# Reconstruction 3D : des boules projetées en radiographies, des diracs 3D qui les retrouvent

2026-09-14. Le pendant 3D du sinogramme : `otrec.Radiographs` empile une IMAGE 2D par angle
( rotation autour de `z`, détecteur `( u, v )`, `add_sphere` par quadrature de Gauss-Legendre par
pixel ), et `otrec.models.ProjectedDiracModel` confronte des diracs 3D à ces images par un
transport semi-discret 2D PAR ANGLE ( `sdot.OtPlan` ), le coût étant la somme des `W_2^2`.

Ce qu'il a fallu ajouter à `sdot` pour ça :

* `PowerDiagram.moments` -- `int rho`, `int x rho`, `int |x|^2 rho` par cellule
  ( `diagram/Ops.h::integrate_moments_into`, formes closes sur les simplexes d'un morceau à densité
  constante, quadrature de `PointwiseDensity` sinon ) ;
* `OtPlan.transport()` / `cost` / `cost_and_position_grad()` -- le coût de transport et sa
  dérivée par rapport aux POSITIONS par le théorème de l'enveloppe, `2 m_i ( p_i - b_i )` ;
* `OtPlan( objective = "dual" )` -- la fonctionnelle duale de Kantorovich descendue par L-BFGS :
  gradient = résidu, valeur et gradient d'UN balayage `moments`, pas d'adjoint, pas de barrière.
  Sur ce cas ( une image avec un fond de 1e-3, des diracs dans des déserts ), la moindre-carrés
  n'avait pas convergé en 100 pas ( résidu 0.08 ) là où le dual converge en ~100 pas à 1e-5 ;
  et un pas coûte un kernel au lieu de trois. Limite connue : sur une densité LISSE la valeur n'est
  connue qu'à la quadrature près ( `rtol` 1e-5 ), et la descente s'arrête là ( ~1e-3 de résidu sur
  le cas dur des gaussiennes séparées, 2e-4 sur le cas doux ) -- la moindre-carrés, qui n'a besoin
  que des mesures ( exactes en 2D ), va plus loin sur ces cibles-là.

Le modèle 3D n'a qu'une évaluation FUSIONNÉE ( `value_and_grad`, `FusedLBFGS` ) : pas d'autodiff à
travers un ajustement itératif. Les poids de chaque angle sont gardés d'une évaluation à l'autre
( `weights0` ). `Reconstruction` lit la dimension sur la donnée ( `world_dim` ) et choisit le
modèle diracs qui va avec.

## À grande échelle : Newton, l'enveloppe visuelle, et ce qui a été appris

L-BFGS sur le dual ne tient pas le passage à l'échelle ( 2000 diracs : 300 pas sans converger ) :
le nombre de pas croît avec le conditionnement, donc avec `n`. D'où :

* `PowerDiagram.hessian_rows` -- la jacobienne des mesures par rapport aux poids, creuse, une
  entrée par facette ( `diagram/Ops.h::hessian_row`, `Local2::for_each_facet` / `LocalN` en 3D par
  l'accumulation des faces ) ; vérifiée contre la différence finie à 1e-10 ;
* `OtPlan( objective = "newton" )` -- le Newton amorti de Kitagawa-Mérigot-Thibert : direction par
  `spsolve` sur `H + eps I`, pas retenu dès que la NORME DU RÉSIDU baisse d'un facteur `1 - t/2`
  sans qu'une cellule passe sous le plancher ( la moitié de la plus petite mesure de départ ).
  Sur une image bien conditionnée : 7 pas à 1e-13. Trois choses qu'il a fallu apprendre :
  - un départ chaud qui VIDE une cellule ( les poids d'un nuage voisin, après un déplacement de
    `h / 2` ) est pire que le Voronoï : ligne de hessienne nulle, `t ~ 1e-7`, 200 pas sans rien.
    On repart de zéro dès que le départ donné a une cellule sous `1e-3` de la masse cible ;
  - la phase LINÉAIRE de KMT ( des cellules presque vides, `t ~ 1e-3` ) coûtait dix évaluations
    par pas à repartir de `t = 1` : le pas d'essai repart de quatre fois le dernier accepté ;
  - une région de confiance ( borner le pas à `h^2` ) fait ramper cette même phase : retirée.
  Ce qui reste cher est le DÉPART À FROID d'un nuage aléatoire : la plus petite cellule de
  Voronoï de `n` points au hasard pèse `~1/n` de la moyenne, et la vitesse de KMT est bornée par
  ce plancher -- ~50 pas quel que soit le fond. Chaud ( un nuage qui a peu bougé ) : 2 à 5 pas.
* `Reconstruction.hull_points` / `Radiographs.visual_hull_points` -- le point de départ dans
  l'ENVELOPPE VISUELLE ( par rejet : les points dont toutes les projections tombent sur de la
  matière ). Un dirac qui projette dans le vide a une cellule quasi nulle et un transport aussi
  mal conditionné qu'il est loin ; avec 6 angles l'enveloppe est déjà les boules à 98 %.
* `models.ProjectedDiracModel` : Newton, `max_backtracks = 30` ( 8 ne suffisent pas au premier pas
  d'un départ à froid, et l'ajustement s'arrêtait... au Voronoï, ce qui faisait tout de même
  baisser le coût ), `mass_tol` relatif `1e-4` ( ce que le noyau FP32 sait sur l'aire d'une
  cellule ).
* `Visualizer.write_vtk` : le rayon des points en donnée de cellule ( `radius` -- un `Glyph`
  ParaView à l'échelle 1 dessine les boules à leur taille ), et les points sans boucle Python.

Mesuré ( 4 threads OMP ) : 5000 diracs, 4 angles, 96² -- départ à froid 3 à 10 s par angle, chaud
0.2 à 0.4 s ; un pas de L-BFGS externe 20 à 30 s. À 20 000 diracs, 6 angles, 128² : ~140 s par
pas ( presque chaque évaluation repart du Voronoï : un pas externe déplace les points de plus
qu'il ne faut pour vider des cellules ), 10 pas en 26 min, coût 9.0e-3 -> 4.7e-3, 96.7 % des
diracs dans les boules -- l'enveloppe visuelle à 6 angles en met 98 % d'emblée, et les 3 % qui
en sortent vont dans ses régions FANTÔMES ( là où les projections restent dans les ombres ) : la
limite des angles, pas du solveur. Les expériences `rec 3D spheres` / `rec 3D random spheres`
( `otrec/tests/test_reconstruction_3d.py`, 10 000 diracs et 30 pas par défaut, `min_iter =
max_iter` pour que scipy ne conclue pas sur le bruit des ajustements ) écrivent HTML + `.pvd` +
courbe tous les dix pas.

## Par étages : peu de diracs, converger, raffiner

`Reconstruction.multiscale` ( déjà là pour le 2D ) fonctionne tel quel en 3D : le premier étage
tire dans l'enveloppe visuelle, chaque dirac est remplacé par `factor` enfants bruités une fois
l'étage quasi convergé ( l'arrêt de scipy, ou `max_iter` ), et `ProjectedDiracModel` abandonne ses
poids chauds quand le nuage change de taille. Expérience `rec 3D multiscale`.

Mesuré, 8 boules au hasard, 6 angles, 128², 8000 diracs au final ( 4 threads OMP ) :

| | pas | temps | perte | dans les boules |
|---|---|---|---|---|
| par étages 500 -> 2000 -> 8000 | 30 + 27 + 12 | 39 + 53 + 254 = 346 s | 4.96e-2 -> 2.24e-3 | 79 % -> 93.3 % |
| direct à 8000 | 30 | 587 s | 2.38e-2 -> 2.22e-3 | 82 % -> 93.0 % |

Même perte finale, même fraction dans les boules, en 1.7 fois moins de temps -- et l'étage
final n'a eu que 12 pas à faire ( le direct part de 2.4e-2, les étages de 4.9e-3 ).
Le raffinement ne coûte presque rien à la perte ( 9.5e-3 -> 8.6e-3 en passant de 500 à 2000,
5.2e-3 -> 4.9e-3 de 2000 à 8000 ) : chaque étage hérite de la structure du précédent et n'a plus
qu'à raffiner localement. Ce qui reste cher est l'étage final lui-même ( 21 s par pas à 8000 ) :
les ajustements par angle y repartent encore du Voronoï à chaque évaluation.

## Les zéros des projections : flouter d'abord, et un départ qui loge dans le domaine

2026-09-15. Les projections ont des zéros, et tout ce qui y projette est sans cellule ni gradient.
Deux réponses ( proposées par HL ) :

* `Radiographs.blurred( sigma )` / `Sinogram.blurred`, et `Reconstruction.anneal_blur( blurs )` :
  des itérations sur les projections FLOUTÉES d'abord ( `sigma` = la largeur du détecteur ), puis
  resserrées étage par étage ( 1, 0.25, 0.06, 0.015, 0 en fraction de la largeur ). Mesuré sur
  3000 diracs, `OtPlan` Newton ( KMT ) : floutée à l'échelle du domaine, la cible se résout en
  6 à 11 pas depuis N'IMPORTE quel nuage -- l'enveloppe visuelle, le cube, ou hors du détecteur --
  là où la donnée nette demande 34 pas depuis l'enveloppe et ne converge pas depuis le cube ;
* un TEST avant Newton : si le Voronoï laisse une cellule vide, le départ est le Voronoï d'une
  SIMILITUDE qui ramène le nuage dans le domaine ( contraction + translation ), écrit comme
  diagramme de puissance du nuage d'origine, `w_i = | p_i |^2 - | a p_i + b |^2 / a`
  ( `OtPlan._similarity_start` ) : toutes les cellules nourries, et le cas « hors du détecteur »
  converge comme les autres.

Le Newton NON amorti ( `damping = "none"` : pas plein dès qu'il baisse le résidu, retour arrière
sinon, pas de plancher ) a été essayé et mesuré 5 à 50 fois plus lent que KMT dans tous les cas,
floutage ou pas ( 1000 à 5000 évaluations contre 10 à 90 ) : son retour arrière repart de 1 à
chaque pas, là où KMT repart du dernier pas accepté. Il reste disponible, KMT reste le défaut.

Expériences `rec 3D blur` ( depuis le cube ) et `rec 3D blur multiscale` ( le flou resserré sur
un petit nuage, puis le raffinement par étages sur la donnée nette ). Mesuré, 4 boules, 4 angles,
64², 1500 diracs depuis le cube : 5 étages, 40 pas, 59 s, 92.9 % des diracs dans les boules.
Puis 8 boules, 6 angles, 128², depuis le CUBE, 500 diracs floutés ( 5 étages, 109 s ) puis
raffinés 500 -> 2000 -> 8000 sur la donnée nette : 639 s en tout, perte finale 2.27e-3 et 94.3 %
des diracs dans les boules -- le même résultat que depuis l'enveloppe visuelle ( 2.24e-3, 93.3 %,
346 s ), obtenu sans elle, en partant de n'importe où. L'étage à 8000 reste le poste principal
( 454 s, 23 s par pas ).

## Ce que ça coûte, et ce qui a été mesuré

* 60 diracs, 4 angles, 48² pixels, 30 pas de L-BFGS : 96 s, 90 % des diracs dans les boules, coût
  1.48 -> 0.027 ( `OMP_NUM_THREADS=4` ; test : 3 angles, 40², 45 diracs, 20 pas, 43 s ).
* Le coût d'un pas est dominé par la SURCHARGE PAR APPEL de `driver.call` sur de petits kernels,
  pas par les kernels : mesuré 23 ms par appel avec 16 threads OMP sur une machine chargée
  ( load ~8 ), 1 ms avec 4 -- l'oversubscription des barrières OMP. Et côté loom, `shared_header`
  relisait ~30 fichiers par appel ( `build_dir()` sondait l'écriture à chaque fois ) : mis en cache,
  4.7 s -> 2.5 s par ajustement. Il reste ~6 ms d'analyse Python par appel ( `_render_call`,
  `CallArgsAnalysis` ) : c'est là que se joue la suite pour les petits problèmes.
* `experiment "rec 3D spheres"` ( `otrec/tests/test_reconstruction_3d.py` ) : la descente dans le
  `Visualizer`, un pas par frame.

## Ce qui reste

* un modèle SPHÈRES ( le pendant de `DiskModel` : rayon fixe, projection différentiable ) ;
* le départ à froid de Newton ( ~50 pas ) : une initialisation des poids qui nourrit toutes les
  cellules ( multi-échelle, ou un Voronoï de points mieux répartis ) raccourcirait la phase
  linéaire de KMT ;
* ramener la surcharge par appel de loom sous la milliseconde, ou batcher les angles dans un seul
  appel ( un `OtPlan` batché sur `num_angle`, comme `OtPlan1d` ) ; le `spsolve` scipy à 1e5+ diracs.
