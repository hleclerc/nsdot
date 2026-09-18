# `solvers_des_familles` — le banc des solveurs

Le pendant de [`2d_des_familles`](../2d_des_familles/README.md), de l'autre côté de la porte.
Là-bas on a cherché **comment construire un diagramme de puissance vite** ; ici on prend ce qui a
gagné, on le range une fois pour toutes dans `src/cell/` (à priori on n'y touche plus), et on
s'en sert pour **essayer des solveurs** — le problème de transport semi-discret résolu de bout en
bout, chronométré par poste.

Du C++ et un compilateur. `xmake`, `-O3 -march=native`, des `std::thread` pour le diagramme,
OpenMP pour AMGCL. Rien d'autre à installer : asimd vient de `../sdot/include`, AMGCL et Eigen
sont en-têtes seuls.

```
xmake f -m release && xmake      # trois binaires, 30 s
xmake run check                  # l'engin contre le balayage complet -- 2D, 3D, Voronoi, Laguerre
xmake run diagramme              # un diagramme chronométré, sur la suite
xmake run newton --help          # le transport résolu, chronométré par poste
xmake run ecrasement --help      # jusqu'où une direction de Newton peut aller avant une cellule vide
```

Les nuages durs sont lus dans `../2d_des_familles/cases/` (`--cases DIR` pour les mettre
ailleurs). Les options communes sont les mêmes partout (`-n`, `--threads`, `--leaf`, `--2d`/`--3d`,
`--load`, `--kernel`, `--maxnv`, …) : elles vivent dans `src/bench/Args.h`, pas dans les `main`.

---

# 1. OÙ EST QUOI, ET QUI FAIT QUOI

```
src/util/      common.h      TF, SI, Vec<D>, now()
               parallel.h    Parallel { threads, pin }, parallel_for -- tranches contiguës, un join

src/accel/     WeightMajorant.h   w( y ) <= a . y + b sur une tranche de germes
               AaBsp.h            le BSP médian : permutation, boîtes, majorants ; refresh_weights

src/cell/      L'ENGIN. La cellule DIRIGE : elle demande un plan à un fournisseur, coupe, redemande.
               Contrat2D.h        Plan2, Atelier, la coupe scalaire en place, le trait `Local`
               Noyau2D.h          la cellule dans trois registres de huit voies, machine à états sur NB
               Elagage2D.h        « un germe de cette boîte peut-il encore couper ? » -- exact, SIMD
               FournisseurBsp2D.h le parcours de l'arbre, SUSPENDU entre deux demandes
               FournisseurAlpha2D.h le même parcours en w + alpha d, sans rafraîchir, à chaud (§ 7.1)
               Contrat3D.h        Plan3, l'état que le fournisseur voit
               Cellule3D.h        le polytope simple porté par ses sommets (3 coupes, 3 voisins chacun)
               Elagage3D.h        le même test, un axe de plus, sur des sommets en mémoire
               FournisseurBsp3D.h le même parcours
               Noyau3D.h          la boucle 3D
               Balayage.h         le TÉMOIN : tous les autres germes, sans élagage

src/diagram/   PowerDiagram.h     PowerDiagram<D,TK,MaxNv> : build, set_weights, measures,
                                  measures_and_facets. Le seul endroit qui distingue 2D et 3D.

src/solver/    Laplacien.h        c_ij = |facette| / ( 2 |p_i - p_j| ), assemblé sans tri ; le CRS réduit
               Lineaire.h         Amg (AMGCL, trois hiérarchies) et Cholesky (Eigen) -- même surface
               Newton.h           Newton amorti (KMT), jauge w_0 = 0, le temps par poste
               Ecrasement.h       le polynôme d'une cellule le long de w + alpha d, à combinatoire figée

src/bench/     Nuages.h/.cpp      uniforme, lecture/écriture de cases/, la suite
               Direction.h        un nuage + des poids + une direction de Newton, sauvés dans directions/
               Args.h/.cpp        les options communes
               Dispatch.h         --kernel et --maxnv deviennent des paramètres de template, ici seulement

src/mains/     main_check.cpp     l'exactitude
               main_diagramme.cpp la ligne de base
               main_newton.cpp    le solveur
               main_ecrasement.cpp les cellules qui se vident le long d'une direction (§ 7)

directions/    les directions « à problème » sauvées, leurs CSV et leurs figures
scripts/       ecrasement_plot.py, les figures depuis les CSV
```

**Les responsabilités, en une phrase chacune.** L'arbre range les germes et borne leurs poids ; il
ne parcourt rien. Le fournisseur parcourt l'arbre et décide ce qu'il propose ; il ne coupe rien.
Le noyau coupe ; il ne connaît ni dirac, ni arbre, ni critère d'arrêt. `PowerDiagram` fait le tour
des cellules et rend des mesures et des facettes indexées par l'identifiant de l'appelant ; il ne
sait pas ce qu'est un poids optimal. Le laplacien s'assemble depuis les facettes. Un solveur
linéaire résout `L d = b` et rapporte son temps. Newton enchaîne.

---

# 2. CE QUI A ÉTÉ GARDÉ, ET CE QUI NE L'A PAS ÉTÉ

Tout ce qui est ici a une mesure derrière lui dans `2d_des_familles` (README § 4, et le journal).

**Gardé.** Le BSP médian aligné, descente fils-le-plus-proche, et le **majorant affine** des poids
(×3 sur le cas dur). La **cellule qui dirige** avec le parcours retourné en fournisseur suspendu et
l'élagage **exact** sur les sommets (280 → 26 candidats par cellule). En 2D la cellule en **trois
registres** avec `NB` constante de compilation, la coupe sans boucle et l'excursion en mémoire ;
la **machine à états** plutôt que `musttail` (mesurée équivalente, et portable). En 3D le
**polytope simple porté par ses sommets**, les sommets gardés qui ne bougent pas (−15 %), la
première passe en asimd à huit voies, le volume et les aires **par accumulation** sans parcours de
cycle. Newton amorti tel quel, l'assemblage **sans tri** (4.2 → 0.67 s à n = 10⁶), **AMGCL**
(4.9× sur le total à 10⁶ contre Cholesky) avec ses trois hiérarchies, et Cholesky en témoin parce
qu'il gagne encore sur le nuage de lignes.

**Pas gardé.** Le multi-échelle (écrit, mesuré, pas abouti : plus cher que Newton depuis `w = 0`),
`--memo` (−19 % au mieux, jamais confirmé), le front sur diagramme grossier dans Newton (−21 à
−46 % au total), le bouclier, la grille, les BSP à quatre fils et obliques, la pré-passe, le
gradient conjugué maison (5× plus lent que Cholesky), les découpages `strided`, la boîte de cellule
(sans objet avec l'élagage sur les sommets).

**Ajouté.** Le flottant du noyau est un paramètre (`TK`, option `--kernel`) : `float` est ce que
`sdot` fait en production, `double` est le défaut ici — et § 4 dit pourquoi.

---

# 3. LES PREMIERS CHIFFRES

Xeon W-2145, 8 fils épinglés, `kernel=double`, mêmes nuages que `2d_des_familles`.

## Un diagramme (`xmake run diagramme --threads 8`)

| | n | temps | ns/germe | ancien engin |
|---|---|---|---|---|
| 2D uniforme | 10⁶ | **0.207 s** | 207 | 0.213 s |
| 2D lignes / Voronoï | 10⁵ | 0.021 s | 211 | 0.023 s |
| 2D lignes / aires égales | 10⁵ | 0.073 s | 733 | 0.093 s |
| 3D uniforme | 10⁶ | **2.768 s** | 2768 | (3425 ns/germe à 2·10⁵) |
| 3D plans / Voronoï | 10⁵ | 0.260 s | 2602 | 0.352 s |
| 3D plans / volumes égaux | 10⁵ | 0.460 s | 4603 | 0.703 s |

`--kernel float` : 0.199 s et 2.655 s sur les deux uniformes. Le `double` coûte 4 %.

## Newton (`xmake run newton --threads 8 -n 100000`)

Mêmes itérations, mêmes diagrammes, mêmes reculs que l'ancien banc, au diagramme près :

| | it / diag (reculs) | diagrammes | résolution | TOTAL | ancien TOTAL |
|---|---|---|---|---|---|
| 2D lignes / aires égales | 24 / 113 (89) | 7.9 s (44 %) | 9.2 s | **17.9 s** | 22.5 s |
| 3D uniforme | 6 / 9 (2) | 2.9 s (56 %) | 2.1 s | **5.2 s** | — |
| 3D plans / Voronoï | 13 / 27 (13) | 11.0 s (68 %) | 4.8 s | **16.3 s** | 20.8 s |

Sur les lignes, les poids trouvés collent à ceux du fichier (L-BFGS, pysdot) à **1.3e-14** sur une
amplitude de 0.128. La boucle y sort en `STAGNATION` à `max|a−ν|/ν = 3e-6` : c'est le plancher du
cas (le fichier annonce 2.35e-6 en en-tête), pas un défaut du solveur — l'ancien banc s'arrête au
même endroit.

Le solveur linéaire, sur les lignes : Cholesky **11.9 s** au total, Ruge-Stüben+GS 15.2 s,
agrégation+spai0 17.9 s. Sur l'uniforme c'est l'inverse, et à 10⁶ Cholesky ne monte pas en
charge. La question reste ouverte, et c'est pour ça que les deux sont là.

---

# 4. LE FLOTTANT DU NOYAU, ET POURQUOI LE DÉFAUT EST `double`

`--kernel float` sur les cas durs :

* **3D plans / Voronoï** : `STAGNATION` à 3.8e-3, 118 diagrammes pour 18 itérations, 100 reculs.
  L'amortissement demande une décroissance stricte du résidu, et le bruit d'un volume calculé en
  `float` (~1e-7 relatif) la refuse bien avant la tolérance.
* **2D lignes** : le solveur linéaire échoue à la deuxième itération (20 000 itérations, résidu 20).
  Des germes à 1e-5 l'un de l'autre donnent des longueurs d'arête et des distances qui n'ont plus de
  chiffres en `float`, et `c_ij` n'a plus de sens.

Le noyau `float` est donc **une question de solveur** — jusqu'où peut-on descendre la tolérance,
que faut-il calculer en `double` pour que le reste puisse rester en `float` — et pas un réglage.
C'est pour pouvoir la poser que le paramètre existe.

---

# 5. AJOUTER UN SOLVEUR

Un solveur linéaire est un objet avec

```cpp
bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d );  // d[ 0 ] == 0
const char *nom() const;
StatsLin st;
```

`Laplacien` livre `n`, le CSR des hors-diagonaux, la diagonale, et `crs_reduit()` pour qui veut le
système à `n − 1` inconnues, colonnes triées. On l'instancie dans `main_newton.cpp` derrière une
option, et `Newton<PD,Lin>` fait le reste.

Une variante de Newton — un autre amortissement, un autre départ, une autre cible — se met dans
`src/solver/` à côté de `Newton.h`, avec la même surface : `resout( w_init )`, `st`, `w`. Le
diagramme n'a rien à en savoir.

---

# 6. LES TÉMOINS

`xmake run check` compare, cellule par cellule et à n = 5000, le fournisseur BSP au balayage
complet dans les deux sens : la somme des mesures, les voisinages, et en 3D l'invariant
d'adjacence. Le balayage **inverse** est là pour calibrer : même algorithme, seul l'ordre des coupes
change, et son désaccord avec le balayage direct est le plancher du flottant. Résultat aujourd'hui,
`double` comme `float`, 2D comme 3D, Voronoï comme Laguerre : **zéro** cellule différente.

La cellule du témoin est bien plus grande que celle de l'engin (512 / 1024 sommets) : proposer les
germes dans l'ordre du tableau laisse la cellule énorme pendant des centaines de coupes, et une
liste de coupes de 128 sature avant que les vraies voisines arrivent.

---

# 7. L'ÉCRASEMENT : JUSQU'OÙ PEUT-ON SUIVRE UNE DIRECTION ?

Sur les lignes, Newton recule 8 fois à la première itération (`t = 3.9e-3`, 9 diagrammes) et
89 fois en tout. Ces reculs sont **entièrement** dus au plancher d'aire : le long de `w + α d`,
le critère de décroissance du résidu (`|r| ≤ (1 − α/2) |r₀|`) tient jusqu'à `α ≈ 0.6`, et au pas
plein le résidu tombe à `0.19 |r₀|` — avec 50 000 cellules vides. Ce sont les cellules qui se font
écraser qui limitent le pas, et `ecrasement` cherche à **prédire depuis le diagramme en `w` seul**
à partir de quel `α` la première se vide.

**Le polynôme à combinatoire figée.** Le plan entre `i` et `j` est `(p_j − p_i)·x ≤ c_ij + α (d_i − d_j)/2` :
la normale ne bouge pas, le décalage est affine en `α`. Tant que la cellule garde les mêmes
arêtes, chaque sommet est affine en `α`, et l'aire est un **polynôme de degré 2** (`Ecrasement.h`
le calcule exactement sommet par sommet ; degré 3 en 3D, pas encore écrit). On en tire la
première racine (la cellule vide), et la première arête qui s'annule (la longueur signée d'une
arête est affine en `α` : c'est le premier événement combinatoire visible depuis la cellule).
Au-delà d'une arête annulée, le polynôme continue à compter le triangle inversé négativement :
il **sous-estime** l'aire, doucement (l'erreur est du second ordre), donc reste conservatif.
Ce qu'il ne voit pas : un germe qui n'était pas voisin et dont le plan entre dans la cellule.
Là, il **surestime**.

**Le banc.** `xmake run ecrasement --load FILE --it K --ecrire directions/X.txt` extrait la
direction de Newton à l'itération `K` (`NewtonOptions::extraire`) et la sauve ; `--direction
directions/X.txt` la relit. Ensuite : les polynômes de toutes les cellules (0.03 s à n = 10⁵),
41 diagrammes sur une grille géométrique d'`α` (1.3 s), et pour chaque `α` la comparaison cellule
par cellule ; puis la première cellule vide et la première sous `eps`, par bissection, contre la
prédiction. `--csv PREFIX` écrit la grille et les cellules suivies, `scripts/ecrasement_plot.py`
en fait la figure.

**Ce qu'on a mesuré** (`directions/`, lignes, `s = 0.005`) :

| direction | reculs de Newton | prédit (cellule) | exact (cellule) | rapport |
|---|---|---|---|---|
| n = 2000, it 0 | 5, `t = 3.1e-2` | 3.49e-2 (1693) | 3.87e-2 (1693) | 0.90 |
| n = 10⁵, it 0 | 8, `t = 3.9e-3` | **4.2341e-3** (22524) | **4.2341e-3** (22524) | **1.0000** |
| n = 10⁵, it 5 | 4, `t = 6.2e-2` | 2.01e-1 (93705) | 8.2e-2 (90645) | 2.46 |

(`α` du premier passage sous `eps`, le critère de l'amortissement.) À l'itération 0, la cellule
qui meurt est un quadrilatère mince (`a₀ = 0.011 ν`, `a₁ = 0.989 ν` — la linéarisation de Newton
vise bien `ν` en `α = 1` —, mais `a₂ = −853 ν`) qui s'écrase d'un bloc, sans événement
combinatoire avant : la prédiction est exacte, et aurait donné le pas de Newton **sans aucun
diagramme d'essai**. Cellule par cellule, le polynôme colle à `1e-4 ν` sur 99.8 % des cellules à
`α = 4e-3`, et sur 100 % jusqu'à `α ≈ 1e-3`. À l'itération 5, la cellule qui se vide en premier
(90645) a un polynôme **croissant** (`a₂ > 0`) et aucune arête qui s'annule avant `α = 0.30` :
elle est mangée par un nouveau voisin, l'événement que le polynôme d'une seule cellule ne peut
pas voir, et la prédiction est optimiste d'un facteur 2.5 — Newton a dû refuser `t = 0.125`.

**Ce qui ne marche pas : décomposer en triangles.** On a testé deux estimateurs bâtis sur
l'éventail `(p_i, v_j, v_{j+1})`, un polynôme par triangle : la somme des *parties positives*,
et la somme où chaque triangle est mis à zéro passé son premier changement de signe (colonnes
`pos` et `zero` de la grille). Résultat, lignes n = 10⁵, it 0 : à `α = 4e-3` (le pas de
Newton) 47 % des cellules à `1e-4 ν` contre 99.8 % pour le polynôme ; à `α = 1`, **0** cellule
vide prédite contre 50 030 vraies. La raison est géométrique : une cellule mince qui s'écrase
d'un bloc est un polygone qui *s'inverse* — la somme signée devient négative, c'est la racine du
polynôme — mais certains de ses triangles restent positifs, donc la somme des parties positives
ne s'annule jamais. Et après une arête annulée, le triangle inversé *négatif* compense au premier
ordre le dépassement de ses deux voisins (l'erreur de la somme signée est `−|A'B'C|`, du second
ordre) ; l'enlever laisse une erreur du premier ordre. En Laguerre (it 5) c'est faux dès
`α = 0` : la cellule ne contient pas toujours son germe. La somme *signée* est le bon objet.

## 7.1 Prédire, vérifier, corriger

Le polynôme seul ne voit pas le nouveau voisin ; une vraie cellule voit tout. D'où la passe
`limites` (`Ecrasement.h`), une cellule à la fois, dans le même parcours :

1. le polynôme en `α = 0` donne une limite prédite (le premier `α` où l'aire passe sous `eps`) ;
2. on calcule la cellule **exacte** en `0.99 × prédit`, avec `FournisseurAlpha2D.h` : l'arbre
   n'est pas rafraîchi — chaque nœud porte un majorant de `w` et un majorant de `d`, et
   `w + α d ≤ (a_w + α a_d)·y + (b_w + α b_d)` est exact pour tout `α ≥ 0`, donc `α` peut
   changer d'une cellule à l'autre ; et le départ est **à chaud** : les voisins de `α = 0` sont
   coupés d'abord, sans parcours, puis le parcours élagué ne trouve presque plus rien (les
   germes déjà proposés sont sautés : les reproposer recouperait une lamelle d'aire nulle par
   arrondi, et la combinatoire serait fausse) ;
3. **mêmes arêtes** ⇒ le polynôme était exact jusque-là, la limite est confirmée — un test
   exact, pas un seuil. Sinon la cellule calculée porte la nouvelle combinatoire, donc un nouveau
   polynôme, donc une limite corrigée, et on repart de là ; une cellule trouvée trop petite
   borne par au-dessus et son polynôme, lu à rebours, dit où revenir ; une cellule vide ne
   porte rien, on resserre par bissection. Au-delà de l'horizon (`α = 1`, le pas plein) on ne
   vérifie qu'à l'horizon.

Le fournisseur est vérifié contre le diagramme rafraîchi (`--check A` : 0 cellule différente
sur 10⁵, à quatre `α`). Résultat sur les trois directions, `--coeff 0.99 --tol 1e-2` :

| direction | limite globale (cellule) | exact | cellules calculées / cellule | temps | diagrammes d'essai de Newton |
|---|---|---|---|---|---|
| n = 2000, it 0 | 3.874695e-2 (1693) | 3.874695e-2 | 2.1 | 6 ms | 6 |
| n = 10⁵, it 0 | 4.234123e-3 (22524) | 4.234123e-3 | 2.7 | 0.18 s | 9 (0.38 s) |
| n = 10⁵, it 5 | 8.171690e-2 (90645) | 8.171690e-2 | 1.05 | 0.13 s | 5 |

Les trois limites globales sont **exactes** — y compris à l'itération 5, où le polynôme seul se
trompait de ×2.5 —, et cellule par cellule les limites tombent dans l'encadrement que donne la
grille de 41 diagrammes pour **100 %** des cellules (avec `--coeff 0.9`, trois cellules sur 10⁵
sortent de 4 % : l'extrapolation des derniers 10 %). Le coût : à l'itération 0 la moitié des
cellules survivent au pas plein et sont vérifiées une fois à `α = 1` ; l'autre moitié change de
combinatoire avant sa limite prédite (la direction depuis Voronoï est violente) et demande 2 à 5
tours ; à l'itération 5 c'est une cellule par cellule. Une cellule à chaud coûte ~1.8 fois une
cellule de diagramme, parce qu'on l'évalue à grand `α`, là où le diagramme lui-même est plus
cher. Au total ~2 fois moins que les diagrammes d'essai, **et une limite par cellule**, ce que
les diagrammes d'essai ne donnent pas.

**Ce qu'on pourrait encore gagner.** `--tol 0.3` fait 1.8 cellule par cellule à l'itération 0
(la limite globale tombe alors à 0.96 de l'exact — Newton ne teste que des puissances de 2, une
précision grossière lui suffit) ; pour la seule limite globale, vérifier chaque cellule au
minimum courant plutôt qu'à sa propre prédiction ramènerait à une cellule par cellule, à petit
`α` ; et le majorant de `d` pourrait être calculé avec celui de `w`, en un seul passage.

## 7.2 Dans Newton : `--pas dyadique | facteur`

`Newton.h` calcule les limites (`OptionsLimites::global` : une cellule dont la prédiction
dépasse le minimum courant n'est vérifiée qu'à `1.1 ×` ce minimum — une cellule à petit `α`,
pas à l'horizon — et la cellule en `α = 0` se refait par ses seuls voisins, lus dans le CSR du
laplacien, sans parcours), puis `t` est la puissance de deux sous `α*` (`dyadique`) ou
`0.9 α*` (`facteur`), et le diagramme de ce pas — nécessaire de toute façon — confirme la
décroissance du résidu ; s'il refuse, on recule comme avant. Lignes, n = 10⁵, 8 fils :

| solveur | essais (KMT) | dyadique | facteur 0.9 |
|---|---|---|---|
| Cholesky | 26 it, 117 diag, **11.2 s** | 26 it, 62 diag, 10.4 s | 22 it, 61 diag, **9.2 s** |
| AMG Ruge-Stüben + GS | 27 it, 119 diag, **14.0 s** | 27 it, 64 diag, 13.5 s | 22 it, 57 diag, **11.5 s** |

`dyadique` reproduit **exactement** les pas que KMT trouve par essais (mêmes itérations, 0 à
3 refus sur 26), pour moitié moins de diagrammes : la passe coûte ~0.08 s par itération, soit
1.3 diagramme, contre 3.3 diagrammes d'essai en moyenne. Le gros de la passe est le parcours
de l'α-cellule, incompressible — il faut chercher les nouveaux voisins ; le départ à chaud
n'économise que les coupes, pas le parcours (mesuré : 0.029 s à chaud contre 0.027 s à froid
pour 10⁵ cellules), parce que les plans des voisins touchent la cellule et empêchent d'élaguer
leurs feuilles. `facteur` gagne en plus quatre à cinq itérations, donc autant de résolutions
linéaires, le poste dominant : **−17 %** au total. Ce qui reste : la sortie en `STAGNATION`
brûle 34 diagrammes à descendre jusqu'à `t = 1e-10` (`--t-min`), la moitié du poste diagrammes
en mode limites (`--t-min 1e-3` : 37 diagrammes, 7.7 s avec Cholesky + facteur, contre ~9.9 s
pour KMT au même `t_min`) ; et l'AMG agrégation+spai0, le défaut, **échoue** sur certaines trajectoires
(20 000 itérations, 50 s, une fois par run) là où Cholesky et Ruge-Stüben passent — une
fragilité du solveur linéaire sur des cellules proches de `eps` (2.5e-5 ν ici), pas du pas.

## 7.3 Modifier la direction : ce que les chiffres disent

**Le segment droit de Voronoï vers `w*` est admissible.** Avec `d = w*` (les poids du fichier),
`α` de 0 à 1 : **aucune** cellule vide, l'aire minimale croît de façon monotone de 8e-10 à ν
(`directions/lignes100000_versW.txt`). Une bonne direction existe donc. La direction de Newton
lui est corrélée à 0.991, avec 18 % d'écart (`|d − w*|/|w*|`), par régions (une ligne poussée
40 % trop fort). Et la géométrie amplifie : à `α = 1`, le plan entre une cellule et son plus
proche voisin se déplace de 150× la distance à ce voisin (médiane ; 570× au 90ᵉ centile) — pour
`w*` aussi, qui survit parce que la combinatoire se réarrange de façon cohérente. C'est l'erreur
de direction, amplifiée ×150, qui tue. Le polynôme figé prédit 18 310 cellules vides à `α = 1`
le long de `w*` : au-delà de `α ~ 0.02–0.05` la combinatoire change trop pour lui.

**Pondérer les aires ne change pas la direction** : `(ΩL) d = Ω r` a la même solution que
`L d = r`. Un pas par cellule (`w + T d`) n'est pas viable non plus : le plan `ij` bouge de
`(t_i d_i − t_j d_j)/2|p_i − p_j|`, et un écart `t_i − t_j` de 1e-7 fait un espacement.

**Le pas tensoriel** (`--tenseur` du banc, `--pas tenseur` de Newton) : le modèle quadratique de
*chaque* cellule dans tout l'espace des poids (`ModeleCellule` : son polygone figé, décalages
`c_k + (δ_i − δ_j)/2`), résolu par Newton *sur le modèle* — le jacobien est le laplacien aux
longueurs d'arêtes **signées**, même motif que `L`, refactorisé, amorti sur le résidu du modèle
— pour la cible partielle `a + θ(ν − a)`. À l'itération 0, `θ = 0.02` : 4 itérations, 0.37 s,
et le vrai diagramme en `w + δ` donne **0 cellule vide** pour le même gain de résidu que le pas
droit `0.02·d`, qui en vide 351 — cinq fois le pas que KMT accepte. À `θ = 0.05` la solution
exacte du modèle laisse encore 2 cellules vides (le nouveau voisin) et le Newton interne stalle
(Picard sur-amorti y arrive en 276 itérations). Picard avec `L` figé diverge dès `θ = 0.01`.

Branché dans Newton (`θ = 5 α*`, puis `limites` sur `δ`, critère de décroissance ramené à `θ`) :
20 itérations au lieu de 22, mais 4 s de pas tensoriels — **11.8 s contre 7.7 s**. Il n'est
utile qu'aux itérations 0–1 ; de 2 à 12, avec des pas de 0.04 à 0.8, sa portée (validité du
modèle × θ) ne dépasse pas l'`α*` de la direction de Newton et il est écarté. La conclusion
tient au-delà de cette implémentation : le pas est borné par la durée de validité de la
combinatoire figée, pas par la direction.

## 7.4 La continuation par série (MAN) : le modèle figé a des plis

Le long du vrai chemin (`versW`), la combinatoire figée tient à `1e-4 ν` pour **1 %** du chemin,
à `1e-2 ν` pour **3 %** ; 82 546 cellules sur 10⁵ changent d'arêtes au moins une fois, surtout
dans les premiers 10 %. Toute série sur la combinatoire figée hérite de ça.

La Méthode Asymptotique Numérique sur le modèle quadratique (`--man N`) : la cible glisse
`a + s(ν − a)`, `δ(s) = Σ s^k w_k`, et parce qu'en 2D le modèle figé est *exactement*
quadratique il ne faut que deux opérateurs — `L` factorisé une fois, et la forme bilinéaire
`Q(u,v) = q(u+v) − q(u) − q(v)` par polarisation du modèle. Ordre 12 en 0.45 s. Mais le
**rayon de convergence est 3.7e-4** (critère MAN), dix fois sous le pas KMT (0.004), cinquante
fois sous le pas tensoriel (0.02) : `|w_2|/|w_1| = 0.3`, puis `|w_k|` croît comme `10^k`. Le
modèle figé a un *pli* vers `s ≈ 5e-4` — la parabole de la cellule 22524 culmine à
`α = 5.8e-4` (1.14 % de ν) le long de `d` : la branche issue de `s = 0` tourne, la solution
du pas tensoriel à `θ = 0.02` est sur une autre branche, que Newton atteint en sautant et
qu'aucune série ne suit. Le vrai chemin passe ce pli en réarrangeant la combinatoire. Vérifié
aux petits `s` (ordre 4 à `s = 1e-4` : 5e-8 d'écart au modèle) ; à `s = 3e-3` l'ordre 2 vide
17 cellules là où le pas droit (l'ordre 1, Newton) n'en vide aucune.

Le test direct (`--man N` fait aussi, pour chaque ordre, `s = 1` puis `s/2` tant qu'une cellule
passe sous `eps`) : à l'itération 0 l'ordre 1 (KMT) accepte `3.9e-3`, les ordres 2 à 4 `9.8e-4`,
les ordres 8 et 16 `2.4e-4` — la série fait *moins* bien que la droite ; à l'itération 5, où le
rayon MAN vaut 0.32 et l'ordre 16 tombe sur la cible partielle à 2e-6 près, tous les ordres
s'arrêtent au même `6.25e-2` : la cellule 90645 mangée par un nouveau voisin, invisible à tout
ordre du modèle figé.

La limite exacte de `s` par ordre (bissection) : it 0 — ordre 1 `4.23e-3`, ordre 2 `1.32e-3`,
ordre 12 `3.4e-4` ; it 5 — ordre 1 `8.17e-2`, **ordre 2 `9.32e-2`** (+14 %), ordres ≥ 3
`9.1e-2` (la limite du chemin du modèle lui-même). Le seul ordre qui apporte quelque chose est
le 2, quand le modèle n'a pas de pli (`|w₂|/|w₁|` = 0.12 à l'it 5, 0.31 à l'it 0 où il divise
la portée par 3) ; il faudrait le tester contre l'ordre 1 à chaque pas, pour +14 % au mieux.

**Conclusion.** Prédicteur d'ordre 1, un diagramme par pas — c'est Newton amorti, la
continuation « à combinatoire vivante » qu'on a déjà, et le nombre de pas (~20) est le nombre
d'époques combinatoires du chemin. Le levier restant est le coût du pas (réutiliser la
factorisation d'un pas à l'autre, `limites` pour le pas), pas son ordre. Reste le 3D, et un
chemin dans un autre paramètre (les positions, le multi-échelle) si l'on veut moins d'époques.
