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
xmake run image --help           # une IMAGE pour source : mesurer sur le bord, et résoudre dessous
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
               FournisseurBsp3D.h le même parcours ; `MEMO` : les voisins d'hier proposés d'abord (§ 11)
               Noyau3D.h          la boucle 3D
               Balayage.h         le TÉMOIN : tous les autres germes, sans élagage

src/diagram/   PowerDiagram.h     PowerDiagram<D,TK,MaxNv> : build, set_weights, measures,
                                  measures_and_facets. Le seul endroit qui distingue 2D et 3D.

src/solver/    Laplacien.h        c_ij = |facette| / ( 2 |p_i - p_j| ), assemblé sans tri ; le CRS réduit
               Lineaire.h         Amg (AMGCL, trois hiérarchies) et Cholesky (Eigen) -- même surface
               Newton.h           Newton amorti (KMT), jauge w_0 = 0, le temps par poste
               Ecrasement.h       le polynôme d'une cellule le long de w + alpha d, à combinatoire figée
               Prolongation.h     le multi-échelle : paquets, prolongations, relèvement minimal (§ 8)
               Densite.h          plancher + gaussiennes : la masse d'une cellule par circulation (§ 9)
               Image.h            une IMAGE pour densite : la masse integree SUR LE BORD, jamais de decoupage (§ 12)
               PremierOrdre.h     L-BFGS et gradient conjugué sur le dual, laplacien figé, bascule vers Newton (§ 10)

src/bench/     Nuages.h/.cpp      uniforme, lecture/écriture de cases/, la suite
               Direction.h        un nuage + des poids + une direction de Newton, sauvés dans directions/
               Args.h/.cpp        les options communes
               Dispatch.h         --kernel et --maxnv deviennent des paramètres de template, ici seulement

src/mains/     main_check.cpp     l'exactitude
               main_diagramme.cpp la ligne de base
               main_newton.cpp    le solveur
               main_ecrasement.cpp les cellules qui se vident le long d'une direction (§ 7)
               main_multiechelle.cpp la prolongation du multi-échelle, mesurée (§ 8)
               main_densite.cpp   les densités hétérogènes, la continuation en largeur (§ 9)
               main_image.cpp     la densité IMAGE : le témoin par découpage, les chronos, Newton (§ 12),
                                  le relèvement par moindres carrés pondérés (§ 13), le
                                  sous-problème local à objectif barrière (§ 14), son
                                  branchement dans Newton (§ 15) et sa version par amas (§ 16)
               main_memo.cpp      la mémoire en 3D, sa borne supérieure et ses souvenirs périmés (§ 11)

directions/    les directions « à problème » sauvées, leurs CSV et leurs figures
courbes/       le résidu contre les diagrammes, méthode par méthode (`--courbe`, § 10)
scripts/       ecrasement_plot.py, methodes_plot.py, les figures depuis les CSV
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

**Pas gardé.** Le multi-échelle (écrit, mesuré, pas abouti : plus cher que Newton depuis `w = 0` —
rouvert et refermé au § 8),
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
| 2D lignes / Voronoï, nuage dédupliqué (§ 23.7) | 23 / 78 (54) | — | — | — | — |
| 3D uniforme | 6 / 9 (2) | 2.9 s (56 %) | 2.1 s | **5.2 s** | — |
| 3D plans / Voronoï | 13 / 27 (13) | 11.0 s (68 %) | 4.8 s | **16.3 s** | 20.8 s |

Sur les lignes, les poids trouvés collent à ceux du fichier (L-BFGS, pysdot) à **1.3e-14** sur une
amplitude de 0.128. La boucle y sort en `STAGNATION` à `max|a−ν|/ν = 3e-6` : c'est le plancher du
cas (le fichier annonce 2.35e-6 en en-tête), pas un défaut du solveur — l'ancien banc s'arrête au
même endroit.

> **CE PLANCHER ÉTAIT UN DÉFAUT DE DONNÉES, et il n'existe plus sur le nuage Voronoï (§ 23.7).**
> `lines5_n100000_s0.005_voronoi.txt` portait **56 paires de germes confondus**, dont une à 1.009e-08
> pour un espacement médian de 4.5e-4 : deux germes au même point se partagent une cellule qu'aucun
> poids ne sépare, d'où le plafond. Le nuage a été **dédupliqué en place** (`n = 99 944`) et la boucle
> y **CONVERGE** désormais à 2.00e-07 en 78 diagrammes, contre 116 et `STAGNATION` avant.
>
> `lines5_n100000_s0.005_equal.txt` a été **laissé tel quel**, volontairement : c'est lui qui PORTE la
> solution de référence (L-BFGS, pysdot), et retirer un germe la périme — sa masse doit être reprise
> par les autres. `pysdot` n'étant installé nulle part ici, la régénérer est impossible, et un témoin
> vaut plus qu'un nuage propre de plus. L'entrée « lignes / aires égales » de la suite garde donc le
> nuage dégénéré, son plancher à 2.35e-6, **et son témoin** (écart 9.8e-13) — et elle joue maintenant
> le rôle utile de cas dégénéré délibéré. Les deux entrées « lignes » ne sont plus le même nuage.

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

**→ Elle est posée et à moitié résolue au § 19.** Les deux échecs ci-dessus sont datés : celui du
nuage de lignes a disparu (le solveur linéaire ne part plus), celui du 3D reste une stagnation. Ce
qui les causait se nomme et se mesure — `eps·|w|·n^(2/D)` pour les poids, `eps·n^(1/D)` pour les
positions — et le premier terme a été supprimé.

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
factorisation d'un pas à l'autre, `limites` pour le pas), pas son ordre.

**L'homotopie sur la prescription, à combinatoire vivante** (`xmake run homotopie`) : la cible
glisse `ν_s = a₀ + s(ν − a₀)` et Newton *converge* sur chaque cible intermédiaire (tolérance
lâche), `s` doublé si le palier coûte peu, divisé sinon. Lignes n = 10⁵, Cholesky : 9 paliers,
**32–33 itérations, 41–62 diagrammes, 9.1–10.7 s**, contre 22 itérations, 34 diagrammes, 8.7 s
en direct avec `facteur`. Le chemin est *lisse* — 0 recul sur les paliers intermédiaires, là où
Newton direct en fait 65 — mais pas moins cher : 3 corrections par palier (la combinatoire
bouge dans chaque palier de `Δs = 0.125`) et un dernier palier de 8–9 itérations, la fin de
partie de Newton. Avec le prédicteur d'ordre 2 en `s` (`--ordre 2` : `w + w₁ + w₂`, `w₂ =
−L⁻¹ q₂(w₁)` sur le modèle quadratique, contre `w` et `w + w₁`, un diagramme chacun pour
choisir) : **1 correction par palier** de `s = 0.001` à `0.25` en doublant `Δs` — la série
marche où la combinatoire est calme — mais échec au départ de Voronoï (le pli de l'it 0), à
tout `Δs ≥ 0.25`, et en fin de partie ; au mieux 25–30 itérations, pas moins de 22. Vérifié
aussi après le correctif de l'engin (§ 7.5) : § 7.1 à 7.4 inchangés au chiffre près.

**Pourquoi l'homotopie perd — l'animation** (`--dump` sur `newton` et `homotopie`, puis
`scripts/animation.py` → `directions/homotopie_vs_newton.html`, lignes n = 2000, deux panneaux,
couleur = `log₁₀(a/ν)`, mode « voisinage » = les cellules dont les voisins ont changé). Par
trame : Newton direct fait `t = 0.035, 0.33, 0.61, 0.79, 1, 1, 1, 1` — le pas admissible est
multiplié par 10 dès la deuxième itération, parce que la direction est recalculée sur un
meilleur état — et le résidu tombe à 0.020 en trois itérations. L'homotopie, elle, passe 9
itérations à atteindre `s = 0.5` (résidu 0.035, la moitié de r₀, comme la droite des cibles
l'impose) : à chaque palier, 2 itérations sur 3 servent à *converger* sur une cible dont on ne
veut pas, et le dernier palier fait 90 % du travail en 6 itérations — ce que Newton faisait
depuis le début. En plus, son chemin est plus long : 7 225 changements de voisinage contre
5 449. Newton amorti avec le pas par les limites *est* une homotopie adaptative — la taille du
palier au maximum admissible, une correction par palier ; l'expliciter n'ajoute que le coût des
convergences intermédiaires.

## 7.5 Glisser depuis d'autres positions : le meilleur cas, mesuré (`glissement`)

L'idée : partir de positions `c` où le problème est facile, et faire glisser les diracs vers
`p` en suivant `w*(τ)`. Le meilleur `c` possible, on le connaît en trichant : les **barycentres
des cellules de la solution** (`c_i → p_i` est le transport optimal lui-même, sans croisement).
`xmake run glissement --load FILE_equal` : résolution en `c` depuis `w = 0`, puis
`p(τ) = (1−τ) c + τ p` avec le prédicteur tangent `L dw = ν − a − (∂a/∂p)·dp` (la vitesse
normale du plan `ij` quand un site bouge, intégrée sur l'arête : `∂a_i = Σ_j ℓ_ij/|p_j−p_i|
[dp_j·(p_j − m_ij) + dp_i·(m_ij − p_i)]`) et Newton en correcteur, pas en `τ` adaptatif.

Lignes, n = 2000 : le départ aux barycentres est facile (4 itérations, 0 recul, contre 9 et 5
reculs pour la résolution directe) — mais la continuation coûte **79 itérations de Newton**
(15 pas, 4 refus, 193 diagrammes) contre 9, avec 4.1 changements de voisinage *par cellule* le
long du chemin (400 à 800 cellules sur 2000 à chaque pas de 0.1) et une fin de chemin (`τ → 1`,
les diracs qui se resserrent) qui refuse les pas — la sensibilité `(w_i − w_j)/2|p_i − p_j|²`
prédite. Le prédicteur tangent divise le résidu après prédiction par 5 (0.64 contre 3.0 sans
prédicteur, pour `Δτ = 0.1`) mais `τ ↦ w*` est si non linéaire que 3 itérations de correcteur
restent nécessaires ; réduire `Δτ` donne 1 itération par pas et 10× plus de pas. À n = 10⁵ :
départ aux barycentres en 10 itérations (9 reculs), puis **86 pas, 209 itérations de Newton,
295 diagrammes, 5.2 changements de voisinage par cellule**, `Δτ` tombé à 0.0125 dès le début,
et le dernier pas (`τ = 1`) refusé jusqu'à `Δτ = 2e-4` — contre 22 itérations en direct.
**Un chemin en positions compte plus d'époques combinatoires qu'un chemin en poids**, même
dans le meilleur cas — l'idée est morte pour une heure de travail, comme prévu.

**Trouvé en chemin : un bug de l'engin.** Avec les poids du fichier, la somme des aires valait
1.00001 : la cellule 74917 n'avait pas d'arête contre 60863, qui était *dans* sa bande. Cause :
le majorant affine des poids (`WeightMajorant.h`) sur un nœud de germes clampés au bord
(`x = 0.0001` à 1e-8 près) — matrice normale presque singulière, pivot non nul, pente 1e13,
`b` calculé à 1e9 avec une annulation qui rend le majorant **faux** de 3e-8 ; 11 535 nœuds sur
32 767 violés. Deux garde-fous (pente admise seulement si `|a_d| × étendue_d ≤ 8 ×
étalement`, marge de 8 ulp sur `b`) : zéro violation, somme 1.000000000, témoins inchangés,
même temps — et Newton stagne désormais à **2.35e-6, le plancher annoncé par le fichier**, au
lieu de 3.05e-6 (ce plancher était un défaut du nuage, corrigé depuis sur l'entrée Voronoï : § 23.7). `check --load FILE [--cellule I] [--weights -1]` le vérifie ; **le même code
vit dans `sdot` (`refresh_weight_majorants`), à reporter.**

# 8. LE MULTI-ÉCHELLE : LA PROLONGATION, MESURÉE (`multiechelle`)

L'idée (Mérigot) : résoudre sur `n/R` représentants portant la masse de leur paquet, prolonger,
résoudre en dessous. `2d_des_familles` l'avait écrit et abandonné (« plus cher que Newton depuis
`w = 0` ») ; la question rouverte ici est celle des poids à donner aux germes qui n'étaient pas au
niveau grossier. `src/solver/Prolongation.h` et `xmake run multiechelle --help` : niveaux par un
BSP à feuilles de `R` germes (représentant = le germe le plus proche du centre), Newton
`essai-limites` + Cholesky à chaque niveau, la prolongation puis un **test** (le diagramme fin :
aucune cellule sous `0.5 × min(ν_i, aire min de Voronoi)`, le plancher de l'amortissement), et
une **correction** si ça ne passe pas.

Les prolongations : `copie` (le poids du représentant) ; **`harmonique`** — la proposition : les
représentants imposés, les autres poids résolvent l'équation de la chaleur du graphe de Voronoi
fin, chaque poids libre étant la moyenne pondérée (`c_ij`) de ses voisins ; `mls` — un polynôme
de degré 2 ajusté par moindres carrés mobiles sur les représentants à deux anneaux ; `ctransf`
— `w_i = max_l (w_l − |p_i − p_l|²)`, l'ancienne. Les corrections : **`penal`** — la lecture
invariante par jauge de « `((1−t) I + t M) w* = t w` » : on ne peut pas faire décroître `w*`
vers zéro (une constante ne change aucune cellule, la prolongation ne doit pas en dépendre), on
relâche la condition imposée, `w_k = (moy_j w_j + μ w_k^c)/(1 + μ)`, `μ = t/(1−t)`, `t = 1`
exact, `t → 0` la constante ; `retrait` — `t·w`, l'homothétie vers Voronoi ; `jacobi` — `k`
balayages de Jacobi amorti sur tout le niveau ; `rattrape` — relever chaque cellule vide à
`−ψ(p_i) + marge·h_i²` (le germe rentre dans sa cellule), et recommencer.

## 8.1 Ce que ça donne

Lignes `s = 0.005` (le cas de la suite), `--threads 8`, `R = 8`, niveaux 100000 / 16384 / 2048 /
256. Le niveau 256 converge en 5 itérations, comme avant. La prolongation harmonique vers 2048 :
**1296 cellules sur 2048 sous le plancher, 1240 vides** ; vers 10⁵ : 22 025 vides à `t = 1`, et
`penal` n'en enlève presque pas (`t = 1/2`, 22 068 ; `t = 1/16`, 19 373 ; **`t = 2.4e-4`, 7 521**).
Aucune prolongation ne passe avant `t ≈ 6e-5` avec `retrait` — et ce `w` là *est* Voronoi :
Newton y refait ses 19 itérations et 40 diagrammes (`mls`, 20 et 45), ou stagne sur les cellules
encore vides (`harmonique`, `copie`, `ctransf`). Les niveaux intermédiaires sont dans le même cas.

Pour vérifier que ce n'est pas la faute de la prolongation, des nuages moins contrastés ont été
tirés (`scripts/nuage_lignes.py --sigma 0.1 | 0.05 | 0.02`, sans le clip de `gen_cases.py` qui
confond des germes dans les coins dès que la ligne s'épaissit) :

| σ | Newton depuis `w = 0` | `harmonique`, vides à `t = 1` | `mls`, vides à `t = 1` | `t` qui passe | Newton depuis là |
|---|---|---|---|---|---|
| 0.1   |  9 it, 10 diag |  14 757 / 10⁵ |   681 | 1/128 – 1/64 |  **9 it, 10 diag** ; `mls` 10 it, 13 diag |
| 0.05  | 11 it, 16 diag |  41 320 | 1 042 | 1/256 – 1/128 | **11 it, 16 diag** |
| 0.02  | 15 it, 26 diag |  79 238 | 2 792 | 1/2048 – 1/128 | **15 it, 26 diag** |
| 0.005 | 19 it, 40 diag |  22 025 | (96 %) | 6e-5 | 19–20 it |

**Le niveau fin refait exactement le travail de Newton depuis Voronoi**, à contraste égal, et le
multi-échelle coûte les niveaux grossiers en plus (+1 s). `mls` interpole dix à cinquante fois
mieux que l'harmonique (la courbure, voir plus bas), et ça ne change rien : 0.7 % de cellules
vides suffisent.

## 8.2 Pourquoi : l'admissible est un fil, et il ne survit à aucun lissage

**La borne.** Prendre la SOLUTION fine (`FILE_equal`), la lisser par `k` balayages de Jacobi sur
le graphe de Voronoi, et repartir de là (`--lisse-solution k`). `s = 0.1` : **un seul balayage
vide 374 cellules**, et Newton stagne ; `s = 0.005` : 21 851 vides après un balayage, 74 755
après 64. Le point de départ le plus proche qui se puisse imaginer n'est pas admissible dès
qu'on le touche — aucune prolongation d'un niveau grossier, qui ne connaît pas la solution fine,
ne fera mieux.

**Le mécanisme (corrigé par l'étude 1D, § 8.3).** Avec `w_lin` la valeur en `p_i` de la corde de
`w` entre les deux voisins, la cellule est non vide tant que `w_i − w_lin ≥ −h̄_i²`, et la MARGE
`m_i = 1 + (w_i − w_lin)/h̄_i²` vaut *exactement* (en 1D) `a_i / |Vor_i|` : **la marge d'une cellule
est son taux de compression par rapport à Voronoi**. Voronoi est à 1 partout ; la solution est à
`(1/n)/|Vor_i|` — grande là où les cellules s'étirent (la bande), *petite là où elles doivent
rétrécir* : les germes des queues gaussiennes, dont la cellule de Voronoi vaut dix à mille fois la
cible. Une perturbation de `w` de courbure locale `κ` coûte `κ/2` de marge, et `w` porte, à chaque
germe, une composante *à l'échelle de la cellule* de taille `(1 − m_i) h̄²` — c'est elle qui rend les
aires égales malgré des cellules de Voronoi qui fluctuent d'un facteur 10 d'un germe à l'autre. Un
lissage la brasse entre voisins ; une prolongation ne la connaît pas et n'apporte que la partie
lisse, avec ses erreurs de courbure : l'harmonique annule le laplacien entre les représentants et
concentre toute la courbure de `w` en **plis sur les représentants**, `(H_c/h)·w''`, du mauvais
côté là où `w` est convexe (`w'' = 2(1 − S) > 0` ⇔ compression `S < 1`) — même sur l'uniforme,
1705 vides ; la copie saute de `∇w · H_c` entre deux paquets ; le `mls` porte la courbure mais
hérite du bruit de discrétisation du niveau grossier, `O(H_c²) = O(R h²)`. Le `t` qui passe est celui
qui ramène ces erreurs sous `m_i h̄²` pour les germes les plus comprimés : Voronoi.

**Le rattrapage cascade.** Relever les vides à `−ψ(p_i) + marge·h_i²` : `s = 0.1`, 27 vides
sur 2048 deviennent 696 après 20 passes, 374 sur 10⁵ deviennent 54 185 (`marge` 0.1, 0.01 ou
0.001) — relever un germe lui fait prendre l'aire de ses voisins, qui étaient sur le fil aussi.
C'est ce que `2d_des_familles` avait vu (« le relèvement d'une cellule vide en vide d'autres »).

**Ce qui reste vrai.** Voronoi est le point le plus intérieur de l'admissible (marge 1 partout), et Newton depuis là coûte 5, 7, 11, 19 itérations pour 256,
2048, 16384, 10⁵ germes sur le cas dur, 9 à 19 selon le contraste à 10⁵ : ce n'est pas le
départ qui manque, c'est le nombre d'époques combinatoires à traverser, et il ne dépend que de la
distance entre Voronoi et la solution. Le multi-échelle ne peut payer que là où une prolongation
serait admissible *sans* correction — il faudrait pour ça qu'aucune cellule n'ait à se comprimer
beaucoup (`m_i` jamais petit), et alors Newton depuis zéro converge déjà en 6 itérations (l'uniforme).

## 8.3 En 1D, où tout se dessine (`scripts/multiechelle_1d.py`)

`python scripts/multiechelle_1d.py [-n 400 --sigma 0.005 -R 8]` écrit `figures/multiechelle_1d_*.png`.
En 1D la cellule de `i` est non vide ssi son point relevé `(p_i, p_i² − w_i)` est un sommet de
l'enveloppe convexe inférieure, la marge est exacte, et l'extension harmonique du graphe de Voronoi
est l'interpolation *linéaire* en `p`. Le cas : la moitié des germes en amas gaussien (`S ≈ 40`),
l'autre moitié uniforme. Quatre figures : le problème (cellules, `w`, points relevés — la solution est
une ligne presque droite dans l'amas) ; la marge de la solution (`= a_i/|Vor_i|`, vérifié point par
point : 40 dans l'amas, 0.1 à 0.5 au fond) et celle de la solution lissée (un balayage de Jacobi :
28 vides, tous au fond) ; les quatre prolongations, points relevés et marges (copie 366 vides,
harmonique 79 — ses représentants là où `w` est convexe, marges de −1 à −100, les germes libres
restant exactement à 1 —, spline 7, c-transformée 101) ; le retrait `t·w` et, pour chaque germe, la
marge de la prolongation contre la compression de la solution — l'harmonique renvoie l'image miroir
de la solution autour de 1, amplifiée `H_c/h` fois, la spline la suit.

**Une différence avec la 2D à garder en tête :** en 1D `a(w)` est *linéaire* tant qu'aucune cellule
ne se vide, et le segment de tout départ admissible vers la solution reste admissible (`a(t) =
(1−t) a₀ + t ν > 0`) — Newton converge en **un pas**, depuis Voronoi comme depuis `t·w`. La 1D ne
montre donc que l'admissibilité de la prolongation, pas ce que Newton coûte ensuite ; la seconde
moitié du constat 2D (« le niveau fin refait le travail de Newton depuis Voronoi ») tient à ce que
le seul `t` admissible *est* Voronoi.

## 8.4 Adoucir plutôt qu'aplatir (`scripts/adoucissement_1d.py`)

Deux « lissages » ont été confondus plus haut, et ils n'ont rien à voir : *interpoler* les poids
grossiers par une fonction lisse (la spline, le `mls`) — ça va dans le bon sens, la spline n'a que
7 vides sur 400 — et *filtrer* une `w` donnée sur le graphe fin (Jacobi, la proposition `M`), qui
brasse entre voisins la composante à l'échelle de la cellule. La question était : peut-on rendre
une bonne prolongation admissible en l'*adoucissant* (passe-bas, noyaux de plus en plus larges)
plutôt qu'en l'aplatissant vers Voronoi ? Mesuré en 1D, `figures/adoucissement_1d_*.png` :

| famille | vides selon le paramètre | admissible ? | résidu ℓ² du départ (Voronoi : 1.42) |
|---|---|---|---|
| filtre gaussien de la spline, λ = 5e-4 → 0.2 | 16 → 5 → 14 | jamais | 0.8 → 2.1 |
| filtre gaussien de l'harmonique | 80 → 0 → 14 | à λ = 0.02 | **2.0** (pire que Voronoi) |
| régression polynomiale locale, degré 0 / 1 / 2 | 316 → 43 / 184 → 14 / 197 → 24 | jamais | 2 à 6 |
| ridge à noyau gaussien (λ, α) | 130 à 400 | jamais | 2 à 14 |
| spline de lissage (`lam`) | 7 → 59 | jamais | 0.7 → 2.4 |
| **enveloppe** de la spline, ε = 0.01 → 0.5 | 0 | **oui** | **0.70** |
| enveloppe de l'harmonique | 0 | oui | 1.99 |

**Pourquoi le passe-bas ne peut pas marcher.** Pour une `w` lisse, `m = 1 − w''/2` : l'admissible
est la borne *à sens unique* `w'' < 2`. La solution a `w'' = 2(1 − S)` — très négatif dans l'amas
(−78), proche de 2 au fond où les cellules se compriment — et sa forme est une tente : un pic
concave, des flancs presque droits qui portent, étalée sur tout le domaine, la courbure positive
qu'il faut pour redescendre. Un filtre symétrique prend la concavité du pic et la rend en
convexité sur ses flancs, `~ (saut de pente)/λ`, bien au-dessus du jeu `2S` disponible : il faudrait
`λ > 1`. Les noyaux gaussiens étroits *sonnent* (courbure `A/λ²`), les larges effacent le pic ; les
vides ne tombent jamais à zéro et le résidu remonte au-dessus de Voronoi.

**L'adoucissement à sens unique : l'enveloppe.** Admissible à marge `ε` ⇔ `(1−ε) p² − w` est
convexe (points relevés). La projection est donc l'**enveloppe convexe inférieure** de
`ψ = (1−ε)p² − w` : `w ← (1−ε)p² − H(p_i)`. Elle ne touche que les germes dont le point relevé
est au-dessus de l'enveloppe (les 79 représentants pliés de l'harmonique, les 7 bouts de la
spline), les remonte du *minimum*, d'un coup, sans cascade : tout point est alors un sommet de
marge ≥ ε. Prendre `ε` sous la plus petite compression de la solution (~0.1 ici), sinon la
projection aplatit aussi le fond. Le même objet s'écrit **germe par germe** : une cellule vide
*naît en un sommet* `v` du diagramme des autres, et le poids qui l'y fait naître est

    w_i = min_v ( |p_i − v|² − ψ(v) ),   ψ(v) = min_j ( |v − p_j|² − w_j )

(`H(p) = max_v [2v·p − |v|² + ψ(v)]`, les bouts du domaine comptant comme sommets). C'est le
**relèvement minimal** — le rattrapage de `2d_des_familles` relevait à `−ψ(p_i)`, c'est-à-dire
jusqu'à mettre `p_i` *dans* sa cellule, condition bien plus forte quand la cellule est transportée
loin, d'où la cascade. Fait une cellule à la fois (le diagramme refait entre deux), il converge à
un ping-pong près à l'échelle de `ε` entre vides adjacentes ; d'un coup par l'enveloppe, il est
exact. En 2D, c'est une bissection sur `w_i` seul avec le moteur de cellule (l'aire de `i` en
fonction de son poids), sur les 0.7 % de cellules que le `mls` laisse vides — à essayer.

Ce que ça vaut : spline + enveloppe donne un départ admissible à résidu **0.70 contre 1.42** pour
Voronoi (la solution lissée d'un balayage puis projetée : 0.56) ; l'harmonique projetée reste à
1.99 — ses représentants relevés naissent minuscules là où la solution voulait `1/n`. En 1D Newton
finit en un pas quel que soit le départ ; ce que vaut un résidu divisé par deux se mesure en 2D.

**Bibliographie (de mémoire, à vérifier avant de citer).** Lissage à noyaux : Nadaraya (1964),
Watson (1964) ; régression polynomiale locale : Cleveland (1979, LOESS), Fan & Gijbels, *Local
Polynomial Modelling and Its Applications* (1996) ; splines de lissage : Reinsch (1967), Wahba,
*Spline Models for Observational Data* (1990) ; moindres carrés mobiles : Lancaster & Šalkauskas
(1981), Levin (1998) ; approximation par fonctions radiales et ridge à noyau : Wendland,
*Scattered Data Approximation* (2005), Schölkopf & Smola, *Learning with Kernels* (2002).
Régression sous contrainte de convexité (le bon cadre pour `w'' < 2`) : Hildreth (1954), Seijo &
Sen (2011), Lim & Glynn (2012). Enveloppe convexe et c-concavité en transport : Villani, *Optimal
Transport, Old and New* (2009, ch. 5) ; Aurenhammer, Hoffmann & Aronov (1998) pour le relevé
diagramme de puissance ↔ enveloppe inférieure ; multi-échelle et amortissement : Mérigot (2011),
Kitagawa, Mérigot & Thibert (2019), Lévy (2015). Prolongation lissée en multigrille : Vaněk, Mandel
& Brezina (1996, agrégation lissée).

## 8.5 Le relèvement minimal en 2D : là où le multi-échelle commence à payer

`--corr releve` : sur les cellules que la prolongation laisse sous le plancher, une **bissection sur
le poids de la cellule seule** (`cellule_avec_poids`, le moteur avec un autre `w_i`, l'arbre
inchangé) entre `w_i` et `−ψ(p_i)`, jusqu'à une aire dans `[ε, 2ε] × min(ν_i, |Vor_i|)` —
c'est le poids de naissance de la cellule, à `ε` près. Toutes les vides d'une passe sur le même
diagramme, six passes au plus (deux vides nées au même sommet se disputent la place ; doubler la
cible à chaque reprise, essayé, fait tout exploser : 2 millions de relèvements).

Mesuré sur la solution lissée d'un balayage (`s = 0.1`, 374 vides) : 469 relèvements en 6 passes,
**0 vide, pas de cascade** — là où le rattrapage à `−ψ(p_i)` en fabriquait 54 000. Et Newton
depuis là : **10 itérations, 13 diagrammes**, contre 9 et 10 depuis Voronoi. Même à un balayage de
la solution, une fois réparé, le départ ne fait pas gagner une itération : ce qui compte pour
l'amortissement est le *pire* résidu (`max|a−ν|/ν = 80` ici, les cellules nées minuscules), pas la
distance ℓ².

`mls` + relèvement, `R = 8`, `--threads 8`, contre Newton depuis Voronoi :

| σ | vides du `mls` au niveau fin | relèvements | fine : Newton | total | référence |
|---|---|---|---|---|---|
| 0.1   |   681 |   836 | **8 it, 11 diag** (2.7 s) | 4.25 s | 9 it, 10 diag, 2.84 s |
| 0.05  | 1 042 | 1 134 | **9 it, 12 diag** (3.1 s) | 4.06 s | 11 it, 16 diag, 3.64 s |
| 0.02  | 2 792 | 3 077 | **8 it, 11 diag** (2.4 s) | **3.95 s** | 15 it, 26 diag, 4.40 s |
| 0.005 | ~5 000 | 343 000 : cascade | stagne | — | 19 it, 40 diag, 7.1 s |

Première fois que le niveau fin coûte *moins* que Newton depuis Voronoi — la moitié à `s = 0.02`
(11 diagrammes contre 26) — et le premier gain sur le total (10 %), mangé aux deux tiers par les
niveaux grossiers (0.35 s) et le relèvement (0.86 s : six diagrammes complets et 47 000 cellules).
Le cas dur reste hors de portée : à `s = 0.005` les germes comprimés à `S ~ 1e-3` sont des
milliers, leurs naissances interfèrent, et le relèvement cascade. Le relèvement pourrait ne
re-mesurer que le voisinage de ce qu'il relève, et les niveaux grossiers s'arrêter plus tôt
(`--n-min`) : à faire si la piste est retenue.

## 8.6 Prolonger la GÉOMÉTRIE et non le potentiel : une garantie, et pourquoi elle ne sert pas

Les § 8.1–8.5 disent que le lissage ne peut pas rendre une prolongation admissible (`w'' < 2` est une
borne à sens unique) et que le seul remède mesuré est l'enveloppe. Restait une famille d'idées
entièrement différente, jamais essayée : ne pas prolonger `w`, mais **prolonger l'objet géométrique**
— les frontières, ou la triangulation. Elle est explorée ici jusqu'au bout. Réponse courte : elle
donne une **garantie non triviale** d'admissibilité, calculable sur le niveau grossier seul, et cette
garantie est **inutilisable en 2D** pour une raison de valeur extrême. Les trois scripts qui portent
la mesure : `scripts/frontieres_1d.py`, `scripts/triangulation_1d.py`, `scripts/cstar_2d.py`
(ce dernier lit les niveaux écrits par `multiechelle --ecrire-niveaux PREFIX`).

### 8.6.1 En 1D, se donner les frontières : spectaculaire, et c'est le pas de Newton déguisé

En 1D on peut se donner les frontières `b` entre cellules fines et en **déduire** les poids, exactement,

    w_{i+1} = w_i + ( p_i + p_{i+1} − 2 b_i )( p_{i+1} − p_i )

et l'admissibilité est *gratuite* dès que `b` est croissante. Deux façons de se donner `b` depuis le
niveau grossier : `sousdiv` — garder les frontières grossières et couper chaque cellule grossière en
parts égales entre ses enfants ; `barycentres` — un linspace entre les barycentres des cellules
grossières, les bords du domaine en ancres. Mesuré (`n = 400`, `σ = 0.005`, `R = 8` ; résidu ℓ²
`|a−ν|/|ν|`, Voronoï à 1.42) :

| prolongation | vides | résidu max | ℓ² |
|---|---|---|---|
| harmonique | 79 | 6.88 | 1.99 |
| spline | 7 | 2.89 | 0.70 |
| spline + enveloppe (§ 8.4, le meilleur connu) | 0 | 2.89 | 0.70 |
| **frontières, barycentres** | **0** | **0.14** | **0.017** |
| **frontières, sous-division** | **0** | **0.00** | **0.000** |

Quatre-vingts fois mieux que Voronoï, admissible par construction. **Et c'est un accident de
dimension**, vérifié à 9e-11. Projeter une tessellation prescrite sur les `w` réalisables, au sens des
moindres carrés sur les arêtes,

    min_w Σ_ij ω_ij ( w_i − w_j − c_ij )²,   ω_ij = |f_ij| / 2 d_ij,   c_ij = 2 d_ij s_ij

(`s_ij` le déplacement voulu de la facette entre `i` et `j`), donne les équations normales
`L w = div c` avec

    ( div c )_i = Σ_j ω_ij c_ij = Σ_j |f_ij| s_ij = LA VARIATION D'AIRE VOULUE

— c'est-à-dire **un pas de Newton, et rien d'autre** : la projection ne retient de la tessellation que
sa *divergence*. En 1D il y a `n−1` arêtes pour `n−1` inconnues de jauge, donc résidu des moindres
carrés **nul** (la tessellation est réalisée exactement) et le pas de Newton est exact (§ 8.3) : **les
deux moitiés du miracle 1D sont le même fait.** En 2D il y a ~3n arêtes pour n inconnues, en 3D ~15n :
les 2n (ou 14n) autres contraintes sont jetées, et ce qui survit est la contrainte de *masse*, qu'on
connaissait déjà. Prescrire la tessellation n'apporte **aucune information nouvelle en nD** — c'est
l'obstruction de réalisabilité des figures réciproques, lue en comptage de degrés de liberté.

Corollaire qui explique tout le § 8 d'un coup : **les données du problème (`ν`) sont déjà au niveau
fin.** Le niveau grossier ne peut apporter que la partie *lisse* du potentiel, et l'admissibilité vit
dans la composante à l'échelle de la cellule, sur laquelle il n'a aucune information.

### 8.6.2 Une hypothèse fausse : le pas diagonal ne reconstruit pas la compression

L'identité 1D `m_i = a_i/|Vor_i|` suggère que la composante à l'échelle de la cellule vaut
`( S_i − 1 ) h̄²` avec `S_i = ν_i/|Vor_i|` — une quantité **lisible au niveau fin sans rien résoudre**.
L'hypothèse était qu'un pas de Jacobi sur le **résidu** (`w_i += ω ( ν_i − a_i ) / ( da_i/dw_i )`, à ne
pas confondre avec `lisse_jacobi`, qui est un pas sur le système *homogène* et enlève la haute
fréquence au lieu de l'injecter) la reconstruit. **Elle est fausse** : la marge obtenue est
*anti*-corrélée à `S` (−0.77), parce que le pas déplace aussi les voisins, donc la corde à laquelle la
marge se mesure.

Ce que le pas diagonal fait vraiment, mesuré, et son défaut : il baisse le résidu ℓ² et le résidu max,
mais **il crée des vides sur les prolongations**. Après enveloppe, à `ω = 1/2`, un balayage : la spline
passe de 0 vide / 2.89 à **7 vides** / 2.27 (ℓ² 0.698 → 0.529) ; l'harmonique de 0 à 97. Seule la
*solution* fine lissée d'un balayage puis projetée reste admissible : 0 vide, `max|a−ν|/ν` **2.23 →
1.05**, ℓ² 0.539 → 0.260. Comme le § 8.5 montre que c'est le *pire* résidu qui étrangle
l'amortissement après un relèvement, la piste garde un intérêt — mais elle demande une seconde passe
de relèvement derrière, donc un `relève → diag → relève`, non mesuré.

### 8.6.3 La cellule grossière ne peut pas être l'unité : ce que le relevé dit du couplage

L'idée naturelle est de **résoudre exactement dans chaque cellule grossière** : le sous-problème est
équilibré en masse par construction, donc admissible à l'intérieur. Le relevé dit exactement ce que ça
coûte. Ajouter une constante aux poids d'un agrégat **translate verticalement le nuage relevé de cet
agrégat**, et rien d'autre. Donc :

* la propriété locale « aucune cellule vide dans l'intersection avec sa cellule grossière » est
  **invariante de jauge** : c'est ce qui rend l'idée bien posée ;
* la condition globale (tout point relevé sommet de l'enveloppe inférieure) ne l'est pas, et elle
  s'écrit comme un jeu d'inégalités **linéaires** en les constantes : le couplage entre agrégats est
  exactement **un scalaire par agrégat**, soit une faisabilité linéaire à `n/R` inconnues, non `n`.

Mais une translation verticale ne corrige pas un nuage dont la *forme* est globalement non convexe, et
la résolution locale fixe la forme. Surtout, **l'emboîtement exact `Cell_i ⊂ K_q` est géométriquement
impossible en nD** : une facette entre `i ∈ q` et `j ∈ r` a pour normale `p_i − p_j`, jamais parallèle à
`P_q − P_r`, donc **la frontière grossière ne peut pas être une face du diagramme fin**. Les cellules
fines du bord la traversent quelles que soient les `w`. En 1D toutes les normales sont parallèles —
encore la même dégénérescence.

### 8.6.4 La triangulation régulière est le bon objet, et elle donne une garantie

La contrainte qui tue une cellule est une inégalité à `D+1` points : `u_i = |p_i|² − w_i` doit être sous
l'interpolé affine des autres sur un simplexe contenant `p_i`. **Le bon objet local du niveau grossier
est donc sa triangulation régulière** (le dual du diagramme de puissance grossier), pas ses cellules —
et les deux sont duales, donc on ne peut pas avoir les deux : la *cellule* est l'unité de la **masse**
(équilibrée par construction), le *simplexe* est l'unité de l'**admissibilité**. Résoudre « dans les
triangles » n'a pas de sens comme sous-problème d'OT : le nombre de germes fins d'un triangle et son
aire n'ont aucune raison de correspondre. Le simplexe est une unité de **certificat**.

Avec `Uh` l'enveloppe convexe inférieure du relevé grossier (affine par simplexe), `Q` l'interpolé
affine de `|P|²` sur la **même** triangulation et `g = Q − |p|² ≥ 0` l'écart au paraboloïde :

    u = Uh − c g        soit        w = ( 1 − c )( |p|² − Uh ) + c · w̃

où `w̃` est l'interpolation barycentrique des poids grossiers (`c = 1`). Deux faits, démontrés :

* **dans** chaque simplexe, `u = affine + c |p|²` est *strictement convexe* dès que `c > 0` : tous les
  germes fins d'un même simplexe grossier sont automatiquement des sommets, **sans rien résoudre**.
  Toute cellule vide vient donc d'une *facette* ;
* à une facette, le pli de `u` vaut `pli( Uh ) − c · pli( Q )`, avec `pli( Uh ) ≥ 0` puisque `Uh` est
  convexe. D'où **zéro cellule vide, garanti**, pour tout

      0 < c ≤ c* = min sur les facettes où pli( Q ) > 0 de  pli( Uh ) / pli( Q )

  une borne **calculable sur le niveau grossier seul**, sans relèvement, sans bissection, sans cascade.

En 1D (`n = 400`, `σ = 0.005`, `R = 8`) la garantie tient, la borne est serrée, et la marge minimale
vaut exactement `c` :

| `c` | vides | marge min | ℓ² |
|---|---|---|---|
| 0.1 c\* | 0 | +0.03 | 2.57 |
| **c\* = 0.2814** | **0** | **+0.28** | 2.07 |
| 1.05 c\* | 0 | +0.24 | 2.05 |
| 1.5 c\* | 3 | −0.26 | 1.94 |
| 3 c\* | 68 | −73 | 1.97 |
| 1 (barycentrique nu) | 81 | −106 | 1.99 |

Et une identité, sur quatre cas (`σ` 0.002 à 0.1, `R` 4 à 16) : **`c*` = la compression grossière
minimale** (0.2814 contre 0.281 ; 0.3389 contre 0.339 ; 0.2008 contre 0.201). La localisation des
morts confirme le théorème : **la marge vaut exactement 1 partout sauf aux germes grossiers** (min −106
là, 53 des 81 vides à un germe près d'un représentant). C'est aussi le mécanisme exact de l'échec de
`--prol harmonique` (§ 8.1) : interpoler les *poids* grossiers impose un pli à l'échelle `H` là où les
germes sont espacés de `h`, donc une marge `1 − O( H/h ) = 1 − O( R^{1/D} )`.

**Mauvaise nouvelle dès la 1D :** à `c = c*` le départ garanti a un résidu **pire que Voronoï** (2.07
contre 1.42) dans les quatre cas. Acheter une marge `c` tire `w` d'une fraction `1 − c` vers l'objet
dégénéré `|p|² − Uh`.

### 8.6.5 Le test décisif en 2D : `c*` vaut 1e-3 à 4e-5, et décroît en 1/n

`multiechelle --ecrire-niveaux PREFIX` écrit chaque niveau résolu (`n`, `D`, puis `x.. w ν`) ;
`scripts/cstar_2d.py` en calcule `c*` par l'enveloppe inférieure des points relevés. Sur
`lines5_n100000`, `R = 8`. Le banc déclare admissible « toute aire `≥ 0.5 × min( ν_i, aire min de
Voronoï )` » : une marge `c*` ne franchit ce plancher que si `c* ≳ 0.5`.

> **CORRECTION (§ 8.7).** Ce critère-là est le mauvais, et la comparaison des `c*` au plancher du banc
> ne prouve rien : le plancher d'amortissement de Newton n'est pas absolu, il **s'adapte au départ**
> (`eps = 0.5 min( min ν, min a₀ )`). Des cellules minuscules sont donc parfaitement légales, et le
> § 8.7 mesure que parmi les départs à zéro vide un résidu quatre fois pire ne coûte qu'**une**
> itération. La construction par la triangulation échoue quand même, mais pour une raison mesurée
> ailleurs : dans un simplexe `u = affine + c|p|²`, donc `w_i = ( 1 − c )|p_i|² + ℓ( p_i )`, et le
> bissecteur donne `Cell_i = c · Vor_i − a/2` avec `a = ∇ℓ` — le motif du simplexe **contracté de `c`
> et translaté de `−a/2`**. Mesuré sur le vrai niveau grossier : `|a/2|` médian vaut **0.80** dans un
> domaine de côté 1, pour tout `c < 1` (0.80 à `c*`, 0.72 à 0.1, 0.55 à 0.3, et 0.09 seulement à
> `c = 1`), et 100 % des centres prédits tombent hors du domaine. **Le théorème garantit « non vide
> dans R² », pas « non vide dans `[0,1]²` »** — c'est l'antagonisme du § 15.16 (contraction contre
> compatibilité de niveau), par simplexe, et `c = 1` est la prolongation barycentrique déjà rejetée.

| σ | n du niveau | `c*` | compression grossière min | médiane de `c_F` | facettes contraintes | sous le plancher |
|---|---|---|---|---|---|---|
| 0.1 | 256 | 9.89e-4 | 1.72e-1 | 1.07 | 92 % | ×506 |
| 0.1 | 2048 | 2.45e-4 | 4.21e-2 | 1.05 | 91 % | ×2 042 |
| 0.1 | 16384 | **3.53e-5** | 1.01e-2 | 1.07 | 91 % | ×14 160 |
| 0.05 | 2048 | 6.59e-4 | 1.90e-2 | 1.04 | 88 % | ×759 |
| 0.05 | 16384 | 2.10e-4 | 3.20e-3 | 1.10 | 88 % | ×2 386 |
| 0.02 | 2048 | 2.02e-3 | 1.75e-2 | 1.04 | 83 % | ×248 |
| 0.02 | 16384 | 9.76e-5 | 2.38e-3 | 1.11 | 82 % | ×5 123 |
| 0.005 | 2048 | 6.37e-4 | 2.22e-2 | 0.97 | 79 % | ×785 |
| 0.005 | 16384 | **3.88e-5** | 2.80e-3 | 1.08 | 77 % | ×12 902 |

Trois faits, et la piste est fermée :

1. **`c*` ne suit pas la difficulté du cas.** À `n = 16384`, `σ = 0.1` donne 3.53e-5 et `σ = 0.005`
   donne 3.88e-5 — le même nombre — alors que la compression grossière minimale passe de 1.0e-2 à
   2.8e-3. **L'identité 1D `c* = min compression grossière` est fausse en 2D** : elle valait pour une
   raison de dimension, comme le reste de la 1D. La prédiction qui avait motivé le test (« `c*`
   s'effondrera là où le cas est dur ») était donc mal posée : `c*` est déjà effondré partout, le cas
   le plus facile compris.
2. **`c*` décroît en ~1/n** : 9.89e-4 (256), 2.45e-4 (2048), 3.53e-5 (16384). **Raffiner dégrade la
   garantie** — exactement le contraire de ce qu'un multi-échelle demande. Disqualifiant en soi.
3. **Et c'est un effet de valeur extrême, pas de géométrie du transport.** La médiane de `c_F` vaut
   **1.05** : la facette *typique* autoriserait `c > 1`. `c*` est le minimum sur ~3n facettes d'une loi
   qui a de la masse en 0, et ce sont les facettes **presque dégénérées** de la triangulation régulière
   grossière — quatre germes grossiers presque co-sphériques au sens du relevé, donc `pli( Uh ) → 0` —
   qui l'imposent. Elles sont partout et leur nombre croît avec `n`. Aucun réglage ne rattrape un `min`
   sur 3n tirages.

Ça enterre au passage le raffinement qui semblait évident (un `c_T` par simplexe, résolu en LP : la
condition est par facette et linéaire en les `c`). Il donnerait bien `c ≈ 1` loin des facettes
dégénérées, mais à une facette quasi plate avec `pli( Q ) > 0` la contrainte force `c → 0` de toute
façon, et les germes voisins retomberaient sur le relèvement : on aurait reconstruit `mls +
relèvement`, en plus cher.

### 8.6.6 Ce qui survit

Un seul résidu, mais il est réel : **le certificat se calcule sur le niveau grossier seul et dit *où*
le niveau fin sera malade** — au voisinage des facettes à petit `c_F` — avant de construire le moindre
diagramme fin, là où `releve_minimal` les découvre aujourd'hui en construisant le diagramme fin et en
le testant (et le § 15.9 documente un bug où « la réparation cherchait les malades là où ils avaient
disparu »). À mesurer : le recouvrement entre les cellules vides du niveau fin après `--prol mls` et le
voisinage des facettes à petit `c_F`.

Les deux autres pistes vivantes, par ordre de rendement : le balayage diagonal amorti après
`--corr releve` (§ 8.6.2 ; le `max|a−ν|/ν` de départ est à 1.6e3 à `σ = 0.005`), avec la seconde passe
de relèvement qu'il impose ; et le chaînage des cellules nées du relèvement sur la réparation par amas
du § 15.11–15.18 (départ harmonique, plafond de taille, gradient local), au lieu de passer au Newton
global un départ à `max|a−ν|/ν = 80`. Le § 8.5 fait la passation directement.

**Bibliographie (de mémoire, à vérifier avant de citer).** Diagramme de puissance ↔ enveloppe convexe
inférieure du relevé, et triangulation régulière comme dual : Aurenhammer (1987), Aurenhammer, Hoffmann
& Aronov (1998) ; triangulations régulières et leurs dégénérescences : Ziegler, *Lectures on Polytopes*
(1995, ch. 5), De Loera, Rambau & Santos, *Triangulations* (2010) ; réalisabilité d'une tessellation
comme diagramme de puissance (figures réciproques) : Ash & Bolker (1986), Aurenhammer (1987) ;
multi-échelle en transport semi-discret : Mérigot (2011), Lévy (2015), Kitagawa, Mérigot & Thibert
(2019).

## 8.7 L'ADMISSIBILITÉ SEULE SUFFIT — et le plancher qui s'effondrait sur une cellule vide

Tout le § 8 juge les départs sur leur *résidu*, et le § 8.5 conclut que « ce qui compte pour
l'amortissement est le pire résidu (`max|a−ν|/ν = 80`), pas la distance ℓ² ». La question posée ici
renverse le critère : **si le départ est admissible, combien reste-t-il vraiment à faire ?** La réponse
est « deux ou trois diagrammes », et elle change la cible de tout le chapitre.

### 8.7.1 Parmi les départs à zéro vide, le résidu ne coûte presque rien

Famille de départs de qualité décroissante, tous admissibles : la solution fine lissée par `k`
balayages de Jacobi, puis réparée (`--lisse-solution k --corr releve --passes 30`). `σ = 0.1`,
`n = 10⁵`, contre Newton depuis Voronoï (**9 it, 10 diagrammes**) :

| départ | résidu max | vides | Newton |
|---|---|---|---|
| Voronoï | — | 0 | 9 it, 10 diag |
| lissée 1 balayage + relèvement (6 passes) | 80 | 0 | 10 it, 13 diag |
| lissée 4 balayages + relèvement (19 passes) | 203 | 0 | **10 it, 13 diag** |
| lissée 16 balayages + relèvement (27 passes) | 316 | 0 | **11 it, 15 diag** |

Un facteur **quatre** sur le pire résidu coûte **une** itération. Et la comparaison qui tranche est à
départ *identique*, seule la complétude de la réparation changeant :

| réparation du même départ (lissée 4 balayages) | vides restants | Newton |
|---|---|---|
| relèvement, 6 passes | 42 | **STAGNATION**, 8 it, 61 diag |
| relèvement, 30 passes | 0 | **CONVERGE, 10 it, 13 diag** |

**Ce n'est donc pas le résidu qu'il faut viser, c'est le nombre de cellules vides** — et `0` n'était
pas un objectif approché mais une condition binaire. Pourquoi, c'est la suite.

### 8.7.2 Le plancher d'amortissement s'effondrait à zéro (corrigé)

La trace, sur un départ à **une seule** cellule vide :

```
it 0  |r|_2 8.78e-03  max|a-nu|/nu 2.72e+02     1 vides  alpha* 2.50e-01
it 1  |r|_2 7.46e-03  max|a-nu|/nu 2.37e+02  7750 vides  alpha* 5.00e-01 REFUSE
```

Le premier pas est **accepté** et vide 7 750 cellules. La cause était dans `Newton.h` :

```
eps = 0.5 * min( min nu, min a )        // `min a` sur TOUTES les cellules
```

Une cellule vide met `min a` à zéro, donc `eps = 0`, donc le critère d'acceptation `m2 >= eps` est
satisfait par n'importe quel pas : **le garde-fou d'aire disparaissait en silence**, précisément dans
le cas où il sert. Or un départ avec une poignée de vides est exactement ce que rend une réparation
incomplète.

Le plancher se lit maintenant sur les cellules **vivantes au départ**, et il ne défend que celles-là
(`protegee` dans `Newton.h`) : une cellule déjà vide ne peut pas être remontée par l'amortissement, et
l'exiger au-dessus du plancher refuserait *tout* pas. Non-régression vérifiée — quand aucune cellule
n'est vide au départ, `protegee` vaut 1 partout et le calcul est identique : 9 it / 10 diag à
`σ = 0.1`, 18 it / 30 diag sur le cas dur, inchangés.

### 8.7.3 Le relèvement minimal exact, en UNE enveloppe convexe

Le § 8.4 établit qu'en 1D le relèvement minimal *est* une projection : admissible à marge `ε` ⟺
`ψ = ( 1 − ε )|p|² − w` est convexe, donc on prend son **enveloppe convexe inférieure** `H` et
`w ← ( 1 − ε )|p|² − H( p_i )`. Le banc l'implémente autrement (`--corr releve`) : une bissection sur
le poids de chaque cellule sous le plancher, le diagramme refait entre deux passes — et ça cascade.

Or le même objet en 2D est l'enveloppe convexe inférieure des `n` points relevés **dans R³** : un
`ConvexHull` de 10⁵ points (`scripts/enveloppe_2d.py`, sur le départ écrit par
`multiechelle --ecrire-depart`). Cas dur `σ = 0.005` nettoyé (`n = 99993` : sept germes **confondus**
retirés — le clip de `gen_cases.py` en fabrique à `10⁻⁸` là où l'espacement médian est `4.5·10⁻⁴`, et
c'est ce qui faisait sortir la référence en STAGNATION ; elle **CONVERGE** une fois nettoyée, 18 it,
30 diagrammes). Ce nettoyage-là était fait à la main, sur les seules paires les plus serrées ; il est
devenu l'opération versionnée `cases/nettoie_germes.py`, qui en trouve **56** au seuil d'un pour cent
de l'espacement médian, et le nuage Voronoï de `cases/` est nettoyé en place depuis (§ 23.7). Départ : solution lissée d'un balayage, 21 849 cellules sous le plancher, rmax 274.

| réparation | sous le plancher | **vides dans R²** | relèvements |
|---|---|---|---|
| bissection `--corr releve`, 40 passes | 3 | — | **27 611** |
| **enveloppe exacte, UNE passe** | 78 | **0** | 21 849, tous du minimum |

**21 817 → 0 en une seule passe**, le théorème vérifié numériquement, sans cascade. Mise *après*
l'enveloppe, la bissection **défait** son travail (24 090 relèvements, rmax remonté de 100 à 272) :
c'est le mauvais outil pour finir.

Ce qui reste est le **domaine** : l'enveloppe garantit « non vide dans R² », pas « non vide dans
`[0,1]²` ». On ajoute donc les **miroirs** — l'image d'un germe à travers une paroi, au même poids, fait
de cette paroi son bissecteur exact, donc contient la cellule dans le demi-espace — et on monte `ε` :

| enveloppe | cellules sous le plancher |
|---|---|
| `ε = 1e-3`, sans miroir | 78 |
| `ε = 1e-3`, miroirs bande 0.05 | 35 |
| `ε = 1e-3`, miroirs complets | 26 |
| `ε = 0.3`, miroirs bande 0.05 | **3** |

Jamais zéro : les derniers sont bien des sommets de l'enveloppe, mais d'aire sous la résolution de
l'arithmétique — la non-dégénérescence ne borne pas l'aire par le bas.

**Et `ε` ne peut pas monter librement**, ce qui ferme la fenêtre par le haut. L'ensemble des germes
touchés est `{ marge < ε }` : `ε` dit jusqu'où la projection mord dans la population SAINE. Mesuré
(21 846 cellules réellement vides au départ) :

| `ε` | germes remontés | part du nuage | passes, `H` tronqué | passes, `H` exact |
|---|---|---|---|---|
| 1e-3 | 21 879 | 21.9 % | 1 | 6 |
| 0.03 | 22 637 | 22.6 % | 1 | 1 |
| 0.1 | 25 366 | 25.4 % | 2 | 1 |
| 0.3 | 39 780 | **39.8 %** | 4, NON MONOTONE (67 → 85) | **1** |
| 0.6 | 68 355 | **68.4 %** | **jamais** (oscille 11 → 23 → 13) | **2** |

À `ε = 1e-3` la projection touche exactement les malades ; à `ε = 0.6` elle en touche 68 %, donc
18 000 à 46 000 cellules saines forcées de grossir, leurs voisines rétrécissant d'autant. Le § 8.4
l'avait énoncé en 1D : garder `ε` sous la plus petite compression de la solution (ici `~1e-3`).

**`H` DOIT ÊTRE ÉVALUÉE EXACTEMENT.** Une première version prenait le max des plans des `k = 2000`
facettes les plus proches. `H` étant convexe vaut le max de TOUS ses morceaux affines : tronquer le
max la SOUS-estime, donc **remonte trop**, donc une cellule remontée en avale une voisine — et c'est
la non-monotonie de la colonne « `H` tronqué » ci-dessus. La bonne évaluation est une **localisation**
du germe dans la triangulation projetée de l'enveloppe, par marche de visibilité (partir de la facette
la plus proche, passer à la voisine tant qu'une barycentrique est négative). Sans paramètre, et le
défaut disparaît : `ε = 0.3` passe de quatre passes non monotones à **une**, `ε = 0.6` de « jamais »
à deux.

### 8.7.4 Le cas dur peut converger — mais pas de façon fiable, et pas mieux que Voronoï

Avec le plancher corrigé, une poignée de vides n'est plus fatale, et `σ = 0.005` donne quelque chose
pour la première fois (le § 8.5 le déclarait hors de portée). Mais **il faut lire le `reste`, pas le
mot `CONVERGE`** : sur ce cas la sortie est souvent STAGNATION, qui peut signifier « fini » comme
« mort » selon le résidu atteint. Contre la référence (**18 it, 30 diagrammes, reste 4.9e-08**) :

| départ (solution lissée 1 balayage, puis) | cassées | `H` tronqué | `H` exact |
|---|---|---|---|
| bissection seule, 40 passes | 3 | mort, reste 2.2e+02 | — |
| enveloppe `ε = 1e-3` + miroirs | 35 | mort | mort, reste 1.0e+02 |
| enveloppe `ε = 0.15` + miroirs | 7 | — | mort, reste 8.3e+01 |
| enveloppe `ε = 0.2` + miroirs | 6 | — | **CONVERGE, 18 it, 29 diag** |
| enveloppe `ε = 0.3` + miroirs | 3 | **CONVERGE, 16 it, 27 diag** | mort, reste 5.2e+01 |
| enveloppe `ε = 0.45` + miroirs | 3 | — | mort, reste 4.8e+01 |
| enveloppe `ε = 0.6` + miroirs | 3 | mort | mort, reste 1.0e+00 |

**Le verdict est la FRAGILITÉ.** Une réussite sur six valeurs de `ε`, sans ordre : 0.15 meurt, 0.2
passe, 0.3 meurt, 0.45 meurt. Et la meilleure réussite avec la projection CORRECTE est 18 it / 29 diag
— une **égalité** avec Voronoï, pas un gain. Le 16 / 27 de la colonne « `H` tronqué » a été obtenu
avec l'évaluation FAUSSE de `H`, et il ne survit pas à sa correction.

**Le diagnostic, et il est instructif.** Le max tronqué sous-estimait `H`, donc **sur-remontait** — et
c'est ce qui aidait. Le relèvement exactement MINIMAL fait naître chaque cellule à l'aire la plus
petite possible, ce qui est précisément le défaut que le § 8.5 nomme (« les cellules nées minuscules »,
et c'est le pire résidu qui étrangle l'amortissement). **Minimal au sens de la convexité n'est pas ce
que Newton veut ; ce qu'il veut est une AIRE.** `ε` ne contrôle pas l'aire de la cellule qui naît, il
contrôle la stricte convexité du relevé — deux choses différentes, et c'est pour ça que la fenêtre en
`ε` est erratique.

**Trois suites ont été essayées à partir de là. Les trois échouent, et elles éliminent trois
suspects nommés.** Dans tous les cas le départ est celui à `ε = 1e-3` (35 cellules cassées), sur
lequel Newton meurt à l'itération 0 avec `reste` égal au `rmax` du départ — c'est-à-dire sans
accepter un seul pas.

1. **La marge d'aire `+ δ h_i²`** sur les germes remontés, puis re-projection (`--delta`,
   `--gonfle`) : c'est l'idée de `rattrape`, mais posée sur le relèvement MINIMAL au lieu de
   `−ψ(p_i)`. Sans garantie — abaisser le point relevé de `i` peut faire passer un VOISIN au-dessus
   de l'enveloppe — mais empiriquement douce : le gonflage casse jusqu'à 5 049 cellules et **une
   seule re-projection les répare à chaque fois**. Elle fait ce qu'on lui demande (35 → 10 cellules
   sous le plancher quand `δ` va de 0 à 1) et **Newton meurt identiquement** (`reste` = 99.8, 92.7,
   125, 267 pour `δ` = 0, 1e-2, 0.1, 1).
2. **Le premier ordre sur les itérations de correction** (`--methode lbfgs | cg`, `--precond`,
   `--bascule`, branchés sur `--lisse-solution` ; `newton` ne pouvait partir que de zéro). L-BFGS
   précond `γI` / Jacobi / `L₀⁻¹` et CG : **tous stagnent au `reste` du départ**, pour 52 à 148
   diagrammes. Sans plancher (`--po-sans-plancher`) : deux itérations et `reste` monte à 265. Ce
   n'est donc **ni la direction de Newton** (le premier ordre n'en calcule pas et échoue pareil)
   **ni le plancher d'aire** (l'enlever dégrade).
3. **Le mérite** (`--residu barriere | log`, exposé ici) : l'hypothèse était que les deux recherches
   linéaires acceptent sur `‖a−ν‖₂` alors que l'objet garanti croissant est le dual. Mesuré sur
   trois départs (35, 6, 3 cellules cassées) × deux mérites : **les six meurent à l'itération 0**,
   au `reste` du départ. Et là où `lin` convergeait (`ε = 0.2`), `barriere` et `log` échouent.

**L'INSTRUMENTATION TRANCHE** (`--refus K` : à l'itération `K`, laquelle des deux clauses refuse
chaque essai, et sur quelle cellule). Sur le départ à `ε = 1e-3`, résidu `lin`, échelle complète :

```
t 1.00e+00  AIRE NON ( cellule     0 : a0 3.56e-05 -> 0 )  MERITE NON
t 2.44e-04  AIRE NON ( cellule    70 : a0 4.34e-09 -> 0 )  MERITE ok
t 3.82e-06  AIRE NON ( cellule  1970 : a0 2.56e-12 -> 0 )  MERITE ok
t 2.38e-07  AIRE ok  ( cellule 71280 : a0 7.75e-13 -> 9.08e-13 )  MERITE ok
```

**C'est TOUJOURS la clause d'AIRE**, jamais le mérite. Et les chiffres nomment le mal : `eps` vaut
`2.9e-14` pour une cible `ν = 1e-5` — neuf ordres de grandeur sous elle, parce qu'il se lit sur
`min a₀` et que la plus petite cellule née de la projection fait `5.8e-14`. La cellule qui bloque
**change à chaque essai** (4.3e-9, 2.5e-10, 2.6e-12, 4.7e-11) : ce sont les cellules que le
relèvement vient de faire naître, **exactement à leur seuil**. Le pas ne passe qu'à `t = 2.4e-7`, où
le mérite gagne `1e-9` en relatif — d'où « une itération, aucun progrès ».

**Le relèvement minimal dépose donc le départ sur un PLI du diagramme, par construction.** Tout ce
qui précède s'explique d'un coup : la sur-remontée aidait parce qu'elle écartait du pli ; le mérite
ne changeait rien parce qu'il n'a jamais bloqué ; le premier ordre échouait pareil parce que sa
recherche linéaire porte le même plancher.

**Et `log` ne sauve pas les petites cellules** (l'intuition était qu'elles pourraient passer au prix
d'itérations). Instrumenté : `log` est bien plus permissif sur l'AIRE — il accepte à `t = 3.8e-6`
contre `2.4e-7` pour `lin`, quinze fois plus loin — puis **le mérite prend le relais et bloque**. La
raison est structurelle : avec `g = log x`, une cellule à `a/ν = 5.8e-9` pèse `log(5.8e-9) ≈ −19`
contre `~0` pour une saine, donc le mérite vaut **1617** au lieu de `5.6e-3`. `log` transforme le
mérite en un COMPTE des cellules microscopiques, et aucun pas ne le réduit de la fraction exigée.
C'est le mécanisme du § 9.6 pour la barrière, appliqué à `log` dès qu'il y a des milliers de
microscopiques. Corollaire : rendre le plancher RELATIF (pour que les deux critères « parlent la
même langue ») ne servirait à rien — ça doublerait le blocage au lieu de le lever.

4. **Le `+ δ h²` SANS re-projeter** (`--gonfle 0`), refait parce que la re-projection reposait sur le
   pli les cellules qu'un voisin gonflé venait de faire déborder. Négatif aussi, et l'échelle dit
   pourquoi : `h² ≈ 1e-5 ≈ ν`, donc sortir une cellule de `1e-13` demande `δ ~ 1` — et `δ = 1` en
   vide 5 049 autres. Le `δ` qui soigne les malades est celui qui en fabrique ; il n'y a pas de
   fenêtre (9 / 1 651 / 3 321 / 5 049 vides créés pour `δ` = 1e-3 … 1, et les quatre meurent à
   l'itération 0 au `reste` du départ).

**Le bilan de ces quatre essais tient en une phrase** : le problème n'est pas le critère, c'est que
le départ contient des milliers de cellules microscopiques. Tout critère qui les regarde bloque ;
tout critère qui les ignore accepte des pas qui ne les réparent pas ; et aucun réajustement des
poids *après coup* ne les enlève. **La réparation doit faire naître les cellules avec une aire
réelle À LA CONSTRUCTION** — ce que le relèvement minimal, par définition, ne fait pas. C'est
exactement ce que la bissection de `--corr releve` fait (cible d'aire), au prix de la cascade ; les
deux outils ont donc des défauts complémentaires, et c'est là qu'est la suite.

**À lire avec la bonne réserve.** Ce départ dérive de la SOLUTION (lissée d'un balayage), c'est-à-dire
la *borne* du § 8.2 et non une vraie prolongation. Ce qui est mesuré est donc un majorant de ce qu'une
prolongation pourrait acheter — et ce majorant, correctement calculé, vaut aujourd'hui **zéro**.

### 8.7.5 Ce qui bloque le multi-échelle pour de bon

Essayé : le vrai chemin, `--prol mls --corr aucune` sur le cas dur. **La cascade repart dès le niveau
2** (`n = 2048`, 1 148 cellules sous le plancher à `t = 1`, Newton y stagne), donc le niveau fin part
d'un grossier faux et le dump ne mesure plus rien. Le § 8.5 le disait déjà pour la bissection ; le
constat est que **l'enveloppe doit vivre DANS le banc** pour réparer *chaque* niveau, pas seulement le
plus fin en différé.

C'est donc le travail qui reste, et il est bien délimité : un `--corr enveloppe`, c'est-à-dire une
enveloppe convexe inférieure de `n` points relevés en dimension `D + 1`, appelée à chaque niveau. Le
moteur n'a pas d'enveloppe convexe aujourd'hui (`AaBsp`, `Plan` ne la donnent pas) : soit une
dépendance (qhull), soit un incrémental maison. Le chiffre qui justifie la dépense est celui du
§ 8.7.3 : une passe contre 27 611 relèvements, et zéro vide dans R² au lieu de trois.

### 8.7.6 Le mérite `log` : le seul gain de la session, et il n'a rien à voir avec le multi-échelle

Le test du mérite (§ 8.7.4, point 3) n'a rien donné sur les départs réparés — mais il a donné
quelque chose sur **la référence elle-même**. Newton depuis Voronoï, `essai-limites`, Cholesky,
`lines5_n100000`, `--residu log` (`g = log(a_i/ν_i)`) contre `lin` (`a_i − ν_i`) :

| σ | `lin` | `log` | gain en diagrammes |
|---|---|---|---|
| 0.1 | 9 it, 10 diag | **7 it, 8 diag** | −20 % |
| 0.02 | 15 it, 26 diag | **9 it, 13 diag** | **−50 %** |
| 0.005 | 18 it, 30 diag | **13 it, 20 diag** | −33 % |

Gratuit — c'est un changement de second membre et de mérite, pas de coût par itération — et **le gain
croît avec la difficulté du cas**. `barriere`, lui, perd (33 it, 69 diag à σ = 0.005).

**Et c'est le verdict INVERSE de celui du § 9.6**, qui mesure les mêmes options et conclut que la
barrière et le `log` coûtent dix fois plus. Les deux mesures sont justes ; ce sont deux régimes :

* § 9.6 : densité hétérogène, **continuation** en largeur, des *milliers* de cellules en transition
  à chaque étape. Là, `1/x` (ou `1/x` déguisé en `log`) fait du mérite un minimax où la pire cellule
  décide de tout, et un pas plein en pince toujours quelques-unes de plus qu'il n'en répare ;
* ici : densité **uniforme**, Newton direct depuis Voronoï, pas de transition de masse — la
  population de cellules pincées est petite.

> **CORRECTION (§ 21).** Ce paragraphe attribuait le gain au fait que `log` pondère le résidu par
> `1/ν_i`. C'est faux : dans ce banc `ν_i = 1/n` pour tout `i`, il n'y a aucun étalement de `ν` à
> égaliser. Et il attribuait le gain au **mérite**, alors que la mesure qui sépare les deux rôles
> (§ 21.1) montre que le mérite est **inerte** — le gain est entièrement dans la DIRECTION. Le vrai
> mécanisme est au § 21.3 : `p` est la fraction de l'écart *logarithmique* que le pas réclame, et
> `p = 1` est le seul membre de la famille qui réclame aux cellules affamées un déficit qu'elles ne
> peuvent pas prendre. Les deux réserves de ce paragraphe sont levées au § 21.4 : mesuré aussi **en
> 3D**, en `--pas essais` et avec AMGCL, et le gain y est du même ordre (27 → 16 diagrammes). Et
> `log` n'est pas l'optimum — c'est un point sur un plateau dont l'intérieur (`p = 0.25` à `0.5`)
> fait parfois mieux.

---

# 9. LES DENSITÉS HÉTÉROGÈNES : LA CONTINUATION EN LARGEUR (`densite`)

Jusqu'ici la mesure était Lebesgue sur le carré. `src/solver/Densite.h` ajoute un plancher
uniforme plus une somme de gaussiennes isotropes, `ρ = f + Σ_k m_k G_{σ_k}(x − c_k)`, et la
**convolution** par une gaussienne de largeur `s` — qui, sur une somme de gaussiennes, ne change
que les largeurs : `σ_k' = √(σ_k² + s²)`, le plancher invariant, le support (le carré) inchangé.
`xmake run densite --help` ; 2D seulement.

**La masse d'une cellule sans quadrature de surface.** Le flux `F = f(r)(x − c)` avec
`f(r) = m(1 − e^{−r²/2σ²})/(2π r²)` vérifie `div F = m G_σ`, donc la masse d'un polygone est la
circulation de `F·n` sur son bord, et le long d'une arête `(x − c)·n = d` est constant : la
distance signée du centre à la droite. Reste une intégrale 1D par arête et par gaussienne, d'un
intégrande *lisse* (c'est `−expm1`, à l'échelle `max(|d|, σ)` près du pied de la perpendiculaire,
puis `1/t²`) : des morceaux géométriques depuis le pied, huit points de Gauss chacun, et là où
l'exponentielle est négligeable (`e^{−40}` — « l'amplitude en deçà de laquelle on ne calcule
pas ») la forme close `atan`. La **facette** (l'entrée de la hessienne) est un `erf`, et la
**dérivée de la masse par rapport à `s`** aussi : `∂f/∂σ = −m e^{−r²/2σ²}/(2πσ³)`. `--check` :
la masse contre une quadrature de surface (7 points par triangle, subdivisée à `σ/8`) à
**4e-10** (la limite de la règle de surface), la somme des masses contre la masse exacte du carré
à **1e-15**, la dérivée contre des différences finies à **1e-8**.

Le jeu par défaut : 4 gaussiennes de largeurs `σ × {1, 0.7, 1.3, 1}`, masses `0.35, 0.25, 0.25,
0.15`, centres écartés du bord ; les germes **uniformes** sur le carré (`--diracs rho` : tirés
selon `ρ`). `n = 10⁵`, `--threads 8`, `--maxnv 512` (les cellules autour d'un pic reçoivent les
pointes de centaines d'aiguilles), Cholesky. Avec une densité, `--pas essai-limites` est ramené
aux essais : les limites parlent en aires.

## 9.1 Newton direct : ça casse à `σ = 0.05`

| σ | Newton depuis Voronoï |
|---|---|
| 0.2  | 6 it, 11 diag, 3.1 s |
| 0.1  | 27 it, 155 diag (127 reculs), 22 s |
| 0.05 | **STAGNATION** à la première itération (35 diagrammes) |

Un diagramme coûte **0.09–0.13 s** contre 0.02 en Lebesgue : la circulation (exp, erf, atan par
arête et par gaussienne) pèse cinq fois la géométrie. Pas optimisé (4 points de Gauss au lieu de
8 donnent la même précision et ne changent pas le temps : ce sont les `erf` et les `exp`).

À `σ = 0.05` les cellules loin des pics ont une masse numériquement nulle au départ de Voronoï :
le plancher de l'amortissement est nul, la hessienne a des lignes vides, et la direction est
inutilisable — Newton recule 34 fois et s'arrête. *Ce sont les queues exponentielles qui tuent,
pas l'hétérogénéité* : avec les germes tirés selon `ρ` (Voronoï presque juste), `σ = 0.02`
échoue aussi (100 itérations, 1 062 diagrammes, des pas de `1e-8`) ; le même avec un plancher de
1 % de la masse converge en **37 it, 151 diag**. Une cellule dont la masse est une exponentielle
de sa position n'a pas de linéarisation utile.

## 9.2 La continuation en largeur de convolution

`--conv 0.5 --conv-ratio R` : la densité convolée à `s = 0.5` (presque uniforme : Voronoï est
presque la solution), résolue, puis `s / R`, ... jusqu'à `σ/4`, puis `s = 0`. À chaque étape la
cible est `M(s)/n` avec `M(s)` la masse exacte sur le carré, et le départ est la solution
précédente. `σ = 0.05` :

| ratio | départ de l'étape | it | diag (reculs) | temps |
|---|---|---|---|---|
| 2 | poids précédents | 63 | 188 (118) | 44 s |
| √2 | poids précédents | 72 | **123 (39)** | 39 s |
| 2^¼ | poids précédents | 110 | 149 (16) | 52 s |
| 2 | extrapolés par `dw/ds`, sans garde | 83 | 364 (274) | 74 s |
| √2 | extrapolés, sans garde | 111 | 406 (283) | 92 s |
| 2^¼ | extrapolés, sans garde | 107 | 237 (107) | 67 s |
| √2 | extrapolés, **gardés** (en `s`) | 60 | **107 (17)** | 36 s |
| √2 | extrapolés, gardés (en `s²`) | 61 | 107 (18) | 36 s |

Et là où le direct est hors de portée, `ratio √2` :

| σ | poids précédents | extrapolés, gardés |
|---|---|---|
| 0.02 | 139 it, 400 diag (246), 93 s | 117 it, **332 diag** (168), 70 s |
| 0.01 | 192 it, 645 diag (436), 153 s | 175 it, **606 diag** (368), 149 s |

(`ratio 2` à `σ = 0.02` : 862 diagrammes ; sauter de `s = 0.1` à `0.02` : échec.)

**La dérivée `dw/ds`.** À la solution, `L dw/ds = dν/ds − ∂a/∂s` : une résolution linéaire avec la
hessienne du dernier diagramme, `∂a/∂s` calculé par la formule close à chaque diagramme. La
tangente est juste — sur un petit pas (`0.0625 → 0.055`) le résidu ℓ² du départ tombe de
`4.4e-4` à `3.4e-5`, 13 fois — mais **elle vide des cellules** : sur un pas `√2`, le départ
extrapolé a 15 cellules vides là où le départ sans en a zéro, Newton limpe 13 itérations à
`t ~ 1e-3` et coûte 79 diagrammes au lieu de 18. Le max du résidu est fait de quelques cellules
dont la masse tient à la pointe d'une aiguille, que la tangente pince au second ordre. La
**garde** : `w + θ dw`, `θ = 1, ½, ¼ ... 1/16`, retenu dès qu'aucune cellule ne passe sous la
moitié de la plus petite masse du départ sans extrapolation (le plancher de l'amortissement) et
que le résidu ℓ² est meilleur ; sinon le départ sans. Un diagramme par essai, celui du départ
retenu réutilisé par Newton (`resout( w, deja_mesure )`). En `s` ou en `s²` (la variable naturelle,
`σ'² = σ² + s²`), même chose. Le gain est de **13 % à 20 %** : `θ = 1` passe hors de la zone dure,
et y tombe à `1/16`.

## 9.3 Où ça coûte : la naissance des aiguilles, à `s` fixe

`figures/densite_pic.png` (`scripts/densite_figure.py`, depuis `--dump`) : les cellules autour
du pic `(0.72, 0.26)` aux étapes `s = 0.5, 0.0625, 0.031, 0`. À `s = 0.5` un Voronoï à peine
déformé ; à `0.031` le rebord des aiguilles ; à `0` **25 000 cellules** — le quart des germes du
carré, la masse du pic — convergent sur une fenêtre de `0.2 × 0.2`, en aiguilles venues de tout
le carré.

Le coût par étape, `σ = 0.02` et `0.01`, `ratio √2`, poids précédents :

| s | 0.5 … 0.088 | 0.0625 | 0.044 | 0.031 | 0.022 | 0.0156 | 0.011 | 0.0078 | … 0 |
|---|---|---|---|---|---|---|---|---|---|
| σ = 0.02 : diag | 4–11 | 31 | **92** | **79** | 58 | 47 | 26 | 12 | 7–9 |
| σ = 0.01 : diag | 4–12 | 37 | **130** | **115** | **105** | 84 | 44 | 31 | 9–23 |

La zone dure est **`s ∈ [0.016, 0.06]`, la même pour `σ = 0.05, 0.02, 0.01`** : elle ne dépend
pas de `σ` mais de `σ_eff = √(σ² + s²)` — c'est là que les queues des cellules lointaines
s'éteignent (`e^{−0.3²/2σ_eff²}`) et qu'elles doivent devenir des aiguilles pour atteindre les
pics. Avant, les cellules sont des blobs ; après, les aiguilles ne font que s'affiner et chaque
étape retombe à 10 diagrammes. Dans la zone, Newton fait 20 itérations dont les deux tiers en
reculs (`t = 1/8 … 1e-3`) : les changements de combinatoire — une aiguille a des voisins sur
toute sa longueur — rendent la linéarisation courte, et ni des pas plus petits en `s` (`2^¼`)
ni la tangente ne l'abrègent. Le plancher aide en direct (`σ = 0.05`, 10 % de la masse : 36 it,
175 diag ; 1 % : échec) mais change le problème.

**Ce que ça dit.** La continuation rend résoluble ce que Newton direct ne résout pas, pour un
coût à peu près constant en `σ` (330 à 600 diagrammes, 70 à 150 s, contre 10 diagrammes en
Lebesgue) concentré dans une zone de `s` qui ne bouge pas. La dérivée `dw/ds` est correcte et
utile hors de cette zone, nuisible dedans sans garde. Le levier suivant n'est pas dans le pas en
`s` mais dans Newton lui-même sur la naissance des aiguilles — le pendant, pour une masse, du
pas par les limites de § 7.

## 9.4 L'autre chemin, `(1 − t) + t·ρ`, et l'ordre 2

Si ce sont les queues qui tuent, un **mélange avec Lebesgue** garantit les zéros : `ρ_t = (1 − t)
+ t·ρ`, le plancher `1 − t` descendant de 0.5 vers 0 (`--melange 0.5 --melange-ratio 2
--melange-min F`), et la dérivée est gratuite et exacte, `∂a/∂t = masse_ρ − aire` (vérifiée à
1e-13 contre des différences finies). L'**ordre 2** (`--ordre 2`) : `L w'' = ν'' − φ''` avec
`φ(ε) = a(w + ε w', λ + ε)` mesuré par différences finies le long de la tangente (deux diagrammes
par étape, `--fd 0.25` du pas) — toutes les dérivées secondes de `a` dans la direction `(w', 1)`,
sans tenseur. Même garde `θ` que l'ordre 1. Diagrammes en tout, essais d'extrapolation et
différences finies compris :

| σ = 0.05 | ordre 0 | ordre 1 | ordre 2 |
|---|---|---|---|
| convolution, `√2`, en `s` | 123 | **107** | 138 (dont 22 de DF) |
| convolution, `√2`, en `s²` | — | 107 | 126 |
| mélange, planchers `0.5 … 2e-3`, puis 0 | **stagne à `t = 1`** | stagne | stagne |
| mélange, planchers `0.5 … 1e-6`, puis 0 | 366 | 353 | — |
| mélange, ratio 8, `0.5 … 1e-6` | — | 510 | — |

| σ = 0.02 | ordre 0 | ordre 1 | ordre 2 |
|---|---|---|---|
| convolution, `√2`, en `s` / `s²` | 400 | **332** / 341 | — / 436 |
| mélange, planchers `0.5 … 1e-6` | — | 2 093 (423 s) | — |

**Le mélange bute sur le même mur, et plus loin.** Arrêté à un plancher de `2e-3`, une
soixantaine de cellules vivent encore du plancher — loin des pics, larges de `1e-2` — et au
passage à `t = 1` leur masse tombe à zéro : ligne nulle dans la hessienne, direction sans
information, stagnation à la première itération. Il faut descendre le plancher sous `1e-5`
(l'aire qu'il faudrait dépasse le carré) pour que *toutes* les cellules soient devenues des
aiguilles avant `t = 1` ; alors la fin est gratuite — sous `1e-4` l'extrapolation passe à `θ = 1`
et chaque étape converge en **une itération** (départ à `1e-5`) : là le problème est linéaire en
`t` et la tangente exacte. Mais le chemin y arrive en payant chaque division du plancher par
deux **30–45 diagrammes** (`σ = 0.05`) ou **200–340** (`σ = 0.02`), de `0.5` à `2e-3` : le départ
de chaque étape vaut exactement `0.5` (les cellules de plancher perdent la moitié de leur
masse), et la naissance des aiguilles, que la convolution concentre sur trois ou quatre étapes,
est étalée sur neuf. Et à `t = 0.5` déjà, les pics sont à leur finesse : les cellules qui y
tombent doivent se comprimer 200 fois, 289 diagrammes pour la première étape à `σ = 0.02` là
où la convolution en met 8. C'est ce que le « moins spatial » coûte : le mélange garantit les
zéros mais laisse la géométrie entière à faire d'un coup.

**L'ordre 2 ne paie pas.** Le terme du second ordre vaut 5 à 20 % du premier, la tangente est
bonne — et le problème n'est pas la courbure du chemin mais les quelques cellules pincées, que
l'ordre 2 pince un peu plus : dans la zone dure la garde refuse tout (`θ = 0` après cinq essais)
là où l'ordre 1 gardait `θ = 1/4`, et les deux diagrammes de différences finies sont perdus.
Hors de la zone, `θ = 1` et deux à trois itérations par étape dans les deux cas.

## 9.5 La garde par cellule

`--garde cellule` : au lieu de reculer `θ` pour tout le monde, les cellules que la tangente a
pincées sous le plancher sont **relevées seules** — bissection sur leur poids, les autres poids
et l'arbre inchangés (`cellule_avec_poids`, comme le relèvement de § 8.5 mais en masse), jusqu'à
une masse entre `ν/2` et `3ν/2` — toutes sur le même diagramme ; puis on re-mesure (un diagramme
par passe), et on recommence tant qu'il en reste, six passes au plus ; `θ` ne recule que si ça
ne suffit pas. Diagrammes en tout, ratio `√2`, ordre 1 :

| σ | garde globale | garde par cellule |
|---|---|---|
| 0.05 | 107 (17 reculs) | 113 (10 reculs, 277 relèvements) |
| 0.02 | 332 (168) | 351 (128, 4 219 relèvements) |
| 0.01 | 606 (368) | 728 (354, 13 716 relèvements) |

À `σ = 0.05` elle fait ce qu'on lui demande : `θ = 1` conservé à toutes les étapes de la zone
dure (16, 10, 1, 83 cellules pincées, réparées en 3 à 6 passes), Newton y fait 12–16 diagrammes
au lieu de 15–20, les reculs tombent de 17 à 10 — et le total ne bouge pas, parce que chaque
passe de réparation coûte un diagramme complet. À `σ = 0.02` et `0.01` elle ne sert plus : à
`s = 0.031` la tangente pince **619 cellules**, les relèvements se disputent la place (calculés
sur le même diagramme, deux voisines relevées se vident l'une l'autre : 619 → 249 → 226 → 172 →
135 → 105, ça ne converge pas, et à `θ = 1/4` la liste *grossit* de passe en passe) — et surtout
le résidu ℓ² de l'extrapolation réparée est **pire que le départ nu** (`1.7e-3` contre `1.45e-3`).
Là, la tangente ne dit rien d'utile : `s ↦ w(s)` a un pli — des milliers de cellules changent de
régime entre deux étapes — et aucune garde ne répare une prédiction fausse. Un relèvement
séquentiel avec re-mesure locale (la leçon de la 1D, § 8.4) demanderait de rafraîchir l'arbre à
chaque cellule : 600 rafraîchissements, le prix de 100 diagrammes.

## 9.6 Pénaliser les petites masses : le résidu barrière

> **Le verdict est INVERSE hors de ce régime** : § 8.7.6 mesure `--residu log` sur Newton direct
> depuis Voronoï, densité uniforme, et il GAGNE 20 à 50 % de diagrammes, le gain croissant avec la
> difficulté. Ce qui suit vaut pour la continuation en densité, où des milliers de cellules sont en
> transition — pas pour un solve direct.
>
> **Et le POURQUOI est au § 21**, qui sépare les deux rôles que cette option confond. Deux résultats
> portent directement sur ce qui suit : (a) le mérite de l'amortissement est **inerte** le long d'une
> direction donnée — c'est le plancher d'aire qui décide, donc l'analyse « `1/x` en fait un minimax »
> ci-dessous décrit la DIRECTION, pas le juge ; (b) `x − 1/x` ne répare qu'un côté — sa courbure
> relative s'éteint en `x⁻³` sur les cellules trop grosses, donc il redevient `a − ν` exactement là
> où c'est dur, tandis que la famille `(xᵖ−1)/p` tient les deux queues (§ 21.3).

`--residu barriere | log` (dans `newton` aussi) : Newton sur `g(a_i/ν_i)` au lieu de `a_i − ν_i`,
`g(x) = x − 1/x` (ou `log x`). Même solution, autre direction et autre mérite : une cellule
minuscule reçoit `x → 2x` au lieu de sa masse entière d'un coup, et `|g| ~ 1/x` refuse les pas qui
la pincent. Un détail qui compte : `Σ a = Σ ν` est automatique, donc `g(x_i) = 0` fait `n`
équations pour `n − 1` inconnues — sans projection, la ligne rayée par la jauge porte toute
l'incohérence, le germe 0 explose et Newton stagne dès `s = 0.5` (mesuré : 220 diagrammes). On
résout `g(x_i) = c` avec `c` la moyenne pondérée qui fait sommer le second membre à zéro, et le
mérite est `|g − moyenne(g)|₂`. Continuation `√2`, `σ = 0.05`, diagrammes en tout :

| | `a − ν` | `x − 1/x` | `log x` |
|---|---|---|---|
| direct, `σ = 0.1` | 155 | 284, pas convergé en 60 it | 358, stagne |
| continuation, ordre 0 | **123** | 1 391 | 915 |
| continuation, ordre 1 | **107** | 505 | — |

Hors de la zone dure, la barrière est propre — pas un recul, 4 à 9 diagrammes par étape, comme
`a − ν`. Dans la zone, elle coûte dix fois plus (235, 489, 277 diagrammes aux étapes `s = 0.044,
0.031, 0.022`, contre 19, 18, 12) : le mérite part à `434` (des centaines de cellules à
`x ~ 1e-3`), le pas plein le fait *monter*, et l'itération accepte `t = 1/128 … 1/8` pour un pour
cent de baisse — quarante itérations à ce régime. Le mécanisme est exactement l'inverse de ce
qu'on espérait : avec des milliers de cellules en transition, un pas plein en pince toujours
quelques-unes un peu plus qu'il n'en répare, et `1/x` en fait un minimax où la pire cellule
décide de tout. Le résidu `a − ν`, lui, borne la contribution d'une cellule pincée à `ν` et
accepte ces pas : les cellules pincées se relèvent aux itérations suivantes. Dit autrement, **le
pincement transitoire fait partie du chemin** — une aiguille est une cellule pincée qui a
trouvé sa masse — et pénaliser les petites masses pénalise la transition elle-même. Avec la
tangente (ordre 1, garde sur le mérite barrière), même chose : `θ = 1` partout sauf dans la
zone, et 198 diagrammes à `s = 0.031`.

## 9.7 Le pas par les limites, en masse : là où ça tombe

`--pas essai-limites` avec une densité : l'essai `t = β` d'abord, et si des cellules y passent
sous `ε`, leurs **limites en masse** — pas de polynôme (la masse le long de `w + αd` n'en est
pas un), une bissection sur `α` entre 0 et l'essai, une cellule exacte par tour à chaud depuis
les voisins de la dernière bonne (`FournisseurAlpha`, l'arbre non rafraîchi), à `1e-2` près ;
le pas ramené sous la plus petite, on recommence, et le diagramme du pas retenu sert à
l'itération suivante (`limites_masse`, `Ecrasement.h`). Une limite nulle rend la main aux
essais depuis la moitié du dernier pas calculé, au lieu de stagner. Diagrammes en tout,
continuation `√2` depuis `s = 0.5`, `β₀ = 1` :

| | essais (KMT) | essais + tangente | **limites en masse** | limites + tangente |
|---|---|---|---|---|
| direct, `σ = 0.1` | 155 (127 reculs) | — | **44 (0 recul)** | — |
| `σ = 0.05` | 123 (39) | 107 (17) | **97 (1)** | 101 (0) |
| `σ = 0.02` | 400 (246) | 332 (168) | **204 (2)** | 202 (1) |
| `σ = 0.01` | 645 (436) | 606 (368) | **288 (3)** | 304 (0) |

Les reculs disparaissent : dans la zone dure, 33, 30, 26, 22 diagrammes par étape à
`σ = 0.02` contre 92, 79, 58, 47 — le nombre d'itérations de Newton est le même (19, 17, 15, 13),
c'est le prix de chacune qui tombe de 4–5 diagrammes à 1.7. Les limites coûtent 860 000
cellules et 1.7 s sur 61 (`σ = 0.02`), 1.9 million et 4.4 s sur 115 (`σ = 0.01`) : 3 %. La
tangente n'apporte plus rien (ses essais coûtent ce qu'elle épargne), et `ratio 2` redevient
viable (259 au lieu de 862) sans rattraper `√2`. Le direct à `σ = 0.05` reste hors de portée :
les cellules de masse nulle n'ont pas de limite à trouver.

Pourquoi ça marche là où la barrière (§ 9.6) et les gardes (§ 9.5) échouaient : le pas par les
limites ne cherche pas à *empêcher* les pincements, il mesure exactement jusqu'où la direction
peut aller avant qu'une cellule ne passe sous le plancher, et y va — au lieu de deviner par
moitiés. Les cellules qui s'amincissent sans passer sous `ε` ne le freinent pas.

---

# 10. NEWTON CONTRE LE PREMIER ORDRE : L-BFGS ET LE GRADIENT CONJUGUÉ (`--methode`)

La question : un L-BFGS *ad hoc*, ou un gradient conjugué non linéaire, font-ils mieux que Newton
amorti — sur l'uniforme, sur un nuage compliqué (les lignes, les plans), et sur une densité
hétérogène (`σ = 0.02`) — en 2D et en 3D ? `src/solver/PremierOrdre.h`, branché dans `newton`
et `densite` par `--methode newton | lbfgs | cg`, même diagramme, même critère d'arrêt, même
compteur ; `figures/methodes.png` (`scripts/methodes_plot.py` depuis `courbes/*.csv`, écrits par
`--courbe`).

**Ce qu'on minimise.** `F(w) = −Φ(w)`, convexe, de gradient `a(w) − ν` : exactement ce que le
diagramme livre, sans hessienne ni système linéaire. **La recherche linéaire n'a pas de valeur
de fonction** (il faudrait le second moment de chaque cellule, et un noyau de plus avec une
densité) : elle n'a que `φ'(α) = (a(w + αd) − ν)·d`, croissante en `α` par convexité. Ça
suffit : tout `α` où `φ' ≤ 0` fait décroître `F` (`F(α) ≤ F(0) + α φ'(α)`), donc chaque point
essayé du bon côté est un progrès garanti sans Armijo, et la sécante entre un `φ' < 0` et un
`φ' > 0` converge vers le minimum sur la droite. On accepte à `|φ'(α)| ≤ c₂ |φ'(0)|` (`c₂ = 0.5`,
`0.1` pour CG), et la courbure `s·y > 0` de L-BFGS est automatique. **Le plancher de masse est
gardé** : un pas qui met une cellule sous `ε = min(min ν, min a₀)/2` est refusé et le pas coupé
en deux, comme KMT (les cellules vides au départ sont exemptées tant qu'elles le restent).

**Le préconditionnement, et pourquoi il fait tout.** Trois `H₀` :

* `γ I` (`--precond 0`) et la **diagonale du laplacien** (`--precond 1`, Jacobi, gratuite
  depuis les facettes). Les deux **stagnent** sur tous les cas. Sans plancher, l'uniforme
  n = 20 000 finit avec 19 702 cellules vides sur 20 000 à l'itération 20 : une cellule vide a un
  gradient constant `−ν_i` et une courbure nulle, la paire `(s, y)` de L-BFGS le lui dit
  (`y_i = 0`), l'inverse de la hessienne devient énorme dans ces directions, et le pas suivant
  pousse ces poids sans mesure — elles avalent leurs voisines, qui se vident à leur tour. Avec le
  plancher, c'est la **rugosité** de la direction qui bloque : `d_i` suit le résidu de la cellule
  `i` seule, deux voisines partent en sens opposés, et c'est `d_i − d_j` qui déplace leur
  bissectrice (de `(d_i − d_j)/2|p_i − p_j|`). Mesuré : `|d|max = 2h²`, une cellule à la moitié
  de sa cible vidée par ses voisines dès `α = 0.25`, puis `α ≈ 5e-3` par itération et `φ'` qui ne
  bouge plus — là où la direction de Newton, *lisse*, passe entière. STAGNATION à 3.9 (uniforme)
  et 1.7e3 (lignes), 54 et 46 diagrammes refusés par le plancher.
* le **laplacien figé** `L₀⁻¹` (`--precond 2`, le défaut) : factorisé une fois — la descente de
  Cholesky, ou la hiérarchie AMG gardée (`resout_encore`, ajouté à `Amg` pour ça) — et une
  descente par itération. La direction a la régularité de Newton, les paires corrigent ce que la
  combinatoire a changé depuis. C'est la **méthode de la corde avec une mémoire**, et c'est ce
  qui réutilise la factorisation (le levier que § 7.4 désignait). Elle est refaite quand la corde
  ne mord plus : quand le plancher a borné le pas sous `0.25` (`--refacto-borne`), quand `|r|₂`
  n'a pas été divisé par 2 au dernier pas (`--refacto-taux 0.5`), toutes les K itérations
  (`--refacto K`), et quand la recherche linéaire échoue sur un laplacien périmé. **À chaque
  refonte la mémoire est vidée** : mesuré sur la densité à `s = 0.125`, les paires apprises sur
  l'ancien laplacien et bornées par le plancher tirent la direction fraîche vers la cellule qui
  bloque (`d_i = −9e-4` pour une cellule à `0.066 ν` dont le résidu demande `+0.93 ν`), et le pas
  tombe à 1e-15 — Newton, qui refait son laplacien, passe la même étape en 7 diagrammes.

**La bascule** (`--bascule 0.5`) : dès que `max|a − ν|/ν ≤ 0.5` — toute cellule a au moins la
moitié de sa cible, donc le plancher KMT est raisonnable — on rend la main à Newton depuis là,
diagramme compris. L'hybride : le premier ordre pour partir de loin, Newton pour finir.

## 10.1 Ce que ça donne

n = 100 000, 8 fils. 2D : Cholesky pour tout le monde ; 3D : AMG Ruge-Stüben+GS. Sur les lignes
le plancher du cas est 2.35e-6 (§ 3), la tolérance est mise à 4e-6 pour que personne ne paie
la sortie en STAGNATION. (Ce plancher était un défaut du nuage : le Voronoï dédupliqué du § 23.7
converge à 2.0e-7, donc **une reprise de ce tableau n'aurait plus besoin de la tolérance élargie** —
elle reste ici parce que les chiffres qui suivent ont été mesurés avec elle.) `L₀` = laplacien figé, refait selon les règles ci-dessus ; `refacto 1` =
refait à chaque itération, c'est Newton avec cette recherche linéaire à la place des essais.

| diagrammes (itérations) — temps | 2D uniforme | 2D lignes | 3D uniforme | 3D plans |
|---|---|---|---|---|
| **Newton, essais KMT** | **8** (6) — 2.6 s | 79 (23) — 16.5 s | **9** (6) — 8.7 s | **27** (13) — 19.5 s |
| Newton, `essai-limites` (2D) | 7 (6) — 2.5 s | **30** (18) — 8.4 s | — | — |
| L-BFGS `L₀`, taux 0.5 | 36 (13), 5 facto — 3.9 s | 86 (24), 14 facto — 12.3 s | 22 (12), 3 facto — 14.3 s | 105 (28), 15 facto — 57 s |
| L-BFGS `L₀`, jamais refait sauf blocage | 67 (23), 2 facto — 4.4 s | 161 (47), 15 facto — 21.5 s | 51 (16), 2 facto — 20 s | 228 (48), 13 facto — 121 s |
| L-BFGS `L₀` → Newton à 0.5 | 27 + 2 — 4.0 s | 81 + 2 — 11.7 s | 17 + 2 — 10.1 s | 97 + 4 — 59 s |
| CG (PR+) `L₀`, taux 0.5 | 56 (14) — 4.9 s | 120 (25) — 15.0 s | 36 (11) — 21.6 s | 161 (31) — 86 s |
| L-BFGS `L₀`, refait à chaque itération | 8 (6) — 2.8 s | 87 (20) — 12.4 s | 11 (6) — 8.2 s | 65 (17) — 41 s |
| L-BFGS Jacobi | STAGNATION à 3.9 | STAGNATION à 1.7e3 | — | — |
| L-BFGS `L₀` sans plancher | 62 (32) — 4.9 s | STAGNATION à 24, 733 diag | — | — |

(Les temps de la ligne Newton 2D sont ceux de ce lot, pris pendant qu'un autre tournait : 2.2 et
13.0 s au calme, § 3.) Et la densité, `σ = 0.02`, continuation `√2` depuis `s = 0.5`, 15 étapes :

| | diagrammes (itérations) | temps |
|---|---|---|
| Newton, essais KMT | 400 (139) | 124 s |
| Newton, `essai-limites` | **204** (130) | 108 s |
| L-BFGS `L₀`, taux 0.5 | 557 (208) | 157 s |
| L-BFGS `L₀`, jamais refait sauf blocage | 884 (309) | 227 s |
| L-BFGS `L₀` → Newton à 0.5 | 491 (149) | 156 s |
| CG (PR+) `L₀`, taux 0.5 | 723 (211) | 175 s |
| L-BFGS `L₀`, refait à chaque itération | 428 (129) | 130 s |
| L-BFGS direct (`s = 0`), tout `H₀` | ÉCHEC | |

Le direct à `σ = 0.02` : le laplacien de Voronoï y a des blocs de facettes de masse `1e-300`
(Cholesky échoue), Jacobi donne `φ'(0) = −4.6e22`, et le gradient nu fait `α = 6e-6` avec
33 000 cellules vides. Ce que Newton ne peut pas (§ 9.1), le premier ordre ne le peut pas non
plus : une cellule de masse nulle n'a ni équation ni courbure, et son gradient constant ne dit
pas de combien monter.

## 10.2 Ce qu'il faut en lire

**Newton gagne partout, en diagrammes comme en temps**, de 2× (2D uniforme, 3D uniforme) à 4×
(3D plans, densité) sur la meilleure variante du premier ordre — et `essai-limites` creuse encore
l'écart en 2D. Le seul cas où le premier ordre égale Newton est quand il *est* Newton
(`refacto 1`), avec une recherche linéaire un peu plus chère que les essais (2D uniforme 8 = 8 ;
3D plans 65 contre 27 : la sécante et le test de Wolfe coûtent un à deux diagrammes de plus par
itération, et la trajectoire diffère).

**La phase chère est la même pour tout le monde, et ce n'est pas une affaire de méthode.** Sur
les lignes, les plans, la zone dure de la densité, ce sont les cellules qui s'écrasent le long de
la direction qui bornent le pas (§ 7) ; le premier ordre n'y échappe pas — il a *besoin* du
plancher (sans lui, effondrement) — et sa direction, moins bonne, y est bornée plus tôt et plus
souvent (86 refus sur les lignes contre 55 reculs de Newton). Le levier là est celui de § 7 et
§ 9.7 : mesurer la limite au lieu de deviner, pas changer d'ordre.

**Là où le premier ordre coûte peu, il converge lentement.** Une fois les cellules en place, la
corde `L₀` avance à 1 diagramme par itération sans factorisation — mais au taux 0.5–0.6 par
itération (lignes, it 20 à 44 : `|r|₂` divisé par 1.85 par pas, 24 itérations pour 6 décades), là
où Newton en met 3 quadratiques. À 0.13 s le diagramme et 0.3 s la factorisation en 2D, 0.4 et
0.5 s en 3D (AMG, hiérarchie comprise), refaire le laplacien est *toujours* rentable : la
corde ne paie que si la factorisation coûte plus de dix diagrammes, ce qui n'est vrai nulle part
ici. La mémoire de L-BFGS n'y change rien de lisible : sans elle (`--memoire 0`, la corde nue
avec les mêmes refontes), 28 diagrammes contre 36 sur l'uniforme et 94 contre 86 sur les lignes,
mêmes factorisations.

**L'hybride ne rend que ce que Newton aurait fait.** `→ Newton à 0.5` économise les 3 à 4
dernières itérations de la corde (27 + 2 contre 36 sur l'uniforme), mais tout ce qui précède est
déjà borné par le plancher comme Newton l'aurait été. Il n'y a pas de « phase loin de la
solution » où le premier ordre serait plus robuste : la difficulté du transport semi-discret
n'est pas la non-linéarité du dual, c'est sa *non-régularité* — les cellules qui se vident — et
un pas de gradient s'y écrase comme un pas de Newton, en moins bien.

**2D contre 3D.** Même ordre, mêmes conclusions ; la différence est de prix relatif. En 3D le
diagramme vaut la résolution (0.4 et 0.5 s), donc une méthode qui multiplie les diagrammes par
2–4 pour économiser des factorisations perd sur les deux tableaux ; et la corde figée y est
plus mauvaise (plans : 228 contre 27), les plans changent la combinatoire plus que les lignes.
En 2D avec Cholesky la factorisation (0.3 s) vaut deux diagrammes, et c'est là que le premier
ordre est le moins loin (uniforme : 3.9 s contre 2.6).

Ce qui reste ouvert, si on y revient : une recherche linéaire à la KMT (couper en deux, accepter
le premier pas admissible) rendrait `refacto 1` égal à Newton ; et la 3D d'une densité
hétérogène (flux d'un champ radial à travers des faces polygonales, un niveau d'imbrication de
plus que § 9) n'est pas écrite.

---

# 11. LA MÉMOIRE EN 3D : PROPOSER D'ABORD LES VOISINS D'HIER (`memo`, `newton --memo`)

La question laissée ouverte par `2d_des_familles` (journal, `--memo`) : par germe, se souvenir des
diracs qui portaient une face de la cellule à la passe précédente, les proposer **en premier**,
puis parcourir l'arbre en **complément** (aucun dirac deux fois — `cut` n'est pas idempotente).
En 2D la borne supérieure valait −19 % et la boucle de Newton n'en rendait rien ; en 3D, jamais
mesuré. `FournisseurBsp3<…, MEMO>` (compilé à part : le chemin sans mémoire ne porte ni pointeur
ni compteur de plus), `PowerDiagram::cellule_memo`, `memorise( i, j )` / `oublie()`, et
`main_memo.cpp`.

## 11.1 La borne supérieure : deux passes aux mêmes poids

n = 10⁵, 8 fils, minimum de 10 répétitions, la machine pour le banc seul (`job -b`, § 11.6 —
les premiers chiffres, pris sur une machine partagée, étaient gonflés de 30 à 75 % sur le
*sans mémoire* ; ce sont ceux-ci qui comptent). Le *témoin* est le chemin MEMO à vide (le prix
du code) ; *les voisins seuls* ne parcourent pas l'arbre du tout — c'est le plancher, le prix des
seules coupes utiles. La mémoire ici est la forme A, les rangs des voisins (§ 11.4).

| par cellule | plans proposés | boîtes testées | **coupes effectives** | temps |
|---|---|---|---|---|
| uniforme, sans mémoire | 87.4 | 95.2 | 30.1 | 0.174 s |
| uniforme, témoin | 87.4 | 95.2 | 30.1 | 0.184 s |
| uniforme, **avec mémoire** | 67.9 | 83.9 | **15.1** | **0.112 s (−35 %)** |
| uniforme, les voisins seuls | 15.1 | 0 | 15.1 | 0.058 s (−67 %) |
| plans / Voronoï, sans | 88.5 | 95.6 | 30.5 | 0.181 s |
| plans / Voronoï, **avec** | 67.8 | 83.9 | **15.1** | **0.115 s (−36 %)** |
| plans / volumes égaux (Laguerre), sans | 270 | 258 | 32.0 | 0.324 s |
| plans / volumes égaux, **avec** | 253 | 251 | **15.5** | **0.266 s (−18 %)** |
| uniforme 10⁶, sans | 106 | 101 | 32.3 | 1.88 s |
| uniforme 10⁶, **avec** | 81.6 | 89.4 | **15.3** | **1.17 s (−38 %)** |

**Ce qui est différent de la 2D, et pourquoi.** Les boîtes testées baissent peu (95 → 84) et les
plans proposés de 20 % : l'argument de 2D tient — une boîte qui contient un vrai voisin passe
`peut_couper` quoi qu'il arrive, on ne peut pas élaguer ce qu'on doit regarder. Ce qui change,
c'est la colonne des **coupes effectives : 30 → 15**. Sans mémoire, la moitié des coupes qui
modifient la cellule sont *transitoires* — un dirac proche coupe, puis un vrai voisin le
supplante — et en 3D chaque coupe effective est une mise à jour combinatoire du polyèdre
(`Cellule3D::coupe`, 54 % du diagramme). Avec la mémoire, les quinze coupes effectives sont
exactement les quinze finales, et tout le reste n'est que des premières passes `s = d·v − off`
sans sommet dehors. En 2D la coupe transitoire d'un polygone en registres ne coûte rien de plus
qu'un test ; en 3D elle coûte le diagramme.

## 11.2 Les souvenirs périmés : ce que Newton fait subir à la mémoire

`--perime T` : les souvenirs pris à `T·W`, le diagramme mesuré à `W`, sur les plans / volumes
égaux (Laguerre, 0.324 s sans mémoire) ; A = les rangs, C = la frontière (§ 11.4) :

| souvenirs pris à | voisins retrouvés | coupes effectives | A | C |
|---|---|---|---|---|
| `W` (exacts) | 15.50 | 15.5 | −18 % | −24 % |
| `0.99 W` | 15.50 | 15.5 | −18 % | −24 % |
| `0.9 W` | 15.48 | 15.9 | −18 % | −24 % |
| `0.5 W` | 15.34 | 17.3 | −16 % | −17 % |
| `0` (Voronoï) | 15.09 | 20.0 | −12 % | −10 % |

Même les souvenirs de Voronoï rendent 10 à 12 % sur le diagramme final : les voisinages changent
peu, et un souvenir faux ne coûte qu'une première passe. Entre deux itérations de Newton, c'est
la ligne `0.9 W` ou mieux. C'est le point que la 2D ne pouvait pas montrer.

## 11.3 Dans la boucle de Newton

`newton --3d --memo` : les facettes du dernier diagramme **accepté** (que Newton a de toute façon,
pour le laplacien) deviennent la mémoire du suivant, essais compris — `memorise` coûte 10 ms par
diagramme. Mêmes itérations, mêmes diagrammes, mêmes résidus au chiffre près (l'ordre des coupes
n'a pas changé les arrondis, contrairement aux lignes en 2D) ; machine seule (`job -b`) :

| n = 10⁵, AMG RS+GS | diagrammes | total |
|---|---|---|
| uniforme, 9 diagrammes | 2.06 → **1.45 s (−30 %)** | 5.78 → 5.23 s (−10 %) |
| plans, 27 diagrammes | 8.17 → **6.40 s (−22 %)** | 17.1 → 15.6 s (−9 %) |

Le quart du diagramme 3D, pour soixante octets par germe et une passe sur les facettes. C'est
le contraire de la conclusion 2D, pour une raison qu'on peut nommer : ce que la mémoire épargne
n'est pas du parcours, ce sont les coupes transitoires, et elles n'ont de prix qu'en 3D. Reste le
plancher : les voisins seuls font −61 %, la moitié du diagramme est encore le parcours de l'arbre
pour *confirmer* qu'il n'y a personne d'autre — et ça, la mémoire ne peut pas le savoir.

## 11.4 Comment se souvenir : les rangs, les feuilles et leurs bits, ou la frontière

Trois formes de mémoire, mesurées sur les mêmes souvenirs exacts (uniforme 10⁵, 8 fils, minimum
de 10, machine seule) :

* **A. les rangs des voisins** (ce que § 11.1 mesurait) : une liste triée par germe, proposée
  d'abord, et au parcours un octet par rang, « déjà proposé », posé et retiré par la cellule.
  Le parcours reste entier — pile, tests d'éviction — et les diracs des boîtes extérieures ne
  sont re-proposés que si leur boîte passe encore le test, contre la cellule finale ;
* **B. les feuilles entrées, un bit par germe** : par germe la liste des feuilles où le parcours
  est entré la dernière fois — rang de leur premier germe, trié — et pour chacune un masque de
  64 bits, 1 = voisin final. Les bits à 1 sont proposés d'abord, sans aucun test de boîte ; au
  parcours une feuille mémorisée propose son complément, et un nœud qui contient une feuille
  mémorisée est descendu. Trois politiques : tester quand même toutes les boîtes ; tester les
  feuilles mais plus les nœuds internes qu'on sait entrés ; ne rien tester de ce qu'on sait entré ;
* **C. la frontière, rejouée sans pile** : la liste des feuilles entrées *et* des nœuds rejetés
  (dont on n'est pas descendu), en indices de nœuds. Après les bits à 1, chaque nœud de la
  frontière est testé et, s'il passe, parcouru normalement (une feuille mémorisée propose son
  complément) ; l'arbre n'est redescendu que sous un rejeté d'hier qui passe aujourd'hui. Aucun
  dépilage d'ancêtre, aucune recherche dans une liste. Exact : tout nœud de l'arbre est soit
  ancêtre d'une feuille entrée, soit entré, soit rejeté, soit sous un rejeté.

| par cellule | plans proposés | boîtes testées | coupes effectives | temps |
|---|---|---|---|---|
| sans mémoire | 87 | 95 | 30 | 0.174 s |
| A. rangs | 68 | 84 | 15 | 0.112 s (−35 %) |
| B. feuilles + bits, tout testé | 68 | 84 | 15 | 0.125 s (−28 %) |
| B. feuilles testées, pas les nœuds | 68 | 50 | 15 | 0.140 s (−20 %) |
| B. rien de connu n'est testé | 87 | 35 | 15 | 0.141 s (−19 %) |
| **C. la frontière, sans pile** | 68 | **48** | 15 | **0.109 s (−38 %)** |
| les voisins seuls | 15 | 0 | 15 | 0.058 s (−67 %) |

Sur les plans / Voronoï : A −36 %, C −39 % ; à 10⁶ germes : A −38 %, C −39 % ; sur les plans /
volumes égaux (Laguerre, 43 feuilles entrées et 85 nœuds rejetés par cellule, 7.5 % de frontières
au-delà des tampons de 64 / 256 et donc sans souvenir) : A −18 %, **C −24 %**. Avec des souvenirs
périmés (`0.9 W` → `W`) : A −18 %, C −24 % ; depuis Voronoï : A −12 %, C −10 % (20 coupes
effectives au lieu de 32 ; 209 boîtes testées au lieu de 251 pour C — la frontière d'hier ne
colle plus, on redescend).

Ce que les colonnes disent. **Ne pas tester les feuilles qu'on sait entrées coûte** : avec la
cellule finale dès le départ, 3 des 14.5 feuilles entrées hier sont *rejetées* aujourd'hui, et
chacune vaut six ou sept premières passes — les plans proposés remontent de 68 à 87. **Les tests
des nœuds internes connus valent peu** : ce sont les 34 qui *passent*, donc les moins chers (la
sortie anticipée dès le premier bloc de huit sommets), et les *chercher* dans une liste à chaque
dépilage (B, deuxième politique) coûte plus que les tests qu'on évite. Les tests chers sont les
rejets, et ce sont précisément ceux que l'exactitude oblige à refaire : une boîte rejetée hier peut
couper aujourd'hui. **C les épargne sans les chercher** : la frontière *est* la liste de ce qu'il
faut tester, 14.5 feuilles et 33.6 rejetés, et le reste de l'arbre n'existe plus pour cette
cellule. C'est la forme la plus courte en travail (48 tests contre 84) ; elle coûte 48 entrées
de 4 octets plus 14.5 masques par germe (~370 octets contre 60 pour A), et le gain sur A, de 1 à
6 points selon le cas (le plus sur le cas Laguerre, où la frontière compte 128 nœuds), est celui
des tests d'ancêtres — les moins chers.

(Ce que la 2D disait déjà : le masque limité à la feuille du germe ne peut rien gagner, la descente
l'atteint en premier de toute façon. Ici la mémoire couvre *toutes* les feuilles entrées, et ce
qu'elle épargne n'est pas de l'ordre mais des coupes transitoires.)

## 11.5 Dans `sdot`

`PowerDiagram( ..., memory = K )` : le stockage `PowerDiagram_Bsp` porte `memo_nbrs [ n, K ]` et
`memo_counts [ n ]` en rangs de l'arbre, écrits par le kernel de `measures` (les voisins de chaque
cellule, triés, ou rien au-delà de `K`) et lus par `FournisseurBsp<…, MEMO>` — la pré-passe, puis
le parcours avec un curseur de saut par feuille (pas d'octet par rang : rien par work-item, ce qui
va aussi au GPU). Les entrées et les sorties d'un appel étant disjointes, la mémoire sort dans deux
tenseurs neufs repris après l'appel — sauf sous une trace (`jit`, `grad`), où l'ancienne reste.
Effacée quand les positions changent. Mesuré (`./run bench "test_PowerDiagram::pd accelerated"
--nb-dims=3 --nb-points=200000 --memory=0|32`, machine partagée, minimum de 7) : Laguerre
0.450 → **0.337 s (−25 %)**, Voronoï 0.436 → **0.307 s (−30 %)** ; en 2D à 10⁶ la mémoire *coûte*
+13 % (0.227 → 0.256 s : la coupe d'un polygone en registres ne vaut pas la pré-passe). D'où le
défaut : `32` en 3D et au-delà, `0` en 2D.

## 11.6 Comment ces chiffres sont pris

`job -b -- <cmd>` (`scripts/job` à la racine du dépôt, `~/.local/bin/job` sur la machine) : le
banc attend que les travaux en cours finissent, tient les nouveaux à la porte, et tourne seul.
Les premiers chiffres de § 11 avaient été pris pendant que trois sessions et des compilations se
partageaient les seize cœurs : le *sans mémoire* de l'uniforme y valait 0.232 à 0.305 s contre
0.174 s seul, et les écarts relatifs bougeaient de ±15 %. Les *comptes* (plans, boîtes, coupes)
ne dépendent pas de la charge ; les temps, si. Ne jamais chronométrer hors d'un `job -b`.

---

# 12. LA DENSITÉ IMAGE : INTÉGRER SUR LE BORD (`image`)

Le § 9 prenait pour source une somme de gaussiennes — lisse, connue en fermé, jamais nulle. Une
application réelle donne une **image** : une densité constante par pixel sur une grille régulière,
**discontinue**, et qui peut valoir **exactement zéro** sur des régions entières. C'est le portage
sur CPU de ce que [`gpu_des_familles`](../gpu_des_familles/doc/08-densites.md) a mesuré sur la
carte — même méthode, même image de synthèse — pour deux raisons : voir ce que ça donne ici, et
**pouvoir instrumenter la simple précision**, ce qui est bien plus facile sur le CPU.

```
xmake run image --check -n 20000 --image 512                     la justesse, contre le découpage en pixels
xmake run image --chrono -n 1000000 --image 512 --threads 8      le bord contre le découpage, à machine égale
xmake run image --threads 8 -n 20000 --pas essai-limites         résoudre
xmake run image --threads 8 -n 20000 --pas essai-limites --chemin conv --adaptatif --chooseur   le meilleur réglage (§ 12.6)
xmake run image --check --acc float                              ce que la fp32 coûte
```

## 12.1 Ne pas découper

La façon naturelle de mesurer une cellule sous une image est de la **découper par les bords de
pixels** et de sommer `ρ × aire` : c'est ce que fait `sdot` en production, et ça coûte quatre
coupes et une copie de la cellule **par pixel de la boîte englobante**. Ici on ne coupe rien.
Green donne

        masse( P ) = ∫_P ρ dA = ∮_(∂P) G( x, y ) dy,      G( x, y ) = ∫_0^x ρ( t, y ) dt

et pour une image `G` est connue en fermé : sur la ligne `j`, `ρ` ne dépend plus de `y`, donc
`G_j( x ) = S[ j ][ i ] + ρ[ j ][ i ] ( x − i hx )` avec `S` la **somme préfixe de la ligne**,
calculée une fois. Sur un morceau d'arête qui reste dans un pixel, `G` est **affine en `x`** : son
intégrale en `y` est exacte au point milieu. Le coût passe de `pixels dans la cellule` à `pixels
sous le bord`, et le parcours est un Amanatides-Woo de quatre lignes (`src/solver/Image.h`). C'est
**exact**, pas approché — contrairement à la quadrature de `Densite.h`.

**Deux sorties pour une seule marche.** La même sous-arête donne aussi `∫ ρ ds`, le coefficient de
hessienne (§ 12.4). C'est pourquoi tout le fichier travaille **arête par arête**.

**La référence `sref`.** Sur un polygone fermé `Σ dy = 0` : retrancher une constante à `G` ne change
rien. Sans elle les termes sont d'ordre `taille de cellule × S`, pour une masse mille fois plus
petite — trois chiffres perdus par annulation. On retranche `S` au premier sommet.

**Le contraste est appliqué à la sortie, pas aux pixels.** On ne stocke que l'image nue
(normalisée) ; `ρ_t = ( 1 − t ) + t ρ` se lit `masse = ( 1 − t ) aire + t masse_ρ`, et la marche
rend les deux morceaux. Donc `d masse / d t = masse_ρ − aire` **ne coûte rien** — la tangente de la
continuation est gratuite, comme sur le chemin `mélange` du § 9.4.

## 12.2 Le témoin, et ce qu'il dit

Le témoin **découpe** (Sutherland-Hodgman) et n'a pas une ligne commune avec le noyau ; celui des
facettes **liste les traversées de lignes de grille et les trie**, sans un incrément. Deux
algorithmes opposés pour les mêmes nombres. `--check`, `n = 2·10⁴`, image 512², `t = 1` :

| | médian | p99.99 | max |
|---|---|---|---|
| masse / découpage en pixels (rapporté à la masse moyenne) | 1.0e−13 | 3.9e−11 | 3.9e−11 |
| `∫ ρ ds` / traversées triées | 0.0 | 7.5e−16 | 7.5e−16 |
| `d m_i / d w_i` / `Σ_j c_ij` (différence finie centrée) | 2.2e−10 | 6.7e−09 | 4.4e−08 |
| somme des masses | | | `1` à 1.3e−14 |

La troisième ligne est le seul contrôle de la hessienne **qui ne suppose ni le signe ni le
facteur** : on ne bouge que le poids de `i` (`cellule_avec_poids`), et la dérivée attendue est
exactement la somme des coefficients de ses facettes intérieures. Comparer deux implémentations de
la même formule n'aurait rien prouvé.

## 12.3 Ce que la méthode vaut, à machine égale

Les mêmes cellules du même moteur, mesurées deux fois dans le même binaire, `double`, 8 fils,
`job -b`. Les deux colonnes rendent **masse et facettes** : ce sont deux algorithmes complets, pas
deux fragments.

| `n` | image | `W / √n` | Lebesgue | **bord** | découpage | |
|---|---|---|---|---|---|---|
| 10⁶ | 512² | 0.5 | 0.172 s | **0.196 s** | 0.276 s | ×1.4 |
| 10⁶ | 2048² | 2.0 | 0.171 s | **0.206 s** | 0.497 s | ×2.4 |
| 10⁶ | 8192² | 8.2 | 0.169 s | **0.261 s** | 1.832 s | ×7.0 |
| 10⁵ | 8192² | 25.9 | 0.017 s | **0.044 s** | 1.039 s | ×23.7 |
| 2.5·10⁴ | 8192² | 51.8 | 0.005 s | **0.026 s** | 0.930 s | ×35.8 |

Le rapport suit la **surface contre le périmètre** : la seule grandeur qui compte est `W / √n`, le
nombre de pixels que traverse un côté de cellule. Et le **surcoût sur Lebesgue** est de +14 % à
512², +20 % à 2048², +54 % à 8192² — à comparer aux +7 % et +14 % de la carte aux mêmes
`W / √n` : le CPU paie la marche un peu plus cher, non pas parce qu'elle y est plus lente, mais
parce qu'elle s'y greffe sur un noyau de cellules qui, lui, est déjà bien plus lent que celui du
GPU.

## 12.4 Résoudre dessous : le pas par les limites débloque tout

Dès que `ρ` n'est pas uniforme, `c_ij = ∫_(facette ij) ρ ds / ( 2 |p_i − p_j| )` et non plus
`|facette| / ( 2 |p_i − p_j| )`. Newton n'a rien à savoir de la source : `Newton<PD,Lin,Rho>` est
maintenant paramétré par la densité, et `Densite` comme `Image` offrent la même
`mesure( cel, facette, dl )`.

`n = 2·10⁴`, image de synthèse 512² **sans zéros** (contraste 60:1), tolérance `1e−6` :

| | itérations | diagrammes (reculs) | temps | fin |
|---|---|---|---|---|
| contraste 3:1, essais (KMT) | 19 | 65 (45) | 1.35 s | converge |
| contraste 3:1, **limites en masse** | 14 | **23 (0)** | 0.88 s | converge |
| 60:1, essais (KMT) | 100 | 1041 (940) | 12.1 s | **résidu 18.2 → 15.5, PAS CONVERGÉ** |
| 60:1, **limites en masse** | 226 | **448 (2)** | 16.5 s | converge |
| 60:1, limites + continuation `K = 8` | 195 | 398 (22) | 13.7 s | converge |

**Le mécanisme du § 9.7 s'applique tel quel** — `limites_masse` : une bissection sur `α`, pas de
polynôme, la masse sous une image n'étant pas plus polynomiale que sous une gaussienne — et ici il
ne rapporte pas un pourcentage, il fait la différence entre converger et ne pas converger. Les 940
reculs tombent à 2.

Il faut lire ce chiffre pour ce qu'il est. Ce qui coûte 940 reculs, c'est une recherche linéaire
qui **repart de `t = 1` à chaque itération** alors que le pas admissible sous une image vaut
`1e−3` : c'est la faiblesse que le banc GPU a corrigée autrement, en repartant du pas précédent
doublé (2322 → 967 diagrammes), sans calculer aucune limite. `limites_masse` prend la même part de
gain, et en plus le **pas exact** au lieu du barreau dyadique en dessous. Les deux remèdes visent
la même chose ; celui-ci est simplement le plus précis, et il est déjà écrit ici.

Et il ne faut pas le confondre avec le **relèvement par cellule** (`--garde cellule`, § 9.5), qui
reste perdant : le banc GPU a tracé la population des cellules condamnées en fonction du pas et n'y
trouve **aucun palier** — ce n'est pas une cellule isolée qui borne le pas sous une image, c'est
une fraction de la population proportionnelle au pas. Les deux idées se ressemblent et n'ont rien
à voir.

La **continuation en contraste** ne gagne ici que ×1.2, et la **tangente** (`--ordre 1`) ne gagne
rien du tout : elle est pourtant gratuite à calculer, mais ses essais gardés coûtent quatre à cinq
diagrammes par étape et sont presque toujours refusés (`θ ≤ 1/16`). Même leçon qu'au § 9.7 : une
fois le pas réglé par les limites, il ne reste plus grand-chose à gagner sur le départ. **Sur la
continuation, ce chiffre de ×1.2 est trompeur** : il est celui du chemin par le plancher, qui est
le mauvais chemin — la continuation en **largeur de convolution** fait ×4 à ×5, et le § 12.6 la
mesure contre celui-ci.

## 12.5 Les zéros : il faut la bonne échelle de continuation, pas autre chose

L'image de synthèse du banc CPU a en plus un **carré exactement nul** (5.8 % des pixels ;
`--sans-trou` l'enlève). Ce qu'on en croyait d'abord — que `t = 1` serait un problème *différent* —
est faux, et la mesure le dit sans ambiguïté.

**Ce qui casse, et pourquoi.** Une cellule dont tout le voisinage est dans le trou ne vit que du
plancher `1 − t` : sa masse vaut `( 1 − t ) × aire`, donc pour tenir `ν = 1/n` il lui faut une aire
`1 / ( n ( 1 − t ) )`. Les `≈ 5.8 %` de germes tombés dans le trou réclament ensemble une aire
`0.058 / ( 1 − t )`, qui dépasse celle du trou dès que `1 − t < 1` et **celle du carré entier** dès
que `1 − t < 0.058`, soit **`t > 0.942`**. Passé ce seuil, ces cellules ne peuvent plus être
nourries par le plancher : il faut qu'elles **sortent du trou** et aillent chercher du vrai `ρ`.
C'est une réorganisation géométrique, pas un pas de Newton, et une échelle uniforme `k/8` **enjambe
le seuil d'un seul pas** — de 0.875 à 1. D'où la stagnation à résidu 2.43, avec 151 cellules sous
`ν/2` et une masse minimale de −2e−19.

**Ce qu'il fallait faire** : ne pas sauter à `1` quand ça vide des cellules, mais passer par un `t`
intermédiaire. La bonne variable est **`log( 1 − t )`** — `--etapes-geo K` : `1 − t = 1/2, 1/4, …
2⁻ᴷ`, puis `t = 1`. `n = 2·10⁴`, image 512² avec le trou :

| échelle | issue | diagrammes |
|---|---|---|
| directe (`t = 1`) | STAGNATION, résidu 18.4 | 33 |
| uniforme `k/8` | STAGNATION à la dernière marche, résidu 2.43 | 323 |
| **géométrique, `K = 14`** | **converge** (reste 7.3e−7) | 476 |

Et la suite des `|w|max` dit pourquoi ça marche : 0.173, 0.190, 0.198, 0.2025, 0.2047, 0.2058,
0.2063, 0.2066, 0.2067, **0.2069**. Les poids **convergent géométriquement** — ils ne divergent pas.
Une solution finie existe bien à `t = 1`, et le seul obstacle était d'y arriver par un chemin
admissible. Les dernières marches sont d'ailleurs les moins chères : 7, 5, 5, 5, 4 itérations, et
`t = 1` lui-même 4 itérations / 5 diagrammes / 0 recul.

Reste une chose vraie de ce qu'on croyait : à `t = 1` **exactement**, une cellule encore prisonnière
du trou aurait une ligne de hessienne entièrement nulle (`∫ ρ ds = 0` sur toutes ses facettes), et
Newton n'aurait rien à lui dire. La continuation ne contourne pas ce fait — elle fait en sorte
qu'il n'y ait plus personne dans ce cas quand on y arrive.

---

## 12.6 Deux chemins, et un pas qui se règle tout seul

Le § 12.5 ne dit rien sur le **choix du chemin**, et il n'y a aucune raison de s'en tenir au
plancher. `Image.h` et `main_image.cpp` portent les deux, paramétrés par **un seul nombre `λ` qui
décroît vers zéro** — ce qui les rend comparables à armes égales :

| `--chemin` | `λ` | la densité de l'étape | ce que `λ` bouche |
|---|---|---|---|
| `melange` (défaut) | `1 − t` | `λ + ( 1 − λ ) ρ` | les zéros, par un plancher **uniforme** |
| `conv` | `σ` | `ρ * G_σ`, `σ` en fraction du côté | les zéros, par la **matière voisine** |

La convolution est **approchée et pas chère** : trois passes de moyenne glissante séparable (une
B-spline d'ordre 3, à quelques pour cent d'une gaussienne), `O( W H )` par passe *quel que soit le
rayon* grâce à une somme courante, puis renormalisation pour que la masse totale reste `1`. Elle n'a
aucune raison d'être exacte — c'est un chemin, pas un résultat, et seule la dernière étape
(`σ = 0`, l'image nue) porte la réponse. On part de `σ = 1`, c'est-à-dire d'un flou à l'échelle du
domaine, qui rend une densité quasi constante : Lebesgue, comme `t = 0` de l'autre côté.

**Le pas adaptatif** (`--adaptatif`) ne pose alors plus aucune échelle : on résout la densité plate,
puis **on vise la cible à chaque fois**, et c'est le refus qui fabrique les étapes intermédiaires
(retour sur `√( λ_ok · λ )`, ou `λ_ok / 2` quand la cible est zéro). Le juge est le **diagramme de
départ** de l'étape — les cellules aux poids courants sous la densité proposée — et il ne coûte
rien quand il passe, puisque c'est exactement le diagramme dont Newton a besoin pour démarrer. Deux
critères de refus, sur ce même diagramme :

* **des cellules meurent** : `min_i a_i < 0.05 ν` (`--seuil`) ;
* **la densité a trop bougé** : `max_i |a_i − ν| / ν > 2` (`--seuil-residu`).

`n = 2·10⁴`, image 512², `--pas essai-limites`, en diagrammes (et secondes) :

| | mélange, avec trou | conv, avec trou | mélange, sans trou | conv, sans trou |
|---|---|---|---|---|
| directe (pas de continuation) | ✗ 33 | ✗ 33 | ✗ 1041 (essais) / 448 | — |
| échelle géométrique réglée à la main | 476 (16.9 s) | **100 (3.9 s)** | 451 (16.2 s) | **90 (3.6 s)** |
| adaptatif, critère « cellules mortes » seul | 469 (16.3 s) | 124 (4.7 s) | **544 (20.7 s)** | **531 (20.1 s)** |
| adaptatif, **les deux critères** | 475 (16.3 s) | 124 (4.9 s) | **387 (13.9 s)** | **95 (3.7 s)** |

**Trois choses à retenir.**

**1. La convolution gagne un facteur 4 à 5, sur les deux images.** Et pas parce qu'elle éviterait
une difficulté : elle arrive à la même solution (`|w|max` = 0.2069 des deux côtés). La raison est
que les deux chemins ne déplacent pas la même chose. Le plancher change la densité **partout à
chaque étape** — le diagramme entier doit se réorganiser, 20 à 40 itérations de Newton par étape.
La convolution ne change la densité **que là où l'image est rugueuse** : la structure grossière est
juste dès les premières étapes, et affiner `σ` ne perturbe les poids que localement — 4 à 8
itérations par étape, **zéro recul sur tout le parcours**. C'est le même verdict que le § 9.2 avait
rendu pour les gaussiennes, où la continuation en largeur de convolution était déjà le bon chemin ;
l'image le confirme, et le chemin par le plancher, hérité du banc GPU, était le mauvais défaut.

**2. Le critère « cellules vides » est juste, et il est aveugle à la moitié du problème.** Sur
l'image *sans trou*, il ne se déclenche **jamais** (0 refus) : aucune cellule ne meurt d'un coup, la
difficulté vient du contraste. L'adaptatif saute alors directement à la cible et Newton paie
l'addition — 531 à 544 diagrammes, pire que n'importe quelle échelle fixe. Le second critère, qui
regarde simplement **à quelle distance de la cible le diagramme de départ se trouve**, coûte le même
diagramme et rattrape tout : 387 et 95. Les deux ensemble couvrent les deux causes, et on ne connaît
pas de troisième.

**3. Avec les deux critères, l'adaptatif fait aussi bien qu'une échelle réglée à la main** — mieux
sur `mélange` sans trou (387 contre 451), à 5 % près sur `conv` (95 contre 90), 24 % de plus sur
`conv` avec trou (124 contre 100) — et il n'y a **plus de `K` à choisir**. Le prix est d'un
diagramme par refus (8 à 17 sur un parcours), ce qui se lit directement dans l'écart.

Le meilleur réglage du banc est donc `--chemin conv --adaptatif --pas essai-limites` : **95
diagrammes et 3.7 s** là où le Newton direct amorti en demandait 448 et n'aboutissait pas du tout
dès que l'image a des zéros.


### 12.6.1 Trois prédicteurs, et pourquoi celui qui porte le gradient ne peut pas marcher

Le critère « la densité a trop bougé » regarde `r_max = max_i |a_i − ν| / ν` au départ de l'étape.
C'est l'effet **visible** du changement de densité ; ce n'est pas ce qu'il **coûte**. Ce qu'il coûte,
c'est la correction de poids qu'il réclame,

        d = L⁻¹ ( ν − a ),      L la hessienne du diagramme de départ, c'est-à-dire `∂a / ∂w`

rapportée à l'échelle naturelle `h² = 1/n`. C'est bien le gradient qui entre : deux étapes de même
`r_max` n'ont pas du tout le même `d` — là où la densité est forte les facettes pèsent lourd et une
petite correction suffit, là où elle est faible `L` est presque singulière et il faut beaucoup
bouger. Et **c'est bon marché** : une résolution linéaire, pas un diagramme, et c'est exactement le
système que la première itération de Newton résoudra de toute façon. `--seuil-dw`, `--diagnostic`.

Mesuré, `n = 2·10⁴`, image 512², au départ de chaque étape :

| l'étape | `r_max` | `‖d‖∞ / h²` | cellules mortes | ce qu'elle a coûté |
|---|---|---|---|---|
| géométrique, sept marches `0.75 → 0.996` | **0.500 exactement**, sept fois | 1379 → 30 (décroît) | 0 | 5 à 39 it. |
| géométrique, cinq dernières marches | 0.46 → 0.017 | 11 → 0.34 | 0 | 4 à 7 it. |
| convolution, `σ = 0.5 → 0` | 0.15 → 1.0 | 420 → 0.19 | 0 à 11 | 4 à 8 it. |
| **saut direct** depuis la densité plate | **7.2** | **1.1e4** | 1118 (trou) / 0 (sans) | ✗ / 273 it. |
| **la marche fatale `0.875 → 1`** | **1.000 exactement** | **490** | **124** | ✗ stagnation |

**Le prédicteur de gradient voit le saut direct — et il est aveugle sur la marche fatale.** 490, au
milieu de la plage des bonnes étapes (479 à 1379) : il annonce une étape *plus facile que la
moyenne* au moment précis où elle est infaisable. La raison est structurelle, et elle vaut mieux
qu'une constatation : les cellules qui meurent ont **toute leur ligne de `L` nulle**
(`∫ ρ ds = 0` sur chacune de leurs facettes). Le système qu'on résout pour les interroger est
exactement celui qui a perdu son sens à cause d'elles. **Un prédicteur construit sur `L` ne peut pas
prévenir de la dégénérescence de `L`.** Il faut regarder les masses, pas le système.

**Et `r_max` est plafonné.** `( ν − a_i ) / ν ≤ 1` toujours, puisque `a_i ≥ 0` : le résidu ne peut
dépasser `1` que par des cellules **trop grosses**, jamais par des cellules qui se vident. C'est
pour ça qu'il vaut `7.2` sur le saut direct (des cellules explosent) et exactement `1.000` sur la
marche fatale. Pire, sur le chemin par le plancher il est **arithmétique** : diviser le plancher par
deux divise par deux la masse des cellules qui n'en vivent que, donc `r_max = 0.5` — sept fois de
suite, quelle que soit la difficulté réelle. Il mesure le **rapport de l'échelle**, pas le problème.

Le balayage du seuil le confirme, et répond au « 0.1 ? » qu'on pouvait avoir en tête :

| `--seuil-residu` | 0.1 | 0.3 | 0.5 | 1 | **2** | 5 | éteint |
|---|---|---|---|---|---|---|---|
| conv, avec trou (diagrammes) | 399 | 424 | 399 | 124 | **124** | 124 | 124 |
| mélange, sans trou (diagrammes) | 685 | 492 | 409 | 423 | **390** | 412 | 554 |

Sous `1`, le seuil mord sur le plafond au lieu de mordre sur la difficulté : il refuse en boucle
(162 refus au lieu de 17) et triple la facture. L'optimum est plat autour de `2`, et il **doit** être
au-dessus de `1` — non par réglage, mais parce qu'au-dessous il ne teste plus rien d'autre que
l'arithmétique du plafond.

**Ce à quoi `‖d‖∞ / h²` sert quand même.** C'est le seul des trois qui décroît **proprement** le
long d'une continuation qui marche : 1379 → 0.34 sur les douze marches géométriques, pendant que
`r_max` reste collé à 0.500. Il ne dit pas « refuse cette étape », il dit « la continuation
converge, tu peux accélérer ». C'est un **chooseur de pas**, pas un critère de refus — c'est ainsi
qu'il est branché, et le § 12.6.2 le mesure. En refus, le code reste là (`--seuil-dw`, éteint par
défaut) avec le verdict ci-dessus.


### 12.6.2 Le chooseur de pas : ce que l'accélérateur donne

Le § 12.6.1 finit sur une proposition : `amp = ‖L⁻¹( ν − a )‖∞ / h²` ne sait pas dire « refuse »,
mais il sait dire « accélère ». `--chooseur` la met en œuvre. On suppose localement
`amp ≈ C · Δlog λ`, on estime `C` sur l'étape qui vient de passer, et on choisit le `Δlog` suivant
pour viser `--amp-cible` — borné à `[ 1.1, 64 ]` en rapport, et on vise directement la cible dès que
le `λ` proposé passe sous un tiers de pixel. L'échelle fixe disparaît ; le refus reste le filet.

**C'est gratuit.** La première itération de chaque étape calcule déjà `d = L⁻¹( ν − a )` : Newton la
rapporte maintenant (`NewtonStats::amp_d0`, deux lignes) au lieu qu'on la recalcule dans un juge.
Zéro diagramme, zéro résolution de plus.

**Il a fallu lui apprendre ses refus.** Écrit naïvement, il oscille : la correction devient petite,
il saute à la cible, le critère des cellules mortes refuse, il replie, la correction est toujours
petite, il ressaute — douze refus sur un parcours qui en demandait cinq. C'est le § 12.6.1 en
action : la correction de poids **ne voit pas mourir les cellules**, c'est donc au refus de le lui
dire. Une ligne (`dlog ← log( λ_ok / λ_replié )`) et l'oscillation disparaît.

Diagrammes (et secondes), `n = 2·10⁴`, image 512², `--pas essai-limites`, refus actif partout :

| | conv, trou | conv, sans trou | mélange, trou | mélange, sans trou |
|---|---|---|---|---|
| échelle géométrique réglée à la main | 100 (3.95) | 90 (3.69) | 469 (16.5) | 458 (15.9) |
| adaptatif seul | 124 (4.73) | 95 (3.82) | 474 (16.3) | **394 (14.5)** |
| chooseur, cible 200 | 203 | 173 | 653 | 621 |
| chooseur, cible 400 | 156 | 117 | 502 | 489 |
| chooseur, cible 800 | **113 (4.36)** | 85 (3.41) | 475 | 448 |
| chooseur, cible 1600 | 114 | 85 | 436 | 439 |
| chooseur, cible 3200 | 123 | **83 (3.31)** | **425 (14.9)** | 405 (14.7) |

**Ça paie, modestement et sans réglage fin.** Au-dessus de 800, le chooseur bat l'échelle réglée à
la main sur les quatre cas (−15 %, −8 %, −9 %, −12 %) et l'adaptatif seul sur trois. Le meilleur
point du banc devient **83 diagrammes et 3.31 s** (conv, sans trou) contre 448 et 16.5 s pour le
Newton direct amorti de départ. Le plateau est large : 800 à 3200 se tiennent à 10 % près partout.

**Et il faut viser haut.** À 200 et 400, il est une fois et demie à deux fois plus lent que tout le
reste : viser une petite correction, c'est prendre des pas minuscules et payer un Newton complet
pour chacun. Le seul réglage dangereux est celui qu'on croirait prudent — la même leçon qu'au
§ 12.6.1 pour le seuil de résidu.

**Ce que la trace apprend sur ce qui travaille vraiment.** À cible 800, les sept premières étapes
sont réellement pilotées : rapports 2.00, 3.75, 1.30, 1.81, 2.43, 4.33, 13.32 — le contrôleur
ralentit là où la densité bouge (`σ` autour de 0.1) et accélère ensuite. Mais la queue sature au
plafond de 64, et à cible 3200 c'est presque tout le parcours qui sature. Autrement dit le gain se
partage : le **contrôle** dans la phase difficile, le **plafond plus le refus** dans la phase facile.
Ce n'est pas « le prédicteur choisit le pas » de bout en bout, et il ne faut pas le lire comme ça.

**Une limite du noyau de convolution, énoncée pour qu'on ne s'y trompe pas.** Les trois moyennes
glissantes ont un **support compact** : une région nulle plus large que `3 ( 2r + 1 )` pixels reste
nulle. Les zéros du trou réapparaissent donc dès `σ ≲ 0.03`, et le parcours les traverse pourtant
sans un recul — parce qu'à ce moment-là les cellules nées dans le trou en sont **déjà sorties** et
n'y reviennent pas. Une vraie gaussienne ne changerait rien d'utile : à dix écarts-types elle vaut
`1e−22`, ce qui demanderait une cellule d'aire `1e22`. Rencontrer les zéros est une propriété du
problème, pas du noyau ; ce qui compte est de les rencontrer **quand le diagramme est déjà rangé**.

---

## 12.7 La simple précision, instrumentée

`--acc A` choisit le flottant de **la mesure** (la marche, les sommes préfixes, les accumulations),
indépendamment de `--kernel`, qui est celui de **la géométrie**. Même image, mêmes cellules, `float`
contre `double` :

| | médian | p99 | p99.99 | max |
|---|---|---|---|---|
| masse (écart absolu / masse moyenne) | 1.1e−06 | 2.9e−05 | 8.3e−05 | 9.8e−05 |
| `∫ ρ ds` (écart relatif) | 2.5e−06 | 1.5e−04 | 1.1e−02 | **1.5e−01** |

Le médian est celui qu'on attend d'un `float` (1e−6 ≈ 2⁻²³ à quelques annulations près). Ce sont
les **queues** qui décident : `1e−4` sur une masse rapportée à la moyenne, c'est cent fois la
tolérance de Newton (`1e−6`) — et **15 % d'erreur sur une facette** interdit la hessienne telle
quelle. Les deux viennent du même endroit : `S[ j ][ i ] − sref`, une différence de sommes préfixes
dont les deux termes sont d'ordre `1` quand leur différence est d'ordre `1/W`. `sref` sauve trois
chiffres, pas huit.

Ce que ça dit pour la suite : en `float`, la marche doit accumuler **relativement au pixel de
départ de l'arête**, pas à celui du premier sommet de la cellule — ou bien la table des sommes
préfixes doit être stockée **par blocs**, avec une origine par bloc. C'est mesurable maintenant, et
c'était le but du portage.

**→ Réglé au § 19.7, et pas comme prévu.** Une référence par arête n'aurait rien changé : l'erreur
est faite au **stockage**, pas à la soustraction. La marche reste en `double` et seule l'image est
en `float` — le gros tableau, celui dont la bande passante décide. Médiane 1.1e−06 → **1.4e−09**,
pire facette 1.5e−01 → **5.9e−08**.

## 12.8 Ce qui reste

* ~~La référence prise **par arête** plutôt que par cellule~~ — **fait autrement** : § 19.7. Le
  diagnostic était juste, le remède non ; c'est le *stockage* de la table qui devait passer en
  double, pas la référence qui devait bouger.
* **Le plafond de rapport du chooseur** (64) borne le parcours facile plus souvent que le
  contrôleur lui-même (§ 12.6.2) : dans la phase où `amp` s'écroule, c'est lui et le refus qui
  décident, pas la prédiction. Le relever, ou proposer directement la cible dès que `amp` passe
  sous un seuil, économiserait les dernières étapes — quelques pour cent, pas plus.
* **Un critère de refus qui verrait la dégénérescence avant qu'elle arrive.** Les deux d'aujourd'hui
  sont réactifs : ils constatent des cellules mortes ou une mesure trop loin. Le prédicteur bâti sur
  `L` ne peut pas le faire (§ 12.6.1). Une piste non mesurée : suivre, par cellule, la masse
  `∫ρ` de son **voisinage** plutôt que d'elle-même — une cellule dont tout le voisinage s'éteint est
  condamnée une étape avant de l'être.
* **La convolution en 3D** viendra avec le reste : trois passes séparables au lieu de deux, même
  coût par voxel.
* La **3D** : le même raisonnement donne `∮_(∂P) G dS`, `G` la primitive en `x` du voxel, et une
  marche sur les faces du polyèdre. `Image.h` est écrit 2D.
* Le nuage `--diracs rho` (germes tirés selon l'image) n'a pas encore servi : c'est le départ
  naturel pour un vrai cas, et il devrait supprimer une bonne partie des cellules mortes.

---

# 13. RELEVER DES CELLULES PAR DES MOINDRES CARRÉS PONDÉRÉS (`image --relevement`)

L'idée, posée en une ligne : plutôt que `L d = ν − a`, résoudre

        min_d  ½ Σ_i C_i ( a_i + ( L d )_i − ν_i )²

avec des `C_i` **grands sur les cellules qui se vident et autour d'elles** (une cloche de largeur à
trouver), pour que la direction s'occupe d'abord d'elles et qu'on puisse prendre un plus grand
coefficient de relaxation. Pas comme substitut à Newton — ce n'est pas le même système — mais pour
fabriquer, depuis un état **sain**, un **point de départ pas trop mauvais** quand on constate que la
direction de Newton mène à un état dégénéré.

Les notations : `L` est la matrice du système de Newton, `L_ij = ∂a_i / ∂w_j` (le laplacien de
Laguerre, § 1) ; `a` les masses courantes, `ν` la cible, `d` le pas cherché. `a_i + ( L d )_i` est
donc la masse de la cellule `i` **prédite** après le pas.

```
xmake run image --relevement --sans-trou --rel-depuis 0.125 --rel-vers 0 --rel-solve
```

## 13.1 Les poids sont invisibles, et c'est une identité

`L` est **carrée** et inversible sur les moyennes nulles (son noyau est exactement les constantes).
Il existe donc un `d` qui met **chaque terme** de la somme à zéro exactement — c'est `L d = ν − a`,
la direction de Newton. Le minimum du problème pondéré vaut alors **zéro**, et il est atteint là,
**quels que soient les `C_i` positifs**. C'est la même chose que d'ajuster une droite à deux points :
avec autant de paramètres que d'équations on passe par tous les points de toute façon, et pondérer
ne change rien.

Ce n'est pas non plus un artefact de la linéarisation : Gauss-Newton sur `Σ C_i ( a_i(w) − ν_i )²`
refait ce calcul à chaque itération, donc redonne le pas de Newton à chaque itération.

Vérifié plutôt qu'affirmé. Les équations normales `L C L d = L C ( ν − a )` sont résolues par un CG
sans matrice (deux produits `L x` par itération — le remplissage en voisins-de-voisins n'existe
jamais), avec une cloche `κ = 1000` de largeur 2 :

| `n` | itérations de CG | écart relatif à la direction de Newton (à la jauge près) |
|---|---|---|
| 2 000 | 15 817 | **7.8e−09** |
| 5 000 | 30 215 | **7.6e−09** |

soit la tolérance du CG. Le conditionnement est carré (`κ(L)²`), d'où les dizaines de milliers
d'itérations : **on paie très cher pour retrouver exactement ce que Cholesky donne d'un coup.**

Une remarque qui explique un chiffre du § 12.6.1 : une cellule de masse nulle a toutes ses `c_ij`
nulles, `Laplacien::assemble` **neutralise sa ligne** (`dia = 1`), et alors `d_i = ν_i` — une
correction de poids de l'ordre de `h²` là où il en faudrait une de l'ordre du domaine. Aucun choix
de `C`, aucun amortissement et aucun patch ne changent ce `d_i` : `C_i` se simplifie des deux côtés.
**Une cellule déjà morte est hors de portée de toute cette famille d'idées.**

## 13.2 Deux façons de rendre les poids visibles

L'identité tient tant que le système est carré, résolu exactement, sur tout le diagramme. Il faut
donc casser l'une de ces trois choses. Deux façons sont implémentées et mesurées :

* **amortir** (`--rel-mus`) : `( L C L + μ I ) d = L C ( ν − a )`, Levenberg-Marquardt. À `μ = 0`
  c'est Newton quels que soient les poids ; quand `μ` domine, `d → ( 1/μ ) L C ( ν − a )` et les
  `C_i` poussent franchement les cellules visées. Global, sans patch. `μ` est donné en fraction de
  la diagonale moyenne, faute de quoi il n'a pas d'échelle ;
* **restreindre** (`--rel-largeurs`) : `d` nul hors d'un patch `P` (une boule de rayon donné autour
  des cellules visées, par parcours en largeur). Le système devient sur-déterminé, le résidu ne peut
  plus être annulé, et les `C_i` décident **ce qu'on sacrifie**. C'est bon marché : `A x` n'est que
  `L` appliqué au `x` complété de zéros, `Aᵀ y` la restriction de `L y` au patch — rien à assembler.

## 13.3 Le protocole, et le cas d'essai

Exactement celui qu'on veut : partir d'un **état sain**, constater que Newton n'y va plus, essayer
autre chose. Image 512² sans zéros, `n = 2·10⁴` ; continuation par le plancher **convergée** en
`t = 0.875` ; on pousse la densité à `t = 1`. **16 cellules passent sous `0.1 ν`** — un pincement
localisé, pas une réorganisation générale — et le pas admissible de Newton tombe à `1.6e−02` au lieu
de 1.

`α*` est le plus grand pas essayé qui garde toutes les masses au-dessus de `ε` : le coefficient de
relaxation que l'amortissement pourrait prendre. Au départ, masse min `0.556 ν`, `‖r‖₂ = 1.123e−3`,
`max|a−ν|/ν = 0.444`.

| direction | inconnues | `‖d‖∞ / h²` | `α*` | masse min | **les 16 visées** | `max|a−ν|/ν` |
|---|---|---|---|---|---|---|
| **newton** | 20 000 | **716** | 1.6e−02 | 0.559 ν | 0.563 ν | 0.441 |
| amorti `μ=0.01 κ=1` | 20 000 | 0.133 | 1.0 | 0.544 ν | 0.553 ν | 0.456 |
| amorti `μ=0.01 κ=100` | 20 000 | 0.771 | 5.0e−01 | 0.339 ν | 0.560 ν | 0.661 |
| amorti `μ=0.01 κ=10⁴` | 20 000 | 1.02 | 1.2e−01 | 0.429 ν | 0.557 ν | 0.659 |
| amorti `μ=1 κ=1` | 20 000 | 9.4e−03 | 1.0 | 0.554 ν | 0.556 ν | 0.446 |
| amorti `μ=1 κ=10⁴` | 20 000 | 7.6e−02 | 1.0 | 0.448 ν | 0.554 ν | 0.552 |
| amorti `μ=100 κ=10⁴` | 20 000 | 4.1e−03 | 1.0 | 0.553 ν | 0.556 ν | 0.447 |
| local `κ=1 l=1` | 67 | 0.552 | 1.0 | 0.527 ν | 0.556 ν | 0.473 |
| local `κ=100 l=1` | 67 | 0.738 | 1.0 | 0.483 ν | **0.614 ν** | 0.517 |
| local `κ=10⁴ l=1` | 67 | 0.740 | 1.0 | 0.482 ν | 0.615 ν | 0.518 |

**Les poids agissent maintenant** — `κ = 1`, `100`, `10⁴` donnent trois directions différentes à
`μ = 0.01` (`‖d‖∞` = 0.133, 0.771, 1.02). La mécanique fonctionne. Mais regarder la colonne `‖d‖∞`
suffit à voir le piège : **l'amortissement écrase la direction de trois ordres de grandeur** (716 →
1) bien avant que les poids ne pèsent. Les cellules visées passent de 0.556 ν à 0.557 — autant dire
qu'on n'a pas bougé. Il n'y a pas de fenêtre où `μ` soit assez grand pour que `C` compte et assez
petit pour que le pas serve à quelque chose.

Le patch, lui, relève pour de bon : **0.556 → 0.614 ν** sur les cellules visées, au pas plein, avec
67 inconnues et 63 itérations de CG — le coût d'un dixième de diagramme.

## 13.4 La mesure qui tranche : et après ?

Relever n'est utile que si le Newton d'après en profite. Newton mené à **convergence** depuis chaque
point relevé :

| départ de Newton | itérations | **diagrammes** |
|---|---|---|
| le point de départ (référence) | 53 | **105** |
| amorti `μ=0.01 κ=1` — *pas de pondération du tout* | 51 | **100** |
| amorti `μ=0.01 κ=10⁴` | 50 | **98** |
| amorti `μ=1 κ=100` | 50 | 98 |
| amorti `μ=100`, tous `κ` | 53 | 105 |
| local `κ=100 l=1` | 51 | 102 |
| local `κ=100 l=4` | 51 | **101** |

**Sept diagrammes sur cent cinq, au mieux.** Et le contrôle est sans appel : `κ = 1`, c'est-à-dire
**aucune pondération**, en gagne déjà cinq. Le peu qu'on gagne vient de l'amortissement — un petit
pas prudent avant Newton — et non du fait d'avoir forcé les cellules pincées. La pondération
elle-même vaut deux diagrammes sur cent cinq.

Pour le patch, deux observations disent pourquoi. **La hauteur de la cloche ne compte pas** :
`κ = 100` et `κ = 10 000` donnent des directions identiques à trois chiffres. Ce qui agit est le
**patch**, pas la pondération dedans — une fois `d` restreint, forcer plus ne force rien de plus. Et
**la cellule qui borne le pas de Newton n'est pas celle qu'on a relevée** : après le relèvement, la
masse minimale globale au pas de Newton est *plus basse* qu'avant (0.491 ν contre 0.559), une autre
cellule ayant pris la place. C'est le même constat que le banc GPU a tiré de sa courbe `N(t)` sans
palier et que le § 9.5 avait tiré de `--garde cellule` : **ce n'est pas une cellule isolée qui borne
le pas, c'est une population**, et la réparer déplace le problème au lieu de le résoudre.

Sur un pincement moins localisé (793 cellules sous `0.1 ν`, continuation par convolution
`σ = 0.0156 → 0`), même histoire en plus net : la direction locale quadruple son propre pas
admissible (3.9e−3 → 1.6e−2) et le Newton d'après revient à 3.9e−3, pour un résidu meilleur de
0.2 %.

## 13.5 Ce que l'étude laisse ouvert

Elle ferme la porte qu'on voulait pousser — et surtout elle dit **pourquoi** elle est fermée, ce qui
vaut mieux qu'un chiffre décevant. Deux autres restent entrebâillées, et elles ne sont **pas**
mesurées :

* **Pondérer le mérite, pas la direction.** `Σ C_i r_i²` ne change pas `d`, mais il change quels pas
  l'amortissement **accepte**. C'est le seul endroit où des `C_i` peuvent agir sans que l'algèbre les
  simplifie ni que l'amortissement les écrase, et c'est un objet entièrement différent de celui-ci.
* **Un sous-problème local résolu NON LINÉAIREMENT.** Ici le patch ne restreint qu'**un pas
  linéaire**. Résoudre à convergence le vrai problème de transport sur le patch, les poids extérieurs
  gelés, est autre chose — `--garde cellule` (§ 9.5) en est le cas dégénéré à une cellule, et il
  perd ; rien ne dit qu'un patch de rayon 2 ou 3 se comporte pareil. **C'est fait, et c'est le
  § 14.**

---

# 14. LE SOUS-PROBLÈME LOCAL : SAUTER PUIS RÉPARER, OU SUIVRE UN CHEMIN (`image --local-nl`)

Le § 13 a fermé la voie des directions pondérées et en laissait une ouverte : ne pas chercher une
direction, mais **résoudre** sur un morceau du diagramme. Le schéma :

* on prend le pas de Newton `w₀ + F·d` et on **repère les cellules qui pincent là** — pas celles qui
  sont petites au départ : au point convergé tout est à la cible, c'est le **pas** qui les tue ;
* on va chercher **`N` anneaux plus loin** autour d'elles (parcours en largeur ; les composantes qui
  se touchent fusionnent d'elles-mêmes) ; seuls les poids du **bord** sont imposés ;
* l'intérieur est cherché en minimisant

        Φ = Σ_( boule et anneau )  ( A_i / ν_i  −  ν_i / A_i )²

**L'anneau est dans l'objectif, et c'est ce qui le protège.** Descendre une cellule d'anneau de 1 à
0.9 ne coûte presque rien (`g = −0.21`) ; la descendre à 0.1 coûte `g = −9.9`. La barrière arbitre
d'elle-même, sans plancher à poser, et vaut `+∞` si une cellule meurt. Ce n'est pas le système de
Newton : `A_i = ν_i` sur toute la boule serait en général infaisable (le budget de masse local est
fermé), alors que le minimum de `Φ` existe toujours. On ne cherche pas la solution, on cherche un
**point de départ sans écrasement**.

Gauss-Newton sur `Φ`, sans matrice, amorti sur `Φ` elle-même. (Un L-BFGS conviendrait aussi ; ici
Gauss-Newton est meilleur parce que la jacobienne exacte est `L`, qui sort **gratuitement** avec les
cellules.) Le coût est en **cellules calculées** — chaque itération ne recalcule que la boule et son
anneau.

## 14.1 La seule question qui compte : saute-t-on, ou suit-on un chemin ?

Deux façons d'amener les poids du bord de `w₀` à `w₀ + F·d` :

* **le saut** (`--local-saut`) : on pose tout le monde à `w₀ + F·d`, puis on répare ;
* **la continuation** (le défaut) : on part de `w₀`, **qui est sain et où il n'y a rien à résoudre**,
  et on fait glisser le bord vers `w₀ + F·d` par sous-pas, l'intérieur étant re-résolu à chaque
  sous-pas. On ne traverse alors **jamais** un état où une cellule est déjà morte : la barrière les
  maintient en vie *le long du chemin*, au lieu d'avoir à ressusciter ce qui est mort — ce qu'aucune
  méthode passant par `L` ne sait faire (§ 13.1).

**La différence est décisive.** Le saut plafonne à `F = 0.033` : au-delà une cellule atteint
exactement zéro, `Φ = 10¹⁸`, et le solveur cale en une itération. La continuation atteint `s = 1`
**à chaque fois, jusqu'à `F = 0.2` au moins** — et les cibles y ont pourtant des cellules mortes
(colonne « min boule à la cible » = 0.000 partout). Il n'y a pas de verrou ; il y avait un mauvais
chemin.

**Deux détails d'implémentation, chacun payé par une mesure fausse.** (i) Le critère de succès d'un
sous-pas est « l'état est sain », pas « la boule est à la cible » : le budget de masse local étant
fermé, l'exiger fait rejeter des sous-pas où la réparation marchait. (ii) Au démarrage à chaud d'un
sous-pas, **l'intérieur doit avancer du même incrément que le bord** le long de `d` avant d'être
corrigé ; le laisser figé pendant que le bord avance recrée une marche différentielle à la frontière
de la boule, qui tue une cellule avant que le solveur ne réagisse — le sous-pas admissible tombe
alors à 0.002, en falaise.

## 14.2 Ce que ça donne

Image 512² sans zéros, `n = 2·10⁴`, continuation convergée en `t = 0.875`, densité poussée à `t = 1`.
Sans réparation, le pas de Newton est refusé au-delà de `α* = 1.6e−02`, et **107 diagrammes** suffisent
ensuite à converger depuis `w₀`.

`s` vaut **1.00 partout** : la continuation atteint la cible dans tous les cas, y compris quand
celle-ci a des cellules mortes (colonne « min boule à la cible » = 0.000). `N` est le nombre de
couches, et la colonne **amas** compte les composantes connexes de la boule.

| `F` | `N` | pincées | boule | **amas** | **+ gros** | itér. | cellules | fuites | min boule (cible → fin) | min **globale** | diag. après |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | **103** |
| 0.03 | 1 | 5 | 23 | **2** | **18** | 42 | 12 000 | 21 | 0.137 → 0.875 | 0.532 | — |
| 0.03 | **2** | 5 | 63 | **1** | **63** | 128 | **39 000** | 131 | 0.137 → 0.616 | 0.532 | **98** |
| 0.03 | 3 | 5 | 114 | 1 | 114 | 125 | 65 000 | 128 | 0.137 → 0.620 | 0.532 | — |
| 0.05 | 1 | 38 | 130 | **22** | **21** | 519 | 580 000 | 9 179 | 0.000 → 0.807 | 0.117 | — |
| 0.05 | **2** | 38 | 343 | **20** | **62** | 106 | **224 000** | 415 | 0.000 → 0.617 | **0.523** | **96** |
| 0.05 | 3 | 38 | 684 | 17 | 141 | 277 | 669 000 | 1 729 | 0.000 → 0.569 | 0.510 | 92 |
| 0.10 | 1 | 234 | 584 | 143 | 66 | 1 515 | 5 266 000 | 45 289 | 0.000 → 0.379 | **0.000** ✗ | ✗ |
| 0.10 | **2** | 234 | 1 241 | 126 | 248 | 341 | **1 539 000** | 3 884 | 0.000 → 0.603 | 0.482 | **87** |
| 0.10 | 3 | 234 | 2 046 | 117 | 488 | 397 | 2 485 000 | 3 309 | 0.000 → 0.584 | 0.525 | 91 |
| 0.20 | 3 | 809 | 3 448 | — | — | 1 342 | 18 716 000 | 51 884 | 0.000 → 0.610 | 0.525 | **72** |

**Le pas admissible passe de `0.016` à `0.2` — un facteur treize** — et l'état obtenu est
franchement sain. Le Newton qui repart de là converge en **72 diagrammes au lieu de 103**.

**L'épaisseur de couronne ne sert pas à donner de la marge de masse, elle sert à contenir la
fuite.** Avec une seule couche, la réparation pousse le dommage hors de la zone surveillée — la
boule finit à 0.379 pendant que le minimum *global* tombe à zéro, et tout est perdu dès `F = 0.1`.
C'est exactement le risque de changement de connectivité au bord : réel, mesurable, et il se paie en
épaisseur. **Deux couches suffisent**, et le coût est **fortement non monotone** en `N` : une
couronne trop mince coûte *plus* (580 000 cellules à `N = 1` contre 224 000 à `N = 2`, à `F = 0.05`)
parce que la fuite fait patiner la continuation en sous-pas refusés.

**La boule n'est pas un bloc, c'est une poussière d'amas** — et ça change le dessin du solveur. À
`F = 0.05` et deux couches : 20 composantes connexes, la plus grosse de 62 cellules. À `F = 0.1` :
126 composantes. Même épaissie, la zone reste une collection de petits problèmes indépendants de
quelques dizaines d'inconnues chacun. Or le code les résout **tous ensemble**, comme un seul système
couplé de 343 ou 1 241 inconnues, par gradient conjugué, à travers l'arbre global. C'est une triple
maladresse : le couplage entre amas éloignés est fictif ; le sous-pas de continuation est le
**minimum sur tous les amas**, donc un amas difficile impose sa lenteur aux autres ; et sur vingt
inconnues il n'y a **aucune raison de passer par une structure d'accélération** — une cellule d'amas
se calcule en force brute contre l'amas et sa couronne, quelques dizaines de germes, au lieu d'un
parcours de BSP précédé d'un `set_weights` en `O(n)`. Le découpage par amas ne rend pas ce coût plus
petit : il le fait disparaître.

**Le bilan net, en comptant un diagramme pour `n` calculs de cellule** (et sans compter le
`set_weights`, qui se paie en plus) :

| `F` (avec `N = 2`) | dépensés | économisés | net |
|---|---|---|---|
| 0.03 | ≈ 2 | 5 | **+3** |
| 0.05 | ≈ 11 | 7 | −4 |
| 0.10 | ≈ 77 | 16 | −61 |

À `F = 0.03` — **un seul amas de 63 cellules** — le relèvement est net positif pour la première
fois. Le régime favorable est donc celui des foyers rares et petits, ce qui est précisément le
régime où le découpage par amas s'impose.

## 14.3 Ce qui empêche encore d'en faire quelque chose

**Le coût, et il est entièrement dans l'implémentation.** 18.7 millions de cellules calculées pour la
réparation à `F = 0.2`, soit l'équivalent de **940 diagrammes** — contre 35 économisés. Attention à
ne pas mal lire ce chiffre : ce ne sont pas des cellules *réparées* mais des **évaluations**, soit
4 138 passes sur l'ensemble surveillé. Les cellules à réparer, elles, sont bien peu nombreuses — ce
sont les **anneaux** qui font le volume :

| | pincées | boule `N = 3` | boule + anneau | passes |
|---|---|---|---|---|
| `F = 0.05` | 38 = **0.19 %** | 684 = 3.4 % | 1 124 = **5.6 %** | 674 |
| `F = 0.10` | 234 = 1.2 % | 2 046 = 10 % | 2 944 = 15 % | 2 186 |
| `F = 0.20` | 809 = 4.0 % | 3 448 = 17 % | 4 523 = **23 %** | 4 138 |

À `F = 0.2` on « répare » près d'un quart du diagramme : ce n'est plus local. Le régime où le schéma
tient sa promesse est celui de `F = 0.05` — deux pour mille de cellules pincées, cinq pour cent
surveillées — et il donne déjà 92 diagrammes contre 107.

Deux causes au reste, toutes deux réductibles et aucune inhérente au schéma :

* **LE PRÉDICTEUR. C'est fait, et ça paie.** Au démarrage à chaud d'un sous-pas, extrapoler
  l'intérieur le long de `d` (la direction de Newton *globale*) n'est pas la bonne tangente. Celle du
  sous-problème l'est : à masses constantes dans la boule, `L_PP dw = − L_(P,bord) db`, c'est-à-dire
  une résolution de plus sur `L_PP` — SPD, Dirichlet, bien mieux conditionnée que les équations
  normales de Gauss-Newton (`--local-sans-tangente` pour l'enlever). À état final **rigoureusement
  identique** :

  | `F` | prédicteur | itér. de Gauss-Newton | cellules calculées | diagrammes après |
  |---|---|---|---|---|
  | 0.05 | le long de `d` | 234 | 757 576 | 92 |
  | 0.05 | **la tangente** | 277 | **668 780** (−12 %) | 92 |
  | 0.10 | le long de `d` | 863 | 6 435 584 | 91 |
  | 0.10 | **la tangente** | **397** (−54 %) | **2 484 736** (−61 %) | **87** |

* **Le `set_weights` global.** Chaque évaluation locale rafraîchit l'arbre en `O(n)` alors qu'on ne
  touche que quelques centaines de poids. C'est le chantier restant, et il est purement mécanique.

Avec la tangente et sans le `set_weights` local, on en est à `F = 0.1` pour 2.5 M évaluations
(≈ 124 diagrammes) contre 20 économisés : **le schéma gagne sur le pas et perd encore sur le temps,
d'un facteur six au lieu de quinze**. Ce qui est acquis et ne dépend pas de l'implémentation, c'est
qu'il n'y a **pas d'obstacle de principe** là où on en voyait un : la continuation traverse les
cellules mortes sans les rencontrer, et l'épaisseur d'anneau contrôle la fuite.

---

# 15. LE RELÈVEMENT BRANCHÉ DANS NEWTON (`image --newton-releve`)

Le § 14 a montré que la réparation locale marche sur **un** pas. Reste la vraie question : dans la
boucle, où le pincement se reforme à chaque itération. Le schéma implémenté est celui qui a été
proposé, avec une condition de plus que la mesure impose :

1. **Le pas** n'est plus borné par « aucune cellule sous `ε` » mais par « **au plus `ratio·n`
   cellules sous `ε`** » — on avance donc bien plus loin, en acceptant un nombre borné de malades ;
2. **et aucune cellule sous `ε_mort`**. Ce n'est pas un raffinement : une cellule de masse nulle a
   toutes ses `c_ij` nulles, donc une ligne de hessienne neutralisée (§ 13.1), et la réparation
   échoue à coup sûr. Le § 14.3 a mesuré cette frontière, et elle est franche ;
3. **le coloriage** : les malades, `N` anneaux autour, coalescés par le parcours en largeur ; seuls
   les poids du bord sont imposés ;
4. **la résolution** de l'intérieur sur l'objectif barrière `Φ = Σ (A/ν − ν/A)²` (§ 14) ;
5. si la réparation échoue, **on redescend le pas** — les « poids imposés intermédiaires » — jusqu'à
   ce qu'elle passe ; et le pas n'est accepté que si le **mérite descend**, sans quoi on perdrait la
   garantie de l'amortissement pour un point « sain » mais plus mauvais.

Sur le solveur local, **Gauss-Newton bat L-BFGS** ici : la jacobienne exacte est `L`, qui sort
gratuitement avec les cellules, et le patch fait trente inconnues — dix itérations suffisent. Le
line-search conservatif est bien le bon ingrédient, mais sous la forme de l'amortissement sur `Φ`
elle-même, qui vaut `+∞` si une cellule meurt.

## 15.1 Ce que ça donne sur un solve complet

Image 512² sans zéros, `n = 2·10⁴`, continuation géométrique à 14 marches, jusqu'à `t = 1`.

| | itérations | **diagrammes** | reculs | temps |
|---|---|---|---|---|
| **A.** Newton `essai-limites` (la référence du banc) | 239 | **451** | 33 | **15.8 s** |
| **B.** la même boucle, **sans** réparation (témoin) | 264 | 1 135 | 842 | 21.2 s |
| **C.** avec réparation, `ratio` 0.5 %, 1 anneau | 220 | **918** | 615 | 21.8 s |
| C. `ratio` 1 %, 1 anneau | 220 | 918 | 615 | 21.6 s |
| C. `ratio` 3 %, 1 anneau | 220 | 918 | 615 | 22.3 s |
| C. `ratio` 10 %, 1 anneau | 220 | 918 | 615 | 22.0 s |
| C. `ratio` 0.5 % … 10 %, **2 anneaux** | 219 | 923 | 619 | 23.6–24.0 s |

**La réparation marche dans la boucle : −19 % de diagrammes à boucle égale** (1 135 → 918), et 44
itérations de moins. C'est le résultat positif.

**Mais le ratio ne sert à rien.** De 0.5 % à 10 % — vingt fois plus de malades autorisées — le
résultat est **strictement identique** : 918 diagrammes, 615 reculs. Ce n'est donc pas le plafond de
malades qui borne le pas, c'est la **mort de la première cellule** (`ε_mort`), exactement comme le
§ 14.3 le prévoyait. La formulation « jusqu'où avancer sans dépasser 3 % de malades » est la bonne
question, mais en pratique le pas s'arrête avant, sur la première mort. Et un deuxième anneau ne
sert à rien non plus (923 contre 918, pour 8 % de temps en plus) — le § 14.2 l'avait déjà vu sur un
pas isolé : élargir dilue l'effort au lieu de le concentrer.

**En temps, c'est un match nul** (21.2 → 21.6 s) : les diagrammes économisés sont repayés en
cellules recalculées par les réparations.

## 15.2 Ce qu'il faut en conclure

Le gain est réel mais **il est petit devant celui de la règle de pas elle-même**. Sur la même
instance, passer de l'amortissement dyadique naïf à `essai-limites` — qui calcule le pas admissible
exact au lieu de le chercher par moitiés (§ 9.7) — fait 1 135 → 451 diagrammes. Le relèvement en
fait 1 135 → 918. **Calculer le bon pas vaut trois fois ce que vaut réparer après coup.**

Les deux ne sont pas concurrents pour autant : `essai-limites` s'arrête au pas où la **première**
cellule atteint le plancher, et le relèvement sert précisément à aller **au-delà**. Les combiner est
le prolongement naturel, et c'est ce que la mesure indique :

* **Mettre la réparation par-dessus `essai-limites`** plutôt que par-dessus le dyadique. L'ordre de
  grandeur attendu est celui du § 14.2 sur un pas isolé : un pas deux fois plus grand pour un
  cinquième de diagramme, soit environ −10 % sur les 451. Ça demande de toucher à `Newton.h`, ce qui
  n'a pas été fait ici : la boucle du § 15 est écrite dans `main_image.cpp` pour ne rien casser
  ailleurs, et c'est pour ça que son témoin est le dyadique naïf.
* **Lever le plafond des cellules mortes** : c'est fait, et autrement qu'on ne le croyait — la
  réparation par **continuation** (§ 14.1) ne rencontre jamais l'état mort, donc il n'y avait pas de
  verrou. La boucle a été reprise avec elle, et le § 15.3 dit ce que ça donne.

## 15.3 La boucle avec la réparation par amas : la séquence des pas, et ce que ça coûte

La réparation par amas (§ 16) est maintenant câblée dans la boucle, avec **le ratio pour seul
paramètre** : à chaque direction on cherche le plus grand `F` qui laisse au plus `ratio·n` cellules
sous `ε`, on répare, on avance. Le filtre `ε_mort` est retiré dans ce mode — il n'avait de sens que
pour la réparation par saut. Une résolution complète produit donc **une séquence de `F`**, une par
itération, et c'est elle qu'on veut lire.

**La séquence, sans réparation** (`n = 5·10³`, quatre étapes de continuation) :

```
étape 1 : 0.031 0.016 0.031 0.031 … 0.062 … 0.125 … 0.25 … 1
étape 3 : 0.031 0.062 0.062 … 0.125 … 0.25 0.25 0.5 0.5 1 1 1 1 1
```

**C'est l'information qui manquait à toute la discussion.** Le coefficient de relaxation n'est petit
qu'**au début de chaque étape** et remonte à 1 : sur 22 à 31 itérations par étape, une bonne moitié
se fait déjà à pas plein. Le relèvement ne peut donc agir que sur la première moitié — ce qui
**borne son gain possible**, indépendamment de ce qu'il coûte.

**Et ce qu'il coûte l'exclut.** `n = 5·10³`, quatre étapes :

| | itérations | diagrammes | temps |
|---|---|---|---|
| Newton `essai-limites` | 98 | 186 | **2.57 s** |
| la boucle, sans réparation | 113 | 471 | 4.40 s |
| la boucle **avec** réparation par amas | — | — | **> 20 min, interrompue** |

La ligne du bas est un **mur d'horloge, pas une divergence** : le calcul a été tué, on ne savait donc
rien de son profil. Le § 15.4 le mesure à `n` réduit, et le verdict s'inverse.

La raison est arithmétique : la réparation coûte 0.1 à 7 s, elle est appelée à **chacune** des ~110
itérations, et chaque appel paie en plus un diagramme global de rafraîchissement par sous-pas. Les
mesures isolées du § 16 — un diagramme dépensé, treize gagnés sur un foyer unique — **ne se
transposent pas** à la boucle, où la plupart des itérations ont des dizaines d'amas.

Tant que le rafraîchissement n'est pas **local**, le relèvement dans la boucle n'est pas utilisable.
Et la séquence des pas dit que, même rendu gratuit, son gain serait plafonné par la fraction
d'itérations qui ne sont pas déjà à pas plein.

## 15.4 Le profil de convergence, à `n` réduit — et le résidu ne remonte pas

Le temps n'est pas le bon instrument tant que le rafraîchissement est global : il mesure le
rafraîchissement, pas la méthode. À `n = 10³` la boucle va au bout dans les trois configurations, et
on peut lire ce qui nous intéresse — le **nombre d'itérations**, et ce que la réparation fait au
**résidu**. La trace imprime à chaque itération trois quantités : `|r|` avant, `|r|` après
réparation, et `|r|` du **pas nu au même `F`**, qui est la seule référence honnête.

| `n = 10³`, quatre étapes | itérations | diagrammes | verdict |
|---|---|---|---|
| la boucle, sans réparation | 56 | **166** | converge |
| avec réparation, aire de bord pénalisée | **40** | 342 | converge |
| avec réparation, objectif gradient d'aire | 41 | 349 | converge |

**Ça converge, et en 29 % d'itérations de moins.** Les diagrammes doublent, et ce doublement est
**entièrement** le rafraîchissement global — c'est-à-dire précisément la partie qu'on sait être
locale par nature et qui ne l'est pas encore.

**Le résidu ne remonte pas ; il descend plus vite.** Sur les quinze réparations acceptées, le rapport
`|r| après réparation / |r| du pas nu` vaut

```
0.21  0.29  0.36  0.37  0.46  0.51  0.51  0.52  0.67  0.92  0.97  0.98  0.99  1.00  1.03
```

— treize sur quinze **au-dessous de 1**, médiane 0.51, la meilleure à 0.21. La réparation ne défait
donc pas le progrès du pas : elle rend un état **meilleur en résidu que le pas nu** qu'elle corrige,
souvent d'un facteur deux à cinq. C'est cohérent avec ce qu'on cherchait — les cellules écrasées
sont aussi celles qui portent le gros du résidu.

Et la séquence des pas remonte en conséquence : `0.25 0.5 0.5 1 1 1 1 1` sur la dernière étape,
contre `0.125 ×5 0.25 ×4 0.5 1 1 1 1` sans réparation.

**Un défaut du critère d'acceptation, visible dans la trace.** Cinq pas sont refusés pour « encore
malade » alors que l'état réparé est **meilleur que les deux autres** — le plus net :

```
it 7  F 0.5  REFUS ( encore malade )   |r| 1.374e-02 -> 7.96e-03   pas nu 1.033e-02
```

On jette un état dont le résidu a chuté de 42 % parce qu'une cellule reste sous le seuil. Le critère
`amin3 > eps` est trop raide : accepter dès que l'état réparé est **sain ou strictement moins malade
que le pas nu**, et meilleur en résidu, supprimerait plusieurs reculs. Non fait.

## 15.5 Le rafraîchissement sur les patchs (`--local-patch`), et le seuil qui faussait tout

Le rafraîchissement global était la dernière dépendance en `n` de la réparation : un diagramme
complet par sous-pas, alors que tout ce qu'on en tire est le graphe de Laguerre **autour des amas**.
On le relit donc en force brute sur les seuls coupeurs, patch par patch, et on le donne au même
constructeur d'amas.

**À quelle condition c'est légitime.** Ne plus relire l'extérieur ne se défend que si l'extérieur n'a
pas bougé — et il ne bouge que par la couronne, sa frontière avec la zone réparée. C'est exactement
ce que `--local-bord` tient : l'**aire** de la couronne, pas seulement son poids. Le mode l'exige
donc et l'allume s'il ne l'est pas. Une cellule du patch est exacte dès que tous ses coupeurs y sont,
soit jusqu'à la distance `N+1+coupeurs` — la profondeur dont la construction a besoin, sans un poil
de marge ; comme la géométrie bouge, le mode élargit `C` d'une couche.

**Le seuil qui faussait tout.** La première mesure donnait **666 inconnues pour deux cellules
malades**, et 9.6 s sur 10.7 dans un seul amas. Les deux critères de « malade » n'étaient pas le
même : la boucle compte sous `eps`, un plancher **absolu** ; la réparation construisait ses amas
autour de `a < local_pince·ν` avec `local_pince = 0.5`, la **moitié de la cible cellule par
cellule**. Au départ d'une étape de continuation, où tout le diagramme est encore loin de sa cible,
des centaines de cellules passent sous cette barre sans être en danger. `repare_amas` prend
désormais le seuil de son appelant ; l'étude isolée du § 16 garde `local_pince`.

`n = 10³`, quatre étapes — le seuil corrigé ramène les amas à 1–4 par pas, 15 à 107 inconnues :

| | itérations | diagrammes | temps |
|---|---|---|---|
| sans réparation | 56 | 166 | 0.47 s |
| réparation, rafraîchissement global | **41** | 280 | 1.52 s |
| réparation, rafraîchissement **sur les patchs** | 44 | **141** | **0.91 s** |

Les 141 diagrammes du mode patch sont **sous** les 166 de la boucle nue : le relèvement ne se paie
plus du tout en `n`. L'écart 280 − 141, c'est exactement le rafraîchissement global.

`n = 5·10³`, quatre étapes — **le mur des 20 minutes est levé** :

| | itérations | diagrammes | temps | cellules locales |
|---|---|---|---|---|
| Newton `essai-limites` | 100 | **193** | **2.76 s** | — |
| la boucle, sans réparation | 113 | 471 | 4.19 s | — |
| la boucle, patchs | **95** | 478 | 19.2 s | 6.2 M |

**Les itérations sont les meilleures des trois** — 95, contre 100 pour la référence du banc. Mais le
temps est sept fois celui de la référence, et les diagrammes ne baissent pas. Deux causes, mesurées :

* **les cellules locales**, 6.2 M, soit 14.8 s des 19.2 : à `n = 5·10³` il y a une dizaine d'amas par
  appel et une centaine d'appels, et la force brute sans arbre les paie tous ;
* **les réparations refusées**, 96 en tout, dont **chacune paie un diagramme global** pour être
  jugée. C'est là que sont les 478 − 193 diagrammes, et c'est ce que le § 15.4 avait déjà désigné :
  le critère `amin3 > eps` refuse des états strictement meilleurs que le pas nu.

## 15.6 Où passe le temps dans la réparation — et ce qui n'y passe pas

Trois questions, trois mesures. Le binaire est *stripped*, donc `perf` ne rend que des adresses : on
instrumente à la main, en cinq postes disjoints posés au même endroit que le travail, plus un résidu
pour éviter de se raconter que la somme fait le total.

**Les inconnues.** Le système ne porte que sur `p < m`, c'est-à-dire `P` seul. La couronne a ses
poids **imposés** et n'est jamais mise à jour : la recherche linéaire n'écrit que `w[A.ens[p]]` pour
`p < m`, et la matrice locale ne garde une colonne que si `p < m`. La couronne n'entre que par son
**aire**, dans l'objectif.

**Le parallélisme : il n'y en a pas.** `mesures_et_facettes` tourne sur huit threads
(`parallel_for`) ; `resout_amas` et le rafraîchissement par patchs sont des boucles nues, un seul
cœur. Toute comparaison de temps entre la réparation et la référence du banc compare donc un cœur à
huit. Les amas d'un même sous-pas ont des inconnues disjointes par construction, donc la boucle est
parallélisable telle quelle — au prix d'un passage de Gauss-Seidel à Jacobi entre amas voisins.

**Le profil**, `n = 5·10³`, quatre étapes cumulées, sur les ~13 s de réparation :

| poste | temps | appels |
|---|---|---|
| **géométrie** (recherche linéaire) | **10.0 s** | **274 426** mesures |
| assemblage (dont `place` / `coupeur`) | 1.57 s | 24 910 |
| rafraîchissement sur les patchs | 0.98 s | |
| verdict | 0.33 s | |
| algèbre (Gauss-Newton + CG) | 0.20 s | |

Les deux suspects désignés avant mesure — les recherches linéaires de `place` et `coupeur` dans le
chemin chaud — sont dans l'assemblage, qui pèse 12 %. L'algèbre pèse 1.5 %. **Le coût est la
recherche linéaire** : onze évaluations géométriques par itération de Gauss-Newton.

### Ce qui a marché, ce qui n'a pas marché

**Le départ à chaud : presque rien.** Repartir du dernier pas accepté (doublé) au lieu de `t = 1` ne
retire que 5 % des évaluations (274 426 → 261 625). L'hypothèse « on repaie la descente à chaque
itération » était donc fausse.

**Le plafond de halvings : tout, à résultat identique.** Les 11 essais de moyenne étaient tirés par
les recherches **qui ne trouvent rien** : 9 371 échecs à 26.8 essais, soit **94 à 98 % du travail
géométrique**. Une recherche qui échoue descend jusqu'à `10⁻⁸`, et elle termine de toute façon la
boucle de Gauss-Newton (`break` au premier échec). En plafonnant à douze essais :

| | avant | après |
|---|---|---|
| évaluations | 261 625 | **133 701** |
| géométrie | 8.67 s | **5.24 s** |
| total | 15.25 s | **11.84 s** |
| itérations / diagrammes | 95 / 478 | **95 / 478** |

**L'arrêt « sain avec marge » : moins cher, moins bon, éteint** (`--local-marge`, défaut 0). On ne
cherche pas le minimum de la barrière mais un état non dégénéré, donc s'arrêter dès que le plancher
dépasse le seuil semblait gratuit. Mesure : 1.80 M cellules au lieu de 3.75 M et 9.92 s au lieu de
11.84 — mais **103 itérations et 549 diagrammes contre 95 et 478**. L'état rendu est *tout juste*
sain et ne survit pas au sous-pas suivant : les réparations acceptées tombent de 22 à 15, les refus
montent de 96 à 119, et chaque refus paie un diagramme global.

### Ce qui reste

Les 9 371 recherches en échec coûtent encore treize évaluations chacune, soit **91 % du travail
géométrique, uniquement pour constater que le sous-problème est fini**. Le bon critère d'arrêt n'est
ni la barrière (trop tard) ni la santé (trop tôt) : il reste à trouver. Les deux autres leviers non
tirés sont le parallélisme sur les amas et les listes de coupeurs par cellule — chaque cellule est
calculée contre les ~300 coupeurs de l'amas alors que les siens sont une vingtaine.

## 15.7 L'échelle : les événements exceptionnels étranglent le pas, et de plus en plus

La moyenne de la suite des pas cachait ce qu'on cherchait. Ce qui compte, ce sont le **minimum** et
la **fraction d'itérations sous 1/8** — la signature d'une poignée de cellules qui impose son pas à
tout le monde. Image 512², quatre étapes géométriques, boucle pilotée par le ratio de malades :

| `n` | min du pas | itér. sous 1/8 | diag. boucle nue | diag. `essai-limites` | rapport |
|---|---|---|---|---|---|
| 10³ | 6.3·10⁻² | 13 % | 166 | 96 | 1.7 |
| 5·10³ | 1.6·10⁻² | 51 % | 471 | 186 | 2.5 |
| 2·10⁴ | 7.8·10⁻³ | 72 % | 1 175 | 353 | 3.3 |
| 10⁵ | **2.0·10⁻³** | **88 %** | **3 308** | 786 | **4.2** |

Le pas minimal décroît à peu près comme `n^-3/4`, la fraction d'itérations étranglées sature vers
90 %, et le surcoût par rapport au pas exact **double tous les facteurs dix**. À 10⁵ diracs, la
boucle passe l'essentiel de son temps à des pas de l'ordre de 1/256. C'est le meilleur argument pour
le relèvement local — et le pire pour la façon dont on le pilote, puisque `essai-limites` fait quatre
fois mieux sans rien réparer.

## 15.8 Le gain potentiel, corrections supposées gratuites

Chaque tentative de réparation paie **exactement un diagramme global** pour être jugée : on les
compte, donc on sait retrancher ce que coûterait une correction rendue gratuite (le temps local, lui,
se lit directement dans le poste « relèvement »).

> **Ces chiffres sont périmés** : ils ont été pris avant la correction du § 15.9, et ils mesurent en
> grande partie un défaut de la réparation plutôt que la méthode. Conservés pour le raisonnement.

| diagrammes | boucle nue | avec réparation | **réparation gratuite** | `essai-limites` |
|---|---|---|---|---|
| 10³ | 166 | 141 | **114** | 96 |
| 5·10³ | 471 | 478 | **360** | 186 |
| 2·10⁴ | 1 175 | 1 159 | **852** | 353 |

Deux lectures, et elles ne disent pas la même chose.

**Contre la même boucle sans réparation**, le gain à correction gratuite est réel et stable : −31 %,
−24 %, −27 % de diagrammes, et −21 %, −16 %, −22 % d'itérations. Il ne dépend pas de la taille.

**Contre `essai-limites`, il ne rattrape rien et l'écart se creuse** : 1.19, 1.94, 2.41 fois plus de
diagrammes. Le pas exact par limites de masse reste très supérieur à « le plus grand pas qui garde
1 % de malades, puis on répare ».

Le coût local, lui, reste massif tant qu'il est monocœur : à `n = 2·10⁴`, 108 s de réparation sur
131 s, pour 34 M cellules — à comparer aux 1 159 × 2·10⁴ = 23 M cellules de tous les diagrammes du
Newton. **La réparation calcule plus de cellules que le Newton entier**, sur un cœur contre huit. Le
sous-problème est pourtant bien petit : 13 à 16 évaluations et quelques centaines de cellules
chacun ; c'est leur **nombre** qui fait la facture — des dizaines de milliers de sous-résolutions.

## 15.9 D'où venaient les refus : la réparation cherchait les malades là où ils avaient disparu

« Les diagrammes d'écart sont les refus » décrivait une comptabilité, pas une cause — et la cause ne
devait pas exister : la continuation ne valide un sous-pas que si l'état est sain, et au pire elle
rend `s = 0`, l'état de départ. On a donc classé chaque refus au lieu de le supposer : `s = 0`,
résidu, cellule morte **dans** la zone réparée (inconnue ou couronne), cellule morte **hors** d'elle
— et, pour chacune, ce que la mesure **locale** annonçait sur les mêmes poids.

Le verdict a été net : **100 % des refus étaient des cellules mortes dans la zone, que la mesure
locale déclarait saines.** Avec, en prime, un symptôme absurde :

```
it 1  F 6.250e-02  s 1.000  REFUS ( inconnue )  --  LOCAL dit 5.000e+03, GLOBAL dit 0.000e+00
```

`5.000e+03` à `n = 5000` : la cellule locale a **cinq mille fois sa masse cible**, c'est-à-dire tout
le domaine. Elle n'est coupée par rien.

**La cause.** `S`, l'ensemble à réparer, était repéré sur le diagramme de la **cible** `w₀ + F·d`. Or
une cellule déjà morte là-bas **ne produit aucune facette** : elle n'a donc aucun voisin dans le
graphe de Laguerre de la cible, son amas se réduit à elle seule, sans couronne et sans coupeurs. La
force brute la calcule contre une liste vide, rend le domaine entier, et la déclare florissante. La
continuation allait jusqu'à `s = 1` sans rien réparer, puis la mesure globale découvrait le cadavre.
**On construisait la structure de voisinage là où l'information avait précisément disparu.**

**Les corrections**, dans l'ordre de leur effet à `n = 5·10³` :

| | diagrammes | itérations | refus |
|---|---|---|---|
| avant | 478 | 95 | 96 |
| amas bâtis sur le diagramme de `w₀`, qui est sain | 420 | 87 | 76 |
| **le graphe local ne peut que croître** | **208** | **59** | **8** |
| seuils de santé alignés sur l'appelant | 215 | 61 | 5 |

La deuxième est la décisive, et c'est le même défaut un cran plus bas : le rafraîchissement rebâtit
la structure à partir des facettes de l'état **courant**, et une cellule qui meurt en chemin y perd
à nouveau tous ses voisins. On donne donc toujours à la construction l'**union** des facettes
fraîches et de celles de `w₀` — filtrées une fois par appel sur les patchs, pas une par sous-pas. Le
graphe ne peut alors plus rétrécir. Le rafraîchissement global souffrait du même défaut.

La troisième est **neutre en mesure** et gardée pour la cohérence : la réparation validait sur
`a/ν > local_eps` (relatif) là où la boucle exige `a > eps` (absolu), donc les deux pouvaient
diverger sans que ni l'une ni l'autre ait tort. Ce désaccord existait bien — il était visible dans
les derniers refus, `LOCAL dit 2.850e-01, GLOBAL dit 2.850e-01, eps/nu 3.095e-01` — mais il ne
coûtait presque plus rien une fois le vrai défaut corrigé.

**Ce que ça change au § 15.8.** Le tableau du gain à correction gratuite mesurait le bug, pas la
méthode : les diagrammes de jugement que j'y attribuais à un coût structurel du relèvement étaient
presque tous des cellules mortes invisibles à la réparation. Après correction :

| | itérations | diagrammes | `essai-limites` |
|---|---|---|---|
| `n = 10³` | **36** | **93** | 52 / 96 |
| `n = 5·10³` | **61** | 215 | 98 / 186 |

Le relèvement fait **40 % d'itérations de moins** que le pas exact par limites de masse, et le bat en
diagrammes à `10³`.

## 15.10 L'échelle, remesurée sur le code corrigé

Le § 15.7 mesurait la boucle nue, qui reste valable. Le § 15.8 mesurait le relèvement, et il mesurait
le défaut du § 15.9. Voici la même campagne sur le code corrigé, image 512², quatre étapes
géométriques :

| `n` | `essai-limites` | boucle nue | **avec relèvement** | relèvement, **correction gratuite** |
|---|---|---|---|---|
| 10³ | 52 it / 96 diag | 56 / 166 | **36 / 93** | **73** |
| 5·10³ | 100 / 193 | 113 / 471 | **61 / 215** | **170** |
| 2·10⁴ | 181 / 355 | 223 / 1175 | **108 / 504** | **395** |

**Le rapport d'itérations au pas exact est stable à 0.60** — 0.69, 0.61, 0.60. Le relèvement fait
constamment 40 % d'itérations de moins, et **ça ne se dégrade pas avec la taille**. En diagrammes à
correction gratuite, 0.76, 0.88, 1.11 fois `essai-limites` : l'avantage s'érode, lentement.

Sur les pas, à `n = 2·10⁴`, c'est le nombre d'étapes qui s'effondre — **32, 33, 19, 24 itérations par
étape contre 54, 62, 41, 66** — et le pas minimal remonte de 7.8·10⁻³ à 1.3·10⁻².

**Le coût est désormais le seul obstacle, et il est entier** : 360 s de réparation sur les 371 à
`n = 2·10⁴`, 86 M de cellules, **sur un cœur**. Il a beaucoup augmenté par rapport au § 15.6, et pour
une bonne raison : la réparation travaille maintenant au lieu de faire semblant — 5 914 cellules
malades traitées à `n = 2·10⁴`, contre une centaine avant la correction.

Les trois leviers restants, tous identifiés et aucun tiré :

* **le parallélisme**, un facteur 8 gratuit : les amas d'un sous-pas ont des inconnues disjointes ;
* **les listes de coupeurs par cellule** — chaque cellule est calculée contre les ~300 coupeurs de
  son amas alors que les siens sont une vingtaine ;
* **le critère d'arrêt du sous-problème**, qui coûte encore treize évaluations par sous-résolution
  juste pour constater qu'elle est finie (§ 15.6).

## 15.11 Le départ par prolongement harmonique : plus de sous-pas, et un relèvement enfin bon marché

La continuation faisait glisser la couronne de `w₀` à `w₀ + F·d` par sous-pas, et c'était son coût :
les cellules de bord bougent **beaucoup** avec `F`, donc il faut beaucoup de sous-pas, et chacun paie
une résolution locale complète. On pose donc la couronne **directement** à sa valeur finale — tous
les poids sont ceux de `F`, sauf ceux de `E` qu'on reconstruit — et c'est l'intérieur qu'on place.

### Ce qui n'a pas marché, et pourquoi

**L'homothétie exacte des cellules.** En diagramme de Laguerre, une homothétie de rapport `λ` et une
translation `−c` du diagramme s'écrivent en forme fermée :
`w_i = λ w0_i + (1−λ)|p_i|² + 2c·p_i`, et alors `C_i(w) = λ C_i(w0) − c`. Toutes les cellules
rétrécissent du même facteur `λ²`, donc aucune ne meurt — c'est séduisant, et c'est inutilisable
ici : le terme `|p_i|²` est d'ordre 1 là où tout ce qui se passe localement est d'ordre `10⁻⁴`. Les
seuls `λ` de cette famille qui ne détruisent pas l'amas sont ceux qu'on ne distingue pas de 1.
Mesure : `λ` retenu entre 0.93 et 1.00, et 80 à 123 amas abandonnés sur 96 à 143.

**Le Voronoï décalé** (`w_i = λ(w0_i − w̄) + β`, dont `λ = 0` donne tous les poids égaux). Il a une
garantie apparente — pour `β` assez grand chaque germe appartient à sa propre cellule — et elle est
fausse, parce que la couronne doit survivre aussi. Le balayage de `β` le montre sans appel, sur un
amas de 31 inconnues et 19 de couronne dont les poids s'étalent de `−1.4·10⁻³` à `+4.4·10⁻³` :

```
beta -4.28e-03   interieures mortes  30      couronne morte 0
beta -6.62e-04   interieures mortes  15      couronne morte 0
beta +6.40e-04   interieures mortes  10      couronne morte 0
beta +1.07e-03   interieures mortes   8      couronne morte 1   <-- la couronne lâche
```

**Aucun `β` ne passe, et il s'en faut de huit à dix cellules.** L'arithmétique le confirme :
l'étalement des poids de couronne vaut `5.8·10⁻³` quand le carré de la distance entre germes vaut
`9·10⁻⁴` ; la condition pour qu'un `β` unique existe est que l'étalement reste sous `2·dist²`, on en
est à **six fois trop**.

### Ce qui marche : le prolongement harmonique

Le défaut est maintenant nommable : **un niveau constant ne peut pas suivre une frontière qui
varie**. On donne donc à l'intérieur le prolongement harmonique du bord — `L_PP w_P = −L_PR w_R` sur
le graphe de l'amas, avec les conductivités de Laguerre, résolu en Gauss-Seidel sur quelques dizaines
d'inconnues. Le champ obtenu épouse la frontière par construction et, étant harmonique, il n'a **ni
maximum ni minimum intérieur** — or c'est la courbure du champ de poids qui écrase une cellule, pas
son niveau. La famille complète est

```
w_i = harm( w_F )_i  +  λ ( w0_i − harm( w0 )_i )  +  β
```

où `λ = 1` garde le détail local de `w₀` en le reposant sur la nouvelle frontière, `λ = 0` rend le
prolongement pur, et `β` est encadré par bissection (les inconnues grossissent avec lui, la couronne
rétrécit : deux monotonies opposées). Le Voronoï décalé en est le cas « frontière constante ».

| `n = 10³`, 2 anneaux | itérations | diagrammes | amas abandonnés | cellules / étape |
|---|---|---|---|---|
| `essai-limites` | 52 | 96 | — | — |
| Voronoï décalé | 48 | 151 | 9 sur 11 | 0.5 – 0.7 M |
| **harmonique** | **36** | **91** | **0** | **7 k – 27 k** |

Zéro amas abandonné, zéro à un refus par étape, et le coût de la réparation divisé par **cinquante**.
La boucle bat `essai-limites` sur les deux comptes à la fois.

`n = 5·10³`, 2 anneaux : **59 itérations et 223 diagrammes** contre 100 et 193 — 41 % d'itérations en
moins, et 170 diagrammes à correction gratuite (53 des 223 sont des jugements), sous les 193 de la
référence.

**Quatre anneaux sont moins bons que deux** : mêmes itérations, coût multiplié par cinq.

## 15.12 Les amas abandonnés sont les gros, et il n'y a rien à y chercher

À `n = 5·10³`, la première étape gardait 17 amas abandonnés coûtant 2.4 M cellules à eux seuls. Le
balayage de `β` sur l'un d'eux dit pourquoi — **266 inconnues et 177 de couronne**, soit 5 % du
diagramme dans un seul bloc :

```
beta  0.000e+00   interieures mortes 109      couronne morte   8
beta +4.85e-04    interieures mortes  76      couronne morte  97
```

À `β = 0`, c'est-à-dire **le prolongement harmonique pur**, 109 des 266 cellules intérieures sont
déjà mortes. Et la raison est dimensionnelle, donc sans recours : les poids de couronne s'étalent sur
`10⁻²` à travers un amas large d'une dizaine de cellules, soit `10⁻³` d'écart **par arête**, contre
`dist² ≈ 9·10⁻⁴`. Le champ de poids est **trop raide pour l'espacement des germes**, et aucune
interpolation ne peut l'annuler puisque la pente est imposée par le bord. Il n'y a pas de placement
initial à trouver : la géométrie demandée est elle-même écrasée.

Le remède est donc de **refuser tôt** (`--local-amas-max K`) : au-delà de `K` inconnues on décline
sans chercher, la boucle divise `F` par deux, et l'amas suivant est plus petit. La taille des amas
devient le vrai paramètre de la réparation — plus fidèle que le ratio global, qui ne dit rien de la
façon dont les malades se groupent.

| `n = 5·10³` | itérations | diagrammes | temps | cellules | abandons |
|---|---|---|---|---|---|
| sans plafond | 59 | 223 | 9.39 s | 3.6 M | 23 |
| **K = 128** | **60** | **238** | **3.22 s** | **0.54 M** | **3** |
| K = 64 | 71 | 303 | 3.10 s | | |
| K = 32 | 83 | 389 | 3.63 s | | |

À 128 on garde le résultat du cas sans plafond pour **trois fois moins de travail**. En dessous, le
plafond mord et il faut payer en itérations. C'est le défaut.

### Où en est le relèvement

`n = 5·10³`, contre la référence du banc : **60 itérations contre 97** (0.62×), 238 diagrammes contre
184 — dont 53 de jugement, donc **185 à correction gratuite, à égalité**, avec 38 % d'itérations en
moins. La réparation coûte 1.25 s sur 3.22, **monocœur**, face à un Newton sur huit threads.

Ce qui reste, par ordre de rendement : le parallélisme sur les amas (facteur 8, les inconnues de deux
amas d'un même pas sont disjointes) ; le critère d'arrêt du sous-problème (§ 15.6), qui brûle encore
treize évaluations par sous-résolution pour constater qu'elle est finie ; les 53 diagrammes de
jugement, qu'un verdict local suffirait peut-être à remplacer.

## 15.13 `U` : le pas se lit au lieu de se chercher

La boucle cherchait `F` à tâtons — un **diagramme complet par essai**, 113 reculs pour 60 itérations à
`n = 5·10³`. Or l'information est déjà là. Le flux d'aire qui sort de la cellule `i` par la facette
`j` vaut exactement `c_ij ( d_j − d_i )`, le terme du laplacien déjà assemblé. Le temps que cette
facette met à consommer toute la cellule est donc

```
U_i  =  a_i  /  max_j [ c_ij ( d_j − d_i ) ]        sur les j qui font perdre de l'aire
```

**Aucune géométrie supplémentaire, aucun calcul de cellule** : une passe sur les arêtes. (Cohérence :
la somme des flux vaut `( L d )_i = ν_i − a_i`.) La bissection de `limites_masse` était le mauvais
chemin — 8 695 cellules pour 100 itérations en ne traitant que les mauvaises, donc une dizaine de
diagrammes par itération si on l'étendait à toutes.

**La courbe des amas est gratuite elle aussi.** On insère les cellules par `U_i` croissant dans une
union-find en fusionnant avec les voisines déjà insérées : à chaque insertion on connaît la taille du
plus gros amas. Un tri et `O( n α( n ) )` donnent donc, **pour tous les `F` à la fois**, le nombre de
malades, le nombre d'amas et la taille du plus gros. Pas besoin d'un essai par `F`, ni d'un thread
par proposition.

**Le prédicteur est bon.** Le `F` prédit pour un amas de malades ≤ 8 tombe systématiquement juste
au-dessus du dyadique que la boucle finissait par retenir : 6.49·10⁻² contre 6.25·10⁻² à l'itération
0, 1.23·10⁻¹ contre 1.25·10⁻¹ à la 8, 2.53·10⁻¹ contre 2.50·10⁻¹ à la 13. Il sur-estime parfois, donc
la dyadique reste en repli — mais elle ne sert plus qu'aux exceptions.

| | `essai-limites` | relèvement, dyadique | **relèvement par `U`** |
|---|---|---|---|
| `n = 10³` | 52 it / 96 diag / 0.32 s | 36 / 91 / 0.40 s | **36 / 79 / 0.83 s** |
| `n = 5·10³` | 98 / 186 / 2.45 s | 60 / 238 / 2.92 s | **61 / 159 / 2.91 s** |
| `n = 2·10⁴` | 181 / 355 / 12.41 s | 118 / 659 / 16.36 s | **116 / 384 / 13.37 s** |

Les reculs s'effondrent — 113 → 38, 364 → 102 — et avec eux un tiers à 42 % des diagrammes. **À `10³`
et `5·10³` le relèvement bat désormais `essai-limites` en diagrammes bruts**, plus seulement à
correction gratuite. À `2·10⁴` il reste 8 % au-dessus, avec 116 itérations contre 181, et le temps est
à la parité à 8 % près — la réparation étant encore **monocœur** face à un Newton sur huit threads.

**Le seuil `M` n'est pas critique** : `M = 4` et `M = 8` donnent exactement le même résultat, 16 et 32
dégradent nettement (455 diagrammes à 32 contre 384 à 8). Défaut : 8. Et il recoupe le plafond de
taille du § 15.12, mesuré par un chemin indépendant : 8 malades entourés de deux anneaux donnent 15 à
110 inconnues, soit exactement la plage sous 128.

## 15.14 `U` par la somme des flux, et le relèvement passe devant

Les reculs restants venaient d'une approximation grossière dans `U` : prendre `max_j` du flux revient
à supposer qu'**une seule** facette dévore la cellule, alors que plusieurs la mangent en même temps.
La somme des flux sortants est strictement plus conservative et se calcule dans la **même boucle** :

```
U_i  =  a_i  /  Σ_j max( 0, c_ij ( d_j − d_i ) )
```

Ce n'était donc pas d'un plafond de confiance sur `F` qu'on avait besoin — un plafond n'aurait fait
que masquer l'imprécision. C'était que la quantité était fausse d'un facteur qui dépend du nombre de
facettes dévorantes.

| | `essai-limites` | `U` max | **`U` somme** |
|---|---|---|---|
| `n = 5·10³` | 100 it / 193 diag | 61 / 159 (38 reculs) | **61 / 122** (15 reculs) |
| `n = 2·10⁴` | 183 / 362 | 116 / 384 (102 reculs) | **123 / 331** (65 reculs) |

**−37 % de diagrammes contre la référence du banc à `5·10³`, −9 % à `2·10⁴`** : pour la première fois
le relèvement gagne sur les deux tailles en diagrammes bruts. Les reculs tombent encore de moitié.
(Les temps mesurés — 1.57 s contre 2.45 à 3.07 s selon les passes à `5·10³`, 11.4 contre 12.2 s à
`2·10⁴` — vont dans le même sens, mais ils sont pris hors mode banc et ne valent que comme ordre de
grandeur.)

**Deux ajouts au passage.** La recherche linéaire du sous-problème est parallélisée : sans
assemblage, sa boucle n'écrit que `a[q]`, donc aucune dépendance entre cellules. Paralléliser **sur
les amas** demanderait en revanche de supprimer le `w` partagé, leurs coupeurs se recouvrant — pas
fait. Et un **buffer tournant** garde les quatre derniers états sains : si une itération démarrait
d'un état dégénéré, la boucle n'aurait aucun recours, puisqu'une cellule morte n'a plus de voisins
donc plus de réparation possible (§ 15.9). Il ne s'est jamais déclenché, ce qui est le résultat
attendu — il reste comme filet, avec son compteur.

## 15.15 Le `U` exact est moins bon que le `U` grossier — et ce qui marche à la place

`U` est une linéarisation en `t = 0` : les flux y sont figés, et la géométrie accélère quand des
facettes disparaissent. On a donc calculé la **vraie** limite, par bissection, pour les seules
cellules dont le `U` bon marché tombe sous un horizon `β` — c'est le seul rôle utile de `β` : borner
le nombre de candidates, une bissection coûtant des calculs de cellule.

**La bissection doit être géométrique.** En arithmétique, `mid = (a_ok + a_bad)/2` depuis `a_ok = 0`
met dix-sept demi-pas rien que pour descendre jusqu'à `10⁻⁵·β`, alors que `max_tours` en vaut douze :
elle rend `a_ok = 0`, c'est-à-dire un pas nul — d'autant plus sûrement que `n` est grand, les pas
admissibles y étant minuscules. On descend donc par facteur 4 puis on prend la moyenne géométrique
(`OptionsLimites::log_ech`).

**Et le verdict est monotone, dans le mauvais sens** (diagrammes, bissection géométrique) :

| | `n = 5·10³` | `n = 2·10⁴` |
|---|---|---|
| sans bissection | **122** | **331** |
| β = 1/64 | 122 (7 cellules raffinées) | 331 (460) |
| β = 1/16 | 133 | 369 |
| β = 1/4 | 161 | 430 |

Plus on raffine, pire c'est, et `β → 0` retrouve le comportement non raffiné. Le mécanisme se lit
dans les reculs — 65 sans raffinement, 122 à `β = 1/4` — alors même que `β` **plafonne** `F`. Le
raffinement **remonte** `U` (la vraie limite est bien plus loin que ce que le flux en `t = 0`
suggère), donc `F` monte, donc la réparation échoue plus souvent. La conservativité accidentelle du
`U` grossier faisait du bon travail.

**Ce n'est donc pas `U` qui est faux, c'est le critère.** « Le plus grand `F` laissant au plus 8
malades » ne prédit pas la réussite de la réparation ; avec un `U` exact il est appliqué exactement,
et il choisit mal.

### La raideur, qui est le bon critère

Ce qui décide vraiment a été mesuré au § 15.12 : un amas échoue quand l'étalement des poids **à
travers lui** dépasse quelques `dist²`. C'est un gradient accumulé sur une dizaine de cellules, pas
une condition par arête. (Une paire `i, j` avec `|w_i − w_j| > |p_i − p_j|²` signifie seulement que le
bissecteur passe au-delà d'un germe — parfaitement banal en Laguerre, et sans conséquence.)

L'union-find peut le suivre : en insérant par `U` croissant, on maintient par amas les `min`/`max` de
`w` et de `d` et le plus petit `dist²` de ses arêtes, donc la borne

```
étalement( F )  ≤  ( w_max − w_min )  +  F ( d_max − d_min )
```

croissante en `F`, lisible dans le même balayage.

| diagrammes | `n = 5·10³` | `n = 2·10⁴` |
|---|---|---|
| `essai-limites` | 186 | 363 |
| taille d'amas seule | 122 (15 reculs) | 331 (65 reculs) |
| **+ raideur** | 123 (**8 reculs**) | **257** (**26 reculs**) |

**−22 % de diagrammes contre la version précédente à `2·10⁴`, −29 % contre `essai-limites`**, et les
reculs divisés par 2.5.

**Mais `R` n'est pas un vrai réglage** : 1, 2, 4 et 8 donnent le même résultat au diagramme près. La
borne ne franchit pas le plafond progressivement, elle **saute** — et ce saut est la fusion de deux
amas, qui fait bondir l'étalement d'un coup. Le critère revient donc en pratique à « s'arrêter à la
première grosse fusion ». Défaut : 2.

**Le seuil de « malade » en ratio `a/ν`** (`--releve-ratio-eps`) est neutre à 0.05 et **casse la
boucle** à 0.2 et 0.5 — quatre itérations puis stagnation. Élargir la définition de malade sans
élargir le budget `--releve-ratio` rend tout pas inacceptable : les deux réglages sont couplés, et ils
n'ont pas été balayés ensemble.

**Une limite structurelle à garder en tête** : `U` comme la raideur se lisent sur le graphe à
`F = 0`, donc **tous deux sont aveugles aux coupes qui n'existent pas encore**. C'est la même cécité
qu'au § 15.9 — un coupeur absent ne produit aucune facette, donc rien dans `L` ne le signale.

## 15.16 « Peut-on toujours y placer un diagramme ? » — non, et pourquoi

L'homothétie exacte des cellules est un fait, pas une approximation :

```
w_i = λ w⁰_i + (1−λ)|p_i|² − 2 t·p_i + β      ⟹      C_i(w) = λ·C_i(w⁰) + t
```

Le motif entier se contracte, **aucune cellule ne disparaît**. La question était donc : peut-on
toujours insérer un tel motif, aussi petit qu'il faille, dans un amas dont la couronne est gelée à
`w_F` ? La réponse mesurée est **non**, et le chemin pour y arriver a coûté quatre hypothèses
théoriques successives, toutes fausses.

**Ce qu'il a fallu corriger pour que la mesure veuille dire quelque chose.**

* `x*` **n'est pas `argmin g`** avec `g(y) = max_k(2y·q_k − |q_k|² + v_k)`. « Là où l'intérieur gagne
  le moins cher » veut dire « là où l'arrangement gelé résiste le moins », donc là où une cellule
  gelée est déjà la plus mince : on la tue au moment même où l'intérieur apparaît.
* La version « centrée » par transformée de distance sur grille rendait un rayon libre de 0.099 sur
  un amas de diamètre 0.05 — la région marquée touchait le bord de la grille. `x*` est finalement le
  **barycentre des cellules intérieures valides à `F`**, sans grille ni paramètre.
* **La translation dépend de `λ`** : `t = x* − λ z₀`, avec `z₀` le centre du motif d'origine. Poser
  `t = x*` est inoffensif à `λ = 10⁻⁸` mais place le motif à `(0.19, 0.17)` de la cible à `λ = 0.25`
  — hors d'un amas large de 0.15. Les deux bouts de l'échelle échouaient donc pour deux raisons
  différentes, ce qui donnait l'illusion d'une impossibilité uniforme.
* Un balayage géométrique de `β` **ne peut pas répondre** : la fenêtre cherchée peut être `10⁷` fois
  plus étroite que son pas. Il faut une **bissection encadrée**, et imprimer la largeur.

**Le verdict, une fois tout cela corrigé.** Sur deux amas, sept décades de `λ`, bissection à
l'epsilon machine : aucun `β` ne garde intérieur et couronne vivants. Et le comptage au passage dit
pourquoi — ce n'est pas une transition, c'est un **recouvrement** :

```
transition autour de beta +9.248e-01 :
  beta -1e-12 :   1 inconnue morte / 66,  54 couronnes mortes / 55
  beta +1e-16 :   0 inconnue morte / 66,  54 couronnes mortes / 55
```

Au moment où la dernière inconnue s'allume, **54 cellules de couronne sur 55 sont mortes depuis
longtemps**.

**Le mécanisme.** Contracter d'un facteur `λ` impose le profil `(1−λ)|p_i − z|²`, dont l'étalement à
travers l'amas est non nul dès que `λ < 1`. Les germes **loin de `z`** ont donc un poids bien plus
élevé et avalent la couronne, pendant que ceux **près de `z`**, de poids plus faible, dorment encore.
Le `β` qui réveille les derniers a depuis longtemps tué les premières. **Contraction et compatibilité
de niveau avec la couronne sont antagonistes** : l'une exige un profil quadratique, l'autre un profil
plat.

**Ce qui réconcilie tout.** La famille qui marche (§ 15.11) est bien un *scaling* — mais de l'écart
au champ compatible avec le bord, pas du motif géométrique :

```
w_i  =  harm( w_F )_i  +  λ ( w⁰_i − harm( w⁰ )_i )  +  β
```

À `λ = 0` c'est le champ harmonique pur, dont l'étalement est celui de la couronne **par
construction** ; à `λ = 1` on garde tout le détail local de `w⁰`. C'est le bon objet à contracter.

## 15.17 Les tailles, remesurées sur le code complet — et deux réglages qui se périment

Le plafond `K = 128` (§ 15.12) et le seuil `M = 8` (§ 15.13) avaient été réglés **avant** `U`, avant
le critère de raideur et avant le parallélisme. Remesurés sur le code complet, en diagrammes :

| | `n = 5·10³` | `n = 2·10⁴` |
|---|---|---|
| `K = 32` | 202 | 431 |
| `K = 64` | 165 | 317 |
| `K = 128` *(ancien défaut)* | 123 | 257 |
| `K = 256` | 114 | 227 |
| **`K = 0`, sans plafond** | **114** | **227** |

**Le plafond nuit désormais, et on l'éteint.** `K = 256` égale `K = 0` : plus aucun amas n'atteint
cette taille. Il était la bonne réponse mesurée à un problème qui n'existe plus — à l'époque les gros
amas coûtaient 2.4 M cellules pour rien ; depuis, le critère de raideur choisit `F` de sorte qu'ils
ne se forment plus, et le plafond ne décline que des amas qui auraient réussi.

**`M` est devenu inerte** : 4, 8, 16, 32 donnent des résultats identiques au diagramme près, parce
que la raideur choisit `F` en premier à chaque fois. C'est le signe qu'un mécanisme en amont a pris
la main.

**Les anneaux confirment 2**, franchement : 174 / **123** / 177 à `5·10³`, 316 / **257** / 322 à
`2·10⁴` pour 1, 2, 3 anneaux.

### Raideur contre `M` : deux tempéraments, et lequel tient à l'échelle

Raideur éteinte, `M` seul pilote `F`. Les deux critères se partagent le terrain en sens opposés —
`M` prend de plus grands pas mais se trompe plus souvent :

| | `essai-limites` | raideur | `M = 8` seul |
|---|---|---|---|
| `n = 5·10³` | 98 it / 186 diag | 72 / 114 (5 reculs) | **57 / 106** (10 reculs) |
| `n = 2·10⁴` | 179 / 351 | **134 / 227** (15) | 113 / 277 (47) |
| `n = 10⁵` | 397 / 792 / 188 s | **324 / 574 / 167 s** (55) | 267 / 747 / 265 s (156) |

À `5·10³` l'audace gagne ; dès `2·10⁴` la prudence l'emporte en diagrammes, et à `10⁵` l'écart est
franc — 574 contre 747, et 167 s contre 265 s. **La raideur reste le défaut**, et son avance grandit
avec `n`. `R` n'est d'ailleurs plus tout à fait insensible : 0.5 et 4 coûtent 233 diagrammes contre
227 pour 1 et 2 à `2·10⁴` — optimum plat entre 1 et 2, ce qui lève la réserve du § 15.15.

### Où en est le relèvement, sur trois décades

| `n` | `essai-limites` | **relèvement** | diagrammes |
|---|---|---|---|
| 5·10³ | 98 it / 186 diag | **72 / 114** | **−39 %** |
| 2·10⁴ | 179 / 351 | **134 / 227** | **−35 %** |
| 10⁵ | 397 / 792 / 188 s | **324 / 574 / 167 s** | **−28 %** |

**Une leçon de méthode.** Deux réglages mesurés se sont périmés dans cette session par l'arrivée d'un
mécanisme en amont : le plafond `K`, et les 307 diagrammes de jugement du § 15.8 que je croyais
structurels et qui étaient un bug. Un réglage devenu **inerte** — comme `M` aujourd'hui — signale que
quelque chose en amont a pris la main, et mérite qu'on aille voir.

## 15.18 Le gradient local après le placement : indispensable, et quatre itérations suffisent

Le placement harmonique (§ 15.11) pose un motif **vivant mais très contracté** — les aires valent
`lam²` de leur taille. Ce qui le redéploie, c'est la boucle de Gauss-Newton locale sur

    Phi = somme_( aretes ) ( x_i - x_j )²,   x = A / nu

qui tourne juste après, dans `resout_amas`. Question légitime : le Newton global qui suit ne
ferait-il pas le travail tout seul ? Balayage du plafond d'itérations, `n = 10⁵` :

| `--local-maxit` | itérations | diagrammes | reculs | temps |
|---|---|---|---|---|
| **0** (placement seul) | 416 | **1063** | 243 | 209,7 s |
| 1 | 384 | 897 | 180 | 182,7 s |
| 2 | 333 | 622 | 72 | 145,4 s |
| **4** | 323 | **566** | 53 | **141,4 s** |
| 40 *(ancien défaut)* | 324 | 574 | 55 | 142,2 s |

**Non.** Sans ces itérations le relèvement est *pire qu'`essai-limites`* (1063 diagrammes contre 792)
— le Newton global n'absorbe pas un motif contracté, il l'encaisse en 243 reculs. Le placement seul
ne suffit donc pas : il rend un état **vivant**, pas un état **utilisable**.

Mais **quatre itérations font aussi bien que quarante**, et légèrement mieux. Le défaut passe de 40 à
4 : même résultat, coût borné — et comme cette boucle est séquentielle, son plafond est aussi un
plafond sur la part non parallélisable du relèvement (§ 17.9).

---

# 16. LE RELÈVEMENT PAR AMAS (`image --local-nl --local-amas`)

Le § 14 répare la zone pincée comme **un seul système couplé**, résolu par gradient conjugué à
travers l'arbre global. C'est une maladresse, et la mesure la désigne : la zone n'est pas un bloc
mais une **poussière d'amas** — 20 composantes connexes dont la plus grosse fait 62 cellules, à
`F = 0.05` et deux couches ; 126 composantes à `F = 0.1`. Le couplage entre amas éloignés est
fictif, et sur vingt inconnues il n'y a **aucune raison de passer par une structure
d'accélération** : une cellule d'amas se calcule en force brute contre l'amas et sa couronne,
quelques dizaines de germes (`Balayage2`, le fournisseur témoin du banc), au lieu d'un parcours de
BSP précédé d'un `set_weights` en `O( n )`. Le découpage par amas ne réduit pas ce coût : il le
supprime.

Les trois ensembles d'un amas : `P` les **inconnues** (la composante connexe) ; `R` la **couronne**,
à poids imposés mais **dans l'objectif** — c'est ce qui la protège ; `C` les **coupeurs**,
`P + R + N( R )`, dont la force brute rend les cellules de `P + R` exactes.

**Deux choses qui ont chacune coûté une mesure fausse.** Les amas se fusionnent dès que **leurs
couronnes** se touchent (composantes de `dist ≤ N+1`, pas de `dist ≤ N`) : sinon deux amas voisins
partagent des cellules de bord, chacun les protège dans son coin, et leurs effets s'additionnent
pour les tuer. Et **le chemin reste commun** : donner à chaque amas sa propre continuation ne marche
pas, chacun résolvant en supposant les autres non réparés, la superposition n'est cohérente avec
aucune des solutions.

## 16.1 Le coût s'effondre, comme prévu

`n = 2·10⁴`, image 512², deux couches, référence à 100–103 diagrammes :

| `F` | amas | + gros | inconnues | sous-pas (refus) | itér. | cellules | **temps** | diag. après |
|---|---|---|---|---|---|---|---|---|
| 0.03 | **1** | 63 | 63 | 4 (0) | 18 | 30 438 | **0.05 s** | **94** |
| 0.05 | 17 | 76 | 343 | 21 (10) | 732 | 442 152 | 0.77 s | ✗ |
| 0.10 | 117 | 289 | 1 241 | 31 (21) | 4 646 | 2 994 350 | 11.1 s | ✗ |

À `F = 0.03` — **un seul amas de 63 inconnues** — la réparation coûte **cinquante millisecondes**,
soit environ un diagramme, et le Newton qui repart de là converge en 98 diagrammes au lieu de 100.
C'est le régime que l'idée visait, et il est net.

## 16.2 Le défaut, trouvé puis corrigé : la structure de voisinage vieillit

Au-delà de `F = 0.03`, la première version tombait à une masse globale nulle. Trois hypothèses
raisonnables ont été **éliminées par la mesure** : la liste de coupeurs trop courte (deux couches de
plus ne changent rien au chiffre près) ; la dérive de l'aire de la couronne (lui donner pour cible
**l'aire qu'elle avait avant la réparation**, résidu `g( x ) + √λ ( x − x_ref )`, **divise le coût
par deux** — 0.60 → 0.31 s à `λ = 10` — et ne change pas la masse minimale) ; le verdict pris amas
par amas (remesuré après la boucle maintenant, correct mais sans effet).

**La cause, trouvée par contradiction** : la mesure locale d'un amas annonçait `0.303 ν` pour sa pire
cellule là où la mesure globale des **mêmes poids** donnait `0.000` — pour une **inconnue**, à
distance 2. La cellule locale était un **sur-ensemble** : il lui manquait un coupeur. Aucun compteur
de facettes ne pouvait le voir, puisqu'un coupeur manquant ne produit **aucune facette**. Les listes
de candidats étaient calculées **une fois**, sur la géométrie de la cible, alors que la réparation
**déplace beaucoup** les cellules — c'est son objet : un amas qui gonfle avale une cellule d'un amas
voisin qui ne l'avait jamais eue pour candidat.

**Le remède** (`--local-refresh K`) : refaire la structure de voisinage **une fois par sous-pas**,
pas par évaluation. Une passe globale par sous-pas contre des centaines d'évaluations locales — le
facteur cent est préservé. Et il corrige : à `F = 0.05`, masse minimale globale **0.280** au lieu de
0.000, et le Newton d'après converge en **87 diagrammes contre 105**.

## 16.3 Deux objectifs, et celui qui a l'air mieux posé perd

Viser `ν` sur l'intérieur **force** à prendre de la masse à la couronne : on s'impose une contrainte
de conservation dont on n'a pas besoin, puisqu'on ne cherche pas la solution mais un état non
dégénéré. Un objectif sans cible absolue s'impose donc naturellement — le **gradient d'aire**
(`--local-grad`) :

        Φ = Σ_( arêtes de la zone ) ( A_i/ν_i − A_j/ν_j )²

qui ne pousse sur rien, et dont un zéro au milieu d'un champ lisse est impossible. Gauss-Newton sans
matrice : `Lᵀ D K D L d = − Lᵀ D K x`, avec `K` le laplacien du graphe et `D = diag( 1/ν )`.

`n = 2·10⁴`, image 512², deux couches, référence **105 diagrammes** :

À la sortie, les poids valent `w₀ + s·F·d` partout sauf sur les inconnues : **le bord n'est le pas
de Newton demandé que si `s = 1`**. Sinon le coefficient de relaxation réellement obtenu est
`s·F`, et c'est lui qu'il faut lire — la colonne est là pour ça. Rappel : `α* = 0.0156` sans
réparation.

| objectif | rafr. | `F` | `s` | **coeff. réel `s·F`** | min globale | cellules | passes | temps | **diag. après** |
|---|---|---|---|---|---|---|---|---|---|
| barrière | non | 0.03 | 1.00 | 0.030 | 0.358 | 25 764 | 0 | 0.05 s | **92** |
| barrière | non | 0.05 | 1.00 | 0.050 | **0.000** ✗ | 213 989 | 0 | 0.29 s | ✗ |
| barrière | oui | 0.03 | 1.00 | 0.030 | 0.338 | 21 030 | 4 | 0.10 s | 95 |
| barrière | oui | 0.05 | 1.00 | **0.050** | **0.280** | 326 076 | 12 | 0.84 s | **87** |
| barrière | oui | 0.10 | 0.34 | 0.034 | 0.041 | 1 358 717 | 21 | 7.08 s | 89 |
| **gradient** | oui | 0.03 | 1.00 | 0.030 | 0.213 | 18 935 | 5 | 0.14 s | **114** |
| **gradient** | oui | 0.05 | 1.00 | 0.050 | 0.240 | 544 666 | 19 | 1.40 s | 92 |
| **gradient** | oui | 0.10 | 0.12 | **0.0125** | 0.326 | 814 853 | 13 | 3.70 s | 102 |

**Deux lignes ne disent pas ce que leur `F` laisse croire.** « Barrière, `F = 0.10` » ne prend pas un
pas de 0.1 mais de **0.034** : elle est comparable à la ligne `F = 0.03`, pas à un pas trois fois plus
grand. Et « gradient, `F = 0.10` » prend un pas de **0.0125**, c'est-à-dire **sous `α*`** : ce run
avance moins loin que le Newton amorti tout seul, et demande quand même 102 diagrammes ensuite.

**Le gradient d'aire tient toutes ses promesses et perd quand même.** Il rend bien un état non
dégénéré (0.213 à 0.326), sans aucune pression sur la couronne, et il est moins cher par itération.
Mais le point qu'il produit est un **plus mauvais départ pour Newton** : 114, 92 et 102 diagrammes
contre 95, 87 et 89 pour la barrière — et à `F = 0.03` il est carrément **nuisible** (114 contre 105
sans rien faire).

La raison est instructive, et elle retourne l'argument : en visant `ν`, la barrière fait *une partie
du travail de Newton*. Sa « pression » sur la couronne n'est pas un défaut à corriger — c'est le
prix de l'avance qu'elle prend. Lisser le champ d'aires garantit la non-dégénérescence, ce qui était
bien le but affiché, mais ne rapproche de rien.

## 16.4 Le bilan net, et où est le régime utile

En comptant un diagramme pour `n` calculs de cellule, et les passes de rafraîchissement pour ce
qu'elles sont :

| | dépensés | économisés | **net** |
|---|---|---|---|
| barrière, `F = 0.03`, sans rafraîchissement | 1.3 | 13 | **+11.7** |
| barrière, `F = 0.05`, avec | 16 + 12 | 18 | −10 |
| gradient, `F = 0.03`, avec | 1 + 5 | −9 | −15 |

Le régime utile est donc **un amas, pas de rafraîchissement, objectif barrière** : là où le pincement
est un foyer unique, la réparation coûte un diagramme et en fait gagner treize. Le rafraîchissement
n'est nécessaire **que** dès qu'il y a plusieurs amas susceptibles de s'avaler l'un l'autre — et son
coût mange alors le gain. Rendre le rafraîchissement **local** (ne relire que le voisinage des amas
qui ont bougé, au lieu d'un diagramme complet) est le chantier qui déciderait de tout le reste.

**L'acquis** est ailleurs et il est solide : la décomposition en amas fait tomber le coût d'un
facteur qui n'a rien de marginal — un diagramme au lieu de quarante pour le même relèvement — parce
qu'elle supprime la dépendance en `n`. C'est la seule voie mesurée qui rende le relèvement
économiquement défendable.

---

# 17. LE SOLVEUR LINÉAIRE, OU 60 % DU TEMPS DANS UN SEUL FIL

## 17.1 Le constat : `--threads 8` n'achetait pas ce qu'on croyait

Un œil sur le CPU pendant un banc suffit à voir le problème : la machine passe le plus clair de son
temps **mono-thread**. Mesuré, `n = 10⁵`, `--threads 8`, huit cœurs physiques :

| | temps écoulé | `task-clock` | **CPU occupés** |
|---|---|---|---|
| `essai-limites` | 155,7 s | 410,3 s | **2,64 / 8** |
| relèvement | 144,1 s | 369,3 s | **2,56 / 8** |

Le diagramme, lui, est bien réparti. Le coupable était ailleurs, et la comptabilité interne ne le
montrait pas : la ligne du relèvement affichait `lin 0.00`, **un trou dans les compteurs et pas un
solveur gratuit** — `resout_releve` appelait `lin.resout` sans le chronométrer. Une fois le poste
rebranché, tout est visible :

```
newton CONVERGE : 51 it, 32.63 s  [ diag 8.24  asm 0.28  lin 13.27  lim 0.00 ]
```

Par étape de continuation, à `n = 10⁵`, relèvement : **`lin` = 84,4 s sur 142,0 s, soit 59 %** — et
68 % pour `essai-limites`. Un seul fil, tout du long.

## 17.2 Trois réglages jamais remis en question

La cause tenait en trois lignes de `Opts`, chacune posée pour de bonnes raisons devenues fausses.

**`solver = "chol"`.** `main_image` résolvait par Eigen `SimplicialLDLT` — une factorisation de
Cholesky creuse, **strictement séquentielle**, en `O( n^1.5 )`. Or `Lineaire.h` documente lui-même
l'alternative depuis le début : *« AMGCL […] se construit en parallèle (OpenMP). 4.9x sur le total à
n=1e6 contre Cholesky »*. Elle n'avait simplement jamais été activée ici.

**`amgvar = RS_GS`.** Le variant par défaut d'AMGCL était Ruge-Stuben + Gauss-Seidel, choisi sur le
**nuage de lignes**, où son choix de nœuds grossiers arête par arête divise les itérations par trois.
Sur une densité image il perd — et il est **doublement séquentiel** chez amgcl, le coarsening comme
le lisseur. L'agrégation lissée + `spai0` est parallèle des deux côtés.

**`tol = 1e-10`, jamais remplacée.** Une direction de Newton **amortie** ne demande pas dix chiffres :
la recherche linéaire vérifie le pas de toute façon. Mesure à `n = 2·10⁴` (`amg var 0`) : 8,27 s à
`1e-10`, 7,58 à `1e-6`, 7,06 à `1e-4`. On s'arrête à `1e-6` — à `1e-4` le compte de diagrammes
remonte (566 → 574 à `n = 10⁵`) et le gain de solveur commence à être repayé en géométrie.

## 17.3 Ce que ça vaut, sur trois décades

Nouveau défaut : `--solver amg --amg-var 0 --amg-tol 1e-6`.

| `n` | méthode | avant (`chol`) | après | gain | diagrammes |
|---|---|---|---|---|---|
| 5·10³ | `essai-limites` | 2,63 s | **1,67 s** | −36 % | 186 = 186 |
| 5·10³ | relèvement | 1,91 s | 1,88 s | — | 114 = 114 |
| 2·10⁴ | `essai-limites` | 12,52 s | **7,31 s** | −42 % | 353 → 357 |
| 2·10⁴ | relèvement | 10,98 s | **7,92 s** | −28 % | 218 = 218 |
| **10⁵** | `essai-limites` | 159,34 s | **90,72 s** | **−43 %** | 809 → 796 |
| **10⁵** | relèvement | 144,96 s | **93,37 s** | **−36 %** | 566 = 566 |

Le gain **croît avec `n`**, ce qui est la signature attendue : `O( n^1.5 )` séquentiel contre `O( n )`
parallèle. Le nombre de diagrammes ne bouge pas — la direction est la même — donc le gain est pur.
L'occupation passe de **2,32 à 5,63 CPU** sur 8 à `n = 10⁵`.

**Hors du cas facile.** Tout le réglage ci-dessus a été fait sur `--sans-trou`, et un solveur itératif
n'a pas les marges d'une factorisation directe : on vérifie.

| cas | `chol` | défaut | diagrammes |
|---|---|---|---|
| avec trous (zéros dans l'image), `2·10⁴` | 8,16 s | **5,12 s** | 170 = 170 |
| avec trous, `10⁵` | 115,37 s | **69,51 s** | 442 = 442 |
| diracs tirés selon `rho`, `2·10⁴` | 9,78 s | **6,99 s** | 204 = 204 |

Résultats **identiques au diagramme près** dans tous les cas, et plus rapides dans tous les cas. Les
cas à trous ne convergent pas — mais ni avant ni après : c'est le problème des zéros de l'image
(§ 11), indépendant du solveur.

## 17.4 Le multigrille maison, porté du GPU (`--solver mg`)

`gpu_des_familles/src/gpu/Amg2D.cuh` contient un multigrille écrit pour la carte, dont l'idée
centrale est belle et *a priori* aussi valable sur CPU :

> **L'agrégation est gratuite.** Les germes sont rangés dans l'ordre de l'arbre, qui est une courbe
> remplissante : des rangs consécutifs sont voisins dans le plan. Agréger, c'est `rang >> 2` — quatre
> germes par paquet, sans appariement, sans matching, sans compaction. Le niveau suivant s'agrège
> pareil (`a >> 2`) : la hiérarchie entière tient dans un décalage.

`src/solver/Multigrille.h` le porte fidèlement — même agrégation, même cycle en V (Jacobi amorti
`ω = 0.7`, `ν = 2`), même K-cycle, mêmes constantes. **Quatre écarts assumés**, chacun pour une
raison mesurée.

**La carte inverse est gratuite elle aussi, et on s'en sert.** Le paquet `a` contient exactement les
rangs `4a..4a+3`. On assemble donc le grossier **en balayant les lignes grossières**, un fil par
ligne, sans atomique et **sans le tri CUB** que le GPU doit faire (il émet un triplet par arête, trie,
réduit par clef). La restriction se fait de même par **ramassage** au lieu de dispersion.

**Jacobi à deux tampons.** Le noyau CUDA lit `x[ col ]` pendant que d'autres fils l'écrivent : c'est
un hybride Jacobi/Gauss-Seidel non déterministe. Sur la carte ça passe ; dans un préconditionneur de
CG c'est faux en droit — CG exige un opérateur **linéaire fixe**. Un vecteur de plus, et l'opérateur
redevient exactement symétrique.

**OpenMP et pas `parallel_for`.** `util/parallel.h` crée et joint ses fils à chaque appel, avec
épinglage : une cinquantaine de microsecondes. Le niveau le plus grossier en demande cent vingt par
cycle sur mille inconnues — la création coûterait cent fois le calcul.

**Le niveau le plus grossier est résolu, pas lissé** (§ 17.6).

### La jauge : un piège qui coûte tout

Premier essai : Newton **stagne immédiatement** — résidu `1.14e+01` inchangé après deux itérations,
31 reculs. La cause n'est pas dans le multigrille, elle est dans son contrat avec l'appelant.

La moyenne nulle est la bonne jauge **pour résoudre** : le laplacien a les constantes pour noyau,
`b = ν − a` est déjà de somme nulle, et projeter est symétrique là où rayer une ligne ne l'est pas.
Mais le reste du code suppose l'autre — `Newton.h` écrit

```cpp
for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
w2[ 0 ] = 0;                         // la jauge, imposee et non esperee
```

Avec `d[ 0 ] ≠ 0`, cette ligne **n'impose plus une jauge, elle mutile la direction sur une
composante**. Les deux jauges décrivent la même direction à une constante près — on translate donc
la solution en sortie. Une ligne, et le multigrille rend exactement les mêmes 114 diagrammes que
Cholesky.

**La leçon** : une jauge n'est pas un détail interne au solveur, c'est une **interface**. Quand deux
codes en supposent deux, l'un des deux échoue silencieusement — ici par stagnation, le symptôme le
plus difficile à rattacher à sa cause.

## 17.5 Le niveau le plus grossier était le poste dominant

Le portage tel quel coûtait **23,84 s** à `n = 2·10⁴`, contre 11,07 pour Cholesky. Trois mesures
indépendantes désignent le même coupable :

| | temps |
|---|---|
| `mg` défaut (K-cycle 2, 120 Jacobi au fond) | 23,84 s |
| `--mg-gros 40` (moins de balayages) | 15,94 s |
| `--mg-k 0` (moins de visites) | 16,01 s |
| `--mg-stop 200` (un fond plus petit) | 17,80 s |
| `--mg-gros 300` | 40,67 s |
| `--mg-stop 4000` | 54,72 s |
| `--mg-nu 1 / 3 / 4` | 26,0 / 24,1 / 24,0 |

Moins de balayages, moins de visites, ou un niveau plus petit donnent **le même −30 %** : c'est bien
le travail au fond du cycle qu'on paie, et pas le lissage. Le K-cycle visite le fond **quatre fois**
par application, à 120 balayages chacune — 480 boucles minuscules par itération de CG. Sur la carte
c'est un bon arbitrage : le parallélisme est massif et un lancement de noyau coûte 5 µs. Sur huit
cœurs, c'est le poste dominant.

## 17.6 Le fond résolu, et le K-cycle qui se périme

À mille inconnues, une factorisation de Cholesky creuse coûte quelques millisecondes **une fois par
hiérarchie** et vingt microsecondes par visite. On résout donc, au lieu de lisser — et la correction
grossière devient **exacte**.

| | temps |
|---|---|
| `mg` réglé, sans fond exact | 13,68 s |
| **`mg` fond exact, `--mg-k 0`** | **11,23 s** |
| `mg` fond exact, `--mg-k 1` | 11,68 s |
| `mg` fond exact, `--mg-k 2` | 12,47 s |

**Le K-cycle coûte désormais au lieu de rapporter**, et c'est cohérent : il n'était là que pour
compenser la faiblesse de la correction grossière de l'agrégation non lissée. Une fois cette
correction exacte, il ne reste que son prix. C'est le deuxième réglage de cette étude qui se périme
par l'arrivée d'un mécanisme en amont (cf. § 15.17).

## 17.7 Le verdict : le portage est juste, et c'est le matériel qui a changé d'avis

`n = 10⁵`, relèvement, **566 diagrammes pour tous** — les trois solveurs rendent la même direction.

| solveur | total | mise en forme | **hiérarchie** | résolution | it. CG | CPU / 8 |
|---|---|---|---|---|---|---|
| `chol` | 145,1 s | 3,6 | 25,8 | 55,9 | — | 2,3 |
| `amg0` | 103,8 s | 3,0 | 13,1 | 33,0 | 14 373 | 5,9 |
| `amg0 --amg-tol 1e-4` | **87,5 s** | 3,0 | 13,0 | 15,9 | 6 809 | 5,5 |
| `mg --mg-k 0` | 157,4 s | 0 | **1,77** | 100,7 | 47 146 | **6,8** |
| `mg k0 stop3000 tol1e-4` | 98,8 s | 0 | **2,37** | 40,4 | 18 607 | 6,0 |

**Le pari tient exactement là où il était annoncé, et échoue exactement là où l'agrégation non lissée
est faible.** La hiérarchie du portage coûte **2,4 s contre 13,0 s** pour AMGCL — l'agrégation par
`rang >> 2` est bien gratuite, 5,5× moins cher, et c'est son seul avantage. Mais il lui faut
**18 607 itérations de CG contre 6 809**, soit 2,7×, et sur CPU ce facteur écrase l'économie de
montée. Aucun réglage du fond n'y change rien : ce n'est pas le cycle, c'est le **taux de
convergence** de l'agrégation non lissée.

Sur carte l'arbitrage s'inverse : les itérations y sont quasi gratuites, le tri et la réduction de la
montée coûtent, et le produit triple est exactement ce qu'un GPU n'aime pas — le chemin lissé y
existe (`AMG_LISSE`, via `cusparseSpGEMM`) mais reste **éteint**. **Le portage est fidèle ; c'est le
matériel qui a changé d'avis.** Ce qui manquait sur CPU était donc identifié et chiffré, et la suite
le corrige (§ 17.10).

## 17.10 La prolongation lissée, portée sur CPU

`P̂ = ( I − ω D⁻¹ A ) P₀`, soit, ligne par ligne :

    P̂[ i ][ a ] = ( 1 − ω ) [ mᵢ = a ] + ( ω / diaᵢ ) Σ_( j ≠ i, mⱼ = a ) c_ij

Toutes les entrées sont positives pour `ω ≤ 1`, et **les lignes somment à un** : le vecteur constant
reste exactement dans l'image de `P̂`, donc le noyau du laplacien est représenté à tous les niveaux et
`A_c = P̂ᵀ A P̂` est encore un laplacien.

**La carte inverse gratuite sert une troisième fois.** Le produit triple se fait en deux passes à
accumulateur dense, une ligne par fil : `AP = A P̂` par ligne fine, puis `A_c = P̂ᵀ ( AP )` par ligne
grossière. Et `P̂ᵀ` **s'obtient sans passe de transposition** : `P̂[ i ][ a ] ≠ 0` exige `mᵢ = a` ou un
voisin de `i` dans le paquet `a`, donc la ligne `a` de `P̂ᵀ` est portée par les membres du paquet et
leurs voisins — que `Sa .. Sa+S−1` donne immédiatement.

**Le résultat est net sur ce qu'on visait** : à `n = 2·10⁴`, les itérations de CG tombent de **8 641
à 3 051**, c'est-à-dire *en dessous* des 3 320 d'AMGCL. Le diagnostic du § 17.7 était le bon.

### Ce que le lissage coûte, et les trois remèdes

Il déplace le problème sur la montée : 0,29 → 6,97 s. La cause se lit dans la trace, en non-nuls par
ligne niveau par niveau (`--mg-trace`) :

| `n = 2·10⁴` | remplissage | complexité |
|---|---|---|
| brut | 6,0 → 8,0 → 7,9 → 7,2 | 1,44 |
| lissé, sans rien | 6,0 → 27,1 → 115,4 → **278,1** | 4,08 |
| lissé, `--mg-tronque 0.1` | 6,0 → 21,7 → 56,0 → 105,7 | 2,78 |
| lissé, `--mg-tronque 0.2` | 6,0 → 18,2 → 31,0 → 46,2 | 2,21 |

**La troncature** (`tronque`) jette les entrées de `P̂` sous une fraction du maximum de la ligne, puis
**renormalise pour que la ligne somme à un** — la renormalisation n'est pas cosmétique, c'est elle qui
garde le vecteur constant dans l'image. Monotone jusqu'à 0,35 sans que les itérations bougent.

**La hiérarchie gardée** (`refaire`) : entre deux itérations de Newton la hessienne change mais son
graphe bouge à peine, et un *préconditionneur* n'a pas besoin d'être exact — seule la matrice que voit
le CG doit l'être, et on la rafraîchit. AMGCL ne permet pas cette dissociation (son `make_solver` lie
matrice et hiérarchie) ; posséder le solveur, si. `112,3 → 95,9 s` à `n = 10⁵`, optimum à 4.

**Le filtre de force est un échec, et c'est un résultat.** Le remède standard — ne lisser que le long
des `c_ij ≥ θ·max_k c_ik` — ne change **rien** à la valeur classique θ = 0,08 : complexité 2,69 contre
2,67. La raison est structurelle : dans un graphe de Laguerre les `c_ij = |facette| / (2|pᵢ−pⱼ|)` sont
toutes du même ordre, **il n'y a pas de connexion faible à jeter**. La densification vient du *motif*
(`P̂` a `1 + deg` entrées, `deg ≈ 6` partout), pas d'un contraste de valeurs. Le filtre de force est
l'outil des problèmes anisotropes ; sur un diagramme de puissance il est hors sujet. Défaut : `0`.

## 17.11 La taille du paquet, et ce que l'arbre autorise

Le vrai levier était ailleurs. La complexité est dominée par **le niveau 1** : à `n = 3·10⁵`,
`75 000 × 18,6 = 1,4 M` non-nuls contre `1,8 M` au niveau fin, soit 0,78 à lui seul. Des paquets plus
gros donnent moins de lignes grossières *et* moins de niveaux.

La carte en prend quatre, parce que quatre est ce qu'un noyau CUDA aime — et parce qu'avec une courbe
de Morton, quatre est ce qu'on a en 2D. Mais **`AaBsp.h` ne fait pas du Morton** : « coupes MÉDIANES
sur l'axe le plus long ». Une fenêtre alignée de `2^k` rangs consécutifs est donc *exactement un
sous-arbre* — localité parfaite — et sa boîte a été coupée `k` fois sur son côté le plus long : `k`
pair donne une boîte carrée, `k` impair une boîte 2:1, ce qui pour de l'agrégation va très bien.
**Toutes les puissances de deux sont donc disponibles, pas seulement les puissances de quatre.**

Ce qui reste interdit, ce sont les tailles qui ne sont pas des puissances de deux : une fenêtre de
neuf rangs n'est alignée sur aucune frontière de sous-arbre, et certains paquets enjamberaient une
coupe de haut niveau — deux moitiés du domaine dans le même agrégat.

| paquet | `n = 2·10⁴` | `n = 10⁵` | complexité |
|---|---|---|---|
| 4 | 8,84 s | 96,75 s | 2,25 |
| **8** | **7,38 s** | 97,99 s | **1,34** |
| 16 | 8,49 s | 105,70 s | 1,13 |
| 64 | 11,09 s | — | 1,02 |
| AMGCL | 7,31 s | 92,14 s | — |

La courbe est en U et son fond est plat entre 4 et 16 : la complexité tombe quand le paquet grossit,
les itérations montent, et les deux se croisent **vers huit** — pile là où AMGCL se place avec son
ensemble indépendant maximal (sept à neuf), et là où une courbe de Morton ne nous aurait pas laissés
aller. Au-delà de seize l'espace grossier devient trop pauvre et rien ne rattrape : 981 s à
`n = 3·10⁵` contre 757 à paquet 4.

**`spai0` plutôt que Jacobi amorti**, enfin : `m_i = A_ii / Σ_j A_ij²` est la meilleure approximation
diagonale de `A⁻¹` au sens de Frobenius, coûte exactement un balayage de Jacobi, et son amortissement
se règle tout seul ligne par ligne au lieu d'un `ω` global deviné — 8,98 s contre 9,86 à paquets
égaux. Dans les deux cas le coefficient est **précalculé par ligne**, ce qui retire une division de la
boucle la plus chaude du cycle.

## 17.12 Où en est le solveur maison

| | `2·10⁴` | `10⁵` | `3·10⁵` |
|---|---|---|---|
| `chol` | 11,1 s | 145,4 s | — |
| **AMGCL `var 0`** | **7,31 s** | **92,1 s** | **667,6 s** |
| `mg` maison, réglé | 7,38 s | 96,8 s | 756,7 s |

**De 2,2× plus lent que Cholesky à l'égalité avec AMGCL à `2·10⁴`, et à 5 % à `10⁵`.** Il restait
13 % derrière à `3·10⁵`, et c'est le même poste qui l'expliquait depuis le début : à taux de
convergence égal, notre cycle brasse plus de non-nuls. Trois ajouts ont fait le reste du chemin
(§ 17.13).

## 17.13 Ce qu'on prend à la littérature : le recyclage, Chebyshev, et un échec

### La hiérarchie gardée ne paie que si on possède le solveur

`Mg` gardait déjà sa hiérarchie plusieurs résolutions (§ 17.10). AMGCL expose la même dissociation —
`make_solver::operator()( A, rhs, x )` résout avec une matrice **neuve** contre la hiérarchie déjà
montée. On l'a branchée (`--amg-refaire`) et mesurée à `n = 10⁵` :

| | hiérarchie | résolution | it. CG | total |
|---|---|---|---|---|
| `amg --amg-refaire 1` | 13,3 s | 21,9 s | **9 369** | **93,9 s** |
| `amg --amg-refaire 4` | 3,4 s | 41,1 s | **16 764** *(+79 %)* | 103,0 s |
| `mg --mg-refaire 4` | 3,5 s | 39,3 s | 16 944 | 98,1 s |

**Ça échoue sur AMGCL et ça marche chez nous, pour une raison qu'on peut nommer** : `Mg::rebranche`
**recalcule les coefficients du lisseur au niveau fin** avec les nouvelles valeurs, et n'y perd que
11 % d'itérations ; AMGCL garde son `spai0` du niveau fin construit sur l'ancienne matrice, et paie
79 %. Il n'existe pas d'API amgcl pour rafraîchir ce seul niveau. Défaut `--amg-refaire 1`, l'option
reste pour vérifier.

C'est l'argument le plus net en faveur de posséder le solveur : la dissociation entre *la matrice que
voit le CG* et *celle qui a servi au préconditionneur* n'est exploitable que si on contrôle le
rafraîchissement, et une bibliothèque ne l'expose pas.

### Le recyclage de sous-espace : −19 %, et il sature à deux vecteurs

On ne résout pas un système mais des centaines qui se ressemblent. On garde les `k` dernières
solutions dans `U` et on démarre sur la projection de Galerkin `x₀ = U(UᵀAU)⁻¹Uᵀb` — la meilleure
approximation dans `span(U)` au sens de l'énergie. Le résidu `r₀ = b − (AU)y` sort du calcul déjà
fait, donc le prix est exactement `k` produits matrice-vecteur (Parks et al., *Recycling Krylov
Subspaces for Sequences of Linear Systems*).

| `n = 2·10⁴` | it. CG |
|---|---|
| `--mg-recycle 0` | 5 117 |
| **`--mg-recycle 2`** | **4 337** *(−15 %)* |
| `--mg-recycle 4 / 8 / 16` | 4 332 / 4 334 / 4 336 |

**Il sature à deux**, ce qui est le résultat intéressant : la solution précédente porte à elle seule
presque toute l'information, et les directions de Newton successives n'engendrent utilement qu'un ou
deux degrés de liberté. On s'arrête donc à 2, où le surcoût est négligeable.

### Chebyshev, et une passe de trop qui coûtait tout le gain

Un polynôme de degré `nu` en `M⁻¹A`, minimisant le maximum sur `[λ_max/r, λ_max]` — la partie du
spectre que le grossier ne corrige pas. Uniquement des produits matrice-vecteur, **donc le même code
sur les deux machines**, ce que Gauss-Seidel n'est pas. `λ_max` ne se devine pas, il se **borne** par
Gershgorin sur `M⁻¹A` : exact, une passe sur les arêtes, et deux pour un laplacien pur.

Premier essai : −8 % d'itérations, mais +12 % de coût par itération — un lavage. La cause était mon
implémentation, pas la méthode : j'écrivais `A y` dans un tampon puis refaisais une passe pour mettre
à jour `r` et la direction. Sur un cycle limité par la bande passante, cette passe coûtait exactement
ce que le polynôme gagnait. Le double tampon **reste nécessaire** — le produit de la ligne `i` lit
`y` chez les voisins, on ne peut pas écrire dedans — mais la seconde passe non : on lit `y`, on met
`r` à jour sur place, on écrit la direction suivante dans `z`, et on échange.

| `n = 2·10⁴` | it. CG | résolution |
|---|---|---|
| Chebyshev `nu2 cheb10`, deux passes | 4 687 | 2,19 s |
| Chebyshev `nu2 cheb10`, **fusionné** | **4 043** | **1,83 s** |

### Ce que les trois donnent ensemble

| `n = 10⁵` | it. CG | hiérarchie | résolution | **total** |
|---|---|---|---|---|
| `amg` (témoin) | 9 369 | 13,32 | 21,71 | 93,04 s |
| `mg`, sans recyclage | 16 941 | 3,45 | 39,62 | 98,17 s |
| `mg` + `recycle 2` | 13 641 | 3,48 | 32,14 | 90,42 s |
| **`mg` + Chebyshev + `recycle 2`** | **10 732** | 3,50 | 31,61 | **90,19 s** |

Les deux briques se composent : le recyclage retire 19 % des itérations, Chebyshev 21 % de plus, et
la montée gratuite fait le reste. Sur le **temps**, Chebyshev et `spai0` sont à égalité (90,19 contre
90,42) ; ce qui le fait choisir par défaut est ailleurs — **−21 % d'itérations pour le même temps**,
donc de la marge quand le problème durcit, et un lisseur qui se porte à l'identique sur carte.

## 17.14 Le solveur maison devient le défaut

| | `5·10³` | `2·10⁴` | `2·10⁴` trous | `2·10⁴` diracs ρ | `10⁵` |
|---|---|---|---|---|---|
| `chol` | 1,91 | 10,99 | — | — | — |
| `amg var 0` | 1,70 | **7,26** | 5,63 | 7,05 | 92,57 |
| **`mg` maison** | **1,54** | 7,44 | **4,48** | **6,47** | **90,43** |

Quatre cas sur cinq, diagrammes identiques partout, et le cinquième est à 2,5 % — sous le bruit de
±8 % qu'on mesure en rejouant la même commande. `--solver mg` devient le défaut.

**Avec une réserve honnête** : nos réglages (`agreg 8`, `tronque 0,2`, `cheb 10`, `nu 3`, `recycle 2`)
ont été calés sur *ce* cas d'usage, alors qu'AMGCL ne l'a pas été. Il reste le témoin, à un drapeau,
et c'est à ce titre qu'il doit rester dans le code. Sur carte, tout sera à re-mesurer : c'est
précisément l'arbitrage matériel qui change, et c'est ce que toute cette section raconte.

## 17.15 Et en 3D (`newton --3d --solver mg`)

Le multigrille est branché dans `main_newton`, qui déroule les deux dimensions. Le 3D n'est pas le
2D avec une coordonnée de plus : **le graphe de Laguerre y a 15,1 non-nuls par ligne contre 6,0**,
et il n'est plus planaire.

### Cholesky n'est pas une option en 3D

| `n = 10⁵`, 3D uniforme | partie linéaire |
|---|---|
| `chol` (Eigen LDLT, AMD) | **> 600 s** (plafond atteint) |
| `amg` | 1,53 s |
| `mg` | **0,83 s** |

Plus de **400×**, et c'est structurel : la dissection emboîtée d'un maillage 3D coûte `O(n²)` de
flops contre `O(n^1,5)` en 2D. En 2D `chol` restait un témoin utilisable ; en 3D il ne finit pas.

### Le paquet optimal est 8 dans les deux dimensions — mais il faut régler le lissage avec

Partie linéaire (hiérarchie + résolution), 3D uniforme :

| | `n = 10⁵` it / s | `n = 5·10⁵` it / s |
|---|---|---|
| `amg` (témoin) | 138 / 1,53 | 129 / 7,32 |
| `a16 nu3` | 151 / 1,00 | 156 / 7,16 |
| `a16 nu2` | 168 / 0,85 | 176 / 6,40 |
| `a16 nu1` | 239 / 0,83 | 250 / 5,64 |
| **`a8 nu1`** | 200 / **0,84** | 165 / **4,81** |
| `a8 nu2` | 144 / 0,95 | 120 / 5,30 |
| `a32 nu1` | 340 / 0,94 | 288 / 6,14 |

**−46 % sur AMGCL à `10⁵`, −34 % à `5·10⁵`.**

Et une leçon de méthode qui a failli me faire écrire l'inverse. En balayant la taille de paquet
**à `nu = 3`** — le réglage 2D — l'optimum 3D semblait être 16, et j'ai commencé à l'écrire. C'était
un artefact : **un cycle trop lissé force à grossir les paquets pour rester payable**. À `nu = 1`
l'optimum redescend à 8, c'est-à-dire au bloc 2×2×2, celui que la géométrie suggérait. Les deux
réglages ne sont pas séparables, et les balayer l'un après l'autre donne le mauvais point.

Pourquoi `nu = 1` en 3D et pas en 2D : avec quinze voisins au lieu de six, **un seul passage de
Jacobi propage déjà l'information bien plus loin**. Passer de `nu 3` à `nu 1` coûte **+69 %
d'itérations en 2D mais seulement +32 % en 3D** — et en 3D ces 32 % sont largement repayés par le
tiers de bande passante économisé. En 2D, `nu` est **plat** entre 1 et 3 (88,6 / 88,0 / 89,6 s à
`n = 10⁵`, sous le bruit de ±8 %) : on y garde 3, qui fait le moins d'itérations à temps égal.

`main_newton` met donc `nu = 1` en 3D et laisse le défaut de `Multigrille.h` en 2D. Chebyshev gagne
dans les deux dimensions (3D, `n = 5·10⁵` : 176 it / 5,61 s contre 191 / 6,48 pour `spai0`).

### Et en 2D uniforme, il perd — ce qui dit ce qui décide vraiment

Le réglage 2D venait entièrement du cas **image** de `main_image`. Sur le **nuage uniforme** de
`main_newton`, il ne tient pas. `n = 5·10⁵`, 2D :

| | it. CG | hiérarchie | résolution | TOTAL |
|---|---|---|---|---|
| **`amg`** | 258 | 1,655 | 4,029 | **8,03 s** |
| `a8 nu3` *(le défaut image)* | 310 | 0,806 | 6,780 | 9,76 s |
| `a8 nu1` | 495 | 0,803 | 5,624 | 8,56 s |
| `a4 nu2` | 233 | 1,510 | 4,913 | 8,68 s |
| `a16 nu1` | 935 | 0,567 | 9,799 | 12,50 s |

**Aucun réglage ne renverse** : le meilleur reste à +6,5 %. Notre cycle est pourtant *moins cher par
itération* (11,4 ms contre 15,6) — il en faut simplement deux fois plus.

**Ce n'est donc pas « mg perd en 2D ».** Dans `main_image`, à dimension égale, il gagne. La
différence n'est pas la dimension, c'est le **régime** : la continuation sous image enchaîne 133 à
323 itérations de Newton, donc des centaines de systèmes voisins, et c'est de ça que le recyclage
vit. Le cas uniforme converge en une poignée d'itérations : `U` ne se remplit pas, et il ne reste
que la qualité brute du préconditionneur, où AMGCL est devant.

`main_newton` prend donc `--solver auto` : **`mg` en 3D, `amg` en 2D**. `main_image` garde `mg`.
Un défaut qui dépend du cas est moins élégant qu'un défaut unique, mais c'est ce que trois bancs
disent, et forcer l'uniformité coûterait 6 à 20 % quelque part.

### Ce que ce banc ne dit PAS : le recyclage

En 3D uniforme, le recyclage ne change rien (151 itérations avec, 155 sans). Ce n'est **pas** un
résultat sur la méthode, c'est un banc inadapté à la question : ce cas converge en **six itérations
de Newton et neuf diagrammes**, donc `U` n'a pas le temps de se remplir et les systèmes successifs
ne se ressemblent pas assez. Le recyclage vit du régime « longue séquence de systèmes voisins »,
que le cas image 2D fournit (133 à 323 itérations de Newton) et que celui-ci ne fournit pas. Pour
le juger en 3D il faudrait une continuation 3D — ce qui n'existe pas encore ici.

## 17.8 Le critère d'arrêt à 1 % : gratuit, et sans effet

Proposition : arrêter Newton sur `max_i | A_i / ν_i − 1 | < 1 %` au lieu de `1e-6`. C'est exactement
la quantité déjà affichée (`max|a-nu|/nu`) et déjà comparée à `NewtonOptions::tol` — donc rien à
coder, `--newton-tol 1e-2`.

| `n = 2·10⁴`, relèvement | diagrammes | temps |
|---|---|---|
| `chol` | 218 | 11,09 s |
| `chol --newton-tol 1e-2` | 211 | 10,84 s |
| `amg0` | 218 | 8,25 s |
| `amg0 --newton-tol 1e-2` | 211 | 8,39 s |

Sept diagrammes de moins sur 218 (−3 %), et **aucun gain de temps mesurable**. La raison est
mécanique : les itérations supprimées sont les **dernières** de chaque étape, celles où le résidu est
déjà petit. Le diagramme y coûte le prix normal, mais le système linéaire y converge en quelques
itérations de CG — on coupe la queue la moins chère. Newton étant quadratique à l'arrivée, il n'y a
que deux itérations entre 1 % et `1e-6`.

Le réglage reste disponible pour qui veut un plan approché ; il n'y a pas de raison d'en faire le
défaut, puisqu'il dégrade la précision du plan sans rien rendre en échange.

## 17.9 Ce qui reste séquentiel

Une fois le solveur linéaire parallèle, le poste séquentiel suivant est **la recherche `( λ, β )` du
placement harmonique** : 4,62 s sur les 8,35 s de réparation à la dernière étape de `n = 10⁵` (le
reste : géométrie 0,57, assemblage 0,69, algèbre 0,52, construction des amas 0,92). C'est un balayage
géométrique suivi d'une bissection, amas par amas, sur 628 amas — et les amas sont indépendants.

Le chantier suivant est donc **paralléliser sur les amas**, ce qui demande de supprimer le `w`
partagé : les coupeurs de deux amas se recouvrent. La bonne forme est un `w` gelé en lecture plus, par
amas, les poids de ses propres inconnues — ce qui rend au passage la réparation **déterministe**, là
où l'ordre de traitement la fait aujourd'hui dépendre de l'amas précédent.

Un dernier détail mesuré au passage : `perf` comptait **10 % des cycles dans `libgomp`**, les fils qui
attendent sur les barrières de boucles OpenMP portant sur trente cellules. Une clause `if( M >= 256 )`
sur la recherche linéaire locale rend la boucle séquentielle quand elle est courte.

---

# 18. LE TEMPS DE COMPILATION : CE N'ÉTAIT PAS LA TAILLE DU FICHIER

`main_image.cpp` mettait **476 s et 4,0 Go** à compiler — au point que `-g` le faisait tuer par le
plafond mémoire de `job`, et qu'on avait renoncé aux numéros de ligne dans `perf` pour ça (§ 17).
Le réflexe est d'accuser les 3 891 lignes et de les éclater en plusieurs fichiers. **C'est le
mauvais coupable** : `main_ecrasement.cpp` fait 787 lignes et mettait 86 s, soit à peu près le même
temps par ligne. Ce qui coûte, c'est le nombre de fois que ces lignes sont **instanciées**.

Elles l'étaient **quarante-huit fois** :

| axe | valeurs | qui décide |
|---|---|---|
| flottant du noyau `TK` | `float`, `double` | `--kernel`, à l'exécution |
| capacité de cellule `MaxNv` | 64, 128, 256, 512 | `--maxnv`, à l'exécution |
| flottant de la mesure `TA` | `float`, `double` | `--acc`, à l'exécution |
| solveur linéaire `Lin` | `Cholesky`, `Mg`, `Amg` | `--solver`, à l'exécution |

**Les quatre sont choisis à l'exécution, et les trois quarts des combinaisons ne servent jamais.**
Deux d'entre elles n'avaient aucune raison d'être des paramètres de template.

## 18.1 Le solveur linéaire : une interface virtuelle (÷3)

`Lin` était un paramètre de template *par habitude*. Or un solveur linéaire est appelé **une fois
par itération de Newton**, sur un système à `n` inconnues : l'appel dure des millisecondes et il n'y
a rien à y inliner. `solver/Lineaire.h` déclare maintenant

```cpp
struct Lineaire {
    StatsLin st;
    virtual const char *nom() const = 0;
    virtual bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d ) = 0;
    virtual bool sait_encore() const { return false; }
    virtual void resout_encore( const std::vector<TF> &b, std::vector<TF> &d ) {}
    virtual void ordre( const std::int32_t *ids, SI nb ) {}
};
```

et `Amg`, `Cholesky`, `Mg` en dérivent. Les **réglages** restent sur le type concret — ils se posent
dans `main` avant que le template ne commence, donc ils n'ont jamais eu besoin d'être visibles
dedans. Deux `if constexpr ( requires { ... } )` disparaissent au passage : `sait_encore()` dit à
l'exécution ce que le `requires` devinait à la compilation, et le dit mieux (Cholesky *a* une
factorisation, mais seulement après la première résolution).

## 18.2 La capacité de cellule : une seule compilée (÷4 en 2D)

`MaxNv` est la taille d'un tableau dans la frame : il ne peut pas cesser d'être un paramètre de
template. Mais rien n'oblige à compiler toute l'échelle. Par défaut, `bench/Dispatch.h` en compile
**une seule en 2D (64)** et **deux en 3D (128 et 256)** — l'asymétrie n'est pas un compromis, elle
est mesurée : une cellule de Laguerre plane a six voisins en moyenne et le banc n'a jamais vu de
débordement, alors qu'en 3D le débordement arrive dès `n = 2·10⁴` sur l'uniforme et que le banc
lui-même répond « relancer avec `--maxnv 256` ». Retirer 256 rendrait son propre conseil
inapplicable.

`-DSF_NV_TOUS` rend l'échelle entière. Et demander une capacité non compilée **le dit** au lieu de
retomber en silence sur une cellule plus petite — ce qui ferait déborder les cellules sans autre
trace qu'un compteur qu'on ne regarde pas toujours.

## 18.3 Ce que ça donne

Les deux axes restants sont `--kernel` et `--acc`, c'est-à-dire **exactement les deux flottants de
l'étude de simple précision** : ils doivent vivre dans le même binaire pour qu'on puisse les
comparer sur les mêmes cellules. `-DSF_TK_UN` n'en garde qu'un pour qui n'en a pas besoin.

| cible | avant | après | mémoire |
|---|---|---|---|
| **image** | **476,1 s** | **56,4 s** | 4 003 → 1 117 Mo |
| newton | 117,7 | 29,5 | 1 528 → 861 |
| ecrasement | 86,1 | 27,2 | 1 584 → 895 |
| densite | 81,4 | 24,2 | 1 338 → 859 |
| homotopie | 58,1 | 20,2 | 1 249 → 843 |
| glissement | 56,6 | 19,9 | 1 173 → 840 |
| multiechelle | 45,0 | 20,6 | 1 356 → 1 100 |
| check | 17,4 | 10,2 | 545 → 513 |
| diagramme | 12,8 | 7,1 | 502 → 480 |
| memo | 8,3 | 6,0 | 522 → 460 |
| **total séquentiel** | **959 s** | **221 s** | |

Build complet en parallèle : **62 s**. Résultats **identiques** — `check`, `image --check`, et les
solves de `newton` 2D/3D et `image` rendent les mêmes comptes d'itérations, de diagrammes et les
mêmes résidus, au chiffre près.

**Ce qui reste sur la table**, et qui n'est plus le levier : éclater `main_image.cpp` en plusieurs
unités de traduction. Ça demanderait des instanciations explicites — fragiles — pour un fichier qui
compile maintenant en moins d'une minute. Le pic mémoire ayant été divisé par 3,6, `-g` redevient
d'ailleurs envisageable si on veut les numéros de ligne dans `perf`.


---

# 19. LA SIMPLE PRÉCISION : CE QUI CASSE, ET CE QUI SE RÉPARE (`fp32`)

Le banc a trois flottants, et ils ne jouent pas le même rôle : `TK` celui de **la géométrie**
(`--kernel`), `TA` celui de **la mesure image** (`--acc`), et `TF = double` partout ailleurs
(l'arbre, les poids, la hessienne, le solveur). En `float`, le § 4 rapportait deux échecs francs et
le § 12.7 une erreur de **15 % sur une facette**. On y revient pour répondre à une question
précise : *qu'est-ce qui casse exactement, et est-ce réparable ?*

La cible `fp32` construit **le même nuage, les mêmes poids, le même moteur, le même élagage, deux
fois — seul `TK` change** — et compare cellule par cellule.

## 19.1 Un seul mécanisme, trois fois

Le plan qui sépare deux germes est rendu sous la forme `dx·x + dy·y = off`, avec
`off = ½(|p_j|² − |p_0|²) + ½(w_0 − w_j)`, et le noyau l'évalue en chaque sommet :
`s = dx·vx + dy·vy − off`. **Deux annulations s'y cachent, et elles n'ont pas la même cause.**

* **Le poids.** `w_0 − w_j` est la différence de deux poids déjà arrondis : erreur `eps·|w|` sur
  `off`, donc un plan déplacé de `eps·|w| / (2h)` le long de sa normale.
* **La position.** `dx` vaut `h`, `vx` vaut `1`, donc `dx·vx` vaut `h` — alors que `s` doit valoir
  `h²`. On perd `log(1/h)` chiffres à chaque coupe.

Rapporté à la taille de cellule `h = n^(−1/D)` :

```
    erreur relative  ~  eps·|w| / h²  +  eps·|p| / h
                     =  eps·|w|·n^(2/D)  +  eps·n^(1/D)
```

**Les deux croissent avec `n`, et la 2D est le cas dur à `n` fixé** — `h² = 1/n` contre `n^(−2/3)`.
C'est l'intuition qu'il fallait vérifier ; le facteur entre les deux dimensions est `n^(1/D)`.

Et c'est **le même mécanisme que `S[j][i] − sref` dans la mesure image** (§ 12.7) : partout où une
quantité **grande dans l'absolu** sert à produire une quantité **à l'échelle `h`**, on perd
`log(grand/petit)` chiffres. Le remède est toujours le même : **porter la différence, pas la
valeur.**

## 19.2 Le terme de position, vérifié sur Voronoï (où il n'y a pas de poids)

Écart de masse médian `float` / `double`, rapporté à la masse **moyenne** `1/n` :

| `n` | 2D mesuré | 2D modèle `eps·√n` | 3D mesuré | 3D modèle `eps·n^(1/3)` |
|---|---|---|---|---|
| 10³ | 1.09e−06 | 1.9e−06 | 4.07e−07 | 6.0e−07 |
| 10⁴ | 3.49e−06 | 6.0e−06 | 8.66e−07 | 1.3e−06 |
| 10⁵ | 1.13e−05 | 1.9e−05 | 1.89e−06 | 2.8e−06 |
| 10⁶ | 3.54e−05 | 6.0e−05 | 4.09e−06 | 6.0e−06 |

**×3.16 par décade en 2D (= `√n`), ×2.15 en 3D (= `n^(1/3)`)**, préfacteur constant 0.57 et 0.68.
Le modèle n'est pas une analogie, c'est la loi. À `n = 10⁶`, **la 2D perd huit fois plus que la 3D**.

## 19.3 Le terme de poids : quatre ordres de grandeur au point de fonctionnement

Les poids aléatoires de `--weights W` valent `W·h²`, donc le terme de poids y vaut `eps·W` : on ne
le voit pas tant que `W` est petit, et au-delà il **vide** les cellules au lieu de les fausser (la
médiane tombe à zéro : la plupart sont vides des deux côtés). Le balayage en `W` ne répond donc pas
à la question. **Il faut un point de fonctionnement réel** — les fichiers `cases/*_equal.txt`, dont
les poids *résolvent* déjà les masses égales.

`|w|max` y vaut **0.13 à 0.18** pour un `h²` de `10⁻⁵`. Quatre ordres de grandeur : c'est tout le
problème, et c'est invisible sur un nuage de jouet.

**Testé et éliminé : centrer les poids.** `w ← w − moyenne(w)` est une invariance *exacte* du
diagramme de puissance, donc c'était le remède à un sou. Il ne change rien (2.01e−03 → 2.01e−03) :
ces champs de poids sont déjà à moyenne quasi nulle, et ce n'est pas le niveau qui nuit, c'est
l'amplitude.

## 19.4 Ce qu'on a fait : deux réparations, le même principe

**(a) Porter la différence des poids.** Les poids ne sont grands que *dans l'absolu*. Pour une
paire qui partage vraiment une facette, le bissecteur tombe *dans* les deux cellules, donc
`|w_0 − w_j| ≲ 2|d|·h ~ h²` : **la différence qui compte est du même ordre que le terme
géométrique.** L'arbre range germes et poids en `double` de toute façon — il suffisait de ne pas
arrondir avant de soustraire.

`cell/Plan.h` est désormais **le seul endroit où un plan bissecteur se construit** (il y en avait
quatre copies : les deux fournisseurs BSP, le balayage témoin, le fournisseur en alpha), et ses
entrées sont en `TF` pour une sortie en `TK`. Le majorant affine de l'élagage suit : `Boite2::cb`
porte `w_0 − b`, calculé en double, au lieu de `b`.

**(b) La cellule vit dans le repère du germe.** `Atelier::lx / ly` et `Cellule3::vx / vy / vz` sont
**relatifs au germe**, et l'offset du plan prend sa forme locale — qui est plus simple :

```
    off_local = off − d·p₀ = |d|²/2 + (w₀ − w_j)/2
```

Les deux membres de `s = d·v − off` sont alors d'ordre `h²`, l'ordre même du résultat. Trois
bénéfices en prime : l'élagage n'a plus de `p₀` à soustraire (il est à l'origine, deux opérations
SIMD de moins par sommet), l'aire par la formule du lacet cesse d'être une différence de termes
d'ordre 1, et `Atelier::x(i)` rend l'absolu par une somme **en double**, donc avec l'erreur `eps·h`
du repère local au lieu de `eps·|p|`.

Un détail qui aurait faussé la suite : **le balayage témoin a reçu le même traitement**. Nourri en
`TK`, il serait devenu quatre chiffres moins précis que ce qu'il teste, et `check --kernel float`
aurait accusé l'élagage de ses propres arrondis.

*Piège payé au passage* : déplacer les sommets sans déplacer le plan donne une somme des mesures de
0.318 et 306 cellules fausses sur 5000. `check` l'a vu immédiatement — c'est exactement ce pour
quoi il existe.

## 19.5 Ce que ça donne

Écart `float` / `double`, mêmes germes, mêmes poids, en trois temps :

| cas | `n` | \|w\|max | | départ | + différence des poids | + repère local |
|---|---|---|---|---|---|---|
| lignes σ=0.005 (2D) | 10⁵ | 0.130 | masse médiane | 2.01e−03 | 1.37e−05 | **3.76e−06** |
| | | | masse p99 | 2.19e−01 | 1.89e−04 | **4.87e−05** |
| | | | arêtes en désaccord | 2978 | 39 | **6** |
| lignes σ=0.1 (2D) | 10⁵ | 0.145 | masse médiane | 8.44e−04 | 1.08e−05 | **3.22e−06** |
| lignes σ=0.005 (2D) | 2·10³ | 0.129 | masse médiane | 3.91e−05 | 1.89e−06 | **5.32e−07** |
| plans σ=0.02 (3D) | 10⁵ | 0.181 | masse médiane | 3.49e−05 | 1.77e−06 | **4.80e−07** |

**×530 sur la médiane et ×4500 sur le p99** dans le pire cas. La vérification qui compte, pour la
première réparation : après elle, la valeur obtenue (1.37e−05 en 2D à `n = 10⁵`) est celle du
**Voronoï au même `n`** (1.13e−05). Le terme de poids avait bien disparu, exactement comme prévu.

### Dans la boucle de Newton, `n = 10⁵`, 2D

| | avant | après |
|---|---|---|
| `double`, uniforme | CONVERGE 2.04e−09, 6 it, 8 diag | CONVERGE **2.58e−10**, 6 it, 8 diag |
| `double`, lignes | STAGNATION 2.35e−06, 116 diag | STAGNATION 2.35e−06, 116 diag |
| ... le même, nuage dédupliqué (§ 23.7) | — | CONVERGE **2.00e−07**, **78 diag** |
| `float`, uniforme | STAGNATION 3.67e−03, 133 diag | STAGNATION **5.40e−04**, **91 diag** |
| **`float`, lignes** | **SOLVEUR LINÉAIRE EN ÉCHEC** (résidu 1.65e+03, 20 119 itérations de CG) | STAGNATION **1.13e−03** |

**L'échec franc du § 4 a disparu** — le solveur linéaire ne part plus. Et le `double` y gagne
aussi, d'un facteur huit sur le résidu final à comptes de diagrammes identiques : le repère local
n'est pas une affaire de `float`, c'est du conditionnement.

### Et ça ne coûte presque rien

`n = 10⁶`, uniforme, ns/germe :

| | 2D `double` | 2D `float` | 3D `double` | 3D `float` |
|---|---|---|---|---|
| Voronoï avant / après | 137 / **145** | 142 / **150** | 1960 / **1946** | 1807 / **1818** |
| Laguerre avant / après | 164 / **163** | 161 / **168** | 1964 / **1982** | 1786 / **1820** |

**Entre 0 et +6 %**, et le +6 % est sur le cas le plus rapide (2D Voronoï, 137 ns/germe), là où un
calcul de plan en double par candidat pèse le plus. En 3D c'est indiscernable de zéro.

## 19.6 Le dernier terme : la mémoire des premières coupes

Le repère du germe rend exactes les coupes **tardives**, pas les premières. Un sommet créé quand la
cellule mesure encore `L` porte l'erreur `eps·L` ; la cellule part à la taille du **domaine**, d'où
un plancher `eps·L/h` qui vaut `eps·n^(1/D)` avec `L = 1`. Il survivait au repère local — divisé
par cinq, mais avec la même loi.

Un diagnostic le prouve avant de le réparer (`SF_R`, qui annonce sur `stderr` que les cellules
dépassant son carré sont fausses) : partir d'un carré de demi-côté 0.05 au lieu du domaine divise
l'erreur par 7.3 à `n = 10⁶`, et la loi reste la même avec un préfacteur vingt fois moindre.

## 19.7 Les boîtes : la feuille du BSP, et un filet

`cell/Boite.h`. **La feuille du BSP est la boîte qu'on cherchait** : elle tient une poignée de
germes, donc sa taille *est* l'espacement local — y compris là où les germes sont serrés, ce qu'un
`h` global ne saurait pas faire. On part de la feuille du germe, dilatée, intersectée avec le
domaine.

**Le filet, parce que rien ne garantit qu'une cellule tienne dans sa boîte.** Les faces de la
boîte qui ne sont pas des faces du domaine reçoivent un identifiant *artificiel* ; si l'une d'elles
survit, la cellule touche la boîte et on recommence plus grand. C'est exact et pas seulement
prudent : si `C ⊄ B`, alors soit `C ∩ B = ∅`, soit `C ∩ B` a une face portée par `∂B`. Le test est
conservatif — on recommence parfois pour rien, jamais l'inverse.

**Le cas (a) est le piège, et il a mordu.** On est tenté de raisonner « la cellule contient son
germe, qui est dans la boîte, donc elle ne peut pas être vide » : **c'est faux en Laguerre.** Un
poids assez bas met la cellule ailleurs que sur son germe, ou la supprime. `check` disait 0/5000 et
Newton divergeait à la première itération — parce que `check` codait ses poids en dur à `0` et
`h²`, deux échelles où chaque cellule reste posée sur son germe. **Le témoin ne pouvait pas voir ce
défaut-là**, et il balaye maintenant trois échelles dont une (`ws = 100`) où les cellules quittent
leurs germes.

**La reprise agrandit, elle ne saute pas au domaine**, et ce n'est pas une économie de temps : une
cellule reprise au domaine retrouve l'erreur `eps·1` qu'on vient d'enlever, et comme le plancher de
Newton suit le **maximum** de l'erreur, quelques pour cent de cellules suffisent à le fixer. Mesuré
à `n = 10⁵` : le saut au domaine laissait un max de 4.50e−05, l'agrandissement le met à 1.52e−06
(×30) et le plancher de Newton passe de 1.13e−04 à 1.14e−05. Le facteur est **4** et non 2, parce
que sur le nuage de **lignes** dix germes colinéaires font une feuille en lamelle alors que leurs
cellules s'étendent *perpendiculairement* jusqu'à la ligne voisine : il faut plusieurs ordres de
grandeur.

### La loi en `n` disparaît

Écart de masse médian `float` / `double`, uniforme Voronoï :

| `n` | 10³ | 10⁴ | 10⁵ | 10⁶ |
|---|---|---|---|---|
| **2D** sans boîte | 2.34e−07 | 6.00e−07 | 2.37e−06 | 6.41e−06 |
| **2D** avec | 7.51e−08 | 7.92e−08 | 6.88e−08 | **7.68e−08** |
| **3D** sans boîte | 8.09e−08 | 1.40e−07 | 3.31e−07 | 6.25e−07 |
| **3D** avec | 5.12e−08 | 5.54e−08 | 5.20e−08 | **5.53e−08** |

**Plat.** 7.7e−08, c'est 1.3 fois l'epsilon du `float` : la cellule médiane est juste à l'arrondi
près, et elle le reste quand `n` grandit. Au départ de cette étude, `n = 10⁶` en 2D donnait
3.54e−05 : **×460**.

La dilatation est le paramètre, et l'erreur y est maintenant monotone — c'est bien le `L` du
modèle. À `n = 10⁵`, 2D, poids `h²` :

| dilatation | médiane | p99 | max | reprises |
|---|---|---|---|---|
| 0 (domaine) | 6.78e−07 | 2.07e−05 | 6.69e−05 | — |
| **1** (défaut) | **3.44e−08** | **4.59e−07** | **1.52e−06** | 142 %* |
| 2 | 5.23e−08 | 7.35e−07 | 2.81e−06 | |
| 4 | 8.45e−08 | 1.39e−06 | 4.57e−06 | 112 %* |
| 8 | 1.40e−07 | 2.71e−06 | 7.94e−06 | |

\* à `n = 10⁶`. **Le taux de reprise est le prix, et il dépend des poids** : 0.05 % en Voronoï,
**142 % en Laguerre** même avec des poids d'ordre `h²`. C'est 1.4 agrandissement par cellule, donc
à peu près deux fois le travail de découpe — le diagramme uniforme Laguerre passe de 169 à
277 ns/germe en 2D. On l'accepte : la question de cette session est la précision, pas la vitesse.
Mais c'est le fait qu'il faut retenir pour la carte, où une reprise par cellule est une divergence
de warp : **avec des poids, presque toute cellule touche une boîte de rayon `3h`.**

## 19.8 Est-ce que la chaîne passe en `fp32` ? En 3D oui, en 2D pas encore

Newton sur l'uniforme, `--kernel float` contre `--kernel double`, tolérance `1e−06` :

| | `double` | `float` |
|---|---|---|
| 2D `n = 2·10⁴` | CONVERGE 4.06e−07 — 5 it, 9 diag | STAGNATION 1.44e−05 — 10 it, 63 diag |
| 2D `n = 10⁵` | CONVERGE 2.58e−10 — 6 it, 8 diag | STAGNATION 1.97e−05 — 15 it, 148 diag |
| **3D `n = 2·10⁴`** | CONVERGE 2.78e−08 — 5 it, 7 diag | **CONVERGE 9.55e−07 — 5 it, 7 diag** |
| 3D `n = 10⁵` | CONVERGE 1.66e−11 — 6 it, 9 diag | STAGNATION 2.28e−06 — 20 it, 165 diag |

**En 3D à `n = 2·10⁴`, `float` converge par le même chemin que `double`** : cinq itérations, sept
diagrammes, un recul — le compte est identique, ligne pour ligne. C'est la première fois que la
chaîne géométrique passe entièrement en simple précision. À `n = 10⁵` elle manque de peu
(2.28e−06 contre 1e−06) ; en 2D elle s'arrête quatorze à vingt fois trop haut.

La trace dit exactement où ça s'arrête :

```
        double                          float
it 3    |r|_2 4.425e-06                 4.426e-06
it 4    |r|_2 1.661e-07                 1.662e-07
it 5    |r|_2 4.456e-10   pas 1.00      2.757e-09   pas 3.05e-05   16 diag
it 6    |r|_2 3.055e-15   CONVERGE      2.757e-09   pas 9.54e-07   21 diag
```

`float` suit `double` **chiffre pour chiffre pendant cinq itérations**, puis la recherche linéaire
s'effondre : le pas est divisé jusqu'à 5.8e−11 sans jamais faire décroître `|r|_2`. Ce n'est pas
une divergence, c'est un **plancher de bruit** — et le critère d'arrêt de Newton porte sur le
`max` sur `n` cellules, donc sur la **queue** de l'erreur géométrique et non sur sa médiane. La
médiane est à 1.3·eps ; le max est à 1.5e−06, et Newton s'arrête dix fois au-dessus.

Les deux dimensions ne diffèrent plus par la loi — les deux sont plates en `n` — mais par ce max
et par le nombre de cellules sur lequel on le prend. C'est ce qui reste de l'asymétrie 2D/3D du
§ 19.1 : elle n'est plus dans l'exposant, elle est dans la queue.

**Ce qui resterait à faire** pour la 2D : ce n'est plus un problème d'échelle — les trois
mécanismes d'échelle sont réparés — mais de **conditionnement local**. Les cellules qui font la
queue sont celles dont une facette est sur le point d'apparaître ou de disparaître : `t = s₀/(s₀−s₁)`
y a un dénominateur qui s'annule, et aucune translation de repère n'y peut rien. La piste n'est
donc plus arithmétique : elle serait d'accepter l'arrêt de Newton au plancher de bruit (un critère
sur `|r|_2` plutôt que sur le `max`), ou de recalculer en `double` les quelques cellules mal
conditionnées — qu'on sait détecter, ce sont celles dont une facette a une aire relative sous
quelques `eps`.

## 19.9 L'assemblage doit être SYMÉTRIQUE, ou le CG n'avance plus

En cherchant combien la simple précision ferait gagner, on est tombé sur un défaut qui n'a rien
d'un arrondi : en 3D, `n = 2·10⁴`, la phase `float` passait **44,7 s dans le solveur linéaire pour
100 000 itérations de CG** — contre 0,10 s et 155 itérations en `double`. Le diagramme, lui,
prenait 0,32 s des deux côtés.

`Laplacien.h` assemblait la hessienne **sans un seul tri**, en gardant pour chaque ligne la mesure
de *sa* cellule : `L_ij` venait de la cellule `i`, `L_ji` de la cellule `j`. Les deux vues d'une
même facette diffèrent de `1e−16` en double et le gradient conjugué ne s'en aperçoit pas. **En
simple précision elles diffèrent de 100 %** sur les facettes presque dégénérées — et le CG, qui
suppose un opérateur symétrique, cesse de converger.

Le remède ne coûte rien et ne demande toujours aucun tri : ne garder que la vue `i < j` et la
**miroiter** dans les deux lignes. Le comptage compte les deux côtés, la somme préfixe place tout
le monde, chaque ligne somme toujours exactement à zéro, et `L` est symétrique **au bit près**.
Une facette vue d'un seul côté n'est gardée que si `i < j` — sur les millions d'arêtes d'un
diagramme à `n = 10⁶`, `fp32` en compte entre zéro et cinq, et une arête qu'une des deux cellules
ne voit même pas est microscopique.

**100 000 itérations → 186. 44,7 s → 0,12 s.** Le `double` est inchangé au bit près.

C'est le genre de défaut qu'on ne voit qu'en `float` : la double précision le masquait depuis le
début, et il attendait qu'on descende la précision pour se manifester.

## 19.10 Ce que la bascule `fp32 → fp64` ferait gagner (`--kernel mixte`)

La forme utile de la simple précision n'est pas « tout en `float` » : c'est **commencer en `float`
et finir en `double`**, puisque `float` suit `double` chiffre pour chiffre pendant les premières
itérations avant de s'écraser sur son plancher de bruit. `--kernel mixte` fait exactement ça, et
imprime les deux phases plus un témoin tout-double.

### Rendre la main au bon moment : trois critères, un seul marche

Les deux critères **réactifs** se trompent, chacun dans son sens :

* **le progrès** (`--mixte-progres`) : les premières itérations gagnent légitimement ~50 % sur
  `|r|_2` (1.364e−3 → 6.604e−4 → 3.341e−4 en 3D), donc un seuil à 0,5 coupe la phase `float` dès
  la deuxième ;
* **le pas** (`--mixte-tmin` à 0,25) : la *première* itération demande légitimement un petit pas,
  et à `n = 2·10⁴` en 2D elle rendait la main tout de suite, quatre diagrammes gaspillés.

Le bon critère se **pose d'avance**. L'écart de mesure d'une cellule vaut `κ·eps·ν_i`, donc le
plancher du mérite vaut `‖bruit‖₂ = κ·eps/√n`. À `n = 10⁵` en `float` ça prédit **1.9e−10** — et la
trace montre Newton bloqué à **2.0e−10**. `--mixte-kappa` (défaut 30) le pose, et il se déclenche
proprement dans les six cas mesurés.

Sans lui, le coût est spectaculaire : sur l'uniforme 3D à `n = 10⁵`, les six premières itérations
`float` sont identiques à celles du `double` et coûtent **neuf diagrammes** ; les quatorze
suivantes en coûtent **cent quarante-cinq** pour faire passer `|r|_2` de 4.3e−10 à 2.0e−10.

### Le partage, qui est la quantité qui se transporte

Avec le plancher, la bascule tombe **toujours au même endroit : cinq itérations en `float`, une en
`double`**.

| uniforme | diag. `float` | diag. `double` | total | témoin tout-`double` |
|---|---|---|---|---|
| 2D `n = 2·10⁴` | 9 | 2 | 11 | 9 |
| 2D `n = 10⁵` | 7 | 2 | 9 | 8 |
| 2D `n = 5·10⁵` | 11 | 2 | 13 | 12 |
| 3D `n = 2·10⁴` | 7 | 2 | 9 | 7 |
| 3D `n = 10⁵` | 8 | 2 | 10 | 9 |
| 3D `n = 5·10⁵` | 8 | 2 | 10 | 8 |

**70 à 85 % des diagrammes passent en `fp32`, pour un surcoût de un à deux diagrammes** (le
diagramme de reprise et l'itération de finition). Le résidu final est le même ou meilleur que
celui du témoin — souvent bien meilleur, l'itération `double` finale allant plus loin.

### Et donc, combien ?

Le surcoût en **nombre** de diagrammes est de +8 à +25 %. Si `fp32` est `s` fois plus rapide par
diagramme, la bascule vaut `N_f/s + N_d` contre `N₀`, donc **le seuil de rentabilité est
`s = N_f / (N₀ − N_d)`, soit 1,10 à 1,33 selon le cas** :

| `s` | 2D `n = 10⁵` | 3D `n = 10⁵` | 2D `n = 5·10⁵` |
|---|---|---|---|
| 1 (ce CPU) | +12 % | +11 % | +8 % |
| 1,5 | −7 % | −11 % | −17 % |
| **2** | **−31 %** | **−33 %** | **−37 %** |
| 4 | −45 % | −50 % | −52 % |

**Sur ce CPU, `s ≈ 1` et il n'y a rien à gagner.** Diagramme isolé, `n = 10⁶`, mêmes germes et
mêmes poids des deux côtés, avec les défauts de la § 19.11 :

| | `double` | `float` | `s` |
|---|---|---|---|
| 2D Voronoï | 146 | 160 ns/germe | 0,91 |
| 2D Laguerre | 167 | 179 | 0,93 |
| 3D Voronoï | 1880 | 1912 | 0,98 |
| 3D Laguerre | 1921 | 1887 | 1,02 |

L'engin est limité par le débit d'instructions SIMD et par les dépendances du découpage, pas par
la bande passante : l'AVX ferait huit `float` par cycle contre quatre `double`, mais le découpage
n'est pas ce qui sature. **La bascule y perd donc exactement son surcoût de un à deux diagrammes,
en 2D comme en 3D.**

*(Une mesure antérieure donnait `s = 1,17` en 3D. Elle comparait les temps des PHASES de la
bascule — des diagrammes pris à des points différents du chemin de Newton, donc sur des cellules
différentes. Ce n'est pas une comparaison propre, et le tableau ci-dessus la remplace.)*

La conclusion transportable est donc : *le partage 5/1 tient, le surcoût est de un à deux
diagrammes, le seuil est à 1,2, et tout ce qui dépasse est du gain.* C'est une barre basse pour une
carte, où `fp32` vaut au minimum deux fois le débit du `fp64` et trente-deux fois sur une puce
grand public.

### Mettre les cellules problématiques de côté : mesuré, et ça ne peut pas marcher

L'idée était naturelle : le plancher est fixé par le **maximum** de l'erreur géométrique ; si ce
maximum vient d'une poignée de cellules mal conditionnées, les recalculer en `double` coûterait ce
pour cent et rendrait des itérations à la phase `float`.

**Un détecteur gratuit et sans modèle existe** : chaque facette est mesurée **deux fois**, une par
cellule, et on jetait la seconde vue. En double les deux coïncident à `1e−16` ; en `float` leur
écart *est* une estimation de l'erreur locale. L'indicateur est
`ind_i = max_j |c_ij − c_ji| / (c_ij + c_ji)`, et il ne coûte rien.

**Il ne sépare pas.** Le maximum de l'erreur qui reste après avoir recalculé les plus suspectes
(uniforme 2D, `n = 10⁵`, poids `h²`) :

| critère | 0 % | 0,1 % | 1 % | 5 % | 20 % |
|---|---|---|---|---|---|
| **ORACLE** (l'erreur vraie) | 2.38e−06 | 7.33e−07 | 4.65e−07 | 2.87e−07 | 1.35e−07 |
| désaccord `c_ij` / `c_ji` | 2.38e−06 | 2.38e−06 | 2.38e−06 | 2.38e−06 | 2.38e−06 |
| plus petite facette | 2.38e−06 | 2.38e−06 | 2.38e−06 | 2.38e−06 | 2.38e−06 |

Aucun des deux critères ne bouge le maximum, même en recalculant une cellule sur cinq. Pour le
désaccord des facettes la raison se voit : il mesure la **longueur** de la facette, alors que ce
qui change l'aire d'une cellule est son déplacement **perpendiculaire** — les deux cellules
s'accordent sur le plan (elles le calculent des mêmes données) et se distinguent sur les sommets,
dont la longueur ne retient qu'une projection.

**Mais la ligne ORACLE dit qu'il n'y a de toute façon rien à chercher.** Même un détecteur PARFAIT
ne divise le maximum que par 3 en recalculant 0,1 % des cellules, par 5 à 1 %, par 18 à 20 %. La
queue n'est pas une poignée de pathologies isolées, c'est **une distribution lisse** : il faudrait
en recalculer une grande fraction pour gagner un ordre, ce qui ôte tout intérêt à la manœuvre.

**Et surtout, le maximum n'est pas ce qui arrête Newton.** La recherche linéaire s'écrase sur le
mérite `|r|_2`, qui est une norme `L2` sur `n` cellules : elle est portée par **le gros de la
distribution**, chaque cellule à 1,3·eps, et un maximum à 70 fois la médiane ne pèse rien dedans
sur 10⁵ termes. Baisser la queue ne rendrait donc aucune itération à la phase `float`.

### Pourquoi le partage 5/1 est optimal, et non réglé

Il est **structurel**. Newton double le nombre de chiffres justes à chaque itération ; `float` en
porte sept, `double` seize. Les dernières itérations sont donc exactement celles qui demandent les
chiffres que `float` n'a pas, et tout ce qui précède est gratuit. *On ne peut pas faire mieux que
« toutes sauf la dernière »* — et c'est là qu'on est : cinq en `float`, une en `double`, dans les
six cas mesurés, en 2D comme en 3D, de `n = 2·10⁴` à `5·10⁵`.

Le seul levier restant est donc `s`, le rapport de vitesse par diagramme. Il ne se gagne pas en
arithmétique : il se gagne sur une machine où `fp32` va vraiment plus vite.

---

## 19.11 LES SOMMETS RÉSOLUS DEPUIS LEURS PLANS — les boîtes deviennent inutiles

Les boîtes de départ (§ 19.7) achetaient la précision au prix d'une **reprise par cellule** :
acceptable sur CPU, mauvais sur carte, où une reprise est une divergence de warp. Or elles
combattaient une erreur bien particulière — l'erreur **portée** : le noyau construit ses sommets
par interpolations successives, si bien qu'un sommet né quand la cellule mesurait `L` garde `eps·L`.

**Mais un sommet d'un convexe ne dépend pas de l'histoire des coupes.** Il est l'intersection de
`D` plans, et de rien d'autre. Une fois la cellule finie, on peut donc le **résoudre** depuis les
deux (2D) ou trois (3D) coupes qui le portent — un Cramer, en `double`, dans le repère du germe.
Plus d'histoire, donc plus de `L`, et **aucune branche** : sur carte, pas de divergence.

L'information nécessaire était déjà là : en 3D `Cellule3::vk0/vk1/vk2` stocke précisément les trois
coupes de chaque sommet ; en 2D `cid[i−1]` et `cid[i]` donnent les deux. Il ne manquait que de
relire le plan depuis l'identifiant de la coupe, ce que l'arbre sait faire.

`n = 10⁶`, uniforme Laguerre, écart de masse `float` / `double` :

| | médiane | p99 | max | 2D, ns/germe |
|---|---|---|---|---|
| ni l'un ni l'autre | 1.71e−06 | 6.49e−05 | 2.52e−04 | |
| boîtes | 3.82e−08 | 5.10e−07 | 1.20e−05 | 219 |
| **sommets résolus** | **6.24e−09** | **1.09e−07** | **3.39e−07** | **175** |
| les deux | 6.24e−09 | 1.09e−07 | 3.39e−07 | |

**En 2D le raffinement rend la boîte strictement inutile** : six fois meilleur sur la médiane,
trente-cinq fois sur le maximum, vingt pour cent plus rapide, et les reprises en moins. Les deux
ensemble ne donnent rien de plus, au chiffre près. 6,24e−09, c'est **un dixième de l'epsilon du
`float`** — le sommet est calculé en `double` et ne subit qu'un seul arrondi.

**En 3D il gagne sur le gros et perd sur la queue** (max 4,15e−06 contre 2,00e−06 avec les boîtes ;
3,87e−07 avec les deux), pour le même temps. La raison se lit dans le compte d'arêtes en
désaccord : **9 sans boîte contre 2 avec**. La boîte ne sert pas qu'aux coordonnées, elle fiabilise
les **décisions** — un sommet mal placé peut se tromper de côté d'un plan, et aucun raffinement
postérieur ne rattrape une topologie déjà fausse. (Le seuil de conditionnement du Cramer n'y est
pour rien : le balayer de `1e−3` à zéro ne change pas un chiffre.)

**Le défaut est donc : sommets résolus, pas de boîtes** — `SF_DIL` reste pour qui veut la queue en
3D. Le maximum ne pilote rien (§ 19.10 : Newton s'arrête sur `|r|_2`, portée par le gros), et la
bascule le confirme : **partage 5/1 et comptes de diagrammes identiques** dans les quatre
configurations mesurées, avec ou sans boîtes.

Sur ce qui inquiétait — la divergence : à `n = 10⁶` en 2D avec des poids dix fois `h²`, **zéro
reprise, et un écart de masse médian de 0.00e+00** — plus de la moitié des cellules sont identiques
au bit près entre `float` et `double`, p99 2,5e−07, max 7,9e−07.

Le raffinement est actif **en simple précision seulement** : en `double` l'erreur portée vaut
`1e−16·L` et il n'y a rien à réparer. `SF_RAFF=0` l'éteint.

## 19.12 La mesure image : le stockage peut être `float`, la marche non

Troisième instance du même mécanisme. Le coupable nommé au § 12.7 était `S[j][i] − sref` : la somme
préfixe court de 0 à ~1 sur une ligne, la différence entre deux pixels d'une même cellule vaut
`h·ρ`. **`sref` n'y change rien** — il ramène le *terme* de 1 à `h`, mais l'erreur a été faite au
**stockage**. Le second terme est `(x_p + x_c)/2 − i·h_x`, deux nombres d'ordre 1 pour une
différence d'ordre `h_x` (il est maintenant écrit `½((x_p − i·h_x) + (x_c − i·h_x))`, deux
soustractions au lieu d'une somme puis une soustraction).

`ImageT` a donc **deux flottants** : `TA` le stockage (`v`, le tableau `W×H` — celui dont la bande
passante décide sur une carte) et `TW` le calcul. Image 512², `n = 2·10⁴` :

| | masse médiane | masse max | facette médiane | **facette max** |
|---|---|---|---|---|
| tout en `float` | 1.02e−06 | 7.84e−05 | 2.52e−06 | **1.51e−01** |
| **image seule en `float`** | **1.41e−09** | **1.63e−07** | **1.13e−08** | **5.90e−08** |

**Mille fois mieux sur la masse, deux millions et demi de fois sur la pire facette** — et surtout
*la queue disparaît* : 5.90e−08 est l'epsilon du `float`, sans aucun événement rare. L'erreur
résiduelle est exactement la quantisation de ρ, propagée linéairement.

C'est le résultat utile pour la carte : **le gros tableau peut être en `float`** (moitié de bande
passante, moitié de mémoire), les quelques scalaires de la marche non. `--acc float` désigne
désormais le stockage ; `ImageT<float,float>` garde l'ancien comportement, et `image --check`
imprime les deux pour qu'on ne les reconfonde pas.

---

# 20. DÉFORMER LA CIBLE POUR ALLONGER LE PAS (`newton --cible`)

L'idée, posée en une ligne : on sait lire pour presque rien quand chaque cellule va s'éteindre le
long de `d` (§ 15.14) ; **est-ce qu'on peut donner plus de masse cible aux plus exposées, pour que
le coefficient de relaxation admissible monte ?** On ne touche alors qu'au **second membre** — pas
de nouveau diagramme, pas de nouvel assemblage, une descente de plus sur la factorisation déjà
faite — et la cible déformée est **transitoire** : l'itération suivante repart de `ν`.

C'est un objet différent du § 13, où pondérer les équations ne changeait rien du tout (le système
est carré, l'identité mange les poids). Changer le second membre change vraiment `d`. Tout est dans
`src/solver/Cible.h`, branché dans `Newton.h` derrière `--cible`.

## 20.1 Ce que le second membre peut atteindre, et ce qu'il ne peut pas

Deux faits d'algèbre décident d'avance de la forme que l'idée doit prendre.

**À l'ordre un, la direction de Newton ne tue personne.** `a_i( w + t d ) = a_i + t ( L d )_i =
( 1 − t ) a_i + t ν_i` : chaque masse interpole linéairement de la sienne vers sa cible. Une
extinction est donc **entièrement un effet de courbure**, et raisonner sur `ν` au premier ordre ne
la voit même pas.

**Le second membre fixe la divergence, jamais le gradient.** Ce qui tue la cellule `i` est le flux
sortant **brut** `S_i = Σ_j max( 0, c_ij ( d_j − d_i ) )`, pendant que `( L d )_i = ν_i − a_i` n'est
que le flux **net**. Une cellule peut encaisser beaucoup d'un côté et en perdre autant de l'autre :
son net est petit, son brut la dévore. Nourrir `i` ne borne que le net.

Deux conséquences, et les deux se mesurent :

* le **contrôle obligatoire** : `ν~ = a + s ( ν − a )` donne `d~ = s d` **exactement**, donc un pas
  admissible `U*/s`. Toute déformation **uniforme** du second membre est un pas plus court déguisé.
  Seule une déformation non uniforme peut payer ;
* le **plafond** : sur un patch où l'on pose `ν_i := a_i` (donc `b_i = 0`), `d` devient
  **harmonique**, et l'harmonique minimise l'énergie de Dirichlet `Σ c_ij ( d_i − d_j )²` à bord
  donné. C'est le plus petit gradient qu'un second membre puisse obtenir dans le patch : le plafond
  de toute la famille. `--cible plafond` le calcule sans rien modifier, une résolution par épaisseur.

## 20.2 Le plafond, mesuré : il est sous 1 là où ça compte

Lignes `n = 10⁵`, `s = 0.005`, Cholesky, `--pas essai-limites`. `U*` est le pas lu sur les flux avant
déformation ; la colonne donne `U*` après gel du patch, en multiple de `U*`.

| itération | `U*` | 1 anneau | 2 anneaux | 4 anneaux | 8 anneaux | qui borne après |
|---|---|---|---|---|---|---|
| 0 | 3.08e−05 | ×0.92 | ×0.98 | ×0.78 | ×1.19 | la **même** cellule |
| 1 | 1.24e−04 | ×0.79 | ×0.69 | ×0.68 | ×0.69 | la **même** cellule |
| 2 | 2.69e−04 | ×0.96 | ×0.95 | ×0.93 | ×0.98 | la **même** cellule |
| 3 | 2.28e−04 | ×0.99 | ×0.96 | ×0.99 | ×0.83 | la **même** cellule |
| 5 | 1.56e−03 | ×0.87 | ×0.86 | ×0.84 | ×1.87 | 31862, dedans |
| 9 | 4.64e−03 | ×0.87 | ×0.86 | ×0.85 | ×0.81 | la **même** cellule |
| 10 | 8.50e−03 | ×1.67 | ×2.78 | ×2.15 | ×2.82 | 34411, dehors |

**Le plafond est sous 1 dans toutes les itérations difficiles**, et il ne passe au-dessus qu'à la
fin, quand le pas vaut déjà 1/2. Geler la demande autour du foyer **raccourcit** le pas, et la
raison est nette : sur ce nuage la cellule qui borne est minuscule devant sa cible (`b_i = ν_i − a_i`
très positif), et **cette demande est précisément l'ordre de sauvetage qui la maintient en vie**. La
taire, c'est la lâcher. Le gel est donc le mauvais signe ; l'intuition de départ — *nourrir* — a le
bon.

## 20.3 Nourrir : ça marche, et ça ne sert à rien

La dose a une échelle naturelle, et elle s'annule d'elle-même sur les cellules qui tiennent :

```
delta_i  =  kappa * max( 0, S_i − ( a_i − eps ) / F )
```

`kappa = 1` est la dose qui suffirait **si** tout le supplément passait par les facettes dévorantes.
La masse est reprise au prorata de `ν` sur le reste (`Σ b~ = 0` est obligatoire : sans ça la ligne
rayée par la jauge porte toute l'incohérence, § 9.6). Balayage de `κ`, itération 0 :

| `κ` | masse donnée | `‖δb‖/‖b‖` | `U*` | qui borne |
|---|---|---|---|---|
| 0.25 | 0.01 % | 5.1e−04 | ×1.13 | la visée |
| **1** | **0.03 %** | **2.0e−03** | **×1.80** | la visée |
| 4 | 0.12 % | 8.2e−03 | ×1.64 | **une autre** |
| 16 | 0.49 % | 3.3e−02 | ×0.53 | une autre |
| 64 | 1.94 % | 1.3e−01 | ×0.11 | une autre |

**Le mécanisme fonctionne exactement comme annoncé** : pour **trois centièmes de pour cent** de la
masse déplacée et **une résolution linéaire**, le pas lu sur les flux **double**. Avec la dose
auto-réglée, `--cible-f 2` atteint sa consigne au premier essai : `U* 3.076e−05 → 6.268e−05`
(×2.04, `vise 6.151e−05`, 7 cellules nourries, 0.04 % de masse).

Et pourtant, sur un solve complet (mêmes réglages, jusqu'à stagnation) :

| | itérations | **diagrammes** | résolutions de plus | total |
|---|---|---|---|---|
| `essai-limites` (témoin) | 19 | **64** | — | 8.66 s |
| `--cible gel --cible-f 2` | 23 | 77 | 90 (1.11 s) | 11.04 s |
| `--cible gel --cible-f 4` | 21 | 68 | 85 (1.03 s) | 9.80 s |
| `--cible gel --cible-f 8` | 20 | 66 | 75 (0.91 s) | 9.58 s |

**Rien, et même un peu moins que rien.** Deux causes, toutes deux lisibles dans la trace.

**`U*` n'est pas ce qui borne.** À l'itération 0 le témoin accepte `α* = 3.81e−03`, soit **124 fois**
`U* = 3.08e−05`. La linéarisation en `t = 0` est conservative d'un facteur cent (§ 15.15 le disait
déjà : « ce n'est pas `U` qui est faux, c'est le critère »). Doubler `U*` ne déplace donc pas le pas
accepté — pire, la direction déformée le fait **tomber** de `3.81e−03` à `1.70e−03`.

**Passé les premières itérations, aucune dose n'améliore quoi que ce soit.** Sur les 18 itérations,
**douze** affichent `RIEN NE FAIT MIEUX` : sauver les exposées en nomme d'autres, exactement le
constat de population du § 13.4 et de la courbe `N( t )` du banc GPU.

## 20.4 Et le critère « population » ne sauve pas l'idée

Puisque ce n'est pas une cellule isolée qui borne, on a remplacé le critère « maximiser `min_i U_i` »
par « minimiser le **nombre** de cellules qui ne tiennent pas jusqu'à l'horizon `β` du pas »
(`--cible-critere pop`). Le résultat est un **chiffre qui ferme la porte** : à `β = 0.25`, `U`
déclare **99 580 cellules malades sur 100 000**, quand le diagramme à ce pas n'en trouve que **5 991**
sous `ε`. Aucune dose ne peut rien pour 99 % du diagramme, le balayage rend `κ = 0` partout, et le
solve est identique au témoin à 0.85 s près.

C'est un fait utile en soi et il vaut au-delà d'ici : **`U` sert à choisir un pas — son minimum est
un minorant utilisable — mais ne sert pas à classer les cellules une par une.** Sa conservativité,
que le § 15.14 avait trouvée « accidentellement bonne », est de deux ordres de grandeur.

## 20.5 L'équilibrage : rendre la masse à côté, et ce que ça révèle

La compensation du § 20.3 reprend la masse **au prorata de `ν` sur tout le dehors**. C'est un
**monopôle** : `δ` a une partie positive ponctuelle et une partie négative étalée, donc `L⁺δ` porte
en 2D un potentiel logarithmique — la correction de direction traîne sur tout le diagramme. Rendre
la masse **à côté** (somme nulle localement) rend le champ dipolaire, donc amorti. Deux règles
(`--cible-repris`), contre `global` :

* **`anneau`** : les cellules malades qui se touchent sont agrégées en **amas** (union-find sur le
  graphe du laplacien), et chaque amas prend à **sa propre couronne** (`--cible-ep`), au prorata
  de `ν`. Somme nulle par amas ;
* **`mangeurs`** : chaque victime prend à **ses mangeurs**, au prorata du flux qu'ils lui volent.
  Somme nulle par cellule — le dipôle le plus court qui existe, et celui que la résistance
  effective désigne comme le plus efficace (`c_ij R_eff( i, j ) ≤ 1`, l'égalité pour le voisin
  direct).

`FUITE` = la part de l'énergie de `d~ − d` (jauge ôtée) qui vit **hors du cœur et de son anneau**.

| itération 0 | `U*` | masse | `‖δb‖/‖b‖` | **FUITE** | `α*` accepté |
|---|---|---|---|---|---|
| Newton (témoin) | 3.08e−05 | — | — | — | **3.81e−03** |
| `global` | ×2.04 | 0.04 % | 2.7e−03 | **99.0 %** | 1.70e−03 |
| `anneau` | ×2.08 | 0.04 % | 3.1e−03 | **50.9 %** | 3.65e−04 |
| `mangeurs` | ×2.09 | **0.02 %** | 1.9e−03 | **58.6 %** | 2.84e−04 |

**Le diagnostic était bon** : avec la reprise globale, **99 % de la correction est ailleurs**. La
reprise locale divise la fuite par deux, et elle fait mieux sur tous ses propres indicateurs — sur
les itérations suivantes `mangeurs` monte à ×2.7, ×2.9, ×4.1, ×9.7, ×10.0 sur `U*`, guérit la
victime à chaque fois (`malades 1 → 0`) et coûte **quatre fois moins de masse** que la reprise
globale.

**Et c'est exactement ce qui la condamne.** Sur un solve complet :

| | itér. | **diagrammes** | gain moyen sur `U*` |
|---|---|---|---|
| `essai-limites` (témoin) | 19 | **64** | — |
| `global` `F = 8` | 20 | 66 | ×1.13 |
| `global` `F = 2` | 23 | 77 | ×1.50 |
| `mangeurs` `F = 8` | 25 | 76 | ×3.53 |
| `anneau` `F = 8` | 32 | 90 | ×3.75 |
| `anneau` `F = 2` | 44 | 115 | ×2.83 |
| **`mangeurs` `F = 2`** | **100, NE CONVERGE PAS** | **201** | **×6.23** |

**L'anti-corrélation est parfaite : plus la déformation réussit sur son critère, pire est le
solve.** À `mangeurs F = 2`, `U*` gagne un facteur six, le résidu ne bouge plus (`max|a−ν|/ν` reste
à 1.5e+03 après cent itérations, contre 2.35e−06 pour le témoin) et il n'y a **aucun recul** : tous
les pas sont acceptés, ils sont simplement microscopiques.

Deux mécanismes, et ils tirent dans le même sens :

* **le dipôle court achète `U*` en fabriquant un gradient de poids concentré.** `‖δb‖/‖b‖` vaut
  2e−03, mais il est porté par deux ou trois cellules voisines : c'est un **écart de poids à travers
  une facette**, donc précisément ce qui tue une cellule — et `U`, linéarisation en `t = 0`, ne le
  voit pas. Le pas réel mesuré par le diagramme tombe de `3.81e−03` à `2.84e−04` pendant que `U*`
  double. On a échangé un facteur treize réel contre un facteur deux sur le papier ;
* **la direction déformée cesse d'être une direction de descente pour le vrai résidu.** À l'ordre un
  le résidu devient `( 1 − t ) r − t δ` : la décroissance garantie est `( 1 − ‖δ‖/‖r‖ ) t / 2`, et
  quand `δ` est local et `t` minuscule le test `n2r < nr` passe par un cheveu à chaque itération.
  Newton accepte cent pas de rien du tout.

**Ce que l'équilibrage local apprend vraiment** est donc un fait sur le critère, pas sur la
méthode : **optimiser `U*` fort le rend d'autant plus mauvais comme proxy.** La conservativité de
`U` que le § 15.14 avait trouvée « accidentellement bonne » ne survit pas à ce qu'on l'optimise —
c'est la loi de Goodhart, mesurée.

## 20.6 Le bon prédicteur : l'aire à combinatoire figée, pas les flux d'arêtes

Tout ce qui précède est jugé par `U`, qui n'est pas un prédicteur d'extinction mais une
**linéarisation des flux en `t = 0`** (§ 15.13). Le prédicteur, c'est le **polynôme d'aire à
combinatoire figée** du § 7 — mesuré *exact* à l'itération 0 de ce nuage même.

**Et il ne coûte pas ce qu'on croit.** `ModeleCellule` (§ 7.3) porte les droites de la cellule ;
l'aire s'en déduit pour un déplacement de poids **quelconque**, par pure arithmétique. On construit
donc la géométrie **une fois**, et tout le balayage — toutes les directions `d~( κ )`, tous les `α` —
se lit dessus sans un seul calcul de cellule. Mieux : `κ` entre **linéairement** dans le second
membre (le cœur et les donneurs n'en dépendent pas), donc `d~( κ ) = d + κ e` avec `L e = δ( 1 )` :
**une** résolution pour tout le balayage, au lieu d'une par essai. `--cible-juge polynome`.

### La calibration, qui vaut pour elle-même

| it | `U*` (flux) | **polynôme** | pas accepté | `U` trop petit de |
|---|---|---|---|---|
| 0 | 3.08e−05 | 5.49e−03 | 3.81e−03 | **×178** |
| 1 | 1.24e−04 | 1.89e−02 | 1.35e−02 | ×152 |
| 2 | 2.69e−04 | 5.36e−02 | 5.96e−02 | ×200 |
| 3 | 2.28e−04 | 1.56e−01 | 1.30e−01 | ×681 |
| 4 | 4.29e−04 | 3.06e−01 | 1.44e−01 | ×714 |
| 9 | 4.64e−03 | 5.66e−01 | 5.00e−01 | ×122 |
| 10 | 8.50e−03 | 6.16e−01 | 6.48e−01 | ×72 |

**`U` est pessimiste de deux à trois ordres de grandeur ; le polynôme tombe à moins d'un facteur
deux du pas réellement pris** (et légèrement optimiste, comme le § 7 l'annonçait). Le chiffre est à
retenir au-delà d'ici : `U` sert à *choisir* un pas — son minimum est un minorant — et à rien d'autre.

### Ce que ça change quand il juge ET désigne

Le cœur est alors `{ i : α_polynôme( i ) < F }` — les cellules dont le bon prédicteur dit qu'elles
s'éteignent avant la cible — et non plus celles que `U` accusait. Solve complet :

| | itér. | **diag.** | déformations retenues | coût de la passe |
|---|---|---|---|---|
| `essai-limites` (témoin) | 19 | **63** | — | — |
| polynôme, `mangeurs` `F = 8` | 19 | **64** | **0 sur 19** | 3.18 s |
| polynôme, `mangeurs` `F = 2` | 19 | **64** | 1 sur 19 | 2.80 s |
| polynôme, `anneau` `F = 2` | 20 | 64 | 5 sur 20 | 2.70 s |
| polynôme, `anneau` `F = 8` | 20 | 64 | 3 sur 20 | 3.16 s |
| polynôme, `global` `F = 8` | 21 | 91 | 5 sur 21 | 3.13 s |

**Avec le bon juge, le doseur refuse presque toujours de déformer** — `κ = 0` à chaque itération ou
presque — et quand il accepte, il ne gagne rien. Les ×2 à ×10 des § 20.3 et § 20.5 étaient
**entièrement un artefact de `U`** : ils achetaient une quantité fausse d'un facteur cent.

Un cas est instructif : à l'itération 0, **une seule** cellule est à nourrir, `κ = 0.25` fait passer
la limite du polynôme de 5.49e−03 à 1.95e−02 (**×3.55**) — et le pas accepté ne bouge pas d'un
chiffre (3.81e−03). La colonne `manques` dit pourquoi : 41 773 cellules non modélisées ont un `U`
sous la limite trouvée, donc le ×3.55 est lu sur un jeu de modèles incomplet. Aux itérations
suivantes, où `manques` tombe à zéro, `κ` tombe à zéro aussi.

**Le prix du bon juge.** Le crible est `U_i < α_polynôme`, et comme `U` est cent fois pessimiste il
attrape 95 000 à 99 000 cellules sur 100 000 : **0.26 s par itération**, soit l'équivalent d'environ
un diagramme, 35 % du solve. C'est exactement ce que `essai-limites` évite en ne calculant de
limites que pour les cellules qu'un diagramme d'essai a trouvées mauvaises. Le bon prédicteur n'est
pas cher **par cellule** ; c'est de savoir *quelles* cellules modéliser qui coûte.

## 20.7 Le doseur branché sur `seules` (`--cible seules`)

Le § 20.6 se termine sur un constat de coût : le bon prédicteur n'est pas cher **par cellule**,
c'est de savoir **lesquelles** modéliser qui coûte — un crible en `U` en attrape 99 %. Or la liste
existe déjà, et elle ne soupçonne pas : `essai-limites` prend un diagramme d'essai en `t = β` et en
sort les cellules qu'il a **vues** passer sous `ε`. Elle est courte, elle est vraie, et elle est
gratuite. On s'y branche.

Le schéma, dans la boucle de `essai-limites` : au lieu de diviser `t` quand des cellules meurent,

* on modélise ces cellules-là (`ModeleCellule`, une cellule chacune, `pd` remis en `w`) ;
* on les nourrit au rythme `S_i`, la masse reprise à côté (§ 20.5) ;
* `κ` entrant linéairement, `d~( κ ) = d + κ e` avec **une** résolution, et tout le balayage se lit
  sur les polynômes — zéro géométrie par essai ;
* on vise `max( 1.5, F ) × ` la limite **courante** — viser `β` serait un facteur soixante sur six
  mille cellules, et le doseur a raison de refuser ;
* si une dose y arrive, on **re-essaye un pas plus long** avec la direction corrigée, et un vrai
  diagramme tranche.

**Le polynôme sur la bonne liste retrouve la réponse de `limites`.** À l'itération 0 : 4.234e−03
contre 4.234e−03 ; à la 1 : 1.494e−02 contre 1.494e−02 ; ensuite 5.361e−02 / 6.619e−02, 1.329e−01 /
1.443e−01, 1.928e−01 / 1.970e−01. Pour 0.02 à 0.035 s contre 0.014 à 0.021 s — comparable, et il
rend le polynôme entier (utilisable pour **toute** direction) au lieu d'un seul nombre.

**Deux gardes, chacune payée par une mesure fausse.** (i) Une cellule dont l'aire du modèle est déjà
au plancher donne `a_av = 0`, donc une cible nulle que **tout** `κ` « atteint » : le doseur
déclarait victoire en imposant un pas nul. (ii) Si le polynôme donne la cellule vivante **au-delà**
de `t`, alors qu'un diagramme vient de l'y voir morte, c'est qu'un voisin **nouveau** l'a mangée
(§ 7, invisible depuis la cellule seule) : on ne dose pas sur une prédiction qu'on sait fausse. Sans
elles, deux déformations sur dix-neuf étaient retenues sur des chiffres qui ne voulaient rien dire.

### Le compromis, balayé — parce qu'il n'y a pas de raison de viser ×2

Un premier essai fixait une **cible** (« atteindre deux fois la limite, ou ne rien faire »). C'est
un mauvais protocole : il ne teste pas le compromis, il teste une consigne arbitraire. Le vrai
réglage est un **budget de déformation** `‖δb‖/‖b‖` (`--cible-budget`) : pas de cible, on prend la
meilleure limite achetable dedans. Le budget se traduit exactement en plafond sur `κ` — `δ` lui
étant proportionnel — donc aucun essai n'est perdu.

**Un filet est indispensable avant de comparer quoi que ce soit.** Une cible déformée peut allonger
le pas *et* ne plus faire descendre le résidu : l'amortissement échoue alors et Newton sort en
STAGNATION — à un résidu qui n'a presque pas bougé. Sans filet, un réglage affichait **11 itérations
et 60 diagrammes contre 64** ; il n'avait simplement **pas convergé** (`max|a−ν|/ν = 1.6e+03` contre
`2.4e−06`). L'amortissement se fait donc en **deux passes** : si la direction déformée ne rend rien,
on revient à la direction de Newton et au pas que les limites lui avaient donné (gardés pour ça),
et on recommence. Toutes les lignes ci-dessous convergent au même résidu.

| budget `‖δb‖/‖b‖` | déformations prises | **diag. `anneau`** | **diag. `mangeurs`** |
|---|---|---|---|
| — (témoin) | 0 | **64** | **64** |
| 3e−04 | 0 | 64 | 64 |
| 1e−03 | 0 | 64 | 64 |
| 2e−03 | 2 – 4 | 65 | 69 |
| 3e−03 | 3 – 5 | 79 | 67 |
| 5e−03 | 15 | 133 | 132 |
| 1e−02 | 12 – 21 | 177 | 123 |
| sans plafond | 29 | 198 | — |

**La courbe est monotone dans le mauvais sens et n'a pas d'optimum intérieur.** Tant que le budget
est trop petit pour qu'une déformation soit retenue, on retrouve le témoin exactement ; dès qu'une
seule est prise, le compte de diagrammes monte, et il ne redescend jamais. L'optimum du compromis
est **zéro**. Ça vaut pour les deux règles de reprise, pour une ou deux rondes de réparation par
itération, sur deux décades de budget.

### Pourquoi, en une ligne

Le supplément de cible dont une cellule a besoin pour tenir jusqu'à `F` vaut son **débit sortant**
`S_i`, c'est-à-dire `a_i / U_i` : il explose comme l'inverse du pas qu'on cherche à allonger. Là où
le pas est étranglé, la dose est inabordable — nourrir les six mille cellules de la première
itération au rythme `S_i` coûte **66 % de la masse totale**, pour 1 % de déplacement de direction et
un gain nul. Là où elle est abordable, le pas ne posait plus problème, et la direction déformée —
qui n'est plus celle de Newton — ralentit la descente.

Ce qui reste acquis et réutilisable : la calibration du § 20.6 (`U` pessimiste de ×72 à ×714), le
fait que `ModeleCellule` donne la limite de **n'importe quelle** direction sans géométrie nouvelle,
et le filet à deux passes de l'amortissement — qui servira à toute idée qui déforme la direction.

## 20.8 Ce qui reste ouvert

L'étude ferme la porte et dit pourquoi : le second membre ne contrôle que la divergence, l'extinction
est un fait de gradient, et le plafond harmonique — la meilleure chose qu'un second membre puisse
faire — est **sous 1** dans le régime difficile. Il reste deux ouvertures, et elles ne sont pas
mesurées :

* **doser sur la vraie limite plutôt que sur `U`.** `limites_masse` donne la limite exacte par
  bissection ; un balayage sur `κ` la paierait en calculs de cellule par essai, ce qui tue la
  promesse « une descente de plus, rien d'autre ». Il faudrait un doseur qui ne demande la vraie
  limite qu'**une fois**, et en déduise `κ` par le modèle plutôt que par balayage ;
* **le résidu pondéré** (§ 13.5, toujours pas mesuré) : `Σ C_i r_i²` ne change pas `d` mais change
  quels pas l'amortissement **accepte**. C'est le seul endroit de cette famille où des poids ne se
  simplifient pas — et c'est la seule case encore vide.

# 21. LE RÉSIDU ET LE JUGE : CE QUE `p` RÈGLE VRAIMENT (`newton --residu`, `--merite`, `--profil`)

`--residu log` gagnait 20 à 50 % de diagrammes sur Newton direct (§ 8.7.6) et on ne savait pas
pourquoi. La question posée ici est précise : **pourquoi `log x` ferait-il mieux que `x − 1/x` ?**
Elle a trois réponses, et la troisième était invisible parce que l'option en changeait deux à la
fois.

Les trois balayages sont dans `scripts/` (`matrice_merite.sh`, `scan_puissance.sh`,
`scan_relax.sh`). Seul `lines5_n100000_s0.005` est versionné ; les deux autres nuages se
régénèrent par `2d_des_familles/cases/gen_cases.py --sigma 0.02` (et `--sigma 0.1`), graine par
défaut.

## 21.1 Deux rôles confondus dans une seule option (`--merite`)

`--residu` changeait **le second membre de Newton** (le modèle local) **et le mérite de
l'amortissement** (la norme qui accepte le pas). Ce sont deux choses sans rapport. `--merite` les
sépare ; la matrice 3 × 3, `n = 10⁵`, `--pas essais`, diagrammes :

| cas | dir. `lin` | | | dir. `barriere` | | | dir. `log` | | |
|---|---|---|---|---|---|---|---|---|---|
| **juge** | lin | bar | log | lin | bar | log | lin | bar | log |
| 2D uniforme | 8 | 8 | 8 | 9 | 9 | 9 | **7** | **7** | **7** |
| 2D lignes `σ=0.1` | 13 | 13 | 13 | 19 | 18 | 18 | **11** | **11** | **11** |
| 2D lignes `σ=0.02` | 41 | 38 | *63* | *60* | 30 | *35* | **29** | **29** | **29** |
| 3D uniforme | 9 | 9 | 9 | 8 | 8 | 8 | **6** | **6** | **6** |
| 3D plans `σ=0.02` | 27 | 27 | 27 | *47* | 18 | *35* | **16** | **16** | **16** |

(*en italique* : sorti en STAGNATION loin de la tolérance — la comparaison de diagrammes n'y veut
rien dire.) Le verdict est net : **le long d'une direction donnée, le juge ne fait rien.** Les trois
colonnes d'un même bloc sont identiques chiffre pour chiffre, sauf quand le juge ne correspond pas à
la direction — et alors il ne fait que casser. Le gain de `log` est **entièrement dans la
direction**.

Le contrôle qui le confirme : `--merite pire`, c'est-à-dire l'amortissement jugé sur `max|a−ν|/ν`,
le critère d'arrêt lui-même (ce qui ruine la preuve de décroissance de KMT, mais on mesure).
Résultat : **rigoureusement les mêmes chiffres** — 13 / 27 / 41 diagrammes à `σ = 0.1 / 0.02` pour
`p = 1`, comme avec le mérite `l²`. Remplacer la norme ne change rien parce que **la clause du
mérite ne mord jamais** : c'est le plancher d'aire qui décide de tout (ce que `--refus` disait déjà
au § 8.7.4, ici confirmé hors du départ réparé).

## 21.2 Le profil le long de la direction (`--profil K`)

`--profil K` balaye `t` à l'itération `K` et imprime les quatre mérites côte à côte. Lignes
`σ = 0.005`, itération 0, départ Voronoï (`max|a−ν|/ν = 1666`, mérite `lin` 6.29e-2) :

| | `t = 1` | | premier `t` sans vide | |
|---|---|---|---|---|
| direction | vides | mérite `lin` | `t` | `max\|a−ν\|/ν` |
| `lin` | **50 030** | 1.19e-2 | 3.9e-3 | 1659 (−0.4 %) |
| `barriere` | 13 299 | 1.47e-2 | 6.2e-2 | 1564 (−6 %) |
| `log` | 22 376 | 1.27e-2 | 6.2e-2 | **1380 (−17 %)** |

Deux choses s'y lisent d'un coup.

**Le mérite `lin` récompense le pas qui vide la moitié du diagramme.** Il *décroît à tous les pas
essayés*, et son minimum est en `t = 1`, là où 50 030 cellules sur 100 000 sont vides. La raison est
arithmétique : avec `ν` uniforme, une cellule affamée ne peut coûter que `ν` au mérite (elle est
bornée par zéro), une gloutonne coûte `(x−1)ν` sans borne. Une gloutonne à `x = 50` pèse donc autant
que **2 401 cellules vides**. C'est pour ça que l'amortissement KMT a besoin d'un plancher d'aire
*dur* : sans lui, son propre mérite pousserait Newton dans un diagramme dégénéré.

**Et le pas admissible, lui, dépend violemment de la direction** : `lin` ne survit qu'à
`t = 3.9e-3` — huit reculs — pour un gain de 0.4 % sur le pire écart, quand `log` passe à
`t = 6.2e-2` et gagne 17 %.

## 21.3 L'algèbre : `p` est la fraction de l'écart LOGARITHMIQUE qu'on réclame

Newton sur `g(a_i/ν_i) = c` résout `L d = b` avec `b_i = ν_i/g'(x_i)·(c − g(x_i))`, et `b_i` est le
changement d'aire **demandé**. Pour la famille des puissances `g_p(x) = (xᵖ − 1)/p`
(`g_p' = x^{p−1}`), le calcul se ferme exactement (à `c` près, qui ne sert qu'à faire sommer `b` à
zéro) :

```
b_i = ( nu_i x_i^(1-p) - a_i ) / p
```

Autrement dit **la cible de la cellule `i` n'est plus `ν_i` mais `ν_i x_i^{1−p} = ν_i^p a_i^{1−p}`,
l'interpolée géométrique entre son aire actuelle et sa cible.** En échelle logarithmique, l'écart
restant après la demande vaut `(1−p) log x_i` : **`p` est exactement la fraction de l'écart
logarithmique que le pas réclame, la même pour toutes les cellules.** `p = 1` réclame tout,
`p = 0.5` la moitié, `p → 0` une fraction infinitésimale — et c'est la limite `log`.

Ce que ça donne aux deux bouts, et la comparaison avec `x − 1/x` :

| | gloutonne `x ≫ 1` | affamée `x ≪ 1` |
|---|---|---|
| `lin` (`p = 1`) | `−a` | `+ν` — **son déficit ENTIER** |
| `g_p`, `p < 1` | `−a/p` | `+ν x^{1−p}/p` → **0** |
| `barriere` (`x − 1/x`) | `−a` (à l'identique de `lin`) | `+a` |

**Voilà la réponse à la question posée.** `x − 1/x` ne répare **qu'un côté** : il adoucit la demande
des affamées, mais sa courbure relative `g''/g' = −2/(x³+x)` s'éteint en `x⁻³` sur les gloutonnes —
il *redevient* `lin` exactement là où le problème est dur. La famille `(xᵖ−1)/p`, elle, tient les
deux queues avec le même exposant.

Et pourquoi la demande des affamées est celle qui coûte : **elle est inexauçable.** Une cellule
affamée n'est pas affamée parce que son poids est trop bas, elle l'est parce que ses voisines ont un
poids trop haut ; elle ne grandira que quand celles-là baisseront. Lui réclamer `ν` d'un coup, c'est
demander au système linéaire quelque chose hors de son domaine de validité, et **ça pollue toute la
direction** — ce que l'amortissement ne peut pas réparer, puisqu'il ne sait que multiplier la
direction entière par un scalaire. C'est aussi pourquoi le gain croît avec la difficulté : la
dynamique de `x` au départ est de 1 à 2 000, donc l'écart logarithmique est grand, et c'est lui que
`p` comprime.

Symétriquement, `p < 0` donne une cible `ν x^{1−p}` **au-delà** de la valeur actuelle : à
`p = −0.5`, une gloutonne à `x = 2160` se voit assigner `x^{1.5} ≈ 10⁵`, donc un `b` de 90 fois son
aire. Le dépassement croît avec `|p|` — et la dégradation mesurée est monotone.

## 21.4 Le balayage de l'exposant (`--residu puissance --puis P`)

`p = 1` **est** `lin` et `p = 0` **est** `log` — la famille recouvre les deux options discrètes, et
les chiffres le vérifient (8 / 13 / 41 / 9 / 27 diagrammes à `p = 1`, identiques à la colonne
`lin/lin` de la matrice ; seul `σ = 0.005` diffère d'un cheveu, 113 contre 116, parce que le mérite
de la famille est centré et celui de `LIN` ne l'est pas). Diagrammes, `n = 10⁵`, `--pas essais` :

| `p` | 2D unif. | 2D `σ=0.1` | 2D `σ=0.02` | 3D unif. | 3D plans |
|---|---|---|---|---|---|
| **1** (= `lin`) | 8 | 13 | 41 | 9 | 27 |
| 0.9 | 7 | 17 | 32 | 7 | 22 |
| 0.75 | 6 | 14 | 30 | 7 | 19 |
| 0.5 | **5** | **11** | 32 | **5** | **16** |
| 0.25 | 6 | 16 | **27** | **5** | 17 |
| 0.1 | 6 | 13 | 29 | **5** | **16** |
| **0** (= `log`) | 7 | **11** | 29 | 6 | **16** |
| −0.25 | 9 | 16 | — | 6 | 18 |
| −0.5 | 10 | 22 | — | 8 | 20 |
| −1 | 11 | 37 | — | 10 | 24 |
| −2 | 17 | 48 | — | 14 | 39 |

La forme est la même partout : **une pente franche de `p = 1` vers `p ≈ 0.25`, un plateau large sur
`[0, 0.5]`, et une dégradation monotone sous zéro.** Donc `log` n'est pas un bout de chemin, c'est
un point sur un plateau — et sur trois des cinq cas l'optimum est *à l'intérieur*, en `p = 0.5` ou
`0.25`, jamais en `p = 0` seul. Le gain le plus net est celui qui compte : le cas 3D dur passe de 27
à 16 diagrammes (−41 %), le 2D dur de 41 à 27 (−34 %).

Sur `σ = 0.005` ces chiffres-là ont été mesurés **avant** la déduplication du nuage : tous les essais
sortaient en STAGNATION vers 2.4e-6, le plancher du nuage et non celui de la méthode. À plancher égal
le classement tenait déjà (113 diagrammes à `p = 1`, 90 à `p = 0.5`, 84 à `p = 0.25`, 73 à `p = 0`), et
**sur le nuage dédupliqué (§ 23.7) il est plus net encore**, tout convergeant :

| `p` | 1 | 0.9 | 0.75 | 0.5 | 0.25 | 0.1 | 0 |
|---|---|---|---|---|---|---|---|
| diagrammes | 78 | 64 | 64 | 56 | 50 | 44 | **39** |

**78 → 39, un facteur deux** — le plus fort gain de l'exposant sur tout le banc, et c'est le cas le
plus dur qui le donne. La dégénérescence le masquait.

## 21.5 La relaxation à la main, sous `essai-limites` (`--facteur`)

La passe des limites rend `alpha*`, le pas exact qui met la première cellule au plancher, et retient
`facteur · alpha*`. Ce `facteur` **est** le coefficient de relaxation, et il n'avait jamais été
balayé contre l'exposant. Lignes `σ = 0.02`, `--pas essai-limites`, diagrammes (aucun recul, dans
tout le tableau) :

| `facteur` | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| `p = 1` | 32 | 26 | 22 | 26 | **20** | 32 |
| `p = 0.5` | 25 | 20 | 18 | 16 | **15** | **15** |
| `p = 0` | 17 | 15 | 14 | **13** | **13** | 14 |

Trois choses, dont une qui n'était pas attendue.

**Les deux mécanismes se composent, presque proprement.** Le pas par les limites fait 41 → 20 avec
`lin`, l'exposant fait 20 → 13 par-dessus : **13 diagrammes contre 41 pour la référence**, un
facteur 3.2, et la meilleure valeur absolue de tout ce banc sur ce cas.

**Et le bon résidu rend la relaxation presque indifférente.** À `p = 1` la courbe est irrégulière et
pointue (32 → 20 → 32, avec un creux parasite à 0.8) : il y a un réglage à trouver, et il est
étroit. À `p = 0` elle est plate entre 0.7 et 0.99 (15 → 13 → 14). C'est cohérent avec le § 21.3 :
quand la direction demande l'inexauçable, *jusqu'où* on la suit devient critique ; quand elle
demande une fraction de l'écart logarithmique, la longueur du pas cesse d'être le paramètre
sensible. **Un réglage de moins à porter, ce qui vaut plus que les trois diagrammes gagnés.**

## 21.6 Plusieurs directions et leur modèle : le span vaut mieux, mais pas selon le mérite

Trois résidus donnent trois directions, et **elles se résolvent sur la même factorisation** — deux
descentes de plus, aucun diagramme, aucun assemblage. Si le modèle polynomial de l'aire (§ 7) était
étendu à plusieurs directions, il donnerait `a_i` sur tout leur span sans diagramme : à combinatoire
figée chaque sommet est affine en `w`, donc l'aire est **quadratique** en les coefficients du
mélange (cubique en 3D), avec les termes croisés. La question qui décide s'il faut le construire :
**le span contient-il nettement mieux que ses bouts ?**

`--combi K` répond à l'itération `K` à la force brute. Lignes `σ = 0.02`, itération 0, pour chaque
mélange le plus grand pas admissible et le pire écart qu'il atteint (départ 2159) :

| `λ` (lin, log, bar) | `t` admis | `max\|a−ν\|/ν` |
|---|---|---|
| 1, 0, 0 (`lin` pur) | 6.2e-2 | 2025 |
| 0, 1, 0 (`log` pur) | 1.2e-1 | 1072 |
| 0, 0, 1 (`barriere` pur) | 2.5e-1 | 1640 |
| **0.25, 0.75, 0** | 2.5e-1 | **602** |
| 0, 0.75, 0.25 | 2.5e-1 | 602 |

Le mélange ¼ `lin` + ¾ `log` fait **1.8 fois mieux que la meilleure direction pure**, et le
mécanisme se lit : `log` pur ne survivait pas à `t = 0.25`, et c'est l'ajout de `lin` qui le permet.
Normal — `lin` est précisément la direction qui *nourrit les affamées* (§ 21.3). **Les deux rôles
sont complémentaires : `log` fait le travail sur les gloutonnes, `lin` protège le plancher.** Aucun
résidu seul ne peut exprimer ça.

`--oracle Q` pousse jusqu'au bout : à *chaque* itération, le meilleur mélange du simplexe, choisi à
la force brute. Ce n'est pas un algorithme (~50 diagrammes par itération pour choisir) mais la
**borne supérieure** de ce que le modèle rendrait. Itérations, `n = 10⁵` :

| cas | meilleure direction pure | oracle, choix sur le **mérite** | oracle, choix sur **`max\|a−ν\|/ν`** |
|---|---|---|---|
| 2D `σ = 0.1` | 7 | 7 | **6** |
| 2D `σ = 0.02` | 11 | *13* | **8** |
| 2D `σ = 0.005` | 15 (stagne à 2.4e-6) | *18* (stagne) | **15, et CONVERGE à 6.8e-7** |

**Et c'est le critère de sélection qui décide, pas le mélange.** Choisi sur le mérite `l²`, l'oracle
fait *moins bien* que la direction pure (13 contre 11) : il a tout le span à sa disposition et il
choisit mal. Choisi sur `max|a−ν|/ν`, il gagne un tiers des itérations — et sur le nuage dégénéré
`σ = 0.005` il est **le seul essai de tout ce banc qui converge**, là où toutes les directions pures
stagnent à 2.4e-6.

Le contrôle du § 21.1 verrouille l'interprétation : changer le juge pour `pire` sur une direction
*seule* ne change **rien** (mêmes chiffres exactement). Donc les deux tiers d'itérations gagnés ici
viennent bien du **mélange**, et le critère `pire` n'est nécessaire que pour *choisir dedans* — là
où il y a un vrai choix à faire. Dit autrement : **le mérite `l²` est inerte quand il n'y a qu'une
direction, et nuisible dès qu'il y en a plusieurs.**

C'est donc positif, et c'est la suite : le modèle polynomial multi-directions prédirait `a_i` sur le
span, donc `max|a−ν|/ν` sur le span, donc exactement le critère qui marche — sans diagramme.

> **CONSTRUIT ET MESURÉ AU § 22.** Il prédit le critère à `0.01 %` près sept itérations sur dix, la
> recherche dans le span est bien gratuite (0.006 s par itération), et il fait 11 diagrammes à
> `σ = 0.02` contre 41 pour la référence et 16 pour le même code privé de son span. Le risque nommé
> ici s'est réalisé, mais par excès et non par défaut : le polynôme est **pessimiste** sur le plancher
> (il annonce `alpha* = 0.118` quand l'exact vaut `0.273`), ce qui est le bon sens de l'erreur. Ce qui
> n'était pas prévu, c'est que le coût ne serait pas la recherche mais **le balayage global** qui bâtit
> le modèle — un diagramme par itération — et que ça suffit à laisser `essai-limites --residu log`
> devant en temps de paroi (5.1 s contre 7.5 s).

## 21.7 Plus simple que le span : résoudre en `log` puis basculer sur `lin` (`--bascule-residu`)

Les deux bouts ne servent pas au même moment. `log` ne réclame à une cellule affamée qu'une fraction
de son écart *logarithmique*, ce qu'il faut tant que la dynamique de `a/ν` est de 1 à 2000 (§ 21.3) ;
`lin` est le vrai Newton du problème et c'est lui qui donne la convergence quadratique à la fin. D'où
l'idée, qui n'a besoin ni de span ni de modèle : **`log` au départ, `lin` pour finir**, avec une
bascule quand `max|a−ν|/ν` descend sous un seuil.

Elle ne peut pas nuire tard, et pour une raison algébrique. Près de la solution `g(x) ≈ g'(1)(x−1)`
et `g'(x) ≈ g'(1)`, donc

```
b_i = ν_i / g'( x_i ) · ( c − g( x_i ) )   →   ν_i − a_i        pour TOUT résidu
```

**Tous les résidus deviennent la même direction de Newton au premier ordre.** Le choix n'a donc plus
d'objet là où il ne sert plus — ce qui se voit aussi sur les `λ` que le modèle du § 22 retient tout
seul : `log` pur pendant cinq itérations, puis `lin`, puis n'importe quoi (les directions sont
colinéaires). La bascule est la version explicite et gratuite de ce que le modèle découvre.

`--bascule-residu R`. Elle est **latchée** (`max|a−ν|/ν` n'est pas monotone) et se décide **avant le
second membre**, donc avant le mérite de l'itération : `b`, `nr` et `n2r` parlent tous du même
résidu — sans ça le premier pas se calcule avec un résidu et se juge avec un autre. Diagrammes,
`--pas essai-limites --facteur 0.9`, `n = 10⁵` (`R = 0` : jamais ; `R = 10⁹` : dès l'itération 0, donc
`lin` pur) :

| `R` | uniforme | `σ = 0.1` | `σ = 0.02` | `σ = 0.005` |
|---|---|---|---|---|
| **0** — `log` pur | 8 | 8 | 13 | 20 |
| 0.5 | **7** | 8 | 13 | **19** |
| 2 | **7** | **7** | 13 | **19** |
| 10 | **7** | **7** | 13 | **19** |
| 50 | **7** | 8 | **12** | 20 |
| 200 | **7** | 8 | 14 | 22 |
| 1000 | **7** | 10 | 18 | 33 |
| **10⁹** — `lin` pur | **7** | 10 | 26 | 30 |

**Ça marche, le seuil n'est pas critique, et le gain est modeste.** La fenêtre `R ∈ [0.5, 10]` est large
et plate, et elle gagne un diagramme sur trois des quatre cas — sans rien coûter, et en supprimant un
réglage (voir plus bas).

> **CORRECTION.** La colonne `σ = 0.005` annonçait d'abord 536 diagrammes pour `log` pur contre 53 avec
> la bascule — « un facteur dix », et j'en avais fait le vrai gain de l'idée. C'était **un artefact du
> nuage dégénéré** : `log` s'acharnait 486 reculs sur une cellule que rien ne pouvait réparer, puisque
> deux germes y étaient confondus. Sur le nuage dédupliqué (§ 23.7) la même colonne fait 20 contre 19.
> La bascule reste utile — un diagramme partout, un réglage de moins — mais son gain est du même ordre
> sur tous les nuages, et le facteur dix n'existait pas.

Et **la bascule simplifie la relaxation** au lieu de la compliquer — c'était la crainte inverse.
Diagrammes à `σ = 0.02` contre `--facteur` :

| `facteur` | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| `log` pur | 17 | 15 | 14 | **13** | **13** | 14 |
| `log` + bascule 2 | 17 | 15 | **13** | **13** | **13** | **13** |

Le plateau passe de `[0.9, 0.95]` à `[0.8, 0.99]` : une fois la fin confiée à `lin`, qui prend `t = 1`
de lui-même, la longueur du pas de la phase `log` cesse d'être un réglage fin.

En temps de paroi (`job -b`, Cholesky, `σ = 0.02`), c'est **le plus rapide de tout ce banc** :

| | diagrammes | temps |
|---|---|---|
| `lin` / essais | 41 | 14.0 s |
| `log` / limites | 13 | 5.13 s |
| **`log` + bascule 2** | 13 | **5.04 s** |
| modèle `K = 3` (§ 22) | **11** | 7.97 s |

Donc les deux approches ne se remplacent pas, elles se partagent le terrain : **sur un cas normal la
bascule gagne** (aussi peu de diagrammes que `log`, le meilleur temps, et deux lignes de code contre
un polynôme multivarié) ; **sur le nuage dégénéré le modèle gagne largement** (13 diagrammes, 11.8 s
et il CONVERGE, contre 53, 23.1 s et STAGNATION). La bascule est ce qu'il faut mettre par défaut ; le
modèle est ce qu'il faut sortir quand ça ne passe pas.

> **Ce que la dégénérescence faisait aux comparaisons de solveurs.** Sur l'ancien nuage sale, `log` pur
> faisait 536 diagrammes avec AMGCL et 58 avec Cholesky : les poids de laplacien montaient à 1.4e4 contre
> une médiane de 0.29 (§ 23.2), et un Krylov à tolérance relative n'y rendait pas la même direction
> qu'une factorisation. Comparer deux variantes à solveur différent n'y voulait donc rien dire. Sur le
> nuage dédupliqué le problème disparaît — c'est l'une des raisons de la bascule du § 23.7.


# 22. LE MODÈLE POLYNOMIAL MULTI-DIRECTIONS (`Ecrasement.h` : `PolyMulti`, `newton --pas modele`)

Le § 21.6 laissait une ouverture mesurée mais pas construite : le span de plusieurs directions
contient des pas bien meilleurs qu'aucune direction seule, mais les trouver coûtait un diagramme par
essai. Voici l'objet qui les trouve sans diagramme.

## 22.1 L'objet : l'aire sur un span, exacte à combinatoire figée

C'est la même algèbre que `PolyCellule` (§ 7), avec `K` décalages au lieu d'un. Le décalage de chaque
coupe est affine en `t = (t_1 … t_K)`, un sommet est l'intersection de deux droites donc **linéaire en
leurs décalages**, et l'aire — une somme de produits vectoriels de sommets — est **quadratique en `t`,
termes croisés compris** (cubique en 3D) :

```
A_i( t ) = c0 + sum_k g_k t_k + sum_{k >= l} q_kl t_k t_l          1 + K + K(K+1)/2 coefficients
```

`PolyMulti` les porte, plus le gradient en forme fermée — c'est lui qui rend une vraie descente
possible au lieu d'une grille — et `rayon`, ce que `alpha_arete` devient sur un span : le rayon en
norme infinie sous lequel **aucune arête ne s'annule**, donc sous lequel le polynôme est exact,
`min_j l0_j / Σ_k |l^k_j|`. Attention, ça ne couvre qu'**un** des deux modes de rupture : un germe qui
n'était pas voisin peut le devenir sans qu'aucune arête présente ne disparaisse, et ça ne se voit pas
depuis la cellule seule.

Les `K` directions viennent des `K` résidus, et **elles se résolvent sur la même factorisation** :
`K − 1` descentes de plus, aucun assemblage, aucun diagramme.

## 22.2 Le contrôle algébrique tombe au dernier chiffre (`--modele K`)

`--modele K` bâtit le modèle à l'itération `K` et le confronte à **deux** témoins qui ne disent pas la
même chose : `ModeleCellule::aire`, qui évalue la même aire à combinatoire figée mais point par point
(contrôle **purement algébrique** — s'il ne tombe pas, les coefficients sont faux), puis le **vrai
diagramme** (l'erreur de combinatoire, la seule qui décide). Lignes `σ = 0.02`, itération 0, `K = 3`,
erreurs en unités de `ν` :

| `‖t‖∞` | % prouvé exact | contre l'algèbre, méd. | contre le réel, méd. | contre le réel, max | `max\|a−ν\|/ν` prédit / réel | sous `eps` prédit / réel |
|---|---|---|---|---|---|---|
| 6.2e-2 | 27 % | 9.1e-13 | **1.9e-12** | 1.0e-1 | 2025.3 / 2025.3 | 0 / 0 |
| 1.9e-1 | 1.6 % | 9.7e-13 | 3.0e-4 | 9.6 | 1230.5 / 1230.5 | 1 / 0 |
| 2.5e-1 | 0.6 % | 9.1e-13 | 2.3e-3 | 6.4 | 1637.6 / 1637.6 | 156 / 18 |
| 1.0 | 0 % | 1.1e-12 | 4.6e-1 | 2.5e+3 | 2533.7 / **21.3** | 15 777 / 47 576 |

Le contrôle algébrique est à `1e-12 ν`, soit `1e-17` absolu : **les coefficients sont justes**. Contre
le réel, il y a un régime : sous `‖t‖∞ ≈ 0.06` l'erreur médiane est au niveau machine, à `0.19` elle
est de `3e-4 ν` et le critère est juste **à cinq chiffres**, et au-delà de `0.5` le modèle ne vaut
plus rien.

Deux remarques qui comptent pour la suite. D'abord `rayon` est très **pessimiste** : à `‖t‖∞ = 6e-2`
il ne certifie que 27 % des cellules alors que l'erreur médiane est au niveau machine — normal, il
borne le pire cas sur toute la boule, pas le long du rayon. Ensuite, et c'est ce qui rend la recherche
sûre, **le modèle est conservateur sur le critère** : dès que la combinatoire d'une cellule casse, son
aire prédite part n'importe où, donc le `max` prédit **explose** (2534 annoncé contre 21 en vérité).
La recherche fuit donc d'elle-même les régions où le modèle ne vaut rien.

## 22.3 Le pas cherché dans le span (`--pas modele`)

Pour un `λ` **fixé** le modèle se réduit à une quadratique **scalaire** en `t` : le plus grand pas qui
respecte le plancher est donc une **racine**, exactement comme pour une direction seule, pas une
échelle dyadique. Et une fois `alpha*(λ)` connu, évaluer le critère à plusieurs fractions de ce pas ne
coûte rien — c'est le profil du § 21.2, gratuit, et **le coefficient de relaxation du § 21.5 choisi par
la mesure au lieu d'être réglé à la main**. Deux passages sur les cellules en tout : les racines, puis
le critère des 75 candidates, jugées pendant que les coefficients de la cellule sont en registres.

Ce que ça prédit, itération après itération : `max|a−ν|/ν` annoncé à l'itération `k` contre celui
mesuré à `k + 1`, lignes `σ = 0.02` :

| it | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| écart | +0.00 % | −0.01 % | −0.00 % | **+4.3 %** | **+6.3 %** | **+3.8 %** | +0.01 % | −0.00 % | +0.01 % | +0.01 % |

Sept itérations sur dix prédites à `0.01 %` près, et les trois écarts de quelques pour cent sont
exactement celles où le pas est le plus long (`0.40`, `0.74`, `0.99`).

Et le bilan en diagrammes, `n = 10⁵`, contre toutes les références des sections précédentes
(`scripts/bilan_modele.sh`) :

| variante | uniforme | `σ=0.1` | `σ=0.02` | `σ=0.005` (dédupliqué, § 23.7) |
|---|---|---|---|---|
| `lin` / essais — la référence du banc | 8 | 13 | 41 | 78 |
| `p = 0.25` / essais (§ 21.4) | 6 | 16 | 27 | 50 |
| `lin` / limites, relaxation 0.95 (§ 21.5) | 7 | 10 | 20 | 31 |
| `log` / limites, relaxation 0.9 (§ 21.5) | 8 | **8** | 13 | 20 |
| **modèle `K = 1`** — LE CONTRÔLE, pas de span | 7 | 11 | 16 | 25 |
| **modèle `K = 2`** — le défaut | 6 | **8** | 11 | **13** |
| modèle `K = 3` | 6 | **7** | 11 | 14 |
| modèle `K = 3` + limites exactes | **5** | **7** | **10** | **13** |

Le contrôle `K = 1` est ce qui rend le tableau lisible : c'est le même code, le même critère, le même
choix de pas, mais **un seul vecteur dans le span** — donc tout l'écart entre sa ligne et celle de
`K = 3` est le span et rien d'autre. Il vaut 16 → 11 diagrammes à `σ = 0.02` (−31 %) et 11 → 7 à
`σ = 0.1` (−36 %). Contre la référence du banc, 41 → 11 (−73 %).

Sur `σ = 0.005` — le cas le plus dur, une fois dédupliqué — le modèle fait **13 diagrammes contre 78**
pour la référence et 20 pour la meilleure combinaison réglée à la main, et **sans un seul recul**.

> **CORRECTION.** Ce paragraphe disait que le modèle à `K = 3` était « le seul essai de tout ce banc qui
> converge » sur ce nuage, en 13 diagrammes contre 47 pour `K = 2`, et j'en avais tiré qu'il fallait les
> trois directions. C'était **entièrement la dégénérescence** : sur le nuage dédupliqué tout converge,
> `K = 2` fait 13 et `K = 3` en fait 14. Le défaut est donc revenu à **deux directions**, ce qui économise
> une résolution linéaire par itération — la troisième ne gagne qu'un diagramme sur `σ = 0.1` et en perd
> un ici. C'est à prendre avec la prudence qu'un cas unique mérite :
il faut les **trois** ingrédients à la fois (trois directions, le pas du modèle, le garde-fou du
§ 22.4), chacun retiré ramène la stagnation, et une conjonction aussi serrée sur un seul nuage ne fait
pas une loi.

## 22.4 Un `L∞` ne doit pas avoir d'otage (`--mod-hors`)

Le § 21.6 avait établi que le mélange doit être choisi sur `max|a−ν|/ν` et pas sur le mérite `l²`.
Premier essai avec le maximum **strict** : les trois cas sains vont bien, et `σ = 0.005` explose —
**1 139 diagrammes, 1 078 reculs, MAX ITERATIONS**. Le diagnostic est direct : un `L∞` est otage de
**la** cellule que rien ne peut réparer. Le choix du mélange devient du bruit, et l'amortissement
refuse tout.

Deux parades mesurées :

| juge du modèle | uniforme / `σ=0.1` / `σ=0.02` | `σ=0.005` **sale** (le cas qui a révélé le problème) |
|---|---|---|
| maximum strict | 5 / 7 / 10 | **1 139**, MAX IT |
| mérite `log` (robuste, mais plus d'extrême) | 6 / 8 / 13 | 124 |
| **`k`-ième pire, `k = 10⁻⁴ n`** | **5 / 7 / 10** | **51** |

Le mérite `log` répare le cas dégénéré mais coûte 30 % sur les cas sains — il a perdu la nature
extrémale qui faisait trouver les bons pas. Enjamber une poignée d'aberrantes garde les deux :
**rigoureusement les mêmes chiffres que le maximum strict** sur les cas sains, et le blocage disparaît.
C'est le défaut (`mod_hors = 1e-4`, soit 10 cellules à `n = 10⁵`).

## 22.5 En temps, c'est l'incumbent qui gagne — et on sait pourquoi

Le § 21 comptait les diagrammes, comme demandé. En temps de paroi, `job -b`, lignes `σ = 0.02`,
`n = 10⁵`, un fil :

| variante | diagrammes | Cholesky | AMGCL |
|---|---|---|---|
| `lin` / essais | 41 | 14.5 s | 19.2 s |
| `log` / `essai-limites` 0.9 | 13 | **5.1 s** | **8.0 s** |
| modèle `K = 3` | 11 | 7.5 s | 14.0 s |
| modèle `K = 3` + limites exactes | 10 | 10.9 s | 16.6 s |

**Le modèle bat largement la référence du banc et perd contre `essai-limites --residu log`.** Le
partage du coût le dit exactement (Cholesky, par itération) :

* bâtir le modèle : **0.27 s, soit un diagramme** — il parcourt toutes les cellules, comme un
  diagramme ;
* **chercher dans le span : 0.006 s.** La promesse est tenue, la recherche est gratuite (mesuré en
  retombant à une seule candidate, `--mod-q 1` : le temps ne bouge pas) ;
* la passe des limites exactes, quand on la demande : 0.41 s, soit 1.6 diagramme.

Autrement dit **le prix n'est pas la recherche, c'est le balayage global**. Et `essai-limites` gagne
précisément parce qu'il ne balaye jamais : il essaye un pas et ne calcule les limites que des cellules
qui ont cassé — sa passe coûte 0.057 s **au total** contre 2.8 s pour bâtir le modèle. Le prix du span
lui-même, les `K − 1` résolutions de plus, est le poste secondaire et il dépend du solveur : presque
gratuit avec une factorisation directe (une descente triangulaire), 0.4 s par direction avec un Krylov.
C'est la première fois dans ce banc que Cholesky est le bon choix en 2D à `n = 10⁵`, et c'est pour
cette raison-là.

La suite est donc nommée, pas faite : **bâtir le modèle paresseusement**, seulement pour les cellules
qui peuvent mordre — ce que `essai-limites` sait déjà trouver. Le point dur est que le critère `max`
a besoin de toutes les cellules, mais le `k`-ième pire du § 22.4, lui, n'a besoin que des `k`
premières : il y a peut-être là de quoi ne jamais toucher les autres. Ce n'est pas une ligne de code.

Et comme tout `Ecrasement.h`, **c'est 2D seulement** : en 3D le volume est cubique en `t` et la marche
dans la cellule est autre ; `--pas modele` y retombe sur les essais, en le disant.

# 23. Y A-T-IL QUELQUE CHOSE À CONDITIONNER ? (`newton --diag-lap`)

Question posée : maintenant que le résidu et le pas ont été travaillés, pourrait-on **mieux
conditionner le système** pour que le multigrille y passe mieux ? La réponse est non, et c'est la
mesure qui le dit — mais elle désigne autre chose, qui paye davantage.

## 23.1 Le résidu ne touche pas la matrice

Premier point, et il ferme une porte avant qu'on l'ouvre. La matrice assemblée est toujours

```
c_ij = |facette ij| / ( 2 |p_i − p_j| ),   L_ii = Σ_j c_ij,   L_ij = −c_ij
```

Elle ne dépend **que des facettes**. Le résidu (`lin`, `log`, `puissance`, la bascule du § 21.7) ne
change que le **second membre** : `L d = ν/g'(x) · (c − g(x))`. Donc rien de tout le § 21 ne peut ni
améliorer ni dégrader le conditionnement — et les chiffres le confirment : ~35 itérations de Krylov
par résolution dans toutes les variantes.

## 23.2 Ce qui rend `L` dure, mesuré (`--diag-lap`)

`--diag-lap` imprime à chaque itération l'étalement de la diagonale, celui des poids d'arêtes, et
l'**anisotropie par ligne** `max_j c_ij / Σ_j c_ij` — une ligne proche de 1 est un nœud couplé à un
seul voisin, donc une chaîne, et c'est ce qui met une agrégation en échec. Au départ Voronoï,
`n = 10⁵` :

| cloud | diag méd. | diag max/min | `c` méd. | `c` max | anisotropie méd. | lignes > 0.9 |
|---|---|---|---|---|---|---|
| uniforme 2D | 2.61 | 1.4e2 | 0.287 | 1.4e2 | 0.440 | 0.68 % |
| lignes `σ=0.1` | 2.60 | 3.5e2 | 0.286 | 3.2e2 | 0.439 | 0.75 % |
| lignes `σ=0.02` | 2.60 | 2.0e2 | 0.289 | 1.8e2 | 0.438 | 0.69 % |
| lignes `σ=0.005` | 2.65 | **1.5e4** | 0.293 | **1.4e4** | 0.442 | 1.06 % |
| uniforme 3D | 0.071 | 5.1e1 | 2.1e-3 | 0.86 | 0.305 | 0.04 % |
| plans `σ=0.02` | 0.048 | 2.2e2 | 1.5e-3 | 1.4 | 0.304 | 0.04 % |

**L'anisotropie est identique partout**, y compris sur l'uniforme : 0.44 en 2D, 0.30 en 3D, et moins
de 1.1 % de lignes au-dessus de 0.9. Le graphe de Laguerre n'est pas une chaîne, c'est un graphe
régulier à six voisins — l'hypothèse « les nuages durs font un problème anisotrope » est **fausse**.

Et sur toute une résolution (`σ = 0.02`, `log` + bascule, AMGCL), le compte de Krylov est **plat** :

| it | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Krylov | 34 | 35 | 34 | 36 | 35 | 39 | 36 | 34 | 31 |
| anisotropie | 0.438 | 0.442 | 0.445 | 0.449 | 0.453 | 0.456 | 0.459 | 0.459 | 0.459 |

35 par résolution du début à la fin — et l'uniforme en coûte 35 aussi. **Il n'y a rien à
conditionner : le solveur voit partout la même matrice facile.**

## 23.3 Sauf sur un nuage, et la cause est un défaut de données

Une seule ligne du tableau sort : `σ = 0.005`, où `c` monte à **1.4e4** contre une médiane de 0.29.
Cinq décades, sur une poignée d'arêtes. La cause est connue (§ 8.7.3) : `gen_cases.py` rabat les
germes dans le carré unité, et à petit `σ` le rabattement en empile — **56 paires à moins de 1 % de
l'espacement médian, dont une à 1.009e-08**. Comme `c_ij = |facette| / 2|p_i − p_j|`, une distance de
`1e-8` donne un poids de `1e4`.

Et là, effectivement, les solveurs itératifs souffrent : à huit fils, `σ = 0.005`, partie linéaire —
Cholesky **2.15 s**, AMG agrégation 3.34 s, Ruge-Stüben 3.28 s, multigrille maison 4.51 s. Le direct
gagne, ce qui n'arrive nulle part ailleurs en 2D à ce `n`. Pire, un lissage de plus (`--mg-nu 2`)
fait passer la partie linéaire de 4.5 s à **50 s**.

**La réparation n'est donc pas un préconditionneur, c'est le nuage.**
`cases/nettoie_germes.py` retire d'une grappe tous les germes sauf le premier (on ne déplace personne :
retirer un germe confondu avec son voisin ne change aucune cellule visible). 56 germes sur 100 000, et :

| | `c` max | diag max | Newton (`log` / limites) | partie linéaire à 8 fils (chol / amg / mg) |
|---|---|---|---|---|
| sale | 1.4e4 | 1.4e4 | 53 diag, **STAGNATION** à 2.35e-6 | 2.15 / 3.34 / 4.51 s |
| **propre** | **1.3e2** | **1.4e2** | **19 diag, CONVERGE** à 2.7e-8 | **1.97 / 2.65 / 2.94 s** |

L'étalement des poids est divisé par cent, le cas **converge** au lieu de plafonner, et les diagrammes
passent de 53 à 19. Les solveurs itératifs sont ceux qui gagnent le plus (−21 % pour AMG, −34 % pour
le multigrille maison, contre −7 % pour Cholesky) : c'était bien eux que l'étalement pénalisait. Ils
ne dépassent pas Cholesky pour autant sur ce nuage — sur les *lignes* le direct gagne de toute façon
(§ 3), et c'est l'uniforme qui est le terrain de l'AMG.

## 23.4 Le piège de mesure : toute comparaison de solveurs se fait au bon nombre de fils

Il faut l'écrire, parce qu'on y tombe. À **un** fil, sur `σ = 0.02` (partie linéaire) : Cholesky
1.54 s, Ruge-Stüben 3.62 s, agrégation 4.25 s, multigrille maison 6.06 s — Cholesky domine, et
Ruge-Stüben bat l'agrégation de 15 %. À **huit** fils, même cas : multigrille maison **1.32 s**,
agrégation 1.47 s, Cholesky 1.65 s, Ruge-Stüben **4.08 s**. Tout s'inverse :

* **Cholesky ne monte pas en charge du tout** (1.72 → 1.77 s de 1 à 8 fils sur l'uniforme). Il gagne
  à un fil et perd à huit, donc une comparaison mono-fil le fait paraître le meilleur solveur 2D
  jusqu'à `n = 10⁶`, ce qui est faux dès qu'on utilise la machine ;
* **le Gauss-Seidel de Ruge-Stüben se parallélise mal** : meilleur à un fil, presque trois fois pire
  à huit. Son gain en itérations (−33 à −61 % de Krylov, réel et croissant avec la difficulté) est
  mangé par le coût du lissage.

Les conclusions du § 17 tiennent donc telles quelles, et le défaut 2D (`amg`) reste le bon. Ce qui
change, c'est qu'on sait maintenant **pourquoi** il n'y a rien à gagner côté conditionnement, et
**où** est le vrai levier : les germes du nuage, pas la matrice.

## 23.5 Tous les nuages, relevés — et un seul était malade

`cases/nettoie_germes.py --essai` relève sans écrire. Espacement au plus proche voisin, `n = 10⁵`
sauf mention, seuil à 1 % de la médiane :

| nuage | min | médiane | min/médiane | paires |
|---|---|---|---|---|
| lignes `σ=0.005` | **1.009e-08** | 4.53e-04 | **2.2e-5** | 56 |
| lignes `σ=0.02` | 2.25e-06 | 8.22e-04 | 2.7e-3 | 2 |
| lignes `σ=0.05` | 4.92e-06 | 1.12e-03 | 4.4e-3 | 3 |
| lignes `σ=0.1` | 4.27e-06 | 1.31e-03 | 3.3e-3 | 5 |
| lignes `σ=0.005`, `n=2000` | 8.06e-06 | 3.18e-03 | 2.5e-3 | 1 |
| **plans `σ=0.02` (3D)** | 2.65e-04 | 7.63e-03 | 3.5e-2 | **0** |

Les nuages 3D sont **propres** : aucune paire sous le seuil. Et parmi les 2D, un seul est malade.
Newton par défaut, 8 fils, avant / après déduplication :

| σ | sale | propre |
|---|---|---|
| 0.1 | 9 it, 13 diag, CONVERGE 3.48e-8 | 9 it, 13 diag, CONVERGE 3.47e-8 |
| 0.05 | 13 it, 23 diag, CONVERGE 6.60e-8 | 13 it, 23 diag, CONVERGE 6.56e-8 |
| 0.02 | 18 it, 41 diag, CONVERGE 2.47e-10 | 18 it, 41 diag, CONVERGE 2.47e-10 |
| **0.005** | 24 it, **113** diag, **STAGNATION** 2.35e-6 | 23 it, **78** diag, **CONVERGE** 2.00e-7 |

**Trois nuages sur quatre : rien ne change**, au diagramme près. Ce qui est cohérent avec le § 23.2 :
leurs paires sont à ~0.3 % de l'espacement médian, donc `c_ij ≈ h / 2·dist` reste à 1e2–3e2, dans la
plage saine. Le seuil à 1 % de la médiane est **prudent, pas nécessaire** — il retire deux à cinq
germes inutilement. Le mal commence vers 0.2 % de la médiane, là où `c_ij` sort de la plage ; à
2.2e-5 de la médiane, `σ = 0.005` est quatre ordres de grandeur au-delà.

Deux précautions dans le script. Il ne **déplace** personne (retirer un germe confondu avec son voisin
ne change aucune cellule visible, le déplacer changerait le problème). Et sur un fichier `_equal`, qui
**porte la solution** et sert de témoin indépendant au banc, il **annule les poids** en le disant :
retirer un germe change le problème, donc les poids stockés ne le résolvent plus, et un témoin faux
est pire que pas de témoin (le banc ignore de lui-même un `W` nul).

**La bascule a été faite** : § 23.7.

## 23.6 Ce qu'il faudra faire : FUSIONNER les germes confondus, pas les retirer

Retirer marche pour un nuage de test, et c'est faux en général : **la masse du germe retiré
disparaît**. Avec `ν = 1/n` uniforme c'est bénin (chaque cible passe de 1/100 000 à 1/99 944), mais
dès que `ν` est une donnée — une mesure à transporter — supprimer un point n'est pas permis.

La bonne opération est la **fusion** :

1. une grappe de germes à moins de `δ` devient **UN** germe, placé au barycentre pondéré par `ν`
   (celui qui minimise le second moment), de cible `ν = Σ ν_i` ;
2. on résout le problème réduit, où plus aucune arête ne porte un `c_ij` aberrant ;
3. on **redivise** la cellule obtenue entre les membres de la grappe, avec leurs masses prescrites.

Ce qui rend l'étape 3 presque gratuite, et c'est le point : **pour des germes exactement confondus, le
découpage est arbitraire.** Le coût `∫|x − p_i|²` est le même quel que soit le membre auquel on
attribue un morceau, puisqu'ils sont au même point — donc *n'importe quelle* partition de la cellule
aux bonnes masses est optimale, et il n'y a rien à résoudre. Pour des germes à distance `δ`, l'écart
de coût entre deux attributions est `O(δ · diam)` : le découpage optimal est le diagramme de puissance
LOCAL de la grappe restreint à la cellule fusionnée — deux ou trois germes, donc un problème minuscule
et exact — et les poids qu'il demande sont `O(δ · diam)`, une perturbation.

Côté code c'est un **enrobage** de `Newton`, pas une modification : construire la liste de germes
fusionnée plus la table de correspondance, résoudre, puis découper chaque cellule fusionnée localement.
Le moteur ne voit jamais la dégénérescence, donc le laplacien ne porte jamais de poids à 1e4 — la
réparation du § 23.3 devient structurelle au lieu d'être un prétraitement de fichier. Et ça donne au
passage la bonne réponse à une question qui se posera de toute façon sur des données réelles, où l'on
ne peut pas jeter des points.

À mesurer quand ce sera écrit : que le `w` de la solution fusionnée-puis-découpée coïncide avec celui
du problème complet là où le problème complet est soluble, et que `σ = 0.005` converge sans rien
perdre — ce qui, contrairement à la déduplication, vaudra pour n'importe quel `ν`.

## 23.7 LA BASCULE : les nuages Voronoï dédupliqués en place

Faite. `cases/lines5_*_voronoi.txt` (les cinq : `σ = 0.005`, `0.02`, `0.05`, `0.1`, et `n = 2000`) sont
remplacés par leurs versions dédupliquées. Les fichiers `_equal` sont **laissés tels quels** : ils
portent la solution de référence, retirer un germe la périme, `pysdot` n'est installé nulle part ici
donc on ne peut pas la régénérer, et un témoin vaut plus qu'un nuage propre de plus. Conséquence
assumée : les deux entrées « lignes » de la suite ne sont plus le même nuage, et « lignes / aires
égales » garde le nuage dégénéré, son plancher à 2.35e-6 **et son témoin** (écart 9.8e-13) — ce qui lui
donne un rôle utile de cas dégénéré délibéré.

Ce que ça change, entrée par entrée de la suite (défaut, un fil) :

| | avant | après |
|---|---|---|
| 2D uniforme | 6 it, 8 diag, CONVERGE 2.58e-10 | inchangé |
| **2D lignes / Voronoï** | 25 it, 116 diag, **STAGNATION** 2.35e-6 | **23 it, 78 diag, CONVERGE 2.00e-7** |
| 2D lignes / aires égales | 25 it, 116 diag, STAGNATION 2.35e-6 | inchangé (volontairement) |
| 3D (les deux) | — | inchangé (ces nuages étaient propres) |
| 2D lignes `σ = 0.02 / 0.05 / 0.1` | 41 / 23 / 13 diag | 41 / 23 / 13 diag — **identiques** |

Les six sections annoncées sont reprises : § 3 (le plancher), § 7 (la mention incidente), § 8.7.3 (le
nettoyage à la main devenu versionné), § 10.1 (la tolérance élargie à 4e-6, désormais inutile), § 19.10
(le tableau `fp32`), § 21.4 et § 21.7 (la colonne `σ = 0.005`), § 22.3 (le tableau du modèle) et
§ 23.3/23.5. **Deux de mes conclusions y tombent**, et c'est le vrai intérêt de l'opération :

* le « facteur dix » de la bascule de résidu (536 → 53 diagrammes) **n'existait pas** : c'était `log`
  s'acharnant sur une cellule irréparable. Nuage propre : 20 contre 19 ;
* la troisième direction du modèle ne servait qu'à ça. Nuage propre : `K = 2` fait 13 diagrammes,
  `K = 3` en fait 14 — le défaut est revenu à **deux**, une résolution linéaire de moins par itération.

En revanche l'exposant du § 21.4 ressort **plus fort** : 78 → 39 diagrammes sur le cas dédupliqué, un
facteur deux, contre 113 → 73 avant. La dégénérescence le masquait.

## 23.8 Ce que coûterait la phase d'agglomération (`newton --agglo D`)

Question : une première phase d'agglomération paraît obligatoire pour un code robuste, mais on n'a pas
les structures de données pour ça. **On les a**, et c'est le point.

En `w = 0` le diagramme **est** celui de Voronoï, donc sa liste de facettes **est** le graphe de
Delaunay. Or Delaunay contient l'arbre couvrant minimal euclidien, et les grappes du lien simple au
seuil `δ` sont exactement les composantes connexes des arêtes de l'ACM sous `δ`. Donc **balayer `fa` et
faire un union-find rend exactement les grappes voulues** — pas de kd-tree, pas de grille, pas de tri,
et le diagramme est payé de toute façon. (La réciproque est la contrainte : un diagramme de Laguerre
n'est pas Delaunay et ne contient plus l'ACM, donc la détection doit se faire *avant* la résolution —
ce qui est justement le moment voulu.)

`--agglo D` le fait et se chronomètre, `n = 10⁵` :

| nuage | arêtes de Delaunay | sous le seuil | grappes | taille max | temps | en % d'un diagramme |
|---|---|---|---|---|---|---|
| lignes `σ=0.005` **sale** | 297 347 | 56 | **56** | 2 | 5.3 ms | **3.9 %** |
| lignes `σ=0.005` propre | 297 230 | 0 | 0 | — | 5.2 ms | 3.7 % |
| uniforme 2D (seuil 3e-5) | 298 957 | 16 | 16 | 2 | 6.1 ms | 4.6 % |
| plans `σ=0.02` (3D) | 754 614 | 0 | 0 | — | 13.9 ms | **0.9 %** |

**La détection coûte 4 % d'un diagramme en 2D, 1 % en 3D**, et elle trouve exactement les 56 grappes du
nuage sale, toutes de taille 2. Autrement dit la phase 1 est gratuite, et sur un solve de 78 diagrammes
elle se paie 700 fois.

Ce qui reste à écrire est la phase 2, et son coût est de la même nature (§ 23.6) : construire la liste
réduite plus la table de correspondance est `O(n)` ; résoudre le problème réduit est **moins cher** que
l'original ; et redécouper les cellules fusionnées est une poignée de problèmes locaux à deux ou trois
germes — 56 ici — dont le cas exactement confondu est **libre** (toute partition aux bonnes masses est
optimale). Le travail est du code, pas du calcul.

## 23.9 La recherche de relaxation ne sert à rien, et on peut la supprimer (`--mod-frac`)

Il y avait une ambiguïté à lever. Sous `essai-limites` (§ 21.5) le `facteur` est une **constante**, pas
une recherche : la passe des limites cherche `alpha*`, le pas exact où la première cellule touche le
plancher, et on retient `facteur · alpha*`. Il n'y a jamais eu de recherche de relaxation là.

Le modèle, lui, en faisait une : il évaluait le critère à cinq fractions de `alpha*(λ)` — c'était
gratuit, donc pourquoi pas. Mais gratuit n'est pas utile. `--mod-frac 1` (la plus longue fraction
seulement, `0.99 · alpha*`) contre `5` :

| | uniforme | `σ=0.1` | `σ=0.02` | `σ=0.005` |
|---|---|---|---|---|
| 5 fractions | 6 | 8 | 11 | 13 |
| **1 fraction** | **6** | **8** | **11** | **13** |

**Identique partout**, et la trace montre pourquoi : le choix tombait sur `0.99` neuf fois sur dix. Le
défaut est donc passé à **une seule fraction** — un cinquième du coût de la recherche, et un concept de
moins.

Et la raison est déjà dans le profil du § 21.2 : **le critère décroît de façon monotone le long du
rayon jusqu'à ce que le plancher morde.** Le meilleur point admissible est donc toujours au bord, et il
n'y a rien à chercher — seulement à s'arrêter juste avant. Ce qui répond à la question posée : oui, on
utilise toujours ~0.9 à 0.99 du pas admissible exact, et autant l'écrire une fois pour toutes.

## 23.10 La résolution en DEUX ÉTAPES, faite et mesurée (`newton --agrege D`)

Le § 23.8 n'avait fait que la détection. Voici les deux étapes : agréger, résoudre. `Agglo.h` porte la
détection (hachage de grille au pas `δ` + union-find, sans diagramme, donc utilisable avant tout le
reste) et la réduction (un germe par grappe, au **barycentre pondéré par `ν`**, de cible `Σ ν_i`) ;
`lance_agrege` enchaîne, remonte les poids, et mesure ce que ça vaut **sur le nuage complet** — le seul
chiffre honnête.

### Ce que la « dégénérescence » est vraiment

Deux germes à distance `δ` sont séparés par un plan dont le décalage vaut `(w_i − w_j) / 2δ`. Placer ce
plan à `ε` près du diamètre `L` de la cellule demande donc `w_i − w_j` à `2δεL` près : **l'écart de
poids est divisé par `δ`**. À `δ = 10⁻⁸`, `L = 10⁻³`, `ε = 10⁻⁶`, il faut `w_i − w_j` à 2·10⁻¹⁷ près,
sous l'epsilon machine relatif de poids qui valent ~10⁻¹. Ce n'est donc pas une insolubilité, c'est un
**plancher de précision** — et il explique le 2.35e-6 exactement là où le banc s'arrêtait.

### Une seule paire suffisait

`lines5_n100000_s0.005_equal.txt` (le nuage dégénéré conservé, § 23.7), 8 fils :

| `δ` | grappes | le problème RÉDUIT | germes seuls > 1e-6 | membres agrégés |
|---|---|---|---|---|
| — (sans agrégation) | — | **STAGNATION 2.35e-6**, 113 diag | — | — |
| 5e-8 | **1** | **CONVERGE 2.12e-7**, 79 diag | **3** / 99 998 | max 1.06, 1 vide sur 2 |
| 1e-7 | 3 | CONVERGE 2.12e-7, 79 diag | 11 / 99 994 | max 1.06, 3 vides sur 6 |
| 1e-6 | 7 | CONVERGE 2.12e-7, 79 diag | 34 / 99 986 | max 5.21, 7 vides sur 14 |
| 4.5e-6 | 56 | CONVERGE 3.38e-7, 74 diag | — | max 14.4, 56 vides sur 112 |

**Tout le plancher à 2.35e-6 était UNE paire de germes à 10⁻⁸.** L'agréger suffit : le problème réduit
converge à 2.12e-7, et les diagrammes passent de 113 à 79. C'est le résultat de la section.

### Mais la remontée sans redécoupage ne vaut rien, et de deux façons

Tous les membres d'une grappe reçoivent le **même** poids, donc leurs plans mutuels passent par le
milieu : la cellule fusionnée se partage selon la **géométrie** et pas selon les masses voulues. Mesuré :
chaque paire finit à ~100 % d'écart, et **la moitié des membres est vide**. Sans surprise, et c'est
exactement ce que le § 23.6 corrigerait.

L'autre façon est moins évidente et c'est la vraie contrainte. Agréger **déplace** la grappe sur son
barycentre, et un voisin voit son aire bouger de `déplacement × périmètre / aire`. Sur une cellule en
**lamelle** — ce que `σ = 0.005` fabrique — ce rapport explose : un déplacement de 2.5·10⁻⁸ donne 4.3 %
d'erreur d'aire. Donc **le critère sur `δ` n'est pas « `δ` ≪ espacement » mais « `δ` ≪ épaisseur locale
de cellule »**, et sur des lamelles l'épaisseur est minuscule.

La bonne nouvelle est que l'erreur reste **strictement locale** : 3 germes seuls touchés pour une
grappe, 11 pour trois, 34 pour sept — soit 3 à 7 voisins immédiats par grappe, sur 10⁵ germes dont tout
le reste est sous 1e-6. Même forme sur l'uniforme (5 touchés pour une grappe, 109 pour seize).

### Donc non, on n'est pas au bout — et la pièce qui manque n'est pas décorative

Le redécoupage (§ 23.6) répare **les deux** problèmes d'un coup, et c'est ce qui le rend obligatoire
plutôt qu'optionnel : une fois chaque membre remis à sa vraie position avec son propre poids, le
diagramme est le vrai diagramme — les masses des membres sont bonnes *et* les voisins ne sont plus
perturbés, puisque plus rien n'a bougé. Ce qui reste à écrire, par grappe :

1. le plan entre deux membres a une normale **fixe** (`p_j − p_i`) ; seul son décalage est libre ;
2. on le cherche par bissection sur l'aire du morceau — **monotone et parfaitement conditionné**,
   parce que le paramètre est la POSITION du plan et non l'écart de poids ;
3. on en **déduit** `w_i − w_j = 2δ · décalage` à la fin. L'amplification par `1/δ` devient une simple
   sortie, jamais une inconnue — et c'est là tout l'intérêt de faire le découpage à part.

Pour des germes **exactement** confondus, l'étape 2 est libre : le coût `∫|x − p_i|²` ne dépend pas du
membre auquel on attribue un morceau, donc toute partition aux bonnes masses est optimale. La liberté
dont on dispose est exactement celle dont on a besoin.

### Deux notes de mise en œuvre

`Agglo.h` détecte par hachage de grille (0.08 s à `n = 10⁵`) parce qu'il doit pouvoir tourner **sans
diagramme**. Quand le diagramme est de toute façon payé, la variante Delaunay de `--agglo` est vingt
fois plus rapide (5 ms, § 23.8) : c'est elle qu'il faudra brancher dans le solveur réel.

Et le nuage dégénéré reste disponible comme **test** : `lines5_n100000_s0.005_equal.txt` porte les mêmes
germes que l'ancien `_voronoi` (ils sont écrits depuis les mêmes positions), donc `--load` dessus donne
le cas dégénéré, avec en prime son témoin. `pysdot` est désormais installé, donc `gen_cases.py`
fonctionne à nouveau et les nuages hors dépôt sont régénérables.

## 23.11 LE REDÉCOUPAGE, fait — et ce qu'il démontre (`--agrege`, `--agrege-brut`, `--agrege-fin`)

Troisième étape écrite (`Agglo.h` : `sommets_cellule`, `aire_coupee`, `coupe_a_l_aire`, `redecoupe`).
Le principe est le choix du paramètre : le plan qui sépare deux membres a une normale **fixe**
(`p_j − p_i`), seul son décalage est libre, et on le cherche **directement** par bissection sur l'aire
du morceau — monotone, sur une quantité de l'ordre de la cellule, donc parfaitement conditionnée.
L'écart de poids n'est calculé qu'à la fin :

```
w_i − w_j = 2 ( p_j − p_i ) . x_plan − ( |p_j|² − |p_i|² )
```

**L'amplification par `1/δ` devient une sortie, jamais une inconnue.** C'est toute la différence avec
ce que Newton peut faire sur le nuage complet, où les poids *sont* les inconnues.

### Il est exact, et il fait ce qu'on attendait

Nuage dégénéré (`lines5_n100000_s0.005_equal.txt`), `δ = 5e-8`, une grappe :

| étape | résultat |
|---|---|
| réduit (99 999 germes) | **CONVERGE 2.12e-7**, 79 diagrammes |
| **découpe** | 1 grappe, **écart local 5.17e-10** |
| remontée, membres agrégés | **4.29e-2, 0 vide** — contre **1.06 et 1 vide sur 2** sans découpe |
| correction finale (nuage complet) | STAGNATION 3.42e-6, **3 it / 39 diag** — contre **8 it / 141 diag** sans découpe |

Sur l'uniforme (`δ = 3e-5`, 16 grappes) : écart local **3.4e-12**, membres à 2.6e-2 sans un vide
(contre 1.04 et 7 vides), et la correction finale **CONVERGE en 2 itérations / 3 diagrammes** au lieu de
6 / 20. Le découpage vaut donc un facteur 4 à 7 sur la phase de correction, et il supprime les cellules
vides — qui étaient la vraie raison pour laquelle cette phase coûtait cher.

### Mais bout à bout, la chaîne ne bat pas le solve direct — et c'est le résultat

118 diagrammes et STAGNATION à 3.42e-6, contre 113 diagrammes et 2.35e-6 en direct. La correction
finale **retombe sur le même plancher**, et elle ne peut pas faire autrement : elle est revenue à
paramétrer la paire par deux poids.

Ce plancher se calcule, et il tombe juste. Résolution d'un poids en `double` : `ε |w| = 2.2e-16 ×
0.1285 = 2.8e-17`. Résolution du décalage du plan : divisée par `2d = 2 × 1.009e-8`, soit **3.1e-9**.
Rapportée à l'étendue de la cellule le long de l'axe de la paire, `h = 4.53e-4`, ça donne une erreur
d'aire relative de **3.1e-6** — mesurée 2.35e-6. L'ordre de grandeur est donc le bon, et c'est bien
pourquoi le plancher est vers `10⁻⁶` et pas `10⁻¹²` : **le plancher est l'epsilon machine vu à travers
`1/δ`.**

> **Nuance, ajoutée après coup (§ 24.3).** L'estimation ci-dessus utilise l'amplitude *globale* des poids
> (0.1285), donc elle **majore** : ce qui compte est l'amplitude locale. Et 2.35e-6 n'est pas une barrière
> que rien ne franchit — `--pas modele` atteint 9.65e-07 sur ce même nuage. C'est **où l'amortissement de
> KMT cale**, et l'argument en `1/δ` en donne l'ordre de grandeur, pas la valeur exacte.

### Donc la conclusion porte sur l'interface, pas sur l'algorithme

Les trois étapes marchent, chacune vérifiée :

* le **problème réduit** converge à 2.12e-7 en 79 diagrammes, soit **dix fois mieux et moins cher** que
  les 2.35e-6 / 113 diagrammes du direct ;
* le **découpage** rend à chaque membre sa masse à 1e-10 près ;
* et la **conversion en poids** détruit le tout.

Autrement dit **la sortie de cette méthode ne peut pas être un vecteur de poids.** C'est le couple
(poids du problème réduit, décalage du plan par grappe) qui porte la précision, et c'est exactement ce
que `redecoupe` produit avant de le convertir. Un solveur dont l'interface est `w` ne peut pas exprimer
la réponse — pas parce qu'il calcule mal, mais parce que deux `double` ne suffisent pas à coder un plan
placé à 1e-9 près entre deux points distants de 1e-8.

Ce qui donne la règle pratique, et elle est simple : **si le consommateur veut des cellules**, on lui
donne le diagramme réduit plus les plans de coupe, et tout est exact ; **s'il veut des poids**, il
faut accepter le plancher `ε|w| / (2 δ h)` ou dédupliquer les germes en amont (§ 23.7). Il n'y a pas de
troisième possibilité, et c'est ce que cette section établit.

Reste une limite de l'implémentation : `redecoupe` est **exact pour `k = 2`** — le seul cas mesuré, la
taille maximale de grappe étant 2 sur tous les nuages. Pour `k > 2` il retire les membres un à un par
coupes successives : les masses sont bonnes, donc la partition est optimale si les germes sont
exactement confondus (le coût ne dépend alors pas du membre), mais elle n'est plus forcément un
diagramme de puissance et les poids rendus ne sont qu'approchés. Et c'est 2D seulement, comme tout
`Ecrasement.h`.

## 23.12 RENONCER aux `w` des germes agrégés : ce que ça coûte vraiment (`--cout`)

Puisque l'écart de poids d'une paire à `δ = 10⁻⁸` est sous la précision machine (§ 23.11), autant
**décider de ne pas le résoudre** : la solution EST alors un agrégat, et c'est aux fonctionnelles en
aval d'en tenir compte. Reste à savoir ce que ça leur coûte. Pour le coût de transport, rien — et pas
« négligeable » : rien, à seize chiffres.

### La décomposition est exacte

```
Σ_{i∈r} ∫_{C_i} |x − p_i|²  =  ∫_{C_r} |x − q|²  +  Σ_{i∈r} ν_i |p_i − q|²  −  2 Σ_{i∈r} (p_i − q)·m_i
```

avec `m_i = ∫_{C_i}(x − q)`. Les **deux premiers termes ne demandent que la cellule fusionnée et les
positions** : aucun découpage. Le troisième est le seul qui en dépende, et il s'annule au premier ordre
**parce que `q` est le barycentre pondéré par `ν`** — si `m_i ≈ (ν_i/ν_r) M_r`, il vaut
`−2 (M_r/ν_r)·Σ ν_i (p_i − q) = 0`. C'est la vraie raison de ce choix de `q`, que j'avais d'abord
justifié par le second moment.

Les moments (`aire`, `∫(x−p)`, `∫|x−p|²`) sont exacts, sommés sur les triangles `(p, v_j, v_{j+1})` avec
leurs aires **signées** — ce qui vaut que `p` soit dedans ou dehors. Ils n'existaient nulle part dans le
banc : `PremierOrdre.h` notait « il faudrait le second moment de chaque cellule ». Ils sont dans
`Agglo.h` (`moments_cellule`, `cout_transport`).

### Mesure : le coût est identique à 1.2e-17

Nuage dégénéré, `n = 10⁵`, `δ = 5e-8` (une grappe) :

| | coût de transport |
|---|---|
| solve **direct** (qui stagne à 2.35e-6 sur les aires) | `1.923535254484273e-02` |
| **agrégé, sans jamais résoudre les `w` de la paire** | `1.923535254484250e-02` |

**Écart relatif 1.2e-17.** Et la décomposition se lit : le terme de variance interne vaut `5.09e-22`
(soit `ν δ² ≈ 10⁻⁵ × 10⁻¹⁶`, comme prévu), donc tout le coût est dans la cellule fusionnée. L'aire
totale est la même (`0.999999999999` des deux côtés, le déficit étant la quadrature du diagramme).

### Et la raison en une ligne

Déplacer une masse `Δa` d'un membre à l'autre change le coût de `(w_i − w_j) · Δa` — parce que sur le
plan de puissance qui les sépare, l'intégrande `|x−p_i|² − |x−p_j|²` vaut exactement `w_i − w_j`. Or
c'est précisément cette quantité qui est minuscule (`~2δL ≈ 10⁻¹¹`). **Le coût de transport est aveugle
exactement à la dégénérescence que les poids ne savent pas résoudre.** Ce n'est pas une coïncidence :
les deux sont la même petitesse, vue une fois au numérateur et une fois au dénominateur.

### Ce que l'agrégat change, et ce qu'il ne change pas

* **rien à changer** pour tout ce qui est une somme pondérée par les masses : le coût de transport, la
  masse totale, le barycentre global, tout moment global. La cellule fusionnée et les `ν_i` suffisent ;
* **le terme de variance interne `Σ ν_i |p_i − q|²` est à ajouter**, et il est exact, explicite, et
  calculable sans rien connaître du découpage ;
* **le découpage n'est nécessaire que pour une quantité géométrique PAR MEMBRE** — le barycentre de la
  cellule d'un membre, sa forme, son voisinage. `redecoupe` (§ 23.11) le fournit exactement pour `k = 2`,
  et son coût est celui d'une poignée de bissections ;
* et **l'interface doit porter l'agrégat**, ce qui est la conclusion du § 23.11 sous une autre forme : un
  `std::vector<TF> w` de taille `n` ne peut pas représenter la solution, un (diagramme réduit + table de
  grappes + masses) si.

C'est le compromis raisonnable : on ne demande pas au solveur de calculer ce que le format ne peut pas
stocker, et on constate que la quantité qui intéresse vraiment n'en dépend pas.

---

# 24. BILAN : L'ALGORITHME PAR DÉFAUT, ET CE QU'IL VAUT CONTRE KMT

## 24.1 Le défaut, c'est KMT — et il n'a pas bougé

À une exception près, tout ce que les § 21 à 23 ont ajouté est **en option**. Le défaut du banc est
l'amortissement de Kitagawa–Mérigot–Thibert :

* **direction** : Newton sur le dual de Kantorovich, `L d = ν − a` ;
* **pas** : l'échelle `t = 1, 1/2, 1/4 …`, un diagramme par essai, accepté quand (a) toute cellule reste
  au-dessus du plancher d'aire `eps` et (b) `|a − ν|₂` décroît d'au moins `1 − t/2`.

L'exception est une **correction**, pas une accélération, et elle est dans le défaut : le plancher ne
défend plus que les cellules **non vides au départ** (§ 8.7). Avant, une seule cellule vide mettait
`eps = 0` et désactivait le garde-fou **en silence** — un départ avec une cellule vide pouvait en vider
7 750 après un pas. C'est le seul changement au comportement par défaut de tout ce travail.

## 24.2 Contre KMT (`job -b`, 8 fils, `n = 10⁵`)

`scripts/bilan_kmt.sh`. « reste » est le `max|a−ν|/ν` atteint, et il compte : un nombre de diagrammes ne
veut rien dire si les variantes ne s'arrêtent pas au même endroit.

| 2D lignes / Voronoï | it | diag | temps | reste |
|---|---|---|---|---|
| **KMT (le défaut)** | 23 | 78 | 10.94 s | 2.00e-07 |
| KMT + `p = 0.25` | 16 | 50 | 8.07 s | 8.27e-09 |
| KMT + `log` + bascule | 14 | 39 | 6.44 s | 8.23e-07 |
| **limites + `log` + bascule** | 12 | **19** | **4.25 s** | 2.69e-08 |
| modèle (span, `K = 2`) | 12 | **13** | 6.57 s | 8.08e-07 |

| 2D lignes / aires égales — le nuage **dégénéré** | it | diag | temps | reste |
|---|---|---|---|---|
| **KMT** | 24 | 113 | 14.21 s | STAGNATION 2.35e-06 |
| KMT + `log` + bascule | 16 | 74 | 10.01 s | STAGNATION 2.35e-06 |
| limites + `log` + bascule | 13 | 53 | 7.50 s | STAGNATION 2.35e-06 |
| **modèle (span, `K = 2`)** | 12 | **13** | **7.40 s** | **CONVERGE 9.65e-07** |

| 3D plans / Voronoï | it | diag | temps | reste |
|---|---|---|---|---|
| **KMT** | 13 | 27 | 10.48 s | 1.16e-10 |
| **KMT + `p = 0.25`** | 9 | **17** | **7.53 s** | 1.63e-12 |
| KMT + `log` + bascule | 9 | **17** | **7.50 s** | 1.96e-09 |

Sur l'uniforme (2D comme 3D) tout se tient dans le bruit, sauf l'exposant qui gagne un tiers du temps
en 3D (9 → 5 diagrammes, 3.05 → 1.93 s). C'est cohérent : il n'y a rien à gagner là où KMT accepte
déjà `t = 1`.

**Donc, contre KMT : −61 % de temps et −76 % de diagrammes sur le cas dur 2D, −28 % en 3D, et le seul
essai qui converge sur le nuage dégénéré.** Le gain croît avec la difficulté et s'annule sur les cas
faciles, ce qui est la bonne forme.

## 24.3 Deux réserves qu'il faut lire avec les chiffres

**Les variantes rapides s'arrêtent parfois plus près de la tolérance.** `limites + log` finit à 6.8e-07
sur l'uniforme 2D là où KMT descend à 2.6e-10 : le dernier pas de Newton est quadratique, donc KMT
dépasse largement la cible pour le même prix. Sur les cas durs la réserve ne tient pas — `limites + log`
fait **à la fois** moins de diagrammes (19 contre 78) et un meilleur reste (2.7e-8 contre 2.0e-7).

**Et le 2.35e-6 du nuage dégénéré n'est pas une barrière absolue.** Le § 23.11 l'explique par l'epsilon
machine vu à travers `1/δ`, et l'ordre de grandeur est le bon — c'est bien pourquoi le plancher est vers
`10⁻⁶` et pas `10⁻¹²`. Mais l'estimation utilisait l'amplitude *globale* des poids, donc elle majore :
`--pas modele` atteint 9.65e-07 sur ce même nuage. Le 2.35e-6 est donc **où l'amortissement de KMT
cale**, pas une limite que rien ne franchit.

## 24.4 Ce qu'il faudrait mettre par défaut

D'après ces mesures, et c'est un changement d'une ligne que je n'ai pas fait :

* **`--residu log`** (la bascule vers `lin` étant déjà le défaut) : elle aide dans les deux dimensions,
  n'a coûté nulle part, et c'est la seule amélioration disponible en 3D ;
* **`--pas essai-limites`** en 2D : c'est la ligne la plus rapide, et elle ne fait aucun recul.

Je laisserais le **modèle** en option : il donne le moins de diagrammes de tout le banc (13 partout sur
les cas durs) et il est le seul à passer le nuage dégénéré, mais son balayage global coûte un diagramme
par itération, donc il perd en temps de paroi sur les cas sains — et c'est le code le plus récent, donc
le moins éprouvé.

Ce qui reste hors de ce bilan : le **multi-échelle** (§ 8, toujours sans solution), l'**agrégation**
(§ 23.10–12, dont la phase 2 est écrite et vérifiée mais pas branchée dans le solveur par défaut), et
les deux limites structurelles du modèle et des limites — **2D seulement**, comme tout `Ecrasement.h`.

## 24.5 `--residu log` par défaut : ce qu'il a fallu corriger

Fait — `NewtonOptions::residu = LOG`, la bascule vers `lin` étant déjà le défaut. Sur la suite :

| | avant (`lin`) | après (`log` + bascule) |
|---|---|---|
| 2D uniforme | 8 diag | **7** |
| 2D lignes / Voronoï | 78 diag | **39** |
| 2D lignes / aires égales | 116 diag | **74** |
| 3D uniforme | 9 diag | **6** |
| 3D plans (les deux) | 27 diag | **17** |

Le témoin passe toujours (écart 3.9e-12 sur une amplitude 0.128). Les `reste` sont un peu plus grands
(8.2e-07 au lieu de 2.0e-07 sur les lignes), toujours loin sous la tolérance.

**Mais le défaut de la bibliothèque a cassé la continuation en densité, et il a fallu l'y remettre à
`lin`.** Le § 9.6 avait déjà mesuré que `log` y coûte dix fois plus ; j'espérais que la bascule l'en
protégerait, puisque chaque étape repart d'un résidu déjà petit. À `σ = 0.05` c'est le cas — la bascule
tire à l'itération 0 de chacune des 7 étapes, et le résultat est un match nul (192 diagrammes contre
188, 152.3 s contre 152.1). À `σ = 0.02` **non** : certaines étapes démarrent au-dessus du seuil, et le
défaut `log` **échoue** — 6 étapes sur 8, `STAGNATION` puis `SOLVEUR LINÉAIRE EN ÉCHEC`, 1 204
diagrammes et 642 s, là où `lin` passe les 8 en 862 diagrammes et 503 s.

Donc `main_densite` et `main_image` remettent `residu = LIN` dans leur `Opts`, avec la raison écrite sur
place. Pour `densite` c'est **mesuré** ; pour `image` c'est **par précaution** — même régime de
continuation (§ 12), mais pas remesuré ici, et c'est dit ainsi dans le code. Vérification : `densite
--sigma 0.02 --conv 0.5` reproduit `lin` ligne pour ligne (`212 (176)`, `56 (40)`, `15 (6)`).

La leçon est générale et vaut d'être écrite : **un défaut appartient au régime où il a été mesuré.**
`log` a été mesuré sur des solves directs ; le mettre dans `NewtonOptions` le poussait dans un régime
où le banc avait déjà la mesure contraire.

## 24.6 MINIMISER le mérite au lieu de prendre le premier pas qui passe (`--pas merite`)

Question posée : le pas du `log` n'était pas choisi en minimisant le résidu `log`, mais en cherchant
jusqu'où aller avant qu'une cellule casse — est-ce qu'il ne faudrait pas plutôt minimiser ce qu'on veut
minimiser ? Réponse : **si, et ça vaut 67 % des diagrammes sur le cas dur.**

D'abord la mise au point, parce que les trois modes ne faisaient pas la même chose :

* `--pas essais` (KMT) : échelle `t = 1, ½, ¼ …` et on prend **le premier** `t` qui passe les deux
  clauses (plancher d'aire, décroissance `1 − t/2` du mérite). Le mérite `log` est donc bien utilisé,
  mais comme **veto** et non comme objectif ;
* `--pas essai-limites` : le pas est `facteur · α*`, c'est-à-dire **où les cellules cassent**, et le
  mérite ne fait que vétoyer ;
* `--pas modele` : pareil pour la longueur (`0.99 · α*`), le critère ne choisit que la direction.

Et le profil (§ 21.2) montre que le mérite `log` a un **vrai minimum intérieur** — lignes `σ = 0.005`,
direction `log`, itération 0 :

| `t` | 1.0 | 0.5 | 0.25 | **0.125** | 0.0625 | → 0 |
|---|---|---|---|---|---|---|
| mérite `log` | 2374 | 643 | 334.6 | **322.5** | 333.0 | ↗ 350.3 |
| cellules vides | 22 334 | 1 061 | 59 | 2 | 0 | 0 |

Deux choses s'y lisent. Le minimum est à `t = 0.125`, et **le plancher d'aire le refuse** (2 cellules
vides) alors que le mérite y est meilleur qu'au pas retenu. Et le mérite `log` **pénalise déjà les
cellules vides**, monotonement (643 pour 1 061 vides, 334 pour 59, 322 pour 2) — ce qui rend le plancher
possiblement redondant.

`--pas merite` descend l'échelle jusqu'à ce que le mérite remonte et prend l'argmin. Avec une précaution
qui vaut tout : **si le premier barreau passe déjà la condition de KMT, on le prend** et on ne cherche
pas plus loin — sans ça, la minimisation paie un diagramme de plus par itération là où `t = 1` était
déjà bon (mesuré : 13 diagrammes au lieu de 7 sur l'uniforme 2D, pour le même résultat).

| | défaut (premier qui passe) | **`--pas merite`** | `--pas merite --sans-plancher-aire` |
|---|---|---|---|
| 2D uniforme | 7 | **7** | 7 |
| **2D lignes `σ=0.005`** | 39, reste 8.2e-07 | **13, reste 1.8e-08** | **13** |
| 2D lignes dégénéré | 74 | **49** | **49** |
| 3D uniforme | 6 | **6** | 6 |
| 3D plans | **17** | 21 | *SOLVEUR EN ÉCHEC* |

**−67 % de diagrammes sur le cas dur 2D et un résidu 45 fois meilleur**, pour un changement qui ne
touche qu'au choix du pas. C'est le même compte que `--pas modele` (13 diagrammes) sans aucune
machinerie polynomiale, et sans les `K − 1` résolutions linéaires de plus.

> **CORRECTION (§ 24.9) : ces −67 % ne viennent PAS de la minimisation.** `--pas merite` entrait aussi
> dans la passe des limites — elle tourne pour tout `pas != essais` — donc son premier barreau était
> `facteur · α*` et non `t₀`, ce que les pas affichés (`0.107`, `0.123`, `0.168`) disaient déjà sans que
> je le voie. Sur une échelle **vraiment** dyadique (`--mer-sans-limites`) la minimisation fait **48
> diagrammes, pire que les 39 de KMT**. Le gain était l'`α*` exact, pas l'argmin.

**Et le plancher d'aire devient effectivement redondant — en 2D.** Les colonnes avec et sans plancher
sont **identiques** (7 / 13 / 49) : le mérite `log` seul suffit à garder les cellules vivantes, ce qui
était l'intuition. Mais deux réserves fermes : avec le premier-pas-acceptable, éteindre le plancher est
**catastrophique** (136 diagrammes au lieu de 39 sur les lignes, 170 au lieu de 74 sur le dégénéré) — la
minimisation est ce qui le rend superflu, pas `log` ; et **en 3D ça casse** (`SOLVEUR LINÉAIRE EN
ÉCHEC`), donc le plancher reste obligatoire là. L'écrêtage de `g` à `x ≥ 1e-8` en est la cause probable :
il borne la pénalité d'une cellule vide à `−18.4` au lieu de `−∞`, donc le mérite peut en tolérer
quelques-unes ; un `log` non écrêté rendrait le plancher inutile par construction, et c'est la prochaine
chose à essayer.

En 3D `--pas merite` coûte 4 diagrammes de plus sur les plans (21 contre 17) : le barreau
supplémentaire qui confirme le minimum n'y est pas amorti. **Je ne l'ai donc pas mis par défaut** — ce
serait un défaut dépendant de la dimension, et je viens de voir (§ 24.5) ce que coûte un défaut posé hors
du régime où il a été mesuré.

## 24.7 Le `log` NON ÉCRÊTÉ : le plancher d'aire devient redondant — pendant la phase `log`

`g` était écrêté à `x ≥ 1e-8`, donc `log` était borné à `−18.4` et une cellule vide ne coûtait qu'un
montant **fini** au mérite : c'est pour ça qu'il en tolérait quelques-unes (§ 21.2). `--g-ecrete 0`
retire l'écrêtage **du mérite seulement** : une cellule vide y coûte alors `+∞`, donc aucun pas qui en
vide une n'est acceptable. L'écrêtage reste dans la **direction**, où `b_i = ν_i x_i (c − log x_i)`
vaudrait `0 × ∞` sur une cellule vide — et où il faut au contraire pouvoir en **remplir** une (§ 8.7).

### Mesure : il remplace le plancher exactement

`--pas essais --sans-plancher-aire`, diagrammes :

| | avec écrêtage | **sans écrêtage** | pour mémoire, plancher ACTIF |
|---|---|---|---|
| 2D uniforme | 7 | **7** | 7 |
| 2D lignes `σ=0.005` | **136** | **39** | 39 |
| 2D lignes dégénéré | **170** | **74** | 74 |
| 3D plans | *ÉCHEC* | **22** | 17 |

**Sur les deux cas durs 2D, les chiffres sans plancher sont identiques à ceux avec, au diagramme près.**
Le plancher d'aire de KMT est donc *exactement* redondant avec un mérite `log` non écrêté — c'était
l'intuition, et l'écrêtage était bien la seule raison pour laquelle il ne l'était pas.

### Mais il a fallu une règle de plus, et elle est instructive

Le premier essai échouait encore en 3D (`SOLVEUR LINÉAIRE EN ÉCHEC`). `--refus 4` dit pourquoi, et ce
n'était pas ce que je supposais : à l'itération 4, `t = 0.5`, la cellule 5754 passe de `1.019e-05` à
**exactement zéro**, l'aire refuse — **et le mérite accepte**. Parce que la bascule (§ 21.7) est déjà
passée à `lin`, et **le mérite `lin` récompense le vidage** : son minimum le long de la direction est en
`t = 1`, là où 50 030 cellules sur 100 000 sont vides (§ 21.2).

D'où la règle, qui n'est pas un réglage mais une conséquence : **le plancher ne peut être éteint que
pendant que le mérite interdit lui-même les cellules vides**, donc avec un `log`/`puissance` non écrêté,
et **jamais en `lin`**. Avec elle, les plans 3D convergent (22 diagrammes, reste 1.45e-12).

### Ce que ça vaut, et ce que ça ne vaut pas

Ce n'est **pas un gain** : les chiffres sont au mieux identiques, et en 3D le plancher éteint coûte
5 diagrammes (22 contre 17). C'est une **simplification** — un garde-fou de moins à porter, et un
morceau de la théorie KMT qui s'avère être une conséquence du mérite plutôt qu'une hypothèse séparée,
dès qu'on prend le bon mérite et qu'on ne l'écrête pas.

Les défauts ne changent pas : `plancher_aire = true` et `g_ecrete = 1e-8`. Ce qui change est qu'on sait
maintenant **pourquoi** le plancher est là — il couvre la phase `lin`, et rien d'autre.

## 24.8 Et si on minimisait VRAIMENT ? Le pas hors de la grille dyadique (`--mer-raffine`)

Objection juste : les pas affichés tombent toujours sur `1, ½, ¼, ⅛ …` parce que **l'échelle est
dyadique**. `--pas merite` prend donc l'argmin *sur les barreaux*, pas le vrai minimum — une grille de
facteur deux, et rien ne dit que le minimum tombe dessus. `--mer-raffine K` ajoute `K` évaluations de
section dorée sur `[ t*/2, min(t₀, 2t*) ]`, où le profil est unimodal.

Les pas deviennent effectivement non dyadiques — lignes `σ = 0.005`, `raffine 8` : `0.117`, `0.136`,
`0.144`, `0.119`, `0.154`, `0.303`, `0.569`, `0.641`, … Donc l'objection porte. Mais :

| diagrammes | `raffine 0` | 2 | 4 | 8 |
|---|---|---|---|---|
| 2D uniforme | **7** | 19 | 31 | 46 |
| 2D lignes `σ=0.005` | **13** | 40 | 66 | 145 |
| 2D lignes dégénéré | **49** | 102 | 130 | 230 |
| 3D plans | **21** | 47 | 68 | 110 |

**Raffiner coûte `K` diagrammes par itération et ne rachète rien** — le compte d'itérations ne bouge
presque pas (12 → 13 → 13 → 16 sur les lignes), donc c'est une perte sèche. Deux raisons, et la seconde
est la vraie.

**La première est banale :** au voisinage d'un minimum la fonction est quadratique, donc gagner sur la
position du minimum ne gagne presque rien sur sa valeur. Le barreau dyadique est déjà « assez bon ».

**La seconde se lit dans la trace, et elle est plus intéressante.** Avec `raffine 8`, le mérite continue
de descendre proprement (`1.969e-04 → 1.694e-04 → 1.364e-04 → 5.682e-05`) pendant que le **vrai**
critère REMONTE : `max|a−ν|/ν` passe de `0.976` à `1.897`, `2.114`, `2.467` sur quatre itérations. Ces
itérations sont **après la bascule**, donc le mérite minimisé est le mérite `lin` — et le § 21.1 avait
établi que celui-là est un mauvais juge (son minimum le long de la direction est là où 50 030 cellules
sont vides). Le minimiser *finement* revient donc à **sur-ajuster un mauvais objectif**.

> **CETTE SECTION ÉTAIT MAL POSÉE, voir le § 24.9.** Le raffinement s'appliquait aussi **après la
> bascule**, donc il minimisait le mérite `lin` — ce qui n'est pas la proposition et explique à lui seul
> la remontée de `max|a−ν|/ν`. Corrigé (minimisation pendant la phase `log` seulement), le raffinement
> coûte deux fois moins mais reste une perte, et pour une raison qui n'a rien à voir avec celle écrite
> ici. La conclusion « la grossièreté nous protégeait » est retirée : c'était l'aveu de ne pas contrôler
> ce que je mesurais, pas un résultat.

Sur la précision numérique, qui était l'autre moitié de la question : les aires sont des aires de
polygone **exactes** (les sommets sont résolus depuis leurs plans, § 19.11), donc `log(a/ν)` est calculé
à ~1e-16 près sauf sur les cellules quasi dégénérées. L'imprécision n'est pas numérique — c'est
l'objectif qui n'est pas le bon.

Le défaut reste donc `mer_raffine = 0`. Ce qu'il faudrait pour que la question se repose utilement :
minimiser `max|a−ν|/ν` lui-même le long du pas, et non le mérite — ce qui demande de l'évaluer sans
diagramme, donc le modèle polynomial du § 22. C'est exactement ce que `--pas modele` fait pour la
*direction*, et il atteint les mêmes 13 diagrammes.

## 24.9 Ce que minimiser le résidu `log` donne vraiment — et pourquoi `α*` EST la réponse

Deux objections justes ont défait les § 24.6 et 24.8, et la résolution est plus propre que ce que
j'avais écrit.

### Première erreur : je minimisais le mérite APRÈS la bascule

La proposition était : minimiser le résidu `log` **pendant la phase `log`**, et protéger l'aire pendant
la phase `lin`. Or `--pas merite` minimisait `merite()`, qui suit le résidu courant — donc **après la
bascule il minimisait le mérite `lin`**, dont le § 21.1 a établi qu'il est un mauvais juge. La remontée
de `max|a−ν|/ν` que j'attribuais à « la finesse » venait de là. Corrigé : la minimisation ne s'applique
plus que tant que le résidu n'est pas `lin` ; après, KMT reprend avec son plancher.

### Deuxième erreur : `--pas merite` n'était pas l'échelle dyadique que j'annonçais

La passe des limites tourne pour tout `pas != essais`, donc le premier barreau était déjà
`facteur · α*`. Les pas affichés le disaient (`0.107`, `0.123`, `0.168` — pas des puissances de deux) et
je ne l'ai pas vu. `--mer-sans-limites` sépare enfin les deux. Lignes `σ = 0.005`, diagrammes :

| | uniforme | lignes `σ=0.005` | dégénéré | plans 3D |
|---|---|---|---|---|
| KMT (`essais`) | 7 | 39 | 74 | **17** |
| **`merite`, départ `α*`** | **7** | **13** | **47** | 21 |
| `merite`, départ `α*`, raffiné 4 | 11 | 42 | 76 | 31 |
| `merite`, échelle dyadique | 7 | **48** | 83 | 21 |
| `merite`, dyadique, raffiné 4 | 11 | 70 | 108 | 31 |
| `merite`, dyadique, raffiné 8 | 15 | 101 | 146 | 43 |

**Les 13 diagrammes venaient du `α*` exact, pas de l'argmin.** Sur une vraie échelle dyadique, minimiser
fait 48 — *pire* que KMT. Et raffiner est une perte dans les deux variantes.

### Et la résolution : le mérite NON ÉCRÊTÉ est monotone, donc son argmin EST le bord

J'avais d'abord écrit que le minimum libre du mérite était « au-delà de ce que le plancher autorise
(2 cellules vides) ». **C'est incohérent** : si le mérite vaut `+∞` dès qu'une cellule est vide, un point
à 2 cellules vides ne peut pas être son minimum. L'erreur était de citer un profil mesuré avec le `log`
**écrêté**, où une cellule vide ne coûte que `−18.4` et reste donc bon marché.

Le profil fin avec le `log` **non écrêté** (`--profil-ratio 1.05`, itération 0, lignes `σ = 0.005`) dit
ce qui se passe réellement :

| `t` | ≥ 0.1228 | **0.1169** | 0.1114 | 0.1061 | … | 0.0591 |
|---|---|---|---|---|---|---|
| mérite `log` non écrêté | **`inf`** | **322.96** | 323.81 | 324.66 | ↗ | 333.77 |
| aire min | 0 | 1.59e-09 | 2.97e-09 | 4.19e-09 | | 3.94e-09 |
| cellules vides | ≥ 1 | **0** | 0 | 0 | | 0 |

À cette itération-là, **le mérite non écrêté décroît de façon monotone jusqu'au seuil de vidage**, puis
saute à `+∞` : son argmin est le plus grand pas qui ne vide aucune cellule, donc « minimiser » et « aller
au bord » sont la même règle.

> **MAIS CE N'EST PAS GÉNÉRAL, et je l'avais écrit comme tel (§ 24.10).** Un mérite purement monotone
> serait mal conçu ; celui-ci ne l'est pas. À l'itération 0 on est simplement trop loin pour que son
> minimum soit atteignable. Voir le § 24.10, qui mesure les deux régimes.

Une nuance quantitative reste, et elle est petite. Le plancher `eps` est **plus strict** que
« non vide » : le bord du vidage est à `t₀ ≈ 0.123`, le plancher mord à `α* = 0.107`, donc le plancher
interdit une plage de 15 % en `t` où le mérite s'améliorait encore — pour 0.5 % de mérite (322.96 contre
324.66). C'est pourquoi remplacer `eps` par zéro ne change aucun chiffre (mesuré au § 24.7 : identique
au diagramme près). Viser `t₀` exactement plutôt que `α*` serait la version exacte de la règle, et
`premiere_racine( niveau )` sait déjà le faire avec `niveau = 0` ; ça ne rapporterait que ces 0.5 %.

Ce qui reste vrai de la proposition, et qui vaut : **le critère d'acceptation, lui, doit bien être le
mérite `log`** — et c'est ce que fait `merite, départ α*`, qui prend `facteur · α*` et l'accepte si le
mérite `log` décroît. 13 diagrammes contre 19 pour `essai-limites` et 39 pour KMT.

### Ce que je retire

La phrase du § 24.8 — « la grossièreté de l'échelle dyadique nous protégeait » — est retirée. Elle
disait qu'un réglage arbitraire compensait une erreur d'objectif, ce qui n'est pas un résultat mais
l'aveu de ne pas maîtriser ce qui était mesuré. La vraie raison est géométrique et se démontre : la
contrainte est active à l'optimum, donc c'est la contrainte qu'il faut calculer.

## 24.10 Le mérite n'est PAS monotone — la contrainte est active tôt, inactive tard

Objection juste : un résidu purement monotone le long du rayon serait **mal conçu**, puisqu'il ne
pénaliserait jamais le dépassement. J'avais écrit la monotonie comme une propriété du mérite ; c'en est
une propriété **de l'itéré**, et c'est ce que deux profils fins montrent.

### Loin de la solution (itération 0, `max|a−ν|/ν = 1665`) : monotone sur tout l'admissible

| `t` | ≥ 0.126 | **0.1167** | 0.1081 | 0.1000 | … | 0.0397 |
|---|---|---|---|---|---|---|
| mérite `log` (non écrêté) | `inf` | **322.99** | 324.34 | 325.68 | ↗ | 338.41 |
| `log2` (NON CENTRÉ) | `inf` | **660.54** | 670.33 | 679.77 | ↗ | 764.74 |
| cellules vides | ≥ 1 | 0 | 0 | 0 | | 0 |

La direction de Newton visait `t = 1`, mais le modèle casse bien avant : les cellules commencent à mourir
vers `t ≈ 0.126`. **Tout le segment admissible est donc sur la branche descendante** — le minimum du
mérite est hors d'atteinte, et c'est la contrainte qui fixe le pas.

**Et `log2` non centré ne change rien** : même forme, même seuil, même absence de minimum intérieur. Le
centrage n'était pas la cause.

### Près de la solution (itération 10, BASCULE COUPÉE) : un minimum intérieur net, contrainte inactive

> **LA CONDITION SANS LAQUELLE CE PROFIL NE VEUT RIEN DIRE** — elle manquait ici, et c'est une objection
> de l'utilisateur qui l'a fait apparaître. Sous le réglage **par défaut**, la bascule tombe à
> l'itération 9 (`max|a−ν|/ν = 9.1e-01 ≤ 2`) : à partir de là `res_cur = LIN`, la branche `MERITE` est
> court-circuitée (`res_cur != LIN`), et le mérite `log` **n'est plus le juge de rien** — `--profil`
> l'imprime en spectateur. Le profil ci-dessous n'est donc valable que parce qu'il a été pris
> `--bascule-residu 0 --g-ecrete 0`, c'est-à-dire en `log` pur jusqu'à la convergence. **C'est un énoncé
> sur le `log` pur, pas sur l'algorithme par défaut.**

| `t` | 4.0 | 2.56 | 1.64 | 1.31 | **1.049** | 0.839 | 0.671 | … | 0.058 |
|---|---|---|---|---|---|---|---|---|---|
| mérite `log` | 19.50 | 10.46 | 4.23 | 1.93 | **0.2786** | 1.554 | 2.797 | ↗ | 7.53 |
| aire min | 7.6e-06 | 8.7e-06 | 9.3e-06 | 9.4e-06 | 8.9e-06 | 7.9e-06 | 7.1e-06 | | 4.6e-06 |
| cellules vides | 0 | 0 | 0 | 0 | 0 | 0 | 0 | | 0 |

**Minimum à `t ≈ 1.05`** — le pas de Newton, comme il se doit — et **aucune cellule vide même à `t = 4`**,
l'aire minimale restant à 7.6e-06 contre un plancher `eps ~ 4e-10`. La contrainte est totalement
inactive, et le mérite choisit `t ≈ 1` tout seul. `log` et `log2` y coïncident à sept chiffres, parce que
`moyenne(log x) → 0` près de la solution : c'est pourquoi le centrage ne pouvait rien changer.

### Et sous le DÉFAUT, la bascule sépare exactement les deux régimes

La conséquence de la note ci-dessus est mesurable. Sous le défaut, la dernière itération où le mérite
`log` juge quelque chose est la **8** (`max|a−ν|/ν = 7.5`), et son profil fin y est celui du régime
contraint, pas celui du régime libre :

| `t` | 1.000 | 0.800 | **0.640** | 0.512 | 0.410 | … | 0.055 |
|---|---|---|---|---|---|---|---|
| mérite `log` (écrêté) | 570.9 | 183.6 | **42.36** | 53.26 | 65.01 | ↗ | 105.2 |
| cellules vides | 929 | 85 | **0** | 0 | 0 | | 0 |

Le minimum admissible est au **dernier barreau sans cellule vide** : `t = 0.64`, le bord. (Les 570.9 et
183.6 ne sont finis que par l'écrêtage `g_ecrete = 1e-8` ; non écrêtés ce sont des `inf`, et la colonne
« vides » dit pourquoi.)

**Donc dans l'algorithme par défaut le mérite `log` ne gouverne QUE le régime contraint**, et le régime
libre est remis à `lin`, qui y prend `t = 1` — ce que le profil en `log` pur dit être précisément
l'optimum. Les deux régimes ne sont pas seulement « tôt » et « tard » : ils sont **de part et d'autre de
la bascule**.

Et ça donne au seuil `R` un sens qu'il n'avait pas. La contrainte cesse d'être active quand
`max|a−ν|/ν` tombe à O(1) — exactement là où `R = 2` déclenche. Le seuil n'est pas une constante
ajustée : il **marque la transition de régime**, ce qui explique le plateau du balayage du § 21.7 (`R` de
0.5 à 50 : 19–20 diagrammes ; `R = 1e9`, donc bascule immédiate : 30 ; `R = 0`, jamais : 20). Sur le
nuage propre et l'échelle dyadique, bascule et `log` pur font d'ailleurs le même compte (39 diagrammes
chacun) : ce que la bascule achète est ailleurs (les nuages dégénérés, et `essai-limites`).

### Ce que ça explique enfin, sans invoquer la chance

Les deux régimes sont le tableau classique — **contraint tôt, libre tard** — et ils expliquent d'un coup
tous les échecs de recherche linéaire mesurés plus haut :

* **tôt**, la contrainte est active : le pas optimal est **au bord**, donc il faut *calculer le bord*
  (`α*`, par cellule, pour moins qu'un diagramme) et non chercher un minimum qui n'est pas là. C'est
  pourquoi `merite, départ α*` fait 13 diagrammes et l'échelle dyadique 48 (§ 24.9) ;
* **tard**, la contrainte est inactive et le minimum est à `t ≈ 1` : Newton le prend déjà — et sous le
  défaut c'est `lin` qui le prend, la bascule ayant déjà eu lieu. Il n'y a rien à chercher non plus.

**Dans aucun des deux régimes une recherche linéaire n'a de quoi gagner** — dans le premier parce que
l'optimum est sur la frontière, dans le second parce qu'il est en `t = 1`. C'est la vraie raison, et elle
remplace la formule retirée au § 24.8 (« la grossièreté nous protégeait »), qui n'expliquait rien.

Ce qui resterait à essayer, et que ces profils désignent : la seule marge est entre `α*` (le plancher
`eps`) et `t₀` (le vidage), soit 15 % en `t` pour 0.4 % de mérite à l'itération 0. Autrement dit il n'y a
rien à gagner là non plus, et le pas est un problème **résolu** — ce qui reste est ailleurs (le § 22 pour
la direction, le § 8 pour le multi-échelle).

## 24.11 Peut-on remonter la bascule plus tôt ? Non — et le seuil est déjà au bord

Le § 24.10 dit que `R = 2` tombe **sur** la transition de régime. Monter `R`, c'est donc confier à `lin`
des itérations où la contrainte d'aire mord encore — exactement ce que le § 21 avait mesuré comme son
point faible. Le balayage le confirme sur les six cas (`scripts/scan_bascule_tot.sh`, diagrammes) :

| cas | `R=0` (jamais) | **`R=2`** | `R=10` | `R=50` | `R=200` | `R=1e9` (tout de suite) |
|---|---|---|---|---|---|---|
| 2D uniforme, dyadique | 7 | **7** | 8 (+1 recul) | 8 | 8 | 8 |
| **3D uniforme** | 6 | **6** | **9** (+2 reculs) | 9 | 9 | 9 |
| 2D lignes s0.005 | 39 | **39** | 41 | 42 | 45 | 78 |
| 2D lignes s0.02 | 29 | **29** | 28 | 27 | 28 | 41 |
| 2D aires égales (DÉGÉNÉRÉ) | 73 | **74** | 75 | 78 | 79 | 113 |
| 3D plans s0.02 | 16 | **17** | 17 | 18 | 22 | 27 |

Le seul gain apparent, 2D s0.02 à `R=50` (27 contre 29), s'arrête à un résidu 2000 fois plus lâche
(3.65e-07 contre 1.81e-10) : c'est une arrivée sous la tolérance une itération plus tôt, pas une
convergence plus rapide. Le 3D uniforme est le contre-exemple franc : **+50 % de diagrammes dès `R=10`**.

**L'asymétrie est le vrai enseignement.** Basculer plus TARD est presque gratuit partout (`R=0` ≈ `R=2`
à un diagramme près) — sauf sur le nuage dégénéré avec `essai-limites`, où ne jamais basculer explose :
**736 diagrammes et MAX ITERATIONS contre 53**. La contrainte n'est donc pas « tôt ou tard » mais
« pas trop tôt, et pas jamais », et `2` est le bord haut de la fenêtre gratuite.

### L'autre lecture : mesurer la transition au lieu de la seuiller ( `--bascule-pas` )

`R` est une constante à régler, et le § 24.10 dit ce qu'elle approxime : le moment où la contrainte
cesse de mordre. Ça ne s'approxime pas, ça **se mesure** — un pas plein accepté *est* la preuve que la
contrainte n'a pas mordu. D'où `--bascule-pas T` : basculer dès qu'un pas `>= T` a été accepté. Aucune
constante d'échelle, rien qui dépende du cas (diagrammes, `--bascule-residu 0` pour isoler) :

| cas | `R = 2` | `pas >= 1` | `pas >= 0.9` | `pas >= 0.5` |
|---|---|---|---|---|
| 2D uniforme (dyad. / limites) | 7 / 7 | 7 / 7 | 7 / 7 | 7 / 7 |
| 2D lignes s0.005 | 39 / 19 | 39 / 19 | 39 / 19 | 39 / 19 |
| 2D lignes s0.02 | 29 / 13 | 29 / 13 | 29 / 13 | 29 / 13 |
| 2D aires égales (DÉGÉNÉRÉ) | 74 / **53** | 73 / **53** | 73 / **53** | 74 / **53** |
| 3D uniforme | 6 / 6 | 6 / 6 | 6 / 6 | 6 / 6 |
| 3D plans s0.02 | 17 / 17 | 16 / 16 | 16 / 16 | 17 / 17 |

Égalité partout, aux deux `−1` près — qui sont encore des arrivées sous la tolérance une itération plus
tôt (3D plans : reste 4.01e-07 contre 1.96e-09). **Et le critère mesuré protège le cas dégénéré** : 53
diagrammes comme le seuil, contre 736 sans bascule du tout. C'est donc un remplacement viable, qui
supprime un paramètre.

**Mais il ne peut pas être plus tôt que le seuil, et il y a un cas où il est strictement moins bon.** Le
test porte sur le pas de l'itération précédente, et `t_prec = 0` à l'itération 0 : le critère mesuré ne
peut PAS tirer à l'itération 0. Or c'est précisément ce que la continuation de densité demande (§ 9.6,
et la note de `bascule_residu`) : chaque étape repart d'un résidu déjà petit, le seuil tire donc
immédiatement et le solve est `lin` du début à la fin — ce qui est vital, puisque `log` seul y est
catastrophique. Le seuil garde donc un rôle que la mesure ne peut pas reprendre, et les deux sont
cumulables (ils sont en OU, chacun latché). Le défaut reste `R = 2`, `bascule_pas = 0`.

### Ce que « plus tôt » voudrait dire vraiment

Aucun des deux mécanismes ne peut avancer la bascule, parce que la transition n'est pas un réglage :
c'est le moment où le pas de Newton `log` devient admissible, donc une propriété de l'itéré. Pour
l'atteindre en moins d'itérations il faut une meilleure **direction** (le § 22, qui fait 13 diagrammes)
ou un meilleur **départ** (le § 8, multi-échelle). La bascule, elle, est déjà au bon endroit.

## 24.12 Deux directions à la fois : `span{ Newton log, déplacement précédent }` (`--pas grille2`)

Maintenant que le mérite `log` a un vrai minimum intérieur dans la phase `log` (§ 24.10), une recherche
à **deux** variables a un sens qu'elle n'avait pas. L'instrument : `w + α d + β ( déplacement
précédent )`, évalué sur une grille — `α = t / 2^k` sur les lignes, `β` symétrique autour de 0 sur les
colonnes. Deux choix rendent la mesure lisible :

* **la colonne `β = 0` EST l'algorithme actuel** (la descente dyadique le long de la direction de
  Newton), donc la grille contient la référence et tout gain se lit comme un écart à cette colonne — un
  argmin qui reste en `β = 0` est un négatif franc, pas une absence de résultat ;
* la seconde direction est le **déplacement** précédent (`t_prec · d_prec`), pas la direction brute :
  sa norme est celle d'un pas qui a été accepté, donc `β` est sans dimension.

Coût : `g2_na × g2_nb` diagrammes par itération. C'est un instrument, personne ne propose ça en défaut.

### Le garde-fou qu'il faut lire AVANT la matrice : le cosinus

Si le déplacement précédent est colinéaire à la direction de Newton, `β` ne fait que réhausser `α` : le
plan est un rayon et un argmin en `β = 0` ne veut rien dire. Le mode imprime donc `cos( d, précédent )`.
Mesuré (2D lignes s0.005, itérations 1 à 6) : **0.989, 0.977, 0.906, 0.963, 0.902, 0.506**. Les deux
directions sont donc **quasi colinéaires** en champ lointain. C'est la limite honnête du négatif 2D qui
suit : il dit surtout que réhausser le pas ne sert pas, ce qu'on savait.

### En 2D : rien, et pour une raison structurelle

`β = 0` gagne à **toutes** les itérations de grille, sur les trois nuages 2D (uniforme, lignes s0.005,
aires égales). Et toute la moitié `β > 0` est **refusée par le plancher d'aire**, à tous les `α`, y
compris les plus petits. La raison n'est pas numérique : le pas précédent avait déjà été poussé jusqu'à
**sa propre limite** (`α*`), donc en redemander empile deux déplacements dont le premier touchait déjà le
bord. **Le déplacement précédent est inutilisable parce qu'il a déjà été utilisé à fond.**

### En 3D il se passe quelque chose — mais ce n'est pas une meilleure descente

3D plans s0.02, itération 3 (`cos = 0.981`), mérite `log` :

| `α` | `β = −1` | `β = −0.5` | `β = 0` | `β = 0.5` |
|---|---|---|---|---|
| **1** | 39.53 | **25.88** | *refusé* | *refusé* |
| 0.5 | 106.8 | 83.76 | 58.77 | *refusé* |
| 0.25 | 137.1 | 115.7 | 92.97 | 69.06 |

À `α = 1`, la colonne `β = 0` est **refusée** par le plancher d'aire — mais `β = −0.5` passe, et donne
25.88 contre 58.77 pour le meilleur point admissible de la colonne : **un facteur 2.3**. Même effet à
l'itération 1 (167.3 contre 202.1). Le mécanisme est donc clair, et ce n'est pas celui qu'on cherchait :
**défaire une partie du déplacement précédent RELÂCHE la contrainte d'aire** et laisse passer un `α`
plus long. La seconde direction ne sert pas de meilleure descente, elle sert de dégagement.

### Le verdict, et ce qu'il désigne

| cas | défaut (diag) | `grille2` (diag) | itérations |
|---|---|---|---|
| 2D uniforme | 7 | 11 | 6 → 6 |
| 2D lignes s0.005 | 39 ( 19 avec limites ) | 161 | 14 → 12 |
| 2D aires égales (dég.) | 74 | 219 | 16 → 13 |
| 3D plans s0.02 | 17 | 86 | 9 → **9** |

**Aucune itération gagnée là où la grille sort de la colonne** (3D : 9 contre 9), pour cinq fois les
diagrammes. Les gains de mérite par itération, réels (×1.2 à ×2.3), ne se transforment pas en
itérations — ce qui est cohérent avec le § 24.10 : gagner sur le mérite d'un pas contraint ne rapproche
pas de la solution, il ne fait que mieux occuper le bord.

Mais l'information est utile pour la suite, et elle est précise : **la seconde direction qui vaudrait
quelque chose est celle qui LIBÈRE la contrainte**, pas celle qui descend mieux. Le déplacement
précédent le fait par accident (en se défaisant) ; un span de directions de **résidus différents** le
fait par construction, puisque chacune place la cible ailleurs — et c'est exactement le span du § 22,
le seul essai qui ait atteint 13 diagrammes. Le prochain essai de direction est donc là, pas dans la
mémoire du pas.

## 24.13 La direction SONDÉE au bout du rayon : la courbure, enfin une seconde direction qui sert

Le § 24.12 a donné le critère : la seconde direction utile est celle qui **libère la contrainte**. Le
déplacement précédent ne le fait que par accident. La bonne candidate vient de la non-linéarité
elle-même : `d` est la direction de Newton **au départ**, et le problème est non linéaire, donc elle
n'est plus la bonne au bout du pas. On recalcule donc Newton **au point de sonde** — juste avant qu'une
cellule casse — et on prend

`e = d( sonde ) − d`,

qui est une différence finie de la direction le long du rayon, c'est-à-dire la **courbure en `t`**. La
grille devient un mélange, `w + α ( d + β e )` : `β = 0` est Newton pur (l'algorithme actuel), `β = 1`
est la direction de l'arrivée, et les deux axes sont sans dimension. (`--pas grille2 --g2-dir sonde`.)

### Le point de sonde doit être ADMISSIBLE — un piège qui coûte 363 secondes

Première version : sonder en `w + t d` avec `t` le pas courant. En 2D ça marche, parce que la passe des
limites a déjà ramené `t` sous `α*`. **En 3D il n'y a pas de passe de limites**, donc `t = 1`, donc le
point de sonde a des cellules vides — et le laplacien y a des lignes nulles. Mesure : le multigrille y
a passé **20 032 itérations et 363 s** avant d'échouer, et le mode retombait silencieusement sur `β = 0`
(`|e|/|d| = 0` dans la trace : le signe qui a permis de le voir).

La correction n'est pas un rustine, c'est le bon ordre : on descend d'abord la colonne `β = 0`, qui est
l'échelle de KMT, et **son premier point admissible EST le point de sonde**. Il devient alors gratuit :
c'est le point que l'amortissement allait prendre de toute façon.

### La mesure : `e` n'est pas colinéaire, et il est largement OPPOSÉ à `d`

| | `cos( d, e )` | `|e|/|d|` | part neuve `|e_⊥|/|d|` |
|---|---|---|---|
| 2D lignes s0.005, it. 0 à 6 | −0.66 à **−0.93** | 0.17 à 0.78 | 0.13 à 0.27 |
| 3D plans s0.02, it. 0 à 3 | −0.88 à **−0.97** | 0.56 à 0.70 | 0.16 à 0.26 |

Au bout du rayon, Newton veut **revenir** : voilà la courbure, et elle est grosse. À comparer au
déplacement précédent du § 24.12, colinéaire à 0.90–0.99. Attention cependant : comme `e` est presque
antiparallèle à `d`, l'essentiel de `β` ne fait que **raccourcir le pas**, ce que `α` couvre déjà. La
part réellement neuve est la composante orthogonale, `|e_⊥|/|d| ≈ 0.2` — modeste, et c'est elle qui
mesure ce qu'il y a à gagner.

### Ce qu'elle gagne : elle DÉBLOQUE le pas plein

3D plans s0.02, itération 1, mérite `log` :

| `α` | `β = 0` | `β = 0.5` | `β = 1` |
|---|---|---|---|
| **1** | *refusé* | *refusé* | **49.49** |
| 0.5 | *refusé* | 105.1 | 122.8 |
| 0.25 | 141.6 | 149.2 | 156.2 |

Le meilleur point de la colonne `β = 0` est 141.6 à `α = 0.25` ; le mélange passe à `α = 1` et donne
**49.49**, soit un **facteur 2.9**. Même effet aux itérations 2 (14.8 contre 36.1) et 3 (3.27 contre
6.64). L'argmin sort de la colonne `β = 0` à **toutes** les itérations de la phase `log`.

### Le bilan, honnête : la direction est réelle, la recherche coûte plus qu'elle ne rapporte

Itérations et diagrammes, `--residu log` partout, référence = `essai-limites 0.9` :

| variante | 2D uniforme | 2D lignes s0.005 | 2D aires égales (dég.) | 3D plans s0.02 |
|---|---|---|---|---|
| référence ( limites + log ) | 6 it / **7** | 12 it / **19** | 13 it / **53** | 9 it / **17** |
| sonde, grille 5×5 | 6 it / 31 | **11** it / 180 | **12** it / 214 | **7** it / 104 |
| sonde, `β ≥ 0` ∈ {0, ½, 1}, 3 `α` | — | **11** it / 68 | — | **7** it / 40 |
| sonde, `β ≥ 0` ∈ {0, 1}, 3 `α` | — | 11 it / 47 | — | 8 it / 29 |
| sonde, 1 `α` × 3 `β` | 6 it / 9 | 11 it / 26 | 12 it / 60 | 9 it / 21 |
| précédent ( § 24.12 ), grille 5×5 | 6 it / 11 | 12 it / 161 | 13 it / 219 | 9 it / 86 |

Trois lectures nettes :

1. **la sonde gagne des itérations là où le précédent n'en gagnait aucune** : une en 2D dur, **deux sur
   neuf en 3D** (7 contre 9) ;
2. **il faut de la résolution en `β`** : retirer `β = 0.5` coûte une itération en 3D (8 au lieu de 7), et
   se limiter à un seul `α` la perd entièrement (9 it) — ce qui explique aussi pourquoi les variantes
   les moins chères ne gagnent rien en 3D ;
3. **et la recherche coûte plus qu'elle ne rapporte** : le meilleur compromis mesuré fait 40 diagrammes
   en 3D contre 17, pour 2 itérations économisées. Comme instrument c'est concluant ; comme algorithme
   ce n'est pas encore un gain.

### Ce que ça désigne, précisément

Le coût est **entièrement** dans le balayage de `β` : chaque point d'essai est un diagramme. Or c'est
exactement ce que le § 22 sait faire sans diagramme — `PolyMulti` donne le polynôme **exact** de l'aire
de chaque cellule sur un span de directions, avec son rayon d'exactitude. Jusqu'ici on le nourrissait
d'un span de **résidus** (`lin`, `log`, `barrière`) ; le § 24.13 fournit un span meilleur, mesuré
non-colinéaire et dont on sait ce qu'il apporte. La suite est donc `span{ d, d( sonde ) }` évalué par le
modèle polynomial, pas par des diagrammes.

Deux remarques de coût qui vont dans le même sens : le solve de la sonde est **celui de l'itération
suivante** quand `β = 0` gagne (même point, même laplacien, même second membre), donc son coût marginal
n'est payé que lorsque le mélange sert ; et un `β` obtenu par le polynôme ne coûterait aucun diagramme
du tout.

## 24.14 Le modèle polynomial sur le span `{ d, d(sonde) }` : l'exploration devient négligeable — mais le TEMPS ne suit pas

Le § 24.13 avait localisé le coût : chaque point d'essai en `(α, β)` était un diagramme. Or à
**connectivité fixe** l'aire de chaque cellule est un polynôme exact du déplacement (§ 22), donc
l'exploration ne coûte plus rien. `--g2-modele` construit `PolyMulti` sur le span `{ d, e }` et évalue
le mérite `log2` depuis les aires modélisées ; `--g2-desc K` ajoute une **descente de gradient** par
`PolyMulti::gradient`. Coût en diagrammes : une construction de l'ordre d'**un**
diagramme — c'est ce que le § 22 avait mesuré — plus **un** diagramme de vérification à l'argmin. Elle
est comptée comme un diagramme ci-dessous. **En temps de paroi le mode perd quand même de 31 à 75 %**,
et la sous-section « le temps de paroi » dit exactement pourquoi : ce n'est pas l'exploration.

Deux détails qui font toute la différence sur le coût :

* la colonne `β = 0` s'arrête au **premier point admissible** — c'est tout ce dont la sonde a besoin, le
  reste de l'échelle étant donné par le modèle. Descendre l'échelle entière en diagrammes coûtait
  `g2_na − 1` diagrammes par itération pour rien : **54 diagrammes au lieu de 24** sur 2D lignes s0.005 ;
* la vérification est unique : on ne paie pas la grille, on paie le point retenu.

### Les chiffres ( `--residu log` partout, itérations / diagrammes )

| variante | 2D uniforme | 2D lignes s0.02 | 2D lignes s0.005 | 2D aires égales (dég.) |
|---|---|---|---|---|
| référence ( limites + log ) | 6 it / **7** | 9 it / **13** | 12 it / **19** | 13 it / **53** |
| grille par DIAGRAMMES ( § 24.13 ) | 6 it / 15 | 9 it / 50 | 11 it / 68 | 12 it / 102 |
| **modèle 6 × 21 + descente** | **5** it / 8 | 9 it / 20 | **11** it / 24 | **12** it / 58 |
| modèle, sans descente | 6 it / 9 | 9 it / 20 | 11 it / 26 | 12 it / 60 |
| modèle 6 × **81** + descente 40 | 5 it / 8 | 9 it / 20 | 11 it / 24 | 12 it / 58 |
| modèle, `β ≤ 2` | 5 it / 8 | 9 it / 20 | 11 it / 26 | 13 it / 61 |

Trois lectures, dont une négative importante :

1. **le modèle remplace la grille de diagrammes à −65 %** (24 contre 68 sur le cas dur, 8 contre 15 sur
   l'uniforme) pour le même nombre d'itérations, et il gagne une itération sur trois cas des quatre —
   l'uniforme 2D descend à **5 itérations**, le meilleur compte jamais mesuré sur ce cas ;
2. **la descente de gradient sert** : elle vaut une itération sur l'uniforme (5 au lieu de 6) et −8 % de
   diagrammes sur le cas dur (24 au lieu de 26). Elle sort des lignes de la grille, ce qu'un raffinement
   ne fait pas ;
3. **raffiner la grille ne sert à RIEN** : 21 → 81 valeurs de `β` donne des résultats *identiques* sur
   les quatre cas, et élargir à `β ≤ 2` ne gagne rien non plus. Donc ce qui reste n'est **pas** un
   problème de résolution de la recherche.

### Ce qui limite, et c'est le § 22 qui l'avait déjà dit

La trace donne la mesure : `rayon min` vaut **1.3e-07 à 1.6e-05** alors que les pas retenus sont de
**0.1 à 0.5**, et **91 701 cellules sur 99 944** sont hors de leur rayon au point évalué. L'écart du
mérite modélisé au mérite réel suit :

| itération | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| mérite modèle | 616.5 | 357.6 | 265.6 | 212.6 | 147.4 | 76.5 |
| mérite réel | 310.5 | 252.4 | 219.8 | 191.0 | 141.9 | 75.8 |
| écart | **−50 %** | −29 % | −17 % | −10 % | −3.7 % | **−0.9 %** |

Ce n'est pas une surprise et ça ne contredit pas le § 22, qui avait mesuré les deux faits qui
expliquent tout : `rayon` est **très pessimiste** (à `‖t‖∞ = 6e-2` il ne certifie que 27 % des cellules
alors que l'erreur médiane est au niveau machine), et le modèle est **conservateur sur le critère** —
quand la combinatoire d'une cellule casse, son aire prédite part n'importe où et le mérite prédit
**explose**. C'est exactement ce qu'on lit ici : le modèle annonce toujours **plus** que la vérité, donc
la recherche fuit d'elle-même les régions où il ne vaut rien, et l'argmin qu'elle rend reste bon. Le
modèle **classe** bien avant de **chiffrer** juste.

Et il devient juste là où on en aurait le moins besoin : à 0.9 % près à l'itération 5, juste avant la
bascule. Sa zone de validité est la **fin** de la phase `log`, pas le début.

### Le temps de paroi, qui tranche ( `job -b` )

Les diagrammes ne disent pas tout, et ici ils mentent par omission.

| | itérations | diagrammes | **temps** |
|---|---|---|---|
| 2D uniforme, défaut ( KMT + log ) | 6 | 7 | **1.10 s** |
| 2D uniforme, référence ( limites ) | 6 | 7 | 1.14 s |
| 2D uniforme, **modèle + sonde** | **5** | 8 | 1.49 s ( **+31 %** ) |
| 2D lignes s0.005, défaut | 14 | 39 | 6.40 s |
| 2D lignes s0.005, référence | 12 | 19 | **4.30 s** |
| 2D lignes s0.005, **modèle + sonde** | **10** à 11 | 23 à 24 | 7.4 à 7.7 s ( **+75 %** ) |
| 2D aires égales, référence | 13 | 53 | **7.55 s** |
| 2D aires égales, **modèle + sonde** | **12** | 58 | 11.85 s ( +57 % ) |

**Le meilleur compte d'itérations jamais mesuré sur ce banc** (10 contre 12 sur le cas dur, 5 contre 6
sur l'uniforme) et pourtant une **perte de 31 à 75 % en temps**. Le détail dit où, et ce n'est pas là
qu'on l'attendait (2D lignes s0.005, contre la référence) :

| poste | référence | modèle | écart |
|---|---|---|---|
| diagrammes | 1.29 s | 1.18 s | **≈ 0** |
| résolution | 2.74 | 3.79 | +1.05 |
| limites | 0.07 | 1.30 | +1.24 |
| reste ( construction + grille + descente ) | 0.06 | 1.11 | +1.05 |
| TOTAL | **4.36** | 7.66 | +3.30 |

**Les diagrammes coûtent exactement la même chose** — le pari est tenu de ce côté. Le surcoût est
**trois postes à peu près égaux** : la construction du modèle, la passe de limites globale qui donne
`α*` à chaque itération, et le **solve linéaire supplémentaire de la sonde** (17 hiérarchies AMG au lieu
de 12, 956 itérations linéaires au lieu de 681).

L'évaluation, elle, n'est plus un sujet, et ça s'est mérité : en mono-thread elle coûtait 1.69 s. Un
seul `parallel_for` sur les cellules avec un accumulateur par ( fil, point ) — le motif du § 22, une
seule traversée de `pm2` pour toute la grille — la ramène à 1.11 s. Et surtout, **réduire la grille de
126 points à 6 ne change plus le temps** (7.68 s contre 7.43 s) : la preuve que l'exploration est
devenue négligeable. Ce qui laisse deux leviers identifiés et **non faits** :

* quand `β = 0` gagne, le solve de la sonde **est** celui de l'itération suivante (même point, même
  laplacien, même second membre) : il est réutilisable au lieu d'être refait ;
* `α*` peut sortir des **racines scalaires du modèle** (c'est ce que fait le § 22.3) au lieu d'une passe
  de limites globale.

Ces deux-là valent environ les deux tiers du surcoût. Sans eux, le mode reste un instrument.

### Une remarque au passage sur le `--pas modele` du § 22

Le même banc le fait sortir de route sur le nuage dégénéré : **40 itérations, 619 diagrammes, 70 s**
contre 13 / 53 / 7.6 pour la référence. Son score de 13 diagrammes du § 22.3 est un score de **nuage
propre** ; il ne survit pas aux germes quasi coïncidents. Le span `{ d, sonde }`, lui, y tient (12 / 58).

### Portée : 2D seulement

`polynomes_multi` porte un `static_assert( PD::dim == 2 )`. Le gain le plus net du § 24.13 — **deux
itérations sur neuf en 3D** — n'est donc **pas** récupérable par cette voie en l'état : il faudrait
écrire le polynôme multi-directions du volume d'une cellule 3D. C'est la limite à connaître avant de
choisir ce qu'on garde.

## 24.15 Le pas SANS AUCUN DIAGRAMME — les deux leviers, et ce qu'il reste

Correction d'abord, parce qu'elle change la structure : **il n'y a pas de « diagramme de
vérification »**. Le diagramme au point choisi *est* celui de l'itération suivante, qu'on doit calculer
de toute façon. Donc on est **optimiste** : on va au point que le modèle désigne, et on ne recule que si
le vrai mérite n'y descend pas — un backtracking qui, mesuré, **n'arrive jamais** sur les nuages du banc.
Et le point de sonde lui-même n'a pas besoin de diagramme : ses aires sortent du modèle, à connectivité
fixe.

Le pas se choisit donc **sans reconstruire une seule cellule** :

1. le modèle à **une** direction, bâti là où `pd` est déjà (aux poids `w`) : `α*` sort des **racines**
   (§ 7) et le modèle donne les **aires** au point de sonde — c'est le **levier 2**, et la passe de
   limites globale disparaît ;
2. le résidu en ce point se calcule depuis ces aires, et son système se résout avec **le même
   laplacien** — donc sans assemblage et **sans nouvelle hiérarchie AMG** : 13 hiérarchies pour 13
   itérations, pas 26. C'est ce qui remplace le levier 1 (voir plus bas) ;
3. `e = d(sonde) − d` est la courbure, et le modèle à **deux** directions rend l'exploration de
   `(α, β)` gratuite — grille puis descente de gradient ;
4. on va à l'argmin, un diagramme, et c'est celui de l'itération suivante.

### Ce que ça donne ( nuages PROPRES seulement )

Les germes quasi coïncidents sont **sortis du banc** : ils sont agrégés avant la résolution (§ 23), donc
les faire porter un jugement sur l'amortissement n'a pas de sens.

| | itérations | diagrammes | temps | Krylov |
|---|---|---|---|---|
| 2D uniforme, référence | 6 | 7 | **1.11 s** | 226 |
| 2D uniforme, **modèle** | **5** | **6** | 1.12 s | 221 |
| 2D lignes s0.02, référence | **9** | 13 | **2.17 s** | 314 |
| 2D lignes s0.02, **modèle** | 10 | **11** | 3.48 s | 556 |
| 2D lignes s0.005, référence | **12** | 19 | **4.24 s** | 681 |
| 2D lignes s0.005, **modèle** | 13 | **15** | 6.53 s | 1164 |

**Les diagrammes passent sous la référence partout** (6 contre 7, 11 contre 13, 15 contre 19), avec
**un seul diagramme par itération et zéro recul**. Sur l'uniforme le temps est maintenant à égalité
(1.12 contre 1.11 s) pour une itération de moins. Sur les lignes il reste 60 % de retard, et la colonne
Krylov dit exactement pourquoi : **le solve de la sonde double le travail linéaire** (1164 contre 681).

### Deux négatifs mesurés, à ne pas répéter

**Le levier 1 tel qu'il était conçu n'a plus d'objet.** « Réutiliser le solve de la sonde comme celui de
l'itération suivante » supposait que le point de sonde soit un vrai point où l'on irait. Il n'en est plus
un : c'est un point du modèle, et le point retenu est ailleurs (`w + α d + β e`). Ce qui reste, et qui
marche, est la reprise de **hiérarchie** : le système de la sonde a le même laplacien, donc AMGCL ne
remonte rien.

**Une tolérance lâche pour la sonde ne sert à rien** — alors que l'idée est bonne (cette résolution ne
sert qu'à définir une direction de recherche). Mesure : `--g2-tol` à 1e-10, 1e-3 et 1e-2 donnent
**exactement** 1164 itérations de Krylov. La raison est dans `Lineaire.h` : AMGCL fige la tolérance dans
l'objet solveur au moment où la hiérarchie est montée (`prm.solver.tol`), et `resout_encore` rappelle cet
objet — changer `tol` après coup n'a aucun effet. Le crochet est en place (`Lineaire::tolerance`), mais
il faudrait monter un second solveur sur la même hiérarchie pour que ça porte.

**Et la précision de la sonde n'est PAS ce qui achetait les itérations.** Le témoin `--g2-sonde-reelle`
(un vrai diagramme au point de sonde, donc `e` exact) est **moins bon** : 12 it / 21 diag contre 10 / 11
sur s0.02, 15 / 27 contre 13 / 15 sur s0.005. Payer un diagramme pour une meilleure courbure est donc
une perte sèche — ce qui contredit l'hypothèse que j'avais avancée, et c'est le témoin qui tranche.

### Où en est le compte

Le mode fait maintenant **moins de diagrammes que la référence** et le **meilleur compte d'itérations**
sur l'uniforme, à temps égal. Ce qui l'empêche de gagner sur les cas durs est un poste unique et
identifié : **un solve linéaire de plus par itération**. Les deux façons de l'attaquer, dans l'ordre de
promesse : monter un second solveur AMGCL à tolérance lâche sur la hiérarchie existante (le crochet est
là) ; ou obtenir `e` sans résoudre, par exemple en extrapolant la direction depuis l'itération précédente
— mais le § 24.12 a mesuré que la mémoire du pas est colinéaire, donc cette seconde piste part avec un
handicap connu.

## 24.16 Le coût du solve de plus : ce qui marche, et le Cholesky incomplet qui ne marche pas

### Une seule construction de modèle par itération

La construction à **une** direction ne servait qu'à deux choses : `α*` et les aires du point de sonde.
Les deux sortent gratuitement de ce qu'on a déjà. Le système résolu est `L d = b`, et `L` **est** la
dérivée des aires (`da = L d`) — donc au premier ordre

`A_i( α ) = a_i + α b_i`,

avec le second membre **déjà calculé** : pas même un produit matrice-vecteur. Et le `α*` du premier ordre
se lit de la même façon, `min_i ( eps − a_i ) / b_i` sur les `b_i < 0`. Le `α*` exact, lui, sort ensuite
du modèle à deux directions restreint à `β = 0`.

Le gain est **plus petit que je ne l'avais annoncé** : `reste` passe de 1.60 à 1.39 s sur 2D lignes
s0.005, pas de 1.60 à 0.8. La construction `nk = 1` était bon marché ; ce qui coûte, c'est la
construction `nk = 2` **et** la recherche, ensemble ~0.09 s par itération — un diagramme en vaut 0.07.
Et le point de sonde au premier ordre est moins précis : avec AMGCL et Cholesky on passe de 13 à 14
itérations. Le multigrille maison, lui, garde **12 itérations / 13 diagrammes**.

### Le solveur change tout, parce qu'on résout DEUX FOIS LA MEME MATRICE

Le solve de la sonde porte exactement le même `L`. Trois façons d'en profiter, mesurées (2D lignes
s0.005, `job -b`) :

| | it | diag | total | `lin` | Krylov |
|---|---|---|---|---|---|
| référence, AMGCL | 12 | 19 | 4.24 s | 1.72 | 681 |
| référence, **Cholesky** | 12 | 19 | **3.75 s** | **0.85** | — |
| modèle, AMGCL | 14 | 16 | 7.22 s | 3.57 | — |
| modèle, **mg maison `recycle 8`** | **12** | **13** | 5.49 s | 3.24 | — |
| modèle, **Cholesky** | 14 | 16 | **5.04 s** | **1.06** | — |

* **Cholesky rend le solve de plus presque gratuit** : 13 factorisations pour 13 itérations et **26**
  descentes-remontées, donc `lin` 1.06 s pour 26 solves contre 0.85 s pour 12. Le surcoût du second
  solve tombe à **+0.21 s** au lieu de +1.17. C'est la bonne réponse à « mettre en commun la
  préparation » : quand le coût est dans la factorisation et qu'on l'utilise deux fois, il est amorti.
* **Le recyclage de sous-espace de notre multigrille donne le meilleur compte d'itérations** (12 / 13
  contre 14 / 16). Il garde les `recycle` dernières solutions et démarre sur la projection de Galerkin
  sur leur span ; le mode modèle enchaîne désormais **deux systèmes corrélés par itération**, donc il a
  enfin de quoi vivre — ce que l'en-tête de `main_newton.cpp` avait prédit : « c'est le RÉGIME qui
  décide ». `recycle 8` vaut mieux que le défaut 2 (5.49 s contre 6.23).
* Réserve : « Cholesky est le bon solveur » serait faux. En 3D il met **790 s** contre 7.4 pour le
  multigrille maison. Ce qui est général, c'est le principe, pas le solveur.

### Le Cholesky incomplet sans nouveaux termes : essayé, et il perd d'un ordre de grandeur

Ça n'avait jamais été testé. C'est fait (`--solver amg --amg-var 3` : CG préconditionné par un IC(0) sur
le motif de `L`, sans aucun terme de remplissage, refait à chaque itération) :

| ( 8 fils ) | AMG agrégation+spai0 | **CG + IC(0)** | mg maison |
|---|---|---|---|
| 2D uniforme | 1.09 s, **226** Krylov | 7.82 s, **5 424** | 1.57 s, 277 |
| 2D lignes s0.005 | 4.50 s, **681** | 29.17 s, **22 642** | 4.60 s, 937 |
| 3D plans s0.02 | 8.20 s, **212** | 11.91 s, **2 141** | 7.44 s, 334 |

**De 10 à 33 fois plus d'itérations de Krylov.** Et la raison n'est pas le coût de construction — la
montée de l'IC(0) est comparable à celle de l'AMG (0.66 s contre 0.87 sur le cas dur) : c'est que sans
correction grossière, une factorisation incomplète ne touche pas aux **basses fréquences**. Le
conditionnement reste en `h^-2`, donc le nombre d'itérations croît comme `n^(1/2)` en 2D. C'est
exactement ce que le multigrille est fait pour éviter, et pourquoi il est optimal ici.

Ce constat ne dépend pas du fait de le garder « du début à la fin » : l'échec est l'absence de niveau
grossier, pas l'obsolescence des coefficients. Sur ce second point d'ailleurs la mesure existait déjà,
dans la note de `Lineaire::refaire` : **geler le préconditionneur d'AMGCL entre itérations coûte +79 %
d'itérations de CG**, parce que son `spai0` du niveau fin reste celui de l'ancienne matrice et qu'aucune
API ne permet de le rafraîchir seul — alors que le `rebranche` de notre multigrille **recalcule** les
coefficients du niveau fin sur les nouvelles valeurs et n'y perd que **11 %**.

Conclusion sur la question posée : la bonne façon d'exploiter « beaucoup de systèmes qui se ressemblent »
n'est pas une factorisation incomplète figée, c'est ce que notre solveur fait déjà — garder la
hiérarchie, **rafraîchir le niveau fin**, et recycler le sous-espace. Et quand deux solves d'une même
itération partagent la matrice, une factorisation directe bat tout le monde sur le second, là où elle
tient en mémoire.

## 24.17 Le Cholesky multi-échelle : 9 itérations de Krylov par résolution, et une factorisation scalaire qui ruine tout

Le § 24.16 a mesuré qu'un IC(0) sur le motif brut perd un ordre de grandeur, et a nommé la cause : sans
niveau grossier, une factorisation incomplète ne touche pas aux basses fréquences. Le Cholesky
multi-échelle (**Chen, Schäfer, Huang, Desbrun, SIGGRAPH 2021**) est la réponse exacte à ce défaut —
garder le remplissage **nul**, mais sur un motif **multi-échelle** et dans un ordre
**fin-vers-grossier**. Implémenté dans `Multichol.h` (`--solver mchol`) :

1. **l'ordre** : la structure de niveaux de l'ordre maximin inversé, par décimation — le niveau `k+1`
   est un sous-ensemble maximal du niveau `k` dont les points sont à plus de `2^k` espacements l'un de
   l'autre. **Le test se fait par un parcours du graphe du laplacien borné en distance euclidienne**,
   pas par une grille de pas fixe : ce détail vaut un facteur 4 sur la qualité, parce qu'il suit la
   densité et bâtit 10 à 12 niveaux au lieu de 5 ;
2. **le motif** : `S = { (i,j) : dist(x_i,x_j) ≤ ρ · min(ℓ_i, ℓ_j) }`, union **tous** les non-nuls de
   `A` — le papier interdit d'en perdre un seul ;
3. **la factorisation** : Cholesky incomplet à remplissage nul sur ce motif, colonne par colonne. La
   section que le papier consacre aux pivots négatifs ne nous concerne pas : notre matrice est une
   **M-matrice** à diagonale dominante, classe sur laquelle l'IC ne casse pas (Meijerink–van der
   Vorst). Le garde-fou est là et compte ses passages : **zéro** partout.

Le motif est purement **géométrique** : il ne dépend que des positions des germes, qui ne bougent pas.
Il est bâti **une fois** pour tout le solve, et chaque itération ne refait que les valeurs — le critère
du § 24.16 que le `spai0` gelé d'AMGCL ne sait pas tenir.

### La qualité : le papier a raison, et largement

2D uniforme, `n = 1e5`, 6 résolutions, **itérations de Krylov par résolution** :

| préconditionneur | Krylov / solve | termes / colonne |
|---|---|---|
| **mchol, ρ = 7** | **9** | 174 |
| **mchol, ρ = 5** | **18** | 90 |
| AMGCL agrégation+spai0 | 38 | — |
| multigrille maison | 46 | — |
| mchol, ρ = 2 | 281 | 16 |
| **IC(0) du § 24.16** | **904** | 7 |

**De 904 à 9 itérations par résolution — un facteur 100 — et quatre fois mieux qu'AMGCL.** Même
remplissage nul, même type de factorisation : seuls l'ordre et le motif changent. Le diagnostic du
§ 24.16 est confirmé de la façon la plus directe possible.

### Le coût : la factorisation scalaire, et c'est exactement ce que le papier optimise

| 2D uniforme | facto ( 6 ) | solve | TOTAL |
|---|---|---|---|
| AMGCL | 0.25 s | 0.53 s | **1.14 s** |
| mchol ρ = 2 | 2.29 | 10.02 | 12.69 |
| mchol ρ = 5 | 21.11 | 3.36 | 24.88 |
| mchol ρ = 7 | **57.91** | 3.26 | 61.60 |

À `ρ = 7` l'application n'est **plus** le problème : 54 itérations de Krylov en tout, 3.26 s. C'est la
**factorisation** qui coûte, 9.6 s par itération de Newton pour 174 termes par colonne. Or c'est
précisément là que le papier met son ingénierie : **supernœuds** (BLAS-3) et **coloriage multicolore**,
dont je n'ai ni l'un ni l'autre — ici c'est une boucle colonne par colonne scalaire et séquentielle.
La réserve était écrite dans l'en-tête de `Multichol.h` **avant** la mesure ; elle porte maintenant sur
le seul poste qui reste.

À noter que ça corrige mon attente : je prévoyais que la descente-remontée séquentielle serait le
goulot (c'est ce qui avait tué l'IC(0)). À bon `ρ` elle ne l'est pas, parce qu'il y a **neuf**
itérations à payer.

### L'échelle `ℓ` : le point dur, et la mesure y contredit l'intuition

Le papier donne pour `ℓ` un raccourci « quasi uniforme » : **une échelle uniforme par niveau**,
`( volume / nb du niveau )^(1/d)`. Sur nos nuages **groupés** (cinq lignes à σ = 0.005) il explose,
parce que l'espacement global surestime l'espacement local de plusieurs ordres :

| 2D lignes s0.005 | Krylov / solve | termes / col | TOTAL |
|---|---|---|---|
| AMGCL | 57 | — | **4.15 s** |
| mchol ρ = 2 | 150 | 121 | 126 s |
| mchol ρ = 5 | 55 | **440** | 418 s |

J'ai donc essayé une échelle **locale** — la distance maximin au plus proche germe de niveau au moins
égal, littéralement le `ℓ_i` de l'article. Elle tient le remplissage **mais elle détruit la qualité** :
sur l'uniforme, à 65 termes par colonne elle fait **242** itérations par résolution, quand l'échelle par
niveau en fait **18** à 90 termes. **Un facteur 13 en faveur du papier**, à densité comparable.

Ce n'est pas un détail d'implémentation : la théorie veut que le support colle à celui de l'ondelette
**du niveau**, donc à une propriété de niveau ; la faire varier d'un point à l'autre casse l'uniformité
de la multirésolution. Les deux options restent disponibles (`--mchol-ech 0 | 1`), défaut celle du
papier.

Reste donc un manque **précis** : une échelle **uniforme par niveau** mais dont la constante est
estimée **localement**. Aucune des deux variantes testées ne l'est.

> **ET L'IMPLEMENTATION DE REFERENCE NE LE FAIT PAS NON PLUS.** Le code des auteurs est public
> (`gitlab.inria.fr/geomerix/ichol`, GPL-3.0). Son `chol_level::calc_supp_scale` lit :
> `h = sqrt( Vol / node_num ); l = sqrt( 2 ) * h * Ones( node_num )` en 2D, `cbrt` et `sqrt( 3 )` en
> 3D. C'est **exactement** le raccourci global, le `sqrt( 2 )` ne faisant que redimensionner `rho`.
> Donc la limite mesurée ici sur les nuages groupés est une limite **de la méthode publiée**, pas de
> cette implémentation : ces nuages sont hors de son domaine validé. La question est close.

### Verdict

La méthode fait ce que le papier annonce, et mieux que je ne l'attendais : **le meilleur
préconditionneur du banc en nombre d'itérations, d'un facteur 4 sur AMGCL**. Elle n'est pas utilisable
ici en l'état, pour deux raisons qu'il faut distinguer : une d'**ingénierie** — supernœuds et
parallélisme, que le papier a et que je n'ai pas, et qui est tout l'écart sur le nuage uniforme — et une
de **fond pour notre cas d'usage** — l'échelle sur un nuage non uniforme. Le code reste dans le banc
comme instrument : c'est la seule mesure qui explique *pourquoi* l'IC(0) échouait.

### Ce qui existe déjà, et sous quelle licence

| brique | où | licence | état sur la machine |
|---|---|---|---|
| l'IC multi-échelle du papier, **supernœuds compris** | `gitlab.inria.fr/geomerix/ichol` | **GPL-3.0** | clonable ( `supernode.cc`, `ichol_pattern.h`, `ptree.cc`, `nanoflann` ) |
| Cholesky **supernodal** complet | CHOLMOD + `Eigen/CholmodSupport` | **LGPL-2.1+** ( le GPL-2+ est CHOLMOD/GPU, SPQR, RBio ) | `libcholmod5` **installé**, en-têtes `libsuitesparse-dev` absentes |
| HSC, « sparsify and compensate » | `github.com/dilipkay/hsc` | — | MATLAB + mex |
| FSAI ( application par **matvec**, donc parallèle ) | hypre | Apache-2.0 / MIT | absent |
| ISAI | Ginkgo | BSD-3 | absent |
| BLR / HSS pour la 3D | MUMPS, STRUMPACK | CeCILL-C, BSD | absents |
| Cholesky sparsifié pour laplaciens | `Laplacians.jl` ( `approxchol_lap` ) | MIT | Julia |
| AMG, ILU(0), SPAI | AMGCL | MIT | **installé**, c'est notre défaut |

Deux conséquences pratiques. D'abord la licence tranche avant la technique : notre dépôt n'a **aucun**
fichier de licence, donc lier du GPL-3 engagerait l'ensemble ; lire `supernode.cc` pour savoir ce que
coûterait une version supernodale de notre IC, non.

Ensuite l'essai le plus rentable ne demande presque pas de code : le § 24.16 a mesuré que le **Cholesky
complet est déjà le meilleur solveur 2D** ( linéaire 0.85 s contre 1.72 pour AMGCL ). Or notre Cholesky
est `Eigen::SimplicialLDLT`, **scalaire** : `CholmodSupernodalLLT` le remplace en quelques lignes et
sous LGPL.

> **FAIT, ET LE RESULTAT EST AU § 24.18** -- avec une correction : j'ajoutais ici que « son coût est sa
> factorisation ( 1.02 s ) », ce qui lisait la mauvaise colonne. 1.02 s est l'analyse **symbolique** ;
> la factorisation numérique tient dans les 0.85 s. Le supernodal ne gagne rien en 2D pour cette
> raison même -- mais il fait **×38** en 3D.

## 24.18 Le Cholesky SUPERNODAL : ×38 en 3D, rien en 2D — et les deux gains ne se recouvrent pas

> **CORRECTION, et c'était ma prémisse.** Au § 24.16 et au § 24.17 j'ai écrit que le coût du Cholesky
> était « sa factorisation ( 1.02 s ) ». **Mauvaise colonne.** Chez `Cholesky`, `hierarchie/analyse`
> compte l'analyse **symbolique** et `resolution` la factorisation **numérique plus** les
> descentes-remontées. Donc 1.02 s était l'analyse symbolique, et la factorisation numérique tenait
> dans les 0.85 s. L'essai valait quand même d'être fait — il transforme la 3D — mais la raison que
> j'en donnais était fausse.

`CholeskySuper` (`--solver chsup`) enveloppe `Eigen::CholmodSupernodalLLT`. La factorisation
supernodale regroupe les colonnes de même structure en blocs **denses** et les traite par des appels
**BLAS 3** ; le parallélisme vient alors du BLAS, pas du code. C'est l'ingénierie que le § 24.17
désignait comme manquante, et elle était déjà sur la machine.

### En 2D : rien à gagner, et les fils BLAS NUISENT

| 2D uniforme, `n = 1e5` | symbolique | numérique + solve | TOTAL |
|---|---|---|---|
| simplicial ( Eigen LDLT ) | 0.55 s | 1.15 / 1.14 / 1.10 | 2.19 / 2.17 / 2.14 |
| **supernodal**, BLAS 1 fil | 0.50 | **0.98** | 1.97 |
| supernodal, BLAS 8 fils | 0.49 | 1.04 | 1.91 |
| supernodal, BLAS **16 fils** | 0.49 | **1.46** | 2.48 |
| AMGCL ( référence ) | 0.25 | 0.56 | **1.20** |

| 2D lignes s0.005 | symbolique | numérique + solve | TOTAL |
|---|---|---|---|
| simplicial | 1.00 s | **0.85** | **3.75** |
| supernodal, BLAS 1 / 8 / 16 | 0.97 | 1.70 / 1.88 / 2.75 | 4.60 / 4.72 / 5.76 |
| AMGCL | 0.86 | 1.77 | 4.29 |

Au mieux −15 % sur l'uniforme, et **jusqu'à ×3 de perte** sur le nuage de lignes. La raison est
structurelle : en 2D la dissection emboîtée produit des **séparateurs minces**, donc des supernœuds
petits, et un appel BLAS 3 sur un bloc de quelques dizaines de lignes coûte plus en préparation qu'il
ne rapporte en débit.

**Et le piège de parallélisme est le contraire de ce qu'on attend.** OpenBLAS prend par défaut **tous**
les cœurs (16 ici), et c'est le **pire** réglage en 2D : 1.46 s contre 0.98 à un seul fil. Tout solveur
qui passe par le BLAS doit donc **épingler ses fils** — sans quoi on paie de la synchronisation sur des
blocs trop petits, en plus de la sursouscription avec les 8 fils du diagramme.

### En 3D : facteur 38, et là les fils servent

| 3D plans s0.02 | symbolique | numérique + solve | TOTAL |
|---|---|---|---|
| simplicial, BLAS 1 / 8 / 16 | 3.39 s | **806 / 805 / 809** | **816 / 815 / 819** |
| supernodal, BLAS 1 fil | 8.87 | 10.55 | 25.9 |
| **supernodal, BLAS 8 fils** | 8.91 | **6.17** | **21.3** |
| supernodal, BLAS 16 fils | 8.84 | 13.77 | 29.1 |
| multigrille maison | 1.19 | 1.09 | **8.46** |

**×131 sur la factorisation numérique** (806 → 6.17 s) et **×38 sur le total** (816 → 21.3 s). En 3D les
séparateurs sont des **surfaces**, donc les fronts sont gros, et le BLAS 3 travaille enfin dans son
régime. C'est aussi le seul endroit où le parallélisme BLAS paie : **8 fils valent 1.7×** sur la
factorisation (10.55 → 6.17) — mais 16 la dégradent encore (13.77), le même plafond qu'en 2D.

### Ce que ça règle, et ce que ça ne règle pas

**Le gain supernodal et l'avantage du solveur direct ne se recouvrent pas.** En 2D, où le direct *est*
le meilleur (§ 24.16), les fronts sont trop petits pour que le BLAS 3 paie. En 3D, où il paie
massivement, le direct reste **2.5×** derrière notre multigrille (21.3 contre 8.46 s) — et c'est déjà
après avoir divisé son coût par 38. Le « Cholesky hors jeu en 3D » du § 24.16 devient donc « Cholesky
jouable en 3D mais toujours battu », ce qui est un déplacement réel mais pas un renversement.

Reste que `chsup` est maintenant la bonne variante partout où l'on veut un **direct** : il domine le
simplicial en 3D d'un facteur 38 et l'égale en 2D sur l'uniforme. Le simplicial garde l'avantage sur le
seul nuage de lignes.

### Licences, puisque le dépôt passe en MIT

* **CHOLMOD CPU : LGPL-2.1+** (`Files: *` dans le fichier de copyright, les exceptions GPL-2+ étant
  `CHOLMOD/GPU`, SPQR, RBio, MATLAB_Tools). Un projet MIT peut le lier **en dynamique** sans
  contamination, c'est ce qu'on fait ici.
* **La voie GPU de CHOLMOD est fermée** : son module `CHOLMOD/GPU` est **GPL-2+**. Si la 3D sur carte
  doit passer par un direct, il faudra regarder ailleurs — cuDSS (NVIDIA, binaire propriétaire mais
  libre d'usage), STRUMPACK (BSD, GPU), Ginkgo (BSD), rocALUTION.
* Et l'observation qui compte pour la suite : **le seul solveur que nous possédons est le multigrille
  maison**, et c'est aussi celui qui gagne en 3D. C'est donc la pièce qu'on peut porter sur GPU sans
  contrainte de licence ni dépendance externe.

## 24.19 Une factorisation gelée comme préconditionneur : les VALEURS dérivent lentement, le MOTIF non

Le cadrage, posé proprement : on ne résout pas un système, on en résout des centaines qui se
ressemblent. Un solveur à **préparation chère et résolutions très rapides** est alors le bon candidat,
à condition d'amortir la préparation. Et comme il ne sert que de **préconditionneur**, le CG voit la
**vraie matrice courante** : la solution reste exacte quel que soit l'âge du préconditionneur, seul le
nombre d'itérations se dégrade. C'est précisément ce qui manquait à la reprise de hiérarchie d'AMGCL
(§ 24.16), dont le `spai0` gelé faussait le préconditionneur sans qu'aucune API ne permette de le
rafraîchir seul.

`CholPrec` (`--solver chprec`) fait ça : une factorisation de Cholesky (supernodale par défaut) gardée
d'une itération sur l'autre, préconditionnant un CG sur la matrice du jour.

### La première moitié de l'hypothèse est VRAIE, et spectaculairement

Nombre d'itérations de CG **pour tout le solve** (pas par résolution) :

| | itérations de Newton | CG au total | temps des résolutions |
|---|---|---|---|
| 2D uniforme, préconditionneur rafraîchi chaque fois | 6 | **6** | **0.095 s** |
| 2D lignes s0.02, idem | 9 | **9** | **0.123 s** |
| 2D lignes s0.005, idem | 12 | **14** | **0.193 s** |
| 3D plans s0.02, idem | 9 | **11** | **0.577 s** |

**Une à deux itérations de CG par résolution.** Une factorisation vieille d'un pas de Newton est donc
un préconditionneur quasi parfait : les valeurs dérivent assez lentement pour ça. À comparer aux
résolutions des solveurs itératifs : 1.81 s pour AMGCL sur 2D s0.005, 1.94 s pour le direct — le
préconditionnement par factorisation gelée divise le temps de résolution par **4 à 10**.

### La seconde moitié est FAUSSE : 20 à 30 % des termes sont neufs

En gelant aussi le **motif** (variante `--cp-seuil 1`), la trace donne la mesure, 2D lignes s0.005 :

| itération | 0 | 1 | 2 | 3 | 4 | … | 8 |
|---|---|---|---|---|---|---|---|
| termes hors motif | 0 % | 0 % | 0 % | **20.6 %** | 23.3 % | ↗ | **30.4 %** |
| itérations de CG | **1** | 20 | 39 | **914** | 1233 | ↗ | 3948 puis **ÉCHEC** |

La lecture est sans appel. Tant que le motif est exact, la factorisation gelée fait converger le CG en
**1 à 39** itérations malgré la dérive des valeurs. Dès que le graphe de Laguerre gagne des arêtes — et
il en gagne **un cinquième à un tiers** — le préconditionneur s'écroule, et refaire la factorisation
*numérique* n'y change rien (à l'itération 8, 151 itérations de CG avec une factorisation fraîche en 3D).
Ce ne sont donc pas les valeurs qui vieillissent, c'est la **structure** qui change.

> **Et la compensation diagonale n'y fait rien.** Retrancher de la diagonale les hors-diagonaux jetés
> pour préserver la somme de ligne — le `modified ILU` des classiques, qui est la cure habituelle pour
> un laplacien — donne **34 itérations de CG contre 34**. Les termes manquants ne nuisent donc pas par
> leur masse mais par leur **place**. Négatif net, et contraire à ce que j'attendais.

### Donc le motif doit être réanalysé, et c'est ça qui mange le gain

`--cp-seuil F` refait l'analyse symbolique dès que la fraction hors motif dépasse `F`. Totaux
(seuls comparables : la colonne « prépa » de `chprec` contient aussi les factorisations numériques,
alors que pour les autres la factorisation numérique est dans « solve ») :

| | 2D uniforme | 2D s0.02 | 2D s0.005 | 3D plans |
|---|---|---|---|---|
| AMGCL | **1.20 s** | **2.13** | **4.29** | 8.31 |
| multigrille maison | 1.24 | 2.27 | 4.63 | **7.25** |
| `chsup` direct | 1.91 | 2.99 | 4.75 | 21.5 |
| **`chprec`, seuil 1 %** | 1.91 | 2.97 | 4.72 | **18.8** |
| `chprec`, seuil 5 % | 1.83 | 6.07 | 8.82 | 18.3 |
| `chprec`, rafraîchi chaque fois | 2.09 | 3.31 | 5.10 | 21.4 |

Au seuil de 1 %, **dix réanalyses sur douze itérations** : le motif change presque à chaque pas, donc
la préparation ne s'amortit pas. Et tolérer plus de dérive ne marche pas non plus — à 5 % le CG repart
(378 itérations) et le total **double**. Il n'y a pas de bon compromis : la dégradation est brutale.

### Ce que ça vaut, et le seul verrou qui reste

Le bilan est donc : `chprec` égale `chsup` partout et reste derrière AMGCL et notre multigrille. Mais
**le plafond de l'idée est réel et attirant**, et il se chiffre. Sur 2D s0.005, si la préparation
n'était faite qu'**une** fois, le travail linéaire tomberait à ≈ 0.25 s (une préparation) + 0.19 s
(douze résolutions) ≈ **0.45 s**, contre **2.65 s** pour AMGCL (0.84 d'analyse + 1.81 de résolutions).
Le total passerait de 4.29 à environ **2.1 s** — divisé par deux.

Tout tient donc à un seul verrou : **stabiliser le motif**. La piste que ces mesures désignent est
précise : un motif **surensemble**, incluant les voisins à **deux sauts**. Les arêtes que le diagramme
de Laguerre gagne relient des germes qui étaient *presque* voisins, donc à deux sauts dans le graphe de
départ ; un tel motif serait stable et permettrait **une** analyse symbolique pour tout le solve. Le
prix est du remplissage en plus, donc une factorisation et des descentes-remontées plus chères — et
c'est exactement l'arbitrage qu'il faudrait mesurer.

## 24.20 Le motif à DEUX SAUTS : il couvre 99 % de la dérive, et ça ne suffit pas

Le § 24.19 avait désigné le verrou — 20 à 30 % des termes du laplacien sont neufs après trois
itérations — et la piste : une arête que le diagramme de Laguerre **gagne** relie deux germes qui
étaient déjà presque voisins, donc à **deux arêtes** dans le graphe de départ. Le motif gelé devient
donc le support de `A + A²` (`--cp-sauts 2`).

### Le diagnostic était juste : la dérive structurelle s'effondre

2D lignes s0.005, fraction de termes hors motif :

| | itération 3 | 4 | 5 | … | 8 | analyses symboliques |
|---|---|---|---|---|---|---|
| motif à **1 saut** | 20.6 % | 23.3 % | 25.9 % | ↗ | **30.4 %** | 10 sur 12 |
| motif à **2 sauts** | **0.93 %** | 0.06 % | 0.15 % | ↗ | **0.28 %** | **2** sur 12 |

**Facteur 100 sur la dérive**, et cinq fois moins d'analyses. L'intuition est donc confirmée : les
arêtes neuves sont bien à deux sauts.

### Et la qualité, quand le motif est vraiment un surensemble, est parfaite

Avec `--cp-seuil 0 --cp-refaire 1` — on réanalyse dès qu'un seul terme manque, et on refait la
factorisation à chaque pas — le préconditionneur est la factorisation **exacte** de la matrice du jour,
et le CG le dit :

| | itérations de Newton | CG au total | temps des résolutions |
|---|---|---|---|
| 2D uniforme | 6 | **6** | **0.168 s** |
| 2D lignes s0.02 | 9 | **9** | 0.228 |
| 2D lignes s0.005 | 12 | **15** | 0.363 |
| 3D plans s0.02 | 9 | **9** | 1.163 |

**Une itération de CG par résolution**, et un temps de résolution 5 à 10 fois sous celui d'AMGCL
(1.75 s sur le cas dur). C'est aussi la preuve que l'implémentation est juste : un surensemble exact
donne exactement un pas de CG.

### Mais la préparation ne s'amortit toujours pas, et le 2 sauts l'aggrave

| totaux | 2D uniforme | 2D s0.02 | 2D s0.005 | 3D plans |
|---|---|---|---|---|
| AMGCL | **1.11 s** | 2.14 | **4.27** | 8.29 |
| multigrille maison | 1.46 | **2.09** | 4.53 | **7.51** |
| `chsup` direct | 1.96 | 2.99 | 4.73 | 21.5 |
| `chprec`, 1 saut, motif exact | 2.16 | 3.40 | 5.19 | 21.9 |
| `chprec`, **2 sauts**, surensemble | 3.62 | 5.18 | 8.71 | **56.7** |

Le motif à deux sauts **double à triple le coût de préparation** (2D s0.005 : 3.10 → 6.51 s ; 3D :
14.8 → 49.2 s), parce qu'il est **3.1 fois plus dense** que le laplacien en 2D et bien davantage en 3D.
Et le nombre d'analyses ne baisse presque pas dans ce réglage — 11 sur 12 au lieu de 12 — parce que
l'exactitude exige de réanalyser dès qu'**un** terme manque, ce qui arrive presque à chaque pas même
avec un motif généreux.

### L'arbitrage est sans issue, et c'est ça le résultat

On a mesuré les deux bouts, et il n'y a rien entre eux :

* **motif exact** → 1 à 2 itérations de CG, mais une analyse symbolique par pas de Newton ;
* **motif généreux gelé** (2 sauts, seuil 1 %) → 2 analyses seulement, mais le CG passe à **590**
  itérations, parce que le résidu de 0.06 à 0.28 % de termes manquants suffit à multiplier le compte
  par 20 à 70. Le § 24.19 avait déjà mesuré cette brutalité ; le 2 sauts la confirme à l'autre échelle.

La raison de fond est structurelle, et elle vaut pour **toute** factorisation directe : une
factorisation est attachée à **son** motif. Elle ne peut donc pas s'amortir sur une suite de matrices
dont le motif change, et le nôtre change à chaque pas — couvrir 99 % de la dérive ne sert à rien
puisque le dernier pourcent coûte un facteur 70.

**Ce qui exploite vraiment le régime reste donc ce que nous possédons déjà** : le multigrille maison,
dont la hiérarchie se garde en rafraîchissant le niveau fin (−11 % d'itérations seulement, § 24.16) et
dont le recyclage de sous-espace trouve de quoi vivre. Un préconditionneur **algébrique** tolère le
changement de motif ; une factorisation, non. C'est la ligne de partage que cette série de mesures
établit.

---

# 25. LE GROSSIER COMME DIRECTION, ET NON COMME DÉPART (`grossier`, `scripts/span_1d.py`)

Le § 8 a fermé le multi-échelle classique : on résout sur `n/R` représentants, on prolonge, et le
niveau fin **refait exactement le travail de Newton depuis Voronoï**. Le § 9 a montré que les densités
méchantes, elles, coûtent 200 à 600 diagrammes par continuation en largeur. La question rouverte ici
est le produit des deux, mais avec un changement de rôle : **la prolongation n'est pas un départ, c'est
une DIRECTION**, et on cherche jusqu'où on peut la suivre.

`src/mains/main_grossier.cpp` porte la mesure (`--reference`, `--balaye`, `--echelle`, `--reste`,
`--span`, `--alpha0`, `--poly`, `--fige`), `scripts/span_1d.py` la boîte à outils 1D exacte (cellules =
intervalles, masses = différences d'`erf`, `L` tridiagonale) qui a servi à tout comprendre avant de
porter.

## 24.21 Le span construit à connectivité gelée : il SATURE, et le coût est dans la construction

### Le protocole, posé explicitement

On est dans la phase `log`, densité fixe. Le diagramme du point de départ `w` donne les aires, le
laplacien `L` et le polynôme **exact** des aires sur un span (§ 22). Ensuite, **plus aucun diagramme** :

1. `d₁` = direction de Newton `log` en `w` ;
2. sur le span courant `{ d₁ … d_k }`, on minimise `log2` **par le modèle** — descente de gradient, une
   grille étant hors de portée des `K = 4` dimensions ;
3. au point optimal, les aires **modélisées** donnent le second membre `log`, résolu avec **le même**
   `L` : c'est `d_{k+1}`. Le span grandit, retour en 2.

Deux approximations à connaître : `L` reste celui du départ (à connectivité fixe la vraie matrice y
serait calculable, les longueurs de facette étant affines en `w`, mais ce n'est pas branché) ; et les
aires de l'optimum viennent du modèle. `--span K` le fait et sort.

### Le modèle est excellent à ces pas

| itération | 0 | 3 | 5 | 7 |
|---|---|---|---|---|
| écart `log2` modèle / réel | **0.01 %** | 0.01 % | −0.13 % | −1.4 % |

C'est bien meilleur que les −50 % du § 24.14, et la raison est instructive : là le point évalué était
poussé à `α*` dans un **mélange** loin du départ ; ici `‖t‖∞ ≈ 0.11` et le polynôme tient. **À ces
pas-là, la connectivité gelée n'est pas une approximation gênante.**

### Mais le span SATURE : les directions sortent colinéaires

> **CETTE SOUS-SECTION ET SA CONCLUSION SONT FAUSSES, voir le § 24.23.** La quasi-colinéarité est
> réelle, mais elle ne réduit pas le span : `{ d₁, d₂ }` et `{ d₁, d₂ − proj( d₂ ) }` sont le **même**
> span. Ce qu'elle fait, c'est le rendre très mal conditionné — et ma descente à pas unique, calée sur
> l'échelle de `t₁`, affamait la seconde coordonnée. Il manquait une **orthogonalisation**. Avec elle,
> la variante A gagne 16 à 29 % au lieu de 0.3 à 6 %. C'est une question de l'utilisateur qui l'a
> trouvé.

`cos( d_{k+1}, d_k )` : **0.98** puis **1.0000** puis **1.0000**, sur les quatre itérations testées. La
nouvelle direction est la précédente. La cause se lit dans les coefficients : l'optimum du span reste
collé à celui de `K = 1` (0.1129 → 0.1088 + 0.0021 → 0.1042 + 0.0032 + 0.0012), donc le résidu y est
presque le même, donc la direction de Newton y est presque la même. **Le span ne s'ouvre pas.**

Et la raison pour laquelle l'optimum ne bouge pas est celle du § 24.10 : tôt dans la phase `log`, le
minimum de `log2` le long de `d₁` est **sur la frontière d'admissibilité** (aire minimale 2.6e-09, une
cellule au bord du vide). D'un point de bord, ajouter une direction ne sert qu'à raccourcir `t₁`.

### D'où un gain qui suit exactement les deux régimes

`log2` atteint, par dimension du span :

| | `K = 1` | `K = 2` | `K = 3` | `K = 4` | aire min à `K=1` |
|---|---|---|---|---|---|
| it 0 ( contrainte ACTIVE ) | **664.7** | 665.7 | 666.8 | 668.0 | 2.6e-09 |
| it 3 ( active ) | **297.7** | 297.8 | 298.0 | 298.1 | 5.2e-10 |
| it 5 | 183.8 | 182.6 | 180.9 | **179.1** | 1.6e-07 |
| it 7 ( contrainte INACTIVE ) | 33.7 | 31.3 | 28.5 | **25.6** | 2.4e-06 |

**Tôt, le span fait PERDRE** (668.0 contre 664.7 : la contrainte mord, et chaque direction ajoutée ne
fait que reculer sur `t₁`). **Tard, il gagne vraiment : −24 % sur `log2` à l'itération 7.** C'est le même
partage que le § 24.10 — contrainte active tôt, inactive tard — et il décide ici aussi.

### Ce que ça dit du solveur linéaire, et c'est la question posée

Le coût d'une direction de plus, à connectivité gelée : **un solve linéaire + une construction de
modèle**, zéro diagramme. Mais la construction de modèle coûte **environ un diagramme** (§ 22, § 24.14 :
0.03 à 0.09 s contre 0.07 pour un diagramme en 2D), et il en faut **une par direction** — parce que
`PolyMulti` a besoin de toutes les directions du span pour ses termes croisés, donc le span qui grandit
force à rebâtir.

**Donc la variante A ne gagne pas de diagrammes en net**, et ce n'est pas le solveur linéaire qui la
bloque : c'est la reconstruction du modèle. Pour une itération de Newton ordinaire on paie 1 solve +
1 diagramme et `log2` est divisé par trois (it 7 : 100.9 → 33.7) ; pour −24 % de plus la variante A
demande 3 solves + 3 constructions. Le compte ne passe pas.

**C'est la variante B qui a l'argument décisif, et le diagnostic le montre par la négative.** Les
dérivées de `w(t)` se lisent dans les coefficients du **même** polynôme : `dA/dt = L d` et
`d²A/dt² = ( dL/dt ) d`, or les coefficients **quadratiques** de `PolyMulti` sont exactement
`d²A/dt²`. Un modèle à **une** direction — une seule construction — donne donc `d₁` **et** la direction
de courbure `d₂ = L⁻¹ ( d²A/dt² )`, pour deux solves et aucun diagramme. C'est d'ailleurs ce que la
sonde du § 24.13 approchait par différence finie, avec `cos( d, e ) ≈ −0.9` : une direction franchement
neuve, là où la variante A rend des colinéaires.

La suite est donc : **variante B, une construction pour tout le span**. Et alors, oui, le solveur
linéaire devient le poste dominant — ce qui ramène au multigrille maison, seul solveur que nous
possédions et seul à tolérer le changement de motif (§ 24.20).

## 24.22 Variante B — les dérivées de `w(t)` : la direction est neuve, et le chemin de Newton est trop COURBÉ pour qu'elle serve

### La dérivation, et son contrôle

À connectivité fixe `A` est **exactement quadratique** en `w` : `A = a + L d + Q(d,d)`. Le long d'un
chemin, `A' = L w'` et `A'' = L w'' + 2 Q(w', w')`, et `2 Q(w',w')` est exactement le coefficient
quadratique du modèle à **une** direction. En imposant au résidu `log` de décroître linéairement,
`r(t) = (1−t) r₀`, avec `u_i = g'(x_i)/ν_i = 1/A_i` pour le `log` :

`u_i A'_i = cste` ⟹ `A''_i = −(u'_i/u_i) A'_i = (A'_i)² / A_i`

et comme `A'_i = (L w')_i = b_i` au départ, il vient

**`L w'' = b²/a − 2q`** — **une seule** construction de modèle, un solve de plus, zéro diagramme.

Le contrôle qui valide les deux : `dA/dt` du modèle contre `L d₁ = b`, écart relatif **4.1e-11**,
1.3e-10, 1.7e-09 aux itérations 0, 5, 7. Le coefficient linéaire du polynôme **est** le second membre
de Newton. (La dérivée seconde n'est validée qu'indirectement — voir plus bas.)

### La direction est franchement neuve, contrairement à la variante A

| | `cos( w', w'' )` | `|w''| / |w'|` |
|---|---|---|
| itération 0 | **−0.447** | **144.4** |
| itération 5 | +0.585 | 135.3 |
| itération 7 | +0.644 | 18.6 |

Là où la variante A rendait des directions à `cos = 0.98` puis **1.0000** (§ 24.21), la courbure donne
une direction à 0.45–0.64. **Le span s'ouvre enfin.**

### Et pourtant elle ne sert à rien, pour une raison chiffrable

`log2` à `K = 2`, en partant **exactement** de l'optimum à `K = 1` (le span le contient, donc `log2` ne
peut que descendre : ce qu'on lit est exactement ce que la direction ajoutée apporte) :

| | `K = 1` | variante **A** à `K = 2` | variante **B** à `K = 2` | coefficient sur `w''` |
|---|---|---|---|---|
| it 0 | 664.6916 | 664.6722 ( −0.003 % ) | 664.6912 ( **−0.00006 %** ) | **−3.1e-08** |
| it 5 | 183.9768 | 183.4217 ( −0.30 % ) | 183.9767 ( **0 %** ) | **+3.4e-07** |
| it 7 | 34.1027 | 31.9856 ( **−6.2 %** ) | 34.1020 ( **0 %** ) | **+1.6e-05** |

**La recherche dans le span refuse `w''`** : le coefficient qu'elle retient est numériquement nul. Et la
raison est dans le rapport des normes. Le développement de Taylor vaut `w₀ + t w' + (t²/2) w''` ; au pas
utile `t = 0.113`, le second terme pèse `(t²/2)·144 = 0.92` contre `t = 0.113` pour le premier — il est
**huit fois plus gros**. Le chemin de Newton `log` tourne donc sur une échelle `t ≈ 1/144 ≈ 0.007`,
**quinze fois plus courte que le pas qu'on prend**. Un développement d'ordre deux n'a aucune validité
là où on en aurait besoin.

> La dérivée seconde n'est validée que par cette absence : une erreur de signe se verrait comme un gros
> coefficient de signe opposé, et les trois itérations donnent un coefficient nul des deux côtés
> ( −3e-08, +3e-07, +1.6e-05 ). Le contrôle exact ne porte que sur l'ordre un.

### Ce que ça explique rétroactivement, et c'est le vrai acquis

La sonde du § 24.13 faisait, elle, une **différence finie sur tout le pas** : `e = d(sonde) − d` avec la
sonde prise en `w + α* d`. Elle moyennait donc la courbure **sur le pas réellement parcouru**, au lieu de
la lire au point de départ — et elle donnait `cos(d,e) ≈ −0.9` et débloquait `α = 1` en 3D (§ 24.13).

**La différence finie en travers du pas bat la dérivée à l'origine**, et on sait maintenant pourquoi avec
un chiffre : la courbure varie d'un facteur ~100 entre `t = 0` et `t = 0.11`, donc la dérivée locale ne
dit rien de ce qui se passe au bout. C'était le bon instrument, et ce n'était pas un hasard.

### Le bilan des deux variantes

Aucune ne paie, et pour des raisons **opposées** :

* **A** produit des directions presque colinéaires (le span reste collé à l'optimum de `K=1`, lui-même
  sur la frontière d'admissibilité), gagne 0 à 6 %, et coûte **une construction de modèle par
  direction** — soit l'équivalent d'un diagramme chacune ;
* **B** produit une direction franchement neuve pour **une seule** construction, mais le chemin est trop
  courbé pour qu'un ordre deux serve : coefficient nul.

Ce qui reste donc de toute cette série est la sonde du § 24.13 — une différence finie en travers du pas,
une construction, deux solves — et la limite mesurée reste celle du § 24.15 : elle gagne des itérations
(deux sur neuf en 3D) et perd en temps de paroi, parce que la construction du modèle vaut un diagramme.

## 24.23 La SOUSTRACTION qui manquait : la variante A gagne 16 à 29 %, pas 0.3 %

### D'où venait l'erreur

Objection de l'utilisateur, et elle porte : au minimum de `log2`, il ne devrait y avoir aucun intérêt à
aller vers les directions déjà dans le span — donc des directions qui ressortent colinéaires sentent le
problème de soustraction.

Deux choses à séparer. D'abord le point théorique, qui ne tient pas tout à fait : au minimum c'est le
**gradient** de `log2` qui est orthogonal au span, et la direction que je calcule n'est pas ce gradient.
`∇log2 ∝ L( g / A )` alors que `d = L⁻¹( A( c − g ) )` — le même résidu `g`, mais pondéré par `A` d'un
côté et par `1/A` de l'autre, et dans une autre métrique. Rien ne force donc `d` hors du span. (Et aux
itérations 0, 3 et 5 l'optimum est **sur la frontière** d'admissibilité, aire minimale 2.6e-09, donc le
gradient n'y est même pas nul.)

Mais l'intuition « problème de soustraction » était la bonne, et le défaut était dans **ma mesure**. La
quasi-colinéarité ne réduit pas le span — `{ d₁, d₂ }` et `{ d₁, d₂ − proj( d₂ ) }` sont identiques —
elle le rend **mal conditionné** : la part utile de la seconde direction ne vaut que quelques pour cent
de sa norme, et une descente de gradient à **pas unique**, calé sur l'échelle de `t₁`, ne la fait
pratiquement pas bouger. Il fallait orthogonaliser (Gram-Schmidt) et **renormaliser à la norme de `d₁`**,
pour que les coordonnées soient comparables.

### Ce que ça change : tout, pour la variante A

`log2` par dimension du span, et `|d⊥|/|d|` = la fraction réellement neuve de la direction ajoutée
(l'écart modèle / réel reste de 0.01 à 1.8 %, donc les gains sont vérifiés par de vrais diagrammes) :

| | `K=1` | `K=2` | `K=3` | `K=4` | gain | `|d⊥|/|d|` des directions ajoutées |
|---|---|---|---|---|---|---|
| it 0 ( contrainte ACTIVE ) | 664.69 | 664.68 | 664.66 | 664.66 | ~0 | 0.205 → 0 |
| it 5 | 183.98 | **153.90** | 150.84 | **150.55** | **−18 %** | **0.589** → 0.080 → 0.006 |
| it 7 | 34.10 | 30.59 | 29.86 | **24.13** | **−29 %** | 0.134 → 0.008 → 0.0003 |

Avant orthogonalisation les mêmes colonnes donnaient −0.003 %, −0.30 % et −6.2 %. **La soustraction
valait un facteur 5 à 60 sur le gain.**

Et la colonne `|d⊥|` dit où est la vraie saturation : la deuxième direction est neuve à 13–59 %, la
troisième à 0.6–8 %, la quatrième à rien. **Le span sature après deux ou trois directions**, mais la
première ajoutée vaut 16 à 18 % à elle seule. Ce qui reste vrai du § 24.21 est le seul cas de
l'itération 0 : là l'optimum est sur la frontière, la contrainte mord, et aucune direction n'y change
quoi que ce soit — le partage du § 24.10 tient.

### La variante B, elle, résiste à la correction

Avec la même orthogonalisation et la même renormalisation — donc des coordonnées comparables — le
coefficient retenu sur `w''` reste **numériquement nul** : −4.2e-06, +1.2e-05, +4.0e-05, +2.3e-04 aux
itérations 0, 3, 5, 7, pour un `log2` inchangé. Et pourtant `|d⊥|/|d|` y vaut **0.77 à 0.99** — bien plus
neuf que les directions de la variante A.

**Donc l'échec de B n'était pas un problème de conditionnement : c'est bien la courbure.** Une direction
peut être presque entièrement neuve et ne porter aucune descente utile. Le chiffre du § 24.22 reste
l'explication : `|w''|/|w'| = 18` à 163, donc le chemin tourne sur une échelle 15 à 150 fois plus courte
que le pas qu'on prend, et la dérivée à l'origine ne dit rien du bout du pas.

### Le compte, refait

Une direction de plus coûte **un solve + une construction de modèle**, et la construction vaut environ
un diagramme (0.03 à 0.09 s contre 0.07 en 2D). Une itération de Newton ordinaire divise `log2` par
trois (it 7 : 100.9 → 34.1) pour un solve + un diagramme. La deuxième direction de la variante A vaut
donc **16 à 18 %**, soit le tiers à la moitié de ce qu'apporte une itération, pour à peu près le même
coût linéaire mais **sans diagramme**.

En 2D c'est donc **à peu près l'équilibre** — et non la perte sèche que le § 24.21 annonçait. Ce qui
ferait basculer, c'est un régime où le diagramme coûte beaucoup plus que le solve : la 3D, où il vaut
0.38 s contre 0.12 pour un solve. Mais `polynomes_multi` porte un `static_assert( dim == 2 )` (§ 24.17),
donc **le span n'est pas disponible là où il paierait**. C'est, à ce stade, le verrou le plus clairement
identifié de toute la série.

## 24.24 La GRILLE : ma descente était piégée, et le span vaut jusqu'à −46 %

**Deuxième correction du même résultat, et c'est encore un doute de l'utilisateur qui l'a trouvée.**
Le § 24.23 avait corrigé le conditionnement (il manquait l'orthogonalisation) ; il restait la question
« est-ce qu'il y a un problème sur la minimisation elle-même ? Peut-on tester sur des grilles ? ».
Réponse : oui, il y en avait un.

### Ce que la grille trouve que la descente ne trouvait pas

Le modèle ne coûte rien à évaluer, donc une **grille complète** sur les coefficients du span est à
portée (`--span-grille N`, 40 × 40 ici, soit 1681 points × 100 000 cellules, parallélisé). Elle est
sans appel sur deux points :

* **itération 5, `K = 2`** : la descente donnait 153.90, la grille trouve **141.97** — et l'optimum est
  en `( 0.086, 0.286 )`, donc **dominé par la seconde direction**. Ma descente, partant de
  `( 0.191, 0 )` et ne sachant que descendre, n'a jamais atteint ce bassin ;
* **itération 7, `K = 2`** : 30.59 pour la descente, **21.16** à `( 1.05, 0.067 )` — et `t₁ = 1.05`,
  c'est-à-dire **le pas plein de Newton**, inatteignable depuis un départ à 0.699.

Le défaut est donc structurel et pas un réglage : une descente monotone depuis l'optimum **contraint**
de `K = 1` ne peut pas trouver un optimum qui demande d'**allonger** `t₁`.

### Les chiffres corrigés

`log2` par dimension du span (grille à `K ≤ 2`, puis descente ; écart modèle / réel de 0.01 à 2 %, donc
tout est vérifié par de vrais diagrammes) :

| | `K=1` | `K=2` | `K=3` | `K=4` | gain |
|---|---|---|---|---|---|
| it 0 ( contrainte ACTIVE ) | 664.69 | 664.68 | 664.66 | 664.66 | ~0 |
| it 3 ( active ) | 297.65 | 297.61 | 297.56 | 295.34 | −0.8 % |
| it 5 | 183.98 | **141.95** | 141.32 | **124.02** | **−33 %** |
| it 7 | 34.10 | **21.16** | 19.00 | **18.55** | **−46 %** |

Les trois états successifs de ma mesure sur l'itération 7, pour mémoire : **−6.2 %** (base dégénérée),
**−29 %** (orthogonalisée, descente seule), **−46 %** (orthogonalisée + grille). Les deux corrections
venaient d'un doute de l'utilisateur, et aucune des deux n'était un détail.

### Ce que ça vaut, enfin

Une itération de Newton divise `log2` par trois (it 7 : 100.9 → 34.1, soit −66 %). Le span à `K = 4` en
rend **−46 %**, c'est-à-dire **70 % d'une itération** — pour trois solves et trois constructions de
modèle, et **aucun diagramme**. Le partage du § 24.10 reste la frontière : là où la contrainte est
active (it 0 et 3), le span ne donne rien, parce que l'optimum est collé au bord.

### Et la variante B tient, maintenant pour de bon

La même grille, appliquée au span `{ w', w'' }`, retrouve **exactement** la valeur de `K = 1` aux
itérations 0, 5 et 7. Une recherche exhaustive ne trouve donc rien : l'échec de la direction de
courbure n'était ni un conditionnement ni un piège de descente. C'est bien ce que le § 24.22 dit —
`|w''|/|w'|` de 18 à 163, un chemin qui tourne 15 à 150 fois plus vite que le pas qu'on prend.

**Donc : la direction utile s'obtient par différence finie EN TRAVERS du pas (variante A, § 24.13), pas
par dérivation à l'origine (variante B).** Et elle vaut beaucoup plus que ce que j'avais mesuré.

### Ce que ça change pour la suite

Le verrou identifié au § 24.23 devient décisif : `polynomes_multi` est **2D seulement**, et c'est en 3D
que l'échange serait favorable — un diagramme y coûte 0.38 s contre 0.12 pour un solve, alors qu'en 2D
c'est 0.07 contre 0.15. Troquer des diagrammes contre des solves et des constructions n'a donc d'intérêt
qu'en 3D. **Le polynôme multi-directions 3D est la suite, et il est maintenant clairement justifié.**

Un fait mathématique à poser avant de l'écrire : en 3D le volume d'une cellule à connectivité fixe est
**cubique** en `t`, pas quadratique. Les sommets restent affines (chacun est l'intersection de ses trois
coupes, et `Cellule3D` porte justement cette information), mais le volume est une forme **cubique** des
sommets — `V = (1/6) Σ_f ε_f ( v_o − g ) · S_f` avec `S_f` quadratique. `PolyMulti` devra donc porter des
termes cubiques, et le coût de l'expansion limite le span utile à `K = 2` ou 3 — ce qui tombe bien,
puisque c'est là que le gain est (−38 % à `K = 2` sur l'itération 7).

## 25.1 Le prix du niveau grossier, et la zone dure ne bouge pas

`densite --pas essai-limites --conv 0.5 --conv-ratio 1.414 --sigma 0.02`, `job -b`, 8 fils :

| `n` | diagrammes | itérations | temps | **en équivalents fins** |
|---|---|---|---|---|
| 100 000 | 200 | 138 | 58.47 s | 200 |
| 25 000 (`R = 4`) | 134 | 103 | 9.16 s | **31** |
| 6 250 (`R = 16`) | **105** | 85 | **2.13 s** | **7.3** |

Le compte de diagrammes tombe **aussi** (200 → 105), pas seulement le prix de chacun : à `R = 16` la
phase grossière coûte **27 fois moins**, soit 3.6 % de la référence. Elle est donc négligeable, et tout
le problème est dans ce qu'on fait ensuite.

Et **la zone dure ne bouge pas avec l'espacement** : le pic est à `s = 0.031–0.044` pour les trois `n`
(31, 16, 11 diagrammes à l'étape la plus chère). Le § 9.3 l'avait établi pour `σ` ; c'est vrai de `h`
aussi. Le nuage grossier traverse la **même** zone dure — mêmes aiguilles, plus grasses.

## 25.2 Quelle agrégation : la distorsion est le critère, et elle n'est pas un choix de goût

Le critère n'est pas à inventer. Le § 23.12 donne la décomposition **exacte**

```
Σ_{i∈r} ∫_{C_i} |x − p_i|²  =  ∫_{C_r} |x − q|²  +  Σ_{i∈r} ν_i |p_i − q|²  −  2 Σ (p_i − q)·m_i
```

le terme croisé s'annulant au premier ordre parce que `q` est le barycentre pondéré par `ν`. **L'écart
entre l'objectif grossier et l'objectif fin est donc exactement la distorsion de quantification**
`Σ ν_i |p_i − q_r|²`. Et le § 8.2 identifie l'erreur de prolongation comme une erreur de courbure à
l'échelle du paquet, `H_c²·w''` — la même quantité pondérée par `ν`.

`R = 16`, `n = 10⁵`, `σ = 0.02`, prolongation `mls` :

| `--paq` | distorsion (en `h²` local) | cellules sous le plancher | **PIRE résidu** |
|---|---|---|---|
| `bsp` (AaBsp, représentant = germe au centre) | 2.233 | 40.6 % | 1660 |
| `bsp-bary` (même partition, barycentre pondéré) | 2.019 | 42.0 % | 1527 |
| **`lloyd`** (quantification optimale pondérée) | **1.293** | **23.1 %** | **897** |

La distorsion classe les trois bras **de façon monotone**, et le classement se transporte sur la
qualité de la prolongation. Le barycentre seul ne vaut presque rien (2.23 → 2.02) : c'est la **forme**
des paquets qui paie. Coût de Lloyd : 1.0 s à `n = 10⁵`, négligeable.

## 25.3 Le mur : la prolongation cesse d'être utile là où les aiguilles naissent

`--balaye` juge la prolongation à chaque `s` de la continuation (Lloyd + `mls`, `σ = 0.02`) :

| `s` | sous le plancher | PIRE résidu, prolongé | PIRE résidu, Voronoï |
|---|---|---|---|
| 0.50 | 0.17 % | 9.7 | 4.5 |
| **0.125** | 0.44 % | **12.3** | **12.3** |
| **0.088** | 2.3 % | **17.7** | **20.6** |
| 0.0626 | 6.2 % | 102 | 32 |
| 0 | 23.1 % | 897 | 341 |

La prolongation ne bat Voronoï que dans **une seule fenêtre, `s ≈ 0.09–0.12`** — et la zone dure
commence à `s = 0.0626`. Or elle pèse 120 des 200 diagrammes. **Le mur est exactement au seuil de la
zone dure**, donc la structure « continuation grossière jusqu'au bout, puis une prolongation » ne peut
économiser que ce qui ne coûte rien.

### Deux comptes qui ne mesurent pas ce qu'on croit

**Les « cellules sous le plancher » ne mesurent pas le départ mais les ZÉROS de la densité.** À `s = 0`
avec `--plancher 0`, Voronoï lui-même en a **90 692 sur 100 000** : la densité est nulle sur presque
tout le carré, donc presque toutes les cellules ont une masse nulle. Le compte est inutilisable.

**Et le résidu est un GAVAGE, pas une famine.** `min a/ν` vaut 0 à tous les `α₀`, y compris 0, et une
cellule vide est bornée par 1 dans `max|a−ν|/ν` : tout ce qui dépasse 1 vient de `max a/ν`. Ce qui
bloque est donc un **plafond**, et tout l'outillage du pas (§ 7, § 9.7) est construit sur le
**plancher**. À `s = 0.0442` la population au-delà de `10 ν` fait 2702 cellules à Voronoï, 89 à
`α₀ = 0.8`, 237 à `α₀ = 1` : quelques centaines, pas des dizaines de milliers.

## 25.4 En 1D, la loi qui gouverne tout : `α* × saut = 2h²`

`scripts/span_1d.py` donne la 1D exacte. La limite d'admissibilité le long d'une prolongation s'y lit
en forme close, et elle obéit à une loi vérifiée à quatre chiffres (`n = 400`, `2h² = 1.2500e-05`) :

| `R` | saut max aux interfaces | `α*` | `α* × saut` |
|---|---|---|---|
| 4 | 3.607e-3 | 3.466e-3 | 1.30e-5 |
| 8 | 7.213e-3 | 1.733e-3 | **1.25e-5** |
| 16 | 1.403e-2 | 8.910e-4 | 1.25e-5 |

**Ce qui borne une prolongation n'est pas son amplitude, c'est la différence seconde qu'elle porte à
l'échelle de la cellule.** La projection par **copie** met tout le défaut aux interfaces d'agrégats :
un saut de `w` de 7.2e-3 entre deux germes distants de 2.5e-3 déplace le bissecteur de **576 largeurs
de cellule**, d'où `α* = 1.7e-3`. Elle est donc inadmissible au premier ordre.

L'**interpolation affine** remplace le saut par un **pli** sur chaque représentant (en 1D c'est
exactement la prolongation harmonique du § 8.2) : la différence seconde passe de `saut` à `h·H·w''`,
et `α*` gagne **un facteur 175** (1.73e-3 → 0.303). Ce n'est toujours pas 1.

### Le seuil est `α* > 1`, et c'est une transition de phase

| prolongation | `α*` | `α₀` atteint | **it restantes** (réf. 60) |
|---|---|---|---|
| copie | 1.73e-3 | 2.7e-5 | 60 |
| affine | 0.303 | 0.168 | 53 |
| affine + 4 lissages | 0.615 | 0.580 | 42 |
| **affine + 16 lissages** | **1.067** | **0.963** | **10** |
| copie + 64 lissages | 1.210 | 0.998 | 11 |

`α* = 0.30` → 53 itérations, `0.62` → 42, `1.07` → **10**. Ce n'est pas un continuum. D'où une règle
opérationnelle sans réglage à l'aveugle : **lisser jusqu'à ce que `α*` dépasse 1**, `α*` se calculant
pour rien. Le lissage optimal est à `m ≈ R²` en 1D (longueur de diffusion `√m` = une largeur
d'agrégat) et `R²/4` pour l'affine, qui a déjà supprimé le saut.

## 25.5 `k > 1` est obligatoire, et les compagnes doivent LISSER

Minimiser le mérite sur le span n'est qu'un **proxy** : son minimum peut être à `α₀` petit alors qu'un
grand `α₀` est atteignable avec les bonnes compagnes. Ce qu'on veut est `α₀ → 1` sous contrainte
d'admissibilité, les autres coefficients libres, par **continuation** en `α₀` avec enrichissement du
span quand ça bloque. Sans aucun pré-lissage, `n = 400`, `R = 8`, référence 60 itérations :

| `k` | `α₀` (copie) | it | `α₀` (affine) | it |
|---|---|---|---|---|
| 1 | 0.0017 | 60 | 0.3028 | 58 |
| 2 | 0.0167 | 60 | 0.7715 | 43 |
| 3 | 0.4116 | 45 | **1.0000** | **9** |
| **4** | **1.0000** | **11** | — | — |

**À `k = 1` il ne se passe rien** ; c'est l'enrichissement qui porte `α₀` d'un facteur 600.

### Les compagnes sont des incréments de lissage, et c'est structurel

Prendre « ce qui manque » comme **direction de Newton au point bloqué** échoue, et pas par maladresse :
là où une cellule est à `1e-8 ν`, les lignes du jacobien `J_ik = (L d_k)_i / a_i` valent `1e8`, le
moindre carré est entièrement dominé par elles, et la tangente de la variété des minimiseurs sort à
**1.1e+07**. *La barrière qui protège détruit le conditionnement de toute algèbre linéaire à son
voisinage.* On prend donc les compagnes **littéralement** comme des directions qui lissent,
`d_j = lisse(w_prol, 4^j) − w_prol` — une échelle dyadique d'incréments, bien bornée et bien
conditionnée.

Et le contrôle qui rassure : à `k = 4`, les coefficients sont `(−0.002, +0.014, +0.988)` sur les
échelles 4, 16, 64 — **tout sur 64, qui est `R²`.** L'optimiseur retrouve seul le lissage trouvé à la
main, au même `log2` (54.4 contre 54.65). Le réglage deviné devient inutile.

### `α₀ = 1` est une cible, pas un maximand

Laissé monter jusqu'à 1.21 (le bord de l'admissible), le point a un `log2` de **2634** — pire que
Voronoï (1888) — et il reste 41 itérations. Plafonné à 1 avec `log2` minimisé sur les compagnes :
54.4 et **11** itérations. Maximiser `α₀` pousse exactement au mauvais endroit.

### Gauss-Newton, et `α*` sort de la recherche

`log2` étant une somme de carrés, le pas résout `min |g + J dα|²` avec `J_ik = (L d_k)_i / a_i` — exact
à connectivité fixe, **vérifié à 1.71e-09** contre des différences finies centrées. La barrière du
logarithme étant *dans* l'objectif, la recherche de pas ne peut pas franchir le bord : plus besoin de
le clamper. La version à grille reste disponible et portait un défaut instructif — balayer `[−0.5, 1.5]`
par pas de 0.05 quand `α*` vaut 1.7e-3 **ne visite jamais le domaine admissible**, et rendait
`α₀ = 0` par artefact.

## 25.6 Le portage 2D : la densité gelée, et ce qu'elle lève

`PolyMulti` donne **l'aire** ; avec une densité l'objectif porte sur la **masse**, qui n'est pas un
polynôme (§ 9.7). Mesure du désastre : `log2` sur les aires vaut **20** à la base quand le vrai mérite
vaut **6.9e4** — le modèle égaliserait les aires.

Le remède est de geler une densité **constante par morceaux** : avec `ρ_i` fixe, `masse_i(t) = ρ_i A_i(t)`
redevient un polynôme, et comparer `ρ_i A_i` à `ν_i` revient à comparer `A_i(t)` à une **cible d'aire**
`â_i = ν_i / ρ_i`. On retombe sur du Lebesgue pondéré, cible par cible. Quatre variantes,
`n = 25600`, germes réguliers, écart max masse prédite / vraie **à la base** :

| `--rho-gel` | écart | `α₀` poly | `α₀` retenu | it restantes |
|---|---|---|---|---|
| `plateau` (`ν_r/\|C_r\|` par cellule grossière) | 5.40e+02 | 0.96 | identique | identique |
| `interp` (le même, interpolé par MLS) | 5.22e+02 | 0.96 | identique | identique |
| `germe` (`ρ(p_i)`) | 2.55e-02 | 0.96 | identique | identique |
| `cellule` (`a_i/A_i` à la base) | **1.10e-16** | 0.96 | identique | identique |

**Le gel était nécessaire et n'est pas le verrou** : sur dix-huit ordres de grandeur de précision, le
résultat ne bouge pas. La continuation bloque sur une aire polynomiale négative, purement géométrique,
qui ignore `ρ`.

Et l'interpolation ne répare pas le plateau (540 → 522), pour une raison structurelle : **à la solution
grossière les cellules grossières ont toutes la même masse**, donc celles de la queue sont énormes et
couvrent une plage de `ρ` d'un facteur plusieurs centaines. Le niveau grossier **sous-résout la densité
précisément là où ses cellules sont grandes**, et interpoler entre des nœuds aussi espacés n'y change
rien. La densité gelée doit s'évaluer **à l'échelle fine**.

## 25.7 Jusqu'où la connectivité fixe emmène (`--fige`)

Trois limitations se confondaient. `mesures_connectivite_figee` (`Ecrasement.h`) refait la cellule
**exactement** mais en ne coupant QUE par les voisins connus (`FournisseurAlpha` avec
`parcours = false` : aucune exploration, pas d'AaBsp), ce qui les sépare :

| `α` | min A poly | min A **figée** | min a/ν vrai | vides vrais | écart masse figée/vraie |
|---|---|---|---|---|---|
| 0.50 | 1.086e-05 | 1.086e-05 | 1.797e-02 | 0 | 9.0e-02 |
| 0.95 | 7.220e-07 | 7.259e-07 | 5.922e-03 | 0 | 5.8e-01 |
| **1.00** | **−6.292e-06** | **+3.028e-07** | 0 | **1** | — |

1. **Le polynôme EST l'aire à connectivité figée jusqu'à `α = 0.95`** — trois à quatre chiffres
   identiques. Le **repliement** (les « parties positives » qui manquent au polynôme) ne mord qu'à
   partir de `α = 1`.
2. **La connectivité fixe tient jusqu'à `α ≈ 1`** : aucune cellule vide avant 1.05.
3. **Et le diagramme réel est sain à `α = 0.95`.**

## 25.8 La chaîne qui marche, et ce qu'elle coûte

`σ = 0.05`, `plancher = 0.05`, pas de continuation en `s`, `n = 25600`, `R = 16`, germes réguliers.
Grossier 56 it / 109 diag ; **témoin depuis Voronoï 98 it / 193 diag**. Et
`cos(w_prol, d_newton) = +0.63` : la direction grossière porte enfin de l'information neuve (contre
+0.98 dans les régimes essayés d'abord).

**`--poly 1 --rho-gel germe --span-garde 0`** : un diagramme construit le modèle, toute la recherche
est gratuite, et une vérification confirme le point.

```
beta = 1.0000 ( aucun recul ),  alpha_0 = 0.9600,  min a/nu = 4.0e-03
13 iterations / 20 diagrammes      contre      98 / 193
```

Bout à bout, le grossier valant 6.8 équivalents fins plus deux diagrammes : **≈ 29 contre 193, soit
×6.7.** Et c'est mieux que le pré-lissage fait à la main (16 it / 26 diag).

### `α₀( k )` en 2D, sur la copie brute

| `k` | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `α₀` | 0.0020 | 0.4760 | **0.8360** | 0.9860 |
| **it restantes** | 97 | 80 | **31** | 50 |
| `β` | 1.0 | 1.0 | 1.0 | 1.0 |

Les compagnes portent `α₀` d'un facteur 500 et le diagramme réel accepte le point du modèle à chaque
tour. Noter que le meilleur point réel est à `k = 3`, pas à `k = 4` où `α₀` est plus grand *et* le
`log2` du modèle plus bas : **pousser `α₀` à 0.986 dépasse**, et l'objectif du modèle n'est pas
parfaitement aligné sur la qualité réelle du départ.

## 25.9 Ce qui résiste : le nuage de Poisson

Tout ce qui précède est sur des germes **réguliers**. Sur un tirage, rien ne marche — et la loi est
simple : **`α* ≈ (plus petit écart) / h`.** Balayage du bruit d'une grille (`n = 25600`) :

| bruit | 0.05 h | 0.15 h | 0.25 h | 0.35 h | 0.50 h |
|---|---|---|---|---|---|
| `α*` | 0.990 | 0.803 | 0.637 | 0.511 | **0.079** |

L'effondrement à 0.50 est le moment où deux voisins peuvent se rejoindre. Et ça recoupe Poisson : son
plus petit écart vaut `O(h²)`, donc `α* ≈ h = 6.25e-3` — mesuré **8.6e-3**.

**L'agglomération du § 23 aide et ne suffit pas** (`--agglo C`, en préalable de tout) :

| `δ` | grappes / 25600 | taille max | `α*` |
|---|---|---|---|
| 0.25 h | 23 198 | 4 | 1.47e-1 |
| **0.50 h** | 16 997 | 11 | **2.70e-1** |
| 0.75 h | 9 484 | 39 | 4.36e-1 |
| 1.00 h | 3 413 | **197** | 3.46e-1 |

×31 à `δ = 0.5 h`, puis **ça retombe** : la fermeture transitive de « à moins de `δ` » **percole** dès
que `δ` approche l'espacement, et les barycentres de grappes de 1 à 197 membres fabriquent leur propre
irrégularité. C'est la limite structurelle du lien simple, et elle explique pourquoi le § 23.8
l'employait à `δ = 3e-5` sur un nuage d'espacement 3e-3. **La voie demande donc un nuage régulier par
construction (bruit ≤ 0.35 h), pas un nuage réparé.**

## 25.10 Les pièges de mesure, et il y en a sept

Cette section a coûté plus en faux diagnostics qu'en calcul. Les voici, pour ne pas les repayer.

1. **`res_cur` n'est posé que dans `resout`.** Le mérite d'un span reste `lin` si on ne le pose pas
   soi-même — et `resout` l'écrase ensuite, la bascule `log → lin` du § 24.5 le laissant à `lin` dès
   que le solve converge. Deux tableaux sur des échelles différentes sans que rien ne le signale.
2. **`nw.merite` rend la NORME, le jacobien du span dérive la SOMME DES CARRÉS.** Les mélanger fait un
   facteur `2·mérite` : le contrôle par différences finies annonçait 5.5e+02 d'écart, ce qui ressemble
   exactement à un jacobien faux. Objectif mis au carré, le contrôle passe à 4.6e-08.
3. **`polynomes_multi` prend les CELLULES des poids que `pd` PORTE**, et les coefficients de `w` qu'on
   lui passe. Entre deux tours, `nw.resout` laisse `pd` sur la solution convergée : le modèle du tour
   `k` était construit sur les cellules du tour précédent. Le même `nk = 1` au même `α` donnait
   `+3.24e-02` puis `−1.21e-01`, ce que j'avais pris pour une **dépendance en `nk` du noyau** — qui
   n'existe pas (avec `t = (α, 0, …)` tous les termes supplémentaires portent un facteur nul, et après
   `pd.set_weights(wb)` les valeurs sont bit à bit identiques de `nk = 1` à `4`).
4. **Une garde d'admissibilité relative mentait.** Exiger `min a/ν ≥ 0.5 ×` celui de la base refusait
   des points parfaitement sains : `α₀ = 0.96` y tombait à 0.35 par recul, 84 itérations au lieu de 13,
   alors que le diagramme réel n'y a aucune cellule vide. **Seul `min a/ν > 0` est un critère.**
5. **L'écart de pavage `Σ A_i(t) − 1` ne teste RIEN** : c'est une identité algébrique de l'aire signée,
   qui fait s'annuler exactement recouvrements et replis dans la somme. Il vaut 9e-15 là où le modèle
   se trompe d'un facteur 5. Et les −6 % à −88 % que j'avais rapportés mesuraient l'aire des cellules
   **exclues** (`etat != OK`), pas une erreur de modèle.
6. **`PolyMulti::rayon` en MINIMUM est inutilisable** : 5e-7 à 2e-5 là où le pas utile est 0.9, parce
   qu'une seule cellule le fixe. Il faut un quantile. (Et l'objection « une arête qui meurt ne change
   rien à l'aire » est juste sur la *valeur* — c'est au-delà que le polynôme continue avec une longueur
   négative, ce qui est le repliement.)
7. **Les germes tirés au hasard faussent `α*`** pour une raison étrangère au sujet (point 25.9), et une
   grille de recherche mal échelonnée ne visite jamais le domaine admissible (§ 25.5).

De tous les garde-fous essayés — pavage, `rayon`, écart de masse au pire cas, garde relative — **le
seul qui ait tenu est `min a/ν > 0`**. L'écart modèle / vraie masse au point retenu vaut 16.9 et le
point est excellent : l'erreur au pire cas n'est pas prédictive.

## 25.11 Ce qui reste

* le nuage de Poisson, qui résiste (§ 25.9) — et le remède que toutes les mesures désignent est de ne
  pas raccourcir le pas **globalement** sur la pire cellule mais de **relever les cellules qui
  bloquent**, c'est-à-dire l'enveloppe convexe inférieure du § 8.7.3, que le § 8.7.5 avait déjà nommée ;
* l'objectif du modèle, qui n'est pas aligné sur la qualité réelle du départ au-delà de `α₀ ≈ 0.85` ;
* et `main_grossier.cpp`, qui porte maintenant deux questions distinctes et devrait se couper en deux
  (la qualité des agrégations d'un côté, la recherche dans le span de l'autre), contre la convention du
  dépôt — un binaire par question.
