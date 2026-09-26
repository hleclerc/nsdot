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
lieu de 3.05e-6. `check --load FILE [--cellule I] [--weights -1]` le vérifie ; **le même code
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
la sortie en STAGNATION. `L₀` = laplacien figé, refait selon les règles ci-dessus ; `refacto 1` =
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

## 12.8 Ce qui reste

* La référence prise **par arête** plutôt que par cellule (§ 12.6) — le seul vrai obstacle à la
  chaîne complète en `float`, et il est bon marché.
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
