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

n = 10⁵, 8 fils, minimum de 10 répétitions. Le *témoin* est le chemin MEMO à vide (le prix du
code) ; *les voisins seuls* ne parcourent pas l'arbre du tout — c'est le plancher, le prix des
seules coupes utiles.

| par cellule | plans proposés | boîtes testées | **coupes effectives** | temps |
|---|---|---|---|---|
| uniforme, sans mémoire | 87.4 | 95.2 | 30.1 | 0.305 s |
| uniforme, témoin | 87.4 | 95.2 | 30.1 | 0.313 s |
| uniforme, **avec mémoire** | 67.9 | 83.9 | **15.1** | **0.211 s (−31 %)** |
| uniforme, les voisins seuls | 15.1 | 0 | 15.1 | 0.118 s (−61 %) |
| plans / Voronoï, sans | 88.5 | 95.6 | 30.5 | 0.306 s |
| plans / Voronoï, **avec** | 67.8 | 83.9 | **15.1** | **0.179 s (−42 %)** |
| plans / volumes égaux (Laguerre), sans | 270 | 258 | 32.0 | 0.499 s |
| plans / volumes égaux, **avec** | 253 | 251 | **15.5** | **0.395 s (−21 %)** |
| uniforme 10⁶, sans | 106 | 101 | 32.3 | 3.16 s |
| uniforme 10⁶, **avec** | 81.6 | 89.4 | **15.3** | **2.04 s (−36 %)** |

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
égaux (Laguerre, 0.5 s sans mémoire) :

| souvenirs pris à | voisins retrouvés | coupes effectives | temps |
|---|---|---|---|
| `W` (exacts) | 15.50 | 15.5 | −21 % |
| `0.9 W` | 15.48 | 15.9 | −18 % |
| `0.5 W` | 15.34 | 17.3 | −18 % |
| `0` (Voronoï) | 15.09 | 20.0 | −18 % |

Même les souvenirs de Voronoï rendent 18 % sur le diagramme final : les voisinages changent
peu, et un souvenir faux ne coûte qu'une première passe. C'est le point que la 2D ne pouvait pas
montrer.

## 11.3 Dans la boucle de Newton

`newton --3d --memo` : les facettes du dernier diagramme **accepté** (que Newton a de toute façon,
pour le laplacien) deviennent la mémoire du suivant, essais compris — `memorise` coûte 10 ms par
diagramme. Mêmes itérations, mêmes diagrammes, mêmes résidus au chiffre près (l'ordre des coupes
n'a pas changé les arrondis, contrairement aux lignes en 2D) :

| n = 10⁵, AMG RS+GS | diagrammes | total |
|---|---|---|
| uniforme, 9 diagrammes | 2.92 → **2.07 s (−29 %)** | 7.08 → 6.30 s (−11 %) |
| plans, 27 diagrammes | 11.8 → **8.8 s (−25 %)** | 21.8 → 19.1 s (−12 %) |

Le quart du diagramme 3D, pour soixante octets par germe et une passe sur les facettes. C'est
le contraire de la conclusion 2D, pour une raison qu'on peut nommer : ce que la mémoire épargne
n'est pas du parcours, ce sont les coupes transitoires, et elles n'ont de prix qu'en 3D. Reste le
plancher : les voisins seuls font −61 %, la moitié du diagramme est encore le parcours de l'arbre
pour *confirmer* qu'il n'y a personne d'autre — et ça, la mémoire ne peut pas le savoir.
