# `2d_des_familles` — le banc, en C++ nu

Ni jax, ni AdaptiveCpp, ni FFI. Un `xmake`, `-O3 -march=native`, des `std::thread`. L'intention est
d'essayer une idée en trente secondes au lieu de trois minutes de compilation SYCL, et de pouvoir
lire l'assembleur qui en sort. **Ce n'est pas la bibliothèque** : ce qu'on y mesure doit ensuite
être reporté dans `sdot`.

Le nom du répertoire est resté, mais le banc **n'est plus 2D** : le cœur est templaté sur la
dimension, et chaque banc déroule les cas importants **en 2D puis en 3D**.

Ce fichier est la **synthèse**. Les mesures datées, dans l'ordre où elles ont été faites et avec le
raisonnement qui les a amenées, sont dans **[JOURNAL.md](JOURNAL.md)** — chaque section ci-dessous y
renvoie.

---

# 1. LES CHIFFRES

Machine : Xeon W-2145, 8 cœurs physiques, FP64, `-O3 -march=native`. Domaine = le carré (ou le
cube) unité. Toutes les mesures ci-dessous sont **appariées** — les deux binaires alternés dans la
même boucle, plusieurs tours, machine au repos. Sur ce banc c'est la seule façon valide de comparer
(§ 1.5).

## 1.1 Le cas de référence : 10⁶ germes uniformes, 2D

| par cellule, 8 cœurs | ce banc | `sdot` (SYCL) | pysdot |
|---|---|---|---|
| temps total | **0.213 s** | 1.185 s | 0.446 s |
| références L3 | **1.7** | 199 | 5.9 |
| instructions | **12 360** | 17 500 | 26 950 |
| IPC | 1.60 | 0.456 | 1.96 |

Le même algorithme (BSP médian, descente fils-le-plus-proche, coupe en place, mêmes formules) monte
à **7.6×** sur 8 cœurs ici et à 2.1× sous SYCL. C'est la question de fond que ce banc a tranchée, et
elle se lit dans la colonne « références L3 ».

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| arbre | 1.620 s | 0.838 s | 0.423 s | **0.213 s** (7.61×) |
| grille | 1.161 s | 0.606 s | 0.306 s | **0.154 s** (7.54×) |

Et le tableau à un cœur, tout le monde sur la même machine : **grille 1.161 s**, arbre 1.637 s,
CGAL (`Regular_triangulation_2`) 1.522 s, pysdot (`SpZGrid`) 3.364 s, `sdot` (SYCL) 2.489 s.

## 1.2 La suite 2D, tous les accélérateurs

`n = 10⁶` pour l'uniforme, `10⁵` pour les nuages durs. **1 fil / 8 fils**, en secondes.

| | uniforme 10⁶ | lignes / Voronoï | lignes / aires égales |
|---|---|---|---|
| `front` | **1.091** / **0.156** | **0.104** / **0.014** | **0.111** / **0.015** |
| `hull` | 1.332 / 0.176 | 0.151 / 0.021 | 0.336 / 0.054 |
| `pack` | 1.675 / 0.219 | 0.165 / 0.023 | 0.379 / 0.060 |
| `bsp4l` | 1.820 / 0.238 | 0.177 / 0.025 | 0.588 / 0.098 |
| `bsp` | 1.823 / 0.235 | 0.173 / 0.023 | 0.564 / 0.093 |
| `packed` | 1.884 / 0.243 | 0.180 / 0.023 | 0.526 / 0.086 |
| `obsp` | 1.910 / 0.252 | 0.274 / 0.057 | 0.720 / 0.124 |
| `bsp4` | 2.535 / 0.328 | 0.252 / 0.033 | 0.580 / 0.093 |
| `grid` | 2.693 / 0.359 | 1.317 / 0.192 | 22.338 / 3.283 |
| `pre` | 3.092 / 0.402 | 0.310 / 0.042 | 0.898 / 0.146 |

Trois choses à lire dans ce tableau. Le nuage dur coûte **3.1×** l'uniforme au BSP et **8×** à la
grille : c'est le rapport qui compte, pas la ligne isolée. `front` gagne partout, mais son index est
fait d'un diagramme grossier qu'il faut construire — voir § 4 pour ce que ça implique dans Newton.
Et la grille, imbattable sur l'uniforme (§ 1.1), s'effondre d'un facteur 40 sur le cas dur.

## 1.3 La 3D

`n = 2·10⁵` pour l'uniforme, `10⁵` pour les nuages de plans, `--maxnv 128`.

| `bsp` | 1 fil | 8 fils | montée |
|---|---|---|---|
| uniforme 2·10⁵ | **5.313 s** (26 570 ns/germe) | **0.685 s** (3 425) | ×7.8 |
| plans / Voronoï 10⁵ | **2.658 s** (26 580) | **0.352 s** (3 519) | ×7.6 |
| plans / volumes égaux 10⁵ | **5.050 s** (50 495) | **0.703 s** (7 029) | ×7.2 |

La montée en charge tient exactement comme en 2D. Deux rapports méritent d'être retenus : une
cellule 3D coûte **15×** une cellule 2D, et le nuage dur n'y coûte que **1.9×** l'uniforme là où en
2D il coûte 3.1×.

La grille en 3D, mesurée avant le travail sur les mesures : 15.5 / 2.05 s sur l'uniforme, 12.9 /
1.83 ⚠ et 43.4 / 6.34 ⚠ sur les plans. Le ⚠ signale qu'elle **déborde** les 128 sommets là où le
BSP ne déborde pas : ce sont les polyèdres intermédiaires, donc l'ordre des coupes, pas le résultat.

**Le gain des mesures par accumulation** (§ 5), mesure appariée, trois tours :

| | avant | après | |
|---|---|---|---|
| uniforme, 1 fil | 6.246 s | 5.313 s | **−14.9 %** |
| plans / Voronoï, 1 fil | 3.126 s | 2.658 s | **−15.0 %** |
| plans / volumes égaux, 1 fil | 5.586 s | 5.050 s | **−9.6 %** |
| uniforme, 8 fils | 0.807 s | 0.685 s | **−15.1 %** |
| plans / Voronoï, 8 fils | 0.413 s | 0.352 s | **−14.8 %** |
| plans / volumes égaux, 8 fils | 0.772 s | 0.703 s | **−9.0 %** |

Et l'ablation, qui dit ce qu'il reste à prendre : `measure()` pesait **21.7 %** du diagramme 3D, il
en pèse **13 %**. Le poste dominant est maintenant `Cell3::cut`, à **54 %**.

## 1.4 Le problème résolu : Newton

**2D, 8 fils** — deux régimes, et ils n'appellent pas le même travail :

| | uniforme 10⁵ | uniforme 10⁶ | lignes 10⁵ |
|---|---|---|---|
| itérations / diagrammes | 7 / 10 | 6 / 11 | **23 / 113** (89 reculs) |
| diagrammes | 0.42 s (27 %) | 3.65 s (20 %) | 7.62 s (**63 %**) |
| résolution linéaire | 0.98 s (**65 %**) | 13.19 s (**71 %**) | 3.84 s (32 %) |
| TOTAL | 1.52 s | 18.50 s | 12.03 s |

Sur l'uniforme, l'algèbre linéaire pèse les deux tiers : accélérer le diagramme ne s'y verrait pas.
Sur les lignes, c'est l'inverse.

**3D, 8 fils, plans `n = 10⁵`** — 13 itérations, 27 diagrammes (13 reculs), résidu `1.16e-10`,
identiques avant et après :

| | avant | après | |
|---|---|---|---|
| diagrammes | 20.07 s (83 %) | 16.54 s (80 %) | **−17.6 %** |
| résolution linéaire | 3.81 s | 3.81 s | — |
| TOTAL | 24.29 s | **20.75 s** | **−14.6 %** |

Le diagramme gagne plus ici que dans le banc pur (−17.6 % contre −15 %) parce que Newton demande le
volume **et** les facettes, et que les deux sortent désormais d'une seule accumulation.

En 3D le diagramme pèse **80 %** du total, contre 20 à 63 % en 2D : c'est là qu'il faut travailler,
et ça confirme le § 9.

**Une réserve à dire** : sur ce nuage, 439 coupes débordent 128 sommets pendant la résolution —
avant comme après. Les poids intermédiaires de Newton font des cellules bien plus grosses que la
solution, et le compte est signalé, pas silencieux.

## 1.5 Ce que ces chiffres ne disent pas

**Le compilateur décide parfois plus que l'algorithme.** Ce n'est pas une précaution de style,
c'est une mesure, rencontrée trois fois :

* la grille, cas dur : **cinq variantes du même code aux compteurs identiques** (4546 boîtes, 719
  balayées, 671.3 gardées, 871.8 coupes tentées) s'étalent de **14.55 à 25.59 s**. Une seule des
  cinq différences est algorithmique (l'ordre des anneaux, 8 %) ;
* la table de hachage de `gather_faces` : 128 entrées mesurent 2.8 % de plus que 256, alors que les
  deux sont vides à 94 % ;
* **la 2D perd 2 à 4 % dans ce commit sans avoir une instruction de différence.** La seule
  modification de `Cell.h` est une méthode template que `pd_bsp` n'instancie pas — donc elle
  n'engendre aucun code. Ce qui a changé, c'est la place du code 2D dans un binaire dont la partie
  3D a été récrite.

**Donc** : une mesure non appariée ne vaut rien ici, et un écart de moins de 5 % entre deux binaires
différents ne prouve rien. Les tableaux ci-dessus qui annoncent un pourcentage sont tous des
mesures appariées, tours alternés, machine au repos.

**Deux autres réserves.** `-march=native` est assumé : ces binaires ne quittent pas la machine.
Et les nuages durs 3D sont fabriqués par le banc lui-même, faute de pysdot sur cette machine (§ 8) —
ce sont des cas de chronométrage valides, pas des témoins indépendants pour Newton.

---

# 2. COMMENT ÇA S'UTILISE

```
xmake f -m release && xmake      # les douze binaires, 63 s ; un seul, 18 s
xmake run pd_check               # tous les accélérateurs contre le balayage complet, 2D et 3D
xmake run pd_bsp                 # LA SUITE : uniforme, nuage dur, masses égales -- 2D puis 3D
xmake run pd_bsp --help
```

**UN BINAIRE PAR ACCÉLÉRATEUR**, plus deux bancs transversaux. Chacun déroule la même suite, de
sorte que deux lignes de deux binaires différents se comparent directement.

| binaire | ce qu'il mesure | 3D | sondes qu'il porte |
|---|---|:---:|---|
| `pd_bsp` | le BSP médian aligné sur les axes — **la référence** | ✓ | `--stats`, `--majorant` |
| `pd_grid` | la grille régulière, tri par comptage, parcours par anneaux | ✓ | `--stats` |
| `pd_packed` | le même arbre dans UNE arène, points collés à leur feuille | | `--stats` |
| `pd_bsp4` | le BSP à QUATRE fils (`--lazy` pour `bsp4l`) — rejeté | | `--stats` |
| `pd_obsp` | coupes NON alignées, boîtes alignées — rejeté | | `--stats` |
| `pd_pre` | la pré-passe contre un sous-échantillon — rejeté | | `--stats` |
| `pd_hull` | la sur-cellule anisotrope (k-DOP par paquet) | | `--stats` |
| `pd_pack` | l'index par paquets | | `--enclos`, `--baisse` |
| `pd_front` | le FRONT sur un diagramme grossier — le plus rapide en 2D | | `--psigrid`, `--front-rate` |
| `pd_memo` | l'arbre qui se souvient des coupes | | (deux passes) |
| `pd_newton` | le problème de transport RÉSOLU, chronométré par poste | ✓ | `--ecrire`, `--ms-*`, `--memo` |
| `pd_check` | tous les accélérateurs contre le balayage complet | ✓ | `--grand` |

Les options communes sont les mêmes partout (`-n`, `--threads`, `--reps`, `--maxnv`, `--2d`/`--3d`,
`--load`, `--weights`, `--seed`, …) : elles vivent dans `src/bench/`, pas dans les `main`.

---

# 3. OÙ EST QUOI

```
src/util/           common.h (TF, SI, Vec<D>, CutResult), parallel.h (les fils, l'épinglage)
src/geometry/       Cell.h        la cellule 2D, sommets en ordre cyclique
                    Cell3.h       la cellule 3D, sommets + arêtes, polytope simple
                    PowerDiagram.h  la boucle : élagage, coupe, mesure
                    WeightMajorant.h  le majorant affine des poids
src/spatial_accel/  LES ACCÉLÉRATEURS : « quels germes peuvent couper cette cellule ? »
                    AaBsp.h AaBsp4.h AaBspHull.h AaBspMemo.h AaBspPack.h
                    AaBspPacked.h AaBspPre.h ObBsp.h Grid.h FrontPd.h
src/solver/         Newton.h      Newton amorti, le Laplacien de Laguerre, AMGCL / Cholesky / CG
src/bench/          Bench.h/.cpp  options, nuages, chronomètre, affichage -- LA SUITE
src/mains/          main_Xyz.cpp  un par accélérateur
cases/              les nuages durs, et gen_cases_3d.py qui fabrique ceux de 3D
```

`spatial_accel` et pas `accel` : dans ce dépôt « accélérateur » désigne aussi un GPU, et ce
répertoire n'en contient aucun.

---

# 4. LES DÉCISIONS, RÉSUMÉES

Une ligne par piste, le verdict, et la raison en une phrase. Le détail — les compteurs, les
contre-mesures, ce qui a été essayé avant d'abandonner — est dans le journal.

## Ce qui est retenu

| piste | ce que ça vaut | pourquoi ça marche |
|---|---|---|
| **BSP médian aligné, descente fils-le-plus-proche** | la référence | le nœud tient dans une ligne de cache, la coupe est en place |
| **`alignas( 64 )` sur les tableaux de sommets** | −8.5 % | une cellule ne chevauche plus deux lignes |
| **majorant AFFINE des poids** | ×3.1 sur le cas dur | `w( y ) ≤ a·y + b` par nœud : 466 → 135 boîtes, 538 → 61 coupes tentées |
| **boîte de cellule** (`--cellbox`) | dépend | achète un rejet en `O( 1 )` ; **gagnant pour la grille**, marginal pour l'arbre |
| **la grille pour l'uniforme** | 0.154 s à 8 fils | 2 germes par case, parcours par anneaux |
| **le FRONT sur diagramme grossier** | le plus rapide en 2D | l'index est fait du seul diagramme grossier |
| **mesures 3D par accumulation de faces** | **−15 %** sur le diagramme 3D | ni le volume ni les aires n'ont besoin de l'ORDRE des sommets (§ 5) |
| **`measure_and_facets` en un appel** | **−14.6 %** sur Newton 3D | volume et facettes sortent de la même accumulation |

## Ce qui est rejeté, et la raison

| piste | verdict | la raison, en une ligne |
|---|---|---|
| AVX-512 (`-mprefer-vector-width=512`) | rejeté | downclocking : 1.888 s contre 1.893, sans gain |
| BSP à 4 fils, `bsp4` / `bsp4l` | rejeté | `bsp4l` fait MOINS de travail (38 boîtes contre 42) et met **9 à 14 % de plus** |
| BSP à coupes obliques, `obsp` | rejeté | les boîtes filles se recouvrent : +41 % de coupes EFFECTIVES |
| pré-passe sur sous-échantillon, `pre` | rejeté | l'enclos par sous-échantillon est 6.9× trop large en 2D — c'est la dimension qui le tue |
| test d'éviction `O( 1 )` EXACT | rejeté | +5.5 % sur l'uniforme pour −5.2 % sur le cas dur |
| majorant de DEGRÉ 2 | pas fait | mesuré comme utile (0.072 contre 0.223), mais le nœud sortirait de sa ligne de cache |
| le BOUCLIER (index réutilisable dans Newton) | fermé | exact, et la mesure dit qu'il ne peut pas être rentable |
| `--front` dans la boucle de Newton | fermé | 2 à 4× sur le diagramme, **−21 à −46 % au total** : l'amorce coûte plus que ce qu'elle rend |

## Ce qui est ouvert

`--memo` (l'arbre qui se souvient des coupes) : borne supérieure mesurée à **−19 %** en 2D, jamais
essayé en 3D. Le multi-échelle dans Newton : écrit, mesuré, pas abouti. Et les accélérateurs
d'étude (`pack`, `hull`, `front`) ne sont pas portés en 3D.

---

# 5. LA GÉOMÉTRIE

## La cellule 2D : les sommets en ordre cyclique SONT la géométrie

L'aire se lit par le lacet, et l'invariant « la coupe `i` porte l'arête `[ v_i, v_i+1 ]` » donne les
voisins sans rien chercher. Tout tient dans deux tableaux `alignas( 64 )` de coordonnées et un
tableau d'identifiants de coupe.

## La cellule 3D : rien de tout cela ne survit

Il n'y a plus d'ordre cyclique global, et une coupe porte une FACE — un cycle d'arêtes qu'il faut
savoir retrouver. `Cell3T` stocke donc les **sommets avec leurs trois coupes** et les **arêtes**.
Ni les faces, ni les équations des plans : les premières se relisent, les secondes ne servent à
rien une fois la coupe faite.

L'hypothèse est celle du **polytope simple** — trois plans par sommet — et c'est elle qui rend la
coupe purement combinatoire : deux nouveaux sommets sont voisins exactement quand ils partagent une
ANCIENNE coupe.

Deux bornes, qui ne sont pas des réglages mais des identités d'Euler : `E = 3V/2` arêtes et
`F = 2 + V/2` faces. Le défaut est à **128 sommets en 3D** (32 en 2D) : à `n = 2·10⁵`, 64 débordait
135 fois et la somme des volumes tombait à `1.000000001`.

## Les mesures en 3D : accumuler les faces plutôt que les ordonner

C'est le poste qui a été travaillé en dernier, et le seul de la 3D à l'avoir été.

Rendre les faces **comme des cycles** coûte `O( F ( V + E ) )` : pour chaque face il faut balayer
tous les sommets, puis toutes les arêtes, pour savoir lesquels sont dessus. Sur une cellule de
Voronoï poissonienne 3D — 27 sommets, 40 arêtes, 15 faces — c'est quinze balayages complets de la
cellule. **Ablation mesurée : `measure()` pesait 21.7 % du temps du diagramme 3D.**

Or ni le volume ni l'aire d'une face n'ont besoin de l'ORDRE. Il suffit, par face, d'UN de ses
sommets `v_f` et de la somme `S_f` des produits vectoriels de ses arêtes vues depuis lui :

* l'**aire** vaut `|S_f| / 2` — la face est plane et convexe, donc les triangles `( v_f, arête )` la
  pavent et leurs produits vectoriels sont tous parallèles ;
* le **volume** vaut `Σ_f |( v_f − g ) · S_f| / 6` pour n'importe quel `g` intérieur : chaque terme
  est `2 × aire × hauteur`, et la valeur absolue dispense d'orienter les faces.

Tout cela s'accumule en **deux passes sans jamais chercher** : une sur les sommets, qui portent
chacun leurs trois faces ; une sur les arêtes, qui sont chacune sur exactement deux faces — les
deux coupes communes à leurs bouts. `O( V + E )` au lieu de `O( F ( V + E ) )`.

Trois choses ont été mesurées séparément, dans cet ordre :

| | uniforme 2·10⁵, 1 fil | ce que ça a apporté |
|---|---|---|
| cycles (départ) | 6.246 s | |
| accumulation, éventail depuis le centre de la face | 5.741 s | −8.2 % |
| éventail depuis un SOMMET de la face | 5.625 s | −2.0 % de plus, et plus aucun flottant dans la passe 1 |
| numéro de face par HACHAGE au lieu d'une recherche linéaire | 5.511 s | −2.0 % de plus |
| une autre écriture du même test, à sémantique identique | **5.313 s** | −3.7 % de plus — et ce n'est pas de l'algorithme (ci-dessous) |

Le dernier point mérite d'être dit : il y a `3 V` demandes de numéro de face par cellule — 81 sur
une cellule moyenne — pour une quinzaine de faces. La recherche linéaire pesait **un tiers** de
l'accumulation, moins par son nombre de comparaisons que par sa sortie de boucle imprévisible.

Deux détails qui ne se devinent pas et ont été mesurés :

* une table de hachage de **128 entrées mesure 2.8 % de plus que 256** sur le cas uniforme, alors
  que les deux sont vides à 94 % — c'est la sondation qui décide, pas le remplissage, et le `memset`
  de 1 Ko ne se voit pas ;
* **la dernière ligne du tableau ci-dessus n'est pas un progrès, c'est un avertissement.**
  `SI k = ht[h] >= 0 ? ht[h] : f.nf; if ( k == f.nf )` et `SI k = ht[h]; if ( k < 0 )` disent
  exactement la même chose, et mesurent 3.7 % d'écart. Ce n'est ni le système de compilation
  (`xmake` et `g++` à la main donnent 5.51 s tous les deux) ni l'ordre des déclarations sur la pile
  (essayé, nul) : c'est cette ligne-là. Et il faut dire l'autre moitié de la mesure — **la même
  modification fait PERDRE 1.9 % à `pd_newton`**. C'est du placement de code, propre à chaque
  binaire. La forme retenue est celle qui gagne sur le banc de diagramme, qui est ce que ce fichier
  sert à mesurer ; elle n'est pas « meilleure ». Même leçon que pour la grille, § 1.5.

**Effet de bord qui compte** : le parcours de cycle pouvait ÉCHOUER sur une dégénérescence et la
face était alors silencieusement omise. Il n'y a plus de parcours, donc plus d'échec possible.

`measure_by_cycles()` reste dans le fichier, hors du chemin chaud. C'est un **second chemin**, qui
ne partage avec le premier ni la façon de retrouver les faces, ni la décomposition en triangles, ni
le point d'où partent les tétraèdres. `pd_check` compare les deux cellule par cellule : en 3D il
n'existe aucune référence extérieure pour le volume d'un polyèdre, et leur accord EST le témoin.

---

# 6. L'ÉLAGAGE : le majorant affine des poids

Le seul endroit où une idée a rapporté un facteur, et pas quelques pour cent.

`ψ( x ) = min_i ( |x − p_i|² − w_i )`. Pour élaguer un nœud il faut majorer `w` sur ce nœud. Avec un
majorant **constant** (le max des poids du sous-arbre), le cas dur — des germes en lignes, résolus
pour des aires égales — teste 466 boîtes et tente 538 coupes par cellule pour n'en garder que 13.

Avec un majorant **affine** `w( y ) ≤ a·y + b`, ajusté par nœud : **135 boîtes, 61 coupes tentées**,
les mêmes 13 effectives. 1.632 s → 0.525 s à un cœur.

Les pentes sont stockées en `float` — un choix, pas une mesure, et `b` est donc calculé avec les
pentes ARRONDIES pour que le majorant reste un majorant. La formule de Cramer est gardée telle
quelle pour `D == 2`, ce qui rend le majorant 2D bit-pour-bit identique à ce qu'il était avant que
la dimension devienne un paramètre.

Un degré 2 resserrerait encore (0.072 contre 0.223 sur un nœud de 128 germes), mais il ferait sortir
le nœud de sa ligne de cache. Mesuré, écrit dans le journal, pas retenu.

---

# 7. LE SOLVEUR

Newton amorti (Kitagawa–Mérigot–Thibert) sur le Laplacien de Laguerre, jauge `w_0 = 0`.
`c_ij = |facette| / ( 2 |p_i − p_j| )` — longueur sur distance en 2D, aire sur distance en 3D : la
formule ne dépend pas de la dimension, seule la MESURE de la facette en dépend, et la cellule la
livre par `for_each_facet`. C'est la seule porte par laquelle Newton lit la géométrie, et c'est ce
qui lui permet d'ignorer la dimension.

Trois solveurs linéaires (`--solver amg | chol | cg`). Deux défauts ont été trouvés par la trace par
poste, et c'est pour ça qu'elle existe : un assemblage qui coûtait plus cher que le diagramme
lui-même (un tri), et un recul mal compté.

---

# 8. LES NUAGES DURS : `cases/`

Un nuage uniforme ne teste presque rien. Les cas qui font mal sont ceux où les cellules sont
anisotropes et où les germes sont très inégalement répartis :

* `lines5_n100000_s0.005_voronoi.txt` — cinq lignes, poids nuls ;
* `lines5_n100000_s0.005_equal.txt` — les mêmes germes, poids résolus pour des **aires égales** ;
* `planes4_n100000_s0.02_voronoi.txt` et `..._equal.txt` — leurs équivalents 3D, quatre plans qui
  traversent le cube, `σ` à l'échelle de `h = n^(−1/3)`.

Les 2D viennent de pysdot. **Les 3D non** : pysdot n'est plus installé sur cette machine (compilé
pour un interprète absent). `gen_cases_3d.py` fabrique les germes en numpy pur, et le cas à volumes
égaux est produit par le banc lui-même (`pd_newton --ecrire`). C'est donc un cas de
**chronométrage** valide, et **pas** un témoin indépendant pour Newton — la réserve est écrite dans
le générateur aussi.

---

# 9. CE QUI RESTE À ESSAYER, EN 3D D'ABORD

* **`Cell3T::cut`**, qui pèse maintenant **54 %** du diagramme 3D. C'est le prochain poste, et de
  loin le plus gros ; `measure` vient d'être traité et pèse 13 %.
* **le nœud de BSP en 3D** ne tient plus dans une ligne de cache (six bornes en `double` plus trois
  pentes). Les bornes en FP32 le feraient rentrer — même piste qu'en 2D, mais elle y était
  facultative et ici elle ne l'est plus.
* **`--memo` en 3D**, la seule dimension où le gain de 3× d'`old_pd` a été observé.
* **les accélérateurs d'étude en 3D** — `pack`, `hull`, `front`. `front` s'appuie sur les sommets
  d'une cellule grossière, ce qui est dimension-générique tel quel ; `hull` a besoin d'un k-DOP 3D.
* l'ancienne liste, toujours ouverte : `CellAoS` contre `CellSoA` ; du SIMD explicite sur le test
  d'éviction, qui ne se vectorise pas à cause de son `return` anticipé ; une grille / quadtree en
  ordre Z à côté du BSP.

Et surtout : **reporter dans `sdot` ce qui a payé ici**, à commencer par la question de fond que ce
banc a tranchée — pourquoi le même algorithme y monte à 7.6× et n'y monte qu'à 2.1× sous SYCL.
