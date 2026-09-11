# Le journal du banc

Les mesures **datées**, dans l'ordre où elles ont été faites, avec le raisonnement qui les a
amenées et les impasses telles qu'elles ont été rencontrées. Le [README](README.md) en donne la
synthèse ; ce fichier-ci en est la preuve, et il n'est pas réécrit après coup — une mesure datée
qu'on récrit n'est plus une mesure.

Les sections d'avant le 2026-09-04 parlent de `--tree X` : c'était l'ancien `main.cpp` unique. La
correspondance est mécanique — `--tree bsp` est `pd_bsp`, `--tree front` est `pd_front`, et ainsi de
suite.

---

## Ce qu'il a déjà servi à établir

Même algorithme que `sdot` (BSP médian, descente en profondeur fils-le-plus-proche, coupe en place,
mêmes formules), 1e6 germes uniformes dans le carré unité, FP64, Xeon W-2145 :

| threads | temps | accélération |
|---|---|---|
| 1 | 1.893 s | 1.00× |
| 2 | 0.968 s | 1.96× |
| 4 | 0.488 s | 3.88× |
| 8 | **0.248 s** | **7.63×** |

(état initial ; après les optimisations ci-dessous : **0.213 s** à 8 threads.)

Et le comparatif à 8 cœurs physiques, même machine, même travail :

| par cellule | ce banc | `sdot` (SYCL) | pysdot |
|---|---|---|---|
| temps total | **0.213 s** | 1.185 s | 0.446 s |
| références L3 | **1.7** | 199 | 5.9 |
| instructions | **12 360** | 17 500 | 26 950 |
| IPC | 1.60 | 0.456 | 1.96 |

**L'algorithme est parfaitement scalable** — ce que la version SYCL ne montrait pas (2.1× sur huit
cœurs), et ce qui avait fait conclure, à tort, à une limite structurelle. Le même algorithme en C++
nu monte à 7.6×, avec **115× moins de défauts de cache par cellule**.

La différence la plus visible entre les deux : ici la cellule est un objet de PILE (`Cell c;`), donc
privée, chaude en L1, ~1 Ko ; dans `sdot` les cellules de travail sont des tampons FFI globaux
indexés par work-item, donc chaque accès à un sommet traverse la hiérarchie mémoire et les seize
cellules cohabitent dans une même allocation. C'est la piste à instruire côté `sdot`.

## Les pistes essayees (2026-09-02)

Bruit : ±0.001 s a 8 threads une fois `--reps 8` (trois passes donnent le meme chiffre au
milliseme), donc 2 % est significatif.

| piste | 1 thread | 8 threads | verdict |
|---|---|---|---|
| depart | 1.893 s | 0.248 s | |
| **`alignas( 64 )` sur `vx`, `vy`, ...** | 1.726 s | 0.227 s | **−8.5 %**, garde |
| AVX-512 (`-mprefer-vector-width=512`) | 1.888 s | 0.246 s | perdant (downclocking), rejete |
| **feuilles empaquetees (`--tree packed`)** | 1.847 s | 0.238 s | **−2 %**, garde |
| pre-tri de la premiere feuille | 1.853 s | 0.245 s | perdant a TOUTES les feuilles, rejete |
| `--skip-inside` | | 0.218 s | −2.5 % vs 0.224 |
| **`--no-cellbox`** | | **0.213 s** | **−4.9 %** vs 0.224, garde |
| `--maxnv` 16 / 24 / 48 / 64 | | 0.213–0.216 s | AUCUN effet ; 12 CASSE (debordement) |

**Configuration retenue : `--tree packed --no-cellbox`, `nv = 32`, `leaf = 10`.**

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| temps | 1.620 s | 0.838 s | 0.423 s | **0.213 s** |
| acceleration | 1.00x | 1.93x | 3.83x | **7.61x** |

### Ce que chacune apprend

* **Alignement** : le vectoriseur passe la boucle des produits scalaires en vecteurs de 32 octets
  (AVX2, 4 `double`), verifie par `-fopt-info-vec-optimized`. Pas besoin de SIMD a la main pour le
  clip. L'AVX-512 perd : le downclocking mange le gain.
* **Feuilles empaquetees** : `[ en-tete ][ germe 0 ][ germe 1 ] ...` d'un seul tenant dans une
  arene, offsets au lieu d'index. Visiter une feuille touche des lignes CONSECUTIVES au lieu de
  quatre regions eloignees (`nodes`, `px`, `py`, `order`). Gain reel mais modeste -- la localite
  etait deja bonne.
* **Pas de boite de cellule** (`--no-cellbox`) : la boite achetait un rejet en `O( 1 )` avant le
  balayage des sommets, mais il fallait la REFAIRE apres chaque coupe effective. Mesure : elle ne
  se rembourse pas. Le balayage des sommets sort des le premier sommet dans le cas courant, ce que
  le rejet en `O( 1 )` n'ameliorait donc pas beaucoup.
* **Pas de test pour la boite qui contient le germe** (`--skip-inside`) : une telle boite ne peut
  jamais etre evincee, et il y en a une par niveau sur le chemin racine -> feuille. Gagne seule
  (−2.5 %), mais fait double emploi avec `--no-cellbox` : les deux ensemble donnent 0.220, moins
  bon que `--no-cellbox` seul.
* **Pre-tri de la premiere feuille** par distance au germe : perdant a 6, 10, 16, 24, 40 et 64
  germes par feuille. Les points d'une feuille sont deja groupes -- c'est ce qu'EST une feuille --
  donc l'ordre interne ne change presque rien a la vitesse de retrecissement.
* **Nombre max de sommets** : sans effet de 16 a 64. A 12 la somme des aires devient 1.0000027 :
  des cellules debordent, la coupe est refusee et la mesure est FAUSSE. Le banc l'attrape (il rend
  un code d'erreur), et ca fixe le plancher.

## LA GRILLE : amorcer, puis completer

L'idee n'est pas de remplacer l'arbre par une grille, c'est de **chercher les premiers candidats**
sans descendre, puis de completer contre une cellule DEJA PETITE.

1. les 3x3 cases autour du germe, en acces direct, **sans test d'eviction** -- une case voisine est
   presque toujours utile, la tester couterait plus que de la couper ;
2. puis les anneaux `r = 2, 3, ...`, case par case, contre la cellule que l'etape 1 a deja reduite.

Ce qui rend l'etape 2 exacte ET courte : un germe `y` ne peut couper que s'il existe `p` dans la
cellule avec `|p - y| <= |p - p0|` ; or `|p - y| >= |y - p0| - |p - p0|`, donc **au-dela de `2R`**,
avec `R = max |p - p0|` sur la cellule, **rien ne peut couper**. Les cases de l'anneau `r` sont a au
moins `( r - 1 ) h` du germe, ce qui donne un critere d'arret par anneau -- et `R` retrecit a chaque
coupe, donc la premiere passe fait exactement ce qu'il faut pour que la seconde s'arrete tout de
suite.

**Verifie germe par germe contre le balayage complet : 1.46e-16, avec ET sans poids.**

| germes vises par case | 1 | **2** | 3 | 4 | 6 | 10 | 16 | 24 |
|---|---|---|---|---|---|---|---|---|
| temps (8 threads) | 0.169 | **0.154** | 0.169 | 0.188 | 0.228 | 0.312 | 0.434 | 0.591 |

L'optimum est a **2 germes par case**, bien plus fin que le `leaf = 10` de l'arbre -- et c'est
logique : le voisinage 3x3 est de taille FIXE, donc il amorce avec `9 x leaf` candidats pour
~6 voisins reels. A 10 par case on en tente 90 pour en garder 6.

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| temps | 1.161 s | 0.606 s | 0.306 s | **0.154 s** |
| acceleration | 1.00x | 1.92x | 3.79x | **7.54x** |

Et la construction coute **36 ms** contre 313 pour l'arbre : un tri par comptage contre une
recursion de medianes.

### LES POIDS, et un critere qui etait FAUX

Le critere d'arret que j'avais ecrit d'abord -- « au-dela de `2R`, rien ne peut couper » -- n'est
valable QUE SANS POIDS. La derivation correcte : `y` coupe s'il existe `p` dans la cellule avec
`|p - y|^2 - w_y <= |p - p0|^2 - w0` ; avec `d = |y - p0|`, `r = |p - p0| <= R` et
`|p - y| >= d - r`, la condition necessaire est `d^2 - 2 d r + w0 - w_y <= 0`. Donc

    rien ne peut couper des que    d >= R    et    d^2 - 2 d R + w0 - wm > 0

ou `wm` MAJORE les poids de la region. A poids nuls ca redonne `d > 2R` ; avec des poids, un germe
lourd rapproche son plan de `p0`, donc une region qu'on croyait trop loin peut couper quand meme.

**Le bug etait reel, pas theorique.** Avec l'ancien critere, l'ecart contre le balayage complet :

| `--weights` | 0 | 100 | 1000 |
|---|---|---|---|
| ancien critere | 1.5e-16 OK | **3.9e-05 ECHEC** | **5.9e-04 ECHEC** |
| critere corrige | 1.5e-16 OK | 1.0e-16 OK | 1.0e-16 OK |

Des cellules trop grandes, silencieusement. Le meme oubli existait dans le test d'eviction
(`may_be_cut` n'avait pas son terme `- wb`), et les deux sont corriges : chaque region -- case de
grille ou noeud d'arbre -- porte desormais un MAJORANT de ses poids.

Ce majorant est ici CONSTANT (le poids max de la region). `sdot` en utilise un AFFINE, bien plus
serre quand les poids sont un potentiel qui varie regulierement dans l'espace -- le regime du
transport optimal semi-discret. Un constant suffit a etre correct, pas a etre bon.

Et pour la grille, le majorant utilise par le critere d'ANNEAU est GLOBAL : un seul germe tres
lourd force a elargir partout. Correct, mais lache -- le resserrer demanderait une hierarchie de
maxima (une mipmap sur la grille), ce que l'arbre a par construction. C'est une limite de plus a
mettre au passif de la grille, avec celle de l'uniformite.

**Ce que ca coute** : 0.154 -> 0.164 s a huit threads sur un nuage SANS poids, soit 6 %. Ce n'est
pas le critere (identique sans poids) mais le stockage : `Seed` passe de 24 a 32 octets en gagnant
son champ de poids, un tiers de trafic memoire en plus. Un parametre de template « avec ou sans
poids » le rendrait gratuit dans le cas euclidien -- pas fait.

### Ou ca met tout le monde, a 1e6 germes 2D FP64 sur cette machine

| | 1 coeur | 8 coeurs | construction |
|---|---|---|---|
| **ce banc, grille** | 1.161 s | **0.164 s** | 47 ms |
| ce banc, arbre | 1.637 s | 0.214 s | 313 ms |
| pysdot (`SpZGrid`) | 3.364 s | 0.446 s | inclus |
| CGAL (`Regular_triangulation_2`) | 1.522 s | pas de mode parallele en 2D | inclus |
| `sdot` (SYCL) | 2.489 s | 1.185 s | ~300 ms |

**2.9x devant pysdot** a huit coeurs, et devant CGAL a UN coeur.

RESERVE : une grille suppose une densite a peu pres uniforme. Des amas la rendraient vide par
endroits et bondee ailleurs, et les anneaux s'etendraient loin -- le critere reste EXACT, mais plus
le cout. Pour le cas general il faut un decoupage (Metis) ou l'arbre, et c'est la que la structure
hierarchique existante garde son sens (avec ses bornes sur les poids).

## DES NUAGES DURS : `cases/`

Tout ce qui precede est mesure sur de l'UNIFORME, et l'uniforme cache deux hypotheses dont aucune
n'est vraie en transport optimal :

1. la densite est a peu pres constante -- c'est ce qui fait qu'une grille reguliere est bien remplie ;
2. la cellule d'un germe est AUTOUR de ce germe -- c'est ce qui fait que descendre vers sa feuille
   amorce bien la cellule, et que le rayon `R` decroit vite.

`cases/gen_cases.py` produit des nuages qui violent les deux : des diracs serres autour de quelques
lignes qui traversent le carre. Deux jeux de poids sur LE MEME nuage :

* **`voronoi`** -- poids nuls. La densite varie de plusieurs ordres de grandeur : a 2000 germes les
  aires vont de `1.5e-6` a `3.0e-2`, soit un facteur 20 000. C'est l'hypothese 1 qui tombe.
* **`equal`** -- les poids qui rendent TOUTES les aires egales a `1/n`. C'est l'hypothese 2 qui
  tombe, et bien plus violemment qu'on ne l'imagine : la distance mediane d'un germe au centre de
  sa propre cellule passe de `0.003` (Voronoi) a `0.114`, avec un maximum de `0.366`. La cellule
  mesure `sqrt( 5e-4 ) = 0.022` de cote -- une cellule est donc typiquement a CINQ FOIS sa propre
  taille de son germe, et parfois au tiers du domaine.

Le format est un texte : des lignes `#`, puis `n`, puis `n` fois « x y w ». Le banc le lit avec
`--load`, et `--check --load` verifie les accelerateurs contre le balayage complet sur le prefixe
du fichier.

### Le solveur : Newton, et pourquoi pas L-BFGS

Les poids du cas `equal` maximisent le dual de Kantorovich

    Phi( w ) = somme_i [ integrale_{Lag_i(w)} |x - p_i|^2 dx - w_i |Lag_i(w)| ] + somme_i w_i nu_i

concave, de gradient `nu_i - |Lag_i(w)|`. Le gradient etant gratuit -- c'est l'aire, qu'on calcule
de toute facon -- L-BFGS est le premier reflexe, et c'est ce que le script fait avec `--solver
lbfgs`. MESURE : il stagne vers `3e-4` d'erreur relative d'aire, apres 3000 evaluations et 40 s a
n=2000. La raison est dans la formule : `Phi` porte le terme `somme_i integrale |x - p_i|^2`,
d'ordre `1`, alors que ce qui reste a gagner pres de l'optimum est d'ordre `1e-12` -- la
decroissance disparait sous l'arrondi de `Phi` bien avant que le gradient ne s'annule. Ce n'est pas
un probleme de reglage, c'est un plancher.

Newton amorti (celui de pysdot, `--solver newton`, le defaut) n'evalue JAMAIS `Phi`, seulement son
gradient et sa hessienne : le plancher n'existe pas. **10 iterations, 0.1 s, erreur `3e-11`** au
meme n. Un cas de test dont les aires ne seraient egales qu'a `3e-4` pres ne serait pas le cas
qu'on croit tester.

### Le debordement, qui etait silencieux

Une cellule qui atteint `max_nb_vertices` etait laissee TELLE QUELLE, donc trop grande, et son aire
etait fausse sans que rien ne le dise. Sur l'uniforme le cas ne se produit jamais et la question ne
se posait pas ; sur ces nuages elle se pose. `PowerDiagram` compte desormais les debordements
(atomique, touchee uniquement quand ca deborde, donc hors du chemin chaud), le banc les signale, et
`--check` echoue s'il y en a.

### Ce que ces nuages mesurent (n=1e5, `--maxnv 64`, aucun debordement)

| | arbre 1c | arbre 8c | grille 1c | grille 8c |
|---|---|---|---|---|
| uniforme | 0.164 s | 0.022 s | 0.255 s | 0.034 s |
| lignes, Voronoi | 0.166 s | 0.022 s | 1.112 s | 0.159 s |
| lignes, aires egales | 1.632 s | 0.274 s | 24.42 s | 3.774 s |

L'arbre est INSENSIBLE au regroupement (0.166 contre 0.164) ; la grille prend 4.4x sur le meme
nuage, et 96x sur le cas a aires egales. Le regime ou la grille gagnait -- l'uniforme -- etait bien
le seul.

### D'ou vient le cout : ni l'anisotropie, ni le maillage

`--stats` compte ce que le parcours fait vraiment, par cellule :

| arbre | boites testees | coupes tentees | coupes EFFECTIVES | sommets finaux | rayon `max abs(v - p0)` |
|---|---|---|---|---|---|
| uniforme | 42.5 | 25.3 | 12.00 | 5.99 | 0.0031 |
| lignes, Voronoi | 42.2 | 25.8 | 12.01 | 5.97 | 0.0017 |
| lignes, aires egales | **466.3** | **538.0** | **13.22** | **5.97** | **0.1273** |

**La cellule finale est la MEME** : 5.97 sommets, 13.2 coupes effectives contre 12.0. Ce n'est donc
pas l'anisotropie, et un maillage adapte a la FORME des cellules ne rapporterait rien -- les
cellules sont des hexagones ordinaires dans les trois cas. Ce qui explose est la RECHERCHE : 538
coupes tentees pour 13 utiles, soit 97.5 % de calculs qui n'enlevent rien.

La cause est dans la derniere colonne. Le rayon `max abs( v - p0 )` -- exactement la quantite qui
majore la cellule dans le test d'eviction -- passe de 0.0031 a 0.1273, un facteur 41. La cellule
n'est plus autour de son germe, donc la sphere qui la majore depuis `p0` englobe des milliers de
germes qu'il faut tous examiner. Et ce qui devrait les ecarter est le majorant de POIDS : un germe
lointain ne coupe que si son poids est assez grand.

## LE MAJORANT AFFINE

Le majorant etait CONSTANT par noeud -- le poids maximum de la region -- ce qui traite toute la
boite comme si le germe le plus lourd etait partout. Les poids d'un probleme de transport sont un
potentiel qui varie regulierement dans l'espace : c'est le pire cas pour une borne constante.

Il est desormais AFFINE, `w( y ) <= a . y + b` (voir `WeightMajorant.h`, repris de `sdot`), et le
point decisif est que **le degre 1 ne coute rien de plus a tester** : le minimum de
`abs( p - y )^2 - a . y` sur une boite reste SEPARABLE par axe, son minimum libre est en
`y = p + a / 2`, et un `clamp` par axe y repond exactement. Le majorant constant est le meme code
avec `a = 0`. Un degre 2 casserait cette separabilite.

Ce que ca donne, sur le cas a aires egales (n=1e5) :

| | constant | affine |
|---|---|---|
| boites testees / cellule | 466.3 | **135.2** |
| coupes tentees / cellule | 538.0 | **60.7** |
| coupes effectives / cellule | 13.22 | 13.22 |
| temps, 1 coeur | 1.632 s | **0.525 s** |
| temps, 8 coeurs | 0.274 s | **0.092 s** |

**3.1x**, et les coupes effectives ne bougent pas d'un chiffre -- c'est bien de l'elagage gagne, pas
du travail deplace.

**Ce que ca coute quand il ne sert a rien** : sur un nuage sans poids, de l'ordre de 4 % (0.164 ->
0.170 s) -- a prendre avec des pincettes, les deux mesures venant de binaires differents et
l'agencement du code valant a lui seul plusieurs pour cent ici (voir plus bas). Le gain de 3.1x sur
le cas dur, lui, est trop grand pour etre de cet ordre.
Pas l'arithmetique -- `Weighted` est un parametre de TEMPLATE, donc sans poids les deux produits,
les deux maximums et le decalage par sommet disparaissent a la compilation, comme dans `sdot` ou le
tenseur des poids est simplement absent. Ce qui reste est la TAILLE DU NOEUD : 56 -> 64 octets,
parce qu'il porte deux pentes de plus. Elles sont en `float` et non en `TF`, et ce n'est pas une
approximation : `a` est un CHOIX et non une mesure, donc n'importe quel `a` donne un majorant
valide pourvu que `b` soit calcule AVEC lui -- on arrondit les pentes d'abord, `b` ensuite.

Le choix entre affine et constant se fait NOEUD PAR NOEUD, sur l'etalement des residus, avec une
correction pour le resserrement que le hasard donne deja a trois parametres sur peu de points
(`sqrt( 1 - d / ( m - 1 ) )`). Sans elle, un noeud de poids purement aleatoires retenait l'affine
une fois sur trois : toujours valide, mais un vecteur de plus a lire pour une borne qui ne vaut
pas mieux.

## CGAL SUR LES MEMES NUAGES, et ce que ca dit

`sdot/bench/cgal/power_2d --load <fichier>` lit les memes fichiers (la convention de poids est la
meme : CGAL minimise `abs( x - p )^2 - w`).

| n=1e5 | nous, 1 coeur | nous, 8 coeurs | CGAL, 1 coeur |
|---|---|---|---|
| uniforme | 0.170 s | 0.023 s | **0.141 s** |
| lignes, Voronoi | 0.176 s | 0.023 s | **0.142 s** |
| lignes, aires egales | 0.525 s | 0.092 s | **0.112 s** |

CGAL est INSENSIBLE au nuage, et le cas a aires egales lui est meme PLUS RAPIDE que l'uniforme. La
raison est structurelle : il construit le DUAL (la triangulation reguliere) une fois, donc son cout
est combinatoire -- de l'ordre de `3n` aretes, le meme pour les trois nuages -- et il ne se demande
jamais « quels germes pourraient couper cette cellule ». Nous faisons une RECHERCHE par cellule, et
c'est la recherche qui explose. Le cas a aires egales lui est plus rapide parce que ses cellules
sont presque hexagonales, donc mieux conditionnees qu'un Voronoi de nuage regroupe.

A un coeur il est devant partout. Notre avantage est le PARALLELISME (7.4x), que la
`Regular_triangulation_2` n'a pas en 2D -- et c'est exactement l'echange : CGAL entretient une
structure mutable partagee, nous construisons chaque cellule independamment, sur la pile, sans rien
partager. Sur le cas dur la marge tombe de 6x a 1.2x.

## LE VOISINAGE DE FEUILLES : le plafond est mesure

Les feuilles d'un BSP forment un maillage (elles pavent le domaine sans recouvrement), donc on
pourrait leur calculer une table d'adjacence et remplacer la descente racine -> feuille par une
propagation de proche en proche. Ce que ca economiserait est EXACTEMENT ce que `--skip-inside`
enleve deja : les boites qui contiennent le germe, c'est-a-dire le chemin de descente.

| n=1e5, 1 coeur | sans | `--skip-inside` |
|---|---|---|
| uniforme | 0.170 s | 0.160 s (**-6 %**) |
| lignes, Voronoi | 0.176 s | 0.169 s (**-4 %**) |
| lignes, aires egales | 0.538 s | 0.564 s (**+5 %**) |

Le plafond est donc de 6 % sur les cas faciles, et NUL sur le cas cher -- ou le test de contenance
coute meme plus qu'il ne rapporte, la descente n'y etant qu'une quinzaine de boites sur 135.

Et il y a pire pour cette piste : dans le cas a aires egales, la cellule N'EST PAS dans la feuille
de son germe. Une propagation partant de cette feuille devrait traverser une quarantaine de
feuilles avant d'atteindre la cellule. L'adjacence donne les REGIONS voisines, pas les CELLULES
voisines, et on ne sait pas d'avance quelle feuille porte la cellule.

Deux remarques pour memoire, si la piste revient :

* « quelle boite a deja ete ouverte » se resout par un petit ensemble a adressage ouvert SUR LA
  PILE (64 entrees suffisent pour ~40 feuilles visitees), donc sans etat partage. Mais ca coute de
  l'ordre de ce que ca economise, la ou l'arbre donne l'absence de revisite gratuitement.
* avec des POIDS, une propagation ne peut pas prouver seule qu'elle a fini. Geometriquement
  l'argument marche (tout chemin de la cellule vers une feuille non visitee traverse la frontiere,
  donc `dist( C, frontiere ) <= dist( C, dehors )`), mais le critere n'est pas la distance seule :
  `dist^2 + w0 - w_L`, et un germe lointain et lourd coupe a travers une frontiere qui a passe le
  test. Il faudrait donc GARDER une hierarchie pour le champ lointain -- exactement ce que le
  `wmax_all` global fait pour les anneaux de la grille.

## LE BSP A QUATRE FILS : `--tree bsp4`, `--tree bsp4l`

On coupe comme avant — médiane sur l'axe le plus long — mais on recoupe tout de suite chaque
moitié et on range **les deux niveaux dans un seul nœud**. Les feuilles sont donc exactement
celles de `bsp` à `--leaf` égal : seule la façon de descendre change. Le nœud passe de 64 octets
(une ligne de cache) à 244 (quatre boîtes en SoA, quatre majorants, quatre indices), mais il y a
quatre fois moins de nœuds internes, et une visite ne suit plus qu'un seul pointeur.

Ce que ça vise : une visite de `bsp` touche **trois** nœuds — le sien pour la boîte et le
majorant, puis ceux des deux fils pour la `nearness` qui décide de l'ordre d'empilement. Les deux
derniers sont des chargements DÉPENDANTS, dont la latence ne se recouvre pas.

Il y a deux façons de le faire, et il fallait les deux :

* `bsp4` teste les quatre boîtes **à la sortie du père** et n'empile que celles qui passent ;
* `bsp4l` empile les quatre fils sans les tester et teste chacun **à sa propre sortie**, comme
  `bsp` — la pile porte alors un couple `( père, rang )`, la boîte d'un fils étant chez son père.

La distinction n'est pas cosmétique : `bsp` teste une boîte contre la cellule TELLE QU'ELLE EST au
moment du dépilement, et chaque coupe faite depuis l'empilement rend le « non » plus probable.

| n=1e5, 1 cœur, même binaire | `bsp` | `bsp4` | `bsp4l` |
|---|---|---|---|
| uniforme | **0.163 s** | 0.222 (+36 %) | 0.176 (+8 %) |
| lignes, Voronoï | **0.174 s** | 0.226 (+30 %) | 0.177 (+2 %) |
| lignes, aires égales | **0.538 s** | 0.573 (+7 %) | 0.588 (+9 %) |

À 8 fils : 0.022 / 0.031 / 0.023, 0.023 / 0.031 / 0.024, 0.088 / 0.091 / 0.098.

### Le test précoce coûte cher, et les compteurs disent exactement combien

`--stats`, par cellule, sur le nuage uniforme :

| | boîtes | balayées | gardées | coupes tentées |
|---|---|---|---|---|
| `bsp` | 42.5 | 28.5 | 25.1 | 25.3 |
| `bsp4` | 109.3 | 38.2 | 33.7 | 43.7 |
| `bsp4l` | **38.2** | **16.5** | **12.7** | **24.1** |

`bsp4` paie **2.6 fois plus de tests de boîte** — il en fait quatre par visite alors que `bsp`
rejette deux petits-fils d'un coup en testant leur moitié — et surtout il présente 73 % de germes
en plus à la coupe : jugées trop tôt, contre une cellule encore grande, les feuilles lointaines
passent le test qu'elles auraient raté cinq coupes plus tard. Le mécanisme annoncé dans l'en-tête
de `AaBsp.h` se lit donc directement dans le compteur.

### `bsp4l` fait moins de travail et met plus de temps

C'est le résultat intéressant. **Tous** les compteurs sont meilleurs que ceux de `bsp` — 38.2
boîtes contre 42.5, 16.5 balayages contre 28.5, 12.7 boîtes gardées contre 25.1 (les quarts sont
plus serrés que les moitiés) — et l'horloge donne quand même +8 %.

Ce n'est pas le débordement de cache : à trois tailles de nuage, l'écart ne bouge pas.

| uniforme, 1 cœur | `bsp` | `bsp4l` |
|---|---|---|
| n=1e4 (l'arbre tient en L2) | 1457 ns/germe | 1629 (+12 %) |
| n=1e5 | 1622 | 1764 (+9 %) |
| n=1e6 | 1712 | 1955 (+14 %) |

Ce qui reste est le **coût par visite** : quatre `nearness` au lieu de deux, un tri à quatre
éléments au lieu d'une comparaison, quatre empilements au lieu de deux, et un nœud lu deux fois
(chez le père pour la boîte, chez le fils pour l'ordre). L'arité fait bien tomber le nombre de
tests, mais chaque test de boîte est **déjà** si bon marché — quelques `max` séparables par axe —
que l'échanger contre du tri est perdant.

Sur le cas à aires égales, où la boîte gardée compte pour beaucoup plus (77.2 contre 25.1), `bsp4l`
descend à 41.1 gardées et perd quand même 9 % : là c'est le balayage de sommets (111.8 contre
128.0) qui aurait dû payer, et il ne suffit pas.

**Conclusion : rejeté, dans les deux variantes.** Ce que ça vaut quand même, c'est la mesure du
poids de « tester à la sortie » : 36 % contre 8 % sur le même arbre, pour le seul déplacement du
test d'un niveau plus haut.

## LES BSP NON ALIGNES : `--tree obsp`

L'argument de principe d'abord, parce qu'il décide de ce qu'on peut essayer. Ce qui coûte n'est pas
l'orientation des coupes mais la FORME des régions. Le test d'éviction demande
`min abs( p - y )^2` et `max_y( a . y )` sur la région : sur une boîte alignée les deux sont
séparables par axe, donc deux `clamp` et deux coins. Sur un polytope convexe général, le point le
plus proche est un problème quadratique — plus rien n'est en `O( 1 )`. Un k-DOP à directions
FIXES ET PARTAGÉES garderait une minoration en `O( k )`, donc l'orienté n'est abordable que si les
directions ne sont pas propres à chaque nœud.

`obsp` teste donc la seule variante praticable sans toucher au test : **la coupe est oblique, le
volume englobant reste la boîte alignée des germes du sous-arbre**. Tout l'aval — parcours, test
d'éviction, `nearness` — est le code de `bsp` inchangé, ce qui isole vraiment l'effet de
l'orientation. La direction est choisie par un critère SAH parmi six candidates (`x`, `y`, les deux
diagonales, l'axe principal du nuage local et sa perpendiculaire), notées par la somme des
demi-périmètres des deux boîtes filles, évaluées sur un échantillon de 256 germes.

| n=1e5, 1 cœur, même binaire | `bsp` | `obsp` | `obsp` bridé aux axes |
|---|---|---|---|
| uniforme | **0.161 s** | 0.176 (+9 %) | 0.173 (+7 %) |
| lignes, Voronoï | **0.173 s** | 0.257 (**+48 %**) | 0.175 (+1 %) |
| lignes, aires égales | **0.538 s** | 0.644 (+20 %) | 0.533 (−1 %) |

La troisième colonne (`PD2D_OBSP_AA=1`, les deux candidates alignées seulement) est le témoin : elle
mesure ce que coûte le fait d'être une AUTRE instanciation du même code. Sur l'uniforme elle
absorbe presque tout l'écart — la perte n'y est pas imputable à l'obliquité. Sur les deux nuages de
lignes elle est nulle, donc **le +48 % et le +20 % sont bien de l'obliquité**.

### Pourquoi ça perd : les boîtes filles se recouvrent

Une coupe alignée laisse deux boîtes DISJOINTES le long de l'axe coupé. Une coupe oblique laisse
deux boîtes qui se recouvrent — couper un carré par sa diagonale donne deux triangles dont les
boîtes font chacune presque tout le carré. Les compteurs, sur le nuage de lignes Voronoï :

| | boîtes | balayées | gardées | coupes tentées | coupes **effectives** |
|---|---|---|---|---|---|
| `bsp` | 42.2 | 28.5 | 25.0 | 25.8 | 12.01 |
| `obsp` | 52.1 | 38.2 | 31.8 | 37.6 | **16.95** |

La cellule finale a 5.97 sommets des deux côtés : ce sont donc **5 coupes effectives sur 17 qui
taillent un morceau aussitôt retaillé**. Le recouvrement des boîtes ne dégrade pas seulement
l'élagage, il dégrade l'ORDRE : `nearness` classe des boîtes qui s'interpénètrent, le parcours
attaque la cellule de loin, et le polygone intermédiaire enfle. Ça se voit jusque dans le
débordement : `obsp` dépasse 64 sommets sur 8 à 9 cellules là où `bsp` n'y touche jamais.

Et le critère SAH n'y est pour rien : `PD2D_OBSP_DIRS=1` compte les directions retenues, et il
choisit `x` ou `y` dans **81 %** des nœuds (40 % `x`, 41 % `y`, 12 % une diagonale, 6 % l'axe
principal). Ce sont les 19 % de coupes obliques restantes,
choisies parce qu'elles resserrent les boîtes, qui coûtent 48 %. **Minimiser la taille des boîtes
est le mauvais objectif dès qu'elles peuvent se recouvrir** — une pénalité de recouvrement dans le
score les ramènerait aux axes, c'est-à-dire à `bsp`.

Enfin la construction passe de 16-35 ms à 42-87 ms, soit un quart de passe de plus à payer avant
même de commencer.

**Conclusion : rejeté.** Et l'argument de principe est renforcé plutôt qu'infirmé : l'oblique ne
peut payer que si le volume englobant l'est aussi (k-DOP à directions partagées), c'est-à-dire au
prix d'un test d'éviction en `O( k )` — le même compromis que `--tree hull` a déjà tranché ailleurs,
avec la différence qu'`hull` ne l'applique qu'une fois par paquet et non à chaque nœud.

Note : le banc dit par ailleurs que le problème n'est pas là. La cellule finale a 5.97 sommets sur
les trois nuages ; des coupes orientées répondraient à une anisotropie qu'on n'a pas observée.

## ESSAYE ET REJETE : le test `O( 1 )` EXACT

Le test en `O( 1 )` minore `dist^2( C, B )` et majore `max abs( p - p0 )^2` SEPAREMENT, donc en des
points differents de la cellule. Quand la cellule est loin de son germe la borne devient tres lache,
et le compteur le montre : sur le cas a aires egales il n'evince que **7 boites sur 135**, les 128
autres partant au balayage des sommets.

Or les trois termes sont separables par axe, donc leur somme peut etre minimisee EXACTEMENT axe par
axe : `h( t )` est lineaire dans les deux regimes ou le point de `B` est colle a un bord (le `t^2`
s'annule) et concave dans celui ou il suit `t`, donc quatre candidats suffisent -- `clo`, `chi`, et
les deux ruptures.

Ecrit, verifie, mesure :

| n=1e5, 1 coeur | sans | avec le test exact |
|---|---|---|
| uniforme | 0.182 s | 0.192 s (+5.5 %) |
| lignes, Voronoi | 0.183 s | 0.194 s (+6.0 %) |
| lignes, aires egales | 0.553 s | 0.524 s (**-5.2 %**) |

Les balayages de sommets tombent bien de 128 a 90 par cellule sur le cas dur -- mais les boites
GARDEES ne bougent pas d'un chiffre, 77.2 avant comme apres. **Le test exact ne change jamais la
reponse** : il ne fait que deplacer des decisions du balayage vers le `O( 1 )`, parce que le
balayage etait deja aussi serre que la boite le permet. On paie quatre fois le test sur toutes les
boites pour eviter 38 balayages, et ca s'annule.

Ce que ca N'INFIRME PAS : un meilleur majorant de POIDS. Lui change la reponse elle-meme -- il fait
tomber « gardees » -- donc il elague des sous-arbres entiers, ce que ce test-ci ne faisait pas.

## LE MAJORANT DE DEGRE 2 : ce qu'il aurait a donner

`--majorant` compare, noeud par noeud, l'ETALEMENT des residus de quatre ajustements (c'est
exactement le mou de la borne, puisque `b` est releve jusqu'au pire germe). Rapporte au majorant
constant, sur le cas a aires egales :

| germes/noeud | affine | diagonal `x^2, y^2` | (hasard) | complet, avec `x y` | (hasard) |
|---|---|---|---|---|---|
| 8 - 15 | 0.058 | 0.029 | 0.802 | 0.005 | 0.744 |
| 128 - 255 | 0.223 | 0.156 | 0.990 | **0.072** | 0.987 |
| 2048 - 4095 | 0.273 | 0.226 | 0.999 | **0.141** | 0.999 |
| 65536 - 131071 | 0.921 | 0.890 | 1.000 | **0.419** | 1.000 |

La colonne « hasard » est le resserrement qu'un ajustement a `k` parametres donne sur du bruit pur ;
elle vaut 1.00 pour les gros noeuds, donc rien ici n'est un artefact.

Le degre 2 DIAGONAL -- le seul qui garde la separabilite par axe -- ne gagne qu'un facteur 1.3 a 1.4
sur l'affine. **C'est le terme croise qui porte le degre 2** (0.072 contre 0.156, et 0.419 contre
0.890 a la racine), et c'est lui qui casse la separabilite.

Mais il n'est pas perdu pour autant, EN 2D : minimiser une forme quadratique quelconque sur une
boite est un probleme quadratique sous contraintes de boite, et en dimension 2 la boite n'a que
quatre aretes -- le minimum est soit au point stationnaire interieur, soit sur une arete (une
minimisation a une variable, donc un `clamp`), soit a un coin. Cela fait cinq evaluations en
`O( 1 )`, environ cinq fois le test affine. En tant que SECOND etage, teste seulement sur les 77
boites que le premier garde, l'arithmetique tient. Ce qui reste inconnu est ce que « trois fois
moins de mou » enleve vraiment de « gardees » -- et c'est la seule chose qui compte.

## Ce qu'il contient

Le cœur est **templaté sur la dimension** : `Cell::dim`, `Accel::dim`, et `PowerDiagram` qui
vérifie que les deux coïncident.

* `common.h` — `TF`, `SI`, `CutResult`, et `Vec<D>`, le point passé PAR VALEUR (un agrégat de deux
  ou trois `double` voyage en registres ; un `const TF *` forcerait à relire la mémoire, d'autant
  que le banc compile `-fno-strict-aliasing`).
* `Cell.h` — `CellSoAT`, **2D**, 32 sommets par défaut, aucune allocation. `cut` est EN PLACE
  (seuls les deux points d'intersection sont écrits, plus le décalage), `measure` est un lacet.
  L'invariant qui rend tout ça possible : *la coupe `i` porte l'arête `[ v_i, v_i+1 ]`*.
* `Cell3.h` — `Cell3T`, **3D** : sommets + leurs TROIS coupes, arêtes. Ni faces ni équations de
  plans — les faces se relisent, le volume se calcule par tétraèdres sur un point intérieur.
  Hypothèse : polytope SIMPLE (trois plans par sommet), ce qui rend la coupe combinatoire.
* `AaBsp.h` — `AaBspT<D>`, le BSP médian, numéroté **en préordre** (fils gauche en `n+1`, fils droit
  stocké), un nœud dans une ligne de cache **en 2D** (en 3D il en prend deux, et rien n'est gratuit
  pour l'éviter). Plus `EverySeedT<D>`, l'oracle en force brute, et `OneSeedT<D>`.
  Chaque accélérateur **porte son parcours** (`for_each_candidate`) : le concept `Walk` séparé
  décrivait un arbre binaire à boîtes, donc ne se généralisait à rien — la grille ne pouvait pas
  l'implémenter, et son parcours ne pouvait pas s'appliquer à un arbre. Un accélérateur ne doit
  qu'une chose à l'appelant, PROPOSER DES GERMES, et c'est sa seule méthode. Le parcours
  meilleur-d'abord (file de priorité, ce que fait `SpZGrid`) est parti avec le concept : il avait
  été mesuré perdant, 0.279 s contre 0.246.
* `WeightMajorant.h` — `WMajT<D>`, le majorant **affine** des poids d'une région, et le choix nœud
  par nœud entre lui et le constant. La formule de Cramer est gardée en `D == 2` pour que le
  majorant reste bit pour bit celui d'avant.
* `Grid.h` — `GridT<D>`, la grille régulière (tri par comptage, germes groupés par case) et son
  parcours en deux temps. Ce que la dimension y change n'est pas le critère mais le PRIX d'un
  anneau de trop : `8 r` cases en 2D, `24 r² + 2` en 3D.
* `PowerDiagram.h` — le pilote, templaté sur `Cell` et `Accel`, plus `CellBox`, `SkipInside`,
  `Stats` et `Weighted` — tous des paramètres de TEMPLATE, pour qu'aucun ne survive à la
  compilation quand il ne sert pas.
* `Newton.h` — le problème de transport RÉSOLU : Newton amorti, laplacien de Laguerre, jauge
  `w_0 = 0`, et le solveur linéaire (AMGCL / Eigen / gradient conjugué maison). **Il ignore la
  dimension** : les deux cellules exposent `for_each_facet( f( coupe, mesure ) )`, et
  `c_ij = |facette| / ( 2 |p_i − p_j| )` ne demande rien d'autre.
* `parallel.h` — `parallel_for` avec découpe `blocks` ou `strided` et épinglage. Mesuré : aucune
  différence entre les deux (0.245 contre 0.244 s).
* `bench/` — les options communes, les nuages, le chronomètre, l'affichage, et **LA SUITE** : les
  cas importants en 2D et en 3D, que chaque `main_Xyz.cpp` déroule.

Les accélérateurs d'étude, tous **2D** (ils ont été écrits pour répondre à une question 2D précise,
et les porter en 3D sans la question serait du travail sans mesure au bout) :

* `AaBspPacked.h` — le même arbre, tout dans une arène, les points collés à leur feuille.
* `AaBsp4.h` — le même arbre à QUATRE fils par nœud (`bsp4` / `bsp4l`), mesuré et rejeté.
* `ObBsp.h` — le même arbre à coupes NON ALIGNÉES et boîtes alignées (`obsp`), mesuré et rejeté.
* `AaBspPre.h` — la pré-passe contre un sous-échantillon.
* `AaBspPack.h` — l'index par paquets. `AaBspHull.h` — la sur-cellule anisotrope.
* `AaBspMemo.h` — l'arbre qui se souvient des coupes de l'itération précédente.
* `FrontPd.h` — le FRONT sur un diagramme grossier : plus de marche dans l'arbre, une liste de
  candidats par germe.

## L'ENCLOS PAR SOUS-ÉCHANTILLON : ce que la 1D en dit

L'idée : borner la cellule d'un germe **sans la calculer**, en calculant d'abord le diagramme d'un
sous-ensemble `S`. Elle repose sur une seule observation, et c'est elle qui la rend gratuite —
`ψ(x) = minᵢ( |x−pᵢ|² − wᵢ )` est un **minimum**, donc retirer des germes ne peut que le remonter :
`ψ_S ≥ ψ` partout, sans correction ni marge. La cellule de `k` calculée contre `S` seul a moins de
contraintes que la vraie, donc elle la **contient** — pour tout `k`, qu'il soit dans `S` ou non. Les
germes omis ne feront que la rogner. C'est un enclos valide par construction, et il donne
directement le rayon initial du parcours, qui est aujourd'hui le domaine entier.

`cases/sub1d.py` mesure de combien cet enclos est trop large, en 1D où tout est exact : l'enveloppe
inférieure s'y calcule en O(n log n) par la pile monotone, et les deux régimes de poids du banc ont
une forme close — les poids à cellules toutes égales se lisent d'une récurrence sur les points de
rupture, puisque le transport 1D est croissant. Le script vérifie à chaque taux que l'enclos
contient bien la vraie cellule. n = 4096, quatre amas serrés :

| n/\|S\| | stratifié : médiane | décile 9 | max | aléatoire : médiane | décile 9 | max |
|---|---|---|---|---|---|---|
| 8  | 4.1 | 7.1 | 10.9 | 6.6 | 15.0 | 45.1 |
| 16 | 8.1 | 9.9 | 22.9 | 13.0 | 32.2 | 62.6 |
| 32 | 16.1 | 18.4 | 47.8 | 27.5 | 53.9 | 111.6 |

(cas à aires égales ; la largeur médiane d'une vraie cellule vaut `1/n`, donc l'enclos du domaine
entier vaut 4096.)

Trois choses en sortent.

**La médiane vaut exactement `n/|S|`**, sur les deux régimes. C'est ce qu'on attend en 1D : un
enclos contient environ `n/|S|` cellules. **En 2D ce serait la même chose en AIRE**, donc `√(n/|S|)`
en rayon — un seizième des germes donnerait un rayon initial ~4× le vrai.

**Le cas dur est le cas SAIN.** À aires égales la distribution est serrée (décile 9 à 9.9, pire cas
à 22.9 pour une médiane de 8.1) parce que toutes les cellules ont la même taille. En Voronoï, où les
largeurs couvrent quatre ordres de grandeur, la queue explose : décile 9 à 31, **pire cas à 16850**.
Une cellule minuscule au cœur d'un amas se retrouve enfermée dans un enclos énorme dès que l'amas
est sous-représenté dans `S`. C'est la bonne nouvelle : le régime où la méthode se tient est
justement celui qui coûte cher.

**La stratification n'est pas un détail.** Un tirage aléatoire de même taille perd un facteur 1.6 à
1.7 sur la médiane et 2 à 3 sur la queue. En 2D le sous-ensemble stratifié est gratuit — un germe
par sous-arbre du BSP à une profondeur fixée — donc c'est celui-là qu'il faut, pas le tirage.

## L'ENCLOS EN 2D : mesuré, et REJETÉ

`AaBspPre` fait la chose : il coupe d'abord contre `S` (un germe sur `pre_rate`), puis contre le
reste. `--enclos` mesure l'enclos, `--stats` et le chrono mesurent ce qu'il coûte. `--check` et
`--cross` vérifient qu'il ne change rien — la cellule est l'intersection de TOUS les demi-plans,
donc l'ordre des coupes ne peut pas changer le polygone.

**L'enclos est aussi serré que la 1D le promettait, et mieux.** Le rapport des MESURES vaut
exactement `n/|S|` sur les trois nuages, comme prédit. Le rapport des RAYONS suit `√(n/|S|)` sur
l'uniforme (3.49 pour 4 attendu à `n/|S| = 16`)... mais vaut **1.06** sur le cas dur :

| n/\|S\| | uniforme : rayon | mesure | Voronoï : rayon | mesure | aires égales : rayon | mesure |
|---|---|---|---|---|---|---|
| 8  | 2.37 | 6.0 | 2.58 | 6.6 | **1.03** | 6.0 |
| 16 | 3.49 | 12.7 | 3.95 | 14.7 | **1.06** | 12.0 |
| 64 | 6.86 | 48.5 | 11.01 | 83.4 | **1.14** | 46.1 |

La raison est celle qui fait la difficulté du nuage : la cellule y est LOIN de son germe, donc le
rayon `max |v − p₀|` est dominé par le trajet germe → cellule et non par la taille de la cellule.
Grossir la cellule de 16× en aire ne le bouge que de 6 %.

**Et ça ne sert à rien.** Sur le cas dur, `bsp` ouvre 135.2 boîtes par cellule ; `pre` en ouvre
199.0, dont 65.5 pour la pré-passe — la seconde passe tombe donc de 135.2 à 133.5. **1.7 boîte sur
135.** Au chrono, à un cœur :

| n=1e5, 1 cœur | bsp | pre r=8 | r=16 | r=32 | r=64 |
|---|---|---|---|---|---|
| uniforme | **0.170** | 0.267 | 0.273 | 0.272 | 0.265 |
| lignes / Voronoï | **0.173** | 0.274 | 0.287 | 0.287 | 0.282 |
| lignes / aires égales | **0.524** | 0.899 | 0.852 | 0.793 | 0.742 |

1.4× à 1.7× plus lent, et monotone en `pre_rate` : le mieux qu'on puisse faire est de rendre la
pré-passe négligeable, c'est-à-dire de ne pas la faire.

**Ce que ça apprend**, et c'est le vrai résultat : **le rayon n'était pas la contrainte qui mord.**
On donne au parcours un rayon à 6 % du rayon final avant sa première boîte, et il n'ouvre pas une
boîte de moins. La descente plus-proche-d'abord atteignait donc déjà ce rayon-là dès sa première
feuille. Les 77 boîtes gardées le sont parce qu'elles PEUVENT vraiment couper, pas parce que la
cellule serait encore trop grosse — le même constat que le test `O(1)` exact, par un autre chemin.
Ce qui reste à serrer est le majorant de POIDS, pas la borne sur la cellule.

**Le coût, lui, était prévisible et il est confirmé** : la pré-passe coûte 65.5 boîtes pour un
arbre de 6250 germes contre 135.2 pour 100 000. Un arbre 16× plus petit n'enlève que quatre
niveaux, et ce sont les moins chers — près de la racine les boîtes sont grandes et toujours
gardées. Le coût par cellule est en `O(1)` et non en `O(n)` : c'est ce qui condamne toute pré-passe
faite germe par germe.

**Un défaut trouvé au passage, qui vaut plus que l'expérience.** Dans la première version les deux
arbres se recouvraient, donc un germe de `S` était proposé deux fois et la cellule coupée deux fois
par le MÊME plan. Résultat : **un germe sur 100 000 avec une aire fausse de 10 %**, et une somme à
`1.000001886`. C'est ce qui a motivé `--cross` — le même calcul par deux accélérateurs, germe par
germe, au `n` demandé, là où `--check` est plafonné à quelques milliers de germes par son oracle en
`O(n²)`. La somme des aires ne l'aurait pas montré : un germe sur cent mille s'y voit à peine.

La cause n'était PAS le recouvrement mais l'interpolation, et le recouvrement ne faisait que la
révéler. `Cell::cut` écrivait l'intersection sous forme symétrique,

    pax = ( s1 * vx[ j0 ] - s0 * vx[ i1 ] ) / ( s1 - s0 )

pour que le point ne dépende pas du bout de l'arête par lequel on commence. Mais ce choix n'existe
jamais — le sommet dedans est toujours désigné en premier — et la symétrie coûtait la propriété qui
compte : avec `s_in == 0` cette forme rend `( s1 · v0 ) / s1`, qui **n'est pas `v0`** en flottant.
Le sommet posé sur le plan par la première coupe se décale donc d'un cran à la seconde, parfois de
l'autre côté du plan, et la suite du code suppose que les sommets dehors forment une plage cyclique
unique. La forme ancrée sur le sommet dedans,

    pax = vx[ j0 ] + ( vx[ i1 ] - vx[ j0 ] ) * s0 / ( s0 - s1 )

est algébriquement identique, vaut exactement `v_in` quand `t = 0`, et fait retomber l'écart à
`2.2e-16`. Elle est aussi **3 % plus rapide** — deux divisions au lieu de quatre (0.183 → 0.177 s
sur l'uniforme, 0.556 → 0.540 sur le cas dur, mesure appariée). Adoptée.

`--pre-overlap` reste, et devient un TEST : il rétablit le recouvrement exprès, donc force chaque
germe de `S` à être coupé deux fois par le même plan. C'est la configuration dégénérée la plus
facile à provoquer, et `--cross --pre-overlap` la vérifie sur les trois nuages.

## LES PAQUETS : un index fait du seul diagramme de puissance grossier

La suite logique : si l'enclos est bon et que c'est la *marche du BSP* qui coûte (65.5 des 199
boîtes de `pre`), autant supprimer le BSP. On range chaque germe fin `j` dans les **cellules
grossières** que son enclos borde ; les candidats de `k` sont l'union des paquets des cellules
grossières que **son** enclos borde. Plus une boîte à tester, plus de descente : quelques listes
contiguës.

C'est un sur-ensemble, et la preuve tient en une ligne : si `C(j)` et `C(k)` se touchent en `x`,
alors `x` est dans les deux enclos, donc la cellule grossière qui gagne en `x` rencontre les deux —
elle est dans les deux listes, et `j` sort du paquet. Et les cellules grossières qu'un enclos
rencontre se lisent **gratuitement**, ce sont les `cid` de ses arêtes : sur `D_r`, `h_r` *est* le
minimum sur `S`, donc `D_r ∩ C_S(j) = D_r ∩ { h_j ≤ h_r }` est d'aire non nulle exactement quand la
bissectrice `(j,r)` coupe `D_r`, c'est-à-dire quand `r` borde l'enclos.

`--enclos` mesure les deux quantités qui décident (n=1e5) :

| n/\|S\| | cellules grossières bordées | candidats : médiane | décile 9 | max | manques |
|---|---|---|---|---|---|
| 2  | 5.88 | **36** | 46 | 73 | 0 |
| 4  | 5.70 | 66 | 84 | 130 | 0 |
| 8  | 5.45 | 121 | 151 | 220 | 2 |
| 16 | 5.57 | 251 | 316 | 505 | 2 |
| 64 | 5.35 | 945 | 1197 | 1793 | 2 |

(uniforme ; le cas dur donne les mêmes chiffres à 3 % près, et **zéro manque à tous les taux**.)

**Le nombre de cellules grossières bordées vaut 5.5, quel que soit le taux.** C'est la borne
d'Euler et non un accident : un enclos est une cellule d'un diagramme, donc il a six côtés, que `S`
fasse `n/2` ou `n/64` germes. C'est ce fait qui gouverne tout le reste — et il rend faux le compte
« 2 à 4 cellules, donc 32 à 64 candidats » qu'on pouvait faire de tête à partir du rapport des
mesures.

**Les listes sont donc proportionnelles au TAUX** : total des paquets = `5.5 n`, réparti sur
`n/ρ` paquets, soit `5.5 ρ` germes chacun et `5.5 × 5.5 ρ` candidats avant recouvrement. Pour
raccourcir les listes il faut un diagramme grossier FIN, et c'est là que la hiérarchie se paie : les
niveaux successifs coûtent `1/(ρ−1)` du niveau de base. En unités « un test de boîte ≈ une coupe
tentée » (grossier, mais sur le cas dur 128 des 135 tests de boîte vont jusqu'au balayage des
sommets, donc pas absurde) :

| | candidats + 6 coupes grossières | × surcoût hiérarchie | total |
|---|---|---|---|
| ρ=2 | 42 | × 2.00 | **84** |
| ρ=4 | 72 | × 1.33 | 96 |
| ρ=8 | 134 | × 1.14 | 153 |
| aujourd'hui, uniforme | 25.3 tentées + 42.5 boîtes | | **68** |
| aujourd'hui, cas dur | 60.7 tentées + 135.2 boîtes | | **196** |

Donc : **une perte sur les nuages faciles, et environ 2.3× sur le cas dur.** Pas le 3 à 4× qu'on
pouvait espérer, mais du bon côté là où ça coûte — et avec un profil mémoire bien meilleur (des
tableaux contigus au lieu du pointer-chasing d'un arbre), ce que le compte en unités ne dit pas.

**Ce qui n'est pas réglé, et qui décide.** La colonne « manques » compte, sur les vrais voisins de
Laguerre, ceux que la règle ne proposerait pas : 0 à 4 sur ~600 000 relations, uniquement sur les
nuages faciles, exactement 0 sur le cas dur. Ce n'est pas du bruit, c'est le cas prévu — une
cellule grossière **avalée tout entière** par un enclos ne borde rien, donc n'apparaît dans aucun
`cid`. Quatre manques sur six cent mille suffisent à faire d'un accélérateur *prouvé complet* un
accélérateur *presque toujours juste*, et c'est exactement ce que le BSP + majorant apportait. Tant
que ce cas n'est pas traité — un test de cellule avalée à la construction, ou un paquet élargi — les
paquets ne remplacent pas le BSP, ils le doublent.

Le code reste (`AaBspPre.h`, `--tree pre`, `--enclos`), gaté hors du chemin chaud : c'est la preuve
de la mesure, et l'idée reviendra sous la forme que la première itération ne permettait pas — le
réamorçage d'une itération de Newton à la suivante.

![principe](cases/sub1d_principe.png)

Le panneau bas-droit est le cas dur en miniature : les cellules avancent régulièrement (elles ont
toutes la même largeur) pendant que les germes restent groupés en quatre amas — une cellule est
**loin de son germe**, exactement ce qui casse le raccourci de tout accélérateur.

## LE CRITERE DE L'ENVELOPPE SUR UNE GRILLE

Un troisième index candidat, et d'une autre nature que les deux précédents : il ne borne pas les
germes ni les cellules, il borne **ψ sur une région de l'espace**. Sur une boîte `B`, `min_B ψ` se
calcule exactement (`min_i min_B h_i`) et

```
M(B) := minᵢ max_B hᵢ     majore     max_B ψ
```

puisque `ψ ≤ hᵢ` partout. D'où, pour un dirac `i` :

```
min_B hᵢ > M(B)   ⟹   C(i) ∩ B = ∅
```

C'est le critère que la grille actuelle **n'a pas** — elle s'arrête sur un rayon, d'où ses 4546
boîtes par cellule sur le cas dur. Et il est *statique* : `M` se calcule une fois pour tous les
diracs, là où le test du BSP dépend de la cellule en cours.

`--psigrid` le mesure. `M(B)` y est pris comme `max_B h_r` avec `r` le dirac qui gagne au centre de
`B` (obtenu en rastérisant les vraies cellules) : c'est un `min` sur un singleton, donc valide.
Puis, boîte par boîte, on compte les diracs retenus, on inverse, et les candidats de chaque dirac
sont l'union des listes des boîtes où il a été retenu. La colonne « manques » vérifie que ses vrais
voisins y sont tous.

### La version scalaire perd, et on sait pourquoi

n=1e5, candidats par cellule :

| | g=32 | g=64 | g=128 | g=256 | le BSP |
|---|---|---|---|---|---|
| uniforme | 1474 | 398 | 124 | **45** | 25.3 + 42.5 = 68 |
| lignes / Voronoï | 12050 | 3591 | 661 | 244 | 68 |
| lignes / aires égales | 23230 | 14279 | 8796 | 4462 | 60.7 + 135.2 = 196 |

Zéro manque partout : le critère est bien une implication. Il **gagne** sur l'uniforme (45 contre
68) et perd d'un facteur 23 sur le cas dur — où raffiner ne sert presque à rien, les candidats ne
tombant que de 8796 à 4462 quand les boîtes quadruplent.

C'est diagnostiquable : le nombre de retenus décroît en `1/h` et non en `1/h²`. Le mou du critère
vaut au moins **l'oscillation de ψ sur la boîte**, `M(B)` devant majorer `max_B ψ` alors qu'on le
compare à un minimum ; et `osc_B(ψ) ≈ 2·rayon·h`, avec un rayon de 0.127 ici contre 0.003 sur
l'uniforme. Le même coupable que partout sur ce nuage : la cellule est loin de son germe, donc ψ a
une grande pente.

### Garder le dirac plutôt que le scalaire

Le mou vient de ce qu'on a résumé `h_r` par son maximum. En gardant **le propriétaire `r`** et non
le scalaire, le critère devient `min_B (hᵢ − h_r) > 0` — valide pour la même raison (`ψ ≤ h_r`), au
même coût, mais `hᵢ − h_r` est **affine** : le `|x|²` s'en va, son minimum sur la boîte est exact,
à un coin. Tout le terme en `osc(ψ)` disparaît ; il ne reste que « `i` bat-il le propriétaire
quelque part dans la boîte ».

| n=1e5, g=256 | retenus / boîte | candidats / cellule | le BSP |
|---|---|---|---|
| uniforme (1.5 diracs/boîte) | 10 | **28** | 68 |
| lignes / Voronoï | 1 | 197 | 68 |
| lignes / aires égales | 18 | **101** | 196 |

Sur le cas dur : **4462 → 101**, un facteur 44 au même coût arithmétique. Zéro manque, à toutes les
résolutions et sur les trois nuages.

Et une observation qui vaut d'être notée : la grille régulière marche *mieux* sur le cas à aires
égales que sur le Voronoï groupé. Ce n'est pas un paradoxe — le critère porte sur les **cellules**,
pas sur les germes, et le cas à aires égales a justement des cellules toutes de même taille. C'est
le Voronoï groupé, où les aires couvrent quatre ordres de grandeur, qui ne rentre dans aucune
grille.

### Ce qui n'est pas réglé

Construire les listes demande, par boîte, d'énumérer les diracs qui passent le test — donc une
structure de recherche, c'est-à-dire le BSP. À g=256 pour n=1e5 il y a 0.66 boîte par dirac : on
échangerait « une marche par cellule contre une cellule qui rétrécit » contre « 0.66 marche par
cellule à seuil FIXE, plus un balayage de 101 candidats ». C'est peut-être un gain, ce n'est pas
une évidence, et l'énoncé initial — inonder à partir de la boîte du germe — ne marche pas : rien ne
garantit que la boîte de `p_i` soit retenue, et sur le cas dur c'est justement un autre dirac qui
gagne en `p_i`.

### Deux défauts trouvés, et la même leçon deux fois

Le critère bissectrice ratait d'abord 8700 voisins sur l'uniforme à g=256, avec des arêtes
manquées de 7.5e-3 — plus longues qu'une cellule entière — alors que la démonstration interdit tout
manque. La cause : `|pᵢ|² − |p_r|²` écrit littéralement ne rend **pas zéro** quand `i = r`
(`fl(px²+py²) − px² − py²` vaut ~1e-19, parfois positif), donc le propriétaire de la boîte
s'excluait de sa propre liste. Écrit `dx·(px+rx) + dy·(py+ry)`, c'est exact.

C'est mot pour mot la leçon de `Cell::cut` quelques sections plus haut : **la forme factorisée est
exacte là où la forme développée ne l'est pas**, et dans les deux cas le symptôme était petit,
silencieux, et n'apparaissait que sur une configuration dégénérée. Le diagnostic qui les a trouvés
est le même : ne pas se fier à une somme globale, mais vérifier germe par germe contre une
définition indépendante.

## LA PARABOLE ABAISSÉE : un enclos par PAQUET, avec certificat

`cases/sub1d_baisse.py` (figures en sombre : `baisse1d_principe.png`,
`baisse1d_chevauchement.png`, `baisse1d_gain.png`). C'est la version **démontrable** de la section
« LES PAQUETS » ci-dessus, qui elle laissait passer quelques manques.

On prend `|S| = n / ρ` germes, on rattache chaque germe `m` au `k` de `S` le plus proche — c'est le
paquet de `k` — et on **abaisse la parabole de `k`** (son sommet reste sur `p_k`, on n'augmente que
`w_k`) juste assez pour qu'elle minore tout son paquet. L'abaissement se calcule exactement, sans
majoration, parce que le terme `x²` est commun aux deux paraboles :

    h_k( x ) − h_m( x ) = −2 ( p_k − p_m ) x + ( p_k² − w_k ) − ( p_m² − w_m )

est **affine**, donc son maximum sur une boîte est atteint à un coin. C'est mot pour mot la même
structure que le critère bissectrice de la section précédente, et pour la même raison.

L'intervalle du paquet est alors `I_k = { x : h_k − δ_k ≤ h_j pour tout j de S }`, et il contient
toutes les cellules du paquet :

    x ∈ Lag( m ) ⟹ h_m( x ) ≤ h_j( x ) ∀ j ∈ S ⟹ h_k( x ) − δ_k ≤ h_j( x ) ∀ j ∈ S ⟹ x ∈ I_k

Deux germes ne peuvent se couper que si leurs cellules se touchent, donc que si les deux
intervalles se chevauchent. La liste de candidats est l'union des paquets qui chevauchent : c'est
un **certificat**, pas une heuristique, et le script l'assène à chaque tirage.

Le réglage tient entièrement dans le domaine `B_m` sur lequel on exige `h_k − δ_k ≤ h_m` :

| `n = 2048`, médianes | ρ=2 | ρ=4 | ρ=8 | ρ=16 | ρ=32 | ρ=64 |
|---|---|---|---|---|---|---|
| débordement, `B_m = E_m` | 1.3 | 1.5 | 1.6 | 1.6 | 1.5 | 1.6 |
| candidats / germe, `B_m = E_m` | 6 | 13 | 25 | 49 | 97 | 192 |
| débordement, `B_m = dom` | 27 | 32 | 27 | 20 | 14 | 10 |
| candidats / germe, `B_m = dom` | 124 | 250 | 390 | 590 | 827 | 1120 |

Deux choses, et la seconde est celle qui compte.

**Il faut `B_m = E_m`** — l'enclos de `m` contre `S` seul, celui de la section « L'ENCLOS PAR
SOUS-ÉCHANTILLON ». Pris sur le domaine entier, l'abaissement paie la variation de la différence
affine sur toute sa longueur et l'intervalle déborde d'un facteur 10 à 30 : inutilisable. Avec
`E_m`, il ne déborde plus que de **1.3 à 1.6 fois** l'étendue vraie du paquet. C'est serré, et ça
le reste à tous les taux et dans les deux régimes de poids.

**Et pourtant les candidats valent exactement `3ρ`.** Sur la figure de gain la courbe mesurée se
pose sur la droite `3ρ` sans s'en écarter. Ce n'est pas une coïncidence, c'est un **plancher** :
`I_k` doit contenir l'étendue de son paquet, les paquets voisins la touchent par construction,
donc au moins trois paquets se chevauchent toujours — soit `3ρ` germes. La matrice de
chevauchement le dit autrement : `3.1` paquets retenus par paquet contre `2.9` réellement adjacents.
**Le critère est à 7 % de la perfection, et c'est le regroupement lui-même qui coûte.**

C'est la même leçon que l'enclos en 2D, sous un autre angle : le mou n'est pas là où on le
cherchait. Améliorer `δ_k` ne rapportera rien — il n'y a que 7 % à prendre. Le seul levier est `ρ`,
et il pousse vers `ρ = 1`, c'est-à-dire vers pas de paquets du tout. Un index par paquets ne peut
gagner que si le balayage de `3ρ` candidats est *tellement* plus régulier qu'une descente d'arbre
que le facteur `ρ` se paie en bande passante — ce qui est exactement le pari que le chiffrage de la
section « LES PAQUETS » a trouvé perdant en 2D (84 contre 68 sur l'uniforme, à ρ = 2).

### La même chose en 2D : `--baisse`

`I_k` est la cellule de `k` dans le diagramme de `S` **où `k` seul porte le poids `w_k + δ_k`** —
donc `|S|` convexes qui se recouvrent. `C_S(k)` est un vrai diagramme de puissance --
ses cellules pavent ; `I_k` n'en est plus un. Le maximum de la différence affine se lit
sur les **sommets de `E_m`**, que la passe contre `S` seul fournit déjà. Trois précautions :

- `|p_r|² − |p_m|²` écrit **factorisé** `dx(rx+mx) + dy(ry+my)`, sinon il ne rend pas zéro quand
  `r = m` et le paquet s'abaisse d'un cran pour rien — le défaut de la section précédente ;
- le test d'axe séparateur avec une **tolérance** : deux paquets adjacents se touchent le long
  d'une arête, donc s'intersectent en mesure nulle. Strict, il « manquait » 3754 couples à ρ = 2 ;
- `OneSeed`, un accélérateur d'un seul germe, plutôt que de paramétrer `make_cell` — mesuré, la
  factorisation coûtait 3 % sur le chemin chaud.

`n = 10⁵`, rattachement au germe de `S` le plus proche, **zéro manque partout** :

| ρ | paquets retenus / vrais | aire `I_k` / paquet | candidats / germe | le BSP |
|---|---|---|---|---|
| uniforme, ρ=2 | 13.3 / 6.6 | 2.37 | **31** | 68 |
| uniforme, ρ=8 | 20.1 / 6.9 | 4.68 | 167 | 68 |
| Voronoï, ρ=2 | 19.1 / 6.6 | 2.37 | **40** | 68 |
| Voronoï, ρ=8 | 45.5 / 6.9 | 5.36 | 294 | 68 |
| aires égales, ρ=2 | 17.7 / 6.7 | 2.52 | **42** | 196 |
| aires égales, ρ=8 | 33.4 / 7.3 | 7.50 | 281 | 196 |

**La colonne « vrais » vaut 6.9 quel que soit ρ**, et ce n'est pas un hasard : le graphe
d'adjacence des paquets est une contraction du graphe de Delaunay, donc planaire, donc de degré
moyen inférieur à 6. Le plancher est ainsi `6.9 ρ` — soit 14 candidats à ρ = 2. On en mesure 31.

C'est la différence de fond avec la 1D, où le critère était à 7 % de son plancher (3.1 contre 2.9).
En 2D il retient **2 fois trop à ρ = 2, et près de 4 fois trop à ρ = 64** : un paquet est ici un
patatoïde, son enclos convexe déborde de 2.4 fois son aire, et ce débordement touche des paquets
qui ne le touchent pas. La 1D ne pouvait pas le montrer — un intervalle n'a que deux voisins.

En cadence : `ρ = 2` gagne sur les trois nuages (31 / 40 / 42 contre 68 / 68 / 196), `ρ = 4` ne
gagne plus que sur le cas dur, `ρ = 8` perd partout. Avec le modèle de coût de la section « LES
PAQUETS » — `( candidats + 6 ) × ( 1 + 1/(ρ−1) )`, qui paie le sous-diagramme récursivement — on
obtient 74 / 92 / 96 contre 68 / 68 / 196 : **à peu près nul sur les cas faciles, 2× sur le cas
dur**. Et c'est un certificat, ce que la version « cid » de la section précédente n'était pas.

Deux réserves de plus. La queue : sur le nuage groupé le décile 9 vaut 60 mais le maximum 2620, un
facteur 65 — un paquet dont l'enclos couvre le domaine, donc un problème d'équilibrage avant d'être
un problème de coût moyen. Et la version bon marché du chevauchement (boîtes englobantes seules)
coûte +29 % à ρ = 2, mais 2.1× à ρ = 16 : elle n'est acceptable qu'aux petits taux.

**Le rattachement, lui, est déjà le bon.** J'ai mesuré la variante qui semblait plus naturelle —
rattacher `m` au propriétaire de `p_m` dans le diagramme grossier plutôt qu'au germe le plus
proche.
Sur les nuages sans poids les deux règles coïncident (à poids nuls, `argmin_j h_j( p_m )` **est**
le plus proche). Sur le nuage à poids elle s'effondre : 22 436 candidats contre 42, et une aire
`I_k / paquet` de 465. La raison est dans la formule : la pente de `h_k − h_m` vaut `2 | p_k − p_m |`,
une quantité de l'espace des **germes**. Le propriétaire de `p_m` peut être très loin dans cet
espace-là — c'est exactement ce que « aires égales » fabrique — et `δ` explose. Prendre le plus
proche minimise directement la pente.

### Pourquoi ça déborde autant : `I_k` est un DISQUE de rayon `√δ_k`

`cases/baisse2d.py` met les deux diagrammes côte à côte (`baisse2d_uniforme.png`,
`baisse2d_groupe.png`). Le quatrième panneau donne le mécanisme, et il se démontre.

`I_k` est la cellule de `p_k` portant le poids `δ_k` contre des germes qui n'en portent pas. Sa
frontière dans la direction d'un germe `j` situé à `d_j` est à

    d_j / 2 + δ_k / ( 2 d_j )   ≥   √δ_k        (égalité en `d_j = √δ_k`)

Donc `I_k` **contient toujours le disque de rayon `√δ_k`**, et dès que les germes sont denses dans
toutes les directions il lui est à peu près égal. C'est exactement ce qu'on voit : un polygone
quasi circulaire, bien plus gros que `C_S(k)` et sans aucun rapport avec la forme du paquet.

Voilà la réponse à « pourquoi ça déborde ». **`I_k` ne retient du paquet qu'un seul scalaire.** Le
paquet est un patatoïde de ρ cellules ; on le remplace par un disque dont le rayon est fixé par le
PIRE de ses membres — et par la racine carrée, si bien qu'un `δ` quatre fois trop grand ne coûte
qu'un facteur deux en rayon, mais qu'un `δ` juste ne rapporte pas grand-chose non plus. En 1D
l'objet était un intervalle, déterminé par deux nombres et n'ayant que deux voisins : le défaut ne
pouvait pas s'y voir.

Mesuré (`n = 1200`, ρ = 12) : chaque `I_k` en chevauche **17.7** alors que **6.2** paquets
seulement lui sont adjacents. Le recouvrement du domaine — combien de `I_k` contiennent un point
donné — a pour médiane 5 sur l'uniforme et 3 sur le nuage groupé, là où un diagramme de puissance
en donnerait 1 partout par définition.

### Où est le mou : un `δ` ORACLE le partage en deux

`δ_k` est le max sur `E_m`, alors que la démonstration n'en a besoin que sur `Lag( m )`. Un `δ`
calculé sur la vraie cellule reste donc **valide** (la colonne « manq. » reste à zéro) mais n'est
pas calculable sans la réponse. Il sépare la question en deux : ce que coûte `E_m`, et ce que coûte
la géométrie.

| candidats / germe, `n = 10⁵` | ρ=2 | ρ=8 | ρ=64 | plancher `6.9 ρ` |
|---|---|---|---|---|
| uniforme, `δ` sur `E_m` | 31 | 167 | 1708 | 14 / 55 / 441 |
| uniforme, `δ` ORACLE | 29 | 120 | 967 | |
| aires égales, `δ` sur `E_m` | 42 | 281 | 2751 | 14 / 55 / 441 |
| aires égales, `δ` ORACLE | 38 | 214 | 1961 | |

`E_m` coûte **1.3 à 1.4×**, la géométrie **2.2× (uniforme) à 3.7× (cas dur)**. Resserrer `E_m` ne
rapporterait donc qu'un tiers, et c'est le plafond de tout raffinement de `δ` : le reste est dans
le fait de résumer un paquet par un disque. Si quelque chose doit changer, c'est ça — plusieurs
représentants par paquet, ou une enveloppe anisotrope — pas la façon de calculer `δ`.

### Comment `S` est choisi

Dans le banc, un germe sur ρ dans la **permutation du BSP** (`AaBspPre::build` : `k % r ? out : in`
sur `full.order`). Le BSP coupe aux médianes, donc un sous-arbre est une plage contiguë de `order`
et un pas de ρ prend un germe par bloc : c'est stratifié, et ça ne coûte rien de plus que l'arbre
qu'on construit déjà. Dans les illustrations Python, faute de BSP sous la main, c'est un code de
**Morton** puis `[::ρ]`. **Hilbert n'a pas été essayé.** Ce qui est mesuré, c'est stratifié contre
aléatoire, en 1D : 1.6× sur la médiane et 2 à 3× sur la queue.

Dans `AaBspPack`, la question ne se pose plus : le paquet **est** une feuille du BSP, et le
représentant est le germe le plus proche du centre de sa boîte.

### `--tree pack` : la fin de l'algorithme, chronométrée

`src/AaBspPack.h`. Plus de descente depuis la racine : une **liste plate précalculée** par paquet,
parcourue du plus proche au plus lointain. La préparation (représentants, `δ`, enclos,
chevauchements) est dans `build`, donc comptée dans « arbre XX ms » et **hors du chrono**.

Trois choses que la première version faisait mal, et qui ont chacune été mesurées :

- **La taille du paquet n'est pas la granularité d'élagage.** Un paquet est un **nœud** du BSP
  (`--pack-rate` = ρ) et on **descend dans son sous-arbre** au lieu de balayer ses ρ germes
  (`--leaf` reste 10). Sans ça, `pack` tentait déjà 63 coupes contre 58 au BSP à ρ = 8. Gain
  mesuré : 11 % à ρ = 16.
- **Un rayon de sécurité termine la liste.** Elle est triée par distance **boîte à boîte**, qui
  minore `dist( p₀, boîte_v )` puisque `p₀` est dans sa propre boîte : un seul `break` suffit.
  Mesure appariée : **−17 à −20 %** sur le nuage groupé, neutre sur l'uniforme, **+4 %** sur le cas
  dur (où `wmax − w₀` rend la borne lâche). Gardé.
- **La marge de tangence**, ci-dessous.

`n = 10⁵`, un fil, `--maxnv 64`, même binaire — la disposition du code déplace la référence de 10 %
d'une compilation à l'autre, donc rien ne se compare entre deux tableaux :

| ρ | uniforme | Voronoï | aires égales | liste / cellule | index | préparation |
|---|---|---|---|---|---|---|
| `bsp` | 0.165 | 0.167 | 0.534 | — | — | 16–33 ms |
| 8 | 0.122 (**1.35×**) | 0.130 (1.28×) | 0.265 (**2.01×**) | 16 / 41 / 28 | 1.6–3.2 Mo | 153–395 ms |
| 12 | 0.144 (1.15×) | 0.151 (1.11×) | 0.350 (**1.53×**) | 18 / 48 / 31 | 1.2–2.3 Mo | 129–275 ms |
| 16 | 0.147 (1.12×) | 0.153 (1.09×) | 0.365 (1.46×) | 19 / 49 / 32 | 1.1–2.0 Mo | 133–237 ms |
| 32 | 0.163 (1.01×) | 0.165 (1.01×) | 0.435 (1.23×) | 16 / 48 / 33 | 0.7–1.2 Mo | 86–160 ms |

Les 24 configurations somment à `1.000000000`, et `--cross` sur le cas dur ne trouve **aucun** germe
au-delà de 1e-15 contre le BSP.

**Le gain s'effondre avec ρ, et la colonne « liste » dit pourquoi : elle ne bouge pas.** 16 à 49
paquets candidats par cellule quel que soit ρ, contre **6.9 réellement adjacents**. Le coût par
cellule est donc celui du parcours de liste et non de ρ — et c'est le débordement des `I_k` (le
disque de rayon `√δ`) qui le fixe. Tant qu'il n'est pas réduit, ρ ≥ 10 plafonne autour de **1.15×
sur les cas faciles et 1.5× sur le cas dur**.

**L'amortissement**, à ρ = 12, en passes de calcul du diagramme : le cas dur rembourse en **1.3
passe** (244 ms d'index contre 184 ms gagnés par passe), l'uniforme en **4.6**, le Voronoï groupé en
**16**. La mémoire n'est pas la contrainte : 12 à 23 octets par germe, et elle décroît en `1/ρ`.

Ce qui l'est : **les poids changent à chaque itération de Newton**, donc `δ` et les `I_k` aussi, et
l'index doit être refait. Le rendre valide sur toute une plage de poids est exactement le
*shielding* de Schmitzer — ce n'est pas fait ici, et c'est ce qui décide si l'amortissement existe.

Le mécanisme du gain, dans `--stats` (ρ = 4), n'est pas le nombre de coupes :

| | boîtes testées | coupes tentées |
|---|---|---|
| uniforme, `bsp` | 51.1 | 17.1 |
| uniforme, `pack` | **15.0** | 14.7 |
| aires égales, `bsp` | 155.4 | 58.0 |
| aires égales, `pack` | **21.7** | 63.4 |

`pack` tente autant voire plus de coupes — la liste est un sur-ensemble moins fin que l'arbre — mais
teste **3.4× à 7× moins de boîtes**. Ce que la descente dépensait, c'était le chemin.

### `--tree hull` : la sur-cellule ANISOTROPE, et elle règle le plafond

`src/AaBspHull.h`. Même squelette que `pack`, mais la sur-cellule n'est plus une paraboloïde
abaissée : c'est l'**enveloppe convexe des `E_m` du paquet**. Comme `E_m ⊇ Lag(m)` par construction
(`ψ_S ≥ ψ`), l'enveloppe contient toutes les cellules du paquet — et **il ne reste ni `δ`, ni
paraboloïde, ni test d'axe séparateur**. L'idée est plus simple que celle qu'elle remplace.

L'enveloppe exacte serait chère à intersecter. On garde sa **fonction d'appui dans K = 8 directions
fixes** : le k-DOP contient l'enveloppe, donc travailler dessus ne fait qu'agrandir — le certificat
tient — et le test de chevauchement devient huit comparaisons,
`sup_i(u) + sup_{i+K/2}(v) < 0 ⟹ disjoints`. Enfin `init_cell` fait **partir la cellule de la
sur-cellule** au lieu du carré unité : même ensemble final (`H ∩ tous les demi-plans = Lag` puisque
`Lag ⊂ H`), mais le test d'éviction mord dès la première boîte.

`n = 10⁵`, un fil, même binaire :

| ρ | uniforme | Voronoï | aires égales |
|---|---|---|---|
| `bsp` | 0.172 | 0.172 | 0.531 |
| `pack` ρ=12 (paraboloïde) | 0.145 (1.19×) | 0.153 (1.13×) | 0.351 (1.51×) |
| `hull` ρ=8 | 0.122 (**1.41×**) | 0.122 (**1.41×**) | 0.235 (**2.26×**) |
| `hull` ρ=12 | 0.140 (1.23×) | 0.144 (1.19×) | 0.313 (**1.70×**) |
| `hull` ρ=16 | 0.144 (1.19×) | 0.146 (1.18×) | 0.330 (1.61×) |
| `hull` ρ=32 | 0.157 (1.10×) | 0.161 (1.07×) | 0.403 (1.32×) |

À huit fils, ρ=12 : 1.22× / 1.21× / **1.74×**. `--cross` ne trouve **aucun** germe au-delà de 1e-15
contre le BSP, à ρ = 8, 12 et 16, sur les deux nuages durs.

**Ce que l'anisotropie achète, c'est la longueur des listes** — la grandeur qui plafonnait tout :

| candidats / cellule, ρ=12 | uniforme | Voronoï | aires égales |
|---|---|---|---|
| `pack` (disque de rayon `√δ`) | 18.3 | 47.6 | 31.1 |
| `hull` (k-DOP de l'enveloppe) | **13.5** | **14.8** | **25.5** |
| *réellement adjacents* | *6.9* | *6.9* | *6.9* |

Le Voronoï groupé passe de 47.6 à 14.8 : c'est exactement le cas que le disque ne pouvait pas
décrire, des cellules très allongées le long des lignes. Il reste 2× à 3.7× au-dessus du vrai, dont
une part vient du k-DOP (qui contient l'enveloppe) et une part de `E_m` (qui contient `Lag(m)`).

**Et la préparation tombe de 1.5 à 3×** — il n'y a plus de `δ`, plus de diagramme `I_u`, plus de
SAT. Mais il faut la compter honnêtement, ce que je n'avais pas fait : `build` parallélisait sur
tous les cœurs alors que la passe chronométrée tournait à un fil, ce qui donnait un amortissement
en « 0.5 passe » qui n'existe pas. La préparation suit maintenant `--threads`.

    hull: preparation 216 ms = arbre 34 + paquets/sous-arbre 2
                             | enclos E_m 144 + sur-cellules 5 + listes 32   (POIDS-DEPENDANT 83%)

ρ = 12, `n = 10⁵`, préparation et passe au MÊME nombre de fils :

| | préparation | dont poids-dépendant | gain par passe | amorti en |
|---|---|---|---|---|
| uniforme, 1 fil | 216 ms | 83 % | 32 ms | **6.8 passes** |
| Voronoï, 1 fil | 228 ms | 92 % | 31 ms | **7.4 passes** |
| aires égales, 1 fil | 522 ms | 94 % | 211 ms | **2.5 passes** |
| uniforme, 8 fils | 70 ms | 71 % | 3 ms | 23 passes |
| aires égales, 8 fils | 192 ms | 83 % | 35 ms | 5.5 passes |

**Deux conséquences, et elles changent la conclusion.**

D'abord, la part mutualisable — l'arbre et le découpage en paquets, qui ne dépendent que des
positions — ne pèse que **6 à 29 %**. Partager la recherche de voisins d'une itération de Newton à
l'autre ne retire donc presque rien.

Ensuite, ce qui coûte est `E_m` : **63 à 67 % de toute la préparation**, et c'est une cellule de
puissance par germe contre `S`. Autrement dit *l'index contient une passe de diagramme presque
complète* — 144 ms contre les 179 ms de la passe qu'il accélère. On ne peut pas l'optimiser
« un peu » : il faudrait ne pas la faire.

Dans une boucle de Newton où l'index est refait à chaque itération, le bilan est donc **une perte
d'un facteur 2** (216 + 147 contre 179 ms). Le schéma ne gagne que si l'index **survit à plusieurs
itérations**. Comme n'importe quel sur-ensemble convexe de `Lag(m)` convient, la voie est de
réutiliser les cellules de l'itération précédente en bornant la variation des poids — c'est
exactement le *shielding* de Schmitzer, et c'est ce qu'il faudrait mesurer ensuite.

Partir de la sur-cellule plutôt que du domaine (`--no-hull-init` pour l'inverse) vaut **4 %** :
0.140 contre 0.146, 0.144 contre 0.149, 0.313 contre 0.327.

Le prix est la mémoire : 3.9 à 5.3 Mo à ρ=12 (39 à 53 octets par germe) contre 1.2 à 2.3 pour
`pack`, parce qu'on range les huit appuis ET le polygone de la sur-cellule. Les deux se mettraient
en `float` sans rien changer au résultat — l'index n'est qu'un sur-ensemble.

**`K = 16`, essayé et rejeté.** Les listes raccourcissent bien (25.5 → 19.7 sur le cas dur) mais la
sur-cellule passe à 16 sommets, donc chaque `cut` en balaie deux fois plus au départ : 1.10× contre
1.23× sur l'uniforme, 1.09× contre 1.19× sur le Voronoï, 1.73× contre 1.70× sur le cas dur, et 50 %
de mémoire en plus. Au passage, la première version écrivait les indices des directions axiales en
dur (`0, 2, 4, 6`) : justes à `K = 8`, ce sont des diagonales dès que `K` change. Boîtes fausses,
grille dégénérée, **12 s de préparation au lieu de 90 ms**. Un `static_assert` et `K/2`, `K/4`.

### Le décalage AFFINE : `w` et la position

Abaisser `w` seul revient à choisir `b` dans `h_q( x ) − ω = |x|² + a·x + b` (avec `q = −a/2`,
`ω = |q|² − b`) en laissant `a = −2p_k` imposé par le représentant. Ça ne corrige donc pas la
**pente** de `h_k − h_m`, qui vaut `2|p_k − p_m|` — précisément le terme identifié plus haut.
Déplacer aussi la graine, c'est choisir `a` : chercher le meilleur **minorant affine** de
`min_m L_m` sur les enclos du paquet, un programme linéaire à trois inconnues et une contrainte par
sommet d'enclos.

Mesuré (`cases/baisse2d.py --affine`, LP par `linprog`) :

| `n = 1200`, uniforme | aire `I_k`/paquet | paquets chevauchés |
|---|---|---|
| `q` = le plus proche de `S` | 4.80 | 17.7 |
| `q` = barycentre du paquet | 4.02 | 15.2 |
| `( a, b )` optimal (LP) | 3.47 | **12.9** |
| *vraiment adjacents* | | *6.2* |

Le barycentre prend la moitié du gain pour rien du tout ; le LP la totalité, soit **27 %**. Mais
même optimal, il en reste **2.1×** au-dessus de l'adjacence réelle. C'est cohérent avec le
diagnostic : l'objet reste une **paraboloïde**, donc `I_k` reste un disque, et le disque ne sait
rien de la forme du paquet. Le décalage affine corrige l'inclinaison, pas la nature. C'est pour ça
que `AaBspPack` prend le barycentre et pas le LP : le tiers de gain restant ne vaut pas un
programme linéaire par paquet.

### La tangence, une troisième fois

Sans marge, `pack` à ρ = 2 sur le cas dur donnait **somme 3.86** — 310 voisins réels absents des
listes. Diagnostic, dans cet ordre : `δ` d'abord (zéro violation de `h_q − δ ≤ h_m` sur les sommets
des **vraies** cellules), puis les listes (310 manques), puis l'**écart** des convexes manqués :
`3.5e-18` à `1.5e-11`. De la tangence — deux paquets adjacents se touchent le long d'une arête,
donc s'intersectent en mesure nulle, et l'arrondi les sépare.

Corrigé en **un** endroit plutôt qu'en deux tolérances de comparaison : `δ += 1e-9`, ce qui ne peut
qu'agrandir `I_u` et reste donc un certificat. C'est la troisième fois que le même défaut se
présente, après l'interpolation de `Cell::cut` et le `|p_i|² − |p_r|²` du critère de la grille.

## `--newton` : LE PROBLÈME RÉSOLU, et où va vraiment le temps

Jusqu'ici le banc mesurait **une** construction de diagramme, sur des poids donnés d'avance. Or le
seul régime dans lequel un accélérateur servira est une dizaine de diagrammes enchaînés, sur des
poids qui bougent de moins en moins. `--newton` résout donc vraiment `|Lag_i(w)| = ν_i`, par Newton
amorti (Kitagawa–Mérigot–Thibert) sur le dual de Kantorovich.

* la hessienne est le **laplacien du graphe de Laguerre**, `c_ij = |arête| / (2 |p_i − p_j|)`,
  assemblé depuis les arêtes que `Cell::cid` porte déjà ;
* son noyau est exactement les constantes (ajouter la même constante à tous les poids ne change
  aucune cellule), donc on **fixe `w_0 = 0`** et on raye la ligne et la colonne 0 ;
* on part de `w = 0`, donc du diagramme de **Voronoï**, dont aucune cellule n'est vide : le départ
  est toujours admissible ;
* `refresh_weights` refait les majorants sans reconstruire l'arbre — dans une boucle de Newton les
  positions ne bougent pas.

**Vérification.** Les poids obtenus depuis `w = 0` collent à ceux du fichier `_equal`, calculés par
L-BFGS dans `gen_cases.py` par un tout autre chemin, à **4e-15 près sur une amplitude de 0.128**.

### Deux défauts trouvés par la trace, et c'est pour ça qu'elle existe

**Un pas nul accepté.** Le test d'amortissement `|r(w+tδ)| ≤ (1 − t/2)·|r(w)|` devient
`|r| ≤ |r|` quand `t → 0` : **vrai par égalité**. Arrivé au plancher numérique, la boucle acceptait
`t = 2⁻⁵⁴`, ne bougeait pas, et repartait — à 54 diagrammes et 4 s par itération, indéfiniment.
Corrigé par une décroissance **stricte** plus un plancher sur `t` ; la boucle sort maintenant en
`STAGNATION` en annonçant le résidu atteint.

**Un diagramme sur deux inutile.** La boucle recalculait l'itéré accepté au début de l'itération
suivante, juste pour en tirer les arêtes. Le pas essayé livre désormais les aires ET les arêtes :
**17 → 10 diagrammes** sur l'uniforme.

### Le solveur linéaire : AMGCL contre Cholesky

Le solveur linéaire n'est pas l'objet de l'étude et il ne faut pas qu'il le devienne. Le banc prend
donc **AMGCL** s'il est là (en-têtes seuls, `-DAMGCL_NO_BOOST` — c'est `boost::property_tree` qui
manque, pas AMGCL), sinon **Eigen** `SimplicialLDLT`, sinon un gradient conjugué maison.
`omp_set_num_threads` suit `--threads`, sans quoi les deux moitiés du chronomètre ne tourneraient
pas sur le même nombre de cœurs.

| 8 fils, résolution / TOTAL | AMGCL | Cholesky (Eigen) |
|---|---|---|
| uniforme n=1e5 | 0.98 / **1.52 s** | 1.95 / 2.59 s |
| uniforme n=1e6 | 13.2 / **18.5 s** | 84.7 / 90.3 s |
| lignes n=1e5 | 5.8 / 13.4 s | 3.8 / **12.0 s** |

**À 1e6 c'est 4.9× sur le total**, et la raison est structurelle : le `SimplicialLDLT` d'Eigen est
scalaire et séquentiel (84.7 s à 8 fils contre 99.0 à 1 fil : il ne monte pas), alors qu'AMGCL
monte à 7.2× et que son coût reste en `O(n)`. Le gradient conjugué maison, lui, est 5× plus lent
que Cholesky : il est resté comme filet de sécurité, pas comme option.

Mais **sur le nuage de lignes, Cholesky gagne encore**, et c'est instructif : les `c_ij` y
s'étalent sur plusieurs ordres de grandeur, ce qui est exactement l'hypothèse que l'agrégation
lissée fait et que ce nuage viole.

| lignes n=1e5, 8 fils | itérations de CG | hiérarchie | résolution |
|---|---|---|---|
| agrégation lissée + `spai0` | 1462 | 1.95 s | 4.31 s |
| agrégation lissée + Gauss-Seidel | 1245 | 3.33 s | 4.30 s |
| **Ruge-Stüben + Gauss-Seidel** | **524** | 3.56 s | **1.95 s** |

Ruge-Stüben, qui choisit ses nœuds grossiers arête par arête, divise les itérations par trois — et
paie la moitié du gain dans la construction de sa hiérarchie. `--amg-var` les expose toutes les
trois.

### L'assemblage : un tri qui coûtait plus cher que le diagramme

Le laplacien était assemblé en triant les arêtes par `(i, j)` pour apparier les deux mesures d'une
même arête et les moyenner. Défendable — un gradient conjugué veut une matrice vraiment symétrique
— mais à n=1e6 le tri de douze millions d'enregistrements coûtait **4.2 s, soit plus que le
diagramme lui-même** (3.9 s). Et il ne servait à rien : les deux mesures ne diffèrent qu'à 1e-16
relatif, cinq ordres de grandeur sous la tolérance du solveur. En gardant la valeur mesurée par la
ligne, un comptage et une somme préfixe suffisent — et chaque ligne somme alors **exactement** à
zéro, donc les constantes restent exactement dans le noyau. Mesure : **0.36 → 0.04 s** à n=1e5,
**4.2 → 0.67 s** à n=1e6.

### Où va le temps maintenant

| 8 fils | uniforme 1e5 | uniforme 1e6 | lignes 1e5 |
|---|---|---|---|
| itérations de Newton | 7 | 6 | **23** |
| diagrammes | 10 | 11 | **113** (89 reculs) |
| **diagrammes** | 0.42 s (27 %) | 3.65 s (20 %) | 7.62 s (**63 %**) |
| **résolution** | 0.98 s (65 %) | 13.19 s (71 %) | 3.84 s (32 %) |
| assemblage | 0.04 s | 0.67 s | 0.13 s |
| arbre + majorants | 0.07 s | 0.92 s | 0.40 s |
| TOTAL | **1.52 s** | **18.50 s** | **12.03 s** |

Deux régimes, et ils n'appellent pas le même travail :

* sur l'**uniforme**, même avec AMGCL, l'algèbre linéaire pèse les deux tiers. Gagner sur le
  diagramme ne se verrait pas dans le total.
* sur les **lignes**, le diagramme pèse 63 % — c'est là qu'une accélération se mesure. Et les 113
  diagrammes pour 23 itérations n'y sont pas une fatalité : depuis `w = 0` l'amortissement rampe à
  des pas de 1/256 pendant 16 itérations avant 6 itérations quadratiques. C'est le régime où il
  faut du **multi-échelle** (Mérigot) : résoudre sur un sous-échantillon avec les masses cibles
  agrégées, puis prolonger en donnant à chaque germe fin le poids de son représentant — tous les
  germes d'un paquet portant alors le même poids, le diagramme fin est localement un **Voronoï** et
  globalement la solution grossière. Le BSP donne la hiérarchie gratuitement (`build_sel` existe
  déjà), et l'erreur restante est purement locale, donc pleinement dans le domaine de Newton.

### LE MULTI-ÉCHELLE : écrit, mesuré, PAS ABOUTI

Les 113 diagrammes des lignes viennent des 89 reculs de l'amortissement depuis `w = 0`. Le remède
classique (Mérigot) est de partir d'ailleurs : résoudre le même problème sur `n / R` représentants
portant la masse **agrégée** de leur paquet, puis prolonger. `--ms-ratio R` l'active (défaut **1**,
c'est-à-dire désactivé — voir plus bas).

Ce qui est en place :

* **la hiérarchie** vient d'un BSP dédié, à feuilles de `--ms-ratio` germes. Les paquets sont ses
  nœuds, donc des sous-arbres, donc spatialement compacts ;
* **la cible agrégée** : `ν_k = |paquet| / n` ;
* **la prolongation par c-transformée** : `w_i = −ψ_grossier(p_i)`, la parabole du germe fin
  touchant le potentiel grossier en `p_i` ;
* **le rattrapage** des cellules vides, en relevant leur poids jusqu'à `max_j (w_j − |p_i−p_j|²)`.

**Ce qui marche.** Le niveau grossier converge remarquablement : 512 germes, **4 itérations à pas
plein**, sur l'uniforme comme sur les lignes. La machinerie est juste.

**Ce qui casse.** Chaque prolongation produit des cellules **vides**, et Newton amorti n'est défini
que si aucune ne l'est — une cellule vide sort du graphe de Laguerre, le laplacien se disconnecte,
et sa ligne devient une équation sans rapport avec la géométrie. Sur l'uniforme, n=2e4,
`--ms-ratio 2`, cellules vides juste après chaque prolongation puis à chaque passe de rattrapage :

| niveau | 512 | 1024 | 2048 | 4096 | 7712 | 11808 | 20000 |
|---|---|---|---|---|---|---|---|
| après prolongation | — | 1 | 9 | 59 | 166 | 190 | 752 |
| après rattrapage | — | 0 | 0 | 0 | 0 | **19** | **1337** (diverge) |

Le rattrapage converge jusqu'à ~10⁴ germes puis décroche. Résultat net : le multi-échelle est
aujourd'hui **plus cher que Newton depuis `w = 0`** (83 itérations et 548 diagrammes contre 7 et
10 sur l'uniforme). Il est donc **désactivé par défaut**.

**Trois erreurs trouvées en chemin, et elles valent d'être notées :**

1. **Les paquets ne peuvent pas être des tranches à pas fixe de la permutation.** Le BSP coupe à la
   médiane, donc un sous-arbre EST une tranche contiguë — mais la réciproque est fausse : les
   frontières tombent aux médianes (10000, 5000, 2500, 1250, 625, 313…), qui ne sont pas des
   multiples de `R`. Un bloc sur deux enjambait deux sous-arbres, parfois cousins éloignés, et la
   prolongation donnait le même poids à deux germes à l'autre bout du carré.
2. **Les paquets ne peuvent pas descendre sous la feuille du BSP.** Avec `--leaf 10`, aucun paquet
   ne fait moins de dix germes : quel que soit `--ms-ratio`, la **dernière** prolongation était
   toujours un saut d'un facteur dix. D'où le second arbre, à feuilles de `--ms-ratio` germes.
3. **Recopier le poids du représentant est faux dès que le nuage est serré.** Deux germes voisins
   de paquets différents reçoivent alors des poids séparés par un SAUT, et une cellule est vide dès
   que ce saut dépasse le carré de la distance qui les sépare. Sur les lignes, des germes à `1e-5`
   l'un de l'autre recevaient des poids écartés de `1e-3` — `1e8` fois trop. La c-transformée est
   continue et fait tomber le résidu après prolongation de 121 à 22 fois la cible, mais ne suffit
   pas.
4. **Le rattrapage ne doit viser que les cellules VIDES.** Imposer `p_i ∈ Lag_i` à tous les germes
   (la c-concavité) relevait 3229 germes sur 4096 — 79 % — là où le diagramme n'avait aucune
   cellule vide, et l'itération de point fixe n'avait pas convergé après quarante passes.

**Ce qu'il faudrait essayer ensuite** : le relèvement d'une cellule vide en vide d'autres, et c'est
ce cycle qui diverge au niveau fin. Une piste est de ne pas relever à la borne mais de faire une
recherche linéaire sur la prolongation elle-même — `w = t · w_prolongé`, `t` divisé par deux tant
qu'une cellule est vide — puisque `t = 0` (Voronoï) est toujours admissible. Le départ serait alors
au pire aussi bon que `w = 0`, et jamais inadmissible.

### `--memo` : SE SOUVENIR DE L'ITÉRATION PRÉCÉDENTE

Dans une boucle de Newton le même diagramme est reconstruit une dizaine de fois sur des poids qui
bougent de moins en moins, et rien n'en était gardé. Le dessin est repris de
`old_pd/src/cpp/sdot/PrevCutInfo.h` : par germe, une poignée de couples **( feuille, masque )**, un
bit par dirac de la feuille, disant qu'il porte un côté de la cellule finale.

**Le masque sert deux fois, et c'est tout le mécanisme :**

1. **en pré-passe**, avant de descendre l'arbre : on coupe avec les diracs retenus de **toutes** les
   feuilles mémorisées ;
2. **au parcours**, quand la descente atteint une feuille : on applique le **complément** de son
   masque, donc uniquement ce qui n'a pas été fait en pré-passe.

Le second point rend le premier gratuit : aucun dirac n'est proposé deux fois, et il n'y a rien à
chercher dirac par dirac — une lecture de masque par feuille *visitée*.

#### Ça marche, et ça ne rapporte presque rien

| uniforme n=1e5, 1 fil | coupes rejouées | feuilles | **boîtes testées** |
|---|---|---|---|
| sans mémoire | 0 | 0 | **50.3** |
| avec | 5.38 | 2.80 | **48.9** (−2.8 %) |

La cellule a 5.97 côtés : **la pré-passe reconstruit donc le voisinage entier**. Et pourtant les
tests de boîte ne bougent quasiment pas.

**La raison est structurelle.** `may_be_cut` demande « cette boîte peut-elle encore atteindre la
cellule ? ». Une boîte qui contient un **vrai voisin** passe ce test quoi qu'il arrive — son plan
est tangent à la cellule, par définition. Arriver avec la cellule finale ne rend donc pas ses
voisins rejetables : les boîtes qu'il faut visiter sont exactement celles des voisins, plus le
chemin de descente, et c'est déjà l'essentiel des cinquante. **On ne peut pas élaguer ce qu'on doit
de toute façon regarder.**

| ms par diagramme, 8 fils, `--leaf 10`, trois passes | lignes n=1e5 | uniforme n=1e5 |
|---|---|---|
| `bsp` | **63.50** | 35.50 |
| témoin : `--memo --no-memo-bits` | 65.24 | 35.50 |
| avec la mémoire | 66.40 | **34.30** |

Et sur les lignes il y a pire que le temps par diagramme : l'ordre des coupes change les arrondis,
donc le chemin de l'amortissement, et la résolution passe de **113 à 140 diagrammes**. Total
17.3 s contre 14.9 s.

#### Les deux variantes plus faibles, mesurées avant celle-ci

* **un masque limité à la feuille du germe** : +3 %. Il ne peut rien gagner, et c'est démontrable —
  le parcours descend fils-le-plus-proche en premier, donc la première feuille atteinte est celle
  du germe ; ses diracs sont proposés avant le moindre test de boîte extérieure, et une cellule est
  l'intersection de ses demi-plans, donc elle est la même une fois la feuille balayée quel que soit
  l'ordre. Le réordonnancement ne change **aucune** réponse de `may_be_cut`.
* **l'indice du dirac qui a vidé la cellule**, pour recommencer par lui : il se déclenche **une fois
  sur dix mille cellules**. Un pas d'amortissement n'est pas refusé parce que des cellules se
  vident, mais parce que le résidu ne baisse pas assez — le critère de Kitagawa-Mérigot-Thibert
  impose `min a >= eps` précisément pour que l'itéré ne sorte jamais de la région où toutes les
  cellules sont non vides.

#### `Cell::cut` n'est pas idempotente

Une première version rejouait les coupes **sans** appliquer le complément ensuite : le parcours
reproposait les mêmes diracs, chacun était donc coupé deux fois. Le README affirmait qu'un plan
appliqué deux fois donne la même **aire** à 2.2e-16 près depuis que le `lerp` est ancré sur le
sommet dedans. C'est vrai de l'aire, **pas du nombre de sommets** : les deux points d'intersection
créés par une coupe ne sont pas exactement sur son plan, `s` y vaut ±1e-17, et un `+1e-17` suffit à
ce que la seconde application trouve `nb_out == 1` et ajoute un sommet dégénéré. Mesure :
**116 305 coupes débordent 64 sommets**, et Newton stagne à `max|a-nu|/nu = 1.67e+03`. Le
complément est une **condition de correction**, pas une optimisation.

#### Deux notes de méthode

* Quand il n'y a rien à rejouer, `AaBspMemo` **rend la main à `AaBsp` mot pour mot**. Porter une
  copie du parcours, même identique, coûte 3 à 6 % — deux pointeurs de plus à garder vivants dans la
  boucle chaude, cf. la note de `may_be_cut` sur le `this` qui coûtait 7 %. Sans ça la première
  passe et le témoin paient une taxe qui masque l'effet mesuré.
* Les compteurs de diagnostic sont compilés dehors (`AaBspMemo::compte`). Membres non atomiques, ils
  mettent quand même une ligne de cache **partagée** sur le chemin chaud : 7.69 → 8.34 s à huit
  fils, 8 % sur un compteur qui ne sert à rien.

#### Sur les 3× de `old_pd`, ce qu'on peut et ne peut pas dire

Le dessin y est ; l'implémentation, non. `make_prev_cuts` est déclarée dans `PointTree.h` et
`PointTreeWithValues.h` et **définie nulle part**, et `PointTreeWithValues.h` est tronqué en plein
milieu de la classe (y compris dans la version commitée). Le chemin `prev_cuts` de ce dépôt ne
compile pas en l'état : impossible de reproduire ou de disséquer le facteur 3.

Une hypothèse vérifiable en lisant `PowerDiagram.cxx` : dans ce code, le chemin **sans** `prev_cuts`
fait, pour **chaque feuille visitée**, un `std::sort` des diracs de la feuille par distance au
germe. Le chemin `prev_cuts` ne le fait pas. Une bonne part du facteur 3 pourrait donc mesurer
« la mémoire évite un tri par feuille » plutôt que « la mémoire évite du parcours ». Ce banc-ci n'a
pas ce tri — le pré-tri de la première feuille y a été mesuré perdant et retiré — donc il n'a pas
cette marge à récupérer. Et son parcours n'est pas le même : `old_pd` marche de feuille en feuille
(`RemainingBoxes`), pas en redescendant de la racine.

## LE FRONT SUR GRILLE : `--front`

L'objectif : supprimer la marche dans le BSP au profit d'un **étalement** de proche en proche, tant
que la parabole du dirac peut passer sous un majorant de ψ — au sens large.

Le majorant est **affine par tuile**, et il n'y a rien à ajuster : toutes les paraboles partagent
`|x|²`, donc `ψ − |x|²` est un min de fonctions affines, donc **concave**, et un majorant affine
minimal en est un **hyperplan d'appui** — c'est-à-dire la partie affine de la parabole d'un dirac
qui gagne quelque part dans la tuile. *Stocker le propriétaire EST stocker un majorant affine*, et
le test `min_B (h_i − h_r) > 0` est exact, à un coin, parce que la différence est affine.

Trois choses à établir, et `--front` les mesure. n=1e5 :

| | g=64 | g=128 | g=256 | g=512 |
|---|---|---|---|---|
| **uniforme** — tuiles du front / cellule | 5.1 | 5.4 | 6.4 | 9.9 |
| pas de descente | 0.01 | 0.05 | 0.17 | 0.15 |
| sans certificat | 96 % | 84 % | 41 % | 4 % |
| **manques** | 0 | 0 | 41 | 6 |
| **lignes / Voronoï** — front / cellule | 4.8 | 5.3 | 6.0 | 8.5 |
| front **maximum** | 99 | 332 | 1184 | **4508** |
| **manques** | 47 | 16 | 16 | 12 |
| **lignes / aires égales** — front / cellule | 10.9 | 11.4 | 12.8 | 17.1 |
| pas de descente | 7.2 | 14.5 | 29.2 | **58.0** |
| sans certificat | 96 % | 85 % | 48 % | 9.5 % |
| **manques** | 2101 | 1866 | 1323 | 764 |

### Ce qui marche

**Le front est petit** : 5 à 17 tuiles par cellule, contre 42 (uniforme) à 135 (lignes) tests de
boîte pour le BSP. Et le test est *indépendant de la cellule en cours* — plus d'ordre imposé, plus
de « tester à la sortie de pile contre la cellule telle qu'elle est ».

**Les majorants se construisent pour rien** : une requête par tuile, parallèle sans partage, 1 à
255 ms pour toute la grille. À g=256 sur l'uniforme c'est **3 ms** contre 34 ms pour une passe de
diagramme. C'est le seul endroit où le BSP reste nécessaire — il disparaît bien du parcours.

### Ce qui bloque : l'AMORCE, et c'est le même problème qu'avant

Le front n'est valide que depuis une tuile qui **rencontre** `Lag_i`, pas seulement une tuile
retenue : l'ensemble retenu peut avoir plusieurs composantes, et partir de la mauvaise fait manquer
la cellule. Or `Lag_i` est convexe, donc les tuiles qu'elle rencontre forment un ensemble
**connexe** et toutes passent le critère — une bonne amorce suffit à tout couvrir.

Le seul certificat en `O(1)` qu'une tuile rencontre `Lag_i` est **« son propriétaire est `i` »**.
Et il est rare : à g=64 il y a 4096 tuiles pour 100 000 diracs, donc **96 % des diracs ne possèdent
aucune tuile**. Sans certificat on part de là où la descente s'est arrêtée — d'où 12 à 2101 manques.

Pire, sur le cas dur **la descente coûte plus cher que le front** : 7 à 58 pas contre 11 à 17 tuiles.
C'est la maladie de ce nuage, la même partout dans ce banc — la cellule est loin de son germe, donc
la *localiser* est le travail, et c'est exactement ce que la marche dans le BSP faisait.

Et raffiner ne sauve pas : de g=64 à g=512, soit 64 fois plus de tuiles, les manques ne tombent que
de 2101 à 764, pendant que la descente passe de 7 à 58 pas et la construction de 17 à 255 ms.

**ESSAYÉ ET REJETÉ** : suivre la direction de descente `p_i − p_r` (« s'éloigner du concurrent »),
qui est le vrai gradient de la pièce affine courante — une marche de visibilité ordinaire. Elle n'a
pas de critère d'arrêt utilisable : quand le dirac ne possède aucune tuile, elle court jusqu'à sa
borne. Mesure : **480 à 2810 pas** contre 7 à 58 pour le glouton à huit voisins.

### Où ça mène

Le critère et l'étalement sont validés et bon marché. Ce qui manque est une **amorce venue
d'ailleurs**, et il y en a deux sources naturelles :

* **l'itération précédente de Newton** : n'importe quel sommet de la cellule précédente donne une
  tuile, et `propriétaire == i` s'y vérifie en `O(1)`. C'est la mémoire de `--memo` qui, ici,
  paierait — elle ne servait à rien pour réordonner des coupes, elle supprimerait la descente ;
* **un pavage hiérarchique** (quadtree raffiné jusqu'à ce que chaque dirac possède une tuile), qui
  réglerait du même coup le front maximum de 4508 tuiles du Voronoï groupé — la grille régulière ne
  peut pas suivre des aires qui couvrent quatre ordres de grandeur.

### LE PAVAGE PAR UN DIAGRAMME GROSSIER : `--front-rate R`

La grille régulière échouait sur deux points : pas de certificat d'amorce (96 % des diracs ne
possèdent aucune tuile) et un front maximum de 4508 tuiles sur le Voronoï groupé. Les deux
disparaissent si le pavage est le **diagramme de puissance d'un germe sur R**, avec leurs vrais
poids.

**Le majorant devient gratuit.** Sur la cellule grossière `T_k`, le germe `k` est lui-même un vrai
dirac, donc `ψ ≤ h_k` partout. Le majorant affine de `T_k` est la parabole de son propre germe : pas
de requête, pas de rastérisation, rien à stocker que le pavage lui-même.

**Et l'ensemble retenu devient CONNEXE.** Sur `T_k`, `ψ_S` vaut exactement `h_k`, donc le critère
`min_{T_k}(h_i − h_k) ≤ 0` dit exactement « `T_k` rencontre `E_i` », où `E_i = {x : h_i ≤ ψ_S}` est
l'**enclos** de `i` contre `S` seul — une cellule de puissance, donc **convexe**. Les cellules
grossières qu'elle rencontre forment donc un ensemble connexe, qui contient toutes celles que
rencontre `Lag_i` puisque `Lag_i ⊆ E_i`. Le piège de la grille — plusieurs composantes, on part dans
la mauvaise — ne peut plus se produire. Et l'adjacence est donnée : `Cell::cid` la porte déjà.

n=1e5, **zéro manque partout** :

| | ρ=4 | ρ=8 | ρ=16 | ρ=32 | ρ=64 |
|---|---|---|---|---|---|
| **uniforme** — cellules du front | 5.95 | 5.58 | 5.63 | 5.40 | 5.37 |
| front maximum | 14 | 12 | 12 | 12 | 11 |
| candidats | 67.8 | 122.8 | 252.5 | 469.0 | 941.6 |
| **Voronoï** — front | 5.94 | 5.59 | 5.57 | 5.47 | 5.38 |
| front maximum | 20 | 18 | 18 | 17 | 17 |
| candidats | 68.5 | 126.0 | 254.7 | 505.3 | 993.2 |
| **aires égales** — front | 5.99 | 5.73 | 5.70 | 5.57 | 5.49 |
| front maximum | 18 | 15 | 13 | 12 | 12 |
| candidats | 69.5 | 130.0 | 260.7 | 501.5 | 977.5 |
| amorce gratuite | 0.5 % | 1.0 % | 2.0 % | 3.6 % | 6.4 % |
| pas de descente | 17.0 | 12.1 | 8.4 | 5.8 | 4.1 |

Sur l'uniforme et le Voronoï, **l'amorce est gratuite dans 100 % des cas** : la cellule grossière qui
contient `p_i` est déjà retenue, il n'y a rien à chercher.

### Ce que ça vaut, comparé au BSP

À ρ=4, en « unités de travail » (tests de région + coupes tentées) :

| n=1e5 | le BSP | le front sur cellules grossières |
|---|---|---|
| uniforme | 42.5 boîtes + 25.3 coupes = **68** | 6.0 cellules + 67.8 candidats = 74 |
| lignes / Voronoï | 42.2 + 25.8 = **68** | 5.9 + 68.5 = 74 |
| lignes / aires égales | 135.2 + 60.7 = **196** | 6.0 + 69.5 = **76** |

Deux lectures :

* **le nombre de régions à tester tombe de 42-135 à 6**, et le test est *indépendant de la cellule
  en cours* — plus d'ordre imposé, donc quelque chose qui se vectorise ;
* **le total est le même sur les trois nuages**, ~75, là où le BSP passe de 68 à 196. La méthode est
  insensible à la difficulté du nuage, et c'est exactement ce qu'on lui demandait.

Le front est remarquablement stable : **5.4 à 6.0 cellules par dirac**, sur les trois nuages et à
tous les ρ, avec un maximum de 11 à 20. La grille régulière montait à 4508.

### Ce qui reste à payer

* **L'amorce sur le cas dur** : 17 pas à ρ=4, 4 pas à ρ=64. C'est encore une marche, et c'est le
  compromis à régler — grand ρ raccourcit la descente et gonfle les candidats. C'est aussi
  exactement là que la mémoire d'une itération à l'autre paierait : dans une boucle de Newton, la
  cellule grossière trouvée au tour précédent est une amorce gratuite et vérifiable en `O(1)`.
* **Le BSP n'a pas disparu, il a été rétrogradé** : il sert encore à localiser `p_i` dans le
  diagramme grossier, une requête par dirac sur un arbre ρ fois plus petit.
* **Le pavage dépend des poids**, donc il est à refaire à chaque itération de Newton : un diagramme
  de `n/ρ` germes, soit `1/ρ` de passe. À ρ=4 c'est 25 %, à ρ=16 c'est 6 %.

### La tangence, une TROISIÈME fois

Sans marge, 30 arêtes sur 600 000 disparaissaient des listes sur l'uniforme à ρ=4 (434 sur le cas
dur). Le diagnostic, chiffré : **le pire écart d'un manque vaut 1.7e-18**. Quand `i` et `j` sont
tous deux des germes grossiers, leur arête *fine* est portée par leur arête *grossière* — le minimum
de `h_i − h_k` sur la tuile vaut alors exactement zéro, et l'arrondi le rend positif. Une marge de
`1e-12` sur le critère suffit, et elle est sûre : elle ne fait qu'élargir l'ensemble retenu, donc
l'implication reste vraie.

### `--tree front` : LE FRONT CHRONOMÉTRÉ, et il gagne

L'accélérateur complet : l'index porte tout (diagramme grossier, amorce, étalement, inversion,
listes triées), et la passe ne fait plus que **lire une liste et couper** — plus de pile, plus de
boîte, plus de test d'éviction. Vérifié contre le balayage complet sur cinq nuages, écart max
**1.8e-16**.

| n=1e5, `--no-cellbox` | `bsp` | front ρ=2 | ρ=4 | ρ=8 | ρ=16 |
|---|---|---|---|---|---|
| **1 fil** uniforme | 0.155 s | **0.068 (2.3×)** | 0.104 | 0.168 | 0.308 |
| lignes / Voronoï | 0.153 s | **0.067 (2.3×)** | 0.102 | 0.164 | 0.295 |
| lignes / aires égales | 0.417 s | **0.074 (5.6×)** | 0.111 | 0.177 | 0.311 |
| **8 fils** uniforme | 0.021 s | **0.009 (2.3×)** | 0.014 | 0.023 | 0.043 |
| lignes / Voronoï | 0.021 s | **0.009 (2.3×)** | 0.014 | 0.023 | 0.043 |
| lignes / aires égales | 0.070 s | **0.010 (7.0×)** | 0.015 | 0.025 | 0.045 |

Le résultat qui compte n'est pas le facteur, c'est la **colonne** : à ρ=2 le front met 0.067 à
0.074 s sur les trois nuages, là où le BSP passe de 0.153 à 0.417. **La méthode ne voit pas la
difficulté du nuage.** C'est exactement ce qu'on lui demandait — le BSP, lui, paie l'anisotropie et
les cellules loin de leur germe.

`ρ = 2` est l'optimum, et il ne se discute pas : les candidats croissent linéairement avec ρ
(37 → 68 → 126 → 255) alors que le front reste à 5.6-6.4 cellules. Doubler ρ double le travail de
la passe.

### Ce que ça coûte

* **La préparation** : 532 à 1180 ms à un fil, 225 à 516 ms à huit — soit **3.4 à 7.6 passes** à un
  fil, **22 à 50** à huit. Même régime que `--tree hull` : l'index ne se rembourse que s'il survit à
  plusieurs itérations. Mais le gain par passe, lui, est bien plus gros (5.6-7× contre 1.7-2.3×).
* **La mémoire** : 19.1 Mo à ρ=2 pour n=1e5, soit **191 octets par germe** — contre 39 à 53 pour
  `hull`. C'est le prix de la liste matérialisée.

### Trois choses apprises en route, toutes mesurées

**Matérialiser la liste est obligatoire.** Faire l'union et le tri à chaque cellule, depuis le front
et la relation inverse, coûte six fois moins de mémoire — et **4837 ns par germe contre 1600 pour le
BSP**. Le tri de ~144 entrées par cellule coûtait plus que tout le reste.

**Trier par distance est obligatoire.** Sans ordre, le polygone *intermédiaire* enfle : 247 cellules
débordent 64 sommets sur le nuage à aires égales et la somme des aires part à `1.000000086`. Le BSP
obtenait cet ordre gratuitement en descendant fils-le-plus-proche d'abord ; ici il faut le payer,
mais **une fois**.

**Une cellule VIDE n'a pas de front, et sa cellule restait le domaine entier.** Un germe dont la
cellule est vide a un enclos vide — il perd contre `S` partout — donc aucune cellule grossière ne le
retient, donc aucun candidat, donc rien ne le coupe : `--check --weights 1.0` sortait un écart de
**1.000e+00**, une cellule d'aire 1 là où la vraie est vide. Vider une cellule demande jusqu'à trois
demi-plans (en 2D, une intersection vide de demi-plans en a une sous-famille vide d'au plus trois) et
rien ne dit lesquels : pour ces germes-là, rares, on repasse par l'arbre. Exact, et ça ne coûte que
sur eux.

### LE FRONT DANS LA BOUCLE DE NEWTON : 2 à 4× sur le diagramme, et une perte de 21 à 46 % au total

`--newton --tree front` mesure l'affaire de bout en bout. L'index est refait à **chaque** évaluation
de diagramme, y compris les pas refusés de l'amortissement.

| n=1e5 | `bsp` | front ρ=2 |
|---|---|---|
| **1 fil, uniforme** — diagrammes (10) | 1.980 s | **0.807 (2.5×)** |
| index | 0.070 s | 3.227 s |
| TOTAL | **6.233 s** | 8.725 s (+40 %) |
| **1 fil, aires égales** — diagrammes (113) | 41.362 s | **9.668 (4.3×)** |
| index | 0.832 s | 47.900 s |
| TOTAL | **69.519 s** | 84.413 s (+21 %) |
| **8 fils, uniforme** — diagrammes | 0.338 s | **0.161 (2.1×)** |
| TOTAL | **1.461 s** | 2.116 s (+45 %) |
| **8 fils, aires égales** — diagrammes | 6.916 s | **1.916 (3.6×)** |
| index | 0.301 s | 11.989 s |
| TOTAL | **14.695 s** | 21.406 s (+46 %) |

Le diagramme lui-même est **2.1 à 4.3× plus rapide**, sur les deux nuages et aux deux niveaux de
parallélisme. Et le total est **perdant de 21 à 46 %**, uniquement parce que l'index est reconstruit
113 fois. Une reconstruction coûte 0.42 s là où un diagramme en coûte 0.086 : **cinq diagrammes**.

La préparation a été optimisée avant de conclure, sans quoi le chiffre aurait menti :

* les listes de candidats dédoublonnées par un `std::sort` + `unique` par germe, et reçues dans `n`
  petits `vector` : **0.516 s sur 0.83 s d'index**. Une MARQUE par thread (74 tests au lieu de ~460
  comparaisons) et un tampon plat par thread — `Split::blocks` donne à chaque thread une plage
  contiguë, donc les tampons se recollent sans rien trier — ramènent à **0.248 s** ;
* le diagramme grossier était construit en série ; parallélisé de la même façon.

### Ce qui rendrait l'index réutilisable, et c'est exact

L'index n'a pas besoin d'être *serré*, seulement **valide**, et `ψ ≤ h_k` ne dépend ni du pavage ni
des poids. Il reste donc à borner la dérive. Avec `h_i(x) = |x − p_i|² − w_i`, un changement de
poids `w → w + Δ` donne

```
h_i^neuf − h_k^neuf  =  ( h_i^vieux − h_k^vieux ) − Δ_i + Δ_k  ≥  ( h_i^vieux − h_k^vieux ) − 2ε
```

avec `ε = max |Δ|`. Donc **un index construit avec une marge `2ε` reste valide pour tout changement
de poids borné par `ε`** : retenir toute tuile telle que `min_T (h_i^vieux − h_k^vieux) ≤ 2ε`, c'est
retenir un sur-ensemble de ce que le critère neuf retiendrait. Rien à reconstruire — ni le pavage,
ni les fronts, ni les listes — tant que les poids restent dans la boule.

C'est le *shielding* de Schmitzer, écrit dans ce cadre-ci, et c'est la pièce que ce banc identifie
comme manquante depuis `--tree hull`. Ce qu'il reste à mesurer : de combien la marge gonfle les
listes, et combien d'itérations un index survit — sachant que dans une boucle de Newton amortie les
poids bougent de moins en moins, donc `ε` décroît d'itération en itération.

### LE BOUCLIER : exact, et inutilisable — la mesure qui ferme la piste

L'idée est juste et la démonstration tient : `ψ ≤ h_k` ne dépend ni du pavage ni des poids, donc un
index construit avec une marge `2ε` reste **valide** pour toute dérive de poids bornée par `ε`. La
connexité survit aussi — l'ensemble retenu devient `{T : T ∩ E_i^{2ε} ≠ ∅}` avec
`E_i^c = {x : h_i − ψ_S ≤ c}`, sous-niveau d'un **max de fonctions affines**, donc convexe.

Implémenté (`--front-bouclier`, marge adaptative à quatre fois la dérive qui vient de la
déclencher), ça marche exactement comme annoncé : **3 reconstructions pour 9 évaluations, 6
réutilisations**, et `--check` passe.

Et c'est inutilisable :

| n=2e4, ρ=2 | sans bouclier | avec |
|---|---|---|
| front / germe | 6.35 | **970.9** |
| candidats / germe | 37.3 | **6409.4** |
| index | 3.8 Mo | **637 Mo** |
| préparation | 0.40 s | **61 s** |

**Pourquoi**, chiffré. `ecart` a la dimension d'une longueur au carré, et l'échelle d'une cellule à
n=2e4 vaut ~6e-5. La dérive de poids entre deux évaluations de Newton, mesurée itération par
itération :

| | it 0 | it 1 | it 2 | it 3 | it 4+ |
|---|---|---|---|---|---|
| dérive globale `max abs D` | 8.5e-3 | 4.2e-3 | 6.4e-3 | 7.0e-5 | 1.1e-4 |
| dérive **locale** `max abs( D_i − D_k )` | 6.5e-4 | 1.8e-3 | 2.7e-3 | 8.8e-5 | 1.6e-4 |
| rapport | 13.1 | 2.3 | 2.3 | 0.8 | 0.7 |

La quantité qui compte n'est pas `max abs D` mais `abs( D_i − D_k )` — le critère ne se décale que
de `− D_i + D_k`. L'espoir était que le pas de Newton, étant un potentiel lisse, fasse bouger les
germes voisins **ensemble** et rende cette différence beaucoup plus petite. Mesuré : elle n'est que
**2 à 13 fois** plus petite, et parfois plus grande.

Et surtout elle vaut **1.6e-4 à 2.7e-3 contre une échelle de cellule de 6e-5** : la marge nécessaire
est de **trois à quarante-cinq fois** la taille de ce que l'index encode. L'ensemble retenu croît
comme le carré, d'où le facteur 150.

**Conclusion : l'index ne peut pas être rendu réutilisable en bornant la dérive des poids, parce que
les poids bougent de plus que la géométrie que l'index encode.** C'est la même raison, chiffrée
autrement, qui a fait échouer `--tree hull` (« il ne paie que si l'index survit à plusieurs
itérations ») et le multi-échelle. Ce banc l'aura donc rencontrée trois fois par trois chemins
différents.

Ce que ça ne condamne pas : le front reste **2.1 à 4.3× plus rapide par diagramme**. Il paie
partout où l'index n'a pas à être refait — un diagramme isolé, une boucle de Lloyd où seules les
positions bougent, ou tout usage où les poids sont donnés. Ce qu'il faudrait pour l'amener dans
Newton n'est pas un bouclier mais une **construction moins chère** : elle coûte aujourd'hui cinq
diagrammes, dont 0.18 s de diagramme grossier et 0.25 s de listes pour 0.12 s de fronts.

## UN BINAIRE PAR ACCÉLÉRATEUR, ET LA DIMENSION EN PARAMÈTRE (2026-09-04)

Deux changements de structure, faits ensemble parce que le second était impraticable sans le premier.

### Pourquoi éclater `main.cpp`

Il faisait **2185 lignes** et portait tout : le chargement des nuages, le chronomètre, la
vérification, sept sondes de recherche, dix accélérateurs, et **quarante-trois drapeaux**. Chaque
idée nouvelle y ajoutait un drapeau et une branche, et plus rien ne disait quel drapeau allait avec
quel accélérateur — `--baisse` et `--enclos` ne servaient qu'à `pack`, `--psigrid` et `--front`
qu'à `front`, `--hull-rate` qu'à `hull`, et il fallait lire le code pour le savoir.

Ce que le découpage donne, au-delà de la lisibilité :

* **chaque banc n'instancie plus que SES combinaisons.** Le dispatch est en templates — `Cell`
  (cinq tailles), `CellBox`, `SkipInside`, `Weighted` — donc l'ancien binaire les instanciait pour
  *tous* les accélérateurs à la fois. MESURE, même machine, même `-j8` :

  | | temps de compilation |
  |---|---|
  | ancien `pd2d`, un seul binaire | **167 à 170 s** |
  | les douze nouveaux, de zéro | **63 s** |
  | un seul `main_Xyz.cpp` | **~18 s** |

  C'est un facteur dix sur la boucle d'essai, et c'est exactement ce que le banc existe pour offrir.
* **les sondes vivent à côté de ce qu'elles sondent.** `--enclos` et `--baisse` dans
  `main_pack.cpp`, `--psigrid` et `--front` dans `main_front.cpp`, `--majorant` dans `main_bsp.cpp`.
  Ce sont des mesures qui ont mené à un accélérateur ; les garder à côté de lui, c'est garder le
  raisonnement à côté de son résultat.
* **la partie commune est nommée.** `src/bench/` porte les options, les nuages, le chronomètre,
  l'affichage — et **LA SUITE**, la liste des cas importants. Un `main_Xyz.cpp` la déroule, en 2D
  puis en 3D, et c'est ce qui rend deux bancs comparables sans convention tacite.

Un détail qui n'en est pas un : **chaque cas dit de combien de sommets il a besoin**. L'uniforme se
contente de 32 en 2D, le nuage de lignes en veut 64, le nuage de plans 128. Sans cela, une suite
lancée sans argument déborderait sur deux cas sur trois et afficherait des aires fausses — ce
qu'elle a fait à la première version. `--maxnv` reste là pour passer outre.

### La dimension : ce qui a changé dans le contrat

Le concept d'accélérateur ne bouge pas — *proposer des germes* — mais ses deux rappels sont
maintenant typés :

```cpp
may_cut ( Vec<D> lo, Vec<D> hi, const WMajT<D> &wm ) -> bool
cut_with( Vec<D> p1, TF w1, SI i1 )                  -> bool
```

et l'accélérateur annonce `static constexpr int dim`, la cellule aussi (`Cell::dim`), sur quoi
`PowerDiagram` fait un `static_assert`. Tout le reste de `may_be_cut` était **déjà séparable par
axe** — l'écart entre deux boîtes, le point le plus loin d'une boîte, le coin qui maximise une forme
linéaire — donc les deux blocs écrits à la main pour `x` et `y` sont devenus une boucle `for d`,
et rien d'autre n'a bougé.

**Les boîtes passent PAR VALEUR, et ce n'était pas le choix évident.** La boîte d'un nœud de `AaBsp`
est déjà contiguë en mémoire : un `const TF *` aurait évité de la recopier. Mais `AaBsp4` range ses
boîtes en SoA (`lo[ axe ][ fils ]`) et la grille calcule les siennes à la volée — toutes deux
auraient dû matérialiser un tableau sur la pile juste pour en prendre l'adresse, et **prendre
l'adresse est précisément ce qui empêche le compilateur de garder la valeur en registres**. Le banc
compile avec `-fno-strict-aliasing` (l'arène de `AaBspPacked` l'exige), donc il ne peut même pas
prouver que rien n'écrit entre deux lectures.

Une précaution pour que la comparaison avec tout ce qui précède reste honnête : `weight_majorant`
garde **la formule de Cramer** en `D == 2` au lieu de passer par l'élimination générique. Les pentes
sont un CHOIX, pas une mesure — n'importe quelles pentes donnent un majorant valide pourvu que `b`
soit calculé avec elles — donc un arrondi différent donnerait un arbre qui n'élague plus tout à
fait pareil, et les mesures d'avant ne seraient plus opposables à celles d'après.

### Ce que ça coûte en 2D : la mesure appariée

Ancien binaire et nouveau, **alternés dans la même boucle**, machine au repos, un seul fil, trois
tours. C'est la seule forme de comparaison qui vaille : deux séries prises à dix minutes d'écart ne
sont pas opposables — ce chantier en a fourni la démonstration à répétition.

| 1 fil | ancien | nouveau | écart |
|---|---|---|---|
| `bsp`, uniforme 10⁶, nv=32 | 1.843 / 1.842 / 1.843 s | 1.826 / 1.819 / 1.837 s | **−0.9 %** |
| `bsp`, lignes / aires égales 10⁵, nv=64 | 0.538 / 0.537 / 0.537 s | 0.565 / 0.567 / 0.566 s | **+5.4 %** |
| `grid`, lignes / aires égales 10⁵ | 25.544 / 25.585 / 25.512 s | 22.361 / 22.362 / 22.357 s | **−12.5 %** |

Le paramètre de dimension est donc **gratuit sur le cas facile** et coûte **5 % sur le cas dur**.
Ce n'est pas ce qu'il coûtait au départ : la première version générique perdait 5.2 % sur
l'uniforme et 7.6 % sur le cas dur. Ce qui a rattrapé la différence, mesuré changement par
changement :

* **construire les `Vec` en INITIALISATION D'AGRÉGAT**, et pas par une boucle qui remplit un
  temporaire. `seed( k )` écrivait `r[ d ] = p[ d ][ k ]` coordonnée par coordonnée : la boucle
  alterne une lecture et une ÉCRITURE, et sous `-fno-strict-aliasing` le compilateur doit supposer
  qu'un store de `TF` peut écraser le `TF *` interne du `std::vector` d'où vient la lecture. Il
  recharge donc le pointeur entre deux coordonnées. En initialisation d'agrégat, toutes les lectures
  précèdent la construction de l'objet. **+5.2 % → +2.4 %** sur l'uniforme, à ce seul changement.
* **hoister les pentes du majorant** hors des deux boucles de `may_be_cut`, ce que la version 2D
  faisait à la main avec deux `const TF`. Neutre isolément — et c'est pourtant en le combinant au
  retrait du point suivant que l'uniforme est repassé sous l'ancien.

Et ce qui a été **essayé, mesuré et retiré** : un accesseur `c.coord( v, d )` pour éviter de
matérialiser `c.vertex( v )` dans le balayage des sommets. Neutre à lui seul (1.889 / 1.893 / 1.910
contre 1.897 / 1.885 / 1.887), et le code se lisait moins bien. Le retirer, une fois les pentes
hoistées, a rendu 1.823 s là où la combinaison des deux donnait 1.889 : ces deux micro-choix
**interagissent**, et aucun ne se juge isolément. C'est la leçon générale de la section suivante.

### LA GRILLE : le compilateur décide plus que l'algorithme

Le cas le plus instructif de tout ce chantier, et il faut le dire tel qu'il est. Sur le nuage à
aires égales, un fil, la grille a été mesurée dans **cinq variantes du même code**, dont les
DÉCISIONS sont identiques :

| variante | temps |
|---|---|
| ancien `main.cpp`, `--cellbox` (défaut) | 25.51 – 25.59 s |
| ancien `main.cpp`, `--no-cellbox` | 17.34 s |
| générique, première version, `--cellbox` | **14.55 s** |
| générique, ordre d'anneau remis comme l'ancien | 15.70 s |
| générique **final**, `--cellbox` | 22.36 s |
| générique final, `--no-cellbox` | 25.44 s |

Les compteurs, relevés avec `--stats` dans la MÊME configuration (`CellBox = true`), sont identiques
d'un bout à l'autre :

```
ancien : boites 4546.0  balayees 719.0  gardees 671.3  coupes tentees 871.8  effectives 62.06
nouveau: boites 4546.0  balayees 719.0  gardees 671.3  coupes tentees 871.8  effectives 60.91
```

Mêmes boîtes visitées, mêmes boîtes évincées, mêmes germes proposés. Seules les coupes EFFECTIVES
bougent, de 2 %, parce que le parcours d'anneau générique énumère les cases dans l'ordre inverse de
la boucle écrite à la main — et cet effet-là est mesuré à part : remettre l'ordre d'origine coûtait
8 % (15.70 contre 14.55).

**Tout le reste — un facteur qui va jusqu'à 1.75 — est de la mise en forme par le compilateur.** Sur
ce cas précis, changer `c.vertex( v )` en `c.coord( v, d )`, ou hoister deux conversions
`float → double`, déplace le temps de 14.5 à 22.4 s sans qu'une seule décision géométrique change.

Trois choses à en retenir, et elles valent au-delà de la grille :

1. **`--no-cellbox`, la configuration que ce README avait retenue, n'est plus le bon choix pour la
   grille** : elle est aujourd'hui la plus lente des deux, alors qu'elle était la plus rapide ;
2. le cas dur de la grille n'est **pas une mesure d'algorithme**, c'est une mesure de compilateur.
   Les conclusions qu'on en a tirées — « la grille s'effondre quand la densité varie », un facteur
   40 contre le BSP — tiennent parce qu'elles portent sur des ordres de grandeur, pas sur des
   pourcentages ;
3. et c'est la raison pour laquelle **toute mesure de ce banc doit être appariée** : la même source,
   compilée à deux moments du chantier, ne donne pas le même temps.

### La suite 2D, avec le code final

`n = 10⁶` pour l'uniforme, `n = 10⁵` pour les nuages de lignes, un fil / huit fils, en secondes.
Les conclusions d'avant tiennent toutes ; c'est le point de cette table.

| | uniforme 10⁶ | lignes / Voronoi | lignes / aires égales |
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

(`obsp` déborde 64 sommets sur les deux nuages de lignes, 9 et 8 coupes : ses deux dernières
colonnes portent une mesure fausse, comme elles le faisaient déjà avant le découpage.)

### `--memo` : la borne supérieure, et elle n'est pas nulle

`pd_memo` mesure autre chose que ce que `--newton --memo` mesurait, et c'est ce qui change la
conclusion. Ici **les deux passes ont les mêmes poids**, donc chaque coupe rejouée est exactement
une coupe de la cellule finale : c'est la BORNE SUPÉRIEURE de ce que l'idée peut rendre. La même
mesure avec le même code, mémoire éteinte, sert de témoin.

| seconde passe, mémoire pleine | 1 fil | 8 fils |
|---|---|---|
| uniforme 10⁶ | 1.940 → 1.568 s, **−19.2 %** | 0.251 → 0.205 s, **−18.5 %** |
| lignes / Voronoi | 0.190 → 0.159 s, **−16.3 %** | 0.025 → 0.021 s, **−17.6 %** |
| lignes / aires égales | 0.555 → 0.546 s, −1.6 % | 0.091 → 0.090 s, −1.3 % |

Ce que ça corrige de ce qui précède : l'idée **n'est pas neutre**, elle vaut un sixième du temps sur
les cas où la cellule est autour de son germe. Ce qui était mesuré avant, et qui reste vrai, c'est
qu'elle ne rend rien DANS UNE BOUCLE DE NEWTON — parce que les poids bougent entre deux passes et
que les souvenirs se périment. Les deux mesures ne se contredisent pas, elles répondent à deux
questions ; il fallait un banc séparé pour les distinguer.

Et le cas dur reste imperméable (−1.6 %), pour la raison déjà donnée : quand la cellule est loin de
son germe, ce ne sont pas les coupes qui coûtent, c'est de les trouver.

### La cellule 3D : pourquoi ce n'est pas la 2D avec un axe de plus

En 2D les sommets **en ordre cyclique** sont toute la géométrie : l'aire se lit par le lacet, et les
coupes se retrouvent par l'invariant *la coupe `i` porte l'arête `[ v_i, v_i+1 ]`*. Rien de cela ne
survit en 3D — il n'y a plus d'ordre cyclique global, et une coupe porte une FACE, c'est-à-dire un
cycle d'arêtes qu'il faut savoir retrouver. C'est le même constat que fait `sdot`, où `cut` et
`measure` ont deux versions selon le régime de dimension.

Ce qui est stocké : **un sommet porte ses TROIS coupes**, une arête porte ses deux sommets. C'est
tout — ni les faces, ni les équations des plans.

* les **faces** ne sont pas stockées parce qu'elles se relisent : la face de la coupe `c` est
  l'ensemble des sommets dont le triplet contient `c`, et ses arêtes celles dont les deux bouts la
  contiennent. Les stocker obligerait à les maintenir à chaque coupe, alors qu'on n'en a besoin
  qu'à la fin, une fois.
* les **équations des plans** non plus. Le volume se calcule par des tétraèdres sur un point
  intérieur (le centre de gravité des sommets, qui est intérieur puisque la cellule est convexe),
  ce qui ne demande que des sommets ; et l'identité du voisin, seule chose dont l'appelant ait
  besoin, est l'indice de coupe lui-même.

L'hypothèse est que le polytope est **simple** : chaque sommet est sur exactement trois plans. Vrai
en position générale, et c'est ce qui rend la coupe purement **combinatoire** — deux nouveaux
sommets de la face créée sont voisins exactement quand ils partagent une ANCIENNE coupe, l'arête qui
les joint étant l'intersection de cette coupe-là avec la nouvelle. Le convexe garantit qu'une
ancienne coupe ne peut apparier qu'une seule paire : sinon sa trace sur la face neuve aurait deux
morceaux.

Newton n'a rien eu à savoir de tout ça. Les deux cellules exposent la même porte,
`for_each_facet( f( coupe, mesure ) )` — longueur d'arête en 2D, aire de face en 3D — et
`c_ij = |facette| / ( 2 |p_i − p_j| )` ne dépend pas de la dimension.

**Le bug que la somme des volumes a attrapé.** La compaction des sommets gardés envoie `i` vers
`map[ i ] <= i`, et je l'avais écrite **en descendant** — l'écriture passait alors devant la lecture
et écrasait un sommet pas encore recopié. Le résultat était un polyèdre de **bonne topologie et de
mauvaise géométrie** : `nb`, `ne` et le nombre de faces étaient tous corrects, `cut` ne signalait
rien, et la somme des volumes valait **0.33** au lieu de 1. C'est exactement ce que le seul contrôle
du banc est là pour attraper, et rien d'autre ne l'aurait vu.

**La validation, une fois corrigé** (`pd_check`, 1500 germes uniformes dans le cube, contre le
balayage complet — qui n'a aucune ligne de code géométrique en commun avec l'arbre) :

| poids | somme des volumes | `bsp` vs balayage | `grille` vs balayage |
|---|---|---|---|
| nuls | `1.000000000000` | 3.14e-18 | 3.69e-18 |
| `--weights 1` | `1.000000000000` | 6.94e-18 | 3.90e-18 |
| `--weights 10` | `1.000000000000` | 1.39e-17 | 1.04e-17 |

Les cellules pavent le cube exactement, et deux chemins de code indépendants s'accordent au bit
près. Ce n'est pas deux fois le même bug.

**Ce qui reste cher, et se voit :** `measure()` retrouve les faces à chaque appel, en
`O( F ( V + E ) )` — pour une cellule poissonienne moyenne (V=27, E=41, F=15) c'est un millier de
comparaisons par cellule, avant même de compter les tétraèdres. C'est le premier endroit à
optimiser en 3D, et il n'a pas été touché : ce qu'on voulait d'abord était une cellule JUSTE.

### Les nuages durs en 3D, et une réserve à dire

`cases/gen_cases_3d.py` fait l'analogue exact des lignes 2D : des diracs serrés autour de quelques
**plans** traversant le cube — une variété de codimension 1, donc une densité qui s'effondre entre
deux nappes. L'écart-type transverse est pris à l'échelle de l'espacement moyen `h = n^(-1/3)`,
comme en 2D où `sigma = 0.005` vaut 1.6 fois `h = n^(-1/2)`.

**La réserve.** En 2D, les poids du cas `equal` viennent de `pysdot`, donc d'un code indépendant du
banc : les comparer à ceux que `pd_newton` retrouve est une vraie vérification croisée, et c'est
elle qui donne le `3.5e-15` sur une amplitude de `0.128` cité plus haut. Ici pysdot n'est pas
disponible — il est compilé pour un interprète qui n'existe plus sur cette machine — donc le cas
`equal` 3D est produit **par le banc lui-même** :

```
xmake run pd_newton --3d --load cases/planes4_n100000_s0.02_voronoi.txt \
                    --ecrire cases/planes4_n100000_s0.02_equal.txt
```

Il reste un cas de **chronométrage** parfaitement valide — c'est un nuage dont les poids éloignent
les cellules de leurs germes, et c'est tout ce qu'on lui demande — mais il n'est **pas** un témoin
indépendant pour Newton. Le fichier le dit dans son en-tête.

Ce que la résolution elle-même a donné, à `n = 1e5`, huit fils :

| | itérations | diagrammes | résidu | total |
|---|---|---|---|---|
| plans / volumes égaux, 3D | 13 | 27 (13 reculs) | 1.16e-10 | 24.5 s, dont **82 % de diagramme** |

Deux choses à en retenir. D'abord **`--maxnv 128` débordait 439 fois** sur les 2.7 millions de
cellules construites, et une instanciation temporaire à 256 sommets a donné le même chemin
d'itérations au chiffre près (`1.162e-10`, 13 itérations, 27 diagrammes) pour le même temps à 1 %
près — la taille de la cellule ne coûte rien tant qu'on ne touche pas les tableaux au-delà de `nb`.
Le fichier livré a donc été produit sans un seul débordement. (Cette instanciation ne reste PAS dans
le banc : `Cell3T<256>` coûtait à elle seule près de trois minutes de compilation dans chaque
`main`, ce qui est le contraire de ce que ce banc doit être. Pour la refaire, ajouter `case 256`
dans `mesure_dispatch` et dans `deroule_newton`.)

Ensuite la part du diagramme monte à **82 %** contre ~40 % en 2D : en 3D, tout ce qui n'est pas le
diagramme devient du bruit, et c'est l'accélérateur qui décide de tout.

### Les premiers chiffres en 3D

Un fil et huit fils, `n = 2·10⁵` pour l'uniforme et `n = 10⁵` pour les nuages de plans,
`nv = 128` partout, temps en secondes et (ns/germe) :

| | `bsp` 1 fil | `bsp` 8 fils | montée | `grid` 1 fil | `grid` 8 fils |
|---|---|---|---|---|---|
| uniforme 2·10⁵ | 6.251 (31 257) | 0.807 (4 033) | **×7.7** | 15.492 (77 459) | 2.051 (10 254) |
| plans / Voronoi 10⁵ | 3.126 (31 264) | 0.414 (4 143) | **×7.6** | 12.949 (129 493) ⚠ | 1.831 ⚠ |
| plans / volumes égaux 10⁵ | 5.597 (55 974) | 0.777 (7 770) | **×7.2** | 43.387 (433 871) ⚠ | 6.339 ⚠ |

Ce qu'il faut y lire, et rien de tout cela n'était acquis :

* **la montée en charge tient.** ×7.2 à ×7.7 sur huit cœurs physiques, exactement comme en 2D
  (×7.6). Le passage en 3D ne réintroduit ni partage ni synchronisation — la cellule reste sur la
  pile, chaque thread n'écrit que dans `res[ i ]`.
* **une cellule 3D coûte 17 fois une cellule 2D** sur l'uniforme (31.3 µs contre 1.82). C'est
  beaucoup, et une bonne part est dans `measure()`, qui retrouve les faces à chaque appel — c'est le
  premier chantier d'optimisation 3D, pas une fatalité.
* **le nuage dur est RELATIVEMENT moins dur en 3D.** Les poids à volumes égaux coûtent 1.8 fois
  l'uniforme (56.0 contre 31.3) là où en 2D les aires égales coûtent 3.1 fois (5.64 contre 1.82).
  L'interprétation : ce qui rend le cas 2D difficile est que la cellule s'éloigne de son germe, et
  en 3D il y a une direction de plus pour la retenir.
* **la grille se dégrade dans les deux sens.** Sur l'uniforme, là où elle devrait être bonne, elle
  prend déjà 2.5× le BSP (contre 1.5× en 2D) : le voisinage immédiat qu'elle balaie **sans rien
  tester** passe de `3² = 9` cases à `3³ = 27`, soit ~270 germes proposés au lieu de ~90, quand le
  BSP en propose une trentaine. Et sur le nuage dur elle prend 7.8× — moins que les 40× de la 2D,
  mais depuis une base déjà mauvaise.
* ⚠ **la grille déborde 128 sommets sur les nuages de plans** (132 et 133 coupes) là où le BSP ne
  déborde pas du tout sur les mêmes nuages. Même cause qu'en 2D pour le front : ce n'est pas la
  cellule finale qui est grosse, c'est le polyèdre INTERMÉDIAIRE, et il dépend de l'ordre des
  coupes. Le BSP, qui descend fils-le-plus-proche, obtient le bon ordre gratuitement.

Et un réglage que la 3D impose : **le défaut est passé à 128 sommets**. À `n = 2·10⁵`, `--maxnv 64`
débordait 135 fois sur l'uniforme et la somme des volumes tombait à `1.000000001`. La borne ne coûte
rien à l'exécution (les tableaux ne sont touchés que jusqu'à `nb`) ; elle coûte à la COMPILATION, et
c'est ce qui a fixé le plafond des instanciations à 128.

## Ce qui reste à essayer, en 3D d'abord

* **`measure()` en 3D**, qui retrouve les faces à chaque appel en `O( F ( V + E ) )`. C'est le poste
  le plus gros d'une cellule 3D et il n'a pas été touché : on voulait d'abord une cellule JUSTE.
* **le nœud de BSP en 3D** ne tient plus dans une ligne de cache (six bornes en `double` plus trois
  pentes). Les bornes en FP32 le feraient rentrer — c'est la même piste qu'en 2D, mais elle y était
  facultative et ici elle ne l'est plus.
* **`--memo` en 3D**, la seule dimension où le gain de 3× d'`old_pd` a été observé. La borne
  supérieure mesurée en 2D (−19 %) dit que l'idée n'est pas morte ; il reste à voir si la 3D la
  rend décisive.
* **les accélérateurs d'étude en 3D** — `pack`, `hull`, `front`. Leur porter demande d'abord de
  savoir quelle question 3D ils répondent : `front` s'appuie sur les sommets d'une cellule grossière,
  ce qui est dimension-générique tel quel, alors que `hull` a besoin d'un k-DOP 3D.
* et l'ancienne liste, toujours ouverte : `CellAoS` contre `CellSoA` ; du SIMD explicite sur le test
  d'éviction, qui ne se vectorise pas à cause de son `return` anticipé ; une grille / quadtree en
  ordre Z à côté du BSP.

Et surtout : **reporter dans `sdot` ce qui a payé ici**, à commencer par la question de fond que ce
banc a tranchée — pourquoi le même algorithme y monte à 7.6× et n'y monte qu'à 2.1× sous SYCL.


## LE RANGEMENT, ET LES MESURES 3D (2026-09-04, après-midi)

Deux choses ce jour-là. Les `.h` sont rangés en `util/`, `geometry/`, `spatial_accel/`, `solver/`,
`bench/`, `mains/` — un déplacement pur, aucune ligne de code touchée, `xmake -j8` remesuré à 62 s
comme avant. Et `measure()` en 3D a été récrit.

Le détail de la démarche est en **README § 5**, qui est écrit pour être lu ; ce qui suit est le
relevé.

**L'ablation d'abord**, parce que c'est elle qui a décidé d'y aller : en remplaçant `measure()` par
une constante (résultat faux, tout le reste du travail conservé), le diagramme 3D uniforme passe de
6.25 à 4.88 s. `measure()` pesait donc **21.7 %**. Après le travail, la même ablation donne 5.31
contre 4.88 : **13 %**. Le profil pointe maintenant `Cell3::cut` à 54 %.

Les quatre étapes, chacune mesurée appariée sur `uniforme n=2·10⁵`, 1 fil, trois tours :

| | temps | |
|---|---|---|
| cycles (départ) | 6.246 s | |
| accumulation, éventail depuis le centre de face | 5.741 s | −8.2 % |
| éventail depuis un SOMMET | 5.625 s | −2.0 % |
| numéro de face par hachage | 5.511 s | −2.0 % |
| autre écriture du même test | 5.313 s | −3.7 % |

Total **−14.9 %** sur l'uniforme, −15.0 % sur les plans/Voronoï, −9.6 % sur les plans/volumes
égaux ; identique à 8 fils (−15.1 / −14.8 / −9.0 %). Newton 3D : 24.29 → 20.75 s, **−14.6 %**, dont
−17.6 % sur le poste diagramme — plus que le banc pur, parce que Newton demande volume ET facettes
et que `measure_and_facets` ne fait plus qu'une accumulation pour les deux.

**Trois choses à ne pas oublier de cette journée.**

1. La dernière ligne du tableau (−3.7 %) n'est pas de l'algorithme : deux écritures sémantiquement
   identiques du même test. Vérifié que ce n'est ni le système de compilation ni l'ordre des
   déclarations sur la pile. Et la même modification fait **perdre 1.9 % à `pd_newton`**. C'est du
   placement de code, et c'est la deuxième fois après la grille.
2. Dans le même esprit : **la 2D perd 2 à 4 % dans ce commit sans une instruction de différence**.
   La seule modification de `Cell.h` est une méthode template que `pd_bsp` n'instancie pas.
3. Le nouveau chemin ne peut pas échouer là où l'ancien le pouvait en silence : le parcours de
   cycle abandonnait une face dont un sommet n'avait pas le degré 2, et la face disparaissait du
   volume. Il n'y a plus de parcours. `measure_by_cycles()` est gardée hors du chemin chaud comme
   SECOND chemin, et `pd_check` compare les deux cellule par cellule — écart maximum `1.08e-18`
   sans poids, `6.94e-18` à `--weights 10`. En 3D, aucune référence extérieure ne donne le volume
   d'un polyèdre : cet accord-là est le seul témoin qu'on ait.


## LES SUR-CELLULES D'AGRÉGATS (2026-09-09)

L'idée de départ, reprise d'une discussion ancienne : au lieu de chercher les voisins d'un germe un
par un, grouper les germes en **agrégats**, calculer pour chaque agrégat une **sur-cellule** qui
contient la réunion des cellules de ses membres, et s'en servir pour savoir qui peut déborder sur
qui. On avait buté là-dessus parce que la sur-cellule était vue comme une réunion de cellules,
donc chère. La question relancée était : existe-t-il une forme d'enceinte qui rende la sur-cellule
**continue** — pas d'échantillonnage à l'intérieur — et bon marché ?

Tout ce qui suit est en 2D, dans `cases/surcell2d*.py`, plus une application interactive
`cases/surcell2d_ui.html`.

### Le principe, et ce qu'il autorise

Un minorant de `psi_A = min_{i dans A} h_i` donne un sur-ensemble :

    f_A <= psi_A   =>   S_A = { f_A <= h_j pour tout j hors de A }   contient   U_{i dans A} Lag_i

**L'agrégation est arbitraire.** Aucune propriété de correction ne dépend de la façon dont on
groupe les germes — seulement la performance. C'est ce qui rend tout le schéma souple.

Avec une enceinte convexe `K` contenant l'agrégat et un majorant affine des poids `w <= a.y + b`,
le minorant s'écrit comme un **paraboloïde tronqué et tangenté**,
`Phi_K( x ) = max_{p dans K} ( 2 p.x - |p|^2 ) + b`, et le bord de `{ Phi_K < l_j }` a **trois
régimes** selon l'élément de `K` le plus proche : un sommet donne une **droite** (bissectrice de
puissance ordinaire), une arête une **parabole** de foyer `p_j`, l'intérieur un **cercle**. Les
raccords sont C¹ (`grad Phi_K = 2 proj_K( x )`).

Deux résultats de structure, tous deux mesurés :

* **`omega = 0` est impossible.** Un minorant fini de `psi_A` de la forme « pseudo-germes sans
  cran » n'existe pas : l'évaluer en `x = p_i` donne la contradiction. Il faut un **cran constant**
  `omega`, et alors il suffit que `conv( Q )` contienne `conv( P )`.
* **La réunion est forcée.** `{ max_v f_v >= max_j g_j } = U_v { f_v >= max_j g_j } = U_v Lag( v )`.
  Un majorant en forme de `max` produit nécessairement une RÉUNION de cellules convexes, une par
  pseudo-germe. Ce n'est pas un choix d'implémentation.

Le bord exact est **linéaire en nombre de coupes** — 1.2 à 1.6 arcs par coupe, aucun trou —
conforme au théorème des pseudo-disques (`Phi_K - l_j` est convexe sur la droite `{ l_j = l_k }`,
donc au plus deux traversées, donc `6n - 12` arcs pour `n` cellules).

### L'ellipsoïde ne gagne jamais

Comparées à enceinte égale (`cases/surcell2d_formes.py`) : enveloppe convexe, k-DOP, capsule,
ellipse, boîte, disque. L'ellipse est la seule candidate anisotrope sans distance en forme close,
et elle ne gagne sur aucune des deux métriques. **Le k-DOP est la bonne enceinte** — en 2D comme en
3D, et il donne le cran `omega` par ses arêtes.

Un bogue instructif au passage : la bissection de la distance à l'ellipse effondrait l'encadrement
(`hi = where( m, mid, hi )` au lieu de `where( m, hi, mid )`), et le symptôme était une ellipse
**plus petite que l'enveloppe convexe** — géométriquement impossible. L'invariant qui l'aurait
attrapé tout de suite, ajouté depuis : `d2` doit s'annuler sur les germes de l'agrégat.

### Le cran `omega`, et où il est valide

Pour `p` sur l'arête `[ v, v' ]` de longueur `L` : `l_p - l_v = s^2 - ( s - t )^2 <= s^2` et
symétriquement `l_p - l_v' <= ( L - s )^2`, donc `min( s^2, ( L - s )^2 ) <= L^2 / 4`. D'où
`omega_v = max( arêtes adjacentes )^2 / 4`, et la parabole est remplacée par une **intersection de
demi-plans** — un « U carré » inscrit dans le U rond.

**Ce minorant n'est valide qu'À L'EXTÉRIEUR de l'enveloppe**, et je m'étais trompé en le croyant
global. La preuve ne parle que des `p` du BORD, alors que `Phi_K` est atteint en `proj_K( x )`, qui
est un point INTÉRIEUR dès que `x` est dans `K` — `l_p` est concave en `p`, pas linéaire. Mesure :
sur un pentagone régulier de rayon 0.2, le minorant colle à `dist^2` à **8e-08** près dehors et le
dépasse de **0.026** au centre. Le morceau `K` de `S_A = K u ( U_v Lag( v, ... ) )` est donc
porteur, pas décoratif. Après correction : 1500 configurations aléatoires, **0 violation**.

### Sous-découper les arêtes : `omega` divisé par `N^2`

Un pseudo-germe ne peut pas porter deux crans différents (le `min` prendrait le plus gros), donc
`omega_q = max( sous-cordes adjacentes )^2 / 4` est forcé et **sous-découper est le seul levier**.
Sur-cellule plane contre sur-cellule courbe, à enceinte FIXÉE, `N` = sites par arête :

| forme | N=1 | N=2 | N=3 | N=4 | N=6 |
|---|---|---|---|---|---|
| pentagone | 1.041 | 1.010 | 1.003 | 1.003 | 1.001 |
| boîte 8:1 | **1.322** | 1.085 | 1.038 | 1.019 | 1.008 |
| 8-DOP | 1.019 | 1.006 | 1.003 | 1.001 | 1.001 |

Le coupable est l'arête la plus longue ; c'est pour ça que seule la boîte allongée souffre.

**À nombre de plans égal, sous-découper ou ajouter des directions ?** Les deux régimes s'inversent
selon la forme de l'agrégat (contre la sur-cellule exacte des Diracs, 40 Diracs, 10 tirages) :

| plans | agrégat rond | agrégat allongé 6:1 |
|---|---|---|
| 8 | 8-DOP·1 = **1.164** · 4-DOP·2 = 1.343 | 8-DOP·1 = 1.367 · 4-DOP·2 = **1.217** |
| 12 | 12-DOP·1 = **1.121** · 4-DOP·3 = 1.292 | 12-DOP·1 = 1.288 · 4-DOP·3 = **1.152** |
| 24 | 12-DOP·2 = **1.052** · 8-DOP·3 = 1.062 | 12-DOP·2 = 1.085 · 8-DOP·3 = **1.077** |

Sur un agrégat allongé, les directions ne rachètent rien : `sqrt( omega )` reste à **0.428** pour un
12-DOP contre **0.262** pour un 4-DOP sous-découpé deux fois, parce que les directions d'un k-DOP
sont réparties uniformément en angle et ne raccourcissent pas les longs côtés. Et les coûts ne sont
pas symétriques : **une direction de plus coûte `O( |A| )`, un sous-découpage coûte `O( 1 )`.**

### Le pari des « fonds de U » : PERDU

Proposition testée (`--convexe`) : puisque les côtés des U carrés sont mangés par les voisins, ne
garder qu'UN demi-plan par voisin — le fond du U — et obtenir une sur-cellule convexe.

| nuage | sommets sortis | somme des aires |
|---|---|---|
| uniforme, Voronoï | 541 | 1.0033 |
| uniforme, poids aléatoires | 389 | 1.0012 |
| 5 lignes, Voronoï | 1092 | **1.0958** |
| 5 lignes, aires égales | 3869 | **1.0984** |

Ça ne pouvait pas marcher : **aucun demi-plan n'est contenu dans un coin convexe**, donc retirer le
demi-plan porté par un fond de U enlève strictement plus que le U. Le prix était pourtant réel — la
préparation tombait de 11.9 à 1.8 coupes par germe.

### LA CONVEXIFICATION EXTÉRIEURE : ÷2.8 à ÷3.6, et elle est exacte par construction

Le pari perdu remplaçait la réunion par quelque chose de CONTENU dedans. La bonne version fait
l'inverse : `conv( U morceaux )` **contient** la réunion, donc la couverture ne peut que grandir et
l'algorithme reste juste sans rien à prouver. Vérifié sommet par sommet : pire sortie **−7e-17** sur
250 agrégats.

Travail par germe (étapes 3 et 4, préparation exclue ; modèle `2 |P| ( |P| + |Q| )` par séparation,
`4 |poly|` par coupe — donc une implémentation à la C++, en Python le surcoût d'appel masquerait
tout) :

| nuage, ρ | | table | juge | coupes | **total** | | tentées |
|---|---|---|---|---|---|---|---|
| uniforme V, 8 | morceaux | 4306 | 12028 | 986 | **17320** | | 44.3 |
| | enveloppe, table seule | 960 | 12042 | 986 | 13988 | ×1.24 | 44.3 |
| | **enveloppe partout** | 960 | 3329 | 1042 | **5331** | **×3.25** | 46.8 |
| uniforme V, 16 | morceaux | 2735 | 15317 | 2081 | **20133** | | 91.3 |
| | **enveloppe partout** | 665 | 4266 | 2257 | **7188** | **×2.80** | 98.9 |
| 5 lignes V, 8 | morceaux | 6679 | 13733 | 1385 | **21797** | | 57.9 |
| | **enveloppe partout** | 1122 | 3462 | 1466 | **6050** | **×3.60** | 61.6 |
| aires égales, 8 | morceaux | 12343 | 18964 | 3448 | **34755** | | 159.4 |
| | **enveloppe partout** | 1848 | 5291 | 3545 | **10684** | **×3.25** | 163.5 |
| aires égales, 16 | morceaux | 4090 | 16173 | 2878 | **23141** | | 133.0 |
| | **enveloppe partout** | 720 | 4262 | 3034 | **8016** | **×2.89** | 139.6 |

Le résultat qui décide : **la densité de la table ne bouge pas** — 12.7 → 12.7, 18.9 → 19.0,
16.2 → 16.2. Le test morceau par morceau, exact, ne retire qu'environ 1.1 candidat sur les 13.8
que laisse déjà la phase large k-DOP, et l'enveloppe en retire autant. La précision qu'on payait
`M x M'` ne servait à rien.

Trois lectures :

* **Convexifier pour la table seule ne vaut pas le détour** (×1.24). Le poste dominant était le
  **juge** (69 % du total), pas la table. Il faut convexifier partout.
* Le coût de construction est négligeable : **40 à 45 points en entrée, 7.9 à 8.7 sommets en
  sortie** par agrégat, et les points d'entrée sont déjà là.
* **En 3D l'argument se renforce** : le nombre de morceaux est le nombre de sommets de l'enceinte,
  ~24 pour un 14-DOP. Le test de table passe de `M x M'` ≈ 600 séparations polyèdre-polyèdre — bien
  plus chères qu'en 2D, il faut les axes croisés arête×arête — à UNE. Le rapport gagne un facteur
  `M`, donc il croît avec la dimension.

### Ce que le prototype a appris en se trompant

`cases/surcell2d_algo.py` déroule les quatre étapes (Voronoï grossier, sur-cellules, table,
cellules finales) et se vérifie contre la triangulation régulière. Exact sur les seize
configurations. Trois défauts trouvés en route, tous du même genre : **sûrs mais faux, et
silencieux**.

1. **Les germes dominés.** Un germe de cellule vide n'a aucun voisin de facette, donc l'argument de
   validité de la liste de candidats est VIDE et l'algorithme peut rendre une cellule fantôme
   (somme 1.356, 6 cellules sur 2000). Ce n'est pas un bogue de codage : c'est l'argument qui ne
   couvrait pas le cas. Réparé par une **amorce** (couper d'abord par tous les sites grossiers),
   avec preuve.
2. **`K` impair casse la phase large.** Elle suppose les directions opposées deux à deux ; avec
   `K = 3` la table se vide, les aires partent à 2.0, et **le certificat ne le voit pas**. C'est le
   même défaut que les indices `0, 2, 4, 6` codés en dur d'il y a quelques mois. Un `assert` est en
   place.
3. **La séparation par axe était incomplète.** `disjoints` ne testait chaque axe que dans un seul
   sens, avec la normale RENTRANTE. Le prédicat restait sûr — il ne déclare jamais disjoints deux
   convexes qui se touchent — mais il ratait de vraies séparations et gonflait la table. Corrigé
   (les deux sens, ce qui le rend aussi indépendant de l'orientation) : les `tentées` tombent de
   **48.9 à 44.3** par germe sur le nuage uniforme, sans rien changer d'autre. Tous les chiffres de
   ce prototype antérieurs à cette correction sont gonflés d'environ 9 %.

Ce qui a mis sur la piste du troisième : la table **rétrécissait** avec la convexification, ce qui
est impossible si l'enveloppe contient la réunion.

**La tangence, une quatrième fois** (trois fois déjà dans ce cahier) : deux ensembles qui se
touchent en mesure nulle doivent compter comme se recouvrant, sinon le plancher est franchi par le
bas. Dilatation d'un pixel côté grille, `eps` exigeant un vrai jour côté séparation.

### L'application interactive

`cases/surcell2d_ui.html` — un fichier, aucune dépendance, aucun serveur. Un polygone plein à cinq
sommets et cinq germes déplaçables ; molette pour les poids ; les trois régimes coloriés ; le
curseur de sous-découpage ; les aires et le nombre d'arcs affichés.

Pas d'échantillonnage à l'intérieur du polygone, finalement : le `&=` sur les images de coupe
contre plein de points `p` est exactement `{ |x - A|^2 - w < dist( x, P )^2 }`, donc l'intersection
sur `p` a une forme close. On la calcule par pixel, ce qui donne le même résultat sans le risque de
rater le bon `p` — et les régimes viennent avec, puisque l'élément le plus proche est déjà connu.

Deux pièges d'implémentation notés parce qu'ils se reproduiront : les arcs se comptent en
**8-connexité** (un bord à 45° donne un arc par pixel en 4-connexité — 617 au lieu de 15), et
l'enveloppe est réorientée CCW pour que le test d'intérieur ne dépende pas de l'ordre des sommets.

### Où ça en est, et ce qui reste

Comparaison honnête avec le BSP, opérations comptées des deux côtés (le BSP paie aussi ses tests de
boîte) : à parité sur l'uniforme, meilleur sur le cas dur. Mais **l'amorce, dans le prototype, est
un balayage trié en force brute** — 7 coupes par germe sur l'uniforme, 114 sur le cas dur. C'est
aujourd'hui le premier poste, et c'est réparable : la connectivité grossière est calculée à
l'étape 1 et n'est pas utilisée.

Après convexification, le juge pèse encore 50 à 62 % du total. Deux leviers, dans l'ordre :

1. la **liste des candidats n'est pas coupée par le critère d'anneau** — elle est triée par distance
   mais parcourue jusqu'au bout ;
2. un **pré-filtre par boîte** avant le juge : 76 % des juges sont des rejets.

Et le réglage à retenir : **peu de directions, plusieurs sites par arête** — `M = 4` ou `6`,
`N = 2` ou `3` — plutôt que `M = 12`.

## LA SUR-CELLULE CALCULÉE, ET LES ENCEINTES (2026-09-09, suite)

La section précédente décrivait une sur-cellule **minorée** : on approchait `psi_A` par des plans
issus d'une enceinte convexe, ce qui imposait des pseudo-germes, un cran `omega = (arête/2)^2`
irréductible, et un sous-découpage pour le diviser. Tout ça est mort, et c'est une bonne nouvelle.

L'objection était la bonne : *« comme le nombre de vertices des englobants d'agrégats risque de
correspondre plus ou moins au nombre de points par agrégat, autant calculer en 1ère passe toutes
les cellules pour tous les diracs vs les échantillons extérieurs et surtout les autres points de
l'agrégat »*. Ce n'est pas une variante de la sur-cellule minorée : **c'est la sur-cellule exacte.**
Les cellules des membres de `A`, coupées par LE MÊME jeu de plans `A u ext( A )`, pavent exactement

    U_A = { x : min_{i dans A} h_i( x ) <= min_{e dans ext( A )} h_e( x ) }

et `U_A` contient la réunion des vraies cellules de `A`, parce que `ext( A )` est un sous-ensemble
des germes et qu'ajouter des germes ne peut que rétrécir. Il n'y a plus rien à minorer, donc
`omega = 0`, donc plus aucune borne de poids nulle part.

Deux conséquences structurelles s'en déduisent et servent partout :

* un sommet né de plans **tous internes** est partagé par plusieurs cellules de `A`, donc intérieur
  à `U_A`. Seuls les sommets portant un plan extérieur (ou une face du domaine) sont sur le bord :
  le filtre est gratuit et exact ;
* `somme des |C_i| >= 1` sur tous les diracs, parce que les `U_A` recouvrent le domaine (pour tout
  `x`, le germe qui minimise `h` sur TOUT le nuage met son agrégat dans le compte). **L'excès sur 1
  est exactement le recouvrement des sur-cellules, c'est-à-dire leur prix.** C'est le témoin de
  l'étape, l'exact analogue du « somme des mesures = 1 » d'un diagramme complet.

### LE CERTIFICAT DE LA PASSE 1 N'EXISTE PAS

L'espoir était qu'une bonne part des diracs soient **finis** dès la passe 1 — tous leurs vrais
voisins déjà dans `A u S`. Mesuré exactement (Delaunay de scipy, `n = 40 000`, `S` = tous les sites
échantillonnés, donc la version la plus généreuse) :

| `rho` | 2D, fini passe 1 | 3D, fini passe 1 |
|---|---|---|
| 4 | 5.6 % | 0.0 % |
| 8 | 11.9 % | 0.1 % |
| 16 | 24.2 % | 0.7 % |
| 32 | 38.8 % | 2.7 % |
| 64 | 52.6 % | 6.6 % |

**Mort en 3D**, et pas de peu : à `rho = 8`, 99.9 % des diracs ont au moins un vrai voisin dehors.
Ce n'est pas une constante à ajuster — en 3D un dirac a ~15.6 voisins et un agrégat de 8 n'en
contient structurellement pas assez. Il n'y aura pas de dirac fini d'avance ; la passe 3 touchera
tout le monde.

### L'AGRÉGAT A LE DEGRÉ D'UNE CELLULE, ET AUCUN ANNEAU N'EST COMPLET

Le nombre d'agrégats voisins **utiles** (contenant au moins un vrai voisin d'un membre) ne dépend
pas de `rho` : 6.0 en 2D, 16.9 en 3D, identiquement à `rho` = 8, 16 ou 32. C'est le degré de
Voronoï, et c'est normal — un agrégat est une cellule grossière, il a le degré d'une cellule.
Grossir les agrégats ne réduit pas le nombre de voisins, ça grossit chaque voisin.

D'où la formule qui résume le schéma : **l'agrégat amortit la recherche, jamais les coupes.** Le
nombre total de coupes est celui du diagramme, invariant. Ce qu'on achète, c'est de trouver les
candidats une fois par agrégat au lieu d'une fois par dirac.

Et l'adjacence du Voronoï grossier ne suffit pas à les énumérer :

| | utiles/agrégat | degré grossier | couvert par l'anneau 1 | par l'anneau 2 |
|---|---|---|---|---|
| 2D `rho`=8 | 6.02 | 6.00 | 90.9 % | 99.6 % |
| 3D `rho`=8 | 16.84 | 15.52 | 78.2 % | 99.1 % |
| 3D `rho`=32 | 16.87 | 15.57 | 82.0 % | 98.7 % |

**Aucun anneau fixe n'est complet.** L'anneau 1 rate 9 % des paires utiles en 2D et 22 % en 3D ;
l'anneau 2 en rate encore 0.4 % et 0.9 %. Toute construction de table qui s'appuie sur un anneau
est fausse, et c'est mesuré, pas supposé.

### LA PRÉCISION DU FILTRE ENCEINTE-CONTRE-CELLULE

Prototype 2D exact (cellules par intersection de demi-plans, SAT complet). Colonne « excès » =
agrégats retenus / agrégats utiles. Zéro rejet à tort sur ~150 000 paires : le filtre est sain,
comme la preuve le dit.

| | anneau 2 | → boîte | → SAT | utiles | excès |
|---|---|---|---|---|---|
| **2D** `rho`=2 | 19.68 | 5.95 | 4.88 | 3.89 | **×1.25** |
| **2D** `rho`=4 | 19.85 | 5.70 | 4.15 | 2.55 | ×1.63 |
| **2D** `rho`=8 | 19.69 | 5.73 | 3.82 | 1.72 | ×2.22 |
| **2D** `rho`=32 | 19.71 | 5.70 | 3.62 | 0.82 | ×4.44 |
| **3D** `rho`=2 | 84.4 | 26.6 | 15.12 | 10.80 | **×1.40** |
| **3D** `rho`=8 | 85.6 | 29.5 | 15.60 | 7.72 | ×2.76 |

**Le filtre est bon ; c'est l'agrégation qui coûte.** L'excès descend à ×1.25 quand `rho` → 2 et
croît régulièrement avec lui : tout l'excès est la pénalité de granularité, le fait de retenir ou
rejeter `rho` plans à la fois. Intrinsèque au schéma, pas améliorable par un meilleur filtre.

En unités du journal (un test de boîte ≈ une coupe tentée), 2D uniforme, le compte complet donne
un minimum vers `rho` = 3–4 à **62 unités contre 68 aujourd'hui**. Match nul sur nuage facile. Le
gain doit venir du cas dur (196 unités aujourd'hui) ou du rejeu entre itérations de Newton.

### LE C++ : `src/supercell/`, ÉTAPES 1 À 3

    Ordre.h          l'ordre spatial des germes (Morton, BSP)
    GrilleSites.h    « quel site est le plus proche ? »
    Agregats.h       ÉTAPE 1 : diagramme grossier, agrégats, médians
    Enceinte.h       enveloppe convexe, k-DOP fixe / orienté, boîte orientée
    Memoire.h        les trois stockages de la recette d'une cellule
    Table.h          ÉTAPE 3 : la table des enceintes qui se touchent
    mains/main_supercellules.cpp   les options et l'enchaînement, rien d'autre

Chaque étape porte SON bilan et SA ligne d'affichage. Trois aiguillages emboîtés (stockage,
enceinte, poids), tous à la compilation : le corps d'une étape ne contient pas un `if` sur le mode
choisi, donc mesurer un mode mesure bien ce mode.

**Étape 1**, `n = 10^6`, `rho = 8`, Morton : 2D `deg 5.98`, `rayon 1.05 h`, ordre 0.131 / gros
0.255 / affect 0.333 s. 3D `deg 15.08`, `rayon 0.91 h`, gros 3.397 / affect 1.412 s. Les degrés
sont ceux de Poisson-Voronoï — témoin gratuit que le diagramme grossier est ce qu'on croit.

**Morton contre l'ordre du BSP : les agrégats sont indiscernables.** Même rayon, même degré, même
distribution des tailles. Ce n'est pas surprenant après coup — l'agrégat est déterminé par *où
tombent les sites*, pas par l'ordre. Seul le coût diffère : Morton 2 à 3× moins cher, et sans
construire d'arbre sur les `n` germes pour n'en garder qu'une permutation.

**Le dirac médian** (le membre le plus proche du barycentre) remplace le site échantillonné comme
représentant. Il faut un vrai DIRAC et pas le barycentre : le plan doit être un demi-espace de
Laguerre, sans quoi la cellule cesse d'être un sur-ensemble.

**L'anneau 2 est le point fixe.** Avec les seuls voisins directs, l'opposition ne compte que ~6
médians en 2D : il reste des directions où un membre bat les six, et la sur-cellule part très loin.

| anneau | 1 | 2 | 3 |
|---|---|---|---|
| `somme des \|C_i\|`, 2D uniforme | **24.4** | 2.065 | 2.065 |

**Étape 2**, balayage en `rho`, anneau 2, `n = 10^6` (2D) et `5.10^5` (3D) :

| 2D `rho` | candidats | sommets (dont ext) | enveloppe | `S\|C\|` | `S\|env\|` | cellules | env. |
|---|---|---|---|---|---|---|---|
| 2 | 21.0 | 5.46 (5.12) | 6.58 | 1.422 | 1.525 | 0.620 s | 0.258 s |
| 4 | 23.3 | 5.36 (3.87) | 7.22 | 1.689 | 1.930 | 0.665 s | 0.199 s |
| 8 | 27.7 | 5.44 (2.70) | 7.85 | 2.075 | 2.472 | 0.771 s | 0.146 s |
| 16 | 36.0 | 5.55 (1.76) | 8.18 | 2.345 | 2.850 | 0.904 s | 0.094 s |
| 32 | 53.2 | 5.67 (1.16) | 8.74 | 2.627 | 3.255 | 1.187 s | 0.063 s |

| 3D `rho` | candidats | sommets (dont ext) | enveloppe | `S\|C\|` | `S\|env\|` |
|---|---|---|---|---|---|
| 2 | 83.1 | 23.13 (23.07) | 31.30 | 1.449 | 1.675 |
| 4 | 84.7 | 21.25 (20.21) | 36.86 | 2.087 | 2.721 |
| 8 | 88.5 | 20.61 (16.91) | 42.42 | 2.856 | 4.005 |
| 16 | 94.4 | 20.70 (13.12) | 46.67 | 3.557 | 5.148 |

**L'enveloppe convexe est en `O( 1 )`, pas en `O( |A| )`** — l'agrégat passe de 2 à 32 membres (×16)
et l'enveloppe de 6.58 à 8.74 sommets (×1.33). Confirmé indépendamment par le prototype Python.
L'inquiétude « autant de sommets que de points » ne se réalise pas.

**Le filtre des sommets extérieurs ne vaut pas la même chose selon la dimension** : en 2D il enlève
de 6 % (`rho`=2) à 80 % (`rho`=32) ; en 3D à petit `rho` il ne sert à rien du tout (23.07 sur 23.13
extérieurs à `rho`=2), parce qu'avec deux cellules par agrégat presque tout est du bord.

**Les trois stockages**, `n = 10^6`, `rho = 8`. Le rejeu coûte **5.44 coupes contre 27.7 candidats
essayés** en 2D, **12.1 contre 88.5** en 3D — sans une seule recherche.

| | 2D | perdus | 3D | perdus |
|---|---|---|---|---|
| `bits` | **8 o/dirac** | 0 | **16 o/dirac** | 6 297 (0.6 %) |
| `packe` | 17 o/dirac | 0 | 97 o/dirac | 18 803 (1.9 %) |
| `complet` | 1 472 o/dirac | — | 6 336 o/dirac | — |

La borne simple sur l'index **n'existe pas**, et le code l'avoue : `bits` sur un mot déborde en 2D
dès `rho`=32 (16 % de diracs perdus), il en faut deux en 3D où l'anneau 2 présente ~89 candidats.
`packe` en 3D coûte 97 octets pour la même information que 16 — il ne se justifie que si on veut la
combinatoire et pas seulement l'ensemble des coupes.

### LES ENCEINTES : QUATRE, ET UN RENVERSEMENT

`Enceinte.h` porte quatre réalisations derrière une interface unique (`build`, `taille`, `volume`,
`ecart`, plus `nb_axes` / `axe` / `bande` / `support` / `boite` pour l'étape 3). `volume()` est
SÉPARÉ de `build()` : c'est un témoin, pas un calcul de production, et le laisser dedans biaiserait
la comparaison (presque gratuit pour l'enveloppe, une découpe de cellule pour un k-DOP).

**Finesse** (`somme des |enc|`, `n = 2.10^5`, `rho` = 8) :

| 2D | uniforme | lignes / Voronoï | lignes / aires égales |
|---|---|---|---|
| enveloppe | 2.458 | 7.301 | 13.176 |
| k-DOP 4 fixe | +18.0 % | +76.4 % | **+146.8 %** |
| k-DOP 4 **orienté** | +17.9 % | **+44.0 %** | **+61.9 %** |
| k-DOP 8 fixe | +8.2 % | +36.9 % | +75.2 % |
| k-DOP 8 **orienté** | +8.3 % | **+27.8 %** | **+48.2 %** |

| 3D | uniforme | plans / Voronoï | plans / volumes égaux |
|---|---|---|---|
| enveloppe | 3.950 | 4.360 | 4.627 |
| k-DOP 7 fixe | +61.8 % | +88.8 % | **+155.8 %** |
| k-DOP 7 **orienté** | +61.3 % | **+79.0 %** | **+90.7 %** |
| k-DOP 13 fixe | +38.5 % | +56.0 % | +92.5 % |
| k-DOP 13 **orienté** | +38.5 % | **+51.6 %** | **+60.0 %** |

**Orienter coupe l'excès de moitié sur les cas durs et ne change rien sur l'uniforme** — c'est la
signature qu'on cherchait, pas un gain diffus. Les agrégats y sont allongés dans une direction que
la table fixe ne contient pas ; le repère propre (covariance, forme close en 2D, Jacobi en 3D)
coûte un passage de plus et une décomposition `O( 1 )`.

Règle utile : **orienter vaut à peu près un doublement de `K`, pour bien moins cher.** Quatre
directions orientées battent huit fixes en 2D (+62 % contre +75 %), sept orientées égalent treize
fixes en 3D (+91 % contre +93 %). Doubler `K` double le balayage ; orienter n'ajoute qu'un passage.

**Vitesse** : l'enveloppe est le poste dominant de l'étape 2 en 3D (10.47 s contre 5.3 s pour
*toutes* les cellules, `n`=5.10^5) ; un k-DOP la remplace pour **0.062 s**, soit **169×** moins.
En 2D 0.105 s contre 0.015 s. Et le k-DOP ne refuse jamais rien (`enc ko 0`) là où l'enveloppe
incrémentale 3D refuse 0.5 à 2 % des agrégats.

**LE RENVERSEMENT : la meilleure enceinte n'est pas le meilleur index.** Table par agrégat :

| 3D uniforme, `n`=2.10^5 | `S\|enc\|` | construction | table / agrégat |
|---|---|---|---|
| enveloppe convexe | **3.950** | 4.071 s | 73.93 |
| k-DOP 13 fixe | 5.469 | **0.014 s** | **61.20** |
| k-DOP 13 orienté | 5.356 | 0.030 s | 63.88 |
| boîte orientée, 13 repères | 8.785 | 0.081 s | 86.68 |

L'enveloppe, la plus fine, donne la table la plus longue. La raison est nette : dans le test de
séparation elle ne sait donner d'intervalle exact que sur les axes canoniques, donc `separe` s'y
réduit exactement au test de boîte et ne rejette rien de plus (son entonnoir montre
`boîte -> gardées` sans un seul rejet). **Sa finesse est dans ses sommets, pas dans des directions
qu'un SAT puisse consommer.**

Même leçon sur le k-DOP orienté, et corrigée : sans les axes canoniques, sa boîte ALIGNÉE est celle
de sa boîte ORIENTÉE, donc gonflée — la grille présentait 855 candidats par agrégat au lieu de 284
et la table finissait plus longue qu'avec des directions fixes. Il porte désormais `D` axes
canoniques **plus** `K` directions propres, et repasse devant partout sauf sur l'uniforme 3D à 13
directions (où les vecteurs propres d'un agrégat rond sont arbitraires et font double emploi).

### LA BOÎTE ORIENTÉE : ESSAYÉE ET REJETÉE COMME INDEX

L'idée était bonne et vaut d'être notée : une boîte orientée a ses `2^D` sommets par construction,
donc un support **exact dans toute direction** (le k-DOP doit majorer hors de ses axes), un volume
qui est un produit d'arêtes (plus de découpe, plus de tangence, plus de `nan`), et de vrais points
pour l'étape 4. On la choisit en balayant `M` repères candidats — le repère propre et ses rotations
— en un seul passage sur les points, et on garde le plus petit volume.

**Le balayage marche** : de `M`=1 (repère propre seul) à `M`=13, l'enceinte se resserre de 12.5 %
(10.043 → 8.785) et la table de 12 % (98.30 → 86.68), monotonement.

**Mais l'objet n'est pas compétitif** : +61 % de volume contre le k-DOP 13, +42 % de table, et ceci
MALGRÉ le SAT arête-contre-arête exact que ses 8 sommets autorisent (les six normales de faces sont
un test incomplet en 3D ; les neuf produits vectoriels le rendent exact). Ce qui coûte n'est pas la
façon de choisir le repère, c'est de n'avoir que `D` directions. Une boîte EST un k-DOP à trois
directions ; treize valent mieux et coûtent moins cher (0.014 s contre 0.081). S'y ajoute la boîte
alignée gonflée : 721 candidats présentés à la grille contre 258.

Et sur le fond — **l'étape 4 n'a pas besoin de sommets.** Son prédicat est « l'enceinte de `B`
rencontre-t-elle la cellule `C_i` ? », un SAT qui se contente de fonctions de support : exact des
deux côtés sur les directions du k-DOP, majoré sur les normales de faces de la cellule. Majorer là
ne coûte que quelques plans candidats de trop, jamais la correction. Les sommets ne deviendraient
indispensables que pour DÉCOUPER la cellule par l'enceinte, ce qu'on n'a pas à faire.

L'`Obb` reste dans le code : c'est l'objet le plus propre du lot (aucune fragilité numérique) et il
est prêt si l'étape 4 réclame de la vraie géométrie.

### L'ÉTAPE 3 : LA TABLE, PAR UNE GRILLE

Trois algorithmes étaient en balance, et le choix se raisonne avant de coder :

* **le tri 1D (sweep-and-prune)** : à écarter en 3D. La liste active le long d'un axe n'est pas un
  voisinage mais une TRANCHE — à 125 000 agrégats dans le cube unité, l'intervalle en `x` d'un
  agrégat en croise ~`na^(2/3)` ≈ 2 500 ;
* **le front par la connectivité grossière** : le graphe est gratuit depuis l'étape 1, mais son
  critère d'arrêt est le problème — aucun anneau fixe n'est complet (voir plus haut), et une règle
  géométrique saine demande un majorant GLOBAL du rayon que la dispersion des `|A|` rend très lâche ;
* **la grille régulière**, retenue : elle n'a besoin d'aucun majorant. Si deux enceintes se
  rencontrent elles partagent un point, donc une case — l'exactitude est STRUCTURELLE et non
  conditionnée à une constante bien choisie. Et les objets sont homogènes en taille (tous ~`rho`
  diracs), ce qui est le cas favorable. Pas pris sur la MÉDIANE des côtés de boîte : la
  distribution des `|A|` a une queue qui tirerait une moyenne.

Une quatrième piste reste ouverte : **le diagramme grossier est lui-même un index spatial**, mieux
adapté à la densité qu'une grille régulière, et l'ensemble des cellules grossières qu'un convexe
rencontre est CONNEXE, donc explorable par un front exact. Elle demande un test convexe-contre-
cellule que la grille n'a pas besoin d'écrire.

Le parcours est un **gather** et non un balayage de cases : pour `A` on traverse le contenu des
cases que sa boîte couvre, on marque chaque `B` vu avec le numéro `A`, et un `B` déjà marqué est
sauté. Le dédoublonnage est un test d'entier, la ligne CSR sort dans l'ordre, rien n'est alloué.

Coût : **0.15 s** pour 125 000 agrégats en 2D à `n`=10^6 (contre 0.75 s pour toutes les cellules),
0.95 à 1.13 s en 3D à `n`=5.10^5 (contre 5.3 s). La table n'est pas le poste dominant.

Taille : ~12 entrées par agrégat en 2D pour ~6 voisins réellement utiles, ~61 en 3D pour ~17
utiles — **un facteur 2 en 2D et 4 en 3D**. C'est le prix de l'enceinte, et c'est ce que l'étape 4
paiera. Et la table a une **queue lourde** : sur `lignes / Voronoï` un agrégat en touche jusqu'à
6 984 (4 388 avec le k-DOP orienté). Ce sont des diracs isolés dont la sur-cellule couvre une
grande région ; à 12 entrées de moyenne c'est invisible, mais l'étape 4 devra le prévoir.

### LES PIÈGES, tous trouvés par un témoin et pas par la lecture

1. **`cut_id` n'est pas l'indice accélérateur** mais l'indice du germe dans le nuage. Un commentaire
   affirmait le contraire ; il fallait la table inverse.
2. **Le minorant par plans est faux À L'INTÉRIEUR de l'enveloppe** (mesuré `+0.026` au centre,
   `-7.7e-08` dehors) : `ell_p` est concave en `p`, la preuve par les arêtes ne couvre pas
   `p = proj_K( x )` quand `x` est dans `K`. La pièce `K` de `S_A = K u (...)` est porteuse.
3. **Le dédoublonnage des sommets est nécessaire** : un sommet du bord de `U_A` est calculé par deux
   cellules voisines dans deux ordres de coupes, donc il revient à un ulp près. Gardés tels quels,
   ces jumeaux font une arête d'enveloppe de longueur 6e-20 dont la normale ne veut rien dire, et
   tout ce qui divise par elle explose.
4. **Le test de visibilité d'une enveloppe incrémentale 3D doit être invariant d'échelle.** Un seuil
   homogène à `ech^3` déclare « pas visible » une petite face que le point survole largement ;
   l'enveloppe repartait fausse de 4 à 7 `h`. En comparant une DISTANCE, plus des passes de
   rattrapage *et leur vérification*, l'écart tombe à 5e-12 `h`. La vérification manquait d'abord :
   les passes ne convergeaient pas et le code repartait quand même.
5. **`Cell3T` réserve les identifiants négatifs aux faces du domaine et exige qu'ils soient
   DISTINCTS.** Passer `-1` partout fait que deux faces opposées deviennent la même et le parcours
   des cycles part en vrille — on lisait 0.02 au lieu de 2.9.
6. **`CellSoAT::cut` lit `s[ nb - 1 ]` avant tout test** : l'appeler sur une cellule déjà vidée lit
   hors du tampon.
7. **Les plans tangents cassent l'invariant de `Cell`.** Avec `K` directions et peu de points,
   plusieurs bandes sont portées par le MÊME sommet ; la coupe le rogne de 1e-17 et insère deux
   points confondus. Au bout de quelques-unes, « les sommets dehors forment une plage cyclique
   unique » est faux, `cut` interpole entre deux sommets non adjacents, et le résultat est `nan`.
   Un desserrage de `1e-9 x taille` rend ces plans `unchanged` : 200 000 cas de test, plus un seul
   `nan`. **Ce n'est pas un défaut de `Cell`** — son invariant suppose des plans bien séparés, ce
   que garantit un diagramme de puissance mais pas un k-DOP.
8. **Un `return` là où il fallait un `continue`** dans la lambda de case de la grille : il sortait
   de la case entière au lieu de passer au candidat suivant, et la table perdait plus de la moitié
   de ses entrées (11.81 → 5.42) en ayant l'air parfaitement plausible. Trouvé par la force brute
   sur échantillon.
9. **Le rejeu se juge sur les FACETTES, pas sur l'aire.** Sur une cellule en lame l'aire est une
   différence de grands termes et son écart relatif (2.4e-09) ne dit rien de la justesse — zéro
   facette différait.
10. **La force brute de vérification doit appliquer LE MÊME test que la grille**, boîte comprise.
    Sans elle, elle garde des paires que la grille écarte à juste titre (`separe` est conservatif),
    et on compte comme fautive la plus fine des deux.
11. **`zsh` ne découpe pas les mots à l'expansion.** `--enceinte $e` avec `e="kdop --dirs 8"` passe
    UN argument, qui retombait silencieusement sur le défaut : deux campagnes de mesures entières
    perdues, avec des lignes assez ressemblantes pour passer pour des résultats. Le symptôme est
    reconnaissable — **des colonnes identiques au centième entre variantes**. `--enceinte` et
    `--stockage` refusent désormais une valeur inconnue.

### OÙ ÇA EN EST

Étapes 1 à 3 en C++, vérifiées : `somme des |C_i| >= 1`, `somme des |enc| >= somme des |C_i|`, le
rejeu rend les mêmes facettes, l'enceinte contient les points à 5e-12 `h` près, la table est
symétrique et coïncide avec la force brute sur ~300 agrégats échantillonnés. Tout `ok` sur les six
nuages de la suite, en 2D et en 3D.

Réglages retenus : **Morton**, **anneau 2**, **`bits`** (deux mots en 3D), **k-DOP 13 fixe** ou
**orienté** selon l'anisotropie du nuage.

Reste l'étape 4 : rejouer la cellule de base depuis sa recette, et ne l'opposer qu'aux agrégats de
la table dont l'enceinte la rencontre encore. C'est là que se joue la comparaison avec les 68
unités du BSP — et le témoin y sera le plus fort possible, puisque la somme des mesures devra
valoir 1.
