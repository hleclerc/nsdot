# `2d_des_familles` — le banc 2D, en C++ nu

Ni jax, ni AdaptiveCpp, ni FFI. Un `xmake`, un binaire, `-O3 -march=native`, des `std::thread`.
L'intention est d'essayer une idée en trente secondes au lieu de trois minutes de compilation SYCL,
et de pouvoir lire l'assembleur qui en sort. **Ce n'est pas la bibliothèque** : ce qu'on y mesure
doit ensuite être reporté dans `sdot`.

```
xmake f -m release && xmake
xmake run pd2d --check                       # contre le balayage complet
xmake run pd2d -n 1000000 --threads 8
xmake run pd2d --help
```

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

* `Cell.h` — `CellSoA`, 2D, **32 sommets au maximum**, aucune allocation. `cut` est EN PLACE
  (seuls les deux points d'intersection sont écrits, plus le décalage), `measure` est un lacet.
  L'invariant qui rend tout ça possible : *la coupe `i` porte l'arête `[ v_i, v_i+1 ]`*.
* `AaBsp.h` — le BSP médian, numéroté **en préordre** (fils gauche en `n+1`, fils droit stocké),
  un nœud dans une ligne de cache. Plus `EverySeed`, l'oracle en force brute.
  Chaque accélérateur **porte son parcours** (`for_each_candidate`) : le concept `Walk` séparé
  décrivait un arbre binaire à boîtes, donc ne se généralisait à rien — la grille ne pouvait pas
  l'implémenter, et son parcours ne pouvait pas s'appliquer à un arbre. Un accélérateur ne doit
  qu'une chose à l'appelant, PROPOSER DES GERMES, et c'est sa seule méthode. Le parcours
  meilleur-d'abord (file de priorité, ce que fait `SpZGrid`) est parti avec le concept : il avait
  été mesuré perdant, 0.279 s contre 0.246.
* `WeightMajorant.h` — le majorant **affine** des poids d'une région, et le choix nœud par nœud
  entre lui et le constant.
* `AaBspPacked.h` — le même arbre, tout dans une arène, les points collés à leur feuille.
* `AaBsp4.h` — le même arbre à QUATRE fils par nœud (`bsp4` / `bsp4l`), mesuré et rejeté.
* `ObBsp.h` — le même arbre à coupes NON ALIGNÉES et boîtes alignées (`obsp`), mesuré et rejeté.
* `Newton.h` — le problème de transport RÉSOLU : Newton amorti, laplacien de Laguerre,
  jauge `w_0 = 0`, et le solveur linéaire (AMGCL / Eigen / gradient conjugué maison).
* `Grid.h` — la grille régulière (tri par comptage, germes groupés par case) et son parcours en
  deux temps, décrit plus haut.
* `parallel.h` — `parallel_for` avec découpe `blocks` ou `strided` et épinglage. Mesuré : aucune
  différence entre les deux (0.245 contre 0.244 s).
* `PowerDiagram.h` — le pilote, templaté sur `Cell` et `Accel`, plus `CellBox`, `SkipInside`,
  `Stats` et `Weighted` — tous des paramètres de TEMPLATE, pour qu'aucun ne survive à la
  compilation quand il ne sert pas.

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

## Ce qui reste à essayer

`CellAoS` pour comparer avec `CellSoA` ; les boîtes de nœud en FP32 (deux nœuds par ligne de
cache, et ça rendrait aussi les 8 octets que le majorant affine vient de prendre) ; du SIMD
explicite sur le test d'éviction, qui ne se vectorise pas à cause de son `return` anticipé ; une
grille / quadtree en ordre Z à côté du BSP ; scinder `build( X, Y, W )` en `build( X, Y )` +
`refresh_weights( W )`, parce que dans une résolution de transport optimal les positions ne bougent
pas et seuls les majorants changent d'une itération de Newton à l'autre.

Et surtout : **reporter dans `sdot` ce qui a payé ici**, à commencer par la question de fond que ce
banc a tranchée — pourquoi le même algorithme y monte à 7.6× et n'y monte qu'à 2.1× sous SYCL.
