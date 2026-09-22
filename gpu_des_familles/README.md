# `gpu_des_familles` — le banc GPU

Le troisième banc. [`2d_des_familles`](../2d_des_familles/README.md) a cherché comment construire
un diagramme de puissance vite ; [`solvers_des_familles`](../solvers_des_familles/README.md) a
rangé l'engin qui a gagné et s'en sert pour essayer des solveurs. Ici on prend **le même arbre et
les mêmes nuages**, et on refait les cellules **en CUDA** — pour savoir comment mapper une cellule
sur un GPU, et ce que ça rapporte.

Le moteur CPU est le **témoin** : même nuage, même arbre, même flottant de noyau, l'écart maximal
par cellule et la somme des mesures sont imprimés à chaque ligne. Rien d'autre n'est vérifié, et
c'est suffisant : une coupe ratée ou une coupe en trop se voit sur la cellule.

```
xmake f -m release --cuda=/usr/local/cuda-13.3 && xmake      # nvcc 13.3 ( 13.4 s'effondre sur les noyaux générés )
xmake run mesures --threads 8                                 # la suite, 2D puis 3D, CPU témoin puis GPU par variante
xmake run mesures --threads 8 --kernel float --3d             # le flottant du noyau, des deux côtés
xmake run mesures --threads 8 --variante voies --load uniforme -n 1000000
```

Les options communes (`-n`, `--load`, `--kernel`, `--maxnv`, `--leaf`, `--cases`, …) sont celles
de `solvers_des_familles` (`src/bench/Args.h`), plus `--variante fil | filreg | filregc | filmix{4,6,8,12,16} | filbrk{6,8,10,12,16} | voies | voies16 |
voies32 | paquet{8,32}x{1,2,4}[S] | toutes` et `--reps-gpu`. Sur la machine, `--threads 8` est à donner (`hardware_concurrency` rend 1
dans le bac à sable) et **tout chronométrage passe par `job -b`**.

---

# 1. OÙ EST QUOI

```
src/gpu/Arbre.cuh       l'arbre tel que le GPU le lit : noeuds AoS alignés sur 16 octets ( boîte,
                        majorant affine, tranche, fils droit ), germes dans l'ordre de l'arbre,
                        le plan bissecteur, le test d'élagage pour UN sommet, la proximité
src/gpu/Fil2D.cuh       UNE CELLULE PAR THREAD, 2D : `coupe_large` mot pour mot, la pile en local
src/gpu/FilReg2D.cuh    UNE CELLULE PAR THREAD, LES SOMMETS EN REGISTRES, 2D : le noyau à registres
                        du CPU exécuté par un thread scalaire, l'excursion au-delà de huit
src/gpu/FilMix2D.cuh    le même, `R` sommets en registres et la queue en mémoire par une boucle
                        ordinaire, sans excursion — le gagnant sur les lignes / Laguerre à `R = 6`
src/gpu/FilBrk2D.cuh    tout en registres, tout déroulé, chaque boucle SORT à `nb`, et les cellules
                        qui dépassent `R` refaites en SECONDE PASSE — LE GAGNANT en uniforme à `R = 8`
src/gpu/Voies2D.cuh     LA CELLULE SUR LES VOIES, 2D : voie = sommet, `V` = 8, 16 ou 32 voies par
                        cellule ; le débordement est une excursion sur la voie 0
src/gpu/Paquet2D.cuh    PLUSIEURS CELLULES PAR VOIE, un parcours par warp, les plans d'une feuille
                        testés en bloc — mesuré, et perdu (§ 4)
src/gpu/Fil3D.cuh       une cellule par thread, 3D : `Cellule3` mot pour mot, tout en mémoire locale
src/gpu/Voies3D.cuh     LA CELLULE SUR LE WARP, 3D : la voie `l` porte les sommets `l, l+32, …`
                        dans `S` cases de registres, coupes et voisins en octets ; deux passes
src/gpu/Mesures.h/.cu   `DiagrammeGpu<D,TK>` : téléversement, le choix du noyau ( variante,
                        Voronoï / Laguerre, sommets max ), le chrono par événements CUDA
src/mains/main_mesures.cpp   le banc
```

Ce qui est repris de `../solvers_des_familles/src` sans y toucher : `accel/AaBsp.h` (l'arbre,
bâti sur l'hôte), `bench/` (les nuages, les options, le dispatch), et tout `cell/` +
`diagram/PowerDiagram.h` pour le témoin. Le `.cu` est compilé par nvcc, le `main` et le témoin par
g++ avec `-march=native` : la frontière est `Mesures.h`, qui ne contient pas un type CUDA.

---

# 2. LES MAPPAGES

Un GPU exécute 32 voies en lock-step. Une cellule de Voronoï est une suite de coupes dont chacune
dépend de la précédente, sur un parcours d'arbre qui dépend de la cellule. Deux façons de mettre
ça sur des voies :

**`fil` — une cellule par thread.** Chaque voie est un petit cœur scalaire qui fait SA cellule du
début à la fin, comme un fil CPU : les sommets dans des tableaux locaux, la pile du parcours
locale, la coupe scalaire en place. C'est ce que fait le chemin GPU de `sdot` (« la boucle nue en
mémoire »). Les 32 cellules d'un warp sont voisines dans l'arbre, donc leurs parcours se
ressemblent ; rien d'autre ne limite la divergence.

**`filreg` — une cellule par thread, les sommets en registres.** Le noyau à registres du CPU
(`cell/Noyau2D.h`) exécuté par UN thread scalaire : huit `x`, huit `y` dans des registres (tout
est déroulé, une lecture à indice dynamique est une chaîne de `select`), `nb` un entier, la coupe
**sans boucle ni branche** — masque de signe, ses deux bouts par `ffs`, deux intersections, et le
remontage par huit `select` à huit entrées. Les `cid` en mémoire locale (`filreg`) ou en
registres aussi (`filregc`). Au-delà de huit sommets, l'excursion : la cellule se pose en mémoire
locale et `coupe2` continue en scalaire jusqu'au retour à huit.

**`filmix` — `R` sommets en registres, la queue en mémoire.** La généralisation : les sommets
`0 .. R-1` en registres et déroulés, les sommets `R .. nb-1` dans des tableaux locaux parcourus
par une boucle ordinaire — qui diverge entre les threads du warp mais ne fait pas plus de travail,
et ne fait rien pour une cellule qui tient dans ses registres. Plus d'excursion, plus de second
mode : la coupe est une seule suite d'instructions, `nb` va jusqu'à 64 (masques sur 64 bits). `R`
règle un compromis : le code déroulé fait toujours `R` cases, occupées ou non ; la queue coûte
une boucle divergente et des chargements à la place des `select`.

**`filbrk` — tout en registres, tout déroulé, avec des sorties, et une seconde passe.** `R`
sommets en registres et chaque boucle déroulée sort dès que `i >= nb` (`break` dans la boucle
déroulée : une branche par case, mais pas de travail sur une case vide, et le warp n'exécute que
jusqu'au plus grand `nb` de ses voies). Une cellule qui dépasserait `R` sommets n'est pas gérée :
son rang est poussé dans une liste (atomique) et une **seconde passe**, `filmix` à 8 registres et
64 sommets, refait ces cellules-là entre elles.

**`voies` — la cellule sur les voies.** Le pendant CUDA du noyau à registres du CPU : la voie `l`
porte le sommet `l`, le nombre de sommets est un scalaire uniforme, et la coupe est le même calcul
sans boucle — masque de signe par `ballot`, ses deux bouts par `ffs`, deux intersections en une
division, deux permutations par `shfl`. Le test d'élagage coûte une opération par voie et un
`ballot`. En 2D une cellule finie a six sommets et 98 % des états intermédiaires en ont huit ou
moins : **8 voies par cellule, quatre cellules par warp** (`voies`), ou 16, ou 32 (`voies16`,
`voies32`) pour mesurer le compromis. Au-delà de `V` sommets, l'excursion du CPU : la cellule se
pose dans des tableaux locaux de la voie 0, qui coupe en scalaire pendant que les autres attendent,
et remonte sur les voies dès qu'elle redescend.

En 3D une cellule a vingt à quarante sommets : **une cellule par warp**, la voie `l` portant les
sommets `l, l + 32, l + 64, l + 96` dans `S` cases de registres (indexées à la compilation, jamais
en mémoire locale). Le polyèdre est celui du CPU — trois coupes et trois voisins par sommet, le
voisin `j` en face de la coupe `j` — les trois coupes dans un mot (un octet chacune), les trois
voisins dans un autre. Ce que le warp fait ensemble à chaque coupe :

* la première passe : un `fma` par case et un `ballot` par case — les masques des sommets dehors
  SONT l'état de la coupe ;
* les sommets neufs : un **candidat par voie** (un sommet dehors, une de ses trois arêtes), gardé
  si l'autre bout est dedans, compacté par `ballot` / `popc` ; le neuf `j` vit dans la voie `j` le
  temps de la coupe ; leurs voisins entre eux par une boucle sur les neufs ;
* la **renumérotation par masques** : pas de trous à boucher (l'astuce du CPU pour ne pas
  toucher les survivants n'a pas de sens quand toutes les voies travaillent de toute façon), les
  survivants sont compactés dans l'ordre, les neufs mis à la suite, et le nouveau numéro d'un
  sommet est le `popc` du masque des survivants sous lui — les `ballot` sont la table, aucune
  mémoire partagée ;
* chaque voie rassemble ses nouveaux sommets par `shfl` (une par case source et par champ) et
  recolle les voisins : un octet qui nommait un sommet dehors nomme le neuf né sur cette arête.

Le volume est la somme des tétraèdres `( g, o_f, a, b )` sur les arêtes et leurs deux faces, `o_f`
le plus petit sommet de la face (`atomicMin` partagé) — équivalent à l'accumulation par face du
CPU sur un polygone convexe, sans accumulateur.

**`paquet` — plusieurs cellules par voie, un parcours par warp (2D).** La voie `l` porte le
sommet `l` de `K` cellules (des cases de registres), un warp porte `G K` cellules consécutives
dans l'arbre, le parcours est partagé (une boîte est descendue si une cellule du paquet peut
encore être coupée), les deux fils d'un nœud sont testés ensemble (`S`) ou chaque nœud à sa
sortie de pile, et les plans d'une feuille sont testés **en bloc** : un `fma` par (germe, case),
un bit par couple qui coupe, une réduction OU du warp, et seuls ces couples passent par le code
de coupe, exécuté sans branche pour tous les groupes (prédiqué). L'idée : tard dans le parcours il
n'y a plus que des coupes inutiles, et là tout est du SIMD pur. Mesuré et perdu, § 4 dit pourquoi.

**Deux passes.** À deux cases par voie (64 sommets) 0,02 % des cellules débordent en cours de
route (223 sur 10⁶ en uniforme) ; à quatre cases le code et les registres doublent et tout le
monde paie (681 contre 247 ns/germe). Donc : deux cases pour tout le monde, les rangs qui ont
débordé poussés dans une liste par atomique, et une seconde passe à quatre cases sur cette liste.
La cellule n'étant écrite qu'à la fin d'une coupe, une cellule qui déborde est restée intacte
jusqu'au débordement et repart de zéro proprement.

---

# 3. LES CHIFFRES

RTX 2080 Ti (Turing, 68 SM, 11 Go, **FP64 au 1/32**), CUDA 13.3. Le témoin : Xeon W-2145, 8 fils
épinglés. Machine seule (`job -b`), tour de chauffe puis minimum de 10 répétitions au GPU, 3 au
CPU ; le chrono GPU est celui du noyau seul (événements CUDA), le téléversement et la descente des
résultats sont comptés à part. Mêmes nuages que les deux autres bancs. Les vitesses sont contre le
**CPU à 8 fils**, dans le même flottant.

## `float`, ce pour quoi cette carte est faite

| | n | CPU 8 fils | `fil` | `filreg` | `filregc` | `filmix6` | `filbrk8` | `voies` 8 | `voies16` | `voies32` |
|---|---|---|---|---|---|---|---|---|---|---|
| 2D uniforme | 10⁶ | 145 ns/germe | 68 (×2.1) | 16 (×8.9) | 15 (×9.7) | 13.2 (×11) | **9.8 (×15)** | 29 (×5.0) | 34 | 34 |
| 2D lignes / Voronoï | 10⁵ | 141 | 63 (×2.3) | 23 (×6.1) | 21 (×6.7) | 20.0 (×7) | **20.5 (×7)** | 36 (×4.0) | 32 | 32 |
| 2D lignes / aires égales | 10⁵ | 696 | 188 (×3.7) | 61 (×11.5) | 51 (×13.6) | **45.5 (×15)** | 50 (×14) | 106 (×6.6) | 108 | 128 |
| 3D uniforme | 10⁶ | 1831 | 3855 (×0.5) | — | — | — | — | **229 (×8.0)** | — | — |
| 3D plans / Voronoï | 10⁵ | 1774 | 3458 (×0.5) | — | — | — | — | **231 (×7.7)** | — | — |
| 3D plans / volumes égaux | 10⁵ | 3054 | 4651 (×0.7) | — | — | — | — | **405 (×7.5)** | — | — |

(chiffres 2D repris après l'ajout d'une chauffe de 300 ms avant le premier chrono, § 6 : sans elle
le premier noyau d'un banc tournait à fréquence réduite, 17 au lieu de 13.)

**Les micro-optimisations de `filbrk`** (`filbrk8nu` est sans) : une cellule non vide a trois
sommets au moins et une coupe en laisse `nb_in + 2 ≥ 3`, donc les trois premières cases ne
testent pas `i < nb` ; et `__builtin_expect` d'après les compteurs (58 % des plans ne coupent
pas, une cellule vide, un débordement, son propre germe sont rares) : 10.2 → 9.8, 22.2 → 20.5,
52 → 50, soit −3 à −7 %. La taille de feuille de l'arbre (`--leaf`) : 6 → 9.7 / 21.1 / 52.5, 8 →
9.9 / 21.0 / 51.0, **10** → 9.8 / 20.5 / 50, 12 → 10.0 / 23.4 / 53.2 ; le 10 du CPU tient.

`filbrk` selon `R` (uniforme / lignes Voronoï / lignes Laguerre, float, sans les
micro-optimisations) : `R = 6` 13 / 21 / 57,
`R = 8` **10 / 21 / 52**, `R = 10` 11 / 30 / 61, `R = 12` 13 / 47 / 86, `R = 16` 17 / 55 / 114 ;
la seconde passe reçoit 10 % des cellules à `R = 8` (elles dépassent huit sommets *en cours de
route*, même si 98 % des états sont à huit ou moins), 0,03 à 1 % à `R = 12`. Et pourtant `R = 12`
perd sur les lignes en envoyant moins de cellules en seconde passe : **la seconde passe regroupe
les grandes cellules entre elles**. Dans un warp, une cellule à douze sommets et trente coupes
fait attendre trente et une voies ; renvoyée à une passe où toutes ses voisines lui ressemblent,
elle ne coûte que sa part. C'est la vraie raison du gain de `filbrk8`, plus que les cases vides
— et une piste : trier les cellules par taille attendue (dans Newton, celle du diagramme d'avant)
pour que chaque warp soit homogène.

`filmix` selon `R` (uniforme / lignes Voronoï / lignes Laguerre, float) : `R = 4` 31 / 38 / 83,
`R = 6` **13 / 20 / 45**, `R = 8` 14 / 22 / 47, `R = 12` 28 / 36 / 68, `R = 16` 56 / 70 / 107 ; en
double `R = 6` 93 / 112 / 385 contre 114 / 131 / 472 pour `filregc` (−18 %). Aucun ne déborde ses
registres (59, 71, 77, 99, 123 registres) : `R = 4` perd parce que la queue en mémoire est chaude
pour presque toutes les cellules (nb ≥ 5) et sa boucle divergente remplace des `select` par des
chargements ; `R ≥ 12` perd par le déroulage lui-même — chaînes de `select` à 12 ou 16 entrées,
occupation en baisse — pour des cases presque toujours vides.

(en 3D, `voies` est le warp entier, en deux passes.) Les paquets, 2D uniforme : `paquet8x1` 34,
`paquet8x2` 83, `paquet8x4` 272, `paquet32x1` 41, `paquet32x2` 47, `paquet32x4` 95, et les mêmes
avec le test des deux fils (`S`) 38 / 48 / 112 — tous derrière `voies`, et derrière `filreg` de
loin ; sur les lignes les paquets à `K ≥ 2` débordent même les 64 sommets de l'excursion. Le téléversement d'un arbre à 10⁶ germes
coûte 20 ms en 3D, 140 ms en 2D avec le premier contexte CUDA ; la descente des 10⁶ mesures, 1 à
5 ms.

## `double`

| | n | CPU 8 fils | `fil` | `filreg` | `filmix6` | `voies` |
|---|---|---|---|---|---|---|
| 2D uniforme | 10⁶ | 138 | 160 (×0.9) | 116 (×1.2) | **93 (×1.5)** | 186 (×0.7) |
| 2D lignes / Voronoï | 10⁵ | 137 | 174 (×0.8) | 131 (×1.0) | **112 (×1.2)** | 198 (×0.7) |
| 2D lignes / aires égales | 10⁵ | 688 | 688 (×1.0) | 493 (×1.4) | **385 (×1.8)** | 871 (×0.8) |
| 3D uniforme | 10⁶ | 1975 | 5648 (×0.3) | — | **1103 (×1.8)** |
| 3D plans / Voronoï | 10⁵ | 1908 | 5435 (×0.4) | — | 1066 (×1.8) |
| 3D plans / volumes égaux | 10⁵ | 3302 | 8107 (×0.4) | — | 2926 (×1.1) |

Le `double` coûte 5× au GPU (1/32 du débit FP64, mais les noyaux ne sont pas bornés par le
flottant, § 4) là où il coûte 4 % au CPU. Sur cette carte le GPU est une histoire de `float` ; et
`float` est ce que `sdot` fait en production — le plancher du résidu de Newton est alors celui de
la géométrie (`solvers_des_familles`, § 4).

**Exactitude.** En `double` l'écart GPU / CPU par cellule est 2e-10 en 2D et 7e-14 en 3D (l'ordre
des opérations), la somme est 1 à 1e-9. En `float` l'écart est le bruit du flottant lui-même —
une cellule de côté 1e-3 avec des sommets à 6e-8 près a son aire à 4e-4 près, et les deux côtés
ne contractent pas les `fma` pareil — la somme est tenue à 1e-6. Aucune cellule ne déborde après
la seconde passe.

---

# 4. CE QUE LE PROFIL DIT (`ncu`)

**`fil` ne remplit rien.** En 3D, **2,3 threads actifs par warp sur 32** : le warp exécute
l'union des chemins de 32 cellules qui ne coupent ni au même moment ni de la même façon, et la
mémoire locale (12 Ko par thread) sort du L1 (12 % de succès) vers la DRAM (224 Go/s pour 6 % de
calcul). En 2D, 3,9 threads actifs, L1 à 49 %. C'est le mappage de `sdot` sur GPU aujourd'hui, et
c'est pour ça qu'il vaut ×2 en 2D et rien en 3D.

**`voies` en 2D : 10 threads actifs par warp**, L1 à 91 %, DRAM à 1 %, calcul à 48 %, 6 000
instructions par cellule. Ce qui borne est la divergence entre les quatre groupes d'un warp (dix
actifs sur trente-deux, soit 1,3 groupe sur 4 en moyenne) et la latence (12 cycles par instruction
émise par warp). `voies32` — 26 voies dorment, zéro divergence — fait **jeu égal** avec `voies8` :
le noyau n'est pas borné par les voies qui dorment mais par la latence de la chaîne nœud → test →
plan → coupe.

**`filreg` en 2D : 3 000 instructions par cellule**, la moitié de `voies`, 6 threads actifs par
warp, 85 % d'occupation, calcul à 47 %. Ce qui a changé par rapport à `fil` (68 → 16 ns) : plus
une lecture de sommet en mémoire (L1 passait de 49 % de succès à rien à lire), et une coupe en
**ligne droite** — des `select`, pas une boucle sur `nb` ni un `if` par cas de plage — donc les 32
cellules d'un warp, même désynchronisées, exécutent le même flot d'instructions avec des prédicats
différents. La divergence qui reste est celle du parcours (quel nœud, feuille ou pas, combien de
germes). C'est le meilleur mappage 2D, et de loin ; le `cid` en registres (`filregc`) rapporte 5 à
15 % de plus là où il y a beaucoup de coupes. Plafonner à 128 registres n'apporte rien (64 sans
débordement).

**Les paquets, et pourquoi ils perdent** — les compteurs par cellule (uniforme 10⁶, float) :

| | plans testés / tentés | coupes effectives | excursions | boîtes testées |
|---|---|---|---|---|
| `voies8` (un parcours par cellule, test au pop) | 29.7 | 12.5 | 0.14 | 47 |
| `paquet32x1` (batch de plans, test au pop) | 19.2 | 12.7 | 0 | — |
| `paquet32x1S` (batch, les deux fils au push) | 19.2 | 12.7 | 0 | 78 |
| `paquet8x1` (4 cellules par warp, un parcours) | 50.3 | **19.5** | 0.40 | — |
| `paquet8x4` (16 cellules par warp) | 83.0 | **34.3** | 0.96 | — |

* **Tester les deux fils au push** descend 78 boîtes au lieu de 47 : au push la cellule est
  encore grande, au pop chaque coupe faite entre-temps rend le « non » plus probable. C'est la
  règle du CPU (`FournisseurBsp2D.h`), elle vaut au GPU.
* **Partager le parcours** présente à chaque cellule les plans les plus proches *du paquet* et non
  les siens : ils coupent le grand carré initial pour rien — les coupes effectives passent de 12,5
  à 19,5 par cellule à 4 cellules par warp, à 34 à 16 — la cellule grossit, déborde des 8 voies
  (excursions ×3 à ×7), et sur les lignes déborde même les 64 sommets de l'excursion. Le paquet
  coûte plus de travail qu'il n'économise de latence.
* **Le test en bloc** marche pour ce qu'il fait (19 tentatives au lieu de 30 tests, 12,7
  effectives dans les deux cas) et perd quand même (41 contre 34 ns à `V = 32`) : un test de plan
  coûte un `fma` et un `ballot`, la réduction 64 bits par feuille et les deux passes sur ses germes
  coûtent plus que les dix tests évités. Le nearest-first par cellule avec test au pop est déjà
  le minimum de travail ; ce qu'il reste à cacher est de la latence, et ça se cache par
  l'occupation (`filreg`, 85 %), pas en ajoutant du travail.

**`voies` en 3D : 27 threads actifs par warp**, L1 à 98 %, DRAM à 0,2 %. La première version
faisait 66 000 instructions par cellule avec 9,6 cycles par instruction émise, dont **4 à attendre
qu'on lui cherche son instruction** (« No Instruction ») : le code est énorme (tout est déroulé
sur les cases), 191 registres, 25 % d'occupation, et huit warps sur des chemins différents font
tomber le cache d'instructions. Trois corrections, aucune ne touchant l'algorithme :

* `__fns` (« le n-ième bit à 1 ») n'est **pas une instruction** sur sm_75 mais une boucle
  logicielle : 7 % des instructions à elle seule, 12 % avec `nieme`. Remplacée par une dichotomie
  sur `popc`, cinq étages de `select` (`bit_nieme`) ;
* `rassemble` (la valeur d'un champ pour un sommet quelconque) gardait ses `shfl` sous un
  `if ( case en usage )` : une branche et sa barrière de reconvergence par case, 4 % des
  instructions. À deux cases, faire toutes les `shfl` coûte moins ;
* les `clamp` du test d'élagage et de la proximité sortaient en branchements ; `fminf` / `fmaxf`
  donnent `FMNMX`.

Puis deux cases au lieu de quatre (deux passes) : 901 → 348 → 247 ns/germe ; et le plafond à 128
registres (`__launch_bounds__( 128, 4 )`, quatre blocs par SM au lieu de trois) : 216 en séance,
229 machine seule (à 102 registres ça déborde en mémoire locale : 251 ; à 85 : 343).

Où vont les 49 000 instructions par cellule qui restent (uniforme, `float`) : le parcours et les
tests d'élagage ~14 000, la première passe des ~90 plans testés ~4 000, et les ~30 coupes
effectives ~25 000 — **800 instructions par coupe**, dont 10 000 en tout dans `rang` / `nieme` /
`sel` (les petits helpers de la renumérotation) et 6 000 dans le nouvel état. Le SASS est à 30 %
de branchements et barrières (`BRA`, `BSSY`, `BSYNC`, `BMOV` : chaque petit `if` divergent en
coûte trois), 8 % de `shfl`, 11 % de flottant. Six cycles par instruction émise, 28 % d'occupation
réelle.

---

# 5. CE QUI RESTE

* **`filbrk8` en uniforme (10 ns/germe, ×14), `filmix6` sur les lignes / Laguerre (46)** sont
  les références 2D ; `filreg` / `filregc` à 8 registres et une excursion sont derrière. La piste
  suivante est celle que la seconde passe a révélée : **des warps homogènes en taille de
  cellule** (§ 3), par un tri des cellules sur le nombre de sommets du diagramme précédent (dans
  Newton on l'a gratuitement) — à faire. Ce qui reste : la divergence du
  parcours entre les 32 cellules d'un warp (6 actifs) — un tri des cellules par profondeur de
  parcours ou une pile en mémoire partagée ne changeraient pas le fond ; les 3 000 instructions
  par cellule sont à lire ligne à ligne comme pour le 3D.
* **Les paquets, le test en bloc et les deux fils au push sont mesurés et perdent** (§ 4) : ne pas
  y revenir sans une idée qui réduise les coupes transitoires.
* **La coupe 3D à 800 instructions.** Les survivants en place plutôt que renumérotés (moins de
  `rassemble`, mais les trous du CPU à gérer par masques) ; les helpers `rang` / `nieme` appelés
  moins de fois (le nouveau numéro d'un voisin calculé une fois par sommet et non par octet) ;
  moins de petits `if` (chaque `select` sauve trois instructions de convergence).
* **Les facettes**, pour Newton : `cid` est déjà porté par les voies, il manque les aires de faces
  (l'accumulateur par face, en mémoire partagée) et une sortie `( i, j, aire )` par atomique ou par
  compactage.
* **Les poids neufs sans re-téléverser** : `refresh_weights` a son pendant naturel (les majorants
  par nœud, un thread par nœud) ; l'arbre lui-même reste bâti sur l'hôte (`notes/2026-09-02-arbre-morton.md`
  dit pourquoi et par quoi le remplacer sur GPU).
* **`double`.** Sur une carte à FP64 plein (A100 / H100) le rapport `float` / `double` reviendrait
  à celui du CPU ; ici on ne le saura pas.

---

# 6. COMMENT LES CHIFFRES SONT PRIS

Avant chaque chrono, 300 ms de noyau en boucle : le CPU (l'arbre, le témoin) laisse le GPU
redescendre en fréquence, et un seul tour de chauffe ne le remonte pas — le même noyau donnait
17 puis 13 ns/germe selon qu'il passait premier ou second. Puis 10 tours, minimum.

`job -b -- ./build/linux/x86_64/release/mesures --threads 8 --reps 3 --reps-gpu 10 --kernel float`
puis la même en `double` — machine seule, rien d'autre ne tourne. Un `PowerDiagram` CPU bâtit
l'arbre ; le témoin mesure (chauffe, minimum de 3) ; `DiagrammeGpu` reçoit l'arbre ; chaque
variante fait un tour de chauffe puis 10 tours chronométrés par événements CUDA autour du (ou des
deux) noyau(x), et le minimum est gardé. Les profils sont pris avec `ncu` (CUDA 13.3) sur la
première passe (`--kernel-name-base demangled --kernel-name "regex:voies<\(bool\)0, \(int\)2, float>"`)
et la répartition par ligne source vient de `--page source --print-source sass,cuda --csv`,
agrégée par un petit script.
