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
de `solvers_des_familles` (`src/bench/Args.h`), plus `--variante fil | filreg | filregc | filmix{4,6,8,12,16} | filbrk{6,8,10,12,16} | filbrk8nu | filrot{6,8} | filnrm8 | filord8 | filsuc8 | filmsk8 | filmsk8c{6,8} | filnrm8c{6,8} | filuni8 | filuni8np | filshm8 | filnrm8tri | filnrm8tril | filph8 | filph8g | filph8b | filph8a | filph8c | filph8o | filph8m | filph8m4 | voies | voies16 |
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
                        qui dépassent `R` refaites en SECONDE PASSE
src/gpu/FilRot2D.cuh    le même avec le remontage par DÉCALAGE EN BARILLET au lieu de lectures
                        indexées
src/gpu/FilNrm2D.cuh    la cellule NORMALISÉE AVANT la coupe : tout se lit à des positions fixes,
                        rien n'est recomposé après — LE GAGNANT en 2D à `R = 8`
src/gpu/FilOrd2D.cuh    LES SOMMETS NE BOUGENT PLUS : un registre de 64 bits porte l'ordre
                        cyclique ( un octet = le slot, en one-hot ) — −21 % de registres, à
                        vitesse égale (§ 4)
src/gpu/FilMsk2D.cuh    registres TRIÉS, la frontière par quatre masques de rôle et le remontage
                        en UN SEUL barillet — le plus petit noyau à registres triés : 96 / 59
                        registres contre 128 / 74, pour +6 % de temps (§ 4)
src/gpu/FilSuc2D.cuh    TOUT EN MASQUES : la cellule est une relation de succession ( deux
                        registres `succ` / `pred` ), plus une position ni un index — mêmes
                        registres, même vitesse (§ 4)
src/gpu/FilUni2D.cuh    le même en UNE SEULE BOUCLE ( un pas par itération ) et avec des lanes
                        persistantes — mesuré, et perdu (§ 4)
src/gpu/FilShm2D.cuh    le même avec la rotation en MÉMOIRE PARTAGÉE ( layout [case][thread] ) :
                        −26 % d'instructions, −25 % d'occupation, égalité — perdu de peu (§ 4)
src/gpu/FilPh2D.cuh     LES PHASES : un noyau PERSISTANT par SM, trois files par bloc ( attend une
                        boîte / a une feuille / finie ), l'état des cellules en vol en RAM, et la
                        file de coupe GROUPÉE PAR FEUILLE ( tri bitonique par blocs de 64 ) —
                        écrit et mesuré (§ 4) : perdant en `float`, −12 % en `double` (uniforme)
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
src/mains/main_bande.cu      le débit en streaming SoA : ce que coûte une phase si l'état va en RAM
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

**`filrot` — `filbrk` avec le remontage par décalage.** Dans `filbrk` chaque case de sortie va
chercher son sommet par une lecture à indice dynamique (`selR` : sept compares, sept `select`),
trois tableaux, huit cases — la moitié des instructions du noyau. Ici l'entrée `i1` et le compte
`nb_out` viennent du masque (`ffs`, `popc`), et les sommets gardés sont **décalés en bloc** d'un
pas `d` dynamique, en barillet (trois étages : de 1, de 2, de 4 selon les bits de `d`, un
`select` par case et par étage). Si la plage extérieure boucle, les gardés sont contigus et la
sortie est `[ A, B, v_j3 … v_j0 ]` — les points créés à des positions FIXES, le reste décalé de
`j3 − 2` ; sinon `[ v_0 … v_i1−1, A, B, v_j3 … ]`, la tête immobile, la queue décalée de
`nb_out − 2`. Une seule formule pour les deux : `new[o] = o < a ? old[o] : o == a ? A : o == a+1 ?
B : old[o + d]`.

**`filnrm` — la cellule normalisée avant la coupe.** L'idée (H. L.) : mettre les sommets dans un
ordre canonique AVANT de calculer `ta`, `tb`, pour que tout se lise à des positions fixes et que
rien ne soit à recomposer après. La sortie est toujours `[ A, B, v_j3 … v_j0 ]` : les gardés dans
l'ordre cyclique depuis `j3`, A et B en 0 et 1. Les gardés y sont amenés par une **rotation
modulo `nb`** de `j3 − 2`, faite en deux barillets (à gauche pour ce qui vient de `[ j3, nb )`, à
droite pour ce qui vient de `[ 0, i1 )`) et un `select` par case. Les quatre sommets des
intersections : `v_j2, v_j3` sont en 1 et 2 de la moitié gauche ; `v_j0, v_i1` sont adjacents, un
barillet à droite de `R − 1 − i1` les met en `R − 2, R − 1` ; les deux cas de bord (`j3 = 0`,
`i1 = 0`) lisent **le dernier sommet**, tenu dans un registre à part (c'est `v_j0` à chaque coupe).
`s` n'est pas décalé : recalculé pour ces quatre sommets, un `fma` chacun. Plus une lecture
indexée dans la coupe. Et plus de test « c'est mon germe » : son plan a `d = 0` et `off = 0`
exactement, donc `s = 0`, jamais `> 0` — le test coûtait une lecture par germe (7 %).

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

| | n | CPU 8 fils | `fil` | `filreg` | `filregc` | `filmix6` | `filbrk8` | `filrot8` | `filnrm8` | `voies` 8 | `voies16` | `voies32` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2D uniforme | 10⁶ | 145 ns/germe | 68 (×2.1) | 16 (×8.9) | 15 (×9.7) | 13.2 (×11) | 9.9 (×15) | 8.7 (×17) | **7.7 (×19)** | 29 (×5.0) | 34 | 34 |
| 2D lignes / Voronoï | 10⁵ | 141 | 63 (×2.3) | 23 (×6.1) | 21 (×6.7) | 20.0 (×7) | 21.5 (×6.6) | 20 (×7) | **19 (×7.4)** | 36 (×4.0) | 32 | 32 |
| 2D lignes / aires égales | 10⁵ | 696 | 188 (×3.7) | 61 (×11.5) | 51 (×13.6) | **45.5 (×15)** | 51 (×14) | 51 (×14) | **47 (×15)** | 106 (×6.6) | 108 | 128 |
| 3D uniforme | 10⁶ | 1831 | 3855 (×0.5) | — | — | — | — | — | — | **229 (×8.0)** | — | — |
| 3D plans / Voronoï | 10⁵ | 1774 | 3458 (×0.5) | — | — | — | — | — | — | **231 (×7.7)** | — | — |
| 3D plans / volumes égaux | 10⁵ | 3054 | 4651 (×0.7) | — | — | — | — | — | — | **405 (×7.5)** | — | — |

(chiffres 2D repris après l'ajout d'une chauffe de 300 ms avant le premier chrono, § 6 : sans elle
le premier noyau d'un banc tournait à fréquence réduite, 17 au lieu de 13.)

**Le remontage par décalage (`filrot8`).** Le SASS de `filbrk8` : `selR` seul fait **25 % des
instructions**, le remontage entier la moitié — 24 % de `FSEL`, 21 % de compares (des `select`,
pas des branches : `BRA` 5 %, la divergence n'a pas bougé). Avec le décalage en barillet :
2867 → **2167 instructions par cellule (−24 %)**, mais 9.9 → 9.0 ns/germe (−9 %) et −4 % sur les
lignes : le noyau est borné par la latence (9 cycles par instruction émise, 6,5 threads actifs
par warp), une instruction de moins ne rend pas son temps. Ce qui reste de `selR` (14 %) : les
douze lectures des sommets d'intersection (`j0, i1, j2, j3`) et l'aire ; dans le cas qui boucle
elles sont à des positions connues du tableau décalé (`j2 → 1, j3 → 2, j0 → nb_in + 1`), pas
dans l'autre — non fait. `filrot6` 13.8 : `R = 6` reste perdant.

**La cellule normalisée (`filnrm8`).** 2167 → **1816 instructions par cellule** (2867 pour
`filbrk8`, −37 %), `FSEL` de 27 à 17 %, plus une lecture indexée dans la coupe, et le test du
propre germe en moins (7 %) : 8.7 → **7.7 ns/germe**, 20 → 19, 50 → 47. Ce qui reste en tête du
profil : la première passe (`s` et le masque, 11 %), les barillets (17 %), le test d'élagage.

**Une seule boucle, des lanes persistantes (`filuni`) : perdu, et pourquoi.** 6,8 threads actifs
par warp, ce n'est pas les registres (72, 87 % d'occupation) : c'est de la divergence, de deux
sources — les **phases** (un lane dépile un nœud, un autre teste un germe, un troisième coupe : le
warp sérialise les trois chemins) et la **queue** (les 32 cellules d'un warp finissent à des
moments différents, le warp vit jusqu'à la plus lente). Deux remèdes essayés : une seule boucle
où chaque itération fait UN pas (le germe suivant de la feuille ouverte, sinon le nœud suivant de
la pile, sinon la cellule est finie), l'état du parcours vivant dans le lane ; et des lanes
persistantes (un lane qui finit prend une autre cellule par `atomicAdd`, autant de threads que
la carte en loge). Résultat : la boucle unique seule 7.9 → **12.4** ns/germe (+57 %), avec les
lanes persistantes **14.9**, 3,4 threads actifs par warp au lieu de 6,8, 3 800 instructions de
warp par cellule au lieu de 1 800. Ce que ça enseigne : avec deux boucles imbriquées les lanes
d'une même phase **avancent ensemble** — la boucle des germes d'une feuille tourne dix fois en
lock-step pour tous ceux qui y sont — alors qu'une boucle unique remélange les phases à chaque
itération et paie les trois chemins à chaque tour ; et les 32 cellules d'un warp, voisines dans
l'arbre et parties ensemble, sont **corrélées en phase** — les lanes persistantes détruisent cette
corrélation, et la queue qu'elles rattrapent vaut moins qu'elle. La divergence qui reste n'est
pas une affaire de structure de boucle mais de cellules dissemblables dans un warp : c'est le
tri par taille (§ 5).

**La rotation en mémoire partagée (`filshm8`).** La mémoire partagée fait ce qui manque aux
registres : l'indexation dynamique. La rotation de `filnrm` devient huit stores à indices fixes
et six chargements à indices calculés, les quatre sommets des intersections quatre chargements —
plus un barillet, plus un `select`. Layout `[ case ][ thread ]` (une ligne par thread mettrait
les 32 threads d'un warp sur quatre bancs). Mesuré : **1816 → 1343 instructions par cellule
(−26 %)**, 60 registres, 7,5 actifs par warp… et 8.0–8.3 ns/germe contre 7.7. Ce qui perd est
l'**empreinte** : `x`, `y`, `cid` × 8 cases = 96 octets par thread, 12 Ko par bloc, cinq blocs par
SM au lieu de huit, 62 % d'occupation au lieu de 87 — sur un noyau borné par la latence, c'est ce
qui compte. Réutiliser le même tampon pour `x` puis `y` (8 Ko, la dépendance store → load est dans
le thread) remonte à 75 % mais sérialise deux allers-retours : 8.3 ; le carveout à 100 % de
mémoire partagée prend le L1 (la pile, les nœuds) : 8.3 aussi. À garder pour la 3D, où la
rotation d'une cellule à trente sommets a plus à rendre et où le warp entier partage une cellule.

**L'oracle de l'homogénéité (`filnrm8tri`, `filnrm8tril`) — ce qu'une file par phases pourrait
rendre, mesuré sans la construire.** L'idée d'une file (des warps spécialisés par phase — couper
une cellule par les germes d'une boîte, fournir des boîtes aux cellules en attente, mesurer — qui
se passent l'état par la mémoire partagée) vise deux choses : des warps **homogènes** (32 items
de coût semblable, donc max ≈ moyenne) et pas de queue. Le modèle SIMT dit que l'*alignement*
des phases par lui-même ne rend rien (l'ordonnanceur fait déjà partager les instructions d'une
boucle à des lanes à des itérations différentes ; le coût est la somme sur les phases du max sur
les lanes) ; ce qui rend, c'est l'homogénéité. On la mesure sans la file : un premier tour compte
le coût de chaque cellule (plans + boîtes testés), puis le noyau tourne sur les cellules **triées
par coût** — chaque warp reçoit 32 cellules semblables, sans queue :

| | uniforme / lignes V / lignes L | actifs / warp | L1 |
|---|---|---|---|
| `filnrm8`, l'ordre de l'arbre | 7.8 / 19 / 46 | 6.8 | 91 % |
| trié par coût, globalement | **14.8** / 29 / 65 | **9.4** | 49 %, DRAM 40 %, 22 cycles par instruction |
| trié par coût par tranches de 4096 rangs | 7.7 / **25** / 55 | | |

L'homogénéité fait ce qu'on attendait, 6,8 → 9,4 lanes actifs (+38 %), et c'est tout ce qu'elle
peut donner ; et elle coûte 2× parce qu'elle casse ce qui porte le noyau : **les 32 cellules d'un
warp sont voisines dans l'arbre** et lisent les mêmes nœuds et les mêmes feuilles. Le tri local
garde la localité à 4096 près et perd quand même 30 % sur les lignes : la localité qui compte est
à l'échelle du warp. Une file par phases redistribue les items entre lanes et paie ce prix, plus
son état en mémoire partagée (~220 octets par cellule en vol — la pile surtout — soit ~300
cellules par SM, 30 % d'occupation, là où `filshm` a montré que 62 % annulaient déjà −26 %
d'instructions), plus la compaction et les barrières. Verdict : pas en 2D sur cette carte ; le
tri par taille du diagramme précédent (§ 5) tombe avec.

**Les phases avec l'état en RAM du GPU — l'enveloppe.** La mémoire partagée ne peut pas porter
32k cellules en vol ; la RAM, si. Le schéma : des chunks de 32k cellules, une phase « couper par
les germes d'une boîte » qui écrit en RAM les cellules pas finies (`[x, y, cid]` + la boîte
courante), une phase « avancer le parcours » qui donne à chacune sa boîte suivante, une phase
« mesurer » quand le tampon des finies est assez plein. L'intérêt par rapport au tri par coût
(ci-dessus) : en groupant les couples (cellule, feuille) **par feuille**, la phase de coupe aurait
une localité *meilleure* que l'actuelle (32 lanes, une seule feuille lue) et une homogénéité
parfaite. Le prix est le trafic. Les deux nombres qui décident :

* **le nombre de phases** = les feuilles visitées par cellule — mesuré 3.80 en uniforme (médiane
  4, p99 8), 4.14 sur les lignes Voronoï, **32.7** sur les lignes Laguerre ;
* **le coût d'une phase** = lire et réécrire l'état — `xmake run bande 1000000 8` : **0.36 ns par
  cellule et par passe**, 532 Go/s, en SoA parfaitement coalescé ;
* **et ce coût est-il payé ?** `xmake run bande 1000000 8 F` injecte `F` fma par sommet dans la
  passe : de `F = 0` à `F = 128` le temps ne bouge pas (0.356 → 0.373 ms) alors que le calcul
  monte à 5.5 Tfma/s, ~80 % du pic fp32 de la carte. **Le trafic est entièrement masqué par le
  calcul** tant que celui-ci reste sous ~1000 opérations par cellule et par passe — une phase de
  coupe en fait ~600 (dix plans à une soixantaine d'instructions). Le temps est
  `max( calcul, trafic )`, pas leur somme.

Donc l'enveloppe est : ~4 phases de coupe et ~4 de parcours, 2.4 ns/germe de trafic **masqué**,
et le gain vaut ce que vaut l'homogénéité — l'oracle donne +38 % de lanes actifs, la moitié du
temps étant dans la phase de coupe, soit au mieux **7.7 → 5.8 ns/germe (−25 %)**. Ce n'est pas
perdu d'avance ; ce qui décide, ce sont les frais que l'enveloppe ne compte pas : la compaction
des listes, les atomiques, et surtout les **lancements de noyaux** — trois par phase, huit phases,
~5 µs pièce : sur des chunks de 32k cellules c'est 1.9 ms pour 10⁶ cellules (25 % du temps
total, le gain y passe), sur des chunks de 10⁶ (96 Mo d'état, la carte en a 11 Go) c'est 60 µs,
négligeable. **Le schéma se joue donc en chunks aussi gros que la RAM le permet**, et son gain
espéré est de l'ordre de −15 à −25 % pour une complexité élevée.

Deux corollaires du recouvrement : ( 1 ) ne stocker que les `nb` sommets réels plutôt que huit
(par un allocateur atomique), ou laisser les `cid` de côté quand seules les mesures comptent,
ne gagnerait **rien en temps** — le trafic est déjà gratuit — seulement de l'empreinte ( 96 →
~50 octets par cellule ) ; ( 2 ) sur les lignes Laguerre, 32.7 phases à 0.63 ns font 20 ns contre
46 ns de calcul : encore sous le toit, mais la marge est mince.

**Et sur une autre génération ?** Deux choses changent, en sens contraire.

| | fp32 | bande passante | FLOP / octet | mémoire partagée / SM |
|---|---|---|---|---|
| RTX 2080 Ti (Turing, ici) | 13.4 T | 616 Go/s (532 mesuré) | 22 | **64 Ko** |
| A100 (Ampere, HBM) | 19.5 T | 1555 Go/s | 13 | 164 Ko |
| RTX 4090 (Ada) | 82.6 T | 1008 Go/s | 82 | 100 Ko |
| H100 SXM (Hopper, HBM) | 67 T | 3350 Go/s | 20 | 228 Ko |

Le ratio FLOP/octet dit si le trafic reste masqué : sur les cartes HBM (A100, H100) il l'est
autant ou mieux qu'ici ; sur les GeForce récentes (Ada, Blackwell) il est 3 à 4 fois moins
favorable, et le schéma par phases y perdrait. La mémoire partagée par SM dit autre chose, qui
concerne `filshm` (ci-dessus) : **notre 64 Ko est le pire cas de toutes les générations
récentes**. Les 12 Ko par bloc qui plafonnent l'occupation à 62 % ici n'en plafonneraient aucune
sur Ampere ou Hopper — la conclusion « la rotation en mémoire partagée perd » est donc
spécifique à Turing et à revérifier ailleurs. S'y ajoutent des mécanismes qui n'existent pas
ici : `cp.async` (Ampere) recouvre global → partagé sans passer par les registres, et la mémoire
partagée *distribuée* entre les blocs d'un cluster (Hopper) permettrait de garder l'état des
cellules en vol sans jamais descendre en RAM — c'est-à-dire le schéma par phases sans son trafic.

**Les phases, écrites (`filph8`).** Le schéma chiffré ci-dessus, implémenté : un noyau
**persistant, un bloc par SM** — pas de noyau global par phase, les phases sont internes au bloc
et se synchronisent par `__syncthreads()` (~20 cycles) au lieu de lancements (~5 µs) — chaque
bloc prenant des cellules **consécutives** dans l'ordre de l'arbre pour garder la localité. Trois
files doubles par bloc (attend une boîte / a une feuille à couper / finie), `CAP = 512` cellules
en vol par bloc, l'état (sommets, `cid`, pile, quatre entiers ≈ 200 o) en RAM. Les files sont des
**bitmaps parcourus dans l'ordre** et non des listes remplies par `atomicAdd` : les lanes
consécutives prennent alors des slots consécutifs, ce qui rend la coalescence (avec des listes :
8.8 Ko lus par cellule et 365 Go/s de DRAM ; avec les bitmaps, 20.7 au lieu de 28.5 ns/germe).

| ns/germe | uniforme | lignes V | lignes L | | uniforme | lignes V | lignes L |
|---|---|---|---|---|---|---|---|
| | *float* | | | | *double* | | |
| `filnrm8` | **7.7** | **18.6** | **45.8** | | 100.5 | 133.2 | **492** |
| `filph8` (phases) | 20.7 | 28.8 | 110.4 | | 97.7 | 136.2 | 805 |
| `filph8g` (groupé : tri) | 20.7 | 35.1 | 132.2 | | 87.9 | 132.4 | 802 |
| `filph8b` (groupé : binning) | 22.4 | 33.8 | 119.6 | | 87.6 | **124.8** | 802 |
| `filph8a` (groupé : arène à trous) | 26.5 | 35.5 | 134.1 | | 102.7 | 139.7 | 846 |
| `filph8c` (arène **compactée**) | 20.8 | **31.7** | **113.3** | | **86.4** | 129.3 | 803 |

**Le schéma marche, et le profil dit exactement quand.** Les lanes actifs passent de 6,8 à
**11,3–11,8 sur 32 (+70 %)** — mieux que les +38 % que l'oracle du tri laissait espérer, parce
que grouper par phase est plus efficace que trier par coût. Mais le régime change tout :

* en `float`, DRAM à 49–56 % et calcul à 10–15 % : **borné par la mémoire**, le trafic n'est plus
  masqué — le noyau de base est trop rapide pour le payer, et `filph8` perd 2,7× ;
* en `double` (FP64 au 1/32 sur cette carte), DRAM à 21 % et calcul à 83 % : **borné par le
  calcul**, le trafic est masqué, et le gain d'homogénéité sort : 101.3 → **98.5** sur l'uniforme.

D'où vient le trafic ? Pas de l'état des cellules (12 % des requêtes) ni de la pile (13 %), mais
de **l'arbre lui-même** (~50 % : `ar.nodes[]`). Dans `filnrm8` les 32 lanes d'un warp portent des
cellules voisines qui parcourent les mêmes nœuds — L1 à 91 % ; ici la redistribution fait que
chaque lane en est à un endroit différent du parcours — L1 à 40 %. C'est la même perte de
localité que le tri par coût, arrivée par un autre chemin. Le paramètre `CAP` l'arbitre :
`CAP = 128` garde les cellules d'un bloc plus corrélées et gagne en `float` (15.3 au lieu de
20.7) mais perd l'homogénéité et le gain en `double` (109 au lieu de 98.5).

**Le groupement par feuille (`filph8g`), qui suivait de cette analyse.** La file « a une feuille »
est triée par numéro de feuille avant la phase de coupe : une clef `( feuille << 9 ) | slot` et un
**tri bitonique en mémoire partagée**, de sorte que les lanes d'un warp partagent quelques feuilles
au lieu d'en avoir trente-deux. Les cellules d'un bloc étant voisines dans l'arbre, elles visitent
largement les mêmes feuilles : le groupement a de la matière. Mesuré : les lanes actifs montent
encore, **11.0 → 14.9 sur 32** (le tri rend aussi la phase homogène : toutes les cellules d'une
même feuille y testent le même nombre de plans), et le L2 passe de 58 à 67 %. Le tri complet
(45 passes sur 512) coûte cependant plus qu'il ne rend en `float` ; **un tri partiel par blocs de
64** (21 passes) garde le gain sans le prix — la file est déjà presque triée, les slots ordonnés
correspondant à des cellules voisines :

| tri sur des blocs de | float (uniforme / lignes V / L) | double |
|---|---|---|
| 512 (complet) | 23.6 / 39.6 / 158 | 88.5 / 134.5 / 839 |
| 128 | 20.9 / 36.2 / 140 | 88.0 / 129.8 / 827 |
| **64** | **20.7** / 35.9 / 134 | **88.0** / 132.2 / 838 |

**Trois façons de grouper, et laquelle gagne.** Le tri ci-dessus réorganise la file après coup ;
deux autres s'en passent de plus en plus :

* `filph8b`, le **binning** : compter par zone (`feuille % 512`), scanner, placer — trois passes,
  pas une comparaison. À égalité avec le tri (89.4 contre 88.7 en `double`) ;
* `filph8a`, l'**arène** : la **première écriture** atterrit directement à la bonne place — dès que
  la phase de parcours connaît la feuille, le slot est posé dans la zone de cette feuille
  (`feuille % 64`), à une place prise par un compteur atomique, et part dans un pool commun si la
  zone est pleine. **Aucune passe de réorganisation** ; en échange la phase de coupe balaie
  l'arène et saute les trous. Mesuré : 103.7 en `double`, **plus lent que les deux autres**.
  (Le pool doit avoir `CAP` places — au plus `CAP` entrées en tout : un pool plus petit avec un
  `% ARN` écrase des entrées, ce qui donnait 1603 cellules perdues sur 10⁶ et des temps
  flatteurs. Le réglage des zones a été balayé, il ne renverse rien.)

Pourquoi l'arène à trous perd alors qu'elle supprime du travail : dans le régime où le groupement
sert (`double`, borné par le calcul), **le tri ne coûte rien** — il est masqué comme le reste du
trafic — donc il n'y a rien à économiser en l'évitant ; tandis que les trous, eux, coûtent
vraiment : **21 % des entrées partent au pool** (mesuré, mêmes 20–21 % sur les trois cas, avec
64 zones de 8 places), donc l'arène est remplie à ~80 % et la phase de coupe fait un tour de
boucle de plus, chaque lane tombée sur un trou étant une voie perdue dans son warp.

* `filph8c`, l'**arène compactée sans déplacer les données** : la première écriture reste directe,
  mais `NZA + 1` offsets (un préfixe sur les tailles de zone, une soixantaine d'additions faites
  par un seul thread) donnent une numérotation compacte, et la phase de coupe retrouve la zone
  d'un item par une **recherche binaire** en sept étapes. Plus un trou parcouru, et toujours
  aucune donnée déplacée. **C'est la meilleure des quatre** : 86.4 ns/germe en `double` sur
  l'uniforme (contre 87.9 pour le tri et 102.7 pour l'arène à trous), et la meilleure aussi en
  `float` sur les deux nuages de lignes. La recherche binaire coûte moins que les trous qu'elle
  évite, et bien moins que de déplacer les données.

Bilan : en `double`, le schéma complet (phases + arène compactée) fait **86.4 contre 101.4
ns/germe pour `filnrm8`, −15 %**
sur l'uniforme et −2 % sur les lignes Voronoï (DRAM à 12 %, calcul à 83 % : le trafic est
entièrement masqué, comme l'enveloppe le prévoyait). En `float` il reste 2,6× derrière : le noyau
de base y est trop rapide pour payer le trafic, quelle que soit l'homogénéité. Et sur les lignes
Laguerre (33 feuilles par cellule, des cellules très inégales) il perd dans les deux précisions :
trop de phases, et un bloc dont quelques cellules traînent bloque ses slots.

**Où ça bloque — et ce n'est PAS l'occupation.** C'était l'hypothèse qui tenait jusqu'ici (168
registres, trois blocs par SM, 38 % d'occupation, 94 % des cycles sans un warp éligible). Elle est
maintenant **réfutée par l'expérience**. `filph8m` porte la coupe de `filmsk` dans le noyau des
phases (§ plus haut : trois tableaux temporaires au lieu de huit) et tombe à **137 registres en
`double`, 86 en `float`** ; `filph8m4` le force à quatre blocs par SM, ce qui donne **exactement
128 registres, zéro octet de débordement, et 50 % d'occupation mesurée** — le palier que le
paragraphe précédent croyait manquer « de trois registres ».

| `double`, ns/germe | registres | blocs/SM | locale | uniforme | lignes V | lignes L |
|---|---|---|---|---|---|---|
| `filph8c` (coupe `filnrm`) | 168 | 3 | 0 | **84.6** | **131.2** | **788** |
| `filph8m` (coupe `filmsk`) | 137 | 3 | 0 | 94.3 | 132.0 | 903 |
| `filph8m4` (idem, 4 blocs forcés) | **128** | **4** | **0** | 94.0 | 136.0 | 948 |

Franchir le palier **ne change rien** (94.0 contre 94.3), et la coupe la plus petite est la plus
lente : l'occupation n'était pas le frein. Le profil dit où il est vraiment — par instruction
émise, les cycles d'attente se répartissent en **34 à 39 sur le `long scoreboard`** (la mémoire
globale), 8 à 13 sur les **barrières** (`__syncthreads()` entre phases), 1.7 sur les latences
fixes, et **0.1 seulement en « pas sélectionné »**. Ce dernier chiffre est le verdict : il n'y a
quasiment jamais un warp prêt à émettre qui attende son tour, donc **ajouter des warps n'ajoute
rien** — ils attendraient la même mémoire. Ce qui borne le noyau des phases, c'est la **latence
des allers-retours de l'état en RAM**, plus la synchronisation entre phases ; pas les registres.

Ce que ça vaut pour la suite : **le schéma est bon quand le calcul domine le trafic**, ce qui est
exactement le régime de la 3D (800 instructions par coupe contre 60) — et c'est là qu'il faudrait
l'essayer, pas en 2D `float` sur cette carte.

**Ne plus déplacer les sommets : l'ordre dans un registre (`filord8`, idée de H. L.).** `filnrm`
normalise la cellule à chaque coupe — trois barillets sur `x`, `y`, `cid`, huit tableaux
temporaires. Ici les valeurs **ne bougent jamais** : un sommet reste dans son slot tant qu'il vit,
et un registre `O` de 64 bits porte l'ordre cyclique — son octet `i` est le masque **one-hot** du
slot où se trouve le sommet de position `i`. Un masque `vivant` de huit bits dit quels slots sont
occupés, son complément donne les slots libres. Par coupe :

* la première passe calcule `s` pour **les huit slots**, occupés ou non (du gâchis, mais pas une
  branche), et rend un masque `M` **par slot** ;
* le masque **par position** se recompose en sept instructions entières :
  `t = O & ( M × 0x0101010101010101 )` a un octet non nul là où le sommet est dehors, et
  « octet non nul → bit » se fait par trois `or` décalés puis une multiplication magique
  (`× 0x0102040810204080 >> 56`) ;
* les quatre sommets de la frontière se lisent par leur slot (`ffs` de l'octet), et leur `s` se
  **recalcule** — un `fma` coûte moins qu'une lecture indexée ;
* les deux sommets neufs prennent deux slots libres (les sortants viennent de se libérer) : deux
  écritures masquées, rien d'autre ne bouge ;
* **`O` se met à jour presque gratuitement** : comme la sortie est normalisée depuis `j3`, les
  gardés sont contigus dans l'ordre cyclique — une rotation de `j3` octets, une troncature à
  `nb_in` octets, et les deux octets neufs à la suite. Six instructions, là où il y avait trois
  barillets.

**Où va le temps, et pourquoi ce n'est pas le mode masque.** Le profil par ligne de `filord8` :
`octets_non_nuls` — la recomposition du masque par position, le cœur de l'idée — ne fait que
**2,2 %** du noyau. Ce qui coûte est ailleurs : les lectures indexées `selR` (13 %), la première
passe et le test d'élagage qui balaient **les huit slots au lieu de `nb`** (11,4 % + 10,2 % : le
gâchis annoncé), et les écritures masquées des deux sommets neufs (8 %). Borner ces boucles par
le slot vivant le plus haut (`31 − clz( vivant )`, les slots libres étant toujours pris par le
plus petit) a été essayé et **perd** — les branches coûtent plus que les itérations épargnées
(8.3 / 20.7 / 51.0 contre 8.0 / 19.4 / 47.1).

Le fait marquant : `filord8` exécute **1762 instructions par cellule contre 1816** pour
`filnrm8` — moins — et reste 4 % plus lent, parce qu'il lui faut **9.0 cycles par instruction
émise contre 7.1**. La chaîne `M → masque par position → i1, j2 → slot → valeur → s → t → A` est
**séquentielle**, là où les trois barillets de `filnrm` sont soixante-douze `select` indépendants
que l'ordonnanceur entrelace à volonté. Autrement dit : en masques on fait moins de travail, mais
on le fait en file indienne. C'est la limite de l'approche telle qu'elle est écrite, et la piste
pour aller plus loin serait d'y **rendre du parallélisme** plutôt que d'économiser encore des
instructions — par exemple en maintenant, à côté de `O`, les masques `succ` et `pred` (octet `i` =
one-hot du successeur / prédécesseur du slot `i`), qui donneraient les quatre slots de la
frontière directement depuis `M` (`j0 = dedans & pred( M )`, `j3 = dedans & succ( M )`,
`i1 = succ( j0 )`, `j2 = pred( j3 )`) en deux branches **indépendantes**, sans passer par les
positions.

**Cette version existe : `filsuc8`.** Deux registres de 64 bits portent la succession et la
précédence (octet `i` = masque one-hot du successeur / prédécesseur du slot `i`) ; il n'y a plus
de positions du tout. `succ` et `pred` d'un *ensemble* se lisent en `hor_or( SU & spread( X ) )`
— `spread` étalant un masque de huit bits en huit octets par une multiplication et trois `or` —
et d'un *singleton* dont on a l'indice, en un simple décalage. La mise à jour est purement
locale : le cycle devient `j0 → A → B → j3`, soit trois octets à écrire dans `SU` et trois dans
`PR`, sans rotation ni ordre global à maintenir. Machine seule :

| ns/germe | uniforme | lignes V | lignes L | registres (double / float) | instr / cellule | cycles / instr |
|---|---|---|---|---|---|---|
| `filnrm8` | **7.6** | 18.5 | **45.2** | 121–128 / 68 | 1816 | **7.1** |
| `filord8` (ordre en registre) | 8.0 | 19.5 | 48.1 | **96 / 58** | **1762** | 9.0 |
| `filsuc8` (succession) | 8.1 | **18.4** | 47.6 | **96 / 58** | 1782 | 9.2 |

Les trois se tiennent en 5 %, et `filsuc8` égale `filnrm8` sur les lignes / Voronoï — le nuage où
les cellules sont les plus régulières. Mais **la chaîne de dépendances n'a pas raccourci** : 9.2
cycles par instruction émise, contre 9.0 pour `filord` et 7.1 pour `filnrm`. Supprimer
l'aller-retour masque↔index a bien rendu `j0` et `j3` indépendants, mais chaque branche reste
longue (`M → spread → hor_or → ffs → voisin → ffs → selR → s → t → A → écritures`), et le profil
ne bouge pas : `selR` 13 %, la première passe 11 %, l'élagage 10 %, les écritures masquées 8 %,
les six `pose_voisin` 5 %. Ce qui fait gagner `filnrm`, ce sont ses barillets — soixante-douze
`select` sans aucune dépendance entre eux, que l'ordonnanceur émet en continu.

Bilan de la famille « masques » : **même vitesse, 21 % de registres en moins**, et une écriture
nettement plus simple (plus de barillet, plus de tableau temporaire). Un bon point de départ pour
la 3D ou pour une carte où la pression de registres décide — pas un gain en 2D ici.

**Les masques de rôle et le barillet unique (`filmsk8`, idée de H. L.).** `filord` et `filsuc`
payaient le désordre : les sommets restant sur place, il fallait « refaire » `x`, `y` et `c` à
chaque lecture. On revient donc à des registres **triés** — les sommets sont toujours en
`0 .. nb - 1` — et deux choses seulement changent par rapport à `filrot`.

*Les quatre sommets de la frontière sortent de quatre masques de rôle*, fabriqués en quatre
instructions à partir de `m`, `prev` et `next`. Chacun n'a qu'un seul bit, puisque la plage
extérieure est un arc cyclique contigu :

```
r0 = ~m & next   ( dedans, le suivant dehors   -> v_j0 )      r2 = m & ~next  ( -> v_j2 )
r3 = ~m & prev   ( dedans, le précédent dehors -> v_j3 )      r1 = m & ~prev  ( -> v_i1 )
```

Un `__ffs` donne l'indice, et « la plage boucle » se lit `r1 > r2` — les deux masques étant
one-hot, les comparer c'est comparer `i1` et `j2`. `filrot` faisait le même travail en deux `__ffs`
suivis de quatre corrections cycliques (`i1 ? i1 - 1 : nb - 1`…).

*Les lectures indexées sont mutualisées et bornées.* Neuf `selR` — `x` et `y` pour les quatre
sommets, plus le `cid` de `v_j2` — balayaient chacun les `R` cases **sans s'arrêter à `nb`**,
contrairement à la première passe. Ils tiennent maintenant dans **une seule boucle**, avec la même
sortie à `nb` : quatre compares par case, partagés par neuf `select`. L'aire fait de même
(`aire_triee`, le sommet précédent gardé dans un registre au lieu de deux `selR` par sommet), et
`filnrm8` en profite aussi. Mesuré : **neutre** (1685 M d'instructions contre 1667, 1413 M pour
`filnrm8` contre 1414) — la branche qui borne coûte ce que les cases épargnées rapportent, comme
pour le « borner par le slot vivant le plus haut » de `filord`. Le code est juste et lisible, ce
n'est pas une optimisation.

*Le remontage tient en un seul barillet.* La sortie est
`new[ o ] = o < a ? old[ o ] : o == a ? A : o == a + 1 ? B : old[ o + d ]`, et `d` peut valoir −1
(une seule coupe sortante : le cas le plus fréquent). `filrot` payait ce cas par une copie décalée
à droite, soit un `select` de plus par case et par tableau. Ici on décale **a priori d'une case**
— ce qui est *gratuit*, un simple renommage de registres à la compilation — en travaillant sur
`u[ k ] = old[ k - 1 ]` de neuf cases, et le barillet part de `e = d + 1 ≥ 0`.

| ns/germe (`float`) | uniforme | lignes V | lignes L | registres (double / float) | blocs / SM | instr (uniforme) |
|---|---|---|---|---|---|---|
| `filnrm8` | **7.7** | **17.7** | 46.2 | 128 / 74 | 4 / 6 | **1414 M** |
| `filrot8` | 8.8 | 19.9 | 49.4 | 106 / 64 | 5 / 8 | 1808 M |
| `filmsk8` | 8.2 | 19.6 | 46.9 | **96 / 59** | **5 / 8** | 1667 M |

Contre `filrot8`, c'est **−8 % d'instructions et −8 % de temps** sur les trois nuages. Contre
`filnrm8`, c'est +6 % de temps pour **−25 % de registres**, registres triés compris — ce que
`filord` et `filsuc` avaient abandonné pour exactement le même prix. Ce qui part, ce sont les huit
tableaux temporaires de huit cases de `filnrm` (les deux barillets de normalisation, doublés en
`double`) : il n'en reste trois, de neuf.

**La variante où les masques servent DIRECTEMENT à cueillir `x`, `y` et `c`** — un masque plein
par rôle et par case, partagé entre les trois tableaux, `ax0 |= bits( x[ i ] ) & k0` en un seul
`LOP3` — a été écrite, déroulée à la main, et **perd** : +11 % d'instructions (1852 contre
1667 M) pour 8.6 contre 8.1. La raison est que **le partage existait déjà** : `selR( x, j )`,
`selR( y, j )` et `selR( c, j )` émettent *un seul* `ISETP.EQ j, i` par case, réutilisé par trois
`SEL` — ptxas le met en facteur tout seul. Le compte : côté indices, 4 rôles × 8 cases de compare
(32) plus 9 × 8 `SEL` (72) ≈ 104 ; côté masques, 4 × 8 fabrications de masque plein
(`-( ( r >> i ) & 1 )`, deux à trois instructions chacune, ≈ 96) plus 9 × 8 `LOP3` (72) ≈ 168.
L'écart mesuré, ≈ 77 instructions par coupe, tombe pile dessus. **Sur cette machine un prédicat
est déjà un masque partagé, et il se fabrique en une instruction au lieu de trois.** Les masques
rendaient un peu d'ILP (3.19 contre 2.96 instructions émises par cycle — les huit cases sont
indépendantes là où la chaîne compare → `select` ne l'est pas), pas assez pour payer. Le noyau a
été simplifié en conséquence : il ne reste que la version par indices.

**Combien de threads sont vraiment en vol ? (et faut-il raboter les registres ?)** La grille n'est
jamais la limite — un thread par cellule, 7813 blocs pour 68 SM — donc ce qui borne les threads en
vol est l'**occupation**, c'est-à-dire les registres par thread. Le banc l'imprime maintenant pour
chaque variante (registres, blocs par SM, occupation, mémoire locale ; au-delà des 192 octets de
la pile, c'est un débordement de registres). Les variantes `filnrm8c{6,8}` et `filmsk8c{6,8}`
forcent `__launch_bounds__( 128, BSM )`, ce qui fait raboter ptxas : à 128 threads par bloc sur
Turing, 4 blocs ⇔ 128 registres, 5 ⇔ 102, 6 ⇔ 85, 8 ⇔ 64 et l'occupation pleine.

| `float` | registres | blocs/SM | occupation | locale | uniforme | lignes V | lignes L |
|---|---|---|---|---|---|---|---|
| `filnrm8` | 74 | 6 | 75 % | 192 | **7.7** | **17.7** | 46.2 |
| `filnrm8c8` | 64 | 8 | **100 %** | 240 ⚠ | 8.7 | 19.7 | 54.8 |
| `filmsk8` | 59 | 8 | **100 %** | 192 | 8.2 | 19.6 | 46.9 |

| `double` | registres | blocs/SM | occupation | locale | uniforme | lignes V | lignes L |
|---|---|---|---|---|---|---|---|
| `filnrm8` | 128 | 4 | 50 % | 192 | **100.3** | **132.0** | **495** |
| `filnrm8c6` | 80 | 6 | 75 % | 432 ⚠ | 101.6 | 146.1 | 514 |
| `filnrm8c8` | 64 | 8 | 100 % | 528 ⚠ | 163.6 | 205.8 | 590 |
| `filmsk8` | 96 | 5 | 62 % | 192 | 100.5 | 137.5 | 501 |
| `filmsk8c6` | 80 | 6 | 75 % | 256 ⚠ | 100.6 | 134.7 | 503 |
| `filmsk8c8` | 64 | 8 | 100 % | 336 ⚠ | 101.2 | 139.5 | 550 |

Trois choses en sortent.

* **`filmsk8` est déjà à 100 % d'occupation en `float`, sans rien forcer** — et il reste 6 % plus
  lent que `filnrm8` à 75 %. En 2D `float`, l'occupation **n'est pas le levier** : le noyau est
  borné par les instructions, pas par la latence. En `double` non plus (62 % contre 50 % pour le
  même temps) : là c'est le débit FP64 de Turing, 1/32.
* **Arrondir à la puissance de deux en dessous fait toujours déborder, et fait toujours perdre.**
  Ces noyaux tiennent la cellule *dans* les registres (8 `x`, 8 `y`, 8 `cid`, plus les
  temporaires) ; raboter les pousse en mémoire locale, et chaque accès débordé coûte bien plus que
  ce que les warps en plus rapportent. Le pire cas, `filnrm8c8` en `double` : 100 % d'occupation,
  336 octets de débordement, **+63 % de temps**.
* **Le plancher de bruit du banc**, au passage : `filnrm8c6` en `float` produit exactement le même
  code que `filnrm8` (74 registres, 6 blocs), et les deux mesures donnent 7.7 / 18.1 / 43.9 contre
  7.7 / 17.7 / 46.2. Donc ±2 % sur l'uniforme et les lignes Voronoï, **±5 % sur les lignes
  Laguerre** — les écarts inférieurs à ça, sur ce nuage, ne veulent rien dire.

Reste que `filmsk8` est **le plus petit noyau de la famille à registres triés**, et c'est le
candidat à porter dans le noyau des phases — le seul endroit où l'occupation a été mesurée
limitante (94 % des cycles sans un warp éligible, 168 registres en `double`, 130 en `float`, et le
palier des quatre blocs par SM à ≤ 128).

Mesuré aussi : **les registres tombent de 121–128 à 96 en `double`, de 68 à 58 en `float`**
(−21 % et −14 %), et le temps est à 3–5 % près celui de `filnrm8` (8.0 / 19.4 / 47.1 en `float` contre
7.6 / 17.8 / 45.1) : les instructions entières du masque et les écritures masquées rendent ce que
les barillets économisent. C'est donc **neutre en vitesse et gagnant en registres** — à garder en
tête là où la pression compte.

Et justement, elle compte dans le noyau des phases (166 registres, 25 % d'occupation) : porté là
(`filph8o`), il descend à **131 registres en `double`** (112 → 92 en `float`)… sans que
l'occupation bouge, et il perd 11 % (95.4 contre 85.8). La raison est un seuil : à 128 threads
par bloc, trois blocs par SM demandent ≤ 170 registres — `filph8c` y est déjà à 166 — et le palier
suivant est à **≤ 128 registres** pour quatre blocs. Avec 131, on le manque de trois registres, et
forcer le plafond fait déborder. Une idée juste, arrêtée par un seuil matériel.

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

# 3 bis. LE PASSAGE À L'ÉCHELLE ( jusqu'à 3·10⁷, et ce que 10⁹ demanderait )

Question de H. L. : à 10⁹ diracs, est-ce que quelque chose change de nature ? Mesuré sur
l'uniforme, `--reps 1 --reps-gpu 5`, machine seule.

| ns/germe | 10⁵ | 10⁶ | 10⁷ | 3·10⁷ |
|---|---|---|---|---|
| `filnrm8` *float* | 14.7 | **7.7** | 8.0 | 7.9 |
| `filmsk8` *float* | 11.3 | 8.2 | 8.6 | 8.4 |
| `filph8c` *float* | 20.7 | 22.6 | 22.3 | 24.0 |
| `filnrm8` *double* | 107.2 | 100.7 | 107.0 | 107.5 |
| `filmsk8` *double* | 106.6 | 100.8 | 107.9 | 109.3 |
| `filph8c` *double* | **87.0** | **86.2** | **92.1** | 103.9 |
| écart max par cellule, *float* | 1.0e-4 | 3.9e-4 | 1.7e-3 | 2.7e-3 |
| écart max par cellule, *double* | 1.8e-11 | 2.1e-10 | 2.5e-9 | 8.3e-9 |
| construction de l'arbre (CPU, 8 fils) | 19 ms | 295 ms | 5.6 s | **22 s** |

**Le noyau passe à l'échelle.** De 10⁶ à 3·10⁷ (×30), `filnrm8` fait +3 % par germe en `float`,
+7 % en `double`. La raison de fond : dans un diagramme de puissance **le travail par cellule ne
croît pas avec `n`** — une cellule a ~6 voisins quelle que soit la taille du nuage. Seule la
descente de l'arbre grandit, en log `n` (16 niveaux à 10⁶, 22 à 3·10⁷), et l'élagage par le
majorant affine en absorbe l'essentiel. Le 14.7 à 10⁵ n'est pas de la mauvaise échelle mais son
contraire : le noyau ne dure que 1.5 ms, les frais fixes de lancement dominent.

**Le schéma par phases, lui, PERD à l'échelle.** Son avance en `double` s'érode : −19 % à 10⁵,
−14 % à 10⁶ et 10⁷, **−3 % seulement à 3·10⁷**. La fenêtre de cellules en vol est FIXE
(`grid × CAP` = 104 448 slots) ; à mesure que `n` grandit elle couvre une fraction décroissante du
problème, et surtout l'arbre cesse de tenir en cache, si bien que les allers-retours de l'état en
RAM entrent en concurrence avec le trafic de l'arbre au lieu d'être masqués par lui. C'est
cohérent avec le diagnostic du § 4 (le frein est la latence mémoire, pas l'occupation). À 10⁹ il
faudrait donc faire grandir `CAP` avec `n`, ou renoncer.

**Ce qui ne passe pas à l'échelle : la mémoire.** Empreinte GPU calculée sur le code (`c[2]`,
`ids`, `res`, `liste`, plus ~2 nœuds de 48 ou 80 octets par feuille de 10 germes) :

| 2D | octets/germe | 10⁹ | tient sur 11 Gio jusqu'à |
|---|---|---|---|
| `float` Voronoï | 33.6 | **31 Gio** | 352 M |
| `float` Laguerre | 37.6 | 35 Gio | 314 M |
| `double` Voronoï | 48.0 | 45 Gio | 246 M |
| `double` Laguerre | 56.0 | 52 Gio | 211 M |

10⁹ diracs ne tient pas sur une carte : il faut plusieurs GPU, ou un découpage spatial avec halo
— que l'arbre rend naturel, puisque les germes y sont déjà triés par boîte. À noter en revanche
que **l'état en vol du noyau par phases est indépendant de `n`** (22 Mio en `float`, 28 en
`double`) : c'est le seul tampon qui aurait pu exploser, et il n'explose pas.

**Ce qui ne passe pas à l'échelle non plus : la construction de l'arbre.** 22 s à 3·10⁷ sur 8
fils, soit ~15 min extrapolées à 10⁹, contre ~8 s pour la mesure GPU en `float`. À cette échelle
**ce n'est plus le diagramme qu'il faut optimiser, c'est le `build`** — et il est encore sur CPU.

**Et la précision en `float` devient rédhibitoire.** L'écart max relatif par cellule croît comme
√n : les sommets sont à ~6e-8 près en absolu, le côté d'une cellule vaut 1/√n. Extrapolé à 10⁹ :
**~1.5e-2, soit 1.5 % par cellule**. La somme, elle, reste bonne (1e-8), mais ce n'est pas elle
qui pilote un Newton sur les mesures. Donc **à 10⁹ il faut le `double`** — et sur cette carte le
`double` coûte 13× le `float` parce que Turing fait le FP64 à 1/32. Sur une carte à FP64 rapide
(A100, H100 : 1/2) ce facteur tomberait vers 2, et c'est aussi là que le schéma par phases, qui
gagne déjà en `double` ici, aurait le plus à rapporter.


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

* **`filnrm8` est la référence 2D** (7.7 ns/germe en uniforme, ×19 ; 19 et 47 sur les lignes,
  `filmix6` à égalité sur Laguerre). Ce qui reste de divergence (6,8 actifs sur 32) est mesuré
  incompressible à peu de frais : l'oracle du tri par coût (§ 4) dit +38 % de lanes au mieux,
  contre une localité par warp qui vaut 2× ; `filreg` / `filregc` à 8 registres et une excursion sont derrière. La piste
  suivante était celle que la seconde passe a révélée : des warps homogènes en taille de
  cellule — mesurée par l'oracle du § 4 : elle ne vaut pas la localité qu'elle coûte. Ce qui reste : la divergence du
  parcours entre les 32 cellules d'un warp (6 actifs) — un tri des cellules par profondeur de
  parcours ou une pile en mémoire partagée ne changeraient pas le fond ; les 3 000 instructions
  par cellule sont à lire ligne à ligne comme pour le 3D.
* **Les paquets, le test en bloc, les deux fils au push, la boucle unique, les lanes
  persistantes, la rotation en mémoire partagée et le tri par coût sont mesurés et perdent**
  (§ 4) : ne pas y revenir sans une idée neuve.
* **`filmsk8` est le même algorithme que `filnrm8` à 96 / 59 registres au lieu de 128 / 74**
  (−25 %), registres triés compris, pour +6 % de temps (§ 4) : c'est le noyau à porter dans le
  noyau des phases, dont le seul frein restant est l'occupation. Le barillet unique y gagne 8 % à
  lui seul ; la cueillette par masques, elle, perd — un prédicat SASS est déjà un masque partagé
  entre `x`, `y` et `c`, et il coûte une instruction au lieu de trois.
* **L'occupation n'est pas le levier en 2D, et raboter les registres fait perdre** (§ 4) :
  `filmsk8` est déjà à 100 % en `float` sans rien forcer et reste derrière `filnrm8` à 75 % ;
  forcer la puissance de deux en dessous fait déborder à tous les coups (jusqu'à +63 %). Le banc
  imprime désormais registres / blocs par SM / occupation / mémoire locale pour chaque variante,
  et `filnrm8c{6,8}`, `filmsk8c{6,8}` refont le balayage. Plancher de bruit du banc : ±2 %, et
  **±5 % sur les lignes Laguerre**.
* **L'ordre dans un registre (`filord8`) et la succession en masques (`filsuc8`) coûtent 21 % de
  registres en moins à vitesse égale** (§ 4) ; leur limite est la longueur de la chaîne de
  dépendances, pas le nombre d'instructions.
  Dans le noyau des phases il manque **trois registres** (131) pour franchir le palier des quatre
  blocs par SM : sortir les `cid` des registres, ou coder `O` autrement, le ferait basculer —
  c'est le chantier le plus court à essayer.
* **À l'échelle (§ 3 bis), le noyau tient (+3 % par germe de 10⁶ à 3·10⁷) mais trois choses
  cassent** : la mémoire (33.6 o/germe en `float`, donc 31 Gio à 10⁹ — plusieurs GPU ou un
  découpage spatial), la construction de l'arbre sur CPU (22 s à 3·10⁷, ~15 min à 10⁹, contre 8 s
  de mesure GPU : c'est le `build` qu'il faut porter), et la précision en `float` (erreur par
  cellule en √n, ~1.5 % à 10⁹ — le `double` devient obligatoire). Le schéma par phases perd son
  avance à l'échelle (−14 % à 10⁶, −3 % à 3·10⁷) : sa fenêtre en vol est fixe.
* **Les phases (`filph8`, `filph8g`, `filph8b`, `filph8a`, `filph8c`, `filph8m`) sont écrites et
  mesurées** (§ 4), groupement par feuille compris — par tri, par binning, par écriture directe en
  arène et par arène compactée (la meilleure : −15 % en `double`). Ce qui les bride n'est ni le
  trafic ni l'**occupation** : `filph8m4` atteint 128 registres, quatre blocs par SM et 50 %
  d'occupation sans un octet de débordement, et ne gagne rien. C'est la **latence** des
  allers-retours de l'état en RAM (34–39 cycles de `long scoreboard` par instruction émise) plus
  les barrières entre phases (8–13), avec 0.1 seulement de « pas sélectionné » — il n'y a
  quasiment jamais un warp prêt qui attende son tour : 6,8 → 14,9 lanes actifs, −12 % en `double` sur l'uniforme, mais bornées par la DRAM
  en `float`. **À reprendre en 3D**, où une coupe coûte 800 instructions au lieu de 60 : c'est le
  régime où le trafic est masqué et où le schéma gagne.
* **Revérifier sur une autre carte** : Turing a la plus petite mémoire partagée par SM des
  générations récentes (64 Ko contre 164–228), et un ratio FLOP/octet médian. Deux de nos
  conclusions en dépendent (§ 4) : « la rotation en mémoire partagée perd » et « le trafic des
  phases est masqué ».
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
