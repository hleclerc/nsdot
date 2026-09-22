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
de `solvers_des_familles` (`src/bench/Args.h`), plus `--variante fil | filreg | filregc | filmix{4,6,8,12,16} | filbrk{6,8,10,12,16} | filbrk8nu | filrot{6,8} | filnrm8 | filuni8 | filuni8np | filshm8 | filnrm8tri | filnrm8tril | filph8 | filph8g | voies | voies16 |
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
| `filph8g` (+ groupé par feuille) | 20.2 | 36.2 | 126.2 | | **88.7** | **130.5** | 837 |

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

Bilan : en `double`, le schéma complet fait **88.7 contre 100.5 ns/germe pour `filnrm8`, −12 %**
sur l'uniforme et −2 % sur les lignes Voronoï (DRAM à 12 %, calcul à 83 % : le trafic est
entièrement masqué, comme l'enveloppe le prévoyait). En `float` il reste 2,6× derrière : le noyau
de base y est trop rapide pour payer le trafic, quelle que soit l'homogénéité. Et sur les lignes
Laguerre (33 feuilles par cellule, des cellules très inégales) il perd dans les deux précisions :
trop de phases, et un bloc dont quelques cellules traînent bloque ses slots.

Ce que ça vaut pour la suite : **le schéma est bon quand le calcul domine le trafic**, ce qui est
exactement le régime de la 3D (800 instructions par coupe contre 60) — et c'est là qu'il faudrait
l'essayer, pas en 2D `float` sur cette carte.

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
* **Les phases (`filph8`, `filph8g`) sont écrites et mesurées** (§ 4), groupement par feuille
  compris : 6,8 → 14,9 lanes actifs, −12 % en `double` sur l'uniforme, mais bornées par la DRAM
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
