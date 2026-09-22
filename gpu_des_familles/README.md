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
de `solvers_des_familles` (`src/bench/Args.h`), plus `--variante fil | voies | voies16 | voies32 |
toutes` et `--reps-gpu`. Sur la machine, `--threads 8` est à donner (`hardware_concurrency` rend 1
dans le bac à sable) et **tout chronométrage passe par `job -b`**.

---

# 1. OÙ EST QUOI

```
src/gpu/Arbre.cuh       l'arbre tel que le GPU le lit : noeuds AoS alignés sur 16 octets ( boîte,
                        majorant affine, tranche, fils droit ), germes dans l'ordre de l'arbre,
                        le plan bissecteur, le test d'élagage pour UN sommet, la proximité
src/gpu/Fil2D.cuh       UNE CELLULE PAR THREAD, 2D : `coupe_large` mot pour mot, la pile en local
src/gpu/Voies2D.cuh     LA CELLULE SUR LES VOIES, 2D : voie = sommet, `V` = 8, 16 ou 32 voies par
                        cellule ; le débordement est une excursion sur la voie 0
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

# 2. LES DEUX MAPPAGES

Un GPU exécute 32 voies en lock-step. Une cellule de Voronoï est une suite de coupes dont chacune
dépend de la précédente, sur un parcours d'arbre qui dépend de la cellule. Deux façons de mettre
ça sur des voies :

**`fil` — une cellule par thread.** Chaque voie est un petit cœur scalaire qui fait SA cellule du
début à la fin, comme un fil CPU : les sommets dans des tableaux locaux, la pile du parcours
locale, la coupe scalaire en place. C'est ce que fait le chemin GPU de `sdot` (« la boucle nue en
mémoire »). Les 32 cellules d'un warp sont voisines dans l'arbre, donc leurs parcours se
ressemblent ; rien d'autre ne limite la divergence.

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

| | n | CPU 8 fils | `fil` | `voies` 8 | `voies16` | `voies32` |
|---|---|---|---|---|---|---|
| 2D uniforme | 10⁶ | 142 ns/germe | 68 (×2.1) | **26 (×5.4)** | 33 (×4.3) | 27 (×5.2) |
| 2D lignes / Voronoï | 10⁵ | 138 | 70 (×2.0) | 34 (×4.1) | 31 (×4.5) | **25 (×5.6)** |
| 2D lignes / aires égales | 10⁵ | 694 | 207 (×3.4) | **98 (×7.1)** | 109 (×6.4) | 127 (×5.4) |
| 3D uniforme | 10⁶ | 1831 | 3855 (×0.5) | **229 (×8.0)** | — | — |
| 3D plans / Voronoï | 10⁵ | 1774 | 3458 (×0.5) | **231 (×7.7)** | — | — |
| 3D plans / volumes égaux | 10⁵ | 3054 | 4651 (×0.7) | **405 (×7.5)** | — | — |

(en 3D, `voies` est le warp entier, en deux passes.) Le téléversement d'un arbre à 10⁶ germes
coûte 20 ms en 3D, 140 ms en 2D avec le premier contexte CUDA ; la descente des 10⁶ mesures, 1 à
5 ms.

## `double`

| | n | CPU 8 fils | `fil` | `voies` |
|---|---|---|---|---|
| 2D uniforme | 10⁶ | 142 | 160 (×0.9) | 191 (×0.7) |
| 2D lignes / aires égales | 10⁵ | 680 | 693 (×1.0) | 875 (×0.8) |
| 3D uniforme | 10⁶ | 1975 | 5648 (×0.3) | **1103 (×1.8)** |
| 3D plans / Voronoï | 10⁵ | 1908 | 5435 (×0.4) | 1066 (×1.8) |
| 3D plans / volumes égaux | 10⁵ | 3302 | 8107 (×0.4) | 2926 (×1.1) |

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
émise par warp). La preuve par `voies32` : 26 voies dorment, zéro divergence, et il fait **jeu
égal** avec `voies8` (27 contre 26 ns/germe en uniforme, mieux sur lignes / Voronoï, moins bien
sur Laguerre où les cellules ont plus de sommets et débordent plus). Le noyau 2D n'est donc pas
borné par les ALU mais par la latence de la chaîne (nœud → test → plan → coupe) ; les voies libres
d'un warp sont à employer à **casser cette chaîne** — tester plusieurs boîtes ou plusieurs plans
d'une feuille en même temps — plutôt qu'à porter plus de cellules (§ 5).

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

* **Casser la chaîne en 2D.** Un warp entier par cellule ne coûte rien de plus que quatre
  groupes (§ 4) : les 24 voies libres peuvent tester **quatre boîtes** ou **quatre plans** d'un
  coup — la première passe d'une feuille de dix germes sur ses ~8 sommets tient dans trois
  opérations de warp, et les plans qui ne coupent rien (la majorité : ~40 testés pour ~15
  effectifs) tombent en bloc. Les coupes effectives restent séquentielles, mais elles ne sont plus
  qu'un tiers des plans.
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

`job -b -- ./build/linux/x86_64/release/mesures --threads 8 --reps 3 --reps-gpu 10 --kernel float`
puis la même en `double` — machine seule, rien d'autre ne tourne. Un `PowerDiagram` CPU bâtit
l'arbre ; le témoin mesure (chauffe, minimum de 3) ; `DiagrammeGpu` reçoit l'arbre ; chaque
variante fait un tour de chauffe puis 10 tours chronométrés par événements CUDA autour du (ou des
deux) noyau(x), et le minimum est gardé. Les profils sont pris avec `ncu` (CUDA 13.3) sur la
première passe (`--kernel-name-base demangled --kernel-name "regex:voies<\(bool\)0, \(int\)2, float>"`)
et la répartition par ligne source vient de `--page source --print-source sass,cuda --csv`,
agrégée par un petit script.
