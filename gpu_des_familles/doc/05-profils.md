# CE QUE LE PROFIL DIT (`ncu`)

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

## Le bilan d'occupation, et pourquoi « une cellule sur V voies » perd VRAIMENT

Question de H. L. : pourquoi ce noyau serait-il plus divergent qu'un noyau à une cellule par voie ?
Réponse mesurée : **il ne l'est pas**. Uniforme, 10⁶, `float`, machine seule :

| | ns/germe | registres | blocs/SM | occup. théorique | occup. **atteinte** | instructions | voies actives / 32 | travail réel (inst × voies) |
|---|---|---|---|---|---|---|---|---|
| `filnrm8` (1 cellule / voie) | **7.6** | 74 | 6 | 75 % | 66 % | **1413 M** | 7.20 | **10 174** |
| `filmsk8f` (idem) | 8.2 | **63** | **8** | **100 %** | **88 %** | 1705 M | 6.45 | 10 997 |
| `voies` (V = 8) | 28.7 | 80 | 6 | 75 % | 61 % | 6171 M | **10.34** | 63 808 |
| `voies16` | 33.3 | 95 | 5 | 62 % | 51 % | 6892 M | 16.22 | 111 787 |
| `voies32` | 33.7 | 137 | 3 | 38 % | 30 % | 6455 M | 23.75 | 153 306 |

Trois choses en sortent, et la première corrige ce que disait ce README.

* **La divergence n'est pas le problème — c'est l'inverse.** `voies` à huit voies a **10.34 voies
  actives par warp contre 7.20** pour `filnrm8` : il est *moins* divergent, de 44 %. L'explication
  « divergence » était fausse.
* **Ce qui le tue, c'est le travail RÉDUNDANT** : 6171 M d'instructions contre 1413, soit ×4.4 — et
  ×6.3 en travail de voie réel (63 808 contre 10 174 par cellule). La raison est structurelle :
  dans l'architecture « la cellule dirige le fournisseur », **la moitié du travail est un parcours
  d'arbre SCALAIRE** (dépiler, tester la boîte, calculer le plan), et ce parcours est **répliqué
  sur les V voies du groupe**. Le compteur de voies actives compte cette redondance comme du
  travail actif — d'où le piège : **« voies actives » n'est pas « voies utiles »**.
* **La prémisse « moins de registres, donc l'attente serait amortie » ne tient pas non plus** :
  `voies` prend **plus** de registres, pas moins (80 contre 63), justement parce que chaque voie
  porte l'état scalaire répliqué (`nb`, `haut`, le nœud, `p0`, `w0`) *en plus* de son sommet — et
  il déborde en mémoire locale (1024 octets, les tableaux d'excursion au-delà de huit sommets).
  Pendant ce temps `filmsk8f` est déjà à **88 % d'occupation atteinte** : en 2D il ne reste presque
  rien à gagner de ce côté.

**Ce que ça dit de l'idée « 32 voies pour les intersections 2×2 ».** Calculer, pour chaque plan
proposé, ses intersections avec tous les plans déjà posés puis tester chaque candidat contre les
autres, c'est joli et sans branche — mais ça tombe dans le même piège : si le warp entier travaille
sur une cellule, le parcours scalaire (la moitié du travail) est répliqué **32 fois**, et le calcul
lui-même passe de O(k) à O(k²) par coupe. Le modèle de coût mesuré ci-dessus dit que ça perd, et de
beaucoup. Le seul régime où ça pourrait basculer est celui où la coupe écrase le parcours — la 3D,
où une coupe coûte 800 instructions contre 60 en 2D, et où `voies3` (la cellule sur le warp) gagne
déjà ×8 sur le CPU.

**La conclusion générale**, qui vaut pour toutes les tentatives de ce genre : tant que le parcours
scalaire pèse la moitié du travail, **répartir UNE cellule sur plusieurs voies réplique cette
moitié**. Ce n'est pas la divergence qu'il faut attaquer, c'est la part scalaire — soit en la
rendant vectorielle (tester plusieurs boîtes à la fois : essayé, perdu, ci-dessus), soit en la sortant du
noyau de coupe (les phases : écrites, ci-dessus).

---

[← sommaire](../README.md)
