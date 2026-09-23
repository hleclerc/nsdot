# LES CHIFFRES

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

(chiffres 2D repris après l'ajout d'une chauffe de 300 ms avant le premier chrono, [§ méthode](07-methode.md) : sans elle
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
tri par taille ([§ ce qui reste](06-ce-qui-reste.md)).

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
tri par taille du diagramme précédent ([§ ce qui reste](06-ce-qui-reste.md)) tombe avec.

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
phases (ci-dessus : trois tableaux temporaires au lieu de huit) et tombe à **137 registres en
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
flottant, [§ profils](05-profils.md)) là où il coûte 4 % au CPU. Sur cette carte le GPU est une histoire de `float` ; et
`float` est ce que `sdot` fait en production — le plancher du résidu de Newton est alors celui de
la géométrie (`solvers_des_familles`, [§ profils](05-profils.md)).

**Exactitude.** En `double` l'écart GPU / CPU par cellule est 2e-10 en 2D et 7e-14 en 3D (l'ordre
des opérations), la somme est 1 à 1e-9. En `float` l'écart est le bruit du flottant lui-même —
une cellule de côté 1e-3 avec des sommets à 6e-8 près a son aire à 4e-4 près, et les deux côtés
ne contractent pas les `fma` pareil — la somme est tenue à 1e-6. Aucune cellule ne déborde après
la seconde passe.

---

---

[← sommaire](../README.md)
