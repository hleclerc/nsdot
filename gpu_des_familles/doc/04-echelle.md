# LE PASSAGE À L'ÉCHELLE ( jusqu'à 3·10⁷, et ce que 10⁹ demanderait )

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

L'oracle de coût (`filnrm8tri` avec `MESURES_DEBUG=1`) dit exactement ce qui grandit et ce qui ne
grandit pas :

| par cellule | 10⁵ | 10⁶ | 10⁷ |
|---|---|---|---|
| feuilles visitées (moyenne / médiane / p99 / max) | 4.13 / 4 / 8 / 13 | 3.80 / 4 / 8 / 13 | 3.43 / 3 / 7 / 14 |
| coût total (plans + boîtes testés) | 65.3 | 73.4 | **81.4** |

Les **feuilles visitées ne croissent pas** — elles décroissent même un peu : c'est le voisinage de
la cellule, et il ne dépend pas de `n`. Ce qui croît, c'est la descente : +8 unités de coût par
décade, soit **+12 % par décade**, pour ~3.3 niveaux d'arbre de plus. Extrapolé à 10⁹ : ~97 contre
81 à 10⁷, **+19 %** — et comme ces unités-là sont des tests de boîte (trois `fma` et un compare,
bien élagués) et non des coupes, le temps mesuré bouge encore moins. C'est la traduction chiffrée
du « travail par cellule constant ».

**Le schéma par phases, lui, PERD à l'échelle.** Son avance en `double` s'érode : −19 % à 10⁵,
−14 % à 10⁶ et 10⁷, **−3 % seulement à 3·10⁷**. La fenêtre de cellules en vol est FIXE
(`grid × CAP` = 104 448 slots) ; à mesure que `n` grandit elle couvre une fraction décroissante du
problème, et surtout l'arbre cesse de tenir en cache, si bien que les allers-retours de l'état en
RAM entrent en concurrence avec le trafic de l'arbre au lieu d'être masqués par lui. C'est
cohérent avec le diagnostic du [§ profils](05-profils.md) (le frein est la latence mémoire, pas l'occupation). À 10⁹ il
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


## Quelle carte pour 10⁹ ? ( A100 / H100 contre RTX PRO 6000 )

Le noyau se coupe en deux parts que la mesure sépare : sur le 2080 Ti, `filnrm8` fait 7.70 ns/germe
en `float` et 100.51 en `double`, et comme le FP64 y tourne à 1/32, on résout
`I + F = 7.70`, `I + 32 F = 100.51` → **`I` = 4.71 ns de travail entier / contrôle** (masques,
`select`, barillet, pile) et **`F` = 2.99 ns de travail flottant**. La première part suit le débit
d'émission (≈ SM × horloge, le noyau étant borné par les instructions — [§ profils](05-profils.md)), la seconde suit le
débit FP64. D'où la projection, *extrapolée et non mesurée* :

| | émission | FP64 | `float` | `double` | 10⁹ en `double` | mémoire |
|---|---|---|---|---|---|---|
| RTX 2080 Ti (ici) | 1.00× | 1.0× | 7.70 | 100.5 | 100 s | 11 Go — ne tient pas |
| A100 80 Go SXM | 1.45× | 23× | 5.3 | 7.4 | **7.4 s** | 80 Go |
| H100 SXM5 | 2.21× | 81× | 3.5 | **3.3** | **3.3 s** | 80 Go |
| RTX PRO 6000 Blackwell | **4.65×** | 4.6× | **1.65** | 21.7 | 21.6 s | **96 Go** |

*(specs de mémoire, à revérifier avant tout achat : A100 108 SM à 1.41 GHz / FP64 9.7 T ; H100 SXM
132 SM à 1.755 GHz / FP64 34 T ; RTX PRO 6000 Blackwell 188 SM à ~2.6 GHz, FP32 ~125 T mais **FP64
à 1/64**, soit ~1.95 T.)*

**Le verdict bascule entièrement sur `float` contre `double`.** En `float` la RTX PRO 6000 est
devant tout le monde d'un facteur 2 sur la H100 — c'est une carte à débit d'instructions, et notre
noyau n'est que ça. En `double` elle est **6× derrière la H100 et 3× derrière une A100**, parce que
son FP64 est bridé à 1/64. Et le ci-dessus dit que le `double` est obligatoire à 10⁹ (erreur en
√n). Donc, en l'état du code : **H100 ou A100**.

Deux choses ne se voient pas dans le tableau.

* **Personne n'est borné par la bande passante**, donc la projection tient. Mesuré (`filnrm8`,
  n = 10⁷) : **235 octets de DRAM par cellule en `float`**, 179 en `double`, soit **7.7 %** de la
  bande passante du 2080 Ti en `float` et 0.26 % en `double`. Reporté sur les autres cartes : 3.4 %
  (A100), 3.1 % (H100), 12.3 % (RTX PRO 6000). Large marge partout.
* **Le L2 de la RTX PRO 6000 (~100 Mo, contre 5.5 ici) tiendrait l'arbre entier** jusqu'à ~10⁷
  germes (92 Mio en `float`) — l'arbre fait 50 % des requêtes mémoire ([§ profils](05-profils.md)), et c'est précisément
  ce qui fait perdre son avance au schéma par phases à grande échelle (ci-dessus). Un avantage réel
  qu'aucune ligne de TFLOPS ne montre. À 10⁹ l'arbre fait 9.2 Gio et plus rien ne tient, sur
  aucune carte.

**Ce qui changerait le verdict** (piste écrite et mesurée : voir juste après). Une partie de
l'erreur du `float` vient de ce que `bissect2` travaille en coordonnées **absolues** —
`off = ½ ( dx·( xj + x0 ) + dy·( yj + y0 ) )`, où `dx` est petit (~1/√n) mais `xj + x0` est
d'ordre 1, si bien que `off` perd `log₂ √n` bits par rapport à la taille de la cellule. Dans un
repère **centré sur le germe** (`x0 = y0 = 0`), la même expression devient `off = ½ ( dx² + dy² )`
et tout est à l'échelle de la cellule : la précision relative redeviendrait indépendante de `n`.
**Écrit et mesuré : ça ne suffit pas** — la perte est en amont, dans le stockage des positions en
`float`. Le verdict reste donc **H100 / A100**.

## Le repère centré sur le germe : écrit, mesuré, et il ne sert PAS à ça

D'abord une correction de méthode, qui change les chiffres de précision de tout ce qui précède.
Le banc comparait le GPU au **témoin CPU dans le MÊME flottant** — deux calculs également faux et
*corrélés* : on mesurait leur écart, pas leur justesse. `--temoin-double` rejoue le nuage avec le
moteur `double` et compare à lui. En `float` :

| écart max par cellule | 10⁵ | 10⁶ | 10⁷ | lignes V (10⁵) | lignes L (10⁵) |
|---|---|---|---|---|---|
| contre le témoin `float` (ce qu'on lisait) | 1.0e-4 | 3.9e-4 | 1.7e-3 | 9.7e-5 | 1.1e-3 |
| **contre le témoin `double` (la vérité)** | **5.7e-4** | **4.8e-3** | **3.0e-2** | 2.4e-3 | **2.7e+00** |

Le `float` est **déjà à 3 % d'erreur par cellule à 10⁷**, pas 0.17 %, et sur les lignes à aires
égales (Laguerre à poids forts) il est **complètement faux dès 10⁵**. (L'écart *max* est une
statistique d'extrême : une part de sa croissance vient de ce qu'on tire plus de cellules. La
conclusion ne change pas.)

Ensuite le repère centré lui-même (`filmsk8g`) : `bissect2c` donne `off = ½ ( dx² + dy² )`,
`bilan_sommet_c` décale la boîte du nœud, la coupe et l'aire ne changent pas d'une ligne (le lacet
est invariant par translation). Et **l'écart max ne bouge pas** : 5.6e-4 / 4.9e-3 / 3.0e-2 contre
5.7e-4 / 4.8e-3 / 3.0e-2.

Sauf que **l'écart max est le mauvais chiffre** — c'est une statistique d'extrême, fixée par une
poignée de cellules dégénérées (une arête presque tangente au plan : `s0 ≈ s1`, et `ta = s0 /
( s0 - s1 )` perd tout), et qui grandit toute seule quand on tire plus de cellules. Le banc
imprime maintenant les **quantiles** de l'écart relatif, et ils disent autre chose (uniforme, 10⁶) :

| | moyenne | médiane | p99 | p99.99 | max |
|---|---|---|---|---|---|
| `filnrm8` (absolu) | 4.3e-5 | 3.2e-5 | 1.8e-4 | 7.6e-4 | 4.8e-3 |
| `filmsk8g` (centré) | 2.1e-5 | **1.4e-5** | 1.1e-4 | 7.7e-4 | 4.9e-3 |

**Le repère centré divise l'erreur de la cellule ordinaire par 2.3.** Il ne touche pas à la queue,
parce que là c'est le conditionnement de l'intersection qui parle, pas la formulation. La part
qu'il ne peut pas atteindre, elle, est **perdue avant le noyau** : `ar.c[ d ][ q ]` est stocké en
`float`, donc `xj` et `x0` portent déjà ~3e-8 d'erreur absolue et `dx = xj - x0` avec eux. D'où la
suite.

Ce que la manche rapporte quand même : **−4 % en `double`**, `off = ½ ( dx² + dy² )` étant plus
court que `½ ( dx ( xj + x0 ) + dy ( yj + y0 ) )`. Confirmé sur deux tailles, au-dessus du
plancher de bruit :

| `double`, ns/germe | 10⁶ | 10⁷ |
|---|---|---|
| `filnrm8` | 100.4 | 107.9 |
| `filmsk8` | 101.4 | 108.7 |
| `filmsk8g` (repère centré) | **96.7** | **104.7** |

**`filmsk8g` est donc le meilleur noyau 2D en `double`** — la précision qui compte à 10⁹ — et il y
est avec 96 registres contre 128.

## La virgule fixe 32 bits (`filmsk8f`)

Le `float` range [0,1] avec un pas de ~6e-8 près de 1 ; en **virgule fixe sur 31 bits** le pas est
**uniforme et vaut 2⁻³⁰ = 9.3e-10**, soit ~64× mieux **pour les mêmes quatre octets**. Et la
différence de deux positions devient une **soustraction entière exacte**, convertie ensuite en
flottant avec toute sa précision relative. [0,1] devient [0, 2³⁰], si bien que le sommet du carré
unité tombe pile et que la cellule de départ est exacte. `bissect2f` fait
`dx = TK( u[ q ] - u0 ) × 2⁻³⁰`, puis `off = ½ ( dx² + dy² )` comme le repère centré.

Écart relatif contre le témoin `double`, en `float` :

| uniforme, 10⁶ | moyenne | médiane | p99 | p99.99 | max |
|---|---|---|---|---|---|
| `filnrm8` | 4.3e-5 | 3.2e-5 | 1.8e-4 | 7.6e-4 | 4.8e-3 |
| `filmsk8g` (centré) | 2.1e-5 | 1.4e-5 | 1.1e-4 | 7.7e-4 | 4.9e-3 |
| `filmsk8f` (centré + fixe) | **1.3e-5** | **6.8e-6** | **8.8e-5** | **2.4e-4** | **1.4e-3** |

| uniforme, 10⁷ | moyenne | médiane | p99 | p99.99 | max |
|---|---|---|---|---|---|
| `filnrm8` | 1.4e-4 | 1.0e-4 | 5.7e-4 | 2.3e-3 | 3.0e-2 |
| `filmsk8g` | 6.5e-5 | 4.3e-5 | 3.5e-4 | 2.3e-3 | 3.0e-2 |
| `filmsk8f` | **4.0e-5** | **2.0e-5** | **2.8e-4** | **7.9e-4** | **9.9e-3** |

| lignes / Voronoï, 10⁵ | médiane | p99 | max | | lignes / aires égales, 10⁵ | médiane | p99 | max |
|---|---|---|---|---|---|---|---|---|
| `filnrm8` | 3.4e-6 | 3.0e-5 | 2.4e-3 | | `filnrm8` | 2.0e-3 | 2.2e-1 | 2.7 |
| `filmsk8f` | **8.2e-7** | **1.4e-5** | **2.1e-4** | | `filmsk8f` | **5.4e-4** | 1.2e-1 | 2.3 |

**Les deux changements ensemble divisent l'erreur de la cellule ordinaire par 4 à 5, et l'écart max
par 3 à 11**, pour **+1 % de temps** (8.2 contre 8.1 ns/germe à 10⁶) et **pas un octet de plus**
(le tableau `float` des positions n'est plus nécessaire, `p0` se reconstruit depuis la virgule
fixe).

Et surtout, **la médiane retrouve la loi en √n propre** : 6.8e-6 à 10⁶, 2.0e-5 à 10⁷, soit ×2.9
par décade contre ×3.16 attendu. **Extrapolée à 10⁹ : 2e-4 de médiane** — 0.02 % sur une cellule
ordinaire, ce qui est utilisable. La queue, elle, garde son exposant : p99.99 ~8e-3 et max ~0.5 à
10⁹, donc une cellule sur 10⁴ à 1 % près et quelques-unes fausses.

**La virgule fixe 64 bits (`filmsk8h`) ne rapporte RIEN de plus.** Essayée : échelle 2⁵², pas de
2.2e-16 (soit l'arrondi exact du `double` d'entrée — on ne peut pas faire mieux), `dx` par
soustraction 64 bits exacte. Résultat, à trois chiffres près **identique** au 32 bits :

| uniforme, 10⁶ | moyenne | médiane | p99 | max | ns/germe |
|---|---|---|---|---|---|
| `filmsk8f` (fixe 32) | 1.3e-5 | 6.8e-6 | 8.8e-5 | 1.4e-3 | **8.1** |
| `filmsk8h` (fixe 64) | 1.3e-5 | 6.9e-6 | 8.9e-5 | 1.4e-3 | 8.3 |

À 10⁷ : médiane 2.0e-5 des deux côtés, max 9.9e-3 des deux côtés. Sur les lignes / Voronoï :
8.2e-7 contre 8.4e-7. Pour +2 % de temps et **huit octets par coordonnée au lieu de quatre**.

Le 32 bits **sature déjà le bénéfice** : à 2⁻³⁰ la quantification des positions (9.3e-10, soit
2.9e-6 rapporté au côté d'une cellule à 10⁷) est passée **sous le plancher de la géométrie en
`float`**, qui vaut 2.0e-5. Descendre l'entrée à 2.2e-16 ne sert donc à rien — c'est l'arithmétique
d'après qui parle.

**D'où vient ce plancher ?** La cellule démarre comme le **carré unité**, dont les coins sont à
distance ~1 du germe, alors que la cellule finale fait ~1/√n. Un sommet final est calculé par
interpolation (`x0v + ( x1 - x0v ) ta`) entre des points qui étaient d'ordre 1 : l'erreur absolue
reste ~6e-8 pendant que le résultat descend à 1/√n, d'où une erreur **relative en √n** — c'est
exactement la loi observée pour le plancher lui-même (6.8e-6 à 10⁶, 2.0e-5 à 10⁷). Deux façons de
l'attaquer, non essayées : **partir d'une boîte serrée autour du germe** (celle de la feuille de
l'arbre, quitte à la vérifier) au lieu du carré unité ; ou **calculer chaque sommet depuis les deux
plans qui le définissent** (un système 2×2 dont tous les coefficients sont à l'échelle de la
cellule) au lieu de l'interpoler.

Deux réserves qui comptaient **et que la suite lève** :

* **La virgule fixe 31 bits est réservée au `float`.** En `double` elle *détruit* la précision
  (1.2e-4 au lieu de 2.1e-10 : 31 bits contre 53) — mesuré, `filmsk8f` en `double` est faux. Cela
  reste vrai : la virgule fixe n'est allumée qu'en `float`.
* **Le Laguerre à poids forts reste hors-jeu en `float`** : médiane 5.4e-4 dès 10⁵ et p99 à 12 %,
  même avec la virgule fixe. Or c'est exactement le cas du transport optimal. **Ce point-là n'est
  plus vrai** — voir la section suivante : la médiane y tombe à 7.7e-7.

## LES TROIS RÉPARATIONS DU `float`, portées du banc CPU

`solvers_des_familles` a passé une session entière à démonter ce qui casse en `fp32` (son
README § 19). Le modèle qui en sort est en une ligne :

> **erreur relative ~ eps · |w| · n^(2/D) + eps · n^(1/D)**

et le remède est toujours le même : **porter la différence, pas la valeur**. Partout où une
quantité *grande dans l'absolu* sert à fabriquer une quantité *à l'échelle `h`*, on perd
`log( grand / petit )` chiffres, et il suffit de ne jamais former la grande. Les trois termes se
lisent directement dans ce qui précède : le repère du germe tue `eps n^(1/D)` sur les positions,
la virgule fixe tue ce qu'il en reste au stockage, et le terme en `|w|` n'avait jamais été
attaqué ici.

Ce qui est arrivé sur la carte, dans l'ordre :

**1. La différence des poids, en `double`.** Le plan bissecteur porte `( w0 - wj ) / 2`. Arrondir
CHAQUE poids avant la soustraction coûte `eps |w|`, ce qui déplace le plan de `eps |w| / ( 2 |d| )`
le long de sa normale, soit `eps |w| / h²` rapporté à la taille d'une cellule. Or les poids ne
sont grands que *dans l'absolu* : pour deux germes qui partagent vraiment une facette,
`| w0 - wj | <~ h²`, du même ordre que le terme géométrique. Un tableau `double` de plus
(`Arbre::w64`), la différence calculée en `double`, un seul arrondi à la fin.

**2. Le repère du germe pour la SECONDE PASSE aussi.** `filmsk` travaillait dans le repère du
germe, `filmix` — le noyau qui finit les cellules trop grosses pour les registres, soit 13 % en
uniforme — était resté en repère absolu. Une cellule sur huit gardait donc l'erreur qu'on venait
d'enlever aux autres, et comme le maximum suit la pire cellule, ce sont elles qui le fixaient.

**3. Les sommets résolus depuis leurs plans.** C'est la seconde des deux idées laissées ouvertes
ci-dessus, et elle bat la première (la boîte serrée) : même sur CPU, où les deux ont été écrites,
la boîte est devenue inutile. Un sommet d'un convexe est l'intersection de `D` plans et ne dépend
PAS de l'histoire des coupes ; le noyau, lui, l'interpole, donc il porte l'erreur de la cellule
telle qu'elle était **quand le sommet est né**. On le résout donc à la fin, en `double`, depuis
les deux coupes qui le portent — elles sont déjà là, l'arête `i` va du sommet `i` au sommet
`i + 1`. Ce qu'il fallait pour cela : que `Plan2::id` porte le **rang** du germe dans l'arbre et
non son identifiant d'appelant, sans quoi le plan ne se relit pas. Au passage cela supprime une
lecture dispersée par germe testé dans la boucle la plus chaude, et ne la laisse que sur les six
arêtes du polygone fini.

Sur une carte, ce qui compte autant que le chiffre : **ça ne branche pas**. Une élimination de
Gauss par sommet, la même suite d'instructions pour tous les threads du warp, pas de reprise, pas
de file de rattrapage. C'est ce qui disqualifiait la boîte de départ, dont le rattrapage
divergeait — une boîte mal dimensionnée fait recommencer la cellule, donc son warp entier.

### Ce que ça donne, contre le témoin `double` (`--temoin-double`)

**Uniforme, `n = 10⁶`, Voronoï**, écart relatif par cellule :

| | moyenne | médiane | p99 | p99.99 | max | ns/germe |
|---|---|---|---|---|---|---|
| `filnrm8` (repère absolu) | 4.3e-05 | 3.2e-05 | 1.8e-04 | 7.6e-04 | 4.8e-03 | 7.5 |
| `filmsk8f` (germe + fixe 32) | 1.0e-05 | 5.6e-06 | 5.7e-05 | 1.2e-04 | 1.9e-04 | 9.0 |
| `filmsk8f` + sommets résolus | 3.0e-07 | 2.1e-07 | 1.6e-06 | 1.5e-05 | 1.2e-04 | 9.4 |
| `filmsk8h` (germe + fixe 64) | 1.0e-05 | 5.7e-06 | 5.8e-05 | 1.2e-04 | 2.2e-04 | 9.5 |
| **`filmsk8h` + sommets résolus** | **1.4e-08** | **1.1e-08** | **4.9e-08** | **8.4e-08** | **1.4e-07** | **9.9** |

**1.1e-08 de médiane, c'est un dixième de l'epsilon du `float`.** Le prix est **+5 %** sur
`filmsk8f` et **+32 %** sur `filnrm8`.

> J'avais écrit ici « la cellule ordinaire est exacte à l'arrondi près, et il n'y a plus rien à
> gagner sans changer de flottant ». **C'est faux d'un facteur 220 000**, et la section suivante
> le corrige : 1.4e-07 ≈ 2.4 · 2⁻²⁴ n'est pas un reste de géométrie, c'est l'epsilon du `float`
> tout nu — donc un arrondi de STOCKAGE, celui qui reconvertit en `float` le sommet qu'on venait
> de résoudre en `double`.

**LA VIRGULE FIXE 64 BITS CHANGE DE STATUT.** La section précédente concluait qu'elle « ne
rapporte RIEN de plus », et c'était juste : à 32 bits la quantification (9.3e-10) était déjà
passée **sous le plancher de la géométrie en `float`**. Le raffinement supprime ce plancher — et
la quantification devient alors le terme dominant. Facteur **19 sur la médiane, 850 sur le max**.
Les deux réparations ne valent QUE prises ensemble ; c'est pour cela que le chemin de Newton
(`chaine`) est passé à `FIX = 64`, pour +2 % et quatre octets de plus par coordonnée.

**Lignes / Voronoï, `n = 10⁵`** (nuage groupé, sans poids) :

| | médiane | p99 | max |
|---|---|---|---|
| `filnrm8` | 3.4e-06 | 3.0e-05 | 2.4e-03 |
| `filmsk8f` | 7.0e-07 | 8.2e-06 | 4.2e-05 |
| `filmsk8f` + résolus | 2.4e-08 | 3.5e-07 | 1.0e-05 |
| **`filmsk8h` + résolus** | **1.2e-09** | **2.9e-08** | **4.2e-07** |

**Lignes / aires égales, `n = 10⁵`** — le Laguerre à poids résolus, `|w|max = 0.13`, celui qui
était déclaré hors-jeu :

| | médiane | p99 | max |
|---|---|---|---|
| `filnrm8` | 2.0e-03 | 2.2e-01 | 2.7e+00 |
| `filmsk8f` | 4.6e-05 | 3.2e-03 | 1.0e+00 |
| `filmsk8f` + résolus | 4.5e-05 | 3.2e-03 | 1.0e+00 |
| `filmsk8h` + résolus, poids arrondis avant la différence | 4.6e-04 | 1.2e-01 | 2.3e+00 |
| **`filmsk8h` + résolus, différence des poids en `double`** | **7.7e-07** | **1.2e-05** | **7.1e-05** |

Trois choses à lire dans ce tableau. **(a)** Le raffinement ne rapporte RIEN tant que la virgule
fixe est à 32 bits : sur un nuage groupé, l'espacement local est bien plus fin que le pas de
9.3e-10, donc le PLAN lui-même est faux, et résoudre un sommet depuis un plan faux ne gagne rien.
**(b)** La différence des poids en `double` vaut à elle seule **×600 sur la médiane et ×32000 sur
le max** — exactement le terme `eps |w| n^(2/D)` du modèle, et il domine tout le reste sur ce
nuage. **(c)** Les trois réparations ensemble font **×2600 sur la médiane et ×38000 sur le max**
contre le noyau de référence.

Sur l'uniforme à poids `h²` (`--weights 1`, `n = 10⁶`), le terme des poids est en revanche presque
invisible — médiane 8.2e-09 contre 4.7e-09, max 1.0e-06 contre 2.0e-07 — et c'est ce que le
modèle prédit : `|w| ~ h²` y rend `eps |w| / h²` égal à `eps`.

## LA QUATRIÈME RÉPARATION : ne pas jeter le sommet qu'on vient de résoudre

Les trois réparations ci-dessus laissaient **1.4e-07 de max et 1.1e-08 de médiane** sur
l'uniforme, et j'avais écrit que « la cellule ordinaire est exacte à l'arrondi près, il n'y a plus
rien à gagner sans changer de flottant ». C'était faux, et l'erreur était d'un facteur **220 000**.

Le compte est pourtant simple : 1.4e-07 ≈ 2.4 · 2⁻²⁴. Ce n'est pas un reste de géométrie, c'est
**l'epsilon du `float` tout nu** — donc un arrondi de stockage, pas un arrondi de calcul. Et il
n'y en a qu'un : le raffinement résout le sommet en `double`, à `2⁻⁵³` près, puis

```cpp
x[ i ] = TK( vx );   // et on jette les vingt-neuf bits qu'on vient de gagner
```

**Le sommet résolu existe déjà dans un registre `double`.** Il suffit de s'en servir avant de le
jeter : la mesure est prise *dans* la boucle de résolution, et le lacet se ferme en **streaming**
— deux `double` pour le sommet précédent, deux pour le premier, cinq registres en tout. On ne
garde jamais les `R` sommets en `double`, ce qui coûterait 32 registres et l'occupation avec.
Les longueurs de facettes suivent le même chemin : `| v_{i+1} − v_i |` prise sur des sommets
`float` perd sa précision *relative* sur les arêtes courtes — deux grands qui donnent un petit,
encore — et c'est exactement la longueur presque nulle qui faisait diverger la hessienne.

C'est la variante `filmsk8m` (`SF_RES64=0` la désactive), et elle est **le défaut** du chemin de
Newton.

### Ce que ça donne (`--temoin-double`, `--reps-gpu 5`)

**Uniforme, `n = 10⁶`, Voronoï :**

| | médiane | p99.99 | max | ns/germe | registres |
|---|---|---|---|---|---|
| `filnrm8` | 3.2e-05 | 7.6e-04 | 4.8e-03 | 7.7 | 74 |
| `filmsk8h` (les trois réparations) | 1.1e-08 | 8.4e-08 | 1.4e-07 | 10.1 | 78 |
| **`filmsk8m`** (+ la mesure résolue) | **4.9e-14** | **4.9e-12** | 4.6e-08 | 10.6 | 94 |

**Médiane 1.1e-08 → 4.9e-14, pour +5 % de temps.** La cellule ordinaire n'est plus « exacte à
l'arrondi du `float` » : elle est exacte **à l'arrondi du `double`**, dans un noyau dont tous les
registres géométriques sont des `float`. Ce qui est cohérent, et qu'il faut dire pour ne pas
laisser croire à un miracle : le sommet EST calculé en `double` depuis des positions en virgule
fixe 64 bits — le `float` ne sert plus qu'à décider *quelles* coupes s'appliquent, et cette
décision-là n'a jamais eu besoin de précision.

### Ce que ça coûte en BANDE PASSANTE, et pourquoi la réponse est « rien »

La résolution **relit** les positions du voisin (`u64[0..1]`, 16 octets) et son poids (8 octets) :
24 octets par plan, ~6 plans par cellule. La question légitime est de savoir si ce sont des accès
DRAM. Le test décisif est la **loi en `n`** — si c'était de la bande passante, le surcoût
grandirait quand le jeu de travail sort du cache :

| uniforme, `float`, ns/germe | `n = 10⁶` | `n = 10⁷` |
|---|---|---|
| sans résolution (`SF_RAFF=0`) | 9.2 | 9.8 |
| + la résolution | 9.7 (**+5.4 %**) | 10.1 (**+3.1 %**) |
| + la mesure en `double` | 10.0 | 10.6 |
| **les deux** | **+8.7 %** | **+8.2 %** |

**Le surcoût ne grandit pas, il diminue.** À `n = 10⁷` les positions font 160 Mo — trente fois les
5 Mo de L2 — et la résolution coûte proportionnellement *moins*. La raison est structurelle :
**le parcours de l'arbre vient de lire ~25 plans candidats par cellule, la résolution n'en relit
que 6.** C'est un sous-ensemble de ce que le même warp a touché quelques microsecondes plus tôt,
donc des hits L2 ; ce qui coûte, c'est l'arithmétique `double`.

Et c'est le bon côté de l'arbitrage. L'alternative — garder les plans en `double` dans des
registres pendant la coupe — coûterait 24 registres par cellule (trois `double` fois huit), soit
bien plus de 8 % d'occupation. **On échange 24 octets de trafic caché contre 24 registres**, et
sur une carte c'est toujours le bon sens.

**Le max, lui, ne suit pas** : 4.6e-08 au lieu de ~1e-13. C'est le garde-fou `SEUIL_DET = 1e-6`
de `croise2` — quand les deux plans porteurs sont trop parallèles, on garde le sommet du noyau
plutôt que de le remplacer par un quotient qui explose. Une poignée de cellules par million
retombent donc au niveau `float` ; le p99.99 à 4.9e-12 dit combien peu. **Le seuil est désormais
le seul terme qui reste**, et c'est le bon endroit où regarder ensuite.

**Les deux nuages « lignes », `n = 10⁵` :**

| | Voronoï, dédupliqué | | aires égales, dégénéré | |
|---|---|---|---|---|
| | médiane | max | médiane | max |
| `filnrm8` | 3.4e-06 | 6.4e-04 | 2.0e-03 | 2.7e+00 |
| `filmsk8h` | 1.2e-09 | 4.2e-07 | 7.7e-07 | 7.1e-05 |
| **`filmsk8m`** | **5.8e-15** | **1.8e-10** | **1.0e-11** | **2.1e-07** |

Sur le nuage **propre**, `filmsk8m` rejoint le témoin `double` : 1.8e-10 de maximum, zéro cellule
fausse au débogueur. Sur le nuage **dégénéré** — conservé exprès, § 23.7 du banc CPU — il reste
2.1e-07 de maximum, et c'est la paire de germes à 10⁻⁸ qui le fixe.

## LES SOMMETS EN ENTIERS 32 BITS (`filent8`, `filent8m`)

La question posée était : jusqu'où peut-on aller en entiers dans la construction d'une cellule ?
`src/gpu/FilEnt2D.cuh` est `filmsk` à l'identique — mêmes masques de rôle, même barillet, même
résolution finale — mais la cellule vit sur la **grille `2⁻³⁰`** dans le repère du germe, et la
coupe s'écrit **sans un seul arrondi** :

```
2 ( du . V )  ≤  |du|² + ( W₀ − Wⱼ )        du = uⱼ − u₀,   W = w · ECH_FIXE²
```

**Où vivent les `w`, enfin.** `w` a la dimension d'une longueur au carré, donc sa grille naturelle
est le **carré** de celle des positions : à `2⁻³⁰` sur `u`, c'est `2⁻⁶⁰` sur `w`. Avec cette
échelle `|du|²` et `W₀ − Wⱼ` sont sur la MÊME grille et l'expression est homogène sans facteur
correctif. Il n'y a rien à régler, et c'est la réponse à « les ordres de grandeur des `w` sont
moins maîtrisés » : ils le sont exactement autant que ceux des positions, une fois la bonne
puissance de deux choisie.

Toutes les bornes tiennent sous `2⁶³` (`|du| ≤ 2³⁰`, `|V| ≤ 2³⁰`, poids écrêté à `2⁶¹`) et la
comparaison se fait **sans soustraction** — `2 l > off` — pour qu'aucun intermédiaire ne déborde
quel que soit l'éloignement du plan. Deux `IMAD.WIDE`, une addition, une comparaison.

**Ce qui reste approché, et pourquoi c'est volontaire.** L'interpolation `s₀ / (s₀ − s₁)` est en
`float` : l'exiger exacte demanderait une division de 71 bits par 40, que la carte n'a pas, et ça
ne servirait à rien puisque **les sommets sont résolus** à la fin. L'élagage reste en `float` : il
est conservatif par construction. L'aire sur la grille, elle, est exacte — accumulateur
`__int128`, parce que la somme des produits croisés ne tient pas dans 64 bits.

### Les chiffres

**Uniforme, `n = 10⁶` :**

| | médiane | max | ns/germe |
|---|---|---|---|
| `filmsk8h` | 1.1e-08 | 1.4e-07 | 10.1 |
| `filmsk8m` | 4.9e-14 | 4.6e-08 | 10.6 |
| `filent8` (mesure sur la grille) | 1.9e-07 | 1.7e-06 | 11.8 |
| `filent8m` (mesure résolue) | **4.9e-14** | 5.4e-08 | 12.0 |

**Trois conclusions, et elles sont nettes.**

**(a) La grille entière SEULE est un recul, d'un facteur 17.** `filent8` fait 1.9e-07 de médiane
contre 1.1e-08 pour `filmsk8h`. La raison est structurelle et se calcule d'avance : la grille est
celle de **la boîte**, pas celle de la cellule — le polygone part du carré unité, donc on ne peut
pas la resserrer. Le sommet porte `2⁻³¹` d'erreur ABSOLUE, quand le même sommet en `float` *dans
le repère du germe* en porte `h · 2⁻²⁴`. Les deux se croisent à `h = 2⁻⁷`, soit **n ≈ 1.6 · 10⁴** :
en dessous l'entier gagne, au-dessus le repère du germe gagne, et à `n = 10⁶` il gagne d'un
facteur 8 sur le sommet — 17 sur l'aire.

**(b) Une fois la mesure résolue, la grille entière ne coûte plus rien en précision.**
`filent8m` = 4.9e-14, au chiffre près comme `filmsk8m`. C'est attendu et c'est le point : les
entiers ne servent qu'à **décider**, et la décision n'a besoin que de vingt bits sur une cellule
de `2⁻¹⁰`. Le prix est **+13 %** (12.0 contre 10.6 ns/germe).

**(c) Ce que ces 13 % achètent, c'est la cohérence, pas la précision.** Le masque `m` dit
désormais la vérité sur les sommets tels qu'ils sont stockés : plus d'écart possible entre « le
bit dit dehors » et « la géométrie dit dedans », donc l'arc extérieur reste contigu, le polygone
reste convexe, l'ordre cyclique reste juste. Sur ces nuages ça ne se voit pas — `filmsk8m` ne
rate rien. C'est une assurance, et il faut la juger comme telle.

### Et le contre-exemple, qui est le plus instructif

Sur `lignes / aires égales` — le nuage dégénéré — **`filent8` et `filent8m` sont FAUX**, avec un
écart maximal de **1.0**, pendant que `filmsk8m` y fait 2.1e-07 sans une seule cellule fausse.
Le débogueur donne le compte exact : **une cellule vidée et quatre fausses sur 100 000**, de rangs
35999 / 36000 et 36159 / 36160 — des paires **adjacentes dans l'arbre**, donc voisines dans
l'espace. C'est la paire de germes à `δ = 10⁻⁸`, et sa voisine.

Le calcul dit pourquoi, et il tient en une ligne : **`δ = 10⁻⁸` fait 11 pas d'une grille à
`2⁻³⁰ = 9.3e-10`.** `filmsk8h/m`, qui lit les positions en virgule fixe **64 bits**, voit la même
paire à 4.5 · 10⁶ pas. Quantifier les positions à `2⁻³⁰` *fusionne* la paire, et une cellule
avale l'autre — d'où l'écart de 1.0, qui est exactement « deux fois l'aire attendue » sur le
survivant et zéro sur l'autre.

**Et le `double` n'y échappe pas, il recule seulement l'échéance.** Sous un noyau `double` avec
`FIXE = 64`, les positions sont quantifiées à `2⁻⁵² = 2.2e-16` ; pour la même paire à `10⁻⁸`
cela fait `2.2e-08` d'erreur RELATIVE sur `Δp`, donc sur la normale du plan — et le banc rend
2.1e-07 d'écart maximal sur ce nuage, **en `double`**, contre 1e-13 partout ailleurs. Ce n'est
donc pas une faiblesse des entiers : c'est la propriété de toute grille fixe face à une paire
assez proche, et le seul remède est en amont.

**C'est la leçon centrale de l'exercice** : l'arithmétique exacte sur une entrée mal quantifiée
donne une réponse exactement fausse. La discipline EGC — quantifier l'entrée UNE fois, puis être
exact dessus — n'a de sens que si la quantification est fine devant **la plus petite distance du
nuage**, et pas devant l'espacement typique. Ce qui rejoint directement ce que
`solvers_des_familles` vient d'établir (§ 23.10) : **la grille entière devient sûre exactement
quand les diracs trop proches ont été agrégés en amont** — c'est la même précondition, pour la
même raison.

### Conséquence pour le choix de carte

Elle est renversée, et deux fois. **Le `float` n'est plus disqualifié par le Laguerre à poids
forts** : sur `lignes / aires égales` la médiane passe de 5.4e-4 à 7.7e-7 avec les trois
réparations, puis à **1.0e-11** avec la mesure résolue. Et sur le chemin de Newton, le `float`
ne se contente plus de suivre le `double` : **il finit sur le même résidu**, 3.700e-12 contre
3.699e-12, avec les mêmes 162 itérations de CG ([§ ce qui reste](06-ce-qui-reste.md)).

Le verdict « `double` obligatoire, donc H100 / A100 » ne tient donc plus du tout, et la **RTX PRO
6000** devient le bon choix pour le transport optimal aussi — le FP64 à 1/32 ou 1/64 n'est plus
sur le chemin critique. Ce qui n'a pas changé : l'arbre et la mémoire. Ce qui est devenu faux et
qu'il faut retirer partout : « le `float` seul ne descend pas sous ~1e-7 de résidu, la dernière
itération restera en `double` ». Il n'y a plus de dernière itération à basculer.

**Ce que le `float` n'achète toujours pas**, en revanche, c'est la **topologie** : 44 facettes
manquantes et 53 en trop sur 6·10⁶ à `n = 10⁶` (1.1e-09 du poids de la hessienne), identiques
avant et après la réparation, nulles en `double`. C'est la seule chose qu'un prédicat exact
achèterait — et c'est ce que mesure `filent8` ci-dessus.

---

[← sommaire](../README.md)
