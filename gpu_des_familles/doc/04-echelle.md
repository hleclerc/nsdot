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

Deux réserves qui comptent.

* **La virgule fixe 31 bits est réservée au `float`.** En `double` elle *détruit* la précision
  (1.2e-4 au lieu de 2.1e-10 : 31 bits contre 53) — mesuré, `filmsk8f` en `double` est faux. Pour
  le `double` il faudrait une virgule fixe 64 bits, donc huit octets.
* **Le Laguerre à poids forts reste hors-jeu en `float`** : médiane 5.4e-4 dès 10⁵ et p99 à 12 %,
  même avec la virgule fixe. Or c'est exactement le cas du transport optimal.

**Conséquence pour le choix de carte.** Pour du Voronoï ou du Laguerre à poids faibles, le `float`
devient défendable à 10⁹ et la **RTX PRO 6000** reprend l'avantage (1.65 contre 3.5 ns/germe
projetés, 96 Go, et un L2 qui tiendrait l'arbre). Pour du Laguerre à poids forts — le transport
optimal — le `double` reste obligatoire et c'est **H100 / A100**. Le choix de carte est donc en
réalité un choix sur le *type de problème*, pas sur le noyau.

---

[← sommaire](../README.md)
