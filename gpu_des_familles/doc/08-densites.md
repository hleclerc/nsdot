# LES DENSITÉS, ET L'IMAGE

Jusqu'ici la source du transport était **Lebesgue** : mesurer une cellule, c'était son aire. Une
application réelle donne une **image** — une densité constante par pixel sur une grille régulière —
et il faut alors, pour chaque cellule, `∫_cellule ρ`, plus `∫_facette ρ ds` pour chaque arête,
parce que **le coefficient de la hessienne n'est plus la longueur de la facette** dès que la source
n'est pas uniforme.

    xmake run chaine --threads 8 --arbre-gpu --image 2048 --densite toutes
    xmake run chaine --threads 8 --arbre-gpu --image-pgm photo.pgm --newton 2000 --image-etapes 8

---

## 1. NE PAS DÉCOUPER

La façon naturelle de mesurer une cellule sous une image est de la **découper par les bords de
pixels** et de sommer `ρ × aire`. C'est ce que fait le moteur CPU de référence (`sdot`,
`Image::_for_each_piece`) : pour **chaque** pixel de la boîte englobante, il recopie la cellule
entière et refait `2d` coupes. Le coût est en `pixels DANS la cellule`, et mesuré ailleurs il
multiplie le temps par 1.5 en 2D et 2.4 en 3D quand les cellules font la taille d'un pixel.

Ici on ne coupe rien. Green donne

        masse( P ) = ∫_P ρ dA = ∮_(∂P) G( x, y ) dy,      G( x, y ) = ∫_0^x ρ( t, y ) dt

et pour une **image**, `G` est connue en fermé : sur la ligne `j`, `ρ` ne dépend plus de `y`, donc

        G_j( x ) = S[ j ][ i ] + ρ[ j ][ i ] ( x − i hx ),    S[ j ][ i ] = hx Σ_(k<i) ρ[ j ][ k ]

la **somme préfixe de la ligne**, calculée une fois à la montée de l'image. Sur un morceau d'arête
qui reste dans le pixel `( i, j )`, de `( xa, ya )` à `( xb, yb )`, `G` est **affine en `x`** : son
intégrale en `y` est donc exacte au point milieu,

        ( yb − ya ) ( S[ j ][ i ] + ρ[ j ][ i ] ( ( xa + xb ) / 2 − i hx ) )

Le coût passe de `pixels dans la cellule` à `pixels sous le bord` — sans une coupe, sans une copie,
et le parcours tient en un Amanatides-Woo de quatre lignes (`src/gpu/Image2D.cuh`).

**Deux sorties pour une seule marche.** La même sous-arête, le même pixel, donne aussi
`∫ ρ ds` — le coefficient de hessienne. C'est la raison de fond pour laquelle tout le fichier
travaille **arête par arête** plutôt que cellule par cellule.

**La référence `sref`.** Sur un polygone fermé `Σ dy = 0` : retrancher une constante à `G` ne change
rien. Sans elle les termes sont d'ordre `taille de cellule × S ≈ 1e−3` pour une masse qui vaut
`1e−6` — trois chiffres perdus par annulation. On retranche `S` au premier sommet, et les termes
tombent à l'ordre de la masse.

---

## 2. CE QUE LA MÉTHODE VAUT, À MACHINE ÉGALE

Le témoin du banc découpe la cellule en pixels (Sutherland-Hodgman, `main_chaine.cpp`) : il n'a
**pas une ligne commune** avec le noyau, ce qui est le seul moyen qu'il témoigne de quelque chose.
Mais on a aussi écrit l'intégrale de bord **sur le CPU**, sur les mêmes cellules du même moteur, ce
qui sépare ce qui revient à l'**algorithme** de ce qui revient à la **machine** :

| `double`, CPU 8 fils | découpage en pixels | intégrale de bord | |
|---|---|---|---|
| 10⁶ germes, image 512² | 0.332 s | 0.179 s | **×1.9** |
| 10⁶ germes, image 16384² | 6.080 s | 0.364 s | **×16.7** |
| 10⁵ germes, image 16384² | 4.528 s | 0.080 s | **×56.9** |
| 2.5·10⁴ germes, image 16384² | 3.889 s | 0.042 s | **×93.1** |

Le rapport suit la **surface contre le périmètre** : plus la cellule couvre de pixels, plus le
découpage paie cher ce que le bord ne paie pas. C'est un gain d'algorithme, pas de matériel.

---

## 3. CE QUE LA DENSITÉ COÛTE SUR LA CARTE

Mesures + facettes, `double`, uniforme, RTX 2080 Ti, en ns par germe. `W / √n` est le nombre de
pixels que traverse un côté de cellule — la seule grandeur qui compte.

| `n` | sans image | 512² | 2048² | 8192² | 16384² |
|---|---|---|---|---|---|
| 10⁶ | **96.6** | 103.9 (`W/√n` = 0.5) | 101.6 (2.0) | 110.4 (8.2) | 120.5 (16.4) |
| 10⁵ | 103.4 | — | — | 142.3 (25.9) | 180.4 (51.8) |
| 2.5·10⁴ | 136.2 | — | — | — | 322.8 (103.6) |

Soit **+7 % à 10⁶ pour une image 512²**, +14 % pour 8192², +25 % pour 16384². Sur les nuages de
lignes, image 4096² : Voronoï 134.0 → 161.1 (+20 %), aires égales 502.2 → 551.4 (+10 %).

**La densité image est presque gratuite tant que la cellule ne couvre pas des dizaines de pixels.**
Elle ne le devient pas par magie : le noyau est dominé par le parcours de l'arbre et par les
coupes, et la marche dans la grille se greffe dessus avec une très bonne localité — deux threads
voisins font des cellules voisines, donc lisent des pixels voisins.

---

## 4. LES PHASES : ÉCRITES, MESURÉES, ET ELLES NE PAIENT PAS

L'idée était bonne à poser : une fois la cellule finie, ou bien on intègre **sur place**, ou bien
on **dépose** le polygone pour un second noyau qui traitera le lot avec moins de divergence — et
par lots, parce qu'à 10⁹ germes on n'a pas la place de garder tous les polygones (128 octets par
cellule). Trois modes, un paramètre de template (`DENS_AUCUNE` / `DENS_DIRECTE` / `DENS_DEPOT`) :

| | ce que fait le noyau des cellules | ce que fait le second noyau |
|---|---|---|
| `directe` | l'arbre, les coupes, **et** la marche dans la grille | — |
| `depot` | l'arbre, les coupes, puis **écrit** les 8 sommets en SoA | une voie par cellule |
| `depot-arete` | idem | **une voie par arête**, 8 voies par cellule, somme par `__shfl_down_sync` |

Le troisième est le seul qui attaque vraiment la divergence : le coût du warp devient le **max sur
les arêtes** au lieu de la **somme sur les arêtes de la cellule la plus lente**.

Rapporté à `directe`, sur seize configurations mesurées (uniforme 2.5·10⁴ … 10⁶, images 512² …
16384², les deux nuages de lignes) :

| | `depot` | `depot-arete` |
|---|---|---|
| `double` | **×0.88 à ×1.00** | ×0.97 à ×1.03 |
| `float`, 10⁶, 2048² | ×0.89 | ×0.89 |

**Le dépôt ne gagne jamais, et perd jusqu'à 12 %.** Les deux raisons qu'on lui prêtait tombent
toutes les deux :

* **Les registres ne bougent pas.** `filmsk` avec facettes fait 116 registres ; avec la densité
  intégrée, **114** — la marche n'en coûte aucun, elle en économise même deux (elle remplace le
  lacet et la racine carrée des longueurs). Les deux donnent 4 blocs par SM et 50 % d'occupation.
  Il n'y a pas de marche d'occupation à sauver. En `float` le dépôt fait bien monter l'occupation
  (71 registres et 88 %, contre 86 et 62 %) — **et il perd quand même 11 %** : la preuve que ce
  n'est pas l'occupation qui limite.
* **La divergence ajoutée est petite.** Le noyau diverge déjà beaucoup plus sur le parcours de
  l'arbre que sur la marche dans la grille ; mettre celle-ci à part enlève une variance mineure et
  paie en échange un aller-retour complet du polygone par la mémoire globale.

En `float`, le dépôt **coûte en plus de la précision** : l'écart entre `directe` et `depot` est de
9.7e−4 (relatif à la cible) contre 8e−14 en `double`, parce que le polygone est **re-quantifié** au
passage par la mémoire au lieu de rester dans les registres.

Le mécanisme reste dans le code et il est correct (les trois modes s'accordent à 8e−14 en
`double`) : il servira le jour où le travail par cellule après la construction sera vraiment lourd
— ce n'est pas le cas d'une image en 2D.

---

## 5. LA HESSIENNE SOUS UNE DENSITÉ, ET COMMENT ON SAIT QU'ELLE EST JUSTE

Dès que `ρ` n'est pas uniforme,

        c_ij = ∫_(facette ij) ρ ds / ( 2 | p_i − p_j | )

et non plus `| facette | / ( 2 | p_i − p_j | )`. Comparer la hessienne du GPU à celle du CPU ne
prouve rien ici : les deux peuvent porter la même formule fausse. Le seul contrôle qui ne suppose
ni le signe ni le facteur est la **différence finie centrée** sur les mesures elles-mêmes :

    NEWTON  d m / d w = +L, verifie par difference finie centree :
            ecart median 1.4e-14, p99 1.0e-12, p99.99 1.8e-05, max 5.8e-03

Deux leçons au passage. Une différence **à droite** donnait 1.6e−3 sur une hessienne juste : la
troncature se confondait avec le verdict. Et il faut juger **en quantiles**, pas au maximum : sur
10⁵ cellules il y en a toujours quelques-unes au bord d'un changement de voisinage, où `m` n'est
plus deux fois dérivable et où aucune différence finie sur un `ε` fini ne peut rendre la dérivée.
C'est exactement la leçon déjà apprise sur les `c_ij`.

Le reste du contrôle est celui de la chaîne : somme des masses `1.000000000`, écart par cellule au
témoin par découpage ≤ 2e−9 (rapporté à la moyenne), **0 facette manquante, 0 en trop**, `c_ij`
médian 3e−14.

---

## 6. LE NEWTON SOUS UNE IMAGE : CE QUI BLOQUE EST **UNE** CELLULE

L'image de synthèse du banc (fond lisse + disque net + bande fine) a un contraste de **90 pour 1**.
À poids nuls le résidu vaut 1.89 au lieu de 0.53 en Lebesgue, et le Newton **n'avance plus** : pas
de 2e−4, et le résidu ne bouge pas.

La trace (`CHAINE_DEBUG=1`) dit précisément pourquoi :

    essai t=3.125e-02 : plus petite 0.00e+00 ( seuil 3.29e-03 ) REFUS, residu 1.826458e+00 contre 1.840716e+00 ok
    ...
    essai t=9.766e-04 : plus petite 7.59e-03 ( seuil 3.29e-03 ) ok,    residu 1.867900e+00 contre 1.869020e+00 ok

**Ce n'est pas la décroissance du résidu qui borne le pas, c'est la mort d'une cellule.** À
`t = 0.03` le résidu baisse déjà franchement — et une cellule tombe à zéro. On perd un facteur
**trente** sur le pas pour **une** cellule. (C'est le même obstacle que le banc CPU avait rencontré
sur les densités concentrées, et qu'il avait réglé en relevant la cellule pincée toute seule.)

Trois corrections, mesurées :

1. **La recherche linéaire repart du pas précédent doublé**, pas de `1`. Sous une image le pas
   admissible est de l'ordre de `1e−3` et ne remonte que lentement : repartir de `1` coûtait huit à
   treize diagrammes d'essai **par itération** pour retrouver ce qu'on savait déjà.
   → 2322 diagrammes → **967**, 88.6 s → 68.8 s.
2. **Le plafond de la suite de forçage du CG passe de `1e−8` à `1e−2`.** Loin de la solution, une
   direction résolue à `1e−8` est du luxe pur. → 68.8 s → **35.3 s**, et un meilleur résidu final
   (2.9e−10 contre 3.4e−8). Sur le Lebesgue à 2·10⁵ : 1.64 s → **1.03 s**, à résidu identique. Le
   nouveau défaut.
3. **La continuation en contraste** (`--image-etapes K`) : on résout sur `ρ_s = ( 1 − s ) + s ρ`,
   `s` montant de `1/K` à `1`, chaque étape partant des poids de la précédente. À `s = 0` c'est
   Lebesgue, où tout va bien.

### Comment repérer les cellules à relever

L'instrument est gratuit : le noyau du résidu balaie déjà les `n` mesures pour faire `‖ m − ν ‖` et
le minimum. On lui ajoute un compteur atomique, la liste plafonnée des cellules sous le seuil, et
**trois histogrammes cumulés** (`BilanCel`). Un essai de pas ne rend plus un booléen, il rend une
distribution — et sur trois échelles, parce qu'elles ne disent pas la même chose :

| | ce qu'elle mesure | ce qu'elle distingue |
|---|---|---|
| `m / cible` | la santé absolue | une cellule saine d'une cellule en danger |
| `m / max` | la taille relative dans le diagramme | l'étalement de la distribution |
| `m / m avant` | **ce que le pas lui a pris** | « elle était déjà minuscule » de « ce pas la tue » |

Les comptes sont **cumulés** : la colonne « < f » donne le nombre de cellules sous la fraction `f`,
ce qui se lit directement comme « combien tomberaient sous `f` si je prenais ce pas ». `--echelle K`
trace le tout sur une échelle logarithmique du pas, aux `K` premières itérations de chaque étape.

**Première leçon : la mort est abrupte.** À l'échelle `m / m avant`, la case `= 0` et la case
`< 1/2` sont presque égales : les cellules ne rétrécissent pas, elles disparaissent. Il n'y a
quasiment pas de population intermédiaire à sauver.

**Deuxième leçon : le point de départ est sain.** À l'étape 1 de la continuation, avant le pas,
*zéro* cellule vide, **2** sous `10⁻²` de la cible, 485 sous `10⁻¹`. Les milliers de cellules qui
meurent ensuite n'étaient donc pas des malades : ce sont des cellules ordinaires, écrasées par
leurs voisines qui grandissent.

### Ce que le compte vaut vraiment : le croiser avec le résidu

Le compte seul ne décide de rien. Ce qui décide, c'est **combien il faut réparer pour gagner
combien**. Étape 1 de la continuation, `n` = 2·10⁵, résidu de départ 0.575 :

| `t` | cellules mortes | résidu atteint | Armijo |
|---|---|---|---|
| 1 | 5 282 | 0.335 | refusé |
| **1/2** | **1 685** | **0.318** | accepté |
| 1/4 | 355 | 0.437 | accepté |
| 1/8 | **6** | 0.503 | accepté |
| 1/16 | 0 | 0.539 | accepté ← **le pas réellement pris** |

Le pas que la recherche linéaire retient aujourd'hui fait passer le résidu de 0.575 à **0.539**.
Le pas que le résidu réclame (`t = 1/2`) le ferait passer à **0.318** — presque deux fois mieux, en
une itération, et le critère de décroissance le valide. Il n'est refusé que par **1 685 morts, soit
0.84 % des cellules**. Les relever coûterait ~34 000 constructions de cellule, c'est-à-dire **17 %
d'un diagramme**, alors que l'étape 1 en dépense 22 itérations complètes.

Ce n'est donc pas « trois cellules isolées », mais ce n'est pas non plus hors de prix : **le
relèvement vaut d'être essayé, et c'est dans la phase froide qu'il vaut quelque chose.**

**Une fois chaud, il ne vaut plus rien** — et pour une raison qui n'a rien à voir avec les
cellules. Étape 2, résidu de départ 0.168 :

| `t` | cellules mortes | résidu atteint |
|---|---|---|
| 1/4 | 1 416 | 0.205 |
| 1/8 | 197 | 0.171 |
| 1/16 | **1** | **0.160** ← le meilleur de toute l'échelle |
| 1/32 | 0 | 0.162 ← le pas pris |

Ici **une seule** cellule bloque, et la relever ferait gagner 1.4 % sur le résidu. Le meilleur pas
de toute l'échelle ne gagne que 5 % : **la direction elle-même est plate**, parce qu'on vient de
changer `ρ` et que le point est froid pour la nouvelle densité. Aucune réparation de cellule ne
répare ça.

Les deux régimes se résument ainsi : **là où relever rapporterait, il y a un millier de cellules à
relever ; là où il n'y en a qu'une, relever ne rapporte rien.** Le compte et le gain bougent
ensemble — ce qui était invisible tant qu'on ne regardait que le compte.

( À garder en tête pour l'essai : le banc CPU a mesuré cette idée (`--garde cellule`) comme
perdante — 113 diagrammes au lieu de 107, 351 au lieu de 332, 728 au lieu de 606 — parce que les
relèvements se concurrencent, chaque cellule relevée volant à ses voisines (la liste descend
619 → 249 → … → 105 en passes successives) et parce que le point réparé s'est révélé un plus
mauvais départ. Ce qui avait gagné là-bas est `limites_masse`, qui est autre chose : le pas
admissible **exact** au lieu des halvings. Son gain principal — tuer les 40 à 440 reculs par
itération — est déjà pris ici par la recherche linéaire qui repart du pas précédent : elle en fait
1 à 3. )

### Ce que la continuation vaut

`n` = 2·10⁵, image 512² :

| source | `K = 1` | `K = 8` |
|---|---|---|
| Lebesgue | 7 it., **1.0 s** | — |
| `¾ + ¼ ρ` ( ≈ 3:1 ) | **64 it., 11.2 s** | 85 it., 16.5 s |
| `½ + ½ ρ` ( ≈ 10:1 ) | **200 it., 38.2 s** | 193 it., 40.6 s |
| `ρ` ( ≈ 90:1 ) | 1341 it., 318 s | **662 it., 164 s** |

**La continuation ne paie qu'au contraste fort** (×1.9 à 90:1) et gêne légèrement en dessous : elle
ajoute des étapes là où une seule suffisait. C'est un bouton, pas un réglage par défaut.

Et le prix du contraste est **dans le solveur, pas dans la densité** : à 90:1 les 662 itérations
coûtent 93 s de diagrammes et 70 s de CG, alors que le même Newton en Lebesgue en coûte 7 au total.
Mesurer sous une image coûte +7 % ; **résoudre** sous une image coûte ×160.

---

## 7. CE QUI RESTE

* **Une échelle de continuation adaptée.** Les huit étapes coûtent 22, 34, 48, 61, 77, 96, 127 et
  197 itérations : la difficulté est toute entière dans le haut de la rampe. Un pas en `s` choisi
  sur le résidu atteint, plutôt qu'uniforme, est le chantier le mieux indiqué par les mesures.
* **Le pas admissible exact** au lieu des halvings (`limites_masse` du banc CPU) : une bissection
  par cellule sur `α`, toutes les cellules en parallèle, au lieu d'un diagramme complet par
  barreau. Son gain principal -- tuer les dizaines de reculs par itération -- est déjà pris par la
  recherche linéaire qui repart du pas précédent ; il resterait le facteur `√2` que coûte
  l'arrondi du pas à une puissance de deux. — La densité image est maintenant **portée sur le
  CPU** ([`solvers_des_familles` § 12](../../solvers_des_familles/README.md)), où `limites_masse`
  s'applique sans une ligne de changement : sous une image 512² à 60:1, il fait 448 diagrammes et
  **2 reculs** là où les essais repartant de `t = 1` en font 1041 et 940 sans converger. Le portage
  CPU sert aussi à instrumenter la simple précision, et il dit où elle casse (§ 12.6 là-bas :
  `S − sref`, jusqu'à 15 % sur une facette).
* **Relever les cellules écrasées, dans la phase froide.** Le compte croisé avec le résidu
  ( ci-dessus ) dit que le pas utile est bridé par 0.84 % des cellules, et que les relever
  coûterait 17 % d'un diagramme pour presque doubler la décroissance d'une itération. À essayer
  là, et seulement là : une fois chaud, une seule cellule bloque et la relever ne rapporte rien,
  la direction étant plate. Le banc CPU a mesuré l'idée comme perdante dans un autre régime --
  c'est la raison de la mesurer ici plutôt que de la supposer.
* La **3D** : le même raisonnement donne `∮_(∂P) G dS` avec `G` la primitive en `x` du voxel, et le
  parcours devient une marche sur les faces du polyèdre — plus de travail par facette, mais la
  même absence de découpage.
* Le **dépôt par lots** existe et marche ; il attend un cas où le travail par cellule après la
  construction est vraiment lourd.

---

[← sommaire](../README.md)
