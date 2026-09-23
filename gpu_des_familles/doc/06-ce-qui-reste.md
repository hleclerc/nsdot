# CE QUI RESTE

## LA CHAÎNE 2D SUR GPU ( cible : `chaine` )

Le squelette existe : `src/mains/main_chaine.cpp` fait l'arbre (CPU), le téléversement, puis
**les mesures ET les facettes en un seul noyau**, et compare au moteur CPU poste par poste. Les
facettes étaient le maillon manquant entre « mesurer des cellules » et « faire un Newton » : la
hessienne du transport est le laplacien du graphe de Laguerre, `c_ij = |facette| / ( 2 |p_i - p_j| )`
(`solver/Laplacien.h`), et ses coefficients sont **les arêtes de chaque cellule** — que le noyau
portait déjà, chacune avec le `cid` de son voisin. `noyau2_filmsk` et `noyau2_filmix` les écrivent
en SoA (`[ arête * n + identifiant ]`), `NF = 16` arêtes par cellule.

| `double`, 10⁶ | CPU 8 fils | GPU | |
|---|---|---|---|
| uniforme | 203 ns/germe, 5 994 454 facettes | **95 ns/germe, 5 994 454** | ×2.1, **0 manquante, 0 en trop** |
| lignes / Voronoï (10⁵) | 324 | 131 | ×2.5, 44 manquantes (3.0e-5 du poids) |
| lignes / aires égales (10⁵) | 907 | 481 | ×1.9, 244 manquantes (1.7e-5 du poids) |

En `float` sur l'uniforme : **23.7 ns/germe, ×8.5**, 261 manquantes et 251 en trop sur 6 M
(2.8e-8 du poids) — des arêtes quasi nulles que le `float` fait apparaître ou disparaître.

**Deux leçons de la mise au point.** D'abord, la première passe fait **déborder 13 % des cellules
en uniforme** (`nn > R` pendant la construction, pas à l'arrivée) : c'est `filmix` qui les finit, et
tant qu'il n'émettait pas de facettes il manquait 13 % de la hessienne — invisible sur les mesures,
qui étaient justes. Ensuite, **juger un `c_ij` sur lui-même ne veut rien dire** : une arête quasi
nulle a une erreur relative énorme et un poids nul ; le contrôle les rapporte au `c` moyen et pèse
les non appariées.

## L'ARBRE CONSTRUIT SUR LE GPU ( `src/gpu/Bsp2D.cu`, `--arbre-gpu` )

Le CPU est récursif : boîte englobante serrée, coupe **médiane par rang** sur l'axe le plus long
(`nth_element`), feuille dès que la tranche tient dans `leaf`. Le GPU fait la même chose **niveau
par niveau** :

1. **Les boîtes, par atomiques.** Chaque germe connaît son nœud et pousse sa position dans son
   min / max. Les flottants passent par un **codage entier ordonné** (bit de signe retourné,
   négatifs complémentés) qui rend `atomicMin` / `atomicMax` exacts — et qui sert **aussi** de clef
   de tri, gratuitement.
2. **L'axe le plus long**, un thread par nœud, et la décision feuille / coupe.
3. **Le tri.** Une coupe médiane par rang, c'est « trier la tranche et prendre le milieu ». Les
   tranches étant contiguës, **un seul tri radix global** sur la clef `( beg << 32 ) | coordonnée`
   trie toutes les tranches à la fois, sans segmentation. `beg` et non le numéro du nœud : il est
   constant sur la tranche **et croissant avec la position**, donc les tranches ne se mélangent
   pas — y compris celles des feuilles déjà finies, que leur clef basse nulle et la stabilité du
   tri laissent en place. C'est le seul poste coûteux : `log2( n / leaf )` tris de `n` clefs.
4. **Les fils**, deux par nœud coupé, pris sur un compteur atomique.

Le **préordre** vient après : on remonte les tailles de sous-arbre (niveaux à l'envers) puis on
redescend les indices (gauche = moi + 1, droit = moi + 1 + taille du gauche). La sortie a
exactement la forme que `Arbre.cuh` attend.

Le **majorant affine des poids** (`accel/WeightMajorant.h`, le seul endroit du banc où une idée a
rapporté un facteur) suit en **quatre passes** sur les germes, chacune une réduction par nœud faite
en atomiques — chaque germe remonte sa branche depuis sa feuille, `parent` donnant un saut par
niveau :

* **A** — les sommes et les extrêmes (`Σw`, `Σy`, `w` min / max) ;
* **B** — la matrice normale centrée et le second membre ; puis un thread par nœud résout le 2×2
  et applique les garde-fous de pente (une pente qui, sur l'étendue du nœud, dépasse de loin
  l'étalement des poids est un artefact du conditionnement) ;
* **C** — l'étalement des résidus `w − a·y`, qui décide d'accepter la pente ou de la jeter (seuil
  « nettement mieux que le hasard ») ;
* **D** — `b = max( w − a·y )` avec les pentes **arrondies**, plus la marge de quelques ulp.

Les atomiques flottantes ne somment pas dans le même ordre que le CPU, donc `a` peut différer d'un
arrondi : sans importance, `a` est un **choix** et `b` est calculé après lui, avec marge — le
majorant reste valide. Les min / max en `double` passent par le même codage entier ordonné, en 64
bits.

**Rien ne redescend** : `DiagrammeGpu` a un constructeur qui prend les positions et les poids bruts,
construit l'arbre sur la carte et garde nœuds, permutation, positions permutées et codes en virgule
fixe là où ils sont.

| `double` | arbre CPU (8 fils) | arbre GPU (mur) | noyaux seuls | nœuds | mesures + facettes |
|---|---|---|---|---|---|
| uniforme 10⁶ | 301 ms | 287 ms (×1.0) | **44 ms (×6.8)** | 262 143 = 262 143 | exact, **0 facette manquante** |
| uniforme 10⁷ | 5574 ms | **751 ms (×7.4)** | **474 ms (×11.8)** | 2 097 151 = 2 097 151 | exact, **0 manquante** |
| lignes / Voronoï 10⁵ | 16 ms | 5 ms (×2.9) | 3.8 ms | 32 767 = 32 767 | 44 manquantes (NF > 16) |
| **lignes / aires égales 10⁵ (Laguerre)** | 33 ms | **10 ms (×3.4)** | 7.1 ms | 32 767 = 32 767 | 2.0e-11, 244 manquantes |

**Le majorant affine fait bien son travail** : sur le nuage à aires égales, la mesure GPU coûte
508 ns/germe avec l'arbre GPU contre 481 avec l'arbre CPU, soit +6 % — alors qu'un majorant
constant coûterait, lui, un facteur (466 boîtes testées par cellule contre 135, `2d_des_familles`
§ 6). Les pentes sont donc trouvées, à quelques arrondis près de celles du CPU.

### Le régime de Newton : `refresh_poids` et les tampons gardés

Dans une boucle de Newton **les positions ne bougent pas, seuls les poids changent** : il n'y a
donc ni tri, ni boîtes, ni permutation à refaire — seulement les majorants. `pour_newton = true`
à la construction garde l'atelier (structure des nœuds, place de chacun, branche de chaque germe,
accumulateurs, positions en `double` dans l'ordre de l'arbre), et `refresh_poids( W )` rejoue les
seules passes A→D. **Pas une allocation.**

| lignes / aires égales, 10⁵ (Laguerre) | |
|---|---|
| construction complète, CPU 8 fils | 34 ms |
| construction complète, GPU (noyaux) | 7.1 ms |
| **majorants refaits seuls, GPU** | **3.27 ms (×10 sur le CPU)** |

Et `tour_newton( W )` fait le tour complet sur la carte — poids, majorants, mesures **et**
facettes — sans qu'un octet redescende. C'est le coût de régime, celui qui compte pour un solveur,
et le banc le mesure sur `--iterations K` après une seule construction :

| `double`, 20 tours | GPU par tour | CPU par tour | |
|---|---|---|---|
| uniforme 10⁶ (Voronoï) | **97 ms** | 183 ms | ×1.9 |
| lignes / Voronoï 10⁵ | **13.7 ms** | 18.1 ms | ×1.3 |
| lignes / aires égales 10⁵ (Laguerre) | **55.4 ms** | 82.6 ms | ×1.5 |

Les ×1.3 à ×1.9 ne sont pas flatteurs, et ils disent où en est la chaîne : le tour est maintenant
**dominé par la mesure elle-même en `double`** (96 des 97 ms sur l'uniforme), c'est-à-dire par le
FP64 à 1/32 de Turing. Ni l'arbre, ni les majorants, ni le trafic ne pèsent plus. Sur une carte à
FP64 rapide le même tour tomberait vers 7 ms, et le rapport au CPU avec.

Ce qui reste ici : le temps de mur à 10⁶ est encore dominé par ~250 ms de frais fixes (contexte
CUDA et une vingtaine d'allocations), constants et donc invisibles à 10⁷ ; et les nœuds sont
écrits en `float` pour la boîte (élargie d'un ulp) là où le CPU la garde en `double`.

### La hessienne assemblee sur la carte

La hessienne du transport est le laplacien du graphe de Laguerre
(`c_ij = |facette| / ( 2 |p_i − p_j| )`, `L_ii = Σ_j c_ij`, `L_ij = −c_ij`). Les facettes sortent
déjà du noyau de mesure, `NF = 32` cases par cellule en SoA : l'assemblage est donc **compter,
scanner, remplir** — les trois passes d'un CSR, **sans un tri**.

* **compter** — une ligne par thread, `NF` lectures espacées de `n`, donc des voies consécutives
  lisent des adresses consécutives : tout est coalescé ;
* **scanner** — une somme préfixe exclusive (CUB) donne `row` ;
* **remplir** — la même boucle écrit `col`, `val`, et accumule la diagonale au passage.

Les colonnes ne sont pas triées : un gradient conjugué n'en a pas besoin (le CPU les trie pour un
solveur direct). Une ligne sans voisin est neutralisée à un, comme au CPU.

| `double` | GPU | CPU 8 fils | | coefficients |
|---|---|---|---|---|
| uniforme 10⁶ | **4.9 ms** | 59 ms | **×12** | 5 994 454 = 5 994 454 |
| lignes / Voronoï 10⁵ | 0.49 ms | 3 ms | ×5 | 594 694 = 594 694 |
| lignes / aires égales 10⁵ | 0.49 ms | 3 ms | ×5 | 594 300 = 594 300 |

**La vérification** est un produit `y = L x` sur un vecteur quelconque — un seul nombre exerce
toute la matrice — plus la propriété de noyau `L · 1 = 0`, ligne par ligne. Écart au CPU :
**7.0e-14 relatif** sur l'uniforme, 1.5e-13 et 1.4e-15 sur les lignes ; `| L · 1 |` à 5e-16
partout. `NF` est passé de 16 à 32 : à 16 il restait 44 et 244 cellules à plus d'arêtes sur les
nuages de lignes, et elles perdaient leurs facettes. **Il n'en reste aucune.**

### Le tour de Newton complet

`tour_newton( W )` puis `assemble( H )` font le tour entier sur la carte — poids, majorants,
mesures, facettes, hessienne — sans qu'un octet redescende. `--iterations K` le mesure après une
seule construction :

| `double`, 20 tours complets | GPU par tour | CPU par tour | |
|---|---|---|---|
| uniforme 10⁶ | **101 ms** | 408 ms | **×4.0** |
| lignes / Voronoï 10⁵ | **13.9 ms** | 28.1 ms | ×2.0 |
| lignes / aires égales 10⁵ (Laguerre) | **54.7 ms** | 92.6 ms | ×1.7 |

Sur l'uniforme, les 101 ms se répartissent en **96 de mesure**, 5 d'assemblage, et rien d'autre :
ni l'arbre (construit une fois), ni les majorants (3.3 ms quand il y a des poids), ni le trafic.
Le tour est donc **borné par le FP64 à 1/32 de Turing**, et c'est tout ce qui reste à gagner ici.

### Le gradient conjugué préconditionné

**La jauge.** Le laplacien a les constantes pour noyau — ajouter la même chose à tous les poids ne
change aucune cellule — donc il est singulier. Le CPU raie la ligne et la colonne zéro
(`crs_reduit`) ; ici on fait la même chose **sans rien recopier** : le produit saute la colonne
zéro et rend zéro sur la ligne zéro. L'opérateur est alors défini positif sur `{ x : x₀ = 0 }`, et
comme `b₀ = 0` et `x = 0` au départ, **tous** les vecteurs du CG y restent.

**Le préconditionneur est Jacobi** : la diagonale est déjà assemblée. **Les réductions** tiennent
en un noyau — réduction dans le warp par `__shfl_down`, une case partagée par warp, puis un seul
atomique par bloc — et **les scalaires restent sur la carte** (`alpha`, `beta` sont calculés par un
thread) : seul le test d'arrêt redescend un nombre par itération.

Vérifié contre un CG Jacobi écrit dans le banc, **même algorithme**, jauge comprise :

| `double`, résidu relatif 1e-10 | itérations | GPU | CPU 8 fils | | écart des solutions |
|---|---|---|---|---|---|
| uniforme 10⁶ | 8167 | **11.76 s** | 314 s (8131 it.) | **×26.7** | 1.2e-11 relatif |
| uniforme 2·10⁵ | 3656 | 0.51 s | 19.3 s (3656 it.) | ×37.6 | 4.7e-13 |
| lignes / Voronoï 10⁵ | 3631 | **0.30 s** | 6.15 s | ×20.5 | 5.8e-12 |
| lignes / aires égales 10⁵ | 7981 | **0.67 s** | 13.5 s | ×20.2 | 1.9e-11 |

Les deux côtés font **le même nombre d'itérations à quelques unités près** et trouvent la même
solution à 1e-11 : le portage est juste.

**Mais l'algorithme est le mauvais, et c'est le résultat qui compte.** 8167 itérations à 10⁶ contre
3656 à 2·10⁵ : c'est la loi en √n d'un CG à préconditionneur diagonal sur un laplacien, dont le
conditionnement croît comme `n`. Résultat : **le CG pèse 11.8 s là où le tour de Newton complet en
pèse 0.10** — il est devenu, à lui seul, 99 % de l'itération. Le CPU ne fait pas cette erreur : sa
chaîne de production utilise un **multigrille algébrique** (AMGCL, ×4.9 sur Cholesky à 10⁶,
`solver/Lineaire.h`), dont le nombre d'itérations ne dépend pas de `n`.

Le ×27 sur le CG du CPU est donc réel mais trompeur : il compare deux fois le mauvais solveur. Le
chantier suivant n'est pas d'accélérer ce CG, c'est de lui donner un **préconditionneur
multi-niveaux** — la hiérarchie se construit une fois par motif (les positions ne bougent pas dans
un Newton) et se réutilise à chaque tour, ce qui est exactement le régime qu'on mesure ici.

### Le multigrille algébrique maison

**L'agrégation est gratuite, et c'est tout l'intérêt.** Les germes sont rangés **dans l'ordre de
l'arbre**, qui est une courbe remplissante : des rangs consécutifs sont voisins dans le plan.
Agréger, c'est donc `rang >> 2` — quatre germes par paquet, sans noyau d'appariement, sans
matching, sans compaction. Et comme les indices d'agrégat restent ordonnés par rang, le niveau
suivant s'agrège pareil (`a >> 2`). **La hiérarchie entière tient dans un décalage entier.**

Le grossier est le produit de Galerkin `A_c = Pᵀ A P` avec `P` constant par morceaux. Sur un
laplacien de graphe c'est encore un laplacien : il suffit de sommer les poids d'arêtes entre
paquets, et la diagonale est la somme de la ligne. Un triplet par arête, un tri (CUB), une
réduction par clef, et le CSR sort de là.

Cycle en V, Jacobi amorti (ω = 0.7) deux fois avant et deux fois après — même `ω`, donc
l'opérateur est symétrique et le CG l'accepte. **La jauge passe de `x₀ = 0` à moyenne nulle** :
c'est la bonne pour un multigrille (projeter est symétrique là où rayer une ligne ne l'est pas),
et `b = mesures − cible` est déjà de somme nulle.

| `double`, résidu 1e-10 | itérations Jacobi | **itérations AMG** | GPU | CPU 8 fils (Jacobi) | |
|---|---|---|---|---|---|
| uniforme 10⁶ | 8167 | **357** | **3.22 s** | 314 s | **×97** |
| uniforme 2·10⁵ | 3656 | **168** | 0.33 s | 19.6 s | ×59 |
| lignes / Voronoï 10⁵ | 3631 | **133** | 0.23 s | 6.5 s | ×28 |
| lignes / aires égales 10⁵ | 7981 | **173** | 0.30 s | 13.7 s | ×46 |

**×20 à ×46 d'itérations en moins**, la hiérarchie se montant en 15 ms à 10⁶. Les solutions
coïncident avec le CG du CPU à 1e-10 près, une fois les deux jauges recentrées.

### Le K-cycle, et la comparaison à AMGCL

Le V-cycle ci-dessus laissait un défaut net : **168 itérations à 2·10⁵ et 357 à 10⁶**, donc une
croissance encore en √n, et descendre à 16 inconnues au lieu de 1000 la faisait passer à **411**.
Les deux symptômes disent la même chose : *la correction grossière est trop faible et l'erreur
s'accumule d'un niveau à l'autre*. Deux remèdes classiques :

* **la prolongation lissée** remplace `P` (une marche d'escalier : tous les germes d'un paquet
  reçoivent la même valeur) par `( I − ω D⁻¹A ) P`, c'est-à-dire un pas de Jacobi appliqué à
  *l'opérateur d'interpolation lui-même*, ce qui arrondit les marches. C'est ce que fait AMGCL.
  Le prix est un vrai **produit triple creux** `P̂ᵀ A P̂`, lourd sur GPU en mémoire comme en code ;
* **le K-cycle** ne touche ni à `P` ni au Galerkin : au lieu d'appeler récursivement le niveau
  grossier *une fois*, on y fait **deux pas d'un gradient conjugué** dont le préconditionneur est
  le niveau d'en dessous — on accélère chaque niveau par Krylov, récursivement. C'est l'idée
  d'AGMG, et elle vise exactement notre symptôme.

C'est le K-cycle qui est implémenté. Les coefficients des deux pas restent **sur la carte** (un
thread les calcule), sinon chaque niveau de chaque cycle coûterait une synchronisation. Deux
réglages comptent, et ils ont été balayés :

| (2·10⁵) | V pur | K sur 1 niveau | **K sur 2** | K sur 3 | K sur 4 |
|---|---|---|---|---|---|
| itérations | 169 | 82 | **67** | 61 | 61 |
| temps, 60 lissages au plus grossier | 269 ms | 148 | **143** | 214 | 356 |
| temps, 300 lissages | 332 ms | 285 | 396 | 721 | 1369 |

Le K-cycle visite le niveau `l` **2ˡ fois** : au-delà de deux niveaux accélérés, le coût des
visites mange le gain en itérations, et le niveau le plus grossier doit être lissé peu (60 fois,
pas 300) pour la même raison. **K sur deux niveaux, 60 lissages** : c'est le réglage retenu.

**Contre AMGCL, à plateforme égale.** Comparer notre multigrille GPU au CPU ne prouvait rien : il
fallait le comparer à **AMGCL sur la même carte**, avec son backend CUDA (`src/gpu/RefAmgcl.cu`) et
en balayant ses configurations. Le backend CUDA d'AMGCL n'accepte pas Gauss-Seidel (séquentiel par
nature) ; restent `spai0`, `damped_jacobi` et `chebyshev` comme lisseurs, et
`smoothed_aggregation`, `aggregation` (non lissée, comme la nôtre) et `ruge_stuben` comme
grossissements. Uniforme 10⁶, `double`, résidu 1e-10, **tout sur le RTX 2080 Ti** :

| | itér. | mise en forme | hiérarchie | résolution | **total** |
|---|---|---|---|---|---|
| AMGCL/CUDA agrégation lissée + spai0 | 49 | 62 ms | 3341 ms | **229 ms** | 3632 ms |
| AMGCL/CUDA agrégation lissée + Jacobi | 46 | 62 | 3422 | **211 ms** | 3695 ms |
| AMGCL/CUDA agrégation lissée + Chebyshev | 145 | 62 | 3342 | 1827 | 5230 ms |
| AMGCL/CUDA **agrégation non lissée + spai0** | 74 | 62 | 1782 | 331 | **2177 ms** |
| AMGCL/CUDA Ruge-Stuben + spai0 | 30 | 62 | 4230 | 164 | 4456 ms |
| **nous** | 85 | 0 | **14 ms** | 873 ms | **887 ms** |

**Sur le total, nous gagnons ×2.5** contre la meilleure configuration d'AMGCL. Mais le détail dit
autre chose, et il faut le dire :

* **notre mise en place est 130 fois moins chère** (14 ms contre 1782) — et c'est tout notre
  avantage. La raison est qu'**AMGCL construit sa hiérarchie sur le CPU** : ses temps de hiérarchie
  en CUDA sont ceux du CPU, parfois pires, alors que sa résolution, elle, profite bien de la carte.
  Chez nous l'agrégation est un décalage et le Galerkin est un tri sur la carte ;
* **notre résolution est 2.6 fois plus lente** (873 ms pour 85 itérations contre 331 pour 74).
  Par itération : 10.3 ms contre 4.5. Notre cycle coûte plus cher — le K-cycle visite le niveau `l`
  deux puissance `l` fois, et on lisse deux fois avant et deux fois après là où AMGCL lisse une
  fois avec `spai0`. C'est là qu'il reste un facteur deux à trois à prendre ;
* **la prolongation lissée d'AMGCL tient ses promesses sur la convergence** : 49 itérations contre
  74 pour la même agrégation non lissée, et 85 pour la nôtre. Notre K-cycle rattrape l'essentiel
  de l'écart mais pas tout.

**Et le contexte décide.** Dans un Newton, les positions ne bougent pas mais les *coefficients*
changent à chaque pas : la hiérarchie doit être remontée, et c'est notre 14 ms contre leurs
1782 ms qui compte. Si au contraire on pouvait figer la hiérarchie et ne refaire que les
résolutions, AMGCL passerait devant d'un facteur 2.6. Le balayage complet est rejouable :
`AMGCL_VAR` choisit la configuration, `AMG_K`, `AMG_NU` et `AMG_GROS` règlent la nôtre.

Notre propre balayage (2·10⁵) : **`AMG_NU = 1` ne converge pas** (le cycle se dégrade trop), deux
lissages avant et après sont le minimum ; `AMG_GROS` vaut mieux à 30 pour 2·10⁵ (118 ms contre 144)
mais à **60 pour 10⁶** (873 ms contre 1085) — c'est 60 qui est retenu, la grande taille décidant.

**Ce que le V-cycle seul ne faisait pas.****Ce que le V-cycle seul ne faisait pas.** 168 itérations à 2·10⁵ et 357 à 10⁶ : le nombre d'itérations croît
encore comme √n, avec une constante 22 fois meilleure. Un vrai multigrille serait indépendant de
`n` ; celui-ci ne l'est pas, parce que l'agrégation est **non lissée** — c'est sa faiblesse
connue, et elle se voit aussi en descendant plus bas : s'arrêter à 16 inconnues au lieu de 1000
fait passer de 168 à **411** itérations, la dégradation classique avec le nombre de niveaux.
C'est ce que le K-cycle a corrigé (§ ci-dessus).

**Ce qui reste, dans l'ordre.** 1) Les quelques dizaines de cellules à plus de `NF` arêtes sur les
nuages de lignes (compteur en place, `Chrono::deborde`). 2) L'assemblage CSR sur GPU (comptage +
somme préfixe, deux passes). 3) Le gradient conjugué préconditionné. 4) La boucle de Newton et sa
recherche linéaire. 5) Les densités, notamment l'image : intersecter la cellule avec la grille de
pixels. 6) Les frais fixes de la
construction (allocations en cache plutôt que refaites), et la 3D.


* **`filnrm8` est la référence 2D** (7.7 ns/germe en uniforme, ×19 ; 19 et 47 sur les lignes,
  `filmix6` à égalité sur Laguerre). Ce qui reste de divergence (6,8 actifs sur 32) est mesuré
  incompressible à peu de frais : l'oracle du tri par coût ([§ profils](05-profils.md)) dit +38 % de lanes au mieux,
  contre une localité par warp qui vaut 2× ; `filreg` / `filregc` à 8 registres et une excursion sont derrière. La piste
  suivante était celle que la seconde passe a révélée : des warps homogènes en taille de
  cellule — mesurée par l'oracle du [§ profils](05-profils.md) : elle ne vaut pas la localité qu'elle coûte. Ce qui reste : la divergence du
  parcours entre les 32 cellules d'un warp (6 actifs) — un tri des cellules par profondeur de
  parcours ou une pile en mémoire partagée ne changeraient pas le fond ; les 3 000 instructions
  par cellule sont à lire ligne à ligne comme pour le 3D.
* **Les paquets, le test en bloc, les deux fils au push, la boucle unique, les lanes
  persistantes, la rotation en mémoire partagée et le tri par coût sont mesurés et perdent**
  ([§ profils](05-profils.md)) : ne pas y revenir sans une idée neuve.
* **`filmsk8` est le même algorithme que `filnrm8` à 96 / 59 registres au lieu de 128 / 74**
  (−25 %), registres triés compris, pour +6 % de temps ([§ profils](05-profils.md)) : c'est le noyau à porter dans le
  noyau des phases, dont le seul frein restant est l'occupation. Le barillet unique y gagne 8 % à
  lui seul ; la cueillette par masques, elle, perd — un prédicat SASS est déjà un masque partagé
  entre `x`, `y` et `c`, et il coûte une instruction au lieu de trois.
* **L'occupation n'est pas le levier en 2D, et raboter les registres fait perdre** ([§ profils](05-profils.md)) :
  `filmsk8` est déjà à 100 % en `float` sans rien forcer et reste derrière `filnrm8` à 75 % ;
  forcer la puissance de deux en dessous fait déborder à tous les coups (jusqu'à +63 %). Le banc
  imprime désormais registres / blocs par SM / occupation / mémoire locale pour chaque variante,
  et `filnrm8c{6,8}`, `filmsk8c{6,8}` refont le balayage. Plancher de bruit du banc : ±2 %, et
  **±5 % sur les lignes Laguerre**.
* **L'ordre dans un registre (`filord8`) et la succession en masques (`filsuc8`) coûtent 21 % de
  registres en moins à vitesse égale** ([§ profils](05-profils.md)) ; leur limite est la longueur de la chaîne de
  dépendances, pas le nombre d'instructions.
  Dans le noyau des phases il manque **trois registres** (131) pour franchir le palier des quatre
  blocs par SM : sortir les `cid` des registres, ou coder `O` autrement, le ferait basculer —
  c'est le chantier le plus court à essayer.
* **À l'échelle ([§ échelle](04-echelle.md)), le noyau tient (+3 % par germe de 10⁶ à 3·10⁷) mais trois choses
  cassent** : la mémoire (33.6 o/germe en `float`, donc 31 Gio à 10⁹ — plusieurs GPU ou un
  découpage spatial), la construction de l'arbre sur CPU (22 s à 3·10⁷, ~15 min à 10⁹, contre 8 s
  de mesure GPU : c'est le `build` qu'il faut porter), et la précision en `float` (erreur par
  cellule en √n, ~1.5 % à 10⁹ — le `double` devient obligatoire). Le schéma par phases perd son
  avance à l'échelle (−14 % à 10⁶, −3 % à 3·10⁷) : sa fenêtre en vol est fixe.
* **Les phases (`filph8`, `filph8g`, `filph8b`, `filph8a`, `filph8c`, `filph8m`) sont écrites et
  mesurées** ([§ profils](05-profils.md)), groupement par feuille compris — par tri, par binning, par écriture directe en
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
  conclusions en dépendent ([§ profils](05-profils.md)) : « la rotation en mémoire partagée perd » et « le trafic des
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

---

[← sommaire](../README.md)
