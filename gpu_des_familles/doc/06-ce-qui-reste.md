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

Ce qui reste ici : le temps de mur à 10⁶ est encore dominé par ~250 ms de frais fixes (contexte
CUDA et une vingtaine d'allocations), constants et donc invisibles à 10⁷ ; et les nœuds sont
écrits en `float` pour la boîte (élargie d'un ulp) là où le CPU la garde en `double`.

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
