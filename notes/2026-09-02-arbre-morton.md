# L'arbre du BSP : Morton + tri radix ?

Réponse à la question 4. Court : **oui, et je pense que c'est la bonne réponse — mais sous une
forme précise, où Morton est un ORDRE et non la règle de découpage.** Les deux se ressemblent et
n'ont pas du tout les mêmes conséquences sur le reste du code.

## Le problème, redit

Le build actuel est sériel PAR NIVEAU, et le parallélisme est pris sur les NŒUDS. Au niveau 0 il
y a un nœud : un seul work-item fait un quickselect sur 1e6 points. Mesuré sur la 2080 Ti,
1e6 germes 2D :

| niveau | nœuds | temps |
|---|---|---|
| 0 | 1 | 1.095 s |
| 1 | 2 | 0.983 s |
| 2 | 4 | 0.673 s |
| 3 | 8 | 0.619 s |
| 4–16 | 16 → 65536 | 1.14 s cumulé |

3.37 s sur 4.51 s dans les quatre premiers niveaux. Sur CPU le même travail coûte 0.35 s au total :
un cœur à 3.7 GHz avale un quickselect sur 1e6 points sans se plaindre, un thread CUDA non.

Le vrai défaut n'est pas le quickselect, c'est que **chaque niveau du haut doit REGARDER TOUS LES
POINTS**. Tant que la règle de coupe est « la médiane sur l'axe le plus long », il n'y a pas moyen
d'y couper : la médiane du niveau 0 est une fonction de tout le nuage.

## (A) Morton comme RÈGLE DE COUPE — l'arbre radix binaire (Karras/LBVH)

Le schéma classique du GPU : coder chaque point en Morton, trier, puis bâtir un arbre radix binaire
sur les codes triés — un work-item par nœud interne, **tous les niveaux à la fois**, aucune
dépendance sérielle du tout. Les boîtes remontent ensuite par une passe bottom-up.

Ce que ça casse, et c'est central : **les coupes deviennent DYADIQUES** (des milieux
géométriques fixes), donc l'arbre n'est plus équilibré en compte. Sa profondeur est bornée par le
nombre de bits (2 × 21 = 42 en 2D), pas par `log2( n / leaf )`. Or tout ce qui est écrit autour de
l'arbre s'appuie sur la forme équilibrée : `max_depth_for`, `max_nb_nodes_for`, la numérotation en
tas (`node_left = 2k+1`), la taille de la pile de descente, l'écriture par niveau côté kernel. Le
« aucune capacité à deviner » de la docstring, c'est exactement cette propriété-là. Il faudrait
tout rouvrir. Plus le traitement des points confondus (Karras s'en sort par un départage sur
l'indice).

## (B) Morton comme ORDRE, coupe au milieu de l'INDICE — ma recommandation

Trier les germes par code de Morton, puis bâtir l'arbre exactement comme aujourd'hui, sauf que la
coupe d'un nœud n'est plus « la médiane sur l'axe le plus long » mais **le milieu de sa tranche
dans l'ordre trié**. Alors :

* **la forme de l'arbre ne bouge pas d'un iota.** Toujours équilibré en compte, toujours
  `ceil( log2( n / leaf ) ) + 1` de profondeur, toujours un arbre binaire parfait numéroté en tas.
  `AaBsp.cxx` (la descente, la pile, `nearness`), `max_depth_for`, `thread_scratch`, tout le côté
  Python : **inchangés**. Seul `bsp_build_level.h` est remplacé.
* **plus rien ne regarde tous les points, sauf une fois.** Le travail en `O(n)` est fait UNE seule
  fois, dans le tri — qui est justement l'opération que le GPU fait le mieux (histogramme + scan +
  scatter, occupation pleine). Ensuite :
  * les tranches sont connues sans calcul : le nœud `k` du niveau `L` est `[ k·n/2^L, (k+1)·n/2^L )`,
    de l'arithmétique d'indices ;
  * les boîtes remontent **bottom-up** : celle d'un nœud est l'union des deux boîtes de ses
    enfants, donc `O(1)` par nœud. Le niveau 0 ne coûte plus 1.1 s, il coûte **deux `min` et deux
    `max`**. C'est là qu'est tout le gain, et il ne vient pas d'avoir mieux parallélisé la
    sélection : il vient de l'avoir supprimée.
  * seul le niveau des FEUILLES balaie encore des points (pour leurs boîtes), et il y a
    `n / leaf` feuilles — le régime parfaitement parallèle.

* **ce que ça coûte** : les boîtes sont moins serrées. Une coupe médiane sur l'axe le plus long
  sépare géométriquement ; une coupe au milieu de l'ordre de Morton sépare bien la plupart du
  temps, mais Morton a ses « grands sauts » (deux cases voisines dans l'ordre peuvent être loin
  l'une de l'autre quand un bit de poids fort bascule). Des boîtes qui se chevauchent davantage,
  c'est de l'élagage en moins, donc des coupes tentées pour rien. **C'est la seule inconnue, et
  elle se mesure directement** : le banc `pd accelerated` rend des ns/germe, un mauvais arbre s'y
  voit tout de suite.

  Si le chevauchement gêne : **Hilbert au lieu de Morton**, qui n'a pas de grands sauts. Le code se
  calcule un peu plus cher par point, mais il est calculé UNE fois puis trié — négligeable devant
  le tri. (C'est d'ailleurs ce que CGAL utilise pour son `spatial_sort`.)

* **le majorant affine des poids** se recompose bottom-up, ce qui n'est pas évident et vaut d'être
  dit. Les ÉQUATIONS NORMALES sont faites de moments — `Σ1, Σy, Σyyᵀ, Σw, Σyw` — et un moment est
  une somme : il se fusionne EXACTEMENT depuis les enfants. L'ajustement `a` devient donc `O(1)`
  par nœud. Le relèvement `b = max( w − a·y )`, lui, n'est pas additif (`a` change d'un nœud à
  l'autre) — mais il se majore depuis les enfants sans relire un point : sur l'enfant `c` on a
  `w ≤ a_c·y + b_c`, donc `w − a·y ≤ ( a_c − a )·y + b_c`, dont le max sur la BOÎTE de l'enfant se
  lit en `O(d)` (un `clamp` par axe, exactement comme `cell_may_be_cut`). Majorant valide, un peu
  plus lâche, `O(d)` par nœud. La construction entière devient alors : un tri, puis une remontée
  où aucun nœud ne relit ses points.

* **la dimension** : 2D sur 32 bits/axe et 3D sur 21 bits/axe tiennent dans un `uint64`, donc un
  tri radix ordinaire. Au-delà (on gère la 4D) Morton s'amincit — il faudrait soit moins de bits
  par axe, soit garder le build actuel en `d > 3`. Ça ne me gêne pas : le build actuel n'est un
  problème qu'en GPU, et la 4D en GPU à 1e6 germes n'est pas le cas qu'on court après.

## Le petit pas, si tu veux voir le gain avant de payer la réécriture

Garder la coupe médiane, et ne changer que les QUATRE PREMIERS niveaux : y remplacer le
quickselect par une sélection par histogramme, parallélisée **sur les points** (histogramme de la
coordonnée par (nœud, seau) → le seau qui contient la médiane → raffinement → partition par somme
préfixe). L'arbre produit est alors le MÊME, au bit près, donc le banc compare deux constructions
identiques et ne mesure que la parallélisation. C'est moins de code que (B), ça ne remet rien en
cause, et ça dit tout de suite combien vaut le poste. Mais ça reste un pansement : (B) supprime le
travail, là où le petit pas se contente de le répartir.

## Ce que je propose

1. le petit pas d'abord (mesure honnête du plafond),
2. puis (B) avec Hilbert, en mesurant le chevauchement à travers le banc.

À valider avant que je m'y mette — c'est le seul point de ta liste qui soit une vraie réécriture.
