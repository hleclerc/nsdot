# LES MAPPAGES

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
n'y a plus que des coupes inutiles, et là tout est du SIMD pur. Mesuré et perdu, [§ profils](05-profils.md) dit pourquoi.

**Deux passes.** À deux cases par voie (64 sommets) 0,02 % des cellules débordent en cours de
route (223 sur 10⁶ en uniforme) ; à quatre cases le code et les registres doublent et tout le
monde paie (681 contre 247 ns/germe). Donc : deux cases pour tout le monde, les rangs qui ont
débordé poussés dans une liste par atomique, et une seconde passe à quatre cases sur cette liste.
La cellule n'étant écrite qu'à la fin d'une coupe, une cellule qui déborde est restée intacte
jusqu'au débordement et repart de zéro proprement.

---

---

[← sommaire](../README.md)
