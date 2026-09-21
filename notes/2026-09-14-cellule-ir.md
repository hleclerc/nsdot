# La cellule en production : la représentation intermédiaire, et le moteur du banc dans `sdot`

Session du 2026-09-14. Ce qui a été décidé, ce qui a été fait, ce qui reste.

## Le point de départ

Le banc `2d_des_familles` avait tranché l'algorithme : **la cellule dirige** (elle demande des
plans de coupe à un fournisseur, coupe, redemande), la cellule 2D tient dans **trois registres
SIMD** avec une machine à états sur son nombre de sommets, la 3D est un **polytope simple** porté
par ses sommets (`D` coupes et `D` voisins chacun), et l'accélérateur BSP est **retourné** en
fournisseur suspendu. Restait à le mettre dans `sdot`, et le premier obstacle était les tenseurs
qui représentaient les cellules : `vertex_positions [nv, d]`, `cut_directions / cut_offsets`,
`vertex_indices`, `edge_indices`, `is_fully_bounded`, `nb_edges` -- une H-représentation redondante,
un treillis d'arêtes, un drapeau -- plus, dans `PowerDiagram`, la navette `ws_0 / ws_1`, `corr`,
`facet_apex`, `grad_vp`, `acc_ws`, `piece_ws` : sept tenseurs de travail par work-item.

## Les décisions

* **pas de rejeu** depuis `cid` : le débordement du noyau à registres est une excursion en
  mémoire, le débordement de la mémoire remonte au tampon d'erreurs ;
* **FP32 dans le noyau par défaut**, `TF` (celui des positions) pour tout ce qu'on en tire :
  mesures, gradients. `kernel_dtype = "FP64"` au choix, `SDOT_KTYPE` pour le défaut ;
* **asimd** (vendored dans `sdot/include/asimd`, commit `dd45dad`) ;
* **un autre chemin sur GPU** : la boucle nue en mémoire, même représentation ;
* **le non-borné** : le simplexe de remplacement à parois `INFINITE` repoussées avant chaque coupe
  (`grow_for`), tenu **en mémoire** tant qu'il reste des parois -- le relais passe aux registres
  dès que la cellule est bornée (`Moteur.h::run`). Les « états jumeaux » non bornés en registres
  ne sont pas écrits : le régime est rare (pas de domaine), et il coûte les plans, que le régime
  borné ne garde pas.

## Ce qui est stocké, et ce que le noyau manipule

Deuxième passe, après relecture : le stockage Python est le format **pratique** -- un sommet par
ligne, dans le flottant de l'appelant -- et c'est le kernel qui le transfère vers sa propre forme à
chaque appel. Une classe par régime, avec exactement ses attributs (`Cell( nb_dims, ... )` choisit) :

    Cell_1   vertex_positions [ nv, 1 ]   cut_ids [ nc ]          un segment
    Cell_2   vertex_positions [ nv, 2 ]   cut_ids [ nv ]          ordre cyclique, la coupe i porte [ v_i, v_i+1 ]
    Cell_N   vertex_positions [ nv, d ]   cut_ids [ nc ]          polytope simple : d coupes et d voisins par sommet
             vertex_cuts [ nv, d ]        vertex_nbrs [ nv, d ]

Ni plans, ni arêtes, ni drapeau : les plans se relisent sur les sommets (`plane_of_edge`,
`plane_of_cut`) ou se refabriquent depuis les germes, les arêtes se lisent sur `vertex_nbrs`, et une
paroi `INFINITE` dans `cut_ids` dit que la cellule n'est pas bornée. `cut_planes`, `faces`, `edges`,
`is_bounded`, `vertices( i )` sont dérivés côté hôte.

La forme du noyau (`cell/Local1.h`, `Local2.h`, `LocalN.h` : SoA dans le flottant du noyau, les
temporaires de la coupe compris) vit dans **un seul tenseur de scratch** par work-item
(`Cell.py::CellScratch`, `cell/Scratch.h`), que le C++ découpe. Sa taille est décidée par l'hôte avec
la formule que les deux côtés partagent (`Local*::words_for` / `Cell_*.scratch_words`) :
**exactement** pour une opération de `Cell_*` -- une coupe ajoute au plus un sommet en 2D, et en 3D
Euler borne `V = 2 F - 4` sur un polytope simple, donc pas de second tour -- et une supposition que
loom double sur débordement pour un diagramme entier, comme pour toute capacité. Plus de constante de
compilation, plus de recompilation.

## Ce qui a été enlevé

* l'adjoint de `Cell.cut` et de `init_as_hypercube` (`cut_bwd`, `init_as_hypercube_bwd`) : rien
  d'autre que les tests ne s'en servait, et l'adjoint qui compte -- celui de `measures` par rapport
  aux germes -- ne passe pas par là ;
* `max_nb_cuts` (devient `scratch_capacity`, une supposition de départ), `thread_scratch` /
  `bytes_per_thread` des accélérateurs, `PieceWorkspace` à deux cellules.

## `PowerDiagram`, une vue sur un stockage

Le même principe que pour `Cell` : `PowerDiagram.py` est le CONTRAT ( `positions` / `weights` en
lecture et en écriture dans l'ordre de l'utilisateur, `measures`, `cells`, `cell( i )` ) et le
tronc commun ( le domaine, la distribution, le scratch, les trois kernels ) ; COMMENT les germes
sont rangés est l'affaire d'une spécialisation choisie à la construction :

* `PowerDiagram_Plain` -- les germes tels qu'ils sont venus, chaque cellule coupée par les `n - 1`
  bissectrices ( `FournisseurTous` ). Le plancher, et ce qui reste quand les positions sont un
  traceur ;
* `PowerDiagram_Bsp` -- `sorted_positions` / `sorted_weights` dans l'ordre de l'arbre `tree`
  ( `AaBsp`, un agrégat imbriqué ), une feuille se lisant d'un seul tenant ; `FournisseurBsp`
  émet le RANG comme identifiant de coupe, et c'est le kernel qui traduit vers l'ordre de
  l'utilisateur ( `user_id( k ) = tree.seed_indices( k )` ) pour les mesures, les cellules et
  les gradients. Le rassemblement `sorted = positions[ seed_indices ]` est une opération du
  backend, donc DÉRIVABLE : les dérivées par rapport à `sorted_*` reviennent sur `positions` /
  `weights` toutes seules.

Côté C++, `diagram/Ops.h` ne demande au stockage que `ct_dim`, `TF`, `has_weights`, `nb_seeds()`,
`point( k )`, `weight( k )`, `user_id( k )` et `fournisseur<TK>( k0 )` ( `diagram/Common.h` ).

Poser des poids neufs ( `pd.weights = w` ) sur un stockage BSP ne rebâtit rien : les poids sont
re-triés et le majorant affine de chaque nœud est refait en UN kernel batché sur les nœuds
( `AaBsp.refresh_weight_majorants`, `AaBsp.h::refresh_majorant` ), gradient coupé à l'entrée. C'est
ce dont `OtPlan` vit désormais : UN diagramme, bâti une fois, auquel chaque évaluation pose ses
poids -- y compris sous `driver.grad` / `jit`, où ce qui reste dans l'objet après la trace est un
traceur, sans conséquence puisque toute lecture repasse par `power_diagram( w )`. Poser des
positions rebâtit l'arbre. `AaBsp` disparaît de la signature de l'utilisateur ordinaire : sans
`accelerator`, l'arbre est bâti par le diagramme ; `"plain"` le refuse ; un `AaBsp` donné est
adopté ( son majorant refait sur les poids du diagramme ).

## Où c'est

    sdot/include/sdot/cell/     Ids.h Plane.h Etat.h Scratch.h Ops.h Local1.h Local2.h LocalN.h
                                Moteur.h Moteur2Reg.h Elagage.h Fournisseurs.h
    sdot/include/sdot/diagram/  Ops.h ( les trois opérations, génériques sur le stockage ) Common.h
    sdot/include/sdot/          Cell_1.h Cell_2.h Cell_N.h PowerDiagram_Plain.h PowerDiagram_Bsp.h
                                PieceWorkspace.h AaBsp.h ( refresh_majorant )
    sdot/src/sdot/              Cell.py ( le contrat et le tronc commun ) Cell_1.py Cell_2.py Cell_N.py
                                CellScratch.py cell_viz.py
                                PowerDiagram.py ( idem ) PowerDiagram_Plain.py PowerDiagram_Bsp.py AaBsp.py

## Les chiffres

`./run bench "test_PowerDiagram::pd accelerated" --nb-points=1000000 --plain=0 --leaf-size=10`,
Xeon W-2145, 16 threads, positions en FP64, le carré unité :

| | mesures |
|---|---|
| avant ( `Ct dim in shape and strides`, AOT `-O3`, 2026-09-02 ) | 0.979 s |
| pysdot, 16 threads | 0.397 s |
| ce commit, x86-64 de base ( `SimdVec<float,8>` en deux `xmm` ) | 1.00 s |
| **ce commit, `-march=native`** ( désormais le défaut sur la cible CPU ) | **0.208 s** |
| idem, noyau FP64 | 0.200 s |
| idem, Laguerre ( `--weights=1` ) | 0.246 s |
| 3D, 2·10⁵ germes | 0.457 s ( le banc : 0.685 s à 8 fils ; c'était 0.94 s avec la cellule en tableaux de pile à taille fixe -- le scratch a rendu ×2 en 3D ) |

Le banc C++ nu donnait 0.213 s sur le même cas : on y est. Le noyau en `double` ne coûte rien de
plus -- ce n'est pas lui qui borne.

`-march=native` est devenu le défaut de la cible CPU ( `adaptive_cpp._march_flags` ), et le nom
du `.so` porte les flags et le modèle de processeur ( `build_signature` ) : un cache partagé entre
machines ne sert plus un binaire AVX-512 à une machine qui n'en a pas. `SDOT_NO_MARCH_NATIVE=1`
pour un `.so` à emporter.

Le banc chronomètre désormais ce qu'un pas d'ajustement paie -- poser les poids ( tri + majorant )
puis mesurer -- sur un diagramme bâti une fois : mesuré à 10⁶ germes SOUS CHARGE ( load ≈ 8,
d'autres calculs sur la machine ) 0.264 s en Voronoï, 0.344 s en Laguerre, la construction
( `pd.weights = w` sur CPU ) valant ~0.05 s hors charge -- à remesurer sur une machine calme.

## Ce qui reste

* la 3D : l'élagage (`peut_couper_boite3` du banc) et la première passe de la coupe sont encore
  des boucles scalaires ici. La mesure par accumulation de faces est portée
  ( `LocalN::measure_3d` ) ; son adjoint reste l'éventail ;
* les états jumeaux non bornés en registres, si le régime sans domaine devient courant ;
* la construction de l'arbre ( 0.6 à 0.7 s à 10⁶ germes ) pèse maintenant TROIS fois le diagramme.

## 2026-09-21 : la mémoire (`memory`)

Le banc (`solvers_des_familles`, README § 11) a montré qu'en 3D proposer d'abord les voisins de
la cellule au dernier diagramme épargne la moitié des coupes effectives -- les transitoires, celles
qu'un germe proche fait avant qu'un vrai voisin ne le supplante -- pour -25 à -42 % du diagramme,
et que des souvenirs périmés (ceux de Voronoï sur un Laguerre) rendent encore -18 %. Porté ici :

* `PowerDiagram_Bsp` porte `memo_nbrs [ n, K ]` / `memo_counts [ n ]` en rangs de l'arbre,
  `num_memo` / `nb_memo` ; `PowerDiagram( ..., memory = K )`, défaut 32 en 3D+ et 0 en 2D
  (mesuré +13 % à 10⁶ en 2D : le polygone en registres se coupe pour moins que la pré-passe) ;
* `cell/Fournisseurs.h::FournisseurBsp<..., MEMO>` : la pré-passe (`ipre`), puis le parcours avec
  un curseur de saut par feuille (`isaut`, une recherche dichotomique à l'ouverture) -- pas de
  tableau par work-item, donc rien qui ne passe pas au GPU. Les deux curseurs sont DISTINCTS :
  un seul compteur pour les deux rôles fait re-proposer un souvenir à chaque feuille, et `cut`
  n'étant pas idempotente la cellule enfle jusqu'au débordement (mesuré : 140 sommets) ;
* `diagram/Ops.h::memorise` : après `integrate_into`, `tidy()` et les coupes vivantes ≥ 0 en
  insertion triée dans `memo_nbrs_out( k, . )` ; `memo_counts_out( k ) = 0` au-delà de `K`.
  Écrit dans deux tenseurs de SORTIE (entrées et sorties disjointes), repris par
  `_memo_after_call` -- sauf sous une trace, où l'ancienne mémoire reste ;
* `has_memo` est une constante de compilation (`memo_counts.is_valid()`), comme `has_weights`.

Mesuré : 3D 2·10⁵, Laguerre 0.450 → 0.337 s, Voronoï 0.436 → 0.307 s. Test :
`the_memory_changes_nothing_but_the_cost`.
