# `gpu_des_familles` — le banc GPU

Le troisième banc. [`2d_des_familles`](../2d_des_familles/README.md) a cherché comment construire
un diagramme de puissance vite ; [`solvers_des_familles`](../solvers_des_familles/README.md) a
rangé l'engin qui a gagné et s'en sert pour essayer des solveurs. Ici on prend **le même arbre et
les mêmes nuages**, et on refait les cellules **en CUDA** — pour savoir comment mapper une cellule
sur un GPU, et ce que ça rapporte.

Le moteur CPU est le **témoin** : même nuage, même arbre, même flottant de noyau, l'écart maximal
par cellule et la somme des mesures sont imprimés à chaque ligne. Rien d'autre n'est vérifié, et
c'est suffisant : une coupe ratée ou une coupe en trop se voit sur la cellule.

```
xmake f -m release --cuda=/usr/local/cuda-13.3 && xmake      # nvcc 13.3 ( 13.4 s'effondre sur les noyaux générés )
xmake run mesures --threads 8                                 # la suite, 2D puis 3D, CPU témoin puis GPU par variante
xmake run mesures --threads 8 --kernel float --3d             # le flottant du noyau, des deux côtés
xmake run mesures --threads 8 --variante voies --load uniforme -n 1000000
```

Les options communes (`-n`, `--load`, `--kernel`, `--maxnv`, `--leaf`, `--cases`, …) sont celles
de `solvers_des_familles` (`src/bench/Args.h`), plus `--variante fil | filreg | filregc | filmix{4,6,8,12,16} | filbrk{6,8,10,12,16} | filbrk8nu | filrot{6,8} | filnrm8 | filord8 | filsuc8 | filmsk8 | filmsk8g | filmsk8f | filmsk8h | filmsk8m | filent8 | filent8m | filmsk8c{6,8} | filnrm8c{6,8} | filuni8 | filuni8np | filshm8 | filnrm8tri | filnrm8tril | filph8 | filph8g | filph8b | filph8a | filph8c | filph8o | filph8m | filph8m4 | voies | voies16 |
voies32 | paquet{8,32}x{1,2,4}[S] | toutes` et `--reps-gpu`. Sur la machine, `--threads 8` est à donner (`hardware_concurrency` rend 1
dans le bac à sable) et **tout chronométrage passe par `errand -x`**.

---

## Le sommaire

| | |
|---|---|
| [1. Où est quoi](doc/01-organisation.md) | les fichiers, ce que chacun tient |
| [2. Les mappages](doc/02-mappages.md) | une cellule par thread, par groupe de voies, par warp ; les phases |
| [3. Les chiffres](doc/03-chiffres.md) | le tableau des variantes, `float` et `double`, 2D et 3D |
| [4. L'échelle et la précision](doc/04-echelle.md) | jusqu'à 3·10⁷, ce que 10⁹ demanderait, le repère centré, la virgule fixe, **les trois réparations du `float`**, **quelle carte** |
| [5. Ce que le profil dit](doc/05-profils.md) | `ncu` : où va le temps, le bilan d'occupation, les idées mesurées et perdues |
| [6. Ce qui reste](doc/06-ce-qui-reste.md) | l'état des lieux et les chantiers ouverts |
| [7. Comment les chiffres sont pris](doc/07-methode.md) | `errand -x`, la chauffe, le témoin, le plancher de bruit |
| [8. Les densités, et l'image](doc/08-densites.md) | intégrer `ρ` **sur le bord**, ce que ça coûte, les phases mesurées, le Newton sous une image |

**Les trois résultats à retenir.** En 2D `float`, `filnrm8` fait **7.6 ns/germe** (×19 sur le CPU à
8 fils) et `filmsk8f` **8.2** avec 25 % de registres en moins, une occupation atteinte de 88 % et
une erreur divisée par 4 à 5 ; en 2D `double`, `filmsk8g` fait **96.7** ; en 3D `voies` fait
**229** (×8). Le temps par germe est **plat de 10⁶ à 3·10⁷**. Ce qui casse avant le noyau à 10⁹,
c'est la mémoire et la construction de l'arbre (sur CPU) — [§ échelle](doc/04-echelle.md).

**Et le `float` ne casse plus — il fait jeu égal.** Quatre réparations portées du banc CPU : la
**différence des poids portée en `double`**, le **repère du germe jusque dans la seconde passe**,
**chaque sommet résolu depuis les deux plans qui le portent** au lieu d'être interpolé, et enfin
— la plus rentable, et celle que j'avais manquée — **la mesure prise sur le sommet résolu**, sans
le faire repasser par le `float` avant de calculer l'aire. L'écart au témoin `double` sur
l'uniforme à 10⁶ tombe de 3.2e-05 à **4.9e-14 de médiane**, et de 2.0e-03 à **1.0e-11** sur le
Laguerre à poids forts qui était déclaré hors-jeu. Avec la **hessienne rendue symétrique au bit
près** — un bug d'une ligne qui faisait passer le CG de 71 à 20 000 itérations en `fp32` — **la
boucle de Newton entière tourne en `float` et finit sur le résidu du `double`** : 3.700e-12
contre 3.699e-12, mêmes pas, mêmes 162 itérations de CG, 2.79 s contre 4.24 à 10⁶.
[§ échelle](doc/04-echelle.md), [§ ce qui reste](doc/06-ce-qui-reste.md).

**Et la recherche de pas n'en est plus une.** `--pas limites` calcule **`alpha*`, le pas exact où
la première cellule touche le plancher d'aire**, par le polynôme de degré deux que l'aire suit le
long du rayon — en forme close, sans un essai. La passe ne reparcourt pas l'arbre : la
connectivité du diagramme courant suffit, et le sommet `i` est l'intersection des plans des arêtes
`i − 1` et `i`. Elle coûte **5 à 11 % d'un diagramme**, et son coefficient linéaire est vérifiable
exactement contre `−(L d)` (écart médian 4.6e-15 en `double`). Sur l'uniforme à 10⁶ : 18 → **13
diagrammes**. Sur le nuage dégénéré : la recherche par essais **ne converge pas** en 60 itérations
(résidu 2.16), `alpha*` converge en **33** (résidu 7.5e-08) avec 192 → **71 diagrammes**.
[§ ce qui reste](doc/06-ce-qui-reste.md).

**Les sommets en entiers 32 bits** (`filent8`) ont été écrits et mesurés : le prédicat de coupe
devient exact pour +13 %, les poids tombent sur la grille `2⁻⁶⁰` que l'homogénéité leur impose,
et la précision de la mesure est **identique** — les entiers achètent la cohérence, pas les
chiffres. Avec une réserve nette : sur une paire de germes à 10⁻⁸, la grille `2⁻³⁰` les fusionne
et le résultat devient exactement faux. L'arithmétique exacte suppose une entrée bien quantifiée,
donc l'agrégation des diracs trop proches en amont — [§ échelle](doc/04-echelle.md).
