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
de `solvers_des_familles` (`src/bench/Args.h`), plus `--variante fil | filreg | filregc | filmix{4,6,8,12,16} | filbrk{6,8,10,12,16} | filbrk8nu | filrot{6,8} | filnrm8 | filord8 | filsuc8 | filmsk8 | filmsk8g | filmsk8c{6,8} | filnrm8c{6,8} | filuni8 | filuni8np | filshm8 | filnrm8tri | filnrm8tril | filph8 | filph8g | filph8b | filph8a | filph8c | filph8o | filph8m | filph8m4 | voies | voies16 |
voies32 | paquet{8,32}x{1,2,4}[S] | toutes` et `--reps-gpu`. Sur la machine, `--threads 8` est à donner (`hardware_concurrency` rend 1
dans le bac à sable) et **tout chronométrage passe par `job -b`**.

---

## Le sommaire

| | |
|---|---|
| [1. Où est quoi](doc/01-organisation.md) | les fichiers, ce que chacun tient |
| [2. Les mappages](doc/02-mappages.md) | une cellule par thread, par groupe de voies, par warp ; les phases |
| [3. Les chiffres](doc/03-chiffres.md) | le tableau des variantes, `float` et `double`, 2D et 3D |
| [4. L'échelle et la précision](doc/04-echelle.md) | jusqu'à 3·10⁷, ce que 10⁹ demanderait, le repère centré, la virgule fixe, **quelle carte** |
| [5. Ce que le profil dit](doc/05-profils.md) | `ncu` : où va le temps, le bilan d'occupation, les idées mesurées et perdues |
| [6. Ce qui reste](doc/06-ce-qui-reste.md) | l'état des lieux et les chantiers ouverts |
| [7. Comment les chiffres sont pris](doc/07-methode.md) | `job -b`, la chauffe, le témoin, le plancher de bruit |

**Les trois résultats à retenir.** En 2D `float`, `filnrm8` fait **7.6 ns/germe** (×19 sur le CPU à
8 fils) et `filmsk8f` **8.2** avec 25 % de registres en moins, une occupation atteinte de 88 % et
une erreur divisée par 4 à 5 ; en 2D `double`, `filmsk8g` fait **96.7** ; en 3D `voies` fait
**229** (×8). Le temps par germe est **plat de 10⁶ à 3·10⁷**. Ce qui casse avant le noyau à 10⁹,
c'est la mémoire, la construction de l'arbre (sur CPU) et la précision du `float` —
[§ échelle](doc/04-echelle.md).
