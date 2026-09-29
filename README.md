# nsdot

**Un plan de travail**, plus un monorepo : les quatre paquets ont chacun leur dépôt, et ce
dépôt-ci ne les suit plus — il les *accueille*, à côté de la recherche qui les consomme.

```bash
scripts/bootstrap.sh          # clone les quatre ici, aux noms attendus
errand --setup --env nsdot    # l'environnement, avec les quatre en éditable
```

```
loom/     git@github.com:hleclerc/loom.git       Jax/Torch → noyaux C++ (tensor, Aggregate, ffi_call)
sdot/     git@github.com:hleclerc/sdot-ffi.git   Transport optimal semi-discret (Cell, PowerDiagram, OtPlan)
otrec/    git@github.com:hleclerc/otrec.git      Reconstruction CT (Reconstruction, Sinogram)
errand/   git@github.com:hleclerc/errand.git     Le lanceur de travaux
```

Dépendances : `otrec` → `sdot` → `loom`. Chacun a son `pyproject.toml`.

Le couplage entre eux est **physique, pas git** : `errandfile.py` déclare `src = [ "loom/src",
"sdot/src", "otrec/src" ]` et les conteneurs montent `loom` sur `/opt/sdot/loom`. D'où les noms de
répertoire imposés ci-dessus — `sdot/` vient du dépôt `sdot-ffi`, le nom local étant celui du
paquet python.

**Ce qui reste suivi ici** : la recherche (`2d_des_familles/`, `solvers_des_familles/`,
`gpu_des_familles/`, `unidim/`), les notes et la doc, et la glu qui n'a de sens qu'à travers les
quatre (`errandfile.py`, `scripts/`, `containers/`).

**Ce qui n'y est plus** : la CI des wheels et `catalogue_record/` sont partis dans `sdot-ffi`, avec
le wheel qui porte les binaires du catalogue.

`unidim/` est un prototype à part (pas de `pyproject.toml`, pas installé) : reconstruction CT
par transport optimal 1D, voir [Prototype `unidim`](#prototype-unidim) plus bas.

## Quick start

Tout passe par **[`errand`](errand/README.md)**, un paquet à part qui ne sait rien de loom : ce
dépôt lui dit ce qu'il a à savoir dans `errandfile.py`, à la racine. Il n'y a plus de `./run` --
lancer le travail, fabriquer les environnements, bâtir les images et entrer dans une machine sont
la même commande.

```bash
# Première fois : l'environnement déclaré dans errandfile.py, avec ses paquets et les editables
errand --setup --env nsdot

# Lancer le travail
errand                              # tout ce qui doit passer (C++ + Python)
errand test_Cell                    # tout test_Cell.py
errand test_Cell::batch             # le cas "batch" de test_Cell.py
errand "test_Cell::grad_*"          # glob sur le nom
errand -k bench "test_OtPlan1d::*" --nb-diracs 5000
errand -k experiment exp_lung --nb-diracs 5000,10000   # une sortie par valeur

# Ailleurs, et à plusieurs
errand --env lmo-cuda-jax           # rsync -> ssh -> run -> rsync retour
errand -j 8                         # huit à la fois
errand --batch                      # détaché : rend la main, se suit après coup

# L'écran : chercher un cas parmi trois cents, le lancer, suivre ce qu'il écrit
errand --tui

# Entrer dans un environnement pour autre chose que du travail déclaré
errand --env nsdot -- loom-toolchain          # ce que la compilation trouve
errand --env lmo-cuda-jax -- nvidia-smi       # ce que la carte de là-bas répond
```

`errand --help` liste les cas sélectionnés avec leurs paramètres, `errand --envs` les
environnements et leur état. La sélection par motif, les matrices de paramètres, l'arborescence de
sortie, la file d'attente, le détachement et l'écran sont documentés une fois pour toutes dans
[`errand/README.md`](errand/README.md).

## Commandes

| Commande | Description |
|---|---|
| `errand [motif]` | Le travail : tests C++ + Python, benchs, expériences |
| `errand --tui` | L'écran : chercher, lancer, suivre |
| `errand --envs` | Les environnements, et lequel est à jour |
| `errand --setup --env X` | Fabriquer ou mettre à jour un env (micromamba, image, pip) |
| `errand --setup force --env X` | Le refaire, quoi qu'il en dise |
| `errand --env X -- <cmd>` | Cette commande-là, dans cet environnement-là |

## Ce que ce dépôt déclare à errand

`errandfile.py`, à la racine, en Python ordinaire :

* **où est le code** -- `src = [ "loom/src", "sdot/src", "otrec/src" ]` : les trois projets DE
  CE CHECKOUT passent devant ce qu'un `pip install -e` fait depuis un autre checkout aurait
  installé. C'est l'ancien `PYTHONPATH` de `./run`, dit une fois.
* **où ça tourne** -- un `env( nom, [ couches ], **tags )` par environnement. Une machine
  distante est un env de plus, dont la première couche est `Ssh`.
* **les tests C++ de loom** -- un *provider* : une entrée par `loom/tests/cpp/test_*.cpp`, un
  paramètre `--device cpu,cuda`, compilée par `loom.compilation.make_executable`. Tout ce qui
  entoure le run -- répertoire de sortie, `result.yaml`, résumés, matrices, file d'attente,
  rapatriement -- est le même que pour une entrée Python.

**Le driver n'est plus une couche, c'est un tag.** Il ne change pas la façon d'atteindre la
machine, il dit ce qu'on y trouve : il sélectionne (`--driver torch`, `-t 'driver=jax'`), il
apparaît dans `result.yaml` et dans le nom du répertoire de sortie, et un fichier qui a besoin
de savoir demande `has_tag( "driver=torch" )` plutôt que de lire une variable d'environnement.

```python
env( "nsdot", [ Micromamba( "nsdot", python = "3.13" ) ], driver = "jax", cuda = True )
env( "lmo-cuda-jax", [ Ssh( host = "lmo", root = "/home/leclerc/nsdot" ),
                       Micromamba( "vfs" ) ], driver = "jax", cuda = True )
```

```bash
$ errand --envs

Environments
  nsdot             cuda  driver=jax     micromamba:nsdot  <- default
  lmo-cuda-jax      cuda  driver=jax     ssh:lmo -> micromamba:vfs
  lmo-cuda-torch    cuda  driver=torch   ssh:lmo -> micromamba:torch

  choose one with --env <name>, or by tag: --cuda --driver
```

## Déclarer un cas

```python
from errand import test, bench, experiment, Param
from loom.testing import check_grad          # ce qui reste propre à loom

if test( "a position is not something one solves for" ):
    assert ...

if p := bench( "cost", nb_diracs = Param( 1000, help = "nb diracs" ) ):
    p.results[ "cost" ] = run_bench( p.nb_diracs )      # -> result.yaml

if p := experiment( "viz 3D" ):
    v.write_html( p.out_dir / "cell_3d.html" )          # une sortie à REGARDER
```

Les trois partagent tout -- enregistrement, paramètres, `p.out_dir`, `result.yaml` -- et ne
diffèrent que par ce qu'on en ATTEND : un test doit passer, un bench doit être rapide (ses
chiffres sont gardés, il prend la machine pour lui seul), une expérience doit être regardée
(`latest/` est un chemin stable, l'onglet resté ouvert dessus se recharge).


## Expériences

Une expérience est une entrée parmi les autres : plusieurs par fichier, mêlées aux tests du
même fichier -- `sdot/tests/test_Cell.py` en a cinq, une par régime d'affichage, à côté de ses
tests de géométrie.

```python
from errand import Param, experiment

if p := experiment( "viz 3D" ):
    c = Cell.make_hypercube( 3, [ 0, 0, 0 ], numpy.eye( 3 ).tolist() )
    c.cut( [ 1, 1, 1 ], 2.5 )
    v = Visualizer(); c.add_to_viz( v )
    v.write_html( p.out_dir / "cell_3d.html" )   # la page autonome
    v.write_vtk ( p.out_dir / "cell_3d.vtu" )    # et ParaView
```

```bash
errand -k experiment test_Cell                # les cinq
errand -k experiment "test_Cell::viz 3D"      # une seule
errand -k experiment "test_Cell::viz cut*" --nb-cuts 4,8   # une sortie par valeur
errand -k experiment --help                   # toutes celles du dépôt, avec leurs params
```

Chaque entrée affiche, en fin de run, son répertoire et ce qu'elle y a écrit, et `latest/`
pointe sur le dernier -- l'onglet resté ouvert sur `runs/test_Cell/viz_3d/latest/cell_3d.html`
se recharge.


## Prototype `unidim`

Prototype de reconstruction CT (pas un des 3 projets pip-installables ci-dessus —
vit à la racine, sans `pyproject.toml`, mais déclare ses `bench` comme tout le
monde) : distance de Wasserstein 1D en forme fermée (pas de plan de
transport explicite) entre un nuage de points 2D projeté et un sinogramme, optimisée
par L-BFGS.

Deux implémentations parallèles, mêmes maths, backends différents :

| Fichier | Backend | Gradient |
|---|---|---|
| `reconstruction_jax.py` | JAX/optax (LBFGS + line search zoom) | autodiff |
| `reconstruction_cuda.py` | noyau CUDA fusionné (projection + tri CUB + cost/grad en un seul kernel), compilé via `torch.utils.cpp_extension.load_inline` | écrit à la main |

Les deux découpent les angles en *chunks* dimensionnés sur la mémoire GPU
RÉELLEMENT libre (`gpu_mem.py`), pour ne jamais matérialiser un tenseur
`[nb_angles, n]` complet (`nb_diracs` visé jusqu'à ~1e11) :

```bash
errand -k bench reconstruction_jax --nb-diracs 5000
errand -k bench --env lmo-cuda-jax reconstruction_jax
errand -k bench --env lmo-cuda-torch reconstruction_cuda
```

Au premier appel (par taille de problème), chaque backend affiche un message
`[warmup]` — compilation JIT XLA côté JAX, compilation nvcc de l'extension côté
CUDA — avant la boucle réellement chronométrée (`p.results["ms_per_grad_by_n"]`).

## Conteneurs Apptainer

Les images `.sif` sont déclarées comme des couches dans `errandfile.py`, avec de quoi les bâtir
(`recipe`, `fakeroot`, `scratch`) et de quoi y entrer (`flags`, `mounts`). Voir
`containers/README.md`.

```bash
errand --setup --env cuda-jax                 # une image, en local
errand --setup force --env lmo-cuda-jax-sif   # la refaire, là-bas (rsync -> ssh -> build)
errand --setup --dry-run --env cuda-jax       # ce que ça lancerait, sans le lancer
```

Elles se bâtissent aussi toutes seules : `errand --env cuda-jax <...>` vérifie l'image contre ce
que la déclaration dit avant de lancer quoi que ce soit. `--no-setup` dit de ne pas le faire.

## C++

Les headers C++ sont dans `loom/include/loom/support/` (runtime générique)
et `sdot/include/sdot/` (transport optimal). Les headers générés (JIT)
atterrissent dans `build/include/`.

Compilation : le compilateur C++ hôte pour le CPU (`c++`/`clang++`/`g++`, `LOOM_CXX` pour en
imposer un ; `-O3 -march=native`), `nvcc` autour de lui pour CUDA (celui du paquet pip
`nvidia-cuda-nvcc-cu13` s'il est là, sinon `/usr/local/cuda`, sinon PATH ; `LOOM_NVCC`). Chaque
device dit avec quoi il se compile (`Device.compiler`, voir `loom/src/loom/compilation/Compiler.py`) ;
le runtime C++ d'un device est sa queue (`loom/include/loom/support/kernels/CpuQueue.h`,
`CudaQueue.h`), qui porte le lancement des noyaux -- `run_parallel` ne connaît aucun device.

Sous nvcc, tout ce qu'un noyau atteint porte `HD` (`__host__ __device__`, vide ailleurs) : toute
nouvelle fonction atteignable par un noyau s'écrit avec ; `scripts/annotate_hd.py` (libclang) le
pose sur un fichier neuf. Les mathématiques passent par `sdot::sqrt` & co (`loom/support/math.h`),
les atomiques par `atomic_add.h`, les étiquettes globales par `LOOM_TAG` : les seuls `#if` sur la
cible.

Un wheel embarque un **catalogue** de noyaux précompilés (`sdot/_catalogue`, une bibliothèque par
variante : `cpu-x86-64-v3`, `cuda`, ...) : l'usage standard n'y compile rien. Tout passe par la
commande `loom-kernels`, qui est l'artefact par lequel **n'importe quel système de construction**
(cmake, bazel, xmake, un Makefile) fait produire les noyaux à l'avance plutôt qu'à l'exécution :

```bash
loom-kernels record  --out catalogue_record -- python -m sdot.catalogue    # le relevé, versionné
loom-kernels compile --record catalogue_record --out sdot/catalogue --import sdot --variant x86-64-v3
```

`record` ne sait rien de ce qu'on lance (la commande vient après `--`) et `compile` n'exécute rien
du projet, `--import` nommant seulement les modules qui enregistrent leur racine C++. C'est ce que
fait `.github/workflows/wheels.yml`. `LOOM_KERNELS=auto|catalogue|atelier`. Voir
`loom/src/loom/compilation/{catalogue,cli}.py`.

Le transport semi-discret (`sdot.OtPlan`) est résolu **en un appel**, tout en C++
(`sdot/include/sdot/otplan/`) : le Newton amorti du banc `solvers_des_familles`, le laplacien
assemblé sans tri, Cholesky (Eigen) / AMG (AMGCL) / CG en unité de domaine, le pas par les limites
en 2D, la continuation en largeur pour les densités qui se concentrent ; le domaine est le support
que la densité déclare (`bounding_half_spaces`). Eigen et AMGCL sont téléchargés par loom lui-même au premier
noyau compilé (`loom/src/loom/compilation/externals.py` : archive épinglée + SHA-256, dans le cache
utilisateur, puis sur le chemin d'inclusion ; `LOOM_EXTERNALS=0` pour s'en passer -- le gradient
conjugué maison reste). Voir `notes/2026-09-22-otplan-cpp.md`.

La compilation passe par **deux graphes**, le critère étant *avoir des dépendants ou non*
(`loom/src/loom/compilation/build.py`) :

* le graphe **commun** (`build/build.ninja`, réécrit depuis `build/ninja/manifest.json`) porte
  `libloom_runtime` (la file de threads, une par signature de compilateur) et les **unités de
  domaine** -- `FfiCode( sources = [ ( "x.cpp", { "DEF": "v" } ) ] )` compile une source une fois
  par configuration et la lie dans tous les noyaux qui la nomment ;
* le graphe **propre** d'un noyau (`build/noyaux/<cible>/`) porte sa source engendrée, son objet et
  sa bibliothèque : zéro dépendant, donc effacer son répertoire suffit à l'oublier
  (`loom-kernels prune [--older-than JOURS]`). Le verrou est par graphe : deux processus qui
  compilent deux noyaux différents ne s'attendent pas.

Une unité n'est refaite que si l'un de SES en-têtes a changé (depfiles du compilateur) -- c'est la
seule chose difficile, et elle coûte 88 ms de graphe par noyau contre ~8 s de compilation.
`LOOM_FORCE_BUILD=1` force la cible demandée.

**Réglages** : le préfixe est `LOOM_` (`LOOM_BUILD_DIR`, `LOOM_CACHE_DIR`, `LOOM_KERNELS`,
`LOOM_CXX`, `LOOM_FTYPE`, ...), lus en un seul endroit, `loom/src/loom/util/env.py`. L'ancien
préfixe `SDOT_` est encore accepté, avec un avertissement émis une fois par variable. Ce qui reste
légitimement en `SDOT_` est ce que sdot lit pour lui-même : `SDOT_KTYPE` (le type du noyau de ses
cellules) et `SDOT_CATALOGUE_DIR`.

**Synchroniser le dépôt** : `.rsync-exclude` dit ce qui ne doit pas traverser, et le dit une fois
pour tous les scripts qui en ont besoin. Tout ce que loom fabrique dans le dépôt tient sous un
**seul** répertoire (`build/`, en-têtes engendrés compris) et rien n'en sort : le graphe porte des
chemins absolus et une signature de compilateur qui inclut le modèle de CPU sous `-march=native`,
donc il ne vaut que sur la machine qui l'a écrit. Ce qui DOIT traverser, et n'y est donc pas :
`catalogue_record/`, indépendant de la machine, qui est précisément ce qui permet de compiler les
noyaux ailleurs.
