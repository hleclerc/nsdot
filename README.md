# nsdot

Monorepo — 3 projets indépendants :

```
loom/     Interface agnostique Jax/Torch → noyaux C++ (tensor, Aggregate, drivers, compilation)
sdot/     Transport optimal semi-discret (Cell, PowerDiagram, OtPlan, OtPlan1d, distributions)
otrec/    Application de reconstruction CT (Reconstruction, Sinogram)
```

Chaque projet a son propre `pyproject.toml`. Dépendances : `otrec` → `sdot` → `loom`.

`unidim/` est un prototype à part (pas de `pyproject.toml`, pas installé) : reconstruction CT
par transport optimal 1D, voir [Prototype `unidim`](#prototype-unidim) plus bas.

## Quick start

Le travail -- tests, benchs, expériences -- est lancé par **[`errand`](errand/README.md)**, un
paquet à part qui ne sait rien de loom : ce dépôt lui dit ce qu'il a à savoir dans
`errandfile.py`, à la racine. `./run` ne garde que ce qui FABRIQUE la machine : installer,
diagnostiquer, bâtir les images, créer les envs micromamba.

```bash
# Première fois : l'env micromamba déclaré dans errandfile.py, puis les paquets en editable
./run env create
./run install                          # errand, loom, sdot, otrec

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
```

`errand --help` liste les cas sélectionnés avec leurs paramètres ; `errand --envs` les
environnements. La sélection par motif, les matrices de paramètres, l'arborescence de sortie,
la file d'attente, le détachement et l'écran sont documentés une fois pour toutes dans
[`errand/README.md`](errand/README.md).

## Commandes

| Commande | Description |
|---|---|
| `errand [motif]` | Le travail : tests C++ + Python, benchs, expériences |
| `errand --tui` | L'écran : chercher, lancer, suivre |
| `./run install` | `pip install -e` d'errand + des 3 projets, dans l'ordre |
| `./run toolchain` | Diagnostic (compilateur hôte, nvcc) |
| `./run build-sif` | Build des images Apptainer (.sif depuis .def) |
| `./run env` / `./run env create` | Lister / fabriquer les envs micromamba |

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

Les images `.sif` sont construites depuis les `.def` dans `containers/`. Voir
`containers/README.md` pour les détails.

```bash
./run build-sif --env cuda-jax               # Une image, en local
./run build-sif                               # Toutes les images (chaque env avec un layer Apptainer)
./run build-sif --env lmo-cuda-jax --fakeroot # Build distant (env dont le seq a un Remote)
```

## C++

Les headers C++ sont dans `loom/include/loom/support/` (runtime générique)
et `sdot/include/sdot/` (transport optimal). Les headers générés (JIT)
atterrissent dans `build/include/`.

Compilation : le compilateur C++ hôte pour le CPU (`c++`/`clang++`/`g++`, `SDOT_CXX` pour en
imposer un ; `-O3 -march=native`), `nvcc` autour de lui pour CUDA (celui du paquet pip
`nvidia-cuda-nvcc-cu13` s'il est là, sinon `/usr/local/cuda`, sinon PATH ; `SDOT_NVCC`). Chaque
device dit avec quoi il se compile (`Device.compiler`, voir `loom/src/loom/compilation/Compiler.py`) ;
le runtime C++ d'un device est sa queue (`loom/include/loom/support/kernels/CpuQueue.h`,
`CudaQueue.h`), qui porte le lancement des noyaux -- `run_parallel` ne connaît aucun device.

Sous nvcc, tout ce qu'un noyau atteint porte `HD` (`__host__ __device__`, vide ailleurs) : toute
nouvelle fonction atteignable par un noyau s'écrit avec ; `scripts/annotate_hd.py` (libclang) le
pose sur un fichier neuf. Les mathématiques passent par `sdot::sqrt` & co (`loom/support/math.h`),
les atomiques par `atomic_add.h`, les étiquettes globales par `LOOM_TAG` : les seuls `#if` sur la
cible.

Un wheel embarque un **catalogue** de noyaux précompilés (`sdot/_catalogue`, une bibliothèque par
variante : `cpu-x86-64-v3`, `cuda`, ...) : l'usage standard n'y compile rien. Le relevé
(`catalogue_record/`, versionné) vient de `scripts/build_catalogue.py record`, qui exerce
`python -m sdot.catalogue` ; la compilation par variante est `build_catalogue.py compile`, ce que
fait `.github/workflows/wheels.yml`. `SDOT_KERNELS=auto|catalogue|atelier`. Voir
`loom/src/loom/compilation/catalogue.py`.

Le transport semi-discret (`sdot.OtPlan`) est résolu **en un appel**, tout en C++
(`sdot/include/sdot/otplan/`) : le Newton amorti du banc `solvers_des_familles`, le laplacien
assemblé sans tri, Cholesky (Eigen) / AMG (AMGCL) / CG en unité de domaine, le pas par les limites
en 2D, la continuation en largeur pour les densités qui se concentrent ; le domaine est le support
que la densité déclare (`bounding_half_spaces`). Eigen et AMGCL sont téléchargés par loom lui-même au premier
noyau compilé (`loom/src/loom/compilation/externals.py` : archive épinglée + SHA-256, dans le cache
utilisateur, puis sur le chemin d'inclusion ; `SDOT_EXTERNALS=0` pour s'en passer -- le gradient
conjugué maison reste). Voir `notes/2026-09-22-otplan-cpp.md`.

La compilation passe par un graphe ninja (`loom/src/loom/compilation/build.py`, `build/build.ninja`
réécrit depuis `build/ninja/manifest.json`) : une unité n'est refaite que si l'un de SES en-têtes a
changé (depfiles du compilateur). `libloom_runtime` (la file de threads, une par processus) est liée
par chaque noyau ; `FfiCode( sources = [ ( "x.cpp", { "DEF": "v" } ) ] )` compile une source de
domaine une fois par configuration et la lie. `SDOT_FORCE_BUILD=1` force la cible demandée.
