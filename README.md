# nsdot

Monorepo — 3 projets indépendants :

```
loom/     Interface agnostique Jax/Torch → SYCL (tensor, Aggregate, drivers, compilation)
sdot/     Transport optimal semi-discret (Cell, OtPlan1d, distributions)
otrec/    Application de reconstruction CT (Reconstruction, Sinogram)
```

Chaque projet a son propre `pyproject.toml`. Dépendances : `otrec` → `sdot` → `loom`.

`unidim/` est un prototype à part (pas de `pyproject.toml`, pas installé) : reconstruction CT
par transport optimal 1D, voir [Prototype `unidim`](#prototype-unidim) plus bas.

## Quick start

`./run` lui-même ne dépend que d'un Python (stdlib seule, aucun paquet tiers) : pas besoin
d'activer quoi que ce soit pour lancer `./run env`, `./run env create` ou `./run install`.

```bash
# Première fois : fabrique l'env micromamba déclaré dans .envs.py (no-op s'il existe déjà)
./run env create

# Installer (une fois sur l'env par défaut)
./run install

# Lancer les tests
./run test                          # Tous les tests (C++ + Python)
./run test test_Cell                # Tout test_Cell.py
./run test test_Cell::batch         # Juste le test "batch" de test_Cell.py
./run test "test_Cell::grad_*"      # Glob sur le nom
./run test --fp=FP32                # Précision FP32

# Benchmark -- même mécanisme que test(), dans les mêmes fichiers
./run bench "test_OtPlan1d::*" --nb-diracs=5000

# Expérience -- même mécanisme encore, pour ce qui se REGARDE (html, ParaView, png)
./run experiment exp_lung                   # tout exp_lung.py
./run experiment "test_Cell::viz 3D"        # une entrée précise
./run experiment exp_lung --nb-diracs=5000  # override, `a,b` pour balayer
```

## Commandes

| Commande | Description |
|---|---|
| `./run test [pattern]` | Tests C++ (via acpp) + Python (tous les projets) |
| `./run bench [pattern]` | Benchmarks Python (même mécanisme que `test`) |
| `./run experiment [pattern]` | Expériences Python (même mécanisme que `test`), avec balayage de params |
| `./run install` | `pip install -e` des 3 projets dans l'ordre |
| `./run toolchain` | Diagnostic (acpp, LLVM, CUDA) |
| `./run build-sif` | Build des images Apptainer (.sif depuis .def) |
| `./run env` | Lister les environnements configurés |
| `./run env create` | Fabriquer les envs micromamba déclarés (no-op sur ceux qui existent déjà) |

### `test` / `bench` / `experiment` : sélection par pattern

`test()`/`bench()`/`experiment()` (`loom.testing`) sont trois variantes d'un
même mécanisme — une garde comme `if __name__ == "__main__":`, en plus élaboré :
plusieurs par fichier, identifiées par site d'appel (pas par nom, donc les
homonymes sont permis), et mixables dans un même fichier sous `{projet}/tests/`.

Ce qui les sépare n'est pas la mécanique mais l'ATTENTE, donc la commande qui
les lance : un `test` doit passer, un `bench` doit être rapide (et laisse des
chiffres datés dans `p.results`), une `experiment` doit être REGARDÉE — sa
sortie est un fichier qu'on ouvre.

Le `pattern` positionnel est une liste de specs séparées par `,`, chacune de
la forme `fichier[::nom]` :

```
./run test                          tout (tous les tests, tous les projets)
./run test test_Cell                 tout test_Cell.py (nom de fichier complet, préfixe inclus)
./run test test_Cell::batch          le test "batch" de test_Cell.py
./run test "test_Cell::grad_*"       glob sur le nom (fnmatch ; sans `*` = match exact)
./run test "test_Cell::a,test_OtPlan1d::b"   plusieurs specs
```

La recherche du fichier se fait sur tout le dépôt (récursif, aucune
distinction de projet ni de répertoire), filtrée aux fichiers qui référencent
`loom.testing` — sans ça, un fichier source qui porte le nom de son test
(`Cell.py` vs `test_Cell.py`, le cas courant) créerait une fausse ambiguïté.
La partie fichier matche le stem COMPLET (préfixe `test_`/`bench_` inclus —
`Cell` ne matche plus rien, il faut `test_Cell`). Sans `*` dans la partie
fichier, le nom doit désigner un fichier unique — sinon erreur (utiliser un
glob pour en sélectionner plusieurs). Pour ne prendre qu'un projet, c'est le
motif qui le dit -- `./run test "test_Cell::*,test_PowerDiagram::*"` -- il n'y
a pas d'option pour restreindre à un répertoire.

N'importe laquelle des trois peut déclarer des `Param` typés, listés via
`--help` et résumés avant chaque exécution :

```python
from loom.testing import bench, Param

if p := bench( "cost", nb_diracs = Param( 1000, help = "nb diracs" ) ):
    run_bench( p.nb_diracs )
```

```bash
./run bench cost --help          # liste les params déclarés
./run bench cost --nb-diracs=5000
```

`./run experiment` ajoute le BALAYAGE : `--nb-diracs=1000,2000` lance chaque
combinaison, chacune dans son propre répertoire (le hash des params diffère),
ce qui est exactement ce qu'on veut pour comparer des images. `test`/`bench`
ne le font pas — on n'asserte pas un produit cartésien.

### Répertoires de sortie

Chaque (entrée, jeu de params, env, date) a son propre répertoire feuille,
effacé et recréé à chaque lancement :

```
tmp/{test|bench}/{fichier}__{nom}/[hash-des-params/]{env}/{date}/
tmp/experiment/{fichier}__{nom}/[hash-des-params/]{env}/          <- sans la date
```

Une expérience s'arrête à `{env}` : ce qu'une date achète est un HISTORIQUE à
comparer, et une expérience n'a rien de comparable à produire — sa sortie est
un fichier qu'on ouvre. Ce qu'une date coûte, là, est la seule chose qui
compte : un chemin qui bouge sous l'onglet resté ouvert dessus. Chemin stable,
rechargement, fin.

(le hash n'apparaît que s'il y a des params). La feuille contient toujours
`result.yaml` (status, durée, RAM pic, params résolus, `p.results`), et
`output.txt` si le corps a produit du texte :

```python
if p := bench( "cost", nb_diracs = Param( 1000 ) ):
    p.results[ "cost" ] = run_bench( p.nb_diracs )   # -> result.yaml
    ( p.out_dir / "plot.png" ).write_bytes( fig )     # fichier ad hoc dans le même répertoire
```

Deux niveaux de résumé, recalculés à chaque run (relecture des `result.yaml`
voisins, pas un historique en mémoire) :
- `[hash]/{env}/summary.yaml` — une ligne par date, pour cet env (ok/pas ok,
  min/max des valeurs numériques de `p.results`, ou de la durée à défaut).
- `[hash]/summary.yaml` — une ligne par `{env}/{date}`, tous envs confondus.

Pour une expérience il n'y en a qu'un — `[hash]/summary.yaml`, une ligne par
env : sans niveau de date, il n'y a pas d'historique par env à résumer.

En exécution distante, seuls les `[hash]/` des cas effectivement sélectionnés
par le pattern sont rapatriés (par `rsync`, un par entrée) — pas tout
`tmp/test`/`tmp/bench` : `tmp/` n'est pas remis à zéro par le push du repo, un
hôte distant peut donc porter des runs plus anciens sans rapport avec
l'invocation en cours. Le contrôleur local prédit le chemin exact (mêmes
règles de hash/date) avant même que le run distant n'ait eu lieu — pas de
mécanisme de marqueurs (`OUTPUT:`) déclarés à l'exécution.

C'est vrai des trois : une expérience écrit dans son `p.out_dir` comme un
test, donc son `[hash]/` se prédit et se rapatrie pareil.

Options communes à toutes les commandes :

| Flag | Effet |
|---|---|
| `--env <nom>` | Sélectionne un environnement de `.envs.py` |
| `--driver jax\|torch` | Sélection auto du 1er env avec ce driver |
| `--device cpu\|cuda` | Définit `SDOT_DEVICE` + `JAX_PLATFORMS` |
| `--fp FP32\|FP64` | Définit `SDOT_FTYPE` |

## Environnements

Le runner lit `.envs.py` (copier depuis `.envs.py.example`). Ce fichier déclare, en
Python, **où** et **comment** les commandes s'exécutent — pas de fichier de config
séparé pour les machines distantes : une machine distante est juste un env de plus.

### Résolution d'environnement

```
./run test --env cuda-jax     → l'env "cuda-jax"                 (explicite)
./run test --driver torch     → 1er env avec un layer Driver("torch")  (par driver)
./run test                    → env nommé "default"              (fallback)
```

Si aucun `--env` ni `--driver` n'est donné, l'environnement nommé `default` est utilisé.
S'il n'existe pas, le premier env de la liste est pris.

### Layers

Un env est une séquence de *layers* (`loom/src/loom/cli/layers.py`), composés
outside-in, qui décrivent comment atteindre le sous-processus final :

```python
from loom.cli.layers import env, Driver, Micromamba, Apptainer, Remote

JAX = [Driver("jax")]
MM = [Micromamba("mon_env_mm")]

env("default", MM + JAX)
```

`Driver(name, pip=...)` : `pip`, si présent, est la spec exacte que `./run install`
installe pour ce driver (ex. `Driver("jax", pip="jax[cuda13]")`) au lieu de compter sur
le `jax`/`torch` tiré en transitif par un projet (ex. `otrec` → `optax` → `jax`, en CPU
par défaut) — l'extra CUDA/ROCm dépend de l'env/du hardware, donc pas exprimable dans
un seul `pyproject.toml` partagé. Installé en premier, avant les `-e` des 3 projets.

`Micromamba` wrappe avec `micromamba -n vfs run ...` (no-op si l'env est déjà activé
localement). `python`/`channels`/`packages` ne servent qu'à `./run env create` (fabrique
l'env s'il n'existe pas encore ; sans effet sur un env déjà là) :

```python
MM = [Micromamba("vfs", python="3.13")]
```

`Apptainer` wrappe avec `apptainer exec --bind ... image.sif ...`, et
utilise toujours le `python` du conteneur (les packages nsdot y sont déjà en editable
install — aucun `PYTHONPATH` ni mount n'est nécessaire pour exécuter les tests).
`Remote` (ssh) doit être le premier layer de la séquence quand il est présent :

```python
LMO = [Remote(host="lmo", remote_dir="/home/leclerc/nsdot",
              python="/data/venvs/sdot/bin/python",
              apptainer_scratch="/data/singularity_tmp")]
CUDA_JAX_SIF = [Apptainer(image="containers/cuda-jax.sif", flags=["--nvccli"])] + JAX

env("lmo-cuda-jax", LMO + CUDA_JAX_SIF)
```

Le flux distant (géré par `Remote`) : `rsync` du repo → `ssh` → exécution →
`rsync` ciblé de retour des chemins passés à `pull=[...]` par l'appelant
(déterministe : les `tmp/{kind}/…/[hash]/` des entrées sélectionnées). Ce flux
s'annonce en gris avant de s'exécuter :

```
  → machine=lmo (/home/leclerc/nsdot)  driver=jax
  rsync push → lmo:/home/leclerc/nsdot
  ...
  rsync pull ← lmo:/home/leclerc/nsdot [tmp/bench/...]
```

(en local, seule la ligne `machine=... driver=...` s'affiche — pas de rsync).

Mutualise les briques communes avec du Python normal (variables, fonctions, `+` de
listes) plutôt qu'avec un mécanisme dédié — voir `.envs.py.example`.

### Lister les environnements

```bash
$ ./run env

Environnements (.envs.py):
  default             driver=jax     micromamba=vfs ← default
  cuda-jax            driver=jax     apptainer=containers/cuda-jax.sif
  lmo-cuda-jax        driver=jax     remote=lmo apptainer=containers/cuda-jax.sif
                      scratch: /data/singularity_tmp

  Select with: --env <name>  (or --driver <jax, torch>)
```

### Exemples complets

```bash
# Local, env par défaut
./run test

# Local, env micromamba torch
./run test --env torch

# Local, conteneur apptainer JAX
./run test --env cuda-jax --device cuda

# Distant (rsync → ssh → run), sur lmo dans le conteneur cuda-jax
./run test --env lmo-cuda-jax --device cuda
```

## Expériences

Même déclaration, même découverte et même sélection par pattern que
`test`/`bench` (voir ci-dessus) : une expérience est une entrée parmi les
autres, plusieurs par fichier, mixables avec des tests dans le même fichier —
`sdot/tests/test_Cell.py` en a cinq, une par régime d'affichage, à côté de ses
tests de géométrie.

```python
from loom.testing import experiment, Param

if p := experiment( "viz 3D" ):
    c = Cell.make_hypercube( 3, [ 0, 0, 0 ], numpy.eye( 3 ).tolist() )
    c.cut( [ 1, 1, 1 ], 2.5 )
    v = Visualizer(); c.add_to_viz( v )
    v.write_html( p.out_dir / "cell_3d.html" )   # la page autonome
    v.write_vtk ( p.out_dir / "cell_3d.vtu" )    # et ParaView
```

```bash
./run experiment test_Cell                 # les cinq
./run experiment "test_Cell::viz 3D"       # une seule
./run experiment "test_Cell::viz cut*" --nb-cuts=4,8   # balayage : une sortie par valeur
./run experiment --help                    # toutes celles du dépôt, avec leurs params
```

Chaque entrée affiche, en fin de run, son répertoire et ce qu'elle y a écrit —
il n'y a donc pas de schéma de nommage à reconstituer pour retrouver le
fichier à ouvrir.

Les fichiers écrits contre l'ancien harnais (`from loom.cli import experiment`,
un fichier = une expérience) marchent tels quels : `loom.cli` ré-exporte
`experiment`/`Param`, et `./run experiment exp_lung` reste le stem du fichier
comme pattern.

## Prototype `unidim`

Prototype de reconstruction CT (pas un des 3 projets pip-installables ci-dessus —
vit à la racine, sans `pyproject.toml`, mais utilise le même mécanisme `bench` de
`loom.testing`) : distance de Wasserstein 1D en forme fermée (pas de plan de
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
./run bench reconstruction_jax --nb-diracs 5000
./run bench --env lmo-cuda-jax reconstruction_jax
./run bench --env lmo-cuda-torch reconstruction_cuda
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

Compilation : AdaptiveCpp (`acpp`), téléchargé automatiquement au premier `driver.call`.
