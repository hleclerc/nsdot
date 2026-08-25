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

```bash
# Installer (une fois sur l'env par défaut)
./run install

# Lancer les tests
./run test                          # Tous les tests (C++ + Python)
./run test test_Cell                # Tout test_Cell.py
./run test test_Cell::batch         # Juste le test "batch" de test_Cell.py
./run test "test_Cell::grad_*"      # Glob sur le nom
./run test --project=sdot           # Un seul projet
./run test --fp=FP32                # Précision FP32

# Benchmark -- même mécanisme que test(), dans les mêmes fichiers
./run bench "test_OtPlan1d::*" --nb-diracs=5000

# Expérience -- un fichier = une expérience, params typés
./run experiment exp_lung               # Nom de fichier complet
./run experiment exp_lung --nb-diracs=5000  # Override de paramètre
```

## Commandes

| Commande | Description |
|---|---|
| `./run test [pattern]` | Tests C++ (via acpp) + Python (tous les projets) |
| `./run bench [pattern]` | Benchmarks Python (même mécanisme que `test`) |
| `./run experiment <fichier>` | Expérience, nom de fichier complet (ex. `exp_lung`) |
| `./run install` | `pip install -e` des 3 projets dans l'ordre |
| `./run toolchain` | Diagnostic (acpp, LLVM, CUDA) |
| `./run build-sif` | Build des images Apptainer (.sif depuis .def) |
| `./run env` | Lister les environnements configurés |

### `test` / `bench` : sélection par pattern

`test()`/`bench()` (`loom.testing`) sont deux variantes d'un même mécanisme —
une garde comme `if __name__ == "__main__":`, en plus élaboré : plusieurs par
fichier, identifiées par site d'appel (pas par nom, donc les homonymes sont
permis), et mixables dans un même fichier sous `{projet}/tests/`.

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
glob pour en sélectionner plusieurs). `--project` restreint au premier
segment du chemin (ex. `loom`, `sdot`, `otrec`), mais n'importe quel
répertoire de premier niveau marche.

`bench` (et `test`, à l'occasion) peut déclarer des `Param` typés, listés via
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

### Répertoires de sortie

Chaque (test/bench, jeu de params, env, date) a son propre répertoire feuille,
effacé et recréé à chaque lancement :

```
tmp/{test|bench}/{fichier}__{nom}/[hash-des-params/]{env}/{date}/
```

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

En exécution distante, seuls les `[hash]/` des cas effectivement sélectionnés
par le pattern sont rapatriés (par `rsync`, un par entrée) — pas tout
`tmp/test`/`tmp/bench` : `tmp/` n'est pas remis à zéro par le push du repo, un
hôte distant peut donc porter des runs plus anciens sans rapport avec
l'invocation en cours. Le contrôleur local prédit le chemin exact (mêmes
règles de hash/date) avant même que le run distant n'ait eu lieu — pas de
mécanisme de marqueurs (`OUTPUT:`) déclarés à l'exécution.

`./run experiment` n'a pas ce chemin déterministe (fichiers choisis librement
par l'auteur) : rien n'est rapatrié automatiquement pour l'instant.

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
localement). `Apptainer` wrappe avec `apptainer exec --bind ... image.sif ...`, et
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
(déterministe — `tmp/test`/`tmp/bench` pour `./run test`/`bench`, `tmp` en
entier pour `./run experiment`, qui écrit sous des chemins choisis par
l'auteur plutôt qu'un schéma fixe). Ce flux s'annonce en gris avant de
s'exécuter :

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

Même découverte que `test`/`bench` (tout le dépôt, filtrée au contenu du
fichier — ici `from loom.cli import`, plutôt que `loom.testing` — aucun
répertoire réservé) : nom de fichier complet obligatoire, préfixe inclus
(`exp_lung`, pas `lung`) — même convention que `test_Cell.py`/`bench_cost.py`.

Une expérience utilise le harnais déclaratif :

```python
from loom.cli import experiment, Param

if p := experiment("lung BFGS", nb_diracs=Param(10_000, help="Nombre de Diracs")):
    # p.nb_diracs vaut 10000 (ou la valeur passée en CLI)
    ...
```

```bash
./run experiment exp_lung
./run experiment exp_lung --nb-diracs=5000
./run experiment exp_lung --help          # liste les params déclarés
```

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
