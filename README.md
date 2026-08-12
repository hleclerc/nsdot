# nsdot

Monorepo — 3 projets indépendants :

```
loom/     Interface agnostique Jax/Torch → SYCL (tensor, Aggregate, drivers, compilation)
sdot/     Transport optimal semi-discret (Cell, OtPlan1d, distributions)
otrec/    Application de reconstruction CT (Reconstruction, Sinogram)
```

Chaque projet a son propre `pyproject.toml`. Dépendances : `otrec` → `sdot` → `loom`.

## Quick start

```bash
# Installer (une fois)
make install

# Lancer les tests
./run test                          # Tous les tests (C++ + Python)
./run test --name=Cell              # Filtre par nom
./run test --project=sdot           # Un seul projet
./run test --fp=FP32                # Précision FP32

# Expérience
./run experiment lung               # Nom du fichier exp_*.py
./run experiment lung --nb-diracs=5000  # Override de paramètre

# Benchmark
./run bench speed                   # Nom du fichier bench_*.py
```

## Commandes

| Commande | Description |
|---|---|
| `./run test` | Tests C++ (via acpp) + Python (tous les projets) |
| `./run experiment <nom>` | Expérience auto-découverte depuis `exp_*.py` |
| `./run bench <nom>` | Benchmark auto-découvert depuis `bench_*.py` |
| `./run install` | `pip install -e` des 3 projets dans l'ordre |
| `./run toolchain` | Diagnostic (acpp, LLVM, CUDA) |
| `./run build-sif` | Build des images Apptainer (.sif depuis .def) |
| `./run env list` | Lister les environnements et hôtes configurés |

Options communes à toutes les commandes :

| Flag | Effet |
|---|---|
| `--env <nom>` | Sélectionne un environnement de `.hosts.toml` |
| `--driver jax\|torch` | Sélection auto du 1er env avec ce driver |
| `--host <nom>` | Exécute sur la machine distante |
| `--device cpu\|cuda` | Définit `SDOT_DEVICE` + `JAX_PLATFORMS` |
| `--fp FP32\|FP64` | Définit `SDOT_FTYPE` |

## Environnements

Le runner lit `.hosts.toml` (copier depuis `.hosts.toml.example`). Ce fichier déclare
**où** et **comment** les commandes s'exécutent.

### Résolution d'environnement

```
./run test --env cuda-jax     → utilise [envs.cuda-jax]          (explicite)
./run test --driver torch     → 1er env avec driver="torch"      (par driver)
./run test                    → env nommé "default"              (fallback)
```

Si aucun `--env` ni `--driver` n'est donné, l'environnement nommé `default` est utilisé.
S'il n'existe pas, le premier env de la liste est pris.

### Types d'environnement

**micromamba** (local, venv conda) :

```toml
[envs.default]
type = "micromamba"
name = "vfs"
driver = "jax"
```

Le runner wrappe la commande avec `micromamba -n vfs run ...`.
Si l'env est déjà activé (`CONDA_DEFAULT_ENV=vfs`), aucun wrapping n'est fait.

**apptainer** (conteneur .sif) :

```toml
[envs.cuda-jax]
type = "apptainer"
image = "containers/cuda-jax.sif"
driver = "jax"
flags = ["--nvccli"]

# Optionnel : mounts pour développement live (shadow les editable installs du conteneur)
mounts = { "loom" = "/opt/sdot/loom", "sdot" = "/opt/sdot/sdot", "otrec" = "/opt/sdot/otrec" }
```

Le runner wrappe avec `apptainer exec --nvccli --bind ... containers/cuda-jax.sif ...`.
Les packages nsdot sont déjà en editable install dans le conteneur — aucun `PYTHONPATH`
ni mount n'est nécessaire pour exécuter les tests.

### Lister les environnements

```bash
$ ./run env list

Environnements (.hosts.toml → [envs]):
  default              driver=jax     type=micromamba   name=vfs ← default
  cuda-jax             driver=jax     type=apptainer    image=containers/cuda-jax.sif
                        mount: loom → /opt/sdot/loom
                        mount: sdot → /opt/sdot/sdot
                        mount: otrec → /opt/sdot/otrec

  Select with: --env <name>  (or --driver <jax, torch>)

Hosts (.hosts.toml → [hosts]):
  lmo                  hostname=lmo  scratch=/data/singularity_tmp

  Select with: --host <name>  |  builds sync via rsync → ssh
```

### Exemples complets

```bash
# Local, env par défaut
./run test

# Local, env micromamba torch
./run test --env torch

# Local, conteneur apptainer JAX
./run test --env cuda-jax --device cuda

# Distant (rsync → ssh → run), env par défaut
./run test --host lmo

# Distant, env explicite
./run test --env cuda-jax --host lmo --device cuda
```

## Hôtes distants

Chaque hôte déclare la machine, le python distant, et optionnellement le répertoire
scratch pour les builds Apptainer :

```toml
[hosts.lmo]
hostname = "lmo"
remote_dir = "/home/leclerc/nsdot"
python = "/data/venvs/sdot/bin/python"
apptainer_scratch = "/data/singularity_tmp"  # ~30 GB free pour les builds .sif
```

Le flux distant : `rsync` du repo → `ssh` → exécution → parse des `OUTPUT:` → `rsync` ciblé.

Pour que les fichiers soient rapatriés automatiquement, l'expérience émet des marqueurs :

```python
print("OUTPUT:tmp/resultat.html")
print("OUTPUT:tmp/figure.png")
```

## Expériences et benchmarks

Les fichiers sont auto-découverts par convention de nommage :

| Type | Pattern | Emplacement |
|---|---|---|
| Tests | `test_*.py` | `{projet}/tests/` |
| Expériences | `exp_*.py` | `{projet}/src/{projet}/experiments/` |
| Benchmarks | `bench_*.py` | `{projet}/src/{projet}/benchmarks/` |

Une expérience utilise le harnais déclaratif :

```python
from loom.cli import experiment, Param

if p := experiment("lung BFGS", nb_diracs=Param(10_000, help="Nombre de Diracs")):
    # p.nb_diracs vaut 10000 (ou la valeur passée en CLI)
    ...
```

## Conteneurs Apptainer

Les images `.sif` sont construites depuis les `.def` dans `containers/`. Voir
`containers/README.md` pour les détails.

```bash
./run build-sif --env cuda-jax               # Une image
./run build-sif                               # Toutes les images
./run build-sif --host lmo --fakeroot         # Build distant
```

## C++

Les headers C++ sont dans `loom/include/loom/support/` (runtime générique)
et `sdot/include/sdot/` (transport optimal). Les headers générés (JIT)
atterrissent dans `build/include/`.

Compilation : AdaptiveCpp (`acpp`), téléchargé automatiquement au premier `driver.call`.
