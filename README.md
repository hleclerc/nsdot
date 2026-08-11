# nsdot

Monorepo — 3 projets indépendants :

```
loom/     Interface agnostique Jax/Torch → SYCL (tensor, Aggregate, drivers, compilation)
sdot/     Transport optimal semi-discret (Cell, OtPlan1d, distributions)
otrec/    Application de reconstruction CT (Reconstruction, Sinogram)
```

Chaque projet a son propre `pyproject.toml`. Dépendances : `otrec` → `sdot` → `loom`.

## Utilisation

Le runner unifié `./run` couvre tout :

```bash
./run test                          # Tous les tests (C++ + Python)
./run test --name=Cell              # Filtre par nom
./run test --fp=FP32                # Précision FP32

./run experiment lung               # Expérience (découverte automatique depuis exp_*.py)
./run experiment lung --nb-diracs=5000  # Avec override de paramètres

./run bench speed                   # Benchmark (découvert depuis bench_*.py)

./run install                       # pip install -e des 3 projets

./run toolchain                     # Diagnostic (acpp, LLVM, CUDA)

./run env list                      # Lister les environnements configurés
```

### Expériences et benchmarks

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

Pour que les fichiers de sortie soient rapatriés automatiquement en exécution distante,
l'expérience doit émettre des marqueurs `OUTPUT:` dans stdout :

```python
print("OUTPUT:tmp/resultat.html")
print("OUTPUT:tmp/figure.png")
```

## Hôtes distants

Configurer `.hosts.toml` (copier depuis `.hosts.toml.example`) :

```toml
[envs.default]
type = "micromamba"
name = "vfs"
driver = "jax"

[hosts.lmo]
hostname = "lmo"
remote_dir = "/home/leclerc/nsdot"
python = "/data/venvs/sdot/bin/python"
```

Exécution distante :

```bash
./run test --host lmo                # Tests sur lmo
./run experiment --host lmo lung     # Expérience sur lmo, rsync auto des outputs
./run toolchain --host lmo           # Diagnostic distant
```

Le flux distant : `rsync` → `ssh` → exécution → parse `OUTPUT:` → `rsync` ciblé.

## Environnements

`.hosts.toml` peut déclarer plusieurs environnements (micromamba, apptainer/sif).
Le runner sélectionne automatiquement l'env selon le driver (`--driver jax|torch`).

```toml
[envs.default]
type = "micromamba"
name = "vfs"
driver = "jax"

[envs.torch]
type = "micromamba"
name = "vfs-torch"
driver = "torch"
```

## C++

Les headers C++ sont dans `loom/include/loom/support/` (runtime générique)
et `sdot/include/sdot/` (transport optimal). Les headers générés (JIT)
atterrissent dans `build/include/`.

Compilation : AdaptiveCpp (`acpp`), téléchargé automatiquement au premier `driver.call`.
