# 🔧 OPTIMISATION de la reconstruction 1D

HC Laptop
fedora 44
GPU : A1000 avec 4Go
# A. Environnements 
## 1. jax cpu only
```bash
conda create -n jaxcpu_recons python=3.12 -y
conda activate jaxcpu_recons
pip install jax optax
./run bench reconstruction_jax
```
1er bench
[warmup] compiling/stabilizing JIT (n=51200)... done (231.56s)
n=   51200: 11818.256 ms/grad (38 calls)
[warmup] compiling/stabilizing JIT (n=100000)... done (285.62s)
n=  100000: 17400.780 ms/grad (35 calls)


## 2. jax avec gpu
```bash
conda create -n jaxcuda_recons python=3.12 -y
conda activate jaxcuda_recons
pip install jax[cuda13] # bonne version de cuda à mettre
pip install optax
pip install tqdm nvidia-ml-py3 matplotlib  pandas mlflow tabulate
./run bench reconstruction_jax
``` 
1er bench
[warmup] compiling/stabilizing JIT (n=51200)... done (40.78s)
n=   51200: 1485.731 ms/grad (37 calls)
[warmup] compiling/stabilizing JIT (n=100000)... done (69.36s)
n=  100000: 3414.831 ms/grad (35 calls)


## 3. Cuda optimized et Torch
```bash
conda create -n cuda_recons python=3.12 -y
conda activate cuda_recons
pip install torch[cuda13] # # pouyr la compil cuda, on utilise torch , pb n'installe pas nvcc via le cuda-toolkit=13.0 (tiré par pip tiré par torch!
conda install cuda-toolkit=13.0 # le paquet conda !=  paquet pipy dans ~/miniconda3/envs/cuda_recons/bin/nvcc
pip install tqdm nvidia-ml-py3 matplotlib  pandas mlflow tabulate
./run bench reconstruction_cuda
```
# B. Historiques des changement dans unidim

## reconstruction multiscale
- `reconstruction_cuda.py` : la version du code original, bug de compatibilité entre les headers CUDA 13.0 et les headers glibc de la machine
- `reconstruction_jax.py` : la version du code original est dans, mais très commentée
- `reconstruction_jax_simpler.py` : une version élaguée sans commentaires, standalone avec _get_chunk_size, 
   et mem_budget calculé dans le script directement via jax.device.memory_stats()
- `reconstruction_torch.py` une 1ere version : 
  - instrumentée avec ml_flow et torch_profiler 
  - Warning : `torch.searchsorted()`: input value tensor is non-contiguous , this will lower the performance

## décomposition du pipeline
- `W2_precision_and_bench_batchsize.py`: 
  - influence de la précission sur le calcul de la distance W2, en float32 parfois W2 < 0
  - Mesurer l'effet de batch_size, compromis mémoire ↔ parallélisme ↔ temps de calcul.
- `lab_jax_device.py` : comprendre la gestion de la mémoire jax notamment via `jax_device.memory_stats()`, préallocation, pool_bytes, used_bytes
- `lab_wasser.py` : chgt sur w2 : `_w2_1d_new` a plus de code compilé, mais moins de mémoire temporaire.
- `lab_jax_map` : à cette échelle, les trois implémentations de W2 sont quasiment identiques en coût d'exécution et en mémoire pour
- `lab_lgbfs2_compare_optax_jax.py` : comparaison `optax.value_and_grad_from_state` à `jax.value_and_grad`,
  la version optax plus l"gérement plus rapide en compilation
- `lab_lgbfs_linesearch_optim.py` : rapidité → `max_linesearch_steps`=1, meilleure convergence → `max_linesearch_steps`=4

## optimisations monoscale
- `lab_lgbfs2_jax.py` : optim avec optax sur N points, mesure gpu et temps , reporting dans mlflow
- `lab_lgbfs2_torch.py` : optim avec torch.optim.LBFGS sur N points, mesure gpu et temps , reporting dans mlflow