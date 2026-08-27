# INFO EXEMPLE RECONSTRUCTION
HC Laptop
fedora 44
GPU : A1000 avec 4Go

# 1. jax cpu only
```bash
conda create -n jaxcpu_recons python=3.12 -y
conda activate jaxcpu_recons
pip install -e ~/Projects/nsdot/loom/  # installe les dépendances nécessaires pour le ./run
pip install jax # bonne version de cuda à mettre
pip install optax
./run bench reconstruction_jax
```
[warmup] compiling/stabilizing JIT (n=51200)... done (231.56s)
n=   51200: 11818.256 ms/grad (38 calls)
[warmup] compiling/stabilizing JIT (n=100000)... done (285.62s)
n=  100000: 17400.780 ms/grad (35 calls)


# 2. jax + cuda
```bash
conda create -n jaxcuda_recons python=3.12 -y
conda activate jaxcuda_recons
pip install -e ~/Projects/nsdot/loom/  # installe les dépendances nécessaires pour le ./run
pip install jax[cuda13] # bonne version de cuda à mettre
pip install optax
./run bench reconstruction_jax
``` 
[warmup] compiling/stabilizing JIT (n=51200)... done (40.78s)
n=   51200: 1485.731 ms/grad (37 calls)

[warmup] compiling/stabilizing JIT (n=100000)... done (69.36s)
n=  100000: 3414.831 ms/grad (35 calls)


# 3. Cuda optimized
```bash
conda create -n test_GCHA python=3.12 -y
conda activate test_GCHA
pip install -e ~/Projects/nsdot/loom/ # installe les dépendances nécessaires pour le ./run
pip install torch[cuda13] # # pouyr la compil cuda, on utilise torch , pb n'installe pas nvcc via le cuda-toolkit=13.0 (tiré par pip tiré par torch!
conda install cuda-toolkit=13.0 # le paquet conda !=  paquet pipy dans ~/miniconda3/envs/test_GCHA/bin/nvcc
./run bench reconstruction_cuda
```
[warmup] compiling CUDA extension with nvcc (first use, ~30-50s)... done (123.7s)
n=  100000: 244.336 ms/grad (32 calls)


# TODO : version torch