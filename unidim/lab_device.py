import os
# https://docs.jax.dev/en/latest/gpu_memory_allocation.html
preallocate = True
mem_fraction = 0.75
print(str(preallocate).lower())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]= str(preallocate).lower()
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]= str(mem_fraction)

mem_frac = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.75)
print("XLA_PYTHON_CLIENT_PREALLOCATE=", preallocate)
print("XLA_PYTHON_CLIENT_MEM_FRACTION", mem_frac)
import jax
import nvidia_smi

print("jax version ",jax.__version__)
device = jax.devices()[0]
print(device)

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

def info(device):
    print(device.memory_stats())
    stats = device.memory_stats()
    # print(device.addressable_memories())
    # print(device.default_memory())
    mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f" JAX used/total  : {stats["bytes_in_use"]/1024**3:.3f} / {stats["bytes_limit"]/1024**3:.3f} GiB")
    print(f" JAX 'pool_bytes'  : {stats['pool_bytes'] / 1024 ** 3:.3f} ")
    print(f"NVML used/total: {mem.used / 1024**3:.3f} / {mem.total / 1024**3:.3f} GiB")

print("->A vide")
info(device)
x = jax.numpy.ones((10000, 10000))
x.block_until_ready()
print(" -> chargé par ones(10000, 10000)")
info(device)
"""

GPU
┌──────────────────────────────────────┐
│              pool_bytes              │
│  ┌──────────────┬─────────────────┐  │
│  │ bytes_in_use │ espace libre    │  │
│  │              │ dans le pool    │  │
│  └──────────────┴─────────────────┘  │
└──────────────────────────────────────┘
pool_bytes = 130 MB
bytes_in_use = 66 MB
130 MB réservés par JAX au pool
├── 66 MB actuellement utilisés
└── 64 MB disponibles dans le pool
Les 64 MB libres ne sont donc pas rendus immédiatement au GPU/driver. JAX peut les réutiliser pour une allocation future.

Parce que JAX utilise un memory pool / allocator pour éviter de faire constamment des allocations/libérations CUDA

XLA_PYTHON_CLIENT_PREALLOCATE=true

JAX, au premier calcul GPU, réserve à l'avance une grosse partie de la mémoire GPU disponible.

Par défaut, c'est environ 75 % de la mémoire GPU.

XLA_PYTHON_CLIENT_PREALLOCATE=false
GPU
┌─────────────────────────────┐
│ pool = 130 MB               │
│ ┌──────────┬──────────────┐ │
│ │ 66 MB    │ 64 MB libre  │ │
│ │ utilisés │              │ │
│ └──────────┴──────────────┘ │
└─────────────────────────────┘
Avec le preallocator activé, tu pourrais avoir quelque chose comme :
GPU
┌────────────────────────────────────┐
│ pool ≈ 3 Go                        │
│ ┌──────────┬─────────────────────┐ │
│ │ 66 MB    │ ~3 Go - 66 MB       │ │
│ │ utilisés │ libres dans le pool │ │
│ └──────────┴─────────────────────┘ │
└────────────────────────────────────┘

Donc :

bytes_in_use → devrait rester autour de 66 MB.

pool_bytes → peut devenir ~3 Go.

NVML used → peut également monter fortement, car le driver voit la mémoire réservée par JAX."""