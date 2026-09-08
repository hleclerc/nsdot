import jax
import jax.numpy as jnp
from lab_wasser import _w2_1d as wasser_dist
import time
print(f"wasser_dist used : {wasser_dist.__name__}")
print("jax_lmap module, jax_enable_x64 :", jax.config.read("jax_enable_x64"))

def loss_lax_map(points, normals, bin_mass, bin_edges, batch_size, ext_dtype, use_checkpoint=True):
    # normals = jax.lax.stop_gradient(normals)
    # bin_mass = jax.lax.stop_gradient(bin_mass)
    # bin_edges = jax.lax.stop_gradient(bin_edges)
    def angle_cost(normal):
        projections = points @ normal
        w = wasser_dist(projections, bin_mass, bin_edges, ext_dtype)
        return w

    if use_checkpoint:
        angle_cost = jax.checkpoint(angle_cost)
    costs = jax.lax.map(angle_cost, normals, batch_size=batch_size,)

    return jnp.sum(costs).astype(jnp.float32)

if __name__ == '__main__':
    NB_POINTS = 10_000
    NB_ANGLES = 600
    NB_BINS = 4096
    use_checkpoint = False

    # Points initiaux aléatoires dans [-1, 1]²
    key = jax.random.PRNGKey(0)
    points = jax.random.uniform(key,shape=(NB_POINTS, 2),minval=-1.0,maxval=1.0,dtype=jnp.float32)
    # 600 normales réparties uniformément sur [0, pi)
    theta = jnp.linspace(0.0,jnp.pi,NB_ANGLES,endpoint=False,dtype=jnp.float32)
    normals = jnp.stack([jnp.cos(theta), jnp.sin(theta)],axis=1,)
    # Mesure cible uniforme sur [-1, 1]
    bin_edges = jnp.linspace( -1.0,1.0,NB_BINS + 1,dtype=jnp.float32)
    bin_mass = jnp.ones(NB_BINS,dtype=jnp.float32,)
    bin_mass /= jnp.sum(bin_mass)
    # Taille de batch utilisée par lax.map
    # Précision utilisée à l'intérieur du calcul de W2
    ext_dtype = jnp.float64
    print("dtype du calcul W2 :", ext_dtype)
    print("points dtype :", points.dtype)
    print("normals dtype :", normals.dtype)
    print("bin_edges dtype :", bin_edges.dtype)
    print("bin_mass dtype :", bin_mass.dtype)
    print(f"checkpoint : {use_checkpoint}")
    batch_size = 16 #  [1,16,64,600]:
    print("-"*60)
    print(f"Opti lax.map with batchsize : {batch_size} , wasser : {wasser_dist.__name__}")
    start = time.perf_counter()
    loss_lax_map(points, normals, bin_mass, bin_edges, batch_size, ext_dtype, use_checkpoint=use_checkpoint)
    time.perf_counter() - start
    print("Time for loss_lax_map Compil  + run  ", round(time.perf_counter() - start, 2) )
    loss_jit = jax.jit(
        lambda points: loss_lax_map(points,normals,bin_mass,bin_edges,batch_size,ext_dtype))

    start = time.perf_counter()
    loss = loss_jit(points)
    loss.block_until_ready()
    print("Time run compiled version :", round(time.perf_counter() - start, 2) )

    loss_compiled = loss_jit.lower(points).compile()

    print("\nMémoire du loss_lax_map compilé")
    print(loss_compiled.memory_analysis())
    print(f"Loss = {float(loss):.12e}")

    """  
    il faut distinguer le choix de batch_size pour la loss seule du choix pour L-BFGS + gradient.  
    Tes résultats avec checkpoint ON/OFF sont pratiquement identiques :
    | batch_size |     run ON |  mémoire ON |    run OFF | mémoire OFF |
    | ---------: | ---------: | ----------: | ---------: | ----------: |
    |          1 |     1.15 s |     0.52 MB |     1.16 s |     0.52 MB |
    |         16 |     1.18 s |     1.93 MB | **1.09 s** |     1.93 MB |
    |         64 |     1.08 s |     7.69 MB |     1.14 s |     7.69 MB |
    |    **600** | **0.91 s** | **72.0 MB** | **0.92 s** | **72.0 MB** |
    
    
    600 → 0.92 s / 72 MB
    64  → 1.14 s / 7.7 MB compromis pour préparer L-BFGS
    """



    loss_and_grad = jax.jit(
        jax.value_and_grad(lambda points: loss_lax_map(points, normals, bin_mass, bin_edges,
                batch_size, ext_dtype,use_checkpoint)
        )
    )
    start = time.perf_counter()

    loss_value, grad = loss_and_grad(points)
    loss_value.block_until_ready()
    grad.block_until_ready()

    time_grad_compile_run = time.perf_counter() - start
    print("Time gradient Compil + run :", round(time_grad_compile_run, 2),"s")
    start = time.perf_counter()
    loss_value, grad = loss_and_grad(points)

    loss_value.block_until_ready()
    grad.block_until_ready()

    time_grad_run = time.perf_counter() - start
    print("Time gradient run compiled :",round(time_grad_run, 2),"s")
    grad_compiled = loss_and_grad.lower(points,).compile()
    print("\nMémoire du loss + gradient compilé")
    print(grad_compiled.memory_analysis())
    grad_norm = jnp.linalg.norm(grad,)
    print(f"Loss = {float(loss_value):.12e}")
    print(f"Gradient dtype = {grad.dtype}")
    print(f"Gradient shape = {grad.shape}")
    print(f"||gradient|| = {float(grad_norm):.12e}")
    """
    batch_size=64
    |                        | Checkpoint ON | Checkpoint OFF |
    | ---------------------- | ------------: | -------------: |
    | Loss compilée          |        0.16 s |     **0.11 s** |
    | Temp mémoire loss+grad |   **15.5 MB** |        82.1 MB |
    | Code généré            |        110 KB |        77.6 KB |
    | Gradient               |       float32 |        float32 |
    | Norme gradient         |    0.31720358 |     0.31720358 |
    """

    """"
    | Configuration |   Gradient | Temp memory |
    | ------------- | ---------: | ----------: |
    | 16 + OFF      | **0.10 s** |     74.5 MB |
    | **16 + ON**   | **0.13 s** | **3.95 MB** |
    | 64 + OFF      |     0.11 s |     82.1 MB |
    | 64 + ON       |     0.15 s |     15.5 MB |
    
    
    
    """