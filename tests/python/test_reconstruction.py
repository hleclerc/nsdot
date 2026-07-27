from pathlib import Path
import sys

# le code vit dans applications/ (hors package sdot) : on ajoute `applications/` au path
sys.path.insert( 0, str( Path( __file__ ).resolve().parents[ 2 ] / "applications" ) )

import numpy as np

from reconstruction.Sinogram import Sinogram
from reconstruction.reconstruction import loss, reconstruct, random_positions
from reconstruction.optimizers import GradientDescent, LBFGS
from . import test


def _disk_sinogram( nb_angles = 8, nb_bins = 201, extent = 6.0, center = ( 0.3, -0.2 ), radius = 1.0 ):
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    s.add_disk( center = list( center ), radius = radius )
    return s


if test( "in_disk_gives_small_loss" ):
    # des diracs échantillonnant le disque cible reproduisent ses projections : la perte doit être
    # petite. Exerce aussi le chemin OT sur des profils à densité NULLE (queues du détecteur).
    sino = _disk_sinogram()

    rng = np.random.default_rng( 0 )
    pts = []
    while len( pts ) < 400:
        xy = ( rng.random( 2 ) - 0.5 ) * 2
        if xy[ 0 ] ** 2 + xy[ 1 ] ** 2 < 1:
            pts.append( xy + np.array( [ 0.3, -0.2 ] ) )
    pts = np.array( pts )

    l = float( loss( sino, pts ) )
    assert np.isfinite( l )
    assert l < 0.05, f"perte trop grande pour des diracs dans le disque : { l }"


if test( "reconstruct_converges" ):
    # à partir de diracs aléatoires, la descente doit faire chuter la perte et amener les diracs
    # DANS le disque (position moyenne ~ centre).
    center, radius = np.array( [ 0.3, -0.2 ] ), 1.0
    sino = _disk_sinogram( nb_angles = 8, center = center, radius = radius )

    nb_steps = 100
    p0 = random_positions( 100, extent = 5.0, seed = 3 )
    l0 = float( loss( sino, p0 ) )

    # on capture ~10 instantaneous positions le long de la descente, pour l'animation
    frames = [ np.asarray( p0 ) ]
    every = max( 1, nb_steps // 10 )
    def snap( step, pts ):
        if step % every == every - 1 or step == nb_steps - 1:
            frames.append( np.asarray( pts ) )

    p = reconstruct( sino, p0, lr = 0.5, nb_steps = nb_steps, callback = snap )
    l1 = float( loss( sino, p ) )

    assert l1 < l0 / 10, f"la perte n'a pas suffisamment chuté : { l0 } -> { l1 }"

    pn = np.asarray( p )
    assert np.allclose( pn.mean( axis = 0 ), center, atol = 0.1 )
    dist = np.linalg.norm( pn - center, axis = 1 )
    assert ( dist <= radius + 0.1 ).mean() > 0.9, "la plupart des diracs devraient être dans le disque"

    # -- animation de la convergence (disque cible en repère) --------------
    # from matplotlib import pyplot
    # from matplotlib.animation import FuncAnimation

    # fig, ax = pyplot.subplots()
    # theta = np.linspace( 0, 2 * np.pi, 200 )
    # ax.plot( center[ 0 ] + radius * np.cos( theta ), center[ 1 ] + radius * np.sin( theta ), "-", color = "0.7" )
    # ax.set_aspect( "equal" )
    # ax.set_xlim( center[ 0 ] - 2.5, center[ 0 ] + 2.5 )
    # ax.set_ylim( center[ 1 ] - 2.5, center[ 1 ] + 2.5 )
    # scat = ax.plot( frames[ 0 ][ :, 0 ], frames[ 0 ][ :, 1 ], ".", color = "C0" )[ 0 ]

    # def update( i ):
    #     f = frames[ i ]
    #     scat.set_data( f[ :, 0 ], f[ :, 1 ] )
    #     ax.set_title( f"image { i + 1 } / { len( frames ) }" )
    #     return scat,

    # anim = FuncAnimation( fig, update, frames = len( frames ), interval = 400, blit = False )
    # pyplot.show()


if test( "lbfgs_vs_gradient_descent" ):
    # Comparer la convergence de LBFGS et gradient descent sur le même problème.
    # LBFGS devrait converger plus vite (moins d'itérations) et atteindre une perte comparable ou meilleure.
    center, radius = np.array( [ 0.3, -0.2 ] ), 1.0
    sino = _disk_sinogram( nb_angles = 12, center = center, radius = radius )

    nb_diracs = 80
    p0 = random_positions( nb_diracs, extent = 5.0, seed = 42 )
    l0 = float( loss( sino, p0 ) )

    # Gradient Descent : 200 itérations
    gd_losses = []
    def gd_callback( step, pts ):
        gd_losses.append( float( loss( sino, pts ) ) )

    p_gd = reconstruct( sino, p0, optimizer=GradientDescent( lr=0.2, nb_steps=200 ), callback=gd_callback )
    l_gd = gd_losses[ -1 ]

    # LBFGS : jusqu'à 200 itérations
    lbfgs_losses = []
    def lbfgs_callback( step, pts ):
        lbfgs_losses.append( float( loss( sino, pts ) ) )

    p_lbfgs = reconstruct( sino, p0, optimizer=LBFGS( max_iter=200, ftol=1e-8 ), callback=lbfgs_callback )
    l_lbfgs = lbfgs_losses[ -1 ]

    print( f"\n  GD:    {len(gd_losses):3d} steps, loss {l0:.6f} -> {l_gd:.6f} (ratio {l_gd/l0:.4f})" )
    print( f"  LBFGS: {len(lbfgs_losses):3d} steps, loss {l0:.6f} -> {l_lbfgs:.6f} (ratio {l_lbfgs/l0:.4f})" )
    print( f"  Speedup: {len(gd_losses)/len(lbfgs_losses):.1f}x fewer iterations with LBFGS" )

    # LBFGS devrait converger en beaucoup moins d'itérations
    assert len( lbfgs_losses ) < len( gd_losses ) / 2, \
        f"LBFGS should converge faster: {len(lbfgs_losses)} vs {len(gd_losses)}"

    # LBFGS devrait atteindre une perte similaire ou meilleure
    assert l_lbfgs <= l_gd * 1.01, \
        f"LBFGS should reach similar or better loss: {l_lbfgs} vs {l_gd}"


if test( "lbfgs_quality" ):
    # LBFGS seul : vérifier qu'il atteint une bonne qualité de reconstruction.
    center, radius = np.array( [ 0.5, 0.1 ] ), 0.8
    sino = _disk_sinogram( nb_angles = 16, center = center, radius = radius )

    p0 = random_positions( 100, extent = 4.0, seed = 123 )
    l0 = float( loss( sino, p0 ) )

    p = reconstruct( sino, p0, optimizer=LBFGS( max_iter=150, ftol=1e-9 ) )
    l = float( loss( sino, p ) )

    print( f"\n  LBFGS quality: loss {l0:.6f} -> {l:.6f} (ratio {l/l0:.4f})" )

    # Vérifier convergence suffisante
    assert l < l0 / 20, f"LBFGS should reduce loss significantly: {l0} -> {l}"

    # Vérifier que les diracs sont bien positionnés
    pn = np.asarray( p )
    assert np.allclose( pn.mean( axis=0 ), center, atol=0.15 ), "diracs should cluster around center"
    dist = np.linalg.norm( pn - center, axis=1 )
    assert ( dist <= radius + 0.15 ).mean() > 0.85, "most diracs should be in the disk"
