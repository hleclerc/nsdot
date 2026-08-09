"""Coupe de poumon synthétique (binaire) : deux lobes pleins percés de ~1e4 alvéoles.

Tout est construit ANALYTIQUEMENT dans le domaine de Radon, par superposition de disques
(`Sinogram.add_disk`) : un lobe (densité +1) puis chaque alvéole (densité -1, un trou d'air) --
la transformée de Radon est linéaire, donc le sinogramme obtenu est EXACT, quels que soient
`nb_angles`/`nb_bins` -- jamais de discrétisation pixel de l'image d'origine.

Les alvéoles sont placées sur une grille hexagonale jittérée : le jitter est borné
analytiquement (voir `_hex_sites`) pour garantir qu'aucune paire ne peut se chevaucher, quelle
que soit la réalisation aléatoire -- le fantôme reste rigoureusement binaire (0/1 après passage
au signe : lobe moins alvéoles), pas juste "en général".
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from ..Reconstruction import Reconstruction
from ..Sinogram import Sinogram
from ..optimizers import LBFGS, GradientDescentLineSearch


def _hex_sites( lobes, max_radius, spacing_factor = 2.15, jitter_factor = 0.02, seed = 0 ):
    """Centres candidats pour des disques de rayon <= `max_radius`, sur une grille hexagonale
    jittérée, filtrés à l'intérieur de `lobes` (liste de `(center, radius)`).

    Le jitter par point est borné à `jitter_factor * spacing` (uniforme par axe, amplitude
    `a`) de façon à ce que même dans le pire cas (deux voisins jitterés l'un vers l'autre), la
    distance résiduelle reste >= 2*max_radius : `spacing - 2*a*sqrt(2) >= 2*max_radius`. Avec
    `spacing = spacing_factor*max_radius` et `jitter_factor <= (spacing_factor - 2) / (2*sqrt(2)
    * spacing_factor)`, cette garantie est déterministe (indépendante du tirage) -- pas de passe
    de rejet a posteriori nécessaire.
    """
    rng = np.random.default_rng( seed )
    spacing = spacing_factor * max_radius
    row_h = spacing * np.sqrt( 3 ) / 2
    jitter_amp = jitter_factor * spacing

    bound = max( r + float( np.linalg.norm( c ) ) for c, r in lobes ) + spacing

    rows = int( 2 * bound / row_h ) + 2
    cols = int( 2 * bound / spacing ) + 2

    ii, jj = np.meshgrid( np.arange( -rows, rows ), np.arange( -cols, cols ), indexing = "ij" )
    y = ii.ravel() * row_h
    x = jj.ravel() * spacing + np.where( ii.ravel() % 2 != 0, spacing / 2, 0.0 )
    pts = np.stack( [ x, y ], axis = 1 )
    pts = pts[ ( np.abs( pts[ :, 0 ] ) <= bound ) & ( np.abs( pts[ :, 1 ] ) <= bound ) ]

    pts = pts + ( rng.random( pts.shape ) * 2 - 1 ) * jitter_amp

    inside = np.zeros( len( pts ), dtype = bool )
    for center, radius in lobes:
        d = np.linalg.norm( pts - center[ None, : ], axis = 1 )
        inside |= d <= ( radius - max_radius * 1.05 )
    return pts[ inside ], rng


def make_lung_phantom(
    nb_angles: int = 600,
    nb_bins: int = 2000,
    extent: float = 44.0,
    nb_alveoli: int | None = None,
    alveolus_radius: float | None = None,
    scale: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
):
    """Construit le fantôme : deux lobes (disques densité 1) percés d'alvéoles (disques densité
    -1, non chevauchants -- voir `_hex_sites`). Renvoie `( sinogram, lobes, alveoli )` --
    `lobes`/`alveoli` = listes `( center, radius )`, la "vérité terrain" analytique (il n'existe
    jamais d'image pixel : seuls le sinogramme et ces disques existent).

    À `scale = 1.0`, les deux lobes tiennent dans le détecteur ([-extent/2, extent/2], marge ~2
    unités) SANS se chevaucher entre eux (sinon leur superposition dépasserait densité 1,
    brisant le caractère binaire) -- valeurs calibrées pour extent=44, à réajuster si `extent`
    change fortement. `scale > 1` agrandit lobes ET alvéoles (`alveolus_radius` par défaut suit,
    sauf si explicitement fourni) SANS changer `extent` : l'objet DÉPASSE alors le détecteur --
    voir `Sinogram.debias_and_equalize_mass` pour la correction associée (masse par angle non
    constante quand une partie de l'ombre sort de la fenêtre visible).

    `alveolus_radius=None` (par défaut) vaut `0.075 * scale` -- les alvéoles grandissent avec le
    fantôme, pour occuper la même fraction relative de la surface.

    `nb_alveoli=None` (par défaut) REMPLIT tous les sites disponibles (empaquetage hexagonal
    maximal compatible avec `_hex_sites`, ~63% de la surface des lobes en air à la densité
    par défaut) -- les alvéoles réelles occupent une grande fraction du volume pulmonaire,
    pas juste quelques trous épars ; passer un entier pour un compte explicite (plus petit,
    plus lisible à l'oeil -- utile pour un exemple pédagogique).
    """
    if alveolus_radius is None:
        alveolus_radius = 0.075 * scale

    lobes = [
        ( np.array( [ 0.0, 0.0 ] ) * scale, 0.5 * extent * scale ),
    ]

    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    for center, radius in lobes:
        sino.add_disk( center = center, radius = radius, density = 1.0 )

    sites, rng = _hex_sites( lobes, alveolus_radius, seed = seed )
    if nb_alveoli is None:
        nb_alveoli = len( sites )
    if len( sites ) < nb_alveoli:
        nb_alveoli = len( sites )
        print(
            f"seulement { len( sites ) } sites disponibles pour { nb_alveoli } alvéoles -- "
            "réduire alveolus_radius ou nb_alveoli, ou agrandir les lobes" )
    rng.shuffle( sites )
    sites = sites[ :nb_alveoli ]
    # 0.8-1.0 (plutôt que 0.55-1.0) : alvéoles plus uniformes en taille, remplissage plus dense --
    # les alvéoles réelles occupent une grande fraction du volume pulmonaire, pas juste des trous
    # épars dans un bloc de tissu.
    radii = alveolus_radius * ( 0.8 + 0.2 * rng.random( nb_alveoli ) )

    t0 = time.time()
    for i, ( center, radius ) in enumerate( zip( sites, radii ) ):
        # densité -1 : creuse un trou d'air dans le lobe (densité +1) -- la Radon transform est
        # linéaire donc la superposition reste exacte, quel que soit l'ordre d'ajout.
        sino.add_disk( center = center, radius = float( radius ), density = -1.0 )
        if verbose and ( i + 1 ) % 2000 == 0:
            print( f"  alvéole { i + 1 } / { nb_alveoli } ({ time.time() - t0:.1f}s)" )

    alveoli = list( zip( sites, radii ) )
    if verbose:
        print( f"phantom: { len( alveoli ) } alvéoles, { time.time() - t0:.1f}s" )
    return sino, lobes, alveoli


def plot_phantom( lobes, alveoli, extent, ax = None, bound = None, show_detector = False ):
    """Dessine la vérité terrain (lobes pleins, alvéoles en blanc) -- jamais rastérisée, juste
    les disques analytiques eux-mêmes.

    `bound` : demi-largeur des axes affichés. Par défaut `extent / 2` (fenêtre du détecteur) --
    mais l'objet PHYSIQUE (les lobes) peut être plus grand que ça (`scale > 1`, voir
    `make_lung_phantom`) : passer explicitement le rayon englobant des lobes pour ne pas les
    couper à l'affichage. `show_detector` dessine alors la fenêtre du détecteur en pointillés,
    pour visualiser ce que le capteur voit réellement.
    """
    if ax is None:
        _, ax = plt.subplots( figsize = ( 6, 6 ) )
    if bound is None:
        bound = extent / 2
    ax.add_collection( PatchCollection(
        [ Circle( c, r ) for c, r in lobes ], facecolor = "black", edgecolor = "none" ) )
    ax.add_collection( PatchCollection(
        [ Circle( c, r ) for c, r in alveoli ], facecolor = "white", edgecolor = "none" ) )
    if show_detector:
        ax.axvline( -extent / 2, color = "red", linestyle = "--", linewidth = 1 )
        ax.axvline(  extent / 2, color = "red", linestyle = "--", linewidth = 1 )
    ax.set_xlim( -bound, bound )
    ax.set_ylim( -bound, bound )
    ax.set_aspect( "equal" )
    ax.set_title( f"{ len( alveoli ) } alvéoles (vérité terrain)" )
    return ax


def plot_sinogram( sino, ax = None ):
    if ax is None:
        _, ax = plt.subplots( figsize = ( 6, 6 ) )
    ax.imshow( np.asarray( sino.values ), aspect = "auto", cmap = "gray",
               extent = [ sino.s_min, sino.s_min + sino.extent, int( sino.nb_angles.value ), 0 ] )
    ax.set_xlabel( "détecteur s" )
    ax.set_ylabel( "angle k" )
    ax.set_title( "sinogramme" )
    return ax


def run( nb_alveoli = 1_000, alveolus_radius = 0.75, nb_diracs = 10_000, max_iter = 60,
         out = "tmp/lung_alveoli.png", plot_max_points = 300_000 ):
    print( f"génération du fantôme ({ nb_alveoli } alvéoles)..." )
    sino, lobes, alveoli = make_lung_phantom( nb_alveoli = nb_alveoli, alveolus_radius = alveolus_radius )

    fig, axes = plt.subplots( 1, 3, figsize = ( 18, 6 ) )
    plot_phantom( lobes, alveoli, extent = sino.extent, ax = axes[ 0 ] )
    plot_sinogram( sino, ax = axes[ 1 ] )

    print( f"reconstruction ({ nb_diracs } diracs, LBFGS max_iter={ max_iter })..." )
    rec = Reconstruction( sino, max_iter = max_iter, ftol = 1e-10, verbose = True )
    rec.random_points( nb_diracs, seed = 1 )
    t0 = time.time()
    losses = []
    def callback( step, pos ):
        if step % 10 == 0 or step == -1:
            l = rec.loss( points = pos )
            losses.append( ( step, l ) )
            print( f"  step { step }: loss = { l:.6f} ({ time.time() - t0:.1f}s)" )
    # backend="sycl" : kernel SYCL fusionné (`dirac_sycl.diracs_cost_grad`), ~10-30x plus rapide
    # que le chemin pur Jax par défaut à cette échelle (voir
    # `benchmarks/execution_speed/benchmark_fused.py`).
    rec.diracs( callback = callback, backend = "sycl" )
    print( f"reconstruction terminée en { time.time() - t0:.1f}s" )

    pos = rec.positions
    if len( pos ) > plot_max_points:
        # sous-échantillonnage pour l'affichage seulement -- la reconstruction elle-même a
        # utilisé les `nb_diracs` points en entier, ceci n'affecte que le rendu matplotlib
        # (un scatter à 1e7 points serait ingérable en mémoire/temps de rendu).
        idx = np.random.default_rng( 0 ).choice( len( pos ), plot_max_points, replace = False )
        pos = pos[ idx ]
    axes[ 2 ].plot( pos[ :, 0 ], pos[ :, 1 ], '.', markersize = 0.5, color = "black" )
    axes[ 2 ].set_xlim( -sino.extent / 2, sino.extent / 2 )
    axes[ 2 ].set_ylim( -sino.extent / 2, sino.extent / 2 )
    axes[ 2 ].set_aspect( "equal" )
    axes[ 2 ].set_title( f"reconstruction ({ nb_diracs } diracs)" )

    fig.tight_layout()
    fig.savefig( out, dpi = 150 )
    print( f"figure sauvée: { out }" )


def run_truncated( nb_alveoli = 1000, scale = 1.0, nb_diracs_final = 4992*4,
                    out_phantom = "tmp/lung_truncated_phantom.png",
                    out_reconstruction = "tmp/lung_truncated_reconstruction.html",
                    point_radius = 0.05, animate = True, polish_steps = 0, polish_lr = 20.0,
                    disks_min_iter = 10, disks_disp_tol_frac = 1e-3, disks_split_before = 1 ):
    """Objet plus grand que le détecteur (`scale=2`, `extent` inchangé) : l'ombre de
    l'objet dépasse la fenêtre visible à certains angles, la masse mesurée par angle n'est donc
    plus constante. `Sinogram.debias_and_equalize_mass` corrige ça (décalage additif par angle,
    masses égalisées avant tout rescale -- voir sa docstring) ; on ne montre ici que la vérité
    terrain et la reconstruction obtenue à partir du sinogramme CORRIGÉ (voir git log pour la
    comparaison avec le sinogramme brut non corrigé, qui produisait un artefact en sablier).

    Sortie : `out_phantom` (PNG, vérité terrain -- peu de disques, matplotlib convient très bien)
    et `out_reconstruction` (HTML autonome, nuage de points -- voir `export_positions_html`,
    bien plus lisible qu'un scatter matplotlib à `nb_diracs_final` points).

    `animate` (par défaut True) : capture une frame à CHAQUE pas d'optimiseur de CHAQUE étage de
    `Reconstruction.multiscale` (grâce à `record`) -- la page HTML obtient
    alors une barre de temps + un bouton play/pause pour rejouer le déplacement/raffinement des
    points jusqu'à convergence. `False` revient au comportement historique (une seule frame, la
    reconstruction finale).

    `polish_steps` : après `multiscale`, un pas supplémentaire de descente de
    gradient AVEC line search (`GradientDescentLineSearch`) sur `nb_diracs_final` points --
    LBFGS bloque parfois sur ce problème (pas vraiment différentiable, points selles : la
    line search de scipy s'arrête sans avoir vraiment convergé). Le backtracking simple de
    `GradientDescentLineSearch` n'a pas ces exigences de courbure et peut continuer à
    grappiller de la loss là où LBFGS s'est arrêté. `0` désactive cette étape.

    `polish_lr` : le gradient au point où LBFGS s'est arrêté est déjà petit (résidu d'un
    quasi-point-critique) -- `lr=1.0` n'y produit qu'un déplacement de l'ordre de 1e-4 fois
    l'étendue, invisible à l'affichage. Le backtracking étant sans risque (il ne fait que
    RÉDUIRE le pas si `lr` overshoot, jamais l'inverse), on part volontairement large ; mesuré
    empiriquement, la condition d'Armijo n'a besoin de reculer qu'à partir de `lr` de l'ordre de
    100 sur ce problème -- 20 laisse de la marge tout en donnant un pas très supérieur à 1.0.

    `disks_min_iter`/`disks_disp_tol_frac` : chaque étage `disks()` (voir `Reconstruction.disks`)
    hérite d'un nuage venant de `diracs()` -- essentiellement un dirac par case du sinogramme
    (`models.sinogram_diracs`). Sur ce point de départ, la perte disques a souvent des directions
    PLATES (un disque peut glisser sans changer le résidu tant qu'il ne chevauche personne) : sans
    garde-fou, L-BFGS-B peut y conclure à convergence après 0-1 pas, alors que les CENTRES restent
    loin d'une disposition régulière. `disks_min_iter` force ce nombre de pas minimum ;
    `disks_disp_tol_frac` (fraction du rayon courant de l'étage) arrête ENSUITE dès que les points
    cessent de bouger significativement -- `None` désactive ce second critère (seul `ftol`
    scipy natif arrête alors la phase suivant `disks_min_iter`).

    `disks_split_before` : si > 1, SUBDIVISE le nuage (voir `Reconstruction.split`) juste avant
    chaque étage disques -- plus de centres pour un même rayon donnent plus de degrés de liberté
    de réarrangement local, utile si le nuage hérité des diracs est trop clairsemé pour le rayon
    demandé et se retrouve VERROUILLÉ (chevauchements imposés par le voisinage). `1` (défaut) ne
    change rien -- à activer/tuner à la main si `disks_min_iter` seul ne suffit pas.
    """
    print( f"génération du fantôme surdimensionné (scale={ scale })..." )
    sino, lobes, alveoli = make_lung_phantom( nb_alveoli = nb_alveoli, scale = scale, alveolus_radius = 0.7, nb_bins = 2000 )
    bound = max( r + float( np.linalg.norm( c ) ) for c, r in lobes ) * 1.05
    recon_extent = 2 * bound

    raw_mass = np.asarray( sino.mass() )
    corrected = sino # .debias_and_equalize_mass()
    corr_mass = np.asarray( corrected.mass() )
    print( f"masse brute par angle : min={ raw_mass.min():.1f} max={ raw_mass.max():.1f} "
           f"(écart-type { raw_mass.std():.1f})" )
    print( f"masse corrigée par angle : min={ corr_mass.min():.1f} max={ corr_mass.max():.1f} "
           f"(écart-type { corr_mass.std():.2e})" )

    ax = plot_phantom( lobes, alveoli, extent = sino.extent, bound = bound, show_detector = True )
    ax.set_title( f"vérité terrain (scale={ scale }, fenêtre détecteur en pointillés)" )
    ax.figure.tight_layout()
    ax.figure.savefig( out_phantom, dpi = 150 )
    print( f"figure sauvée: { out_phantom }" )

    print( "reconstruction (sinogramme corrigé)..." )
    t0 = time.time()
    # une seule `Reconstruction` pour TOUTE la chaîne (multiscale puis polish) : elle garde le
    # nuage courant, la trajectoire (`record`) et l'historique d'un étage à l'autre.
    rec = Reconstruction( corrected, extent = recon_extent, record = animate, verbose = True )

    rec.random_points( nb_diracs_final // 4**2 )
    for i in range( 3 ):
        if i:
            rec.split( factor = 4, noise_frac = 1e-2 )
        # backend="sycl" : kernel SYCL fusionné, ~10-30x plus rapide que le chemin pur Jax à cette
        # échelle (600 angles, milliers de diracs -- voir `benchmarks/execution_speed/benchmark_fused.py`).
        rec.diracs( backend = "sycl" )
        # r = 0.08 * 2 ** ( 2 - i )
        # rec.disks( r, min_iter = disks_min_iter,
        #           disp_tol = None if disks_disp_tol_frac is None else disks_disp_tol_frac * r,
        #           split_before = disks_split_before )
        # rec.multiscale(
        #     nb_points_final = nb_diracs_final, nb_points_init = nb_diracs_final // 4**2, factor = 4,
        #     optimizer_factory = lambda n: LBFGS( max_iter = 40, ftol = 1e-9 ),
        #     noise_frac = 1e-2
        # )

        # if polish_steps > 0:
        #     print( f"polish (descente de gradient + line search, { polish_steps } pas)..." )
        #     def polish_callback( step, positions ):
        #         if step % 5 == 0 or step == -1:
        #             l = rec.loss( points = positions )
        #             print( f"  step { step }: loss = { l:.6f} ({ time.time() - t0:.1f}s)" )
        #     rec.diracs( optimizer = GradientDescentLineSearch( lr = polish_lr, nb_steps = polish_steps ),
        #                 callback = polish_callback, label = "polish" )

        # rec.disks( 0.18/3, max_iter = 200, ftol = 1e-13 )

    print( f"reconstruction terminée en { time.time() - t0:.1f}s" )

    rec.export_html( out_reconstruction, point_radius = point_radius,
                     title = "reconstruction poumon" )


if __name__ == "__main__":
    run_truncated()
