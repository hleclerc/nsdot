import numpy

from loom.testing import test, experiment, Param

from sdot import Visualizer, Voronoi


def _measures( v ):
    return numpy.asarray( v.measures.tensor ).reshape( -1 )


def _monte_carlo_measures( pos, mi, ma, nb_samples = 200000, seed = 0 ):
    """Les mesures par ÉCHANTILLONNAGE : on tire des points du pavé et on compte, pour chaque
    germe, la fraction qui lui est plus proche qu'à tout autre.

    Aucun code géométrique en commun avec le kernel -- c'est la DÉFINITION du diagramme (« plus
    proche germe »), pas une autre façon de découper des demi-espaces. Ce qu'il vérifie, ce n'est
    donc pas le découpage mais ce qu'on prétend découper.
    """
    rng = numpy.random.default_rng( seed )
    mi, ma = numpy.asarray( mi, float ), numpy.asarray( ma, float )
    pts = rng.uniform( mi, ma, size = ( nb_samples, mi.size ) )
    d2 = ( ( pts[ :, None, : ] - pos[ None, :, : ] ) ** 2 ).sum( axis = 2 )
    nearest = d2.argmin( axis = 1 )
    volume = float( numpy.prod( ma - mi ) )
    return numpy.bincount( nearest, minlength = len( pos ) ) / nb_samples * volume


if test( "basic_2D" ):
    # deux germes symétriques dans le carré unité : chacun en prend la moitié.
    v = Voronoi( numpy.array( [ [ 0.25, 0.5 ], [ 0.75, 0.5 ] ] ), box = ( [ 0, 0 ], [ 1, 1 ] ) )
    assert numpy.allclose( _measures( v ), 0.5 )

if test( "one_seed_takes_everything" ):
    # pas de bissectrice du tout : il ne reste que le domaine.
    for d in ( 2, 3, 4 ):
        v = Voronoi( numpy.full( ( 1, d ), 0.3 ), box = ( numpy.zeros( d ), numpy.full( d, 2.0 ) ) )
        assert abs( float( _measures( v )[ 0 ] ) - 2.0 ** d ) < 1e-12

if test( "sum_is_the_domain" ):
    # les cellules PAVENT le domaine : leur somme en est le volume, quelle que soit la dimension.
    # C'est le seul test qui prend le diagramme comme un tout -- une cellule fausse d'un côté et
    # fausse à l'envers de l'autre est ce qu'il attrape.
    for d, n in ( ( 2, 60 ), ( 3, 40 ), ( 4, 20 ) ):
        rng = numpy.random.default_rng( 12 + d )
        pos = rng.uniform( 0.05, 0.95, size = ( n, d ) )
        v = Voronoi( pos, box = ( numpy.zeros( d ), numpy.ones( d ) ) )
        m = _measures( v )
        assert m.shape == ( n, )
        assert ( m > 0 ).all()
        assert abs( float( m.sum() ) - 1 ) < 1e-10

if test( "matches_the_cell_by_cell_build" ):
    # la même géométrie par une orchestration entièrement différente : `Voronoi.cell` refait la
    # cellule côté Python, un `driver.call` par coupe, sur les tampons d'une `Cell` ordinaire.
    for d, n in ( ( 2, 12 ), ( 3, 10 ), ( 4, 6 ) ):
        rng = numpy.random.default_rng( 5 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        v = Voronoi( pos, box = ( numpy.zeros( d ), numpy.ones( d ) ) )
        m = _measures( v )
        ref = numpy.array( [ float( v.cell( i ).measure ) for i in range( n ) ] )
        assert numpy.allclose( m, ref, atol = 1e-12 ), ( d, m - ref )

if test( "is_the_nearest_seed_partition" ):
    # ce que le découpage prétend découper. Erreur d'échantillonnage en 1/sqrt(N) : la tolérance
    # est lâche exprès, ce test-là ne cherche pas la précision mais la bonne partition.
    for d, n in ( ( 2, 8 ), ( 3, 8 ) ):
        rng = numpy.random.default_rng( 100 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        v = Voronoi( pos, box = ( numpy.zeros( d ), numpy.ones( d ) ) )
        m = _measures( v )
        mc = _monte_carlo_measures( pos, numpy.zeros( d ), numpy.ones( d ) )
        assert numpy.abs( m - mc ).max() < 5e-3, ( d, m, mc )

if test( "an_off_centre_box" ):
    # le domaine n'a rien à voir avec l'origine ni avec les germes : les demi-espaces sont coupés
    # AVANT les bissectrices (c'est ce qui rend la cellule bornée), et un pavé quelconque le montre.
    d = 3
    mi, ma = numpy.array( [ -2.0, 1.0, 0.5 ] ), numpy.array( [ 1.0, 4.0, 2.5 ] )
    rng = numpy.random.default_rng( 7 )
    pos = rng.uniform( mi + 0.1, ma - 0.1, size = ( 15, d ) )
    v = Voronoi( pos, box = ( mi, ma ) )
    assert abs( float( _measures( v ).sum() ) - float( numpy.prod( ma - mi ) ) ) < 1e-10

if test( "a_domain_that_is_not_a_box" ):
    # `box` n'est qu'un raccourci : le domaine est une LISTE DE DEMI-ESPACES, donc n'importe quel
    # convexe polyédrique. Ici le simplexe `x, y, z >= 0, x + y + z <= 1` (volume 1/6).
    dirs = numpy.array( [ [ -1.0, 0, 0 ], [ 0, -1.0, 0 ], [ 0, 0, -1.0 ], [ 1.0, 1, 1 ] ] )
    offs = numpy.array( [ 0.0, 0, 0, 1 ] )
    rng = numpy.random.default_rng( 3 )
    pos = rng.dirichlet( numpy.ones( 4 ), size = 12 )[ :, :3 ] * 0.9
    v = Voronoi( pos, boundaries = ( dirs, offs ) )
    assert abs( float( _measures( v ).sum() ) - 1 / 6 ) < 1e-10

if test( "no_domain_leaves_the_cells_infinite" ):
    # sans domaine, toutes les cellules d'un si petit nuage partent à l'infini -- et `Cell::measure`
    # le dit comme il le dit ailleurs (`TF::max`), sans que `Voronoi` ait à en connaître.
    v = Voronoi( numpy.array( [ [ 0.0, 0 ], [ 1.0, 0 ], [ 0.0, 1 ] ] ) )
    assert ( _measures( v ) > 1e300 ).all()

if test( "more_seeds_than_work_items" ):
    # les tampons sont PAR WORK-ITEM et repassés d'un germe au suivant : une cellule qui laisserait
    # un reste derrière elle (un compte, une ligne de scratch) se verrait ici et pas sur 10 germes.
    d, n = 3, 500
    rng = numpy.random.default_rng( 42 )
    pos = rng.uniform( 0.02, 0.98, size = ( n, d ) )
    v = Voronoi( pos, box = ( numpy.zeros( d ), numpy.ones( d ) ) )
    m = _measures( v )
    assert ( m > 0 ).all()
    assert abs( float( m.sum() ) - 1 ) < 1e-9

if test( "a_capacity_too_small_is_grown_and_retried" ):
    # `max_nb_cuts` n'est qu'une supposition. Trop petite, le kernel enregistre ce qu'il aurait
    # fallu et n'écrit rien de faux ; la plateforme réserve le double et relance (`driver.call`).
    # Le résultat doit être le MÊME que celui obtenu d'emblée avec de la place.
    d, n = 3, 30
    rng = numpy.random.default_rng( 8 )
    pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
    box = ( numpy.zeros( d ), numpy.ones( d ) )
    tight = _measures( Voronoi( pos, box = box, max_nb_cuts = 5 ) )
    roomy = _measures( Voronoi( pos, box = box, max_nb_cuts = 64 ) )
    assert numpy.allclose( tight, roomy )
    assert abs( float( tight.sum() ) - 1 ) < 1e-10

if test( "seeds_on_a_grid" ):
    # la configuration DÉGÉNÉRÉE par excellence : une grille régulière, où quatre cellules se
    # rejoignent en un même point et où les bissectrices passent exactement par des sommets. Le
    # clip ne teste jamais `s == 0` (cf. `Cell::cut`) et rend donc une cellule éventuellement
    # dégénérée, mais jamais fausse : les carrés font toujours 1/9.
    xs = numpy.array( [ 1, 3, 5 ] ) / 6
    pos = numpy.array( [ [ x, y ] for x in xs for y in xs ] )
    v = Voronoi( pos, box = ( [ 0, 0 ], [ 1, 1 ] ) )
    assert numpy.allclose( _measures( v ), 1 / 9 )

if test( "seeds_on_a_3D_grid" ):
    xs = numpy.array( [ 1, 3 ] ) / 4
    pos = numpy.array( [ [ x, y, z ] for x in xs for y in xs for z in xs ] )
    v = Voronoi( pos, box = ( [ 0, 0, 0 ], [ 1, 1, 1 ] ) )
    assert numpy.allclose( _measures( v ), 1 / 8 )

if test( "a_seed_outside_the_domain_has_no_cell" ):
    # rien d'anormal : la cellule se vide, et une cellule vide mesure zéro. Ce qui compte est que
    # les autres ne s'en portent pas plus mal -- elles pavent toujours le domaine.
    pos = numpy.array( [ [ 0.5, 0.5 ], [ 0.2, 0.2 ], [ 5.0, 5.0 ] ] )
    v = Voronoi( pos, box = ( [ 0, 0 ], [ 1, 1 ] ) )
    m = _measures( v )
    assert abs( float( m[ 2 ] ) ) < 1e-14
    assert abs( float( m.sum() ) - 1 ) < 1e-12


# -- toutes les cellules d'un coup ---------------------------------------------------------------
# `measures` réduit chaque cellule à un nombre et l'oublie ; `cells` les GARDE, ce dont il n'y a
# qu'un usage -- les dessiner. Les deux passent par le même `make_cell`, donc ce qu'il reste à
# vérifier ici est la copie vers la sortie (`Cell::copy_into`) et l'espace d'items (un work-item
# par germe, plus la boucle striée de `measures`).

if test( "cells_are_the_cells" ):
    # confronté à l'oracle des autres tests : la cellule construite CÔTÉ PYTHON, un appel par
    # coupe. Même géométrie, deux orchestrations sans rien en commun.
    for d in ( 2, 3 ):
        rng = numpy.random.default_rng( 5 )
        pos = rng.uniform( 0.1, 0.9, size = ( 9, d ) )
        v = Voronoi( pos, box = ( [ 0 ] * d, [ 1 ] * d ) )

        cs = v.cells
        nvs = numpy.asarray( cs.nb_vertices.value )
        vps = numpy.asarray( cs.vertex_positions )
        for i in range( len( pos ) ):
            ref = v.cell( i )
            nv = int( ref.nb_vertices.value )
            assert nvs[ i ] == nv, ( d, i, nvs[ i ], nv )
            # en ORDRE quelconque : les deux chemins n'appliquent pas les coupes dans le même
            # ordre, donc la numérotation des sommets n'a aucune raison de coïncider.
            a = numpy.sort( numpy.asarray( ref.vertex_positions )[ : nv ].round( 9 ), axis = 0 )
            b = numpy.sort( vps[ i, : nv ].round( 9 ), axis = 0 )
            assert numpy.allclose( a, b ), ( d, i )

if test( "cells_measure_like_measures" ):
    # la copie porte TOUT ce qu'il faut pour mesurer -- y compris le treillis de faces en d > 2,
    # que rien d'autre ne relit. Une cellule batchée sait se mesurer, et doit rendre exactement
    # ce que rend le chemin qui ne garde rien.
    for d in ( 2, 3 ):
        rng = numpy.random.default_rng( 11 )
        pos = rng.uniform( 0.1, 0.9, size = ( 20, d ) )
        v = Voronoi( pos, box = ( [ 0 ] * d, [ 1 ] * d ) )
        got = numpy.asarray( v.cells.measure.tensor ).reshape( -1 )
        assert numpy.allclose( got, _measures( v ) ), d
        assert abs( float( got.sum() ) - 1 ) < 1e-10

if test( "cells_of_an_unbounded_diagram" ):
    # sans domaine, les cellules du bord restent infinies : elles gardent des plans `INFINITE`,
    # et c'est l'affichage qui sait quoi en faire (cf. `Cell.add_to_viz`). Ce qui doit tenir ici
    # est que le drapeau dise vrai -- une cellule intérieure est bornée, une du bord ne l'est pas.
    pos = numpy.array( [ [ x, y ] for x in ( 0.25, 0.5, 0.75 ) for y in ( 0.25, 0.5, 0.75 ) ] )
    bounded = numpy.asarray( Voronoi( pos ).cells.is_fully_bounded ).reshape( -1 )
    assert bounded[ 4 ] == 1                    # celle du centre, entourée
    assert bounded.sum() == 1                   # les huit autres partent à l'infini


# -- ce qu'on REGARDE ----------------------------------------------------------------------------
# Un diagramme se juge à l'oeil bien plus qu'à un nombre. Un `experiment` par régime, chacun
# écrivant la page HTML et le VTK de ParaView -- voir `Cell.add_to_viz` pour ce qu'une cellule non
# bornée montre d'elle-même (les plans factices ne sont pas envoyés, les arêtes tronquées sont en
# pointillés, celles qui ne sont que la fermeture factice ne sont pas tracées).
#
#   ./run experiment test_Voronoi                      # toutes
#   ./run experiment "test_Voronoi::vor 2D open"       # une seule
#   ./run experiment test_Voronoi --nb-points=20,200   # un balayage

def _write_both( p, viz, stem ):
    f = viz.write_html( p.out_dir / f"{ stem }.html" ).absolute()
    v = viz.write_vtk( p.out_dir / f"{ stem }.vtu" )
    print( f"  html : file://{ f }"  )
    print( "  vtk  :", v )


def _seeds( d, n, seed ):
    return numpy.random.default_rng( seed ).uniform( 0.05, 0.95, size = ( n, d ) )


if p := experiment( "vor 2D",
                    nb_points = Param( 40, help = "nombre de germes" ),
                    seed      = Param( 0, help = "graine du tirage" ) ):
    # le cas de référence : un diagramme borné par un carré. Toutes les cellules sont de vrais
    # polygones, une couleur chacune, et le pavage se voit.
    v = Voronoi( _seeds( 2, p.nb_points, p.seed ), box = ( [ 0, 0 ], [ 1, 1 ] ) )
    viz = Visualizer( title = f"Voronoï 2D, { p.nb_points } germes" )
    v.add_to_viz( viz )
    viz.add_points( numpy.asarray( v.positions ), color = "#ffffff" )
    _write_both( p, viz, "vor_2d" )

if p := experiment( "vor 2D open",
                    nb_points = Param( 25, help = "nombre de germes" ),
                    seed      = Param( 1, help = "graine du tirage" ) ):
    # SANS domaine : les cellules du bord partent à l'infini. C'est l'expérience qui montre les
    # trois règles d'un fond factice -- pas de paroi inventée, une arête tronquée en pointillés,
    # et rien du tout là où la cellule n'est fermée que par le simplexe de remplacement.
    v = Voronoi( _seeds( 2, p.nb_points, p.seed ) )
    viz = Visualizer( title = f"Voronoï 2D non borné, { p.nb_points } germes" )
    v.add_to_viz( viz )
    viz.add_points( numpy.asarray( v.positions ), color = "#ffffff" )
    _write_both( p, viz, "vor_2d_open" )

if p := experiment( "vor 3D",
                    nb_points = Param( 30, help = "nombre de germes" ),
                    seed      = Param( 2, help = "graine du tirage" ) ):
    # en 3D chaque cellule est un polyèdre plein : c'est le cas où l'opacité sert, et celui qu'on
    # ouvre dans ParaView pour couper le pavage plutôt que de le regarder de l'extérieur.
    v = Voronoi( _seeds( 3, p.nb_points, p.seed ), box = ( [ 0 ] * 3, [ 1 ] * 3 ) )
    viz = Visualizer( title = f"Voronoï 3D, { p.nb_points } germes" )
    v.add_to_viz( viz, opacity = 0.55 )
    _write_both( p, viz, "vor_3d" )

if p := experiment( "vor 3D open",
                    nb_per_side = Param( 3, help = "germes par côté de la grille" ),
                    jitter      = Param( 0.25, help = "désordre, en fraction du pas" ),
                    seed        = Param( 4, help = "graine du tirage" ) ):
    # une grille SECOUÉE plutôt qu'un tirage uniforme, et pour une raison qui n'a rien de
    # cosmétique : un sommet de Voronoï est un centre de sphère circonscrite, et quatre germes
    # presque coplanaires en donnent un très loin. Il est réel, il cadre donc la scène -- et en 3D
    # sans domaine, avec peu de germes, il est probable et il écrase tout le reste de l'image.
    # Une grille secouée n'en produit pas -- mais une grille EXACTE, elle, est le pire cas de
    # tous : ses quadruplets sont rigoureusement coplanaires, leur centre de sphère circonscrite
    # est à l'infini, et le clip en rend la valeur finie honnête, autour de 1e16 (essayer
    # `--jitter=0`). Il y a un désordre à viser, ni zéro ni trop.
    rng = numpy.random.default_rng( p.seed )
    k = p.nb_per_side
    xs = ( numpy.arange( k ) + 0.5 ) / k
    pos = numpy.array( [ [ x, y, z ] for x in xs for y in xs for z in xs ] )
    pos = pos + rng.uniform( -p.jitter, p.jitter, size = pos.shape ) / k
    v = Voronoi( pos )
    viz = Visualizer( title = f"Voronoï 3D non borné, { len( pos ) } germes" )
    v.add_to_viz( viz, opacity = 0.45 )
    _write_both( p, viz, "vor_3d_open" )

if p := experiment( "vor moving seeds",
                    nb_points = Param( 30, help = "nombre de germes" ),
                    nb_frames = Param( 12, help = "nombre d'images" ),
                    seed      = Param( 7, help = "graine du tirage" ) ):
    # une IMAGE par pas : les germes tournent, le diagramme se refait entièrement à chaque fois --
    # ce qui est littéralement ce que fait la classe, aucun diagramme n'étant conservé. La page se
    # déroule toute seule, ParaView reçoit une série temporelle.
    rng = numpy.random.default_rng( p.seed )
    pos = rng.uniform( 0.15, 0.85, size = ( p.nb_points, 2 ) )
    dirs = rng.normal( size = pos.shape )
    dirs /= numpy.linalg.norm( dirs, axis = 1, keepdims = True )

    viz = Visualizer( title = "Voronoï 2D, germes en mouvement", frame_axis = "pas" )
    for k in range( p.nb_frames ):
        if k:
            viz.new_frame( k )
        # une trajectoire circulaire : les germes reviennent, donc l'animation boucle proprement
        a = 2 * numpy.pi * k / p.nb_frames
        cur = pos + 0.08 * ( numpy.cos( a ) - 1 ) * dirs + 0.08 * numpy.sin( a ) * dirs[ :, ::-1 ]
        Voronoi( cur, box = ( [ 0, 0 ], [ 1, 1 ] ) ).add_to_viz( viz )
    _write_both( p, viz, "vor_moving" )
