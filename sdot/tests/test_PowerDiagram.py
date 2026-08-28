import numpy

from loom import driver
from loom.testing import bench, check_grad, test, experiment, Param

from sdot import AaBsp, PowerDiagram, Visualizer, Voronoi


def _measures( v ):
    return numpy.asarray( v.measures.tensor ).reshape( -1 )


def _monte_carlo_measures( pos, mi, ma, weights = None, nb_samples = 200000, seed = 0 ):
    """Les mesures par ÉCHANTILLONNAGE : on tire des points du pavé et on compte, pour chaque
    germe, la fraction qui lui est plus proche qu'à tout autre AU SENS DE LA PUISSANCE.

    Aucun code géométrique en commun avec le kernel -- c'est la DÉFINITION du diagramme
    (« `|x - d|² - w` minimal »), pas une autre façon de découper des demi-espaces. Ce qu'il
    vérifie, ce n'est donc pas le découpage mais ce qu'on prétend découper : la convention des
    poids est écrite ici une seconde fois, et sans un seul plan.
    """
    rng = numpy.random.default_rng( seed )
    mi, ma = numpy.asarray( mi, float ), numpy.asarray( ma, float )
    pts = rng.uniform( mi, ma, size = ( nb_samples, mi.size ) )
    d2 = ( ( pts[ :, None, : ] - pos[ None, :, : ] ) ** 2 ).sum( axis = 2 )
    if weights is not None:
        d2 = d2 - numpy.asarray( weights, float )[ None, : ]
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


# -- les poids -----------------------------------------------------------------------------------
# Tout ce qui précède est le cas euclidien, obtenu SANS poids (`Voronoi`). Ce qui suit fixe la
# convention -- `|x - d_i|² - w_i <= |x - d_j|² - w_j` -- et rien d'autre : la géométrie, elle, est
# déjà testée, les poids ne font que déplacer des plans.

if test( "the_bisector_is_shifted_by_the_weight_gap" ):
    # LE test de convention, et il se calcule à la main. Deux germes sur l'axe des x, en 0 et 1 :
    # le plan est en `( 1 + w_0 - w_1 ) / 2`, donc la cellule de gauche mesure exactement ça dans
    # le carré unité. Un signe à l'envers, un facteur 2 oublié, une normalisation qui traîne -- les
    # trois erreurs possibles se voient ici, chiffrées.
    for gap in ( -0.6, -0.25, 0.0, 0.25, 0.6 ):
        pd = PowerDiagram( numpy.array( [ [ 0.0, 0.5 ], [ 1.0, 0.5 ] ] ),
                           weights = numpy.array( [ gap, 0.0 ] ),
                           box = ( [ 0, 0 ], [ 1, 1 ] ) )
        m = _measures( pd )
        assert abs( float( m[ 0 ] ) - ( 1 + gap ) / 2 ) < 1e-12, ( gap, m )
        assert abs( float( m.sum() ) - 1 ) < 1e-12

if test( "equal_weights_are_no_weights" ):
    # ce que dit `Voronoi.py` : seules les DIFFÉRENCES de poids atteignent les plans. Des poids
    # tous égaux -- à n'importe quelle valeur -- doivent rendre le diagramme euclidien au bit près,
    # ce qui vérifie du même coup que le terme ajouté est bien une différence et pas une somme.
    for d, n in ( ( 2, 30 ), ( 3, 20 ) ):
        rng = numpy.random.default_rng( 20 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        box = ( numpy.zeros( d ), numpy.ones( d ) )
        ref = _measures( Voronoi( pos, box = box ) )
        for w in ( 0.0, 1.7, -3.0 ):
            got = _measures( PowerDiagram( pos, weights = numpy.full( n, w ), box = box ) )
            assert numpy.allclose( got, ref, atol = 1e-12 ), ( d, w, got - ref )

if test( "a_shift_of_every_weight_changes_nothing" ):
    # le même invariant, mais sur des poids QUELCONQUES : le diagramme ne dépend que de `w` modulo
    # les constantes. Ça ne se déduit pas du test précédent, qui ne voit que le cas plat.
    d, n = 2, 25
    rng = numpy.random.default_rng( 33 )
    pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
    w = rng.uniform( -0.05, 0.05, size = n )
    box = ( numpy.zeros( d ), numpy.ones( d ) )
    a = _measures( PowerDiagram( pos, weights = w, box = box ) )
    b = _measures( PowerDiagram( pos, weights = w + 2.5, box = box ) )
    assert numpy.allclose( a, b, atol = 1e-12 ), a - b

if test( "weighted_cells_still_tile_the_domain" ):
    # les cellules de puissance PAVENT le domaine comme celles de Voronoï -- y compris quand
    # certaines sont vides, ce que les poids rendent possible et que l'euclidien ne fait pas.
    for d, n in ( ( 2, 40 ), ( 3, 25 ), ( 4, 12 ) ):
        rng = numpy.random.default_rng( 50 + d )
        pos = rng.uniform( 0.05, 0.95, size = ( n, d ) )
        w = rng.uniform( -0.1, 0.1, size = n )
        m = _measures( PowerDiagram( pos, weights = w, box = ( numpy.zeros( d ), numpy.ones( d ) ) ) )
        assert ( m >= -1e-14 ).all(), ( d, m.min() )
        assert abs( float( m.sum() ) - 1 ) < 1e-10, ( d, m.sum() )

if test( "is_the_least_power_distance_partition" ):
    # ce que le découpage prétend découper, repris de l'euclidien avec la puissance à la place de
    # la distance. Erreur d'échantillonnage en 1/sqrt(N) : la tolérance est lâche exprès.
    for d, n in ( ( 2, 8 ), ( 3, 8 ) ):
        rng = numpy.random.default_rng( 70 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        w = rng.uniform( -0.15, 0.15, size = n )
        m = _measures( PowerDiagram( pos, weights = w, box = ( numpy.zeros( d ), numpy.ones( d ) ) ) )
        mc = _monte_carlo_measures( pos, numpy.zeros( d ), numpy.ones( d ), weights = w )
        assert numpy.abs( m - mc ).max() < 5e-3, ( d, m, mc )

if test( "matches_the_cell_by_cell_build_with_weights" ):
    # l'oracle des autres tests, côté Python : `PowerDiagram.cell` refait la cellule un
    # `driver.call` par coupe. La convention des poids y est écrite une seconde fois, en numpy --
    # si le kernel et lui s'accordent, c'est deux fois la même formule et pas deux fois le même bug.
    for d, n in ( ( 2, 12 ), ( 3, 10 ), ( 4, 6 ) ):
        rng = numpy.random.default_rng( 90 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        w = rng.uniform( -0.08, 0.08, size = n )
        pd = PowerDiagram( pos, weights = w, box = ( numpy.zeros( d ), numpy.ones( d ) ) )
        ref = numpy.array( [ float( pd.cell( i ).measure ) for i in range( n ) ] )
        assert numpy.allclose( _measures( pd ), ref, atol = 1e-12 ), ( d, _measures( pd ) - ref )

if test( "a_dominated_seed_loses_its_cell" ):
    # ce qu'un Voronoï ne sait pas faire : un germe INTÉRIEUR au domaine, et pourtant sans cellule.
    # Un poids assez bas le fait perdre partout -- et les autres pavent toujours le domaine, ce qui
    # est la vraie affirmation ici (une cellule vide ne doit rien laisser derrière elle).
    pos = numpy.array( [ [ 0.25, 0.5 ], [ 0.5, 0.5 ], [ 0.75, 0.5 ] ] )
    m = _measures( PowerDiagram( pos, weights = numpy.array( [ 0.0, -1.0, 0.0 ] ),
                                 box = ( [ 0, 0 ], [ 1, 1 ] ) ) )
    assert abs( float( m[ 1 ] ) ) < 1e-14, m
    assert abs( float( m.sum() ) - 1 ) < 1e-12, m

if test( "a_big_enough_weight_swallows_the_domain" ):
    # l'autre extrême, et la borne du précédent : un poids assez haut prend TOUT, les autres
    # cellules se vident. Le pavage doit tenir là aussi.
    pos = numpy.array( [ [ 0.25, 0.4 ], [ 0.5, 0.6 ], [ 0.8, 0.3 ] ] )
    m = _measures( PowerDiagram( pos, weights = numpy.array( [ 0.0, 10.0, 0.0 ] ),
                                 box = ( [ 0, 0 ], [ 1, 1 ] ) ) )
    assert abs( float( m[ 1 ] ) - 1 ) < 1e-12, m
    assert float( m[ 0 ] ) + float( m[ 2 ] ) < 1e-14, m

if test( "voronoi_is_the_power_diagram_without_weights" ):
    # `Voronoi` n'est pas une classe : c'est `PowerDiagram` sans le membre `weights` (voir
    # `Voronoi.py`). Il en construit donc bien un, il refuse les poids, et il rend le même
    # diagramme que des poids explicitement nuls.
    pos = numpy.random.default_rng( 4 ).uniform( 0.1, 0.9, size = ( 15, 2 ) )
    box = ( [ 0, 0 ], [ 1, 1 ] )
    v = Voronoi( pos, box = box )
    assert isinstance( v, PowerDiagram )
    assert not v.weights.is_defined              # `Unbound` -> `NoneTensor`, pas un tampon de zéros
    zeros = PowerDiagram( pos, weights = numpy.zeros( 15 ), box = box )
    assert numpy.allclose( _measures( v ), _measures( zeros ), atol = 1e-12 )

    try:
        Voronoi( pos, weights = numpy.zeros( 15 ), box = box )
    except TypeError:
        pass
    else:
        raise AssertionError( "Voronoi should refuse weights" )


# -- les dérivées -------------------------------------------------------------------------------
# `measures` se dérive par rapport aux GERMES, positions et poids (voir
# `PowerDiagram.cxx::measures_bwd`). Deux familles de tests, et elles ne se recouvrent pas :
# `check_grad` compare l'adjoint à une différence finie -- il attrape un signe, un facteur, un
# terme oublié -- tandis que les deux tests analytiques ci-dessous confrontent la jacobienne
# COMPLÈTE à sa formule classique, écrite en facettes (aire et barycentre) là où le kernel, lui,
# passe par un petit système linéaire par sommet. Deux dérivations sans rien en commun.

def _in_fp64( f ):
    """`f()` en double précision. Le kernel tourne en FP32 par défaut, ce qui suffit pour un
    volume mais pas pour comparer une jacobienne à 1e-9 près."""
    previous = driver.ftype
    driver.ftype = "FP64"
    try:
        return f()
    finally:
        driver.ftype = previous


def _facets_2d( pd, n ):
    """`( longueurs, milieux )` de chaque facette germe-germe du diagramme 2D, `[ n, n ]` et
    `[ n, n, 2 ]`, lus sur `pd.cells`. L'invariant 2D fait tout le travail : la coupe `c` porte
    l'arête `[ v_c, v_c+1 ]`, donc la facette entre `i` et `cut_ids( c )` est ce segment-là.
    """
    cs = pd.cells
    nvs = numpy.asarray( cs.nb_vertices.value ).reshape( -1 )
    vps = numpy.asarray( cs.vertex_positions )
    cis = numpy.asarray( cs.cut_ids )

    length = numpy.zeros( ( n, n ) )
    middle = numpy.zeros( ( n, n, 2 ) )
    for i in range( n ):
        nv = int( nvs[ i ] )
        for c in range( nv ):
            j = int( cis[ i, c ] )
            if j < 0:                       # le domaine : une constante, sans part de gradient
                continue
            a, b = vps[ i, c ], vps[ i, ( c + 1 ) % nv ]
            length[ i, j ] = float( numpy.linalg.norm( b - a ) )
            middle[ i, j ] = ( a + b ) / 2
    return length, middle


def _jacobian( f, x, nb_rows ):
    """La jacobienne de `f` en `x`, ligne par ligne : `nb_rows` passages de l'adjoint, chacun avec
    une cotangente un-hot. On la matérialise parce qu'on a de quoi la comparer terme à terme."""
    _, pullback = driver.vjp( lambda a: f( a ).tensor, x )
    rows = []
    for i in range( nb_rows ):
        seed = numpy.zeros( nb_rows )
        seed[ i ] = 1.0
        rows.append( numpy.asarray( pullback( seed )[ 0 ] ) )
    return numpy.array( rows )


if test( "measures_derive_wrt_the_seeds" ):
    # l'adjoint contre la différence finie, positions ET poids en même temps -- `check_grad` tire
    # une tangente au hasard sur chacune, donc un terme croisé manquant se voit aussi.
    for d, n in ( ( 2, 10 ), ( 3, 8 ) ):
        rng = numpy.random.default_rng( 200 + d )
        pos = rng.uniform( 0.15, 0.85, size = ( n, d ) )
        w = rng.uniform( -0.02, 0.02, size = n )
        box = ( numpy.zeros( d ), numpy.ones( d ) )
        check_grad( lambda p, q: PowerDiagram( p, weights = q, box = box ).measures, pos, w )

if test( "measures_derive_wrt_positions_alone" ):
    # sans poids du tout : `weights` est un `NoneTensor`, donc son gradient aussi, et la branche
    # qui l'écrit doit disparaître à la compilation sans emporter celle des positions.
    for d, n in ( ( 2, 10 ), ( 3, 8 ) ):
        rng = numpy.random.default_rng( 220 + d )
        pos = rng.uniform( 0.15, 0.85, size = ( n, d ) )
        box = ( numpy.zeros( d ), numpy.ones( d ) )
        check_grad( lambda p: Voronoi( p, box = box ).measures, pos )

if test( "measures_derive_wrt_weights_alone" ):
    # l'autre moitié : positions figées, seuls les poids bougent. C'est la dérivée dont vit un
    # solveur de transport optimal semi-discret, donc celle qu'on veut isoler.
    for d, n in ( ( 2, 12 ), ( 3, 8 ) ):
        rng = numpy.random.default_rng( 240 + d )
        pos = rng.uniform( 0.15, 0.85, size = ( n, d ) )
        w = rng.uniform( -0.02, 0.02, size = n )
        box = ( numpy.zeros( d ), numpy.ones( d ) )
        check_grad( lambda q: PowerDiagram( pos, weights = q, box = box ).measures, w )

if test( "the_weight_jacobian_is_the_facet_over_the_gap" ):
    # la formule classique : `dm_i/dw_j = -|F_ij| / ( 2 |d_i - d_j| )` pour `j != i`, et la
    # diagonale est l'opposé de la somme de sa ligne -- déplacer tous les poids ensemble ne change
    # rien, donc chaque ligne somme à zéro. C'est la matrice du Newton d'un transport optimal
    # semi-discret, et elle sort ici d'un kernel qui n'a jamais calculé une aire de facette.
    def body():
        n = 8
        rng = numpy.random.default_rng( 260 )
        pos = rng.uniform( 0.15, 0.85, size = ( n, 2 ) )
        w = rng.uniform( -0.02, 0.02, size = n )
        box = ( numpy.zeros( 2 ), numpy.ones( 2 ) )
        pd = PowerDiagram( pos, weights = w, box = box )

        length, _ = _facets_2d( pd, n )
        gap = numpy.linalg.norm( pos[ :, None, : ] - pos[ None, :, : ], axis = 2 )
        expected = numpy.zeros( ( n, n ) )
        off = length > 0
        expected[ off ] = - length[ off ] / ( 2 * gap[ off ] )
        expected[ numpy.diag_indices( n ) ] = - expected.sum( axis = 1 )

        got = _jacobian( lambda q: PowerDiagram( pos, weights = q, box = box ).measures, w, n )
        assert numpy.abs( got - expected ).max() < 1e-9, numpy.abs( got - expected ).max()
        assert numpy.abs( got.sum( axis = 1 ) ).max() < 1e-9      # invariance par translation
        assert numpy.abs( got - got.T ).max() < 1e-9              # et elle est symétrique
    _in_fp64( body )

if test( "the_position_jacobian_is_the_facet_moment" ):
    # l'autre moitié de la même dérivation : `dm_i/dd_j = |F_ij| ( d_j - b_ij ) / |d_j - d_i|`,
    # où `b_ij` est le BARYCENTRE de la facette (son milieu, en 2D). Le terme de barycentre est ce
    # qui distingue un plan qui se translate d'un plan qui pivote -- c'est lui qu'une dérivée
    # naïve oublie, et il n'apparaît nulle part dans le kernel.
    def body():
        n = 8
        rng = numpy.random.default_rng( 280 )
        pos = rng.uniform( 0.15, 0.85, size = ( n, 2 ) )
        w = rng.uniform( -0.02, 0.02, size = n )
        box = ( numpy.zeros( 2 ), numpy.ones( 2 ) )
        pd = PowerDiagram( pos, weights = w, box = box )

        length, middle = _facets_2d( pd, n )
        expected = numpy.zeros( ( n, n, 2 ) )
        for i in range( n ):
            for j in range( n ):
                if i == j or length[ i, j ] == 0:
                    continue
                nrm = float( numpy.linalg.norm( pos[ j ] - pos[ i ] ) )
                expected[ i, j ] = length[ i, j ] * ( pos[ j ] - middle[ i, j ] ) / nrm
                expected[ i, i ] += length[ i, j ] * ( middle[ i, j ] - pos[ i ] ) / nrm

        got = _jacobian( lambda p: PowerDiagram( p, weights = w, box = box ).measures, pos, n )
        assert numpy.abs( got - expected ).max() < 1e-9, numpy.abs( got - expected ).max()
    _in_fp64( body )

if test( "the_domain_carries_no_gradient" ):
    # une LIMITE, pas une propriété : une coupe du domaine porte `cut_id == BOUNDARY`, qui dit
    # « pas un germe » sans dire LEQUEL, donc le backward n'a nulle part où ranger sa part et la
    # laisse à zéro. Le test l'épingle -- pour que ça reste une décision et pas une surprise.
    # (Le gradient par rapport aux GERMES, lui, est complet : une facette du domaine ne dépend
    # d'aucun germe, sa contribution y est nulle pour de bon.)
    n = 6
    rng = numpy.random.default_rng( 300 )
    pos = rng.uniform( 0.2, 0.8, size = ( n, 2 ) )
    dirs = numpy.array( [ [ 1.0, 0 ], [ 0, 1.0 ], [ -1.0, 0 ], [ 0, -1.0 ] ] )
    offs = numpy.array( [ 1.0, 1.0, 0.0, 0.0 ] )
    _, pullback = driver.vjp(
        lambda o: PowerDiagram( pos, boundaries = ( dirs, o ) ).measures.tensor, offs )
    assert numpy.abs( numpy.asarray( pullback( numpy.ones( n ) )[ 0 ] ) ).max() == 0


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


if test( "weighted_cells_are_the_weighted_cells" ):
    # le chemin d'affichage passe par le même `make_cell`, donc les poids y sont par construction --
    # ce qui reste à vérifier est que la copie vers la sortie ne les perd pas en route, et une
    # cellule VIDE (ce que les poids rendent possible) est le cas qui le dirait.
    for d in ( 2, 3 ):
        rng = numpy.random.default_rng( 320 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( 14, d ) )
        w = rng.uniform( -0.06, 0.06, size = 14 )
        w[ 3 ] = -2.0                                  # celui-là ne gagne nulle part
        pd = PowerDiagram( pos, weights = w, box = ( [ 0 ] * d, [ 1 ] * d ) )
        got = numpy.asarray( pd.cells.measure.tensor ).reshape( -1 )
        assert numpy.allclose( got, _measures( pd ), atol = 1e-12 ), d
        assert abs( float( got[ 3 ] ) ) < 1e-14, ( d, got[ 3 ] )
        assert abs( float( got.sum() ) - 1 ) < 1e-10, d

# -- l'ACCÉLÉRATION SPATIALE ---------------------------------------------------------------------
# Un accélérateur ne peut pas changer le diagramme : il ne peut que taire des coupes qui
# n'auraient rien enlevé. C'est ce que tout ce qui suit vérifie, et c'est la SEULE chose à
# vérifier -- la géométrie, elle, est déjà testée plus haut, et elle est la même.


def _cut_sets( pd ):
    """Pour chaque cellule, l'ensemble des germes qu'elle touche vraiment.

    Plus fort qu'une mesure : deux découpages différents peuvent rendre le même volume (une coupe
    tangente n'enlève rien), mais pas le même ENSEMBLE de voisins. C'est donc le témoin direct de
    « aucune coupe utile n'a été tue » -- ce que ni les mesures ni les sommets ne diraient aussi
    précisément.
    """
    cs = pd.cells
    nbc = numpy.asarray( cs.nb_cuts.value ).reshape( -1 )
    ids = numpy.asarray( cs.cut_ids )
    return [ frozenset( int( j ) for j in ids[ i, : int( nbc[ i ] ) ] if j >= 0 )
             for i in range( len( nbc ) ) ]


def _both_ways( pos, weights = None, box = None, **kwargs ):
    """Le même diagramme, une fois en balayage complet et une fois accéléré."""
    plain = PowerDiagram( pos, weights = weights, box = box )
    acc = AaBsp.of( plain, **kwargs )
    return plain, PowerDiagram( pos, weights = weights, box = box, accelerator = acc )


if test( "the_bsp_holds_every_seed_exactly_once" ):
    # la propriété structurelle dont TOUT le reste dépend : un germe tu par l'arbre serait une
    # coupe perdue, et rien dans la marche ne pourrait la rattraper.
    for d, n, leaf in ( ( 2, 200, 30 ), ( 3, 137, 8 ), ( 2, 1, 30 ), ( 4, 40, 1 ) ):
        rng = numpy.random.default_rng( 400 + d * 17 + n )
        pos = rng.uniform( 0, 1, size = ( n, d ) )
        bsp = AaBsp( pos, max_seeds_per_leaf = leaf )

        order = numpy.asarray( bsp.seed_indices ).reshape( -1 )
        assert sorted( order.tolist() ) == list( range( n ) ), ( d, n, leaf )

        left = numpy.asarray( bsp.node_left ).reshape( -1 )
        beg = numpy.asarray( bsp.node_begin ).reshape( -1 )
        end = numpy.asarray( bsp.node_end ).reshape( -1 )
        # les feuilles PARTITIONNENT `seed_indices`, dans l'ordre : c'est ce qui rend une feuille
        # lisible d'un seul tenant.
        slices = sorted( ( int( beg[ k ] ), int( end[ k ] ) ) for k in range( len( left ) ) if left[ k ] < 0 )
        assert slices[ 0 ][ 0 ] == 0 and slices[ -1 ][ 1 ] == n, ( d, n, leaf )
        assert all( a[ 1 ] == b[ 0 ] for a, b in zip( slices, slices[ 1: ] ) ), ( d, n, leaf )
        assert all( b - a <= max( leaf, 1 ) for a, b in slices ), ( d, n, leaf )


if test( "the_tree_shape_does_not_depend_on_the_data" ):
    # la coupe est MÉDIANE, donc profondeur et nombre de nœuds sont fonction de `n` seul. C'est ce
    # qui dit qu'une construction en kernel n'aurait aucune capacité à deviner -- ni pour la pile
    # de la descente, ni pour les tableaux de nœuds. Testé jusqu'aux nuages qu'on croirait pires :
    # dix positions répétées `n` fois, où l'arbre ferme des feuilles tôt et ne fait donc que
    # rétrécir.
    for n in ( 1, 7, 30, 31, 100, 1057, 5000 ):
        for leaf in ( 1, 8, 30 ):
            for d in ( 2, 3 ):
                for kind in ( "uniforme", "degenere" ):
                    rng = numpy.random.default_rng( 401 + n + leaf )
                    if kind == "uniforme":
                        pos = rng.uniform( 0, 1, size = ( n, d ) )
                    else:
                        pos = rng.uniform( 0, 1, size = ( 10, d ) )[ rng.integers( 0, 10, n ) ]
                    bsp = AaBsp( pos, max_seeds_per_leaf = leaf )
                    want = AaBsp.max_depth_for( n, leaf )
                    # ATTEINT quand les germes sont distincts, seulement majoré quand ils ne le
                    # sont pas : des positions confondues ferment des feuilles tôt, donc l'arbre
                    # rétrécit -- ce qui est le bon sens pour une capacité.
                    if kind == "uniforme":
                        assert bsp.max_depth == want, ( n, leaf, d, kind, bsp.max_depth, want )
                    else:
                        assert bsp.max_depth <= want, ( n, leaf, d, kind, bsp.max_depth, want )
                    nb = int( numpy.asarray( bsp.node_left ).size )
                    assert nb <= AaBsp.max_nb_nodes_for( n, leaf ), ( n, leaf, d, kind, nb )


if test( "a_bsp_node_contains_its_subtree" ):
    # la boîte d'un nœud doit contenir TOUS les germes du sous-arbre, pas seulement ceux de ses
    # feuilles directes : c'est elle que la marche teste avant de refuser de descendre.
    rng = numpy.random.default_rng( 411 )
    pos = rng.normal( size = ( 300, 3 ) )
    bsp = AaBsp( pos, max_seeds_per_leaf = 7 )

    left = numpy.asarray( bsp.node_left ).reshape( -1 )
    right = numpy.asarray( bsp.node_right ).reshape( -1 )
    beg = numpy.asarray( bsp.node_begin ).reshape( -1 )
    end = numpy.asarray( bsp.node_end ).reshape( -1 )
    lo = numpy.asarray( bsp.node_lo )
    hi = numpy.asarray( bsp.node_hi )
    order = numpy.asarray( bsp.seed_indices ).reshape( -1 )

    def seeds_of( k ):
        if left[ k ] < 0:
            return list( order[ int( beg[ k ] ) : int( end[ k ] ) ] )
        return seeds_of( int( left[ k ] ) ) + seeds_of( int( right[ k ] ) )

    total = 0
    for k in range( len( left ) ):
        sub = pos[ seeds_of( k ) ]
        assert ( sub >= lo[ k ] - 1e-12 ).all() and ( sub <= hi[ k ] + 1e-12 ).all(), k
        total += 1
    assert total == len( left )
    assert len( seeds_of( 0 ) ) == len( pos )          # la racine, c'est tout le monde


if test( "the_weight_majorant_majorates" ):
    # `w_i <= a . y_i + b` pour tout germe du sous-arbre : sans ça la marche élaguerait un nœud
    # qui contenait pourtant une coupe. Testé sur des poids TENDANCIELS (le régime où l'affine
    # sert) autant que sur du bruit pur (celui où il doit se taire).
    rng = numpy.random.default_rng( 420 )
    pos = rng.uniform( 0, 1, size = ( 400, 2 ) )
    for name, w in ( ( "affine", 0.5 * pos[ :, 0 ] - 0.3 * pos[ :, 1 ] ),
                     ( "bruit", rng.normal( size = 400 ) ),
                     ( "affine bruite", 0.5 * pos[ :, 0 ] + 0.02 * rng.normal( size = 400 ) ) ):
        bsp = AaBsp( pos, w, max_seeds_per_leaf = 12 )
        left = numpy.asarray( bsp.node_left ).reshape( -1 )
        right = numpy.asarray( bsp.node_right ).reshape( -1 )
        beg = numpy.asarray( bsp.node_begin ).reshape( -1 )
        end = numpy.asarray( bsp.node_end ).reshape( -1 )
        wa = numpy.asarray( bsp.node_wa )
        wb = numpy.asarray( bsp.node_wb ).reshape( -1 )
        order = numpy.asarray( bsp.seed_indices ).reshape( -1 )

        def seeds_of( k ):
            if left[ k ] < 0:
                return list( order[ int( beg[ k ] ) : int( end[ k ] ) ] )
            return seeds_of( int( left[ k ] ) ) + seeds_of( int( right[ k ] ) )

        nb_affine = 0
        for k in range( len( left ) ):
            sub = seeds_of( k )
            slack = wa[ k ] @ pos[ sub ].T + wb[ k ] - w[ sub ]
            assert slack.min() >= 0, ( name, k, slack.min() )
            # RELEVÉ jusqu'à toucher : un majorant qui ne touche jamais est un majorant lâche. Ce
            # qui l'empêche de toucher exactement est la marge d'arrondi de `_weight_majorant`,
            # relative -- donc c'est à elle qu'on compare, pas à zéro.
            room = 1e-4 * ( 1 + numpy.abs( w[ sub ] ).max() )
            assert slack.min() < room, ( name, k, slack.min(), room )
            nb_affine += bool( numpy.abs( wa[ k ] ).max() > 0 )

        # le choix affine/constant est fait nœud par nœud : sur une tendance il doit être pris,
        # sur du bruit pur il doit rester marginal. « Marginal » et pas « jamais » : le seuil
        # corrige le resserrement que le hasard donne EN MOYENNE, pas celui d'un tirage
        # particulier, et un nœud qui passe quand même reste parfaitement valide -- c'est le
        # bloc au-dessus qui le dit.
        frac = nb_affine / len( left )
        if name == "bruit":
            assert frac < 0.2, ( name, frac )
        else:
            assert frac > 0.5, ( name, frac )


if test( "an_accelerator_for_other_seeds_is_refused" ):
    # un accélérateur INDEXE les germes : celui d'un autre nuage désignerait autre chose, et la
    # réponse serait fausse sans rien qui le signale. Le cas courant est un `AaBsp` gardé d'un pas
    # précédent où le nuage a changé de taille -- c'est celui-là que le compte attrape.
    rng = numpy.random.default_rng( 425 )
    pos = rng.uniform( 0, 1, size = ( 20, 2 ) )
    bsp = AaBsp( pos )
    try:
        PowerDiagram( pos[ :15 ], box = ( [ 0, 0 ], [ 1, 1 ] ), accelerator = bsp ).measures
        raise AssertionError( "expected a ValueError" )
    except ValueError as e:
        assert "20 seeds" in str( e ) and "15" in str( e ), str( e )


if test( "the_accelerator_changes_nothing_to_the_measures" ):
    # le test central, en balayant les régimes : dimension, poids, et une distribution TRÈS
    # inhomogène (des amas), qui est le cas où un arbre médian et un arbre géométrique divergent.
    cases = []
    for d in ( 2, 3 ):
        rng = numpy.random.default_rng( 430 + d )
        cases.append( ( f"uniforme {d}D", rng.uniform( 0.02, 0.98, size = ( 120, d ) ), None ) )
        pos = rng.uniform( 0.02, 0.98, size = ( 120, d ) )
        cases.append( ( f"poids {d}D", pos, rng.uniform( -0.02, 0.02, 120 ) ) )
        # des amas : dix paquets serrés, donc des boîtes très inégales
        centres = rng.uniform( 0.15, 0.85, size = ( 10, d ) )
        clust = ( centres[ rng.integers( 0, 10, 150 ) ] + 0.02 * rng.normal( size = ( 150, d ) ) ).clip( 0.01, 0.99 )
        cases.append( ( f"amas {d}D", clust, None ) )
        cases.append( ( f"amas+poids {d}D", clust, 0.01 * rng.normal( size = 150 ) ) )

    for name, pos, w in cases:
        d = pos.shape[ 1 ]
        box = ( [ 0 ] * d, [ 1 ] * d )
        plain, fast = _both_ways( pos, w, box )
        a, b = _measures( plain ), _measures( fast )
        assert numpy.allclose( a, b, rtol = 0, atol = 1e-9 ), ( name, numpy.abs( a - b ).max() )


if test( "the_accelerator_keeps_every_facet" ):
    # la même chose, mais au niveau des VOISINS et non des volumes : c'est ce qui dit qu'aucune
    # coupe utile n'a été tue, y compris celles qui n'enlèvent presque rien.
    for d in ( 2, 3 ):
        rng = numpy.random.default_rng( 440 + d )
        pos = rng.uniform( 0.02, 0.98, size = ( 60, d ) )
        w = rng.uniform( -0.02, 0.02, 60 )
        for weights in ( None, w ):
            plain, fast = _both_ways( pos, weights, ( [ 0 ] * d, [ 1 ] * d ) )
            assert _cut_sets( plain ) == _cut_sets( fast ), ( d, weights is not None )


if test( "the_accelerator_survives_the_degenerate_layouts" ):
    # les cas où l'arbre lui-même est bizarre : tous les germes au même endroit (une feuille qu'on
    # ne peut pas couper), un seul germe, des germes alignés, une feuille d'UN seul germe (donc
    # l'arbre le plus profond possible, et la pile la plus sollicitée).
    box2 = ( [ 0, 0 ], [ 1, 1 ] )

    same = numpy.full( ( 12, 2 ), 0.5 )
    plain, fast = _both_ways( same, None, box2 )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )

    one = numpy.array( [ [ 0.3, 0.7 ] ] )
    plain, fast = _both_ways( one, None, box2 )
    assert numpy.allclose( _measures( fast ), 1.0, atol = 1e-9 )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )

    line = numpy.stack( [ numpy.linspace( 0.05, 0.95, 40 ), numpy.full( 40, 0.5 ) ], axis = 1 )
    plain, fast = _both_ways( line, None, box2 )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )

    rng = numpy.random.default_rng( 451 )
    pos = rng.uniform( 0.02, 0.98, size = ( 90, 2 ) )
    plain, fast = _both_ways( pos, None, box2, max_seeds_per_leaf = 1 )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )
    # mêmes germes, grain opposé : une seule feuille, donc l'arbre est réduit à sa racine
    plain, fast = _both_ways( pos, None, box2, max_seeds_per_leaf = 10 ** 6 )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )


if test( "an_unbounded_diagram_falls_back_to_the_full_sweep" ):
    # sans domaine, une cellule n'est pas l'enveloppe de ses sommets tant qu'elle n'est pas
    # bornée, donc il n'y a rien contre quoi élaguer : la marche doit tout visiter, et le
    # résultat rester le bon. C'est le seul régime où l'accélérateur n'accélère pas, et c'est
    # exprès (voir `cell_may_be_cut`).
    rng = numpy.random.default_rng( 460 )
    pos = rng.uniform( 0, 1, size = ( 30, 2 ) )
    plain = PowerDiagram( pos )
    fast = PowerDiagram( pos, accelerator = AaBsp.of( plain ) )
    a, b = _measures( plain ), _measures( fast )
    # `Cell::measure` dit « infinie » par `TF::max`, comme partout ailleurs
    inf_a, inf_b = a > 1e300, b > 1e300
    assert ( inf_a == inf_b ).all() and inf_a.sum() > 0
    assert numpy.allclose( a[ ~inf_a ], b[ ~inf_b ], atol = 1e-9 )
    assert _cut_sets( plain ) == _cut_sets( fast )


if test( "a_domain_that_is_not_a_box_is_accelerated_too" ):
    # l'élagage ne connaît que la cellule et les boîtes de l'arbre : la FORME du domaine ne lui
    # dit rien, il suffit qu'il borne. Un simplexe le vérifie.
    rng = numpy.random.default_rng( 470 )
    pos = rng.uniform( 0.05, 0.4, size = ( 50, 2 ) )
    bnd = ( numpy.array( [ [ -1.0, 0 ], [ 0, -1.0 ], [ 1.0, 1.0 ] ] ), numpy.array( [ 0.0, 0.0, 1.0 ] ) )
    plain = PowerDiagram( pos, boundaries = bnd )
    fast = PowerDiagram( pos, boundaries = bnd, accelerator = AaBsp.of( plain ) )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )
    assert abs( float( _measures( fast ).sum() ) - 0.5 ) < 1e-9


if test( "an_accelerator_built_on_other_weights_is_still_right" ):
    # le majorant n'est qu'une BORNE : le construire sur d'autres poids que ceux du diagramme
    # élague moins bien mais ne peut pas mentir... à condition qu'il majore encore. Ici on
    # accélère un diagramme AVEC poids par un arbre construit SANS -- donc un majorant nul, qui
    # ne majore plus rien -- et par un arbre construit sur des poids plus grands, qui majore.
    rng = numpy.random.default_rng( 480 )
    pos = rng.uniform( 0.02, 0.98, size = ( 70, 2 ) )
    w = rng.uniform( -0.03, 0.03, 70 )
    box = ( [ 0, 0 ], [ 1, 1 ] )
    ref = _measures( PowerDiagram( pos, weights = w, box = box ) )

    generous = PowerDiagram( pos, weights = w, box = box,
                             accelerator = AaBsp( pos, w + 0.05 ) )     # majore : correct
    assert numpy.allclose( ref, _measures( generous ), atol = 1e-9 )

    # et l'arbre sans poids, lui, n'est PAS un majorant valide : on ne le teste que pour dire
    # qu'il ne l'est pas -- c'est pour cela que `AaBsp.of` existe.
    assert AaBsp( pos ).node_wa.is_undefined


if test( "the_accelerator_changes_nothing_to_the_derivatives" ):
    # les dérivées passent par les coupes SURVIVANTES, donc elles ne peuvent différer que si une
    # coupe a été perdue. La jacobienne complète, terme à terme, le dit sans détour -- et c'est le
    # test qui protège l'arbre contre une remontée subtilement fausse plutôt que juste un volume
    # qui tombe juste.
    def run():
        rng = numpy.random.default_rng( 490 )
        n = 12
        pos = rng.uniform( 0.1, 0.9, size = ( n, 2 ) )
        w = rng.uniform( -0.02, 0.02, n )
        box = ( [ 0, 0 ], [ 1, 1 ] )
        bsp = AaBsp( pos, w, max_seeds_per_leaf = 3 )

        for name, f_plain, f_fast, x in (
            ( "positions",
              lambda a: PowerDiagram( a, weights = w, box = box ).measures,
              lambda a: PowerDiagram( a, weights = w, box = box, accelerator = bsp ).measures,
              pos ),
            ( "poids",
              lambda a: PowerDiagram( pos, weights = a, box = box ).measures,
              lambda a: PowerDiagram( pos, weights = a, box = box, accelerator = bsp ).measures,
              w ),
        ):
            ja = _jacobian( f_plain, x, n )
            jb = _jacobian( f_fast, x, n )
            assert numpy.abs( ja - jb ).max() < 1e-9, ( name, numpy.abs( ja - jb ).max() )

        # et l'adjoint contre la différence finie, sur le chemin accéléré cette fois : la
        # comparaison ci-dessus dirait « pareil » si les deux étaient faux de la même façon.
        check_grad( lambda a, b: PowerDiagram( a, weights = b, box = box, accelerator = bsp ).measures,
                    pos, w )
    _in_fp64( run )


if test( "an_accelerated_diagram_draws_the_same_cells" ):
    # le chemin d'affichage (`cells`) a son propre appel et son propre scratch par work-item :
    # il faut donc lui aussi le vérifier, et pas seulement `measures`.
    rng = numpy.random.default_rng( 495 )
    pos = rng.uniform( 0.05, 0.95, size = ( 40, 2 ) )
    w = rng.uniform( -0.02, 0.02, 40 )
    plain, fast = _both_ways( pos, w, ( [ 0, 0 ], [ 1, 1 ] ) )
    a = numpy.asarray( plain.cells.measure.tensor ).reshape( -1 )
    b = numpy.asarray( fast.cells.measure.tensor ).reshape( -1 )
    assert numpy.allclose( a, b, atol = 1e-9 )
    assert numpy.allclose( b, _measures( fast ), atol = 1e-9 )


# -- ce que ça COÛTE ------------------------------------------------------------------------------
#   ./run bench test_PowerDiagram --nb-points=2000,32000,128000

if p := bench( "pd accelerated",
               nb_points = Param( 32000, help = "nombre de germes" ),
               nb_dims   = Param( 2, help = "dimension" ),
               leaf_size = Param( 30, help = "germes par feuille du BSP" ),
               weights   = Param( 0, help = "1 pour un diagramme de puissance" ),
               seed      = Param( 0, help = "graine du tirage" ) ):
    # Une vraie boucle, et le MINIMUM : les noms d'axes de batch sont empruntés à une réserve
    # (`loom.tensor.batch`), donc deux appels identiques produisent la même source C++ et le second
    # touche le cache de compilation. Le premier appel de chaque variante compile encore -- d'où
    # le tour de chauffe hors chronomètre.
    import time

    d, n = p.nb_dims, p.nb_points
    rng = numpy.random.default_rng( p.seed )
    pos = rng.uniform( 0.01, 0.99, size = ( n, d ) )
    # les poids À L'ÉCHELLE : un plan est décalé de `dw / ( 2 |d1 - d0| )`, donc pour que le
    # décalage soit une fraction de l'espacement `h` il faut `dw ~ h²`. Des poids « au hasard entre
    # -0.002 et 0.002 » videraient presque toutes les cellules à `n` grand -- et un balayage qui
    # sort tout de suite ne mesure plus rien.
    h = n ** ( -1.0 / d )
    w = rng.uniform( -0.3, 0.3, n ) * h * h if p.weights else None
    box = ( [ 0 ] * d, [ 1 ] * d )

    t = time.perf_counter()
    bsp = AaBsp( pos, w, max_seeds_per_leaf = p.leaf_size )
    t_build = time.perf_counter() - t

    def run( acc, reps = 3 ):
        def once():
            t = time.perf_counter()
            m = numpy.asarray( PowerDiagram( pos, weights = w, box = box, accelerator = acc ).measures.tensor )
            return time.perf_counter() - t, m.reshape( -1 )
        once()                                          # chauffe : c'est celui-là qui compile
        best, m = once()
        for _ in range( reps - 1 ):
            best = min( best, once()[ 0 ] )
        return best, m

    t_plain, m_plain = run( None )
    t_acc, m_acc = run( bsp )

    p.results[ "t_plain" ] = t_plain
    p.results[ "t_accelerated" ] = t_acc
    p.results[ "t_bsp_build" ] = t_build
    p.results[ "speedup" ] = t_plain / t_acc
    p.results[ "max_abs_diff" ] = float( numpy.abs( m_plain - m_acc ).max() )
    p.results[ "bsp_depth" ] = bsp.max_depth
    p.results[ "bsp_leaves" ] = bsp.nb_leaves
    print( f"  {n} germes en {d}D : {t_plain:.2f} s -> {t_acc:.2f} s"
           f" (x{t_plain / t_acc:.0f}), arbre bâti en {t_build * 1e3:.0f} ms"
           f", écart max {numpy.abs( m_plain - m_acc ).max():.1e}" )


# -- ce qu'on REGARDE ----------------------------------------------------------------------------
# Un diagramme se juge à l'oeil bien plus qu'à un nombre. Un `experiment` par régime, chacun
# écrivant la page HTML et le VTK de ParaView -- voir `Cell.add_to_viz` pour ce qu'une cellule non
# bornée montre d'elle-même (les plans factices ne sont pas envoyés, les arêtes tronquées sont en
# pointillés, celles qui ne sont que la fermeture factice ne sont pas tracées).
#
#   ./run experiment test_PowerDiagram                      # toutes
#   ./run experiment "test_PowerDiagram::vor 2D open"       # une seule
#   ./run experiment test_PowerDiagram --nb-points=20,200   # un balayage

def _write_both( p, viz, stem ):
    viz.write_html( p.out_dir / f"{ stem }.html" )
    v = viz.write_vtk( p.out_dir / f"{ stem }.vtu" )
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


if p := experiment( "pd 2D weights",
                    nb_points = Param( 30, help = "nombre de germes" ),
                    spread    = Param( 0.02, help = "amplitude des poids" ),
                    seed      = Param( 3, help = "graine du tirage" ) ):
    # ce que les poids FONT, à germes fixés. Deux images : la première sans poids (le Voronoï), la
    # seconde avec -- les germes n'ont pas bougé d'un pouce, seuls les plans ont glissé. À forte
    # amplitude (`--spread=0.1`) on voit apparaître ce qu'un Voronoï ne produit jamais : des germes
    # à l'extérieur de leur propre cellule, et des germes qui n'en ont plus du tout.
    pos = _seeds( 2, p.nb_points, p.seed )
    w = numpy.random.default_rng( p.seed + 1 ).uniform( -p.spread, p.spread, p.nb_points )
    box = ( [ 0, 0 ], [ 1, 1 ] )

    viz = Visualizer( title = f"Voronoï -> puissance, { p.nb_points } germes", frame_axis = "poids" )
    for k, weights in enumerate( ( None, 0.2 * w, 0.4 * w, 0.6 * w, 0.8 * w, w ) ):
        if k:
            viz.new_frame( k )
        PowerDiagram( pos, weights = weights, box = box ).add_to_viz( viz )
        viz.add_points( pos, color = "#ffffff" )
    _write_both( p, viz, "pd_2d_weights" )
