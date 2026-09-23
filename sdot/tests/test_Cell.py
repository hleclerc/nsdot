import numpy

from loom.testing import check_grad, test, experiment, Param

from loom import driver, new_batch_axis
from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype

# La géométrie se coupe en FP32 par défaut ( voir `Cell.py` ) : les tests de JUSTESSE tournent avec
# le noyau en double, et le noyau `float` a ses tests à lui ( « the_float_kernel_* » ).
set_kernel_dtype( "FP64" )


if test( "basic" ):
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 2, 0 ], [ 0, 1 ] ] )
    assert c.measure == 2

    # 2D : `vertex_positions` est déjà le polygone en ordre cyclique, donc pas de treillis --
    # `Cell_2` n'a même pas ces attributs
    from sdot import Cell_2
    assert isinstance( c, Cell_2 ) and not hasattr( c, "vertex_cuts" )
    assert numpy.allclose( numpy.asarray( c.vertex_positions ), [ [ 0, 0 ], [ 2, 0 ], [ 2, 1 ], [ 0, 1 ] ] )

if test( "basic_1D" ):
    # 1D : la cellule est un SEGMENT ( `Cell_1` ) -- deux bouts, une coupe par bout
    from sdot import Cell_1
    c = Cell.make_hypercube( 1, [ 0.5 ], [ [ 2 ] ] )
    assert isinstance( c, Cell_1 )
    assert c.measure == 2
    c.cut( [ 1 ], 1.5 )
    assert abs( float( c.measure ) - 1 ) < 1e-12
    assert numpy.allclose( c.vertices(), [ [ 0.5 ], [ 1.5 ] ] )
    c.cut( [ -1 ], -0.7 )
    assert abs( float( c.measure ) - 0.8 ) < 1e-12
    c.cut( [ 1 ], 0.6 )                                     # tout est dehors
    assert int( c.nb_vertices.value ) == 0 and float( c.measure ) == 0

if test( "unbounded_1D" ):
    c = Cell.make_unbounded( 1 )
    assert float( c.measure ) > 1e300 and not c.is_bounded
    c.cut( [ 1 ], 3.0 )
    assert float( c.measure ) > 1e300 and not c.is_bounded      # une demi-droite
    c.cut( [ -1 ], 1.0 )
    assert c.is_bounded and abs( float( c.measure ) - 4 ) < 1e-12

if test( "batch" ):
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 2, 0 ], [ 0, 1 ] ], batch_axes = [ new_batch_axis( 2 ) ] )
    assert tuple( c.measure.tensor ) == ( 2, 2 )
    assert c.nb_items == 2 and numpy.allclose( c.vertices( 1 ), c.vertices( 0 ) )

def _cell_with_coords( d, coords, cut_ids, vertex_cuts = None, vertex_nbrs = None ):
    """Une cellule posée DIRECTEMENT par ses tenseurs -- ce qui rend la mesure dérivable par
    rapport aux coordonnées, `coords` pouvant être un traceur."""
    c = Cell( d, init_as_unbounded = False )
    c.vertex_positions = coords
    c.cut_ids = cut_ids
    if vertex_cuts is not None:
        c.vertex_cuts = vertex_cuts
        c.vertex_nbrs = vertex_nbrs
    return c

if test( "grad_measure" ):
    # l'adjoint du lacet, vérifié par différence finie sur les coordonnées elles-mêmes
    c = Cell.make_hypercube( 2, [ 0.3, -0.2 ], [ [ 2.0, 0.1 ], [ -0.3, 1.0 ] ] )
    c.cut( [ 1, 1 ], 1.5 )
    coords = driver.array( numpy.asarray( c.vertex_positions.tensor ) )
    ids = numpy.asarray( c.cut_ids.tensor )
    check_grad( lambda x: _cell_with_coords( 2, x, ids ).measure, coords )

if p := test( "cut" ):
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 0 ], 0.1 )

    # une page écrite au passage : ce test ne juge pas l'IMAGE ( voir les `experiment` en fin de
    # fichier pour ça ), il vérifie seulement que l'écriture ne casse pas.
    v = Visualizer()
    c.add_to_viz( v )
    v.write_html( p.out_dir / "cut.html" )

    assert c.measure == 0.1

if test( "cut_keeps_the_edge_to_cut_correspondence" ):
    # L'invariant 2D dont vit `Local2` : la coupe i porte l'arête [ v_i, v_i+1 ], donc
    # `nb_cuts == nb_vertices`. On coupe un coin du carré unité et on le vérifie arête par arête --
    # les DEUX extrémités de l'arête i doivent être sur le plan i, relu sur la géométrie.
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 1 ], 1.5 )

    n = int( c.nb_vertices.value )
    assert n == 5 and int( c.nb_cuts.value ) == n
    assert abs( float( c.measure ) - 0.875 ) < 1e-12      # 1 - le triangle de côté 1/2

    vp = c.vertices()
    cd, co = c.cut_planes
    for i in range( n ):
        j = ( i + 1 ) % n
        assert abs( cd[ i ] @ vp[ i ] - co[ i ] ) < 1e-12
        assert abs( cd[ i ] @ vp[ j ] - co[ i ] ) < 1e-12

    # la coupe demandée se retrouve, normalisée, parmi les plans relus
    assert any( numpy.allclose( cd[ i ], numpy.array( [ 1, 1 ] ) / numpy.sqrt( 2 ) ) and abs( co[ i ] - 1.5 / numpy.sqrt( 2 ) ) < 1e-12 for i in range( n ) )

if test( "cut_direction_is_not_normalized" ):
    # `offset` est le produit scalaire tel quel, donc ( 2n, 2o ) est le MÊME demi-espace que
    # ( n, o ) : une direction trois fois plus longue avec un offset trois fois plus grand doit
    # rendre exactement la même cellule.
    a = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    b = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    a.cut( [ 1, 1 ], 1.5 )
    b.cut( [ 3, 3 ], 4.5 )

    assert int( a.nb_vertices.value ) == int( b.nb_vertices.value )
    assert numpy.allclose( a.vertices(), b.vertices() )
    assert abs( float( a.measure ) - float( b.measure ) ) < 1e-12

if test( "cut_degenerate" ):
    # deux dégénérescences. Une coupe qui ne coupe rien ne doit RIEN ajouter ; une coupe qui exclut
    # tout doit vider la cellule proprement ( 0 sommet, 0 coupe, mesure nulle ).
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )

    c.cut( [ 1, 0 ], 5 )
    assert int( c.nb_vertices.value ) == 4 and int( c.nb_cuts.value ) == 4
    assert abs( float( c.measure ) - 1 ) < 1e-12

    c.cut( [ 1, 0 ], -1 )
    assert int( c.nb_vertices.value ) == 0 and int( c.nb_cuts.value ) == 0
    assert float( c.measure ) == 0

if test( "cut_capacity_is_exact" ):
    # La place qu'une coupe demande est BORNÉE d'avance ( `Cell_2._cut_capacities` : un sommet de
    # plus au plus ), donc un `cut` n'a jamais de second tour à faire : la capacité allouée suit le
    # compte d'un cran par coupe, et une coupe qui n'ajoute rien ne fait pas grossir non plus.
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    for k in range( 12 ):                                   # un dodécagone inscrit : 12 sommets
        t = 2 * numpy.pi * ( k + 0.5 ) / 12
        n = [ numpy.cos( t ), numpy.sin( t ) ]
        c.cut( n, float( numpy.dot( n, [ 0.5, 0.5 ] ) + 0.45 * numpy.cos( numpy.pi / 12 ) ) )
        assert c.nb_vertices.allocated_capacity() == 4 + k + 1
    assert int( c.nb_vertices.value ) == 12
    exp = 12 * ( 0.45 * numpy.cos( numpy.pi / 12 ) ) ** 2 * numpy.tan( numpy.pi / 12 )
    assert abs( float( c.measure ) - exp ) < 1e-12

if test( "cut_tangent" ):
    # Les deux tangences, celles où un sommet tombe sur le plan de coupe. Le clip ne pose qu'une
    # question, `s > 0`, et un sommet à l'epsilon près du plan répond comme un sommet dedans.
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 0 ], 1 )
    assert int( c.nb_vertices.value ) == 4 and int( c.nb_cuts.value ) == 4
    assert abs( float( c.measure ) - 1 ) < 1e-12

    # Tangente à un SOMMET : trois sommets dehors, les deux arêtes vers le coin ( 0, 0 ) donnent
    # chacune un croisement -- qui tombe sur ce coin. Aire nulle, position juste.
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 1 ], 0 )
    vp = c.vertices()
    assert int( c.nb_vertices.value ) == int( c.nb_cuts.value )     # l'invariant 2D tient
    assert numpy.allclose( vp, 0 )                                  # toutes au même point
    assert float( c.measure ) == 0

def _half_planes_of( *corners ):
    """Les demi-espaces `n . x <= o` d'un polygone convexe donné par ses sommets, en ordre CCW."""
    res = []
    for i in range( len( corners ) ):
        p, q = numpy.asarray( corners[ i ], float ), numpy.asarray( corners[ ( i + 1 ) % len( corners ) ], float )
        d = q - p
        n = numpy.array( [ d[ 1 ], - d[ 0 ] ] )
        res.append( ( list( n ), float( n @ p ) ) )
    return res


if test( "cut_unbounded_becomes_bounded" ):
    # La cellule non bornée est un SIMPLEXE FACTICE ( `init_as_unbounded` ) : ses plans portent des
    # offsets inventés, et une coupe les classerait selon l'échelle arbitraire de ce simplexe.
    # `cut` les repousse d'abord ( `Local2::grow_for` ) jusqu'à ce que le classement soit celui
    # qu'il serait à l'infini. Les plans factices sont mangés un à un et il reste le carré unité.
    c = Cell.make_unbounded( 2 )
    assert not c.is_bounded

    for n, o in ( ( [ -1, 0 ], 0 ), ( [ 1, 0 ], 1 ), ( [ 0, -1 ], 0 ) ):
        c.cut( n, o )
        assert not c.is_bounded                     # une bande, un demi-plan : infinis
        assert float( c.measure ) > 1e300

    c.cut( [ 0, 1 ], 1 )    # la quatrième referme la cellule
    assert c.is_bounded
    assert int( c.nb_vertices.value ) == 4
    assert float( c.measure ) == 1

if test( "cut_unbounded_triangle" ):
    # hors des axes et avec un ordre de coupes quelconque : trois demi-plans découpent leur
    # triangle dans le plan entier. La poussée doit trouver la BONNE configuration.
    corners = ( ( 0, 0 ), ( 4, 1 ), ( 1, 3 ) )
    exact = 5.5

    for order in ( ( 0, 1, 2 ), ( 2, 0, 1 ), ( 1, 2, 0 ) ):
        c = Cell.make_unbounded( 2 )
        hp = _half_planes_of( *corners )
        for i in order:
            c.cut( *hp[ i ] )

        assert c.is_bounded
        assert int( c.nb_vertices.value ) == 3
        assert abs( float( c.measure ) - exact ) < 1e-12

if test( "cut_unbounded_far_from_the_origin" ):
    # Le simplexe factice est bâti à l'ORIGINE et à l'échelle 1 : pour une cellule lointaine, il
    # faut le pousser jusque-là, et le résultat est juste à la précision de CES coordonnées-là.
    shift = numpy.array( [ 1000.0, -500.0 ] )
    corners = [ tuple( numpy.array( p, float ) + shift ) for p in ( ( 0, 0 ), ( 4, 1 ), ( 1, 3 ) ) ]

    c = Cell.make_unbounded( 2 )
    for n, o in _half_planes_of( *corners ):
        c.cut( n, o )

    assert c.is_bounded
    assert int( c.nb_vertices.value ) == 3
    assert abs( float( c.measure ) - 5.5 ) < 1e-8

if test( "cut_batched" ):
    # tout un batch se coupe d'un seul appel, un item par work-item, chacun sur sa pile
    nb_items = 64
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ], batch_axes = [ new_batch_axis( nb_items ) ] )
    c.cut( [ 1, 1 ], 1.5 )

    m = numpy.asarray( c.measure.tensor )
    assert m.shape == ( nb_items, )
    assert numpy.allclose( m, 0.875 )

if test( "the_float_kernel_cuts_the_same_cells" ):
    # le noyau par défaut est en `float32` : même combinatoire, géométrie à 1e-6 près
    corners = ( ( 0.1, 0.2 ), ( 0.9, 0.15 ), ( 0.8, 0.9 ), ( 0.2, 0.7 ) )
    for kd in ( "FP32", None ):
        a = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ], kernel_dtype = kd )
        b = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ], kernel_dtype = "FP64" )
        for n, o in _half_planes_of( *corners ):
            a.cut( n, o )
            b.cut( n, o )
        assert int( a.nb_vertices.value ) == int( b.nb_vertices.value ) == 4
        assert numpy.allclose( a.vertices(), b.vertices(), atol = 1e-6 )
        assert abs( float( a.measure ) - float( b.measure ) ) < 1e-6
        # ... et la mesure sort dans le flottant de l'appelant, pas dans celui du noyau
        assert numpy.asarray( a.measure.tensor ).dtype == numpy.float64


# -- l'oracle des tests d > 2 -----------------------------------------------------------------
# Au-delà de la 2D, la cellule n'est plus décrite par ses seuls sommets : il y a un TREILLIS DE
# FACES à vérifier, et le comparer à la main ne tiendrait pas. On le confronte donc à une
# référence indépendante, construite par force brute depuis la seule H-représentation -- résoudre
# tous les d-uplets de plans et garder les solutions admissibles. C'est O( nb_plans^d ), sans
# rapport avec l'algorithme testé, ce qui est précisément ce qu'on veut d'un oracle.

def _reference_cell( planes, tol = 1e-9 ):
    """V-représentation + « sur quels plans » pour chaque sommet, depuis la H-représentation."""
    from itertools import combinations

    # un plan coupé deux fois n'est ajouté qu'une fois : la seconde passe ne trouve rien de
    # strictement dehors et ne fait rien. La référence se dédoublonne donc pareil.
    seen, uniq = set(), []
    for n, o in planes:
        k = tuple( numpy.round( numpy.append( numpy.asarray( n, float ), o ), 12 ) + 0.0 )
        if k not in seen:
            seen.add( k )
            uniq.append( ( numpy.asarray( n, float ), float( o ) ) )

    dirs = numpy.array( [ p[ 0 ] for p in uniq ] )
    offs = numpy.array( [ p[ 1 ] for p in uniq ] )
    d = dirs.shape[ 1 ]

    verts, on_planes = [], []
    for combo in combinations( range( len( offs ) ), d ):
        rows = list( combo )
        if abs( numpy.linalg.det( dirs[ rows ] ) ) < 1e-10:
            continue
        x = numpy.linalg.solve( dirs[ rows ], offs[ rows ] )
        if numpy.any( dirs @ x - offs > tol * max( 1.0, numpy.abs( x ).max() ) ):
            continue
        if any( numpy.allclose( x, v, atol = 1e-7 ) for v in verts ):
            continue
        verts.append( x )
        on_planes.append( frozenset( numpy.nonzero( numpy.abs( dirs @ x - offs ) <= 1e-7 )[ 0 ].tolist() ) )
    return dirs, offs, verts, on_planes


def alive_cuts( vi ):
    """les coupes qu'un sommet porte encore, triées"""
    return sorted( set( int( r ) for row in vi for r in row ) )


def _check_against_reference( c, planes, name, exact = True ):
    """La cellule EST l'intersection de `planes` : sommets, coupes et arêtes, tous les trois, lus
    sur les tenseurs DÉRIVÉS ( `vertex_positions`, `vertex_cut_indices`, `edges`, `cut_planes` ).

    `exact = False` quand le plan de coupe passe par des sommets existants : le clip ne teste
    jamais `s == 0`, donc il produit alors des sommets CONFONDUS à ~1e-16 et les arêtes de longueur
    nulle qui les relient. On vérifie toujours la géométrie mais plus l'absence de doublons.
    """
    nv, nc = int( c.nb_vertices.value ), int( c.nb_cuts.value )
    vp = c.vertices()
    vi = c.vertex_cut_indices
    ev = c.edges
    cd, co = c.cut_planes
    ci = numpy.asarray( c.cut_ids.tensor )
    d = vp.shape[ 1 ]

    dirs, offs, ref, ref_on = _reference_cell( planes )
    rnd = lambda v: tuple( numpy.round( v, 7 ) + 0.0 )

    # 1. les sommets, en tant qu'ENSEMBLE de points
    where = [ next( ( i for i, r in enumerate( ref ) if numpy.allclose( p, r, atol = 1e-7 ) ), None )
              for p in vp ]
    assert None not in where, f"{ name } : sommet hors de la référence"
    assert set( where ) == set( range( len( ref ) ) ), f"{ name } : sommet de la référence manquant"
    if exact:
        assert sorted( map( rnd, vp ) ) == sorted( map( rnd, ref ) ), f"{ name } : sommets"

    # 2. le treillis est cohérent avec la géométrie : chaque sommet est bien SUR chacune de ses
    #    d coupes, et dedans pour toutes les autres ( les plans sont relus sur la géométrie, donc
    #    une coupe morte -- sans sommet -- a une direction nulle et ne contraint rien ).
    #    Une face DÉGÉNÉRÉE ( ses sommets confondus ou alignés, ce que laisse un plan passé par des
    #    sommets existants ) n'a pas de plan à relire : direction nulle, et rien à vérifier dessus.
    flat = numpy.linalg.norm( cd, axis = 1 ) == 0
    assert exact <= ( not flat[ alive_cuts( vi ) ].any() ), f"{ name } : une face vivante sans plan"
    for k, p in enumerate( vp ):
        assert list( vi[ k ] ) == sorted( vi[ k ] ), f"{ name } : liste de coupes non triée en v{ k }"
        for r in vi[ k ]:
            assert flat[ r ] or abs( cd[ r ] @ p - co[ r ] ) < 1e-9, f"{ name } : v{ k } n'est pas sur la coupe { r }"
    assert numpy.all( vp @ cd.T - co < 1e-9 ), f"{ name } : un sommet est dehors"

    # 3. les coupes VIVANTES sont exactement les plans que touche un sommet ( une coupe morte peut
    #    traîner dans la liste jusqu'à la prochaine compaction : elle n'a plus de sommet )
    used = sorted( set().union( *ref_on ) ) if ref_on else []
    alive = alive_cuts( vi )
    assert len( alive ) == len( used ), f"{ name } : { len( alive ) } coupes vivantes, attendu { len( used ) }"
    for r, u in zip( alive, used ):
        nrm = numpy.linalg.norm( dirs[ u ] )
        assert flat[ r ] or ( numpy.allclose( cd[ r ], dirs[ u ] / nrm ) and abs( co[ r ] - offs[ u ] / nrm ) < 1e-9 ), \
            f"{ name } : la coupe { r } n'est pas le plan { u }"

    # 4. les arêtes. Deux sommets sont voisins quand la PLUS PETITE FACE qui les contient ne
    #    contient qu'eux. Une arête DÉGÉNÉRÉE ( ses deux bouts sur le même point ) est mise de côté.
    def _adjacent( a, b ):
        shared = ref_on[ a ] & ref_on[ b ]
        if len( shared ) < d - 1:
            return False
        return not any( shared <= ref_on[ c ] for c in range( len( ref ) ) if c != a and c != b )

    expected = { ( a, b ) for a in range( len( ref ) ) for b in range( a + 1, len( ref ) )
                 if _adjacent( a, b ) }
    got, nb_degenerate = set(), 0
    for row in ev:
        a, b = where[ row[ 0 ] ], where[ row[ 1 ] ]
        if a == b:
            nb_degenerate += 1
            continue
        got.add( ( min( a, b ), max( a, b ) ) )
    assert got == expected, (
        f"{ name } : arêtes -- en trop { [ ( ref[ a ], ref[ b ] ) for a, b in sorted( got - expected ) ] }, "
        f"manquantes { [ ( ref[ a ], ref[ b ] ) for a, b in sorted( expected - got ) ] }" )
    if exact:
        assert nb_degenerate == 0 and len( got ) == len( ev ), f"{ name } : arête en double"


def _cube_planes( d ):
    """Les 2d demi-espaces du cube unité, dans l'ordre où `init_as_hypercube` les écrit."""
    res = []
    for b in range( d ):
        res.append( ( [ -float( i == b ) for i in range( d ) ], 0.0 ) )
        res.append( ( [ +float( i == b ) for i in range( d ) ], 1.0 ) )
    return res


def _unit_cube( d ):
    return Cell.make_hypercube( d, [ 0 ] * d, numpy.eye( d ).tolist() )


if test( "basic_3D" ):
    # au-dessus de la 2D, le treillis EXISTE : les d coupes et les d voisins de chaque sommet
    c = _unit_cube( 3 )
    assert c.vertex_cuts.is_defined
    assert c.vertex_nbrs.is_defined
    assert int( c.nb_vertices.value ) == 8
    assert int( c.nb_cuts.value ) == 6
    assert len( c.edges ) == 12
    assert len( c.faces ) == 6 and all( len( f ) == 4 for f in c.faces )
    _check_against_reference( c, _cube_planes( 3 ), "cube 3D" )

if test( "cut_3D" ):
    # une suite de coupes génériques sur le cube unité, chacune revérifiée en entier
    for extra in ( [ ( [ 1, 1, 1 ], 2.5 ) ],
                   [ ( [ 1, 1, 1 ], 2.5 ), ( [ -1, -1, -1 ], -0.4 ) ],
                   [ ( [ 1, 1, 1 ], 2.5 ), ( [ 1, -1, 0.5 ], 0.6 ), ( [ -0.3, 1, 0.7 ], 0.9 ) ],
                   [ ( [ 1, 1, 1 ], 10.0 ) ] ):          # celle-ci rate la cellule : sans effet
        c, planes = _unit_cube( 3 ), _cube_planes( 3 )
        for n, o in extra:
            c.cut( n, o )
            planes = planes + [ ( n, o ) ]
            _check_against_reference( c, planes, f"cube 3D + { extra }" )

    # un coin de cube tranché : un sommet en moins, trois en plus, une coupe en plus
    c = _unit_cube( 3 )
    c.cut( [ 1, 1, 1 ], 2.5 )
    assert ( int( c.nb_vertices.value ), len( c.edges ), int( c.nb_cuts.value ) ) == ( 10, 15, 7 )

if test( "cut_3D_random" ):
    # des plans quelconques, en série : les combinatoires qu'on n'écrirait pas à la main
    rng = numpy.random.default_rng( 12345 )
    for d in ( 3, 4 ):
        for trial in range( 12 ):
            c, planes = _unit_cube( d ), _cube_planes( d )
            for step in range( 5 ):
                n = rng.normal( size = d )
                o = float( n @ rng.uniform( 0.2, 0.8, size = d ) )   # passe par l'intérieur
                c.cut( n.tolist(), o )
                if int( c.nb_vertices.value ) == 0:
                    break
                planes = planes + [ ( n.tolist(), o ) ]
                _check_against_reference( c, planes, f"aléatoire d={ d } t={ trial } s={ step }" )

if test( "cut_through_existing_vertices" ):
    # LE cas dégénéré : des plans qui passent EXACTEMENT par des sommets. Le clip ne teste jamais
    # `s == 0` : ce qui sort a des sommets confondus et des arêtes de longueur nulle, mais c'est
    # une cellule COHÉRENTE.
    for d in ( 3, 4 ):
        for n, o in ( ( [ 1, 1 ] + [ 0 ] * ( d - 2 ), 1.0 ),      # par une 2-face entière
                      ( [ 1 ] * d, float( d - 1 ) ),              # par une arête
                      ( [ 1 ] + [ 0 ] * ( d - 1 ), 1.0 ),         # CONFONDUE avec une facette
                      ( [ -1 ] * ( d - 1 ) + [ 1 ], 0.0 ) ):      # par d sommets
            c = _unit_cube( d )
            c.cut( n, o )
            _check_against_reference( c, _cube_planes( d ) + [ ( n, o ) ],
                                      f"dégénérée d={ d } n={ n }", exact = False )

if test( "cut_through_a_vertex_2D" ):
    # le même piège en 2D, sur une cellule dont les coordonnées ne tombent pas rond
    origin, axes = [ 0.13, -0.27 ], [ [ 1.7, 0.29 ], [ -0.41, 1.13 ] ]
    o, a0, a1 = ( numpy.asarray( v, float ) for v in ( origin, axes[ 0 ], axes[ 1 ] ) )
    corners = ( o, o + a0, o + a0 + a1, o + a1 )
    for corner in corners:
        for n in ( [ 1.0, 0.37 ], [ -0.83, 1.0 ], [ 0.61, -1.19 ] ):
            offset = float( numpy.dot( n, corner ) )
            c = Cell.make_hypercube( 2, origin, axes )
            c.cut( n, offset )

            nv = int( c.nb_vertices.value )
            vp = c.vertices()
            cd, co = c.cut_planes
            assert nv == int( c.nb_cuts.value )                    # l'invariant 2D tient toujours
            assert numpy.all( vp @ cd.T - co < 1e-9 )              # tout est dans toutes les coupes

            keep = [ p for p in corners if numpy.dot( n, p ) - offset <= 0 ]
            ref = []
            for a, b in zip( corners, corners[ 1 : ] + corners[ : 1 ] ):
                sa, sb = numpy.dot( n, a ) - offset, numpy.dot( n, b ) - offset
                if sa <= 0:
                    ref.append( a )
                if ( sa > 0 ) != ( sb > 0 ):
                    ref.append( ( sb * a - sa * b ) / ( sb - sa ) )
            area = abs( sum( ref[ i ][ 0 ] * ref[ ( i + 1 ) % len( ref ) ][ 1 ]
                           - ref[ ( i + 1 ) % len( ref ) ][ 0 ] * ref[ i ][ 1 ]
                             for i in range( len( ref ) ) ) ) / 2 if len( ref ) > 2 else 0.0
            assert abs( float( c.measure ) - area ) < 1e-9, \
                f"aire fausse pour n={ n } par { corner } : { float( c.measure ) } != { area }"

if test( "cut_5D" ):
    # rien de 3D-spécifique dans le clip : la même boucle vaut en 5D
    c, planes = _unit_cube( 5 ), _cube_planes( 5 )
    for n, o in ( ( [ 1, 1, 1, 1, 1 ], 4.5 ), ( [ 1, -0.4, 0.3, 0.2, 0.1 ], 0.75 ) ):
        c.cut( n, o )
        planes = planes + [ ( n, o ) ]
        _check_against_reference( c, planes, "cube 5D" )

if test( "cut_nd_empty" ):
    c = _unit_cube( 3 )
    c.cut( [ 1, 1, 1 ], -1.0 )
    assert ( int( c.nb_vertices.value ), int( c.nb_cuts.value ) ) == ( 0, 0 )
    assert float( c.measure ) == 0

if test( "cut_nd_unbounded" ):
    # le simplexe factice repoussé jusqu'à ce que la coupe le classe comme elle le ferait à
    # l'infini -- en nD la poussée résout le même système que le sommet ( `LocalN::growth_rate` ).
    for d in ( 3, 4 ):
        c, planes = Cell.make_unbounded( d ), []
        for n, o in _cube_planes( d ):
            c.cut( n, o )
            planes.append( ( n, o ) )
            assert c.is_bounded == ( len( planes ) == 2 * d )
        _check_against_reference( c, planes, f"non bornée { d }D -> cube" )
        assert int( c.nb_vertices.value ) == 2 ** d

if test( "cut_nd_capacity_is_exact" ):
    # en 3D la borne vient d'Euler sur un polytope simple ( `V = 2 F - 4` ) : une coupe de plus, et
    # autant de sommets qu'un tel polytope peut en avoir avec ces faces-là. Jamais de second tour.
    c = _unit_cube( 3 )
    cap_c = c.nb_cuts.allocated_capacity()
    for n, o in ( ( [ 1, 1, 1 ], 2.5 ), ( [ 1, -1, 0.5 ], 0.6 ), ( [ -0.3, 1, 0.7 ], 0.9 ) ):
        c.cut( n, o )
    _check_against_reference( c, _cube_planes( 3 ) + [ ( [ 1, 1, 1 ], 2.5 ), ( [ 1, -1, 0.5 ], 0.6 ), ( [ -0.3, 1, 0.7 ], 0.9 ) ], "trois coupes" )
    assert c.nb_cuts.allocated_capacity() == cap_c + 3
    assert c.nb_vertices.allocated_capacity() == 2 * ( cap_c + 3 ) - 4

if test( "cut_nd_batched" ):
    nb_items = 64
    c = Cell.make_hypercube( 3, [ 0, 0, 0 ], numpy.eye( 3 ).tolist(),
                             batch_axes = [ new_batch_axis( nb_items ) ] )
    c.cut( [ 1, 1, 1 ], 2.5 )

    nvs = numpy.asarray( c.nb_vertices.value )
    assert nvs.shape == ( nb_items, ) and ( nvs == 10 ).all()

    assert c.nb_items == nb_items
    for b in range( nb_items ):
        assert numpy.allclose( c.vertices( b ), c.vertices( 0 ) )


# -- l'oracle de la MESURE en d > 2 --------------------------------------------------------------
# Le kernel découpe la V-représentation en simplexes et somme des déterminants. La référence prend
# l'autre bout : théorème de la divergence sur la H-représentation, `vol = ( 1/d ) somme_facettes
# h_F A_F`, chaque aire de facette obtenue en ÉLIMINANT une coordonnée puis en récursant. Aucune
# ligne de code en commun, et c'est exact.

def _reference_volume( dirs, offs ):
    from itertools import combinations

    dirs, offs = numpy.asarray( dirs, float ), numpy.asarray( offs, float )
    d = dirs.shape[ 1 ]

    # éliminer une coordonnée fait RÉAPPARAÎTRE des plans déjà là (sur la facette x = 0 d'un
    # prisme, `x + y <= 1` redevient `y <= 1`) : sans dédoublonnage la facette correspondante
    # serait comptée deux fois. Au passage on jette les plans devenus vides (`0 . x <= b`).
    seen, rows = set(), []
    for n, o in zip( dirs, offs ):
        norm = numpy.linalg.norm( n )
        if norm < 1e-12:
            continue
        key = tuple( numpy.round( numpy.append( n, o ) / norm, 9 ) + 0.0 )
        if key not in seen:
            seen.add( key )
            rows.append( ( n, o ) )
    dirs = numpy.array( [ r[ 0 ] for r in rows ] )
    offs = numpy.array( [ r[ 1 ] for r in rows ] )

    if d == 1:
        lo, hi = -numpy.inf, numpy.inf
        for n, o in zip( dirs[ :, 0 ], offs ):
            if n > 0: hi = min( hi, o / n )
            elif n < 0: lo = max( lo, o / n )
        return max( 0.0, hi - lo )

    verts = []
    for combo in combinations( range( len( offs ) ), d ):
        rows = list( combo )
        if abs( numpy.linalg.det( dirs[ rows ] ) ) < 1e-10:
            continue
        x = numpy.linalg.solve( dirs[ rows ], offs[ rows ] )
        if numpy.any( dirs @ x - offs > 1e-9 * max( 1.0, numpy.abs( x ).max() ) ):
            continue
        if not any( numpy.allclose( x, v, atol = 1e-7 ) for v in verts ):
            verts.append( x )       # un sommet dégénéré sort de plusieurs d-uplets de plans
    if len( verts ) < d + 1:
        return 0.0
    x0 = numpy.mean( verts, axis = 0 )

    total = 0.0
    for k in range( len( offs ) ):
        on = [ v for v in verts if abs( dirs[ k ] @ v - offs[ k ] ) < 1e-7 ]
        if len( on ) < d or numpy.linalg.matrix_rank( numpy.array( on ) - on[ 0 ], tol = 1e-7 ) < d - 1:
            continue                                       # plan redondant, pas une facette
        # x_j = ( o_k - somme_{i != j} n_i x_i ) / n_j, reporté dans tous les autres plans
        nk, ok = dirs[ k ], offs[ k ]
        j = int( numpy.argmax( numpy.abs( nk ) ) )
        keep = [ i for i in range( d ) if i != j ]
        others = [ i for i in range( len( offs ) ) if i != k ]
        A = dirs[ others ][ :, keep ] - numpy.outer( dirs[ others ][ :, j ] / nk[ j ], nk[ keep ] )
        b = offs[ others ] - dirs[ others ][ :, j ] * ok / nk[ j ]
        area = _reference_volume( A, b ) * numpy.linalg.norm( nk ) / abs( nk[ j ] )
        total += ( ok - nk @ x0 ) / numpy.linalg.norm( nk ) * area
    return total / d


if test( "measure_3D" ):
    c = _unit_cube( 3 )
    assert float( c.measure ) == 1

    c.cut( [ 1, 1, 1 ], 2.5 )
    assert abs( float( c.measure ) - ( 1 - 0.5 ** 3 / 6 ) ) < 1e-15

if test( "measure_5D" ):
    c = _unit_cube( 5 )
    assert float( c.measure ) == 1

    c.cut( [ 1, 1, 1, 1, 1 ], 4.5 )
    assert abs( float( c.measure ) - ( 1 - 0.5 ** 5 / 120 ) ) < 1e-15

if test( "measure_nd_vs_reference" ):
    rng = numpy.random.default_rng( 7 )
    for d in ( 3, 4 ):
        for trial in range( 8 ):
            c, planes = _unit_cube( d ), _cube_planes( d )
            for step in range( 3 ):
                n = rng.normal( size = d )
                o = float( n @ rng.uniform( 0.2, 0.8, size = d ) )
                c.cut( n.tolist(), o )
                planes = planes + [ ( n.tolist(), o ) ]
                got = float( c.measure )
                exp = _reference_volume( [ p[ 0 ] for p in planes ], [ p[ 1 ] for p in planes ] )
                assert abs( got - exp ) < 1e-10 * max( 1.0, exp ), \
                    f"d={ d } t={ trial } s={ step } : { got } != { exp }"

if test( "measure_nd_is_additive" ):
    rng = numpy.random.default_rng( 3 )
    for d in ( 3, 4 ):
        for _ in range( 6 ):
            n = rng.normal( size = d )
            o = float( n @ rng.uniform( 0.2, 0.8, size = d ) )
            a = _unit_cube( d ); a.cut( n.tolist(), o )
            b = _unit_cube( d ); b.cut( ( -n ).tolist(), -o )
            assert abs( float( a.measure ) + float( b.measure ) - 1 ) < 1e-12

if test( "measure_nd_degenerate" ):
    for d in ( 3, 4 ):
        for n, o in ( ( [ 1, 1 ] + [ 0 ] * ( d - 2 ), 1.0 ),
                      ( [ 1 ] * d, float( d - 1 ) ),
                      ( [ -1 ] * ( d - 1 ) + [ 1 ], 0.0 ) ):
            c = _unit_cube( d )
            c.cut( n, o )
            exp = _reference_volume( [ p[ 0 ] for p in _cube_planes( d ) ] + [ n ],
                                     [ p[ 1 ] for p in _cube_planes( d ) ] + [ o ] )
            assert abs( float( c.measure ) - exp ) < 1e-11, f"d={ d } n={ n }"

if test( "measure_nd_unbounded_and_empty" ):
    assert float( Cell.make_unbounded( 3 ).measure ) > 1e300
    c = _unit_cube( 3 )
    c.cut( [ 1, 1, 1 ], -1.0 )
    assert float( c.measure ) == 0

if test( "grad_measure_nd" ):
    # l'adjoint du découpage : la triangulation est COMBINATOIRE ( elle ne bouge pas sous une
    # petite perturbation ), donc le backward rejoue le même parcours et ne dérive que les
    # déterminants. Vérifié par différence finie sur les coordonnées.
    for d in ( 3, 4 ):
        c = Cell.make_hypercube( d, numpy.zeros( d ), numpy.eye( d ) + 0.07 * numpy.arange( d * d ).reshape( d, d ) / ( d * d ) )
        c.cut( numpy.linspace( 0.8, 1.3, d ).tolist(), float( d ) - 0.7 )
        coords = driver.array( numpy.asarray( c.vertex_positions.tensor ) )
        ids = numpy.asarray( c.cut_ids.tensor )
        vc, vn = numpy.asarray( c.vertex_cuts.tensor ), numpy.asarray( c.vertex_nbrs.tensor )
        check_grad( lambda x: _cell_with_coords( d, x, ids, vc, vn ).measure, coords )

if test( "measure_nd_batched" ):
    nb_items = 48
    c = Cell.make_hypercube( 3, [ 0, 0, 0 ], numpy.eye( 3 ).tolist(),
                             batch_axes = [ new_batch_axis( nb_items ) ] )
    c.cut( [ 1, 1, 1 ], 2.5 )

    m = numpy.asarray( c.measure.tensor )
    assert m.shape == ( nb_items, )
    assert numpy.allclose( m, 1 - 0.5 ** 3 / 6 )


# -- ce que l'affichage d'une cellule NON BORNÉE laisse tomber -----------------------------------
# Une image ne s'asserte pas, mais ces règles-là si : elles portent sur QUOI est envoyé au
# visualiseur, et le visualiseur sait dire ce qu'il a reçu. Voir `Cell.add_to_viz`.

if test( "viz_drops_the_infinite_planes" ):
    # les parois du simplexe factice ne sont pas des faces de la cellule -- elles n'ont ni la bonne
    # position (leur offset est une invention que les coupes repoussent) ni d'existence. Ce qui
    # part est exactement l'ensemble des demi-espaces demandés, et rien d'autre.
    asked = ( ( [ 1.0, 0.0 ], 1.0 ), ( [ 0.0, 1.0 ], 1.0 ), ( [ 0.0, -1.0 ], 0.0 ) )
    c = Cell.make_unbounded( 2 )
    for n, o in asked:
        c.cut( n, o )
    assert not c.is_bounded                                       # une demi-bande : non bornée

    v = Visualizer()
    c.add_to_viz( v )
    assert len( v.polytopes ) == 1
    dirs, offs, _, with_edges = v.polytopes[ 0 ]
    assert not with_edges          # en 2D/3D c'est la cellule qui trace ses arêtes, pas la coupe
    # les plans partent NORMALISÉS ( relus sur la géométrie ) : on compare à la même échelle
    got = sorted( tuple( numpy.round( numpy.append( d, o ), 6 ) ) for d, o in zip( dirs, offs ) )
    exp = sorted( tuple( numpy.round( numpy.append( n, o ) / numpy.linalg.norm( n ), 6 ) ) for n, o in asked )
    assert got == exp, ( got, exp )

if test( "viz_dashes_what_runs_off_and_hides_what_is_made_up" ):
    # la même demi-bande `{ x <= 1, 0 <= y <= 1 }`, qui part vers les x négatifs. Trois sortes
    # d'arêtes, et on les distingue par la GÉOMÉTRIE de ce qui arrive au visualiseur :
    #
    # - `x = 1, 0 <= y <= 1` : ses deux bouts sont de vrais sommets -> tracée pleine ;
    # - `y = 0` et `y = 1` : un vrai bout, l'autre sur une paroi factice -> pointillés, donc un
    #   paquet de petits segments (7 tirets, cf. `Visualizer.add_edges( dashed = True )`) ;
    # - la fermeture du simplexe factice -> rien du tout.
    c = Cell.make_unbounded( 2 )
    for n, o in ( ( [ 1, 0 ], 1.0 ), ( [ 0, 1 ], 1.0 ), ( [ 0, -1 ], 0.0 ) ):
        c.cut( n, o )

    v = Visualizer()
    c.add_to_viz( v )
    segs = numpy.asarray( v.positions )[ numpy.asarray( v.edges ) ]      # [ m, 2, 2 ]

    def _is( seg, a, b ):
        return ( numpy.allclose( seg, [ a, b ], atol = 1e-6 )
              or numpy.allclose( seg, [ b, a ], atol = 1e-6 ) )

    full = [ s for s in segs if _is( s, [ 1, 0 ], [ 1, 1 ] ) ]
    rest = [ s for s in segs if not _is( s, [ 1, 0 ], [ 1, 1 ] ) ]
    assert len( full ) == 1, f"l'arête réelle devrait être tracée pleine, une fois ({ len( full ) })"

    # tout le reste est un tiret, sur l'une des deux demi-droites -- rien de la fermeture factice
    # (`x` très négatif fermé par un plan oblique) ne doit être passé.
    assert len( rest ) == 2 * 7, f"{ len( rest ) } tirets, attendu 14"
    for s in rest:
        on_line = numpy.allclose( s[ :, 1 ], 0, atol = 1e-6 ) or numpy.allclose( s[ :, 1 ], 1, atol = 1e-6 )
        assert on_line and ( s[ :, 0 ] <= 1 + 1e-6 ).all(), s

if test( "viz_the_clipping_box_is_not_geometry" ):
    # Un polytope OUVERT n'a rien à montrer tel quel : c'est la boîte de la scène qui lui donne
    # des sommets (`polytope.clip_planes`, et la même chose côté page à chaque coupe). Elle est un
    # moyen de le voir, pas une partie de lui -- elle referme donc la face par où il sort du champ,
    # mais elle ne donne AUCUNE arête. Sans quoi la boîte se dessine elle-même, et se retrouve
    # toute seule à l'écran dès qu'on décoche les faces.
    from sdot.viz.polytope import polytope_mesh

    lim = 2.0
    verts, edges, faces = polytope_mesh( [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ], [ 1.0, 1.0, 1.0 ],
                                         bounds = [ [ -lim, lim ] ] * 3 )
    assert len( faces ) > 0                          # refermé : il y a bien de quoi remplir

    # les trois vraies arêtes, et rien d'autre : celles qui partent du coin ( 1, 1, 1 ) le long
    # des trois axes ( intersections deux à deux des trois plans demandés ).
    assert len( edges ) == 3, f"{ len( edges ) } arêtes, attendu 3"
    for a, b in edges:
        for k in range( 3 ):
            for side in ( -lim, lim ):
                assert not ( abs( verts[ a ][ k ] - side ) < 1e-9
                         and abs( verts[ b ][ k ] - side ) < 1e-9 ), \
                    f"arête posée sur la boîte : { verts[ a ] } -> { verts[ b ] }"

if test( "viz_of_a_bounded_cell_is_untouched" ):
    # aucune coupe factice sur une cellule bornée : rien à retirer, rien à pointiller. Le carré
    # unité doit sortir en UNE face de 4 sommets et 4 arêtes pleines, comme avant.
    v = Visualizer()
    Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] ).add_to_viz( v )
    assert len( v.polytopes ) == 0
    assert len( v.polygons ) == 1 and len( v.polygons[ 0 ] ) == 4
    assert len( v.edges ) == 4

if test( "viz_of_a_batch_uses_each_item_own_counts" ):
    # deux cellules de TAILLES DIFFÉRENTES dans un même batch : les tableaux sont denses au plus
    # grand, donc dessiner la petite sur le compte de la grande lui ferait traîner des places
    # qu'elle n'utilise pas. C'est le cas courant d'un diagramme de Voronoï.
    from sdot import Voronoi
    pos = numpy.array( [ [ 0.2, 0.5 ], [ 0.55, 0.5 ], [ 0.9, 0.2 ], [ 0.9, 0.8 ] ] )
    cs = Voronoi( pos, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) ).cells
    nvs = numpy.asarray( cs.nb_vertices.value )
    assert nvs.min() < nvs.max(), f"il faut des tailles différentes pour que le test dise quelque chose ({ nvs })"

    v = Visualizer()
    cs.add_to_viz( v )
    assert [ len( f ) for f in v.polygons ] == list( nvs )

if test( "viz_colors_are_the_seed_and_nothing_else" ):
    # une couleur dit QUEL germe. Elle se prend donc au rang de l'item, pas au nombre de choses
    # déjà dessinées -- et la preuve est une cellule VIDE : le germe du milieu, dominé par un poids
    # bas, n'a plus rien à montrer, et les deux autres doivent garder EXACTEMENT la couleur qu'ils
    # avaient sans lui. Avec un compteur d'appels, le troisième héritait de celle du deuxième.
    from sdot import PowerDiagram
    from sdot.viz.Visualizer import scale_color

    pos = numpy.array( [ [ 0.2, 0.5 ], [ 0.5, 0.5 ], [ 0.8, 0.5 ] ] )
    box = ( [ 0, 0 ], [ 1, 1 ] )

    def face_colors( viz ):
        return [ tuple( viz.colors[ c ][ :3 ] ) for c in viz.polygon_colors ]

    full = Visualizer()
    PowerDiagram( pos, boundaries = box_half_spaces( *box )).add_to_viz( full )
    assert len( full.polygons ) == 3
    assert numpy.allclose( face_colors( full ), [ scale_color( i ) for i in range( 3 ) ] )

    holed = Visualizer()
    PowerDiagram( pos, weights = numpy.array( [ 0.0, -1.0, 0.0 ] ), boundaries = box_half_spaces( *box )).add_to_viz( holed )
    assert len( holed.polygons ) == 2                      # celle du milieu a disparu
    assert numpy.allclose( face_colors( holed ), [ scale_color( 0 ), scale_color( 2 ) ] )

if test( "viz_colors_do_not_drift_from_one_frame_to_the_next" ):
    # le même diagramme, redessiné image après image : chaque germe doit y reprendre SA couleur.
    # C'est ce que la remise à zéro par image achète -- sans elle, l'image `k` commençait là où la
    # précédente s'était arrêtée et toute l'animation clignotait.
    from sdot import Voronoi
    pos = numpy.array( [ [ 0.2, 0.2 ], [ 0.75, 0.3 ], [ 0.45, 0.8 ], [ 0.9, 0.85 ] ] )
    n, nb_frames = len( pos ), 3

    v = Visualizer( frame_axis = "pas" )
    for k in range( nb_frames ):
        if k:
            v.new_frame( k )
        Voronoi( pos + 0.02 * k, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) ).add_to_viz( v )

    cols = [ tuple( v.colors[ c ][ :3 ] ) for c in v.polygon_colors ]
    assert len( cols ) == n * nb_frames
    for k in range( 1, nb_frames ):
        assert cols[ k * n : ( k + 1 ) * n ] == cols[ : n ], k
    assert len( set( cols[ : n ] ) ) == n                  # et les quatre sont bien distinctes


# -- ce qu'on REGARDE ----------------------------------------------------------------------------
# Des `experiment` et non des `test` : une image ne s'asserte pas. Ce qu'on vérifie ici est que les
# deux SORTIES du visualiseur (la page HTML autonome et le VTK de ParaView) sortent bien pour
# chacun des régimes de `Cell.add_to_viz` -- V-représentation en 2D, treillis de faces en 3D,
# H-représentation au-delà et sur une cellule non bornée -- plus la série d'images.
#
#   ./run experiment test_Cell                    # toutes
#   ./run experiment "test_Cell::viz 3D"          # une seule
#   ./run experiment "test_Cell::viz cut*" --nb-cuts=4,8    # un balayage : une sortie par valeur
#
# Chaque entrée écrit dans son `p.out_dir` -- tmp/experiment/test_Cell__viz_3D/<env>/, sans date :
# le chemin ne bouge pas d'un jour à l'autre, donc l'onglet ouvert dessus se recharge.

def _write_both( p, viz, stem ):
    """Les deux sorties, côte à côte -- c'est le geste que ces expériences vérifient."""
    print( "  html :", viz.write_html( p.out_dir / f"{ stem }.html" ) )
    print( "  vtk  :", viz.write_vtk( p.out_dir / f"{ stem }.vtu" ) )


if p := experiment( "viz 2D" ):
    # le régime 2D : `vertex_coords` EST le polygone, en ordre cyclique -- une face et son tour
    # d'arêtes, sans rien à reconstruire. Deux cellules pour voir la palette automatique jouer.
    v = Visualizer( title = "Cell 2D" )
    for shift, normal in ( ( 0.0, [ 1, 1 ] ), ( 1.3, [ -1, 2 ] ) ):
        c = Cell.make_hypercube( 2, [ shift, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
        c.cut( normal, float( numpy.dot( normal, [ shift + 0.75, 0.75 ] ) ) )
        c.add_to_viz( v, points = True )
    _write_both( p, v, "cell_2d" )

if p := experiment( "viz 3D" ):
    # le régime 3D : les faces se relisent sur le treillis ( `Cell.faces` ). C'est LE cas où la page a quelque chose à cacher -- faces
    # pleines, arêtes de derrière -- et celui qu'on ouvre dans ParaView pour tourner autour.
    c = _unit_cube( 3 )
    c.cut( [ 1, 1, 1 ], 2.5 )
    v = Visualizer( title = "Cube 3D, un coin tranché" )
    c.add_to_viz( v, opacity = 0.55, points = True )
    _write_both( p, v, "cell_3d" )

if p := experiment( "viz 5D" ):
    # au-delà de la 3D il n'y a plus de treillis à envoyer : `add_to_viz` passe la H-représentation,
    # la page en montre une COUPE 3D ( couper des demi-espaces redonne des demi-espaces ) et y
    # ajoute le fil de fer PROJETÉ. Côté VTK, les coordonnées au-delà de la 3e partent en données de
    # points, à ParaView de faire ses coupes lui-même.
    c = _unit_cube( 5 )
    c.cut( [ 1, 1, 1, 1, 1 ], 4.5 )
    c.cut( [ 1, -0.4, 0.3, 0.2, 0.1 ], 0.75 )
    v = Visualizer( title = "Cell 5D ( coupe )" )
    c.add_to_viz( v )
    _write_both( p, v, "cell_5d" )

if p := experiment( "viz unbounded" ):
    # une cellule NON BORNÉE n'a pas de sommets à montrer -- le simplexe factice de
    # `init_as_unbounded` en a, mais ce sont ceux d'un stand-in, pas de la cellule. C'est donc la
    # H-représentation qui part, et c'est le visualiseur qui la referme sur la boîte de la scène.
    v = Visualizer( title = "Cellules non bornées" )
    c = Cell.make_unbounded( 3 )
    for n, o in ( ( [ -1, 0, 0 ], 0 ), ( [ 0, -1, 0 ], 0 ), ( [ 0, 0, -1 ], 0 ) ):
        c.cut( n, o )
    assert not c.is_bounded                                    # l'octant positif : un cône
    c.add_to_viz( v, opacity = 0.55 )
    _write_both( p, v, "cell_unbounded" )

if p := experiment( "viz cut by cut",
                    dim     = Param( 3, help = "dimension de la cellule" ),
                    nb_cuts = Param( 4, help = "nombre de coupes aléatoires" ),
                    seed    = Param( 0, help = "graine du tirage des plans" ) ):
    # une IMAGE par coupe : le cube unité rogné plan après plan. La page se déroule toute seule
    # ( axe « coupe », lecture/pause ) et ParaView reçoit un `.pvd`, c'est-à-dire une série
    # temporelle -- donc un `.vtu` par image, ce que les sorties mono-image ci-dessus ne couvrent
    # pas. On part du cube, borné, et non de la cellule infinie : le cadrage de la scène est
    # COMMUN à toutes les images, donc une première image grande comme le monde écraserait toutes
    # les suivantes en un trait ( c'est `viz unbounded` qui montre ce régime-là ).
    d   = p.dim
    rng = numpy.random.default_rng( p.seed )
    c   = _unit_cube( d )
    v   = Visualizer( title = f"cube { d }D, coupe par coupe", frame_axis = "coupe" )
    c.add_to_viz( v, opacity = 0.55 )
    nb_drawn = 1
    for k in range( p.nb_cuts ):
        n = rng.normal( size = d )
        c.cut( n.tolist(), float( n @ rng.uniform( 0.2, 0.8, size = d ) ) )   # passe par l'intérieur
        # des plans tirés au hasard finissent par tout exclure ; la cellule vide est un état
        # légitime, mais elle ne fait pas une image -- on s'arrête là plutôt que d'en ajouter.
        if int( numpy.max( numpy.asarray( c.nb_vertices.value ) ) ) == 0:
            print( f"  cellule vide à la coupe { k }" )
            break
        v.new_frame( k + 1 )
        c.add_to_viz( v, opacity = 0.55 )
        nb_drawn += 1
    print( f"  { nb_drawn } image(s)" )
    _write_both( p, v, f"cut_by_cut_{ d }d" )
