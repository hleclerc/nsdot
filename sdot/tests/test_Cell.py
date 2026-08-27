import numpy

from loom.testing import check_grad, test

from loom import driver, new_batch_axis
from sdot import Cell, Visualizer

if test( "basic" ):
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 2, 0 ], [ 0, 1 ] ] )
    assert c.measure == 2

    # 2D : `vertex_positions` est déjà le polygone en ordre cyclique, donc pas de treillis de
    # faces -- ces attributs-là restent Unbound (cf. `Cell.py::_output_exceptions_for_init`).
    assert c.vertex_indices.is_undefined
    assert c.edge_indices.is_undefined
    assert c.nb_edges.allocated_capacity() is None   # ni buffer ni capacité : jamais alloué

if test( "basic_1D" ):
    # 1D : la cellule est un SEGMENT -- `vertex_positions` en donne les deux extrémités, et il n'y
    # a ni treillis de faces (`vertex_indices` / `edge_indices`) ni scratch de triangulation
    # (`item_map`) à allouer. Voir les deux surcharges de `Cell.h::measure`.
    c = Cell.make_hypercube( 1, [ 0.5 ], [ [ 2 ] ] )
    assert c.measure == 2

    # les attributs du régime d > 2 restent Unbound : rien n'a été alloué pour eux, ni les
    # tenseurs du treillis de faces, ni le compte qui en dimensionne un.
    assert c.vertex_indices.is_undefined
    assert c.edge_indices.is_undefined
    assert c.nb_edges.allocated_capacity() is None   # ni buffer ni capacité : jamais alloué

if test( "unbounded_1D" ):
    # la cellule non bornée (simplexe aligné marqué INFINITE) : sa mesure est "infinie". En 1D les
    # coupes sont x >= 0 et x <= 1, écrites par une branche à part -- la version 2D indexerait un
    # `dim = 1` qui n'existe pas ici.
    c = Cell.make_unbounded( 1 )
    assert float( c.measure ) > 1e300

if test( "batch" ):
    # c = Cell.make_hypercube( 3, [ 0, 0, 0 ], [ [ 2, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ], batch_axes = [ new_batch_axis( 2 ) ] )
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 2, 0 ], [ 0, 1 ] ], batch_axes = [ new_batch_axis( 2 ) ] )
    # info( c.vertex_positions )
    assert tuple( c.measure.tensor ) == ( 2, 2 )

if test( "grad_hypercube" ):
    # Dérivées des sorties d'un hypercube 2D par rapport à ses entrées `origin` et `axes`.
    # `check_grad` est agnostique : il confronte l'adjoint du driver à une différence finie, sans
    # jamais importer le framework. `f` renvoie un `Tensor` : `check_grad` en prend la vue dense
    # (le padding de capacité est retiré tout seul, sans `.raw[ :n ]`).
    origin = driver.array( [ 0.3, -0.2 ] )
    axes   = driver.array( [ [ 2.0, 0.1 ], [ -0.3, 1.0 ] ] )

    check_grad( lambda o, a: Cell.make_hypercube( 2, o, a ).vertex_positions, origin, axes )
    check_grad( lambda o, a: Cell.make_hypercube( 2, o, a ).cut_directions  , origin, axes )
    check_grad( lambda o, a: Cell.make_hypercube( 2, o, a ).cut_offsets     , origin, axes )

    # une seule entrée dérivée à la fois : l'autre n'est pas perturbée, donc son gradient arrive
    # au backward en NoneTensor ( is_valid() == faux ) et le bloc qui l'écrit est supprimé à la
    # compilation. On couvre ainsi les deux branches `! is_valid()`.
    check_grad( lambda o: Cell.make_hypercube( 2, o, axes   ).vertex_positions, origin )
    check_grad( lambda o: Cell.make_hypercube( 2, o, axes   ).cut_offsets     , origin )
    check_grad( lambda a: Cell.make_hypercube( 2, origin, a ).cut_directions  , axes   )
    check_grad( lambda a: Cell.make_hypercube( 2, origin, a ).vertex_positions, axes   )


if test( "grad_measure" ):
    # `measure` (l'aire, en 2D) par rapport à `origin`/`axes` -- passe par `measure_bwd`'s
    # shoelace adjoint, et par `vertex_positions` seul (`cut_directions`/`cut_offsets` sont hors
    # du call, voir `input_exceptions` dans `Cell.py::measure`).
    origin = driver.array( [ 0.3, -0.2 ] )
    axes   = driver.array( [ [ 2.0, 0.1 ], [ -0.3, 1.0 ] ] )

    check_grad( lambda o, a: Cell.make_hypercube( 2, o, a ).measure, origin, axes )

    # une seule entrée dérivée : couvre la branche `! grad_for_cell.vertex_positions.is_valid()`
    # côté `init_as_hypercube_bwd` (l'autre entrée n'est pas perturbée).
    check_grad( lambda o: Cell.make_hypercube( 2, o, axes   ).measure, origin )
    check_grad( lambda a: Cell.make_hypercube( 2, origin, a ).measure, axes   )

if test( "grad_measure_1D" ):
    # même chemin `measure` qu'en 2D (d <= 2), mais l'adjoint y est celui du segment : seuls les
    # sommets extrêmes portent le gradient.
    origin = driver.array( [ 0.3 ] )
    axes   = driver.array( [ [ 2.0 ] ] )

    check_grad( lambda o, a: Cell.make_hypercube( 1, o, a ).vertex_positions, origin, axes )
    check_grad( lambda o, a: Cell.make_hypercube( 1, o, a ).cut_directions  , origin, axes )
    check_grad( lambda o, a: Cell.make_hypercube( 1, o, a ).cut_offsets     , origin, axes )
    check_grad( lambda o, a: Cell.make_hypercube( 1, o, a ).measure         , origin, axes )

    check_grad( lambda o: Cell.make_hypercube( 1, o, axes   ).measure, origin )
    check_grad( lambda a: Cell.make_hypercube( 1, origin, a ).measure, axes   )

if test( "batch_1D" ):
    c = Cell.make_hypercube( 1, [ 0 ], [ [ 3 ] ], batch_axes = [ new_batch_axis( 2 ) ] )
    assert tuple( c.measure.tensor ) == ( 3, 3 )

if p := test( "cut" ):
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 0 ], 0.1 )

    v = Visualizer()
    c.add_to_viz( v )
    v.write_html( p.out_dir / "cut.html" )

    assert c.measure == 0.1

if test( "cut_keeps_the_edge_to_cut_correspondence" ):
    # L'invariant 2D dont vit `Cell.cxx::cut` : la coupe i porte l'arête [ v_i, v_i+1 ], donc
    # `nb_cuts == nb_vertices`. On coupe un coin du carré unité et on le vérifie arête par arête --
    # les DEUX extrémités de l'arête i doivent être sur la coupe i.
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 1 ], 1.5 )

    n = int( c.nb_vertices.value )
    assert n == 5 and int( c.nb_cuts.value ) == n
    assert abs( float( c.measure ) - 0.875 ) < 1e-12      # 1 - le triangle de côté 1/2

    vp, cd, co = ( numpy.asarray( t ) for t in ( c.vertex_positions, c.cut_directions, c.cut_offsets ) )
    for i in range( n ):
        j = ( i + 1 ) % n
        assert abs( cd[ i ] @ vp[ i ] - co[ i ] ) < 1e-12
        assert abs( cd[ i ] @ vp[ j ] - co[ i ] ) < 1e-12

    # la coupe demandée est reprise TELLE QUELLE, sans normalisation
    assert any( numpy.allclose( cd[ i ], [ 1, 1 ] ) and abs( co[ i ] - 1.5 ) < 1e-12 for i in range( n ) )

if test( "cut_direction_is_not_normalized" ):
    # `offset` est le produit scalaire tel quel, donc ( 2n, 2o ) est le MÊME demi-espace que
    # ( n, o ) : une direction trois fois plus longue avec un offset trois fois plus grand doit
    # rendre exactement la même cellule.
    a = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    b = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    a.cut( [ 1, 1 ], 1.5 )
    b.cut( [ 3, 3 ], 4.5 )

    assert int( a.nb_vertices.value ) == int( b.nb_vertices.value )
    assert numpy.allclose( numpy.asarray( a.vertex_positions ), numpy.asarray( b.vertex_positions ) )
    assert abs( float( a.measure ) - float( b.measure ) ) < 1e-12

if test( "cut_degenerate" ):
    # deux dégénérescences. Une coupe qui ne coupe rien (tout le polygone est dedans) ne doit RIEN
    # ajouter -- sinon chaque coupe redondante ferait grossir la H-représentation. Une coupe qui
    # exclut tout doit vider la cellule proprement (0 sommet, 0 coupe, mesure nulle) plutôt que de
    # laisser une géométrie à moitié écrite.
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )

    c.cut( [ 1, 0 ], 5 )
    assert int( c.nb_vertices.value ) == 4 and int( c.nb_cuts.value ) == 4
    assert abs( float( c.measure ) - 1 ) < 1e-12

    c.cut( [ 1, 0 ], -1 )
    assert int( c.nb_vertices.value ) == 0 and int( c.nb_cuts.value ) == 0
    assert float( c.measure ) == 0

if test( "cut_capacity_overflow" ):
    # Une capacité est une SUPPOSITION : seul le kernel sait combien de sommets il produit. On en
    # impose ici une trop petite. Le kernel s'en aperçoit AVANT d'écrire quoi que ce soit
    # (`res.nb_vertices.set( nb + 1 )` renvoie faux), enregistre ce qu'il voulait et sort ; la
    # plateforme réserve plus et relance. Le résultat doit être celui de la bonne capacité.
    orig = Cell._cut_capacities
    Cell._cut_capacities = lambda self: { "res.nb_vertices": 3, "res.nb_cuts": 3 }
    try:
        c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
        c.cut( [ 1, 1 ], 1.5 )     # 5 sommets, pour une capacité de 3
        assert int( c.nb_vertices.value ) == 5
        assert abs( float( c.measure ) - 0.875 ) < 1e-12
        # la preuve que le second tour a bien eu lieu : la capacité a grossi ( max( 5, 2 * 3 ) )
        assert c.vertex_positions.capacity == ( 6, 2 ) and c.cut_offsets.capacity == ( 6, )
    finally:
        Cell._cut_capacities = orig

if test( "cut_tangent" ):
    # Les deux tangences, celles où un sommet tombe sur le plan de coupe. Le clip ne les traite PAS
    # à part : il ne pose qu'une question, `s > 0`, et un sommet à l'epsilon près du plan répond
    # comme un sommet dedans. Ce qui en sort peut être dégénéré -- une arête de longueur nulle --
    # mais c'est une cellule cohérente, et la géométrie, elle, est juste.
    #
    # Tangente à une ARÊTE : la coupe longe le côté droit. Aucun sommet n'est dehors, donc aucun
    # croisement n'est créé : rien n'est retiré et rien n'est ajouté, ni sommet ni coupe.
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 0 ], 1 )
    assert int( c.nb_vertices.value ) == 4 and int( c.nb_cuts.value ) == 4
    assert abs( float( c.measure ) - 1 ) < 1e-12

    # Tangente à un SOMMET : là, trois sommets sont dehors, donc les deux arêtes qui les relient au
    # coin ( 0, 0 ) sont bien traversées et donnent chacune un croisement -- qui tombe sur ce coin.
    # La cellule sort avec trois sommets CONFONDUS au lieu d'un. Aire nulle, position juste : le
    # coût de ne jamais tester `s == 0`, et il est en 1e-16, pas en géométrie.
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 1 ], 0 )
    vp = numpy.asarray( c.vertex_positions )[ : int( c.nb_vertices.value ) ]
    assert int( c.nb_vertices.value ) == int( c.nb_cuts.value )     # l'invariant 2D tient
    assert numpy.allclose( vp, 0 )                                  # toutes au même point
    assert float( c.measure ) == 0

def _half_planes_of( *corners ):
    """Les demi-espaces `n . x <= o` d'un polygone convexe donné par ses sommets, en ordre CCW.

    Une arête P -> Q donne la normale sortante `( dy, -dx )` -- pas normalisée, exprès : `cut` ne le
    demande pas, et la garder brute évite une racine carrée dans la valeur attendue du test.
    """
    res = []
    for i in range( len( corners ) ):
        p, q = numpy.asarray( corners[ i ], float ), numpy.asarray( corners[ ( i + 1 ) % len( corners ) ], float )
        d = q - p
        n = numpy.array( [ d[ 1 ], - d[ 0 ] ] )
        res.append( ( list( n ), float( n @ p ) ) )
    return res


if test( "cut_unbounded_becomes_bounded" ):
    # La cellule non bornée est un SIMPLEXE FACTICE (`init_as_unbounded`) : ses plans portent des
    # offsets inventés, et une coupe les classerait selon l'échelle arbitraire de ce simplexe. `cut`
    # les repousse d'abord jusqu'à ce que le classement soit celui qu'il serait à l'infini.
    #
    # Sans cette poussée, aucune des quatre coupes ci-dessous ne toucherait le simplexe unité (il
    # est tout entier dedans) : la cellule resterait le simplexe, et resterait « non bornée ». Avec,
    # les plans factices sont mangés un à un et il reste le carré unité, exactement.
    c = Cell.make_unbounded( 2 )
    assert not bool( numpy.asarray( c.is_fully_bounded ) )

    for n, o in ( ( [ -1, 0 ], 0 ), ( [ 1, 0 ], 1 ), ( [ 0, -1 ], 0 ) ):
        c.cut( n, o )
        assert not bool( numpy.asarray( c.is_fully_bounded ) )   # une bande, un demi-plan : infinis
        assert float( c.measure ) > 1e300

    c.cut( [ 0, 1 ], 1 )    # la quatrième referme la cellule
    assert bool( numpy.asarray( c.is_fully_bounded ) )
    assert int( c.nb_vertices.value ) == 4
    assert float( c.measure ) == 1

if test( "cut_unbounded_triangle" ):
    # Le même, hors des axes et avec un ordre de coupes quelconque : trois demi-plans découpent leur
    # triangle dans le plan entier. C'est le test qui dit que la poussée trouve la BONNE
    # configuration, pas seulement une configuration bornée.
    corners = ( ( 0, 0 ), ( 4, 1 ), ( 1, 3 ) )
    exact = 5.5                                  # |( B - A ) x ( C - A )| / 2 = | 4*3 - 1*1 | / 2

    for order in ( ( 0, 1, 2 ), ( 2, 0, 1 ), ( 1, 2, 0 ) ):
        c = Cell.make_unbounded( 2 )
        hp = _half_planes_of( *corners )
        for i in order:
            c.cut( *hp[ i ] )

        assert bool( numpy.asarray( c.is_fully_bounded ) )
        assert int( c.nb_vertices.value ) == 3
        assert abs( float( c.measure ) - exact ) < 1e-12

if test( "cut_unbounded_far_from_the_origin" ):
    # Le simplexe factice est bâti à l'ORIGINE et à l'échelle 1 : pour une cellule lointaine, il faut
    # le pousser jusque-là, et les coordonnées intermédiaires montent à l'échelle de la translation.
    # Le résultat reste juste, mais à la précision de CES coordonnées-là, pas des siennes : ci-dessous
    # un décalage de 1e3 coûte ~3e-12 sur une aire de 5.5 (contre ~1e-15 à l'origine, et 0 en partant
    # d'une boîte bornée qui contient déjà le triangle -- donc c'est bien la poussée qui le paie).
    # C'est le prix du stand-in, pas de l'algorithme ; le réduire demanderait de recalculer les
    # sommets finaux comme intersections de leurs deux plans, une fois la cellule bornée.
    shift = numpy.array( [ 1000.0, -500.0 ] )
    corners = [ tuple( numpy.array( p, float ) + shift ) for p in ( ( 0, 0 ), ( 4, 1 ), ( 1, 3 ) ) ]

    c = Cell.make_unbounded( 2 )
    for n, o in _half_planes_of( *corners ):
        c.cut( n, o )

    assert bool( numpy.asarray( c.is_fully_bounded ) )
    assert int( c.nb_vertices.value ) == 3
    assert abs( float( c.measure ) - 5.5 ) < 1e-9

if test( "grad_cut" ):
    # L'adjoint de `cut` par rapport à TOUT ce qui entre : la cellule d'origine (via `origin`/`axes`)
    # et le demi-espace lui-même. Le clip est un scatter -- un sommet d'entrée alimente jusqu'à trois
    # sommets de sortie -- donc `cut_bwd` accumule, et `cut_bwd_setup` remet à zéro avant.
    #
    # `1.5` est strictement entre 1 et 2 : la combinatoire du clip ne change pas sous la
    # perturbation, sans quoi la différence finie ne mesurerait pas la même fonction des deux côtés.
    origin = driver.array( [ 0.0, 0.0 ] )
    axes   = driver.array( [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ] )
    normal = driver.array( [ 1.0, 1.0 ] )
    offset = driver.array( 1.5 )

    check_grad( lambda o, a, n, c: Cell.make_hypercube( 2, o, a ).cut( n, c ).measure,
                origin, axes, normal, offset )

    # une seule entrée dérivée à la fois : les autres arrivent au backward en NoneTensor, et les
    # blocs qui les écrivent disparaissent à la compilation. On couvre ainsi chaque `is_valid()`.
    check_grad( lambda o: Cell.make_hypercube( 2, o, axes ).cut( normal, offset ).measure, origin )
    check_grad( lambda a: Cell.make_hypercube( 2, origin, a ).cut( normal, offset ).measure, axes )
    check_grad( lambda n: Cell.make_hypercube( 2, origin, axes ).cut( n, offset ).measure, normal )
    check_grad( lambda c: Cell.make_hypercube( 2, origin, axes ).cut( normal, c ).measure, offset )

if test( "grad_cut_outputs" ):
    # Pas seulement la mesure : chaque sortie différentiable de `cut` a son adjoint. Les coupes de
    # sortie sont des COPIES (une des nôtres, ou le demi-espace donné), donc leur cotangente doit
    # revenir telle quelle sur celle dont elles ont été copiées.
    origin = driver.array( [ 0.0, 0.0 ] )
    axes   = driver.array( [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ] )
    normal = driver.array( [ 1.0, 1.0 ] )
    offset = driver.array( 1.5 )

    for out in ( lambda c: c.vertex_positions, lambda c: c.cut_directions, lambda c: c.cut_offsets ):
        check_grad( lambda o, a, n, c: out( Cell.make_hypercube( 2, o, a ).cut( n, c ) ),
                    origin, axes, normal, offset )

if test( "grad_cut_chained" ):
    # Deux coupes de suite : la cotangente de la seconde doit retraverser la première. C'est ce qui
    # vérifie que `cut` ne se donne pas `self` en entrée -- le backward est construit APRÈS que la
    # mise à jour en place a rebranché la cellule sur des buffers d'une autre capacité.
    origin = driver.array( [ 0.0, 0.0 ] )
    axes   = driver.array( [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ] )
    normal = driver.array( [ 1.0, 1.0 ] )
    offset = driver.array( 1.5 )

    check_grad( lambda o, a, n, c: Cell.make_hypercube( 2, o, a ).cut( n, c ).cut( [ -1.0, 0.3 ], 0.1 ).measure,
                origin, axes, normal, offset )

if test( "grad_cut_batched" ):
    # `direction` et `offset` sont PARTAGÉS par tous les items du batch : leur gradient est une
    # somme sur les items, donc chacun l'accumule dans les deux mêmes cases -- atomiquement, sur un
    # buffer que `cut_bwd_setup` a remis à zéro. Un seed manquant ou un `+=` non atomique se verrait
    # ici, et seulement ici.
    origin = driver.array( [ 0.0, 0.0 ] )
    axes   = driver.array( [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ] )
    normal = driver.array( [ 1.0, 1.0 ] )
    offset = driver.array( 1.5 )

    def cut_a_batch( n, c ):
        cell = Cell.make_hypercube( 2, origin, axes, batch_axes = [ new_batch_axis( 4 ) ] )
        return cell.cut( n, c ).measure

    check_grad( cut_a_batch, normal, offset )

if test( "cut_batched" ):
    # `cut` ne réserve RIEN par thread (contrairement à `_measure_nd`, cf.
    # `_measure_bytes_per_thread`) : rien ne plafonne donc le nombre de threads, et tout un batch
    # se coupe d'un seul appel, un item par thread. Chaque item doit sortir la même cellule.
    nb_items = 64
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ], batch_axes = [ new_batch_axis( nb_items ) ] )
    c.cut( [ 1, 1 ], 1.5 )

    m = numpy.asarray( c.measure.tensor )
    assert m.shape == ( nb_items, )
    assert numpy.allclose( m, 0.875 )


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


def _check_against_reference( c, planes, name, exact = True ):
    """La cellule EST l'intersection de `planes` : sommets, coupes et arêtes, tous les trois.

    `exact = False` quand le plan de coupe passe par des sommets existants : le clip ne teste
    jamais `s == 0` (voir `Cell.cxx::cut`), donc il produit alors des sommets CONFONDUS à ~1e-16 et
    les arêtes de longueur nulle qui les relient. On vérifie toujours la géométrie -- l'ENSEMBLE
    des points, chaque sommet sur ses propres coupes, chaque arête non dégénérée attendue -- mais
    plus l'absence de doublons.
    """
    nv, ne, nc = int( c.nb_vertices.value ), int( c.nb_edges.value ), int( c.nb_cuts.value )
    vp = numpy.asarray( c.vertex_positions )[ : nv ]
    vi = numpy.asarray( c.vertex_indices   )[ : nv ]
    ei = numpy.asarray( c.edge_indices     )[ : ne ]
    cd = numpy.asarray( c.cut_directions   )[ : nc ]
    co = numpy.asarray( c.cut_offsets      )[ : nc ]
    d = vp.shape[ 1 ]

    dirs, offs, ref, ref_on = _reference_cell( planes )
    rnd = lambda v: tuple( numpy.round( v, 7 ) + 0.0 )

    # 1. les sommets, en tant qu'ENSEMBLE de points : chacun de la cellule est un de la référence,
    #    et chacun de la référence est là. En mode `exact`, un pour un -- donc sans doublon, ce
    #    qu'une dégénérescence mal traitée produirait en premier.
    where = [ next( ( i for i, r in enumerate( ref ) if numpy.allclose( p, r, atol = 1e-7 ) ), None )
              for p in vp ]
    assert None not in where, f"{ name } : sommet hors de la référence"
    assert set( where ) == set( range( len( ref ) ) ), f"{ name } : sommet de la référence manquant"
    if exact:
        assert sorted( map( rnd, vp ) ) == sorted( map( rnd, ref ) ), f"{ name } : sommets"

    # 2. le treillis est cohérent avec la géométrie : chaque sommet est bien SUR chacune de ses
    #    d coupes ( c'est ce que `vertex_indices` prétend ), et dedans pour toutes les autres.
    for k, p in enumerate( vp ):
        assert list( vi[ k ] ) == sorted( vi[ k ] ), f"{ name } : liste de coupes non triée en v{ k }"
        for r in vi[ k ]:
            assert abs( cd[ r ] @ p - co[ r ] ) < 1e-9, f"{ name } : v{ k } n'est pas sur la coupe { r }"
    assert numpy.all( vp @ cd.T - co < 1e-9 ), f"{ name } : un sommet est dehors"

    # 3. les coupes survivantes sont exactement les plans que touche un sommet, dans l'ordre des
    #    plans ( la compaction est stable )
    used = sorted( set().union( *ref_on ) ) if ref_on else []
    assert len( cd ) == len( used ), f"{ name } : { len( cd ) } coupes, attendu { len( used ) }"
    for r, u in enumerate( used ):
        assert numpy.allclose( cd[ r ], dirs[ u ] ) and abs( co[ r ] - offs[ u ] ) < 1e-9, \
            f"{ name } : la coupe { r } n'est pas le plan { u }"

    # 4. les arêtes. Le critère : deux sommets sont voisins quand la PLUS PETITE FACE qui les
    #    contient -- l'intersection de tous les plans qui portent les deux -- ne contient qu'eux.
    #    Le raccourci « ils partagent d-1 plans » n'est valable que sur un polytope SIMPLE, et une
    #    coupe passant par des sommets existants n'en donne justement pas un : là, des sommets
    #    portent plus de d plans, et deux diagonales d'une face carrée en partagent d-1 sans être
    #    voisines pour autant.
    #
    #    La liste de coupes de chaque arête doit être incluse dans ce partage. Une arête DÉGÉNÉRÉE
    #    ( ses deux bouts sur le même point de la référence ) est mise de côté : c'est le résidu
    #    attendu d'un plan passant par un sommet, pas une arête de la cellule.
    def _adjacent( a, b ):
        shared = ref_on[ a ] & ref_on[ b ]
        if len( shared ) < d - 1:
            return False
        return not any( shared <= ref_on[ c ] for c in range( len( ref ) ) if c != a and c != b )

    expected = { ( a, b ) for a in range( len( ref ) ) for b in range( a + 1, len( ref ) )
                 if _adjacent( a, b ) }
    got, nb_degenerate = set(), 0
    for row in ei:
        a, b = where[ row[ 0 ] ], where[ row[ 1 ] ]
        if a == b:
            nb_degenerate += 1
            continue
        got.add( ( min( a, b ), max( a, b ) ) )
        assert { used[ r ] for r in row[ 2 : ] } <= ( ref_on[ a ] & ref_on[ b ] ), \
            f"{ name } : liste de coupes fausse pour l'arête { row }"
    assert got == expected, (
        f"{ name } : arêtes -- en trop { [ ( ref[ a ], ref[ b ] ) for a, b in sorted( got - expected ) ] }, "
        f"manquantes { [ ( ref[ a ], ref[ b ] ) for a, b in sorted( expected - got ) ] }" )
    if exact:
        assert nb_degenerate == 0 and len( got ) == len( ei ), f"{ name } : arête en double"


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
    # le pendant de `basic` : au-dessus de la 2D, le treillis de faces EXISTE, et c'est lui qui
    # porte la cellule ( les sommets seuls ne suffisent plus : aucun ordre cyclique à lire ).
    c = _unit_cube( 3 )
    assert c.vertex_indices.is_defined
    assert c.edge_indices.is_defined
    assert int( c.nb_vertices.value ) == 8
    assert int( c.nb_edges.value ) == 12
    assert int( c.nb_cuts.value ) == 6
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

    # le compte est celui qu'on attend d'un coin de cube tranché : un sommet en moins, trois en
    # plus, trois arêtes en plus, une coupe en plus.
    c = _unit_cube( 3 )
    c.cut( [ 1, 1, 1 ], 2.5 )
    assert ( int( c.nb_vertices.value ), int( c.nb_edges.value ), int( c.nb_cuts.value ) ) == ( 10, 15, 7 )

if test( "cut_3D_random" ):
    # des plans quelconques, en série : c'est ce qui explore les combinatoires qu'on n'écrirait
    # pas à la main ( sommets retirés en paquet, coupes qui deviennent inutiles et disparaissent ).
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
    # LE cas dégénéré, et il n'a rien d'exotique : des données alignées sur les axes donnent des
    # plans qui passent EXACTEMENT par des sommets, et le produit scalaire ne tombe alors pas rond
    # en flottant -- il sort à ~1e-17 du plan. Le clip ne cherche pas à le reconnaître ( il ne
    # teste jamais `s == 0`, aucun test ne saurait distinguer ce sommet-là d'un sommet vraiment à
    # 1e-17 du plan ) : il le range dedans et recrée un croisement sur chacune de ses arêtes
    # traversées. Ce qui sort a des sommets confondus et des arêtes de longueur nulle, mais c'est
    # une cellule COHÉRENTE -- même ensemble de points, même treillis une fois les doublons
    # fusionnés, chaque sommet sur les coupes qu'il déclare. C'est ce qu'on vérifie ici.
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
    # le même piège en 2D, sur une cellule dont les coordonnées ne tombent pas rond : `offset` est
    # calculé ici dans un ordre différent de celui du kernel, donc le sommet traversé sort à ~1e-17
    # du plan. Ce qui compte n'est pas le nombre de sommets ( il peut y avoir un doublon confondu )
    # mais que la cellule reste JUSTE : l'invariant 2D tenu, aucun sommet du mauvais côté d'une de
    # ses coupes, et une aire égale à celle du polygone de référence.
    origin, axes = [ 0.13, -0.27 ], [ [ 1.7, 0.29 ], [ -0.41, 1.13 ] ]
    o, a0, a1 = ( numpy.asarray( v, float ) for v in ( origin, axes[ 0 ], axes[ 1 ] ) )
    corners = ( o, o + a0, o + a0 + a1, o + a1 )
    for corner in corners:
        for n in ( [ 1.0, 0.37 ], [ -0.83, 1.0 ], [ 0.61, -1.19 ] ):
            offset = float( numpy.dot( n, corner ) )
            c = Cell.make_hypercube( 2, origin, axes )
            c.cut( n, offset )

            nv = int( c.nb_vertices.value )
            vp = numpy.asarray( c.vertex_positions )[ : nv ]
            cd = numpy.asarray( c.cut_directions )[ : nv ]
            co = numpy.asarray( c.cut_offsets )[ : nv ]
            assert nv == int( c.nb_cuts.value )                    # l'invariant 2D tient toujours
            assert numpy.all( vp @ cd.T - co < 1e-9 )              # tout est dans toutes les coupes

            # l'aire de référence : le polygone d'origine rogné à la main ( shoelace sur les
            # sommets gardés + les croisements ), en flottants de l'hôte donc par un autre chemin.
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
    # rien de 3D-spécifique dans le clip : la même boucle vaut en 5D ( le simplexe de départ, les
    # listes de coupes, le recollage de la nouvelle facette sont tous écrits en `ct_dim` ).
    c, planes = _unit_cube( 5 ), _cube_planes( 5 )
    for n, o in ( ( [ 1, 1, 1, 1, 1 ], 4.5 ), ( [ 1, -0.4, 0.3, 0.2, 0.1 ], 0.75 ) ):
        c.cut( n, o )
        planes = planes + [ ( n, o ) ]
        _check_against_reference( c, planes, "cube 5D" )

if test( "cut_nd_empty" ):
    # tout est dehors : la cellule devient vide, et vide veut dire les trois comptes à zéro ( pas
    # seulement les sommets -- une arête ou une coupe qui traîne serait une cellule incohérente ).
    c = _unit_cube( 3 )
    c.cut( [ 1, 1, 1 ], -1.0 )
    assert ( int( c.nb_vertices.value ), int( c.nb_edges.value ), int( c.nb_cuts.value ) ) == ( 0, 0, 0 )

if test( "cut_nd_unbounded" ):
    # le simplexe factice de `init_as_unbounded` repoussé jusqu'à ce que la coupe le classe comme
    # elle le ferait à l'infini -- en nD la poussée résout le même système que le sommet, avec les
    # taux au second membre ( `growth_rate` ). Six demi-espaces plus tard : le cube, exactement.
    for d in ( 3, 4 ):
        c, planes = Cell.make_unbounded( d ), []
        for n, o in _cube_planes( d ):
            c.cut( n, o )
            planes.append( ( n, o ) )
            # bornée seulement une fois le dernier plan infini évincé, pas avant
            assert int( numpy.asarray( c.is_fully_bounded ) ) == ( len( planes ) == 2 * d )
        _check_against_reference( c, planes, f"non bornée { d }D -> cube" )
        assert int( c.nb_vertices.value ) == 2 ** d

if test( "cut_nd_capacity_overflow" ):
    # une capacité trop petite n'est pas une erreur : le kernel l'enregistre et sort sans rien
    # écrire de faux, la plateforme réserve plus et relance. Ici les trois comptes débordent.
    saved = Cell._cut_capacities
    Cell._cut_capacities = lambda self: { "res.nb_vertices": 4, "res.nb_edges": 4, "res.nb_cuts": 4 }
    try:
        c = _unit_cube( 3 )
        c.cut( [ 1, 1, 1 ], 2.5 )
        _check_against_reference( c, _cube_planes( 3 ) + [ ( [ 1, 1, 1 ], 2.5 ) ], "débordement" )
        assert c.vertex_positions.capacity[ 0 ] > 4     # la relance a bien élargi
    finally:
        Cell._cut_capacities = saved

if test( "cut_nd_batched" ):
    # `_cut_nd` réserve une ligne de `corr` PAR WORK-ITEM et plafonne les threads là-dessus
    # ( `thread_cap`, lu à l'exécution ). Un batch plus grand que ce plafond force donc un thread
    # à enchaîner plusieurs items en réutilisant sa ligne : si elle fuyait d'un item au suivant,
    # c'est ici que ça se verrait.
    nb_items = 64
    c = Cell.make_hypercube( 3, [ 0, 0, 0 ], numpy.eye( 3 ).tolist(),
                             batch_axes = [ new_batch_axis( nb_items ) ] )
    c.cut( [ 1, 1, 1 ], 2.5 )

    nv = int( c.nb_vertices.value )
    vp = numpy.asarray( c.vertex_positions )
    assert nv == 10 and vp.shape[ 0 ] == nb_items
    for b in range( nb_items ):
        assert numpy.allclose( vp[ b, : nv ], vp[ 0, : nv ] )

if test( "grad_cut_nd" ):
    # L'adjoint du clip nD, sortie par sortie. `measure` n'est pas encore écrite au-delà de la 2D,
    # donc on dérive directement les sorties de `cut` -- ce qui est de toute façon plus fin qu'un
    # scalaire : chaque sommet et chaque coupe a sa cotangente.
    origin = driver.array( numpy.zeros( 3 ) )
    axes   = driver.array( numpy.eye( 3 ) )
    normal = driver.array( [ 1.0, 0.9, 1.1 ] )
    offset = driver.array( 2.3 )

    for out in ( lambda c: c.vertex_positions, lambda c: c.cut_directions, lambda c: c.cut_offsets ):
        check_grad( lambda o, a, n, k: out( Cell.make_hypercube( 3, o, a ).cut( n, k ) ),
                    origin, axes, normal, offset )
        # une seule entrée dérivée à la fois : les autres arrivent en NoneTensor et les blocs qui
        # les écrivent disparaissent à la compilation.
        check_grad( lambda n: out( Cell.make_hypercube( 3, origin, axes ).cut( n, offset ) ), normal )
        check_grad( lambda k: out( Cell.make_hypercube( 3, origin, axes ).cut( normal, k ) ), offset )

if test( "grad_cut_nd_chained" ):
    # deux coupes de suite : la cotangente de la seconde doit retraverser la première ( et donc la
    # compaction, la renumérotation des coupes, le recollage de facette ).
    origin = driver.array( numpy.zeros( 3 ) )
    axes   = driver.array( numpy.eye( 3 ) )
    normal = driver.array( [ 1.0, 0.9, 1.1 ] )
    offset = driver.array( 2.3 )

    check_grad( lambda o, a, n, k: Cell.make_hypercube( 3, o, a ).cut( n, k )
                                       .cut( [ -1.0, 0.3, 0.2 ], 0.1 ).vertex_positions,
                origin, axes, normal, offset )

if test( "grad_cut_nd_batched" ):
    # `direction` et `offset` sont PARTAGÉS par tous les items : leur gradient est une somme sur
    # le batch, accumulée atomiquement dans les deux mêmes cases, sur un buffer que
    # `cut_bwd_setup` a remis à zéro. Et le backward se réalloue son propre `corr`.
    normal = driver.array( [ 1.0, 0.9, 1.1 ] )
    offset = driver.array( 2.3 )

    def cut_a_3d_batch( n, k ):
        cell = Cell.make_hypercube( 3, [ 0, 0, 0 ], numpy.eye( 3 ).tolist(),
                                    batch_axes = [ new_batch_axis( 4 ) ] )
        return cell.cut( n, k ).vertex_positions

    check_grad( cut_a_3d_batch, normal, offset )


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
    # deux volumes connus d'avance, à la précision de la machine : le cube, puis le cube dont on a
    # tranché un coin ( le tétraèdre retiré a trois côtés de 1/2, donc ( 1/2 )^3 / 3! ).
    c = _unit_cube( 3 )
    assert float( c.measure ) == 1

    c.cut( [ 1, 1, 1 ], 2.5 )
    assert abs( float( c.measure ) - ( 1 - 0.5 ** 3 / 6 ) ) < 1e-15

if test( "measure_5D" ):
    # rien dans le découpage n'est écrit pour la 3D : le cube unité fait 1 en toute dimension, et
    # le coin tranché suit la même formule.
    c = _unit_cube( 5 )
    assert float( c.measure ) == 1

    c.cut( [ 1, 1, 1, 1, 1 ], 4.5 )
    assert abs( float( c.measure ) - ( 1 - 0.5 ** 5 / 120 ) ) < 1e-15

if test( "measure_nd_vs_reference" ):
    # des coupes quelconques, confrontées à une référence qui n'a rien en commun avec le kernel
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
    # un plan sépare le cube en deux : les deux morceaux doivent rendre 1. Ça teste ce qu'aucune
    # comparaison à une référence ne teste vraiment -- que le découpage PAVE la cellule, sans trou
    # ni recouvrement -- et ça le teste à 1e-16.
    rng = numpy.random.default_rng( 3 )
    for d in ( 3, 4 ):
        for _ in range( 6 ):
            n = rng.normal( size = d )
            o = float( n @ rng.uniform( 0.2, 0.8, size = d ) )
            a = _unit_cube( d ); a.cut( n.tolist(), o )
            b = _unit_cube( d ); b.cut( ( -n ).tolist(), -o )
            assert abs( float( a.measure ) + float( b.measure ) - 1 ) < 1e-12

if test( "measure_nd_degenerate" ):
    # sur une cellule à sommets confondus ( plan passant par des sommets existants, cf.
    # `cut_through_existing_vertices` ) : les simplexes plats que ça produit doivent peser zéro.
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
    # L'adjoint du découpage : la triangulation est COMBINATOIRE ( elle ne bouge pas sous une
    # petite perturbation ), donc le backward rejoue le même parcours et ne dérive que les
    # déterminants. Vérifié en bout de chaîne, à travers `init_as_hypercube` puis `cut`.
    for d in ( 3, 4 ):
        origin = driver.array( numpy.zeros( d ) )
        axes   = driver.array( numpy.eye( d ) + 0.07 * numpy.arange( d * d ).reshape( d, d ) / ( d * d ) )
        normal = driver.array( numpy.linspace( 0.8, 1.3, d ) )
        offset = driver.array( float( d ) - 0.7 )

        check_grad( lambda a: Cell.make_hypercube( d, origin, a ).measure, axes )
        check_grad( lambda p, a: Cell.make_hypercube( d, p, a ).measure, origin, axes )
        check_grad( lambda p, a, n, k: Cell.make_hypercube( d, p, a ).cut( n, k ).measure,
                    origin, axes, normal, offset )

if test( "measure_nd_batched" ):
    # le scratch est PAR THREAD, pas par item : un batch plus grand que le plafond de threads fait
    # donc repasser chaque ligne d'un item à l'autre. Une ligne mal réinitialisée se verrait ici.
    nb_items = 48
    c = Cell.make_hypercube( 3, [ 0, 0, 0 ], numpy.eye( 3 ).tolist(),
                             batch_axes = [ new_batch_axis( nb_items ) ] )
    c.cut( [ 1, 1, 1 ], 2.5 )

    m = numpy.asarray( c.measure.tensor )
    assert m.shape == ( nb_items, )
    assert numpy.allclose( m, 1 - 0.5 ** 3 / 6 )
