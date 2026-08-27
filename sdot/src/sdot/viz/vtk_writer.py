"""Sortie ParaView d'un `Visualizer` : XML VTK binaire compressé.

Trois choix qui ne se devinent pas, et qui sont ici :

- FORMAT. `.vtu` (grille non structurée) en binaire ajouté (`AppendedData encoding="raw"`) et
  compressé zlib -- ce que ParaView préfère, et bien plus léger que l'ASCII. Un `.vtu` accepte
  les trois natures de cellule d'un coup : polygones, segments, points isolés.

- LE TEMPS. Une image par fichier, rassemblées par un `.pvd` qui porte les abscisses. C'est ce
  que ParaView attend pour une série temporelle dont la géométrie change complètement d'un pas à
  l'autre -- ce qui est notre cas : ni les sommets ni la connectivité ne se correspondent.

- AU-DELÀ DE LA 3D. ParaView est 3D. On écrit les trois dimensions choisies en GÉOMÉTRIE et les
  autres coordonnées en DONNÉES DE POINTS (`x3`, `x4`, ...) : ParaView peut alors couper et
  seuiller dessus lui-même, ce qui vaut mieux que de figer une coupe à l'écriture.

Un polytope donné en demi-espaces n'a pas de sommets : il est énuméré ici, en Python
(`polytope.polytope_mesh`).
"""
import struct
import zlib
from pathlib import Path

import numpy as np

from .polytope import polytope_mesh


VTK_VERTEX, VTK_LINE, VTK_POLYGON = 1, 3, 7

_VTK_DTYPE = {
    "float32": "Float32", "float64": "Float64",
    "int32": "Int32", "int64": "Int64", "uint8": "UInt8",
}


class _Appended:
    """Les tableaux du fichier, mis bout à bout dans le bloc binaire final.

    `add` rend l'OFFSET à écrire dans l'attribut du `DataArray` ; `data` rend le bloc entier.
    Chaque tableau est compressé par morceaux, avec l'en-tête que VTK attend :
    `[ nb_morceaux, taille_morceau, taille_du_dernier, taille_compressée_de_chacun ]`, tous en
    UInt64 (d'où le `header_type` du fichier).
    """
    BLOCK = 32768

    def __init__( self ):
        self._chunks = []
        self._size   = 0

    def add( self, arr ):
        raw = np.ascontiguousarray( arr ).tobytes()
        blob = self._encode( raw )
        offset = self._size
        self._chunks.append( blob )
        self._size += len( blob )
        return offset

    def _encode( self, raw ):
        if not raw:
            return struct.pack( "<Q", 0 )                  # tableau vide : zéro morceau
        parts = [ raw[ i : i + self.BLOCK ] for i in range( 0, len( raw ), self.BLOCK ) ]
        comp  = [ zlib.compress( part, 6 ) for part in parts ]
        head  = struct.pack( f"<{ 3 + len( parts ) }Q", len( parts ), self.BLOCK,
                             len( parts[ -1 ] ), *( len( c ) for c in comp ) )
        return head + b"".join( comp )

    def data( self ):
        return b"".join( self._chunks )


def _array_tag( name, arr, offset, nb_components = 1 ):
    kind = _VTK_DTYPE[ arr.dtype.name ]
    name = f' Name="{ name }"' if name else ""
    return ( f'<DataArray type="{ kind }"{ name } NumberOfComponents="{ nb_components }" '
             f'format="appended" offset="{ offset }"/>' )


def _write_vtu( path, coords, extra, cells, colors, extra_names ):
    """Un fichier `.vtu`.

    `coords` : `[n, 3]` la géométrie. `extra` : `[n, k]` les coordonnées au-delà de la 3D, nommées
    par `extra_names`. `cells` : `( types [m], connectivity, offsets )`. `colors` : `[m, 4]` uint8.
    """
    types, conn, offs = cells
    app = _Appended()
    o_pts  = app.add( coords.astype( np.float32 ) )
    o_conn = app.add( conn.astype( np.int64 ) )
    o_offs = app.add( offs.astype( np.int64 ) )
    o_type = app.add( types.astype( np.uint8 ) )
    o_col  = app.add( colors.astype( np.uint8 ) )
    o_extra = [ app.add( np.ascontiguousarray( extra[ :, k ], np.float32 ) )
                for k in range( extra.shape[ 1 ] ) ]

    xml = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64" compressor="vtkZLibDataCompressor">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{ len( coords ) }" NumberOfCells="{ len( types ) }">',
        '      <Points>',
        '        ' + _array_tag( None, coords.astype( np.float32 ), o_pts, 3 ),
        '      </Points>',
        '      <Cells>',
        '        ' + _array_tag( "connectivity", conn.astype( np.int64 ), o_conn ),
        '        ' + _array_tag( "offsets", offs.astype( np.int64 ), o_offs ),
        '        ' + _array_tag( "types", types.astype( np.uint8 ), o_type ),
        '      </Cells>',
        '      <CellData Scalars="RGBA">',
        '        ' + _array_tag( "RGBA", colors.astype( np.uint8 ), o_col, 4 ),
        '      </CellData>',
    ]
    if extra_names:
        xml.append( f'      <PointData Scalars="{ extra_names[ 0 ] }">' )
        for name, off in zip( extra_names, o_extra ):
            xml.append( '        ' + _array_tag( name, np.zeros( 0, np.float32 ), off ) )
        xml.append( '      </PointData>' )
    xml += [
        '    </Piece>',
        '  </UnstructuredGrid>',
        '  <AppendedData encoding="raw">',
        '   _',
    ]
    head = ( "\n".join( xml ) ).encode( "ascii" )
    tail = b"\n  </AppendedData>\n</VTKFile>\n"
    path.write_bytes( head + app.data() + tail )
    return path


def _frame_mesh( viz, index, axes ):
    """L'image `index`, mise à plat : sommets, cellules VTK, couleurs.

    Les sommets sont RENUMÉROTÉS : le vivier porte toute la scène, un fichier ne doit contenir que
    ce que son image utilise.
    """
    fr   = viz.frame( index )
    pool = viz.positions
    d    = viz.nb_dims
    colors = np.array( viz.colors, np.float64 ).reshape( -1, 4 )

    polys = [ list( f ) for f in fr[ "polygons" ] ]
    edges = np.asarray( fr[ "edges" ] ).reshape( -1, 2 )
    pts   = np.asarray( fr[ "points" ] ).reshape( -1 )

    used = np.unique( np.concatenate( [
        np.array( [ i for f in polys for i in f ], np.int64 ),
        edges.reshape( -1 ).astype( np.int64 ),
        pts.astype( np.int64 ) ] ) ) if ( polys or len( edges ) or len( pts ) ) \
        else np.zeros( 0, np.int64 )
    remap = { int( v ): k for k, v in enumerate( used ) }

    verts = [ pool[ used ].reshape( -1, d ) ]
    cells, rgba = [], []                      # ( type, [ indices ] ) et la couleur de la cellule

    for f, ci in zip( polys, fr[ "polygon_colors" ] ):
        cells.append( ( VTK_POLYGON, [ remap[ int( i ) ] for i in f ] ) )
        rgba.append( colors[ int( ci ) ] )
    for ( a, b ), ci in zip( edges, fr[ "edge_colors" ] ):
        cells.append( ( VTK_LINE, [ remap[ int( a ) ], remap[ int( b ) ] ] ) )
        rgba.append( colors[ int( ci ) ] )
    for v, ci in zip( pts, fr[ "point_colors" ] ):
        cells.append( ( VTK_VERTEX, [ remap[ int( v ) ] ] ) )
        rgba.append( colors[ int( ci ) ] )

    # polytopes : ils n'ont pas de sommets, on les énumère (la boîte de la scène les borne, sans
    # quoi un polytope ouvert n'aurait rien à montrer).
    bounds = viz.bounds()
    for dirs, offs, col in fr[ "polytopes" ]:
        pv, pe, pf = polytope_mesh( dirs, offs, bounds = bounds )
        if len( pv ) == 0:
            continue
        base = sum( len( v ) for v in verts )
        verts.append( pv )
        for f in pf:
            cells.append( ( VTK_POLYGON, [ base + i for i in f ] ) )
            rgba.append( np.asarray( col, np.float64 ) )
        for a, b in pe:
            cells.append( ( VTK_LINE, [ base + int( a ), base + int( b ) ] ) )
            # l'arête reprend la teinte de la face, assombrie, et toujours OPAQUE (l'opacité
            # d'une face n'a pas de sens pour un trait)
            rgba.append( np.array( [ 0.72 * col[ 0 ], 0.72 * col[ 1 ], 0.72 * col[ 2 ], 1.0 ] ) )

    allv = np.concatenate( verts, axis = 0 ) if verts else np.zeros( ( 0, d ) )
    return allv, cells, ( np.array( rgba ).reshape( -1, 4 ) * 255 ).astype( np.uint8 )


def write_vtk( viz, filename, axes = ( 0, 1, 2 ) ):
    """Voir `Visualizer.write_vtk`."""
    path = Path( filename )
    if viz.nb_dims is None:
        raise ValueError( "write_vtk: rien à écrire (aucune primitive ajoutée)" )

    d = viz.nb_dims
    axes = tuple( a for a in axes if a < d )
    rest = [ k for k in range( d ) if k not in axes ]
    extra_names = [ f"x{ k }" for k in rest ]

    def one( index, out ):
        allv, cells, rgba = _frame_mesh( viz, index, axes )
        coords = np.zeros( ( len( allv ), 3 ), np.float32 )
        for j, a in enumerate( axes ):
            coords[ :, j ] = allv[ :, a ]
        extra = ( allv[ :, rest ] if rest else np.zeros( ( len( allv ), 0 ) ) ).astype( np.float32 )

        conn = np.array( [ i for _, ids in cells for i in ids ], np.int64 )
        offs = np.cumsum( [ len( ids ) for _, ids in cells ], dtype = np.int64 )
        types = np.array( [ t for t, _ in cells ], np.uint8 )
        return _write_vtu( out, coords, extra, ( types, conn, offs ), rgba, extra_names )

    if viz.nb_frames == 1:
        return one( 0, path.with_suffix( ".vtu" ) )

    stem = path.with_suffix( "" )
    stem.parent.mkdir( parents = True, exist_ok = True )
    entries = []
    for i in range( viz.nb_frames ):
        out = Path( "%s_%04d.vtu" % ( stem, i ) )
        one( i, out )
        entries.append( ( viz.frame( i )[ "value" ], out.name ) )

    pvd = [ '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            '  <Collection>' ]
    pvd += [ f'    <DataSet timestep="{ t }" group="" part="0" file="{ name }"/>'
             for t, name in entries ]
    pvd += [ '  </Collection>', '</VTKFile>', '' ]
    out = stem.with_suffix( ".pvd" )
    out.write_text( "\n".join( pvd ) )
    return out
