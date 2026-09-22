"""L'ENVELOPPE d'un nuage, approchée par l'EXTÉRIEUR : l'intersection de `K` demi-espaces qui
s'appuient sur le nuage, un par direction.

Ce dont un transport a besoin quand rien ne borne le domaine ( des gaussiennes, dont le support est
le plan entier ) n'est pas l'enveloppe convexe exacte : elle peut avoir autant d'arêtes que de
germes ( des points sur un cercle ), et chaque cellule du diagramme est coupée par TOUS les plans
du domaine -- `n` coupes par cellule, le diagramme deviendrait quadratique. Ce qu'il lui faut est
un convexe BORNÉ qui contient tous les germes, avec peu de faces : pour `K` directions `u_k`, le
demi-espace `u_k . x <= max_i u_k . p_i + marge` s'appuie sur le nuage, et leur intersection est
un polyèdre convexe qui contient l'enveloppe, de `K` faces, à `marge` près -- les directions des
axes en font partie, donc un pavé de départ en sort ( `PowerDiagram.axis_aligned_box` ), et les
autres directions ne sont que `K - 2d` coupes par cellule.

Chaque germe est dedans ( ou sur le bord ), donc sa cellule de Voronoï y a une mesure positive :
c'est ce qui rend le départ de `OtPlan` admissible ( `otplan/Solve.h` ). Ce qui déborde du polyèdre
n'a pas de masse : le transport est celui vers la densité RESTREINTE à ce domaine.

Le maximum sur les `n` germes est un NOYAU ( un work-item par direction, là où vivent les
positions ) : rien du nuage ne passe par l'hôte. Seuls les `K` décalages en sortent -- la
description du domaine, ce que `PowerDiagram` lit de toute façon côté hôte pour poser sa cellule
de départ, comme il le fait des demi-espaces d'une image.
"""

import itertools

import numpy as np

from loom.compilation.FfiCode import FfiCodeParallel
from loom.drivers.driver import driver
from loom.tensor import Axis, CtShapeVar, RealTensor, ShapeVar, new_batch_axis
from loom.util import Aggregate


class _Support( Aggregate ):
    """un demi-espace d'appui : sa direction ( donnée ) et son décalage ( calculé ) -- batché sur
    les directions, un work-item par direction"""
    direction : RealTensor[ "dim" ]
    offset    : RealTensor
    dim       : Axis[ "nb_dims" ]
    nb_dims   : CtShapeVar


def hull_directions( nb_dims, nb_directions = None ):
    """`[ K, d ]` directions unitaires : en 2D `K` angles réguliers ( 16 par défaut, les axes
    compris ) ; au-delà, tous les vecteurs de `{ -1, 0, 1 }^d` à une ou deux composantes non
    nulles ( `2d + 4 C( d, 2 )` : 18 en 3D ), plus les coins en 3D ( 26 )."""
    d = int( nb_dims )
    if d == 1:
        return np.array( [ [ 1.0 ], [ -1.0 ] ] )
    if d == 2:
        K = int( nb_directions or 16 )
        a = 2 * np.pi * np.arange( K ) / K
        dirs = np.stack( [ np.cos( a ), np.sin( a ) ], axis = 1 )
        dirs[ np.abs( dirs ) < 1e-12 ] = 0.0            # les axes EXACTEMENT : ce que `axis_aligned_box` reconnaît
        return dirs
    dirs = []
    max_nz = 3 if d == 3 else 2
    for v in itertools.product( ( -1, 0, 1 ), repeat = d ):
        nz = sum( 1 for x in v if x )
        if 1 <= nz <= max_nz:
            dirs.append( v )
    dirs = np.array( dirs, dtype = float )
    return dirs / np.linalg.norm( dirs, axis = 1, keepdims = True )


def supporting_half_spaces( positions, nb_directions = None, margin = 0.0 ):
    """`( directions, offsets )`, les demi-espaces `direction . x <= offset` qui s'appuient sur le
    nuage `positions` ( un `Tensor` `[ n, d ]`, ou un tableau ), écartés de `margin`. Le format
    qu'attend `PowerDiagram( boundaries = ... )` : `directions` en tableau hôte ( des constantes ),
    `offsets` un `Tensor` `[ K ]` calculé par le noyau."""
    if not hasattr( positions, "raw" ):
        pos = np.asarray( positions, dtype = float ).reshape( -1, np.asarray( positions ).shape[ -1 ] )
        positions = RealTensor[ Axis( ShapeVar( pos.shape[ 0 ] ), name = "num_dirac" ), Axis( ShapeVar( pos.shape[ 1 ] ), name = "dim" ) ]( pos )
    d = int( positions.shape[ -1 ] )
    dirs = hull_directions( d, nb_directions )

    num_dir = new_batch_axis( len( dirs ), prefix = "hulldir" )
    support = _Support( nb_dims = d, direction = dirs, batch_axes = [ num_dir ] )
    driver.call(
        FfiCodeParallel( name = "supporting_half_spaces",
            fwd_code = "auto sup = support( batch_index );"
                       "using TF = DECAYED_TYPE_OF( sup.offset )::TF;"
                       "constexpr int nd = CT_VALUE( sup.nb_dims );"
                       "const SI n = SI( positions.shape( 0 ) );"
                       "TF best = 0;"
                       "for ( SI i = 0; i < n; ++i ) {"
                       "    TF s = 0;"
                       "    for ( int c = 0; c < nd; ++c ) s += TF( sup.direction( c ) ) * TF( positions( i, c ) );"
                       "    best = i == 0 || s > best ? s : best;"
                       "}"
                       "sup.offset = best + TF( margin );" ),
        output_attributes = [ "support.offset" ],
        has_dynamic_capacity = False,
        positions = positions,
        support = support,
        margin = RealTensor[ () ]( float( margin ) ),
    )
    return dirs, support.offset
