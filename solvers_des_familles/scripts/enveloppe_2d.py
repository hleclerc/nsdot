#!/usr/bin/env python
"""LE RELEVEMENT MINIMAL EN 2D, EXACT, EN UNE ENVELOPPE CONVEXE ( README § 8.6 ).

En 1D ( § 8.4 ) le relevement minimal EST une projection : admissible a marge `eps` <=> `psi =
( 1 - eps )|p|^2 - w` est convexe ( points releves ), donc la projection est l'ENVELOPPE CONVEXE
INFERIEURE de `psi`, et `w <- ( 1 - eps )|p|^2 - H( p_i )`. Elle ne touche QUE les germes dont le point
releve est au-dessus de l'enveloppe, les remonte du MINIMUM, d'un coup, sans cascade ni ping-pong.

En 2D le banc l'implemente autrement ( `--corr releve` ) : une bissection sur le poids de CHAQUE
cellule sous le plancher, le diagramme refait entre deux passes. Deux vides nes au meme sommet se
disputent la place, et ca cascade -- a `sigma = 0.005`, 27 585 relevements en 40 passes et il RESTE des
cellules vides, ce qui tue Newton ( trois vides suffisent, § 8.6 ).

Or le meme objet en 2D est simplement l'enveloppe convexe inferieure des `n` points releves DANS R^3 :
un `ConvexHull` de 10^5 points, exact, en un coup. C'est ce que fait ce script. `H` etant convexe, elle
vaut le MAX de ses morceaux affines, et l'argmax en `p_i` est la facette qui le contient.

CE QUI RESTE au banc : l'enveloppe garantit « non vide dans R^2 », pas « non vide dans le DOMAINE ».
Les cellules nees hors de `[0,1]^2` sont a finir par `--corr releve`, qui connait les bords.

    scripts/enveloppe_2d.py --depart DEP.txt --eps 1e-3 --out PROJ.txt
    multiechelle --load PROJ.txt --2d --threads 8 --lisse-solution 0 [--corr releve]
"""
import argparse, os, sys
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from triangulation_2d import lit_cas, ecrit_cas


def enveloppe_inferieure( P, z ):
    """les facettes de l'enveloppe convexe INFERIEURE de `( P, z )` ( normale sortante vers le bas )."""
    h = ConvexHull( np.column_stack( [ P, z ] ) )
    return h.simplices[ h.equations[ :, P.shape[ 1 ] ] < 0 ]


def pas_sommets_de( P, sx ):
    """les indices qui n'apparaissent dans AUCUNE facette : ils sont au-dessus de l'enveloppe."""
    return np.setdiff1d( np.arange( len( P ) ), np.unique( sx ) )


def pas_sommets( P, z ):
    """les germes dont le point releve `( p, z )` n'est PAS un sommet de l'enveloppe inferieure -- c'est
    exactement la liste des cellules vides ( dans R^2 ; le domaine est une autre affaire )."""
    return pas_sommets_de( P, enveloppe_inferieure( P, z ) )


def barycentriques( P, sx, t, Q ):
    """les coordonnees barycentriques de `Q[i]` dans le triangle `sx[ t[i] ]`."""
    V = P[ sx[ t ] ]                                           # ( m, 3, 2 )
    M = np.stack( [ V[ :, 1 ] - V[ :, 0 ], V[ :, 2 ] - V[ :, 0 ] ], axis = 2 )
    l12 = np.linalg.solve( M, ( Q - V[ :, 0 ] )[ :, :, None ] )[ :, :, 0 ]
    return np.column_stack( [ 1 - l12.sum( axis = 1 ), l12 ] )


def hauteur_enveloppe( P, psi, sx, nb, qui, pas_max = 200, tol = -1e-12 ):
    """`H( p_i )` EXACTEMENT, par LOCALISATION dans la triangulation projetee de l'enveloppe.

    L'ancienne version prenait le max des plans des `k` facettes les plus proches. `H` etant convexe,
    elle vaut le max de TOUS ses morceaux affines : tronquer le max la SOUS-estime, donc remonte TROP,
    donc une cellule remontee peut en avaler une voisine -- c'est exactement le defaut que le compte de
    vides non monotone trahissait ( 21846 -> 67 -> 85 -> ... a `eps = 0.3` ).

    Ici on cherche la facette qui CONTIENT `p_i`, par une marche de visibilite : partir de la facette
    de centre le plus proche, et tant qu'une coordonnee barycentrique est negative, passer a la facette
    voisine par l'arete opposee. La marche se termine dans une triangulation d'un convexe, et un germe
    qui n'est pas sommet est strictement a l'interieur de l'enveloppe projetee.

    Rend `H` et le nombre de germes pour lesquels la marche n'a pas conclu ( 0 attendu )."""
    Q = P[ qui ]
    cen = P[ sx ].mean( axis = 1 )
    _, t = cKDTree( cen ).query( Q, k = 1 )
    for _ in range( pas_max ):
        lam = barycentriques( P, sx, t, Q )
        j = lam.argmin( axis = 1 )
        dehors = lam[ np.arange( len( t ) ), j ] < tol
        if not dehors.any(): break
        idx = np.nonzero( dehors )[ 0 ]
        v = nb[ t[ idx ], j[ idx ] ]                           # la facette d'en face
        bouge = v >= 0                                         # -1 : voisine du dessus, on s'arrete
        if not bouge.any(): break
        t[ idx[ bouge ] ] = v[ bouge ]
    lam = barycentriques( P, sx, t, Q )
    return np.einsum( "mv,mv->m", lam, psi[ sx[ t ] ] ), int( ( lam.min( axis = 1 ) < tol ).sum() )


def miroirs( P, bande ):
    """LE DOMAINE. L'enveloppe garantit « non vide dans R^2 » ; pour que ce soit « non vide dans
    `[0,1]^2` » il suffit que la cellule TIENNE dans le domaine, et l'image d'un germe a travers une
    paroi fait de cette paroi son bissecteur EXACT ( meme poids ). On ne double donc que les germes a
    moins de `bande` d'une paroi. Rend les positions images et l'indice de leur original."""
    img, src = [], []
    for d in range( P.shape[ 1 ] ):
        for mur in ( 0.0, 1.0 ):
            pres = np.nonzero( np.abs( P[ :, d ] - mur ) < bande )[ 0 ]
            Q = P[ pres ].copy()
            Q[ :, d ] = 2 * mur - Q[ :, d ]
            img.append( Q ); src.append( pres )
    return np.concatenate( img ), np.concatenate( src )


def projette( P, w, eps, passes = 6, bande = 0.0 ):
    """`w <- ( 1 - eps )|p|^2 - H( p )`. Les germes qui sont DEJA sommets de l'enveloppe ont `H = psi`
    exactement et ne bougent pas ; seuls les autres demandent une evaluation. Avec `H` exacte la
    projection converge en UNE passe ( 21846 -> 0 a `eps = 0.3` ) ; on boucle quand meme, parce
    qu'avec les miroirs le poids d'une image SUIT son original, donc l'enveloppe change.

    ATTENTION a ce que ca vaut : la projection est CORRECTE, et les departs qu'elle produit sont PIRES
    pour Newton que ceux de la version fausse ( README § 8.7.4 ). Le relevement exactement minimal fait
    naitre chaque cellule a l'aire la plus petite possible ; ce que Newton veut est une AIRE, que `eps`
    ne controle pas. A essayer : `+ delta h_i^2` sur les germes remontes."""
    n = len( P )
    if bande > 0:
        Q, src = miroirs( P, bande )
        A = np.vstack( [ P, Q ] )
    else:
        Q, src, A = None, None, P
    a2 = ( A * A ).sum( axis = 1 )
    w = w.copy()
    bilan = []
    for it in range( passes + 1 ):
        wa = w if src is None else np.concatenate( [ w, w[ src ] ] )
        # LA MESURE porte sur le VRAI releve `|p|^2 - w` : c'est lui dont la convexite stricte dit
        # qu'aucune cellule n'est vide. Apres projection le point est exactement SUR l'enveloppe de
        # `psi`, donc jamais sommet de CELLE-LA -- la mesurer sur `psi` ne dirait rien.
        pas_som = pas_sommets( A, a2 - wa )
        pas_som = pas_som[ pas_som < n ]                        # les images ne sont pas des cellules
        bilan.append( len( pas_som ) )
        if len( pas_som ) == 0 or it == passes: break
        psi = ( 1 - eps ) * a2 - wa
        h = ConvexHull( np.column_stack( [ A, psi ] ) )
        bas = h.equations[ :, 2 ] < 0
        sx = h.simplices[ bas ]
        # les voisines, RENUMEROTEES sur les seules facettes du dessous ( -1 = voisine du dessus )
        ren = np.full( len( h.simplices ), -1, dtype = int )
        ren[ np.nonzero( bas )[ 0 ] ] = np.arange( len( sx ) )
        nb = np.where( h.neighbors[ bas ] >= 0, ren[ h.neighbors[ bas ] ], -1 )
        au_dessus = pas_sommets_de( A, sx )
        au_dessus = au_dessus[ au_dessus < n ]
        if len( au_dessus ) == 0: break
        H, perdus = hauteur_enveloppe( A, psi, sx, nb, au_dessus )
        if perdus: print( "  ( %d germes dont la marche n'a pas conclu )" % perdus )
        H = np.minimum( H, psi[ au_dessus ] )
        w[ au_dessus ] = ( 1 - eps ) * a2[ au_dessus ] - H
    return w, bilan, len( A )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "--depart", required = True )
    ap.add_argument( "--eps", type = float, default = 1e-3 )
    ap.add_argument( "--passes", type = int, default = 6 )
    ap.add_argument( "--delta", type = float, default = 0.0, help = "gonflage : + delta h_i^2 sur les germes remontes, puis re-projection" )
    ap.add_argument( "--gonfle", type = int, default = 1, help = "tours de gonflage" )
    ap.add_argument( "--bande", type = float, default = 0.0, help = "miroirs : largeur de la bande le long des parois ( 0 : pas de miroir )" )
    ap.add_argument( "--out", required = True )
    o = ap.parse_args()

    P, w = lit_cas( o.depart )
    w1, bilan, nf = projette( P, w, o.eps, o.passes, o.bande )

    # ---- LE GONFLAGE : `+ delta h_i^2` sur les germes remontes, puis RE-PROJECTION.
    #
    # Pas de garantie : abaisser le point releve de `i` le rend plus extremal, mais peut faire passer
    # un VOISIN au-dessus de l'enveloppe. Ce qui se recupere, c'est l'admissibilite APRES coup -- la
    # re-projection la restaure toujours ( elle ne fait que remonter ). Le risque est donc que le
    # gonflage SE PROPAGE : c'est ce que la colonne « remontes » mesure d'un tour a l'autre.
    if o.delta > 0:
        h = cKDTree( P ).query( P, k = 2 )[ 0 ][ :, 1 ]       # l'echelle locale : le plus proche voisin
        for tour in range( o.gonfle ):
            m = w1 > w
            w1 = w1.copy(); w1[ m ] += o.delta * h[ m ] ** 2
            w1, b2, _ = projette( P, w1, o.eps, o.passes, o.bande )
            print( "  gonflage %d : delta = %g sur %d germes ; vides par passe : %s"
                   % ( tour + 1, o.delta, int( m.sum() ), " -> ".join( str( b ) for b in b2 ) ) )
    d = w1 - w
    remontes = int( ( d > 0 ).sum() )
    print( "n = %d ( %d avec les miroirs ) ; eps = %g : cellules VIDES dans R^2 par passe : %s"
           % ( len( P ), nf, o.eps, " -> ".join( str( b ) for b in bilan ) ) )
    print( "  %d germes remontes ( %.2f %% ), hausse mediane %.2e, max %.2e"
           % ( remontes, 100 * remontes / len( P ), np.median( d[ d > 0 ] ) if remontes else 0, d.max() ) )
    ecrit_cas( o.out, P, w1,
               "# relevement minimal par l'enveloppe convexe inferieure ( scripts/enveloppe_2d.py )\n"
               "# depart : %s ; eps = %g ; passes %s ; %d germes remontes\n"
               "# format : n, puis n lignes « x y w »\n" % ( os.path.basename( o.depart ), o.eps, bilan, remontes ) )
    print( "  -> %s" % o.out )


if __name__ == "__main__":
    main()
