#!/usr/bin/env python
"""LA TRIANGULATION GROSSIERE COMME OBJET PORTEUR ( 1D ) : une prolongation a GARANTIE.

Le point de depart est la caracterisation exacte de l'admissibilite : la cellule de `i` est non vide
SSI le point releve `( p_i, u_i )`, `u_i = |p_i|^2 - w_i`, est un SOMMET de l'enveloppe convexe
inferieure. Autrement dit `w` est admissible ssi `u` est strictement convexe au sens discret. La
contrainte qui tue une cellule est donc une inegalite a `D+1` points -- un SIMPLEXE -- et non une
inegalite de cellule : le bon objet local du niveau grossier est sa TRIANGULATION REGULIERE ( le dual
du diagramme de puissance grossier ), pas ses cellules.

LA CONSTRUCTION. Soient `P_q, W_q` le niveau grossier, `U_q = |P_q|^2 - W_q` son releve, et `Uh` son
enveloppe convexe inferieure ( affine par simplexe de la triangulation reguliere ). Soit `Q` l'interpole
affine de `|P|^2` sur la MEME triangulation, et `g = Q - |p|^2 >= 0` l'ecart au paraboloide ( nul aux
germes grossiers, strictement concave dans chaque simplexe ). On pose

    u = Uh - c g        c'est-a-dire      w = ( 1 - c )( |p|^2 - Uh ) + c * w_tilde

ou `w_tilde = interpolation barycentrique des poids grossiers` ( on retrouve `c = 1` ). Alors :

  * DANS chaque simplexe, `u = affine - c ( affine - |p|^2 ) = affine + c |p|^2` est STRICTEMENT convexe
    des que `c > 0` : tous les germes fins d'un meme simplexe grossier sont automatiquement des sommets,
    sans rien resoudre. Toute cellule vide vient donc d'une FACETTE de la triangulation grossiere ;
  * a une facette `F`, le pli de `u` vaut `pli_F( Uh ) - c * pli_F( Q )`, avec `pli_F( Uh ) >= 0` puisque
    `Uh` est convexe. La convexite globale est donc garantie pour tout

        0 < c <= c* = min over F, pli_F( Q ) > 0 of  pli_F( Uh ) / pli_F( Q )

    -- une borne CALCULABLE SUR LE NIVEAU GROSSIER SEUL, qui ne coute rien, et qui donne ZERO cellule
    vide sans relevement, sans bissection, sans cascade.

POURQUOI `c = 1` ( l'interpolation barycentrique nue ) NE PEUT PAS marcher, et c'est le mecanisme de
l'echec de `--prol harmonique` du § 8.1 : a un germe grossier la marge d'une cellule fine vaut
`m = 1 - pli / ( h_l + h_r )` avec `h` l'espacement FIN et `pli ~ 2H` l'echelle GROSSIERE, soit
`m ~ 1 - H/h ~ 1 - R^(1/D)` -- les marges de -1 a -100 mesurees en 1D au § 8.3. Interpoler les POIDS
grossiers impose un pli a l'echelle grossiere la ou les germes sont espaces de `h`.

Usage : scripts/triangulation_1d.py [-n 400 --sigma 0.005 -R 8]
"""
import argparse, os, sys
import numpy as np

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
import multiechelle_1d as me
from adoucissement_1d import enveloppe, bilan
from frontieres_1d import niveau_grossier, corr_diag

# --------------------------------------------------------------- l'interpole affine sur la triangulation

def pl( x, xc, yc ):
    """interpolation affine par morceaux, PROLONGEE affinement au-dela des extremes ( pas de clamp :
    hors de l'enveloppe des germes grossiers le simplexe du bord est etendu, et `u = affine + c|p|^2`
    y est encore strictement convexe )."""
    i = np.clip( np.searchsorted( xc, x ) - 1, 0, len( xc ) - 2 )
    t = ( x - xc[ i ] ) / ( xc[ i + 1 ] - xc[ i ] )
    return yc[ i ] * ( 1 - t ) + yc[ i + 1 ] * t


def prolonge_triangulation( p, pc, wc, c ):
    """`w = ( 1 - c )( |p|^2 - Uh ) + c * w_tilde`, par `u = Uh - c g`."""
    U  = pc * pc - wc
    Uh = pl( p, pc, U )
    Q  = pl( p, pc, pc * pc )
    return p * p - ( Uh - c * ( Q - p * p ) )


def c_etoile( pc, wc ):
    """`c* = min_F pli_F( Uh ) / pli_F( Q )` sur les facettes ou `pli_F( Q ) > 0`. En 1D les facettes
    sont les germes grossiers interieurs, `pli_F( Q ) = P_{q+1} - P_{q-1} > 0` toujours."""
    U  = pc * pc - wc
    sU = np.diff( U ) / np.diff( pc )
    sQ = np.diff( pc * pc ) / np.diff( pc )                  # = P_q + P_{q+1}
    kU = sU[ 1: ] - sU[ :-1 ]                                # pli de Uh, >= 0 ( convexe )
    kQ = sQ[ 1: ] - sQ[ :-1 ]                                # = P_{q+1} - P_{q-1}
    ok = kQ > 0
    r  = kU[ ok ] / kQ[ ok ]
    return float( r.min() ), kU, kQ

# --------------------------------------------------------------- le banc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "-n", type = int, default = 400 )
    ap.add_argument( "--sigma", type = float, default = 0.005 )
    ap.add_argument( "-R", type = int, default = 8 )
    ap.add_argument( "--graine", type = int, default = 0 )
    ap.add_argument( "--eps", type = float, default = 0.1 )
    o = ap.parse_args()

    n, R = o.n, o.R
    p, _ = me.nuage( n, o.sigma, o.graine )
    nu = np.full( n, 1 / n )
    avor, _, _ = me.cellules( p, np.zeros( n ) )
    print( "n = %d, sigma = %g, R = %d" % ( n, o.sigma, R ) )
    print( "  Voronoi : %d vides, rmax %.2f, l2 %.3f ; compression fine min %.3f" % ( bilan( p, np.zeros( n ), nu ) + ( ( nu / avor ).min(), ) ) )

    k, reps = me.paquets( n, R )
    pc, nuc, wc, ac, B, itc = niveau_grossier( p, k, reps )
    cet, kU, kQ = c_etoile( pc, wc )
    Sc = nuc / me.cellules( pc, np.zeros( len( pc ) ) )[ 0 ]
    print( "  niveau grossier : n = %d ; compression grossiere min %.3f ; c* = %.4f" % ( len( pc ), Sc.min(), cet ) )
    print( "  plis : pli( Uh ) dans [ %.2e, %.2e ], pli( Q ) dans [ %.2e, %.2e ]" % ( kU.min(), kU.max(), kQ.min(), kQ.max() ) )

    print( "\n-- le balayage de c : la borne c* est-elle la bonne, et est-elle serree ?" )
    print( "   %-22s | vides | rmax     | l2     | marge min" % "c" )
    for etiq, c in [ ( "0.1 c*", 0.1 * cet ), ( "0.5 c*", 0.5 * cet ), ( "c*", cet ),
                     ( "1.05 c*", 1.05 * cet ), ( "1.5 c*", 1.5 * cet ), ( "3 c*", 3 * cet ),
                     ( "0.5", 0.5 ), ( "1 ( barycentrique )", 1.0 ) ]:
        w = prolonge_triangulation( p, pc, wc, c )
        m = me.marge( p, w )
        print( "   %-22s | %5d | %8.2f | %6.3f | %+9.2f" % ( ( "%s = %.4f" % ( etiq, c ), ) + bilan( p, w, nu ) + ( np.nanmin( m ), ) ) )

    print( "\n-- a c = c*, contre tout ce qui est connu ( residu l2 ; Voronoi = 1.422 )" )
    w_bary = prolonge_triangulation( p, pc, wc, 1.0 )
    connus = [ ( "Voronoi", np.zeros( n ) ),
               ( "harmonique ( = barycentrique en 1D )", me.prolonge( "harmonique", p, pc, wc, k ) ),
               ( "barycentrique par la triangulation", w_bary ),
               ( "spline", me.prolonge( "spline", p, pc, wc, k ) ),
               ( "spline + enveloppe", enveloppe( p, me.prolonge( "spline", p, pc, wc, k ), o.eps )[ 0 ] ),
               ( "triangulation c = c*", prolonge_triangulation( p, pc, wc, cet ) ),
               ( "triangulation c = c* + diag 1/2", corr_diag( p, prolonge_triangulation( p, pc, wc, cet ), nu, 0.5 ) ),
               ( "triangulation c = c* + enveloppe", enveloppe( p, prolonge_triangulation( p, pc, wc, cet ), o.eps )[ 0 ] ) ]
    for nom, w in connus:
        v, rmax, r2 = bilan( p, w, nu )
        adm = "admissible" if v == 0 else "%d vides" % v
        try:
            _, its, diag = me.newton( p, nu, w )
            nw = "%d it, %d diag" % ( its, diag )
        except AssertionError:
            nw = "-"
        print( "   %-36s : %-12s rmax %8.2f, l2 %6.3f | Newton : %s" % ( nom, adm, rmax, r2, nw ) )

    print( "\n-- ou sont les vides de c = 1, et est-ce bien AUX FACETTES ( les germes grossiers ) ?" )
    a1, _, _ = me.cellules( p, w_bary )
    vides = np.nonzero( a1 <= 0 )[ 0 ]
    est_rep = np.isin( vides, reps )
    voisin_rep = np.isin( vides, np.concatenate( [ reps, reps + 1, reps - 1 ] ) )
    print( "   %d vides : %d sont des germes grossiers, %d sont a un germe pres ( %d representants en tout )"
           % ( len( vides ), int( est_rep.sum() ), int( voisin_rep.sum() ), len( reps ) ) )
    m1 = me.marge( p, w_bary )
    interieurs = reps[ ( reps > 0 ) & ( reps < n - 1 ) ]
    print( "   marge aux germes grossiers interieurs : min %.1f, mediane %.2f ( ailleurs : min %.2f )"
           % ( np.nanmin( m1[ interieurs ] ), np.nanmedian( m1[ interieurs ] ),
               np.nanmin( np.delete( m1, interieurs ) ) ) )
    print( "   ( le pli de `w_tilde` est a l'echelle GROSSIERE la ou les germes sont espaces de h :" )
    print( "     la marge y plonge comme `1 - O( H/h ) = 1 - O( R^(1/D) )`, d'ou les -100 )" )


if __name__ == "__main__":
    main()
