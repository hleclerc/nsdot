#!/usr/bin/env python
"""L'ALGORITHME COMPLET, EN GEOMETRIE EXACTE : agregats, sur-cellules, table, cellules.

Tout ce qui precede etait mesure sur GRILLE. Ici tout est en polygones : coupe par demi-plan,
test de separation par axe, aire par lacet. Le but est le compteur qui manquait -- le nombre de
coupes REELLEMENT tentees quand le juge de paix travaille sur la cellule EN COURS et non sur la
cellule finale.

= LES QUATRE ETAPES

  1. VORONOI GROSSIER. Un germe sur `rho` dans la permutation du BSP ( stratifie, gratuit ), chaque
     germe au site le plus proche. On garde par agregat : les membres ( position, index, poids ),
     la cellule grossiere et son adjacence, et le RESSERREMENT -- le k-DOP des membres, pas la
     cellule grossiere, qui est bien plus large.

  2. SUR-CELLULES. Avec le majorant affine `w_i <= a.p_i + b` et le k-DOP `K` de rayon d'arete
     `L`, on pose `omega = max L^2 / 4` et

         S_A  =  ( K - a/2 )  U  ( U_v  Lag( v, a.v + b + omega ) )

     ou `Lag` est une cellule de puissance ORDINAIRE, calculee contre les SITES GROSSIERS des
     autres agregats. Le pseudo-germe au sommet `v` porte donc simplement le majorant affine
     EVALUE EN `v`, releve de `omega`. Que des demi-plans : `clip` suffit.

  3. LA TABLE. Quelle sur-cellule deborde sur quelle sur-cellule : k-DOP en phase large, puis
     separation par axe entre morceaux convexes. Un contact TANGENT compte comme un chevauchement
     ( sinon on perd de vrais voisins -- c'est la quatrieme fois que ce defaut se presente ).

  4. LES CELLULES. Pour chaque germe : L'AMORCE par les sites grossiers, puis les membres de son
     propre agregat, puis, agregat candidat par agregat candidat pris du plus proche au plus loin,
     LE JUGE DE PAIX -- la sur-cellule de `B` rencontre-t-elle la cellule EN COURS ? Si non, ses
     `|B|` germes ne sont jamais presentes a la coupe.

= POURQUOI L'AMORCE EST OBLIGATOIRE, ET CE QU'ELLE REPARE

Sans elle, le schema est exact pour tout germe dont la cellule est NON VIDE : `j` donne une
facette a `Lag_i` => `Lag_i` et `Lag_j` se touchent => `S_A` et `S_B` aussi => `B` est dans la
table. Mais un germe DOMINE ( poids trop faible, cellule vide ) n'a aucune facette, l'implication
est vide, et rien ne force sa cellule a s'effondrer : l'algorithme lui rend une cellule fantome.
Mesure : 6 germes sur 2000 sur le nuage a poids aleatoires, aucun sur les trois autres, et une
somme des aires a 1.356 au lieu de 1.

L'amorce ferme le trou, et ca se demontre. Si `x` survit aux coupes des sites grossiers alors
`h_i( x ) <= psi_S( x )`, donc `f_A( x ) <= psi_A( x ) <= h_i( x ) <= psi_S( x )` : `x` est dans
`S_A`. Si en plus `x` n'est pas dans `Lag_i`, un germe `j` de `B` verifie `h_j( x ) < h_i( x )`,
donc `f_B( x ) <= h_j( x ) < psi_S( x )` : `x` est aussi dans `S_B`, donc `B` est dans la table,
donc `j` a ete presente -- contradiction. **La correction vaut pour les cellules vides comme pour
les autres**, et elle ne coute que quelques coupes : les sites sont `n / rho`, et le critere
d'anneau arrete la liste tout de suite.

= CE QUI EST VERIFIE

  * chaque cellule finale contre la REFERENCE ( triangulation reguliere par enveloppe convexe
    relevee ), aire par aire ;
  * la somme des aires vaut l'aire du domaine ;
  * le CERTIFICAT lui-meme : chaque sommet de chaque vraie cellule est dans la sur-cellule de son
    agregat. C'est la propriete dont tout le reste depend, et elle est testee et non supposee.
"""

import argparse, os, sys, time
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
from surcell2d import charge, uniforme, ordre_median, majorant_affine

# Le TRAVAIL, et pas seulement le nombre d'appels : un test sur deux enveloppes de 8 sommets n'a
# pas le prix d'un test sur deux morceaux de 5. On compte les operations reellement faites --
# `2 |P| ( |P| + |Q| )` par orientation pour la separation, `4 |poly|` par coupe.
TRAVAIL = dict( sat = 0, clip = 0 )

EPS = 1e-12
DOM = np.array( [ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 1.0, 1.0 ], [ 0.0, 1.0 ] ] )

# ------------------------------------------------------------------- les primitives

def clip( poly, a, c ):
    """`poly ∩ { x : a.x <= c }`. L'intersection est ANCREE sur le sommet DEDANS -- avec `s0 = 0`
    elle rend exactement ce sommet, ce que la forme symetrique ne fait pas en flottant. C'est le
    correctif que `Cell::cut` a deja recu."""
    TRAVAIL[ "clip" ] += 4 * len( poly )
    if len( poly ) == 0:
        return poly
    s = poly @ a - c
    dedans = s <= 0
    if dedans.all():
        return poly
    if not dedans.any():
        return poly[ :0 ]
    out, m = [], len( poly )
    for i in range( m ):
        j = ( i + 1 ) % m
        if dedans[ i ]:
            out.append( poly[ i ] )
        if dedans[ i ] != dedans[ j ]:
            t = s[ i ] / ( s[ i ] - s[ j ] )
            out.append( poly[ i ] + t * ( poly[ j ] - poly[ i ] ) )
    return np.array( out )

def bisec( p, wp, q, wq ):
    """Le demi-plan ou `p` GAGNE contre `q` : `|x-p|^2 - wp <= |x-q|^2 - wq`."""
    return 2 * ( q - p ), float( q @ q - p @ p - wq + wp )

def aire( poly ):
    if len( poly ) < 3:
        return 0.0
    x, y = poly[ :, 0 ], poly[ :, 1 ]
    return 0.5 * abs( float( np.dot( x, np.roll( y, -1 ) ) - np.dot( y, np.roll( x, -1 ) ) ) )

def cellule( p, w, Q, WQ, depart = DOM ):
    """La cellule de puissance de `( p, w )` contre les germes `Q`, dans `depart`."""
    c = depart
    for q, wq in zip( Q, WQ ):
        a, cc = bisec( p, w, q, wq )
        c = clip( c, a, cc )
        if len( c ) < 3:
            return c[ :0 ]
    return c

def disjoints( A, B, eps = 1e-11 ):
    """Separation par axe. Un contact TANGENT n'est PAS une separation : il faut un vrai jour,
    sinon deux sur-cellules qui se touchent le long d'une arete sont declarees disjointes et on
    perd de vrais voisins."""
    if len( A ) < 3 or len( B ) < 3:
        return True
    for P, Q in ( ( A, B ), ( B, A ) ):
        e = np.roll( P, -1, 0 ) - P
        N = np.column_stack( [ -e[ :, 1 ], e[ :, 0 ] ] )
        # LES DEUX SENS sur chaque axe. N'en tester qu'un revient a n'essayer que la moitie des
        # normales -- le test reste SUR ( il ne separe que sur une vraie separation ) mais il est
        # INCOMPLET, et une table batie dessus est gonflee de fausses paires. Tester les deux sens
        # rend aussi le predicat independant de l'orientation des polygones.
        TRAVAIL[ "sat" ] += 2 * len( P ) * ( len( P ) + len( Q ) )
        pq, pp = Q @ N.T, P @ N.T
        if ( ( pq.min( 0 ) > pp.max( 0 ) + eps ) | ( pp.min( 0 ) > pq.max( 0 ) + eps ) ).any():
            return True
    return False

def kdop( polys, U ):
    """Les appuis d'une reunion de polygones dans les directions `U`."""
    v = [ ( p @ U.T ).max( 0 ) for p in polys if len( p ) ]
    return np.max( v, 0 ) if v else None

# ------------------------------------------------------------------- LA REFERENCE

def reference( P, W ):
    """Les cellules exactes, par la TRIANGULATION REGULIERE : l'enveloppe convexe des points
    releves `( p, |p|^2 - w )` ; ses facettes du BAS donnent les voisins, et la cellule d'un germe
    est determinee par ses seuls voisins. Un germe absent de l'enveloppe basse est domine, sa
    cellule est vide."""
    L = np.column_stack( [ P, ( P * P ).sum( 1 ) - W ] )
    h = ConvexHull( L )
    vois = [ set() for _ in range( len( P ) ) ]
    vus = set()
    for eq, sx in zip( h.equations, h.simplices ):
        if eq[ 2 ] < -1e-12:                       # facette tournee vers le bas
            for i in sx:
                vus.add( int( i ) )
                for j in sx:
                    if i != j:
                        vois[ i ].add( int( j ) )
    out = []
    for i in range( len( P ) ):
        if i not in vus:
            out.append( np.zeros( ( 0, 2 ) ) ); continue
        v = sorted( vois[ i ] )
        out.append( cellule( P[ i ], W[ i ], P[ v ], W[ v ] ) )
    return out

# ------------------------------------------------------- 1. LE VORONOI GROSSIER

class Agregat:
    __slots__ = ( "idx", "P", "W", "site", "V", "om", "a", "b", "Kd", "morceaux", "tab",
                  "kd", "cen" )

def etape1( P, W, rho, K = 8 ):
    """Sites = un germe sur `rho` dans l'ordre du BSP. Chaque germe au site le plus proche. Puis
    le RESSERREMENT : le k-DOP des membres, qui est bien plus serre que la cellule grossiere."""
    o = ordre_median( P )
    sites = np.sort( o[ :: rho ] )
    C = P[ sites ]
    lab = np.empty( len( P ), np.int64 )
    for i in range( 0, len( P ), 4096 ):
        d = ( ( P[ i : i + 4096, None, : ] - C[ None ] ) ** 2 ).sum( -1 )
        lab[ i : i + 4096 ] = d.argmin( 1 )
    th = np.arange( K ) * 2 * np.pi / K
    U = np.column_stack( [ np.cos( th ), np.sin( th ) ] )

    ags = []
    for c in range( len( sites ) ):
        m = np.flatnonzero( lab == c )
        A = Agregat()
        A.idx, A.P, A.W, A.site = m, P[ m ], W[ m ], int( sites[ c ] )
        A.a, A.b = majorant_affine( A.P, A.W )
        # le k-DOP des MEMBRES, obtenu en rognant un grand carre par ses K demi-plans
        h = ( A.P @ U.T ).max( 0 )
        V = DOM * 0 + np.array( [ [ -9, -9 ], [ 9, -9 ], [ 9, 9 ], [ -9, 9 ] ] ) + A.P.mean( 0 )
        for u, hh in zip( U, h ):
            V = clip( V, u, hh )
        A.V = V if len( V ) >= 3 else A.P
        L = np.linalg.norm( np.roll( A.V, -1, 0 ) - A.V, axis = 1 ) if len( A.V ) >= 3 else np.zeros( 1 )
        A.om = float( ( L * L ).max() / 4 )
        A.cen = A.P.mean( 0 )
        ags.append( A )
    return ags, sites, lab, U

# ------------------------------------------------------ 2. LES SUR-CELLULES

def etape2( ags, sites, P, W, U ):
    """`S_A = ( K - a/2 ) U ( U_v Lag( v, a.v + b + omega ) )`, contre les SITES des autres
    agregats. Le pseudo-germe en `v` porte le majorant affine EVALUE EN `v`, releve de `omega` :

        f_v( x ) = |x + a/2 - v|^2 - a.x - |a|^2/4 - b - omega  =  |x - v|^2 - ( a.v + b + omega )

    donc rien de nouveau a coder -- c'est une cellule de puissance ordinaire."""
    SP, SW = P[ sites ], W[ sites ]
    nb_coupes = 0
    for c, A in enumerate( ags ):
        garde = np.ones( len( sites ), bool ); garde[ c ] = False
        Q, WQ = SP[ garde ], SW[ garde ]
        d = np.linalg.norm( Q - A.cen, axis = 1 )
        o = np.argsort( d )
        Q, WQ, d = Q[ o ], WQ[ o ], d[ o ]
        wsuf = np.maximum.accumulate( WQ[ ::-1 ] )[ ::-1 ]      # majorant des poids restants

        A.morceaux = [ A.V - A.a / 2 ] if len( A.V ) >= 3 else []
        for v in ( A.V if len( A.V ) >= 3 else A.P ):
            wv = float( A.a @ v + A.b + A.om )
            cell = DOM
            for k in range( len( Q ) ):
                # le critere d'ANNEAU du banc : au-dela, plus rien ne peut couper
                R = float( np.abs( cell - v ).max() ) if len( cell ) else 0.0
                dk = d[ k ] - np.linalg.norm( v - A.cen )
                if dk >= R and dk * dk - 2 * dk * R + wv - wsuf[ k ] > 0:
                    break
                a_, c_ = bisec( v, wv, Q[ k ], WQ[ k ] )
                cell = clip( cell, a_, c_ )
                nb_coupes += 1
                if len( cell ) < 3:
                    break
            if len( cell ) >= 3:
                A.morceaux.append( cell )
        A.tab = A.morceaux                                 # ce que lit la TABLE ( etape 3 )
        A.kd = kdop( A.morceaux, U )
    return nb_coupes

# ------------------------------------------------ 2 bis. LA CONVEXIFICATION EXTERIEURE
#
# `S_A` est une REUNION de `M` convexes, donc un test `S_A ^ S_B` coute `M x M'` separations. La
# proposition est de remplacer la reunion par son ENVELOPPE CONVEXE : UN test par paire.
#
# A NE PAS CONFONDRE avec `--convexe` plus bas. Celui-la remplace la reunion par une INTERSECTION
# de demi-plans, qui est contenue dedans : c'est un pari, et il est perdu. Ici l'enveloppe CONTIENT
# la reunion, donc la couverture ne peut que grandir : l'algorithme reste exact par construction,
# et le seul prix est la DENSITE DE LA TABLE. C'est ca qu'on mesure.
#
# Le cout de construction est lui aussi mesure : `sommets_in -> sommets_out`. Les sommets d'entree
# sont deja la ( ce sont les cellules de l'etape 2 ), donc il ne reste qu'une enveloppe convexe sur
# quelques dizaines de points -- et en 3D, sur quelques centaines.

def convexifie( ags, juge_aussi = True ):
    nin = nout = 0
    for A in ags:
        if not A.morceaux:
            continue
        X = np.vstack( A.morceaux )
        nin += len( X )
        try:
            env = X[ ConvexHull( X ).vertices ]
        except Exception:                                  # degenere ( aligne, ou trop peu )
            env = X
        nout += len( env )
        A.tab = [ env ]
        if juge_aussi:
            A.morceaux = [ env ]
    return nin, nout

# ------------------------------------------------------ 3. LA TABLE DE DEBORDEMENT

def etape3( ags, U ):
    """ATTENTION : la phase large suppose les directions OPPOSEES DEUX A DEUX ( `u_{d+K/2} =
    -u_d` ), ce qui exige `K` PAIR. Avec `K` impair elle declare disjoints des agregats qui se
    touchent, la table se vide et les aires sont fausses SANS que le certificat le voie -- c'est
    le meme defaut que les indices `0, 2, 4, 6` codes en dur du journal.

    Phase large sur les k-DOP ( `sup_i( u ) + sup_{i+K/2}( -u ) < 0` => disjoints ), puis
    separation par axe morceau contre morceau."""
    assert len( U ) % 2 == 0, "k-DOP : nombre de directions PAIR ( voir l'en-tete )"
    n = len( ags )
    KD = np.array( [ A.kd for A in ags ] )
    K2 = len( U ) // 2
    table, nsat, nlarge = [ ], 0, 0
    for i in range( n ):
        # large : disjoints des que, dans une direction, l'appui de i et celui de j opposé se croisent
        sep = ( KD[ i ][ None, : ] + KD[ :, np.roll( np.arange( len( U ) ), K2 ) ] < -1e-11 ).any( 1 )
        cand = np.flatnonzero( ~sep )
        nlarge += len( cand )
        v = [ ]
        for j in cand:
            if j == i:
                continue
            ok = False
            for a_ in ags[ i ].tab:
                for b_ in ags[ j ].tab:
                    nsat += 1
                    if not disjoints( a_, b_ ):
                        ok = True; break
                if ok: break
            if ok:
                v.append( int( j ) )
        table.append( v )
    return table, nsat, nlarge

# ------------------------------------------------------ 4. LES CELLULES FINALES

def anneau( cell, p, w, d, R0, wsuf, k ):
    """Le critere d'arret du banc : `d >= R` et `d^2 - 2 d R + w - wm > 0` => plus rien ne peut
    couper. `R` majore `max |x - p|` sur la cellule, `wm` majore les poids restants."""
    if len( cell ) < 3:
        return True
    R = float( np.abs( cell - p ).max() )
    dk = d - R0
    return dk >= R and dk * dk - 2 * dk * R + w - wsuf > 0

def etape4( ags, table, P, W, sites, juge = True, amorce = True ):
    cells = [ None ] * len( P )
    st = dict( tentees = 0, effectives = 0, juges = 0, retenus = 0, amorce = 0 )
    SP, SW = P[ sites ], W[ sites ]
    for c, A in enumerate( ags ):
        vois = sorted( table[ c ],
                       key = lambda j: float( np.linalg.norm( ags[ j ].cen - A.cen ) ) )
        for k, i in enumerate( A.idx ):
            cell = DOM
            if amorce:                                        # 0. LES SITES GROSSIERS
                d = np.linalg.norm( SP - P[ i ], axis = 1 )
                o = np.argsort( d )
                wsuf = np.maximum.accumulate( SW[ o ][ ::-1 ] )[ ::-1 ]
                for t in range( len( o ) ):
                    if anneau( cell, P[ i ], W[ i ], d[ o[ t ] ], 0.0, wsuf[ t ], t ):
                        break
                    j = int( sites[ o[ t ] ] )
                    if j == i: continue
                    a_, c_ = bisec( P[ i ], W[ i ], P[ j ], W[ j ] )
                    nv = clip( cell, a_, c_ )
                    st[ "amorce" ] += 1
                    st[ "effectives" ] += not np.array_equal( nv, cell )
                    cell = nv
                    if len( cell ) < 3: break
            for l, j in enumerate( A.idx ):                   # 1. son propre agregat
                if j == i: continue
                a_, c_ = bisec( P[ i ], W[ i ], P[ j ], W[ j ] )
                nv = clip( cell, a_, c_ )
                st[ "tentees" ] += 1
                st[ "effectives" ] += len( nv ) != len( cell ) or not np.array_equal( nv, cell )
                cell = nv
                if len( cell ) < 3: break
            for j in vois:                                     # 2. les agregats candidats
                if len( cell ) < 3: break
                if juge:
                    st[ "juges" ] += 1
                    if all( disjoints( cell, m ) for m in ags[ j ].morceaux ):
                        continue
                    st[ "retenus" ] += 1
                for q in ags[ j ].idx:
                    a_, c_ = bisec( P[ i ], W[ i ], P[ q ], W[ q ] )
                    nv = clip( cell, a_, c_ )
                    st[ "tentees" ] += 1
                    st[ "effectives" ] += len( nv ) != len( cell ) or not np.array_equal( nv, cell )
                    cell = nv
                    if len( cell ) < 3: break
            cells[ i ] = cell if len( cell ) >= 3 else np.zeros( ( 0, 2 ) )
    return cells, st

# ------------------------------------------------------------------ LES VERIFICATIONS

def dedans( poly, x, eps = 1e-9 ):
    if len( poly ) < 3:
        return False
    e = np.roll( poly, -1, 0 ) - poly
    d = x[ None ] - poly
    return bool( ( e[ :, 0 ] * d[ :, 1 ] - e[ :, 1 ] * d[ :, 0 ] >= -eps ).all() )

def verifie( ags, cells, ref, P ):
    """Trois controles : le CERTIFICAT ( toute vraie cellule est dans la sur-cellule de son
    agregat ), les aires cellule par cellule, et leur somme."""
    manques, pire_v = 0, 0.0
    for A in ags:
        for i in A.idx:
            for x in ref[ i ]:
                if not any( dedans( m, x ) for m in A.morceaux ):
                    manques += 1
                    for m in A.morceaux:
                        if len( m ) < 3: continue
                        e = np.roll( m, -1, 0 ) - m; dd = x[ None ] - m
                        sg = float( np.min( e[ :, 0 ] * dd[ :, 1 ] - e[ :, 1 ] * dd[ :, 0 ] ) )
                        pire_v = max( pire_v, -sg )
    ec = np.array( [ abs( aire( cells[ i ] ) - aire( ref[ i ] ) ) for i in range( len( P ) ) ] )
    return manques, pire_v, ec.max(), sum( aire( c ) for c in cells )

# ------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "--n", type = int, default = 2000 )
    ap.add_argument( "--rho", type = int, nargs = "*", default = [ 8, 16 ] )
    ap.add_argument( "--kdop", type = int, default = 8 )
    ap.add_argument( "--graine", type = int, default = 0 )
    ap.add_argument( "--cas", nargs = "*", default = None )
    ap.add_argument( "--conv", action = "store_true",
                     help = "enveloppe convexe des morceaux : 1 test par paire au lieu de M x M" )
    ap.add_argument( "--conv-table", action = "store_true",
                     help = "convexifier pour la TABLE seulement, garder les morceaux pour le juge" )
    ap.add_argument( "--convexe", action = "store_true",
                     help = "un seul demi-plan par voisin ( le pari des fonds de U )" )
    a = ap.parse_args()

    ici = os.path.dirname( os.path.abspath( __file__ ) )
    Pu, Wu = uniforme( a.n, a.graine, 0.0 )
    Pw, Ww = uniforme( a.n, a.graine, 1.0 )
    Pv, Wv = charge( os.path.join( ici, "lines5_n2000_s0.005_voronoi.txt" ) )
    Pe, We = charge( os.path.join( ici, "lines5_n2000_s0.005_equal.txt" ) )
    cas = [ ( "uniforme, Voronoi", Pu, Wu ), ( "uniforme, poids aleatoires", Pw, Ww ),
            ( "5 lignes, Voronoi", Pv, Wv ), ( "5 lignes, aires egales", Pe, We ) ]
    if a.cas:
        cas = [ c for c in cas if any( s in c[ 0 ] for s in a.cas ) ]

    for nom, P, W in cas:
        t0 = time.time()
        ref = reference( P, W )
        tref = time.time() - t0
        sref = sum( aire( c ) for c in ref )
        print( f"\n=== {nom}   n = {len( P )}   reference : somme des aires {sref:.12f}"
               f"   ( {tref:.1f} s )" )
        print( f"    {'rho':>4s} {'|A|':>4s} | {'prep':>6s} {'SAT':>7s} {'table':>6s} {'kdop':>5s} |"
               f" {'amorce':>6s} {'juges':>6s} {'retenus':>7s} {'tentees':>8s} {'effect.':>7s} |"
               f" {'sans juge':>9s} | {'ecart':>9s} {'manq':>5s} {'cert':>5s}" )
        for rho in a.rho:
            ags, sites, lab, U = etape1( P, W, rho, a.kdop )
            npre = ( etape2_convexe if a.convexe else etape2 )( ags, sites, P, W, U )
            nenv = ( convexifie( ags, juge_aussi = not a.conv_table )
                     if ( a.conv or a.conv_table ) else None )
            cmq, cpire = certificat( ags, ref )
            TRAVAIL[ "sat" ] = TRAVAIL[ "clip" ] = 0
            table, nsat, nlarge = etape3( ags, U )
            tsat3 = TRAVAIL[ "sat" ]
            cells, st = etape4( ags, table, P, W, sites, juge = True )
            tsat4, tclip = TRAVAIL[ "sat" ] - tsat3, TRAVAIL[ "clip" ]
            _, st0 = etape4( ags, table, P, W, sites, juge = False )
            mq, pv, ec, som = verifie( ags, cells, ref, P )
            n = len( P )
            print( f"    {rho:4d} {n // len( ags ):4d} | {npre / n:6.2f} {nsat / n:7.1f}"
                   f" {sum( len( t ) for t in table ) / len( ags ):6.1f}"
                   f" {nlarge / len( ags ) - 1:5.1f} |"
                   f" {st[ 'amorce' ] / n:6.1f} {st[ 'juges' ] / n:6.1f} {st[ 'retenus' ] / n:7.1f}"
                   f" {( st[ 'tentees' ] + st[ 'amorce' ] ) / n:8.1f}"
                   f" {st[ 'effectives' ] / n:7.2f} |"
                   f" {( st0[ 'tentees' ] + st0[ 'amorce' ] ) / n:9.1f} | {ec:9.1e} {mq:5d}"
                   + ( f"   somme {som:.9f}" if abs( som - sref ) > 1e-9 else "" )
                   + ( f"   PARI PERDU, sortie max {cpire:.2e}" if cmq else "" )
                   + f"   travail/germe  table {tsat3 / n:6.0f}  juge {tsat4 / n:5.0f}"
                     f"  coupes {tclip / n:5.0f}"
                   + ( f"   env {nenv[ 0 ] / len( ags ):.0f}->{nenv[ 1 ] / len( ags ):.1f} sommets"
                       if nenv else "" ) )


# ============================================ LA SUR-CELLULE CONVEXE : UN SEUL DEMI-PLAN PAR VOISIN
#
# L'identite `{ max_v f_v >= max_j g_j } = U_v Lag( v )` fait apparaitre une REUNION de `M`
# cellules. La proposition testee ici est de n'en garder qu'UN demi-plan par voisin -- le « fond du
# U carre », le plus permissif le long de l'axe `c -> p_j` -- en pariant que les cotes du U sont de
# toute facon manges par les voisins.
#
# CE N'EST PAS SUR A PRIORI : aucun demi-plan n'est contenu dans un coin convexe, donc le cut est
# plus agressif que la reunion. C'est un PARI, et `--certificat` le juge : il verifie que tout
# sommet de toute vraie cellule reste dans la sur-cellule convexe de son agregat.
#
# Si le pari tient, la sur-cellule devient UNE cellule convexe par agregat au lieu de `M` : la
# preparation est divisee par `M`, le k-DOP est exact, et les tests de l'etape 3 comme du juge
# deviennent un seul SAT au lieu de `M x M'`.

def surcellule_convexe( A, P, W, sites ):
    """`M` demi-plans candidats par voisin, on n'en garde qu'UN : celui qui coupe le plus loin le
    long de l'axe `centre -> p_j`."""
    V = A.V if len( A.V ) >= 3 else A.P
    wv = V @ A.a + A.b + A.om                      # le pseudo-poids de chaque sommet
    Q = np.array( [ P[ j ] for j in sites if j not in set( A.idx.tolist() ) ] )
    WQ = np.array( [ W[ j ] for j in sites if j not in set( A.idx.tolist() ) ] )
    if len( Q ) == 0:
        return DOM, 0
    d = np.linalg.norm( Q - A.cen, axis = 1 )
    o = np.argsort( d ); Q, WQ, d = Q[ o ], WQ[ o ], d[ o ]
    wsuf = np.maximum.accumulate( WQ[ ::-1 ] )[ ::-1 ]
    wmin = float( wv.min() )

    cell, nb = DOM, 0
    for k in range( len( Q ) ):
        if len( cell ) < 3:
            break
        R = float( np.abs( cell - A.cen ).max() )
        if d[ k ] >= R and d[ k ] ** 2 - 2 * d[ k ] * R + wmin - wsuf[ k ] > 0:
            break
        u = Q[ k ] - A.cen
        aa = 2 * ( Q[ k ][ None, : ] - V )                     # ( M, 2 )
        cc = Q[ k ] @ Q[ k ] - ( V * V ).sum( 1 ) - WQ[ k ] + wv
        den = aa @ u
        t = np.where( den > 1e-300, ( cc - aa @ A.cen ) / np.where( den > 1e-300, den, 1 ), np.inf )
        m = int( np.argmax( t ) )
        if not np.isfinite( t[ m ] ):
            continue                                            # aucun sommet ne coupe cet axe
        cell = clip( cell, aa[ m ], float( cc[ m ] ) )
        nb += 1
    return cell, nb

def etape2_convexe( ags, sites, P, W, U ):
    nb = 0
    for A in ags:
        c, k = surcellule_convexe( A, P, W, sites )
        nb += k
        A.morceaux = [ c ] if len( c ) >= 3 else []
        A.kd = kdop( A.morceaux, U )
        if A.kd is None:
            A.morceaux = [ DOM ]; A.kd = kdop( A.morceaux, U )
    return nb

def certificat( ags, ref ):
    """Le pari tient-il ? Tout sommet de toute vraie cellule doit rester dans la sur-cellule de son
    agregat. Rend le nombre de sommets sortis et de combien ( en distance signee )."""
    manq, pire = 0, 0.0
    for A in ags:
        for i in A.idx:
            for x in ref[ i ]:
                if not any( dedans( m, x ) for m in A.morceaux ):
                    manq += 1
                    for m in A.morceaux:
                        if len( m ) < 3: continue
                        e = np.roll( m, -1, 0 ) - m
                        dd = x[ None ] - m
                        s = float( np.min( e[ :, 0 ] * dd[ :, 1 ] - e[ :, 1 ] * dd[ :, 0 ] ) )
                        L = float( np.linalg.norm( e, axis = 1 ).max() )
                        pire = max( pire, -s / max( L, 1e-30 ) )
    return manq, pire

if __name__ == "__main__":
    main()
