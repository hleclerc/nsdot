#!/usr/bin/env python
"""LA FORME DU GERME-POLYTOPE : ce que chaque enclos coute, et ce qu'il serre.

Suite de `surcell2d.py`, qui ne mesurait QUE l'enveloppe convexe. La question ici est le choix de
`K` : l'enveloppe est le plus serre des enclos, elle est aussi le plus cher a interroger. Entre le
point ( `pack` ) et l'enveloppe ( `hull` ) il y a une famille, et c'est elle qu'on mesure.

= LE CADRE

Un agregat `A`, un majorant AFFINE de ses poids `w_i <= a.p_i + b`, et n'importe quel ensemble
`K ⊇ { p_i }` donnent

    f_K( x ) = d( x + a/2, K )^2 - a.x - |a|^2/4 - b        <=  psi_A = min_{i in A} h_i

donc la SUR-CELLULE `{ f_K <= psi }` contient la reunion des vraies cellules de `A`. Verification :
`K = { p }`, `a = 0`, `b = w` redonne `f = |x-p|^2 - w`.

= POURQUOI LE CONTINU EST OBLIGATOIRE, ET C'EST TESTE ICI

Un minorant a support FINI n'existe pas. Avec `f = min_m ( |x-q_m|^2 - w_m )`, l'evaluer en
`x = p_i` impose `min_m ( |p_i - q_m|^2 - w_m ) <= -w_i`, et la limite `x -> inf` borne les `w_m` :
il faut `p_i ∈ { q_m }`. C'est exactement pourquoi `pack` a besoin d'un `delta` MESURE SUR UNE
REGION `E_m` -- une cellule de puissance par germe, 63 a 67 % de sa preparation.

La ligne `paraboloide, delta local` le montre : `delta` calcule sur les seules donnees de l'agregat
( la valeur juste au point `q` ) ne couvre PAS la reunion. La colonne `couv` tombe sous 1.00, donc
ce n'est pas un certificat. La ligne `paraboloide, delta ORACLE` prend le `delta` minimal qui
couvre -- il demande la reponse, il n'est pas calculable, et c'est le meilleur que `pack` puisse
esperer. Les six enclos continus, eux, couvrent par construction et ne demandent rien.

= CE QUI EST COMPTE

  couv  : part de la reunion des vraies cellules couverte par la sur-cellule.  DOIT valoir 1.00.
  aire  : aire de la sur-cellule / aire de la reunion.
  cand  : nombre de VRAIES cellules rencontrees par la sur-cellule / |A| -- la longueur de liste.
  aire* cand* : les MEMES, avec `b` ORACLE -- le plus petit `b` qui couvre encore la reunion, au
          lieu de celui que rend l'ajustement affine. Il n'est pas calculable ( il demande la
          reponse ) et il ne sert qu'a SEPARER les deux sources de mou : ce qui reste dans `aire*`
          est GEOMETRIQUE ( la forme de `K` ), l'ecart `aire - aire*` est le majorant de POIDS.
          Il est gratuit : `b` ne fait que translater `f`, donc un seul `max` sur la reunion.

`psi` est pris EXACT ( tous les germes ) : c'est la sur-cellule la plus serree possible, donc un
PLANCHER. Un pavage grossier ne peut que l'agrandir.
"""

import argparse, os, sys
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
from surcell2d import charge, uniforme, agregats, majorant_affine, dist2_polygone

# ------------------------------------------------------------------ l'enveloppe VRAIE

def psi_kd( P, W ):
    """`psi( x ) = min_j ( |x-p_j|^2 - w_j )` et son gagnant, par un KD-tree RELEVE : avec
    `s_j = sqrt( C - w_j )`, `|( x, 0 ) - ( p_j, s_j )|^2 = |x-p_j|^2 - w_j + C`, donc l'argmin est
    le meme. Exact, et en `O( log n )` au lieu du balayage."""
    C = float( W.max() ) + 1.0
    L = np.column_stack( [ P, np.sqrt( C - W ) ] )
    arbre = cKDTree( L )
    def ev( X ):
        d, j = arbre.query( np.column_stack( [ X, np.zeros( len( X ) ) ] ), workers = -1 )
        return d * d - C, j
    return ev

# ------------------------------------------------------------------------- les enclos
#
# Chacun rend `d2( Y )` -- la distance au carre a `K` -- et un trace. TOUS contiennent les germes
# de l'agregat, donc tous sont des certificats.

class Enclos:
    def __init__( self, Q ): self.Q = Q
    def d2( self, Y ): raise NotImplementedError
    def trace( self ): return None

class Germes( Enclos ):
    """`K` = les germes EUX-MEMES, qui est bien un ensemble contenant les germes. Mou geometrique
    NUL par definition : `d( y, K )^2 = min_i |y - p_i|^2` est exactement le terme que le majorant
    remplace. Ce n'est donc PAS un resume -- il coute `O( rho )` par requete, ce qu'on cherche
    justement a eviter -- mais c'est LE PLAFOND : ce qui reste au-dessus de 1.00 sur cette ligne
    n'est imputable qu'au majorant AFFINE des poids, et rien d'autre."""
    nom, coul, cout, style = "germes ( LE PLAFOND )", "#444444", "O( rho ) -- pas un resume", "-."
    def d2( self, Y ):
        b, r = 4096, np.empty( len( Y ) )
        for i in range( 0, len( Y ), b ):
            r[ i : i + b ] = ( ( Y[ i : i + b, None, : ] - self.Q[ None ] ) ** 2 ).sum( -1 ).min( 1 )
        return r
    def trace( self ): return None

class Enveloppe( Enclos ):
    """`K` = enveloppe convexe des germes. Le plus serre possible : mou geometrique NUL.
    Cout : le point le plus proche d'un polygone a `m` sommets, donc GJK ou `O( m )`."""
    nom, coul, cout = "enveloppe", "#0072B2", "O( m ) / GJK"
    def __init__( self, Q ):
        super().__init__( Q )
        self.V = Q[ ConvexHull( Q ).vertices ] if len( Q ) >= 3 else Q
    def d2( self, Y ):
        if len( self.V ) < 3:
            return ( ( Y[ :, None, : ] - self.V[ None ] ) ** 2 ).sum( -1 ).min( 1 )
        return dist2_polygone( Y, self.V )
    def trace( self ): return self.V

class KDop( Enclos ):
    """`K` = intersection de 8 demi-plans a directions FIXES ET PARTAGEES. Mou majore par
    `1 / cos( pi / 8 ) = 1.08` sur la fonction d'appui. Cout : `O( K )` pour les appuis, mais le
    point le plus proche reste celui d'un polygone -- l'appui est bon marche, la distance non."""
    nom, coul, cout = "k-DOP 8", "#009E73", "O( K )"
    NDIR = 8
    def __init__( self, Q ):
        super().__init__( Q )
        th = np.arange( self.NDIR ) * 2 * np.pi / self.NDIR
        D = np.column_stack( [ np.cos( th ), np.sin( th ) ] )
        h = ( Q @ D.T ).max( 0 )
        self.V = clip_demiplans( D, h, Q )
    def d2( self, Y ): return dist2_polygone( Y, self.V )
    def trace( self ): return self.V

class Boite( Enclos ):
    """`K` = la boite alignee des germes. C'est celle que le noeud du BSP porte DEJA, et
    `d( y, boite )^2` est SEPARABLE par axe -- exactement le `clamp` du test d'eviction. Mou nul
    sur les axes, `sqrt( 2 )` au pire."""
    nom, coul, cout = "boite", "#E69F00", "SEPARABLE, 2 clamp"
    def __init__( self, Q ):
        super().__init__( Q )
        self.lo, self.hi = Q.min( 0 ), Q.max( 0 )
    def d2( self, Y ):
        d = np.maximum( np.maximum( self.lo - Y, Y - self.hi ), 0.0 )
        return ( d * d ).sum( 1 )
    def trace( self ):
        lo, hi = self.lo, self.hi
        return np.array( [ lo, [ hi[ 0 ], lo[ 1 ] ], hi, [ lo[ 0 ], hi[ 1 ] ] ] )

class Capsule( Enclos ):
    """`K` = segment ( axe principal ) epaissi du rayon transverse. Anisotropie de RANG UN, et
    forme close : une projection sur le segment, un `clamp`, une norme. C'est l'enclos fait pour un
    paquet allonge -- le nuage de lignes."""
    nom, coul, cout = "capsule", "#D55E00", "proj + clamp + norme"
    def __init__( self, Q ):
        super().__init__( Q )
        c = Q.mean( 0 )
        D = Q - c
        if len( Q ) >= 2:
            _, _, Vt = np.linalg.svd( D, full_matrices = False )
            u = Vt[ 0 ]
        else:
            u = np.array( [ 1.0, 0.0 ] )
        t = D @ u
        n = np.array( [ -u[ 1 ], u[ 0 ] ] )
        self.A, self.B = c + t.min() * u, c + t.max() * u
        self.r = float( np.abs( D @ n ).max() ) if len( Q ) else 0.0
    def d2( self, Y ):
        e = self.B - self.A
        ee = float( e @ e )
        d = Y - self.A
        t = np.clip( ( d @ e ) / ee, 0, 1 ) if ee > 0 else np.zeros( len( Y ) )
        pr = d - t[ :, None ] * e
        dd = np.sqrt( np.einsum( "ij,ij->i", pr, pr ) )
        return np.maximum( dd - self.r, 0.0 ) ** 2
    def trace( self ):
        e = self.B - self.A
        L = np.hypot( *e )
        u = e / L if L > 0 else np.array( [ 1.0, 0.0 ] )
        n = np.array( [ -u[ 1 ], u[ 0 ] ] )
        th = np.linspace( -np.pi / 2, np.pi / 2, 24 )
        arc = lambda c, s: c + self.r * ( np.outer( np.cos( th ), s * u ) + np.outer( np.sin( th ), n ) )
        return np.vstack( [ arc( self.B, +1 ), arc( self.A, -1 ) ] )

class Ellipse( Enclos ):
    """`K` = l'ellipse d'inertie dilatee jusqu'a contenir les germes. Le mou est celui du k-DOP,
    mais `d( y, ellipse )` N'A PAS DE FORME CLOSE : il faut resoudre l'equation SECULAIRE
    `sum_k ( alpha_k v_k / ( alpha_k^2 + lam ) )^2 = 1`. On compte les iterations."""
    nom, coul, cout = "ellipse", "#CC79A7", "eq. SECULAIRE, iteratif"
    iters = 0
    def __init__( self, Q ):
        super().__init__( Q )
        self.c = Q.mean( 0 )
        D = Q - self.c
        S = ( D.T @ D ) / max( len( Q ), 1 ) + 1e-18 * np.eye( 2 )
        val, vec = np.linalg.eigh( S )
        # PLANCHER RELATIF : un paquet plat ( le nuage de lignes ) donne une valeur propre nulle,
        # l'ellipse degenere en segment et l'equation seculaire perd tous ses chiffres. On borne
        # l'aplatissement a 1e3, et on calcule le facteur d'echelle APRES -- l'enclos contient
        # alors les germes par construction, quel que soit le plancher.
        al0 = np.sqrt( np.maximum( val, val.max() * 1e-6 ) )
        V = D @ vec
        s = float( np.max( np.linalg.norm( V / al0, axis = 1 ) ) )
        self.al = al0 * max( s, 1e-300 )                      # demi-axes de l'enclos
        self.R = vec                                          # ses directions propres
    def d2( self, Y ):
        V = ( Y - self.c ) @ self.R                           # dans le repere propre
        al = self.al
        z = V / al
        dedans = ( z * z ).sum( 1 ) <= 1.0
        av = al * V
        lam = np.zeros( len( Y ) )
        # `g( lam ) = sum ( av_k / ( al_k^2 + lam ) )^2 - 1` decroit ; bisection sur un crochet sur.
        hi = np.linalg.norm( av, axis = 1 ) + al.max() ** 2      # crochet SUR : g( hi ) < 0
        lo = np.zeros( len( Y ) )
        g = lambda l: ( ( av / ( al * al + l[ :, None ] ) ) ** 2 ).sum( 1 ) - 1.0
        for _ in range( 50 ):                                 # 50 bisections = ~1e-15 relatif
            mid = 0.5 * ( lo + hi )
            m = g( mid ) > 0
            lo = np.where( m, mid, lo ); hi = np.where( m, hi, mid )
            Ellipse.iters += 1
        lam = 0.5 * ( lo + hi )
        U = av / ( al * al + lam[ :, None ] )
        pr = al * U - V
        return np.where( dedans, 0.0, np.einsum( "ij,ij->i", pr, pr ) )
    def trace( self ):
        th = np.linspace( 0, 2 * np.pi, 96 )
        return self.c + ( np.column_stack( [ self.al[ 0 ] * np.cos( th ),
                                             self.al[ 1 ] * np.sin( th ) ] ) @ self.R.T )

class Disque( Enclos ):
    """`K` = la boule centree au barycentre qui contient les germes. Isotrope : le mou est le
    rayon dans TOUTES les directions. Cout minimal apres la boite."""
    nom, coul, cout = "disque", "#56B4E9", "1 norme"
    def __init__( self, Q ):
        super().__init__( Q )
        self.c = Q.mean( 0 )
        self.R = float( np.linalg.norm( Q - self.c, axis = 1 ).max() )
    def d2( self, Y ):
        return np.maximum( np.linalg.norm( Y - self.c, axis = 1 ) - self.R, 0.0 ) ** 2
    def trace( self ):
        th = np.linspace( 0, 2 * np.pi, 96 )
        return self.c + self.R * np.column_stack( [ np.cos( th ), np.sin( th ) ] )

CONTINUS = [ Germes, Enveloppe, KDop, Capsule, Ellipse, Boite, Disque ]

def clip_demiplans( D, h, Q ):
    """Le polygone `{ x : D x <= h }`, obtenu en rognant un grand carre. Sutherland-Hodgman."""
    c = Q.mean( 0 ); R = 8.0 * ( np.linalg.norm( Q - c, axis = 1 ).max() + 1e-9 )
    V = [ c + np.array( v ) * R for v in ( ( -1, -1 ), ( 1, -1 ), ( 1, 1 ), ( -1, 1 ) ) ]
    for d, hh in zip( D, h ):
        out, m = [], len( V )
        for i in range( m ):
            A, B = V[ i ], V[ ( i + 1 ) % m ]
            sa, sb = d @ A - hh, d @ B - hh
            if sa <= 0: out.append( A )
            if ( sa > 0 ) != ( sb > 0 ):
                out.append( A + ( B - A ) * sa / ( sa - sb ) )
        V = out
        if len( V ) < 3: break
    return np.array( V )

# ---------------------------------------------------------- les deux paraboloides temoins

class ParabLocal:
    """`pack` avec le seul `delta` calculable sans region : celui qui rend l'inegalite juste AU
    POINT `q`, `delta = max_i ( w_i - |p_i - q|^2 )`. Il N'EST PAS un certificat, et c'est ce que
    la colonne `couv` montre."""
    nom, coul, style, cout = "parab., delta local", "#999999", ":", "1 norme + UNE CELLULE"
    oracle = False
    def __init__( self, Q, w, psi_U ):
        self.q = Q.mean( 0 )
        self.om = float( np.max( w - ( ( Q - self.q ) ** 2 ).sum( 1 ) ) )
    def f( self, X ): return ( ( X - self.q ) ** 2 ).sum( 1 ) - self.om
    def trace( self ): return None

class ParabOracle( ParabLocal ):
    """`pack` AU MIEUX : `delta` minimal qui couvre la reunion des vraies cellules. Il demande la
    reponse, donc il n'est pas calculable -- c'est la borne inferieure de la famille `pack`, celle
    que le journal appelle `delta` ORACLE."""
    nom, coul, style, cout = "parab., delta ORACLE", "#666666", "--", "1 norme + LA REPONSE"
    oracle = True
    def __init__( self, Q, w, psi_U ):
        self.q = Q.mean( 0 )
        XU, ps = psi_U
        self.om = float( np.max( ( ( XU - self.q ) ** 2 ).sum( 1 ) - ps ) ) if len( XU ) else 0.0

TEMOINS = [ ParabOracle, ParabLocal ]

# ------------------------------------------------------------------------- la campagne

def fenetre( xs, ys, masque, g, fac ):
    """Une sous-grille autour de la reunion, dilatee de `fac` fois sa diagonale."""
    idx = np.flatnonzero( masque )
    r, c = idx // g, idx % g
    di = max( int( fac * np.hypot( np.ptp( r ) + 1, np.ptp( c ) + 1 ) ), 3 )
    r0, r1 = max( r.min() - di, 0 ), min( r.max() + di + 1, g )
    c0, c1 = max( c.min() - di, 0 ), min( c.max() + di + 1, g )
    return r0, r1, c0, c1

def campagne( nom, P, W, S, g, nech, graine, verbose = True ):
    lo = P.min( 0 ) - 0.02; hi = P.max( 0 ) + 0.02
    xs = np.linspace( lo[ 0 ], hi[ 0 ], g ); ys = np.linspace( lo[ 1 ], hi[ 1 ], g )
    Xg, Yg = np.meshgrid( xs, ys )
    X = np.column_stack( [ Xg.ravel(), Yg.ravel() ] )
    ev = psi_kd( P, W )
    psi_g, gag = ev( X )

    lab, sites = agregats( P, S )
    na = len( sites )
    tailles = np.bincount( lab, minlength = na )
    ok = np.flatnonzero( tailles >= max( 2, S // 2 ) )
    rng = np.random.default_rng( graine )
    ech = rng.permutation( ok )[ : nech ]

    res = { }
    for cls in TEMOINS + CONTINUS:
        res[ cls.nom ] = { "couv" : [], "aire" : [], "cand" : [], "aire*" : [], "cand*" : [] }
    bord = 0
    for c in ech:
        m = np.flatnonzero( lab == c )
        U = np.isin( gag, m )
        if U.sum() < 4: continue
        Xw, psw, gaw, Uw = X, psi_g, gag, U
        aU = int( Uw.sum() )

        a, b = majorant_affine( P[ m ], W[ m ] )
        Y = Xw + a / 2
        dec = - Xw @ a - ( a @ a ) / 4 - b
        psi_U = ( Xw[ Uw ], psw[ Uw ] )

        for cls in TEMOINS:
            t = cls( P[ m ], W[ m ], psi_U )
            enregistre( res[ cls.nom ], t.f( Xw ) - psw, Uw, gaw, aU, len( m ) )
        for cls in CONTINUS:
            K = cls( P[ m ] )
            ech0 = K.d2( P[ m ] )
            if ech0.max() > 1e-18 + 1e-9 * float( np.ptp( P[ m ] ) ) ** 2:
                raise SystemExit( f"{cls.nom} NE CONTIENT PAS ses germes : d2 max {ech0.max():.3e}" )
            enregistre( res[ cls.nom ], K.d2( Y ) + dec - psw, Uw, gaw, aU, len( m ) )

    if verbose:
        print( f"\n--- {nom}   S = {S}   |A| median {int( np.median( tailles[ ok ] ) )}"
               f"   agregats {len( ech )}   grille {g}x{g}" )
        print( f"    {'enclos':26s} {'couv':>5s} {'aire':>6s} {'p90':>6s} {'aire*':>6s}"
               f" {'cand':>6s} {'p90':>6s} {'cand*':>6s}   {'cout de d2':s}" )
        couts = { c.nom : c.cout for c in TEMOINS + CONTINUS }
        for k, v in res.items():
            co, ar, ca, aro, cao = ( np.array( v[ x ] )
                                     for x in ( "couv", "aire", "cand", "aire*", "cand*" ) )
            drap = "" if co.min() > 0.9999 else "   <- PAS UN CERTIFICAT"
            print( f"    {k:26s} {co.min():5.2f} {np.median( ar ):6.2f}"
                   f" {np.percentile( ar, 90 ):6.2f} {np.median( aro ):6.2f}"
                   f" {np.median( ca ):6.2f} {np.percentile( ca, 90 ):6.2f}"
                   f" {np.median( cao ):6.2f}   {couts[ k ]:24s}{drap}" )
    return res

def enregistre( acc, ecart, Uw, gaw, aU, nm ):
    """`ecart = f - psi`. La sur-cellule est `{ ecart <= 0 }` ; avec `b` ORACLE elle devient
    `{ ecart <= max_reunion( ecart ) }`, puisque `b` ne fait que translater `f`."""
    seuils = ( 1e-12, float( ecart[ Uw ].max() ) + 1e-12 )
    for cle, s in zip( ( "", "*" ), seuils ):
        d = ecart <= s
        acc[ "aire" + cle ].append( float( d.sum() ) / aU )
        acc[ "cand" + cle ].append( len( np.unique( gaw[ d ] ) ) / nm )
    acc[ "couv" ].append( float( ( ( ecart <= seuils[ 0 ] ) & Uw ).sum() ) / aU )

def bord_touche( dedans, nr, nc ):
    D = dedans.reshape( nr, nc )
    return D[ 0 ].any() or D[ -1 ].any() or D[ :, 0 ].any() or D[ :, -1 ].any()

# --------------------------------------------------------------------------- le dessin

import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt

ENCRE, ENCRE2, FOND = "#1a1a1a", "#5c5c5c", "#fcfcfb"

def choisit( P, lab, na, S ):
    """L'agregat le plus ALLONGE parmi ceux qui sont bien garnis : c'est celui ou les enclos se
    separent. Sur un nuage isotrope il n'y a rien a voir, et c'est aussi un resultat."""
    best, bc = -1.0, None
    for c in range( na ):
        m = np.flatnonzero( lab == c )
        if len( m ) < max( 3, S // 2 ): continue
        s = np.linalg.svd( P[ m ] - P[ m ].mean( 0 ), compute_uv = False )
        r = s[ 0 ] / max( s[ 1 ], 1e-12 )
        if r > best: best, bc = r, m
    return bc, best

def prepare( nom, P, W, S, g, graine ):
    lab, sites = agregats( P, S )
    m, aniso = choisit( P, lab, len( sites ), S )
    a, b = majorant_affine( P[ m ], W[ m ] )
    ev = psi_kd( P, W )

    dom_lo, dom_hi = P.min( 0 ) - 0.02, P.max( 0 ) + 0.02
    xg = np.linspace( dom_lo[ 0 ], dom_hi[ 0 ], 200 ); yg = np.linspace( dom_lo[ 1 ], dom_hi[ 1 ], 200 )
    Ag, Bg = np.meshgrid( xg, yg )
    _, gg = ev( np.column_stack( [ Ag.ravel(), Bg.ravel() ] ) )
    pts = np.column_stack( [ Ag.ravel(), Bg.ravel() ] )[ np.isin( gg, m ) ]
    pts = np.vstack( [ pts, P[ m ] ] ) if len( pts ) else P[ m ]
    lo, hi = pts.min( 0 ), pts.max( 0 )
    c = 0.5 * ( lo + hi ); demi = 0.62 * np.maximum( hi - lo, 1e-6 )
    lo = np.maximum( c - demi, dom_lo ); hi = np.minimum( c + demi, dom_hi )
    xs = np.linspace( lo[ 0 ], hi[ 0 ], g )
    ys = np.linspace( lo[ 1 ], hi[ 1 ], g )
    Xg, Yg = np.meshgrid( xs, ys )
    X = np.column_stack( [ Xg.ravel(), Yg.ravel() ] )
    psw, gaw = ev( X )
    U = np.isin( gaw, m ).reshape( g, g )
    aU = max( int( U.sum() ), 1 )
    dec = - X @ a - ( a @ a ) / 4 - b
    psi_U = ( X[ U.ravel() ], psw[ U.ravel() ] )

    objets = [ ( cls.nom, cls.coul, getattr( cls, "style", "-" ),
                 cls( P[ m ], W[ m ], psi_U ).f( X ), None ) for cls in TEMOINS ]
    for cls in CONTINUS:
        K = cls( P[ m ] )
        objets.append( ( cls.nom, cls.coul, getattr( cls, "style", "-" ),
                         K.d2( X + a / 2 ) + dec, K.trace() ) )

    return dict( nom = nom, P = P, m = m, aniso = aniso, xs = xs, ys = ys, g = g,
                 U = U, aU = aU, psw = psw, objets = objets )

def dessine( fig, gs, ligne, d ):
    nom, P, m, aniso = d[ "nom" ], d[ "P" ], d[ "m" ], d[ "aniso" ]
    xs, ys, g, U, aU, psw = d[ "xs" ], d[ "ys" ], d[ "g" ], d[ "U" ], d[ "aU" ], d[ "psw" ]
    for j, ( nomK, coul, style, f, tr ) in enumerate( d[ "objets" ] ):
        ax = fig.add_subplot( gs[ ligne, j ] )
        dedans = ( f <= psw + 1e-12 ).reshape( g, g )
        ax.set_facecolor( FOND )
        ax.contourf( xs, ys, U.astype( float ), levels = [ 0.5, 1.5 ], colors = [ "#d9d9d9" ] )
        ax.contourf( xs, ys, dedans.astype( float ), levels = [ 0.5, 1.5 ],
                     colors = [ coul ], alpha = 0.30 )
        ax.contour( xs, ys, dedans.astype( float ), levels = [ 0.5 ], colors = [ coul ],
                    linewidths = 2.0, linestyles = style )
        if tr is not None:
            ax.fill( *np.vstack( [ tr, tr[ :1 ] ] ).T, facecolor = "none", edgecolor = coul,
                     lw = 1.0, ls = "--" )
        ax.scatter( P[ :, 0 ], P[ :, 1 ], s = 1.2, c = "#bdbdbd", linewidths = 0, zorder = 3 )
        ax.scatter( P[ m, 0 ], P[ m, 1 ], s = 5.0, c = ENCRE, linewidths = 0, zorder = 4 )
        manque = int( ( U & ~dedans ).sum() )
        eti = f"{nomK}\naire x{dedans.sum() / aU:.2f}"
        if manque: eti += f"   MANQUE {100 * manque / aU:.0f} %"
        ax.set_title( eti, fontsize = 8.5, color = ENCRE if not manque else "#b03030" )
        ax.set_xticks( [] ); ax.set_yticks( [] )
        ax.set_xlim( xs[ 0 ], xs[ -1 ] ); ax.set_ylim( ys[ 0 ], ys[ -1 ] )
        ax.set_aspect( "equal" )
        for s in ax.spines.values(): s.set_color( "#dddddd" )
        if j == 0:
            court = nom.replace( "uniforme", "unif." ).replace( "poids aleatoires", "poids alea." )
            court = court.replace( "5 lignes", "lignes" ).replace( ", ", "\n" )
            ax.set_ylabel( court, fontsize = 7.5, color = ENCRE )
            eti = f"|A| = {len( m )}, allongement {aniso:.1f}\n" + eti
            ax.set_title( eti, fontsize = 8.5, color = ENCRE if not manque else "#b03030" )

def figure_formes( cas, S, g, graine, out ):
    n = len( TEMOINS ) + len( CONTINUS )
    ds = [ prepare( nom, P, W, S, g, graine ) for nom, P, W in cas ]
    # a aspect egal, la hauteur d'une ligne est celle de sa fenetre. Bornee : une bande tres plate
    # deviendrait illisible.
    hr = [ min( max( ( d[ "ys" ][ -1 ] - d[ "ys" ][ 0 ] ) / ( d[ "xs" ][ -1 ] - d[ "xs" ][ 0 ] ),
                     0.34 ), 1.15 ) for d in ds ]
    larg = 1.95
    fig = plt.figure( figsize = ( larg * n, larg * sum( hr ) + 0.55 * len( cas ) + 0.9 ),
                      facecolor = FOND )
    gs = fig.add_gridspec( len( cas ), n, hspace = 0.42, wspace = 0.06, height_ratios = hr )
    for i, d in enumerate( ds ):
        dessine( fig, gs, i, d )
    fig.suptitle( f"LA FORME DE L'ENCLOS  |  S = {S}  |  gris : la reunion des VRAIES cellules de "
                  f"l'agregat   -   couleur : la sur-cellule   -   tirets : l'enclos K",
                  fontsize = 11, color = ENCRE )
    fig.savefig( out, dpi = 135, facecolor = FOND, bbox_inches = "tight" )
    print( "->", out )

def figure_bilan( bilan, out ):
    """Deux mesures d'echelles differentes -> deux facettes, jamais deux axes."""
    cles = [ c.nom for c in TEMOINS ] + [ c.nom for c in CONTINUS ]
    coul = { c.nom : c.coul for c in TEMOINS + CONTINUS }
    fig, axs = plt.subplots( len( bilan ), 2, figsize = ( 12.0, 2.7 * len( bilan ) + 0.9 ),
                             facecolor = FOND, sharey = True )
    axs = np.atleast_2d( axs )
    for i, ( nom, res ) in enumerate( bilan ):
        for j, ( mes, titre ) in enumerate( ( ( "aire", "aire de la sur-cellule / aire de la reunion" ),
                                              ( "cand", "cellules rencontrees / membre" ) ) ):
            ax = axs[ i ][ j ]; ax.set_facecolor( FOND )
            v = [ np.median( res[ k ][ mes ] ) for k in cles ]
            y = np.arange( len( cles ) )[ ::-1 ]
            faux = [ np.min( res[ k ][ "couv" ] ) < 0.9999 for k in cles ]
            ax.barh( y, v, height = 0.62, color = [ coul[ k ] for k in cles ], linewidth = 0,
                     hatch = None )
            for yy, vv, f in zip( y, v, faux ):
                if f:
                    ax.barh( [ yy ], [ vv ], height = 0.62, color = "none", edgecolor = "#b03030",
                             hatch = "///", linewidth = 1.0 )
            for yy, vv, f in zip( y, v, faux ):
                ax.text( vv + max( v ) * 0.015, yy,
                         f"{vv:.2f}" + ( "  PAS UN CERTIFICAT" if f else "" ), va = "center",
                         fontsize = 8, color = "#b03030" if f else ENCRE )
            vo = [ np.median( res[ k ][ mes + "*" ] ) for k in cles ]
            ax.scatter( vo, y, marker = "|", s = 260, linewidths = 1.6, color = ENCRE, zorder = 4 )
            ax.axvline( 1.0, color = ENCRE2, lw = 1.0, ls = ":" )
            ax.set_yticks( y ); ax.set_yticklabels( cles, fontsize = 8.5, color = ENCRE )
            ax.set_xlim( 0, max( v ) * 1.34 )
            ax.tick_params( axis = "x", labelsize = 8, colors = ENCRE2 )
            for s in ( "top", "right", "left" ): ax.spines[ s ].set_visible( False )
            ax.spines[ "bottom" ].set_color( "#dddddd" )
            ax.grid( axis = "x", color = "#eeeeee", lw = 0.8 )
            ax.set_axisbelow( True )
            if i == 0: ax.set_title( titre, fontsize = 9.5, color = ENCRE )
            if j == 0: ax.text( -0.42, 1.06, nom, transform = ax.transAxes, fontsize = 9.5,
                                color = ENCRE, fontweight = "bold" )
    fig.suptitle( "MEDIANE PAR AGREGAT, psi EXACT\npointille : le plancher ( la reunion "
                  "elle-meme )   -   trait noir sur la barre : la meme mesure avec b ORACLE, "
                  "donc le mou purement GEOMETRIQUE", fontsize = 9.5, color = ENCRE )
    fig.tight_layout( rect = [ 0.02, 0, 1, 0.93 ] )
    fig.savefig( out, dpi = 140, facecolor = FOND )
    print( "->", out )

# ------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "--n", type = int, default = 2000 )
    ap.add_argument( "--g", type = int, default = 700 )
    ap.add_argument( "--gd", type = int, default = 340, help = "grille des dessins" )
    ap.add_argument( "--nech", type = int, default = 60 )
    ap.add_argument( "--esses", type = int, nargs = "*", default = [ 8, 16, 64 ] )
    ap.add_argument( "--graine", type = int, default = 0 )
    ap.add_argument( "--sfig", type = int, default = 16 )
    ap.add_argument( "--pas-de-figure", action = "store_true" )
    ap.add_argument( "--sans-mesure", action = "store_true", help = "les figures seulement" )
    a = ap.parse_args()

    ici = os.path.dirname( os.path.abspath( __file__ ) )
    Pu, Wu = uniforme( a.n, a.graine, 0.0 )
    Pw, Ww = uniforme( a.n, a.graine, 1.0 )
    Pv, Wv = charge( os.path.join( ici, "lines5_n2000_s0.005_voronoi.txt" ) )
    Pe, We = charge( os.path.join( ici, "lines5_n2000_s0.005_equal.txt" ) )
    cas = [ ( "uniforme, Voronoi", Pu, Wu ),
            ( "uniforme, poids aleatoires", Pw, Ww ),
            ( "5 lignes, Voronoi", Pv, Wv ),
            ( "5 lignes, aires egales", Pe, We ) ]

    bilan = [ ]
    for nom, P, W in ( [ ] if a.sans_mesure else cas ):
        for S in a.esses:
            r = campagne( nom, P, W, S, a.g, a.nech, a.graine )
            if S == a.sfig: bilan.append( ( f"{nom}  ( S = {S} )", r ) )
    if Ellipse.iters:
        print( f"\n[ ellipse : {Ellipse.iters} passes de bisection sur l'equation seculaire, "
               f"50 par appel -- les cinq autres enclos n'en font aucune ]" )

    if not a.pas_de_figure:
        figure_formes( cas, a.sfig, a.gd, a.graine,
                       os.path.join( ici, "surcell2d_formes.png" ) )
        if bilan:
            figure_bilan( bilan, os.path.join( ici, "surcell2d_formes_bilan.png" ) )

if __name__ == "__main__":
    main()
