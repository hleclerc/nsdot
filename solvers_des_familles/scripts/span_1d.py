#!/usr/bin/env python
"""LE SPAN { PROLONGATION , CORRECTION } EN 1D : la boite a outils, et les figures.

POURQUOI LA 1D. Le schema a valider est celui-ci : on part d'un `w` SAIN ( Voronoi au debut ), on
ajoute `alpha_0 ( w_prol - w_sain )` ou `w_prol` est la solution grossiere PROJETEE PAR COPIE ( le
meme poids a tous les germes fins d'un agregat ), et `log2 = sum ( log a_i/nu_i )^2` descend puis
REMONTE -- parce que la copie fait un saut de `w` aux interfaces d'agregats, et que les cellules qui
s'y trouvent se pincent. Au meilleur `alpha_0` on resout le residu log pour obtenir `w_1`, qui dit
comment relever ces cellules-la, puis on minimise `log2` sur le plan
    w_sain + alpha_0 ( w_prol - w_sain ) + alpha_1 ( w_1 - w_sain )
a `L` et polynomes GELES. La question est de savoir si `alpha_0` va alors beaucoup plus loin.

En 1D tout est EXACT et se dessine : les cellules sont des intervalles, les masses des differences
d'`erf`, `L` est tridiagonale, et la limite d'admissibilite est la premiere longueur d'intervalle qui
s'annule -- donc une fonction AFFINE de `alpha`, dont la racine se lit.

LA 1D MONTRE AUSSI LES ITERATIONS, contrairement a ce que le § 8.3 laisse croire -- et c'est une
nuance mesuree ici. Le § 8.3 dit qu'en 1D Newton converge en UN pas depuis tout depart admissible :
c'est vrai pour LEBESGUE, ou `a( w )` est lineaire tant qu'aucune cellule ne se vide. Avec une
DENSITE, `a( w )` ne l'est plus, et la reference depuis Voronoi demande des dizaines d'iterations.
Le compte d'iterations est donc un chiffre honnete ici.

LE CHOIX DES GERMES N'EST PAS NEUTRE, et c'est le piege de ce banc. La limite d'admissibilite d'une
direction est fixee par la PAIRE DE GERMES LA PLUS SERREE : une cellule se vide quand la difference
seconde de `w` depasse `h_l h_r`, et sur `n` germes tires au hasard le plus petit ecart vaut `O( 1/n^2 )`
au lieu de `O( 1/n )`. Le plus petit `h_l h_r` est alors `n` fois plus petit que le typique, et
`alpha*` s'effondre d'autant -- pour une raison qui n'a rien a voir avec la prolongation ( c'est la
meme pathologie que les 56 paires de germes confondus du § 23.7, qu'il a fallu dedupliquer ).
`--germes regulier` ( le defaut ) met les germes a `( i + 1/2 ) / n` et laisse voir le phenomene
qu'on veut voir ; `--germes aleatoire` montre l'autre.

LE SPAN EST LE MEME QU'AVEC `d`. `w_1 = w( alpha_0* ) + d` avec `d` la direction de Newton en ce
point, donc `w_1 - w_sain = alpha_0* ( w_prol - w_sain ) + d` : le sous-espace engendre est
`span{ w_prol - w_sain , d }`, identique. Orthogonaliser ne change que le conditionnement vu par
l'optimiseur, pas ce qui est atteignable.

LA BOITE A OUTILS, en appels simples :

    from span_1d import Cas
    cas = Cas( n = 200, R = 8 )         # germes uniformes, densite = plancher + gaussiennes
    a, b = cas.masses( w )              # les masses, et les bords de cellules
    cas.log2( w )                       # sum ( log a/nu )^2, +inf si une cellule est vide
    cas.residu( w )                     # log( a_i / nu_i ), par cellule
    cas.newton( w )                     # la direction du residu log ( jauge : d[0] = 0 )
    cas.alpha_max( w, d )               # le premier alpha ou une cellule se vide ( exact )
    cas.resout( w0 )                    # Newton amorti -> w*, nb d'iterations
    w_prol = cas.grossier()             # le solve a n/R, projete PAR COPIE sur les germes fins
    cas.span_min( w0, [ d0, d1 ] )      # minimise log2 sur le span -> alphas, w, log2

    python scripts/span_1d.py                       # ecrit figures/span_1d_*.png
    python scripts/span_1d.py -n 400 -R 16 --sigma 0.02
"""
import argparse, os
import numpy as np
from scipy.special import erf


# ---------------------------------------------------------------- la densite
class Densite:
    """`rho( x ) = plancher + sum_k m_k N( c_k, s_k )` sur [0,1], avec sa primitive EXACTE.

    Le plancher n'est pas un ornement : `log2` est infini des qu'une cellule a une masse nulle, donc
    une densite a zeros rend l'objectif inutilisable ( c'est ce qui se passe en 2D avec
    `--plancher 0` : 90 % des cellules ont une masse nulle a Voronoi ). Il faut `rho > 0` partout.
    """

    def __init__( self, sigma = 0.03, plancher = 0.05, jeu = None ):
        # des centres pas trop symetriques, des masses et des largeurs inegales -- l'esprit du
        # `densite_jeu` de `Densite.h`, en une dimension
        self.g = jeu if jeu is not None else [ ( 0.26, 1.0 * sigma, 0.45 ),
                                              ( 0.58, 0.7 * sigma, 0.30 ),
                                              ( 0.82, 1.3 * sigma, 0.25 ) ]
        tot = sum( m for _, _, m in self.g )
        self.g = [ ( c, s, m * ( 1 - plancher ) / tot ) for c, s, m in self.g ]
        self.plancher = plancher

    def rho( self, x ):
        x = np.asarray( x, dtype = float )
        r = np.full( x.shape, self.plancher )
        for c, s, m in self.g:
            r = r + m * np.exp( -0.5 * ( ( x - c ) / s ) ** 2 ) / ( s * np.sqrt( 2 * np.pi ) )
        return r

    def primitive( self, x ):
        """`F( x ) = int_0^x rho` -- exacte, par `erf`."""
        x = np.asarray( x, dtype = float )
        f = self.plancher * x
        for c, s, m in self.g:
            f = f + m * 0.5 * ( erf( ( x - c ) / ( s * np.sqrt( 2 ) ) ) + erf( c / ( s * np.sqrt( 2 ) ) ) )
        return f

    def masse( self ):
        return float( self.primitive( 1.0 ) - self.primitive( 0.0 ) )


# ---------------------------------------------------------------- le cas
class Cas:
    """`n` germes uniformes sur [0,1], la densite, et la cible `nu_i = M / n`.

    LES CELLULES. En 1D le bord entre `i` et `i+1` est
        b_i = ( p_i + p_{i+1} ) / 2 + ( w_i - w_{i+1} ) / ( 2 ( p_{i+1} - p_i ) )
    et la cellule de `i` est `[ b_{i-1}, b_i ]` ( avec `b_{-1} = 0`, `b_{n-1} = 1` ). La formule
    suppose que les voisins de Laguerre sont les voisins CONSECUTIFS, ce qui est exact partout ou
    aucune cellule n'est vide -- c'est-a-dire exactement le domaine ou `log2` est fini, donc le seul
    ou le schema a le droit d'aller. Hors de la, `log2` rend `+inf` et on n'y va pas.
    """

    def __init__( self, n = 200, R = 8, sigma = 0.03, plancher = 0.05, graine = 0, germes = "regulier" ):
        self.n, self.R = n, R
        self.rho = Densite( sigma, plancher )
        if germes == "regulier":
            p = ( np.arange( n ) + 0.5 ) / n
        else:
            rng = np.random.default_rng( graine )
            p = np.sort( rng.uniform( 0.5 / n, 1 - 0.5 / n, n ) )
        self.germes = germes
        self.p = p
        self.dp = np.diff( p )                       # `p_{i+1} - p_i`, taille n-1
        self.nu = np.full( n, self.rho.masse() / n )
        # les agregats : `R` germes consecutifs ( en 1D, le BSP a feuilles de `R` EST le decoupage
        # consecutif -- il n'y a pas de forme de paquet a discuter )
        self.paquet = np.minimum( np.arange( n ) // R, ( n - 1 ) // R )
        self.nb_paquets = int( self.paquet.max() ) + 1
        # les interfaces : les germes dont le voisin de droite est dans un autre paquet
        self.interfaces = np.where( np.diff( self.paquet ) != 0 )[ 0 ]

    # ---- les mesures
    def bords( self, w ):
        b = 0.5 * ( self.p[ :-1 ] + self.p[ 1: ] ) + ( w[ :-1 ] - w[ 1: ] ) / ( 2 * self.dp )
        return np.concatenate( ( [ 0.0 ], b, [ 1.0 ] ) )

    def masses( self, w ):
        b = self.bords( w )
        return np.diff( self.rho.primitive( b ) ), b

    def longueurs( self, w ):
        return np.diff( self.bords( w ) )

    def admissible( self, w, eps = 0.0 ):
        return bool( np.all( self.longueurs( w ) > eps ) )

    # ---- l'objectif
    def residu( self, w ):
        """`log( a_i / nu_i )`, par cellule. `None` si une cellule est vide."""
        a, _ = self.masses( w )
        if np.any( a <= 0 ):
            return None
        return np.log( a / self.nu )

    def log2( self, w ):
        """`sum_i ( log a_i / nu_i )^2`, `+inf` si une cellule est vide.

        LA BARRIERE EST DANS L'OBJECTIF : `a_i -> 0` donne `-inf` au logarithme, donc `+inf` ici. Il
        n'y a aucune contrainte a imposer de l'exterieur -- le minimum ne peut pas vider une cellule.
        """
        g = self.residu( w )
        return np.inf if g is None else float( np.sum( g * g ) )

    # ---- la derivee et la direction
    def laplacien( self, w ):
        """`L = da / dw`, tridiagonale : `c_i = rho( b_i ) / ( 2 ( p_{i+1} - p_i ) )`, et
        `L = diag( c_{i-1} + c_i ) - offdiag( c )`. C'est `c_ij = |facette| rho / 2 |p_i - p_j|`
        avec `|facette| = 1` en une dimension."""
        b = self.bords( w )
        c = self.rho.rho( b[ 1:-1 ] ) / ( 2 * self.dp )
        L = np.zeros( ( self.n, self.n ) )
        idx = np.arange( self.n - 1 )
        L[ idx, idx + 1 ] = -c
        L[ idx + 1, idx ] = -c
        d = np.zeros( self.n )
        d[ :-1 ] += c
        d[ 1: ] += c
        L[ np.arange( self.n ), np.arange( self.n ) ] = d
        return L

    def newton( self, w ):
        """LA DIRECTION DU RESIDU LOG, comme `Newton.h` la construit : le systeme est
        `diag( g' / nu ) L d = c - g` avec `g( x ) = log x`, `x = a / nu`, donc `g' = 1/x` et
        `nu / g' = a` :
            `L d = a ( c - log( a / nu ) )`,   `c = sum a log( a/nu ) / sum a`
        La constante `c` n'est pas un choix esthetique : `L` a le vecteur constant dans son noyau,
        donc son image est orthogonale a 1 et le second membre DOIT etre de somme nulle. Jauge :
        `d[ 0 ] = 0`, puisque `w` et `w + cste` donnent le meme diagramme.
        """
        a, _ = self.masses( w )
        g = np.log( a / self.nu )
        c = float( np.sum( a * g ) / np.sum( a ) )
        rhs = a * ( c - g )
        L = self.laplacien( w )
        d = np.zeros( self.n )
        d[ 1: ] = np.linalg.solve( L[ 1:, 1: ], rhs[ 1: ] )   # la jauge raye la premiere ligne
        return d

    def newton_gele( self, w, L0 ):
        """LA DIRECTION DU RESIDU LOG SUR UN `L` GELE : le second membre est celui du point courant,
        la matrice celle du point de base. C'est `lin.resout_encore` du banc 2D -- une re-resolution
        sur la factorisation deja payee."""
        a, _ = self.masses( w )
        g = np.log( a / self.nu )
        c = float( np.sum( a * g ) / np.sum( a ) )
        rhs = a * ( c - g )
        d = np.zeros( self.n )
        d[ 1: ] = np.linalg.solve( L0[ 1:, 1: ], rhs[ 1: ] )
        return d

    def alpha_max( self, w, d ):
        """LE PREMIER `alpha` OU UNE CELLULE SE VIDE, exact. Les bords sont affines en `alpha`, donc
        chaque longueur de cellule l'est : `len_i( alpha ) = len_i( 0 ) + alpha * pente_i`, et on
        prend la plus petite racine positive. `inf` si aucune longueur ne decroit."""
        l0 = self.longueurs( w )
        pente = self.longueurs( w + d ) - l0
        m = np.inf
        neg = pente < 0
        if np.any( neg ):
            m = float( np.min( -l0[ neg ] / pente[ neg ] ) )
        return m

    # ---- les solveurs
    def resout( self, w0, tol = 1e-10, maxit = 60, facteur = 0.9 ):
        """Newton amorti sur le residu log : le pas plein s'il passe, sinon `facteur * alpha*`, puis
        des moities tant que `log2` ne descend pas. Rend `( w, nb_iterations )`.

        En 1D c'est presque toujours UN pas ( § 8.3 ) : `a` est lineaire et le segment vers la
        solution reste admissible. Les iterations qui restent sont celles que la densite impose,
        `rho` rendant `a( w )` non lineaire.
        """
        w = np.array( w0, dtype = float )
        for it in range( 1, maxit + 1 ):
            g = self.residu( w )
            if g is None:
                raise ValueError( "depart non admissible" )
            if np.max( np.abs( g ) ) < tol:
                return w, it - 1
            d = self.newton( w )
            t = min( 1.0, facteur * self.alpha_max( w, d ) )
            base = self.log2( w )
            for _ in range( 40 ):
                if self.log2( w + t * d ) < base:
                    break
                t /= 2
            else:
                return w, it
            w = w + t * d
        return w, maxit

    def grossier( self ):
        """LE NIVEAU GROSSIER : un germe par agregat, au barycentre, de cible la somme des cibles,
        resolu avec le meme code. Rend `( w_grossier, positions_grossieres, iterations )` -- la
        PROJECTION est un choix a part ( `prolonge` )."""
        pg = np.array( [ self.p[ self.paquet == k ].mean() for k in range( self.nb_paquets ) ] )
        cg = Cas.__new__( Cas )
        cg.n, cg.R, cg.rho, cg.germes = len( pg ), 1, self.rho, "grossier"
        cg.p, cg.dp = pg, np.diff( pg )
        cg.nu = np.array( [ self.nu[ self.paquet == k ].sum() for k in range( self.nb_paquets ) ] )
        cg.paquet = np.arange( cg.n )
        cg.nb_paquets, cg.interfaces = cg.n, np.arange( cg.n - 1 )
        wg, itg = cg.resout( np.zeros( cg.n ) )
        return wg, pg, itg

    def lisse( self, w, m = 1 ):
        """`m` passes de moyenne `( w_{i-1} + 2 w_i + w_{i+1} ) / 4` ( bords reflechis ). La longueur
        de diffusion est `sqrt( m )` germes : a `m = R^2` le lissage couvre exactement un agregat, et
        c'est la que `alpha*` passe au-dessus de 1."""
        w = np.array( w, dtype = float )
        for _ in range( m ):
            g = np.concatenate( ( [ w[ 0 ] ], w[ :-1 ] ) )
            d = np.concatenate( ( w[ 1: ], [ w[ -1 ] ] ) )
            w = 0.25 * ( g + 2 * w + d )
        return w

    def prolonge( self, wg, pg, mode = "copie", lissages = 0 ):
        """LA PROJECTION du grossier sur les germes fins. Ce qui distingue les modes n'est pas leur
        amplitude mais la DIFFERENCE SECONDE qu'ils portent a l'echelle de la cellule, puisque
        `alpha* = 2 h^2 / ( difference seconde )` :

          `copie`   un poids constant par agregat. La difference seconde est le SAUT entier aux
                    interfaces -- inadmissible au premier ordre, `alpha* = O( h / R )`.
          `affine`  l'interpolation lineaire entre les germes grossiers. En 1D c'est EXACTEMENT la
                    prolongation harmonique du § 8.2, `C^0` avec un PLI sur chaque representant : la
                    difference seconde y vaut `h H w''` au lieu de `h^2 w''`, donc `alpha* = O( 1/R )`.
                    C'est la forme qui se generalise le plus simplement en nD ( barycentrique sur une
                    triangulation des germes grossiers ).
          `lissages` passes de `( 1, 2, 1 )/4` par-dessus : a `m = R^2` la longueur de diffusion
                    couvre un agregat et le resultat est `C^1`, donc `alpha* = O( 1 )`.
        """
        if mode == "copie":
            w = wg[ self.paquet ].astype( float ).copy()
        elif mode == "affine":
            w = np.interp( self.p, pg, wg )               # extrapolation constante aux bords
        else:
            raise ValueError( f"mode de prolongation inconnu : {mode}" )
        return self.lisse( w, lissages ) if lissages else w

    # ---- le gradient, et Gauss-Newton dans l'espace des alpha
    def jacobien_span( self, w, dirs, L_gel = None ):
        """`( J, g )` avec `g_i = log( a_i / nu_i )` et `J_ik = d g_i / d alpha_k = ( L d_k )_i / a_i`.

        EXACT A CONNECTIVITE FIXE : les bords sont affines en `alpha`, donc `a_i( alpha )` est
        analytique, et sa derivee est `da = L dw` -- la tridiagonale de `laplacien`. Rien n'est
        approche, et c'est ce qui permet de se passer de `alpha*` : la barriere du logarithme EST
        dans l'objectif, donc une recherche de pas ne peut pas franchir le bord.
        """
        a, _ = self.masses( w )
        # `L_gel` : la variante du schema propose, ou `L` n'est PAS refait. `J` devient approche ( `L`
        # depend de `alpha` par les longueurs de facettes ), mais reste une direction de descente -- et
        # en 2D ca fait toute la difference, un `L` refait etant un diagramme de plus.
        L = self.laplacien( w ) if L_gel is None else L_gel
        J = np.empty( ( self.n, len( dirs ) ) )
        for k, d in enumerate( dirs ):
            J[ :, k ] = ( L @ np.asarray( d, dtype = float ) ) / a
        return J, np.log( a / self.nu )

    def gradient_span( self, w, dirs, L_gel = None ):
        """`d log2 / d alpha_k = 2 ( J^T g )_k`."""
        J, g = self.jacobien_span( w, dirs, L_gel )
        return 2.0 * ( J.T @ g )

    def verifie_gradient( self, w0, dirs, h = 1e-7 ):
        """LE GRADIENT CONTRE DES DIFFERENCES FINIES CENTREES ( l'usage du banc, cf. `--check` de
        `densite` ). Rend `( ecart relatif max, analytique, numerique )`."""
        al = np.zeros( len( dirs ) )
        W = np.array( dirs )
        ana = self.gradient_span( w0, dirs )
        num = np.empty_like( ana )
        for k in range( len( al ) ):
            e = np.zeros( len( al ) )
            e[ k ] = h
            num[ k ] = ( self.log2( w0 + ( al + e ) @ W ) - self.log2( w0 + ( al - e ) @ W ) ) / ( 2 * h )
        return float( np.max( np.abs( ana - num ) / np.maximum( np.abs( num ), 1e-300 ) ) ), ana, num

    def span_min( self, w0, dirs, itmax = 80, tol = 1e-13, trace = False, gel = False, al0 = None ):
        """MINIMISE `log2` SUR `w0 + sum_k alpha_k dirs[ k ]` PAR GAUSS-NEWTON dans l'espace des
        `alpha` : `log2` etant une somme de carres, le bon pas resout `min | g + J da |^2`, et c'est
        un solve `k x k` puisque `k` est petit.

        PAS DE `alpha*`, ET C'EST LE POINT : la recherche de pas recule tant que `log2` n'est pas fini
        ou ne descend pas, et `log2` vaut `+inf` des qu'une cellule se vide. Le bord n'est jamais
        franchi, et l'optimum est atteint a la precision de la machine au lieu du pas d'une grille.
        Rend `( alphas, w, log2, iterations, |grad| )`.
        """
        al = np.zeros( len( dirs ) ) if al0 is None else np.array( al0, dtype = float )
        W = np.array( dirs )

        def en( a ):
            return w0 + a @ W

        f = self.log2( en( al ) )
        if not np.isfinite( f ):
            return al, en( al ), np.inf, 0, np.inf    # depart inadmissible : l'appelant decide
        L0 = self.laplacien( w0 ) if gel else None       # `L` GELE au point de base, une fois pour tout
        it = 0
        for it in range( 1, itmax + 1 ):
            J, g = self.jacobien_span( en( al ), dirs, L0 )
            da, *_ = np.linalg.lstsq( J, -g, rcond = None )
            t, f2, pris = 1.0, np.inf, False
            for _ in range( 80 ):
                f2 = self.log2( en( al + t * da ) )
                if np.isfinite( f2 ) and f2 < f:
                    pris = True
                    break
                t /= 2
            if not pris:
                break                                     # plus de descente : on est a l'optimum
            al = al + t * da
            fini = abs( f - f2 ) <= tol * abs( f )
            f = f2
            if trace:
                print( f"      gn {it:3d}  t {t:.3e}  log2 {f:.12e}  |grad| "
                       f"{np.linalg.norm( self.gradient_span( en( al ), dirs, L0 ) ):.3e}" )
            if fini:
                break
        return al, en( al ), f, it, float( np.linalg.norm( self.gradient_span( en( al ), dirs ) ) )

    def span_min_grille( self, w0, dirs, grille = 61, balayages = 6, facteur = 0.999 ):
        """MINIMISE `log2` SUR `w0 + sum_k alpha_k dirs[ k ]`, coordonnee par coordonnee, la recherche
        1-D etant faite EN UNITES DE `alpha*` : a chaque passe on recalcule l'intervalle admissible de
        la coordonnee AU POINT COURANT et on y balaye une grille mixte -- lineaire sur tout
        l'intervalle, plus logarithmique de chaque cote du point courant, parce que l'optimum peut
        etre a `1e-5 alpha*` comme au bord. Rend `( alphas, w, log2 )`."""
        al = np.zeros( len( dirs ) )
        W = np.array( dirs )

        def en( a ):
            return w0 + a @ W

        best = self.log2( en( al ) )
        for _ in range( balayages ):
            for k in range( len( al ) ):
                w_cur, c = en( al ), al[ k ]
                lo, hi = self.bornes( w_cur, W[ k ] )
                lo = facteur * lo if np.isfinite( lo ) else -max( 1.0, 4 * abs( c ) )
                hi = facteur * hi if np.isfinite( hi ) else max( 1.0, 4 * abs( c ) )
                pas = [ np.linspace( lo, hi, grille ) ]
                if hi > 0:
                    pas.append( np.geomspace( 1e-6 * hi, hi, grille // 2 ) )
                if lo < 0:
                    pas.append( -np.geomspace( -1e-6 * lo, -lo, grille // 2 ) )
                meilleur = c
                for v in np.unique( np.concatenate( pas ) ):
                    al[ k ] = c + v
                    m = self.log2( en( al ) )
                    if m < best:
                        best, meilleur = m, al[ k ]
                al[ k ] = meilleur
        return al, en( al ), best


# ---------------------------------------------------------------- les figures
def figures( cas, sortie, lissages = 0, mode = "copie" ):
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt

    n = cas.n
    w_sain = np.zeros( n )                          # le depart SAIN : Voronoi
    wg, pg, it_g = cas.grossier()
    w_prol = cas.prolonge( wg, pg, mode, lissages )
    w_prol = w_prol - w_prol[ 0 ]                   # la jauge, pour que les figures se comparent
    d_prol = w_prol - w_sain
    w_ref, it_ref = cas.resout( w_sain )

    print( f"  n = {n}, R = {cas.R}, germes {cas.germes}, {cas.nb_paquets} agregats,"
           f" {len( cas.interfaces )} interfaces  ;  ecart min {cas.dp.min():.3e} contre 1/n = {1.0/n:.3e}" )
    print( f"  grossier : {it_g} iterations  ;  reference fine depuis Voronoi : {it_ref} iterations" )
    print( f"  log2 : Voronoi {cas.log2( w_sain ):.4e}, solution {cas.log2( w_ref ):.4e}" )

    # ---- 1. log2 LE LONG DE alpha_0, et la limite d'admissibilite
    a_lim = cas.alpha_max( w_sain, d_prol )
    al = np.unique( np.concatenate( ( np.linspace( 0, min( 1.5, 1.05 * a_lim ), 300 ),
                                     np.geomspace( max( 1e-6, 1e-4 * a_lim ), 1.02 * a_lim, 300 ) ) ) )
    vals = np.array( [ cas.log2( w_sain + t * d_prol ) for t in al ] )
    fini = np.isfinite( vals )
    i_best = int( np.argmin( np.where( fini, vals, np.inf ) ) )
    a0 = float( al[ i_best ] )
    print( f"  |w_prol|inf = {np.max( np.abs( w_prol ) ):.3e}, |w*|inf = {np.max( np.abs( w_ref ) ):.3e}"
           f"  ;  saut max de la copie aux interfaces = "
           f"{np.max( np.abs( np.diff( w_prol )[ cas.interfaces ] ) ):.3e}" )
    print( f"  alpha* ( premiere cellule vide ) = {a_lim:.3e}  ;  R^-2 = {1.0 / cas.R ** 2:.3e}"
           f"  ;  alpha* R^2 = {a_lim * cas.R ** 2:.4f}" )
    print( f"  meilleur alpha_0 = {a0:.4e} = {a0 / a_lim:.4f} alpha*  ;  log2 {vals[ i_best ]:.10e}"
           f" contre {vals[ 0 ]:.10e} a Voronoi  ( gain relatif {1 - vals[ i_best ] / vals[ 0 ]:.3e} )" )

    fig, ax = plt.subplots( figsize = ( 7, 4.2 ) )
    ax.semilogy( al[ fini ], vals[ fini ], lw = 1.8, label = r"$\log_2$ le long de $\alpha_0$" )
    ax.axvline( a_lim, color = "crimson", ls = "--", lw = 1,
                label = rf"$\alpha^*$ = {a_lim:.3f} ( 1re cellule vide )" )
    ax.axvline( a0, color = "seagreen", ls = ":", lw = 1.4, label = rf"meilleur $\alpha_0$ = {a0:.3f}" )
    ax.axhline( cas.log2( w_ref ), color = "gray", ls = "-.", lw = 1, label = "la solution" )
    ax.set_xlabel( r"$\alpha_0$  dans  $w_{sain} + \alpha_0 ( w_{prol} - w_{sain} )$" )
    ax.set_ylabel( r"$\sum_i ( \log a_i / \nu_i )^2$" )
    ax.set_title( rf"La prolongation suivie comme direction ( lissee {lissages}x )" )
    ax.legend( fontsize = 8 )
    fig.tight_layout()
    fig.savefig( os.path.join( sortie, "span_1d_alpha0.png" ), dpi = 130 )
    plt.close( fig )

    # ---- 2. OU ca pince : le residu par cellule, et les interfaces d'agregats
    w0 = w_sain + a0 * d_prol
    g0, gv = cas.residu( w0 ), cas.residu( w_sain )
    fig, ax = plt.subplots( 2, 1, figsize = ( 8, 5.6 ), sharex = True )
    ax[ 0 ].plot( cas.p, w_prol, lw = 1.2, drawstyle = "steps-mid", label = r"$w_{prol}$ ( copie )" )
    ax[ 0 ].plot( cas.p, w_ref, lw = 1.2, label = r"$w^*$ ( la solution )" )
    ax[ 0 ].plot( cas.p, w0, lw = 1.0, ls = ":", label = rf"$\alpha_0 = {a0:.3f}$" )
    ax[ 0 ].set_ylabel( "poids" )
    ax[ 0 ].legend( fontsize = 8 )
    ax[ 0 ].set_title( rf"Le saut aux interfaces d'agregats ( lissage {lissages}x )" )
    ax[ 1 ].plot( cas.p, gv, lw = 0.9, color = "gray", label = "Voronoi" )
    ax[ 1 ].plot( cas.p, g0, lw = 1.2, color = "crimson", label = rf"$\alpha_0 = {a0:.3f}$" )
    for j, i in enumerate( cas.interfaces ):
        ax[ 1 ].axvline( 0.5 * ( cas.p[ i ] + cas.p[ i + 1 ] ), color = "steelblue", lw = 0.5,
                         alpha = 0.55, label = "interface" if j == 0 else None )
    ax[ 1 ].axhline( 0, color = "black", lw = 0.6 )
    ax[ 1 ].set_xlabel( "x" )
    ax[ 1 ].set_ylabel( r"$\log a_i / \nu_i$" )
    ax[ 1 ].legend( fontsize = 8 )
    fig.tight_layout()
    fig.savefig( os.path.join( sortie, "span_1d_pincement.png" ), dpi = 130 )
    plt.close( fig )

    # ---- 3. LE PLAN ( alpha_0, alpha_1 ) : `w_1` debloque-t-il `alpha_0` ?
    d_new = cas.newton( w0 )                        # `w_1 = w0 + d_new`, donc `w_1 - w_sain`
    d_1 = ( w0 + d_new ) - w_sain
    al1, w_span, l_span, gn1, ng1 = cas.span_min( w_sain, [ d_prol, d_1 ] )
    lim1 = cas.alpha_max( w_sain, d_1 )
    print( f"  span : alpha_0 = {al1[ 0 ]:.4e} ( {al1[ 0 ] / a_lim:+.4f} alpha*_0 ),"
           f" alpha_1 = {al1[ 1 ]:.4e} ( {al1[ 1 ] / lim1:+.4f} alpha*_1 )" )
    print( f"  log2 du span = {l_span:.10e}  ( gain relatif {1 - l_span / vals[ 0 ]:.3e} )"
           f"  ;  Gauss-Newton {gn1} iterations, |grad| = {ng1:.2e}" )
    _, it_span = cas.resout( w_span )
    print( f"  iterations restantes : {it_span} depuis le span, {it_ref} depuis Voronoi" )

    # LES AXES EN UNITES DE `alpha*` : sur [ 0, 1.5 ] la carte serait grise partout
    A0 = np.linspace( 0, 1.4 * a_lim, 90 )
    A1 = np.linspace( 0, 1.4 * lim1, 90 )
    Z = np.empty( ( len( A1 ), len( A0 ) ) )
    for i, y in enumerate( A1 ):
        for j, x in enumerate( A0 ):
            Z[ i, j ] = cas.log2( w_sain + x * d_prol + y * d_1 )
    fig, ax = plt.subplots( figsize = ( 6.6, 5.4 ) )
    fini = np.isfinite( Z )
    lz = np.where( fini, np.log10( np.where( fini, Z, 1 ) ), np.nan )
    im = ax.pcolormesh( A0, A1, lz, shading = "auto", cmap = "viridis" )
    ax.contour( A0, A1, lz, levels = 14, colors = "white", linewidths = 0.4, alpha = 0.6 )
    ax.contourf( A0, A1, ( ~fini ).astype( float ), levels = [ 0.5, 1.5 ],
                 colors = [ "0.25" ], alpha = 0.95 )
    ax.plot( [ a0 ], [ 0 ], "o", color = "crimson", ms = 6, label = rf"meilleur $\alpha_0$ seul = {a0:.3f}" )
    ax.plot( [ al1[ 0 ] ], [ al1[ 1 ] ], "*", color = "gold", ms = 15,
             label = rf"optimum du span ( {al1[ 0 ]:.3f}, {al1[ 1 ]:.3f} )" )
    fig.colorbar( im, ax = ax, label = r"$\log_{10} \sum ( \log a_i/\nu_i )^2$" )
    ax.set_xlabel( rf"$\alpha_0$ ( prolongation ), $\alpha^*_0$ = {a_lim:.2e}" )
    ax.set_ylabel( rf"$\alpha_1$ ( correction ), $\alpha^*_1$ = {lim1:.2e}" )
    ax.set_title( "Le gris est INADMISSIBLE ( une cellule vide )" )
    ax.legend( fontsize = 8, loc = "upper right" )
    fig.tight_layout()
    fig.savefig( os.path.join( sortie, "span_1d_carte.png" ), dpi = 130 )
    plt.close( fig )
    return a0, al1, l_span, it_span, it_ref


def monte_alpha0( cas, d0, kmax, amax = 1.0, trace = True, compagnes = "lissage" ):
    """CONTINUATION EN `alpha_0`, LES COMPAGNES LIBRES -- la mesure qui repond a la question.

    Minimiser `log2` sur le span n'est qu'un PROXY : son minimum peut etre a `alpha_0` petit alors
    qu'un grand `alpha_0` est atteignable avec les bonnes compagnes. Ce qu'on veut est
        max alpha_0  sous contrainte que le point reste ADMISSIBLE,
    les autres coefficients libres. C'est le role des compagnes : ACCOMPAGNER `w_prol` en le lissant
    la ou il pince, au lieu de le pre-lisser a la main avec un `m` devine.

    LES COMPAGNES SONT DES INCREMENTS DE LISSAGE, et c'est le point. Prendre « ce qui manque » comme
    direction de Newton au point bloque ne marche pas, et la raison est structurelle : la ou une
    cellule est a `1e-8 nu`, les lignes du jacobien `J_ik = ( L d_k )_i / a_i` valent `1e8`, le moindre
    carre est entierement domine par elles et la tangente sort a `1e+07` ( mesure ). La barriere qui
    protege detruit le conditionnement de toute algebre lineaire a son voisinage.
    On prend donc les compagnes LITTERALEMENT comme des directions qui LISSENT :
        d_j = lisse( w_prol, 4^j ) - w_prol,     j = 0, 1, 2, ...
    une echelle dyadique d'increments de lissage. Le span est alors
        w = alpha_0 w_prol + sum_j alpha_j ( lisse_j( w_prol ) - w_prol )
    et l'optimiseur choisit LUI-MEME le profil de lissage, au lieu qu'on devine un `m`. C'est bien
    borne, bien conditionne, et ca generalise en nD ( `lisse_jacobi` sur le graphe de Voronoi ).

    PREDICTEUR-CORRECTEUR, et il le faut : `log2` valant `+inf` hors de l'admissible, aucune methode
    de descente ne peut y ENTRER -- une continuation naive reste bloquee au premier pas qui sort ( les
    coefficients compagnons restent a zero, mesure ). On suit donc la VARIETE DES MINIMISEURS. A
    l'optimum sur les compagnes, `d log2 / d alpha_j = 0` pour `j >= 1` ; en derivant par rapport a
    `alpha_0`, la tangente est
        d alpha_compagnes / d alpha_0 = - ( Jc^T Jc )^-1 Jc^T j_0
    avec `Jc` les colonnes des compagnes et `j_0` celle de `w_prol` ( Gauss-Newton ). On predit par
    cette tangente, on corrige par `span_min`, et on halve le pas quand ca ne passe pas.

    Quand le pas ne peut plus croitre, on AJOUTE une direction -- « ce qui manque » au point courant,
    c'est-a-dire la direction du residu log, qui releve exactement les cellules qui bloquent.
    `amax = 1` ET NON PLUS : maximiser `alpha_0` est le MAUVAIS objectif, et c'est mesure -- pousse a
    1.21 ( le bord de l'admissible ) le point a un `log2` de 2634, PIRE que Voronoi, et il reste 41
    iterations ; plafonne a 1 avec `log2` minimise sur les compagnes, c'est le bon point. `alpha_0 = 1`
    est une CIBLE, pas un maximand.
    Rend la liste `( k, alpha_0 atteint, log2, iterations restantes )`.
    """
    w_sain = np.zeros( cas.n )
    dirs, al = [], []
    a0 = 0.0
    f = cas.log2( w_sain )
    hist = []

    def point( a, coefs ):
        w = w_sain + a * d0
        for c, d in zip( coefs, dirs ):
            w = w + c * d
        return w

    for k in range( 1, kmax + 1 ):
        pas = max( 1e-4, 0.05 * amax )
        while a0 < amax and pas > 1e-9:
            # LA TANGENTE au point courant ( rien a predire s'il n'y a pas de compagne )
            tang = np.zeros( len( dirs ) )
            if dirs:
                J, _ = cas.jacobien_span( point( a0, al ), [ d0 ] + dirs )
                tang, *_ = np.linalg.lstsq( J[ :, 1: ], -J[ :, 0 ], rcond = None )
            essai = min( a0 + pas, amax )
            pred = list( np.array( al ) + ( essai - a0 ) * tang ) if dirs else []
            base = w_sain + essai * d0
            if dirs:
                al2, w2, f2, _, _ = cas.span_min( base, dirs, al0 = pred )
            else:
                al2, f2 = [], cas.log2( base )
            if np.isfinite( f2 ):
                a0, al, f = essai, list( al2 ), f2
                pas *= 1.5
            else:
                if trace and dirs and pas > 1e-6:
                    print( f"       refus a alpha_0 {essai:.5f} ( pas {pas:.2e} ) :"
                           f" tangente {np.array2string( tang, precision = 3 )},"
                           f" predit {np.array2string( np.array( pred ), precision = 3 )}" )
                pas /= 2
        # LA METRIQUE : ce que Newton coute depuis le point atteint
        try:
            _, it_k = cas.resout( point( a0, al ) )
        except ValueError:
            it_k = -1
        hist.append( ( k, a0, f, it_k ) )
        if trace:
            print( f"  {k:2d} | {a0:10.4f} | {f:12.5e} | {it_k:4d} | "
                   + " ".join( f"{v:+.3f}" for v in al ) )
        if a0 >= amax - 1e-9 or k == kmax:
            break
        w_bloc = point( a0, al )
        a_bloc, _ = cas.masses( w_bloc )
        if compagnes == "lissage":
            d_new = cas.lisse( d0, 4 ** k ) - d0         # l'echelle dyadique : 4, 16, 64, ...
        else:
            d_new = cas.newton( w_bloc )
        if trace:
            print( f"       au point bloque : min a/nu {np.min( a_bloc / cas.nu ):.3e},"
                   f" cellules < 0.01 nu : {int( np.sum( a_bloc < 0.01 * cas.nu ) )}"
                   f"  ;  direction ajoutee |d|inf {np.max( np.abs( d_new ) ):.3e}"
                   f" ( |w_prol|inf {np.max( np.abs( d0 ) ):.3e} ), finie : {np.all( np.isfinite( d_new ) )}" )
        dirs.append( d_new )
        al.append( 0.0 )
    return hist


def balaye_k( cas, kmax, lissages = 0, gel = False, mode = "copie" ):
    """LA QUESTION : quand on ajoute des directions, jusqu'ou `alpha_0` va-t-il ?

    Le schema exactement tel qu'il est propose : au meilleur point du span courant, on resout le
    residu log ( `L` recalcule LA -- en 1D c'est trois lignes, en 2D ce serait `L` gele ), on en fait
    une direction de plus mesuree depuis `w_sain`, et on re-minimise `log2` sur le span elargi. On
    rapporte `alpha_0` EN UNITES DE SON PROPRE `alpha*`, parce que c'est la seule echelle qui ait un
    sens, et le nombre d'iterations qui restent -- la metrique.
    """
    w_sain = np.zeros( cas.n )
    wg, pg, it_g = cas.grossier()
    w_prol = cas.prolonge( wg, pg, mode, lissages )
    w_prol = w_prol - w_prol[ 0 ]
    d_prol = w_prol - w_sain
    a_lim = cas.alpha_max( w_sain, d_prol )
    saut = float( np.max( np.abs( np.diff( w_prol )[ cas.interfaces ] ) ) )
    l0 = cas.log2( w_sain )
    _, it_ref = cas.resout( w_sain )
    print( f"  n = {cas.n}, R = {cas.R}, germes {cas.germes}  ;  grossier {it_g} it  ;"
           f"  reference depuis Voronoi {it_ref} it  ;  log2 Voronoi {l0:.6e}" )
    print( f"  prolongation {mode}, lissages {lissages}  ;  saut aux interfaces {saut:.4e}  ;"
           f"  alpha*_0 = {a_lim:.4e}  ;  alpha* x saut = {a_lim * saut:.4e}  ( 2h^2 = {2.0 / cas.n ** 2:.4e} )" )
    ec, _, _ = cas.verifie_gradient( w_sain, [ d_prol ] )
    print( f"  gradient analytique contre differences finies centrees : ecart relatif {ec:.2e}" )
    print( "   k |        log2 |   gain rel |    alpha_0 | alpha_0/alpha*_0 | gn it |    |grad| | it restantes" )
    L0_base = cas.laplacien( w_sain ) if gel else None
    dirs, al = [ d_prol ], None
    for k in range( 1, kmax + 1 ):
        al, w_opt, lk, gn, ng = cas.span_min( w_sain, dirs, gel = gel )
        _, it_k = cas.resout( w_opt )
        print( f"  {k:2d} | {lk:11.5e} | {1 - lk / l0:10.3e} | {al[ 0 ]:10.4e} |"
               f" {al[ 0 ] / a_lim:16.4f} | {gn:5d} | {ng:9.2e} | {it_k:12d}" )
        if k == kmax:
            break
        # « ce qui manque » : la direction du residu log au meilleur point, mesuree depuis `w_sain`
        # « ce qui manque » : sur le `L` GELE aussi, si on gele -- c'est `resout_encore` en 2D
        d = cas.newton_gele( w_opt, L0_base ) if gel else cas.newton( w_opt )
        dirs.append( ( w_opt + d ) - w_sain )
    return al


def main():
    ap = argparse.ArgumentParser( description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter )
    ap.add_argument( "-n", type = int, default = 200, help = "germes fins" )
    ap.add_argument( "-R", type = int, default = 8, help = "germes par agregat" )
    ap.add_argument( "--sigma", type = float, default = 0.03, help = "largeur des gaussiennes" )
    ap.add_argument( "--plancher", type = float, default = 0.05,
                     help = "fraction de la masse dans le plancher uniforme ( > 0 : log2 doit etre fini )" )
    ap.add_argument( "--graine", type = int, default = 0 )
    ap.add_argument( "--germes", default = "regulier", choices = [ "regulier", "aleatoire" ],
                     help = "regulier : ( i + 1/2 ) / n, pour que la paire la plus serree ne decide pas tout" )
    ap.add_argument( "-k", type = int, default = 0,
                     help = "> 0 : balayer le nombre de directions du span, et ne pas faire les figures" )
    ap.add_argument( "--prol", default = "copie", choices = [ "copie", "affine" ],
                     help = "copie ( constante par agregat ) | affine ( interpolation lineaire )" )
    ap.add_argument( "--compagnes", default = "lissage", choices = [ "lissage", "newton" ],
                     help = "lissage : d_j = lisse( w_prol, 4^j ) - w_prol | newton : le residu log" )
    ap.add_argument( "--alpha0", type = int, default = 0,
                     help = "> 0 : CONTINUATION EN alpha_0, les compagnes libres, jusqu'a K directions" )
    ap.add_argument( "--gel", action = "store_true",
                     help = "geler `L` au point de base ( le schema propose ) au lieu de le refaire" )
    ap.add_argument( "--lisse", type = int, default = 0,
                     help = "passes de moyenne sur la prolongation par copie ( attaque alpha*, pas k )" )
    ap.add_argument( "--sortie", default = None, help = "dossier des figures ( defaut : ../figures )" )
    o = ap.parse_args()
    sortie = o.sortie or os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), "..", "figures" )
    os.makedirs( sortie, exist_ok = True )
    cas = Cas( o.n, o.R, o.sigma, o.plancher, o.graine, o.germes )
    if o.alpha0 > 0:
        wg, pg, itg = cas.grossier()
        w_prol = cas.prolonge( wg, pg, o.prol, o.lisse )
        w_prol = w_prol - w_prol[ 0 ]
        _, it_ref = cas.resout( np.zeros( cas.n ) )
        print( f"  n = {cas.n}, R = {cas.R}, germes {cas.germes}, prolongation {o.prol}"
               f" + {o.lisse} lissages  ;  grossier {itg} it  ;  reference {it_ref} it" )
        print( f"  alpha* seul ( k = 1 ) = {cas.alpha_max( np.zeros( cas.n ), w_prol ):.4e}" )
        print( "   k |   alpha_0  |        log2  |  it | coefficients des compagnes" )
        monte_alpha0( cas, w_prol, o.alpha0, compagnes = o.compagnes )
        return
    if o.k > 0:
        balaye_k( cas, o.k, o.lisse, o.gel, o.prol )
        return
    figures( cas, sortie, o.lisse, o.prol )
    print( f"  figures dans {os.path.normpath( sortie )}/span_1d_*.png" )


if __name__ == "__main__":
    main()
