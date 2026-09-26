#pragma once

// =====================================================================================
// LA DENSITE IMAGE, SUR LE CPU : integrer `rho` SUR LE BORD, jamais sur l'interieur.
//
// Une image est une densite CONSTANTE PAR PIXEL sur une grille reguliere de `W x H` cases
// couvrant `[ 0, 1 ]^2`. Elle differe d'une densite reguliere ( `Densite.h` ) par ce qui compte :
// elle est DISCONTINUE, et elle peut valoir ZERO sur des regions entieres.
//
// La facon naturelle de mesurer une cellule sous une telle densite est de la DECOUPER par les
// bords de pixels et de sommer `rho x aire` -- c'est ce que fait le moteur de production
// ( `sdot/include/sdot/Image.cxx` ), et ca coute `2 d` coupes et une copie de la cellule PAR
// PIXEL de la boite englobante. Ici on ne coupe rien. Green :
//
//      masse( P ) = int_P rho dA = int_(bord P) G( x, y ) dy,      G( x, y ) = int_0^x rho( t, y ) dt
//
// et pour une image `G` est CONNUE EN FERME : sur la ligne `j`, `rho` ne depend plus de `y`, donc
//
//      G_j( x ) = S[ j ][ i ] + rho[ j ][ i ] ( x - i hx ),   S[ j ][ i ] = hx somme_(k<i) rho[j][k]
//
// -- la somme prefixe de la ligne, calculee UNE FOIS. Sur un morceau d'arete qui reste dans le
// pixel `( i, j )`, de `( xa, ya )` a `( xb, yb )`, `G` est AFFINE EN `x` : son integrale en `y`
// est donc exacte au point milieu,
//
//      ( yb - ya ) ( S[ j ][ i ] + rho[ j ][ i ] ( ( xa + xb ) / 2 - i hx ) )
//
// Le cout passe de `pixels DANS la cellule` a `pixels SOUS LE BORD`, sans une coupe ni une copie,
// et le parcours est un Amanatides-Woo de quatre lignes. C'est EXACT, pas approche : aucune
// quadrature, contrairement aux gaussiennes de `Densite.h`.
//
// DEUX SORTIES POUR UNE SEULE MARCHE. Sous une densite, le coefficient de la hessienne n'est plus
// la longueur de la facette mais `int_facette rho ds / ( 2 |p_i - p_j| )`. C'est la MEME marche,
// le meme pixel, la meme sous-arete. C'est la raison de fond pour laquelle tout travaille ARETE
// PAR ARETE plutot que cellule par cellule.
//
// LA REFERENCE `sref`. La somme des `dy` sur un polygone ferme vaut zero : RETRANCHER UNE
// CONSTANTE A `G` NE CHANGE RIEN. Sans elle les termes sont d'ordre `taille de cellule x S ~ 1e-3`
// pour une masse qui vaut `1e-6` -- trois chiffres perdus par annulation. On retranche `S` au
// premier sommet, et les termes tombent a l'ordre de la masse.
//
// = LE CONTRASTE, ET SA DERIVEE GRATUITE
//
// On ne stocke que l'image NUE ( normalisee : `int rho = 1` ). Le melange avec Lebesgue
//
//      rho_t = ( 1 - t ) + t rho
//
// est applique a la sortie : `masse = ( 1 - t ) aire + t masse_rho`, et
// `int_facette rho_t ds = ( 1 - t ) L + t int_facette rho ds`. Le parcours rend les deux morceaux,
// donc `d masse / d t = masse_rho - aire` NE COUTE RIEN -- c'est ce qui permet l'extrapolation
// tangente d'une etape de continuation a la suivante ( comme le chemin MELANGE de `Densite.h` ).
// `t = 0` rend Lebesgue, `t = 1` l'image nue, et la masse totale vaut 1 pour tout `t`.
//
// C'est UN des deux chemins de continuation, et MESURE COMME LE MOINS BON : `convolue()` plus bas
// fait l'autre -- l'image floutee a une largeur decroissante -- et il va quatre a cinq fois plus
// vite ( README § 12.6 ), parce qu'il ne deplace la densite que la ou elle est rugueuse au lieu de
// la changer partout a chaque etape.
//
// = `TA`, LE FLOTTANT DE LA MESURE
//
// Tout le calcul de la marche se fait en `TA` ( les sommes prefixes comprises ). Le defaut est
// `double` ; `ImageT<float>` existe pour pouvoir MESURER ce que la simple precision coute --
// c'est la question ouverte du banc, et c'est sur le CPU qu'elle s'instrumente.
//
// 2D seulement.
// =====================================================================================

#include "util/common.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <limits>
#include <vector>

namespace sf {

template<class TA = double>
struct ImageT {
    int W = 0, H = 0;
    std::vector<TA> v;            ///< la densite du pixel, `v[ j * W + i ]`, `j = 0` en bas
    std::vector<TA> s;            ///< la somme prefixe de la ligne, `hx` compris ( `prepare()` )
    TA  t = 1;                    ///< le contraste : `rho_t = ( 1 - t ) + t rho`
    TA  hx = 0, hy = 0, ihx = 0, ihy = 0;

    static constexpr int borne( int i, int n ) { return i < 0 ? 0 : ( i >= n ? n - 1 : i ); }
    int col( TA x ) const { return borne( int( x * ihx ), W ); }
    int lig( TA y ) const { return borne( int( y * ihy ), H ); }
    TA  pixel( int i, int j ) const { return v[ size_t( j ) * W + i ]; }

    /// les pas, les sommes prefixes. LES SOMMES SONT FAITES EN `double` puis arrondies a `TA` :
    /// c'est une table, pas un calcul de cellule -- la question de la precision porte sur la marche.
    void prepare() {
        hx = TA( 1 ) / W; hy = TA( 1 ) / H;
        ihx = TA( W ); ihy = TA( H );
        s.assign( v.size(), TA( 0 ) );
        for ( int j = 0; j < H; ++j ) {
            double a = 0;
            for ( int i = 0; i < W; ++i ) { s[ size_t( j ) * W + i ] = TA( a ); a += double( v[ size_t( j ) * W + i ] ) / W; }
        }
    }

    /// la masse totale ramenee a un ( le Newton vise `1 / n` par cellule ). Rend le facteur applique.
    double normalise() {
        double tot = 0;
        for ( TA u : v ) tot += double( u );
        tot /= double( W ) * H;
        if ( ! ( tot > 0 ) ) return 0;
        for ( TA &u : v ) u = TA( double( u ) / tot );
        prepare();
        return tot;
    }

    // ------------------------------------------------------------------ ce que le solveur demande
    /// la densite en un point ( le contraste compris )
    TF rho( TF x, TF y ) const { return TF( 1 - t ) + TF( t ) * TF( pixel( col( TA( x ) ), lig( TA( y ) ) ) ); }

    /// la masse sur le carre unite : `1` a l'arrondi de la normalisation pres, pour tout `t`
    TF masse_carre() const {
        double m = 0;
        for ( TA u : v ) m += double( u );
        m /= double( W ) * H;
        return TF( 1 - t ) + TF( t ) * TF( m );
    }

    TF max_rho() const {
        TA m = 0;
        for ( TA u : v ) m = std::max( m, u );
        return TF( 1 - t ) + TF( t ) * TF( m );
    }

    /// LA MASSE d'une cellule 2D ( `cel.nb`, `cel.vx`, `cel.vy`, `cel.cid` ), ses facettes contre
    /// les autres germes ( `facette( j, int_facette rho_t ds )` ), et si `dl` n'est pas nul la
    /// derivee de la masse par rapport au contraste `t`.
    ///
    /// La MEME SURFACE que `Densite::mesure` -- c'est ce qui permet a `Newton` de ne rien savoir
    /// de la source.
    template<class Cel, class Facette>
    TF mesure( const Cel &cel, Facette &&facette, TF *dl = nullptr ) const {
        const int nb = cel.nb;
        if ( nb <= 0 ) { if ( dl ) *dl = 0; return 0; }

        TA a2 = 0;                                       // l'aire signee, deux fois : l'ORIENTATION
        for ( int i = 0, j = nb - 1; i < nb; j = i++ )
            a2 += TA( cel.vx[ j ] ) * TA( cel.vy[ i ] ) - TA( cel.vx[ i ] ) * TA( cel.vy[ j ] );
        const TA sgn = a2 >= 0 ? TA( 1 ) : TA( -1 );     // `+1` : le tour est direct, et alors la
        const TA aire = TA( 0.5 ) * std::fabs( a2 );     // circulation rend `+ masse`

        // la reference retranchee a `G` : n'importe laquelle convient ( voir l'en-tete )
        const TA sref = s[ size_t( lig( TA( cel.vy[ 0 ] ) ) ) * W + col( TA( cel.vx[ 0 ] ) ) ];

        TA acc = 0;                                      // `int_(bord) ( G - sref ) dy`, SIGNE
        for ( int i = 0, j = nb - 1; i < nb; j = i++ ) { // l'arete [ v_j, v_i ], portee par `cid[ j ]`
            TA mas = 0, lon = 0;
            arete( TA( cel.vx[ j ] ), TA( cel.vy[ j ] ), TA( cel.vx[ i ] ), TA( cel.vy[ i ] ), sref, mas, lon );
            acc += mas;
            if ( cel.cid[ j ] >= 0 ) {
                const TA dx = TA( cel.vx[ i ] ) - TA( cel.vx[ j ] ), dy = TA( cel.vy[ i ] ) - TA( cel.vy[ j ] );
                const TA L = std::sqrt( dx * dx + dy * dy );
                facette( cel.cid[ j ], TF( ( 1 - t ) * L + t * lon ) );
            }
        }
        const TA masse_rho = sgn * acc;
        if ( dl ) *dl = TF( masse_rho - aire );          // `d / d t` : `rho - 1`
        return TF( ( 1 - t ) * aire + t * masse_rho );
    }

    /// LE PARCOURS D'UNE ARETE dans la grille ( Amanatides-Woo ), et ses DEUX sorties :
    ///   `mas` la contribution de l'arete a `int_(bord) ( G - sref ) dy`, c'est-a-dire A LA MASSE
    ///         de l'image nue ( signee : l'orientation est rendue par l'appelant ) ;
    ///   `lon` `int_arete rho ds`, c'est-a-dire LE COEFFICIENT DE HESSIENNE de l'image nue
    ///         ( `rho = 1` rend bien la longueur ).
    void arete( TA x0, TA y0, TA x1, TA y1, TA sref, TA &mas, TA &lon ) const {
        mas = 0; lon = 0;
        const TA dx = x1 - x0, dy = y1 - y0;
        const TA lg = std::sqrt( dx * dx + dy * dy );
        if ( ! ( lg > 0 ) ) return;

        constexpr TA INF = std::numeric_limits<TA>::max();
        int i = col( x0 ), j = lig( y0 );
        const int si = dx > 0 ? 1 : -1, sj = dy > 0 ? 1 : -1;
        // le `t` de la premiere frontiere verticale / horizontale, puis le pas entre deux frontieres
        TA tx = dx == 0 ? INF : ( ( dx > 0 ? ( i + 1 ) * hx : i * hx ) - x0 ) / dx;
        TA ty = dy == 0 ? INF : ( ( dy > 0 ? ( j + 1 ) * hy : j * hy ) - y0 ) / dy;
        const TA ax = dx == 0 ? INF : std::fabs( hx / dx );
        const TA ay = dy == 0 ? INF : std::fabs( hy / dy );
        tx = tx < 0 ? 0 : tx; ty = ty < 0 ? 0 : ty;      // depart pile sur une frontiere

        TA tp = 0, xp = x0;
        const int garde = W + H + 4;                     // un segment du carre n'en traverse pas plus
        for ( int g = 0; g < garde; ++g ) {
            const bool par_x = tx < ty;
            TA tn = par_x ? tx : ty;
            const bool der = ! ( tn < TA( 1 ) );
            if ( der ) tn = TA( 1 );
            const TA xc = x0 + dx * tn;
            const size_t o = size_t( j ) * W + i;
            const TA dt = tn - tp;
            mas += dt * dy * ( s[ o ] - sref + v[ o ] * ( TA( 0.5 ) * ( xp + xc ) - i * hx ) );
            lon += dt * lg * v[ o ];
            if ( der ) break;
            if ( par_x ) { i = borne( i + si, W ); tx += ax; } else { j = borne( j + sj, H ); ty += ay; }
            tp = tn; xp = xc;
        }
    }
};

using Image = ImageT<double>;

// =====================================================================================
// D'OU VIENT L'IMAGE
// =====================================================================================

/// LA CONVOLUTION, APPROCHEE ET PAS CHERE. Une moyenne glissante de demi-largeur `r` passee TROIS
/// fois ( separable : lignes puis colonnes ) approche une gaussienne a quelques pour cent pres --
/// c'est la B-spline d'ordre 3 -- et coute `O( W H )` PAR PASSE QUEL QUE SOIT `r`, parce qu'une
/// somme courante ne relit pas la fenetre. Trois boites de demi-largeur `r` ont pour variance
/// `r ( r + 1 ) / 2` en pixels : on prend `r` tel que ca vaille `( sigma W )^2`.
///
/// Les bords sont REPLIQUES : le carre est tout le domaine, il n'y a pas de masse dehors, et une
/// densite constante doit rester constante. La masse totale ne se conserve alors pas exactement --
/// on RENORMALISE apres coup, ce qui la remet a `1` : le Newton vise `1 / n` quelle que soit
/// l'etape.
///
/// La convolution n'a PAS besoin d'etre exacte : elle n'est qu'un chemin vers la vraie image, et
/// seule la derniere etape ( `sigma = 0`, l'image nue ) porte le resultat.
///
/// UN AVERTISSEMENT QUI COMPTE : ce noyau est a SUPPORT COMPACT ( `3 ( 2r + 1 )` pixels ). Une
/// region nulle plus large que ca reste NULLE apres convolution. Une vraie gaussienne y mettrait
/// `exp( -d^2 / 2 sigma^2 )`, ce qui pour `d = 10 sigma` fait `1e-22` -- numeriquement nul aussi,
/// et de toute facon inutilisable ( il faudrait une cellule d'aire `1e22` ). Le chemin par
/// convolution rencontre donc les zeros, tot ou tard ; c'est une propriete du chemin, pas du noyau.
inline void moyenne_glissante( const std::vector<double> &in, std::vector<double> &out, int W, int H, int r, bool par_ligne ) {
    const int nl = par_ligne ? H : W, nc = par_ligne ? W : H;
    const int pas = par_ligne ? 1 : W, saut = par_ligne ? W : 1;
    const double inv = 1.0 / ( 2 * r + 1 );
    for ( int l = 0; l < nl; ++l ) {
        const double *a = in.data() + size_t( l ) * saut;
        double *b = out.data() + size_t( l ) * saut;
        auto lit = [ & ]( int i ) { return a[ size_t( i < 0 ? 0 : ( i >= nc ? nc - 1 : i ) ) * pas ]; };
        double som = 0;                                  // la fenetre [ -r, r ] du premier point
        for ( int i = -r; i <= r; ++i ) som += lit( i );
        for ( int i = 0; i < nc; ++i ) {
            b[ size_t( i ) * pas ] = som * inv;
            som += lit( i + r + 1 ) - lit( i - r );      // la somme courante : deux lectures par point
        }
    }
}

/// `im.v <- brut * G_sigma`, renormalisee. `sigma` en fraction du cote du carre ; `0` rend `brut`.
template<class TA>
void convolue( const std::vector<double> &brut, ImageT<TA> &im, double sigma ) {
    const int W = im.W, H = im.H;
    const double sp = sigma * W;                         // l'ecart type, en pixels
    // `3 x boite( r )` a pour variance `r ( r + 1 ) / 2`
    int r = int( std::floor( ( std::sqrt( 1 + 8 * sp * sp ) - 1 ) / 2 + 0.5 ) );
    if ( sigma <= 0 || r < 1 ) {
        for ( size_t q = 0; q < brut.size(); ++q ) im.v[ q ] = TA( brut[ q ] );
        im.prepare();
        im.normalise();
        return;
    }
    std::vector<double> u = brut, t( brut.size() );
    for ( int passe = 0; passe < 3; ++passe ) {
        moyenne_glissante( u, t, W, H, r, true );
        moyenne_glissante( t, u, W, H, r, false );
    }
    for ( size_t q = 0; q < u.size(); ++q ) im.v[ q ] = TA( u[ q ] );
    im.prepare();
    im.normalise();
}

/// UNE IMAGE DE SYNTHESE, faite pour etre dure : un fond lisse, un disque net, une bande fine, et
/// un CARRE A ZERO. Les DISCONTINUITES et les zeros sont ce qui distingue une image d'une densite
/// reguliere -- une methode qui les rate ne se voit pas sur un fond lisse. ( La meme que
/// `gpu_des_familles`, plus le trou : sur le CPU on veut aussi voir ce qu'un zero fait au solveur. )
template<class TA>
ImageT<TA> image_synthese( int N, bool trou = true ) {
    ImageT<TA> im;
    im.W = N; im.H = N;
    im.v.assign( size_t( N ) * N, TA( 0 ) );
    for ( int j = 0; j < N; ++j )
        for ( int i = 0; i < N; ++i ) {
            const double x = ( i + 0.5 ) / N, y = ( j + 0.5 ) / N;
            double r = 0.10 + 0.9 * std::pow( std::sin( 3 * M_PI * x ) * std::cos( 2 * M_PI * y ), 2 );
            const double dx = x - 0.30, dy = y - 0.70;
            if ( dx * dx + dy * dy < 0.15 * 0.15 ) r += 3.0;                         // un disque net
            if ( std::fabs( x - y ) < 0.02 ) r += 5.0;                               // une bande fine
            if ( trou && x > 0.62 && x < 0.86 && y > 0.10 && y < 0.34 ) r = 0.0;      // un carre VIDE
            im.v[ size_t( j ) * N + i ] = TA( r );
        }
    im.prepare();
    return im;
}

/// un PGM ( P2 ascii ou P5 binaire ), la ligne du haut devenant `j = H - 1`
template<class TA>
bool lit_pgm( const char *chemin, ImageT<TA> &im ) {
    std::FILE *f = std::fopen( chemin, "rb" );
    if ( ! f ) return false;
    auto jeton = [ & ]( long &val ) {
        int c;
        while ( true ) {
            while ( ( c = std::fgetc( f ) ) != EOF && std::isspace( c ) ) {}
            if ( c == '#' ) { while ( ( c = std::fgetc( f ) ) != EOF && c != '\n' ) {} continue; }
            break;
        }
        if ( c == EOF ) return false;
        long u = 0;
        while ( c != EOF && std::isdigit( c ) ) { u = u * 10 + ( c - '0' ); c = std::fgetc( f ); }
        val = u;
        return true;
    };
    char magique[ 3 ] = {};
    if ( std::fscanf( f, "%2s", magique ) != 1 || magique[ 0 ] != 'P' || ( magique[ 1 ] != '2' && magique[ 1 ] != '5' ) ) { std::fclose( f ); return false; }
    long W = 0, H = 0, mx = 0;
    if ( ! jeton( W ) || ! jeton( H ) || ! jeton( mx ) || W <= 0 || H <= 0 || mx <= 0 ) { std::fclose( f ); return false; }
    im.W = int( W ); im.H = int( H );
    im.v.assign( size_t( W ) * H, TA( 0 ) );
    if ( magique[ 1 ] == '5' ) {
        std::vector<unsigned char> buf( size_t( W ) * H * ( mx > 255 ? 2 : 1 ) );
        if ( std::fread( buf.data(), 1, buf.size(), f ) != buf.size() ) { std::fclose( f ); return false; }
        for ( long j = 0; j < H; ++j )
            for ( long i = 0; i < W; ++i ) {
                const size_t o = size_t( j ) * W + i;
                const double u = mx > 255 ? ( buf[ 2 * o ] * 256.0 + buf[ 2 * o + 1 ] ) : double( buf[ o ] );
                im.v[ size_t( H - 1 - j ) * W + i ] = TA( u / double( mx ) );
            }
    } else {
        for ( long j = 0; j < H; ++j )
            for ( long i = 0; i < W; ++i ) {
                long u = 0;
                if ( ! jeton( u ) ) { std::fclose( f ); return false; }
                im.v[ size_t( H - 1 - j ) * W + i ] = TA( double( u ) / double( mx ) );
            }
    }
    std::fclose( f );
    im.prepare();
    return true;
}

} // namespace sf
