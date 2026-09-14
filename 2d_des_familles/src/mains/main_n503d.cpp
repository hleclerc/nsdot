// =====================================================================================
// LA CELLULE 3D QUI DIRIGE -- tous contre tous, aucune structure d'acceleration.
//
// Le pendant 3D de `pd_n50`, et pour la meme raison : aucune politique dans l'horloge, donc ce qui
// reste est le COUT D'UNE COUPE. Le fournisseur propose tous les autres germes, le nuage est trie
// en Morton, et le parcours part du germe courant pour s'en ecarter.
//
// = LES DEUX TEMOINS
//
//   1. LA SOMME DES VOLUMES DOIT VALOIR 1. C'est le seul controle qui teste la COMPLETUDE : si une
//      coupe est ratee, une cellule deborde sur sa voisine et la somme monte.
//
//   2. `Cell3T`, l'implementation POUSSEE qui existait avant, recoit exactement la meme suite de
//      plans. Elle est en `double` la ou la nouvelle est en `float`, donc les volumes ne peuvent
//      pas coincider au bit pres -- mais la COMBINATOIRE, elle, doit coincider exactement : meme
//      nombre de sommets, meme nombre d'aretes, meme ensemble de voisins. C'est la seule chose qui
//      distingue une erreur d'algorithme d'un ecart d'arrondi.
// =====================================================================================

#include "geometry/Cell3.h"
#include "supercell/Fournisseurs3D.h"
#include "supercell/Noyau3D.h"
#include "util/common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace pd;

static double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

#ifndef MAXNV_BANC
#define MAXNV_BANC 128
#endif
static constexpr int MAXNV = MAXNV_BANC;

/// LA LARGEUR SIMD DE LA PREMIERE PASSE, en voies de `float`. Un binaire par largeur : deux
/// instanciations dans le meme executable se genent dans le cache d'instructions et la comparaison
/// ne veut plus rien dire ( meme lecon qu'en 2D, voir l'en-tete de `main_n50.cpp` ).
#ifndef LARGEUR_BANC
#define LARGEUR_BANC 8
#endif
using Cel3 = noyau3d::Cellule3<MAXNV,LARGEUR_BANC>;

/// LE CODE DE MORTON EN 3D : dix bits par axe, etales deux bits sur trois, entrelaces.
static inline unsigned etale3( unsigned x ) {
    x &= 0x3ffu;
    x = ( x | ( x << 16 ) ) & 0x030000ffu;
    x = ( x | ( x <<  8 ) ) & 0x0300f00fu;
    x = ( x | ( x <<  4 ) ) & 0x030c30c3u;
    x = ( x | ( x <<  2 ) ) & 0x09249249u;
    return x;
}
static inline unsigned morton3( float x, float y, float z ) {
    auto q = []( float v ) { return (unsigned) std::min( 1023.f, std::max( 0.f, v * 1023.f ) ); };
    return etale3( q( x ) ) | ( etale3( q( y ) ) << 1 ) | ( etale3( q( z ) ) << 2 );
}

int main( int argc, char **argv ) {
    int n = 50, rep = 0, tri = 1, autour = 1, seuil = 0;   // seuil 0 = le defaut de la cellule
    unsigned graine = 12345;
    double frac = 0;                                     // poids en fraction de h^2
    for ( int i = 1; i + 1 < argc; i += 2 ) {
        const std::string o = argv[ i ];
        if      ( o == "--n"      ) n      = atoi( argv[ i + 1 ] );
        else if ( o == "--rep"    ) rep    = atoi( argv[ i + 1 ] );
        else if ( o == "--morton" ) tri    = atoi( argv[ i + 1 ] );
        else if ( o == "--autour" ) autour = atoi( argv[ i + 1 ] );
        else if ( o == "--graine" ) graine = atoi( argv[ i + 1 ] );
        else if ( o == "--compacte" ) seuil = atoi( argv[ i + 1 ] );
        else if ( o == "--poids"  ) frac = atof( argv[ i + 1 ] );
        else { fprintf( stderr, "option inconnue %s\n", argv[ i ] ); return 1; }
    }
    if ( rep <= 0 ) {                                    // ~4e6 coupes proposees par passe
        const double c = (double) n * ( n - 1 );
        rep = (int) std::max( 3.0, 4e6 / ( c > 0 ? c : 1 ) );
    }

    std::mt19937 gen( graine );
    std::uniform_real_distribution<float> u( 0, 1 );
    std::vector<float> px( n ), py( n ), pz( n ), pw( n, 0.f );
    // LES POIDS A L'ECHELLE : un plan est decale de `dw / ( 2 |p1 - p0| )`, donc pour que le
    // decalage soit une fraction de l'espacement `h ~ n^(-1/3)` il faut `dw ~ h^2`.
    const double h2 = std::pow( double( n ), -2.0 / 3.0 );
    for ( int i = 0; i < n; ++i ) {
        px[i] = u( gen ); py[i] = u( gen ); pz[i] = u( gen );
        pw[i] = float( frac * h2 * u( gen ) );
    }

    if ( tri ) {
        std::vector<int> ord( n );
        std::iota( ord.begin(), ord.end(), 0 );
        std::vector<unsigned> cle( n );
        for ( int i = 0; i < n; ++i ) cle[ i ] = morton3( px[i], py[i], pz[i] );
        std::sort( ord.begin(), ord.end(), [&]( int a, int b ) { return cle[a] < cle[b]; } );
        std::vector<float> qx( n ), qy( n ), qz( n );
        for ( int i = 0; i < n; ++i ) { qx[i] = px[ord[i]]; qy[i] = py[ord[i]]; qz[i] = pz[ord[i]]; }
        px.swap( qx ); py.swap( qy ); pz.swap( qz );
    }

    // TOUT LE BANC EST UN PATRON SUR LE FOURNISSEUR, comme en 2D : une politique, une
    // instanciation, et le noyau reste specialise pour elle. Le fournisseur n'est jamais un objet
    // polymorphe.
    int pire_nv = 0, pire_ne = 0, pire_nc = 0, deborde = 0, deborde_ref = 0;
    long long som_nv = 0, som_ne = 0, som_nc = 0, som_vois = 0;
    double t = 1e30, vol = 0, t_ref = 1e30, vol_ref = 0;
    long long compactions = 0;
    int faux_nv = 0, faux_ne = 0, faux_vois = 0, nok = 0;
    long long prop = 0, eff = 0;
    int faux_adj = 0;                                    ///< l'invariant d'adjacence ( variante ne )                         ///< nuls si le binaire ne compte pas

    auto banc = [ & ]( auto fabrique ) {

    // ---- HORS CHRONOMETRE : les tailles atteintes, et ce que la liste de coupes coute
    {
        Cel3 c;
        if ( seuil ) c.seuil_compacte = seuil;
        for ( int i = 0; i < n; ++i ) {
            auto f = fabrique( i );
            if ( noyau3d::moteur( &f, &c ) ) { ++deborde; continue; }
            pire_nv = std::max( pire_nv, c.nv );
            pire_ne = std::max( pire_ne, c.ne );
            pire_nc = std::max( pire_nc, c.nc );
            som_nv += c.nv; som_ne += c.ne; som_nc += c.nc;
            int vo[ MAXNV ];
            som_vois += c.voisins( vo, MAXNV );
            faux_adj += c.verifie();             // l'invariant d'adjacence, sur chaque cellule
        }
        nok = n - deborde;
#if NOYAU3D_COMPTE
        prop = c.nb_prop; eff = c.nb_eff;
#endif
    }

    // ---- LE CHRONOMETRE
    for ( int r = 0; r < rep; ++r ) {
        double v = 0;
        const double t0 = now();
        Cel3 c;
        if ( seuil ) c.seuil_compacte = seuil;
        c.nb_compactions = 0;
        for ( int i = 0; i < n; ++i ) {
            auto f = fabrique( i );
            if ( noyau3d::moteur( &f, &c ) ) continue;
            v += c.volume();
        }
        compactions = c.nb_compactions;
        const double dt = now() - t0;
        if ( dt < t ) { t = dt; vol = v; }
    }

    // ---- LE TEMOIN : `Cell3T`, MEME suite de plans, en `double`. Le fournisseur est le meme
    // objet, deroule a la main -- c'est la seule facon d'etre sur que les deux voient les memes
    // plans dans le meme ordre.
    auto deroule_ref = [ & ]( Cell3T<MAXNV> &c, int i ) {
        c.init_as_unit_cube();
        auto f = fabrique( i );
        noyau3d::RienDeLocal rl;
        noyau3d::Plan3 p;
        while ( f.suivant( noyau3d::EtatCell3{ 0, nullptr, nullptr, nullptr }, rl, p ) ) {
            const CutResult cr = c.cut( Vec<3>{ p.dx, p.dy, p.dz }, p.off, p.id );
            if ( cr == CutResult::overflow ) return false;
            if ( cr == CutResult::empty ) return true;
        }
        return true;
    };
    for ( int r = 0; r < rep; ++r ) {
        double v = 0; int dr = 0;
        const double t0 = now();
        for ( int i = 0; i < n; ++i ) {
            Cell3T<MAXNV> c;
            if ( ! deroule_ref( c, i ) ) { ++dr; continue; }
            v += c.measure();
        }
        const double dt = now() - t0;
        if ( dt < t_ref ) { t_ref = dt; vol_ref = v; deborde_ref = dr; }
    }

    // ---- LA COMPARAISON COMBINATOIRE, cellule par cellule
    {
        Cel3 a;
        if ( seuil ) a.seuil_compacte = seuil;
        std::vector<int> va, vb;
        for ( int i = 0; i < n; ++i ) {
            auto f = fabrique( i );
            if ( noyau3d::moteur( &f, &a ) ) continue;
            Cell3T<MAXNV> b;
            if ( ! deroule_ref( b, i ) ) continue;

            if ( a.nv != b.nb ) { ++faux_nv; continue; }
            if ( a.ne != b.ne ) ++faux_ne;

            int vo[ MAXNV ]; const int m = a.voisins( vo, MAXNV );
            va.assign( vo, vo + m );
            vb.clear();
            for ( SI k = 0; k < b.nb; ++k )
                for ( int q = 0; q < 3; ++q ) vb.push_back( (int) b.vc[k][q] );
            std::sort( va.begin(), va.end() );
            std::sort( vb.begin(), vb.end() );
            vb.erase( std::unique( vb.begin(), vb.end() ), vb.end() );
            if ( va != vb ) ++faux_vois;
        }
    }

    }; // fin du patron `banc`

    #define LANCE( P )                                                                          \
        ( autour ? banc( [ & ]( int i ) { return noyau3d::AutourDeMoi3<P>(                       \
                              px.data(), py.data(), pz.data(), pw.data(), n, i ); } )            \
                 : banc( [ & ]( int i ) { return noyau3d::TousLesAutres3<P>(                     \
                              px.data(), py.data(), pz.data(), pw.data(), n, i ); } ) )
    if ( frac > 0 ) LANCE( true );
    else            LANCE( false );
    #undef LANCE

    const double coupes = (double) n * ( n - 1 );
    printf( "%d diracs en 3D, tous contre tous, aucune acceleration -- %.0f coupes proposees par passe\n",
            n, coupes );
    printf( "largeur SIMD %d voies   ( MaxNv = %d )\n", LARGEUR_BANC, MAXNV );
    printf( "nuage : %s %s   parcours : %s   -- %d passes, on garde la meilleure\n\n",
            tri ? "TRI MORTON" : "brut", frac > 0 ? "/ Laguerre" : "/ Voronoi",
            autour ? "en s'ecartant de soi" : "depuis l'indice 0", rep );

    printf( "                    temps/passe    par cellule    par coupe proposee\n" );
    printf( "cellule qui dirige  %9.2f us     %7.1f ns        %6.3f ns\n",
            t * 1e6, t * 1e9 / n, t * 1e9 / coupes );
    printf( "Cell3T ( poussee )  %9.2f us     %7.1f ns        %6.3f ns   <- en double\n",
            t_ref * 1e6, t_ref * 1e9 / n, t_ref * 1e9 / coupes );
    printf( "\nvolumes : cellule %.9f   Cell3T %.9f   <- doivent valoir 1\n", vol, vol_ref );
    printf( "debordements ( MaxNv = %d ) : %d ici, %d dans Cell3T\n", MAXNV, deborde, deborde_ref );
    printf( "\ncombinatoire contre Cell3T : %d sommets, %d aretes, %d voisinages differents / %d\n",
            faux_nv, faux_ne, faux_vois, nok );
    printf( "\ntailles atteintes ( sur %d cellules ) :\n", nok );
    printf( "  sommets  moyenne %5.2f   pire %3d      ( Euler : E = 3 V / 2, F = 2 + V / 2 )\n",
            double( som_nv ) / nok, pire_nv );
    printf( "  aretes   moyenne %5.2f   pire %3d\n", double( som_ne ) / nok, pire_ne );
    printf( "  voisins  moyenne %5.2f                ( = les faces VIVANTES )\n",
            double( som_vois ) / nok );
    printf( "  liste de coupes moyenne %5.2f   pire %3d   <- dont %.2f mortes\n",
            double( som_nc ) / nok, pire_nc, double( som_nc - som_vois ) / nok );
    printf( "  compactions : %lld pour %d cellules ( seuil %d )\n",
            compactions, n, seuil ? seuil : Cel3::max_nc );
    printf( "\nCSV3D n=%d morton=%d autour=%d cell=%.3f ref=%.3f vol=%.9f faux=%d nv=%.2f nc=%.2f pire_nc=%d comp=%lld seuil=%d prop=%lld eff=%lld w=%d adj=%d\n",
            n, tri, autour, t * 1e9 / coupes, t_ref * 1e9 / coupes, vol,
            faux_nv + faux_ne + faux_vois, double( som_nv ) / nok, double( som_nc ) / nok, pire_nc,
            compactions, seuil ? seuil : Cel3::max_nc, prop, eff, LARGEUR_BANC, faux_adj );
    return 0;
}
