// =====================================================================================
// TOUS CONTRE TOUS -- le banc le plus bas qu'on puisse ecrire.
//
// Aucune structure d'acceleration, aucun elagage : chaque cellule est coupee par TOUS les
// autres diracs, dans l'ordre ou ils sont ranges. Le fournisseur est `TousLesAutres`, qui ne
// regarde jamais la cellule.
//
// CE QUE CA ISOLE. Les bancs precedents mesuraient un melange -- une liste d'opposants triee
// par distance ( donc une politique ) et un moteur de coupe. Ici il n'y a plus de politique,
// donc ce qui reste dans l'horloge est le MOTEUR : le cout d'une coupe, sur exactement le meme
// travail et le meme ordre, pour trois implementations.
//
// LE TRI MORTON ( `--morton 1` ) NE CHANGE PAS LE TRAVAIL, seulement l'ORDRE des diracs -- donc
// deux choses a la fois, qu'il faut se garder de confondre :
//   - la LOCALITE des lectures `px[ j ] / py[ j ]`, qui ne peut compter que lorsque le nuage
//     deborde du cache ;
//   - l'ORDRE D'ARRIVEE des coupes, qui decide par quels etats intermediaires la cellule passe,
//     donc combien de sommets elle porte au pire, donc combien de fois le noyau echappe.
// Le banc mesure les deux ensemble et rapporte le nombre de sommets maximum atteint, qui est la
// grandeur par laquelle le second effet se lit.
// =====================================================================================

#include "supercell/Fournisseurs.h"
#include "supercell/Noyau2DEtats.h"
#include "supercell/Noyau2DSca.h"
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

static double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

#ifndef MAXNB_BANC
#define MAXNB_BANC 64
#endif
static constexpr int MAXNB = MAXNB_BANC;

/// ---- LE CODE DE MORTON. Seize bits par axe, etales un bit sur deux, entrelaces.
static inline unsigned etale( unsigned x ) {
    x &= 0xffffu;
    x = ( x | ( x << 8 ) ) & 0x00ff00ffu;
    x = ( x | ( x << 4 ) ) & 0x0f0f0f0fu;
    x = ( x | ( x << 2 ) ) & 0x33333333u;
    x = ( x | ( x << 1 ) ) & 0x55555555u;
    return x;
}
static inline unsigned morton( float x, float y ) {
    const unsigned ix = (unsigned) std::min( 65535.f, std::max( 0.f, x * 65535.f ) );
    const unsigned iy = (unsigned) std::min( 65535.f, std::max( 0.f, y * 65535.f ) );
    return etale( ix ) | ( etale( iy ) << 1 );
}

/// UN FOURNISSEUR QUI REGARDE. Il ne change pas ce qui est propose -- il note seulement combien
/// de sommets la cellule porte au moment ou on lui demande le dirac suivant. C'est la
/// demonstration la plus courte de ce que l'interface permet : une politique peut LIRE l'etat,
/// et celle-ci ne fait que ca.
/// il transmet le `Local` de celui qu'il enveloppe, sans quoi la politique du fournisseur
/// interieur serait perdue en chemin.
template<class F>
struct Espion {
    using Local = noyau2d::Local<F>;
    static constexpr bool veut_changement = noyau2d::veut_changement<F>();

    F f;
    int *pire;
    template<class Etat> bool suivant( const Etat &e, Local &l, noyau2d::Plan &p ) {
        if ( e.nb > *pire ) *pire = e.nb;
        return f.suivant( e, l, p );
    }
};
template<class F> static Espion<F> espionne( F f, int *pire ) { return Espion<F>{ f, pire }; }

template<class T>
static double lacet( const T *vx, const T *vy, int nb ) {
    double s = 0;
    for ( int v = 0; v < nb; ++v ) {
        const int w = v + 1 < nb ? v + 1 : 0;
        s += (double) vx[ v ] * (double) vy[ w ] - (double) vx[ w ] * (double) vy[ v ];
    }
    return 0.5 * s;
}

int main( int argc, char **argv ) {
    int n = 50, rep = 0, tri = 0, autour = 0;   // rep = 0 : dimensionne d'apres n
    unsigned graine = 12345;
    for ( int i = 1; i + 1 < argc; i += 2 ) {
        if      ( ! std::string( argv[ i ] ).compare( "--n"      ) ) n      = atoi( argv[ i + 1 ] );
        else if ( ! std::string( argv[ i ] ).compare( "--rep"    ) ) rep    = atoi( argv[ i + 1 ] );
        else if ( ! std::string( argv[ i ] ).compare( "--morton" ) ) tri    = atoi( argv[ i + 1 ] );
        else if ( ! std::string( argv[ i ] ).compare( "--autour" ) ) autour = atoi( argv[ i + 1 ] );
        else if ( ! std::string( argv[ i ] ).compare( "--graine" ) ) graine = atoi( argv[ i + 1 ] );
        else { fprintf( stderr, "option inconnue %s\n", argv[ i ] ); return 1; }
    }

    if ( rep <= 0 ) {                  // ~4e7 coupes proposees par variante, quel que soit n
        const double c = (double) n * ( n - 1 );
        rep = (int) std::max( 3.0, 4e7 / ( c > 0 ? c : 1 ) );
    }

    // ---- le nuage. LE MEME dans les deux modes : seul l'ordre change.
    std::mt19937 gen( graine );
    std::uniform_real_distribution<float> u( 0, 1 );
    std::vector<float> px( n ), py( n );
    for ( int i = 0; i < n; ++i ) { px[ i ] = u( gen ); py[ i ] = u( gen ); }

    if ( tri ) {
        std::vector<int> ord( n );
        std::iota( ord.begin(), ord.end(), 0 );
        std::vector<unsigned> cle( n );
        for ( int i = 0; i < n; ++i ) cle[ i ] = morton( px[ i ], py[ i ] );
        std::sort( ord.begin(), ord.end(), [ & ]( int a, int b ) { return cle[ a ] < cle[ b ]; } );
        std::vector<float> qx( n ), qy( n );
        for ( int i = 0; i < n; ++i ) { qx[ i ] = px[ ord[ i ] ]; qy[ i ] = py[ ord[ i ] ]; }
        px.swap( qx ); py.swap( qy );
    }

    std::vector<int> nbv[ 4 ];
    std::vector<int> idv[ 4 ];
    for ( int p = 0; p < 4; ++p ) { nbv[ p ].assign( n, 0 ); idv[ p ].assign( (size_t) n * MAXNB, 0 ); }
    double aire[ 4 ] = { 0, 0, 0, 0 };
    double t[ 4 ] = { 1e30, 1e30, 1e30, 1e30 };
    int nb_echap = 0, nb_debord = 0, nb_abandon = 0;

    // ---- TOUT LE BANC EST UN PATRON SUR LE FOURNISSEUR. Une politique, une instanciation :
    // le fournisseur n'est jamais un objet polymorphe, et le noyau reste specialise pour lui.
    auto banc = [ & ]( auto fabrique ) {

    // ---- le pire nombre de sommets atteint, et combien de cellules font une EXCURSION,
    // HORS chronometre. L'espion ne change rien a ce qui est propose : il regarde passer
    // l'`Etat`, ce que l'interface permet et qui ne coute rien quand personne ne le fait.
    int pire = 0, nb_excur = 0;
    for ( int i = 0; i < n; ++i ) {
        noyau2d::Atelier<MAXNB> at;
        int pi = 0;
        auto e = espionne( fabrique( i ), &pi );
        noyau2d::en_place<MAXNB>( &at, &e );
        if ( pi > 8 ) ++nb_excur;
        if ( pi > pire ) pire = pi;
    }

    // ================= 0. SIMD, ALLER-RETOUR ( atelier de MAXNB ) =================
    for ( int r = 0; r < rep; ++r ) {
        double a = 0; int ech = 0;
        const double t0 = now();
        for ( int i = 0; i < n; ++i ) {
            noyau2d::Atelier<MAXNB> at;
            auto f = fabrique( i );
            noyau2d::etats::moteur( &f, &at );
            nbv[ 0 ][ i ] = at.nb;
            if ( at.nb < 0 ) { ++ech; continue; }
            for ( int v = 0; v < at.nb; ++v ) idv[ 0 ][ (size_t) i * MAXNB + v ] = at.cid[ v ];
            a += lacet( at.vx, at.vy, at.nb );
        }
        const double dt = now() - t0;
        if ( dt < t[ 0 ] ) { t[ 0 ] = dt; aire[ 0 ] = a; nb_echap = ech; }
    }

    // ================= 3. SIMD, ECHAPPEMENT ( atelier de 8 : l'ancien comportement ) ======
    //
    // Il ne rend PAS un resultat complet -- les cellules de plus de huit sommets sont
    // abandonnees. Son temps n'est donc pas comparable aux autres : une cellule qui echappe
    // cesse de travailler. Il est ici pour une seule question : ce que l'aller-retour coute.
    for ( int r = 0; r < rep; ++r ) {
        int ech = 0;
        const double t0 = now();
        for ( int i = 0; i < n; ++i ) {
            noyau2d::Atelier<8> at;
            auto f = fabrique( i );
            noyau2d::etats::moteur( &f, &at );
            if ( at.nb < 0 ) { ++ech; continue; }
            nbv[ 3 ][ i ] = at.nb;
        }
        const double dt = now() - t0;
        if ( dt < t[ 3 ] ) { t[ 3 ] = dt; nb_abandon = ech; }
    }

    // ================= 1. SCALAIRE EN PLACE / 2. SCALAIRE TAMPON NEUF =================
    //
    // UN PATRON PAR VARIANTE, une boucle par patron. Une seule boucle avec `if ( p == 1 )` au
    // centre du chronometre donnait au tampon neuf 18.7 ns la ou il en vaut 13.7 : le test
    // reste dans le corps, la fonction n'est plus specialisee, et c'est le banc qu'on mesure.
    auto mesure = [ & ]( int p, auto &&appel ) {
        for ( int r = 0; r < rep; ++r ) {
            double a = 0; int deb = 0;
            const double t0 = now();
            for ( int i = 0; i < n; ++i ) {
                noyau2d::Atelier<MAXNB> at;
                auto f = fabrique( i );
                appel( &at, &f );
                nbv[ p ][ i ] = at.nb;
                if ( at.nb < 0 ) { ++deb; continue; }
                for ( int v = 0; v < at.nb; ++v ) idv[ p ][ (size_t) i * MAXNB + v ] = at.cid[ v ];
                a += lacet( at.vx, at.vy, at.nb );
            }
            const double dt = now() - t0;
            if ( dt < t[ p ] ) { t[ p ] = dt; aire[ p ] = a; nb_debord = deb; }
        }
    };
    mesure( 1, []( auto *at, auto *f ) { noyau2d::en_place   <MAXNB>( at, f ); } );
    mesure( 2, []( auto *at, auto *f ) { noyau2d::tampon_neuf<MAXNB>( at, f ); } );

    // ---- justesse : meme cycle de `cid`, a rotation pres, entre les trois.
    auto rot_eq = []( const int *a, const int *b, int m ) {
        for ( int r = 0; r < m; ++r ) {
            bool ok = true;
            for ( int v = 0; v < m; ++v )
                if ( a[ ( v + r ) % m ] != b[ v ] ) { ok = false; break; }
            if ( ok ) return true;
        }
        return m == 0;
    };
    int faux_simd = 0, faux_sca = 0, compares = 0;
    for ( int i = 0; i < n; ++i ) {
        if ( nbv[ 1 ][ i ] < 0 || nbv[ 2 ][ i ] < 0 ) continue;
        if ( nbv[ 1 ][ i ] != nbv[ 2 ][ i ] ||
             ! rot_eq( &idv[ 1 ][ (size_t) i * MAXNB ], &idv[ 2 ][ (size_t) i * MAXNB ], nbv[ 2 ][ i ] ) ) ++faux_sca;
        if ( nbv[ 0 ][ i ] < 0 ) continue;
        ++compares;
        if ( nbv[ 0 ][ i ] != nbv[ 1 ][ i ] ||
             ! rot_eq( &idv[ 0 ][ (size_t) i * MAXNB ], &idv[ 1 ][ (size_t) i * MAXNB ], nbv[ 1 ][ i ] ) ) ++faux_simd;
    }

    const double coupes = (double) n * ( n - 1 );
    const char *nom[ 4 ] = { "SIMD aller-ret", "scalaire place", "scalaire neuf ", "SIMD echappe  " };
    printf( "%d diracs, tous contre tous, aucune acceleration -- %.0f coupes proposees par passe\n",
            n, coupes );
    printf( "nuage : %s   parcours : %s   -- %d passes, on garde la meilleure\n\n",
            tri ? "TRI MORTON" : "brut", autour ? "en s'ecartant de soi" : "depuis l'indice 0", rep );
    printf( "                   temps/passe    par cellule    par coupe proposee\n" );
    for ( int p = 0; p < 4; ++p )
        printf( "%s   %9.2f us     %7.1f ns        %6.3f ns%s\n",
                nom[ p ], t[ p ] * 1e6, t[ p ] * 1e9 / n, t[ p ] * 1e9 / coupes,
                p == 3 ? "   <- incomplet" : "" );
    const double best_sca = std::min( t[ 1 ], t[ 2 ] );
    printf( "\nSIMD / meilleur scalaire : x%.2f   ( place / neuf : x%.2f )\n\n",
            best_sca / t[ 0 ], t[ 1 ] / t[ 2 ] );
    printf( "sommets au pire ( etats intermediaires ) : %d\n", pire );
    printf( "cellules faisant une EXCURSION ( > 8 )   : %d / %d\n", nb_excur, n );
    printf( "  abandonnees par l'atelier de 8         : %d\n", nb_abandon );
    printf( "  non terminees par l'aller-retour       : %d\n", nb_echap );
    printf( "debordements scalaire ( > %d )           : %d\n", MAXNB, nb_debord );
    printf( "aires : SIMD %.9f  place %.9f  neuf %.9f\n", aire[ 0 ], aire[ 1 ], aire[ 2 ] );
    printf( "combinatoire : %d fausses SIMD/%d comparees, %d entre les deux scalaires\n",
            faux_simd, compares, faux_sca );
    printf( "CSV n=%d morton=%d autour=%d simd=%.3f place=%.3f neuf=%.3f ech=%.3f gain=%.3f pire=%d excur=%d faux=%d\n",
            n, tri, autour, t[ 0 ] * 1e9 / coupes, t[ 1 ] * 1e9 / coupes, t[ 2 ] * 1e9 / coupes,
            t[ 3 ] * 1e9 / coupes, best_sca / t[ 0 ], pire, nb_excur, faux_simd + faux_sca );
    }; // fin du patron `banc`

    if ( autour ) banc( [ & ]( int i ) { return noyau2d::AutourDeMoi ( px.data(), py.data(), n, i ); } );
    else          banc( [ & ]( int i ) { return noyau2d::TousLesAutres( px.data(), py.data(), n, i ); } );
    return 0;
}
