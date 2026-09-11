// LE BSP ALIGNE SUR LES AXES -- l'accelerateur de reference du banc, celui auquel tous les autres
// se comparent.
//
//   xmake run pd_bsp --help

#include "spatial_accel/AaBsp.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include "bench/Bench.h"
#include <cstdio>
#include <string>

using namespace pd;
using namespace pd::bench;

namespace {

/// CE QUE VAUDRAIT UN MAJORANT DE DEGRE 2, mesure AVANT de l'ecrire.
///
/// Un majorant vaut ce que vaut l'ETALEMENT de ses residus, `max( w - q ) - min( w - q )` : c'est
/// exactement le mou de la borne, puisque `b` est releve jusqu'au pire germe et que tous les autres
/// paient l'ecart. On compare donc, noeud par noeud, l'etalement de quatre ajustements :
///
///   constant  [ 1 ]                        -- la borne classique
///   affine    [ 1, x, y ]                  -- celle qu'on vient de mettre, en O( 1 )
///   diagonal  [ 1, x, y, x^2, y^2 ]        -- degre 2 SANS terme croise, encore en O( 1 )
///   complet   [ 1, x, y, x^2, y^2, x y ]   -- degre 2 entier, qui lui n'est PAS separable
///
/// Le complet n'est pas utilisable ; il est la pour dire si c'est le degre qui manque ou la
/// restriction a la diagonale.
///
/// La precaution qui decide de tout : un ajustement a `k` parametres sur `m` points resserre
/// l'etalement MEME QUAND IL N'Y A RIEN A AJUSTER, d'un facteur `sqrt( 1 - ( k - 1 ) / ( m - 1 ) )`.
/// Une feuille a dix germes et le diagonal cinq parametres : il y gagnerait 0.75 sur du bruit pur.
/// On affiche donc ce plancher a cote, et seul ce qui passe NETTEMENT dessous compte.
double fit_spread( const std::vector<double> &B, const std::vector<double> &w, int m, int nb ) {
    // equations normales `B^T B c = B^T w`, resolues par Gauss avec pivot partiel. `nb <= 6`, donc
    // le cout ne compte pas : c'est un diagnostic, pas un chemin chaud.
    std::vector<double> A( nb * nb, 0 ), r( nb, 0 );
    for ( int i = 0; i < m; ++i ) {
        for ( int u = 0; u < nb; ++u ) {
            r[ u ] += B[ i * nb + u ] * w[ i ];
            for ( int v = 0; v < nb; ++v )
                A[ u * nb + v ] += B[ i * nb + u ] * B[ i * nb + v ];
        }
    }
    for ( int u = 0; u < nb; ++u ) {
        int piv = u;
        for ( int v = u + 1; v < nb; ++v )
            if ( std::fabs( A[ v * nb + u ] ) > std::fabs( A[ piv * nb + u ] ) )
                piv = v;
        if ( std::fabs( A[ piv * nb + u ] ) < 1e-300 )
            return -1;                                  // systeme degenere : on ne compte pas
        if ( piv != u ) {
            for ( int v = 0; v < nb; ++v )
                std::swap( A[ u * nb + v ], A[ piv * nb + v ] );
            std::swap( r[ u ], r[ piv ] );
        }
        for ( int v = u + 1; v < nb; ++v ) {
            const double f = A[ v * nb + u ] / A[ u * nb + u ];
            for ( int k = u; k < nb; ++k )
                A[ v * nb + k ] -= f * A[ u * nb + k ];
            r[ v ] -= f * r[ u ];
        }
    }
    for ( int u = nb - 1; u >= 0; --u ) {
        for ( int v = u + 1; v < nb; ++v )
            r[ u ] -= A[ u * nb + v ] * r[ v ];
        r[ u ] /= A[ u * nb + u ];
    }

    double lo = 1e300, hi = -1e300;
    for ( int i = 0; i < m; ++i ) {
        double f = 0;
        for ( int u = 0; u < nb; ++u )
            f += B[ i * nb + u ] * r[ u ];
        const double e = w[ i ] - f;
        lo = std::min( lo, e ); hi = std::max( hi, e );
    }
    return hi - lo;
}

int majorant_stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    if ( ! W ) {
        std::printf( "  pas de poids : rien a majorer\n" );
        return 0;
    }
    AaBsp bs;
    bs.build( X.data(), Y.data(), W, a.n, a.leaf );

    // par TAILLE de noeud : c'est elle qui dit si le majorant sert a rejeter un sous-arbre entier
    // (gros noeud, haut dans l'arbre) ou juste une feuille.
    struct Acc { double s1 = 0, s2 = 0, s3 = 0, chance2 = 0, chance3 = 0; long long m = 0, n = 0; };
    std::vector<Acc> by( 32 );

    for ( const AaBsp::Node &nd : bs.nodes ) {
        const int m = int( nd.end - nd.beg );
        if ( m < 12 )
            continue;                                   // trop peu de points pour que 6 parametres aient un sens
        int bucket = 0;
        while ( ( 1 << ( bucket + 1 ) ) <= m )
            ++bucket;

        double mx = 0, my = 0;
        for ( SI k = nd.beg; k < nd.end; ++k ) { mx += bs.p[ 0 ][ k ]; my += bs.p[ 1 ][ k ]; }
        mx /= m; my /= m;

        std::vector<double> B( size_t( m ) * 6 ), w( m );
        for ( int i = 0; i < m; ++i ) {
            const SI k = nd.beg + i;
            const double qx = bs.p[ 0 ][ k ] - mx, qy = bs.p[ 1 ][ k ] - my;
            B[ i * 6 + 0 ] = 1; B[ i * 6 + 1 ] = qx; B[ i * 6 + 2 ] = qy;
            B[ i * 6 + 3 ] = qx * qx; B[ i * 6 + 4 ] = qy * qy; B[ i * 6 + 5 ] = qx * qy;
            w[ i ] = bs.pw[ k ];
        }
        // les colonnes ne sont pas contigues pour `nb < 6` : on recopie le prefixe voulu.
        auto sub = [ & ]( int nb ) {
            std::vector<double> C( size_t( m ) * nb );
            for ( int i = 0; i < m; ++i )
                for ( int u = 0; u < nb; ++u )
                    C[ i * nb + u ] = B[ i * 6 + u ];
            return fit_spread( C, w, m, nb );
        };

        const double s0 = sub( 1 ), s1 = sub( 3 ), s2 = sub( 5 ), s3 = fit_spread( B, w, m, 6 );
        if ( s0 <= 0 || s1 < 0 || s2 < 0 || s3 < 0 )
            continue;
        Acc &A = by[ bucket ];
        A.s1 += s1 / s0; A.s2 += s2 / s0; A.s3 += s3 / s0;
        A.chance2 += std::sqrt( std::max( 0.0, 1.0 - 4.0 / ( m - 1 ) ) );
        A.chance3 += std::sqrt( std::max( 0.0, 1.0 - 5.0 / ( m - 1 ) ) );
        A.m += m; ++A.n;
    }

    std::printf( "  etalement des residus, RAPPORTE au majorant constant (plus petit = plus serre)\n" );
    std::printf( "  %-14s %7s   %8s   %8s %8s   %8s %8s\n",
                 "germes/noeud", "noeuds", "affine", "diagonal", "(hasard)", "complet", "(hasard)" );
    for ( size_t b = 0; b < by.size(); ++b ) {
        const Acc &A = by[ b ];
        if ( ! A.n )
            continue;
        std::printf( "  %6d - %-6d %7lld   %8.3f   %8.3f %8.3f   %8.3f %8.3f\n",
                     1 << b, ( 2 << b ) - 1, A.n, A.s1 / A.n,
                     A.s2 / A.n, A.chance2 / A.n, A.s3 / A.n, A.chance3 / A.n );
    }
    return 0;
}


} // namespace

int main( int argc, char **argv ) {
    Args a;
    bool stats = false, majorant = false;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( parse_commun( a, s, i, argc, argv ) )
            continue;
        if ( s == "--stats" )         stats = true;
        else if ( s == "--majorant" ) majorant = true;
        else {
            std::printf( "usage: pd_bsp [options]\n" );
            usage_commun();
            std::printf(
                "  --stats         compte boites/coupes par cellule au lieu de chronometrer\n"
                "  --majorant      de combien le majorant AFFINE des poids resserre la borne\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    finalise( a );

    if ( stats ) {
        if ( a.dims != 3 ) {
            std::printf( "=== 2D  ce que le parcours fait par cellule\n" );
            for ( const Cloud<2> &cl : suite_2d( a ) ) {
                if ( cl.absent ) continue;
                std::printf( "  %s\n", cl.nom.c_str() );
                AaBsp tr; tr.build( cl.P, cl.W, cl.n, a.leaf );
                stats_une<AaBsp, CellSoAT<64>, true>( tr );
            }
        }
        if ( a.dims != 2 ) {
            std::printf( "=== 3D  ce que le parcours fait par cellule\n" );
            for ( const Cloud<3> &cl : suite_3d( a ) ) {
                if ( cl.absent ) continue;
                std::printf( "  %s\n", cl.nom.c_str() );
                AaBsp3 tr; tr.build( cl.P, cl.W, cl.n, a.leaf );
                stats_une<AaBsp3, Cell3T<128>, true>( tr );
            }
        }
        return 0;
    }

    if ( majorant ) {
        int bad = 0;
        for ( const Cloud<2> &cl : suite_2d( a ) ) {
            if ( cl.absent ) continue;
            std::printf( "=== 2D  %s\n", cl.nom.c_str() );
            Args b = a;
            b.n = cl.n;
            bad += majorant_stats( b, cl.c[ 0 ], cl.c[ 1 ], cl.W );
        }
        return bad;
    }

    return banc<AaBsp, AaBsp3>( a );
}
