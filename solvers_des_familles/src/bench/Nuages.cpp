#include "bench/Nuages.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace sf {

template<int D>
Nuage<D> nuage_uniforme( SI n, unsigned graine, double wscale ) {
    Nuage<D> nu;
    nu.nom = "uniforme n=" + std::to_string( n );
    std::mt19937_64 rng( graine );
    std::uniform_real_distribution<TF> uni( 0.001, 0.999 );
    for ( int d = 0; d < D; ++d ) {
        nu.c[ d ].resize( n );
        for ( SI i = 0; i < n; ++i ) nu.c[ d ][ i ] = uni( rng );
    }
    nu.w.assign( n, TF( 0 ) );
    if ( wscale != 0 ) {
        const TF h = std::pow( TF( 1 ) / n, TF( 1 ) / D );
        std::uniform_real_distribution<TF> uw( -wscale * h * h, wscale * h * h );
        for ( SI i = 0; i < n; ++i ) nu.w[ i ] = uw( rng );
    }
    nu.finish();
    return nu;
}

/// `strtod` sur un tampon lu d'un coup : a 1e6 germes, `operator>>` coute plus que la mesure.
template<int D>
bool charge_nuage( const std::string &chemin, Nuage<D> &nu, bool bavard ) {
    std::FILE *f = std::fopen( chemin.c_str(), "rb" );
    if ( ! f ) {
        if ( bavard ) std::printf( "impossible d'ouvrir '%s'\n", chemin.c_str() );
        return false;
    }
    std::fseek( f, 0, SEEK_END );
    const long sz = std::ftell( f );
    std::fseek( f, 0, SEEK_SET );
    std::string buf( size_t( sz ) + 1, '\0' );
    const size_t rd = std::fread( buf.data(), 1, size_t( sz ), f );
    std::fclose( f );
    buf.resize( rd + 1 );
    buf[ rd ] = 0;

    const char *p = buf.data(), *e = p + rd;
    auto num = [ & ]( TF &out ) {
        for ( ;; ) {
            while ( p < e && ( *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' ) ) ++p;
            if ( p < e && *p == '#' ) { while ( p < e && *p != '\n' ) ++p; continue; }
            break;
        }
        if ( p >= e ) return false;
        char *q = nullptr;
        out = std::strtod( p, &q );
        if ( q == p ) return false;
        p = q;
        return true;
    };

    TF v;
    if ( ! num( v ) ) {
        if ( bavard ) std::printf( "'%s' : pas de nombre de germes\n", chemin.c_str() );
        return false;
    }
    const SI n = SI( v );
    for ( int d = 0; d < D; ++d ) nu.c[ d ].assign( n, TF( 0 ) );
    nu.w.assign( n, TF( 0 ) );
    for ( SI i = 0; i < n; ++i ) {
        bool ok = true;
        for ( int d = 0; d < D && ok; ++d ) ok = num( nu.c[ d ][ i ] );
        if ( ok ) ok = num( nu.w[ i ] );
        if ( ! ok ) {
            if ( bavard ) std::printf( "'%s' : tronque au germe %d\n", chemin.c_str(), int( i ) );
            return false;
        }
    }
    nu.finish();
    return true;
}

template<int D>
bool ecrit_nuage( const std::string &chemin, const Nuage<D> &nu, const TF *W, const std::string &entete ) {
    std::FILE *f = std::fopen( chemin.c_str(), "w" );
    if ( ! f )
        return false;
    std::fprintf( f, "%s", entete.c_str() );
    std::fprintf( f, "%d\n", int( nu.n ) );
    for ( SI i = 0; i < nu.n; ++i ) {
        for ( int d = 0; d < D; ++d )
            std::fprintf( f, "%.17g ", double( nu.P[ d ][ i ] ) );
        std::fprintf( f, "%.17g\n", double( W ? W[ i ] : TF( 0 ) ) );
    }
    std::fclose( f );
    return true;
}

namespace {
template<int D>
Nuage<D> depuis_fichier( const std::string &chemin, const char *nom, bool temoin ) {
    Nuage<D> nu;
    nu.nom = nom;
    nu.temoin = temoin;
    if ( ! charge_nuage<D>( chemin, nu, false ) )
        nu.absent = true;
    return nu;
}
} // namespace

template<int D>
std::vector<Nuage<D>> suite( SI n, unsigned graine, double wscale, const std::string &cases ) {
    std::vector<Nuage<D>> r;
    r.push_back( nuage_uniforme<D>( n, graine, wscale ) );
    if constexpr ( D == 2 ) {
        r.push_back( depuis_fichier<2>( cases + "/lines5_n100000_s0.005_voronoi.txt", "lignes / Voronoi", false ) );
        r.push_back( depuis_fichier<2>( cases + "/lines5_n100000_s0.005_equal.txt", "lignes / aires egales", true ) );
    } else {
        r.push_back( depuis_fichier<3>( cases + "/planes4_n100000_s0.02_voronoi.txt", "plans / Voronoi", false ) );
        // fabrique par le banc lui-meme, donc un cas de chronometrage et PAS un temoin independant
        r.push_back( depuis_fichier<3>( cases + "/planes4_n100000_s0.02_equal.txt", "plans / volumes egaux", false ) );
    }
    return r;
}

template Nuage<2> nuage_uniforme<2>( SI, unsigned, double );
template Nuage<3> nuage_uniforme<3>( SI, unsigned, double );
template bool charge_nuage<2>( const std::string &, Nuage<2> &, bool );
template bool charge_nuage<3>( const std::string &, Nuage<3> &, bool );
template bool ecrit_nuage<2>( const std::string &, const Nuage<2> &, const TF *, const std::string & );
template bool ecrit_nuage<3>( const std::string &, const Nuage<3> &, const TF *, const std::string & );
template std::vector<Nuage<2>> suite<2>( SI, unsigned, double, const std::string & );
template std::vector<Nuage<3>> suite<3>( SI, unsigned, double, const std::string & );

} // namespace sf
