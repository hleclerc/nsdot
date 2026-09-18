#pragma once

// =====================================================================================
// UNE DIRECTION SAUVEGARDEE : un nuage, des poids `w`, et une direction `d` proposee par Newton
// depuis ces poids. C'est le materiau du banc `ecrasement` -- des ensembles de diracs qui font
// beaucoup reculer l'amortissement, gardes tels quels pour qu'on puisse y revenir.
//
// Format : des lignes `#`, puis `n`, puis `n` fois `x y [z] w d`, en `%.17g`.
// =====================================================================================

#include "bench/Nuages.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace sf {

template<int D>
struct Direction {
    Nuage<D>        nuage;      ///< les positions ( `nuage.w` porte `w`, voir `finish` )
    std::vector<TF> w;          ///< les poids au depart du pas
    std::vector<TF> d;          ///< la direction proposee ( `d[ 0 ] == 0` )
    std::string     entete;     ///< les lignes `#` du fichier, pour savoir d'ou ca vient

    SI n() const { return nuage.n; }
};

template<int D>
bool ecrit_direction( const std::string &chemin, const Direction<D> &dir ) {
    std::FILE *f = std::fopen( chemin.c_str(), "w" );
    if ( ! f )
        return false;
    std::fprintf( f, "%s", dir.entete.c_str() );
    std::fprintf( f, "%d\n", int( dir.n() ) );
    for ( SI i = 0; i < dir.n(); ++i ) {
        for ( int k = 0; k < D; ++k )
            std::fprintf( f, "%.17g ", double( dir.nuage.P[ k ][ i ] ) );
        std::fprintf( f, "%.17g %.17g\n", double( dir.w[ i ] ), double( dir.d[ i ] ) );
    }
    std::fclose( f );
    return true;
}

template<int D>
bool charge_direction( const std::string &chemin, Direction<D> &dir ) {
    std::ifstream f( chemin );
    if ( ! f ) {
        std::printf( "impossible d'ouvrir '%s'\n", chemin.c_str() );
        return false;
    }
    std::string ligne;
    SI n = -1;
    dir.entete.clear();
    while ( n < 0 && std::getline( f, ligne ) ) {
        if ( ligne.empty() ) continue;
        if ( ligne[ 0 ] == '#' ) { dir.entete += ligne + "\n"; continue; }
        n = std::atoi( ligne.c_str() );
    }
    if ( n < 0 ) {
        std::printf( "'%s' : pas de nombre de germes\n", chemin.c_str() );
        return false;
    }
    dir.nuage.nom = chemin;
    for ( int k = 0; k < D; ++k ) dir.nuage.c[ k ].assign( n, TF( 0 ) );
    dir.w.assign( n, TF( 0 ) );
    dir.d.assign( n, TF( 0 ) );
    for ( SI i = 0; i < n; ++i ) {
        bool ok = bool( std::getline( f, ligne ) );
        std::istringstream ss( ligne );
        for ( int k = 0; k < D && ok; ++k ) ok = bool( ss >> dir.nuage.c[ k ][ i ] );
        if ( ok ) ok = bool( ss >> dir.w[ i ] >> dir.d[ i ] );
        if ( ! ok ) {
            std::printf( "'%s' : tronque au germe %d\n", chemin.c_str(), int( i ) );
            return false;
        }
    }
    dir.nuage.w = dir.w;
    dir.nuage.finish();
    return true;
}

} // namespace sf
