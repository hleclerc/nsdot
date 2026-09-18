#pragma once

// =====================================================================================
// LES TRAMES : un diagramme par ligne JSON, pour animer un chemin de resolution ( `scripts/
// animation.py` assemble le HTML ). Par cellule : le polygone, l'aire, et si son voisinage a
// change depuis la trame precedente.
// =====================================================================================

#include "diagram/PowerDiagram.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace sf {

struct Trames {
    std::FILE *f = nullptr;
    std::vector<std::vector<int>> vois_prec;

    bool ouvre( const std::string &chemin ) { f = std::fopen( chemin.c_str(), "w" ); return f != nullptr; }
    ~Trames() { if ( f ) std::fclose( f ); }

    /// `pd` doit porter les poids de la trame ; `nu` la cible par cellule
    template<class PD>
    void ecrit( const PD &pd, const std::vector<TF> &nu, const Parallel &par, const std::string &etiquette, double s, int it, double t, int reculs ) {
        if constexpr ( PD::dim != 2 ) { return; } else {
        if ( ! f ) return;
        const SI n = pd.n;
        std::vector<typename PD::Cell> cels( n );
        parallel_for( n, par, [ & ]( SI k, int ) { pd.cellule( k, cels[ pd.ids[ k ] ] ); } );
        std::vector<std::vector<int>> vois( n );
        std::vector<TF> aire( n );
        for ( SI i = 0; i < n; ++i ) {
            aire[ i ] = PD::mesure( cels[ i ] );
            for ( int j = 0; j < cels[ i ].nb; ++j ) vois[ i ].push_back( cels[ i ].cid[ j ] );
            std::sort( vois[ i ].begin(), vois[ i ].end() );
        }
        TF pire = 0, r2 = 0;
        SI chang = 0;
        for ( SI i = 0; i < n; ++i ) {
            pire = std::max( pire, std::fabs( aire[ i ] - nu[ i ] ) / nu[ i ] );
            r2 += ( aire[ i ] - nu[ i ] ) * ( aire[ i ] - nu[ i ] );
            chang += ! vois_prec.empty() && vois[ i ] != vois_prec[ i ];
        }
        std::fprintf( f, "{\"etiquette\":\"%s\",\"s\":%.6g,\"it\":%d,\"t\":%.4g,\"reculs\":%d,\"pire\":%.4g,\"r2\":%.4g,\"changes\":%d,\"aires\":[",
                      etiquette.c_str(), s, it, t, reculs, double( pire ), double( std::sqrt( r2 ) ), int( chang ) );
        for ( SI i = 0; i < n; ++i ) std::fprintf( f, "%s%.4g", i ? "," : "", double( aire[ i ] / nu[ i ] ) );
        std::fprintf( f, "],\"change\":[" );
        for ( SI i = 0; i < n; ++i ) std::fprintf( f, "%s%d", i ? "," : "", int( ! vois_prec.empty() && vois[ i ] != vois_prec[ i ] ) );
        std::fprintf( f, "],\"cellules\":[" );
        for ( SI i = 0; i < n; ++i ) {
            std::fprintf( f, "%s[", i ? "," : "" );
            for ( int j = 0; j < cels[ i ].nb; ++j )
                std::fprintf( f, "%s%.4f,%.4f", j ? "," : "", double( cels[ i ].vx[ j ] ), double( cels[ i ].vy[ j ] ) );
            std::fprintf( f, "]" );
        }
        std::fprintf( f, "]}\n" );
        vois_prec.swap( vois );
        }
    }
};

} // namespace sf
