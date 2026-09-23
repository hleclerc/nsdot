#pragma once

// =====================================================================================
// UNE CELLULE PAR THREAD, 2D. La « boucle nue en memoire » : les sommets en ordre cyclique dans
// des tableaux locaux du thread, la coupe scalaire en place de `cell/Contrat2D.h` ( `coupe_large` )
// et le parcours de l'arbre avec sa pile, egalement locale. Rien de cooperatif : chaque thread
// est un coeur scalaire qui fait SA cellule du debut a la fin.
//
// CE QUI EN DECOULE. Les tableaux indexes dynamiquement vont en MEMOIRE LOCALE ( du global cache
// en L1 / L2, entrelace par thread pour que le sommet `i` de trente-deux threads voisins tienne en
// une ligne ). Un warp execute l'UNION des chemins de ses trente-deux cellules : deux voisines
// dans l'arbre suivent a peu pres le meme parcours, ce qui borne la divergence -- mais pas les
// coupes, dont le nombre et la forme varient d'une cellule a l'autre.
//
// Le test d'elagage est celui du CPU en excursion : une boucle sur les sommets, sortie au
// premier qui peut etre coupe.
// =====================================================================================

#include "gpu/Arbre.cuh"

namespace sf::gpu {

/// LA COUPE SCALAIRE, EN PLACE -- `coupe_large` de `Contrat2D.h`, mot pour mot. Rend le nouveau
/// nombre de sommets : `nb` inchange si le plan ne coupe rien, `0` si la cellule est vide, `-1`
/// si la sortie ne tient pas dans `MaxNb` ( la cellule reste alors INTACTE ).
template<int MaxNb, class TK>
__device__ __forceinline__ int coupe2( TK *vx, TK *vy, int *cid, int nb, const Plan2<TK> &p, TK *s ) {
    int nb_out = 0;
    for ( int i = 0; i < nb; ++i ) {
        s[ i ] = p.dx * vx[ i ] + p.dy * vy[ i ] - p.off;
        nb_out += s[ i ] > 0;
    }
    if ( nb_out == 0 )
        return nb;
    if ( nb_out == nb )
        return 0;

    int i1 = 0;                                          // l'exterieur d'un convexe coupe est d'un
    for ( int i = 0; i < nb; ++i ) {                     // seul tenant : `i1` est unique
        const int q = i ? i - 1 : nb - 1;
        if ( s[ i ] > 0 && ! ( s[ q ] > 0 ) ) { i1 = i; break; }
    }

    const int nb_in = nb - nb_out;
    const int new_nb = nb_in + 2;
    if ( new_nb > MaxNb )
        return -1;

    const int j0 = ( i1 + nb - 1 ) % nb;                 // dernier DEDANS avant la plage
    const int j2 = ( i1 + nb_out - 1 ) % nb;             // dernier DEHORS
    const int j3 = ( j2 + 1 ) % nb;                      // premier DEDANS apres

    // les deux intersections, ANCREES sur le sommet dedans
    const TK s0 = s[ j0 ], s1 = s[ i1 ], s2 = s[ j2 ], s3 = s[ j3 ];
    const TK ta  = s0 / ( s0 - s1 );
    const TK pax = vx[ j0 ] + ( vx[ i1 ] - vx[ j0 ] ) * ta;
    const TK pay = vy[ j0 ] + ( vy[ i1 ] - vy[ j0 ] ) * ta;
    const TK tb  = s3 / ( s3 - s2 );
    const TK pbx = vx[ j3 ] + ( vx[ j2 ] - vx[ j3 ] ) * tb;
    const TK pby = vy[ j3 ] + ( vy[ j2 ] - vy[ j3 ] ) * tb;
    const int bid = cid[ j2 ];                           // lu MAINTENANT : `j2` va etre ecrase

    auto move = [ & ]( int d, int t ) { vx[ d ] = vx[ t ]; vy[ d ] = vy[ t ]; cid[ d ] = cid[ t ]; };

    if ( i1 <= j2 ) {
        if ( nb_out == 1 ) {                             // un cran de plus : la queue va A DROITE
            for ( int i = nb; i > i1 + 1; --i ) move( i, i - 1 );
        } else if ( nb_out > 2 ) {                       // trop de place : la queue revient A GAUCHE
            const int gap = nb_out - 2;
            for ( int i = j2 + 1; i < nb; ++i ) move( i - gap, i );
        }
        vx[ i1 ] = pax; vy[ i1 ] = pay; cid[ i1 ] = p.id;
        vx[ i1 + 1 ] = pbx; vy[ i1 + 1 ] = pby; cid[ i1 + 1 ] = bid;
    } else {
        // la plage BOUCLE, donc l'interieur est contigu : `[ j3, j3 + nb_in )`.
        if ( j3 >= 2 ) for ( int o = 0; o < nb_in; ++o ) move( 2 + o, j3 + o );
        else           for ( int o = nb_in - 1; o >= 0; --o ) move( 2 + o, j3 + o );
        vx[ 0 ] = pax; vy[ 0 ] = pay; cid[ 0 ] = p.id;
        vx[ 1 ] = pbx; vy[ 1 ] = pby; cid[ 1 ] = bid;
    }
    return new_nb;
}

/// « un germe de cette boite peut-il encore couper ? » -- sur les sommets tels qu'ils sont.
template<bool POIDS, class TK>
__device__ __forceinline__ bool peut_couper2( const Noeud<TK,2> &B, const TK *p0, TK w0, int nb, const TK *vx, const TK *vy ) {
    for ( int i = 0; i < nb; ++i ) {
        const TK v[ 2 ] = { vx[ i ], vy[ i ] };
        if ( bilan_sommet<POIDS>( B, v, p0, w0 ) <= 0 ) return true;
    }
    return false;
}

/// LA CELLULE DU THREAD : depuis le carre unite ( cotes `-1 .. -4` ), le parcours, les coupes.
/// Rend le nombre de sommets, `0` si vide, `-1` si `MaxNb` n'a pas suffi. `area` : l'aire.
template<bool POIDS, int MaxNb, class TK>
__device__ int cellule2_fil( const Arbre<TK,2> &ar, int k, TK *vx, TK *vy, int *cid, double &area ) {
    const TK p0[ 2 ] = { ar.c[ 0 ][ k ], ar.c[ 1 ][ k ] };
    const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
    const int i0 = ar.ids[ k ];

    vx[ 0 ] = 0; vy[ 0 ] = 0; cid[ 0 ] = -1;
    vx[ 1 ] = 1; vy[ 1 ] = 0; cid[ 1 ] = -2;
    vx[ 2 ] = 1; vy[ 2 ] = 1; cid[ 2 ] = -3;
    vx[ 3 ] = 0; vy[ 3 ] = 1; cid[ 3 ] = -4;
    int nb = 4;

    TK  s[ MaxNb ];
    int pile[ PILE ];
    int haut = 0;
    pile[ haut++ ] = 0;
    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,2> nd = ar.nodes[ h ];
        if ( ! peut_couper2<POIDS>( nd, p0, w0, nb, vx, vy ) )
            continue;
        if ( nd.right < 0 ) {                            // une feuille : ses germes, dans l'ordre
            for ( int q = nd.beg; q < nd.end; ++q ) {
                if ( ar.ids[ q ] == i0 ) continue;
                const Plan2<TK> p = bissect2<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], w0 );
                nb = coupe2<MaxNb>( vx, vy, cid, nb, p, s );
                if ( nb <= 0 ) { area = 0; return nb; }
            }
            continue;
        }
        const int g = h + 1, dr = nd.right;              // PREORDRE : le gauche est juste a cote
        const bool gauche_pres = proximite( ar.nodes[ g ], p0 ) <= proximite( ar.nodes[ dr ], p0 );
        pile[ haut++ ] = gauche_pres ? dr : g;           // le plus proche est depile en premier
        pile[ haut++ ] = gauche_pres ? g : dr;
    }

    double a = 0;                                        // le lacet, en double depuis `TK`
    for ( int i = 0; i < nb; ++i ) {
        const int j = i + 1 < nb ? i + 1 : 0;
        a += double( vx[ i ] ) * double( vy[ j ] ) - double( vx[ j ] ) * double( vy[ i ] );
    }
    area = 0.5 * fabs( a );
    return nb;
}

template<bool POIDS, int MaxNb, class TK>
__global__ void noyau2_fil( Arbre<TK,2> ar, double *res, int *deborde ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= ar.n ) return;
    TK  vx[ MaxNb ], vy[ MaxNb ];
    int cid[ MaxNb ];
    double area;
    const int nb = cellule2_fil<POIDS,MaxNb>( ar, k, vx, vy, cid, area );
    if ( nb < 0 ) atomicAdd( deborde, 1 );
    res[ ar.ids[ k ] ] = area;
}

} // namespace sf::gpu
