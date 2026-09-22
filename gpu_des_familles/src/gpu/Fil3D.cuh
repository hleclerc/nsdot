#pragma once

// =====================================================================================
// UNE CELLULE PAR THREAD, 3D. Le polyedre de `cell/Cellule3D.h` porte par ses sommets -- trois
// coupes et trois voisins chacun, le voisin `j` EN FACE de la coupe `j` -- et sa coupe purement
// combinatoire, mot pour mot, la premiere passe asimd remplacee par une boucle. Le volume par
// accumulation sans parcours de cycle, comme au CPU.
//
// CE QUE LE GPU CHANGE : tout est en memoire locale, et il y en a beaucoup -- `MaxNv` sommets a
// neuf mots, la liste des coupes, et les tampons de la coupe. Les tampons des sommets NEUFS sont
// bornes par `MaxNm` et non `MaxNv` : une coupe cree autant de sommets qu'elle traverse d'aretes,
// une douzaine d'ordinaire, et un depassement est compte comme un debordement, pas gere.
// =====================================================================================

#include "gpu/Arbre.cuh"

namespace sf::gpu {

enum : int { INCHANGEE = 0, COUPEE = 1, VIDE = 2, DEBORDE = 3 };

template<class TK, int MaxNv, int MaxNm>
struct Cel3 {
    int nv, nc;
    TK  vx[ MaxNv ], vy[ MaxNv ], vz[ MaxNv ];
    int vk0[ MaxNv ], vk1[ MaxNv ], vk2[ MaxNv ];        ///< les trois coupes du sommet, TRIEES
    int vn0[ MaxNv ], vn1[ MaxNv ], vn2[ MaxNv ];        ///< les trois voisins, `vnj` EN FACE de `vkj`
    int cid[ MaxNv ];                                    ///< l'identifiant GLOBAL de chaque coupe

    __device__ static void faces_de( const int k[ 3 ], int j, int &f0, int &f1 ) {
        f0 = k[ j == 0 ? 1 : 0 ];
        f1 = k[ j == 2 ? 1 : 2 ];
    }

    /// LE CUBE UNITE. Coupes `0:x=0 1:x=1 2:y=0 3:y=1 4:z=0 5:z=1`, identifiants `-1 .. -6`.
    __device__ void init_cube() {
        nc = 6;
#pragma unroll
        for ( int k = 0; k < 6; ++k ) cid[ k ] = -1 - k;
        nv = 8;
#pragma unroll
        for ( int b = 0; b < 8; ++b ) {
            const int i = b & 1, j = ( b >> 1 ) & 1, k = ( b >> 2 ) & 1;
            vx[ b ] = TK( i ); vy[ b ] = TK( j ); vz[ b ] = TK( k );
            vk0[ b ] = i; vk1[ b ] = 2 + j; vk2[ b ] = 4 + k;
            vn0[ b ] = b ^ 1; vn1[ b ] = b ^ 2; vn2[ b ] = b ^ 4;
        }
    }

    /// LA COUPE par `d . x <= off`. Rien n'est ecrit avant que les tailles finales soient connues.
    __device__ int coupe( const Plan3<TK> &p ) {
        TK s[ MaxNv ];
        int nb_out = 0;
        for ( int i = 0; i < nv; ++i ) {
            s[ i ] = p.dx * vx[ i ] + p.dy * vy[ i ] + p.dz * vz[ i ] - p.off;
            nb_out += s[ i ] > 0;
        }
        if ( nb_out == 0 )
            return INCHANGEE;
        if ( nb_out == nv ) { nv = 0; nc = 0; return VIDE; }

        if ( nc >= MaxNv ) {
            compacte();
            if ( nc >= MaxNv ) return DEBORDE;
        }
        const int knew = nc;

        // ---- les TROUS que laissent les sommets dehors et, pour chacun, ses aretes traversantes
        int trou[ MaxNm ], nt = 0;
        TK  nx[ MaxNm ], ny[ MaxNm ], nz[ MaxNm ];
        int n0[ MaxNm ], n1[ MaxNm ];                    ///< les deux coupes HERITEES, triees
        int rec_v[ MaxNm ], rec_f[ MaxNm ];              ///< le sommet DEDANS a recoller, et sa fente
        int nm = 0;
        for ( int o = 0; o < nv; ++o ) {
            if ( ! ( s[ o ] > 0 ) ) continue;
            if ( nt >= MaxNm ) return DEBORDE;
            trou[ nt++ ] = o;
            const int k[ 3 ] = { vk0[ o ], vk1[ o ], vk2[ o ] };
            const int w[ 3 ] = { vn0[ o ], vn1[ o ], vn2[ o ] };
#pragma unroll
            for ( int j = 0; j < 3; ++j ) {
                const int u = w[ j ];
                if ( s[ u ] > 0 )
                    continue;                            // arete entierement dehors : elle meurt
                if ( nm >= MaxNm )
                    return DEBORDE;
                const TK t = s[ u ] / ( s[ u ] - s[ o ] );   // ANCRE SUR LE SOMMET DEDANS
                nx[ nm ] = vx[ u ] + ( vx[ o ] - vx[ u ] ) * t;
                ny[ nm ] = vy[ u ] + ( vy[ o ] - vy[ u ] ) * t;
                nz[ nm ] = vz[ u ] + ( vz[ o ] - vz[ u ] ) * t;
                faces_de( k, j, n0[ nm ], n1[ nm ] );
                rec_v[ nm ] = u;
                rec_f[ nm ] = vn0[ u ] == o ? 0 : ( vn1[ u ] == o ? 1 : 2 );
                ++nm;
            }
        }

        const int nn = nv - nt;
        const int new_nv = nn + nm;
        if ( new_nv > MaxNv )
            return DEBORDE;

        // OU VA CHAQUE SOMMET NEUF : dans un trou tant qu'il en reste, puis a la suite
        int dest[ MaxNm ];
        for ( int j = 0; j < nm; ++j ) dest[ j ] = j < nt ? trou[ j ] : nv + ( j - nt );

        // ---- LES VOISINS DES SOMMETS NEUFS : deux neufs sont voisins exactement quand ils
        // partagent une ANCIENNE coupe ; en face de `knew`, le bout dedans dont chacun vient
        int m0[ MaxNm ], m1[ MaxNm ];
        for ( int i = 0; i < nm; ++i ) { m0[ i ] = -1; m1[ i ] = -1; }
        for ( int i = 0; i < nm; ++i )
            for ( int j = i + 1; j < nm; ++j ) {
                int kc;
                if      ( n0[ i ] == n0[ j ] || n0[ i ] == n1[ j ] ) kc = n0[ i ];
                else if ( n1[ i ] == n0[ j ] || n1[ i ] == n1[ j ] ) kc = n1[ i ];
                else continue;
                if ( kc == n0[ i ] ) m1[ i ] = dest[ j ]; else m0[ i ] = dest[ j ];
                if ( kc == n0[ j ] ) m1[ j ] = dest[ i ]; else m0[ j ] = dest[ i ];
            }

        // ---- COMMIT
        for ( int j = 0; j < nm; ++j ) {
            const int m = dest[ j ];
            vx[ m ] = nx[ j ]; vy[ m ] = ny[ j ]; vz[ m ] = nz[ j ];
            vk0[ m ] = n0[ j ]; vk1[ m ] = n1[ j ]; vk2[ m ] = knew;
            vn0[ m ] = m0[ j ]; vn1[ m ] = m1[ j ]; vn2[ m ] = rec_v[ j ];
        }
        for ( int i = 0; i < nm; ++i ) {                 // le recollage, cote sommet DEDANS
            const int u = rec_v[ i ];
            if      ( rec_f[ i ] == 0 ) vn0[ u ] = dest[ i ];
            else if ( rec_f[ i ] == 1 ) vn1[ u ] = dest[ i ];
            else                        vn2[ u ] = dest[ i ];
        }

        // ---- LES TROUS QUI RESTENT, quand la coupe enleve plus de sommets qu'elle n'en cree
        if ( nm < nt ) {
            int th = nt;
            while ( th > nm && trou[ th - 1 ] >= new_nv ) --th;
            int nouv[ MaxNm ], src[ MaxNm ], dst[ MaxNm ], nmv = 0;
            int ct = th, cd = nm;
            for ( int i = new_nv; i < nv; ++i ) {
                if ( ct < nt && trou[ ct ] == i ) { ++ct; continue; }
                src[ nmv ] = i;
                dst[ nmv ] = trou[ cd++ ];
                nouv[ i - new_nv ] = dst[ nmv ];
                ++nmv;
            }
            for ( int t = 0; t < nmv; ++t ) {
                const int a = src[ t ], b = dst[ t ];
                vx[ b ] = vx[ a ]; vy[ b ] = vy[ a ]; vz[ b ] = vz[ a ];
                vk0[ b ] = vk0[ a ]; vk1[ b ] = vk1[ a ]; vk2[ b ] = vk2[ a ];
                vn0[ b ] = vn0[ a ]; vn1[ b ] = vn1[ a ]; vn2[ b ] = vn2[ a ];
            }
            for ( int t = 0; t < nmv; ++t ) {
                const int a = src[ t ], b = dst[ t ];
                const int w[ 3 ] = { vn0[ b ], vn1[ b ], vn2[ b ] };
#pragma unroll
                for ( int j = 0; j < 3; ++j ) {
                    const int q = w[ j ] >= new_nv ? nouv[ w[ j ] - new_nv ] : w[ j ];
                    if      ( vn0[ q ] == a ) vn0[ q ] = b;
                    else if ( vn1[ q ] == a ) vn1[ q ] = b;
                    else                      vn2[ q ] = b;
                }
            }
        }

        cid[ knew ] = p.id;
        nc = knew + 1;
        nv = new_nv;
        return COUPEE;
    }

    /// ENLEVER LES COUPES MORTES. La renumerotation est MONOTONE, donc les triplets restent tries.
    __device__ void compacte() {
        int m[ MaxNv ];
        for ( int k = 0; k < nc; ++k ) m[ k ] = -1;
        for ( int i = 0; i < nv; ++i ) { m[ vk0[ i ] ] = 0; m[ vk1[ i ] ] = 0; m[ vk2[ i ] ] = 0; }
        int q = 0;
        for ( int k = 0; k < nc; ++k )
            if ( m[ k ] == 0 ) { cid[ q ] = cid[ k ]; m[ k ] = q++; }
        for ( int i = 0; i < nv; ++i ) { vk0[ i ] = m[ vk0[ i ] ]; vk1[ i ] = m[ vk1[ i ] ]; vk2[ i ] = m[ vk2[ i ] ]; }
        nc = q;
    }

    /// LE VOLUME : par face un sommet `v_f` et la somme des produits vectoriels de ses aretes vues
    /// depuis lui, `sum_f | ( v_f - g ) . S_f | / 6`.
    __device__ double volume() const {
        if ( nv < 4 ) return 0;
        int v0[ MaxNv ];
        double sx[ MaxNv ], sy[ MaxNv ], sz[ MaxNv ];
        for ( int k = 0; k < nc; ++k ) { v0[ k ] = -1; sx[ k ] = sy[ k ] = sz[ k ] = 0; }
        for ( int i = nv - 1; i >= 0; --i ) { v0[ vk0[ i ] ] = i; v0[ vk1[ i ] ] = i; v0[ vk2[ i ] ] = i; }

        double gx = 0, gy = 0, gz = 0;
        for ( int i = 0; i < nv; ++i ) { gx += vx[ i ]; gy += vy[ i ]; gz += vz[ i ]; }
        gx /= nv; gy /= nv; gz /= nv;

        for ( int a = 0; a < nv; ++a ) {
            const int k[ 3 ] = { vk0[ a ], vk1[ a ], vk2[ a ] };
            const int w[ 3 ] = { vn0[ a ], vn1[ a ], vn2[ a ] };
#pragma unroll
            for ( int j = 0; j < 3; ++j ) {
                const int b = w[ j ];
                if ( b <= a ) continue;
                int f0, f1;
                faces_de( k, j, f0, f1 );
#pragma unroll
                for ( int r = 0; r < 2; ++r ) {
                    const int f = r ? f1 : f0;
                    const int o = v0[ f ];
                    const double ax = vx[ a ] - vx[ o ], ay = vy[ a ] - vy[ o ], az = vz[ a ] - vz[ o ];
                    const double bx = vx[ b ] - vx[ o ], by = vy[ b ] - vy[ o ], bz = vz[ b ] - vz[ o ];
                    double cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
                    if ( sx[ f ] * cx + sy[ f ] * cy + sz[ f ] * cz < 0 ) { cx = -cx; cy = -cy; cz = -cz; }
                    sx[ f ] += cx; sy[ f ] += cy; sz[ f ] += cz;
                }
            }
        }
        double v = 0;
        for ( int k = 0; k < nc; ++k ) {
            const int o = v0[ k ];
            if ( o < 0 ) continue;
            v += fabs( ( vx[ o ] - gx ) * sx[ k ] + ( vy[ o ] - gy ) * sy[ k ] + ( vz[ o ] - gz ) * sz[ k ] );
        }
        return v / 6;
    }
};

/// « un germe de cette boite peut-il encore couper ? » -- la boucle sur les sommets
template<bool POIDS, class TK, class Cel>
__device__ __forceinline__ bool peut_couper3( const Noeud<TK,3> &B, const TK *p0, TK w0, const Cel &c ) {
    for ( int i = 0; i < c.nv; ++i ) {
        const TK v[ 3 ] = { c.vx[ i ], c.vy[ i ], c.vz[ i ] };
        if ( bilan_sommet<POIDS>( B, v, p0, w0 ) <= 0 ) return true;
    }
    return false;
}

template<bool POIDS, int MaxNv, int MaxNm, class TK>
__global__ void noyau3_fil( Arbre<TK,3> ar, double *res, int *deborde ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= ar.n ) return;
    const TK p0[ 3 ] = { ar.c[ 0 ][ k ], ar.c[ 1 ][ k ], ar.c[ 2 ][ k ] };
    const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
    const int i0 = ar.ids[ k ];

    Cel3<TK,MaxNv,MaxNm> c;
    c.init_cube();
    int pile[ PILE ];
    int haut = 0;
    pile[ haut++ ] = 0;
    int r = 0;
    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,3> nd = ar.nodes[ h ];
        if ( ! peut_couper3<POIDS>( nd, p0, w0, c ) )
            continue;
        if ( nd.right < 0 ) {
            for ( int q = nd.beg; q < nd.end; ++q ) {
                if ( ar.ids[ q ] == i0 ) continue;
                const Plan3<TK> p = bissect3<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], p0[ 2 ], w0 );
                r = c.coupe( p );
                if ( r == VIDE || r == DEBORDE ) { haut = 0; break; }
            }
            continue;
        }
        const int g = h + 1, dr = nd.right;
        const bool gp = proximite( ar.nodes[ g ], p0 ) <= proximite( ar.nodes[ dr ], p0 );
        pile[ haut++ ] = gp ? dr : g;
        pile[ haut++ ] = gp ? g : dr;
    }
    if ( r == DEBORDE ) atomicAdd( deborde, 1 );
    res[ i0 ] = r == DEBORDE ? 0.0 : c.volume();
}

} // namespace sf::gpu
