#pragma once

// =====================================================================================
// LA CELLULE 3D : un polyedre convexe qui se coupe lui-meme.
//
// LA CONNECTIVITE EST PORTEE PAR LES SOMMETS : chacun nomme ses trois coupes ( `vk`, indices dans
// `cid`, tries croissant ) et ses trois voisins ( `vn` ). L'INVARIANT : `vn[ j ]` est le voisin DE
// L'AUTRE COTE de l'arete portee par les deux autres coupes que `vk[ j ]` -- le voisin `j` est
// « en face » de la coupe `j`. Les deux faces qui portent l'arete `j` sont donc `vk` prive de
// `vk[ j ]`, sans rien chercher.
//
// LA COUPE part des sommets DEHORS ( `3 nb_out` predicats, une douzaine ) et non des aretes, et LES
// SOMMETS GARDES NE BOUGENT PAS : les neufs remplissent les trous laisses par les sommets dehors,
// l'adjacence des survivants reste valable telle quelle. Le commit est en `O( nm )`, pas `O( nv )`.
// Le seul cas ou un sommet garde bouge est une coupe qui enleve plus de sommets qu'elle n'en
// cree -- un cycle parmi les sommets coupes -- et alors `nt - nm` sommets demenagent.
//
// LA PREMIERE PASSE ( `s = d . v - off` sur tous les sommets ) est en asimd par blocs de `W`
// voies, chargements pleins puis UNE queue partielle ; le compte des sommets dehors est un
// `popcount` de masque. Huit voies est la largeur mesuree optimale ( `2d_des_familles`, en-tete de
// `Cellule3D.h` : le detail et les variantes rejetees ).
//
// LE VOLUME ET LES AIRES DE FACES s'accumulent SANS parcourir de cycle : par face, un sommet `v_f`
// et la somme `S_f` des produits vectoriels de ses aretes vues depuis lui. L'aire vaut `|S_f| / 2`,
// le volume `sum_f |( v_f - g ) . S_f| / 6`. Deux passes en `O( V + E )`.
// =====================================================================================

#include "cell/Contrat3D.h"
#include <asimd/asimd.h>
#include <cmath>

namespace sf::d3 {

/// `MaxNv` borne les sommets et les coupes, `W` est la largeur SIMD de la premiere passe.
template<class TK, int MaxNv, int W = 8>
struct Cellule3 {
    using V  = asimd::SimdVec<TK,W>;
    using TKernel = TK;

    static constexpr int max_nv = MaxNv;
    static constexpr int max_nc = MaxNv;
    static_assert( MaxNv % W == 0, "les sommets se lisent par groupes de W" );
    static_assert( ( W & ( W - 1 ) ) == 0, "la largeur SIMD est une puissance de deux" );

    int nv = 0, nc = 0;

    alignas( 64 ) TK vx[ MaxNv ];
    alignas( 64 ) TK vy[ MaxNv ];
    alignas( 64 ) TK vz[ MaxNv ];

    /// les trois coupes du sommet, en indices dans `cid`, TRIEES CROISSANT
    alignas( 64 ) int vk0[ MaxNv ];
    alignas( 64 ) int vk1[ MaxNv ];
    alignas( 64 ) int vk2[ MaxNv ];

    /// les trois voisins du sommet. `vnj` est EN FACE de `vkj` -- voir l'en-tete.
    alignas( 64 ) int vn0[ MaxNv ];
    alignas( 64 ) int vn1[ MaxNv ];
    alignas( 64 ) int vn2[ MaxNv ];

    alignas( 64 ) SI32 cid[ max_nc ];                    ///< l'identifiant GLOBAL de chaque coupe

    /// LES DEUX FACES QUI PORTENT L'ARETE `j` DU SOMMET : ses coupes, privees de la `j`-ieme.
    static void faces_de( const int k[ 3 ], int j, int &f0, int &f1 ) {
        f0 = k[ j == 0 ? 1 : 0 ];
        f1 = k[ j == 2 ? 1 : 2 ];
    }

    /// LE CUBE UNITE. Coupes `0:x=0 1:x=1 2:y=0 3:y=1 4:z=0 5:z=1`, identifiants `-1 .. -6`.
    void init_cube() {
        nc = 6;
        for ( int k = 0; k < 6; ++k ) cid[ k ] = -1 - k;
        nv = 8;
        for ( int b = 0; b < 8; ++b ) {
            const int i = b & 1, j = ( b >> 1 ) & 1, k = ( b >> 2 ) & 1;
            vx[ b ] = TK( i ); vy[ b ] = TK( j ); vz[ b ] = TK( k );
            vk0[ b ] = i;
            vk1[ b ] = 2 + j;
            vk2[ b ] = 4 + k;                            // deja croissant
            vn0[ b ] = b ^ 1;                            // en face de la coupe en x : l'arete qui
            vn1[ b ] = b ^ 2;                            // court le long de x, et ainsi de suite
            vn2[ b ] = b ^ 4;
        }
    }

    EtatCell3<TK> etat() const { return { nv, vx, vy, vz }; }

    /// LA COUPE par `p.dx x + p.dy y + p.dz z <= p.off`. Rien n'est ecrit avant que les tailles
    /// finales soient connues : sur debordement la cellule reste INTACTE.
    int coupe( const Plan3<TK> &p ) {
        alignas( 64 ) TK s[ MaxNv ];

        // ---- LA PREMIERE PASSE, qui COMPTE les sommets dehors.
        int nb_out = 0;
        {
            const V dx( p.dx ), dy( p.dy ), dz( p.dz ), mof( -p.off ), zero( TK( 0 ) );
            const int plein = nv & ~( W - 1 );
            int i = 0;
            for ( ; i < plein; i += W ) {
                const V sv = asimd::fma( dx, V::load_aligned( vx + i ),
                             asimd::fma( dy, V::load_aligned( vy + i ),
                             asimd::fma( dz, V::load_aligned( vz + i ), mof ) ) );
                sv.store_aligned( s + i );
                nb_out += __builtin_popcountll( asimd::to_bits( sv > zero ) );
            }
            if ( i < nv ) {
                const auto q = asimd::LaneRange<0>( nv - i );
                const V sv = asimd::fma( dx, V::load_partial( vx + i, q ),
                             asimd::fma( dy, V::load_partial( vy + i, q ),
                             asimd::fma( dz, V::load_partial( vz + i, q ), mof ) ) );
                sv.store_partial( s + i, q );
                nb_out += __builtin_popcountll( asimd::to_bits( sv > zero, q ) );
            }
        }

        if ( nb_out == 0 )                               // LE CAS FREQUENT, ET IL EST LE PREMIER
            return INCHANGEE;
        if ( nb_out == nv ) { nv = 0; nc = 0; return VIDE; }

        if ( nc >= max_nc ) {
            compacte();
            if ( nc >= max_nc ) return DEBORDE;
        }
        const int knew = nc;

        // ---- UNE SEULE PASSE SUR LES SOMMETS : les TROUS que laissent les sommets dehors et, pour
        // chacun d'eux, ses aretes traversantes.
        int   trou[ MaxNv ], nt = 0;
        TK    nx[ MaxNv ], ny[ MaxNv ], nz[ MaxNv ];
        int   n0[ MaxNv ], n1[ MaxNv ];                  ///< les deux coupes HERITEES, triees
        int   rec_v[ MaxNv ], rec_f[ MaxNv ];            ///< le sommet DEDANS a recoller, et sa fente
        int   nm = 0;

        for ( int o = 0; o < nv; ++o ) {
            if ( ! ( s[ o ] > 0 ) ) continue;
            trou[ nt++ ] = o;

            const int k[ 3 ] = { vk0[ o ], vk1[ o ], vk2[ o ] };
            const int w[ 3 ] = { vn0[ o ], vn1[ o ], vn2[ o ] };
            for ( int j = 0; j < 3; ++j ) {
                const int u = w[ j ];
                if ( s[ u ] > 0 )
                    continue;                            // arete entierement dehors : elle meurt
                if ( nm >= MaxNv )
                    return DEBORDE;

                // ANCRE SUR LE SOMMET DEDANS : avec `s_u == 0` la forme symetrique ne rend pas
                // `v_u` en flottant, et le sommet passerait de l'autre cote du plan.
                const TK t = s[ u ] / ( s[ u ] - s[ o ] );
                nx[ nm ] = vx[ u ] + ( vx[ o ] - vx[ u ] ) * t;
                ny[ nm ] = vy[ u ] + ( vy[ o ] - vy[ u ] ) * t;
                nz[ nm ] = vz[ u ] + ( vz[ o ] - vz[ u ] ) * t;
                faces_de( k, j, n0[ nm ], n1[ nm ] );

                rec_v[ nm ] = u;                         // la fente de `u` qui pointait vers `o`
                rec_f[ nm ] = vn0[ u ] == o ? 0 : ( vn1[ u ] == o ? 1 : 2 );
                ++nm;
            }
        }

        const int nn = nv - nt;
        const int new_nv = nn + nm;
        if ( new_nv > MaxNv )
            return DEBORDE;

        // OU VA CHAQUE SOMMET NEUF : dans un trou tant qu'il en reste, puis a la suite.
        int dest[ MaxNv ];
        for ( int j = 0; j < nm; ++j ) dest[ j ] = j < nt ? trou[ j ] : nv + ( j - nt );

        // ---- LES VOISINS DES SOMMETS NEUFS. Le sommet neuf porte `( n0, n1, knew )` : en face de
        // `knew` il y a le bout DEDANS dont il vient. Les deux autres sont ses voisins sur la face
        // neuve : deux sommets neufs sont voisins exactement quand ils partagent une ANCIENNE coupe.
        int m0[ MaxNv ], m1[ MaxNv ], m2[ MaxNv ];
        for ( int i = 0; i < nm; ++i ) {
            m2[ i ] = rec_v[ i ];
            m0[ i ] = m1[ i ] = -1;
        }
        for ( int i = 0; i < nm; ++i )
            for ( int j = i + 1; j < nm; ++j ) {
                int kc;
                if      ( n0[ i ] == n0[ j ] || n0[ i ] == n1[ j ] ) kc = n0[ i ];
                else if ( n1[ i ] == n0[ j ] || n1[ i ] == n1[ j ] ) kc = n1[ i ];
                else continue;
                if ( kc == n0[ i ] ) m1[ i ] = dest[ j ]; else m0[ i ] = dest[ j ];
                if ( kc == n0[ j ] ) m1[ j ] = dest[ i ]; else m0[ j ] = dest[ i ];
            }

        // ---- COMMIT. Rien n'a bouge jusqu'ici.
        for ( int j = 0; j < nm; ++j ) {                 // les sommets NEUFS, dans les trous
            const int m = dest[ j ];
            vx[ m ] = nx[ j ]; vy[ m ] = ny[ j ]; vz[ m ] = nz[ j ];
            vk0[ m ] = n0[ j ]; vk1[ m ] = n1[ j ]; vk2[ m ] = knew;   // la neuve en dernier : trie
            vn0[ m ] = m0[ j ]; vn1[ m ] = m1[ j ]; vn2[ m ] = m2[ j ];
        }
        for ( int i = 0; i < nm; ++i ) {                 // le recollage, cote sommet DEDANS
            const int u = rec_v[ i ];
            switch ( rec_f[ i ] ) {
                case 0:  vn0[ u ] = dest[ i ]; break;
                case 1:  vn1[ u ] = dest[ i ]; break;
                default: vn2[ u ] = dest[ i ]; break;
            }
        }

        // ---- LES TROUS QUI RESTENT, quand la coupe enleve plus de sommets qu'elle n'en cree.
        if ( nm < nt ) {
            int th = nt;                                 // les trous deja au-dela de la fin
            while ( th > nm && trou[ th - 1 ] >= new_nv ) --th;

            int nouv[ MaxNv ], src[ MaxNv ], dst[ MaxNv ], nmv = 0;
            int ct = th, cd = nm;
            for ( int i = new_nv; i < nv; ++i ) {
                if ( ct < nt && trou[ ct ] == i ) { ++ct; continue; }   // ce slot EST un trou
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
            // et les voisins qui pointaient vers eux ; un voisin peut lui-meme avoir demenage
            for ( int t = 0; t < nmv; ++t ) {
                const int a = src[ t ], b = dst[ t ];
                const int w[ 3 ] = { vn0[ b ], vn1[ b ], vn2[ b ] };
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
    void compacte() {
        int m[ max_nc ];
        for ( int k = 0; k < nc; ++k ) m[ k ] = -1;
        for ( int i = 0; i < nv; ++i ) { m[ vk0[ i ] ] = 0; m[ vk1[ i ] ] = 0; m[ vk2[ i ] ] = 0; }
        int q = 0;
        for ( int k = 0; k < nc; ++k )
            if ( m[ k ] == 0 ) { cid[ q ] = cid[ k ]; m[ k ] = q++; }
        for ( int i = 0; i < nv; ++i ) {
            vk0[ i ] = m[ vk0[ i ] ]; vk1[ i ] = m[ vk1[ i ] ]; vk2[ i ] = m[ vk2[ i ] ];
        }
        nc = q;
    }

    /// LE VOLUME, et les faces si on les demande : `face( id_global, aire )` pour chaque coupe
    /// encore portee par un sommet -- les faces du domaine ( `id < 0` ) comprises, a l'appelant de
    /// les filtrer.
    template<class Face>
    double volume_et_faces( Face &&face ) const {
        if ( nv < 4 ) return 0;
        int v0[ max_nc ];
        double sx[ max_nc ], sy[ max_nc ], sz[ max_nc ];
        for ( int k = 0; k < nc; ++k ) { v0[ k ] = -1; sx[ k ] = sy[ k ] = sz[ k ] = 0; }
        for ( int i = nv - 1; i >= 0; --i ) { v0[ vk0[ i ] ] = i; v0[ vk1[ i ] ] = i; v0[ vk2[ i ] ] = i; }

        double gx = 0, gy = 0, gz = 0;
        for ( int i = 0; i < nv; ++i ) { gx += vx[ i ]; gy += vy[ i ]; gz += vz[ i ]; }
        gx /= nv; gy /= nv; gz /= nv;

        // chaque arete, vue une fois, contribue aux DEUX faces qui la portent : l'eventail depuis
        // `v0[ f ]`, et l'orientation ramenee sur celle de la somme courante
        for ( int a = 0; a < nv; ++a ) {
            const int k[ 3 ] = { vk0[ a ], vk1[ a ], vk2[ a ] };
            const int w[ 3 ] = { vn0[ a ], vn1[ a ], vn2[ a ] };
            for ( int j = 0; j < 3; ++j ) {
                const int b = w[ j ];
                if ( b <= a ) continue;
                int f0, f1;
                faces_de( k, j, f0, f1 );
                for ( int r = 0; r < 2; ++r ) {
                    const int f = r ? f1 : f0;
                    const int o = v0[ f ];
                    const double ax = vx[a] - vx[o], ay = vy[a] - vy[o], az = vz[a] - vz[o];
                    const double bx = vx[b] - vx[o], by = vy[b] - vy[o], bz = vz[b] - vz[o];
                    double cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
                    if ( sx[f] * cx + sy[f] * cy + sz[f] * cz < 0 ) { cx = -cx; cy = -cy; cz = -cz; }
                    sx[f] += cx; sy[f] += cy; sz[f] += cz;
                }
            }
        }

        double v = 0;
        for ( int k = 0; k < nc; ++k ) {
            const int o = v0[ k ];
            if ( o < 0 ) continue;                       // coupe morte, pas encore compactee
            v += std::abs( ( vx[o] - gx ) * sx[k] + ( vy[o] - gy ) * sy[k] + ( vz[o] - gz ) * sz[k] );
            face( cid[ k ], 0.5 * std::sqrt( sx[k] * sx[k] + sy[k] * sy[k] + sz[k] * sz[k] ) );
        }
        return v / 6;
    }

    double volume() const { return volume_et_faces( []( SI32, double ) {} ); }

    /// LE CONTROLE DE L'INVARIANT : rend le nombre de violations. Pour chaque sommet `a` et chaque
    /// fente `j`, `b = vnj[ a ]` doit avoir UNE fente qui pointe vers `a`, portant les memes faces.
    int verifie() const {
        int faux = 0;
        for ( int a = 0; a < nv; ++a ) {
            const int ka[ 3 ] = { vk0[ a ], vk1[ a ], vk2[ a ] };
            const int wa[ 3 ] = { vn0[ a ], vn1[ a ], vn2[ a ] };
            if ( wa[ 0 ] == wa[ 1 ] || wa[ 0 ] == wa[ 2 ] || wa[ 1 ] == wa[ 2 ] ) ++faux;
            for ( int j = 0; j < 3; ++j ) {
                const int b = wa[ j ];
                if ( b < 0 || b >= nv ) { ++faux; continue; }
                const int kb[ 3 ] = { vk0[ b ], vk1[ b ], vk2[ b ] };
                const int wb[ 3 ] = { vn0[ b ], vn1[ b ], vn2[ b ] };
                int t = -1;
                for ( int r = 0; r < 3; ++r ) if ( wb[ r ] == a ) t = r;
                if ( t < 0 ) { ++faux; continue; }
                int f0, f1, g0, g1;
                faces_de( ka, j, f0, f1 );
                faces_de( kb, t, g0, g1 );
                if ( f0 != g0 || f1 != g1 ) ++faux;
            }
        }
        return faux;
    }

    /// LES VOISINS : les identifiants GLOBAUX des coupes encore portees par un sommet.
    int voisins( SI32 *out, int cap ) const {
        bool vivant[ max_nc ] = {};
        for ( int i = 0; i < nv; ++i ) { vivant[ vk0[i] ] = true; vivant[ vk1[i] ] = true; vivant[ vk2[i] ] = true; }
        int m = 0;
        for ( int k = 0; k < nc; ++k )
            if ( vivant[ k ] && m < cap ) out[ m++ ] = cid[ k ];
        return m;
    }
};

} // namespace sf::d3
