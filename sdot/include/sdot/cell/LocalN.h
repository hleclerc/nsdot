#pragma once

// =====================================================================================
// LA CELLULE EN DIMENSION `D >= 3` : un polytope SIMPLE qui se coupe lui-meme.
//
// Rien de la 2D ne survit ici : il n'y a plus d'ordre cyclique global, et une coupe porte une
// FACE. La connectivite est donc portee par les SOMMETS, chacun nommant ses `D` coupes et ses `D`
// voisins :
//
//     vk[ r ][ i ]   la `r`-ieme coupe du sommet `i`, comme INDICE dans `cid` -- triees croissant
//     vn[ r ][ i ]   le voisin de `i` DE L'AUTRE COTE de l'arete portee par les `D - 1` autres
//                    coupes que `vk[ r ][ i ]` : le voisin `r` est « en face » de la coupe `r`
//
// Un sommet d'un polytope simple a `D` coupes, donc `D` paquets de `D - 1` coupes, donc `D`
// aretes, et cette mise en correspondance est une bijection. Elle rend GRATUIT ce qu'une liste
// d'aretes faisait chercher : les faces qui portent l'arete `r` sont les coupes du sommet privees
// de la `r`-ieme, sans rien parcourir.
//
// LA LISTE DE COUPES EST LOCALE : `cid[ 0 .. nc )` porte les identifiants globaux, tout le reste ne
// manipule que des indices dedans. Elle ne fait qu'AJOUTER pendant la vie de la cellule -- une
// coupe qui perd tous ses sommets y laisse une entree morte -- et `compacte()` ne l'enleve que
// lorsqu'elle est pleine.
//
// L'HYPOTHESE : chaque sommet est sur EXACTEMENT `D` plans ( position generale ). C'est ce qui rend
// la coupe purement combinatoire : deux sommets neufs de la face creee sont voisins exactement
// quand ils partagent `D - 2` ANCIENNES coupes. Un plan qui passe exactement par un sommet ne
// l'enleve pas ( `s > 0` strict ), ce qui est la seule concession faite aux configurations
// degenerees.
//
// LA COUPE REMPLIT LES TROUS : les sommets dehors laissent des places libres, les sommets neufs
// s'y installent, et les survivants gardent leur indice -- donc leur adjacence reste valable
// telle quelle, et il n'y a pas de table de renumerotation. Le commit est en `O( nm )`, pas en
// `O( nv )` ; le seul cas ou un sommet garde bouge est celui d'une coupe qui enleve plus de
// sommets qu'elle n'en cree, et il n'en bouge alors que la difference.
//
// C'est le `Cellule3D.h` du banc, la dimension en parametre.
// =====================================================================================

#include <loom/support/common_types.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/containers/Vector.h>
#include "Scratch.h"
#include "Plane.h"
#include "Etat.h"
#include "Ids.h"

#include <type_traits>
#include <limits>
#include <cmath>

namespace sdot {

template<class TK,int D>
struct LocalN {
    static_assert( D >= 3, "en dessous de 3D, c'est `Local2`" );
    static constexpr int ct_dim = D;
    using TKernel = TK;
    using PlaneT  = Plane<TK,D>;

    int  nv = 0, nc = 0;
    int  cap = 0;                                        ///< la capacite : sommets ET coupes
    bool unbounded  = false;
    bool has_planes = false;

    TK  *v [ D ];
    int *vk[ D ];
    int *vn[ D ];
    int *cid;
    TK  *pd[ D ], *po;                                   ///< le plan de la coupe `k` ( si `has_planes` )

    // les temporaires, dans le scratch eux aussi
    TK  *s, *nx[ D ], *rate[ D ], *s3[ 3 ];
    int *trou, *nk[ D - 1 ], *rec_v, *rec_f, *dest, *nn_[ D ], *nouv, *src, *dst, *m, *apex[ D ], *v0, *on_face;

    // ---- le scratch --------------------------------------------------------------------------

    /// ce qu'il faut de mots pour `cap` sommets ( et autant de coupes ) -- LA MEME FORMULE que
    /// `Cell_N.scratch_words`
    static constexpr SI words_for( SI cap ) {
        return ( 4 * D + 5 ) * words_of<TK>( cap ) + ( 5 * D + 10 ) * words_of<int>( cap );
    }

    bool attach( Carver &c, SI capacity ) {
        cap = int( capacity );
        for ( int d = 0; d < D; ++d ) { v[ d ] = c.take<TK>( cap ); vk[ d ] = c.take<int>( cap ); vn[ d ] = c.take<int>( cap ); }
        cid = c.take<int>( cap );
        for ( int d = 0; d < D; ++d ) pd[ d ] = c.take<TK>( cap );
        po = c.take<TK>( cap );
        s = c.take<TK>( cap );
        for ( int d = 0; d < D; ++d ) { nx[ d ] = c.take<TK>( cap ); rate[ d ] = c.take<TK>( cap ); }
        for ( int d = 0; d < 3; ++d ) s3[ d ] = c.take<TK>( cap );
        trou = c.take<int>( cap );
        for ( int d = 0; d + 1 < D; ++d ) nk[ d ] = c.take<int>( cap );
        rec_v = c.take<int>( cap ); rec_f = c.take<int>( cap ); dest = c.take<int>( cap );
        for ( int d = 0; d < D; ++d ) nn_[ d ] = c.take<int>( cap );
        nouv = c.take<int>( cap ); src = c.take<int>( cap ); dst = c.take<int>( cap ); m = c.take<int>( cap );
        for ( int d = 0; d < D; ++d ) apex[ d ] = c.take<int>( cap );
        v0 = c.take<int>( cap ); on_face = c.take<int>( cap );
        nv = 0; nc = 0; unbounded = false; has_planes = false;
        return ! c.overflow;
    }

    bool copy_from( const LocalN &o ) {
        if ( o.nv > cap || o.nc > cap )
            return false;
        nv = o.nv; nc = o.nc; unbounded = o.unbounded; has_planes = o.has_planes;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) { v[ d ][ i ] = o.v[ d ][ i ]; vk[ d ][ i ] = o.vk[ d ][ i ]; vn[ d ][ i ] = o.vn[ d ][ i ]; }
        for ( int k = 0; k < nc; ++k ) {
            cid[ k ] = o.cid[ k ];
            if ( has_planes ) { for ( int d = 0; d < D; ++d ) pd[ d ][ k ] = o.pd[ d ][ k ]; po[ k ] = o.po[ k ]; }
        }
        return true;
    }

    // ---- ce que tout le monde lit ------------------------------------------------------------
    int  nb_vertices() const { return nv; }
    int  nb_cuts    () const { return nc; }
    bool bounded    () const { return ! unbounded; }
    TK   coord      ( int i, int d ) const { return v[ d ][ i ]; }
    int  vertex_cut ( int i, int r ) const { return vk[ r ][ i ]; }

    EtatMemN<TK,D> etat() const {
        EtatMemN<TK,D> e;
        e.nb = nv;
        for ( int d = 0; d < D; ++d )
            e.v[ d ] = v[ d ];
        e.bounded = ! unbounded;
        return e;
    }

    // ---- les etats de depart -----------------------------------------------------------------

    /// le parallelotope `origin + sum_j t_j axes( j )`, `t` dans `[ 0, 1 ]^D`. Le sommet `b` a pour
    /// coordonnees les bits de `b` ; la coupe `2 j + bit_j( b )` le porte, et son voisin en face
    /// de cette coupe est `b ^ ( 1 << j )`. Les coupes sont dans l'ordre des axes, donc triees.
    bool init_hypercube( const auto &origin, const auto &axes, int cut_id ) {
        if ( ( 1 << D ) > cap )
            return false;
        nv = 1 << D;
        nc = 2 * D;
        for ( int b = 0; b < nv; ++b ) {
            for ( int d = 0; d < D; ++d ) {
                TK x = TK( origin[ d ] );
                for ( int j = 0; j < D; ++j )
                    if ( b & ( 1 << j ) )
                        x += TK( axes( j, d ) );
                v[ d ][ b ] = x;
            }
            for ( int j = 0; j < D; ++j ) {
                vk[ j ][ b ] = 2 * j + ( ( b >> j ) & 1 );
                vn[ j ][ b ] = b ^ ( 1 << j );
            }
        }
        for ( int k = 0; k < nc; ++k )
            cid[ k ] = cut_id;
        unbounded  = cut_id == cell_ids::INFINITE;
        has_planes = false;
        if ( unbounded )
            planes_from_vertices();
        return true;
    }

    /// « TOUT L'ESPACE » : le simplexe unite, dont les `D + 1` parois sont marquees `INFINITE`. Le
    /// sommet 0 est l'origine, sur les coupes `0 .. D-1` ( `x_c >= 0` ) ; le sommet `n >= 1` est
    /// `e_{n-1}`, sur les memes privees de `n-1`, plus la coupe `D` ( `sum x <= 1` ).
    bool init_unbounded() {
        if ( D + 1 > cap )
            return false;
        nv = D + 1;
        nc = D + 1;
        for ( int n = 0; n <= D; ++n )
            for ( int d = 0; d < D; ++d )
                v[ d ][ n ] = TK( d + 1 == n );
        for ( int r = 0; r < D; ++r ) {
            vk[ r ][ 0 ] = r;
            vn[ r ][ 0 ] = r + 1;                        // en face de `x_r >= 0` : le long de `e_r`
        }
        for ( int n = 1; n <= D; ++n ) {
            for ( int r = 0; r + 1 < D; ++r ) {
                const int c = r + ( r >= n - 1 );
                vk[ r ][ n ] = c;
                vn[ r ][ n ] = c + 1;                    // le sommet qui manque aussi la coupe `c`
            }
            vk[ D - 1 ][ n ] = D;
            vn[ D - 1 ][ n ] = 0;                        // en face de la fermeture : l'origine
        }
        for ( int k = 0; k <= D; ++k )
            cid[ k ] = cell_ids::INFINITE;
        for ( int c = 0; c < D; ++c ) {
            for ( int d = 0; d < D; ++d )
                pd[ d ][ c ] = - TK( d == c );
            po[ c ] = 0;
        }
        for ( int d = 0; d < D; ++d )
            pd[ d ][ D ] = 1;
        po[ D ] = 1;
        unbounded  = true;
        has_planes = true;
        return true;
    }

    void make_empty() { nv = 0; nc = 0; unbounded = false; has_planes = false; }

    // ---- les plans, relus sur la geometrie ---------------------------------------------------

    /// la normale SORTANTE de la face `k` et son offset, relus sur ses sommets. `false` si la face
    /// n'engendre pas un hyperplan ( morte, ou degeneree ).
    ///
    /// PAS « les `D` premiers sommets trouves » : sur une facette de dimension `D - 1 >= 3`, `D`
    /// sommets peuvent tres bien tenir dans une meme 2-face, et deux sommets confondus a 1e-15
    /// ( une coupe passee par un sommet ) sont le cas courant. On construit donc une base
    /// ORTHONORMEE de l'espace engendre par `p - p0`, par Gram-Schmidt, en ne gardant qu'un point
    /// dont le residu est franc ; la normale est le produit vectoriel generalise de cette base.
    bool plane_of_cut( int k, TK *dir, TK &off ) const {
        int p0 = -1;
        TK  base[ D - 1 ][ D ];
        TK  scale = 0;
        int nb = 0;
        for ( int i = 0; i < nv; ++i ) {
            bool on = false;
            for ( int r = 0; r < D; ++r )
                on |= vk[ r ][ i ] == k;
            on_face[ i ] = on;
            if ( ! on )
                continue;
            if ( p0 < 0 ) { p0 = i; continue; }
            if ( nb == D - 1 )
                continue;
            TK u[ D ];
            TK n2 = 0;
            for ( int d = 0; d < D; ++d ) { u[ d ] = v[ d ][ i ] - v[ d ][ p0 ]; n2 += u[ d ] * u[ d ]; }
            if ( n2 > scale ) scale = n2;
            for ( int b = 0; b < nb; ++b ) {
                TK dot = 0;
                for ( int d = 0; d < D; ++d ) dot += u[ d ] * base[ b ][ d ];
                for ( int d = 0; d < D; ++d ) u[ d ] -= dot * base[ b ][ d ];
            }
            TK r2 = 0;
            for ( int d = 0; d < D; ++d ) r2 += u[ d ] * u[ d ];
            if ( r2 <= scale * TK( 1e-12 ) )
                continue;                                // dans l'espace deja engendre, ou confondu
            const TK inv = 1 / std::sqrt( r2 );
            for ( int d = 0; d < D; ++d ) base[ nb ][ d ] = u[ d ] * inv;
            ++nb;
        }
        if ( nb < D - 1 )
            return false;

        // le produit vectoriel generalise des `D - 1` vecteurs de base : `n_i` est le mineur sans
        // la colonne `i`, au signe `( -1 )^i`
        for ( int i = 0; i < D; ++i ) {
            const auto M = Matrix<TK,D-1>::with_func( [&]( auto r, auto c ) {
                const int col = int( c ) + ( int( c ) >= i );
                return base[ int( r ) ][ col ];
            } );
            const TK m = M.determinant();
            dir[ i ] = ( i % 2 ) ? - m : m;
        }
        off = 0;
        for ( int d = 0; d < D; ++d )
            off += dir[ d ] * v[ d ][ p0 ];

        // sortante : le sommet LE PLUS LOIN du plan est dedans ( pas le premier venu, qui peut etre
        // un sommet confondu avec la face, a 1e-16 d'un cote ou de l'autre )
        TK far = 0;
        for ( int i = 0; i < nv; ++i ) {
            if ( on_face[ i ] )
                continue;
            TK s = - off;
            for ( int d = 0; d < D; ++d )
                s += dir[ d ] * v[ d ][ i ];
            if ( ( s < 0 ? - s : s ) > ( far < 0 ? - far : far ) )
                far = s;
        }
        if ( far > 0 ) {
            for ( int d = 0; d < D; ++d )
                dir[ d ] = - dir[ d ];
            off = - off;
        }
        return true;
    }

    void planes_from_vertices() {
        for ( int k = 0; k < nc; ++k ) {
            TK dir[ D ], off;
            if ( ! plane_of_cut( k, dir, off ) ) {
                for ( int d = 0; d < D; ++d ) dir[ d ] = 0;
                off = 0;
            }
            for ( int d = 0; d < D; ++d )
                pd[ d ][ k ] = dir[ d ];
            po[ k ] = off;
        }
        has_planes = true;
    }

    template<class T>
    void plane( int k, T *dir, T &off ) const {
        if ( has_planes ) {
            for ( int d = 0; d < D; ++d ) dir[ d ] = T( pd[ d ][ k ] );
            off = T( po[ k ] );
        } else {
            TK dk[ D ], o = 0;
            if ( ! plane_of_cut( k, dk, o ) )
                for ( int d = 0; d < D; ++d ) dk[ d ] = 0;
            for ( int d = 0; d < D; ++d ) dir[ d ] = T( dk[ d ] );
            off = T( o );
        }
    }

    // ---- la coupe ----------------------------------------------------------------------------

    int cut( const PlaneT &p ) {
        if ( unbounded )
            grow_for( p );
        return cut_impl( p );
    }

    int nb_outside( const PlaneT &p ) const {
        int res = 0;
        for ( int i = 0; i < nv; ++i ) {
            TK s = - p.off;
            for ( int d = 0; d < D; ++d )
                s += p.dir[ d ] * v[ d ][ i ];
            res += s > 0;
        }
        return res;
    }

    int cut_impl( const PlaneT &p ) {
        int nb_out = 0;
        for ( int i = 0; i < nv; ++i ) {
            TK si = - p.off;
            for ( int d = 0; d < D; ++d )
                si += p.dir[ d ] * v[ d ][ i ];
            s[ i ] = si;
            nb_out += si > 0;
        }
        if ( nb_out == 0 )
            return CutStatus::UNCHANGED;
        if ( nb_out == nv ) {
            nv = 0; nc = 0;
            unbounded = false;
            return CutStatus::EMPTY;
        }

        if ( nc >= cap ) {
            compacte();
            if ( nc >= cap )
                return CutStatus::OVERFLOW;
        }
        const int knew = nc;

        // ---- UNE SEULE PASSE SUR LES SOMMETS : les trous que laissent les sommets dehors et, pour
        // chacun d'eux, ses aretes traversantes -- d'ou naissent les sommets neufs. `nk` : les
        // coupes HERITEES, triees ; `rec_v` / `rec_f` : le sommet DEDANS a recoller, et sa fente.
        int nt = 0, nm = 0;

        for ( int o = 0; o < nv; ++o ) {
            if ( ! ( s[ o ] > 0 ) )
                continue;
            trou[ nt++ ] = o;
            for ( int j = 0; j < D; ++j ) {
                const int u = vn[ j ][ o ];
                if ( s[ u ] > 0 )
                    continue;                            // arete entierement dehors : elle meurt
                if ( nm >= cap )
                    return CutStatus::OVERFLOW;

                // ANCRE SUR LE SOMMET DEDANS : avec `s_u == 0` la forme symetrique ne rend pas `v_u`
                // en flottant, et le sommet passerait de l'autre cote du plan.
                const TK t = s[ u ] / ( s[ u ] - s[ o ] );
                for ( int d = 0; d < D; ++d )
                    nx[ d ][ nm ] = v[ d ][ u ] + ( v[ d ][ o ] - v[ d ][ u ] ) * t;
                for ( int r = 0, q = 0; r < D; ++r )
                    if ( r != j )
                        nk[ q++ ][ nm ] = vk[ r ][ o ];

                rec_v[ nm ] = u;
                int f = 0;
                for ( int r = 0; r < D; ++r )
                    if ( vn[ r ][ u ] == o )
                        f = r;
                rec_f[ nm ] = f;
                ++nm;
            }
        }

        const int nn = nv - nt;
        const int new_nv = nn + nm;
        if ( new_nv > cap )
            return CutStatus::OVERFLOW;

        // ou va chaque sommet neuf : dans un trou tant qu'il en reste, puis a la suite
        for ( int j = 0; j < nm; ++j )
            dest[ j ] = j < nt ? trou[ j ] : nv + ( j - nt );

        // ---- LES VOISINS DES SOMMETS NEUFS. En face de `knew` ( fente `D - 1` ) : le bout dedans
        // dont il vient. Les autres sont ses voisins SUR LA FACE NEUVE : deux sommets neufs sont
        // voisins exactement quand ils partagent `D - 2` anciennes coupes, et l'arete qui les joint
        // est alors en face de la coupe heritee qu'ils NE partagent PAS.
        for ( int i = 0; i < nm; ++i ) {
            for ( int r = 0; r + 1 < D; ++r )
                nn_[ r ][ i ] = -1;
            nn_[ D - 1 ][ i ] = rec_v[ i ];
        }
        for ( int i = 0; i < nm; ++i ) {
            for ( int j = i + 1; j < nm; ++j ) {
                // les deux listes sont triees : on les fusionne en comptant les communs
                int a = 0, b = 0, common = 0, ai = -1, bj = -1;
                while ( a < D - 1 && b < D - 1 ) {
                    if      ( nk[ a ][ i ] == nk[ b ][ j ] ) { ++common; ++a; ++b; }
                    else if ( nk[ a ][ i ] <  nk[ b ][ j ] ) { ai = a; ++a; }
                    else                                     { bj = b; ++b; }
                }
                if ( common != D - 2 )
                    continue;
                if ( a < D - 1 ) ai = a;                 // le reste, non partage
                if ( b < D - 1 ) bj = b;
                nn_[ ai ][ i ] = dest[ j ];
                nn_[ bj ][ j ] = dest[ i ];
            }
        }

        // ---- COMMIT. Rien n'a bouge jusqu'ici.
        for ( int j = 0; j < nm; ++j ) {
            const int m = dest[ j ];
            for ( int d = 0; d < D; ++d )
                v[ d ][ m ] = nx[ d ][ j ];
            for ( int r = 0; r + 1 < D; ++r )
                vk[ r ][ m ] = nk[ r ][ j ];             // < `knew`, donc trie
            vk[ D - 1 ][ m ] = knew;
            for ( int r = 0; r < D; ++r )
                vn[ r ][ m ] = nn_[ r ][ j ];
        }
        for ( int i = 0; i < nm; ++i )                   // le recollage, cote sommet DEDANS
            vn[ rec_f[ i ] ][ rec_v[ i ] ] = dest[ i ];

        // ---- LES TROUS QUI RESTENT, quand la coupe enleve plus de sommets qu'elle n'en cree
        if ( nm < nt ) {
            int th = nt;
            while ( th > nm && trou[ th - 1 ] >= new_nv ) --th;

            int nmv = 0;
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
                for ( int d = 0; d < D; ++d ) v[ d ][ b ] = v[ d ][ a ];
                for ( int r = 0; r < D; ++r ) { vk[ r ][ b ] = vk[ r ][ a ]; vn[ r ][ b ] = vn[ r ][ a ]; }
            }
            // et les voisins qui pointaient vers eux ; un voisin peut lui-meme avoir demenage
            for ( int t = 0; t < nmv; ++t ) {
                const int a = src[ t ], b = dst[ t ];
                for ( int j = 0; j < D; ++j ) {
                    const int w = vn[ j ][ b ];
                    const int q = w >= new_nv ? nouv[ w - new_nv ] : w;
                    for ( int r = 0; r < D; ++r )
                        if ( vn[ r ][ q ] == a ) { vn[ r ][ q ] = b; break; }
                }
            }
        }

        cid[ knew ] = p.id;
        if ( has_planes ) {
            for ( int d = 0; d < D; ++d )
                pd[ d ][ knew ] = p.dir[ d ];
            po[ knew ] = p.off;
        }
        nc = knew + 1;
        nv = new_nv;

        if ( unbounded ) {
            unbounded = false;
            for ( int i = 0; i < nv; ++i )
                for ( int r = 0; r < D; ++r )
                    unbounded |= cid[ vk[ r ][ i ] ] == cell_ids::INFINITE;
            if ( ! unbounded )
                has_planes = false;
        }
        return CutStatus::CUT;
    }

    /// avant de poser la cellule en memoire : les coupes mortes ne sortent pas d'ici
    void tidy() { compacte(); }

    /// ENLEVER LES COUPES MORTES. La renumerotation est MONOTONE, donc les listes restent triees.
    void compacte() {
        for ( int k = 0; k < nc; ++k ) m[ k ] = -1;
        for ( int i = 0; i < nv; ++i )
            for ( int r = 0; r < D; ++r )
                m[ vk[ r ][ i ] ] = 0;
        int q = 0;
        for ( int k = 0; k < nc; ++k ) {
            if ( m[ k ] != 0 )
                continue;
            cid[ q ] = cid[ k ];
            if ( has_planes ) {
                for ( int d = 0; d < D; ++d ) pd[ d ][ q ] = pd[ d ][ k ];
                po[ q ] = po[ k ];
            }
            m[ k ] = q++;
        }
        for ( int i = 0; i < nv; ++i )
            for ( int r = 0; r < D; ++r )
                vk[ r ][ i ] = m[ vk[ r ][ i ] ];
        nc = q;
    }

    // ---- le simplexe de remplacement -----------------------------------------------------------

    /// la vitesse du sommet `i` quand on repousse les parois `INFINITE` : il resout le `D x D` de
    /// ses coupes avec les indicatrices `INFINITE` en second membre
    void growth_rate( int i, TK *rate ) const {
        bool any = false;
        for ( int r = 0; r < D; ++r )
            any |= cid[ vk[ r ][ i ] ] == cell_ids::INFINITE;
        if ( ! any ) {
            for ( int d = 0; d < D; ++d ) rate[ d ] = 0;
            return;
        }
        const auto A = Matrix<TK,D>::with_func( [&]( auto r, auto c ) { return pd[ int( c ) ][ vk[ int( r ) ][ i ] ]; } );
        const auto b = Vector<TK,D>::with_func( [&]( PI r ) { return TK( cid[ vk[ r ][ i ] ] == cell_ids::INFINITE ); } );
        const auto x = Matrix<TK,D>::solve_ge( A, b );
        for ( int d = 0; d < D; ++d )
            rate[ d ] = x[ d ];
    }

    void grow_for( const PlaneT &p ) {
        if ( ! has_planes )
            planes_from_vertices();

        static constexpr int max_rounds = 4;
        // LA MARGE N'EST PAS UN EPSILON MACHINE : un sommet repousse d'un epsilon retombe a la
        // precision du produit scalaire, et deux sommets confondus a 1e-15 se classent alors
        // chacun de son cote du plan -- ce qui casse la combinatoire de la coupe ( vu en 4D ).
        // Elle est petite devant la geometrie, grande devant l'arrondi : ou se posent les sommets
        // FACTICES n'a de toute facon aucun sens geometrique.
        const TK margin = std::is_same_v<TK,float> ? TK( 1e-5 ) : TK( 1e-6 );

        for ( int i = 0; i < nv; ++i ) {
            TK r[ D ];
            growth_rate( i, r );
            for ( int d = 0; d < D; ++d )
                rate[ d ][ i ] = r[ d ];
        }

        // « la vitesse ne fait pas varier la distance au plan » se juge A UNE TOLERANCE, pas a zero :
        // les plans sont RELUS sur la geometrie, donc une normale exactement axiale sort avec des
        // composantes a 1e-17, et `root = - s / rate` ferait alors une poussee de 1e17 -- et 1e38 au
        // tour suivant. Un rayon parallele au plan a 1e-9 pres l'est.
        TK nd = 0;
        for ( int d = 0; d < D; ++d )
            nd += p.dir[ d ] * p.dir[ d ];
        const TK tol = TK( 1e-9 ) * std::sqrt( nd );

        TK g = 0;
        for ( int round = 0; round < max_rounds; ++round ) {
            bool push = false;
            TK grow = 0;
            for ( int i = 0; i < nv; ++i ) {
                TK dr = 0, s = - p.off, nr = 0;
                for ( int d = 0; d < D; ++d ) {
                    dr += p.dir[ d ] * rate[ d ][ i ];
                    s  += p.dir[ d ] * ( v[ d ][ i ] + g * rate[ d ][ i ] );
                    nr += rate[ d ][ i ] * rate[ d ][ i ];
                }
                if ( nr == 0 || ( dr < 0 ? - dr : dr ) <= tol * std::sqrt( nr ) )
                    continue;
                const TK root = - s / dr;
                if ( root >= 0 ) {
                    push = true;
                    if ( root > grow )
                        grow = root;
                }
            }
            if ( ! push )
                break;
            g += grow + ( grow + 1 ) * margin;
        }
        if ( g == 0 )
            return;

        for ( int k = 0; k < nc; ++k )
            if ( cid[ k ] == cell_ids::INFINITE )
                po[ k ] += g;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d )
                v[ d ][ i ] += g * rate[ d ][ i ];
    }

    // ---- la mesure : un eventail de simplexes sur le treillis des faces ----------------------

    bool has_cut( int i, int k ) const {
        for ( int r = 0; r < D; ++r )
            if ( vk[ r ][ i ] == k )
                return true;
        return false;
    }

    /// `func( chain )` pour chaque simplexe de la triangulation standard : un sommet de la cellule,
    /// cone sur chaque facette qui ne le contient pas, chacune trianguee pareil une dimension plus
    /// bas. Il faut, pour chaque face rencontree, UN de ses sommets ( « l'apex » ) : `apex` en garde
    /// une ligne par profondeur et une case par coupe, remplie d'un seul passage sur les sommets
    /// de la face -- `D * nc` mots, la ou indexer les faces par leur ENSEMBLE de coupes coute
    /// `nc^( D - 1 )`.
    void for_each_simplex( auto &&func ) const {
        if ( nv == 0 )
            return;
        Vector<SI,D+1> chain;
        Vector<SI,D> face_cuts;
        chain[ 0 ] = 0;
        for_each_simplex_rec( chain, face_cuts, func, Ct<int,D>() );
    }

    template<int K>
    void for_each_simplex_rec( Vector<SI,D+1> &chain, Vector<SI,D> &face_cuts, auto &&func, Ct<int,K> ) const {
        constexpr int depth = D - K;                     // le nombre de coupes qui definissent la face
        if constexpr ( K == 0 ) {
            func( chain );
        } else {
            const SI p = chain[ depth ];                 // l'apex de la face, fixe pour ce sous-arbre
            for ( int c = 0; c < nc; ++c )
                apex[ depth ][ c ] = -1;
            for ( int i = 0; i < nv; ++i ) {
                bool on = true;
                for ( int m = 0; m < depth; ++m )
                    on &= has_cut( i, int( face_cuts[ m ] ) );
                if ( ! on )
                    continue;
                for ( int r = 0; r < D; ++r ) {
                    const int c = vk[ r ][ i ];
                    bool in = false;
                    for ( int m = 0; m < depth; ++m )
                        in |= int( face_cuts[ m ] ) == c;
                    if ( ! in && apex[ depth ][ c ] < 0 )
                        apex[ depth ][ c ] = i;
                }
            }
            // ... puis coner `p` sur les facettes qui ne le contiennent PAS ( les autres donneraient
            // des simplexes plats )
            for ( int c = 0; c < nc; ++c ) {
                const int a = apex[ depth ][ c ];
                if ( a < 0 || has_cut( int( p ), c ) )
                    continue;
                face_cuts[ depth ] = c;
                chain[ depth + 1 ] = a;
                for_each_simplex_rec( chain, face_cuts, func, Ct<int,K-1>() );
            }
        }
    }

    /// EN 3D : ACCUMULER LES FACES PLUTOT QUE LES ORDONNER. Ni le volume ni l'aire d'une face n'ont
    /// besoin de l'ORDRE des sommets : il suffit, par face, d'UN de ses sommets `v_f` et de la somme
    /// `S_f` des produits vectoriels de ses aretes vues depuis lui -- la face est plane et convexe,
    /// donc les triangles `( v_f, arete )` la pavent et leurs produits vectoriels sont paralleles.
    /// Le volume vaut `sum_f | ( v_f - g ) . S_f | / 6` pour n'importe quel `g` interieur. Deux
    /// passes sans jamais chercher : `O( V + E )` au lieu de `O( F ( V + E ) )` pour les cycles --
    /// mesure sur le banc : -15 % sur le diagramme 3D entier.
    /// EN 3D : par coupe `f`, UN sommet `v0[ f ]` et la somme `s3[ . ][ f ]` des produits vectoriels
    /// de ses aretes vues depuis lui -- deux fois le vecteur aire de la face. Ce que `measure_3d` et
    /// `for_each_facet` lisent tous deux ( voir `measure_3d` pour pourquoi accumuler plutot qu'ordonner ).
    void accumulate_faces_3d() const {
        static_assert( D == 3 );
        TK *const *s = s3;
        for ( int k = 0; k < nc; ++k ) { v0[ k ] = -1; s[ 0 ][ k ] = s[ 1 ][ k ] = s[ 2 ][ k ] = 0; }
        for ( int i = nv - 1; i >= 0; --i )
            for ( int r = 0; r < 3; ++r )
                v0[ vk[ r ][ i ] ] = i;

        for ( int a = 0; a < nv; ++a ) {
            for ( int j = 0; j < 3; ++j ) {
                const int b = vn[ j ][ a ];
                if ( b <= a )
                    continue;                            // chaque arete vue une seule fois
                // les deux faces qui portent l'arete `j` : les coupes du sommet privees de la `j`-ieme
                for ( int r = 0; r < 3; ++r ) {
                    if ( r == j )
                        continue;
                    const int f = vk[ r ][ a ];
                    const int o = v0[ f ];
                    const TK ax = v[0][a] - v[0][o], ay = v[1][a] - v[1][o], az = v[2][a] - v[2][o];
                    const TK bx = v[0][b] - v[0][o], by = v[1][b] - v[1][o], bz = v[2][b] - v[2][o];
                    TK cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
                    if ( s[0][f] * cx + s[1][f] * cy + s[2][f] * cz < 0 ) { cx = -cx; cy = -cy; cz = -cz; }
                    s[0][f] += cx; s[1][f] += cy; s[2][f] += cz;
                }
            }
        }
    }

    /// `func( c, mesure )` pour chaque coupe `c` qui porte une face : son AIRE ( 3D seulement )
    template<class TF>
    void for_each_facet( auto &&func ) const {
        static_assert( D == 3, "for_each_facet : 3D seulement au-dela du plan" );
        if ( nv < 4 )
            return;
        accumulate_faces_3d();
        TK *const *s = s3;
        for ( int k = 0; k < nc; ++k ) {
            if ( v0[ k ] < 0 )
                continue;
            const TF sx = TF( s[0][k] ), sy = TF( s[1][k] ), sz = TF( s[2][k] );
            func( k, sycl::sqrt( sx * sx + sy * sy + sz * sz ) / 2 );
        }
    }

    template<class TF>
    TF measure_3d() const {
        static_assert( D == 3 );
        if ( nv < 4 )
            return 0;
        // l'accumulation se fait dans le flottant du NOYAU ( les tableaux `s3` sont en `TK` ) ;
        // la somme finale est en `TF`
        accumulate_faces_3d();
        TK *const *s = s3;

        TF g[ 3 ] = { 0, 0, 0 };
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < 3; ++d )
                g[ d ] += TF( v[ d ][ i ] );
        for ( int d = 0; d < 3; ++d )
            g[ d ] /= nv;

        TF vol = 0;
        for ( int k = 0; k < nc; ++k ) {
            const int o = v0[ k ];
            if ( o < 0 )
                continue;                                // coupe morte, pas encore compactee
            const TF t = ( TF( v[0][o] ) - g[0] ) * TF( s[0][k] ) + ( TF( v[1][o] ) - g[1] ) * TF( s[1][k] ) + ( TF( v[2][o] ) - g[2] ) * TF( s[2][k] );
            vol += t < 0 ? -t : t;
        }
        return vol / 6;
    }

    template<class TF>
    TF measure() const {
        if ( unbounded )
            return std::numeric_limits<TF>::max();
        if constexpr ( D == 3 )
            return measure_3d<TF>();
        TF sum = 0;
        for_each_simplex( [&]( const auto &chain ) {
            const auto M = Matrix<TF,D>::with_func( [&]( auto r, auto c ) {
                return TF( v[ int( r ) ][ chain[ int( c ) + 1 ] ] ) - TF( v[ int( r ) ][ chain[ 0 ] ] );
            } );
            const TF det = M.determinant();
            sum += det < 0 ? - det : det;
        } );
        TF fact = 1;
        for ( int i = 2; i <= D; ++i )
            fact *= i;
        return sum / fact;
    }

    /// l'adjoint : `grad_vp( i, d )` ACCUMULE ( un sommet est dans plusieurs simplexes ), donc il
    /// est mis a zero d'abord, sur les `nv` sommets.
    template<class TF>
    void measure_bwd( TF grad_res, auto &&grad_vp ) const {
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d )
                grad_vp( i, d ) = 0;
        if ( unbounded )
            return;

        TF fact = 1;
        for ( int i = 2; i <= D; ++i )
            fact *= i;
        const TF g = grad_res / fact;

        // `d|det|/dM = sign( det ) * cofactor( M )`, et chaque colonne de `M` est un apex moins le
        // premier, qui recoit donc MOINS la somme des colonnes
        for_each_simplex( [&]( const auto &chain ) {
            const auto M = Matrix<TF,D>::with_func( [&]( auto r, auto c ) {
                return TF( v[ int( r ) ][ chain[ int( c ) + 1 ] ] ) - TF( v[ int( r ) ][ chain[ 0 ] ] );
            } );
            const TF det = M.determinant();
            const TF sg = det < 0 ? - g : g;
            for ( int r = 0; r < D; ++r ) {
                TF row_sum = 0;
                for ( int c = 0; c < D; ++c ) {
                    const TF minor = M.without_row_and_col( r, c ).determinant();
                    const TF cof = ( ( r + c ) % 2 ? - minor : minor ) * sg;
                    grad_vp( chain[ c + 1 ], r ) += cof;
                    row_sum += cof;
                }
                grad_vp( chain[ 0 ], r ) -= row_sum;
            }
        } );
    }

    void bbox( TK *lo, TK *hi ) const {
        for ( int d = 0; d < D; ++d )
            lo[ d ] = hi[ d ] = nv ? v[ d ][ 0 ] : TK( 0 );
        for ( int i = 1; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) {
                lo[ d ] = v[ d ][ i ] < lo[ d ] ? v[ d ][ i ] : lo[ d ];
                hi[ d ] = v[ d ][ i ] > hi[ d ] ? v[ d ][ i ] : hi[ d ];
            }
    }

    // ---- les tenseurs ------------------------------------------------------------------------

    /// depuis une vue `Cell_N` : `vertex_positions [ nv, D ]`, `vertex_cuts` / `vertex_nbrs`
    /// `[ nv, D ]`, `cut_ids [ nc ]`. `false` si elle ne tient pas dans `cap`.
    bool load( const auto &c ) {
        const int n = int( SI( c.nb_vertices ) ), k = int( SI( c.nb_cuts ) );
        if ( n > cap || k > cap )
            return false;
        nv = n; nc = k;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) {
                v [ d ][ i ] = TK( c.vertex_positions( i, d ) );
                vk[ d ][ i ] = int( SI( c.vertex_cuts( i, d ) ) );
                vn[ d ][ i ] = int( SI( c.vertex_nbrs( i, d ) ) );
            }
        for ( int q = 0; q < nc; ++q )
            cid[ q ] = int( SI( c.cut_ids( q ) ) );
        // sur les sommets et non sur la liste : une coupe MORTE ( sans sommet ) peut y trainer
        // jusqu'a la prochaine compaction, et une paroi morte ne borne rien
        unbounded = false;
        for ( int i = 0; i < nv; ++i )
            for ( int r = 0; r < D; ++r )
                unbounded |= cid[ vk[ r ][ i ] ] == cell_ids::INFINITE;
        has_planes = false;
        return true;
    }

    bool store( auto &&c ) const {
        if ( ! c.nb_vertices.set( nv ) )
            return false;
        if ( ! c.nb_cuts.set( nc ) )
            return false;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) {
                c.vertex_positions( i, d ) = v [ d ][ i ];
                c.vertex_cuts     ( i, d ) = vk[ d ][ i ];
                c.vertex_nbrs     ( i, d ) = vn[ d ][ i ];
            }
        for ( int k = 0; k < nc; ++k )
            c.cut_ids( k ) = cid[ k ];
        return true;
    }
};

} // namespace sdot
