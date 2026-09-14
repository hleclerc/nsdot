#pragma once

// =====================================================================================
// LA CELLULE 2D EN MEMOIRE : la representation intermediaire, celle que les noyaux manipulent.
//
// Les sommets en ORDRE CYCLIQUE sont toute la geometrie : l'aire se lit par le lacet, et
// l'invariant « LA COUPE `i` PORTE L'ARETE `[ v_i, v_i+1 ]` » donne la connectivite sans rien
// chercher -- le sommet `i` est le coin des coupes `i-1` et `i`, et `nb_cuts == nb_vertices`. Tout
// tient dans deux tableaux de coordonnees et un tableau d'identifiants de coupe, decoupes dans le
// scratch du work-item ( `Scratch.h` ) a une capacite `cap` que l'hote a decidee.
//
// Ce n'est PAS ce qu'un utilisateur voit (`Cell.py` en derive les tenseurs « pratiques ») : c'est
// ce sur quoi les algorithmes deja ecrits tournent -- la coupe scalaire en place, l'excursion du
// noyau a registres (`Moteur2Reg.h`), le calcul de la mesure et son adjoint.
//
// = LES PLANS NE SONT PAS STOCKES, sauf quand il le faut
//
// Une cellule BORNEE n'a pas besoin de ses plans : ils se relisent sur ses aretes
// (`planes_from_vertices`), et ni la coupe ni la mesure ne les lisent. Une cellule NON BORNEE, en
// revanche, est un simplexe de remplacement dont les parois `INFINITE` portent des offsets
// inventes qu'il faut REPOUSSER avant chaque coupe (`grow_for`) -- et pour cela il faut les plans.
// Ils ne sont donc tenus (`has_planes`) que dans ce regime, qui est le rare.
//
// = LA COUPE SCALAIRE, EN PLACE
//
// L'exterieur d'un convexe coupe par un demi-plan est une plage CYCLIQUE contigue : on compte les
// sommets dehors, on trouve le debut de la plage, et la sortie fait EXACTEMENT
// `nb - nb_out + 2` sommets. Les deux intersections sont ancrees sur le sommet DEDANS
// (`v_in + ( v_out - v_in ) * t` vaut exactement `v_in` en `t == 0`, ce que la forme symetrique
// n'a pas -- et sans quoi la plage cesse d'etre contigue).
//
// Le scan est en DEUX passes, et c'est le contraire de ce qu'on croit : compter et chercher le
// debut de la plage dans la meme boucle fait moins d'instructions et va plus lentement, la
// dependance portee empechant le vectoriseur de prendre le produit scalaire. Mesure sur le banc :
// 17.1 ns par coupe en une passe, 13.2 en deux.
// =====================================================================================

#include <loom/support/common_types.h>
#include <loom/support/containers/Vector.h>
#include "Scratch.h"
#include "Plane.h"
#include "Etat.h"
#include "Ids.h"

#include <type_traits>
#include <limits>
#include <cmath>

namespace sdot {

template<class TK>
struct Local2 {
    static constexpr int ct_dim = 2;
    using TKernel = TK;
    using PlaneT  = Plane<TK,2>;

    int  nb         = 0;      ///< sommets ( = coupes ) ; 0 = vide
    int  cap        = 0;      ///< la capacite des tableaux
    bool unbounded  = false;  ///< il reste des parois `INFINITE`
    bool has_planes = false;  ///< `pdx / pdy / po` sont a jour

    TK  *vx = nullptr, *vy = nullptr;
    int *cid = nullptr;
    TK  *pdx = nullptr, *pdy = nullptr, *po = nullptr;   ///< le plan de la coupe `i` ( si `has_planes` )
    TK  *s = nullptr;                                    ///< les produits scalaires de la coupe en cours
    TK  *rx = nullptr, *ry = nullptr;                    ///< les vitesses de poussee ( non borne )

    // ---- le scratch --------------------------------------------------------------------------

    /// ce qu'il faut de mots pour `cap` sommets -- LA MEME FORMULE que `Cell_2.scratch_words`
    static constexpr SI words_for( SI cap ) { return 8 * words_of<TK>( cap ) + words_of<int>( cap ); }

    /// pose les tableaux dans `c`. `false` s'il n'y a pas la place ( rien n'est alors utilisable )
    bool attach( Carver &c, SI capacity ) {
        cap = int( capacity );
        vx = c.take<TK>( cap ); vy = c.take<TK>( cap ); cid = c.take<int>( cap );
        pdx = c.take<TK>( cap ); pdy = c.take<TK>( cap ); po = c.take<TK>( cap );
        s = c.take<TK>( cap ); rx = c.take<TK>( cap ); ry = c.take<TK>( cap );
        nb = 0; unbounded = false; has_planes = false;
        return ! c.overflow;
    }

    /// recopie `o` ( geometrie et etat ) : `cap` doit suffire
    bool copy_from( const Local2 &o ) {
        if ( o.nb > cap )
            return false;
        nb = o.nb; unbounded = o.unbounded; has_planes = o.has_planes;
        for ( int i = 0; i < nb; ++i ) {
            vx[ i ] = o.vx[ i ]; vy[ i ] = o.vy[ i ]; cid[ i ] = o.cid[ i ];
            if ( has_planes ) { pdx[ i ] = o.pdx[ i ]; pdy[ i ] = o.pdy[ i ]; po[ i ] = o.po[ i ]; }
        }
        return true;
    }

    // ---- ce que tout le monde lit ------------------------------------------------------------
    int  nb_vertices() const { return nb; }
    int  nb_cuts    () const { return nb; }
    bool bounded    () const { return ! unbounded; }
    TK   coord      ( int i, int d ) const { return d ? vy[ i ] : vx[ i ]; }

    /// les deux coupes du sommet `i`, dans l'ordre : `r == 0` -> la coupe `i-1`, `r == 1` -> `i`
    int  vertex_cut ( int i, int r ) const { return r ? i : ( i ? i - 1 : nb - 1 ); }

    /// ce que le fournisseur voit ( voir `Moteur.h` )
    EtatMem<TK> etat() const { return { nb, vx, vy, cid, ! unbounded }; }

    // ---- les etats de depart -----------------------------------------------------------------

    /// le parallelogramme `origin + s * axes( 0 ) + t * axes( 1 )`, `s, t` dans `[ 0, 1 ]`, en ordre
    /// cyclique direct si `axes` l'est. Toutes les coupes portent `cut_id`.
    bool init_hypercube( const auto &origin, const auto &axes, int cut_id ) {
        if ( cap < 4 )
            return false;
        const TK ox = TK( origin[ 0 ] ), oy = TK( origin[ 1 ] );
        const TK ax = TK( axes( 0, 0 ) ), ay = TK( axes( 0, 1 ) );
        const TK bx = TK( axes( 1, 0 ) ), by = TK( axes( 1, 1 ) );
        vx[ 0 ] = ox;           vy[ 0 ] = oy;
        vx[ 1 ] = ox + ax;      vy[ 1 ] = oy + ay;
        vx[ 2 ] = ox + ax + bx; vy[ 2 ] = oy + ay + by;
        vx[ 3 ] = ox + bx;      vy[ 3 ] = oy + by;
        for ( int i = 0; i < 4; ++i )
            cid[ i ] = cut_id;
        nb         = 4;
        unbounded  = cut_id == cell_ids::INFINITE;
        has_planes = false;
        if ( unbounded )
            planes_from_vertices();
        return true;
    }

    /// « TOUT LE PLAN » : le triangle `( 0, 0 ), ( 1, 0 ), ( 0, 1 )` dont les trois cotes sont
    /// marques `INFINITE`. Ses offsets sont inventes ; `grow_for` les repousse coupe apres coupe.
    bool init_unbounded() {
        if ( cap < 3 )
            return false;
        vx[ 0 ] = 0; vy[ 0 ] = 0;
        vx[ 1 ] = 1; vy[ 1 ] = 0;
        vx[ 2 ] = 0; vy[ 2 ] = 1;
        for ( int i = 0; i < 3; ++i )
            cid[ i ] = cell_ids::INFINITE;
        nb        = 3;
        unbounded = true;
        planes_from_vertices();
        return true;
    }

    void make_empty() { nb = 0; unbounded = false; has_planes = false; }
    void tidy() {}                                       ///< rien a ranger : pas de coupe morte en 2D

    // ---- les plans, relus sur la geometrie ---------------------------------------------------

    /// le plan de l'arete `i` : la normale SORTANTE de `[ v_i, v_i+1 ]` pour un polygone direct,
    /// et l'offset lu sur `v_i`. Exact pour un plan reel comme pour une paroi repoussee -- les
    /// deux bouts d'une arete sont sur son plan.
    void plane_of_edge( int i, TK &dx, TK &dy, TK &off ) const {
        const int j = i + 1 < nb ? i + 1 : 0;
        dx  = vy[ j ] - vy[ i ];
        dy  = vx[ i ] - vx[ j ];
        off = dx * vx[ i ] + dy * vy[ i ];
    }

    void planes_from_vertices() {
        for ( int i = 0; i < nb; ++i )
            plane_of_edge( i, pdx[ i ], pdy[ i ], po[ i ] );
        has_planes = true;
    }

    /// le plan de la coupe `i`, dans le flottant `T` de l'appelant ( relu sur les sommets, ou
    /// pris dans la table quand elle est tenue )
    template<class T>
    void plane( int i, T *dir, T &off ) const {
        if ( has_planes ) {
            dir[ 0 ] = T( pdx[ i ] ); dir[ 1 ] = T( pdy[ i ] ); off = T( po[ i ] );
        } else {
            TK dx, dy, o;
            plane_of_edge( i, dx, dy, o );
            dir[ 0 ] = T( dx ); dir[ 1 ] = T( dy ); off = T( o );
        }
    }

    // ---- la coupe ----------------------------------------------------------------------------

    /// Coupe par `p`, EN PLACE. Rend un `CutStatus` ; sur `OVERFLOW` la cellule est restee intacte.
    int cut( const PlaneT &p ) {
        if ( unbounded ) {
            grow_for( p );
            return cut_impl<true>( p );
        }
        return has_planes ? cut_impl<true>( p ) : cut_impl<false>( p );
    }

    /// Combien de sommets le demi-espace laisse DEHORS -- le test « rien a enlever », a part et
    /// petit : le predicat `s > 0` n'est ecrit qu'ici et dans `cut_impl`, donc les deux ne peuvent
    /// pas repondre differemment sur un sommet a l'epsilon du plan.
    int nb_outside( const PlaneT &p ) const {
        int res = 0;
        for ( int i = 0; i < nb; ++i )
            res += ( p.dir[ 0 ] * vx[ i ] + p.dir[ 1 ] * vy[ i ] - p.off ) > 0;
        return res;
    }

    template<bool PL>
    int cut_impl( const PlaneT &p ) {
        int nb_out = 0;
        for ( int i = 0; i < nb; ++i ) {                 // reduction pure : le vectoriseur la prend
            s[ i ] = p.dir[ 0 ] * vx[ i ] + p.dir[ 1 ] * vy[ i ] - p.off;
            nb_out += s[ i ] > 0;
        }
        if ( nb_out == 0 )
            return CutStatus::UNCHANGED;
        if ( nb_out == nb ) {
            nb = 0;
            unbounded = false;
            return CutStatus::EMPTY;
        }

        int i1 = 0;                                      // unique : l'exterieur d'un convexe
        for ( int i = 0; i < nb; ++i ) {                 // coupe est d'un seul tenant
            const int q = i ? i - 1 : nb - 1;
            if ( s[ i ] > 0 && ! ( s[ q ] > 0 ) ) { i1 = i; break; }
        }

        const int nb_in  = nb - nb_out;
        const int new_nb = nb_in + 2;
        if ( new_nb > cap )
            return CutStatus::OVERFLOW;                  // la cellule reste INTACTE

        const int j0 = ( i1 + nb - 1 ) % nb;             // dernier DEDANS avant la plage
        const int j2 = ( i1 + nb_out - 1 ) % nb;         // dernier DEHORS
        const int j3 = ( j2 + 1 ) % nb;                  // premier DEDANS apres

        const TK s0 = s[ j0 ], s1 = s[ i1 ], s2 = s[ j2 ], s3 = s[ j3 ];
        const TK ta  = s0 / ( s0 - s1 );
        const TK pax = vx[ j0 ] + ( vx[ i1 ] - vx[ j0 ] ) * ta;
        const TK pay = vy[ j0 ] + ( vy[ i1 ] - vy[ j0 ] ) * ta;
        const TK tb  = s3 / ( s3 - s2 );
        const TK pbx = vx[ j3 ] + ( vx[ j2 ] - vx[ j3 ] ) * tb;
        const TK pby = vy[ j3 ] + ( vy[ j2 ] - vy[ j3 ] ) * tb;
        // la coupe `j2` porte l'arete `[ v_j2, v_j3 ]`, qui survit tronquee : elle est relue
        // MAINTENANT, `j2` va etre ecrase.
        const int bid = cid[ j2 ];
        TK bdx = 0, bdy = 0, bo = 0;
        if constexpr ( PL ) { bdx = pdx[ j2 ]; bdy = pdy[ j2 ]; bo = po[ j2 ]; }

        auto move = [ & ]( int d, int t ) {
            vx[ d ] = vx[ t ]; vy[ d ] = vy[ t ]; cid[ d ] = cid[ t ];
            if constexpr ( PL ) { pdx[ d ] = pdx[ t ]; pdy[ d ] = pdy[ t ]; po[ d ] = po[ t ]; }
        };
        auto put = [ & ]( int d, TK x, TK y, int id, TK dx, TK dy, TK o ) {
            vx[ d ] = x; vy[ d ] = y; cid[ d ] = id;
            if constexpr ( PL ) { pdx[ d ] = dx; pdy[ d ] = dy; po[ d ] = o; }
        };

        if ( i1 <= j2 ) {
            if ( nb_out == 1 ) {                         // un cran de plus : la queue va A DROITE
                for ( int i = nb; i > i1 + 1; --i ) move( i, i - 1 );
            } else if ( nb_out > 2 ) {                   // trop de place : la queue revient A GAUCHE
                const int gap = nb_out - 2;
                for ( int i = j2 + 1; i < nb; ++i ) move( i - gap, i );
            }                                            // `nb_out == 2` : rien a decaler
            put( i1,     pax, pay, p.id, p.dir[ 0 ], p.dir[ 1 ], p.off );
            put( i1 + 1, pbx, pby, bid,  bdx, bdy, bo );
        } else {
            // la plage BOUCLE, donc l'interieur est contigu : `[ j3, j3 + nb_in )`.
            if ( j3 >= 2 ) for ( int o = 0; o < nb_in; ++o ) move( 2 + o, j3 + o );
            else           for ( int o = nb_in - 1; o >= 0; --o ) move( 2 + o, j3 + o );
            put( 0, pax, pay, p.id, p.dir[ 0 ], p.dir[ 1 ], p.off );
            put( 1, pbx, pby, bid,  bdx, bdy, bo );
        }
        nb = new_nb;

        if ( unbounded ) {
            unbounded = false;
            for ( int i = 0; i < nb; ++i )
                unbounded |= cid[ i ] == cell_ids::INFINITE;
            if ( ! unbounded )
                has_planes = false;                      // plus rien a repousser : on ne les tient plus
        }
        return CutStatus::CUT;
    }

    // ---- le simplexe de remplacement -----------------------------------------------------------

    /// La VITESSE du sommet `i` quand on repousse les parois `INFINITE` de `g` : il est le coin de
    /// ses deux coupes, donc il resout le meme 2x2 avec les indicatrices `INFINITE` en second
    /// membre. Nulle pour un sommet reel, qui ne bouge pas.
    void growth_rate( int i, TK &rx, TK &ry ) const {
        const int c0 = vertex_cut( i, 0 ), c1 = vertex_cut( i, 1 );
        const TK f0 = cid[ c0 ] == cell_ids::INFINITE, f1 = cid[ c1 ] == cell_ids::INFINITE;
        if ( f0 == 0 && f1 == 0 ) { rx = 0; ry = 0; return; }
        const TK a1 = pdx[ c0 ], b1 = pdy[ c0 ], a2 = pdx[ c1 ], b2 = pdy[ c1 ];
        const TK det = a1 * b2 - b1 * a2;
        if ( det == 0 ) { rx = 0; ry = 0; return; }
        rx = ( f0 * b2 - b1 * f1 ) / det;
        ry = ( a1 * f1 - f0 * a2 ) / det;
    }

    /// REPOUSSE les parois `INFINITE` jusqu'a ce que le classement des sommets par `p` soit celui
    /// qu'il a a l'infini. Chaque sommet voyage en ligne droite, donc sa distance signee au plan est
    /// AFFINE en la poussee et « quand changerait-il de cote ? » est une division ; on pousse
    /// au-dela du plus lointain, et un peu plus, pour qu'aucun sommet ne reste SUR le plan.
    void grow_for( const PlaneT &p ) {
        if ( ! has_planes )
            planes_from_vertices();

        static constexpr int max_rounds = 4;   // 2 suffisent en arithmetique exacte
        // LA MARGE N'EST PAS UN EPSILON MACHINE : un sommet repousse d'un epsilon retombe a la
        // precision du produit scalaire, et deux sommets confondus a 1e-15 se classent alors
        // chacun de son cote du plan -- ce qui casse la combinatoire de la coupe ( vu en 4D ).
        // Elle est petite devant la geometrie, grande devant l'arrondi : ou se posent les sommets
        // FACTICES n'a de toute facon aucun sens geometrique.
        const TK margin = std::is_same_v<TK,float> ? TK( 1e-5 ) : TK( 1e-6 );

        for ( int i = 0; i < nb; ++i )
            growth_rate( i, rx[ i ], ry[ i ] );

        // « la vitesse ne fait pas varier la distance au plan » se juge A UNE TOLERANCE, pas a zero :
        // les plans sont RELUS sur la geometrie, donc une normale exactement axiale sort avec des
        // composantes a 1e-17, et `root = - s / rate` ferait alors une poussee de 1e17. Un rayon
        // parallele au plan a 1e-9 pres l'est.
        const TK tol = TK( 1e-9 ) * std::sqrt( p.dir[ 0 ] * p.dir[ 0 ] + p.dir[ 1 ] * p.dir[ 1 ] );

        TK g = 0;
        for ( int round = 0; round < max_rounds; ++round ) {
            bool push = false;
            TK grow = 0;
            for ( int i = 0; i < nb; ++i ) {
                const TK rate = p.dir[ 0 ] * rx[ i ] + p.dir[ 1 ] * ry[ i ];
                const TK nr = std::sqrt( rx[ i ] * rx[ i ] + ry[ i ] * ry[ i ] );
                if ( nr == 0 || ( rate < 0 ? - rate : rate ) <= tol * nr )
                    continue;
                const TK s = p.dir[ 0 ] * ( vx[ i ] + g * rx[ i ] ) + p.dir[ 1 ] * ( vy[ i ] + g * ry[ i ] ) - p.off;
                // `>= 0`, pas `> 0` : un sommet exactement SUR le plan n'est pas encore du cote ou
                // il serait a l'infini -- c'est precisement un cas a pousser.
                const TK root = - s / rate;
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

        for ( int i = 0; i < nb; ++i ) {
            if ( cid[ i ] == cell_ids::INFINITE )
                po[ i ] += g;
            vx[ i ] += g * rx[ i ];
            vy[ i ] += g * ry[ i ];
        }
    }

    // ---- la mesure, dans le flottant `TF` de l'appelant ----------------------------------------

    template<class TF>
    TF measure() const {
        if ( unbounded )
            return std::numeric_limits<TF>::max();
        TF sum = 0;
        for ( int i = 0; i < nb; ++i ) {
            const int j = i + 1 < nb ? i + 1 : 0;
            sum += TF( vx[ i ] ) * TF( vy[ j ] ) - TF( vx[ j ] ) * TF( vy[ i ] );
        }
        return sum / 2;
    }

    /// l'adjoint du lacet : `grad_vp( i, d )` recoit la cotangente du sommet `i` ( ECRITE, pas
    /// accumulee ). Rien pour une cellule non bornee, dont la mesure est une constante.
    template<class TF>
    void measure_bwd( TF grad_res, auto &&grad_vp ) const {
        if ( unbounded )
            return;
        for ( int i = 0; i < nb; ++i ) {
            const int p = i ? i - 1 : nb - 1;
            const int n = i + 1 < nb ? i + 1 : 0;
            grad_vp( i, 0 ) = grad_res * ( TF( vy[ n ] ) - TF( vy[ p ] ) ) / 2;
            grad_vp( i, 1 ) = grad_res * ( TF( vx[ p ] ) - TF( vx[ n ] ) ) / 2;
        }
    }

    /// un eventail depuis le sommet 0 : les triangles `( 0, i, i+1 )` PAVENT le convexe.
    /// `func( chain )` recoit trois indices de sommets.
    /// `func( c, mesure )` pour chaque coupe `c` qui porte une arete : sa LONGUEUR ( la coupe `i`
    /// porte l'arete `[ v_i, v_i+1 ]` ) -- ce que la hessienne d'un transport lit ( `diagram::hessian_row` )
    template<class TF>
    void for_each_facet( auto &&func ) const {
        for ( int i = 0; i < nb; ++i ) {
            const int j = i + 1 < nb ? i + 1 : 0;
            const TF dx = TF( vx[ j ] ) - TF( vx[ i ] ), dy = TF( vy[ j ] ) - TF( vy[ i ] );
            func( i, sycl::sqrt( dx * dx + dy * dy ) );
        }
    }

    void for_each_simplex( auto &&func ) const {
        Vector<SI,3> chain;
        chain[ 0 ] = 0;
        for ( int i = 1; i + 1 < nb; ++i ) {
            chain[ 1 ] = i;
            chain[ 2 ] = i + 1;
            func( chain );
        }
    }

    void bbox( TK *lo, TK *hi ) const {
        lo[ 0 ] = hi[ 0 ] = nb ? vx[ 0 ] : TK( 0 );
        lo[ 1 ] = hi[ 1 ] = nb ? vy[ 0 ] : TK( 0 );
        for ( int i = 1; i < nb; ++i ) {
            lo[ 0 ] = vx[ i ] < lo[ 0 ] ? vx[ i ] : lo[ 0 ]; hi[ 0 ] = vx[ i ] > hi[ 0 ] ? vx[ i ] : hi[ 0 ];
            lo[ 1 ] = vy[ i ] < lo[ 1 ] ? vy[ i ] : lo[ 1 ]; hi[ 1 ] = vy[ i ] > hi[ 1 ] ? vy[ i ] : hi[ 1 ];
        }
    }

    // ---- les tenseurs ( voir `Cell.py` pour leur forme ) ----------------------------------------

    /// depuis une vue `Cell_2` ( un item deja indexe ) : `vertex_positions [ nv, 2 ]`, `cut_ids`.
    /// `false` si la cellule ne tient pas dans `cap`.
    bool load( const auto &c ) {
        const int n = int( SI( c.nb_vertices ) );
        if ( n > cap )
            return false;
        nb = n;
        unbounded = false;
        for ( int i = 0; i < nb; ++i ) {
            vx [ i ] = TK( c.vertex_positions( i, 0 ) );
            vy [ i ] = TK( c.vertex_positions( i, 1 ) );
            cid[ i ] = int( SI( c.cut_ids( i ) ) );
            unbounded |= cid[ i ] == cell_ids::INFINITE;
        }
        has_planes = false;
        return true;
    }

    /// vers une vue `Cell_2`. Rend `false` si elle est trop petite : le compte voulu est alors
    /// enregistre ( `ShapeVarView::set` ) et RIEN n'est ecrit -- l'hote reserve plus et relance.
    bool store( auto &&c ) const {
        if ( ! c.nb_vertices.set( nb ) )
            return false;
        c.nb_cuts.set( nb );
        for ( int i = 0; i < nb; ++i ) {
            c.vertex_positions( i, 0 ) = vx[ i ];
            c.vertex_positions( i, 1 ) = vy[ i ];
            c.cut_ids( i ) = cid[ i ];
        }
        return true;
    }
};

} // namespace sdot
