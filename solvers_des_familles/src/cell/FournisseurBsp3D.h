#pragma once

// =====================================================================================
// LE BSP RETOURNE EN FOURNISSEUR, 3D. Le pendant exact de `FournisseurBsp2D.h` : meme parcours
// suspendu, meme preordre, meme ordre des fils ; seul le test d'elagage change de dimension.
//
// Exactitude mesuree dans le banc : somme des volumes a 1.3e-8, zero voisinage different du
// parcours sans elagage sur 3000 cellules ( `2d_des_familles/src/mains/main_bspf3d.cpp` ).
// =====================================================================================

#include "accel/AaBsp.h"
#include "cell/Contrat3D.h"
#include "cell/Elagage3D.h"

namespace sf::d3 {

/// LE PLAN BISSECTEUR de `[ p0, pj ]`, oriente pour que `p0` soit DEDANS :
///     |x - p0|^2 - w0 <= |x - pj|^2 - wj   <=>   d . x <= off ,
///     d = pj - p0 ,  off = d . ( pj + p0 ) / 2 + ( w0 - wj ) / 2
/// La forme `d . ( pj + p0 ) / 2` annule les termes dominants quand les deux germes sont proches.
template<bool POIDS, class TK>
inline void bissect3( TK xj, TK yj, TK zj, TK wj, TK x0, TK y0, TK z0, TK w0, SI32 id, Plan3<TK> &p ) {
    p.dx = xj - x0;
    p.dy = yj - y0;
    p.dz = zj - z0;
    p.off = TK( 0.5 ) * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) + p.dz * ( zj + z0 ) );
    if constexpr ( POIDS ) p.off += TK( 0.5 ) * ( w0 - wj );
    p.id = id;
}

/// `MEMO` : LA MEMOIRE ( `main_memo.cpp`, la borne superieure de `--memo` en 3D ). Le fournisseur
/// propose d'abord les rangs de `pre[ 0 .. npre )` -- les voisins de la cellule finale, connus
/// d'une passe precedente -- puis le parcours ordinaire, en SAUTANT les rangs marques dans
/// `saute` ( le complement : aucun dirac deux fois, `cut` n'est pas idempotente ). Compile a part
/// pour que le chemin sans memoire ne porte ni pointeur ni compteur de plus.
template<class TK, bool POIDS, int W = 8, bool MEMO = false>
struct FournisseurBsp3 {
    using Arbre = AaBspT<3>;

    struct Local {
        int  pile[ 48 ];
        int  haut = 0;
        int  k = 0, fin = 0;
        bool amorce = false;
        int  ipre = 0;                                   ///< MEMO : ou on en est dans `pre`
        int  nb_prop = 0, nb_boites = 0, nb_coupees = 0; ///< MEMO : plans proposes, boites testees, coupes EFFECTIVES
    };

    const Arbre *arbre;
    TK   x0, y0, z0, w0;
    SI32 i0;
    const SI32          *pre = nullptr;                  ///< MEMO : les rangs a proposer d'abord
    int                  npre = 0;
    const unsigned char *saute = nullptr;                ///< MEMO : par rang, 1 = deja propose
    bool                 parcours = true;                ///< MEMO : `false` = les souvenirs seuls, sans parcours ( le plancher )

    FournisseurBsp3( const Arbre *arbre, TK x0, TK y0, TK z0, TK w0, SI32 i0 )
        : arbre( arbre ), x0( x0 ), y0( y0 ), z0( z0 ), w0( w0 ), i0( i0 ) {}

    template<class Etat>
    bool peut_couper( const typename Arbre::Node &nd, const Etat &e ) const {
        Boite3<TK> B;
        for ( int d = 0; d < 3; ++d ) { B.lo[ d ] = TK( nd.lo[ d ] ); B.hi[ d ] = TK( nd.hi[ d ] ); }
        if constexpr ( POIDS ) {
            for ( int d = 0; d < 3; ++d ) B.a[ d ] = TK( nd.wm.a[ d ] );
            B.b = TK( nd.wm.b );
        }
        return peut_couper_boite3<POIDS,W>( e, x0, y0, z0, w0, B );
    }

    TK proximite( int n ) const {
        const auto &nd = arbre->nodes[ n ];
        const TK x[ 3 ] = { x0, y0, z0 };
        TK s = 0;
        for ( int d = 0; d < 3; ++d ) {
            const TK lo = TK( nd.lo[ d ] ), hi = TK( nd.hi[ d ] );
            const TK e = x[ d ] < lo ? lo - x[ d ] : x[ d ] > hi ? x[ d ] - hi : TK( 0 );
            s += e * e;
        }
        return s;
    }

    template<class Etat>
    bool suivant( const Etat &e, Local &l, Plan3<TK> &p ) {
        if ( ! l.amorce ) { l.pile[ l.haut++ ] = 0; l.amorce = true; }

        if constexpr ( MEMO ) {                          // la pre-passe : les voisins d'hier
            if ( l.ipre < npre ) {
                const int k = pre[ l.ipre++ ];
                ++l.nb_prop;
                bissect3<POIDS>( TK( arbre->seed_c( k, 0 ) ), TK( arbre->seed_c( k, 1 ) ),
                                 TK( arbre->seed_c( k, 2 ) ), POIDS ? TK( arbre->seed_w( k ) ) : TK( 0 ),
                                 x0, y0, z0, w0, SI32( arbre->order[ k ] ), p );
                return true;
            }
            if ( ! parcours ) return false;
        }

        for ( ;; ) {
            while ( l.k < l.fin ) {
                const int k = l.k++;
                const SI32 id = SI32( arbre->order[ k ] );
                if ( id == i0 ) continue;
                if constexpr ( MEMO ) { if ( saute && saute[ k ] ) continue; ++l.nb_prop; }
                bissect3<POIDS>( TK( arbre->seed_c( k, 0 ) ), TK( arbre->seed_c( k, 1 ) ),
                                 TK( arbre->seed_c( k, 2 ) ), POIDS ? TK( arbre->seed_w( k ) ) : TK( 0 ),
                                 x0, y0, z0, w0, id, p );
                return true;
            }

            if ( l.haut == 0 )
                return false;

            const int h = l.pile[ --l.haut ];
            const auto &nd = arbre->nodes[ h ];
            if constexpr ( MEMO ) ++l.nb_boites;
            if ( ! peut_couper( nd, e ) )
                continue;
            if ( nd.right < 0 ) { l.k = int( nd.beg ); l.fin = int( nd.end ); continue; }

            const int g = h + 1, dr = int( nd.right );
            if ( proximite( g ) <= proximite( dr ) ) { l.pile[ l.haut++ ] = dr; l.pile[ l.haut++ ] = g; }
            else                                       { l.pile[ l.haut++ ] = g;  l.pile[ l.haut++ ] = dr; }
        }
    }
};

} // namespace sf::d3
