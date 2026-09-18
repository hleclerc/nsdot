#pragma once

// =====================================================================================
// LE FOURNISSEUR « A ALPHA » : la cellule de `i` pour les poids `w + alpha d`, SANS rafraichir
// l'arbre, et avec un DEPART A CHAUD.
//
// = Le majorant qui ne se rafraichit pas
//
// Le majorant affine est lineaire en les poids : si un noeud porte `w( y ) <= a_w . y + b_w` et
// `d( y ) <= a_d . y + b_d`, alors pour tout `alpha >= 0`
//
//     w( y ) + alpha d( y ) <= ( a_w + alpha a_d ) . y + ( b_w + alpha b_d ).
//
// C'est exact, donc l'elagage reste exact, et `alpha` peut CHANGER D'UNE CELLULE A L'AUTRE. Le
// majorant de `d` est calcule une fois par direction, en dehors de l'arbre ( `dm`, un par noeud ).
//
// = Le depart a chaud
//
// Avant le parcours, on propose les plans d'une liste de germes donnee -- les voisins de la cellule
// en `alpha = 0`, typiquement. Six coupes sans parcours, et la cellule est deja petite quand le
// parcours commence : l'elagage, exact, ecarte presque tout. Le resultat est le meme que sans
// depart a chaud ( l'elagage ne rejette qu'un noeud qui ne peut plus rien retrancher a la cellule
// COURANTE, sur-ensemble de la cellule finale ), en moins cher. Un germe deja propose a chaud est
// SAUTE par le parcours : le reproposer ne couperait rien en arithmetique exacte, mais par arrondi
// il peut retrancher une lamelle d'aire nulle, et la cellule aurait alors deux aretes consecutives
// de meme normale -- une combinatoire fausse.
// =====================================================================================

#include "accel/AaBsp.h"
#include "cell/Contrat2D.h"
#include "cell/Elagage2D.h"

namespace sf::d2 {

template<class TK>
struct FournisseurAlpha {
    using Arbre = AaBspT<2>;

    struct Local {
        int  pile[ 48 ];
        int  haut = 0;
        int  k = 0, fin = 0;
        int  chaud = 0;                                  ///< le prochain plan de la liste chaude
        bool amorce = false;
    };

    const Arbre    *arbre;
    const WMajT<2> *dm;          ///< le majorant de `d`, par noeud
    const TF       *dt;          ///< `d` dans l'ordre de l'arbre
    const TF *const *P;          ///< les positions, par identifiant ( pour les plans chauds )
    const TF       *w, *d;       ///< les poids et la direction, par identifiant
    TF              alpha;
    TK              x0, y0, w0;  ///< le germe courant, `w0 = w_i + alpha d_i`
    SI32            i0;
    const SI32     *chauds;      ///< les germes a proposer d'abord ( `< 0` : ignore )
    int             nb_chauds;

    FournisseurAlpha( const Arbre *arbre, const WMajT<2> *dm, const TF *dt, const TF *const *P,
                      const TF *w, const TF *d, TF alpha, SI32 i0, const SI32 *chauds, int nb_chauds )
        : arbre( arbre ), dm( dm ), dt( dt ), P( P ), w( w ), d( d ), alpha( alpha ),
          x0( TK( P[ 0 ][ i0 ] ) ), y0( TK( P[ 1 ][ i0 ] ) ), w0( TK( w[ i0 ] + alpha * d[ i0 ] ) ),
          i0( i0 ), chauds( chauds ), nb_chauds( nb_chauds ) {}

    /// le plan de `j`, aux poids `w + alpha d`
    void plan( SI32 j, TK xj, TK yj, TK wj, Plan2<TK> &p ) const {
        p.dx  = xj - x0;
        p.dy  = yj - y0;
        p.off = TK( 0.5 ) * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) + ( w0 - wj ) );
        p.id  = j;
    }

    template<class Etat>
    bool peut_couper( int n, const Etat &e ) const {
        const auto &nd = arbre->nodes[ n ];
        Boite2<TK> B;
        for ( int k = 0; k < 2; ++k ) {
            B.lo[ k ] = TK( nd.lo[ k ] ); B.hi[ k ] = TK( nd.hi[ k ] );
            B.a[ k ] = TK( nd.wm.a[ k ] + alpha * dm[ n ].a[ k ] );
        }
        B.b = TK( nd.wm.b + alpha * dm[ n ].b );
        return peut_couper_boite<true>( e, x0, y0, w0, B );
    }

    TK proximite( int n ) const {
        const auto &nd = arbre->nodes[ n ];
        TK s = 0;
        for ( int k = 0; k < 2; ++k ) {
            const TK x = k ? y0 : x0;
            const TK lo = TK( nd.lo[ k ] ), hi = TK( nd.hi[ k ] );
            const TK e = x < lo ? lo - x : x > hi ? x - hi : TK( 0 );
            s += e * e;
        }
        return s;
    }

    bool deja_chaud( SI32 j ) const {
        for ( int c = 0; c < nb_chauds; ++c )
            if ( chauds[ c ] == j ) return true;
        return false;
    }

    template<class Etat>
    bool suivant( const Etat &e, Local &l, Plan2<TK> &p ) {
        while ( l.chaud < nb_chauds ) {                  // ---- le depart a chaud
            const SI32 j = chauds[ l.chaud++ ];
            if ( j < 0 || j == i0 ) continue;
            plan( j, TK( P[ 0 ][ j ] ), TK( P[ 1 ][ j ] ), TK( w[ j ] + alpha * d[ j ] ), p );
            return true;
        }
        if ( ! l.amorce ) { l.pile[ l.haut++ ] = 0; l.amorce = true; }

        for ( ;; ) {
            while ( l.k < l.fin ) {
                const int k = l.k++;
                const SI32 id = SI32( arbre->order[ k ] );
                if ( id == i0 || deja_chaud( id ) ) continue;
                plan( id, TK( arbre->seed_c( k, 0 ) ), TK( arbre->seed_c( k, 1 ) ),
                      TK( arbre->seed_w( k ) + alpha * dt[ k ] ), p );
                return true;
            }
            if ( l.haut == 0 )
                return false;
            const int h = l.pile[ --l.haut ];
            const auto &nd = arbre->nodes[ h ];
            if ( ! peut_couper( h, e ) )
                continue;
            if ( nd.right < 0 ) { l.k = int( nd.beg ); l.fin = int( nd.end ); continue; }
            const int g = h + 1, dr = int( nd.right );
            if ( proximite( g ) <= proximite( dr ) ) { l.pile[ l.haut++ ] = dr; l.pile[ l.haut++ ] = g; }
            else                                       { l.pile[ l.haut++ ] = g;  l.pile[ l.haut++ ] = dr; }
        }
    }
};

} // namespace sf::d2
