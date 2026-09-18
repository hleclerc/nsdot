#pragma once

// =====================================================================================
// LE BSP RETOURNE EN FOURNISSEUR, 2D.
//
// Le parcours de l'arbre ne POUSSE plus ses candidats vers la cellule : il est SUSPENDU. Sa pile
// explicite vit dans le `Local` que le moteur loge pour lui, et chaque `suivant` reprend ou le
// precedent s'etait arrete. Profondeur d'abord, fils le plus proche du germe en premier -- les
// coupes qui mordent le plus arrivent tot, la cellule retrecit vite, et l'elagage mord plus.
//
// Le test d'elagage est fait a la SORTIE de la pile, contre la cellule TELLE QU'ELLE EST : chaque
// coupe faite depuis l'empilement rend le « non » plus probable.
//
// `POIDS` est une constante de compilation : le cas euclidien ne paie ni les pentes du majorant
// ni les termes en `a . y`.
// =====================================================================================

#include "accel/AaBsp.h"
#include "cell/Contrat2D.h"
#include "cell/Elagage2D.h"

namespace sf::d2 {

template<class TK, bool POIDS>
struct FournisseurBsp {
    using Arbre = AaBspT<2>;

    /// L'ETAT DU PARCOURS, un par cellule. La pile est bornee par la PROFONDEUR de l'arbre : a
    /// chaque niveau on depile un noeud et on en empile deux. 48 niveaux valent 2^48 germes.
    struct Local {
        int  pile[ 48 ];
        int  haut = 0;
        int  k = 0, fin = 0;                             ///< la tranche de la feuille ouverte
        bool amorce = false;
    };

    const Arbre *arbre;
    TK   x0, y0, w0;                                     ///< le germe courant
    SI32 i0;                                             ///< son identifiant : on ne se coupe pas

    FournisseurBsp( const Arbre *arbre, TK x0, TK y0, TK w0, SI32 i0 )
        : arbre( arbre ), x0( x0 ), y0( y0 ), w0( w0 ), i0( i0 ) {}

    template<class Etat>
    bool peut_couper( const typename Arbre::Node &nd, const Etat &e ) const {
        Boite2<TK> B;
        for ( int d = 0; d < 2; ++d ) { B.lo[ d ] = TK( nd.lo[ d ] ); B.hi[ d ] = TK( nd.hi[ d ] ); }
        if constexpr ( POIDS ) {
            B.a[ 0 ] = TK( nd.wm.a[ 0 ] ); B.a[ 1 ] = TK( nd.wm.a[ 1 ] ); B.b = TK( nd.wm.b );
        }
        return peut_couper_boite<POIDS>( e, x0, y0, w0, B );
    }

    /// le carre de la distance du germe a la boite du noeud : une clef d'ordre, pas un test.
    TK proximite( int n ) const {
        const auto &nd = arbre->nodes[ n ];
        TK s = 0;
        for ( int d = 0; d < 2; ++d ) {
            const TK x = d ? y0 : x0;
            const TK lo = TK( nd.lo[ d ] ), hi = TK( nd.hi[ d ] );
            const TK e = x < lo ? lo - x : x > hi ? x - hi : TK( 0 );
            s += e * e;
        }
        return s;
    }

    template<class Etat>
    bool suivant( const Etat &e, Local &l, Plan2<TK> &p ) {
        if ( ! l.amorce ) { l.pile[ l.haut++ ] = 0; l.amorce = true; }

        for ( ;; ) {
            // ---- une feuille est ouverte : le germe suivant de sa tranche
            while ( l.k < l.fin ) {
                const int k = l.k++;
                const SI32 id = SI32( arbre->order[ k ] );
                if ( id == i0 ) continue;
                const TK xj = TK( arbre->seed_c( k, 0 ) ), yj = TK( arbre->seed_c( k, 1 ) );
                p.dx  = xj - x0;
                p.dy  = yj - y0;
                p.off = TK( 0.5 ) * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) );
                if constexpr ( POIDS ) p.off += TK( 0.5 ) * ( w0 - TK( arbre->seed_w( k ) ) );
                p.id  = id;
                return true;
            }

            if ( l.haut == 0 )
                return false;                            // l'arbre est epuise

            const int h = l.pile[ --l.haut ];
            const auto &nd = arbre->nodes[ h ];
            if ( ! peut_couper( nd, e ) )                // aucun germe de ce sous-arbre ne peut rien
                continue;
            if ( nd.right < 0 ) { l.k = int( nd.beg ); l.fin = int( nd.end ); continue; }

            const int g = h + 1, dr = int( nd.right );   // PREORDRE : le gauche est juste a cote
            if ( proximite( g ) <= proximite( dr ) ) { l.pile[ l.haut++ ] = dr; l.pile[ l.haut++ ] = g; }
            else                                       { l.pile[ l.haut++ ] = g;  l.pile[ l.haut++ ] = dr; }
        }
    }
};

} // namespace sf::d2
