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

/// `MEMO` : LA MEMOIRE ( `main_memo.cpp`, la borne superieure de `--memo` en 3D ), sous deux formes.
///   A. `pre[ 0 .. npre )` : les RANGS des voisins de la cellule finale, connus d'une passe
///      precedente, proposes d'abord ; puis le parcours ordinaire, en SAUTANT les rangs marques
///      dans `saute` ( le complement : aucun dirac deux fois, `cut` n'est pas idempotente ).
///   B. `fbeg / fmask [ 0 .. nf )` : les FEUILLES entrees a la passe precedente ( par le rang de
///      leur premier germe, triees ), et pour chacune un bit par germe disant « voisin final ».
///      Les bits a 1 sont proposes d'abord, sans aucun test de boite ; puis le parcours : une
///      feuille memorisee propose ses bits a 0 ( le complement ), un noeud qui contient une
///      feuille memorisee est descendu ; `tester = false` leur epargne le test d'eviction ( on
///      sait qu'on y entre ), `true` le garde ( la cellule finale peut les rejeter maintenant ).
///   C. `front [ 0 .. nfront )` : LA FRONTIERE du parcours d'hier -- les feuilles entrees puis les
///      noeuds REJETES ( dont on n'est pas descendu ), en indices de noeuds -- rejouee SANS PILE :
///      apres les bits a 1 ( forme B ), chaque noeud de la frontiere est teste et, s'il passe,
///      parcouru normalement ( une feuille memorisee propose son complement ) ; l'arbre n'est
///      redescendu que sous un rejete d'hier qui passe aujourd'hui. Exact : tout noeud est soit
///      ancetre d'une feuille entree, soit entre, soit rejete, soit sous un rejete.
/// Compile a part pour que le chemin sans memoire ne porte ni pointeur ni compteur de plus.
template<class TK, bool POIDS, int W = 8, bool MEMO = false>
struct FournisseurBsp3 {
    using Arbre = AaBspT<3>;

    struct Local {
        int  pile[ 48 ];
        int  haut = 0;
        int  k = 0, fin = 0;
        bool amorce = false;
        int  ipre = 0;                                   ///< MEMO A : ou on en est dans `pre`
        int  fa = 0; unsigned long long bits = 0; bool fa_ouverte = false;   ///< MEMO B : la feuille en cours de la pre-passe
        unsigned long long saut = 0; int kbeg = 0;      ///< MEMO B : au parcours, les bits deja proposes de la feuille ouverte
        int  nb_prop = 0, nb_boites = 0, nb_coupees = 0; ///< MEMO : plans proposes, boites testees, coupes EFFECTIVES
        int  entrees[ 64 ]; int nentrees = 0;            ///< MEMO : les feuilles entrees ( indices de noeuds ), pour batir la memoire
        int  rejets[ 256 ]; int nrejets = 0;             ///< MEMO : les noeuds rejetes par l'elagage ( indices ), idem
        int  ifront = 0;                                 ///< MEMO C : ou on en est de la frontiere
    };

    const Arbre *arbre;
    TK   x0, y0, z0, w0;
    SI32 i0;
    const SI32          *pre = nullptr;                  ///< MEMO A : les rangs a proposer d'abord
    int                  npre = 0;
    const unsigned char *saute = nullptr;                ///< MEMO A : par rang, 1 = deja propose
    const SI32          *fbeg = nullptr;                 ///< MEMO B : les feuilles memorisees ( rang du premier germe, triees )
    const unsigned long long *fmask = nullptr;           ///< MEMO B : leurs bits « voisin final »
    int                  nf = 0;
    int                  tester = 1;                     ///< MEMO B : 0 = ne pas tester les boites qu'on sait entrees, 1 = les tester toutes, 2 = tester les feuilles seules ( les noeuds internes connus sont descendus sans test )
    bool                 parcours = true;                ///< MEMO : `false` = les souvenirs seuls, sans parcours ( le plancher )
    const SI32          *front = nullptr;                ///< MEMO C : la frontiere d'hier ( indices de noeuds )
    int                  nfront = 0;

    /// MEMO B : l'indice de la premiere feuille memorisee de rang >= `r`
    int feuille_des( int r ) const {
        int lo = 0, hi = nf;
        while ( lo < hi ) { const int m = ( lo + hi ) / 2; if ( fbeg[ m ] < r ) lo = m + 1; else hi = m; }
        return lo;
    }

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
        if ( ! l.amorce ) {                              // la racine -- sauf avec une frontiere ( C ) : elle la remplace
            if ( ! ( MEMO && nfront > 0 ) ) l.pile[ l.haut++ ] = 0;
            l.amorce = true;
        }

        if constexpr ( MEMO ) {                          // la pre-passe : les voisins d'hier
            if ( l.ipre < npre ) {                       // A. par rang
                const int k = pre[ l.ipre++ ];
                ++l.nb_prop;
                bissect3<POIDS>( TK( arbre->seed_c( k, 0 ) ), TK( arbre->seed_c( k, 1 ) ),
                                 TK( arbre->seed_c( k, 2 ) ), POIDS ? TK( arbre->seed_w( k ) ) : TK( 0 ),
                                 x0, y0, z0, w0, SI32( arbre->order[ k ] ), p );
                return true;
            }
            while ( l.fa < nf ) {                        // B. par feuille, les bits a 1
                if ( ! l.fa_ouverte ) { l.bits = fmask[ l.fa ]; l.fa_ouverte = true; }
                if ( ! l.bits ) { ++l.fa; l.fa_ouverte = false; continue; }
                const int b = __builtin_ctzll( l.bits );
                l.bits &= l.bits - 1;
                const int k = fbeg[ l.fa ] + b;
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
                if constexpr ( MEMO ) {
                    if ( saute && saute[ k ] ) continue;
                    if ( ( l.saut >> ( k - l.kbeg ) ) & 1 ) continue;
                    ++l.nb_prop;
                }
                bissect3<POIDS>( TK( arbre->seed_c( k, 0 ) ), TK( arbre->seed_c( k, 1 ) ),
                                 TK( arbre->seed_c( k, 2 ) ), POIDS ? TK( arbre->seed_w( k ) ) : TK( 0 ),
                                 x0, y0, z0, w0, id, p );
                return true;
            }

            if ( l.haut == 0 ) {
                if constexpr ( MEMO ) {                  // C : le noeud suivant de la frontiere
                    if ( l.ifront < nfront ) { l.pile[ l.haut++ ] = front[ l.ifront++ ]; continue; }
                }
                return false;
            }

            const int h = l.pile[ --l.haut ];
            const auto &nd = arbre->nodes[ h ];
            if constexpr ( MEMO ) {
                // B : ce noeud contient-il une feuille memorisee ? ( est-ce cette feuille ? )
                int fi = -1; bool connu = false;
                if ( nf && ( tester != 1 || nd.right < 0 ) ) {   // en mode 1 seules les feuilles ont besoin de leur masque
                    const int q = feuille_des( int( nd.beg ) );
                    connu = q < nf && fbeg[ q ] < int( nd.end );
                    if ( connu && nd.right < 0 ) fi = q;
                }
                if ( ! connu || tester == 1 || ( tester == 2 && nd.right < 0 ) ) {
                    ++l.nb_boites;
                    if ( ! peut_couper( nd, e ) ) {
                        if ( l.nrejets < 256 ) l.rejets[ l.nrejets++ ] = h;
                        continue;
                    }
                }
                if ( nd.right < 0 ) {
                    l.k = int( nd.beg ); l.fin = int( nd.end ); l.kbeg = int( nd.beg );
                    l.saut = fi >= 0 ? fmask[ fi ] : 0;
                    if ( l.nentrees < 64 ) l.entrees[ l.nentrees++ ] = h;
                    continue;
                }
            } else {
                if ( ! peut_couper( nd, e ) )
                    continue;
                if ( nd.right < 0 ) { l.k = int( nd.beg ); l.fin = int( nd.end ); continue; }
            }

            const int g = h + 1, dr = int( nd.right );
            if ( proximite( g ) <= proximite( dr ) ) { l.pile[ l.haut++ ] = dr; l.pile[ l.haut++ ] = g; }
            else                                       { l.pile[ l.haut++ ] = g;  l.pile[ l.haut++ ] = dr; }
        }
    }
};

} // namespace sf::d3
