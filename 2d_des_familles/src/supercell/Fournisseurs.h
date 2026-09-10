#pragma once

// =====================================================================================
// LES FOURNISSEURS -- « quel demi-plan couper maintenant ? »
//
// Le noyau ne connait aucune structure d'acceleration, et depuis qu'il recoit des PLANS il ne
// connait meme plus les diracs. Un fournisseur est un objet avec une seule methode,
//
//      template<class Etat, class Local> bool suivant( const Etat &e, Local &l, Plan &p );
//
// qui remplit `p` et rend `true`, ou rend `false` quand il n'a plus rien.
//
// `l` EST A LUI. S'il declare un type `Local`, le moteur en cree un par CELLULE dans sa propre
// frame et le lui repasse a chaque appel ; le noyau ne le lit jamais. Un fournisseur sans etat
// n'en declare pas, et ne paie rien. C'est la que vont un compteur de rejets, une boule, une
// boite -- tout ce qui doit survivre d'une coupe a l'autre sans etre l'affaire du noyau.
//
// Et s'il declare `veut_changement = true`, `Etat::change` lui dit si la cellule a bouge depuis
// l'appel precedent. C'est ce qui permet de compter les rejets CONSECUTIFS, donc de decider d'une
// transition, sans rien relire. Il est appele APRES
// chaque coupe et voit la cellule telle qu'elle est devenue -- c'est tout l'interet : une
// politique qui recoit `e` peut decider en fonction de ce que la cellule EST, alors qu'une
// liste construite d'avance doit decider avant qu'elle existe.
//
// CE QUI A DEMENAGE ICI. La bissectrice de deux germes est une facon parmi d'autres de
// fabriquer un demi-plan ; elle vit desormais de ce cote de la porte, avec les positions. Un
// diagramme de LAGUERRE ( des poids dans le produit scalaire ) s'ecrit en changeant `bissect`
// et rien d'autre -- le noyau n'a pas a le savoir.
//
// `Etat` est un patron, pas un type : le meme fournisseur sert le chemin SIMD ( ou `e.nb` est
// une constante de compilation ), l'excursion et les chemins scalaires ( ou c'est un entier ).
// Il n'y a donc qu'une seule version de la politique a ecrire.
// =====================================================================================

#include "supercell/Contrat2D.h"

#include <immintrin.h>

namespace noyau2d {

/// LA BISSECTRICE de [ ( x0, y0 ), ( xj, yj ) ], sous la forme `d . x <= off`. C'est ce que le
/// noyau faisait avant, et qui n'avait rien a y faire.
inline void bissect( float xj, float yj, float x0, float y0, int id, Plan &p ) {
    p.dx  = xj - x0;
    p.dy  = yj - y0;
    p.off = 0.5f * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) );
    p.id  = id;
}

/// LE FOURNISSEUR NUL : tous les autres diracs, dans l'ordre, sans jamais regarder la cellule.
/// C'est le « tous contre tous » -- aucune acceleration, aucun tri, aucun elagage. Il ne sert
/// pas a aller vite : il sert de PLANCHER, et il isole le cout du moteur de celui de la
/// politique.
struct TousLesAutres {
    const float *px, *py;
    float x0, y0;
    int   n, moi, k;

    TousLesAutres( const float *px, const float *py, int n, int moi )
        : px( px ), py( py ), x0( px[ moi ] ), y0( py[ moi ] ), n( n ), moi( moi ), k( 0 ) {}

    template<class Etat>
    bool suivant( const Etat &, RienDeLocal &, Plan &p ) {
        if ( k == moi ) ++k;                             // on ne se coupe pas soi-meme
        if ( k >= n ) return false;
        const int j = k++;
        bissect( px[ j ], py[ j ], x0, y0, j, p );
        return true;
    }
};

/// EN S'ELOIGNANT DE SOI. Meme travail que `TousLesAutres` -- tous les autres diracs, aucun
/// elagage -- mais dans un ORDRE different : i+1, i-1, i+2, i-2, ... en indice.
///
/// Seul, il ne vaut rien de plus. Couple a un nuage trie en MORTON, il devient « du plus proche
/// au plus lointain », approximativement : c'est toute la raison d'etre du tri. Deux voisins en
/// indice Morton sont voisins dans l'espace, donc partir de `moi` et s'ecarter propose d'abord
/// les coupes qui mordent FORT, et la cellule s'effondre en quelques coupes.
///
/// A l'inverse, `TousLesAutres` sur un nuage Morton est le PIRE ordre possible : il part du coin
/// bas-gauche du domaine quel que soit `moi`, donc commence par les coupes les plus lointaines
/// et les plus faibles. Le tri ne suffit pas -- il faut que le parcours s'en serve.
///
/// = DEUX CURSEURS, ET PAS UNE BOUCLE DE REJET
///
/// La forme evidente tient en trois lignes : un ecart `d` qui monte, l'indice `moi ± d/2`, et on
/// RECOMMENCE quand il sort des bornes. Elle coute le double du reste. La raison est
/// arithmetique : pour rendre `n-1` indices valides il faut monter `d` jusqu'a `2n`, donc environ
/// deux tours par indice rendu -- et pour un `moi` proche d'un bout de l'ordre Morton, tout un
/// cote est rejete, donc c'est deux tours pour tous. Chaque tour porte en prime un test de bornes
/// que le predicteur ne voit pas venir.
///
/// MESURE ( n = 5000, nuage Morton, chemin SIMD, ns par coupe proposee ) :
///
///     suite deja calculee, lue en memoire   1.93     <- le plancher
///     deux curseurs                         2.14     +11 %
///     boucle de rejet                       4.53     +135 %
///
/// Un vrai MODULO ( `( moi + d ) % n` ) est encore pire : a suite identique il ajoute ~3.1 ns
/// par coupe, quelle que soit la taille -- c'est une division entiere par une valeur que le
/// compilateur ne connait pas. La soustraction conditionnelle l'evite, mais elle enroule au lieu
/// de s'arreter aux bords, et l'ordre enroule est spatialement moins bon : a n = 5000 il fait
/// passer les echappements de 681 a 3699. Ni modulo, ni enroulement : deux curseurs.
struct AutourDeMoi {
    const float *px, *py;
    float x0, y0;
    int   lo, hi, n;            ///< les deux curseurs, et la borne
    bool  cote;                 ///< a qui le tour

    AutourDeMoi( const float *px, const float *py, int n, int moi )
        : px( px ), py( py ), x0( px[ moi ] ), y0( py[ moi ] ),
          lo( moi - 1 ), hi( moi + 1 ), n( n ), cote( true ) {}

    template<class Etat>
    bool suivant( const Etat &, RienDeLocal &, Plan &p ) {
        int j;
        if      ( cote && hi < n ) { cote = false; j = hi++; }
        else if ( lo >= 0 )        { cote = true;  j = lo--; }
        else if ( hi < n )         {               j = hi++; }
        else return false;
        bissect( px[ j ], py[ j ], x0, y0, j, p );
        return true;
    }
};

/// LE FOURNISSEUR A LISTE : il deroule une liste construite d'avance, et ne regarde pas plus la
/// cellule que les precedents. C'est ce que faisait le noyau AVANT d'avoir un fournisseur, ecrit
/// comme un fournisseur -- de sorte que les bancs qui trient leurs opposants par distance
/// continuent de tourner sans que le noyau sache qu'une liste existe.
struct ListeFournie {
    const float *px, *py;
    float x0, y0;
    const int *opp;
    int nopp, k;

    ListeFournie( const float *px, const float *py, float x0, float y0, const int *opp, int nopp )
        : px( px ), py( py ), x0( x0 ), y0( y0 ), opp( opp ), nopp( nopp ), k( 0 ) {}

    template<class Etat>
    bool suivant( const Etat &, RienDeLocal &, Plan &p ) {
        if ( k >= nopp ) return false;
        const int j = opp[ k++ ];
        bissect( px[ j ], py[ j ], x0, y0, j, p );
        return true;
    }
};

/// DEUX PHASES, ET LA BOULE EST A LUI. C'est la demonstration de ce que `Local` et
/// `veut_changement` permettent, et le noyau n'en sait rien.
///
///   PHASE 1  on ne paie rien. On compte les coupes SANS EFFET consecutives -- `Etat::change`
///            les donne gratuitement, sans relire quoi que ce soit.
///   TRANSITION  au `SEUIL`-ieme rejet d'affilee, la cellule a probablement fini de bouger. On
///            calcule ALORS une boule, UNE SEULE FOIS, et jamais plus.
///   PHASE 2  la liste etant triee par distance et la boule ne pouvant que majorer une cellule
///            qui retrecit, le premier candidat a distance >= 2 R arrete TOUT.
///
/// POURQUOI LA BOULE N'EST PLUS JAMAIS REMISE A JOUR. La cellule ne fait que retrecir, donc un
/// `R` calcule plus tot la majore encore : une boule perimee reste EXACTE, seulement plus lache.
/// La resserrer a chaque coupe couterait plus que les quelques tentatives qu'elle economiserait.
///
/// LE SIMD EST A SA CHARGE. `Etat::vx / vy` sont des registres ; c'est ici qu'on en fait une
/// reduction horizontale, pas dans le noyau. Pendant une excursion l'etat est en memoire, d'ou
/// les deux branches -- et c'est bien le fournisseur qui choisit comment lire.
template<int SEUIL = 4>
struct ListeDeuxPhases {
    using Local = struct { int rates; float r2; bool phase2; };
    static constexpr bool veut_changement = true;

    const float *px, *py;
    float x0, y0;
    const int *opp;
    int nopp, k;

    ListeDeuxPhases( const float *px, const float *py, float x0, float y0,
                     const int *opp, int nopp )
        : px( px ), py( py ), x0( x0 ), y0( y0 ), opp( opp ), nopp( nopp ), k( 0 ) {}

    /// le carre du rayon autour du GERME -- ce que le noyau ne pourrait pas calculer, faute de
    /// connaitre le germe.
    template<class Etat>
    float rayon2( const Etat &e ) const {
        if constexpr ( requires { e.vx + e.vx; } ) {      // registres : reduction horizontale
            const __m256 ex = _mm256_sub_ps( e.vx, _mm256_set1_ps( x0 ) );
            const __m256 ey = _mm256_sub_ps( e.vy, _mm256_set1_ps( y0 ) );
            const __m256 d2 = _mm256_fmadd_ps( ex, ex, _mm256_mul_ps( ey, ey ) );
            const __m256 ok = _mm256_mask_blend_ps( (__mmask8)( ( 1u << Etat::nb ) - 1 ),
                                                    _mm256_set1_ps( -1.f ), d2 );
            __m128 q = _mm_max_ps( _mm256_castps256_ps128( ok ), _mm256_extractf128_ps( ok, 1 ) );
            q = _mm_max_ps( q, _mm_movehl_ps( q, q ) );
            q = _mm_max_ss( q, _mm_shuffle_ps( q, q, 1 ) );
            return _mm_cvtss_f32( q );
        } else {                                         // excursion : l'etat est en memoire
            float r2 = 0;
            for ( int i = 0; i < e.nb; ++i ) {
                const float ex = e.vx[ i ] - x0, ey = e.vy[ i ] - y0;
                const float d2 = ex * ex + ey * ey;
                r2 = d2 > r2 ? d2 : r2;
            }
            return r2;
        }
    }

    template<class Etat>
    bool suivant( const Etat &e, Local &l, Plan &p ) {
        if ( k >= nopp )
            return false;

        if ( ! l.phase2 ) {
            if ( e.change ) l.rates = 0;
            else if ( ++l.rates >= SEUIL ) { l.r2 = rayon2( e ); l.phase2 = true; }
        }

        const int j = opp[ k ];
        const float dx = px[ j ] - x0, dy = py[ j ] - y0;
        if ( l.phase2 && dx * dx + dy * dy >= 4 * l.r2 )
            return false;                                // lui et tous les suivants sont rejetes

        ++k;
        bissect( px[ j ], py[ j ], x0, y0, j, p );
        return true;
    }
};

} // namespace noyau2d
