#pragma once

#include "AaBsp.h"

namespace pd2d {

/// Le TAUX de sous-echantillonnage : un germe sur `pre_rate`. Reglage de banc, donc une variable
/// et non un parametre de template -- il ne sert qu'a la construction, jamais dans la marche.
inline SI pre_rate = 16;

/// Refaire EXPRES le recouvrement : le grand arbre reprend tous les germes, donc ceux de `S` sont
/// proposes deux fois. Sert uniquement a reproduire le defaut qu'il a cause, pour le dissequer.
inline bool pre_overlap = false;

/// LA PRE-PASSE PAR SOUS-ECHANTILLON : on coupe d'abord contre un seizieme des germes.
///
/// = Pourquoi c'est exact, et gratuit
///
/// `psi( x ) = min_i ( |x - p_i|^2 - w_i )` est un MINIMUM : retirer des germes ne peut que le
/// remonter, donc `psi_S >= psi` partout. La cellule de `k` calculee contre `S` seul a moins de
/// contraintes que la vraie, donc elle la CONTIENT -- pour tout `k`, qu'il soit dans `S` ou non.
/// Aucune correction, aucune marge : c'est le SENS de l'inegalite qui rend l'idee gratuite.
///
/// Et il n'y a meme pas de « pre-passe » a proprement parler dans le resultat : la cellule finale
/// est l'intersection de TOUS les demi-plans, donc couper d'abord par un sous-ensemble puis par le
/// reste donne exactement la meme cellule. Ce qui change n'est que l'ORDRE, et donc le RAYON dont
/// dispose le test d'eviction quand il ouvre sa premiere boite du grand arbre.
///
/// = Pourquoi c'est un accelerateur et pas une option de `PowerDiagram`
///
/// « proposer des germes » est deja toute l'interface. Un accelerateur qui propose d'abord le
/// sous-ensemble puis tout le reste est un accelerateur ordinaire : rien a changer ailleurs.
///
/// = Le sous-ensemble est STRATIFIE, et ce n'est pas un detail
///
/// `full.order` est deja un tri SPATIAL -- un sous-arbre du BSP y occupe une plage contigue --
/// donc un germe sur `pre_rate` de cette permutation, c'est un germe par sous-arbre a la
/// profondeur ou un sous-arbre porte `pre_rate` germes. C'est gratuit. MESURE EN 1D
/// (`cases/sub1d.py`) : contre un tirage aleatoire de meme taille, le stratifie gagne 1.6x sur la
/// mediane du rapport de largeur et 2 a 3x sur la queue.
///
/// = Le grand arbre porte le COMPLEMENT, et il le faut
///
/// Si les deux arbres se recouvraient, un germe de `S` serait propose DEUX FOIS, donc la cellule
/// serait coupee deux fois par le MEME plan. Le second passage trouve alors deux sommets a `s ~ 0`
/// -- ceux que le premier vient de poser sur ce plan -- et l'intersection `( s1 vx0 - s0 vx1 ) /
/// ( s1 - s0 )` y perd toute sa precision : denominateur minuscule, numerateur de deux termes qui
/// s'annulent. MESURE : un germe sur 100 000 (le nuage uniforme, graine 0, un germe sur 8) sortait
/// avec une aire fausse de 10 %, et la somme des aires a `1.000001886` -- c'est `--cross` qui l'a
/// nomme, pas la somme, parce qu'un seul germe sur cent mille s'y voit a peine.
///
/// Partitionner coute un troisieme arbre a la construction et rend le grand un seizieme plus
/// petit. C'est la seule forme ou aucun plan n'est applique deux fois.
///
/// `SubOnly` : ne faire QUE la pre-passe. La cellule rendue est alors l'ENCLOS et non la cellule --
/// c'est faux comme resultat, et c'est exactement ce que `--enclos` veut mesurer.
template<bool SubOnly>
struct AaBspPreT {
    AaBsp full;                 ///< tous les germes : l'ENUMERATION et l'ordre spatial
    AaBsp sub;                  ///< un sur `pre_rate`, avec les identites d'ORIGINE
    AaBsp rest;                 ///< tous les AUTRES -- la partition, pas un recouvrement

    static constexpr const char *name = SubOnly ? "sub" : "pre";

    TF seed_x( SI k ) const { return full.seed_x( k ); }
    TF seed_y( SI k ) const { return full.seed_y( k ); }
    TF seed_w( SI k ) const { return full.seed_w( k ); }
    SI seed_id( SI k ) const { return full.seed_id( k ); }
    SI nb_seeds() const { return full.nb_seeds(); }

    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        // `alive` : si la cellule s'est VIDEE pendant la pre-passe, la seconde n'a plus rien a
        // couper. `cut_with` le dit en rendant `false`, mais `for_each_candidate` ne rend rien --
        // on le retient donc au passage plutot que d'elargir le concept pour ce seul cas.
        const TF p0x = full.px[ k0 ], p0y = full.py[ k0 ];
        const SI i0 = full.order[ k0 ];

        bool alive = true;
        sub.for_each_candidate_at( p0x, p0y, i0, may_cut,
            [ & ]( TF x, TF y, TF w, SI id ) { return alive = cut_with( x, y, w, id ); },
            reach2 );

        if constexpr ( ! SubOnly )
            if ( alive )
                rest.for_each_candidate_at( p0x, p0y, i0, may_cut, cut_with, reach2 );
    }

    void build( const TF *X, const TF *Y, const TF *W, SI n, SI leaf ) {
        full.build( X, Y, W, n, leaf );

        const SI r = pre_rate < 1 ? SI( 1 ) : pre_rate;
        std::vector<SI> in, out;
        in.reserve( n / r + 1 );
        out.reserve( n );
        for ( SI k = 0; k < n; ++k )
            ( k % r ? out : in ).push_back( full.order[ k ] );
        sub .build_sel( X, Y, W, in .data(), SI( in .size() ), leaf );
        if ( pre_overlap )
            rest.build( X, Y, W, n, leaf );
        else
            rest.build_sel( X, Y, W, out.data(), SI( out.size() ), leaf );
    }
};

using AaBspPre = AaBspPreT<false>;
using AaBspSub = AaBspPreT<true>;

} // namespace pd2d
