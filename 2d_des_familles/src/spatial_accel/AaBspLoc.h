#pragma once

#include "spatial_accel/AaBsp.h"
#include <vector>

namespace pd {

/// Les germes vises par PAQUET. Distinct de `--leaf`, qui reste l'unite d'ELAGAGE : le paquet est
/// un NOEUD du Bsp, donc un sous-arbre entier, et l'elagage continue de se faire feuille par
/// feuille a l'interieur. A `loc_rate` egal a `--leaf` le paquet EST la feuille du germe, que la
/// marche ordinaire visite deja en premier -- l'accelerateur doit alors mesurer comme `AaBsp`, et
/// c'est l'auto-test de son implementation.
inline SI loc_rate = 64;

/// SONDE DE CHRONOMETRAGE, pas un reglage :
///
///   0 = les deux phases -- l'accelerateur veritable, la reference ;
///   1 = LE PAQUET SEUL, donc la cellule reste inachevee et les volumes sont FAUX ;
///   2 = les deux phases completes, PUIS un troisieme parcours identique au second mais qui ne
///       coupe pas, sur la cellule FINALE. Les volumes restent justes.
///
/// Ce qu'on en tire par soustraction :
///
///   `t(0) - t(1)` : ce que coute la phase 2 aujourd'hui, coupes comprises. Biais connu, petit et
///                   dans le sens qui SURESTIME : la cellule du mode 1 est coupee par 43 plans au
///                   lieu de 88, donc elle a moins de sommets (22.3 contre 26.5) et sa `measure`
///                   est un peu moins chere.
///   `t(2) - t(0)` : le prix d'une CERTIFICATION par le majorant affine sur une cellule deja finie
///                   -- exactement ce que paierait le second parcours si un noyau local (CGAL)
///                   avait deja termine la cellule dans le paquet.
///
/// = Pourquoi le mode 2 n'est pas « le second parcours sans couper »
///
/// C'etait ma premiere version, et elle mesurait deux fois rien : `may_cut` etant sans effet de
/// bord et la coupe etant devenue vide, le compilateur SUPPRIMAIT le parcours entier -- meme compte
/// d'instructions a 0.04 % pres entre les modes 1 et 2. Et meme non supprimee elle n'aurait rien
/// voulu dire : sur une cellule que le paquet n'a pas finie, rien n'est evince, le parcours devient
/// quadratique. Le mode 2 part donc de la cellule FINALE, et son accumulateur s'echappe dans
/// `loc_sink` pour que le parcours ne puisse pas etre efface.
inline int loc_phase = 0;

/// L'evier du mode 2. Non atomique : la sonde se chronometre a `--threads 1`.
inline SI loc_sink = 0;

/// LE PAQUET D'ABORD, LE RESTE DE L'ARBRE ENSUITE.
///
/// = Ce que ca fait
///
/// La cellule est coupee d'abord contre les germes de SON paquet -- un noeud du Bsp qui contient
/// une centaine de voisins immediats -- puis contre tout le reste de l'arbre, sous-arbre du paquet
/// SAUTE. Rien d'autre ne change : le resultat est l'intersection des memes demi-espaces, dans un
/// ordre different.
///
/// = Pourquoi c'est exact sans rien ajouter
///
/// Une cellule est une INTERSECTION, donc l'ordre des coupes ne change pas la reponse. Et il n'y a
/// aucun certificat a ecrire : le test d'eviction du second parcours -- la boite du noeud contre la
/// cellule DEJA retrecie, avec le majorant AFFINE des poids -- est precisement ce qui certifie
/// qu'aucun germe exterieur ne peut plus couper. Une cellule dont tous les voisins sont dans le
/// paquet voit donc son second parcours rejeter les noeuds tout en haut, et ne paie que la
/// descente. Il n'y a pas deux branches a maintenir, ni deux facons pour une cellule d'etre finie.
///
/// = Ce qu'on espere, et ce qui peut le manger
///
/// Le paquet donne au second parcours une cellule DEJA PETITE, donc un test d'eviction qui mord des
/// la premiere boite. Ce qu'on paie en echange : le paquet est propose EN ENTIER, sans elagage
/// utile en son sein, donc `loc_rate` germes coupes a coup sur la ou l'arbre en aurait ecarte une
/// partie. C'est le banc qui tranche, et le reglage est `--rate`.
template<int D>
struct AaBspLocT {
    static constexpr int dim = D;
    static constexpr const char *name = "loc";

    AaBspT<D> t;
    std::vector<SI> pack;       ///< pour chaque germe, dans l'ordre de l'arbre, la RACINE de son paquet

    SI nb_seeds() const { return t.nb_seeds(); }
    Vec<D> seed( SI k ) const { return t.seed( k ); }
    TF seed_w( SI k ) const { return t.seed_w( k ); }
    SI seed_id( SI k ) const { return t.seed_id( k ); }

    void build( const TF *const *P, const TF *W, SI n, SI leaf ) {
        t.build( P, W, n, leaf );
        pack.assign( n, 0 );
        marque( 0 );
    }

    /// Le premier noeud, en descendant, dont le sous-arbre tient dans `loc_rate` germes. Une
    /// feuille en est toujours un, donc la recursion se termine quel que soit le reglage.
    void marque( SI h ) {
        const auto &nd = t.nodes[ h ];
        if ( nd.right < 0 || nd.end - nd.beg <= loc_rate ) {
            for ( SI k = nd.beg; k < nd.end; ++k )
                pack[ k ] = h;
            return;
        }
        marque( h + 1 );                                // PREORDRE : le fils gauche est juste a cote
        marque( nd.right );
    }

    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        const Vec<D> p0 = t.seed( k0 );
        const SI i0 = t.seed_id( k0 ), pr = pack[ k0 ];
        // la cellule peut se VIDER dans le paquet : `for_each_candidate_from` le dit en rendant
        // `false`, et il n'y a alors plus rien a couper.
        if ( ! t.for_each_candidate_from( pr, p0, i0, may_cut, cut_with, reach2 ) )
            return;
        if ( loc_phase == 1 )
            return;
        t.for_each_candidate_skip( 0, pr, p0, i0, may_cut, cut_with, reach2 );
        if ( loc_phase == 2 ) {
            SI acc = 0;
            t.for_each_candidate_skip( 0, pr, p0, i0, may_cut,
                                       [ &acc ]( Vec<D>, TF, SI id ) { acc += id; return true; },
                                       reach2 );
            loc_sink += acc;
        }
    }
};

using AaBspLoc  = AaBspLocT<2>;
using AaBspLoc3 = AaBspLocT<3>;

} // namespace pd
