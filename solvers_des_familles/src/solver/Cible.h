#pragma once

// =====================================================================================
// MODIFIER LE SECOND MEMBRE POUR ALLONGER LE PAS ADMISSIBLE
//
// L'idee : `L d = nu - a` donne une direction que l'amortissement ne peut suivre que jusqu'a
// `F`, parce qu'une poignee de cellules s'y eteint. Plutot que de refaire le systeme ou de
// reparer apres coup ( § 13, § 14, § 15 ), on change la CIBLE -- donc le seul second membre --
// et on resout une fois de plus sur LA MEME FACTORISATION. Pas de diagramme, pas d'assemblage.
// La cible modifiee est transitoire : l'iteration suivante repart de `nu`.
//
// = Ce que le second membre peut, et ce qu'il ne peut pas
//
// Le flux d'aire qui sort de `i` par la facette `j` vaut `c_ij ( d_j - d_i )` ( § 15.13 ). La
// cellule meurt quand le flux SORTANT BRUT l'a consommee :
//
//     S_i = sum_j max( 0, c_ij ( d_j - d_i ) )        U_i = ( a_i - eps ) / S_i
//
// Or le second membre ne fixe que la DIVERGENCE : `( L d )_i = sum_j c_ij ( d_i - d_j )` est le
// flux NET. Nourrir une cellule exposee -- lui donner `nu_i + delta` -- ne borne que le net ; le
// brut peut rester le meme ( la cellule encaisse plus d'un cote et en perd autant de l'autre ).
// Le gradient de `d`, lui, n'est pas une quantite que le second membre atteint.
//
// = D'ou le seul levier honnete : ANNULER la demande autour du foyer
//
// Sur un patch `P` ou l'on pose `nu_i := a_i` ( donc `b_i = 0` ), `d` devient HARMONIQUE : parmi
// tous les champs qui valent la meme chose au bord de `P`, l'harmonique MINIMISE l'energie de
// Dirichlet `sum c_ij ( d_i - d_j )^2`, donc les flux. C'est le plus petit gradient qu'un second
// membre puisse obtenir dans `P` sans toucher au reste -- le PLAFOND de toute la famille d'idees.
// Ce qui reste du gradient est ce que le bord impose, et aucun choix de cible dans `P` ne l'enleve.
//
// Le mode PLAFOND mesure exactement ca, sans rien modifier : une resolution de plus par epaisseur
// d'anneau, et la trace dit de combien `U*` monterait au mieux. Le mode GEL s'en sert :
// `b( lambda ) = lambda b` dans `P`, `b` dehors ( plus la compensation qui garde `sum b = 0` --
// sans elle la ligne rayee par la jauge porte l'incoherence, § 9.6 ), et une bissection sur
// `lambda` cherche la plus grande fidelite a `nu` qui atteigne le `F` vise.
//
// = Le controle a ne pas oublier
//
// `nu~ = a + s ( nu - a )` donne `d~ = s d` EXACTEMENT : le pas admissible est `U*/s`, et on n'a
// rien gagne -- c'est un pas plus court deguise. Toute modification UNIFORME du second membre est
// ce controle. Seule une deformation non uniforme peut payer, et `lambda` sur un patch en est une.
// =====================================================================================

#include "solver/Laplacien.h"
#include "solver/Lineaire.h"
#include "util/common.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sf {

struct OptionsCible {
    /// SEULES : le doseur branche sur la liste `seules` de `essai-limites` -- les cellules qu'un
    /// DIAGRAMME d'essai a trouvees mortes en `t`, pas celles qu'un crible soupconne. C'est la
    /// seule facon de payer le bon predicteur sans payer de le chercher : la liste est deja la,
    /// elle est courte, et elle est VRAIE. On nourrit ces cellules juste assez pour que le
    /// polynome les donne vivantes en `t`, et on re-essaye LE MEME PAS au lieu de le diviser.
    enum Mode : int { NON = 0, PLAFOND, GEL, SEULES };
    int mode    = NON;
    int anneaux = 2;        ///< epaisseur du patch autour des cellules qui bornent le pas
    TF  facteur = 2;        ///< le `F` vise, en multiples du `U*` de la direction de Newton
    int essais  = 5;        ///< bissections sur `lambda` ( GEL ) -- autant de resolutions
    TF  seuil   = 0.5;      ///< au-dessus de ce `U*`, on ne touche a rien : le pas est deja bon
    /// CE QU'ON OPTIMISE. `MIN` : le pas de la cellule la plus exposee -- c'est la lecture
    /// naturelle, et c'est celle que le § 15.15 a deja disqualifiee ailleurs ( « ce n'est pas `U`
    /// qui est faux, c'est le critere » ). `POPULATION` : le NOMBRE de cellules qui ne tiennent
    /// pas jusqu'a l'horizon du pas -- parce que ce n'est pas une cellule isolee qui borne, c'est
    /// une population ( § 13.4 ), et en sauver une en nomme une autre.
    enum Critere : int { MIN = 0, POPULATION };
    int critere = MIN;
    /// D'OU VIENT LA MASSE QU'ON DONNE, et c'est une question de FREQUENCE. Reprise au prorata de
    /// `nu` sur tout le dehors ( GLOBAL ), la deformation est un MONOPOLE : `delta` a une partie
    /// positive ponctuelle et une partie negative etalee, donc `L^+ delta` porte en 2D un
    /// potentiel logarithmique -- la correction traine sur tout le diagramme et deregle les basses
    /// frequences que le pas plein resolvait bien. Reprise A COTE, la somme est nulle
    /// LOCALEMENT : le champ devient dipolaire et s'amortit.
    ///
    ///   ANNEAU    : par AMAS de cellules malades qui se touchent, chacun prend a sa propre
    ///               couronne ( `ep_anneau` couches ), au prorata de `nu` -- somme nulle par amas
    ///   MANGEURS  : chaque victime prend a SES mangeurs, au prorata du flux qu'ils lui volent --
    ///               somme nulle par cellule, le dipole le plus court qui existe
    enum Repris : int { GLOBAL = 0, ANNEAU, MANGEURS };
    int repris = GLOBAL;
    int ep_anneau = 1;      ///< ANNEAU : epaisseur de la couronne donneuse
    /// QUI JUGE LA DOSE. `FLUX` : `U`, la linearisation des flux en `t = 0` ( § 15.13 ) -- gratuite
    /// mais fausse d'un facteur cent, et le § 20.5 montre que l'optimiser la rend pire encore.
    /// `POLYNOME` : l'AIRE A COMBINATOIRE FIGEE ( § 7 ), qui est le bon predicteur -- exacte tant
    /// que la combinatoire tient, et mesuree exacte a l'iteration 0 de ce nuage meme.
    ///
    /// Elle ne coute pas ce qu'on croit. `ModeleCellule` porte les droites de la cellule ; l'aire
    /// s'en deduit pour un deplacement de poids QUELCONQUE, par pure arithmetique. On construit
    /// donc la geometrie UNE fois, pour les seules candidates, et tout le balayage sur `kappa`
    /// -- toutes les directions, tous les `alpha` -- se lit dessus sans un calcul de cellule.
    ///
    /// Et `kappa` entre LINEAIREMENT dans le second membre ( le coeur et les donneurs n'en
    /// dependent pas ), donc `d~( kappa ) = d + kappa e` avec `L e = delta( 1 )` : UNE resolution
    /// pour tout le balayage, au lieu d'une par essai.
    enum Juge : int { FLUX = 0, POLYNOME };
    int juge = FLUX;
    TF  filtre = 8;         ///< POLYNOME : on modelise les cellules dont `U` tombe sous `filtre * vise`
    SI  max_seules = 0;     ///< SEULES : au-dela de tant de mauvaises, on renonce ( 0 : aucune limite )
    /// LE BUDGET DE DEFORMATION, en `| db | / | b |`. C'est le vrai reglage du compromis : on ne
    /// se fixe PAS une cible a atteindre ( tout-ou-rien ), on prend la meilleure limite qu'on
    /// puisse acheter dedans. `0` : aucun plafond.
    ///
    /// Il se traduit exactement en un plafond sur `kappa`, sans essai : `delta` est proportionnel
    /// a `kappa`, donc `kappa_max = budget | b | / | delta( 1 ) |`.
    TF  budget = 0;
};

/// ce qu'une passe a trouve, pour la trace et pour les stats
struct EtatCible {
    TF  u0 = 0, u1 = 0;     ///< le pas lu sur les flux, avant et apres
    TF  vise = 0;           ///< le `F` demande
    TF  lambda = 1;         ///< la deformation retenue ( 1 : Newton, 0 : patch gele )
    SI  nb_noyau = 0;       ///< cellules sous le `F` vise
    SI  nb_patch = 0;       ///< elles plus leurs anneaux
    SI  lie0 = -1, lie1 = -1; ///< la cellule qui borne le pas, avant et apres
    bool dedans = false;    ///< celle d'apres est-elle dans le patch ?
    TF  masse = 0;          ///< masse deplacee, en fraction de `sum nu`
    TF  ecart = 0;          ///< `| b~ - b |_2 / | b |_2`
    int nb_res = 0;         ///< resolutions lineaires depensees
};

/// LE PAS LU SUR LES FLUX ( § 15.14 ) : aucune geometrie, une passe sur les aretes du laplacien.
/// Rend `min_i U_i` et remplit `U`. Une cellule que personne ne mange ne borne rien.
inline TF pas_par_flux( const Laplacien &L, const std::vector<TF> &a, const std::vector<TF> &d,
                        TF eps, std::vector<TF> &U, SI &lie, std::vector<TF> *Sor = nullptr ) {
    static constexpr TF INF = std::numeric_limits<TF>::infinity();
    const SI n = L.n;
    U.assign( n, INF );
    if ( Sor ) Sor->assign( n, TF( 0 ) );
    TF u = INF;
    lie = -1;
    for ( SI i = 0; i < n; ++i ) {
        TF s = 0;
        for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) {
            const TF f = L.c[ k ] * ( d[ L.col[ k ] ] - d[ i ] );
            if ( f > 0 ) s += f;
        }
        if ( Sor ) ( *Sor )[ i ] = s;
        if ( ! ( s > 0 ) )                               // aucun voisin ne lui prend d'aire
            continue;
        const TF reste = a[ i ] - eps;
        U[ i ] = reste > 0 ? reste / s : TF( 0 );
        if ( U[ i ] < u ) { u = U[ i ]; lie = i; }
    }
    return u;
}

/// LE PATCH : les cellules marquees, plus `anneaux` couches de voisins ( parcours en largeur sur
/// le graphe du laplacien -- les composantes qui se touchent fusionnent d'elles-memes ).
inline SI dilate( const Laplacien &L, std::vector<char> &dans, int anneaux ) {
    const SI n = L.n;
    std::vector<SI> front, suiv;
    for ( SI i = 0; i < n; ++i )
        if ( dans[ i ] ) front.push_back( i );
    for ( int r = 0; r < anneaux; ++r ) {
        suiv.clear();
        for ( SI i : front )
            for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) {
                const SI j = L.col[ k ];
                if ( ! dans[ j ] ) { dans[ j ] = 1; suiv.push_back( j ); }
            }
        front.swap( suiv );
    }
    SI c = 0;
    for ( SI i = 0; i < n; ++i ) c += dans[ i ];
    return c;
}

/// `b~ = lambda b` dans le patch, `b` dehors, et la masse relachee REDISTRIBUEE au prorata de `nu`
/// sur le dehors -- `sum b~ = 0` exactement, sans quoi la ligne de la jauge porte l'incoherence.
inline void melange_cible( const std::vector<TF> &b, const std::vector<TF> &nu,
                           const std::vector<char> &dans, TF lambda, std::vector<TF> &b2 ) {
    const SI n = SI( b.size() );
    TF sb = 0, sn = 0;
    for ( SI i = 0; i < n; ++i ) {
        if ( dans[ i ] ) sb += b[ i ];
        else             sn += nu[ i ];
    }
    const TF r = sn > 0 ? ( 1 - lambda ) * sb / sn : TF( 0 );
    b2.resize( n );
    for ( SI i = 0; i < n; ++i )
        b2[ i ] = dans[ i ] ? lambda * b[ i ] : b[ i ] + r * nu[ i ];
}

/// NOURRIR LES EXPOSEES ( l'idee telle qu'elle se pose ) : `nu~_i = nu_i + kappa S_i` sur le
/// NOYAU seul -- `S_i` est la bonne echelle, c'est le debit qui la tue -- et la masse prise au
/// prorata de `nu` sur tout le reste, pour que `sum b~ = 0`.
///
/// Ce que `kappa` achete est un flux NET, pas une baisse du flux BRUT : `kappa = 1` donne a la
/// cellule de quoi compenser exactement son debit sortant SI tout le supplement passait par les
/// facettes devorantes. La mesure dit quelle fraction y passe vraiment.
inline TF nourri_cible( const std::vector<TF> &b, const std::vector<TF> &nu, const std::vector<TF> &S,
                        const std::vector<char> &coeur, TF kappa, std::vector<TF> &b2 ) {
    const SI n = SI( b.size() );
    TF sd = 0, sn = 0;
    for ( SI i = 0; i < n; ++i ) {
        if ( coeur[ i ] ) sd += kappa * S[ i ];
        else              sn += nu[ i ];
    }
    const TF r = sn > 0 ? sd / sn : TF( 0 );
    b2.resize( n );
    for ( SI i = 0; i < n; ++i )
        b2[ i ] = coeur[ i ] ? b[ i ] + kappa * S[ i ] : b[ i ] - r * nu[ i ];
    return sd;                                           // la masse donnee, a diviser par `sum nu`
}

/// NOURRIR AU DEFICIT, et c'est la seule dose qui ait une echelle. La cellule `i` tient jusqu'a
/// `U_i = ( a_i - eps ) / S_i` ; pour tenir jusqu'a `F` il lui faut un debit sortant d'au plus
/// `( a_i - eps ) / F`, donc un supplement de cible de
///
///     delta_i = kappa max( 0, S_i - ( a_i - eps ) / F )
///
/// nul de lui-meme des que `U_i >= F` -- aucune liste a tenir, aucun seuil a poser. `kappa = 1`
/// est la dose qui suffirait SI tout le supplement passait par les facettes devorantes ; elle ne
/// le fait pas, parce que le second membre ne fixe que le flux NET. La masse est reprise au
/// prorata de `nu` sur les cellules qui ne recoivent rien.
inline TF nourri_deficit( const std::vector<TF> &b, const std::vector<TF> &nu, const std::vector<TF> &S,
                          const std::vector<TF> &a, TF eps, TF F, TF kappa, std::vector<TF> &b2,
                          std::vector<TF> &del ) {
    const SI n = SI( b.size() );
    del.assign( n, TF( 0 ) );
    TF sd = 0, sn = 0;
    for ( SI i = 0; i < n; ++i ) {
        const TF manque = S[ i ] - ( a[ i ] - eps ) / F;
        if ( manque > 0 ) { del[ i ] = kappa * manque; sd += del[ i ]; }
        else              sn += nu[ i ];
    }
    const TF r = sn > 0 ? sd / sn : TF( 0 );
    b2.resize( n );
    for ( SI i = 0; i < n; ++i )
        b2[ i ] = del[ i ] > 0 ? b[ i ] + del[ i ] : b[ i ] - r * nu[ i ];
    return sd;
}

/// LA MEME DOSE, MAIS LA MASSE REPRISE A COTE ( `OptionsCible::Repris` ). Rend la masse donnee,
/// remplit `del` ( somme nulle ), et marque dans `touche` le coeur et ses donneurs -- de quoi
/// mesurer ensuite quelle part de la correction de direction FUIT hors de la zone traitee.
inline TF nourri_local( const Laplacien &L, const std::vector<TF> &b, const std::vector<TF> &nu,
                        const std::vector<TF> &S, const std::vector<TF> &a, const std::vector<TF> &d,
                        TF eps, TF F, TF kappa, int repris, int ep,
                        std::vector<TF> &b2, std::vector<TF> &del, std::vector<char> &touche,
                        SI &nb_coeur, SI &nb_amas, SI &nb_don,
                        const std::vector<char> *coeur_impose = nullptr ) {
    const SI n = L.n;
    del.assign( n, TF( 0 ) );
    touche.assign( n, 0 );
    std::vector<char> coeur( n, 0 );
    nb_coeur = nb_amas = nb_don = 0;
    // LE COEUR : soit le deficit de flux ( regle interne ), soit une liste IMPOSEE par un
    // meilleur predicteur -- on nourrit alors au meme rythme `S_i`, qui reste la bonne echelle
    // de masse par unite de pas, mais on nourrit LES BONNES cellules.
    for ( SI i = 0; i < n; ++i ) {
        const TF manque = coeur_impose ? ( ( *coeur_impose )[ i ] ? S[ i ] : TF( 0 ) )
                                       : S[ i ] - ( a[ i ] - eps ) / F;
        if ( manque > 0 ) { del[ i ] = kappa * manque; coeur[ i ] = 1; ++nb_coeur; }
    }
    if ( ! nb_coeur ) { b2 = b; return 0; }

    if ( repris == OptionsCible::MANGEURS ) {
        for ( SI i = 0; i < n; ++i ) {
            if ( ! coeur[ i ] ) continue;
            // les poids : le flux que le voisin lui vole, a defaut la conductance. Un voisin
            // malade n'est pas un donneur -- on ne deshabille pas Pierre pour habiller Paul.
            TF tot = 0;
            for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) {
                const SI j = L.col[ k ];
                if ( coeur[ j ] ) continue;
                const TF f = L.c[ k ] * ( d[ j ] - d[ i ] );
                tot += f > 0 ? f : TF( 0 );
            }
            bool par_flux = tot > 0;
            if ( ! par_flux )
                for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k )
                    if ( ! coeur[ L.col[ k ] ] ) tot += L.c[ k ];
            if ( ! ( tot > 0 ) ) { del[ i ] = 0; continue; }   // entouree de malades : on renonce
            for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) {
                const SI j = L.col[ k ];
                if ( coeur[ j ] ) continue;
                const TF f = L.c[ k ] * ( d[ j ] - d[ i ] );
                const TF p = par_flux ? ( f > 0 ? f : TF( 0 ) ) : L.c[ k ];
                if ( p > 0 ) del[ j ] -= del[ i ] * p / tot;
            }
        }
        nb_amas = nb_coeur;                              // une victime, un dipole
    } else if ( repris == OptionsCible::ANNEAU ) {
        // LES AMAS : union-find sur les cellules malades qui se touchent
        std::vector<SI> pere( n );
        for ( SI i = 0; i < n; ++i ) pere[ i ] = i;
        auto racine = [ & ]( SI i ) { while ( pere[ i ] != i ) { pere[ i ] = pere[ pere[ i ] ]; i = pere[ i ]; } return i; };
        for ( SI i = 0; i < n; ++i ) {
            if ( ! coeur[ i ] ) continue;
            for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) {
                const SI j = L.col[ k ];
                if ( coeur[ j ] ) { const SI ri = racine( i ), rj = racine( j ); if ( ri != rj ) pere[ ri ] = rj; }
            }
        }
        std::vector<SI> tete;                            // les racines, une par amas
        std::vector<std::vector<SI>> amas;
        std::vector<SI> ou( n, -1 );
        for ( SI i = 0; i < n; ++i ) {
            if ( ! coeur[ i ] ) continue;
            const SI r = racine( i );
            if ( ou[ r ] < 0 ) { ou[ r ] = SI( amas.size() ); amas.emplace_back(); tete.push_back( r ); }
            amas[ ou[ r ] ].push_back( i );
        }
        nb_amas = SI( amas.size() );
        std::vector<SI> front, suiv, vus;
        std::vector<char> vu( n, 0 );
        for ( auto &cl : amas ) {
            TF don = 0;
            for ( SI i : cl ) don += del[ i ];
            // la couronne de CET amas : `ep` couches de cellules saines
            vus.clear();
            front = cl;
            for ( int r = 0; r < ep; ++r ) {
                suiv.clear();
                for ( SI i : front )
                    for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) {
                        const SI j = L.col[ k ];
                        if ( coeur[ j ] || vu[ j ] ) continue;
                        vu[ j ] = 1; suiv.push_back( j ); vus.push_back( j );
                    }
                front = suiv;
            }
            TF sn = 0;
            for ( SI j : vus ) sn += nu[ j ];
            for ( SI j : vus ) {
                if ( sn > 0 ) del[ j ] -= don * nu[ j ] / sn;
                vu[ j ] = 0;                             // rendu disponible pour l'amas suivant
            }
            if ( ! ( sn > 0 ) )                          // aucun donneur : on renonce pour cet amas
                for ( SI i : cl ) del[ i ] = 0;
        }
    } else {
        TF sd = 0, sn = 0;
        for ( SI i = 0; i < n; ++i ) { if ( coeur[ i ] ) sd += del[ i ]; else sn += nu[ i ]; }
        const TF r = sn > 0 ? sd / sn : TF( 0 );
        for ( SI i = 0; i < n; ++i ) if ( ! coeur[ i ] ) del[ i ] = -r * nu[ i ];
        nb_amas = 1;
    }

    // LA ZONE DE MESURE, la meme pour les trois regles : le coeur et ses `ep` anneaux. C'est
    // contre elle qu'on lira la part de la correction qui fuit ailleurs.
    touche = coeur;
    nb_don = dilate( L, touche, std::max( ep, 1 ) ) - nb_coeur;

    TF sd = 0;
    b2.resize( n );
    for ( SI i = 0; i < n; ++i ) {
        if ( del[ i ] > 0 ) sd += del[ i ];
        b2[ i ] = b[ i ] + del[ i ];
    }
    return sd;
}

/// LA PORTEE de la correction de direction : quelle part de son energie vit HORS de la zone
/// traitee. C'est la question des basses frequences, posee en un nombre. La jauge est otee
/// ( une constante ne change aucune cellule ).
inline TF fuite_direction( const std::vector<TF> &d, const std::vector<TF> &d2,
                           const std::vector<char> &touche, TF &emax ) {
    const SI n = SI( d.size() );
    TF m = 0;
    for ( SI i = 0; i < n; ++i ) m += d2[ i ] - d[ i ];
    m /= TF( n );
    TF dedans = 0, dehors = 0;
    emax = 0;
    for ( SI i = 0; i < n; ++i ) {
        const TF e = d2[ i ] - d[ i ] - m;
        emax = std::max( emax, std::fabs( e ) );
        ( touche[ i ] ? dedans : dehors ) += e * e;
    }
    return dedans + dehors > 0 ? dehors / ( dedans + dehors ) : TF( 0 );
}

} // namespace sf
