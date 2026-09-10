#pragma once

// =====================================================================================
// LE CONTRAT -- ce que le moteur echange avec le monde, et qui ne depend d'aucune machine.
//
// Ce fichier a ete separe de `Noyau2D.h` le jour ou il a fallu compiler le noyau POUR PLUSIEURS
// CIBLES DANS UN MEME BINAIRE. Le mecanisme de highway ( `foreach_target.h` ) reinclut le code
// vectoriel une fois par cible, et ce qui doit rester UNIQUE -- les types echanges, la coupe
// scalaire -- doit alors vivre ailleurs. La separation etait de toute facon la bonne : `Plan`,
// `Atelier` et `coupe_large` ne contiennent pas une instruction machine.
//
// Y RESTE DONC : le demi-plan, la vue que le fournisseur recoit d'une cellule en memoire,
// l'atelier, et la coupe scalaire en place.
// N'Y EST PAS : tout ce qui parle de registres -- c'est `Noyau2D.h` ( intrinseques x86 ) ou
// `Noyau2DHwy-inl.h` ( highway, portable ).
// =====================================================================================

namespace noyau2d {

// =====================================================================================
// CE QUE LE FOURNISSEUR PEUT DEMANDER AU MOTEUR
//
// Le noyau ne connait aucune politique. Il n'offre que deux choses, et seulement si on les lui
// demande -- silence radio sinon, et cout nul.
//
//   `Fourn::Local`         un objet par CELLULE, que le moteur cree dans SA frame et repasse a
//                          chaque appel. Le fournisseur y met ce qu'il veut -- un compteur de
//                          rejets, une boule, un rayon, une boite -- et le noyau ne le lit
//                          JAMAIS. Vide par defaut, donc gratuit.
//
//   `Fourn::veut_changement`   si vrai, `Etat::change` dit si la cellule a bouge depuis l'appel
//                          precedent. C'est ce qui permet de compter les rejets consecutifs, donc
//                          de decider d'une transition, sans rien recalculer.
//
// CE QUE LE NOYAU NE FAIT PAS. Il ne calcule ni boule, ni boite, ni rayon. S'il le faisait il
// choisirait a la place du fournisseur, et il choisirait mal : une boule centree sur la boite des
// sommets est plus lache qu'un rayon autour du germe, et le noyau ne connait pas le germe.
// Mesure : 15.5 tentatives par cellule contre 12.9, et x1.05 contre x1.18. Le fournisseur, lui,
// voit `Etat::vx / vy` -- des registres -- et fait son SIMD lui-meme.

/// pour un fournisseur qui n'a rien a garder d'une coupe a l'autre.
struct RienDeLocal {};

template<class F> struct local_de { using type = RienDeLocal; };
template<class F> requires requires { typename F::Local; }
struct local_de<F> { using type = typename F::Local; };
template<class F> using Local = typename local_de<F>::type;

template<class F> consteval bool veut_changement() {
    if constexpr ( requires { F::veut_changement; } ) return F::veut_changement;
    else return false;
}

/// LE DEMI-PLAN, tel qu'un fournisseur le rend : `dx * x + dy * y <= off`. `id` est ce qui ira
/// dans `cid` -- l'IDENTITE de la coupe, dont la geometrie ne depend pas. C'est elle qui porte
/// la connectivite, et c'est depuis elle qu'on refabrique le plan quand on en a besoin.
struct Plan {
    float dx, dy, off;
    int   id;
};

/// LA MEME SURFACE, quand la cellule est en memoire -- pendant l'excursion, et pour les chemins
/// scalaires. Le flottant est un parametre : `float` pour l'excursion et le jumeau scalaire,
/// `double` pour le repli. Un seul fournisseur sert donc TOUS les chemins.
template<class T>
struct EtatLarge {
    int nb;
    const T   *vx, *vy;
    const int *cid;

    /// meme sens que dans `Etat<NB>` : la cellule a-t-elle bouge depuis l'appel precedent, et une
    /// boule englobante. Renseignes seulement si la politique du moteur le demande.
    bool change;

    T   x ( int i ) const { return vx [ i ]; }
    T   y ( int i ) const { return vy [ i ]; }
    int id( int i ) const { return cid[ i ]; }
};

/// L'ATELIER -- ce qui reste de `Ctx` une fois les entrees parties chez le fournisseur : la
/// SORTIE, et l'endroit ou la cellule se pose quand elle deborde des registres.
///
/// `MaxNb` decide du comportement au neuvieme sommet, et c'est le seul endroit ou ce choix
/// s'ecrit :
///   `MaxNb == 8`  : pas de place pour l'excursion, le noyau rend `nb = -1` et laisse
///                   `attente` renseigne -- l'ancien echappement, pour qui veut reprendre en
///                   double ( cf. `Repli2D.h` ) ;
///   `MaxNb > 8`   : la cellule y tient, l'excursion a lieu, et le noyau rend toujours un
///                   resultat complet.
template<int MaxNb>
struct Atelier {
    static_assert( MaxNb >= 8, "l'atelier doit au moins pouvoir recevoir les huit registres" );
    static constexpr int max_nb = MaxNb;

    int  nb;                    ///< SORTIE : nb de sommets, 0 = vide, -1 = atelier trop petit
    Plan attente;               ///< si `nb < 0` : la coupe qui deborde, PAS appliquee
    alignas( 32 ) float vx[ MaxNb ], vy[ MaxNb ];
    alignas( 32 ) int   cid[ MaxNb ];
};

/// LA COUPE SCALAIRE, EN PLACE -- celle de l'excursion, et celle du jumeau scalaire. Rend le
/// nouveau nombre de sommets : `nb` inchange si le plan ne coupe rien, `0` si la cellule est
/// vide, `-1` si la sortie ne tient pas dans `MaxNb`.
///
/// `__restrict` N'EST PAS DECORATIF. Les tableaux etaient des LOCALES quand la coupe vivait
/// dans le corps de son appelant : le compilateur savait qu'elles ne se recouvraient pas. En
/// devenant des parametres ils sont redevenus des pointeurs quelconques, et la premiere boucle
/// -- qui ecrit `s` en lisant `vx` et `vy` -- a cesse d'etre vectorisable. Cout mesure : +36 %
/// sur le chemin scalaire. Le mot-cle rend ce que la mise en facteur avait pris.
///
/// LE SCAN EST EN DEUX PASSES, ET C'EST LE CONTRAIRE DE CE QU'ON CROIT. Compter les sommets
/// dehors et trouver le debut de la plage dans la MEME boucle, en portant le signe du
/// precedent, fait moins d'instructions et va plus lentement : la dependance portee empeche le
/// vectoriseur de prendre le produit scalaire, qui est la seule partie qui compte. Mesure sur
/// le banc scalaire, n = 50 : 17.1 ns par coupe en une passe, 13.2 en deux.
template<int MaxNb>
[[gnu::always_inline]] inline int coupe_large( float *__restrict vx, float *__restrict vy, int *__restrict cid,
                 int nb, const Plan &p, float *__restrict s ) {
    int nb_out = 0;
    for ( int i = 0; i < nb; ++i ) {                     // reduction pure : le vectoriseur la prend
        s[ i ] = p.dx * vx[ i ] + p.dy * vy[ i ] - p.off;
        nb_out += s[ i ] > 0;
    }
    if ( nb_out == 0 )
        return nb;
    if ( nb_out == nb )
        return 0;

    int i1 = 0;                                          // `i1` est unique : l'exterieur d'un
    for ( int i = 0; i < nb; ++i ) {                     // convexe coupe est d'un seul tenant
        const int q = i ? i - 1 : nb - 1;
        if ( s[ i ] > 0 && ! ( s[ q ] > 0 ) ) { i1 = i; break; }
    }

    const int nb_in = nb - nb_out;
    const int new_nb = nb_in + 2;
    if ( new_nb > MaxNb )
        return -1;                                       // la cellule reste INTACTE

    const int j0 = ( i1 + nb - 1 ) % nb;                 // dernier DEDANS avant la plage
    const int j2 = ( i1 + nb_out - 1 ) % nb;             // dernier DEHORS
    const int j3 = ( j2 + 1 ) % nb;                      // premier DEDANS apres

    // les deux intersections, ancrees sur le sommet DEDANS : `v_in + ( v_out - v_in ) * t` vaut
    // EXACTEMENT `v_in` en `t == 0`, ce que la forme symetrique n'a pas -- et sans quoi la plage
    // cesse d'etre contigue.
    const float s0 = s[ j0 ], s1 = s[ i1 ], s2 = s[ j2 ], s3 = s[ j3 ];
    const float ta  = s0 / ( s0 - s1 );
    const float pax = vx[ j0 ] + ( vx[ i1 ] - vx[ j0 ] ) * ta;
    const float pay = vy[ j0 ] + ( vy[ i1 ] - vy[ j0 ] ) * ta;
    const float tb  = s3 / ( s3 - s2 );
    const float pbx = vx[ j3 ] + ( vx[ j2 ] - vx[ j3 ] ) * tb;
    const float pby = vy[ j3 ] + ( vy[ j2 ] - vy[ j3 ] ) * tb;
    const int   bid = cid[ j2 ];                         // lu MAINTENANT : `j2` va etre ecrase

    auto move = [ & ]( int d, int t ) { vx[ d ] = vx[ t ]; vy[ d ] = vy[ t ]; cid[ d ] = cid[ t ]; };

    if ( i1 <= j2 ) {
        if ( nb_out == 1 ) {                             // un cran de plus : la queue va A DROITE
            for ( int i = nb; i > i1 + 1; --i ) move( i, i - 1 );
        } else if ( nb_out > 2 ) {                       // trop de place : la queue revient A GAUCHE
            const int gap = nb_out - 2;
            for ( int i = j2 + 1; i < nb; ++i ) move( i - gap, i );
        }                                                // `nb_out == 2` : rien a decaler
        vx[ i1 ] = pax; vy[ i1 ] = pay; cid[ i1 ] = p.id;
        vx[ i1 + 1 ] = pbx; vy[ i1 + 1 ] = pby; cid[ i1 + 1 ] = bid;
    } else {
        // la plage BOUCLE, donc l'interieur est contigu : `[ j3, j3 + nb_in )`.
        if ( j3 >= 2 ) for ( int o = 0; o < nb_in; ++o ) move( 2 + o, j3 + o );
        else           for ( int o = nb_in - 1; o >= 0; --o ) move( 2 + o, j3 + o );
        vx[ 0 ] = pax; vy[ 0 ] = pay; cid[ 0 ] = p.id;
        vx[ 1 ] = pbx; vy[ 1 ] = pby; cid[ 1 ] = bid;
    }
    return new_nb;
}


/// RECONSTRUIRE UNE CELLULE DEPUIS SES SEULS `cid`.
///
/// C'est l'affirmation de l'en-tete du noyau prise au mot : l'ordre porte la connectivite, donc la
/// geometrie est redondante. Le plan de la coupe `i` se refabrique depuis `cid[ i ]`, et
/// l'invariant « la coupe `i` porte l'arete [ v_i, v_i+1 ] » dit que le sommet `i` est
/// l'intersection des coupes `i-1` et `i`. Un systeme 2x2 par sommet.
///
/// Ce qu'on y gagne : une cellule se STOCKE en `nb` entiers, pas en `nb` points. Pour la phase 2
/// des sur-cellules -- ou il faut garder toutes les cellules entre les deux passes -- c'est la
/// difference entre garder la connectivite et garder la geometrie.
///
/// `plan_de` doit rendre le demi-plan d'un identifiant : `plan_de( id, dx, dy, off )`.
template<int MaxNb, class PlanDe>
void reconstruit( float *vx, float *vy, const int *cid, int nb, PlanDe &&plan_de ) {
    float dx[ MaxNb ], dy[ MaxNb ], of[ MaxNb ];
    for ( int i = 0; i < nb; ++i )
        plan_de( cid[ i ], dx[ i ], dy[ i ], of[ i ] );
    for ( int i = 0, j = nb - 1; i < nb; j = i++ ) {
        const float a1 = dx[ j ], b1 = dy[ j ], f1 = of[ j ];
        const float a2 = dx[ i ], b2 = dy[ i ], f2 = of[ i ];
        const float det = a1 * b2 - b1 * a2;             // nul seulement pour deux coupes
        vx[ i ] = ( f1 * b2 - b1 * f2 ) / det;           // consecutives PARALLELES, ce qu'un
        vy[ i ] = ( a1 * f2 - f1 * a2 ) / det;           // convexe non degenere n'a pas
    }
}

/// les quatre cotes du carre unite, tels que `carre_unite` les numerote.
inline void plan_domaine( int id, float &dx, float &dy, float &off ) {
    static const float cx[ 4 ] = {  0,  1,  0, -1 };
    static const float cy[ 4 ] = { -1,  0,  1,  0 };
    static const float cf[ 4 ] = {  0,  1,  1,  0 };
    const int s = -id - 1;
    dx = cx[ s ]; dy = cy[ s ]; off = cf[ s ];
}

} // namespace noyau2d
