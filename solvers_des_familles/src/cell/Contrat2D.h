#pragma once

// =====================================================================================
// LE CONTRAT 2D : ce que le noyau et le fournisseur echangent. Pas une instruction machine ici.
//
// = LA CELLULE DIRIGE
//
// Le noyau ne connait ni structure d'acceleration, ni dirac, ni politique. Il recoit un
// FOURNISSEUR -- un objet a une seule methode,
//
//      template<class Etat> bool suivant( const Etat &e, Local &l, Plan2<TK> &p );
//
// qui remplit `p` et rend `true`, ou rend `false` quand il n'a plus rien -- et il coupe. Le
// fournisseur voit la cellule TELLE QU'ELLE EST apres chaque coupe ( `e` ), donc il decide apres
// coup, pas d'avance ; c'est ce qui rend l'elagage exact.
//
// `Local` est a lui : s'il declare un type `Local`, le moteur en cree un par CELLULE dans sa frame
// et le lui repasse a chaque appel. C'est la que vit la pile du parcours d'arbre. Un fournisseur
// sans etat n'en declare pas et ne paie rien.
//
// = LA CELLULE EST TROIS TABLEAUX
//
// Les sommets en ORDRE CYCLIQUE et l'identite de la coupe qui porte l'arete `[ v_i, v_i+1 ]`.
// Les plans ne sont pas gardes : l'ordre porte la connectivite, la geometrie est redondante.
// `cid < 0` designe un cote du domaine ( `-1 .. -4` pour le carre unite ).
//
// `TK` est le flottant du noyau : `float` ou `double`.
// =====================================================================================

#include <cstdint>

namespace sf::d2 {

using SI32 = std::int32_t;

/// pour un fournisseur qui n'a rien a garder d'une coupe a l'autre.
struct RienDeLocal {};

template<class F> struct local_de { using type = RienDeLocal; };
template<class F> requires requires { typename F::Local; }
struct local_de<F> { using type = typename F::Local; };
template<class F> using Local = typename local_de<F>::type;

/// LE DEMI-PLAN, tel qu'un fournisseur le rend : `dx * x + dy * y <= off`. `id` ira dans `cid`.
template<class TK>
struct Plan2 {
    TK   dx, dy, off;
    SI32 id;
};

/// CE QUE LE FOURNISSEUR VOIT quand la cellule est en MEMOIRE ( pendant une excursion ).
template<class TK>
struct EtatLarge {
    int         nb;
    const TK   *vx, *vy;
    const SI32 *cid;
};

/// L'ATELIER : la SORTIE du noyau, et l'endroit ou la cellule se pose quand elle deborde des
/// huit registres. `MaxNb > 8` : l'excursion a lieu, et le noyau rend toujours une cellule
/// complete -- sauf si l'atelier lui-meme est trop petit, auquel cas `nb == -1`.
template<class TK, int MaxNb>
struct Atelier {
    static_assert( MaxNb >= 16, "l'atelier doit pouvoir heberger une excursion" );
    static constexpr int max_nb = MaxNb;

    int  nb;                    ///< SORTIE : nb de sommets, 0 = vide, -1 = atelier trop petit
    Plan2<TK> attente;          ///< la coupe qui a fait deborder, PAS encore appliquee
    alignas( 64 ) TK   vx[ MaxNb ], vy[ MaxNb ];
    alignas( 64 ) SI32 cid[ MaxNb ];
};

/// LA COUPE SCALAIRE, EN PLACE -- celle de l'excursion. Rend le nouveau nombre de sommets : `nb`
/// inchange si le plan ne coupe rien, `0` si la cellule est vide, `-1` si la sortie ne tient pas
/// dans `MaxNb` ( la cellule reste alors INTACTE ).
///
/// `__restrict` n'est pas decoratif : sans lui la premiere boucle cesse d'etre vectorisable
/// ( +36 % mesure ). Et le scan est en DEUX passes -- compter et chercher le debut de la plage
/// dans la meme boucle fait moins d'instructions et va plus lentement ( 17.1 contre 13.2 ns ).
template<int MaxNb, class TK>
[[gnu::always_inline]] inline int coupe_large( TK *__restrict vx, TK *__restrict vy,
                                               SI32 *__restrict cid, int nb, const Plan2<TK> &p,
                                               TK *__restrict s ) {
    int nb_out = 0;
    for ( int i = 0; i < nb; ++i ) {                     // reduction pure : le vectoriseur la prend
        s[ i ] = p.dx * vx[ i ] + p.dy * vy[ i ] - p.off;
        nb_out += s[ i ] > 0;
    }
    if ( nb_out == 0 )
        return nb;
    if ( nb_out == nb )
        return 0;

    int i1 = 0;                                          // l'exterieur d'un convexe coupe est d'un
    for ( int i = 0; i < nb; ++i ) {                     // seul tenant : `i1` est unique
        const int q = i ? i - 1 : nb - 1;
        if ( s[ i ] > 0 && ! ( s[ q ] > 0 ) ) { i1 = i; break; }
    }

    const int nb_in = nb - nb_out;
    const int new_nb = nb_in + 2;
    if ( new_nb > MaxNb )
        return -1;

    const int j0 = ( i1 + nb - 1 ) % nb;                 // dernier DEDANS avant la plage
    const int j2 = ( i1 + nb_out - 1 ) % nb;             // dernier DEHORS
    const int j3 = ( j2 + 1 ) % nb;                      // premier DEDANS apres

    // les deux intersections, ANCREES sur le sommet dedans : `v_in + ( v_out - v_in ) * t` vaut
    // exactement `v_in` en `t == 0`, sans quoi la plage cesse d'etre contigue.
    const TK s0 = s[ j0 ], s1 = s[ i1 ], s2 = s[ j2 ], s3 = s[ j3 ];
    const TK ta  = s0 / ( s0 - s1 );
    const TK pax = vx[ j0 ] + ( vx[ i1 ] - vx[ j0 ] ) * ta;
    const TK pay = vy[ j0 ] + ( vy[ i1 ] - vy[ j0 ] ) * ta;
    const TK tb  = s3 / ( s3 - s2 );
    const TK pbx = vx[ j3 ] + ( vx[ j2 ] - vx[ j3 ] ) * tb;
    const TK pby = vy[ j3 ] + ( vy[ j2 ] - vy[ j3 ] ) * tb;
    const SI32 bid = cid[ j2 ];                          // lu MAINTENANT : `j2` va etre ecrase

    auto move = [ & ]( int d, int t ) { vx[ d ] = vx[ t ]; vy[ d ] = vy[ t ]; cid[ d ] = cid[ t ]; };

    if ( i1 <= j2 ) {
        if ( nb_out == 1 ) {                             // un cran de plus : la queue va A DROITE
            for ( int i = nb; i > i1 + 1; --i ) move( i, i - 1 );
        } else if ( nb_out > 2 ) {                       // trop de place : la queue revient A GAUCHE
            const int gap = nb_out - 2;
            for ( int i = j2 + 1; i < nb; ++i ) move( i - gap, i );
        }
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

} // namespace sf::d2
