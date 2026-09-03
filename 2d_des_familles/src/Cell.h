#pragma once

#include "common.h"
#include <cmath>

namespace pd2d {

/// LA CELLULE 2D, en tampons FIXES : pas une allocation, pas un `std::vector`.
///
/// `max_nb_vertices = 32` est un choix de banc, pas une verite : une cellule de Laguerre en 2D sur
/// un nuage uniforme en a six ou sept en moyenne, et une queue tres fine au-dela. On borne pour
/// que la cellule tienne SUR LA PILE et se copie sans indirection ; quand la borne est atteinte,
/// `cut` rend `overflow` et laisse la cellule intacte -- l'appelant decide (ici : on compte, et le
/// banc le signale, plutot que de rendre un resultat faux en silence).
///
/// = L'INVARIANT, et tout en decoule
///
///     la coupe `i` porte l'arete [ v_i, v_i+1 ]        (donc nb_cuts == nb_vertices)
///
/// C'est lui qui permet de reecrire la geometrie ET la representation en demi-espaces en UN seul
/// passage cyclique, sans rien tabuler.
///
/// = SoA, et pourquoi
///
/// Les abscisses d'un cote, les ordonnees de l'autre. Le noyau du clip est
/// `s[ i ] = dx * vx[ i ] + dy * vy[ i ] - off` : en SoA c'est deux chargements contigus et un FMA,
/// la forme exacte qu'un vectoriseur prend. En AoS (`x, y, x, y, ...`) il faudrait desentrelacer.
/// `CellAoS` existe a cote pour que ce soit MESURE et non suppose.
/// `MaxNv` : combien de sommets une cellule peut porter. Ce n'est pas qu'une borne de surete, c'est
/// la TAILLE DE L'OBJET -- six tableaux de `MaxNv` elements, sur la pile, plus le `s[]` du clip.
/// A 32 en FP64 la cellule fait ~1.4 Ko ; a 16, 0.7 Ko. Ce qui se joue est de savoir si ce qui
/// reste chaud en L1 vaut ce qu'on perd en cellules qui debordent (une cellule de Laguerre 2D
/// uniforme en a six ou sept en moyenne, mais la queue est longue).
template<int MaxNv>
struct CellSoAT {
    static constexpr int max_nb_vertices = MaxNv;

    /// ALIGNES SUR UNE LIGNE DE CACHE, et ce n'est pas cosmetique : sans garantie d'alignement, un
    /// vectoriseur qui veut charger quatre `double` d'un coup doit soit emettre des chargements non
    /// alignes, soit prologuer la boucle jusqu'a tomber juste. Avec `alignas( 64 )` il sait, a la
    /// COMPILATION, que `vx` commence sur une frontiere -- et 32 `double` font exactement 4 lignes,
    /// donc `vy`, `cdx`... commencent toutes sur une frontiere elles aussi.
    SI nb = 0;                                          ///< sommets == coupes
    alignas( 64 ) TF vx[ max_nb_vertices ];             ///< les sommets, en ordre CYCLIQUE
    alignas( 64 ) TF vy[ max_nb_vertices ];
    alignas( 64 ) TF cdx[ max_nb_vertices ];            ///< la coupe `i` : `cd . x <= co`
    alignas( 64 ) TF cdy[ max_nb_vertices ];
    alignas( 64 ) TF co [ max_nb_vertices ];
    alignas( 64 ) SI cid[ max_nb_vertices ];            ///< le germe d'en face, ou < 0 pour le domaine

    /// le carre unite, en ordre direct -- l'invariant impose que la coupe `i` porte l'arete
    /// sortante du sommet `i`.
    void init_as_unit_square() {
        nb = 4;
        vx[ 0 ] = 0; vy[ 0 ] = 0;
        vx[ 1 ] = 1; vy[ 1 ] = 0;
        vx[ 2 ] = 1; vy[ 2 ] = 1;
        vx[ 3 ] = 0; vy[ 3 ] = 1;
        //            arete 0->1 : y >= 0    1->2 : x <= 1     2->3 : y <= 1     3->0 : x >= 0
        const TF dx[ 4 ] = {  0,  1,  0, -1 };
        const TF dy[ 4 ] = { -1,  0,  1,  0 };
        const TF of[ 4 ] = {  0,  1,  1,  0 };
        for ( SI i = 0; i < 4; ++i ) {
            cdx[ i ] = dx[ i ]; cdy[ i ] = dy[ i ]; co[ i ] = of[ i ]; cid[ i ] = -1;
        }
    }

    /// LA COUPE, EN PLACE. Intersecte avec `dx * x + dy * y <= off`.
    ///
    /// Ce qu'elle evite n'est pas le calcul mais la RECOPIE : le clip habituel reecrit la cellule
    /// entiere dans un second tampon, y compris les sommets que la coupe ne touche pas. Ici seuls
    /// les deux points d'intersection sont ecrits, plus le decalage qu'impose le changement de
    /// taille.
    ///
    /// Ce qui la rend possible : la sortie fait EXACTEMENT `nb - nb_dehors + 2` sommets, et les
    /// sommets conserves forment UNE plage cyclique -- l'exterieur d'un convexe coupe par un
    /// demi-espace est d'un seul tenant. Le sens du decalage est donc connu avant de bouger quoi
    /// que ce soit.
    CutResult cut( TF dx, TF dy, TF off, SI cut_id ) {
        // ---- les produits scalaires, dans une boucle QUI NE FAIT QUE CA : pas de branche, pas de
        // compteur, un `i` qui avance. C'est la seule forme du clip qu'un vectoriseur sache
        // prendre, et c'est pour l'isoler qu'ils sont ranges dans `s` au lieu d'etre recalcules.
        alignas( 64 ) TF s[ max_nb_vertices ];
        for ( SI i = 0; i < nb; ++i )
            s[ i ] = dx * vx[ i ] + dy * vy[ i ] - off;

        // ---- combien dehors, et OU commence la plage exterieure. `i1` est UNIQUE (voir plus haut).
        SI nb_out = 0, i1 = 0;
        bool prev_out = s[ nb - 1 ] > 0;
        for ( SI i = 0; i < nb; ++i ) {
            const bool out = s[ i ] > 0;
            if ( out ) {
                ++nb_out;
                if ( ! prev_out )
                    i1 = i;
            }
            prev_out = out;
        }

        if ( nb_out == 0 )
            return CutResult::unchanged;
        if ( nb_out == nb ) {
            nb = 0;
            return CutResult::empty;
        }

        const SI nb_in = nb - nb_out;
        const SI new_nb = nb_in + 2;
        if ( new_nb > max_nb_vertices )
            return CutResult::overflow;         // la cellule reste INTACTE : rien n'a encore bouge

        const SI j0 = ( i1 + nb - 1 ) % nb;     // dernier DEDANS avant la plage
        const SI j2 = ( i1 + nb_out - 1 ) % nb; // dernier DEHORS
        const SI j3 = ( j2 + 1 ) % nb;          // premier DEDANS apres

        // ---- LES DEUX INTERSECTIONS, ancrees sur le sommet DEDANS.
        //
        // `t = s_in / ( s_in - s_out )` est dans `[ 0, 1 ]` par construction, et
        // `v_in + ( v_out - v_in ) * t` vaut EXACTEMENT `v_in` quand `t` vaut zero.
        //
        // L'autre forme, algebriquement identique, etait ecrite ici :
        //
        //     pax = ( s1 * vx[ j0 ] - s0 * vx[ i1 ] ) / ( s1 - s0 )
        //
        // pour etre SYMETRIQUE -- le meme point quel que soit le bout de l'arete par lequel on
        // commence. Mais la symetrie ne servait a rien (le sommet dedans est toujours designe en
        // premier, donc il n'y a jamais de choix a faire) et elle coutait la propriete qui compte :
        // avec `s_in == 0` elle rend `( s1 * v0 ) / s1`, qui n'est PAS `v0` en flottant. Le sommet
        // se decale alors d'un cran, parfois DE L'AUTRE COTE du plan -- et la suite suppose que les
        // sommets dehors forment une plage cyclique unique.
        //
        // MESURE : un plan applique DEUX FOIS (ce que `--pre-overlap` provoque exprès) donnait avec
        // l'ancienne forme une cellule fausse de 10 % sur un germe sur 100 000, et une somme des
        // aires a `1.000001886` -- assez petit pour passer inapercu, assez gros pour n'etre pas du
        // bruit. Avec celle-ci, l'ecart retombe a `2.2e-16`, et le banc gagne 3 % au passage : deux
        // divisions au lieu de quatre.
        const TF s0 = s[ j0 ], s1 = s[ i1 ], s2 = s[ j2 ], s3 = s[ j3 ];
        const TF ta = s0 / ( s0 - s1 );
        const TF pax = vx[ j0 ] + ( vx[ i1 ] - vx[ j0 ] ) * ta;
        const TF pay = vy[ j0 ] + ( vy[ i1 ] - vy[ j0 ] ) * ta;
        const TF tb = s3 / ( s3 - s2 );
        const TF pbx = vx[ j3 ] + ( vx[ j2 ] - vx[ j3 ] ) * tb;
        const TF pby = vy[ j3 ] + ( vy[ j2 ] - vy[ j3 ] ) * tb;

        // ce que `pb` PORTE : un morceau de l'ancienne arete `j2`. Lu MAINTENANT, `j2` etant dans
        // la plage donc sur le point d'etre ecrase.
        const TF bdx = cdx[ j2 ], bdy = cdy[ j2 ], bof = co[ j2 ];
        const SI bid = cid[ j2 ];

        auto move = [ & ]( SI dst, SI src ) {
            vx[ dst ] = vx[ src ]; vy[ dst ] = vy[ src ];
            cdx[ dst ] = cdx[ src ]; cdy[ dst ] = cdy[ src ];
            co[ dst ] = co[ src ]; cid[ dst ] = cid[ src ];
        };
        auto put = [ & ]( SI k, TF x, TF y, TF ddx, TF ddy, TF dof, SI did ) {
            vx[ k ] = x; vy[ k ] = y;
            cdx[ k ] = ddx; cdy[ k ] = ddy; co[ k ] = dof; cid[ k ] = did;
        };

        if ( i1 <= j2 ) {
            // la plage ne boucle pas : la sortie garde sa place, `v_0 .. v_j0`, `pa`, `pb`, la queue.
            if ( nb_out == 1 ) {                        // un cran de plus : la queue va A DROITE
                for ( SI i = nb; i > i1 + 1; --i )
                    move( i, i - 1 );
            } else if ( nb_out > 2 ) {                  // trop de place : la queue revient A GAUCHE
                const SI gap = nb_out - 2;
                for ( SI i = j2 + 1; i < nb; ++i )
                    move( i - gap, i );
            }                                           // `nb_out == 2` : rien a decaler
            put( i1 + 0, pax, pay, dx, dy, off, cut_id );
            put( i1 + 1, pbx, pby, bdx, bdy, bof, bid );
        } else {
            // la plage BOUCLE, donc l'interieur est contigu : `[ j3, j3 + nb_in )`. On l'amene en
            // `[ 2, 2 + nb_in )`, puis `pa` en 0 et `pb` en 1.
            if ( j3 >= 2 )
                for ( SI o = 0; o < nb_in; ++o )
                    move( 2 + o, j3 + o );
            else
                for ( SI o = nb_in - 1; o >= 0; --o )
                    move( 2 + o, j3 + o );
            put( 0, pax, pay, dx, dy, off, cut_id );
            put( 1, pbx, pby, bdx, bdy, bof, bid );
        }

        nb = new_nb;
        return CutResult::done;
    }

    /// L'AIRE, par la formule du lacet. Rien a enumerer : en 2D les sommets EN ORDRE CYCLIQUE sont
    /// deja la geometrie.
    TF measure() const {
        TF a = 0;
        for ( SI i = 0, j = nb - 1; i < nb; j = i++ )
            a += vx[ j ] * vy[ i ] - vx[ i ] * vy[ j ];
        return TF( 0.5 ) * ( a < 0 ? -a : a );
    }

    /// La BOITE de la cellule : ce qui rend l'elagage bon marche. Une cellule convexe est majoree
    /// par sa boite bien plus finement que par une sphere, et ca ne coute pas plus cher.
    void bounds( TF &lox, TF &loy, TF &hix, TF &hiy ) const {
        lox = hix = vx[ 0 ]; loy = hiy = vy[ 0 ];
        for ( SI i = 1; i < nb; ++i ) {
            lox = vx[ i ] < lox ? vx[ i ] : lox;  hix = vx[ i ] > hix ? vx[ i ] : hix;
            loy = vy[ i ] < loy ? vy[ i ] : loy;  hiy = vy[ i ] > hiy ? vy[ i ] : hiy;
        }
    }
};

using CellSoA = CellSoAT<32>;

} // namespace pd2d
