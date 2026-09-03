#pragma once

#include "AaBsp.h"
#include <cstdint>

namespace pd2d {

/// L'ARBRE QUI SE SOUVIENT DE L'ITERATION PRECEDENTE.
///
/// Dans une boucle de Newton, le meme diagramme est reconstruit une dizaine de fois sur des poids
/// qui bougent de moins en moins. La geometrie de la fin est donc presque celle du debut -- et rien
/// n'en etait garde. Deux souvenirs, tous les deux tenus dans des entiers, tous les deux VALIDES
/// QUOI QU'IL ARRIVE :
///
/// = LA LISTE DES VOISINS (`vois`), et c'est celui qui compte
///
/// La cellule finale de `k` a six ou sept cotes, et `Cell::cid` porte EXACTEMENT les germes qui les
/// portent. On les garde tous -- ou qu'ils soient dans l'arbre -- et on recoupe avec eux avant de
/// descendre. Si le voisinage n'a pas bouge, la cellule est alors DEJA la reponse quand le parcours
/// commence : chaque test de boite echoue au premier essai au lieu d'etre garde.
///
/// = UN BIT PAR GERME DE LA BOITE D'ORIGINE (`--memo-boite`)
///
/// La variante ou l'on ne retient que les voisins de SA PROPRE FEUILLE, un bit chacun. Elle ne peut
/// pas payer, et c'est demontrable : le parcours descend FILS LE PLUS PROCHE EN PREMIER, donc la
/// toute premiere feuille atteinte est celle du germe. Ses germes sont donc proposes AVANT le
/// moindre test de boite exterieure -- une fois la feuille balayee, la cellule est la meme, qu'on
/// ait rejoue ou non, parce qu'une cellule est l'INTERSECTION de ses demi-plans. Le masque ne
/// reordonne que l'interieur de la premiere feuille, et ne peut changer AUCUNE reponse de
/// `may_be_cut`. Il est garde pour que la mesure existe.
///
/// = L'INDICE DU COUPABLE
///
/// Une cellule vide coute aujourd'hui un parcours complet avant qu'un germe ne la vide. On garde
/// donc QUI l'a videe, et on recommence par lui : la coupe rend `empty`, `cut_with` rend `false`,
/// et la construction s'arrete a la premiere coupe. C'est le pas de recul de l'amortissement qui
/// paie ca -- il y en a 89 sur le nuage de lignes, et un pas refuse l'est justement parce que des
/// cellules se sont videes.
///
/// = POURQUOI C'EST SUR
///
/// Une cellule est l'INTERSECTION de tous ses demi-plans. Changer l'ORDRE des coupes ne change donc
/// pas le polygone -- au bit pres si l'ordre des coupes effectives est le meme, a l'arrondi sinon.
/// Un souvenir perime ne rend pas un resultat faux : il rend une coupe inutile, qui repond
/// `unchanged`. Il n'y a rien a invalider.
struct AaBspMemo {
    AaBsp tr;

    std::vector<SI>       pos;      ///< id d'origine -> place dans l'ordre de l'arbre
    std::vector<SI>       lbeg;     ///< place -> debut de SA feuille (l'origine des bits)
    std::vector<SI>       lend;
    mutable std::vector<uint32_t> masque;   ///< les voisins de la feuille, un bit chacun
    mutable std::vector<SI>       vois;     ///< les PLACES des voisins, `max_vois` par germe
    mutable std::vector<uint8_t>  nvois;    ///< combien sont valides
    mutable std::vector<SI>       coupable; ///< qui a vide la cellule, ou `-1`

    bool liste = true;              ///< le souvenir des VOISINS, ou qu'ils soient
    bool bits = false;              ///< ... ou seulement ceux de la boite d'origine
    bool vides = true;              ///< le souvenir du coupable
    bool saute = true;              ///< ne pas representer a la coupe ce qui vient d etre rejoue
    mutable bool actif = false;     ///< rien a rejouer tant qu'une passe n'a pas eu lieu

    /// LES COMPTEURS SONT COMPILES DEHORS, et ce n'est pas de la coquetterie : membres non
    /// atomiques, ils mettent quand meme une ligne de cache PARTAGEE sur le chemin chaud, et huit
    /// threads qui s'y incrementent se la volent. Mesure de l'instrumentation elle-meme : le temps
    /// de diagramme du nuage de lignes passait de 7.69 a 8.34 s -- 8 % -- sur un compteur qui ne
    /// sert a rien. Passer `compte` a `true` et recompiler pour les lire, a `--threads 1`.
    static constexpr bool compte = false;
    mutable long long nb_rejoue = 0;    ///< coupes rejouees
    mutable long long nb_vide = 0;      ///< fois ou le coupable a vide la cellule tout de suite

    static constexpr const char *name = "memo";
    static constexpr SI max_bits = 32;
    static constexpr SI max_vois = 8;   ///< 5.97 cotes en moyenne, la queue est courte

    TF seed_x( SI k ) const { return tr.px[ k ]; }
    TF seed_y( SI k ) const { return tr.py[ k ]; }
    TF seed_w( SI k ) const { return tr.seed_w( k ); }
    SI seed_id( SI k ) const { return tr.order[ k ]; }
    SI nb_seeds() const { return tr.nb_seeds(); }

    void build( const TF *X, const TF *Y, const TF *W, SI n, SI leaf ) {
        tr.build( X, Y, W, n, leaf );
        pos.resize( n );
        lbeg.resize( n );
        lend.resize( n );
        for ( SI k = 0; k < n; ++k )
            pos[ tr.order[ k ] ] = k;
        for ( const AaBsp::Node &nd : tr.nodes )
            if ( nd.right < 0 )
                for ( SI k = nd.beg; k < nd.end; ++k ) { lbeg[ k ] = nd.beg; lend[ k ] = nd.end; }
        masque.assign( n, 0 );
        vois.assign( size_t( n ) * max_vois, -1 );
        nvois.assign( n, 0 );
        coupable.assign( n, -1 );
        actif = false;
        if ( tr.leaf_size > max_bits )                  // le masque ne tiendrait pas
            bits = false;
    }

    /// Rien n'est rejoue tant qu'une passe complete n'a pas rempli les souvenirs.
    void arme() const { actif = true; }

    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        const SI i0 = tr.order[ k0 ];

        // CE QUI A DEJA ETE REJOUE. Sans cette liste, le parcours represente les memes germes a la
        // coupe : chacun est alors PAYE DEUX FOIS, une fois pour de bon et une fois pour s'entendre
        // repondre `unchanged` -- et un `unchanged` n'est pas gratuit, il balaie tous les sommets.
        // La liste tient en registres (quatre a six entrees), donc la comparer coute moins que le
        // balayage qu'elle evite.
        SI deja[ max_bits + max_vois + 1 ];
        SI nd = 0;

        auto cw = [ & ]( TF x, TF y, TF w, SI id ) {
            if ( cut_with( x, y, w, id ) )
                return true;
            coupable[ k0 ] = id;                        // c'est lui qui a vide la cellule
            return false;
        };
        auto cw2 = [ & ]( TF x, TF y, TF w, SI id ) {   // le meme, mais qui saute le deja-vu
            for ( SI i = 0; i < nd; ++i )
                if ( deja[ i ] == id )
                    return true;
            return cw( x, y, w, id );
        };

        if ( actif ) {
            // ---- le coupable d'abord : s'il vide encore, on sort a la premiere coupe
            if ( vides ) {
                const SI j = coupable[ k0 ];
                if ( j >= 0 && j != i0 ) {
                    const SI p = pos[ j ];
                    deja[ nd++ ] = j;
                    if ( ! cw( tr.px[ p ], tr.py[ p ], tr.seed_w( p ), j ) ) {
                        if constexpr ( compte ) ++nb_vide;
                        return;
                    }
                }
            }
            // ---- puis les voisins connus, ou qu'ils soient
            if ( liste ) {
                const SI *v = &vois[ size_t( k0 ) * max_vois ];
                const SI nv = nvois[ k0 ];
                for ( SI i = 0; i < nv; ++i ) {
                    const SI p = v[ i ];
                    const SI id = tr.order[ p ];
                    if ( id == i0 )
                        continue;
                    deja[ nd++ ] = id;
                    if ( ! cw( tr.px[ p ], tr.py[ p ], tr.seed_w( p ), id ) )
                        return;
                }
            }
            // ---- ou seulement ceux de la boite d'origine
            if ( bits ) {
                const SI b = lbeg[ k0 ];
                for ( uint32_t m = masque[ k0 ]; m; m &= m - 1 ) {
                    const SI u = b + __builtin_ctz( m );
                    const SI id = tr.order[ u ];
                    if ( id == i0 )
                        continue;
                    deja[ nd++ ] = id;
                    if ( ! cw( tr.px[ u ], tr.py[ u ], tr.seed_w( u ), id ) )
                        return;
                }
            }
            if constexpr ( compte ) nb_rejoue += nd;
        }

        if ( nd && saute )
            tr.for_each_candidate( k0, may_cut, cw2, reach2 );
        else
            tr.for_each_candidate( k0, may_cut, cw, reach2 );
    }

    /// Ce qu'on retient de la cellule finie. `cid` porte deja EXACTEMENT les germes qui ont un cote
    /// dans la cellule -- c'est l'invariant de `Cell` -- donc il n'y a rien a compter pendant la
    /// construction : on lit six ou sept entrees a la fin.
    template<class Cell>
    void note_cell( const Cell &c, SI k0 ) const {
        if ( ! c.nb )                                   // cellule vide : le coupable est deja note,
            return;                                     // et l'ancien masque reste valide
        coupable[ k0 ] = -1;
        if ( liste ) {
            SI *v = &vois[ size_t( k0 ) * max_vois ];
            SI nv = 0;
            for ( SI i = 0; i < c.nb && nv < max_vois; ++i )
                if ( c.cid[ i ] >= 0 )
                    v[ nv++ ] = pos[ c.cid[ i ] ];
            nvois[ k0 ] = uint8_t( nv );
        }
        if ( ! bits )
            return;
        const SI b = lbeg[ k0 ], e = lend[ k0 ];
        uint32_t m = 0;
        for ( SI v = 0; v < c.nb; ++v ) {
            const SI id = c.cid[ v ];
            if ( id < 0 )
                continue;
            const SI p = pos[ id ];
            if ( p >= b && p < e )
                m |= uint32_t( 1 ) << ( p - b );
        }
        masque[ k0 ] = m;
    }
};

} // namespace pd2d
