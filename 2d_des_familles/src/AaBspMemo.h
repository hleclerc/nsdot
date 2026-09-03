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
    mutable std::vector<SI>       coupable; ///< qui a vide la cellule, ou `-1`

    bool bits = true;               ///< le souvenir des coupes de la boite d'origine
    bool vides = true;              ///< le souvenir du coupable
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
        coupable.assign( n, -1 );
        actif = false;
        if ( tr.leaf_size > max_bits )                  // le masque ne tiendrait pas
            bits = false;
    }

    /// Rien n'est rejoue tant qu'une passe complete n'a pas rempli les souvenirs.
    void arme() const { actif = true; }

    /// LE PARCOURS, avec la feuille du germe balayee EN DEUX PASSES.
    ///
    /// C'est une copie de celui de `AaBsp` -- meme pile, meme ordre fils-le-plus-proche -- avec une
    /// seule difference : quand la feuille visitee est CELLE DU GERME, ses diracs sont proposes
    /// dans l'ordre du masque, ceux a un d'abord. Aucun dirac n'est propose deux fois, il n'y a
    /// rien a memoriser pendant la construction, et le cout est un decalage et un `et` logique par
    /// dirac de cette seule feuille.
    ///
    /// ESSAYE ET REJETE : rejouer les coupes retenues AVANT de lancer le parcours. C'etait le
    /// meme souvenir, mais applique au mauvais endroit : le parcours ne sait pas qu'on vient de
    /// couper, il repropose les memes diracs, et il faut alors tenir une liste des deja-rejoues
    /// pour ne pas les recouper -- 25 a 60 comparaisons par cellule. Pire, sans cette liste le
    /// resultat devient FAUX (voir plus bas). Ici la question ne se pose pas : le rejeu n'existe
    /// pas, seul l'ordre change.
    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        const TF p0x = tr.px[ k0 ], p0y = tr.py[ k0 ];
        const SI i0 = tr.order[ k0 ];
        const SI b0 = lbeg[ k0 ];
        const uint32_t m = ( actif && bits ) ? masque[ k0 ] : 0;

        // le seul germe qu'il faille eventuellement ne pas represente : le coupable, coupe avant
        // tout le monde parce qu'il vide peut-etre encore la cellule. UN entier, une comparaison.
        SI saut = -1;

        auto essaie = [ & ]( SI k ) {
            const SI id = tr.order[ k ];
            if ( id == i0 || id == saut )
                return true;
            if ( cut_with( tr.px[ k ], tr.py[ k ], tr.seed_w( k ), id ) )
                return true;
            coupable[ k0 ] = id;                        // c'est lui qui a vide la cellule
            return false;
        };
        auto proche = [ & ]( SI h ) {
            const AaBsp::Node &nd = tr.nodes[ h ];
            const TF ex = p0x < nd.lo[ 0 ] ? nd.lo[ 0 ] - p0x
                        : ( p0x > nd.hi[ 0 ] ? p0x - nd.hi[ 0 ] : TF( 0 ) );
            const TF ey = p0y < nd.lo[ 1 ] ? nd.lo[ 1 ] - p0y
                        : ( p0y > nd.hi[ 1 ] ? p0y - nd.hi[ 1 ] : TF( 0 ) );
            return ex * ex + ey * ey;
        };

        if ( actif && vides ) {
            const SI j = coupable[ k0 ];
            if ( j >= 0 && j != i0 ) {
                const SI p = pos[ j ];
                if ( ! cut_with( tr.px[ p ], tr.py[ p ], tr.seed_w( p ), j ) ) {
                    coupable[ k0 ] = j;
                    if constexpr ( compte ) ++nb_vide;
                    return;
                }
                saut = j;
            }
        }

        SI stack[ 64 ];
        SI top = 0;
        stack[ top++ ] = 0;
        while ( top > 0 ) {
            const SI h = stack[ --top ];
            const AaBsp::Node &nd = tr.nodes[ h ];

            if ( ! may_cut( nd.lo[ 0 ], nd.lo[ 1 ], nd.hi[ 0 ], nd.hi[ 1 ], nd.wm ) )
                continue;

            if ( nd.right < 0 ) {
                if ( m && nd.beg == b0 ) {
                    if constexpr ( compte ) nb_rejoue += __builtin_popcount( m );
                    for ( SI k = nd.beg; k < nd.end; ++k )
                        if ( ( m >> ( k - nd.beg ) ) & 1 )
                            if ( ! essaie( k ) ) return;
                    for ( SI k = nd.beg; k < nd.end; ++k )
                        if ( ! ( ( m >> ( k - nd.beg ) ) & 1 ) )
                            if ( ! essaie( k ) ) return;
                } else {
                    for ( SI k = nd.beg; k < nd.end; ++k )
                        if ( ! essaie( k ) ) return;
                }
                continue;
            }

            const SI l = h + 1, r = nd.right;           // PREORDRE : le gauche est juste a cote
            if ( proche( l ) <= proche( r ) ) { stack[ top++ ] = r; stack[ top++ ] = l; }
            else                              { stack[ top++ ] = l; stack[ top++ ] = r; }
        }
    }

    /// Ce qu'on retient de la cellule finie. `cid` porte deja EXACTEMENT les germes qui ont un cote
    /// dans la cellule -- c'est l'invariant de `Cell` -- donc il n'y a rien a compter pendant la
    /// construction : on lit six ou sept entrees a la fin.
    template<class Cell>
    void note_cell( const Cell &c, SI k0 ) const {
        if ( ! c.nb )                                   // cellule vide : le coupable est deja note,
            return;                                     // et l'ancien masque reste valide
        coupable[ k0 ] = -1;
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
