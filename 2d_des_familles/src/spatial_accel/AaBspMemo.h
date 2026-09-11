#pragma once

#include "spatial_accel/AaBsp.h"
#include <cstdint>

namespace pd {

/// L'ARBRE QUI SE SOUVIENT DE L'ITERATION PRECEDENTE.
///
/// Dans une boucle de Newton le meme diagramme est reconstruit une dizaine de fois sur des poids qui
/// bougent de moins en moins, et rien n'en etait garde. On garde donc, par germe, QUELS DIRACS ONT
/// COUPE -- sous la forme d'une poignee de couples ( feuille, masque ), un bit par dirac de la
/// feuille. Repris de `old_pd/src/cpp/sdot/PrevCutInfo.h`.
///
/// = LE MASQUE SERT DEUX FOIS, et c'est tout le mecanisme
///
///   1. EN PRE-PASSE, avant de descendre l'arbre : on coupe avec les diracs retenus de TOUTES les
///      feuilles memorisees. Si le voisinage n'a pas bouge, la cellule est alors deja la reponse ;
///   2. AU PARCOURS, quand la descente atteint une feuille : on applique le COMPLEMENT de son
///      masque, donc uniquement les diracs qui n'ont pas ete faits en pre-passe.
///
/// Le second point est ce qui rend le premier gratuit : aucun dirac n'est propose deux fois, et il
/// n'y a rien a chercher dirac par dirac -- une seule lecture de masque par feuille VISITEE.
///
/// = POURQUOI IL FAUT LES AUTRES FEUILLES
///
/// Un masque limite a la feuille du germe ne peut RIEN gagner, et c'est demontrable : le parcours
/// descend fils-le-plus-proche en premier, donc la premiere feuille atteinte est celle du germe ;
/// ses diracs sont proposes avant le moindre test de boite exterieure, et une cellule est
/// l'INTERSECTION de ses demi-plans, donc elle est la meme une fois la feuille balayee quel que
/// soit l'ordre. Mesure de cette variante-la : +3 %.
///
/// Ce sont les coupes venues des AUTRES feuilles qui changent tout : appliquees en pre-passe, elles
/// retrecissent la cellule AVANT le premier `may_be_cut` -- et c'est `may_be_cut` qui coute, 42
/// (uniforme) a 135 (lignes) fois par cellule.
///
/// = POURQUOI LE COMPLEMENT N'EST PAS QU'UNE OPTIMISATION
///
/// Une cellule est l'intersection de ses demi-plans, donc l'ordre des coupes ne change pas le
/// polygone. Mais recouper avec un plan DEJA applique n'est pas neutre : `Cell::cut` n'est pas
/// idempotente (les sommets d'intersection ne sont pas exactement sur leur plan, `s` y vaut ±1e-17,
/// et un `+1e-17` fait ajouter un sommet degenere). Mesure sans le complement : 116 305 coupes
/// debordent 64 sommets et Newton stagne. Le complement garantit qu'un dirac n'est propose qu'une
/// fois.
///
/// Un souvenir PERIME, lui, est inoffensif : il fait couper avec un germe qui n'est plus voisin,
/// donc une coupe qui ne retire rien, et le parcours fera le reste.
///
/// = CE QUE CA DONNE, ET POURQUOI C'EST SI PEU
///
/// Le souvenir marche : 5.38 coupes rejouees par cellule dans 2.80 feuilles, sur une cellule qui a
/// 5.97 cotes -- la pre-passe reconstruit donc bien le voisinage entier. Et pourtant les boites
/// testees ne tombent que de 50.3 a 48.9, soit 2.8 %.
///
/// La raison est structurelle : `may_be_cut` demande « cette boite peut-elle encore atteindre la
/// cellule ? ». Une boite qui contient un VRAI voisin passe ce test quoi qu'il arrive -- son plan
/// est tangent a la cellule, par definition. Arriver avec la cellule finale ne rend donc PAS ses
/// voisins rejetables : les boites qu'il faut visiter sont exactement celles des voisins, plus le
/// chemin de descente, et c'est deja l'essentiel des cinquante. On ne peut pas elaguer ce qu'on
/// doit de toute facon regarder.
struct AaBspMemo {
    static constexpr int dim = 2;

    AaBsp tr;

    std::vector<SI>       pos;      ///< id d'origine -> place dans l'ordre de l'arbre
    std::vector<SI>       lbeg;     ///< place -> debut de SA feuille (l'origine des bits)

    /// Les souvenirs, `max_feuilles` couples par germe. A plat plutot qu'en `vector` de `vector` :
    /// c'est lu a chaque cellule, donc ca doit tenir dans quelques lignes de cache contigues.
    mutable std::vector<SI>       fbeg;
    mutable std::vector<uint32_t> fmsk;
    mutable std::vector<uint8_t>  fnb;

    bool bits = true;               ///< rejouer les coupes retenues
    mutable bool actif = false;     ///< rien a rejouer tant qu'une passe n'a pas eu lieu

    /// LES COMPTEURS SONT COMPILES DEHORS. Membres non atomiques, ils mettent quand meme une ligne
    /// de cache PARTAGEE sur le chemin chaud, et huit threads qui s'y incrementent se la volent :
    /// mesure de l'instrumentation elle-meme, 7.69 -> 8.34 s a huit fils, 8 % sur un compteur qui
    /// ne sert a rien. Passer `compte` a `true` et recompiler pour les lire, a `--threads 1`.
    static constexpr bool compte = false;
    mutable long long nb_rejoue = 0, nb_feuilles = 0, nb_cell = 0, nb_boites = 0;

    static constexpr const char *name = "memo";
    static constexpr SI max_bits = 32;      ///< un masque est un `uint32_t`
    static constexpr SI max_feuilles = 8;   ///< 5.97 cotes en moyenne, donc rarement plus de feuilles

    Vec<2> seed( SI k ) const { return tr.seed( k ); }
    TF seed_x( SI k ) const { return tr.seed_x( k ); }
    TF seed_y( SI k ) const { return tr.seed_y( k ); }
    TF seed_w( SI k ) const { return tr.seed_w( k ); }
    SI seed_id( SI k ) const { return tr.order[ k ]; }
    SI nb_seeds() const { return tr.nb_seeds(); }

    void build( const TF *const *P, const TF *W, SI n, SI leaf ) {
        tr.build( P, W, n, leaf );
        pos.resize( n );
        lbeg.resize( n );
        for ( SI k = 0; k < n; ++k )
            pos[ tr.order[ k ] ] = k;
        for ( const AaBsp::Node &nd : tr.nodes )
            if ( nd.right < 0 )
                for ( SI k = nd.beg; k < nd.end; ++k )
                    lbeg[ k ] = nd.beg;
        fbeg.assign( size_t( n ) * max_feuilles, -1 );
        fmsk.assign( size_t( n ) * max_feuilles, 0 );
        fnb.assign( n, 0 );
        actif = false;
        if ( tr.leaf_size > max_bits )                  // un masque ne tiendrait pas
            bits = false;
    }

    /// Rien n'est rejoue tant qu'une passe complete n'a pas rempli les souvenirs.
    void arme() const { actif = true; }

    /// Le meme parcours que `AaBsp` -- meme pile, meme ordre fils-le-plus-proche -- precede de la
    /// pre-passe, et dont chaque feuille consulte son masque.
    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        const Vec<2> p0 = tr.seed( k0 );
        const SI i0 = tr.order[ k0 ];

        // le meme test d'eviction, compte quand on instrumente
        auto mc = [ & ]( Vec<2> lo, Vec<2> hi, const WMaj &wm ) {
            if constexpr ( compte ) ++nb_boites;
            return may_cut( lo, hi, wm );
        };

        const SI nf = ( actif && bits ) ? SI( fnb[ k0 ] ) : 0;
        const SI *fb = &fbeg[ size_t( k0 ) * max_feuilles ];
        const uint32_t *fm = &fmsk[ size_t( k0 ) * max_feuilles ];

        // RIEN A REJOUER : on rend la main a `AaBsp`, mot pour mot. Ce n'est pas une elegance,
        // c'est ce qui rend la mesure lisible -- porter une copie du parcours, meme identique,
        // coute 6 % (deux pointeurs de plus a garder vivants dans la boucle chaude, cf. la note de
        // `may_be_cut` sur le `this` qui coutait 7 %). La premiere passe, et toute passe sans
        // souvenir, doivent couter EXACTEMENT ce que coute `bsp`.
        if ( ! nf ) {
            if constexpr ( compte ) ++nb_cell;
            tr.for_each_candidate( k0, mc, cut_with, reach2 );
            return;
        }

        auto essaie = [ & ]( SI k ) {
            const SI id = tr.order[ k ];
            return id == i0 || cut_with( tr.seed( k ), tr.seed_w( k ), id );
        };

        // ---- 1. LA PRE-PASSE : les coupes retenues, de toutes les feuilles memorisees
        for ( SI i = 0; i < nf; ++i ) {
            const SI b = fb[ i ];
            for ( uint32_t m = fm[ i ]; m; m &= m - 1 )
                if ( ! essaie( b + __builtin_ctz( m ) ) )
                    return;
        }
        if constexpr ( compte ) {
            ++nb_cell;
            nb_feuilles += nf;
            for ( SI i = 0; i < nf; ++i )
                nb_rejoue += __builtin_popcount( fm[ i ] );
        }

        // ---- 2. LE PARCOURS, chaque feuille n'appliquant que le COMPLEMENT de son masque
        SI stack[ 64 ];
        SI top = 0;
        stack[ top++ ] = 0;
        while ( top > 0 ) {
            const SI h = stack[ --top ];
            const AaBsp::Node &nd = tr.nodes[ h ];

            if ( ! mc( vec_of<2>( nd.lo ), vec_of<2>( nd.hi ), nd.wm ) )
                continue;

            if ( nd.right < 0 ) {
                // le masque de CETTE feuille : une seule recherche par feuille visitee, sur une
                // poignee d'entrees -- et non une recherche par dirac propose.
                //
                // Les DEUX boucles sont ecrites separement, et ce n'est pas de la redondance : une
                // feuille sans masque -- la majorite -- doit couter exactement ce qu'elle coute
                // dans `AaBsp`, sans un decalage ni un `et` de plus par dirac. Mesure de ce seul
                // detail sur le nuage de lignes : 6.90 -> 7.30 s, soit 6 %.
                uint32_t fait = 0;
                for ( SI i = 0; i < nf; ++i )
                    if ( fb[ i ] == nd.beg ) { fait = fm[ i ]; break; }
                if ( fait ) {
                    for ( SI k = nd.beg; k < nd.end; ++k )
                        if ( ! ( ( fait >> ( k - nd.beg ) ) & 1 ) && ! essaie( k ) )
                            return;
                } else {
                    for ( SI k = nd.beg; k < nd.end; ++k )
                        if ( ! essaie( k ) )
                            return;
                }
                continue;
            }

            const SI l = h + 1, r = nd.right;           // PREORDRE : le gauche est juste a cote
            if ( loin( l, p0 ) <= loin( r, p0 ) ) {
                stack[ top++ ] = r;
                stack[ top++ ] = l;
            } else {
                stack[ top++ ] = l;
                stack[ top++ ] = r;
            }
        }
    }

    /// Ce qu'on retient de la cellule finie. `cid` porte deja EXACTEMENT les germes qui ont un cote
    /// -- c'est l'invariant de `Cell` -- donc il n'y a rien a compter pendant la construction : on
    /// lit six ou sept entrees a la fin et on les range par feuille.
    template<class Cell>
    void note_cell( const Cell &c, SI k0 ) const {
        if ( ! bits || ! c.nb )                         // cellule vide : l'ancien souvenir reste
            return;                                     // valide, il ne fera que des coupes inutiles
        SI *fb = &fbeg[ size_t( k0 ) * max_feuilles ];
        uint32_t *fm = &fmsk[ size_t( k0 ) * max_feuilles ];
        SI nf = 0;
        for ( SI v = 0; v < c.nb; ++v ) {
            const SI id = c.cid[ v ];
            if ( id < 0 )                               // une arete du DOMAINE, pas un voisin
                continue;
            const SI p = pos[ id ], b = lbeg[ p ];
            SI i = 0;
            for ( ; i < nf && fb[ i ] != b; ++i )
                ;
            if ( i == nf ) {
                if ( nf == max_feuilles )               // au-dela on oublie : le parcours les
                    continue;                           // proposera, c'est tout
                fb[ nf ] = b;
                fm[ nf ] = 0;
                ++nf;
            }
            fm[ i ] |= uint32_t( 1 ) << ( p - b );
        }
        fnb[ k0 ] = uint8_t( nf );
    }

private:
    /// Carre de la distance du germe a la boite du noeud -- uniquement une cle d'ordre.
    TF loin( SI h, Vec<2> q ) const {
        const AaBsp::Node &nd = tr.nodes[ h ];
        const TF ex = q[ 0 ] < nd.lo[ 0 ] ? nd.lo[ 0 ] - q[ 0 ] : ( q[ 0 ] > nd.hi[ 0 ] ? q[ 0 ] - nd.hi[ 0 ] : TF( 0 ) );
        const TF ey = q[ 1 ] < nd.lo[ 1 ] ? nd.lo[ 1 ] - q[ 1 ] : ( q[ 1 ] > nd.hi[ 1 ] ? q[ 1 ] - nd.hi[ 1 ] : TF( 0 ) );
        return ex * ex + ey * ey;
    }
};

} // namespace pd
