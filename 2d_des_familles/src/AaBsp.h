#pragma once

#include "WeightMajorant.h"
#include "common.h"
#include <algorithm>
#include <vector>

namespace pd2d {

/// UN BSP ALIGNE SUR LES AXES : coupes MEDIANES sur l'axe le plus long, une poignee de germes par
/// feuille.
///
/// = L'accelerateur porte SON PARCOURS
///
/// `for_each_candidate` est ici, et pas dans une classe `Walk` a part. Le concept qu'il fallait
/// sinon exposer -- `root`, `node`, `is_leaf`, `lo_x`..., `left`, `right`, `nearness`, `emit_leaf`,
/// `w_max` -- decrivait un arbre binaire a boites, donc ne se generalisait a rien : la grille ne
/// pouvait pas l'implementer, et le parcours de grille ne pouvait pas s'appliquer a un arbre. Un
/// accelerateur ne doit qu'une chose a l'appelant, PROPOSER DES GERMES, et c'est sa seule methode.
///
/// = La numerotation est en PREORDRE, et c'est le point
///
/// Le fils gauche est en `n + 1` -- JUSTE A COTE -- et le droit est stocke. En numerotation en tas
/// (`2n+1` / `2n+2`), un chemin racine -> feuille saute vers des adresses qui DOUBLENT a chaque
/// niveau, et le pire est en bas de l'arbre, la ou les niveaux sont les plus gros et les plus
/// visites. En preordre les sauts RETRECISSENT en descendant : un sous-arbre est contigu.
///
/// = Le noeud tient dans UNE ligne de cache
///
/// La boite, le majorant de poids, la tranche, le fils droit : tout ce qu'une visite demande est
/// CONTIGU. La marche est du pointer-chasing, donc ce qui coute n'est pas le nombre d'octets lus
/// mais le nombre de LIGNES touchees.
struct AaBsp {
    struct Node {
        TF lo[ 2 ], hi[ 2 ];    ///< la boite englobante du sous-arbre
        WMaj wm;                ///< MAJORANT AFFINE des poids du sous-arbre (voir `may_be_cut`)
        SI beg, end;            ///< sa tranche dans `order` / `px` / `py`
        SI right;               ///< le fils droit ; `< 0` dit FEUILLE (le gauche est `n + 1`)
    };
    static_assert( sizeof( Node ) <= 64, "un noeud doit tenir dans une ligne de cache" );

    std::vector<Node> nodes;    ///< en PREORDRE
    std::vector<SI>   order;    ///< la permutation des germes
    std::vector<TF>   px, py;   ///< les positions PERMUTEES, lues d'un seul tenant par feuille
    std::vector<TF>   pw;       ///< les poids permutes (vide = cas euclidien)
    SI                leaf_size = 10;

    static constexpr const char *name = "bsp";

    TF seed_x( SI k ) const { return px[ k ]; }
    TF seed_y( SI k ) const { return py[ k ]; }
    TF seed_w( SI k ) const { return pw.empty() ? TF( 0 ) : pw[ k ]; }
    SI seed_id( SI k ) const { return order[ k ]; }
    SI nb_seeds() const { return SI( order.size() ); }

    /// LES CANDIDATS : profondeur d'abord, fils le plus proche en premier, sur une pile explicite.
    ///
    /// Bornee par la PROFONDEUR : a chaque niveau on depile un noeud et on en empile deux, donc la
    /// pile ne depasse jamais `depth + 1` -- allouable sur le cadre, sans rien a deviner.
    ///
    /// Le plus proche est empile EN DERNIER, donc depile en PREMIER : la premiere feuille atteinte
    /// est celle du germe lui-meme, la cellule retrecit tout de suite, et tout le reste est elague
    /// contre une cellule deja petite. C'est ce qui rend la pile SUFFISANTE la ou il faudrait
    /// sinon une file de priorite -- ce qu'on a essaye (`WalkBestFirst`, ce que fait `SpZGrid`) et
    /// qui perdait : 0.279 s contre 0.246. Le code est parti avec le concept `Walk`.
    ///
    /// Le test est fait a la SORTIE de la pile et non a l'entree, donc contre la cellule TELLE
    /// QU'ELLE EST : chaque coupe faite depuis l'empilement rend la reponse « non » plus probable.
    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        for_each_candidate_at( px[ k0 ], py[ k0 ], order[ k0 ], may_cut, cut_with, reach2 );
    }

    /// Le meme, pour un germe donne par sa POSITION et son identite plutot que par sa place dans
    /// l'arbre -- donc pour un germe qui n'y est PAS. C'est ce que demande une pre-passe : marcher
    /// un arbre construit sur un sous-ensemble, pour un germe qui n'en fait pas forcement partie.
    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate_at( TF p0x, TF p0y, SI i0, MayCut &&may_cut, CutWith &&cut_with,
                                Reach2 &&reach2 ) const {
        for_each_candidate_from( 0, p0x, p0y, i0, may_cut, cut_with, reach2 );
    }

    /// Le meme, borne a UN SOUS-ARBRE. Ce que ca separe : la granularite de l'INDEX (quel bloc
    /// regarder) de celle de l'ELAGAGE (quoi tester dedans). Un index par paquets veut des paquets
    /// gros -- pour la memoire et pour amortir sa preparation -- alors que l'elagage veut des
    /// boites fines. Sans ca, un paquet retenu fait tester ses `rho` germes en bloc.
    ///
    /// Rend `false` si `cut_with` a demande l'arret (cellule vide) : l'appelant doit alors sortir.
    template<class MayCut, class CutWith, class Reach2>
    bool for_each_candidate_from( SI root, TF p0x, TF p0y, SI i0, MayCut &&may_cut,
                                  CutWith &&cut_with, Reach2 && ) const {
        SI stack[ 64 ];
        SI top = 0;
        stack[ top++ ] = root;

        while ( top > 0 ) {
            const SI h = stack[ --top ];
            const Node &nd = nodes[ h ];

            if ( ! may_cut( nd.lo[ 0 ], nd.lo[ 1 ], nd.hi[ 0 ], nd.hi[ 1 ], nd.wm ) )
                continue;

            if ( nd.right < 0 ) {
                for ( SI k = nd.beg; k < nd.end; ++k ) {
                    const SI id = order[ k ];
                    if ( id != i0 && ! cut_with( px[ k ], py[ k ], seed_w( k ), id ) )
                        return false;
                }
                continue;
            }

            const SI l = h + 1, r = nd.right;           // PREORDRE : le gauche est juste a cote
            if ( nearness( l, p0x, p0y ) <= nearness( r, p0x, p0y ) ) {
                stack[ top++ ] = r;
                stack[ top++ ] = l;
            } else {
                stack[ top++ ] = l;
                stack[ top++ ] = r;
            }
        }
        return true;
    }

    void build( const TF *X, const TF *Y, const TF *W, SI n, SI leaf ) {
        build_sel( X, Y, W, nullptr, n, leaf );
    }

    /// L'arbre sur la SELECTION `idx` (`nullptr` = tous). `order` part des indices d'origine et
    /// n'est ensuite que permute, donc `seed_id` rend l'identite dans le nuage ENTIER meme quand
    /// l'arbre n'en porte qu'un seizieme -- sans quoi un germe de la selection ne pourrait pas se
    /// reconnaitre lui-meme et se couperait avec son propre plan.
    void build_sel( const TF *X, const TF *Y, const TF *W, const SI *idx, SI n, SI leaf ) {
        leaf_size = leaf;
        order.resize( n );
        for ( SI i = 0; i < n; ++i )
            order[ i ] = idx ? idx[ i ] : i;

        // les positions sont PERMUTEES a cote des indices, pas relues a travers eux : une feuille
        // se lit alors d'un seul tenant, la ou une indirection en ferait une collecte eparse.
        px.resize( n ); py.resize( n );
        if ( W )
            pw.resize( n );

        nodes.clear();
        nodes.reserve( 2 * ( n / std::max<SI>( leaf, 1 ) + 1 ) );
        _build( X, Y, W, 0, n );

        for ( SI k = 0; k < n; ++k ) {
            px[ k ] = X[ order[ k ] ];
            py[ k ] = Y[ order[ k ] ];
            if ( W )
                pw[ k ] = W[ order[ k ] ];
        }
    }

protected:                                          // `ObBsp` reprend `Node`, `nearness` et la boucle
    /// Carre de la distance du point a la boite du noeud -- 0 dedans. UNIQUEMENT une cle d'ordre,
    /// donc volontairement ce bon marche : le vrai test balaie les sommets de la cellule, et le
    /// payer sur les deux fils juste pour choisir lequel regarder d'abord doublerait la marche.
    TF nearness( SI n, TF x, TF y ) const {
        const Node &nd = nodes[ n ];
        const TF ex = x < nd.lo[ 0 ] ? nd.lo[ 0 ] - x : ( x > nd.hi[ 0 ] ? x - nd.hi[ 0 ] : TF( 0 ) );
        const TF ey = y < nd.lo[ 1 ] ? nd.lo[ 1 ] - y : ( y > nd.hi[ 1 ] ? y - nd.hi[ 1 ] : TF( 0 ) );
        return ex * ex + ey * ey;
    }

    /// Recursif, donc PREORDRE par construction : le noeud est pousse avant ses fils, le fils
    /// gauche juste apres lui.
    SI _build( const TF *X, const TF *Y, const TF *W, SI beg, SI end ) {
        const SI me = SI( nodes.size() );
        nodes.push_back( Node{} );

        TF lox = X[ order[ beg ] ], hix = lox, loy = Y[ order[ beg ] ], hiy = loy;
        for ( SI k = beg + 1; k < end; ++k ) {
            const TF x = X[ order[ k ] ], y = Y[ order[ k ] ];
            lox = x < lox ? x : lox;  hix = x > hix ? x : hix;
            loy = y < loy ? y : loy;  hiy = y > hiy ? y : hiy;
        }
        nodes[ me ].lo[ 0 ] = lox; nodes[ me ].lo[ 1 ] = loy;
        nodes[ me ].hi[ 0 ] = hix; nodes[ me ].hi[ 1 ] = hiy;
        nodes[ me ].beg = beg;
        nodes[ me ].end = end;

        // le MAJORANT des poids du sous-arbre. Sans poids il reste a zero et tous ses termes
        // disparaissent du test.
        if ( W )
            nodes[ me ].wm = weight_majorant( beg, end, [ & ]( SI k, TF &x, TF &y, TF &w ) {
                const SI i = order[ k ];
                x = X[ i ]; y = Y[ i ]; w = W[ i ];
            } );

        const int ax = ( hix - lox ) >= ( hiy - loy ) ? 0 : 1;
        const TF span = ax ? hiy - loy : hix - lox;

        // `span <= 0` : tous les germes au meme endroit, aucune coupe ne les separerait, et
        // insister ferait une descente sans fin.
        if ( end - beg <= leaf_size || ! ( span > 0 ) ) {
            nodes[ me ].right = -1;
            return me;
        }

        // la MEDIANE et non le milieu de la boite : c'est ce qui borne la profondeur par
        // `log2( n / leaf )` quelle que soit la distribution.
        const SI mid = beg + ( end - beg ) / 2;
        const TF *C = ax ? Y : X;
        std::nth_element( order.begin() + beg, order.begin() + mid, order.begin() + end,
                          [ & ]( SI a, SI b ) { return C[ a ] < C[ b ]; } );

        _build( X, Y, W, beg, mid );                    // == me + 1
        nodes[ me ].right = _build( X, Y, W, mid, end );
        return me;
    }
};

/// L'accelerateur qui n'accelere rien : tous les germes, dans l'ordre. Pas un bouche-trou -- c'est
/// l'ORACLE contre lequel les autres se verifient, et il fait de « pas d'acceleration » un cas
/// ordinaire du meme code plutot qu'une seconde implementation a tenir en phase.
struct EverySeed {
    std::vector<SI> order;
    std::vector<TF> px, py, pw;

    static constexpr const char *name = "all";

    TF seed_x( SI k ) const { return px[ k ]; }
    TF seed_y( SI k ) const { return py[ k ]; }
    TF seed_w( SI k ) const { return pw.empty() ? TF( 0 ) : pw[ k ]; }
    SI seed_id( SI k ) const { return order[ k ]; }
    SI nb_seeds() const { return SI( order.size() ); }

    /// Le majorant n'est meme pas demande : une borne qui ne sert jamais a sauter quoi que ce soit
    /// est une borne a ne pas calculer.
    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&, CutWith &&cut_with, Reach2 && ) const {
        const SI i0 = order[ k0 ], n = nb_seeds();
        for ( SI k = 0; k < n; ++k ) {
            const SI id = order[ k ];
            if ( id != i0 && ! cut_with( px[ k ], py[ k ], seed_w( k ), id ) )
                return;
        }
    }

    void build( const TF *X, const TF *Y, const TF *W, SI n, SI ) {
        order.resize( n ); px.resize( n ); py.resize( n );
        if ( W ) pw.resize( n );
        for ( SI i = 0; i < n; ++i ) {
            order[ i ] = i; px[ i ] = X[ i ]; py[ i ] = Y[ i ];
            if ( W ) pw[ i ] = W[ i ];
        }
    }
};

/// UN SEUL GERME, pose sur un arbre auquel il n'appartient pas -- ou auquel il appartient avec un
/// AUTRE poids. C'est un accelerateur comme un autre : il ne doit qu'une chose, proposer des
/// germes, et il delegue le parcours a l'arbre qu'il enveloppe.
///
/// Pourquoi passer par la plutot que par un `make_cell` parametre : le concept d'accelerateur
/// existe deja et ne coute rien, alors que factoriser `make_cell` coutait 3 % (cf. `PowerDiagram`).
struct OneSeed {
    const AaBsp &tr;
    TF x, y, w;
    SI id;

    TF seed_x( SI ) const { return x; }
    TF seed_y( SI ) const { return y; }
    TF seed_w( SI ) const { return w; }
    SI seed_id( SI ) const { return id; }
    SI nb_seeds() const { return 1; }

    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        tr.for_each_candidate_at( x, y, id, may_cut, cut_with, reach2 );
    }
};

} // namespace pd2d
