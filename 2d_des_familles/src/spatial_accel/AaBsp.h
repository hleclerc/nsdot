#pragma once

#include "geometry/WeightMajorant.h"
#include "util/common.h"
#include <algorithm>
#include <vector>

namespace pd {

/// UN BSP ALIGNE SUR LES AXES : coupes MEDIANES sur l'axe le plus long, une poignee de germes par
/// feuille. `D` est la dimension -- rien dans ce qui suit n'en depend autrement que par des boucles
/// `for d`.
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
/// = Le noeud tient dans une ligne de cache EN 2D, et plus en 3D
///
/// La boite, le majorant de poids, la tranche, le fils droit : tout ce qu'une visite demande est
/// contigu. La marche est du pointer-chasing, donc ce qui coute n'est pas le nombre d'octets lus
/// mais le nombre de LIGNES touchees -- et c'est la premiere chose que la 3D change, sans qu'on y
/// puisse rien de gratuit : `2 D` bornes en `double` plus `D` pentes font deja 60 octets a `D = 3`.
/// Le `static_assert` est donc conditionne a `D == 2` : en 3D on veut savoir qu'on paie deux
/// lignes, pas s'interdire de compiler.
template<int D>
struct AaBspT {
    static constexpr int dim = D;

    struct Node {
        TF lo[ D ], hi[ D ];    ///< la boite englobante du sous-arbre
        WMajT<D> wm;            ///< MAJORANT AFFINE des poids du sous-arbre (voir `may_be_cut`)
        SI beg, end;            ///< sa tranche dans `order` / `p`
        SI right;               ///< le fils droit ; `< 0` dit FEUILLE (le gauche est `n + 1`)
    };
    static_assert( D != 2 || sizeof( Node ) <= 64, "en 2D un noeud doit tenir dans une ligne de cache" );

    std::vector<Node> nodes;    ///< en PREORDRE
    std::vector<SI>   order;    ///< la permutation des germes
    std::vector<TF>   p[ D ];   ///< les positions PERMUTEES, lues d'un seul tenant par feuille
    std::vector<TF>   pw;       ///< les poids permutes (vide = cas euclidien)
    SI                leaf_size = 10;

    static constexpr const char *name = "bsp";

    /// EN INITIALISATION D'AGREGAT : voir `vec_of`. C'est la methode la plus chaude de tout le banc
    /// -- une fois par germe propose a la coupe -- et la remplir coordonnee par coordonnee coutait
    /// 5 a 7 %.
    Vec<D> seed( SI k ) const {
        if constexpr ( D == 2 )
            return { p[ 0 ][ k ], p[ 1 ][ k ] };
        else if constexpr ( D == 3 )
            return { p[ 0 ][ k ], p[ 1 ][ k ], p[ 2 ][ k ] };
        else {
            Vec<D> r;
            for ( int d = 0; d < D; ++d ) r[ d ] = p[ d ][ k ];
            return r;
        }
    }
    TF seed_c( SI k, int d ) const { return p[ d ][ k ]; }
    TF seed_x( SI k ) const { return p[ 0 ][ k ]; }
    TF seed_y( SI k ) const { return p[ 1 ][ k ]; }
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
        for_each_candidate_at( seed( k0 ), order[ k0 ], may_cut, cut_with, reach2 );
    }

    /// Le meme, pour un germe donne par sa POSITION et son identite plutot que par sa place dans
    /// l'arbre -- donc pour un germe qui n'y est PAS. C'est ce que demande une pre-passe : marcher
    /// un arbre construit sur un sous-ensemble, pour un germe qui n'en fait pas forcement partie.
    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate_at( Vec<D> p0, SI i0, MayCut &&may_cut, CutWith &&cut_with,
                                Reach2 &&reach2 ) const {
        for_each_candidate_from( 0, p0, i0, may_cut, cut_with, reach2 );
    }

    /// Le meme, borne a UN SOUS-ARBRE. Ce que ca separe : la granularite de l'INDEX (quel bloc
    /// regarder) de celle de l'ELAGAGE (quoi tester dedans). Un index par paquets veut des paquets
    /// gros -- pour la memoire et pour amortir sa preparation -- alors que l'elagage veut des
    /// boites fines. Sans ca, un paquet retenu fait tester ses `rho` germes en bloc.
    ///
    /// Rend `false` si `cut_with` a demande l'arret (cellule vide) : l'appelant doit alors sortir.
    template<class MayCut, class CutWith, class Reach2>
    bool for_each_candidate_from( SI root, Vec<D> p0, SI i0, MayCut &&may_cut,
                                  CutWith &&cut_with, Reach2 && ) const {
        SI stack[ 64 ];
        SI top = 0;
        stack[ top++ ] = root;

        while ( top > 0 ) {
            const SI h = stack[ --top ];
            const Node &nd = nodes[ h ];

            if ( ! may_cut( vec_of<D>( nd.lo ), vec_of<D>( nd.hi ), nd.wm ) )
                continue;

            if ( nd.right < 0 ) {
                for ( SI k = nd.beg; k < nd.end; ++k ) {
                    const SI id = order[ k ];
                    if ( id != i0 && ! cut_with( seed( k ), seed_w( k ), id ) )
                        return false;
                }
                continue;
            }

            const SI l = h + 1, r = nd.right;           // PREORDRE : le gauche est juste a cote
            if ( nearness( l, p0 ) <= nearness( r, p0 ) ) {
                stack[ top++ ] = r;
                stack[ top++ ] = l;
            } else {
                stack[ top++ ] = l;
                stack[ top++ ] = r;
            }
        }
        return true;
    }

    void build( const TF *const *P, const TF *W, SI n, SI leaf ) {
        build_sel( P, W, nullptr, n, leaf );
    }
    /// les deux commodites qui evitent de monter un tableau de pointeurs sur chaque site d'appel.
    void build( const TF *X, const TF *Y, const TF *W, SI n, SI leaf ) requires ( D == 2 ) {
        const TF *P[ 2 ] = { X, Y };
        build_sel( P, W, nullptr, n, leaf );
    }
    void build( const TF *X, const TF *Y, const TF *Z, const TF *W, SI n, SI leaf ) requires ( D == 3 ) {
        const TF *P[ 3 ] = { X, Y, Z };
        build_sel( P, W, nullptr, n, leaf );
    }

    /// Le meme, en SAUTANT un sous-arbre entier -- celui d'un PAQUET qu'on vient de balayer.
    ///
    /// = Pourquoi la boucle est RECOPIEE et non factorisee
    ///
    /// Elle l'a ete, avec un `if constexpr` sur un parametre de template, ce qui ne devait rien
    /// couter a l'instanciation sans saut. MESURE : `pd_bsp` passait de 5.301 a 5.505 s sur
    /// l'uniforme 3D et de 5.058 a 5.562 sur le nuage dur -- 4 a 10 %, sans qu'une instruction du
    /// parcours ordinaire ait change. C'est le meme effet de placement de code que ce depot a deja
    /// mesure ailleurs, et il tombe ici sur la fonction la plus chaude du banc. La duplication
    /// achete une reference qui ne bouge pas, et c'est ce qui compte le plus.
    ///
    /// = Pourquoi sauter un SOUS-ARBRE
    ///
    /// C'est tout l'interet de prendre un NOEUD du Bsp comme paquet : un seul test, en haut, la ou
    /// une partition quelconque demanderait une lecture par candidat. Et ce n'est pas un confort :
    /// reproposer un germe deja coupe recoupe avec un plan CONFONDU avec une face existante, dont
    /// les sommets ont un produit scalaire nul au signe d'arrondi pres -- c'est le defaut que
    /// `AaBspPre::pre_overlap` garde pour le dissequer, et il donne des volumes faux.
    template<class MayCut, class CutWith, class Reach2>
    bool for_each_candidate_skip( SI root, SI skip, Vec<D> p0, SI i0, MayCut &&may_cut,
                                  CutWith &&cut_with, Reach2 && ) const {
        SI stack[ 64 ];
        SI top = 0;
        stack[ top++ ] = root;

        while ( top > 0 ) {
            const SI h = stack[ --top ];
            if ( h == skip )
                continue;
            const Node &nd = nodes[ h ];

            if ( ! may_cut( vec_of<D>( nd.lo ), vec_of<D>( nd.hi ), nd.wm ) )
                continue;

            if ( nd.right < 0 ) {
                for ( SI k = nd.beg; k < nd.end; ++k ) {
                    const SI id = order[ k ];
                    if ( id != i0 && ! cut_with( seed( k ), seed_w( k ), id ) )
                        return false;
                }
                continue;
            }

            const SI l = h + 1, r = nd.right;           // PREORDRE : le gauche est juste a cote
            if ( nearness( l, p0 ) <= nearness( r, p0 ) ) {
                stack[ top++ ] = r;
                stack[ top++ ] = l;
            } else {
                stack[ top++ ] = l;
                stack[ top++ ] = r;
            }
        }
        return true;
    }

    /// L'arbre sur la SELECTION `idx` (`nullptr` = tous). `order` part des indices d'origine et
    /// n'est ensuite que permute, donc `seed_id` rend l'identite dans le nuage ENTIER meme quand
    /// l'arbre n'en porte qu'un seizieme -- sans quoi un germe de la selection ne pourrait pas se
    /// reconnaitre lui-meme et se couperait avec son propre plan.
    void build_sel( const TF *X, const TF *Y, const TF *W, const SI *idx, SI n, SI leaf )
        requires ( D == 2 ) {
        const TF *P[ 2 ] = { X, Y };
        build_sel( P, W, idx, n, leaf );
    }
    void build_sel( const TF *const *P, const TF *W, const SI *idx, SI n, SI leaf ) {
        leaf_size = leaf;
        order.resize( n );
        for ( SI i = 0; i < n; ++i )
            order[ i ] = idx ? idx[ i ] : i;

        // les positions sont PERMUTEES a cote des indices, pas relues a travers eux : une feuille
        // se lit alors d'un seul tenant, la ou une indirection en ferait une collecte eparse.
        for ( int d = 0; d < D; ++d )
            p[ d ].resize( n );
        if ( W )
            pw.resize( n );

        nodes.clear();
        nodes.reserve( 2 * ( n / std::max<SI>( leaf, 1 ) + 1 ) );
        _build( P, W, 0, n );

        for ( SI k = 0; k < n; ++k ) {
            for ( int d = 0; d < D; ++d )
                p[ d ][ k ] = P[ d ][ order[ k ] ];
            if ( W )
                pw[ k ] = W[ order[ k ] ];
        }
    }

protected:                                          // `ObBsp` reprend `Node`, `nearness` et la boucle
    /// Carre de la distance du point a la boite du noeud -- 0 dedans. UNIQUEMENT une cle d'ordre,
    /// donc volontairement ce bon marche : le vrai test balaie les sommets de la cellule, et le
    /// payer sur les deux fils juste pour choisir lequel regarder d'abord doublerait la marche.
    TF nearness( SI n, Vec<D> x ) const {
        const Node &nd = nodes[ n ];
        TF s = 0;
        for ( int d = 0; d < D; ++d ) {
            const TF e = x[ d ] < nd.lo[ d ] ? nd.lo[ d ] - x[ d ]
                                             : ( x[ d ] > nd.hi[ d ] ? x[ d ] - nd.hi[ d ] : TF( 0 ) );
            s += e * e;
        }
        return s;
    }

    /// Recursif, donc PREORDRE par construction : le noeud est pousse avant ses fils, le fils
    /// gauche juste apres lui.
    SI _build( const TF *const *P, const TF *W, SI beg, SI end ) {
        const SI me = SI( nodes.size() );
        nodes.push_back( Node{} );

        TF lo[ D ], hi[ D ];
        for ( int d = 0; d < D; ++d )
            lo[ d ] = hi[ d ] = P[ d ][ order[ beg ] ];
        for ( SI k = beg + 1; k < end; ++k ) {
            const SI i = order[ k ];
            for ( int d = 0; d < D; ++d ) {
                const TF v = P[ d ][ i ];
                lo[ d ] = v < lo[ d ] ? v : lo[ d ];
                hi[ d ] = v > hi[ d ] ? v : hi[ d ];
            }
        }
        for ( int d = 0; d < D; ++d ) { nodes[ me ].lo[ d ] = lo[ d ]; nodes[ me ].hi[ d ] = hi[ d ]; }
        nodes[ me ].beg = beg;
        nodes[ me ].end = end;

        // le MAJORANT des poids du sous-arbre. Sans poids il reste a zero et tous ses termes
        // disparaissent du test.
        if ( W )
            nodes[ me ].wm = weight_majorant<D>( beg, end, [ & ]( SI k, Vec<D> &y, TF &w ) {
                const SI i = order[ k ];
                for ( int d = 0; d < D; ++d ) y[ d ] = P[ d ][ i ];
                w = W[ i ];
            } );

        int ax = 0;
        for ( int d = 1; d < D; ++d )
            if ( hi[ d ] - lo[ d ] > hi[ ax ] - lo[ ax ] ) ax = d;
        const TF span = hi[ ax ] - lo[ ax ];

        // `span <= 0` : tous les germes au meme endroit, aucune coupe ne les separerait, et
        // insister ferait une descente sans fin.
        if ( end - beg <= leaf_size || ! ( span > 0 ) ) {
            nodes[ me ].right = -1;
            return me;
        }

        // la MEDIANE et non le milieu de la boite : c'est ce qui borne la profondeur par
        // `log2( n / leaf )` quelle que soit la distribution.
        const SI mid = beg + ( end - beg ) / 2;
        const TF *C = P[ ax ];
        std::nth_element( order.begin() + beg, order.begin() + mid, order.begin() + end,
                          [ & ]( SI a, SI b ) { return C[ a ] < C[ b ]; } );

        _build( P, W, beg, mid );                       // == me + 1
        nodes[ me ].right = _build( P, W, mid, end );
        return me;
    }
};

using AaBsp  = AaBspT<2>;
using AaBsp3 = AaBspT<3>;

/// L'accelerateur qui n'accelere rien : tous les germes, dans l'ordre. Pas un bouche-trou -- c'est
/// l'ORACLE contre lequel les autres se verifient, et il fait de « pas d'acceleration » un cas
/// ordinaire du meme code plutot qu'une seconde implementation a tenir en phase.
template<int D>
struct EverySeedT {
    static constexpr int dim = D;

    std::vector<SI> order;
    std::vector<TF> p[ D ], pw;

    static constexpr const char *name = "all";

    Vec<D> seed( SI k ) const {
        if constexpr ( D == 2 )
            return { p[ 0 ][ k ], p[ 1 ][ k ] };
        else if constexpr ( D == 3 )
            return { p[ 0 ][ k ], p[ 1 ][ k ], p[ 2 ][ k ] };
        else {
            Vec<D> r;
            for ( int d = 0; d < D; ++d ) r[ d ] = p[ d ][ k ];
            return r;
        }
    }
    TF seed_c( SI k, int d ) const { return p[ d ][ k ]; }
    TF seed_x( SI k ) const { return p[ 0 ][ k ]; }
    TF seed_y( SI k ) const { return p[ 1 ][ k ]; }
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
            if ( id != i0 && ! cut_with( seed( k ), seed_w( k ), id ) )
                return;
        }
    }

    void build( const TF *const *P, const TF *W, SI n, SI ) {
        order.resize( n );
        for ( int d = 0; d < D; ++d ) p[ d ].resize( n );
        if ( W ) pw.resize( n );
        for ( SI i = 0; i < n; ++i ) {
            order[ i ] = i;
            for ( int d = 0; d < D; ++d ) p[ d ][ i ] = P[ d ][ i ];
            if ( W ) pw[ i ] = W[ i ];
        }
    }
    void build( const TF *X, const TF *Y, const TF *W, SI n, SI l ) requires ( D == 2 ) {
        const TF *P[ 2 ] = { X, Y };
        build( P, W, n, l );
    }
    void build( const TF *X, const TF *Y, const TF *Z, const TF *W, SI n, SI l ) requires ( D == 3 ) {
        const TF *P[ 3 ] = { X, Y, Z };
        build( P, W, n, l );
    }
};

using EverySeed  = EverySeedT<2>;
using EverySeed3 = EverySeedT<3>;

/// UN SEUL GERME, pose sur un arbre auquel il n'appartient pas -- ou auquel il appartient avec un
/// AUTRE poids. C'est un accelerateur comme un autre : il ne doit qu'une chose, proposer des
/// germes, et il delegue le parcours a l'arbre qu'il enveloppe.
///
/// Pourquoi passer par la plutot que par un `make_cell` parametre : le concept d'accelerateur
/// existe deja et ne coute rien, alors que factoriser `make_cell` coutait 3 % (cf. `PowerDiagram`).
template<int D>
struct OneSeedT {
    static constexpr int dim = D;

    const AaBspT<D> &tr;
    Vec<D> pos;
    TF w;
    SI id;

    Vec<D> seed( SI ) const { return pos; }
    TF seed_x( SI ) const { return pos[ 0 ]; }
    TF seed_y( SI ) const { return pos[ 1 ]; }
    TF seed_w( SI ) const { return w; }
    SI seed_id( SI ) const { return id; }
    SI nb_seeds() const { return 1; }

    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        tr.for_each_candidate_at( pos, id, may_cut, cut_with, reach2 );
    }
};

using OneSeed = OneSeedT<2>;

} // namespace pd
