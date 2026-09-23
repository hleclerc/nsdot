#pragma once

// =====================================================================================
// L'ECRASEMENT DES CELLULES LE LONG D'UNE DIRECTION : jusqu'ou peut-on aller ? ( 2D )
//
// Le port de `solvers_des_familles/src/solver/Ecrasement.h` et `src/cell/FournisseurAlpha2D.h`
// ( README § 7, § 9.7 ) sur la cellule et les fournisseurs de sdot.
//
// Newton propose `d` ; l'amortissement essaye `w + t d` et refuse tant qu'une cellule passe sous le
// plancher. Chaque essai est un diagramme. Ici, pour les seules cellules que l'essai a trouvees
// sous le plancher, on calcule la LIMITE le long de `d` -- le premier `alpha` ou la cellule passe
// sous `eps` -- avec des cellules EXACTES calculees une par une, a chaud :
//
//   * le POLYNOME ( mesure de Lebesgue ) : tant que la cellule garde les memes aretes, chaque sommet
//     est AFFINE en `alpha` et l'aire est un polynome de degre 2 ; on predit sa racine, on verifie
//     par une cellule exacte a `0.99 x` la prediction ( memes aretes = polynome exact = limite
//     confirmee ), sinon la cellule calculee porte la nouvelle combinatoire et on recommence ;
//   * la BISSECTION en masse ( toute autre densite ) : la masse le long de `w + alpha d` n'est pas
//     un polynome ; une bissection sur `alpha` entre 0 et l'horizon, une cellule exacte par tour.
//
// = La cellule en `w + alpha d`, sans rafraichir l'arbre ( `PdAlpha` )
//
// Le majorant affine est lineaire en les poids : si un noeud porte `w( y ) <= a_w . y + b_w` et
// `d( y ) <= a_d . y + b_d`, alors pour tout `alpha >= 0`
//
//     w( y ) + alpha d( y ) <= ( a_w + alpha a_d ) . y + ( b_w + alpha b_d ).
//
// C'est exact, donc l'elagage reste exact, et `alpha` peut CHANGER D'UNE CELLULE A L'AUTRE. Le
// majorant de `d` est calcule une fois par direction, en dehors de l'arbre. `PdAlpha` est le
// stockage vu ainsi : les memes points, les memes boites, des poids et des majorants decales --
// et c'est le `FournisseurBsp` ORDINAIRE qui le parcourt.
//
// = Le depart a chaud
//
// Avant le parcours, on propose les plans des voisins de la cellule en `alpha = 0` ( le laplacien
// de Newton les porte deja ) : c'est la MEMOIRE du fournisseur ( `MEMO` ), qui les propose en
// premier et les saute ensuite. Six coupes sans parcours, et la cellule est deja petite quand le
// parcours commence : l'elagage, exact, ecarte presque tout.
// =====================================================================================

#include "Balayage.h"
#include "../UnitDensity.h"
#include <atomic>
#include <limits>

namespace sdot {
namespace otplan {

constexpr double INFINI = std::numeric_limits<double>::infinity();

// ---- le stockage en `w + alpha d` -------------------------------------------------------------------

/// le majorant d'un noeud, decale : `wa + alpha da`
template<class View>
struct MajAlphaA {
    const View   &wa;
    const double *da;
    double        alpha;
    double operator()( SI n, int d ) const { return double( wa( n, d ) ) + alpha * da[ 2 * n + d ]; }
};
template<class View>
struct MajAlphaB {
    const View   &wb;
    const double *db;
    double        alpha;
    double operator()( SI n ) const { return double( wb( n ) ) + alpha * db[ n ]; }
};

/// l'arbre vu par `FournisseurBsp` : les boites et les tranches telles quelles, les majorants decales
template<class Tree>
struct ArbreAlpha {
    const DECAYED_TYPE_OF( std::declval<Tree>().node_box )   &node_box;
    const DECAYED_TYPE_OF( std::declval<Tree>().node_begin ) &node_begin;
    const DECAYED_TYPE_OF( std::declval<Tree>().node_end )   &node_end;
    MajAlphaA<DECAYED_TYPE_OF( std::declval<Tree>().node_wa )> node_wa;
    MajAlphaB<DECAYED_TYPE_OF( std::declval<Tree>().node_wb )> node_wb;
};

/// LE STOCKAGE BSP en `w + alpha d`, la liste chaude comme memoire ( des RANGS, tries croissant )
template<class PD>
struct PdAlphaBsp {
    using TF = typename PD::TF;
    static constexpr int  ct_dim = PD::ct_dim;
    static constexpr bool on_cpu = true;
    static constexpr bool has_weights = true;

    const PD     &pd;
    const double *w, *d;                                 ///< ordre utilisateur
    double        alpha;
    ArbreAlpha<DECAYED_TYPE_OF( std::declval<PD>().tree )> tree;
    const int    *chauds;
    int           nb_chauds;

    SI   nb_seeds() const { return pd.nb_seeds(); }
    auto point( SI k ) const { return pd.point( k ); }
    SI   user_id( SI k ) const { return pd.user_id( k ); }
    TF   weight( SI k ) const { const SI i = pd.user_id( k ); return TF( w[ i ] + alpha * d[ i ] ); }
    int  memo_counts( SI ) const { return nb_chauds; }
    int  memo_nbrs( SI, int q ) const { return chauds[ q ]; }

    template<class TK>
    auto fournisseur( SI k0 ) const { return FournisseurBsp<PdAlphaBsp,TK,ct_dim,true,true>( *this, k0 ); }
};

/// LE STOCKAGE PLAT en `w + alpha d` : tous les germes, pas de depart a chaud
template<class PD>
struct PdAlphaPlain {
    using TF = typename PD::TF;
    static constexpr int  ct_dim = PD::ct_dim;
    static constexpr bool on_cpu = true;
    static constexpr bool has_weights = true;

    const PD     &pd;
    const double *w, *d;
    double        alpha;

    SI   nb_seeds() const { return pd.nb_seeds(); }
    auto point( SI k ) const { return pd.point( k ); }
    SI   user_id( SI k ) const { return k; }
    TF   weight( SI k ) const { return TF( w[ k ] + alpha * d[ k ] ); }

    template<class TK>
    auto fournisseur( SI k0 ) const { return FournisseurTous<PdAlphaPlain,TK,ct_dim>( *this, k0 ); }
};

/// LES SEULS PLANS D'UNE LISTE ( des rangs ) : pour refaire une cellule dont on connait les voisins
template<class PDA,class TK>
struct FournisseurListe {
    const PDA &pd;
    SI  k0;
    const int *liste;
    int nb, q = 0;
    typename PDA::TF p0[ 2 ], w0;

    FournisseurListe( const PDA &pd, SI k0, const int *liste, int nb ) : pd( pd ), k0( k0 ), liste( liste ), nb( nb ) {
        const auto p = pd.point( k0 );
        p0[ 0 ] = p[ 0 ]; p0[ 1 ] = p[ 1 ];
        w0 = pd.weight( k0 );
    }
    template<class Etat>
    bool suivant( const Etat &, RienDeLocal &, Plane<TK,2> &p ) {
        if ( q >= nb ) return false;
        const SI k = liste[ q++ ];
        const auto pj = pd.point( k );
        typename PDA::TF p1[ 2 ] = { pj[ 0 ], pj[ 1 ] };
        p = bisector<TK,2>( p0, w0, p1, pd.weight( k ), int( k ) );
        return true;
    }
};

// ---- le polynome d'une cellule -----------------------------------------------------------------------

/// LE POLYNOME D'UNE CELLULE : `q( alpha ) = a0 + a1 alpha + a2 alpha^2`, et ce qu'on en tire.
struct PolyCellule {
    enum Etat : int { OK = 0, VIDE_AU_DEPART, DEBORDE, DEGENERE };

    double a0 = 0, a1 = 0, a2 = 0;
    double alpha_arete = INFINI;   ///< premiere arete qui s'annule : la combinatoire change
    int    nb_aretes = 0;
    int    etat = OK;

    double operator()( double alpha ) const { return a0 + alpha * ( a1 + alpha * a2 ); }

    /// les racines reelles de `q( alpha ) == niveau`, triees ; rend leur nombre ( 0, 1 ou 2 ).
    int racines( double niveau, double &r1, double &r2 ) const {
        const double c = a0 - niveau;
        if ( std::fabs( a2 ) <= 1e-300 ) {
            if ( a1 == 0 ) return 0;
            r1 = -c / a1;
            return 1;
        }
        const double disc = a1 * a1 - 4 * a2 * c;
        if ( disc < 0 ) return 0;
        const double s = std::sqrt( disc );
        const double q = -0.5 * ( a1 + ( a1 >= 0 ? s : -s ) );
        double x1 = q / a2, x2 = ( q != 0 ) ? c / q : x1;
        if ( x1 > x2 ) std::swap( x1, x2 );
        r1 = x1; r2 = x2;
        return 2;
    }

    /// la premiere racine POSITIVE de `q == niveau` ( `INFINI` s'il n'y en a pas ) : la limite predite
    double premiere_racine( double niveau ) const {
        double r1, r2;
        const int nr = racines( niveau, r1, r2 );
        if ( nr >= 1 && r1 > 0 ) return r1;
        if ( nr >= 2 && r2 > 0 ) return r2;
        return INFINI;
    }
};

/// Une droite `n . x <= c + alpha delta`, la normale fixe.
struct Droite2 { double nx, ny, c, delta; };

/// LE POLYNOME D'UNE CELLULE `cel` du germe de rang `k0`, calculee aux poids `w + alpha0 d`, le long
/// de `d` : `q( beta )` est l'aire en `w + ( alpha0 + beta ) d`, combinatoire figee. `pda` donne les
/// points par rang et `w`, `d` par identifiant.
template<class Local,class PDA>
PolyCellule polynome_cellule( const Local &cel, SI k0, const PDA &pda, double alpha0, std::vector<Droite2> &dr,
                              std::vector<double> &v0x, std::vector<double> &v0y, std::vector<double> &v1x, std::vector<double> &v1y ) {
    PolyCellule q;
    if ( cel.nb == 0 ) { q.etat = PolyCellule::VIDE_AU_DEPART; return q; }
    const int nb = cel.nb;
    q.nb_aretes = nb;
    dr.resize( nb ); v0x.resize( nb ); v0y.resize( nb ); v1x.resize( nb ); v1y.resize( nb );

    // ---- les droites : l'arete `j` va de `v_j` a `v_j+1`, portee par la coupe `cid[ j ]`
    const SI i0 = pda.user_id( k0 );
    const auto pi = pda.point( k0 );
    const double xi = double( pi[ 0 ] ), yi = double( pi[ 1 ] ), wi = pda.w[ i0 ] + alpha0 * pda.d[ i0 ], di = pda.d[ i0 ];
    for ( int j = 0; j < nb; ++j ) {
        const int id = cel.cid[ j ];
        if ( id >= 0 ) {
            const SI ij = pda.user_id( id );
            const auto pj = pda.point( id );
            const double xj = double( pj[ 0 ] ), yj = double( pj[ 1 ] ), wj = pda.w[ ij ] + alpha0 * pda.d[ ij ];
            const double nx = xj - xi, ny = yj - yi;
            dr[ j ] = { nx, ny, 0.5 * ( nx * ( xj + xi ) + ny * ( yj + yi ) + wi - wj ), 0.5 * ( di - pda.d[ ij ] ) };
        } else {                                         // le domaine : la droite lue sur l'arete, immobile
            typename Local::TKernel dx, dy, off;
            cel.plane_of_edge( j, dx, dy, off );
            dr[ j ] = { double( dx ), double( dy ), double( off ), 0 };
        }
    }

    // ---- les sommets, affines : `v_j` est l'intersection des aretes `j-1` et `j`
    for ( int j = 0; j < nb; ++j ) {
        const Droite2 &a = dr[ j ? j - 1 : nb - 1 ], &b = dr[ j ];
        const double det = a.nx * b.ny - a.ny * b.nx;
        if ( ! ( std::fabs( det ) > 0 ) ) { q.etat = PolyCellule::DEGENERE; return q; }
        v0x[ j ] = ( a.c * b.ny - b.c * a.ny ) / det;
        v0y[ j ] = ( a.nx * b.c - b.nx * a.c ) / det;
        v1x[ j ] = ( a.delta * b.ny - b.delta * a.ny ) / det;
        v1y[ j ] = ( a.nx * b.delta - b.nx * a.delta ) / det;
    }

    // ---- l'aire signee, et la longueur signee de chaque arete
    double a0 = 0, a1 = 0, a2 = 0;
    for ( int j = 0; j < nb; ++j ) {
        const int l = j + 1 < nb ? j + 1 : 0;
        a0 += v0x[ j ] * v0y[ l ] - v0x[ l ] * v0y[ j ];
        a1 += v0x[ j ] * v1y[ l ] - v0x[ l ] * v1y[ j ] + v1x[ j ] * v0y[ l ] - v1x[ l ] * v0y[ j ];
        a2 += v1x[ j ] * v1y[ l ] - v1x[ l ] * v1y[ j ];
    }
    const double sg = a0 < 0 ? -0.5 : 0.5;
    q.a0 = sg * a0; q.a1 = sg * a1; q.a2 = sg * a2;

    for ( int j = 0; j < nb; ++j ) {
        const int l = j + 1 < nb ? j + 1 : 0;
        const double tx = -dr[ j ].ny, ty = dr[ j ].nx; // le long de l'arete `j`
        double l0 = ( v0x[ l ] - v0x[ j ] ) * tx + ( v0y[ l ] - v0y[ j ] ) * ty;
        double l1 = ( v1x[ l ] - v1x[ j ] ) * tx + ( v1y[ l ] - v1y[ j ] ) * ty;
        if ( l0 < 0 ) { l0 = -l0; l1 = -l1; }
        if ( l1 < 0 )
            q.alpha_arete = std::min( q.alpha_arete, -l0 / l1 );
    }
    return q;
}

// ---- la passe -----------------------------------------------------------------------------------------

/// CE QU'ON SAIT D'UNE CELLULE a la fin : sa limite, et comment on l'a obtenue.
struct LimiteCellule {
    enum Etat : int { CONFIRMEE = 0, CORRIGEE, HORIZON, VIDE_AU_DEPART, ECHEC };
    double alpha      = INFINI;    ///< le premier `alpha` ou la cellule passe sous `niveau`
    double alpha_poly = INFINI;    ///< ce que le polynome en 0 predisait
    int    tours      = 0;         ///< cellules calculees en plus de celle en 0
    int    etat       = ECHEC;
};

struct OptionsLimites {
    double coeff     = 0.99;       ///< on verifie en `a_ok + coeff * ( predit - a_ok )`
    double tol       = 1e-2;       ///< precision relative demandee sur la limite
    int    max_tours = 12;
};

/// LES LIMITES, sur un `Balayage` 2D : la passe « predire, verifier, corriger » ( polynome, mesure de
/// Lebesgue ) ou la bissection en masse ( toute autre densite ), pour les seules cellules demandees.
template<class Bal>
struct Limites2D {
    using PD    = DECAYED_TYPE_OF( std::declval<Bal>().pd );
    using Local = typename Bal::Local;
    using TK    = typename Local::TKernel;
    using TF    = typename PD::TF;
    using Dist  = DECAYED_TYPE_OF( *std::declval<Bal>().dist );
    static constexpr bool bsp = requires( const PD &p ) { p.tree; };
    static constexpr bool polynome = std::is_same_v<Dist,UnitDensity>;
    using PDA = std::conditional_t<bsp,PdAlphaBsp<PD>,PdAlphaPlain<PD>>;

    Bal            &bal;
    OptionsLimites  o;
    std::vector<double> dt, da, db;                      ///< `d` dans l'ordre de l'arbre, son majorant par noeud
    std::vector<std::vector<std::int32_t>> scratch;      ///< le scratch de CETTE passe, par fil ( il grossit seul )
    std::vector<SI> caps;
    std::vector<LimiteCellule> lim;                      ///< par identifiant

    struct Fil {                                         ///< ce qu'un fil garde d'une cellule a l'autre
        std::vector<Droite2> dr;
        std::vector<double> v0x, v0y, v1x, v1y;
        std::vector<int> chauds;                         ///< les voisins de la derniere bonne cellule ( rangs tries )
        std::vector<int> cids_ok, c2;                    ///< TOUTES ses coupes, domaine compris ( la combinatoire )
    };

    Limites2D( Bal &bal ) : bal( bal ) {
        scratch.resize( bal.nt );
        caps.assign( bal.nt, bal.cap );
        for ( int t = 0; t < bal.nt; ++t )
            scratch[ t ].assign( size_t( diagram::words_for<Local,TF>( caps[ t ], Bal::nbc, false ) ) + 16, 0 );
    }

    /// le stockage en `w + alpha d`, la liste chaude `chauds` ( rangs tries )
    PDA pd_alpha( const double *w, const double *d, double alpha, const int *chauds, int nb ) const {
        if constexpr ( bsp )
            return PDA{ bal.pd, w, d, alpha,
                        { bal.pd.tree.node_box, bal.pd.tree.node_begin, bal.pd.tree.node_end,
                          { bal.pd.tree.node_wa, da.data(), alpha }, { bal.pd.tree.node_wb, db.data(), alpha } },
                        chauds, nb };
        else
            return PDA{ bal.pd, w, d, alpha };
    }

    /// UNE FOIS PAR DIRECTION : `d` dans l'ordre de l'arbre, et son majorant par noeud
    void prepare( const double *d ) {
        if constexpr ( bsp ) {
            const SI n = bal.n();
            dt.resize( n );
            for ( SI k = 0; k < n; ++k ) dt[ k ] = d[ bal.pd.user_id( k ) ];
            const SI nb_nodes = SI( bal.pd.tree.node_begin.shape( 0 ) );
            da.assign( 2 * nb_nodes, 0.0 );
            db.assign( nb_nodes, 0.0 );
            struct Poids { const double *v; double operator()( SI k ) const { return v[ k ]; } };
            struct Wa { double *v; double &operator()( int d ) const { return v[ d ]; } };
            bal.queue.run_threads( bal.nt, [&]( int t ) {
                SI b, e;
                Bal::tranche( nb_nodes, t, bal.nt, b, e );
                for ( SI m = b; m < e; ++m ) {
                    const SI nb = SI( bal.pd.tree.node_begin( m ) ), ne = SI( bal.pd.tree.node_end( m ) );
                    if ( ne <= nb ) continue;
                    bsp_weight_majorant<2>( bal.pd.sorted_positions, Poids{ dt.data() }, nb, ne, Wa{ da.data() + 2 * m }, db[ m ] );
                }
            } );
        }
    }

    /// LA CELLULE du rang `k` en `w + alpha d`, a chaud depuis `chauds` ( `parcours = false` : ces plans
    /// seuls ). Rend `false` si le scratch du fil n'a pas suffi apres avoir grossi ( jamais, en pratique ).
    bool cellule( int t, const PDA &pda, SI k, Local &c, Local &piece, bool parcours, const int *liste = nullptr, int nb_liste = 0 ) {
        for ( int essai = 0; essai < 8; ++essai ) {
            Carver cv{ scratch[ t ].data(), SI( scratch[ t ].size() ) - 16 };
            c.attach( cv, caps[ t ] );
            if constexpr ( Bal::nbc > 1 ) piece.attach( cv, caps[ t ] );
            else                          piece = c;
            bool ok;
            if ( parcours ) {
                ok = diagram::make_cell( pda, c, k, bal.dom );
            } else {
                ok = c.load( bal.dom );
                if ( ok ) {
                    FournisseurListe<PDA,TK> f( pda, k, liste, nb_liste );
                    ok = run<true>( c, f ) != CutStatus::OVERFLOW;
                }
            }
            if ( ok ) return true;
            caps[ t ] *= 2;
            scratch[ t ].assign( size_t( diagram::words_for<Local,TF>( caps[ t ], Bal::nbc, false ) ) + 16, 0 );
        }
        return false;
    }

    /// la masse de `c` contre la distribution du balayage
    double masse( Local &c, Local &piece ) const {
        if ( c.nb == 0 ) return 0;
        TF m = 0;
        if ( ! diagram::integrate_into<TF>( m, c, piece, *bal.dist ) ) return 0;
        return double( m );
    }

    /// LES LIMITES des cellules `mauvaises` ( identifiants ), le long de `d` depuis `w`, sous `horizon`,
    /// au niveau `eps` ; `L` porte les voisins en `alpha = 0`. Rend `min_i alpha_i` ; `nb_cellules`
    /// compte les cellules calculees. `lim[ i ]` est rempli pour chaque `i` de `mauvaises`.
    double alpha_min( const std::vector<double> &w, const std::vector<double> &d, const std::vector<SI> &mauvaises,
                      double horizon, double eps, const Laplacien &L, SI &nb_cellules ) {
        const SI n = bal.n();
        if ( SI( lim.size() ) != n ) lim.assign( n, LimiteCellule{} );
        prepare( d.data() );
        std::atomic<double> courant{ INFINI };
        auto abaisse = [&]( double a ) {
            double c = courant.load();
            while ( a < c && ! courant.compare_exchange_weak( c, a ) ) {}
        };
        std::atomic<SI> nb_cel{ 0 };
        std::vector<Fil> fils( bal.nt );
        const SI nm = SI( mauvaises.size() );

        bal.queue.run_threads( bal.nt, [&]( int t ) {
            Fil &f = fils[ t ];
            Local c, piece;
            SI b, e;
            Bal::tranche( nm, t, bal.nt, b, e );
            for ( SI j = b; j < e; ++j ) {
                const SI i = mauvaises[ j ], k = bal.rang_de[ i ];
                LimiteCellule &Li = lim[ i ];
                Li = LimiteCellule{};
                // les voisins en 0, en rangs tries : le depart a chaud, et la cellule en 0 sans parcours
                f.chauds.clear();
                for ( SI q = L.row[ i ]; q < L.row[ i + 1 ]; ++q )
                    f.chauds.push_back( int( bal.rang_de[ L.col[ q ] ] ) );
                std::sort( f.chauds.begin(), f.chauds.end() );
                auto garde_voisins = [&]() {             // la cellule courante est bonne : on repart d'elle
                    f.chauds.clear();
                    f.cids_ok.assign( c.cid, c.cid + c.nb );
                    std::sort( f.cids_ok.begin(), f.cids_ok.end() );
                    for ( int q = 0; q < c.nb; ++q ) if ( c.cid[ q ] >= 0 ) f.chauds.push_back( c.cid[ q ] );
                    std::sort( f.chauds.begin(), f.chauds.end() );
                };
                auto fini = [&]( double a, int etat ) { Li.alpha = a; Li.etat = etat; if ( etat != LimiteCellule::HORIZON ) abaisse( a ); };

                if constexpr ( polynome ) {
                    // ---- predire, verifier, corriger
                    PDA p0 = pd_alpha( w.data(), d.data(), 0, f.chauds.data(), int( f.chauds.size() ) );
                    cellule( t, p0, k, c, piece, false, f.chauds.data(), int( f.chauds.size() ) );
                    PolyCellule q = polynome_cellule( c, k, p0, 0, f.dr, f.v0x, f.v0y, f.v1x, f.v1y );
                    if ( q.etat != PolyCellule::OK ) { Li.etat = LimiteCellule::VIDE_AU_DEPART; Li.alpha = 0; abaisse( 0 ); continue; }
                    Li.alpha_poly = q.premiere_racine( eps );
                    double a_ok = 0, a_bad = INFINI, pred = Li.alpha_poly, cible = pred;
                    garde_voisins();
                    bool termine = false;
                    for ( ; Li.tours < o.max_tours && ! termine; ) {
                        bool sur_pred = cible == pred;
                        double a_test = std::min( cible, horizon );
                        if ( ! ( a_test < a_bad ) ) { a_test = 0.5 * ( a_ok + a_bad ); sur_pred = false; }
                        if ( sur_pred && a_test < horizon ) a_test = a_ok + o.coeff * ( a_test - a_ok );
                        if ( ! ( a_test > a_ok ) ) { fini( a_ok, LimiteCellule::CORRIGEE ); termine = true; break; }

                        PDA pa = pd_alpha( w.data(), d.data(), a_test, f.chauds.data(), int( f.chauds.size() ) );
                        cellule( t, pa, k, c, piece, true );
                        ++Li.tours;
                        ++nb_cel;

                        // ---- memes aretes ( le domaine compris ) : le polynome etait exact de `a_ok` a `a_test`
                        f.c2.assign( c.cid, c.cid + c.nb );
                        std::sort( f.c2.begin(), f.c2.end() );
                        const bool memes = c.nb > 0 && f.c2 == f.cids_ok;
                        if ( memes ) {
                            if ( a_test >= horizon ) { fini( pred < INFINI && pred < horizon ? pred : horizon, LimiteCellule::HORIZON ); termine = true; break; }
                            if ( sur_pred ) { fini( pred, Li.tours == 1 ? LimiteCellule::CONFIRMEE : LimiteCellule::CORRIGEE ); termine = true; break; }
                            a_ok = a_test;
                            cible = pred;
                            if ( a_bad < INFINI && a_bad - a_ok <= o.tol * a_bad ) { fini( a_ok, LimiteCellule::CORRIGEE ); termine = true; break; }
                            continue;
                        }

                        // ---- la combinatoire a change : la cellule calculee porte le nouveau polynome
                        const PolyCellule q2 = polynome_cellule( c, k, pa, a_test, f.dr, f.v0x, f.v0y, f.v1x, f.v1y );
                        const double m = q2.etat == PolyCellule::OK ? q2.a0 : 0.0;
                        if ( m >= eps ) {                // bonne : on repart d'ici
                            a_ok = a_test;
                            garde_voisins();
                            const double beta = q2.premiere_racine( eps );
                            pred = cible = a_test + beta;
                            if ( a_test >= horizon ) { fini( horizon, LimiteCellule::HORIZON ); termine = true; break; }
                            if ( ! ( pred < a_bad ) ) cible = 0.5 * ( a_ok + a_bad );
                            if ( beta <= o.tol * a_test ) { fini( a_test, LimiteCellule::CORRIGEE ); termine = true; break; }
                        } else {                         // mauvaise : la limite est avant
                            a_bad = a_test;
                            double r1, r2, beta = -INFINI;
                            if ( q2.etat == PolyCellule::OK ) {
                                const int nr = q2.racines( eps, r1, r2 );
                                if ( nr >= 1 && r1 < 0 ) beta = r1;
                                if ( nr >= 2 && r2 < 0 ) beta = r2;
                            }
                            cible = a_test + beta;
                            if ( ! ( cible > a_ok && cible < a_bad ) ) cible = 0.5 * ( a_ok + a_bad );
                            if ( cible == pred ) cible = std::nextafter( cible, a_ok );
                        }
                        if ( a_bad < INFINI && a_bad - a_ok <= o.tol * a_bad ) { fini( a_ok, LimiteCellule::CORRIGEE ); termine = true; break; }
                    }
                    if ( ! termine ) fini( a_ok, LimiteCellule::ECHEC );   // le conservatif, faute de mieux
                } else {
                    // ---- la bissection en masse, entre 0 ( admissible ) et l'horizon ( l'essai l'a vue en dessous )
                    double a_ok = 0, a_bad = horizon;
                    for ( ; Li.tours < o.max_tours && a_bad - a_ok > o.tol * a_bad; ) {
                        const double mid = 0.5 * ( a_ok + a_bad );
                        PDA pa = pd_alpha( w.data(), d.data(), mid, f.chauds.data(), int( f.chauds.size() ) );
                        cellule( t, pa, k, c, piece, true );
                        ++Li.tours;
                        ++nb_cel;
                        if ( masse( c, piece ) >= eps ) { a_ok = mid; garde_voisins(); }
                        else a_bad = mid;
                    }
                    Li.alpha = a_ok;
                    Li.etat = LimiteCellule::CORRIGEE;
                    abaisse( a_ok );
                }
            }
        } );
        nb_cellules += nb_cel.load();
        double res = courant.load();
        return res < INFINI ? res : horizon;
    }
};

} // namespace otplan
} // namespace sdot
