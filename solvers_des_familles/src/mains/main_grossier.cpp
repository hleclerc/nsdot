// =====================================================================================
// LA CONTINUATION FAITE SUR `n/R` GERMES, PUIS PROLONGEE : ce que le grossier fait gagner sur une
// densite mechante, et QUELLE agregation.
//
// POURQUOI UN BINAIRE DE PLUS. `densite` porte la densite et la continuation en largeur de
// convolution ( README § 9 ) ; `multiechelle` porte les paquets et les prolongations ( § 8 ). La
// question est leur produit, et elle a trois parties.
//
// 1. LE PRIX EST DEJA MESURE, et il est bon : la continuation coute 200 diagrammes / 58.5 s a
//    `n = 1e5` ( `sigma = 0.02`, ratio `sqrt2`, `essai-limites` ) contre 105 diagrammes / 2.13 s a
//    `n = 6250`. Le compte de diagrammes tombe AUSSI ( 200 -> 105 ), pas seulement le prix de
//    chacun : a `R = 16` la phase grossiere coute 3.6 % de la reference, donc elle est negligeable
//    et TOUT est dans la phase fine. La zone dure en `s` ne bouge pas avec l'espacement
//    ( pic a `s = 0.031..0.044` pour les trois `n` ) : le nuage grossier traverse la MEME zone.
//
// 2. CE QUI DECIDE N'EST PAS L'ADMISSIBILITE. Le § 8.5 l'a mesure : un depart REPARE, a un
//    balayage de la solution exacte, donne 10 it / 13 diag contre 9 / 10 depuis Voronoi -- « ce qui
//    compte pour l'amortissement est le PIRE residu ( `max|a-nu|/nu` ), pas la distance l2 », parce
//    que les cellules qui viennent de naitre sont minuscules et que c'est la pire qui regle le pas.
//    Ce binaire sort donc QUATRE nombres pour la prolongation -- cellules sous le plancher, aire
//    minimale, PIRE residu, merite -- puis le cout du Newton fin, qui est le seul qui tranche.
//
// 3. QUELLE AGREGATION. Trois bras, qui isolent les deux effets ( `--paq` ) :
//      `bsp`      les feuilles a `R` germes d'un AaBsp, representant = le germe le plus proche du
//                 centre. C'est ce que `multiechelle` fait.
//      `bsp-bary` la MEME partition, mais le germe grossier est mis au BARYCENTRE PONDERE par `nu`.
//      `lloyd`    la quantification optimale ponderee : les `n/R` points qui minimisent
//                 `sum_i nu_i |p_i - q_{r(i)}|^2`, par Lloyd depuis les representants du BSP.
//    Le critere n'est pas arbitraire, et c'est le point : cette distorsion est A LA FOIS l'ecart
//    exact entre l'objectif grossier et l'objectif fin ( § 23.12 : le terme de variance interne de
//    la decomposition du cout de transport, le terme croise s'annulant AU PREMIER ORDRE parce que
//    `q` est le barycentre pondere par `nu` ) et le terme dominant de l'erreur de prolongation
//    ( § 8.2 : une erreur de COURBURE a l'echelle du paquet, `H_c^2 w''` ). La distorsion est donc
//    rapportee pour les trois bras, en unites de l'espacement local -- avec son maximum, qui est
//    ce qui detecte le paquet pathologique ( un germe isole agrege a un amas voisin ).
//
//   xmake run grossier --sigma 0.02 -n 100000 --conv 0.5 --conv-ratio 1.414 -R 16 --paq lloyd
//   xmake run grossier --sigma 0.02 -n 100000 --conv 0.5 --conv-ratio 1.414 --reference
//
// 2D seulement, comme tout ce qui touche a `Densite.h`.
// =====================================================================================

#include "bench/Dispatch.h"
#include "solver/Densite.h"
#include "solver/Ecrasement.h"
#include "solver/Lineaire.h"
#include "solver/Newton.h"
#include "solver/Prolongation.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>
#include <functional>
#include <random>
#include <string>

using namespace sf;

namespace {

struct Opts {
    NewtonOptions newton;
    // LE RESIDU EST `lin` ICI, comme dans `main_densite.cpp` et pour la meme raison : le defaut
    // `log` de la bibliotheque a ete regle sur des solves DIRECTS, et la continuation en densite est
    // l'autre regime ( § 9.6 ). La reference a battre a ete prise en `lin`.
    Opts() { newton.residu = NewtonOptions::LIN; }
    std::string solver = "chol";
    int         amgvar = Amg::RS_GS;
    TF          sigma = 0.02;
    int         nb_gauss = 4;
    std::string gauss;
    TF          plancher = 0;
    std::string germes = "aleatoire";    ///< aleatoire | regulier ( grille a peine bruitee )
    std::string diracs = "uniforme";    ///< uniforme | rho ( germes tires selon la densite )
    TF          conv0 = 0.5, conv_ratio = 2, conv_min = 0;
    std::vector<TF> liste;
    SI          R = 16;                 ///< germes par paquet
    std::string paq = "bsp";            ///< bsp | bsp-bary | lloyd
    int         lloyd_it = 20;          ///< iterations de Lloyd
    std::string prol = "mls";           ///< copie | harmonique | mls | ctransf
    int         mls_anneaux = 2;
    TF          mls_largeur = 1;
    TF          seuil = 0.5;            ///< sous le plancher : `seuil * min( nu_i, |Vor_i| )`
    TF          tolg = 0;               ///< > 0 : la tolerance de Newton au niveau GROSSIER
    bool        reference = false;      ///< la continuation fine seule, pour le denominateur
    bool        fin = true;             ///< enchainer le Newton fin apres la prolongation
    bool        balaye = false;         ///< juger la prolongation a CHAQUE `s`, pas seulement au dernier
    bool        sans_zero = false;      ///< ne PAS finir la liste a s = 0 ( ou la densite a des zeros )
    // PAS DE GARDE D'ADMISSIBILITE PAR DEFAUT, et c'est le point : avec le residu `log` la barriere est
    // DANS l'objectif ( `a_i -> 0` donne `+inf` ), donc le minimum du merite ne peut pas affamer une
    // cellule. Il n'y a rien a interdire de l'exterieur. ( `--span-garde F > 0` ajoute quand meme un
    // plancher relatif a la base, utile seulement pour mesurer en `lin`, ou la barriere n'existe pas. )
    int         prol_lisse = 0;          ///< passes de Jacobi amorti SUR la prolongation ( `lisse_jacobi` )
    int         span_grille = 33;        ///< span : points de la grille 1-D par coordonnee et par balayage
    TF          span_garde = 0.5;
    int         span = 0;               ///< `--span K` : minimiser le merite sur un span de 1..K directions, `L` gele
    bool        reste = false;          ///< LA METRIQUE : combien d'iterations restent depuis `alpha_0 * w_prol`
    bool        echelle = false;        ///< balayer `alpha_0` : le verdict de `alpha_0 * w_prol`, alpha_0 = 1, 1/2, ...
};

/// des germes tires SELON `rho` ( rejet ) -- le nuage d'AMAS et de points isoles, qui est le cas ou
/// l'agregation devrait se voir
Nuage<2> nuage_selon( const Densite &rho, SI n, unsigned graine ) {
    Nuage<2> nu;
    nu.nom = "selon rho n=" + std::to_string( n );
    std::mt19937_64 rng( graine + 17 );
    std::uniform_real_distribution<TF> uni( 0.001, 0.999 ), u01( 0, 1 );
    const TF rmax = rho.max_rho();
    while ( SI( nu.c[ 0 ].size() ) < n ) {
        const TF x = uni( rng ), y = uni( rng );
        if ( u01( rng ) * rmax <= rho.rho( x, y ) ) { nu.c[ 0 ].push_back( x ); nu.c[ 1 ].push_back( y ); }
    }
    nu.w.assign( n, TF( 0 ) );
    nu.finish();
    return nu;
}

/// UNE GRILLE A PEINE BRUITEE : `m x m` germes a `( i + 1/2 ) / m`, deplaces d'au plus `bruit * h`
/// pour casser la degenerescence du reseau carre ( sommets quadruples ) sans toucher a l'ECART
/// MINIMAL, qui est ce qui fixe `alpha*`.
Nuage<2> nuage_regulier( SI n, unsigned graine, TF bruit = 0.05 ) {
    Nuage<2> nu;
    const SI m = SI( std::lround( std::sqrt( double( n ) ) ) );
    const TF h = TF( 1 ) / TF( m );
    nu.nom = "grille " + std::to_string( m ) + "x" + std::to_string( m );
    std::mt19937_64 rng( graine + 101 );
    std::uniform_real_distribution<TF> u( -bruit, bruit );
    for ( SI i = 0; i < m; ++i )
        for ( SI j = 0; j < m; ++j ) {
            nu.c[ 0 ].push_back( ( TF( i ) + TF( 0.5 ) + u( rng ) ) * h );
            nu.c[ 1 ].push_back( ( TF( j ) + TF( 0.5 ) + u( rng ) ) * h );
        }
    nu.w.assign( nu.c[ 0 ].size(), TF( 0 ) );
    nu.finish();
    return nu;
}

/// LE PLUS PROCHE de `x` dans l'arbre, par bornes sur les boites ( la meme coupure que `moins_psi`,
/// sans les poids ). Rend l'identifiant, pas le rang.
template<int D>
SI plus_proche( const AaBspT<D> &tr, const TF *x ) {
    SI best = -1;
    TF bd = std::numeric_limits<TF>::infinity();
    SI pile[ 128 ];
    int np = 0;
    pile[ np++ ] = 0;
    while ( np ) {
        const SI ni = pile[ --np ];
        const auto &nd = tr.nodes[ ni ];
        TF dd = 0;
        for ( int d = 0; d < D; ++d ) {
            const TF e = x[ d ] < nd.lo[ d ] ? nd.lo[ d ] - x[ d ] : ( x[ d ] > nd.hi[ d ] ? x[ d ] - nd.hi[ d ] : TF( 0 ) );
            dd += e * e;
        }
        if ( ! ( dd < bd ) ) continue;                   // la boite entiere est plus loin que le champion
        if ( nd.right < 0 ) {
            for ( SI k = nd.beg; k < nd.end; ++k ) {
                TF d2 = 0;
                for ( int d = 0; d < D; ++d ) { const TF e = x[ d ] - tr.p[ d ][ k ]; d2 += e * e; }
                if ( d2 < bd ) { bd = d2; best = tr.order[ k ]; }
            }
        } else if ( np + 2 <= 128 ) {
            pile[ np++ ] = ni + 1;
            pile[ np++ ] = nd.right;
        }
    }
    return best;
}

/// LA QUANTIFICATION OPTIMALE PONDEREE ( Lloyd ) : `q` minimise `sum_i nu_i | p_i - q_{r(i)} |^2`,
/// par alternance « chaque germe au `q` le plus proche » / « chaque `q` au barycentre pondere de son
/// paquet ». Le depart est la sortie du BSP -- des paquets deja equilibres en effectif, donc Lloyd
/// n'a plus qu'a en corriger la FORME, qui est justement le defaut des boites ( anisotropes des
/// qu'une feuille chevauche un bord d'amas, et `H_c^2` explose dans une direction ).
///
/// Les paquets VIDES sont retires : Lloyd peut en fabriquer, et un germe grossier sans masse n'a pas
/// de cible. `q` est rendu dans `qc`, et `pq.rep` porte le germe fin le plus proche de `q` -- dont
/// `prolonge_harmonique` a besoin ( il impose les poids grossiers SUR des germes fins ).
template<int D>
void quantifie( const TF *const *P, const std::vector<TF> &nuf, SI n, int nb_it, const Parallel &par,
                Paquets &pq, std::vector<TF> qc[ D ] ) {
    SI nb = pq.nb;
    for ( int d = 0; d < D; ++d ) qc[ d ].assign( nb, TF( 0 ) );
    std::vector<TF> poids( nb, TF( 0 ) );
    for ( SI i = 0; i < n; ++i ) {                       // le barycentre pondere des paquets du BSP
        const SI k = pq.paquet[ i ];
        for ( int d = 0; d < D; ++d ) qc[ d ][ k ] += nuf[ i ] * P[ d ][ i ];
        poids[ k ] += nuf[ i ];
    }
    for ( SI k = 0; k < nb; ++k )
        for ( int d = 0; d < D; ++d ) qc[ d ][ k ] /= std::max( poids[ k ], TF( 1e-300 ) );

    for ( int it = 0; it < nb_it; ++it ) {
        const TF *Q[ D ];
        for ( int d = 0; d < D; ++d ) Q[ d ] = qc[ d ].data();
        AaBspT<D> tr;
        tr.build( Q, nullptr, nb, 8 );
        parallel_for( n, par, [ & ]( SI i, int ) {
            TF x[ D ];
            for ( int d = 0; d < D; ++d ) x[ d ] = P[ d ][ i ];
            pq.paquet[ i ] = plus_proche<D>( tr, x );
        } );
        std::vector<TF> nc[ D ];
        for ( int d = 0; d < D; ++d ) nc[ d ].assign( nb, TF( 0 ) );
        poids.assign( nb, TF( 0 ) );
        for ( SI i = 0; i < n; ++i ) {
            const SI k = pq.paquet[ i ];
            for ( int d = 0; d < D; ++d ) nc[ d ][ k ] += nuf[ i ] * P[ d ][ i ];
            poids[ k ] += nuf[ i ];
        }
        for ( SI k = 0; k < nb; ++k )
            if ( poids[ k ] > 0 )
                for ( int d = 0; d < D; ++d ) qc[ d ][ k ] = nc[ d ][ k ] / poids[ k ];
    }

    // les paquets vides retires, et la renumerotation
    std::vector<SI> neuf( nb, -1 );
    SI nb2 = 0;
    for ( SI k = 0; k < nb; ++k ) if ( poids[ k ] > 0 ) neuf[ k ] = nb2++;
    if ( nb2 < nb ) {
        for ( int d = 0; d < D; ++d ) {
            std::vector<TF> c( nb2 );
            for ( SI k = 0; k < nb; ++k ) if ( neuf[ k ] >= 0 ) c[ neuf[ k ] ] = qc[ d ][ k ];
            qc[ d ].swap( c );
        }
        for ( SI i = 0; i < n; ++i ) pq.paquet[ i ] = neuf[ pq.paquet[ i ] ];
        nb = nb2;
    }
    pq.nb = nb;
    pq.taille.assign( nb, 0 );
    for ( SI i = 0; i < n; ++i ) ++pq.taille[ pq.paquet[ i ] ];

    // le representant : le germe fin le plus proche de `q`
    pq.rep.assign( nb, -1 );
    std::vector<TF> dmin( nb, std::numeric_limits<TF>::infinity() );
    for ( SI i = 0; i < n; ++i ) {
        const SI k = pq.paquet[ i ];
        TF d2 = 0;
        for ( int d = 0; d < D; ++d ) { const TF e = P[ d ][ i ] - qc[ d ][ k ]; d2 += e * e; }
        if ( d2 < dmin[ k ] ) { dmin[ k ] = d2; pq.rep[ k ] = i; }
    }
}

struct Bilan {
    int    it = 0, diag = 0, recul = 0;
    double temps = 0;
    TF     reste = 0;
    bool   ok = true;
};

/// LA CONTINUATION EN LARGEUR sur le nuage que `nw` porte. `part[ i ]` est la FRACTION de la masse
/// totale visee par le germe `i` -- `1/n` sur le nuage complet, `taille_paquet / n` sur le grossier :
/// la cible est `part[ i ] * M( s )`, et elle change a chaque etape puisque `M` en depend.
template<class PD>
Bilan continuation( Newton<PD> &nw, Densite &rho, const std::vector<TF> &liste, const std::vector<TF> &part,
                    std::vector<TF> &w, const char *quoi, const std::function<void( TF )> &apres = {} ) {
    Bilan b;
    const SI n = SI( part.size() );
    const double t0 = now();
    for ( SI e = 0; e < SI( liste.size() ); ++e ) {
        rho.s = liste[ e ];
        const TF M = rho.masse_carre();
        nw.nu.resize( n );
        for ( SI i = 0; i < n; ++i ) nw.nu[ i ] = part[ i ] * M;
        nw.st = NewtonStats{};
        nw.resout( w );
        w = nw.w;
        b.it += nw.st.nb_iter; b.diag += nw.st.nb_diag; b.recul += nw.st.nb_recul;
        b.reste = nw.st.reste;
        std::printf( "  %s s = %-10.5g  %2d it, %3d diag ( %d reculs ), reste %.2e  %s\n",
                     quoi, double( liste[ e ] ), nw.st.nb_iter, nw.st.nb_diag, nw.st.nb_recul,
                     double( nw.st.reste ), nw.st.fin );
        if ( std::string( nw.st.fin ) != "CONVERGE" && nw.st.reste > 10 * nw.o.tol ) { b.ok = false; break; }
        if ( apres ) apres( liste[ e ] );
    }
    b.temps = now() - t0;
    return b;
}

template<class PD>
int lance( const Args &a, const Opts &o, const Nuage<2> &nu0, Lineaire &lin ) {
    constexpr int D = 2;
    const SI n = nu0.n;
    Densite rho = densite_jeu( o.sigma, o.nb_gauss, o.gauss, o.plancher );
    rho.chemin = Densite::CONV;
    std::printf( "  densite : %d gaussiennes, sigma %g, plancher %.3f  ;  germes %s, n = %d  ;  %d etapes en s\n",
                 int( rho.g.size() ), double( o.sigma ), double( rho.plancher ), o.diracs.c_str(), int( n ), int( o.liste.size() ) );
#ifdef _OPENMP
    omp_set_num_threads( a.par.threads );
#endif

    PD pd;
    pd.build( nu0.P, nullptr, n, a.leaf );
    Newton<PD> nw( pd, lin, nu0.P, a.par, o.newton );
    nw.rho = &rho;
    const std::vector<TF> part_fin( n, TF( 1 ) / n );

    // ---- LA REFERENCE : la continuation sur le nuage COMPLET, dans CE binaire ( deux binaires du
    //      meme code compiles differemment s'ecartent de 4 % -- la comparaison se fait ici )
    if ( o.reference ) {
        std::vector<TF> w( n, TF( 0 ) );
        const Bilan b = continuation( nw, rho, o.liste, part_fin, w, "reference" );
        std::printf( "\n  REFERENCE : %d iterations, %d diagrammes ( %d reculs ), %.2f s, reste %.2e  --  %s\n",
                     b.it, b.diag, b.recul, b.temps, double( b.reste ), b.ok ? "converge" : "ECHEC" );
        return b.ok ? 0 : 1;
    }

    // ---- 1. LES PAQUETS
    double t0 = now();
    Paquets pq = fait_paquets<D>( nu0.P, n, o.R );
    std::vector<TF> qc[ D ];
    if ( o.paq == "lloyd" )
        quantifie<D>( nu0.P, part_fin, n, o.lloyd_it, a.par, pq, qc );
    else if ( o.paq == "bsp-bary" ) {
        for ( int d = 0; d < D; ++d ) qc[ d ].assign( pq.nb, TF( 0 ) );
        for ( SI i = 0; i < n; ++i )
            for ( int d = 0; d < D; ++d ) qc[ d ][ pq.paquet[ i ] ] += nu0.P[ d ][ i ] / TF( pq.taille[ pq.paquet[ i ] ] );
    } else if ( o.paq == "bsp" ) {
        for ( int d = 0; d < D; ++d ) {
            qc[ d ].resize( pq.nb );
            for ( SI k = 0; k < pq.nb; ++k ) qc[ d ][ k ] = nu0.P[ d ][ pq.rep[ k ] ];
        }
    } else { std::printf( "  paquets inconnus : %s\n", o.paq.c_str() ); return 1; }
    const double t_paq = now() - t0;
    const SI nc = pq.nb;
    const TF *Pc[ D ];
    for ( int d = 0; d < D; ++d ) Pc[ d ] = qc[ d ].data();

    // LA DISTORSION, en unites de l'espacement local : le seul chiffre qui juge une agregation
    // AVANT de resoudre quoi que ce soit ( § 23.12 pour l'objectif, § 8.2 pour la prolongation ).
    Laplacien Lvor;
    std::vector<TF> avor;
    laplacien_de( pd, nu0.P, nullptr, a.par, Lvor, &avor );
    const TF amin_vor = *std::min_element( avor.begin(), avor.end() );
    std::vector<TF> h2( n, TF( 0 ) );
    for ( SI i = 0; i < n; ++i ) {
        TF s = 0; int nh = 0;
        for ( SI e = Lvor.row[ i ]; e < Lvor.row[ i + 1 ]; ++e ) {
            const SI j = Lvor.col[ e ];
            TF d2 = 0;
            for ( int d = 0; d < D; ++d ) { const TF t = nu0.P[ d ][ j ] - nu0.P[ d ][ i ]; d2 += t * t; }
            s += d2; ++nh;
        }
        h2[ i ] = nh ? s / nh : TF( 1 ) / n;
    }
    TF dist = 0, dist_rel = 0, pire_rayon = 0;
    SI tmin = n, tmax = 0;
    for ( SI i = 0; i < n; ++i ) {
        const SI k = pq.paquet[ i ];
        TF d2 = 0;
        for ( int d = 0; d < D; ++d ) { const TF e = nu0.P[ d ][ i ] - qc[ d ][ k ]; d2 += e * e; }
        dist += d2 / n;                                  // `sum nu_i | p_i - q |^2`, `nu_i = 1/n`
        dist_rel += d2 / h2[ i ] / n;
        pire_rayon = std::max( pire_rayon, d2 / h2[ i ] );
    }
    for ( SI k = 0; k < nc; ++k ) { tmin = std::min( tmin, pq.taille[ k ] ); tmax = std::max( tmax, pq.taille[ k ] ); }
    std::printf( "  paquets %s : %d -> %d germes ( R = %d, tailles %d a %d ), %.3f s\n",
                 o.paq.c_str(), int( n ), int( nc ), int( o.R ), int( tmin ), int( tmax ), t_paq );
    std::printf( "    DISTORSION sum nu |p - q|^2 = %.4e  ;  en unites de h^2 local : moyenne %.3f, PIRE %.1f\n",
                 double( dist ), double( dist_rel ), double( pire_rayon ) );

    // ---- 2. LA PROLONGATION ET SON VERDICT, en une lambda -- parce qu'on veut pouvoir l'appeler a
    //         CHAQUE etape en `s` ( `--balaye` ) et pas seulement a la derniere. Ce qui nous interesse
    //         n'est pas « est-ce que ca echoue » mais A PARTIR DE QUEL `s` ca echoue.
    PD pdc;
    pdc.build( Pc, nullptr, nc, a.leaf );
    NewtonOptions oc = o.newton;
    if ( o.tolg > 0 ) oc.tol = o.tolg;
    Newton<PD> nwc( pdc, lin, Pc, a.par, oc );
    nwc.rho = &rho;
    std::vector<TF> part_c( nc );
    for ( SI k = 0; k < nc; ++k ) part_c[ k ] = TF( pq.taille[ k ] ) / TF( n );
    std::vector<TF> wc( nc, TF( 0 ) );

    std::vector<TF> w;
    double t_prol = 0;
    bool prol_ok = true;
    // `rho.s` vaut DEJA l'etape courante quand la lambda est appelee ; la cible fine en decoule.
    // LE JUGEMENT d'un depart quelconque : les trois nombres qui comptent. `nw.nu` doit etre a jour.
    // LES DEUX COTES DU RESIDU, SEPARES -- et c'est indispensable sur une densite. `max|a-nu|/nu` est
    // borne par 1 du cote AFFAME ( une cellule vide donne 1 ) : tout ce qui depasse 1 vient donc du
    // cote GAVE, `max a/nu`. Les confondre, c'est ne pas savoir de quoi un depart souffre.
    //   `amin` = min a/nu ( la famine ),  `amax` = max a/nu ( le gavage ),  `gaves` = combien au-dela de 10.
    auto juge = [ & ]( const std::vector<TF> &ww, SI &mauv, TF &amin, TF &amax, SI &gaves, TF &pire, TF &mer ) {
        std::vector<TF> af;
        std::vector<Facette> fa;
        nw.mesures_et_facettes( ww, af, fa );
        mauv = 0; gaves = 0;
        amin = std::numeric_limits<TF>::infinity(); amax = 0; pire = 0;
        for ( SI i = 0; i < n; ++i ) {
            const TF x = af[ i ] / nw.nu[ i ];
            amin = std::min( amin, x );
            amax = std::max( amax, x );
            gaves += x > 10;
            pire = std::max( pire, std::fabs( af[ i ] - nw.nu[ i ] ) / nw.nu[ i ] );
            mauv += af[ i ] < o.seuil * std::min( nw.nu[ i ], amin_vor );
        }
        mer = nw.merite( af );
    };

    auto prolonge_et_juge = [ & ]( const char *quoi ) {
        const TF M = rho.masse_carre();
        nw.nu.assign( n, M / n );
        const double tp = now();
        Laplacien Lc, Llag;
        if ( o.prol == "ctransf" ) laplacien_de( pdc, Pc, wc.data(), a.par, Llag );  // le graphe de LAGUERRE grossier
        if ( o.prol == "mls" )     laplacien_de( pdc, Pc, nullptr, a.par, Lc );      // le graphe de VORONOI grossier
        SI retombes = 0;
        if ( o.prol == "copie" )           prolonge_copie( pq, wc, w );
        else if ( o.prol == "harmonique" ) prolonge_harmonique( Lvor, pq, wc, std::numeric_limits<TF>::infinity(), w );
        else if ( o.prol == "mls" )        retombes = prolonge_mls<D>( nu0.P, pq, Pc, Lc, wc, a.par, w, o.mls_anneaux, o.mls_largeur );
        else if ( o.prol == "ctransf" )    prolonge_ctransf<D>( nu0.P, pq, Pc, Llag, wc, w );
        else { prol_ok = false; return; }
        if ( o.prol_lisse ) lisse_jacobi( Lvor, o.prol_lisse, w );   // `Prolongation.h`
        t_prol += now() - tp;

        // LES QUATRE NOMBRES. Ce n'est PAS `teste_admissible` ( qui mesure du Lebesgue ) : la mesure
        // doit etre celle de la DENSITE, donc celle de Newton lui-meme.
        SI mauv, gaves; TF amin, amax, pire, mer;
        juge( w, mauv, amin, amax, gaves, pire, mer );
        // ET LE TEMOIN, au MEME `s` : Voronoi -- le pire residu qu'un solve DIRECT verrait a cette
        // largeur, donc ce que la prolongation doit battre pour servir a quelque chose.
        std::vector<TF> w0( n, TF( 0 ) );
        SI m0, g0; TF am0, ax0, pire0, mer0;
        juge( w0, m0, am0, ax0, g0, pire0, mer0 );
        std::printf( "  %s s = %-10.5g prol %-10s : %6d sous le plancher ( %5.2f %% ), aire min %10.2e nu,"
                     "  PIRE residu %.3e  ( Voronoi : %.3e )%s\n",
                     quoi, double( rho.s ), o.prol.c_str(), int( mauv ), 100.0 * double( mauv ) / double( n ),
                     double( amin ), double( pire ), double( pire0 ),
                     retombes ? ( ", " + std::to_string( retombes ) + " germes retombes" ).c_str() : "" );
    };

    // ---- 3. LA CONTINUATION GROSSIERE
    const Bilan bc = continuation( nwc, rho, o.liste, part_c, wc, "grossier ",
                                   o.balaye ? std::function<void( TF )>( [ & ]( TF ) { prolonge_et_juge( "  verdict" ); } )
                                            : std::function<void( TF )>() );
    std::printf( "  GROSSIER : %d iterations, %d diagrammes, %.2f s, reste %.2e  --  %s\n",
                 bc.it, bc.diag, bc.temps, double( bc.reste ), bc.ok ? "converge" : "ECHEC" );
    if ( ! bc.ok ) return 1;
    if ( o.balaye ) return 0;

    // ---- 4. LE VERDICT A LA DENSITE FINALE
    rho.s = o.liste.back();
    prolonge_et_juge( "PROLONGATION" );
    if ( ! prol_ok ) { std::printf( "  prolongation inconnue : %s\n", o.prol.c_str() ); return 1; }

    // ---- 5. LE SPAN A DIAGRAMME ET `L` GELES ( `--span K` )
    if ( o.span > 0 ) {
    // =====================================================================================
    // LE SPAN A DIAGRAMME ET `L` GELES ( `--span K` ) -- inclus par `main_grossier.cpp`.
    //
    // LE SCHEMA MESURE, exactement celui demande et rien de plus : on part d'un point ADMISSIBLE
    // ( Voronoi ), on paie UN diagramme et UNE factorisation, et on ne les refait plus. On minimise le
    // merite sur le span, on regarde ce qui manque, on en fait une direction de plus ( orthogonalisee ),
    // et on recommence. La question est le COMPORTEMENT QUAND `k` MONTE, pas le cout.
    //
    // CE QUI EST GELE, ET CE QUI NE PEUT PAS L'ETRE. `L` est gele : chaque direction ajoutee est une
    // re-resolution sur la MEME factorisation ( `lin.resout_encore` ), pas une factorisation de plus.
    // En revanche le MERITE est evalue avec les vraies masses : `PolyMulti` donne l'aire exacte sur le
    // span ( les sommets sont affines dans les coefficients, donc l'aire est quadratique ), mais la
    // MASSE sous des gaussiennes n'en est pas un polynome ( § 9.7 ) -- minimiser le residu d'AIRE serait
    // minimiser autre chose que ce qu'on veut. On paie donc une mesure par evaluation. Ca ne change pas
    // la reponse a la question posee, seulement son prix, et le prix est le sujet d'apres.
    //
    // LE RAYON DE VALIDITE est rapporte parce qu'il est deja calcule : `PolyMulti::rayon` est le plus
    // grand `|t|inf` sous lequel aucune arete ne s'annule, donc le domaine ou le modele gele est exact.
    // =====================================================================================

    // la moyenne retiree : les poids sont definis a une constante pres, autant que les directions le soient
    auto centre = [ & ]( std::vector<TF> &v ) {
        TF m = 0;
        for ( SI i = 0; i < n; ++i ) m += v[ i ];
        m /= TF( n );
        for ( SI i = 0; i < n; ++i ) v[ i ] -= m;
    };
    auto pscal = [ & ]( const std::vector<TF> &u, const std::vector<TF> &v ) {
        TF s = 0;
        for ( SI i = 0; i < n; ++i ) s += u[ i ] * v[ i ];
        return s;
    };

    // ---- LE POINT DE BASE : Voronoi, a la densite du dernier `s`. UN diagramme.
    nw.res_cur = o.newton.residu;            // sinon le merite reste `lin` : `res_cur` n'est pose que dans `resout`
    std::vector<TF> wb( n, TF( 0 ) ), ab;
    std::vector<Facette> fab;
    nw.mesures_et_facettes( wb, ab, fab );
    Laplacien L;
    L.assemble( n, fab );
    TF amin_base = INFINI, plancher_span = 0;
        {
            TF ax = 0;
            for ( SI i = 0; i < n; ++i ) { amin_base = std::min( amin_base, ab[ i ] / nw.nu[ i ] ); ax = std::max( ax, ab[ i ] / nw.nu[ i ] ); }
            plancher_span = o.span_garde * amin_base;   // 0 par defaut : seul `a_i > 0` est exige
            std::printf( "  base = Voronoi a s = %g : merite %.4e, min a/nu %.3e, max a/nu %.1f  ( plancher du span %.3e )\n",
                         double( rho.s ), double( nw.merite( ab ) ), double( amin_base ), double( ax ), double( plancher_span ) );
        }

    // ---- LA PREMIERE DIRECTION : la prolongation grossiere. LA DEUXIEME : Newton au point de base, sur
    //      la factorisation qu'on garde pour tout le reste.
    std::vector<std::vector<TF>> D;
    {
        std::vector<TF> d0 = w;                              // `w` porte la prolongation ( § 3 )
        for ( SI i = 0; i < n; ++i ) d0[ i ] -= wb[ i ];
        centre( d0 );
        D.push_back( d0 );
    }
    std::vector<TF> b, dn( n );
    nw.membre_de( ab, nw.res_cur, nw.o.puis, b );
    const bool fact_ok = lin.resout( L, b, dn );
    if ( ! fact_ok ) { std::printf( "  la factorisation a echoue\n" ); return 1; }
    std::printf( "  factorisation faite ( %s ), re-resolutions possibles : %s\n",
                 lin.nom(), lin.sait_encore() ? "oui" : "NON ( chaque direction coutera une factorisation )" );

    // le COSINUS entre la direction grossiere et celle de Newton -- le garde-fou du § 24.12, a lire avant
    // tout le reste : deux directions colineaires ne font pas un span.
    {
        std::vector<TF> dc = dn;
        centre( dc );
        const TF c = pscal( D[ 0 ], dc ) / std::sqrt( std::max( pscal( D[ 0 ], D[ 0 ] ) * pscal( dc, dc ), TF( 1e-300 ) ) );
        std::printf( "  COSINUS( w_prol , d_newton ) = %+.4f\n", double( c ) );
    }

    // ---- L'EVALUATION : le merite vrai en `wb + sum t_k D_k`, et l'admissibilite. Un point non
    //      admissible est REFUSE ( merite infini ) : le schema ne quitte jamais l'admissible.
    std::vector<TF> wt( n ), at;
    std::vector<Facette> fat;
    int nb_eval = 0;
    auto evalue = [ & ]( const std::vector<TF> &t, TF &amin, TF &amax ) {
        for ( SI i = 0; i < n; ++i ) {
            TF v = wb[ i ];
            for ( size_t k = 0; k < t.size(); ++k ) v += t[ k ] * D[ k ][ i ];
            wt[ i ] = v;
        }
        ++nb_eval;
        // LE RESIDU, REPOSE A CHAQUE FOIS. `nw.resout` ( le temoin `k = 0`, et le « il reste combien »
        // de chaque `k` ) ecrase `res_cur`, et la bascule `log -> lin` du defaut ( § 24.5 ) le laisse a
        // `lin` des que le solve converge. Sans ca les lignes du tableau ne sont pas sur la meme echelle.
        nw.res_cur = o.newton.residu;
        nw.mesures_et_facettes( wt, at, fat );
        amin = INFINI; amax = 0;
        for ( SI i = 0; i < n; ++i ) {
            const TF x = at[ i ] / nw.nu[ i ];
            amin = std::min( amin, x );
            amax = std::max( amax, x );
        }
        // LE CARRE du merite : `nw.merite` rend la NORME, et le jacobien du span derive la somme des
        // carres. Les melanger fait un facteur `2 * merite` sur le gradient -- ce qui a failli me faire
        // conclure a un jacobien faux ( -2.565e4 contre -46.6 en differences finies, soit exactement
        // `2 x 275` ). Le minimiseur travaille donc sur `S = merite^2`, dont l'argmin est le meme.
        if ( ! ( amin > 0 && amin >= plancher_span ) ) return INFINI;
        const TF m = nw.merite( at );
        return m * m;
    };

    // ---- LA MINIMISATION SUR LE SPAN : des recherches 1-D cycliques. Le merite sur le span n'est pas
    //      quadratique ( la masse ne l'est pas ), donc pas de formule : on balaye, on raffine, on cycle.
    std::vector<TF> t;
    // LA MINIMISATION : une GRILLE FINE, coordonnee par coordonnee, resserree a chaque balayage. Rien
    // de malin -- pas de gradient, justement parce qu'en `lin` avec un plancher dur l'optimum est SUR
    // le bord du domaine admissible, et qu'un gradient n'y mene pas. Le premier balayage couvre
    // `[ -0.5, 1.5 ]` ( `alpha_0 = 1` est dedans ), les suivants raffinent autour du meilleur point.
    // Les points non admissibles sont refuses : le blocage se fait AVANT que la cellule se vide.
    // ALPHA* LE LONG D'UNE DIRECTION, par bissection sur de vrais diagrammes. Ce n'est plus le
    // parametre de la recherche ( la barriere du `log` s'en charge ) mais le DIAGNOSTIC qui decide :
    // la mesure 1D dit que le seuil est `alpha* > 1`, en transition de phase et non en continuum
    // ( `alpha* = 0.30` -> 53 iterations, `0.62` -> 42, `1.07` -> 10 ). On le lit AVANT tout le reste.
    auto alpha_etoile = [ & ]( const std::vector<TF> &d, TF haut = 4 ) {
        std::vector<TF> ww( n ), aa;
        std::vector<Facette> ff;
        auto ok = [ & ]( TF al ) {
            for ( SI i = 0; i < n; ++i ) ww[ i ] = wb[ i ] + al * d[ i ];
            nw.mesures_et_facettes( ww, aa, ff );
            for ( SI i = 0; i < n; ++i ) if ( ! ( aa[ i ] > 0 ) ) return false;
            return true;
        };
        if ( ok( haut ) ) return haut;                   // la direction ne vide rien jusqu'a `haut`
        TF lo = 0, hi = haut;
        for ( int it = 0; it < 40 && hi - lo > 1e-6 * std::max( hi, TF( 1e-6 ) ); ++it ) {
            const TF mi = TF( 0.5 ) * ( lo + hi );
            if ( ok( mi ) ) lo = mi; else hi = mi;
        }
        return lo;
    };

    // LE PRODUIT `L v`, pour le jacobien du span
    auto Lfois = [ & ]( const Laplacien &L, const std::vector<TF> &v, std::vector<TF> &r ) {
        r.assign( n, TF( 0 ) );
        for ( SI i = 0; i < n; ++i ) {
            TF x = L.dia[ i ] * v[ i ];
            for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) x -= L.c[ e ] * v[ L.col[ e ] ];
            r[ i ] = x;
        }
    };

    // GAUSS-NEWTON DANS L'ESPACE DES `alpha`. `log2` etant une somme de carres, le pas resout
    // `min | r + J da |^2` avec `r_i = g( a_i / nu_i ) - moyenne` et `J_ik = ( L d_k )_i / a_i` -- c'est
    // `da = d g / d alpha` en passant par `da = L dw`, exact a combinatoire figee. Un solve `k x k`.
    //
    // PAS DE `alpha*` DANS LA RECHERCHE, et c'est le point : avec le residu `log` la barriere est DANS
    // l'objectif ( `a_i -> 0` donne `+inf` ), donc la recherche de pas recule d'elle-meme et le bord
    // n'est jamais franchi. Le grille du debut etait mal echelonnee ( elle balayait `[ -0.5, 1.5 ]`
    // quand `alpha*` valait 1e-3 ) et rendait `alpha_0 = 0` par artefact ; elle reste sous
    // `--span-grille` pour comparaison.
    auto minimise_gn = [ & ]( int itmax, int &gn_it, TF &norme_grad ) {
        const int k = int( t.size() );
        std::vector<TF> aa, r, col, grad( k );
        std::vector<Facette> ff;
        Laplacien Lc2;
        Eigen::MatrixXd JtJ( k, k );
        Eigen::VectorXd Jtr( k ), da( k );
        TF amin, amax;
        TF best = evalue( t, amin, amax );
        gn_it = 0; norme_grad = 0;
        for ( int it = 1; it <= itmax; ++it ) {
            // l'etat au point courant : les masses, les facettes, `L`
            for ( SI i = 0; i < n; ++i ) {
                TF v = wb[ i ];
                for ( int j = 0; j < k; ++j ) v += t[ j ] * D[ j ][ i ];
                wt[ i ] = v;
            }
            nw.mesures_et_facettes( wt, aa, ff );
            bool sain = true;
            for ( SI i = 0; i < n; ++i ) if ( ! ( aa[ i ] > 0 ) ) { sain = false; break; }
            if ( ! sain ) break;
            Lc2.assemble( n, ff );
            r.assign( n, TF( 0 ) );
            TF moy = 0;
            for ( SI i = 0; i < n; ++i ) { r[ i ] = std::log( aa[ i ] / nw.nu[ i ] ); moy += r[ i ]; }
            moy /= TF( n );
            for ( SI i = 0; i < n; ++i ) r[ i ] -= moy;
            std::vector<std::vector<TF>> J( k );
            for ( int j = 0; j < k; ++j ) {
                Lfois( Lc2, D[ j ], col );
                J[ j ].resize( n );
                for ( SI i = 0; i < n; ++i ) J[ j ][ i ] = col[ i ] / aa[ i ];
            }
            for ( int j = 0; j < k; ++j ) {
                TF g = 0;
                for ( SI i = 0; i < n; ++i ) g += J[ j ][ i ] * r[ i ];
                grad[ j ] = 2 * g;
                Jtr( j ) = -g;
                for ( int l = 0; l <= j; ++l ) {
                    TF v = 0;
                    for ( SI i = 0; i < n; ++i ) v += J[ j ][ i ] * J[ l ][ i ];
                    JtJ( j, l ) = JtJ( l, j ) = v;
                }
            }
            norme_grad = 0;
            for ( int j = 0; j < k; ++j ) norme_grad += grad[ j ] * grad[ j ];
            norme_grad = std::sqrt( norme_grad );
            da = JtJ.ldlt().solve( Jtr );
            // la recherche de pas : des moities tant que `log2` n'est pas fini ou ne descend pas
            const std::vector<TF> t0 = t;
            TF pas = 1, f2 = INFINI;
            bool pris = false;
            for ( int e = 0; e < 40; ++e, pas /= 2 ) {
                for ( int j = 0; j < k; ++j ) t[ j ] = t0[ j ] + pas * TF( da( j ) );
                f2 = evalue( t, amin, amax );
                if ( f2 < best ) { pris = true; break; }
            }
            if ( ! pris ) { t = t0; break; }
            gn_it = it;
            const bool fini = std::fabs( best - f2 ) <= TF( 1e-12 ) * std::fabs( best );
            best = f2;
            if ( fini ) break;
        }
        return best;
    };

    // LE TEMOIN, `k = 0` : Newton depuis la base elle-meme, au meme `s`, dans le meme binaire.
        {
            nw.st = NewtonStats{};
            const double t0 = now();
            nw.resout( wb );
            std::printf( "  TEMOIN k=0 ( Newton depuis la base ) : %d it, %d diag ( %d reculs ), %.2f s, reste %.2e, %s\n",
                         nw.st.nb_iter, nw.st.nb_diag, nw.st.nb_recul, now() - t0, double( nw.st.reste ), nw.st.fin );
        }
        // LE GRADIENT CONTRE DES DIFFERENCES FINIES CENTREES. Sans ce controle, un `|grad|` enorme
        // sans descente ne se distingue pas d'un jacobien faux -- et c'est exactement ce qu'on voit.
        {
            const int k = 1;
            std::vector<TF> aa, ff_r, col;
            std::vector<Facette> ff;
            Laplacien Lc2;
            nw.mesures_et_facettes( wb, aa, ff );
            Lc2.assemble( n, ff );
            Lfois( Lc2, D[ 0 ], col );
            TF moy = 0;
            std::vector<TF> r( n );
            for ( SI i = 0; i < n; ++i ) { r[ i ] = std::log( aa[ i ] / nw.nu[ i ] ); moy += r[ i ]; }
            moy /= TF( n );
            TF ana = 0;
            for ( SI i = 0; i < n; ++i ) ana += 2 * ( col[ i ] / aa[ i ] ) * ( r[ i ] - moy );
            std::vector<TF> tt( 1 );
            TF amin, amax;
            const TF hh = 1e-6;
            tt[ 0 ] = hh;  const TF fp = evalue( tt, amin, amax );
            tt[ 0 ] = -hh; const TF fm = evalue( tt, amin, amax );
            const TF num = ( fp - fm ) / ( 2 * hh );
            std::printf( "  CONTROLE du gradient du span ( k = 1, en t = 0 ) : analytique %.6e,"
                         " differences finies %.6e, ecart relatif %.2e\n",
                         double( ana ), double( num ), double( std::fabs( ana - num ) / std::max( std::fabs( num ), TF( 1e-300 ) ) ) );
            (void) k; (void) ff_r;
        }
        std::printf( "  ALPHA* le long de w_prol depuis la base = %.4e"
                     "   ( la mesure 1D dit : le seuil est 1 )\n", double( alpha_etoile( D[ 0 ] ) ) );
        std::printf( "  k  |  merite         |  min a/nu   |  max a/nu  |  coef w_prol  |  gn it |    |grad| |  diag  ||  IT RESTANTES  |  diag ( reculs )  |  reste  |  fin\n" );
    for ( int k = 1; k <= o.span && k <= int( PolyMulti::KMAX ); ++k ) {
        if ( int( D.size() ) < k ) break;
        t.resize( k, TF( 0 ) );
        nb_eval = 0;
        int gn_it = 0;
        TF ngrad = 0;
        const TF mer = minimise_gn( 60, gn_it, ngrad );
        // l'etat au minimum : les masses, et le rayon de validite du modele gele sur CE span
        TF amin, amax;
        evalue( t, amin, amax );
        const std::vector<TF> w_opt = wt, a_opt = at;
        // LES CONTRAINTES ACTIVES : les cellules posees sur le plancher. C'est ce qui dit si « projeter
        // sur le bord » est une projection ( quelques contraintes ) ou un programme lineaire ( des
        // milliers ) -- et donc si un span de quelques directions peut suivre ce bord.
        (void) plancher_span;
        TF rayon = INFINI, pave = 0, tinf = 0;
        {
            std::vector<const TF *> dp( k );
            for ( int j = 0; j < k; ++j ) dp[ j ] = D[ j ].data();
            std::vector<PolyMulti> pm;
            polynomes_multi( pd, nu0.P, wb, dp.data(), k, a.par, pm );
            // L'ECART DE PAVAGE, qui est la BONNE mesure de l'erreur du modele gele : une arete qui
            // meurt ne change RIEN a l'aire ( une facette de longueur nulle contribue zero ), donc un
            // rayon par arete flague des evenements qui ne coutent rien. Les cellules PAVENT le carre,
            // donc `sum_i a_i^poly = 1` exactement si le modele est juste ; tout ecart est la somme des
            // recouvrements, c'est-a-dire son erreur `L1` -- et de signe connu, puisque le modele
            // ignore des coupes et SURESTIME. Gratuit, et sans un diagramme.
            std::vector<TF> tt( k );
            for ( int j = 0; j < k; ++j ) { tt[ j ] = t[ j ]; tinf = std::max( tinf, std::fabs( t[ j ] ) ); }
            for ( const PolyMulti &p : pm ) {
                if ( p.etat != PolyCellule::OK ) continue;
                rayon = std::min( rayon, p.rayon );
                pave += p( tt.data(), k );
            }
            pave -= 1;
        }
        (void) rayon; (void) tinf;
        // ET LA SEULE METRIQUE : ce que Newton coute DEPUIS CE POINT.
        nw.st = NewtonStats{};
        const double t0 = now();
        nw.resout( w_opt );
        std::printf( "  %d  |  %.6e  |  %9.3e  |  %8.1f  |  %+9.4f  |  %5d | %8.1e  |  %5d  ||  %4d  |  %4d ( %3d )  |  %.2e  |  %s\n",
                     k, double( mer ), double( amin ), double( amax ), double( t[ 0 ] ), gn_it, double( ngrad ), nb_eval,
                     nw.st.nb_iter, nw.st.nb_diag, nw.st.nb_recul, double( nw.st.reste ), nw.st.fin );

        // ---- CE QUI MANQUE : la direction de Newton AU MINIMUM, sur la factorisation GELEE, puis
        //      orthogonalisee contre le span ( sinon elle n'apporte rien de neuf ).
        if ( k == o.span || k == int( PolyMulti::KMAX ) ) break;
        nw.membre_de( a_opt, nw.res_cur, nw.o.puis, b );
        std::vector<TF> dnk( n );
        if ( lin.sait_encore() ) lin.resout_encore( b, dnk );
        else                     lin.resout( L, b, dnk );
        centre( dnk );
        for ( int j = 0; j < int( D.size() ); ++j ) {
            const TF c = pscal( dnk, D[ j ] ) / std::max( pscal( D[ j ], D[ j ] ), TF( 1e-300 ) );
            for ( SI i = 0; i < n; ++i ) dnk[ i ] -= c * D[ j ][ i ];
        }
        const TF nrm = std::sqrt( pscal( dnk, dnk ) );
        if ( ! ( nrm > 0 ) ) { std::printf( "    la direction ajoutee est DANS le span ( norme nulle ) : rien a ajouter\n" ); break; }
        std::printf( "    direction %d ajoutee : norme apres orthogonalisation %.3e\n", int( D.size() ) + 1, double( nrm ) );
        D.push_back( dnk );
    }
    return 0;

    }

    // ---- 6. LA SEULE METRIQUE QUI COMPTE : COMBIEN D'ITERATIONS RESTE-T-IL ( `--reste` ).
    //          Ni `alpha_0`, ni les cellules sous un plancher, ni le merite : ce que Newton coute
    //          depuis ce depart. `alpha_0 = 0` EST Voronoi, donc le temoin est dans le meme tableau, au
    //          meme `s`, dans le meme binaire. Le second temoin est l'etape de la continuation elle-meme
    //          ( qui part de la solution du `s` precedent, bien meilleure que Voronoi ).
    if ( o.reste ) {
        std::printf( "  Newton depuis `alpha_0 * w_prol` a s = %g -- ce qu'il RESTE\n", double( rho.s ) );
        std::printf( "    alpha_0  |  it  |  diag ( reculs )  |  temps  |  reste  |  fin\n" );
        std::vector<TF> wa( n );
        const TF fs[] = { 0, 0.35, 0.6, 0.8, 1.0 };
        for ( TF f : fs ) {
            for ( SI i = 0; i < n; ++i ) wa[ i ] = f * w[ i ];
            nw.st = NewtonStats{};
            const double t = now();
            nw.resout( wa );
            std::printf( "    %7.3f  |  %3d  |  %4d ( %3d )  |  %6.2f s  |  %.2e  |  %s\n",
                         double( f ), nw.st.nb_iter, nw.st.nb_diag, nw.st.nb_recul, now() - t,
                         double( nw.st.reste ), nw.st.fin );
        }
        return 0;
    }

    // ---- 7. JUSQU'OU LA DIRECTION PORTE ( `--echelle` ). C'est le `t` de la correction `retrait`
    //         du § 8.1, remesure sur une densite : le plus grand `alpha_0` tel que
    //         `alpha_0 * w_prol` soit encore admissible. Si ce `alpha_0` est minuscule, aucun span
    //         de deux ou trois directions ne le remontera a 1 -- le deficit de la prolongation est
    //         PAR GERME ( § 8.2 : la composante `( 1 - m_i ) h^2` ), pas dans un sous-espace.
    if ( o.echelle ) {
        std::printf( "  alpha_0 : le verdict de `alpha_0 * w_prol` a s = %g\n", double( rho.s ) );
        std::vector<TF> wa( n );
        std::printf( "    alpha_0  |  min a/nu ( famine )  |  max a/nu ( gavage )  |  cellules > 10 nu  |  PIRE  |  merite\n" );
        const TF fs[] = { 1.4, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01, 0 };
        for ( TF f : fs ) {
            for ( SI i = 0; i < n; ++i ) wa[ i ] = f * w[ i ];
            SI mauv, gaves; TF amin, amax, pire, mer;
            juge( wa, mauv, amin, amax, gaves, pire, mer );
            std::printf( "    %7.3f  |  %18.3e  |  %18.1f  |  %16d  |  %6.1f  |  %.4e\n",
                         double( f ), double( amin ), double( amax ), int( gaves ), double( pire ), double( mer ) );
        }
        return 0;
    }

    // ---- 8. LE NEWTON FIN, le seul chiffre qui tranche
    if ( ! o.fin ) return 0;
    nw.st = NewtonStats{};
    t0 = now();
    nw.resout( w );
    const double t_fin = now() - t0;
    std::printf( "  FIN : %d iterations, %d diagrammes ( %d reculs ), %.2f s, reste %.2e  --  %s\n",
                 nw.st.nb_iter, nw.st.nb_diag, nw.st.nb_recul, t_fin, double( nw.st.reste ), nw.st.fin );
    std::printf( "\n  TOTAL : %d diagrammes, %.2f s  ( grossier %d / %.2f s, paquets + prolongation %.2f s, fin %d / %.2f s )\n",
                 bc.diag + nw.st.nb_diag, bc.temps + t_paq + t_prol + t_fin, bc.diag, bc.temps, t_paq + t_prol,
                 nw.st.nb_diag, t_fin );
    return 0;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    a.n = 100000;
    a.dims = 2;
    Opts o;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if ( a.parse( s, i, argc, argv ) ) continue;
        else if ( s == "--sigma" )      o.sigma = std::atof( val() );
        else if ( s == "--nb-gauss" )   o.nb_gauss = std::atoi( val() );
        else if ( s == "--gauss" )      o.gauss = val();
        else if ( s == "--plancher" )   o.plancher = std::atof( val() );
        else if ( s == "--diracs" )     o.diracs = val();
        else if ( s == "--germes" )     o.germes = val();
        else if ( s == "--conv" )       o.conv0 = std::atof( val() );
        else if ( s == "--conv-ratio" ) o.conv_ratio = std::atof( val() );
        else if ( s == "--conv-min" )   o.conv_min = std::atof( val() );
        else if ( s == "-R" )           o.R = std::atoi( val() );
        else if ( s == "--paq" )        o.paq = val();
        else if ( s == "--lloyd-it" )   o.lloyd_it = std::atoi( val() );
        else if ( s == "--prol" )       o.prol = val();
        else if ( s == "--mls-anneaux" ) o.mls_anneaux = std::atoi( val() );
        else if ( s == "--mls-largeur" ) o.mls_largeur = std::atof( val() );
        else if ( s == "--seuil" )      o.seuil = std::atof( val() );
        else if ( s == "--tolg" )       o.tolg = std::atof( val() );
        else if ( s == "--reference" )  o.reference = true;
        else if ( s == "--sans-fin" )   o.fin = false;
        else if ( s == "--balaye" )     { o.balaye = true; o.fin = false; }
        else if ( s == "--echelle" )    { o.echelle = true; o.fin = false; }
        else if ( s == "--sans-zero" )  o.sans_zero = true;
        else if ( s == "--reste" )      { o.reste = true; o.fin = false; }
        else if ( s == "--span" )       { o.span = std::atoi( val() ); o.fin = false; }
        else if ( s == "--span-garde" ) o.span_garde = std::atof( val() );
        else if ( s == "--span-grille" ) o.span_grille = std::atoi( val() );
        else if ( s == "--prol-lisse" ) o.prol_lisse = std::atoi( val() );
        else if ( s == "--solver" )     o.solver = val();
        else if ( s == "--amg-var" )    o.amgvar = std::atoi( val() );
        else if ( s == "--newton-tol" ) o.newton.tol = std::atof( val() );
        else if ( s == "--newton-max" ) o.newton.maxit = std::atoi( val() );
        else if ( s == "--pas" )        o.newton.pas = std::string( val() ) == "essai-limites" ? NewtonOptions::ESSAI_LIMITES : NewtonOptions::ESSAIS;
        else if ( s == "--residu" ) {
            const std::string r = val();
            o.newton.residu = r == "log" ? NewtonOptions::LOG : r == "barriere" ? NewtonOptions::BARRIERE : NewtonOptions::LIN;
        }
        else if ( s == "--quiet" )      o.newton.trace = false;
        else {
            Args::usage();
            std::printf(
                "  --sigma S       l'echelle des largeurs du jeu de gaussiennes            (0.02)\n"
                "  --nb-gauss N --gauss SPEC --plancher F    la densite ( voir densite --help )\n"
                "  --diracs D      uniforme | rho ( germes tires selon la densite )   (uniforme)\n"
                "  --germes G      aleatoire | regulier : une grille a peine bruitee. `alpha*` est fixe par\n"
                "                  la paire de germes la PLUS SERREE, et sur un tirage aleatoire le plus petit\n"
                "                  ecart est `O( 1/n )` au lieu de `O( 1/sqrt n )` en 2D -- la pathologie du\n"
                "                  § 23.7, qui ecrase `alpha*` pour une raison etrangere au sujet (aleatoire)\n"
                "  --conv S0       la continuation : S0, S0/R, ... >= Smin, puis 0          (0.5)\n"
                "  --conv-ratio R                                                            (2)\n"
                "  --conv-min S                                                 (0 : sigma / 4)\n"
                "  -R K            germes par paquet                                        (16)\n"
                "  --paq P         bsp ( representant = germe le plus proche du centre )\n"
                "                  | bsp-bary ( meme partition, germe au barycentre pondere )\n"
                "                  | lloyd ( quantification optimale ponderee )            (bsp)\n"
                "  --lloyd-it K    iterations de Lloyd                                      (20)\n"
                "  --prol P        copie | harmonique | mls | ctransf                      (mls)\n"
                "  --prol-lisse M  passes de Jacobi amorti SUR la prolongation. Ce qui borne un pas le\n"
                "                  long d'une prolongation n'est pas son amplitude mais la difference\n"
                "                  seconde qu'elle porte a l'echelle de la cellule : `alpha* x saut = 2h^2`.\n"
                "                  Regle mesuree en 1D : LISSER JUSQU'A CE QUE `alpha*` DEPASSE 1. En 2D un\n"
                "                  agregat de R germes a un rayon de sqrt( R ), donc M ~ R et non R^2   (0)\n"
                "  --mls-anneaux K --mls-largeur F   le pochoir du MLS                     (2, 1)\n"
                "  --seuil F       sous le plancher : F * min( nu_i, aire min de Voronoi )  (0.5)\n"
                "  --tolg T        la tolerance de Newton au niveau GROSSIER   (0 : la meme)\n"
                "  --reference     la continuation sur le nuage COMPLET, seule ( le denominateur )\n"
                "  --sans-fin      s'arreter apres la prolongation ( le verdict, gratuit )\n"
                "  --balaye        juger la prolongation a CHAQUE etape en s : OU est le mur\n"
                "  --sans-zero     la liste s'arrete a --conv-min, PAS a s = 0 ( les zeros de la densite\n"
                "                  rendent le residu et le log indefinis sur la plupart des cellules )\n"
                "  --span-garde F  span : un point n'est admissible que si min a/nu >= F * ( celui de la\n"
                "                  base ) -- le blocage AVANT que les cellules se vident        (0.5)\n"
                "  --span-grille N points de la grille 1-D, par coordonnee et par balayage       (33)\n"
                "  --span K        minimiser le merite sur un span de 1..K directions, a DIAGRAMME et `L`\n"
                "                  GELES : w_prol d'abord, puis « ce qui manque » orthogonalise, K <= 4\n"
                "  --reste         LA METRIQUE : Newton depuis alpha_0 * w_prol, et combien d'iterations\n"
                "                  il reste -- `alpha_0 = 0` est Voronoi, donc le temoin est dans le tableau\n"
                "  --echelle       balayer alpha_0 dans `alpha_0 * w_prol` : JUSQU'OU la direction porte\n"
                "  --pas P         essais ( KMT ) | essai-limites                      (essais)\n"
                "  --residu R      lin | barriere | log                                   (lin)\n"
                "  --solver S      chol | amg   --amg-var V   --newton-tol T   --newton-max K\n"
                "  --quiet         pas de trace par iteration\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    a.finalise();
    if ( o.liste.empty() ) {
        const TF smin = o.conv_min > 0 ? o.conv_min : o.sigma / 4;
        if ( o.conv0 > 0 )
            for ( TF s = o.conv0; s >= smin * ( 1 - 1e-12 ); s /= o.conv_ratio ) o.liste.push_back( s );
        if ( ! o.sans_zero ) o.liste.push_back( 0 );
    }
    const Opts &oc = o;
    const Nuage<2> nu = o.diracs == "rho" ? nuage_selon( densite_jeu( o.sigma, o.nb_gauss, o.gauss, o.plancher ), a.n, a.graine )
                       : o.germes == "regulier" ? nuage_regulier( a.n, a.graine )
                                                : nuage_uniforme<2>( a.n, a.graine, 0 );
    return dispatch<2>( a, [ & ]( auto tag ) {
        using PD = typename decltype( tag )::type;
#ifdef SF_EIGEN
        if ( oc.solver == "chol" ) {
            Cholesky lin;
            return lance<PD>( a, oc, nu, lin );
        }
#endif
#ifdef SF_AMGCL
        Amg lin;
        lin.variante = oc.amgvar;
        return lance<PD>( a, oc, nu, lin );
#else
        std::printf( "  AMGCL absent : --solver chol\n" );
        return 1;
#endif
    } );
}
