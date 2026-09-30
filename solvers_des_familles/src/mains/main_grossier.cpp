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

    // ---- 5. JUSQU'OU LA DIRECTION PORTE ( `--echelle` ). C'est le `t` de la correction `retrait`
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

    // ---- 6. LE NEWTON FIN, le seul chiffre qui tranche
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
                "  --conv S0       la continuation : S0, S0/R, ... >= Smin, puis 0          (0.5)\n"
                "  --conv-ratio R                                                            (2)\n"
                "  --conv-min S                                                 (0 : sigma / 4)\n"
                "  -R K            germes par paquet                                        (16)\n"
                "  --paq P         bsp ( representant = germe le plus proche du centre )\n"
                "                  | bsp-bary ( meme partition, germe au barycentre pondere )\n"
                "                  | lloyd ( quantification optimale ponderee )            (bsp)\n"
                "  --lloyd-it K    iterations de Lloyd                                      (20)\n"
                "  --prol P        copie | harmonique | mls | ctransf                      (mls)\n"
                "  --mls-anneaux K --mls-largeur F   le pochoir du MLS                     (2, 1)\n"
                "  --seuil F       sous le plancher : F * min( nu_i, aire min de Voronoi )  (0.5)\n"
                "  --tolg T        la tolerance de Newton au niveau GROSSIER   (0 : la meme)\n"
                "  --reference     la continuation sur le nuage COMPLET, seule ( le denominateur )\n"
                "  --sans-fin      s'arreter apres la prolongation ( le verdict, gratuit )\n"
                "  --balaye        juger la prolongation a CHAQUE etape en s : OU est le mur\n"
                "  --sans-zero     la liste s'arrete a --conv-min, PAS a s = 0 ( les zeros de la densite\n"
                "                  rendent le residu et le log indefinis sur la plupart des cellules )\n"
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
