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
#include "solver/Agglo.h"
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
    TF          bruit = 0.05;            ///< `--germes regulier` : amplitude du bruit, en fraction de `h`
    TF          agglo = 0;               ///< > 0 : agglomerer les germes a moins de `agglo * h` ( § 23 )
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
    // ZERO PAR DEFAUT, ET C'EST UNE CORRECTION : a `0.5` cette garde exigeait `min a/nu >= 0.5 x` celui
    // de la base, et elle REFUSAIT des points parfaitement sains -- `alpha_0 = 0.96` y tombait a 0.35 par
    // recul, 84 iterations au lieu de 13. Le diagramme reel n'y a aucune cellule vide ; c'etait ma garde
    // qui mentait, pas la realite. Seul `min a/nu > 0` est exige.
    TF          span_garde = 0;
    TF          pave_tol = 1e-3;         ///< continuation : on s'arrete quand |somme A_i - 1| depasse ca
    std::string rho_gel = "interp";       ///< interp | plateau | cellule | germe ( cf. --help )
    bool        fige = false;            ///< `--fige` : jusqu'ou la CONNECTIVITE FIXE emmene, le long de w_prol
    int         poly = 0;                ///< `--poly K` : la recherche SUR LES POLYNOMES, puis un recul scalaire
    TF          pas0 = 0.05;             ///< continuation en `alpha_0` : le pas MAXIMAL
    int         alpha0 = 0;              ///< `--alpha0 K` : continuation en `alpha_0`, compagnes = increments de lissage
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
int lance( const Args &a, const Opts &o, const Nuage<2> &nu0, Lineaire &lin, const std::vector<TF> &part ) {
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
    // LES FRACTIONS DE CIBLE PAR GERME. Uniformes sans agglomeration ; apres `--agglo` un germe qui
    // represente une grappe de `k` membres en porte `k` fois plus -- et `nw.nu` etant un vecteur, ca ne
    // coute rien a la formulation ( § 23.10 ).
    const std::vector<TF> part_fin = part.empty() ? std::vector<TF>( n, TF( 1 ) / n ) : part;

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
        nw.res_cur = o.newton.residu;        // sinon le merite reste `lin` ( `res_cur` n'est pose que dans `resout` )
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
        nw.nu.resize( n );
        for ( SI i = 0; i < n; ++i ) nw.nu[ i ] = part_fin[ i ] * M;
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
        // ============ JUSQU'OU LA CONNECTIVITE FIXE EMMENE ( `--fige` ) ============
        if ( o.fige ) {
            Voisinage vois;
            vois.row = Lvor.row.data();
            vois.col = Lvor.col.data();                  // le voisinage de Voronoi = celui de la base
            std::vector<PolyMulti> pm1;
            const TF *dp1[ 1 ] = { D[ 0 ].data() };
            pd.set_weights( wb.data(), a.par );
            polynomes_multi( pd, nu0.P, wb, dp1, 1, a.par, pm1 );
            std::printf( "  JUSQU'OU LA CONNECTIVITE FIXE EMMENE, le long de w_prol\n" );
            std::printf( "   alpha  | min A poly |  min A fige | vides fige | min a/nu vrai | vides vrais |"
                         " ecart masse fige/vraie\n" );
            std::vector<TF> af, am_f, av;
            std::vector<Facette> ff;
            for ( int j = 0; j <= 25; ++j ) {
                const TF al = TF( 1.25 ) * TF( j ) / TF( 25 );
                // 1. LE POLYNOME
                TF mp = INFINI;
                for ( SI i = 0; i < n; ++i )
                    if ( pm1[ i ].etat == PolyCellule::OK ) mp = std::min( mp, pm1[ i ]( &al, 1 ) );
                // 2. LA CELLULE REFAITE, connectivite figee : l'aire, puis la masse
                const SI vf = mesures_connectivite_figee( pd, nu0.P, wb, D[ 0 ], al, vois, a.par,
                                                          []( const typename PD::Cell &c ) { return PD::mesure( c ); }, af );
                mesures_connectivite_figee( pd, nu0.P, wb, D[ 0 ], al, vois, a.par,
                                            [ & ]( const typename PD::Cell &c ) { return rho.mesure( c, []( int, TF ) {} ); }, am_f );
                TF mf = INFINI;
                for ( SI i = 0; i < n; ++i ) mf = std::min( mf, af[ i ] );
                // 3. LA VRAIE CELLULE
                std::vector<TF> wa( n );
                for ( SI i = 0; i < n; ++i ) wa[ i ] = wb[ i ] + al * D[ 0 ][ i ];
                nw.mesures_et_facettes( wa, av, ff );
                TF mv = INFINI, ec = 0;
                SI vv = 0;
                for ( SI i = 0; i < n; ++i ) {
                    mv = std::min( mv, av[ i ] / nw.nu[ i ] );
                    vv += ! ( av[ i ] > 0 );
                    ec = std::max( ec, std::fabs( am_f[ i ] - av[ i ] ) / std::max( av[ i ], TF( 1e-300 ) ) );
                }
                std::printf( "  %6.3f  | %10.3e | %11.3e | %10d | %13.3e | %11d | %.3e\n",
                             double( al ), double( mp ), double( mf ), int( vf ), double( mv ), int( vv ), double( ec ) );
            }
            return 0;
        }

        // ============ LA RECHERCHE SUR LES POLYNOMES, PUIS UN RECUL SCALAIRE ( `--poly K` ) ============
        //
        // LA CONCEPTION, et c'est elle qui rend le schema bon marche. Calculer les cellules coute cher,
        // donc la recherche ne doit pas en calculer : `polynomes_multi` construit UNE FOIS, a la base,
        // l'aire de chaque cellule comme POLYNOME des coefficients du span ( les sommets sont affines en
        // `t`, donc l'aire est quadratique -- § 22 ), et tout se cherche dessus, gratuitement. On ne paie
        // de vrais diagrammes que pour VERIFIER le `t` trouve.
        //
        // UNE AIRE POLYNOMIALE NEGATIVE EST L'ARTEFACT A GARDER, PAS UN BUG : a combinatoire figee le
        // polygone peut SE REPLIER, et la formule de l'aire signee rend alors du negatif. C'est le
        // critere d'inadmissibilite du modele, et il n'a rien a voir avec un changement de combinatoire.
        //
        // ET LE RECUL EST UN SCALAIRE : le `t` propose peut donner, EN REALITE, un diagramme a cellules
        // malades. On cherche alors le plus grand `beta` tel que `beta t` soit sain -- un seul scalaire
        // devant tout le vecteur, par bissection, donc une poignee de diagrammes -- pour avoir un depart
        // sain qui avance quand meme.
        if ( o.poly > 0 ) {
            // ---- LA DENSITE GELEE, qui rend le modele utilisable sur une densite. `PolyMulti` donne
            //      l'AIRE ; avec une densite l'objectif porte sur la MASSE, et la masse n'est pas un
            //      polynome ( § 9.7 ) -- mesure : `log2` sur les aires vaut 20 a la base quand le vrai
            //      merite vaut 6.9e4, donc le modele egaliserait les AIRES, pas les masses.
            //
            //      LE REMEDE : geler une densite CONSTANTE PAR MORCEAUX. Avec `rho_i` fixe par cellule
            //      fine, `masse_i( t ) = rho_i A_i( t )` redevient un POLYNOME, et comparer `rho_i A_i` a
            //      `nu_i` revient a comparer `A_i( t )` a une CIBLE D'AIRE `a^_i = nu_i / rho_i`. On
            //      retombe donc exactement sur du Lebesgue pondere, cible par cible.
            //
            //      Et le `rho` naturel est celui que le solve grossier donne gratuitement : la cellule
            //      grossiere `r` porte la masse `nu_r` sur l'aire `|C_r|`, donc `rho_r = nu_r / |C_r|` et
            //      `a^_i = ( nu_i / nu_r ) |C_r|` -- la part d'aire de la cellule grossiere qui revient au
            //      germe fin.
            std::vector<TF> cible_aire( n ), rho_fige( n ), aire_base;
            {
                std::vector<TF> ab2;
                std::vector<Facette> fb2;
                nw.mesures_et_facettes( wb, ab2, fb2 );      // les MASSES a la base
                pd.set_weights( wb.data(), a.par );
                pd.measures( aire_base, a.par );             // les AIRES de Lebesgue a la base
                if ( o.rho_gel == "plateau" || o.rho_gel == "interp" ) {
                    // `rho_r = nu_r / |C_r|` : la masse cible de la cellule grossiere sur son volume de
                    // LEBESGUE, donc la densite moyenne qui y regne. C'est une propriete de la DENSITE.
                    std::vector<TF> ac;
                    pdc.set_weights( wc.data(), a.par );
                    pdc.measures( ac, a.par );
                    const TF M2 = rho.masse_carre();
                    std::vector<TF> lrc( nc );
                    for ( SI r = 0; r < nc; ++r )
                        lrc[ r ] = std::log( std::max( part_c[ r ] * M2 / std::max( ac[ r ], TF( 1e-300 ) ), TF( 1e-300 ) ) );
                    if ( o.rho_gel == "plateau" ) {
                        for ( SI i = 0; i < n; ++i ) rho_fige[ i ] = std::exp( lrc[ pq.paquet[ i ] ] );
                    } else {
                        // INTERPOLE aux germes fins, sur `log rho` ( positivite, et `rho` s'etale sur des
                        // ordres de grandeur ). Le MLS de degre 2 du § 8.2, qui est le meilleur
                        // interpolant du depot -- et ce qui disparait ainsi, ce sont les SAUTS du plateau
                        // aux bords d'agregats, exactement la pathologie qui lui coute un facteur 540.
                        Laplacien Lcv;
                        laplacien_de( pdc, Pc, nullptr, a.par, Lcv );   // le graphe de Voronoi grossier
                        std::vector<TF> lrf;
                        const SI retombes = prolonge_mls<2>( nu0.P, pq, Pc, Lcv, lrc, a.par, lrf,
                                                             o.mls_anneaux, o.mls_largeur );
                        for ( SI i = 0; i < n; ++i ) rho_fige[ i ] = std::exp( lrf[ i ] );
                        if ( retombes )
                            std::printf( "    ( interpolation de log rho : %d germes retombes sur le lineaire"
                                         " ou la copie )\n", int( retombes ) );
                    }
                } else if ( o.rho_gel == "germe" ) {
                    for ( SI i = 0; i < n; ++i ) rho_fige[ i ] = rho.rho( nu0.P[ 0 ][ i ], nu0.P[ 1 ][ i ] );
                } else {
                    // LA MOYENNE DE LA CELLULE FINE : `rho_i = a_i / A_i` a la base. Le modele
                    // `masse_i( t ) = rho_i A_i( t )` est alors EXACT en `t = 0`, et son erreur ne croit
                    // qu'avec le deplacement de la cellule -- contre un facteur 540 des le depart pour un
                    // plateau par cellule grossiere.
                    for ( SI i = 0; i < n; ++i )
                        rho_fige[ i ] = ab2[ i ] / std::max( aire_base[ i ], TF( 1e-300 ) );
                }
                TF ecart = 0, aire_tot = 0;
                for ( SI i = 0; i < n; ++i ) {
                    cible_aire[ i ] = nw.nu[ i ] / std::max( rho_fige[ i ], TF( 1e-300 ) );
                    aire_tot += cible_aire[ i ];
                    const TF m_pred = rho_fige[ i ] * aire_base[ i ];
                    ecart = std::max( ecart, std::fabs( m_pred - ab2[ i ] ) / std::max( ab2[ i ], TF( 1e-300 ) ) );
                }
                std::printf( "  DENSITE GELEE ( %s ) : somme des cibles d'aire %.6f ( le carre vaut 1 ),"
                             "  ecart max masse predite / vraie A LA BASE %.2e\n",
                             o.rho_gel.c_str(), double( aire_tot ), double( ecart ) );
            }
            std::printf( "  RECHERCHE SUR LES POLYNOMES ( un diagramme par k, le reste gratuit )\n" );
            std::printf( "   k | alpha_0 poly |  log2 poly  |  beta  | alpha_0 retenu |  min a/nu  | ecart mod |  IT RESTANTES  |  diag\n" );
            std::vector<PolyMulti> pm;
            std::vector<TF> tt;
            for ( int k = 1; k <= o.poly && k <= int( PolyMulti::KMAX ); ++k ) {
                // ---- UN diagramme : le modele du span courant
                std::vector<const TF *> dp( k );
                for ( int j = 0; j < k; ++j ) dp[ j ] = D[ j ].data();
                // `polynomes_multi` prend les CELLULES des poids que `pd` PORTE, et les coefficients
                // de `wb` : il faut donc remettre `wb` dans `pd`. Sans ca, le modele du tour `k` est
                // construit sur les cellules de la SOLUTION du tour precedent ( `nw.resout` laisse `pd`
                // dessus ) -- mesure : le meme `nk = 1` au meme `alpha` donnait +3.2e-02 au tour 1 et
                // -1.2e-01 au tour 2, ce que j'avais pris pour une dependance en `nk`.
                pd.set_weights( wb.data(), a.par );
                polynomes_multi( pd, nu0.P, wb, dp.data(), k, a.par, pm );
                SI hors = 0;
                for ( const PolyMulti &q : pm ) hors += q.etat != PolyCellule::OK;
                // CONTROLE : `A( alpha, 0, ..., 0 )` ne doit PAS dependre de `nk` ( tous les termes
                // supplementaires portent un facteur `t_k = 0` ). On le verifie au lieu de le supposer.
                {
                    const TF als[] = { TF( 0.001 ), TF( 0.002 ), TF( 0.003 ), TF( 0.004 ) };
                    std::printf( "  CONTROLE de l'independance en nk ( min A / cible, par cellule )\n" );
                    for ( TF al : als ) {
                        std::printf( "    alpha %7.4f :", double( al ) );
                        for ( int nk = 1; nk <= k; ++nk ) {
                                std::vector<const TF *> dpc( nk );
                                for ( int j = 0; j < nk; ++j ) dpc[ j ] = D[ j ].data();
                                std::vector<PolyMulti> pmc;
                                polynomes_multi( pd, nu0.P, wb, dpc.data(), nk, a.par, pmc );
                                std::vector<TF> tc( nk, TF( 0 ) );
                                tc[ 0 ] = al;
                                TF mn = INFINI;
                                SI arg = -1, nb_ok = 0;
                                for ( SI i = 0; i < n; ++i ) {
                                    if ( pmc[ i ].etat != PolyCellule::OK ) continue;
                                    ++nb_ok;
                                    const TF v = pmc[ i ]( tc.data(), nk ) / cible_aire[ i ];
                                    if ( v < mn ) { mn = v; arg = i; }
                                }
                                std::printf( "   nk=%d : %+.6e ( germe %d, %d cellules )", nk, double( mn ), int( arg ), int( nb_ok ) );
                        }
                        std::printf( "\n" );
                    }
                }


                // ---- l'objectif SUR LE MODELE, et son jacobien ( tout est analytique )
                // L'ECART DE PAVAGE, gratuit : `somme A_i( t ) - 1`
                auto pavage = [ & ]( const std::vector<TF> &t2 ) {
                    TF s2 = 0;
                    for ( SI i = 0; i < n; ++i )
                        if ( pm[ i ].etat == PolyCellule::OK ) s2 += pm[ i ]( t2.data(), k );
                    return s2 - 1;
                };
                auto modele = [ & ]( const std::vector<TF> &t2, TF &amin ) {
                    TF s2 = 0, moy = 0;
                    SI nb = 0;
                    amin = INFINI;
                    for ( SI i = 0; i < n; ++i ) {
                        const PolyMulti &q = pm[ i ];
                        if ( q.etat != PolyCellule::OK ) continue;
                        const TF A = q( t2.data(), k );
                        amin = std::min( amin, A / cible_aire[ i ] );
                        if ( ! ( A > 0 ) ) return INFINI;        // l'aire NEGATIVE : le polygone se replie
                        moy += std::log( A / cible_aire[ i ] );
                        ++nb;
                    }
                    if ( ! nb ) return INFINI;
                    moy /= TF( nb );
                    for ( SI i = 0; i < n; ++i ) {
                        const PolyMulti &q = pm[ i ];
                        if ( q.etat != PolyCellule::OK ) continue;
                        const TF e = std::log( q( t2.data(), k ) / cible_aire[ i ] ) - moy;
                        s2 += e * e;
                    }
                    return s2;
                };
                // GAUSS-NEWTON sur les compagnes seules, `t[ 0 ]` fixe ( `libre0 = false` ) ou sur tout
                auto gn_modele = [ & ]( std::vector<TF> &t2, bool libre0 ) {
                    const int d0 = libre0 ? 0 : 1, kc = k - d0;
                    if ( kc <= 0 ) return;
                    TF amin;
                    TF f = modele( t2, amin );
                    for ( int it = 0; it < 60 && std::isfinite( double( f ) ); ++it ) {
                        Eigen::MatrixXd A2( kc, kc );
                        Eigen::VectorXd b2( kc );
                        A2.setZero(); b2.setZero();
                        TF moy = 0;
                        SI nb = 0;
                        for ( SI i = 0; i < n; ++i ) {
                            if ( pm[ i ].etat != PolyCellule::OK ) continue;
                            moy += std::log( pm[ i ]( t2.data(), k ) / cible_aire[ i ] ); ++nb;
                        }
                        moy /= TF( nb );
                        TF gr[ PolyMulti::KMAX ];
                        for ( SI i = 0; i < n; ++i ) {
                            const PolyMulti &q = pm[ i ];
                            if ( q.etat != PolyCellule::OK ) continue;
                            const TF A = q( t2.data(), k );
                            q.gradient( t2.data(), k, gr );
                            const TF r = std::log( A / cible_aire[ i ] ) - moy;
                            for ( int j = 0; j < kc; ++j ) {
                                const TF Jj = gr[ j + d0 ] / A;
                                b2( j ) -= Jj * r;
                                for ( int l = 0; l <= j; ++l ) {
                                    const TF v = Jj * gr[ l + d0 ] / A;
                                    A2( j, l ) += v;
                                    if ( l != j ) A2( l, j ) += v;
                                }
                            }
                        }
                        const Eigen::VectorXd dd = A2.ldlt().solve( b2 );
                        const std::vector<TF> t0 = t2;
                        TF pas = 1, f2 = INFINI;
                        bool pris = false;
                        for ( int e = 0; e < 50; ++e, pas /= 2 ) {
                            for ( int j = 0; j < kc; ++j ) t2[ j + d0 ] = t0[ j + d0 ] + pas * TF( dd( j ) );
                            f2 = modele( t2, amin );
                            if ( f2 < f ) { pris = true; break; }
                        }
                        if ( ! pris ) { t2 = t0; break; }
                        const bool fini = std::fabs( f - f2 ) <= TF( 1e-12 ) * std::fabs( f );
                        f = f2;
                        if ( fini ) break;
                    }
                };

                // ---- LA CONTINUATION EN `t[ 0 ]`, sur le modele : gratuite, donc a pas fin
                tt.assign( k, TF( 0 ) );
                TF a0 = 0, amin_mod = 0;
                int nb_pas = 0;
                for ( TF cible = o.pas0; cible <= TF( 1 ) + 1e-12; cible += o.pas0 ) {
                    const std::vector<TF> garde = tt;
                    tt[ 0 ] = std::min( cible, TF( 1 ) );
                    const TF f_av = modele( tt, amin_mod );
                    gn_modele( tt, false );
                    const TF f_ap = modele( tt, amin_mod );
                    const TF pv = pavage( tt );
                    if ( ! std::isfinite( double( f_ap ) ) || std::fabs( pv ) > o.pave_tol ) {
                        std::printf( "       ARRET a cible %.5f ( pas %d ) : %s  ( modele %.4e,"
                                     " min A/a^ %.3e, ecart de pavage %+.3e )\n",
                                     double( cible ), nb_pas,
                                     std::isfinite( double( f_ap ) ) ? "LE MODELE N'EST PLUS VALIDE ( pavage )"
                                                                     : "aire polynomiale NEGATIVE",
                                     double( f_ap ), double( amin_mod ), double( pv ) );
                        tt = garde;
                        break;
                    }
                    (void) f_av;
                    a0 = tt[ 0 ];
                    ++nb_pas;
                }
                const TF l2_mod = modele( tt, amin_mod );

                // ---- UNE verification reelle, et le RECUL SCALAIRE si des cellules sont malades
                TF beta = 1, am = 0, ax = 0;
                std::vector<TF> coefs( k );
                auto sain = [ & ]( TF b3 ) {
                    for ( int j = 0; j < k; ++j ) coefs[ j ] = b3 * tt[ j ];
                    const TF m2 = evalue( coefs, am, ax );
                    return std::isfinite( double( m2 ) ) && am > 0;
                };
                if ( ! sain( 1 ) ) {
                    TF lo = 0, hi = 1;
                    for ( int it = 0; it < 20; ++it ) {
                        const TF mi = TF( 0.5 ) * ( lo + hi );
                        if ( sain( mi ) ) lo = mi; else hi = mi;
                    }
                    beta = lo;
                    sain( beta );
                }

                // ---- LE CONTROLE DU MODELE AU POINT RETENU : c'est la que sa validite se juge, pas a
                //      la base. `rho_i A_i( t )` contre la vraie masse, et l'ecart de pavage.
                TF ec_pt = 0;
                {
                    std::vector<TF> aa2;
                    pd.set_weights( wt.data(), a.par );
                    pd.measures( aa2, a.par );               // les AIRES reelles au point retenu
                    std::vector<TF> am2;
                    std::vector<Facette> fm2;
                    nw.mesures_et_facettes( wt, am2, fm2 );  // et les MASSES reelles
                    for ( SI i = 0; i < n; ++i ) {
                        const TF m_pred = rho_fige[ i ] * aa2[ i ];
                        ec_pt = std::max( ec_pt, std::fabs( m_pred - am2[ i ] ) / std::max( am2[ i ], TF( 1e-300 ) ) );
                    }
                }
                // ---- LA METRIQUE
                nw.st = NewtonStats{};
                nw.resout( wt );
                std::printf( "  %2d | %12.4f | %11.4e | %6.4f | %14.4f | %10.3e | %9.2e | %12d  | %5d%s\n",
                             k, double( a0 ), double( l2_mod ), double( beta ), double( beta * a0 ),
                             double( am ), double( ec_pt ), nw.st.nb_iter, nw.st.nb_diag,
                             hors ? ( "   ( " + std::to_string( hors ) + " cellules hors modele )" ).c_str() : "" );
                if ( k == o.poly || k == int( PolyMulti::KMAX ) ) break;
                std::vector<TF> ws = w;
                SI m = 1;
                for ( int j = 0; j <= k; ++j ) m *= 4;
                lisse_jacobi( Lvor, int( m ), ws );
                for ( SI i = 0; i < n; ++i ) ws[ i ] -= w[ i ];
                D.push_back( ws );
                t.resize( D.size(), TF( 0 ) );
            }
            return 0;
        }

        // ================= LA CONTINUATION EN `alpha_0` ( `--alpha0 K` ) =================
        // Les compagnes sont les INCREMENTS DE LISSAGE de la prolongation, aux echelles 4, 16, 64 ... :
        //   `d_j = lisse_jacobi( Lvor, 4^j, w_prol ) - w_prol`.
        // Mesure en 1D : prendre « ce qui manque » comme direction de Newton au point bloque echoue
        // structurellement -- la ou une cellule est a `1e-8 nu` les lignes de `J_ik = ( L d_k )_i / a_i`
        // valent `1e8`, le moindre carre est domine par elles et la tangente sort a `1e+07`. Les
        // increments de lissage sont bornes, bien conditionnes, et l'optimiseur trouve SEUL le profil.
        if ( o.alpha0 > 0 ) {
            // l'etat au point courant : `a`, `L`, et les colonnes du jacobien
            auto etat = [ & ]( const std::vector<TF> &coefs, std::vector<TF> &aa, Laplacien &Lc2,
                               std::vector<std::vector<TF>> &J, std::vector<TF> &r ) {
                for ( SI i = 0; i < n; ++i ) {
                    TF v = wb[ i ];
                    for ( size_t j = 0; j < coefs.size(); ++j ) v += coefs[ j ] * D[ j ][ i ];
                    wt[ i ] = v;
                }
                std::vector<Facette> ff;
                nw.mesures_et_facettes( wt, aa, ff );
                for ( SI i = 0; i < n; ++i ) if ( ! ( aa[ i ] > 0 ) ) return false;
                Lc2.assemble( n, ff );
                r.resize( n );
                TF moy = 0;
                for ( SI i = 0; i < n; ++i ) { r[ i ] = std::log( aa[ i ] / nw.nu[ i ] ); moy += r[ i ]; }
                moy /= TF( n );
                for ( SI i = 0; i < n; ++i ) r[ i ] -= moy;
                J.assign( coefs.size(), {} );
                std::vector<TF> col;
                for ( size_t j = 0; j < coefs.size(); ++j ) {
                    Lfois( Lc2, D[ j ], col );
                    J[ j ].resize( n );
                    for ( SI i = 0; i < n; ++i ) J[ j ][ i ] = col[ i ] / aa[ i ];
                }
                return true;
            };
            // GAUSS-NEWTON SUR LES COMPAGNES SEULES, `alpha_0` fixe
            auto corrige = [ & ]( TF a0, std::vector<TF> &al, TF &f ) {
                const int kc = int( al.size() );
                std::vector<TF> coefs( kc + 1 ), aa, r;
                Laplacien Lc2;
                std::vector<std::vector<TF>> J;
                coefs[ 0 ] = a0;
                for ( int j = 0; j < kc; ++j ) coefs[ j + 1 ] = al[ j ];
                TF amin, amax;
                f = evalue( coefs, amin, amax );
                if ( ! std::isfinite( double( f ) ) ) return false;
                if ( kc == 0 ) return true;
                for ( int it = 0; it < 40; ++it ) {
                    if ( ! etat( coefs, aa, Lc2, J, r ) ) return false;
                    Eigen::MatrixXd A( kc, kc );
                    Eigen::VectorXd b2( kc );
                    for ( int j = 0; j < kc; ++j ) {
                        TF g = 0;
                        for ( SI i = 0; i < n; ++i ) g += J[ j + 1 ][ i ] * r[ i ];
                        b2( j ) = -g;
                        for ( int l = 0; l <= j; ++l ) {
                            TF v = 0;
                            for ( SI i = 0; i < n; ++i ) v += J[ j + 1 ][ i ] * J[ l + 1 ][ i ];
                            A( j, l ) = A( l, j ) = v;
                        }
                    }
                    const Eigen::VectorXd dd = A.ldlt().solve( b2 );
                    const std::vector<TF> c0 = coefs;
                    TF pas = 1, f2 = INFINI;
                    bool pris = false;
                    for ( int e = 0; e < 40; ++e, pas /= 2 ) {
                        for ( int j = 0; j < kc; ++j ) coefs[ j + 1 ] = c0[ j + 1 ] + pas * TF( dd( j ) );
                        f2 = evalue( coefs, amin, amax );
                        if ( f2 < f ) { pris = true; break; }
                    }
                    if ( ! pris ) { coefs = c0; break; }
                    const bool fini = std::fabs( f - f2 ) <= TF( 1e-12 ) * std::fabs( f );
                    f = f2;
                    if ( fini ) break;
                }
                for ( int j = 0; j < kc; ++j ) al[ j ] = coefs[ j + 1 ];
                return true;
            };

            std::printf( "  CONTINUATION EN alpha_0 ( compagnes = increments de lissage, alpha_0 plafonne a 1 )\n" );
            std::printf( "   k |   alpha_0  |      merite^2  |  IT RESTANTES  |  diag  |  fin  | coefficients\n" );
            std::vector<TF> al;                          // les coefficients des compagnes
            TF a0 = 0, f = 0;
            {
                TF am0, ax0;
                f = evalue( std::vector<TF>{ TF( 0 ) }, am0, ax0 );   // le merite^2 a la base
            }
            for ( int k = 1; k <= o.alpha0; ++k ) {
                TF pas = o.pas0;
                while ( a0 < 1 && pas > TF( 1e-9 ) ) {
                    // LA TANGENTE de la variete des minimiseurs : `- ( Jc^T Jc )^-1 Jc^T j_0`
                    std::vector<TF> tang( al.size(), TF( 0 ) );
                    if ( ! al.empty() ) {
                        std::vector<TF> coefs( al.size() + 1 ), aa, r;
                        Laplacien Lc2;
                        std::vector<std::vector<TF>> J;
                        coefs[ 0 ] = a0;
                        for ( size_t j = 0; j < al.size(); ++j ) coefs[ j + 1 ] = al[ j ];
                        if ( etat( coefs, aa, Lc2, J, r ) ) {
                            const int kc = int( al.size() );
                            Eigen::MatrixXd A( kc, kc );
                            Eigen::VectorXd b2( kc );
                            for ( int j = 0; j < kc; ++j ) {
                                TF g = 0;
                                for ( SI i = 0; i < n; ++i ) g += J[ j + 1 ][ i ] * J[ 0 ][ i ];
                                b2( j ) = -g;
                                for ( int l = 0; l <= j; ++l ) {
                                    TF v = 0;
                                    for ( SI i = 0; i < n; ++i ) v += J[ j + 1 ][ i ] * J[ l + 1 ][ i ];
                                    A( j, l ) = A( l, j ) = v;
                                }
                            }
                            const Eigen::VectorXd x = A.ldlt().solve( b2 );
                            for ( int j = 0; j < kc; ++j ) tang[ j ] = TF( x( j ) );
                        }
                    }
                    const TF essai = std::min( a0 + pas, TF( 1 ) );
                    std::vector<TF> al2 = al;
                    for ( size_t j = 0; j < al2.size(); ++j ) al2[ j ] += ( essai - a0 ) * tang[ j ];
                    TF f2 = INFINI;
                    if ( corrige( essai, al2, f2 ) ) { a0 = essai; al = al2; f = f2; pas = std::min( o.pas0, pas * TF( 1.5 ) ); }
                    else pas /= 2;
                }
                // LE POLISSAGE FINAL, et il faut le faire : la continuation s'arrete des qu'elle touche
                // `alpha_0 = 1` sans jamais reminimiser `log2` LIBREMENT. Mesure : sans lui, 42
                // iterations ; le point lissee a la main ( qui est un point du meme span ) en donne 16.
                // On relache donc TOUS les coefficients, `alpha_0` compris, depuis le point atteint.
                t.assign( al.size() + 1, TF( 0 ) );
                t[ 0 ] = a0;
                for ( size_t j = 0; j < al.size(); ++j ) t[ j + 1 ] = al[ j ];
                int gn_fin = 0;
                TF ng_fin = 0;
                const TF f_poli = minimise_gn( 60, gn_fin, ng_fin );
                const std::vector<TF> coefs = t;
                TF am, ax;
                evalue( coefs, am, ax );
                if ( f_poli < f ) f = f_poli;
                nw.st = NewtonStats{};
                nw.resout( wt );
                std::printf( "  %2d | %10.4f | %14.6e | %12d  | %5d  | %s |", k, double( coefs[ 0 ] ), double( f ),
                             nw.st.nb_iter, nw.st.nb_diag, nw.st.fin );
                for ( size_t j = 1; j < coefs.size(); ++j ) std::printf( " %+.3f", double( coefs[ j ] ) );
                std::printf( "\n" );
                if ( a0 >= 1 || k == o.alpha0 ) break;
                // UNE COMPAGNE DE PLUS : l'increment de lissage a l'echelle suivante
                std::vector<TF> ws = w;
                SI m = 1;
                for ( int j = 0; j <= k; ++j ) m *= 4;
                lisse_jacobi( Lvor, int( m ), ws );
                for ( SI i = 0; i < n; ++i ) ws[ i ] -= w[ i ];
                D.push_back( ws );
                al.push_back( TF( 0 ) );
                t.resize( D.size(), TF( 0 ) );
            }
            return 0;
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
            pd.set_weights( wb.data(), a.par );
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
        std::printf( "    alpha_0  |  min a/nu ( famine )  |  max a/nu ( gavage )  |  cellules > 10 nu  |    PIRE  |  merite^2 ( %s )\n",
                     o.newton.residu == NewtonOptions::LOG ? "log" : o.newton.residu == NewtonOptions::BARRIERE ? "barriere" : "lin" );
        // UNE GRILLE FINE, de 0 a 1.25 : le profil doit se lire sans trou, sinon on confond un
        // minimum local avec un pas de grille trop grand.
        std::vector<TF> fs;
        for ( int j = 0; j <= 50; ++j ) fs.push_back( TF( 1.25 ) * TF( j ) / TF( 50 ) );
        for ( TF f : fs ) {
            for ( SI i = 0; i < n; ++i ) wa[ i ] = f * w[ i ];
            SI mauv, gaves; TF amin, amax, pire, mer;
            juge( wa, mauv, amin, amax, gaves, pire, mer );
            std::printf( "    %7.4f  |  %18.3e  |  %18.1f  |  %16d  |  %8.2f  |  %.6e\n",
                         double( f ), double( amin ), double( amax ), int( gaves ), double( pire ), double( mer * mer ) );
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
        else if ( s == "--agglo" )      o.agglo = std::atof( val() );
        else if ( s == "--bruit" )      o.bruit = std::atof( val() );
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
        else if ( s == "--alpha0" )     { o.alpha0 = std::atoi( val() ); o.span = std::max( o.span, 1 ); o.fin = false; }
        else if ( s == "--pas0" )       o.pas0 = std::atof( val() );
        else if ( s == "--poly" )       { o.poly = std::atoi( val() ); o.span = std::max( o.span, 1 ); o.fin = false; }
        else if ( s == "--rho-gel" )    o.rho_gel = val();
        else if ( s == "--fige" )       { o.fige = true; o.span = std::max( o.span, 1 ); o.fin = false; }
        else if ( s == "--pave-tol" )   o.pave_tol = std::atof( val() );
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
                "  --bruit F       `--germes regulier` : le bruit, en fraction de `h`. Interpole entre la\n"
                "                  grille ( 0 ) et un tirage ( ~0.5 ), donc mesure COMBIEN de regularite il\n"
                "                  faut pour que `alpha*` depasse 1                               (0.05)\n"
                "  --agglo C       AGGLOMERER les germes a moins de `C * h` les uns des autres avant tout le\n"
                "                  reste ( `Agglo.h`, § 23 : hachage de grille + union-find, sans diagramme ).\n"
                "                  UN germe par grappe, au barycentre pondere par `nu`, de cible `somme nu`.\n"
                "                  C'est le PREALABLE du multi-echelle : `alpha*` est proportionnel au plus\n"
                "                  petit ecart, donc une seule paire serree le plafonne pour tout le nuage  (0)\n"
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
                "  --pave-tol F    L'ECART DE PAVAGE qui arrete la continuation. Les cellules PAVENT le carre,\n"
                "                  donc `somme_i A_i( t ) = 1` exactement si le modele est juste ; tout ecart est\n"
                "                  la somme des recouvrements, c'est-a-dire son erreur L1 -- gratuit, sans un\n"
                "                  diagramme, et sans passer par les aretes ( une arete qui meurt ne change\n"
                "                  RIEN a l'aire, donc le rayon par arete flague ce qui ne coute pas )  (1e-3)\n"
                "  --rho-gel M     la DENSITE GELEE qui rend la masse polynomiale ( `masse_i( t ) = rho_i A_i( t )`\n"
                "                  est un polynome si `rho_i` est fixe, donc on compare `A_i( t )` a la CIBLE\n"
                "                  D'AIRE `nu_i / rho_i` et on retombe sur du Lebesgue pondere ) :\n"
                "                    interp   `nu_r / |C_r|` par cellule GROSSIERE, puis INTERPOLE ( MLS sur\n"
                "                             `log rho` ) aux germes fins. C'est le bon objet : une densite, pas\n"
                "                             une propriete d'une configuration de cellules            (defaut)\n"
                "                    plateau  le meme, mais constant par agregat -- les SAUTS aux bords\n"
                "                             d'agregats lui coutent un facteur 540 sur la masse predite\n"
                "                    cellule  `a_i / A_i` par cellule FINE a la base : exact en t = 0, mais ca\n"
                "                             attache `rho` a UNE configuration, et pres de `w_prol` beaucoup de\n"
                "                             cellules fines sont degenerees -- donc la quantite n'a plus de sens\n"
                "                    germe    la densite analytique au germe, `rho( p_i )`\n"
                "  --fige          JUSQU'OU LA CONNECTIVITE FIXE EMMENE. Le long de `w_prol`, on compare trois\n"
                "                  choses : le POLYNOME ( qui laisse l'aire devenir negative au lieu de prendre\n"
                "                  les parties positives -- le polygone se replie ), la cellule REFAITE\n"
                "                  EXACTEMENT mais en ne coupant QUE par les voisins connus ( aucune\n"
                "                  exploration, pas d'AaBsp ), et la VRAIE cellule. Ce qui separe l'erreur du\n"
                "                  modele de celle de la connectivite\n"
                "  --poly K        LA RECHERCHE SUR LES POLYNOMES ( § 22 ), et c'est la version bon marche :\n"
                "                  UN diagramme par `k` construit le modele, toute la recherche est ensuite\n"
                "                  GRATUITE ( `A_i( t )` en forme close ), et on ne paie de vrais diagrammes que\n"
                "                  pour VERIFIER le `t` propose -- avec un recul sur un SCALAIRE `beta` devant\n"
                "                  tout le vecteur si des cellules sont malades. Une aire polynomiale NEGATIVE\n"
                "                  est le polygone qui se replie a combinatoire fixe : c'est inadmissible\n"
                "  --pas0 F        continuation : le pas MAXIMAL en `alpha_0`. Il faut partir de 0 et avancer\n"
                "                  LISSEMENT : le profil de `log2` le long du rayon est un PLATEAU a rides de\n"
                "                  0.2 % entre 0 et 0.25 ( mesure, grille fine ), et une methode de descente\n"
                "                  s'y arrete sur une ride au lieu de traverser                      (0.05)\n"
                "  --alpha0 K      LA CONTINUATION EN `alpha_0`, jusqu'a K directions. Les compagnes sont les\n"
                "                  INCREMENTS DE LISSAGE de la prolongation, `lisse_jacobi( 4^j ) - w_prol` :\n"
                "                  c'est a elles de lisser `w_prol` la ou il pince, et l'optimiseur trouve seul\n"
                "                  le profil. `alpha_0` est plafonne a 1 -- c'est une CIBLE, pas un maximand :\n"
                "                  pousse au bord de l'admissible le point est pire que Voronoi ( § 1D )\n"
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
    const Nuage<2> nu_brut = o.diracs == "rho" ? nuage_selon( densite_jeu( o.sigma, o.nb_gauss, o.gauss, o.plancher ), a.n, a.graine )
                       : o.germes == "regulier" ? nuage_regulier( a.n, a.graine, o.bruit )
                                                : nuage_uniforme<2>( a.n, a.graine, 0 );

    // ---- L'AGGLOMERATION DU § 23, EN PREALABLE. `alpha*` le long d'une prolongation est
    //      PROPORTIONNEL au plus petit ecart entre germes voisins : une seule paire serree le plafonne
    //      pour tout le nuage, et sur un tirage aleatoire le plus petit ecart vaut `O( h^2 )` au lieu
    //      de `O( h )`. Mesure : `alpha*` plafonne a 1.5e-2 sur un tirage contre 0.99 sur une grille.
    //      La detection est celle du § 23.8 ( hachage de grille + union-find, sans diagramme ), la
    //      reduction celle du § 23.10 ( un germe par grappe, au barycentre pondere par `nu` ).
    Nuage<2> nu = nu_brut;
    std::vector<TF> part;
    if ( o.agglo > 0 ) {
        const SI n0 = nu_brut.n;
        const TF h = TF( 1 ) / std::sqrt( TF( n0 ) );
        std::vector<SI> rep, vers, taille;
        const SI perdus = grappes_proches<2>( nu_brut.P, n0, o.agglo * h, rep );
        std::vector<TF> nu0v( n0, TF( 1 ) / TF( n0 ) ), Q[ 2 ], nur;
        reduis<2>( nu_brut.P, nu0v, rep, Q, nur, vers, taille );
        const SI m = SI( nur.size() );
        nu = Nuage<2>{};
        nu.nom = nu_brut.nom + " agglomere";
        for ( int d = 0; d < 2; ++d ) nu.c[ d ] = Q[ d ];
        nu.w.assign( m, TF( 0 ) );
        nu.finish();
        part = nur;                                      // deja `somme nu_i` par grappe, donc les fractions
        SI tmax = 0;
        for ( SI r = 0; r < m; ++r ) tmax = std::max( tmax, taille[ r ] );
        std::printf( "  AGGLOMERATION ( § 23 ) : delta = %.3g h, %d germes -> %d grappes"
                     " ( %d disparus, taille max %d )\n",
                     double( o.agglo ), int( n0 ), int( m ), int( perdus ), int( tmax ) );
    }
    return dispatch<2>( a, [ & ]( auto tag ) {
        using PD = typename decltype( tag )::type;
#ifdef SF_EIGEN
        if ( oc.solver == "chol" ) {
            Cholesky lin;
            return lance<PD>( a, oc, nu, lin, part );
        }
#endif
#ifdef SF_AMGCL
        Amg lin;
        lin.variante = oc.amgvar;
        return lance<PD>( a, oc, nu, lin, part );
#else
        std::printf( "  AMGCL absent : --solver chol\n" );
        return 1;
#endif
    } );
}
