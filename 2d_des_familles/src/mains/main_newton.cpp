// LE SOLVEUR : un probleme de transport semi-discret resolu de bout en bout, chronometre par
// poste. C'est le banc qui dit ce qu'un accelerateur vaut VRAIMENT -- une passe isolee ne mesure
// que la moitie de la question, puisqu'un index qui coute une passe et en fait gagner un quart se
// rembourse sur douze iterations et pas sur une.
//
//   xmake run pd_newton --help

#include "spatial_accel/AaBsp.h"
#include "spatial_accel/AaBspMemo.h"
#include "spatial_accel/FrontPd.h"
#include "solver/Newton.h"
#include "bench/Bench.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>
#include <string>

using namespace pd;
using namespace pd::bench;

namespace {

/// Les options PROPRES au solveur. Elles n'ont rien a faire dans `bench::Args` : aucun autre banc
/// ne les lit, et c'est tout l'interet d'avoir un `main` par sujet.
struct Opts {
    double ntol      = 1e-6;       ///< arret : `max_i |a_i - nu| <= ntol * nu`
    int    nmax      = 100;        ///< iterations de Newton au maximum
    double cgtol     = 1e-10;      ///< arret du gradient conjugue, en residu RELATIF
    int    cgmax     = 20000;
    std::string solver = "amg";    ///< amg (AMGCL) | chol (Eigen) | cg (maison)
    int    amgvar    = 0;          ///< 0 = SA+spai0, 1 = SA+Gauss-Seidel, 2 = Ruge-Stuben+GS
    SI     msratio   = 1;          ///< multi-echelle : germes par paquet d un niveau au suivant
                                   ///< 1 = AUCUN (defaut : la piste n aboutit pas, cf. README)
    SI     msmin     = 500;        ///< ... taille du niveau le plus grossier
    double mstol     = 1e-2;       ///< ... tolerance des niveaux grossiers
    int    mspasses  = 4;          ///< ... passes de rattrapage des cellules vides
    double msmarge   = 0;          ///< ... de combien on releve, en fraction de la cible
    bool   memo      = false;      ///< l arbre qui se souvient des coupes
    bool   memobits  = true;
    std::string tree = "bsp";      ///< bsp | front
    std::string ecrire;            ///< ou ecrire les poids trouves, au format de `cases/`
};


/// `suite_2d` et `suite_3d` ont des types de retour differents ; ces deux enveloppes permettent au
/// corps generique ci-dessous de n'en nommer qu'une.
template<int D> std::vector<Cloud<D>> suite_2d_ou_vide( const Args &a ) { (void) a; return {}; }
template<> inline std::vector<Cloud<2>> suite_2d_ou_vide<2>( const Args &a ) { return suite_2d( a ); }
template<int D> std::vector<Cloud<D>> suite_3d_ou_vide( const Args &a ) { (void) a; return {}; }
template<> inline std::vector<Cloud<3>> suite_3d_ou_vide<3>( const Args &a ) { return suite_3d( a ); }

/// Le nombre d iterations de CG, ou rien du tout quand on a factorise.
inline const char *cg_txt( int nb ) {
    static char buf[ 64 ];
    if ( ! nb ) return "";
    std::snprintf( buf, sizeof( buf ), " (%d iterations)", nb );
    return buf;
}

/// UN NIVEAU de la hierarchie multi-echelle : un nuage grossier, et la masse cible de chaque germe
/// grossier -- qui est celle de TOUT SON PAQUET.
template<int D>
struct Niveau {
    SI              R = 1, m = 0;
    std::vector<SI> cl;                 ///< place dans `order` -> paquet
    std::vector<SI> beg, end;           ///< la tranche de chaque paquet
    std::vector<TF> C[ D ], nu;         ///< le nuage grossier
    const TF *P[ D ] = {};
    void finish() { for ( int d = 0; d < D; ++d ) P[ d ] = C[ d ].data(); }
};

/// LES PAQUETS SONT DES NOEUDS DU BSP, pas des tranches a pas fixe de sa permutation.
///
/// ESSAYE ET REJETE : decouper `order` en blocs de `R` positions consecutives. C'etait tentant --
/// le BSP coupe a la MEDIANE, donc un sous-arbre EST une tranche contigue -- mais la reciproque est
/// fausse : les frontieres de sous-arbres tombent aux medianes (10000, 5000, 2500, 1250, 625,
/// 313...), qui ne sont pas des multiples de `R`. Un bloc sur deux enjambait donc deux sous-arbres,
/// parfois cousins eloignes, et la prolongation donnait LE MEME POIDS a deux germes a l'autre bout
/// du carre. Mesure : le niveau grossier convergeait en 3 iterations a pas plein, et le niveau
/// suivant demarrait avec des cellules VIDES et un residu de 67 fois la cible -- pire que depuis
/// `w = 0`.
///
/// Ici on descend l'arbre et on coupe des qu'un sous-arbre ne depasse plus `R` germes : les paquets
/// sont alors compacts par construction, et restent des tranches contigues de `order`.
template<int D>
Niveau<D> faire_niveau( const AaBspT<D> &tr, const TF *const *P, SI R ) {
    const SI n = SI( tr.order.size() );
    Niveau<D> L;
    L.R = R;
    L.cl.assign( n, -1 );

    std::vector<SI> pile{ 0 };
    while ( ! pile.empty() ) {
        const SI h = pile.back();
        pile.pop_back();
        const typename AaBspT<D>::Node &nd = tr.nodes[ h ];
        if ( nd.end - nd.beg <= R || nd.right < 0 ) {
            for ( SI k = nd.beg; k < nd.end; ++k )
                L.cl[ k ] = L.m;
            L.beg.push_back( nd.beg );
            L.end.push_back( nd.end );
            ++L.m;
        } else {
            pile.push_back( h + 1 );                    // PREORDRE : le fils gauche est juste a cote
            pile.push_back( nd.right );
        }
    }

    // le representant est le germe le plus proche du barycentre du paquet -- pas le premier venu :
    // la prolongation donne son poids a tout le paquet, donc il doit en etre au centre.
    for ( int d = 0; d < D; ++d ) L.C[ d ].resize( L.m );
    L.nu.resize( L.m );
    for ( SI c = 0; c < L.m; ++c ) {
        const SI b = L.beg[ c ], e = L.end[ c ];
        Vec<D> g = {};
        for ( SI k = b; k < e; ++k )
            for ( int d = 0; d < D; ++d ) g[ d ] += P[ d ][ tr.order[ k ] ];
        for ( int d = 0; d < D; ++d ) g[ d ] /= ( e - b );
        SI best = b;
        TF bd = -1;
        for ( SI k = b; k < e; ++k ) {
            TF dd = 0;
            for ( int d = 0; d < D; ++d ) {
                const TF u = P[ d ][ tr.order[ k ] ] - g[ d ];
                dd += u * u;
            }
            if ( bd < 0 || dd < bd ) { bd = dd; best = k; }
        }
        for ( int d = 0; d < D; ++d ) L.C[ d ][ c ] = P[ d ][ tr.order[ best ] ];
        L.nu[ c ] = TF( e - b ) / n;                    // la masse AGREGEE du paquet
    }
    L.finish();
    return L;
}

/// LA BASELINE : un probleme de transport resolu de bout en bout, chronometre par poste.
///
/// Ce qu'on veut en lire n'est pas le temps total mais sa REPARTITION. Un accelerateur qui doit
/// survivre d'une iteration a l'autre ne se juge que la : s'il faut douze diagrammes pour
/// converger, un index qui coute une passe et en fait gagner un quart se rembourse -- alors qu'il
/// est une perte seche sur une passe isolee.
///
/// = LE MULTI-ECHELLE, et ce qu'il repare
///
/// Depuis `w = 0` sur un nuage tres non uniforme, l'amortissement rampe : les cellules de Voronoi
/// s'etalent sur six ordres de grandeur, la direction de Newton est enorme devant la region ou
/// toutes les cellules restent non vides, et le pas tombe a 1/256. Mesure sur le nuage de lignes :
/// SEIZE iterations a ramper avant six iterations quadratiques.
///
/// Le remede n'est pas de mieux amortir, c'est de PARTIR D'AILLEURS. On resout d'abord le meme
/// probleme sur `n / R` representants portant la masse agregee de leur paquet -- moins cher, et
/// surtout bien mieux conditionne, puisque les masses cibles y sont comparables par construction.
/// Puis on PROLONGE : chaque germe fin recoit le poids de son representant.
///
/// Pourquoi cette prolongation est la bonne : tous les germes d'un paquet portant alors LE MEME
/// poids, la bissectrice de puissance entre deux d'entre eux est la mediatrice ordinaire. Le
/// diagramme fin est donc, a l'interieur d'un paquet, un VORONOI -- et globalement la solution
/// grossiere. L'erreur restante est purement LOCALE, et c'est exactement le regime ou Newton prend
/// des pas pleins.
template<class Cell, bool BOX, bool IN, class Tree, int D>
int newton_go( const Args &a, const Opts &o, const Cloud<D> &cl, const TF *Wref ) {
    const SI n = cl.n;
    const TF *const *P = cl.P;

    // l'arbre du niveau FIN est bati une fois pour toutes, sur des poids nuls, et ne sera plus que
    // rafraichi : les positions ne bougent pas d'une iteration de Newton a l'autre. Sa permutation
    // sert AUSSI a decouper les paquets de tous les niveaux grossiers.
    Tree tr;
    std::vector<TF> zero( n, TF( 0 ) );
    const double tb = now();
    tr.build( P, zero.data(), n, a.leaf );
    if constexpr ( requires ( Tree &t ) { t.bits; } ) { tr.bits = o.memobits; }
    const double t_avant = now() - tb;              // hors de `tot`, donc a rajouter a la fin
    double t_arbre = t_avant;

#ifdef _OPENMP
    // AMGCL est parallelise en OpenMP, le diagramme en `std::thread` : sans ca les deux moities du
    // chronometre ne tourneraient pas sur le meme nombre de coeurs.
    omp_set_num_threads( a.threads );
#endif
    Newton nw;
    nw.quel = o.solver == "chol" ? 1 : ( o.solver == "cg" ? 2 : 0 );
    nw.variante = o.amgvar;

    const double t0 = now();

    // ---- les niveaux, du plus fin au plus grossier.
    //
    // ESSAYE ET REJETE : decouper les paquets dans l'arbre du DIAGRAMME. Ses feuilles portent
    // `--leaf` germes (dix), donc aucun paquet ne peut etre plus petit que dix : quel que soit
    // `--ms-ratio`, la DERNIERE prolongation etait toujours un saut d'un facteur dix. Mesure : les
    // niveaux grossiers convergeaient tous en quatre a cinq iterations a pas plein, et le niveau
    // fin stagnait des sa premiere iteration. On batit donc un second arbre, a feuilles de
    // `--ms-ratio` germes, qui ne sert qu'a decouper -- et qu'on relache aussitot les niveaux
    // construits, parce qu'il pese deux fois plus de noeuds que celui du diagramme.
    std::vector<Niveau<D>> niv;
    if ( o.msratio > 1 ) {
        AaBspT<D> hier;
        const double th0 = now();
        hier.build( P, zero.data(), n, o.msratio );
        t_arbre += now() - th0;
        for ( SI R = o.msratio; ; R *= o.msratio ) {
            Niveau<D> L = faire_niveau<D>( hier, P, R );
            if ( L.m < o.msmin )
                break;
            if ( ! niv.empty() && L.m == niv.back().m )
                continue;
            niv.push_back( std::move( L ) );
        }
    }

    // LA PROLONGATION, par c-transformee : `w_i = -psi_grossier( p_i )`. Voir `psi_min`.
    std::vector<TF> wp;                                 // la solution du niveau precedent
    AaBspT<D> tsup;                                     // et son arbre, pour l'interroger
    auto prolonge = [ & ]( AaBspT<D> &tc, const std::vector<TF> &wc, const TF *const *Pf,
                           SI mf, std::vector<TF> &wf ) {
        refresh_weights( tc, wc.data(), a.threads, a.split, a.pin );
        wf.assign( mf, TF( 0 ) );
        parallel_for( mf, a.threads, a.split, a.pin, [ & ]( SI i, int ) {
            Vec<D> q;
            for ( int d = 0; d < D; ++d ) q[ d ] = Pf[ d ][ i ];
            wf[ i ] = -psi_min<D>( tc, q );
        } );
    };

    for ( int li = int( niv.size() ) - 1; li >= 0; --li ) {
        const Niveau<D> &L = niv[ li ];
        std::vector<TF> w0( L.m, TF( 0 ) );
        if ( li + 1 < int( niv.size() ) )
            prolonge( tsup, wp, L.P, L.m, w0 );

        AaBspT<D> tl;
        const double tl0 = now();
        tl.build( L.P, w0.data(), L.m, a.leaf );
        t_arbre += now() - tl0;
        PowerDiagram<Cell, AaBspT<D>, BOX, IN, false, true> pl{ tl };
        if ( li + 1 < int( niv.size() ) )
            rattrape_vides<D>( pl, tl, L.P, L.m, w0, TF( o.msmarge ) / L.m, o.mspasses,
                               a.threads, a.split, a.pin, true );
        nw.n = L.m;
        nw.nu = L.nu;
        std::printf( "    -- niveau R=%d, %d germes\n", int( L.R ), int( L.m ) );
        nw.resout<Cell>( pl, tl, L.P, w0, TF( o.mstol ), o.nmax,
                         TF( o.cgtol ), o.cgmax, a.threads, a.split, a.pin, true );
        wp = nw.w;
        tsup = std::move( tl );
    }

    // ---- le niveau FIN : chaque germe recoit le poids du paquet auquel il appartient.
    std::vector<TF> w0( n, TF( 0 ) );
    if ( ! niv.empty() ) {
        prolonge( tsup, wp, P, n, w0 );
    }

    PowerDiagram<Cell, Tree, BOX, IN, false, true> pd{ tr };
    if ( ! niv.empty() )
        rattrape_vides<D>( pd, tr, P, n, w0, TF( o.msmarge ) / n, o.mspasses, a.threads,
                           a.split, a.pin, true );
    nw.n = n;
    nw.nu.assign( n, TF( 1 ) / n );
    if ( ! niv.empty() )
        std::printf( "    -- niveau R=1, %d germes\n", int( n ) );
    const bool ok = nw.resout<Cell>( pd, tr, P, w0, TF( o.ntol ), o.nmax,
                                     TF( o.cgtol ), o.cgmax, a.threads, a.split, a.pin, true );
    const double tot = now() - t0;

    const double total = tot + t_avant;
    const double autre = total - t_arbre - nw.t_diag - nw.t_maj - nw.t_syst - nw.t_cg;
    std::printf( "  newton %s (max|a-nu|/nu = %.2e) : %dD n=%d threads=%d nv=%d box=%d leaf=%d %s %d iterations,"
                 " %d diagrammes (%d reculs), solveur %s%s\n",
                 nw.fin, double( nw.reste ), D, int( n ), a.threads, Cell::max_nb_vertices, int( BOX ),
                 int( a.leaf ), Tree::name, nw.nb_iter, nw.nb_diag, nw.nb_recul,
                 nw.quel == 0 ? "AMGCL" : ( nw.quel == 1 ? "Cholesky creux" : "gradient conjugue" ),
                 cg_txt( nw.nb_cg ) );
    std::printf( "         arbres %.3f | diagrammes %.3f | majorants %.3f | assemblage %.3f"
                 " | resolution %.3f | reste %.3f | TOTAL %.3f s\n",
                 t_arbre, nw.t_diag, nw.t_maj, nw.t_syst, nw.t_cg, autre, total );
    std::printf( "         soit %.0f %% de diagramme, %.3f s par diagramme, %.1f us/germe en tout\n",
                 100 * nw.t_diag / total, nw.t_diag / std::max( nw.nb_diag, 1 ), 1e6 * total / n );
    std::printf( "         resolution en detail : mise en forme %.3f | hierarchie/analyse %.3f"
                 " | resolution %.3f | descente %.3f\n",
                 nw.t_tri, nw.t_ana, nw.t_fac, nw.t_sol );
    if ( nw.nb_ana )
        std::printf( "         (%d montages pour %d iterations, pire residu lineaire %.2e)\n",
                     nw.nb_ana, nw.nb_iter, double( nw.pire_lin ) );

    if constexpr ( requires ( const Tree &t ) { t.t_gros; } )
        std::printf( "         index du front, %d preparations : arbres %.3f | diagramme grossier"
                     " %.3f | fronts %.3f | listes %.3f\n",
                     tr.nb_prep, tr.t_arbres, tr.t_gros, tr.t_front, tr.t_listes );
    if constexpr ( requires ( const Tree &t ) { t.nb_reuse; } )
        std::printf( "         bouclier : %d reconstructions pour %d evaluations, %d reutilisations\n",
                     tr.nb_prep, tr.nb_prep + tr.nb_reuse, tr.nb_reuse );

    if constexpr ( requires ( const Tree &t ) { t.nb_rejoue; } )
        std::printf( "         memo : %.2f coupes rejouees dans %.2f feuilles, et %.1f boites"
                     " testees par cellule (compteurs NON atomiques : a lire a --threads 1)\n",
                     double( tr.nb_rejoue ) / std::max( tr.nb_cell, 1LL ),
                     double( tr.nb_feuilles ) / std::max( tr.nb_cell, 1LL ),
                     double( tr.nb_boites ) / std::max( tr.nb_cell, 1LL ) );

    const SI novf = pd.nb_overflow.load();
    if ( novf )
        std::printf( "  ATTENTION : %d coupes ont DEBORDE %d sommets pendant la resolution.\n",
                     int( novf ), Cell::max_nb_vertices );

    // LES POIDS SUR DISQUE, quand on demande a fabriquer un cas de test. Le format est celui de
    // `cases/` : `n` puis `x [y z] w`, en `%.17g` -- la representation la plus courte qui relit
    // exactement le meme double, ce qui compte pour des poids issus d'une optimisation.
    if ( ! o.ecrire.empty() ) {
        std::FILE *f = std::fopen( o.ecrire.c_str(), "w" );
        if ( ! f ) {
            std::printf( "  impossible d'ecrire '%s'\n", o.ecrire.c_str() );
        } else {
            std::fprintf( f, "# nuage %dD a masses egales, poids obtenus par pd_newton (%s,"
                             " max|a-nu|/nu = %.3e)\n", D, nw.fin, double( nw.reste ) );
            std::fprintf( f, "# ATTENTION : produits par le banc lui-meme, donc PAS un temoin\n"
                             "# independant pour Newton -- seulement un cas de chronometrage.\n" );
            std::fprintf( f, "%d\n", int( n ) );
            for ( SI i = 0; i < n; ++i ) {
                for ( int d = 0; d < D; ++d )
                    std::fprintf( f, "%.17g ", double( P[ d ][ i ] ) );
                std::fprintf( f, "%.17g\n", double( nw.w[ i ] ) );
            }
            std::fclose( f );
            std::printf( "  poids ecrits dans '%s'\n", o.ecrire.c_str() );
        }
    }

    // La verification qui ne coute rien : le nuage `_equal` PORTE deja la solution, obtenue par
    // L-BFGS dans `gen_cases.py`. Elle n'est definie qu'a une constante pres, donc on recale sur
    // le germe 0 -- exactement la jauge que Newton impose.
    if ( Wref ) {
        TF m = 0, ampl = 0;
        for ( SI i = 0; i < n; ++i ) {
            m = std::max( m, std::fabs( ( nw.w[ i ] - nw.w[ 0 ] ) - ( Wref[ i ] - Wref[ 0 ] ) ) );
            ampl = std::max( ampl, std::fabs( Wref[ i ] - Wref[ 0 ] ) );
        }
        std::printf( "  contre les poids du fichier : ecart max %.3e sur une amplitude %.3e\n",
                     double( m ), double( ampl ) );
    }
    return ok && novf == 0 ? 0 : 1;
}

/// La suite du solveur : le nuage uniforme, puis les deux nuages durs de `cases/`. Un `--newton`
/// sur l'uniforme converge en dix iterations ; c'est sur le nuage a masses egales que le nombre
/// d'iterations, et donc la question de l'index, se pose vraiment.
template<int D>
int deroule_newton( const Args &a, const Opts &o ) {
    int bad = 0;
    for ( const Cloud<D> &cl : ( D == 2 ? suite_2d_ou_vide<D>( a ) : suite_3d_ou_vide<D>( a ) ) ) {
        if ( cl.absent ) {
            std::printf( "  %-28s : ABSENT (lancer cases/gen_cases.py)\n", cl.nom.c_str() );
            continue;
        }
        std::printf( "-- %s\n", cl.nom.c_str() );
        // les poids du fichier ne servent PAS de depart -- on part de zero -- mais de TEMOIN quand
        // ils resolvent deja le probleme de masses egales.
        const TF *ref = cl.nom.find( "egal" ) != std::string::npos ? cl.W : nullptr;
        const int nv = a.nv( D, cl.nv );
        auto go_n = [ & ]( auto cell_tag ) {
            using Cell = decltype( cell_tag );
            auto avec = [ & ]( auto tree_tag ) {
                using Tree = decltype( tree_tag );
                if ( a.cellbox && ! a.skipin )   return newton_go<Cell, true,  false, Tree, D>( a, o, cl, ref );
                if ( a.cellbox &&   a.skipin )   return newton_go<Cell, true,  true,  Tree, D>( a, o, cl, ref );
                if ( ! a.cellbox && ! a.skipin ) return newton_go<Cell, false, false, Tree, D>( a, o, cl, ref );
                return newton_go<Cell, false, true, Tree, D>( a, o, cl, ref );
            };
            if constexpr ( D == 2 ) {
                if ( o.tree == "front" ) return avec( FrontPd{} );
                if ( o.memo )            return avec( AaBspMemo{} );
            }
            return avec( AaBspT<D>{} );
        };
        if constexpr ( D == 2 ) {
            switch ( nv ) {
                case 16: bad += go_n( CellFor<2, 16>{} ); break;
                case 48: bad += go_n( CellFor<2, 48>{} ); break;
                case 64: bad += go_n( CellFor<2, 64>{} ); break;
                default: bad += go_n( CellFor<2, 32>{} ); break;
            }
        } else {
            switch ( nv ) {
                case 96:  bad += go_n( CellFor<3, 96>{} ); break;
                case 128: bad += go_n( CellFor<3, 128>{} ); break;
                default:  bad += go_n( CellFor<3, 64>{} ); break;
            }
        }
    }
    return bad;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    Opts o;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if ( parse_commun( a, s, i, argc, argv ) ) continue;
        else if ( s == "--newton-tol" ) o.ntol = std::atof( val() );
        else if ( s == "--newton-max" ) o.nmax = std::atoi( val() );
        else if ( s == "--cg-tol" )     o.cgtol = std::atof( val() );
        else if ( s == "--cg-max" )     o.cgmax = std::atoi( val() );
        else if ( s == "--solver" )     o.solver = val();
        else if ( s == "--amg-var" )    o.amgvar = std::atoi( val() );
        else if ( s == "--ms-ratio" )   o.msratio = std::atoi( val() );
        else if ( s == "--ms-min" )     o.msmin = std::atoi( val() );
        else if ( s == "--ms-tol" )     o.mstol = std::atof( val() );
        else if ( s == "--ms-passes" )  o.mspasses = std::atoi( val() );
        else if ( s == "--ms-marge" )   o.msmarge = std::atof( val() );
        else if ( s == "--memo" )       o.memo = true;
        else if ( s == "--no-memo-bits" ) o.memobits = false;
        else if ( s == "--tree" )       o.tree = val();
        else if ( s == "--ecrire" )     o.ecrire = val();
        else if ( s == "--front-rate" ) front_rate = std::atoi( val() );
        else if ( s == "--front-bouclier" ) front_bouclier = true;
        else {
            std::printf( "usage: pd_newton [options]\n" );
            usage_commun();
            std::printf(
                "  --tree T        bsp (defaut) | front   -- 2D seulement pour « front »\n"
                "  --newton-tol T  arret sur max|a_i - nu| / nu           (1e-6)\n"
                "  --newton-max K  iterations au maximum                  (100)\n"
                "  --cg-tol T      arret du gradient conjugue, relatif    (1e-10)\n"
                "  --solver S      amg (AMGCL, defaut) | chol (Eigen) | cg (maison)\n"
                "  --amg-var V     0 = agregation+spai0 | 1 = agregation+GS | 2 = Ruge-Stuben+GS\n"
                "  --ms-ratio R    MULTI-ECHELLE : rapport entre deux niveaux, 1 = aucun (1)\n"
                "  --ms-min M      ... taille du niveau le plus grossier          (500)\n"
                "  --ms-tol T      ... tolerance des niveaux grossiers            (1e-2)\n"
                "  --memo          l arbre qui garde les coupes d une iteration a l autre\n"
                "  --front-rate R  « front » : un germe grossier sur R\n"
                "  --ecrire FILE   ecrire les poids trouves au format de cases/ -- c est\n"
                "                  ainsi qu on fabrique un cas « masses egales » en 3D,\n"
                "                  faute de pysdot sur cette machine (cf. gen_cases_3d.py)\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    finalise( a );
    front_threads = a.threads;

    int bad = 0;
    if ( a.dims != 3 ) { std::printf( "=== 2D\n" ); bad += deroule_newton<2>( a, o ); }
    if ( a.dims != 2 ) { std::printf( "=== 3D\n" ); bad += deroule_newton<3>( a, o ); }
    return bad ? 1 : 0;
}
