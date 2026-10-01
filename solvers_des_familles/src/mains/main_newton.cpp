// =====================================================================================
// LE SOLVEUR : un probleme de transport semi-discret resolu de bout en bout, chronometre par poste,
// sur la suite ( uniforme, nuage dur / Voronoi, nuage dur / masses egales ), en 2D puis en 3D.
//
// Ce qu'on veut y lire n'est pas le temps total mais sa REPARTITION : sur l'uniforme l'algebre
// lineaire pese les deux tiers, sur les lignes c'est le diagramme, et les deux regimes n'appellent
// pas le meme travail ( `2d_des_familles/README.md`, § 1.4 ).
//
//   xmake run newton --help
//   xmake run newton --2d -n 100000 --solver amg --amg-var 2
//   xmake run newton --3d --kernel float
//   xmake run newton --methode lbfgs --bascule 0.5   L-BFGS depuis Voronoi, Newton des que toute
//                                                    cellule est a moins de 50 % de sa cible
//   xmake run newton --methode cg --c2 0.1 --courbe courbes.csv
// =====================================================================================

#include "bench/Dispatch.h"
#include "bench/Trames.h"
#include "solver/Agglo.h"
#include "solver/Lineaire.h"
#include "solver/Multichol.h"
#include "solver/Multigrille.h"
#include "solver/Newton.h"
#include "solver/PremierOrdre.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

using namespace sf;

namespace {

/// les options PROPRES au solveur.
struct Opts {
    NewtonOptions newton;
    // LE DEFAUT DEPEND DE LA DIMENSION, PARCE QUE LA MESURE LE DIT. `auto` = `mg` en 3D, `amg`
    // en 2D. Partie lineaire, nuage uniforme :
    //
    //      3D  n = 1e5   mg 0.83 s   amg 1.53 s   chol > 600 s ( plafond )
    //      3D  n = 5e5   mg 4.79     amg 6.73
    //      2D  n = 5e5   mg 6.43     amg 5.68
    //
    // En 3D le multigrille maison gagne largement, et Cholesky n'est plus une option du tout. En
    // 2D UNIFORME il perd de 6 % au mieux, et aucun reglage ne renverse : notre cycle est moins
    // cher par iteration ( 11.4 ms contre 15.6 ) mais il en faut deux fois plus.
    //
    // Attention, ce n'est pas « mg perd en 2D » : dans `main_image`, ou la continuation enchaine
    // des CENTAINES de systemes voisins, il gagne ( 90.4 s contre 93.0 a n = 1e5 ) -- parce que
    // le recyclage de sous-espace y trouve de quoi vivre, ce qu'une poignee d'iterations de
    // Newton ne donne pas. C'est le REGIME qui decide, pas la dimension seule.
    std::string   solver = "auto"; ///< auto ( mg en 3D, amg en 2D ) | mg | amg | chol | chsup | mchol
    double        mchol_rho = 0;   ///< MCHOL : le `rho` du motif ( 0 : 7 en 2D, 3 en 3D )
    int           cp_refaire = 0;  ///< CHPREC : factorisation numerique toutes les K resolutions ( 0 : une )
    int           cp_cgmax = 0;    ///< CHPREC : au-dela de ce compte de CG, on rafraichit ( 0 : le defaut )
    bool          cp_super = true; ///< CHPREC : supernodal plutot que simplicial
    bool          cp_comp = true;  ///< CHPREC : compensation diagonale des termes hors motif
    double        cp_seuil = -1;   ///< CHPREC : fraction hors motif au-dela de laquelle on re-analyse
    int           cp_sauts = 0;    ///< CHPREC : sauts du motif gele ( 0 : le defaut, 2 )
    int           mchol_ech = 0;   ///< MCHOL : l'echelle -- 0 uniforme par niveau ( le papier ), 1 locale
    // LES REGLAGES DU MULTIGRILLE MAISON. Les defauts viennent de `Multigrille.h`, ou ils ont ete
    // mesures EN 2D SUR UNE DENSITE IMAGE : `agreg 8` en particulier vaut ce que vaut son cas
    // d'usage, et il n'y a aucune raison qu'il tienne en 3D ou le graphe a deux fois plus de
    // voisins et ou le paquet naturel est 2x2x2. C'est ce que ce banc est la pour dire.
    int           mg_agreg = 0, mg_lisseur = -1, mg_recycle = -1, mg_nu = 0;
    double        mg_cheb = 0, mg_tronque = -1;
    int           mg_trace = 0;
    int           amgvar = Amg::SA_SPAI0;
    TF            lintol = 1e-10;
    int           linmax = 20000;
    std::string   ecrire;          ///< ou ecrire les poids trouves, au format de `cases/`
    std::string   dump;            ///< les trames de l'animation ( JSONL )
    std::string   methode = "newton"; ///< newton | lbfgs | cg ( `PremierOrdre.h` )
    PremierOrdreOptions po;
    std::string   courbe;          ///< CSV : le residu apres chaque pas, contre les diagrammes et le temps
    // LE PLANCHER DE BRUIT EST LE BON CRITERE, et les deux autres sont des filets. On a essaye :
    //   * LE PROGRES ( `mixte_progres` ) se trompe -- les premieres iterations gagnent
    //     legitimement ~50 % sur `|r|_2` ( 1.364e-3 -> 6.604e-4 -> 3.341e-4 en 3D ), donc un
    //     seuil a 0.5 coupe la phase `float` des la deuxieme ;
    //   * LE PAS ( `mixte_tmin` a 0.25 ) se trompe aussi, dans l'autre sens : la PREMIERE
    //     iteration demande legitimement un petit pas, et a `n = 2e4` en 2D elle rendait la main
    //     tout de suite, quatre diagrammes gaspilles.
    // Les deux sont REACTIFS : ils constatent apres coup. Le plancher, lui, se POSE d'avance.
    double        mixte_tmin = 1e-3;  ///< `--kernel mixte` : le pas sous lequel la phase `float` passe la main
    double        mixte_progres = 0;  ///< ... ou le gain minimal sur `|r|_2` par iteration ( 0 : eteint )
    double        mixte_kappa = 30;   ///< ... et le plancher de bruit vise : `kappa eps / sqrt( n )`
    double        agrege = 0;         ///< > 0 : resoudre EN DEUX ETAPES, germes agreges sous ce seuil ( `Agglo.h` )
    bool          agrege_corr = true;  ///< ... et REDECOUPER les cellules fusionnees ( troisieme etape, 2D )
    bool          agrege_fin = false;  ///< ... puis finir par un Newton sur le nuage COMPLET depuis ces poids
    bool          cout = false;       ///< calculer le COUT DE TRANSPORT a la fin ( 2D, `Agglo.h` )
};

/// LE SOLVEUR LINEAIRE demande par les options, construit UNE FOIS et hors du dispatch : depuis
/// qu'il est une interface ( § 18 ), son type ne depend plus de celui du diagramme. C'est ce qui
/// permet a la bascule `--kernel mixte` d'en garder UN SEUL a travers les deux phases -- donc de
/// garder aussi sa hierarchie et son sous-espace recycle.
template<int D>
std::unique_ptr<Lineaire> fabrique( const Opts &o ) {
    const std::string sol = o.solver == "auto" ? ( D == 3 ? "mg" : "amg" ) : o.solver;
#ifdef SF_EIGEN
    if ( sol == "chprec" ) {                             // la factorisation GELEE comme preconditionneur
        auto p = std::make_unique<CholPrec>();
        p->tol = o.lintol;
        p->maxit = o.linmax;
        p->refaire = o.cp_refaire;
        if ( o.cp_cgmax > 0 ) p->cg_max = o.cp_cgmax;
        p->super = o.cp_super;
        p->compense = o.cp_comp;
        if ( o.cp_seuil >= 0 ) p->seuil_motif = TF( o.cp_seuil );
        if ( o.cp_sauts > 0 ) p->sauts = o.cp_sauts;
        p->trace = o.mg_trace;
        return p;
    }
#endif
#ifdef SF_CHOLMOD
    if ( sol == "chsup" )                                // le supernodal ( § 24.18 )
        return std::make_unique<CholeskySuper>();
#endif
#ifdef SF_EIGEN
    if ( sol == "chol" )
        return std::make_unique<Cholesky>();
#endif
    if ( sol == "mchol" ) {                              // le Cholesky multi-echelle ( § 24.17 )
        auto p = std::make_unique<MultiChol>();
        p->tol = o.lintol;
        p->maxit = o.linmax;
        if ( o.mchol_rho > 0 ) p->rho = TF( o.mchol_rho );
        p->echelle = o.mchol_ech;
        p->trace = o.mg_trace;
        return p;
    }
    if ( sol == "mg" ) {
        auto p = std::make_unique<Mg>();
        p->tol = o.lintol;
        p->maxit = o.linmax;
        if ( o.mg_agreg > 0 ) p->agreg = o.mg_agreg;
        // UN SEUL LISSAGE EN 3D, ET C'EST LA MESURE QUI LE DIT. Les defauts de `Multigrille.h`
        // viennent du 2D sous une densite image, ou `nu` est PLAT entre 1 et 3 ( 88.6 / 88.0 /
        // 89.6 s a `n = 1e5`, sous le bruit de +/- 8 % ). En 3D il ne l'est pas du tout -- partie
        // lineaire a `n = 5e5`, paquets de 8 :
        //
        //      nu 1 :  165 it,  4.81 s        nu 2 :  120 it,  5.30 s
        //      nu 3 :  ( 156 it a paquet 16 ), 7.16 s      AMGCL : 129 it, 7.32 s
        //
        // Le graphe 3D a 15.1 non-nuls par ligne contre 6.0 en 2D : un seul passage de Jacobi y
        // propage deja l'information bien plus loin, donc le deuxieme et le troisieme ne font
        // plus qu'ajouter de la bande passante. Passer de `nu 3` a `nu 1` fait +69 %
        // d'iterations en 2D mais seulement +32 % en 3D.
        if      ( o.mg_nu > 0 ) p->nu = o.mg_nu;
        else if ( D == 3 )      p->nu = 1;
        if ( o.mg_lisseur >= 0 ) p->lisseur = o.mg_lisseur;
        if ( o.mg_recycle >= 0 ) p->recycle = o.mg_recycle;
        if ( o.mg_cheb    > 0 )  p->cheb    = TF( o.mg_cheb );
        if ( o.mg_tronque >= 0 ) p->tronque = TF( o.mg_tronque );
        p->trace = o.mg_trace;
        return p;
    }
#ifdef SF_AMGCL
    auto p = std::make_unique<Amg>();
    p->variante = o.amgvar;
    p->tol = o.lintol;
    p->maxit = o.linmax;
    return p;
#else
    std::printf( "  AMGCL absent : on retombe sur Cholesky\n" );
    return std::make_unique<Cholesky>();
#endif
}

template<class PD>
int lance( const Args &a, const Opts &o, const Nuage<PD::dim> &nu, Lineaire &lin ) {
    constexpr int D = PD::dim;
    const SI n = nu.n;

    // l'arbre est bati une fois pour toutes, sur des poids nuls, et ne sera plus que rafraichi
    double t0 = now();
    PD pd;
    pd.build( nu.P, nullptr, n, a.leaf );
    const double t_arbre = now() - t0;
    // L'ORDRE DE L'ARBRE, pour qui sait s'en servir : c'est l'agregation du multigrille maison.
    lin.ordre( pd.ids.data(), n );
    lin.positions( nu.P, n, PD::dim );                   // pour le Cholesky multi-echelle

#ifdef _OPENMP
    // AMGCL est parallelise en OpenMP, le diagramme en `std::thread` : sans ca les deux moities
    // du chronometre ne tourneraient pas sur le meme nombre de coeurs.
    omp_set_num_threads( a.par.threads );
#endif

    Newton<PD> nw( pd, lin, nu.P, a.par, o.newton );
    nw.nu.assign( n, TF( 1 ) / n );
    Trames trames;
    const bool dump = ! o.dump.empty() && trames.ouvre( o.dump );
    FILE *courbe = o.courbe.empty() ? nullptr : std::fopen( o.courbe.c_str(), "a" );
    const std::string etiquette = o.methode == "newton" ? ( o.newton.pas == NewtonOptions::ESSAI_LIMITES ? "newton essai-limites" : "newton" ) : o.po.nom();
    t0 = now();
    nw.o.apres_pas = [ & ]( int it, TF t, int reculs ) {
        if ( dump ) trames.ecrit( pd, nw.nu, a.par, "newton", 0, it, double( t ), reculs );
        if ( courbe ) {                              // le residu du point ACCEPTE, contre ce qu'il a coute
            TF p = 0;
            for ( SI i = 0; i < n; ++i ) p = std::max( p, std::fabs( nw.nu[ i ] - nw.a[ i ] ) / nw.nu[ i ] );
            std::fprintf( courbe, "%s;%s;%dD;%d;%d;%.3f;%.6e;%.6e\n", etiquette.c_str(), nu.nom.c_str(), D, it + 1, nw.st.nb_diag, now() - t0, double( p ), double( nw.merite( nw.a ) ) );
        }
    };
    PremierOrdre<PD> po( nw, o.po );
    const bool ok = o.methode == "newton" ? nw.resout( std::vector<TF>( n, TF( 0 ) ) ) : po.resout( std::vector<TF>( n, TF( 0 ) ) );
    const double total = now() - t0 + t_arbre;
    if ( courbe ) std::fclose( courbe );
    const NewtonStats &st = nw.st;
    const StatsLin &sl = lin.st;
    const double autre = total - t_arbre - st.t_maj - st.t_diag - st.t_asm - st.t_lin - st.t_lim;
    if ( o.methode != "newton" ) {
        const PremierOrdreStats &ps = po.st;
        std::printf( "  %s %s ( max|a-nu|/nu = %.2e ) : %dD n=%d -- %d iterations, %d diagrammes ( %d en recherche lineaire, %d refuses par le plancher, %d redemarrages, %d factorisations )%s, %.3f s\n",
                     o.methode == "cg" ? "gradient conjugue ( PR+ )" : "L-BFGS", ps.fin, double( ps.reste ), D, int( n ), ps.nb_iter, ps.nb_diag, ps.nb_ls, ps.nb_plancher, ps.nb_restart, ps.nb_facto,
                     ps.it_bascule >= 0 ? ( " puis NEWTON depuis l'iteration " + std::to_string( ps.it_bascule ) ).c_str() : "", ps.temps );
    }

    std::printf( "  newton %s ( max|a-nu|/nu = %.2e ) : %dD n=%d threads=%d kernel=%s maxnv=%d leaf=%d"
                 " -- %d iterations, %d diagrammes ( %d reculs ), %s%s\n",
                 st.fin, double( st.reste ), D, int( n ), a.par.threads, a.kernel.c_str(), PD::max_nv,
                 int( a.leaf ), st.nb_iter, st.nb_diag - ( o.methode != "newton" ? po.st.nb_diag : 0 ), st.nb_recul, lin.nom(),
                 sl.nb_iter ? ( " ( " + std::to_string( sl.nb_iter ) + " iterations )" ).c_str() : "" );
    std::printf( "         arbre %.3f | majorants %.3f | diagrammes %.3f | assemblage %.3f"
                 " | resolution %.3f | limites %.3f | reste %.3f | TOTAL %.3f s%s\n",
                 t_arbre, st.t_maj, st.t_diag, st.t_asm, st.t_lin, st.t_lim, autre, total,
                 o.newton.memo ? ( "   ( memoire " + std::to_string( st.t_memo ).substr( 0, 5 ) + " s, dans reste )" ).c_str() : "" );
    if ( st.nb_cell_lim )
        std::printf( "         limites : %d cellules calculees ( %.2f par germe et par iteration ), %d pas refuses par le diagramme\n",
                     int( st.nb_cell_lim ), double( st.nb_cell_lim ) / n / std::max( st.nb_iter, 1 ), st.nb_lim_refus );
    if ( st.nb_tours_essai )
        std::printf( "         essai-limites : %d essais corriges, %d cellules sous eps en tout\n", st.nb_tours_essai, int( st.nb_cell_mauvaises ) );
    if ( st.nb_cible_res )
        std::printf( "         cible : %d resolutions de plus ( %.3f s ), %d directions deformees retenues,"
                     " gain geometrique moyen sur la limite x%.2f, %d rejetees par l'amortissement\n",
                     st.nb_cible_res, st.t_cible, st.nb_cible_pris,
                     st.nb_cible_pris ? std::exp( double( st.cible_gain ) / st.nb_cible_pris ) : 1.0,
                     st.nb_cible_refus );
    if ( st.nb_tenseur )
        std::printf( "         tenseur : %d pas tensoriels, %.3f s ( compris dans limites )\n", st.nb_tenseur, st.t_tenseur );
    std::printf( "         soit %.0f %% de diagramme, %.3f s par diagramme, %.1f us/germe en tout\n",
                 100 * st.t_diag / total, st.t_diag / std::max( st.nb_diag, 1 ), 1e6 * total / n );
    std::printf( "         resolution en detail : mise en forme %.3f | hierarchie/analyse %.3f ( %d )"
                 " | resolution %.3f | pire residu lineaire %.2e\n",
                 sl.t_forme, sl.t_hier, sl.nb_hier, sl.t_res, double( sl.pire ) );
    if ( st.nb_deborde )
        std::printf( "  ATTENTION : %d cellules ont DEBORDE %d sommets pendant la resolution --"
                     " relancer avec --maxnv %d.\n", int( st.nb_deborde ), PD::max_nv, 2 * PD::max_nv );

    if ( ! o.ecrire.empty() ) {
        char entete[ 512 ];
        std::snprintf( entete, sizeof( entete ),
                       "# nuage %dD a masses egales, poids obtenus par newton ( %s, max|a-nu|/nu = %.3e )\n"
                       "# ATTENTION : produits par le banc lui-meme, donc PAS un temoin independant\n"
                       "# pour Newton -- seulement un cas de chronometrage.\n", D, st.fin, double( st.reste ) );
        if ( ecrit_nuage<D>( o.ecrire, nu, nw.w.data(), entete ) )
            std::printf( "  poids ecrits dans '%s'\n", o.ecrire.c_str() );
        else
            std::printf( "  impossible d'ecrire '%s'\n", o.ecrire.c_str() );
    }

    if ( o.cout ) {
        if constexpr ( D == 2 ) {
            TF ct = 0, at = 0, pb = 0;
            cout_transport( pd, nu.P, nw.w.data(), a.par, ct, at, pb );
            std::printf( "         COUT de transport = %.15e  ( aire totale %.15e, pire decalage de"
                         " barycentre %.3e )\n", double( ct ), double( at ), double( pb ) );
        } else
            std::printf( "         COUT : les moments ne sont ecrits qu'en 2D.\n" );
    }

    // La verification qui ne coute rien : le nuage `_equal` PORTE deja la solution, obtenue par
    // un tout autre chemin ( L-BFGS, pysdot ). Elle n'est definie qu'a une constante pres, donc on
    // recale sur le germe 0 -- exactement la jauge que Newton impose.
    if ( nu.temoin && nu.W ) {
        TF m = 0, ampl = 0;
        for ( SI i = 0; i < n; ++i ) {
            m = std::max( m, std::fabs( ( nw.w[ i ] - nw.w[ 0 ] ) - ( nu.W[ i ] - nu.W[ 0 ] ) ) );
            ampl = std::max( ampl, std::fabs( nu.W[ i ] - nu.W[ 0 ] ) );
        }
        std::printf( "  contre les poids du fichier : ecart max %.3e sur une amplitude %.3e\n",
                     double( m ), double( ampl ) );
    }
    return ok && st.nb_deborde == 0 ? 0 : 1;
}

/// LA RESOLUTION EN DEUX ETAPES ( `--agrege D`, voir `Agglo.h` ) : agreger les germes a moins de `D`,
/// resoudre le probleme REDUIT, remonter les poids -- et surtout MESURER ce que ca vaut sur le nuage
/// COMPLET, en separant les germes seuls des germes agreges. C'est ce dernier chiffre qui dit si
/// l'etape de redecoupage ( § 23.6 ) est necessaire ou decorative.
template<class PD>
int lance_agrege( const Args &a, const Opts &o, const Nuage<PD::dim> &nu, Lineaire &lin ) {
    constexpr int D = PD::dim;
    const SI n = nu.n;
    const std::vector<TF> nu_plein( n, TF( 1 ) / TF( n ) );

#ifdef _OPENMP
    omp_set_num_threads( a.par.threads );
#endif

    // ---- etape 0 : les grappes, sans diagramme et sans structure nouvelle
    double t0 = now();
    std::vector<SI> rep;
    const SI perdus = grappes_proches<D>( nu.P, n, TF( o.agrege ), rep );
    const double t_det = now() - t0;

    if ( perdus == 0 ) {
        std::printf( "  AGREGE : aucune grappe sous %.3e ( detection %.4f s ) -- rien a agreger,"
                     " on resout normalement\n", o.agrege, t_det );
        return lance<PD>( a, o, nu, lin );
    }

    // ---- etape 1 : le probleme reduit
    Nuage<D> red;
    std::vector<TF> nur;
    std::vector<SI> vers, taille;
    reduis<D>( nu.P, nu_plein, rep, red.c, nur, vers, taille );
    red.nom = nu.nom + " ( agrege )";
    red.w.assign( nur.size(), TF( 0 ) );
    red.finish();
    const SI m = red.n;
    SI nb_gr = 0, nb_dedans = 0, tmax = 0;
    for ( SI r = 0; r < m; ++r )
        if ( taille[ r ] > 1 ) { ++nb_gr; nb_dedans += taille[ r ]; tmax = std::max( tmax, taille[ r ] ); }
    std::printf( "  AGREGE seuil %.3e : %d germes -> %d, soit %d grappes ( %d germes dedans,"
                 " taille max %d ), detection %.4f s\n",
                 o.agrege, int( n ), int( m ), int( nb_gr ), int( nb_dedans ), int( tmax ), t_det );

    // ---- etape 2 : resoudre le reduit, avec le meme Newton et rien de special
    t0 = now();
    PD pdr;
    pdr.build( red.P, nullptr, m, a.leaf );
    lin.ordre( pdr.ids.data(), m );
    Newton<PD> nw( pdr, lin, red.P, a.par, o.newton );
    nw.nu = nur;
    const bool ok = nw.resout( std::vector<TF>( m, TF( 0 ) ) );
    const double t_red = now() - t0;
    const NewtonStats sr = nw.st;
    std::printf( "        reduit  : %s ( max|a-nu|/nu = %.2e ) -- %d iterations, %d diagrammes"
                 " ( %d reculs ), %.3f s\n",
                 sr.fin, double( sr.reste ), sr.nb_iter, sr.nb_diag, sr.nb_recul, t_red );

    // ---- etape 3 : remonter les poids, et MESURER sur le nuage complet
    //
    // Tous les membres d'une grappe recoivent LE MEME poids : leurs plans mutuels passent alors par le
    // milieu, donc la cellule fusionnee se partage selon la GEOMETRIE et non selon les masses voulues.
    // C'est exactement ce que le redecoupage du § 23.6 corrigerait, et la mesure ci-dessous le chiffre.
    std::vector<TF> w_plein( n );
    for ( SI i = 0; i < n; ++i ) w_plein[ i ] = nw.w[ vers[ i ] ];

    // ---- etape 3 : LE REDECOUPAGE. Chaque cellule fusionnee est partagee entre les membres de sa
    // grappe par des plans dont SEUL LE DECALAGE est libre, trouve par bissection sur l'aire. L'ecart
    // de poids n'est deduit qu'a la fin, donc l'amplification par `1 / delta` est une sortie.
    TF pire_loc = 0;
    SI nb_coupees = 0, nb_ratees = 0;
    if constexpr ( D == 2 ) {
        if ( o.agrege_corr ) {
            pdr.set_weights( nw.w.data(), a.par );
            std::vector<std::vector<SI>> membres( m );
            for ( SI i = 0; i < n; ++i )
                if ( taille[ vers[ i ] ] > 1 ) membres[ vers[ i ] ].push_back( i );
            // les cellules fusionnees, une par grappe non triviale
            std::vector<ModeleCellule> mods( m );
            std::vector<char> veut( m, 0 );
            for ( SI r = 0; r < m; ++r ) veut[ r ] = taille[ r ] > 1;
            parallel_for( m, a.par, [ & ]( SI k, int ) {
                const SI r = pdr.ids[ k ];
                if ( ! veut[ r ] ) return;
                typename PD::Cell cel;
                pdr.cellule( k, cel );
                mods[ r ].depuis( cel, r, red.P, nw.w.data() );
            } );
            std::vector<TF> dw, aires;
            for ( SI r = 0; r < m; ++r ) {
                if ( ! veut[ r ] ) continue;
                if ( ! redecoupe( mods[ r ], nu.P, nu_plein, membres[ r ], dw, aires ) ) { ++nb_ratees; continue; }
                ++nb_coupees;
                for ( size_t t = 0; t < membres[ r ].size(); ++t ) {
                    w_plein[ membres[ r ][ t ] ] = nw.w[ r ] + dw[ t ];
                    const SI i = membres[ r ][ t ];
                    pire_loc = std::max( pire_loc, std::fabs( nu_plein[ i ] - aires[ t ] ) / nu_plein[ i ] );
                }
            }
            std::printf( "        DECOUPE : %d grappes redecoupees ( %d echecs ), pire ecart LOCAL"
                         " ( sur la cellule fusionnee ) %.3e\n",
                         int( nb_coupees ), int( nb_ratees ), double( pire_loc ) );
        }
    }
    NewtonOptions om = o.newton;
    om.tol = 1e300;                                  // une seule mesure, pas de resolution
    om.trace = false;
    om.agglo = 0;
    PD pdp;
    pdp.build( nu.P, nullptr, n, a.leaf );
    lin.ordre( pdp.ids.data(), n );
    Newton<PD> nm( pdp, lin, nu.P, a.par, om );
    nm.nu = nu_plein;
    nm.resout( w_plein );
    TF pire_seuls = 0, pire_gr = 0;
    SI nv_gr = 0;
    // COMBIEN de germes seuls sont touches, et pas seulement le pire : c'est la question, parce qu'un
    // maximum sur 10^5 germes ne dit pas si l'erreur est LOCALE ( les voisins immediats d'une grappe )
    // ou repandue. L'agregation deplace une grappe sur son barycentre, et un voisin voit son aire
    // bouger de `deplacement x perimetre / aire` -- ce qui explose sur une cellule en LAMELLE.
    const TF seuils[] = { TF( 1e-6 ), TF( 1e-4 ), TF( 1e-2 ) };
    SI au_dessus[ 3 ] = { 0, 0, 0 };
    for ( SI i = 0; i < n; ++i ) {
        const TF e = std::fabs( nu_plein[ i ] - nm.a[ i ] ) / nu_plein[ i ];
        if ( taille[ vers[ i ] ] > 1 ) { pire_gr = std::max( pire_gr, e ); nv_gr += ! ( nm.a[ i ] > 0 ); }
        else {
            pire_seuls = std::max( pire_seuls, e );
            for ( int k = 0; k < 3; ++k ) au_dessus[ k ] += e > seuils[ k ];
        }
    }
    std::printf( "        COMPLET : sur les %d germes SEULS, max %.3e et %d / %d / %d au-dessus de"
                 " 1e-6 / 1e-4 / 1e-2 ; sur les %d AGREGES, max %.3e ( %d vides )%s\n",
                 int( n - nb_dedans ), double( pire_seuls ), int( au_dessus[ 0 ] ), int( au_dessus[ 1 ] ),
                 int( au_dessus[ 2 ] ), int( nb_dedans ), double( pire_gr ), int( nv_gr ),
                 pire_gr > pire_seuls * 10 ? "  <- le redecoupage manque ( § 23.6 )" : "" );

    // ---- LE COUT DE TRANSPORT, VU PAR L'AGREGAT ( § 23.12 )
    //
    // Si on renonce a resoudre les `w` des germes agreges -- ce qui est raisonnable, puisqu'ils sont
    // sous la precision machine -- alors la solution EST un agregat, et les fonctionnelles en aval
    // doivent en tenir compte. Pour le cout, la decomposition est EXACTE :
    //
    //     sum_{i in r} int_{C_i} |x - p_i|^2
    //         = int_{C_r} |x - q|^2  +  sum_{i in r} nu_i |p_i - q|^2  -  2 sum_{i in r} ( p_i - q ) . m_i
    //
    // avec `m_i = int_{C_i} ( x - q )`. Les DEUX PREMIERS termes ne demandent QUE la cellule fusionnee et
    // les positions : aucun decoupage. Le troisieme est le seul qui en depende -- et il s'annule au
    // premier ordre PARCE QUE `q` est le barycentre pondere par `nu` ( si `m_i ~ ( nu_i / nu_r ) M_r`,
    // il vaut `-2 ( M_r / nu_r ) . sum nu_i ( p_i - q ) = 0` ). Voila la vraie raison de ce choix de `q`.
    if ( o.cout ) {
        if constexpr ( D == 2 ) {
            TF ct = 0, at = 0, pb = 0;
            cout_transport( pdr, red.P, nw.w.data(), a.par, ct, at, pb );
            TF var = 0;                                  // le terme de VARIANCE INTERNE des grappes
            for ( SI i = 0; i < n; ++i ) {
                const SI r = vers[ i ];
                if ( taille[ r ] < 2 ) continue;
                TF s2 = 0;
                for ( int d = 0; d < D; ++d ) { const TF u = nu.P[ d ][ i ] - red.P[ d ][ r ]; s2 += u * u; }
                var += nu_plein[ i ] * s2;
            }
            std::printf( "        COUT vu par l'agregat = %.15e  =  %.15e ( cellules fusionnees )"
                         " + %.15e ( variance interne )\n"
                         "              ( aire totale %.15e, pire decalage de barycentre %.3e )\n",
                         double( ct + var ), double( ct ), double( var ), double( at ), double( pb ) );
        }
    }

    // ---- etape 4 : LA CORRECTION, un Newton sur le nuage COMPLET depuis ces poids.
    //
    // Ce qui reste apres le redecoupage n'est pas une erreur de partage -- celui-la est exact a 1e-10 --
    // mais la PERTURBATION que l'agregation a introduite en deplacant la grappe sur son barycentre : les
    // voisins voient leur bord bouger de `delta`, et sur une cellule en lamelle une aire bouge de
    // `delta x perimetre / aire`. Personne d'autre que le nuage complet ne peut la corriger.
    if ( o.agrege_fin ) {
        const double tf0 = now();
        PD pdf;
        pdf.build( nu.P, nullptr, n, a.leaf );
        lin.ordre( pdf.ids.data(), n );
        Newton<PD> nf( pdf, lin, nu.P, a.par, o.newton );
        nf.nu = nu_plein;
        const bool okf = nf.resout( w_plein );
        const NewtonStats sf = nf.st;
        std::printf( "        FIN     : %s ( max|a-nu|/nu = %.2e, depart %.2e ) -- %d iterations,"
                     " %d diagrammes ( %d reculs ), %.3f s\n",
                     sf.fin, double( sf.reste ), double( sf.reste0 ), sf.nb_iter, sf.nb_diag,
                     sf.nb_recul, now() - tf0 );
        return okf ? 0 : 1;
    }
    return ok ? 0 : 1;
}

/// LA BASCULE `--kernel mixte` : resoudre en SIMPLE PRECISION tant que ca avance, FINIR en
/// double. C'est la forme utile du fp32, parce que la mesure du § 19.8 est celle-ci : `float`
/// suit `double` chiffre pour chiffre pendant les premieres iterations, puis la recherche
/// lineaire s'effondre sur un plancher de bruit. Il n'y a donc rien a deviner -- on laisse la
/// phase `float` stagner, ce QU'ELLE SIGNALE, et on reprend en double a partir de ses poids.
///
/// Les deux phases partagent LE MEME solveur lineaire : il est une interface depuis le § 18,
/// donc il ne sait pas dans quel flottant le diagramme a ete calcule, et il garde sa hierarchie
/// et son sous-espace recycle a travers la bascule.
///
/// L'arbre est bati DEUX FOIS ( un par type de diagramme ) : c'est quelques pour cent du total et
/// ca evite de rendre `PowerDiagram` polymorphe pour une etude.
///
/// ON NE LAISSE PAS LA PHASE `float` STAGNER JUSQU'AU BOUT. Descendre le pas de 1 a `t_min = 1e-10`
/// coute 34 diagrammes, et Newton le fait plusieurs fois avant de declarer la stagnation : sur
/// l'uniforme 2D a `n = 1e5`, 35 des 41 diagrammes de la phase `float` sont des reculs. On lui
/// donne donc un `t_min` genereux ( `--mixte-tmin`, defaut 1e-3, dix demi-pas ) : des qu'elle ne
/// peut plus avancer d'un pas franc, elle passe la main au lieu d'insister.
template<int D>
int lance_mixte( const Args &a, const Opts &o, const Nuage<D> &nu, Lineaire &lin ) {
    constexpr int NV = D == 2 ? 64 : 128;
    const SI n = nu.n;
    const std::vector<TF> zero( n, TF( 0 ) );
    NewtonOptions of = o.newton;
    of.t_min = TF( o.mixte_tmin );
    of.progres_min = TF( o.mixte_progres );
    // LE PLANCHER DE BRUIT, POSE D'AVANCE. `|bruit|_2 = kappa eps / sqrt( n )` avec l'epsilon du
    // `float` : c'est la seule facon de rendre la main AU BON MOMENT. Les criteres reactifs -- pas
    // trop petit, progres trop faible -- se declenchent tous APRES coup, donc apres avoir paye.
    of.plancher = TF( o.mixte_kappa * 6e-8 / std::sqrt( double( n ) ) );

#ifdef _OPENMP
    omp_set_num_threads( a.par.threads );
#endif
    const double t0 = now();

    // ---- phase 1 : `float`
    PowerDiagram<D,float,NV> pdf;
    pdf.build( nu.P, nullptr, n, a.leaf );
    lin.ordre( pdf.ids.data(), n );
    Newton<PowerDiagram<D,float,NV>> n1( pdf, lin, nu.P, a.par, of );
    n1.nu.assign( n, TF( 1 ) / n );
    const StatsLin l0 = lin.st;
    n1.resout( zero );
    const double t1 = now();
    const NewtonStats s1 = n1.st;
    const StatsLin l1 = lin.st;

    // ---- phase 2 : `double`, depuis les poids de la phase 1
    PowerDiagram<D,double,NV> pdd;
    pdd.build( nu.P, nullptr, n, a.leaf );
    lin.ordre( pdd.ids.data(), n );
    Newton<PowerDiagram<D,double,NV>> n2( pdd, lin, nu.P, a.par, o.newton );
    n2.nu.assign( n, TF( 1 ) / n );
    const bool ok = n2.resout( n1.w );
    const double t2 = now();
    const NewtonStats s2 = n2.st;
    const StatsLin l2 = lin.st;

    // ---- et le TEMOIN : tout en double, depuis zero
    PowerDiagram<D,double,NV> pdt;
    pdt.build( nu.P, nullptr, n, a.leaf );
    lin.ordre( pdt.ids.data(), n );
    Newton<PowerDiagram<D,double,NV>> n3( pdt, lin, nu.P, a.par, o.newton );
    n3.nu.assign( n, TF( 1 ) / n );
    n3.resout( zero );
    const double t3 = now();
    const NewtonStats s3 = n3.st;
    const StatsLin l3 = lin.st;

    std::printf( "  MIXTE %s ( max|a-nu|/nu = %.2e ) : %dD n=%d threads=%d\n", s2.fin, double( s2.reste ), D, int( n ), a.par.threads );
    std::printf( "        phase float  : %2d it, %4d diag ( %3d reculs ), reste %.2e, %7.3f s"
                 "   [ diag %6.3f  lineaire %7.3f / %6d it ]  %s\n",
                 s1.nb_iter, s1.nb_diag, s1.nb_recul, double( s1.reste ), t1 - t0,
                 s1.t_diag, l1.total() - l0.total(), l1.nb_iter - l0.nb_iter, s1.fin );
    std::printf( "        phase double : %2d it, %4d diag ( %3d reculs ), reste %.2e, %7.3f s"
                 "   [ diag %6.3f  lineaire %7.3f / %6d it ]\n",
                 s2.nb_iter, s2.nb_diag, s2.nb_recul, double( s2.reste ), t2 - t1,
                 s2.t_diag, l2.total() - l1.total(), l2.nb_iter - l1.nb_iter );
    std::printf( "        TOTAL MIXTE  : %2d iterations, %4d diagrammes, %.3f s   ( diagramme seul %.3f s )\n",
                 s1.nb_iter + s2.nb_iter, s1.nb_diag + s2.nb_diag, t2 - t0, s1.t_diag + s2.t_diag );
    std::printf( "        TEMOIN double: %2d it, %4d diag ( %3d reculs ), reste %.2e, %7.3f s"
                 "   [ diag %6.3f  lineaire %7.3f / %6d it ]  %s\n",
                 s3.nb_iter, s3.nb_diag, s3.nb_recul, double( s3.reste ), t3 - t2,
                 s3.t_diag, l3.total() - l2.total(), l3.nb_iter - l2.nb_iter, s3.fin );
    const double g = t3 - t2 > 0 ? 100.0 * ( 1 - ( t2 - t0 ) / ( t3 - t2 ) ) : 0;
    const double gd = s3.t_diag > 0 ? 100.0 * ( 1 - ( s1.t_diag + s2.t_diag ) / s3.t_diag ) : 0;
    std::printf( "        BILAN : %+.1f %% sur le total, %+.1f %% sur le diagramme seul\n", g, gd );
    return ok ? 0 : 1;
}

template<int D>
int deroule( const Args &a, const Opts &o ) {
    int bad = 0;
    std::printf( "=== %dD\n", D );
    for ( const Nuage<D> &nu : a.nuages<D>() ) {
        if ( nu.absent ) {
            std::printf( "  %-28s : ABSENT ( --cases DIR, ou lancer 2d_des_familles/cases/gen_cases.py )\n",
                         nu.nom.c_str() );
            continue;
        }
        std::printf( "-- %s\n", nu.nom.c_str() );
        auto lin = fabrique<D>( o );
        if ( a.kernel == "mixte" ) {
            bad += lance_mixte<D>( a, o, nu, *lin );
            continue;
        }
        bad += dispatch<D>( a, [ & ]( auto tag ) {
            using T = typename decltype( tag )::type;
            return o.agrege > 0 ? lance_agrege<T>( a, o, nu, *lin ) : lance<T>( a, o, nu, *lin );
        } );
    }
    return bad;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    a.n = 100000;
    Opts o;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if ( a.parse( s, i, argc, argv ) ) continue;
        else if ( s == "--newton-tol" ) o.newton.tol = std::atof( val() );
        else if ( s == "--newton-max" ) o.newton.maxit = std::atoi( val() );
        else if ( s == "--lin-tol" )    o.lintol = std::atof( val() );
        else if ( s == "--lin-max" )    o.linmax = std::atoi( val() );
        else if ( s == "--solver" )     o.solver = val();
        else if ( s == "--mg-agreg" )   o.mg_agreg = std::atoi( val() );
        else if ( s == "--mg-nu" )      o.mg_nu = std::atoi( val() );
        else if ( s == "--mg-lisseur" ) o.mg_lisseur = std::atoi( val() );
        else if ( s == "--mg-recycle" ) o.mg_recycle = std::atoi( val() );
        else if ( s == "--mg-cheb" )    o.mg_cheb = std::atof( val() );
        else if ( s == "--mg-tronque" ) o.mg_tronque = std::atof( val() );
        else if ( s == "--mg-trace" )   o.mg_trace = 1;
        else if ( s == "--amg-var" )    o.amgvar = std::atoi( val() );
        else if ( s == "--mchol-rho" )  o.mchol_rho = std::atof( val() );
        else if ( s == "--cp-refaire" ) o.cp_refaire = std::atoi( val() );
        else if ( s == "--cp-cgmax" )   o.cp_cgmax = std::atoi( val() );
        else if ( s == "--cp-simplicial" ) o.cp_super = false;
        else if ( s == "--cp-sans-compense" ) o.cp_comp = false;
        else if ( s == "--cp-seuil" )   o.cp_seuil = std::atof( val() );
        else if ( s == "--cp-sauts" )   o.cp_sauts = std::atoi( val() );
        else if ( s == "--mchol-ech" )  o.mchol_ech = std::atoi( val() );
        else if ( s == "--ecrire" )     o.ecrire = val();
        else if ( s == "--quiet" )      o.newton.trace = false;
        else if ( s == "--pas" ) {
            const std::string v = val();
            o.newton.pas = v == "dyadique" ? NewtonOptions::DYADIQUE : v == "facteur" ? NewtonOptions::FACTEUR
                         : v == "tenseur" ? NewtonOptions::TENSEUR : v == "essai-limites" ? NewtonOptions::ESSAI_LIMITES
                         : v == "modele" ? NewtonOptions::MODELE : v == "merite" ? NewtonOptions::MERITE
                         : v == "grille2" ? NewtonOptions::GRILLE2
                         : NewtonOptions::ESSAIS;
        }
        else if ( s == "--facteur" )    o.newton.facteur = std::atof( val() );
        else if ( s == "--theta-mult" ) o.newton.theta_mult = std::atof( val() );
        else if ( s == "--confiance" )  o.newton.confiance = std::atof( val() );
        else if ( s == "--beta0" )      o.newton.beta0 = std::atof( val() );
        else if ( s == "--mult-ok" )    o.newton.mult_ok = std::atof( val() );
        else if ( s == "--dump" )       o.dump = val();
        else if ( s == "--methode" )    o.methode = val();
        else if ( s == "--memoire" )    o.po.memoire = std::atoi( val() );
        else if ( s == "--c2" )         o.po.c2 = std::atof( val() );
        else if ( s == "--precond" )    o.po.precond = std::atoi( val() );
        else if ( s == "--refacto" )    o.po.refacto = std::atoi( val() );
        else if ( s == "--sauter-borne" ) o.po.sauter_borne = std::atoi( val() );
        else if ( s == "--refacto-borne" ) o.po.refacto_borne = std::atof( val() );
        else if ( s == "--refacto-taux" ) o.po.refacto_taux = std::atof( val() );
        else if ( s == "--plancher" )   o.po.plancher = std::atoi( val() );
        else if ( s == "--max-ls" )     o.po.max_ls = std::atoi( val() );
        else if ( s == "--bascule" )    o.po.bascule = std::atof( val() );
        else if ( s == "--bascule-it" ) o.po.bascule_it = std::atoi( val() );
        else if ( s == "--courbe" )     o.courbe = val();
        else if ( s == "--mixte-tmin" ) o.mixte_tmin = std::atof( val() );
        else if ( s == "--mixte-progres" ) o.mixte_progres = std::atof( val() );
        else if ( s == "--mixte-kappa" ) o.mixte_kappa = std::atof( val() );
        else if ( s == "--agrege" )     o.agrege = std::atof( val() );
        else if ( s == "--agrege-brut" ) o.agrege_corr = false;
        else if ( s == "--agrege-fin" )  o.agrege_fin = true;
        else if ( s == "--cout" )       o.cout = true;
        else if ( s == "--memo" )       o.newton.memo = true;
        else if ( s == "--cible" ) {
            const std::string v = val();
            o.newton.cible.mode = v == "plafond" ? OptionsCible::PLAFOND : v == "gel" ? OptionsCible::GEL
                                : v == "seules" ? OptionsCible::SEULES : OptionsCible::NON;
        }
        else if ( s == "--cible-anneaux" ) o.newton.cible.anneaux = std::atoi( val() );
        else if ( s == "--cible-f" )      o.newton.cible.facteur = std::atof( val() );
        else if ( s == "--cible-essais" ) o.newton.cible.essais = std::atoi( val() );
        else if ( s == "--cible-seuil" )  o.newton.cible.seuil = std::atof( val() );
        else if ( s == "--cible-repris" ) {
            const std::string v = val();
            o.newton.cible.repris = v == "anneau" ? OptionsCible::ANNEAU
                                  : v == "mangeurs" ? OptionsCible::MANGEURS : OptionsCible::GLOBAL;
        }
        else if ( s == "--cible-ep" ) o.newton.cible.ep_anneau = std::atoi( val() );
        else if ( s == "--cible-juge" ) o.newton.cible.juge =
            std::string( val() ) == "polynome" ? OptionsCible::POLYNOME : OptionsCible::FLUX;
        else if ( s == "--cible-filtre" ) o.newton.cible.filtre = std::atof( val() );
        else if ( s == "--cible-max-seules" ) o.newton.cible.max_seules = std::atoi( val() );
        else if ( s == "--cible-budget" ) o.newton.cible.budget = std::atof( val() );
        else if ( s == "--cible-critere" ) o.newton.cible.critere =
            std::string( val() ) == "pop" ? OptionsCible::POPULATION : OptionsCible::MIN;
        else if ( s == "--lim-tol" )    o.newton.lim.tol = std::atof( val() );
        else if ( s == "--lim-coeff" )  o.newton.lim.coeff = std::atof( val() );
        else if ( s == "--t-min" )      o.newton.t_min = std::atof( val() );
        else if ( s == "--residu" ) {
            const std::string v = val();
            o.newton.residu = v == "barriere" ? NewtonOptions::BARRIERE : v == "log" ? NewtonOptions::LOG
                            : v == "puissance" ? NewtonOptions::PUISSANCE : NewtonOptions::LIN;
        }
        else if ( s == "--merite" ) {
            const std::string v = val();
            o.newton.merite_res = v == "barriere" ? NewtonOptions::BARRIERE : v == "log" ? NewtonOptions::LOG
                                : v == "puissance" ? NewtonOptions::PUISSANCE : v == "pire" ? NewtonOptions::PIRE
                                : v == "log2" ? NewtonOptions::LOG2 : NewtonOptions::LIN;
        }
        else if ( s == "--puis" )       o.newton.puis = std::atof( val() );
        else if ( s == "--bascule-residu" ) o.newton.bascule_residu = std::atof( val() );
        else if ( s == "--bascule-pas" ) o.newton.bascule_pas = std::atof( val() );
        else if ( s == "--g2-dir" ) {
            const std::string v = val();
            o.newton.g2_dir = v == "sonde" ? NewtonOptions::SONDE : NewtonOptions::PREC;
        }
        else if ( s == "--g2-na" )      o.newton.g2_na = std::atoi( val() );
        else if ( s == "--g2-nb" )      o.newton.g2_nb = std::atoi( val() );
        else if ( s == "--g2-bmax" )    o.newton.g2_bmax = std::atof( val() );
        else if ( s == "--g2-muet" )    o.newton.g2_trace = false;
        else if ( s == "--g2-bpos" )    o.newton.g2_bpos = true;
        else if ( s == "--g2-modele" )  o.newton.g2_modele = true;
        else if ( s == "--g2-desc" )    o.newton.g2_desc = std::atoi( val() );
        else if ( s == "--g2-sonde-reelle" ) o.newton.g2_sonde_reelle = true;
        else if ( s == "--g2-back" )    o.newton.g2_back = std::atoi( val() );
        else if ( s == "--g2-tol" )     o.newton.g2_tol = std::atof( val() );
        else if ( s == "--g2-sans-verif" ) o.newton.g2_verif = false;
        else if ( s == "--profil" )     o.newton.profil = std::atoi( val() );
        else if ( s == "--combi" )      o.newton.combi = std::atoi( val() );
        else if ( s == "--modele" )     o.newton.modele = std::atoi( val() );
        else if ( s == "--mod-q" )      o.newton.mod_q = std::atoi( val() );
        else if ( s == "--mod-k" )      o.newton.mod_k = std::atoi( val() );
        else if ( s == "--mod-limites" ) o.newton.mod_limites = true;
        else if ( s == "--mod-hors" )   o.newton.mod_hors = std::atof( val() );
        else if ( s == "--mod-juge" ) {
            const std::string v = val();
            o.newton.mod_juge = v == "log" ? NewtonOptions::LOG : v == "barriere" ? NewtonOptions::BARRIERE
                              : v == "puissance" ? NewtonOptions::PUISSANCE : NewtonOptions::PIRE;
        }
        else if ( s == "--oracle" )     o.newton.oracle = std::atoi( val() );
        else if ( s == "--oracle-pire" ) o.newton.oracle_pire = true;
        else if ( s == "--profil-nb" )  o.newton.profil_nb = std::atoi( val() );
        else if ( s == "--profil-ratio" ) o.newton.profil_ratio = std::atof( val() );
        else if ( s == "--relax" )      o.newton.t0 = std::atof( val() );
        else if ( s == "--refus" )      o.newton.refus = std::atoi( val() );
        else if ( s == "--diag-lap" )   o.newton.diag_lap = true;
        else if ( s == "--agglo" )      o.newton.agglo = std::atof( val() );
        else if ( s == "--mod-frac" )   o.newton.mod_frac = std::atoi( val() );
        else if ( s == "--mer-patience" ) o.newton.mer_patience = std::atoi( val() );
        else if ( s == "--mer-raffine" ) o.newton.mer_raffine = std::atoi( val() );
        else if ( s == "--mer-sans-limites" ) o.newton.mer_limites = false;
        else if ( s == "--sans-plancher-aire" ) o.newton.plancher_aire = false;
        else if ( s == "--g-ecrete" )   o.newton.g_ecrete = std::atof( val() );
        else {
            std::printf( "usage: newton [options]\n" );
            Args::usage();
            std::printf(
                "  --solver S      auto ( defaut : mg en 3D, amg en 2D ) | mg | amg | chol | chsup ( supernodal )\n"
                "                  | chprec ( factorisation GELEE en preconditionneur d.un CG, § 24.19 )\n"
                "                  | mchol ( Cholesky multi-echelle, § 24.17 )\n"
                "  --cp-refaire K --cp-cgmax K --cp-simplicial   chprec : refaire la factorisation tous les K solves,\n"
                "                  le compte de CG au-dela duquel on rafraichit, et le simplicial au lieu du supernodal\n"
                "  --mg-agreg S --mg-nu N --mg-lisseur 0|1|2 --mg-recycle K --mg-cheb R\n"
                "  --mg-tronque T --mg-trace   les reglages du multigrille maison\n"
                "  --amg-var V     0 = agregation+spai0 | 1 = agregation+GS | 2 = Ruge-Stuben+GS  (0)\n"
                "  --newton-tol T  arret sur max|a_i - nu| / nu             (1e-6)\n"
                "  --newton-max K  iterations au maximum                    (100)\n"
                "  --lin-tol T     arret du solveur lineaire, relatif       (1e-10)\n"
                "  --lin-max K     iterations du solveur lineaire au plus   (20000)\n"
                "  --ecrire FILE   ecrire les poids trouves au format de cases/ ( le dernier nuage deroule )\n"
                "  --mixte-tmin T  --kernel mixte : le pas sous lequel la phase float passe la main au double  (1e-3)\n"
                "  --mixte-progres F  ... ou le gain minimal sur |r|_2 par iteration ( 0 : eteint )              (0)\n"
                "  --mixte-kappa K    ... et le PLANCHER DE BRUIT vise, en unites de eps_float / sqrt( n )  (30)\n"
                "  --agrege D      resoudre EN DEUX ETAPES : agreger les germes a moins de D, resoudre le probleme reduit,\n"
                "                  remonter les poids, et MESURER ce que ca vaut sur le nuage complet ( Agglo.h )\n"
                "  --agrege-brut   ... sans le redecoupage des cellules fusionnees ( pour voir ce qu.il apporte )\n"
                "  --agrege-fin    ... puis finir par un Newton sur le nuage COMPLET depuis les poids obtenus\n"
                "  --cout          calculer le COUT DE TRANSPORT sum_i int |x - p_i|^2 a la fin ( 2D )\n"
                "  --quiet         pas de trace par iteration\n"
                "  --pas P         essais ( KMT, defaut ) | dyadique | facteur | tenseur | essai-limites ( 2D )\n"
                "                  | merite : le pas qui MINIMISE le merite le long de la direction ( au lieu du premier qui passe )\n"
                "                  | grille2 : RECHERCHE A DEUX VARIABLES, w + alpha d + beta ( deplacement precedent ) -- un INSTRUMENT\n"
                "  --g2-dir D      grille2 : la seconde direction -- prec ( deplacement precedent ) | sonde ( Newton au bout du rayon )  (prec)\n"
                "  --g2-na N --g2-nb N --g2-bmax B --g2-muet   grille2 : barreaux en alpha, en beta, amplitude de beta, silence   (5, 5, 1)\n"
                "  --g2-bpos       grille2 : ne balayer que beta >= 0 ( avec la sonde, le cote negatif ne gagne jamais )\n"
                "  --g2-modele     grille2 : evaluer la grille par le POLYNOME EXACT ( 2D ) au lieu de diagrammes -- la grille devient gratuite\n"
                "  --g2-desc K     grille2 modele : K evaluations de DESCENTE DE GRADIENT apres la grille        (0)\n"
                "  --g2-sonde-reelle  grille2 modele : aires du point de sonde par un VRAI diagramme ( temoin )\n"
                "  --g2-back K     grille2 modele : divisions par deux permises si le vrai merite ne descend pas   (4)\n"
                "                  | modele ( 2D ) : le pas cherche dans le SPAN de plusieurs directions, sur le modele\n"
                "                  polynomial d.aire -- aucun diagramme pour chercher, un seul pour verifier\n"
                "  --mod-q Q       modele : le pas du simplexe cherche, 1/Q                        (4)\n"
                "  --mod-k K       modele : combien de directions ( 1 | 2 = + log | 3 = + barriere ; 2 SUFFIT )  (2)\n"
                "  --mod-limites   modele : le pas par les LIMITES EXACTES au lieu de la racine du modele ( un diagramme de\n"
                "                  moins sur les cas sains, 34 de plus sur le nuage degenere, et 30 % de temps en plus )\n"
                "  --mod-juge J    modele : sur quoi il choisit son melange -- pire ( defaut ) | log | barriere | puissance\n"
                "  --mod-hors F    modele, juge pire : la fraction de cellules laissee DEHORS du maximum ( 0 = max strict )  (1e-4)\n"
                "  --beta0 B       essai-limites : le premier essai                          (0.25)\n"
                "  --mult-ok M     essai-limites : apres un essai passe direct, beta *= M     (2)\n"
                "  --confiance C   essai-limites : apres un pas corrige, au moins C * t       (0 = beta inchange)\n"
                "  --theta-mult M  tenseur : cible partielle theta = M * alpha*        (5)\n"
                "  --facteur F     t = F * alpha* en mode facteur                (0.9)\n"
                "  --cible M       non ( defaut ) | plafond ( mesure seulement ) | gel : deformer nu pour\n"
                "                  allonger le pas admissible, sans refaire le diagramme ( Cible.h )\n"
                "                  | seules ( le doseur sur la liste de essai-limites, le bon predicteur )\n"
                "  --cible-max-seules K  seules : au-dela de K mauvaises on renonce      (0 = pas de limite)\n"
                "  --cible-budget B   seules : deformation autorisee, en |db|/|b| -- LE compromis  (0 = sans plafond)\n"
                "  --cible-anneaux N  epaisseur du patch autour des cellules qui bornent      (2)\n"
                "  --cible-f F     le pas vise, en multiples du pas lu sur les flux         (2)\n"
                "  --cible-essais K   bissections sur lambda ( = resolutions de plus )      (5)\n"
                "  --cible-seuil S    au-dessus de ce U*, on ne touche a rien               (0.5)\n"
                "  --cible-critere C  min ( le pas de la pire ) | pop ( le nombre de malades )  (min)\n"
                "  --cible-repris R   ou la masse est reprise : global | anneau ( par amas ) | mangeurs  (global)\n"
                "  --cible-ep N       anneau : epaisseur de la couronne donneuse                (1)\n"
                "  --cible-juge J     flux ( U, § 15.13 ) | polynome ( l'aire a combinatoire figee, § 7 )  (flux)\n"
                "  --cible-filtre F   polynome : on modelise les cellules dont U < F * vise      (8)\n"
                "  --lim-tol T     precision relative des limites               (1e-2)\n"
                "  --lim-coeff C   ou verifier la prediction                    (0.99)\n"
                "  --t-min T       sous ce pas, STAGNATION                      (1e-10)\n"
                "  --residu R      lin ( a - nu ) | barriere ( x - 1/x, x = a/nu ) | log | puissance  (log)\n"
                "  --puis P        puissance : g = ( x^P - 1 ) / P -- P = 1 EST lin, P = 0 EST log  (0.5)\n"
                "  --bascule-residu R  repasser a lin des que max|a-nu|/nu <= R ( 0 : jamais ; inerte si --residu lin )  (2)\n"
                "  --bascule-pas T     repasser a lin des que le pas accepte atteint T -- la contrainte MESUREE ( 0 : inactive )\n"
                "  --merite R      LE JUGE DE L'AMORTISSEMENT, separement de la direction : lin | barriere | log |\n"
                "                  puissance | pire ( max|a-nu|/nu ) | log2 ( sum ( log x )^2, NON CENTRE )   ( defaut : comme --residu )\n"
                "  --profil K      a l'iteration K, balayer t et imprimer LES TROIS merites le long de la direction, puis sortir\n"
                "  --profil-nb N   nombre de pas du profil ( t = relax / ratio^k )              (24)\n"
                "  --profil-ratio F   le rapport entre deux barreaux du profil ( 2 : dyadique )    (2)\n"
                "  --oracle Q      le MEILLEUR melange des trois directions a chaque iteration, force brute ( pas du simplexe 1/Q )\n"
                "  --modele K      a l.iteration K, batir le modele multi-directions ( PolyMulti ) et mesurer ce qu.il predit\n"
                "  --combi K       a l.iteration K, balayer le SIMPLEXE des directions lin/log/barriere et dire ce que gagne chaque melange\n"
                "  --relax R       le premier pas essaye ( 1 = Newton entier ) -- la relaxation a la main  (1)\n"
                "  --refus K       a l'iteration K, dire laquelle des deux clauses refuse chaque essai\n"
                "  --agglo D       a l'iteration 0, detecter les grappes de germes a moins de D ( union-find sur les aretes\n"
                "                  de Delaunay du diagramme de Voronoi : aucune structure de plus ), puis sortir\n"
                "  --mod-frac N    modele : combien de fractions de alpha* on essaye ( 1 = aucune recherche de relaxation )  (1)\n"
                "  --mer-patience K   merite : barreaux qu.on laisse remonter avant de s.arreter        (1)\n"
                "  --mer-raffine K    merite : evaluations de SECTION DOREE apres l.argmin ( 0 : aucune )  (0)\n"
                "  --mer-sans-limites merite : partir de t0 au lieu du pas des limites exactes ( echelle vraiment dyadique )\n"
                "  --sans-plancher-aire   ETEINDRE le plancher d.aire de l.amortissement ( le merite log le penalise deja )\n"
                "  --g-ecrete X    l.ecretage de g dans le MERITE ( 0 : aucun, une cellule vide coute +infini )  (1e-8)\n"
                "  --diag-lap      tracer ce qui rend L dure : etalement de la diagonale, des poids d'aretes, et l'ANISOTROPIE par ligne\n"
                "  --methode M     newton ( defaut ) | lbfgs | cg : le premier ordre sur le dual ( PremierOrdre.h )\n"
                "  --memoire K     L-BFGS : paires gardees                                     (10)\n"
                "  --c2 C          Wolfe forte |phi'( alpha )| <= C |phi'( 0 )|                   (0.5 ; CG : 0.1)\n"
                "  --precond P     H0 : 0 = gamma I | 1 = diagonale du laplacien | 2 = laplacien du depart, factorise une fois  (2)\n"
                "  --refacto K     precond 2 : refactoriser toutes les K iterations                  (0 : jamais)\n"
                "  --refacto-borne F  precond 2 : refactoriser quand le plancher a borne le pas sous F   (0.25)\n"
                "  --refacto-taux T   precond 2 : refactoriser quand |r|_2 n'a pas ete divise par 1/T   (0.5 ; 0 : jamais)\n"
                "  --plancher 0|1  refuser un pas qui met une cellule sous eps ( KMT )                (1)\n"
                "  --sauter-borne 0|1  L-BFGS : ne pas garder la paire d'un pas borne par le plancher (1)\n"
                "  --max-ls K      diagrammes par recherche lineaire, au plus                     (12)\n"
                "  --bascule R     passer a Newton des que max|a-nu|/nu <= R                     (0 : jamais)\n"
                "  --bascule-it K  passer a Newton apres K iterations                            (0 : jamais)\n"
                "  --courbe FILE   CSV ( ajoute ) : methode;cas;dim;it;diagrammes;temps;max|a-nu|/nu;|r|_2 apres chaque pas\n"
                "  --memo          3D : les facettes du dernier diagramme accepte proposees en premier au suivant\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    a.finalise();
    o.po.tol = o.newton.tol;
    o.po.trace = o.newton.trace;
    o.po.maxit = 10 * o.newton.maxit;                    // le premier ordre a besoin de bien plus d'iterations
    if ( o.methode == "cg" ) { o.po.methode = PremierOrdreOptions::CG; if ( o.po.c2 == TF( 0.5 ) ) o.po.c2 = 0.1; }

    int bad = 0;
    if ( a.dims != 3 ) bad += deroule<2>( a, o );
    if ( a.dims != 2 ) bad += deroule<3>( a, o );
    return bad ? 1 : 0;
}
