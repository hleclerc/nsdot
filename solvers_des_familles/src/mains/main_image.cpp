// =====================================================================================
// LA DENSITE IMAGE SUR LE CPU : une grille de pixels pour source, l'integrale SUR LE BORD
// ( `solver/Image.h` ), et le solveur du banc par-dessus.
//
// Trois questions, une par mode :
//
//   --check    EST-CE JUSTE ? Le temoin est le DECOUPAGE EN PIXELS ( Sutherland-Hodgman ), ecrit
//              ici et qui ne partage pas une ligne avec le noyau : deux algorithmes opposes pour
//              le meme nombre. Les coefficients de hessienne sont verifies par DIFFERENCE FINIE
//              CENTREE sur la masse de la cellule -- le seul controle qui ne suppose ni le signe
//              ni le facteur -- et la derivee en contraste de meme.
//
//   --chrono   CE QUE LA METHODE VAUT, A MACHINE EGALE : le bord contre le decoupage, sur les
//              memes cellules du meme moteur, dans le meme binaire. Le rapport suit la SURFACE
//              CONTRE LE PERIMETRE ( `W / sqrt( n )` pixels par cote de cellule ).
//
//   ( defaut ) RESOUDRE : Newton amorti sous l'image, avec la continuation en contraste
//              `rho_t = ( 1 - t ) + t rho` et le pas par les LIMITES EN MASSE ( `--pas
//              essai-limites` : la cellule pincee est relevee seule au lieu de raboter le pas
//              global -- c'est ce qui manque au banc GPU ).
//
// Et partout `--acc float` : L'IMAGE STOCKEE en simple precision -- le gros tableau, celui dont
// la bande passante decide. La MARCHE, elle, reste en double, et le § 19 dit pourquoi : ses
// quantites sont d'ordre 1 pour un resultat d'ordre `h`. `--check` mesure les deux separement.
//
//   xmake run image --check -n 20000 --image 512
//   xmake run image --chrono -n 1000000 --image 512 --threads 8
//   xmake run image --threads 8 -n 200000 --image 512 --pas essai-limites
//   xmake run image --threads 8 -n 200000 --image 512 --etapes 8
//
// 2D seulement.
// =====================================================================================

#include "bench/Dispatch.h"
#include "cell/Balayage.h"
#include "solver/Image.h"
#include "solver/Lineaire.h"
#include "solver/Multigrille.h"
#include "solver/Newton.h"
#include "solver/Ecrasement.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace sf;

namespace {

struct Opts {
    NewtonOptions newton;
    // LE RESIDU REVIENT A `lin` ICI, contre le defaut de la bibliotheque ( `log`, § 24 ), PAR PRECAUTION
    // ET NON PAR MESURE : ce main enchaine des centaines de systemes voisins ( § 12 ), donc il est dans
    // le meme regime de continuation ou `log` a ete mesure mauvais ( § 9.6 ) et ou il fait echouer
    // `densite` a `sigma = 0.02`. A remesurer ici avant de conclure quoi que ce soit.
    // ( `--residu log` pour l'essayer quand meme. )
    Opts() { newton.residu = NewtonOptions::LIN; }
    // LE SOLVEUR LINEAIRE EST LE PREMIER POSTE, ET C'ETAIT LE SEQUENTIEL. Eigen `SimplicialLDLT`
    // etait le defaut : 84 s sur 142 pour le relevement a `n = 1e5`, 106 s sur 157 pour
    // `essai-limites`, soit 59 a 68 % du temps dans UN SEUL FIL -- 2.3 CPU occupes sur 8 demandes.
    // C'est ce que `--threads 8` n'achetait pas. Mesure a `n = 1e5`, relevement, memes 566
    // diagrammes pour tous ( la direction est la meme ) :
    //
    //      chol                        145.1 s   ( 2.3 CPU )   hierarchie 25.8 s
    //      amg var 0                   103.8 s   ( 5.9 CPU )   hierarchie 13.1 s
    //      mg maison, porte tel quel   157.4 s   ( 6.8 CPU )   hierarchie  1.8 s
    //
    // LE SOLVEUR MAISON EST PASSE DEVANT, et c'est lui le defaut. Trois ajouts l'y ont mene ( § 17 ) :
    // la prolongation LISSEE, le RECYCLAGE du sous-espace ( -19 % d'iterations ) et le lisseur de
    // CHEBYSHEV ( -21 % de plus ). Mesure finale, a diagrammes identiques partout :
    //
    //                          5e3     2e4    2e4 trous   2e4 rho     1e5
    //      chol               1.91   10.99        -          -          -
    //      amg var 0          1.70    7.26      5.63       7.05      92.57
    //      mg maison          1.54    7.44      4.48       6.47      90.43
    //
    // Quatre cas sur cinq, et le cinquieme ( 2e4 ) est a 2.5 %, sous le bruit de +/- 8 % mesure en
    // rejouant la meme commande. `amg` reste a un drapeau, et reste le TEMOIN : il n'a pas ete
    // regle sur ce cas d'usage, le notre si.
    std::string   solver = "mg";
    // RUGE-STUBEN ETAIT LE DEFAUT, ET C'EST L'AUTRE MOITIE DU PROBLEME. Il avait ete choisi sur le
    // NUAGE DE LIGNES, ou son choix de noeuds grossiers arete par arete divise les iterations par
    // trois ( `Lineaire.h` ). Sur une densite image il perd, et il est doublement sequentiel chez
    // amgcl : le coarsening comme le lisseur de Gauss-Seidel. L'agregation lissee + spai0 est
    // parallele des deux cotes.
    int           amgvar = Amg::SA_SPAI0;
    // `1e-10` N'ETAIT JAMAIS REMPLACE, et une direction de Newton AMORTIE n'en demande pas tant :
    // la recherche lineaire verifie le pas, et une direction a dix chiffres n'est pas meilleure
    // qu'une a six. Mesure a `n = 2e4` ( amg var 0 ) : 8.27 s a 1e-10, 7.58 a 1e-6, 7.06 a 1e-4.
    // On s'arrete a `1e-6` : a `1e-4` le compte de diagrammes remonte ( 566 -> 574 a `n = 1e5` ),
    // donc le gain de solveur commence a etre repaye en geometrie.
    double        amgtol = 1e-6;           ///< residu RELATIF demande au solveur lineaire
    int           mg_nu = 3, mg_gros = 120, mg_k = 0, mg_stop = 1000;   ///< le multigrille maison
    int           mg_lisse = 1;            ///< la PROLONGATION LISSEE ( sinon : constante par morceaux )
    int           mg_agreg = 8;            ///< germes par paquet ( puissance de deux )
    int           mg_lisseur = 2;          ///< 0 : Jacobi amorti ; 1 : spai0 ; 2 : Chebyshev
    double        mg_cheb = 10;            ///< `lmin = lmax / cheb` pour Chebyshev
    int           mg_recycle = 2;          ///< solutions gardees pour le demarrage de Galerkin
    int           amg_refaire = 1;         ///< la hierarchie d'AMGCL gardee N resolutions ( mesure : 1 )
    int           mg_refaire = 4;          ///< la hierarchie refaite toutes les N resolutions
    double        mg_omega_p = 0.7;        ///< l'amortissement du lissage de `P`
    double        mg_tronque = 0.2;        ///< troncature de `P`, en fraction du max de la ligne
    double        mg_force = 0.0;          ///< connexions FORTES seules ( 0 : eteint, cf. § 17 )
    int           mg_trace = 0;            ///< la taille et le remplissage de chaque niveau
    int           mg_exact = 1;            ///< le niveau grossier RESOLU ( Cholesky ) au lieu de lisse
    int           taille = 512;        ///< l'image de synthese : `taille x taille`
    std::string   pgm;                 ///< a la place de la synthese
    bool          trou = true;         ///< l'image de synthese a un carre a zero
    TF            contraste = 1;       ///< `rho <- ( 1 - F ) + F rho` une fois pour toutes ( avant tout )
    int           etapes = 1;          ///< la continuation UNIFORME : `t = 1/K ... 1`
    int           etapes_geo = 0;      ///< la continuation GEOMETRIQUE : `t = 1 - 2^-1 ... 1 - 2^-K`, puis 1
    std::string   chemin = "melange";  ///< melange ( le plancher `1 - t` ) | conv ( la largeur `sigma` )
    double        sigma0 = 1;          ///< conv : la largeur de la premiere etape, en fraction du cote
    bool          adaptatif = false;   ///< reculer sur une etape intermediaire quand une etape est refusee
    TF            seuil = TF( 0.05 );  ///< adaptatif : une etape est refusee si `min a_i < seuil * nu`
    TF            seuil_res = TF( 2 );  ///< adaptatif : ... ou si `max |a_i - nu| / nu > seuil_res` ( 0 : eteint )
    TF            seuil_dw = 0;        ///< adaptatif : ... ou si `max |d| / h^2 > seuil_dw`, `d` la correction
                                       ///< de poids reclamee par le changement de densite ( 0 : eteint )
    bool          diagnostic = false;  ///< imprimer les predicteurs de difficulte de chaque etape
    bool          relevement = false;  ///< l'etude du relevement par moindres carres ponderes
    double        rel_depuis = -1;     ///< le `lambda` ou l'on s'arrete et converge ( defaut : l'avant-dernier )
    double        rel_vers = 0;        ///< le `lambda` cible, celui qui pince
    TF            rel_seuil = TF( 0.5 );   ///< les cellules a relever : `a_i < rel_seuil * nu`
    int           rel_largeur_max = 8;
    std::vector<int> rel_largeurs{ 0, 1, 2, 4 };
    std::vector<TF>  rel_kappas{ 1, 100, 10000 };
    std::vector<TF>  rel_mus;              ///< l'amortissement de Levenberg-Marquardt, en fraction de la diagonale
    int           rel_cg = 20000;
    bool          rel_identite = true;     ///< verifier que le pondere GLOBAL rend Newton
    bool          rel_puis_newton = true;  ///< apres le relevement local, un Newton complet
    bool          rel_solve = false;       ///< ... et jusqu'a CONVERGENCE, pour compter les diagrammes
    bool          local_nl = false;        ///< LE SOUS-PROBLEME LOCAL, resolu non lineairement
    bool          local_barriere = true;   ///< son merite : `x - 1/x` ( penalise les petites aires )
    // QUATRE ITERATIONS DE GAUSS-NEWTON LOCAL SUFFISENT, ET ELLES SONT INDISPENSABLES. Le
    // placement harmonique pose un motif VIVANT mais tres contracte ( les aires valent `lam^2` de
    // leur taille ) ; c'est cette boucle qui le redeploie. Mesure a `n = 1e5`, en diagrammes :
    //
    //      maxit  0 : 1063 ( 243 reculs )   -- PIRE qu'essai-limites ( 792 ) : le Newton global
    //      maxit  1 :  897 ( 180 )             qui suit ne rattrape pas le motif contracte
    //      maxit  2 :  622 (  72 )
    //      maxit  4 :  566 (  53 )   <-- le defaut
    //      maxit 40 :  574 (  55 )
    //
    // Au-dela de quatre on ne gagne plus rien -- et cette boucle est SEQUENTIELLE, donc son
    // plafond est aussi un plafond sur la partie non parallelisable du relevement.
    int           local_maxit = 4;
    TF            local_tol = TF( 1e-3 );
    TF            local_eps = TF( 0.05 );  ///< le plancher, en fraction de `nu`
    TF            local_pince = TF( 0.5 );  ///< « cette cellule pince sous le pas » : `a < local_pince * nu`
    int           local_cg = 2000;
    bool          newton_releve = false;   ///< la boucle de Newton AVEC relevement local ( § 15 )
    bool          releve_actif = true;     ///< ... ou sans, pour le temoin a boucle identique
    TF            releve_ratio = TF( 0.03 );  ///< le plafond de cellules malades, en fraction de `n`
    TF            releve_mort = TF( 1e-3 );  ///< sous `releve_mort * nu`, la cellule est hors de portee
    // DEUX ANNEAUX, MESURE SUR LES DEUX TAILLES ( diagrammes ) : 1 -> 174 et 316, 2 -> 123 et 257,
    // 3 -> 177 et 322. L'optimum est franc, et il ne bouge pas avec `n`.
    int           releve_anneaux = 2;
    bool          local_cont = true;       ///< la reparation par CONTINUATION sur le bord ( § 16 )
    bool          local_tangente = true;   ///< le predicteur : la tangente du sous-probleme, ou `d`
    bool          local_amas = false;      ///< un probleme PAR AMAS, en force brute, sans arbre ( § 16 )
    int           local_coupeurs = 0;      ///< couches de COUPEURS en plus autour de la couronne
    TF            local_bord = 10;         ///< penalisation qui tient l'AIRE de la couronne ( 0 : eteinte )
    // L'OBJECTIF EST L'EQUILIBRAGE, PAS LA CIBLE. La barriere `g( x ) = x - 1/x` FORCE l'aire vers
    // `nu`, donc impose un budget de masse que la zone ne peut pas tenir : on s'y bat contre une
    // contrainte dont on n'a aucun besoin, puisqu'on ne cherche pas la solution mais un etat non
    // degenere. On minimise donc l'ecart de `a / nu` D'UN VOISIN A L'AUTRE, qui n'a aucune cible
    // absolue -- et un zero au milieu d'un champ lisse est impossible. La couronne, elle, est tenue
    // a l'aire qu'elle a au pas de Newton pur, pour que l'exterieur ne bouge pas.
    bool          local_grad = true;       ///< l'objectif : le GRADIENT d'aire, au lieu de la barriere
    int           local_refresh = 1;       ///< refaire la structure de voisinage tous les K sous-pas ( 0 : jamais )
    bool          local_patch = false;     ///< le rafraichissement se fait SUR LES PATCHS, sans diagramme global
    // 128 EST MESURE, PAS CHOISI. n=5e3 : sans plafond 59 it / 223 diag / 3.6 M cellules ; a 128,
    // 60 / 238 / 0.54 M -- meme resultat pour TROIS FOIS MOINS de travail, les abandons tombant de 23
    // a 3. En dessous, le plafond mord et il faut payer en iterations : 71 / 303 a 64, 83 / 389 a 32.
    TF            releve_ratio_eps = 0;    ///< « malade » sur le RATIO `a / nu` au lieu du plancher absolu ( 0 : off )
    // LA RAIDEUR EST ALLUMEE PAR DEFAUT, ET `R` N'EST PAS UN VRAI REGLAGE. Mesure a `n = 2e4` :
    // 257 diagrammes contre 331 sans elle et 363 pour `essai-limites`, les reculs passant de 65 a 26.
    // Mais `R` = 1, 2, 4 et 8 donnent le MEME resultat au diagramme pres : la borne ne franchit pas
    // le plafond progressivement, elle SAUTE -- et ce saut est la fusion de deux amas, qui fait bondir
    // l'etalement d'un coup. Le critere revient donc en pratique a « s'arreter a la premiere grosse
    // fusion ». C'est defendable, mais ce n'est pas la lecture fine qu'on pourrait croire.
    TF            releve_raideur = 2;       ///< plafond sur l'etalement des poids d'un amas, en unites de dist^2 ( 0 : off )
    TF            releve_beta = 0.25;      ///< L'HORIZON : on ne raffine `U` que la-dessous ( 0 : jamais )
    TF            releve_bis_tol = 3e-2;   ///< precision relative de la bissection ( pas besoin de mieux )
    int           demo_homo = 0;           ///< table lam -> ( vides GEOMETRIQUES / masses nulles ) sur N amas
    bool          placement_homo = false;  ///< placer l'interieur par HOMOTHETIE centree sur `x*`
    bool          u_max = false;           ///< `U` par le flux MAXIMAL au lieu de la somme des flux
    bool          releve_u = true;         ///< choisir `F` par le predicteur `U` au lieu de la dyadique
    SI            releve_mal_max = 8;      ///< le plus gros amas de MALADES tolere, en cellules
    bool          diag_u = false;          ///< comparer le predicteur `U` au pas reellement accepte
    // LE PLAFOND DE TAILLE D'AMAS EST DESORMAIS ETEINT, ET C'EST UN RENVERSEMENT ASSUME. Il avait ete
    // mesure utile ( § 15.12 ) quand les gros amas coutaient 2.4 M cellules pour rien ; depuis, le
    // critere de RAIDEUR choisit `F` de sorte que ces amas ne se forment plus, et le plafond ne fait
    // que decliner des amas qui auraient reussi. Mesure sur le code actuel, en diagrammes :
    //
    //      n = 5e3 :  K=32 202,  K=64 165,  K=128 123,  K=256 114,  K=0 114
    //      n = 2e4 :  K=32 431,  K=64 317,  K=128 257,  K=256 227,  K=0 227
    //
    // `K = 256` egale `K = 0` : plus aucun amas n'atteint cette taille. Le plafond etait la bonne
    // reponse a un probleme qui n'existe plus.
    SI            local_amas_max = 0;      ///< au-dela, on DECLINE au lieu de chercher ( 0 : sans plafond )
    int           local_scan = 0;          ///< imprimer le balayage de `beta` pour les N premiers abandons
    int           local_dump = 0;          ///< dessiner les N premiers amas ABANDONNES ( SVG )
    TF            local_marge = 0;         ///< arret anticipe sur `plancher > marge * local_eps` ( 0 : eteint )
    // MESURE, ET POURQUOI C'EST ETEINT PAR DEFAUT. L'idee etait bonne en principe -- on cherche un
    // etat non degenere, pas le minimum de la barriere -- et elle coute effectivement moins cher
    // ( 1.80 M cellules au lieu de 3.75 M, 9.9 s au lieu de 11.8 ). Mais elle rend un relevement
    // moins bon, et le Newton le paie plus cher qu'il n'economise : 103 iterations et 549
    // diagrammes, contre 95 et 478. S'arreter « des que c'est sain » laisse un etat tout juste sain,
    // qui ne survit pas au sous-pas suivant -- le nombre de reparations acceptees tombe de 22 a 15
    // et celui des refus monte de 96 a 119. On garde donc l'option, eteinte.
    bool          local_trace = false;
    std::vector<TF> local_F{ 0.25, 0.5, 1 };   ///< les coefficients de relaxation imposes au bord
    std::string   rel_csv, rel_cas = "cas";
    bool          chooseur = false;    ///< choisir le pas sur la correction de poids de l'etape ecoulee
    TF            amp_cible = 800;     ///< chooseur : la correction visee, en unites de `h^2`
    TF            lam_min = 0;         ///< chooseur : sous cette valeur on vise directement la cible
                                       ///< ( 0 : `1e-6`, ou un tiers de pixel sur le chemin conv )
    int           max_etapes = 200;
    std::vector<TF> liste;             ///< les valeurs de `t`, explicites
    int           ordre = 0;           ///< 0 : les poids precedents ; 1 : la tangente en `t`
    std::string   diracs = "uniforme"; ///< uniforme | rho ( tires selon l'image )
    std::string   acc = "double";      ///< le flottant de la MESURE
    bool          check = false, chrono = false;
    std::string   ecrire;
};

// =====================================================================================
// LE TEMOIN. Il ne doit RIEN partager avec `Image.h`, sinon il ne temoigne de rien : la mesure
// par le bord ne coupe jamais, le temoin ne fait que couper.
// =====================================================================================

constexpr int NMX = 600;                                 ///< `MaxNv` plus la marge des coupes

int borne( int i, int n ) { return i < 0 ? 0 : ( i >= n ? n - 1 : i ); }

/// Sutherland-Hodgman contre `a x + b y <= c`. Le polygone est convexe : au plus un sommet de plus.
int coupe_demi( const double *ix, const double *iy, int n, double *ox, double *oy, double a, double b, double c ) {
    int m = 0;
    for ( int i = 0, j = n - 1; i < n; j = i++ ) {
        const double sj = a * ix[ j ] + b * iy[ j ] - c, si = a * ix[ i ] + b * iy[ i ] - c;
        if ( ( sj <= 0 ) != ( si <= 0 ) ) {
            const double t = sj / ( sj - si );
            ox[ m ] = ix[ j ] + t * ( ix[ i ] - ix[ j ] );
            oy[ m ] = iy[ j ] + t * ( iy[ i ] - iy[ j ] );
            ++m;
        }
        if ( si <= 0 ) { ox[ m ] = ix[ i ]; oy[ m ] = iy[ i ]; ++m; }
    }
    return m;
}

double aire_pol( const double *x, const double *y, int n ) {
    double a = 0;
    for ( int i = 0, j = n - 1; i < n; j = i++ ) a += x[ j ] * y[ i ] - x[ i ] * y[ j ];
    return 0.5 * std::fabs( a );
}

/// LA MASSE, PAR DECOUPAGE EN PIXELS : la boite englobante, quatre demi-plans par pixel, et
/// `rho x aire`. Le cout est en pixels DANS la cellule -- c'est tout le sujet.
template<class Im, class Cel>
double masse_pixels( const Im &im, const Cel &cel ) {
    const int nb = cel.nb;
    if ( nb < 3 ) return 0;
    double px[ NMX ], py[ NMX ];
    for ( int q = 0; q < nb; ++q ) { px[ q ] = cel.x( q ); py[ q ] = cel.y( q ); }
    double x0 = px[ 0 ], x1 = px[ 0 ], y0 = py[ 0 ], y1 = py[ 0 ];
    for ( int i = 1; i < nb; ++i ) {
        x0 = std::min( x0, px[ i ] ); x1 = std::max( x1, px[ i ] );
        y0 = std::min( y0, py[ i ] ); y1 = std::max( y1, py[ i ] );
    }
    const double hx = 1.0 / im.W, hy = 1.0 / im.H;
    const int i0 = borne( int( x0 * im.W ), im.W ), i1 = borne( int( x1 * im.W ), im.W );
    const int j0 = borne( int( y0 * im.H ), im.H ), j1 = borne( int( y1 * im.H ), im.H );
    double m = 0;
    double ax[ NMX ], ay[ NMX ], bx[ NMX ], by[ NMX ];
    for ( int j = j0; j <= j1; ++j )
        for ( int i = i0; i <= i1; ++i ) {
            const double r = double( im.v[ size_t( j ) * im.W + i ] );
            int n = nb;
            for ( int q = 0; q < nb; ++q ) { ax[ q ] = px[ q ]; ay[ q ] = py[ q ]; }
            n = coupe_demi( ax, ay, n, bx, by,  1,  0,  ( i + 1 ) * hx );
            n = coupe_demi( bx, by, n, ax, ay, -1,  0, -i * hx );
            n = coupe_demi( ax, ay, n, bx, by,  0,  1,  ( j + 1 ) * hy );
            n = coupe_demi( bx, by, n, ax, ay,  0, -1, -j * hy );
            if ( n >= 3 ) m += r * aire_pol( ax, ay, n );
        }
    return m;
}

/// `int_arete rho ds` -- le coefficient de hessienne. Temoin : on LISTE les traversees de lignes
/// de grille, on les TRIE, et on evalue `rho` au milieu de chaque morceau. Pas un increment, pas
/// une marche : rien du parcours de `Image.h`.
template<class Im>
double long_ponderee( const Im &im, double x0, double y0, double x1, double y1 ) {
    const double dx = x1 - x0, dy = y1 - y0;
    const double lg = std::sqrt( dx * dx + dy * dy );
    if ( lg == 0 ) return 0;
    const double hx = 1.0 / im.W, hy = 1.0 / im.H;
    std::vector<double> t{ 0, 1 };
    if ( dx != 0 ) {
        const int a = borne( int( std::min( x0, x1 ) * im.W ), im.W ), b = borne( int( std::max( x0, x1 ) * im.W ), im.W );
        for ( int i = a; i <= b + 1; ++i ) { const double u = ( i * hx - x0 ) / dx; if ( u > 0 && u < 1 ) t.push_back( u ); }
    }
    if ( dy != 0 ) {
        const int a = borne( int( std::min( y0, y1 ) * im.H ), im.H ), b = borne( int( std::max( y0, y1 ) * im.H ), im.H );
        for ( int j = a; j <= b + 1; ++j ) { const double u = ( j * hy - y0 ) / dy; if ( u > 0 && u < 1 ) t.push_back( u ); }
    }
    std::sort( t.begin(), t.end() );
    double s = 0;
    for ( size_t q = 1; q < t.size(); ++q ) {
        const double u = 0.5 * ( t[ q - 1 ] + t[ q ] );
        const int i = borne( int( ( x0 + dx * u ) * im.W ), im.W ), j = borne( int( ( y0 + dy * u ) * im.H ), im.H );
        s += double( im.v[ size_t( j ) * im.W + i ] ) * ( t[ q ] - t[ q - 1 ] ) * lg;
    }
    return s;
}

// =====================================================================================

/// les quantiles d'une liste d'ecarts, triee sur place
void quantiles( std::vector<double> &e, const char *quoi ) {
    if ( e.empty() ) { std::printf( "    %-34s : rien\n", quoi ); return; }
    std::sort( e.begin(), e.end() );
    auto q = [ & ]( double f ) { return e[ std::min( e.size() - 1, size_t( f * e.size() ) ) ]; };
    std::printf( "    %-34s : median %.2e  p99 %.2e  p99.99 %.2e  max %.2e\n", quoi, q( 0.5 ), q( 0.99 ), q( 0.9999 ), e.back() );
}

/// LA VERIFICATION, cellule par cellule sur TOUT le diagramme.
///   1. la masse contre le decoupage en pixels ;
///   2. `int_facette rho ds` contre le temoin par traversees triees ;
///   3. `d m_i / d w_i` contre `somme_j c_ij` -- la difference finie CENTREE sur la masse de la
///      cellule, le seul controle qui ne suppose ni le signe ni le facteur de la hessienne. On ne
///      bouge QUE le poids de `i` ( `cellule_avec_poids` ), donc la derivee attendue est la somme
///      des coefficients des facettes interieures, et rien d'autre.
///   4. la somme des masses contre `1`.
template<class PD, class Im>
void verifie( const PD &pd, const Nuage<2> &nu, const Im &im ) {
    const SI n = pd.n;
    std::vector<double> em, ef, ed;
    double somme = 0, mmin = 1e300, mmax = 0;
    SI nvide = 0, nnul = 0;
    typename PD::Cell cel, cp, cm;
    std::vector<TF> fv;
    const TF h2 = TF( 1 ) / n;
    for ( SI k = 0; k < n; ++k ) {
        if ( ! pd.cellule( k, cel ) ) continue;
        if ( cel.nb <= 0 ) { ++nvide; continue; }
        const SI id = pd.ids[ k ];
        double sc = 0;                                   // somme_j c_ij : la derivee attendue
        fv.clear();
        const TF m = im.mesure( cel, [ & ]( int j, TF f ) {
            fv.push_back( f );
            const double dx = double( nu.c[ 0 ][ j ] ) - double( nu.c[ 0 ][ id ] );
            const double dy = double( nu.c[ 1 ][ j ] ) - double( nu.c[ 1 ][ id ] );
            sc += double( f ) / ( 2 * std::sqrt( dx * dx + dy * dy ) );
        } );
        somme += double( m );
        mmin = std::min( mmin, double( m ) );
        mmax = std::max( mmax, double( m ) );

        // 1. la masse contre le decoupage en pixels. L'ecart est rapporte a LA MASSE MOYENNE et
        //    non a la masse de la cellule : sous une image a zeros, des cellules entieres ont une
        //    masse nulle, et un ecart relatif y n'a aucun sens ( c'est la moyenne qui fixe
        //    l'echelle de ce que Newton voit ).
        const double mp = masse_pixels( im, cel );
        em.push_back( std::fabs( double( m ) - mp ) );
        nnul += ! ( double( m ) > 0 );

        // 2. `int_facette rho ds` contre les traversees triees ( un echantillon : le temoin est cher )
        if ( k % 37 == 0 ) {
            size_t q = 0;
            const double tt = double( im.t );
            for ( int i = 0, j = cel.nb - 1; i < cel.nb; j = i++ ) {
                if ( cel.cid[ j ] < 0 ) continue;
                const double x0 = cel.x( j ), y0 = cel.y( j );
                const double x1 = cel.x( i ), y1 = cel.y( i );
                const double L = std::hypot( x1 - x0, y1 - y0 );
                const double att = ( 1 - tt ) * L + tt * long_ponderee( im, x0, y0, x1, y1 );
                if ( q < fv.size() && att > 0 ) ef.push_back( std::fabs( double( fv[ q ] ) - att ) / att );
                ++q;
            }
        }

        // 3. `d m_i / d w_i` contre `somme_j c_ij`, par difference finie CENTREE
        const TF hw = TF( 1e-5 ) * h2;
        if ( sc > 0 && pd.cellule_avec_poids( k, TF( pd.w[ k ] ) + hw, cp ) && pd.cellule_avec_poids( k, TF( pd.w[ k ] ) - hw, cm )
             && cp.nb > 0 && cm.nb > 0 ) {
            const TF mp2 = im.mesure( cp, []( int, TF ) {} ), mm2 = im.mesure( cm, []( int, TF ) {} );
            ed.push_back( std::fabs( double( mp2 - mm2 ) / ( 2 * hw ) - sc ) / sc );
        }
    }
    const double moy = somme / n;
    for ( double &e : em ) e /= moy;
    std::printf( "  %d cellules ( %d debordees ou vides, %d de masse NULLE OU NEGATIVE ), masse par cellule : min %.3e  max %.3e  moyenne %.3e\n",
                 int( n ), int( nvide ), int( nnul ), mmin, mmax, moy );
    quantiles( em, "masse / decoupage ( / moyenne )" );
    quantiles( ef, "facette / traversees triees" );
    quantiles( ed, "d m / d w  /  somme_j c_ij  ( d.f. )" );
    std::printf( "    %-34s : %.12f ( ecart a 1 : %.2e )\n", "somme des masses", somme, std::fabs( somme - 1 ) );
}

} // namespace

// =====================================================================================

namespace {

/// LE CHRONO A MACHINE EGALE : les memes cellules du meme moteur, mesurees deux fois -- par le
/// bord et par le decoupage. Ce qui separe l'ALGORITHME de la MACHINE.
template<class PD, class Im>
void chrono( PD &pd, const Im &im, const Parallel &par, int reps ) {
    const SI n = pd.n;
    std::vector<TF> res( n );
    double t_bord = 1e300, t_pix = 1e300, t_leb = 1e300;
    for ( int r = 0; r < reps; ++r ) {
        double t0 = now();
        pd.measures_and_facets_avec( res, par, []( int, d2::SI32, d2::SI32, TF ) {},
                                     [ & ]( const typename PD::Cell &cel, auto &&fac, SI ) { return im.mesure( cel, fac ); } );
        t_bord = std::min( t_bord, now() - t0 );
        t0 = now();
        pd.measures_and_facets( res, par, []( int, d2::SI32, d2::SI32, TF ) {} );
        t_leb = std::min( t_leb, now() - t0 );
        t0 = now();
        pd.measures_and_facets_avec( res, par, []( int, d2::SI32, d2::SI32, TF ) {},
                                     [ & ]( const typename PD::Cell &cel, auto &&fac, SI ) {
                                         TF s = 0;                       // les facettes du decoupage : le temoin
                                         for ( int i = 0, j = cel.nb - 1; i < cel.nb; j = i++ )
                                             if ( cel.cid[ j ] >= 0 )
                                                 fac( cel.cid[ j ], TF( long_ponderee( im, cel.x( j ), cel.y( j ),
                                                                                           cel.x( i ), cel.y( i ) ) ) );
                                         s = TF( masse_pixels( im, cel ) );
                                         return s;
                                     } );
        t_pix = std::min( t_pix, now() - t0 );
    }
    std::printf( "  %8d germes, image %5d x %-5d ( W / sqrt( n ) = %5.1f ) : Lebesgue %.3f s | BORD %.3f s | decoupage %.3f s  --  x%.1f\n",
                 int( n ), im.W, im.H, im.W / std::sqrt( double( n ) ), t_leb, t_bord, t_pix, t_pix / std::max( t_bord, 1e-12 ) );
}

// =====================================================================================
// LE RELEVEMENT PAR MOINDRES CARRES PONDERES : peut-on prendre un plus grand pas en forcant
// certaines cellules ?
//
// L'idee a l'essai : au lieu de `L d = nu - a`, resoudre
//
//      min_d  1/2 sum_i C_i ( a_i + ( L d )_i - nu_i )^2
//
// avec des `C_i` GRANDS sur les cellules qui se vident et autour d'elles ( une cloche ), pour que
// la direction s'occupe d'abord d'elles.
//
// = CE QU'IL FAUT VOIR AVANT DE MESURER : LES POIDS SONT INVISIBLES
//
// `L` est CARREE et inversible sur les moyennes nulles ( son noyau est exactement les constantes ).
// Le residu linearise `r + L d` peut donc etre annule EXACTEMENT, et le minimum du probleme
// pondere vaut ZERO -- atteint en `L d = nu - a`, c'est-a-dire la direction de Newton, **quels que
// soient les `C_i` positifs**. Ponderer ne change donc rigoureusement rien. ( Mesure : avec un CG
// mene a convergence, `kappa = 1` et `kappa = 1000` rendent le meme pas admissible, la meme masse
// minimale et le meme residu que Newton, au chiffre pres -- les directions ne different que par
// la constante de jauge. )
//
// Et ce n'est pas un accident de la linearisation : Gauss-Newton sur
// `sum_i C_i ( a_i( w ) - nu_i )^2` refait ce calcul a chaque iteration, donc redonne le pas de
// Newton a chaque iteration. Les poids n'apparaissent nulle part.
//
// = LES DEUX FACONS DE LEUR RENDRE UN SENS
//
//   1. PONDERER LE MERITE, pas la direction. `sum C_i r_i^2` change quels pas sont ACCEPTES par
//      l'amortissement, meme si la direction ne bouge pas. C'est un autre levier, pas celui-ci.
//   2. RESTREINDRE `d` A UN MORCEAU du diagramme. Si `d` est nul hors d'un patch `P`, le systeme
//      devient SUR-DETERMINE ( `|P|` inconnues, les lignes de `P` et de ses voisins ), le residu
//      ne peut plus etre annule, et les `C_i` decident **ce qu'on sacrifie**. C'est la version
//      « locale », et ce n'est pas une variante economique de la premiere : c'est la SEULE ou
//      l'idee a un contenu.
//
// Une derniere borne a garder en tete : une cellule de masse nulle a toutes ses `c_ij` nulles,
// `Laplacien::assemble` neutralise sa ligne ( `dia = 1` ), et alors `d_i = nu_i` quels que soient
// les poids et le patch -- une correction de l'ordre de `h^2` la ou il en faudrait une de l'ordre
// du domaine. Aucun systeme local ne releve une cellule DEJA morte ; seules les cellules PINCEES
// VIVANTES sont en jeu.
// =====================================================================================

/// `y = L x`
template<class L_>
void applique_L( const L_ &L, const std::vector<TF> &x, std::vector<TF> &y ) {
    const SI n = L.n;
    y.resize( n );
    for ( SI i = 0; i < n; ++i ) {
        TF s = L.dia[ i ] * x[ i ];
        for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) s -= L.c[ e ] * x[ L.col[ e ] ];
        y[ i ] = s;
    }
}

inline void moyenne_nulle( std::vector<TF> &x ) {
    TF s = 0;
    for ( TF v : x ) s += v;
    s /= TF( x.size() );
    for ( TF &v : x ) v -= s;
}

/// LES MOINDRES CARRES PONDERES, `d` NUL HORS DU PATCH : `( L_P )^T C ( L_P ) d = ( L_P )^T C b`,
/// par CG sans matrice. `L_P x` n'est autre que `L` applique au `x` complete par des zeros, et
/// `( L_P )^T y` la restriction de `L y` au patch -- `L` est symetrique, il n'y a rien a assembler
/// et le remplissage en voisins-de-voisins n'existe jamais.
///
/// `dans_p` vide : le patch est TOUT le diagramme, et on projette sur les moyennes nulles ( le
/// noyau des constantes ). Ce cas est le temoin : il doit rendre la direction de Newton.
int resout_pondere( const Laplacien &L, const std::vector<TF> &C, const std::vector<TF> &b,
                    const std::vector<char> &dans_p, std::vector<TF> &d, int maxit, TF tol,
                    TF mu_rel = 0 ) {
    const SI n = L.n;
    const bool global = dans_p.empty();
    std::vector<TF> rhs( n ), tmp( n ), r( n ), z( n ), p( n ), Ap( n ), dia( n );
    auto restreint = [ & ]( std::vector<TF> &x ) {
        if ( global ) { moyenne_nulle( x ); return; }
        for ( SI i = 0; i < n; ++i ) if ( ! dans_p[ i ] ) x[ i ] = 0;
    };
    for ( SI i = 0; i < n; ++i ) tmp[ i ] = C[ i ] * b[ i ];
    applique_L( L, tmp, rhs );
    restreint( rhs );
    TF moy = 0;
    for ( SI i = 0; i < n; ++i ) {                       // `diag( A^T C A )_j = sum_i L_ij^2 C_i`
        TF s = L.dia[ i ] * L.dia[ i ] * C[ i ];
        for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) s += L.c[ e ] * L.c[ e ] * C[ L.col[ e ] ];
        dia[ i ] = s > 0 ? s : TF( 1 );
        moy += dia[ i ];
    }
    // L'AMORTISSEMENT ( Levenberg-Marquardt ) : `( A^T C A + mu I ) d = A^T C b`. C'est LUI qui rend
    // les poids visibles sans restreindre `d` -- a `mu = 0` le residu s'annule exactement et `C`
    // se simplifie ( voir l'en-tete ) ; des que `mu > 0` le systeme n'est plus celui de Newton, et
    // quand `mu` domine, `d -> ( 1 / mu ) L C ( nu - a )` : la poussee pondere cellule par cellule.
    // `mu` est donne EN FRACTION de la diagonale moyenne, faute de quoi il n'a pas d'echelle.
    const TF mu = mu_rel * moy / TF( n );
    for ( SI i = 0; i < n; ++i ) dia[ i ] += mu;
    auto applique_A = [ & ]( const std::vector<TF> &x, std::vector<TF> &y ) {
        applique_L( L, x, tmp );
        for ( SI i = 0; i < n; ++i ) tmp[ i ] *= C[ i ];
        applique_L( L, tmp, y );
        if ( mu > 0 ) for ( SI i = 0; i < n; ++i ) y[ i ] += mu * x[ i ];
        restreint( y );
    };
    d.assign( n, TF( 0 ) );
    r = rhs;
    TF n0 = 0;
    for ( TF v : r ) n0 += v * v;
    n0 = std::sqrt( n0 );
    if ( ! ( n0 > 0 ) ) return 0;
    TF rz = 0;
    for ( SI i = 0; i < n; ++i ) { z[ i ] = r[ i ] / dia[ i ]; rz += r[ i ] * z[ i ]; }
    restreint( z );
    p = z;
    for ( int it = 0; it < maxit; ++it ) {
        applique_A( p, Ap );
        TF pAp = 0;
        for ( SI i = 0; i < n; ++i ) pAp += p[ i ] * Ap[ i ];
        if ( ! ( pAp > 0 ) ) return -it;
        const TF al = rz / pAp;
        TF nr = 0;
        for ( SI i = 0; i < n; ++i ) { d[ i ] += al * p[ i ]; r[ i ] -= al * Ap[ i ]; nr += r[ i ] * r[ i ]; }
        nr = std::sqrt( nr );
        if ( nr <= tol * n0 ) return it + 1;
        TF rz2 = 0;
        for ( SI i = 0; i < n; ++i ) { z[ i ] = r[ i ] / dia[ i ]; rz2 += r[ i ] * z[ i ]; }
        restreint( z );
        const TF be = rz2 / rz;
        rz = rz2;
        for ( SI i = 0; i < n; ++i ) p[ i ] = z[ i ] + be * p[ i ];
    }
    return -maxit;
}

/// LE BALAYAGE D'UNE DIRECTION : `w + t d` sur une grille geometrique, et ce que chaque pas donne.
/// `alpha*` est le plus grand `t` essaye qui garde toutes les masses au-dessus de `eps` -- le
/// « coefficient de relaxation » que l'amortissement pourrait prendre.
struct Bilan {
    std::string nom;
    TF   amp = 0;            ///< `|d|inf / h^2`
    TF   alpha = 0;          ///< le plus grand pas admissible trouve
    TF   mmin = 0;           ///< la plus petite masse a `alpha`, en unites de `nu`
    TF   mmin_p = 0;         ///< la plus petite masse DES CELLULES VISEES, a `alpha`
    TF   r2 = 0, rmax = 0;   ///< le merite et `max |a - nu| / nu` a `alpha`
    SI   sous = 0;           ///< cellules sous `eps` a `alpha`
    int  cg = 0;             ///< iterations du CG ( 0 : resolution directe )
    SI   taille = 0;         ///< inconnues du systeme
};

template<class PD, class Rho>
Bilan balaye( Newton<PD,Rho> &nw, const std::vector<TF> &w0, const std::vector<TF> &d,
              const std::vector<SI> &vise, TF eps, const std::string &nom, int cg, SI taille,
              FILE *csv, const char *cas, std::vector<TF> *w_alpha = nullptr ) {
    const SI n = SI( w0.size() );
    Bilan b;
    b.nom = nom;
    b.cg = cg;
    b.taille = taille;
    for ( SI i = 0; i < n; ++i ) b.amp = std::max( b.amp, std::fabs( d[ i ] ) );
    b.amp *= TF( n );
    std::vector<TF> w( n ), a;
    std::vector<Facette> fa;
    for ( TF t = 1; t >= TF( 1e-5 ); t /= 2 ) {
        for ( SI i = 0; i < n; ++i ) w[ i ] = w0[ i ] + t * d[ i ];
        nw.mesures_et_facettes( w, a, fa );
        TF mmin = a[ 0 ], rmax = 0, r2 = 0, mmin_p = 1e300;
        SI sous = 0;
        for ( SI i = 0; i < n; ++i ) {
            mmin = std::min( mmin, a[ i ] );
            rmax = std::max( rmax, std::fabs( a[ i ] - nw.nu[ i ] ) / nw.nu[ i ] );
            r2 += ( a[ i ] - nw.nu[ i ] ) * ( a[ i ] - nw.nu[ i ] );
            sous += a[ i ] < eps;
        }
        for ( SI i : vise ) mmin_p = std::min( mmin_p, a[ i ] );
        r2 = std::sqrt( r2 );
        if ( csv ) std::fprintf( csv, "%s;%s;%.6e;%.6e;%.6e;%.6e;%.6e;%d\n", cas, nom.c_str(), double( t ),
                                 double( mmin / nw.nu[ 0 ] ), double( mmin_p / nw.nu[ 0 ] ), double( r2 ),
                                 double( rmax ), int( sous ) );
        if ( b.alpha == 0 && mmin >= eps ) {             // le premier `t` admissible en descendant
            b.alpha = t; b.mmin = mmin / nw.nu[ 0 ]; b.mmin_p = mmin_p / nw.nu[ 0 ];
            b.r2 = r2; b.rmax = rmax; b.sous = sous;
            if ( w_alpha ) *w_alpha = w;
        }
    }
    return b;
}

// =====================================================================================
// LA RESOLUTION : la continuation en contraste, et Newton a chaque etape.
// =====================================================================================

// =====================================================================================
// LE SOUS-PROBLEME LOCAL, A OBJECTIF BARRIERE ( `--local-nl` )
//
// Le schema, tel qu'il a ete propose :
//
//   * on prend le pas de Newton `w0 + F d` et on REPERE LES CELLULES QUI PINCENT LA -- pas celles
//     qui sont petites au depart : au depart tout va bien, c'est le PAS qui les tue ;
//   * on va chercher `N` anneaux plus loin autour d'elles ( parcours en largeur ; les composantes
//     qui se touchent fusionnent d'elles-memes ), ce qui donne une boule dont le bord est, PAR
//     CONSTRUCTION, fait de cellules saines -- et on le verifie, au depart comme au pas ;
//   * les poids du bord sont FIXES a `w0 + F d`, ceux de l'interieur sont cherches en minimisant
//
//         Phi = somme_( boule et anneau )  ( A_i / nu_i  -  nu_i / A_i )^2
//
//     L'ANNEAU EST DANS L'OBJECTIF, et c'est ce qui le protege : une cellule d'anneau qu'on fait
//     descendre de 1 a 0.9 ne coute presque rien ( `g = -0.21` ), la faire descendre a 0.1 coute
//     `g = -9.9`. La barriere arbitre d'elle-meme entre « relever la boule » et « ne pas ecraser
//     l'anneau », sans qu'on ait a poser de plancher -- et elle vaut `+infini` si une cellule meurt.
//
// Ce n'est PAS le systeme de Newton : `A_i = nu_i` sur toute la boule serait en general infaisable
// ( le budget de masse local est ferme ), alors que le minimum de `Phi` existe toujours. On ne
// cherche pas la solution, on cherche un POINT DE DEPART SANS ECRASEMENT.
//
// Gauss-Newton sur `Phi` : avec `x_i = A_i / nu_i`, `g( x ) = x - 1/x`, `g'( x ) = 1 + 1/x^2`,
// `r_i = g( x_i )` et `u_i = g'( x_i ) / nu_i`, la jacobienne est `J_ip = u_i L_ip` et le pas resout
//
//         ( L diag( u^2 ) L )_boule d = -( L ( u . r ) )_boule
//
// sans matrice ( `L` est symetrique : les deux produits se font sur la meme structure locale ), et
// le pas est amorti sur `Phi` lui-meme -- la barriere refuse toute seule de tuer une cellule.
//
// Le cout est en CELLULES CALCULEES : chaque iteration ne recalcule que la boule et son anneau.
// =====================================================================================

struct BilanLocal {
    TF   F = 0;
    int  N = 0;
    SI   nS = 0, taille = 0, n_ring = 0, nb_cel = 0;
    int  iter = 0;
    TF   ring_w0 = 0, ring_try = 0;   ///< la sante de l'anneau, au depart et sous le pas
    TF   ball_try = 0;                ///< la masse minimale de la boule sous le pas, avant reparation
    TF   ball_fin = 0, ring_fin = 0, glob_fin = 0;
    TF   phi0 = 0, phi1 = 0;
    TF   r2 = 0;
    int  diag_apres = -1;
    TF   s = 0;                       ///< la fraction du pas cible atteinte par la continuation
    SI   fuites = 0;                  ///< facettes de la boule vers hors de la zone surveillee
    SI   nb_amas = 0, amas_max = 0;   ///< composantes connexes de la boule, et la plus grosse
    std::string fin;
};

/// `g( x ) = x - 1/x` et sa derivee, bornees loin de zero pour que la trace reste lisible
inline TF bar_g( TF x )  { x = std::max( x, TF( 1e-9 ) ); return x - 1 / x; }
inline TF bar_gp( TF x ) { x = std::max( x, TF( 1e-9 ) ); return 1 + 1 / ( x * x ); }

/// LA REPARATION, depuis un etat DEJA MESURE ( `w_try`, `a_try`, `fa_try` ) : on repere les cellules
/// pincees, on prend `N` anneaux autour, on fixe le bord et on minimise `Phi` a l'interieur.
/// C'est le coeur reutilisable -- l'etude du § 14 et la boucle de Newton du § 15 l'appellent tous
/// les deux.
template<class PD, class Rho>
BilanLocal repare_local( Newton<PD,Rho> &nw, const std::vector<SI> &rang, const std::vector<TF> &w_try,
                         const std::vector<TF> &a_try, const std::vector<Facette> &fa_try, int N,
                         const Opts &o, std::vector<TF> &w_out ) {
    PD &pd = nw.pd;
    const SI n = pd.n;
    BilanLocal bi;
    bi.N = N;
    std::vector<SI> S;
    for ( SI i = 0; i < n; ++i ) if ( a_try[ i ] < o.local_pince * nw.nu[ i ] ) S.push_back( i );
    bi.nS = SI( S.size() );
    w_out = w_try;
    if ( S.empty() ) return bi;                          // le pas passe : rien a reparer

    // ---- 2. LA BOULE : `N` anneaux plus loin, sur la geometrie du PAS
    Laplacien Lt;
    Lt.assemble( n, fa_try );
    std::vector<int> dist( n, -1 );
    std::vector<SI> file( S.begin(), S.end() );
    for ( SI i : S ) dist[ i ] = 0;
    for ( size_t q = 0; q < file.size(); ++q ) {
        const SI i = file[ q ];
        if ( dist[ i ] >= N + 1 ) continue;
        for ( SI e = Lt.row[ i ]; e < Lt.row[ i + 1 ]; ++e ) {
            const SI j = Lt.col[ e ];
            if ( dist[ j ] < 0 ) { dist[ j ] = dist[ i ] + 1; file.push_back( j ); }
        }
    }
    std::vector<SI> ens;                                 // la boule PUIS l'anneau
    std::vector<SI> loc( n, -1 );
    for ( SI i = 0; i < n; ++i ) if ( dist[ i ] >= 0 && dist[ i ] <= N ) { loc[ i ] = SI( ens.size() ); ens.push_back( i ); }
    const SI m = SI( ens.size() );
    for ( SI i = 0; i < n; ++i ) if ( dist[ i ] == N + 1 ) { loc[ i ] = SI( ens.size() ); ens.push_back( i ); }
    const SI M = SI( ens.size() );
    bi.taille = m;
    bi.n_ring = M - m;
    // LA SANTE DU BORD, qu'on annonce « par construction » : on la verifie.
    bi.ring_w0 = 1e300; bi.ring_try = 1e300; bi.ball_try = 1e300;
    for ( SI q = m; q < M; ++q )
        bi.ring_try = std::min( bi.ring_try, a_try[ ens[ q ] ] / nw.nu[ ens[ q ] ] );
    for ( SI q = 0; q < m; ++q ) bi.ball_try = std::min( bi.ball_try, a_try[ ens[ q ] ] / nw.nu[ ens[ q ] ] );
    if ( M == m ) bi.ring_try = 1;

    // ---- 3. LE SOLVEUR LOCAL
    std::vector<TF> w = w_try, a( M ), dia( M ), u( M ), r( M );
    std::vector<SI> row( M + 1 ), col;
    std::vector<TF> cc;
    auto mesure_locale = [ & ]( bool avec_facettes ) {
        pd.set_weights( w.data(), nw.par );
        if ( avec_facettes ) { row.assign( M + 1, 0 ); col.clear(); cc.clear(); dia.assign( M, TF( 0 ) ); }
        typename PD::Cell cel;
        for ( SI q = 0; q < M; ++q ) {
            const SI id = ens[ q ];
            a[ q ] = 0;
            if ( ! pd.cellule( rang[ id ], cel ) ) continue;
            a[ q ] = nw.rho->mesure( cel, [ & ]( int j, TF mes ) {
                if ( ! avec_facettes ) return;
                TF d2 = 0;
                for ( int d = 0; d < PD::dim; ++d ) { const TF e = nw.P[ d ][ j ] - nw.P[ d ][ id ]; d2 += e * e; }
                if ( ! ( d2 > 0 ) ) return;
                const TF cij = mes / ( 2 * std::sqrt( d2 ) );
                dia[ q ] += cij;                         // TOUS les voisins : c'est `d a_i / d w_i`
                if ( loc[ j ] >= 0 ) { col.push_back( loc[ j ] ); cc.push_back( cij ); ++row[ q + 1 ]; }
            } );
            ++bi.nb_cel;
        }
        if ( avec_facettes ) {
            for ( SI q = 0; q < M; ++q ) row[ q + 1 ] += row[ q ];
            for ( SI q = 0; q < M; ++q ) if ( ! ( dia[ q ] > 0 ) ) dia[ q ] = 1;
        }
    };
    auto phi = [ & ]() {                                 // l'objectif, sur la boule ET l'anneau
        TF s = 0;
        for ( SI q = 0; q < M; ++q ) {
            const TF g = bar_g( a[ q ] / nw.nu[ ens[ q ] ] );
            s += g * g;
        }
        return s;
    };
    // `y ( sur M ) = L x`, `x` porte par la BOULE seulement
    auto Lx = [ & ]( const std::vector<TF> &x, std::vector<TF> &y ) {
        for ( SI q = 0; q < M; ++q ) {
            TF s = q < m ? dia[ q ] * x[ q ] : TF( 0 );
            for ( SI e = row[ q ]; e < row[ q + 1 ]; ++e ) if ( col[ e ] < m ) s -= cc[ e ] * x[ col[ e ] ];
            y[ q ] = s;
        }
    };
    // `y ( sur la boule ) = L^T z`, `z` porte par la boule ET l'anneau
    auto Ltz = [ & ]( const std::vector<TF> &z, std::vector<TF> &y ) {
        for ( SI p = 0; p < m; ++p ) {
            TF s = dia[ p ] * z[ p ];
            for ( SI e = row[ p ]; e < row[ p + 1 ]; ++e ) s -= cc[ e ] * z[ col[ e ] ];
            y[ p ] = s;
        }
    };

    mesure_locale( true );
    bi.phi0 = phi();
    TF ph = bi.phi0;
    std::vector<TF> t1( M ), t2( M ), rhs( m ), d( m ), pcg( m ), Ap( m ), zz( m ), rr( m ), jac( m ), w_sauve;
    for ( int it = 0; it < o.local_maxit; ++it ) {
        for ( SI q = 0; q < M; ++q ) {
            const TF x = a[ q ] / nw.nu[ ens[ q ] ];
            r[ q ] = bar_g( x );
            u[ q ] = bar_gp( x ) / nw.nu[ ens[ q ] ];
        }
        for ( SI q = 0; q < M; ++q ) t1[ q ] = u[ q ] * r[ q ];
        Ltz( t1, rhs );
        for ( SI p = 0; p < m; ++p ) rhs[ p ] = -rhs[ p ];
        TF nr0 = 0;
        for ( TF v : rhs ) nr0 += v * v;
        if ( ! ( nr0 > 0 ) ) break;
        for ( SI p = 0; p < m; ++p ) {                   // Jacobi : `diag( L u^2 L )_p = sum_q u_q^2 L_qp^2`
            TF s = u[ p ] * u[ p ] * dia[ p ] * dia[ p ];
            for ( SI e = row[ p ]; e < row[ p + 1 ]; ++e ) s += u[ col[ e ] ] * u[ col[ e ] ] * cc[ e ] * cc[ e ];
            jac[ p ] = s > 0 ? s : TF( 1 );
        }
        auto A = [ & ]( const std::vector<TF> &x, std::vector<TF> &y ) {
            Lx( x, t1 );
            for ( SI q = 0; q < M; ++q ) t2[ q ] = u[ q ] * u[ q ] * t1[ q ];
            Ltz( t2, y );
        };
        d.assign( m, TF( 0 ) );                          // CG preconditionne
        rr = rhs;
        TF rz = 0;
        for ( SI p = 0; p < m; ++p ) { zz[ p ] = rr[ p ] / jac[ p ]; rz += rr[ p ] * zz[ p ]; }
        pcg = zz;
        for ( int k = 0; k < o.local_cg; ++k ) {
            A( pcg, Ap );
            TF pAp = 0;
            for ( SI p = 0; p < m; ++p ) pAp += pcg[ p ] * Ap[ p ];
            if ( ! ( pAp > 0 ) ) break;
            const TF al = rz / pAp;
            TF nr = 0;
            for ( SI p = 0; p < m; ++p ) { d[ p ] += al * pcg[ p ]; rr[ p ] -= al * Ap[ p ]; nr += rr[ p ] * rr[ p ]; }
            if ( nr <= TF( 1e-16 ) * nr0 ) break;
            TF rz2 = 0;
            for ( SI p = 0; p < m; ++p ) { zz[ p ] = rr[ p ] / jac[ p ]; rz2 += rr[ p ] * zz[ p ]; }
            const TF be = rz2 / rz;
            rz = rz2;
            for ( SI p = 0; p < m; ++p ) pcg[ p ] = zz[ p ] + be * pcg[ p ];
        }
        // le pas amorti SUR L'OBJECTIF : la barriere refuse d'elle-meme de tuer une cellule
        w_sauve = w;
        bool pris = false;
        for ( TF t = 1; t >= TF( 1e-8 ); t /= 2 ) {
            for ( SI p = 0; p < m; ++p ) w[ ens[ p ] ] = w_sauve[ ens[ p ] ] + t * d[ p ];
            mesure_locale( false );
            const TF p2 = phi();
            if ( p2 < ph * ( 1 - 1e-12 ) ) { ph = p2; pris = true; break; }
        }
        ++bi.iter;
        if ( ! pris ) { w = w_sauve; mesure_locale( true ); break; }
        mesure_locale( true );
        if ( o.local_trace )
            std::printf( "      local N=%d : it %2d  Phi %.4e  min boule %.3e  min anneau %.3e\n",
                         N, it, double( ph ),
                         double( [ & ]{ TF v = 1e300; for ( SI q = 0; q < m; ++q ) v = std::min( v, a[ q ] / nw.nu[ ens[ q ] ] ); return v; }() ),
                         double( [ & ]{ TF v = 1e300; for ( SI q = m; q < M; ++q ) v = std::min( v, a[ q ] / nw.nu[ ens[ q ] ] ); return M > m ? v : TF( 1 ); }() ) );
    }
    bi.phi1 = ph;
    bi.ball_fin = 1e300; bi.ring_fin = 1e300;
    for ( SI q = 0; q < m; ++q ) bi.ball_fin = std::min( bi.ball_fin, a[ q ] / nw.nu[ ens[ q ] ] );
    for ( SI q = m; q < M; ++q ) bi.ring_fin = std::min( bi.ring_fin, a[ q ] / nw.nu[ ens[ q ] ] );
    if ( M == m ) bi.ring_fin = 1;
    w_out = w;
    return bi;
}

// =====================================================================================
// LE RELEVEMENT PAR AMAS ( `--local-amas` )
//
// La boule des cellules a reparer n'est pas un bloc : c'est une POUSSIERE DE PETITS AMAS ( mesure :
// 20 composantes connexes dont la plus grosse fait 62 cellules, a `F = 0.05` et deux couches ). Les
// traiter comme un seul systeme couple de 343 inconnues, resolu par gradient conjugue a travers
// l'arbre global, est une maladresse : le couplage entre amas eloignes est fictif, et sur vingt
// inconnues il n'y a AUCUNE RAISON de passer par une structure d'acceleration.
//
// Ici chaque amas a ses propres inconnues et ses cellules calculees EN FORCE BRUTE contre une
// petite liste de germes ( `Balayage2`, le fournisseur temoin du banc ). Plus d'arbre, plus de
// `set_weights` en `O( n )` : le cout de la reparation cesse de dependre de `n`.
//
// LES TROIS ENSEMBLES d'un amas :
//   `P` les INCONNUES ( la composante connexe ) ;
//   `R` la COURONNE : a poids IMPOSES, mais DANS l'objectif -- c'est ce qui la protege ;
//   `C` les COUPEURS : `P + R + les voisines de R`. Une cellule de `P + R` a toutes ses voisines
//       dans `C`, donc la force brute contre `C` rend la cellule EXACTE -- tant que la connectivite
//       ne change pas, ce qu'on compte ( `fuites` ).
//
// DEUX CHOSES QUI ONT CHACUNE COUTE UNE MESURE FAUSSE.
//
//   * LES AMAS SE FUSIONNENT DES QUE LEURS COURONNES SE TOUCHENT ( composantes de `dist <= N+1`, pas
//     de `dist <= N` ). Sinon deux amas voisins partagent des cellules de bord, chacun les protege
//     dans son coin, et leurs effets s'additionnent pour les tuer.
//   * LE CHEMIN EST COMMUN. Donner a chaque amas sa propre continuation ne marche pas : chacun
//     resout en supposant les autres non repares, et la superposition des solutions n'est coherente
//     avec aucune d'elles -- les boules finissent saines et la masse minimale GLOBALE tombe quand
//     meme a zero. On garde donc un `s` unique et on boucle sur les amas a chaque sous-pas :
//     l'algebre reste petite et independante, l'etat reste coherent.
// =====================================================================================

struct StatsAmas {
    SI nb_amas = 0, taille_max = 0, nb_cel = 0, fuites = 0, n_inconnues = 0;
    int iter = 0, nb_pas = 0, nb_refus = 0, nb_refresh = 0;
    SI  nb_abandon = 0;            ///< amas pour lesquels AUCUN `( lam, beta )` ne donne un etat vivant
    TF  lam_min = 1e300, lam_max = 0;   ///< la contraction effectivement retenue
    SI  ech_fenetre = 0;           ///< abandons ou LES DEUX bords echouent : la fenetre est VIDE
    SI  ech_inconnue = 0;          ///< ... ou seules les inconnues meurent, quel que soit `beta`
    SI  ech_couronne = 0;          ///< ... ou seule la couronne meurt
    SI  nb_dump = 0, nb_scan = 0, nb_demo = 0;
    double rho_max = 0;            ///< le plus grand rayon libre trouve dans `R`
    int nb_refresh_loc = 0;        ///< rafraichissements faits SUR LES PATCHS : ils ne coutent pas un diagramme
    SI  cel_refresh = 0;           ///< cellules depensees par ces rafraichissements
    // OU PASSE LE TEMPS. Quatre postes disjoints, mesures au meme endroit que le travail.
    double t_geo = 0;              ///< `mesure( false )` : la geometrie seule, dans la recherche lineaire
    double t_asm = 0;              ///< `mesure( true )`  : geometrie + assemblage ( dont `place` / `coupeur` )
    double t_cg  = 0;              ///< l'algebre : equations normales et gradient conjugue
    double t_ref = 0;              ///< le rafraichissement sur les patchs
    double t_ver = 0;              ///< le verdict : la remesure de tous les amas apres chaque sous-pas
    double t_bis = 0;              ///< la recherche `( lam, beta )` : les appels a `etat()`, SEQUENTIELS
    double t_pre = 0;              ///< `construit()` : le graphe, les composantes, les coupeurs
    SI     n_geo = 0, n_asm = 0;
    SI     n_sous = 0;             ///< appels a `resout_amas` : les sous-resolutions
    SI     n_ls_ech = 0;           ///< recherches lineaires qui n'ont RIEN trouve
    SI     n_geo_ech = 0;          ///< evaluations geometriques brulees par celles-la
    TF s = 0;
    TF ball_fin = 1e300;
};

template<class PD, class Rho>
StatsAmas repare_amas( Newton<PD,Rho> &nw, const std::vector<TF> &w0, const std::vector<TF> &dn,
                       const std::vector<TF> &a_cible, const std::vector<Facette> &fa_cible,
                       TF F, int N, const Opts &o, std::vector<TF> &w_out, TF seuil_abs = 0,
                       std::vector<char> *zone = nullptr, const std::vector<Facette> *fa0 = nullptr ) {
    using TK = typename PD::TKernel;
    const SI n = nw.pd.n;
    StatsAmas st;

    // LE SEUIL DE « MALADE » EST CELUI DE L'APPELANT. `local_pince` vaut la moitie de `nu` : au
    // depart d'une etape de continuation, ou tout le diagramme est encore loin de sa cible, des
    // CENTAINES de cellules passent sous cette barre sans etre le moins du monde en danger. Les
    // amas construits autour d'elles ne sont plus locaux -- on a mesure 666 inconnues pour DEUX
    // cellules malades au sens de la boucle -- et la force brute sans arbre y devient le mauvais
    // outil. Quand l'appelant a son propre seuil ( la boucle de Newton en a un ), c'est le sien.
    std::vector<SI> S;
    const TF seuil_i = seuil_abs > 0 ? seuil_abs : TF( 0 );
    for ( SI i = 0; i < n; ++i ) {
        const TF lim = seuil_abs > 0 ? seuil_i : o.local_pince * nw.nu[ i ];
        if ( a_cible[ i ] < lim ) S.push_back( i );
    }
    for ( SI i = 0; i < n; ++i ) w_out[ i ] = w0[ i ] + F * dn[ i ];
    if ( zone ) zone->assign( n, 0 );
    if ( S.empty() ) { st.s = 1; return st; }

    Laplacien L;

    // ---- LA STRUCTURE DE VOISINAGE, REJOUABLE. Les listes de candidats calculees une fois sur la
    //      geometrie de la cible deviennent fausses en cours de route : la reparation DEPLACE
    //      beaucoup les cellules, c'est son objet, et un amas qui gonfle peut avaler une cellule
    //      d'un amas voisin qui ne l'avait jamais eue pour candidat. Le symptome mesure : la mesure
    //      locale annonce 0.303 nu la ou la mesure globale des memes poids donne 0.000 -- la cellule
    //      locale est un SUR-ENSEMBLE, il lui manque un coupeur, et aucun compteur de facettes ne
    //      peut le voir puisqu'un coupeur manquant ne produit AUCUNE facette.
    //      On refait donc la structure tous les `local_refresh` sous-pas : une passe globale par
    //      sous-pas, pas par evaluation -- les evaluations restent locales, c'est la qu'est le gain.
    struct Prep {
        std::vector<SI> ens, C;
        SI m = 0;
        std::vector<TF> cx, cy;
        std::vector<d2::SI32> cid;
    };
    std::vector<Prep> pr;
    std::vector<char> vu( n, 0 );
    auto construit = [ & ]( const std::vector<Facette> &fa ) {
        const double t_pre0 = now();
        L.assemble( n, fa );
        std::vector<int> dist( n, -1 );
        std::vector<SI> file( S.begin(), S.end() );
        for ( SI i : S ) dist[ i ] = 0;
        for ( size_t q = 0; q < file.size(); ++q ) {
            const SI i = file[ q ];
            if ( dist[ i ] >= N + 2 + o.local_coupeurs + ( o.local_patch ? 1 : 0 ) ) continue;
            for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) {
                const SI j = L.col[ e ];
                if ( dist[ j ] < 0 ) { dist[ j ] = dist[ i ] + 1; file.push_back( j ); }
            }
        }
        std::vector<int> num( n, -1 );
        std::vector<std::vector<SI>> groupes;
        std::vector<SI> pile;
        for ( SI i = 0; i < n; ++i ) {
            if ( dist[ i ] < 0 || dist[ i ] > N + 1 || num[ i ] >= 0 ) continue;
            const int g = int( groupes.size() );
            groupes.emplace_back();
            pile.assign( 1, i );
            num[ i ] = g;
            while ( ! pile.empty() ) {
                const SI r = pile.back();
                pile.pop_back();
                if ( dist[ r ] <= N ) groupes[ g ].push_back( r );
                for ( SI e = L.row[ r ]; e < L.row[ r + 1 ]; ++e ) {
                    const SI j = L.col[ e ];
                    if ( dist[ j ] >= 0 && dist[ j ] <= N + 1 && num[ j ] < 0 ) { num[ j ] = g; pile.push_back( j ); }
                }
            }
        }
        groupes.erase( std::remove_if( groupes.begin(), groupes.end(),
                                       []( const std::vector<SI> &v ) { return v.empty(); } ), groupes.end() );
        pr.assign( groupes.size(), Prep{} );
        st.nb_amas = 0; st.taille_max = 0; st.n_inconnues = 0;
        for ( size_t g = 0; g < groupes.size(); ++g ) {
            Prep &A = pr[ g ];
            A.ens = groupes[ g ];
            A.m = SI( A.ens.size() );
            for ( SI i : A.ens ) vu[ i ] = 1;
            for ( SI q = 0; q < A.m; ++q )
                for ( SI e = L.row[ A.ens[ q ] ]; e < L.row[ A.ens[ q ] + 1 ]; ++e )
                    if ( ! vu[ L.col[ e ] ] ) { vu[ L.col[ e ] ] = 1; A.ens.push_back( L.col[ e ] ); }
            A.C = A.ens;
            size_t deb = size_t( A.m );
            // UNE COUCHE DE MARGE EN MODE PATCH. Le rafraichissement local est exact pour les
            // cellules dont TOUS les coupeurs sont dans `C`, c'est-a-dire jusqu'a la distance
            // `N+1+coupeurs` -- soit exactement la profondeur dont `construit` a besoin, sans un
            // poil de marge. Or la geometrie bouge, c'est tout l'objet de la reparation : une
            // cellule peut gagner une voisine qui etait hors du patch. On elargit donc d'une couche.
            const int n_couches = o.local_coupeurs + ( o.local_patch ? 1 : 0 );
            for ( int couche = 0; couche <= n_couches; ++couche ) {
                const size_t fin = A.C.size();
                for ( size_t q = deb; q < fin; ++q )
                    for ( SI e = L.row[ A.C[ q ] ]; e < L.row[ A.C[ q ] + 1 ]; ++e )
                        if ( ! vu[ L.col[ e ] ] ) { vu[ L.col[ e ] ] = 1; A.C.push_back( L.col[ e ] ); }
                deb = fin;
            }
            for ( SI i : A.C ) vu[ i ] = 0;
            const SI nc = SI( A.C.size() );
            A.cx.resize( nc ); A.cy.resize( nc ); A.cid.resize( nc );
            for ( SI q = 0; q < nc; ++q ) {
                A.cx[ q ] = nw.P[ 0 ][ A.C[ q ] ];
                A.cy[ q ] = nw.P[ 1 ][ A.C[ q ] ];
                A.cid[ q ] = d2::SI32( A.C[ q ] );
            }
            st.nb_amas += 1;
            st.taille_max = std::max( st.taille_max, A.m );
            st.n_inconnues += A.m;
        }
        st.t_pre += now() - t_pre0;
    };
    // ON BATIT LES AMAS SUR LE DIAGRAMME DE `w0`, PAS SUR CELUI DE LA CIBLE.
    //
    // C'est la correction d'un defaut qui expliquait la TOTALITE des refus. `S` est l'ensemble des
    // cellules malades SOUS LE PAS, donc repere sur la cible -- mais une cellule deja morte la-bas
    // NE PRODUIT AUCUNE FACETTE. Elle n'a donc aucun voisin dans le graphe de Laguerre de la cible :
    // son amas se reduit a elle seule, sans couronne et sans coupeurs. La force brute la calcule
    // alors contre une liste vide, rend le DOMAINE ENTIER, et la declare florissante -- on a mesure
    // « LOCAL dit 5.000e+03 » a `n = 5000`, soit cinq mille fois la masse cible. La continuation
    // avance jusqu'a `s = 1` sans rien reparer, et la mesure globale decouvre le cadavre.
    //
    // Le diagramme de `w0` est sain par construction ( c'est l'etat courant du Newton, accepte ),
    // donc la cellule y est vivante et y a tous ses voisins. C'est la bonne base, et le
    // rafraichissement par sous-pas fait le reste du chemin.
    // LE GRAPHE EST L'UNION DES DEUX, et chacun des deux est necessaire.
    //
    //   * celui de `w0` IDENTIFIE `E` : une cellule morte a `F` ne produit aucune facette, donc n'a
    //     aucun voisin dans le graphe de la cible -- son amas se reduirait a elle seule, sans
    //     couronne ni coupeurs, et la force brute lui rendrait le domaine entier ( § 15.9 ) ;
    //   * celui de `F` DONNE LA FRONTIERE : `dE` est ce qui coupe et ce qui borne la zone
    //     reconstruite, et c'est dans la geometrie d'arrivee que ca se joue.
    //
    // L'union ne peut que faire grossir la zone, jamais la retrecir, ce qui est le seul sens sur.
    {
        std::vector<Facette> fu;
        if ( fa0 ) fu = *fa0;
        fu.insert( fu.end(), fa_cible.begin(), fa_cible.end() );
        construit( fu );
    }

    // ---- LE CHEMIN COMMUN : `s` de 0 a 1 ; a chaque sous-pas, chaque amas est re-resolu chez lui
    std::vector<TF> w( n );
    std::vector<TF> cw;
    std::vector<TF> a_ref;                               ///< l'aire des cellules de `E` au pas pur
    std::vector<TF> a, dia, u, r, cc, t1, t2, rhs, d, pcg, Ap, zz, rr, jac, w_sauve;
    std::vector<SI> row, col;
    // LE RANG DU GERME, EN O(1). `place` et `coupeur` etaient deux recherches LINEAIRES -- dans
    // `ens` ( M ) et dans `C` ( nc ) -- refaites POUR CHAQUE FACETTE DE CHAQUE CELLULE a chaque
    // assemblage. Avec `M ~ 100` et `nc ~ 300`, c'est un demi-million de comparaisons par mesure
    // pour une information que deux tableaux indexes par le germe rendent immediatement. On les
    // remplit en O( M + nc ) a l'entree de l'amas et on les rend a la sortie.
    std::vector<SI>   rang_ens( n, -1 );
    std::vector<char> dans_C( n, 0 );

    /// resout l'amas `A` a bord fige, dans `w`. Rend la plus petite masse obtenue ( boule et couronne ).
    auto resout_amas = [ & ]( const Prep &A ) {
        ++st.n_sous;
        const SI m = A.m, M = SI( A.ens.size() ), nc = SI( A.C.size() );
        cw.resize( nc );
        a.assign( M, TF( 0 ) ); dia.assign( M, TF( 0 ) ); u.assign( M, TF( 0 ) ); r.assign( M, TF( 0 ) );
        t1.assign( M, TF( 0 ) ); t2.assign( M, TF( 0 ) );
        rhs.assign( m, TF( 0 ) ); d.assign( m, TF( 0 ) ); pcg.assign( m, TF( 0 ) ); Ap.assign( m, TF( 0 ) );
        zz.assign( m, TF( 0 ) ); rr.assign( m, TF( 0 ) ); jac.assign( m, TF( 0 ) );
        for ( SI q = 0; q < M; ++q ) rang_ens[ A.ens[ q ] ] = q;
        for ( SI c : A.C ) dans_C[ c ] = 1;
        auto rend = [ & ]() {
            for ( SI q = 0; q < M; ++q ) rang_ens[ A.ens[ q ] ] = -1;
            for ( SI c : A.C ) dans_C[ c ] = 0;
        };
        auto place = [ & ]( SI id ) -> SI { return rang_ens[ id ]; };
        auto coupeur = [ & ]( SI id ) { return dans_C[ id ] != 0; };
        auto mesure = [ & ]( bool fac ) {
            const double t_dep = now();
            for ( SI q = 0; q < nc; ++q ) cw[ q ] = w[ A.C[ q ] ];
            if ( fac ) { row.assign( M + 1, 0 ); col.clear(); cc.clear(); dia.assign( M, TF( 0 ) ); }
            // LA RECHERCHE LINEAIRE EN PARALLELE. Sans assemblage, la boucle n'ecrit QUE `a[ q ]` :
            // aucune dependance entre cellules, aucune structure partagee. C'est 76 % du temps de la
            // reparation ( § 15.6 ), et c'est le seul endroit ou le parallelisme est immediat --
            // paralleliser SUR LES AMAS demanderait de supprimer le `w` partage, leurs coupeurs se
            // recouvrant.
            if ( ! fac ) {
                SI cel_loc = 0;
                #pragma omp parallel for schedule( static ) reduction( + : cel_loc ) if( M >= 256 )
                for ( SI q = 0; q < M; ++q ) {
                    const SI id = A.ens[ q ];
                    typename PD::Cell cl;
                    Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc ), 0, 1,
                                          nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], w[ id ], d2::SI32( id ) };
                    d2::moteur<TK>( &f, &cl );
                    ++cel_loc;
                    a[ q ] = cl.nb < 0 ? TF( 0 ) : nw.rho->mesure( cl, []( int, TF ) {} );
                }
                st.nb_cel += cel_loc;
                st.t_geo += now() - t_dep; ++st.n_geo;
                return;
            }
            typename PD::Cell cel;
            for ( SI q = 0; q < M; ++q ) {
                const SI id = A.ens[ q ];
                Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc ), 0, 1,
                                      nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], w[ id ], d2::SI32( id ) };
                d2::moteur<TK>( &f, &cel );
                ++st.nb_cel;
                a[ q ] = 0;
                if ( cel.nb < 0 ) continue;
                a[ q ] = nw.rho->mesure( cel, [ & ]( int j, TF mes ) {
                    if ( ! fac ) return;
                    const SI p = place( j );
                    if ( q < M && ! coupeur( j ) ) ++st.fuites;   // une voisine hors des coupeurs :
                    // la cellule locale n'est alors PAS exacte, c'est la seule fuite qui compte
                    TF d2v = 0;
                    for ( int dd = 0; dd < 2; ++dd ) { const TF e = nw.P[ dd ][ j ] - nw.P[ dd ][ id ]; d2v += e * e; }
                    if ( ! ( d2v > 0 ) ) return;
                    const TF cij = mes / ( 2 * std::sqrt( d2v ) );
                    dia[ q ] += cij;
                    if ( p >= 0 && p < m ) { col.push_back( p ); cc.push_back( cij ); ++row[ q + 1 ]; }
                } );
            }
            if ( fac ) {
                for ( SI q = 0; q < M; ++q ) row[ q + 1 ] += row[ q ];
                for ( SI q = 0; q < M; ++q ) if ( ! ( dia[ q ] > 0 ) ) dia[ q ] = 1;
            }
            if ( fac ) { st.t_asm += now() - t_dep; ++st.n_asm; }
            else       { st.t_geo += now() - t_dep; ++st.n_geo; }
        };
        std::function<TF()> phi;                     // definie plus bas, quand `x_ref` existe
        // LE PLANCHER DE SANTE EST CELUI DE L'APPELANT. La reparation validait sur `a / nu >
        // local_eps` ( relatif, 0.05 ) tandis que la boucle exige `a > eps` ( absolu ) : pour une
        // cellule de petite cible le premier critere est le plus permissif, et la reparation rendait
        // un etat qu'elle croyait sain et que la boucle comptait malade. On rend donc la MARGE
        // NORMALISEE -- `> 1` veut dire sain -- calculee sur le seuil de l'appelant quand il en a un.
        auto lim = [ & ]( SI id ) { return seuil_abs > 0 ? seuil_abs : o.local_eps * nw.nu[ id ]; };
        auto plancher = [ & ]() {
            TF v = 1e300;
            for ( SI q = 0; q < M; ++q ) v = std::min( v, a[ q ] / lim( A.ens[ q ] ) );
            return v;
        };
        // VIVANT N'EST PAS SAIN, et les confondre bloquait tout. Le depart par homothetie est
        // DELIBEREMENT petit -- les cellules valent `lam^2` de leur taille -- donc tester le seuil de
        // l'appelant a l'entree faisait rendre la main au sous-probleme avant qu'il ait rien fait, et
        // faisait refuser a la recherche lineaire chacun de ses essais. La seule chose qu'on ne doit
        // jamais perdre en chemin, c'est la VIE : une cellule d'aire nulle n'a plus de facettes, sort
        // du graphe et ne peut plus revenir. Le seuil de l'appelant, lui, ne sert qu'au verdict final.
        auto vivant = [ & ]() {
            TF v = 1e300;
            for ( SI q = 0; q < M; ++q ) v = std::min( v, a[ q ] );
            return v;
        };
        mesure( true );
        if ( ! ( vivant() > 0 ) ) { rend(); return plancher(); }
        // L'AIRE DE REFERENCE DE LA COURONNE. Geler le POIDS d'une cellule de couronne ne gele pas
        // son AIRE : celle-ci depend des poids de ses voisines, qu'on est justement en train de
        // changer. La couronne derive donc, et sa derive se propage a ses propres voisines, hors de
        // la zone surveillee -- c'est le mecanisme de la fuite. On lui donne donc pour cible l'aire
        // qu'elle avait AVANT la reparation, avec une forte penalisation : `r_i` devient
        // `g( x_i ) + sqrt( lam ) ( x_i - x_ref_i )`, dont le zero est en `x_ref` des que `lam` est
        // grand -- et la barriere reste, donc une cellule de couronne ne peut toujours pas mourir.
        // L'AIRE DE REFERENCE DE LA COURONNE EST CELLE QU'ELLE A AU PAS DE NEWTON PUR. C'est l'etat
        // dont on ne veut pas perturber l'exterieur : la couronne est la frontiere entre la zone
        // reconstruite et le reste du diagramme, et la tenir la tient tout ce qu'il y a derriere.
        std::vector<TF> x_ref( M, TF( 0 ) );
        for ( SI q = m; q < M; ++q ) {
            const SI id = A.ens[ q ];
            x_ref[ q ] = ( a_ref.empty() ? a[ q ] : a_ref[ id ] ) / nw.nu[ id ];
        }
        const TF rl = std::sqrt( std::max( o.local_bord, TF( 0 ) ) );
        auto res_g = [ & ]( SI q, TF x ) { return q < m ? bar_g( x ) : bar_g( x ) + rl * ( x - x_ref[ q ] ); };
        auto res_gp = [ & ]( SI q, TF x ) { return q < m ? bar_gp( x ) : bar_gp( x ) + rl; };
        // L'OBJECTIF DE GRADIENT D'AIRE ( `--local-grad` ). Viser `nu` sur l'interieur force a lui
        // prendre de la masse quelque part, et ce quelque part est la couronne : on s'impose une
        // contrainte de conservation dont on n'a aucun besoin, puisqu'on ne cherche pas la solution
        // mais un etat non degenere. Minimiser plutot
        //
        //      Phi = somme_( aretes de la zone ) ( x_i - x_j )^2,   x = A / nu
        //
        // n'a AUCUNE cible absolue, donc ne pousse sur rien -- et un zero au milieu d'un champ lisse
        // est impossible, ce qui suffit a garantir les aires non nulles.
        //
        // Gauss-Newton : `r = B x` ( `B` l'incidence ), `J = B diag( 1/nu ) L`, donc les equations
        // normales sont `L^T D K D L d = - L^T D K x` avec `K = B^T B` le laplacien du graphe et
        // `D = diag( 1/nu )`. Tout se fait sans matrice sur la meme structure locale.
        // LA COURONNE EST TENUE A SON AIRE, DANS L'OBJECTIF DE GRADIENT AUSSI. Geler le POIDS d'une
        // cellule de couronne ne gele pas son AIRE : celle-ci depend des poids de ses voisines,
        // qu'on est en train de changer. Si la couronne derive, sa derive se propage a l'exterieur,
        // qu'on ne surveille pas. L'objectif complet est donc
        //
        //      Phi = somme_aretes ( x_i - x_j )^2  +  lam_b somme_couronne ( x_i - x_ref_i )^2
        //
        // ce qui remplace le laplacien de graphe `K` par `K + lam_b diag( couronne )` dans les
        // equations normales, et ajoute `- lam_b x_ref` au second membre. Rien d'autre ne change.
        const TF lb = std::max( o.local_bord, TF( 0 ) );
        auto Kx = [ & ]( const std::vector<TF> &x, std::vector<TF> &y, bool avec_ref ) {
            for ( SI q = 0; q < M; ++q ) {
                TF deg = 0, som = 0;
                for ( SI e = row[ q ]; e < row[ q + 1 ]; ++e ) { deg += 1; som += x[ col[ e ] ]; }
                y[ q ] = deg * x[ q ] - som;
                if ( q >= m ) y[ q ] += lb * ( x[ q ] - ( avec_ref ? x_ref[ q ] : TF( 0 ) ) );
            }
        };
        phi = [ & ]() {
            TF v = 0;
            if ( o.local_grad ) {
                for ( SI q = 0; q < M; ++q ) {
                    const TF xq = a[ q ] / nw.nu[ A.ens[ q ] ];
                    for ( SI e = row[ q ]; e < row[ q + 1 ]; ++e ) {
                        const SI j = col[ e ];
                        if ( j > q ) continue;            // chaque arete une fois
                        const TF xj = a[ j ] / nw.nu[ A.ens[ j ] ];
                        v += ( xq - xj ) * ( xq - xj );
                    }
                    if ( q >= m ) { const TF e = xq - x_ref[ q ]; v += lb * e * e; }
                }
                return v;
            }
            for ( SI q = 0; q < M; ++q ) { const TF g = res_g( q, a[ q ] / nw.nu[ A.ens[ q ] ] ); v += g * g; }
            return v;
        };
        TF ph = phi();
        // LA RECHERCHE LINEAIRE EST LE POSTE DOMINANT, ET C'EST SON POINT DE DEPART QUI COUTE.
        // Mesure : 274 000 evaluations geometriques pour 25 000 assemblages, soit ONZE essais par
        // iteration de Gauss-Newton -- onze fois toutes les cellules de l'amas recalculees en force
        // brute, 10 s sur les 13 de la reparation. La cause n'est pas le pas retenu mais le point de
        // depart : on repart de `t = 1` a CHAQUE iteration alors que le pas admissible ne bouge
        // guere d'une iteration a l'autre. On repart donc du dernier pas accepte, double -- ce qui
        // laisse le pas remonter si la non-linearite se calme, sans repayer la descente.
        TF t_dep_ls = 1;
        for ( int it = 0; it < o.local_maxit; ++it ) {
            for ( SI q = 0; q < M; ++q ) {
                const TF x = a[ q ] / nw.nu[ A.ens[ q ] ];
                if ( o.local_grad ) { r[ q ] = x; u[ q ] = TF( 1 ) / nw.nu[ A.ens[ q ] ]; }
                else                 { r[ q ] = res_g( q, x ); u[ q ] = res_gp( q, x ) / nw.nu[ A.ens[ q ] ]; }
            }
            if ( o.local_grad ) { Kx( r, t2, true ); for ( SI q = 0; q < M; ++q ) t1[ q ] = u[ q ] * t2[ q ]; }
            else                  for ( SI q = 0; q < M; ++q ) t1[ q ] = u[ q ] * r[ q ];
            for ( SI p = 0; p < m; ++p ) {               // `- L^T ( u r )`, restreint aux inconnues
                TF v = dia[ p ] * t1[ p ];
                for ( SI e = row[ p ]; e < row[ p + 1 ]; ++e ) v -= cc[ e ] * t1[ col[ e ] ];
                rhs[ p ] = -v;
                TF jj = u[ p ] * u[ p ] * dia[ p ] * dia[ p ];
                for ( SI e = row[ p ]; e < row[ p + 1 ]; ++e ) jj += u[ col[ e ] ] * u[ col[ e ] ] * cc[ e ] * cc[ e ];
                if ( o.local_grad ) jj *= TF( row[ p + 1 ] - row[ p ] ) + 1;   // le degre de `K`

                jac[ p ] = jj > 0 ? jj : TF( 1 );
            }
            TF nr0 = 0;
            for ( SI p = 0; p < m; ++p ) nr0 += rhs[ p ] * rhs[ p ];
            if ( ! ( nr0 > 0 ) ) break;
            auto applique = [ & ]( const std::vector<TF> &x, std::vector<TF> &y ) {
                for ( SI q = 0; q < M; ++q ) {           // `L x`, `x` porte par les inconnues
                    TF v = q < m ? dia[ q ] * x[ q ] : TF( 0 );
                    for ( SI e = row[ q ]; e < row[ q + 1 ]; ++e ) v -= cc[ e ] * x[ col[ e ] ];
                    t2[ q ] = u[ q ] * v;
                }
                if ( o.local_grad ) {                    // `D ( K + lam_b R ) D L x` au lieu de `u^2 L x`
                    Kx( t2, t1, false );
                    for ( SI q = 0; q < M; ++q ) t2[ q ] = u[ q ] * t1[ q ];
                } else
                    for ( SI q = 0; q < M; ++q ) t2[ q ] *= u[ q ];
                for ( SI p = 0; p < m; ++p ) {           // `L^T ( u^2 L x )`
                    TF v = dia[ p ] * t2[ p ];
                    for ( SI e = row[ p ]; e < row[ p + 1 ]; ++e ) v -= cc[ e ] * t2[ col[ e ] ];
                    y[ p ] = v;
                }
            };
            const double t_cg0 = now();
            d.assign( m, TF( 0 ) );
            rr = rhs;
            TF rz = 0;
            for ( SI p = 0; p < m; ++p ) { zz[ p ] = rr[ p ] / jac[ p ]; rz += rr[ p ] * zz[ p ]; }
            pcg = zz;
            for ( int k = 0; k < o.local_cg; ++k ) {
                applique( pcg, Ap );
                TF pAp = 0;
                for ( SI p = 0; p < m; ++p ) pAp += pcg[ p ] * Ap[ p ];
                if ( ! ( pAp > 0 ) ) break;
                const TF al = rz / pAp;
                TF nr = 0;
                for ( SI p = 0; p < m; ++p ) { d[ p ] += al * pcg[ p ]; rr[ p ] -= al * Ap[ p ]; nr += rr[ p ] * rr[ p ]; }
                if ( nr <= TF( 1e-16 ) * nr0 ) break;
                TF rz2 = 0;
                for ( SI p = 0; p < m; ++p ) { zz[ p ] = rr[ p ] / jac[ p ]; rz2 += rr[ p ] * zz[ p ]; }
                const TF be = rz2 / rz;
                rz = rz2;
                for ( SI p = 0; p < m; ++p ) pcg[ p ] = zz[ p ] + be * pcg[ p ];
            }
            st.t_cg += now() - t_cg0;
            w_sauve.assign( m, TF( 0 ) );
            for ( SI p = 0; p < m; ++p ) w_sauve[ p ] = w[ A.ens[ p ] ];
            // LE PLAFOND DE HALVINGS. Mesure : 94 a 98 % des evaluations geometriques etaient
            // brulees par des recherches lineaires QUI NE TROUVENT RIEN, a 27 essais chacune --
            // parce qu'une recherche qui echoue descend jusqu'a `1e-8`. Or une recherche qui echoue
            // n'echoue pas faute d'etre allee assez bas : elle echoue parce que le sous-probleme est
            // arrive au bout, et un pas de `2^-27` n'y changera rien. Douze essais suffisent a le
            // constater, et la boucle de Gauss-Newton s'arrete de toute facon au premier echec.
            bool pris = false;
            SI n_ess = 0;
            const TF t_min_ls = t_dep_ls * TF( 1 ) / 4096;
            for ( TF t = t_dep_ls; t >= t_min_ls; t /= 2 ) {
                ++n_ess;
                for ( SI p = 0; p < m; ++p ) w[ A.ens[ p ] ] = w_sauve[ p ] + t * d[ p ];
                mesure( false );
                // SANS BARRIERE, C'EST LA RECHERCHE LINEAIRE QUI TIENT LE PLANCHER : l'objectif de
                // gradient est lisse et ne repousse pas les cellules vides tout seul.
                if ( o.local_grad && ! ( vivant() > 0 ) ) continue;
                const TF p2 = phi();
                if ( p2 < ph * ( 1 - 1e-12 ) ) {
                    ph = p2; pris = true;
                    t_dep_ls = std::min( TF( 1 ), t * 2 );
                    break;
                }
            }
            ++st.iter;
            if ( ! pris ) { st.n_ls_ech += 1; st.n_geo_ech += n_ess; }
            if ( ! pris ) {
                for ( SI p = 0; p < m; ++p ) w[ A.ens[ p ] ] = w_sauve[ p ];
                mesure( true );
                break;
            }
            mesure( true );
            // LE CRITERE D'ARRET EST L'OBJECTIF REEL, PAS LE MINIMUM DE `phi`. Mesure : apres le
            // plafond de halvings, 89 a 96 % des evaluations restantes sont encore brulees par des
            // recherches lineaires en echec -- treize evaluations, a chaque fois, pour DECOUVRIR que
            // le sous-probleme est fini. Or on ne cherche pas le minimum de la barriere : on cherche
            // un etat non degenere, et des que le plancher est confortablement au-dessus du seuil,
            // il n'y a plus rien a gagner. On le teste donc AVANT de repartir pour une iteration.
            if ( o.local_marge > 0 && plancher() > o.local_marge ) break;
        }
        rend();
        return plancher();
    };

    /// mesure l'amas `A` dans `w`, sans rien resoudre : rend sa plus petite masse ( boule et couronne )
    auto mesure_amas = [ & ]( const Prep &A, bool inconnues_seules = false ) {
        const SI M = inconnues_seules ? A.m : SI( A.ens.size() ), nc = SI( A.C.size() );
        cw.resize( nc );
        for ( SI q = 0; q < nc; ++q ) cw[ q ] = w[ A.C[ q ] ];
        TF v = 1e300;
        typename PD::Cell cel;
        for ( SI q = 0; q < M; ++q ) {
            const SI id = A.ens[ q ];
            Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc ), 0, 1,
                                  nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], w[ id ], d2::SI32( id ) };
            d2::moteur<TK>( &f, &cel );
            ++st.nb_cel;
            const TF m = cel.nb < 0 ? TF( 0 ) : nw.rho->mesure( cel, []( int, TF ) {} );
            v = std::min( v, m / ( seuil_abs > 0 ? seuil_abs : o.local_eps * nw.nu[ id ] ) );
        }
        return v;
    };

    // =====================================================================================
    // LE DEPART PAR HOMOTHETIE ( il remplace la continuation par sous-pas )
    //
    // La couronne allait de `w0` a `w0 + F d` par sous-pas. C'est ce qui coutait : les cellules de
    // BORD bougent beaucoup avec `F`, donc il faut beaucoup de sous-pas, et chacun paie une
    // resolution locale complete. On pose donc la couronne DIRECTEMENT a sa valeur finale -- tous
    // les poids sont ceux de `F`, sauf ceux de `E` qu'on reconstruit -- et c'est l'INTERIEUR qu'on
    // place pour qu'il tienne dedans.
    //
    // POURQUOI UNE HOMOTHETIE, ET POURQUOI ELLE EST EXACTE. En diagramme de Laguerre, une homothetie
    // de rapport `lam` et une translation `-c` du DIAGRAMME s'ecrivent en forme fermee sur les poids :
    //
    //      w_i = lam w0_i + ( 1 - lam ) |p_i|^2 + 2 c . p_i        =>        C_i( w ) = lam C_i( w0 ) - c
    //
    // ( la condition `|x-p_i|^2 - w_i <= |x-p_j|^2 - w_j` se recrit, en posant `y = ( x + c ) / lam`,
    // exactement en `y dans C_i( w0 )` -- les termes en `|x|^2` se simplifient, le reste se divise
    // par `lam`. Les germes NE BOUGENT PAS : c'est le motif des cellules qui se contracte. )
    //
    // TOUTES les cellules retrecissent du meme facteur `lam^2` en aire, donc AUCUNE NE MEURT. C'est
    // ce qui distingue ce depart d'un abaissement uniforme des poids, qui tue les plus petites en
    // premier. Si `lam` doit etre petit, l'interieur se ramasse en un paquet minuscule -- mais
    // vivant et bien conditionne, ce qui est tout ce qu'on demande a un point de depart.
    //
    // LES DEUX k-DOP donnent `lam` sans recherche : l'empreinte de `E` sous `w0` ( source ) et sous
    // `w0 + F d` ( destination ). Ce qui ecrase les cellules, c'est que la PLACE allouee a `E`
    // retrecit sous le pas ; contracter le motif du meme facteur le fait rentrer. Le centre est
    // celui du k-DOP destination : c'est la que la place est.
    //
    // L'homothetie n'etant exacte que sur le diagramme ENTIER, l'appliquer au seul interieur avec la
    // couronne gelee ne garantit rien. On mesure donc, et si une inconnue est morte on divise `lam`
    // par deux. Le predicteur fait le gros du travail ; le garde-fou rend le resultat sur.
    // =====================================================================================

    static constexpr int KD = 8;                         // huit fentes, de 0 a pi
    TF ux[ KD ], uy[ KD ];
    for ( int q = 0; q < KD; ++q ) {
        const double th = 3.14159265358979323846 * q / KD;
        ux[ q ] = TF( std::cos( th ) ); uy[ q ] = TF( std::sin( th ) );
    }

    /// l'empreinte de `A.ens` sous les poids `wv` : k-DOP dans `lo` / `hi`, et les aires dans `aa`
    auto empreinte = [ & ]( const Prep &A, const std::vector<TF> &wv, TF *lo, TF *hi, std::vector<TF> *aa ) {
        const SI M = SI( A.ens.size() ), nc = SI( A.C.size() );
        cw.resize( nc );
        for ( SI q = 0; q < nc; ++q ) cw[ q ] = TK( wv[ A.C[ q ] ] );
        for ( int q = 0; q < KD; ++q ) { lo[ q ] = 1e300; hi[ q ] = -1e300; }
        typename PD::Cell cel;
        for ( SI q = 0; q < M; ++q ) {
            const SI id = A.ens[ q ];
            Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc ), 0, 1,
                                  nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], wv[ id ], d2::SI32( id ) };
            d2::moteur<TK>( &f, &cel );
            ++st.nb_cel;
            if ( cel.nb <= 0 ) { if ( aa ) ( *aa )[ id ] = 0; continue; }
            if ( aa ) ( *aa )[ id ] = nw.rho->mesure( cel, []( int, TF ) {} );
            for ( int v = 0; v < cel.nb; ++v )
                for ( int m = 0; m < KD; ++m ) {
                    const TF pr_ = ux[ m ] * cel.x( v ) + uy[ m ] * cel.y( v );
                    lo[ m ] = std::min( lo[ m ], pr_ ); hi[ m ] = std::max( hi[ m ], pr_ );
                }
        }
    };

    // LE PLAFOND DE TAILLE. Mesure : les amas abandonnes sont LES GROS. Sur un amas de 266 inconnues
    // et 177 de couronne, le prolongement harmonique pur laisse deja 109 cellules interieures mortes,
    // et aucun `beta` n'arrange rien. La raison est dimensionnelle : les poids de couronne s'etalent
    // sur `1e-2` a travers un amas large d'une dizaine de cellules, soit `1e-3` d'ecart par arete,
    // contre `dist^2 ~ 9e-4`. Le champ est TROP RAIDE POUR L'ESPACEMENT DES GERMES, et aucune
    // interpolation ne peut l'annuler puisque la pente est imposee par le bord. Il n'y a donc pas de
    // placement initial a trouver : chercher revient a bruler des millions de cellules pour rien.
    // On decline, la boucle divise `F` par deux, et l'amas suivant sera plus petit.
    if ( o.local_amas_max > 0 && st.taille_max > o.local_amas_max ) { st.s = 0; return st; }

    // LE DESSIN D'UN AMAS ( `--local-dump` ). Deux panneaux, `w0` a gauche et `w_F` a droite, meme
    // cadrage : c'est la comparaison qui parle. Inconnues en orange, couronne en bleu, coupeurs en
    // gris. Une cellule vide n'a pas de polygone : son germe est marque d'une croix rouge.
    auto dessine = [ & ]( const Prep &A, const std::vector<TF> &wa, const std::vector<TF> &wb,
                          const char *chem ) {
        const SI M = SI( A.ens.size() ), nc = SI( A.C.size() );
        struct Poly { std::vector<double> x, y; SI id; int rang; };
        std::vector<Poly> pg[ 2 ];
        double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
        for ( int pan = 0; pan < 2; ++pan ) {
            const std::vector<TF> &wv = pan ? wb : wa;
            cw.resize( nc );
            for ( SI q = 0; q < nc; ++q ) cw[ q ] = TK( wv[ A.C[ q ] ] );
            typename PD::Cell cel;
            for ( SI q = 0; q < nc; ++q ) {
                const SI id = A.C[ q ];
                Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc ), 0, 1,
                                      nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], wv[ id ],
                                      d2::SI32( id ) };
                d2::moteur<TK>( &f, &cel );
                Poly po; po.id = id;
                po.rang = q < A.m ? 0 : ( q < M ? 1 : 2 );   // inconnue / couronne / coupeur
                for ( int v = 0; v < std::max( cel.nb, 0 ); ++v ) {
                    po.x.push_back( cel.x( v ) );
                    po.y.push_back( cel.y( v ) );
                    if ( po.rang < 2 ) {
                        x0 = std::min( x0, po.x.back() ); x1 = std::max( x1, po.x.back() );
                        y0 = std::min( y0, po.y.back() ); y1 = std::max( y1, po.y.back() );
                    }
                }
                pg[ pan ].push_back( std::move( po ) );
            }
        }
        if ( ! ( x1 > x0 ) ) return;
        const double mx = 0.15 * std::max( x1 - x0, y1 - y0 );
        x0 -= mx; x1 += mx; y0 -= mx; y1 += mx;
        const double lg = x1 - x0, ht = y1 - y0, ech = 460.0 / std::max( lg, ht );
        auto XX = [ & ]( int pan, double x ) { return 20 + pan * 500 + ( x - x0 ) * ech; };
        auto YY = [ & ]( double y ) { return 20 + ( y1 - y ) * ech; };   // `y` vers le haut

        std::ofstream f( chem );
        f << "<svg xmlns='http://www.w3.org/2000/svg' width='1020' height='540'>\n"
          << "<rect width='1020' height='540' fill='white'/>\n";
        static const char *rem[ 3 ] = { "#e8811a", "#3a7bd5", "#dddddd" };
        for ( int pan = 0; pan < 2; ++pan ) {
            f << "<text x='" << XX( pan, x0 ) << "' y='14' font-family='sans-serif' font-size='13'>"
              << ( pan ? "w_F ( pas de Newton pur )" : "w0 ( etat sain )" ) << "</text>\n";
            for ( int couche = 2; couche >= 0; --couche )    // coupeurs d'abord, inconnues au-dessus
                for ( const Poly &po : pg[ pan ] ) {
                    if ( po.rang != couche ) continue;
                    if ( po.x.empty() ) {                    // cellule VIDE : une croix rouge au germe
                        const double gx = XX( pan, double( nw.P[ 0 ][ po.id ] ) ), gy = YY( double( nw.P[ 1 ][ po.id ] ) );
                        if ( couche < 2 )
                            f << "<path d='M" << gx - 5 << " " << gy - 5 << "L" << gx + 5 << " " << gy + 5
                              << "M" << gx - 5 << " " << gy + 5 << "L" << gx + 5 << " " << gy - 5
                              << "' stroke='red' stroke-width='2'/>\n";
                        continue;
                    }
                    f << "<polygon points='";
                    for ( size_t v = 0; v < po.x.size(); ++v )
                        f << XX( pan, po.x[ v ] ) << "," << YY( po.y[ v ] ) << " ";
                    f << "' fill='" << rem[ po.rang ] << "' fill-opacity='"
                      << ( couche == 2 ? "0.35" : "0.75" ) << "' stroke='#444' stroke-width='0.6'/>\n";
                }
            for ( const Poly &po : pg[ pan ] ) {
                if ( po.rang == 2 ) continue;
                f << "<circle cx='" << XX( pan, double( nw.P[ 0 ][ po.id ] ) ) << "' cy='"
                  << YY( double( nw.P[ 1 ][ po.id ] ) ) << "' r='1.6' fill='#111'/>\n";
            }
        }
        f << "<text x='20' y='534' font-family='sans-serif' font-size='12'>"
          << "orange = inconnues, bleu = couronne, gris = coupeurs, croix rouge = cellule vide"
          << "</text>\n</svg>\n";
    };

    for ( SI i = 0; i < n; ++i ) w[ i ] = w0[ i ] + F * dn[ i ];   // TOUT a `F`, couronne comprise
    const std::vector<TF> wF = w;
    a_ref.assign( n, TF( 0 ) );
    st.s = 1;                                            // il n'y a plus de chemin : on EST a `F`

    TF lo_s[ KD ], hi_s[ KD ], lo_d[ KD ], hi_d[ KD ];
    for ( const Prep &A : pr ) {
        empreinte( A, w,  lo_d, hi_d, &a_ref );          // au pas de Newton pur : l'aire de la couronne

        // LE DEPART EST UN VORONOI DECALE, PAS UNE HOMOTHETIE DES CELLULES.
        //
        //      w_i = lam ( w0_i - moyenne )  +  beta,      i dans l'interieur de `E`
        //
        // A `lam = 1` c'est le motif d'origine ; a `lam = 0` TOUS LES POIDS SONT EGAUX, donc
        // l'interieur se partage la region laissee par la couronne en VORONOI PUR. Et la il y a une
        // garantie, pas un espoir : pour `beta` assez grand, chaque germe interieur appartient a sa
        // propre cellule ( il suffit que `-beta <= |p_i - p_k|^2 - w_k` pour toute cellule `k` de
        // couronne ), donc AUCUNE cellule interieure n'est vide. C'est le point de depart non
        // degenere que la construction demande, et il est demontrable.
        //
        // C'est aussi la bonne echelle. L'homothetie exacte des CELLULES s'ecrit
        // `w_i = lam w0_i + ( 1 - lam ) |p_i|^2 + 2 c . p_i`, et son terme `|p_i|^2` est d'ordre 1 la
        // ou tout ce qui se passe localement est d'ordre `1e-4` : mesure faite, les seuls `lam` de
        // cette famille qui ne detruisent pas l'amas sont ceux qu'on ne distingue pas de 1, c'est a
        // dire aucune transformation. Le scaling des POIDS n'a pas ce defaut, puisqu'il ne fait
        // intervenir aucune grandeur etrangere a l'echelle locale.
        // LE PROLONGEMENT HARMONIQUE DES POIDS DE BORD.
        //
        // Un niveau CONSTANT a l'interieur ne peut pas suivre une couronne dont le niveau varie.
        // Mesure : sur un amas de 31 inconnues, les poids de couronne s'etalent sur `5.8e-3` alors
        // que le carre de la distance entre germes vaut `9e-4` ; la condition pour qu'un `beta`
        // unique existe est que l'etalement reste sous `2 dist^2`, on en est a SIX FOIS trop. Le
        // balayage le montre sans ambiguite : quand la couronne commence a mourir il reste huit a
        // dix cellules interieures mortes, l'intervalle n'est pas juste vide, il manque largement.
        //
        // On donne donc a l'interieur le prolongement HARMONIQUE du bord : `L_PP w_P = - L_PR w_R`
        // sur le graphe de l'amas, avec les conductivites de Laguerre. Le champ obtenu epouse la
        // frontiere par construction, et etant harmonique il n'a NI MAXIMUM NI MINIMUM INTERIEUR --
        // or c'est la courbure du champ de poids qui ecrase une cellule, pas son niveau. Le Voronoi
        // decale en est le cas particulier « frontiere constante ».
        //
        // Deux prolongements : celui de `w_F` ( la frontiere d'arrivee ) et celui de `w0` ( celle de
        // depart ). La famille est alors
        //
        //      w_i = harm( w_F )_i  +  lam ( w0_i - harm( w0 )_i )  +  beta
        //
        // dont `lam = 1` garde le detail local de `w0` en le posant sur la nouvelle frontiere, et
        // `lam = 0` rend le prolongement pur, le plus lisse donc le plus sur.
        // `x*`, LE SEUL CENTRE QUI CONVIENT POUR L'HOMOTHETIE.
        //
        // Dans la limite `lam -> 0`, les poids valent `|p_i|^2 + 2 c.p_i + beta` et la fonction
        // puissance de TOUTES les inconnues au point `y = -c` vaut `|y|^2 - beta`. L'interieur y gagne
        // donc ssi `beta >= |y|^2 - f_out( y )`, c'est a dire
        //
        //      beta >= g( y ) = max_k ( 2 y.q_k - |q_k|^2 + v_k )      sur les cellules GELEES
        //
        // un MAX DE FONCTIONS AFFINES : `g` est convexe et lineaire par morceaux, `{ g <= beta }` est
        // exactement le polygone que l'interieur gagne, et son minimiseur `x*` est le point ou il
        // gagne le moins cher. Placer le centre ailleurs -- au milieu d'un k-DOP, par exemple -- fait
        // atterrir le motif contracte la ou l'exterieur domine, et TOUT MEURT quel que soit `beta`.
        // C'est ce qu'on avait mesure ( « a lam -> 0, 109 des 266 interieures mortes » ) et lu a tort
        // comme une impossibilite.
        TF xsx = 0, xsy = 0, z0x = 0, z0y = 0;
        if ( o.placement_homo ) {
            std::vector<SI> gel;                         // les cellules gelees qui coupent l'amas
            for ( SI c : A.C ) if ( rang_ens[ c ] < 0 || true ) gel.push_back( c );
            {   // on enleve les inconnues : elles ne sont pas gelees
                std::vector<char> inc( 0 );
                gel.clear();
                for ( SI q = 0; q < SI( A.C.size() ); ++q ) {
                    const SI c = A.C[ q ];
                    bool est_inc = false;
                    for ( SI r = 0; r < A.m; ++r ) if ( A.ens[ r ] == c ) { est_inc = true; break; }
                    if ( ! est_inc ) gel.push_back( c );
                }
            }
            auto g_de = [ & ]( TF yx, TF yy ) {
                TF v = -1e300;
                for ( SI k : gel ) {
                    const TF qx = nw.P[ 0 ][ k ], qy = nw.P[ 1 ][ k ];
                    v = std::max( v, 2 * ( yx * qx + yy * qy ) - ( qx * qx + qy * qy ) + wF[ k ] );
                }
                return v;
            };
            // `g` est convexe : une grille grossiere puis trois raffinements locaux suffisent, et
            // c'est plus robuste qu'un sous-gradient sur un max de fonctions affines ( qui zigzague ).
            TF x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
            for ( SI q = 0; q < SI( A.ens.size() ); ++q ) {
                const SI id = A.ens[ q ];
                x0 = std::min( x0, nw.P[ 0 ][ id ] ); x1 = std::max( x1, nw.P[ 0 ][ id ] );
                y0 = std::min( y0, nw.P[ 1 ][ id ] ); y1 = std::max( y1, nw.P[ 1 ][ id ] );
            }
            // `x*` EST LE BARYCENTRE DES CELLULES INTERIEURES VALIDES A `F`.
            //
            // Deux tentatives precedentes, toutes deux fausses et mesurees comme telles.
            // `argmin g` -- le point ou l'interieur gagne LE MOINS CHER -- est un piege : « le moins
            // cher » veut dire « la ou l'arrangement gele resiste le moins », donc la ou une cellule
            // gelee est deja la plus mince ; y placer le paquet la tue au moment meme ou l'interieur
            // apparait. Et la version « centree » par transformee de distance sur une grille rendait
            // un rayon libre de 0.099 sur un amas de diametre 0.05 -- signe que la region marquee
            // touchait le bord de la grille et que le maximum atterrissait n'importe ou.
            //
            // Le barycentre des cellules qui EXISTENT a `F` n'a aucun de ces defauts : il est par
            // construction dans la region que l'interieur occupe deja, il ne demande ni grille ni
            // parametre, et il coute un balayage de l'amas.
            {
                const SI nc2 = SI( A.C.size() );
                cw.resize( nc2 );
                for ( SI q = 0; q < nc2; ++q ) cw[ q ] = TK( wF[ A.C[ q ] ] );
                TF sx_ = 0, sy_ = 0, sa_ = 0;
                typename PD::Cell cl;
                for ( SI q = 0; q < A.m; ++q ) {
                    const SI id = A.ens[ q ];
                    Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc2 ), 0, 1,
                                          nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], wF[ id ], d2::SI32( id ) };
                    d2::moteur<TK>( &f, &cl );
                    ++st.nb_cel;
                    if ( cl.nb <= 0 ) continue;
                    TF a2_ = 0, cxx = 0, cyy = 0;        // aire signee et centroide du polygone
                    for ( int v = 0, nb = cl.nb; v < nb; ++v ) {
                        const int v2 = v + 1 == nb ? 0 : v + 1;
                        const TF cr = cl.x( v ) * cl.y( v2 ) - cl.x( v2 ) * cl.y( v );
                        a2_ += cr;
                        cxx += ( cl.x( v ) + cl.x( v2 ) ) * cr;
                        cyy += ( cl.y( v ) + cl.y( v2 ) ) * cr;
                    }
                    if ( ! ( std::fabs( a2_ ) > 0 ) ) continue;
                    const TF ai = std::fabs( a2_ ) / 2;
                    sx_ += ai * ( cxx / ( 3 * a2_ ) ); sy_ += ai * ( cyy / ( 3 * a2_ ) ); sa_ += ai;
                }
                if ( sa_ > 0 ) { xsx = sx_ / sa_; xsy = sy_ / sa_; }
                else {                                   // aucune cellule valide : le barycentre des germes
                    for ( SI q = 0; q < A.m; ++q ) { xsx += nw.P[ 0 ][ A.ens[ q ] ]; xsy += nw.P[ 1 ][ A.ens[ q ] ]; }
                    xsx /= TF( std::max( SI( 1 ), A.m ) ); xsy /= TF( std::max( SI( 1 ), A.m ) );
                }
                // LE CENTRE DU MOTIF DE REFERENCE, a `w0`. Il faut les deux : `x*` dit OU poser le
                // motif, `z0` dit d'ou il part -- et la translation les relie en dependant de `lam`.
                {
                    for ( SI q = 0; q < nc2; ++q ) cw[ q ] = TK( w0[ A.C[ q ] ] );
                    TF tx = 0, ty = 0, ta = 0;
                    typename PD::Cell cl2;
                    for ( SI q = 0; q < A.m; ++q ) {
                        const SI id = A.ens[ q ];
                        Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc2 ), 0, 1,
                                              nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], w0[ id ], d2::SI32( id ) };
                        d2::moteur<TK>( &f, &cl2 );
                        ++st.nb_cel;
                        if ( cl2.nb <= 0 ) continue;
                        TF a2b = 0, cxx = 0, cyy = 0;
                        for ( int v = 0, nb = cl2.nb; v < nb; ++v ) {
                            const int v2 = v + 1 == nb ? 0 : v + 1;
                            const TF cr = cl2.x( v ) * cl2.y( v2 ) - cl2.x( v2 ) * cl2.y( v );
                            a2b += cr;
                            cxx += ( cl2.x( v ) + cl2.x( v2 ) ) * cr;
                            cyy += ( cl2.y( v ) + cl2.y( v2 ) ) * cr;
                        }
                        if ( ! ( std::fabs( a2b ) > 0 ) ) continue;
                        const TF ai = std::fabs( a2b ) / 2;
                        tx += ai * ( cxx / ( 3 * a2b ) ); ty += ai * ( cyy / ( 3 * a2b ) ); ta += ai;
                    }
                    if ( ta > 0 ) { z0x = tx / ta; z0y = ty / ta; }
                    else {
                        for ( SI q = 0; q < A.m; ++q ) { z0x += nw.P[ 0 ][ A.ens[ q ] ]; z0y += nw.P[ 1 ][ A.ens[ q ] ]; }
                        z0x /= TF( std::max( SI( 1 ), A.m ) ); z0y /= TF( std::max( SI( 1 ), A.m ) );
                    }
                }
            }
        }

        std::vector<TF> harmF( A.m, 0 ), harm0( A.m, 0 );
        {
            const SI M = SI( A.ens.size() );
            for ( SI q = 0; q < M; ++q ) rang_ens[ A.ens[ q ] ] = q;
            auto dirichlet = [ & ]( const std::vector<TF> &wb, std::vector<TF> &sol ) {
                TF mb = 0;
                for ( SI q = A.m; q < M; ++q ) mb += wb[ A.ens[ q ] ];
                mb /= TF( std::max( SI( 1 ), M - A.m ) );
                sol.assign( A.m, mb );
                for ( int it = 0; it < 500; ++it ) {     // Gauss-Seidel : petit et a diagonale dominante
                    TF dm = 0;
                    for ( SI q = 0; q < A.m; ++q ) {
                        const SI i = A.ens[ q ];
                        TF num = 0, den = 0;
                        for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) {
                            const SI j = L.col[ e ], pj = rang_ens[ j ];
                            if ( pj < 0 ) continue;      // hors de l'amas : pas une donnee du probleme
                            const TF c = L.c[ e ];
                            num += c * ( pj < A.m ? sol[ pj ] : wb[ j ] );
                            den += c;
                        }
                        if ( ! ( den > 0 ) ) continue;
                        const TF nv = num / den;
                        dm = std::max( dm, std::fabs( nv - sol[ q ] ) );
                        sol[ q ] = nv;
                    }
                    if ( ! ( dm > TF( 1e-15 ) ) ) break;
                }
            };
            dirichlet( wF, harmF );
            dirichlet( w0, harm0 );
            for ( SI q = 0; q < M; ++q ) rang_ens[ A.ens[ q ] ] = -1;
        }

        // LE QUATRIEME PARAMETRE : `beta`, UNE CONSTANTE ADDITIVE SUR LES SEULES INCONNUES.
        //
        // « Translation + homothetie » donne trois parametres en 2D, et il en faut quatre. Sans le
        // quatrieme, contracter NE SAUVE PAS : quand `lam -> 0` le poids tend vers
        // `|p_i|^2 + 2 c . p_i`, donc la fonction puissance de TOUTES les inconnues au point `-c`
        // vaut la meme constante `|c|^2`. Face a une couronne GELEE dont la valeur y est finie et
        // arbitraire, ou bien le paquet gagne, ou bien la couronne gagne PARTOUT et toutes les
        // inconnues sont vides d'un coup. C'est ce qu'on a mesure : zero reparation acceptee, et le
        // sous-probleme rendant la main avant d'avoir rien fait.
        //
        // Entre inconnues, seules les DIFFERENCES de poids comptent : `beta` ne change donc rien au
        // motif interne, il ne deplace que l'interface avec la couronne. `lam` regle la contraction
        // du motif -- aucune cellule ecrasee par ses voisines internes -- et `beta` regle la place
        // que le paquet prend contre la couronne. Les deux roles sont disjoints, et c'est pour ca
        // qu'un seul des deux ne peut pas suffire.
        //
        // La recherche sur `beta` se termine : quand il croit, la region du paquet croit, donc
        // chaque cellule inconnue croit. On balaie une echelle geometrique DES DEUX COTES -- trop
        // grand tue la couronne, trop petit tue les inconnues -- et on prend la premiere valeur ou
        // TOUT `E` est vivant.
        // LA RECHERCHE SUR `beta` EST ENCADREE, PAS BALAYEE. Un balayage geometrique partant de `d^2`
        // ( l'extension du k-DOP au carre ) ne trouve jamais rien : l'echelle qui compte pour les
        // poids est celle des ECARTS entre voisins, de l'ordre de `1e-4` quand `d^2` vaut `1e-3`. Le
        // premier essai non nul etait deja mille fois trop gros, et on ne descendait jamais en
        // dessous -- mesure : zero reparation, residu strictement egal a celui du pas nu.
        //
        // On exploite donc la structure : `min` des aires INCONNUES croit avec `beta` ( le paquet
        // prend de la place contre la couronne ), `min` des aires de COURONNE decroit. Deux
        // monotonies opposees, donc une fenetre. On part du petit, on double jusqu'a la franchir, et
        // on bissecte. Si les deux bords sont morts des `beta = 0`, il n'y a pas de fenetre : c'est
        // `lam` qu'il faut baisser.
        TF dmax = 0;
        for ( int m = 0; m < KD; ++m ) dmax = std::max( dmax, hi_d[ m ] - lo_d[ m ] );
        const TF d_beta = dmax > 0 ? dmax * dmax : TF( 1 );

        // ON DESCEND `lam` JUSQU'A CONTRACTER POUR DE BON. Le predicteur k-DOP vaut ~1 : l'empreinte
        // EXTERIEURE de `E` ne retrecit presque pas sous le pas, les cellules se rearrangent au
        // DEDANS. Trois paliers a partir de la ne contractaient donc rien, et on comparait l'ancien
        // motif a la nouvelle couronne -- ce qu'aucun `beta` ne peut concilier. Or en principe il
        // existe toujours une solution : quand `lam -> 0` le paquet se contracte vers un point,
        // chaque cellule garde sa part du motif a l'echelle `lam^2`, toutes vivantes, et un paquet
        // minuscule ne peut pas tuer la couronne. Si un plancher subsiste, il est ailleurs.
        // L'ECHELLE EST CELLE DE `1 - lam`, PAS DE `lam`. Le terme d'homothetie est `( 1 - lam ) |p|^2` :
        // avec des coordonnees d'ordre 1, il varie a travers l'amas d'environ `( 1 - lam ) * 2 |p| diam`.
        // Pour un amas de diametre `0.05` autour de `|p| ~ 0.5`, `lam = 0.93` donne deja `3e-3` --
        // TRENTE FOIS les ecarts de poids entre voisines, qui valent `1e-4`. Diviser `lam` par deux a
        // partir de la ne visite que le domaine ou la transformation est violente ; la plage utile est
        // `1 - lam ~ 1e-4`, et elle n'etait jamais essayee. On balaie donc `1 - lam` en geometrique.
        //
        // ( Le predicteur k-DOP, qui rendait `lam ~ 1`, avait donc raison : c'est s'en eloigner qui
        // etait l'erreur. Le compteur de « fenetre vide » est remis A CHAQUE PALIER -- accumule sur
        // toute l'echelle, il se declenchait des qu'un bord echouait ici et l'autre la, ce qui ne dit
        // rien sur un `lam` donne. )
        // LA TABLE QUI TRANCHE. « Morte » recouvre deux choses tres differentes : l'intersection des
        // demi-plans est VIDE ( `cel.nb <= 0` ), ce qui infirmerait la construction ; ou le polygone
        // existe mais sa masse mesuree vaut zero, l'aire etant en `lam^2`, ce qui est une obstruction
        // NUMERIQUE et confirme au contraire que le placement existe. Le test `masse > 0` confond les
        // deux, et c'est la seule mesure qui separe « impossible » de « trop petit pour se mesurer ».
        // LA TABLE QUI TRANCHE, PAR BISSECTION ENCADREE.
        //
        // Un balayage geometrique de `beta` ne peut pas repondre : a `lam = 1.5e-8` le motif contracte
        // tient dans une boule de rayon `lam D ~ 1e-9`, donc la fenetre en `beta` ou l'interieur
        // existe SANS avaler la couronne a la meme largeur -- alors que le balayage, autour de la
        // valeur pertinente `beta_min ~ 1e-2`, a un pas de `1e-2`. Il saute par-dessus une fenetre dix
        // millions de fois plus etroite que lui, et rend « couronne morte » la ou il faudrait lire
        // « je n'ai pas cherche assez fin ».
        //
        // La structure est monotone : `beta` petit -> l'interieur meurt ; `beta` grand -> la couronne
        // meurt ; entre les deux, la fenetre, si elle existe. On l'encadre donc, et on IMPRIME SA
        // LARGEUR -- c'est elle qui donne le `lam` maximal exploitable. Si la bissection resserre le
        // crochet sous toute largeur mesurable sans jamais voir un etat sain, la transition est
        // DIRECTE et la construction ne se transporte pas au sous-ensemble.
        if ( o.demo_homo > 0 && st.nb_demo < o.demo_homo ) {
            ++st.nb_demo;
            const SI M = SI( A.ens.size() ), nc = SI( A.C.size() );
            // LE CONTROLE QU'IL FALLAIT FAIRE D'ABORD : la couronne est-elle SAINE au pas nu ? Elle est
            // choisie comme voisinage des malades, et une voisine peut etre tres mince sans passer
            // sous `eps` -- auquel cas toute intrusion la tue, et la transition directe s'explique
            // sans rien invoquer sur `x*`.
            {
                cw.resize( nc );
                for ( SI q = 0; q < nc; ++q ) cw[ q ] = TK( wF[ A.C[ q ] ] );
                TF mi = 1e300, mc = 1e300;
                typename PD::Cell cl;
                for ( SI q = 0; q < M; ++q ) {
                    const SI id = A.ens[ q ];
                    Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc ), 0, 1,
                                          nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], wF[ id ], d2::SI32( id ) };
                    d2::moteur<TK>( &f, &cl );
                    const TF mm = cl.nb <= 0 ? TF( 0 ) : nw.rho->mesure( cl, []( int, TF ) {} );
                    if ( q < A.m ) mi = std::min( mi, mm / nw.nu[ id ] );
                    else           mc = std::min( mc, mm / nw.nu[ id ] );
                }
                std::printf( "   AU PAS NU : plus petite inconnue %.3e nu, plus petite COURONNE %.3e nu\n",
                             double( mi ), double( mc ) );
            }
            {
                TF bx0 = 1e300, bx1 = -1e300, by0 = 1e300, by1 = -1e300;
                for ( SI q = 0; q < A.m; ++q ) {
                    const SI id = A.ens[ q ];
                    bx0 = std::min( bx0, nw.P[ 0 ][ id ] ); bx1 = std::max( bx1, nw.P[ 0 ][ id ] );
                    by0 = std::min( by0, nw.P[ 1 ][ id ] ); by1 = std::max( by1, nw.P[ 1 ][ id ] );
                }
                std::printf( "   DEMO amas %d : %d inconnues, %d couronne, x* = ( %.6f, %.6f )"
                             "  --  boite des germes [ %.4f, %.4f ] x [ %.4f, %.4f ]\n",
                             int( st.nb_demo ), int( A.m ), int( M - A.m ), double( xsx ), double( xsy ),
                             double( bx0 ), double( bx1 ), double( by0 ), double( by1 ) );
              }
            for ( int tl = 2; tl <= 26; tl += 3 ) {
                const TF lh = TF( 1 ) / TF( SI( 1 ) << tl );
                // LA TRANSLATION DEPEND DE `lam`, ET C'EST LE COUPLAGE QUI MANQUAIT.
                //
                // `C_i( w ) = lam C_i( w0 ) + t`, donc pour que le motif atterrisse en `x*` il faut
                // `t = x* - lam z0`, ou `z0` est le centre du motif d'origine. J'avais pose `t = x*`
                // tout court : inoffensif pour `lam` minuscule, mais a `lam = 0.25` le motif atterrit
                // en `x* + 0.25 z0`, soit a `( 0.19, 0.17 )` de la cible -- hors d'un amas large de
                // 0.15. Les `lam` moderes, justement ceux dont les aires restent mesurables, posaient
                // le paquet n'importe ou. ( Et dans la formule des poids, `c = -t`. )
                const TF cx = -( xsx - lh * z0x ), cy = -( xsy - lh * z0y );
                SI n_ev = 0;
                /// +1 une inconnue morte ( monter beta ), -1 une couronne morte, 0 tout vivant
                auto etat_b = [ & ]( TF beta ) -> int {
                    for ( SI q = 0; q < A.m; ++q ) {
                        const SI id = A.ens[ q ];
                        const TF px = nw.P[ 0 ][ id ], py = nw.P[ 1 ][ id ];
                        w[ id ] = lh * w0[ id ] + ( 1 - lh ) * ( px * px + py * py )
                                + 2 * ( cx * px + cy * py ) + beta;
                    }
                    cw.resize( nc );
                    for ( SI q = 0; q < nc; ++q ) cw[ q ] = w[ A.C[ q ] ];
                    bool mort_int = false, mort_cour = false;
                    typename PD::Cell cl;
                    for ( SI q = 0; q < M; ++q ) {
                        const SI id = A.ens[ q ];
                        Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc ), 0, 1,
                                              nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], w[ id ],
                                              d2::SI32( id ) };
                        d2::moteur<TK>( &f, &cl );
                        ++n_ev;
                        const TF mm = cl.nb <= 0 ? TF( 0 ) : nw.rho->mesure( cl, []( int, TF ) {} );
                        if ( q < A.m ) mort_int |= ! ( mm > 0 );
                        else           mort_cour |= ! ( mm > 0 );
                    }
                    if ( mort_int ) return +1;
                    if ( mort_cour ) return -1;
                    return 0;
                };

                // ---- le crochet : `lo` ou l'interieur meurt, `hi` ou la couronne meurt
                TF lo = 0, hi = 0;
                bool ok = false, crochet = false;
                const int e0 = etat_b( 0 );
                if ( e0 == 0 ) { ok = true; lo = hi = 0; }
                else {
                    const TF signe = e0 > 0 ? TF( 1 ) : TF( -1 );
                    TF pas = 1e-12;
                    for ( int k = 0; k < 60 && ! ok && ! crochet; ++k, pas *= 2 ) {
                        const int e = etat_b( signe * pas );
                        if ( e == 0 ) { ok = true; lo = hi = signe * pas; break; }
                        if ( e != e0 ) { crochet = true;
                            if ( e0 > 0 ) { lo = signe * pas / 2; hi = signe * pas; }
                            else          { hi = signe * pas / 2; lo = signe * pas; } }
                    }
                }
                TF larg = 0;
                if ( ! ok && crochet ) {                 // on resserre jusqu'a voir un etat sain
                    for ( int b = 0; b < 200 && ! ok; ++b ) {
                        const TF mid = ( lo + hi ) / 2;
                        if ( ! ( std::fabs( hi - lo ) > 0 ) ) break;
                        const int e = etat_b( mid );
                        if ( e == 0 ) { ok = true; lo = hi = mid; break; }
                        if ( e > 0 ) lo = mid; else hi = mid;
                    }
                    larg = std::fabs( hi - lo );         // la fenetre est AU PLUS large de ca
                }
                if ( ok ) {                              // on mesure la largeur reelle de la fenetre
                    const TF b0 = lo;
                    TF bas = b0, haut = b0, p2 = 1e-14;
                    for ( int k = 0; k < 60; ++k, p2 *= 2 ) { if ( etat_b( b0 - p2 ) != 0 ) { bas = b0 - p2; break; } bas = b0 - p2; }
                    p2 = 1e-14;
                    for ( int k = 0; k < 60; ++k, p2 *= 2 ) { if ( etat_b( b0 + p2 ) != 0 ) { haut = b0 + p2; break; } haut = b0 + p2; }
                    larg = haut - bas;
                }
                // COMBIEN DE CELLULES BASCULENT AU PASSAGE ? C'est la mesure qui distingue une
                // transition LOCALE ( une inconnue meurt, une couronne meurt : le phenomene est reel )
                // d'un BASCULEMENT GLOBAL ( 66 -> 55 : `R` saute d'un point au tout, ce qui est
                // incompatible avec la croissance continue que l'algebre predit, et designe un bug ou
                // un minimum plat de `G_out - H_int` ).
                if ( crochet || ok ) {
                    auto compte = [ & ]( TF beta, SI &n_int, SI &n_cour ) {
                        for ( SI q = 0; q < A.m; ++q ) {
                            const SI id = A.ens[ q ];
                            const TF px = nw.P[ 0 ][ id ], py = nw.P[ 1 ][ id ];
                            w[ id ] = lh * w0[ id ] + ( 1 - lh ) * ( px * px + py * py )
                                    + 2 * ( cx * px + cy * py ) + beta;
                        }
                        cw.resize( nc );
                        for ( SI q = 0; q < nc; ++q ) cw[ q ] = w[ A.C[ q ] ];
                        n_int = 0; n_cour = 0;
                        typename PD::Cell cl;
                        for ( SI q = 0; q < M; ++q ) {
                            const SI id = A.ens[ q ];
                            Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc ), 0, 1,
                                                  nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], w[ id ],
                                                  d2::SI32( id ) };
                            d2::moteur<TK>( &f, &cl );
                            const TF mm = cl.nb <= 0 ? TF( 0 ) : nw.rho->mesure( cl, []( int, TF ) {} );
                            if ( q < A.m ) n_int += ! ( mm > 0 );
                            else           n_cour += ! ( mm > 0 );
                        }
                    };
                    const TF b0 = ( lo + hi ) / 2;
                    std::printf( "       transition autour de beta %+.9e :\n", double( b0 ) );
                    for ( int e = -3; e <= 3; ++e ) {
                        const TF dd = ( e == 0 ? TF( 0 ) : ( e < 0 ? -1 : 1 ) )
                                    * std::pow( TF( 10 ), TF( -16 + 4 * ( std::abs( e ) - 1 ) ) );
                        SI ni = 0, ncr = 0;
                        compte( b0 + dd, ni, ncr );
                        std::printf( "         beta %+.3e : %3d inconnues mortes / %d, %3d couronnes mortes / %d\n",
                                     double( dd ), int( ni ), int( A.m ), int( ncr ), int( M - A.m ) );
                    }
                }
                std::printf( "     lam %.3e : %s  beta %+.9e, largeur de fenetre %.3e"
                             "  ( %d evaluations )\n",
                             double( lh ), ok ? "FENETRE TROUVEE " : "aucun etat sain,",
                             double( lo ), double( larg ), int( n_ev ) );
            }
        }

        bool place = false;
        TF lam_pris = -1;
        for ( int tour = 0; tour < ( o.placement_homo ? 25 : 13 ) && ! place; ++tour ) {
            // `lam` : 1, 1/2, 1/4, ... puis 0 -- on garde le motif d'origine tant qu'il passe, et on
            // retombe sur le Voronoi pur, qui lui ne peut pas echouer.
            const TF lam = tour >= 12 ? TF( 0 ) : TF( 1 ) / TF( SI( 1 ) << tour );
            // l'homothetie demande de descendre BIEN plus bas : le motif doit tenir dans `R`, dont
            // le rayon est petit -- et c'est la que l'anisotropie mord, `lam` etant borne par la
            // direction MINCE de `R`, donc les aires par `lam^2`.
            const TF lam_h = tour >= 24 ? TF( 0 ) : TF( 1 ) / TF( SI( 1 ) << tour );
            bool vu_plus = false, vu_moins = false;      // PAR PALIER
            /// pose `beta` et rend : 0 tout vivant, +1 une inconnue morte, -1 une couronne morte
            auto etat = [ & ]( TF beta ) -> int {
                const double t_bis0 = now();
                st.t_bis -= t_bis0;                      // rendu a la sortie : trois `return`
                struct Fin { double *p; ~Fin() { *p += now(); } } fin{ &st.t_bis };
                for ( SI q = 0; q < A.m; ++q ) {
                    const SI id = A.ens[ q ];
                    if ( o.placement_homo ) {            // l'homothetie exacte, centree sur `x*`
                        const TF px = nw.P[ 0 ][ id ], py = nw.P[ 1 ][ id ];
                        const TF cx = -( xsx - lam_h * z0x ), cy = -( xsy - lam_h * z0y );
                        w[ id ] = lam_h * w0[ id ] + ( 1 - lam_h ) * ( px * px + py * py )
                                + 2 * ( cx * px + cy * py ) + beta;
                    } else
                        w[ id ] = harmF[ q ] + lam * ( w0[ id ] - harm0[ q ] ) + beta;
                }
                ++st.nb_pas;
                const SI M = SI( A.ens.size() ), nc = SI( A.C.size() );
                cw.resize( nc );
                for ( SI q = 0; q < nc; ++q ) cw[ q ] = w[ A.C[ q ] ];
                TF mu = 1e300, mr = 1e300;
                typename PD::Cell cel;
                for ( SI q = 0; q < M; ++q ) {
                    const SI id = A.ens[ q ];
                    Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc ), 0, 1,
                                          nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], w[ id ],
                                          d2::SI32( id ) };
                    d2::moteur<TK>( &f, &cel );
                    ++st.nb_cel;
                    const TF mm = cel.nb <= 0 ? TF( 0 ) : nw.rho->mesure( cel, []( int, TF ) {} );
                    if ( q < A.m ) mu = std::min( mu, mm );
                    else           mr = std::min( mr, mm );
                }
                if ( ! ( mu > 0 ) ) return +1;
                if ( ! ( mr > 0 ) ) return -1;
                return 0;
            };

            const int e0 = etat( 0 );
            vu_plus |= ( e0 > 0 ); vu_moins |= ( e0 < 0 );
            if ( e0 == 0 ) { place = true; break; }
            const TF signe = e0 > 0 ? TF( 1 ) : TF( -1 );
            TF pas = d_beta * TF( 1e-9 ), lo = 0, hi = 0;
            bool franchi = false;
            for ( int k = 0; k < 46 && ! place; ++k, pas *= 2 ) {
                const int e = etat( signe * pas );
                vu_plus |= ( e > 0 ); vu_moins |= ( e < 0 );
                if ( e == 0 ) { place = true; break; }
                if ( e != e0 ) { lo = signe * pas / 2; hi = signe * pas; franchi = true; break; }
            }
            for ( int b = 0; b < 40 && franchi && ! place; ++b ) {
                const TF mid = ( lo + hi ) / 2;
                const int e = etat( mid );
                if ( e == 0 ) { place = true; break; }
                if ( e == e0 ) lo = mid; else hi = mid;
            }
            if ( place ) lam_pris = lam;
            else {
                ++st.nb_refus;
                if ( vu_plus && vu_moins ) ++st.ech_fenetre;      // vide A CE PALIER
                else if ( vu_plus )        ++st.ech_inconnue;
                else                       ++st.ech_couronne;
            }
        }
        if ( place ) { st.lam_min = std::min( st.lam_min, lam_pris ); st.lam_max = std::max( st.lam_max, lam_pris ); }
        else {
            ++st.nb_abandon;
            // LE BALAYAGE QUI TRANCHE. A `lam = 0` tous les poids interieurs valent `beta`. Monter
            // `beta` fait grossir l'interieur ( monotone ) et retrecir la couronne ( monotone en sens
            // inverse ). Si les deux comptes de morts ne s'annulent jamais ENSEMBLE, l'intervalle
            // admissible est vide : ce n'est pas la recherche qui echoue, c'est que DEUX parametres
            // ne peuvent pas satisfaire une contrainte par paire ( interieure, couronne ).
            if ( o.local_scan > 0 && st.nb_scan < o.local_scan ) {
                ++st.nb_scan;
                const SI M = SI( A.ens.size() ), nc = SI( A.C.size() );
                TF wlo = 1e300, whi = -1e300;
                for ( SI q = A.m; q < M; ++q ) { wlo = std::min( wlo, w[ A.ens[ q ] ] ); whi = std::max( whi, w[ A.ens[ q ] ] ); }
                std::printf( "   SCAN amas %d inconnues %d couronne %d  --  w de couronne de %.6e a %.6e\n",
                             int( st.nb_scan ), int( A.m ), int( M - A.m ), double( wlo ), double( whi ) );
                for ( int k = 0; k <= 40; ++k ) {
                    const TF beta = ( whi - wlo ) * ( TF( k ) / 40 - TF( 1 ) / 2 );
                    for ( SI q = 0; q < A.m; ++q ) w[ A.ens[ q ] ] = harmF[ q ] + beta;
                    cw.resize( nc );
                    for ( SI q = 0; q < nc; ++q ) cw[ q ] = w[ A.C[ q ] ];
                    SI mi = 0, mr = 0;
                    TF ai = 1e300, ar = 1e300;
                    typename PD::Cell cel;
                    for ( SI q = 0; q < M; ++q ) {
                        const SI id = A.ens[ q ];
                        Balayage2<TK,true> f{ A.cx.data(), A.cy.data(), cw.data(), A.cid.data(), int( nc ), 0, 1,
                                              nw.P[ 0 ][ id ], nw.P[ 1 ][ id ], w[ id ],
                                              d2::SI32( id ) };
                        d2::moteur<TK>( &f, &cel );
                        const TF mm = cel.nb <= 0 ? TF( 0 ) : nw.rho->mesure( cel, []( int, TF ) {} );
                        const TF rr = mm / nw.nu[ id ];
                        if ( q < A.m ) { mi += ! ( mm > 0 ); ai = std::min( ai, rr ); }
                        else           { mr += ! ( mm > 0 ); ar = std::min( ar, rr ); }
                    }
                    std::printf( "     beta %+.6e   interieures mortes %3d ( min a/nu %.3e )"
                                 "   couronne morte %3d ( min a/nu %.3e )%s\n",
                                 double( beta ), int( mi ), double( ai ), int( mr ), double( ar ),
                                 ( mi == 0 && mr == 0 ) ? "   <-- VIVANT" : "" );
                }
            }
            if ( o.local_dump > 0 && st.nb_dump < o.local_dump ) {
                char chem[ 64 ];
                std::snprintf( chem, sizeof chem, "/tmp/amas_%02d.svg", int( st.nb_dump ) );
                std::vector<TF> wF( n );
                for ( SI i = 0; i < n; ++i ) wF[ i ] = w0[ i ] + F * dn[ i ];
                dessine( A, w0, wF, chem );
                ++st.nb_dump;
            }
        }
        if ( place ) resout_amas( A );
        else for ( SI q = 0; q < A.m; ++q ) {            // on rend l'amas au pas de Newton pur
            const SI id = A.ens[ q ];
            w[ id ] = w0[ id ] + F * dn[ id ];
        }
    }

    w_out = w;
    for ( const Prep &A : pr ) {
        if ( zone ) {
            for ( SI q = 0; q < A.m; ++q ) ( *zone )[ A.ens[ q ] ] = 1;                            // inconnue
            for ( size_t q = size_t( A.m ); q < A.ens.size(); ++q ) ( *zone )[ A.ens[ q ] ] = 2;   // couronne
        }
        st.ball_fin = std::min( st.ball_fin, mesure_amas( A ) );
    }
    return st;
}

// =====================================================================================
// LA REPARATION PAR CONTINUATION SUR LE BORD ( `repare_continuation` )
//
// La difference avec `repare_local` ( § 14 ) est la seule qui compte, et elle est entiere : on ne
// SAUTE PAS a `w0 + F d` pour reparer ensuite. On part de `w0`, QUI EST SAIN ET OU IL N'Y A RIEN A
// RESOUDRE, et on fait glisser les poids IMPOSES du bord de `w0` vers `w0 + F d` par sous-pas ; a
// chaque sous-pas l'interieur est re-resolu, a chaud. On ne traverse donc jamais un etat ou une
// cellule est deja morte : la barriere les maintient en vie LE LONG DU CHEMIN, au lieu d'avoir a
// ressusciter ce qui est deja mort -- ce qu'aucune methode passant par `L` ne sait faire ( § 13.1 ).
//
// Le plafond du § 14 ( reparable a `F = 0.033`, hors de portee a `0.036` ) etait donc un artefact
// du saut, pas une propriete du probleme.
//
// CE QU'ON SURVEILLE EN CHEMIN. La geometrie bouge, donc la frontiere de la boule aussi : une
// cellule interieure qui gonfle peut se mettre a toucher une cellule HORS de la zone surveillee.
// C'est le risque que le schema porte en lui. On le compte ( `fuites` ) : une facette d'une cellule
// de la boule vers un germe qui n'est ni dans la boule ni dans l'anneau.
//
// Le cout reste en cellules calculees, sur une zone qui ne depend que de `N`.
// =====================================================================================

template<class PD, class Rho>
BilanLocal repare_continuation( Newton<PD,Rho> &nw, const std::vector<SI> &rang, const std::vector<TF> &w0,
                                const std::vector<TF> &dn, TF F, int N, const Opts &o,
                                std::vector<TF> &w_out ) {
    PD &pd = nw.pd;
    const SI n = pd.n;
    BilanLocal bi;
    bi.F = F;
    bi.N = N;

    // ---- 1. LA CIBLE, ET LES CELLULES QU'ELLE PINCE
    std::vector<TF> w_cible( n ), a_cible;
    std::vector<Facette> fa_cible;
    for ( SI i = 0; i < n; ++i ) w_cible[ i ] = w0[ i ] + F * dn[ i ];
    nw.mesures_et_facettes( w_cible, a_cible, fa_cible );
    std::vector<SI> S;
    for ( SI i = 0; i < n; ++i ) if ( a_cible[ i ] < o.local_pince * nw.nu[ i ] ) S.push_back( i );
    bi.nS = SI( S.size() );
    w_out = w_cible;
    if ( S.empty() ) { bi.s = 1; return bi; }            // la cible passe : rien a reparer

    // ---- 2. LA BOULE ET L'ANNEAU, sur la geometrie de la cible
    Laplacien Lc;
    Lc.assemble( n, fa_cible );
    std::vector<int> dist( n, -1 );
    std::vector<SI> file( S.begin(), S.end() );
    for ( SI i : S ) dist[ i ] = 0;
    for ( size_t q = 0; q < file.size(); ++q ) {
        const SI i = file[ q ];
        if ( dist[ i ] >= N + 1 ) continue;
        for ( SI e = Lc.row[ i ]; e < Lc.row[ i + 1 ]; ++e ) {
            const SI j = Lc.col[ e ];
            if ( dist[ j ] < 0 ) { dist[ j ] = dist[ i ] + 1; file.push_back( j ); }
        }
    }
    std::vector<SI> ens;
    std::vector<SI> loc( n, -1 );
    for ( SI i = 0; i < n; ++i ) if ( dist[ i ] >= 0 && dist[ i ] <= N ) { loc[ i ] = SI( ens.size() ); ens.push_back( i ); }
    const SI m = SI( ens.size() );
    for ( SI i = 0; i < n; ++i ) if ( dist[ i ] == N + 1 ) { loc[ i ] = SI( ens.size() ); ens.push_back( i ); }
    const SI M = SI( ens.size() );
    bi.taille = m;
    bi.n_ring = M - m;
    // LA STRUCTURE EN FOYERS. La boule est-elle un bloc, ou une poussiere de petits amas ? C'est
    // ce qui decide s'il faut un solveur couple sur des milliers d'inconnues ou une collection de
    // problemes de trente inconnues, chacun resoluble sans structure d'acceleration.
    {
        std::vector<char> vu( m, 0 );
        std::vector<SI> pile;
        for ( SI q = 0; q < m; ++q ) {
            if ( vu[ q ] ) continue;
            SI taille = 0;
            pile.assign( 1, q );
            vu[ q ] = 1;
            while ( ! pile.empty() ) {
                const SI r = pile.back();
                pile.pop_back();
                ++taille;
                for ( SI e = Lc.row[ ens[ r ] ]; e < Lc.row[ ens[ r ] + 1 ]; ++e ) {
                    const SI j = Lc.col[ e ];
                    if ( loc[ j ] >= 0 && loc[ j ] < m && ! vu[ loc[ j ] ] ) { vu[ loc[ j ] ] = 1; pile.push_back( loc[ j ] ); }
                }
            }
            ++bi.nb_amas;
            bi.amas_max = std::max( bi.amas_max, taille );
        }
    }
    bi.ball_try = 1e300; bi.ring_try = 1e300;
    for ( SI q = 0; q < m; ++q ) bi.ball_try = std::min( bi.ball_try, a_cible[ ens[ q ] ] / nw.nu[ ens[ q ] ] );
    for ( SI q = m; q < M; ++q ) bi.ring_try = std::min( bi.ring_try, a_cible[ ens[ q ] ] / nw.nu[ ens[ q ] ] );
    if ( M == m ) bi.ring_try = 1;

    // ---- 3. LA MACHINERIE LOCALE
    std::vector<TF> w = w0, a( M ), dia( M ), u( M ), r( M );
    std::vector<SI> row( M + 1 ), col;
    std::vector<TF> cc;
    SI fuites = 0;
    auto mesure_locale = [ & ]( bool avec_facettes ) {
        pd.set_weights( w.data(), nw.par );
        if ( avec_facettes ) { row.assign( M + 1, 0 ); col.clear(); cc.clear(); dia.assign( M, TF( 0 ) ); }
        typename PD::Cell cel;
        for ( SI q = 0; q < M; ++q ) {
            const SI id = ens[ q ];
            a[ q ] = 0;
            if ( ! pd.cellule( rang[ id ], cel ) ) continue;
            a[ q ] = nw.rho->mesure( cel, [ & ]( int j, TF mes ) {
                if ( ! avec_facettes ) return;
                if ( q < m && loc[ j ] < 0 ) ++fuites;   // la boule touche hors de la zone surveillee
                TF d2 = 0;
                for ( int d = 0; d < PD::dim; ++d ) { const TF e = nw.P[ d ][ j ] - nw.P[ d ][ id ]; d2 += e * e; }
                if ( ! ( d2 > 0 ) ) return;
                const TF cij = mes / ( 2 * std::sqrt( d2 ) );
                dia[ q ] += cij;
                if ( loc[ j ] >= 0 ) { col.push_back( loc[ j ] ); cc.push_back( cij ); ++row[ q + 1 ]; }
            } );
            ++bi.nb_cel;
        }
        if ( avec_facettes ) {
            for ( SI q = 0; q < M; ++q ) row[ q + 1 ] += row[ q ];
            for ( SI q = 0; q < M; ++q ) if ( ! ( dia[ q ] > 0 ) ) dia[ q ] = 1;
        }
    };
    auto phi = [ & ]() {
        TF s = 0;
        for ( SI q = 0; q < M; ++q ) { const TF g = bar_g( a[ q ] / nw.nu[ ens[ q ] ] ); s += g * g; }
        return s;
    };
    auto plancher = [ & ]() {
        TF v = 1e300;
        for ( SI q = 0; q < M; ++q ) v = std::min( v, a[ q ] / nw.nu[ ens[ q ] ] );
        return v;
    };
    auto Lx = [ & ]( const std::vector<TF> &x, std::vector<TF> &y ) {
        for ( SI q = 0; q < M; ++q ) {
            TF s = q < m ? dia[ q ] * x[ q ] : TF( 0 );
            for ( SI e = row[ q ]; e < row[ q + 1 ]; ++e ) if ( col[ e ] < m ) s -= cc[ e ] * x[ col[ e ] ];
            y[ q ] = s;
        }
    };
    auto Ltz = [ & ]( const std::vector<TF> &z, std::vector<TF> &y ) {
        for ( SI p = 0; p < m; ++p ) {
            TF s = dia[ p ] * z[ p ];
            for ( SI e = row[ p ]; e < row[ p + 1 ]; ++e ) s -= cc[ e ] * z[ col[ e ] ];
            y[ p ] = s;
        }
    };
    std::vector<TF> t1( M ), t2( M ), rhs( m ), d( m ), pcg( m ), Ap( m ), zz( m ), rr( m ), jac( m ), w_sauve;

    // LA TANGENTE DU SOUS-PROBLEME. Extrapoler l'interieur le long de `d` ( la direction de Newton
    // GLOBALE ) n'est pas la bonne tangente : quand le bord avance de `db`, l'interieur qui garde
    // ses masses suit `L_PP dw = - L_(P,bord) db`, c'est-a-dire `dw_p = sum_(j au bord) c_pj db_j`
    // au second membre. C'est une resolution de plus sur `L_PP` -- SPD, Dirichlet, bien mieux
    // conditionnee que les equations normales de Gauss-Newton -- et elle doit rendre le premier pas
    // de chaque sous-pas presque juste.
    std::vector<TF> tg( m ), rhs_t( m );
    auto tangente = [ & ]( TF db_ech ) {                 // `db_ech` : l'increment du bord, en fraction de `F d`
        for ( SI p = 0; p < m; ++p ) {
            TF acc = 0;
            for ( SI e = row[ p ]; e < row[ p + 1 ]; ++e )
                if ( col[ e ] >= m ) acc += cc[ e ] * db_ech * F * dn[ ens[ col[ e ] ] ];
            rhs_t[ p ] = acc;
        }
        tg.assign( m, TF( 0 ) );
        std::vector<TF> r2( rhs_t ), z2( m ), p2( m ), Ap2( m );
        TF n0 = 0;
        for ( TF v : r2 ) n0 += v * v;
        if ( ! ( n0 > 0 ) ) return;
        TF rz = 0;
        for ( SI p = 0; p < m; ++p ) { z2[ p ] = r2[ p ] / dia[ p ]; rz += r2[ p ] * z2[ p ]; }
        p2 = z2;
        for ( int k = 0; k < o.local_cg; ++k ) {
            for ( SI p = 0; p < m; ++p ) {               // `L_PP p2`
                TF acc = dia[ p ] * p2[ p ];
                for ( SI e = row[ p ]; e < row[ p + 1 ]; ++e ) if ( col[ e ] < m ) acc -= cc[ e ] * p2[ col[ e ] ];
                Ap2[ p ] = acc;
            }
            TF pAp = 0;
            for ( SI p = 0; p < m; ++p ) pAp += p2[ p ] * Ap2[ p ];
            if ( ! ( pAp > 0 ) ) break;
            const TF al = rz / pAp;
            TF nr = 0;
            for ( SI p = 0; p < m; ++p ) { tg[ p ] += al * p2[ p ]; r2[ p ] -= al * Ap2[ p ]; nr += r2[ p ] * r2[ p ]; }
            if ( nr <= TF( 1e-14 ) * n0 ) break;
            TF rz2 = 0;
            for ( SI p = 0; p < m; ++p ) { z2[ p ] = r2[ p ] / dia[ p ]; rz2 += r2[ p ] * z2[ p ]; }
            const TF be = rz2 / rz;
            rz = rz2;
            for ( SI p = 0; p < m; ++p ) p2[ p ] = z2[ p ] + be * p2[ p ];
        }
    };

    /// minimise `Phi` sur l'interieur, bord fige. Rend `false` si une cellule passe sous le plancher.
    auto resout_interieur = [ & ]() {
        mesure_locale( true );
        if ( plancher() <= o.local_eps ) return false;
        TF ph = phi();
        for ( int it = 0; it < o.local_maxit; ++it ) {
            for ( SI q = 0; q < M; ++q ) {
                const TF x = a[ q ] / nw.nu[ ens[ q ] ];
                r[ q ] = bar_g( x );
                u[ q ] = bar_gp( x ) / nw.nu[ ens[ q ] ];
            }
            for ( SI q = 0; q < M; ++q ) t1[ q ] = u[ q ] * r[ q ];
            Ltz( t1, rhs );
            for ( SI p = 0; p < m; ++p ) rhs[ p ] = -rhs[ p ];
            TF nr0 = 0;
            for ( TF v : rhs ) nr0 += v * v;
            if ( ! ( nr0 > 0 ) ) break;
            for ( SI p = 0; p < m; ++p ) {
                TF s = u[ p ] * u[ p ] * dia[ p ] * dia[ p ];
                for ( SI e = row[ p ]; e < row[ p + 1 ]; ++e ) s += u[ col[ e ] ] * u[ col[ e ] ] * cc[ e ] * cc[ e ];
                jac[ p ] = s > 0 ? s : TF( 1 );
            }
            auto A = [ & ]( const std::vector<TF> &x, std::vector<TF> &y ) {
                Lx( x, t1 );
                for ( SI q = 0; q < M; ++q ) t2[ q ] = u[ q ] * u[ q ] * t1[ q ];
                Ltz( t2, y );
            };
            d.assign( m, TF( 0 ) );
            rr = rhs;
            TF rz = 0;
            for ( SI p = 0; p < m; ++p ) { zz[ p ] = rr[ p ] / jac[ p ]; rz += rr[ p ] * zz[ p ]; }
            pcg = zz;
            for ( int k = 0; k < o.local_cg; ++k ) {
                A( pcg, Ap );
                TF pAp = 0;
                for ( SI p = 0; p < m; ++p ) pAp += pcg[ p ] * Ap[ p ];
                if ( ! ( pAp > 0 ) ) break;
                const TF al = rz / pAp;
                TF nr = 0;
                for ( SI p = 0; p < m; ++p ) { d[ p ] += al * pcg[ p ]; rr[ p ] -= al * Ap[ p ]; nr += rr[ p ] * rr[ p ]; }
                if ( nr <= TF( 1e-16 ) * nr0 ) break;
                TF rz2 = 0;
                for ( SI p = 0; p < m; ++p ) { zz[ p ] = rr[ p ] / jac[ p ]; rz2 += rr[ p ] * zz[ p ]; }
                const TF be = rz2 / rz;
                rz = rz2;
                for ( SI p = 0; p < m; ++p ) pcg[ p ] = zz[ p ] + be * pcg[ p ];
            }
            w_sauve = w;
            bool pris = false;
            for ( TF t = 1; t >= TF( 1e-8 ); t /= 2 ) {
                for ( SI p = 0; p < m; ++p ) w[ ens[ p ] ] = w_sauve[ ens[ p ] ] + t * d[ p ];
                mesure_locale( false );
                const TF p2 = phi();
                if ( p2 < ph * ( 1 - 1e-12 ) ) { ph = p2; pris = true; break; }
            }
            ++bi.iter;
            if ( ! pris ) { w = w_sauve; mesure_locale( true ); break; }
            mesure_locale( true );
        }
        return plancher() > o.local_eps;
    };

    // ---- 4. LA CONTINUATION SUR LE BORD : `s` de 0 a 1, et a `s = 0` IL N'Y A RIEN A RESOUDRE.
    TF s = 0, ds = TF( 1 ) / 8;
    std::vector<TF> w_ok = w0;
    for ( int e = 0; e < 400 && s < 1 - 1e-9; ++e ) {
        const TF s2 = std::min( s + ds, TF( 1 ) );
        // LE DEMARRAGE A CHAUD. Il ne faut surtout pas laisser l'interieur fige pendant que le bord
        // avance : ca recree une marche differentielle a la frontiere de la boule, qui tue une
        // cellule avant que le solveur ne reagisse ( mesure : sous-pas admissible 0.002, en
        // falaise ). Deux predicteurs : le long de `d` ( la direction de Newton globale ), ou LA
        // TANGENTE DU SOUS-PROBLEME, qui est la bonne.
        if ( o.local_tangente && s > 0 ) tangente( s2 - s );
        else tg.assign( m, TF( 0 ) );
        for ( SI i = 0; i < n; ++i )
            w[ i ] = loc[ i ] >= 0 && loc[ i ] < m
                   ? ( o.local_tangente && s > 0 ? w_ok[ i ] + tg[ loc[ i ] ]
                                                 : w_ok[ i ] + ( s2 - s ) * F * dn[ i ] )
                   : w0[ i ] + s2 * F * dn[ i ];
        const SI f0 = fuites;
        const bool ok = resout_interieur();
        if ( o.local_trace )
            std::printf( "      cont N=%d F=%.4g : s %.4f %s ( plancher %.3e, %d fuites )\n",
                         N, double( F ), double( s2 ), ok ? "PASSE" : "refuse", double( plancher() ), int( fuites - f0 ) );
        if ( ok ) { s = s2; w_ok = w; ds = std::min( ds * 2, TF( 1 ) ); }
        else { ds /= 2; if ( ds < TF( 1e-4 ) ) break; }
    }
    bi.s = s;
    bi.fuites = fuites;
    w = w_ok;
    mesure_locale( true );
    bi.ball_fin = 1e300; bi.ring_fin = 1e300;
    for ( SI q = 0; q < m; ++q ) bi.ball_fin = std::min( bi.ball_fin, a[ q ] / nw.nu[ ens[ q ] ] );
    for ( SI q = m; q < M; ++q ) bi.ring_fin = std::min( bi.ring_fin, a[ q ] / nw.nu[ ens[ q ] ] );
    if ( M == m ) bi.ring_fin = 1;
    bi.phi1 = phi();
    w_out = w_ok;
    return bi;
}

/// L'ETUDE DU § 14 : le pas `w0 + F d`, puis la reparation. Mesure en plus la sante du bord AU
/// DEPART, qui est ce qu'on annonce « par construction ».
template<class PD, class Rho>
BilanLocal local_barriere( Newton<PD,Rho> &nw, const std::vector<TF> &w0, const std::vector<TF> &dn,
                           const std::vector<SI> &rang, int N, TF F, const Opts &o,
                           std::vector<TF> &w_out ) {
    const SI n = nw.pd.n;
    std::vector<TF> w_try( n ), a_try, a_0;
    std::vector<Facette> fa_try, fa_0;
    for ( SI i = 0; i < n; ++i ) w_try[ i ] = w0[ i ] + F * dn[ i ];
    nw.mesures_et_facettes( w_try, a_try, fa_try );
    nw.mesures_et_facettes( w0, a_0, fa_0 );
    BilanLocal bi = repare_local( nw, rang, w_try, a_try, fa_try, N, o, w_out );
    bi.F = F;
    return bi;
}

// =====================================================================================
// NEWTON AVEC RELEVEMENT LOCAL ( `--newton-releve`, § 15 )
//
// L'amortissement classique borne le pas par « AUCUNE cellule sous `eps` ». On le remplace par
//
//      au plus `ratio x n` cellules sous `eps`,  ET  AUCUNE sous `eps_mort`
//
// -- on avance donc bien plus loin, en acceptant un nombre BORNE de cellules malades, puis on les
// repare : quelques anneaux autour d'elles, coalesces par le parcours en largeur, bord impose, et
// l'interieur resolu sur l'objectif barriere ( § 14 ). Si la reparation echoue, on redescend le
// pas -- ce sont les « poids imposes intermediaires » -- jusqu'a ce qu'elle passe.
//
// LA SECONDE CONDITION N'EST PAS UN DETAIL. Une cellule de masse exactement nulle a toutes ses
// `c_ij` nulles, donc une ligne de hessienne neutralisee : aucune methode passant par `L` ne peut
// la ranimer ( § 13.1 ), et la reparation echoue a coup sur. Le § 14 a mesure la frontiere, et elle
// est franche : reparable a `F = 0.033`, hors de portee a `0.036`.
//
// Le pas n'est accepte que si le MERITE DESCEND apres reparation -- sans quoi on perdrait la
// garantie de l'amortissement pour un point « sain » mais plus mauvais.
// =====================================================================================

struct StatsReleve {
    int nb_iter = 0, nb_diag = 0, nb_recul = 0, nb_repare = 0, nb_echec = 0;
    SI  nb_cel = 0;                ///< cellules calculees par les reparations
    SI  nb_amas = 0, nb_abandon = 0, ech_fenetre = 0, ech_inconnue = 0, ech_couronne = 0, nb_dump = 0;
    SI  nb_bis = 0, cel_bis = 0;   ///< bissections de raffinement de `U`, et leur cout
    SI  nb_remonte = 0;            ///< retours au buffer tournant : ne devrait jamais arriver
    double lam_min = 1e300, lam_max = 0;
    double t_geo = 0, t_asm = 0, t_cg = 0, t_ref = 0, t_ver = 0, t_bis = 0, t_pre = 0;
    SI     n_geo = 0, n_asm = 0, n_ls_ech = 0, n_geo_ech = 0, n_sous = 0;
    // D'OU VIENNENT LES REFUS. Par construction il ne devrait pas y en avoir : la continuation ne
    // valide un sous-pas que si l'etat est sain, et au pire elle rend `s = 0`, l'etat de depart.
    // Quatre causes possibles, qu'on separe au lieu de les supposer.
    SI     ref_s0 = 0;             ///< `s = 0` : la continuation n'a pas pu avancer d'un pouce
    SI     ref_res = 0;            ///< le residu n'a pas baisse, alors que l'etat est sain
    SI     ref_mal_dans = 0;       ///< une cellule malade DANS la zone reparee : la reparation ment
    SI     ref_mal_cour = 0;       ///< ... ou dans la COURONNE, dont on ne resout pas les poids
    SI     ref_mal_hors = 0;       ///< une cellule malade HORS d'elle : la zone etait mal choisie
    SI     ref_contradiction = 0;  ///< refus ou la mesure LOCALE annonçait pourtant un etat sain
    SI     ref_deja_morte = 0;     ///< ... et la cellule fautive etait DEJA morte avant la reparation
    double pire_ratio = 1e300;     ///< `a / nu` de la pire cellule refusee, pour comparer les seuils
    double s_cumul = 0;            ///< somme des pas effectivement pris apres reparation
    std::vector<double> suite_F;   ///< LA SEQUENCE DES COEFFICIENTS DE RELAXATION, une par iteration
    SI  nb_malades = 0;            ///< cellules malades reparees, en tout
    double t_rep = 0;
    TF  reste = 0;
    const char *fin = "?";
};

template<class PD, class Rho>
bool resout_releve( Newton<PD,Rho> &nw, Lineaire &lin, std::vector<TF> &w, const Opts &o, StatsReleve &st ) {
    const SI n = nw.pd.n;
    std::vector<SI> rang( n );
    for ( SI k = 0; k < n; ++k ) rang[ nw.pd.ids[ k ] ] = k;
    std::vector<TF> a, a2, b, d, w2, w3;
    std::vector<char> zone_rep;
    std::vector<TF> a_pur;
    std::vector<Facette> fa, fa2;
    Laplacien L;

    nw.mesures_et_facettes( w, a, fa );
    ++st.nb_diag;
    TF eps = 0;
    {
        TF am = a[ 0 ], nm = nw.nu[ 0 ];
        for ( SI i = 0; i < n; ++i ) { am = std::min( am, a[ i ] ); nm = std::min( nm, nw.nu[ i ] ); }
        eps = TF( 0.5 ) * std::min( nm, am );
    }
    // « MALADE » EN RATIO. Le plancher absolu `eps = 0.5 min( min nu, min a )` est severe pour les
    // cellules de grosse cible et laxiste pour les petites ; le ratio `a / nu` les traite toutes
    // pareil. On garde le plancher absolu par defaut, faute d'avoir mesure lequel vaut mieux.
    const bool eps_en_ratio = o.releve_ratio_eps > 0;
    auto malade = [ & ]( SI i, TF ai ) { return eps_en_ratio ? ai < o.releve_ratio_eps * nw.nu[ i ] : ai < eps; };
    const TF eps_mort = o.releve_mort * nw.nu[ 0 ];
    const SI max_malades = SI( o.releve_ratio * n );

    // LE BUFFER TOURNANT DES ETATS SAINS. Les etats acceptes sont verifies sains, donc en principe on
    // ne repart jamais d'un etat degenere -- mais « en principe » n'est pas une garantie, et le jour
    // ou ca arrive la boucle n'a aucun recours : une cellule morte n'a plus de facettes, donc plus de
    // voisins, donc plus de reparation possible ( § 15.9 ). On garde donc les quatre derniers etats
    // sains ; si l'etat courant est degenere on y revient et on repart d'un pas moitie moindre.
    struct Etat { std::vector<TF> w, a; std::vector<Facette> fa; TF pas = 1; bool plein = false; };
    Etat anneau[ 4 ];
    int i_anneau = 0;

    for ( int it = 0; it < o.newton.maxit; ++it ) {
        {
            TF am = a[ 0 ];
            for ( SI i = 0; i < n; ++i ) am = std::min( am, a[ i ] );
            if ( am > 0 ) {                              // sain : on le range
                Etat &e = anneau[ i_anneau ];
                e.w = w; e.a = a; e.fa = fa; e.plein = true;
                i_anneau = ( i_anneau + 1 ) % 4;
            } else {                                     // degenere : on remonte le plus recent sain
                int k = -1;
                for ( int q = 1; q <= 4; ++q ) {
                    const int r = ( i_anneau - q + 8 ) % 4;
                    if ( anneau[ r ].plein ) { k = r; break; }
                }
                if ( k < 0 ) { st.fin = "ETAT DEGENERE SANS RECOURS"; return false; }
                w = anneau[ k ].w; a = anneau[ k ].a; fa = anneau[ k ].fa;
                anneau[ k ].pas /= 2;
                ++st.nb_remonte;
            }
        }
        TF pire = 0, mer = 0;
        for ( SI i = 0; i < n; ++i ) {
            pire = std::max( pire, std::fabs( nw.nu[ i ] - a[ i ] ) / nw.nu[ i ] );
            mer += ( nw.nu[ i ] - a[ i ] ) * ( nw.nu[ i ] - a[ i ] );
        }
        mer = std::sqrt( mer );
        st.reste = pire;
        if ( pire <= o.newton.tol ) { st.fin = "CONVERGE"; return true; }
        ++st.nb_iter;

        // LE TEMPS DE L'ALGEBRE GLOBALE N'ETAIT PAS COMPTE DANS CE CHEMIN, et il est le premier
        // poste : la ligne « lin 0.00 » du relevement etait un trou dans la comptabilite, pas un
        // solveur gratuit. On le verse dans les memes compteurs que la boucle de Newton nue.
        {
            const double t_a0 = now();
            L.assemble( n, fa );
            nw.st.t_asm += now() - t_a0;
        }
        b.resize( n );
        for ( SI i = 0; i < n; ++i ) b[ i ] = nw.nu[ i ] - a[ i ];
        {
            const double t_l0 = now();
            const bool ok_lin = lin.resout( L, b, d );
            nw.st.t_lin += now() - t_l0;
            if ( ! ok_lin ) { st.fin = "SOLVEUR LINEAIRE EN ECHEC"; return false; }
        }

        // ---- LE PREDICTEUR `U`, GRATUIT ( `--diag-u` )
        //
        // Le flux d'aire qui sort de la cellule `i` par la facette `j` vaut exactement
        // `c_ij ( d_j - d_i )` -- c'est le terme du laplacien, deja assemble. Le temps que cette
        // facette met a consommer toute la cellule est donc `a_i / c_ij ( d_j - d_i )`, et
        //
        //      U_i = a_i / max_j [ c_ij ( d_j - d_i ) ]     sur les `j` qui font perdre de l'aire
        //
        // est une estimation du pas admissible de la cellule `i`. AUCUNE geometrie supplementaire :
        // une passe sur les aretes. ( Coherence : la somme des flux vaut `( L d )_i = nu_i - a_i`. )
        //
        // LA COURBE DES AMAS se lit alors d'une seule passe d'UNION-FIND : on insere les cellules par
        // `U_i` croissant en fusionnant avec les voisines deja inserees, et a chaque insertion on
        // connait la taille du plus gros amas. Un tri et `O( n alpha( n ) )` donnent donc, POUR TOUS
        // LES `F` A LA FOIS, le nombre de malades, le nombre d'amas et la taille du plus gros -- pas
        // besoin d'un essai par `F`, ni d'un thread par proposition.
        TF t_depart = 1;
        if ( o.diag_u || o.releve_u ) {
            std::vector<TF> U( n, 1e300 );
            // LA SOMME DES FLUX, PAS LE MAXIMUM. Prendre `max_j` revient a supposer qu'UNE SEULE
            // facette devore la cellule ; en realite plusieurs la mangent en meme temps. La somme des
            // flux sortants est donc strictement plus conservative, se calcule dans la MEME boucle,
            // et corrige une partie des sur-estimations -- la ou un plafond de confiance sur `F` ne
            // ferait que les masquer.
            for ( SI i = 0; i < n; ++i ) {
                TF sor = 0;
                for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) {
                    const TF fl = L.c[ e ] * ( d[ L.col[ e ] ] - d[ i ] );
                    if ( fl <= 0 ) continue;
                    sor = o.u_max ? std::max( sor, fl ) : sor + fl;
                }
                U[ i ] = sor > 0 ? a[ i ] / sor : TF( 1e300 );
            }
            // LE RAFFINEMENT PAR BISSECTION, SOUS L'HORIZON `beta`.
            //
            // Le `U` bon marche n'est pas fiablement conservatif -- le flux est evalue en `t = 0` et
            // la geometrie accelere quand des facettes disparaissent -- et la mesure le dit : 65
            // reculs pour 123 iterations a `n = 2e4`, donc une sur-estimation une fois sur deux. On
            // calcule donc la VRAIE limite, par bissection, pour les seules cellules qui comptent.
            //
            // C'EST LA QUE `beta` SERT, et c'est son seul role : borner le nombre de candidates. Une
            // bissection coute des calculs de cellule ; `beta` petit en laisse peu, et donne en plus
            // une meilleure evaluation la ou elle compte, l'intervalle a bissecter etant plus court.
            // Avec beaucoup de diracs, partir a 1 n'a de toute facon aucun sens.
            SI n_bis = 0, cel_bis = 0;
            if ( o.releve_beta > 0 && nw.rho ) {
                std::vector<SI> cand;
                for ( SI i = 0; i < n; ++i ) if ( U[ i ] < o.releve_beta ) cand.push_back( i );
                n_bis = SI( cand.size() );
                if ( n_bis ) {
                    OptionsLimites ol;
                    ol.niveau = eps;
                    ol.horizon = o.releve_beta;
                    ol.tol = o.releve_bis_tol;
                    ol.global = false;                   // on veut le `alpha` de CHAQUE candidate
                    ol.log_ech = true;                   // les pas admissibles sont minuscules a grand `n`
                    ol.max_tours = 32;                   // le geometrique en demande plus, et ils sont petits
                    std::vector<LimiteCellule> lim;
                    nw.pd.set_weights( w.data(), nw.par );
                    limites_masse( nw.pd, nw.P, w, d, nw.par, ol, lim,
                                   Voisinage{ L.row.data(), L.col.data() }, cand,
                                   [ & ]( const typename PD::Cell &cel ) { return nw.rho->mesure( cel, []( int, TF ) {} ); } );
                    for ( SI i : cand ) {
                        cel_bis += lim[ i ].tours;
                        if ( lim[ i ].etat == LimiteCellule::VIDE_AU_DEPART ) { U[ i ] = 0; continue; }
                        U[ i ] = lim[ i ].alpha;         // HORIZON compris : la cellule tient jusqu'a `beta`
                    }
                    st.nb_cel += cel_bis; st.nb_bis += n_bis; st.cel_bis += cel_bis;
                }
            }

            std::vector<SI> ord( n );
            for ( SI i = 0; i < n; ++i ) ord[ i ] = i;
            std::sort( ord.begin(), ord.end(), [ & ]( SI x, SI y ) { return U[ x ] < U[ y ]; } );

            // L'ETALEMENT DES POIDS PAR AMAS. La condition mesuree au § 15.12 est que l'amas echoue
            // quand l'etalement de `w` a travers lui depasse ~2 dist^2 : le champ y est trop raide
            // pour l'espacement des germes. A `F` donne, `w_F = w + F d`, donc
            //
            //      etalement( F )  <=  ( w_max - w_min ) + F ( d_max - d_min )
            //
            // qui est CROISSANT en `F` -- donc exploitable dans le meme balayage croissant, en
            // maintenant par amas les min/max de `w` et de `d`, et le plus petit `dist^2` de ses
            // aretes. ( C'est une borne : les extremes de `w` et de `d` ne sont pas forcement sur la
            // meme cellule. )
            std::vector<TF> wlo( n ), whi( n ), dlo( n ), dhi( n ), d2min( n, 1e300 );
            for ( SI i = 0; i < n; ++i ) { wlo[ i ] = whi[ i ] = w[ i ]; dlo[ i ] = dhi[ i ] = d[ i ]; }
            std::vector<SI> pere( n ), taille( n, 1 );
            std::vector<char> dedans( n, 0 );
            for ( SI i = 0; i < n; ++i ) pere[ i ] = i;
            std::function<SI(SI)> trouve = [ & ]( SI x ) { while ( pere[ x ] != x ) { pere[ x ] = pere[ pere[ x ] ]; x = pere[ x ]; } return x; };
            SI gros = 0, nb_amas = 0;
            TF F_raid = 1e300;                           // le `F` ou un amas depasse le plafond d'etalement
            const SI paliers[ 6 ] = { 8, 16, 32, 64, 128, 256 };
            TF F_pour[ 6 ] = { 0, 0, 0, 0, 0, 0 };
            SI k_pal = 0;
            for ( SI q = 0; q < n; ++q ) {
                const SI i = ord[ q ];
                if ( ! ( U[ i ] < 1e299 ) ) break;
                dedans[ i ] = 1; ++nb_amas;
                for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) {
                    const SI j = L.col[ e ];
                    if ( ! dedans[ j ] ) continue;
                    {
                        TF dd = 0;
                        for ( int q = 0; q < 2; ++q ) { const TF t = nw.P[ q ][ j ] - nw.P[ q ][ i ]; dd += t * t; }
                        const SI r0 = trouve( i );
                        d2min[ r0 ] = std::min( d2min[ r0 ], dd );
                    }
                    const SI ri = trouve( i ), rj = trouve( j );
                    if ( ri == rj ) continue;
                    pere[ ri ] = rj; taille[ rj ] += taille[ ri ]; --nb_amas;
                    wlo[ rj ] = std::min( wlo[ rj ], wlo[ ri ] ); whi[ rj ] = std::max( whi[ rj ], whi[ ri ] );
                    dlo[ rj ] = std::min( dlo[ rj ], dlo[ ri ] ); dhi[ rj ] = std::max( dhi[ rj ], dhi[ ri ] );
                    d2min[ rj ] = std::min( d2min[ rj ], d2min[ ri ] );
                    gros = std::max( gros, taille[ rj ] );
                }
                gros = std::max( gros, taille[ trouve( i ) ] );
                while ( k_pal < 6 && gros > paliers[ k_pal ] ) {   // on vient de depasser ce palier
                    F_pour[ k_pal ] = U[ i ];
                    ++k_pal;
                }
                if ( o.releve_raideur > 0 && ! ( F_raid < 1e299 ) ) {
                    const SI r0 = trouve( i );
                    const TF sw = whi[ r0 ] - wlo[ r0 ], sd = dhi[ r0 ] - dlo[ r0 ];
                    const TF plaf = o.releve_raideur * ( d2min[ r0 ] < 1e299 ? d2min[ r0 ] : TF( 0 ) );
                    if ( plaf > 0 && sw + U[ i ] * sd > plaf ) F_raid = U[ i ];
                }
            }
            for ( ; k_pal < 6; ++k_pal ) F_pour[ k_pal ] = 1;

            // LE PAS DE DEPART EST CELUI QUE `U` DESIGNE, pas `1`. La dyadique cherchait a tatons, un
            // DIAGRAMME COMPLET par essai ; l'union-find rend la meme information d'avance. Mesure :
            // le `F` predit pour un amas <= 8 tombe systematiquement juste au-dessus du dyadique que
            // la boucle finissait par retenir, a 20 % pres et souvent bien mieux. `U` sur-estime
            // parfois, donc on GARDE la dyadique en repli -- mais elle ne sert plus qu'aux exceptions.
            if ( o.releve_u ) {
                TF f = 1;
                for ( int q = 0; q < 6; ++q ) if ( paliers[ q ] >= o.releve_mal_max ) { f = F_pour[ q ]; break; }
                // la table est en paliers fixes : on prend le premier qui couvre le seuil demande
                if ( o.releve_mal_max <= paliers[ 0 ] ) f = F_pour[ 0 ];
                if ( o.releve_raideur > 0 && F_raid < 1e299 ) f = std::min( f, F_raid );
                t_depart = std::min( TF( 1 ), std::max( f, o.newton.t_min ) );
            }
            if ( o.diag_u )
                std::printf( "    U  it %2d : min %.4e  |  %d bissectees ( %d cellules )  |  amas <= 8 %.4e,"
                             " 16 %.4e, 32 %.4e, 64 %.4e, 128 %.4e, 256 %.4e  |  depart %.4e\n",
                             it, double( U[ ord[ 0 ] ] ), int( n_bis ), int( cel_bis ),
                             double( F_pour[ 0 ] ), double( F_pour[ 1 ] ), double( F_pour[ 2 ] ),
                             double( F_pour[ 3 ] ), double( F_pour[ 4 ] ), double( F_pour[ 5 ] ),
                             double( t_depart ) );
        }

        // ---- LE PAS : le plus grand `t` dyadique qui garde le nombre de malades sous le plafond
        //      et personne au-dessous de `eps_mort`. Puis la reparation ; si elle echoue, on
        //      redescend `t` ( les « poids imposes intermediaires » ).
        bool pris = false;
        w2.resize( n );
        for ( TF t = t_depart; t >= o.newton.t_min; t /= 2 ) {
            for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
            w2[ 0 ] = w[ 0 ];
            nw.mesures_et_facettes( w2, a2, fa2 );
            ++st.nb_diag;
            SI nmal = 0;
            TF amin = a2[ 0 ], m2 = 0;
            for ( SI i = 0; i < n; ++i ) {
                nmal += malade( i, a2[ i ] );
                amin = std::min( amin, a2[ i ] );
                m2 += ( nw.nu[ i ] - a2[ i ] ) * ( nw.nu[ i ] - a2[ i ] );
            }
            m2 = std::sqrt( m2 );
            // LE RATIO EST LE PARAMETRE, PAS `F`. Le plancher `eps_mort` n'a de sens que pour la
            // reparation PAR SAUT, qui ne sait pas ressusciter une cellule morte ; la reparation
            // par CONTINUATION ne rencontre jamais l'etat mort, donc elle n'en a pas besoin.
            const bool mort_interdit = ! ( o.local_cont || o.local_amas );
            if ( ( mort_interdit && amin <= eps_mort ) || nmal > max_malades ) { ++st.nb_recul; continue; }

            if ( nmal == 0 ) {                           // le pas classique suffit
                if ( m2 < mer ) {
                    w.swap( w2 ); a.swap( a2 ); fa.swap( fa2 ); pris = true; st.suite_F.push_back( double( t ) );
                    if ( o.newton.trace )
                        std::printf( "    it %2d  F %.3e  ( pas nu )                            "
                                     "  |r| %.6e -> %.6e  ( x %.4f )\n",
                                     it, double( t ), double( mer ), double( m2 ),
                                     double( mer ) > 0 ? double( m2 / mer ) : 0.0 );
                    break;
                }
                ++st.nb_recul;
                continue;
            }
            if ( ! o.releve_actif ) { ++st.nb_recul; continue; }

            // ---- LA REPARATION
            const double t0 = now();
            BilanLocal bl;
            if ( o.local_amas ) {
                w3.assign( n, TF( 0 ) );
                a_pur = a2;                              // l'etat du pas NU, avant qu'on y touche
                StatsAmas sa = repare_amas( nw, w, d, a2, fa2, t, o.releve_anneaux, o, w3, eps, &zone_rep, &fa );
                bl.s = sa.s; bl.taille = sa.n_inconnues; bl.nb_cel = sa.nb_cel;
                bl.nb_amas = sa.nb_amas; bl.amas_max = sa.taille_max; bl.iter = sa.iter;
                bl.ball_fin = sa.nb_amas ? sa.ball_fin : TF( 1 );   // CE QUE LA MESURE LOCALE ANNONCE
                st.nb_diag += sa.nb_refresh;             // chaque rafraichissement est un diagramme
                st.nb_amas += sa.nb_amas;
                st.nb_abandon += sa.nb_abandon;
                st.ech_fenetre += sa.ech_fenetre; st.ech_inconnue += sa.ech_inconnue;
                st.ech_couronne += sa.ech_couronne; st.nb_dump = sa.nb_dump;
                if ( sa.lam_max > 0 ) { st.lam_min = std::min( st.lam_min, double( sa.lam_min ) );
                                        st.lam_max = std::max( st.lam_max, double( sa.lam_max ) ); }
                st.t_geo += sa.t_geo; st.t_asm += sa.t_asm; st.t_cg += sa.t_cg;
                st.t_ref += sa.t_ref; st.t_ver += sa.t_ver;
                st.t_bis += sa.t_bis; st.t_pre += sa.t_pre;
                st.n_geo += sa.n_geo; st.n_asm += sa.n_asm;
                st.n_ls_ech += sa.n_ls_ech; st.n_geo_ech += sa.n_geo_ech; st.n_sous += sa.n_sous;
                st.s_cumul += double( sa.s ) * double( t );
            } else
                bl = o.local_cont
                   ? repare_continuation( nw, rang, w, d, t, o.releve_anneaux, o, w3 )
                   : repare_local( nw, rang, w2, a2, fa2, o.releve_anneaux, o, w3 );
            st.t_rep += now() - t0;
            st.nb_cel += bl.nb_cel;
            nw.mesures_et_facettes( w3, a2, fa2 );
            ++st.nb_diag;
            TF amin3 = a2[ 0 ], m3 = 0;
            SI nmal3 = 0, i_pire = 0;
            for ( SI i = 0; i < n; ++i ) {
                if ( a2[ i ] < amin3 ) { amin3 = a2[ i ]; i_pire = i; }
                nmal3 += malade( i, a2[ i ] );
                m3 += ( nw.nu[ i ] - a2[ i ] ) * ( nw.nu[ i ] - a2[ i ] );
            }
            m3 = std::sqrt( m3 );
            if ( amin3 > eps && m3 < mer ) {             // sain ET meilleur : on prend
                w.swap( w3 ); a.swap( a2 ); fa.swap( fa2 );
                ++st.nb_repare;
                st.nb_malades += nmal;
                pris = true;
                st.suite_F.push_back( double( t ) * ( bl.s > 0 ? double( bl.s ) : 1.0 ) );
                if ( o.newton.trace )
                    std::printf( "    it %2d  F %.3e  s %.3f  -> pas %.3e  ( %d mal, %d amas, %d inc )  "
                                 "|r| %.6e -> %.6e  ( x %.4f )  nu %.6e  ( reparation x %.4f )\n",
                                 it, double( t ), double( bl.s ), double( t ) * double( bl.s > 0 ? bl.s : TF( 1 ) ),
                                 int( nmal ), int( bl.nb_amas ), int( bl.taille ),
                                 double( mer ), double( m3 ), double( mer ) > 0 ? double( m3 / mer ) : 0.0,
                                 double( m2 ), double( m2 ) > 0 ? double( m3 / m2 ) : 0.0 );
                break;
            }
            // LA CAUSE DU REFUS, SEPAREE. Et pour le cas « malade », le rapport `a / nu` de la
            // cellule fautive face aux DEUX seuils : celui de la boucle ( `eps`, absolu ) et celui
            // de la reparation ( `local_eps`, relatif ). S'ils ne disent pas la meme chose, le refus
            // n'est pas une limite de la methode mais une incoherence de seuils.
            if ( bl.s <= 0 )          ++st.ref_s0;
            else if ( amin3 > eps )   ++st.ref_res;
            else if ( ! zone_rep.empty() && zone_rep[ i_pire ] == 1 ) ++st.ref_mal_dans;
            else if ( ! zone_rep.empty() && zone_rep[ i_pire ] == 2 ) ++st.ref_mal_cour;
            else                      ++st.ref_mal_hors;
            if ( bl.ball_fin > o.local_eps ) ++st.ref_contradiction;
            if ( ! a_pur.empty() && ! ( a_pur[ i_pire ] > 0 ) ) ++st.ref_deja_morte;
            if ( amin3 <= eps ) {
                const double rr = double( amin3 / nw.nu[ i_pire ] );
                st.pire_ratio = std::min( st.pire_ratio, rr );
            }
            if ( o.newton.trace ) {
                const char *cause = bl.s <= 0 ? "s = 0" : amin3 > eps ? "residu"
                                  : ( ! zone_rep.empty() && zone_rep[ i_pire ] ) ? "malade DANS la zone"
                                                                                 : "malade HORS zone";
                const char *ou = zone_rep.empty() ? "?" : zone_rep[ i_pire ] == 1 ? "inconnue"
                               : zone_rep[ i_pire ] == 2 ? "couronne" : "hors zone";
                // L'AIRE AVANT REPARATION tranche entre « elle etait deja condamnee par le pas » et
                // « c'est la reparation qui l'a tuee ». Sans ce chiffre on ne peut que supposer.
                const double av = double( a_pur[ i_pire ] / nw.nu[ i_pire ] );
                // LA CONTRADICTION : `ball_fin` est le plancher que la mesure LOCALE annonce, sur
                // les memes poids que la mesure globale. S'il est confortable alors que le global
                // trouve zero, la cellule locale est un SUR-ENSEMBLE -- il lui manque un coupeur, et
                // aucun compteur de facettes ne peut le voir puisqu'un coupeur manquant n'en produit
                // aucune.
                std::printf( "    it %2d  F %.3e  s %.3f  REFUS ( %s, %s )  |r| %.6e -> %.6e  nu %.6e"
                             "  --  marge LOCALE %.3e, a/nu APRES %.3e, a/nu AVANT %.3e  ( eps/nu %.3e )\n",
                             it, double( t ), double( bl.s ), cause, ou,
                             double( mer ), double( m3 ), double( m2 ),
                             double( bl.ball_fin ), double( amin3 / nw.nu[ i_pire ] ), av,
                             double( eps / nw.nu[ i_pire ] ) );
            }
            ++st.nb_echec;
            ++st.nb_recul;
        }
        if ( ! pris ) { st.fin = "STAGNATION"; return false; }
    }
    st.fin = "MAX ITERATIONS";
    return false;
}

/// L'ETUDE, au point `w0` sous la densite DEJA reglee sur la cible ( `nw.a`, `nw.fa` mesures la ).
template<class PD, class Rho>
void etude_relevement( Newton<PD,Rho> &nw, Lineaire &lin, const std::vector<TF> &w0, const Opts &o ) {
    const SI n = SI( w0.size() );
    Laplacien L;
    L.assemble( n, nw.fa );

    // ---- LES CELLULES A RELEVER, et la distance a leur groupe
    std::vector<SI> S;
    for ( SI i = 0; i < n; ++i ) if ( nw.a[ i ] < o.rel_seuil * nw.nu[ i ] ) S.push_back( i );
    const bool repli = S.empty();
    if ( repli ) {
        std::vector<SI> ord( n );
        for ( SI i = 0; i < n; ++i ) ord[ i ] = i;
        std::partial_sort( ord.begin(), ord.begin() + 16, ord.end(), [ & ]( SI x, SI y ) { return nw.a[ x ] < nw.a[ y ]; } );
        S.assign( ord.begin(), ord.begin() + 16 );
    }
    SI nmortes = 0;
    for ( SI i : S ) nmortes += ! ( nw.a[ i ] > 0 );
    std::vector<int> dist( n, -1 );
    std::vector<SI> file( S.begin(), S.end() );
    for ( SI i : S ) dist[ i ] = 0;
    for ( size_t q = 0; q < file.size(); ++q ) {         // parcours en largeur sur le graphe de Laguerre
        const SI i = file[ q ];
        if ( dist[ i ] >= o.rel_largeur_max ) continue;
        for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) {
            const SI j = L.col[ e ];
            if ( dist[ j ] < 0 ) { dist[ j ] = dist[ i ] + 1; file.push_back( j ); }
        }
    }
    TF amin = nw.a[ 0 ], numin = nw.nu[ 0 ], rmax0 = 0;
    for ( SI i = 0; i < n; ++i ) {
        amin = std::min( amin, nw.a[ i ] );
        numin = std::min( numin, nw.nu[ i ] );
        rmax0 = std::max( rmax0, std::fabs( nw.a[ i ] - nw.nu[ i ] ) / nw.nu[ i ] );
    }
    const TF eps = TF( 0.5 ) * std::min( numin, std::max( amin, TF( 0 ) ) );
    const TF r2_0 = nw.merite( nw.a );
    std::printf( "  ETUDE DU RELEVEMENT : %d cellules visees%s ( dont %d de masse NULLE : hors de portee ),"
                 " plancher eps = %.2e nu\n", int( S.size() ),
                 repli ? " ( AUCUNE sous le seuil : ce sont LES PLUS PETITES, toutes saines )" : " sous le seuil",
                 int( nmortes ), double( eps / nw.nu[ 0 ] ) );
    std::printf( "                        depart : masse min %.3e nu, |r|_2 %.3e, max|a-nu|/nu %.3e\n",
                 double( amin / nw.nu[ 0 ] ), double( r2_0 ), double( rmax0 ) );

    FILE *csv = o.rel_csv.empty() ? nullptr : std::fopen( o.rel_csv.c_str(), "a" );
    std::vector<Bilan> bilans;
    std::vector<TF> b( n ), d, dn, C( n, TF( 1 ) ), w_lev;
    for ( SI i = 0; i < n; ++i ) b[ i ] = nw.nu[ i ] - nw.a[ i ];
    const std::vector<char> tout;                        // patch vide = tout le diagramme

    // ---- 1. NEWTON, le temoin
    Bilan bn;
    if ( lin.resout( L, b, dn ) ) {
        bn = balaye( nw, w0, dn, S, eps, "newton", 0, n, csv, o.rel_cas.c_str() );
        bilans.push_back( bn );
    }

    // ---- 2. LE TEMOIN DE L'IDENTITE : les moindres carres ponderes SUR TOUT LE DIAGRAMME doivent
    //      rendre la direction de Newton, quels que soient les poids ( voir l'en-tete ). On le
    //      verifie une fois, en ecart relatif A LA CONSTANTE DE JAUGE PRES.
    if ( o.rel_identite ) {
        for ( SI i = 0; i < n; ++i ) C[ i ] = dist[ i ] >= 0 && dist[ i ] <= 2 ? TF( 1000 ) : TF( 1 );
        const int it = resout_pondere( L, C, b, tout, d, o.rel_cg, TF( 1e-10 ) );
        TF mo = 0, ec = 0, no = 0;
        for ( SI i = 0; i < n; ++i ) mo += d[ i ] - dn[ i ];
        mo /= n;
        for ( SI i = 0; i < n; ++i ) { const TF e = d[ i ] - dn[ i ] - mo; ec = std::max( ec, std::fabs( e ) ); no = std::max( no, std::fabs( dn[ i ] ) ); }
        std::printf( "  identite : moindres carres GLOBAUX avec kappa = 1000 contre Newton -- ecart relatif %.2e"
                     " ( %d iterations de CG )%s\n", double( ec / no ), it,
                     it < 0 ? "  [ CG NON CONVERGE : l'ecart ne prouve rien ]" : "" );
    }

    // ---- LA REFERENCE, quand on veut trancher : Newton mene a convergence depuis le depart.
    int diag_ref = -1;
    std::string fin_ref;
    if ( o.rel_solve ) {
        const std::vector<TF> sauve_a = nw.a, sauve_w = w0;
        const std::vector<Facette> sauve_fa = nw.fa;
        NewtonStats sauve_st = nw.st;
        nw.st = NewtonStats{};
        const bool f = nw.resout( sauve_w, true );
        diag_ref = nw.st.nb_diag;
        fin_ref = std::string( nw.st.fin ) + ( f ? "" : " (!)" );
        std::printf( "  reference : Newton depuis le depart -- %d iterations, %d diagrammes, reste %.2e, %s\n",
                     nw.st.nb_iter, diag_ref, double( nw.st.reste ), nw.st.fin );
        nw.a = sauve_a; nw.fa = sauve_fa; nw.st = sauve_st;
    }

    // ---- 3. LA VERSION LOCALE : `d` nul hors du patch. Le systeme devient sur-determine, et les
    //      poids decident ce qu'on sacrifie.
    // ---- LE SOUS-PROBLEME LOCAL A OBJECTIF BARRIERE. On ne part PAS des cellules petites au
    //      depart ( il n'y en a pas : au point converge tout est a la cible ) mais de celles que
    //      LE PAS pince -- et le pas, donc `S`, donc la boule, dependent de `F`.
    if ( o.local_nl ) {
        std::vector<SI> rang( n );
        for ( SI k = 0; k < n; ++k ) rang[ nw.pd.ids[ k ] ] = k;
        std::vector<BilanLocal> bl;
        std::vector<TF> w_loc;
        for ( TF F : o.local_F )
            for ( int N : o.rel_largeurs ) {
                BilanLocal z;
                if ( o.local_amas ) {
                    std::vector<TF> w_cible( n ), a_cib;
                    std::vector<Facette> fa_cib;
                    for ( SI i = 0; i < n; ++i ) w_cible[ i ] = w0[ i ] + F * dn[ i ];
                    nw.mesures_et_facettes( w_cible, a_cib, fa_cib );
                    w_loc.assign( n, TF( 0 ) );
                    const double t0 = now();
                    StatsAmas sa = repare_amas( nw, w0, dn, a_cib, fa_cib, F, N, o, w_loc );
                    z.F = F; z.N = N; z.s = sa.s; z.iter = sa.iter; z.nb_cel = sa.nb_cel;
                    z.fuites = sa.fuites; z.nb_amas = sa.nb_amas; z.amas_max = sa.taille_max;
                    z.taille = sa.n_inconnues;
                    z.ball_fin = sa.nb_amas ? sa.ball_fin : TF( 1 );
                    z.ball_try = 1e300;
                    for ( SI i = 0; i < n; ++i ) if ( a_cib[ i ] < o.local_pince * nw.nu[ i ] ) {
                        ++z.nS;
                        z.ball_try = std::min( z.ball_try, a_cib[ i ] / nw.nu[ i ] );
                    }
                    std::printf( "   amas : %d amas ( le plus gros %d, %d inconnues en tout ), s %.4f,"
                                 " %d sous-pas ( %d refus ), %d iterations, %d cellules, %d fuites, %.2f s\n",
                                 int( sa.nb_amas ), int( sa.taille_max ), int( sa.n_inconnues ), double( sa.s ),
                                 sa.nb_pas, sa.nb_refus, sa.iter, int( sa.nb_cel ), int( sa.fuites ), now() - t0 );
                    std::printf( "          ( %d rafraichissements de la structure de voisinage )\n", sa.nb_refresh );
                    // QUI MEURT, ET OU ? La distance de la cellule la plus maigre a l'ensemble pince
                    // dit si le dommage est DANS la zone reparee, sur sa couronne, ou plus loin.
                    {
                        std::vector<TF> ab;
                        std::vector<Facette> fb;
                        nw.mesures_et_facettes( w_loc, ab, fb );
                        Laplacien Lb;
                        Lb.assemble( n, fa_cib );
                        std::vector<int> dd( n, -1 );
                        std::vector<SI> fl;
                        for ( SI i = 0; i < n; ++i ) if ( a_cib[ i ] < o.local_pince * nw.nu[ i ] ) { dd[ i ] = 0; fl.push_back( i ); }
                        for ( size_t q = 0; q < fl.size(); ++q ) {
                            const SI i = fl[ q ];
                            if ( dd[ i ] >= 12 ) continue;
                            for ( SI e = Lb.row[ i ]; e < Lb.row[ i + 1 ]; ++e )
                                if ( dd[ Lb.col[ e ] ] < 0 ) { dd[ Lb.col[ e ] ] = dd[ i ] + 1; fl.push_back( Lb.col[ e ] ); }
                        }
                        SI pire = 0;
                        for ( SI i = 0; i < n; ++i ) if ( ab[ i ] < ab[ pire ] ) pire = i;
                        SI nmal = 0, hors = 0;
                        for ( SI i = 0; i < n; ++i ) if ( ab[ i ] < TF( 0.05 ) * nw.nu[ i ] ) {
                            ++nmal;
                            hors += dd[ i ] < 0 || dd[ i ] > N + 1;
                        }
                        std::printf( "   qui meurt : cellule %d, masse %.3e nu, a DISTANCE %d de l'ensemble pince"
                                     " ( boule <= %d, couronne %d ) ; %d cellules sous 0.05 nu dont %d HORS de la zone\n",
                                     int( pire ), double( ab[ pire ] / nw.nu[ pire ] ), dd[ pire ], N, N + 1,
                                     int( nmal ), int( hors ) );
                    }
                } else
                    z = o.local_cont ? repare_continuation( nw, rang, w0, dn, F, N, o, w_loc )
                                     : local_barriere( nw, w0, dn, rang, N, F, o, w_loc );
                std::vector<TF> a2;
                std::vector<Facette> fa2;
                nw.mesures_et_facettes( w_loc, a2, fa2 );
                z.glob_fin = 1e300; z.r2 = 0;
                for ( SI i = 0; i < n; ++i ) {
                    z.glob_fin = std::min( z.glob_fin, a2[ i ] / nw.nu[ i ] );
                    z.r2 += ( a2[ i ] - nw.nu[ i ] ) * ( a2[ i ] - nw.nu[ i ] );
                }
                z.r2 = std::sqrt( z.r2 );
                if ( o.rel_solve && z.glob_fin > 0 ) {
                    const std::vector<TF> ga = nw.a;
                    const std::vector<Facette> gf = nw.fa;
                    NewtonStats gs = nw.st;
                    nw.a = a2; nw.fa = fa2; nw.st = NewtonStats{};
                    const bool f = nw.resout( w_loc, true );
                    z.diag_apres = nw.st.nb_diag;
                    z.fin = std::string( nw.st.fin ) + ( f ? "" : " (!)" );
                    nw.a = ga; nw.fa = gf; nw.st = gs;
                }
                bl.push_back( z );
            }
        std::printf( "  | F | s atteint | N | pincees | boule | AMAS | + GROS | anneau | it | cellules | fuites | min boule ( pas -> fin ) | GLOBALE fin | diag apres |\n"
                     "  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n" );
        for ( const BilanLocal &z : bl )
            std::printf( "  | %.4g | %.4f | %d | %6d | %6d | %5d | %6d | %6d | %4d | %9d | %6d | %.3e -> %.3e | %.3e | %d %s |\n",
                         double( z.F ), double( z.s ), z.N, int( z.nS ), int( z.taille ), int( z.nb_amas ),
                         int( z.amas_max ), int( z.n_ring ), z.iter, int( z.nb_cel ), int( z.fuites ),
                         double( z.ball_try ), double( z.ball_fin ), double( z.glob_fin ),
                         z.diag_apres, z.fin.c_str() );
        if ( csv ) std::fclose( csv );
        return;
    }

    char nom[ 64 ];
    std::vector<char> dans_p( n );

    // ---- 2 bis. L'AMORTI GLOBAL : `( L C L + mu I ) d = L C ( nu - a )`, sans patch. A `mu = 0`
    //      c'est Newton quels que soient les poids ; on balaie `mu` pour voir a partir de quand les
    //      poids agissent, et ce que ca vaut.
    for ( TF mu : o.rel_mus )
        for ( TF kappa : o.rel_kappas ) {
            for ( SI i = 0; i < n; ++i ) {
                const int dd = dist[ i ];
                const TF f = ( dd < 0 || dd > o.rel_largeur_max ) ? TF( 0 )
                                                                  : TF( 1 ) - TF( dd ) / TF( o.rel_largeur_max + 1 );
                C[ i ] = 1 + ( kappa - 1 ) * f;
            }
            const int it = resout_pondere( L, C, b, tout, d, o.rel_cg, TF( 1e-10 ), mu );
            std::snprintf( nom, sizeof( nom ), "amorti mu=%g k=%g", double( mu ), double( kappa ) );
            Bilan ba = balaye( nw, w0, d, S, eps, nom, it, n, csv, o.rel_cas.c_str(), &w_lev );
            bilans.push_back( ba );
            if ( ba.alpha > 0 && o.rel_solve ) {
                std::vector<TF> a2;
                std::vector<Facette> fa2;
                nw.mesures_et_facettes( w_lev, a2, fa2 );
                const std::vector<TF> ga = nw.a;
                const std::vector<Facette> gf = nw.fa;
                NewtonStats gs = nw.st;
                nw.a = a2; nw.fa = fa2; nw.st = NewtonStats{};
                const bool f = nw.resout( w_lev, true );
                std::printf( "  amorti  mu=%g k=%g : Newton depuis le point releve -- %d iterations, %d diagrammes"
                             " ( reference %d ), reste %.2e, %s\n", double( mu ), double( kappa ),
                             nw.st.nb_iter, nw.st.nb_diag, diag_ref, double( nw.st.reste ),
                             f ? nw.st.fin : ( std::string( nw.st.fin ) + " (!)" ).c_str() );
                nw.a = ga; nw.fa = gf; nw.st = gs;
            }
        }
    for ( int largeur : o.rel_largeurs )
        for ( TF kappa : o.rel_kappas ) {
            SI taille = 0;
            for ( SI i = 0; i < n; ++i ) {
                dans_p[ i ] = dist[ i ] >= 0 && dist[ i ] <= largeur;
                taille += dans_p[ i ];
                const int dd = dist[ i ];
                const TF f = ( dd < 0 || dd > o.rel_largeur_max ) ? TF( 0 )
                                                                  : TF( 1 ) - TF( dd ) / TF( o.rel_largeur_max + 1 );
                C[ i ] = 1 + ( kappa - 1 ) * f;
            }
            const int it = resout_pondere( L, C, b, dans_p, d, o.rel_cg, TF( 1e-10 ) );
            std::snprintf( nom, sizeof( nom ), "local k=%g l=%d", double( kappa ), largeur );
            Bilan bl = balaye( nw, w0, d, S, eps, nom, it, taille, csv, o.rel_cas.c_str(), &w_lev );
            bilans.push_back( bl );

            // ---- 4. CE QUI COMPTE VRAIMENT : le relevement sert-il de POINT DE DEPART ? On se
            //      place au meilleur pas de la direction locale, on y refait un Newton COMPLET, et
            //      on regarde le pas que CELUI-LA peut prendre.
            if ( bl.alpha > 0 && o.rel_puis_newton ) {
                std::vector<TF> a2, b2( n ), d2;
                std::vector<Facette> fa2;
                nw.mesures_et_facettes( w_lev, a2, fa2 );
                Laplacien L2;
                L2.assemble( n, fa2 );
                for ( SI i = 0; i < n; ++i ) b2[ i ] = nw.nu[ i ] - a2[ i ];
                if ( lin.resout( L2, b2, d2 ) ) {
                    std::vector<TF> sauve_a = nw.a, sauve_w = w_lev;
                    std::vector<Facette> sauve_fa = nw.fa;
                    nw.a = a2; nw.fa = fa2;              // `balaye` lit `nw.nu` seulement, mais restons propres
                    std::snprintf( nom, sizeof( nom ), "  puis newton k=%g l=%d", double( kappa ), largeur );
                    bilans.push_back( balaye( nw, sauve_w, d2, S, eps, nom, 0, n, csv, o.rel_cas.c_str() ) );
                    nw.a = sauve_a; nw.fa = sauve_fa;
                }
                if ( o.rel_solve ) {
                    const std::vector<TF> ga = nw.a;
                    const std::vector<Facette> gf = nw.fa;
                    NewtonStats gs = nw.st;
                    nw.a = a2; nw.fa = fa2; nw.st = NewtonStats{};
                    const bool f = nw.resout( w_lev, true );
                    std::printf( "  releve  k=%g l=%d : Newton depuis le point releve -- %d iterations, %d diagrammes"
                                 " ( reference %d ), reste %.2e, %s\n", double( kappa ), largeur,
                                 nw.st.nb_iter, nw.st.nb_diag, diag_ref, double( nw.st.reste ),
                                 f ? nw.st.fin : ( std::string( nw.st.fin ) + " (!)" ).c_str() );
                    nw.a = ga; nw.fa = gf; nw.st = gs;
                }
            }
        }
    if ( csv ) std::fclose( csv );

    std::printf( "  | direction | inconnues | |d|max/h2 | alpha* | masse min | dont visees | |r|_2 | max|a-nu|/nu | sous eps | CG |\n"
                 "  |---|---|---|---|---|---|---|---|---|---|\n" );
    for ( const Bilan &z : bilans )
        std::printf( "  | %-20s | %7d | %8.2e | %7.1e | %8.2e | %8.2e | %9.3e | %9.3e | %5d | %d |\n",
                     z.nom.c_str(), int( z.taille ), double( z.amp ), double( z.alpha ), double( z.mmin ),
                     double( z.mmin_p ), double( z.r2 ), double( z.rmax ), int( z.sous ), z.cg );
    std::printf( "  ( au depart : |r|_2 %.3e, masse min %.2e nu. Une direction n'est utile que si elle releve\n"
                 "    les visees SANS degrader le reste -- et surtout si le Newton d'apres prend un plus grand pas. )\n",
                 double( r2_0 ), double( amin / nw.nu[ 0 ] ) );
}

template<class PD, class TA>
int lance( const Args &a, const Opts &o, const Nuage<2> &nu, Lineaire &lin, ImageT<TA> &im ) {
    const SI n = nu.n;
    double t0 = now();
    PD pd;
    pd.build( nu.P, nullptr, n, a.leaf );
    const double t_arbre = now() - t0;
    // L'ORDRE DE L'ARBRE, POUR QUI SAIT S'EN SERVIR. Le multigrille maison agrege par `rang >> 2`
    // -- des rangs consecutifs sont voisins dans le plan, l'arbre etant une courbe remplissante --
    // et c'est tout ce qu'il demande de plus qu'un solveur ordinaire. Les germes ne bougent pas,
    // donc cet ordre est pose UNE FOIS.
    lin.ordre( pd.ids.data(), n );
#ifdef _OPENMP
    omp_set_num_threads( a.par.threads );
#endif

    if ( o.check ) {
        const std::vector<TF> w0( n, TF( 0 ) );          // Laguerre a poids nuls : `cellule_avec_poids` en a besoin
        pd.set_weights( w0.data(), a.par );
        std::printf( "-- verification, t = %g\n", double( im.t ) );
        verifie( pd, nu, im );
        if constexpr ( std::is_same_v<TA,double> ) {
            // LA SIMPLE PRECISION, ET SES DEUX MOITIES SEPAREMENT ( § 19 ). `v` en `float` est le
            // STOCKAGE -- le gros tableau, celui dont la bande passante decide sur une carte ; la
            // marche en `float` est le CALCUL. Les mesurer ensemble, comme on le faisait, ne dit
            // pas laquelle des deux coute.
            ImageT<float,TF>    imS;                     // stockage float, marche double
            ImageT<float,float> imT;                     // tout en float ( l'ancien `--acc float` )
            imS.W = imT.W = im.W; imS.H = imT.H = im.H;
            imS.t = TF( im.t ); imT.t = float( im.t );
            imS.v.assign( im.v.begin(), im.v.end() );
            imT.v.assign( im.v.begin(), im.v.end() );
            imS.prepare(); imT.prepare();
            typename PD::Cell cel;
            std::vector<double> emS, efS, emT, efT;
            double somme = 0;
            for ( SI k = 0; k < n; ++k ) {
                if ( ! pd.cellule( k, cel ) || cel.nb <= 0 ) continue;
                std::vector<double> fd, fS, fT;
                const double md = double( im .mesure( cel, [ & ]( int, TF f ) { fd.push_back( double( f ) ); } ) );
                const double mS = double( imS.mesure( cel, [ & ]( int, TF f ) { fS.push_back( double( f ) ); } ) );
                const double mT = double( imT.mesure( cel, [ & ]( int, TF f ) { fT.push_back( double( f ) ); } ) );
                emS.push_back( std::fabs( mS - md ) );
                emT.push_back( std::fabs( mT - md ) );
                somme += md;
                for ( size_t q = 0; q < fd.size(); ++q ) {
                    if ( ! ( fd[ q ] > 0 ) ) continue;
                    if ( q < fS.size() ) efS.push_back( std::fabs( fS[ q ] - fd[ q ] ) / fd[ q ] );
                    if ( q < fT.size() ) efT.push_back( std::fabs( fT[ q ] - fd[ q ] ) / fd[ q ] );
                }
            }
            for ( double &e : emS ) e /= somme / n;
            for ( double &e : emT ) e /= somme / n;
            std::printf( "-- TOUT en simple precision ( stockage ET marche ), ecart a la double :\n" );
            quantiles( emT, "masse ( / moyenne )" );
            quantiles( efT, "facette ( relatif )" );
            std::printf( "-- l'IMAGE SEULE en simple precision, la marche en double ( `--acc float` ) :\n" );
            quantiles( emS, "masse ( / moyenne )" );
            quantiles( efS, "facette ( relatif )" );
        }
        return 0;
    }

    if ( o.chrono ) {
        const std::vector<TF> w0( n, TF( 0 ) );
        pd.set_weights( w0.data(), a.par );
        chrono( pd, im, a.par, std::max( a.reps, 1 ) );
        return 0;
    }

    // =============================================================================================
    // LA CONTINUATION, PARAMETREE PAR UN SEUL NOMBRE `lam` QUI DECROIT VERS ZERO
    //
    //   chemin MELANGE : `lam = 1 - t`, LE PLANCHER. `rho = lam + ( 1 - lam ) rho_nue`.
    //   chemin CONV    : `lam = sigma`, LA LARGEUR de la convolution. `rho = rho_nue * G_lam`.
    //
    // Les deux partent d'une densite plate ( `lam = 1` : Lebesgue d'un cote, une image floutee a
    // l'echelle du domaine de l'autre ) et finissent a `lam = 0`, l'image nue. Les deux ont la
    // meme propriete : `lam` mesure a quel point les zeros de l'image sont bouches. Un seul jeu de
    // code, donc, et une comparaison a armes egales.
    //
    // LE PAS ADAPTATIF ( `--adaptatif` ). Le diagramme de depart d'une etape -- les cellules aux
    // POIDS COURANTS sous la NOUVELLE densite -- est le juge : si sa plus petite masse tombe sous
    // `seuil x nu`, le pas etait trop grand, et on le reprend a la valeur INTERMEDIAIRE
    // `sqrt( lam_ok x lam )` ( geometrique : c'est la bonne variable, cf. § 12.5 ), ou `lam_ok / 2`
    // quand la cible est zero. Le test ne coute rien quand il passe : ce diagramme est exactement
    // celui dont Newton a besoin pour demarrer.
    // =============================================================================================
    const bool conv = o.chemin == "conv";
    std::vector<double> brut;                            // l'image nue, gardee pour la convolution
    if ( conv ) {
        brut.resize( im.v.size() );
        for ( size_t q = 0; q < brut.size(); ++q ) brut[ q ] = double( im.v[ q ] );
    }
    auto regle = [ & ]( double lam ) {
        if ( conv ) convolue( brut, im, lam * o.sigma0 );
        else        im.t = TA( 1 - lam );
    };
    auto nom_de = [ & ]( double lam ) {
        char b[ 64 ];
        if ( conv ) std::snprintf( b, sizeof( b ), "sigma %.4g", lam * o.sigma0 );
        else        std::snprintf( b, sizeof( b ), "t %.9g", 1 - lam );
        return std::string( b );
    };

    // la suite des `lam`, decroissante, finissant par zero
    std::vector<double> liste;
    for ( TF v : o.liste ) liste.push_back( conv ? double( v ) : 1 - double( v ) );
    if ( liste.empty() && ( o.adaptatif || o.chooseur ) && o.etapes <= 1 && o.etapes_geo <= 0 ) {
        // ADAPTATIF SEUL : on ne pose aucune echelle. On resout la densite plate, puis on VISE LA
        // CIBLE a chaque fois, et c'est le refus qui fabrique les etapes intermediaires.
        liste = { 1.0, 0.0 };
    }
    if ( liste.empty() ) {
        if ( conv || o.etapes_geo > 0 ) {
            const int K = o.etapes_geo > 0 ? o.etapes_geo : o.etapes;
            double u = conv ? 1.0 : 0.5;                 // conv : on part de la largeur du domaine
            for ( int e = 0; e < K; ++e, u /= 2 ) liste.push_back( u );
        } else
            for ( int e = 1; e <= o.etapes; ++e ) liste.push_back( 1 - double( e ) / o.etapes );
    }
    // LA CIBLE de la continuation. `--relevement` s'arrete AVANT le pas qui casse : on converge
    // en `rel_depuis`, et c'est de la que l'etude part.
    const double lam_fin = o.relevement ? ( o.rel_depuis >= 0 ? o.rel_depuis : 0.125 ) : 0.0;
    while ( ! liste.empty() && liste.back() <= lam_fin ) liste.pop_back();
    liste.push_back( lam_fin );

    Newton<PD,ImageT<TA>> nw( pd, lin, nu.P, a.par, o.newton );
    nw.rho = &im;
    nw.derivee = o.ordre > 0 && ! conv;                  // `d a / d sigma` n'est pas en forme close

    std::vector<TF> w( n, TF( 0 ) ), dw, b, w1, a_p, da_p, w_try, a_try, da_try;
    std::vector<Facette> fa_p, fa_try;
    int tot_it = 0, tot_diag = 0, tot_recul = 0, nb_refus = 0, nb_solve_juge = 0;
    SI  tot_cel_lim = 0;
    double t_lim = 0;
    bool ok = true;
    std::vector<std::string> lignes;
    const double debut = now();
    double lam_ok = liste.front() * 2;                   // la derniere valeur RESOLUE ( fictive au depart )
    size_t suivante = 0;                                 // la prochaine valeur de la liste a viser
    double lam = liste[ 0 ];
    // LE CHOOSEUR DE PAS. `amp = | d |inf / h^2` de la premiere direction de Newton mesure ce que
    // l'etape ecoulee a RECLAME en poids. C'est le seul des trois predicteurs du § 12.6.1 qui
    // decroisse proprement le long d'une continuation qui marche -- il ne sait pas dire « refuse »,
    // il sait dire « accelere ». On suppose `amp ~ C x Dlog( lam )`, on estime `C` sur l'etape qui
    // vient de passer, et on choisit le `Dlog` suivant pour viser `amp_cible`.
    double dlog = std::log( 2.0 );                       // le premier pas : un rapport deux
    int nb_pas_faits = 0;
    const double dlog_min = std::log( 1.1 ), dlog_max = std::log( 64.0 );
    const double lam_min = o.lam_min > 0 ? double( o.lam_min ) : ( conv ? 0.33 / im.W : 1e-6 );
    for ( int e = 0; e < o.max_etapes; ++e ) {
        regle( lam );
        const TF M = im.masse_carre();
        nw.nu.assign( n, M / n );
        nw.st = NewtonStats{};
        lin.st = StatsLin{};

        // ---- LE JUGE : les cellules aux poids courants, sous la densite proposee
        bool mesure = false;
        if ( e > 0 ) {
            nw.mesures_et_facettes( w, nw.a, nw.fa, nw.derivee ? &nw.da : nullptr );
            ++tot_diag;
            // DEUX CRITERES, LE MEME DIAGRAMME. Le premier voit les cellules qui MEURENT -- c'est
            // ce que les zeros de l'image font. Le second voit que la densite a trop BOUGE, meme
            // sans tuer personne -- c'est ce que le contraste fait. Le premier seul laisse passer
            // un saut que Newton paie ensuite en 274 iterations ( mesure ).
            TF amin = nw.a[ 0 ], pire = 0;
            SI nmort = 0;
            for ( SI i = 0; i < n; ++i ) {
                amin = std::min( amin, nw.a[ i ] );
                pire = std::max( pire, std::fabs( nw.a[ i ] - nw.nu[ i ] ) / nw.nu[ i ] );
                nmort += nw.a[ i ] < o.seuil * nw.nu[ i ];
            }
            // LE TROISIEME PREDICTEUR, CELUI QUI PORTE LE GRADIENT. `r = max |a - nu| / nu` dit de
            // combien la densite a bouge LA MESURE ; il ne dit rien de ce qu'il en COUTE. Ce qu'il
            // en coute, c'est la correction de poids que le changement reclame,
            //
            //      d = L^-1 ( nu - a ),   L la hessienne ( = d a / d w ) du diagramme de depart
            //
            // rapportee a l'echelle naturelle des poids, `h^2 = 1/n`. Deux etapes de meme `r` n'ont
            // pas du tout le meme `d` : la ou la densite est forte les facettes pesent lourd et une
            // petite correction suffit ; la ou elle est faible, `L` est presque singuliere et il
            // faut deplacer les poids beaucoup. C'est exactement le systeme que la premiere
            // iteration de Newton resoudra -- une RESOLUTION de plus, pas un diagramme.
            TF amp_dw = -1;
            if ( o.diagnostic || o.seuil_dw > 0 ) {
                Laplacien Ld;
                Ld.assemble( n, nw.fa );
                b.assign( n, TF( 0 ) );
                for ( SI i = 0; i < n; ++i ) b[ i ] = nw.nu[ i ] - nw.a[ i ];
                if ( lin.resout( Ld, b, w1 ) ) {
                    amp_dw = 0;
                    for ( SI i = 0; i < n; ++i ) amp_dw = std::max( amp_dw, std::fabs( w1[ i ] ) );
                    amp_dw *= TF( n );                   // en unites de `h^2 = 1/n`
                }
                ++nb_solve_juge;
            }
            if ( o.diagnostic )
                std::printf( "   [ juge %s ] r_max %.3e  r_2 %.3e  |d|max/h2 %.3e  masse min %.2e x nu  %d mortes\n",
                             nom_de( lam ).c_str(), double( pire ), double( nw.merite( nw.a ) ),
                             double( amp_dw ), double( amin / nw.nu[ 0 ] ), int( nmort ) );
            const bool trop_mort = amin < o.seuil * nw.nu[ 0 ];
            const bool trop_loin = o.seuil_res > 0 && pire > o.seuil_res;
            const bool trop_dw   = o.seuil_dw > 0 && amp_dw > o.seuil_dw;
            if ( o.adaptatif && ( trop_mort || trop_loin || trop_dw ) ) {
                const double mid = lam > lam_fin ? std::sqrt( lam_ok * lam ) : ( lam_ok + lam_fin ) / 2;
                if ( ! ( mid < lam_ok * ( 1 - 1e-9 ) ) ) {
                    std::printf( "-- %s : REFUSE et plus de place pour un intermediaire -- on y va quand meme\n", nom_de( lam ).c_str() );
                } else {
                    std::printf( "-- %s REFUSE ( %s ) : masse min %.1e x nu, %d cellules sous le seuil, depart %.2e -> on recule sur %s\n",
                                 nom_de( lam ).c_str(),
                                 trop_mort ? "cellules mortes" : trop_loin ? "depart trop loin" : "correction de poids trop grande",
                                 double( amin / nw.nu[ 0 ] ), int( nmort ), double( pire ), nom_de( mid ).c_str() );
                    // LE CHOOSEUR APPREND DU REFUS. Sans ça il repart du rapport qu'il avait
                    // choisi -- qui vient d'être refusé -- et il oscille : saut à la cible, refus,
                    // repli, saut à la cible... ( 12 refus mesurés ). La correction de poids ne
                    // voit pas mourir les cellules ( § 12.6.1 ) ; c'est au refus de le lui dire.
                    if ( o.chooseur && mid > 0 ) dlog = std::log( lam_ok / mid );
                    lam = mid;
                    ++nb_refus;
                    continue;
                }
            }
            mesure = true;
        }

        std::printf( "-- %s : masse sur le carre %.9f, rho max %.4g, %s\n",
                     nom_de( lam ).c_str(), double( M ), double( im.max_rho() ),
                     e == 0 ? "depuis Voronoi" : dw.empty() ? "depuis les poids precedents" : "depuis les poids extrapoles" );
        const double te = now();

        // L'EXTRAPOLATION TANGENTE, GARDEE ( `--ordre 1`, chemin melange seulement ). `theta = 1,
        // 1/2, ...` tant qu'une cellule passe sous le plancher ou que le residu ne s'ameliore pas.
        TF theta = 0;
        int nb_essais = 0;
        if ( ! dw.empty() ) {
            std::vector<TF> *pdt = nw.derivee ? &da_try : nullptr;
            a_p = nw.a; fa_p = nw.fa; if ( nw.derivee ) da_p = nw.da;
            TF plancher = nw.nu[ 0 ], r_p = nw.merite( a_p );
            for ( SI i = 0; i < n; ++i ) plancher = std::min( plancher, a_p[ i ] );
            plancher *= TF( 0.5 );
            w_try.resize( n );
            for ( theta = 1; theta >= TF( 1. / 16 ); theta /= 2 ) {
                ++nb_essais;
                for ( SI i = 0; i < n; ++i ) w_try[ i ] = w[ i ] + theta * dw[ i ];
                nw.mesures_et_facettes( w_try, a_try, fa_try, pdt );
                TF amin = a_try[ 0 ];
                for ( SI i = 0; i < n; ++i ) amin = std::min( amin, a_try[ i ] );
                if ( amin >= plancher && nw.merite( a_try ) < r_p ) break;
            }
            if ( theta >= TF( 1. / 16 ) ) {
                w.swap( w_try ); nw.a.swap( a_try ); nw.fa.swap( fa_try ); if ( nw.derivee ) nw.da.swap( da_try );
                std::printf( "   extrapolation retenue : theta %.4f ( %d essais ), |r|_2 %.3e contre %.3e sans\n",
                             double( theta ), nb_essais, double( nw.merite( nw.a ) ), double( r_p ) );
            } else {
                theta = 0;
                nw.a.swap( a_p ); nw.fa.swap( fa_p ); if ( nw.derivee ) nw.da.swap( da_p );
                std::printf( "   extrapolation REFUSEE ( %d essais ) : depart sans\n", nb_essais );
            }
        }

        bool fini;
        StatsReleve sr;
        if ( o.newton_releve ) {
            if ( ! mesure ) nw.mesures_et_facettes( w, nw.a, nw.fa );
            fini = resout_releve( nw, lin, w, o, sr );
            nw.w = w;
            nw.st.nb_iter = sr.nb_iter; nw.st.nb_diag = sr.nb_diag; nw.st.nb_recul = sr.nb_recul;
            nw.st.reste = sr.reste; nw.st.fin = sr.fin;
        } else
            fini = nw.resout( w, mesure );
        const double dt = now() - te;
        NewtonStats &st = nw.st;
        if ( o.newton_releve ) {
            std::printf( "   relevement : %d pas repares ( %d cellules malades en tout, %d echecs, %d amas ),"
                         " %d cellules calculees, %d amas abandonnes, lam de %.3e a %.3e, %.2f s\n",
                         sr.nb_repare, int( sr.nb_malades ), sr.nb_echec, int( sr.nb_amas ),
                         int( sr.nb_cel ), int( sr.nb_abandon ),
                         sr.lam_max > 0 ? sr.lam_min : 0.0, sr.lam_max, sr.t_rep );
            // LA SEQUENCE DES COEFFICIENTS DE RELAXATION : c'est elle qui dit si le relevement
            // permet d'avancer plus vite, iteration par iteration.
            if ( sr.t_rep > 0 )
                std::printf( "   ou passe le temps : geometrie %.2f s ( %d mesures ), assemblage %.2f s"
                             " ( %d ), algebre %.2f s, placement %.2f s, amas %.2f s,"
                             " rafraichissement %.2f s, verdict %.2f s  --  reste %.2f s\n",
                             sr.t_geo, int( sr.n_geo ), sr.t_asm, int( sr.n_asm ), sr.t_cg,
                             sr.t_bis, sr.t_pre, sr.t_ref, sr.t_ver,
                             sr.t_rep - sr.t_geo - sr.t_asm - sr.t_cg - sr.t_bis - sr.t_pre
                                      - sr.t_ref - sr.t_ver );
            if ( sr.nb_bis )
                std::printf( "   raffinement de U : %d cellules bissectees, %d cellules calculees"
                             " ( %.1f par bissection )\n",
                             int( sr.nb_bis ), int( sr.cel_bis ),
                             sr.nb_bis ? double( sr.cel_bis ) / double( sr.nb_bis ) : 0.0 );
            if ( sr.nb_remonte )
                std::printf( "   ATTENTION : %d retours au buffer tournant ( etat degenere au depart d'une iteration )\n",
                             int( sr.nb_remonte ) );
            std::printf( "   paliers de lam en echec : %d fenetre VIDE ( les deux bords, au MEME palier ),"
                         " %d inconnue morte partout, %d couronne morte partout\n",
                         int( sr.ech_fenetre ), int( sr.ech_inconnue ), int( sr.ech_couronne ) );
            std::printf( "   refus : %d avec s = 0, %d sur le residu, %d inconnue morte, %d couronne morte,"
                         " %d hors zone  --  dont %d DEJA MORTES AU PAS NU ( amas abandonne, rendu tel quel )"
                         "  ( pire a/nu %.3e, seuil local %.3e )\n",
                         int( sr.ref_s0 ), int( sr.ref_res ), int( sr.ref_mal_dans ), int( sr.ref_mal_cour ),
                         int( sr.ref_mal_hors ), int( sr.ref_deja_morte ),
                         sr.pire_ratio < 1e299 ? sr.pire_ratio : 0.0, double( o.local_eps ) );
            if ( sr.t_rep > 0 )
                std::printf( "   recherches lineaires : %d en echec, %d evaluations brulees par elles"
                             " ( %.0f %% des %d )  --  %d sous-resolutions, %.0f evaluations et"
                             " %.0f cellules chacune\n",
                             int( sr.n_ls_ech ), int( sr.n_geo_ech ),
                             sr.n_geo > 0 ? 100.0 * double( sr.n_geo_ech ) / double( sr.n_geo ) : 0.0,
                             int( sr.n_geo ), int( sr.n_sous ),
                             sr.n_sous > 0 ? double( sr.n_geo ) / double( sr.n_sous ) : 0.0,
                             sr.n_sous > 0 ? double( sr.nb_cel ) / double( sr.n_sous ) : 0.0 );
            // LA MOYENNE CACHE CE QU'ON CHERCHE. Les « evenements exceptionnels » -- une poignee de
            // cellules qui imposent un pas minuscule a tout le monde -- ne se voient que dans le
            // MINIMUM et dans la fraction d'iterations sous un pas donne.
            double som = 0, mx = 0, mn = 1e300;
            int n_petit = 0;
            for ( double f : sr.suite_F ) {
                som += f; mx = std::max( mx, f ); mn = std::min( mn, f );
                n_petit += f < 0.125;
            }
            std::printf( "   suite des pas ( %d ) : min %.3e, moyenne %.3e, max %.3e,"
                         " %d sous 1/8 ( %.0f %% )  |",
                         int( sr.suite_F.size() ), sr.suite_F.empty() ? 0.0 : mn,
                         sr.suite_F.empty() ? 0.0 : som / sr.suite_F.size(), mx, n_petit,
                         sr.suite_F.empty() ? 0.0 : 100.0 * n_petit / double( sr.suite_F.size() ) );
            for ( size_t q = 0; q < sr.suite_F.size() && q < 24; ++q ) std::printf( " %.3g", sr.suite_F[ q ] );
            std::printf( "%s\n", sr.suite_F.size() > 24 ? " ..." : "" );
        }
        ok = fini;
        tot_it += st.nb_iter; tot_diag += st.nb_diag; tot_recul += st.nb_recul;
        tot_cel_lim += st.nb_cell_lim; t_lim += st.t_lim;
        char buf[ 512 ];
        std::snprintf( buf, sizeof( buf ), "| %-16s | %.2f (%d) | %.2e | %d | %d (%d) | %.2e | %.2f s | %s |",
                       nom_de( lam ).c_str(), double( theta ), nb_essais, double( st.reste0 ), st.nb_iter,
                       st.nb_diag, st.nb_recul, double( st.reste ), dt, st.fin );
        lignes.push_back( buf );
        // CE QUE LE SOLVEUR LINEAIRE FAIT DE SON TEMPS. `lin` est le premier poste de la boucle et
        // le moins parallele : la HIERARCHIE, refaite a chaque iteration, est sequentielle chez
        // amgcl, la resolution ne l'est pas. Les separer dit lequel des deux on doit attaquer.
        std::printf( "   lineaire %s : mise en forme %.2f s, hierarchie %.2f s ( %d ),"
                     " resolution %.2f s ( %d iterations de CG )\n",
                     lin.nom(), lin.st.t_forme, lin.st.t_hier, lin.st.nb_hier,
                     lin.st.t_res, lin.st.nb_iter );
        std::printf( "   newton %s : depart %.2e, %d iterations, %d diagrammes ( %d reculs ), reste %.2e, %.2f s"
                     " [ diag %.2f  asm %.2f  lin %.2f  lim %.2f ]%s\n",
                     st.fin, double( st.reste0 ), st.nb_iter, st.nb_diag, st.nb_recul, double( st.reste ), dt,
                     st.t_diag, st.t_asm, st.t_lin, st.t_lim,
                     st.nb_deborde ? ( "  DEBORDEMENTS : " + std::to_string( st.nb_deborde ) + " cellules ( --maxnv )" ).c_str() : "" );
        w = nw.w;
        {   // CE QUE LA CONTINUATION FAIT AUX CELLULES MOURANTES : ou bien leur POIDS reste borne
            // -- elles ont rejoint le support, et `lam = 0` est atteignable -- ou bien il diverge.
            TF amin = nw.a[ 0 ], amax_w = 0;
            SI nmaigre = 0;
            for ( SI i = 0; i < n; ++i ) {
                amin = std::min( amin, nw.a[ i ] );
                amax_w = std::max( amax_w, std::fabs( w[ i ] ) );
                nmaigre += nw.a[ i ] < TF( 0.5 ) * nw.nu[ i ];
            }
            std::printf( "   etat : masse min %.3e ( cible %.3e, rapport %.2e ), %d cellules sous nu/2, |w|max %.3e ( h^2 = %.3e )\n",
                         double( amin ), double( nw.nu[ 0 ] ), double( amin / nw.nu[ 0 ] ), int( nmaigre ),
                         double( amax_w ), 1.0 / n );
        }
        dw.clear();
        if ( ! fini && st.fin != std::string( "STAGNATION" ) ) break;
        lam_ok = lam;
        if ( lam <= lam_fin ) break;                     // la cible de la continuation est faite

        // LA TANGENTE vers l'etape suivante : `L w' = nu' - d a / d t`, `dw = ( t' - t ) w'`.
        while ( suivante < liste.size() && liste[ suivante ] >= lam ) ++suivante;
        const double lam_but = suivante < liste.size() ? liste[ suivante ] : 0.0;
        if ( o.ordre > 0 && ! conv && ! nw.da.empty() ) {
            const TF dtc = TF( lam - lam_but );          // `d t = -( d lam )`, et `da` est en `t`
            Laplacien L;
            L.assemble( n, nw.fa );
            b.resize( n );
            TF sda = 0;
            for ( SI i = 0; i < n; ++i ) sda += nw.da[ i ];
            for ( SI i = 0; i < n; ++i ) b[ i ] = sda / n - nw.da[ i ];
            if ( lin.resout( L, b, w1 ) ) {
                dw.resize( n );
                TF amp = 0;
                for ( SI i = 0; i < n; ++i ) { dw[ i ] = dtc * w1[ i ]; amp = std::max( amp, std::fabs( dw[ i ] ) ); }
                std::printf( "   extrapolation vers %s : |dw|max %.3e\n", nom_de( lam_but ).c_str(), double( amp ) );
            } else
                std::printf( "   extrapolation : solveur lineaire en echec\n" );
        }
        if ( o.chooseur && lam > lam_fin ) {
            const double amp = double( nw.st.amp_d0 ) * n;           // en unites de `h^2`
            // la PREMIERE etape part de Voronoi sous une densite plate : sa correction ne mesure
            // aucun pas de continuation, et on ne s'en sert pas pour calibrer.
            if ( nb_pas_faits > 0 ) {
                if ( amp > 0 ) dlog = std::min( dlog_max, std::max( dlog_min, dlog * double( o.amp_cible ) / amp ) );
                else           dlog = dlog_max;
            }
            ++nb_pas_faits;
            const double prop = lam * std::exp( -dlog );
            lam = prop < std::max( lam_min, lam_fin ) ? lam_fin : prop;
            std::printf( "   chooseur : correction reclamee %.3e h^2 ( cible %.0f ) -> rapport %.2f, %s\n",
                         amp, double( o.amp_cible ), std::exp( dlog ), nom_de( lam ).c_str() );
        } else
            lam = lam_but;
    }
    const double total = now() - debut + t_arbre;
    std::printf( "  TOTAL : %d iterations, %d diagrammes ( %d reculs, %d etapes refusees, %d resolutions de juge ), %.2f s ( arbre %.3f )  --  %s\n",
                 tot_it, tot_diag, tot_recul, nb_refus, nb_solve_juge, total, t_arbre, ok ? "converge" : "PAS CONVERGE" );
    if ( tot_cel_lim )
        std::printf( "  limites en masse : %d cellules calculees, %.2f s\n", int( tot_cel_lim ), t_lim );
    std::printf( "  | %s | theta (essais) | depart | it | diag (reculs) | reste | temps | fin |\n  |---|---|---|---|---|---|---|---|\n",
                 conv ? "sigma" : "t" );
    for ( const std::string &l : lignes ) std::printf( "  %s\n", l.c_str() );

    // ---- L'ETUDE DU RELEVEMENT : on pousse la densite jusqu'a la cible QUI PINCE, aux poids de
    //      l'etape convergee, et on compare les directions.
    if ( o.relevement && ok ) {
        regle( o.rel_vers );
        const TF M = im.masse_carre();
        nw.nu.assign( n, M / n );
        nw.mesures_et_facettes( w, nw.a, nw.fa );
        std::printf( "-- RELEVEMENT : converge en %s, on pousse a %s\n",
                     nom_de( lam_fin ).c_str(), nom_de( o.rel_vers ).c_str() );
        etude_relevement( nw, lin, w, o );
    }

    if ( ! o.ecrire.empty() ) {
        char entete[ 256 ];
        std::snprintf( entete, sizeof( entete ), "# densite image %d x %d, poids obtenus par newton\n", im.W, im.H );
        if ( ecrit_nuage<2>( o.ecrire, nu, w.data(), entete ) ) std::printf( "  poids ecrits dans '%s'\n", o.ecrire.c_str() );
    }
    return ok ? 0 : 1;
}

/// des germes tires SELON l'image ( rejet )
template<class Im>
Nuage<2> nuage_selon( const Im &im, SI n, unsigned graine ) {
    Nuage<2> nu;
    nu.nom = "selon l'image n=" + std::to_string( n );
    std::mt19937_64 rng( graine + 17 );
    std::uniform_real_distribution<TF> uni( 1e-6, 1 - 1e-6 ), u01( 0, 1 );
    const TF rmax = im.max_rho();
    while ( SI( nu.c[ 0 ].size() ) < n ) {
        const TF x = uni( rng ), y = uni( rng );
        if ( u01( rng ) * rmax <= im.rho( x, y ) ) { nu.c[ 0 ].push_back( x ); nu.c[ 1 ].push_back( y ); }
    }
    nu.w.assign( n, TF( 0 ) );
    nu.finish();
    return nu;
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
        else if ( s == "--image" )      o.taille = std::atoi( val() );
        else if ( s == "--image-pgm" )  o.pgm = val();
        else if ( s == "--sans-trou" )  o.trou = false;
        else if ( s == "--contraste" )  o.contraste = std::atof( val() );
        else if ( s == "--etapes" )     o.etapes = std::atoi( val() );
        else if ( s == "--etapes-geo" ) o.etapes_geo = std::atoi( val() );
        else if ( s == "--chemin" )     o.chemin = val();
        else if ( s == "--sigma0" )     o.sigma0 = std::atof( val() );
        else if ( s == "--adaptatif" )  o.adaptatif = true;
        else if ( s == "--seuil" )      o.seuil = std::atof( val() );
        else if ( s == "--seuil-residu" ) o.seuil_res = std::atof( val() );
        else if ( s == "--seuil-dw" )   o.seuil_dw = std::atof( val() );
        else if ( s == "--diagnostic" ) o.diagnostic = true;
        else if ( s == "--chooseur" )   o.chooseur = true;
        else if ( s == "--relevement" ) o.relevement = true;
        else if ( s == "--rel-depuis" ) o.rel_depuis = std::atof( val() );
        else if ( s == "--rel-vers" )   o.rel_vers = std::atof( val() );
        else if ( s == "--rel-seuil" )  o.rel_seuil = std::atof( val() );
        else if ( s == "--rel-cg" )     o.rel_cg = std::atoi( val() );
        else if ( s == "--rel-sans-identite" ) o.rel_identite = false;
        else if ( s == "--rel-sans-newton" )   o.rel_puis_newton = false;
        else if ( s == "--rel-solve" )  o.rel_solve = true;
        else if ( s == "--local-nl" )   o.local_nl = true;
        else if ( s == "--local-lin" )  o.local_barriere = false;
        else if ( s == "--local-trace" ) o.local_trace = true;
        else if ( s == "--rel-largeur-max" ) o.rel_largeur_max = std::atoi( val() );
        else if ( s == "--local-maxit" ) o.local_maxit = std::atoi( val() );
        else if ( s == "--local-tol" )  o.local_tol = std::atof( val() );
        else if ( s == "--local-eps" )  o.local_eps = std::atof( val() );
        else if ( s == "--local-pince" ) o.local_pince = std::atof( val() );
        else if ( s == "--local-cg" )   o.local_cg = std::atoi( val() );
        else if ( s == "--newton-releve" ) o.newton_releve = true;
        else if ( s == "--releve-off" ) o.releve_actif = false;
        else if ( s == "--releve-ratio" ) o.releve_ratio = std::atof( val() );
        else if ( s == "--releve-mort" ) o.releve_mort = std::atof( val() );
        else if ( s == "--releve-anneaux" ) o.releve_anneaux = std::atoi( val() );
        else if ( s == "--local-saut" ) o.local_cont = false;
        else if ( s == "--local-sans-tangente" ) o.local_tangente = false;
        else if ( s == "--local-amas" ) o.local_amas = true;
        else if ( s == "--local-coupeurs" ) o.local_coupeurs = std::atoi( val() );
        else if ( s == "--local-bord" ) o.local_bord = std::atof( val() );
        else if ( s == "--local-grad" ) o.local_grad = true;
        else if ( s == "--local-barriere" ) o.local_grad = false;
        else if ( s == "--local-refresh" ) o.local_refresh = std::atoi( val() );
        else if ( s == "--local-patch" ) { o.local_patch = true; if ( ! ( o.local_bord > 0 ) ) o.local_bord = 10; }
        else if ( s == "--local-marge" ) o.local_marge = std::atof( val() );
        else if ( s == "--local-dump" ) o.local_dump = std::atoi( val() );
        else if ( s == "--local-scan" ) o.local_scan = std::atoi( val() );
        else if ( s == "--diag-u" ) o.diag_u = true;
        else if ( s == "--releve-sans-u" ) o.releve_u = false;
        else if ( s == "--u-max" ) o.u_max = true;
        else if ( s == "--placement-homo" ) o.placement_homo = true;
        else if ( s == "--demo-homo" ) { o.demo_homo = std::atoi( val() ); o.placement_homo = true; }
        else if ( s == "--releve-beta" ) o.releve_beta = std::atof( val() );
        else if ( s == "--releve-ratio-eps" ) o.releve_ratio_eps = std::atof( val() );
        else if ( s == "--releve-raideur" ) o.releve_raideur = std::atof( val() );
        else if ( s == "--releve-bis-tol" ) o.releve_bis_tol = std::atof( val() );
        else if ( s == "--releve-mal-max" ) o.releve_mal_max = std::atoi( val() );
        else if ( s == "--local-amas-max" ) o.local_amas_max = std::atoi( val() );
        else if ( s == "--local-F" ) {
            o.local_F.clear();
            std::stringstream ss( val() ); std::string it;
            while ( std::getline( ss, it, ',' ) ) o.local_F.push_back( std::atof( it.c_str() ) );
        }
        else if ( s == "--rel-mus" ) {
            o.rel_mus.clear();
            std::stringstream ss( val() ); std::string it;
            while ( std::getline( ss, it, ',' ) ) o.rel_mus.push_back( std::atof( it.c_str() ) );
        }
        else if ( s == "--rel-csv" )    o.rel_csv = val();
        else if ( s == "--rel-cas" )    o.rel_cas = val();
        else if ( s == "--rel-largeurs" ) {
            o.rel_largeurs.clear();
            std::stringstream ss( val() ); std::string it;
            while ( std::getline( ss, it, ',' ) ) o.rel_largeurs.push_back( std::atoi( it.c_str() ) );
        }
        else if ( s == "--rel-kappas" ) {
            o.rel_kappas.clear();
            std::stringstream ss( val() ); std::string it;
            while ( std::getline( ss, it, ',' ) ) o.rel_kappas.push_back( std::atof( it.c_str() ) );
        }
        else if ( s == "--amp-cible" )  o.amp_cible = std::atof( val() );
        else if ( s == "--lam-min" )    o.lam_min = std::atof( val() );
        else if ( s == "--max-etapes" ) o.max_etapes = std::atoi( val() );
        else if ( s == "--liste" ) {
            std::stringstream ss( val() ); std::string it;
            while ( std::getline( ss, it, ',' ) ) o.liste.push_back( std::atof( it.c_str() ) );
        }
        else if ( s == "--ordre" )      o.ordre = std::atoi( val() );
        else if ( s == "--diracs" )     o.diracs = val();
        else if ( s == "--acc" )        o.acc = val();
        else if ( s == "--check" )      o.check = true;
        else if ( s == "--chrono" )     o.chrono = true;
        else if ( s == "--solver" )     o.solver = val();
        else if ( s == "--amg-var" )    o.amgvar = std::atoi( val() );
        else if ( s == "--amg-tol" )    o.amgtol = std::atof( val() );
        else if ( s == "--mg-nu" )      o.mg_nu = std::atoi( val() );
        else if ( s == "--mg-gros" )    o.mg_gros = std::atoi( val() );
        else if ( s == "--mg-k" )       o.mg_k = std::atoi( val() );
        else if ( s == "--mg-stop" )    o.mg_stop = std::atoi( val() );
        else if ( s == "--mg-exact" )   o.mg_exact = std::atoi( val() );
        else if ( s == "--mg-lisse" )   o.mg_lisse = std::atoi( val() );
        else if ( s == "--mg-agreg" )   o.mg_agreg = std::atoi( val() );
        else if ( s == "--mg-lisseur" ) o.mg_lisseur = std::atoi( val() );
        else if ( s == "--mg-cheb" )    o.mg_cheb = std::atof( val() );
        else if ( s == "--mg-recycle" ) o.mg_recycle = std::atoi( val() );
        else if ( s == "--amg-refaire" ) o.amg_refaire = std::atoi( val() );
        else if ( s == "--mg-refaire" ) o.mg_refaire = std::atoi( val() );
        else if ( s == "--mg-omega-p" ) o.mg_omega_p = std::atof( val() );
        else if ( s == "--mg-tronque" ) o.mg_tronque = std::atof( val() );
        else if ( s == "--mg-force" )   o.mg_force = std::atof( val() );
        else if ( s == "--mg-trace" )   o.mg_trace = 1;
        else if ( s == "--newton-tol" ) o.newton.tol = std::atof( val() );
        else if ( s == "--newton-max" ) o.newton.maxit = std::atoi( val() );
        else if ( s == "--t-min" )      o.newton.t_min = std::atof( val() );
        else if ( s == "--pas" )        o.newton.pas = std::string( val() ) == "essai-limites" ? NewtonOptions::ESSAI_LIMITES : NewtonOptions::ESSAIS;
        else if ( s == "--beta0" )      o.newton.beta0 = std::atof( val() );
        else if ( s == "--mult-ok" )    o.newton.mult_ok = std::atof( val() );
        else if ( s == "--facteur" )    o.newton.facteur = std::atof( val() );
        else if ( s == "--lim-tol" )    o.newton.lim.tol = std::atof( val() );
        else if ( s == "--residu" ) {
            const std::string v = val();
            o.newton.residu = v == "barriere" ? NewtonOptions::BARRIERE : v == "log" ? NewtonOptions::LOG : NewtonOptions::LIN;
        }
        else if ( s == "--quiet" )      o.newton.trace = false;
        else if ( s == "--ecrire" )     o.ecrire = val();
        else {
            std::printf( "usage: image [options]\n" );
            Args::usage();
            std::printf(
                "  --image N       l'image de synthese : N x N ( fond lisse, disque net, bande fine, carre vide )  (512)\n"
                "  --image-pgm F   une image lue dans un PGM ( P2 ou P5 ) a la place\n"
                "  --sans-trou     l'image de synthese sans le carre a zero\n"
                "  --contraste F   rho <- ( 1 - F ) + F rho une fois pour toutes ( F = 0 : Lebesgue )   (1)\n"
                "  --etapes K      LA CONTINUATION UNIFORME : t = 1/K ... 1, chaque etape partant de la precedente  (1)\n"
                "  --etapes-geo K  LA CONTINUATION GEOMETRIQUE : 1 - t = 1/2, 1/4, ... 2^-K, puis t = 1. LA BONNE ECHELLE\n"
                "                  des que l'image a des zeros ( § 12.5 ) -- les dernieres etapes y coutent 5 diagrammes\n"
                "  --liste L       les valeurs explicites du parametre du chemin ( t, ou sigma avec --chemin conv )\n"
                "  --chemin C      melange ( le plancher ( 1 - t ) + t rho ) | conv ( rho convolee, largeur decroissante )  (melange)\n"
                "  --sigma0 S      conv : la largeur de la premiere etape, en fraction du cote du carre   (1)\n"
                "  --adaptatif     LE PAS ADAPTATIF : une etape dont le diagramme de depart a une cellule sous\n"
                "                  `--seuil x nu` est REFUSEE, et reprise a la valeur intermediaire geometrique\n"
                "  --seuil F       refus si une masse tombe sous F x nu ( les cellules qui MEURENT )        (0.05)\n"
                "  --seuil-residu R  refus si le depart max|a-nu|/nu depasse R ( la densite a trop BOUGE ), 0 : eteint  (2)\n"
                "  --seuil-dw D    refus si la CORRECTION DE POIDS reclamee, max|L^-1( nu - a )| / h^2, depasse D\n"
                "                  ( le predicteur qui porte le gradient : une resolution lineaire, pas un diagramme )  (0 : eteint)\n"
                "  --relevement    L'ETUDE DU RELEVEMENT : converger en --rel-depuis, pousser la densite a --rel-vers\n"
                "                  ( le pas qui pince ), puis comparer la direction de Newton aux directions de\n"
                "                  MOINDRES CARRES PONDERES  min sum C_i ( a_i - nu_i )^2, C = cloche sur les pincees\n"
                "  --rel-depuis L --rel-vers L   les deux valeurs du parametre du chemin      (0.125 -> 0)\n"
                "  --rel-seuil F   les cellules a relever : a_i < F nu                          (0.5)\n"
                "  --rel-largeurs L  les largeurs de cloche, en nombre d'aretes             (0,1,2,4)\n"
                "  --rel-kappas K  les hauteurs de cloche                                  (1,100,10000)\n"
                "  --local-nl      LE SOUS-PROBLEME LOCAL, resolu NON LINEAIREMENT : une boule de rayon R autour\n"
                "                  des cellules a problème, le bord impose a w0 + F d_newton, l'interieur resolu\n"
                "                  par un Newton amorti a merite BARRIERE. Les rayons : --rel-largeurs\n"
                "  --local-F F     les coefficients de relaxation du pas de Newton essaye        (0.25,0.5,1)\n"
                "  --local-pince F « cette cellule pince sous le pas » : a < F nu                      (0.5)\n"
                "  --local-cg K    budget du CG de Gauss-Newton local                                 (2000)\n"
                "  --newton-releve LA BOUCLE DE NEWTON AVEC RELEVEMENT ( § 15 ) : le pas est borne par « au plus\n"
                "                  --releve-ratio de cellules malades », puis elles sont reparees localement\n"
                "  --releve-ratio F  le plafond de cellules malades, en fraction de n                  (0.03)\n"
                "  --releve-mort F   sous F x nu la cellule est HORS DE PORTEE et le pas est refuse    (1e-3)\n"
                "  --releve-anneaux N  le nombre d'anneaux autour des malades                             (1)\n"
                "  --releve-off    la meme boucle SANS reparation : le temoin a boucle identique\n"
                "  --releve-mal-max M  LE PLUS GROS AMAS DE MALADES tolere ( 8 ) : `F` est choisi comme le plus\n"
                "                  grand pas dont l'union-find sur U ne fait pas depasser cette taille\n"
                "  --demo-homo N   LA TABLE QUI TRANCHE, sur les N premiers amas : pour chaque `lam`, combien de\n"
                "                  cellules interieures sont VIDES GEOMETRIQUEMENT ( intersection de demi-plans vide,\n"
                "                  ce qui infirmerait la construction ) et combien ont un POLYGONE mais une masse\n"
                "                  mesuree nulle ( aire en lam^2 sous la resolution : obstruction NUMERIQUE )\n"
                "  --placement-homo  PLACER L'INTERIEUR PAR HOMOTHETIE EXACTE, centree sur `x*`.\n"
                "                  `C_i( lam w0 + (1-lam)|p|^2 + 2 c.p + beta ) = lam C_i( w0 ) - c` : le motif entier se\n"
                "                  contracte, AUCUNE cellule ne disparait. Et `x*` n'est pas un centre geometrique : la\n"
                "                  condition pour que l'interieur gagne en `y` est `beta >= g( y )` avec\n"
                "                  `g( y ) = max_k ( 2 y.q_k - |q_k|^2 + v_k )` sur les cellules GELEES -- un max de\n"
                "                  fonctions affines, donc convexe, dont `{ g <= beta }` est exactement la region gagnee.\n"
                "  --releve-ratio-eps E  « MALADE » SUR LE RATIO `a / nu < E` au lieu du plancher absolu `eps`.\n"
                "                  Le plancher absolu vaut `0.5 min nu`, donc il est severe pour les grosses cibles et\n"
                "                  laxiste pour les petites -- le ratio traite toutes les cellules de la meme facon.\n"
                "  --releve-raideur R  LE PLAFOND D'ETALEMENT : `F` est choisi pour qu'aucun amas n'ait un etalement\n"
                "                  de poids depassant R fois `dist^2`. C'est la condition mesuree au § 15.12 -- un amas\n"
                "                  echoue quand le champ de poids y est trop raide pour l'espacement des germes --\n"
                "                  encodee a l'avance, au lieu du simple comptage de malades.  (0 : off)\n"
                "  --releve-beta B L'HORIZON ( 0.25 ) : les cellules dont le `U` bon marche tombe sous B voient\n"
                "                  leur vraie limite calculee PAR BISSECTION. B petit = peu de candidates, donc peu de\n"
                "                  bissections, et une meilleure evaluation la ou elle compte.  (0 : pas de bissection)\n"
                "  --releve-bis-tol T  precision relative de ces bissections                    (3e-2)\n"
                "  --u-max         `U` par le flux MAXIMAL ( une seule facette devore ) au lieu de la SOMME des\n"
                "                  flux sortants, qui est plus conservative et ne coute rien de plus\n"
                "  --releve-sans-u LA DYADIQUE A L'ANCIENNE : `F` cherche a tatons, un diagramme par essai\n"
                "  --diag-u        COMPARER LE PREDICTEUR U au pas reellement accepte : U_i = a_i / max_j c_ij ( d_j - d_i )\n"
                "                  est le temps que met la facette la plus devorante a consommer la cellule i.\n"
                "                  Imprime min U, le pas pris, et la courbe « plus gros amas en fonction de F »\n"
                "  --local-amas-max K  LE PLAFOND DE TAILLE : au-dela de K inconnues dans un amas, on DECLINE\n"
                "                  sans chercher. Un gros amas n'a pas de placement initial -- la pente du champ de\n"
                "                  poids imposee par son bord y depasse le carre de la distance entre germes -- donc\n"
                "                  chercher revient a bruler des millions de cellules pour abandonner. Decliner fait\n"
                "                  baisser F, et l'amas suivant sera plus petit.  (0 : sans plafond)\n"
                "  --local-scan N  BALAYER beta a lam = 0 pour les N premiers amas abandonnes, et imprimer le\n"
                "                  nombre d'interieures mortes ET de couronnes mortes : si les deux comptes ne\n"
                "                  s'annulent jamais ensemble, l'intervalle est VIDE et deux parametres ne suffisent pas\n"
                "  --local-dump N  DESSINER les N premiers amas abandonnes ( /tmp/amas_XX.svg ) : deux panneaux,\n"
                "                  w0 et w_F, inconnues / couronne / coupeurs en couleurs, cellules mortes marquees\n"
                "  --local-marge M ON S'ARRETE DES QUE L'ETAT EST SAIN AVEC MARGE ( plancher > M * local-eps ) :\n"
                "                  le sous-probleme cherche un point de depart pas trop mauvais, pas un minimum\n"
                "  --local-patch   LE RAFRAICHISSEMENT SUR LES PATCHS : la structure de voisinage est relue en\n"
                "                  force brute sur les seuls coupeurs des amas, PAS par un diagramme global.\n"
                "                  Implique --local-bord ( defaut 10 ) : sans l'aire du bord tenue, rien ne\n"
                "                  garantit que l'exterieur, qu'on ne relit plus, n'a pas bouge.\n"
                "  --local-bord L  LA COURONNE TENUE A SON AIRE : penalisation L sur ( A_i - A_i avant reparation ),\n"
                "                  parce que geler le POIDS d'une cellule de bord ne gele pas son AIRE  (0 : eteinte)\n"
                "  --local-grad    L'OBJECTIF EST LE GRADIENT D'AIRE  sum_( aretes ) ( A_i/nu_i - A_j/nu_j )^2,\n"
                "                  au lieu de la barriere : pas de cible absolue, donc AUCUNE pression sur la couronne\n"
                "  --local-refresh K  refaire la structure de voisinage tous les K sous-pas ( 0 : jamais )   (1)\n"
                "  --local-amas    UN PROBLEME PAR AMAS ( § 16 ) : chaque composante connexe a ses inconnues, sa\n"
                "                  continuation, et ses cellules calculees EN FORCE BRUTE -- sans arbre, sans set_weights\n"
                "  --local-saut    reparer APRES avoir saute a w0 + F d ( § 14 ) au lieu de la CONTINUATION\n"
                "                  sur le bord depuis w0 ( § 16, le defaut ) -- le saut bute sur les cellules mortes\n"
                "  --local-lin     merite lineaire au lieu de la barriere ( le temoin )\n"
                "  --local-maxit K --local-tol T --local-eps F   ( 60, 1e-3, 0.05 )\n"
                "  --rel-mus M     L'AMORTI GLOBAL ( L C L + mu I ) d = L C ( nu - a ), mu en fraction de la diagonale\n"
                "                  moyenne. C'est lui qui rend les poids visibles SANS restreindre d     (vide)\n"
                "  --rel-sans-identite / --rel-sans-newton   enlever le temoin d'identite / le Newton d'apres\n"
                "  --rel-solve     LA SEULE MESURE QUI TRANCHE : Newton mene a CONVERGENCE depuis le point releve,\n"
                "                  en diagrammes, contre Newton depuis le point de depart\n"
                "  --rel-cg K      budget du CG sur les equations normales                     (4000)\n"
                "  --rel-csv F --rel-cas NOM   le balayage complet en CSV\n"
                "  --diagnostic    imprimer les trois predicteurs de difficulte au depart de chaque etape, et son cout\n"
                "  --chooseur      LE CHOOSEUR DE PAS : le pas suivant est choisi pour que la CORRECTION DE POIDS\n"
                "                  reclamee reste voisine de --amp-cible. Gratuit : Newton rapporte deja sa premiere\n"
                "                  direction. Remplace l'echelle fixe ; le refus ( --adaptatif ) reste le filet.\n"
                "  --amp-cible A   la correction visee, en unites de h^2 = 1/n                        (800)\n"
                "  --lam-min L     sous cette valeur du parametre, viser directement la cible  (1e-6, ou 1/3 pixel en conv)\n"
                "  --max-etapes K  garde-fou sur le nombre total d'etapes                               (200)\n"
                "  --ordre K       0 ( les poids precedents ) | 1 ( la tangente en t, GRATUITE )   (0)\n"
                "  --diracs D      uniforme | rho ( germes tires selon l'image )              (uniforme)\n"
                "  --acc A         double | float : le flottant du STOCKAGE de l'image ( la marche reste en double,\n"
                "                  cf. § 19 ) ; `--kernel` est celui de la geometrie                        (double)\n"
                "  --check         la masse contre le DECOUPAGE EN PIXELS, la hessienne contre des differences finies,\n"
                "                  et la simple precision contre la double\n"
                "  --amg-tol T     le residu RELATIF demande au solveur lineaire ( defaut 1e-10 -- une\n"
                "                  direction de Newton amortie n'en demande pas tant )\n"
                "  --chrono        le bord contre le decoupage, sur les memes cellules ( `errand -x` )\n"
                "  --solver S      amg ( AMGCL, defaut ) | chol ( Eigen, sequentiel ) | mg ( § 17 )\n"
                "  --amg-refaire N  la hierarchie d'AMGCL gardee N resolutions ( defaut 4 )\n"
                "  --mg-lisseur 0|1|2  Jacobi amorti | spai0 | Chebyshev ; --mg-cheb R le rapport\n"
                "                  `lmax / lmin` de Chebyshev ; --mg-recycle K le sous-espace garde\n"
                "  --mg-lisse 0|1 --mg-refaire N --mg-omega-p W --mg-nu N --mg-gros G --mg-k K\n"
                "  --mg-stop S --mg-exact 0|1   les reglages du multigrille maison ( § 17 ) :\n"
                "                  `lisse` la prolongation lissee, `refaire` la hierarchie gardee N\n"
                "                  resolutions, `exact` le fond resolu au lieu d'etre lisse\n"
                "  --amg-var V     0 = agregation+spai0 | 1 = agregation+GS | 2 = Ruge-Stuben+GS   (2)\n"
                "  --pas P         essais ( KMT, defaut ) | essai-limites ( les cellules pincees relevees seules )\n"
                "  --newton-tol T --newton-max K --t-min T --beta0 B --mult-ok M --facteur F --lim-tol T --residu R --quiet\n"
                "  --ecrire FILE   les poids trouves, au format de cases/\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    a.finalise();

    const Opts &oc = o;
    return dispatch<2>( a, [ & ]( auto tag ) {
        using PD = typename decltype( tag )::type;
        auto avec = [ & ]( auto ta ) {
            using TA = typename decltype( ta )::type;
            ImageT<TA> im;
            if ( ! oc.pgm.empty() ) {
                if ( ! lit_pgm( oc.pgm.c_str(), im ) ) { std::printf( "image illisible : %s\n", oc.pgm.c_str() ); return 1; }
            } else
                im = image_synthese<TA>( oc.taille, oc.trou );
            im.normalise();
            if ( oc.contraste != 1 ) {                   // le contraste FIGE, avant la continuation
                for ( TA &u : im.v ) u = TA( ( 1 - oc.contraste ) + oc.contraste * double( u ) );
                im.normalise();
            }
            TA vmin = im.v[ 0 ], vmax = im.v[ 0 ], vpos = im.v[ 0 ] > 0 ? im.v[ 0 ] : TA( 1e30 );
            SI nz = 0;
            for ( TA u : im.v ) {
                vmin = std::min( vmin, u ); vmax = std::max( vmax, u );
                if ( u > 0 ) vpos = std::min( vpos, u );
                nz += ( u == 0 );
            }
            std::printf( "  image %d x %d, rho dans [ %.4g, %.4g ] ( contraste %.1f:1 sur les pixels NON NULS ), %d pixels a zero ( %.1f %% ),"
                         " mesure en %s, noyau en %s\n",
                         im.W, im.H, double( vmin ), double( vmax ), double( vmax ) / double( vpos ),
                         int( nz ), 100.0 * nz / im.v.size(), sizeof( TA ) == 4 ? "float" : "double", a.kernel.c_str() );

            const Nuage<2> nu = oc.diracs == "rho" ? nuage_selon( im, a.n, a.graine ) : nuage_uniforme<2>( a.n, a.graine, 0 );
#ifdef SF_EIGEN
            if ( oc.solver == "chol" ) {
                Cholesky lin;
                return lance<PD>( a, oc, nu, lin, im );
            }
#endif
            if ( oc.solver == "mg" ) {
                Mg lin;
                lin.tol = TF( oc.amgtol );
                lin.nu = oc.mg_nu; lin.gros = oc.mg_gros; lin.kcycle = oc.mg_k; lin.stop = oc.mg_stop;
                lin.exact = oc.mg_exact != 0;
                lin.lisse = oc.mg_lisse != 0;
                lin.agreg = oc.mg_agreg;
                lin.lisseur = oc.mg_lisseur;
                lin.cheb = TF( oc.mg_cheb );
                lin.recycle = oc.mg_recycle;
                lin.refaire = oc.mg_refaire;
                lin.omega_p = TF( oc.mg_omega_p );
                lin.tronque = TF( oc.mg_tronque );
                lin.force = TF( oc.mg_force );
                lin.trace = oc.mg_trace;
                return lance<PD>( a, oc, nu, lin, im );
            }
#ifdef SF_AMGCL
            Amg lin;
            lin.variante = oc.amgvar;
            lin.tol = TF( oc.amgtol );
            lin.refaire = oc.amg_refaire;
            return lance<PD>( a, oc, nu, lin, im );
#else
            // AMGCL ABSENT : on retombe sur Cholesky plutot que d'abandonner, maintenant que `amg`
            // est le defaut. Le binaire reste utilisable, il est seulement plus lent.
            std::printf( "  AMGCL absent : on retombe sur Cholesky ( sequentiel )\n" );
            Cholesky lin;
            return lance<PD>( a, oc, nu, lin, im );
#endif
        };
        if ( oc.acc == "float" ) return avec( std::type_identity<float>{} );
        return avec( std::type_identity<double>{} );
    } );
}
