#pragma once

#include "AaBsp.h"
#include "AaBspMemo.h"
#include "PowerDiagram.h"
#include "parallel.h"
#include <algorithm>
#include <chrono>
#include <type_traits>
#include <cmath>
#include <cstdio>
#include <vector>

/// LE SOLVEUR LINEAIRE N'EST PAS L'OBJET DE L'ETUDE, et il ne faut pas qu'il le devienne. Le banc
/// est en C++ nu pour tout ce qui touche a la GEOMETRIE ; pour la factorisation creuse, la
/// reecrire serait long, mal fait, et sans rapport avec la question posee. On prend donc Eigen
/// s'il est la -- en-tetes seuls, rien a lier -- et on retombe sur un gradient conjugue maison
/// sinon. `--solver cg` force le second pour pouvoir les comparer.
#if __has_include( <eigen3/Eigen/SparseCholesky> )
#  define PD2D_EIGEN 1
#  include <eigen3/Eigen/SparseCholesky>
#  include <eigen3/Eigen/SparseCore>
#endif

/// AMGCL : un multigrille ALGEBRIQUE. C'est le bon outil pour ce systeme et pas un luxe -- la
/// hessienne est le laplacien d'un graphe presque planaire, exactement l'objet pour lequel
/// l'agregation lissee est faite. Contre une factorisation, ce qu'on echange : le cout ne croit
/// plus qu'en `O( n )` au lieu du remplissage de Cholesky, et la hierarchie se construit en
/// parallele.
#if __has_include( <amgcl/make_solver.hpp> )
#  define PD2D_AMGCL 1
#  include <amgcl/adapter/crs_tuple.hpp>
#  include <amgcl/amg.hpp>
#  include <amgcl/backend/builtin.hpp>
#  include <amgcl/coarsening/smoothed_aggregation.hpp>
#  include <amgcl/make_solver.hpp>
#  include <amgcl/coarsening/ruge_stuben.hpp>
#  include <amgcl/relaxation/gauss_seidel.hpp>
#  include <amgcl/relaxation/spai0.hpp>
#  include <amgcl/solver/cg.hpp>
#endif

namespace pd2d {

/// LES POIDS CHANGENT, PAS LES POSITIONS. C'est le regime de Newton : la permutation, les boites
/// et la forme de l'arbre sont les memes d'une iteration a l'autre, seuls les MAJORANTS bougent.
/// Les refaire ne coute aucun `nth_element` -- trois passes de `weight_majorant` par noeud -- la ou
/// reconstruire paierait le tri a chaque fois.
inline void refresh_weights( AaBsp &tr, const TF *W, int nb_threads, Split split, bool pin ) {
    const SI n = SI( tr.order.size() );
    if ( SI( tr.pw.size() ) != n )
        tr.pw.resize( n );
    for ( SI k = 0; k < n; ++k )
        tr.pw[ k ] = W[ tr.order[ k ] ];
    parallel_for( SI( tr.nodes.size() ), nb_threads, split, pin, [ & ]( SI i, int ) {
        AaBsp::Node &nd = tr.nodes[ i ];
        nd.wm = weight_majorant( nd.beg, nd.end, [ & ]( SI k, TF &x, TF &y, TF &w ) {
            x = tr.px[ k ]; y = tr.py[ k ]; w = tr.pw[ k ];
        } );
    } );
}

/// Le meme, pour l'arbre qui se souvient : seuls les MAJORANTS bougent, les souvenirs restent
/// valides -- un souvenir perime ne rend pas un resultat faux, il rend une coupe `unchanged`.
inline void refresh_weights( AaBspMemo &t, const TF *W, int nb_threads, Split split, bool pin ) {
    refresh_weights( t.tr, W, nb_threads, split, pin );
}

/// L'arbre nu, derriere l'accelerateur qui l'enveloppe : `psi_min` marche sur des boites et des
/// germes, pas sur des souvenirs.
inline const AaBsp &base( const AaBsp &t ) { return t; }
inline const AaBsp &base( const AaBspMemo &t ) { return t.tr; }

/// `psi( x ) = min_k ( |x - p_k|^2 - w_k )`, cherche dans l'arbre plutot que balaye.
///
/// C'est LA PROLONGATION du multi-echelle. Recopier le poids du representant sur tout son paquet --
/// ce que la litterature decrit comme « donner a chaque germe fin le poids de son representant » --
/// est faux des que le nuage est serre : deux germes voisins de paquets differents recoivent alors
/// des poids qui different d'un SAUT, et une cellule est vide des que ce saut depasse le carre de
/// la distance qui les separe. Sur le nuage de lignes, des germes a 1e-5 l'un de l'autre recevaient
/// des poids ecartes de 1e-3, soit 1e8 fois trop : mesure, chaque prolongation produisait des
/// cellules vides et un residu PIRE que le niveau qu'elle venait de resoudre (121 fois la cible
/// apres un niveau grossier converge a 1e-3).
///
/// La bonne prolongation est la C-TRANSFORMEE : on demande a la parabole du germe fin de TOUCHER le
/// potentiel grossier en `p_i`, donc `-w_i = psi_grossier( p_i )`, soit
///
///     w_i = w_k - |p_i - p_k|^2   ou `k` est la cellule grossiere qui contient `p_i`
///
/// Elle vaut `w_k` exactement sur le germe grossier, elle est CONTINUE en `p_i` -- les deux
/// expressions coincident sur la bissectrice de puissance -- donc deux germes fins voisins recoivent
/// des poids voisins, et le saut disparait.
///
/// L'elagage : sur une boite, `w` est majore par son majorant affine, donc `|x - p|^2 - w` est
/// minore par `dist^2( x, boite ) - max_boite( w )`. C'est le meme argument que `may_be_cut`, en
/// beaucoup plus simple parce qu'il n'y a pas de cellule, juste un point.
inline TF psi_min( const AaBsp &tr, TF x, TF y, SI hors = -1 ) {
    TF best = 1e300;
    tr.for_each_candidate_at( x, y, hors,
        [ & ]( TF lox, TF loy, TF hix, TF hiy, const WMaj &wm ) {
            const TF ex = x < lox ? lox - x : ( x > hix ? x - hix : TF( 0 ) );
            const TF ey = y < loy ? loy - y : ( y > hiy ? y - hiy : TF( 0 ) );
            const TF sx = TF( wm.ax ) * lox, tx = TF( wm.ax ) * hix;
            const TF sy = TF( wm.ay ) * loy, ty = TF( wm.ay ) * hiy;
            const TF wmax = ( sx > tx ? sx : tx ) + ( sy > ty ? sy : ty ) + wm.b;
            return ex * ex + ey * ey - wmax < best;
        },
        [ & ]( TF px, TF py, TF w, SI ) {
            const TF dx = px - x, dy = py - y;
            const TF h = dx * dx + dy * dy - w;
            if ( h < best )
                best = h;
            return true;
        },
        [] { return TF( 0 ); } );
    return best;
}

/// LE RATTRAPAGE DES CELLULES VIDES, et c'est ce qui rend le multi-echelle utilisable.
///
/// Newton amorti n'est defini que tant qu'AUCUNE cellule n'est vide : une cellule vide sort du
/// graphe de Laguerre, le laplacien se disconnecte, et sa ligne devient une equation sans rapport
/// avec la geometrie. Or aucune prolongation ne le garantit -- mesure sur l'uniforme, de niveau en
/// niveau : 0, 1, 9, 59, 166, 180 puis 758 cellules vides sur 20000. Trois pour cent, et Newton
/// stagne des sa premiere iteration.
///
/// Relever `w_i` jusqu'a `max_{j != i} ( w_j - |p_i - p_j|^2 )` met le germe DANS sa cellule, donc
/// la rend non vide. Le membre de droite est exactement `-psi( p_i )` calcule sans `i`.
///
/// ESSAYE ET REJETE : imposer cette borne a TOUS les germes. C'est la condition de c-concavite, et
/// elle est bien trop forte : sur l'uniforme a 4096 germes elle relevait 3229 germes sur 4096 --
/// 79 % -- alors que le diagramme n'avait aucune cellule vide, et l'iteration de point fixe n'avait
/// pas converge apres quarante passes. Ce qu'on veut n'est pas « chaque germe dans sa cellule »,
/// c'est « aucune cellule vide » : on ne releve donc que les fautives, qu'on trouve en mesurant.
template<class PD, class Tree>
SI rattrape_vides( PD &pd, Tree &tr, const TF *X, const TF *Y, SI m, std::vector<TF> &w,
                   TF marge, int passes, int th, Split sp, bool pin, bool trace ) {
    std::vector<TF> a;
    SI nv = 0;
    for ( int p = 0; p < passes; ++p ) {
        refresh_weights( tr, w.data(), th, sp, pin );
        pd.measures( a, th, sp, pin );
        nv = 0;
        for ( SI i = 0; i < m; ++i )
            nv += ! ( a[ i ] > 0 );
        if ( trace )
            std::printf( "%s %d", p ? "" : "       cellules vides :", int( nv ) );
        if ( ! nv )
            break;
        std::vector<TF> w2( w );
        parallel_for( m, th, sp, pin, [ & ]( SI i, int ) {
            if ( a[ i ] > 0 )
                return;
            // `marge` a la dimension d'une AIRE, comme un poids : elle donne a la cellule relevee
            // un rayon de l'ordre de celui qu'elle doit finir par avoir, au lieu de la laisser
            // exactement sur la frontiere.
            w2[ i ] = -psi_min( base( tr ), X[ i ], Y[ i ], i ) + marge;
        } );
        w.swap( w2 );
    }
    if ( trace ) {
        std::printf( "\n" );
        std::fflush( stdout );
    }
    return nv;
}

/// LE TRANSPORT SEMI-DISCRET, RESOLU. Jusqu'ici le banc ne mesurait qu'UNE construction de
/// diagramme, sur des poids donnes d'avance. Ici on resout vraiment
///
///     trouver `w` tel que `|Lag_i( w )| = nu_i` pour tout `i`
///
/// ce qui est le seul regime dans lequel un accelerateur servira : une dizaine de diagrammes
/// enchaines, sur des poids qui bougent de moins en moins.
///
/// = Le dual, et pourquoi Newton
///
///     Phi( w ) = integrale min_i ( |x - p_i|^2 - w_i ) dx + somme_i w_i nu_i
///
/// est CONCAVE, de gradient `nu_i - |Lag_i( w )|`. Sa hessienne est, au signe pres, le LAPLACIEN du
/// graphe de Laguerre :
///
///     c_ij = |arete commune| / ( 2 |p_i - p_j| ),   L_ii = somme_j c_ij,   L_ij = -c_ij
///
/// Un laplacien de graphe connexe a pour noyau les CONSTANTES -- et c'est exact, pas un accident
/// numerique : ajouter la meme constante a tous les poids ne change aucune cellule. On fixe donc
/// `w_0 = 0`, ce qui revient a rayer la ligne et la colonne 0 et rend le systeme defini positif.
///
/// = L'amortissement, et ce qu'il protege
///
/// La hessienne n'est definie que tant qu'aucune cellule n'est vide -- une cellule vide sort du
/// graphe, le laplacien se disconnecte, et le pas n'a plus de sens. Le critere de
/// Kitagawa-Merigot-Thibert demande donc DEUX choses au pas essaye : qu'aucune aire ne descende
/// sous un plancher `eps` fixe au depart, et que le residu decroisse d'au moins `1 - t / 2`. On
/// part de `w = 0`, donc du diagramme de VORONOI, dont aucune cellule n'est vide par construction :
/// le depart est toujours admissible.
struct Newton {
    /// Une arete du graphe de Laguerre, vue DEPUIS la cellule `i` : c'est elle qui en a mesure la
    /// longueur. Chaque paire est donc vue deux fois, une par cellule, et les deux valeurs ne
    /// different qu'a l'arrondi.
    struct Arete {
        SI i, j;
        TF c;
    };

    /// Le laplacien en CSR, diagonale a part -- elle est la somme des hors-diagonaux, donc la
    /// ranger separement evite de la chercher dans la ligne a chaque produit.
    struct Systeme {
        SI              n = 0;
        std::vector<SI> row, col;
        std::vector<TF> c;          ///< `c_ij > 0` ; le signe moins est dans le produit
        std::vector<TF> dia;

        /// `y = L x`, avec `x_0` FORCE A ZERO et la ligne 0 rayee : le systeme reduit, sans le
        /// construire.
        void mul( const std::vector<TF> &x, std::vector<TF> &y ) const {
            y[ 0 ] = 0;
            for ( SI i = 1; i < n; ++i ) {
                TF s = dia[ i ] * x[ i ];
                for ( SI k = row[ i ]; k < row[ i + 1 ]; ++k )
                    if ( const SI j = col[ k ] )
                        s -= c[ k ] * x[ j ];
                y[ i ] = s;
            }
        }
    };

    SI              n  = 0;         ///< les germes DU NIVEAU
    std::vector<TF> nu;             ///< la mesure cible, par germe -- au niveau grossier un germe
                                    ///< porte la masse AGREGEE de tout son paquet
    TF              eps = 0;        ///< le plancher d'aire de l'amortissement
    std::vector<TF> w;              ///< les poids courants, `w[ 0 ] == 0`

    bool        direct = true;      ///< factoriser plutot qu iterer, si Eigen est la
    const char *fin = "?";          ///< pourquoi la boucle s est arretee
    int         quel = 0;           ///< 0 = AMG, 1 = Cholesky, 2 = gradient conjugue
    int         variante = 0;       ///< AMG : 0 = SA+spai0, 1 = SA+Gauss-Seidel, 2 = RS+GS
    TF          reste = 0;          ///< le `max_i |a_i - nu| / nu` atteint
    double t_tri = 0, t_ana = 0, t_fac = 0, t_sol = 0;  ///< le detail de la factorisation
    int    nb_iter = 0, nb_diag = 0, nb_cg = 0, nb_recul = 0, nb_ana = 0;
    TF     pire_lin = 0;            ///< le pire residu relatif rendu par le solveur lineaire
    double t_diag = 0, t_maj = 0, t_syst = 0, t_cg = 0;

    static double now() {
        using namespace std::chrono;
        return duration<double>( steady_clock::now().time_since_epoch() ).count();
    }

    /// LES AIRES ET LES ARETES. Une passe de plus par cellule -- longueur d'arete, distance des
    /// deux germes -- faite une fois par iteration acceptee et non a chaque pas essaye.
    template<class Cell, class PD, class Tree>
    void aires_et_aretes( PD &pd, Tree &tr, const TF *X, const TF *Y, const TF *W,
                          std::vector<TF> &res, std::vector<Arete> &ar, int th, Split sp,
                          bool pin ) {
        double t0 = now();
        refresh_weights( tr, W, th, sp, pin );
        t_maj += now() - t0;
        t0 = now();

        res.assign( n, TF( 0 ) );
        std::vector<std::vector<Arete>> par( std::max( th, 1 ) );
        parallel_for( n, th, sp, pin, [ & ]( SI k, int t ) {
            Cell c;
            pd.make_cell( c, k );
            const SI i = tr.seed_id( k );
            res[ i ] = c.measure();
            const TF p0x = X[ i ], p0y = Y[ i ];
            for ( SI v = 0; v < c.nb; ++v ) {
                const SI j = c.cid[ v ];
                if ( j < 0 )                            // une arete du DOMAINE, pas un voisin
                    continue;
                const SI u = v + 1 < c.nb ? v + 1 : 0;
                const TF ex = c.vx[ u ] - c.vx[ v ], ey = c.vy[ u ] - c.vy[ v ];
                const TF dx = X[ j ] - p0x, dy = Y[ j ] - p0y;
                const TF d2 = dx * dx + dy * dy;
                if ( ! ( d2 > 0 ) )
                    continue;
                const TF cc = std::sqrt( ( ex * ex + ey * ey ) / d2 ) / 2;
                par[ t ].push_back( Arete{ i, j, cc } );
            }
        } );
        ar.clear();
        for ( auto &v : par )
            ar.insert( ar.end(), v.begin(), v.end() );
        // la passe a eu lieu : les souvenirs sont remplis, on peut les rejouer
        if constexpr ( requires ( const Tree &t ) { t.arme(); } )
            tr.arme();
        t_diag += now() - t0;
        ++nb_diag;
    }

    /// LE LAPLACIEN, SANS UN SEUL TRI. Chaque arete sait a quelle LIGNE elle appartient, donc un
    /// comptage puis une somme prefixe suffisent a placer tout le monde -- deux passes lineaires.
    ///
    /// ESSAYE ET REJETE : trier les aretes par `( i, j )` pour apparier les deux mesures d'une meme
    /// arete et les moyenner. C'etait defendable -- un gradient conjugue veut une matrice vraiment
    /// symetrique -- mais le tri de douze millions d'enregistrements coutait 4.2 s a n=1e6, soit
    /// PLUS que le diagramme lui-meme (3.9 s). Et il ne servait a rien : les deux mesures ne
    /// different qu'a 1e-16 relatif, cinq ordres de grandeur sous la tolerance du solveur lineaire.
    /// En prenant la valeur de la ligne, chaque ligne somme en outre EXACTEMENT a zero, donc les
    /// constantes restent exactement dans le noyau.
    void assemble( const std::vector<Arete> &ar, Systeme &S ) const {
        S.n = n;
        S.row.assign( n + 1, 0 );
        for ( const Arete &e : ar )
            ++S.row[ e.i + 1 ];
        for ( SI i = 0; i < n; ++i )
            S.row[ i + 1 ] += S.row[ i ];

        S.col.resize( S.row[ n ] );
        S.c.resize( S.row[ n ] );
        S.dia.assign( n, TF( 0 ) );
        std::vector<SI> at( S.row.begin(), S.row.end() - 1 );
        for ( const Arete &e : ar ) {
            const SI p = at[ e.i ]++;
            S.col[ p ] = e.j;
            S.c[ p ] = e.c;
            S.dia[ e.i ] += e.c;
        }
        // une cellule sans voisin ne peut pas arriver tant qu'aucune n'est vide, mais une ligne
        // nulle rendrait le systeme singulier SANS LE DIRE : on la neutralise.
        for ( SI i = 0; i < n; ++i )
            if ( ! ( S.dia[ i ] > 0 ) )
                S.dia[ i ] = 1;
    }

    /// GRADIENT CONJUGUE preconditionne par la diagonale. Le laplacien reduit est defini positif et
    /// CREUX -- six voisins par cellule en 2D -- donc le produit matrice-vecteur est tout le cout.
    /// Pas de factorisation : elle demanderait une renumerotation et une bibliotheque, et ce banc
    /// n'en a pas.
    int cg( const Systeme &S, const std::vector<TF> &b, std::vector<TF> &x, TF tol,
            int maxit ) const {
        std::vector<TF> r( n ), z( n ), p( n ), q( n );
        x.assign( n, TF( 0 ) );
        TF nb2 = 0;
        for ( SI i = 0; i < n; ++i ) { r[ i ] = i ? b[ i ] : TF( 0 ); nb2 += r[ i ] * r[ i ]; }
        if ( ! ( nb2 > 0 ) )
            return 0;
        TF rz = 0;
        for ( SI i = 0; i < n; ++i ) {
            z[ i ] = i ? r[ i ] / S.dia[ i ] : TF( 0 );
            rz += r[ i ] * z[ i ];
        }
        p = z;
        const TF cible = tol * tol * nb2;
        int it = 0;
        for ( ; it < maxit; ++it ) {
            S.mul( p, q );
            TF pq = 0;
            for ( SI i = 1; i < n; ++i ) pq += p[ i ] * q[ i ];
            if ( ! ( pq > 0 ) )
                break;
            const TF al = rz / pq;
            TF nr2 = 0;
            for ( SI i = 1; i < n; ++i ) {
                x[ i ] += al * p[ i ];
                r[ i ] -= al * q[ i ];
                nr2 += r[ i ] * r[ i ];
            }
            if ( nr2 <= cible ) { ++it; break; }
            TF rz2 = 0;
            for ( SI i = 1; i < n; ++i ) { z[ i ] = r[ i ] / S.dia[ i ]; rz2 += r[ i ] * z[ i ]; }
            const TF be = rz2 / rz;
            rz = rz2;
            for ( SI i = 1; i < n; ++i ) p[ i ] = z[ i ] + be * p[ i ];
        }
        return it;
    }

#ifdef PD2D_EIGEN
    /// LA FACTORISATION. Le systeme reduit (le germe 0 raye) est symetrique defini positif et
    /// creux ; `SimplicialLDLT` avec renumerotation AMD est ce qu'un vrai code de transport
    /// utilise. Elle est REFAITE a chaque iteration -- la hessienne change avec les cellules --
    /// et c'est bien ce qu'il faut chronometrer.
    using SpM = Eigen::SparseMatrix<double>;
    Eigen::SimplicialLDLT<SpM, Eigen::Lower, Eigen::AMDOrdering<int>> so;
    std::vector<SI> motif;              ///< le motif de la derniere analyse symbolique

    /// LE MOTIF NE BOUGE PRESQUE PAS. Le graphe de Laguerre change de quelques aretes par
    /// iteration au debut et de zero a la fin, alors que la renumerotation AMD et l'analyse
    /// symbolique, elles, coutent le tiers de la factorisation. On les refait donc SEULEMENT
    /// quand le motif a change -- compare tel quel, ce qui coute une passe lineaire.
    bool chol( const Systeme &S, const std::vector<TF> &b, std::vector<TF> &x ) {
        const double tdeb = now();
        const SI m = n - 1;
        std::vector<Eigen::Triplet<double>> tri;
        tri.reserve( size_t( S.row[ n ] ) / 2 + n );
        for ( SI i = 1; i < n; ++i ) {
            tri.emplace_back( i - 1, i - 1, double( S.dia[ i ] ) );
            for ( SI k = S.row[ i ]; k < S.row[ i + 1 ]; ++k ) {
                const SI j = S.col[ k ];
                if ( j >= 1 && j < i )              // le triangle INFERIEUR seul
                    tri.emplace_back( i - 1, j - 1, -double( S.c[ k ] ) );
            }
        }
        SpM A( m, m );
        A.setFromTriplets( tri.begin(), tri.end() );
        double t0 = now();
        t_tri += t0 - tdeb;
        std::vector<SI> mot( A.outerIndexPtr(), A.outerIndexPtr() + m + 1 );
        mot.insert( mot.end(), A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros() );
        if ( mot != motif ) {
            so.analyzePattern( A );
            motif.swap( mot );
            ++nb_ana;
        }
        double t1 = now();
        t_ana += t1 - t0;
        so.factorize( A );
        t0 = now();
        t_fac += t0 - t1;
        if ( so.info() != Eigen::Success )
            return false;
        Eigen::VectorXd rb( m );
        for ( SI i = 1; i < n; ++i )
            rb[ i - 1 ] = double( b[ i ] );
        const Eigen::VectorXd sol = so.solve( rb );
        t_sol += now() - t0;
        if ( so.info() != Eigen::Success )
            return false;
        x.assign( n, TF( 0 ) );
        for ( SI i = 1; i < n; ++i )
            x[ i ] = TF( sol[ i - 1 ] );
        return true;
    }
#endif

    /// LE SYSTEME REDUIT EN CRS, colonnes TRIEES. Le germe 0 est raye -- pas neutralise : une
    /// ligne identite laissee dans la matrice donnerait a l'agregation un noeud isole a traiter,
    /// et ce n'est pas ce qu'on veut lui montrer.
    void crs( const Systeme &S, std::vector<int> &ptr, std::vector<int> &col,
              std::vector<double> &val ) const {
        const SI m = n - 1;
        ptr.assign( m + 1, 0 );
        for ( SI i = 1; i < n; ++i ) {
            int k = 1;                                  // la diagonale, toujours presente
            for ( SI e = S.row[ i ]; e < S.row[ i + 1 ]; ++e )
                k += S.col[ e ] >= 1;
            ptr[ i ] = k;
        }
        for ( SI i = 0; i < m; ++i )
            ptr[ i + 1 ] += ptr[ i ];
        col.resize( ptr[ m ] );
        val.resize( ptr[ m ] );
        for ( SI i = 1; i < n; ++i ) {
            int k = ptr[ i - 1 ];
            col[ k ] = int( i - 1 );  val[ k ] = double( S.dia[ i ] );  ++k;
            for ( SI e = S.row[ i ]; e < S.row[ i + 1 ]; ++e )
                if ( S.col[ e ] >= 1 ) {
                    col[ k ] = int( S.col[ e ] - 1 );  val[ k ] = -double( S.c[ e ] );  ++k;
                }
            // par insertion : sept entrees par ligne, et AMGCL veut des colonnes croissantes
            for ( int u = ptr[ i - 1 ] + 1; u < k; ++u ) {
                const int c = col[ u ];
                const double v = val[ u ];
                int j = u;
                for ( ; j > ptr[ i - 1 ] && col[ j - 1 ] > c; --j ) {
                    col[ j ] = col[ j - 1 ]; val[ j ] = val[ j - 1 ];
                }
                col[ j ] = c; val[ j ] = v;
            }
        }
    }

#ifdef PD2D_AMGCL
    using AmgBack = amgcl::backend::builtin<double>;
    template<template<class> class Coarse, template<class> class Relax>
    using AmgOf = amgcl::make_solver<amgcl::amg<AmgBack, Coarse, Relax>, amgcl::solver::cg<AmgBack>>;

    /// LE MULTIGRILLE ALGEBRIQUE. `smoothed_aggregation` + `spai0` : l'agregation lissee suppose
    /// que le noyau local est la CONSTANTE, ce qui est exactement vrai d'un laplacien de graphe,
    /// et `spai0` est un lisseur diagonal donc parallelisable sans coloration.
    bool amg( const Systeme &S, const std::vector<TF> &b, std::vector<TF> &x, TF tol, int maxit ) {
        const SI m = n - 1;
        double t0 = now();
        std::vector<int> ptr, col;
        std::vector<double> val;
        crs( S, ptr, col, val );
        double t1 = now();
        t_tri += t1 - t0;

        std::vector<double> rb( m ), sol( m, 0.0 );
        for ( SI i = 1; i < n; ++i )
            rb[ i - 1 ] = double( b[ i ] );
        int it = 0;
        double err = 0;

        // Trois hierarchies, parce que le nuage decide. L'agregation lissee suppose que le noyau
        // local est la CONSTANTE -- vrai d'un laplacien de graphe -- mais elle suppose aussi que
        // les poids d'aretes sont comparables, et sur un nuage de lignes les `c_ij` s'etalent sur
        // plusieurs ordres de grandeur. Ruge-Stuben, qui choisit ses noeuds grossiers arete par
        // arete, n'a pas cette hypothese.
        auto lance = [ & ]( auto tag ) {
            using Solv = typename decltype( tag )::type;
            typename Solv::params prm;
            prm.solver.tol = double( tol );
            prm.solver.maxiter = maxit;
            Solv so( std::tie( m, ptr, col, val ), prm );
            const double ta = now();
            t_ana += ta - t1;                           // la CONSTRUCTION de la hierarchie
            ++nb_ana;
            std::tie( it, err ) = so( rb, sol );
            t_fac += now() - ta;
        };
        using SaSpai = AmgOf<amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>;
        using SaGs   = AmgOf<amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::gauss_seidel>;
        using RsGs   = AmgOf<amgcl::coarsening::ruge_stuben, amgcl::relaxation::gauss_seidel>;
        if ( variante == 1 )      lance( std::type_identity<SaGs>{} );
        else if ( variante == 2 ) lance( std::type_identity<RsGs>{} );
        else                      lance( std::type_identity<SaSpai>{} );
        nb_cg += it;
        pire_lin = std::max( pire_lin, TF( err ) );

        x.assign( n, TF( 0 ) );
        for ( SI i = 1; i < n; ++i )
            x[ i ] = TF( sol[ i - 1 ] );
        return err < 1;                                 // `1` : le solveur n'a rien fait du tout
    }
#endif

    static TF norme2( const std::vector<TF> &v ) {
        TF s = 0;
        for ( TF x : v ) s += x * x;
        return std::sqrt( s );
    }

    /// LA BOUCLE. Rend `true` si le critere d'arret a ete atteint.
    ///
    /// Une iteration coute UN diagramme par pas essaye, et rien de plus : le pas accepte livre a la
    /// fois les aires (pour le residu) et les aretes (pour la hessienne suivante).
    ///
    /// `w_init` est le point de depart -- zero au niveau le plus grossier, la PROLONGATION du
    /// niveau precedent ensuite. C'est tout ce que le multi-echelle demande a cette fonction.
    template<class Cell, class PD, class Tree>
    bool resout( PD &pd, Tree &tr, const TF *X, const TF *Y, const std::vector<TF> &w_init,
                 TF tol, int maxit, TF cgtol, int cgmax, int th, Split sp, bool pin, bool trace ) {
        std::vector<TF> a, a2, b, d, w2;
        std::vector<Arete> ar, ar2;
        Systeme S;

        w = w_init;
        const TF g = w[ 0 ];
        for ( SI i = 0; i < n; ++i )                    // la jauge : `w_0 = 0`, imposee ici et
            w[ i ] -= g;                                // maintenue par `d[ 0 ] = 0` ensuite
        aires_et_aretes<Cell>( pd, tr, X, Y, w.data(), a, ar, th, sp, pin );

        for ( int it = 0; it < maxit; ++it ) {
            TF plancher = a[ 0 ] / nu[ 0 ], pire = 0;
            SI nvide = 0;                               // combien de cellules VIDES : c'est la
            b.assign( n, TF( 0 ) );                     // seule chose qui sorte Newton de son domaine
            for ( SI i = 0; i < n; ++i ) {
                nvide += ! ( a[ i ] > 0 );
                plancher = std::min( plancher, a[ i ] / nu[ i ] );
                b[ i ] = nu[ i ] - a[ i ];              // `-r`, le second membre de Newton
                pire = std::max( pire, std::fabs( b[ i ] ) / nu[ i ] );
            }
            if ( it == 0 ) {
                TF am = a[ 0 ], nm = nu[ 0 ];
                for ( SI i = 0; i < n; ++i ) { am = std::min( am, a[ i ] ); nm = std::min( nm, nu[ i ] ); }
                eps = TF( 0.5 ) * std::min( nm, am );
            }
            const TF nr = norme2( b );
            reste = pire;

            if ( pire <= tol ) {
                if ( trace )
                    std::printf( "    it %2d  |r|_2 %.3e  max|a-nu|/nu %.3e  CONVERGE\n",
                                 it, double( nr ), double( reste ) );
                fin = "CONVERGE";
                return true;
            }
            ++nb_iter;
            const double d0 = t_diag, c0 = t_cg, s0 = t_syst, m0 = t_maj;
            const int    g0 = nb_diag;

            double t0 = now();
            assemble( ar, S );
            t_syst += now() - t0;
            t0 = now();
            bool fait = false;
#ifdef PD2D_AMGCL
            if ( quel == 0 )
                fait = amg( S, b, d, cgtol, cgmax );
#endif
#ifdef PD2D_EIGEN
            if ( ! fait && quel <= 1 )
                fait = chol( S, b, d );
#endif
            if ( ! fait )
                nb_cg += cg( S, b, d, cgtol, cgmax );
            t_cg += now() - t0;

            // ---- L'AMORTISSEMENT
            TF t = 1;
            bool pris = false;
            w2.resize( n );
            for ( int essai = 0; essai < 60; ++essai ) {
                for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
                w2[ 0 ] = 0;                            // la jauge, imposee et non esperee
                aires_et_aretes<Cell>( pd, tr, X, Y, w2.data(), a2, ar2, th, sp, pin );
                TF m2 = a2[ 0 ], n2 = 0;      // le plancher `eps` est une aire ABSOLUE
                for ( SI i = 0; i < n; ++i ) {
                    m2 = std::min( m2, a2[ i ] );
                    const TF e = nu[ i ] - a2[ i ];
                    n2 += e * e;
                }
                const TF n2r = std::sqrt( n2 );
                // `n2r < nr` STRICT, et c'est le point. Sans lui, un pas qui tend vers zero passe
                // le test `n2r <= ( 1 - t / 2 ) nr` PAR EGALITE des que `t` est negligeable : la
                // boucle accepte alors un pas nul et Newton tourne sur place indefiniment. Mesure
                // avant correction : 54 diagrammes par iteration, a residu rigoureusement constant,
                // sur le nuage de lignes une fois le plancher numerique atteint.
                if ( m2 >= eps && n2r <= ( 1 - t / 2 ) * nr && n2r < nr ) { pris = true; break; }
                t /= 2;
                ++nb_recul;
                if ( t < TF( 1e-10 ) )
                    break;
            }
            if ( trace ) {
                std::printf( "    it %2d  |r|_2 %.3e  max|a-nu|/nu %.3e  %d vides"
                             "  pas %.2e  %d diag  [diag %.2f  maj %.2f  asm %.2f  sol %.2f]\n",
                             it, double( nr ), double( reste ), int( nvide ),
                             double( t ), nb_diag - g0, t_diag - d0, t_maj - m0, t_syst - s0,
                             t_cg - c0 );
                std::fflush( stdout );
            }
            if ( ! pris ) {
                // plus aucun pas ne fait decroitre le residu : on est au PLANCHER NUMERIQUE, pas
                // en echec. La difference se lit sur `reste`.
                fin = "STAGNATION";
                return false;
            }
            w.swap( w2 );
            a.swap( a2 );
            ar.swap( ar2 );
        }
        fin = "MAX ITERATIONS";
        return false;
    }
};

} // namespace pd2d
