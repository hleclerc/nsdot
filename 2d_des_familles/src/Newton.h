#pragma once

#include "AaBsp.h"
#include "PowerDiagram.h"
#include "parallel.h"
#include <algorithm>
#include <chrono>
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
    /// Une arete du graphe de Laguerre, `i < j`. Chaque paire est vue DEUX FOIS, une par cellule,
    /// et c'est ce qui sert a symetriser.
    struct Arete {
        SI i, j;
        TF c;
        bool operator<( const Arete &o ) const { return i != o.i ? i < o.i : j < o.j; }
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

    SI              n  = 0;
    TF              nu = 0;         ///< la mesure cible, la meme pour tous
    TF              eps = 0;        ///< le plancher d'aire de l'amortissement
    std::vector<TF> w;              ///< les poids courants, `w[ 0 ] == 0`

    bool        direct = true;      ///< factoriser plutot qu iterer, si Eigen est la
    const char *fin = "?";          ///< pourquoi la boucle s est arretee
    TF          reste = 0;          ///< le `max_i |a_i - nu| / nu` atteint
    double t_tri = 0, t_ana = 0, t_fac = 0, t_sol = 0;  ///< le detail de la factorisation
    int    nb_iter = 0, nb_diag = 0, nb_cg = 0, nb_recul = 0, nb_ana = 0;
    double t_diag = 0, t_maj = 0, t_syst = 0, t_cg = 0;

    static double now() {
        using namespace std::chrono;
        return duration<double>( steady_clock::now().time_since_epoch() ).count();
    }

    /// LES AIRES ET LES ARETES. Une passe de plus par cellule -- longueur d'arete, distance des
    /// deux germes -- faite une fois par iteration acceptee et non a chaque pas essaye.
    template<class Cell, class PD>
    void aires_et_aretes( PD &pd, AaBsp &tr, const TF *X, const TF *Y, const TF *W,
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
                par[ t ].push_back( i < j ? Arete{ i, j, cc } : Arete{ j, i, cc } );
            }
        } );
        ar.clear();
        for ( auto &v : par )
            ar.insert( ar.end(), v.begin(), v.end() );
        t_diag += now() - t0;
        ++nb_diag;
    }

    /// Le laplacien. Les doublons sont MOYENNES : les deux cellules d'une meme arete en mesurent
    /// la longueur chacune de son cote, et les deux valeurs ne different qu'a l'arrondi -- mais un
    /// gradient conjugue veut une matrice VRAIMENT symetrique.
    void assemble( std::vector<Arete> &ar, Systeme &S ) const {
        std::sort( ar.begin(), ar.end() );
        S.n = n;
        S.dia.assign( n, TF( 0 ) );
        std::vector<SI> deg( n, 0 );
        std::vector<Arete> uniq;
        uniq.reserve( ar.size() / 2 + 1 );
        for ( size_t k = 0; k < ar.size(); ) {
            size_t e = k;
            TF s = 0;
            while ( e < ar.size() && ar[ e ].i == ar[ k ].i && ar[ e ].j == ar[ k ].j )
                s += ar[ e++ ].c;
            const TF c = s / TF( e - k );
            uniq.push_back( Arete{ ar[ k ].i, ar[ k ].j, c } );
            ++deg[ ar[ k ].i ]; ++deg[ ar[ k ].j ];
            S.dia[ ar[ k ].i ] += c;
            S.dia[ ar[ k ].j ] += c;
            k = e;
        }

        S.row.assign( n + 1, 0 );
        for ( SI i = 0; i < n; ++i )
            S.row[ i + 1 ] = S.row[ i ] + deg[ i ];
        S.col.assign( S.row[ n ], 0 );
        S.c.assign( S.row[ n ], TF( 0 ) );
        std::vector<SI> at( S.row.begin(), S.row.end() - 1 );
        for ( const Arete &e : uniq ) {
            S.col[ at[ e.i ] ] = e.j;  S.c[ at[ e.i ]++ ] = e.c;
            S.col[ at[ e.j ] ] = e.i;  S.c[ at[ e.j ]++ ] = e.c;
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

    static TF norme2( const std::vector<TF> &v ) {
        TF s = 0;
        for ( TF x : v ) s += x * x;
        return std::sqrt( s );
    }

    /// LA BOUCLE. Rend `true` si le critere d'arret a ete atteint.
    ///
    /// Une iteration coute UN diagramme par pas essaye, et rien de plus : le pas accepte livre a la
    /// fois les aires (pour le residu) et les aretes (pour la hessienne suivante). La version
    /// precedente refaisait un diagramme au debut de chaque iteration juste pour les aretes, ce qui
    /// coutait 17 diagrammes la ou 9 suffisent.
    template<class Cell, class PD>
    bool resout( PD &pd, AaBsp &tr, const TF *X, const TF *Y, TF tol, int maxit, TF cgtol,
                 int cgmax, int th, Split sp, bool pin, bool trace ) {
        std::vector<TF> a, a2, b, d, w2;
        std::vector<Arete> ar, ar2;
        Systeme S;

        w.assign( n, TF( 0 ) );
        nu = TF( 1 ) / n;
        aires_et_aretes<Cell>( pd, tr, X, Y, w.data(), a, ar, th, sp, pin );

        for ( nb_iter = 0; nb_iter < maxit; ++nb_iter ) {
            TF amin = a[ 0 ], pire = 0;
            b.assign( n, TF( 0 ) );
            for ( SI i = 0; i < n; ++i ) {
                amin = std::min( amin, a[ i ] );
                b[ i ] = nu - a[ i ];                   // `-r`, le second membre de Newton
                pire = std::max( pire, std::fabs( b[ i ] ) );
            }
            if ( nb_iter == 0 )
                eps = TF( 0.5 ) * std::min( nu, amin );
            const TF nr = norme2( b );
            reste = pire / nu;

            if ( pire <= tol * nu ) {
                if ( trace )
                    std::printf( "    it %2d  |r|_2 %.3e  max|a-nu|/nu %.3e  CONVERGE\n",
                                 nb_iter, double( nr ), double( reste ) );
                fin = "CONVERGE";
                return true;
            }
            const double d0 = t_diag, c0 = t_cg, s0 = t_syst, m0 = t_maj;
            const int    g0 = nb_diag;

            double t0 = now();
            assemble( ar, S );
            t_syst += now() - t0;
            t0 = now();
            bool fait = false;
#ifdef PD2D_EIGEN
            if ( direct )
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
                TF m2 = a2[ 0 ], n2 = 0;
                for ( SI i = 0; i < n; ++i ) {
                    m2 = std::min( m2, a2[ i ] );
                    const TF e = nu - a2[ i ];
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
                std::printf( "    it %2d  |r|_2 %.3e  max|a-nu|/nu %.3e  min a/nu %.3e"
                             "  pas %.2e  %d diag  [diag %.2f  maj %.2f  asm %.2f  sol %.2f]\n",
                             nb_iter, double( nr ), double( reste ), double( amin / nu ),
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
