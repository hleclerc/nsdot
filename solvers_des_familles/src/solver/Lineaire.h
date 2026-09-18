#pragma once

// =====================================================================================
// LES SOLVEURS LINEAIRES : `L d = b` sur le systeme reduit, la jauge `d_0 = 0` etant a leur
// charge. Tous ont la meme surface :
//
//     bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d );
//     const char *nom() const;
//     StatsLin st;        // le temps par poste, et ce que le solveur a fait
//
// Le solveur lineaire n'est pas l'objet de l'etude et il ne faut pas qu'il le devienne : on prend
// des bibliotheques en-tetes seuls.
//
//   AMG       AMGCL, multigrille algebrique + CG. Le bon outil pour ce systeme : la hessienne est
//             le laplacien d'un graphe presque planaire, le cout reste en O( n ) et la hierarchie
//             se construit en parallele ( OpenMP ). 4.9x sur le total a n=1e6 contre Cholesky.
//             Trois hierarchies, parce que le nuage decide : l'agregation lissee suppose des poids
//             d'aretes comparables, ce que le nuage de lignes viole ; Ruge-Stuben, qui choisit ses
//             noeuds grossiers arete par arete, y divise les iterations par trois.
//   CHOLESKY  Eigen `SimplicialLDLT`, renumerotation AMD, l'analyse symbolique refaite SEULEMENT
//             quand le motif change. Scalaire et sequentiel, mais il gagne encore sur le nuage de
//             lignes a n=1e5 ; c'est le temoin.
// =====================================================================================

#include "solver/Laplacien.h"
#include <algorithm>
#include <tuple>
#include <type_traits>

#if __has_include( <amgcl/make_solver.hpp> )
#  define SF_AMGCL 1
#  include <amgcl/adapter/crs_tuple.hpp>
#  include <amgcl/amg.hpp>
#  include <amgcl/backend/builtin.hpp>
#  include <amgcl/coarsening/ruge_stuben.hpp>
#  include <amgcl/coarsening/smoothed_aggregation.hpp>
#  include <amgcl/make_solver.hpp>
#  include <amgcl/relaxation/gauss_seidel.hpp>
#  include <amgcl/relaxation/spai0.hpp>
#  include <amgcl/solver/cg.hpp>
#endif

#if __has_include( <Eigen/SparseCholesky> )
#  define SF_EIGEN 1
#  include <Eigen/SparseCholesky>
#  include <Eigen/SparseCore>
#endif

namespace sf {

/// ce qu'un solveur lineaire rapporte, cumule sur toutes ses resolutions.
struct StatsLin {
    double t_forme = 0;        ///< la mise en forme de la matrice ( CRS, triplets )
    double t_hier  = 0;        ///< la hierarchie AMG, ou l'analyse symbolique
    double t_res   = 0;        ///< la resolution proprement dite ( ou factorisation + descente )
    int    nb_hier = 0;        ///< combien de hierarchies / analyses
    int    nb_iter = 0;        ///< iterations de CG, en tout
    TF     pire    = 0;        ///< le pire residu relatif rendu
    double total() const { return t_forme + t_hier + t_res; }
};

#ifdef SF_AMGCL
struct Amg {
    enum Variante : int { SA_SPAI0 = 0, SA_GS = 1, RS_GS = 2 };
    int      variante = SA_SPAI0;
    TF       tol      = 1e-10;     ///< residu RELATIF
    int      maxit    = 20000;
    StatsLin st;

    const char *nom() const {
        return variante == RS_GS ? "AMGCL Ruge-Stuben+GS"
             : variante == SA_GS ? "AMGCL agregation+GS" : "AMGCL agregation+spai0";
    }

    bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d ) {
        const SI n = L.n, m = n - 1;
        const double t0 = now();
        std::vector<int> ptr, col;
        std::vector<double> val;
        L.crs_reduit( ptr, col, val );
        std::vector<double> rb( m ), sol( m, 0.0 );
        for ( SI i = 1; i < n; ++i )
            rb[ i - 1 ] = double( b[ i ] );
        const double t1 = now();
        st.t_forme += t1 - t0;

        using Back = amgcl::backend::builtin<double>;
        int it = 0;
        double err = 0;
        auto lance = [ & ]( auto tag ) {
            using Solv = typename decltype( tag )::type;
            typename Solv::params prm;
            prm.solver.tol = double( tol );
            prm.solver.maxiter = maxit;
            Solv so( std::tie( m, ptr, col, val ), prm );
            const double ta = now();
            st.t_hier += ta - t1;
            ++st.nb_hier;
            std::tie( it, err ) = so( rb, sol );
            st.t_res += now() - ta;
        };
        using SaSpai = amgcl::make_solver<amgcl::amg<Back, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>, amgcl::solver::cg<Back>>;
        using SaGs   = amgcl::make_solver<amgcl::amg<Back, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::gauss_seidel>, amgcl::solver::cg<Back>>;
        using RsGs   = amgcl::make_solver<amgcl::amg<Back, amgcl::coarsening::ruge_stuben, amgcl::relaxation::gauss_seidel>, amgcl::solver::cg<Back>>;
        if      ( variante == SA_GS ) lance( std::type_identity<SaGs>{} );
        else if ( variante == RS_GS ) lance( std::type_identity<RsGs>{} );
        else                          lance( std::type_identity<SaSpai>{} );
        st.nb_iter += it;
        st.pire = std::max( st.pire, TF( err ) );

        d.assign( n, TF( 0 ) );
        for ( SI i = 1; i < n; ++i )
            d[ i ] = TF( sol[ i - 1 ] );
        return err < 1;                                  // `1` : le solveur n'a rien fait du tout
    }
};
#endif

#ifdef SF_EIGEN
struct Cholesky {
    using SpM = Eigen::SparseMatrix<double>;
    Eigen::SimplicialLDLT<SpM, Eigen::Lower, Eigen::AMDOrdering<int>> so;
    std::vector<SI> motif;         ///< le motif de la derniere analyse symbolique
    StatsLin st;

    const char *nom() const { return "Cholesky creux ( Eigen LDLT, AMD )"; }

    /// UNE DESCENTE DE PLUS sur la derniere factorisation : pour qui a plusieurs seconds membres.
    void resout_encore( const std::vector<TF> &b, std::vector<TF> &d ) {
        const SI n = SI( b.size() ), m = n - 1;
        const double t0 = now();
        Eigen::VectorXd rb( m );
        for ( SI i = 1; i < n; ++i ) rb[ i - 1 ] = double( b[ i ] );
        const Eigen::VectorXd sol = so.solve( rb );
        d.assign( n, TF( 0 ) );
        for ( SI i = 1; i < n; ++i ) d[ i ] = TF( sol[ i - 1 ] );
        st.t_res += now() - t0;
    }

    /// LE MOTIF NE BOUGE PRESQUE PAS : quelques aretes par iteration au debut, zero a la fin, alors
    /// que la renumerotation et l'analyse symbolique coutent le tiers de la factorisation. On les
    /// refait SEULEMENT quand le motif a change -- compare tel quel, une passe lineaire.
    bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d ) {
        const SI n = L.n, m = n - 1;
        const double t0 = now();
        std::vector<Eigen::Triplet<double>> tri;
        tri.reserve( size_t( L.row[ n ] ) / 2 + n );
        for ( SI i = 1; i < n; ++i ) {
            tri.emplace_back( i - 1, i - 1, double( L.dia[ i ] ) );
            for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) {
                const SI j = L.col[ k ];
                if ( j >= 1 && j < i )                   // le triangle INFERIEUR seul
                    tri.emplace_back( i - 1, j - 1, -double( L.c[ k ] ) );
            }
        }
        SpM A( m, m );
        A.setFromTriplets( tri.begin(), tri.end() );
        const double t1 = now();
        st.t_forme += t1 - t0;

        std::vector<SI> mot( A.outerIndexPtr(), A.outerIndexPtr() + m + 1 );
        mot.insert( mot.end(), A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros() );
        if ( mot != motif ) {
            so.analyzePattern( A );
            motif.swap( mot );
            ++st.nb_hier;
        }
        const double t2 = now();
        st.t_hier += t2 - t1;

        so.factorize( A );
        if ( so.info() != Eigen::Success ) { st.t_res += now() - t2; return false; }
        Eigen::VectorXd rb( m );
        for ( SI i = 1; i < n; ++i )
            rb[ i - 1 ] = double( b[ i ] );
        const Eigen::VectorXd sol = so.solve( rb );
        st.t_res += now() - t2;
        if ( so.info() != Eigen::Success )
            return false;

        d.assign( n, TF( 0 ) );
        for ( SI i = 1; i < n; ++i )
            d[ i ] = TF( sol[ i - 1 ] );
        return true;
    }
};
#endif

} // namespace sf
