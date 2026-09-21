// L'UNITE DE DOMAINE des solveurs lineaires ( voir `Lineaire.h` ) : compilee une fois par
// compilateur, liee par les noyaux qui nomment `sdot/otplan/Lineaire.cpp` dans leurs `sources`.
// Eigen est cherche sous `<eigen3/...>` ( son emplacement Debian, dans un repertoire d'inclusion
// par defaut ), AMGCL sous `<amgcl/...>` ; ce qui manque n'est simplement pas propose.

#include "Lineaire.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <tuple>

#if __has_include( <amgcl/make_solver.hpp> )
#  define SDOT_AMGCL 1
#  ifndef AMGCL_NO_BOOST
#    define AMGCL_NO_BOOST
#  endif
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

#if __has_include( <eigen3/Eigen/SparseCholesky> )
#  define SDOT_EIGEN 1
#  include <eigen3/Eigen/SparseCholesky>
#  include <eigen3/Eigen/SparseCore>
#elif __has_include( <Eigen/SparseCholesky> )
#  define SDOT_EIGEN 1
#  include <Eigen/SparseCholesky>
#  include <Eigen/SparseCore>
#endif

namespace sdot {
namespace otplan {

static double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

// ---- le gradient conjugue, toujours la ------------------------------------------------------------

/// CG preconditionne par Jacobi sur le systeme complet, la jauge tenue en projetant `d_0 = 0` :
/// on resout sur les inconnues `1 .. n-1` en lisant la matrice complete et en ignorant la colonne 0.
struct Cg : SolveurLineaire {
    double tol = 1e-12;
    int    maxit = 20000;

    const char *nom() const override { return "gradient conjugue ( Jacobi )"; }

    bool resout( const Laplacien &L, const std::vector<double> &b, std::vector<double> &d ) override {
        const SI n = L.n;
        const double t0 = now();
        std::vector<double> r( n ), z( n ), p( n ), q( n );
        d.assign( n, 0.0 );
        auto produit = [&]( const std::vector<double> &x, std::vector<double> &y ) {
            L.produit( x.data(), y.data() );
            y[ 0 ] = 0;
        };
        double nb = 0;
        for ( SI i = 1; i < n; ++i ) { r[ i ] = b[ i ]; nb += b[ i ] * b[ i ]; }
        r[ 0 ] = 0;
        nb = std::sqrt( nb );
        if ( ! ( nb > 0 ) ) { st.t_res += now() - t0; return true; }
        double rz = 0;
        for ( SI i = 1; i < n; ++i ) { z[ i ] = r[ i ] / L.dia[ i ]; p[ i ] = z[ i ]; rz += r[ i ] * z[ i ]; }
        double err = 1;
        int it = 0;
        for ( ; it < maxit; ++it ) {
            produit( p, q );
            double pq = 0;
            for ( SI i = 1; i < n; ++i ) pq += p[ i ] * q[ i ];
            if ( ! ( pq > 0 ) ) break;
            const double alpha = rz / pq;
            double nr = 0;
            for ( SI i = 1; i < n; ++i ) { d[ i ] += alpha * p[ i ]; r[ i ] -= alpha * q[ i ]; nr += r[ i ] * r[ i ]; }
            err = std::sqrt( nr ) / nb;
            if ( err <= tol ) break;
            double rz2 = 0;
            for ( SI i = 1; i < n; ++i ) { z[ i ] = r[ i ] / L.dia[ i ]; rz2 += r[ i ] * z[ i ]; }
            const double beta = rz2 / rz;
            rz = rz2;
            for ( SI i = 1; i < n; ++i ) p[ i ] = z[ i ] + beta * p[ i ];
        }
        st.nb_iter += it;
        st.pire = std::max( st.pire, err );
        st.t_res += now() - t0;
        return err < 1;
    }
};

// ---- AMGCL ----------------------------------------------------------------------------------------

#ifdef SDOT_AMGCL
struct Amg : SolveurLineaire {
    enum Variante : int { SA_SPAI0 = 0, SA_GS = 1, RS_GS = 2 };
    int    variante = RS_GS;
    double tol      = 1e-10;       ///< residu RELATIF
    int    maxit    = 20000;

    const char *nom() const override {
        return variante == RS_GS ? "AMGCL Ruge-Stuben+GS"
             : variante == SA_GS ? "AMGCL agregation+GS" : "AMGCL agregation+spai0";
    }

    bool resout( const Laplacien &L, const std::vector<double> &b, std::vector<double> &d ) override {
        const SI n = L.n, m = n - 1;
        const double t0 = now();
        std::vector<int> ptr, col;
        std::vector<double> val;
        L.crs_reduit( ptr, col, val );
        std::vector<double> rb( m ), sol( m, 0.0 );
        for ( SI i = 1; i < n; ++i )
            rb[ i - 1 ] = b[ i ];
        const double t1 = now();
        st.t_forme += t1 - t0;

        using Back = amgcl::backend::builtin<double>;
        int it = 0;
        double err = 0;
        auto lance = [&]( auto tag ) {
            using Solv = typename decltype( tag )::type;
            typename Solv::params prm;
            prm.solver.tol = tol;
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
        st.pire = std::max( st.pire, err );

        d.assign( n, 0.0 );
        for ( SI i = 1; i < n; ++i )
            d[ i ] = sol[ i - 1 ];
        return err < 1;                                  // `1` : le solveur n'a rien fait du tout
    }
};
#endif

// ---- Eigen ----------------------------------------------------------------------------------------

#ifdef SDOT_EIGEN
struct Cholesky : SolveurLineaire {
    using SpM = Eigen::SparseMatrix<double>;
    Eigen::SimplicialLDLT<SpM, Eigen::Lower, Eigen::AMDOrdering<int>> so;
    std::vector<SI> motif;         ///< le motif de la derniere analyse symbolique

    const char *nom() const override { return "Cholesky creux ( Eigen LDLT, AMD )"; }

    /// LE MOTIF NE BOUGE PRESQUE PAS : quelques aretes par iteration au debut, zero a la fin, alors
    /// que la renumerotation et l'analyse symbolique coutent le tiers de la factorisation. On les
    /// refait SEULEMENT quand le motif a change -- compare tel quel, une passe lineaire.
    bool resout( const Laplacien &L, const std::vector<double> &b, std::vector<double> &d ) override {
        const SI n = L.n, m = n - 1;
        const double t0 = now();
        std::vector<Eigen::Triplet<double>> tri;
        tri.reserve( size_t( L.row[ n ] ) / 2 + n );
        for ( SI i = 1; i < n; ++i ) {
            tri.emplace_back( int( i - 1 ), int( i - 1 ), L.dia[ i ] );
            for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) {
                const SI j = L.col[ k ];
                if ( j >= 1 && j < i )                   // le triangle INFERIEUR seul
                    tri.emplace_back( int( i - 1 ), int( j - 1 ), -L.c[ k ] );
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
            rb[ i - 1 ] = b[ i ];
        const Eigen::VectorXd sol = so.solve( rb );
        st.t_res += now() - t2;
        if ( so.info() != Eigen::Success )
            return false;

        d.assign( n, 0.0 );
        for ( SI i = 1; i < n; ++i )
            d[ i ] = sol[ i - 1 ];
        return true;
    }
};
#endif

// ---- le choix -------------------------------------------------------------------------------------

int methodes_lineaires_disponibles() {
    int res = 1 << int( Lin::CG );
#ifdef SDOT_EIGEN
    res |= 1 << int( Lin::CHOLESKY );
#endif
#ifdef SDOT_AMGCL
    res |= 1 << int( Lin::AMG );
#endif
    return res;
}

std::unique_ptr<SolveurLineaire> solveur_lineaire( Lin methode, SI n, int dim ) {
    const int dispo = methodes_lineaires_disponibles();
    if ( methode == Lin::AUTO )
        methode = dim <= 2 && n <= 300000 ? Lin::CHOLESKY : Lin::AMG;
    if ( methode == Lin::CHOLESKY && ! ( dispo & ( 1 << int( Lin::CHOLESKY ) ) ) ) methode = Lin::AMG;
    if ( methode == Lin::AMG      && ! ( dispo & ( 1 << int( Lin::AMG      ) ) ) ) methode = Lin::CHOLESKY;
    if ( ! ( dispo & ( 1 << int( methode ) ) ) )                                   methode = Lin::CG;
#ifdef SDOT_EIGEN
    if ( methode == Lin::CHOLESKY ) return std::make_unique<Cholesky>();
#endif
#ifdef SDOT_AMGCL
    if ( methode == Lin::AMG ) return std::make_unique<Amg>();
#endif
    return std::make_unique<Cg>();
}

} // namespace otplan
} // namespace sdot
