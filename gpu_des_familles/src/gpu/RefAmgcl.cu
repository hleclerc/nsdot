// =====================================================================================
// LE TEMOIN DE REFERENCE, SUR LA MEME PLATEFORME QUE NOUS : AMGCL avec son backend CUDA.
//
// Comparer notre multigrille au CPU ne prouvait rien -- il fallait le comparer a AMGCL SUR LA
// CARTE, et avec ses meilleurs reglages. Le backend CUDA d'AMGCL n'accepte pas Gauss-Seidel
// ( sequentiel par nature ) ; les lisseurs disponibles sont `spai0`, `damped_jacobi` et
// `chebyshev`, les grossissements `smoothed_aggregation` ( sa force : la prolongation lissee ),
// `aggregation` ( non lissee, comme la notre ) et `ruge_stuben`.
//
// Le systeme donne est LE SYSTEME REDUIT ( `Laplacien::crs_reduit` ), le meme que celui du CPU :
// c'est la matrice qu'AMGCL veut, et elle est definie positive.
// =====================================================================================

// CUDA 13 a deplace ces symboles : `amgcl/backend/cuda.hpp` les utilise sans les inclure
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <amgcl/backend/cuda.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>

#include <chrono>
#include <vector>
#include <tuple>
#include <vector>

namespace sf::gpu {

namespace {

double maintenant() {
    return std::chrono::duration<double>( std::chrono::steady_clock::now().time_since_epoch() ).count();
}

template<template<class> class Coarsen, template<class> class Relax>
bool tourne( int n, const std::vector<int> &ptr, const std::vector<int> &col, const std::vector<double> &val,
             const std::vector<double> &b, std::vector<double> &x,
             double tol, int maxit, int *iters, double *err, double *t_hier, double *t_res ) {
    typedef amgcl::backend::cuda<double> Backend;
    typedef amgcl::make_solver<amgcl::amg<Backend, Coarsen, Relax>, amgcl::solver::cg<Backend>> Solveur;

    typename Backend::params bprm;
    cusparseCreate( &bprm.cusparse_handle );
    typename Solveur::params prm;
    prm.solver.tol = tol;
    prm.solver.maxiter = maxit;

    cudaDeviceSynchronize();
    double t0 = maintenant();
    Solveur solveur( std::tie( n, ptr, col, val ), prm, bprm );
    cudaDeviceSynchronize();
    *t_hier = maintenant() - t0;

    thrust::device_vector<double> f( b ), d( n, 0.0 );
    cudaDeviceSynchronize();
    t0 = maintenant();
    int it = 0;
    double e = 0;
    std::tie( it, e ) = solveur( f, d );
    cudaDeviceSynchronize();
    *t_res = maintenant() - t0;
    *iters = it;
    *err = e;
    thrust::copy( d.begin(), d.end(), x.begin() );
    cusparseDestroy( bprm.cusparse_handle );
    return e <= tol * 10;
}

} // namespace

/// `variante` : 0 agregation lissee + spai0, 1 + Jacobi amorti, 2 + Chebyshev,
///              3 agregation NON lissee + spai0, 4 Ruge-Stuben + spai0
const char *amgcl_cuda_nom( int variante ) {
    switch ( variante ) {
        case 1:  return "AMGCL/CUDA sa+jacobi";
        case 2:  return "AMGCL/CUDA sa+chebyshev";
        case 3:  return "AMGCL/CUDA agreg+spai0";
        case 4:  return "AMGCL/CUDA ruge-stuben";
        default: return "AMGCL/CUDA sa+spai0";
    }
}

bool amgcl_cuda( int n, const std::vector<int> &ptr, const std::vector<int> &col, const std::vector<double> &val,
                 const std::vector<double> &b, std::vector<double> &x,
                 double tol, int maxit, int variante,
                 int *iters, double *err, double *t_hier, double *t_res ) {
    using namespace amgcl;
    switch ( variante ) {
        case 1: return tourne<coarsening::smoothed_aggregation, relaxation::damped_jacobi>( n, ptr, col, val, b, x, tol, maxit, iters, err, t_hier, t_res );
        case 2: return tourne<coarsening::smoothed_aggregation, relaxation::chebyshev>( n, ptr, col, val, b, x, tol, maxit, iters, err, t_hier, t_res );
        case 3: return tourne<coarsening::aggregation, relaxation::spai0>( n, ptr, col, val, b, x, tol, maxit, iters, err, t_hier, t_res );
        case 4: return tourne<coarsening::ruge_stuben, relaxation::spai0>( n, ptr, col, val, b, x, tol, maxit, iters, err, t_hier, t_res );
        default: return tourne<coarsening::smoothed_aggregation, relaxation::spai0>( n, ptr, col, val, b, x, tol, maxit, iters, err, t_hier, t_res );
    }
}

} // namespace sf::gpu
