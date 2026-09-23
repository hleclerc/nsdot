#pragma once
#include <string>
#include <vector>
namespace sf::gpu {
/// AMGCL sur SA carte : `variante` 0 agregation lissee + spai0, 1 + Jacobi amorti, 2 + Chebyshev,
/// 3 agregation non lissee + spai0, 4 Ruge-Stuben + spai0. Le systeme est le REDUIT.
const char *amgcl_cuda_nom( int variante );
bool amgcl_cuda( int n, const std::vector<int> &ptr, const std::vector<int> &col, const std::vector<double> &val,
                 const std::vector<double> &b, std::vector<double> &x,
                 double tol, int maxit, int variante,
                 int *iters, double *err, double *t_hier, double *t_res );
}
