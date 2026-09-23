#pragma once

// =====================================================================================
// LES OPTIONS COMMUNES aux trois bancs : le nuage, les fils, l'arbre, le noyau. Un `main` les
// parse d'abord, puis les siennes.
// =====================================================================================

#include "bench/Nuages.h"
#include "util/parallel.h"
#include <string>

namespace sf {

struct Args {
    SI          n       = 1000000;  ///< germes du cas uniforme
    int         reps    = 3;        ///< repetitions chronometrees ; on garde le MINIMUM
    Parallel    par;                ///< `threads = 0` -> autant que de coeurs
    SI          leaf    = 10;       ///< germes par feuille
    unsigned    graine  = 0;
    double      wscale  = 0;        ///< poids aleatoires de l'uniforme, en fraction de h^2
    std::string load;               ///< UN nuage, au lieu de la suite
    int         dims    = 0;        ///< 0 = 2D puis 3D, 2 ou 3 = cette dimension seule
    std::string cases   = "../2d_des_familles/cases";
    std::string kernel  = "double"; ///< le flottant du noyau : double | float
    int         maxnv   = 0;        ///< 0 = le defaut de la dimension ( 64 en 2D, 128 en 3D )

    int nv( int D ) const { return maxnv ? maxnv : ( D == 2 ? 64 : 128 ); }

    /// rend `true` si l'argument a ete consomme ; `i` avance sur la valeur.
    bool parse( const std::string &s, int &i, int argc, char **argv );
    static void usage();
    /// a appeler apres la boucle : normalise ce qui doit l'etre.
    void finalise();

    /// la suite, ou le seul nuage demande par `--load`.
    template<int D> std::vector<Nuage<D>> nuages() const;
};

} // namespace sf
