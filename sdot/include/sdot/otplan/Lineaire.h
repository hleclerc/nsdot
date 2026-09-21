#pragma once

// =====================================================================================
// LES SOLVEURS LINEAIRES DU TRANSPORT : `L d = b` sur le systeme reduit, la jauge `d_0 = 0` etant a
// leur charge. Le solveur lineaire n'est pas l'objet de l'etude et il ne faut pas qu'il le
// devienne : ce sont des bibliotheques en-tetes seuls, dans UNE unite de domaine compilee une fois
// ( `Lineaire.cpp`, nommee par `FfiCode( sources = ... )` ) et liee par le noyau qui les appelle --
// Eigen et AMGCL n'ont rien a faire dans chaque unite generee.
//
//   CHOLESKY  Eigen `SimplicialLDLT`, renumerotation AMD, l'analyse symbolique refaite SEULEMENT
//             quand le motif change. Scalaire et sequentiel, mais il gagne sur le nuage de lignes
//             a n=1e5, et en 2D jusqu'a quelques centaines de milliers de germes.
//   AMG       AMGCL, multigrille algebrique + CG. La hessienne est le laplacien d'un graphe
//             presque planaire : le cout reste en O( n ). 4.9x sur le total a n=1e6 contre
//             Cholesky. Ruge-Stuben+GS, qui choisit ses noeuds grossiers arete par arete, tient sur
//             les nuages ou l'agregation lissee echoue ( des poids d'aretes incomparables ).
//   CG        le gradient conjugue preconditionne par Jacobi, ecrit ici : ce qui reste quand ni
//             Eigen ni AMGCL ne sont la ( `__has_include` ). Cinq fois plus lent que Cholesky, mais
//             toujours disponible.
//
// Mesure dans `solvers_des_familles` ( README § 3, § 10 ) : Cholesky 11.9 s contre Ruge-Stuben
// 15.2 s sur les lignes 2D a 1e5 ; sur l'uniforme c'est l'inverse, et a 1e6 Cholesky ne monte pas
// en charge. AUTO choisit sur la dimension et la taille.
// =====================================================================================

#include "Laplacien.h"
#include <memory>

namespace sdot {
namespace otplan {

enum class Lin : int { AUTO = 0, CHOLESKY = 1, AMG = 2, CG = 3 };

/// ce qu'un solveur lineaire rapporte, cumule sur toutes ses resolutions.
struct StatsLin {
    double t_forme = 0;        ///< la mise en forme de la matrice ( CRS, triplets )
    double t_hier  = 0;        ///< la hierarchie AMG, ou l'analyse symbolique
    double t_res   = 0;        ///< la resolution proprement dite ( ou factorisation + descente )
    int    nb_hier = 0;        ///< combien de hierarchies / analyses
    int    nb_iter = 0;        ///< iterations de CG, en tout
    double pire    = 0;        ///< le pire residu relatif rendu
    double total() const { return t_forme + t_hier + t_res; }
};

struct SolveurLineaire {
    virtual ~SolveurLineaire() {}
    /// `d[ 0 ] == 0` en sortie. Rend `false` si le solveur n'a pas pu.
    virtual bool resout( const Laplacien &L, const std::vector<double> &b, std::vector<double> &d ) = 0;
    virtual const char *nom() const = 0;
    StatsLin st;
};

/// LE solveur pour `methode` -- AUTO : Cholesky en 2D jusqu'a 3e5 germes s'il est la, AMG s'il est
/// la, CG sinon. Une methode demandee et absente retombe sur la suivante disponible.
std::unique_ptr<SolveurLineaire> solveur_lineaire( Lin methode, SI n, int dim );

/// les methodes compilees ( un masque : bit `int( Lin::X )` )
int methodes_lineaires_disponibles();

} // namespace otplan
} // namespace sdot
