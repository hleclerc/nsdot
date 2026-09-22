#pragma once

// =====================================================================================
// CE QUE L'HOTE VOIT : un `DiagrammeGpu<D,TK>` prend l'arbre bati par `accel/AaBsp.h`, le
// televerse une fois, et rend les mesures des cellules INDEXEES PAR L'IDENTIFIANT de l'appelant,
// comme `PowerDiagram::measures`. Pas un type CUDA ici : ce fichier est inclus par du C++
// compile par g++ ( `-march=native` ), l'implementation est dans `Mesures.cu`.
// =====================================================================================

#include "accel/AaBsp.h"
#include <string>
#include <vector>

namespace sf::gpu {

/// le mappage cellule / threads
enum class Variante { FIL, FILREG, FILREGC, FILMIX4, FILMIX6, FILMIX8, FILMIX12, FILMIX16, FILBRK6, FILBRK8, FILBRK10, FILBRK12, FILBRK16, FILBRK8NU, FILROT6, FILROT8, FILNRM8, FILUNI8, FILUNI8NP, FILSHM8, FILNRM8TRI, FILNRM8TRIL, FILPH8, FILPH8G, FILPH8B, FILPH8A, FILPH8C, VOIES, VOIES16, VOIES32, PAQ8x1, PAQ8x2, PAQ8x4, PAQ32x1, PAQ32x2, PAQ32x4, PAQ8x1S, PAQ32x1S, PAQ32x4S, NB };
inline const char *nom( Variante v ) {
    static const char *noms[] = { "fil", "filreg", "filregc", "filmix4", "filmix6", "filmix8", "filmix12", "filmix16", "filbrk6", "filbrk8", "filbrk10", "filbrk12", "filbrk16", "filbrk8nu", "filrot6", "filrot8", "filnrm8", "filuni8", "filuni8np", "filshm8", "filnrm8tri", "filnrm8tril", "filph8", "filph8g", "filph8b", "filph8a", "filph8c", "voies", "voies16", "voies32", "paquet8x1", "paquet8x2", "paquet8x4", "paquet32x1", "paquet32x2", "paquet32x4", "paquet8x1S", "paquet32x1S", "paquet32x4S" };
    return noms[ int( v ) ];
}
/// `paquet V x K` : `V` voies par cellule, `K` cellules par voie, un parcours par warp ( 2D )
inline bool paquet( Variante v ) { return v >= Variante::PAQ8x1 && v < Variante::NB; }

/// les chiffres d'un `mesures`
struct Chrono {
    double noyau  = 0;      ///< le noyau seul, en secondes, MINIMUM des repetitions ( evenements CUDA )
    double retour = 0;      ///< la descente des mesures, une fois
    int    deborde = 0;     ///< cellules dont les tampons n'ont pas suffi ( mesure fausse )
    long long stats[ 4 ] = {};   ///< par cellule, si le noyau les compte : coupes tentees, effectives, excursions, boites testees
};

template<int D, class TK>
struct DiagrammeGpu {
    explicit DiagrammeGpu( const AaBspT<D> &arbre );
    ~DiagrammeGpu();
    DiagrammeGpu( const DiagrammeGpu & ) = delete;
    DiagrammeGpu &operator=( const DiagrammeGpu & ) = delete;

    double televersement() const { return t_tele; }     ///< le temps de la montee, en secondes

    /// `res[ id ]` pour chaque germe ; `maxnv` : 64 | 128 en 2D, 64 | 128 en 3D. Un tour de chauffe,
    /// puis `reps` tours chronometres.
    Chrono mesures( Variante v, int maxnv, int reps, std::vector<double> &res ) const;

    struct Impl;
    Impl  *impl;
    double t_tele = 0;
};

/// le nom de la carte, pour l'en-tete
std::string carte();

} // namespace sf::gpu
