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
enum class Variante { FIL, VOIES, VOIES16, VOIES32 };
inline const char *nom( Variante v ) {
    switch ( v ) {
        case Variante::FIL:     return "fil";
        case Variante::VOIES:   return "voies";
        case Variante::VOIES16: return "voies16";
        default:                return "voies32";
    }
}

/// les chiffres d'un `mesures`
struct Chrono {
    double noyau  = 0;      ///< le noyau seul, en secondes, MINIMUM des repetitions ( evenements CUDA )
    double retour = 0;      ///< la descente des mesures, une fois
    int    deborde = 0;     ///< cellules dont les tampons n'ont pas suffi ( mesure fausse )
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
