#pragma once

#include <cstdint>

namespace pd {

using TF = double;   ///< le type flottant de la GEOMETRIE. `float` est une variante a essayer.
using SI = int;      ///< un indice. `int` et non `int64` : 2e9 germes suffisent, et ca compte
                     ///< dans un noeud d'arbre, ou chaque octet est une fraction de ligne de cache.

/// UN POINT, en `D` dimensions, PASSE PAR VALEUR.
///
/// Pourquoi par valeur et pas par pointeur : c'est ce qui remplace les arguments scalaires
/// `( px, py )` que le banc passait quand il ne connaissait que la 2D, et il faut que le
/// remplacement soit GRATUIT. Un agregat de deux ou trois `double` sans constructeur ni destructeur
/// tient dans les registres SSE de l'ABI System V ; un `const TF *` forcerait le compilateur a
/// relire la memoire a chaque usage -- d'autant plus que le banc compile avec
/// `-fno-strict-aliasing`, donc sans pouvoir prouver que rien n'ecrit dedans entre deux lectures.
/// MESURE de ce choix : voir README, section « LA DIMENSION EN PARAMETRE ».
template<int D>
struct Vec {
    TF v[ D ];

    TF operator[]( int d ) const { return v[ d ]; }
    TF &operator[]( int d ) { return v[ d ]; }

    /// `|a - b|^2`, deroule a la compilation.
    friend TF dist2( Vec a, Vec b ) {
        TF s = 0;
        for ( int d = 0; d < D; ++d ) { const TF e = a[ d ] - b[ d ]; s += e * e; }
        return s;
    }
    friend TF dot( Vec a, Vec b ) {
        TF s = 0;
        for ( int d = 0; d < D; ++d ) s += a[ d ] * b[ d ];
        return s;
    }
};

/// le meme, depuis un tableau (une boite de noeud, par exemple). Le tableau reste ou il est ; ce
/// qui voyage est la copie en registres.
/// ECRITE EN INITIALISATION D'AGREGAT, et pas par une boucle qui remplit un temporaire. La boucle
/// alterne lecture (`a[ d ]`) et ECRITURE (`r[ d ]`), et le banc compile avec `-fno-strict-aliasing`
/// -- le compilateur doit alors supposer qu'un store de `TF` peut ecraser n'importe quoi, y compris
/// le `TF *` interne du `std::vector` d'ou vient `a`. Il recharge donc le pointeur entre deux
/// coordonnees. En initialisation d'agregat, toutes les lectures precedent la construction de
/// l'objet et le probleme n'existe pas. MESURE : voir README, « LA DIMENSION EN PARAMETRE ».
template<int D>
inline Vec<D> vec_of( const TF *a ) {
    if constexpr ( D == 2 )
        return { a[ 0 ], a[ 1 ] };
    else if constexpr ( D == 3 )
        return { a[ 0 ], a[ 1 ], a[ 2 ] };
    else {
        Vec<D> r;
        for ( int d = 0; d < D; ++d ) r[ d ] = a[ d ];
        return r;
    }
}

using Pt2 = Vec<2>;
using Pt3 = Vec<3>;

/// Ce qu'une coupe a fait de la cellule. Trois etats et non deux, parce que « le plan ne touche pas
/// la cellule » est le cas MAJORITAIRE : un accelerateur propose une feuille entiere de germes,
/// dont six seulement sont vraiment voisins en 2D. Le distinguer permet de ne RIEN ecrire.
enum class CutResult {
    unchanged,   ///< le demi-espace contient deja toute la cellule
    done,        ///< la cellule a ete coupee
    empty,       ///< il ne reste rien
    overflow,    ///< plus de `max_nb_vertices` sommets : la cellule est laissee TELLE QUELLE
};

} // namespace pd
