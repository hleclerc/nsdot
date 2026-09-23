#pragma once

// =====================================================================================
// LE LAPLACIEN DU GRAPHE DE LAGUERRE, assemble depuis les facettes.
//
//     c_ij = |facette ij| / ( 2 |p_i - p_j| ),   L_ii = sum_j c_ij,   L_ij = -c_ij
//
// La formule ne depend pas de la dimension : longueur sur distance en 2D, aire sur distance en
// 3D, et c'est la cellule qui livre la mesure de la facette. Son noyau est exactement les
// CONSTANTES -- ajouter la meme constante a tous les poids ne change aucune cellule.
//
// SANS UN SEUL TRI. Chaque facette est vue DEUX fois, une par cellule, et les deux mesures ne
// different qu'a l'arrondi ; on garde celle de la ligne, donc un comptage et une somme prefixe
// placent tout le monde -- et chaque ligne somme EXACTEMENT a zero. Apparier les deux mesures par
// un tri coutait plus que le diagramme lui-meme ( 4.2 s contre 3.9 a n=1e6 ).
//
// LA JAUGE `w_0 = 0` : le systeme reduit raye la ligne et la colonne 0, et devient defini positif.
// `crs_reduit` le livre ainsi, colonnes triees, pour un solveur qui veut du CRS.
// =====================================================================================

#include "util/common.h"
#include <vector>

namespace sf {

/// Une facette vue DEPUIS la cellule `i` : c'est elle qui en a mesure l'etendue.
struct Facette {
    SI i, j;
    TF c;                          ///< `c_ij > 0` ; le signe moins est dans la matrice
};

struct Laplacien {
    SI              n = 0;
    std::vector<SI> row, col;      ///< CSR des hors-diagonaux, `c[ k ] = c_ij`
    std::vector<TF> c;
    std::vector<TF> dia;           ///< `L_ii`, la somme de la ligne

    void assemble( SI nb, const std::vector<Facette> &fa ) {
        n = nb;
        row.assign( n + 1, 0 );
        for ( const Facette &e : fa )
            ++row[ e.i + 1 ];
        for ( SI i = 0; i < n; ++i )
            row[ i + 1 ] += row[ i ];

        col.resize( row[ n ] );
        c.resize( row[ n ] );
        dia.assign( n, TF( 0 ) );
        std::vector<SI> at( row.begin(), row.end() - 1 );
        for ( const Facette &e : fa ) {
            const SI p = at[ e.i ]++;
            col[ p ] = e.j;
            c[ p ] = e.c;
            dia[ e.i ] += e.c;
        }
        // une cellule sans voisin ne peut pas arriver tant qu'aucune n'est vide, mais une ligne
        // nulle rendrait le systeme singulier SANS LE DIRE : on la neutralise.
        for ( SI i = 0; i < n; ++i )
            if ( ! ( dia[ i ] > 0 ) )
                dia[ i ] = 1;
    }

    /// LE SYSTEME REDUIT ( `n - 1` inconnues, le germe 0 raye -- pas neutralise : une ligne
    /// identite donnerait a l'agregation un noeud isole a traiter ), diagonale comprise, colonnes
    /// TRIEES par insertion -- sept entrees par ligne.
    void crs_reduit( std::vector<int> &ptr, std::vector<int> &cl, std::vector<double> &val ) const {
        const SI m = n - 1;
        ptr.assign( m + 1, 0 );
        for ( SI i = 1; i < n; ++i ) {
            int k = 1;                                  // la diagonale, toujours presente
            for ( SI e = row[ i ]; e < row[ i + 1 ]; ++e )
                k += col[ e ] >= 1;
            ptr[ i ] = k;
        }
        for ( SI i = 0; i < m; ++i )
            ptr[ i + 1 ] += ptr[ i ];
        cl.resize( ptr[ m ] );
        val.resize( ptr[ m ] );
        for ( SI i = 1; i < n; ++i ) {
            int k = ptr[ i - 1 ];
            cl[ k ] = int( i - 1 );  val[ k ] = double( dia[ i ] );  ++k;
            for ( SI e = row[ i ]; e < row[ i + 1 ]; ++e )
                if ( col[ e ] >= 1 ) {
                    cl[ k ] = int( col[ e ] - 1 );  val[ k ] = -double( c[ e ] );  ++k;
                }
            for ( int u = ptr[ i - 1 ] + 1; u < k; ++u ) {
                const int cc = cl[ u ];
                const double v = val[ u ];
                int j = u;
                for ( ; j > ptr[ i - 1 ] && cl[ j - 1 ] > cc; --j ) {
                    cl[ j ] = cl[ j - 1 ]; val[ j ] = val[ j - 1 ];
                }
                cl[ j ] = cc; val[ j ] = v;
            }
        }
    }
};

} // namespace sf
