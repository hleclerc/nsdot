// LA RECETTE D'UNE CELLULE, et trois facons de la garder.
//
// Ce qu'on veut stocker n'est pas la cellule mais de quoi la REFAIRE : les plans qui l'ont
// reellement bordee. Les rejouer coute ~5 coupes en 2D et ~12 en 3D, contre ~28 et ~89 candidats
// essayes -- et sans une seule recherche.

#pragma once

#include "util/common.h"
#include "bench/Bench.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace pd::supercell {

// ------------------------------------------------------------------ LE STOCKAGE DE LA PASSE 1

/// TROIS FACONS DE GARDER CE QUE LA PASSE 1 A TROUVE, et le choix se fait a la COMPILATION.
///
/// Ce qu'on veut garder n'est pas la cellule mais la RECETTE pour la refaire : les plans qui l'ont
/// reellement bordee. Les rejouer, c'est couper ~6 fois au lieu d'en essayer ~13, sans une seule
/// recherche. Et entre deux iterations de Newton, ou les poids bougent peu, la recette reste
/// presque toujours valable -- c'est la que le stockage paie vraiment.
///
///   `Bits`    un bit par candidat, dans l'ordre CANONIQUE de la liste (membres de l'agregat puis
///             medians exterieurs). 8 octets par dirac, quelle que soit la cellule. Quand la liste
///             depasse 64 candidats, on ne peut pas : on COMPTE le dirac perdu.
///   `Packe`   les numeros des coupes AUX SOMMETS FINAUX, un octet chacun -- en 2D dans l'ordre
///             cyclique (la coupe `i` porte l'arete sortante du sommet `i`, c'est l'invariant de
///             `CellSoA`), en 3D les TROIS coupes de chaque sommet. Ca garde la combinatoire et
///             pas seulement l'ensemble. Les faces du domaine prennent les codes `nc + k`, et
///             `0xFF` dit « pas pu stocker » -- ce que la borne sur l'index ne garantit pas, on
///             l'AVOUE au lieu de le supposer.
///   `Complet` la cellule entiere. La reference : rien a rejouer, mais ~1.4 Ko par dirac en 2D.
enum class Stockage { Bits, Packe, Complet };

template<Stockage S, class Cell, int D>
struct Memoire {
    static constexpr int cap = ( D == 2 ? 16 : 96 );    ///< codes gardes par dirac en `Packe`
    /// LA LARGEUR DU MASQUE. Un mot suffit en 2D (`|A| - 1 + ~19` candidats a l'anneau 2) ; en 3D
    /// l'anneau 2 en presente ~84 et il en faut deux. Ce n'est pas un reglage libre : au-dela, on
    /// ne peut pas stocker et on le COMPTE.
    static constexpr int mots = ( D == 2 ? 1 : 2 );
    static constexpr int bits_max = 64 * mots;
    static constexpr uint8_t vide = 0xFF;

    std::vector<uint64_t> bit;
    std::vector<uint8_t>  cod, ncod;
    std::vector<Cell>     cel;
    SI perdus = 0;

    void reserve( SI n ) {
        if constexpr ( S == Stockage::Bits )    bit.assign( size_t( n ) * mots, 0 );
        if constexpr ( S == Stockage::Packe ) { cod.assign( size_t( n ) * cap, vide ); ncod.assign( n, 0 ); }
        if constexpr ( S == Stockage::Complet ) cel.resize( n );
    }

    void note( SI i, const Cell &c, SI nc ) {
        if constexpr ( S == Stockage::Bits ) {
            if ( nc > bits_max ) { ++perdus; return; }
            // `for_each_facet` rend EXACTEMENT les coupes qui portent une facette finale -- donc
            // les non redondantes, en 2D comme en 3D, et sans les faces du domaine (toujours
            // posees par l'initialisation). C'est la recette, et rien de plus qu'elle.
            uint64_t *m = &bit[ size_t( i ) * mots ];
            for ( int w = 0; w < mots; ++w ) m[ w ] = 0;
            c.for_each_facet( [ & ]( SI id, TF ) {
                if ( id >= 0 ) m[ id >> 6 ] |= uint64_t( 1 ) << ( id & 63 ); } );
        }
        if constexpr ( S == Stockage::Packe ) {
            const SI par = ( D == 2 ? 1 : 3 );
            if ( nc + 2 * D > 254 || c.nb * par > cap ) { ++perdus; ncod[ i ] = vide; return; }
            uint8_t *o = &cod[ size_t( i ) * cap ];
            SI k = 0;
            for ( SI v = 0; v < c.nb; ++v ) {
                if constexpr ( D == 2 ) {
                    // LA COUPE DE L'ARETE SORTANTE. Les quatre cotes du domaine portent le MEME
                    // `cid = -1` dans `CellSoA` : les distinguer ici est ce qui rend la cellule
                    // RECONSTRUCTIBLE sans la recouper -- sinon on saurait qu'un cote est le
                    // domaine sans savoir lequel, et le systeme 2x2 n'aurait pas d'equation.
                    const SI id = c.cid[ v ];
                    o[ k++ ] = uint8_t( id >= 0 ? id : nc + cote_domaine( c.cdx[ v ], c.cdy[ v ] ) );
                } else {
                    for ( int e = 0; e < 3; ++e ) {
                        const SI id = c.vc[ v ][ e ];
                        o[ k++ ] = uint8_t( id >= 0 ? id : nc + ( -id - 1 ) );
                    }
                }
            }
            ncod[ i ] = uint8_t( k );
        }
        if constexpr ( S == Stockage::Complet ) cel[ i ] = c;
    }

    /// les candidats a rejouer, lus dans la recette. `Complet` n'a rien a rejouer.
    void relis( SI i, SI nc, std::vector<SI> &out ) const {
        out.clear();
        if constexpr ( S == Stockage::Bits ) {
            if ( nc > bits_max ) return;
            const uint64_t *m = &bit[ size_t( i ) * mots ];
            for ( SI j = 0; j < nc; ++j )
                if ( m[ j >> 6 ] >> ( j & 63 ) & 1 ) out.push_back( j );
        }
        if constexpr ( S == Stockage::Packe ) {
            if ( ncod[ i ] == vide ) return;
            for ( SI k = 0; k < SI( ncod[ i ] ); ++k ) {
                const SI v = cod[ size_t( i ) * cap + k ];
                if ( v < nc && std::find( out.begin(), out.end(), v ) == out.end() ) out.push_back( v );
            }
        }
    }

    /// lequel des quatre cotes du carre unite, d'apres sa normale. L'ordre est celui de
    /// `init_as_unit_square` : `y >= 0`, `x <= 1`, `y <= 1`, `x >= 0`.
    static SI cote_domaine( TF dx, TF dy ) {
        if ( dy < 0 ) return 0;
        if ( dx > 0 ) return 1;
        if ( dy > 0 ) return 2;
        return 3;
    }

    /// LES CODES BRUTS, pour qui veut RECONSTRUIRE au lieu de rejouer. En 2D ils portent toute la
    /// combinatoire : le sommet `v` est l'intersection des coupes `v-1` et `v`, et l'ordre
    /// cyclique EST la connectivite. Il ne reste qu'un systeme `2 x 2` par sommet.
    SI nb_codes( SI i ) const {
        if constexpr ( S == Stockage::Packe ) return ncod[ i ] == vide ? 0 : SI( ncod[ i ] );
        else { (void) i; return 0; }
    }
    SI code( SI i, SI k ) const {
        if constexpr ( S == Stockage::Packe ) return SI( cod[ size_t( i ) * cap + k ] );
        else { (void) i; (void) k; return 0; }
    }

    bool rejouable() const { return S != Stockage::Complet; }

    /// LA CELLULE TELLE QUELLE, quand on l'a gardee entiere. Rend `false` dans les autres modes,
    /// ou il faut la rejouer. C'est le seul endroit ou `Complet` gagne quelque chose : sans lui il
    /// paie le stockage ( 1.4 Ko par dirac ) sans jamais le relire, et l'etape 4 retombait sur la
    /// liste COMPLETE des candidats -- 27.7 coupes au lieu de 5.4, le pire des trois modes.
    bool donne( SI i, Cell &c ) const {
        if constexpr ( S == Stockage::Complet ) { c = cel[ i ]; return true; }
        else { (void) i; (void) c; return false; }
    }

    double octets( SI n ) const {
        if constexpr ( S == Stockage::Bits )    return 8.0 * mots;
        if constexpr ( S == Stockage::Packe )   return cap + 1.0;
        if constexpr ( S == Stockage::Complet ) return double( sizeof( Cell ) );
        return 0;
    }

};

} // namespace pd::supercell
