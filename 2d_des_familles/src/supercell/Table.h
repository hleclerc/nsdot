// L'ETAPE 3 : LA TABLE DES SUR-CELLULES QUI SE TOUCHENT.
//
// Ce qu'elle produit : pour chaque agregat `A`, la liste des `B` dont l'enceinte rencontre la
// sienne. C'est la liste que l'etape 4 opposera a chaque dirac de `A` -- et le critere est exact
// pour la raison qui porte tout le schema : si `enc_A` ne rencontre pas `enc_B`, alors pour tout
// `i` de `A` et tout `j` de `B`, `Lag_i inter Lag_j` est vide ( les vraies cellules sont dans les
// enceintes ), donc `j` n'est pas voisin de `i` et son demi-espace est redondant.
//
// = TROIS ALGORITHMES POSSIBLES, ET POURQUOI CELUI-CI
//
//   * LE TRI 1D ( sweep-and-prune ). A ecarter en 3D : la liste active le long d'un axe n'est pas
//     un voisinage mais une TRANCHE. A 125 000 agregats dans le cube unite, l'intervalle en `x`
//     d'un agregat en croise ~`na^(2/3)`, soit ~2 500 -- des centaines de millions de tests pour
//     quelques millions de paires.
//
//   * LE FRONT par la connectivite grossiere. Le graphe est deja la, gratuit depuis l'etape 1,
//     mais son critere d'ARRET est le probleme. Aucun anneau fixe n'est complet : mesure sur
//     nuage uniforme, l'anneau 1 rate 9 % des paires utiles en 2D et 22 % en 3D, l'anneau 2 en
//     rate encore 0.4 % et 0.9 %. Et une regle geometrique saine (« etendre tant que
//     `|c_A - c_C| <= R_A + R_max` ») demande un MAJORANT GLOBAL du rayon, que la dispersion des
//     tailles d'agregat ( de 1 a 3 rho ) rend tres lache.
//
//   * LA GRILLE REGULIERE, retenue. Elle n'a besoin d'aucun majorant : si deux enceintes se
//     rencontrent elles partagent un point, donc une case -- l'exactitude est STRUCTURELLE et non
//     conditionnee a une constante bien choisie. Et le cas lui est favorable, parce que les objets
//     sont homogenes en taille ( tous ~rho diracs ). Le pas est pris sur la taille MEDIANE des
//     boites, ce qui n'est pas un reglage : c'est la seule echelle du probleme.
//
// Une quatrieme piste reste ouverte, et pourrait battre celle-ci : le DIAGRAMME GROSSIER est
// lui-meme un index spatial, mieux adapte a la densite qu'une grille reguliere, et l'ensemble des
// cellules grossieres qu'un convexe rencontre est CONNEXE -- donc explorable par un front exact.
// Elle demande un test convexe-contre-cellule que celle-ci n'a pas besoin d'ecrire.
//
// = LE PARCOURS EST UN « GATHER », ET C'EST CE QUI DEDOUBLONNE
//
// On ne balaie pas les cases en produisant des paires ( il faudrait ensuite trier et unifier :
// une paire apparait dans autant de cases que les deux boites en partagent ). On balaie les
// AGREGATS : pour `A`, on parcourt le contenu des cases que sa boite couvre, on marque chaque `B`
// vu avec le numero `A`, et un `B` deja marque est saute. Le dedoublonnage est un test d'entier,
// la ligne CSR sort dans l'ordre, et rien n'est alloue.

#pragma once

#include "supercell/Enceinte.h"
#include "util/common.h"
#include "bench/Bench.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace pd::supercell {

using namespace pd::bench;

/// La table, en CSR et SYMETRIQUE : la paire `( A, B )` figure dans la ligne de `A` et dans celle
/// de `B`. C'est ce que l'etape 4 veut -- une liste de candidats par agregat.
struct Table {
    SI ns = 0;
    std::vector<SI> deb, vois;
    SI degre_max() const {
        SI m = 0;
        for ( SI a = 0; a + 1 < SI( deb.size() ); ++a ) m = std::max( m, deb[ a + 1 ] - deb[ a ] );
        return m;
    }
};

struct BilanTable {
    double par_agregat = 0;       ///< entrees de table par agregat
    SI     degre_max = 0;
    double cases = 0;             ///< cases de grille couvertes par une boite
    double vues = 0, boites = 0;  ///< paires presentees / retenues par la boite, par agregat
    double t_boites = 0, t_grille = 0, t_paires = 0;
    SI     manques = 0;           ///< sous `--verif` : paires que la force brute trouve en plus
    SI     verifies = 0;
    bool   ok = false;
};

/// LE TEST DE SEPARATION, sur les axes des DEUX enceintes.
///
/// Sur ses propres axes une enceinte donne l'intervalle EXACT ; sur ceux de l'autre elle ne donne
/// qu'un majorant ( sa boite orientee, pour un k-DOP ). Un majorant ne peut que faire RATER une
/// separation, donc garder une paire de trop : la table reste un SUR-ENSEMBLE de ce qu'il faut,
/// ce qui est le seul sens dans lequel elle a le droit de se tromper.
///
/// En 3D le test est de surcroit incomplet -- il ignore les axes arete-contre-arete. Meme
/// remarque : incomplet du bon cote.
template<int D, class Enc>
bool separe( const Enc &A, const Enc &B ) {
    for ( int tour = 0; tour < 2; ++tour ) {
        const Enc &U = tour ? B : A;
        const Enc &V = tour ? A : B;
        for ( SI k = 0; k < U.nb_axes(); ++k ) {
            TF ul, uh;
            U.bande( k, ul, uh );
            Vec<D> n = U.axe( k ), m;
            for ( int d = 0; d < D; ++d ) m[ d ] = -n[ d ];
            const TF vh = V.support( n ), vl = -V.support( m );
            if ( vl > uh || ul > vh )
                return true;
        }
    }
    // LES AXES ARETE-CONTRE-ARETE, quand les deux enceintes sont des boites orientees. Le test par
    // les seules normales de faces est INCOMPLET en 3D : deux boites peuvent etre disjointes sans
    // qu'aucune des six normales ne les separe, et seul un produit vectoriel d'aretes le voit.
    // Ajouter les neuf rend le test EXACT -- ce qui n'a de sens que si le support l'est aussi,
    // d'ou le `croise`.
    if constexpr ( D == 3 && Enc::croise ) {
        for ( SI i = 0; i < A.nb_axes(); ++i )
            for ( SI j = 0; j < B.nb_axes(); ++j ) {
                Vec<3> n = croix( A.axe( i ), B.axe( j ) ), m;
                if ( dot( n, n ) < TF( 1e-24 ) )
                    continue;                            // axes paralleles : rien a en tirer
                for ( int d = 0; d < 3; ++d ) m[ d ] = -n[ d ];
                const TF ah = A.support( n ), al = -A.support( m );
                const TF bh = B.support( n ), bl = -B.support( m );
                if ( bl > ah || al > bh )
                    return true;
            }
    }
    return false;
}

template<int D, class Enc>
BilanTable etape3( const std::vector<Enc> &enc, Table &T, bool verif ) {
    BilanTable b;
    const SI ns = SI( enc.size() );
    T.ns = ns;
    if ( ns == 0 ) { b.ok = true; return b; }

    // --- les boites, et l'echelle du probleme
    double t = now();
    std::vector<Vec<D>> blo( ns ), bhi( ns );
    std::vector<TF> cote( ns );
    for ( SI a = 0; a < ns; ++a ) {
        enc[ a ].boite( blo[ a ], bhi[ a ] );
        TF c = 0;
        for ( int d = 0; d < D; ++d ) c = std::max( c, bhi[ a ][ d ] - blo[ a ][ d ] );
        cote[ a ] = c;
    }
    // le pas : la MEDIANE des cotes. Pas une moyenne -- la distribution des tailles d'agregat a
    // une queue ( `|A|` va de 1 a 3 rho ), et une moyenne s'y laisse tirer.
    std::vector<TF> tri = cote;
    std::nth_element( tri.begin(), tri.begin() + ns / 2, tri.end() );
    const double pas = std::max( 1e-9, double( tri[ ns / 2 ] ) );
    b.t_boites = now() - t;

    // --- la grille : tri par comptage des ( agregat, case )
    t = now();
    Vec<D> lo, hi;
    lo = blo[ 0 ]; hi = bhi[ 0 ];
    for ( SI a = 1; a < ns; ++a )
        for ( int d = 0; d < D; ++d ) {
            lo[ d ] = std::min( lo[ d ], blo[ a ][ d ] );
            hi[ d ] = std::max( hi[ d ], bhi[ a ][ d ] );
        }
    int res[ D ];
    Vec<D> inv;
    SI nc = 1;
    for ( int d = 0; d < D; ++d ) {
        const double e = std::max( 1e-12, double( hi[ d ] - lo[ d ] ) );
        res[ d ] = std::max( 1, std::min( 1024, int( e / pas ) ) );
        inv[ d ] = TF( res[ d ] / ( e * ( 1 + 1e-12 ) ) );
        nc *= res[ d ];
    }
    auto boucle_cases = [ & ]( SI a, auto &&f ) {
        int i0[ D ], i1[ D ];
        for ( int d = 0; d < D; ++d ) {
            i0[ d ] = std::clamp( int( ( blo[ a ][ d ] - lo[ d ] ) * inv[ d ] ), 0, res[ d ] - 1 );
            i1[ d ] = std::clamp( int( ( bhi[ a ][ d ] - lo[ d ] ) * inv[ d ] ), 0, res[ d ] - 1 );
        }
        if constexpr ( D == 2 ) {
            for ( int y = i0[ 1 ]; y <= i1[ 1 ]; ++y )
                for ( int x = i0[ 0 ]; x <= i1[ 0 ]; ++x ) f( SI( x ) * res[ 1 ] + y );
        } else {
            for ( int z = i0[ 2 ]; z <= i1[ 2 ]; ++z )
                for ( int y = i0[ 1 ]; y <= i1[ 1 ]; ++y )
                    for ( int x = i0[ 0 ]; x <= i1[ 0 ]; ++x )
                        f( ( SI( x ) * res[ 1 ] + y ) * res[ 2 ] + z );
        }
    };
    std::vector<SI> deb( nc + 1, 0 );
    long long tot_cases = 0;
    for ( SI a = 0; a < ns; ++a )
        boucle_cases( a, [ & ]( SI k ) { ++deb[ k + 1 ]; ++tot_cases; } );
    for ( SI k = 0; k < nc; ++k ) deb[ k + 1 ] += deb[ k ];
    std::vector<SI> cont( deb[ nc ] );
    { std::vector<SI> pos( deb.begin(), deb.end() - 1 );
      for ( SI a = 0; a < ns; ++a )
          boucle_cases( a, [ & ]( SI k ) { cont[ pos[ k ]++ ] = a; } ); }
    b.t_grille = now() - t;

    // --- les paires, en GATHER : une ligne de table par agregat, dedoublonnee au passage
    t = now();
    T.deb.assign( ns + 1, 0 );
    T.vois.clear();
    T.vois.reserve( size_t( ns ) * ( D == 2 ? 8 : 24 ) );
    std::vector<SI> marque( ns, -1 );
    long long vues = 0, gard_boite = 0;
    for ( SI a = 0; a < ns; ++a ) {
        boucle_cases( a, [ & ]( SI k ) {
            for ( SI u = deb[ k ]; u < deb[ k + 1 ]; ++u ) {
                const SI c = cont[ u ];
                if ( c == a || marque[ c ] == a ) continue;
                marque[ c ] = a;
                ++vues;
                bool disjoint = false;
                for ( int d = 0; d < D && ! disjoint; ++d )
                    disjoint = bhi[ c ][ d ] < blo[ a ][ d ] || bhi[ a ][ d ] < blo[ c ][ d ];
                if ( disjoint ) continue;      // `continue`, PAS `return` : on est dans la
                                               // lambda de la CASE, et il reste des candidats
                ++gard_boite;
                if ( ! separe<D>( enc[ a ], enc[ c ] ) )
                    T.vois.push_back( c );
            }
        } );
        T.deb[ a + 1 ] = SI( T.vois.size() );
    }
    b.t_paires = now() - t;

    b.par_agregat = double( T.vois.size() ) / ns;
    b.degre_max = T.degre_max();
    b.cases = double( tot_cases ) / ns;
    b.vues = double( vues ) / ns;
    b.boites = double( gard_boite ) / ns;

    // LE CONTROLE : la grille doit trouver EXACTEMENT ce que la force brute trouve. On echantillonne
    // -- la force brute est en `ns` par agregat -- et on compare ligne a ligne.
    if ( verif ) {
        const SI pas_ech = std::max( SI( 1 ), ns / 300 );
        std::vector<SI> brut;
        for ( SI a = 0; a < ns; a += pas_ech ) {
            // La force brute doit appliquer LE MEME test que la grille, boite comprise. Sans
            // elle, elle garde des paires que la grille ecarte a juste titre : `separe` est
            // CONSERVATIF ( il majore le support hors des axes propres ), donc il peut rendre
            // « non separe » pour deux enceintes dont les boites, elles, sont disjointes. La
            // grille est alors la plus fine des deux, et la comptabiliser comme fautive serait
            // exactement l'inverse de la verite.
            brut.clear();
            for ( SI c = 0; c < ns; ++c ) {
                if ( c == a ) continue;
                bool disjoint = false;
                for ( int d = 0; d < D && ! disjoint; ++d )
                    disjoint = bhi[ c ][ d ] < blo[ a ][ d ] || bhi[ a ][ d ] < blo[ c ][ d ];
                if ( ! disjoint && ! separe<D>( enc[ a ], enc[ c ] ) ) brut.push_back( c );
            }
            std::vector<SI> ligne( T.vois.begin() + T.deb[ a ], T.vois.begin() + T.deb[ a + 1 ] );
            std::sort( ligne.begin(), ligne.end() );
            // la DIFFERENCE SYMETRIQUE : ce que la force brute trouve et pas la grille ( le
            // defaut qu'on craint ) comme l'inverse ( un dedoublonnage ou un indice fautif ).
            for ( SI c : brut )
                if ( ! std::binary_search( ligne.begin(), ligne.end(), c ) ) ++b.manques;
            for ( SI c : ligne )
                if ( ! std::binary_search( brut.begin(), brut.end(), c ) ) ++b.manques;
            ++b.verifies;
        }
    }
    // la table doit etre SYMETRIQUE : `B` dans la ligne de `A` implique `A` dans celle de `B`.
    // C'est gratuit a verifier et ca attrape toute asymetrie du parcours ou du test.
    bool sym = true;
    for ( SI a = 0; a < ns && sym; ++a )
        for ( SI u = T.deb[ a ]; u < T.deb[ a + 1 ] && sym; ++u ) {
            const SI c = T.vois[ u ];
            sym = std::find( T.vois.begin() + T.deb[ c ], T.vois.begin() + T.deb[ c + 1 ], a )
                != T.vois.begin() + T.deb[ c + 1 ];
        }
    b.ok = sym && b.manques == 0;
    return b;
}

template<int D>
void ligne_table( const Cloud<D> &cl, const BilanTable &b ) {
    std::printf( "      %-20s table %6.2f /agregat ( max %d ) | cases %5.2f  presentees %6.2f"
                 " -> boite %6.2f -> gardees %6.2f | boites %6.3f  grille %6.3f  paires %6.3f s"
                 " | %d verifies, %d manques | %s\n",
                 cl.nom.c_str(), b.par_agregat, int( b.degre_max ), b.cases, b.vues, b.boites,
                 b.par_agregat, b.t_boites, b.t_grille, b.t_paires,
                 int( b.verifies ), int( b.manques ), b.ok ? "ok" : "FAUX" );
}

} // namespace pd::supercell
