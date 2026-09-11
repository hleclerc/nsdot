// L'ETAPE 1 : le Voronoi grossier, les agregats, et un dirac MEDIAN par agregat.
//
// Elle produit le `Gros<D>` que toute la suite consomme : les sites, l'agregat de chaque dirac, les
// membres en CSR, la connectivite grossiere en CSR, et le median. Tout est indexe par NUMERO
// D'AGREGAT, y compris la connectivite -- les deux parlent la meme langue sans traduction.
//
// L'AGREGATION EST ARBITRAIRE : aucune propriete de correction ne depend de la facon de grouper,
// seulement la performance. L'affectation se fait donc au plus proche EUCLIDIEN, sans se soucier
// des poids ; ce n'est pas une approximation, c'est un choix libre.

#pragma once

#include "supercell/GrilleSites.h"
#include "supercell/Ordre.h"
#include "spatial_accel/AaBsp.h"
#include "geometry/PowerDiagram.h"
#include "bench/Bench.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

namespace pd::supercell {

using namespace pd::bench;

// ------------------------------------------------------------------ L'ETAPE 1

/// Ce que l'etape 1 laisse a l'etape 2. Tout est indexe par l'indice ACCELERATEUR du site, qui
/// sert donc de numero d'agregat -- c'est aussi celui que rend `for_each_facet`, donc la
/// connectivite grossiere et les agregats parlent la meme langue sans traduction.
template<int D>
struct Gros {
    SI ns = 0;
    std::vector<SI> site;                   ///< site -> germe
    std::vector<SI> lab;                    ///< germe -> agregat
    std::vector<SI> mdeb, mem;              ///< CSR des membres
    std::vector<SI> adeb, adj;              ///< CSR de la connectivite grossiere
    std::vector<SI> median;                 ///< agregat -> le germe MEDIAN ( indice PERMUTE )

    /// LE NUAGE RENUMEROTE PAR AGREGAT, et c'est ce qui rend tout le reste contigu.
    ///
    /// `mem` etait deja la permutation ; on l'applique aux coordonnees. Les membres d'un agregat
    /// occupent alors la plage `[ mdeb[ a ], mdeb[ a + 1 ] )` de `Pp`, et ceux d'un agregat voisin
    /// une autre plage contigue -- donc les listes de candidats deviennent des SUITES DE PLAGES au
    /// lieu d'indices disperses. Sans ca, chaque lecture de position est un acces aleatoire dans
    /// un tableau de `n` doubles, et il n'y a pas de SoA possible.
    ///
    /// C'est de la PREPARATION : l'agregation est gelee d'une iteration de Newton a l'autre, donc
    /// la permutation se calcule une fois.
    std::vector<TF> Pp[ D ], Wp;
    const TF *Ppp[ D ] = {};
    bool pese = false;                      ///< le nuage porte-t-il des poids
};

struct BilanGros {
    SI   ns = 0;
    SI   amin = 0, amed = 0, amax = 0;
    double deg = 0;
    double rayon = 0;
    double somme = 0;
    SI   novf = 0;
    double t_ordre = 0, t_gros = 0, t_affect = 0, t_median = 0, t_perm = 0;
    bool ok = false;
};

/// Le corps, une fois la dimension et le drapeau « poids » fixes a la compilation.
template<int D, bool Weighted>
BilanGros etape1( const Cloud<D> &cl, SI rho, bool morton_ordre, SI leaf, Gros<D> &G ) {
    BilanGros b;
    using Cell = CellFor<D, D == 2 ? 64 : 256>;

    // --- l'ordre spatial
    double t = now();
    std::vector<SI> ord;
    if ( morton_ordre ) ordre_morton<D>( cl, ord );
    else                ordre_bsp<D>( cl, ord, leaf );
    b.t_ordre = now() - t;

    // --- l'echantillonnage : un germe tous les `rho` DANS CET ORDRE
    std::vector<SI> ech;
    ech.reserve( cl.n / rho + 1 );
    for ( SI i = 0; i < cl.n; i += rho )
        ech.push_back( ord[ i ] );
    b.ns = SI( ech.size() );

    // --- le diagramme grossier, et sa connectivite
    t = now();
    AaBspT<D> gros;
    gros.build_sel( cl.P, cl.W, ech.data(), b.ns, leaf );
    PowerDiagram<Cell, AaBspT<D>, true, false, false, Weighted> pd{ gros };
    G.ns = b.ns;
    G.site.resize( b.ns );
    for ( SI k = 0; k < b.ns; ++k ) G.site[ k ] = gros.seed_id( k );
    // `cut_with` rend `order[ k ]`, c'est-a-dire l'indice du germe DANS LE NUAGE et non sa place
    // dans l'accelerateur -- donc `for_each_facet` aussi. Il faut l'inverse pour parler en numeros
    // d'agregat, et il ne coute qu'un tableau : les sites sont peu nombreux.
    std::vector<SI> inv( cl.n, -1 );
    for ( SI k = 0; k < b.ns; ++k ) inv[ G.site[ k ] ] = k;
    G.adeb.assign( b.ns + 1, 0 );
    G.adj.clear();
    TF somme = 0;
    for ( SI k = 0; k < b.ns; ++k ) {
        Cell c;
        pd.make_cell( c, k );
        somme += c.measure();
        // la connectivite grossiere sort du diagramme sans un calcul de plus.
        c.for_each_facet( [ & ]( SI j, TF ) { G.adj.push_back( inv[ j ] ); } );
        G.adeb[ k + 1 ] = SI( G.adj.size() );
    }
    b.somme = double( somme );
    b.novf = pd.nb_overflow.load();
    b.deg = double( G.adj.size() ) / double( b.ns );
    b.t_gros = now() - t;

    // --- l'affectation de chaque dirac a son site
    t = now();
    GrilleSites<D> g;
    g.build( cl.P, G.site.data(), b.ns );
    G.lab.resize( cl.n );
    std::vector<SI> taille( b.ns, 0 );
    double rayon = 0;
    for ( SI i = 0; i < cl.n; ++i ) {
        Vec<D> x;
        for ( int d = 0; d < D; ++d ) x[ d ] = cl.P[ d ][ i ];
        const SI s = g.plus_proche( x );
        G.lab[ i ] = s;
        ++taille[ s ];
        Vec<D> y;
        for ( int d = 0; d < D; ++d ) y[ d ] = cl.P[ d ][ G.site[ s ] ];
        rayon += std::sqrt( double( dist2( x, y ) ) );
    }
    G.mdeb.assign( b.ns + 1, 0 );
    for ( SI k = 0; k < b.ns; ++k ) G.mdeb[ k + 1 ] = G.mdeb[ k ] + taille[ k ];
    G.mem.resize( cl.n );
    { std::vector<SI> pos( G.mdeb.begin(), G.mdeb.end() - 1 );
      for ( SI i = 0; i < cl.n; ++i ) G.mem[ pos[ G.lab[ i ] ]++ ] = i; }
    b.t_affect = now() - t;

    // --- LE DIRAC MEDIAN de chaque agregat : le membre le plus proche du barycentre.
    //
    // Pourquoi un DIRAC et pas le barycentre lui-meme : le plan qu'on en tire doit etre un vrai
    // demi-espace de Laguerre, sans quoi la cellule cessee d'etre un SUR-ENSEMBLE de la vraie. Le
    // germe echantillonne ferait l'affaire, mais il est pris dans l'ordre spatial, donc excentre
    // en moyenne ; le median est centre, et un plan issu d'un point central coupe plus court.
    t = now();
    G.median.assign( b.ns, -1 );
    for ( SI k = 0; k < b.ns; ++k ) {
        const SI d0 = G.mdeb[ k ], d1 = G.mdeb[ k + 1 ];
        if ( d0 == d1 ) continue;
        Vec<D> g0{};
        for ( int d = 0; d < D; ++d ) g0[ d ] = 0;
        for ( SI u = d0; u < d1; ++u )
            for ( int d = 0; d < D; ++d ) g0[ d ] += cl.P[ d ][ G.mem[ u ] ];
        for ( int d = 0; d < D; ++d ) g0[ d ] /= TF( d1 - d0 );
        TF best = TF( 1e300 );
        for ( SI u = d0; u < d1; ++u ) {
            Vec<D> x;
            for ( int d = 0; d < D; ++d ) x[ d ] = cl.P[ d ][ G.mem[ u ] ];
            const TF dd = dist2( x, g0 );
            if ( dd < best ) { best = dd; G.median[ k ] = G.mem[ u ]; }
        }
    }
    b.t_median = now() - t;

    // --- LA PERMUTATION DU NUAGE, par agregat.
    t = now();
    for ( int d = 0; d < D; ++d ) {
        G.Pp[ d ].resize( cl.n );
        for ( SI k = 0; k < cl.n; ++k ) G.Pp[ d ][ k ] = cl.P[ d ][ G.mem[ k ] ];
        G.Ppp[ d ] = G.Pp[ d ].data();
    }
    G.pese = cl.W != nullptr;
    if ( G.pese ) {
        G.Wp.resize( cl.n );
        for ( SI k = 0; k < cl.n; ++k ) G.Wp[ k ] = cl.W[ G.mem[ k ] ];
    }
    // les medians passent en indices PERMUTES, comme tout le reste
    {   std::vector<SI> pos( cl.n );
        for ( SI k = 0; k < cl.n; ++k ) pos[ G.mem[ k ] ] = k;
        for ( SI a = 0; a < b.ns; ++a ) if ( G.median[ a ] >= 0 ) G.median[ a ] = pos[ G.median[ a ] ];
    }
    b.t_perm = now() - t;

    const double h = std::pow( 1.0 / double( cl.n ), 1.0 / D );
    b.rayon = rayon / cl.n / h;
    std::vector<SI> tri = taille;
    std::sort( tri.begin(), tri.end() );
    b.amin = tri.front(); b.amed = tri[ tri.size() / 2 ]; b.amax = tri.back();

    // LES DEUX CONTROLES. La somme des mesures grossieres est celle d'un diagramme complet, donc
    // `1` -- c'est le meme temoin que partout ailleurs dans ce banc. Et l'affectation doit etre une
    // PARTITION : chaque dirac dans un agregat et un seul, ce que la somme des tailles verifie.
    const SI stot = std::accumulate( taille.begin(), taille.end(), SI( 0 ) );
    b.ok = std::fabs( b.somme - 1.0 ) < 1e-9 && b.novf == 0 && stot == cl.n;
    return b;
}

template<int D>
void ligne_gros( const Cloud<D> &cl, const BilanGros &b ) {
    std::printf( "  %-24s n=%8d  sites %7d | |A| %4d/%4d/%4d  deg %5.2f  rayon %5.2f h"
                 " | ordre %6.3f  gros %6.3f  affect %6.3f  median %6.3f  perm %6.3f s | %s\n",
                 cl.nom.c_str(), int( cl.n ), int( b.ns ), int( b.amin ), int( b.amed ),
                 int( b.amax ), b.deg, b.rayon, b.t_ordre, b.t_gros, b.t_affect, b.t_median, b.t_perm,
                 b.ok ? "ok" : ( b.novf ? "DEBORDEMENT" : "FAUX" ) );
}

} // namespace pd::supercell
