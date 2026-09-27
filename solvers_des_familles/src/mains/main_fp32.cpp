// =====================================================================================
// CE QUE LA SIMPLE PRECISION COUTE A LA GEOMETRIE, mesure au lieu d'etre suppose.
//
// Le meme nuage, les memes poids, le meme moteur, le meme elagage : SEUL `TK` change. On compare
// cellule par cellule la mesure et les `c_ij`, et on regarde comment l'ecart varie avec `n`, avec
// la dimension, et avec l'amplitude des poids.
//
// = LE MODELE QU'ON VIENT VERIFIER
//
// Le plan qui separe le germe `0` du germe `j` est rendu par le fournisseur sous la forme
//
//      dx x + dy y = off,   off = 1/2 ( |p_j|^2 - |p_0|^2 ) + 1/2 ( w_0 - w_j )
//
// et le noyau l'evalue en chaque sommet : `s = dx vx + dy vy - off`. Deux annulations s'y
// cachent, et elles n'ont pas la meme cause :
//
//   * LA POSITION. `dx` vaut `h`, `vx` vaut `1`, donc `dx vx` vaut `h` -- alors que `s` doit
//     valoir `h^2` ( la distance au plan, fois `|d| = h` ). On perd `log( 1 / h )` chiffres par
//     annulation. Et `dx` lui-meme est la difference de deux flottants DEJA arrondis a `TK` :
//     son erreur absolue est `eps |p|`, pas `eps h`.
//   * LE POIDS. `w_0 - w_j` est la difference de deux poids arrondis a `TK` : erreur `eps |w|`
//     sur `off`, donc un plan deplace de `eps |w| / ( 2 h )` le long de sa normale.
//
// Rapporte a la taille de cellule `h`, ca donne la prediction
//
//      erreur relative ~ eps |p| / h  +  eps |w| / h^2
//
// avec `h = n^( -1/D )`. LE SECOND TERME EST EN `n^( 2/D )` : a `n` fixe, la 2D est le cas dur --
// `h^2 = 1/n` contre `n^( -2/3 )`.
//
// LES DEUX TERMES SONT MAINTENANT REPARES ( README § 19 ), par trois moyens qui sont le meme :
// porter la DIFFERENCE et non la valeur. Le poids par `cell/Plan.h`, la position par le REPERE DU
// GERME ( `cell/Contrat2D.h` ), et la memoire des premieres coupes par la BOITE DE DEPART
// ( `cell/Boite.h` ). Ce qui reste est plat en `n`, a 1.3 fois l'epsilon du `float`.
//
// `SF_DIL=0` rend le comportement d'avant les boites -- c'est comme ca que les tables du § 19.7
// se refont.
//
// `--centre` teste une reparation bon marche du terme de poids : retrancher leur moyenne AVANT de
// les arrondir. C'est une invariance EXACTE du diagramme de puissance, donc tout ecart mesure est
// du flottant et rien d'autre -- et ca ne change RIEN, ces champs etant deja a moyenne nulle.
//
//   fp32 --2d                       le balayage en n, uniforme, poids ~ h^2
//   fp32 --2d --weights 100         des poids cent fois plus gros
//   fp32 --3d
//   fp32 --2d --cas ../2d_des_familles/cases/lines5_n100000_s0.1_equal.txt
// =====================================================================================

#include "bench/Args.h"
#include "bench/Nuages.h"
#include "cell/Boite.h"
#include "diagram/PowerDiagram.h"
#include "solver/Laplacien.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sf;

namespace {

struct Quant { double med, p99, p9999, max; };

Quant quantiles( std::vector<double> &e ) {
    if ( e.empty() ) return { 0, 0, 0, 0 };
    std::sort( e.begin(), e.end() );
    auto q = [ & ]( double f ) { return e[ std::min( e.size() - 1, size_t( f * e.size() ) ) ]; };
    return { q( 0.5 ), q( 0.99 ), q( 0.9999 ), e.back() };
}

/// la mesure de chaque cellule et les facettes, TRIEES par ( i, j ) -- ce qu'on compare.
template<class PD>
void diagramme( const Nuage<PD::dim> &nu, const TF *W, const Parallel &par,
                std::vector<TF> &a, std::vector<Facette> &fa, SI &deborde, SI leaf ) {
    PD pd;
    pd.build( nu.P, W, nu.n, leaf );
    const int nth = std::max( par.threads, 1 );
    std::vector<std::vector<Facette>> par_th( nth );
    deborde = pd.measures_and_facets( a, par, [ & ]( int t, d2::SI32 i, d2::SI32 j, TF m ) {
        par_th[ t ].push_back( Facette{ SI( i ), SI( j ), m } );
    } );
    fa.clear();
    for ( auto &v : par_th ) fa.insert( fa.end(), v.begin(), v.end() );
    std::sort( fa.begin(), fa.end(), []( const Facette &x, const Facette &y ) {
        return x.i != y.i ? x.i < y.i : x.j < y.j; } );
}

/// UNE LIGNE DE TABLE : float contre double, sur le meme nuage et les memes poids.
template<int D>
void compare( const Nuage<D> &nu, const Args &a, bool centre, const char *etiq ) {
    constexpr int NV = D == 2 ? 64 : 128;
    const SI n = nu.n;

    // les poids que voit le diagramme : tels quels, ou CENTRES ( invariance exacte )
    std::vector<TF> wc;
    const TF *W = nu.W;
    double wmax = 0, wmoy = 0;
    if ( nu.W ) {
        for ( SI i = 0; i < n; ++i ) wmoy += nu.W[ i ];
        wmoy /= n;
        for ( SI i = 0; i < n; ++i ) wmax = std::max( wmax, std::fabs( nu.W[ i ] - ( centre ? wmoy : 0 ) ) );
        if ( centre ) {
            wc.resize( n );
            for ( SI i = 0; i < n; ++i ) wc[ i ] = nu.W[ i ] - wmoy;
            W = wc.data();
        }
    }

    std::vector<TF> ad, af;
    std::vector<Facette> fd, ff;
    SI bd = 0, bf = 0;
    nb_reprises.store( 0 ); nb_rep_face.store( 0 ); nb_rep_vide.store( 0 );
    diagramme<PowerDiagram<D,double,NV>>( nu, W, a.par, ad, fd, bd, a.leaf );
    const long long rep_d = nb_reprises.exchange( 0 );
    nb_rep_face.store( 0 ); nb_rep_vide.store( 0 );
    diagramme<PowerDiagram<D,float ,NV>>( nu, W, a.par, af, ff, bf, a.leaf );
    const long long rep_f = nb_reprises.load();
    const long long r_face = nb_rep_face.load(), r_vide = nb_rep_vide.load();

    // ---- la mesure, rapportee a la mesure MOYENNE ( `1 / n` sur le cube unite )
    std::vector<double> em;
    em.reserve( n );
    for ( SI i = 0; i < n; ++i ) em.push_back( std::fabs( double( af[ i ] ) - double( ad[ i ] ) ) * n );
    const Quant qm = quantiles( em );

    // ---- les `c_ij` : erreur RELATIVE sur les paires communes, et les paires qui manquent
    std::vector<double> ec;
    SI manque = 0;
    {
        size_t p = 0, q = 0;
        while ( p < fd.size() && q < ff.size() ) {
            const Facette &x = fd[ p ], &y = ff[ q ];
            if ( x.i == y.i && x.j == y.j ) {
                const double d = double( x.c );
                if ( d > 0 ) ec.push_back( std::fabs( double( y.c ) - d ) / d );
                ++p; ++q;
            } else if ( x.i < y.i || ( x.i == y.i && x.j < y.j ) ) { ++manque; ++p; }
            else                                                   { ++manque; ++q; }
        }
        manque += SI( fd.size() - p ) + SI( ff.size() - q );
    }
    const Quant qc = quantiles( ec );

    // ---- la prediction du modele
    const double eps = 6e-8, h = std::pow( double( n ), -1.0 / D );
    const double t_pos = eps / h, t_poi = eps * wmax / ( h * h );

    std::printf( "  %-24s n %8d |w|max %8.2e h^2 %8.2e | masse med %8.2e p99 %8.2e max %8.2e"
                 " | c_ij med %8.2e p99 %8.2e max %8.2e | %6d aretes dissidentes | modele pos %7.1e poids %7.1e\n",
                 etiq, int( n ), wmax, h * h, qm.med, qm.p99, qm.max, qc.med, qc.p99, qc.max,
                 int( manque ), t_pos, t_poi );
    if ( bd || bf ) std::printf( "      ( debordements : double %d, float %d )\n", int( bd ), int( bf ) );
    if ( rep_d || rep_f )
        std::printf( "      ( reprises : %.2f %% en double, %.2f %% en float -- dont %.2f %% face touchee, %.2f %% VIDE )\n",
                     100.0 * rep_d / n, 100.0 * rep_f / n, 100.0 * r_face / n, 100.0 * r_vide / n );
}

template<int D>
void deroule( const Args &a, bool centre, const std::string &cas ) {
    std::printf( "=== %dD  ( eps_float = 6e-8 ; l'ecart de masse est rapporte a la masse MOYENNE 1/n )\n", D );
    if ( ! cas.empty() ) {
        Nuage<D> nu;
        if ( ! charge_nuage<D>( cas, nu ) ) { std::printf( "  cas illisible : %s\n", cas.c_str() ); return; }
        compare<D>( nu, a, centre, nu.nom.c_str() );
        return;
    }
    for ( SI n : { SI( 1000 ), SI( 10000 ), SI( 100000 ), SI( 1000000 ) } ) {
        if ( n > a.n ) break;
        Nuage<D> nu = nuage_uniforme<D>( n, a.graine, a.wscale );
        compare<D>( nu, a, centre, a.wscale == 0 ? "uniforme / Voronoi" : "uniforme / Laguerre" );
    }
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    a.n = 1000000;
    a.wscale = 1;
    bool centre = false;
    std::string cas;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if ( a.parse( s, i, argc, argv ) ) continue;
        else if ( s == "--centre" ) centre = true;
        else if ( s == "--cas" )    cas = val();
        else {
            std::printf( "usage: fp32 [options]\n" );
            Args::usage();
            std::printf( "  --centre        retrancher la moyenne des poids AVANT de les arrondir ( invariance EXACTE )\n"
                         "  --cas FILE      un nuage de cases/ au lieu du balayage uniforme\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    a.finalise();
    if ( a.dims != 3 ) deroule<2>( a, centre, cas );
    if ( a.dims != 2 ) deroule<3>( a, centre, cas );
    return 0;
}
