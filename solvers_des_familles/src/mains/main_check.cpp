// =====================================================================================
// L'EXACTITUDE DE L'ENGIN : le fournisseur BSP contre le balayage complet, en 2D et en 3D, en
// Voronoi et en Laguerre. C'est le seul controle qui teste l'ELAGAGE : meme noyau, memes plans,
// seul le fournisseur change.
//
// Trois choses sont comparees, cellule par cellule :
//   * la somme des mesures, qui doit valoir 1 ( la COMPLETUDE : un elagage trop agressif fait
//     deborder une cellule sur sa voisine et la somme monte ) ;
//   * les VOISINAGES, BSP contre balayage direct -- une coupe ratee s'y voit meme quand elle ne
//     change pas la mesure ;
//   * les voisinages, balayage direct contre balayage INVERSE : meme algorithme, meme resultat
//     mathematique, seul l'ordre des coupes change. Ce desaccord-la est le plancher du flottant,
//     pas un defaut de l'elagage -- sans lui on lirait « 2 cellules fausses » en `float` et on
//     chercherait un bug qui n'existe pas.
//
//   xmake run check                  # n = 5000, 2D puis 3D
//   xmake run check --kernel float
// =====================================================================================

#include "bench/Dispatch.h"
#include "cell/Balayage.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace sf;

namespace {

/// LA CELLULE DU TEMOIN EST BIEN PLUS GRANDE que celle de l'engin : le balayage propose les
/// germes dans l'ordre du tableau, donc la cellule reste enorme pendant des centaines de coupes
/// effectives avant que les vraies voisines arrivent -- et en 3D la liste de coupes sature a 128.
template<class PD>
using Temoin = std::conditional_t<PD::dim == 2, d2::Atelier<typename PD::TKernel,512>,
                                                d3::Cellule3<typename PD::TKernel,1024,8>>;

/// les voisins de la cellule, en identifiants tries -- la seule chose qu'on compare.
template<int D, class Cel>
std::vector<int> voisins( const Cel &cel ) {
    std::vector<int> v;
    if constexpr ( D == 2 ) {
        for ( int i = 0; i < cel.nb; ++i )
            if ( cel.cid[ i ] >= 0 ) v.push_back( cel.cid[ i ] );
    } else {
        d2::SI32 buf[ Cel::max_nv ];
        const int m = cel.voisins( buf, Cel::max_nv );
        for ( int i = 0; i < m; ++i )
            if ( buf[ i ] >= 0 ) v.push_back( buf[ i ] );
    }
    std::sort( v.begin(), v.end() );
    return v;
}

/// la cellule `k` par le BALAYAGE, dans un sens ou dans l'autre.
template<class PD, bool POIDS>
bool balaye( const PD &pd, SI k, int sens, Temoin<PD> &cel ) {
    using TK = typename PD::TKernel;
    const TK w0 = POIDS ? pd.w[ k ] : TK( 0 );
    if constexpr ( PD::dim == 2 ) {
        Balayage2<TK,POIDS> f{ pd.c[ 0 ].data(), pd.c[ 1 ].data(), pd.w.data(), pd.ids.data(),
                               int( pd.n ), 0, sens, pd.c[ 0 ][ k ], pd.c[ 1 ][ k ], w0, pd.ids[ k ] };
        d2::moteur<TK>( &f, &cel );
        return cel.nb >= 0;
    } else {
        Balayage3<TK,POIDS> f{ pd.c[ 0 ].data(), pd.c[ 1 ].data(), pd.c[ 2 ].data(), pd.w.data(),
                               pd.ids.data(), int( pd.n ), 0, sens,
                               pd.c[ 0 ][ k ], pd.c[ 1 ][ k ], pd.c[ 2 ][ k ], w0, pd.ids[ k ] };
        return d3::moteur( &f, &cel ) == 0;
    }
}

template<class PD>
int verifie( const Args &a, const Nuage<PD::dim> &nu, const std::vector<SI> &seules = {} ) {
    constexpr int D = PD::dim;
    PD pd;
    if ( a.wscale < 0 ) {                                // `--weights -1` : le chemin de Newton, arbre nu puis poids
        pd.build( nu.P, nullptr, nu.n, a.leaf );
        if ( nu.W ) pd.set_weights( nu.W, a.par );
    } else
        pd.build( nu.P, nu.W, nu.n, a.leaf );
    const SI n = nu.n;
    if ( nu.W ) {                                        // les majorants sont-ils VALIDES ?
        SI faux = 0; TF pire = 0;
        for ( const auto &nd : pd.arbre.nodes )
            for ( SI k = nd.beg; k < nd.end; ++k ) {
                TF m = nd.wm.b;
                for ( int d = 0; d < D; ++d ) m += TF( nd.wm.a[ d ] ) * pd.arbre.p[ d ][ k ];
                if ( pd.arbre.pw[ k ] > m ) {
                    if ( faux < 3 )
                        std::printf( "    noeud [%d,%d) a = ( %.6e %.6e ) b = %.10e : germe %d y = ( %.6f %.6f ) w = %.10e > %.10e de %.3e\n",
                                     int( nd.beg ), int( nd.end ), double( nd.wm.a[ 0 ] ), double( nd.wm.a[ 1 ] ), double( nd.wm.b ), int( k ),
                                     double( pd.arbre.p[ 0 ][ k ] ), double( pd.arbre.p[ 1 ][ k ] ), double( pd.arbre.pw[ k ] ), double( m ), double( pd.arbre.pw[ k ] - m ) );
                    ++faux; pire = std::max( pire, pd.arbre.pw[ k ] - m );
                }
            }
        std::printf( "  majorants : %d violations sur %d noeuds, pire %.3e\n", int( faux ), int( pd.arbre.nodes.size() ), double( pire ) );
    }

    std::vector<TF> res;
    const double t0 = now();
    const SI deb = pd.measures( res, a.par );
    const double t_bsp = now() - t0;
    TF somme = 0;
    for ( TF v : res ) somme += v;

    int faux_bsp = 0, faux_ordre = 0, faux_adj = 0, deb_temoin = 0;
    for ( SI k = 0; k < n; ++k ) {
        if ( ! seules.empty() && std::find( seules.begin(), seules.end(), pd.ids[ k ] ) == seules.end() ) continue;
        typename PD::Cell ca;
        Temoin<PD> cb, cc;
        pd.cellule( k, ca );
        const bool ok_b = nu.W ? balaye<PD,true>( pd, k, +1, cb ) : balaye<PD,false>( pd, k, +1, cb );
        const bool ok_c = nu.W ? balaye<PD,true>( pd, k, -1, cc ) : balaye<PD,false>( pd, k, -1, cc );
        if ( ! ok_b || ! ok_c ) { ++deb_temoin; continue; }
        const auto va = voisins<D>( ca ), vb = voisins<D>( cb ), vc = voisins<D>( cc );
        if ( ! seules.empty() ) {
            auto imprime = [ & ]( const char *quoi, const std::vector<int> &v ) {
                std::printf( "      cellule %d %-8s :", int( pd.ids[ k ] ), quoi );
                for ( int x : v ) std::printf( " %d", x );
                std::printf( "\n" );
            };
            imprime( "BSP", va ); imprime( "direct", vb ); imprime( "inverse", vc );
        }
        faux_bsp   += va != vb;
        faux_ordre += vb != vc;
        if constexpr ( D == 3 ) faux_adj += ca.verifie();
    }

    const bool ok = std::fabs( somme - 1 ) < 1e-6 && deb == 0 && deb_temoin == 0
                    && faux_bsp <= faux_ordre && faux_adj == 0;
    std::printf( "  %-24s n=%-6d %-8s : somme %.9f  debordes %d  BSP vs direct %d/%d  direct vs inverse %d/%d"
                 "%s  ( %.3f s )%s\n",
                 nu.nom.c_str(), int( n ), nu.W ? "Laguerre" : "Voronoi", double( somme ), int( deb ),
                 faux_bsp, int( n ), faux_ordre, int( n ),
                 D == 3 ? ( "  adjacence " + std::to_string( faux_adj ) ).c_str() : "",
                 t_bsp, ok ? "" : "   <-- FAUX" );
    if ( deb_temoin )
        std::printf( "      le TEMOIN a deborde %d fois : ces cellules n'ont pas ete comparees\n", deb_temoin );
    return ! ok;
}

template<int D>
int deroule( const Args &a, const std::vector<SI> &seules ) {
    int bad = 0;
    std::printf( "=== %dD  kernel=%s maxnv=%d leaf=%d\n", D, a.kernel.c_str(), a.nv( D ), int( a.leaf ) );
    if ( ! a.load.empty() ) {
        for ( const Nuage<D> &nu : a.nuages<D>() ) {
            if ( nu.absent ) return 1;
            bad += dispatch<D>( a, [ & ]( auto tag ) { return verifie<typename decltype( tag )::type>( a, nu, seules ); } );
        }
        return bad;
    }
    for ( double ws : { 0.0, 1.0 } ) {
        const Nuage<D> nu = nuage_uniforme<D>( a.n, a.graine, ws );
        bad += dispatch<D>( a, [ & ]( auto tag ) {
            return verifie<typename decltype( tag )::type>( a, nu );
        } );
    }
    return bad;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    a.n = 5000;                                          // le balayage est en O( n^2 )
    std::vector<SI> seules;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( a.parse( s, i, argc, argv ) ) continue;
        if ( s == "--cellule" && i + 1 < argc ) { seules.push_back( std::atoi( argv[ ++i ] ) ); continue; }
        std::printf( "usage: check [options]\n" );
        Args::usage();
        std::printf( "  --cellule I     avec --load : ne confronter que cette cellule ( repetable )\n" );
        return s == "--help" || s == "-h" ? 0 : 1;
    }
    a.finalise();

    int bad = 0;
    if ( a.dims != 3 ) bad += deroule<2>( a, seules );
    if ( a.dims != 2 ) bad += deroule<3>( a, seules );
    return bad ? 1 : 0;
}
