// =====================================================================================
// LE TELEVERSEMENT ET LE LANCEMENT. L'arbre de l'hote devient des `Noeud<TK,D>` ( pentes du
// majorant converties dans le flottant du noyau ), les germes montent dans l'ordre de l'arbre, et
// `mesures` choisit le noyau : la dimension et le flottant sont ceux du type, la variante, le
// nombre de sommets et Voronoi / Laguerre deviennent des parametres de template ICI.
//
// Le chrono est celui des evenements CUDA autour du noyau seul ; la descente des `n` doubles est
// comptee a part. `deborde` est un compteur device incremente par atomique.
// =====================================================================================

#include "gpu/Mesures.h"
#include "gpu/Fil2D.cuh"
#include "gpu/Voies2D.cuh"
#include "gpu/Fil3D.cuh"
#include "gpu/Voies3D.cuh"

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace sf::gpu {

#define CUDA_OK( x ) do { cudaError_t e_ = ( x ); if ( e_ != cudaSuccess ) { \
    std::fprintf( stderr, "CUDA : %s ( %s:%d )\n", cudaGetErrorString( e_ ), __FILE__, __LINE__ ); std::exit( 2 ); } } while ( 0 )

template<int D, class TK>
struct DiagrammeGpu<D,TK>::Impl {
    Noeud<TK,D> *nodes = nullptr;
    TK          *c[ D ] = {};
    TK          *w = nullptr;
    int         *ids = nullptr;
    double      *res = nullptr;
    int         *deb = nullptr, *deb2 = nullptr;     ///< les compteurs de debordement des deux passes
    int         *liste = nullptr;                    ///< les rangs qui ont deborde a la premiere passe
    int          n = 0, nn = 0;
    bool         poids = false;

    Arbre<TK,D> arbre() const {
        Arbre<TK,D> a;
        a.nodes = nodes;
        for ( int d = 0; d < D; ++d ) a.c[ d ] = c[ d ];
        a.w = w; a.ids = ids; a.n = n;
        return a;
    }
};

template<int D, class TK>
DiagrammeGpu<D,TK>::DiagrammeGpu( const AaBspT<D> &arbre ) : impl( new Impl ) {
    const double t0 = now();
    Impl &m = *impl;
    m.n  = arbre.nb_seeds();
    m.nn = int( arbre.nodes.size() );
    m.poids = arbre.has_weights();

    std::vector<Noeud<TK,D>> nds( m.nn );
    for ( int i = 0; i < m.nn; ++i ) {
        const auto &s = arbre.nodes[ i ];
        Noeud<TK,D> &d = nds[ i ];
        for ( int k = 0; k < D; ++k ) { d.lo[ k ] = TK( s.lo[ k ] ); d.hi[ k ] = TK( s.hi[ k ] ); d.a[ k ] = m.poids ? TK( s.wm.a[ k ] ) : TK( 0 ); }
        d.b = m.poids ? TK( s.wm.b ) : TK( 0 );
        d.beg = int( s.beg ); d.end = int( s.end ); d.right = int( s.right );
    }
    CUDA_OK( cudaMalloc( &m.nodes, m.nn * sizeof( Noeud<TK,D> ) ) );
    CUDA_OK( cudaMemcpy( m.nodes, nds.data(), m.nn * sizeof( Noeud<TK,D> ), cudaMemcpyHostToDevice ) );

    std::vector<TK> tmp( m.n );
    for ( int d = 0; d < D; ++d ) {
        for ( int k = 0; k < m.n; ++k ) tmp[ k ] = TK( arbre.seed_c( k, d ) );
        CUDA_OK( cudaMalloc( &m.c[ d ], m.n * sizeof( TK ) ) );
        CUDA_OK( cudaMemcpy( m.c[ d ], tmp.data(), m.n * sizeof( TK ), cudaMemcpyHostToDevice ) );
    }
    if ( m.poids ) {
        for ( int k = 0; k < m.n; ++k ) tmp[ k ] = TK( arbre.seed_w( k ) );
        CUDA_OK( cudaMalloc( &m.w, m.n * sizeof( TK ) ) );
        CUDA_OK( cudaMemcpy( m.w, tmp.data(), m.n * sizeof( TK ), cudaMemcpyHostToDevice ) );
    }
    CUDA_OK( cudaMalloc( &m.ids, m.n * sizeof( int ) ) );
    CUDA_OK( cudaMemcpy( m.ids, arbre.order.data(), m.n * sizeof( int ), cudaMemcpyHostToDevice ) );
    CUDA_OK( cudaMalloc( &m.res, m.n * sizeof( double ) ) );
    CUDA_OK( cudaMalloc( &m.deb, sizeof( int ) ) );
    CUDA_OK( cudaMalloc( &m.deb2, sizeof( int ) ) );
    CUDA_OK( cudaMalloc( &m.liste, m.n * sizeof( int ) ) );
    CUDA_OK( cudaDeviceSynchronize() );
    t_tele = now() - t0;
}

template<int D, class TK>
DiagrammeGpu<D,TK>::~DiagrammeGpu() {
    Impl &m = *impl;
    cudaFree( m.nodes );
    for ( int d = 0; d < D; ++d ) cudaFree( m.c[ d ] );
    cudaFree( m.w ); cudaFree( m.ids ); cudaFree( m.res ); cudaFree( m.deb ); cudaFree( m.deb2 ); cudaFree( m.liste );
    delete impl;
}

namespace {

/// LE LANCEMENT d'un noyau deja choisi : `lance()` fait tout ( une ou deux passes ) et rend le
/// compteur de debordement qui fait foi.
template<int D, class TK, class Impl>
Chrono chrono( const Impl &m, int reps, std::vector<double> &res, auto &&lance ) {
    Chrono ch;
    cudaEvent_t e0, e1;
    CUDA_OK( cudaEventCreate( &e0 ) );
    CUDA_OK( cudaEventCreate( &e1 ) );
    ch.noyau = 1e300;
    int *deb = m.deb;
    for ( int r = -1; r < reps; ++r ) {                  // `-1` : la chauffe
        CUDA_OK( cudaMemset( m.deb, 0, sizeof( int ) ) );
        CUDA_OK( cudaMemset( m.deb2, 0, sizeof( int ) ) );
        CUDA_OK( cudaEventRecord( e0 ) );
        deb = lance();
        CUDA_OK( cudaEventRecord( e1 ) );
        CUDA_OK( cudaGetLastError() );
        CUDA_OK( cudaEventSynchronize( e1 ) );
        float ms = 0;
        CUDA_OK( cudaEventElapsedTime( &ms, e0, e1 ) );
        if ( r >= 0 && ms * 1e-3 < ch.noyau ) ch.noyau = ms * 1e-3;
    }
    const double t0 = now();
    res.resize( m.n );
    CUDA_OK( cudaMemcpy( res.data(), m.res, m.n * sizeof( double ), cudaMemcpyDeviceToHost ) );
    CUDA_OK( cudaMemcpy( &ch.deborde, deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
    ch.retour = now() - t0;
    cudaEventDestroy( e0 ); cudaEventDestroy( e1 );
    return ch;
}

template<class TK, bool POIDS, int MaxNb, class Impl>
Chrono lance2( const Impl &m, Variante v, int reps, std::vector<double> &res ) {
    const Arbre<TK,2> ar = m.arbre();
    if ( v != Variante::FIL ) {
        auto voies = [ & ]( auto vv ) {
            constexpr int V = decltype( vv )::value;
            const int cellules_par_bloc = BLOC2 / V;
            const int grid = ( m.n + cellules_par_bloc - 1 ) / cellules_par_bloc;
            return chrono<2,TK>( m, reps, res, [ & ]() { noyau2_voies<POIDS,MaxNb,V><<<grid, BLOC2>>>( ar, m.res, m.deb ); return m.deb; } );
        };
        if ( v == Variante::VOIES16 ) return voies( std::integral_constant<int,16>{} );
        if ( v == Variante::VOIES32 ) return voies( std::integral_constant<int,32>{} );
        return voies( std::integral_constant<int,8>{} );
    }
    const int bloc = 128, grid = ( m.n + bloc - 1 ) / bloc;
    return chrono<2,TK>( m, reps, res, [ & ]() { noyau2_fil<POIDS,MaxNb><<<grid, bloc>>>( ar, m.res, m.deb ); return m.deb; } );
}

template<class TK, bool POIDS, int MaxNv, class Impl>
Chrono lance3( const Impl &m, Variante v, int reps, std::vector<double> &res ) {
    const Arbre<TK,3> ar = m.arbre();
    const int bloc = 128, grid = ( m.n + bloc - 1 ) / bloc;
    if ( v != Variante::FIL ) {
        // une cellule par warp, EN DEUX PASSES : deux cases par voie ( 64 sommets ) pour toutes,
        // puis les rangs qui ont deborde ( 0.02 % en uniforme ) rejoues a `MaxNv / 32` cases
        constexpr int S2 = MaxNv / 32 > 2 ? MaxNv / 32 : 4;
        const int grid3 = ( m.n + BLOC3 / 32 - 1 ) / ( BLOC3 / 32 );
        return chrono<3,TK>( m, reps, res, [ & ]() {
            noyau3_voies<POIDS,2><<<grid3, BLOC3>>>( ar, m.res, m.deb, nullptr, 0, m.liste );
            int nd = 0;
            CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
            if ( nd == 0 ) return m.deb;
            noyau3_voies<POIDS,S2><<<( nd + BLOC3 / 32 - 1 ) / ( BLOC3 / 32 ), BLOC3>>>( ar, m.res, m.deb2, m.liste, nd, nullptr );
            return m.deb2;
        } );
    }
    return chrono<3,TK>( m, reps, res, [ & ]() { noyau3_fil<POIDS,MaxNv,MaxNv><<<grid, bloc>>>( ar, m.res, m.deb ); return m.deb; } );
}

} // namespace

template<int D, class TK>
Chrono DiagrammeGpu<D,TK>::mesures( Variante v, int maxnv, int reps, std::vector<double> &res ) const {
    const Impl &m = *impl;
    if constexpr ( D == 2 ) {
        if ( m.poids ) return maxnv > 64 ? lance2<TK,true,128>( m, v, reps, res ) : lance2<TK,true,64>( m, v, reps, res );
        else           return maxnv > 64 ? lance2<TK,false,128>( m, v, reps, res ) : lance2<TK,false,64>( m, v, reps, res );
    } else {
        if ( m.poids ) return maxnv > 64 ? lance3<TK,true,128>( m, v, reps, res ) : lance3<TK,true,64>( m, v, reps, res );
        else           return maxnv > 64 ? lance3<TK,false,128>( m, v, reps, res ) : lance3<TK,false,64>( m, v, reps, res );
    }
}

std::string carte() {
    int dev = 0;
    cudaDeviceProp p;
    if ( cudaGetDevice( &dev ) != cudaSuccess || cudaGetDeviceProperties( &p, dev ) != cudaSuccess ) return "( pas de GPU )";
    char buf[ 256 ];
    std::snprintf( buf, sizeof buf, "%s, sm_%d%d, %d SM, L2 %d Mo", p.name, p.major, p.minor, p.multiProcessorCount, int( p.l2CacheSize >> 20 ) );
    return buf;
}

template struct DiagrammeGpu<2,float>;
template struct DiagrammeGpu<2,double>;
template struct DiagrammeGpu<3,float>;
template struct DiagrammeGpu<3,double>;

} // namespace sf::gpu
