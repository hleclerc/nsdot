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
#include "gpu/Bsp2D.cuh"
#include "gpu/Hess2D.cuh"
#include "gpu/Fil2D.cuh"
#include "gpu/Voies2D.cuh"
#include "gpu/Paquet2D.cuh"
#include "gpu/FilReg2D.cuh"
#include "gpu/FilMix2D.cuh"
#include "gpu/FilBrk2D.cuh"
#include "gpu/FilRot2D.cuh"
#include "gpu/FilNrm2D.cuh"
#include "gpu/FilOrd2D.cuh"
#include "gpu/FilSuc2D.cuh"
#include "gpu/FilMsk2D.cuh"
#include "gpu/FilUni2D.cuh"
#include "gpu/FilShm2D.cuh"
#include "gpu/FilPh2D.cuh"
#include "gpu/Fil3D.cuh"
#include "gpu/Voies3D.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <algorithm>
#include <numeric>

namespace sf::gpu {

#define CUDA_OK( x ) do { cudaError_t e_ = ( x ); if ( e_ != cudaSuccess ) { \
    std::fprintf( stderr, "CUDA : %s ( %s:%d )\n", cudaGetErrorString( e_ ), __FILE__, __LINE__ ); std::exit( 2 ); } } while ( 0 )

template<int D, class TK>
struct DiagrammeGpu<D,TK>::Impl {
    Noeud<TK,D> *nodes = nullptr;
    TK          *c[ D ] = {};
    int         *u[ D ] = {};                        ///< les memes en virgule fixe 32 bits
    long long   *u64[ D ] = {};                      ///< et en virgule fixe 64 bits
    TK          *w = nullptr;
    int         *ids = nullptr;
    double      *res = nullptr;
    int         *deb = nullptr, *deb2 = nullptr;
    unsigned long long *stats = nullptr;
    int         *cptr = nullptr;                     ///< le compteur des lanes persistantes             ///< 4 compteurs, pour les noyaux qui comptent     ///< les compteurs de debordement des deux passes
    int         *liste = nullptr;                    ///< les rangs qui ont deborde a la premiere passe
    void        *sortie = nullptr;                   ///< `SortieBsp<TK>*` quand l'arbre vient du GPU
    int         *fac_j = nullptr;                    ///< les facettes, allouees une fois pour toutes
    TK          *fac_l = nullptr;
    int         *hrow = nullptr, *hcol = nullptr;    ///< la hessienne, idem
    double      *hval = nullptr, *hdia = nullptr;
    double      *pid[ 2 ] = {};                      ///< positions dans l'ordre DE L'APPELANT
    void        *hscan = nullptr;                    ///< le tampon du scan CUB
    size_t       hscan_o = 0;
    int          hcap = 0;                           ///< places allouees pour `col` / `val`
    int          n = 0, nn = 0;
    bool         poids = false;

    Arbre<TK,D> arbre() const {
        Arbre<TK,D> a;
        a.nodes = nodes;
        for ( int d = 0; d < D; ++d ) { a.c[ d ] = c[ d ]; a.u[ d ] = u[ d ]; a.u64[ d ] = u64[ d ]; }
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
    std::vector<int> fix( m.n );
    std::vector<long long> fix64( m.n );
    for ( int d = 0; d < D; ++d ) {
        for ( int k = 0; k < m.n; ++k ) {
            const double v = arbre.seed_c( k, d );
            tmp[ k ] = TK( v );
            // la virgule fixe : `[ 0, 1 ] -> [ 0, 2^30 ]`, arrondi au plus proche, borne
            const double f = std::round( v * double( ECH_FIXE ) );
            fix[ k ] = int( std::min( std::max( f, 0.0 ), double( ECH_FIXE ) ) );
            fix64[ k ] = ( long long ) std::llround( std::min( std::max( v, 0.0 ), 1.0 ) * double( ECH_F64 ) );
        }
        CUDA_OK( cudaMalloc( &m.c[ d ], m.n * sizeof( TK ) ) );
        CUDA_OK( cudaMemcpy( m.c[ d ], tmp.data(), m.n * sizeof( TK ), cudaMemcpyHostToDevice ) );
        CUDA_OK( cudaMalloc( &m.u[ d ], m.n * sizeof( int ) ) );
        CUDA_OK( cudaMemcpy( m.u[ d ], fix.data(), m.n * sizeof( int ), cudaMemcpyHostToDevice ) );
        CUDA_OK( cudaMalloc( &m.u64[ d ], m.n * sizeof( long long ) ) );
        CUDA_OK( cudaMemcpy( m.u64[ d ], fix64.data(), m.n * sizeof( long long ), cudaMemcpyHostToDevice ) );
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
    CUDA_OK( cudaMalloc( &m.stats, 4 * sizeof( unsigned long long ) ) );
    CUDA_OK( cudaMalloc( &m.cptr, sizeof( int ) ) );
    CUDA_OK( cudaMalloc( &m.liste, m.n * sizeof( int ) ) );
    CUDA_OK( cudaDeviceSynchronize() );
    t_tele = now() - t0;
}

/// L'ARBRE SUR LE GPU : la construction remplit directement les tampons de l'implementation, et
/// rien ne repasse par l'hote ( ni noeuds, ni permutation, ni positions ).
template<int D, class TK>
DiagrammeGpu<D,TK>::DiagrammeGpu( const double *const *P, const double *W, int n, int leaf, double *ms,
                                  bool pour_newton )
    : impl( new Impl ) {
    const double t0 = now();
    Impl &m = *impl;
    m.n = n;
    m.poids = W != nullptr;
    if constexpr ( D == 2 ) {
        SortieBsp<TK> s;
        construit2_dev<TK>( P[ 0 ], P[ 1 ], W, n, leaf, s, ms, pour_newton );
        m.nodes = s.nodes; m.nn = s.nn; m.ids = s.ids; m.w = s.w;
        for ( int d = 0; d < 2; ++d ) { m.c[ d ] = s.c[ d ]; m.u[ d ] = s.u[ d ]; m.u64[ d ] = s.u64[ d ]; }
        m.sortie = new SortieBsp<TK>( s );
        nn_pub = s.nn;
    } else {
        std::fprintf( stderr, "l'arbre sur GPU n'est fait qu'en 2D\n" );
        std::exit( 2 );
    }
    CUDA_OK( cudaMalloc( &m.res, size_t( n ) * sizeof( double ) ) );
    CUDA_OK( cudaMalloc( &m.deb, sizeof( int ) ) );
    CUDA_OK( cudaMalloc( &m.deb2, sizeof( int ) ) );
    CUDA_OK( cudaMalloc( &m.stats, 4 * sizeof( unsigned long long ) ) );
    CUDA_OK( cudaMalloc( &m.cptr, sizeof( int ) ) );
    CUDA_OK( cudaMalloc( &m.liste, size_t( n ) * sizeof( int ) ) );
    CUDA_OK( cudaDeviceSynchronize() );
    t_tele = now() - t0;
}

template<int D, class TK>
DiagrammeGpu<D,TK>::~DiagrammeGpu() {
    Impl &m = *impl;
    cudaFree( m.nodes );
    for ( int d = 0; d < D; ++d ) { cudaFree( m.c[ d ] ); cudaFree( m.u[ d ] ); cudaFree( m.u64[ d ] ); }
    cudaFree( m.w ); cudaFree( m.ids ); cudaFree( m.res ); cudaFree( m.deb ); cudaFree( m.deb2 ); cudaFree( m.liste ); cudaFree( m.stats ); cudaFree( m.cptr );
    cudaFree( m.fac_j ); cudaFree( m.fac_l );
    cudaFree( m.hrow ); cudaFree( m.hcol ); cudaFree( m.hval ); cudaFree( m.hdia ); cudaFree( m.hscan );
    for ( int d = 0; d < 2; ++d ) cudaFree( m.pid[ d ] );
    if ( m.sortie ) { libere_atelier<TK>( *( SortieBsp<TK> * ) m.sortie ); delete ( SortieBsp<TK> * ) m.sortie; }
    delete impl;
}

namespace {

/// CE QUE LE NOYAU COUTE EN REGISTRES, et combien de blocs le SM en loge. La question n'est pas
/// oiseuse : la grille est toujours largement plus grande que la carte ( un thread par cellule,
/// des milliers de blocs pour 68 SM ), donc ce qui limite le nombre de threads EN VOL est
/// uniquement l'occupation -- les registres par thread, arrondis par l'unite d'allocation.
template<class F>
void infos( Chrono &ch, F noy, int bloc ) {
    cudaFuncAttributes at;
    if ( cudaFuncGetAttributes( &at, noy ) != cudaSuccess ) return;
    int par_sm = 0, dev = 0;
    cudaGetDevice( &dev );
    cudaOccupancyMaxActiveBlocksPerMultiprocessor( &par_sm, noy, bloc, 0 );
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, dev );
    ch.regs = at.numRegs;
    ch.blocs = par_sm;
    ch.local = int( at.localSizeBytes );
    ch.occup = double( par_sm * bloc ) / prop.maxThreadsPerMultiProcessor;
}

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
    // LA CHAUFFE : 300 ms de noyau avant le premier chrono. Le CPU ( l'arbre, le temoin ) laisse le
    // GPU redescendre en frequence, et un seul tour ne le remonte pas ( mesure : 17 puis 13 ns
    // par germe pour le meme noyau, selon qu'il passait premier ou second )
    for ( const double t0 = now(); now() - t0 < 0.3; ) {
        CUDA_OK( cudaMemset( m.deb, 0, sizeof( int ) ) );
        CUDA_OK( cudaMemset( m.deb2, 0, sizeof( int ) ) );
        lance();
        CUDA_OK( cudaDeviceSynchronize() );
    }
    for ( int r = 0; r < reps; ++r ) {
        CUDA_OK( cudaMemset( m.deb, 0, sizeof( int ) ) );
        CUDA_OK( cudaMemset( m.deb2, 0, sizeof( int ) ) );
        CUDA_OK( cudaMemset( m.res, 0xff, m.n * sizeof( double ) ) );   // NaN : une cellule non ecrite se voit
        CUDA_OK( cudaMemset( m.stats, 0, 4 * sizeof( unsigned long long ) ) );
        CUDA_OK( cudaEventRecord( e0 ) );
        deb = lance();
        CUDA_OK( cudaEventRecord( e1 ) );
        CUDA_OK( cudaGetLastError() );
        CUDA_OK( cudaEventSynchronize( e1 ) );
        float ms = 0;
        CUDA_OK( cudaEventElapsedTime( &ms, e0, e1 ) );
        if ( ms * 1e-3 < ch.noyau ) ch.noyau = ms * 1e-3;
    }
    const double t0 = now();
    res.resize( m.n );
    CUDA_OK( cudaMemcpy( res.data(), m.res, m.n * sizeof( double ), cudaMemcpyDeviceToHost ) );
    CUDA_OK( cudaMemcpy( &ch.deborde, deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
    CUDA_OK( cudaMemcpy( ch.stats, m.stats, 4 * sizeof( unsigned long long ), cudaMemcpyDeviceToHost ) );
    ch.retour = now() - t0;
    cudaEventDestroy( e0 ); cudaEventDestroy( e1 );
    return ch;
}

template<class TK, bool POIDS, int MaxNb, class Impl>
Chrono lance2( const Impl &m, Variante v, int reps, std::vector<double> &res ) {
    const Arbre<TK,2> ar = m.arbre();
    if ( paquet( v ) ) {
        auto paq = [ & ]( auto vv, auto kk, auto sib ) {
            constexpr int V = decltype( vv )::value, K = decltype( kk )::value;
            constexpr bool SIB = decltype( sib )::value;
            constexpr int cellules_par_bloc = 4 * ( 32 / V ) * K;
            const int grid = ( m.n + cellules_par_bloc - 1 ) / cellules_par_bloc;
            return chrono<2,TK>( m, reps, res, [ & ]() { noyau2_paquet<POIDS,MaxNb,V,K,SIB><<<grid, 128>>>( ar, m.res, m.deb, m.stats ); return m.deb; } );
        };
        using I1 = std::integral_constant<int,1>; using I2 = std::integral_constant<int,2>; using I4 = std::integral_constant<int,4>;
        using V8 = std::integral_constant<int,8>; using V32 = std::integral_constant<int,32>;
        using F = std::false_type; using T = std::true_type;
        switch ( v ) {
            case Variante::PAQ8x1:   return paq( V8{},  I1{}, F{} );
            case Variante::PAQ8x2:   return paq( V8{},  I2{}, F{} );
            case Variante::PAQ8x4:   return paq( V8{},  I4{}, F{} );
            case Variante::PAQ32x1:  return paq( V32{}, I1{}, F{} );
            case Variante::PAQ32x2:  return paq( V32{}, I2{}, F{} );
            case Variante::PAQ32x4:  return paq( V32{}, I4{}, F{} );
            case Variante::PAQ8x1S:  return paq( V8{},  I1{}, T{} );
            case Variante::PAQ32x1S: return paq( V32{}, I1{}, T{} );
            default:                 return paq( V32{}, I4{}, T{} );
        }
    }
    if ( v >= Variante::VOIES ) {
        auto voies = [ & ]( auto vv ) {
            constexpr int V = decltype( vv )::value;
            const int cellules_par_bloc = BLOC2 / V;
            const int grid = ( m.n + cellules_par_bloc - 1 ) / cellules_par_bloc;
            Chrono ch = chrono<2,TK>( m, reps, res, [ & ]() { noyau2_voies<POIDS,MaxNb,V><<<grid, BLOC2>>>( ar, m.res, m.deb, m.stats ); return m.deb; } );
            infos( ch, noyau2_voies<POIDS,MaxNb,V,TK>, BLOC2 );
            return ch;
        };
        if ( v == Variante::VOIES16 ) return voies( std::integral_constant<int,16>{} );
        if ( v == Variante::VOIES32 ) return voies( std::integral_constant<int,32>{} );
        return voies( std::integral_constant<int,8>{} );
    }
    const int bloc = 128, grid = ( m.n + bloc - 1 ) / bloc;
    if ( v == Variante::FILREG )
        return chrono<2,TK>( m, reps, res, [ & ]() { noyau2_filreg<POIDS,MaxNb,false><<<grid, bloc>>>( ar, m.res, m.deb ); return m.deb; } );
    if ( v >= Variante::FILMIX4 && v <= Variante::FILMIX16 ) {
        auto mix = [ & ]( auto rr ) {
            constexpr int R = decltype( rr )::value;
            return chrono<2,TK>( m, reps, res, [ & ]() { noyau2_filmix<POIDS,( MaxNb > 64 ? 64 : MaxNb ),R,true><<<grid, bloc>>>( ar, m.res, m.deb ); return m.deb; } );
        };
        switch ( v ) {
            case Variante::FILMIX4:  return mix( std::integral_constant<int,4>{} );
            case Variante::FILMIX6:  return mix( std::integral_constant<int,6>{} );
            case Variante::FILMIX8:  return mix( std::integral_constant<int,8>{} );
            case Variante::FILMIX12: return mix( std::integral_constant<int,12>{} );
            default:                 return mix( std::integral_constant<int,16>{} );
        }
    }
    if ( v >= Variante::FILPH8 && v <= Variante::FILPH8M4 ) {
        // LE NOYAU PERSISTANT PAR SM : autant de blocs que la carte en loge, `CAP` cellules en vol
        // par bloc, l'etat en RAM
        constexpr int CAP = 512, BLPH = 128;
        auto noy = v == Variante::FILPH8G  ? noyau2_filph<POIDS,8,CAP,BLPH,1,64,0,3,TK>
                 : v == Variante::FILPH8B  ? noyau2_filph<POIDS,8,CAP,BLPH,2,1,0,3,TK>
                 : v == Variante::FILPH8A  ? noyau2_filph<POIDS,8,CAP,BLPH,3,1,0,3,TK>
                 : v == Variante::FILPH8C  ? noyau2_filph<POIDS,8,CAP,BLPH,4,1,0,3,TK>
                 : v == Variante::FILPH8O  ? noyau2_filph<POIDS,8,CAP,BLPH,4,1,1,3,TK>
                 : v == Variante::FILPH8M  ? noyau2_filph<POIDS,8,CAP,BLPH,4,1,2,3,TK>   // l'arene compactee + la coupe de `filmsk`
                 : v == Variante::FILPH8M4 ? noyau2_filph<POIDS,8,CAP,BLPH,4,1,2,4,TK>   // idem, force a quatre blocs par SM
                 : noyau2_filph<POIDS,8,CAP,BLPH,0,1,0,3,TK>;
        int par_sm = 0, dev = 0;
        CUDA_OK( cudaGetDevice( &dev ) );
        CUDA_OK( cudaOccupancyMaxActiveBlocksPerMultiprocessor( &par_sm, noy, BLPH, 0 ) );
        cudaDeviceProp prop;
        CUDA_OK( cudaGetDeviceProperties( &prop, dev ) );
        const int grid_p = std::max( 1, par_sm ) * prop.multiProcessorCount;
        EtatPh<TK> st;
        st.S = grid_p * CAP;
        CUDA_OK( cudaMalloc( &st.x, size_t( st.S ) * 8 * sizeof( TK ) ) );
        CUDA_OK( cudaMalloc( &st.y, size_t( st.S ) * 8 * sizeof( TK ) ) );
        CUDA_OK( cudaMalloc( &st.c, size_t( st.S ) * 8 * sizeof( int ) ) );
        CUDA_OK( cudaMalloc( &st.pile, size_t( st.S ) * PILE_PH * sizeof( int ) ) );
        CUDA_OK( cudaMalloc( &st.meta, size_t( st.S ) * META_PH * sizeof( int ) ) );
        Chrono ch = chrono<2,TK>( m, reps, res, [ & ]() {
            CUDA_OK( cudaMemset( m.cptr, 0, sizeof( int ) ) );
            noy<<<grid_p, BLPH>>>( ar, m.res, m.deb, m.liste, m.cptr, st, m.stats );
            int nd = 0;
            CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
            if ( nd == 0 ) return m.deb;
            noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
            return m.deb2;
        } );
        cudaFree( st.x ); cudaFree( st.y ); cudaFree( st.c ); cudaFree( st.pile ); cudaFree( st.meta );
        infos( ch, noy, BLPH );
        return ch;
    }
    if ( v == Variante::FILNRM8TRI || v == Variante::FILNRM8TRIL ) {
        // L'ORACLE DE L'HOMOGENEITE : un premier tour compte le cout de chaque cellule ( plans et
        // boites testes ), puis les cellules sont TRIEES PAR COUT ( `tri` : globalement ; `tril` :
        // dans des tranches de 4096 rangs, pour garder la localite ) et le noyau tourne dans cet
        // ordre -- chaque warp recoit 32 cellules de cout semblable. C'est ce qu'une file par
        // phases obtiendrait au mieux sur ce point, sans rien construire.
        int *cout = nullptr;
        CUDA_OK( cudaMalloc( &cout, m.n * sizeof( int ) ) );
        CUDA_OK( cudaMemset( m.deb, 0, sizeof( int ) ) );
        noyau2_filnrm<POIDS,8><<<grid, bloc>>>( ar, m.res, m.deb, m.liste, nullptr, 0, cout );
        std::vector<int> hc( m.n ), ordre( m.n );
        CUDA_OK( cudaMemcpy( hc.data(), cout, m.n * sizeof( int ), cudaMemcpyDeviceToHost ) );
        if ( std::getenv( "MESURES_DEBUG" ) ) {
            double sf = 0, sc = 0; int mf = 0;
            std::vector<int> hf( m.n );
            for ( int i = 0; i < m.n; ++i ) { hf[ i ] = hc[ i ] >> 20; sf += hf[ i ]; sc += hc[ i ] & 0xfffff; mf = std::max( mf, hf[ i ] ); }
            std::sort( hf.begin(), hf.end() );
            std::fprintf( stderr, "        feuilles par cellule : moyenne %.2f, mediane %d, p99 %d, max %d ; cout moyen %.1f\n",
                          sf / m.n, hf[ m.n / 2 ], hf[ m.n * 99 / 100 ], mf, sc / m.n );
        }
        for ( int i = 0; i < m.n; ++i ) hc[ i ] &= 0xfffff;
        std::iota( ordre.begin(), ordre.end(), 0 );
        const int tranche = v == Variante::FILNRM8TRI ? m.n : 4096;
        for ( int b = 0; b < m.n; b += tranche )
            std::sort( ordre.begin() + b, ordre.begin() + std::min( b + tranche, m.n ), [ & ]( int a, int c ) { return hc[ a ] < hc[ c ]; } );
        int *dordre = nullptr;
        CUDA_OK( cudaMalloc( &dordre, m.n * sizeof( int ) ) );
        CUDA_OK( cudaMemcpy( dordre, ordre.data(), m.n * sizeof( int ), cudaMemcpyHostToDevice ) );
        const Chrono ch = chrono<2,TK>( m, reps, res, [ & ]() {
            noyau2_filnrm<POIDS,8><<<grid, bloc>>>( ar, m.res, m.deb, m.liste, dordre, m.n, nullptr );
            int nd = 0;
            CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
            if ( nd == 0 ) return m.deb;
            noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
            return m.deb2;
        } );
        cudaFree( cout ); cudaFree( dordre );
        return ch;
    }
    if ( v == Variante::FILSHM8 ) {
        // ( le carveout a 100 % de memoire partagee a ete essaye : il prend le L1 -- la pile, les
        // noeuds -- pour une occupation que 12 Ko par bloc ne remontent pas : 8.3 au lieu de 8.0 )
        return chrono<2,TK>( m, reps, res, [ & ]() {
            noyau2_filshm<POIDS,8><<<grid, bloc>>>( ar, m.res, m.deb, m.liste );
            int nd = 0;
            CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
            if ( nd == 0 ) return m.deb;
            noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
            return m.deb2;
        } );
    }
    if ( v == Variante::FILUNI8NP )                       // la boucle unique, un thread par cellule
        return chrono<2,TK>( m, reps, res, [ & ]() {
            noyau2_filuni<POIDS,8,false><<<grid, bloc>>>( ar, m.res, m.deb, m.liste, m.cptr );
            int nd = 0;
            CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
            if ( nd == 0 ) return m.deb;
            noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
            return m.deb2;
        } );
    if ( v == Variante::FILUNI8 ) {
        // les lanes persistantes : autant de threads que la carte en loge, chacun prend des cellules
        int par_sm = 0, dev = 0;
        CUDA_OK( cudaGetDevice( &dev ) );
        CUDA_OK( cudaOccupancyMaxActiveBlocksPerMultiprocessor( &par_sm, noyau2_filuni<POIDS,8,true,TK>, bloc, 0 ) );
        cudaDeviceProp prop;
        CUDA_OK( cudaGetDeviceProperties( &prop, dev ) );
        const int grid_p = par_sm * prop.multiProcessorCount;
        return chrono<2,TK>( m, reps, res, [ & ]() {
            CUDA_OK( cudaMemset( m.cptr, 0, sizeof( int ) ) );
            noyau2_filuni<POIDS,8,true><<<grid_p, bloc>>>( ar, m.res, m.deb, m.liste, m.cptr );
            int nd = 0;
            CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
            if ( nd == 0 ) return m.deb;
            noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
            return m.deb2;
        } );
    }
    if ( v == Variante::FILMSK8 || v == Variante::FILMSK8G || v == Variante::FILMSK8F || v == Variante::FILMSK8H || v == Variante::FILMSK8C6 || v == Variante::FILMSK8C8 ) {
        // `BSM` : le nombre de blocs par SM que ptxas doit garantir -- il rabote les registres
        // pour y arriver. A 128 threads par bloc sur Turing ( 64 Ko de registres, 32 warps ) :
        // 4 blocs <=> 128 registres, 5 <=> 102, 6 <=> 85, 8 <=> 64 et l'occupation pleine
        auto msk = [ & ]( auto mm, auto gg, auto ff ) {
            constexpr int BSM = decltype( mm )::value;
            constexpr bool CENTRE = decltype( gg )::value;
            constexpr int FIXE = decltype( ff )::value;
            Chrono ch = chrono<2,TK>( m, reps, res, [ & ]() {
                noyau2_filmsk<POIDS,BSM,CENTRE,FIXE><<<grid, bloc>>>( ar, m.res, m.deb, m.liste );
                int nd = 0;
                CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
                if ( nd == 0 ) return m.deb;
                noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
                return m.deb2;
            } );
            infos( ch, noyau2_filmsk<POIDS,BSM,CENTRE,FIXE,TK>, bloc );
            return ch;
        };
        using I1 = std::integral_constant<int,1>;
        using F = std::false_type; using T = std::true_type;
        using N0 = std::integral_constant<int,0>; using N32 = std::integral_constant<int,32>; using N64 = std::integral_constant<int,64>;
        return v == Variante::FILMSK8   ? msk( I1{}, F{}, N0{} )
             : v == Variante::FILMSK8G  ? msk( I1{}, T{}, N0{} )   // le repere centre sur le germe
             : v == Variante::FILMSK8F  ? msk( I1{}, T{}, N32{} )  // + les positions en virgule fixe 32 bits
             : v == Variante::FILMSK8H  ? msk( I1{}, T{}, N64{} )  // + en virgule fixe 64 bits
             : v == Variante::FILMSK8C6 ? msk( std::integral_constant<int,6>{}, F{}, N0{} )
             :                            msk( std::integral_constant<int,8>{}, F{}, N0{} );
    }
    if ( v == Variante::FILNRM8C6 || v == Variante::FILNRM8C8 ) {
        auto nrm = [ & ]( auto mm ) {
            constexpr int BSM = decltype( mm )::value;
            Chrono ch = chrono<2,TK>( m, reps, res, [ & ]() {
                noyau2_filnrm<POIDS,8,BSM><<<grid, bloc>>>( ar, m.res, m.deb, m.liste );
                int nd = 0;
                CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
                if ( nd == 0 ) return m.deb;
                noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
                return m.deb2;
            } );
            infos( ch, noyau2_filnrm<POIDS,8,BSM,TK>, bloc );
            return ch;
        };
        return v == Variante::FILNRM8C6 ? nrm( std::integral_constant<int,6>{} ) : nrm( std::integral_constant<int,8>{} );
    }
    if ( v == Variante::FILSUC8 )
        return chrono<2,TK>( m, reps, res, [ & ]() {
            noyau2_filsuc<POIDS><<<grid, bloc>>>( ar, m.res, m.deb, m.liste );
            int nd = 0;
            CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
            if ( nd == 0 ) return m.deb;
            noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
            return m.deb2;
        } );
    if ( v == Variante::FILORD8 )
        return chrono<2,TK>( m, reps, res, [ & ]() {
            noyau2_filord<POIDS><<<grid, bloc>>>( ar, m.res, m.deb, m.liste );
            int nd = 0;
            CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
            if ( nd == 0 ) return m.deb;
            noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
            return m.deb2;
        } );
    if ( v == Variante::FILNRM8 ) {
        Chrono ch = chrono<2,TK>( m, reps, res, [ & ]() {
            noyau2_filnrm<POIDS,8><<<grid, bloc>>>( ar, m.res, m.deb, m.liste );
            int nd = 0;
            CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
            if ( nd == 0 ) return m.deb;
            noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
            return m.deb2;
        } );
        infos( ch, noyau2_filnrm<POIDS,8,1,TK>, bloc );
        return ch;
    }
    if ( v == Variante::FILROT6 || v == Variante::FILROT8 ) {
        auto rot = [ & ]( auto rr ) {
            constexpr int R = decltype( rr )::value;
            return chrono<2,TK>( m, reps, res, [ & ]() {
                noyau2_filrot<POIDS,R><<<grid, bloc>>>( ar, m.res, m.deb, m.liste );
                int nd = 0;
                CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
                if ( nd == 0 ) return m.deb;
                noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
                return m.deb2;
            } );
        };
        return v == Variante::FILROT6 ? rot( std::integral_constant<int,6>{} ) : rot( std::integral_constant<int,8>{} );
    }
    if ( v >= Variante::FILBRK6 && v <= Variante::FILBRK8NU ) {
        // tout en registres avec des sorties, puis les rangs qui ont deborde `R` refaits par
        // `filmix` a 8 registres et 64 sommets
        auto brk = [ & ]( auto rr, auto oo ) {
            constexpr int R = decltype( rr )::value;
            constexpr bool OPT = decltype( oo )::value;
            return chrono<2,TK>( m, reps, res, [ & ]() {
                noyau2_filbrk<POIDS,R,OPT><<<grid, bloc>>>( ar, m.res, m.deb, m.liste );
                int nd = 0;
                CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
                if ( std::getenv( "MESURES_DEBUG" ) ) std::fprintf( stderr, "        seconde passe : %d cellules\n", nd );
                if ( nd == 0 ) return m.deb;
                noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd );
                return m.deb2;
            } );
        };
        switch ( v ) {
            case Variante::FILBRK6:  return brk( std::integral_constant<int,6>{},  std::true_type{} );
            case Variante::FILBRK8:  return brk( std::integral_constant<int,8>{},  std::true_type{} );
            case Variante::FILBRK10: return brk( std::integral_constant<int,10>{}, std::true_type{} );
            case Variante::FILBRK12: return brk( std::integral_constant<int,12>{}, std::true_type{} );
            case Variante::FILBRK16: return brk( std::integral_constant<int,16>{}, std::true_type{} );
            default:                 return brk( std::integral_constant<int,8>{},  std::false_type{} );   // `filbrk8nu` : sans les micro-optimisations
        }
    }
    if ( v == Variante::FILREGC )
        return chrono<2,TK>( m, reps, res, [ & ]() { noyau2_filreg<POIDS,MaxNb,true><<<grid, bloc>>>( ar, m.res, m.deb ); return m.deb; } );
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

/// DES POIDS NEUFS sur le meme arbre : que les majorants, et pas une allocation.
template<int D, class TK>
double DiagrammeGpu<D,TK>::refresh_poids( const double *W ) {
    if constexpr ( D != 2 ) { ( void ) W; return 0; }
    else {
        SortieBsp<TK> *s = ( SortieBsp<TK> * ) impl->sortie;
        if ( ! s || ! s->at.cru ) { std::fprintf( stderr, "refresh_poids demande `pour_newton`\n" ); std::exit( 2 ); }
        impl->poids = true;
        impl->w = s->w;
        return rafraichit_poids_dev<TK>( *s, W );
    }
}

/// L'ASSEMBLAGE : compter, scanner, remplir.
template<int D, class TK>
double DiagrammeGpu<D,TK>::assemble( Hessienne &H ) {
    if constexpr ( D != 2 ) { ( void ) H; return 0; }
    else {
        Impl &m = *impl;
        const int BL = 256, gr = ( m.n + BL - 1 ) / BL;
        if ( ! m.hrow ) {
            CUDA_OK( cudaMalloc( &m.hrow, size_t( m.n + 1 ) * sizeof( int ) ) );
            CUDA_OK( cudaMalloc( &m.hdia, size_t( m.n ) * sizeof( double ) ) );
            for ( int d = 0; d < 2; ++d ) CUDA_OK( cudaMalloc( &m.pid[ d ], size_t( m.n ) * sizeof( double ) ) );
            SortieBsp<TK> *s = ( SortieBsp<TK> * ) m.sortie;
            if ( ! s || ! s->at.tx ) { std::fprintf( stderr, "assemble demande `pour_newton`\n" ); std::exit( 2 ); }
            k_hess_pos<<<gr, BL>>>( s->at.tx, s->at.ty, m.ids, m.pid[ 0 ], m.pid[ 1 ], m.n );
            CUDA_OK( cub::DeviceScan::ExclusiveSum( m.hscan, m.hscan_o, m.hrow, m.hrow, m.n + 1 ) );
            CUDA_OK( cudaMalloc( &m.hscan, m.hscan_o ) );
        }
        cudaEvent_t e0, e1;
        CUDA_OK( cudaEventCreate( &e0 ) ); CUDA_OK( cudaEventCreate( &e1 ) );
        CUDA_OK( cudaEventRecord( e0 ) );
        k_hess_compte<<<gr, BL>>>( m.fac_j, m.n, NF, m.hrow );
        CUDA_OK( cudaMemsetAsync( m.hrow + m.n, 0, sizeof( int ) ) );
        size_t o = m.hscan_o;
        CUDA_OK( cub::DeviceScan::ExclusiveSum( m.hscan, o, m.hrow, m.hrow, m.n + 1 ) );
        int nnz = 0;
        CUDA_OK( cudaMemcpy( &nnz, m.hrow + m.n, sizeof( int ), cudaMemcpyDeviceToHost ) );
        if ( nnz > m.hcap ) {                            // on ne retrecit jamais
            cudaFree( m.hcol ); cudaFree( m.hval );
            m.hcap = nnz + nnz / 8 + 1024;
            CUDA_OK( cudaMalloc( &m.hcol, size_t( m.hcap ) * sizeof( int ) ) );
            CUDA_OK( cudaMalloc( &m.hval, size_t( m.hcap ) * sizeof( double ) ) );
        }
        k_hess_remplit<TK><<<gr, BL>>>( m.fac_j, m.fac_l, m.pid[ 0 ], m.pid[ 1 ], m.hrow, m.n, NF,
                                        m.hcol, m.hval, m.hdia );
        CUDA_OK( cudaEventRecord( e1 ) );
        CUDA_OK( cudaEventSynchronize( e1 ) );
        float t = 0;
        CUDA_OK( cudaEventElapsedTime( &t, e0, e1 ) );
        CUDA_OK( cudaGetLastError() );
        cudaEventDestroy( e0 ); cudaEventDestroy( e1 );
        H.row = m.hrow; H.col = m.hcol; H.val = m.hval; H.dia = m.hdia; H.n = m.n; H.nnz = nnz;
        return t;
    }
}

template<int D, class TK>
void DiagrammeGpu<D,TK>::applique( const Hessienne &H, const double *x, double *y ) const {
    if constexpr ( D == 2 ) {
        const int BL = 256;
        k_hess_mul<<<( H.n + BL - 1 ) / BL, BL>>>( H.row, H.col, H.val, H.dia, x, y, H.n );
        CUDA_OK( cudaDeviceSynchronize() );
    }
}

/// UN TOUR DE NEWTON, de bout en bout sur la carte : poids, majorants, mesures, facettes.
template<int D, class TK>
double DiagrammeGpu<D,TK>::tour_newton( const double *W ) {
    if constexpr ( D != 2 ) { ( void ) W; return 0; }
    else {
        Impl &m = *impl;
        if ( ! m.fac_j ) {
            CUDA_OK( cudaMalloc( &m.fac_j, size_t( NF ) * m.n * sizeof( int ) ) );
            CUDA_OK( cudaMalloc( &m.fac_l, size_t( NF ) * m.n * sizeof( TK ) ) );
        }
        const Arbre<TK,2> ar = m.arbre();
        const int bloc = 128, grid = ( m.n + bloc - 1 ) / bloc;
        constexpr int FIX = sizeof( TK ) == 4 ? 32 : 0;
        cudaEvent_t e0, e1;
        CUDA_OK( cudaEventCreate( &e0 ) ); CUDA_OK( cudaEventCreate( &e1 ) );
        CUDA_OK( cudaEventRecord( e0 ) );
        if ( W ) {
            SortieBsp<TK> *s = ( SortieBsp<TK> * ) m.sortie;
            m.poids = true; m.w = s->w;
            rafraichit_poids_dev<TK>( *s, W );
        }
        CUDA_OK( cudaMemsetAsync( m.deb, 0, sizeof( int ) ) );
        CUDA_OK( cudaMemsetAsync( m.deb2, 0, sizeof( int ) ) );
        auto tour = [ & ]( auto pp ) {
            constexpr bool POIDS = decltype( pp )::value;
            noyau2_filmsk<POIDS,1,true,FIX><<<grid, bloc>>>( ar, m.res, m.deb, m.liste, m.fac_j, m.fac_l, NF );
            int nd = 0;
            CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
            if ( nd )
                noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd, m.fac_j, m.fac_l, NF, m.cptr );
        };
        if ( m.poids ) tour( std::true_type{} ); else tour( std::false_type{} );
        CUDA_OK( cudaEventRecord( e1 ) );
        CUDA_OK( cudaEventSynchronize( e1 ) );
        float t = 0;
        CUDA_OK( cudaEventElapsedTime( &t, e0, e1 ) );
        CUDA_OK( cudaGetLastError() );
        cudaEventDestroy( e0 ); cudaEventDestroy( e1 );
        return t;
    }
}

template<int D, class TK>
Chrono DiagrammeGpu<D,TK>::facettes( int reps, std::vector<double> &res, std::vector<int> &fj, std::vector<double> &fl ) const {
    if constexpr ( D != 2 ) {                            // les facettes ne sont rendues qu'en 2D
        ( void ) reps; ( void ) res; ( void ) fj; ( void ) fl;
        return Chrono{};
    } else {
    const Impl &m = *impl;
    const Arbre<TK,2> ar = m.arbre();
    const int bloc = 128, grid = ( m.n + bloc - 1 ) / bloc;
    // les MEMES tampons que `tour_newton` et `assemble` : alloues une fois pour toutes
    Impl &mm = *impl;
    if ( ! mm.fac_j ) {
        CUDA_OK( cudaMalloc( &mm.fac_j, size_t( NF ) * m.n * sizeof( int ) ) );
        CUDA_OK( cudaMalloc( &mm.fac_l, size_t( NF ) * m.n * sizeof( TK ) ) );
    }
    int *dj = mm.fac_j;
    TK  *dl = mm.fac_l;
    // la virgule fixe 32 bits est RESERVEE AU `float` : en `double` elle detruit la precision
    // ( 31 bits contre 53 ) -- voir `doc/04-echelle.md`
    constexpr int FIX = sizeof( TK ) == 4 ? 32 : 0;
    auto tour = [ & ]( auto pp ) {
        constexpr bool POIDS = decltype( pp )::value;
        Chrono ch = chrono<2,TK>( m, reps, res, [ & ]() {
            noyau2_filmsk<POIDS,1,true,FIX><<<grid, bloc>>>( ar, m.res, m.deb, m.liste, dj, dl, NF );
            int nd = 0;
            CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
            if ( nd == 0 ) return m.deb;
            // la seconde passe rend AUSSI ses facettes : elle finit 13 % des cellules en uniforme
            noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd, dj, dl, NF, m.cptr );
            return m.deb2;
        } );
        infos( ch, noyau2_filmsk<POIDS,1,true,FIX,TK>, bloc );
        return ch;
    };
    CUDA_OK( cudaMemset( m.cptr, 0, sizeof( int ) ) );   // les polygones finaux de plus de `NF` aretes
    Chrono ch = m.poids ? tour( std::true_type{} ) : tour( std::false_type{} );
    CUDA_OK( cudaMemcpy( &ch.deborde, m.cptr, sizeof( int ), cudaMemcpyDeviceToHost ) );
    const double t0 = now();
    fj.resize( size_t( NF ) * m.n );
    std::vector<TK> tmp( size_t( NF ) * m.n );
    CUDA_OK( cudaMemcpy( fj.data(), dj, fj.size() * sizeof( int ), cudaMemcpyDeviceToHost ) );
    CUDA_OK( cudaMemcpy( tmp.data(), dl, tmp.size() * sizeof( TK ), cudaMemcpyDeviceToHost ) );
    fl.assign( tmp.begin(), tmp.end() );
    ch.retour += now() - t0;
    return ch;
    }
}

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
