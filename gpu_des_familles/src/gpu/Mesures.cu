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
#include "gpu/Cg2D.cuh"
#include "gpu/Amg2D.cuh"
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
#include "gpu/Image2D.cuh"
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
#include <cstring>

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
    double      *cgv = nullptr;                      ///< `r, z, p, q` bout a bout, plus six scalaires
    std::vector<Niveau> niv;                         ///< la hierarchie du multigrille
    std::vector<int *>  map;                         ///< `map[ l ][ i ]` : le paquet du niveau `l + 1`
    int          kcycle = 2;                         ///< niveaux acceleres par Krylov ( 0 : V pur )
    int          gros = 120;                         ///< lissages au niveau le plus grossier
    int          nu = 2;                             ///< lissages avant et apres, par niveau
    int          stop = 1000;                        ///< on arrete de grossir en dessous
    int          lisse = 0;                          ///< la prolongation LISSEE ( `cusparseSpGEMM` )
    cusparseHandle_t cus = nullptr;
    int         *rang_de = nullptr;                  ///< identifiant -> rang ( l'agregation du niveau fin )
    double      *acc = nullptr;                      ///< un scalaire de travail
    int         *cond = nullptr, *ncond = nullptr;   ///< LES CELLULES CONDAMNEES : leur liste, leur compte
    int          cond_cap = 0;
    double2     *img = nullptr;                      ///< LA DENSITE IMAGE : ( somme prefixe, valeur )
    int          iw = 0, ih = 0;
    Densite      dens = Densite::DIRECTE;            ///< le mode, quand une image est chargee
    int          chunk = 1 << 20;                    ///< cellules par lot dans les modes a depot
    TK          *dep_x = nullptr, *dep_y = nullptr;  ///< le depot des polygones, en SoA
    int         *dep_nb = nullptr, *dep_id = nullptr;
    int          dep_cap = 0;
    int          n = 0, nn = 0;
    bool         poids = false;

    /// la grille telle que les noyaux la lisent ( inactive si aucune image n'est chargee )
    Image2 image() const {
        Image2 im;
        if ( ! img ) return im;
        im.p = img; im.W = iw; im.H = ih;
        im.hx = 1.0 / iw; im.hy = 1.0 / ih;
        im.ihx = double( iw ); im.ihy = double( ih );
        return im;
    }
    Densite mode() const { return img ? dens : Densite::AUCUNE; }

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
    cudaFree( m.img ); cudaFree( m.cond ); cudaFree( m.ncond );
    cudaFree( m.dep_x ); cudaFree( m.dep_y ); cudaFree( m.dep_nb ); cudaFree( m.dep_id );
    cudaFree( m.cgv );
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
            infos( ch, noyau2_filmsk<POIDS,BSM,CENTRE,FIXE,DENS_AUCUNE,TK>, bloc );
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

/// `|| m - cible ||`, la plus petite mesure, ET LE COMPTE DES CELLULES CONDAMNEES ( celles qui
/// tombent sous `seuil` ), avec leur liste -- le tout en un noyau, puisqu'il balaie deja tout.
///
/// C'est l'instrument qui manquait a la recherche lineaire. Jusqu'ici un diagramme d'essai ne
/// rendait qu'un scalaire : « la plus petite cellule vaut zero, donc je refuse ». Le compte dit
/// EN PLUS si l'obstruction est LOCALE ( trois cellules sur 10^5 : on les releve ) ou GLOBALE
/// ( le pas est vraiment trop grand ). Et la liste dit lesquelles, sans une passe de plus.
///
/// `ncond` peut depasser `cap` : on sait alors seulement qu'il y en a trop -- ce qui suffit.
__global__ void k_residu( const double *res, double cible, double *b, double *acc, int n,
                          double seuil, int *cond, int *ncond, int cap ) {
    __shared__ double ps[ 32 ], pm[ 32 ];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const double m = i < n ? res[ i ] : 0.0;
    double s = i < n ? ( m - cible ) * ( m - cible ) : 0.0;
    double mn = i < n ? m : 1e300;
    if ( b && i < n ) b[ i ] = m - cible;
    if ( ncond && i < n && m <= seuil ) {
        const int p = atomicAdd( ncond, 1 );
        if ( cond && p < cap ) cond[ p ] = i;
    }
    // les VIDES a part : une cellule sous le seuil se releve, une cellule nulle a deja disparu
    double vd = ncond && i < n && m <= 0 ? 1.0 : 0.0;
#pragma unroll
    for ( int d = 16; d; d >>= 1 ) vd += __shfl_down_sync( 0xffffffffu, vd, d );
    if ( ncond && ( threadIdx.x & 31 ) == 0 && vd ) atomicAdd( acc + 2, vd );
#pragma unroll
    for ( int d = 16; d; d >>= 1 ) { s += __shfl_down_sync( 0xffffffffu, s, d ); mn = fmin( mn, __shfl_down_sync( 0xffffffffu, mn, d ) ); }
    const int voie = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if ( voie == 0 ) { ps[ warp ] = s; pm[ warp ] = mn; }
    __syncthreads();
    if ( warp == 0 ) {
        const bool ok = voie < ( blockDim.x >> 5 );
        s = ok ? ps[ voie ] : 0.0;
        mn = ok ? pm[ voie ] : 1e300;
#pragma unroll
        for ( int d = 16; d; d >>= 1 ) { s += __shfl_down_sync( 0xffffffffu, s, d ); mn = fmin( mn, __shfl_down_sync( 0xffffffffu, mn, d ) ); }
        if ( voie == 0 ) { atomicAdd( acc, s ); atomicMin( ( unsigned long long * ) ( acc + 1 ), ordd( mn ) ); }
    }
}

template<int D, class TK>
double DiagrammeGpu<D,TK>::residu( double cible, double *mini, double *b, double seuil, int *nb_cond, int *nb_vides ) const {
    Impl &m = *impl;
    const int BL = 256, gr = ( m.n + BL - 1 ) / BL;
    if ( ! m.acc ) CUDA_OK( cudaMalloc( &m.acc, 4 * sizeof( double ) ) );
    if ( nb_cond && ! m.cond ) {
        m.cond_cap = std::min( m.n, 1 << 16 );
        CUDA_OK( cudaMalloc( &m.cond, size_t( m.cond_cap ) * sizeof( int ) ) );
        CUDA_OK( cudaMalloc( &m.ncond, sizeof( int ) ) );
    }
    const double init[ 1 ] = { 0.0 };
    CUDA_OK( cudaMemcpy( m.acc, init, sizeof( init ), cudaMemcpyHostToDevice ) );
    CUDA_OK( cudaMemcpy( m.acc + 2, init, sizeof( init ), cudaMemcpyHostToDevice ) );
    const unsigned long long haut = 0xffffffffffffffffull;   // `ordd` du plus grand : le min part de la
    CUDA_OK( cudaMemcpy( m.acc + 1, &haut, 8, cudaMemcpyHostToDevice ) );
    if ( nb_cond ) CUDA_OK( cudaMemset( m.ncond, 0, sizeof( int ) ) );
    k_residu<<<gr, BL>>>( m.res, cible, b, m.acc, m.n, seuil, m.cond, nb_cond ? m.ncond : nullptr, m.cond_cap );
    double s = 0;
    unsigned long long om = 0;
    CUDA_OK( cudaMemcpy( &s, m.acc, 8, cudaMemcpyDeviceToHost ) );
    CUDA_OK( cudaMemcpy( &om, m.acc + 1, 8, cudaMemcpyDeviceToHost ) );
    if ( nb_cond ) CUDA_OK( cudaMemcpy( nb_cond, m.ncond, sizeof( int ), cudaMemcpyDeviceToHost ) );
    if ( nb_vides ) { double v = 0; CUDA_OK( cudaMemcpy( &v, m.acc + 2, 8, cudaMemcpyDeviceToHost ) ); *nb_vides = int( v ); }
    // le meme codage entier ordonne que les boites ( `Bsp2D.cuh` ), decode ici
    const long long bits = ( long long ) ( ( om >> 63 ) ? ( om & 0x7fffffffffffffffull ) : ~om );
    double mn;
    std::memcpy( &mn, &bits, 8 );
    if ( mini ) *mini = mn;
    return std::sqrt( s );
}

/// LA HIERARCHIE : Galerkin niveau par niveau, l'agregation etant `>> 2` sur les rangs.
template<int D, class TK>
void DiagrammeGpu<D,TK>::monte_amg( const Hessienne &H ) {
    Impl &m = *impl;
    const int BL = 256;
    auto gr = [ & ]( int k ) { return ( k + BL - 1 ) / BL; };
    for ( size_t l = 0; l < m.niv.size(); ++l ) {
        Niveau &v = m.niv[ l ];
        if ( l ) { cudaFree( v.row ); cudaFree( v.col ); cudaFree( v.val ); cudaFree( v.dia ); }
        cudaFree( v.x ); cudaFree( v.b ); cudaFree( v.r );
        cudaFree( v.v1 ); cudaFree( v.v2 ); cudaFree( v.t ); cudaFree( v.rc ); cudaFree( v.sc );
        csr_libere( v.P ); csr_libere( v.R );
    }
    for ( int *p : m.map ) cudaFree( p );
    m.niv.clear(); m.map.clear();
    if ( const char *e = std::getenv( "AMG_K" ) ) m.kcycle = std::atoi( e );
    if ( const char *e = std::getenv( "AMG_GROS" ) ) m.gros = std::atoi( e );
    if ( const char *e = std::getenv( "AMG_NU" ) ) m.nu = std::atoi( e );
    if ( const char *e = std::getenv( "AMG_STOP" ) ) m.stop = std::atoi( e );
    if ( const char *e = std::getenv( "AMG_LISSE" ) ) m.lisse = std::atoi( e );
    if ( m.lisse && ! m.cus ) cusparseCreate( &m.cus );
    if ( ! m.rang_de ) {
        CUDA_OK( cudaMalloc( &m.rang_de, size_t( m.n ) * sizeof( int ) ) );
        CUDA_OK( cudaMalloc( &m.acc, 4 * sizeof( double ) ) );   // le CG en veut un, `residu` deux
        k_amg_rang<<<gr( m.n ), BL>>>( m.ids, m.rang_de, m.n );
    }

    // le niveau zero est la hessienne elle-meme ( on ne la recopie pas )
    Niveau f;
    f.row = ( int * ) H.row; f.col = ( int * ) H.col; f.val = ( double * ) H.val; f.dia = ( double * ) H.dia;
    f.n = H.n; f.nnz = H.nnz;
    m.niv.push_back( f );

    int *cpt = nullptr;
    CUDA_OK( cudaMalloc( &cpt, sizeof( int ) ) );
    // on s'arrete a MILLE inconnues. Descendre plus bas a ete essaye et PERD ( 411 iterations au
    // lieu de 168 a n = 2e5 ) : c'est la degradation connue de l'agregation non lissee avec le
    // nombre de niveaux, et trois cents lissages de Jacobi suffisent a ce niveau-la.
    for ( int l = 0; m.niv[ l ].n > m.stop && l < 24; ++l ) {
        const Niveau &g = m.niv[ l ];
        const int nc = ( g.n + 3 ) / 4;
        int *mp = nullptr;
        CUDA_OK( cudaMalloc( &mp, size_t( g.n ) * sizeof( int ) ) );
        if ( l == 0 ) k_amg_map_fin<<<gr( g.n ), BL>>>( m.rang_de, mp, g.n );
        else          k_amg_map<<<gr( g.n ), BL>>>( mp, g.n );
        m.map.push_back( mp );

        unsigned long long *cl, *cl2;
        double *po, *po2;
        CUDA_OK( cudaMalloc( &cl, size_t( g.nnz ) * 8 ) ); CUDA_OK( cudaMalloc( &cl2, size_t( g.nnz ) * 8 ) );
        CUDA_OK( cudaMalloc( &po, size_t( g.nnz ) * 8 ) ); CUDA_OK( cudaMalloc( &po2, size_t( g.nnz ) * 8 ) );
        if ( m.lisse ) {
            // ---- LA PROLONGATION LISSEE. `A` en CSR ordinaire et trie, `P` tentative, puis
            //      `P^ = P - omega D^-1 A P` en place sur `A P`, et `A_c = P^t ( A P^ )`.
            constexpr double OMP = 0.7;
            Csr A;
            A.lignes = A.colonnes = g.n; A.nnz = g.nnz + g.n;
            CUDA_OK( cudaMalloc( &A.row, size_t( A.lignes + 1 ) * 4 ) );
            CUDA_OK( cudaMalloc( &A.col, size_t( A.nnz ) * 4 ) );
            CUDA_OK( cudaMalloc( &A.val, size_t( A.nnz ) * 8 ) );
            k_lis_plein<<<gr( g.n + 1 ), BL>>>( g.row, g.col, g.val, g.dia, A.row, A.col, A.val, g.n );
            trie( m.cus, A );

            Csr P0;
            P0.lignes = g.n; P0.colonnes = nc; P0.nnz = g.n;
            CUDA_OK( cudaMalloc( &P0.row, size_t( g.n + 1 ) * 4 ) );
            CUDA_OK( cudaMalloc( &P0.col, size_t( g.n ) * 4 ) );
            CUDA_OK( cudaMalloc( &P0.val, size_t( g.n ) * 8 ) );
            k_lis_p0<<<gr( g.n + 1 ), BL>>>( mp, P0.row, P0.col, P0.val, g.n );

            Csr Ph, APh, Ac, R;
            spgemm( m.cus, A, P0, Ph );                  // `A P`, puis lisse EN PLACE
            k_lis_lisse<<<gr( g.n ), BL>>>( Ph.row, Ph.col, Ph.val, g.dia, mp, OMP, g.n );
            spgemm( m.cus, A, Ph, APh );
            transpose( m.cus, Ph, R );
            spgemm( m.cus, R, APh, Ac );

            Niveau c2;
            c2.n = nc;
            CUDA_OK( cudaMalloc( &c2.row, size_t( nc + 1 ) * 4 ) );
            CUDA_OK( cudaMalloc( &c2.dia, size_t( nc ) * 8 ) );
            CUDA_OK( cudaMemset( c2.row, 0, size_t( nc + 1 ) * 4 ) );
            k_lis_compte<<<gr( nc ), BL>>>( Ac.row, Ac.col, nc, c2.row );
            void *ts2 = nullptr; size_t tso2 = 0;
            cub::DeviceScan::ExclusiveSum( ts2, tso2, c2.row, c2.row, nc + 1 );
            CUDA_OK( cudaMalloc( &ts2, tso2 ) );
            CUDA_OK( cub::DeviceScan::ExclusiveSum( ts2, tso2, c2.row, c2.row, nc + 1 ) );
            CUDA_OK( cudaMemcpy( &c2.nnz, c2.row + nc, 4, cudaMemcpyDeviceToHost ) );
            CUDA_OK( cudaMalloc( &c2.col, size_t( c2.nnz ) * 4 ) );
            CUDA_OK( cudaMalloc( &c2.val, size_t( c2.nnz ) * 8 ) );
            k_lis_verse<<<gr( nc ), BL>>>( Ac.row, Ac.col, Ac.val, c2.row, c2.col, c2.val, c2.dia, nc );
            CUDA_OK( cudaMalloc( &c2.x, size_t( nc ) * 8 ) ); CUDA_OK( cudaMalloc( &c2.b, size_t( nc ) * 8 ) );
            CUDA_OK( cudaMalloc( &c2.r, size_t( nc ) * 8 ) );
            CUDA_OK( cudaMalloc( &c2.v1, size_t( nc ) * 8 ) ); CUDA_OK( cudaMalloc( &c2.v2, size_t( nc ) * 8 ) );
            CUDA_OK( cudaMalloc( &c2.t, size_t( nc ) * 8 ) );  CUDA_OK( cudaMalloc( &c2.rc, size_t( nc ) * 8 ) );
            CUDA_OK( cudaMalloc( &c2.sc, 8 * 8 ) );
            c2.P = Ph; c2.R = R;                         // gardes : le cycle s'en sert
            csr_libere( A ); csr_libere( P0 ); csr_libere( APh ); csr_libere( Ac );
            cudaFree( ts2 );
            m.niv.push_back( c2 );
            continue;
        }
        CUDA_OK( cudaMemset( cpt, 0, 4 ) );
        k_amg_triples<<<gr( g.n ), BL>>>( g.row, g.col, g.val, mp, g.n, nc, cl, po, cpt );
        int nt = 0;
        CUDA_OK( cudaMemcpy( &nt, cpt, 4, cudaMemcpyDeviceToHost ) );

        void *tmp = nullptr; size_t to = 0;
        cub::DeviceRadixSort::SortPairs( tmp, to, cl, cl2, po, po2, nt );
        CUDA_OK( cudaMalloc( &tmp, to ) );
        CUDA_OK( cub::DeviceRadixSort::SortPairs( tmp, to, cl, cl2, po, po2, nt ) );
        int *nu = nullptr;
        CUDA_OK( cudaMalloc( &nu, 4 ) );
        void *tr = nullptr; size_t tro = 0;
        cub::DeviceReduce::ReduceByKey( tr, tro, cl2, cl, po2, po, nu, ::cuda::std::plus<double>{}, nt );
        CUDA_OK( cudaMalloc( &tr, tro ) );
        CUDA_OK( cub::DeviceReduce::ReduceByKey( tr, tro, cl2, cl, po2, po, nu, ::cuda::std::plus<double>{}, nt ) );
        int nun = 0;
        CUDA_OK( cudaMemcpy( &nun, nu, 4, cudaMemcpyDeviceToHost ) );

        Niveau c;
        c.n = nc; c.nnz = nun;
        CUDA_OK( cudaMalloc( &c.row, size_t( nc + 1 ) * 4 ) );
        CUDA_OK( cudaMalloc( &c.col, size_t( nun ) * 4 ) );
        CUDA_OK( cudaMalloc( &c.val, size_t( nun ) * 8 ) );
        CUDA_OK( cudaMalloc( &c.dia, size_t( nc ) * 8 ) );
        CUDA_OK( cudaMalloc( &c.x, size_t( nc ) * 8 ) ); CUDA_OK( cudaMalloc( &c.b, size_t( nc ) * 8 ) );
        CUDA_OK( cudaMalloc( &c.r, size_t( nc ) * 8 ) );
        CUDA_OK( cudaMalloc( &c.v1, size_t( nc ) * 8 ) ); CUDA_OK( cudaMalloc( &c.v2, size_t( nc ) * 8 ) );
        CUDA_OK( cudaMalloc( &c.t, size_t( nc ) * 8 ) );  CUDA_OK( cudaMalloc( &c.rc, size_t( nc ) * 8 ) );
        CUDA_OK( cudaMalloc( &c.sc, 8 * 8 ) );
        CUDA_OK( cudaMemset( c.row, 0, size_t( nc + 1 ) * 4 ) );
        k_amg_compte<<<gr( nun ), BL>>>( cl, nun, nc, c.row );
        void *ts = nullptr; size_t tso = 0;
        cub::DeviceScan::ExclusiveSum( ts, tso, c.row, c.row, nc + 1 );
        CUDA_OK( cudaMalloc( &ts, tso ) );
        CUDA_OK( cub::DeviceScan::ExclusiveSum( ts, tso, c.row, c.row, nc + 1 ) );
        int *at = nullptr;
        CUDA_OK( cudaMalloc( &at, size_t( nc ) * 4 ) );
        CUDA_OK( cudaMemset( at, 0, size_t( nc ) * 4 ) );
        k_amg_place<<<gr( nun ), BL>>>( cl, po, nun, nc, c.row, at, c.col, c.val );
        k_amg_dia<<<gr( nc ), BL>>>( c.row, c.val, c.dia, nc );
        cudaFree( cl ); cudaFree( cl2 ); cudaFree( po ); cudaFree( po2 );
        cudaFree( tmp ); cudaFree( tr ); cudaFree( ts ); cudaFree( nu ); cudaFree( at );
        m.niv.push_back( c );
    }
    cudaFree( cpt );
    // le niveau fin a besoin de ses vecteurs de travail lui aussi
    CUDA_OK( cudaMalloc( &m.niv[ 0 ].r, size_t( m.n ) * 8 ) );
    CUDA_OK( cudaMalloc( &m.niv[ 0 ].x, size_t( m.n ) * 8 ) );
    CUDA_OK( cudaMalloc( &m.niv[ 0 ].b, size_t( m.n ) * 8 ) );
}

/// UN CYCLE EN V : lissage, restriction, recursion, prolongation, lissage
template<int D, class TK>
void DiagrammeGpu<D,TK>::cycle_v( int l ) {
    Impl &m = *impl;
    const int BL = 256;
    auto gr = [ & ]( int k ) { return ( k + BL - 1 ) / BL; };
    Niveau &g = m.niv[ l ];
    constexpr double OM = 0.7;
    const int NU = m.nu;                                 // pre et post, pour la symetrie
    CUDA_OK( cudaMemsetAsync( g.x, 0, size_t( g.n ) * 8 ) );
    if ( l + 1 == int( m.niv.size() ) ) {
        // les `m.gros` lissages en UN SEUL noyau ( `k_amg_gros` ) ont ete essayes et PERDENT :
        // un seul bloc n'occupe qu'un SM sur soixante-huit, et la perte de parallelisme coute
        // plus que les soixante lancements epargnes ( 969 ms contre 873 a n = 1e6 ).
        for ( int k = 0; k < m.gros; ++k )
            k_amg_jacobi<<<gr( g.n ), BL>>>( g.row, g.col, g.val, g.dia, g.x, g.b, OM, g.n );
        return;
    }
    Niveau &c = m.niv[ l + 1 ];
    for ( int k = 0; k < NU; ++k )
        k_amg_jacobi<<<gr( g.n ), BL>>>( g.row, g.col, g.val, g.dia, g.x, g.b, OM, g.n );
    k_amg_residu<<<gr( g.n ), BL>>>( g.row, g.col, g.val, g.dia, g.x, g.b, g.r, g.n );
    if ( c.R.row ) {
        k_lis_spmv<<<( c.n + BL - 1 ) / BL, BL>>>( c.R.row, c.R.col, c.R.val, g.r, c.b, c.n, false );
    } else {
        CUDA_OK( cudaMemsetAsync( c.b, 0, size_t( c.n ) * 8 ) );
        k_amg_restreint<<<gr( g.n ), BL>>>( g.r, m.map[ l ], c.b, g.n );
    }

    if ( l >= m.kcycle ) {
        cycle_v( l + 1 );
    } else {
        // LE K-CYCLE : deux pas d'un gradient conjugue sur le systeme grossier, preconditionnes
        // par le niveau d'en dessous. `sc` : rho1, a1, g2, b2, a2, c1, c2, libre.
        const int gc = ( c.n + BL - 1 ) / BL;
        double *rho1 = c.sc, *a1 = c.sc + 1, *g2 = c.sc + 2, *b2 = c.sc + 3, *a2 = c.sc + 4,
               *c1 = c.sc + 5, *c2 = c.sc + 6;
        auto dot = [ & ]( const double *u, const double *v, double *acc ) {
            k_cg_zero<<<1,1>>>( acc );
            k_cg_dot<<<gc, BL>>>( u, v, acc, c.n );
        };
        k_amg_copie<<<gc, BL>>>( c.rc, c.b, c.n );       // le second membre du premier pas
        cycle_v( l + 1 );                                // v1 = M^-1 b_c
        k_amg_copie<<<gc, BL>>>( c.v1, c.x, c.n );
        k_amg_matvec<<<gc, BL>>>( c.row, c.col, c.val, c.dia, c.v1, c.t, c.n );
        dot( c.v1, c.t, rho1 );
        dot( c.v1, c.rc, a1 );
        k_kc_c1<<<1,1>>>( rho1, a1, c1 );
        k_kc_rc<<<gc, BL>>>( c.b, c.rc, c.t, c1, c.n );  // le residu devient le second membre
        cycle_v( l + 1 );                                // v2 = M^-1 r
        k_amg_copie<<<gc, BL>>>( c.v2, c.x, c.n );
        dot( c.v2, c.t, g2 );
        dot( c.v2, c.b, a2 );
        k_amg_matvec<<<gc, BL>>>( c.row, c.col, c.val, c.dia, c.v2, c.t, c.n );
        dot( c.v2, c.t, b2 );
        k_kc_c2<<<1,1>>>( rho1, a1, g2, b2, a2, c1, c2 );
        k_kc_comb<<<gc, BL>>>( c.x, c.v1, c.v2, c1, c2, c.n );
    }
    if ( c.P.row ) k_lis_spmv<<<gr( g.n ), BL>>>( c.P.row, c.P.col, c.P.val, c.x, g.x, g.n, true );
    else           k_amg_prolonge<<<gr( g.n ), BL>>>( g.x, c.x, m.map[ l ], g.n );
    for ( int k = 0; k < NU; ++k )
        k_amg_jacobi<<<gr( g.n ), BL>>>( g.row, g.col, g.val, g.dia, g.x, g.b, OM, g.n );
}

/// LE GRADIENT CONJUGUE PRECONDITIONNE. Les scalaires restent sur la carte ; seul le test d'arret
/// redescend un nombre par iteration.
template<int D, class TK>
int DiagrammeGpu<D,TK>::resout( const Hessienne &H, const double *b, double *x, double tol, int maxit,
                                double *ms, double *res ) {
    if constexpr ( D != 2 ) { ( void ) H; ( void ) b; ( void ) x; return -1; }
    else {
        Impl &m = *impl;
        const int n = H.n, BL = 256, gr = ( n + BL - 1 ) / BL;
        if ( ! m.cgv ) CUDA_OK( cudaMalloc( &m.cgv, ( size_t( 4 ) * n + 8 ) * sizeof( double ) ) );
        double *r = m.cgv, *z = r + n, *p = z + n, *q = p + n;
        double *srz = q + n, *spq = srz + 1, *srz2 = spq + 1, *sal = srz2 + 1, *sbe = sal + 1, *sbb = sbe + 1;
        auto zero = [ & ]( double *a ) { k_cg_zero<<<1,1>>>( a ); };
        auto dot  = [ & ]( const double *u, const double *v, double *a ) { zero( a ); k_cg_dot<<<gr, BL>>>( u, v, a, n ); };

        cudaEvent_t e0, e1;
        CUDA_OK( cudaEventCreate( &e0 ) ); CUDA_OK( cudaEventCreate( &e1 ) );
        CUDA_OK( cudaEventRecord( e0 ) );
        CUDA_OK( cudaMemsetAsync( x, 0, size_t( n ) * sizeof( double ) ) );
        CUDA_OK( cudaMemcpyAsync( r, b, size_t( n ) * sizeof( double ), cudaMemcpyDeviceToDevice ) );
        auto centre = [ & ]( double *v ) {              // la jauge : moyenne nulle
            k_cg_zero<<<1,1>>>( m.acc );
            k_amg_somme<<<gr, BL>>>( v, m.acc, n );
            k_amg_centre<<<gr, BL>>>( v, m.acc, n );
        };
        auto precond = [ & ]( const double *rr, double *zz ) {
            if ( m.niv.size() > 1 ) {                    // le multigrille
                CUDA_OK( cudaMemcpyAsync( m.niv[ 0 ].b, rr, size_t( n ) * sizeof( double ), cudaMemcpyDeviceToDevice ) );
                cycle_v( 0 );
                CUDA_OK( cudaMemcpyAsync( zz, m.niv[ 0 ].x, size_t( n ) * sizeof( double ), cudaMemcpyDeviceToDevice ) );
            } else
                k_cg_prec<<<gr, BL>>>( rr, H.dia, zz, n );
            centre( zz );
        };
        centre( r );
        dot( r, r, sbb );
        precond( r, z );
        CUDA_OK( cudaMemcpyAsync( p, z, size_t( n ) * sizeof( double ), cudaMemcpyDeviceToDevice ) );
        dot( r, z, srz );
        double bb = 0;
        CUDA_OK( cudaMemcpy( &bb, sbb, sizeof( double ), cudaMemcpyDeviceToHost ) );
        const double cible = tol * tol * bb;
        int it = 0;
        double rr = bb;
        for ( ; it < maxit && rr > cible; ++it ) {
            k_cg_mul<<<gr, BL>>>( H.row, H.col, H.val, H.dia, p, q, n );
            dot( p, q, spq );
            k_cg_alpha<<<1,1>>>( srz, spq, sal );
            k_cg_avance<<<gr, BL>>>( x, r, p, q, sal, n );
            dot( r, r, sbb );
            CUDA_OK( cudaMemcpy( &rr, sbb, sizeof( double ), cudaMemcpyDeviceToHost ) );
            if ( rr <= cible ) { ++it; break; }
            precond( r, z );
            dot( r, z, srz2 );
            k_cg_beta<<<1,1>>>( srz2, srz, sbe );
            k_cg_dir<<<gr, BL>>>( p, z, sbe, n );
            k_cg_copie<<<1,1>>>( srz, srz2 );
        }
        CUDA_OK( cudaEventRecord( e1 ) );
        CUDA_OK( cudaEventSynchronize( e1 ) );
        float t = 0;
        CUDA_OK( cudaEventElapsedTime( &t, e0, e1 ) );
        CUDA_OK( cudaGetLastError() );
        cudaEventDestroy( e0 ); cudaEventDestroy( e1 );
        if ( ms ) *ms = t;
        if ( res ) *res = bb > 0 ? std::sqrt( rr / bb ) : 0.0;
        return rr <= cible ? it : -1;
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

// =====================================================================================
// LE TOUR DE CELLULES ( 2D ) : `filmsk`, puis `filmix` pour ce qui a deborde -- avec la densite
// image la ou le mode le demande. C'est le SEUL endroit qui connait les quatre modes, et les
// trois chemins ( mesures, facettes, Newton ) passent tous par lui.
// =====================================================================================
template<class TK, bool POIDS, int FIX>
static int *cellules_2d( typename DiagrammeGpu<2,TK>::Impl &m, const Arbre<TK,2> &ar, int *dj, TK *dl, int NF ) {
    const int bloc = 128;
    const Image2 im = m.image();
    const Densite d = m.mode();
    CUDA_OK( cudaMemsetAsync( m.deb, 0, sizeof( int ) ) );
    CUDA_OK( cudaMemsetAsync( m.deb2, 0, sizeof( int ) ) );

    if ( ( d == Densite::DEPOT || d == Densite::DEPOT_ARETE ) && m.dep_cap > 0 ) {
        // PAR LOTS : a 10^9 germes on n'a pas la place de garder tous les polygones a la fois
        for ( int k0 = 0; k0 < m.n; k0 += m.dep_cap ) {
            const int nk = std::min( m.dep_cap, m.n - k0 );
            noyau2_filmsk<POIDS,1,true,FIX,DENS_DEPOT><<<( nk + bloc - 1 ) / bloc, bloc>>>(
                ar, m.res, m.deb, m.liste, dj, dl, NF, im, m.dep_x, m.dep_y, m.dep_nb, m.dep_id, m.dep_cap, k0, nk );
            if ( d == Densite::DEPOT )
                k_dens_cel<TK><<<( nk + bloc - 1 ) / bloc, bloc>>>( im, m.dep_x, m.dep_y, m.dep_nb, m.dep_id, m.dep_cap, nk, m.n, m.res, dl );
            else
                k_dens_arete<TK,8><<<( nk * 8 + bloc - 1 ) / bloc, bloc>>>( im, m.dep_x, m.dep_y, m.dep_nb, m.dep_id, m.dep_cap, nk, m.n, m.res, dl );
        }
    } else if ( d == Densite::AUCUNE ) {
        noyau2_filmsk<POIDS,1,true,FIX,DENS_AUCUNE><<<( m.n + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb, m.liste, dj, dl, NF );
    } else {
        noyau2_filmsk<POIDS,1,true,FIX,DENS_DIRECTE><<<( m.n + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb, m.liste, dj, dl, NF, im );
    }

    int nd = 0;
    CUDA_OK( cudaMemcpy( &nd, m.deb, sizeof( int ), cudaMemcpyDeviceToHost ) );
    if ( nd == 0 ) return m.deb;
    // la seconde passe finit les cellules trop grosses pour les registres, densite comprise
    noyau2_filmix<POIDS,64,8,true><<<( nd + bloc - 1 ) / bloc, bloc>>>( ar, m.res, m.deb2, m.liste, nd, dj, dl, NF, m.cptr, im );
    return m.deb2;
}

// =====================================================================================
// LA DENSITE IMAGE : la montee de la grille, et le choix du mode.
// =====================================================================================
template<int D, class TK>
double DiagrammeGpu<D,TK>::charge_image( const double *v, int W, int H ) {
    Impl &m = *impl;
    cudaFree( m.img );                                   // et RIEN d'autre : la liste des
    m.img = nullptr; m.iw = 0; m.ih = 0;                 // condamnees survit au changement d'image
    if ( ! v || W <= 0 || H <= 0 ) return 0;
    const double hx = 1.0 / W, hy = 1.0 / H;
    // la SOMME PREFIXE de chaque ligne, calculee une fois pour toutes : c'est elle qui rend
    // l'integrale de bord exacte sans jamais couper la cellule ( voir `Image2D.cuh` )
    std::vector<double2> t( size_t( W ) * H );
    double tot = 0;
    for ( int j = 0; j < H; ++j ) {
        double s = 0;
        for ( int i = 0; i < W; ++i ) {
            const double r = v[ size_t( j ) * W + i ];
            t[ size_t( j ) * W + i ] = make_double2( s, r );
            s += r * hx;
        }
        tot += s * hy;
    }
    CUDA_OK( cudaMalloc( &m.img, t.size() * sizeof( double2 ) ) );
    CUDA_OK( cudaMemcpy( m.img, t.data(), t.size() * sizeof( double2 ), cudaMemcpyHostToDevice ) );
    m.iw = W; m.ih = H;
    return tot;
}

template<int D, class TK>
void DiagrammeGpu<D,TK>::regle_densite( Densite d, int chunk ) {
    Impl &m = *impl;
    m.dens = d;
    m.chunk = chunk > 0 ? chunk : 1;
    if ( d != Densite::DEPOT && d != Densite::DEPOT_ARETE ) return;
    const int cap = std::min( m.chunk, m.n );
    if ( cap == m.dep_cap ) return;
    cudaFree( m.dep_x ); cudaFree( m.dep_y ); cudaFree( m.dep_nb ); cudaFree( m.dep_id );
    CUDA_OK( cudaMalloc( &m.dep_x, size_t( 8 ) * cap * sizeof( TK ) ) );
    CUDA_OK( cudaMalloc( &m.dep_y, size_t( 8 ) * cap * sizeof( TK ) ) );
    CUDA_OK( cudaMalloc( &m.dep_nb, size_t( cap ) * sizeof( int ) ) );
    CUDA_OK( cudaMalloc( &m.dep_id, size_t( cap ) * sizeof( int ) ) );
    m.dep_cap = cap;
}

template<int D, class TK>
Densite DiagrammeGpu<D,TK>::densite() const { return impl->mode(); }

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
            cellules_2d<TK,POIDS,FIX>( m, ar, m.fac_j, m.fac_l, NF );
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
    const int bloc = 128;
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
        // la seconde passe rend AUSSI ses facettes : elle finit 13 % des cellules en uniforme
        Chrono ch = chrono<2,TK>( m, reps, res, [ & ]() { return cellules_2d<TK,POIDS,FIX>( mm, ar, dj, dl, NF ); } );
        // les registres et l'occupation sont ceux du noyau REELLEMENT lance
        const Densite md = mm.mode();
        infos( ch, md == Densite::DIRECTE ? ( const void * ) noyau2_filmsk<POIDS,1,true,FIX,DENS_DIRECTE,TK>
                 : md == Densite::AUCUNE  ? ( const void * ) noyau2_filmsk<POIDS,1,true,FIX,DENS_AUCUNE,TK>
                                          : ( const void * ) noyau2_filmsk<POIDS,1,true,FIX,DENS_DEPOT,TK>, bloc );
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
