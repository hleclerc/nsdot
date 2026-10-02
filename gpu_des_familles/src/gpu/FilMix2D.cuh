#pragma once

// =====================================================================================
// UNE CELLULE PAR THREAD, 2D, `R` SOMMETS EN REGISTRES ET LA QUEUE EN MEMOIRE. La generalisation
// de `FilReg2D.cuh` : les sommets `0 .. R-1` sont dans des registres et leur code est deroule,
// les sommets `R .. nb-1` sont dans des tableaux locaux et parcourus par une boucle ordinaire --
// qui diverge entre les threads d'un warp, mais qui ne fait pas plus de travail, et qui ne fait
// RIEN pour une cellule qui tient dans ses registres. Il n'y a plus d'excursion : `nb` va
// jusqu'a `MaxNb` sans changer de mode, la coupe est une seule et meme suite d'instructions.
//
// Ce que `R` regle : le code deroule fait toujours `R` cases, qu'elles soient occupees ou non
// ( une cellule finie a six sommets ) ; la queue coute une boucle divergente. `R = 8` est
// `filreg` sans excursion, `R = 4` parie sur le peu de travail perdu, `R = 16` sur l'absence de
// queue ( Laguerre, ou les cellules ont plus de sommets ).
//
// Les masques sont sur 64 bits ( `MaxNb <= 64` ), `ffsll` / `popcll` les lisent.
// =====================================================================================

#include "gpu/Fil2D.cuh"
#include "gpu/Image2D.cuh"

namespace sf::gpu {

template<class T, int R>
__device__ __forceinline__ T selR( const T ( &a )[ R ], int i ) {
    T r = a[ 0 ];
#pragma unroll
    for ( int q = 1; q < R; ++q ) r = i == q ? a[ q ] : r;
    return r;
}

/// `liste` : les rangs a faire ( `nullptr` : tous ) -- la seconde passe de `filbrk`.
/// `fac_j` / `fac_l` / `NF` : les facettes, comme `noyau2_filmsk` -- indispensables ici, puisque
/// c'est CE noyau qui finit les cellules que la premiere passe a fait deborder ( 13 % en uniforme )
/// et que sans elles leur ligne de la hessienne serait vide. `fac_deb` compte celles dont le
/// polygone final a plus de `NF` aretes.
/// `im` : la densite image, si elle est active. Ce noyau finit les cellules que la premiere passe
/// a fait deborder -- une poignee -- donc la densite s'y traite TOUJOURS sur place, et le choix
/// n'est qu'un test a l'execution : y mettre un parametre de template ne paierait rien.
/// `CENTRE` et `FIXE` : les memes que `noyau2_filmsk`, et pour la meme raison. CE NOYAU FINIT
/// 13 % DES CELLULES en uniforme -- celles qui debordent les registres de la premiere passe --
/// donc le laisser en repere ABSOLU pendant que `filmsk` travaille dans le repere du germe
/// revient a laisser une cellule sur huit avec l'erreur qu'on vient d'enlever aux autres ; et
/// comme le maximum de l'erreur suit la pire cellule, ce sont elles qui le fixent.
template<bool POIDS, int MaxNb, int R, bool CIDREG, bool CENTRE = false, int FIXE = 0, class TK = float, bool RES64 = false>
__global__ void __launch_bounds__( 128 ) noyau2_filmix( Arbre<TK,2> ar, double *res, int *deborde, const int *liste = nullptr, int nl = 0,
                                                        int *fac_j = nullptr, TK *fac_l = nullptr, int NF = 0, int *fac_deb = nullptr,
                                                        bool raff = false, Image2 im = Image2{} ) {
    static_assert( MaxNb <= 64 && R <= MaxNb && R >= 4, "les masques sont sur 64 bits, le carre tient dans les registres" );
    static_assert( FIXE == 0 || CENTRE, "la virgule fixe n'a de sens que dans le repere du germe" );
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = liste ? ( ti < nl ? liste[ ti ] : ar.n ) : ti;
    if ( k >= ar.n ) return;
    const int u0[ 2 ] = { FIXE == 32 ? ar.u[ 0 ][ k ] : 0, FIXE == 32 ? ar.u[ 1 ][ k ] : 0 };
    const long long g0[ 2 ] = { FIXE == 64 ? ar.u64[ 0 ][ k ] : 0, FIXE == 64 ? ar.u64[ 1 ][ k ] : 0 };
    const TK p0[ 2 ] = { FIXE == 32 ? TK( u0[ 0 ] ) * TK( INV_FIXE ) : FIXE == 64 ? TK( double( g0[ 0 ] ) * INV_F64 ) : ar.c[ 0 ][ k ],
                         FIXE == 32 ? TK( u0[ 1 ] ) * TK( INV_FIXE ) : FIXE == 64 ? TK( double( g0[ 1 ] ) * INV_F64 ) : ar.c[ 1 ][ k ] };
    const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
    const double w0d = POIDS ? ar.w64[ k ] : 0.0;
    const int i0 = ar.ids[ k ];

    // ---- les registres, puis la queue
    TK  x[ R ], y[ R ];
    int cr[ R ];                                         // `CIDREG`
    TK  lx[ MaxNb ], ly[ MaxNb ], ls[ MaxNb ];          // la queue ( entrees `>= R` ), et `s` de la queue
    int lc[ MaxNb ];                                     // les cid : tous si `! CIDREG`, la queue sinon
#pragma unroll
    for ( int i = 0; i < R; ++i ) {
        x[ i ] = FIXE == 32 ? TK( ( i == 1 || i == 2 ? ECH_FIXE : 0 ) - u0[ 0 ] ) * TK( INV_FIXE )
               : FIXE == 64 ? TK( ( i == 1 || i == 2 ? ECH_F64 : 0ll ) - g0[ 0 ] ) * TK( INV_F64 )
                            : TK( i == 1 || i == 2 ) - ( CENTRE ? p0[ 0 ] : TK( 0 ) );
        y[ i ] = FIXE == 32 ? TK( ( i == 2 || i == 3 ? ECH_FIXE : 0 ) - u0[ 1 ] ) * TK( INV_FIXE )
               : FIXE == 64 ? TK( ( i == 2 || i == 3 ? ECH_F64 : 0ll ) - g0[ 1 ] ) * TK( INV_F64 )
                            : TK( i == 2 || i == 3 ) - ( CENTRE ? p0[ 1 ] : TK( 0 ) );
        cr[ i ] = i < 4 ? -1 - i : 0;
    }
    lc[ 0 ] = -1; lc[ 1 ] = -2; lc[ 2 ] = -3; lc[ 3 ] = -4;
    int nb = 4;

    auto get_x = [ & ]( int j ) { return j < R ? selR( x, j ) : lx[ j ]; };
    auto get_y = [ & ]( int j ) { return j < R ? selR( y, j ) : ly[ j ]; };
    auto get_c = [ & ]( int j ) { return CIDREG && j < R ? selR( cr, j ) : lc[ j ]; };

    int pile[ PILE ];
    int haut = 0;
    pile[ haut++ ] = 0;
    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,2> nd = ar.nodes[ h ];

        // ---- le test d'elagage : `R` bilans deroules, la queue en boucle
        bool peut = false;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            const TK v[ 2 ] = { x[ i ], y[ i ] };
            peut |= i < nb && ( CENTRE ? bilan_sommet_c<POIDS>( nd, v, p0, w0 ) : bilan_sommet<POIDS>( nd, v, p0, w0 ) ) <= TK( 0 );
        }
        for ( int i = R; i < nb && ! peut; ++i ) {
            const TK v[ 2 ] = { lx[ i ], ly[ i ] };
            peut |= ( CENTRE ? bilan_sommet_c<POIDS>( nd, v, p0, w0 ) : bilan_sommet<POIDS>( nd, v, p0, w0 ) ) <= TK( 0 );
        }
        if ( ! peut )
            continue;

        if ( nd.right >= 0 ) {
            const int g = h + 1, dr = nd.right;
            const bool gp = proximite( ar.nodes[ g ], p0 ) <= proximite( ar.nodes[ dr ], p0 );
            pile[ haut++ ] = gp ? dr : g;
            pile[ haut++ ] = gp ? g : dr;
            continue;
        }

        for ( int q = nd.beg; q < nd.end; ++q ) {
            if ( q == k ) continue;
            const Plan2<TK> p = FIXE == 32 ? bissect2f<POIDS>( ar, q, u0[ 0 ], u0[ 1 ], w0d )
                              : FIXE == 64 ? bissect2g<POIDS>( ar, q, g0[ 0 ], g0[ 1 ], w0d )
                              : CENTRE     ? bissect2c<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], w0d )
                                           : bissect2<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], w0 );

            // LE TEST, QUI EST DEJA LA COUPE : le masque de signe sur 64 bits
            TK s[ R ];
            unsigned long long m = 0;
#pragma unroll
            for ( int i = 0; i < R; ++i ) {
                s[ i ] = p.dx * x[ i ] + p.dy * y[ i ] - p.off;
                m |= ( unsigned long long ) ( i < nb && s[ i ] > TK( 0 ) ) << i;
            }
            for ( int i = R; i < nb; ++i ) {
                ls[ i ] = p.dx * lx[ i ] + p.dy * ly[ i ] - p.off;
                m |= ( unsigned long long ) ( ls[ i ] > TK( 0 ) ) << i;
            }
            if ( ! m )
                continue;
            const unsigned long long valid = nb >= 64 ? ~0ull : ( 1ull << nb ) - 1;
            if ( m == valid ) { nb = 0; goto fin; }

            const unsigned long long prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
            const unsigned long long next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
            const int i1 = __ffsll( ( long long ) ( m & ~prev ) ) - 1;
            const int j2 = __ffsll( ( long long ) ( m & ~next ) ) - 1;
            const int j0 = i1 ? i1 - 1 : nb - 1;
            const int j3 = j2 + 1 < nb ? j2 + 1 : 0;
            const int nb_in = nb - __popcll( m );
            const int nn = nb_in + 2;
            if ( nn > MaxNb ) { nb = -1; goto fin; }

            // les deux intersections, ancrees sur le sommet dedans
            auto get_s = [ & ]( int j ) { return j < R ? selR( s, j ) : ls[ j ]; };
            const TK s0 = get_s( j0 ), s1 = get_s( i1 ), s2 = get_s( j2 ), s3 = get_s( j3 );
            const TK x0v = get_x( j0 ), y0v = get_y( j0 ), x1v = get_x( i1 ), y1v = get_y( i1 );
            const TK x2v = get_x( j2 ), y2v = get_y( j2 ), x3v = get_x( j3 ), y3v = get_y( j3 );
            const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            const TK pax = x0v + ( x1v - x0v ) * ta, pay = y0v + ( y1v - y0v ) * ta;
            const TK pbx = x3v + ( x2v - x3v ) * tb, pby = y3v + ( y2v - y3v ) * tb;
            const int bid = get_c( j2 );

            // LE REMONTAGE `[ v_j3, ..., v_j0, A, B ]` : les registres deroules, la queue en boucle
            // dans un tampon ( elle lit derriere elle ), rien n'est ecrit avant que tout soit lu
            TK  nx[ R ], ny[ R ];
            int nc[ R ];
#pragma unroll
            for ( int o = 0; o < R; ++o ) {
                int og = j3 + o; og = og >= nb ? og - nb : og;
                nx[ o ] = o == nb_in ? pax : ( o == nb_in + 1 ? pbx : get_x( og ) );
                ny[ o ] = o == nb_in ? pay : ( o == nb_in + 1 ? pby : get_y( og ) );
                nc[ o ] = o == nb_in ? p.id : ( o == nb_in + 1 ? bid : get_c( og ) );
            }
            TK  tx[ MaxNb ], ty[ MaxNb ];
            int tc[ MaxNb ];
            for ( int o = R; o < nn; ++o ) {
                int og = j3 + o; og = og >= nb ? og - nb : og;
                tx[ o ] = o == nb_in ? pax : ( o == nb_in + 1 ? pbx : get_x( og ) );
                ty[ o ] = o == nb_in ? pay : ( o == nb_in + 1 ? pby : get_y( og ) );
                tc[ o ] = o == nb_in ? p.id : ( o == nb_in + 1 ? bid : get_c( og ) );
            }
#pragma unroll
            for ( int o = 0; o < R; ++o ) { x[ o ] = nx[ o ]; y[ o ] = ny[ o ]; if ( CIDREG ) cr[ o ] = nc[ o ]; else lc[ o ] = nc[ o ]; }
            for ( int o = R; o < nn; ++o ) { lx[ o ] = tx[ o ]; ly[ o ] = ty[ o ]; lc[ o ] = tc[ o ]; }
            nb = nn;
        }
    }

fin:
    // ---- LES SOMMETS RESOLUS DEPUIS LEURS PLANS, comme `filmsk` -- et il le faut d'autant plus
    //      ici que ce noyau prend les GROSSES cellules, celles dont l'histoire de coupes est la
    //      plus longue, donc l'erreur portee la plus grande.
    //
    // `RES64` : la mesure prise DANS la boucle, sur le sommet resolu et avant qu'il ne repasse par
    // le flottant du noyau ( voir `FilMsk2D.cuh` pour le pourquoi ). Ici la boucle n'est pas
    // deroulee -- `nb` va jusqu'a `MaxNb` -- donc les indices sont a l'execution, et le lacet se
    // ferme de la meme facon : deux `double` pour le sommet precedent, deux pour le premier.
    double a2 = 0;
    bool   pris = false;
    if constexpr ( sizeof( TK ) < 8 && ( CENTRE || FIXE ) ) {
        if ( raff ) {
            double v0x = 0, v0y = 0, ppx = 0, ppy = 0;
            if ( nb >= 3 ) {
                double ax, ay, ao;
                plan_relu<POIDS,FIXE>( ar, get_c( nb - 1 ), u0, g0, p0, w0d, ax, ay, ao );
                for ( int i = 0; i < nb; ++i ) {
                    double bx, by, bo, vx, vy;
                    plan_relu<POIDS,FIXE>( ar, get_c( i ), u0, g0, p0, w0d, bx, by, bo );
                    if ( ! croise2( ax, ay, ao, bx, by, bo, vx, vy ) ) { vx = double( get_x( i ) ); vy = double( get_y( i ) ); }
                    ax = bx; ay = by; ao = bo;
                    if ( i < R ) {
#pragma unroll
                        for ( int o = 0; o < R; ++o ) { x[ o ] = o == i ? TK( vx ) : x[ o ]; y[ o ] = o == i ? TK( vy ) : y[ o ]; }
                    } else { lx[ i ] = TK( vx ); ly[ i ] = TK( vy ); }
                    if constexpr ( RES64 ) {
                        if ( i == 0 ) { v0x = vx; v0y = vy; }
                        else {
                            a2 += ppx * vy - vx * ppy;
                            if ( fac_j && i - 1 < NF ) {
                                const double ex = vx - ppx, ey = vy - ppy;
                                fac_j[ size_t( i - 1 ) * ar.n + i0 ] = id_de( ar, get_c( i - 1 ) );
                                fac_l[ size_t( i - 1 ) * ar.n + i0 ] = TK( sqrt( ex * ex + ey * ey ) );
                            }
                        }
                        ppx = vx; ppy = vy;
                    }
                }
                if constexpr ( RES64 ) {
                    a2 += ppx * v0y - v0x * ppy;
                    if ( fac_j && nb - 1 < NF ) {
                        const double ex = v0x - ppx, ey = v0y - ppy;
                        fac_j[ size_t( nb - 1 ) * ar.n + i0 ] = id_de( ar, get_c( nb - 1 ) );
                        fac_l[ size_t( nb - 1 ) * ar.n + i0 ] = TK( sqrt( ex * ex + ey * ey ) );
                    }
                }
            }
            // la densite image a son PROPRE parcours d'arete, plus bas : on lui laisse la mesure
            if constexpr ( RES64 ) if ( ! im.active() ) {
                if ( fac_j ) {
                    if ( nb > NF && fac_deb ) atomicAdd( fac_deb, 1 );
                    for ( int e = nb > 0 ? nb : 0; e < NF; ++e ) { fac_j[ size_t( e ) * ar.n + i0 ] = -1000000; fac_l[ size_t( e ) * ar.n + i0 ] = 0; }
                }
                pris = true;
            }
        }
    }

    // l'origine du repere des sommets : le germe si `CENTRE`, sinon rien. La densite, elle, vit
    // dans le carre unite -- c'est le seul endroit ou le repere du germe doit etre defait.
    const double ox = CENTRE || FIXE ? double( p0[ 0 ] ) : 0.0;
    const double oy = CENTRE || FIXE ? double( p0[ 1 ] ) : 0.0;

    // ---- LA DENSITE IMAGE : une marche par arete, qui rend la masse ET `integrale rho ds`
    if ( im.active() ) {
        double mas = 0;
        const double sref = nb > 0 ? im.ref( double( get_x( 0 ) ) + ox, double( get_y( 0 ) ) + oy ) : 0.0;
        for ( int e = 0; e < nb; ++e ) {              // les aretes : la masse ne depend pas de `NF`
            const int ee = e + 1 < nb ? e + 1 : 0;
            double mm, ll;
            arete_image( im, double( get_x( e ) ) + ox, double( get_y( e ) ) + oy,
                             double( get_x( ee ) ) + ox, double( get_y( ee ) ) + oy, sref, mm, ll );
            mas += mm;
            if ( fac_j && e < NF ) { fac_j[ size_t( e ) * ar.n + i0 ] = id_de( ar, get_c( e ) ); fac_l[ size_t( e ) * ar.n + i0 ] = TK( ll ); }
        }
        if ( fac_j )
            for ( int e = nb > 0 ? nb : 0; e < NF; ++e ) { fac_j[ size_t( e ) * ar.n + i0 ] = -1000000; fac_l[ size_t( e ) * ar.n + i0 ] = 0; }
        if ( nb > NF && fac_deb ) atomicAdd( fac_deb, 1 );
        if ( nb < 0 ) atomicAdd( deborde, 1 );
        res[ i0 ] = nb > 0 ? fabs( mas ) : 0.0;
        return;
    }

    // ---- LES FACETTES du polygone final ( `get_x` / `get_y` / `get_c` lisent registres ou queue )
    if ( fac_j && ! pris ) {
        if ( nb > NF && fac_deb ) atomicAdd( fac_deb, 1 );
        for ( int k = 0; k < NF; ++k ) {
            int j = -1000000;
            TK  l = 0;
            if ( k < nb ) {
                const int kk = k + 1 < nb ? k + 1 : 0;
                const TK dx = get_x( kk ) - get_x( k ), dy = get_y( kk ) - get_y( k );
                j = id_de( ar, get_c( k ) );
                l = sqrt( dx * dx + dy * dy );
            }
            fac_j[ size_t( k ) * ar.n + i0 ] = j;
            fac_l[ size_t( k ) * ar.n + i0 ] = l;
        }
    }
    double area = 0;
    if ( pris ) area = 0.5 * fabs( a2 );
    else if ( nb > 0 ) {
        double a = 0;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            const int j = i + 1 < nb ? i + 1 : 0;
            if ( i < nb ) a += double( x[ i ] ) * double( get_y( j ) ) - double( get_x( j ) ) * double( y[ i ] );
        }
        for ( int i = R; i < nb; ++i ) {
            const int j = i + 1 < nb ? i + 1 : 0;
            a += double( lx[ i ] ) * double( get_y( j ) ) - double( get_x( j ) ) * double( ly[ i ] );
        }
        area = 0.5 * fabs( a );
    }
    if ( nb < 0 ) atomicAdd( deborde, 1 );
    res[ i0 ] = area;
}

} // namespace sf::gpu
