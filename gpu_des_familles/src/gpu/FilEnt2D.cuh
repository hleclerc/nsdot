#pragma once

// =====================================================================================
// LES SOMMETS EN ENTIERS 32 BITS, ET LE PREDICAT DE COUPE EXACT. `filmsk` a l'identique -- memes
// masques de role, meme barillet, meme resolution finale -- mais la cellule vit sur une GRILLE
// ENTIERE au lieu de vivre dans le flottant du noyau.
//
// = CE QUE LA GRILLE ENTIERE CHANGE, ET CE QU'ELLE NE CHANGE PAS
//
// Le sommet est un `int` a l'echelle `ECH_FIXE = 2^30`, compte a partir du germe. La coupe
// `i | j` s'ecrit alors SANS UN SEUL ARRONDI :
//
//     2 ( du . V )  <=  |du|^2 + ( W0 - Wj )        du = uj - u0,  W = w . ECH_FIXE^2
//
// -- `du` est une soustraction entiere exacte, `du . V` tient dans un `long long` ( deux
// `IMAD.WIDE`, une addition ), et le poids est sur la grille `2^-60` parce que `w` a la dimension
// d'une LONGUEUR AU CARRE : sa grille naturelle est le CARRE de celle des positions, et
// `|du|^2 + dW` est homogene sans facteur d'echelle. C'est la reponse a « ou vivent les `w` ».
//
// LES BORNES, une fois pour toutes : `|du| <= 2^30` et `|V| <= 2^30` donnent `2 du.V <= 2^62`,
// `|du|^2 <= 2^61`, et le poids est ECRETE a `2^61`. Tout reste sous `2^63`, et la comparaison se
// fait SANS SOUSTRACTION -- `2 l > off` -- pour qu'aucun intermediaire ne deborde. Un plan ecrete
// est « tres loin », ce qui est la bonne degenerescence.
//
// = LE PREDICAT EXACT NE REND PAS LA CELLULE EXACTE
//
// Il rend le POLYGONE COHERENT. Le masque `m` dit desormais LA VERITE sur les sommets tels qu'ils
// sont stockes : plus d'ecart possible entre « le bit dit dehors » et « la geometrie dit dedans »,
// donc l'arc exterieur reste contigu, le polygone reste convexe, l'ordre cyclique reste juste.
// C'est de la ROBUSTESSE, pas de la precision -- et c'est exactement ce qu'on peut acheter avec
// des entiers ( cf. le README § 4 : predicats exacts, constructions approchees ).
//
// L'INTERPOLATION RESTE APPROCHEE, deliberement. `s0 / ( s0 - s1 )` se calcule en `float` et le
// point tombe sur la grille par arrondi. L'exiger exacte demanderait une division entiere de 71
// bits par 40 -- que la carte n'a pas -- et ca ne servirait a rien : LES SOMMETS SONT RESOLUS
// depuis leurs deux plans a la fin ( `Arbre.cuh` ), donc le sommet intermediaire ne sert qu'a
// prendre des decisions, et `2^-30` de grille sur une cellule de `2^-10` lui laisse vingt bits.
//
// = CE QUE LA GRILLE ENTIERE COUTE, ET POURQUOI ELLE N'EST PAS GRATUITE
//
// La grille est celle de LA BOITE, pas celle de la cellule -- le polygone part du carre unite,
// donc on ne peut pas la resserrer. Le sommet porte donc `2^-31` d'erreur ABSOLUE, quand le meme
// sommet en `float` DANS LE REPERE DU GERME en porte `h . 2^-24`. Les deux se croisent a
// `h = 2^-7`, soit `n ~ 1.6e4` : EN DESSOUS l'entier gagne, AU-DESSUS le repere du germe gagne,
// et a `n = 1e6` il gagne d'un facteur huit. C'est le prix de l'echelle fixe, et c'est pourquoi
// cette variante ne vaut que par son predicat -- la mesure, elle, vient de la resolution.
//
// L'ELAGAGE reste en `float` : c'est un test CONSERVATIF ( « un germe de cette boite pourrait-il
// retrancher ce sommet ? » ), donc l'exactitude n'y ajoute rien, et la boite du noeud n'existe pas
// en entiers. Le sommet y est converti au vol, deux instructions par coordonnee.
// =====================================================================================

#include "gpu/FilMsk2D.cuh"

namespace sf::gpu {

/// le plan bissecteur TOUT EN ENTIERS ( `off` porte deja `|du|^2 + dW` )
struct Plan2e { int dx, dy; long long off; int id; };

/// l'echelle des poids : le CARRE de celle des positions, par homogeneite
constexpr double ECH_POIDS  = double( ECH_FIXE ) * double( ECH_FIXE );   // 2^60
constexpr double PLAF_POIDS = 2.0 * ECH_POIDS;                           // 2^61, l'ecretage

template<bool POIDS, class TK>
__device__ __forceinline__ Plan2e bissect2e( const Arbre<TK,2> &ar, int q, int ux0, int uy0, double w0 ) {
    Plan2e p;
    p.dx  = ar.u[ 0 ][ q ] - ux0;                        // soustraction entiere EXACTE
    p.dy  = ar.u[ 1 ][ q ] - uy0;
    p.off = ( long long ) p.dx * p.dx + ( long long ) p.dy * p.dy;
    if constexpr ( POIDS ) {
        const double dw = ( w0 - ar.w64[ q ] ) * ECH_POIDS;   // LA DIFFERENCE, arrondie une fois
        p.off += ( long long ) fmin( fmax( dw, -PLAF_POIDS ), PLAF_POIDS );
    }
    p.id  = q;
    return p;
}

/// UNE COUPE sur des sommets ENTIERS -- `filmsk`'s `coupe_msk`, avec le test exact. Rend le
/// nouveau nombre de sommets : `nb` si le plan ne coupe pas, `0` si la cellule se vide, `-1` si
/// elle deborde `R`.
template<int R>
__device__ __forceinline__ int coupe_ent( const Plan2e &p, int nb, int ( &x )[ R ], int ( &y )[ R ], int ( &c )[ R ] ) {
    constexpr int SUR = 3;

    // ---- LE MASQUE, EXACT. `2 l > off` : les deux membres tiennent dans `long long` et on ne
    //      les soustrait PAS, donc rien ne peut deborder quel que soit l'eloignement du plan.
    unsigned m = 0;
#pragma unroll
    for ( int i = 0; i < R; ++i ) {
        if ( i >= SUR && i >= nb ) break;
        const long long l = ( long long ) p.dx * x[ i ] + ( long long ) p.dy * y[ i ];
        m |= unsigned( 2 * l > p.off ) << i;
    }
    if ( PROBABLE( ! m ) )
        return nb;
    const unsigned valid = ( 1u << nb ) - 1;
    if ( IMPROBABLE( m == valid ) )
        return 0;

    // ---- LES QUATRE MASQUES DE ROLE, un seul bit chacun ( identique a `coupe_msk` )
    const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
    const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
    const unsigned r0 = ~m & next & valid;
    const unsigned r1 =  m & ~prev;
    const unsigned r2 =  m & ~next;
    const unsigned r3 = ~m & prev & valid;
    const int j0 = __ffs( int( r0 ) ) - 1, i1 = __ffs( int( r1 ) ) - 1;
    const int j2 = __ffs( int( r2 ) ) - 1, j3 = __ffs( int( r3 ) ) - 1;
    const int nb_out = __popc( m );
    const int nn = nb - nb_out + 2;
    if ( IMPROBABLE( nn > R ) )
        return -1;

    int x0v = x[ 0 ], y0v = y[ 0 ], x1v = x[ 0 ], y1v = y[ 0 ];
    int x2v = x[ 0 ], y2v = y[ 0 ], x3v = x[ 0 ], y3v = y[ 0 ];
    int bid = c[ 0 ];
#pragma unroll
    for ( int i = 1; i < R; ++i ) {
        if ( i >= SUR && i >= nb ) break;
        x0v = i == j0 ? x[ i ] : x0v; y0v = i == j0 ? y[ i ] : y0v;
        x1v = i == i1 ? x[ i ] : x1v; y1v = i == i1 ? y[ i ] : y1v;
        x2v = i == j2 ? x[ i ] : x2v; y2v = i == j2 ? y[ i ] : y2v; bid = i == j2 ? c[ i ] : bid;
        x3v = i == j3 ? x[ i ] : x3v; y3v = i == j3 ? y[ i ] : y3v;
    }

    // ---- LES DEUX POINTS CREES. `s` est ici PETIT ( les quatre sommets encadrent le plan, donc
    //      le plan coupe la cellule ) : la soustraction est sans danger, et la conversion en
    //      `float` ne coute que sa precision relative -- celle-la meme qu'avait `filmsk`.
    auto sig = [ & ]( int vx, int vy ) { return 2 * ( ( long long ) p.dx * vx + ( long long ) p.dy * vy ) - p.off; };
    const long long s0 = sig( x0v, y0v ), s1 = sig( x1v, y1v );
    const long long s2 = sig( x2v, y2v ), s3 = sig( x3v, y3v );
    const float ta = float( s0 ) / float( s0 - s1 ), tb = float( s3 ) / float( s3 - s2 );
    // les differences passent par `long long` : deux sommets opposes du carre en sont a `2^31`
    const int pax = x0v + __float2int_rn( float( ( long long ) x1v - x0v ) * ta );
    const int pay = y0v + __float2int_rn( float( ( long long ) y1v - y0v ) * ta );
    const int pbx = x3v + __float2int_rn( float( ( long long ) x2v - x3v ) * tb );
    const int pby = y3v + __float2int_rn( float( ( long long ) y2v - y3v ) * tb );

    // ---- LE REMONTAGE : un seul barillet, comme `filmsk`
    const bool boucle = r1 > r2;
    const int  a = boucle ? 0 : i1;
    const int  e = boucle ? j3 - 1 : nb_out - 1;
    int ux[ R + 1 ], uy[ R + 1 ], uc[ R + 1 ];
    ux[ 0 ] = x[ 0 ]; uy[ 0 ] = y[ 0 ]; uc[ 0 ] = c[ 0 ];
#pragma unroll
    for ( int o = 1; o < R + 1; ++o ) { ux[ o ] = x[ o - 1 ]; uy[ o ] = y[ o - 1 ]; uc[ o ] = c[ o - 1 ]; }
#pragma unroll
    for ( int b = 1; b < R + 1; b *= 2 ) {
        const bool on = e & b;
#pragma unroll
        for ( int o = 0; o + b < R + 1; ++o ) { ux[ o ] = on ? ux[ o + b ] : ux[ o ]; uy[ o ] = on ? uy[ o + b ] : uy[ o ]; uc[ o ] = on ? uc[ o + b ] : uc[ o ]; }
    }
#pragma unroll
    for ( int o = 0; o < R; ++o ) {
        if ( o >= SUR && o >= nn ) break;
        x[ o ] = o < a ? x[ o ] : ( o == a ? pax  : ( o == a + 1 ? pbx : ux[ o ] ) );
        y[ o ] = o < a ? y[ o ] : ( o == a ? pay  : ( o == a + 1 ? pby : uy[ o ] ) );
        c[ o ] = o < a ? c[ o ] : ( o == a ? p.id : ( o == a + 1 ? bid : uc[ o ] ) );
    }
    return nn;
}

/// LE NOYAU. `RES64` : la mesure prise sur le sommet RESOLU, en `double`, comme `filmsk8m` --
/// c'est elle qui rattrape la grille de `2^-30`, et sans elle la variante entiere n'aurait aucun
/// sens. `RAFF64` : les plans de la resolution relus en virgule fixe 64 bits ( la construction,
/// elle, decide sur les positions quantifiees a `2^-30` : c'est la discipline EGC, on quantifie
/// l'entree UNE FOIS puis on est exact dessus ).
template<bool POIDS, int BSM, bool RES64 = false, int RAFF64 = 64, class TK = float>
__global__ void __launch_bounds__( 128, BSM ) noyau2_filent( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb,
                                                             int *fac_j = nullptr, TK *fac_l = nullptr, int NF = 0,
                                                             bool raff = false ) {
    constexpr int R = 8, SUR = 3;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= ar.n ) return;
    const int u0[ 2 ] = { ar.u[ 0 ][ k ], ar.u[ 1 ][ k ] };
    const long long g0[ 2 ] = { RAFF64 == 64 ? ar.u64[ 0 ][ k ] : 0, RAFF64 == 64 ? ar.u64[ 1 ][ k ] : 0 };
    const TK p0[ 2 ] = { TK( u0[ 0 ] ) * TK( INV_FIXE ), TK( u0[ 1 ] ) * TK( INV_FIXE ) };
    const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
    const double w0d = POIDS ? ar.w64[ k ] : 0.0;
    const int i0 = ar.ids[ k ];

    // le carre unite, dans le repere du germe : EXACT, le sommet tombe pile sur `ECH_FIXE`
    int x[ R ], y[ R ], c[ R ];
#pragma unroll
    for ( int i = 0; i < R; ++i ) {
        x[ i ] = ( i == 1 || i == 2 ? ECH_FIXE : 0 ) - u0[ 0 ];
        y[ i ] = ( i == 2 || i == 3 ? ECH_FIXE : 0 ) - u0[ 1 ];
        c[ i ] = i < 4 ? -1 - i : 0;
    }
    int nb = 4;

    int pile[ PILE ];
    int haut = 0;
    pile[ haut++ ] = 0;
    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,2> nd = ar.nodes[ h ];

        // l'elagage, CONSERVATIF donc laisse en flottant : le sommet y est converti au vol
        bool peut = false;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            if ( i >= SUR && i >= nb ) break;
            const TK v[ 2 ] = { TK( x[ i ] ) * TK( INV_FIXE ), TK( y[ i ] ) * TK( INV_FIXE ) };
            peut |= bilan_sommet_c<POIDS>( nd, v, p0, w0 ) <= TK( 0 );
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
            nb = coupe_ent( bissect2e<POIDS>( ar, q, u0[ 0 ], u0[ 1 ], w0d ), nb, x, y, c );
            if ( IMPROBABLE( nb <= 0 ) ) goto fin;
        }
    }

fin:
    // ---- LES SOMMETS RESOLUS, ET LA MESURE PRISE DEDANS ( voir `FilMsk2D.cuh` )
    double a2 = 0;
    bool   pris = false;
    if ( raff ) {
        if ( nb >= 3 ) {
            double ax, ay, ao;
            plan_relu<POIDS,RAFF64>( ar, selR( c, nb - 1 ), u0, g0, p0, w0d, ax, ay, ao );
            double v0x = 0, v0y = 0, ppx = 0, ppy = 0;
#pragma unroll
            for ( int i = 0; i < R; ++i ) {
                if ( i >= SUR && i >= nb ) break;
                double bx, by, bo, vx, vy;
                plan_relu<POIDS,RAFF64>( ar, c[ i ], u0, g0, p0, w0d, bx, by, bo );
                if ( ! croise2( ax, ay, ao, bx, by, bo, vx, vy ) ) { vx = double( x[ i ] ) * INV_FIXE; vy = double( y[ i ] ) * INV_FIXE; }
                ax = bx; ay = by; ao = bo;
                if constexpr ( RES64 ) {
                    if ( i == 0 ) { v0x = vx; v0y = vy; }
                    else {
                        a2 += ppx * vy - vx * ppy;
                        if ( fac_j ) {
                            const double ex = vx - ppx, ey = vy - ppy;
                            fac_j[ size_t( i - 1 ) * ar.n + i0 ] = id_de( ar, c[ i - 1 ] );
                            fac_l[ size_t( i - 1 ) * ar.n + i0 ] = TK( sqrt( ex * ex + ey * ey ) );
                        }
                    }
                    ppx = vx; ppy = vy;
                } else {                                 // sans `RES64` le sommet redescend sur la grille
                    x[ i ] = __double2int_rn( vx * ECH_FIXE );
                    y[ i ] = __double2int_rn( vy * ECH_FIXE );
                }
            }
            if constexpr ( RES64 ) {
                a2 += ppx * v0y - v0x * ppy;
                if ( fac_j ) {
                    const double ex = v0x - ppx, ey = v0y - ppy;
                    fac_j[ size_t( nb - 1 ) * ar.n + i0 ] = id_de( ar, selR( c, nb - 1 ) );
                    fac_l[ size_t( nb - 1 ) * ar.n + i0 ] = TK( sqrt( ex * ex + ey * ey ) );
                }
            }
        }
        if constexpr ( RES64 ) {
            if ( fac_j )
#pragma unroll
                for ( int e = 0; e < R; ++e )
                    if ( e >= nb ) { fac_j[ size_t( e ) * ar.n + i0 ] = -1000000; fac_l[ size_t( e ) * ar.n + i0 ] = 0; }
            pris = true;
        }
    }

    double mes = 0;
    if ( ! pris ) {
        if ( fac_j ) {
#pragma unroll
            for ( int e = 0; e < R; ++e ) {
                int j = -1000000;
                TK  l = 0;
                if ( e < nb ) {
                    j = id_de( ar, c[ e ] );
                    const int ee = e + 1 < nb ? e + 1 : 0;
                    const double dx = ( double( selR( x, ee ) ) - double( x[ e ] ) ) * INV_FIXE,   // `2^31` ne tient pas dans un `int`
                                 dy = ( double( selR( y, ee ) ) - double( y[ e ] ) ) * INV_FIXE;
                    l = TK( sqrt( dx * dx + dy * dy ) );
                }
                fac_j[ size_t( e ) * ar.n + i0 ] = j;
                fac_l[ size_t( e ) * ar.n + i0 ] = l;
            }
        }
        // L'AIRE PAR LE LACET, EXACTE. Chaque produit croise tient dans un `long long` ( `2^61` )
        // mais leur SOMME n'y tient pas -- la boite fait deja `2^60` d'aire et rien ne garantit
        // que les sommes partielles restent bornees par l'aire finale. L'accumulateur est donc en
        // `__int128`, une fois par cellule : c'est le seul endroit du noyau qui en demande, et
        // c'est ce que « l'aire d'un polygone a sommets entiers est un demi-entier » vaut ici.
        if ( nb > 0 ) {
            __int128 a = 0;
            int xp = x[ 0 ], yp = y[ 0 ];
#pragma unroll
            for ( int i = 1; i < R; ++i ) {
                if ( i >= SUR && i >= nb ) break;
                a += ( __int128 ) ( ( long long ) xp * y[ i ] ) - ( __int128 ) ( ( long long ) x[ i ] * yp );
                xp = x[ i ]; yp = y[ i ];
            }
            a += ( __int128 ) ( ( long long ) xp * y[ 0 ] ) - ( __int128 ) ( ( long long ) x[ 0 ] * yp );
            if ( a < 0 ) a = -a;
            mes = 0.5 * double( a ) * ( INV_FIXE * INV_FIXE );
        }
    } else {
        mes = 0.5 * fabs( a2 );
    }
    if ( fac_j )
        for ( int e = R; e < NF; ++e ) { fac_j[ size_t( e ) * ar.n + i0 ] = -1000000; fac_l[ size_t( e ) * ar.n + i0 ] = 0; }

    if ( nb < 0 ) liste_deb[ atomicAdd( deborde, 1 ) ] = k;
    res[ i0 ] = mes;
}

} // namespace sf::gpu
