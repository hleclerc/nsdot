#pragma once

// =====================================================================================
// LE PAQUET, 2D : PLUSIEURS CELLULES PAR VOIE, UN SEUL PARCOURS PAR WARP.
//
// `voies` ( Voies2D.cuh ) a montre que le noyau 2D n'est borne ni par les ALU ni par la memoire
// mais par la LATENCE de la chaine noeud -> test -> plan -> coupe, et par la divergence entre les
// groupes d'un warp qui parcourent chacun leur arbre. Ici :
//
//   * la voie `l` porte le sommet `l` de `K` CELLULES ( `K` cases de registres ), et un warp de
//     `G = 32 / V` groupes porte `C = G K` cellules CONSECUTIVES dans l'arbre, donc voisines ;
//   * le PARCOURS EST PARTAGE : une pile par warp, une boite est descendue si UNE cellule du
//     paquet peut encore etre coupee par elle ( `any` du warp ), et les deux fils d'un noeud sont
//     testes ENSEMBLE ( deux tests en vol au lieu d'un aller-retour par la pile ) ;
//   * les plans d'une feuille sont testes EN BLOC contre toutes les cellules : un `fma` par
//     ( germe, case ), un bit par ( germe, case ) qui coupe, une reduction OU du warp -- et seuls
//     les couples qui coupent passent par le code de coupe. Les cellules ne faisant que retrecir,
//     un bit a 0 le reste apres les coupes precedentes du bloc ( exact ), un bit a 1 peut etre
//     devenu inutile ( la coupe le voit, `m == 0` ) ;
//   * LE CODE DE COUPE EST SANS BRANCHE : execute par les 32 voies pour tous les groupes a la
//     fois ( `shfl` et `ballot` de largeur `V` sous le masque plein ), un groupe que ce plan ne
//     coupe pas garde ses valeurs par `select`. Du travail perdu, mais sur un noyau qui attend --
//     et plus aucune divergence, sauf l'excursion.
//
// L'EXCURSION est celle de `voies` : au-dela de `V` sommets une cellule se pose dans les tableaux
// locaux de la voie 0 de son groupe, qui coupe et teste en scalaire ; les transitions ( descente,
// remontee ) sont faites par tout le warp sous `any`, avec les groupes concernes en predicat.
//
// L'ordre des fils est la proximite au germe de la cellule MEDIANE du paquet : les coupes qui
// mordent le plus arrivent tot pour tout le paquet, a peu pres.
// =====================================================================================

#include "gpu/Fil2D.cuh"

namespace sf::gpu {

constexpr unsigned PLEIN2 = 0xffffffffu;

/// `V` voies par cellule ( 8, 16, 32 ), `K` cellules par voie.
/// `SIB` : les deux fils testes ensemble au moment de l'empilement ( `true` ), ou chaque noeud teste
/// a sa sortie de pile, contre la cellule telle qu'elle est alors ( `false`, la regle du CPU ).
template<bool POIDS, int MaxNb, int V, int K, bool SIB, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_paquet( Arbre<TK,2> ar, double *res, int *deborde, unsigned long long *stats ) {
    static_assert( V == 8 || V == 16 || V == 32, "un groupe est une fraction du warp" );
    constexpr int G = 32 / V, C = G * K;
    const int lane = threadIdx.x & 31, l = lane & ( V - 1 ), grp = lane / V, wp = threadIdx.x >> 5;
    const int wi = ( blockIdx.x * blockDim.x + threadIdx.x ) >> 5;
    const int k0 = wi * C;
    __shared__ int piles[ 4 ][ PILE ];
    int *pile = piles[ wp ];
    if ( k0 >= ar.n ) return;                            // uniforme dans le warp

    // ---- LES CELLULES DU GROUPE : `k0 + grp * K + slot`
    TK   x0[ K ], y0[ K ], w0[ K ];
    int  i0[ K ];
    bool act[ K ];
    TK   vx[ K ], vy[ K ];
    int  cid[ K ], nb[ K ];
    bool large[ K ];
#pragma unroll
    for ( int s = 0; s < K; ++s ) {
        const int k = k0 + grp * K + s;
        act[ s ] = k < ar.n;
        const int kk = act[ s ] ? k : 0;
        x0[ s ] = ar.c[ 0 ][ kk ]; y0[ s ] = ar.c[ 1 ][ kk ];
        w0[ s ] = POIDS ? ar.w[ kk ] : TK( 0 );
        i0[ s ] = ar.ids[ kk ];
        vx[ s ] = TK( l == 1 || l == 2 ); vy[ s ] = TK( l == 2 || l == 3 );
        cid[ s ] = l < 4 ? -1 - l : 0;
        nb[ s ] = act[ s ] ? 4 : 0;
        large[ s ] = false;
    }
    const int kc = k0 + C / 2 < ar.n ? k0 + C / 2 : ar.n - 1;
    const TK pc[ 2 ] = { ar.c[ 0 ][ kc ], ar.c[ 1 ][ kc ] };   // le centre du paquet, pour l'ordre

    unsigned n_coupes = 0, n_eff = 0, n_exc = 0, n_boites = 0;   // les compteurs, voie 0 de chaque groupe
    TK  lvx[ K ][ MaxNb ], lvy[ K ][ MaxNb ], ls[ MaxNb ];   // l'excursion, voie 0 du groupe
    int lcid[ K ][ MaxNb ];

    // ---- « une cellule du paquet peut-elle encore etre coupee par un germe de cette boite ? »
    auto peut = [ & ]( const Noeud<TK,2> &nd ) -> bool {
        bool mienne = false;
#pragma unroll
        for ( int s = 0; s < K; ++s ) {
            const TK p0[ 2 ] = { x0[ s ], y0[ s ] };
            if ( ! large[ s ] ) {
                const TK v[ 2 ] = { vx[ s ], vy[ s ] };
                mienne |= l < nb[ s ] && bilan_sommet<POIDS>( nd, v, p0, w0[ s ] ) <= TK( 0 );
            } else if ( l == 0 && nb[ s ] > 0 )
                mienne |= peut_couper2<POIDS>( nd, p0, w0[ s ], nb[ s ], lvx[ s ], lvy[ s ] );
        }
        return __any_sync( PLEIN2, mienne );
    };

    // ---- LA COUPE de la case `S` par le germe `q`, pour tous les groupes, sans branche
    auto coupe = [ & ]<int S>( int q ) {
        const TK xj = ar.c[ 0 ][ q ], yj = ar.c[ 1 ][ q ];
        Plan2<TK> p;
        p.dx  = xj - x0[ S ];
        p.dy  = yj - y0[ S ];
        p.off = TK( 0.5 ) * ( p.dx * ( xj + x0[ S ] ) + p.dy * ( yj + y0[ S ] ) );
        if constexpr ( POIDS ) p.off += TK( 0.5 ) * ( w0[ S ] - ar.w[ q ] );
        p.id  = ar.ids[ q ];

        const int n0 = nb[ S ];
        const TK  s  = p.dx * vx[ S ] + p.dy * vy[ S ] - p.off;
        // `valid` est BORNE AUX `V` BITS DU GROUPE : en excursion `n0` depasse `V`, et un masque plus
        // large ramasserait les bits des groupes voisins dans le `ballot` ( mesure : des coupes
        // fantomes, 5 % de cellules vides ) ; une cellule finie ou en excursion a un masque nul
        constexpr unsigned plein_v = V == 32 ? 0xffffffffu : ( 1u << V ) - 1;
        const unsigned valid = ( n0 <= 0 || large[ S ] ) ? 0u : ( n0 >= V ? plein_v : ( 1u << n0 ) - 1 );
        const unsigned m = ( __ballot_sync( PLEIN2, l < n0 && ! large[ S ] && s > TK( 0 ) ) >> ( V * grp ) ) & valid;
        const bool vide = valid != 0 && m == valid;

        // les deux bouts de la plage, les intersections, le remontage : calcules par tous, gardes
        // par les groupes que le plan coupe vraiment
        const int nm1 = n0 > 0 ? n0 - 1 : 0;
        const unsigned prev = ( ( m << 1 ) | ( m >> nm1 ) ) & valid;
        const unsigned next = ( ( m >> 1 ) | ( m << nm1 ) ) & valid;
        const int i1 = __ffs( m & ~prev ) - 1;
        const int j2 = __ffs( m & ~next ) - 1;
        const int j0 = i1 > 0 ? i1 - 1 : n0 - 1;
        const int j3 = j2 + 1 < n0 ? j2 + 1 : 0;
        const int nb_in = n0 - __popc( m );
        const int nn = nb_in + 2;
        const int anc = ( l == 1 ? j3 : j0 ) & ( V - 1 ), oth = ( l == 1 ? j2 : i1 ) & ( V - 1 );
        const TK vax = __shfl_sync( PLEIN2, vx[ S ], anc, V ), vox = __shfl_sync( PLEIN2, vx[ S ], oth, V );
        const TK vay = __shfl_sync( PLEIN2, vy[ S ], anc, V ), voy = __shfl_sync( PLEIN2, vy[ S ], oth, V );
        const TK sa  = __shfl_sync( PLEIN2, s,  anc, V ),      so  = __shfl_sync( PLEIN2, s,  oth, V );
        const TK t   = sa / ( sa - so );
        const TK pcx = vax + ( vox - vax ) * t, pcy = vay + ( voy - vay ) * t;
        int og = j3 + l; og = og >= n0 ? og - n0 : og;
        TK  nvx = __shfl_sync( PLEIN2, vx[ S ],  og & ( V - 1 ), V );
        TK  nvy = __shfl_sync( PLEIN2, vy[ S ],  og & ( V - 1 ), V );
        int nid = __shfl_sync( PLEIN2, cid[ S ], og & ( V - 1 ), V );
        const TK  ax  = __shfl_sync( PLEIN2, pcx, 0, V ), ay = __shfl_sync( PLEIN2, pcy, 0, V );
        const TK  bx  = __shfl_sync( PLEIN2, pcx, 1, V ), by = __shfl_sync( PLEIN2, pcy, 1, V );
        const int bid = __shfl_sync( PLEIN2, cid[ S ], j2 & ( V - 1 ), V );
        nvx = l == nb_in ? ax : ( l == nb_in + 1 ? bx : nvx );
        nvy = l == nb_in ? ay : ( l == nb_in + 1 ? by : nvy );
        nid = l == nb_in ? p.id : ( l == nb_in + 1 ? bid : nid );

        const bool coupee = m != 0 && ! vide && nn <= V;
        const bool descend = m != 0 && ! vide && nn > V;   // l'excursion part de la
        ++n_coupes; n_eff += coupee || vide || descend || large[ S ]; n_exc += descend;
        if ( coupee ) { vx[ S ] = nvx; vy[ S ] = nvy; cid[ S ] = nid; nb[ S ] = nn; }
        if ( vide ) nb[ S ] = 0;

        // ---- L'EXCURSION : les transitions par tout le warp, les groupes concernes en predicat
        if ( __any_sync( PLEIN2, descend ) ) {
#pragma unroll
            for ( int i = 0; i < V; ++i ) {
                const TK  x = __shfl_sync( PLEIN2, vx[ S ], i, V ), y = __shfl_sync( PLEIN2, vy[ S ], i, V );
                const int c = __shfl_sync( PLEIN2, cid[ S ], i, V );
                if ( descend && l == 0 ) { lvx[ S ][ i ] = x; lvy[ S ][ i ] = y; lcid[ S ][ i ] = c; }
            }
            if ( descend ) large[ S ] = true;
        }
        if ( __any_sync( PLEIN2, large[ S ] ) ) {
            int r = 0;
            if ( large[ S ] && l == 0 && nb[ S ] > 0 ) r = coupe2<MaxNb>( lvx[ S ], lvy[ S ], lcid[ S ], nb[ S ], p, ls );
            r = __shfl_sync( PLEIN2, r, 0, V );
            if ( large[ S ] && nb[ S ] > 0 ) nb[ S ] = r;
            const bool retour = large[ S ] && r > 0 && r <= V;
            if ( __any_sync( PLEIN2, retour ) ) {
#pragma unroll
                for ( int i = 0; i < V; ++i ) {
                    const TK  x = __shfl_sync( PLEIN2, l == 0 && i < r ? lvx[ S ][ i ]  : TK( 0 ), 0, V );
                    const TK  y = __shfl_sync( PLEIN2, l == 0 && i < r ? lvy[ S ][ i ]  : TK( 0 ), 0, V );
                    const int c = __shfl_sync( PLEIN2, l == 0 && i < r ? lcid[ S ][ i ] : 0,       0, V );
                    if ( retour && l == i ) { vx[ S ] = x; vy[ S ] = y; cid[ S ] = c; }
                }
                if ( retour ) large[ S ] = false;
            }
        }
    };

    // ---- LE PARCOURS, partage : la racine sans test, puis les deux fils de chaque noeud ensemble
    int haut = 1;
    if ( lane == 0 ) pile[ 0 ] = 0;
    __syncwarp();
    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,2> nd = ar.nodes[ h ];

        if constexpr ( ! SIB ) {
            if ( h != 0 && ! peut( nd ) ) continue;
            ++n_boites;
        }
        if ( nd.right >= 0 ) {
            const int g = h + 1, dr = nd.right;
            const Noeud<TK,2> ng = ar.nodes[ g ], ndr = ar.nodes[ dr ];
            const bool gp = proximite( ng, pc ) <= proximite( ndr, pc );
            const int  n1 = gp ? dr : g, n2 = gp ? g : dr;       // le proche en dernier : depile en premier
            bool p1 = true, p2 = true;
            if constexpr ( SIB ) {
                const bool pg = peut( ng ), pd = peut( ndr );
                n_boites += 2;
                p1 = gp ? pd : pg; p2 = gp ? pg : pd;
            }
            if ( lane == 0 ) { int t = haut; if ( p1 ) pile[ t++ ] = n1; if ( p2 ) pile[ t++ ] = n2; }
            haut += int( p1 ) + int( p2 );
            __syncwarp();
            continue;
        }

        // ---- UNE FEUILLE : ses germes testes EN BLOC, un bit par ( germe, case )
        constexpr int LC = 64 / K;                       // germes par bloc
        for ( int beg = nd.beg; beg < nd.end; beg += LC ) {
            const int fin = beg + LC < nd.end ? beg + LC : nd.end;
            unsigned long long bits = 0;
            for ( int q = beg; q < fin; ++q ) {
                const TK xj = ar.c[ 0 ][ q ], yj = ar.c[ 1 ][ q ];
                const TK wj = POIDS ? ar.w[ q ] : TK( 0 );
#pragma unroll
                for ( int s = 0; s < K; ++s ) {
                    const TK dx = xj - x0[ s ], dy = yj - y0[ s ];
                    TK off = TK( 0.5 ) * ( dx * ( xj + x0[ s ] ) + dy * ( yj + y0[ s ] ) );
                    if constexpr ( POIDS ) off += TK( 0.5 ) * ( w0[ s ] - wj );
                    bool out;
                    if ( ! large[ s ] )
                        out = l < nb[ s ] && dx * vx[ s ] + dy * vy[ s ] - off > TK( 0 );
                    else {
                        out = false;
                        if ( l == 0 )
                            for ( int i = 0; i < nb[ s ]; ++i ) out |= dx * lvx[ s ][ i ] + dy * lvy[ s ][ i ] - off > TK( 0 );
                    }
                    bits |= ( unsigned long long ) out << ( ( q - beg ) * K + s );
                }
            }
#pragma unroll
            for ( int d = 16; d >= 1; d /= 2 ) bits |= __shfl_xor_sync( PLEIN2, bits, d );
            if ( ! bits ) continue;                      // le cas frequent : rien de ce bloc ne coupe
            for ( int q = beg; q < fin; ++q ) {
                const unsigned long long b = bits >> ( ( q - beg ) * K );
                if ( ! ( b & ( ( 1ull << K ) - 1 ) ) ) continue;
                // ( `if constexpr` sur une case : la boucle est deroulee a la compilation )
                if ( b & 1 ) coupe.template operator()<0>( q );
                if constexpr ( K > 1 ) if ( b & 2 ) coupe.template operator()<1>( q );
                if constexpr ( K > 2 ) if ( b & 4 ) coupe.template operator()<2>( q );
                if constexpr ( K > 3 ) if ( b & 8 ) coupe.template operator()<3>( q );
            }
        }
    }

    if ( stats && lane == 0 ) {                          // par warp : `C` cellules
        atomicAdd( stats, ( unsigned long long ) n_coupes * G ); atomicAdd( stats + 1, ( unsigned long long ) n_eff * G );
        atomicAdd( stats + 2, ( unsigned long long ) n_exc * G ); atomicAdd( stats + 3, ( unsigned long long ) n_boites * C );
    }

    // ---- LES AIRES, une par ( groupe, case )
#pragma unroll
    for ( int s = 0; s < K; ++s ) {
        double area = 0;
        const int n0 = nb[ s ];
        if ( ! large[ s ] ) {
            const int j = l + 1 < n0 ? l + 1 : 0;
            const TK xj = __shfl_sync( PLEIN2, vx[ s ], j & ( V - 1 ), V ), yj = __shfl_sync( PLEIN2, vy[ s ], j & ( V - 1 ), V );
            double a = l < n0 ? double( vx[ s ] ) * double( yj ) - double( xj ) * double( vy[ s ] ) : 0.0;
#pragma unroll
            for ( int d = V / 2; d >= 1; d /= 2 ) a += __shfl_xor_sync( PLEIN2, a, d, V );
            area = 0.5 * fabs( a );
        } else if ( l == 0 && n0 > 0 ) {
            double a = 0;
            for ( int i = 0; i < n0; ++i ) {
                const int j = i + 1 < n0 ? i + 1 : 0;
                a += double( lvx[ s ][ i ] ) * double( lvy[ s ][ j ] ) - double( lvx[ s ][ j ] ) * double( lvy[ s ][ i ] );
            }
            area = 0.5 * fabs( a );
        }
        if ( l == 0 && act[ s ] ) {
            if ( n0 < 0 ) atomicAdd( deborde, 1 );
            res[ i0[ s ] ] = area;
        }
    }
}

} // namespace sf::gpu
