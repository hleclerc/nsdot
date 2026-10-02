#pragma once

// =====================================================================================
// `filnrm` / `filrot` AVEC DES MASQUES DE ROLE ET UN SEUL BARILLET ( idee de H. L. ).
// Les registres restent TRIES -- les sommets sont toujours en `0 .. nb - 1` -- et deux choses
// changent par rapport a `filrot` :
//
//   1. LES QUATRE SOMMETS DE LA FRONTIERE sortent de QUATRE MASQUES DE ROLE de huit bits,
//      fabriques en quatre instructions a partir de `m`, `prev` et `next`. Chacun n'a qu'UN SEUL
//      BIT, puisque la plage exterieure est un arc cyclique contigu :
//          r0 = ~m & next    ( dedans, le suivant dehors    -> `v_j0` )
//          r1 =  m & ~prev   ( dehors, le precedent dedans  -> `v_i1`, l'entree )
//          r2 =  m & ~next   ( dehors, le suivant dedans    -> `v_j2` )
//          r3 = ~m & prev    ( dedans, le precedent dehors  -> `v_j3` )
//      Un `__ffs` donne l'indice, et « la plage boucle » se lit `r1 > r2` ( les deux masques sont
//      one-hot : les comparer, c'est comparer `i1` et `j2` ). `filrot` faisait le meme travail en
//      deux `__ffs` puis quatre corrections cycliques ( `i1 ? i1 - 1 : nb - 1` ... ).
//
//      ( La variante ou les quatre masques servaient DIRECTEMENT a cueillir `x`, `y` et `c` par
//        des `et` / `ou` pleine largeur a ete ecrite et mesuree : +11 % d'instructions. Le partage
//        entre les trois tableaux existait deja -- ptxas met un seul `ISETP.EQ` par case en
//        facteur de trois `SEL` -- et un masque plein coute deux a trois instructions la ou un
//        predicat en coute une. Voir README § 4. )
//
//   2. LE REMONTAGE TIENT EN UN SEUL BARILLET. La sortie est
//          new[ o ] = o < a ? old[ o ] : o == a ? A : o == a + 1 ? B : old[ o + d ]
//      avec `( a, d ) = ( 0, j3 - 2 )` si la plage boucle, `( i1, nb_out - 2 )` sinon. Le `d`
//      peut valoir -1 ( une seule coupe sortante : le cas le plus frequent ), et `filrot` payait
//      ce cas par une copie decalee a droite -- un `select` de plus par case et par tableau.
//      Ici on decale A PRIORI d'une case, ce qui est GRATUIT ( un renommage de registres a la
//      compilation ) : on travaille sur `u[ k ] = old[ k - 1 ]`, neuf cases, et le barillet part
//      de `e = d + 1 >= 0`. Mesure : -8 % d'instructions et -8 % de temps contre `filrot8`.
//
//   3. LES LECTURES INDEXEES SONT MUTUALISEES ET BORNEES. Neuf `selR` ( `x`, `y` pour les quatre
//      sommets, plus le `cid` de `v_j2` ) balayaient chacun les `R` cases SANS s'arreter a `nb`.
//      Ils tiennent maintenant dans UNE SEULE boucle, avec la meme sortie a `nb` que la premiere
//      passe : quatre compares partages par neuf `select`, et rien au-dela de `nb`. L'aire fait de
//      meme ( `aire_triee`, le sommet precedent garde dans un registre ).
//
// Trois tableaux temporaires de neuf cases, contre les huit de huit de `filnrm` : 96 registres en
// `double` et 58 en `float`, contre 128 et 74 -- LE PLUS PETIT NOYAU DE LA FAMILLE, a +5 % de
// `FIXE` : les positions sont lues en VIRGULE FIXE ( `Arbre.cuh` ) -- `dx` devient une
// soustraction entiere exacte. `FIXE = 32` : pas de 9.3e-10 pour les memes quatre octets.
// `FIXE = 64` : pas de 2.2e-16, soit `dx` exact a la precision du `float` LUI-MEME ( la conversion
// arrondit a 24 bits RELATIFS a `dx` ), au prix de huit octets par coordonnee. Implique `CENTRE`,
// qui seul leur donne leur sens.
//
// `CENTRE` : la cellule vit dans le REPERE DU GERME ( les sommets comptent a partir de `p0` ).
// Rien ne change dans la coupe -- l'aire par le lacet est invariante par translation, et
// `coupe_msk` ne voit que des differences -- seuls changent le carre de depart ( decale de `p0` ),
// le plan ( `bissect2c` : `off = 1/2 ( dx^2 + dy^2 )` ) et l'elagage ( `bilan_sommet_c` ). Le but
// est la PRECISION en `float` : voir `Arbre.cuh` et le README § 3 bis.
//
// `FACETTES` : le noyau rend AUSSI, pour chaque cellule, ses facettes -- de quoi assembler la
// hessienne du Newton ( `solver/Laplacien.h` : `c_ij = |facette| / ( 2 |p_i - p_j| )` ). Elles
// sont deja dans la cellule : l'arete `k` va du sommet `k` au sommet `k + 1` et porte le `cid`
// `c[ k ]` -- identifiant du voisin s'il est positif, cote de la boite s'il vaut `-1 - d`. On
// ecrit la longueur et le voisin dans `R` cases par cellule, en SoA ( `[ case * n + id ]`, donc
// des voies consecutives ecrivent des adresses consecutives ), `-1` pour les cases inutilisees.
//
// `BSM` force l'occupation ( `__launch_bounds__( 128, BSM )` : ptxas rabote les registres
// pour loger `BSM` blocs par SM ) ; `BSM = 1` ne contraint rien.
// =====================================================================================

#include "gpu/FilNrm2D.cuh"
#include "gpu/Image2D.cuh"

namespace sf::gpu {

/// UNE COUPE, sur des sommets TRIES en `0 .. nb - 1`. Rend le nouveau nombre de sommets : `nb`
/// inchange si le plan ne coupe pas, `0` si la cellule devient vide, `-1` si elle deborde `R`.
/// Partagee telle quelle par `noyau2_filmsk` et par le noyau des phases.
template<int R, class TK>
__device__ __forceinline__ int coupe_msk( const Plan2<TK> &p, int nb, TK ( &x )[ R ], TK ( &y )[ R ], int ( &c )[ R ] ) {
    constexpr int SUR = 3;

    // ---- LA PREMIERE PASSE : `m`, le masque des sommets dehors
    unsigned m = 0;
#pragma unroll
    for ( int i = 0; i < R; ++i ) {
        if ( i >= SUR && i >= nb ) break;
        m |= unsigned( p.dx * x[ i ] + p.dy * y[ i ] - p.off > TK( 0 ) ) << i;
    }
    if ( PROBABLE( ! m ) )
        return nb;
    const unsigned valid = ( 1u << nb ) - 1;
    if ( IMPROBABLE( m == valid ) )
        return 0;

    // ---- LES QUATRE MASQUES DE ROLE, un seul bit chacun
    const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
    const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
    const unsigned r0 = ~m & next & valid;               // `v_j0` : dedans, le suivant dehors
    const unsigned r1 =  m & ~prev;                      // `v_i1` : dehors, le precedent dedans
    const unsigned r2 =  m & ~next;                      // `v_j2` : dehors, le suivant dedans
    const unsigned r3 = ~m & prev & valid;               // `v_j3` : dedans, le precedent dehors
    const int j0 = __ffs( int( r0 ) ) - 1, i1 = __ffs( int( r1 ) ) - 1;
    const int j2 = __ffs( int( r2 ) ) - 1, j3 = __ffs( int( r3 ) ) - 1;
    const int nb_out = __popc( m );
    const int nn = nb - nb_out + 2;
    if ( IMPROBABLE( nn > R ) )
        return -1;                                       // pour la seconde passe

    // ---- LES QUATRE SOMMETS, EN UNE SEULE BOUCLE BORNEE PAR `nb` : quatre compares par case,
    //      partages par neuf `select` ( `selR` en faisait neuf balayages complets des `R` cases )
    TK x0v = x[ 0 ], y0v = y[ 0 ], x1v = x[ 0 ], y1v = y[ 0 ];
    TK x2v = x[ 0 ], y2v = y[ 0 ], x3v = x[ 0 ], y3v = y[ 0 ];
    int bid = c[ 0 ];
#pragma unroll
    for ( int i = 1; i < R; ++i ) {
        if ( i >= SUR && i >= nb ) break;
        x0v = i == j0 ? x[ i ] : x0v; y0v = i == j0 ? y[ i ] : y0v;
        x1v = i == i1 ? x[ i ] : x1v; y1v = i == i1 ? y[ i ] : y1v;
        x2v = i == j2 ? x[ i ] : x2v; y2v = i == j2 ? y[ i ] : y2v; bid = i == j2 ? c[ i ] : bid;
        x3v = i == j3 ? x[ i ] : x3v; y3v = i == j3 ? y[ i ] : y3v;
    }

    // ---- LES DEUX POINTS CREES ; `s` recalcule pour les quatre ( un `fma` chacun )
    const TK s0 = p.dx * x0v + p.dy * y0v - p.off, s1 = p.dx * x1v + p.dy * y1v - p.off;
    const TK s2 = p.dx * x2v + p.dy * y2v - p.off, s3 = p.dx * x3v + p.dy * y3v - p.off;
    const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
    const TK pax = x0v + ( x1v - x0v ) * ta, pay = y0v + ( y1v - y0v ) * ta;
    const TK pbx = x3v + ( x2v - x3v ) * tb, pby = y3v + ( y2v - y3v ) * tb;

    // ---- LE REMONTAGE : `u[ k ] = old[ k - 1 ]` ( gratuit ), puis UN barillet de `e`
    const bool boucle = r1 > r2;                         // `i1 > j2` : la plage dehors boucle
    const int  a = boucle ? 0 : i1;
    const int  e = boucle ? j3 - 1 : nb_out - 1;         // `= d + 1 >= 0`
    TK  ux[ R + 1 ], uy[ R + 1 ];
    int uc[ R + 1 ];
    ux[ 0 ] = x[ 0 ]; uy[ 0 ] = y[ 0 ]; uc[ 0 ] = c[ 0 ];   // jamais lu : `o + e >= 1`
#pragma unroll
    for ( int o = 1; o < R + 1; ++o ) { ux[ o ] = x[ o - 1 ]; uy[ o ] = y[ o - 1 ]; uc[ o ] = c[ o - 1 ]; }
    // trois etages, la meme condition pour les trois tableaux ; les cases au-dela de `o + e = R`
    // ne servent jamais ( `o < nn` et `nn - 1 + e <= R` ), d'ou les bornes
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

/// `DENS` : le traitement de la densite image. `DENS_AUCUNE` rend l'aire ( le noyau d'origine,
/// octet pour octet ) ; `DENS_DIRECTE` integre l'image DANS ce noyau, a la suite du parcours ;
/// `DENS_DEPOT` se contente d'ECRIRE le polygone fini, pour un second noyau ( `Image2D.cuh` ).
/// Le depot travaille par LOTS : `k0` le premier rang du lot, `nk` sa taille, `cap` le pas du SoA.
enum { DENS_AUCUNE = 0, DENS_DIRECTE = 1, DENS_DEPOT = 2 };

template<bool POIDS, int BSM, bool CENTRE, int FIXE, int DENS = DENS_AUCUNE, class TK = float, bool RES64 = false>
__global__ void __launch_bounds__( 128, BSM ) noyau2_filmsk( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb,
                                                            int *fac_j = nullptr, TK *fac_l = nullptr, int NF = 0,
                                                            bool raff = false, Image2 im = Image2{}, TK *dep_x = nullptr, TK *dep_y = nullptr,
                                                            int *dep_nb = nullptr, int *dep_id = nullptr,
                                                            int cap = 0, int k0 = 0, int nk = 0 ) {
    constexpr int R = 8, SUR = 3;
    static_assert( ! RES64 || DENS == DENS_AUCUNE, "RES64 prend la mesure LUI-MEME : la densite image a son propre parcours d'arete" );
    const int k = ( DENS == DENS_DEPOT ? k0 : 0 ) + blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= ar.n ) return;
    if constexpr ( DENS == DENS_DEPOT ) if ( k >= k0 + nk ) return;
    const int u0[ 2 ] = { FIXE == 32 ? ar.u[ 0 ][ k ] : 0, FIXE == 32 ? ar.u[ 1 ][ k ] : 0 };
    const long long g0[ 2 ] = { FIXE == 64 ? ar.u64[ 0 ][ k ] : 0, FIXE == 64 ? ar.u64[ 1 ][ k ] : 0 };
    const TK p0[ 2 ] = { FIXE == 32 ? TK( u0[ 0 ] ) * TK( INV_FIXE ) : FIXE == 64 ? TK( double( g0[ 0 ] ) * INV_F64 ) : ar.c[ 0 ][ k ],
                         FIXE == 32 ? TK( u0[ 1 ] ) * TK( INV_FIXE ) : FIXE == 64 ? TK( double( g0[ 1 ] ) * INV_F64 ) : ar.c[ 1 ][ k ] };
    const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
    const double w0d = POIDS ? ar.w64[ k ] : 0.0;        // le plan porte LA DIFFERENCE des poids
    const int i0 = ar.ids[ k ];

    TK  x[ R ], y[ R ];
    int c[ R ];
#pragma unroll
    for ( int i = 0; i < R; ++i ) {
        // le carre unite, dans le repere du germe. En virgule fixe le sommet tombe pile sur
        // `2^30`, donc le cote de la cellule de depart est EXACT
        x[ i ] = FIXE == 32 ? TK( ( i == 1 || i == 2 ? ECH_FIXE : 0 ) - u0[ 0 ] ) * TK( INV_FIXE )
               : FIXE == 64 ? TK( ( i == 1 || i == 2 ? ECH_F64 : 0ll ) - g0[ 0 ] ) * TK( INV_F64 )
                            : TK( i == 1 || i == 2 ) - ( CENTRE ? p0[ 0 ] : TK( 0 ) );
        y[ i ] = FIXE == 32 ? TK( ( i == 2 || i == 3 ? ECH_FIXE : 0 ) - u0[ 1 ] ) * TK( INV_FIXE )
               : FIXE == 64 ? TK( ( i == 2 || i == 3 ? ECH_F64 : 0ll ) - g0[ 1 ] ) * TK( INV_F64 )
                            : TK( i == 2 || i == 3 ) - ( CENTRE ? p0[ 1 ] : TK( 0 ) );
        c[ i ] = i < 4 ? -1 - i : 0;
    }
    int nb = 4;

    int pile[ PILE ];
    int haut = 0;
    pile[ haut++ ] = 0;
    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,2> nd = ar.nodes[ h ];

        bool peut = false;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            if ( i >= SUR && i >= nb ) break;
            const TK v[ 2 ] = { x[ i ], y[ i ] };
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
            const Plan2<TK> p = FIXE == 32 ? bissect2f<POIDS>( ar, q, u0[ 0 ], u0[ 1 ], w0d )
                              : FIXE == 64 ? bissect2g<POIDS>( ar, q, g0[ 0 ], g0[ 1 ], w0d )
                              : CENTRE     ? bissect2c<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], w0d )
                                           : bissect2<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], w0 );
            nb = coupe_msk( p, nb, x, y, c );
            if ( IMPROBABLE( nb <= 0 ) ) goto fin;
        }
    }

fin:
    // ---- LES SOMMETS RESOLUS DEPUIS LEURS PLANS ( `Arbre.cuh` ). Le sommet `i` est porte par
    //      l'arete qui y ARRIVE ( `c[ i - 1 ]` ) et par celle qui en part ( `c[ i ]` ) ; on
    //      remonte donc les plans un par un, chacun servant deux fois.
    //
    // `RES64` : LA MESURE EST PRISE ICI, dans la boucle, sur le sommet resolu et AVANT qu'il ne
    // repasse par le flottant du noyau. C'est la derniere marche du modele `fp32`, et elle etait
    // restee en travers : la resolution rend le sommet a `2^-53` pres, puis `x[ i ] = TK( vx )` le
    // ramene aussitot a `eps h` -- donc l'aire a `eps` pres, soit le 1.4e-07 qui subsistait sur
    // l'uniforme. Le sommet resolu EXISTE deja en `double` dans un registre ; il suffit de s'en
    // servir avant de le jeter.
    //
    // CE QUE CA NE COUTE PAS : les `R` sommets en `double` ( 32 registres, l'occupation y
    // passerait ). Le lacet se ferme en STREAMING -- deux `double` pour le sommet precedent, deux
    // pour le premier -- donc cinq registres de plus en tout, et le tour du polygone est deja
    // fait par la boucle de resolution.
    //
    // LES FACETTES SUIVENT LE MEME CHEMIN. `| v_{i+1} - v_i |` prise sur des sommets `float` perd
    // sa precision RELATIVE sur les aretes courtes ( deux grands qui donnent un petit, encore ),
    // et c'est exactement la longueur presque nulle qui faisait diverger la hessienne. Prise en
    // `double`, seule la racine redescend au flottant du noyau.
    double a2 = 0;                                       // deux fois l'aire, par le lacet resolu
    bool   pris = false;                                 // la mesure a-t-elle ete prise ici
    if constexpr ( sizeof( TK ) < 8 && ( CENTRE || FIXE ) ) {
        if ( raff ) {
            if ( nb >= 3 ) {
                double ax, ay, ao;
                plan_relu<POIDS,FIXE>( ar, selR( c, nb - 1 ), u0, g0, p0, w0d, ax, ay, ao );
                double v0x = 0, v0y = 0, ppx = 0, ppy = 0;
#pragma unroll
                for ( int i = 0; i < R; ++i ) {
                    if ( i >= SUR && i >= nb ) break;
                    double bx, by, bo, vx, vy;
                    plan_relu<POIDS,FIXE>( ar, c[ i ], u0, g0, p0, w0d, bx, by, bo );
                    if ( ! croise2( ax, ay, ao, bx, by, bo, vx, vy ) ) { vx = double( x[ i ] ); vy = double( y[ i ] ); }
                    ax = bx; ay = by; ao = bo;
                    x[ i ] = TK( vx ); y[ i ] = TK( vy );   // `filmix`, le depot et le debordement les relisent
                    if constexpr ( RES64 ) {
                        if ( i == 0 ) { v0x = vx; v0y = vy; }
                        else {
                            a2 += ppx * vy - vx * ppy;
                            if ( fac_j ) {                  // la facette `i - 1`, du sommet `i - 1` au sommet `i`
                                const double ex = vx - ppx, ey = vy - ppy;
                                fac_j[ size_t( i - 1 ) * ar.n + i0 ] = id_de( ar, c[ i - 1 ] );
                                fac_l[ size_t( i - 1 ) * ar.n + i0 ] = TK( sqrt( ex * ex + ey * ey ) );
                            }
                        }
                        ppx = vx; ppy = vy;
                    }
                }
                if constexpr ( RES64 ) {
                    a2 += ppx * v0y - v0x * ppy;            // le lacet se ferme sur le sommet `0`
                    if ( fac_j ) {
                        const double ex = v0x - ppx, ey = v0y - ppy;
                        fac_j[ size_t( nb - 1 ) * ar.n + i0 ] = id_de( ar, selR( c, nb - 1 ) );
                        fac_l[ size_t( nb - 1 ) * ar.n + i0 ] = TK( sqrt( ex * ex + ey * ey ) );
                    }
                }
            }
            // `pris` est UNIFORME dans le warp ( `raff` l'est, `nb` ne l'est pas ) : les cellules
            // vides passent par la meme branche, avec zero partout.
            if constexpr ( RES64 ) {
                if ( fac_j )
#pragma unroll
                    for ( int e = 0; e < R; ++e )
                        if ( e >= nb ) { fac_j[ size_t( e ) * ar.n + i0 ] = -1000000; fac_l[ size_t( e ) * ar.n + i0 ] = 0; }
                pris = true;
            }
        }
    }

    // l'origine du repere des sommets : le germe si `CENTRE`, sinon rien. La densite, elle, vit
    // dans le carre unite -- c'est le seul endroit ou le repere centre doit etre defait.
    const double ox = CENTRE || FIXE ? double( p0[ 0 ] ) : 0.0;
    const double oy = CENTRE || FIXE ? double( p0[ 1 ] ) : 0.0;
    double mes = 0;

    if constexpr ( DENS == DENS_DIRECTE ) {
        // UNE SEULE MARCHE PAR ARETE, deux sorties : la masse et le coefficient de hessienne.
        // ( Avec une densite, `c_ij` n'est plus `| facette |` mais `integrale_facette rho ds`. )
        const double sref = nb > 0 ? im.ref( double( x[ 0 ] ) + ox, double( y[ 0 ] ) + oy ) : 0.0;
#pragma unroll
        for ( int e = 0; e < R; ++e ) {
            int j = -1000000;
            TK  l = 0;
            if ( e < nb ) {
                const int ee = e + 1 < nb ? e + 1 : 0;
                double mm, ll;
                arete_image( im, double( x[ e ] ) + ox, double( y[ e ] ) + oy,
                                 double( selR( x, ee ) ) + ox, double( selR( y, ee ) ) + oy, sref, mm, ll );
                mes += mm;
                j = id_de( ar, c[ e ] );
                l = TK( ll );
            }
            if ( fac_j ) { fac_j[ size_t( e ) * ar.n + i0 ] = j; fac_l[ size_t( e ) * ar.n + i0 ] = l; }
        }
        mes = fabs( mes );                               // `rho >= 0` : le signe est celui du lacet
    } else {
        // ---- LES FACETTES : une par arete dont le `cid` est un voisin. Au DEPOT, seul le `cid`
        //      est ecrit ici -- la longueur vient du second noyau, qui la veut ponderee par rho.
        if ( fac_j && ! pris ) {
#pragma unroll
            for ( int e = 0; e < R; ++e ) {
                int j = -1000000;                        // case vide ( au-dela de `nb` )
                TK  l = 0;
                if ( e < nb ) {
                    j = id_de( ar, c[ e ] );
                    if constexpr ( DENS != DENS_DEPOT ) {
                        const int ee = e + 1 < nb ? e + 1 : 0;
                        const TK dx = selR( x, ee ) - x[ e ], dy = selR( y, ee ) - y[ e ];
                        l = sqrt( dx * dx + dy * dy );
                    }
                }
                fac_j[ size_t( e ) * ar.n + i0 ] = j;
                fac_l[ size_t( e ) * ar.n + i0 ] = l;
            }
        }
        if constexpr ( DENS != DENS_DEPOT ) mes = pris ? 0.5 * fabs( a2 ) : ( nb > 0 ? aire_triee( x, y, nb ) : 0.0 );
    }
    if ( fac_j )
        for ( int e = R; e < NF; ++e ) {                 // `nb <= R` ici : le reste est vide
            fac_j[ size_t( e ) * ar.n + i0 ] = -1000000;
            fac_l[ size_t( e ) * ar.n + i0 ] = 0;
        }

    if constexpr ( DENS == DENS_DEPOT ) {
        // LE DEPOT : le polygone fini, en coordonnees ABSOLUES et en SoA -- les voies voisines
        // ecrivent des adresses voisines. `nb < 0` ( debordement ) descend tel quel : le second
        // noyau le saute, et `filmix` refera la cellule.
        const int s = k - k0;
#pragma unroll
        for ( int e = 0; e < R; ++e ) { dep_x[ size_t( e ) * cap + s ] = TK( double( x[ e ] ) + ox ); dep_y[ size_t( e ) * cap + s ] = TK( double( y[ e ] ) + oy ); }
        dep_nb[ s ] = nb;
        dep_id[ s ] = i0;
    }

    if ( nb < 0 ) liste_deb[ atomicAdd( deborde, 1 ) ] = k;
    res[ i0 ] = mes;
}

} // namespace sf::gpu
