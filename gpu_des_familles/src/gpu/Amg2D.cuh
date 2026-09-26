#pragma once

// =====================================================================================
// UN MULTIGRILLE ALGEBRIQUE MAISON, sur la carte. Le CG a preconditionneur diagonal converge en
// `sqrt( n )` iterations ( 8167 a n = 1e6 ) et pese 99 % de l'iteration de Newton ; il lui faut
// une hierarchie.
//
// L'AGREGATION EST GRATUITE, et c'est le point. Les germes sont ranges DANS L'ORDRE DE L'ARBRE,
// qui est une courbe remplissante : des RANGS CONSECUTIFS sont voisins dans le plan. Agreger,
// c'est donc `rang >> sh` -- sans un noyau d'appariement, sans matching, sans compaction. Et
// comme les indices d'agregat restent ordonnes par rang, le niveau suivant s'agrege pareil :
// `a >> sh`. La hierarchie entiere tient dans un decalage.
// ( Le niveau fin est indexe par IDENTIFIANT ; on passe par `rang_de` une seule fois. )
//
// LA TAILLE DU PAQUET EST UN REGLAGE, ET ELLE N'AVAIT JAMAIS ETE BALAYEE. Quatre etait le choix
// naturel d'une courbe de Morton en 2D. Mais `AaBsp` ne fait pas du Morton : il coupe A LA
// MEDIANE SUR L'AXE LE PLUS LONG, donc une fenetre alignee de `2^k` rangs est exactement un
// sous-arbre -- localite parfaite -- et `k` impair ne donne qu'une boite 2:1, ce qui pour de
// l'agregation va tres bien. TOUTES LES PUISSANCES DE DEUX sont donc disponibles.
//
// Sur CPU ( `solvers_des_familles/README.md` § 17.11 et § 17.15 ) l'optimum est HUIT dans les
// deux dimensions -- et l'attraper demande de regler le nombre de lissages EN MEME TEMPS : a
// `nu = 3` l'optimum 3D semble etre seize, parce qu'un cycle trop lisse force a grossir les
// paquets pour rester payable. Les balayer separement donne le mauvais point. `AMG_AGREG`.
//
// LE GROSSIER est le produit de Galerkin `A_c = P^T A P` avec `P` constant par morceaux. Sur un
// laplacien de graphe c'est encore un laplacien : il suffit de sommer les poids d'aretes entre
// paquets, `c_ab = somme des c_ij pour i dans a, j dans b`, et la diagonale est la somme de la
// ligne. On emet donc un triplet par arete, on TRIE ( CUB ), on REDUIT PAR CLEF, et le CSR sort
// de la.
//
// LE CYCLE EN V : un lissage avant, un apres ( le meme des deux cotes, donc l'operateur est
// SYMETRIQUE et le CG l'accepte comme preconditionneur ), restriction par somme sur le paquet,
// prolongation par diffusion. Le niveau le plus grossier -- moins de mille inconnues -- est
// lisse cent fois.
//
// DEUX LISSEURS ( `AMG_LISSEUR` ). Jacobi amorti a `omega = 0.7` ( 0, le defaut historique ), ou
// CHEBYSHEV ( 2 ) : un polynome de degre `nu` en `M^-1 A` choisi pour minimiser le maximum sur
// `[ lmax / r, lmax ]`, c'est-a-dire la partie du spectre que le grossier NE corrige pas. Il ne
// demande que des produits matrice-vecteur -- exactement ce qu'une carte veut -- et son
// amortissement ne se devine pas : il sort de `lmax`, qu'on BORNE par Gershgorin sur `M^-1 A`,
// `lmax <= max_i m_i ( dia_i + somme_e |val_e| )`, exact et gratuit.
//
// L'interieur de Chebyshev est `spai0`, la meilleure approximation DIAGONALE de `A^-1` au sens de
// Frobenius : `m_i = A_ii / somme_j A_ij^2`. Sur un laplacien a six voisins egaux elle vaut
// `1 / ( 7 c )` la ou Jacobi non amorti vaut `1 / ( 6 c )` -- un amortissement CALCULE, pas pose.
//
// ET UNE PASSE DE TROP ANNULE LE GAIN. Sur CPU, en ecrivant `A y` dans un tampon puis en refaisant
// une passe pour mettre a jour `r` et la direction, Chebyshev rendait en cout par iteration
// ( +12 % ) ce qu'il gagnait en nombre ( -8 % ). Le double tampon RESTE necessaire -- le produit
// de la ligne `i` lit `y` chez les voisins -- mais la seconde passe non : un seul noyau lit `y`,
// met `r` a jour sur place, et ecrit la direction suivante dans `z`, qu'on echange. Sur carte,
// limitee par la bande passante, ce piege est pire, pas meilleur.
//
// LA JAUGE passe de `x[ 0 ] = 0` a MOYENNE NULLE, qui est la bonne pour un multigrille : le
// laplacien a les constantes pour noyau, `b = mesures - cible` est deja de somme nulle, et
// projeter est symetrique la ou rayer une ligne ne l'est pas. Les deux jauges donnent la meme
// direction de Newton a une constante pres.
// =====================================================================================

#include "gpu/Lisse2D.cuh"
#include <cub/cub.cuh>

namespace sf::gpu {
namespace {

/// un niveau de la hierarchie ( laplacien : hors-diagonaux positifs, diagonale = somme de ligne )
struct Niveau {
    int    *row = nullptr, *col = nullptr;
    double *val = nullptr, *dia = nullptr;
    double *x = nullptr, *b = nullptr, *r = nullptr;     ///< les vecteurs de travail du cycle
    double *v1 = nullptr, *v2 = nullptr, *t = nullptr, *rc = nullptr;   ///< ceux du K-cycle
    double *sc = nullptr;                                ///< huit scalaires, sur la carte
    double *rl = nullptr;                                ///< le coefficient de relaxation, un par ligne
    double *sy = nullptr, *sz = nullptr;                 ///< les deux tampons de direction de Chebyshev
    double  lmax = 2;                                    ///< la borne de Gershgorin sur `M^-1 A`
    Csr     P, R;                                        ///< prolongation lissee et sa transposee
    int     n = 0, nnz = 0;
};

/// `m[ i ]` : le paquet du niveau suivant. Au niveau fin on passe par le rang, ensuite non.
__global__ void k_amg_map_fin( const int *rang_de, int *m, int n, int sh ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) m[ i ] = rang_de[ i ] >> sh;
}
__global__ void k_amg_map( int *m, int n, int sh ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) m[ i ] = i >> sh;
}

/// un triplet par arete inter-paquets : la clef porte `( a, b )`, la valeur le poids
__global__ void k_amg_triples( const int *row, const int *col, const double *val, const int *m,
                               int n, int nc, unsigned long long *clef, double *poids, int *cpt ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    const int a = m[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) {
        const int b = m[ col[ p ] ];
        if ( a == b ) continue;                          // l'interieur du paquet disparait
        const int q = atomicAdd( cpt, 1 );
        clef[ q ] = ( unsigned long long ) a * nc + b;
        poids[ q ] = val[ p ];
    }
}

/// du tableau de clefs uniques ( trie ) au CSR : compter, puis placer
__global__ void k_amg_compte( const unsigned long long *clef, int nu, int nc, int *cnt ) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if ( q < nu ) atomicAdd( &cnt[ int( clef[ q ] / nc ) ], 1 );
}
__global__ void k_amg_place( const unsigned long long *clef, const double *poids, int nu, int nc,
                             const int *row, int *at, int *col, double *val ) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if ( q >= nu ) return;
    const int a = int( clef[ q ] / nc );
    const int p = row[ a ] + atomicAdd( &at[ a ], 1 );
    col[ p ] = int( clef[ q ] % nc );
    val[ p ] = poids[ q ];
}
__global__ void k_amg_dia( const int *row, const double *val, double *dia, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double s = 0;
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) s += val[ p ];
    dia[ i ] = s > 0 ? s : 1.0;
}

/// `r = b - A x` ( laplacien : diagonale moins les hors-diagonaux )
__global__ void k_amg_residu( const int *row, const int *col, const double *val, const double *dia,
                              const double *x, const double *b, double *r, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double s = dia[ i ] * x[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) s -= val[ p ] * x[ col[ p ] ];
    r[ i ] = b[ i ] - s;
}
/// Jacobi amorti : `x += omega D^-1 ( b - A x )`
__global__ void k_amg_jacobi( const int *row, const int *col, const double *val, const double *dia,
                              double *x, const double *b, double omega, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double s = dia[ i ] * x[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) s -= val[ p ] * x[ col[ p ] ];
    x[ i ] += omega * ( b[ i ] - s ) / dia[ i ];
}

/// LE COEFFICIENT DE RELAXATION, UN PAR LIGNE : `spai0`, l'interieur de Chebyshev.
__global__ void k_amg_relax( const int *row, const int *col, const double *val, const double *dia,
                             double *rl, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    ( void ) col;
    const double d = dia[ i ];
    double q = d * d;
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) q += val[ p ] * val[ p ];
    rl[ i ] = q > 0 ? d / q : 0.0;
}

/// LA BORNE DE GERSHGORIN SUR `M^-1 A`, exacte et en une passe. `atomicMax` n'existe pas pour les
/// `double`, mais le motif binaire d'un POSITIF est monotone : on passe par `unsigned long long`.
__global__ void k_amg_gersh( const int *row, const int *col, const double *val, const double *dia,
                             const double *rl, unsigned long long *acc, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    ( void ) col;
    double sm = dia[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) sm += fabs( val[ p ] );
    const double v = rl[ i ] * sm;
    if ( v > 0 ) atomicMax( acc, ( unsigned long long ) __double_as_longlong( v ) );
}

/// CHEBYSHEV, le premier pas : `r = b - A x` ( ou `b` si on part de zero ), puis `y = r / theta`.
__global__ void k_cheb_init( const int *row, const int *col, const double *val, const double *dia,
                             const double *x, const double *b, double *r, double *y,
                             const double *rl, int net, double invth, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double ri = b[ i ];
    if ( ! net ) {
        double sm = dia[ i ] * x[ i ];
        for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) sm -= val[ p ] * x[ col[ p ] ];
        ri = b[ i ] - sm;
    }
    r[ i ] = ri;
    y[ i ] = rl[ i ] * ri * invth;
}

/// CHEBYSHEV, un pas : `x += y`, `r -= A y`, `z = c1 y + c2 M^-1 r`. UN SEUL NOYAU -- c'est tout
/// l'objet : on lit `y`, on met `r` a jour sur place ( chacun son indice ), et la direction
/// suivante part dans `z`, qu'on echange. Deux passes rendraient le gain.
__global__ void k_cheb_pas( const int *row, const int *col, const double *val, const double *dia,
                            double *x, double *r, const double *y, double *z, const double *rl,
                            double c1, double c2, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    const double yi = y[ i ];
    x[ i ] += yi;
    double sm = dia[ i ] * yi;
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) sm -= val[ p ] * y[ col[ p ] ];
    const double ri = r[ i ] - sm;
    r[ i ] = ri;
    z[ i ] = c1 * yi + c2 * rl[ i ] * ri;
}

/// ... et le dernier pas, qui n'a plus de direction a preparer
__global__ void k_cheb_fin( double *x, const double *y, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) x[ i ] += y[ i ];
}

/// LE NIVEAU LE PLUS GROSSIER EN UN SEUL NOYAU. Il fait moins de mille inconnues, donc il tient
/// dans UN bloc et dans la memoire partagee : les soixante lissages deviennent une boucle avec un
/// `__syncthreads` entre deux, au lieu de soixante lancements. Et comme le K-cycle le visite
/// quatre fois par application, c'etaient 244 lancements par preconditionnement -- le quart du
/// temps de resolution, en frais de lancement purs.
/// Jacobi a besoin de l'ancien `x` pour toutes les cases : deux tampons partages, alternes.
__global__ void k_amg_gros( const int *row, const int *col, const double *val, const double *dia,
                            double *x, const double *b, double omega, int n, int nsweep ) {
    extern __shared__ double sh[];
    double *u = sh, *v = sh + n;
    for ( int i = threadIdx.x; i < n; i += blockDim.x ) u[ i ] = 0;
    __syncthreads();
    for ( int k = 0; k < nsweep; ++k ) {
        for ( int i = threadIdx.x; i < n; i += blockDim.x ) {
            double s = dia[ i ] * u[ i ];
            for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) s -= val[ p ] * u[ col[ p ] ];
            v[ i ] = u[ i ] + omega * ( b[ i ] - s ) / dia[ i ];
        }
        __syncthreads();
        double *t = u; u = v; v = t;
    }
    for ( int i = threadIdx.x; i < n; i += blockDim.x ) x[ i ] = u[ i ];
}

__global__ void k_amg_restreint( const double *r, const int *m, double *bc, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) atomicAdd( &bc[ m[ i ] ], r[ i ] );
}
__global__ void k_amg_prolonge( double *x, const double *xc, const int *m, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) x[ i ] += xc[ m[ i ] ];
}

/// la somme d'un vecteur ( warp, bloc, un atomique ) -- pour la projection
__global__ void k_amg_somme( const double *u, double *acc, int n ) {
    __shared__ double part[ 32 ];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double s = i < n ? u[ i ] : 0.0;
#pragma unroll
    for ( int d = 16; d; d >>= 1 ) s += __shfl_down_sync( 0xffffffffu, s, d );
    const int voie = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if ( voie == 0 ) part[ warp ] = s;
    __syncthreads();
    if ( warp == 0 ) {
        s = voie < ( blockDim.x >> 5 ) ? part[ voie ] : 0.0;
#pragma unroll
        for ( int d = 16; d; d >>= 1 ) s += __shfl_down_sync( 0xffffffffu, s, d );
        if ( voie == 0 ) atomicAdd( acc, s );
    }
}

/// `rang_de` : identifiant -> rang dans l'arbre
__global__ void k_amg_rang( const int *ids, int *rang_de, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k < n ) rang_de[ ids[ k ] ] = k;
}

// ---------------------------------------------------------------- LE K-CYCLE
//
// Le defaut mesure de l'agregation non lissee est que LA CORRECTION GROSSIERE EST TROP FAIBLE et
// que l'erreur s'accumule d'un niveau a l'autre ( descendre a 16 inconnues au lieu de 1000 faisait
// passer de 168 a 411 iterations ). Le K-cycle y repond non pas en changeant `P` -- ce que fait la
// prolongation lissee, au prix d'un vrai produit triple creux -- mais en ACCELERANT CHAQUE NIVEAU
// PAR KRYLOV : au lieu d'un appel recursif, DEUX pas d'un gradient conjugue sur le systeme
// grossier, dont le preconditionneur est le niveau d'en dessous.
//
// Les coefficients restent sur la carte ( un thread les calcule ) : sans ca, chaque niveau de
// chaque cycle couterait une synchronisation.

__global__ void k_kc_c1( const double *rho1, const double *a1, double *c1 ) {
    *c1 = *rho1 != 0 ? *a1 / *rho1 : 0.0;
}
/// `rc = b - c1 t`, le residu apres le premier pas
__global__ void k_kc_rc( double *rc, const double *b, const double *t, const double *c1, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) rc[ i ] = b[ i ] - *c1 * t[ i ];
}
/// les coefficients du second pas : `rho2 = b2 - g2^2 / rho1`, et la correction de `c1`
__global__ void k_kc_c2( const double *rho1, const double *a1, const double *g2, const double *b2,
                         const double *a2, double *c1, double *c2 ) {
    const double r1 = *rho1, r2 = *b2 - ( *g2 ) * ( *g2 ) / ( r1 != 0 ? r1 : 1.0 );
    *c2 = r2 != 0 ? *a2 / r2 : 0.0;
    *c1 = ( r1 != 0 ? *a1 / r1 : 0.0 ) - ( r1 != 0 && r2 != 0 ? ( *g2 ) * ( *a2 ) / ( r1 * r2 ) : 0.0 );
}
__global__ void k_kc_comb( double *x, const double *v1, const double *v2,
                           const double *c1, const double *c2, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) x[ i ] = *c1 * v1[ i ] + *c2 * v2[ i ];
}
__global__ void k_amg_matvec( const int *row, const int *col, const double *val, const double *dia,
                              const double *x, double *y, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double s = dia[ i ] * x[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) s -= val[ p ] * x[ col[ p ] ];
    y[ i ] = s;
}
__global__ void k_amg_copie( double *dst, const double *src, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) dst[ i ] = src[ i ];
}

/// `y += a x`, `a` sur l'HOTE -- le recyclage n'en fait que `2 k` par resolution, et les
/// coefficients sortent d'un systeme dense `k x k` resolu sur l'hote de toute facon.
__global__ void k_rec_axpy( double *y, const double *x, double a, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) y[ i ] += a * x[ i ];
}

/// la projection sur la moyenne nulle : le noyau du laplacien
__global__ void k_amg_centre( double *x, const double *somme, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) x[ i ] -= *somme / n;
}

} // namespace
} // namespace sf::gpu
