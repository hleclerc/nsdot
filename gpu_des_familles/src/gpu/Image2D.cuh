#pragma once

// =====================================================================================
// LA DENSITE IMAGE : integrer rho SUR LE BORD, jamais sur l'interieur.
//
// Une image est une densite CONSTANTE PAR PIXEL sur une grille reguliere de `W x H` cases
// couvrant `[ 0, 1 ]^2`. La facon naive de mesurer une cellule sous une telle densite est de
// DECOUPER la cellule par les bords de pixels et de sommer les aires -- c'est ce que fait le
// moteur CPU ( `sdot/include/sdot/Image.cxx` ), et ca coute `2 d` coupes PAR PIXEL de la boite
// englobante, plus une copie de la cellule par pixel. Mesure faite ailleurs : dans le regime qui
// nous interesse ( cellules ~ pixels ) ca multiplie le temps par 1.5 en 2D et par 2.4 en 3D.
//
// Ici on ne coupe rien. Green :
//
//      masse( P ) = integrale_P rho dA = integrale_(bord P) G( x, y ) dy,   G( x, y ) = int_0^x rho
//
// et pour une image, G est CONNUE EN FERME : sur la ligne `j`, rho ne depend plus de `y`, donc
//
//      G_j( x ) = S[ j ][ i ] + rho[ j ][ i ] ( x - i hx ),    S[ j ][ i ] = hx somme_(k<i) rho[j][k]
//
// -- la somme prefixe de la ligne, calculee UNE FOIS a la montee de l'image. Sur un morceau
// d'arete qui reste dans le pixel `( i, j )`, de `( xa, ya )` a `( xb, yb )`, G est affine en `x`
// donc son integrale en `y` est exacte au point milieu :
//
//      ( yb - ya ) ( S[ j ][ i ] + rho[ j ][ i ] ( ( xa + xb ) / 2 - i hx ) )
//
// Le cout n'est donc plus `O( pixels DANS la cellule )` mais `O( pixels SOUS LE BORD )`, sans une
// seule coupe ni une seule copie -- et le parcours est un Amanatides-Woo de quatre lignes.
//
// DEUX SORTIES POUR UNE SEULE MARCHE. Avec une densite, le coefficient de la hessienne n'est plus
// la longueur de la facette mais `integrale_facette rho ds / ( 2 | p_i - p_j | )`. C'est la MEME
// marche, le meme pixel, la meme sous-arete : `arete_image` rend les deux. C'est la raison
// profonde pour laquelle tout ce fichier travaille ARETE PAR ARETE.
//
// LA REFERENCE `sref`. La somme des `dy` sur un polygone ferme vaut zero, donc RETRANCHER UNE
// CONSTANTE A G NE CHANGE RIEN. Sans elle, les termes sont d'ordre `taille de cellule x S ~ 1e-3`
// pour une masse qui vaut `1e-6` : trois chiffres perdus par annulation. On retranche `S` au
// premier sommet : les termes tombent a l'ordre de la masse, et l'erreur avec eux.
// =====================================================================================

namespace sf::gpu {

/// LA GRILLE, telle que le noyau la lit. `p[ j * W + i ].x` est la somme prefixe de la ligne `j`
/// jusqu'a `i` ( `hx` compris ), `.y` la valeur du pixel. Un seul chargement de 16 octets par
/// pixel visite -- les acces sont disperses, ce qui compte est le nombre de transactions.
struct Image2 {
    const double2 *p = nullptr;
    int    W = 0, H = 0;
    double hx = 0, hy = 0, ihx = 0, ihy = 0;

    __host__ __device__ bool active() const { return p != nullptr; }

    __device__ __forceinline__ int col( double x ) const { const int i = int( x * ihx ); return i < 0 ? 0 : ( i >= W ? W - 1 : i ); }
    __device__ __forceinline__ int lig( double y ) const { const int j = int( y * ihy ); return j < 0 ? 0 : ( j >= H ? H - 1 : j ); }

    /// la valeur de la primitive a retrancher ( n'importe laquelle convient : voir l'en-tete )
    __device__ __forceinline__ double ref( double x, double y ) const { return p[ size_t( lig( y ) ) * W + col( x ) ].x; }
};

/// LE PARCOURS D'UNE ARETE dans la grille ( Amanatides-Woo ), et ses DEUX sorties :
///   `mas` la contribution de l'arete a `integrale_(bord) G dy`, c'est-a-dire A LA MASSE ;
///   `lon` `integrale_arete rho ds`, c'est-a-dire LE COEFFICIENT DE LA HESSIENNE ( `rho = 1` rend
///         bien la longueur ).
/// Tout est en `double` : la geometrie peut etre en `float`, la mesure ne l'est jamais.
__device__ __forceinline__ void arete_image( const Image2 &im, double x0, double y0, double x1, double y1,
                                             double sref, double &mas, double &lon ) {
    mas = 0; lon = 0;
    const double dx = x1 - x0, dy = y1 - y0;
    const double lg = sqrt( dx * dx + dy * dy );
    if ( lg == 0 ) return;

    constexpr double INF = 1e300;
    int i = im.col( x0 ), j = im.lig( y0 );
    const int si = dx > 0 ? 1 : -1, sj = dy > 0 ? 1 : -1;
    // `t` de la premiere frontiere verticale / horizontale, puis le pas entre deux frontieres
    double tx = dx == 0 ? INF : ( ( dx > 0 ? ( i + 1 ) * im.hx : i * im.hx ) - x0 ) / dx;
    double ty = dy == 0 ? INF : ( ( dy > 0 ? ( j + 1 ) * im.hy : j * im.hy ) - y0 ) / dy;
    const double ax = dx == 0 ? INF : fabs( im.hx / dx );
    const double ay = dy == 0 ? INF : fabs( im.hy / dy );
    tx = tx < 0 ? 0 : tx; ty = ty < 0 ? 0 : ty;          // depart pile sur une frontiere

    double tp = 0, xp = x0;
    const int garde = im.W + im.H + 4;                   // un segment ne traverse pas plus
    for ( int g = 0; g < garde; ++g ) {
        const bool par_x = tx < ty;
        double tn = par_x ? tx : ty;
        const bool der = ! ( tn < 1.0 );
        if ( der ) tn = 1.0;
        const double xc = x0 + dx * tn;
        const double2 q = im.p[ size_t( j ) * im.W + i ];
        const double dt = tn - tp;
        mas += dt * dy * ( q.x - sref + q.y * ( 0.5 * ( xp + xc ) - i * im.hx ) );
        lon += dt * lg * q.y;
        if ( der ) break;
        if ( par_x ) { i += si; tx += ax; } else { j += sj; ty += ay; }
        i = i < 0 ? 0 : ( i >= im.W ? im.W - 1 : i );    // le polygone est dans le carre : ne
        j = j < 0 ? 0 : ( j >= im.H ? im.H - 1 : j );    // borne que les arrondis de bord
        tp = tn; xp = xc;
    }
}

// =====================================================================================
// LE TRAITEMENT PAR LOT. `DENS_DEPOT` fait ECRIRE la cellule finie par le noyau des cellules, et
// c'est un second noyau qui l'integre. Deux raisons :
//
//   * LA DIVERGENCE. Le parcours d'une arete dure `| di | + | dj | + 1` tours, et le cout d'un
//     warp est le MAX sur ses voies. Mise dans le noyau des cellules, cette attente s'ajoute a
//     celle du parcours de l'arbre, qui diverge deja ; mise a part, elle se paie dans un noyau
//     minuscule ou l'occupation est libre.
//   * LES REGISTRES. `noyau2_filmsk` est regle a 96 registres ; lui ajouter la marche, c'est
//     risquer une marche d'occupation pour du travail qui n'a rien a voir.
//
// `DENS_DEPOT_ARETE` va plus loin : UNE VOIE PAR ARETE, huit voies par cellule, quatre cellules
// par warp. Le cout du warp devient le MAX sur les aretes au lieu de la SOMME sur les aretes de
// la cellule la plus lente -- c'est la seule forme qui attaque vraiment la divergence.
//
// Le depot se fait PAR LOTS de `cap` cellules : a 10^9 germes on n'a pas la place de garder tous
// les polygones ( 8 sommets x 2 coordonnees x 8 octets = 128 octets par cellule ).
// =====================================================================================

/// UNE VOIE PAR CELLULE. `dx` / `dy` sont en SoA `[ sommet * cap + place ]`, en coordonnees
/// ABSOLUES ( le noyau des cellules a deja rajoute le germe s'il travaillait dans son repere ).
/// `fac_l`, s'il est donne, recoit `integrale_arete rho ds` a la place de la longueur.
template<class TK>
__global__ void __launch_bounds__( 128 ) k_dens_cel( Image2 im, const TK *dx, const TK *dy, const int *dnb, const int *did,
                                                     int cap, int nk, int n, double *res, TK *fac_l ) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if ( s >= nk ) return;
    const int nb = dnb[ s ], id = did[ s ];
    if ( nb <= 0 ) { if ( nb == 0 ) res[ id ] = 0; return; }
    const double sref = im.ref( double( dx[ s ] ), double( dy[ s ] ) );
    double mas = 0;
    double xp = double( dx[ s ] ), yp = double( dy[ s ] );
    for ( int e = 0; e < nb; ++e ) {
        const int ee = e + 1 < nb ? e + 1 : 0;
        const double xc = double( dx[ size_t( ee ) * cap + s ] ), yc = double( dy[ size_t( ee ) * cap + s ] );
        double mm, ll;
        arete_image( im, xp, yp, xc, yc, sref, mm, ll );
        mas += mm;
        if ( fac_l ) fac_l[ size_t( e ) * n + id ] = TK( ll );
        xp = xc; yp = yc;
    }
    res[ id ] = fabs( mas );
}

/// UNE VOIE PAR ARETE : `NP` voies par cellule, la somme par `__shfl_down_sync` sur `NP` voies.
/// Aucune sortie anticipee -- les voies inactives restent dans le warp pour les echanges.
template<class TK, int NP>
__global__ void __launch_bounds__( 128 ) k_dens_arete( Image2 im, const TK *dx, const TK *dy, const int *dnb, const int *did,
                                                       int cap, int nk, int n, double *res, TK *fac_l ) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = t / NP, e = t % NP;
    const bool actif = s < nk;
    const int nb = actif ? dnb[ s ] : 0, id = actif ? did[ s ] : 0;

    // la reference, lue par la voie zero du groupe et diffusee
    double sref = 0;
    if ( e == 0 && nb > 0 ) sref = im.ref( double( dx[ s ] ), double( dy[ s ] ) );
    sref = __shfl_sync( 0xffffffff, sref, ( threadIdx.x & 31 ) & ~( NP - 1 ), 32 );

    double mas = 0, lon = 0;
    if ( e < nb ) {
        const int ee = e + 1 < nb ? e + 1 : 0;
        arete_image( im, double( dx[ size_t( e ) * cap + s ] ), double( dy[ size_t( e ) * cap + s ] ),
                         double( dx[ size_t( ee ) * cap + s ] ), double( dy[ size_t( ee ) * cap + s ] ), sref, mas, lon );
    }
    if ( actif && fac_l && e < nb ) fac_l[ size_t( e ) * n + id ] = TK( lon );
#pragma unroll
    for ( int o = NP / 2; o; o >>= 1 ) mas += __shfl_down_sync( 0xffffffff, mas, o, NP );
    if ( actif && e == 0 && nb >= 0 ) res[ id ] = fabs( mas );
}

} // namespace sf::gpu
