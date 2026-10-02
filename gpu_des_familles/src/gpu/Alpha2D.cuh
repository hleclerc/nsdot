#pragma once

// =====================================================================================
// `alpha*` : LE PAS EXACT OU LA PREMIERE CELLULE TOUCHE LE PLANCHER, par le polynome de l'aire.
//
// = POURQUOI L'AIRE EST UN POLYNOME, ET DE DEGRE DEUX
//
// Le long du rayon `w - t d`, le plan qui separe `i` de `j` porte `( w_i - w_j ) / 2`, donc son
// decalage est AFFINE en `t` -- et sa NORMALE, qui ne depend que des positions, ne bouge pas du
// tout. Un sommet est l'intersection de deux plans : meme matrice `2 x 2`, second membre affine,
// donc LE SOMMET EST AFFINE EN `t`. L'aire par le lacet est une forme quadratique des sommets :
//
//     A( t ) = 1/2 somme_k ( v_k + t v'_k ) x ( v_{k+1} + t v'_{k+1} )  =  a0 + a1 t + a2 t^2
//
// exactement, en 2D ( un cubique en 3D ). Il n'y a rien a ajuster et rien a echantillonner :
// `alpha*_i` est la plus petite racine positive de `A_i( t ) = seuil`, formule fermee, et
// `alpha* = min_i alpha*_i` est une reduction.
//
// = POURQUOI CETTE PASSE NE COUTE PRESQUE RIEN
//
// La direction `d` n'existe qu'APRES le gradient conjugue, donc apres le diagramme qui a fourni
// la hessienne : une seconde passe sur les cellules est structurelle, on ne peut pas la fondre
// dans la premiere. Mais elle n'a pas besoin de REFAIRE le diagramme -- LA CONNECTIVITE EST DEJA
// LA, dans `fac_j`, une arete par case et dans l'ordre du polygone. Or l'intersection de
// demi-plans ne depend pas de l'ordre des coupes, et surtout : l'arete `e` va du sommet `e` au
// sommet `e + 1`, donc LE SOMMET `i` EST L'INTERSECTION DES PLANS `i - 1` ET `i`. Il n'y a donc
// ni parcours d'arbre, ni elagage, ni recherche de plan, ni meme decoupe -- huit lectures de
// plan et huit systemes `2 x 2`. Une passe d'essai coutait un diagramme entier ( ~38 ns/germe ) ;
// celle-ci en coute la fraction qui reste une fois l'arbre enleve.
//
// = CE QUE LE POLYNOME NE SAIT PAS
//
// Il est exact TANT QUE LA COMBINATOIRE NE CHANGE PAS. Un voisin qui entre rend l'aire reelle
// PLUS PETITE que le polynome, donc `alpha*` optimiste -- le mauvais sens. On ne traque pas ces
// evenements ( les trouver demanderait l'arbre, c'est-a-dire le diagramme qu'on voulait eviter ) :
// on prend `facteur . alpha*` avec `facteur < 1` et ON VALIDE avec le diagramme du nouvel itere,
// qu'on calcule de toute facon. C'est le dessin du banc CPU.
//
// = LE CONTROLE QUI NE COUTE RIEN
//
// `dm_i / dw_j = L_ij`, donc le long de `w - t d` on a `A'_i( 0 ) = -( L d )_i` -- et `L d` est le
// produit matrice-vecteur que le gradient conjugue vient de faire. Le coefficient `a1` est donc
// VERIFIABLE exactement, et c'est tout le chemin ( plan, derivee du decalage, vitesse du sommet,
// lacet ) qu'il controle d'un coup. Seul `a2` est vraiment neuf.
// =====================================================================================

#include "gpu/FilMsk2D.cuh"

namespace sf::gpu {
namespace {

/// LA DIRECTION, PERMUTEE DANS L'ORDRE DE L'ARBRE ( comme les poids ) : `d64[ rang ] = d[ id ]`.
__global__ void k_dir_rang( const double *d_id, const int *ids, double *d64, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k < n ) d64[ k ] = d_id[ ids[ k ] ];
}

/// LA PLUS PETITE RACINE POSITIVE de `a2 t^2 + a1 t + c`, avec `c >= 0` ( l'etat courant respecte
/// le plancher ). Rend `INF` s'il n'y en a pas -- cette cellule ne contraint pas le pas.
/// La forme stable ( `q = -( a1 + signe( a1 ) sqrt( D ) ) / 2` ) evite la soustraction de deux
/// grands quand `4 a2 c << a1^2`, ce qui est le cas courant : le pas est petit devant la courbure.
__device__ __forceinline__ double racine_pos( double a2, double a1, double c ) {
    constexpr double INF = 1e300;
    if ( c < 0 ) return 0.0;                             // deja sous le seuil : aucun pas admissible
    if ( a2 == 0.0 ) return a1 < 0 ? -c / a1 : INF;      // le cas degenere : l'aire est affine
    const double disc = a1 * a1 - 4.0 * a2 * c;
    if ( disc < 0 ) return INF;                          // l'aire ne touche jamais le seuil
    const double r = sqrt( disc );
    const double q = -0.5 * ( a1 + ( a1 >= 0 ? r : -r ) );
    double t1 = q / a2, t2 = q != 0.0 ? c / q : INF;
    if ( t1 <= 0 ) t1 = INF;
    if ( t2 <= 0 ) t2 = INF;
    return fmin( t1, t2 );
}

/// LE NOYAU. Un thread par cellule, comme `filmsk`, mais SANS l'arbre : la connectivite vient de
/// `fac_j` ( identifiants de l'appelant, `-1 - d` pour un cote de boite, `-1000000` pour une case
/// vide ). `pol` recoit `( a0, a1, a2 )` en SoA, indexes par identifiant ; `amin` recoit le
/// minimum de `alpha*_i` par `atomicMin` sur le motif binaire ( pour un `double` POSITIF, l'ordre
/// des motifs est celui des valeurs ).
/// `conf` : LE RAYON DE CONFIANCE, en fractions du rayon de la cellule. Le polynome suppose la
/// combinatoire FIXE ; il cesse d'etre credible bien avant que la cellule n'ait parcouru sa propre
/// taille. `t_conf = conf . |v| / |v'|` est le pas au bout duquel le sommet LE PLUS RAPIDE a
/// bouge de `conf` fois le rayon de la cellule -- une region de confiance au sens propre, et elle
/// repare le point faible de `croise2v` : la VITESSE est bien moins bien conditionnee que la
/// position. Les deux resolvent le meme systeme, mais le second membre de la vitesse porte
/// l'echelle de la DIRECTION DE NEWTON et non celle de la geometrie, si bien que le meme
/// `SEUIL_DET` y est beaucoup trop laxiste : `|v'| <~ |o'| / ( SEUIL_DET . min( |a|, |b| ) )`.
/// `conf = 0` desactive -- et `amin[ 0 ]` garde le `alpha*` NU, pour qu'on puisse mesurer si la
/// region de confiance mord vraiment.
template<bool POIDS, int FIXE, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_alpha( Arbre<TK,2> ar, const int *fac_j, const int *rang_de,
                                                       int NF, double seuil, double conf, double *pol,
                                                       unsigned long long *amin ) {
    constexpr int SUR = 3;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= ar.n ) return;
    const int i0 = ar.ids[ k ];
    const size_t n = ar.n;

    const int u0[ 2 ] = { FIXE == 32 ? ar.u[ 0 ][ k ] : 0, FIXE == 32 ? ar.u[ 1 ][ k ] : 0 };
    const long long g0[ 2 ] = { FIXE == 64 ? ar.u64[ 0 ][ k ] : 0, FIXE == 64 ? ar.u64[ 1 ][ k ] : 0 };
    const TK p0[ 2 ] = { FIXE == 32 ? TK( u0[ 0 ] ) * TK( INV_FIXE ) : FIXE == 64 ? TK( double( g0[ 0 ] ) * INV_F64 ) : ar.c[ 0 ][ k ],
                         FIXE == 32 ? TK( u0[ 1 ] ) * TK( INV_FIXE ) : FIXE == 64 ? TK( double( g0[ 1 ] ) * INV_F64 ) : ar.c[ 1 ][ k ] };
    const double w0d = POIDS ? ar.w64[ k ] : 0.0;
    const double d0  = ar.d64 ? ar.d64[ k ] : 0.0;

    // ---- combien d'aretes : les cases pleines sont contigues depuis zero
    int nb = 0;
    while ( nb < NF && fac_j[ size_t( nb ) * n + i0 ] != -1000000 ) ++nb;
    if ( nb < SUR ) {                                    // cellule vide ou degeneree : aucun pas
        pol[ i0 ] = 0; pol[ n + i0 ] = 0; pol[ 2 * n + i0 ] = 0; pol[ 3 * n + i0 ] = 0;
        if ( seuil > 0 ) { atomicMin( amin, 0ull ); atomicMin( amin + 1, 0ull ); }
        return;
    }

    // le rang du germe voisin ( les cotes de boite gardent leur code negatif )
    auto cid_de = [ & ]( int e ) {
        const int j = fac_j[ size_t( e ) * n + i0 ];
        return j >= 0 ? rang_de[ j ] : j;
    };
    auto plan = [ & ]( int e, double &nx, double &ny, double &off, double &dof ) {
        plan_relu<POIDS,FIXE>( ar, cid_de( e ), u0, g0, p0, w0d, nx, ny, off, d0, &dof );
    };

    // ---- LE LACET, EN STREAMING. Le sommet `i` est porte par l'arete `i - 1` ( celle qui y
    //      arrive ) et l'arete `i` ( celle qui en part ) : on remonte les plans un par un, chacun
    //      servant deux fois, et on accumule les trois coefficients au passage.
    double ax, ay, ao, ad;
    plan( nb - 1, ax, ay, ao, ad );
    double c0 = 0, c1 = 0, c2 = 0;
    double v0x = 0, v0y = 0, q0x = 0, q0y = 0;           // le sommet zero, pour fermer
    double ppx = 0, ppy = 0, qpx = 0, qpy = 0;           // le precedent : position et vitesse
    bool sur = true;                                     // le polynome est-il digne de foi ?
    double ext2 = 0, vit2 = 0;                           // pour la region de confiance
    for ( int i = 0; i < nb; ++i ) {
        double bx, by, bo, bd, vx, vy, px, py;
        plan( i, bx, by, bo, bd );
        // DEUX PLANS TROP PARALLELES : ici, contrairement au raffinement, il n'y a pas de sommet
        // de repli a garder -- on n'a que les plans. Mettre le sommet a zero corromprait le lacet
        // et rendrait un `alpha*` faux, donc on DECLARE la cellule non fiable : son polynome
        // n'aura pas voix au chapitre, et c'est la validation par un diagramme qui tranchera.
        if ( ! croise2v( ax, ay, ao, ad, bx, by, bo, bd, vx, vy, px, py ) ) { vx = vy = px = py = 0; sur = false; }
        ax = bx; ay = by; ao = bo; ad = bd;
        ext2 = fmax( ext2, vx * vx + vy * vy );          // le rayon de la cellule, au carre
        vit2 = fmax( vit2, px * px + py * py );          // la vitesse du sommet le plus rapide
        if ( i == 0 ) { v0x = vx; v0y = vy; q0x = px; q0y = py; }
        else {
            c0 += ppx * vy  - vx  * ppy;
            c1 += ppx * py  + qpx * vy - vx * qpy - px * ppy;
            c2 += qpx * py  - px  * qpy;
        }
        ppx = vx; ppy = vy; qpx = px; qpy = py;
    }
    c0 += ppx * v0y - v0x * ppy;
    c1 += ppx * q0y + qpx * v0y - v0x * qpy - q0x * ppy;
    c2 += qpx * q0y - q0x * qpy;

    // le lacet a le signe de l'orientation ; l'aire est son module, et le signe ne change pas
    // sur l'intervalle qui nous interesse
    const double sg = c0 >= 0 ? 0.5 : -0.5;
    const double a0 = sg * c0, a1 = sg * c1, a2 = sg * c2;
    const double t_conf = vit2 > 0 ? conf * sqrt( ext2 / vit2 ) : 1e300;
    pol[ i0 ] = a0; pol[ n + i0 ] = a1; pol[ 2 * n + i0 ] = a2; pol[ 3 * n + i0 ] = t_conf;

    if ( seuil > 0 ) {
        // une cellule non fiable ne CONTRAINT pas -- l'optimisme est rattrapable ( on rabote le
        // pas ), le pessimisme gelerait le Newton
        const double al = sur ? racine_pos( a2, a1, a0 - seuil ) : 1e300;
        atomicMin( amin,     ( unsigned long long ) __double_as_longlong( al ) );
        atomicMin( amin + 1, ( unsigned long long ) __double_as_longlong( conf > 0 ? fmin( al, t_conf ) : al ) );
    }
}

} // namespace
} // namespace sf::gpu
