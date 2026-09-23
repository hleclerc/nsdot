#pragma once

// =====================================================================================
// L'ARBRE TEL QUE LE GPU LE LIT. Le meme BSP median que `accel/AaBsp.h`, bati sur l'hote, puis
// televerse : les noeuds en PREORDRE ( le fils gauche est en `n + 1`, le droit est stocke ), les
// germes RANGES DANS L'ORDRE DE L'ARBRE dans le flottant du noyau, et les identifiants.
//
// Un noeud est un agregat ALIGNE SUR 16 OCTETS : les threads d'un warp lisent des noeuds
// differents ( le parcours diverge ), donc ce qui compte est le nombre de transactions par noeud,
// et un agregat aligne se lit en chargements de 16 octets. Les pentes du majorant sont deja dans
// le flottant du noyau : pas de conversion dans le test.
//
// Le thread `k` fait la cellule du germe de RANG `k` : deux threads voisins d'un warp sont voisins
// dans l'arbre, donc voisins dans l'espace, donc leurs parcours se ressemblent -- c'est la seule
// chose qui limite la divergence, et elle est gratuite.
// =====================================================================================

namespace sf::gpu {

template<class TK, int D>
struct alignas( 16 ) Noeud {
    TK  lo[ D ], hi[ D ];       ///< la boite du sous-arbre
    TK  a[ D ], b;              ///< `w( q ) <= a . q + b` sur le sous-arbre ( `a = 0` en Voronoi )
    int beg, end;               ///< sa tranche de germes
    int right;                  ///< le fils droit ; `< 0` dit FEUILLE
};

/// ce qu'un noyau recoit : des pointeurs DEVICE, par valeur.
template<class TK, int D>
struct Arbre {
    const Noeud<TK,D> *nodes;
    const TK          *c[ D ];  ///< positions, ordre de l'arbre
    const TK          *w;       ///< poids, idem ( `nullptr` en Voronoi )
    const int         *ids;     ///< rang -> identifiant
    int                n;
};

/// LE PLAN BISSECTEUR de `[ p0, pj ]`, oriente pour que `p0` soit DEDANS : `d . x <= off`.
template<class TK>
struct Plan2 { TK dx, dy, off; int id; };

template<class TK>
struct Plan3 { TK dx, dy, dz, off; int id; };

template<bool POIDS, class TK>
__device__ __forceinline__ Plan2<TK> bissect2( const Arbre<TK,2> &ar, int q, TK x0, TK y0, TK w0 ) {
    Plan2<TK> p;
    const TK xj = ar.c[ 0 ][ q ], yj = ar.c[ 1 ][ q ];
    p.dx  = xj - x0;
    p.dy  = yj - y0;
    p.off = TK( 0.5 ) * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) );
    if constexpr ( POIDS ) p.off += TK( 0.5 ) * ( w0 - ar.w[ q ] );
    p.id  = ar.ids[ q ];
    return p;
}

template<bool POIDS, class TK>
__device__ __forceinline__ Plan3<TK> bissect3( const Arbre<TK,3> &ar, int q, TK x0, TK y0, TK z0, TK w0 ) {
    Plan3<TK> p;
    const TK xj = ar.c[ 0 ][ q ], yj = ar.c[ 1 ][ q ], zj = ar.c[ 2 ][ q ];
    p.dx  = xj - x0;
    p.dy  = yj - y0;
    p.dz  = zj - z0;
    p.off = TK( 0.5 ) * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) + p.dz * ( zj + z0 ) );
    if constexpr ( POIDS ) p.off += TK( 0.5 ) * ( w0 - ar.w[ q ] );
    p.id  = ar.ids[ q ];
    return p;
}

/// `min` / `max` qui sortent en UNE instruction ( `FMNMX` ) et pas en branchement.
__device__ __forceinline__ float  mn( float a, float b )   { return fminf( a, b ); }
__device__ __forceinline__ float  mx( float a, float b )   { return fmaxf( a, b ); }
__device__ __forceinline__ double mn( double a, double b ) { return fmin( a, b ); }
__device__ __forceinline__ double mx( double a, double b ) { return fmax( a, b ); }

/// LE TEST D'ELAGAGE POUR UN SOMMET, le critere de `cell/Elagage2D.h` : rend `s`, et `s <= 0` dit
/// « un germe de la boite peut retrancher ce sommet ». `min_q ( |v - q|^2 - a . q )` est
/// separable par axe, libre en `q = v + a / 2`, un clamp par axe le donne. Exact.
template<bool POIDS, class TK, int D>
__device__ __forceinline__ TK bilan_sommet( const Noeud<TK,D> &B, const TK *v, const TK *p0, TK w0 ) {
    TK s = POIDS ? w0 - B.b : TK( 0 );
#pragma unroll
    for ( int d = 0; d < D; ++d ) {
        TK y = v[ d ] + ( POIDS ? TK( 0.5 ) * B.a[ d ] : TK( 0 ) );
        y = mn( mx( y, B.lo[ d ] ), B.hi[ d ] );
        const TK u = y - v[ d ], f = v[ d ] - p0[ d ];
        s += u * u - f * f;
        if constexpr ( POIDS ) s -= B.a[ d ] * y;
    }
    return s;
}

/// le carre de la distance du germe a la boite : la clef d'ordre des deux fils, pas un test.
template<class TK, int D>
__device__ __forceinline__ TK proximite( const Noeud<TK,D> &nd, const TK *p0 ) {
    TK s = 0;
#pragma unroll
    for ( int d = 0; d < D; ++d ) {
        const TK e = mx( mx( nd.lo[ d ] - p0[ d ], p0[ d ] - nd.hi[ d ] ), TK( 0 ) );
        s += e * e;
    }
    return s;
}

/// LA PROFONDEUR MAXIMALE de la pile : a chaque niveau on depile un noeud et on en empile deux.
constexpr int PILE = 48;

} // namespace sf::gpu
