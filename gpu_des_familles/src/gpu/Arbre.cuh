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
    const int         *u[ D ];  ///< positions en VIRGULE FIXE 32 bits ( echelle `ECH_FIXE` )
    const long long   *u64[ D ];///< les memes en VIRGULE FIXE 64 bits ( echelle `ECH_F64` )
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

// LA VIRGULE FIXE. Le `float` range `[ 0, 1 ]` avec un pas qui vaut `~6e-8` pres de 1 : une
// cellule de cote `1/sqrt( n )` n'a donc plus que `sqrt( n ) x 6e-8` de precision RELATIVE, et
// cette perte est faite AVANT le noyau, dans le stockage. En virgule fixe sur 31 bits le pas est
// UNIFORME et vaut `2^-30 = 9.3e-10`, soit ~64 fois mieux, POUR LES MEMES QUATRE OCTETS -- et la
// difference de deux positions est une soustraction entiere EXACTE, convertie ensuite en flottant
// avec toute sa precision relative. `[ 0, 1 ]` devient `[ 0, 2^30 ]` ( le sommet du carre unite
// tombe pile sur `2^30` ), et l'ecart tient toujours dans un `int`.
constexpr int    ECH_FIXE = 1 << 30;
constexpr double INV_FIXE = 1.0 / double( ECH_FIXE );

// EN 64 BITS, le pas tombe a `2^-52 = 2.2e-16` : `dx` n'est plus quantifie du tout a l'echelle du
// `float`, et la conversion `long long -> float` arrondit a 24 bits RELATIFS A `dx`, pas a 1.
// L'echelle `2^52` se convertit exactement depuis un `double` d'entree ( `v x 2^52 < 2^53` ), donc
// la position stockee est l'arrondi exact de l'entree -- on ne peut pas faire mieux sans entree
// plus precise que le `double`. Prix : huit octets par coordonnee au lieu de quatre.
constexpr long long ECH_F64 = 1ll << 52;
constexpr double    INV_F64 = 1.0 / double( ECH_F64 );

/// le plan bissecteur DANS LE REPERE DU GERME, les positions lues en virgule fixe
template<bool POIDS, class TK>
__device__ __forceinline__ Plan2<TK> bissect2f( const Arbre<TK,2> &ar, int q, int ux0, int uy0, TK w0 ) {
    Plan2<TK> p;
    p.dx  = TK( ar.u[ 0 ][ q ] - ux0 ) * TK( INV_FIXE );   // soustraction EXACTE, puis conversion
    p.dy  = TK( ar.u[ 1 ][ q ] - uy0 ) * TK( INV_FIXE );
    p.off = TK( 0.5 ) * ( p.dx * p.dx + p.dy * p.dy );
    if constexpr ( POIDS ) p.off += TK( 0.5 ) * ( w0 - ar.w[ q ] );
    p.id  = ar.ids[ q ];
    return p;
}

/// le plan bissecteur dans le repere du germe, positions en virgule fixe 64 bits
template<bool POIDS, class TK>
__device__ __forceinline__ Plan2<TK> bissect2g( const Arbre<TK,2> &ar, int q, long long ux0, long long uy0, TK w0 ) {
    Plan2<TK> p;
    p.dx  = TK( ar.u64[ 0 ][ q ] - ux0 ) * TK( INV_F64 );  // soustraction 64 bits EXACTE
    p.dy  = TK( ar.u64[ 1 ][ q ] - uy0 ) * TK( INV_F64 );
    p.off = TK( 0.5 ) * ( p.dx * p.dx + p.dy * p.dy );
    if constexpr ( POIDS ) p.off += TK( 0.5 ) * ( w0 - ar.w[ q ] );
    p.id  = ar.ids[ q ];
    return p;
}

/// LE MEME PLAN, DANS LE REPERE CENTRE SUR LE GERME ( `v` compte a partir de `p0` ). En absolu
/// `off = 1/2 ( dx ( xj + x0 ) + dy ( yj + y0 ) )` melange du petit ( `dx ~ 1/sqrt( n )` ) et du
/// grand ( `xj + x0 ~ 1` ) : le produit vaut `~1/sqrt( n )` alors que `dx . v` en vaut autant, si
/// bien que `s = dx vx + dy vy - off` est une DIFFERENCE DE DEUX GRANDS QUI DONNE UN PETIT -- la
/// cellule fait `1/sqrt( n )` de cote, donc `s ~ 1/n`, et on perd `log2( sqrt( n ) )` bits.
/// Centre, la meme expression devient `off = 1/2 ( dx^2 + dy^2 )` : tout est a l'echelle de la
/// cellule, `dx vx` et `off` valent `~1/n` tous les deux, et il n'y a plus de cancellation.
template<bool POIDS, class TK>
__device__ __forceinline__ Plan2<TK> bissect2c( const Arbre<TK,2> &ar, int q, TK x0, TK y0, TK w0 ) {
    Plan2<TK> p;
    p.dx  = ar.c[ 0 ][ q ] - x0;
    p.dy  = ar.c[ 1 ][ q ] - y0;
    p.off = TK( 0.5 ) * ( p.dx * p.dx + p.dy * p.dy );
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

/// LE MEME ELAGAGE, dans le repere centre : `V = v - p0`, et la boite du noeud decalee de `p0`.
/// ( `v[ d ] - p0[ d ]` disparait, `y` revient en absolu pour le seul terme des poids. )
template<bool POIDS, class TK, int D>
__device__ __forceinline__ TK bilan_sommet_c( const Noeud<TK,D> &B, const TK *V, const TK *p0, TK w0 ) {
    TK s = POIDS ? w0 - B.b : TK( 0 );
#pragma unroll
    for ( int d = 0; d < D; ++d ) {
        const TK lo = B.lo[ d ] - p0[ d ], hi = B.hi[ d ] - p0[ d ];
        TK y = V[ d ] + ( POIDS ? TK( 0.5 ) * B.a[ d ] : TK( 0 ) );
        y = mn( mx( y, lo ), hi );
        const TK u = y - V[ d ];
        s += u * u - V[ d ] * V[ d ];
        if constexpr ( POIDS ) s -= B.a[ d ] * ( y + p0[ d ] );
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
