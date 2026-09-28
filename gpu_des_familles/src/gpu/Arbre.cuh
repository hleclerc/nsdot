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
    /// LES MEMES POIDS EN `double`. Ce n'est pas un luxe : le plan bissecteur porte
    /// `( w0 - wj ) / 2`, et arrondir CHAQUE POIDS avant la difference lui coute `eps |w|`, donc
    /// deplace le plan de `eps |w| / ( 2 |d| )`. Rapporte a la taille d'une cellule cela fait
    /// `eps |w| n^( 2/D )` -- une erreur QUI GRANDIT AVEC `n`. Or les poids ne sont grands que
    /// dans l'absolu : pour deux germes qui partagent vraiment une facette, `| w0 - wj | <~ h^2`,
    /// du meme ordre que le terme geometrique. On porte donc LA DIFFERENCE, pas la valeur : elle
    /// se calcule en `double` et n'est arrondie qu'une fois. Voir `cell/Plan.h` du banc CPU.
    const double      *w64;
    const int         *u[ D ];  ///< positions en VIRGULE FIXE 32 bits ( echelle `ECH_FIXE` )
    const long long   *u64[ D ];///< les memes en VIRGULE FIXE 64 bits ( echelle `ECH_F64` )
    const int         *ids;     ///< rang -> identifiant
    int                n;
};

/// LE PLAN BISSECTEUR de `[ p0, pj ]`, oriente pour que `p0` soit DEDANS : `d . x <= off`.
///
/// `id` EST LE RANG DU GERME DANS L'ARBRE, pas son identifiant d'appelant. Deux raisons. La
/// premiere est gratuite : la traduction `rang -> identifiant` etait une lecture dispersee PAR
/// GERME TESTE, dans la boucle la plus chaude ; elle n'a lieu maintenant que sur les six aretes
/// du polygone fini. La seconde est qu'avec le rang, LE PLAN SE RELIT : c'est ce qui permet de
/// resoudre chaque sommet depuis les deux coupes qui le portent ( `raffine2`, `FilMsk2D.cuh` ).
/// Les cotes du carre gardent leur code negatif `-1 - d`, un rang etant toujours `>= 0`.
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
    p.id  = q;
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
__device__ __forceinline__ Plan2<TK> bissect2f( const Arbre<TK,2> &ar, int q, int ux0, int uy0, double w0 ) {
    Plan2<TK> p;
    p.dx  = TK( ar.u[ 0 ][ q ] - ux0 ) * TK( INV_FIXE );   // soustraction EXACTE, puis conversion
    p.dy  = TK( ar.u[ 1 ][ q ] - uy0 ) * TK( INV_FIXE );
    p.off = TK( 0.5 ) * ( p.dx * p.dx + p.dy * p.dy );
    if constexpr ( POIDS ) p.off += TK( 0.5 * ( w0 - ar.w64[ q ] ) );   // LA DIFFERENCE, pas la valeur
    p.id  = q;
    return p;
}

/// le plan bissecteur dans le repere du germe, positions en virgule fixe 64 bits
template<bool POIDS, class TK>
__device__ __forceinline__ Plan2<TK> bissect2g( const Arbre<TK,2> &ar, int q, long long ux0, long long uy0, double w0 ) {
    Plan2<TK> p;
    p.dx  = TK( ar.u64[ 0 ][ q ] - ux0 ) * TK( INV_F64 );  // soustraction 64 bits EXACTE
    p.dy  = TK( ar.u64[ 1 ][ q ] - uy0 ) * TK( INV_F64 );
    p.off = TK( 0.5 ) * ( p.dx * p.dx + p.dy * p.dy );
    if constexpr ( POIDS ) p.off += TK( 0.5 * ( w0 - ar.w64[ q ] ) );
    p.id  = q;
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
__device__ __forceinline__ Plan2<TK> bissect2c( const Arbre<TK,2> &ar, int q, TK x0, TK y0, double w0 ) {
    Plan2<TK> p;
    p.dx  = ar.c[ 0 ][ q ] - x0;
    p.dy  = ar.c[ 1 ][ q ] - y0;
    p.off = TK( 0.5 ) * ( p.dx * p.dx + p.dy * p.dy );
    if constexpr ( POIDS ) p.off += TK( 0.5 * ( w0 - ar.w64[ q ] ) );
    p.id  = q;
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
    p.id  = q;
    return p;
}

// =====================================================================================
// LES SOMMETS RESOLUS DEPUIS LEURS PLANS.
//
// Un sommet d'un convexe ne depend PAS de l'histoire des coupes : il est l'intersection de `D`
// plans, et rien d'autre. Le noyau, lui, le construit par interpolations successives, si bien
// qu'il porte l'erreur de la cellule TELLE QU'ELLE ETAIT quand il est ne -- `eps L` pour une
// cellule large de `L`, alors qu'il ne devrait porter que `eps h`. C'est le dernier terme du
// modele `fp32`, celui qui survit au repere du germe et a la virgule fixe.
//
// On resout donc chaque sommet A LA FIN, depuis les deux coupes qui le portent -- elles sont
// deja la, l'arete `i` va du sommet `i` au sommet `i + 1` et le sommet `i` est l'intersection de
// l'arete `i - 1` et de l'arete `i`. Ce que ca demande, c'est de savoir RELIRE un plan depuis
// son `cid`, d'ou le rang plutot que l'identifiant dans `Plan2::id`.
//
// Deux choses en font une reparation faite pour une carte. Elle NE BRANCHE PAS -- une elimination
// de Gauss par sommet, la meme suite d'instructions pour tous les threads, pas de reprise, pas de
// file de rattrapage. Et elle remplace les BOITES DE DEPART du banc CPU, qui elles divergeaient :
// une boite mal dimensionnee faisait recommencer la cellule, donc son warp entier.
//
// Le systeme mal conditionne ( deux plans presque paralleles ) est LAISSE tel quel : le sommet du
// noyau y est au moins aussi bon, et on ne veut pas remplacer une erreur portee par un quotient
// qui explose.
// =====================================================================================

/// sous ce determinant RELATIF, on garde le sommet du noyau ( `SF_DET` au CPU )
constexpr double SEUIL_DET = 1e-6;

/// LE PLAN D'UNE COUPE, RELU DEPUIS SON `cid`, EN `double` et dans le repere du germe. C'est le
/// meme plan que la coupe a utilise, mais sans l'arrondi final au flottant du noyau.
template<bool POIDS, int FIXE, class TK>
__device__ __forceinline__ void plan_relu( const Arbre<TK,2> &ar, int cid, const int *u0,
                                           const long long *g0, const TK *p0, double w0,
                                           double &nx, double &ny, double &off ) {
    if ( cid >= 0 ) {                                    // un germe : le bissecteur
        nx = FIXE == 32 ? double( ar.u[ 0 ][ cid ] - u0[ 0 ] ) * INV_FIXE
           : FIXE == 64 ? double( ar.u64[ 0 ][ cid ] - g0[ 0 ] ) * INV_F64
                        : double( ar.c[ 0 ][ cid ] ) - double( p0[ 0 ] );
        ny = FIXE == 32 ? double( ar.u[ 1 ][ cid ] - u0[ 1 ] ) * INV_FIXE
           : FIXE == 64 ? double( ar.u64[ 1 ][ cid ] - g0[ 1 ] ) * INV_F64
                        : double( ar.c[ 1 ][ cid ] ) - double( p0[ 1 ] );
        off = 0.5 * ( nx * nx + ny * ny );
        if constexpr ( POIDS ) off += 0.5 * ( w0 - ar.w64[ cid ] );
        return;
    }
    // LES QUATRE COTES DU CARRE, dans l'ordre que pose le noyau ( l'arete `i` va du sommet `i` au
    // suivant, et le carre part de `( 0, 0 ), ( 1, 0 ), ( 1, 1 ), ( 0, 1 )` ) : `-1` le bas,
    // `-2` la droite, `-3` le haut, `-4` la gauche. Dans le repere du germe leur `off` est
    // exactement la coordonnee du carre de depart -- donc exact en virgule fixe.
    const int f = -1 - cid;
    const bool vert = ( f == 1 || f == 3 );              // `x = cte`
    const bool haut = ( f == 1 || f == 2 );              // le cote a un
    nx = vert ? 1.0 : 0.0;
    ny = vert ? 0.0 : 1.0;
    const int d = vert ? 0 : 1;
    off = FIXE == 32 ? double( ( haut ? ECH_FIXE : 0 ) - u0[ d ] ) * INV_FIXE
        : FIXE == 64 ? double( ( haut ? ECH_F64 : 0ll ) - g0[ d ] ) * INV_F64
                     : ( haut ? 1.0 : 0.0 ) - double( p0[ d ] );
}

/// LE SOMMET porte par les plans `a` ( l'arete qui arrive ) et `b` ( celle qui part ). Rend
/// `false` si le systeme est trop mal conditionne pour qu'on y gagne.
__device__ __forceinline__ bool croise2( double ax, double ay, double ao,
                                         double bx, double by, double bo, double &vx, double &vy ) {
    const double det = ax * by - ay * bx;
    const double e1 = ax * ax + ay * ay, e2 = bx * bx + by * by;
    if ( ! ( det * det > SEUIL_DET * SEUIL_DET * e1 * e2 ) )
        return false;
    vx = ( ao * by - bo * ay ) / det;
    vy = ( ax * bo - bx * ao ) / det;
    return true;
}

/// LA TRADUCTION `rang -> identifiant d'appelant`, appliquee AUX SEULES ARETES DU POLYGONE FINI
/// ( les cid negatifs, cotes du carre, passent tels quels ).
template<class TK, int D>
__device__ __forceinline__ int id_de( const Arbre<TK,D> &ar, int cid ) {
    return cid >= 0 ? ar.ids[ cid ] : cid;
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
