// LES ENCEINTES : un CONVEXE qui contient la reunion des cellules d'un agregat.
//
// C'est le sur-ensemble sur lequel l'etape 3 posera ses questions (« l'agregat `B` peut-il encore
// couper la cellule de `i` ? »). Deux realisations derriere la MEME interface, et le banc les
// compare sans rien changer d'autre :
//
//   `Enveloppe<D>`  l'enveloppe convexe exacte -- la plus serree qui soit. 2D : chaine monotone
//                   d'Andrew. 3D : incremental, avec passes de rattrapage et refus explicite.
//   `KDop<D,K>`     l'intersection de `K` bandes de directions FIXES -- plus lache, mais un
//                   balayage sans tri, sans allocation et sans cas degenere.
//
// L'INTERFACE : `build( pts )`, `ok`, `taille()`, `volume()`, `ecart( pts )`.
//
// `volume()` est SEPARE de `build` a dessein. C'est un TEMOIN (`somme des |env| >= somme des
// |C_i|`, convexifier ne peut qu'ajouter) et pas un calcul de production : le laisser dans `build`
// biaiserait la comparaison, puisqu'il est presque gratuit pour l'enveloppe (les faces sont deja
// la) et coute une decoupe de cellule pour le k-DOP.
//
// POURQUOI L'ENCEINTE EST EXACTE, quelle qu'elle soit. Les cellules des membres de `A` sont
// coupees par LE MEME jeu de plans, donc elles PAVENT `U_A`. Un sommet ne de plans tous INTERNES
// est partage par plusieurs cellules de `A`, donc interieur ; seuls les sommets portant un plan
// exterieur (ou une face du domaine) sont sur le bord. Tout convexe contenant ces sommets-la
// contient `U_A`, donc la reunion des vraies cellules de `A`. Il n'y a aucun pari a prendre.

#pragma once

#include "util/common.h"
#include "geometry/Cell.h"
#include "geometry/Cell3.h"
#include "bench/Bench.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace pd::supercell {

// ------------------------------------------------------------------ L'ENVELOPPE CONVEXE

/// `Vec` ne porte que `[]`, `dot` et `dist2` : le reste se fait ici, en local.
template<int D> Vec<D> moins( Vec<D> a, Vec<D> b ) {
    Vec<D> r; for ( int d = 0; d < D; ++d ) r[ d ] = a[ d ] - b[ d ]; return r;
}
inline Vec<3> croix( Vec<3> a, Vec<3> b ) {
    Vec<3> r;
    r[ 0 ] = a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ];
    r[ 1 ] = a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ];
    r[ 2 ] = a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ];
    return r;
}

/// L'ENVELOPPE CONVEXE DES SOMMETS DES CELLULES D'UN AGREGAT -- la sur-cellule convexifiee.
///
/// = POURQUOI ELLE EST EXACTE, ET CE QU'ELLE COUTE
///
/// Les cellules des membres de `A` sont toutes calculees contre LE MEME jeu de plans
/// (`A u ext( A )`), donc elles PAVENT la region `U_A = { min_A h <= min_ext h }`. Trois
/// consequences, et les trois servent :
///
///   * `U_A` recouvre le domaine quand `A` parcourt les agregats -- pour tout `x`, le germe qui
///     minimise `h` sur TOUT le nuage met son agregat dans le compte. Donc `somme des |C_i| >= 1`,
///     avec egalite si et seulement si les sur-cellules ne se recouvrent pas. C'est le temoin de
///     l'etape, l'exact analogue du « somme des mesures = 1 » de l'etape 1.
///   * un sommet ne de deux (2D) ou trois (3D) plans INTERNES est partage par plusieurs cellules
///     de `A`, donc il est a l'INTERIEUR de `U_A` : seuls les sommets portant au moins un plan
///     exterieur (ou une face du domaine) sont sur le bord. Le filtre est gratuit et exact, et on
///     mesure ce qu'il enleve.
///   * `conv( U_A ) ⊇ U_A ⊇ la reunion des vraies cellules de A` : l'enveloppe est un sur-ensemble
///     PAR CONSTRUCTION, il n'y a aucun pari a prendre. Son exces de volume est le prix de la
///     convexification, et il se lit dans le meme compte.
///
/// 2D : chaine monotone d'Andrew, exacte et sans cas particulier. 3D : incremental (on part d'un
/// tetraedre et on recolle un cone sur l'horizon a chaque point exterieur). Le 3D peut echouer sur
/// une entree degeneree -- il rend alors `ok = false` et l'appelant COMPTE l'echec au lieu de le
/// masquer.
template<int D>
struct Enveloppe {
    std::vector<Vec<D>>            som;             ///< les sommets de l'enveloppe
    std::vector<std::array<SI,3>>  fac;             ///< 3D : les faces, indexant `som`, DEHORS
    TF     ech = 0;                                 ///< l'echelle du nuage, pour les tolerances
    bool   ok  = false;

    /// LA BOITE DE REPLI, enregistree AVANT toute construction. L'enveloppe incrementale 3D
    /// refuse 0.5 a 2 % des agregats (nuages quasi coplanaires) ; sans repli, l'etape 3 n'aurait
    /// rien a interroger pour ceux-la et la table aurait un trou. La boite alignee des sommets du
    /// bord contient `U_A` tout autant -- moins fine, jamais fausse.
    /// faux : les axes de l'enveloppe sont les axes CANONIQUES, dont les produits
    /// vectoriels redonnent les axes canoniques. Rien a en tirer.
    static constexpr bool croise = false;

    Vec<D> blo{}, bhi{};

    void build( std::vector<Vec<D>> &pts ) {
        som.clear(); fac.clear(); ok = false;
        if ( pts.empty() ) return;
        blo = bhi = pts[ 0 ];
        for ( const Vec<D> &p : pts )
            for ( int d = 0; d < D; ++d ) {
                blo[ d ] = std::min( blo[ d ], p[ d ] ); bhi[ d ] = std::max( bhi[ d ], p[ d ] ); }
        if constexpr ( D == 2 ) build2( pts );
        else                    build3( pts );
    }

    void build2( std::vector<Vec<2>> &pts ) requires ( D == 2 ) {
        if ( pts.size() < 3 ) { som.assign( pts.begin(), pts.end() ); return; }
        std::sort( pts.begin(), pts.end(), []( Vec<2> a, Vec<2> b ) {
            return a[ 0 ] != b[ 0 ] ? a[ 0 ] < b[ 0 ] : a[ 1 ] < b[ 1 ]; } );
        // LE DEDOUBLONNAGE, et il n'est pas cosmetique. Un sommet du bord de `U_A` est calcule par
        // DEUX cellules voisines, dans deux ordres de coupes differents : il revient a un ulp pres.
        // Garde tel quel, il fait une arete d'enveloppe de longueur 1e-20 -- une normale sans
        // signification, et tout ce qui divise par elle explose. On les fond ici, une fois.
        ech = 0;
        for ( const Vec<2> &p : pts ) for ( int d = 0; d < 2; ++d ) ech = std::max( ech, std::fabs( p[ d ] ) );
        const TF tolp = ech * TF( 1e-13 );
        pts.erase( std::unique( pts.begin(), pts.end(), [ & ]( Vec<2> a, Vec<2> b ) {
            return std::fabs( a[ 0 ] - b[ 0 ] ) <= tolp && std::fabs( a[ 1 ] - b[ 1 ] ) <= tolp; } ),
                   pts.end() );
        const SI m = SI( pts.size() );
        if ( m < 3 ) { som.assign( pts.begin(), pts.end() ); return; }
        auto cr = []( Vec<2> o, Vec<2> a, Vec<2> b ) {
            return ( a[ 0 ] - o[ 0 ] ) * ( b[ 1 ] - o[ 1 ] )
                 - ( a[ 1 ] - o[ 1 ] ) * ( b[ 0 ] - o[ 0 ] ); };
        std::vector<Vec<2>> h( 2 * m );
        SI k = 0;
        for ( SI i = 0; i < m; ++i ) {                          // la chaine BASSE
            while ( k >= 2 && cr( h[ k - 2 ], h[ k - 1 ], pts[ i ] ) <= 0 ) --k;
            h[ k++ ] = pts[ i ];
        }
        for ( SI i = m - 2, t = k + 1; i >= 0; --i ) {           // ... puis la HAUTE
            while ( k >= t && cr( h[ k - 2 ], h[ k - 1 ], pts[ i ] ) <= 0 ) --k;
            h[ k++ ] = pts[ i ];
        }
        som.assign( h.begin(), h.begin() + ( k - 1 ) );          // CCW, sans point aligne
        ok = som.size() >= 3;
    }

    void build3( std::vector<Vec<3>> &pts ) requires ( D == 3 ) {
        if ( pts.size() < 4 ) return;
        Vec<3> lo = pts[ 0 ], hi = pts[ 0 ];
        for ( SI i = 1; i < SI( pts.size() ); ++i )
            for ( int d = 0; d < 3; ++d ) {
                lo[ d ] = std::min( lo[ d ], pts[ i ][ d ] );
                hi[ d ] = std::max( hi[ d ], pts[ i ][ d ] );
            }
        ech = 0;
        for ( int d = 0; d < 3; ++d ) ech = std::max( ech, hi[ d ] - lo[ d ] );
        if ( ech <= 0 ) return;
        // meme dedoublonnage qu'en 2D, et pour la meme raison : un sommet du bord est calcule par
        // plusieurs cellules et revient a un ulp pres.
        std::sort( pts.begin(), pts.end(), []( Vec<3> a, Vec<3> b ) {
            for ( int d = 0; d < 3; ++d ) if ( a[ d ] != b[ d ] ) return a[ d ] < b[ d ];
            return false; } );
        const TF tolp = ech * TF( 1e-13 );
        pts.erase( std::unique( pts.begin(), pts.end(), [ & ]( Vec<3> a, Vec<3> b ) {
            for ( int d = 0; d < 3; ++d ) if ( std::fabs( a[ d ] - b[ d ] ) > tolp ) return false;
            return true; } ), pts.end() );
        const TF eps = ech * ech * ech * TF( 1e-13 );

        auto dessus = [ & ]( const std::array<SI,3> &t, Vec<3> p ) {
            const Vec<3> n = croix( moins( pts[ t[ 1 ] ], pts[ t[ 0 ] ] ),
                                    moins( pts[ t[ 2 ] ], pts[ t[ 0 ] ] ) );
            return dot( n, moins( p, pts[ t[ 0 ] ] ) );
        };
        // LA VISIBILITE, NORMALISEE -- et c'est la que se joue la justesse. Le produit mixte
        // `n . ( p - v0 )` vaut `|n| x distance` : sur une PETITE face `|n|` est minuscule, donc un
        // seuil homogene a `ech^3` declare « pas visible » une face que le point survole
        // largement. Le point finit dehors, et l'enveloppe est fausse de plusieurs `h` -- c'est
        // exactement ce qu'on mesurait. On compare donc une DISTANCE, sans racine carree.
        const TF ed = ech * TF( 1e-12 );
        auto visible = [ & ]( const std::array<SI,3> &t, Vec<3> p ) {
            const Vec<3> n = croix( moins( pts[ t[ 1 ] ], pts[ t[ 0 ] ] ),
                                    moins( pts[ t[ 2 ] ], pts[ t[ 0 ] ] ) );
            const TF v = dot( n, moins( p, pts[ t[ 0 ] ] ) );
            return v > 0 && v * v > ed * ed * dot( n, n );
        };

        const SI m = SI( pts.size() );
        if ( m < 4 ) return;

        // --- une base affine : extreme, le plus loin, le plus loin de la droite, du plan
        SI ia = 0;
        for ( SI i = 1; i < m; ++i ) if ( pts[ i ][ 0 ] < pts[ ia ][ 0 ] ) ia = i;
        SI ib = -1; TF bb = 0;
        for ( SI i = 0; i < m; ++i ) { const TF d = dist2( pts[ i ], pts[ ia ] );
                                       if ( d > bb ) { bb = d; ib = i; } }
        if ( ib < 0 ) return;
        SI ic = -1; TF bc = 0;
        const Vec<3> ab = moins( pts[ ib ], pts[ ia ] );
        for ( SI i = 0; i < m; ++i ) { const Vec<3> n = croix( ab, moins( pts[ i ], pts[ ia ] ) );
                                       const TF d = dot( n, n );
                                       if ( d > bc ) { bc = d; ic = i; } }
        if ( ic < 0 ) return;
        const Vec<3> nr = croix( ab, moins( pts[ ic ], pts[ ia ] ) );
        SI id = -1; TF bd = 0;
        for ( SI i = 0; i < m; ++i ) { const TF v = dot( nr, moins( pts[ i ], pts[ ia ] ) );
                                       if ( std::fabs( v ) > std::fabs( bd ) ) { bd = v; id = i; } }
        if ( id < 0 || std::fabs( bd ) <= eps ) return;     // tout est coplanaire

        const SI base[ 4 ] = { ia, ib, ic, id };
        for ( int f = 0; f < 4; ++f ) {
            std::array<SI,3> t{};
            int k = 0;
            for ( int g = 0; g < 4; ++g ) if ( g != f ) t[ k++ ] = base[ g ];
            if ( dessus( t, pts[ base[ f ] ] ) > 0 ) std::swap( t[ 1 ], t[ 2 ] );
            fac.push_back( t );
        }

        std::vector<char> dedans( m, 0 ), vis;
        for ( int g = 0; g < 4; ++g ) dedans[ base[ g ] ] = 1;
        std::vector<std::array<SI,2>> ar, hor;
        std::vector<std::array<SI,3>> nf;
        bool casse = false;
        auto insere = [ & ]( SI i ) {
            vis.assign( fac.size(), 0 );
            bool une = false;
            for ( size_t f = 0; f < fac.size(); ++f )
                if ( visible( fac[ f ], pts[ i ] ) ) { vis[ f ] = 1; une = true; }
            if ( ! une ) return;
            // L'HORIZON : une arete dirigee des faces visibles dont la RECIPROQUE n'y est pas.
            ar.clear();
            for ( size_t f = 0; f < fac.size(); ++f ) if ( vis[ f ] )
                for ( int e = 0; e < 3; ++e )
                    ar.push_back( { fac[ f ][ e ], fac[ f ][ ( e + 1 ) % 3 ] } );
            hor.clear();
            for ( const auto &e : ar ) {
                bool jum = false;
                for ( const auto &g : ar ) if ( g[ 0 ] == e[ 1 ] && g[ 1 ] == e[ 0 ] ) { jum = true; break; }
                if ( ! jum ) hor.push_back( e );
            }
            if ( hor.empty() ) { casse = true; return; }    // surface incoherente : on refuse
            nf.clear();
            for ( size_t f = 0; f < fac.size(); ++f ) if ( ! vis[ f ] ) nf.push_back( fac[ f ] );
            // le cone : l'arete GARDE son sens, donc l'orientation reste coherente de proche en
            // proche -- et comme le tetraedre de depart est oriente dehors, tout l'est.
            for ( const auto &e : hor ) nf.push_back( { e[ 0 ], e[ 1 ], i } );
            fac.swap( nf );
            dedans[ i ] = 1;
            if ( SI( fac.size() ) > 4 * m + 16 ) casse = true;   // garde-fou : Euler dit `2m - 4`
        };
        for ( SI i = 0; i < m && ! casse; ++i ) if ( ! dedans[ i ] ) insere( i );
        // LES PASSES DE RATTRAPAGE. Un point peut rester dehors quand l'ensemble des faces qu'il
        // voit n'est pas connexe -- l'horizon n'est alors pas un cycle unique et le cone est
        // recolle de travers. Plutot que d'esperer, on REPASSE tant qu'il reste un point dehors,
        // et si trois passes n'y suffisent pas on rend `ok = false` : l'appelant compte l'echec.
        for ( int passe = 0; passe < 3 && ! casse; ++passe ) {
            bool reste = false;
            for ( SI i = 0; i < m && ! casse; ++i ) {
                bool dehors = false;
                for ( size_t f = 0; f < fac.size() && ! dehors; ++f ) dehors = visible( fac[ f ], pts[ i ] );
                if ( dehors ) { insere( i ); reste = true; }
            }
            if ( ! reste ) break;
        }
        if ( casse ) return;
        // ... et on VERIFIE que les passes ont suffi. Sans ce test, une enveloppe restee fausse
        // repartait comme bonne : c'est exactement ce qui donnait un ecart de plusieurs `h`.
        for ( SI i = 0; i < m; ++i )
            for ( size_t f = 0; f < fac.size(); ++f )
                if ( visible( fac[ f ], pts[ i ] ) ) return;

        // --- les sommets utilises, et la reindexation de `fac` sur `som`
        std::vector<SI> pos( m, -1 );
        for ( const auto &f : fac ) for ( int e = 0; e < 3; ++e )
            if ( pos[ f[ e ] ] < 0 ) { pos[ f[ e ] ] = SI( som.size() ); som.push_back( pts[ f[ e ] ] ); }
        for ( auto &f : fac ) for ( int e = 0; e < 3; ++e ) f[ e ] = pos[ f[ e ] ];

        ok = som.size() >= 4;
    }

    SI taille() const { return SI( som.size() ); }

    // ---- L'INTERROGATION, ce que l'etape 3 demande a une enceinte.
    //
    // `nb_axes` / `axe` / `bande` : les directions sur lesquelles CETTE enceinte sait donner un
    // intervalle EXACT, et cet intervalle. `support( n )` : un MAJORANT de `max n . x` sur
    // l'enceinte, pour une direction quelconque -- exact ici (on a les sommets), majore pour un
    // k-DOP hors de ses propres directions. Un majorant suffit : il ne peut que faire RATER une
    // separation, donc garder une paire de trop dans la table. Jamais en perdre une.
    SI     nb_axes() const { return D; }
    Vec<D> axe( SI k ) const { Vec<D> n{}; for ( int d = 0; d < D; ++d ) n[ d ] = ( d == k ); return n; }
    void   bande( SI k, TF &l, TF &h ) const { l = blo[ k ]; h = bhi[ k ]; }
    void   boite( Vec<D> &l, Vec<D> &h ) const { l = blo; h = bhi; }
    TF support( Vec<D> n ) const {
        if ( ! ok ) {                                   // repli : le support de la boite
            TF s = 0;
            for ( int d = 0; d < D; ++d )
                s += ( blo[ d ] + bhi[ d ] ) / 2 * n[ d ]
                   + ( bhi[ d ] - blo[ d ] ) / 2 * std::fabs( n[ d ] );
            return s;
        }
        TF m = dot( n, som[ 0 ] );
        for ( const Vec<D> &p : som ) m = std::max( m, dot( n, p ) );
        return m;
    }

    /// LE VOLUME, A PART de la construction. C'est un TEMOIN et pas un calcul de production : le
    /// chronometrer avec `build` biaiserait la comparaison avec le k-DOP, ou il coute une decoupe
    /// de cellule alors qu'il est presque gratuit ici (les faces sont deja la).
    double volume() const {
        if ( ! ok ) return 0;
        if constexpr ( D == 2 ) {
            double a = 0;
            for ( SI i = 0, j = SI( som.size() ) - 1; i < SI( som.size() ); j = i++ )
                a += double( som[ j ][ 0 ] ) * double( som[ i ][ 1 ] )
                   - double( som[ i ][ 0 ] ) * double( som[ j ][ 1 ] );
            return std::fabs( a ) * 0.5;
        } else {
            Vec<3> g{};
            for ( int d = 0; d < 3; ++d ) g[ d ] = 0;
            for ( const Vec<3> &s : som ) for ( int d = 0; d < 3; ++d ) g[ d ] += s[ d ];
            for ( int d = 0; d < 3; ++d ) g[ d ] /= TF( som.size() );
            double v = 0;
            for ( const auto &f : fac ) {
                const Vec<3> n = croix( moins( som[ f[ 1 ] ], som[ f[ 0 ] ] ),
                                        moins( som[ f[ 2 ] ], som[ f[ 0 ] ] ) );
                v += double( dot( n, moins( som[ f[ 0 ] ], g ) ) );
            }
            return std::fabs( v ) / 6;
        }
    }

    /// LE CONTROLE, `O( faces x points )` : de combien l'enveloppe RATE-T-ELLE le point le plus
    /// fautif ? On rend une DISTANCE et non un booleen -- un `false` ne dirait pas si c'est une
    /// enveloppe fausse ou le dernier bit d'un produit vectoriel, et les deux ne se corrigent pas
    /// de la meme facon. Quadratique, donc sous `--verif` seulement.
    TF ecart( const std::vector<Vec<D>> &pts ) const {
        TF pire = 0;
        if constexpr ( D == 3 ) {
            for ( const auto &f : fac ) {
                const Vec<3> n = croix( moins( som[ f[ 1 ] ], som[ f[ 0 ] ] ),
                                        moins( som[ f[ 2 ] ], som[ f[ 0 ] ] ) );
                const TF ln = std::sqrt( dot( n, n ) );
                if ( ln <= ech * ech * TF( 1e-10 ) ) continue;   // face degeneree
                for ( const Vec<3> &p : pts )
                    pire = std::max( pire, dot( n, moins( p, som[ f[ 0 ] ] ) ) / ln );
            }
        } else {
            const SI k = SI( som.size() );
            if ( k < 3 ) return 0;
            for ( SI i = 0; i < k; ++i ) {
                const Vec<2> a = som[ i ], b = som[ ( i + 1 ) % k ];
                const TF ln = std::sqrt( ( b[ 0 ] - a[ 0 ] ) * ( b[ 0 ] - a[ 0 ] )
                                       + ( b[ 1 ] - a[ 1 ] ) * ( b[ 1 ] - a[ 1 ] ) );
                if ( ln <= ech * TF( 1e-10 ) ) continue;          // arete degeneree
                for ( const Vec<2> &p : pts )
                    pire = std::max( pire, -( ( b[ 0 ] - a[ 0 ] ) * ( p[ 1 ] - a[ 1 ] )
                                            - ( b[ 1 ] - a[ 1 ] ) * ( p[ 0 ] - a[ 0 ] ) ) / ln );
            }
        }
        return pire;
    }
};

// ------------------------------------------------------------------ LE k-DOP

/// LES DIRECTIONS, fixes et unitaires. En 2D on prend `K` angles regulierement repartis sur un
/// DEMI-tour : `K = 2` rend exactement les axes, `K = 4` y ajoute les deux diagonales. En 3D il n'y
/// a pas de repartition reguliere pour un `K` quelconque, alors on prend les familles classiques :
/// 3 axes (le 6-DOP), + 4 diagonales de cube (le 14-DOP), + 6 diagonales de face (le 26-DOP).
template<int D, int K>
const std::array<Vec<D>,K> &directions() {
    static const std::array<Vec<D>,K> d = [] {
        std::array<Vec<D>,K> r{};
        if constexpr ( D == 2 ) {
            for ( int k = 0; k < K; ++k ) {
                const double a = M_PI * k / K;
                r[ k ][ 0 ] = TF( std::cos( a ) );
                r[ k ][ 1 ] = TF( std::sin( a ) );
            }
        } else {
            static const double t[ 13 ][ 3 ] = {
                { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 },
                { 1, 1, 1 }, { 1, 1, -1 }, { 1, -1, 1 }, { -1, 1, 1 },
                { 1, 1, 0 }, { 1, -1, 0 }, { 1, 0, 1 }, { 1, 0, -1 }, { 0, 1, 1 }, { 0, 1, -1 } };
            static_assert( K <= 13, "on n'a tabule que jusqu'au 26-DOP" );
            for ( int k = 0; k < K; ++k ) {
                const double n = std::sqrt( t[k][0]*t[k][0] + t[k][1]*t[k][1] + t[k][2]*t[k][2] );
                for ( int j = 0; j < 3; ++j ) r[ k ][ j ] = TF( t[ k ][ j ] / n );
            }
        }
        return r;
    }();
    return d;
}

/// LE REPERE PROPRE d'un nuage de points : les vecteurs propres de sa covariance.
///
/// C'est ce qui manque au k-DOP a directions fixes. Sur un nuage uniforme il ne perd que 4 a 8 %
/// contre l'enveloppe convexe ; sur `lignes / aires egales` il en perd 147 %, parce que les
/// agregats y sont allonges dans une direction que la table ne contient pas -- et ce sont
/// justement les cas durs. Le repere propre coute UN passage de plus sur les points (la
/// covariance) et une decomposition `O( 1 )` ; la suite est le meme balayage.
///
/// 2D : forme fermee, l'angle principal vaut `atan2( 2 b, a - c ) / 2`.
/// 3D : Jacobi cyclique, en annulant a chaque tour la plus grande extra-diagonale. Douze tours
/// suffisent tres largement pour un 3x3.
///
/// Rend `false` si la decomposition n'a pas converge -- l'appelant garde alors le repere
/// canonique. Une enceinte moins fine, jamais une enceinte fausse : le k-DOP contient les points
/// QUEL QUE SOIT le repere, puisque `lo` et `hi` sont des min et des max sur ces points-la.
template<int D>
bool repere_propre( const std::vector<Vec<D>> &pts, Vec<D> e[ D ] ) {
    const TF n = TF( pts.size() );
    Vec<D> g{};
    for ( int d = 0; d < D; ++d ) g[ d ] = 0;
    for ( const Vec<D> &p : pts ) for ( int d = 0; d < D; ++d ) g[ d ] += p[ d ];
    for ( int d = 0; d < D; ++d ) g[ d ] /= n;
    TF c[ D ][ D ] = {};
    for ( const Vec<D> &p : pts )
        for ( int i = 0; i < D; ++i )
            for ( int j = i; j < D; ++j ) c[ i ][ j ] += ( p[ i ] - g[ i ] ) * ( p[ j ] - g[ j ] );
    for ( int i = 0; i < D; ++i ) for ( int j = 0; j < i; ++j ) c[ i ][ j ] = c[ j ][ i ];

    if constexpr ( D == 2 ) {
        const TF t = TF( 0.5 ) * std::atan2( 2 * c[ 0 ][ 1 ], c[ 0 ][ 0 ] - c[ 1 ][ 1 ] );
        const TF co = std::cos( t ), si = std::sin( t );
        e[ 0 ][ 0 ] =  co; e[ 0 ][ 1 ] = si;
        e[ 1 ][ 0 ] = -si; e[ 1 ][ 1 ] = co;
        return true;
    } else {
        TF v[ 3 ][ 3 ] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        TF ech = 0;
        for ( int i = 0; i < 3; ++i ) ech = std::max( ech, std::fabs( c[ i ][ i ] ) );
        const TF seuil = ech * TF( 1e-14 );
        bool fini = false;
        for ( int tour = 0; tour < 12 && ! fini; ++tour ) {
            int p0 = 0, q0 = 1;
            TF m = std::fabs( c[ 0 ][ 1 ] );
            if ( std::fabs( c[ 0 ][ 2 ] ) > m ) { m = std::fabs( c[ 0 ][ 2 ] ); p0 = 0; q0 = 2; }
            if ( std::fabs( c[ 1 ][ 2 ] ) > m ) { m = std::fabs( c[ 1 ][ 2 ] ); p0 = 1; q0 = 2; }
            if ( m <= seuil ) { fini = true; break; }
            const TF th = ( c[ q0 ][ q0 ] - c[ p0 ][ p0 ] ) / ( 2 * c[ p0 ][ q0 ] );
            const TF t  = ( th >= 0 ? TF( 1 ) : TF( -1 ) )
                        / ( std::fabs( th ) + std::sqrt( th * th + 1 ) );
            const TF cs = 1 / std::sqrt( t * t + 1 ), sn = t * cs;
            for ( int k = 0; k < 3; ++k ) {                       // colonnes
                const TF a = c[ k ][ p0 ], b = c[ k ][ q0 ];
                c[ k ][ p0 ] = cs * a - sn * b;  c[ k ][ q0 ] = sn * a + cs * b;
            }
            for ( int k = 0; k < 3; ++k ) {                       // puis lignes
                const TF a = c[ p0 ][ k ], b = c[ q0 ][ k ];
                c[ p0 ][ k ] = cs * a - sn * b;  c[ q0 ][ k ] = sn * a + cs * b;
            }
            for ( int k = 0; k < 3; ++k ) {                       // ... et l'accumulation
                const TF a = v[ k ][ p0 ], b = v[ k ][ q0 ];
                v[ k ][ p0 ] = cs * a - sn * b;  v[ k ][ q0 ] = sn * a + cs * b;
            }
        }
        for ( int d = 0; d < 3; ++d ) for ( int j = 0; j < 3; ++j ) e[ d ][ j ] = v[ j ][ d ];
        return fini;
    }
}

/// L'ENCEINTE PAR `K` BANDES -- l'intersection des `[ lo_k, hi_k ]` sur `K` directions.
///
/// `Oriente = false` : les directions de `directions<D,K>()`, les memes pour tout le monde.
/// `Oriente = true`  : ces memes directions EXPRIMEES DANS LE REPERE PROPRE de l'agregat. Comme
///                     la table est orthonormee dans la base canonique et le repere orthonorme,
///                     les directions restent unitaires -- `ecart` rend donc toujours une
///                     distance, et le k-DOP oriente n'est qu'un k-DOP fixe dans une autre base.
///
/// Contre l'enveloppe convexe il perd sur la finesse et gagne sur le cout, et les deux se lisent
/// dans les memes colonnes : `S|enc|` dit de combien il est plus lache, le chronometre de combien
/// il est moins cher. Sa construction est UN BALAYAGE (deux si oriente), sans tri, sans
/// allocation, sans cas particulier -- donc jamais de refus, la ou l'enveloppe incrementale 3D en
/// refuse 0.5 a 2 %.
template<int D, int K, bool Oriente>
struct KDopT {
    /// LES AXES CANONIQUES EN PLUS, quand il est oriente -- et ce n'est pas du luxe. Sans eux la
    /// boite ALIGNEE d'un k-DOP oriente est celle de sa boite ORIENTEE, donc gonflee : mesure en
    /// 3D, la grille de l'etape 3 presentait 855 candidats par agregat au lieu de 284, et la table
    /// finissait PLUS LONGUE qu'avec des directions fixes. L'enceinte la plus fine faisait le plus
    /// mauvais index. Avec les axes, la boite redevient exacte et les directions propres ne
    /// servent plus qu'a ce qu'elles savent faire : separer.
    /// faux : hors de ses axes propres le k-DOP MAJORE, et il en a trop pour que les croisements
    /// soient bon marche. `separe` s'en tient a ses axes.
    static constexpr bool croise = false;
    static constexpr int nb = Oriente ? K + D : K;
    Vec<D> dir[ nb ];
    TF     lo[ nb ], hi[ nb ];
    bool   ok = false;

    void build( std::vector<Vec<D>> &pts ) {
        ok = false;
        if ( pts.empty() ) return;
        const auto &tab = directions<D,K>();
        if constexpr ( Oriente ) {
            for ( int d = 0; d < D; ++d )
                for ( int j = 0; j < D; ++j ) dir[ d ][ j ] = ( d == j );   // les axes, d'abord
            Vec<D> e[ D ];
            if ( ! repere_propre<D>( pts, e ) )                 // pas converge : repere canonique
                for ( int d = 0; d < D; ++d )
                    for ( int j = 0; j < D; ++j ) e[ d ][ j ] = ( d == j );
            for ( int k = 0; k < K; ++k )
                for ( int j = 0; j < D; ++j ) {
                    TF v = 0;
                    for ( int d = 0; d < D; ++d ) v += tab[ k ][ d ] * e[ d ][ j ];
                    dir[ D + k ][ j ] = v;
                }
        } else {
            for ( int k = 0; k < K; ++k ) dir[ k ] = tab[ k ];
        }
        for ( int k = 0; k < nb; ++k ) lo[ k ] = hi[ k ] = dot( dir[ k ], pts[ 0 ] );
        for ( const Vec<D> &p : pts )
            for ( int k = 0; k < nb; ++k ) {
                const TF v = dot( dir[ k ], p );
                lo[ k ] = std::min( lo[ k ], v );
                hi[ k ] = std::max( hi[ k ], v );
            }
        ok = true;
    }

    SI taille() const { return nb; }

    // ---- L'INTERROGATION. Sur SES directions le k-DOP donne l'intervalle exact ; ailleurs il
    // majore par sa boite ORIENTEE. Celle-ci est gratuite : les `D` premieres directions de la
    // table sont orthonormees ( les axes, ou les vecteurs propres si oriente ) -- en 2D il faut
    // les indices `0` et `K/2`, l'angle `k pi / K` ne rendant l'orthogonal qu'a mi-tour.
    static constexpr SI iax( int d ) { return Oriente ? d : ( D == 2 ? ( d == 0 ? 0 : K / 2 ) : d ); }
    static_assert( D == 3 || K % 2 == 0, "en 2D il faut K pair pour que la table porte un repere" );

    SI     nb_axes() const { return nb; }
    Vec<D> axe( SI k ) const { return dir[ k ]; }
    void   bande( SI k, TF &l, TF &h ) const { l = lo[ k ]; h = hi[ k ]; }

    TF support( Vec<D> n ) const {
        TF s = 0;
        for ( int d = 0; d < D; ++d ) {
            const SI k = iax( d );
            s += ( lo[ k ] + hi[ k ] ) / 2 * dot( dir[ k ], n )
               + ( hi[ k ] - lo[ k ] ) / 2 * std::fabs( dot( dir[ k ], n ) );
        }
        return s;
    }
    void boite( Vec<D> &l, Vec<D> &h ) const {
        for ( int j = 0; j < D; ++j ) {
            TF c = 0, r = 0;
            for ( int d = 0; d < D; ++d ) {
                const SI k = iax( d );
                c += ( lo[ k ] + hi[ k ] ) / 2 * dir[ k ][ j ];
                r += ( hi[ k ] - lo[ k ] ) / 2 * std::fabs( dir[ k ][ j ] );
            }
            l[ j ] = c - r; h[ j ] = c + r;
        }
    }

    /// LE VOLUME, par decoupe d'une cellule avec les `2 K` plans. On ne l'ecrit pas a la main
    /// parce qu'il n'a pas de forme fermee des que `K > D` : le polytope depend de quelles bandes
    /// mordent. `Cell` le fait deja et le fait juste -- et c'est un TEMOIN, donc son cout est
    /// chronometre a part.
    ///
    /// TROIS PRECAUTIONS, apprises en le faisant faux. Les identifiants de coupe doivent etre
    /// POSITIFS ET DISTINCTS : `Cell3T` reserve les negatifs aux six faces du domaine et exige
    /// qu'elles se distinguent, sans quoi deux faces opposees deviennent la meme et le parcours
    /// des cycles part en vrille (on lisait 0.02 au lieu de 2.9). La cellule doit pouvoir loger
    /// `2 K` plans. Et il faut s'arreter des qu'elle est vide : `cut` lit `s[ nb - 1 ]` avant tout
    /// test, donc l'appeler sur une cellule videe lit hors du tampon.
    double volume() const {
        if ( ! ok ) return 0;
        // UNE GARDE, et elle dit quelque chose. Un agregat dont tous les sommets tiennent dans
        // 1e-10 (un dirac isole, deux sommets confondus) donne des bandes de cette largeur :
        // decouper le carre unite avec `2 K` plans quasi confondus a une abscisse d'ordre 1 fait
        // perdre TOUS les chiffres a `t = s_in / ( s_in - s_out )`, et la mesure sort `nan`. A la
        // precision ou l'on travaille ce volume est nul ; on le rend, plutot que de propager un
        // `nan` dans le temoin. ( L'enveloppe traite le meme cas autrement : son dedoublonnage la
        // ramene a moins de trois points et elle se declare degeneree. )
        TF large = 0;
        for ( int k = 0; k < nb; ++k ) large = std::max( large, hi[ k ] - lo[ k ] );
        if ( large < TF( 1e-10 ) ) return 0;
        // ... ET UN DESSERRAGE, qui est la vraie difficulte. Avec `K` directions et peu de points,
        // plusieurs bandes sont portees par LE MEME sommet : les plans y sont TANGENTS. Une coupe
        // tangente rogne un sommet de 1e-17 et insere a sa place deux points confondus ; au bout
        // de quelques-unes le polygone n'est plus convexe, l'invariant de `Cell` (« les sommets
        // dehors forment une plage cyclique UNIQUE ») tombe, et `cut` interpole entre deux sommets
        // non adjacents -- d'ou le `nan`. On ecarte donc chaque plan de `1e-9` fois la taille du
        // k-DOP : les plans tangents rendent alors `unchanged` et ne creent plus rien. Le volume
        // est majore de ~1e-09 en relatif, ce qui est sans effet sur un temoin.
        const TF ec = large * TF( 1e-9 );
        using Cell = bench::CellFor<D, D == 2 ? 128 : 192>;
        static_assert( Cell::max_nb_vertices >= 4 * nb + 8, "la cellule doit loger les 2 K plans" );
        Cell c;
        if constexpr ( D == 2 ) c.init_as_unit_square();
        else                    c.init_as_unit_cube();
        for ( int k = 0; k < nb; ++k ) {
            Vec<D> n = dir[ k ];
            if ( c.cut( n, hi[ k ] + ec, 2 * k ) == CutResult::overflow || c.nb == 0 ) return 0;
            for ( int d = 0; d < D; ++d ) n[ d ] = -n[ d ];
            if ( c.cut( n, ec - lo[ k ], 2 * k + 1 ) == CutResult::overflow || c.nb == 0 ) return 0;
        }
        return double( c.measure() );
    }

    TF ecart( const std::vector<Vec<D>> &pts ) const {
        TF pire = 0;
        for ( const Vec<D> &p : pts )
            for ( int k = 0; k < nb; ++k ) {
                const TF v = dot( dir[ k ], p );
                pire = std::max( pire, std::max( v - hi[ k ], lo[ k ] - v ) );
            }
        return pire;
    }
};

// ------------------------------------------------------------------ LA BOITE ORIENTEE

/// LA BOITE ORIENTEE, choisie parmi `M` reperes candidats -- et ce qu'elle a que le k-DOP n'a pas.
///
/// Un k-DOP est une intersection de bandes : pour en tirer des SOMMETS il faut construire le
/// polytope, ce qui coute une decoupe de cellule et se heurte aux plans tangents. Une boite
/// orientee a ses `2^D` sommets par construction, et il en decoule trois choses qui comptent :
///
///   * son SUPPORT est EXACT dans toute direction, pas seulement les siennes. Le k-DOP doit
///     majorer hors de ses axes propres, ce qui lui fait garder des paires de trop dans la table.
///   * son VOLUME est le produit des aretes. Pas de decoupe, pas de tangence, pas de `nan`, pas
///     de desserrage a 1e-9 -- toute la fragilite de `KDopT::volume` disparait.
///   * elle donne des POINTS a l'etape 4, qui pourra intersecter proprement au lieu d'interroger
///     des fonctions de support.
///
/// LE CHOIX DU REPERE. On part du repere propre ( la covariance, comme le k-DOP oriente ) et on
/// essaie `M` repères : le repere propre lui-meme, puis des rotations de celui-ci. Les `M x D`
/// projections se font en UN SEUL passage sur les points -- le cout est celui d'un k-DOP a `M x D`
/// directions -- et on garde le repere de plus petit volume. Ce n'est pas la boite orientee
/// MINIMALE ( qui coute `O( m^3 )` en 3D ) ; c'est un balayage, et on mesure ce qu'il rapporte.
///
///   2D : `M` angles reguliers a partir de l'angle principal. La boite est de periode `pi / 2`,
///        donc `M` angles couvrent tout.
///   3D : le repere propre, puis des rotations de `pi/2 . j/(J+1)` autour de chacun de ses trois
///        axes. `M = 4` donne le repere propre plus trois rotations a 45 degres ; `M = 7`, six
///        rotations a 30 et 60.
template<int D, int M>
struct Obb {
    Vec<D> e[ D ];                    ///< le repere retenu, orthonorme
    TF     lo[ D ], hi[ D ];
    bool   ok = false;

    /// `support` ne majore pas, il rend le maximum -- et l'enceinte n'a que `D` axes. `separe`
    /// peut donc ajouter les `D x D` axes arete-contre-arete, ce qui rend le test EXACT en 3D.
    static constexpr bool croise = true;

    static void repere( const Vec<D> base[ D ], int m, Vec<D> out[ D ] ) {
        if constexpr ( D == 2 ) {
            const double a = M_PI * m / ( 2.0 * M );
            const TF c = TF( std::cos( a ) ), s = TF( std::sin( a ) );
            for ( int j = 0; j < 2; ++j ) {
                out[ 0 ][ j ] =  c * base[ 0 ][ j ] + s * base[ 1 ][ j ];
                out[ 1 ][ j ] = -s * base[ 0 ][ j ] + c * base[ 1 ][ j ];
            }
        } else {
            for ( int d = 0; d < 3; ++d ) out[ d ] = base[ d ];
            if ( m == 0 ) return;
            const int niv = ( M - 1 + 2 ) / 3;               // niveaux d'angle
            const int ax = ( m - 1 ) % 3, j = ( m - 1 ) / 3 + 1;
            const double a = M_PI / 2 * j / ( niv + 1.0 );
            const TF c = TF( std::cos( a ) ), s = TF( std::sin( a ) );
            const int b = ( ax + 1 ) % 3, g = ( ax + 2 ) % 3;
            for ( int k = 0; k < 3; ++k ) {
                out[ b ][ k ] =  c * base[ b ][ k ] + s * base[ g ][ k ];
                out[ g ][ k ] = -s * base[ b ][ k ] + c * base[ g ][ k ];
            }
        }
    }

    void build( std::vector<Vec<D>> &pts ) {
        ok = false;
        if ( pts.empty() ) return;
        Vec<D> base[ D ];
        if ( ! repere_propre<D>( pts, base ) )
            for ( int d = 0; d < D; ++d )
                for ( int j = 0; j < D; ++j ) base[ d ][ j ] = ( d == j );

        Vec<D> cand[ M ][ D ];
        for ( int m = 0; m < M; ++m ) repere( base, m, cand[ m ] );

        // UN SEUL passage sur les points, pour les `M x D` directions a la fois.
        TF l[ M ][ D ], h[ M ][ D ];
        for ( int m = 0; m < M; ++m )
            for ( int d = 0; d < D; ++d ) l[ m ][ d ] = h[ m ][ d ] = dot( cand[ m ][ d ], pts[ 0 ] );
        for ( const Vec<D> &p : pts )
            for ( int m = 0; m < M; ++m )
                for ( int d = 0; d < D; ++d ) {
                    const TF v = dot( cand[ m ][ d ], p );
                    l[ m ][ d ] = std::min( l[ m ][ d ], v );
                    h[ m ][ d ] = std::max( h[ m ][ d ], v );
                }

        int best = 0;
        double vbest = 1e300;
        for ( int m = 0; m < M; ++m ) {
            double v = 1;
            for ( int d = 0; d < D; ++d ) v *= double( h[ m ][ d ] - l[ m ][ d ] );
            if ( v < vbest ) { vbest = v; best = m; }
        }
        for ( int d = 0; d < D; ++d ) {
            e[ d ] = cand[ best ][ d ];
            lo[ d ] = l[ best ][ d ];
            hi[ d ] = h[ best ][ d ];
        }
        ok = true;
    }

    /// LES SOMMETS, a la demande. On ne les stocke pas -- ils se regenerent en `2^D` additions,
    /// et les garder doublerait la place d'une enceinte qui tient en `D` axes et `2 D` bornes.
    void sommets( Vec<D> out[ 1 << D ] ) const {
        for ( int k = 0; k < ( 1 << D ); ++k )
            for ( int j = 0; j < D; ++j ) {
                TF v = 0;
                for ( int d = 0; d < D; ++d ) v += ( k >> d & 1 ? hi[ d ] : lo[ d ] ) * e[ d ][ j ];
                out[ k ][ j ] = v;
            }
    }

    SI     taille() const { return 1 << D; }
    SI     nb_axes() const { return D; }
    Vec<D> axe( SI k ) const { return e[ k ]; }
    void   bande( SI k, TF &l, TF &h ) const { l = lo[ k ]; h = hi[ k ]; }

    /// EXACT, et sans passer par les sommets : `max ( sum t_d e_d ) . n` se separe par coordonnee.
    TF support( Vec<D> n ) const {
        TF s = 0;
        for ( int d = 0; d < D; ++d ) {
            const TF u = dot( e[ d ], n );
            s += ( lo[ d ] + hi[ d ] ) / 2 * u + ( hi[ d ] - lo[ d ] ) / 2 * std::fabs( u );
        }
        return s;
    }
    void boite( Vec<D> &l, Vec<D> &h ) const {
        for ( int j = 0; j < D; ++j ) {
            TF c = 0, r = 0;
            for ( int d = 0; d < D; ++d ) {
                c += ( lo[ d ] + hi[ d ] ) / 2 * e[ d ][ j ];
                r += ( hi[ d ] - lo[ d ] ) / 2 * std::fabs( e[ d ][ j ] );
            }
            l[ j ] = c - r; h[ j ] = c + r;
        }
    }
    /// Le produit des aretes. Rien a decouper, rien a desserrer.
    double volume() const {
        if ( ! ok ) return 0;
        double v = 1;
        for ( int d = 0; d < D; ++d ) v *= double( hi[ d ] - lo[ d ] );
        return v;
    }
    TF ecart( const std::vector<Vec<D>> &pts ) const {
        TF pire = 0;
        for ( const Vec<D> &p : pts )
            for ( int d = 0; d < D; ++d ) {
                const TF v = dot( e[ d ], p );
                pire = std::max( pire, std::max( v - hi[ d ], lo[ d ] - v ) );
            }
        return pire;
    }
};

template<int D, int K> using KDop        = KDopT<D, K, false>;
template<int D, int K> using KDopOriente = KDopT<D, K, true>;

} // namespace pd::supercell
