#pragma once

// =====================================================================================
// LE DIAGRAMME DE PUISSANCE : ce que le solveur voit, en 2D comme en 3D.
//
// Un `PowerDiagram<D,TK,MaxNv>` possede l'arbre BSP et une copie des germes DANS L'ORDRE DE
// L'ARBRE, dans le flottant du noyau `TK`. Il offre trois choses :
//
//   `build( P, W, n, leaf )`       les germes, dans l'ordre de l'appelant ; `W == nullptr` pour
//                                  Voronoi ( le cas euclidien ne paie alors pas les majorants ) ;
//   `set_weights( W, par )`        des poids neufs, meme arbre : les majorants et la copie
//                                  refaits, rien d'autre ;
//   `measures( res, par )`         la mesure de chaque cellule, INDEXEE PAR L'IDENTIFIANT de
//   `measures_and_facets( ... )`   l'appelant, et pour Newton les facettes `( i, j, mesure )`.
//
// Le choix 2D / 3D est fait ici et nulle part ailleurs : `cellule( k, c )` deroule le moteur qui
// convient, `mesure( c, facette )` lit ce qui en sort. Le reste du fichier est commun.
//
// Les mesures sont accumulees en `TF` ( double ) depuis les sommets en `TK` : avec `TK = float`
// le plancher relatif d'une aire est ~1e-7, ce qui est le plancher du residu de Newton. `double`
// leve cette limite au prix mesure par `diagramme --kernel float|double`.
//
// `c` ET `w` SONT UNE COPIE EN `TK`, ET LES FOURNISSEURS NE S'EN SERVENT PLUS pour construire
// les plans : ils lisent l'arbre, qui range germes et poids en `TF`, et n'arrondissent qu'une
// fois le plan calcule ( `cell/Plan.h` -- c'est la reparation de la simple precision ). La copie
// reste pour les temoins et pour qui veut les germes dans l'ordre de l'arbre.
// =====================================================================================

#include "accel/AaBsp.h"
#include "cell/FournisseurBsp2D.h"
#include "cell/FournisseurBsp3D.h"
#include "cell/Noyau2D.h"
#include "cell/Noyau3D.h"
#include "util/parallel.h"

#include <atomic>
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <vector>

namespace sf {

template<int D, class TK, int MaxNv>
struct PowerDiagram {
    static constexpr int dim = D;
    static constexpr int max_nv = MaxNv;
    using TKernel = TK;
    using Arbre   = AaBspT<D>;
    /// la cellule telle que le moteur la rend : l'atelier en 2D, le polyedre en 3D
    using Cell    = std::conditional_t<D == 2, d2::Atelier<TK,MaxNv>, d3::Cellule3<TK,MaxNv,8>>;

    Arbre            arbre;
    std::vector<TK>  c[ D ];        ///< les positions, ordre de l'arbre, flottant du noyau
    std::vector<TK>  w;             ///< les poids, idem ( vide = Voronoi )
    std::vector<d2::SI32> ids;      ///< rang dans l'arbre -> identifiant de l'appelant
    SI               n = 0;

    /// LA MEMOIRE ( 3D, § 11 ) : par rang, les rangs des voisins du DERNIER diagramme, proposes en
    /// premier par `measures_and_facets*` tant qu'elle est la ( `oublie()` pour l'eteindre ).
    std::vector<SI>       rang;     ///< identifiant -> rang, bati au premier `memorise`
    std::vector<SI>       memo_beg;
    std::vector<d2::SI32> memo_pre;
    mutable std::vector<std::vector<unsigned char>> memo_saute;   ///< par fil : « deja propose »
    mutable SI            memo_coupees = 0;                         ///< coupes effectives, en tout, au dernier diagramme

    /// LE RAFFINEMENT DES SOMMETS ( `raffine` ), actif par defaut EN SIMPLE PRECISION SEULEMENT --
    /// en `double` l'erreur portee vaut `1e-16 L`, il n'y a rien a reparer. `SF_RAFF=0` l'eteint.
    bool raffiner = []{ const char *e = std::getenv( "SF_RAFF" ); return ! e || std::atoi( e ); }();
    /// sous ce determinant RELATIF, on garde le sommet du noyau. `SF_DET`.
    static TF seuil_det() {
        static const TF v = []{ const char *e = std::getenv( "SF_DET" ); return e ? TF( std::atof( e ) ) : TF( 1e-6 ); }();
        return v;
    }

    /// les paires ( identifiant, identifiant ) des facettes d'un diagramme deviennent la memoire
    void memorise( const d2::SI32 *ii, const d2::SI32 *jj, SI nb ) {
        if ( SI( rang.size() ) != n ) { rang.resize( n ); for ( SI k = 0; k < n; ++k ) rang[ ids[ k ] ] = k; }
        memo_beg.assign( n + 1, 0 );
        for ( SI q = 0; q < nb; ++q ) ++memo_beg[ rang[ ii[ q ] ] + 1 ];
        for ( SI k = 0; k < n; ++k ) memo_beg[ k + 1 ] += memo_beg[ k ];
        memo_pre.resize( nb );
        std::vector<SI> at( memo_beg.begin(), memo_beg.end() - 1 );
        for ( SI q = 0; q < nb; ++q ) memo_pre[ at[ rang[ ii[ q ] ] ]++ ] = d2::SI32( rang[ jj[ q ] ] );
    }
    void oublie() { memo_beg.clear(); memo_pre.clear(); }
    bool memoire() const { return ! memo_beg.empty(); }

    // ------------------------------------------------------------------ construction, poids
    void build( const TF *const *P, const TF *W, SI nb, SI leaf ) {
        n = nb;
        arbre.build( P, W, n, leaf );
        for ( int d = 0; d < D; ++d ) c[ d ].resize( n );
        ids.resize( n );
        for ( SI k = 0; k < n; ++k ) {
            for ( int d = 0; d < D; ++d ) c[ d ][ k ] = TK( arbre.seed_c( k, d ) );
            ids[ k ] = d2::SI32( arbre.seed_id( k ) );
        }
        rang.resize( n );                                // identifiant -> rang : `raffine` en a besoin
        for ( SI k = 0; k < n; ++k ) rang[ ids[ k ] ] = k;
        w.clear();
        if ( W )
            copie_poids();
    }

    /// `W` dans l'ordre de l'appelant. L'arbre passe en Laguerre s'il ne l'etait pas.
    void set_weights( const TF *W, const Parallel &par ) {
        arbre.refresh_weights( W, par );
        copie_poids();
    }

    bool laguerre() const { return ! w.empty(); }

    // ------------------------------------------------------------------ une cellule
    /// La cellule du germe de rang `k` dans l'arbre. Rend `false` si elle a DEBORDE `MaxNv`.
    bool cellule( SI k, Cell &cel ) const {
        return laguerre() ? cellule_<true>( k, cel ) : cellule_<false>( k, cel );
    }

    /// La meme cellule avec un AUTRE poids pour le germe `k` seul ( les autres inchanges, l'arbre
    /// aussi : les majorants ne parlent que des autres ). Laguerre seulement.
    bool cellule_avec_poids( SI k, TF wk, Cell &cel ) const {
        if constexpr ( D == 2 ) {
            d2::FournisseurBsp<TK,true> f( &arbre, arbre.seed_c( k, 0 ), arbre.seed_c( k, 1 ), wk, ids[ k ], k );
            d2::moteur<TK>( &f, &cel );
            return cel.nb >= 0;
        } else {
            d3::FournisseurBsp3<TK,true> f( &arbre, arbre.seed_c( k, 0 ), arbre.seed_c( k, 1 ),
                                            arbre.seed_c( k, 2 ), wk, ids[ k ], k );
            return d3::moteur( &f, &cel ) == 0;
        }
    }

    /// LA CELLULE AVEC MEMOIRE ( 3D, `main_memo.cpp` ) : les rangs `pre[ 0 .. npre )` proposes en
    /// premier, ceux marques dans `saute` ( par rang ) sautes au parcours. `prop` et `boites` rendent
    /// ce que le fournisseur a propose et teste. Rend `false` sur debordement.
    struct Memo {                                        ///< ce qu'on donne a `cellule_memo` ( voir `FournisseurBsp3` )
        const d2::SI32 *pre = nullptr; int npre = 0; const unsigned char *saute = nullptr;                 // A
        const d2::SI32 *fbeg = nullptr; const unsigned long long *fmask = nullptr; int nf = 0; int tester = 1;   // B
        const d2::SI32 *front = nullptr; int nfront = 0;                                                    // C
        bool parcours = true;
        int prop = 0, boites = 0, coupees = 0;           ///< rendus
        int entrees[ 64 ]; int nentrees = 0;             ///< rendu : les feuilles entrees ( indices de noeuds )
        int rejets[ 256 ]; int nrejets = 0;              ///< rendu : les noeuds rejetes ( indices )
    };
    bool cellule_memo( SI k, Cell &cel, Memo &m ) const {
        static_assert( D == 3, "la memoire n'est ecrite qu'en 3D" );
        return laguerre() ? cellule_memo_<true>( k, cel, m ) : cellule_memo_<false>( k, cel, m );
    }

    /// La mesure de la cellule, et ses facettes contre d'autres germes : `facette( j, mes )`
    /// avec `j` l'identifiant du voisin -- les parois du domaine ne sont pas rendues.
    template<class Facette>
    static TF mesure( const Cell &cel, Facette &&facette ) {
        if constexpr ( D == 2 ) {
            if ( cel.nb <= 0 ) return 0;
            // EN COORDONNEES LOCALES, et ce n'est pas seulement licite : l'aire et les longueurs
            // sont invariantes par translation, mais en absolu le lacet somme des termes d'ordre
            // `1` pour rendre `h^2`. Dans le repere du germe les termes SONT d'ordre `h^2`.
            TF a = 0;
            for ( int i = 0, j = cel.nb - 1; i < cel.nb; j = i++ ) {
                a += TF( cel.lx[ j ] ) * TF( cel.ly[ i ] ) - TF( cel.lx[ i ] ) * TF( cel.ly[ j ] );
                if ( cel.cid[ j ] >= 0 ) {               // la coupe `j` porte l'arete [ v_j, v_i ]
                    const TF ex = TF( cel.lx[ i ] ) - TF( cel.lx[ j ] );
                    const TF ey = TF( cel.ly[ i ] ) - TF( cel.ly[ j ] );
                    facette( cel.cid[ j ], std::sqrt( ex * ex + ey * ey ) );
                }
            }
            return TF( 0.5 ) * std::fabs( a );
        } else {
            return cel.volume_et_faces( [ & ]( d3::SI32 id, double aire ) {
                if ( id >= 0 ) facette( id, TF( aire ) );
            } );
        }
    }
    static TF mesure( const Cell &cel ) { return mesure( cel, []( d2::SI32, TF ) {} ); }

    // ------------------------------------------------------------------ tout le diagramme
    /// `res[ id ]` pour chaque germe. Rend le nombre de cellules qui ont DEBORDE ( leur mesure est
    /// alors fausse -- relancer avec un `MaxNv` plus grand ).
    SI measures( std::vector<TF> &res, const Parallel &par ) const {
        return measures_and_facets( res, par, []( int, d2::SI32, d2::SI32, TF ) {} );
    }

    /// Le meme, avec `facette( t, i, j, mes )` pour chaque facette de la cellule `i` contre `j`,
    /// depuis le thread `t` -- chaque paire est vue DEUX fois, une par cellule.
    template<class Facette>
    SI measures_and_facets( std::vector<TF> &res, const Parallel &par, Facette &&facette ) const {
        return measures_and_facets_avec( res, par, facette, []( const Cell &cel, auto &&fac, SI ) { return mesure( cel, fac ); } );
    }

    /// Le meme, avec UNE AUTRE MESURE que Lebesgue : `mes( cel, facette( j, mes ), i )` rend la mesure
    /// de la cellule `i` et appelle `facette` pour chacune de ses facettes ( `Densite.h` ).
    template<class Facette, class Mesure>
    SI measures_and_facets_avec( std::vector<TF> &res, const Parallel &par, Facette &&facette, Mesure &&mes ) const {
        res.assign( n, TF( 0 ) );
        std::atomic<SI> deborde{ 0 };
        if constexpr ( D == 3 ) {
            if ( memoire() ) {
                const int nth = std::max( par.threads, 1 );
                if ( SI( memo_saute.size() ) < nth ) memo_saute.resize( nth );
                for ( auto &v : memo_saute ) if ( SI( v.size() ) != n ) v.assign( n, 0 );
                std::atomic<SI> coupees{ 0 };
                parallel_for( n, par, [ & ]( SI k, int t ) {
                    Cell cel;
                    const d2::SI32 i = ids[ k ];
                    const d2::SI32 *p = memo_pre.data() + memo_beg[ k ];
                    const int np = int( memo_beg[ k + 1 ] - memo_beg[ k ] );
                    unsigned char *sa = memo_saute[ t ].data();
                    for ( int q = 0; q < np; ++q ) sa[ p[ q ] ] = 1;
                    Memo m; m.pre = p; m.npre = np; m.saute = sa;
                    const bool ok = cellule_memo( k, cel, m );
                    for ( int q = 0; q < np; ++q ) sa[ p[ q ] ] = 0;
                    coupees += m.coupees;
                    if ( ! ok ) { ++deborde; return; }
                    res[ i ] = mes( cel, [ & ]( d2::SI32 j, TF m ) { facette( t, i, j, m ); }, SI( i ) );
                } );
                memo_coupees = coupees.load();
                return deborde.load();
            }
        }
        parallel_for( n, par, [ & ]( SI k, int t ) {
            Cell cel;
            const d2::SI32 i = ids[ k ];
            if ( ! cellule( k, cel ) ) { ++deborde; return; }
            if constexpr ( sizeof( TK ) < 8 ) if ( raffiner ) raffine( k, cel );
            res[ i ] = mes( cel, [ & ]( d2::SI32 j, TF m ) { facette( t, i, j, m ); }, SI( i ) );
        } );
        return deborde.load();
    }

    // ------------------------------------------------------------------ LES SOMMETS RAFFINES
    /// LE PLAN de la coupe `id`, vu par le germe de rang `k`, EN DOUBLE et dans le repere du germe.
    /// Rend `false` si l'identifiant ne se relit pas ( face artificielle d'une boite de depart ).
    bool plan_local( SI k, d2::SI32 id, TF nrm[ D ], TF &off ) const {
        if ( id >= 0 ) {                                 // un germe : le bissecteur, comme `Plan.h`
            const SI r = rang[ id ];
            TF dd = 0;
            for ( int d = 0; d < D; ++d ) {
                nrm[ d ] = arbre.seed_c( r, d ) - arbre.seed_c( k, d );
                dd += nrm[ d ] * nrm[ d ];
            }
            off = TF( 0.5 ) * dd;
            if ( laguerre() ) off += TF( 0.5 ) * ( arbre.seed_w( k ) - arbre.seed_w( r ) );
            return true;
        }
        const int f = -1 - id;
        if ( f >= 2 * D ) return false;                  // `FACE_ARTIF` : on ne sait pas la relire
        for ( int d = 0; d < D; ++d ) nrm[ d ] = 0;
        // LES FACES DU DOMAINE, dans l'ordre que posent `Noyau2D.h` et `Cellule3D.h` -- et ce
        // n'est PAS le meme : en 2D bas / droite / haut / gauche, en 3D x- / x+ / y- / y+ / z- / z+
        int ax; bool haut;
        if constexpr ( D == 2 ) { const int t[ 4 ] = { 1, 0, 1, 0 }; ax = t[ f ]; haut = f == 1 || f == 2; }
        else                    { ax = f / 2; haut = f & 1; }
        nrm[ ax ] = haut ? TF( 1 ) : TF( -1 );
        const TF o = arbre.seed_c( k, ax );
        off = haut ? TF( 1 ) - o : o;
        return true;
    }

    /// CHAQUE SOMMET RESOLU DEPUIS LES COUPES QUI LE PORTENT -- deux en 2D, trois en 3D.
    ///
    /// Un sommet d'un convexe ne depend PAS de l'histoire des coupes : il est l'intersection de
    /// `D` plans, et rien d'autre. Le noyau, lui, le construit par interpolations successives, si
    /// bien qu'il porte l'erreur de la cellule TELLE QU'ELLE ETAIT quand il est ne -- `eps L` pour
    /// une cellule large de `L`. C'est cette erreur PORTEE que les boites de depart combattaient,
    /// au prix de reprises. La resoudre a la source coute une elimination de Gauss par sommet et
    /// ne branche pas : sur une carte, pas de divergence.
    ///
    /// Le systeme mal conditionne ( plans presque paralleles ) est LAISSE tel quel : le sommet du
    /// noyau y est au moins aussi bon, et on ne veut pas remplacer une erreur portee par un
    /// quotient qui explose.
    void raffine( SI k, Cell &cel ) const {
        if constexpr ( D == 2 ) {
            const int nb = cel.nb;
            if ( nb < 3 ) return;
            TF pn[ MaxNv ][ 2 ], po[ MaxNv ];
            bool ok[ MaxNv ];
            for ( int j = 0; j < nb; ++j ) ok[ j ] = plan_local( k, cel.cid[ j ], pn[ j ], po[ j ] );
            TF nx[ MaxNv ], ny[ MaxNv ];
            for ( int i = 0; i < nb; ++i ) {
                const int j = i ? i - 1 : nb - 1;         // l'arete qui ARRIVE en `i`
                nx[ i ] = TF( cel.lx[ i ] ); ny[ i ] = TF( cel.ly[ i ] );
                if ( ! ok[ j ] || ! ok[ i ] ) continue;
                const TF det = pn[ j ][ 0 ] * pn[ i ][ 1 ] - pn[ j ][ 1 ] * pn[ i ][ 0 ];
                const TF e1 = pn[ j ][ 0 ] * pn[ j ][ 0 ] + pn[ j ][ 1 ] * pn[ j ][ 1 ];
                const TF e2 = pn[ i ][ 0 ] * pn[ i ][ 0 ] + pn[ i ][ 1 ] * pn[ i ][ 1 ];
                if ( ! ( det * det > seuil_det() * seuil_det() * e1 * e2 ) ) continue;
                nx[ i ] = ( po[ j ] * pn[ i ][ 1 ] - po[ i ] * pn[ j ][ 1 ] ) / det;
                ny[ i ] = ( pn[ j ][ 0 ] * po[ i ] - pn[ i ][ 0 ] * po[ j ] ) / det;
            }
            for ( int i = 0; i < nb; ++i ) { cel.lx[ i ] = TK( nx[ i ] ); cel.ly[ i ] = TK( ny[ i ] ); }
        } else {
            const int nv = cel.nv, nc = cel.nc;
            if ( nv < 4 ) return;
            std::vector<TF> pn( size_t( nc ) * 3 ), po( nc );
            std::vector<char> ok( nc );
            for ( int j = 0; j < nc; ++j ) ok[ j ] = plan_local( k, cel.cid[ j ], &pn[ size_t( j ) * 3 ], po[ j ] );
            for ( int i = 0; i < nv; ++i ) {
                const int q[ 3 ] = { cel.vk0[ i ], cel.vk1[ i ], cel.vk2[ i ] };
                if ( ! ok[ q[ 0 ] ] || ! ok[ q[ 1 ] ] || ! ok[ q[ 2 ] ] ) continue;
                const TF *A0 = &pn[ size_t( q[ 0 ] ) * 3 ], *A1 = &pn[ size_t( q[ 1 ] ) * 3 ], *A2 = &pn[ size_t( q[ 2 ] ) * 3 ];
                const TF c0 = A1[ 1 ] * A2[ 2 ] - A1[ 2 ] * A2[ 1 ];
                const TF c1 = A1[ 2 ] * A2[ 0 ] - A1[ 0 ] * A2[ 2 ];
                const TF c2 = A1[ 0 ] * A2[ 1 ] - A1[ 1 ] * A2[ 0 ];
                const TF det = A0[ 0 ] * c0 + A0[ 1 ] * c1 + A0[ 2 ] * c2;
                TF ech = 1;
                for ( const TF *A : { A0, A1, A2 } ) ech *= std::sqrt( A[0]*A[0] + A[1]*A[1] + A[2]*A[2] );
                if ( ! ( std::fabs( det ) > seuil_det() * ech ) ) continue;
                const TF b[ 3 ] = { po[ q[ 0 ] ], po[ q[ 1 ] ], po[ q[ 2 ] ] };
                // Cramer : on remplace une colonne a la fois
                auto d3 = [ ]( const TF *u, const TF *v, const TF *w ) {
                    return u[0]*(v[1]*w[2]-v[2]*w[1]) - u[1]*(v[0]*w[2]-v[2]*w[0]) + u[2]*(v[0]*w[1]-v[1]*w[0]);
                };
                const TF L0[ 3 ] = { b[0], A0[1], A0[2] }, L1[ 3 ] = { b[1], A1[1], A1[2] }, L2[ 3 ] = { b[2], A2[1], A2[2] };
                const TF M0[ 3 ] = { A0[0], b[0], A0[2] }, M1[ 3 ] = { A1[0], b[1], A1[2] }, M2[ 3 ] = { A2[0], b[2], A2[2] };
                const TF N0[ 3 ] = { A0[0], A0[1], b[0] }, N1[ 3 ] = { A1[0], A1[1], b[1] }, N2[ 3 ] = { A2[0], A2[1], b[2] };
                cel.vx[ i ] = TK( d3( L0, L1, L2 ) / det );
                cel.vy[ i ] = TK( d3( M0, M1, M2 ) / det );
                cel.vz[ i ] = TK( d3( N0, N1, N2 ) / det );
            }
        }
    }

private:
    void copie_poids() {
        w.resize( n );
        for ( SI k = 0; k < n; ++k ) w[ k ] = TK( arbre.seed_w( k ) );
    }

    template<bool POIDS>
    bool cellule_memo_( SI k, Cell &cel, Memo &m ) const {
        if constexpr ( D == 3 ) {
            using F = d3::FournisseurBsp3<TK,POIDS,8,true>;
            F f( &arbre, arbre.seed_c( k, 0 ), arbre.seed_c( k, 1 ), arbre.seed_c( k, 2 ),
                 POIDS ? arbre.seed_w( k ) : TF( 0 ), ids[ k ], k );
            f.pre = m.pre; f.npre = m.npre; f.saute = m.saute; f.parcours = m.parcours;
            f.fbeg = m.fbeg; f.fmask = m.fmask; f.nf = m.nf; f.tester = m.tester;
            f.front = m.front; f.nfront = m.nfront;
            typename F::Local loc;
            const int r = d3::moteur( &f, &cel, &loc );
            m.prop = loc.nb_prop; m.boites = loc.nb_boites; m.coupees = loc.nb_coupees;
            m.nentrees = loc.nentrees; m.nrejets = loc.nrejets;
            for ( int q = 0; q < loc.nentrees; ++q ) m.entrees[ q ] = loc.entrees[ q ];
            for ( int q = 0; q < loc.nrejets; ++q ) m.rejets[ q ] = loc.rejets[ q ];
            return r == 0;
        } else
            return false;
    }

    template<bool POIDS>
    bool cellule_( SI k, Cell &cel ) const {
        const TF w0 = POIDS ? arbre.seed_w( k ) : TF( 0 );
        if constexpr ( D == 2 ) {
            d2::FournisseurBsp<TK,POIDS> f( &arbre, arbre.seed_c( k, 0 ), arbre.seed_c( k, 1 ), w0, ids[ k ], k );
            d2::moteur<TK>( &f, &cel );
            return cel.nb >= 0;
        } else {
            d3::FournisseurBsp3<TK,POIDS> f( &arbre, arbre.seed_c( k, 0 ), arbre.seed_c( k, 1 ),
                                             arbre.seed_c( k, 2 ), w0, ids[ k ], k );
            return d3::moteur( &f, &cel ) == 0;
        }
    }
};

} // namespace sf
