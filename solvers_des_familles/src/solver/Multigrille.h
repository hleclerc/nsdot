#pragma once

// =====================================================================================
// UN MULTIGRILLE ALGEBRIQUE MAISON, SUR CPU. C'est le portage de `gpu_des_familles/src/gpu/
// Amg2D.cuh` -- meme agregation, meme cycle, memes constantes, qui y ont ete mesurees.
//
// = Pourquoi le remplacer
//
// `main_image` resolvait par Eigen `SimplicialLDLT`, qui est STRICTEMENT SEQUENTIEL : mesure a
// `n = 1e5`, 84 s sur 142 pour le relevement et 106 s sur 157 pour `essai-limites`, soit 59 a
// 68 % du temps dans un seul fil -- 2.6 CPU occupes sur 8 demandes. AMGCL parallelise, mais son
// agregation coute un appariement, et sa variante par defaut ici ( Ruge-Stuben + Gauss-Seidel )
// est elle-meme sequentielle des deux cotes.
//
// = L'AGREGATION EST GRATUITE, et c'est tout le point
//
// Les germes sont ranges DANS L'ORDRE DE L'ARBRE ( `pd.ids` ), qui est une courbe remplissante :
// des RANGS CONSECUTIFS sont voisins dans le plan. Agreger, c'est donc `rang >> 2` -- quatre
// germes par paquet, sans noyau d'appariement, sans matching, sans compaction. Et comme les
// indices d'agregat restent ordonnes par rang, le niveau suivant s'agrege pareil : `a >> 2`. La
// hierarchie entiere tient dans un decalage.
//
// Mieux : LA CARTE INVERSE EST GRATUITE ELLE AUSSI. Le paquet `a` contient exactement les rangs
// `4a .. 4a+3`, donc les identifiants `ord[ 4a ] .. ord[ 4a+3 ]` au niveau fin et les indices
// `4a .. 4a+3` ensuite. On assemble donc le grossier EN BALAYANT LES LIGNES GROSSIERES, chacune
// par un seul fil, sans atomique et sans tri -- la ou le GPU emet un triplet par arete, trie
// ( CUB ) et reduit par clef. La restriction aussi se fait par RAMASSAGE au lieu de dispersion.
//
// = LE GROSSIER : Galerkin, `A_c = P^T A P` avec `P` constant par morceaux
//
// Sur un laplacien de graphe c'est encore un laplacien : il suffit de sommer les poids d'aretes
// entre paquets, `c_ab = somme des c_ij pour i dans a, j dans b`, et la diagonale est la somme de
// la ligne ( la ligne fine somme deja a zero, donc le Galerkin preserve la propriete ).
//
// = LE CYCLE EN V, et le K-CYCLE
//
// Un lissage de Jacobi amorti avant, un apres ( meme `omega`, donc l'operateur est SYMETRIQUE et
// le CG l'accepte comme preconditionneur ), restriction par somme sur le paquet, prolongation par
// injection.
//
// LE NIVEAU LE PLUS GROSSIER EST RESOLU EXACTEMENT, ET C'EST LA DEUXIEME DIFFERENCE AVEC LA CARTE.
// Le GPU le lisse cent vingt fois par visite, et le K-cycle le visite quatre fois : quatre cent
// quatre-vingts balayages par application du preconditionneur. Sur la carte c'est un bon
// arbitrage -- le parallelisme est massif, un lancement de noyau coute cinq microsecondes. Sur
// huit coeurs c'est le poste dominant, et la mesure le dit trois fois plutot qu'une ( `n = 2e4`,
// en secondes ) : `gros 40` 15.9 contre 23.8, `k 0` 16.0, `stop 200` 17.8 -- moins de balayages,
// moins de visites ou un niveau plus petit donnent le MEME gain, donc c'est bien le travail au
// fond du cycle qu'on paie. Or a mille inconnues une factorisation de Cholesky creuse coute
// quelques millisecondes une fois par hierarchie, et vingt microsecondes par visite. On resout
// donc, au lieu de lisser -- et la correction grossiere devient EXACTE, ce qui enleve aussi la
// faiblesse que le K-cycle etait la pour compenser.
//
// Le defaut mesure de l'agregation NON LISSEE est que la correction grossiere est trop faible.
// Le K-cycle y repond en ACCELERANT LES DEUX PREMIERS NIVEAUX PAR KRYLOV : au lieu d'un appel
// recursif, DEUX pas d'un gradient conjugue sur le systeme grossier, preconditionnes par le
// niveau d'en dessous.
//
// JACOBI A DEUX TAMPONS, ET C'EST UNE DIFFERENCE ASSUMEE AVEC LE GPU. Le noyau CUDA lit
// `x[ col ]` pendant que d'autres fils l'ecrivent : c'est un Jacobi/Gauss-Seidel hybride, non
// deterministe. Sur la carte ca passe ; dans un preconditionneur de CG c'est faux en droit -- CG
// exige un operateur LINEAIRE FIXE. On alterne donc deux tampons, ce qui coute un vecteur et rend
// l'operateur exactement symetrique.
//
// OPENMP ET PAS `parallel_for`. `util/parallel.h` cree et joint ses fils A CHAQUE APPEL, avec
// epinglage : une cinquantaine de microsecondes. Le niveau le plus grossier en demande cent
// vingt par cycle sur mille inconnues -- la creation couterait cent fois le calcul. Le pool
// d'OpenMP, lui, est deja la, et la clause `if` rend la boucle sequentielle quand elle est
// courte.
//
// = LA JAUGE : MOYENNE NULLE POUR RESOUDRE, `d[ 0 ] = 0` POUR RENDRE
//
// La moyenne nulle est la bonne jauge pour un multigrille : le laplacien a les constantes pour
// noyau, `b = nu - a` est deja de somme nulle, et projeter est symetrique la ou rayer une ligne ne
// l'est pas. Mais le reste du code suppose l'autre -- `Newton.h` ecrit `w2 = w + t d` PUIS
// `w2[ 0 ] = 0`, « la jauge, imposee et non esperee ». On TRANSLATE donc la solution en sortie.
// Les deux jauges decrivent la meme direction a une constante pres, et une constante ajoutee a
// tous les poids ne change aucune cellule ; mais rendre l'une quand l'appelant attend l'autre
// mutile une composante, et Newton stagne sur-le-champ ( mesure : residu inchange, 31 reculs ).
// =====================================================================================

#include "solver/Lineaire.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

namespace sf {

/// en dessous, la boucle reste sequentielle : le fork/join coute plus que le travail
static constexpr SI MG_SEUIL_PAR = 2048;

/// un niveau de la hierarchie. Le niveau zero POINTE sur le laplacien de l'appelant ; les autres
/// possedent leur matrice. Convention partout : `y_i = dia_i x_i - somme_e val_e x_( col_e )`.
struct NiveauMg {
    SI               n = 0, nnz = 0;
    const SI        *row = nullptr, *col = nullptr;
    const TF        *val = nullptr, *dia = nullptr;
    std::vector<SI>  arow, acol;                     ///< la possession, pour les niveaux grossiers
    std::vector<TF>  aval, adia;
    std::vector<TF>  x, b, r, y;                     ///< le cycle ( `y` : le second tampon de Jacobi )
    std::vector<TF>  v1, v2, t, rc;                  ///< le K-cycle
};

struct Mg {
    // ---- LES CONSTANTES, mesurees sur la carte ( `gpu_des_familles`, `Amg2D.cuh` )
    int      kcycle = 2;           ///< niveaux acceleres par Krylov ( 0 : cycle en V pur )
    int      gros   = 120;         ///< lissages au niveau le plus grossier
    int      nu     = 2;           ///< lissages avant et apres, par niveau
    int      stop   = 1000;        ///< on arrete de grossir en dessous
    bool     exact  = true;        ///< le niveau le plus grossier RESOLU ( Cholesky ) au lieu de lisse
    TF       omega  = TF( 0.7 );   ///< l'amortissement de Jacobi
    TF       tol    = TF( 1e-10 ); ///< residu RELATIF
    int      maxit  = 20000;

    StatsLin st;

    /// `ids[ k ]` : l'identifiant du germe de rang `k` dans l'arbre. C'est `pd.ids`, et c'est la
    /// SEULE chose que ce solveur demande de plus qu'un autre. Sans lui, l'ordre des identifiants
    /// fait l'affaire -- et l'agregation ne vaut alors que ce que vaut cet ordre.
    template<class TI>
    void ordre( const TI *ids, SI nb ) {
        ord.resize( nb );
        rg.resize( nb );
        for ( SI k = 0; k < nb; ++k ) { ord[ k ] = SI( ids[ k ] ); rg[ ord[ k ] ] = k; }
    }
    bool a_l_ordre() const { return ! ord.empty(); }

    const char *nom() const { return "multigrille maison ( agregation par l'arbre, K-cycle )"; }

    /// combien de niveaux, et leurs tailles -- pour la trace
    const std::vector<NiveauMg> &niveaux() const { return niv; }

    /// UNE RESOLUTION DE PLUS sur la derniere hierarchie. Meme surface que `Amg` et `Cholesky`.
    void resout_encore( const std::vector<TF> &b, std::vector<TF> &d ) {
        const double t0 = now();
        cg( b, d );
        st.t_res += now() - t0;
    }

    bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d ) {
        const double t0 = now();
        monte( L );
        const double t1 = now();
        st.t_hier += t1 - t0;
        ++st.nb_hier;
        const bool ok = cg( b, d );
        st.t_res += now() - t1;
        return ok;
    }

private:
    std::vector<SI>       ord, rg;         ///< rang -> identifiant, et son inverse
    std::vector<NiveauMg> niv;
    std::vector<std::vector<SI>> carte;    ///< `carte[ l ][ i ]` : le paquet de `i` au niveau `l+1`
    std::vector<TF>       cr, cz, cp, cq;  ///< les vecteurs du CG externe
    // les tampons de l'assemblage grossier, gardes d'un appel a l'autre
    std::vector<std::vector<TF>> acc, tval;
    std::vector<std::vector<SI>> tcol;
#ifdef SF_EIGEN
    // LA FACTORISATION DU FOND. La jauge y est `x[ 0 ] = 0` -- on raye la ligne et la colonne --
    // parce que le laplacien grossier est singulier ( les constantes ), et que rayer suffit ici :
    // l'operateur `P A_red^-1 P^t` reste symetrique semi-defini positif, donc CG l'accepte.
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> fgros;
    bool            gros_pret = false;
    Eigen::VectorXd egb, egx;
#endif

    /// l'indice fin du `t`-ieme membre du paquet `a`, au niveau `l` ( -1 s'il n'existe pas )
    SI membre( int l, SI a, int t, SI nf ) const {
        const SI k = 4 * a + t;
        if ( k >= nf ) return -1;
        return ( l == 0 && ! ord.empty() ) ? ord[ k ] : k;
    }

    // ---------------------------------------------------------------- LA HIERARCHIE
    void monte( const Laplacien &L ) {
        niv.clear();
        carte.clear();

        NiveauMg f;
        f.n = L.n; f.nnz = SI( L.col.size() );
        f.row = L.row.data(); f.col = L.col.data(); f.val = L.c.data(); f.dia = L.dia.data();
        niv.push_back( std::move( f ) );

        for ( int l = 0; niv[ l ].n > stop && l < 24; ++l )
            grossit( l );

        for ( NiveauMg &v : niv ) {
            v.x.assign( v.n, TF( 0 ) ); v.b.assign( v.n, TF( 0 ) );
            v.r.assign( v.n, TF( 0 ) ); v.y.assign( v.n, TF( 0 ) );
        }
        for ( int l = 1; l <= kcycle && l < int( niv.size() ); ++l ) {
            NiveauMg &v = niv[ l ];
            v.v1.assign( v.n, TF( 0 ) ); v.v2.assign( v.n, TF( 0 ) );
            v.t.assign( v.n, TF( 0 ) );  v.rc.assign( v.n, TF( 0 ) );
        }
        factorise_le_fond();
    }

    /// le niveau le plus grossier, factorise UNE FOIS par hierarchie
    void factorise_le_fond() {
#ifdef SF_EIGEN
        gros_pret = false;
        if ( ! exact )
            return;
        const NiveauMg &g = niv.back();
        const SI m = g.n - 1;
        if ( m <= 0 )
            return;
        std::vector<Eigen::Triplet<double>> tr;
        tr.reserve( size_t( g.nnz + m ) );
        for ( SI i = 1; i < g.n; ++i ) {
            tr.emplace_back( int( i - 1 ), int( i - 1 ), double( g.dia[ i ] ) );
            for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e )
                if ( g.col[ e ] >= 1 )
                    tr.emplace_back( int( i - 1 ), int( g.col[ e ] - 1 ), -double( g.val[ e ] ) );
        }
        Eigen::SparseMatrix<double> A( m, m );
        A.setFromTriplets( tr.begin(), tr.end() );
        fgros.compute( A );
        gros_pret = fgros.info() == Eigen::Success;
        egb.resize( m ); egx.resize( m );
#endif
    }

    /// LE PRODUIT DE GALERKIN, UNE LIGNE GROSSIERE PAR FIL. Le paquet `a` est fait des rangs
    /// `4a..4a+3`, donc un fil qui tient `a` connait ses membres sans rien chercher : il parcourt
    /// leurs aretes, traduit chaque colonne en paquet, et accumule dans un tampon dense qu'il
    /// remet a zero par la liste des cases touchees. Aucune atomique, aucun tri.
    void grossit( int l ) {
        const NiveauMg &g = niv[ l ];
        const SI nf = g.n, nc = ( nf + 3 ) / 4;
        const int T = omp_get_max_threads();

        std::vector<SI> &m = carte.emplace_back();     // la carte `fin -> paquet`
        m.resize( nf );
        if ( l == 0 && ! rg.empty() ) {
            #pragma omp parallel for schedule( static ) if( nf >= MG_SEUIL_PAR )
            for ( SI i = 0; i < nf; ++i ) m[ i ] = rg[ i ] >> 2;
        } else {
            #pragma omp parallel for schedule( static ) if( nf >= MG_SEUIL_PAR )
            for ( SI i = 0; i < nf; ++i ) m[ i ] = i >> 2;
        }

        acc.resize( T ); tcol.resize( T ); tval.resize( T );
        for ( int t = 0; t < T; ++t ) { acc[ t ].assign( nc, TF( 0 ) ); tcol[ t ].clear(); tval[ t ].clear(); }
        std::vector<SI> len( nc ), loc( nc ), fil( nc );

        #pragma omp parallel for schedule( static ) if( nc >= MG_SEUIL_PAR )
        for ( SI a = 0; a < nc; ++a ) {
            const int t = omp_get_thread_num();
            std::vector<TF> &ac = acc[ t ];
            std::vector<SI> &tc = tcol[ t ];
            std::vector<TF> &tv = tval[ t ];
            const SI deb = SI( tc.size() );
            for ( int q = 0; q < 4; ++q ) {
                const SI i = membre( l, a, q, nf );
                if ( i < 0 ) continue;
                for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) {
                    const SI bb = m[ g.col[ e ] ];
                    if ( bb == a ) continue;             // l'interieur du paquet disparait
                    if ( ac[ bb ] == 0 ) tc.push_back( bb );
                    ac[ bb ] += g.val[ e ];
                }
            }
            for ( SI k = deb; k < SI( tc.size() ); ++k ) { tv.push_back( ac[ tc[ k ] ] ); ac[ tc[ k ] ] = 0; }
            loc[ a ] = deb;
            len[ a ] = SI( tc.size() ) - deb;
            fil[ a ] = t;
        }

        NiveauMg c;
        c.n = nc;
        c.arow.assign( nc + 1, 0 );
        for ( SI a = 0; a < nc; ++a ) c.arow[ a + 1 ] = c.arow[ a ] + len[ a ];
        c.nnz = c.arow[ nc ];
        c.acol.resize( c.nnz );
        c.aval.resize( c.nnz );
        c.adia.assign( nc, TF( 0 ) );
        #pragma omp parallel for schedule( static ) if( nc >= MG_SEUIL_PAR )
        for ( SI a = 0; a < nc; ++a ) {
            const int t = fil[ a ];
            const SI  o = loc[ a ], p = c.arow[ a ];
            TF s = 0;
            for ( SI k = 0; k < len[ a ]; ++k ) {
                c.acol[ p + k ] = tcol[ t ][ o + k ];
                c.aval[ p + k ] = tval[ t ][ o + k ];
                s += tval[ t ][ o + k ];
            }
            c.adia[ a ] = s > 0 ? s : TF( 1 );           // une ligne nulle rendrait Jacobi fou
        }
        c.row = c.arow.data(); c.col = c.acol.data(); c.val = c.aval.data(); c.dia = c.adia.data();
        niv.push_back( std::move( c ) );
    }

    // ---------------------------------------------------------------- LES BRIQUES
    static void matvec( const NiveauMg &v, const std::vector<TF> &x, std::vector<TF> &y ) {
        const SI n = v.n;
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) {
            TF s = v.dia[ i ] * x[ i ];
            for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) s -= v.val[ e ] * x[ v.col[ e ] ];
            y[ i ] = s;
        }
    }
    /// `nb` lissages de Jacobi amorti, DEUX TAMPONS ; `net` : on part de `x = 0`
    void jacobi( NiveauMg &v, int nb, bool net ) const {
        const SI n = v.n;
        if ( net ) std::fill( v.x.begin(), v.x.end(), TF( 0 ) );
        for ( int k = 0; k < nb; ++k ) {
            const TF *xi = v.x.data();
            TF *xo = v.y.data();
            #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
            for ( SI i = 0; i < n; ++i ) {
                TF s = v.dia[ i ] * xi[ i ];
                for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) s -= v.val[ e ] * xi[ v.col[ e ] ];
                xo[ i ] = xi[ i ] + omega * ( v.b[ i ] - s ) / v.dia[ i ];
            }
            v.x.swap( v.y );
        }
    }
    static TF dot( const std::vector<TF> &u, const std::vector<TF> &v ) {
        const SI n = SI( u.size() );
        TF s = 0;
        #pragma omp parallel for schedule( static ) reduction( + : s ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) s += u[ i ] * v[ i ];
        return s;
    }

    // ---------------------------------------------------------------- LE CYCLE
    void cycle( int l ) {
        NiveauMg &g = niv[ l ];
        if ( l + 1 == int( niv.size() ) ) {
#ifdef SF_EIGEN
            if ( gros_pret ) {
                for ( SI i = 1; i < g.n; ++i ) egb[ i - 1 ] = double( g.b[ i ] );
                egx = fgros.solve( egb );
                g.x[ 0 ] = 0;
                for ( SI i = 1; i < g.n; ++i ) g.x[ i ] = TF( egx[ i - 1 ] );
                return;
            }
#endif
            jacobi( g, gros, true );
            return;
        }
        NiveauMg &c = niv[ l + 1 ];
        const SI nf = g.n, ncc = c.n;

        jacobi( g, nu, true );
        #pragma omp parallel for schedule( static ) if( nf >= MG_SEUIL_PAR )
        for ( SI i = 0; i < nf; ++i ) {                  // `r = b - A x`
            TF s = g.dia[ i ] * g.x[ i ];
            for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) s -= g.val[ e ] * g.x[ g.col[ e ] ];
            g.r[ i ] = g.b[ i ] - s;
        }
        // LA RESTRICTION PAR RAMASSAGE : le paquet connait ses membres, donc pas d'atomique.
        #pragma omp parallel for schedule( static ) if( ncc >= MG_SEUIL_PAR )
        for ( SI a = 0; a < ncc; ++a ) {
            TF s = 0;
            for ( int q = 0; q < 4; ++q ) { const SI i = membre( l, a, q, nf ); if ( i >= 0 ) s += g.r[ i ]; }
            c.b[ a ] = s;
        }

        if ( l >= kcycle ) cycle( l + 1 );
        else               kcycle_deux( l + 1 );

        const std::vector<SI> &m = carte[ l ];
        #pragma omp parallel for schedule( static ) if( nf >= MG_SEUIL_PAR )
        for ( SI i = 0; i < nf; ++i ) g.x[ i ] += c.x[ m[ i ] ];
        jacobi( g, nu, false );
    }

    /// DEUX PAS DE GRADIENT CONJUGUE SUR LE NIVEAU `l`, preconditionnes par le niveau d'en
    /// dessous. C'est le K-cycle : on ne change pas `P`, on accelere chaque niveau.
    void kcycle_deux( int l ) {
        NiveauMg &c = niv[ l ];
        const SI n = c.n;
        c.rc = c.b;                                      // le second membre du premier pas
        cycle( l );                                      // `v1 = M^-1 b`
        c.v1 = c.x;
        matvec( c, c.v1, c.t );                          // `t = A v1`
        const TF rho1 = dot( c.v1, c.t );
        const TF a1   = dot( c.v1, c.rc );
        const TF k1   = rho1 != 0 ? a1 / rho1 : TF( 0 );
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) c.b[ i ] = c.rc[ i ] - k1 * c.t[ i ];
        cycle( l );                                      // `v2 = M^-1 r`
        c.v2 = c.x;
        const TF g2 = dot( c.v2, c.t );                  // `t` porte encore `A v1`
        const TF a2 = dot( c.v2, c.b );
        matvec( c, c.v2, c.t );
        const TF b2   = dot( c.v2, c.t );
        const TF rho2 = b2 - ( rho1 != 0 ? g2 * g2 / rho1 : TF( 0 ) );
        const TF k2   = rho2 != 0 ? a2 / rho2 : TF( 0 );
        const TF k1c  = k1 - ( rho1 != 0 && rho2 != 0 ? g2 * a2 / ( rho1 * rho2 ) : TF( 0 ) );
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) c.x[ i ] = k1c * c.v1[ i ] + k2 * c.v2[ i ];
        c.b.swap( c.rc );                                // rendu tel qu'on l'a trouve
    }

    /// la jauge : moyenne nulle, le noyau du laplacien
    static void centre( std::vector<TF> &v ) {
        const SI n = SI( v.size() );
        if ( n <= 0 ) return;
        TF s = 0;
        #pragma omp parallel for schedule( static ) reduction( + : s ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) s += v[ i ];
        const TF mu = s / TF( n );
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) v[ i ] -= mu;
    }

    // ---------------------------------------------------------------- LE CG EXTERNE
    bool cg( const std::vector<TF> &b, std::vector<TF> &d ) {
        const SI n = niv[ 0 ].n;
        d.assign( n, TF( 0 ) );
        cr.assign( b.begin(), b.begin() + n );
        centre( cr );
        cz.assign( n, TF( 0 ) ); cp.assign( n, TF( 0 ) ); cq.assign( n, TF( 0 ) );

        auto precond = [ & ]( const std::vector<TF> &rr, std::vector<TF> &zz ) {
            if ( niv.size() > 1 ) {
                niv[ 0 ].b = rr;
                cycle( 0 );
                zz = niv[ 0 ].x;
            } else {
                const TF *di = niv[ 0 ].dia;
                #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
                for ( SI i = 0; i < n; ++i ) zz[ i ] = rr[ i ] / di[ i ];
            }
            centre( zz );
        };

        const TF bb = dot( cr, cr );
        if ( ! ( bb > 0 ) ) return true;                 // rien a resoudre
        precond( cr, cz );
        cp = cz;
        TF rz = dot( cr, cz );
        const TF cible = tol * tol * bb;
        TF rr = bb;
        int it = 0;
        for ( ; it < maxit && rr > cible; ++it ) {
            matvec( niv[ 0 ], cp, cq );
            const TF pq = dot( cp, cq );
            if ( ! ( pq > 0 ) ) break;                   // la direction est dans le noyau : fini
            const TF al = rz / pq;
            TF s = 0;
            #pragma omp parallel for schedule( static ) reduction( + : s ) if( n >= MG_SEUIL_PAR )
            for ( SI i = 0; i < n; ++i ) {
                d[ i ] += al * cp[ i ];
                cr[ i ] -= al * cq[ i ];
                s += cr[ i ] * cr[ i ];
            }
            rr = s;
            if ( rr <= cible ) { ++it; break; }
            precond( cr, cz );
            const TF rz2 = dot( cr, cz );
            const TF be = rz != 0 ? rz2 / rz : TF( 0 );
            rz = rz2;
            #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
            for ( SI i = 0; i < n; ++i ) cp[ i ] = cz[ i ] + be * cp[ i ];
        }
        st.nb_iter += it;
        // ---- ON REND LA JAUGE `d[ 0 ] = 0`, ET CE N'EST PAS UN DETAIL
        //
        // La moyenne nulle est la bonne jauge POUR RESOUDRE ( le noyau du laplacien est les
        // constantes, et projeter est symetrique la ou rayer une ligne ne l'est pas ), mais le
        // reste du code suppose l'autre : `Newton.h` ecrit `w2[ i ] = w[ i ] + t d[ i ]` PUIS
        // `w2[ 0 ] = 0`, « la jauge, imposee et non esperee ». Avec `d[ 0 ] != 0` cette ligne
        // n'impose plus une jauge, elle MUTILE la direction sur une composante -- et Newton
        // stagnait aussitot ( mesure : residu 1.14e+01 inchange apres deux iterations, 31 reculs ).
        // Les deux jauges decrivent la MEME direction a une constante pres, donc on translate.
        const TF d0 = d[ 0 ];
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) d[ i ] -= d0;
        const TF err = std::sqrt( rr / bb );
        st.pire = std::max( st.pire, err );
        return err < 1;
    }
};

} // namespace sf
