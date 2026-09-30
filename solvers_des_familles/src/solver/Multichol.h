#pragma once
// =====================================================================================
// = LE CHOLESKY MULTI-ECHELLE ( Chen, Schafer, Huang, Desbrun, SIGGRAPH 2021 )
//
// Le § 24.16 a mesure qu'un IC(0) sur le motif brut de `L` perd un ordre de grandeur : sans
// correction grossiere, une factorisation incomplete ne touche pas aux basses frequences et le
// conditionnement reste en `h^-2`. Ce solveur est la reponse de la litterature a exactement ce
// defaut : GARDER le remplissage nul, mais sur un MOTIF MULTI-ECHELLE et dans un ORDRE
// fin-vers-grossier -- ce qui donne au facteur les proprietes d'homogeneisation qui manquaient.
//
// Les trois etapes du papier, telles qu'implementees ici :
//
//   1. ORDRE. L'ordre maximin INVERSE, c'est-a-dire l'echantillonnage du point le plus lointain
//      ( `i_k = argmax_i min_j dist( x_i, x_j )` sur les deja choisis ), puis retourne. Le papier
//      note que pour une distribution quasi uniforme on peut se contenter de la structure de
//      NIVEAUX qu'il implique ; c'est ce qu'on fait, par decimation : le niveau `k + 1` est un
//      sous-ensemble maximal du niveau `k` dont les points sont a plus de `2^(k+1) h` l'un de
//      l'autre. L'ordre final va du niveau 0 ( le plus fin ) au dernier.
//
//   2. MOTIF. `S = { ( i, j ) : dist( x_i, x_j ) <= rho min( l_i, l_j ) }`, avec `l_i` l'echelle
//      du niveau de `i` -- l'espacement moyen de ses points, `( volume / nb )^( 1/d )`, comme le
//      papier le prescrit. On y ajoute TOUS les non-nuls de `A` : le papier insiste pour ne perdre
//      aucun terme brut. `rho` est le seul reglage : il echange la taille du facteur contre sa
//      qualite, et le papier utilise 7 a 8 en 2D, 2.5 a 3.5 en 3D.
//
//   3. FACTORISATION. Cholesky incomplet a remplissage nul sur ce motif, colonne par colonne, de
//      gauche a droite. Le papier consacre une section aux pivots negatifs ; ils ne nous
//      concernent pas, parce que notre matrice est une M-MATRICE a diagonale dominante ( les
//      hors-diagonaux valent `-c_ij <= 0` ) et que l'IC ne casse pas sur cette classe
//      ( Meijerink et van der Vorst ). Le garde-fou est quand meme la, et il compte ses passages.
//
// CE QUI EST VOLONTAIREMENT LAISSE DE COTE, et qu'il faut savoir en lisant les mesures : le papier
// gagne l'essentiel de son temps de paroi sur des SUPERNOEUDS ( BLAS 3 ) et un coloriage
// multicolore pour le parallelisme. Ici la factorisation est scalaire et sequentielle ; seuls le
// produit matrice-vecteur et les operations vectorielles du CG sont paralleles. Un verdict
// defavorable en temps de paroi doit donc se lire avec cette reserve -- mais le COMPTE
// D'ITERATIONS, lui, est celui de la methode.
//
// = POURQUOI IL ENTRE BIEN DANS NOTRE REGIME
//
// Le motif est purement GEOMETRIQUE : il ne depend que des positions des germes, qui ne bougent
// pas. Il est donc bati UNE FOIS pour tout le solve, et chaque iteration de Newton ne refait que
// les valeurs -- c'est le critere que le § 24.16 a degage ( rafraichir les valeurs sans
// reconstruire la structure ), celui que le `spai0` gele d'AMGCL ne sait pas tenir.
// =====================================================================================

#include "Lineaire.h"
#include <algorithm>
#include <cmath>

namespace sf {

struct MultiChol : Lineaire {
    TF   rho      = 0;             ///< `0` : 7 en 2D, 3 en 3D ( les valeurs du papier )
    TF   tol      = 1e-10;         ///< residu RELATIF
    int  maxit    = 20000;
    int  min_niv  = 64;            ///< on arrete la hierarchie sous ce nombre de points
    /// L'ECHELLE `l_i` DU MOTIF. `0` : celle du papier -- UNIFORME PAR NIVEAU, estimee par
    /// `( volume / nb du niveau )^( 1/d )`. `1` : locale, la distance maximin au plus proche germe
    /// de niveau au moins egal.
    ///
    /// La mesure tranche en faveur du papier, et ce n'etait pas evident : A DENSITE EGALE du
    /// facteur, l'echelle uniforme par niveau fait 193 iterations de Krylov ( 102 termes par
    /// colonne ) la ou l'echelle locale en fait 1088 ( 124 termes ). La theorie veut que le support
    /// colle a celui de l'ondelette DU NIVEAU : c'est une propriete de niveau, et la faire varier
    /// d'un point a l'autre casse l'uniformite de la multiresolution. Le defaut de l'echelle
    /// globale est ailleurs : sur un nuage groupe, `( volume / nb )^( 1/d )` surestime
    /// l'espacement local de plusieurs ordres et le motif explose ( 288 termes par colonne ).
    int  echelle  = 0;
    int  trace    = 0;

    const char *nom() const override { return "CG + Cholesky multi-echelle ( IC sur motif multiscale )"; }
    TF tolerance() const override { return tol; }
    void tolerance( TF v ) override { tol = v; }

    void positions( const TF *const *P, SI nb, int dd ) override {
        Px = P; npos = nb; dim = dd;
        perm.clear(); Lp.clear();
    }

    bool sait_encore() const override { return ! Lp.empty(); }

    /// MEME MATRICE, MEME FACTEUR : la sonde du § 24.15 ne change que le second membre.
    void resout_encore( const std::vector<TF> &b, std::vector<TF> &d ) override {
        const double t0 = now();
        pcg( b, d );
        st.t_res += now() - t0;
    }

    bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d ) override {
        double t0 = now();
        m = L.n - 1;
        L.crs_reduit( ptr, col, val );
        st.t_forme += now() - t0;

        t0 = now();
        if ( SI( perm.size() ) != m && ! bati_motif() ) { st.t_hier += now() - t0; return false; }
        permute();
        if ( ! factorise() ) { st.t_hier += now() - t0; return false; }
        st.t_hier += now() - t0;
        ++st.nb_hier;

        t0 = now();
        const bool ok = pcg( b, d );
        st.t_res += now() - t0;
        return ok;
    }

private:
    // ---- les donnees du motif ( geometriques, baties une fois )
    const TF *const *Px = nullptr;
    SI   npos = 0;
    int  dim  = 2;
    SI   m    = 0;
    std::vector<SI>  perm, rang;   ///< `perm[ a ]` = l'indice reduit du a-ieme ; `rang` l'inverse
    std::vector<int> niv;          ///< le niveau de chaque indice reduit
    std::vector<TF>  ell;          ///< son echelle
    // ---- le facteur ( CSC bas-triangulaire, dans l'ordre permute )
    std::vector<SI> Lp, Li;
    std::vector<TF> Lx;
    std::vector<SI> Rp, Ri;        ///< le meme motif vu par LIGNES ( pour la boucle gauche )
    // ---- la matrice permutee ( CSR complet )
    std::vector<SI> Ap, Ai;
    std::vector<TF> Ax;
    // ---- les tampons
    std::vector<int>    ptr, col;
    std::vector<double> val;
    std::vector<TF> r_, z_, p_, q_, y_, bb_, xx_, dnn;
    std::vector<int> vu;
    int tampon = 0;
    std::vector<SI> pile;
    std::vector<SI> pos_;
    int nb_clamp = 0;

    TF pt( SI r, int k ) const { return Px[ k ][ r + 1 ]; }   // l'indice 0 est raye par la jauge


    // =================================================================================
    /// LA BOULE EUCLIDIENNE AUTOUR DE `r`, parcourue PAR LE GRAPHE.
    ///
    /// Le graphe du laplacien est un voisinage de Delaunay : un parcours en largeur borne par la
    /// distance euclidienne visite donc tous les germes de la boule, et il SUIT LA DENSITE -- ce
    /// qu'une grille de pas fixe ne peut pas faire quand l'espacement local varie de plusieurs
    /// ordres de grandeur, ce qui est le cas de nos nuages groupes.
    template<class F>
    void boule( SI r, TF R, F f ) {
        ++tampon;
        pile.clear(); pile.push_back( r ); vu[ r ] = tampon;
        while ( ! pile.empty() ) {
            const SI a = pile.back(); pile.pop_back();
            for ( int e = ptr[ a ]; e < ptr[ a + 1 ]; ++e ) {
                const SI q = col[ e ];
                if ( vu[ q ] == tampon ) continue;
                vu[ q ] = tampon;
                TF d2 = 0;
                for ( int k = 0; k < dim; ++k ) { const TF t = pt( r, k ) - pt( q, k ); d2 += t * t; }
                if ( d2 > R * R ) continue;
                f( q, std::sqrt( d2 ) );
                pile.push_back( q );
            }
        }
    }

    bool bati_motif() {
        if ( ! Px || npos < m + 1 ) return false;
        const int D = dim;
        vu.assign( m, 0 ); tampon = 0;

        // ---- 0. L'ECHELLE LOCALE : la plus courte arete du laplacien en chaque germe.
        //
        // Le papier donne un raccourci pour les distributions « quasi uniformes » : une echelle par
        // NIVEAU, estimee par `( volume / nb )^( 1/d )`. Mesure : sur nos nuages groupes ( cinq
        // lignes a sigma = 0.005 ) ce raccourci explose -- 40 termes par colonne des `rho = 1.5`
        // contre 6.3 sur l'uniforme, parce que l'espacement GLOBAL surestime de plusieurs ordres
        // l'espacement local dans les amas. L'echelle du vrai ordre maximin est une distance
        // d'insertion, donc une quantite LOCALE : on la prend ici par la plus courte arete
        // incidente, qui est gratuite et qui EST l'espacement local.
        TF vmax = 0;
        dnn.assign( m, 0 );
        for ( SI r = 0; r < m; ++r ) {
            TF best = 0;
            for ( int e = ptr[ r ]; e < ptr[ r + 1 ]; ++e ) {
                const SI q = col[ e ];
                if ( q == r ) continue;
                TF d2 = 0;
                for ( int k = 0; k < D; ++k ) { const TF t = pt( r, k ) - pt( q, k ); d2 += t * t; }
                const TF dd = std::sqrt( d2 );
                if ( best == 0 || dd < best ) best = dd;
            }
            dnn[ r ] = best;
            vmax = std::max( vmax, best );
        }
        for ( SI r = 0; r < m; ++r ) if ( ! ( dnn[ r ] > 0 ) ) dnn[ r ] = vmax;

        // ---- 1. LES NIVEAUX : un sous-ensemble maximal dont les points sont a plus de `2^k dnn`
        // l'un de l'autre. C'est la structure de niveaux de l'ordre maximin inverse, mais suivant
        // la densite locale au lieu de la supposer uniforme.
        niv.assign( m, 0 );
        std::vector<SI> cand( m ), pris;
        for ( SI r = 0; r < m; ++r ) cand[ r ] = r;
        std::vector<char> gard( m, 0 );
        int nk = 1;
        for ( int k = 1; k < 32; ++k ) {
            const TF fk = std::pow( TF( 2 ), TF( k ) );
            std::fill( gard.begin(), gard.end(), 0 );
            pris.clear();
            for ( SI r : cand ) {
                bool ok = true;
                boule( r, fk * dnn[ r ], [ & ]( SI q, TF ) { if ( gard[ q ] ) ok = false; } );
                if ( ok ) { gard[ r ] = 1; pris.push_back( r ); }
            }
            if ( SI( pris.size() ) < min_niv ) break;
            for ( SI r : pris ) niv[ r ] = k;
            cand = pris;
            nk = k + 1;
        }
        // ---- L'ECHELLE : LA DISTANCE D'INSERTION MAXIMIN, c'est-a-dire la distance au plus proche
        // germe DE NIVEAU AU MOINS EGAL.
        //
        // C'est le `l_i` du papier, et les deux raccourcis testes avant echouent chacun d'un cote :
        // une echelle globale par niveau ( `( volume / nb )^( 1/d )` ) donne la bonne qualite sur
        // l'uniforme mais explose sur un nuage groupe ( 288 termes par colonne ) ; la plus courte
        // arete incidente tient le remplissage mais SOUS-ESTIME l'espacement, donc les points
        // grossiers perdent le large support qui fait l'homogeneisation ( 23648 iterations de
        // Krylov ). La distance au plus proche point de meme niveau ou plus grossier est locale ET
        // croissante avec le niveau : elle a les deux proprietes.
        ell.assign( m, 0 );
        if ( echelle == 0 ) {
            // L'ECHELLE DU PAPIER : uniforme par niveau, l'espacement moyen de ses points.
            TF lo[ 3 ] = { 0, 0, 0 }, hi[ 3 ] = { 0, 0, 0 };
            for ( int k = 0; k < D; ++k ) { lo[ k ] = pt( 0, k ); hi[ k ] = lo[ k ]; }
            for ( SI r = 0; r < m; ++r )
                for ( int k = 0; k < D; ++k ) { lo[ k ] = std::min( lo[ k ], pt( r, k ) ); hi[ k ] = std::max( hi[ k ], pt( r, k ) ); }
            TF vol = 1;
            for ( int k = 0; k < D; ++k ) vol *= std::max( hi[ k ] - lo[ k ], TF( 1e-12 ) );
            std::vector<SI> nb( nk, 0 );
            for ( SI r = 0; r < m; ++r ) ++nb[ niv[ r ] ];
            std::vector<TF> el( nk );
            for ( int k = 0; k < nk; ++k ) el[ k ] = std::pow( vol / TF( std::max<SI>( nb[ k ], 1 ) ), TF( 1 ) / D );
            for ( int k = 1; k < nk; ++k ) el[ k ] = std::max( el[ k ], el[ k - 1 ] );
            for ( SI r = 0; r < m; ++r ) ell[ r ] = el[ niv[ r ] ];
        } else {
            // L'ECHELLE LOCALE : la distance maximin au plus proche germe de niveau au moins egal.
            for ( SI r = 0; r < m; ++r ) {
                const int k = niv[ r ];
                TF R = std::pow( TF( 2 ), TF( k ) ) * dnn[ r ], best = 0;
                for ( int essai = 0; essai < 24 && best == 0; ++essai, R *= 2 )
                    boule( r, R, [ & ]( SI q, TF dd ) {
                        if ( q != r && niv[ q ] >= k && ( best == 0 || dd < best ) ) best = dd;
                    } );
                ell[ r ] = best > 0 ? best : std::pow( TF( 2 ), TF( k ) ) * dnn[ r ];
            }
        }

        // ---- l'ordre : niveau croissant ( fin -> grossier ), ordre d'entree a l'interieur
        perm.resize( m ); rang.resize( m );
        {
            std::vector<SI> deb( nk + 1, 0 );
            for ( SI r = 0; r < m; ++r ) ++deb[ niv[ r ] + 1 ];
            for ( int k = 0; k < nk; ++k ) deb[ k + 1 ] += deb[ k ];
            std::vector<SI> cur = deb;
            for ( SI r = 0; r < m; ++r ) perm[ cur[ niv[ r ] ]++ ] = r;
            for ( SI a = 0; a < m; ++a ) rang[ perm[ a ] ] = a;
        }

        // ---- 2. LE MOTIF : `dist( x_i, x_j ) <= rho min( l_i, l_j )`, plus TOUS les non-nuls de
        // `A` -- le papier interdit d'en perdre un seul.
        const TF rh = rho > 0 ? rho : ( D == 2 ? TF( 7 ) : TF( 3 ) );
        std::vector<std::vector<SI>> cols( m );
        for ( SI r = 0; r < m; ++r ) {
            const SI a = rang[ r ];
            const TF R = rh * ell[ r ];
            boule( r, R, [ & ]( SI q, TF dd ) {
                if ( dd > rh * std::min( ell[ r ], ell[ q ] ) ) return;
                const SI bq = rang[ q ];
                if ( bq > a ) cols[ a ].push_back( bq );
                else if ( bq < a ) cols[ bq ].push_back( a );
            } );
            for ( int e = ptr[ r ]; e < ptr[ r + 1 ]; ++e ) {
                const SI bq = rang[ col[ e ] ];
                if ( bq > a ) cols[ a ].push_back( bq );
            }
        }
        Lp.assign( m + 1, 0 );
        for ( SI a = 0; a < m; ++a ) {
            auto &v = cols[ a ];
            std::sort( v.begin(), v.end() );
            v.erase( std::unique( v.begin(), v.end() ), v.end() );
            Lp[ a + 1 ] = Lp[ a ] + SI( v.size() ) + 1;   // + la diagonale
        }
        Li.resize( Lp[ m ] ); Lx.assign( Lp[ m ], 0 );
        for ( SI a = 0; a < m; ++a ) {
            SI k = Lp[ a ];
            Li[ k++ ] = a;                                // la diagonale en tete
            for ( SI b : cols[ a ] ) Li[ k++ ] = b;
        }
        // le motif vu par LIGNES : quelles colonnes deja faites mettent a jour la colonne `a`
        Rp.assign( m + 1, 0 );
        for ( SI a = 0; a < m; ++a )
            for ( SI k = Lp[ a ] + 1; k < Lp[ a + 1 ]; ++k ) ++Rp[ Li[ k ] + 1 ];
        for ( SI a = 0; a < m; ++a ) Rp[ a + 1 ] += Rp[ a ];
        Ri.resize( Rp[ m ] );
        {
            std::vector<SI> cur( Rp.begin(), Rp.end() - 1 );
            for ( SI a = 0; a < m; ++a )
                for ( SI k = Lp[ a ] + 1; k < Lp[ a + 1 ]; ++k ) Ri[ cur[ Li[ k ] ]++ ] = a;
        }
        if ( trace )
            std::printf( "      MULTICHOL : %d niveaux, rho %.1f, %.1f termes par colonne"
                         " ( %.2f x le laplacien )\n", nk, double( rh ),
                         double( Lp[ m ] ) / double( m ), double( Lp[ m ] ) / double( ptr[ m ] ) );
        return true;
    }

    /// la matrice dans l'ordre permute, CSR complet
    void permute() {
        Ap.assign( m + 1, 0 );
        for ( SI a = 0; a < m; ++a ) { const SI r = perm[ a ]; Ap[ a + 1 ] = ptr[ r + 1 ] - ptr[ r ]; }
        for ( SI a = 0; a < m; ++a ) Ap[ a + 1 ] += Ap[ a ];
        Ai.resize( Ap[ m ] ); Ax.resize( Ap[ m ] );
        for ( SI a = 0; a < m; ++a ) {
            SI k = Ap[ a ];
            const SI r = perm[ a ];
            for ( int e = ptr[ r ]; e < ptr[ r + 1 ]; ++e ) { Ai[ k ] = rang[ col[ e ] ]; Ax[ k ] = TF( val[ e ] ); ++k; }
        }
    }

    /// CHOLESKY INCOMPLET A REMPLISSAGE NUL sur le motif, colonne par colonne ( gauche a droite )
    bool factorise() {
        nb_clamp = 0;
        std::fill( Lx.begin(), Lx.end(), TF( 0 ) );
        pos_.assign( m, -1 );
        std::vector<TF> w( m, 0 );
        for ( SI a = 0; a < m; ++a ) {
            for ( SI k = Lp[ a ]; k < Lp[ a + 1 ]; ++k ) { pos_[ Li[ k ] ] = k; w[ Li[ k ] ] = 0; }
            for ( SI e = Ap[ a ]; e < Ap[ a + 1 ]; ++e ) {   // la colonne de `A`, restreinte au motif
                const SI i = Ai[ e ];
                if ( i >= a && pos_[ i ] >= 0 ) w[ i ] = Ax[ e ];
            }
            for ( SI u = Rp[ a ]; u < Rp[ a + 1 ]; ++u ) {   // les colonnes deja faites qui touchent `a`
                const SI c = Ri[ u ];
                TF lac = 0;
                for ( SI k = Lp[ c ] + 1; k < Lp[ c + 1 ]; ++k ) if ( Li[ k ] == a ) { lac = Lx[ k ]; break; }
                if ( lac == 0 ) continue;
                for ( SI k = Lp[ c ]; k < Lp[ c + 1 ]; ++k ) {
                    const SI i = Li[ k ];
                    if ( i < a ) continue;
                    if ( pos_[ i ] >= 0 ) w[ i ] -= lac * Lx[ k ];
                }
            }
            TF piv = w[ a ];
            if ( ! ( piv > 0 ) ) { piv = std::max( TF( 1e-30 ), std::fabs( Ax[ Ap[ a ] ] ) * TF( 1e-8 ) ); ++nb_clamp; }
            const TF s = std::sqrt( piv );
            Lx[ Lp[ a ] ] = s;
            for ( SI k = Lp[ a ] + 1; k < Lp[ a + 1 ]; ++k ) Lx[ k ] = w[ Li[ k ] ] / s;
            for ( SI k = Lp[ a ]; k < Lp[ a + 1 ]; ++k ) pos_[ Li[ k ] ] = -1;
        }
        return true;
    }

    void applique( const std::vector<TF> &r, std::vector<TF> &z ) const {
        z = r;
        for ( SI a = 0; a < m; ++a ) {                    // L y = r
            const TF v = z[ a ] / Lx[ Lp[ a ] ];
            z[ a ] = v;
            for ( SI k = Lp[ a ] + 1; k < Lp[ a + 1 ]; ++k ) z[ Li[ k ] ] -= Lx[ k ] * v;
        }
        for ( SI a = m - 1; a >= 0; --a ) {               // L^t z = y
            TF s = z[ a ];
            for ( SI k = Lp[ a ] + 1; k < Lp[ a + 1 ]; ++k ) s -= Lx[ k ] * z[ Li[ k ] ];
            z[ a ] = s / Lx[ Lp[ a ] ];
        }
    }

    void matvec( const std::vector<TF> &x, std::vector<TF> &y ) const {
#ifdef _OPENMP
#       pragma omp parallel for schedule( static )
#endif
        for ( SI a = 0; a < m; ++a ) {
            TF s = 0;
            for ( SI e = Ap[ a ]; e < Ap[ a + 1 ]; ++e ) s += Ax[ e ] * x[ Ai[ e ] ];
            y[ a ] = s;
        }
    }

    bool pcg( const std::vector<TF> &b, std::vector<TF> &d ) {
        bb_.assign( m, 0 ); xx_.assign( m, 0 );
        for ( SI r = 0; r < m; ++r ) bb_[ rang[ r ] ] = b[ r + 1 ];
        r_ = bb_; z_.assign( m, 0 ); p_.assign( m, 0 ); q_.assign( m, 0 );
        TF nb2 = 0;
        for ( SI a = 0; a < m; ++a ) nb2 += bb_[ a ] * bb_[ a ];
        if ( ! ( nb2 > 0 ) ) { d.assign( m + 1, 0 ); return true; }
        applique( r_, z_ );
        p_ = z_;
        TF rz = 0;
        for ( SI a = 0; a < m; ++a ) rz += r_[ a ] * z_[ a ];
        const TF cible = tol * tol * nb2;
        int it = 0;
        TF rr = nb2;
        for ( ; it < maxit && rr > cible; ++it ) {
            matvec( p_, q_ );
            TF pq = 0;
            for ( SI a = 0; a < m; ++a ) pq += p_[ a ] * q_[ a ];
            if ( ! ( pq > 0 ) ) break;
            const TF al = rz / pq;
            rr = 0;
            for ( SI a = 0; a < m; ++a ) { xx_[ a ] += al * p_[ a ]; r_[ a ] -= al * q_[ a ]; rr += r_[ a ] * r_[ a ]; }
            if ( rr <= cible ) { ++it; break; }
            applique( r_, z_ );
            TF rz2 = 0;
            for ( SI a = 0; a < m; ++a ) rz2 += r_[ a ] * z_[ a ];
            const TF be = rz2 / rz;
            rz = rz2;
            for ( SI a = 0; a < m; ++a ) p_[ a ] = z_[ a ] + be * p_[ a ];
        }
        d.assign( m + 1, 0 );
        for ( SI r = 0; r < m; ++r ) d[ r + 1 ] = xx_[ rang[ r ] ];
        st.nb_iter += it;
        st.pire = std::max( st.pire, TF( std::sqrt( rr / nb2 ) ) );
        return true;
    }
};

} // namespace sf
