// =====================================================================================
// LA MEMOIRE EN 3D, SA BORNE SUPERIEURE. La question de `2d_des_familles` ( journal, `--memo` ) :
// que rend le fait de proposer D'ABORD les diracs qui portaient une face de la cellule a la passe
// precedente ? Ici les deux passes ont LES MEMES POIDS, donc chaque souvenir est exact -- c'est
// le mieux que l'idee puisse rendre, et ce qu'une boucle de Newton ne peut qu'eroder.
//
// Par nuage, minimum de `--reps` :
//   sans memoire        le chemin ordinaire ( `cellule` ) ;
//   temoin              le chemin MEMO a vide : le prix du code de plus ;
//   A                   les voisins d'hier ( par rang ) proposes d'abord, le parcours en complement ;
//   B                   les feuilles entrees hier, un bit par voisin : les bits d'abord sans test de
//                       boite, puis le parcours ou ces feuilles proposent leur complement -- testees
//                       avant d'y entrer, ou pas ( on sait qu'on y entre ) ;
//   les voisins seuls   pas de parcours du tout : le plancher, ce que coutent les coupes utiles.
// Et par cellule : plans proposes, boites testees, coupes effectives, feuilles entrees.
//
// `--perime T` ( nuages a poids ) : les souvenirs pris a `T * W`, le diagramme mesure a `W` -- ce
// que Newton fait subir a la memoire, en pire ( `T = 0` : les souvenirs de Voronoi ).
//
//   xmake run memo --threads 8
//   xmake run memo --load uniforme -n 1000000 --3d
//   xmake run memo --perime 0.9
// =====================================================================================

#include "bench/Dispatch.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

using namespace sf;

namespace {

double perime = -1;                                      ///< < 0 : souvenirs exacts

template<class PD>
int mesure( const Args &a, const Nuage<3> &nu ) {
    const SI n = nu.n;
    PD pd;
    pd.build( nu.P, nu.W, n, a.leaf );
    std::vector<TF> w_perime;
    if ( perime >= 0 && nu.W ) {                         // les souvenirs viennent d'AUTRES poids
        w_perime.resize( n );
        for ( SI i = 0; i < n; ++i ) w_perime[ i ] = TF( perime ) * nu.W[ i ];
        pd.set_weights( w_perime.data(), a.par );
    }
    std::vector<SI> rang( n );
    for ( SI k = 0; k < n; ++k ) rang[ pd.ids[ k ] ] = k;

    // ---- sans memoire ( aux vrais poids )
    std::vector<TF> res;
    if ( ! w_perime.empty() ) pd.set_weights( nu.W, a.par );
    pd.measures( res, a.par );
    double t_sans = 1e300;
    for ( int r = 0; r < a.reps; ++r ) { const double t0 = now(); pd.measures( res, a.par ); t_sans = std::min( t_sans, now() - t0 ); }
    TF somme_sans = 0;
    for ( TF v : res ) somme_sans += v;

    // ---- les souvenirs, de la passe temoin ( MEMO a vide ) : les voisins de chaque cellule en rangs
    // ( forme A, CSR ) et les feuilles entrees avec un bit par voisin ( forme B, CSR )
    if ( ! w_perime.empty() ) pd.set_weights( w_perime.data(), a.par );
    const int nth = std::max( a.par.threads, 1 );
    std::vector<SI> feuille_de( n );                     // rang -> rang du premier germe de sa feuille
    for ( const auto &nd : pd.arbre.nodes ) if ( nd.right < 0 ) for ( SI k = nd.beg; k < nd.end; ++k ) feuille_de[ k ] = nd.beg;
    std::vector<std::vector<d2::SI32>> vois( n ), frontiere( n );
    std::vector<std::vector<std::pair<d2::SI32,unsigned long long>>> feuilles( n );
    std::atomic<SI> nb_vois{ 0 }, nb_feuilles{ 0 }, max_feuilles{ 0 }, nb_sature{ 0 }, nb_rejets{ 0 }, nb_front_sature{ 0 };
    parallel_for( n, a.par, [ & ]( SI k, int ) {
        typename PD::Cell cel;
        typename PD::Memo m;
        pd.cellule_memo( k, cel, m );
        d2::SI32 out[ PD::max_nv ];
        const int nv = cel.voisins( out, PD::max_nv );
        // C : la frontiere = les feuilles entrees ( noeuds ) puis les rejetes ; incomplete si un tampon a sature
        if ( m.nentrees < 64 && m.nrejets < 256 ) {
            for ( int q = 0; q < m.nentrees; ++q ) frontiere[ k ].push_back( d2::SI32( m.entrees[ q ] ) );
            for ( int q = 0; q < m.nrejets; ++q ) frontiere[ k ].push_back( d2::SI32( m.rejets[ q ] ) );
        } else ++nb_front_sature;
        nb_rejets += m.nrejets;
        std::vector<d2::SI32> fb( m.entrees, m.entrees + m.nentrees );
        for ( d2::SI32 &b : fb ) b = d2::SI32( pd.arbre.nodes[ b ].beg );   // noeud -> rang du premier germe
        std::sort( fb.begin(), fb.end() );
        for ( d2::SI32 b : fb ) feuilles[ k ].push_back( { b, 0ull } );
        for ( int q = 0; q < nv; ++q ) if ( out[ q ] >= 0 ) {
            const SI r = rang[ out[ q ] ];
            vois[ k ].push_back( d2::SI32( r ) );
            const SI fbk = feuille_de[ r ];
            auto it = std::lower_bound( fb.begin(), fb.end(), d2::SI32( fbk ) );
            if ( it != fb.end() && *it == d2::SI32( fbk ) ) feuilles[ k ][ it - fb.begin() ].second |= 1ull << ( r - fbk );
            else ++nb_sature;                            // un voisin dans une feuille non enregistree ( > 64 entrees )
        }
        nb_vois += SI( vois[ k ].size() );
        nb_feuilles += m.nentrees;
        SI mx = max_feuilles.load(); while ( m.nentrees > mx && ! max_feuilles.compare_exchange_weak( mx, m.nentrees ) ) {}
    } );
    std::vector<SI> beg( n + 1, 0 ), fbeg_c( n + 1, 0 ), fr_c( n + 1, 0 );
    for ( SI k = 0; k < n; ++k ) { beg[ k + 1 ] = beg[ k ] + SI( vois[ k ].size() ); fbeg_c[ k + 1 ] = fbeg_c[ k ] + SI( feuilles[ k ].size() ); fr_c[ k + 1 ] = fr_c[ k ] + SI( frontiere[ k ].size() ); }
    std::vector<d2::SI32> pre( beg[ n ] ), fb_all( fbeg_c[ n ] ), fr_all( fr_c[ n ] );
    std::vector<unsigned long long> fm_all( fbeg_c[ n ] );
    for ( SI k = 0; k < n; ++k ) {
        std::copy( vois[ k ].begin(), vois[ k ].end(), pre.begin() + beg[ k ] );
        std::copy( frontiere[ k ].begin(), frontiere[ k ].end(), fr_all.begin() + fr_c[ k ] );
        for ( SI q = 0; q < SI( feuilles[ k ].size() ); ++q ) { fb_all[ fbeg_c[ k ] + q ] = feuilles[ k ][ q ].first; fm_all[ fbeg_c[ k ] + q ] = feuilles[ k ][ q ].second; }
    }
    if ( ! w_perime.empty() ) pd.set_weights( nu.W, a.par );

    // ---- une passe MEMO parametree
    enum { TEMOIN, A, B_TESTE, B_FEUILLES, B_SANS_TEST, C_FRONT, SEULS };
    std::vector<std::vector<unsigned char>> saute( nth, std::vector<unsigned char>( n, 0 ) );
    std::vector<SI> prop_th( nth ), boites_th( nth ), coupees_th( nth );
    std::atomic<SI> deborde{ 0 };
    auto passe = [ & ]( int mode ) {
        res.assign( n, TF( 0 ) );
        for ( int t = 0; t < nth; ++t ) { prop_th[ t ] = 0; boites_th[ t ] = 0; coupees_th[ t ] = 0; }
        deborde = 0;
        parallel_for( n, a.par, [ & ]( SI k, int t ) {
            typename PD::Cell cel;
            typename PD::Memo m;
            const d2::SI32 *p = pre.data() + beg[ k ];
            const int np = int( beg[ k + 1 ] - beg[ k ] );
            unsigned char *sa = saute[ t ].data();
            if ( mode == A || mode == SEULS ) {
                for ( int q = 0; q < np; ++q ) sa[ p[ q ] ] = 1;
                m.pre = p; m.npre = np; m.saute = sa; m.parcours = mode != SEULS;
            } else if ( mode != TEMOIN ) {
                m.fbeg = fb_all.data() + fbeg_c[ k ]; m.fmask = fm_all.data() + fbeg_c[ k ]; m.nf = int( fbeg_c[ k + 1 ] - fbeg_c[ k ] );
                m.tester = mode == B_TESTE || mode == C_FRONT ? 1 : mode == B_FEUILLES ? 2 : 0;
                if ( mode == C_FRONT ) { m.front = fr_all.data() + fr_c[ k ]; m.nfront = int( fr_c[ k + 1 ] - fr_c[ k ] ); }
            }
            if ( ! pd.cellule_memo( k, cel, m ) ) ++deborde;
            else res[ pd.ids[ k ] ] = PD::mesure( cel );
            if ( mode == A || mode == SEULS ) for ( int q = 0; q < np; ++q ) sa[ p[ q ] ] = 0;
            prop_th[ t ] += m.prop; boites_th[ t ] += m.boites; coupees_th[ t ] += m.coupees;
        } );
        TF s = 0;
        for ( TF v : res ) s += v;
        SI prop = 0, boites = 0, coupees = 0;
        for ( int t = 0; t < nth; ++t ) { prop += prop_th[ t ]; boites += boites_th[ t ]; coupees += coupees_th[ t ]; }
        return std::tuple{ s, prop, boites, coupees };
    };
    auto chrono = [ & ]( int mode ) {
        passe( mode );                                   // chauffe
        double t = 1e300;
        for ( int r = 0; r < a.reps; ++r ) { const double t0 = now(); passe( mode ); t = std::min( t, now() - t0 ); }
        return t;
    };
    std::printf( "  %-24s n=%-7d %-8s  %.2f voisins, %.2f feuilles entrees ( au plus %d ), %.2f noeuds rejetes par cellule ( frontiere incomplete : %d )%s%s\n", nu.nom.c_str(), int( n ), nu.W ? "Laguerre" : "Voronoi",
                 double( nb_vois.load() ) / n, double( nb_feuilles.load() ) / n, int( max_feuilles.load() ), double( nb_rejets.load() ) / n, int( nb_front_sature.load() ),
                 nb_sature ? ( "  ( " + std::to_string( nb_sature.load() ) + " voisins hors des feuilles enregistrees )" ).c_str() : "",
                 w_perime.empty() ? "" : ( "  ( souvenirs pris a " + std::to_string( perime ) + " x W )" ).c_str() );
    std::printf( "     %-34s %8.3f s   somme %.9f\n", "sans memoire", t_sans, double( somme_sans ) );
    TF s_avec = somme_sans, s_seuls = somme_sans;
    for ( int mode : { TEMOIN, A, B_TESTE, B_FEUILLES, B_SANS_TEST, C_FRONT, SEULS } ) {
        const double t = chrono( mode );
        const auto [ sm, prop, boites, coupees ] = passe( mode );
        const char *nom = mode == TEMOIN ? "temoin ( MEMO a vide )" : mode == A ? "A. voisins par rang" : mode == B_TESTE ? "B. feuilles + bits, testees" : mode == B_FEUILLES ? "B. feuilles testees, pas les noeuds" : mode == C_FRONT ? "C. la frontiere, sans pile" : mode == B_SANS_TEST ? "B. feuilles + bits, sans test" : "les voisins seuls ( plancher )";
        std::printf( "     %-34s %8.3f s   somme %.9f   %6.2f plans proposes, %6.2f boites testees, %6.2f coupes effectives par cellule   ( %+.1f %% / sans )\n",
                     nom, t, double( sm ), double( prop ) / n, double( boites ) / n, double( coupees ) / n, 100 * ( t / t_sans - 1 ) );
        if ( mode == A || mode == B_TESTE || mode == B_FEUILLES || mode == B_SANS_TEST || mode == C_FRONT ) if ( std::fabs( sm - somme_sans ) > std::fabs( s_avec - somme_sans ) ) s_avec = sm;
        if ( mode == SEULS ) s_seuls = sm;
    }
    const bool ok = std::fabs( s_avec - somme_sans ) < 1e-9 && ( ! w_perime.empty() || std::fabs( s_seuls - somme_sans ) < 1e-9 ) && deborde == 0;
    if ( ! ok ) std::printf( "     <-- FAUX ( ecart %.2e / %.2e, %d debordements )\n", double( s_avec - somme_sans ), double( s_seuls - somme_sans ), int( deborde.load() ) );
    return ! ok;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    a.n = 100000;
    a.dims = 3;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( a.parse( s, i, argc, argv ) ) continue;
        if ( s == "--perime" && i + 1 < argc ) { perime = std::atof( argv[ ++i ] ); continue; }
        std::printf( "usage: memo [options]   ( 3D seulement )\n" );
        Args::usage();
        std::printf( "  --perime T      les souvenirs pris a T * W, le diagramme a W ( nuages a poids )\n" );
        return s == "--help" || s == "-h" ? 0 : 1;
    }
    a.finalise();
    std::printf( "=== 3D  threads=%d kernel=%s maxnv=%d leaf=%d reps=%d\n", a.par.threads, a.kernel.c_str(), a.nv( 3 ), int( a.leaf ), a.reps );
    int bad = 0;
    for ( const Nuage<3> &nu : a.nuages<3>() ) {
        if ( nu.absent ) { std::printf( "  %-28s : ABSENT ( --cases DIR )\n", nu.nom.c_str() ); continue; }
        bad += dispatch<3>( a, [ & ]( auto tag ) { return mesure<typename decltype( tag )::type>( a, nu ); } );
    }
    return bad ? 1 : 0;
}
