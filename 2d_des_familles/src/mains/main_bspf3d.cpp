// =====================================================================================
// LE DIAGRAMME 3D COMPLET : `AaBsp3` + la cellule qui dirige.
//
// Le pendant 3D de `pd_bspf`. On mesure trois choses, et il faut les trois pour conclure :
//
//   1. LE TEMPS PAR GERME, arbre compris, qui est le seul nombre comparable a CGAL.
//   2. LE NOMBRE DE CANDIDATS PROPOSES par cellule, hors chronometre, qui dit ce que l'elagage
//      rend vraiment. Une cellule de Voronoi 3D a ~15,5 faces ; tout ce qui depasse est du dechet.
//   3. LA MEME CHOSE SANS ELAGAGE ( `AutourDeMoi3`, tous contre tous ) au meme `n`, pour que le
//      gain de la structure soit une mesure et non une impression.
//
// = LES DEUX TEMOINS D'EXACTITUDE
//
//   * LA SOMME DES VOLUMES DOIT VALOIR 1. C'est le seul controle qui teste la COMPLETUDE : si un
//     elagage est trop agressif, une cellule deborde sur sa voisine et la somme monte.
//   * LE VOISINAGE, cellule par cellule, contre le parcours SANS elagage. Une coupe ratee qui ne
//     changerait pas assez le volume pour se voir y apparait tout de suite. En O( n^2 ), donc
//     seulement en dessous de `--verif-max`.
// =====================================================================================

#include "spatial_accel/AaBsp.h"
#include "supercell/FournisseurBsp3.h"
#include "supercell/Noyau3D.h"
#include "util/common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace pd;

static double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

#ifndef MAXNV_BANC
#define MAXNV_BANC 128
#endif
static constexpr int MAXNV = MAXNV_BANC;
#ifndef LARGEUR_BANC
#define LARGEUR_BANC 8
#endif
using Cel3 = noyau3d::Cellule3<MAXNV,LARGEUR_BANC>;

/// COMPTER LES CANDIDATS SANS TOUCHER AU FOURNISSEUR : un enrobage qui transmet `suivant` et
/// incremente. Hors chronometre -- un compteur dans la boucle chaude fausserait ce qu'il mesure.
template<class F>
struct Compte3 {
    F         f;
    long long *n;
    /// le trait, et non `F::Local` : `AutourDeMoi3` n'en declare pas, et `noyau3d::Local` rend
    /// alors `RienDeLocal` -- exactement ce que son `suivant` attend.
    using Local = noyau3d::Local<F>;
    template<class Etat>
    bool suivant( const Etat &e, Local &l, noyau3d::Plan3 &p ) {
        if ( ! f.suivant( e, l, p ) ) return false;
        ++*n;
        return true;
    }
};

using Arbre = pd::AaBspT<3>;

/// CE QUE LA MESURE REND. `POIDS` est une constante de compilation -- le cas euclidien ne doit
/// payer ni les pentes du majorant ni les termes en `a . y` --, donc tout ce qui en depend vit
/// dans un patron, appele une fois. C'est la meme construction que `LANCE` dans `main_n503d.cpp`.
struct Bilan {
    int       pire_nv = 0, deborde = 0, faux_vois = 0, nok = 0, faux_adj = 0, pire_transit = 0;
    long long eff = 0;
    long long som_nv = 0, som_vois = 0, prop = 0, prop_brut = 0;
    double    t_diag = 1e30, vol = 0, t_brut = 1e30, vol_brut = 0;
};

template<bool POIDS>
static void deroule( const Arbre &arbre, const std::vector<float> &px, const std::vector<float> &py,
                     const std::vector<float> &pz, const std::vector<float> &pw,
                     const std::vector<int> &ids, int n, int rep, bool avec_brut, Bilan &R ) {
    using Four = noyau3d::FournisseurBsp3<Arbre,POIDS,LARGEUR_BANC>;
    using Brut = noyau3d::AutourDeMoi3<POIDS>;

    // `k` est le rang dans l'arbre, `ids[ k ]` l'identifiant global : c'est LUI qu'il ne faut pas
    // se proposer a soi-meme, puisque c'est lui que le fournisseur rend dans `p.id`.
    auto fab = [ & ]( int k ) { return Four( &arbre, px[k], py[k], pz[k], pw[k], ids[k] ); };
    auto fab_brut = [ & ]( int k ) {
        return Brut( px.data(), py.data(), pz.data(), pw.data(), n, k );
    };

    // ---- LE DIAGRAMME, chronometre.
    for ( int r = 0; r < rep; ++r ) {
        double v = 0;
        const double t0 = now();
        Cel3 c;
        for ( int k = 0; k < n; ++k ) {
            auto f = fab( k );
            if ( noyau3d::moteur( &f, &c ) ) continue;
            v += c.volume();
        }
        const double dt = now() - t0;
        if ( dt < R.t_diag ) { R.t_diag = dt; R.vol = v; }
    }

    // ---- HORS CHRONOMETRE : les tailles, les candidats, l'invariant d'adjacence.
    {
        Cel3 c;
        for ( int k = 0; k < n; ++k ) {
            Compte3<Four> f{ fab( k ), &R.prop };
            if ( noyau3d::moteur( &f, &c ) ) { ++R.deborde; continue; }
            R.pire_nv = std::max( R.pire_nv, c.nv );
            R.som_nv += c.nv;
            int vo[ MAXNV ];
            R.som_vois += c.voisins( vo, MAXNV );
            R.faux_adj += c.verifie();
            R.pire_transit = std::max( R.pire_transit, c.nv );
        }
        R.nok = n - R.deborde;
#if NOYAU3D_COMPTE
        R.eff = c.nb_eff;                                // les coupes qui ONT eu un effet
#endif
    }

    if ( ! avec_brut )
        return;

    // ---- LE TEMOIN : le meme nuage SANS elagage. O( n^2 ), donc borne par `--verif-max`.
    for ( int r = 0; r < rep; ++r ) {
        double v = 0;
        const double t0 = now();
        Cel3 c;
        for ( int k = 0; k < n; ++k ) {
            auto f = fab_brut( k );
            if ( noyau3d::moteur( &f, &c ) ) continue;
            v += c.volume();
        }
        const double dt = now() - t0;
        if ( dt < R.t_brut ) { R.t_brut = dt; R.vol_brut = v; }
    }
    {
        Cel3 a, b;
        std::vector<int> va, vb;
        for ( int k = 0; k < n; ++k ) {
            auto fa = fab( k );
            if ( noyau3d::moteur( &fa, &a ) ) continue;
            Compte3<Brut> fb{ fab_brut( k ), &R.prop_brut };
            if ( noyau3d::moteur( &fb, &b ) ) continue;
            int oa[ MAXNV ], ob[ MAXNV ];
            const int ma = a.voisins( oa, MAXNV ), mb = b.voisins( ob, MAXNV );
            va.assign( oa, oa + ma );
            vb.clear();                                  // le brut rend des RANGS, l'arbre des ids
            for ( int t = 0; t < mb; ++t ) vb.push_back( ob[t] < 0 ? ob[t] : ids[ ob[t] ] );
            std::sort( va.begin(), va.end() ); std::sort( vb.begin(), vb.end() );
            if ( va != vb ) ++R.faux_vois;
        }
    }
}

int main( int argc, char **argv ) {
    int n = 100000, rep = 3, leaf = 10, verif_max = 3000;
    unsigned graine = 12345;
    double frac = 0;
    for ( int i = 1; i + 1 < argc; i += 2 ) {
        const std::string o = argv[ i ];
        if      ( o == "--n"         ) n         = atoi( argv[ i + 1 ] );
        else if ( o == "--rep"       ) rep       = atoi( argv[ i + 1 ] );
        else if ( o == "--leaf"      ) leaf      = atoi( argv[ i + 1 ] );
        else if ( o == "--graine"    ) graine    = atoi( argv[ i + 1 ] );
        else if ( o == "--poids"     ) frac      = atof( argv[ i + 1 ] );
        else if ( o == "--verif-max" ) verif_max = atoi( argv[ i + 1 ] );
        else { fprintf( stderr, "option inconnue %s\n", argv[ i ] ); return 1; }
    }

    // LE MEME NUAGE QUE `pd_n503d` : meme graine, meme loi, meme formule de poids. Le decalage
    // d'un plan est `dw / ( 2 |p1 - p0| )`, donc pour qu'il soit une fraction de l'espacement
    // `h ~ n^(-1/3)` il faut `dw ~ h^2`.
    std::mt19937 gen( graine );
    std::uniform_real_distribution<float> u( 0, 1 );
    std::vector<double> ax( n ), ay( n), az( n ), aw( n, 0.0 );
    const double h2 = std::pow( double( n ), -2.0 / 3.0 );
    for ( int i = 0; i < n; ++i ) {
        ax[i] = u( gen ); ay[i] = u( gen ); az[i] = u( gen );
        aw[i] = frac * h2 * u( gen );
    }

    // ---- 1. L'ARBRE. Chronometre a part : il se construit une fois pour tout le diagramme.
    double t_arbre = 1e30;
    Arbre arbre;
    for ( int r = 0; r < rep; ++r ) {
        const double t0 = now();
        Arbre a;
        a.build( ax.data(), ay.data(), az.data(), frac > 0 ? aw.data() : nullptr, n, leaf );
        const double dt = now() - t0;
        if ( dt < t_arbre ) { t_arbre = dt; arbre = std::move( a ); }
    }

    // LES GERMES DANS L'ORDRE DE L'ARBRE, en `float`. On parcourt dans cet ordre-la : c'est celui
    // qui donne la localite, et `ids[ k ]` ramene a l'identifiant global.
    std::vector<float> px( n ), py( n ), pz( n ), pw( n, 0.f );
    std::vector<int>   ids( n );
    for ( int k = 0; k < n; ++k ) {
        px[k] = (float) arbre.seed_c( k, 0 );
        py[k] = (float) arbre.seed_c( k, 1 );
        pz[k] = (float) arbre.seed_c( k, 2 );
        pw[k] = (float) arbre.seed_w( k );
        ids[k] = (int) arbre.order[ k ];
    }

    const bool avec_brut = n <= verif_max;
    Bilan R;
    if ( frac > 0 ) deroule<true >( arbre, px, py, pz, pw, ids, n, rep, avec_brut, R );
    else            deroule<false>( arbre, px, py, pz, pw, ids, n, rep, avec_brut, R );
    if ( R.faux_adj ) printf( "  !! invariant d'adjacence viole %d fois\n", R.faux_adj );

    const double t_tot = t_arbre + R.t_diag;
    printf( "%d germes en 3D, %s -- AaBsp3 ( feuilles de %d ), largeur SIMD %d\n",
            n, frac > 0 ? "Laguerre" : "Voronoi", leaf, LARGEUR_BANC );
    printf( "  arbre              : %8.4f s   ( %7.1f ns/germe )\n", t_arbre, t_arbre * 1e9 / n );
    printf( "  diagramme          : %8.4f s   ( %7.1f ns/germe )\n", R.t_diag,  R.t_diag  * 1e9 / n );
    printf( "  TOTAL              : %8.4f s   ( %7.1f ns/germe )\n", t_tot,   t_tot   * 1e9 / n );
    if ( avec_brut )
        printf( "  sans elagage       : %8.4f s   ( %7.1f ns/germe )   x%.1f\n",
                R.t_brut, R.t_brut * 1e9 / n, R.t_brut / t_tot );
    printf( "\n  volume total %.9f   <- doit valoir 1\n", R.vol );
    if ( avec_brut )
        printf( "  voisinages differents du parcours sans elagage : %d / %d\n", R.faux_vois, R.nok );
    printf( "  debordements ( MaxNv = %d ) : %d\n", MAXNV, R.deborde );
#if BSP3_COMPTE
    printf( "\n  DIAGNOSTIC : %.1f noeuds depiles par cellule, %.1f feuilles ouvertes,\n"
            "               %.1f sommets lus par le test d'elagage et par cellule\n",
            double( noyau3d::g_visites ) / R.nok / ( rep + 2 ),
            double( noyau3d::g_feuilles ) / R.nok / ( rep + 2 ),
            double( noyau3d::g_sommets ) / R.nok / ( rep + 2 ) );
#endif
    printf( "\n  candidats proposes par cellule : %.2f   ( faces vivantes : %.2f )\n",
            double( R.prop ) / R.nok, double( R.som_vois ) / R.nok );
    if ( avec_brut )
        printf( "  sans elagage                   : %.2f\n", double( R.prop_brut ) / R.nok );
#if NOYAU3D_COMPTE
    printf( "  coupes EFFECTIVES par cellule : %.2f  sur %.2f proposees\n",
            double( R.eff ) / R.nok, double( R.prop ) / R.nok );
#endif
    printf( "  sommets moyenne %.2f   pire %d\n", double( R.som_nv ) / R.nok, R.pire_nv );
    printf( "\nCSVB3 n=%d poids=%.1f leaf=%d arbre=%.1f diag=%.1f tot=%.1f vol=%.9f faux=%d prop=%.2f vois=%.2f\n",
            n, frac, leaf, t_arbre * 1e9 / n, R.t_diag * 1e9 / n, t_tot * 1e9 / n, R.vol, R.faux_vois,
            double( R.prop ) / R.nok, double( R.som_vois ) / R.nok );
    return 0;
}
