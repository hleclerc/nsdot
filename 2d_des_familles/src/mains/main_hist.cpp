// =====================================================================================
// COMBIEN DE SOMMETS UNE COUPE EFFECTIVE RETRANCHE-T-ELLE ?
//
// Le noyau traite l'exterieur comme une PLAGE CYCLIQUE CONTIGUE : deux `ctz` donnent ses deux
// bouts, une seule division fabrique les deux nouveaux sommets, et la cellule est reassemblee par
// un `vpermps`. Le cout de cette sequence NE DEPEND PAS du nombre de sommets retranches -- retirer
// un sommet coute ce que coutent en retirer six.
//
// Ce banc mesure donc ce que ce choix vaut : la loi de `nb_out` sur les coupes effectives, et son
// conditionnement par la taille de la cellule au moment de la coupe. Un histogramme concentre sur
// `nb_out = 1` dit que la plage contigue ne sert presque jamais et qu'un chemin special a un
// sommet serait tentant ; une queue epaisse dit le contraire.
//
// UNE COUPE EFFECTIVE, ICI, EST UNE COUPE QUI BOUGE LA CELLULE. Les coupes sans effet -- la grande
// majorite des tentatives -- ne sont pas comptees : elles n'ont pas de `nb_out` a montrer, et leur
// nombre est deja mesure ailleurs ( `pd_sc1`, `pd_bspf` ).
//
// LE COMPTAGE PASSE PAR `Fourn::compte`, detecte sur le fournisseur comme `Local` et
// `veut_changement` : un fournisseur qui ne la declare pas ne paie rien, et le noyau n'a ni
// pointeur ni drapeau a porter.
//
// LES CAS SONT CEUX DE LA SUITE 2D, et c'est le point : le nuage uniforme est le seul ou une
// cellule est toujours autour de son germe. Les nuages en LIGNES ont des cellules en lame, et rien
// ne dit a priori qu'elles se coupent de la meme facon.
// =====================================================================================

#include "bench/Bench.h"
#include "spatial_accel/AaBsp.h"
#include "supercell/FournisseurBsp.h"
#include "supercell/Noyau2DEtats.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace pd;
using namespace pd::bench;

static constexpr int M = 64;                             ///< sommets max, et taille des histogrammes

/// LE COMPTEUR, en enveloppe autour d'un fournisseur quelconque. Il ne fait qu'ajouter `compte` :
/// `suivant` est transmis tel quel, et `Local` est celui du fournisseur enveloppe.
template<class F>
struct Compteur {
    using Local = noyau2d::Local<F>;
    F f;
    long long ( *h )[ M + 2 ];                           ///< `h[ nb avant ][ nb_out ]`
    long long *prop;                                     ///< coupes PROPOSEES, effectives ou non

    template<class Etat>
    bool suivant( const Etat &e, Local &l, noyau2d::Plan &p ) {
        const bool r = f.suivant( e, l, p );
        *prop += r;
        return r;
    }

    void compte( int nb_out, int nb ) {
        h[ nb < M + 1 ? nb : M + 1 ][ nb_out < M + 1 ? nb_out : M + 1 ] += 1;
    }
};

static void affiche( const std::string &nom, long long ( *h )[ M + 2 ], long long nc, long long prop ) {
    long long tot = 0, marg[ M + 2 ] = {}, parnb[ M + 2 ] = {};
    for ( int nb = 0; nb <= M + 1; ++nb )
        for ( int o = 0; o <= M + 1; ++o ) {
            tot += h[ nb ][ o ]; marg[ o ] += h[ nb ][ o ]; parnb[ nb ] += h[ nb ][ o ];
        }
    if ( tot == 0 ) { printf( "%s : aucune coupe\n", nom.c_str() ); return; }

    printf( "\n%s\n", nom.c_str() );
    printf( "  %lld coupes proposees ( %.2f par cellule ), %lld effectives ( %.2f ), soit %.1f %%\n",
            prop, double( prop ) / nc, tot, double( tot ) / nc, 100.0 * tot / prop );

    // ---- la loi de `nb_out`, toutes tailles confondues
    printf( "  nb_out |" );
    for ( int o = 1; o <= 8; ++o ) printf( " %7d", o );
    printf( "  %6s\n", ">8" );
    printf( "       %% |" );
    double cum = 0;
    for ( int o = 1; o <= 8; ++o ) { const double q = 100.0 * marg[ o ] / tot; cum += q;
                                     printf( " %7.3f", q ); }
    {   long long q = 0;
        for ( int o = 9; o <= M + 1; ++o ) q += marg[ o ];
        printf( "  %6.3f\n", 100.0 * q / tot ); }
    printf( "     cum |" );
    cum = 0;
    for ( int o = 1; o <= 8; ++o ) { cum += 100.0 * marg[ o ] / tot; printf( " %7.3f", cum ); }
    printf( "\n" );
    double moy = 0;
    for ( int o = 1; o <= M + 1; ++o ) moy += double( o ) * marg[ o ];
    printf( "  moyenne %.3f sommets retranches par coupe effective\n", moy / tot );

    // ---- conditionne par la taille AVANT la coupe : c'est la que le choix du noyau se juge
    printf( "  conditionne par la taille avant coupe ( %% de la ligne ) :\n" );
    printf( "    nb avant |" );
    for ( int o = 1; o <= 7; ++o ) printf( " %6d", o );
    printf( " %6s |  part des coupes\n", "8+" );
    for ( int nb = 3; nb <= 9; ++nb ) {
        const long long l = nb <= 8 ? parnb[ nb ] : ( [ & ] { long long q = 0;
            for ( int k = 9; k <= M + 1; ++k ) q += parnb[ k ]; return q; }() );
        if ( l == 0 ) continue;
        printf( "    %8s |", nb <= 8 ? std::to_string( nb ).c_str() : ">8" );
        for ( int o = 1; o <= 7; ++o ) {
            long long c = 0;
            if ( nb <= 8 ) c = h[ nb ][ o ];
            else for ( int k = 9; k <= M + 1; ++k ) c += h[ k ][ o ];
            printf( " %6.2f", 100.0 * c / l );
        }
        long long c8 = 0;
        for ( int o = 8; o <= M + 1; ++o ) {
            if ( nb <= 8 ) c8 += h[ nb ][ o ];
            else for ( int k = 9; k <= M + 1; ++k ) c8 += h[ k ][ o ];
        }
        printf( " %6.2f |  %11.3f %%\n", 100.0 * c8 / l, 100.0 * l / tot );
    }
}

template<bool POIDS>
static void un_cas( const Cloud<2> &cl ) {
    AaBspT<2> arbre;
    arbre.build( cl.P[ 0 ], cl.P[ 1 ], POIDS ? cl.W : nullptr, cl.n, 10 );

    const SI n = cl.n;
    std::vector<float> px( n ), py( n ), pw( n ); std::vector<int> ids( n );
    for ( SI k = 0; k < n; ++k ) {
        px[ k ] = (float) arbre.seed_x( k ); py[ k ] = (float) arbre.seed_y( k );
        pw[ k ] = (float) arbre.seed_w( k ); ids[ k ] = (int) arbre.order[ k ];
    }

    static long long h[ M + 2 ][ M + 2 ];
    for ( int a = 0; a <= M + 1; ++a ) for ( int b = 0; b <= M + 1; ++b ) h[ a ][ b ] = 0;

    double aire = 0; long long vides = 0, prop = 0;
    for ( SI k = 0; k < n; ++k ) {
        noyau2d::Atelier<M> at;
        using F = noyau2d::FournisseurBsp<AaBspT<2>,POIDS>;
        Compteur<F> f{ F( &arbre, px[ k ], py[ k ], pw[ k ], ids[ k ] ), h, &prop };
        noyau2d::etats::moteur( &f, &at );
        if ( at.nb <= 0 ) { ++vides; continue; }
        for ( int v = 0; v < at.nb; ++v ) { const int w = v + 1 < at.nb ? v + 1 : 0;
            aire += (double) at.vx[v] * at.vy[w] - (double) at.vx[w] * at.vy[v]; }
    }

    char t[ 256 ];
    snprintf( t, sizeof t, "%s  ( %s, n = %lld, aire %.9f, %lld cellules vides )",
              cl.nom.c_str(), POIDS ? "Laguerre" : "Voronoi", (long long) n, 0.5 * aire,
              (long long) vides );
    affiche( t, h, n - vides, prop );
}

int main( int argc, char **argv ) {
    Args a; a.n = 200000; a.dims = 2;
    for ( int i = 1; i < argc; ++i ) {
        std::string s = argv[ i ];
        if ( ! parse_commun( a, s, i, argc, argv ) ) { usage_commun(); return 1; }
    }
    finalise( a );

    for ( const Cloud<2> &cl : suite_2d( a ) ) {
        if ( cl.absent ) { printf( "\n%s : fichier absent, saute\n", cl.nom.c_str() ); continue; }
        if ( cl.W ) un_cas<true >( cl );
        else        un_cas<false>( cl );
    }
    return 0;
}
