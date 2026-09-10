// =====================================================================================
// UN BINAIRE, PLUSIEURS JEUX D'INSTRUCTIONS, CHOISIS A L'EXECUTION.
//
// C'est la reponse de highway au probleme des drapeaux : au lieu de compiler pour la machine
// qui compile ( `-march=native`, et le binaire ne bouge plus ), on compile le MEME code une
// fois par cible et on choisit au demarrage d'apres ce que le processeur annonce.
//
// = COMMENT CA MARCHE, EN TROIS PIECES
//
//  1. `foreach_target.h` REINCLUT CE FICHIER, une fois par cible retenue, en basculant
//     `HWY_TARGET_TOGGLE` a chaque tour. C'est pour ca que le noyau vectoriel est dans un
//     `-inl.h` a garde a bascule et non a `#pragma once` : une garde ordinaire ne laisserait
//     passer que le premier tour. A chaque passage `HWY_NAMESPACE` vaut autre chose --
//     `N_AVX3`, `N_AVX2`, `N_SSE4`... -- et `HWY_BEFORE_NAMESPACE()` pose les options du
//     compilateur pour cette cible-la. On obtient donc N copies de `CalculeImpl`, chacune dans
//     son espace de noms, chacune compilee avec ses instructions.
//
//  2. `HWY_EXPORT( CalculeImpl )` fabrique la TABLE : un pointeur par cible compilee.
//
//  3. `HWY_DYNAMIC_DISPATCH( CalculeImpl )( ... )` lit la table. Au premier appel highway
//     interroge le processeur ( `cpuid` ) et retient la meilleure cible disponible ; ensuite
//     c'est un appel indirect, un seul, hors de la boucle chaude.
//
// = CE QUE CA COUTE, MESURE ( n = 1000, nuage Morton, ns par coupe )
//
//     statique `-march=native`, intrinseques          2.59
//     statique `-march=native`, highway               2.61
//     dispatch, base `-march=native`                  2.67
//     dispatch, base AVX2                             2.94   <- +12 %
//     dispatch, base AVX2 + `-mtune=native`           2.62   <- l'ecart disparait
//
// LE MECANISME NE COUTE RIEN. Les 12 % n'etaient ni la table, ni l'appel indirect, ni une
// frontiere d'inlining -- j'ai cru au fournisseur laisse hors de la region par-cible et la
// mesure a dit non ( 2.939 contre 2.943 ). C'est le TUNING : `HWY_BEFORE_NAMESPACE()` emet un
// `#pragma GCC target`, qui active des INSTRUCTIONS et ne touche pas au modele d'ordonnancement.
// Avec une base AVX2 on herite de `-mtune=generic`, et c'est lui qui coute.
//
// CE QUI RESTE VRAI POUR UN BINAIRE VRAIMENT PORTABLE. `-mtune=native` n'a pas de sens dans un
// binaire qu'on distribue -- il regle l'ordonnancement pour la machine qui COMPILE. Le prix de
// la portabilite sur ce noyau est donc ces ~10 %, et il ne vient pas de l'abstraction SIMD mais
// de l'ordonnanceur. Highway ne dispatche pas le tuning ; si on y tient, il faut plusieurs
// binaires ou un `-mtune` choisi pour la cible dominante.
//
// CE QUE CA COUTE EN APPELS. Un appel indirect par APPEL DE HAUT NIVEAU -- pas par cellule, et surtout
// pas par coupe. C'est pourquoi le point d'entree disptache doit etre GROS : ici « calcule tout
// le nuage ». Un `HWY_DYNAMIC_DISPATCH` autour de la coupe elle-meme couterait plus que le
// vectoriel ne rapporte.
//
// LA CONTRAINTE QUE CA IMPOSE A LA CONCEPTION. La fonction exportee doit etre CONCRETE : on ne
// peut pas exporter `noyau<NB, Fourn, Atl>`, qui est un patron sur la politique. Le fournisseur
// doit donc etre choisi A L'INTERIEUR de la fonction dispatchee. Ce n'est pas une perte -- la
// politique reste un patron, elle est juste instanciee de l'autre cote de la porte -- mais ca
// dit ou passe la frontiere : le dispatch est au niveau du CALCUL, pas de l'operation.
// =====================================================================================

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// ---- IL FAUT ECARTER LES CIBLES TROP ETROITES, ET C'EST LA LIMITE DU PROCEDE.
//
// `foreach_target.h` compile pour TOUTES les cibles atteignables. Or le noyau exige HUIT voies
// de `float` : sur SSE2/SSSE3/SSE4, qui n'en ont que quatre, `FixedTag<float,8>` ne compile pas
// -- « static assertion failed: Too many lanes ». Highway ne decoupe pas une largeur demandee
// sur plusieurs registres ; il refuse.
//
// On les ecarte donc explicitement. Sur x86 c'est sans consequence -- il reste AVX2 et AVX3,
// c'est a dire tout ce qui date d'apres 2013. Sur ARM ce serait fatal : NEON fait 128 bits,
// donc quatre voies, et il ne resterait rien. C'est exactement ce que `asimd` sait faire et pas
// highway : `SimdVec<float,8>` s'y decoupe en deux registres de quatre.
// `HWY_SCALAR` et `HWY_EMU128` sont les filets de securite de highway -- une voie, ou une
// emulation 128 bits. Ils ne savent pas non plus faire huit voies, donc ils tombent aussi.
//
// ET LA BASE DE COMPILATION DOIT SUIVRE. Highway exige qu'au moins une cible de BASE reste
// active, et la base se deduit des drapeaux : sans rien, sur x86-64, c'est SSE2 -- qu'on vient
// d'interdire, d'ou « At least one baseline target must be defined and enabled ». Il faut donc
// compiler ce fichier avec une base a huit voies :
//
//     -mavx2 -mfma -mbmi2 -mf16c -maes -mpclmul
//
// ( `-march=haswell` seul ne suffit pas : gcc n'y active pas `__AES__`, et la cible SSE4 de
//   highway l'exige, donc AVX2 qui en depend retombe silencieusement sur SSSE3. Piege de
//   drapeaux, du meme genre que celui qu'on reproche a xsimd. )
//
// Le binaire exige alors AVX2 au minimum -- 2013 et apres. C'est le prix du choix « huit voies
// toujours », et il est explicite ici plutot que subi.
#define HWY_DISABLED_TARGETS ( HWY_SSE2 | HWY_SSSE3 | HWY_SSE4 | HWY_SCALAR | HWY_EMU128 )

// ---- LE MECANISME. `HWY_TARGET_INCLUDE` nomme CE fichier, `foreach_target.h` le reinclut.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mains/main_dyn.cpp"
#include "hwy/foreach_target.h"                          // IWYU pragma: keep

#include "hwy/highway.h"
#include "supercell/Fournisseurs.h"
#include "supercell/Noyau2DHwy-inl.h"

// ============ CE QUI EST RECOMPILE UNE FOIS PAR CIBLE ============
HWY_BEFORE_NAMESPACE();
namespace app {
namespace HWY_NAMESPACE {

namespace hk = noyau2d::HWY_NAMESPACE;

/// LE FOURNISSEUR, RECOPIE DANS LA REGION PAR-CIBLE. On pouvait croire que le laisser a la
/// base ( `Fournisseurs.h` est inclus avant `HWY_BEFORE_NAMESPACE()` ) empechait son inlining
/// dans un noyau AVX3, et expliquait l'ecart avec le statique. MESURE : 2.939 ns de ce cote,
/// 2.943 de l'autre. Aucun effet. L'ecart vient d'ailleurs -- voir le commentaire d'en-tete.
struct AutourDeMoiCible {
    const float *px, *py;
    float x0, y0;
    int   lo, hi, n;
    bool  cote;
    AutourDeMoiCible( const float *px, const float *py, int n, int moi )
        : px( px ), py( py ), x0( px[ moi ] ), y0( py[ moi ] ),
          lo( moi - 1 ), hi( moi + 1 ), n( n ), cote( true ) {}
    template<class Etat>
    bool suivant( const Etat &, noyau2d::Plan &p ) {
        int j;
        if      ( cote && hi < n ) { cote = false; j = hi++; }
        else if ( lo >= 0 )        { cote = true;  j = lo--; }
        else if ( hi < n )         {               j = hi++; }
        else return false;
        p.dx = px[ j ] - x0; p.dy = py[ j ] - y0;
        p.off = 0.5f * ( p.dx * ( px[ j ] + x0 ) + p.dy * ( py[ j ] + y0 ) );
        p.id = j;
        return true;
    }
};

/// calcule TOUT le nuage. Gros exprès : c'est le grain du dispatch.
void CalculeImpl( const float *px, const float *py, int n, int *nbs, double *aire ) {
    double a = 0;
    for ( int i = 0; i < n; ++i ) {
        noyau2d::Atelier<64> at;
        AutourDeMoiCible f( px, py, n, i );               // la politique, DU BON COTE
        hk::carre_unite( &f, &at );
        nbs[ i ] = at.nb;
        if ( at.nb <= 0 ) continue;
        for ( int v = 0; v < at.nb; ++v ) {
            const int w = v + 1 < at.nb ? v + 1 : 0;
            a += (double) at.vx[ v ] * at.vy[ w ] - (double) at.vx[ w ] * at.vy[ v ];
        }
    }
    *aire = 0.5 * a;
}

} // namespace HWY_NAMESPACE
} // namespace app
HWY_AFTER_NAMESPACE();

// ============ CE QUI N'EST COMPILE QU'UNE FOIS ============
#if HWY_ONCE

namespace app {

HWY_EXPORT( CalculeImpl );                                // la table : un pointeur par cible

void Calcule( const float *px, const float *py, int n, int *nbs, double *aire ) {
    HWY_DYNAMIC_DISPATCH( CalculeImpl )( px, py, n, nbs, aire );
}

} // namespace app

static double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}
static inline unsigned etale( unsigned x ) {
    x &= 0xffffu; x = ( x | ( x << 8 ) ) & 0x00ff00ffu; x = ( x | ( x << 4 ) ) & 0x0f0f0f0fu;
    x = ( x | ( x << 2 ) ) & 0x33333333u; x = ( x | ( x << 1 ) ) & 0x55555555u; return x; }
static inline unsigned morton( float x, float y ) {
    return etale( (unsigned) std::min( 65535.f, std::max( 0.f, x * 65535.f ) ) )
       | ( etale( (unsigned) std::min( 65535.f, std::max( 0.f, y * 65535.f ) ) ) << 1 ); }

int main( int argc, char **argv ) {
    const int n = argc > 1 ? atoi( argv[ 1 ] ) : 1000;

    // `SupportedAndGeneratedTargets` rend les cibles a la fois COMPILEES ici et SUPPORTEES par
    // ce processeur, la meilleure d'abord -- c'est celle que le dispatch retiendra.
    printf( "cibles compilees dans CE binaire et utilisables ici :\n" );
    bool premiere = true;
    for ( int64_t t : hwy::SupportedAndGeneratedTargets() ) {
        printf( "  %-12s %s\n", hwy::TargetName( t ), premiere ? "<- retenue pour ce processeur" : "" );
        premiere = false;
    }

    std::mt19937 gen( 12345 );
    std::uniform_real_distribution<float> u( 0, 1 );
    std::vector<float> px( n ), py( n );
    for ( int i = 0; i < n; ++i ) { px[ i ] = u( gen ); py[ i ] = u( gen ); }
    { std::vector<int> ord( n ); std::iota( ord.begin(), ord.end(), 0 );
      std::vector<unsigned> cle( n );
      for ( int i = 0; i < n; ++i ) cle[ i ] = morton( px[ i ], py[ i ] );
      std::sort( ord.begin(), ord.end(), [ & ]( int a, int b ) { return cle[ a ] < cle[ b ]; } );
      std::vector<float> qx( n ), qy( n );
      for ( int i = 0; i < n; ++i ) { qx[ i ] = px[ ord[ i ] ]; qy[ i ] = py[ ord[ i ] ]; }
      px.swap( qx ); py.swap( qy ); }

    std::vector<int> nbs( n ); double aire = 0, best = 1e30;
    const int rep = (int) std::max( 3.0, 4e7 / ( (double) n * ( n - 1 ) ) );
    for ( int r = 0; r < rep; ++r ) {
        const double t0 = now();
        app::Calcule( px.data(), py.data(), n, nbs.data(), &aire );
        best = std::min( best, now() - t0 );
    }
    printf( "\nn = %d, %d passes : %.3f ns par coupe proposee, somme des aires %.9f\n",
            n, rep, best * 1e9 / ( (double) n * ( n - 1 ) ), aire );
    return 0;
}

#endif // HWY_ONCE
