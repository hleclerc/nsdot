// =====================================================================================
// INTRINSEQUES CONTRE HIGHWAY -- meme algorithme, meme fournisseur, meme nuage, meme atelier.
//
// LES TROIS RESTENT SUR LA FORME `musttail`, exprès : ce banc compare des BIBLIOTHEQUES, pas des
// formes de controle. Le noyau principal du projet est desormais la machine a etats
// ( `Noyau2DEtats.h` ), et les portages highway et asimd devront la suivre -- d'ici la, les
// comparer a une reference qui a change ne mesurerait plus rien.
//
// La question est « un code portable peut-il tenir le meme temps ? », et elle ne se repond pas
// en lisant du code : les deux noyaux tournent ici cote a cote, sur la meme politique, et le
// banc compare AUSSI leurs cellules. C'est indispensable -- un noyau faux paraissait 15 % plus
// rapide, parce qu'une cellule fausse se vide tot et cesse de travailler.
//
// DEUX CIBLES, UNE SOURCE. `pd_asimd` se construit toujours -- asimd est rapatrie dans
// `ext/asimd`, il n'y a rien a installer. `pd_hwy` ajoute la colonne highway si `hwy/highway.h`
// est trouvable ( `apt install libhwy-dev` ).
// =====================================================================================
#include "supercell/Fournisseurs.h"
#include "supercell/Noyau2D.h"
#include "supercell/Noyau2DAsimd.h"
#ifdef AVEC_HIGHWAY
#include "supercell/Noyau2DHwy-inl.h"
#endif
#include "supercell/Noyau2DSca.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>
static double now() { using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count(); }
static constexpr int M = 64;
static inline unsigned etale( unsigned x ) {
    x &= 0xffffu; x = (x|(x<<8))&0x00ff00ffu; x = (x|(x<<4))&0x0f0f0f0fu;
    x = (x|(x<<2))&0x33333333u; x = (x|(x<<1))&0x55555555u; return x; }
static inline unsigned morton( float x, float y ) {
    return etale( (unsigned) std::min(65535.f,std::max(0.f,x*65535.f)) )
       | ( etale( (unsigned) std::min(65535.f,std::max(0.f,y*65535.f)) ) << 1 ); }

int main( int argc, char **argv ) {
    const int n = argc > 1 ? atoi( argv[ 1 ] ) : 50;
    const int rep = (int) std::max( 3.0, 4e7 / ( (double) n * ( n - 1 ) ) );
    std::mt19937 gen( 12345 ); std::uniform_real_distribution<float> u( 0, 1 );
    std::vector<float> px( n ), py( n );
    for ( int i = 0; i < n; ++i ) { px[i]=u(gen); py[i]=u(gen); }
    { std::vector<int> ord(n); std::iota(ord.begin(),ord.end(),0);
      std::vector<unsigned> cle(n);
      for ( int i=0;i<n;++i ) cle[i]=morton(px[i],py[i]);
      std::sort(ord.begin(),ord.end(),[&](int a,int b){return cle[a]<cle[b];});
      std::vector<float> qx(n),qy(n);
      for ( int i=0;i<n;++i ){ qx[i]=px[ord[i]]; qy[i]=py[ord[i]]; }
      px.swap(qx); py.swap(qy); }

    std::vector<int> nbv[4]; std::vector<int> idv[4];
    for ( int p=0;p<4;++p ){ nbv[p].assign(n,0); idv[p].assign((size_t)n*M,0); }
    double best[4] = {1e30,1e30,1e30,1e30}, aire[4]={0,0,0,0};
    const char *nom[4] = { "intrinseques  ", "highway       ", "asimd         ", "scalaire      " };
#ifdef AVEC_HIGHWAY
    constexpr bool avec_hwy = true;
#else
    constexpr bool avec_hwy = false;                     // la colonne 1 n'a pas tourne
#endif

    for ( int p = 0; p < 4; ++p ) for ( int r = 0; r < rep; ++r ) {
        double a = 0;
        const double t0 = now();
        for ( int i = 0; i < n; ++i ) {
            noyau2d::Atelier<M> at;
            noyau2d::AutourDeMoi f( px.data(), py.data(), n, i );
            if      ( p == 0 ) noyau2d::carre_unite( &f, &at );
#ifdef AVEC_HIGHWAY
            else if ( p == 1 ) noyau2d::HWY_NAMESPACE::carre_unite( &f, &at );
#else
            else if ( p == 1 ) continue;                  // highway absent : colonne vide
#endif
            else if ( p == 2 ) noyau2d::asimd2d::carre_unite( &f, &at );
            else               noyau2d::en_place<M>( &at, &f );
            nbv[p][i] = at.nb;
            if ( at.nb <= 0 ) continue;
            for ( int v=0; v<at.nb; ++v ) idv[p][(size_t)i*M+v] = at.cid[v];
            for ( int v=0; v<at.nb; ++v ) { const int w = v+1<at.nb?v+1:0;
                a += (double)at.vx[v]*at.vy[w] - (double)at.vx[w]*at.vy[v]; }
        }
        const double dt = now()-t0;
        if ( dt < best[p] ) { best[p]=dt; aire[p]=0.5*a; }
    }

    auto rot_eq = []( const int *a, const int *b, int m ) {
        for ( int r=0;r<m;++r ){ bool ok=true;
            for ( int v=0;v<m;++v ) if ( a[(v+r)%m]!=b[v] ) { ok=false; break; }
            if ( ok ) return true; }
        return m==0; };
    int faux_h = 0, faux_a = 0;
    for ( int i = 0; i < n; ++i ) {
        if ( nbv[0][i] != nbv[1][i] || ! rot_eq( &idv[0][(size_t)i*M], &idv[1][(size_t)i*M], nbv[0][i] ) ) ++faux_h;
        if ( nbv[0][i] != nbv[2][i] || ! rot_eq( &idv[0][(size_t)i*M], &idv[2][(size_t)i*M], nbv[0][i] ) ) ++faux_a;
    }

    const double coupes = (double) n * ( n - 1 );
    printf( "n = %d, nuage Morton, parcours en s'ecartant, %d passes\n", n, rep );
    for ( int p=0;p<4;++p ) {
        if ( p == 1 && ! avec_hwy ) continue;            // rien plutot qu'un chiffre faux
        printf( "  %s %7.3f ns/coupe   aire %.9f\n", nom[p], best[p]*1e9/coupes, aire[p] );
    }
    if ( avec_hwy )
        printf( "  rapport a l'intrinseque : highway x%.3f, asimd x%.3f\n", best[1]/best[0], best[2]/best[0] );
    else
        printf( "  rapport a l'intrinseque : asimd x%.3f   ( highway non compile : -DAVEC_HIGHWAY )\n",
                best[2]/best[0] );
    if ( avec_hwy ) printf( "  combinatoire differente : highway %d/%d, asimd %d/%d\n", faux_h, n, faux_a, n );
    else            printf( "  combinatoire differente : asimd %d/%d\n", faux_a, n );
    printf( "CSV n=%d intr=%.3f hwy=%.3f asimd=%.3f sca=%.3f rh=%.3f ra=%.3f fh=%d fa=%d\n",
            n, best[0]*1e9/coupes, avec_hwy ? best[1]*1e9/coupes : 0.0, best[2]*1e9/coupes,
            best[3]*1e9/coupes, avec_hwy ? best[1]/best[0] : 0.0, best[2]/best[0],
            avec_hwy ? faux_h : 0, faux_a );
    return 0;
}
