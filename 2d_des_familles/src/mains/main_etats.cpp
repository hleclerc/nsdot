// =====================================================================================
// `musttail` CONTRE MACHINE A ETATS -- meme algorithme, meme fournisseur, meme nuage.
//
// CE QUI EST EN JEU. `[[gnu::musttail]]` est la seule chose du noyau qui ne soit pas portable
// entre COMPILATEURS : gcc et clang l'ont, MSVC n'a rien d'equivalent. La machine a etats fait
// le meme travail avec une boucle et un `switch`, `etape<NB>` etant `always_inline` pour que la
// cellule reste dans ses registres. La question est ce que ca coute.
//
// LE BANC COMPARE AUSSI LES CELLULES, et pas seulement les temps : sans ca, un noyau faux
// paraitrait plus rapide -- une cellule fausse se vide tot et cesse de travailler.
// =====================================================================================
#include "supercell/Fournisseurs.h"
#include "supercell/Noyau2D.h"
#include "supercell/Noyau2DEtats.h"
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
    x &= 0xffffu; x=(x|(x<<8))&0x00ff00ffu; x=(x|(x<<4))&0x0f0f0f0fu;
    x=(x|(x<<2))&0x33333333u; x=(x|(x<<1))&0x55555555u; return x; }
static inline unsigned morton( float x, float y ) {
    return etale((unsigned)std::min(65535.f,std::max(0.f,x*65535.f)))
       | ( etale((unsigned)std::min(65535.f,std::max(0.f,y*65535.f)))<<1 ); }

int main( int argc, char **argv ) {
    const int n = argc>1 ? atoi(argv[1]) : 1000;
    const int rep = (int) std::max( 3.0, 4e7/((double)n*(n-1)) );
    std::mt19937 gen(12345); std::uniform_real_distribution<float> u(0,1);
    std::vector<float> px(n), py(n);
    for ( int i=0;i<n;++i ){ px[i]=u(gen); py[i]=u(gen); }
    { std::vector<int> ord(n); std::iota(ord.begin(),ord.end(),0);
      std::vector<unsigned> cle(n);
      for ( int i=0;i<n;++i ) cle[i]=morton(px[i],py[i]);
      std::sort(ord.begin(),ord.end(),[&](int a,int b){return cle[a]<cle[b];});
      std::vector<float> qx(n),qy(n);
      for ( int i=0;i<n;++i ){ qx[i]=px[ord[i]]; qy[i]=py[ord[i]]; }
      px.swap(qx); py.swap(qy); }

    std::vector<int> nbv[2]; std::vector<int> idv[2];
    for ( int p=0;p<2;++p ){ nbv[p].assign(n,0); idv[p].assign((size_t)n*M,0); }
    double best[2]={1e30,1e30}, aire[2]={0,0};
    const char *nom[2] = { "musttail      ", "machine a etats" };

    for ( int p=0;p<2;++p ) for ( int r=0;r<rep;++r ) {
        double a=0;
        const double t0=now();
        for ( int i=0;i<n;++i ) {
            noyau2d::Atelier<M> at;
            noyau2d::AutourDeMoi f( px.data(), py.data(), n, i );
            if ( p==0 ) noyau2d::carre_unite( &f, &at );
            else        noyau2d::etats::moteur( &f, &at );
            nbv[p][i]=at.nb;
            if ( at.nb<=0 ) continue;
            for ( int v=0;v<at.nb;++v ) idv[p][(size_t)i*M+v]=at.cid[v];
            for ( int v=0;v<at.nb;++v ){ const int w=v+1<at.nb?v+1:0;
                a += (double)at.vx[v]*at.vy[w] - (double)at.vx[w]*at.vy[v]; }
        }
        const double dt=now()-t0;
        if ( dt<best[p] ){ best[p]=dt; aire[p]=0.5*a; }
    }
    auto rot_eq = []( const int *a, const int *b, int m ){
        for ( int r=0;r<m;++r ){ bool ok=true;
            for ( int v=0;v<m;++v ) if ( a[(v+r)%m]!=b[v] ){ ok=false; break; }
            if ( ok ) return true; }
        return m==0; };
    int faux=0;
    for ( int i=0;i<n;++i )
        if ( nbv[0][i]!=nbv[1][i] || !rot_eq(&idv[0][(size_t)i*M],&idv[1][(size_t)i*M],nbv[0][i]) ) ++faux;

    const double coupes=(double)n*(n-1);
    for ( int p=0;p<2;++p )
        printf( "  %s %7.3f ns/coupe   aire %.9f\n", nom[p], best[p]*1e9/coupes, aire[p] );
    printf( "  etats / musttail : x%.3f   cellules differentes : %d / %d\n",
            best[1]/best[0], faux, n );
    printf( "CSV n=%d mt=%.3f et=%.3f ratio=%.3f faux=%d\n",
            n, best[0]*1e9/coupes, best[1]*1e9/coupes, best[1]/best[0], faux );
    return 0;
}
