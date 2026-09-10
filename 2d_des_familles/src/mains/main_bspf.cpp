// =====================================================================================
// LE FOURNISSEUR BSP, EN LAGUERRE -- verifie contre un balayage complet.
//
// Meme noyau, memes cellules, seul l'ELAGAGE differe : si le majorant affine des poids se trompe
// d'un signe, les cellules divergent. C'est la seule facon de tester un elagage -- le comparer a
// l'absence d'elagage.
//
// LA TROISIEME PASSE EST LA POUR CALIBRER LE DESACCORD. Le balayage complet est aussi rejoue dans
// l'ORDRE INVERSE : meme algorithme, meme resultat mathematique, seul l'ordre des coupes change.
// Il diverge de lui-meme sur le meme nombre de cellules que le BSP -- 2 sur 20000 -- donc ce
// desaccord est le plancher du `float`, pas un defaut de l'elagage. Sans cette passe on lirait
// « 2 cellules fausses » et on chercherait un bug qui n'existe pas.
// =====================================================================================
#include "util/common.h"
#include "supercell/FournisseurBsp.h"
#include "supercell/Noyau2DEtats.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
using namespace pd;
static double now() { using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count(); }
static constexpr int M = 64;

/// tous les germes, aucun elagage : la reference.
struct Balayage {
    int sens = 1;                                        ///< +1 = ordre direct, -1 = inverse
    const float *px, *py, *pw;
    float x0, y0, w0;
    int n, k, i0;
    const int *ids;
    template<class Etat> bool suivant( const Etat &, noyau2d::RienDeLocal &, noyau2d::Plan &p ) {
        for ( ;; ) {
            if ( k >= n ) return false;
            const int j = sens > 0 ? k++ : n - 1 - k++;
            if ( ids[ j ] == i0 ) continue;
            const float xj = px[ j ], yj = py[ j ];
            p.dx = xj - x0; p.dy = yj - y0;
            p.off = 0.5f * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) ) + 0.5f * ( w0 - pw[ j ] );
            p.id = ids[ j ];
            return true;
        }
    }
};

int main( int argc, char **argv ) {
    const int n = argc>1 ? atoi(argv[1]) : 20000;
    const double frac = argc>2 ? atof(argv[2]) : 1.0;    // poids en fraction de h^2
    const int leaf = 10;

    std::mt19937 gen( 12345 );
    std::uniform_real_distribution<double> u( 0, 1 );
    std::vector<double> ax(n), ay(n), aw(n);
    const double h2 = 1.0 / n;                           // h^2 ~ aire par germe
    for ( int i=0;i<n;++i ){ ax[i]=u(gen); ay[i]=u(gen); aw[i]= frac * h2 * u(gen); }

    AaBspT<2> arbre;
    arbre.build( ax.data(), ay.data(), aw.data(), n, leaf );

    std::vector<float> px(n), py(n), pw(n); std::vector<int> ids(n);
    for ( int k=0;k<n;++k ){ px[k]=(float)arbre.seed_x(k); py[k]=(float)arbre.seed_y(k);
                             pw[k]=(float)arbre.seed_w(k); ids[k]=(int)arbre.order[k]; }

    std::vector<int> nb_b(n), nb_a(n); std::vector<int> id_b((size_t)n*M), id_a((size_t)n*M);
    double aire_b=0, aire_a=0, t_b=1e30, t_a=1e30;
    long long cand_a=0;

    for ( int r=0;r<3;++r ){ double a=0; const double t0=now();
        for ( int k=0;k<n;++k ){
            noyau2d::Atelier<M> at;
            Balayage f{ 1, px.data(), py.data(), pw.data(), px[k], py[k], pw[k], n, 0, ids[k], ids.data() };
            noyau2d::etats::moteur( &f, &at );
            nb_b[k]=at.nb; if ( at.nb<=0 ) continue;
            for ( int v=0;v<at.nb;++v ) id_b[(size_t)k*M+v]=at.cid[v];
            for ( int v=0;v<at.nb;++v ){ const int w=v+1<at.nb?v+1:0;
                a += (double)at.vx[v]*at.vy[w]-(double)at.vx[w]*at.vy[v]; } }
        const double dt=now()-t0; if ( dt<t_b ){ t_b=dt; aire_b=0.5*a; } }

    for ( int r=0;r<3;++r ){ double a=0; long long c=0; const double t0=now();
        for ( int k=0;k<n;++k ){
            noyau2d::Atelier<M> at;
            noyau2d::FournisseurBsp<AaBspT<2>,true> f( &arbre, px[k], py[k], pw[k], ids[k] );
            noyau2d::etats::moteur( &f, &at );
            nb_a[k]=at.nb; if ( at.nb<=0 ) continue;
            c += at.nb;
            for ( int v=0;v<at.nb;++v ) id_a[(size_t)k*M+v]=at.cid[v];
            for ( int v=0;v<at.nb;++v ){ const int w=v+1<at.nb?v+1:0;
                a += (double)at.vx[v]*at.vy[w]-(double)at.vx[w]*at.vy[v]; } }
        const double dt=now()-t0; if ( dt<t_a ){ t_a=dt; aire_a=0.5*a; cand_a=c; } }

    std::vector<int> nb_c(n); std::vector<int> id_c((size_t)n*M);
    for ( int k=0;k<n;++k ){
        noyau2d::Atelier<M> at;
        Balayage f{ -1, px.data(), py.data(), pw.data(), px[k], py[k], pw[k], n, 0, ids[k], ids.data() };
        noyau2d::etats::moteur( &f, &at );
        nb_c[k]=at.nb;
        for ( int v=0;v<at.nb;++v ) id_c[(size_t)k*M+v]=at.cid[v];
    }

    auto rot_eq=[]( const int *a, const int *b, int m ){
        for ( int r=0;r<m;++r ){ bool ok=true;
            for ( int v=0;v<m;++v ) if ( a[(v+r)%m]!=b[v] ){ ok=false; break; }
            if ( ok ) return true; } return m==0; };
    int faux=0;
    for ( int k=0;k<n;++k )
        if ( nb_b[k]!=nb_a[k] || !rot_eq(&id_b[(size_t)k*M],&id_a[(size_t)k*M],nb_b[k]) ) ++faux;

    printf( "n=%d, poids ~ U[0, %.1f h^2]\n", n, frac );
    printf( "  balayage complet : %8.4f s   aire %.9f\n", t_b, aire_b );
    printf( "  fournisseur BSP  : %8.4f s   aire %.9f   x%.1f   %.2f cotes/cellule\n",
            t_a, aire_a, t_b/t_a, double(cand_a)/n );
    int faux_ordre=0;
    for ( int k=0;k<n;++k )
        if ( nb_b[k]!=nb_c[k] || !rot_eq(&id_b[(size_t)k*M],&id_c[(size_t)k*M],nb_b[k]) ) ++faux_ordre;
    printf( "  BSP vs balayage direct   : %d / %d\n", faux, n );
    printf( "  balayage direct vs INVERSE : %d / %d   <- meme algo, seul l'ordre change\n",
            faux_ordre, n );
    return 0;
}
