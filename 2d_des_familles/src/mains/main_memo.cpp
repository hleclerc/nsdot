// L'ARBRE QUI SE SOUVIENT : un entier par germe, un bit par germe de sa feuille, mis a un quand ce
// germe-la a fourni un cote de la cellule finale. A l'iteration suivante on rejoue ces coupes-la
// d'abord, et le parcours n'applique que le COMPLEMENT du masque.
//
// CE QUE CE BANC MESURE, ET CE QU'IL NE MESURE PAS. Les deux passes ont ICI LES MEMES POIDS, donc
// les souvenirs sont PARFAITS : chaque coupe rejouee est exactement une coupe de la cellule finale.
// C'est la BORNE SUPERIEURE de ce que l'idee peut rendre, et elle n'est pas nulle -- 16 a 19 % sur
// les cas faciles (voir README). Dans une boucle de Newton les poids bougent entre deux passes, les
// souvenirs se periment, et le gain mesure la-bas etait quasi nul : ce sont deux questions
// differentes et il faut les garder separees. Le gain de 3x obtenu dans `old_pd` etait, lui, en 3D,
// ou une coupe inutile coute beaucoup plus cher.
//
//   xmake run pd_memo --help

#include "spatial_accel/AaBspMemo.h"
#include "bench/Bench.h"
#include <cstdio>
#include <string>

using namespace pd;
using namespace pd::bench;

namespace {

/// La memoire ne sert qu'a partir de la SECONDE passe. Un banc qui ne mesure qu'une passe ne
/// mesurerait donc rien du tout : on en fait deux, et on chronometre la seconde.
template<class Cell, bool Weighted>
Mesure deux_passes( const Args &a, const Cloud<2> &cl, bool bits ) {
    AaBspMemo tr;
    const double t0 = now();
    tr.build( cl.P, cl.W, cl.n, a.leaf );
    tr.bits = bits;
    Mesure m;
    m.t_build = now() - t0;

    PowerDiagram<Cell, AaBspMemo, true, false, false, Weighted> pd{ tr };
    std::vector<TF> res;
    pd.measures( res, a.threads, a.split, a.pin );       // la passe qui REMPLIT les souvenirs
    tr.arme();
    pd.measures( res, a.threads, a.split, a.pin );       // chauffe, souvenirs en place

    m.t = 1e300;
    for ( int r = 0; r < a.reps; ++r ) {
        const double t1 = now();
        pd.measures( res, a.threads, a.split, a.pin );
        m.t = std::min( m.t, now() - t1 );
    }
    TF s = 0;
    for ( TF v : res ) s += v;
    m.somme = double( s );
    m.novf = pd.nb_overflow.load();
    m.ok = std::fabs( m.somme - 1.0 ) < 1e-9 && m.novf == 0;
    return m;
}

Mesure passe( const Args &a, const Cloud<2> &cl, bool bits ) {
    const int nv = a.nv( 2, cl.nv );
    auto go = [ & ]( auto tag ) {
        using Cell = decltype( tag );
        return cl.W ? deux_passes<Cell, true>( a, cl, bits )
                    : deux_passes<Cell, false>( a, cl, bits );
    };
    switch ( nv ) {
        case 16: return go( CellSoAT<16>{} );
        case 48: return go( CellSoAT<48>{} );
        case 64: return go( CellSoAT<64>{} );
        default: return go( CellSoAT<32>{} );
    }
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( parse_commun( a, s, i, argc, argv ) ) continue;
        std::printf( "usage: pd_memo [options]\n" );
        usage_commun();
        std::printf( "\n  Chaque cas est mesure DEUX FOIS : « bits » (les coupes rejouees) et\n"
                     "  « eteint » (le meme code, memoire desactivee). C'est l'ecart entre les deux\n"
                     "  qui est le resultat -- le comparer a `pd_bsp` melangerait la memoire et le\n"
                     "  surcout de l'enveloppe.\n"
                     "  ATTENTION : les deux passes ont les MEMES POIDS, donc les souvenirs sont\n"
                     "  parfaits. C'est la BORNE SUPERIEURE, pas le gain dans une boucle de Newton.\n" );
        return s == "--help" || s == "-h" ? 0 : 1;
    }
    finalise( a );

    std::printf( "=== 2D  seconde passe chronometree, souvenirs en place\n" );
    int bad = 0;
    for ( const Cloud<2> &cl : suite_2d( a ) ) {
        if ( cl.absent ) { std::printf( "  %-28s : ABSENT\n", cl.nom.c_str() ); continue; }
        const Mesure e = passe( a, cl, false );
        const Mesure b = passe( a, cl, true );
        std::printf( "  %-24s n=%-7d : eteint %7.3f s | bits %7.3f s  -> %+.1f %%   somme %.9f\n",
                     cl.nom.c_str(), int( cl.n ), e.t, b.t, 100 * ( b.t / e.t - 1 ), b.somme );
        bad += ! ( e.ok && b.ok );
    }
    std::printf( "=== 3D\n  PAS ENCORE : c'est pourtant LA dimension ou l'idee devrait payer.\n" );
    return bad ? 1 : 0;
}
