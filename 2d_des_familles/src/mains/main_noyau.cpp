// =====================================================================================
// LE BANC DU NOYAU A REGISTRES. Il ne mesure QUE la coupe.
//
// Les trois chemins recoivent la MEME liste d'opposants, deja triee par distance et construite
// hors chronometre : ce qui reste dans l'horloge est le clip et rien d'autre. La liste est
// passee au noyau sous la forme d'un FOURNISSEUR ( `ListeFournie` ) -- le noyau ne sait plus
// qu'une liste existe, et `main_n50` mesure le meme moteur avec un fournisseur qui n'en a pas.
//
//   scalaire : `CellSoAT<32>`, double, telle qu'elle tourne dans les bancs
//   hybride  : `noyau2d` en float, et `Repli2D` en double pour les cellules qui echappent
//   noyau nu : le noyau seul, echappees ABANDONNEES -- ce n'est pas un resultat utilisable,
//              c'est la borne haute que le repli ne peut pas depasser
//
// POURQUOI LE « NOYAU NU » FIGURE ICI. Sans lui on ne sait pas lire l'hybride : un hybride a
// 2x ne dit pas si le noyau est a 2x et le repli gratuit, ou le noyau a 4x et le repli qui en
// mange la moitie. Les cellules qui echappent sont les grosses -- 3.7 % des cellules mais bien
// plus que 3.7 % du travail -- donc l'ecart entre les deux colonnes est le vrai prix du repli.
//
// LA COMPARAISON DE JUSTESSE porte sur DEUX choses, qu'il ne faut pas confondre :
//   - la COMBINATOIRE ( la suite cyclique des `cid` ) : c'est elle qui doit etre exacte,
//     puisque de la connectivite on refait la geometrie en double quand on veut ;
//   - la GEOMETRIE en float : on la mesure pour savoir ce qu'elle coute, pas pour s'en servir
//     telle quelle.
// =====================================================================================

#include "geometry/Cell.h"
#include "supercell/Fournisseurs.h"
#include "supercell/Noyau2DEtats.h"
#include "supercell/Repli2D.h"
#include "util/common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace pd;

static double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

/// l'aire d'un cycle, par le lacet.
template<class T>
static double lacet( const T *vx, const T *vy, int nb ) {
    double s = 0;
    for ( int v = 0; v < nb; ++v ) {
        const int w = v + 1 < nb ? v + 1 : 0;
        s += (double) vx[ v ] * (double) vy[ w ] - (double) vx[ w ] * (double) vy[ v ];
    }
    return 0.5 * s;
}

int main( int argc, char **argv ) {
    SI  n = 400000;
    int K = 24, rep = 3;
    for ( int i = 1; i + 1 < argc; i += 2 ) {
        if      ( ! std::string( argv[ i ] ).compare( "--n"   ) ) n   = atoi( argv[ i + 1 ] );
        else if ( ! std::string( argv[ i ] ).compare( "--k"   ) ) K   = atoi( argv[ i + 1 ] );
        else if ( ! std::string( argv[ i ] ).compare( "--rep" ) ) rep = atoi( argv[ i + 1 ] );
        else { fprintf( stderr, "option inconnue %s\n", argv[ i ] ); return 1; }
    }

    // ---- le nuage, trie par cellule de grille : les listes d'opposants deviennent locales.
    std::mt19937 gen( 12345 );
    std::uniform_real_distribution<double> u( 0, 1 );
    std::vector<double> ax( n ), ay( n );
    for ( SI i = 0; i < n; ++i ) { ax[ i ] = u( gen ); ay[ i ] = u( gen ); }

    const int    g = std::max( 1, (int) std::sqrt( double( n ) / 2 ) );
    const double h = 1.0 / g;
    std::vector<SI> cpt( g * g + 1, 0 ), ord( n ), deb;
    auto cell = [ & ]( SI i ) {
        int cx = std::min( g - 1, (int) ( ax[ i ] / h ) ), cy = std::min( g - 1, (int) ( ay[ i ] / h ) );
        return cy * g + cx;
    };
    for ( SI i = 0; i < n; ++i ) ++cpt[ cell( i ) + 1 ];
    for ( SI c = 0; c < g * g; ++c ) cpt[ c + 1 ] += cpt[ c ];
    deb = cpt;
    for ( SI i = 0; i < n; ++i ) ord[ deb[ cell( i ) ]++ ] = i;

    std::vector<double> px( n ), py( n );
    std::vector<float>  fx( n ), fy( n );
    for ( SI k = 0; k < n; ++k ) {
        px[ k ] = ax[ ord[ k ] ]; py[ k ] = ay[ ord[ k ] ];
        fx[ k ] = (float) px[ k ]; fy[ k ] = (float) py[ k ];
    }

    // ---- les listes d'opposants : bloc 5x5 de cellules, tri par distance, K premiers.
    //      HORS CHRONOMETRE, et identiques pour les trois chemins.
    const double t_prep = now();
    std::vector<int> opp( (size_t) n * K );
    std::vector<std::pair<double,int>> tmp;
    for ( SI k = 0; k < n; ++k ) {
        const int cx = std::min( g - 1, (int) ( px[ k ] / h ) ), cy = std::min( g - 1, (int) ( py[ k ] / h ) );
        for ( int r = 2; ; ++r ) {                       // on elargit jusqu'a tenir K candidats
            tmp.clear();
            for ( int dy = -r; dy <= r; ++dy ) for ( int dx = -r; dx <= r; ++dx ) {
                const int bx = cx + dx, by = cy + dy;
                if ( bx < 0 || by < 0 || bx >= g || by >= g ) continue;
                for ( SI q = cpt[ by * g + bx ]; q < cpt[ by * g + bx + 1 ]; ++q ) {
                    if ( q == k ) continue;
                    const double ex = px[ q ] - px[ k ], ey = py[ q ] - py[ k ];
                    tmp.push_back( { ex * ex + ey * ey, (int) q } );
                }
            }
            if ( (int) tmp.size() >= K || r > g ) break;
        }
        const int nk = std::min( K, (int) tmp.size() );
        std::partial_sort( tmp.begin(), tmp.begin() + nk, tmp.end() );
        for ( int q = 0; q < K; ++q ) opp[ (size_t) k * K + q ] = q < nk ? tmp[ q ].second : tmp[ nk - 1 ].second;
    }
    printf( "n = %lld, %d opposants par cellule, %d passes ( on garde la meilleure )\n",
            (long long) n, K, rep );
    printf( "preparation ( hors chrono ) : %.3f s\n\n", now() - t_prep );

    // les sorties, communes aux trois chemins : nb de sommets et suite des `cid`.
    std::vector<int> sca_nb( n ), hyb_nb( n ), rej_nb( n );
    std::vector<int> sca_id( (size_t) n * 32 ), hyb_id( (size_t) n * 32 ), rej_id( (size_t) n * 32 );
    double aire_sca = 0, aire_hyb = 0, aire_rej = 0, aire_nu = 0;
    SI nb_echap = 0, nb_debord = 0;
    std::vector<char> est_echap( n, 0 );
    double t_sca = 1e30, t_nu = 1e30, t_hyb = 1e30, t_rej = 1e30;

    // ================= 1. SCALAIRE PUR =================
    for ( int r = 0; r < rep; ++r ) {
        double a = 0;
        const double t0 = now();
        for ( SI k = 0; k < n; ++k ) {
            CellSoAT<32> c;
            c.init_as_unit_square();
            const double x0 = px[ k ], y0 = py[ k ];
            for ( int q = 0; q < K; ++q ) {
                const int j = opp[ (size_t) k * K + q ];
                const double dx = px[ j ] - x0, dy = py[ j ] - y0;
                const double off = 0.5 * ( dx * ( px[ j ] + x0 ) + dy * ( py[ j ] + y0 ) );
                c.cut( dx, dy, off, j );
            }
            sca_nb[ k ] = (int) c.nb;
            for ( SI v = 0; v < c.nb; ++v ) sca_id[ (size_t) k * 32 + v ] = (int) c.cid[ v ];
            a += lacet( c.vx, c.vy, (int) c.nb );
        }
        t_sca = std::min( t_sca, now() - t0 );
        aire_sca = a;
    }

    // ================= 2. NOYAU NU ( echappees abandonnees ) =================
    //
    // MEME COMPTABILITE QUE LES AUTRES. Une premiere version se contentait de compter les
    // echappees sans ranger ni `nb`, ni `cid`, ni aire : elle sortait a x2.29 -- un chiffre qui
    // ne mesurait pas le noyau mais l'absence d'ecriture. Les quatre colonnes font desormais le
    // meme travail de rangement, seule la coupe differe.
    for ( int r = 0; r < rep; ++r ) {
        SI ech = 0; double a = 0;
        const double t0 = now();
        for ( SI k = 0; k < n; ++k ) {
            noyau2d::Atelier<8> c;
            noyau2d::ListeFournie f( fx.data(), fy.data(), fx[ k ], fy[ k ],
                                     opp.data() + (size_t) k * K, K );
            noyau2d::etats::moteur( &f, &c );
            rej_nb[ k ] = c.nb;
            if ( c.nb < 0 ) { ++ech; continue; }
            for ( int v = 0; v < c.nb; ++v ) hyb_id[ (size_t) k * 32 + v ] = c.cid[ v ];
            a += lacet( c.vx, c.vy, c.nb );
        }
        t_nu = std::min( t_nu, now() - t0 );
        nb_echap = ech; aire_nu = a;
    }

    // ================= 3. HYBRIDE, repli PAR REPRISE =================
    for ( int r = 0; r < rep; ++r ) {
        double a = 0; SI deb2 = 0;
        const double t0 = now();
        for ( SI k = 0; k < n; ++k ) {
            noyau2d::Atelier<8> c;
            noyau2d::ListeFournie f( fx.data(), fy.data(), fx[ k ], fy[ k ],
                                     opp.data() + (size_t) k * K, K );
            noyau2d::etats::moteur( &f, &c );
            if ( c.nb >= 0 ) {
                hyb_nb[ k ] = c.nb;
                for ( int v = 0; v < c.nb; ++v ) hyb_id[ (size_t) k * 32 + v ] = c.cid[ v ];
                a += lacet( c.vx, c.vy, c.nb );
            } else {
                CellSoAT<32> s;
                noyau2d::Local<noyau2d::ListeFournie> lo{};
                if ( ! noyau2d::reprise( s, c, &f, lo, px.data(), py.data(), px[ k ], py[ k ] ) ) ++deb2;
                hyb_nb[ k ] = (int) s.nb;
                for ( SI v = 0; v < s.nb; ++v ) hyb_id[ (size_t) k * 32 + v ] = (int) s.cid[ v ];
                a += lacet( s.vx, s.vy, (int) s.nb );
                est_echap[ k ] = 1;
            }
        }
        t_hyb = std::min( t_hyb, now() - t0 );
        aire_hyb = a; nb_debord = deb2;
    }

    // ================= 4. HYBRIDE, repli PAR REJEU =================
    for ( int r = 0; r < rep; ++r ) {
        double a = 0;
        const double t0 = now();
        for ( SI k = 0; k < n; ++k ) {
            noyau2d::Atelier<8> c;
            noyau2d::ListeFournie f( fx.data(), fy.data(), fx[ k ], fy[ k ],
                                     opp.data() + (size_t) k * K, K );
            noyau2d::etats::moteur( &f, &c );
            if ( c.nb >= 0 ) {
                rej_nb[ k ] = c.nb;
                for ( int v = 0; v < c.nb; ++v ) rej_id[ (size_t) k * 32 + v ] = c.cid[ v ];
                a += lacet( c.vx, c.vy, c.nb );
            } else {
                CellSoAT<32> s;
                noyau2d::ListeFournie f2( fx.data(), fy.data(), fx[ k ], fy[ k ],
                                          opp.data() + (size_t) k * K, K );  // neuf : celui du noyau est entame
                noyau2d::Local<noyau2d::ListeFournie> lo2{};
                noyau2d::rejeu( s, &f2, lo2, px.data(), py.data(), px[ k ], py[ k ] );
                rej_nb[ k ] = (int) s.nb;
                for ( SI v = 0; v < s.nb; ++v ) rej_id[ (size_t) k * 32 + v ] = (int) s.cid[ v ];
                a += lacet( s.vx, s.vy, (int) s.nb );
            }
        }
        t_rej = std::min( t_rej, now() - t0 );
        aire_rej = a;
    }

    // ---- LES DEUX COMPARAISONS, ET L'OUTIL QU'ELLES PARTAGENT.
    //
    // EGAL VEUT DIRE : A ROTATION PRES, ET NEGATIFS REPLIES SUR -1. Les chemins n'ont ni la
    // meme origine de cycle, ni la meme convention de domaine -- `init_as_unit_square` marque
    // les quatre cotes -1, `carre_unite` les distingue -1..-4, et celle du noyau est la plus
    // informative des deux. Comparer voie a voie faisait passer pour fausses toute cellule
    // touchant le bord ( 2.4 % du nuage ), puis 96 % des cellules replies : le test etait faux,
    // pas le code. Ce qu'on verifie est l'identite des VOISINS, et rien d'autre.
    auto rep1 = []( int c ) { return c < 0 ? -1 : c; };
    auto rot_eq = [ & ]( const int *a, const int *b, int m ) {
        for ( int r = 0; r < m; ++r ) {
            bool ok = true;
            for ( int v = 0; v < m; ++v )
                if ( rep1( a[ ( v + r ) % m ] ) != rep1( b[ v ] ) ) { ok = false; break; }
            if ( ok ) return true;
        }
        return m == 0;
    };

    // 1. l'hybride contre le scalaire : la justesse du resultat.
    SI faux = 0, faux_nb = 0, faux_repli = 0;
    for ( SI k = 0; k < n; ++k ) {
        if ( hyb_nb[ k ] != sca_nb[ k ] ) { ++faux; ++faux_nb; faux_repli += est_echap[ k ]; continue; }
        if ( ! rot_eq( &hyb_id[ (size_t) k * 32 ], &sca_id[ (size_t) k * 32 ], sca_nb[ k ] ) ) {
            ++faux; faux_repli += est_echap[ k ];
        }
    }

    // 1 bis. le rejeu contre le scalaire. Ce que ca isole : `reprise` HERITE de la suite de
    // `cid` que le noyau a etablie EN FLOAT, et ne peut donc pas rattraper une decision limite
    // deja prise ; `rejeu` repart de zero en double et n'herite de rien. Comparer les deux
    // colonnes dit si le legs coute de la justesse, et combien.
    SI faux_rej = 0;
    for ( SI k = 0; k < n; ++k ) {
        if ( rej_nb[ k ] != sca_nb[ k ] ) { ++faux_rej; continue; }
        if ( ! rot_eq( &rej_id[ (size_t) k * 32 ], &sca_id[ (size_t) k * 32 ], sca_nb[ k ] ) ) ++faux_rej;
    }

    // 2. reprise contre rejeu : la validation de `Repli2D`, et elle ne se lit nulle part
    // ailleurs. `rejeu` refait tout en double depuis le carre unite ; `reprise` reconstruit la
    // cellule depuis les seuls `cid` legues par le noyau, par systemes 2x2, et enchaine. Si la
    // reconstruction 2x2 etait fausse, c'est ICI que ca se verrait -- l'aire totale, elle, est
    // dominee par les cellules non echappees et ne dirait rien.
    //
    // CE COMPTE N'EST PAS ZERO, ET C'EST NORMAL. Il vaut exactement le nombre de cellules
    // replies fausses ( ligne « dont passees par le repli » ) : ce ne sont pas des erreurs de
    // reconstruction mais l'HERITAGE. La suite de `cid` que la reprise recoit a ete etablie en
    // float ; si le noyau a tranche une configuration limite du mauvais cote avant d'echapper,
    // la reprise repart de cette decision et ne peut plus la rattraper, alors que le rejeu, qui
    // n'herite de rien, retombe sur le scalaire. C'est le prix du legs, et il se lit en
    // comparant les deux lignes « combinatoire fausse ».
    SI repr_vs_rej = 0;
    for ( SI k = 0; k < n; ++k ) {
        if ( hyb_nb[ k ] != rej_nb[ k ] ) { ++repr_vs_rej; continue; }
        if ( ! rot_eq( &hyb_id[ (size_t) k * 32 ], &rej_id[ (size_t) k * 32 ], hyb_nb[ k ] ) )
            ++repr_vs_rej;
    }


    printf( "                        temps        x scalaire     somme des aires\n" );
    printf( "scalaire ( double )   %8.4f s       1.00        %.12f\n", t_sca, aire_sca );
    printf( "noyau nu ( float )    %8.4f s    x %5.2f        %.12f  ( %.2f %% manquants )\n",
            t_nu, t_sca / t_nu, aire_nu, 100.0 * ( aire_sca - aire_nu ) / aire_sca );
    printf( "hybride / reprise     %8.4f s    x %5.2f        %.12f\n", t_hyb, t_sca / t_hyb, aire_hyb );
    printf( "hybride / rejeu       %8.4f s    x %5.2f        %.12f\n", t_rej, t_sca / t_rej, aire_rej );
    printf( "\n" );
    printf( "echappements          : %lld  ( %.2f %% des cellules )\n",
            (long long) nb_echap, 100.0 * nb_echap / n );
    printf( "  prix du repli       : %.1f %% du temps du noyau nu ( reprise ), %.1f %% ( rejeu )\n",
            100.0 * ( t_hyb - t_nu ) / t_nu, 100.0 * ( t_rej - t_nu ) / t_nu );
    printf( "  debordements > 32   : %lld\n", (long long) nb_debord );
    printf( "reprise != rejeu      : %lld  ( = ce que la reprise herite des decisions float )\n",
            (long long) repr_vs_rej );
    printf( "ecart d'aire hybride  : %.2e ( relatif )\n", std::abs( aire_hyb - aire_sca ) / aire_sca );
    // A QUI SONT LES ERREURS. Si le repli en portait une part superieure a son poids ( 3.7 %
    // des cellules ), c'est `Repli2D` qu'il faudrait suspecter et non le float du noyau.
    printf( "combinatoire fausse   : %lld  ( %.4f %% )  dont %lld a nb different\n",
            (long long) faux, 100.0 * faux / n, (long long) faux_nb );
    printf( "  dont passees par le repli : %lld  ( le repli est %.2f %% des cellules )\n",
            (long long) faux_repli, 100.0 * nb_echap / n );
    printf( "combinatoire fausse, rejeu : %lld  -- le rejeu n'herite d'aucune decision float\n",
            (long long) faux_rej );
    return 0;
}
