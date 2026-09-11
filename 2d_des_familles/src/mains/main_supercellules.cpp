// LES SUR-CELLULES D'AGREGATS -- le pilote. Il ne contient que les options et l'enchainement ;
// tout le calcul est dans `src/supercell/`.
//
// = LE PLAN, en quatre etapes
//
//   1. un VORONOI GROSSIER sur un germe echantillonne tous les `rho`, avec sa CONNECTIVITE, la
//      liste des diracs de chaque cellule grossiere et un dirac MEDIAN par cellule ;
//   2. pour chaque dirac, sa cellule contre son AGREGAT et les MEDIANS voisins, la recette pour
//      la rejouer, et l'ENCEINTE de l'agregat -- la sur-cellule ;          <-- JUSQU'ICI
//   3. la TABLE de quelles enceintes se debordent ;
//   4. chaque cellule finale : on rejoue la cellule de base, et on ne l'oppose qu'aux agregats
//      dont l'enceinte la rencontre encore.
//
// = OU EST QUOI
//
//   `supercell/Ordre.h`        l'ordre spatial des germes (Morton, BSP)
//   `supercell/GrilleSites.h`  « quel site est le plus proche ? »
//   `supercell/Agregats.h`     ETAPE 1 : le diagramme grossier, les agregats, les medians
//   `supercell/Enceinte.h`     l'enveloppe convexe et le k-DOP, meme interface
//   `supercell/Memoire.h`      les trois stockages de la recette d'une cellule
//   `supercell/SurCellules.h`  ETAPE 2 : la passe 1 et les enceintes
//
// Chaque etape porte SON bilan et SA ligne d'affichage : ce qu'elle mesure ne se lit nulle part
// ailleurs, et on peut en ajouter une sans toucher aux autres.
//
// = LA SUR-CELLULE EST CALCULEE, PAS APPROCHEE. C'est ce qui a change.
//
// La premiere piste minorait `psi_A` par des plans issus d'une enceinte convexe : il fallait des
// pseudo-germes, un cran `omega = (arete/2)^2` irreductible, et un sous-decoupage pour le diviser.
// Tout cela a disparu -- voir l'en-tete de `SurCellules.h` pour la raison. Aucune borne de poids
// n'intervient plus nulle part.
//
// = LES TEMOINS, et ce qu'ils disent
//
//   `somme des |C_i| >= 1`   les `U_A` recouvrent le domaine et les cellules d'un agregat le
//                            pavent. L'EXCES sur 1 est exactement le recouvrement des
//                            sur-cellules, c'est-a-dire leur prix.
//   `somme des |enc| >= somme des |C_i|`   convexifier ne peut qu'ajouter.
//   le REJEU                 refaire la cellule avec la seule recette doit rendre LES MEMES
//                            FACETTES. Le critere est combinatoire et non metrique : sur une
//                            cellule en lame l'aire est une difference de grands termes et son
//                            ecart relatif (~1e-09) ne dit rien de la justesse.
//   l'ECART D'ENCEINTE       de combien l'enceinte rate le pire point, en unites de `h`. Une
//                            distance et pas un booleen -- un `false` ne distingue pas une
//                            enceinte fausse du dernier bit d'un produit vectoriel.
//
// = L'ANNEAU, et pourquoi 1 ne suffit pas
//
// Avec les seuls voisins DIRECTS, l'opposition ne compte que ~6 medians en 2D et ~15 en 3D : il
// reste des directions ou un membre de `A` bat tous les medians, et la sur-cellule part tres loin.
// Mesure : `somme des |C_i|` vaut 24.4 a l'anneau 1 et 2.07 a l'anneau 2, en 2D uniforme. L'anneau
// 3 rend le meme chiffre a la quatrieme decimale -- l'anneau 2 est le point fixe.
//
//   xmake run pd_supercellules --help

#include "supercell/Agregats.h"
#include "supercell/SurCellules.h"
#include "supercell/Cellules.h"
#include "supercell/Table.h"
#include <cstdio>
#include <string>
#include <vector>

using namespace pd;
using namespace pd::bench;
using namespace pd::supercell;

namespace {

// ------------------------------------------------------------------ LE PILOTE
//
// Trois dispatches emboites -- le stockage, l'enceinte, le drapeau « poids » -- et tous les trois
// a la COMPILATION. C'est la raison d'etre des parametres template : le corps de l'etape 2 ne
// contient pas un seul `if` sur le mode choisi, donc mesurer un mode mesure bien ce mode.

/// `hull` = l'enveloppe convexe exacte. `kdop` = `K` directions FIXES. `kdopo` = les memes `K`
/// directions dans le REPERE PROPRE de l'agregat -- meme cout a un passage pres, mais elles
/// suivent la forme au lieu de la subir.
enum class TypeEnceinte { Hull, KDop, KDopO, Obb };

struct Reglages {
    SI   rho = 8;
    bool morton = false;
    int  etape = 2, anneau = 2, dirs = 0;
    bool verif = false;
    Opposants opp = Opposants::Median;
    StatsE2 *st = nullptr;
    TypeEnceinte enc = TypeEnceinte::Hull;
    Stockage stk = Stockage::Bits;
};

/// L'enchainement des etapes 2 et 3, une fois TOUS les types fixes. C'est ici que les enceintes
/// sont gardees : l'etape 2 les jette par defaut ( elles pesent ~280 octets piece en 3D ), et ne
/// les conserve que si l'etape 3 est demandee.
template<int D, Stockage S, class Enc, Opposants Mode>
bool lance( const Cloud<D> &cl, const Gros<D> &G, const Reglages &r, SI a_leaf ) {
    std::vector<Enc> enc;
    Memoire<S, CellSC<D>, D> mem;
    std::vector<Enc> *garde = r.etape >= 3 ? &enc : nullptr;
    auto *gmem = r.etape >= 4 ? &mem : nullptr;
    const BilanSC b = cl.W ? etape2<D, true,  S, Enc, Mode>( cl, G, r.anneau, r.verif, r.st, garde, gmem )
                           : etape2<D, false, S, Enc, Mode>( cl, G, r.anneau, r.verif, r.st, garde, gmem );
    ligne_surcellules( cl, b );
    if ( r.etape < 3 || ! b.ok )
        return b.ok;
    Table t;
    const BilanTable bt = etape3<D, Enc>( enc, t, r.verif );
    ligne_table( cl, bt );
    if ( r.etape < 4 || ! bt.ok )
        return bt.ok;
    const BilanCel bc = cl.W
        ? etape4<D, true,  S, Enc, Mode>( cl, G, t, enc, mem, r.anneau, r.verif, a_leaf )
        : etape4<D, false, S, Enc, Mode>( cl, G, t, enc, mem, r.anneau, r.verif, a_leaf );
    ligne_cellules( cl, bc );
    return bc.ok;
}

/// L'aiguillage sur les opposants, lui aussi a la COMPILATION : le corps des etapes 2 et 4 ne
/// contient aucun test a l'execution sur ce choix.
template<int D, Stockage S, class Enc>
bool selon_opposants( const Cloud<D> &cl, const Gros<D> &G, const Reglages &r, SI a_leaf ) {
    switch ( r.opp ) {
        case Opposants::Tous:  return lance<D, S, Enc, Opposants::Tous  >( cl, G, r, a_leaf );
        case Opposants::Mixte: return lance<D, S, Enc, Opposants::Mixte >( cl, G, r, a_leaf );
        default:               return lance<D, S, Enc, Opposants::Median>( cl, G, r, a_leaf );
    }
}

template<int D, Stockage S, template<int,int> class Dop>
bool selon_dirs( const Cloud<D> &cl, const Gros<D> &G, const Reglages &r, SI a_leaf ) {
    if constexpr ( D == 2 ) {
        switch ( r.dirs ) {
            case  2: return selon_opposants<D, S, Dop<D,  2>>( cl, G, r, a_leaf );
            case  8: return selon_opposants<D, S, Dop<D,  8>>( cl, G, r, a_leaf );
            case 16: return selon_opposants<D, S, Dop<D, 16>>( cl, G, r, a_leaf );
            default: return selon_opposants<D, S, Dop<D,  4>>( cl, G, r, a_leaf );
        }
    } else {
        switch ( r.dirs ) {
            case  3: return selon_opposants<D, S, Dop<D,  3>>( cl, G, r, a_leaf );
            case  4: return selon_opposants<D, S, Dop<D,  4>>( cl, G, r, a_leaf );
            case 13: return selon_opposants<D, S, Dop<D, 13>>( cl, G, r, a_leaf );
            default: return selon_opposants<D, S, Dop<D,  7>>( cl, G, r, a_leaf );
        }
    }
}

/// `Obb` a son propre aiguillage : `M = 1` ( le seul repere propre ) a un sens pour elle et pas
/// pour un k-DOP, dont une direction unique ne fait pas un repere.
template<int D, Stockage S>
bool selon_obb( const Cloud<D> &cl, const Gros<D> &G, const Reglages &r, SI a_leaf ) {
    if constexpr ( D == 2 ) {
        switch ( r.dirs ) {
            case  1: return selon_opposants<D, S, Obb<D,  1>>( cl, G, r, a_leaf );
            case  2: return selon_opposants<D, S, Obb<D,  2>>( cl, G, r, a_leaf );
            case  4: return selon_opposants<D, S, Obb<D,  4>>( cl, G, r, a_leaf );
            case 16: return selon_opposants<D, S, Obb<D, 16>>( cl, G, r, a_leaf );
            default: return selon_opposants<D, S, Obb<D,  8>>( cl, G, r, a_leaf );
        }
    } else {
        switch ( r.dirs ) {
            case  1: return selon_opposants<D, S, Obb<D,  1>>( cl, G, r, a_leaf );
            case  4: return selon_opposants<D, S, Obb<D,  4>>( cl, G, r, a_leaf );
            case 13: return selon_opposants<D, S, Obb<D, 13>>( cl, G, r, a_leaf );
            default: return selon_opposants<D, S, Obb<D,  7>>( cl, G, r, a_leaf );
        }
    }
}

template<int D, Stockage S>
bool selon_enceinte( const Cloud<D> &cl, const Gros<D> &G, const Reglages &r, SI a_leaf ) {
    switch ( r.enc ) {
        case TypeEnceinte::KDop:  return selon_dirs<D, S, KDop       >( cl, G, r, a_leaf );
        case TypeEnceinte::KDopO: return selon_dirs<D, S, KDopOriente>( cl, G, r, a_leaf );
        case TypeEnceinte::Obb:   return selon_obb<D, S>( cl, G, r, a_leaf );
        default:                  return selon_opposants<D, S, Enveloppe<D>>( cl, G, r, a_leaf );
    }
}

const char *nom_stockage( Stockage s ) {
    return s == Stockage::Bits ? "bits" : ( s == Stockage::Packe ? "packe" : "complet" );
}

template<int D>
void entete( int dim, const Args &a, const Reglages &r ) {
    char enc[ 32 ];
    // les valeurs tabulees, et non celle demandee : `--dirs 8` en 3D retombe sur 7,
    // et l'en-tete doit dire ce qui a TOURNE.
    const int nd = r.enc == TypeEnceinte::Obb
        ? ( D == 2 ? ( r.dirs == 1 || r.dirs == 2 || r.dirs == 4 || r.dirs == 16 ? r.dirs : 8 )
                   : ( r.dirs == 1 || r.dirs == 4 || r.dirs == 13 ? r.dirs : 7 ) )
        : ( D == 2 ? ( r.dirs == 2 || r.dirs == 8 || r.dirs == 16 ? r.dirs : 4 )
                   : ( r.dirs == 3 || r.dirs == 4 || r.dirs == 13 ? r.dirs : 7 ) );
    if      ( r.enc == TypeEnceinte::KDop  ) std::snprintf( enc, sizeof enc, "kdop%d", nd );
    else if ( r.enc == TypeEnceinte::KDopO ) std::snprintf( enc, sizeof enc, "kdopo%d", nd );
    else if ( r.enc == TypeEnceinte::Obb   ) std::snprintf( enc, sizeof enc, "obb%d", nd );
    else                                     std::snprintf( enc, sizeof enc, "hull" );
    std::printf( "=== %dD  etape %d  rho=%d  ordre=%s  leaf=%d  stockage=%s  anneau=%d"
                 "  enceinte=%s%s\n",
                 dim, r.etape, int( r.rho ), r.morton ? "morton" : "bsp", int( a.leaf ),
                 nom_stockage( r.stk ), r.anneau, enc, r.verif ? "  +verif" : "" );
}

/// Le deroule d'une suite de nuages : l'etape 1, puis les etapes 2 et 3 sous leurs types.
template<int D>
int deroule( const Args &a, const std::vector<Cloud<D>> &cas, const Reglages &r ) {
    int bad = 0;
    for ( const Cloud<D> &cl : cas ) {
        if ( cl.absent ) {
            std::printf( "  %-24s : ABSENT (lancer cases/gen_cases.py)\n", cl.nom.c_str() );
            continue;
        }
        Gros<D> G;
        const BilanGros b = cl.W ? etape1<D, true >( cl, r.rho, r.morton, a.leaf, G )
                                 : etape1<D, false>( cl, r.rho, r.morton, a.leaf, G );
        ligne_gros( cl, b );
        bad += ! b.ok;
        if ( r.etape < 2 || ! b.ok )
            continue;
        bad += ! ( r.stk == Stockage::Bits  ? selon_enceinte<D, Stockage::Bits   >( cl, G, r, a.leaf )
                 : r.stk == Stockage::Packe ? selon_enceinte<D, Stockage::Packe  >( cl, G, r, a.leaf )
                                            : selon_enceinte<D, Stockage::Complet>( cl, G, r, a.leaf ) );
    }
    return bad;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    Reglages r;
    bool stats = false;
    StatsE2 st;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if ( parse_commun( a, s, i, argc, argv ) ) continue;
        if      ( s == "--rho" )      r.rho = SI( std::atoi( val() ) );
        else if ( s == "--ordre" )    r.morton = std::string( val() ) == "morton";
        else if ( s == "--etape" )    r.etape = std::atoi( val() );
        else if ( s == "--anneau" )   r.anneau = std::atoi( val() );
        else if ( s == "--dirs" )     r.dirs = std::atoi( val() );
        else if ( s == "--opposants" ) {
            const std::string v = val();
            if      ( v == "median" ) r.opp = Opposants::Median;
            else if ( v == "tous" )   r.opp = Opposants::Tous;
            else if ( v == "mixte" )  r.opp = Opposants::Mixte;
            else { std::printf( "opposants inconnus : `%s`\n", v.c_str() ); return 1; }
        }
        else if ( s == "--verif" )    r.verif = true;
        else if ( s == "--stats" )    stats = true;
        else if ( s == "--enceinte" ) {
            const std::string v = val();
            // PAS de repli silencieux. Un `--enceinte` mal orthographie qui retombait sur `hull`
            // a coute deux campagnes de mesures entieres : les lignes se ressemblaient assez pour
            // passer pour des resultats, et assez peu pour qu'on cherche l'explication ailleurs.
            if      ( v == "hull"  ) r.enc = TypeEnceinte::Hull;
            else if ( v == "kdop"  ) r.enc = TypeEnceinte::KDop;
            else if ( v == "kdopo" ) r.enc = TypeEnceinte::KDopO;
            else if ( v == "obb"   ) r.enc = TypeEnceinte::Obb;
            else { std::printf( "enceinte inconnue : `%s`\n", v.c_str() ); return 1; }
        }
        else if ( s == "--stockage" ) {
            const std::string v = val();
            if      ( v == "bits"    ) r.stk = Stockage::Bits;
            else if ( v == "packe"   ) r.stk = Stockage::Packe;
            else if ( v == "complet" ) r.stk = Stockage::Complet;
            else { std::printf( "stockage inconnu : `%s`\n", v.c_str() ); return 1; }
        } else {
            std::printf( "usage: pd_supercellules [options]\n" );
            usage_commun();
            std::printf( "  --rho N         un site tous les N germes dans l'ordre spatial (8)\n"
                         "  --ordre O       `bsp` (celui de AaBsp, defaut) ou `morton`\n"
                         "  --etape N       1 = le Voronoi grossier, 2 = + les sur-cellules (defaut),\n"
                         "                  3 = + la table des enceintes qui se touchent,\n"
                         "                  4 = + les cellules finales\n"
                         "  --stockage M    `bits` (defaut), `packe` ou `complet`\n"
                         "  --anneau N      medians opposes : 1 = les voisins directs,\n"
                         "                  2 = leurs voisins aussi (defaut), etc.\n"
                         "  --enceinte E    `hull` (enveloppe convexe, defaut), `kdop` (directions\n"
                         "                  fixes), `kdopo` (directions du repere propre) ou `obb`\n"
                         "                  (boite orientee, meilleur de M reperes candidats)\n"
                         "  --opposants O   `median` (defaut), `tous`, ou `mixte` : anneau 1 complet\n"
                         "                  plus medians de anneau 2 ; etape 4 saute alors anneau 1\n"
                         "  --dirs K        directions du k-DOP, ou reperes candidats de `obb` :\n"
                         "                  3D (defaut 4 et 7, soit le 8-DOP et le 14-DOP)\n"
                         "  --stats         histogrammes : sommets par cellule, efficacite par rang\n"
                         "  --verif         rejoue chaque cellule depuis sa recette et mesure de\n"
                         "                  combien l'enceinte rate le pire point (quadratique)\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    if ( stats ) r.st = &st;
    finalise( a );
    if ( r.rho < 1 ) r.rho = 1;
    if ( r.anneau < 1 ) r.anneau = 1;

    int bad = 0;
    if ( a.dims != 3 ) { entete<2>( 2, a, r ); bad += deroule<2>( a, suite_2d( a ), r ); }
    if ( a.dims != 2 ) { entete<3>( 3, a, r ); bad += deroule<3>( a, suite_3d( a ), r ); }
    if ( stats ) {
        std::printf( "\n--- SOMMETS PAR CELLULE ( %lld cellules )\n", st.cellules );
        std::printf( "    nb :" );
        for ( int k = 0; k < StatsE2::MAXNB; ++k ) if ( st.hist_final[ k ] )
            std::printf( " %d=%.2f%%", k, 100.0 * st.hist_final[ k ] / std::max( 1LL, st.cellules ) );
        std::printf( "\n    au fil des coupes ( toutes etapes intermediaires confondues ) :\n    nb :" );
        long long tote = 0;
        for ( int k = 0; k < StatsE2::MAXNB; ++k ) tote += st.hist_evol[ k ];
        for ( int k = 0; k < StatsE2::MAXNB; ++k ) if ( st.hist_evol[ k ] )
            std::printf( " %d=%.2f%%", k, 100.0 * st.hist_evol[ k ] / std::max( 1LL, tote ) );
        std::printf( "\n\n--- COUPES EFFECTIVES SELON LE RANG TESTE\n" );
        std::printf( "    rang :" );
        for ( int q = 0; q < 28; ++q ) std::printf( " %4d", q );
        std::printf( "\n    testes%%:" );
        for ( int q = 0; q < 28; ++q )
            std::printf( " %4.0f", 100.0 * st.testes[ q ] / std::max( 1LL, st.cellules ) );
        std::printf( "\n    effic%% :" );
        for ( int q = 0; q < 28; ++q )
            std::printf( " %4.0f", 100.0 * st.effectifs[ q ] / std::max( 1LL, st.testes[ q ] ) );
        std::printf( "\n    nb moy :" );
        for ( int q = 0; q < 28; ++q )
            std::printf( " %4.1f", double( st.nb_apres[ q ] ) / std::max( 1LL, st.testes[ q ] ) );
        std::printf( "\n" );
    }
    return bad ? 1 : 0;
}
