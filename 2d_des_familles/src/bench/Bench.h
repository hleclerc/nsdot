#pragma once

#include "geometry/Cell.h"
#include "geometry/Cell3.h"
#include "geometry/PowerDiagram.h"
#include "util/common.h"
#include "util/parallel.h"
#include <cstdio>
#include <string>
#include <type_traits>
#include <cmath>
#include <vector>

/// LE BANC, la partie qui ne depend d'aucun accelerateur.
///
/// Pourquoi ce fichier existe : il y avait UN `main.cpp` de deux mille lignes qui portait a la fois
/// le chargement des nuages, le chronometre, la verification, sept sondes de recherche et le
/// dispatch de dix accelerateurs. Chaque idee nouvelle y ajoutait un drapeau et une branche, et
/// plus rien ne disait quel drapeau allait avec quel accelerateur. Ici chaque accelerateur a SON
/// `main_Xyz.cpp`, qui porte ses options a lui et ses sondes a lui, et tout ce qu'ils ont en commun
/// est ce qui suit.
///
/// La regle, pour qu'un `main_Xyz` soit comparable aux autres : il DEROULE LA SUITE -- les cas
/// importants en 2D et en 3D -- et rien ne l'oblige a s'y limiter.
namespace pd::bench {

double now();

/// Les options COMMUNES. Un `main_Xyz` les parse d'abord, puis les siennes.
struct Args {
    SI   n           = 1000000;   ///< germes du cas « uniforme » (les cas charges ont leur taille)
    int  reps        = 3;         ///< repetitions chronometrees ; on garde le MINIMUM
    int  threads     = 0;         ///< 0 -> autant que de coeurs
    SI   leaf        = 10;        ///< germes par feuille
    int  maxnv       = 0;         ///< 0 -> 32 en 2D, 128 en 3D
    bool cellbox     = true;
    bool skipin      = false;
    Split split      = Split::blocks;
    bool pin         = true;
    unsigned seed    = 0;
    double wscale    = 0;         ///< poids aleatoires du cas uniforme, en fraction de h^2
    std::string load;             ///< UN nuage precis, au lieu de la suite
    int  dims        = 0;         ///< 0 = 2D puis 3D, 2 = 2D seule, 3 = 3D seule

    /// LE DEFAUT EST GENEREUX EN 3D, et c'est mesure : une cellule de Laguerre 3D poissonienne a 27
    /// sommets en moyenne mais une queue qui depasse 64 pour un germe sur 1500 -- a `n = 2e5`,
    /// `--maxnv 64` debordait 135 fois et la somme des volumes tombait a `1.000000001`. Et la borne
    /// ne coute rien : les tableaux ne sont touches que jusqu'a `nb`, donc passer de 128 a 256 n'a
    /// pas change le temps d'une resolution de Newton complete de plus de 1 %.
    int  nv( int D, int conseil = 0 ) const {
        return maxnv ? maxnv : ( conseil ? conseil : ( D == 2 ? 32 : 128 ) );
    }
};

/// rend `true` si l'argument a ete consomme. `val()` livre l'argument suivant.
bool parse_commun( Args &a, const std::string &s, int &i, int argc, char **argv );
void usage_commun();
/// a appeler apres la boucle de parsing : normalise ce qui doit l'etre.
void finalise( Args &a );

/// UN NUAGE, en `D` dimensions. Les coordonnees sont en SoA -- un tableau par axe -- parce que
/// c'est ce que tous les accelerateurs demandent a la construction.
template<int D>
struct Cloud {
    std::string nom;
    SI n = 0;
    std::vector<TF> c[ D ];
    std::vector<TF> w;
    const TF *P[ D ] = {};
    const TF *W = nullptr;        ///< `nullptr` si les poids sont TOUS NULS (le cas euclidien)
    bool absent = false;          ///< le fichier n'existe pas : le cas est saute, pas echoue
    int  nv = 0;                  ///< sommets max CONSEILLES pour ce cas ; 0 = le defaut de la
                                  ///< dimension. Ce n'est pas un reglage esthetique : un nuage dur
                                  ///< a des cellules a bien plus de cotes que l'uniforme, et une
                                  ///< suite qui deborderait sur deux cas sur trois mesurerait des
                                  ///< aires fausses. Chaque cas dit donc ce qu'il lui faut, et
                                  ///< `--maxnv` reste la pour passer outre.

    /// Le tableau de poids, ou `nullptr` s'ils sont tous nuls. Un fichier `voronoi` declare des
    /// poids parce que le format en a une colonne, mais des poids nuls SONT le cas euclidien : les
    /// garder ferait payer au diagramme un majorant identiquement nul (mesure : 12 % en 2D), et
    /// surtout ferait mesurer autre chose que ce qu'on croit comparer.
    void finish() {
        n = SI( c[ 0 ].size() );
        for ( int d = 0; d < D; ++d ) P[ d ] = c[ d ].data();
        W = nullptr;
        for ( TF v : w )
            if ( v != TF( 0 ) ) { W = w.data(); break; }
    }
};

/// Lire un nuage produit par `cases/gen_cases.py` : des lignes `#` de commentaire, `n`, puis `n`
/// fois « x y [z] w ».
///
/// Pourquoi ce chargeur existe : le tirage uniforme est le seul regime ou une cellule est toujours
/// autour de son germe et ou une grille est bien remplie. Un accelerateur peut n'etre juste que
/// sous ces deux hypotheses sans qu'on le voie -- il faut donc pouvoir lui donner un nuage qui les
/// viole, et le meme d'une execution a l'autre.
///
/// `strtod` sur un tampon lu d'un coup, et non `operator>>` : a 1e6 germes le second coute
/// plusieurs secondes -- plus que la mesure qu'on veut chronometrer.
bool load_cloud( const std::string &path, Cloud<2> &cl, bool bavard = true );
bool load_cloud( const std::string &path, Cloud<3> &cl, bool bavard = true );

/// Le nuage UNIFORME dans le cube unite, borne loin des faces pour qu'aucune cellule ne soit
/// degeneree. `wscale` : les poids A L'ECHELLE -- un plan est decale de `dw / ( 2 |p1 - p0| )`, donc
/// pour que le decalage soit une fraction de l'espacement `h ~ n^(-1/D)` il faut `dw ~ h^2`. Des
/// poids « au hasard entre -1 et 1 » videraient presque toutes les cellules.
void uniform_cloud( Cloud<2> &cl, SI n, unsigned seed, double wscale );
void uniform_cloud( Cloud<3> &cl, SI n, unsigned seed, double wscale );

/// LA SUITE : les cas importants, dans l'ordre ou on veut les lire. Un cas dont le fichier manque
/// est rendu avec `absent = true` -- le banc le signale et continue, plutot que de refuser de
/// tourner sur une machine ou `gen_cases.py` n'a pas encore ete lance.
std::vector<Cloud<2>> suite_2d( const Args &a );
std::vector<Cloud<3>> suite_3d( const Args &a );

/// La cellule qui va avec la dimension.
template<int D, int MaxNv>
using CellFor = std::conditional_t<D == 2, CellSoAT<MaxNv>, Cell3T<MaxNv>>;

/// Ce qu'une mesure rend, en plus de sa ligne imprimee.
struct Mesure {
    double t = 0;                 ///< le MINIMUM des repetitions
    double t_build = 0;
    double somme = 0;
    SI     novf = 0;
    bool   ok = false;
};

/// Le corps du banc, une fois l'accelerateur choisi. Le tour de chauffe est HORS chrono, et on
/// garde le MINIMUM : c'est la mesure la moins bruitee d'un temps qu'on veut comparer.
template<class Accel, class Cell, bool CellBox, bool SkipIn, bool Weighted, int D>
Mesure mesure_une( const Args &a, const Cloud<D> &cl ) {
    Accel tree;
    const double t0 = now();
    tree.build( cl.P, cl.W, cl.n, a.leaf );
    Mesure m;
    m.t_build = now() - t0;

    PowerDiagram<Cell, Accel, CellBox, SkipIn, false, Weighted> pd{ tree };
    std::vector<TF> res;
    pd.measures( res, a.threads, a.split, a.pin );          // chauffe

    m.t = 1e300;
    for ( int r = 0; r < a.reps; ++r ) {
        const double t1 = now();
        pd.measures( res, a.threads, a.split, a.pin );
        m.t = std::min( m.t, now() - t1 );
    }
    TF s = 0;
    for ( TF v : res )
        s += v;
    m.somme = double( s );
    m.novf = pd.nb_overflow.load();
    // le SEUL controle du banc, et il suffit : une cellule fausse d'un cote et fausse a l'envers de
    // l'autre est ce qu'il attrape.
    m.ok = std::fabs( m.somme - 1.0 ) < 1e-9 && m.novf == 0;
    return m;
}

void ligne( const Cloud<2> &cl, const Mesure &m, int maxnv );
void ligne( const Cloud<3> &cl, const Mesure &m, int maxnv );

/// LE DISPATCH. Les options sont des parametres de TEMPLATE -- elles doivent disparaitre a la
/// compilation, sans quoi on mesurerait le branchement -- donc on instancie les combinaisons qu'on
/// veut pouvoir choisir a l'execution. `Accel` est deja fixe : c'est ce que le decoupage en un
/// `main` par accelerateur a fait gagner, chaque binaire n'instancie plus que le sien.
template<class Accel, int D>
Mesure mesure_dispatch( const Args &a, const Cloud<D> &cl ) {
    const int nv = a.nv( D, cl.nv );
    auto go = [ & ]( auto cell_tag ) {
        using Cell = decltype( cell_tag );
        auto avec = [ & ]( auto w_tag ) {
            constexpr bool W = decltype( w_tag )::value;
            if ( a.cellbox && ! a.skipin )   return mesure_une<Accel, Cell, true,  false, W, D>( a, cl );
            if ( a.cellbox &&   a.skipin )   return mesure_une<Accel, Cell, true,  true,  W, D>( a, cl );
            if ( ! a.cellbox && ! a.skipin ) return mesure_une<Accel, Cell, false, false, W, D>( a, cl );
            return mesure_une<Accel, Cell, false, true, W, D>( a, cl );
        };
        return cl.W ? avec( std::true_type{} ) : avec( std::false_type{} );
    };
    // les tailles instanciees ne sont pas les memes selon la dimension, et ce n'est pas cosmetique :
    // une cellule de Laguerre 3D a 27 sommets en moyenne la ou la 2D en a 6, donc `16` n'a aucun
    // sens en 3D (le cube de depart en a deja 8) et `256` aucun en 2D.
    if constexpr ( D == 2 ) {
        switch ( nv ) {
            case 12: return go( CellFor<D, 12>{} );
            case 16: return go( CellFor<D, 16>{} );
            case 24: return go( CellFor<D, 24>{} );
            case 48: return go( CellFor<D, 48>{} );
            case 64: return go( CellFor<D, 64>{} );
            default: return go( CellFor<D, 32>{} );
        }
    } else {
        switch ( nv ) {
            case 32:  return go( CellFor<D, 32>{} );
            case 48:  return go( CellFor<D, 48>{} );
            case 96:  return go( CellFor<D, 96>{} );
            case 128: return go( CellFor<D, 128>{} );
            default:  return go( CellFor<D, 64>{} );
        }
    }
}

/// Le marqueur « cet accelerateur n'existe pas dans cette dimension ». Ce n'est pas un aveu : la
/// plupart des accelerateurs de ce banc ont ete ecrits pour repondre a une question 2D precise, et
/// les porter en 3D sans la question serait du travail sans mesure au bout.
struct Absent {
    static constexpr const char *name = "-";
};

template<class Acc, int D>
int deroule( const Args &a, const std::vector<Cloud<D>> &cas ) {
    int bad = 0;
    for ( const Cloud<D> &cl : cas ) {
        if ( cl.absent ) {
            std::printf( "  %-28s : ABSENT (lancer cases/gen_cases.py)\n", cl.nom.c_str() );
            continue;
        }
        const Mesure m = mesure_dispatch<Acc, D>( a, cl );
        ligne( cl, m, a.nv( D, cl.nv ) );
        bad += ! m.ok;
    }
    return bad;
}

/// CE QUE LE PARCOURS A REELLEMENT FAIT, par cellule. Un seul thread, compteurs non atomiques.
///
/// Ce qu'on cherche a distinguer : un nuage difficile peut couter cher pour deux raisons opposees,
/// et elles n'appellent pas le meme remede.
///   * la cellule a BEAUCOUP DE COTES (anisotropie, voisinage etendu) -> `coupes effectives` monte,
///     et c'est un maillage adapte a la forme qui aiderait ;
///   * la cellule est LOIN de son germe -> `boites` monte sans que `coupes effectives` bouge : le
///     parcours cherche longtemps avant de trouver de quoi couper, parce qu'il part du mauvais
///     endroit. Aucun maillage ne corrige ca ; il faut partir d'ailleurs.
template<class Accel, class Cell, bool CellBox>
void stats_une( const Accel &tree ) {
    PowerDiagram<Cell, Accel, CellBox, false, true> pd{ tree };
    const SI n = tree.nb_seeds();
    double sv = 0, sr = 0;
    for ( SI k = 0; k < n; ++k ) {
        Cell c;
        pd.make_cell( c, k );
        sv += c.nb;
        const Vec<Cell::dim> p0 = tree.seed( k );
        TF m = 0;
        for ( SI v = 0; v < c.nb; ++v )
            m = std::max( m, dist2( c.vertex( v ), p0 ) );
        sr += std::sqrt( double( m ) );
    }
    const double d = double( n );
    std::printf( "    %-8s boites %8.1f  balayees %7.1f  gardees %6.1f  coupes tentees %6.1f"
                 "  effectives %5.2f  sommets %5.2f  rayon %.4f\n",
                 Accel::name, pd.st_boxes / d, pd.st_sweep / d, pd.st_kept / d, pd.st_tried / d,
                 pd.st_done / d, sv / d, sr / d );
}

/// LA SUITE COMPLETE, 2D puis 3D. C'est la base d'un `main_Xyz.cpp`.
template<class Acc2, class Acc3>
int banc( const Args &a ) {
    int bad = 0;
    if ( a.dims != 3 ) {
        std::printf( "=== 2D  %-7s box=%d in=%d threads=%d leaf=%d%s\n",
                     a.split == Split::blocks ? "blocks" : "strided",
                     int( a.cellbox ), int( a.skipin ), a.threads, int( a.leaf ),
                     a.maxnv ? "  (--maxnv impose)" : "" );
        if constexpr ( std::is_same_v<Acc2, Absent> )
            std::printf( "  (pas de version 2D)\n" );
        else
            bad += deroule<Acc2, 2>( a, suite_2d( a ) );
    }
    if ( a.dims != 2 ) {
        std::printf( "=== 3D  %-7s box=%d in=%d threads=%d leaf=%d%s\n",
                     a.split == Split::blocks ? "blocks" : "strided",
                     int( a.cellbox ), int( a.skipin ), a.threads, int( a.leaf ),
                     a.maxnv ? "  (--maxnv impose)" : "" );
        if constexpr ( std::is_same_v<Acc3, Absent> )
            std::printf( "  PAS ENCORE : cet accelerateur est 2D par construction.\n" );
        else
            bad += deroule<Acc3, 3>( a, suite_3d( a ) );
    }
    return bad ? 1 : 0;
}

} // namespace pd::bench
