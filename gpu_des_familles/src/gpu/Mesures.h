#pragma once

// =====================================================================================
// CE QUE L'HOTE VOIT : un `DiagrammeGpu<D,TK>` prend l'arbre bati par `accel/AaBsp.h`, le
// televerse une fois, et rend les mesures des cellules INDEXEES PAR L'IDENTIFIANT de l'appelant,
// comme `PowerDiagram::measures`. Pas un type CUDA ici : ce fichier est inclus par du C++
// compile par g++ ( `-march=native` ), l'implementation est dans `Mesures.cu`.
// =====================================================================================

#include "accel/AaBsp.h"
#include <string>
#include <vector>

namespace sf::gpu {

/// le mappage cellule / threads
enum class Variante { FIL, FILREG, FILREGC, FILMIX4, FILMIX6, FILMIX8, FILMIX12, FILMIX16, FILBRK6, FILBRK8, FILBRK10, FILBRK12, FILBRK16, FILBRK8NU, FILROT6, FILROT8, FILNRM8, FILORD8, FILSUC8, FILMSK8, FILMSK8G, FILMSK8F, FILMSK8H, FILMSK8M, FILENT8, FILENT8M, FILMSK8C6, FILMSK8C8, FILNRM8C6, FILNRM8C8, FILUNI8, FILUNI8NP, FILSHM8, FILNRM8TRI, FILNRM8TRIL, FILPH8, FILPH8G, FILPH8B, FILPH8A, FILPH8C, FILPH8O, FILPH8M, FILPH8M4, VOIES, VOIES16, VOIES32, PAQ8x1, PAQ8x2, PAQ8x4, PAQ32x1, PAQ32x2, PAQ32x4, PAQ8x1S, PAQ32x1S, PAQ32x4S, NB };
inline const char *nom( Variante v ) {
    static const char *noms[] = { "fil", "filreg", "filregc", "filmix4", "filmix6", "filmix8", "filmix12", "filmix16", "filbrk6", "filbrk8", "filbrk10", "filbrk12", "filbrk16", "filbrk8nu", "filrot6", "filrot8", "filnrm8", "filord8", "filsuc8", "filmsk8", "filmsk8g", "filmsk8f", "filmsk8h", "filmsk8m", "filent8", "filent8m", "filmsk8c6", "filmsk8c8", "filnrm8c6", "filnrm8c8", "filuni8", "filuni8np", "filshm8", "filnrm8tri", "filnrm8tril", "filph8", "filph8g", "filph8b", "filph8a", "filph8c", "filph8o", "filph8m", "filph8m4", "voies", "voies16", "voies32", "paquet8x1", "paquet8x2", "paquet8x4", "paquet32x1", "paquet32x2", "paquet32x4", "paquet8x1S", "paquet32x1S", "paquet32x4S" };
    return noms[ int( v ) ];
}
/// `paquet V x K` : `V` voies par cellule, `K` cellules par voie, un parcours par warp ( 2D )
inline bool paquet( Variante v ) { return v >= Variante::PAQ8x1 && v < Variante::NB; }

/// COMMENT LA DENSITE IMAGE EST TRAITEE. Le calcul est le meme ( l'integrale de bord de
/// `Image2D.cuh` ) ; ce qui change est OU il a lieu :
///   `DIRECTE`     dans le noyau des cellules, a la suite du parcours de l'arbre ;
///   `DEPOT`       le noyau des cellules ECRIT le polygone fini, un second noyau l'integre,
///                 une voie par cellule -- le depot se fait par lots ( `chunk` cellules ) ;
///   `DEPOT_ARETE` le meme, mais UNE VOIE PAR ARETE : le cout du warp devient le max sur les
///                 aretes au lieu de la somme sur les aretes de la cellule la plus lente.
enum class Densite { AUCUNE, DIRECTE, DEPOT, DEPOT_ARETE };
inline const char *nom( Densite d ) {
    static const char *noms[] = { "aucune", "directe", "depot", "depot-arete" };
    return noms[ int( d ) ];
}

/// LA HESSIENNE assemblee sur la carte : le CSR des hors-diagonaux et la diagonale. Les pointeurs
/// sont DEVICE et appartiennent au diagramme ; `nnz = row[ n ]`.
struct Hessienne {
    const int    *row = nullptr;    ///< `n + 1` bornes de ligne
    const int    *col = nullptr;    ///< `nnz` colonnes, NON TRIEES ( un CG n'en a pas besoin )
    const double *val = nullptr;    ///< `nnz` coefficients `c_ij > 0` ; le signe moins est dans la matrice
    const double *dia = nullptr;    ///< `L_ii`, la somme de la ligne
    int n = 0, nnz = 0;
};

/// LE BILAN D'UN ETAT, en une passe sur les mesures : non plus « une cellule est morte », mais
/// LA DISTRIBUTION -- et rapportee a trois echelles, parce qu'elles ne disent pas la meme chose :
///   `par_cible` : `m / cible`      -- la sante absolue de la cellule ;
///   `par_max`   : `m / max( m )`   -- sa taille relative a la plus grosse du diagramme ;
///   `par_ref`   : `m / m_ref`      -- CE QUE LE PAS LUI A PRIS, par rapport a l'etat de depart.
/// La troisieme est la seule qui distingue « cette cellule etait deja minuscule » de « ce pas est
/// en train de la tuer ». Les deux premieres sont en decades, la troisieme en octaves ; la case 0
/// compte les mesures NULLES, la derniere celles qui valent l'unite ou plus.
struct BilanCel {
    double err = 0;                 ///< `|| m - cible ||`
    double mini = 0, maxi = 0;
    long long vides = 0;
    long long par_cible[ 9 ] = {};  ///< 0 ; < 1e-7 ; 1e-6 ; 1e-5 ; 1e-4 ; 1e-3 ; 1e-2 ; 1e-1 ; >= 1
    long long par_max[ 9 ] = {};    ///< les memes decades, rapportees au maximum
    long long par_ref[ 9 ] = {};    ///< 0 ; < 1/128 ; 1/64 ; 1/32 ; 1/16 ; 1/8 ; 1/4 ; 1/2 ; >= 1
};

/// les chiffres d'un `mesures`
struct Chrono {
    double noyau  = 0;      ///< le noyau seul, en secondes, MINIMUM des repetitions ( evenements CUDA )
    double retour = 0;      ///< la descente des mesures, une fois
    int    deborde = 0;     ///< cellules dont les tampons n'ont pas suffi ( mesure fausse )
    long long stats[ 4 ] = {};   ///< par cellule, si le noyau les compte : coupes tentees, effectives, excursions, boites testees
    int    regs = 0;        ///< registres par thread ( 0 : pas renseigne )
    int    blocs = 0;       ///< blocs simultanes par SM
    double occup = 0;       ///< occupation theorique, en fraction de threads
    int    local = 0;       ///< octets de memoire locale par thread ( la pile, plus les debordements )
};

template<int D, class TK>
struct DiagrammeGpu {
    explicit DiagrammeGpu( const AaBspT<D> &arbre );

    /// L'ARBRE CONSTRUIT SUR LE GPU, de bout en bout : rien ne redescend ( 2D ). `P[ d ][ i ]` les
    /// positions, `W` les poids ou `nullptr`. `ms` rend le temps GPU de la construction.
    DiagrammeGpu( const double *const *P, const double *W, int n, int leaf, double *ms = nullptr,
                  bool pour_newton = false );

    /// DES POIDS NEUFS SUR LE MEME ARBRE ( regime de Newton ) : seuls les majorants sont refaits,
    /// ni tri ni boites ni permutation, et aucune allocation. Rend le temps GPU en ms.
    /// Demande `pour_newton` a la construction.
    double refresh_poids( const double *W );

    /// UN TOUR DE NEWTON cote GPU : poids neufs, majorants refaits, puis mesures ET facettes --
    /// rien ne redescend, pas une allocation. Rend le temps GPU en ms. C'est le cout de regime.
    double tour_newton( const double *W );

    /// L'ASSEMBLAGE de la hessienne depuis les facettes deja sur la carte : compter, scanner,
    /// remplir. Rend le temps GPU en ms. Demande un `tour_newton` ou un `facettes` avant.
    double assemble( Hessienne &H );

    /// LE GRADIENT CONJUGUE PRECONDITIONNE ( Jacobi ), sur le systeme reduit ( jauge `x[ 0 ] = 0` ).
    /// `b` et `x` sont DEVICE, `x` est rendu. Rend le nombre d'iterations, `-1` si pas convergé ;
    /// `ms` le temps GPU, `res` le residu relatif final.
    int resout( const Hessienne &H, const double *b, double *x, double tol, int maxit,
                double *ms = nullptr, double *res = nullptr );

    /// `|| m - cible ||` ( norme deux ) et, par `mini`, LA PLUS PETITE MESURE -- c'est elle qui
    /// dit si une cellule est sur le point de disparaitre, donc si le pas est trop grand.
    /// `b` ( device, `n` doubles ) recoit `m - cible` CENTRE, le second membre du pas de Newton.
    ///
    /// `nb_cond`, s'il est donne, recoit LE NOMBRE DE CELLULES SOUS `seuil` -- et leur liste reste
    /// sur la carte. C'est ce compte qui dit si un pas refuse l'est pour TROIS cellules ou pour
    /// dix mille : dans le premier cas il n'y a pas a raboter le pas, il y a a relever ces trois.
    /// `nb_vides` compte a part celles dont la mesure est NULLE : une cellule sous le seuil se
    /// releve, une cellule nulle a deja disparu.
    double residu( double cible, double *mini, double *b = nullptr,
                   double seuil = 0, int *nb_cond = nullptr, int *nb_vides = nullptr ) const;

    /// `alpha*` : LE PAS EXACT ou la premiere cellule touche `seuil`, par le POLYNOME de l'aire
    /// ( `Alpha2D.cuh` ). Le long de `w - t d` l'aire de chaque cellule est un polynome de degre
    /// deux EXACT tant que la combinatoire ne change pas, donc `alpha*_i` est une racine en forme
    /// close et `alpha*` une reduction -- il n'y a rien a essayer.
    ///
    /// `d` est la direction dans l'ordre DE L'APPELANT ( `n` doubles, hote ). Demande un
    /// `tour_newton` juste avant : c'est SA connectivite qui sert, et cette passe ne reparcourt
    /// pas l'arbre. Rend `alpha*` ( `1e300` si aucune cellule ne contraint ), et par `ms` le
    /// temps GPU.
    ///
    /// `pol`, s'il est donne, recoit `( a0, a1, a2 )` par cellule ( `3 n`, en SoA, indexes par
    /// identifiant ). `a0` doit valoir la mesure et `a1` valoir `-( L d )` : c'est le controle
    /// exact de toute la chaine.
    /// `conf` : le RAYON DE CONFIANCE du polynome, en fractions du rayon de cellule ( 0 : off ).
    /// `nu`, s'il est donne, recoit le `alpha*` NU -- celui d'avant la region de confiance --
    /// de quoi mesurer si elle mord. `pol` recoit `( a0, a1, a2, t_conf )`, soit `4 n`.
    double limites( const double *d, double seuil, double conf = 0, double *ms = nullptr,
                    double *nu = nullptr, std::vector<double> *pol = nullptr );

    /// la hierarchie du multigrille, montee depuis `H` ( le motif ne change pas dans un Newton,
    /// seuls les coefficients -- a remonter quand ils bougent )
    void monte_amg( const Hessienne &H );
    void cycle_v( int niveau );
    /// le lissage d'un niveau : Jacobi amorti ou Chebyshev, selon `AMG_LISSEUR`
    void lisse_un( int niveau, int nb, bool net );

    /// `y = L x` sur la carte ( pointeurs device ), pour verifier et pour le gradient conjugue
    void applique( const Hessienne &H, const double *x, double *y ) const;

    /// LA DENSITE IMAGE : `v` les `W * H` valeurs de la grille ( ligne par ligne, elle couvre
    /// `[ 0, 1 ]^2` ), montees UNE FOIS avec la somme prefixe de chaque ligne. Rien n'est
    /// normalise ici : l'appelant decide de la masse totale, qui est rendue. `v = nullptr`
    /// libere l'image et le diagramme revient a Lebesgue.
    /// Des lors, `res` est la MASSE de la cellule et `fac_l` est `integrale_facette rho ds` --
    /// c'est ce que la hessienne du transport demande quand la source n'est pas uniforme.
    double charge_image( const double *v, int W, int H );

    /// GARDE les mesures courantes comme etat de reference du prochain `bilan` ( une copie sur la
    /// carte ). A appeler AVANT d'essayer un pas, pour que `par_ref` ait un sens.
    void garde_mesures();

    /// le bilan complet de l'etat courant. Trois passes triviales sur `n` doubles.
    BilanCel bilan( double cible ) const;

    /// le mode de traitement, et la taille des lots du depot ( en cellules )
    void regle_densite( Densite d, int chunk = 1 << 20 );
    Densite densite() const;
    ~DiagrammeGpu();
    DiagrammeGpu( const DiagrammeGpu & ) = delete;
    DiagrammeGpu &operator=( const DiagrammeGpu & ) = delete;

    int nb_noeuds() const { return nn_pub; }
    double televersement() const { return t_tele; }     ///< le temps de la montee, en secondes

    /// `res[ id ]` pour chaque germe ; `maxnv` : 64 | 128 en 2D, 64 | 128 en 3D. Un tour de chauffe,
    /// puis `reps` tours chronometres.
    Chrono mesures( Variante v, int maxnv, int reps, std::vector<double> &res ) const;

    /// LES MESURES ET LES FACETTES ( 2D ) : le noyau `filmsk8f` rend en plus, pour chaque cellule,
    /// jusqu'a `NF` aretes -- `fj[ s * n + i ]` l'identifiant du voisin ( `< 0` : un cote de la
    /// boite, ou une case vide ) et `fl[ s * n + i ]` la longueur. De quoi assembler la hessienne
    /// du Newton sans repasser par le CPU.
    static constexpr int NF = 32;   ///< aretes gardees par cellule ( le polygone final en a 6 en moyenne )
    Chrono facettes( int reps, std::vector<double> &res, std::vector<int> &fj, std::vector<double> &fl ) const;

    struct Impl;
    Impl  *impl;
    double t_tele = 0;
    int    nn_pub = 0;
};

/// le nom de la carte, pour l'en-tete
std::string carte();

} // namespace sf::gpu
