// L'ETAPE 2 : pour chaque dirac sa cellule contre l'agregat et les medians voisins, la recette
// pour la rejouer, et l'ENCEINTE de l'agregat -- la sur-cellule.
//
// LA SUR-CELLULE EST CALCULEE, PAS APPROCHEE. Les cellules des membres de `A`, coupees par LE MEME
// jeu de plans (`A u ext( A )`), pavent exactement
//
//     U_A = { x : min_{i in A} h_i( x ) <= min_{e in ext( A )} h_e( x ) }
//
// et `U_A` contient la reunion des vraies cellules de `A`, parce que `ext( A )` est un
// sous-ensemble des germes et qu'ajouter des germes ne peut que retrecir. Aucune borne de poids
// n'intervient : ni pseudo-germe, ni cran `omega`, ni sous-decoupage.

#pragma once

#include "supercell/Agregats.h"
#include "supercell/Enceinte.h"
#include "supercell/Memoire.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace pd::supercell {

using namespace pd::bench;

/// La cellule des etapes 2 et 4. Une seule definition : les deux doivent produire le MEME objet,
/// sans quoi la recette rejouee ne serait pas celle qui a ete ecrite.
///
/// `256` en 3D et pas `128`, parce que l'etape 4 coupe PAR AGREGAT : les plans arrivent groupes
/// par voisin et non par distance, donc la cellule passe par des etats transitoires bien plus
/// larges que sa forme finale ( 27 sommets en moyenne ). A 128 elle debordait, `cut` laissait la
/// cellule intacte, et la somme des mesures montait a 1.0026 -- le temoin l'a dit tout de suite.
/// Trier les agregats retenus par distance ramenerait ce transitoire, et c'est l'optimisation a
/// faire ; en attendant on paie la place. Pas plus de `256` : `Cell3T` veut `ht_size`
/// strictement superieur a son nombre de faces.
template<int D> using CellSC = CellFor<D, D == 2 ? 32 : 256>;

/// COMMENT ON CHOISIT LES OPPOSANTS. A la compilation, evidemment.
///
///   `Median` : un germe par agregat voisin, jusqu'a l'anneau 2. Peu de plans, bonne couverture
///              ANGULAIRE -- c'est elle qui borne la sur-cellule, pas le nombre d'opposants.
///   `Tous`   : tous les membres, jusqu'a l'anneau demande. Mesure : a l'anneau 1 c'est
///              catastrophique ( 48 diracs groupes dans 6 agregats laissent des secteurs ouverts,
///              la table explose et l'etape 4 passe a 8.7 s ) ; a l'anneau 2 la qualite est
///              excellente mais l'etape 2 coute 3.1 s au lieu de 0.9.
///   `Mixte`  : TOUT l'anneau 1, plus les MEDIANS de l'anneau 2. On ferme les secteurs
///              intermediaires au prix d'un plan par agregat lointain, et surtout `C_i` contient
///              alors TOUS les plans de l'anneau 1 -- donc l'etape 4 n'a plus a les repasser, ce
///              qui retire d'un coup les 90 % de paires utiles que l'anneau 1 porte en 2D.
enum class Opposants { Median, Tous, Mixte };

/// LES OPPOSANTS de l'agregat `a`. `n1` rend le nombre d'agregats du PREMIER anneau, dont
/// l'etape 4 se sert pour les sauter.
///
/// Extrait en fonction parce que l'etape 4 doit reconstruire EXACTEMENT la meme liste que
/// l'etape 2 : les indices de la recette designent des places dans cette liste, et une place qui
/// glisse d'un cran rend la recette silencieusement fausse.
template<int D, Opposants Mode>
void opposes( const Gros<D> &G, SI a, int anneau, std::vector<SI> &vus, std::vector<SI> &ext,
              SI &n1 ) {
    vus.clear();
    for ( SI u = G.adeb[ a ]; u < G.adeb[ a + 1 ]; ++u ) {
        const SI v = G.adj[ u ];
        if ( v != a && std::find( vus.begin(), vus.end(), v ) == vus.end() ) vus.push_back( v );
    }
    n1 = SI( vus.size() );
    const int nr = Mode == Opposants::Mixte ? 2 : anneau;
    for ( int r = 1; r < nr; ++r ) {
        const SI m = SI( vus.size() );
        for ( SI q = 0; q < m; ++q )
            for ( SI u = G.adeb[ vus[ q ] ]; u < G.adeb[ vus[ q ] + 1 ]; ++u ) {
                const SI v = G.adj[ u ];
                if ( v != a && std::find( vus.begin(), vus.end(), v ) == vus.end() ) vus.push_back( v );
            }
    }
    // LES BLOCS, DU PLUS PROCHE AU PLUS LOIN. L'ordre ne change pas le resultat mais tout le
    // rendement du crible : le rayon `R` de la cellule ne decroit que lorsqu'on coupe, donc
    // commencer par les agregats lointains laisse `R` grand et le critere ne rejette rien. On trie
    // les DEUX anneaux separement pour que `n1` continue de designer le premier -- l'etape 4 s'en
    // sert pour le sauter.
    //
    // Le tri porte sur ~25 agregats et se fait UNE FOIS par agregat, donc amorti sur ses `rho`
    // membres : trier les 72 candidats par membre couterait bien plus que les coupes economisees.
    {   auto dm = [ & ]( SI v ) {
            Vec<D> x, y;
            for ( int d = 0; d < D; ++d ) { x[ d ] = G.Ppp[ d ][ G.median[ a ] ];
                                            y[ d ] = G.Ppp[ d ][ G.median[ v ] ]; }
            return dist2( x, y );
        };
        const auto par_distance = [ & ]( SI x, SI y ) { return dm( x ) < dm( y ); };
        std::sort( vus.begin(), vus.begin() + n1, par_distance );
        std::sort( vus.begin() + n1, vus.end(), par_distance );
    }
    ext.clear();
    for ( SI q = 0; q < SI( vus.size() ); ++q ) {
        const SI v = vus[ q ];
        const bool tout = Mode == Opposants::Tous || ( Mode == Opposants::Mixte && q < n1 );
        if ( tout ) for ( SI u = G.mdeb[ v ]; u < G.mdeb[ v + 1 ]; ++u ) ext.push_back( u );
        else if ( G.median[ v ] >= 0 ) ext.push_back( G.median[ v ] );
    }
}

/// LA LISTE DE CANDIDATS du dirac `i` de l'agregat `a` : les membres sauf lui, PUIS les medians.
/// L'ordre est l'ordre CANONIQUE -- le candidat `j` porte le meme numero pour tous les membres de
/// `a`, donc le bit `j` de la recette veut dire la meme chose partout.
template<int D>
void liste_candidats( const Gros<D> &G, SI a, const std::vector<SI> &ext,
                      std::vector<SI> &cand ) {
    cand.clear();
    for ( SI v = G.mdeb[ a ]; v < G.mdeb[ a + 1 ]; ++v ) cand.push_back( v );
    for ( SI e : ext ) cand.push_back( e );
}

/// LES TAMPONS SoA d'un agregat : les coordonnees des candidats, une composante par tableau.
///
/// Ils ne dependent PAS du membre courant -- la liste est celle de l'agregat entier -- donc ils se
/// remplissent une fois pour ses `rho` membres. C'est ce que la permutation du nuage rend possible :
/// les candidats viennent de quelques PLAGES contigues, pas d'indices disperses.
///
/// Et c'est la forme que le vectoriseur prend : `d2[ j ] = somme_d ( b[ d ][ j ] - p0[ d ] )^2` est
/// une boucle sur des tableaux alignes, sans indirection, sans branchement.
template<int D>
struct TamponSoA {
    std::vector<TF> b[ D ], w, d2;
    void remplit( const Gros<D> &G, const std::vector<SI> &cand, bool pese ) {
        const SI nc = SI( cand.size() );
        for ( int d = 0; d < D; ++d ) {
            b[ d ].resize( nc );
            for ( SI j = 0; j < nc; ++j ) b[ d ][ j ] = G.Ppp[ d ][ cand[ j ] ];
        }
        w.resize( nc );
        if ( pese ) for ( SI j = 0; j < nc; ++j ) w[ j ] = G.Wp[ cand[ j ] ];
        else        std::fill( w.begin(), w.end(), TF( 0 ) );
        d2.resize( nc );
    }
    /// LES DISTANCES, en un passage vectorisable.
    void distances( Vec<D> p0, SI j0, SI j1 ) {
        const TF *bp[ D ];
        for ( int d = 0; d < D; ++d ) bp[ d ] = b[ d ].data();
        TF *o = d2.data();
        for ( SI j = j0; j < j1; ++j ) {
            TF s = 0;
            for ( int d = 0; d < D; ++d ) { const TF e = bp[ d ][ j ] - p0[ d ]; s += e * e; }
            o[ j ] = s;
        }
    }
};

/// LES STATISTIQUES DE L'ETAPE 2, pour repondre a deux questions qu'on ne peut pas deviner :
///
///   * COMBIEN DE SOMMETS a une cellule, et comment ce nombre evolue au fil des coupes. Si la
///     distribution est concentree, on peut engendrer du code specialise par taille -- avec les
///     sommets dans des registres SIMD de largeur EXACTE.
///   * QUELLE EST LA CHANCE qu'une coupe soit EFFECTIVE en fonction de son rang, maintenant que
///     les candidats arrivent a peu pres tries par distance. Une chute brutale dirait ou arreter,
///     ou a partir d'ou tester en parallele sans se soucier du resultat.
struct StatsE2 {
    static constexpr int MAXQ = 48, MAXNB = 24;
    long long testes[ MAXQ ] = {}, effectifs[ MAXQ ] = {}, nb_apres[ MAXQ ] = {};
    long long hist_final[ MAXNB ] = {}, hist_evol[ MAXNB ] = {};
    long long cellules = 0;
};

/// LE CRITERE DE SECURITE, avec les poids et sans racine carree.
///
/// `j` ne peut couper une cellule de rayon `R` autour de `p_i` que si un sommet `v` verifie
/// `|v - p_j|^2 - w_j < |v - p_i|^2 - w_i`. Comme `|v - p_j| >= d - R` avec `d = |p_i - p_j|`, une
/// condition SUFFISANTE de non-coupe est `d ( d - 2 R ) >= w_j - w_i`. En posant
/// `t = d^2 - ( w_j - w_i )`, elle devient `t >= 0 et t^2 >= 4 d^2 R^2` : deux multiplications et
/// deux comparaisons, pas de racine, pas de branchement -- la forme qu'un vectoriseur prend.
///
/// Ce n'est PAS reserve au cas Voronoi. Ce que l'arbre du depot ajoute, c'est un MAJORANT des
/// poids d'un sous-arbre entier ; sur une liste plate on a le `w_j` de chaque candidat, donc le
/// majorant ne sert a rien.
inline bool hors_portee( TF d2, TF dw, TF r2 ) {
    const TF t = d2 - dw;
    return t >= 0 && t * t >= 4 * d2 * r2;
}

/// LE RAYON CARRE d'une cellule autour de son germe.
template<int D, class Cell>
inline TF rayon2( const Cell &c, Vec<D> p0 ) {
    TF r2 = 0;
    for ( SI v = 0; v < c.nb; ++v ) {
        Vec<D> x;
        if constexpr ( D == 2 ) { x[ 0 ] = c.vx[ v ]; x[ 1 ] = c.vy[ v ]; }
        else                      x = c.vertex( v );
        r2 = std::max( r2, dist2( p0, x ) );
    }
    return r2;
}

/// La coupe de Laguerre de `p0, w0` par `p1, w1`, dans la convention de `PowerDiagram`.
template<int D, bool Weighted>
inline void plan( const Gros<D> &G, SI i, Vec<D> p0, TF w0, Vec<D> &dir, TF &off ) {
    Vec<D> p1;
    for ( int d = 0; d < D; ++d ) p1[ d ] = G.Ppp[ d ][ i ];
    const TF w1 = Weighted ? G.Wp[ i ] : TF( 0 );
    off = ( w0 - w1 ) / 2;
    for ( int d = 0; d < D; ++d ) {
        dir[ d ] = p1[ d ] - p0[ d ];
        off += dir[ d ] * ( p0[ d ] + p1[ d ] ) / 2;
    }
}

// ------------------------------------------------------------------ L'ETAPE 2

struct BilanSC {
    double somme_c = 0;           ///< somme des |C_i| sur TOUS les diracs -- doit valoir >= 1
    double somme_h = 0;           ///< somme des volumes d'enveloppe -- >= somme_c
    double nv = 0, nv_ext = 0;    ///< sommets par cellule, et ceux qui sont sur le bord de U_A
    double nc = 0;                ///< candidats presentes par dirac
    double nc_reel = 0;           ///< ... et ceux reellement TENTES apres le crible
    double nf_int = 0, nf_ext = 0;///< facettes portees par un membre / par un median exterieur
    double nh = 0;                ///< sommets par enveloppe
    double octets = 0;
    SI     novf = 0, perdus = 0, env_deg = 0, env_ko = 0, rejeu_ko = 0;
    double rejeu_max = 0;         ///< le pire ecart RELATIF du rejeu
    double env_max = 0;           ///< de combien l'enveloppe rate le pire point, en `h`
    double nv_rejeu = 0;          ///< coupes qu'il faut rejouer, contre `nc` essayees
    double t_cell = 0, t_env = 0, t_vol = 0, t_rejeu = 0;
    bool   ok = false;
};

/// LA PASSE 1 DU SCHEMA : pour chaque dirac, la cellule contre son agregat et les medians voisins.
///
/// La liste de candidats ne depend QUE de l'agregat -- `A \ { i }` puis les medians de `ext( A )`.
/// C'est ce qui fait que les cellules des membres pavent `U_A` (voir `Enveloppe`), et c'est aussi
/// ce qui rend l'ordre CANONIQUE : le candidat `j` a le meme numero pour tous les membres, donc le
/// bit `j` veut dire la meme chose partout et la recette se relit sans table.
///
/// Aucune borne de poids n'intervient. La sur-cellule n'est pas approchee par un minorant : elle
/// est CALCULEE, exactement, contre l'echantillon choisi. C'est tout ce que le schema des agregats
/// demandait et ca dispense du cran `omega`, des pseudo-germes et du sous-decoupage.
template<int D, bool Weighted, Stockage S, class Enc, Opposants Mode>
BilanSC etape2( const Cloud<D> &cl, const Gros<D> &G, int anneau, bool verif,
                StatsE2 *st = nullptr,
                std::vector<Enc> *garde = nullptr,
                Memoire<S, CellSC<D>, D> *garde_mem = nullptr ) {
    const double h_esp = std::pow( 1.0 / double( cl.n ), 1.0 / D );
    using Cell = CellSC<D>;
    BilanSC b;
    Memoire<S, Cell, D> mem;
    mem.reserve( cl.n );

    std::vector<SI>    cand, ext, vus, fa, fb;
    TamponSoA<D>       tam;
    std::vector<std::pair<TF,SI>> ordre;
    std::vector<Vec<D>> bord, pts;
    std::vector<SI>    relu;
    Enc                env;
    if ( garde ) garde->assign( G.ns, Enc{} );
    double t_cell = 0, t_env = 0, t_vol = 0, t_rejeu = 0;
    long long tot_nc_reel = 0;
    long long tot_nv = 0, tot_ext = 0, tot_nc = 0, tot_fi = 0, tot_fe = 0, tot_nh = 0, tot_rj = 0;

    for ( SI a = 0; a < G.ns; ++a ) {
        const SI d0 = G.mdeb[ a ], d1 = G.mdeb[ a + 1 ];
        if ( d0 == d1 ) continue;
        const SI na = d1 - d0;

        // --- les candidats EXTERIEURS : un median par cellule grossiere qui touche
        // `--anneau 2` ajoute les voisins des voisins. Ce n'est pas un raffinement cosmetique :
        // avec les seuls voisins DIRECTS l'opposition ne compte que ~6 points en 2D, et il reste
        // des directions ou un membre de `A` bat les six -- la sur-cellule part alors tres loin.
        SI n1 = 0;
        opposes<D, Mode>( G, a, anneau, vus, ext, n1 );
        liste_candidats<D>( G, a, ext, cand );          // STABLE : la meme pour tous les membres
        tam.remplit( G, cand, G.pese );
        const SI nc = SI( cand.size() );
        const SI i0_ext = na;                           // ou commencent les opposants

        double t = now();
        bord.clear();
        for ( SI u = d0; u < d1; ++u ) {
            const SI i = u;                             // indice PERMUTE : la plage est contigue
            Vec<D> p0;
            for ( int d = 0; d < D; ++d ) p0[ d ] = G.Ppp[ d ][ i ];
            const TF w0 = Weighted ? G.Wp[ i ] : TF( 0 );

            Cell c;
            if constexpr ( D == 2 ) c.init_as_unit_square();
            else                    c.init_as_unit_cube();
            // LES MEMBRES D'ABORD ( ce sont les plus proches ), puis les opposants PASSES AU
            // CRIBLE. La liste est deja rangee dans cet ordre, donc il n'y a rien a trier : on
            // coupe les `|A| - 1` membres, on releve le rayon, et on ecarte les opposants hors de
            // portee. L'etape 2 tentait 27.7 coupes pour une cellule qui finit a 5.44 facettes.
            bool ovf = false;
            SI tentees = 0;

            // LES DISTANCES D'UN COUP, sur TOUTE la liste. Le tampon est en SoA et ne depend pas
            // du membre -- seule `p0` change -- donc c'est un FMA sur des tableaux contigus, sans
            // indirection ni branchement.
            tam.distances( p0, 0, nc );
            const TF *dd = tam.d2.data(), *ww = tam.w.data();

            // LES MEMBRES DE L'AGREGAT, COUPES D'OFFICE. On a essaye de les cribler eux aussi
            // -- les trier par distance, en couper `D + 1` pour obtenir un rayon, soumettre le
            // reste au critere. Ca marche ( 13.5 -> 12.1 tentatives en `median`, 27.2 -> 25.8 en
            // `mixte` ) et ca COUTE PLUS CHER : 0.712 -> 0.792 s et 1.257 -> 1.329 s. Le tri des
            // huit membres et la construction des paires depassent les 1.4 coupes economisees.
            //
            // C'est la meme lecon que partout sur ce noyau : une coupe qui NE COUPE PAS ne fait
            // que la boucle de projection de `cut` et sort, soit une douzaine de flops -- a peine
            // plus que le critere lui-meme. On ne gagne pas a filtrer plus finement, seulement a
            // presenter moins de candidats.
            SI rang = 0;
            for ( SI j = 0; j < i0_ext; ++j ) {
                if ( j == i - d0 ) continue;                 // son propre plan : `dir` serait nul
                Vec<D> dir; TF off;
                plan<D, Weighted>( G, cand[ j ], p0, w0, dir, off );
                ++tentees;
                const CutResult r = c.cut( dir, off, j );
                if ( r == CutResult::overflow ) ovf = true;
                if ( st ) {
                    const int q = int( std::min<SI>( rang, StatsE2::MAXQ - 1 ) );
                    ++st->testes[ q ];
                    st->effectifs[ q ] += ( r == CutResult::done );
                    st->nb_apres[ q ] += c.nb;
                    ++st->hist_evol[ std::min<SI>( c.nb, StatsE2::MAXNB - 1 ) ];
                }
                ++rang;
            }
            TF r2 = rayon2<D>( c, p0 );

            // LES OPPOSANTS, dans l'ordre des blocs deja tries par distance.
            for ( SI j = i0_ext; j < nc; ++j ) {
                const TF dw = Weighted ? ww[ j ] - w0 : TF( 0 );
                if ( hors_portee( dd[ j ], dw, r2 ) )
                    continue;
                Vec<D> dir; TF off;
                plan<D, Weighted>( G, cand[ j ], p0, w0, dir, off );
                ++tentees;
                if ( st ) {
                    const int q = int( std::min<SI>( rang, StatsE2::MAXQ - 1 ) );
                    ++st->testes[ q ];
                    st->nb_apres[ q ] += c.nb;
                }
                ++rang;
                // ON RELEVE `r2` APRES CHAQUE COUPE : le geler faisait remonter les tentatives de
                // 13.2 a 22.5 et le temps de 0.866 a 0.889 s. Le passage sur les ~6 sommets coute
                // moins que les coupes qu'il fait rejeter.
                const CutResult r = c.cut( dir, off, j );
                if ( r == CutResult::overflow ) ovf = true;
                else                            r2 = rayon2<D>( c, p0 );
                if ( st ) {
                    const int q = int( std::min<SI>( rang - 1, StatsE2::MAXQ - 1 ) );
                    st->effectifs[ q ] += ( r == CutResult::done );
                    ++st->hist_evol[ std::min<SI>( c.nb, StatsE2::MAXNB - 1 ) ];
                }
            }
            if ( st ) { ++st->cellules; ++st->hist_final[ std::min<SI>( c.nb, StatsE2::MAXNB - 1 ) ]; }
            tot_nc_reel += tentees;
            b.novf += ovf;
            b.somme_c += double( c.measure() );
            tot_nv += c.nb;
            tot_nc += cand.size();

            // les facettes, et d'ou elles viennent -- c'est ce qui dit si les medians exterieurs
            // suffisent a fermer la cellule ou si l'agregat la ferme tout seul.
            c.for_each_facet( [ & ]( SI j, TF ) { if ( j < i0_ext ) ++tot_fi; else ++tot_fe; } );

            // LE FILTRE DES SOMMETS EXTERIEURS : un sommet dont TOUTES les coupes sont internes
            // est partage par plusieurs cellules de l'agregat, donc interieur a `U_A`.
            for ( SI v = 0; v < c.nb; ++v ) {
                bool dehors = false;
                if constexpr ( D == 2 ) {
                    const SI u1 = c.cid[ v ], u0 = c.cid[ ( v + c.nb - 1 ) % c.nb ];
                    dehors = u1 < 0 || u1 >= i0_ext || u0 < 0 || u0 >= i0_ext;
                } else {
                    for ( int e = 0; e < 3; ++e )
                        dehors |= c.vc[ v ][ e ] < 0 || c.vc[ v ][ e ] >= i0_ext;
                }
                if ( dehors ) {
                    ++tot_ext;
                    Vec<D> x;
                    if constexpr ( D == 2 ) { x[ 0 ] = c.vx[ v ]; x[ 1 ] = c.vy[ v ]; }
                    else                      x = c.vertex( v );
                    bord.push_back( x );
                }
            }
            mem.note( i, c, nc );
        }
        t_cell += now() - t;

        // --- L'ENCEINTE : la sur-cellule convexifiee. Enveloppe exacte ou k-DOP, au choix du
        //     parametre `Enc` -- c'est le seul endroit du fichier qui les distingue.
        t = now();
        pts = bord;
        env.build( pts );
        t_env += now() - t;
        if ( garde ) ( *garde )[ a ] = env;              // l'etape 3 en aura besoin
        if ( ! env.ok ) {
            ++b.env_deg;                                // on n'a pas su la construire : on COMPTE
        } else {
            tot_nh += SI( env.taille() );
            t = now();
            b.somme_h += env.volume();                  // TEMOIN, donc chronometre a part
            t_vol += now() - t;
            if ( verif ) b.env_max = std::max( b.env_max, double( env.ecart( bord ) ) / h_esp );
        }

        // --- LE REJEU : refaire la cellule avec la SEULE recette, et retrouver la meme mesure.
        if ( verif && mem.rejouable() ) {
            t = now();
            for ( SI u = d0; u < d1; ++u ) {
                const SI i = u;                     // indice PERMUTE, comme la boucle principale
                mem.relis( i, nc, relu );
                if ( relu.empty() ) continue;
                Vec<D> p0;
                for ( int d = 0; d < D; ++d ) p0[ d ] = G.Ppp[ d ][ i ];
                const TF w0 = Weighted ? G.Wp[ i ] : TF( 0 );
                Cell c;
                if constexpr ( D == 2 ) c.init_as_unit_square();
                else                    c.init_as_unit_cube();
                for ( SI j : relu ) {
                    Vec<D> p1;
                    for ( int d = 0; d < D; ++d ) p1[ d ] = G.Ppp[ d ][ cand[ j ] ];
                    const TF w1 = Weighted ? G.Wp[ cand[ j ] ] : TF( 0 );
                    Vec<D> dir;
                    TF off = ( w0 - w1 ) / 2;
                    for ( int d = 0; d < D; ++d ) {
                        dir[ d ] = p1[ d ] - p0[ d ];
                        off += dir[ d ] * ( p0[ d ] + p1[ d ] ) / 2;
                    }
                    c.cut( dir, off, j );
                }
                tot_rj += SI( relu.size() );
                // on refait la cellule pleine pour comparer : c'est une VERIFICATION, le cout
                // n'entre pas dans le chronometre de la passe.
                Cell r;
                if constexpr ( D == 2 ) r.init_as_unit_square();
                else                    r.init_as_unit_cube();
                for ( SI j = 0; j < SI( cand.size() ); ++j ) {
                    if ( j == i - d0 ) continue;
                    Vec<D> p1;
                    for ( int d = 0; d < D; ++d ) p1[ d ] = G.Ppp[ d ][ cand[ j ] ];
                    const TF w1 = Weighted ? G.Wp[ cand[ j ] ] : TF( 0 );
                    Vec<D> dir;
                    TF off = ( w0 - w1 ) / 2;
                    for ( int d = 0; d < D; ++d ) {
                        dir[ d ] = p1[ d ] - p0[ d ];
                        off += dir[ d ] * ( p0[ d ] + p1[ d ] ) / 2;
                    }
                    r.cut( dir, off, j );
                }
                // LE CRITERE EST COMBINATOIRE, et pas metrique. Comparer les aires ferait echouer
                // des cellules en lame ou l'aire est une difference de grands termes -- on y lisait
                // 2.4e-09 d'ecart relatif sans qu'aucune coupe ne differe. Ce qu'on veut savoir est
                // si la recette rend LES MEMES FACETTES ; l'ecart d'aire reste imprime, comme
                // information sur le bruit d'arrondi du rejeu, mais il ne juge rien.
                fa.clear(); fb.clear();
                r.for_each_facet( [ & ]( SI j, TF ) { fa.push_back( j ); } );
                c.for_each_facet( [ & ]( SI j, TF ) { fb.push_back( j ); } );
                std::sort( fa.begin(), fa.end() ); std::sort( fb.begin(), fb.end() );
                if ( fa != fb ) ++b.rejeu_ko;
                const double m0 = double( r.measure() ), m1 = double( c.measure() );
                b.rejeu_max = std::max( b.rejeu_max, std::fabs( m0 - m1 ) / std::max( 1e-30, m0 ) );
            }
            t_rejeu += now() - t;
        }
    }

    const double n = double( cl.n ), ns = double( G.ns );
    b.nv = tot_nv / n;  b.nv_ext = tot_ext / n;  b.nc = tot_nc / n;
    b.nc_reel = tot_nc_reel / n;
    b.nf_int = tot_fi / n;  b.nf_ext = tot_fe / n;
    b.nh = tot_nh / std::max( 1.0, ns - b.env_deg );
    b.nv_rejeu = tot_rj / n;
    b.octets = mem.octets( cl.n );
    b.perdus = mem.perdus;
    b.t_cell = t_cell; b.t_env = t_env; b.t_vol = t_vol; b.t_rejeu = t_rejeu;
    if ( garde_mem ) *garde_mem = std::move( mem );      // l'etape 4 rejouera avec

    // LES CONTROLES. `somme_c >= 1` : les `U_A` recouvrent le domaine, donc la somme de leurs
    // mesures majore la sienne -- et l'EXCES est exactement le recouvrement, c'est-a-dire le prix
    // des sur-cellules. `somme_h >= somme_c` : convexifier ne peut qu'ajouter. Et le rejeu doit
    // rendre la meme cellule, sans quoi la recette ne vaut rien.
    b.ok = b.somme_c > 1.0 - 1e-9 && b.somme_h > b.somme_c - 1e-9
           && b.novf == 0 && b.rejeu_ko == 0 && b.env_max < 1e-6;
    return b;
}

template<int D>
void ligne_surcellules( const Cloud<D> &cl, const BilanSC &b ) {
    std::printf( "      %-20s cand %5.1f ->%5.1f tentes -> sommets %5.2f dont %5.2f ext | facettes %4.2f int"
                 " %4.2f ext | enc %5.2f | S|C| %7.4f  S|env| %7.4f"
                 " | rejeu %4.2f coupes (ecart %.1e, %d ko) | %5.1f o/dirac  perdus %d"
                 " | enc ko %d ecart %.1e h | cell %6.3f  env %6.3f  vol %6.3f s | %s\n",
                 cl.nom.c_str(), b.nc, b.nc_reel, b.nv, b.nv_ext, b.nf_int, b.nf_ext, b.nh,
                 b.somme_c, b.somme_h, b.nv_rejeu, b.rejeu_max, int( b.rejeu_ko ), b.octets, int( b.perdus ),
                 int( b.env_deg ), b.env_max, b.t_cell, b.t_env, b.t_vol,
                 b.ok ? "ok" : ( b.novf ? "DEBORDEMENT"
                       : ( b.rejeu_ko ? "REJEU FAUX"
                       : ( b.env_max >= 1e-6 ? "ENCEINTE FAUSSE" : "FAUX" ) ) ) );
}

} // namespace pd::supercell
