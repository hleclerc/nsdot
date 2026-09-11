// L'ETAPE 4 : LA CELLULE FINALE, ET LE TEMOIN LE PLUS FORT DU BANC.
//
// Pour chaque dirac : on REJOUE sa cellule de base depuis la recette de l'etape 2 ( ~5 coupes en
// 2D, ~12 en 3D, sans une seule recherche ), puis on ne l'oppose qu'aux agregats de sa ligne de
// table dont l'enceinte la rencontre ENCORE. Le critere est exact :
//
//     si `enc_B` ne rencontre pas `C_i`, alors pour tout `j` de `B`, `Lag_i inter Lag_j` est vide
//     ( les vraies cellules sont dans `C_i` et dans `enc_B` ), donc `j` n'est pas voisin de `i`
//     et son demi-espace est redondant.
//
// = CE QUI SE MESURE ICI, ET POURQUOI C'EST LE TEMOIN DECISIF
//
// La somme des mesures doit valoir **1**. Pas « au bruit pres un sur-ensemble », pas « la table
// contient les paires utiles » : le diagramme complet, exact, dont les cellules pavent le domaine.
// Toute cellule trop grande ( un plan oublie ) ou trop petite ( un plan de trop -- impossible ici,
// tous les plans sont de vrais germes ) se voit dans ce chiffre. Sous `--verif` on va plus loin et
// on compare CELLULE PAR CELLULE avec le diagramme de reference construit par `AaBsp` : une somme
// juste par compensation ne passerait pas.
//
// = L'ENTONNOIR, ET CE QU'IL FAUT EN ATTENDRE
//
// Par dirac : `table` agregats presentes -> `boite` apres le test de boite -> `retenus` apres le
// SAT -> `plans` coupes tentees ( la somme des `|B|` retenus ). Le prototype 2D annoncait qu'en
// partant de ~6 agregats utiles le filtre en retient 1.72 a `rho`=8 ; en 3D, 15.6 sur 17. C'est
// cette reduction-la, par DIRAC et non par agregat, qui decide de la comparaison avec les 68
// unites du BSP.

#pragma once

#include "supercell/SurCellules.h"
#include "supercell/Table.h"
#include "spatial_accel/AaBsp.h"
#include "geometry/PowerDiagram.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>

namespace pd::supercell {

using namespace pd::bench;

struct BilanCel {
    double somme = 0;             ///< somme des mesures -- doit valoir 1
    double nv = 0, nf = 0;        ///< sommets et facettes de la cellule FINALE
    double rejeu = 0;             ///< coupes rejouees ( la passe 1 )
    double table = 0, boite = 0, retenus = 0;   ///< l'entonnoir, par dirac
    double plans = 0;             ///< coupes tentees en passe 3
    SI     novf = 0, novf_ref = 0, sans_recette = 0;
    double ecart_max = 0;         ///< sous `--verif` : pire ecart relatif d'aire
    double pire_facette = 0;      ///< ... et la plus grande facette en desaccord, en `h^(D-1)`
    SI     faux = 0;              ///< cellules dont la MESURE s'ecarte de la reference
    SI     reconstruites = 0;
    SI     au_bord_faux = 0, dedans_faux = 0;
    SI     etiquettes = 0;        ///< ... et celles dont seul l'ETIQUETAGE des facettes differe
    double t_rejeu = 0, t_filtre = 0, t_coupe = 0, t_ref = 0;
    bool   ok = false;
};

template<int D, class Cell>
inline Vec<D> sommet( const Cell &c, SI v ) {
    Vec<D> x;
    if constexpr ( D == 2 ) { x[ 0 ] = c.vx[ v ]; x[ 1 ] = c.vy[ v ]; }
    else                      x = c.vertex( v );
    return x;
}

/// L'ENCEINTE RENCONTRE-T-ELLE LA CELLULE ? Un SAT sur les axes de l'ENCEINTE seulement.
///
/// Sur chacun d'eux les deux cotes sont EXACTS : l'enceinte par ses bornes, la cellule par la
/// projection de ses sommets. Ne pas y ajouter les normales de faces de la cellule est un choix :
/// elles couteraient une projection de l'enceinte par facette pour un rejet de plus rarement
/// decisif, et le test reste SAIN dans ce sens -- incomplet ne fait que garder un agregat de trop.
template<int D, class Enc, class Cell>
bool rencontre( const Enc &e, const Cell &c ) {
    for ( SI k = 0; k < e.nb_axes(); ++k ) {
        const Vec<D> n = e.axe( k );
        TF bl, bh;
        e.bande( k, bl, bh );
        TF cl_ = dot( n, sommet<D>( c, 0 ) ), ch = cl_;
        for ( SI v = 1; v < c.nb; ++v ) {
            const TF u = dot( n, sommet<D>( c, v ) );
            cl_ = std::min( cl_, u ); ch = std::max( ch, u );
        }
        if ( cl_ > bh || bl > ch )
            return false;
    }
    return true;
}

/// LA CELLULE RECONSTRUITE depuis la recette `packe`, en 2D, SANS UNE SEULE DECOUPE.
///
/// La recette porte deja TOUTE la combinatoire : la suite cyclique des coupes est la connectivite
/// ( l'invariant de `CellSoA` : la coupe `v` porte l'arete sortante du sommet `v` ), donc le
/// sommet `v` est l'intersection des coupes `v-1` et `v`. Il ne reste qu'un systeme `2 x 2` par
/// sommet, soit ~10 operations, la ou rejouer demandait un clip complet du polygone par coupe.
///
/// C'est ce que le stockage `packe` permet et que `bits` ne permet pas : le masque donne
/// l'ENSEMBLE des coupes, pas leur ORDRE, et sans l'ordre il faut redecouper pour le retrouver.
///
/// Rend `false` si la recette manque ( liste trop longue ) ou si un determinant s'annule --
/// l'appelant rejoue alors, ce qui est toujours juste.
template<bool Weighted, class Mem>
bool reconstruit2d( const Gros<2> &G, const Mem &mem, SI i, const std::vector<SI> &cand, SI nc,
                    Vec<2> p0, TF w0, CellSC<2> &c ) {
    using Cell = CellSC<2>;
    const SI nb = mem.nb_codes( i );
    if ( nb < 3 || nb > Cell::max_nb_vertices )
        return false;
    TF dx[ Cell::max_nb_vertices ], dy[ Cell::max_nb_vertices ], of[ Cell::max_nb_vertices ];
    SI id[ Cell::max_nb_vertices ];
    // les quatre cotes du carre unite, dans l'ordre de `init_as_unit_square`
    static const TF bx[ 4 ] = { 0, 1, 0, -1 }, by[ 4 ] = { -1, 0, 1, 0 }, bo[ 4 ] = { 0, 1, 1, 0 };
    for ( SI k = 0; k < nb; ++k ) {
        const SI co = mem.code( i, k );
        if ( co < nc ) {
            Vec<2> d; TF o;
            plan<2, Weighted>( G, cand[ co ], p0, w0, d, o );
            dx[ k ] = d[ 0 ]; dy[ k ] = d[ 1 ]; of[ k ] = o; id[ k ] = cand[ co ];
        } else {
            const SI e = co - nc;
            if ( e < 0 || e > 3 ) return false;
            dx[ k ] = bx[ e ]; dy[ k ] = by[ e ]; of[ k ] = bo[ e ]; id[ k ] = -1;
        }
    }
    for ( SI v = 0; v < nb; ++v ) {
        const SI a = v ? v - 1 : nb - 1;
        const TF det = dx[ a ] * dy[ v ] - dy[ a ] * dx[ v ];
        if ( std::fabs( det ) < TF( 1e-300 ) ) return false;
        c.vx[ v ] = ( of[ a ] * dy[ v ] - dy[ a ] * of[ v ] ) / det;
        c.vy[ v ] = ( dx[ a ] * of[ v ] - of[ a ] * dx[ v ] ) / det;
        c.cdx[ v ] = dx[ v ]; c.cdy[ v ] = dy[ v ]; c.co[ v ] = of[ v ]; c.cid[ v ] = id[ v ];
    }
    c.nb = nb;
    return true;
}

template<int D, bool Weighted, Stockage S, class Enc, Opposants Mode>
BilanCel etape4( const Cloud<D> &cl, const Gros<D> &G, const Table &T,
                 const std::vector<Enc> &enc, const Memoire<S, CellSC<D>, D> &mem,
                 int anneau, bool verif, SI leaf ) {
    using Cell = CellSC<D>;
    BilanCel b;

    // --- LA REFERENCE, si on verifie : le diagramme complet par l'accelerateur du depot. On en
    //     garde la mesure ET LES FACETTES. Le jugement se fait sur les facettes, pas sur l'aire :
    //     sur une cellule en lame l'aire est une difference de grands termes et son ecart relatif
    //     ( ~1e-09 ) ne dit rien de la justesse -- la lecon avait deja ete payee sur le rejeu de
    //     l'etape 2. La mesure reste imprimee, comme information sur le bruit d'arrondi.
    std::vector<TF> mes_ref;
    std::vector<SI> ref_deb, ref_cut;
    std::vector<TF> ref_mes;
    // = POURQUOI COMPARER LES MESURES SUFFIT -- ET LE PROUVE
    //
    // La cellule construite ici est coupee par un SOUS-ENSEMBLE des plans ( seulement les membres
    // des agregats retenus ), donc elle CONTIENT la vraie cellule. La reference EST la vraie
    // cellule. Or deux convexes dont l'un contient l'autre et qui ont la meme mesure sont EGAUX.
    // Comparer les mesures n'est donc pas un controle approche : c'est un critere COMPLET.
    //
    // A une condition : le prendre en ecart ABSOLU rapporte a `h^D`, et non en relatif. Une
    // cellule en lame a une mesure minuscule calculee par un lacet ou les grands termes
    // s'annulent ; 2.6e-09 de SON aire ne represente rien du tout, et c'est ce qui faisait
    // signaler 12 cellules a tort.
    //
    // L'ensemble des FACETTES, lui, differe sur ~2.6 % des cellules, avec des facettes allant
    // jusqu'a 2 `h`. Ce n'est PAS une contradiction avec ce qui precede : un plan qui passe a
    // 1e-12 d'une arete la coupe d'un cote et pas de l'autre selon l'ordre des coupes, et cree
    // alors une facette LONGUE qui borde un eclat d'epaisseur nulle. La geometrie est la meme,
    // l'etiquetage non. On le COMPTE, on ne le juge pas.
    const double hD  = std::pow( 1.0 / double( cl.n ), 1.0 );
    const double hd = std::pow( 1.0 / double( cl.n ), ( D - 1.0 ) / D );
    if ( verif ) {
        const double t = now();
        // LA REFERENCE SE DIMENSIONNE LARGEMENT, et ce n'est pas un detail. Avec la cellule des
        // etapes 2 et 4 ( 32 sommets en 2D ) elle debordait 240 fois sur `lignes / Voronoi` et
        // 1761 fois sur `lignes / aires egales` : `cut` rend alors `overflow`, laisse la cellule
        // INTACTE, et la reference sort fausse en silence -- elle aurait valide n'importe quoi.
        // Le debordement est d'ailleurs transitoire : les cellules FINALES tiennent en 32 sommets
        // ( l'etape 4 n'en deborde aucune ), c'est l'ordre des coupes de l'accelerateur qui passe
        // par des etats plus larges. `256` en 3D et pas plus : `Cell3T` veut `ht_size` strictement
        // superieur a son nombre de faces, et `512` le violerait.
        using CellRef = CellFor<D, D == 2 ? 128 : 256>;
        AaBspT<D> ref;
        ref.build( cl.P, cl.W, cl.n, leaf );
        PowerDiagram<CellRef, AaBspT<D>, true, false, false, Weighted> pd{ ref };
        mes_ref.assign( cl.n, TF( 0 ) );
        b.novf_ref = 0;
        std::vector<std::vector<std::pair<SI,TF>>> fac( cl.n );
        for ( SI k = 0; k < cl.n; ++k ) {
            CellRef c;
            pd.make_cell( c, k );
            const SI i = ref.seed_id( k );
            mes_ref[ i ] = c.measure();
            c.for_each_facet( [ & ]( SI j, TF m ) { fac[ i ].push_back( { j, m } ); } );
            std::sort( fac[ i ].begin(), fac[ i ].end() );
        }
        ref_deb.assign( cl.n + 1, 0 );
        for ( SI i = 0; i < cl.n; ++i ) ref_deb[ i + 1 ] = ref_deb[ i ] + SI( fac[ i ].size() );
        ref_cut.resize( ref_deb[ cl.n ] );
        ref_mes.resize( ref_deb[ cl.n ] );
        for ( SI i = 0; i < cl.n; ++i )
            for ( SI q = 0; q < SI( fac[ i ].size() ); ++q ) {
                ref_cut[ ref_deb[ i ] + q ] = fac[ i ][ q ].first;
                ref_mes[ ref_deb[ i ] + q ] = fac[ i ][ q ].second;
            }
        b.novf_ref = pd.nb_overflow.load();      // la reference doit etre saine, elle aussi
        b.t_ref = now() - t;
    }
    std::vector<std::pair<SI,TF>> mes_fac;

    // --- les boites des enceintes, une fois pour toutes
    std::vector<Vec<D>> blo( G.ns ), bhi( G.ns );
    for ( SI a = 0; a < G.ns; ++a ) enc[ a ].boite( blo[ a ], bhi[ a ] );

    // LES CROCHETS DE DEBOGAGE, LUS UNE FOIS. Ils etaient dans les boucles : `getenv` parcourt
    // l'environnement lineairement et il tournait 12.6 millions de fois par lancer. Les « 71 ns
    // pour quatre comparaisons » du test de boite, c'etait lui -- et j'en avais conclu que le
    // poste etait memoire.
    const bool sans_recette = getenv( "SANS_RECETTE" ) != nullptr;
    const bool sans_filtre  = getenv( "SANS_FILTRE" )  != nullptr;
    const bool sans_boite   = getenv( "SANS_BOITE" )   != nullptr;
    const bool diag         = getenv( "DIAG" )         != nullptr;
    std::vector<SI> vus, ext, cand, relu, garde;
    std::vector<std::pair<TF,SI>> tri;
    std::vector<SI> deja( G.ns, -1 );
    double t_rejeu = 0, t_filtre = 0, t_coupe = 0;
    long long tot_nv = 0, tot_nf = 0, tot_rj = 0, tot_tab = 0, tot_bo = 0, tot_re = 0, tot_pl = 0;
    double somme = 0;

    for ( SI a = 0; a < G.ns; ++a ) {
        const SI d0 = G.mdeb[ a ], d1 = G.mdeb[ a + 1 ];
        if ( d0 == d1 ) continue;
        SI n1 = 0;
        opposes<D, Mode>( G, a, anneau, vus, ext, n1 );
        // L'ANNEAU 1 EST DEJA DANS LA CELLULE DE BASE quand on a coupe avec tous ses membres : la
        // recette les contient tous ( les redondants en moins, ce qui ne change pas
        // l'intersection ), donc les repasser serait un no-op. On les MARQUE pour les sauter.
        if constexpr ( Mode != Opposants::Median )
            for ( SI q = 0; q < n1; ++q ) deja[ vus[ q ] ] = a;
        liste_candidats<D>( G, a, ext, cand );
        const SI nc = SI( cand.size() );

        for ( SI u = d0; u < d1; ++u ) {
            const SI i = u;                             // indice PERMUTE
            Vec<D> p0;
            for ( int d = 0; d < D; ++d ) p0[ d ] = G.Ppp[ d ][ i ];
            const TF w0 = Weighted ? G.Wp[ i ] : TF( 0 );


            // --- LA PASSE 1, REJOUEE. La recette ne garde que les plans qui ont porte une
            //     facette : les rejouer rend la meme cellule, sans essayer les autres.
            // --- LA PASSE 1 : la cellule de base. Ou bien elle a ete gardee entiere et il n'y
            //     a rien a faire, ou bien on la REJOUE depuis la recette -- qui ne garde que les
            //     plans ayant porte une facette, donc les rejouer rend la meme cellule sans
            //     essayer les autres.
            double t = now();
            Cell c;
            bool fait = mem.donne( i, c );
            if constexpr ( S == Stockage::Packe && D == 2 )
                if ( ! fait ) { fait = reconstruit2d<Weighted>( G, mem, i, cand, nc, p0, w0, c );
                                if ( fait ) ++b.reconstruites; }
            if ( ! fait ) {
                if constexpr ( D == 2 ) c.init_as_unit_square();
                else                    c.init_as_unit_cube();
                mem.relis( i, nc, relu );
                const bool recette = ( ! relu.empty() || nc == 0 ) && ! sans_recette;
                if ( ! recette ) ++b.sans_recette;      // liste trop longue pour le masque
                Vec<D> d0v; TF o0v;
                if ( recette ) {
                    for ( SI j : relu ) {
                        plan<D, Weighted>( G, cand[ j ], p0, w0, d0v, o0v );
                        if ( c.cut( d0v, o0v, cand[ j ] ) == CutResult::overflow ) ++b.novf;
                    }
                    tot_rj += SI( relu.size() );
                } else {
                    for ( SI j = 0; j < SI( cand.size() ); ++j ) {
                        if ( j == i - d0 ) continue;
                        plan<D, Weighted>( G, cand[ j ], p0, w0, d0v, o0v );
                        if ( c.cut( d0v, o0v, cand[ j ] ) == CutResult::overflow ) ++b.novf;
                    }
                    tot_rj += SI( cand.size() );
                }
            }
            t_rejeu += now() - t;

            // --- LA PASSE 3 : la ligne de table, filtree contre CETTE cellule
            t = now();
            Vec<D> dir; TF off;
            Vec<D> lo, hi;
            c.bounds( &lo[ 0 ], &hi[ 0 ] );
            // PAS de tableau a taille fixe ici. La table a une queue lourde -- sur `lignes /
            // Voronoi` un agregat en touche jusqu'a 6 984 -- et un plafond y ecrete des agregats
            // en silence : la somme des mesures passait a 1.081 au lieu de 1.
            garde.clear();
            for ( SI q = T.deb[ a ]; q < T.deb[ a + 1 ]; ++q ) {
                const SI bb = T.vois[ q ];
                if constexpr ( Mode != Opposants::Median )
                    if ( deja[ bb ] == a ) continue;     // anneau 1 : deja dans la cellule de base
                ++tot_tab;
                bool disjoint = false;
                for ( int d = 0; d < D && ! disjoint; ++d )
                    disjoint = bhi[ bb ][ d ] < lo[ d ] || hi[ d ] < blo[ bb ][ d ];
                if ( disjoint && ! sans_boite ) continue;
                ++tot_bo;
                if ( ! sans_filtre && ! rencontre<D>( enc[ bb ], c ) ) continue;
                ++tot_re;
                garde.push_back( bb );
            }
            t_filtre += now() - t;

            // --- les coupes qui restent, DU PLUS PROCHE AU PLUS LOIN
            //
            // L'ordre ne change pas le resultat mais change tout le transitoire. En coupant par
            // agregat dans l'ordre de la table, les plans arrivent groupes par voisin et non par
            // distance : la cellule passe par des etats bien plus larges que sa forme finale ( 27
            // sommets en 3D ) et debordait a 128 comme a 256 sommets. `cut` rend alors `overflow`
            // et laisse la cellule INTACTE -- on obtient un objet qui porte toutes les facettes de
            // la vraie cellule tout en etant plus gros, ce qui est impossible pour un convexe et
            // signe la corruption. Trier par distance au median est ce que fait l'accelerateur du
            // depot, et ca ne coute qu'un tri de ~17 elements.
            t = now();
            // la distance UNE FOIS par agregat, pas a chaque comparaison : le comparateur
            // faisait deux indirections et `D` lectures par appel, soit `k log k` fois de trop.
            tri.clear();
            for ( SI bb : garde ) {
                Vec<D> a1;
                const SI rep = Mode == Opposants::Median ? G.median[ bb ] : G.mdeb[ bb ];
                for ( int d = 0; d < D; ++d ) a1[ d ] = G.Ppp[ d ][ rep ];
                tri.push_back( { dist2( p0, a1 ), bb } );
            }
            std::sort( tri.begin(), tri.end() );
            // LE MEME CRIBLE QU'A L'ETAPE 2, et il devrait porter davantage ici : on coupe avec
            // TOUS les membres des agregats retenus, donc 31.8 plans pour 6 facettes finales, et
            // ces membres sont plus loin que ceux de l'agregat.
            TF r2 = rayon2<D>( c, p0 );
            for ( const auto &pr : tri ) {
                const SI bb = pr.second;
                for ( SI v = G.mdeb[ bb ]; v < G.mdeb[ bb + 1 ]; ++v ) {
                    const SI j2 = v;
                    Vec<D> p1;
                    for ( int d = 0; d < D; ++d ) p1[ d ] = G.Ppp[ d ][ j2 ];
                    const TF dw = Weighted ? G.Wp[ j2 ] - w0 : TF( 0 );
                    if ( hors_portee( dist2( p0, p1 ), dw, r2 ) )
                        continue;
                    ++tot_pl;
                    plan<D, Weighted>( G, j2, p0, w0, dir, off );
                    if ( c.cut( dir, off, j2 ) == CutResult::overflow ) ++b.novf;
                }
                // ... releve UNE FOIS PAR AGREGAT : assez rare pour ne rien couter, assez souvent
                // pour que le rayon suive la cellule qui se reduit.
                r2 = rayon2<D>( c, p0 );
            }
            t_coupe += now() - t;

            const TF m = c.measure();
            somme += double( m );
            tot_nv += c.nb;
            c.for_each_facet( [ & ]( SI, TF ) { ++tot_nf; } );
            if ( verif ) {
                // ECART ABSOLU, rapporte a la mesure MOYENNE d'une cellule ( `h^D = 1 / n` ).
                const SI io = G.mem[ i ];   // la reference parle en indices d'ORIGINE
                b.ecart_max = std::max( b.ecart_max,
                                        std::fabs( double( m - mes_ref[ io ] ) ) / hD );
                mes_fac.clear();
                c.for_each_facet( [ & ]( SI j, TF m ) { mes_fac.push_back( { G.mem[ j ], m } ); } );
                std::sort( mes_fac.begin(), mes_fac.end() );
                // la difference symetrique, par fusion des deux listes triees, en gardant la
                // MESURE de ce qui ne figure que d'un cote.
                SI q = 0, r = ref_deb[ io ];
                const SI r1 = ref_deb[ io + 1 ], q1 = SI( mes_fac.size() );
                double pire = 0;
                while ( q < q1 || r < r1 ) {
                    if      ( r >= r1 || ( q < q1 && mes_fac[ q ].first < ref_cut[ r ] ) )
                        pire = std::max( pire, double( mes_fac[ q++ ].second ) );
                    else if ( q >= q1 || ref_cut[ r ] < mes_fac[ q ].first )
                        pire = std::max( pire, double( ref_mes[ r++ ] ) );
                    else { ++q; ++r; }
                }
                b.pire_facette = std::max( b.pire_facette, pire / hd );
                // DIAGNOSTIC : ou sont les cellules fausses, et quel plan leur manque ?
                if ( diag && std::fabs( double( m - mes_ref[ io ] ) ) / hD > 1e-9 ) {
                    static int vu = 0;
                    TF dmin = 1;
                    for ( int d = 0; d < D; ++d )
                        dmin = std::min( dmin, std::min( G.Ppp[ d ][ i ], TF( 1 ) - G.Ppp[ d ][ i ] ) );
                    ++b.au_bord_faux;
                    if ( dmin > TF( 0.02 ) ) ++b.dedans_faux;
                    if ( vu < 2 && dmin > TF( 0.02 ) ) {
                        ++vu;
                        std::printf( "  DIAG dirac %d  agregat %d  mes %.8g ref %.8g  ( %d facettes / %d ref )\n",
                                     int( i ), int( a ), double( m ), double( mes_ref[ io ] ),
                                     int( mes_fac.size() ), int( ref_deb[ io + 1 ] - ref_deb[ io ] ) );
                        std::printf( "     MIENNES :" );
                        for ( const auto &pr : mes_fac )
                            std::printf( " %d(%.2g)", int( pr.first ), double( pr.second ) / hd );
                        std::printf( "\n     REF     :" );
                        for ( SI r3 = ref_deb[ io ]; r3 < ref_deb[ io + 1 ]; ++r3 )
                            std::printf( " %d(%.2g)", int( ref_cut[ r3 ] ), double( ref_mes[ r3 ] ) / hd );
                        std::printf( "\n" );
                        // FORCE BRUTE : on recoupe par TOUS les germes et on regarde lequel
                        // reduit encore MA cellule. Celui-la manque, et son identite dit ou.
                        Cell fb = c;
                        for ( SI j2 = 0; j2 < cl.n; ++j2 ) {
                            if ( j2 == i ) continue;
                            Vec<D> d2; TF o2;
                            plan<D, Weighted>( G, j2, p0, w0, d2, o2 );
                            Cell av = fb;
                            fb.cut( d2, o2, j2 );
                            if ( double( av.measure() - fb.measure() ) > 1e-14 )
                                std::printf( "     COUPABLE germe %d  agregat %d ( le mien : %d )"
                                             "  dans la table : %d  dans cand passe1 : %d\n",
                                             int( j2 ), int( G.lab[ j2 ] ), int( a ),
                                             int( std::find( T.vois.begin() + T.deb[ a ],
                                                             T.vois.begin() + T.deb[ a + 1 ],
                                                             G.lab[ j2 ] )
                                                  != T.vois.begin() + T.deb[ a + 1 ] ),
                                             int( std::find( cand.begin(), cand.end(), j2 ) != cand.end() ) );
                        }
                        std::printf( "     apres force brute : %.8g ( ref %.8g )\n",
                                     double( fb.measure() ), double( mes_ref[ io ] ) );
                        for ( SI r2 = ref_deb[ io ]; r2 < ref_deb[ io + 1 ]; ++r2 ) {
                            const SI j = ref_cut[ r2 ];
                            bool ici = false;
                            for ( const auto &pr : mes_fac ) ici |= pr.first == j;
                            if ( ici ) continue;
                            const SI bj = G.lab[ j ];
                            bool dans_table = false;
                            for ( SI q2 = T.deb[ a ]; q2 < T.deb[ a + 1 ]; ++q2 ) dans_table |= T.vois[ q2 ] == bj;
                            bool dans_cand = false;
                            for ( SI cc : cand ) dans_cand |= cc == j;
                            std::printf( "     manque voisin %d ( facette %.3g h ) agregat %d :"
                                         " table %d, candidat passe1 %d, agregat==le sien %d\n",
                                         int( j ), double( ref_mes[ r2 ] ) / hd, int( bj ),
                                         int( dans_table ), int( dans_cand ), int( bj == a ) );
                        }
                    }
                }
                b.etiquettes += ( pire / hd > 1e-6 );        // information, pas verdict
                if ( std::fabs( double( m - mes_ref[ io ] ) ) / hD > 1e-9 ) ++b.faux;
            }
        }
    }

    const double n = double( cl.n );
    b.somme = somme;
    b.nv = tot_nv / n;  b.nf = tot_nf / n;  b.rejeu = tot_rj / n;
    b.table = tot_tab / n;  b.boite = tot_bo / n;  b.retenus = tot_re / n;
    b.plans = tot_pl / n;
    b.t_rejeu = t_rejeu; b.t_filtre = t_filtre; b.t_coupe = t_coupe;
    // LE TEMOIN : la somme des mesures d'un diagramme complet vaut 1. Et sous `--verif`, aucune
    // cellule ne doit s'ecarter de la reference -- une somme juste par compensation ne passe pas.
    b.ok = std::fabs( b.somme - 1.0 ) < 1e-9 && b.novf == 0 && b.novf_ref == 0 && b.faux == 0;
    return b;
}

template<int D>
void ligne_cellules( const Cloud<D> &cl, const BilanCel &b ) {
    std::printf( "      %-20s cellule %5.2f sommets %5.2f facettes | rejeu %5.2f | table %6.2f"
                 " -> boite %6.2f -> retenus %5.2f -> plans %6.2f"
                 " | somme %.12f  ecart %.1e h^D  ( %d faux, %d ovf ref )"
                 " | %d reconstruites, %d sans recette"
                 " | %d ovf | rejeu %6.3f  filtre %6.3f  coupe %6.3f s | %s\n",
                 cl.nom.c_str(), b.nv, b.nf, b.rejeu, b.table, b.boite, b.retenus, b.plans,
                 b.somme, b.ecart_max, int( b.faux ), int( b.novf_ref ),
                 int( b.reconstruites ), int( b.sans_recette ),
                 int( b.novf ), b.t_rejeu, b.t_filtre, b.t_coupe,
                 b.ok ? "ok" : ( b.novf ? "DEBORDEMENT" : "FAUX" ) );
}

} // namespace pd::supercell
