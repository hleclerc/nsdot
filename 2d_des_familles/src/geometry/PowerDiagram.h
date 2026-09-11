#pragma once

#include "geometry/WeightMajorant.h"
#include "util/common.h"
#include "util/parallel.h"
#include <atomic>
#include <vector>

namespace pd {

/// Le diagramme : pour chaque germe, une cellule construite, mesuree, oubliee.
///
/// Rien n'est garde -- le diagramme n'existe jamais comme un tout. C'est ce qui permet a une
/// cellule de tenir sur la pile et au parallelisme d'etre sans partage : chaque thread a SA
/// cellule, dans son cadre, et n'ecrit que dans `res[ i ]`.
///
/// `Cell` et `Accel` sont des parametres de template, et la DIMENSION se lit sur la cellule
/// (`Cell::dim`). Un accelerateur ne doit qu'UNE chose :
///
///     for_each_candidate( k0, may_cut, cut_with, reach2 )
///
/// c'est-a-dire proposer des germes, dans l'ordre qu'il veut, en s'aidant de `may_cut` pour sauter
/// ce qu'il peut. Il porte donc son propre parcours. C'etait auparavant un concept `Walk` separe,
/// mais ce concept decrivait un arbre binaire a boites -- `root`, `node`, `is_leaf`, `left`,
/// `right`, `nearness`, `emit_leaf` -- donc ne se generalisait a rien : la grille ne pouvait pas
/// l'implementer et son parcours ne pouvait pas s'appliquer a un arbre.
///
/// Les deux rappels sont typees ainsi, et c'est LA SEULE CHOSE que le passage a la dimension
/// quelconque a change dans ce fichier :
///
///     may_cut ( Vec<D> lo, Vec<D> hi, const WMajT<D> &wm ) -> bool
///     cut_with( Vec<D> p1, TF w1, SI i1 )                  -> bool
///
/// TOUT passe par VALEUR, y compris les boites. C'etait le choix a faire et il n'allait pas de soi :
/// la boite d'un noeud de `AaBsp` est deja contigue en memoire, donc un `const TF *` aurait evite de
/// la recopier. Mais `AaBsp4` range ses boites en SoA (`lo[ axe ][ fils ]`) et la grille calcule les
/// siennes a la volee -- toutes deux auraient du materialiser un tableau sur la pile juste pour en
/// prendre l'adresse, et prendre l'adresse est precisement ce qui EMPECHE le compilateur de garder
/// la valeur en registres. Par valeur, un agregat de deux ou trois `double` voyage en `xmm` et
/// personne ne materialise rien.
///
/// `CellBox` : maintenir la BOITE de la cellule pour un rejet en `O( 1 )` avant de regarder les
///             sommets. La boite se refait apres chaque coupe EFFECTIVE (une poignee par cellule) ;
///             sans elle, chaque test d'eviction balaie les sommets.
/// `SkipInside` : ne pas tester une boite qui CONTIENT le germe. Une telle boite ne peut jamais
///             etre evincee -- le germe est dans la cellule et dans la boite, donc le test rend
///             toujours vrai -- et il y en a une par niveau sur le chemin racine -> feuille du
///             germe, soit une quinzaine par cellule.
/// `Stats` : compter ce que le parcours a REELLEMENT fait. Parametre de template et non drapeau
///             d'execution, pour que les compteurs disparaissent a la compilation dans le cas
///             normal -- sans quoi on mesurerait l'instrumentation. Ils ne sont PAS atomiques : le
///             mode `--stats` tourne a un seul thread.
/// `Weighted` : y a-t-il des poids du tout. Sans eux le majorant est identiquement nul, mais le
///             compilateur ne peut pas le savoir : il emet quand meme les produits, les maximums et
///             le decalage par sommet. MESURE : 19 % sur le cas uniforme 2D. C'est donc un
///             parametre de template, comme dans `sdot` ou le tenseur des poids est simplement
///             absent et la branche tombe a la compilation.
template<class Cell, class Accel, bool CellBox = true, bool SkipInside = false, bool Stats = false,
         bool Weighted = true>
struct PowerDiagram {
    static constexpr int D = Cell::dim;
    static_assert( Accel::dim == D, "l'accelerateur et la cellule doivent etre de meme dimension" );

    const Accel &tree;

    /// Combien de coupes ont ATTEINT `max_nb_vertices`. Une cellule qui deborde est laissee telle
    /// quelle, donc trop grande : sa mesure est fausse, et la somme des mesures ne vaut plus 1. Sur
    /// un nuage uniforme ce compteur reste a zero et la question ne se pose pas -- c'est sur un
    /// nuage DUR qu'il faut le regarder, sans quoi on lirait un temps de calcul sans savoir qu'il
    /// porte sur un faux resultat. Atomique et touche UNIQUEMENT en cas de debordement : hors du
    /// chemin chaud.
    mutable std::atomic<SI> nb_overflow{ 0 };

    mutable long long st_boxes = 0;     ///< appels au test d'eviction
    mutable long long st_sweep = 0;     ///< ... ou le test en `O( 1 )` n'a pas conclu
    mutable long long st_kept  = 0;     ///< ... qui n'ont pas evince
    mutable long long st_tried = 0;     ///< germes proposes a la coupe
    mutable long long st_done  = 0;     ///< ... qui ont vraiment coupe

    /// « la boite `[ lo, hi ]`, dont les poids sont majores par `wm`, peut-elle encore atteindre la
    /// cellule ? »
    ///
    /// Le test est en `O( 1 )` puis, s'il ne conclut pas, en `O( nb sommets )`. On majore la
    /// cellule par SA BOITE, pas par une sphere : bien plus serre pour une cellule de Laguerre, et
    /// pas plus cher -- une somme par axe, sans racine. TOUT y est SEPARABLE PAR AXE, et c'est ce
    /// qui rend le passage a la dimension quelconque gratuit : les boucles `for d` ci-dessous sont
    /// exactement les deux blocs `{ ... }` qui etaient ecrits a la main pour `x` et `y`.
    ///
    /// LE MAJORANT DE POIDS est ce qui rend le test correct hors du cas euclidien : un poids eleve
    /// rapproche le plan de `p0`, donc une region qu'on croirait trop loin peut couper quand meme.
    /// Il est AFFINE (`w( y ) <= a . y + b`), et c'est gratuit : le minimum de `|p - y|^2 - a . y`
    /// sur une boite reste separable par axe, son minimum libre est en `y = p + a / 2`, et un
    /// `clamp` par axe le donne exactement. Un majorant constant est le meme code avec `a = 0`.
    ///
    /// `sweep` est le compteur de `--stats` : passe en PARAMETRE pour que la methode reste
    /// `static`. Mesure : la rendre membre pour lire le compteur directement coutait 7 % (0.170 ->
    /// 0.182 s) alors que le compteur lui-meme est compile hors du code -- un `this` de plus a
    /// faire vivre suffit a changer ce que l'inlineur garde en registres.
    static bool may_be_cut( const Cell &c, Vec<D> clo, Vec<D> chi, Vec<D> p0, TF w0,
                            Vec<D> lo, Vec<D> hi, const WMajT<D> &wm, long long &sweep ) {
        // LES PENTES, converties UNE FOIS. C'est ce que la version 2D faisait a la main.
        Vec<D> a = {};
        if constexpr ( Weighted )
            a = pentes<D>( wm );

        if constexpr ( SkipInside ) {
            // la boite contient le germe : elle ne peut pas etre evincee, et le prouver coute
            // `2 D` comparaisons la ou le test general coute un balayage.
            bool in = true;
            for ( int d = 0; d < D; ++d )
                in &= p0[ d ] >= lo[ d ] && p0[ d ] <= hi[ d ];
            if ( in )
                return true;
        }

        if constexpr ( CellBox ) {
        // `dist^2( C, B ) - max_y( a . y ) - b - max_{p in C} abs( p - p0 )^2 + w0`, terme a terme.
        // Les trois quantites sont SEPARABLES par axe -- l'ecart entre deux boites, le point le plus
        // loin d'une boite, le coin qui maximise une forme lineaire -- donc chacune est une somme de
        // termes independants. Positif => on evince sans regarder un seul sommet.
        //
        // ESSAYE ET REJETE (2026-09-02) : la version EXACTE de ce test. Les trois termes etant
        // separables par axe, leur somme peut etre minimisee exactement axe par axe -- `h( t )` est
        // lineaire dans les deux regimes ou le point de `B` est colle a un bord (le `t^2` s'annule)
        // et concave dans celui ou il suit `t`, donc quatre candidats suffisent : `clo`, `chi` et
        // les deux ruptures. C'est strictement plus serre que ce qui suit, qui minore et majore en
        // des points DIFFERENTS de la cellule. Mesure : les balayages de sommets tombent de 128 a
        // 90 par cellule sur le cas dur, mais les boites GARDEES ne bougent pas d'un chiffre --
        // 77.2 avant comme apres. Le test exact ne fait que deplacer des decisions du balayage vers
        // le `O( 1 )`, sans jamais changer la reponse : le balayage etait deja aussi serre que la
        // boite le permet. Mesure APPARIEE : 0.182 -> 0.192 s sur l'uniforme (+5.5 %) et
        // 0.553 -> 0.524 sur le cas dur (-5.2 %). Pas de quoi porter un second test.
        //
        // Ce que ca N'INFIRME PAS : un meilleur majorant de POIDS. Lui change la reponse elle-meme
        // (il fait tomber « gardees »), donc il elague des sous-arbres entiers.
        TF far = 0;
        if constexpr ( Weighted )
            far = w0 - wm.b;
        for ( int d = 0; d < D; ++d ) {
            const TF g = clo[ d ] > hi[ d ] ? clo[ d ] - hi[ d ]
                                            : ( lo[ d ] > chi[ d ] ? lo[ d ] - chi[ d ] : TF( 0 ) );
            const TF u = clo[ d ] - p0[ d ], v = chi[ d ] - p0[ d ];
            far += g * g - ( u * u > v * v ? u * u : v * v );
            if constexpr ( Weighted ) {
                const TF s = a[ d ] * lo[ d ], t = a[ d ] * hi[ d ];
                far -= s > t ? s : t;
            }
        }
        if ( far > 0 )
            return false;
        }
        if constexpr ( Stats ) ++sweep;

        // sinon, le test EXACT sur les sommets. `<=` et non `<` : un plan qui passe exactement par
        // un sommet n'enleve rien, donc l'admettre coute une coupe inutile la ou le refuser sur un
        // arrondi perdrait une coupe VRAIE.
        for ( SI v = 0; v < c.nb; ++v ) {
            const Vec<D> p = c.vertex( v );
            TF s = 0;
            for ( int d = 0; d < D; ++d ) {
                // le point de la boite le plus proche de `p`, DECALE d'une demi-pente : le
                // minimande se separe par axe et son minimum libre est en `p + a / 2`.
                TF y = p[ d ];
                if constexpr ( Weighted ) y += a[ d ] / 2;
                y = y < lo[ d ] ? lo[ d ] : ( y > hi[ d ] ? hi[ d ] : y );
                const TF e = y - p[ d ], f = p[ d ] - p0[ d ];
                s += e * e - f * f;
                if constexpr ( Weighted ) s -= a[ d ] * y;
            }
            if constexpr ( Weighted )
                s += w0 - wm.b;
            if ( s <= 0 )
                return true;
        }
        return false;
    }

    /// UNE cellule, du domaine jusqu'a sa forme finale.
    ///
    /// ESSAYE ET REJETE (2026-09-03) : la factoriser en `make_cell_with( c, p0, w0, walk )` pour
    /// pouvoir construire une cellule avec un poids qui n'est pas celui range dans l'arbre (ce que
    /// demande `--baisse`). Le corps etait identique, le `walk` un lambda generique -- et ca coutait
    /// 3 % : 0.176 -> 0.182 s sur l'uniforme, mesure APPARIEE. Le besoin passe par un ACCELERATEUR
    /// (`OneSeed`) : le concept existe deja pour ca, et le chemin chaud ne bouge pas.
    void make_cell( Cell &c, SI k0 ) const {
        // D'OU PART LA CELLULE. Le domaine, sauf si l'accelerateur sait faire mieux : une
        // SUR-CELLULE qui contient deja toutes les cellules de son paquet part bien plus pres de
        // la reponse, donc le test d'eviction mord des la premiere boite. Detecte a la
        // COMPILATION -- l'accelerateur qui n'a pas la methode ne paie rien.
        if constexpr ( requires ( const Accel &t, Cell &cc ) { t.init_cell( cc, k0 ); } )
            tree.init_cell( c, k0 );
        else if constexpr ( D == 2 )
            c.init_as_unit_square();
        else
            c.init_as_unit_cube();

        const Vec<D> p0 = tree.seed( k0 );
        const TF w0 = tree.seed_w( k0 );
        Vec<D> clo = {}, chi = {};
        if constexpr ( CellBox )
            c.bounds( clo.v, chi.v );
        tree.for_each_candidate( k0,
            [ & ]( Vec<D> lo, Vec<D> hi, const WMajT<D> &wm ) {
                const bool r = may_be_cut( c, clo, chi, p0, w0, lo, hi, wm, st_sweep );
                if constexpr ( Stats ) { ++st_boxes; st_kept += r; }
                return r;
            },
            [ & ]( Vec<D> p1, TF w1, SI i1 ) {
                if constexpr ( Stats ) ++st_tried;
                Vec<D> dir;
                TF off = ( w0 - w1 ) / 2;
                for ( int d = 0; d < D; ++d ) {
                    dir[ d ] = p1[ d ] - p0[ d ];
                    off += dir[ d ] * ( p0[ d ] + p1[ d ] ) / 2;
                }
                const CutResult r = c.cut( dir, off, i1 );
                if constexpr ( Stats ) st_done += r != CutResult::unchanged;
                if ( r == CutResult::unchanged ) return true;
                if ( r == CutResult::empty ) return false;
                if ( r == CutResult::overflow ) {
                    nb_overflow.fetch_add( 1, std::memory_order_relaxed );
                    return true;
                }
                if constexpr ( CellBox )
                    c.bounds( clo.v, chi.v );
                return true;
            },
            [ & ]() {
                TF m = 0;
                for ( SI v = 0; v < c.nb; ++v ) {
                    const TF d = dist2( c.vertex( v ), p0 );
                    m = d > m ? d : m;
                }
                return m;
            } );

        // CE QU'ON RETIENT de la cellule finie. Detecte a la COMPILATION, comme `init_cell` :
        // l'accelerateur qui n'a pas la methode ne paie pas une instruction.
        if constexpr ( requires ( const Accel &t, const Cell &cc ) { t.note_cell( cc, k0 ); } )
            tree.note_cell( c, k0 );
    }

    /// Les mesures de tout le nuage.
    void measures( std::vector<TF> &res, int nb_threads, Split split, bool pin ) const {
        const SI n = tree.nb_seeds();
        res.assign( n, TF( 0 ) );
        nb_overflow.store( 0, std::memory_order_relaxed );
        parallel_for( n, nb_threads, split, pin, [ & ]( SI k, int ) {
            Cell c;
            make_cell( c, k );
            res[ tree.seed_id( k ) ] = c.measure();     // ecrit PAR INDICE : l'ordre du balayage
        } );                                            // ne change rien au resultat
    }
};

} // namespace pd
