#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/Cell.h>
#include <loom/support/common_macros.h>
#include "Cell/CellBoundary.h"

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_Cell
struct Cell {
    SDOT_ATTRIBUTES_OF_Cell

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( vertex_positions )::TF;

    // ---- the artificial simplex of an UNBOUNDED cell ---------------------------------------------
    // `init_as_unbounded` stands in for "the whole space" with a SIMPLEX whose planes are flagged
    // `INFINITE` and carry MADE-UP offsets. A cut would then classify a vertex by where that
    // arbitrary simplex happens to sit, not by where the cell really is -- so before clipping, the
    // infinite planes are pushed out until the answer stops changing (`growth_for_cut`). Pushing
    // them is exact for the real planes: a vertex travels along the ray its own planes prescribe
    // (`growth_rate`), which keeps it exactly on any real plane it lies on. Dimension-generic:
    // which planes a vertex stands on is the only thing that changes with `d` (`vertex_cut`).
    static constexpr int max_growth_rounds = 4;   ///< 2 are enough in exact arithmetic; the rest is slack
    static constexpr TF  growth_margin     = 1e-6;///< how far PAST the plane a pushed vertex lands

    // WHICH cuts vertex `i` stands on -- `ct_dim` of them, in increasing order. The one place the
    // two descriptions of a cell meet: below 3D it is READ OFF THE ORDER (cut i carries the edge
    // leaving v_i, so v_i is the corner of cuts i-1 and i), above it is stored, in
    // `vertex_indices`. Everything dimension-generic here goes through it.
    SI   vertex_cut             ( SI i, SI r ) const;

    auto growth_rate            ( SI i ) const;                             ///< d vertex(i) / d push
    auto grown_vertex           ( SI i, TF g ) const;                       ///< where vertex `i` sits once pushed by `g`
    auto growth_for_cut         ( auto &&direction, auto off ) const;       ///< how far to push, for THIS cut

    void init_as_aligned_simplex( SI cut_id );

    void init_as_hypercube_bwd  ( auto &&origin, auto &&axes, auto &&grad_cell, auto &&grad_for_origin, auto &&grad_for_axes ) const;
    void init_as_hypercube      ( auto &&origin, auto &&axes, SI cut_id = CellBoundary::BOUNDARY );

    void init_as_unbounded      ();

    // Intersects with the half-space `direction . x <= offset` (`direction` need not be normalized
    // -- `offset` is the dot product it is compared to), writing the result into a SEPARATE cell:
    // a call's inputs and outputs are disjoint, so the update in place is a Python-side rebinding
    // (see `Cell.py::cut`). Like `measure`, it comes in TWO versions, one per DIMENSION REGIME --
    // and for the same reason: they do not rewrite the same description of the cell.
    //
    // Returns whether the result FITTED in `res`: on a capacity overflow it has recorded what it
    // would have taken (the host reserves more and runs again, see `driver.call`) and written
    // nothing, so `res` still holds whatever was there before. A caller that goes on cutting the
    // SAME pair of buffers over and over -- `PowerDiagram`, whose cells ping-pong between two
    // work-item-local cells -- must stop there rather than clip stale geometry; the one call per
    // cut of `Cell.py` simply ignores it, the host re-run being the whole answer.
    //
    // d == 2: `vertex_positions` alone IS the cell, and the cuts follow it through the invariant
    // the orderings of `init_as_*` establish -- CUT i CARRIES THE EDGE [ v_i, v_i+1 ], hence
    // `nb_cuts == nb_vertices`. Sutherland-Hodgman then rewrites both in ONE cyclic pass, with
    // nothing tabulated: no scratch, hence no cap on the number of threads.
    // La coupe TELLE QU'ELLE EST UTILISEE PAR UN BALAYAGE : elle peut repondre `unchanged` et ne
    // rien ecrire du tout, `res` restant alors intact (voir `CutResult`). C'est ce que veut celui
    // qui fait la navette entre deux tampons -- il lui suffit de ne pas echanger.
    //
    // Le test « rien a enlever » n'est PAS un pre-passage separe : c'est le meme balayage qui
    // calcule les produits scalaires et compte les sommets dehors. Un seul predicat (`s > 0`), a un
    // seul endroit -- deux copies du meme test finiraient par ne plus repondre pareil sur un sommet
    // a l'epsilon pres du plan, et l'une sauterait la coupe que l'autre inscrirait.
    // Combien de sommets le demi-espace `direction . x <= offset` laisse DEHORS.
    //
    // Zero == « il contient deja toute la cellule », donc la couper n'ecrirait qu'une copie a
    // l'identique -- et c'est le cas MAJORITAIRE (un accelerateur propose des feuilles entieres,
    // dont 6 germes en moyenne sont vraiment voisins en 2D).
    //
    // Une fonction A PART, et petite, pour deux raisons qui tirent dans le meme sens. Le predicat
    // `s > 0` n'est ecrit QU'ICI, donc `cut_into` et son appelant ne peuvent pas se mettre a
    // repondre differemment sur un sommet a l'epsilon du plan. Et l'appelant qui balaie peut
    // l'interroger SANS entrer dans le corps du clip : mesure, le seul fait d'y entrer pour en
    // ressortir aussitot coute +26 % a leaf=12 et +47 % a leaf=30 -- le clip est enorme, il est
    // inline deux fois par `PowerDiagram::cut_by`, et le chemin rejete payait son cadre.
    SI nb_vertices_outside      ( const auto &direction, TF offset ) const;

    // LA COUPE EN PLACE : elle modifie CETTE cellule, sans second tampon.
    //
    // Ce qu'elle evite n'est pas le calcul mais la RECOPIE. Le clip ordinaire reecrit la cellule
    // entiere dans l'autre tampon, y compris les sommets que la coupe ne touche pas ; ici seuls
    // les deux points d'intersection sont ecrits, plus le decalage qu'impose le changement de
    // taille. A 1e6 germes et six coupes utiles par cellule, la recopie evitee se compte en Go.
    //
    // = Ce qui la rend possible
    //
    // La sortie fait EXACTEMENT `nb - nb_dehors + 2` sommets, et les sommets conserves forment UNE
    // plage cyclique -- l'exterieur d'un convexe coupe par un demi-espace est d'un seul tenant. Le
    // sens du decalage est donc connu avant de bouger quoi que ce soit : `nb_dehors == 1` allonge
    // d'un cran, `== 2` laisse la taille inchangee (rien a decaler du tout), `>= 3` raccourcit.
    //
    // Rien n'est stocke : le balayage retient `nb_dehors` et le DEBUT de la plage exterieure (le
    // `i` tel que dehors(i) et dedans(i-1), unique par ce qui precede), et les quatre distances
    // dont les deux intersections ont besoin se recalculent en `O(1)`. Pas de masque, donc pas de
    // borne sur le nombre de sommets.
    //
    // = Pourquoi le CPU seulement
    //
    // Trois branches selon `nb_dehors` et des decalages de longueur variable : sur GPU les voies
    // d'un warp prendraient des chemins differents et boucleraient des nombres de fois differents,
    // ce qui coute plus que la recopie qu'on evite. `PowerDiagram::cut_by` regarde l'espace memoire
    // des tenseurs pour choisir -- rien n'a a traverser depuis Python.
    //
    // Reservee aux cellules BORNEES : une cellule non bornee est un simplexe de remplacement dont
    // la coupe repousse d'abord les plans infinis, donc sa geometrie change meme quand aucun sommet
    // ne sort, et ce n'est plus un simple decoupage.
    CutResult cut_in_place      ( const auto &direction, TF offset, SI cut_id );

    CutResult cut_into          ( auto &&res, auto &&direction, auto &&offset, SI cut_id ) const;
    CutResult cut_into          ( auto &&res, auto &&direction, auto &&offset, SI cut_id, auto &&corr ) const;

    // La coupe COMPLETE : `res` tient toujours le resultat, meme quand la coupe n'enleve rien --
    // elle recopie alors. C'est ce que veut un appelant qui a demande UNE coupe et attend son
    // resultat quelque part (`Cell.py::cut`), par opposition a celui qui balaie.
    bool cut                    ( auto &&res, auto &&direction, auto &&offset, SI cut_id ) const;


    // d > 2: no cyclic order to lean on. The FACE LATTICE (`vertex_indices` / `edge_indices`) is
    // what carries the cell, and the clip rewrites it: vertices are classified, the surviving ones
    // COMPACTED, every crossing edge yields a vertex on the new plane, the new facet is stitched
    // from the pairs of those vertices that share `d-2` old cuts, and the cuts nothing stands on
    // any more are dropped. Compaction is what the scratch is for: `corr` holds the old -> new
    // index maps ( `[ 0, nb_vertices )` for the vertices, `[ nb_vertices, ... )` for the cuts ),
    // one row per work-item, so the call caps its thread count on the room they take.
    bool cut                    ( auto &&res, auto &&direction, auto &&offset, SI cut_id, auto &&corr ) const;

    // Adjoint of `cut`. The clip is a SCATTER -- an input vertex feeds several output ones, an
    // input cut every output that copies it -- so this ACCUMULATES, and what it accumulates into
    // has to start at zero: `cut_bwd_setup` is that pre-pass, run once through the queue before any
    // item's body (see `Cell.py::cut`'s `bwd_setup_code`). `direction` / `offset` are SHARED across
    // the batch, so every item lands on the same two slots -- their adds are atomic.
    //
    // Neither version records anything in the forward: both REPLAY its walk (a handful of dot
    // products) rather than remember which output came from which input, which would be a buffer.
    void cut_bwd_setup          ( auto &&queue, auto &&grad_cell, auto &&grad_direction, auto &&grad_offset ) const;
    void cut_bwd                ( auto &&direction, auto &&offset, auto &&grad_res, auto &&grad_cell, auto &&grad_direction, auto &&grad_offset ) const;
    void cut_bwd                ( auto &&direction, auto &&offset, auto &&grad_res, auto &&grad_cell, auto &&grad_direction, auto &&grad_offset, auto &&corr ) const;

    // Adjoint of the point where the edge `vi -> vj` meets the cutting plane -- the ONE piece of
    // arithmetic both regimes share, so it lives here rather than twice inside them. `q` is the
    // cotangent of that point, already read (a missing one reads as zero, see the callers).
    void crossing_bwd           ( auto &&direction, SI i, SI j, const auto &vi, const auto &vj, TF si, TF sj,
                                  const auto &q, auto &&grad_cell, auto &&grad_direction, auto &&grad_offset ) const;

    // `measure` comes in TWO versions, one per DIMENSION REGIME -- neither is a specialization of
    // the other: they do not read the same description of the cell. `Cell.py::measure` picks one,
    // on `nb_dims`, and builds (or does not build) the scratch the other one needs.
    //
    // d <= 2: `vertex_positions` IS the geometry -- a segment in 1D, a CYCLICALLY ordered polygon
    // in 2D (see `vertex_ordering_2D`). Nothing to enumerate, hence no scratch at all: neither
    // `item_map` nor `nb_map_items` exists on this side (they stay `Unbound`, never allocated).
    void measure_bwd            ( auto &&res, auto &&grad_res, auto &&grad_vertex_positions ) const;
    void measure                ( auto &&res ) const;

    // ... mais découper en SIMPLICES a un sens ici aussi, et c'est ce que demande une grandeur qui
    // n'a pas de formule fermée sur la cellule (une quadrature, par exemple -- voir
    // `PowerDiagram::integrate_into` face à une densité lisse). Un éventail depuis le sommet 0 en
    // 2D (l'ordre cyclique est déjà là, cf. `vertex_ordering_2D`), le segment lui-même en 1D. Même
    // signature de callback que le cas d > 2 -- `ct_dim + 1` INDICES de sommets -- de sorte qu'une
    // fonctionnelle s'écrive une fois pour toutes ; et aucun scratch de ce côté-ci, donc pas de
    // `facet_apex` dans la signature.
    void for_each_simplex       ( auto &&func ) const;

    // d > 2: no formula to read off the vertices any more -- the cell has to be CUT INTO SIMPLICES
    // first, and that is a walk on the face lattice, hence `vertex_indices` (and only it: the fan
    // never needs `edge_indices`, an edge being just a face like any other here).
    //
    // The triangulation is the standard one: pick a vertex of the cell, cone it over every facet
    // that does not contain it, and triangulate each of those facets the same way, one dimension
    // down. What it takes is, for every face met on the way, ONE of its vertices ("the apex") --
    // any one, as long as the choice is fixed while that face is being triangulated.
    //
    // Getting that apex is the whole cost of the algorithm. `facet_apex` is what it takes here: one
    // row per recursion depth, one slot per cut, holding the apex of the facet that cut opens --
    // filled by a SINGLE pass over the face's vertices (a cut not already in the face names a
    // facet, and the first vertex seen carrying it is on that facet, which is all an apex has to
    // be). So `ct_dim * nb_cuts` words, where keying the faces by their cut SET -- the obvious way,
    // and the one the previous implementation took -- costs `nb_cuts^(d-1)`.
    void for_each_simplex_rec   ( auto &&facet_apex, auto &chain, auto &face_cuts, auto &&func, auto face_dim ) const;
    void for_each_simplex       ( auto &&facet_apex, auto &&func ) const;

    bool has_cut                ( SI v, SI c ) const;   ///< does vertex `v` stand on cut `c` ?

    // Copies this cell -- geometry, H-representation, face lattice and counts -- into `res`.
    // Answers whether it FITTED, like `cut`: the counts go through `ShapeVarView::set`, so an
    // under-provisioned `res` records what it would have taken and nothing is written past its
    // capacity (the host reserves more and runs again, see `driver.call`).
    //
    // What it is for: a caller that builds cells in a small pair of work buffers but wants to KEEP
    // them -- `PowerDiagram::build_cell`, whose display path materializes every cell at once. Nothing in
    // the ordinary cut/measure path needs it, a cell being consumed where it is built.
    bool copy_into              ( auto &&res ) const;

    void measure_bwd            ( auto &&res, auto &&facet_apex, auto &&grad_res, auto &&grad_vertex_positions ) const;
    void measure                ( auto &&res, auto &&facet_apex ) const;
};

}

#include "Cell.cxx"
