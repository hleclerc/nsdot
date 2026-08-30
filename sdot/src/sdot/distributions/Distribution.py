from loom.util import Aggregate
from loom.tensor import RealTensor


class Distribution( Aggregate ):
    """Base class for probability distributions.

    Subclasses should override `measure` (property) to return the total mass.
    Supports automatic normalization via `normalized_version()` when `target_mass` is set.

    = Intégrer une CELLULE contre une distribution -- le contrat côté kernel

    Une distribution peut servir de MESURE à `PowerDiagram.measures` : au lieu du volume de la
    cellule, on veut alors l'intégrale de la densité dessus. Le partage est le même que pour les
    accélérateurs spatiaux (voir `SpatialAccelerator.py`) : la distribution sait DÉCOUPER, le
    diagramme sait ce qu'on calcule SUR un morceau -- et c'est ce qui fait qu'une image, une somme
    de gaussiennes, ou rien du tout, se branchent au même endroit sans un second intégrateur.

    La classe C++ homonyme doit donc offrir :

        void for_each_piece( const auto &cell, auto &&ws, auto &&func ) const;

    qui découpe `cell` en morceaux sur lesquels la densité est SIMPLE et appelle, pour chacun :

        func( piece, density )

      * `piece`   -- un polytope convexe (une `Cell`), le morceau. Ses coupes gardent le `cut_id`
                     qu'elles avaient dans la cellule (donc l'indice du germe qu'elles font face) ;
                     celles que le découpage ajoute portent `BOUNDARY`, c.-à-d. « pas un germe ».
                     C'est cette seule propriété qui permet à l'adjoint de traiter un morceau
                     exactement comme une cellule (`PowerDiagram::integrate_bwd_into`).
      * `density` -- COMMENT ce morceau s'intègre. Deux formes, distinguées à la COMPILATION par
                     `density.is_constant` :

    1. `is_constant == true` : `density.value` est la densité, constante sur le morceau, et
       `density.add_value_grad( grad_dist, g )` dit où ranger `g = d(sortie)/d(cette valeur)` -- la
       masse étant linéaire en elle, l'intégrateur passe le VOLUME du morceau, et la distribution
       seule sait dans quelle case de ses paramètres ça tombe. C'est `ConstantDensity`, ce que
       rendent une image (une case de `values` par pavé) et la densité unité (aucune case).
       L'intégration est alors exacte et ne coûte rien : `valeur * mesure`.

    2. `is_constant == false` : la densité S'INTÈGRE ELLE-MÊME, sur un simplexe --

           TF   integrate_over_simplex    ( const auto &pts ) const;
           void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts,
                                            auto &&grad_dist ) const;

       `pts` sont les `d + 1` sommets, `grad_pts` la cotangente à y accumuler (l'intégrateur la
       recolle sur les sommets du morceau, puis `scatter_cell_grad` remonte aux germes comme
       toujours), `grad_dist` celle des paramètres de la densité.

    L'INTÉGRATEUR N'ÉCRIT AUCUNE RÈGLE D'INTÉGRATION. Il apporte le découpage géométrique en
    simplices (`Cell::for_each_simplex`, en toute dimension) et rien d'autre. Une densité qui a une
    formule fermée sur un simplexe -- ou une réduction à une fonction spéciale -- la donne, et elle
    est exacte. Une qui n'est qu'une BOÎTE NOIRE (une fonction écrite ailleurs, une gaussienne dont
    on n'a pas dérivé la formule) s'emballe dans `PointwiseDensity`, qui implémente le même contrat
    par quadrature à partir de `value_at` / `gradient_at` / `add_value_grad_at` :

        void for_each_piece( const auto &cell, auto &&, auto &&func ) const {
            func( cell, PointwiseDensity{ *this } );
        }

    La quadrature n'est donc pas un régime de l'intégrateur : c'est UNE implémentation du contrat,
    à côté des formules exactes, et le choix appartient à la densité.

    « Pas de distribution » est un cas ordinaire du même code, pas une absence de code : c'est
    `UnitDensity` (densité 1, un seul morceau, la cellule), fabriqué côté C++ par
    `PowerDiagram::unit_density()` -- comme `EverySeed` l'est pour les accélérateurs.

    = Le SCRATCH, et pourquoi il ne passe pas par ici

    Découper demande de la place : `Cell::cut` écrit dans une cellule SÉPARÉE, donc une suite de
    coupes fait la navette entre deux tampons. Ces deux tampons-là (plus la table de compaction du
    régime d > 2) sont fournis par l'APPELANT, dans `ws` -- un `PieceWorkspace` (voir
    `PieceWorkspace.h`), que `PowerDiagram.measures` alloue par work-item comme tout le reste.

    Une distribution ne demande donc pas de la mémoire : elle dit seulement, DEPUIS PYTHON,
    combien de coupes de plus qu'une cellule un de ses morceaux peut porter
    (`extra_cuts_per_piece`), et ces cellules-là sont dimensionnées en conséquence. Une taille de
    tampon n'est pas une décision de kernel : c'est ce que la plateforme doit savoir AVANT
    d'allouer, donc ça se dit d'ici.
    """

    def bounding_half_spaces( self ):
        """Le SUPPORT de la distribution, en demi-espaces `direction . x <= offset`, ou `None`.

        Une densité à support compact (une image) borne les cellules pour rien : tout ce qui
        dépasse son support n'apporte aucune masse, donc le couper ne change PAS le résultat --
        c'est une identité, pas une approximation. `PowerDiagram` ajoute donc ces demi-espaces aux
        siens (voir son `__init__`), et il y gagne trois choses : les cellules du bord cessent
        d'être infinies, le test d'élagage d'un accélérateur redevient utilisable
        (`cell_may_be_cut` n'a rien à mordre sur une cellule infinie), et le découpage n'a plus à
        balayer toute la grille faute de boîte englobante (`Image::_for_each_piece`).

        `None` (le défaut) dit « support non borné », ce qui est le cas d'une somme de gaussiennes :
        la tronquer perdrait de la masse, donc on ne le fait pas dans son dos -- c'est alors à
        l'appelant de donner un `box` s'il en veut un.

        Renvoie `None` aussi quand la géométrie n'est pas lisible côté hôte (sous `jit`) : borner
        est une OPTIMISATION, et une optimisation qui ne peut pas se faire ne doit pas casser
        l'appel."""
        return None

    def extra_cuts_per_piece( self, nb_dims ):
        """Combien de coupes de plus qu'une cellule un MORCEAU peut porter.

        `0` (le défaut) dit « le morceau EST la cellule » : aucun découpage, donc pas de cellules
        de rechange à allouer. Une image découpe par les 2d plans d'un pavé de sa grille, donc
        `2 * nb_dims`. C'est la SEULE information de dimensionnement que le contrat demande ; voir
        la docstring de la classe pour pourquoi elle passe par Python et pas par le kernel."""
        return 0

    # current_mass   : Tensor...
    target_mass      : RealTensor
    @property
    def mass( self ):
        """Total mass/measure of this distribution. Implemented by subclasses."""
        if self.current_mass.is_undefined:
            self._update_current_mass()
        return self.current_mass

    def normalized_version( self ):
        """Return a version of this distribution normalized to target_mass, if specified.

        If target_mass is not set, returns self unchanged.
        If target_mass is set, returns a copy with values scaled so that measure == target_mass.
        """
        return self

    def _update_current_mass( self ):
        """  """
        raise NotImplementedError

    def raw_1d_diracs( self ):
        """For a 1D dirac-source distribution (`_is_dirac_source`): `( weights, batched_extra,
        project_fn )`, letting a target distribution's `try_update_otplan1d` read plain,
        differentiable backend arrays and bypass `driver.call` entirely (ordinary autodiff
        differentiates straight through). `None` when this distribution cannot supply this
        cheaply (default: unsupported) -- the caller then falls back to the general
        driver.call/C++ path.

        - `weights`: `[ nb_diracs ]` (shared across the batch) or `[ *batch, nb_diracs ]`.
        - `batched_extra`: a dict of this distribution's OWN per-batch-element leaves needed to
          compute positions (e.g. a per-angle projection normal) -- EMPTY if positions do not
          depend on the batch. The caller merges these into whatever it maps/vmaps over, so
          they are sliced one batch element at a time, not materialized in full.
        - `project_fn( extra )`: given one slice of `batched_extra` (same keys, each now
          unbatched -- `{}` if `batched_extra` is empty), returns this distribution's `[ n ]`
          1D positions for that one batch element. Deferred like this (a function, not a
          materialized array) so a projection that DOES depend on the batch (e.g.
          `ProjectedSumOfDiracs`'s `points·normal`) is computed LAZILY, one angle at a time,
          inside the caller's `lax.map` -- eagerly materializing it for every batch element
          upfront would defeat the whole point of mapping instead of vmapping (see
          `Image.try_update_otplan1d`)."""
        return None

    def try_update_otplan1d( self, plan ):
        """Attempt to solve `plan` (an `OtPlan1d` with `self` as one of its two
        distributions) without going through `driver.call` -- e.g. a closed-form, pure-JAX
        computation. On success: update `plan`'s output fields (at least `plan.cost`) and
        return True. On failure (unsupported combination): change nothing and return False,
        so the caller uses the general driver.call/C++ path instead. Default: always decline
        (default: unsupported)."""
        return False

    def batch_slice( self, index ):
        """An UNBATCHED version of `self` for one element (`index`, a traced int) of its
        (single) batch axis -- lets `OtPlan1d` loop over the batch with `jax.lax.map` (one
        instance, and so one `driver.call`, per iteration) instead of a single call handling
        every batch element's memory at once. This is what lets the driver.call/C++ path scale
        to a large batch count the same way `Image.try_update_otplan1d`'s own `lax.map` already
        does for the pure-JAX path: peak memory bounded by ONE batch element, not the total
        count (see `OtPlan1d._update_outputs_via_angle_loop`). `None` when unsupported (no
        batch axis, or this distribution type does not know how to slice itself) -- the caller
        then falls back to its previous, single-call batched behavior."""
        return None
