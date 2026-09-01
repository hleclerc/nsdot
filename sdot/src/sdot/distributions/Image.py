import numpy

from loom.compilation.FfiCode import FfiCodeParallel
from loom.drivers.driver import driver
from loom.tensor import Axis, AxisList, CtShapeVar, RealTensor, ShapeVar
from loom.util import ComputedAttribute

from .Distribution import Distribution


class Image( Distribution ):
    """
        Piecewise constant function on a grid.

        Each square/cube/hypercube is defined by `origin` and `frame( dir ) * knots( ... )`

        By default, knots is equal to 0, 1, ... for each dim.
    """

    nb_dims          : CtShapeVar
    shape            : ShapeVar[ "dim" ]

    num_knot         : Axis[ "shape + 1" ]
    img_pos          : AxisList[ "dim", "shape" ]
    dim              : Axis[ "nb_dims" ]
    dir              : Axis[ "nb_dims" ]

    values           : RealTensor[ "img_pos..." ]

    origin           : RealTensor[ "dim" ]
    frame            : RealTensor[ "dir", "dim" ]
    knots            : RealTensor[ "dim", "num_knot" ]

    current_mass     : ComputedAttribute[ RealTensor, ( "values", "frame", "knots" ) ]

    # `nb_cells_cum`/`num_cell_cum`: an INDEPENDENT ShapeVar + Axis pair (not derived from the
    # per-dim `shape`, unlike `num_knot`) -- `cell_cum_mass` is a FLAT array over ALL cells
    # (`nb_pieces + 1`), not expressible as an affine function of the per-dim `shape` the way
    # `num_knot` is (that one stays ragged over `dim`, fine for `knots` but wrong-shaped here). Same
    # two-field shape as `SumOfDiracs`'s `nb_diracs`/`num_dirac`. Prescribed once in
    # `_update_cell_cum_mass`, exactly like `OtPlan1d`'s own `nb_diracs`/`nb_dims` are prescribed
    # from elsewhere. A DECLARED axis (as opposed to a bare `Tensor`, fine for the truly-scalar
    # `current_mass`) is required so the generated C++ struct's `cell_cum_mass( c )` accepts an
    # index at all -- a bare `Tensor` field only ever gets a RANK-0 call operator.
    nb_cells_cum     : ShapeVar
    num_cell_cum     : Axis[ "nb_cells_cum" ]

    # exclusive prefix sum of each cell's mass ([nb_pieces+1], see `nb_pieces`) -- lets `OtPlan1d`
    # jump straight into the middle of its sequential walk (`Image::udp_at`) instead of following it
    # there step by step. Depends only on `values`/`frame`/`knots`, so it is CACHED like `current_mass`
    # (`ensure_cell_cum_mass` below) instead of being rebuilt on every `OtPlan1d` forward/backward
    # call -- previously it was, twice per call, see [[otplan1d-kernel-profile]].
    cell_cum_mass    : ComputedAttribute[ RealTensor[ "num_cell_cum" ], ( "values", "frame", "knots" ) ]

    def __init__( self, values, **kwargs ) -> None:
        self.__base_init__( values = values, target_mass = 1.0, **kwargs )

    def _grid_geometry( self ):
        """`( d, frame, origin, lo, hi )` de la grille, côté HÔTE -- ou `None` si elle n'est pas
        lisible ici (un tracer sous `jit`). Borner est une optimisation : on s'en passe plutôt que
        de faire échouer l'appel."""
        try:
            d = int( self.nb_dims.value )
            shape = numpy.asarray( self.shape.value, dtype = int ).reshape( -1 )
            frame = numpy.asarray( self.frame, dtype = float ).reshape( d, d ) if self.frame.is_defined else numpy.eye( d )
            origin = numpy.asarray( self.origin, dtype = float ).reshape( -1 ) if self.origin.is_defined else numpy.zeros( d )
            if self.knots.is_defined:
                knots = numpy.asarray( self.knots, dtype = float ).reshape( d, -1 )
                lo = knots[ :, 0 ]
                hi = numpy.array( [ knots[ a, shape[ a ] ] for a in range( d ) ] )
            else:
                lo, hi = numpy.zeros( d ), shape.astype( float )
        except ( TypeError, ValueError ):
            return None
        return d, frame, origin, lo, hi

    def bounding_half_spaces( self ):
        # voir `Distribution.bounding_half_spaces`. Le support est le pavé de la grille, écrit en
        # coordonnées PHYSIQUES : `t_a = n_a . ( x - origin )` avec `n_a` la colonne `a` de `F^-1`
        # (convention `x = origin + F^T t`, celle de `Image::measure`), et la bande utile va de
        # `knots( a, 0 )` à `knots( a, shape_a )`.
        g = self._grid_geometry()
        if g is None:
            return None
        d, frame, origin, lo, hi = g
        nrm = numpy.linalg.inv( frame ).T                        # ligne `a` = la normale de l'axe `a`

        sh = nrm @ origin
        return ( numpy.concatenate( [ nrm, -nrm ] ),
                 numpy.concatenate( [ hi + sh, -( lo + sh ) ] ) )

    def extra_cuts_per_piece( self, nb_dims ):
        # un morceau est `cellule INTER pavé de la grille`, et un pavé est l'intersection de 2d
        # demi-espaces (voir `Image::_for_each_piece`). Les coupes de la cellule, elles, sont déjà
        # comptées par la capacité de la cellule.
        return 2 * nb_dims

    @property
    def nb_pieces( self ):
        """Total flat cell count for the 1D case `OtPlan1d` consumes (a single `dim`) -- used to
        size `cell_cum_mass` (`nb_pieces + 1`, see `_update_cell_cum_mass`)."""
        return self.shape.static_count()

    def normalized_version( self ):
        # update mass
        mass = self.mass

        # normalize
        if self.target_mass.is_defined:
            return Image(
                nb_dims = self.nb_dims.value,
                shape = self.shape.value,

                values = self.target_mass / mass * self.values,

                origin = self.origin,
                frame = self.frame,
                knots = self.knots,

                current_mass = self.target_mass,
                batch_axes = self.batch_axes,
            )

        return self

    def _update_current_mass( self ):
        # res = RealTensor[ tuple( self.batch_axes ) ]()
        driver.call(
            FfiCodeParallel( name = "mass",
                fwd_code = "image.current_mass( batch_index ) = image( batch_index ).measure();",
                bwd_code = "image( batch_index ).measure_bwd( grad_for_image( batch_index ).values, "
                           "grad_for_image( batch_index ).current_mass );",
            ),
            output_attributes = [ "image.current_mass" ],
            has_dynamic_capacity = False,
            image = self,
        )

    def batch_slice( self, index ):
        # see `Distribution.batch_slice`. Mirrors `Sinogram.image( k )`'s existing unbatched
        # slice: `origin`/`frame`/`knots` are shared geometry (common to every angle), only
        # `values` varies -- `Image.__init__` re-infers `shape`/`nb_dims` from the sliced
        # buffer, nothing else needs restating.
        if not self.batch_axes:
            return None
        kwargs = {}
        for name in ( "origin", "frame", "knots" ):
            attr = getattr( self, name )
            if attr.is_defined:
                kwargs[ name ] = attr.tensor
        return Image( values = self.values.tensor[ index ], **kwargs )

    def ensure_cell_cum_mass( self ):
        """Materializes `cell_cum_mass` (a lazy `ComputedAttribute`, mirrors `mass`/`current_mass`)
        if not already cached. Called once by `OtPlan1d` before it reads `dst_dist.cell_cum_mass` --
        a plain method (not a same-named property) because the FIELD itself must keep the name
        `cell_cum_mass` for the C++ struct it crosses the FFI as."""
        if self.cell_cum_mass.is_undefined:
            self._update_cell_cum_mass()

    def _update_cell_cum_mass( self ):
        # `values`/`frame`/`knots` carry no real gradient THROUGH `cell_cum_mass`: it is a routing
        # helper for `OtPlan1d`'s walk, and `d cost/d values` is already computed there directly (a
        # closed form, `Phi_k`/`second_moment_about`), never via `cell_cum_mass`. `stop_gradient`
        # them going INTO this call so it needs no `bwd_code` at all, and so `cell_cum_mass` itself
        # never carries a gradient trace back to `values` wherever it is read afterwards (once, here
        # -- not at each read site, see `driver.stop_gradient`'s docstring).
        # `origin`/`frame`/`knots` may be unset (`NoneTensor`, defaulted later by `with_defaults`
        # inside `fill_cell_cum_mass`): omitted entirely rather than passed as `None`, so `Image`
        # leaves them unbound instead of trying to `.set( None )` them.
        detached_kwargs = {}
        for name in ( "origin", "frame", "knots" ):
            attr = getattr( self, name )
            if attr.is_defined:
                detached_kwargs[ name ] = driver.stop_gradient( attr.tensor )
        detached = Image(
            nb_dims = self.nb_dims.value,
            shape = self.shape.value,
            values = driver.stop_gradient( self.values.tensor ),
            batch_axes = self.batch_axes,
            **detached_kwargs,
        )

        # prescribe `nb_cells_cum` (the ShapeVar behind `cell_cum_mass`'s declared axis, see its
        # docstring) BEFORE binding the buffer below -- `set_raw` does not observe sizes from the
        # buffer it is handed, it trusts the ShapeVar to already carry the right count.
        self.nb_cells_cum = self.nb_pieces + 1

        # a plain OUTPUT tensor (like `Cell.measure`'s `res`), built from its OWN local axis (a
        # free-standing `RealTensor[...]` cannot resolve a string axis name outside an aggregate's
        # scope) -- not a nested `image.cell_cum_mass` write, so `detached` never needs to carry the
        # result back out itself.
        cum_axis = Axis( ShapeVar( self.nb_pieces + 1 ), name = "num_cell_cum" )
        cell_cum_mass = RealTensor[ *self.batch_axes, cum_axis ]()

        driver.call(
            FfiCodeParallel( name = "cell_cum_mass",
                fwd_code = "image( batch_index ).fill_cell_cum_mass( cell_cum_mass( batch_index ) );",
            ),
            output_attributes = [ "cell_cum_mass" ],
            has_dynamic_capacity = False,
            image = detached,
            cell_cum_mass = cell_cum_mass,
        )
        self.cell_cum_mass.set_raw( cell_cum_mass.raw )

    def try_update_otplan1d( self, plan ):
        """Closed-form, pure-JAX fast path for `OtPlan1d.update_outputs` when `self` is its
        1D piecewise-constant target: bypasses `driver.call`/the C++ kernel entirely (ordinary
        JAX autodiff differentiates straight through `_pure_jax_cost1d.cost_1d_ot`), evaluated
        ~1.2x-6x faster than the C++ kernel from n=1e6 through 1e8 diracs (see
        [[pure-jax-otplan1d]]). Declines (returns False, caller falls back to driver.call)
        when: the backend is not JAX (this path uses `jax.lax.map`/`jnp.argsort` directly, not
        the backend-agnostic `Tensor` algebra), `self` is not 1D, more than one batch axis is
        in play (matches today's ONE `num_angle` axis in production; untested beyond that),
        `plan` wants `barycenters` (not implemented here), or `plan.src_dist` cannot supply
        plain positions/weights (`raw_1d_diracs`, e.g. an unsupported dirac-source type).

        Batched over angles with `jax.lax.map` (sequential), NOT `jax.vmap`: `vmap` batches
        `jnp.argsort`/the fancy-index gathers by materializing a `[ nb_angles * n, ... ]`
        transpose/fusion -- fine at moderate scale, but a genuine OOM at n=1e8 x 5 angles
        (needing >13GiB). `lax.map` processes one angle at a time (no such batched-sort
        buffer) and, measured, is never slower -- often faster -- than `vmap` at every scale
        tried (1e6-1e8), so there is no size threshold to pick between them.
        """
        if plan._with_barycenters:
            return False
        if int( self.nb_dims.value ) != 1:
            return False
        if len( self.batch_axes ) > 1:
            return False
        if driver.framework.module_name != "jax":
            return False

        diracs = plan.src_dist.raw_1d_diracs()
        if diracs is None:
            return False
        weights, batched_extra, project_fn = diracs

        import jax
        from ._pure_jax_cost1d import cost_1d_ot

        values = self.values.tensor
        s_min = self.origin.tensor[ 0 ] if self.origin.is_defined else 0.0
        dw = self.frame.tensor[ 0, 0 ] if self.frame.is_defined else 1.0

        if self.batch_axes:
            # only the genuinely PER-ANGLE leaves are mapped (`lax.map` requires every leaf of
            # its pytree argument to share the same leading/mapped axis) -- shared (unbatched)
            # `weights`, and `batched_extra`'s own projection inputs, are closed over instead
            # where they are not batched, so nothing is ever materialized for more than one
            # angle at a time (see `raw_1d_diracs`'s docstring on why this matters).
            mapped = { "values": values, **batched_extra }
            if weights.ndim > 1:
                mapped[ "weights" ] = weights

            # `jax.checkpoint`: without it, the backward through `lax.map` still retains every
            # angle's forward residuals at once (confirmed: OOMs at n=1e8 x 5 angles needing
            # >13GiB even though the forward itself, thanks to the lazy `project_fn` above,
            # never materializes more than one angle's data) -- recomputing the forward per
            # angle during the backward instead is what actually bounds peak memory to ONE
            # angle's worth, letting this scale to n=1e8 (measured ~920ms/angle, matched
            # against a single-angle-isolated baseline -- i.e. no residual per-angle overhead).
            @jax.checkpoint
            def body( leaves ):
                w = leaves[ "weights" ] if "weights" in leaves else weights
                extra = { k: leaves[ k ] for k in batched_extra }
                p = project_fn( extra )
                return cost_1d_ot( p, w, leaves[ "values" ], s_min, dw )

            cost = jax.lax.map( body, mapped )
        else:
            cost = cost_1d_ot( project_fn( {} ), weights, values, s_min, dw )

        plan.cost = cost
        return True
