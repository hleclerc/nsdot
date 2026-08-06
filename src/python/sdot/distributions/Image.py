from typing import TYPE_CHECKING, cast, overload

from ..tensor.CtShapeVar import CtShapeVar
from ..tensor.ShapeVar import ShapeVar
from ..tensor.AxisList import AxisList
from ..tensor.Tensor import Tensor
from ..tensor.Axis import Axis

from ..util.ComputedAttribute import ComputedAttribute
from ..compilation.FfiCode import FfiCodeParallel
from ..drivers.driver import driver

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

    values           : Tensor[ "img_pos..." ]

    origin           : Tensor[ "dim" ]
    frame            : Tensor[ "dir", "dim" ]
    knots            : Tensor[ "dim", "num_knot" ]

    current_mass     : ComputedAttribute[ Tensor, ( "values", "frame", "knots" ) ]

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
    cell_cum_mass    : ComputedAttribute[ Tensor[ "num_cell_cum" ], ( "values", "frame", "knots" ) ]

    def __init__( self, values, **kwargs ) -> None:
        self.__base_init__( values = values, target_mass = 1.0, **kwargs )

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
        # res = Tensor[ tuple( self.batch_axes ) ]()
        driver.call(
            FfiCodeParallel( name = "mass",
                fwd_code = "image.current_mass( batch_index ) = image( batch_index ).measure();",
                bwd_code = "image( batch_index ).measure_bwd( grad_for_image( batch_index ).values, "
                           "grad_for_image( batch_index ).current_mass );",
            ),
            output_attributes = [ "image.current_mass" ],
            image = self,
        )

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
        # free-standing `Tensor[...]` cannot resolve a string axis name outside an aggregate's
        # scope) -- not a nested `image.cell_cum_mass` write, so `detached` never needs to carry the
        # result back out itself.
        cum_axis = Axis( ShapeVar( self.nb_pieces + 1 ), name = "num_cell_cum" )
        cell_cum_mass = Tensor[ *self.batch_axes, cum_axis ]()

        driver.call(
            FfiCodeParallel( name = "cell_cum_mass",
                fwd_code = "image( batch_index ).fill_cell_cum_mass( cell_cum_mass( batch_index ) );",
            ),
            output_attributes = [ "cell_cum_mass" ],
            image = detached,
            cell_cum_mass = cell_cum_mass,
        )
        self.cell_cum_mass.set_raw( cell_cum_mass.raw )
