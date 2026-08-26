from typing import TYPE_CHECKING, cast, overload

from loom.tensor import CtShapeVar
from loom.tensor import ShapeVar
from loom.tensor import AxisList
from loom.tensor import RealTensor
from loom.tensor import Axis
from loom.util import ComputedAttribute

from loom.compilation.FfiCode import FfiCodeParallel

from .Distribution import Distribution


class SumOfDiracs( Distribution ):
    """Sum of weighted Dirac point masses."""

    # a 1D-dirac source OtPlan1d can sort/sweep (see the `position(i)` C++ contract); the projected
    # variant carries the same flag, so OtPlan1d dispatches on capability, not on a concrete type.
    _is_dirac_source = True

    nb_diracs        : ShapeVar
    nb_dims          : CtShapeVar

    num_dirac        : Axis[ "nb_diracs" ]
    dim              : Axis[ "nb_dims" ]

    positions        : RealTensor[ "num_dirac", "dim" ]
    weights          : RealTensor[ "num_dirac" ]

    # Mass is computed from weights; when weights change, mass is invalidated
    current_mass     : ComputedAttribute[ RealTensor, ( "weights" ) ]

    def __init__( self, positions, weights = None, target_mass = 1.0, **kwargs ):
        self.__base_init__( positions = positions, weights = weights, target_mass = target_mass, **kwargs )

    def normalized_version( self ):
        # update mass
        mass = self.mass

        # normalize
        if self.target_mass.is_defined:
            if self.weights.is_defined:
                weights = self.target_mass / mass * self.weights
            else:
                weights = RealTensor[ *self.batch_axes, self.num_dirac ].full( self.target_mass / self.nb_diracs.value )


            return SumOfDiracs(
                nb_diracs = self.nb_diracs.value,
                nb_dims = self.nb_dims.value,

                positions = self.positions,
                weights = weights,

                current_mass = self.target_mass,
                batch_axes = self.batch_axes,
            )

        return self

    def _update_current_mass( self ):
        if self.weights.is_defined:
            # reduce over the DIRAC axis ONLY, so any batch axis survives: with genuinely per-batch
            # weights (e.g. one dirac per sinogram bin, weighted by THAT angle's pixel value) the
            # mass differs from one batch element to the next, and `current_mass` is batched to
            # match. A plain `sum()` would collapse it to a scalar that no longer fits the declared
            # (batched) shape -- `normalized_version`'s divide then reads it as ragged and fails.
            self.current_mass = self.weights.sum( axis = self.num_dirac )
        else:
            self.current_mass = self.nb_diracs.value

    def raw_1d_diracs( self ):
        # see `Distribution.raw_1d_diracs`. Only meaningful for the 1D case `OtPlan1d`
        # consumes; `positions` carries a trailing size-1 `dim` axis to drop. `positions`
        # itself may be batched (varies per angle) -- unlike `ProjectedSumOfDiracs`'s
        # projection, reading a batched SLICE of it costs nothing extra to defer, so it goes
        # through `batched_extra`/`project_fn` uniformly rather than a special "already batched
        # positions" case.
        if int( self.nb_dims.value ) != 1 or not self.weights.is_defined:
            return None
        positions = self.positions.tensor[ ..., 0 ]
        weights = self.weights.tensor
        if positions.ndim > 1:
            return weights, { "positions": positions }, lambda extra: extra[ "positions" ]
        return weights, {}, lambda extra: positions
