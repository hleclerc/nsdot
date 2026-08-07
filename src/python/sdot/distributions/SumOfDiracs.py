from typing import TYPE_CHECKING, cast, overload

from ..tensor.CtShapeVar import CtShapeVar
from ..tensor.ShapeVar import ShapeVar
from ..tensor.AxisList import AxisList
from ..tensor.Tensor import Tensor
from ..tensor.Axis import Axis
from ..util.ComputedAttribute import ComputedAttribute

from ..compilation.FfiCode import FfiCodeParallel

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

    positions        : Tensor[ "num_dirac", "dim" ]
    weights          : Tensor[ "num_dirac" ]

    # Mass is computed from weights; when weights change, mass is invalidated
    current_mass     : ComputedAttribute[ Tensor, ( "weights" ) ]

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
                weights = Tensor[ *self.batch_axes, self.num_dirac ].full( self.target_mass / self.nb_diracs.value )


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
            self.current_mass = self.weights.sum()
        else:
            self.current_mass = self.nb_diracs.value

    def raw_1d_diracs( self ):
        # see `Distribution.raw_1d_diracs`. Only meaningful for the 1D case `OtPlan1d`
        # consumes; `positions` carries a trailing size-1 `dim` axis to drop.
        if int( self.nb_dims.value ) != 1 or not self.weights.is_defined:
            return None
        return self.positions.tensor[ ..., 0 ], self.weights.tensor
