from ..tensor.CtShapeVar import CtShapeVar
from ..tensor.ShapeVar import ShapeVar
from ..tensor.Tensor import Tensor
from ..tensor.Axis import Axis
from ..util.ComputedAttribute import ComputedAttribute

from .Distribution import Distribution


class ProjectedSumOfDiracs( Distribution ):
    """Diracs whose 1D position is the PROJECTION of a shared ambient point onto a per-item normal,
    computed ON THE FLY inside the kernel -- so the `[ nb_angles, n ]` projected positions are NEVER
    materialized (the whole point: at 1e7 diracs x 1000 angles that array is 80GB).

    `points` `[ n, proj_dim ]` is SHARED across the batch (the reconstructed particles, 2D); `normal`
    `[ proj_dim ]` is PER-ITEM (batched over the angles); `weights` `[ n ]` are the masses. The C++
    struct answers the same `position(i)` / `add_position_grad(...)` contract as `SumOfDiracs`, so
    `OtPlan1d` consumes either without knowing which (polymorphism, not a type test): `position(i)`
    is the dot `points(i)·normal`, and the backward ATOMIC-SCATTERS d cost / d position back onto the
    shared `points` gradient (many angles contribute to the same point). See [[per-thread-scratch-facility]].
    """

    _is_dirac_source = True

    nb_diracs        : ShapeVar
    nb_dims          : CtShapeVar     # the OT (projected) space: 1D
    proj_dims        : CtShapeVar     # the ambient space the points live in (2D)

    num_dirac        : Axis[ "nb_diracs" ]
    proj_dim         : Axis[ "proj_dims" ]

    points           : Tensor[ "num_dirac", "proj_dim" ]
    normal           : Tensor[ "proj_dim" ]
    weights          : Tensor[ "num_dirac" ]

    # mass is computed from weights; invalidated when weights change (as in SumOfDiracs)
    current_mass     : ComputedAttribute[ Tensor, ( "weights" ) ]

    def __init__( self, points, normal, weights = None, target_mass = 1.0, proj_dims = 2, **kwargs ):
        self.__base_init__( points = points, normal = normal, weights = weights,
                            target_mass = target_mass, nb_dims = 1, proj_dims = proj_dims, **kwargs )

    def normalized_version( self ):
        # normalize the WEIGHTS only; the positions stay symbolic (points/normal carried untouched).
        mass = self.mass
        if self.target_mass.is_defined:
            if self.weights.is_defined:
                weights = self.target_mass / mass * self.weights
            else:
                # SHARED across the batch (`[ num_dirac ]`, NOT `[ *batch_axes, num_dirac ]`): the diracs
                # are the SAME particles at every angle, only the projection direction differs, so their
                # masses do not depend on the angle. Batching this to [nb_angles, n] would be an 80GB
                # buffer at scale for no reason. See [[projected-source-fusion]].
                weights = Tensor[ self.num_dirac ].full( self.target_mass / self.nb_diracs.value )

            return ProjectedSumOfDiracs(
                nb_diracs = self.nb_diracs.value,
                proj_dims = self.proj_dims.value,

                points = self.points,
                normal = self.normal,
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
