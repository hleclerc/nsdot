from loom.tensor import CtShapeVar
from loom.tensor import ShapeVar
from loom.tensor import Tensor
from loom.tensor import RealTensor
from loom.tensor import Axis
from loom.util import ComputedAttribute

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

    points           : RealTensor[ "num_dirac", "proj_dim" ]
    normal           : RealTensor[ "proj_dim" ]
    weights          : RealTensor[ "num_dirac" ]

    # mass is computed from weights; invalidated when weights change (as in SumOfDiracs)
    current_mass     : ComputedAttribute[ RealTensor, ( "weights" ) ]

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
                weights = RealTensor[ self.num_dirac ].full( self.target_mass / self.nb_diracs.value )

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

    def raw_1d_diracs( self ):
        # see `Distribution.raw_1d_diracs`. `points` [n,proj_dim] is shared; `normal` is
        # `[proj_dim]` (unbatched) or `[*batch,proj_dim]` (one per angle) -- when batched, the
        # projection (`points·normal`) is DEFERRED (`project_fn`, `normal` routed through
        # `batched_extra`) so the caller computes it ONE ANGLE AT A TIME: materializing the
        # full `[*batch,n]` projection upfront (a plain eager `einsum` over every angle at
        # once) is exactly the memory blow-up `Image.try_update_otplan1d`'s `lax.map` exists to
        # avoid -- confirmed as the actual cause of an OOM at n=1e8 x 5 angles that an EARLIER,
        # eager version of this method hit (the lazy `lax.map` alone did not save it, because
        # the projection was already materialized in full before `lax.map` ever ran).
        if not self.weights.is_defined:
            return None
        import jax.numpy as jnp
        points = self.points.tensor
        normal = self.normal.tensor
        weights = self.weights.tensor
        if normal.ndim > 1:
            return weights, { "normal": normal }, lambda extra: jnp.einsum( "n p, p -> n", points, extra[ "normal" ] )
        return weights, {}, lambda extra: jnp.einsum( "n p, p -> n", points, normal )

    def batch_slice( self, index ):
        # see `Distribution.batch_slice`. `points`/`weights` are shared (unaffected by which
        # angle); only `normal` varies, so only it is actually indexed.
        if not self.batch_axes or not self.normal.is_defined:
            return None
        return ProjectedSumOfDiracs(
            points = self.points,
            normal = Tensor.wrap( self.normal.tensor[ index ], [ self.proj_dim.name ] ),
            weights = self.weights if self.weights.is_defined else None,
        )
