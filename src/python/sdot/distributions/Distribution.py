from ..util.Aggregate import Aggregate
from ..tensor.Tensor import Tensor


class Distribution( Aggregate ):
    """Base class for probability distributions.

    Subclasses should override `measure` (property) to return the total mass.
    Supports automatic normalization via `normalized_version()` when `target_mass` is set.
    """

    # current_mass   : Tensor...
    target_mass      : Tensor


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
        """For a 1D dirac-source distribution (`_is_dirac_source`): its positions/weights as
        plain, differentiable backend arrays -- `( positions, weights )`, each either
        `[ nb_diracs ]` (shared across the batch) or `[ *batch, nb_diracs ]` (varies per batch
        element). Lets a target distribution's `try_update_otplan1d` bypass `driver.call`
        entirely (ordinary autodiff differentiates straight through). `None` when this
        distribution cannot supply this cheaply (default: unsupported) -- the caller then
        falls back to the general driver.call/C++ path."""
        return None

    def try_update_otplan1d( self, plan ):
        """Attempt to solve `plan` (an `OtPlan1d` with `self` as one of its two
        distributions) without going through `driver.call` -- e.g. a closed-form, pure-JAX
        computation. On success: update `plan`'s output fields (at least `plan.cost`) and
        return True. On failure (unsupported combination): change nothing and return False,
        so the caller uses the general driver.call/C++ path instead. Default: always decline
        (default: unsupported)."""
        return False
