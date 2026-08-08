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
