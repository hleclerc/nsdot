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
