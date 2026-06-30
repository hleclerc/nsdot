from numpy.typing import ArrayLike

from ..drivers.driver import driver


class TensorInst:
    """Per-instance value for a `Tensor` declaration.

    Holds the concrete `raw` array (in the chosen backend) and its resolved
    `shape`. Everything structural (axes, dtype, device, name) is read through
    `self.decl` — nothing is copied from the declaration.
    """

    def __init__( self, decl ):
        self.decl = decl
        self.raw = None
        self.shape = None

    def reassign( self, value: ArrayLike | None ):
        from .Tensor import Tensor

        if isinstance( value, TensorInst ):
            value = value.raw  # TODO: shape, shape vars, ...
        elif isinstance( value, Tensor ):
            value = None       # a bare declaration carries no value

        if value is None:
            self.raw = None
            self.shape = None
            return

        self.raw = driver.array( value, dtype = self.decl.dtype, device = self.decl.device )
        self.shape = [ int( i ) for i in self.raw.shape ]

    @property
    def value( self ):
        return self.raw

    def __repr__( self ):
        return f"{ self.decl.name or 'tensor' }[ shape={ self.shape } ]"
