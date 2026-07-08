# from typing_extensions import Sequence, overload
from ..util.Parametrized import Parametrized
from ..util.aggregate import get_attribute
from ..util.Attribute import Attribute
from typing import TYPE_CHECKING

class ShapeVar( Attribute ):
    """A free (symbolic) integer variable used to build shapes.

    This is a *declaration* (shared, class-level schema) and a typed descriptor:
    on an instance, `c.nb_dims` reads as an `int` (the prescribed or solved
    value) and `c.nb_dims = 1222` prescribes it. The per-instance state lives in
    a `ShapeVarInst` cell, reached through the instance's `_bindings`.

    Axis extents are *expressions* built on top of `ShapeVar`s (e.g.
    `nb_dims + 1`). A `ShapeVar` is either prescribed or solved from the shapes
    of the declared tensors that use it (`usage`).

    Sharing: pass a `ShapeVar` to several constructors (`Cell( nb_dims = n )`).
    The objects then point to the same cell, so the value is solved from the
    union of their bound tensors (see `instantiate`).

    - rank 0 (`shape=None`): a scalar unknown.
    - rank >= 1 (`shape=[ ... ]`): a vector of unknowns indexed by the given axes.
    """

    if TYPE_CHECKING:
        def __set__( self, obj, value: int ) -> None: ...

    def __init__( self, parent_inst = None, /, template_args = [], template_kwargs = {} ) -> None:
        from .Axis import Axis

        self.dep_axes = []
        for dep_axis in template_args:
            axis = get_attribute( dep_axis, parent_inst )
            assert isinstance( axis, Axis )
            self.dep_axes.append( axis )

        self.prescribed_value = None

    def set( self, value ):
        if isinstance( value, ShapeVar ):
            value = value.value
        self.prescribed_value = int( value )

    @property
    def value( self ):
        if self.prescribed_value is not None:
            return self.prescribed_value
        raise NotImplementedError


    # def register_tensor( self, tensor_decl ):
    #     if tensor_decl not in self.usage:
    #         self.usage.append( tensor_decl )

    # # --- ShapeExpr ----------------------------------------------------------
    # @property
    # def rank( self ):
    #     return len( self.shape )

    # def to_affine( self ) -> AffineShapeExpr:
    #     return AffineShapeExpr( terms = { self: 1 }, offset = 0 )

    # def __repr__( self ):
    #     return f"{ self.name or 'shape_var' }[ shape={ self.shape } ]"
