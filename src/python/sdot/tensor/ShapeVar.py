from typing_extensions import Sequence, overload
from typing import TYPE_CHECKING

from .ShapeExpr import ShapeExpr

from ..util.Attribute import Attribute


if TYPE_CHECKING:
    from .Axis import Axis


class ShapeVar( Attribute, ShapeExpr ):
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

    def __init__( self, shape : None | Sequence[ 'Axis' ] = None, name = None, prescribed_value = None ) -> None:
        if shape is None:
            shape = []

        self.prescribed_value = prescribed_value
        self.shape = list( shape )
        self.name = name          # if None, set by `Attribute.__set_name__`

        self.usage = []           # `Tensor` *declarations* that reference this var
        self._shared_cell = None  # set when injected, so every injectee shares one cell

    # --- descriptor protocol ------------------------------------------------
    @overload
    def __get__( self, obj: None, objtype = None ) -> 'ShapeVar': ...
    @overload
    def __get__( self, obj: object, objtype = None ) -> int: ...

    def __get__( self, obj, objtype = None ):
        if obj is None:
            return self  # class access (e.g. inside the class body) -> the schema
        return obj._bindings[ self ].value

    def __set__( self, obj, value: int ) -> None:
        obj._bindings[ self ].prescribed_value = int( value )

    def instantiate( self, env, injection = None ) -> ShapeVarInst:
        if isinstance( injection, ShapeVar ):
            # injected for sharing: every object given this var reuses one cell
            if injection._shared_cell is None:
                injection._shared_cell = ShapeVarInst( self )
            cell = injection._shared_cell
        else:
            cell = ShapeVarInst( self )
            if injection is not None:
                cell.prescribed_value = int( injection )
        cell.envs.append( env )
        return cell

    def register_tensor( self, tensor_decl ):
        if tensor_decl not in self.usage:
            self.usage.append( tensor_decl )

    # --- ShapeExpr ----------------------------------------------------------
    @property
    def rank( self ):
        return len( self.shape )

    def to_affine( self ) -> AffineShapeExpr:
        return AffineShapeExpr( terms = { self: 1 }, offset = 0 )

    def __repr__( self ):
        return f"{ self.name or 'shape_var' }[ shape={ self.shape } ]"
