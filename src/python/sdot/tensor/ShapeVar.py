# from typing_extensions import Sequence, overload
from ..util.Parametrized import Parametrized
from ..util.Attribute import Attribute
from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from .Axis import Axis


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

    def __class_getitem__( cls, *deps: str ):
        return Parametrized( cls, *deps )

    def __init__( self, parent, dep_axes : None | list[ str ] = None, /, template_args = [], template_kwargs = {} ) -> None:
        self.dep_axes = []
        # for dep_axis in template_args:
        #     self.dep_axes.append(  )



    # # --- descriptor protocol ------------------------------------------------
    # @overload
    # def __get__( self, obj: None, objtype = None ) -> 'ShapeVar': ...
    # @overload
    # def __get__( self, obj: object, objtype = None ) -> int: ...

    # def __get__( self, obj, objtype = None ):
    #     if obj is None:
    #         return self  # class access (e.g. inside the class body) -> the schema
    #     return obj._bindings[ self ].value

    # def __set__( self, obj, value: int ) -> None:
    #     obj._bindings[ self ].prescribed_value = int( value )

    # def instantiate( self, env, injection = None ) -> ShapeVarInst:
    #     if isinstance( injection, ShapeVar ):
    #         # injected for sharing: every object given this var reuses one cell
    #         if injection._shared_cell is None:
    #             injection._shared_cell = ShapeVarInst( self )
    #         cell = injection._shared_cell
    #     else:
    #         cell = ShapeVarInst( self )
    #         if injection is not None:
    #             cell.prescribed_value = int( injection )
    #     cell.envs.append( env )
    #     return cell

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
