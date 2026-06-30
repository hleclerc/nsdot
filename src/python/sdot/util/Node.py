from .Reassignable import Reassignable


class Node:
    """Base class for objects that hold Attribute fields (Tensor, ShapeVar, Axis, ...).

    Overrides __setattr__ so that assigning to an existing Attribute field routes to
    `field.value = ...` rather than replacing the instance — matching C++ assignment
    semantics and preventing accidental instance replacement.
    """

    def __setattr__( self, name, value ):
        existing = self.__dict__.get( name )
        if isinstance( existing, Reassignable ):
            existing.reassign( value )
        else:
            object.__setattr__( self, name, value )
