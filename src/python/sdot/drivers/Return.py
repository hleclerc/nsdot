from .IoCategory import IoCategory
from .CtorArgs import CtorArgs

class Return:
    """Marks a `driver.call` kwarg as an *object to construct and return*.

    `Return( Cell, max_of_nb_vertices = 8, nb_dims = 2 )` describes a `Cell` output: the
    kwargs are not inputs, they *initialize* the outputs (Jax has no mutable tensors, so the
    kernel seeds its outputs from them). Everything under a `Return` is therefore in the
    output category.

    An initializer aimed at a single nested aggregate goes under its name (`left = { "nb_dims":
    3 }`); anything left at this level reaches every field below it (see `CtorArgs`).
    """

    def __init__( self, klass, **kwargs ) -> None:
        self.kwargs = kwargs
        self.klass = klass

    @classmethod
    def make_CallArg( cls, call_args_analysis, io_category, name, value, ctor_args ):
        # `value` is the Return instance; decompose its target class with value=None (object
        # to build) and the kwargs as ctor_args (initializers picked up by each field by name).
        return call_args_analysis.make_CallArg(
            IoCategory.for_return(), name, value.klass, None, CtorArgs( value.kwargs ),
        )
