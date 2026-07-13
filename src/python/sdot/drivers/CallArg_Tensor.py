from .CallArg import CallArg

class CallArg_Tensor( CallArg ):
    """A tensor buffer crossing the FFI boundary.

    `value` is the concrete input tensor when known (input), else `None` (an output to build).
    `schema` is the `Parametrized` declaration (`Tensor["num_vertex","dim"]`); its `args` are
    the declared axis names, used to resolve the `shape`. Registers itself in `caa.tensors`.

    `shape`:
    * `value` known  -> `value.shape`.
    * output (`value is None`) -> resolved from the declared axes against the sibling
      `ShapeVar` capacities, in `resolve_shape` (needs the owning aggregate).
    """

    name : str
    shape : list

    def __init__( self, call_args_analysis, io_category, name, value = None, schema = None ) -> None:
        super().__init__( io_category )

        self.name = name
        self.value = value
        self.schema = schema
        self.shape = list( value.shape ) if value is not None else None

        call_args_analysis.tensors.append( self )

    def resolve_shape( self, owner ) -> None:
        if self.shape is not None:
            return
        axis_names = self.schema.args if self.schema is not None else []
        self.shape = [ owner.attributes[ a ].extent( owner.attributes ) for a in axis_names ]
