from ..tensor.AbstractAxis import AbstractAxis
from .CallArg import CallArg

class CallArg_Axis( CallArg ):
    """A declared tensor dimension: resolvable, but NOT a buffer (stays out of `caa.tensors`).

    Holds its affine extent, parsed from the declaration (`Axis["2*nb_dims+1"]`): a map
    `{ shape_var_name: coeff }` plus an `offset`. `extent(attributes)` evaluates it over the
    sibling ShapeVar capacities -- the axis computes its own extent, mirroring `Axis.max`.
    Collected in `caa.axes` for the C++ `DEFINE_AXIS(...)` generation.
    """

    name : str

    def __init__( self, call_args_analysis, io_category, name, expr ) -> None:
        super().__init__( io_category )

        self.name = name
        self.coeffs, self.offset = AbstractAxis.parse_affine( expr )

        call_args_analysis.axes.append( self )

    def extent( self, attributes ):
        extent = self.offset
        for shape_var_name, coeff in self.coeffs.items():
            extent += coeff * int( attributes[ shape_var_name ].capacity )
        return extent

    # -- driver-agnostic C++ (DEFINE_AXIS is the same for every driver) --
    def cpp_define( self ):
        # the same axis name may appear several times in `ca.axes` (declared in different
        # tensors / aggregates): guard each DEFINE_AXIS so it lands at most once per TU.
        guard = f"SDOT_AXIS_{ self.name }"
        return ( f"#ifndef { guard }\n"
                 f"#define { guard }\n"
                 f"DEFINE_AXIS( { self.name } );\n"
                 f"#endif" )
