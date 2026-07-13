from .ShapeVar import ShapeVar


class CtShapeVar( ShapeVar ):
    """A `ShapeVar` whose value is known at C++ *compile* time.

    Emitted as a compile-time constant (a literal `static constexpr`) in the generated source
    rather than crossing the FFI as a runtime buffer. Because the value is baked into the
    source, it is part of the library-name hash -- each distinct value yields a distinct
    compiled library (enabling template specialization, loop unrolling, ...).

    It must be *prescribed* (`<name> = v`); a reservation (`max_of_`) denotes a runtime count
    and is therefore rejected.
    """

    @classmethod
    def make_CallArg( cls, caa, io_category, name, value, ctor_args, schema = None ):
        from ..drivers.CallArg_CtShapeVar import CallArg_CtShapeVar
        if ctor_args.has( name, "max_of_" ):
            raise ValueError( f"CtShapeVar '{ name }' is compile-time: it cannot be reserved (max_of_{ name })" )
        return CallArg_CtShapeVar( caa, io_category, name, value = ctor_args.find( name ) )
