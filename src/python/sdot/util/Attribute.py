from .Parametrized import Parametrized

class Attribute:
    """Base protocol for `@aggregate` field declarations.

    This is the *only* thing `@aggregate` knows about: it operates on classes
    whose fields are `Attribute`s, with no knowledge of any concrete declaration
    type (`ShapeVar`, `Tensor`, ... are just examples). A declaration is a
    class-level descriptor -- the immutable, shared *schema* of a field.
    Per-instance mutable state lives in the object returned by `instantiate`,
    stored in the instance's `_bindings` map (`decl -> inst`); that object merely
    *references* its declaration, nothing is copied.

    Subclasses implement the descriptor protocol (`__get__` / `__set__`) to expose
    whatever per-instance *view* the user manipulates (an `int`, a tensor wrapper,
    ...), and override `instantiate` when they carry per-instance state.
    """

    # def __class_getitem__( cls, *deps: str ):
    #     return Parametrized( cls, *deps )
    pass
