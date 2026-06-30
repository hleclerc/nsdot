class ShapeVarInst:
    """Per-instance *cell* for a `ShapeVar` declaration.

    This is the node of the constraint graph, NOT the schema (that is the
    `ShapeVar` decl, shared by all instances). A cell holds only the mutable
    bits: the optional `prescribed_value` and the list of binding scopes
    (`envs`) that reference it.

    Sharing a `ShapeVar` between several aggregate objects = several objects
    whose `_bindings` point to the *same* cell. Each sharer adds its scope to
    `envs`, so the value is solved from the union of the bound tensors across
    all of them (see `value`).
    """

    def __init__( self, decl ):
        self.decl = decl
        self.prescribed_value = decl.prescribed_value
        self.envs = []   # list of `_bindings` dicts (decl -> inst) referencing this cell

    @property
    def value( self ):
        # a prescribed value wins over whatever the tensors would imply
        if self.prescribed_value is not None:
            return self.prescribed_value

        # otherwise solve from the shape of every bound tensor that uses this var,
        # across every scope that shares this cell
        for env in self.envs:
            for tensor_decl in self.decl.usage:
                tensor_inst = env.get( tensor_decl )
                if tensor_inst is not None and tensor_inst.shape is not None:
                    for n, axis in enumerate( tensor_decl.axes ):
                        res = axis.extent.solve_shape_var( self.decl, tensor_inst.shape[ n ] )
                        if res is not None:
                            return res
        return None
