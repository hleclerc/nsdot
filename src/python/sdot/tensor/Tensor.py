from typing_extensions import overload
from numpy.typing import ArrayLike
from typing import TYPE_CHECKING
import numpy

from sdot.tensor.ShapeVar import ShapeVar

from ..util.Attribute import Attribute, resolve_attribute

from ..drivers.driver import driver

from ..devices.Device import Device

from .AbstractAxis import AbstractAxis
from .AxisList import AxisList
from .Dtype import Dtype
from .Axis import Axis


class Tensor( Attribute ):
    """
    Tensor declaration: a thin wrapper around the backend tensor of the chosen
    library (Jax, Torch, ...).

    Inside an `@aggregate`, one `Tensor` is created per parent instance (see
    `get_attribute`) and holds that instance's state: `c.frame = ...` goes
    through `set` and fills `_raw` (a homogeneous driver tensor).

    But a `Tensor` needs no aggregate: it is only a scope in which the NAMES of
    its axes are looked up. Give it the axes themselves (or none) and it stands
    alone, `t.value` playing the part `c.frame` plays above:

        t = Tensor( 17 )                        # rank 0, no declared axis
        t = Tensor[ { "dtype": int } ]( [ 1, 2 ] )
        t = Tensor[ x, y ]( [ [ 1, 2 ] ] )      # x, y being `Axis` objects

    The logical contract is the axis list; axis extents may depend on other axes
    (RAGGED axes), in which case the varying sizes live in the `ShapeVar`s of
    rank > 0. With no declared axis at all, there is nothing to solve and the
    buffer IS the contract (see `shape`). The physical contract (padding / order
    / alignment, per device) is to come, as template kwargs, kept separate from
    the axis list.
    """

    if TYPE_CHECKING:
        def __set__( self, obj, value: ArrayLike | None ) -> None: ...


    def __init__( self, value = None, /, *, template_args = (), template_kwargs = {}, scope = None ) -> None:
        self.device = Device.factory( template_kwargs.get( "device", None ) )
        self.dtype = Dtype.factory( template_kwargs.get( "dtype", None ) )
        self._raw = None          # the backing value, the SINGLE source of truth for what we are:
                                  #   None            -> no value        (lowers to a `NoneTensor`)
                                  #   symbolic zero   -> reads as 0       (lowers to a `ZeroTensor`)
                                  #   a real buffer   -> our data, padded when ragged (`TensorView`)
                                  # A symbolic zero is the framework's own bufferless-zero object
                                  # (`driver.symbolic_zero` / `is_symbolic_zero`), so there is no
                                  # separate flag to keep in sync -- see `raw` (materialized buffer).

        # Our whole logical contract, and -- with `device`, `dtype`, `_raw` -- our WHOLE state: a
        # list of `AbstractAxis`. Everything else (per-dim positions, sizes, names) is derived from
        # these four (see `_spec_dims`, `shape`, `_dim_names`); the observed sizes live on the axes'
        # ShapeVars, not here. An axis is a NAME looked up in `scope`, or an `AbstractAxis` object --
        # a tensor can then BORROW an axis (`Tensor[ cell.num_vertex ]`) without any aggregate. An
        # `AxisList` is unrolled (spanning `nb_dims` array dimensions); that it is, is derived from
        # its TYPE (`_is_unrolled`), so a trailing `...` on its name is only checked, never stored.
        self.axes = []
        for entry in template_args:
            if isinstance( entry, str ):
                unroll = entry.endswith( "..." )
                axis = resolve_attribute( entry[ :-3 ] if unroll else entry, scope, AbstractAxis )
            else:
                if isinstance( entry, ShapeVar ):
                    entry = Axis( entry )
                else:
                    assert isinstance( entry, AbstractAxis )
                unroll = False
                axis = entry
            assert unroll == _is_unrolled( axis ), "only an AxisList is unrolled, and it must be ('name...')"
            self.axes.append( axis )
        assert sum( _is_unrolled( a ) for a in self.axes ) <= 1, "at most one unrolled AxisList per tensor"

        # let each member record, on its ShapeVars, how this tensor constrains their capacity
        for index, axis in enumerate( self.axes ):
            axis.register_in( self, index, _is_unrolled( axis ) )

        if value is not None:
            self.set( value )

    @classmethod
    def make_CallArg( cls, caa, path, name, inst ):
        from ..drivers.CallArg_Tensor import CallArg_Tensor
        return CallArg_Tensor( caa, path, name, inst )

    @classmethod
    def like( cls, other ) -> "Tensor":
        """An empty tensor sharing `other`'s axes, dtype and device -- what the backward pass
        builds to carry a gradient (same logical shape as the value it is the gradient of). What
        goes INTO it then decides its kind: a real cotangent buffer (`set_raw`) makes it a
        `TensorView`, a symbolic-zero cotangent a `ZeroTensor`, nothing at all a `NoneTensor`."""
        res = cls( template_kwargs = { "dtype": other.dtype, "device": other.device } )
        res.axes = list( other.axes )   # same axes, but not re-registered on their ShapeVars
        return res

    @property
    def is_symbolic_zero( self ) -> bool:
        # DERIVED from `_raw` alone: it IS the framework's bufferless-zero object, so there is no
        # flag that could contradict the buffer. A materialized buffer is never a symbolic zero.
        return self._raw is not None and driver.is_symbolic_zero( self._raw )

    def set( self, value ):
        if isinstance( value, Tensor ):
            self._raw = value._raw           # carries the kind along (buffer / symbolic zero / None)
            # re-observe our OWN axes from the source's logical extents, so `shape` resolves even
            # when we do not share the source's axis objects. Only for a plain dense copy: a ragged
            # or unrolled source is well-defined here only through SHARED axes (whose ShapeVars
            # already hold the per-segment counts), so we leave those untouched rather than crush
            # them with a bounding-box max.
            if value.raw is not None and not self._has_unroll() and all( _axis_rank( a ) == 0 for a in self.axes ):
                for axis, size in zip( self.axes, value.shape ):
                    axis.observe_size( numpy.array( size, dtype = int ) )
            return

        if any( _is_unrolled( a ) for a in self.axes ):
            return self._set_unrolled( value )

        # Sizes are read from a *shape tree* (list nesting + `.shape` metadata), so `value`'s data
        # is never touched -- a GPU tensor is not moved. One size per axis; its rank is fixed by the
        # declaration (a dense axis collapses to a scalar, a ragged one keeps the per-segment
        # lengths). Each is PUSHED onto the axis' ShapeVars (`observe_size`), not cached here.
        tree  = _shape_tree( value )
        ranks = [ _axis_rank( axis ) for axis in self.axes ]
        sizes = [
            numpy.array( _collapse( _query( tree, d ), ranks[ d ] ), dtype = int )
            for d in range( len( self.axes ) )
        ]
        for axis, size in zip( self.axes, sizes ):
            axis.observe_size( size )

        # values are stored in a tensor of rank = nb of axes (no compression).
        # Dense maps directly; ragged is assembled into a padded buffer whose per
        # axis capacity is (for now) the max observed size.
        if all( r == 0 for r in ranks ):
            self._raw = driver.array( value, dtype = self.dtype, device = self.device )
        else:
            caps = [ int( size.max() ) for size in sizes ]
            self._raw = _assemble( value, caps, self.dtype, self.device )

    def _set_unrolled( self, value ):
        """Assign a tensor with one unrolled AxisList: its rank is dynamic. Each plain axis
        before/after keeps a single array dimension; the AxisList spans the remaining ones, so the
        value is a dense array (a ragged non-unrolled axis would need nested lists, which this case
        deliberately excludes).

        `nb_dims` is observed from the number of spanned dimensions; each spanned size feeds one
        loop index of the AxisList's ShapeVars; each plain axis from its own (scalar) dimension."""
        tree = _shape_tree( value )
        assert isinstance( tree, tuple ), "an unrolled tensor expects a dense array value"

        # the unroll spans every array dimension not taken by a plain axis
        count = len( tree ) - ( len( self.axes ) - 1 )
        assert count >= 0, "value rank too small for the declared axes"

        sizes = [ numpy.array( s, dtype = int ) for s in tree ]
        self._raw = driver.array( value, dtype = self.dtype, device = self.device )

        # `_spec_dims` is now derivable (`_raw` is set): push each observation onto its axis.
        spec_dims = self._spec_dims()
        for index, axis in enumerate( self.axes ):
            start = spec_dims[ index ]
            if _is_unrolled( axis ):
                axis.observe_span( sizes[ start : start + count ] )
            else:
                axis.observe_size( sizes[ start ] )

    def set_raw( self, raw ):
        """Bind the buffer a kernel produced (a driver tensor). Sizes stay unobserved: an
        output's extents are the ones we ASKED for (`shape`), and its counts live in the
        ShapeVars the kernel wrote -- there is nothing to solve from the data.

        `raw` may also be a symbolic-zero cotangent handed back by the framework: it lands here the
        same way, and `is_symbolic_zero` recognizes it -- no special case."""
        self._raw = raw

    @property
    def capacity( self ):
        """What our buffer IS: the allocated extents, read off it. An input is bound at THIS
        size -- an output that wants to grow must not force us to inflate the input."""
        if self.raw is None:
            return [ 0 ] * self.rank
        return self.raw.shape

    @property
    def allocated_sizes( self ):
        """One allocated size per ARRAY dimension, read off the buffer: what a `ShapeVar` inverts to
        learn the capacity it was allocated with (its `_raw.shape`). `None` while unbound."""
        if self.raw is None:
            return None
        return [ numpy.array( s, dtype = int ) for s in self.raw.shape ]

    # ---- per-dimension geometry, DERIVED from the axes + `_raw` (no cached field) ----
    def _axis_array_dims( self, axis ):
        """How many ARRAY dimensions `axis` occupies: 1 for a plain `Axis`, `nb_dims` (the unroll
        count) for an unrolled `AxisList`."""
        return self._unroll_count() if _is_unrolled( axis ) else 1

    def _unroll_count( self ):
        """The number of array dimensions the (single) unrolled `AxisList` spans -- the rest of the
        buffer's rank once each plain axis has taken one dimension. `None` while unbound."""
        if self._raw is None:
            return None
        return self._raw.ndim - ( len( self.axes ) - 1 )

    def _spec_dims( self ):
        """The first array dimension of each declared axis (accounting for an unrolled AxisList
        sibling spanning several). Derived on the fly from the axes' dimension counts."""
        dims, d = [], 0
        for axis in self.axes:
            dims.append( d )
            d += self._axis_array_dims( axis )
        return dims

    def _unroll_span( self, index ):
        """`( start, count )` of the unrolled `AxisList` at `index`, or `None` while its count is
        not yet knowable (unbound)."""
        count = self._unroll_count()
        return None if count is None else ( self._spec_dims()[ index ], count )

    def _has_unroll( self ):
        return any( _is_unrolled( a ) for a in self.axes )

    @property
    def shape( self ):
        # with no declared axis there is no expression to evaluate: the buffer is the whole contract
        # (a standalone `Tensor( [ 1, 2 ] )`, or a derived tensor with no named axis), and it has
        # none while unvalued. `_raw.shape` also serves a symbolic zero (it carries its shape).
        if not self.axes:
            return list( self._raw.shape ) if self._raw is not None else []

        # each member contributes a LIST of extents (one for an `Axis`, `nb_dims`
        # for an unrolled `AxisList`); concatenation gives the tensor's extents.
        res = []
        for axis in self.axes:
            res += axis.max_list()
        return res

    @property
    def rank( self ):
        if not self.axes:
            return self._raw.ndim if self._raw is not None else 0
        return len( self.axes )

    @property
    def raw( self ):
        """The MATERIALIZED buffer, or `None` when there is nothing to bind -- a symbolic zero has
        no storage, so it reads as `None` here (that is how `is_bound` stays false for it), while
        `_raw` still holds the framework's zero object for `is_symbolic_zero` to recognize."""
        return None if self.is_symbolic_zero else self._raw

    @property
    def tensor( self ):
        """The dense VIEW of `raw`: its logical region, with the capacity padding cropped off.

        `raw` is a homogeneous buffer sized at CAPACITY -- padding included -- because that is what
        a kernel writes into; `tensor` slices it back to the logical `shape`, which is what one
        usually wants to read (`c.vertex_positions.tensor` instead of `c.vertex_positions.raw[ :n ]`).

        Meaningful for a DENSE (non-ragged) tensor: a ragged one has no single box to extract, so
        this returns its bounding box (inner padding kept). Needs a statically known `shape`, so it
        holds eagerly -- a kernel-written count is a device value under a trace, where Python cannot
        slice by it (`shape` raises there). A symbolic zero has no buffer to view -> `None`."""
        if self.raw is None:
            return None
        return self.raw[ tuple( slice( 0, s ) for s in self.shape ) ]

    @property
    def value( self ):
        """The LOGICAL data, backend array: `tensor` (padding cropped), which is what one wants
        when reading a tensor as a value -- the same role `c.nb_dims.value` plays for a `ShapeVar`.
        `raw` stays the padded buffer the FFI needs."""
        return self.tensor

    @value.setter
    def value( self, value ):
        self.set( value )

    # ------------------------------------------------------------------ derived tensors
    # Every op below (operators, reductions, slicing) reads the LOGICAL values (`self.tensor`,
    # padding cropped) and returns a fresh DERIVED tensor built by `_wrap`. A derived tensor is no
    # special case: it carries a real list of `AbstractAxis` like any other. Each surviving dimension
    # gets a fresh DEFAULT axis -- a plain `Axis` over a new `ShapeVar`, observed straight from the
    # (dense) buffer, carrying the inherited name if any. So `shape` / `_dim_names` derive uniformly
    # from the axes, named reductions / slices keep working down the chain, and a partial slice is
    # correct (the fresh axis holds the sliced size, not the original's stale one).

    def _wrap( self, raw, names ):
        return type( self ).wrap( raw, names, dtype = self.dtype, device = self.device )

    @classmethod
    def wrap( cls, raw, names = None, dtype = None, device = None ):
        """A DETACHED tensor around an existing backend buffer: no aggregate, fresh DEFAULT axes
        (one per array dimension, named from `names`, observed straight from the dense buffer). This
        is how an op result is built (`_wrap`), and how a `ShapeVar` hands its count back as a
        `Tensor` -- the buffer is the whole contract, the axes carry the names."""
        res = cls( template_kwargs = { "dtype": dtype, "device": device } )
        res._raw = raw
        if names is not None:
            axes = []
            for index, name in enumerate( names ):
                axis = Axis( ShapeVar() )
                axis.name = name
                if raw is not None and not driver.is_symbolic_zero( raw ):
                    axis.observe_size( numpy.array( raw.shape[ index ], dtype = int ) )
                axes.append( axis )
            res.axes = axes
        return res

    def _dim_names( self ):
        """One axis name (or `None`) per ARRAY dimension, read uniformly off the axes (an unrolled
        AxisList spreads its name over its spanned dimensions)."""
        names = []
        for axis in self.axes:
            if _is_unrolled( axis ):
                count = self._unroll_count()
                count = count if count is not None else len( axis.max_list() )
                names += [ axis.name ] * count
            else:
                names.append( axis.name )
        return names

    def _axis_pos( self, key ):
        """A dimension index, from an int (returned as is) or an axis NAME (looked up in
        `_dim_names`)."""
        if isinstance( key, str ):
            names = self._dim_names()
            if key not in names:
                raise ValueError( f"no axis named '{ key }' in { names }" )
            return names.index( key )
        return int( key )

    # ---- array protocol: makes `numpy.asarray(t)`, `int(t)`, `list(t)`, `assert t == x` work ----
    def __array__( self, dtype = None ):
        arr = numpy.asarray( self.tensor )
        return arr.astype( dtype ) if dtype is not None else arr

    def __int__( self ):
        return int( numpy.asarray( self.tensor ) )

    def __float__( self ):
        return float( numpy.asarray( self.tensor ) )

    def __bool__( self ):
        return bool( numpy.asarray( self.tensor ) )

    def __len__( self ):
        if self.rank == 0:
            raise TypeError( "len() of a rank-0 tensor" )   # numpy's convention for a 0-d value
        return self.shape[ 0 ]

    def __iter__( self ):
        for i in range( len( self ) ):
            yield self[ i ]

    # ---- elementwise operators (scalar or same-shape `Tensor`/array; result keeps our names) ----
    def _binary( self, other, op ):
        b = other.tensor if isinstance( other, Tensor ) else other
        return self._wrap( op( self.tensor, b ), self._dim_names() )

    def __add__     ( self, o ): return self._binary( o, lambda a, b: a +  b )
    def __radd__    ( self, o ): return self._binary( o, lambda a, b: b +  a )
    def __sub__     ( self, o ): return self._binary( o, lambda a, b: a -  b )
    def __rsub__    ( self, o ): return self._binary( o, lambda a, b: b -  a )
    def __mul__     ( self, o ): return self._binary( o, lambda a, b: a *  b )
    def __rmul__    ( self, o ): return self._binary( o, lambda a, b: b *  a )
    def __truediv__ ( self, o ): return self._binary( o, lambda a, b: a /  b )
    def __rtruediv__( self, o ): return self._binary( o, lambda a, b: b /  a )
    def __floordiv__( self, o ): return self._binary( o, lambda a, b: a // b )
    def __mod__     ( self, o ): return self._binary( o, lambda a, b: a %  b )
    def __pow__     ( self, o ): return self._binary( o, lambda a, b: a ** b )
    def __matmul__  ( self, o ): return self._binary( o, lambda a, b: a @  b )

    def __neg__( self ): return self._wrap( -self.tensor, self._dim_names() )
    def __abs__( self ): return self._wrap( abs( self.tensor ), self._dim_names() )

    def __eq__( self, o ): return self._binary( o, lambda a, b: a == b )
    def __ne__( self, o ): return self._binary( o, lambda a, b: a != b )
    def __lt__( self, o ): return self._binary( o, lambda a, b: a <  b )
    def __le__( self, o ): return self._binary( o, lambda a, b: a <= b )
    def __gt__( self, o ): return self._binary( o, lambda a, b: a >  b )
    def __ge__( self, o ): return self._binary( o, lambda a, b: a >= b )

    # `__eq__` returns a Tensor (elementwise), so instances are no longer value-comparable as keys;
    # keep IDENTITY hashing (a `Tensor` is never used as a by-value dict key -- `ShapeVar` is).
    __hash__ = object.__hash__

    # ---- reductions (`axis` = None / int / axis name / a tuple of those) ----
    def _reduce( self, op, axis, identity ):
        """Reduce over the LOGICAL values. A ragged tensor's bounding box (`tensor`) has HOLES
        (padding) that would corrupt the result -- a 0 surviving a `max`, a 1 lost in a `prod` -- so
        they are first filled with the operation's IDENTITY. A dense (or unrolled) tensor has no
        holes: `_hole_mask` returns `None` and the fast path is untouched."""
        data  = self.tensor
        holes = self._hole_mask()
        if holes is not None:
            data = driver.where( holes, identity, data )
        if axis is None:
            return self._wrap( op( data ), [] )
        keys = axis if isinstance( axis, ( tuple, list ) ) else ( axis, )
        pos  = tuple( self._axis_pos( k ) for k in keys )
        names = [ n for d, n in enumerate( self._dim_names() ) if d not in pos ]
        return self._wrap( op( data, axis = pos ), names )

    def sum ( self, axis = None ): return self._reduce( driver.sum,  axis, 0 )
    def prod( self, axis = None ): return self._reduce( driver.prod, axis, 1 )
    def max ( self, axis = None ): return self._reduce( driver.max,  axis, -numpy.inf )
    def min ( self, axis = None ): return self._reduce( driver.min,  axis,  numpy.inf )
    def all ( self, axis = None ): return self._reduce( driver.all,  axis, True )
    def any ( self, axis = None ): return self._reduce( driver.any,  axis, False )

    def mean( self, axis = None ):
        # holes filled with 0 make the SUM correct; divide by the count of REAL cells, not the box.
        return self.sum( axis ) / self._valid_counts( axis )

    def _valid_counts( self, axis ):
        """A `Tensor` of how many non-hole cells fall along the reduced dims (the whole box when
        dense) -- the denominator `mean` divides by."""
        holes = self._hole_mask()
        valid = numpy.ones( tuple( self.shape ), dtype = int ) if holes is None else ( ~holes ).astype( int )
        if axis is None:
            return self._wrap( driver.array( int( valid.sum() ), dtype = int ), [] )
        keys = axis if isinstance( axis, ( tuple, list ) ) else ( axis, )
        pos  = tuple( self._axis_pos( k ) for k in keys )
        names = [ n for d, n in enumerate( self._dim_names() ) if d not in pos ]
        return self._wrap( driver.array( valid.sum( axis = pos ), dtype = int ), names )

    def _hole_mask( self ):
        """A boolean array over the bounding box (`shape`), True at each PADDING position -- the
        holes a ragged tensor leaves inside its box. `None` when there is nothing to mask (a dense
        or unrolled tensor: its box IS its data), so the common path stays allocation-free.

        Built eagerly, cell by cell, from the axes' live extents (like the `__repr__` display): a
        reduction already needs a static `shape`, so an eager mask fits here."""
        if self._has_unroll() or not any( _axis_rank( a ) > 0 for a in self.axes ):
            return None
        shape = tuple( self.shape )
        mask  = numpy.zeros( shape, dtype = bool )
        for idx in numpy.ndindex( *shape ):
            if not _cell_valid( self.axes, idx ):
                mask[ idx ] = True
        return mask

    # ---- indexing: numpy-positional, or ( "axis_name", index ) to select by name ----
    def __getitem__( self, key ):
        names = self._dim_names()
        if isinstance( key, tuple ) and len( key ) and isinstance( key[ 0 ], str ):
            name, idx = key
            pos = self._axis_pos( name )
            key = tuple( idx if d == pos else slice( None ) for d in range( self.rank ) )
        elif not isinstance( key, tuple ):
            key = ( key, )
        # pad the trailing dimensions with full slices, then track which axes survive (an int index
        # drops its dimension; a slice / array keeps it) to carry the right names onto the result.
        key = key + ( slice( None ), ) * ( self.rank - len( key ) )
        result = self.tensor[ key ]
        new_names = [ names[ d ] if d < len( names ) else None
                      for d, k in enumerate( key ) if not isinstance( k, int ) ]
        return self._wrap( result, new_names )

    def __repr__( self ):
        names  = self._dim_names()
        axes   = "" if all( n is None for n in names ) else f", axes={ names }"
        kind   = ", symbolic_zero" if self.is_symbolic_zero else ""
        header = f"Tensor( shape={ self.shape }{ axes }, dtype={ self.dtype.name }, device={ self.device }{ kind } )"
        if self.raw is None:
            return header

        raw = numpy.asarray( self.raw )
        # an unrolled AxisList is always fully dense (no reservation, no padding);
        # otherwise mask out padding cell by cell, from the axes' LIVE extents.
        tree = raw.tolist() if self._has_unroll() else _display_tree( raw, self.axes )
        width = max( ( len( _fmt_scalar( v ) ) for v in _leaves( tree ) if v is not _BLANK ), default = 0 )
        return header + "\n" + _render_tree( tree, width, raw.ndim )


# an axis is unrolled (spans several array dimensions) iff it is an `AxisList` -- the fact lives in
# the type, so a tensor stores only the axis, never a separate unroll flag (see `Tensor.__init__`).
def _is_unrolled( axis ):
    return isinstance( axis, AxisList )


# containers recursed into by `_shape_tree` (a whitelist: anything else is a leaf)
_containers = ( list, tuple )


def _axis_rank( axis ):
    """Number of distinct axes the extent varies along (0 = dense, >0 = ragged).

    The coeffs' ShapeVars may share `dep_axes`, so this is the size of their
    union, not a sum or a max.
    """
    dep_axes = set()
    for shape_var in axis.coeffs:
        dep_axes.update( shape_var.dep_axes )
    return len( dep_axes )


def _shape_tree( value ):
    """Recursive size descriptor of `value`, WITHOUT touching its data.

    A whitelisted container becomes a list of child descriptors; anything else is
    treated as an array and described by its `.shape` (metadata only -- a GPU
    tensor is never moved); a scalar has shape `()`.
    """
    if isinstance( value, _containers ):
        return [ _shape_tree( v ) for v in value ]
    shape = getattr( value, "shape", None )
    return tuple( shape ) if shape is not None else ()


def _query( tree, d ):
    """Size structure along axis `d`, read from a shape tree: a scalar when the
    outer axes are dense, a nested list of ints when they are ragged."""
    if isinstance( tree, list ):
        return len( tree ) if d == 0 else [ _query( child, d - 1 ) for child in tree ]
    return tree[ d ]


def _depth( x ):
    return 1 + _depth( x[ 0 ] ) if isinstance( x, list ) and x else 0


def _collapse( sizes, rank ):
    """Reduce `sizes` to `rank` by dropping outer dims that are uniform (dense);
    a non-uniform dim declared dense is an error."""
    while _depth( sizes ) > rank:
        assert all( s == sizes[ 0 ] for s in sizes ), "non-uniform extent on a dense axis"
        sizes = sizes[ 0 ]
    return sizes


def _leaves( tree ):
    if isinstance( tree, list ):
        for v in tree:
            yield from _leaves( v )
    else:
        yield tree


def _fmt_scalar( v ):
    if v is _BLANK:
        return ""
    return f"{ v:g}" if isinstance( v, float ) else str( v )


def _render_tree( tree, width, rank ):
    """Text form of a (possibly ragged) nested list of numbers: no brackets, each
    number right-justified to `width`, one row per line, a blank line between
    higher-rank blocks. A row/block that is ENTIRELY padding (e.g. the unwritten
    tail of a reservation) is dropped rather than printed as blank -- a row that
    is only partly padding (ragged in some other direction) still prints, with
    `_BLANK` cells shown empty in place, so column alignment is preserved."""
    if rank == 0:
        return _fmt_scalar( tree ).rjust( width )
    if rank == 1:
        return " ".join( _fmt_scalar( v ).rjust( width ) for v in tree )
    sep = "\n" if rank == 2 else "\n\n"
    kept = [ sub for sub in tree if not _is_blank( sub ) ]
    return sep.join( _render_tree( sub, width, rank - 1 ) for sub in kept )


# display sentinel for a raw cell that is padding, not a real value (see `_display_tree`)
_BLANK = object()


def _is_blank( tree ):
    return all( v is _BLANK for v in _leaves( tree ) )


def _shape_var_at( shape_var, axes, idx ):
    """Current value of `shape_var` (its LIVE `.value`, e.g. solved from a kernel
    write -- not a reservation) at raw position `idx`: dense (no `dep_axes`) is a
    single value (same convention as `Axis.max`); ragged is indexed by where its
    `dep_axes` sit among `axes` (a dependency this tensor does not itself carry
    as a dimension falls back to the max over it)."""
    v = shape_var.raw
    if not shape_var.dep_axes:
        return int( v.max() )
    key = tuple( idx[ axes.index( dep ) ] if dep in axes else slice( None ) for dep in shape_var.dep_axes )
    v = v[ key ]
    return int( v.max() ) if v.ndim else int( v )


def _cell_valid( axes, idx ):
    """Whether raw position `idx` holds a real value: every axis's OWN extent,
    evaluated at `idx` from its ShapeVars' current values, must cover it. Checked
    independently per axis, so ragged padding is caught in any direction, not
    only a trailing/horizontal one."""
    for d, axis in enumerate( axes ):
        extent = axis.offset + sum(
            coeff * _shape_var_at( shape_var, axes, idx ) for shape_var, coeff in axis.coeffs.items()
        )
        if idx[ d ] >= extent:
            return False
    return True


def _display_tree( raw, axes, d = 0, idx = () ):
    """Nested list over the full (padded) `raw`, `_BLANK` at every position that
    is padding rather than a real value (see `_cell_valid`)."""
    if d == raw.ndim:
        return raw[ idx ].item() if _cell_valid( axes, idx ) else _BLANK
    return [ _display_tree( raw, axes, d + 1, idx + ( i, ) ) for i in range( raw.shape[ d ] ) ]


def _assemble( value, caps, dtype, device ):
    """Build a padded rank-`len(caps)` buffer from `value`, FUNCTIONALLY: pad each
    block up to `caps` (extension), then `stack` the blocks (assembly). No in-place
    mutation, so it stays valid for Jax tracers / autodiff. Pad value is 0."""
    if not isinstance( value, _containers ):
        leaf = driver.array( value, dtype = dtype, device = device )
        pad_width = [ ( 0, caps[ i ] - leaf.shape[ i ] ) for i in range( len( caps ) ) ]
        return driver.pad( leaf, pad_width ) if any( a for _, a in pad_width ) else leaf

    children = [ _assemble( v, caps[ 1: ], dtype, device ) for v in value ]
    if len( children ) < caps[ 0 ]:
        block = driver.zeros( caps[ 1: ], dtype = dtype )
        children = children + [ block ] * ( caps[ 0 ] - len( children ) )
    return driver.stack( children, axis = 0 )
