from typing_extensions import overload
from numpy.typing import ArrayLike
from typing import TYPE_CHECKING
import numpy

from ..util.aggregate import get_attribute
from ..util.Attribute import Attribute

from ..drivers.driver import driver

from ..devices.Device import Device

from .Dtype import Dtype
from .AbstractAxis import AbstractAxis


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


class Tensor( Attribute ):
    """
    Tensor declaration: a thin wrapper around the backend tensor of the chosen
    library (Jax, Torch, ...).

    One `Tensor` is created per parent instance (see `get_attribute`) and holds
    its own state: `c.frame = ...` goes through `set` and fills `_raw` (a
    homogeneous driver tensor); `c.frame` reads that value back.

    The logical contract is the axis list; axis extents may depend on other axes
    (RAGGED axes), in which case the varying sizes live in the `ShapeVar`s of
    rank > 0. The physical contract (padding / order / alignment, per device) is
    to come, as template kwargs, kept separate from the axis list.
    """

    if TYPE_CHECKING:
        def __set__( self, obj, value: ArrayLike | None ) -> None: ...


    def __init__( self, parent_inst = None, /, template_args = [], template_kwargs = {} ) -> None:
        self.device = Device.factory( template_kwargs.get( "device", None ) )
        self.dtype = Dtype.factory( template_kwargs.get( "dtype", None ) )
        self._raw = None          # homogeneous value buffer (padded when ragged)
        self._sizes = None        # one size per array dimension of the value
        self._spec_dims = None    # spec index -> its first array dimension
        self._unroll_spans = {}   # spec index -> (start, count) for the unrolled AxisList

        # A declared member is either a plain name or `name...` (unroll marker,
        # only valid for an AxisList): (axis, unroll) pairs kept in `specs`.
        # At most one member may be unrolled; plain `Axis`es can sit before and/or
        # after it (they keep one array dimension each; the unroll takes the rest).
        self.specs = []
        for entry in template_args:
            unroll = isinstance( entry, str ) and entry.endswith( "..." )
            axis = get_attribute( entry[ :-3 ] if unroll else entry, parent_inst )
            assert isinstance( axis, AbstractAxis )
            self.specs.append( ( axis, unroll ) )
        assert sum( u for _, u in self.specs ) <= 1, "at most one unrolled AxisList per tensor"

        # let each member record, on its ShapeVars, how this tensor constrains them
        for index, ( axis, unroll ) in enumerate( self.specs ):
            axis.register_in( self, index, unroll )

    @property
    def axes( self ):
        return [ axis for axis, _ in self.specs ]

    def set( self, value ):
        if isinstance( value, Tensor ):
            self._raw = value._raw
            self._sizes = value._sizes
            self._unroll_spans = value._unroll_spans
            return

        if any( unroll for _, unroll in self.specs ):
            return self._set_unrolled( value )

        # Sizes are read from a *shape tree* (list nesting + `.shape` metadata),
        # so `value`'s data is never touched -- a GPU tensor is not moved. One
        # size-tensor per axis; its rank is fixed by the declaration (a dense
        # axis collapses to a scalar, a ragged one keeps the per-segment lengths).
        tree = _shape_tree( value )
        ranks = [ _axis_rank( axis ) for axis, _ in self.specs ]
        self._sizes = [
            numpy.array( _collapse( _query( tree, d ), ranks[ d ] ), dtype = int )
            for d in range( len( self.specs ) )
        ]
        # no unroll here: one array dimension per declared axis
        self._spec_dims = list( range( len( self.specs ) ) )

        # values are stored in a tensor of rank = nb of axes (no compression).
        # Dense maps directly; ragged is assembled into a padded buffer whose per
        # axis capacity is (for now) the max observed size.
        if all( r == 0 for r in ranks ):
            self._raw = driver.array( value, dtype = self.dtype, device = self.device )
        else:
            caps = [ int( size.max() ) for size in self._sizes ]
            self._raw = _assemble( value, caps, self.dtype, self.device )

    def _set_unrolled( self, value ):
        """Assign a tensor with one unrolled AxisList: its rank is dynamic. Each
        plain axis before/after keeps a single array dimension; the AxisList spans
        the remaining ones, so the value is a dense array (a ragged non-unrolled
        axis would need nested lists, which this case deliberately excludes).

        `nb_dims` is solved from the number of spanned dimensions; each spanned
        size solves one loop index of the AxisList's ShapeVars; each plain axis is
        solved from its own (scalar) dimension."""
        u = next( i for i, ( _, unroll ) in enumerate( self.specs ) if unroll )
        tree = _shape_tree( value )
        assert isinstance( tree, tuple ), "an unrolled tensor expects a dense array value"

        # the unroll spans every array dimension not taken by a plain axis
        count = len( tree ) - ( len( self.specs ) - 1 )
        assert count >= 0, "value rank too small for the declared axes"

        self._sizes = [ numpy.array( s, dtype = int ) for s in tree ]
        self._spec_dims = []
        d = 0
        for i in range( len( self.specs ) ):
            self._spec_dims.append( d )
            d += count if i == u else 1
        self._unroll_spans = { u: ( self._spec_dims[ u ], count ) }
        self._raw = driver.array( value, dtype = self.dtype, device = self.device )

    @property
    def capacity( self ):
        if self._raw is None:
            return [ 0 ] * self.rank
        return self._raw.shape

    @property
    def shape( self ):
        # each member contributes a LIST of extents (one for an `Axis`, `nb_dims`
        # for an unrolled `AxisList`); concatenation gives the tensor's extents.
        res = []
        for axis, _ in self.specs:
            res += axis.max_list()
        return res

    @property
    def rank( self ):
        return len( self.axes )

    @property
    def raw( self ):
        return self._raw

    # def __repr__( self ):
    #     return f"{ self.name or 'tensor' }( { ', '.join( a.name or 'axis' for a in self.axes ) } )"
