from jax._src.util import foreach
from typing_extensions import Sequence
from numpy.typing import ArrayLike

from ..util.Reassignable import Reassignable

from ..drivers.driver import driver
from ..devices.Device import Device
from .Dtype import Dtype
from .Axis import Axis

class Tensor( Reassignable ):
    """
    Wrapper fin autour des tenseurs de la librairie choisie (Jax, Torch, ...).

    Le ctor permet de définir les propriétés (noms d'axes, data type, device, padding, ...). On peut ensuite donner les valeur avec un `=`.

    Les extents peuvent dépendre d'autres axes,
    """

    def __init__( self, *axes : Axis, dtype = None, device = None, value = None, name = None ) -> None:
        # parameters
        self.device = Device.factory( device )
        self.dtype = Dtype.factory( dtype )
        self.axes = list( axes )
        self.name = name # set by `@aggregate` from the field name

        # ShapeVar.usage
        for axis in self.axes:
            for shape_var in axis.extent.terms.keys():
                shape_var.register_tensor( self )

        # raw tensor
        self.shape = None
        self.raw = None
        if value is not None:
            self.value = value

    def __del__( self ) -> None:
        # symétrique des `register_tensor` de `__init__` (on suppose que `axes` n'a pas changé)
        for axis in self.axes:
            for shape_var in axis.extent.terms.keys():
                shape_var.unregister_tensor( self )

    # def __set__( self, obj, value: ArrayLike ) -> None:
    #     info( value )
    #     if isinstance( value, Tensor ):
    #         value = value.raw # TODO: shape, shape vars, ...

    #     if value is None:
    #         self.raw = None
    #         return

    #     self.raw = driver.array( value, dtype = self.dtype, device = self.device )
    #     self.shape = [ int( i ) for i in self.raw.shape ]

    # def __repr__( self ) -> str:
    #     return "pouet"

    def reassign( self, value ):
        if isinstance( value, Tensor ):
            value = value.raw # TODO: shape, shape vars, ...

        if value is None:
            self.raw = None
            return

        self.raw = driver.array( value, dtype = self.dtype, device = self.device )
        self.shape = [ int( i ) for i in self.raw.shape ]


    @property
    def value( self ):
        return self.raw

    @value.setter
    def value( self, value: ArrayLike | None ) -> None:
        self.reassign( value )

    def _copy( self, copy_map ):
        return Tensor( *[ a.copy( copy_map ) for a in self.axes ], dtype = self.dtype, device = self.device, value = self.raw, name = self.name )

    def shape_var_value( self, shape_var, forbidden_shape_vars ):
        if shape_var in forbidden_shape_vars:
            return None

        if self.shape is not None:
            for n, axis in enumerate( self.axes ):
                res = axis.extent.solve_shape_var( shape_var, self.shape[ n ] )
                if res is not None:
                    return res

        return None
