from typing_extensions import overload
from numpy.typing import ArrayLike
from typing import TYPE_CHECKING

from ..util.aggregate import get_attribute
from ..util.Attribute import Attribute

from ..drivers.driver import driver

from ..devices.Device import Device

from .Dtype import Dtype
from .Axis import Axis


class Tensor( Attribute ):
    """
    Déclaration d'un tenseur : wrapper fin autour des tenseurs de la librairie
    choisie (Jax, Torch, ...).

    C'est le *schéma* (partagé, au niveau classe). Le ctor définit les propriétés
    (axes, dtype, device, ...). En tant que descripteur, `c.frame` renvoie la
    `TensorInst` de l'instance et `c.frame = ...` lui donne sa valeur. L'état
    mutable (le tableau `raw`, la shape résolue) vit dans la `TensorInst`, qui
    référence cette déclaration sans en copier les attributs.

    Les extents des axes peuvent dépendre d'autres axes (axes ragged).
    """

    if TYPE_CHECKING:
        def __set__( self, obj, value: ArrayLike | None ) -> None: ...


    def __init__( self, parent_inst = None, /, template_args = [], template_kwargs = {} ) -> None:
        self.device = Device.factory( template_kwargs.get( "device", None ) )
        self.dtype = Dtype.factory( template_kwargs.get( "dtype", None ) )
        self.raw = None

        self.axes = []
        for dep_axis in template_args:
            axis = get_attribute( dep_axis, parent_inst )
            assert isinstance( axis, Axis )
            self.axes.append( axis )

    def set( self, value ):
        if isinstance( value, Tensor ):
            value = value._raw
        self.raw = driver.array( value, dtype = self.dtype, device = self.device )

    # def __repr__( self ):
    #     return f"{ self.name or 'tensor' }( { ', '.join( a.name or 'axis' for a in self.axes ) } )"
