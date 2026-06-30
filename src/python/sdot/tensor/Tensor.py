from typing_extensions import overload
from numpy.typing import ArrayLike

from ..util.Attribute import Attribute

from ..devices.Device import Device
from .TensorInst import TensorInst
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

    def __init__( self, *axes : Axis, dtype = None, device = None, name = None ) -> None:
        self.device = Device.factory( device )
        self.dtype = Dtype.factory( dtype )
        self.axes = list( axes )
        self.name = name # if None, set by `Attribute.__set_name__`

        # structural graph: tell each ShapeVar decl that this tensor decl uses it
        for axis in self.axes:
            for shape_var in axis.extent.terms.keys():
                shape_var.register_tensor( self )

    # --- descriptor protocol ------------------------------------------------
    @overload
    def __get__( self, obj: None, objtype = None ) -> 'Tensor': ...
    @overload
    def __get__( self, obj: object, objtype = None ) -> TensorInst: ...

    def __get__( self, obj, objtype = None ):
        if obj is None:
            return self
        return obj._bindings[ self ]

    def __set__( self, obj, value: ArrayLike | None ) -> None:
        obj._bindings[ self ].reassign( value )

    def instantiate( self, env, injection = None ) -> TensorInst:
        inst = TensorInst( self )
        if injection is not None:
            inst.reassign( injection )
        return inst

    def __repr__( self ):
        return f"{ self.name or 'tensor' }( { ', '.join( a.name or 'axis' for a in self.axes ) } )"
