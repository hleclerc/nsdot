import numpy as np
import torch

from .TorchFramework import TorchFramework
from .Device import Device
from .Dtype import Dtype

class TorchDriver:
    """

    """
    def __init__( self, framework: TorchFramework, device: Device | None, ftype: Dtype | None, itype: Dtype | None ):
        if device is None:
            device = TorchDriver.default_device_for( ftype )
        if itype is None:
            itype = TorchDriver.default_itype_for( device )
        if ftype is None:
            ftype = TorchDriver.default_ftype_for( device )

        #
        self.framework = framework
        self.device = device
        self.ftype  = ftype
        self.itype  = itype

        # fill device_type for ftype
        assert ftype.floating_point == True
        match ftype.size:
            case 32:
                ftype.driver_version = torch.float32
            case 64:
                # jax.config.update( "jax_enable_x64", True )
                ftype.driver_version = torch.float64
            case _:
                raise ValueError( f"unsupported ftype size: { ftype.size }" )

        # fill device_type for itype
        assert itype.floating_point == False
        match itype.size:
            case 32:
                itype.driver_version = torch.int32
            case 64:
                itype.driver_version = torch.int64
            case _:
                raise ValueError( f"unsupported itype size: { itype.size }" )


    @staticmethod
    def default_device_for( ftype ):
        # cuda
        if torch.cuda.is_available():
            from .CudaGpu import CudaGpu
            return CudaGpu( 0 )

        # Metal (MPS) only supports FP32
        if torch.backends.mps.is_available() and ftype == "FP32":
            from .AppleGpu import AppleGpu
            return AppleGpu()

        # cpu
        from .Cpu import Cpu
        return Cpu()

    @staticmethod
    def default_ftype_for( device: Device ):
        if device.is_apple_gpu:
            return Dtype.fp( 32 )
        return Dtype.fp( 64 )

    @staticmethod
    def default_itype_for( device: Device ):
        if device.is_apple_gpu:
            return Dtype.si( 32 )
        return Dtype.si( 64 )









    @property
    def array_type( self ):
        return torch.Tensor

    @property
    def int_type( self ):
        return torch.int64

    _INT_DTYPES = frozenset( [ torch.int8, torch.int16, torch.int32, torch.int64,
                                torch.uint8, torch.uint16, torch.uint32, torch.uint64 ] )

    def is_int_dtype( self, dtype ):
        return dtype in TorchDriver._INT_DTYPES

    def any_requires_grad( self, tensors ) -> bool:
        return any( t.requires_grad for t in tensors )

    def t3( self, tensor ):
        """ make a rank 3 tensor """
        return self.tn( tensor, 3 )

    def t2( self, tensor ):
        """ make a rank 2 tensor """
        return self.tn( tensor, 2 )

    def t1( self, tensor ):
        """ make a rank 1 tensor """
        return self.tn( tensor, 1 )

    def t0( self, tensor ):
        """ make a rank 0 tensor """
        return self.tn( tensor, 0 )

    def tn( self, tensor, ndim = None, name = None, dtype = None ):
        """ make a rank ndim tensor """
        if tensor is None:
            return tensor

        if dtype is None or dtype is float:
            dtype = self.dtype
        elif dtype is int:
            dtype = self.int_type
        else:
            raise NotImplementedError( f"for { dtype }" )

        res = torch.as_tensor( tensor, dtype = dtype, device = self.device )

        if ndim is not None and res.ndim != ndim:
            if name is not None:
                raise IndexError( f"expecting for field '{ name }' a { ndim }d tensor, but { res.ndim }d was provided." )
            raise IndexError( f"expecting a { ndim }d tensor, but { res.ndim }d was provided." )

        return res

    def zeros( self, shape, dtype = None ):
        return torch.zeros( shape, dtype = dtype or self.dtype, device = self.device )

    def ones( self, shape, dtype = None ):
        return torch.ones( shape, dtype = dtype or self.dtype, device = self.device )

    def linspace( self, a, b, n, dtype = None ):
        return torch.linspace( a, b, n, dtype = dtype or self.dtype, device = self.device )

    def empty( self, shape, dtype = None ):
        return torch.zeros( shape, dtype = dtype or self.dtype, device = self.device )

    def expand_dims( self, tensor, index ):
        return tensor.unsqueeze( index )

    def repeat( self, tensor, shape ):
        return tensor.repeat( shape )

    def stack( self, tensors, axis ):
        return torch.stack( tensors, dim=axis )

    def linalg_solve( self, A, b ):
        return torch.linalg.solve( A, b )

    def moveaxis( self, tensor, source, destination ):
        return torch.moveaxis( tensor, source, destination )

    def hstack( self, lst ):
        return torch.hstack( lst )

    def to_numpy( self, t ):
        return np.array( t )
        # if isinstance( t, list ):
        # return t.to_numpy()

    def to_nanobind_compatible_objects( self, obj ):
        if isinstance( obj, torch.Tensor ):
            if self.is_int_dtype( obj.dtype ):
                return [ ( obj, "MI" ) ]
            return [ ( obj, "MF" ) ]
        return None

    def forward( self, forward_func: callable, backward_func: callable, args: list, input_tensors: list, _output_args: list, output_tensors: list ):
        """Differentiable wrapper.
            forward_func ( *args ) -> None  (fills output_tensors in-place)
            backward_func( *args, *grad_inputs, *grad_outputs ) -> None  (fills grad_inputs in-place)
        """
        _driver = self
        _output = []

        class Func( torch.autograd.Function ):
            @staticmethod
            def forward( ctx, *_ ):
                _output.append( forward_func( *args ) )
                return tuple( output_tensors )

            @staticmethod
            def backward( ctx, *grad_outputs ):
                grad_inputs = [ _driver.zeros( t.shape ) for t in input_tensors ]
                backward_func( *args, *grad_inputs, *grad_outputs )
                return tuple( grad_inputs )

        Func.apply( *input_tensors )
        return _output[ 0 ]

        # tracked = Func.apply( *input_tensors )
        # if not output_tensors:
        #     return None
        # return tracked[ 0 ] if len( output_tensors ) == 1 else tracked


    def array_conversion( self, value ):
        """ Ensure that every array of value is of the right type """
        if isinstance( value, self.array_type ):
            return value
        if isinstance( value, np.ndarray ):
            return self.tn( value )
        if isinstance( value, list ):
            return [ self.array_conversion( v ) for v in value ]
        if isinstance( value, tuple ):
            return tuple( self.array_conversion( v ) for v in value )
        if hasattr( value, "array_conversion" ):
            return value.array_conversion()
        raise NotImplementedError


    # def plan( self, bindings, f: BatchOfDistributions, g: BatchOfDistributions ):
    #     class SDOTFunction( torch.autograd.Function ):
    #         @staticmethod
    #         def forward( ctx, dirac_xs, *args ):
    #             # get constants
    #             batch_size = dirac_xs.shape[ 0 ]
    #             nb_diracs = dirac_xs.shape[ 1 ]
    #             dim = dirac_xs.shape[ 2 ]

    #             # room for the outputs
    #             barycenters = self.empty( [ batch_size, nb_diracs, dim ] )
    #             potentials = self.empty( [ batch_size, nb_diracs ] )
    #             distances = self.empty( [ batch_size ] )
    #             cuts = self.empty( [ batch_size, nb_diracs, 2 ] )

    #             # arguments as expected by the binding
    #             binding_inputs = unflatten_args( f, g, [ dirac_xs ] + list( args ) )

    #             # call the C++ procedure
    #             bindings.forward( *binding_inputs, distances, barycenters, potentials, cuts )

    #             ctx.save_for_backward( dirac_xs, *args, distances, barycenters, potentials, cuts )

    #             return distances, barycenters, potentials, cuts

    #         @staticmethod
    #         def backward( ctx, grad_distance, grad_barycenters, grad_potentials, grad_cuts ):
    #             # get room for the output gradients
    #             flat_grad_outputs = [ self.empty( arg.shape ) for arg in ctx.saved_tensors[ :-4 ] ]

    #             # arguments as expected by the binding
    #             binding_grad_outputs = unflatten_args( f, g, flat_grad_outputs )
    #             binding_inputs = unflatten_args( f, g, ctx.saved_tensors[ :-4 ] )

    #             bindings.backward( *binding_inputs, *ctx.saved_tensors[ -4: ], grad_distance, grad_barycenters, grad_potentials, grad_cuts, *binding_grad_outputs )

    #             return tuple( flat_grad_outputs )

    #     input_tensors = flat_tensor_list( f ) + flat_tensor_list( g )
    #     outputs = SDOTFunction.apply( *input_tensors )
    #     assert isinstance( outputs, tuple )
    #     return BatchOfOtPlans( *outputs )

    def optimize_using_lbfgs( self, loss, params, max_iter=50, tol_grad=1e-7, on_iter=None ):
        """ small helper to optimize `loss` wrt `params` using L-BFGS.
            - `params`  : torch tensor or list of torch tensors
            - `on_iter` : optional callback( params, iter, grad_norm ) called each iteration
            Returns the optimized params (same type as input).
        """
        is_list = isinstance( params, ( list, tuple ) )
        p_list  = list( params ) if is_list else [ params ]

        for p in p_list:
            p.requires_grad_( True )

        lbfgs = torch.optim.LBFGS( p_list, history_size=15, max_iter=10 )

        def closure():
            lbfgs.zero_grad()
            objective = loss( p_list if is_list else p_list[ 0 ] )
            objective.backward()
            return objective

        for i in range( max_iter ):
            lbfgs.step( closure )

            with torch.no_grad():
                grad_norm = float( torch.norm( torch.stack( [ torch.norm( p.grad ) for p in p_list if p.grad is not None ] ) ) )

                if on_iter:
                    on_iter( p_list if is_list else p_list[ 0 ], i, grad_norm )

                if grad_norm < tol_grad:
                    break

        return p_list if is_list else p_list[ 0 ]

    def optimize_using_sgd( self, loss, params ):
        optimizer = torch.optim.SGD( [ params ] )
        for _ in range( 2000 ):
            l = loss( params )
            print( l )
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
