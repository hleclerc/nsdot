class Device:
    """

    """

    driver_version: any

    @staticmethod
    def factory( value ) -> 'Device':
        if isinstance( value, Device ):
            return value.copy()

        value = str( value ).lower()
        if value.startswith( "cpu" ):
            from .Cpu import Cpu
            return Cpu()

        if value.startswith( "gpu" ) or value.startswith( "cuda" ):
            from .CudaGpu import CudaGpu
            n = 0
            s = value.split( ":" )
            if len( s ) == 2:
                n = int( s[ 1 ] )
            return CudaGpu( n )

        if value.startswith( "metal" ) or value.startswith( "applegpu" ):
            from .AppleGpu import AppleGpu
            return AppleGpu()

        raise ValueError( f"unsupported device name: { value }" )

    @property
    def name( self ) -> str:
        raise NotImplementedError

    @property
    def cpp_type( self ) -> str:
        raise NotImplementedError

    @property
    def mem_type( self ) -> str:
        raise NotImplementedError

    @property
    def signature( self ) -> str:
        raise NotImplementedError

    @property
    def codegen_target( self ) -> str:
        """Codegen context tag for per-context FfiCode selectors ( see FfiCode.select_for )."""
        raise NotImplementedError

    def __eq__( self, value, / ) -> bool:
        if not isinstance( value, Device ):
            value = Device.factory( value )
        return str( self ) == str( value )

    def __neq__( self, value, / ) -> bool:
        return not self.__eq__( value )

    def nb_threads( self, nb_local_bytes_per_thread=0, nb_pinned_bytes_per_thread=0, nb_waves=1 ) -> int:
        raise NotImplementedError

    def driver_version_for_jax( self, devices ):
        raise NotImplementedError

    @property
    def is_apple_gpu( self ):
        return False

    @property
    def is_cuda_gpu( self ):
        return False

    @property
    def is_cpu( self ):
        return False
