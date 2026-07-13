
class Device:
    """

    """

    driver_version: any

    @staticmethod
    def factory( value ) -> 'Device':
        if isinstance( value, Device ):
            return value.copy()

        if value is None:
            return Device.default()

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

    @staticmethod
    def default() -> 'Device':
        from .Cpu import Cpu
        return Cpu()

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

    # ── AdaptiveCpp / Jax-FFI mapping ─────────────────────────────────────────
    # Consumed by sdot.compilation.adaptive_cpp (make_executable / make_library) and by the
    # Jax FFI registration. Defaults describe a device NOT reachable through acpp (e.g. Apple
    # GPU / Metal): `acpp_targets is None` makes the acpp builders raise. Reachable devices
    # override these.
    @property
    def acpp_targets( self ):
        """`--acpp-targets` value (e.g. "omp", "cuda:sm_80"); None if not acpp-reachable."""
        return None

    @property
    def acpp_profile( self ):
        """AdaptiveCpp feature profile ("minimal" | "full")."""
        return "minimal"

    @property
    def acpp_backends( self ):
        """GPU backends to enable when building acpp (e.g. ("cuda",)); () for CPU-only."""
        return ()

    @property
    def ffi_platform( self ) -> str:
        """XLA/Jax platform tag used by jax.ffi.register_ffi_target ("cpu" | "cuda" | ...)."""
        raise NotImplementedError

    @property
    def device_is_present( self ) -> bool:
        """Whether this device is actually usable here (hardware present AND acpp-reachable).

        Default: reachable iff acpp can target it (covers Cpu -> True, Apple GPU / Metal ->
        False). Devices whose hardware may be absent (CUDA) refine this."""
        return self.acpp_targets is not None

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
