
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

    # ── what the device IS, in C++ ────────────────────────────────────────────
    # A device is not a runtime parameter of a kernel: it is part of its TYPE. The queue decides
    # which memory space the kernel dereferences (see Ptr.h), and the memory space of a buffer
    # says where it already lives -- so both are typedefs in the generated source, and a call
    # whose data XLA put on the GPU needs no transfer at all.
    @property
    def cpp_queue_type( self ) -> str:
        """The `sdot::Queue` of a generated kernel ("CpuQueue" | "CudaQueue")."""
        raise NotImplementedError

    @property
    def cpp_memory_space( self ) -> str:
        """Where the buffers this device is given already live ("CpuHostMemorySpace" | ...)."""
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

    # fraction of total device/host RAM per-thread scratch is allowed to use. The rest holds the
    # I/O buffers, the runtime and the rest of the machine, none of which we measure here -- this
    # is a deliberately coarse cap, not a reading of what is actually free (which would be a moving
    # runtime quantity, wrong to freeze into a compiled shape; see `Cell.measure`). Overridable
    # per device (Cuda carries its own `mem_fraction`).
    scratch_ram_fraction = 0.5

    def nb_threads( self, batch_axes = (), **per_thread ) -> int:
        """How many threads to SIZE per-thread scratch for.

        The hardware/RAM ceiling (`_hw_thread_cap`, device-specific) capped by the total number of
        batch items -- no point reserving more lanes than there is work -- and floored at 1. The
        per-thread footprint is passed through in `**per_thread` (`nb_local_bytes_per_thread`, ...)
        to the device hook; `batch_axes` is the list whose sizes multiply into the item count.

        This is a HOST-side, compile-time decision: the result becomes the static extent of a
        scratch axis, so every input must be knowable at trace time (a prescribed batch size, a
        capacity bound). A genuinely dynamic quantity folded in here would freeze into the compiled
        kernel -- which is why `scratch_ram_fraction` is a fixed fraction, not a live free-RAM read."""
        n = self._hw_thread_cap( **per_thread )
        total = 1
        for axis in batch_axes:
            total *= int( axis.max )
        if batch_axes:
            n = min( n, total )
        return max( 1, n )

    def _hw_thread_cap( self, **per_thread ) -> int:
        """The hardware/RAM ceiling on simultaneous threads, given a per-thread footprint. Device
        specific (cores + host RAM on CPU, SM occupancy + device RAM on CUDA); `nb_threads` wraps
        it with the batch cap and the floor."""
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
