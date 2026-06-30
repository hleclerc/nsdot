from .Device import Device
import ctypes
import os

_libcuda = None


class CudaGpu( Device ):
    def __init__( self, device_id, mem_fraction = 0.5 ):
        self.device_id = device_id
        self.mem_fraction = mem_fraction  # fraction of total_dev_mem reserved for per-thread scratch
        self._attrs = None  # (nb_sm, max_thr_per_sm, regs_per_sm, shm_per_sm, total_dev_mem, sm_major, sm_minor)

    @property
    def name( self ):
        return "Cuda"

    @property
    def signature( self ):
        attrs = self._get_attrs()
        if attrs is None:
            return "cuda"
        *_, sm_major, sm_minor = attrs
        return f"cuda_sm{ sm_major }{ sm_minor }"

    @property
    def codegen_target( self ):
        return "cuda"

    @property
    def cpp_type( self ):
        return "ExecutionContext_Cuda"

    @property
    def mem_type( self ):
        return "MemorySpace_GlobalCudaRam"

    @property
    def is_cuda_gpu( self ):
        return True

    def driver_version_for_jax( self, devices ):
        return devices( "gpu" )[ self.device_id ]

    def __repr__( self ) -> str:
        return f"CudaGpu:{ self.device_id }"

    def _get_attrs( self ):
        if self._attrs is not None:
            return self._attrs
        lib = _load_libcuda()
        if lib is None:
            return None

        def attr( a ):
            v = ctypes.c_int( 0 )
            lib.cuDeviceGetAttribute( ctypes.byref( v ), a, self.device_id )
            return v.value

        mem = ctypes.c_size_t( 0 )
        lib.cuDeviceTotalMem_v2( ctypes.byref( mem ), self.device_id )

        # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                 = 16
        # CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR       = 39
        # CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR     = 82
        # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR             = 75
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR             = 76
        self._attrs = attr( 16 ), attr( 39 ), attr( 82 ), attr( 81 ), mem.value, attr( 75 ), attr( 76 )
        return self._attrs

    def nb_threads( self, nb_regs_per_thread=0, nb_shared_bytes_per_thread=0,
                    nb_local_bytes_per_thread=0, nb_pinned_bytes_per_thread=0, nb_waves=4 ):
        attrs = self._get_attrs()
        if attrs is None:
            raise RuntimeError( "CUDA device attributes unavailable (libcuda not found)" )
        nb_sm, max_thr_per_sm, regs_per_sm, shm_per_sm, total_dev_mem, *_ = attrs

        # SM-level occupancy constraints (simultaneous threads per SM)
        thr_per_sm = max_thr_per_sm
        if nb_regs_per_thread > 0:
            thr_per_sm = min( thr_per_sm, regs_per_sm // nb_regs_per_thread )
        if nb_shared_bytes_per_thread > 0:
            thr_per_sm = min( thr_per_sm, shm_per_sm  // nb_shared_bytes_per_thread )

        n = nb_waves * nb_sm * max( 1, thr_per_sm )

        # global memory budgets (constrain total work units, not per-SM occupancy).
        # `total_dev_mem` is the raw physical memory; only a fraction is actually allocatable
        # because the input/output data buffers and the XLA/JAX runtime must also reside on the
        # card. Reserve `mem_fraction` of it for per-thread scratch so the sum of all argument
        # buffers stays under XLA's per-call base limit (~the whole card).
        if nb_local_bytes_per_thread > 0:
            usable_dev_mem = int( total_dev_mem * self.mem_fraction )
            n = min( n, usable_dev_mem // nb_local_bytes_per_thread  )
        if nb_pinned_bytes_per_thread > 0:
            n = min( n, _total_host_ram() // nb_pinned_bytes_per_thread )

        return max( 1, n )

def _load_libcuda():
    global _libcuda
    if _libcuda is None:
        for lib_name in ( "libcuda.so.1", "libcuda.so", "nvcuda.dll" ):
            try:
                lib = ctypes.cdll.LoadLibrary( lib_name )
                lib.cuInit( 0 )
                _libcuda = lib
                break
            except OSError:
                pass
        if _libcuda is None:
            _libcuda = False
    return _libcuda if _libcuda else None


def _total_host_ram():
    try:
        return os.sysconf( 'SC_PHYS_PAGES' ) * os.sysconf( 'SC_PAGE_SIZE' )
    except ( AttributeError, ValueError ):
        return 4 * ( 1 << 30 )
