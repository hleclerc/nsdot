from .Device import Device
import os


class Cpu( Device ):
    @property
    def name( self ):
        return "Cpu"

    @property
    def cpp_type( self ):
        return "ExecutionContext_Cpu"

    @property
    def mem_type( self ):
        return "MemorySpace_CpuRam"

    @property
    def signature( self ):
        return "cpu"

    @property
    def codegen_target( self ):
        return "cpu"

    @property
    def acpp_targets( self ):
        # OpenMP CPU backend; resolves to omp.library-only under the minimal profile.
        return "omp"

    @property
    def is_cpu( self ):
        return True

    def __repr__( self ) -> str:
        return "Cpu"

    def nb_threads( self, nb_local_bytes_per_thread=0, nb_pinned_bytes_per_thread=0, nb_waves=1 ):
        # registers managed by compiler; shared memory not applicable to CPU threads
        # both local and pinned bytes draw from host RAM
        n          = os.cpu_count() or 1
        per_thread = max( nb_local_bytes_per_thread, nb_pinned_bytes_per_thread )
        if per_thread > 0:
            n = min( n, _total_host_ram() // per_thread )
        return max( 1, n )

    def driver_version_for_jax( self, devices ):
        return devices( "cpu" )[ 0 ]


def _total_host_ram():
    try:
        return os.sysconf( 'SC_PHYS_PAGES' ) * os.sysconf( 'SC_PAGE_SIZE' )
    except ( AttributeError, ValueError ):
        return 4 * ( 1 << 30 )  # 4 GB conservative fallback
