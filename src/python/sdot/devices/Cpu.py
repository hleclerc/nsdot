from .Device import Device
import os


class Cpu( Device ):
    def copy( self ) -> 'Device':
        return Cpu()

    @property
    def name( self ):
        return "Cpu"

    @property
    def cpp_queue_type( self ):
        return "CpuQueue"

    @property
    def cpp_memory_space( self ):
        return "CpuHostMemorySpace"

    @property
    def signature( self ):
        return "cpu"

    @property
    def codegen_target( self ):
        return "cpu"

    @property
    def is_cpu( self ):
        return True

    @property
    def acpp_targets( self ):
        return "omp"

    @property
    def ffi_platform( self ):
        return "cpu"

    def __repr__( self ) -> str:
        return "Cpu"

    def _hw_thread_cap( self, nb_local_bytes_per_thread=0, nb_pinned_bytes_per_thread=0, nb_waves=1 ):
        # registers managed by compiler; shared memory not applicable to CPU threads
        # both local and pinned bytes draw from host RAM
        n          = os.cpu_count() or 1
        per_thread = max( nb_local_bytes_per_thread, nb_pinned_bytes_per_thread )
        if per_thread > 0:
            usable = int( _total_host_ram() * self.scratch_ram_fraction )
            n = min( n, usable // per_thread )
        return n

    def driver_version_for_jax( self, devices ):
        return devices( "cpu" )[ 0 ]


def _total_host_ram():
    try:
        return os.sysconf( 'SC_PHYS_PAGES' ) * os.sysconf( 'SC_PAGE_SIZE' )
    except ( AttributeError, ValueError ):
        return 4 * ( 1 << 30 )  # 4 GB conservative fallback
