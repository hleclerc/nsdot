#!/usr/bin/env python3
"""Build & run the C++ tests with AdaptiveCpp (the `acpp` SYCL driver).

  micromamba run -n vfs python scripts/run_cpp_tests.py            # all tests
  micromamba run -n vfs python scripts/run_cpp_tests.py Run Index  # only matching names

Every test is compiled through `acpp` for each candidate device (CPU today; CUDA/HIP/…
once the matching toolchains are available). A device is skipped when AdaptiveCpp has no
target for it (e.g. Apple GPU / Metal). Tests are only *run* when the device is actually
present, otherwise the run is skipped while the build is still validated. Exit code is
non-zero if any build or run failed.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path( __file__ ).resolve().parents[ 1 ]
sys.path.insert( 0, str( ROOT / "src" / "python" ) )

from sdot.compilation.adaptive_cpp import make_executable
from sdot.devices.Device import Device


def discover( filter_names ):
    files = sorted( ( ROOT / "tests" / "cpp" ).glob( "test_*.*" ) )
    # on ne restreint les fichiers que par les noms ; le filtrage par tag est
    # délégué au binaire (qui filtre au niveau de chaque test).
    if filter_names:
        files = [ f for f in files if any( k.lower() in f.stem.lower() for k in filter_names ) ]
    return files


def run_cpp_tests( filters ):
    """Compile et lance les tests C++ correspondant à `filters`.

    Retourne la liste des échecs sous la forme [ ( label, name, phase ), ... ].
    Les `filters` sont transmis tels quels au binaire de test (mêmes sémantiques
    de filtrage noms/tags que tests/cpp/test_main.h).
    """
    filter_names = [ a for a in filters if "[" not in a ]
    files = discover( filter_names )

    failures = []
    for f in files:
        for device in [ Device.factory( "cuda" ), Device.factory( "cpu" ) ]:
            # skip devices AdaptiveCpp cannot target (e.g. Apple GPU / Metal)
            if not device.device_is_present:
                continue

            # build
            print( f"\n=========== [{ device }] { f.stem } ==========", flush = True )
            try:
                exe = make_executable( f"{ f.stem }_{ device }", [ f ], device )
            except Exception as e:
                print( f"  BUILD-FAIL: { e }", flush = True )
                failures.append( ( device, f.stem, "build" ) )
                continue

            # run (on passe les filtres pour filtrer noms/tags au niveau test)
            if subprocess.run( [ str( exe ), *filters ] ).returncode:
                failures.append( ( device, f.stem, "run" ) )

    return failures


def main():
    failures = run_cpp_tests( sys.argv[ 1: ] )

    print( "\n=============== summary =================" )
    if failures:
        for label, name, phase in failures:
            print( f"  [{ label }] { name }: { phase } FAILED" )
        return 1
    print( "  all good" )
    return 0


if __name__ == "__main__":
    sys.exit( main() )
