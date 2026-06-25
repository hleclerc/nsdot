#!/usr/bin/env python3
"""Compile/lance les tests C++ ET Python avec un seul affichage et un seul jeu de filtres.

  micromamba run -n vfs python scripts/run_tests.py            # tout
  micromamba run -n vfs python scripts/run_tests.py Cell       # noms contenant "Cell"
  micromamba run -n vfs python scripts/run_tests.py '[fast]'   # filtre par tag

Les filtres sont identiques au harnais C++ (tests/cpp/test_main.h) : un argument
contenant '[' filtre par tag, le reste filtre par (sous-chaîne de) nom.

La sortie est *flushée* immédiatement (stdout en line-buffering) afin de voir la
progression en direct, avant tout plantage.
"""
from pathlib import Path
import importlib
import importlib.util
import traceback
import sys

ROOT = Path( __file__ ).resolve().parents[ 1 ]
sys.path.insert( 0, str( ROOT / "src" / "python" ) )

# le dossier du script (scripts/) est déjà dans sys.path -> import direct
from run_cpp_tests import run_cpp_tests

PY_DIR   = ROOT / "tests" / "python"
PKG_NAME = "nsdot_tests_python"


def parse_filters( args ):
    names, tags = [], []
    for a in args:
        ( tags if "[" in a else names ).append( a )
    return names, tags


def _load_python_package():
    """Charge tests/python comme un vrai package pour que `from .test_main` marche."""
    if PKG_NAME in sys.modules:
        return sys.modules[ PKG_NAME ]
    spec = importlib.util.spec_from_file_location(
        PKG_NAME, PY_DIR / "__init__.py",
        submodule_search_locations = [ str( PY_DIR ) ],
    )
    pkg = importlib.util.module_from_spec( spec )
    sys.modules[ PKG_NAME ] = pkg
    spec.loader.exec_module( pkg )
    return pkg


def run_python_tests( filters ):
    """Collecte puis exécute les tests Python correspondant à `filters`.

    Retourne la liste des échecs sous la forme [ ( "python", name, "run" ), ... ].
    """
    filter_names, filter_tags = parse_filters( filters )

    _load_python_package()
    tm = importlib.import_module( f"{ PKG_NAME }.main" )

    # --- phase de collecte : on importe chaque fichier test_*.py (corps sautés) ---
    tm.test_phase = tm.PHASE_COLLECT
    tm.all_the_tests.clear()
    files = sorted( PY_DIR.glob( "test_*.py" ) )

    if filter_names:
        # même restriction par nom de fichier que côté C++
        files = [ f for f in files if any( n.split( "::" )[ 0 ].lower() in f.stem.lower() for n in filter_names ) ]
    for f in files:
        importlib.import_module( f"{ PKG_NAME }.{ f.stem }" )

    selected = [ t for t in tm.all_the_tests if tm.matches( t, filter_names, filter_tags ) ]

    failures = []
    if not selected:
        return failures

    # --- phase d'exécution : on recharge le module ciblé, un test à la fois ---
    print( f"\n=========== [python] { len( selected ) } test(s) ==========", flush = True )
    tm.test_phase = tm.PHASE_RUN
    try:
        for t in selected:
            tm.test_filter = t
            module = sys.modules[ t.module ]
            try:
                importlib.reload( module )
                print( f"{ tm.GREEN }PASS:{ tm.RESET } { t.name }", flush = True )
            except Exception as e:
                print( f"{ tm.RED }FAIL:{ tm.RESET } { t.name } - { e }", flush = True )
                traceback.print_exc()
                sys.stdout.flush()
                failures.append( ( "python", t.name, "run" ) )
    finally:
        tm.test_phase  = tm.PHASE_COLLECT
        tm.test_filter = None

    return failures


def main():
    # progression immédiate (avant plantage) : stdout/stderr en line-buffering
    sys.stdout.reconfigure( line_buffering = True )
    sys.stderr.reconfigure( line_buffering = True )

    filters = sys.argv[ 1: ]

    failures = []
    failures += run_cpp_tests( filters )
    failures += run_python_tests( filters )

    print( "\n=============== summary =================" )
    if failures:
        for label, name, phase in failures:
            print( f"  [{ label }] { name }: { phase } FAILED" )
        return 1
    print( "  all good" )
    return 0


if __name__ == "__main__":
    sys.exit( main() )
