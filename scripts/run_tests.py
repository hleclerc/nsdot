#!/usr/bin/env python3
"""Compile/lance les tests C++ ET Python avec un seul affichage et un seul jeu de filtres.

  micromamba run -n vfs python scripts/run_tests.py            # tout
  micromamba run -n vfs python scripts/run_tests.py Cell       # noms contenant "Cell"
  micromamba run -n vfs python scripts/run_tests.py '[fast]'   # filtre par tag

Les filtres sont identiques au harnais C++ (tests/cpp/test_main.h) : un argument
contenant '[' filtre par tag, le reste filtre par (sous-chaîne de) nom.

Découverte des tests Python : tous les `test_*.py` sous `tests/python/` (tests du
cœur) ET sous `applications/*/tests/` (tests propres à chaque application). Le
harnais `test()` est le même pour tous -- il vit dans le package `sdot.testing`,
importé en absolu (`from sdot.testing import test`), donc indépendant de
l'emplacement du fichier.

La sortie est *flushée* immédiatement (stdout en line-buffering) afin de voir la
progression en direct, avant tout plantage.
"""
from pathlib import Path
import importlib
import importlib.util
import traceback
import sys

ROOT = Path( __file__ ).resolve().parents[ 1 ]
# sdot est normalement installé en editable (`pip install -e .`) ; on garde ces insertions pour
# que le runner marche aussi à cru. `applications/` sur le path : les tests d'app importent leur
# code via `from <app>.… import …` (ex. `from reconstruction.Sinogram import Sinogram`).
sys.path.insert( 0, str( ROOT / "src" / "python" ) )
sys.path.insert( 0, str( ROOT / "applications" ) )

# le dossier du script (scripts/) est déjà dans sys.path -> import direct
from run_cpp_tests import run_cpp_tests


def test_roots():
    """Répertoires où chercher des `test_*.py` : le cœur puis un dossier par application."""
    roots = [ ROOT / "tests" / "python" ]
    roots += sorted( ( ROOT / "applications" ).glob( "*/tests" ) )
    return [ r for r in roots if r.is_dir() ]


def parse_filters( args ):
    names, tags = [], []
    for a in args:
        ( tags if "[" in a else names ).append( a )
    return names, tags


def _module_name_for( path ):
    """Nom de module SYNTHÉTIQUE et unique pour un fichier de test importé par chemin.

    Dérivé du chemin relatif au repo -> pas de collision entre deux `test_x.py` de racines
    différentes. Ce nom devient le `__name__` du module (donc le `t.module` enregistré par le
    harnais), ce qui permet le `reload` ciblé en phase d'exécution.
    """
    rel = path.resolve().relative_to( ROOT ).with_suffix( "" )
    return "sdot_test__" + "__".join( rel.parts )


# chemin source par nom de module synthétique -> permet la ré-exécution ciblée en phase RUN
# (importlib.reload ne marche pas : il re-cherche le spec via les finders, introuvable par chemin)
_FILE_BY_MODULE = {}


def _import_test_file( path ):
    """Importe (ou ré-exécute) un fichier de test par chemin, sous un nom stable."""
    name = _module_name_for( path )
    _FILE_BY_MODULE[ name ] = path
    spec = importlib.util.spec_from_file_location( name, path )
    module = importlib.util.module_from_spec( spec )
    sys.modules[ name ] = module
    spec.loader.exec_module( module )
    return module


def _collect_files( filter_names ):
    """Tous les `test_*.py` des racines, filtrés par nom de fichier comme côté C++."""
    files = []
    for root in test_roots():
        files += sorted( root.glob( "test_*.py" ) )
    if filter_names:
        files = [ f for f in files
                  if any( n.split( "::" )[ 0 ].lower() in f.stem.lower() for n in filter_names ) ]
    return files


def run_python_tests( filters ):
    """Collecte puis exécute les tests Python correspondant à `filters`.

    Retourne la liste des échecs sous la forme [ ( "python", name, "run" ), ... ].
    """
    filter_names, filter_tags = parse_filters( filters )

    # le harnais partagé : `test()`, phases et filtrage vivent dans sdot.testing (editable install)
    tm = importlib.import_module( "sdot.testing" )

    # --- phase de collecte : on importe chaque fichier test_*.py (corps sautés) ---
    tm.test_phase = tm.PHASE_COLLECT
    tm.all_the_tests.clear()
    for f in _collect_files( filter_names ):
        _import_test_file( f )

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
            # le site source (fichier:ligne) distingue des tests homonymes, que les noms seuls
            # laisseraient indiscernables dans la sortie.
            where = f"{ Path( t.file ).name }:{ t.line }"
            try:
                _import_test_file( _FILE_BY_MODULE[ t.module ] )
                print( f"{ tm.GREEN }PASS:{ tm.RESET } { t.name } ({ where })", flush = True )
            except Exception as e:
                print( f"{ tm.RED }FAIL:{ tm.RESET } { t.name } ({ where }) - { e }", flush = True )
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
