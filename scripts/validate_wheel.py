#!/usr/bin/env python3
"""Valide le wheel `sdot` de bout en bout : build -> venv PROPRE -> install -> smoke test.

  python scripts/validate_wheel.py                 # build le wheel puis le teste
  python scripts/validate_wheel.py --wheel a.whl   # teste un wheel existant
  python scripts/validate_wheel.py --keep          # garde le venv/cache pour autopsie

Le but est de prouver que le wheel est self-contained : on l'installe dans un venv NEUF (pas de
PYTHONPATH vers le repo, pas de `build/` local réutilisé) et on exécute make_hypercube(2D) +
measure -- tout le cycle génération -> compilation (compilateur hôte) -> enregistrement (Jax FFI) -> exécution.

La preuve clé est que la compilation utilise les en-têtes C++ EMBARQUÉES DANS LES WHEELS
(`.../site-packages/loom/_include`, `sdot/_include`), pas celles du checkout : `include_roots()`
ne doit nommer que des chemins du venv, et le noyau doit compiler avec.

Contrairement à `run_tests.py`, ce script n'insère JAMAIS `src/python` dans sys.path : ce serait
justement l'erreur qui masquerait un wheel cassé en important `sdot` depuis le checkout.
"""
from pathlib import Path
import subprocess
import argparse
import re
import tempfile
import shutil
import glob
import sys
import os

ROOT = Path( __file__ ).resolve().parents[ 1 ]

# Reproduit tests/python/test_Cell.py :: test( "basic" ), plus une assertion chiffrée (le bloc
# original ne fait qu'afficher, utile à l'oeil mais pas exploitable comme signal pass/fail).
SMOKE = """
import numpy as np
from sdot import Cell

c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 2, 0 ], [ 0, 1 ] ] )
print( c.vertex_positions )
print( c.measure )
assert abs( float( np.asarray( c.measure ) ) - 2.0 ) < 1e-9, "measure != 2.0"
print( "SMOKE-OK" )
"""

PRECHECK = """
import sdot, loom.compilation as c
print( sdot.__file__ )
print( c.include_roots() )
"""


def _venv_python( venv_dir: Path ) -> Path:
    sub = "Scripts" if os.name == "nt" else "bin"
    return venv_dir / sub / ( "python.exe" if os.name == "nt" else "python" )


def _run( cmd, **kw ):
    print( "$ " + " ".join( map( str, cmd ) ), flush = True )
    return subprocess.run( [ str( c ) for c in cmd ], **kw )


def build_wheels() -> list:
    """Les deux wheels, `loom` puis `sdot` (le second dépend du premier), sous `dist/`."""
    wheels = []
    for pkg in ( "loom", "sdot" ):
        _run( [ sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "-w", ROOT / "dist", ROOT / pkg ], check = True )
        found = sorted( glob.glob( str( ROOT / "dist" / f"{ pkg }-*.whl" ) ), key = os.path.getmtime )
        if not found:
            raise RuntimeError( f"aucun wheel { pkg } produit sous dist/" )
        wheels.append( Path( found[ -1 ] ) )
    return wheels


def validate( wheels: list, keep: bool ) -> int:
    scratch = Path( tempfile.mkdtemp( prefix = "sdot-validate-" ) )
    venv_dir  = scratch / "venv"
    cache_dir = scratch / "sdot-cache"   # vide -> aucun binaire réutilisé
    work_dir  = scratch / "run"          # cwd du smoke test, jamais la racine du repo
    work_dir.mkdir( parents = True )
    print( f"scratch: { scratch }", flush = True )

    try:
        _run( [ sys.executable, "-m", "venv", venv_dir ], check = True )
        py = _venv_python( venv_dir )
        # `ninja` et `jax` viennent de PyPI ; loom et sdot des wheels fraîchement bâties
        _run( [ py, "-m", "pip", "install", "--quiet", *( f"{ w }[jax]" if w.name.startswith( "loom" ) else str( w ) for w in wheels ) ], check = True )

        # Pré-check rapide : échouer vite si l'install est mal packagée, AVANT de compiler.
        r = _run( [ py, "-c", PRECHECK ], cwd = work_dir, capture_output = True, text = True )
        if r.returncode:
            print( r.stdout + r.stderr, flush = True )
            raise RuntimeError( "pré-check import a échoué" )
        # .resolve() des deux côtés : sur macOS /var est un symlink vers /private/var, et
        # __file__ n'est pas canonicalisé alors que include_roots() l'est.
        venv_real = str( venv_dir.resolve() )
        for line in re.findall( r"/[^\s'\]\[,]+", r.stdout ):
            if not str( Path( line ).resolve() ).startswith( venv_real ):
                raise RuntimeError(
                    f"chemin hors du venv (import depuis le checkout ?) : { line }\n{ r.stdout }"
                )
        print( r.stdout, flush = True )

        # Smoke test complet, dans un env isolé : pas de PYTHONPATH, cache neuf, build par défaut.
        env = dict( os.environ )
        env.pop( "PYTHONPATH", None )
        env.pop( "SDOT_BUILD_DIR", None )
        env[ "SDOT_CACHE_DIR" ] = str( cache_dir )

        r = _run( [ py, "-c", SMOKE ], cwd = work_dir, env = env, capture_output = True, text = True )
        out = r.stdout + r.stderr
        print( out, flush = True )
        if r.returncode or "SMOKE-OK" not in r.stdout:
            raise RuntimeError( "smoke test a échoué" )

        # La preuve que la compilation a utilisé les en-têtes du wheel est le pré-check ci-dessus :
        # `include_roots()` ne nomme que des chemins du venv, et le noyau a compilé avec.

        print( "\nVALIDATION OK", flush = True )
        return 0
    finally:
        if keep:
            print( f"\n--keep : scratch conservé -> { scratch }", flush = True )
        else:
            shutil.rmtree( scratch, ignore_errors = True )


def main() -> int:
    p = argparse.ArgumentParser( description = "valide le wheel sdot dans un venv propre" )
    p.add_argument( "--wheel", type = Path, nargs = "*", help = "wheels existants à tester (loom et sdot ; sinon on les bâtit)" )
    p.add_argument( "--keep", action = "store_true", help = "garder le venv/cache scratch" )
    args = p.parse_args()

    wheels = [ w.resolve() for w in args.wheel ] if args.wheel else build_wheels()
    return validate( wheels, args.keep )


if __name__ == "__main__":
    sys.exit( main() )
