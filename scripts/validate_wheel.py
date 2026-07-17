#!/usr/bin/env python3
"""Valide le wheel `sdot` de bout en bout : build -> venv PROPRE -> install -> smoke test.

  python scripts/validate_wheel.py                 # build le wheel puis le teste
  python scripts/validate_wheel.py --wheel a.whl   # teste un wheel existant
  python scripts/validate_wheel.py --keep          # garde le venv/cache pour autopsie
  python scripts/validate_wheel.py --require-cold   # exige AUSSI un build acpp à froid (CI/machine vierge)

Le but est de prouver que le wheel est self-contained : on l'installe dans un venv NEUF (pas de
PYTHONPATH vers le repo, pas de `build/` local réutilisé) et on exécute make_hypercube(2D) +
measure -- tout le cycle génération -> compilation (acpp) -> enregistrement (Jax FFI) -> exécution.

La preuve clé est que la compilation acpp utilise les en-têtes C++ EMBARQUÉES DANS LE WHEEL
(`.../site-packages/sdot/_cpp`), pas le `src/cpp` du checkout : on vérifie que la ligne `[acpp] $`
référence bien `sdot/_cpp`. Le toolchain acpp lui-même est réutilisé depuis le cache utilisateur
s'il existe (comportement normal) ; `--require-cold` exige en plus un build acpp depuis zéro.

Contrairement à `run_tests.py`, ce script n'insère JAMAIS `src/python` dans sys.path : ce serait
justement l'erreur qui masquerait un wheel cassé en important `sdot` depuis le checkout.
"""
from pathlib import Path
import subprocess
import argparse
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
import sdot, sdot.compilation as c
print( sdot.__file__ )
print( c.cpp_include_root() )
"""


def _venv_python( venv_dir: Path ) -> Path:
    sub = "Scripts" if os.name == "nt" else "bin"
    return venv_dir / sub / ( "python.exe" if os.name == "nt" else "python" )


def _run( cmd, **kw ):
    print( "$ " + " ".join( map( str, cmd ) ), flush = True )
    return subprocess.run( [ str( c ) for c in cmd ], **kw )


def build_wheel() -> Path:
    _run( [ sys.executable, "-m", "build", "--wheel", "--outdir", ROOT / "dist" ], check = True )
    wheels = sorted( glob.glob( str( ROOT / "dist" / "sdot-*.whl" ) ), key = os.path.getmtime )
    if not wheels:
        raise RuntimeError( "aucun wheel produit sous dist/" )
    return Path( wheels[ -1 ] )


def validate( wheel: Path, keep: bool, require_cold: bool ) -> int:
    scratch = Path( tempfile.mkdtemp( prefix = "sdot-validate-" ) )
    venv_dir  = scratch / "venv"
    cache_dir = scratch / "sdot-cache"   # vide -> force un build acpp à froid
    work_dir  = scratch / "run"          # cwd du smoke test, jamais la racine du repo
    work_dir.mkdir( parents = True )
    print( f"scratch: { scratch }", flush = True )

    try:
        _run( [ sys.executable, "-m", "venv", venv_dir ], check = True )
        py = _venv_python( venv_dir )
        _run( [ py, "-m", "pip", "install", "--quiet", f"{ wheel }[jax]" ], check = True )

        # Pré-check rapide : échouer vite si l'install est mal packagée, AVANT le long build acpp.
        r = _run( [ py, "-c", PRECHECK ], cwd = work_dir, capture_output = True, text = True )
        if r.returncode:
            print( r.stdout + r.stderr, flush = True )
            raise RuntimeError( "pré-check import a échoué" )
        # .resolve() des deux côtés : sur macOS /var est un symlink vers /private/var, et
        # __file__ n'est pas canonicalisé alors que cpp_include_root() l'est.
        venv_real = str( venv_dir.resolve() )
        for line in r.stdout.split():
            if line.startswith( "/" ) and not str( Path( line ).resolve() ).startswith( venv_real ):
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

        # Preuve que la compilation a bien utilisé les en-têtes embarquées dans le wheel, et non
        # le src/cpp du checkout : la ligne acpp doit référencer `sdot/_cpp` sous le venv.
        acpp_lines = [ l for l in out.splitlines() if l.startswith( "[acpp] $" ) ]
        if not acpp_lines:
            raise RuntimeError( "aucune compilation acpp observée -- rien n'a été généré/compilé" )
        if not any( "sdot/_cpp" in l for l in acpp_lines ):
            raise RuntimeError(
                "acpp n'a pas utilisé les en-têtes du wheel (sdot/_cpp) :\n" + "\n".join( acpp_lines )
            )

        # Build acpp à froid, seulement si explicitement demandé (sur une machine de dev le
        # toolchain est déjà dans ~/Library/Caches/sdot et sera réutilisé -- c'est normal).
        if require_cold and "[acpp] downloading" not in out:
            raise RuntimeError(
                "--require-cold : aucun '[acpp] downloading' -- toolchain acpp réutilisé du cache."
            )

        print( "\nVALIDATION OK", flush = True )
        return 0
    finally:
        if keep:
            print( f"\n--keep : scratch conservé -> { scratch }", flush = True )
        else:
            shutil.rmtree( scratch, ignore_errors = True )


def main() -> int:
    p = argparse.ArgumentParser( description = "valide le wheel sdot dans un venv propre" )
    p.add_argument( "--wheel", type = Path, help = "wheel existant à tester (sinon on le build)" )
    p.add_argument( "--keep", action = "store_true", help = "garder le venv/cache scratch" )
    p.add_argument( "--require-cold", action = "store_true",
                    help = "exiger en plus un build acpp depuis zéro (CI/machine vierge)" )
    args = p.parse_args()

    wheel = args.wheel if args.wheel else build_wheel()
    return validate( wheel.resolve(), args.keep, args.require_cold )


if __name__ == "__main__":
    sys.exit( main() )
