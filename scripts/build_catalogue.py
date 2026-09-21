"""Bâtir le CATALOGUE des noyaux précompilés (voir `loom/compilation/catalogue.py`).

    # 1. sur une machine de dev (un GPU pour le relevé CUDA) : le relevé des sources
    python scripts/build_catalogue.py record  --out catalogue_record [--device cpu|cuda] [--tests]

    # 2. n'importe où (l'intégration continue, par plateforme) : une variante -> une bibliothèque
    python scripts/build_catalogue.py compile --record catalogue_record --out sdot/catalogue --variant x86-64-v3
    python scripts/build_catalogue.py compile --record catalogue_record --out sdot/catalogue --device cuda --arch sm_70,sm_75,sm_80,sm_86,sm_89,sm_90

    # 3. le wheel `sdot` embarque `sdot/catalogue/` (-> `sdot/_catalogue`), enregistré à l'import

`record` nettoie un répertoire de build à lui (`SDOT_BUILD_DIR`), lance ce qui provoque les
compilations (`python -m sdot.catalogue`, et les suites de tests sdot avec `--tests`) sous
`SDOT_CATALOGUE_RECORD`, et laisse dans `--out` les sources générés + les en-têtes générés.
`compile` compile ce relevé avec la variante demandée, dans un build à lui aussi, et dépose
`libsdot_kernels.<so>` + `catalogue.json` sous `--out/<tag>/`.
"""
from pathlib import Path
import subprocess
import argparse
import tempfile
import shutil
import sys
import os

ROOT = Path( __file__ ).resolve().parents[ 1 ]
SRC  = [ str( ROOT / p / "src" ) for p in ( "loom", "sdot", "otrec" ) ]


def _env( **extra ):
    env = dict( os.environ )
    env[ "PYTHONPATH" ] = os.pathsep.join( SRC + [ p for p in env.get( "PYTHONPATH", "" ).split( os.pathsep ) if p ] )
    env.update( { k: str( v ) for k, v in extra.items() } )
    return env


def record( args ):
    # le relevé d'un genre de device remplace le précédent du même genre ; les autres genres et
    # les en-têtes générés (fusionnés, write-if-changed) restent
    out = Path( args.out ).resolve()
    if ( out / args.device ).exists():
        shutil.rmtree( out / args.device )
    out.mkdir( parents = True, exist_ok = True )
    with tempfile.TemporaryDirectory( prefix = "sdot-catalogue-record-" ) as build:
        env = _env( SDOT_CATALOGUE_RECORD = out, SDOT_BUILD_DIR = build, SDOT_KERNELS = "atelier" )
        device = [ "--device", args.device ]
        cmds = [ [ sys.executable, "-m", "sdot.catalogue", *device ] ]
        if args.tests:
            cmds += [ [ str( ROOT / "run" ), "test", t, *device ]
                      for t in ( "test_Cell", "test_PowerDiagram", "test_OtPlan1d", "test_OtPlan", "test_SumOfDiracs", "test_Image" ) ]
        for cmd in cmds:
            print( "$", " ".join( cmd ), flush = True )
            if subprocess.run( cmd, env = env, cwd = ROOT ).returncode:
                raise SystemExit( "le relevé a échoué" )
    n = len( list( ( out / args.device ).glob( "*.c*" ) ) )
    print( f"relevé : { n } noyau(x) { args.device } dans { out }" )


def compile_( args ):
    with tempfile.TemporaryDirectory( prefix = "sdot-catalogue-build-" ) as build:
        # la variante / les architectures vont au compilateur par l'environnement, et le build a
        # son propre répertoire : rien de la machine ne doit entrer dans la bibliothèque
        os.environ[ "SDOT_BUILD_DIR" ] = build
        if args.device == "cpu":
            os.environ[ "SDOT_CPU_VARIANT" ] = args.variant
            tag = f"cpu-{ args.variant }"
        else:
            os.environ[ "SDOT_CUDA_ARCH" ] = args.arch
            tag = "cuda"
        sys.path[ :0 ] = SRC
        import sdot  # noqa: F401 -- enregistre sa racine C++
        from loom.devices.Device import Device
        from loom.compilation import catalogue
        device = Device.factory( args.device )
        n = catalogue.build( args.record, args.out, device, tag )
    print( f"catalogue `{ tag }` : { n } noyau(x) dans { Path( args.out ) / tag }" )


def main():
    p = argparse.ArgumentParser( description = __doc__.splitlines()[ 0 ] )
    sub = p.add_subparsers( dest = "cmd", required = True )
    r = sub.add_parser( "record" )
    r.add_argument( "--out", default = "catalogue_record" )
    r.add_argument( "--device", default = "cpu", choices = ( "cpu", "cuda" ) )
    r.add_argument( "--tests", action = "store_true", help = "ajouter les suites de tests sdot au relevé" )
    r.set_defaults( func = record )
    c = sub.add_parser( "compile" )
    c.add_argument( "--record", default = "catalogue_record" )
    c.add_argument( "--out", default = str( ROOT / "sdot" / "catalogue" ) )
    c.add_argument( "--device", default = "cpu", choices = ( "cpu", "cuda" ) )
    c.add_argument( "--variant", default = "x86-64-v3", help = "niveau CPU : x86-64-v2 / v3 / v4, armv8-a" )
    c.add_argument( "--arch", default = "sm_70,sm_75,sm_80,sm_86,sm_89,sm_90", help = "architectures CUDA, séparées par des virgules" )
    c.set_defaults( func = compile_ )
    a = p.parse_args()
    a.func( a )


if __name__ == "__main__":
    main()
