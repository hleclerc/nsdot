"""Ce que ce dépôt déclare à `errand` : où est le code, où va la sortie, et où ça tourne.

Trois paquets dans un seul checkout (`loom`, `sdot`, `otrec`) et leurs suites, plus les tests C++
de `loom` qui ne sont pas des appels Python et passent donc par un *provider*.

    errand                       tout ce qui doit passer
    errand -k bench              ce qui doit être rapide
    errand test_Cell             un fichier
    errand "test_Cell::cut*"     des cas dedans
    errand --env lmo-cuda-jax    ailleurs
    errand --tui                 l'écran

Le PYTHONPATH d'antan (`./run` qui plaçait les trois `src` devant tout le reste) est ici
`configure( src = ... )` : les trois projets DE CE CHECKOUT passent devant ce qu'un
`pip install -e` fait depuis un autre checkout aurait installé.
"""
import sys
from pathlib import Path

from errand import Apptainer, Micromamba, Outcome, Param, Provider, Ssh, configure, env, provider

configure(
    out = "runs",
    # Les trois projets de CE checkout, devant tout ce que l'environnement a installé.
    src = [ "loom/src", "sdot/src", "otrec/src" ],
    default = "nsdot",
    # Ce dans quoi la découverte n'entre pas : du code gardé pour mémoire, des artefacts, des
    # répertoires de sortie -- et `errand/`, qui est le checkout d'un AUTRE projet posé ici. Il
    # n'a pas de configuration à lui ( il n'en a pas besoin ), donc la règle « un répertoire qui
    # a sa propre config est un autre projet » ne peut pas le voir : on le dit.
    exclude = [ "errand", "old_pd", "old_sdot", "very_old_sdot", "build", "dist", "tmp", "runs",
                "containers", "catalogue_record", "notes", "docs", "__pycache__" ],
)


# ── où ça tourne ─────────────────────────────────────────────────────────────
#
# Le `driver` n'est plus une couche : c'est un TAG. Il ne change pas la façon d'atteindre la
# machine, il dit ce qu'on y trouve -- donc il sélectionne (`--driver jax`, `-t 'driver=torch'`)
# et il est lisible dans le nom du répertoire de sortie. Un fichier qui a besoin de savoir
# demande `has_tag( "driver=torch" )`.

# Ce que l'environnement CONTIENT, dit ici et nulle part ailleurs : `errand --setup --env
# nsdot` le fabrique s'il manque, `--setup force` le refait. Les quatre `-e` sont les paquets
# de CE checkout, `errand` en tête parce que c'est lui qui lance les autres.
EDITABLE = [ "-e", "errand", "-e", "loom", "-e", "sdot", "-e", "otrec" ]

NSDOT = Micromamba( "nsdot", python = "3.13", packages = [
    "pip",
    # de quoi bâtir CGAL depuis le git et compiler un programme contre lui
    "cmake", "ninja", "gmp", "mpfr", "boost-cpp", "eigen",
], pip = [ "jax[cuda13]", *EDITABLE ] )

VFS   = Micromamba( "vfs", python = "3.13", pip = [ "jax[cuda13]", *EDITABLE ] )
TORCH = Micromamba( "torch", pip = [ "torch", *EDITABLE ] )

LMO = Ssh( host = "lmo", root = "/home/leclerc/nsdot", python = "python3" )

CUDA_JAX_AOT = Apptainer(
    image  = "containers/cuda-jax-aot.sif",
    flags  = [ "--nvccli" ],
    mounts = { "loom": "/opt/sdot/loom", "sdot": "/opt/sdot/sdot", "otrec": "/opt/sdot/otrec" },
)

env( "nsdot", [ NSDOT ], driver = "jax", cuda = True )
env( "vfs",   [ VFS   ], driver = "jax" )
env( "torch", [ TORCH ], driver = "torch" )

env( "lmo-cuda-jax",     [ LMO, VFS ],          driver = "jax",   cuda = True )
env( "lmo-cuda-jax-aot", [ LMO, CUDA_JAX_AOT ], driver = "jax",   cuda = True )
env( "lmo-cuda-torch",   [ LMO, TORCH ],        driver = "torch", cuda = True )


# ── les tests C++ de loom ────────────────────────────────────────────────────

class LoomCpp( Provider ):
    """`loom/tests/cpp/test_*.cpp` : une entrée par fichier, un paramètre `--device`.

    Une entrée par FICHIER et non par `TEST_CASE`, parce que les entrées doivent être connues
    de CE côté-ci d'un saut ssh -- avant que quoi que ce soit soit compilé -- pour que les
    chemins à rapatrier soient calculables d'avance. Un `::nom` dans le motif est passé au
    binaire, qui sait filtrer lui-même ( `main.h` prend des noms et des `[tags]` ).

    Le device est un PARAMÈTRE et pas deux entrées : `--device cpu,cuda` est alors une matrice
    comme une autre, avec un répertoire par device et une ligne par device dans le résumé. Un
    device absent de la machine donne un SKIP -- pas un échec, et pas un silence.
    """

    name = "loom-cpp"
    whole_files = True

    def __init__( self, dir = "loom/tests/cpp", pattern = "test_*.cpp",
                  devices = ( "cpu", "cuda" ) ):
        self.dir     = Path( dir )
        self.pattern = pattern
        self.devices = list( devices )

    def files( self ):
        return sorted( p.resolve() for p in self.dir.glob( self.pattern ) )

    def collect( self, specs ):
        return [ self.entry( name = path.stem, file = path, key = str( path ),
                             tags = [ "cpp" ],
                             params = { "device": Param( self.devices[ 0 ], choices = self.devices,
                                                         help = "sur quoi le compiler" ) } )
                 for path in self.files()
                 if any( path in matched for matched, _ in specs ) ]

    def run( self, entry, ctx ) -> Outcome:
        # Importé ici et pas en tête de fichier : ce fichier est lu à chaque commande, y
        # compris `errand --envs`, et `loom` n'est sur le chemin qu'une fois la configuration
        # lue ( c'est elle qui dit où sont les sources ).
        sys.path[ : 0 ] = [ str( ctx.root / s ) for s in ( "loom/src", "sdot/src", "otrec/src" )
                            if str( ctx.root / s ) not in sys.path ]
        from loom.compilation import make_executable
        from loom.devices.Device import Device

        name = ctx.params.get( "device", self.devices[ 0 ] )
        device = Device.factory( name )
        if not device.device_is_present:
            from errand import skip
            skip( f"pas de {name} sur cette machine" )

        source = Path( entry.key )
        try:
            exe = make_executable( f"{source.stem}_{device}", [ source ], device )
        except Exception as e:
            return Outcome( status = "FAIL", error = f"la compilation a échoué : {e}" )

        argv = [ exe ] + ( [ ctx.selector ] if ctx.selector else [ ] )
        got = self.shell( argv, ctx )
        text = got.stdout + got.stderr
        if got.returncode:
            last = [ l for l in text.splitlines() if l.startswith( ( "FAIL", "  " ) ) ]
            return Outcome( status = "FAIL", output = text,
                            error = last[ 0 ].strip() if last else f"sortie {got.returncode}" )
        return Outcome( status = "PASS", output = text )


provider( LoomCpp() )
