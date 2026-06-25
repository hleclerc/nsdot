"""Framework de test Python, miroir de tests/cpp/test_main.h.

Un fichier de test ressemble à :

    from .test_main import test

    if test( "my test", [ "[fast]" ] ):
        assert 0 == 0

`test( name, tags )` fonctionne en deux phases, pilotées par le runner
(scripts/run_tests.py) :

* phase de collecte -> enregistre le test, retourne False (le corps est sauté)
* phase d'exécution -> retourne True uniquement pour le test en cours

Le runner recharge chaque module de test une fois par test sélectionné, de sorte
que chaque corps de test s'exécute isolément et qu'un échec (assert / exception)
soit capturé test par test, exactement comme le harnais C++.
"""
from sdot.util.info import info, infox
import sys

builtins = __import__( 'builtins' )
setattr( builtins, "infox", infox )
setattr( builtins, "info", info )



GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"


class Test:
    def __init__( self, name, tags, file, line, module ):
        self.name   = name
        self.tags   = tags            # list[ str ]
        self.file   = file
        self.line   = line
        self.module = module          # __name__ du module, pour reload + exécution

    @property
    def tag_text( self ):
        # même représentation que côté C++ : "[fast][core]"
        return "".join( self.tags )


all_the_tests: list[ Test ] = []

# état de pilotage, positionné par le runner
PHASE_COLLECT = 0
PHASE_RUN     = 1
test_phase    = PHASE_COLLECT
test_filter   = None                  # Test en cours d'exécution (PHASE_RUN)


def _normalize_tags( tags ):
    if tags is None:
        return []
    if isinstance( tags, str ):
        return [ tags ]
    return list( tags )


def test( name, tags = None ):
    tags = _normalize_tags( tags )

    if test_phase == PHASE_COLLECT:
        frame  = sys._getframe( 1 )
        file   = frame.f_code.co_filename
        line   = frame.f_lineno
        module = frame.f_globals.get( "__name__" )

        if any( t.name == name for t in all_the_tests ):
            raise RuntimeError( f"un test nommé { name!r } est déjà enregistré" )
        all_the_tests.append( Test( name, tags, file, line, module ) )
        return False

    # phase d'exécution : seul le test ciblé exécute son corps
    return test_filter is not None and name == test_filter.name


def matches( t, filter_names, filter_tags ):
    """Mêmes sémantiques que TestFunc::matches dans test_main.h.

    - un nom filtre par sous-chaîne
    - un tag filtre par sous-chaîne dans la concaténation des tags
    """
    test_names = []
    for filter_name in filter_names:
        s = filter_name.split( "::" )
        if len( s ) == 2:
            test_names.append( s[ 1 ] )
    if test_names:
        if not any( n in t.name for n in test_names ):
            return False
    if filter_tags:
        if not any( tag in t.tag_text for tag in filter_tags ):
            return False
    return True
