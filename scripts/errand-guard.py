#!/usr/bin/env python3
"""Le garde-fou de Claude Code ( PreToolUse sur Bash ), POUR TOUTE LA MACHINE : declare dans
`~/.claude/settings.json` ( `~/.claude/hooks/errand-guard.py` est un lien sur ce fichier ), il vaut dans
chaque projet. Tout ce qui calcule ou compile passe par `errand` ( `~/.local/bin/errand`, un lien sur
`scripts/errand` d'ici ) : une commande lourde lancee a nu est REFUSEE ( code 2 ), et le message dit
quoi faire. Ce qui est lourd : un binaire de banc ( `build/`, `N0/tl`, `N1/n1`, `N1/tests/` ... ),
xmake / make / ninja / un compilateur, `./run`, python sur un script ( `python3 -` et `python3 -c`,
les petits scripts d'edition, passent ), nohup. Ce qui est libre : git, grep, sed, cat, ls, et tout
ce qui est deja sous `errand`. `job`, l'ancien lanceur, n'existe plus : l'appeler est refuse aussi.
Un autre projet ajoute ses binaires a `lourd`."""
import json, re, sys

try:
    data = json.load( sys.stdin )
except Exception:
    sys.exit( 0 )
if data.get( "tool_name" ) != "Bash":
    sys.exit( 0 )
cmd = data.get( "tool_input", {} ).get( "command", "" )

if re.search( r"(^|[\s;&|(])(\S*/)?job(\s|$)", cmd ):     # l'ancien lanceur : retire
    sys.stderr.write(
        "REFUSE : `job` n'existe plus. `errand -- <cmd>` ( partage la machine ), `errand -x -- <cmd>` ( benchmark :\n"
        "la machine pour lui seul ), `errand --queue` ( ce qui tourne ), `errand --claim-wait` ( attendre le calme ).\n" )
    sys.exit( 2 )
if re.search( r"(^|[\s;&|(])(\S*/)?errand(\s|$)", cmd ):   # `errand -- ...`, `errand -x -- ...`, `errand --queue`
    sys.exit( 0 )

lourd = [
    r"(^|[\s;&|(])(\./)?build/",                         # un binaire du banc
    r"\bxmake\s+(run|build|b|r|f|config)?\b",
    r"(^|[\s;&|(])(make|ninja|cmake|g\+\+|gcc|clang\+\+|clang|nvcc|cargo|pytest)\b",
    r"(^|[\s;&|(])\./run\b",
    r"(^|[\s;&|(])python3?\s+(?!-\s|-c\s|-m\s+py_compile|--version)",   # un script python
    r"(^|[\s;&|(])nohup\b",
    r"(^|[\s;&|(])timeout\s+\d+\s+(\./|python)",
    r"N[012]/(tl|n1|tests/|N2\.tl)",                     # tl24_LMO : le compilateur, ses bancs, l'emission
    r"(^|[\s;&|(])(bash|sh)\s+\S*tests/\S+\.sh\b",     # un banc en script
]
for p in lourd:
    if re.search( p, cmd ):
        sys.stderr.write(
            "REFUSE par ~/.claude/hooks/errand-guard.py : cette commande calcule ou compile, elle doit passer par\n"
            "errand : `errand -- <cmd>` ( generique, partage la machine ) ou `errand -x -- <cmd>` ( benchmark,\n"
            "la machine pour lui seul ; les autres attendent ). `errand --queue` montre ce qui tourne.\n"
            "Motif : %s\n" % p )
        sys.exit( 2 )
sys.exit( 0 )
