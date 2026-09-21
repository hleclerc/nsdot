"""Pose `HD` sur les fonctions qu'un noyau peut atteindre -- ce que nvcc exige (`__host__ __device__`)
et que le compilateur hôte ignore (`HD` est vide hors nvcc, voir `common_macros.h`).

    python scripts/annotate_hd.py [--dry-run] [fichiers...]     # défaut : les arbres ci-dessous

La règle : chaque déclaration ET définition de fonction (libre, méthode, constructeur, opérateur,
template) des en-têtes visés reçoit `HD` en tête de ses spécificateurs, après l'en-tête de
template (`template<...>`, ou l'un des raccourcis `T_T`, `UTP`, `SDOT_TEMPLATE_DECL_FOR_*`).
Sont laissées telles quelles : ce qui porte déjà `HD` / `__host__` / `__device__`, et les
fonctions NON template dont le corps touche à ce qui n'existe pas sur un device
(`std::ostream`, `std::vector`, `throw`, ...) -- une fonction template qui y touche ne sera
jamais instanciée côté device, l'annoter est sans risque.

Un outil de MIGRATION, pas une passe à rejouer : une fois les arbres annotés, la convention est
que toute nouvelle fonction atteignable par un noyau s'écrit avec `HD`. Le script reste utile pour
un nouveau fichier.
"""
from pathlib import Path
import argparse
import sys
import re

import clang.cindex as ci

ROOT = Path( __file__ ).resolve().parents[ 1 ]

DEFAULT_TREES = [
    ROOT / "loom" / "include" / "loom" / "support" / "containers",
    ROOT / "loom" / "include" / "loom" / "support" / "algorithms",
    ROOT / "loom" / "include" / "loom" / "support" / "Ct.h",
    ROOT / "loom" / "include" / "loom" / "support" / "CtStr.h",
    ROOT / "loom" / "include" / "loom" / "support" / "CtType.h",
    ROOT / "loom" / "include" / "loom" / "support" / "kernels" / "Ptr.h",
    ROOT / "loom" / "include" / "loom" / "support" / "kernels" / "Group.h",
    ROOT / "loom" / "include" / "loom" / "support" / "kernels" / "Reducer.h",
    ROOT / "loom" / "include" / "loom" / "support" / "kernels" / "IoCategory.h",
    ROOT / "loom" / "include" / "loom" / "support" / "kernels" / "CpuKernelMemorySpace.h",
    ROOT / "loom" / "include" / "loom" / "support" / "kernels" / "CpuHostMemorySpace.h",
    ROOT / "loom" / "include" / "loom" / "support" / "kernels" / "CudaKernelMemorySpace.h",
    ROOT / "loom" / "include" / "loom" / "support" / "kernels" / "CudaGlobalMemorySpace.h",
    ROOT / "sdot" / "include" / "sdot",
]

INCLUDE_DIRS = [ ROOT / "sdot" / "include", ROOT / "loom" / "include", ROOT / "build" / "include" ]


def _resource_include() -> list:
    """Le libclang du paquet pip n'a pas ses en-têtes de ressources (`stddef.h`) : on prend ceux
    d'un clang du système s'il y en a un."""
    import glob
    for pattern in ( "/usr/lib/llvm-*/lib/clang/*/include", "/usr/lib/clang/*/include" ):
        found = sorted( glob.glob( pattern ) )
        if found:
            return [ f"-isystem{ found[ -1 ] }" ]
    return []

# le corps d'une fonction non template qui contient l'un de ces mots n'a rien à faire sur un device
HOST_ONLY = re.compile( r"std::ostream|std::cout|std::cerr|std::string|std::vector|std::map|std::function|std::thread|"
                        r"\bthrow\b|std::runtime_error|printf|\bINFO\b|\bTODO\b|std::barrier|std::mutex" )

TEMPLATE_MACROS = re.compile( r"^(T_[A-Za-z]+|UTP|SDOT_TEMPLATE_DECL_FOR_\w+)$" )

FUNCTION_KINDS = { ci.CursorKind.FUNCTION_DECL, ci.CursorKind.CXX_METHOD, ci.CursorKind.CONSTRUCTOR,
                   ci.CursorKind.DESTRUCTOR, ci.CursorKind.CONVERSION_FUNCTION, ci.CursorKind.FUNCTION_TEMPLATE }


_TOKEN = re.compile( r"[A-Za-z_]\w*|>>|\[\[|\]\]|[^\sA-Za-z_]" )


def _tokenize( text: str, base: int ):
    """(jeton, offset absolu) sur une tranche de source -- libclang ne rend AUCUN jeton pour un
    extent qui commence dans une expansion de macro (`T_T ...`), on découpe donc nous-mêmes. Les
    commentaires sont retirés d'abord (remplacés par des blancs, pour garder les offsets)."""
    def blank( m ):
        return " " * len( m.group( 0 ) )
    text = re.sub( r"//[^\n]*|/\*.*?\*/", blank, text, flags = re.S )
    return [ ( m.group( 0 ), base + m.start() ) for m in _TOKEN.finditer( text ) ]


def insertion_offset( cursor, text: str ):
    """L'offset où poser `HD ` pour cette fonction, ou None (déjà annotée, friend, macro)."""
    ext = cursor.extent
    start, end = ext.start.offset, ext.end.offset
    if start == end or ext.start.file is None:
        # libclang 18 rend un extent VIDE pour un template abrégé (`auto f( auto x )`) : on part du
        # début de la ligne du nom, et on lit jusqu'au nom -- la logique ci-dessous saute ce qui
        # précède les spécificateurs (en-tête de template, macro), donc un début de ligne suffit
        name_off = cursor.location.offset
        start = text.rfind( "\n", 0, name_off ) + 1
        end = text.find( "(", name_off )
        if end < 0:
            return None
        end += 1
    tokens = _tokenize( text[ start:end ], start )
    if not tokens:
        return None
    spell = [ t for t, _ in tokens ]
    if "(" not in spell:
        return None
    # le nom de la fonction doit être dans ses jetons : sinon la déclaration sort d'une MACRO
    # (`SDOT_DIAGRAM_COMMON( ... )`), et c'est la macro qu'il faut annoter, à la main
    name = re.match( r"~?[A-Za-z_]\w*", cursor.spelling or "" )
    name = name.group( 0 ).lstrip( "~" ) if name else ""
    if name and name not in spell:
        return None
    i = 0
    # l'en-tête de template : `template < ... >` équilibré, ou un raccourci macro
    if spell[ 0 ] == "template":
        depth = 0
        for j, s_ in enumerate( spell ):
            if s_ == "<":
                depth += 1
            elif s_ == ">":
                depth -= 1
                if depth == 0:
                    i = j + 1
                    break
            elif s_ == ">>":
                depth -= 2
                if depth <= 0:
                    i = j + 1
                    break
        # une clause `requires` après l'en-tête : trop rare pour valoir un analyseur, on signale
        if i < len( spell ) and spell[ i ] == "requires":
            return None
    elif TEMPLATE_MACROS.match( spell[ 0 ] ):
        i = 1
    # les attributs `[[ ... ]]` en tête
    while i < len( spell ) and spell[ i ] == "[[":
        j = spell.index( "]]", i )
        i = j + 1
    if i >= len( spell ):
        return None
    if spell[ i ] == "friend":
        i += 1
    head = spell[ i: i + 4 ]
    if head[ 0 ] in ( "HD", "HD_INLINE", "__host__", "__device__", "LOOM_EXPORT" ):
        return None
    return tokens[ i ][ 1 ]


def annotate_file( path: Path, dry_run: bool ) -> int:
    # les offsets de libclang sont en OCTETS : on travaille sur le texte décodé octet à octet
    # (latin-1, bijectif), et on ré-encode de même -- l'UTF-8 des commentaires ressort intact
    text = path.read_bytes().decode( "latin-1" )
    index = ci.Index.create()
    args = [ "-x", "c++", "-std=c++20", "-fsyntax-only", *( f"-I{ d }" for d in INCLUDE_DIRS ), *_resource_include(), "-DLOOM_ANNOTATING" ]
    tu = index.parse( str( path ), args = args )
    offsets = set()
    skipped = []

    def visit( cursor ):
        for c in cursor.get_children():
            loc = c.location
            if c.kind in FUNCTION_KINDS:
                if loc.file is not None and Path( loc.file.name ).resolve() == path.resolve():
                    off = insertion_offset( c, text )
                    if off is None:
                        skipped.append( ( loc.line, c.spelling ) )
                    else:
                        # une fonction NON template dont le corps est hôte seulement
                        body = _without_comments( text[ c.extent.start.offset:c.extent.end.offset ] )
                        if c.kind != ci.CursorKind.FUNCTION_TEMPLATE and HOST_ONLY.search( body ) and not _is_abbreviated_template( body ):
                            skipped.append( ( loc.line, c.spelling + " (hôte)" ) )
                        else:
                            offsets.add( off )
                # les fonctions template ont leurs enfants, mais on ne descend pas dans les corps
                continue
            if loc.file is None or Path( loc.file.name ).resolve() == path.resolve():
                visit( c )

    visit( tu.cursor )

    diags = [ d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error ]
    if diags:
        print( f"  {path}: {len(diags)} erreur(s) d'analyse (les annotations tiennent quand même) :", file = sys.stderr )
        for d in diags[ :3 ]:
            print( f"    { d }", file = sys.stderr )

    if skipped:
        for line, name in sorted( skipped ):
            print( f"  {path.relative_to(ROOT)}:{line}: laissé tel quel : { name }" )

    if not offsets:
        return 0
    out = text
    for off in sorted( offsets, reverse = True ):
        out = _insert_aligned( out, off )
    # `HD` vient de `common_macros.h` : un fichier qui ne l'a pas encore le prend
    if "common_macros.h" not in out and "#pragma once\n" in out:
        out = out.replace( "#pragma once\n", "#pragma once\n\n#include <loom/support/common_macros.h> // HD\n", 1 )
    if not dry_run:
        path.write_bytes( out.encode( "latin-1" ) )
    return len( offsets )


def _insert_aligned( text: str, off: int ) -> str:
    """`HD ` à `off`, en reprenant trois espaces à un alignement en colonnes voisin (les
    déclarations de ces en-têtes sont tabulées : le nom doit rester dans sa colonne). D'abord
    dans la course d'espaces qui suit sur la ligne (entre le type et le nom), sinon dans celle
    qui précède."""
    line_end = text.find( "\n", off )
    if line_end < 0:
        line_end = len( text )
    m = re.search( r" {4,}", text[ off:line_end ] )
    if m:
        a, b = off + m.start(), off + m.end()
        return text[ :off ] + "HD " + text[ off:a ] + " " * ( b - a - 3 ) + text[ b: ]
    line_start = text.rfind( "\n", 0, off ) + 1
    m = re.search( r" {4,}$", text[ line_start:off ] )
    if m and m.start() > 0:
        a = line_start + m.start()
        return text[ :a ] + " " * ( off - a - 3 ) + "HD " + text[ off: ]
    return text[ :off ] + "HD " + text[ off: ]


def _without_comments( text: str ) -> str:
    return re.sub( r"//[^\n]*|/\*.*?\*/", " ", text, flags = re.S )


def _is_abbreviated_template( body: str ) -> bool:
    """`auto` dans la liste de paramètres = un template abrégé (libclang le dit FUNCTION_DECL)."""
    head = body.split( "{", 1 )[ 0 ]
    return re.search( r"\bauto\b[^()]*[,)]", head ) is not None or "auto &&" in head or "auto ..." in head


def main():
    p = argparse.ArgumentParser()
    p.add_argument( "paths", nargs = "*" )
    p.add_argument( "--dry-run", action = "store_true" )
    a = p.parse_args()
    roots = [ Path( x ).resolve() for x in a.paths ] or DEFAULT_TREES
    files = []
    for r in roots:
        if r.is_dir():
            files += [ f for f in sorted( r.rglob( "*" ) ) if f.suffix in ( ".h", ".cxx" ) ]
        else:
            files.append( r )
    total = 0
    for f in files:
        n = annotate_file( f, a.dry_run )
        total += n
        print( f"{ f.relative_to( ROOT ) }: { n }" )
    print( f"total: { total } annotation(s){ ' (dry run)' if a.dry_run else '' }" )


if __name__ == "__main__":
    main()
