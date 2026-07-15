"""A store for SHARED, generated C++ headers -- the mechanism only, nothing about what they hold.

A call would otherwise spell its boilerplate (an axis' `DEFINE_AXIS`, an aggregate's `struct`)
straight into its one-off `.cpp` -- a content-hashed file, unreadable and useless to an editor.
Yet that boilerplate is call-INDEPENDENT, so it belongs in a shared place, written ONCE and then
`#include`d by both the generated `.cpp` and any hand-written helper (which then gets declared
symbols to autocomplete, and compiles on its own).

WHERE, not what: WHAT a header contains is the business of the thing it describes -- an axis knows
its own `DEFINE_AXIS`, an aggregate its own `struct`. This module knows only the store. It lives
under the BUILD directory, never the sources: those may be read-only (a `sudo pip install`), and
writing into them is a bad habit regardless. `build_dir()` already picks a writable location (with
a per-user temp fallback), and `include_root()` is added to the compile `-I` path so the headers
resolve as `sdot/generated/...`. Write-if-changed keeps a deterministic content from churning
mtimes (hence rebuilds).
"""

from . import build_dir


def include_root():
    """The `-I` root the generated headers live under -- kept apart from the build artifacts
    (`.cpp`, `.so`) so it is a clean include tree. Added to the compile flags by the driver."""
    root = build_dir() / "include"
    root.mkdir( parents = True, exist_ok = True )
    return root


def shared_header( rel_path: str, content: str ) -> str:
    """Ensure `content` lives at `rel_path` (e.g. `sdot/generated/axes/num_vertex.h`) under
    `include_root()`, writing only when the bytes would differ, and return `rel_path` -- the
    string to `#include`."""
    path = include_root() / rel_path
    if not ( path.exists() and path.read_text() == content ):
        path.parent.mkdir( parents = True, exist_ok = True )
        path.write_text( content )
    return rel_path
