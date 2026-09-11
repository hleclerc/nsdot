#!/usr/bin/env bash
# CGAL DEPUIS SES SOURCES GIT, puis le banc compile contre elles.
#
# Depuis git et non depuis un paquet : CGAL bouge souvent, et un etalon qui compare a une version
# d'il y a deux ans ne dit pas ce qu'on veut savoir. CGAL 6 est une bibliotheque d'EN-TETES -- il
# n'y a donc rien a batir dans le clone, seulement un `-I` a pointer dessus (plus gmp/mpfr a lier,
# qui eux sont des vraies bibliotheques et viennent de l'env micromamba).
#
#   ./sdot/bench/cgal/build.sh          # clone (ou met a jour) + compile
#   ./sdot/bench/cgal/power_2d 1000000  # lance
#
# `CGAL_DIR` : ou le clone atterrit. Par defaut a cote du cache des toolchains, pas dans le
# checkout -- ce n'est ni notre source ni un artefact de build a nous.
set -euo pipefail

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CGAL_DIR="${CGAL_DIR:-$HOME/.cache/sdot/cgal}"
CGAL_REF="${CGAL_REF:-main}"

if [ -d "$CGAL_DIR/.git" ]; then
    echo "== CGAL: mise a jour de $CGAL_DIR ($CGAL_REF)"
    git -C "$CGAL_DIR" fetch --depth 1 origin "$CGAL_REF"
    git -C "$CGAL_DIR" checkout -q FETCH_HEAD
else
    echo "== CGAL: clone dans $CGAL_DIR ($CGAL_REF)"
    mkdir -p "$( dirname "$CGAL_DIR" )"
    git clone --depth 1 --branch "$CGAL_REF" https://github.com/CGAL/cgal.git "$CGAL_DIR"
fi
echo "   $( git -C "$CGAL_DIR" log -1 --format='%h %ad %s' --date=short )"

# les en-tetes : CGAL en depot git est ECLATE en paquets (Kernel_23/include, Triangulation_2/...),
# la ou une release les recolle sous un seul `include/`. On passe donc un `-I` par paquet utile.
incs=()
if [ -d "$CGAL_DIR/Installation/include" ]; then
    for d in "$CGAL_DIR"/*/include; do
        [ -d "$d" ] && incs+=( -I "$d" )
    done
else
    incs+=( -I "$CGAL_DIR/include" )
fi

# gmp / mpfr / boost : de l'env micromamba actif (voir `.envs.py`, paquets de la couche NSDOT).
prefix="${CONDA_PREFIX:-${MAMBA_PREFIX:-}}"
if [ -z "$prefix" ]; then
    echo "!! aucun env conda/micromamba actif : gmp/mpfr/boost introuvables." >&2
    echo "   micromamba activate nsdot   (ou ./run env create)" >&2
    exit 1
fi

echo "== compilation du banc"
${CXX:-g++} -std=c++20 -O3 -march=native -DNDEBUG -DCGAL_NDEBUG \
    "${incs[@]}" -I "$prefix/include" \
    "$here/power_2d.cpp" \
    -L "$prefix/lib" -Wl,-rpath,"$prefix/lib" -lgmp -lmpfr \
    -o "$here/power_2d"

echo "== fait : $here/power_2d"
