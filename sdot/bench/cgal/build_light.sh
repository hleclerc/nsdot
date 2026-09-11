#!/usr/bin/env bash
# Les QUATRE binaires de l'etude « prix de la robustesse » : {2D, 3D} x {Epick, Simple_cartesian}.
#
#   power_2d_light   power_2d_lightk        power_3d_light   power_3d_lightk
#                    ^ `k` = noyau allege (LIGHT_KERNEL)
#
# Meme chaine que `build.sh` (CGAL depuis le clone git, gmp/mpfr depuis l'env), plus un `-I` sur
# `2d_des_familles/src` : le mode `--cells` rejoue NOTRE cellule sur la connectivite de CGAL.
set -euo pipefail

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
repo="$( cd "$here/../../.." && pwd )"
CGAL_DIR="${CGAL_DIR:-$HOME/.cache/sdot/cgal}"
prefix="${CONDA_PREFIX:-${MAMBA_PREFIX:-$HOME/.local/share/mamba/envs/nsdot}}"

incs=()
if [ -d "$CGAL_DIR/Installation/include" ]; then
    for d in "$CGAL_DIR"/*/include; do [ -d "$d" ] && incs+=( -I "$d" ); done
else
    incs+=( -I "$CGAL_DIR/include" )
fi

for dim in 2 3; do
    for k in "" k; do
        def=(); [ -n "$k" ] && def=( -DLIGHT_KERNEL )
        out="$here/power_${dim}d_light${k}"
        echo "== $out"
        ${CXX:-g++} -std=c++20 -O3 -march=native -DNDEBUG -DCGAL_NDEBUG "${def[@]}" \
            "${incs[@]}" -I "$prefix/include" -I "$repo/2d_des_familles/src" \
            "$here/power_${dim}d_light.cpp" \
            -L "$prefix/lib" -Wl,-rpath,"$prefix/lib" -lgmp -lmpfr \
            -o "$out"
    done
done
echo "== fait"
