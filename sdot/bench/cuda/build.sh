#!/usr/bin/env bash
# Le banc GPU. Autonome : il ne depend ni de CGAL ni de `2d_des_familles`, il lit le fichier de
# connectivite ecrit par `power_{2,3}d_light --dump`.
#
# `-arch=sm_75` : Turing (RTX 2080 Ti). A changer sur une autre carte.
# Pas de `--use_fast_math` : il remplacerait les divisions par des reciproques approchees, ce qui
# fausserait justement ce que ce banc essaie de mesurer -- la precision du FP32 sur la geometrie.
set -euo pipefail
here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
nvcc -O3 -std=c++17 -arch="${ARCH:-sm_75}" -lineinfo \
     -Xptxas -v \
     "$here/cells_gpu.cu" -o "$here/cells_gpu" 2>&1 | grep -E "ptxas|error|warning" || true
ls -l "$here/cells_gpu"
