# asimd, vendored

Copie de `src/asimd` du dépôt https://github.com/hleclerc/asimd (commit `dd45dad`, « load_partial /
store_partial: the lanes of a set, not one byte outside »), telle qu'elle est aussi rapatriée dans
`2d_des_familles/ext/asimd` pour le banc.

Ici parce que les kernels engendrés par `loom` se compilent avec `-I sdot/include` et rien d'autre
(voir `loom/compilation/__init__.py::cpp_include_root`), et parce que la roue installée embarque
cet arbre tel quel. Pour mettre à jour : recopier `src/asimd` du dépôt et noter le commit ici.

Écart local (2026-09-21, à remonter) : `HaD` retiré des trois fonctions qui prennent ou rendent un
`__m256` / `__m256d` (`ops/X86.h::avx_lane_ext_ps/pd`, `impl/SimdVecImpl_AVX.h`) -- nvcc refuse un
type vecteur x86 dans une signature `__host__ __device__`, même sur un template jamais instancié
côté device. Ces fonctions sont x86 seulement, `inline` suffit.
