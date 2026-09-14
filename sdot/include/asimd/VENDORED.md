# asimd, vendored

Copie de `src/asimd` du dépôt https://github.com/hleclerc/asimd (commit `dd45dad`, « load_partial /
store_partial: the lanes of a set, not one byte outside »), telle qu'elle est aussi rapatriée dans
`2d_des_familles/ext/asimd` pour le banc.

Ici parce que les kernels engendrés par `loom` se compilent avec `-I sdot/include` et rien d'autre
(voir `loom/compilation/__init__.py::cpp_include_root`), et parce que la roue installée embarque
cet arbre tel quel. Pour mettre à jour : recopier `src/asimd` du dépôt et noter le commit ici.
