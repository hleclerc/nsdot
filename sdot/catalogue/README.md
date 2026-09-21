Le CATALOGUE des noyaux précompilés du wheel `sdot` -- vide dans un checkout.

`scripts/build_catalogue.py compile` y dépose `<tag>/libsdot_kernels.so` + `<tag>/catalogue.json`
(un tag par variante : `cpu-x86-64-v3`, `cuda`, ...), à partir du relevé `catalogue_record/`.
Le wheel embarque ce répertoire tel quel (`sdot/_catalogue`), et `sdot/__init__.py` l'enregistre à
l'import : ce qui s'y trouve ne se compile pas chez l'utilisateur. Voir
`loom/src/loom/compilation/catalogue.py`.
