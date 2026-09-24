# OÙ EST QUOI

```
src/gpu/Arbre.cuh       l'arbre tel que le GPU le lit : noeuds AoS alignés sur 16 octets ( boîte,
                        majorant affine, tranche, fils droit ), germes dans l'ordre de l'arbre,
                        le plan bissecteur, le test d'élagage pour UN sommet, la proximité
src/gpu/Fil2D.cuh       UNE CELLULE PAR THREAD, 2D : `coupe_large` mot pour mot, la pile en local
src/gpu/FilReg2D.cuh    UNE CELLULE PAR THREAD, LES SOMMETS EN REGISTRES, 2D : le noyau à registres
                        du CPU exécuté par un thread scalaire, l'excursion au-delà de huit
src/gpu/FilMix2D.cuh    le même, `R` sommets en registres et la queue en mémoire par une boucle
                        ordinaire, sans excursion — le gagnant sur les lignes / Laguerre à `R = 6`
src/gpu/FilBrk2D.cuh    tout en registres, tout déroulé, chaque boucle SORT à `nb`, et les cellules
                        qui dépassent `R` refaites en SECONDE PASSE
src/gpu/FilRot2D.cuh    le même avec le remontage par DÉCALAGE EN BARILLET au lieu de lectures
                        indexées
src/gpu/FilNrm2D.cuh    la cellule NORMALISÉE AVANT la coupe : tout se lit à des positions fixes,
                        rien n'est recomposé après — LE GAGNANT en 2D à `R = 8`
src/gpu/FilOrd2D.cuh    LES SOMMETS NE BOUGENT PLUS : un registre de 64 bits porte l'ordre
                        cyclique ( un octet = le slot, en one-hot ) — −21 % de registres, à
                        vitesse égale ([§ profils](05-profils.md))
src/gpu/FilMsk2D.cuh    registres TRIÉS, la frontière par quatre masques de rôle et le remontage
                        en UN SEUL barillet — le plus petit noyau à registres triés : 96 / 59
                        registres contre 128 / 74, pour +6 % de temps ([§ profils](05-profils.md))
src/gpu/FilSuc2D.cuh    TOUT EN MASQUES : la cellule est une relation de succession ( deux
                        registres `succ` / `pred` ), plus une position ni un index — mêmes
                        registres, même vitesse ([§ profils](05-profils.md))
src/gpu/FilUni2D.cuh    le même en UNE SEULE BOUCLE ( un pas par itération ) et avec des lanes
                        persistantes — mesuré, et perdu ([§ profils](05-profils.md))
src/gpu/FilShm2D.cuh    le même avec la rotation en MÉMOIRE PARTAGÉE ( layout [case][thread] ) :
                        −26 % d'instructions, −25 % d'occupation, égalité — perdu de peu ([§ profils](05-profils.md))
src/gpu/FilPh2D.cuh     LES PHASES : un noyau PERSISTANT par SM, trois files par bloc ( attend une
                        boîte / a une feuille / finie ), l'état des cellules en vol en RAM, et la
                        file de coupe GROUPÉE PAR FEUILLE ( tri bitonique par blocs de 64 ) —
                        écrit et mesuré ([§ profils](05-profils.md)) : perdant en `float`, −12 % en `double` (uniforme)
src/gpu/Voies2D.cuh     LA CELLULE SUR LES VOIES, 2D : voie = sommet, `V` = 8, 16 ou 32 voies par
                        cellule ; le débordement est une excursion sur la voie 0
src/gpu/Paquet2D.cuh    PLUSIEURS CELLULES PAR VOIE, un parcours par warp, les plans d'une feuille
                        testés en bloc — mesuré, et perdu ([§ profils](05-profils.md))
src/gpu/Fil3D.cuh       une cellule par thread, 3D : `Cellule3` mot pour mot, tout en mémoire locale
src/gpu/Voies3D.cuh     LA CELLULE SUR LE WARP, 3D : la voie `l` porte les sommets `l, l+32, …`
                        dans `S` cases de registres, coupes et voisins en octets ; deux passes
src/gpu/Image2D.cuh     LA DENSITÉ IMAGE : l'intégrale de bord `∮ G dy` avec `G` la somme
                        préfixe de la ligne de pixels, le parcours d'arête en Amanatides-Woo, et
                        les deux noyaux de traitement PAR LOT ( une voie par cellule, une voie par
                        arête ) ([§ densités](08-densites.md))
src/gpu/Bsp2D.cuh       L'ARBRE CONSTRUIT SUR LE GPU : boîtes par atomiques sur entiers ordonnés,
                        un tri radix par niveau, préordre, et le majorant affine en quatre passes
src/gpu/Hess2D.cuh      L'ASSEMBLAGE de la hessienne depuis les facettes : compter, scanner, remplir
src/gpu/Cg2D.cuh        LE GRADIENT CONJUGUÉ, scalaires gardés sur la carte
src/gpu/Amg2D.cuh       LE MULTIGRILLE MAISON : agrégation, Galerkin par triplets, K-cycle
src/gpu/Lisse2D.cuh     la prolongation LISSÉE par `cusparseSpGEMM` — écrite, mesurée, perdante
src/gpu/Mesures.h/.cu   `DiagrammeGpu<D,TK>` : téléversement, le choix du noyau ( variante,
                        Voronoï / Laguerre, sommets max ), le chrono par événements CUDA
src/mains/main_mesures.cpp   le banc
src/mains/main_chaine.cpp    LA CHAÎNE COMPLÈTE : arbre, mesures, facettes, hessienne, CG / AMG,
                        Newton, densité image — et le témoin CPU de chaque poste
src/mains/main_bande.cu      le débit en streaming SoA : ce que coûte une phase si l'état va en RAM
```

Ce qui est repris de `../solvers_des_familles/src` sans y toucher : `accel/AaBsp.h` (l'arbre,
bâti sur l'hôte), `bench/` (les nuages, les options, le dispatch), et tout `cell/` +
`diagram/PowerDiagram.h` pour le témoin. Le `.cu` est compilé par nvcc, le `main` et le témoin par
g++ avec `-march=native` : la frontière est `Mesures.h`, qui ne contient pas un type CUDA.

---

---

[← sommaire](../README.md)
