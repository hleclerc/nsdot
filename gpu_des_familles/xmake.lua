-- =====================================================================================
-- `gpu_des_familles` : le banc GPU. Le diagramme de puissance de `solvers_des_familles` refait en
-- CUDA, avec le moteur CPU en TEMOIN sur le meme arbre. L'arbre et les nuages viennent de
-- `../solvers_des_familles/src` ; asimd ( pour le temoin ) de `../sdot/include`.
--
--   xmake f -m release --cuda=/usr/local/cuda-13.3 && xmake
--   xmake run mesures --threads 8              la suite, 2D puis 3D : CPU temoin, GPU par variante
--   xmake run mesures --kernel float ...       le flottant du noyau, CPU et GPU
--   xmake run mesures --variante fil|voies     une variante seule
--
-- OU EST QUOI :
--   src/gpu/Arbre.cuh       l'arbre tel que le GPU le lit : noeuds AoS alignes, germes dans l'ordre
--   src/gpu/Fil2D.cuh       UNE CELLULE PAR THREAD, 2D : la boucle scalaire en memoire locale
--   src/gpu/FilReg2D.cuh    une cellule par thread, LES SOMMETS EN REGISTRES, l'excursion au-dela de 8
--   src/gpu/FilMix2D.cuh    le meme, R sommets en registres et la queue en memoire par une boucle ( R = 6 : les lignes )
--   src/gpu/FilBrk2D.cuh    tout en registres avec des sorties a nb, les cellules > R en seconde passe
--   src/gpu/FilRot2D.cuh    le meme, le remontage par decalage en barillet au lieu de lectures indexees
--   src/gpu/FilNrm2D.cuh    la cellule normalisee AVANT la coupe, tout a positions fixes, rien a recomposer : LE GAGNANT 2D
--   src/gpu/FilOrd2D.cuh    les sommets ne bougent plus : l'ordre cyclique dans un registre de 64 bits ( -21 % de registres )
--   src/gpu/FilSuc2D.cuh    tout en masques : la cellule est une relation de succession ( succ / pred )
--   src/gpu/FilMsk2D.cuh    registres tries, mais la frontiere cueillie par des masques partages entre x, y et c
--   src/gpu/FilUni2D.cuh    le meme en une seule boucle, lanes persistantes : perdu ( les phases se remelangent )
--   src/gpu/FilShm2D.cuh    la rotation en memoire partagee : -26 % d'instructions, -25 % d'occupation, perdu de peu
--   src/gpu/FilPh2D.cuh     LES PHASES : un noyau persistant par SM, trois files par bloc, l'etat en RAM,
--                           la file de coupe groupee par feuille : tri, binning, arene, arene compactee
--   src/gpu/Paquet2D.cuh    plusieurs cellules par voie, un parcours par warp, plans en bloc : perdu
--   src/gpu/Voies2D.cuh     LA CELLULE SUR HUIT VOIES, 2D : voie = sommet, quatre cellules par warp
--   src/gpu/Fil3D.cuh       une cellule par thread, 3D : le polytope simple porte par ses sommets
--   src/gpu/Voies3D.cuh     LA CELLULE SUR LE WARP, 3D : la voie l porte les sommets l, l+32, ... ; deux passes
--   src/gpu/Mesures.h/.cu   `DiagrammeGpu<D,TK>` : televersement, lancement, chrono par evenements
--   src/mains/main_mesures.cpp   le banc
--   src/mains/main_bande.cu      le debit en streaming SoA : le cout d'une phase si l'etat va en RAM
-- =====================================================================================

set_project( "gpu_des_familles" )
set_languages( "c++20" )
add_rules( "mode.release", "mode.debug" )

target( "bande" )
    set_kind( "binary" )
    add_files( "src/mains/main_bande.cu" )
    set_rundir( "$(projectdir)" )
    add_cugencodes( "sm_75" )
    add_cuflags( "-O3", { force = true } )

target( "mesures" )
    set_kind( "binary" )
    add_files( "src/mains/main_mesures.cpp", "../solvers_des_familles/src/bench/*.cpp", "src/gpu/*.cu" )
    add_includedirs( "src", "../solvers_des_familles/src", "../sdot/include" )
    set_rundir( "$(projectdir)" )                        -- `--cases ../2d_des_familles/cases` s'y lit
    set_warnings( "all" )
    -- l'hote : les memes reglages que le banc CPU, pour que le temoin soit le meme binaire
    if is_mode( "release" ) then
        add_cxflags( "-O3", "-march=native", "-fno-math-errno", { force = true } )
        add_defines( "NDEBUG" )
    end
    -- le GPU de la machine ( RTX 2080 Ti ). PAS de `--use_fast_math` : une division approchee
    -- deplace un sommet, et l'elagage est exact seulement si la geometrie l'est.
    add_cugencodes( "sm_75" )
    add_cuflags( "-O3", "-lineinfo", { force = true } )
    add_syslinks( "pthread" )
