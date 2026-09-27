-- =====================================================================================
-- `solvers_des_familles` : le banc des SOLVEURS, sur l'engin de diagrammes qui a gagne dans
-- `2d_des_familles`. Du C++ et un compilateur ; asimd vient de `../sdot/include`.
--
--   xmake f -m release && xmake
--   xmake run check                  l'engin contre le balayage complet ( 2D, 3D, Voronoi, Laguerre )
--   xmake run diagramme              un diagramme chronometre, sur la suite
--   xmake run newton --help          le transport semi-discret resolu, chronometre par poste
--   xmake run ecrasement --help      jusqu'ou une direction de Newton peut aller avant une cellule vide
--   xmake run multiechelle --help    resoudre sur des representants, prolonger, resoudre en dessous
--   xmake run densite --help         une somme de gaussiennes pour densite, la continuation en largeur
--   xmake run image --help           une IMAGE pour densite, integree SUR LE BORD ( jamais de decoupage )
--   xmake run fp32 --help            ce que la SIMPLE PRECISION coute a la geometrie, mesure
--   xmake run memo                   la memoire en 3D : les voisins d'hier proposes d'abord, borne superieure
--
-- OU EST QUOI :
--   src/util/      les types, l'horloge, les fils
--   src/accel/     le BSP et le majorant affine des poids
--   src/cell/      L'ENGIN : le PLAN bissecteur ( `Plan.h`, le seul endroit ou il se construit ),
--                  la cellule qui dirige, 2D ( registres ) et 3D ( sommets ), les
--                  fournisseurs BSP avec elagage, le balayage temoin -- a priori on n'y touche pas
--   src/diagram/   `PowerDiagram<D,TK,MaxNv>` : ce que le solveur voit
--   src/solver/    le laplacien de Laguerre, les solveurs lineaires, Newton amorti
--   src/bench/     les nuages, les options, le dispatch
--   src/mains/     un binaire par question
-- =====================================================================================

set_project( "solvers_des_familles" )
set_languages( "c++20" )
add_rules( "mode.release", "mode.debug" )

-- `mode.release` DEPOUILLE LE BINAIRE, ce qui rendait `-g` inutile : `perf report` ne sortait que
-- des adresses. On garde les symboles -- ils ne coutent que de la place sur le disque.
set_strip( "none" )

target( "bench" )
    set_kind( "static" )
    add_files( "src/bench/*.cpp" )
    add_includedirs( "src", { public = true } )
    set_warnings( "all" )
    if is_mode( "release" ) then
        add_cxflags( "-O3", "-march=native", "-fno-math-errno", { force = true } )
        add_defines( "NDEBUG" )
    end

-- LES REGLAGES D'UN BINAIRE, EN UN SEUL ENDROIT. Deux binaires du meme code compiles differemment
-- s'ecartent de 4 % ; toute cible passe par ici.
local function reglages()
    add_deps( "bench" )
    add_includedirs( "src", "../sdot/include" )
    set_warnings( "all" )
    if is_mode( "release" ) then
        -- `-march=native` est ASSUME : ces binaires ne quittent pas la machine.
        -- PAS DE `-g` ICI, ET C'ETAIT MESURE. Sur `main_image.cpp`, `-g` faisait passer `cc1plus`
        -- de 3 a 7 Go et la compilation de six a douze minutes ; le plafond memoire de `job`
        -- ( 8 Go ) le tuait une fois sur deux. `set_strip( "none" )` garde la table des symboles,
        -- ce qui suffit a `perf` pour nommer les fonctions -- seules les LIGNES manquent, et les
        -- chronos internes par poste les remplacent avantageusement.
        -- DEPUIS LE § 18 le pic est tombe a 1.1 Go et la compilation a 56 s : `-g` redeviendrait
        -- tenable si on voulait les numeros de ligne. A remesurer avant d'y toucher.
        add_cxflags( "-O3", "-march=native", "-fno-math-errno", { force = true } )
        add_defines( "NDEBUG" )
    end
    -- AMGCL est en-tetes seuls, sans boost si on le lui dit, et son backend « builtin » est
    -- parallelise en OpenMP. Eigen est en-tetes seuls.
    add_defines( "AMGCL_NO_BOOST" )
    add_sysincludedirs( "/usr/include/eigen3" )
    add_cxflags( "-fopenmp" )
    add_ldflags( "-fopenmp" )
    add_syslinks( "pthread" )
end

for _, nom in ipairs( { "check", "diagramme", "newton", "ecrasement", "glissement", "homotopie", "multiechelle", "densite", "image", "memo", "fp32" } ) do
    target( nom )
        set_kind( "binary" )
        add_files( "src/mains/main_" .. nom .. ".cpp" )
        set_rundir( "$(projectdir)" )                    -- `--cases ../2d_des_familles/cases` s'y lit
        reglages()
end
