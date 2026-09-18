-- =====================================================================================
-- `solvers_des_familles` : le banc des SOLVEURS, sur l'engin de diagrammes qui a gagne dans
-- `2d_des_familles`. Du C++ et un compilateur ; asimd vient de `../sdot/include`.
--
--   xmake f -m release && xmake
--   xmake run check                  l'engin contre le balayage complet ( 2D, 3D, Voronoi, Laguerre )
--   xmake run diagramme              un diagramme chronometre, sur la suite
--   xmake run newton --help          le transport semi-discret resolu, chronometre par poste
--   xmake run ecrasement --help      jusqu'ou une direction de Newton peut aller avant une cellule vide
--
-- OU EST QUOI :
--   src/util/      les types, l'horloge, les fils
--   src/accel/     le BSP et le majorant affine des poids
--   src/cell/      L'ENGIN : la cellule qui dirige, 2D ( registres ) et 3D ( sommets ), les
--                  fournisseurs BSP avec elagage, le balayage temoin -- a priori on n'y touche pas
--   src/diagram/   `PowerDiagram<D,TK,MaxNv>` : ce que le solveur voit
--   src/solver/    le laplacien de Laguerre, les solveurs lineaires, Newton amorti
--   src/bench/     les nuages, les options, le dispatch
--   src/mains/     un binaire par question
-- =====================================================================================

set_project( "solvers_des_familles" )
set_languages( "c++20" )
add_rules( "mode.release", "mode.debug" )

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

for _, nom in ipairs( { "check", "diagramme", "newton", "ecrasement", "glissement" } ) do
    target( nom )
        set_kind( "binary" )
        add_files( "src/mains/main_" .. nom .. ".cpp" )
        set_rundir( "$(projectdir)" )                    -- `--cases ../2d_des_familles/cases` s'y lit
        reglages()
end
