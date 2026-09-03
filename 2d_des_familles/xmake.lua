-- Banc 2D autonome : ni jax, ni acpp, ni FFI. Du C++ et un compilateur.
--
-- L'intention : pouvoir essayer une idee en trente secondes au lieu de trois minutes de
-- compilation SYCL, et pouvoir lire l'assembleur qui sort. Ce qu'on y mesure doit ensuite etre
-- reporte dans `sdot` -- ce banc n'est PAS la bibliotheque, c'est le terrain d'essai.
--
--   xmake f -m release && xmake && xmake run pd2d --help

set_project( "2d_des_familles" )
set_languages( "c++20" )
add_rules( "mode.release", "mode.debug" )

target( "pd2d" )
    set_kind( "binary" )
    add_files( "src/main.cpp" )
    add_includedirs( "src" )
    set_warnings( "all" )

    if is_mode( "release" ) then
        -- `-march=native` est ASSUME ici : ce binaire ne quitte pas la machine sur laquelle il a
        -- ete compile, contrairement aux `.so` que `sdot` met en cache.
        -- `-fno-strict-aliasing` : `AaBspPacked` lit ses en-tetes et ses points dans UNE arene de
        -- `TF`, par `reinterpret_cast`. C'est le prix d'une arene, et le meme choix que fait
        -- `sdot` (voir `SDOT_XMAKE_CXXFLAGS`).
        add_cxflags( "-O3", "-march=native", "-fno-math-errno", "-fno-strict-aliasing", { force = true } )
        add_defines( "NDEBUG" )
    end

    -- AMGCL est en-tetes seuls, mais il veut soit `boost::property_tree` pour ses parametres,
    -- soit qu'on lui dise de s'en passer. Et son backend « builtin » est parallelise en OpenMP :
    -- sans le drapeau il compile quand meme, en sequentiel. Aucun des deux ne touche la
    -- geometrie -- il n'y a pas une pragma OpenMP dans le banc.
    add_defines( "AMGCL_NO_BOOST" )
    add_cxflags( "-fopenmp" )
    add_ldflags( "-fopenmp" )

    add_syslinks( "pthread" )
