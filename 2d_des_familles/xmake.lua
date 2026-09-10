-- Banc autonome : ni jax, ni acpp, ni FFI. Du C++ et un compilateur.
--
-- L'intention : pouvoir essayer une idee en trente secondes au lieu de trois minutes de
-- compilation SYCL, et pouvoir lire l'assembleur qui sort. Ce qu'on y mesure doit ensuite etre
-- reporte dans `sdot` -- ce banc n'est PAS la bibliotheque, c'est le terrain d'essai.
--
-- OU EST QUOI :
--
--   src/util/          `common.h` (TF, SI, Vec<D>), les fils
--   src/geometry/      la cellule 2D, la cellule 3D, le diagramme, le majorant des poids
--   src/spatial_accel/ LES ACCELERATEURS : « quels germes peuvent couper cette cellule ? »
--                      -- nomme ainsi et pas `accel/` parce que dans ce depot « accelerateur »
--                      designe aussi un GPU, et ce repertoire n'en contient aucun.
--   src/solver/        Newton amorti
--   src/bench/         le harnais partage : options, nuages, chronometre, affichage
--   src/mains/         un `main_Xyz.cpp` par accelerateur
--
-- UN BINAIRE PAR ACCELERATEUR. Il y avait un `main.cpp` de deux mille lignes qui les portait tous,
-- avec un drapeau par idee et plus rien pour dire quel drapeau allait avec quel accelerateur. Le
-- decoupage a un second effet, mesurable : chaque binaire n'instancie plus que SES combinaisons de
-- templates, donc il compile en quelques secondes au lieu d'une minute.
--
--   xmake f -m release && xmake && xmake run pd_bsp --help

set_project( "2d_des_familles" )
set_languages( "c++20" )
add_rules( "mode.release", "mode.debug" )

-- ce que tous les bancs partagent : les options, les nuages, le chronometre, l'affichage.
target( "bench" )
    set_kind( "static" )
    add_files( "src/bench/*.cpp" )
    add_includedirs( "src", { public = true } )
    set_warnings( "all" )
    if is_mode( "release" ) then
        -- `-march=native` est ASSUME ici : ce binaire ne quitte pas la machine sur laquelle il a
        -- ete compile, contrairement aux `.so` que `sdot` met en cache.
        -- `-fno-strict-aliasing` A ETE RETIRE. Il coutait 10 % au noyau de coupe : la machine a
        -- etats passe de x0.95 a x1.04 face a la version `musttail` sur ce seul drapeau ( mediane
        -- de neuf lancers, cf. `Noyau2DEtats.h` ). `AaBspPacked` lit son arene par
        -- `reinterpret_cast` et etait la raison de sa presence ; si un banc se met a rendre des
        -- resultats faux, c'est la qu'il faut regarder -- `pd_check` le dira.
        add_cxflags( "-O3", "-march=native", "-fno-math-errno", { force = true } )
        add_defines( "NDEBUG" )
    end

-- un `main_Xyz.cpp` par accelerateur, plus les deux bancs transversaux (`check`, `newton`).
local bancs = {
    bsp    = "src/mains/main_bsp.cpp",
    grid   = "src/mains/main_grid.cpp",
    packed = "src/mains/main_packed.cpp",
    bsp4   = "src/mains/main_bsp4.cpp",
    obsp   = "src/mains/main_obsp.cpp",
    hull   = "src/mains/main_hull.cpp",
    pack   = "src/mains/main_pack.cpp",
    pre    = "src/mains/main_pre.cpp",
    loc    = "src/mains/main_loc.cpp",
    front  = "src/mains/main_front.cpp",
    memo   = "src/mains/main_memo.cpp",
    newton = "src/mains/main_newton.cpp",
    check  = "src/mains/main_check.cpp",
    supercellules = "src/mains/main_supercellules.cpp",
    noyau  = "src/mains/main_noyau.cpp",
    n50    = "src/mains/main_n50.cpp",
    etats  = "src/mains/main_etats.cpp",
    bspf   = "src/mains/main_bspf.cpp",
}

-- LE BANC PORTABLE est a part : il demande highway, qui n'est pas toujours la. La cible
-- n'existe que si l'en-tete se trouve, de sorte qu'un `xmake` sans la bibliotheque construit
-- tout le reste sans broncher.
option( "highway" )
    add_cxxincludes( "hwy/highway.h" )
    add_links( "hwy" )
option_end()

for nom, src in pairs( bancs ) do
    target( "pd_" .. nom )
        set_kind( "binary" )
        add_files( src )
        add_deps( "bench" )
        add_includedirs( "src" )
        set_warnings( "all" )
        if is_mode( "release" ) then
            add_cxflags( "-O3", "-march=native", "-fno-math-errno", { force = true } )
            add_defines( "NDEBUG" )
        end
        -- AMGCL est en-tetes seuls, mais il veut soit `boost::property_tree` pour ses parametres,
        -- soit qu'on lui dise de s'en passer. Et son backend « builtin » est parallelise en
        -- OpenMP : sans le drapeau il compile quand meme, en sequentiel. Aucun des deux ne touche
        -- la geometrie -- il n'y a pas une pragma OpenMP dans le banc.
        add_defines( "AMGCL_NO_BOOST" )
        add_cxflags( "-fopenmp" )
        add_ldflags( "-fopenmp" )
        add_syslinks( "pthread" )
end

-- LE BANC DE PORTABILITE, en deux cibles sur une seule source.
--
-- `asimd` est RAPATRIE dans `ext/asimd` ( en-tetes seuls, rien a installer ), donc `pd_asimd`
-- se construit toujours. `pd_hwy` ajoute la colonne highway quand la bibliotheque est la.
target( "pd_asimd" )
    set_kind( "binary" )
    add_files( "src/mains/main_hwy.cpp" )
    add_includedirs( "src", "ext/asimd/src" )
    set_warnings( "all" )
    if is_mode( "release" ) then
        add_cxflags( "-O3", "-march=native", "-fno-math-errno", { force = true } )
        add_defines( "NDEBUG" )
    end

target( "pd_hwy" )
    set_kind( "binary" )
    set_default( false )                                 -- pas construit par un `xmake` nu
    add_files( "src/mains/main_hwy.cpp" )
    add_includedirs( "src", "ext/asimd/src" )
    add_defines( "AVEC_HIGHWAY" )
    add_options( "highway" )
    set_warnings( "all" )
    if is_mode( "release" ) then
        add_cxflags( "-O3", "-march=native", "-fno-math-errno", { force = true } )
        add_defines( "NDEBUG" )
    end
