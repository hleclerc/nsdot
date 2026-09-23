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

-- LES REGLAGES D'UN BANC, EN UN SEUL ENDROIT.
--
-- Toute cible de VARIANTE doit etre construite EXACTEMENT comme la cible de reference a laquelle on
-- la compare. Ce n'est pas de la coquetterie : deux binaires du meme code compile differemment
-- s'ecartent ici de 4 %, et une comparaison entre une cible de la boucle `bancs` ( qui a
-- `add_deps( "bench" )` et `-fopenmp` ) et une cible ecrite a la main sans eux a deja fait conclure
-- l'inverse de la verite. Voir « CE QUE LA METHODE A COUTE » dans l'en-tete de `Cellule3D.h`.
local function reglages_banc()
    add_deps( "bench" )
    add_includedirs( "src", "ext/asimd/src" )
    set_warnings( "all" )
    if is_mode( "release" ) then
        add_cxflags( "-O3", "-march=native", "-fno-math-errno", { force = true } )
        add_defines( "NDEBUG" )
    end
    -- AMGCL est en-tetes seuls, mais il veut soit `boost::property_tree` pour ses parametres, soit
    -- qu'on lui dise de s'en passer. Et son backend « builtin » est parallelise en OpenMP : sans le
    -- drapeau il compile quand meme, en sequentiel. Aucun des deux ne touche la geometrie -- il n'y
    -- a pas une pragma OpenMP dans le banc.
    add_defines( "AMGCL_NO_BOOST" )
    add_cxflags( "-fopenmp" )
    add_ldflags( "-fopenmp" )
    add_syslinks( "pthread" )
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
    par    = "src/mains/main_par.cpp",
    sc1    = "src/mains/main_sc1.cpp",
    sc2    = "src/mains/main_sc2.cpp",
    n503d  = "src/mains/main_n503d.cpp",
    bspf3d = "src/mains/main_bspf3d.cpp",
    hist   = "src/mains/main_hist.cpp",
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
        reglages_banc()
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

-- LE TEMOIN EXTERIEUR. `cgal-dev` n'est pas une dependance du projet : la cible n'existe que si
-- l'en-tete et les deux bibliotheques exactes sont la, et un `xmake` nu ne la construit pas.
option( "cgal" )
    add_cxxincludes( "CGAL/Regular_triangulation_2.h" )
    add_links( "gmp", "mpfr" )
option_end()

option( "cgal3" )
    add_cxxincludes( "CGAL/Regular_triangulation_3.h" )
    add_links( "gmp", "mpfr" )
option_end()

target( "pd_cgal3" )
    set_kind( "binary" )
    set_default( false )
    add_files( "src/mains/main_cgal3.cpp" )
    add_options( "cgal3" )
    set_warnings( "all" )
    if is_mode( "release" ) then
        add_cxflags( "-O3", "-march=native", "-fno-math-errno", { force = true } )
        add_defines( "NDEBUG" )
    end

target( "pd_cgal" )
    set_kind( "binary" )
    set_default( false )
    add_files( "src/mains/main_cgal.cpp" )
    add_options( "cgal" )
    set_warnings( "all" )
    if is_mode( "release" ) then
        add_cxflags( "-O3", "-march=native", "-fno-math-errno", { force = true } )
        add_defines( "NDEBUG" )
    end

-- LE DIAGNOSTIC DU PARCOURS 3D ( noeuds depiles, sommets lus par l'elagage ), a part.
target( "pd_bspf3d_cpt" )
    set_kind( "binary" )
    set_default( false )
    add_files( "src/mains/main_bspf3d.cpp" )
    reglages_banc()
    add_defines( "BSP3_COMPTE=1", "NOYAU3D_COMPTE=1" )

-- LA LARGEUR DE LA PREMIERE PASSE 3D, un binaire par largeur. Meme lecon qu'en 2D : deux
-- instanciations dans le meme executable se genent, et la comparaison ne veut plus rien dire.
for _, w in ipairs( { 1, 4, 8, 16 } ) do
    target( "pd_n503d_w" .. w )
        set_kind( "binary" )
        set_default( false )
        add_files( "src/mains/main_n503d.cpp" )
        reglages_banc()
        add_defines( "LARGEUR_BANC=" .. w )
end

-- LE MELANGE DES COUPES ( proposees / effectives ), hors chronometre : un binaire a part, parce
-- qu'un compteur dans la boucle chaude fausserait ce qu'il mesure.
target( "pd_n503d_cpt" )
    set_kind( "binary" )
    set_default( false )
    add_files( "src/mains/main_n503d.cpp" )
    reglages_banc()
    add_defines( "NOYAU3D_COMPTE=1" )

-- L'ORDRE DES TESTS DU NOYAU, UN PAR BINAIRE. Une seule instanciation de `etape<NB>` par
-- executable : les deux variantes dans le meme binaire se genent dans le cache d'instructions et la
-- comparaison ne veut plus rien dire ( voir l'en-tete de `main_n50.cpp` ).
--
-- ATTENTION : ces cibles ont longtemps ete comparees a `pd_n50` / `pd_bspf`, qui sortent de la
-- boucle `bancs` et n'avaient PAS la meme recette. Les chiffres d'ordre 2D obtenus ainsi sont a
-- refaire maintenant que `reglages_banc()` les aligne.
for _, o in ipairs( { 1, 2, 3 } ) do
    target( "pd_bspf_ord" .. o )
        set_kind( "binary" )
        set_default( false )
        add_files( "src/mains/main_bspf.cpp" )
        reglages_banc()
        add_defines( "ORDRE_NOYAU=" .. o )

    target( "pd_n50_ord" .. o )
        set_kind( "binary" )
        add_files( "src/mains/main_n50.cpp" )
        reglages_banc()
        add_defines( "ORDRE_BANC=" .. o )
end
