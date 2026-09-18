#include "bench/Args.h"
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace sf {

bool Args::parse( const std::string &s, int &i, int argc, char **argv ) {
    auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
    if      ( s == "-n" )        n         = std::atoi( val() );
    else if ( s == "--reps" )    reps      = std::atoi( val() );
    else if ( s == "--threads" ) par.threads = std::atoi( val() );
    else if ( s == "--no-pin" )  par.pin   = false;
    else if ( s == "--leaf" )    leaf      = std::atoi( val() );
    else if ( s == "--seed" )    graine    = unsigned( std::atoi( val() ) );
    else if ( s == "--weights" ) wscale    = std::atof( val() );
    else if ( s == "--load" )    load      = val();
    else if ( s == "--2d" )      dims      = 2;
    else if ( s == "--3d" )      dims      = 3;
    else if ( s == "--cases" )   cases     = val();
    else if ( s == "--kernel" )  kernel    = val();
    else if ( s == "--maxnv" )   maxnv     = std::atoi( val() );
    else return false;
    return true;
}

void Args::usage() {
    std::printf(
        "  -n N            germes du cas uniforme                (1000000)\n"
        "  --reps R        repetitions chronometrees, minimum     (3)\n"
        "  --threads T     0 = autant que de coeurs               (0)\n"
        "  --no-pin        ne pas epingler les threads\n"
        "  --leaf L        germes par feuille de l'arbre          (10)\n"
        "  --seed S        graine du tirage                       (0)\n"
        "  --weights W     poids aleatoires de l'uniforme, en fraction de h^2\n"
        "  --load FILE     UN nuage, au lieu de la suite ( avec --2d ou --3d )\n"
        "  --2d / --3d     ne derouler QUE cette dimension        (les deux)\n"
        "  --cases DIR     le repertoire des nuages durs          (../2d_des_familles/cases)\n"
        "  --kernel K      le flottant du noyau : double | float  (double)\n"
        "  --maxnv M       sommets max par cellule : 64 | 128 en 2D, 128 | 256 en 3D\n" );
}

void Args::finalise() {
    if ( par.threads <= 0 )
        par.threads = int( std::thread::hardware_concurrency() );
    if ( ! load.empty() && dims == 0 )
        dims = 2;                                        // un fichier n'annonce pas sa dimension
}

template<int D>
std::vector<Nuage<D>> Args::nuages() const {
    if ( ! load.empty() ) {
        Nuage<D> nu;
        nu.nom = load;
        if ( ! charge_nuage<D>( load, nu ) ) nu.absent = true;
        return { nu };
    }
    return suite<D>( n, graine, wscale, cases );
}

template std::vector<Nuage<2>> Args::nuages<2>() const;
template std::vector<Nuage<3>> Args::nuages<3>() const;

} // namespace sf
