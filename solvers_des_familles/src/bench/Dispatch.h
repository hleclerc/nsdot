#pragma once

// =====================================================================================
// LE DISPATCH : les options qui doivent DISPARAITRE a la compilation -- le flottant du noyau, la
// taille de cellule -- deviennent des parametres de template ici, et nulle part ailleurs. Chaque
// `main` appelle `dispatch<D>( a, f )` et recoit le type de diagramme qui va avec ses options.
//
// = CHAQUE AXE MULTIPLIE TOUT CE QUI VIT DESSOUS
//
// C'est le fait central, et il se mesure. `main_image.cpp` -- quatre mille lignes templatees --
// s'instanciait QUARANTE-HUIT fois : deux noyaux x quatre capacites x deux flottants de mesure x
// trois solveurs lineaires. Huit minutes et quatre gigaoctets pour un seul fichier, au point que
// `-g` le faisait tuer par le plafond memoire du lanceur. Les trois quarts de ces instanciations
// ne servaient JAMAIS : le solveur est choisi a l'execution, et `--maxnv` garde son defaut.
//
// Deux axes ont donc ete supprimes, chacun par le moyen qui convenait :
//
//   * LE SOLVEUR LINEAIRE est devenu une INTERFACE VIRTUELLE ( `solver/Lineaire.h` ). Un appel
//     virtuel par resolution d'un systeme a `n` inconnues ne se mesure pas.
//   * LA CAPACITE DE CELLULE reste un parametre de template -- c'est la taille d'un tableau dans
//     la frame, on ne peut pas faire autrement -- mais on ne compile plus l'echelle entiere.
//     EN 2D UNE SEULE, 64 : une cellule de Laguerre plane a six voisins en moyenne et le banc
//     n'a jamais vu de debordement. EN 3D DEUX, 128 et 256 : le debordement arrive des
//     `n = 2e4` sur l'uniforme -- c'est le banc lui-meme qui repond « relancer avec --maxnv 256 »
//     -- donc retirer 256 rendrait son propre conseil inapplicable.
//     `-DSF_NV_TOUS` rend l'echelle complete, et demander une valeur non compilee le DIT au lieu
//     de retomber en silence sur une cellule plus petite.
//
// Restent le flottant du noyau ( `--kernel` ) et, dans `image`, celui de la mesure ( `--acc` ) :
// tous deux sont l'objet meme de l'etude de la simple precision, et les deux doivent vivre dans
// LE MEME binaire pour qu'on puisse les comparer sur les memes cellules. `-DSF_TK_UN` n'en garde
// qu'un pour qui n'en a pas besoin.
// =====================================================================================

#include "bench/Args.h"
#include "diagram/PowerDiagram.h"
#include <cstdio>
#include <type_traits>

namespace sf {

template<int D, class F>
int dispatch( const Args &a, F &&f ) {
    const int nv = a.nv( D );
    auto avec = [ & ]( auto tk ) {
        using TK = typename decltype( tk )::type;
#ifdef SF_NV_TOUS
        if constexpr ( D == 2 ) {
            if ( nv > 256 ) return f( std::type_identity<PowerDiagram<2,TK,512>>{} );
            if ( nv > 128 ) return f( std::type_identity<PowerDiagram<2,TK,256>>{} );
            if ( nv > 64 )  return f( std::type_identity<PowerDiagram<2,TK,128>>{} );
            return f( std::type_identity<PowerDiagram<2,TK,64>>{} );
        } else {
            if ( nv > 128 ) return f( std::type_identity<PowerDiagram<3,TK,256>>{} );
            return f( std::type_identity<PowerDiagram<3,TK,128>>{} );
        }
#else
        if constexpr ( D == 3 ) {
            if ( nv > 256 )
                std::printf( "  ATTENTION : --maxnv %d demande, 256 compile. Recompiler avec -DSF_NV_TOUS\n", nv );
            if ( nv > 128 ) return f( std::type_identity<PowerDiagram<3,TK,256>>{} );
            return f( std::type_identity<PowerDiagram<3,TK,128>>{} );
        } else {
            if ( nv != 64 )
                std::printf( "  ATTENTION : --maxnv %d demande, 64 compile. Recompiler avec -DSF_NV_TOUS\n", nv );
            return f( std::type_identity<PowerDiagram<2,TK,64>>{} );
        }
#endif
    };
#ifndef SF_TK_UN
    if ( a.kernel == "float" )
        return avec( std::type_identity<float>{} );
#else
    if ( a.kernel == "float" )
        std::printf( "  ATTENTION : --kernel float demande, seul `double` est compile ( -DSF_TK_UN )\n" );
#endif
    return avec( std::type_identity<double>{} );
}

} // namespace sf
