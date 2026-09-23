#pragma once

// =====================================================================================
// LE DISPATCH : les options qui doivent DISPARAITRE a la compilation -- le flottant du noyau, la
// taille de cellule -- deviennent des parametres de template ici, et nulle part ailleurs. Chaque
// `main` appelle `dispatch<D>( a, f )` et recoit le type de diagramme qui va avec ses options.
// =====================================================================================

#include "bench/Args.h"
#include "diagram/PowerDiagram.h"
#include <type_traits>

namespace sf {

template<int D, class F>
int dispatch( const Args &a, F &&f ) {
    const int nv = a.nv( D );
    auto avec = [ & ]( auto tk ) {
        using TK = typename decltype( tk )::type;
        if constexpr ( D == 2 ) {
            if ( nv > 256 ) return f( std::type_identity<PowerDiagram<2,TK,512>>{} );
            if ( nv > 128 ) return f( std::type_identity<PowerDiagram<2,TK,256>>{} );
            if ( nv > 64 ) return f( std::type_identity<PowerDiagram<2,TK,128>>{} );
            return f( std::type_identity<PowerDiagram<2,TK,64>>{} );
        } else {
            if ( nv > 128 ) return f( std::type_identity<PowerDiagram<3,TK,256>>{} );
            return f( std::type_identity<PowerDiagram<3,TK,128>>{} );
        }
    };
    if ( a.kernel == "float" )
        return avec( std::type_identity<float>{} );
    return avec( std::type_identity<double>{} );
}

} // namespace sf
