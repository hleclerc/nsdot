#pragma once

// =====================================================================================
// LES NUAGES : le tirage uniforme, les fichiers de `cases/`, et LA SUITE -- les cas importants,
// dans l'ordre ou on veut les lire. Un nuage uniforme ne teste presque rien ; les cas qui font
// mal sont ceux ou les germes sont tres inegalement repartis et ou les poids EMMENENT la cellule
// loin de son germe.
// =====================================================================================

#include "util/common.h"
#include <string>
#include <utility>
#include <vector>

namespace sf {

template<int D>
struct Nuage {
    std::string     nom;
    SI              n = 0;
    std::vector<TF> c[ D ];
    std::vector<TF> w;
    const TF       *P[ D ] = {};
    const TF       *W = nullptr;    ///< `nullptr` si les poids sont TOUS NULS ( Voronoi )
    bool            absent = false; ///< le fichier n'existe pas : le cas est saute, pas echoue
    bool            temoin = false; ///< les poids du fichier RESOLVENT deja les masses egales

    /// `P` et `W` pointent DANS l'objet : une copie ou un deplacement doit les refaire.
    Nuage() = default;
    Nuage( const Nuage &o ) { reprend( o ); }
    Nuage( Nuage &&o ) noexcept { reprend( std::move( o ) ); }
    Nuage &operator=( const Nuage &o ) { reprend( o ); return *this; }
    Nuage &operator=( Nuage &&o ) noexcept { reprend( std::move( o ) ); return *this; }

    /// `W` ne pointe sur les poids que s'il y en a un non nul : des poids nuls SONT le cas
    /// euclidien, et les garder ferait payer un majorant identiquement nul ( 12 % mesure ).
    void finish() {
        n = SI( c[ 0 ].size() );
        for ( int d = 0; d < D; ++d ) P[ d ] = c[ d ].data();
        W = nullptr;
        for ( TF v : w )
            if ( v != TF( 0 ) ) { W = w.data(); break; }
    }

private:
    template<class N>
    void reprend( N &&o ) {
        nom = std::forward<N>( o ).nom;
        for ( int d = 0; d < D; ++d ) c[ d ] = std::forward<N>( o ).c[ d ];
        w = std::forward<N>( o ).w;
        absent = o.absent;
        temoin = o.temoin;
        finish();
    }
};

/// Le nuage UNIFORME dans le cube unite, borne loin des faces. `wscale` : des poids tires dans
/// `[ -wscale h^2, wscale h^2 ]`, `h = n^(-1/D)` -- un plan est decale de `dw / ( 2 |p1 - p0| )`,
/// donc il faut `dw ~ h^2` pour que le decalage soit une fraction de l'espacement.
template<int D> Nuage<D> nuage_uniforme( SI n, unsigned graine, double wscale );

/// Un fichier de `cases/` : des lignes `#`, puis `n`, puis `n` fois `x y [z] w`.
template<int D> bool charge_nuage( const std::string &chemin, Nuage<D> &nu, bool bavard = true );

/// Ecrire un nuage et ses poids au meme format, en `%.17g` -- la representation la plus courte qui
/// relit exactement le meme double.
template<int D> bool ecrit_nuage( const std::string &chemin, const Nuage<D> &nu, const TF *W,
                                  const std::string &entete );

/// LA SUITE. `cases` est le repertoire des nuages durs ; un fichier absent est signale, pas fatal.
///   2D : uniforme, lignes / Voronoi, lignes / aires egales ( ce dernier PORTE la solution ) ;
///   3D : uniforme, plans / Voronoi, plans / volumes egaux.
template<int D> std::vector<Nuage<D>> suite( SI n, unsigned graine, double wscale,
                                             const std::string &cases );

} // namespace sf
