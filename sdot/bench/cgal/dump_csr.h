#pragma once

#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

/// LE VIDAGE DE LA CONNECTIVITE FINALE.
///
/// Le banc GPU doit rejouer EXACTEMENT ce que rejoue le banc CPU : les memes germes, les memes
/// voisins, dans le meme ordre. Le refaire de son cote voudrait dire y remettre CGAL, donc une
/// dependance de plus et surtout un risque de divergence silencieuse. On ecrit donc la liste une
/// fois, hors chronometre, et les deux bancs la lisent.
///
/// Le fichier porte AUSSI le volume de reference calcule ici en FP64 par `Cell3T<128>` : c'est le
/// seul temoin qui permette au banc GPU de dire si son FP32 rend la meme geometrie, cellule par
/// cellule, et pas seulement en somme -- une somme peut etre juste avec des cellules fausses qui
/// se compensent, on l'a deja vu sur ce banc.
///
///   magic "PDC1" | dim | n | nv | moi[ n * (dim+1) ] | off[ n+1 ] | vois[ nv * (dim+1) ] | ref[ n ]
///
/// Tout en FP64 sur le disque : la conversion en FP32 est le sujet de la mesure, pas du transport.
inline bool dump_csr( const std::string &path, int dim,
                      const std::vector<double> &moi, const std::vector<std::size_t> &off,
                      const std::vector<double> &vois, const std::vector<double> &ref ) {
    std::FILE *f = std::fopen( path.c_str(), "wb" );
    if ( ! f ) { std::fprintf( stderr, "dump_csr : %s illisible\n", path.c_str() ); return false; }
    const std::int32_t magic = 0x31434450;                  // "PDC1"
    const std::int32_t d = dim;
    const std::int64_t n = std::int64_t( off.size() ) - 1;
    const std::int64_t nv = std::int64_t( vois.size() ) / ( dim + 1 );
    std::vector<std::int64_t> o( off.begin(), off.end() );
    std::fwrite( &magic, 4, 1, f );
    std::fwrite( &d, 4, 1, f );
    std::fwrite( &n, 8, 1, f );
    std::fwrite( &nv, 8, 1, f );
    std::fwrite( moi.data(), 8, moi.size(), f );
    std::fwrite( o.data(), 8, o.size(), f );
    std::fwrite( vois.data(), 8, vois.size(), f );
    std::fwrite( ref.data(), 8, ref.size(), f );
    std::fclose( f );
    std::fprintf( stderr, "dump_csr : %s  dim %d  n %lld  voisins %lld\n",
                  path.c_str(), dim, ( long long ) n, ( long long ) nv );
    return true;
}
