#pragma once

// =====================================================================================
// LES FOURNISSEURS 3D -- pour l'instant, ceux qui n'elaguent rien.
//
// Ils ne regardent jamais la cellule : ils proposent TOUS les autres germes, et c'est le noyau qui
// decide de ce que chaque plan retranche. C'est le point de depart delibere -- on veut d'abord
// mesurer ce que coute une coupe, sans qu'une politique d'elagage vienne s'y melanger.
//
// L'ORDRE, LUI, N'EST PAS NEUTRE. Le nuage est trie en MORTON, et le parcours part du germe courant
// pour s'en ECARTER ( i+1, i-1, i+2, i-2... ), ce qui est approximativement du plus proche au plus
// lointain. C'est ce qui decide par quels etats intermediaires la cellule passe, donc combien de
// sommets elle porte au pire -- et en 3D ce nombre decide de la taille des tampons.
//
// DEUX CURSEURS ET PAS UNE BOUCLE DE REJET : la forme evidente monte l'ecart jusqu'a `2 n` et
// rejette la moitie des indices. Mesure faite en 2D sur exactement ce point : +135 %.
// =====================================================================================

#include "supercell/Contrat3D.h"

namespace noyau3d {

/// LE PLAN BISSECTEUR de `[ p0, pj ]`, oriente pour que `p0` soit DEDANS.
///
///     |x - p0|^2 - w0  <=  |x - pj|^2 - wj      <=>      d . x <= off
///     d = pj - p0 ,  off = ( |pj|^2 - |p0|^2 + w0 - wj ) / 2
///
/// La forme `d . ( pj + p0 ) / 2` est preferee a `( |pj|^2 - |p0|^2 ) / 2` : elle annule les termes
/// dominants quand les deux germes sont proches, ce qui est le cas usuel.
template<bool POIDS = false>
inline void bissect3( float xj, float yj, float zj, float wj,
                      float x0, float y0, float z0, float w0, int id, Plan3 &p ) {
    p.dx = xj - x0;
    p.dy = yj - y0;
    p.dz = zj - z0;
    p.off = 0.5f * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) + p.dz * ( zj + z0 ) );
    if constexpr ( POIDS ) p.off += 0.5f * ( w0 - wj );
    p.id = id;
}

/// TOUS LES AUTRES, DANS L'ORDRE DU TABLEAU. Le plus simple qu'on puisse ecrire, et le temoin
/// auquel on compare l'ordre sortant.
template<bool POIDS = false>
struct TousLesAutres3 {
    const float *px, *py, *pz, *pw;
    float x0, y0, z0, w0;
    int   n, k, moi;

    TousLesAutres3( const float *px, const float *py, const float *pz, const float *pw,
                    int n, int moi )
        : px( px ), py( py ), pz( pz ), pw( pw ),
          x0( px[ moi ] ), y0( py[ moi ] ), z0( pz[ moi ] ), w0( POIDS ? pw[ moi ] : 0.f ),
          n( n ), k( 0 ), moi( moi ) {}

    template<class Etat>
    bool suivant( const Etat &, RienDeLocal &, Plan3 &p ) {
        if ( k == moi ) ++k;
        if ( k >= n ) return false;
        const int j = k++;
        bissect3<POIDS>( px[j], py[j], pz[j], POIDS ? pw[j] : 0.f, x0, y0, z0, w0, j, p );
        return true;
    }
};

/// EN S'ECARTANT DE SOI. Sur un nuage trie en Morton, `i+1, i-1, i+2, i-2...` est approximativement
/// un parcours du plus proche au plus lointain -- sans jamais calculer une distance.
template<bool POIDS = false>
struct AutourDeMoi3 {
    const float *px, *py, *pz, *pw;
    float x0, y0, z0, w0;
    int   lo, hi, n;                                     ///< les deux curseurs, et la borne
    bool  cote;                                          ///< a qui le tour

    AutourDeMoi3( const float *px, const float *py, const float *pz, const float *pw,
                  int n, int moi )
        : px( px ), py( py ), pz( pz ), pw( pw ),
          x0( px[ moi ] ), y0( py[ moi ] ), z0( pz[ moi ] ), w0( POIDS ? pw[ moi ] : 0.f ),
          lo( moi - 1 ), hi( moi + 1 ), n( n ), cote( true ) {}

    template<class Etat>
    bool suivant( const Etat &, RienDeLocal &, Plan3 &p ) {
        int j;
        if      ( cote && hi < n ) { cote = false; j = hi++; }
        else if ( lo >= 0 )        { cote = true;  j = lo--; }
        else if ( hi < n )         {               j = hi++; }
        else return false;
        bissect3<POIDS>( px[j], py[j], pz[j], POIDS ? pw[j] : 0.f, x0, y0, z0, w0, j, p );
        return true;
    }
};

} // namespace noyau3d
