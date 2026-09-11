#pragma once

#include "util/common.h"
#include <cmath>
#include <cstring>

namespace pd {

/// LA CELLULE 3D : un polyedre convexe decrit par ses SOMMETS et ses ARETES, en tampons fixes.
///
/// = Pourquoi ce n'est pas la cellule 2D avec un axe de plus
///
/// En 2D les sommets EN ORDRE CYCLIQUE sont deja toute la geometrie : l'aire se lit par le lacet,
/// les coupes se retrouvent par l'invariant « la coupe `i` porte l'arete `[ v_i, v_i+1 ]` ». Rien
/// de tout cela ne survit en 3D -- il n'y a plus d'ordre cyclique global, et une coupe porte une
/// FACE, c'est-a-dire un cycle d'aretes qu'il faut savoir retrouver. La description change donc de
/// nature, exactement comme dans `sdot` ou `cut` et `measure` ont deux versions selon le regime de
/// dimension.
///
/// = Ce qui est stocke, et ce qui ne l'est pas
///
/// Un sommet porte SES TROIS COUPES (`vc`), une arete porte ses deux sommets. C'est tout : ni les
/// faces, ni les equations des plans.
///
///   * les FACES ne sont pas stockees parce qu'elles se relisent : la face de la coupe `c` est
///     l'ensemble des sommets dont `vc` contient `c`, et ses aretes celles dont les DEUX bouts la
///     contiennent. Les stocker obligerait a les maintenir a chaque coupe, alors qu'on n'en a
///     besoin qu'a la fin, une fois.
///   * les EQUATIONS DES PLANS non plus. Le volume se calcule par des tetraedres sur un point
///     interieur, ce qui ne demande que des sommets ; et l'identite du voisin, seule chose dont
///     l'appelant ait besoin, est l'indice de coupe lui-meme.
///
/// = L'hypothese : polytope SIMPLE
///
/// Chaque sommet est sur EXACTEMENT trois plans. C'est vrai en position generale, et c'est ce qui
/// rend la coupe purement combinatoire : deux nouveaux sommets de la face creee sont voisins
/// exactement quand ils partagent une ANCIENNE coupe. Un nuage degenere (germes cocycliques,
/// grille parfaite) sort de cette hypothese -- c'est la meme classe de fragilite que la tangence
/// deja rencontree trois fois dans ce banc, et elle se traite pareil, par une tolerance.
///
/// `MaxNv` est la borne sur les SOMMETS. Une cellule de Voronoi 3D poissonienne en a 27 en moyenne
/// (contre 6 en 2D) pour 40 aretes et 15 faces : la queue commence beaucoup plus haut qu'en 2D, et
/// `--maxnv 64` est ici le defaut raisonnable la ou 32 suffit en 2D.
template<int MaxNv>
struct Cell3T {
    static constexpr int dim = 3;
    static constexpr int max_nb_vertices = MaxNv;
    /// EULER : un polytope simple a `E = 3 V / 2` aretes exactement. La borne n'est donc pas un
    /// reglage, c'est une identite -- plus deux de marge pour les etats intermediaires.
    static constexpr int max_nb_edges = 3 * MaxNv / 2 + 2;
    /// ... et donc `F = 2 + V / 2` faces, par `V - E + F = 2`. Meme remarque : ce n'est pas un
    /// reglage, et les deux bornes tombent ensemble si `MaxNv` change.
    static constexpr int max_nb_faces = MaxNv / 2 + 2;

    SI nb = 0;                                  ///< sommets
    SI ne = 0;                                  ///< aretes
    alignas( 64 ) TF vx[ max_nb_vertices ];
    alignas( 64 ) TF vy[ max_nb_vertices ];
    alignas( 64 ) TF vz[ max_nb_vertices ];
    alignas( 64 ) SI vc[ max_nb_vertices ][ 3 ];///< les TROIS coupes du sommet, triees croissant
    alignas( 64 ) SI ea[ max_nb_edges ];
    alignas( 64 ) SI eb[ max_nb_edges ];

    Vec<3> vertex( SI i ) const { return { vx[ i ], vy[ i ], vz[ i ] }; }

    /// le cube unite. Les six faces du domaine portent les indices `-1 .. -6` : negatifs, donc
    /// distinguables d'un germe, et DISTINCTS entre eux, sans quoi deux faces opposees seraient la
    /// meme et le parcours des cycles partirait en vrille.
    void init_as_unit_cube() {
        nb = 8;
        ne = 12;
        for ( SI b = 0; b < 8; ++b ) {
            const SI i = b & 1, j = ( b >> 1 ) & 1, k = ( b >> 2 ) & 1;
            vx[ b ] = TF( i ); vy[ b ] = TF( j ); vz[ b ] = TF( k );
            vc[ b ][ 0 ] = i ? -2 : -1;
            vc[ b ][ 1 ] = j ? -4 : -3;
            vc[ b ][ 2 ] = k ? -6 : -5;
            sort3( vc[ b ] );
        }
        SI e = 0;
        for ( SI b = 0; b < 8; ++b )
            for ( int d = 0; d < 3; ++d ) {
                const SI o = b ^ ( 1 << d );
                if ( o > b ) { ea[ e ] = b; eb[ e ] = o; ++e; }
            }
    }

    /// LA COUPE par `d . x <= off`.
    ///
    /// Trois passes, et aucune n'ecrit avant que la taille finale soit connue : sur debordement la
    /// cellule doit rester INTACTE, comme en 2D -- l'appelant compte et signale, il ne recolte pas
    /// un polyedre a moitie reecrit.
    CutResult cut( Vec<3> d, TF off, SI cut_id ) {
        TF s[ max_nb_vertices ];
        SI nb_out = 0;
        for ( SI i = 0; i < nb; ++i ) {
            s[ i ] = d[ 0 ] * vx[ i ] + d[ 1 ] * vy[ i ] + d[ 2 ] * vz[ i ] - off;
            nb_out += s[ i ] > 0;
        }
        if ( nb_out == 0 )
            return CutResult::unchanged;
        if ( nb_out == nb ) {
            nb = 0; ne = 0;
            return CutResult::empty;
        }

        // ---- les sommets GARDES, renumerotes. `map < 0` dit « coupe ».
        SI map[ max_nb_vertices ];
        SI nn = 0;
        for ( SI i = 0; i < nb; ++i )
            map[ i ] = s[ i ] > 0 ? -1 : nn++;

        // ---- les sommets NEUFS, un par arete traversante. Ils sont exactement les sommets de la
        // face creee, donc leur nombre est celui de ses cotes.
        TF nx[ max_nb_vertices ], ny[ max_nb_vertices ], nz[ max_nb_vertices ];
        SI ncut[ max_nb_vertices ][ 3 ];
        SI from[ max_nb_edges ];                        // le sommet DEDANS de l'arete traversante
        SI which[ max_nb_edges ];                       // ... et l'indice du sommet neuf
        SI nm = 0;
        for ( SI e = 0; e < ne; ++e ) {
            const SI a = ea[ e ], b = eb[ e ];
            const bool oa = s[ a ] > 0, ob = s[ b ] > 0;
            which[ e ] = -1;
            if ( oa == ob )
                continue;
            if ( nn + nm >= max_nb_vertices )
                return CutResult::overflow;
            const SI in = oa ? b : a, out = oa ? a : b;

            // ancre sur le sommet DEDANS, comme en 2D et pour la meme raison : avec `s_in == 0` la
            // forme symetrique ne rend pas `v_in` en flottant, et le sommet passe alors DE L'AUTRE
            // COTE du plan -- ce que toute la suite suppose impossible.
            const TF t = s[ in ] / ( s[ in ] - s[ out ] );
            nx[ nm ] = vx[ in ] + ( vx[ out ] - vx[ in ] ) * t;
            ny[ nm ] = vy[ in ] + ( vy[ out ] - vy[ in ] ) * t;
            nz[ nm ] = vz[ in ] + ( vz[ out ] - vz[ in ] ) * t;

            // les deux coupes COMMUNES aux deux bouts sont celles qui portent l'arete ; le sommet
            // neuf est sur elles deux et sur la nouvelle.
            SI c0, c1;
            shared2( vc[ a ], vc[ b ], c0, c1 );
            ncut[ nm ][ 0 ] = c0; ncut[ nm ][ 1 ] = c1; ncut[ nm ][ 2 ] = cut_id;
            sort3( ncut[ nm ] );

            from[ e ] = in;
            which[ e ] = nm;
            ++nm;
        }

        const SI new_nb = nn + nm;
        if ( new_nb > max_nb_vertices )
            return CutResult::overflow;

        // ---- LES ARETES. Trois familles, et la troisieme est la seule qui demande a reflechir.
        SI na = 0;
        SI ta[ max_nb_edges ], tb[ max_nb_edges ];
        auto push = [ & ]( SI a, SI b ) {
            if ( na < max_nb_edges ) { ta[ na ] = a; tb[ na ] = b; }
            ++na;
        };
        for ( SI e = 0; e < ne; ++e ) {
            const SI a = ea[ e ], b = eb[ e ];
            if ( which[ e ] >= 0 )
                push( map[ from[ e ] ], nn + which[ e ] );      // arete TRAVERSANTE, raccourcie
            else if ( map[ a ] >= 0 )
                push( map[ a ], map[ b ] );                     // arete entierement DEDANS
        }
        // les cotes de la FACE NEUVE : deux sommets neufs sont voisins exactement quand ils sont
        // sur une meme ANCIENNE coupe -- l'arete qui les joint est alors l'intersection de cette
        // coupe-la avec la nouvelle. Le convexe garantit qu'une ancienne coupe ne peut appareiller
        // qu'une seule paire : sinon sa trace sur la face neuve aurait deux morceaux.
        for ( SI i = 0; i < nm; ++i )
            for ( SI j = i + 1; j < nm; ++j )
                if ( share_one_but( ncut[ i ], ncut[ j ], cut_id ) )
                    push( nn + i, nn + j );
        if ( na > max_nb_edges )
            return CutResult::overflow;

        // ---- COMMIT. Rien n'a bouge jusqu'ici.
        // EN MONTANT, et pas en descendant : la compaction envoie toujours le sommet `i` vers un
        // indice `map[ i ] <= i`, donc l'ecriture reste DERRIERE la lecture. En descendant elle
        // passerait devant et ecraserait un sommet pas encore recopie -- ce qui donnait un polyedre
        // aux sommets melanges, de bonne topologie et de mauvaise geometrie : la somme des volumes
        // valait 0.33 sans qu'un seul compteur ne s'en plaigne.
        for ( SI i = 0; i < nb; ++i ) {
            const SI m = map[ i ];
            if ( m < 0 ) continue;
            vx[ m ] = vx[ i ]; vy[ m ] = vy[ i ]; vz[ m ] = vz[ i ];
            vc[ m ][ 0 ] = vc[ i ][ 0 ]; vc[ m ][ 1 ] = vc[ i ][ 1 ]; vc[ m ][ 2 ] = vc[ i ][ 2 ];
        }
        for ( SI i = 0; i < nm; ++i ) {
            vx[ nn + i ] = nx[ i ]; vy[ nn + i ] = ny[ i ]; vz[ nn + i ] = nz[ i ];
            vc[ nn + i ][ 0 ] = ncut[ i ][ 0 ]; vc[ nn + i ][ 1 ] = ncut[ i ][ 1 ]; vc[ nn + i ][ 2 ] = ncut[ i ][ 2 ];
        }
        for ( SI e = 0; e < na; ++e ) { ea[ e ] = ta[ e ]; eb[ e ] = tb[ e ]; }
        nb = new_nb;
        ne = na;
        return CutResult::done;
    }

    /// LES FACES SANS LEUR ORDRE : ce que les MESURES demandent vraiment.
    ///
    /// `for_each_face` ci-dessous rend des CYCLES, et les rendre coute cher : pour chaque face il
    /// faut balayer tous les sommets, puis toutes les aretes, pour savoir lesquels sont dessus.
    /// C'est `O( F ( V + E ) )` -- sur une cellule de Voronoi poissonienne 3D, 17 faces pour 27
    /// sommets et 40 aretes, soit quinze balayages complets de la cellule. Ablation mesuree :
    /// `measure()` pesait 22 % du temps du diagramme 3D.
    ///
    /// Or ni le volume ni l'aire d'une face n'ont besoin de l'ORDRE. Il suffit, par face, d'UN de
    /// ses sommets `v_f` -- le sommet d'eventail, le premier rencontre -- et de la somme `S_f` des
    /// produits vectoriels de ses aretes vues depuis lui :
    ///
    ///   * l'AIRE vaut `|S_f| / 2`. La face est plane et convexe, donc les triangles ( v_f, arete )
    ///     la pavent, et leurs produits vectoriels sont TOUS PARALLELES.
    ///   * le VOLUME vaut `sum_f |( v_f - g ) . S_f| / 6` pour n'importe quel `g` interieur --
    ///     `v_f` est sur le plan de la face, c'est tout ce que la formule lui demande. Chaque terme
    ///     est `2 * aire * hauteur`, et la valeur absolue dispense d'orienter les faces : c'est
    ///     exactement la decomposition en tetraedres d'avant, sommee une fois par FACE au lieu
    ///     d'une fois par triangle.
    ///
    /// Et tout cela s'accumule en DEUX passes sans jamais chercher : une sur les sommets, qui
    /// portent chacun leurs trois faces ; une sur les aretes, qui sont chacune sur exactement deux
    /// faces -- les deux coupes communes a leurs bouts. `O( V + E )` au lieu de `O( F ( V + E ) )`.
    ///
    /// Un seul point delicat : l'ordre `( a, b )` d'une arete est arbitraire, donc son produit
    /// vectoriel peut sortir dans un sens ou dans l'autre. On le recale sur la somme deja
    /// accumulee, a laquelle il est parallele -- un signe de produit scalaire tranche.
    ///
    /// Effet de bord qui compte : le parcours de cycle pouvait ECHOUER (degre != 2 sur une
    /// degenerescence) et la face etait alors silencieusement omise. Ici il n'y a pas de parcours,
    /// donc pas d'echec possible.
    struct Faces {
        SI nf = 0;
        SI cut[ max_nb_faces ];                                         ///< la coupe qui la porte
        SI v0[ max_nb_faces ];                                          ///< son SOMMET D'EVENTAIL
        TF sx[ max_nb_faces ], sy[ max_nb_faces ], sz[ max_nb_faces ];  ///< 2 x le vecteur-aire
        TF cgx, cgy, cgz;                                               ///< un point INTERIEUR

        TF area( SI k ) const {
            return TF( 0.5 ) * std::sqrt( sx[ k ] * sx[ k ] + sy[ k ] * sy[ k ] + sz[ k ] * sz[ k ] );
        }
    };

    /// LA TABLE qui donne le numero de face d'une coupe. Il y a `3 V` demandes par cellule -- 81
    /// sur une cellule moyenne -- pour une quinzaine de faces, et la recherche lineaire qu'il y
    /// avait ici pesait un tiers de `gather_faces` : sept comparaisons en moyenne, et surtout une
    /// sortie de boucle imprevisible a chaque fois.
    ///
    /// Deux fois la borne d'Euler, arrondi a la puissance de deux superieure. Le remplissage n'est
    /// pas ce qui decide -- avec quinze faces typiques la table est vide a 94 % dans les deux cas --
    /// mais 128 mesurait 2.8 % de plus que 256 sur le cas uniforme, et le `memset` de 1 Ko ne se
    /// voit pas.
    static constexpr unsigned ht_size = 256;
    static_assert( ht_size > unsigned( max_nb_faces ), "la table doit pouvoir loger toutes les faces" );

    void gather_faces( Faces &f ) const {
        SI slot[ max_nb_vertices ][ 3 ];    // la face de chaque (sommet, coupe), retrouvee une fois
        SI ht[ ht_size ];
        std::memset( ht, -1, sizeof ht );   // -1 : case libre
        f.nf = 0;

        // ---- PASSE 1 : numeroter les faces. Aucun flottant ici -- le sommet d'eventail est le
        // PREMIER rencontre, et il ne coute qu'une affectation.
        for ( SI i = 0; i < nb; ++i )
            for ( int r = 0; r < 3; ++r ) {
                const SI c = vc[ i ][ r ];
                unsigned h = ( unsigned( c ) * 2654435761u ) & ( ht_size - 1 );
                while ( ht[ h ] >= 0 && f.cut[ ht[ h ] ] != c )
                    h = ( h + 1 ) & ( ht_size - 1 );
                // ECRIT AINSI, et pas `SI k = ht[ h ]; if ( k < 0 )`, qui dit pourtant la meme
                // chose : la forme ci-dessous mesure 3.7 % de MOINS sur le cas uniforme. Elle
                // laisse le compilateur sortir la case libre par un `cmov` et garder le corps
                // chaud sans branchement. Verifie autrement : ni le systeme de compilation
                // (xmake ou `g++` a la main donnent 5.51 s tous les deux) ni l'ordre des
                // declarations sur la pile n'y changent quoi que ce soit -- c'est bien cette
                // ligne-ci. Meme lecon que pour la grille : sur ce banc, le compilateur decide
                // parfois plus que l'algorithme, et seule la mesure appariee le dit.
                //
                // Et il faut dire l'autre moitie de la mesure : la MEME modification fait perdre
                // 1.9 % a `pd_newton` (20.36 -> 20.75 s). C'est donc bien un effet de placement de
                // code, propre a chaque binaire, et pas une amelioration de l'algorithme. La forme
                // ci-dessous est gardee parce qu'elle gagne sur le banc de diagramme, qui est ce
                // que ce fichier sert a mesurer -- pas parce qu'elle serait « meilleure ».
                SI k = ht[ h ] >= 0 ? ht[ h ] : f.nf;
                if ( k == f.nf ) {
                    // `nf <= 2 + nb / 2` par Euler tant que la cellule est un polytope simple ;
                    // la borne n'est la que pour qu'une degenerescence ne deborde pas le tampon.
                    // Elle passe AVANT l'ecriture dans la table : un numero hors borne y resterait
                    // et la sondation suivante irait lire `cut` en dehors.
                    if ( k >= max_nb_faces )
                        continue;
                    ht[ h ] = k;
                    ++f.nf;
                    f.cut[ k ] = c;
                    f.v0[ k ] = i;
                    f.sx[ k ] = f.sy[ k ] = f.sz[ k ] = 0;
                }
                slot[ i ][ r ] = k;
            }

        // le point interieur : le centre des sommets. Une boucle a lui seul, contigue et
        // vectorisable, plutot que trois additions dispersees dans la precedente.
        TF gx = 0, gy = 0, gz = 0;
        for ( SI i = 0; i < nb; ++i ) { gx += vx[ i ]; gy += vy[ i ]; gz += vz[ i ]; }
        f.cgx = gx / nb; f.cgy = gy / nb; f.cgz = gz / nb;

        // ---- PASSE 2 : les aretes. Chacune est sur exactement deux faces -- les deux coupes
        // communes a ses bouts -- et donne dans chacune le triangle ( v0, a, b ). Les deux aretes
        // qui TOUCHENT `v0` sont sautees : leur triangle est plat, et c'est autant de produits
        // vectoriels en moins (deux par face, soit un tiers d'entre eux).
        for ( SI e = 0; e < ne; ++e ) {
            const SI a = ea[ e ], b = eb[ e ];
            for ( int r = 0; r < 3; ++r ) {
                const SI c = vc[ a ][ r ];
                if ( c != vc[ b ][ 0 ] && c != vc[ b ][ 1 ] && c != vc[ b ][ 2 ] )
                    continue;
                const SI k = slot[ a ][ r ], o = f.v0[ k ];
                if ( o == a || o == b )
                    continue;
                const TF ux = vx[ a ] - vx[ o ], uy = vy[ a ] - vy[ o ], uz = vz[ a ] - vz[ o ];
                const TF wx = vx[ b ] - vx[ o ], wy = vy[ b ] - vy[ o ], wz = vz[ b ] - vz[ o ];
                TF px = uy * wz - uz * wy;
                TF py = uz * wx - ux * wz;
                TF pz = ux * wy - uy * wx;
                // l'ordre ( a, b ) d'une arete est arbitraire : on recale sur ce qui est deja la.
                // La premiere arete trouve une somme nulle, donc se garde telle quelle.
                if ( f.sx[ k ] * px + f.sy[ k ] * py + f.sz[ k ] * pz < 0 ) {
                    px = -px; py = -py; pz = -pz;
                }
                f.sx[ k ] += px; f.sy[ k ] += py; f.sz[ k ] += pz;
            }
        }
    }

    /// LE VOLUME ET LES FACETTES D'UN SEUL COUP : c'est ce que Newton demande a chaque cellule, et
    /// les deux sortent de la meme accumulation. La porte existe aussi en 2D, ou elle ne mutualise
    /// rien -- l'appelant n'a pas a savoir dans quelle dimension il est.
    template<class F>
    TF measure_and_facets( F &&f ) const {
        if ( nb == 0 )
            return 0;
        Faces fa;
        gather_faces( fa );
        TF vol = 0;
        for ( SI k = 0; k < fa.nf; ++k ) {
            const SI o = fa.v0[ k ];
            const TF d = ( vx[ o ] - fa.cgx ) * fa.sx[ k ]
                       + ( vy[ o ] - fa.cgy ) * fa.sy[ k ]
                       + ( vz[ o ] - fa.cgz ) * fa.sz[ k ];
            vol += d < 0 ? -d : d;
            if ( fa.cut[ k ] >= 0 )
                f( fa.cut[ k ], fa.area( k ) );
        }
        return vol / 6;
    }

    /// LE VOLUME seul. La lambda vide emporte avec elle la racine de `area` -- sous
    /// `-fno-math-errno` elle est pure, donc eliminable.
    TF measure() const {
        return measure_and_facets( []( SI, TF ) {} );
    }

    /// LES FACETTES : `f( coupe, aire )` pour chaque face portee par un VOISIN. Meme contrat qu'en
    /// 2D, ou la mesure d'une facette est une longueur.
    template<class F>
    void for_each_facet( F &&f ) const {
        measure_and_facets( f );
    }

    /// LES FACES, une par coupe encore presente, chacune rendue comme un CYCLE de sommets.
    ///
    /// `f( cut_id, cyc, m )` -- `cyc[ 0 .. m )` sont les sommets de la face, dans l'ordre du bord.
    /// C'est la seule porte par laquelle on lit la topologie : `measure` s'en sert pour le volume,
    /// Newton pour les aires de faces dont il fait ses coefficients de Laplacien.
    template<class F>
    void for_each_face( F &&f ) const {
        SI seen[ 3 * max_nb_vertices ];
        SI ns = 0;
        for ( SI i = 0; i < nb; ++i )
            for ( int r = 0; r < 3; ++r ) {
                const SI c = vc[ i ][ r ];
                bool got = false;
                for ( SI u = 0; u < ns; ++u ) if ( seen[ u ] == c ) { got = true; break; }
                if ( ! got ) seen[ ns++ ] = c;
            }

        for ( SI u = 0; u < ns; ++u ) {
            const SI c = seen[ u ];

            // les deux voisins de chaque sommet SUR CETTE FACE : une face d'un polytope simple est
            // un polygone, donc chaque sommet y a exactement deux voisins. On les remplit en un
            // passage sur les aretes, sans jamais chercher.
            SI nb1[ max_nb_vertices ], nb2[ max_nb_vertices ], deg[ max_nb_vertices ];
            SI cyc[ max_nb_vertices ], m = 0;
            for ( SI i = 0; i < nb; ++i )
                if ( on( i, c ) ) { deg[ i ] = 0; cyc[ m++ ] = i; }
            if ( m < 3 )
                continue;
            for ( SI e = 0; e < ne; ++e ) {
                const SI a = ea[ e ], b = eb[ e ];
                if ( ! on( a, c ) || ! on( b, c ) )
                    continue;
                if ( deg[ a ] == 0 ) nb1[ a ] = b; else nb2[ a ] = b;
                ++deg[ a ];
                if ( deg[ b ] == 0 ) nb1[ b ] = a; else nb2[ b ] = a;
                ++deg[ b ];
            }

            SI ord[ max_nb_vertices ];
            SI prev = -1, cur = cyc[ 0 ], k = 0;
            bool ok = true;
            for ( ; k < m; ++k ) {
                ord[ k ] = cur;
                if ( deg[ cur ] < 2 ) { ok = false; break; }
                const SI nx = nb1[ cur ] == prev ? nb2[ cur ] : nb1[ cur ];
                prev = cur;
                cur = nx;
            }
            if ( ok && cur == ord[ 0 ] )
                f( c, ord, m );
        }
    }

    /// LE VOLUME PAR LES CYCLES : l'implementation d'origine, par des tetraedres sur le CENTRE DE
    /// GRAVITE des sommets -- qui est interieur, la cellule etant convexe. Chaque face est
    /// eventaillee depuis son PREMIER SOMMET ; les tetraedres pavent alors la cellule sans se
    /// recouvrir, donc la somme des `|det| / 6` est le volume, sans avoir a orienter quoi que ce
    /// soit.
    ///
    /// Elle n'est plus dans le chemin chaud -- `measure()` passe par `gather_faces` -- mais elle
    /// reste : c'est un SECOND CHEMIN, qui ne partage avec le premier ni la decomposition en
    /// triangles ni meme la facon de retrouver les faces. `pd_check` compare les deux cellule par
    /// cellule, et c'est le seul temoin qu'on ait en 3D, ou aucune reference exterieure ne donne
    /// le volume d'un polyedre.
    TF measure_by_cycles() const {
        if ( nb == 0 )
            return 0;
        TF gx = 0, gy = 0, gz = 0;
        for ( SI i = 0; i < nb; ++i ) { gx += vx[ i ]; gy += vy[ i ]; gz += vz[ i ]; }
        gx /= nb; gy /= nb; gz /= nb;

        TF vol = 0;
        for_each_face( [ & ]( SI, const SI *ord, SI m ) {
            const SI o = ord[ 0 ];
            const TF ax = vx[ o ] - gx, ay = vy[ o ] - gy, az = vz[ o ] - gz;
            for ( SI t = 1; t + 1 < m; ++t ) {
                const SI p = ord[ t ], q = ord[ t + 1 ];
                const TF bx = vx[ p ] - gx, by = vy[ p ] - gy, bz = vz[ p ] - gz;
                const TF cx = vx[ q ] - gx, cy = vy[ q ] - gy, cz = vz[ q ] - gz;
                const TF det = ax * ( by * cz - bz * cy )
                             - ay * ( bx * cz - bz * cx )
                             + az * ( bx * cy - by * cx );
                vol += det < 0 ? -det : det;
            }
        } );
        return vol / 6;
    }

    /// L'AIRE d'une face, dans le meme esprit : eventail depuis son premier sommet, demi-norme du
    /// produit vectoriel. Elle est PLANE, donc l'eventail est exact et l'orientation sans objet.
    TF face_measure( const SI *ord, SI m ) const {
        TF sx = 0, sy = 0, sz = 0;
        const SI o = ord[ 0 ];
        for ( SI t = 1; t + 1 < m; ++t ) {
            const SI p = ord[ t ], q = ord[ t + 1 ];
            const TF ux = vx[ p ] - vx[ o ], uy = vy[ p ] - vy[ o ], uz = vz[ p ] - vz[ o ];
            const TF wx = vx[ q ] - vx[ o ], wy = vy[ q ] - vy[ o ], wz = vz[ q ] - vz[ o ];
            sx += uy * wz - uz * wy;
            sy += uz * wx - ux * wz;
            sz += ux * wy - uy * wx;
        }
        return TF( 0.5 ) * std::sqrt( sx * sx + sy * sy + sz * sz );
    }

    void bounds( TF *lo, TF *hi ) const {
        if ( nb == 0 ) {
            for ( int d = 0; d < 3; ++d ) { lo[ d ] = 0; hi[ d ] = 0; }
            return;
        }
        lo[ 0 ] = hi[ 0 ] = vx[ 0 ]; lo[ 1 ] = hi[ 1 ] = vy[ 0 ]; lo[ 2 ] = hi[ 2 ] = vz[ 0 ];
        for ( SI i = 1; i < nb; ++i ) {
            lo[ 0 ] = vx[ i ] < lo[ 0 ] ? vx[ i ] : lo[ 0 ];  hi[ 0 ] = vx[ i ] > hi[ 0 ] ? vx[ i ] : hi[ 0 ];
            lo[ 1 ] = vy[ i ] < lo[ 1 ] ? vy[ i ] : lo[ 1 ];  hi[ 1 ] = vy[ i ] > hi[ 1 ] ? vy[ i ] : hi[ 1 ];
            lo[ 2 ] = vz[ i ] < lo[ 2 ] ? vz[ i ] : lo[ 2 ];  hi[ 2 ] = vz[ i ] > hi[ 2 ] ? vz[ i ] : hi[ 2 ];
        }
    }

private:
    bool on( SI i, SI c ) const { return vc[ i ][ 0 ] == c || vc[ i ][ 1 ] == c || vc[ i ][ 2 ] == c; }

    static void sort3( SI *a ) {
        if ( a[ 0 ] > a[ 1 ] ) { const SI t = a[ 0 ]; a[ 0 ] = a[ 1 ]; a[ 1 ] = t; }
        if ( a[ 1 ] > a[ 2 ] ) { const SI t = a[ 1 ]; a[ 1 ] = a[ 2 ]; a[ 2 ] = t; }
        if ( a[ 0 ] > a[ 1 ] ) { const SI t = a[ 0 ]; a[ 0 ] = a[ 1 ]; a[ 1 ] = t; }
    }

    /// les deux coupes communes a deux sommets voisins. Elles existent et sont exactement deux :
    /// une arete d'un polytope simple est l'intersection de deux plans.
    static void shared2( const SI *a, const SI *b, SI &c0, SI &c1 ) {
        c0 = c1 = 0;
        int k = 0;
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                if ( a[ i ] == b[ j ] ) { ( k++ == 0 ? c0 : c1 ) = a[ i ]; break; }
    }

    static bool share_one_but( const SI *a, const SI *b, SI skip ) {
        for ( int i = 0; i < 3; ++i ) {
            if ( a[ i ] == skip ) continue;
            for ( int j = 0; j < 3; ++j )
                if ( a[ i ] == b[ j ] ) return true;
        }
        return false;
    }
};

using Cell3 = Cell3T<64>;

} // namespace pd
