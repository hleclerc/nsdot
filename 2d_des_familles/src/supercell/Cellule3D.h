#pragma once

// =====================================================================================
// LA CELLULE 3D : un polyedre convexe qui se coupe lui-meme.
//
// La description et ses raisons sont dans `Contrat3D.h` ; ce fichier contient la STRUCTURE et la
// COUPE, donc du code machine, donc il est separe.
//
// LA CONNECTIVITE EST PORTEE PAR LES SOMMETS : chacun nomme ses trois voisins. Il y a eu une liste
// d'aretes ici, elle a ete mesuree contre celle-ci et elle a perdu -- le tableau est plus bas, et
// tout ce qui suit en depend.
//
// = L'INVARIANT QUI FAIT MARCHER TOUT LE RESTE
//
// `vnj[ i ]` est le voisin de `i` DE L'AUTRE COTE de l'arete portee par les DEUX AUTRES coupes que
// `vkj[ i ]` -- le voisin `j` est « en face » de la coupe `j ». Un sommet d'un polytope simple a
// trois coupes, donc trois paires de coupes, donc trois aretes, et cette mise en correspondance
// est une bijection.
//
// Elle vaut son prix parce qu'elle rend GRATUIT ce que la liste d'aretes faisait chercher : les
// deux faces qui portent l'arete `j` sont `vk` prive de `vkj`, donc `communes2` -- neuf
// comparaisons par arete -- disparait de `volume()`, et le numero de fente d'une arete se deduit
// de ses coupes sans rien parcourir.
//
// = CE QUE CELA CHANGE DANS LA COUPE, ET C'EST LA LE VRAI POINT
//
// Avec la liste d'aretes il faut lire les DEUX BOUTS DE CHAQUE ARETE pour trouver les
// traversantes : `2 E = 3 V` predicats, soit une soixantaine sur une cellule de Voronoi moyenne,
// et il faut ensuite REECRIRE la liste en entier.
//
// Ici la recherche part des sommets DEHORS : `3 nb_out` predicats, soit une douzaine. Et comme la
// boucle qui construit `map` visite deja tous les sommets et sait lesquels sont dehors, les deux
// FUSIONNENT -- une passe de moins.
//
// C'est le meme gain asymptotique que le schema « clefs des sommets dehors » mesure dans
// `Cellule3D.h`, MAIS SANS SON PRIX : pas de tri, et pas d'equation de plan a garder, parce que le
// voisin NOMME le bout dedans de l'arete au lieu de le laisser deviner. L'interpolation reste
// ancree sur le sommet dedans, exactement comme avant.
//
// = LES SOMMETS GARDES NE BOUGENT PAS -- ET C'EST CE QUI A RENDU LE PLUS
//
// La premiere version renumerotait : les sommets gardes etaient compactes vers le bas, les neufs
// mis a la suite, et toute l'adjacence relue a travers une table `map`. Cela coutait NEUF TABLEAUX
// recopies sur `nv` elements plus trois consultations de `map` par sommet, pour une coupe qui
// n'enleve que trois a cinq sommets.
//
// ON REMPLIT PLUTOT LES TROUS. Les sommets DEHORS laissent des places libres ; les sommets neufs
// s'y installent, et ce qui reste va a la suite. Les survivants gardent donc leur indice, leur
// adjacence reste valable telle quelle, ET IL N'Y A PLUS DE `map` DU TOUT. Le commit passe de
// `O( nv )` a `O( nm )`.
//
// LE SEUL CAS OU UN SOMMET GARDE BOUGE est celui d'une coupe qui enleve plus de sommets qu'elle
// n'en cree. Avec `k` sommets enleves portant `e` aretes entre eux, la coupe cree `nm = 3 k - 2 e`
// sommets, donc `nm < k` demande `e > k`, c'est-a-dire un CYCLE parmi les sommets coupes. Cela
// arrive, et alors `nt - nm` sommets demenagent -- jamais `nv`.
//
// MESURE, mediane de trois placements de code, minimum de trois executions :
//
//   diagramme complet ( AaBsp3 )    n=2000  x1.160    n=50000  x1.153    n=200000  x1.155
//   noyau nu, ordre arbitraire      x1.218
//   noyau nu, Morton en s'ecartant  x1.127 ( n=200 )  x1.038 ( n=1000 )  x0.957 ( n=5000 )
//
// LE x0.957 A n=5000 EST COHERENT et vaut d'etre garde : dans ce regime 99,2 % des coupes n'ont
// aucun effet, donc le commit ne tourne presque jamais et il ne reste que le petit surcout du
// tableau `dest` et de la comptabilite des trous. Ce n'est pas le regime de production -- avec
// elagage, toute coupe proposee ou presque est effective ( voir `FournisseurBsp3.h` ), et le gain
// y est de 15,5 % partout.
//
// La fermeture de la face neuve reste en `nm^2` : deux sommets neufs sont voisins exactement
// quand ils partagent une ANCIENNE coupe. `nm` vaut quatre a huit, donc la boucle est minuscule ;
// ce qui change est seulement qu'il faut ranger le voisin trouve DANS LA BONNE FENTE, ce que
// l'invariant donne sans chercher.
//
// = LA MESURE
//
// Cinq executions entrelacees, minimum, ns par coupe PROPOSEE, les deux binaires construits avec
// EXACTEMENT la meme recette. `f` compte les differences de combinatoire avec `Cell3T`, `a` les
// violations de l'invariant ( `verifie()`, sur chaque cellule ).
//
//                             aretes      voisins        voisins ( u8 )
//   n=50   autour=1          124.92      97.56  x1.280      x1.192
//   n=200  autour=1           55.56      44.85  x1.239      x1.145
//   n=1000 autour=1           23.61      20.25  x1.166      x1.075
//   n=5000 autour=1           13.80      12.83  x1.075      x1.004
//   n=200  autour=0          168.96     129.17  x1.308      x1.188
//   n=1000 autour=0          123.56      90.48  x1.366      x1.226
//   n=1000 Laguerre           16.08      13.38  x1.202      x1.112
//   n=50   nuage brut        159.16     124.48  x1.279      x1.178
//                                       f0 a0
//
// RAPPORTE A LA COUPE EFFECTIVE -- les coupes sans effet sont bit pour bit les memes dans les deux,
// donc tout l'ecart leur revient --, c'est entre 89 et 118 ns de travail supprime sur les six cas
// uniformes, dont le melange va pourtant de 0,82 % a 38,9 % de coupes efficaces. C'est une
// quantite FIXE qui disparait, et la variation du tableau ci-dessus n'est que le melange.
//
// Les deux ecarts a cette plage la confirment plutot qu'ils ne la contredisent, parce qu'ils
// suivent LA TAILLE DE LA CELLULE AU MOMENT DE LA COUPE, comme doit le faire une boucle en `O( E )`
// qu'on supprime : 154 ns a `n=1000 autour=0`, ou une coupe efficace arrive sur une cellule qui a
// encore jusqu'a 128 sommets ; 57 ns en Laguerre, ou les cellules font 10,4 sommets au lieu de 23,6.
//
// C'est le premier des changements de cette serie qui rende quelque chose DANS LE BON ORDRE aussi.
// Le schema « clefs des sommets dehors » plus bas gagnait la meme chose asymptotiquement mais la
// rendait aussitot en tri et en equations de plan ; ici il n'y a rien a rendre.
//
// = LE COMPTE CONTRE LE SIMPLE OU, ET LA PLACE DU TEST DE CELLULE VIDE
//
// Deux reordonnancements ont ete essayes dans la premiere passe, tous deux plausibles, tous deux
// mesures, et AUCUN N'EST GARDE.
//
//   * DEPLACER `nb_out == nv` A LA FIN, puisque la cellule videe d'un seul coup est tres rare. Le
//     test devient `nn == 0` apres la boucle fusionnee, et il est alors gratuit. Mesure : x0.960 a
//     x0.997 -- IL COUTE, sur les cinq cas et sur les quatre placements de code. Le test qu'on
//     croyait supprimer ne coutait deja rien : il est TOUJOURS FAUX, donc parfaitement predit, et
//     le deplacer ne fait que pousser une branche dans le chemin des coupes EFFECTIVES, ou elle
//     n'etait pas.
//
//   * `mb |= to_bits( ... )` AU LIEU DE `nb_out += popcount( ... )`, sur l'hypothese que le
//     `popcount` coute. Il ne coute pas : c'est une micro-operation, et la chaine portee par la
//     boucle est la meme ( un `+=` et un `|=` ont la meme latence ) dans une boucle bornee par
//     trois FMA et une ecriture. Le simple OU ne peut d'ailleurs pas etre garde SEUL : sans
//     `nb_out`, le test de cellule vide doit partir a la fin, donc il traine avec lui la perte
//     ci-dessus. L'ensemble donne x1.015 a x1.027 sur les cas uniformes en bon ordre, mais x0.941
//     et x0.955 sur l'ordre arbitraire et sur Laguerre -- et Laguerre est le cas qui nous
//     interesse. La premiere passe reste donc celle qui COMPTE.
//
// = CE QUE LA METHODE A COUTE, ET IL FAUT QUE CELA RESTE ECRIT
//
// Deux comparaisons ont ete publiees faussees avant d'etre reprises, et les deux fautes sont du
// meme genre : le binaire mesure n'etait pas celui qu'on croyait.
//
//   1. LES RECETTES DE CONSTRUCTION. La cible de reference sortait de la boucle `bancs` de
//      `xmake.lua`, avec `add_deps( "bench" )` et `-fopenmp` ; la cible de variante etait ecrite a
//      la main sans eux. Le drapeau OpenMP lui-meme ne fait rien ( +-0,5 %, verifie ), mais le
//      PLACEMENT du code qui en resulte vaut 4 %, ce qui est l'ordre de grandeur de la plupart des
//      effets qu'on mesure ici. `xmake.lua` a maintenant une fonction `reglages_banc()` unique, et
//      toute cible de variante doit passer par elle.
//
//   2. L'ORDRE DES `-I`. Compiler une variante avec `g++ ... -Isrc -Ivariante` place `src/` EN
//      PREMIER : les trois « variantes » lisaient le meme en-tete, et le tableau « tout est
//      identique » comparait un binaire a lui-meme. Un tableau ou tout est egal a un pour mille
//      pres doit etre tenu pour suspect, pas pour concluant.
//
// D'OU LA REGLE, pour toute mesure sous 5 % : QUATRE PLACEMENTS DE CODE PAR VARIANTE
// ( `-falign-functions` / `-falign-loops` a 16, 32, 64 ), et on compare les MEDIANES. C'est ce qui
// a permis de separer les 2 a 4 % des deux reordonnancements ci-dessus du bruit de placement, qui
// est du meme ordre sur un seul binaire.
//
// = L'INDICE DE SOMMET SUR HUIT BITS : MESURE, PERDANT, RETIRE
//
// Les voisins ont ete essayes en `unsigned char` -- ce qui borne la cellule a 256 sommets. Colonne
// de droite du tableau ci-dessus : ils PERDENT de 4 a 7 % contre `int`, systematiquement, et ne
// valent au mieux que l'egalite a n=5000.
//
// Il n'y avait rien a gagner : a `MaxNv = 128` les trois tableaux font 1,5 ko en `int`, donc ils
// sont deja en L1 et le trafic n'est pas le probleme. Restent les extensions de zero a chaque
// lecture et une ecriture d'octet dans la boucle de recopie -- du cout pur. Le parametre a donc
// ete enleve plutot que laisse en place ; le remettre est trois lignes, et cela ne vaudra la peine
// que la ou la memoire manque vraiment.
//
// A largeur egale L'OCCUPATION EST DE TOUTE FACON LA MEME QUE CELLE DES ARETES : `3 V` entrees
// ici, `2 E = 3 V` la-bas. Il n'y avait rien a gagner de ce cote non plus, et ce n'etait pas la
// question -- tout l'interet est dans le SENS DE PARCOURS, l'adjacence se laissant interroger
// depuis les sommets dehors quand la liste d'aretes ne le permet pas.
// = LA PREMIERE PASSE : CE QUE LE SIMD Y GAGNE, ET OU
//
// `s[ i ] = d . v[ i ] - off` sur tous les sommets est une reduction pure, et `gcc -O3
// -march=native` LA VECTORISE DEJA en 32 octets -- verifie par `-fopt-info-vec-optimized`. Ecrire
// les FMA a la main ne rendait donc rien de ce cote-la.
//
// LE GAIN EST DANS LA REDUCTION, pas dans l'arithmetique. `nb_out += s[ i ] > 0` oblige le
// vectoriseur a materialiser un vecteur de 0 et de 1 puis a le sommer horizontalement ; le
// `popcount` d'un masque ne coute rien. Et c'est pour cela que le gain CROIT avec `n` -- x1.03 a
// n = 50, x1.26 a n = 1000 : plus il y a de germes, plus la proportion de coupes SANS EFFET est
// grande, et une coupe sans effet n'est QUE cette premiere passe.
//
// DEUX HYPOTHESES ONT ETE MESUREES ET ECARTEES, elles sont ici pour qu'on ne les reprenne pas :
//
//   * GARDER LE PREDICAT EN BITS plutot qu'en flottants. Tout le reste de la coupe ne demande a
//     `s` que son SIGNE -- environ 80 lectures par coupe sur une cellule moyenne --, donc un
//     masque de bits devait remplacer tout cela par des tests sur deux mots de 64 bits. Mesure :
//     x1.143 contre x1.263, soit LA MOITIE DU GAIN PERDUE. `( deh[ i >> 6 ] >> ( i & 63 ) ) & 1`
//     est un decalage variable indexe, trois micro-operations, la ou `s[ i ] > 0` est un
//     chargement independant qui alimente un branchement bien predit ; et le construire ajoute une
//     lecture-modification-ecriture dans la boucle qu'on voulait justement laisser libre.
//
//   * MASQUER TOUS LES CHARGEMENTS plutot que d'isoler la queue. Mesure : x1.208 contre x1.263.
//     Le masque devient une dependance dans le corps de boucle alors qu'il n'y sert a rien.
//
// D'OU LA FORME QUI RESTE : chargements PLEINS tant qu'il reste un vecteur entier, puis UNE seule
// queue partielle. Lire au-dela de `nv` marcherait -- les tableaux font `MaxNv` -- mais ce serait
// lire de la memoire non initialisee pour ne rien gagner.
//
// = ELLE EST ECRITE EN ASIMD, ET LA LARGEUR EST UN PARAMETRE
//
// Plus une intrinseque dans ce fichier : `SimdVec<float,W>`, `fma`, `to_bits`, `load_partial` /
// `store_partial` disent exactement la meme chose, et se compilent sur n'importe quelle cible --
// quand le registre n'a pas `W` voies, asimd DECOUPE le vecteur recursivement au lieu de refuser.
// Verifie : `-march=x86-64` ( SSE2, deux moities de quatre voies, `mulps` + `addps` faute de FMA ),
// `-march=x86-64-v2`, `-march=x86-64-v3` ( AVX2, un ymm et de vrais `vfmadd` ) et `-march=native`.
// L'ecriture aux intrinseques, elle, retombait au scalaire des qu'AVX-512VL manquait.
//
// ET CELA NE COUTE RIEN. Cinq executions entrelacees, minimum, ns par coupe PROPOSEE ; la colonne
// `intr` est le binaire aux intrinseques d'avant, garde tel quel pour l'A/B :
//
//                         intr      asimd W=1   asimd W=4   asimd W=8   asimd W=16
//   n=200  autour=1      53.42        67.38       56.35       53.43        60.01
//   n=1000 autour=1      22.83        39.71       26.59       22.92        24.74
//   n=5000 autour=1      13.53        32.20       17.91       13.64        14.10
//   n=1000 autour=0     117.56       134.60      119.76      114.66       133.25
//
// A HUIT VOIES l'ecart a l'intrinseque est de -0,4 % a -0,8 % sur le bon ordre et de +2,5 % sur
// l'autre : rien, et c'etait la seule question. ( Il a fallu deux corrections pour y arriver :
// `nv % W` faisait emettre une division signee -- six instructions au lieu du `and` -- et il
// valait mieux donner le `LaneRange` de la queue a `to_bits` que masquer les bits apres coup. )
//
// LA LARGEUR ETAIT UNE ASSERTION, ELLE EST MAINTENANT UNE MESURE. On disait « pas de registres de
// 512 bits, ils font baisser la frequence sur le Xeon W-2145 » sans l'avoir montre sur ce code :
// W = 16 perd 4 a 12 %, et W = 4 en perd 2 a 25 %. HUIT VOIES EST LE DEFAUT, et c'est aussi la
// largeur qui tient dans un ymm sur AVX2 comme sur AVX-512 et se coupe en deux sur NEON.
//
// W = 1 DIT CE QUE LA VECTORISATION RAPPORTE VRAIMENT : de x1.26 a x2.38 selon `n`. C'est plus que
// le x1.26 mesure autrefois contre la « variante scalaire », parce que celle-la etait une boucle
// que `gcc -O3 -march=native` vectorisait toute seule ; `SimdVec<float,1>` est, lui, du scalaire
// pour de bon. Le gain croit avec `n` pour la raison dite plus haut -- les coupes sans effet.
//
// = UN TROISIEME SCHEMA A ETE ESSAYE -- LA CONNECTIVITE RECOMPOSEE PAR UN TRI, ET IL PERD
//
// La connectivite peut se RECOMPOSER a chaque coupe au lieu d'etre maintenue du tout. Dans un polytope
// simple chaque paire de coupes est portee par EXACTEMENT DEUX sommets -- c'est l'arete ou les
// deux faces se rencontrent, et un convexe n'en a qu'une. Il suffit donc d'emettre par sommet ses
// trois paires, sous la forme d'un entier `[ ka:8 | kb:8 | dehors:1 | sommet:8 ]`, de trier, et de
// lire les entrees deux par deux : celles dont le drapeau differe sont les aretes traversantes, et
// la clef donne leurs DEUX BOUTS. Aucune liste a tenir a jour, et le tri d'entiers natifs tient en
// 32 bits grace aux indices etroits.
//
// CELA COUTE DE 2,2 A 4,2 FOIS PLUS CHER, et la raison est arithmetique, pas une question de tri :
//
//                     tri `std::sort`   tri par insertion   liste d'aretes
//   n=200  ordre bon       120.97            114.13              53.14
//   n=1000 ordre bon        45.79             43.29              23.74
//   n=1000 ordre mauvais   424.19            376.23             116.54
//   n=5000 ordre mauvais   399.83            335.03              94.83
//
// Les clefs sont `3 V`, les aretes sont `E = 3 V / 2` : la liste de clefs fait DEUX FOIS la liste
// d'aretes, et il faut encore la TRIER, soit un facteur `log V` que la liste d'aretes ne paie
// jamais. Changer de tri deplace la mesure de 6 a 16 % ; il en faudrait 300. La liste d'aretes EST
// la liste de clefs triee, maintenue au lieu d'etre refaite -- c'est le meme objet, et le
// maintenir est ce qui gagne.
//
// N'EMETTRE LES CLEFS QUE POUR LES SOMMETS DEHORS echappe a cet argument : `3 nb_out` clefs, une
// dizaine au lieu de quatre cents. Une paire vue deux fois est une arete entierement dehors qui
// meurt, une paire vue UNE fois est traversante ; le bout dedans n'etant plus nomme, le sommet
// neuf se calcule comme intersection de trois plans, ce qui oblige a garder l'equation de chaque
// coupe. Mesure : EGALITE A UN POUR MILLE dans l'ordre qu'on utilise ( x0.995 a x1.002 de n = 50 a
// n = 5000 ), et x1.04 a x1.18 dans l'ordre arbitraire. Le gain suit la TAILLE DE LA CELLULE AU
// MOMENT DE LA COUPE : en partant du germe courant pour s'en ecarter, la cellule tombe a sa taille
// finale en quelques coupes et il n'y a plus d'aretes a relire ; dans l'ordre arbitraire une coupe
// efficace arrive sur une cellule qui a encore jusqu'a 128 sommets, donc 190 aretes.
//
// D'OU LE CHOIX : la liste d'aretes reste, parce que dans l'ordre qu'on utilise l'autre schema ne
// rend rien et qu'il coute une equation de plan par coupe. CE QUI POURRAIT LE ROUVRIR est le
// fournisseur 3D AVEC ELAGAGE, qui n'existe pas encore : une descente d'arbre ne propose pas du
// plus proche au plus loin, et si son ordre ressemble a l'ordre arbitraire il faudra remesurer.
// =====================================================================================

#include "supercell/Contrat3D.h"

#include <asimd/asimd.h>
#include <cmath>

#ifndef NOYAU3D_COMPTE
#define NOYAU3D_COMPTE 0
#endif

namespace noyau3d {

/// `MaxNv` borne les sommets, `W` est la largeur SIMD de la premiere passe.
template<int MaxNv,int W = 8>
struct Cellule3 {
    using V = asimd::SimdVec<float,W>;

    static constexpr int max_nv = MaxNv;
    static constexpr int max_nc = MaxNv;
    static_assert( MaxNv % W == 0, "les sommets se lisent par groupes de W" );
    static_assert( ( W & ( W - 1 ) ) == 0, "la largeur SIMD est une puissance de deux" );

    int nv = 0, ne = 0, nc = 0;

    int seuil_compacte = max_nc;
    long long nb_compactions = 0;
#if NOYAU3D_COMPTE
    long long nb_prop = 0, nb_eff = 0;
#endif

    alignas( 64 ) float vx[ MaxNv ];
    alignas( 64 ) float vy[ MaxNv ];
    alignas( 64 ) float vz[ MaxNv ];

    /// les trois coupes du sommet, en indices dans `cid`, TRIEES CROISSANT
    alignas( 64 ) int vk0[ MaxNv ];
    alignas( 64 ) int vk1[ MaxNv ];
    alignas( 64 ) int vk2[ MaxNv ];

    /// les trois voisins du sommet. `vnj` est EN FACE de `vkj` -- voir l'en-tete.
    alignas( 64 ) int vn0[ MaxNv ];
    alignas( 64 ) int vn1[ MaxNv ];
    alignas( 64 ) int vn2[ MaxNv ];

    alignas( 64 ) int cid[ max_nc ];                     ///< l'identifiant GLOBAL de chaque coupe

    /// LES DEUX FACES QUI PORTENT L'ARETE `j` DU SOMMET : ses coupes, privees de la `j`-ieme. Rien
    /// a chercher -- c'est l'invariant.
    static void faces_de( const int k[ 3 ], int j, int &f0, int &f1 ) {
        f0 = k[ j == 0 ? 1 : 0 ];
        f1 = k[ j == 2 ? 1 : 2 ];
    }

    /// LE CUBE UNITE. Coupes `0:x=0 1:x=1 2:y=0 3:y=1 4:z=0 5:z=1`, identifiants globaux `-1 .. -6`.
    void init_cube() {
        nc = 6;
        for ( int k = 0; k < 6; ++k ) cid[ k ] = -1 - k;
        nv = 8;
        ne = 12;
        for ( int b = 0; b < 8; ++b ) {
            const int i = b & 1, j = ( b >> 1 ) & 1, k = ( b >> 2 ) & 1;
            vx[ b ] = float( i ); vy[ b ] = float( j ); vz[ b ] = float( k );
            vk0[ b ] = i;                                // 0 ou 1
            vk1[ b ] = 2 + j;                            // 2 ou 3
            vk2[ b ] = 4 + k;                            // 4 ou 5   -- deja croissant
            // en face de la coupe en x : l'arete portee par les faces y et z, donc celle qui court
            // le long de x. Et ainsi de suite.
            vn0[ b ] = b ^ 1;
            vn1[ b ] = b ^ 2;
            vn2[ b ] = b ^ 4;
        }
    }

    /// LA COUPE par `p.dx x + p.dy y + p.dz z <= p.off`. Rien n'est ecrit avant que les tailles
    /// finales soient connues : sur debordement la cellule reste INTACTE.
    int coupe( const Plan3 &p ) {
        alignas( 64 ) float s[ MaxNv ];

        // ---- LA PREMIERE PASSE. Elle COMPTE les sommets dehors, et c'est une mesure : voir
        // « LE COMPTE CONTRE LE SIMPLE OU » dans l'en-tete.
        int nb_out = 0;
        {
            const V dx( p.dx ), dy( p.dy ), dz( p.dz ), mof( -p.off ), zero( 0.f );
            const int plein = nv & ~( W - 1 );
            int i = 0;
            for ( ; i < plein; i += W ) {
                const V sv = asimd::fma( dx, V::load_aligned( vx + i ),
                             asimd::fma( dy, V::load_aligned( vy + i ),
                             asimd::fma( dz, V::load_aligned( vz + i ), mof ) ) );
                sv.store_aligned( s + i );
                nb_out += __builtin_popcountll( asimd::to_bits( sv > zero ) );
            }
            if ( i < nv ) {
                const auto q = asimd::LaneRange<0>( nv - i );
                const V sv = asimd::fma( dx, V::load_partial( vx + i, q ),
                             asimd::fma( dy, V::load_partial( vy + i, q ),
                             asimd::fma( dz, V::load_partial( vz + i, q ), mof ) ) );
                sv.store_partial( s + i, q );
                nb_out += __builtin_popcountll( asimd::to_bits( sv > zero, q ) );
            }
        }

#if NOYAU3D_COMPTE
        ++nb_prop;
        nb_eff += nb_out > 0;
#endif
        if ( nb_out == 0 )                               // LE CAS FREQUENT, ET IL EST LE PREMIER
            return INCHANGEE;
        // LA CELLULE VIDE EST TRES RARE, et pourtant elle reste ICI et non a la fin : voir
        // l'en-tete. La deplacer a ete essaye et coute.
        if ( nb_out == nv ) { nv = 0; ne = 0; nc = 0; return VIDE; }

        if ( nc >= seuil_compacte ) {
            compacte();
            if ( nc >= max_nc ) return DEBORDE;
        }
        const int knew = nc;

        // ---- UNE SEULE PASSE SUR LES SOMMETS : on note les TROUS que laissent les sommets
        // DEHORS et, pour chacun d'eux, ses aretes traversantes.
        //
        // LES SOMMETS GARDES NE BOUGENT PAS, et c'est tout l'objet de cette version. Au lieu de
        // renumeroter la cellule entiere -- neuf tableaux recopies plus une consultation de `map`
        // par voisin --, on REMPLIT LES TROUS avec les sommets neufs. L'adjacence des survivants
        // reste alors valable telle quelle, et il n'y a plus de `map` du tout.
        int   trou[ MaxNv ], nt = 0;
        float nx[ MaxNv ], ny[ MaxNv ], nz[ MaxNv ];
        int   n0[ MaxNv ], n1[ MaxNv ];                  ///< les deux coupes HERITEES, triees
        int   rec_v[ MaxNv ], rec_f[ MaxNv ];            ///< le sommet DEDANS a recoller, et sa fente
        int   nm = 0;

        for ( int o = 0; o < nv; ++o ) {
            if ( ! ( s[ o ] > 0 ) ) continue;
            trou[ nt++ ] = o;

            const int k[ 3 ] = { vk0[ o ], vk1[ o ], vk2[ o ] };
            const int w[ 3 ] = { vn0[ o ], vn1[ o ], vn2[ o ] };
            for ( int j = 0; j < 3; ++j ) {
                const int u = w[ j ];
                if ( s[ u ] > 0 )
                    continue;                            // arete entierement dehors : elle meurt
                if ( nm >= MaxNv )
                    return DEBORDE;

                // ANCRE SUR LE SOMMET DEDANS, comme en 2D : avec `s_u == 0` la forme symetrique ne
                // rend pas `v_u` en flottant, et le sommet passerait DE L'AUTRE COTE du plan.
                const float t = s[ u ] / ( s[ u ] - s[ o ] );
                nx[ nm ] = vx[ u ] + ( vx[ o ] - vx[ u ] ) * t;
                ny[ nm ] = vy[ u ] + ( vy[ o ] - vy[ u ] ) * t;
                nz[ nm ] = vz[ u ] + ( vz[ o ] - vz[ u ] ) * t;
                faces_de( k, j, n0[ nm ], n1[ nm ] );

                // la fente de `u` qui pointait vers `o` : c'est elle qui prendra le sommet neuf
                rec_v[ nm ] = u;
                rec_f[ nm ] = vn0[ u ] == o ? 0 : ( vn1[ u ] == o ? 1 : 2 );
                ++nm;
            }
        }

        const int nn = nv - nt;
        const int new_nv = nn + nm;
        if ( new_nv > MaxNv )
            return DEBORDE;

        // OU VA CHAQUE SOMMET NEUF : dans un trou tant qu'il en reste, puis a la suite. Les trous
        // sont pris dans l'ordre CROISSANT, et les `nm` premiers sont toujours `< new_nv` : il y a
        // au plus `nt - nm` trous au-dela de `new_nv`, donc au moins `nm` en deca.
        int dest[ MaxNv ];
        for ( int j = 0; j < nm; ++j ) dest[ j ] = j < nt ? trou[ j ] : nv + ( j - nt );

        // ---- LES VOISINS DES SOMMETS NEUFS. Le sommet neuf porte `( n0, n1, knew )` : en face de
        // `knew` il y a l'arete portee par `( n0, n1 )`, donc le bout DEDANS dont il vient -- et cet
        // indice-la est deja le bon, puisque les gardes ne bougent pas.
        int m0[ MaxNv ], m1[ MaxNv ], m2[ MaxNv ];
        for ( int i = 0; i < nm; ++i ) {
            m2[ i ] = rec_v[ i ];
            m0[ i ] = m1[ i ] = -1;                      // `verifie()` les rattrape si la fermeture
        }                                                // en manquait un

        // Les deux autres sont ses voisins SUR LA FACE NEUVE : deux sommets neufs sont voisins
        // exactement quand ils partagent une ANCIENNE coupe `kc`, et l'arete qui les joint est
        // alors portee par `( kc, knew )` -- donc elle est EN FACE de l'autre coupe heritee.
        for ( int i = 0; i < nm; ++i )
            for ( int j = i + 1; j < nm; ++j ) {
                int kc;
                if      ( n0[ i ] == n0[ j ] || n0[ i ] == n1[ j ] ) kc = n0[ i ];
                else if ( n1[ i ] == n0[ j ] || n1[ i ] == n1[ j ] ) kc = n1[ i ];
                else continue;
                if ( kc == n0[ i ] ) m1[ i ] = dest[ j ]; else m0[ i ] = dest[ j ];
                if ( kc == n0[ j ] ) m1[ j ] = dest[ i ]; else m0[ j ] = dest[ i ];
            }

        // ---- COMMIT. Rien n'a bouge jusqu'ici.
        for ( int j = 0; j < nm; ++j ) {                 // les sommets NEUFS, dans les trous
            const int m = dest[ j ];
            vx[ m ] = nx[ j ]; vy[ m ] = ny[ j ]; vz[ m ] = nz[ j ];
            // les deux heritees sont < `knew`, donc mettre la neuve en dernier garde le tri
            vk0[ m ] = n0[ j ]; vk1[ m ] = n1[ j ]; vk2[ m ] = knew;
            vn0[ m ] = m0[ j ]; vn1[ m ] = m1[ j ]; vn2[ m ] = m2[ j ];
        }
        for ( int i = 0; i < nm; ++i ) {                 // le recollage, cote sommet DEDANS
            const int u = rec_v[ i ];
            switch ( rec_f[ i ] ) {
                case 0: vn0[ u ] = dest[ i ]; break;
                case 1: vn1[ u ] = dest[ i ]; break;
                default: vn2[ u ] = dest[ i ]; break;
            }
        }

        // ---- LES TROUS QUI RESTENT, quand la coupe enleve plus de sommets qu'elle n'en cree.
        // C'est le seul cas ou un sommet garde bouge, et il n'en bouge alors que `nt - nm`, pas
        // `nv`. ( `nm = 3 k - 2 e` pour `k` sommets enleves portant `e` aretes internes : il faut
        // un cycle parmi les sommets coupes pour que `nm < k`, ce qui n'arrive pas a chaque fois. )
        if ( nm < nt ) {
            int th = nt;                                 // les trous deja au-dela de la fin
            while ( th > nm && trou[ th - 1 ] >= new_nv ) --th;

            int nouv[ MaxNv ], src[ MaxNv ], dst[ MaxNv ], nmv = 0;
            int ct = th, cd = nm;
            for ( int i = new_nv; i < nv; ++i ) {
                if ( ct < nt && trou[ ct ] == i ) { ++ct; continue; }   // ce slot EST un trou
                src[ nmv ] = i;
                dst[ nmv ] = trou[ cd++ ];
                nouv[ i - new_nv ] = dst[ nmv ];
                ++nmv;
            }
            for ( int t = 0; t < nmv; ++t ) {
                const int a = src[ t ], b = dst[ t ];
                vx[ b ] = vx[ a ]; vy[ b ] = vy[ a ]; vz[ b ] = vz[ a ];
                vk0[ b ] = vk0[ a ]; vk1[ b ] = vk1[ a ]; vk2[ b ] = vk2[ a ];
                vn0[ b ] = vn0[ a ]; vn1[ b ] = vn1[ a ]; vn2[ b ] = vn2[ a ];
            }
            // et les voisins qui pointaient vers eux. Un voisin peut lui-meme avoir demenage :
            // `nouv` donne sa nouvelle place, et sa copie porte encore les ANCIENS indices, donc
            // chercher `a` y marche.
            for ( int t = 0; t < nmv; ++t ) {
                const int a = src[ t ], b = dst[ t ];
                const int w[ 3 ] = { vn0[ b ], vn1[ b ], vn2[ b ] };
                for ( int j = 0; j < 3; ++j ) {
                    const int q = w[ j ] >= new_nv ? nouv[ w[ j ] - new_nv ] : w[ j ];
                    if      ( vn0[ q ] == a ) vn0[ q ] = b;
                    else if ( vn1[ q ] == a ) vn1[ q ] = b;
                    else                      vn2[ q ] = b;
                }
            }
        }

        cid[ knew ] = p.id;
        nc = knew + 1;
        nv = new_nv;
        ne = 3 * new_nv / 2;                             // Euler, exact pour un polytope simple
        return COUPEE;
    }

    /// ENLEVER LES COUPES MORTES. La renumerotation est MONOTONE, donc les triplets restent tries.
    /// L'adjacence ne la voit pas : elle porte des numeros de SOMMET.
    void compacte() {
        ++nb_compactions;
        int m[ max_nc ];
        for ( int k = 0; k < nc; ++k ) m[ k ] = -1;
        for ( int i = 0; i < nv; ++i ) { m[ vk0[ i ] ] = 0; m[ vk1[ i ] ] = 0; m[ vk2[ i ] ] = 0; }
        int q = 0;
        for ( int k = 0; k < nc; ++k )
            if ( m[ k ] == 0 ) { cid[ q ] = cid[ k ]; m[ k ] = q++; }
        for ( int i = 0; i < nv; ++i ) {
            vk0[ i ] = m[ vk0[ i ] ]; vk1[ i ] = m[ vk1[ i ] ]; vk2[ i ] = m[ vk2[ i ] ];
        }
        nc = q;
    }

    /// LE VOLUME, sans jamais parcourir un cycle -- meme formule que dans `Cellule3D.h`, et voir
    /// son en-tete pour la justification. La seule difference est que les DEUX FACES d'une arete
    /// sont ici donnees par l'invariant au lieu d'etre cherchees par `communes2`.
    double volume() const {
        if ( nv < 4 ) return 0;
        int v0[ max_nc ];
        double sx[ max_nc ], sy[ max_nc ], sz[ max_nc ];
        for ( int k = 0; k < nc; ++k ) { v0[ k ] = -1; sx[ k ] = sy[ k ] = sz[ k ] = 0; }
        for ( int i = nv - 1; i >= 0; --i ) { v0[ vk0[ i ] ] = i; v0[ vk1[ i ] ] = i; v0[ vk2[ i ] ] = i; }

        double gx = 0, gy = 0, gz = 0;
        for ( int i = 0; i < nv; ++i ) { gx += vx[ i ]; gy += vy[ i ]; gz += vz[ i ]; }
        gx /= nv; gy /= nv; gz /= nv;

        for ( int a = 0; a < nv; ++a ) {
            const int k[ 3 ] = { vk0[ a ], vk1[ a ], vk2[ a ] };
            const int w[ 3 ] = { vn0[ a ], vn1[ a ], vn2[ a ] };
            for ( int j = 0; j < 3; ++j ) {
                const int b = w[ j ];
                if ( b <= a ) continue;                  // chaque arete vue une seule fois
                int f0, f1;
                faces_de( k, j, f0, f1 );
                for ( int r = 0; r < 2; ++r ) {
                    const int f = r ? f1 : f0;
                    if ( f < 0 ) continue;
                    const int o = v0[ f ];
                    const double ax = vx[a] - vx[o], ay = vy[a] - vy[o], az = vz[a] - vz[o];
                    const double bx = vx[b] - vx[o], by = vy[b] - vy[o], bz = vz[b] - vz[o];
                    double cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
                    if ( sx[f] * cx + sy[f] * cy + sz[f] * cz < 0 ) { cx = -cx; cy = -cy; cz = -cz; }
                    sx[f] += cx; sy[f] += cy; sz[f] += cz;
                }
            }
        }

        double v = 0;
        for ( int k = 0; k < nc; ++k ) {
            const int o = v0[ k ];
            if ( o < 0 ) continue;                       // coupe morte, pas encore compactee
            v += std::abs( ( vx[o] - gx ) * sx[k] + ( vy[o] - gy ) * sy[k] + ( vz[o] - gz ) * sz[k] );
        }
        return v / 6;
    }

    /// LE CONTROLE DE L'INVARIANT, et il n'est pas optionnel : sans liste d'aretes, plus rien
    /// d'exterieur ne dit que l'adjacence est coherente. Rend le nombre de violations.
    ///
    /// Pour chaque sommet `a` et chaque fente `j`, avec `b = vnj[ a ]` : il doit exister UNE fente
    /// `t` de `b` qui pointe vers `a`, et les deux faces portees par la fente `j` de `a` doivent
    /// etre LES MEMES que celles portees par la fente `t` de `b` -- c'est la definition de l'arete.
    /// On verifie aussi que les trois voisins d'un sommet sont distincts.
    int verifie() const {
        int faux = 0;
        for ( int a = 0; a < nv; ++a ) {
            const int ka[ 3 ] = { vk0[ a ], vk1[ a ], vk2[ a ] };
            const int wa[ 3 ] = { vn0[ a ], vn1[ a ], vn2[ a ] };
            if ( wa[ 0 ] == wa[ 1 ] || wa[ 0 ] == wa[ 2 ] || wa[ 1 ] == wa[ 2 ] ) ++faux;
            for ( int j = 0; j < 3; ++j ) {
                const int b = wa[ j ];
                if ( b < 0 || b >= nv ) { ++faux; continue; }
                const int kb[ 3 ] = { vk0[ b ], vk1[ b ], vk2[ b ] };
                const int wb[ 3 ] = { vn0[ b ], vn1[ b ], vn2[ b ] };
                int t = -1;
                for ( int r = 0; r < 3; ++r ) if ( wb[ r ] == a ) t = r;
                if ( t < 0 ) { ++faux; continue; }
                int f0, f1, g0, g1;
                faces_de( ka, j, f0, f1 );
                faces_de( kb, t, g0, g1 );
                if ( f0 != g0 || f1 != g1 ) ++faux;
            }
        }
        return faux;
    }

    /// LES VOISINS : les identifiants GLOBAUX des coupes encore portees par un sommet.
    int voisins( int *out, int cap ) const {
        bool vivant[ max_nc ] = {};
        for ( int i = 0; i < nv; ++i ) { vivant[ vk0[i] ] = true; vivant[ vk1[i] ] = true; vivant[ vk2[i] ] = true; }
        int m = 0;
        for ( int k = 0; k < nc; ++k )
            if ( vivant[ k ] && m < cap ) out[ m++ ] = cid[ k ];
        return m;
    }
};

} // namespace noyau3d
