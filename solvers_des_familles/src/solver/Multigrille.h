#pragma once

// =====================================================================================
// UN MULTIGRILLE ALGEBRIQUE MAISON, LE MEME QUE SUR LA CARTE, AUX REGLAGES PRES. C'est le portage
// de `gpu_des_familles/src/gpu/Amg2D.cuh` -- meme agregation, meme cycle -- avec les choix que le
// CPU impose et ceux qu'il autorise. Chacun est mesure ; aucun n'est repris tel quel.
//
// = Pourquoi ne pas se contenter d'AMGCL
//
// On le garde en temoin ( `--solver amg` ). Mais un seul solveur pour les deux machines a un
// interet propre : l'agregation est LA meme, donc ce qui est mesure ici se transpose la-bas, et
// les deux codes vieillissent ensemble. Et il y a une raison technique : l'agregation d'AMGCL
// coute un appariement, la notre est gratuite.
//
// = L'AGREGATION EST GRATUITE, et c'est le point de depart
//
// Les germes sont ranges DANS L'ORDRE DE L'ARBRE ( `pd.ids` ), qui est une courbe remplissante :
// des RANGS CONSECUTIFS sont voisins dans le plan. Agreger, c'est donc `rang >> 2` -- quatre
// germes par paquet, sans noyau d'appariement, sans matching, sans compaction. Et comme les
// indices d'agregat restent ordonnes par rang, le niveau suivant s'agrege pareil : `a >> 2`. La
// hierarchie entiere tient dans un decalage.
//
// Mieux : LA CARTE INVERSE EST GRATUITE ELLE AUSSI. Le paquet `a` contient exactement les rangs
// `Sa .. Sa+S-1`. On peut donc parcourir le grossier EN BALAYANT LES LIGNES GROSSIERES, chacune
// par un seul fil, sans atomique et sans tri -- la ou le GPU emet un triplet par arete, trie
// ( CUB ) et reduit par clef. C'est vrai de l'assemblage de Galerkin, de la restriction, ET de la
// transposee de la prolongation, qu'on obtient donc sans passe de transposition.
//
// = LA TAILLE DU PAQUET ( `agreg` ), qui est LE reglage de la complexite
//
// La carte en prend quatre, parce que quatre est ce qu'un noyau CUDA aime. Ce n'est pas le bon
// choix ici, et la trace le dit -- a `n = 3e5`, non-nuls par ligne niveau par niveau :
//
//      300000 ( 6.0 ) -> 75000 ( 18.6 ) -> 18750 ( 35.0 ) -> 4688 ( 60.4 ) -> 1172 ( 90.1 )
//
// La complexite 2.37 est dominee par LE NIVEAU 1 : `75000 x 18.6 = 1.4 M` non-nuls contre `1.8 M`
// au niveau fin, soit 0.78 a lui seul. Avec des paquets de seize -- un bloc 4x4 sur la courbe
// remplissante -- il y a quatre fois moins de lignes grossieres pour un remplissage a peine plus
// grand, et deux niveaux de moins. C'est ce que fait AMGCL sans le dire ainsi : son ensemble
// independant maximal sur le graphe de force donne en 2D des paquets de sept a neuf.
//
// LES PUISSANCES DE DEUX, ET C'EST L'ARBRE QUI LE PERMET. `AaBsp.h` fait des « coupes MEDIANES
// sur l'axe le plus long » : une fenetre ALIGNEE de `2^k` rangs consecutifs est donc EXACTEMENT
// un sous-arbre -- localite parfaite -- et sa boite a ete coupee `k` fois, chaque fois sur son
// cote le plus long. `k` pair donne une boite carree ( 4, 16, 64 ), `k` impair une boite 2:1
// ( 2, 8, 32 ), ce qui pour de l'agregation va tres bien : les paquets d'AMGCL ne sont pas carres
// non plus, et comme la coupe suit l'axe long DU NUAGE LOCAL, le 2:1 est dans la metrique locale.
//
// Huit tombe donc pile dans la zone ou AMGCL se place ( sept a neuf ), ce qu'une courbe de Morton
// -- puissances de quatre en 2D -- n'aurait pas permis. Et c'est le defaut, parce que c'est ce
// que la mesure dit. `n = 2e4` / `n = 1e5`, relevement, en secondes de total :
//
//      paquet    4 :   8.84  /  96.75      complexite 2.25
//      paquet    8 :   7.38  /  97.99      complexite 1.34
//      paquet   16 :   8.49  / 105.70      complexite 1.13
//      paquet   64 :  11.09  /    -        complexite 1.02
//      AMGCL        :   7.31  /  92.14
//
// La courbe est en U et son fond est plat entre 4 et 16 : la complexite tombe quand le paquet
// grossit, les iterations montent, et les deux se croisent vers huit. Au-dela de seize l'espace
// grossier devient trop pauvre et rien ne rattrape ( 981 s a `n = 3e5` contre 757 a paquet 4 ).
//
// CE QUI RESTE INTERDIT, ce sont les tailles qui ne sont PAS des puissances de deux : une fenetre
// de neuf rangs n'est alignee sur aucune frontiere de sous-arbre, donc certains paquets
// enjamberaient une coupe de haut niveau -- deux moities du domaine dans le meme agregat.
//
// = LA PROLONGATION LISSEE, qui est ce que le CPU autorise
//
// L'agregation NON LISSEE -- `P0` constante par morceaux -- donne une correction grossiere faible,
// et le prix se lit en iterations : 2.7 fois plus que l'agregation lissee d'AMGCL, mesure a
// `n = 1e5`. Sur la carte on le paie volontiers ( une iteration y est presque gratuite ) et on
// compense par un K-cycle ; le chemin lisse y existe mais reste ETEINT ( `AMG_LISSE`, via
// `cusparseSpGEMM` ), parce qu'un produit triple creux est exactement ce qu'un GPU n'aime pas.
//
// Sur CPU c'est l'inverse. On lisse donc :
//
//     P = ( I - w D^-1 A ) P0,   soit   P[ i ][ a ] = ( 1 - w ) [ m_i = a ]
//                                                   + ( w / dia_i ) somme_( j != i, m_j = a ) c_ij
//
// -- toutes les entrees positives pour `w <= 1`, et LES LIGNES SOMMENT A UN, donc le vecteur
// constant est exactement dans l'image de `P`. C'est ce qui compte pour un laplacien : le noyau
// est represente exactement a tous les niveaux, et `A_c = P^t A P` est encore un laplacien
// ( `A_c 1 = P^t A P 1 = P^t A 1 = 0` ).
//
// Le produit triple se fait en deux passes, chacune avec un accumulateur DENSE par fil :
//   1. `AP = A P`, une ligne FINE par fil ;
//   2. `A_c = P^t ( AP )`, une ligne GROSSIERE par fil -- et `P^t` se construit sans tri, la
//      ligne `a` de `P^t` etant portee par les membres du paquet `a` et leurs voisins.
//
// = LA TRONCATURE DE `P`, sans laquelle le lissage se paie trop cher
//
// `P` a `1 + deg` entrees par ligne la ou `P0` en avait une, donc `A_c = P^t A P` SE DENSIFIE a
// chaque niveau, et le cout du produit triple croit comme le carre du remplissage. Mesure a
// `n = 2e4` sans troncature : la montee passe de 0.29 s a 6.97 s, et le lissage perd ce qu'il
// gagne en iterations. Deux indices le disaient avant qu'on le nomme -- garder la hierarchie
// aidait beaucoup, et SUPPRIMER UN NIVEAU aussi.
//
// On tronque donc chaque ligne de `P` aux entrees qui valent au moins `tronque` fois le maximum
// de la ligne, PUIS ON RENORMALISE pour que la ligne somme a un. La renormalisation n'est pas un
// detail cosmetique : c'est elle qui garde le vecteur constant exactement dans l'image de `P`,
// donc le noyau du laplacien represente a tous les niveaux. Sans elle, la troncature casse ce que
// le lissage etait venu apporter.
//
// Mesure a `n = 2e4`, non-nuls par ligne niveau par niveau :
//
//      brut                6.0 ->   8.0 ->   7.9 ->   7.2     complexite 1.44
//      lisse, sans rien    6.0 ->  27.1 -> 115.4 -> 278.1     complexite 4.08
//      lisse, tronque 0.1  6.0 ->  21.7 ->  56.0 -> 105.7     complexite 2.78
//      lisse, tronque 0.2  6.0 ->  18.2 ->  31.0 ->  46.2     complexite 2.21
//
// La troncature est MONOTONE jusqu'a 0.35 sans que le compte d'iterations bouge : on ne coupe
// donc pas encore dans le vif. `0.2` est le defaut ; au-dela le gain s'aplatit.
//
// = LE FILTRE DE FORCE ( `force` ) EST INUTILE ICI, ET IL FALLAIT LE MESURER
//
// Le remede standard a la densification est de ne lisser que le long des connexions FORTES,
// `c_ij >= force x max_k c_ik`, en reportant le reste sur la diagonale pour que la ligne de `P`
// somme encore a un. On l'a implemente ( `force`, qui reste disponible ) et mesure : a la valeur
// classique `0.08` il ne change RIEN -- complexite 2.69 contre 2.67, 10.97 s contre 10.64.
//
// La raison est structurelle et vaut d'etre retenue : dans un graphe de Laguerre les
// `c_ij = |facette| / ( 2 |p_i - p_j| )` sont TOUTES DU MEME ORDRE. Il n'y a pas de connexion
// faible a jeter. La densification vient du MOTIF -- `P` a `1 + deg` entrees et `deg ~ 6`
// partout -- et pas d'un contraste de valeurs. Le filtre de force est l'outil des problemes
// ANISOTROPES ; sur un diagramme de puissance il est hors sujet, et a `0.25` il ne fait plus que
// couper au hasard. Le defaut est donc `0`.
//
// Le filtrage, quand on l'active, ne sert QU'A CONSTRUIRE `P`. Le Galerkin se fait sur le vrai
// `A` : on ne change pas l'operateur, seulement l'espace grossier.
//
// = LE LISSEUR DE CHEBYSHEV, qui est le meme sur les deux machines
//
// Un polynome de degre `nu` en `M^-1 A` choisi pour minimiser le maximum sur `[ lmin, lmax ]` --
// la partie du spectre que le grossier NE CORRIGE PAS. Il ne demande que des produits
// matrice-vecteur, donc il se comporte pareil sur CPU et sur carte, la ou Gauss-Seidel demande
// un ordre et se parallelise mal. Adams, Brezina, Hu et Tuminaro ( « Parallel multigrid
// smoothing: polynomial versus Gauss-Seidel », 2003 ) montrent qu'il tient meme en sequentiel.
//
// `lmax` ne se devine pas, il se BORNE : Gershgorin sur `M^-1 A` donne
// `lmax <= max_i m_i ( dia_i + somme_e |val_e| )`, exact et gratuit -- deux pour un laplacien
// pur et son preconditionneur de Jacobi. `lmin = lmax / ratio` : on ne demande au lisseur que la
// moitie haute du spectre, le reste etant l'affaire du niveau grossier.
//
// = LE LISSEUR : `spai0` PLUTOT QUE JACOBI AMORTI
//
// `spai0` est la meilleure approximation DIAGONALE de `A^-1` au sens de Frobenius : minimiser
// `|| I - M A ||_F` sur `M` diagonale donne `m_i = A_ii / somme_j A_ij^2`. C'est ce qu'AMGCL
// emploie dans la variante qui gagne ici, ca coute exactement un balayage de Jacobi, et
// l'amortissement s'y regle TOUT SEUL, ligne par ligne, au lieu d'un `omega` global devine.
// Sur un laplacien a six voisins egaux il vaut `1 / ( 7 c )` la ou Jacobi non amorti vaut
// `1 / ( 6 c )` : c'est un Jacobi amorti a 0.857, mais calcule et non pose.
//
// Dans les deux cas on precalcule UN COEFFICIENT PAR LIGNE, `rlx[ i ]`, applique au residu. La
// boucle la plus chaude du cycle y perd une division.
//
// = LE CYCLE EN V, le fond RESOLU, et le K-cycle qui se perime
//
// Un lissage de Jacobi amorti avant, un apres ( meme `omega`, donc l'operateur est SYMETRIQUE et
// le CG l'accepte comme preconditionneur ). Le niveau le plus grossier, lui, est RESOLU par une
// factorisation de Cholesky creuse, et c'est la deuxieme difference avec la carte : le GPU le
// lisse 120 fois par visite et le K-cycle le visite quatre fois, soit 480 balayages par
// application. Mesure a `n = 2e4` : le fond exact fait tomber 23.8 s a 11.2, et le K-cycle
// devient un COUT ( 11.2 en V pur, 11.7 a K=1, 12.5 a K=2 ) -- il n'etait la que pour compenser
// la faiblesse de la correction grossiere, qui n'existe plus.
//
// = JACOBI A DEUX TAMPONS, troisieme difference
//
// Le noyau CUDA lit `x[ col ]` pendant que d'autres fils l'ecrivent : c'est un Jacobi/Gauss-Seidel
// hybride, non deterministe. Sur la carte ca passe ; dans un preconditionneur de CG c'est faux en
// droit -- CG exige un operateur LINEAIRE FIXE. On alterne donc deux tampons.
//
// = OPENMP ET PAS `parallel_for`, quatrieme
//
// `util/parallel.h` cree et joint ses fils A CHAQUE APPEL, avec epinglage : une cinquantaine de
// microsecondes. Un cycle en demande des dizaines sur des niveaux de mille inconnues. Le pool
// d'OpenMP est deja la, et la clause `if` rend la boucle sequentielle quand elle est courte.
//
// = LE RECYCLAGE DE SOUS-ESPACE ( `recycle` )
//
// On ne resout pas UN systeme, on en resout des centaines qui se ressemblent -- meme graphe a
// quelques aretes pres, second membre correle. Partir de zero a chaque fois jette cette
// information. On garde donc les `recycle` dernieres solutions dans `U` et on demarre sur la
// meilleure combinaison qu'elles permettent, au sens de l'energie :
//
//      x0 = U ( U^t A U )^-1 U^t b
//
// c'est-a-dire la projection de Galerkin sur `span( U )`. Elle coute `k` produits
// matrice-vecteur pour former `AU`, un systeme dense `k x k`, et RIEN de plus -- le residu
// `r0 = b - AU y` sort du calcul deja fait. C'est le premier etage du recyclage de Krylov
// ( Parks et al., « Recycling Krylov Subspaces for Sequences of Linear Systems » ) ; la
// deflation en cours de boucle serait le second.
//
// = LA HIERARCHIE PEUT SERVIR PLUSIEURS FOIS ( `refaire` )
//
// Entre deux iterations de Newton la hessienne CHANGE, mais son graphe bouge a peine -- quelques
// aretes. Un preconditionneur n'a pas besoin d'etre exact : on peut garder la hierarchie et ne
// rafraichir que le pointeur du niveau fin, ce qui amortit la montee sur `refaire` resolutions.
// C'est ce qui rend la prolongation lissee abordable : elle coute plus cher a monter qu'a servir.
//
// = LA JAUGE : MOYENNE NULLE POUR RESOUDRE, `d[ 0 ] = 0` POUR RENDRE
//
// La moyenne nulle est la bonne jauge pour un multigrille : le laplacien a les constantes pour
// noyau, `b = nu - a` est deja de somme nulle, et projeter est symetrique la ou rayer une ligne ne
// l'est pas. Mais le reste du code suppose l'autre -- `Newton.h` ecrit `w2 = w + t d` PUIS
// `w2[ 0 ] = 0`, « la jauge, imposee et non esperee ». On TRANSLATE donc la solution en sortie.
// Les deux jauges decrivent la meme direction a une constante pres ; mais rendre l'une quand
// l'appelant attend l'autre mutile une composante, et Newton stagne sur-le-champ ( mesure :
// residu inchange apres deux iterations, 31 reculs ).
// =====================================================================================

#include "solver/Lineaire.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <omp.h>
#include <vector>

namespace sf {

/// en dessous, la boucle reste sequentielle : le fork/join coute plus que le travail
static constexpr SI MG_SEUIL_PAR = 2048;

/// un niveau de la hierarchie. Le niveau zero POINTE sur le laplacien de l'appelant ; les autres
/// possedent leur matrice. Convention partout : `y_i = dia_i x_i - somme_e val_e x_( col_e )`.
struct NiveauMg {
    SI               n = 0, nnz = 0;
    const SI        *row = nullptr, *col = nullptr;
    const TF        *val = nullptr, *dia = nullptr;
    std::vector<SI>  arow, acol;                     ///< la possession, pour les niveaux grossiers
    std::vector<TF>  aval, adia;
    std::vector<TF>  rlx;                            ///< le coefficient de relaxation, UN PAR LIGNE
    TF               lmax = 2;                       ///< la borne de Gershgorin sur `M^-1 A`
    std::vector<TF>  x, b, r, y, z;                  ///< le cycle ( `y`, `z` : les tampons )
    std::vector<TF>  v1, v2, t, rc;                  ///< le K-cycle
};

/// une matrice creuse rectangulaire en CSR : `P` ( fin x grossier ) ou sa transposee
struct CsrMg {
    std::vector<SI> row, col;
    std::vector<TF> val;
    SI              lignes = 0, colonnes = 0;
    bool            vrai() const { return ! row.empty(); }
};

struct Mg {
    // ---- LES REGLAGES. Les valeurs viennent de la mesure sur CPU ( README § 17 ), pas de la carte.
    int      agreg   = 8;          ///< germes par paquet -- UNE PUISSANCE DE DEUX ( 4, 8, 16, ... )
    // CHEBYSHEV PAR DEFAUT, ET SUR LE TEMPS C'EST UN MATCH NUL. A `n = 1e5` : 90.19 s contre
    // 90.42 pour spai0 -- indiscernable. Ce qui tranche est ailleurs : il fait 10 732 iterations
    // de CG la ou spai0 en fait 13 641, soit -21 % pour le meme temps. C'est de la marge quand le
    // probleme durcit, et c'est un lisseur PUREMENT MATRICE-VECTEUR, donc le meme code sur les
    // deux machines -- ce que Gauss-Seidel n'est pas et ce qu'un `omega` global devine mal.
    int      lisseur = 2;          ///< 0 : Jacobi amorti ; 1 : spai0 ; 2 : Chebyshev
    TF       cheb    = TF( 10 );   ///< `lmin = lmax / cheb` -- la part du spectre laissee au grossier
    // DEUX SUFFISENT, ET C'EST LA MESURE QUI LE DIT. A `n = 2e4` le compte d'iterations de CG
    // passe de 5117 a 4337 avec DEUX vecteurs gardes ( -15 % ), et ne bouge plus ensuite : 4332 a
    // quatre, 4334 a huit, 4336 a seize. La solution precedente porte a elle seule presque toute
    // l'information -- les directions de Newton successives n'engendrent utilement qu'un ou deux
    // degres de liberte. Le prix etant `k` produits matrice-vecteur par resolution, on s'arrete la.
    int      recycle = 2;          ///< solutions gardees pour le demarrage de Galerkin ( 0 : off )
    bool     lisse   = true;       ///< la PROLONGATION LISSEE ( sinon : constante par morceaux )
    TF       omega_p = TF( 0.7 );  ///< l'amortissement du lissage de `P`
    TF       tronque = TF( 0.2 );  ///< on jette les entrees de `P` sous cette fraction du max de la ligne
    TF       force   = TF( 0 );    ///< on ne LISSE que le long des `c_ij >= force x max_k c_ik`
                                   ///< ( 0 : eteint -- mesure inutile sur un graphe de Laguerre )
    int      trace   = 0;          ///< 1 : la taille et le remplissage de chaque niveau, une fois
    int      kcycle  = 0;          ///< niveaux acceleres par Krylov ( le fond etant resolu : aucun )
    int      gros    = 120;        ///< lissages au fond, SI la factorisation n'est pas disponible
    int      nu      = 3;          ///< lissages avant et apres, par niveau ( 3 avec Chebyshev )
    int      stop    = 1000;       ///< on arrete de grossir en dessous
    bool     exact   = true;       ///< le fond RESOLU ( Cholesky creux ) au lieu de lisse
    // QUATRE RESOLUTIONS PAR HIERARCHIE. La prolongation lissee coute plus cher a monter qu'a
    // servir, et le graphe de la hessienne bouge a peine d'une iteration de Newton a l'autre.
    // Mesure a `n = 1e5` ( relevement, en secondes de total ) : 112.3 a `refaire 1`, 95.9 a 4,
    // 96.7 a 8 -- au-dela, le preconditionneur rancit et les iterations reviennent.
    int      refaire = 4;          ///< la hierarchie refaite toutes les `refaire` resolutions
    TF       omega   = TF( 0.7 );  ///< l'amortissement de Jacobi
    TF       tol     = TF( 1e-6 ); ///< residu RELATIF
    int      maxit   = 20000;

    StatsLin st;

    /// `ids[ k ]` : l'identifiant du germe de rang `k` dans l'arbre. C'est `pd.ids`, et c'est la
    /// SEULE chose que ce solveur demande de plus qu'un autre. Sans lui, l'ordre des identifiants
    /// fait l'affaire -- et l'agregation ne vaut alors que ce que vaut cet ordre.
    template<class TI>
    void ordre( const TI *ids, SI nb ) {
        ord.resize( nb );
        rg.resize( nb );
        for ( SI k = 0; k < nb; ++k ) { ord[ k ] = SI( ids[ k ] ); rg[ ord[ k ] ] = k; }
        niv.clear();                                 // l'agregation change : la hierarchie est caduque
        Uv.clear(); AUv.clear();                     // ... et le sous-espace recycle aussi
    }
    bool a_l_ordre() const { return ! ord.empty(); }

    const char *nom() const {
        return lisse ? "multigrille maison ( agregation par l'arbre, prolongation LISSEE )"
                     : "multigrille maison ( agregation par l'arbre, prolongation constante )";
    }
    /// la taille de paquet effective, pour la trace
    int taille_paquet() const { return 1 << decalage(); }

    /// combien de niveaux, et leurs tailles -- pour la trace
    const std::vector<NiveauMg> &niveaux() const { return niv; }

    /// UNE RESOLUTION DE PLUS sur la derniere hierarchie. Meme surface que `Amg` et `Cholesky`.
    void resout_encore( const std::vector<TF> &b, std::vector<TF> &d ) {
        const double t0 = now();
        cg( b, d );
        st.t_res += now() - t0;
    }

    bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d ) {
        const double t0 = now();
        // LA HIERARCHIE PEUT RESSERVIR : entre deux iterations de Newton le graphe bouge a peine,
        // et un preconditionneur n'a pas besoin d'etre exact. On ne rafraichit alors que le
        // pointeur du niveau fin -- `L` est reassemble a chaque fois et peut avoir demenage.
        if ( ! niv.empty() && niv[ 0 ].n != L.n ) { Uv.clear(); AUv.clear(); }
        const bool neuf = niv.empty() || niv[ 0 ].n != L.n || depuis >= std::max( refaire, 1 );
        if ( neuf ) { monte( L ); depuis = 0; ++st.nb_hier; }
        else          rebranche( L );
        ++depuis;
        const double t1 = now();
        st.t_hier += t1 - t0;
        const bool ok = cg( b, d );
        st.t_res += now() - t1;
        return ok;
    }

private:
    std::vector<SI>       ord, rg;         ///< rang -> identifiant, et son inverse
    std::vector<NiveauMg> niv;
    std::vector<std::vector<SI>> carte;    ///< `carte[ l ][ i ]` : le paquet de `i` au niveau `l+1`
    std::vector<CsrMg>    prol, prolt;     ///< `P` ( fin x grossier ) et `P^t`, quand on lisse
    std::vector<TF>       cr, cz, cp, cq;  ///< les vecteurs du CG externe
    std::vector<std::vector<TF>> Uv;       ///< LE SOUS-ESPACE RECYCLE : les dernieres solutions
    std::vector<std::vector<TF>> AUv;      ///< `A U`, refait a chaque resolution ( `A` change )
    int                   depuis = 0;      ///< resolutions depuis la derniere montee
    // les tampons des assemblages, gardes d'un appel a l'autre
    std::vector<std::vector<TF>>   acc, tval;
    std::vector<std::vector<SI>>   tcol, lst;
    std::vector<std::vector<char>> mkc, mkf;
#ifdef SF_EIGEN
    // LA FACTORISATION DU FOND. La jauge y est `x[ 0 ] = 0` -- on raye la ligne et la colonne --
    // parce que le laplacien grossier est singulier ( les constantes ), et que rayer suffit ici :
    // l'operateur `P A_red^-1 P^t` reste symetrique semi-defini positif, donc CG l'accepte.
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> fgros;
    bool            gros_pret = false;
    Eigen::VectorXd egb, egx;
#endif

    /// l'indice fin du `t`-ieme membre du paquet `a`, au niveau `l` ( -1 s'il n'existe pas )
    SI membre( int l, SI a, int t, SI nf ) const {
        const SI k = SI( taille_paquet() ) * a + t;
        if ( k >= nf ) return -1;
        return ( l == 0 && ! ord.empty() ) ? ord[ k ] : k;
    }
    /// `log2( agreg )`, arrondi a la puissance de deux superieure -- toutes sont alignees sur un
    /// sous-arbre, donc toutes sont locales
    int decalage() const {
        int sh = 1;
        while ( ( 1 << sh ) < agreg && sh < 16 ) ++sh;
        return sh;
    }

    void prepare_tampons( int T, SI nc, SI nf ) {
        acc.resize( T ); tval.resize( T ); tcol.resize( T ); lst.resize( T );
        mkc.resize( T ); mkf.resize( T );
        for ( int t = 0; t < T; ++t ) {
            acc[ t ].assign( nc, TF( 0 ) );
            mkc[ t ].assign( nc, 0 );
            if ( nf > 0 ) mkf[ t ].assign( nf, 0 );
            tcol[ t ].clear(); tval[ t ].clear(); lst[ t ].clear();
        }
    }

    /// LE CSR SANS TRI NI ATOMIQUE. Chaque fil a rempli son arene dans l'ordre de SES lignes ; on
    /// n'a donc qu'a sommer les longueurs et recopier, chaque ligne sachant ou elle est.
    void assemble_csr( SI nl, const std::vector<SI> &len, const std::vector<SI> &loc,
                       const std::vector<SI> &fil,
                       std::vector<SI> &row, std::vector<SI> &col, std::vector<TF> &val ) const {
        row.assign( nl + 1, 0 );
        for ( SI i = 0; i < nl; ++i ) row[ i + 1 ] = row[ i ] + len[ i ];
        col.resize( row[ nl ] );
        val.resize( row[ nl ] );
        #pragma omp parallel for schedule( static ) if( nl >= MG_SEUIL_PAR )
        for ( SI i = 0; i < nl; ++i ) {
            const int t = fil[ i ];
            const SI  o = loc[ i ], p = row[ i ];
            for ( SI k = 0; k < len[ i ]; ++k ) { col[ p + k ] = tcol[ t ][ o + k ]; val[ p + k ] = tval[ t ][ o + k ]; }
        }
    }

    // ---------------------------------------------------------------- LA HIERARCHIE
    void rebranche( const Laplacien &L ) {
        niv[ 0 ].row = L.row.data(); niv[ 0 ].col = L.col.data();
        niv[ 0 ].val = L.c.data();   niv[ 0 ].dia = L.dia.data();
        niv[ 0 ].nnz = SI( L.col.size() );
        calcule_relax( niv[ 0 ] );                   // les valeurs ont change, les coefficients aussi
    }

    /// LE COEFFICIENT DE RELAXATION, UN PAR LIGNE, precalcule une fois. `spai0` minimise
    /// `|| I - M A ||_F` sur les `M` diagonales, ce qui donne `m_i = A_ii / somme_j A_ij^2` --
    /// un amortissement qui se regle tout seul, la ou Jacobi demande un `omega` devine.
    void calcule_relax( NiveauMg &v ) const {
        const SI n = v.n;
        v.rlx.resize( n );
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) {
            const TF d = v.dia[ i ];
            if ( lisseur == 0 ) { v.rlx[ i ] = d > 0 ? omega / d : TF( 0 ); continue; }
            TF q = d * d;                                // `spai0`, aussi pour l'interieur de Chebyshev
            for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) q += v.val[ e ] * v.val[ e ];
            v.rlx[ i ] = q > 0 ? d / q : TF( 0 );
        }
        if ( lisseur != 2 )
            return;
        // GERSHGORIN SUR `M^-1 A` : exact, et une passe sur les aretes.
        TF mx = 0;
        #pragma omp parallel for schedule( static ) reduction( max : mx ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) {
            TF s = v.dia[ i ];
            for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) s += std::fabs( v.val[ e ] );
            mx = std::max( mx, v.rlx[ i ] * s );
        }
        v.lmax = mx > 0 ? mx : TF( 2 );
    }

    /// LE LISSEUR DE CHEBYSHEV, la recurrence a trois termes. `deg` produits matrice-vecteur,
    /// exactement comme `deg` balayages de Jacobi -- mais un polynome choisi pour ecraser la
    /// partie haute du spectre, celle que le niveau grossier ne voit pas.
    void chebyshev( NiveauMg &v, int deg, bool net ) const {
        const SI n = v.n;
        if ( deg <= 0 ) { if ( net ) std::fill( v.x.begin(), v.x.end(), TF( 0 ) ); return; }
        const TF hi = v.lmax, lo = hi / std::max( cheb, TF( 1.01 ) );
        const TF th = ( hi + lo ) / 2, de = ( hi - lo ) / 2;
        const TF si = th / de;
        TF rh = 1 / si;
        if ( net ) {
            std::fill( v.x.begin(), v.x.end(), TF( 0 ) );
            v.r = v.b;
        } else {
            #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
            for ( SI i = 0; i < n; ++i ) {
                TF s = v.dia[ i ] * v.x[ i ];
                for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) s -= v.val[ e ] * v.x[ v.col[ e ] ];
                v.r[ i ] = v.b[ i ] - s;
            }
        }
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) v.y[ i ] = v.rlx[ i ] * v.r[ i ] / th;     // `y` porte la direction
        for ( int k = 0; k < deg; ++k ) {
            #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
            for ( SI i = 0; i < n; ++i ) v.x[ i ] += v.y[ i ];
            if ( k + 1 == deg ) break;
            // UNE SEULE PASSE, ET LE DOUBLE TAMPON RESTE NECESSAIRE. Le produit `A y` de la ligne
            // `i` lit `y` chez les voisins : on ne peut donc pas ecrire la nouvelle direction dans
            // `y`. Mais on n'a pas besoin d'une passe de plus pour autant -- on lit `y`, on met
            // `r` a jour sur place ( chacun son indice ), et on ecrit la direction suivante dans
            // `z`, qu'on echange. Mesure de l'erreur inverse : avec deux passes, Chebyshev rendait
            // en cout par iteration ( 0.467 ms contre 0.418 ) ce qu'il gagnait en nombre.
            const TF r2 = 1 / ( 2 * si - rh ), c1 = r2 * rh, c2 = 2 * r2 / de;
            #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
            for ( SI i = 0; i < n; ++i ) {
                TF s = v.dia[ i ] * v.y[ i ];
                for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) s -= v.val[ e ] * v.y[ v.col[ e ] ];
                const TF ri = v.r[ i ] - s;
                v.r[ i ] = ri;
                v.z[ i ] = c1 * v.y[ i ] + c2 * v.rlx[ i ] * ri;
            }
            v.y.swap( v.z );
            rh = r2;
        }
    }

    /// le lissage demande, quel qu'il soit
    void lisse_un( NiveauMg &v, int nb, bool net ) const {
        if ( lisseur == 2 ) chebyshev( v, nb, net );
        else                jacobi( v, nb, net );
    }

    void monte( const Laplacien &L ) {
        niv.clear(); carte.clear(); prol.clear(); prolt.clear();

        NiveauMg f;
        f.n = L.n; f.nnz = SI( L.col.size() );
        f.row = L.row.data(); f.col = L.col.data(); f.val = L.c.data(); f.dia = L.dia.data();
        niv.push_back( std::move( f ) );

        for ( int l = 0; niv[ l ].n > stop && l < 24; ++l )
            grossit( l );

        for ( NiveauMg &v : niv ) {
            v.x.assign( v.n, TF( 0 ) ); v.b.assign( v.n, TF( 0 ) );
            v.r.assign( v.n, TF( 0 ) ); v.y.assign( v.n, TF( 0 ) );
            if ( lisseur == 2 ) v.z.assign( v.n, TF( 0 ) );
            calcule_relax( v );
        }
        for ( int l = 1; l <= kcycle && l < int( niv.size() ); ++l ) {
            NiveauMg &v = niv[ l ];
            v.v1.assign( v.n, TF( 0 ) ); v.v2.assign( v.n, TF( 0 ) );
            v.t.assign( v.n, TF( 0 ) );  v.rc.assign( v.n, TF( 0 ) );
        }
        factorise_le_fond();
        if ( trace ) {
            trace = 0;                               // une fois suffit
            SI tot = 0;
            std::printf( "   multigrille ( paquets de %d, %s ) : ", taille_paquet(),
                         lisseur ? "spai0" : "Jacobi amorti" );
            for ( size_t l = 0; l < niv.size(); ++l ) {
                tot += niv[ l ].nnz;
                std::printf( "%s%d ( %.1f nz/l )", l ? " -> " : "", int( niv[ l ].n ),
                             niv[ l ].n ? double( niv[ l ].nnz ) / double( niv[ l ].n ) : 0.0 );
            }
            std::printf( "  --  complexite d'operateur %.2f\n",
                         niv[ 0 ].nnz ? double( tot ) / double( niv[ 0 ].nnz ) : 0.0 );
        }
    }

    void grossit( int l ) {
        const SI S = taille_paquet();
        const SI nf = niv[ l ].n, nc = ( nf + S - 1 ) / S;

        std::vector<SI> &m = carte.emplace_back();     // la carte `fin -> paquet`
        m.resize( nf );
        {
            const std::vector<SI> &r = rg;
            const bool par_rang = ( l == 0 && ! r.empty() );
            const int sh = decalage();
            #pragma omp parallel for schedule( static ) if( nf >= MG_SEUIL_PAR )
            for ( SI i = 0; i < nf; ++i ) m[ i ] = ( par_rang ? r[ i ] : i ) >> sh;
        }
        prol.emplace_back();
        prolt.emplace_back();
        if ( lisse ) galerkin_lisse( l, m, nc );
        else         galerkin_brut( l, m, nc );
    }

    /// LE GALERKIN DE L'AGREGATION BRUTE : `P0` constante par morceaux, donc `A_c` s'obtient en
    /// sommant les poids d'aretes entre paquets. Une ligne grossiere par fil, pas de tri.
    void galerkin_brut( int l, const std::vector<SI> &m, SI nc ) {
        const NiveauMg &g = niv[ l ];
        const SI nf = g.n;
        const int T = omp_get_max_threads();
        prepare_tampons( T, nc, 0 );
        std::vector<SI> len( nc ), loc( nc ), fil( nc );

        #pragma omp parallel for schedule( static ) if( nc >= MG_SEUIL_PAR )
        for ( SI a = 0; a < nc; ++a ) {
            const int t = omp_get_thread_num();
            std::vector<TF> &ac = acc[ t ];
            std::vector<char> &mk = mkc[ t ];
            std::vector<SI> &tc = tcol[ t ];
            std::vector<TF> &tv = tval[ t ];
            const SI deb = SI( tc.size() );
            for ( int q = 0; q < taille_paquet(); ++q ) {
                const SI i = membre( l, a, q, nf );
                if ( i < 0 ) continue;
                for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) {
                    const SI b = m[ g.col[ e ] ];
                    if ( b == a ) continue;              // l'interieur du paquet disparait
                    if ( ! mk[ b ] ) { mk[ b ] = 1; tc.push_back( b ); }
                    ac[ b ] += g.val[ e ];
                }
            }
            for ( SI k = deb; k < SI( tc.size() ); ++k ) {
                tv.push_back( ac[ tc[ k ] ] );
                ac[ tc[ k ] ] = 0; mk[ tc[ k ] ] = 0;
            }
            loc[ a ] = deb; len[ a ] = SI( tc.size() ) - deb; fil[ a ] = t;
        }

        NiveauMg c;
        c.n = nc;
        assemble_csr( nc, len, loc, fil, c.arow, c.acol, c.aval );
        c.nnz = c.arow[ nc ];
        c.adia.assign( nc, TF( 0 ) );
        #pragma omp parallel for schedule( static ) if( nc >= MG_SEUIL_PAR )
        for ( SI a = 0; a < nc; ++a ) {
            TF s = 0;
            for ( SI e = c.arow[ a ]; e < c.arow[ a + 1 ]; ++e ) s += c.aval[ e ];
            c.adia[ a ] = s > 0 ? s : TF( 1 );           // une ligne nulle rendrait Jacobi fou
        }
        pose( std::move( c ) );
    }

    /// LE GALERKIN DE LA PROLONGATION LISSEE : `P`, `P^t`, `AP = A P`, puis `A_c = P^t ( AP )`.
    void galerkin_lisse( int l, const std::vector<SI> &m, SI nc ) {
        const NiveauMg &g = niv[ l ];
        const SI nf = g.n;
        const int T = omp_get_max_threads();
        CsrMg &P = prol.back(), &Pt = prolt.back();

        // ---- 1. `P = ( I - w D^-1 A ) P0`, une ligne FINE par fil
        {
            prepare_tampons( T, nc, 0 );
            std::vector<SI> len( nf ), loc( nf ), fil( nf );
            #pragma omp parallel for schedule( static ) if( nf >= MG_SEUIL_PAR )
            for ( SI i = 0; i < nf; ++i ) {
                const int t = omp_get_thread_num();
                std::vector<TF> &ac = acc[ t ];
                std::vector<char> &mk = mkc[ t ];
                std::vector<SI> &tc = tcol[ t ];
                std::vector<TF> &tv = tval[ t ];
                const SI deb = SI( tc.size() );
                // LES CONNEXIONS FORTES, ET ELLES SEULES. Le reste est reporte sur la diagonale
                // ( `diaf` ne somme que les fortes ), donc la ligne du `A` filtre somme toujours a
                // zero -- et la ligne de `P` somme toujours a un.
                TF mxc = 0;
                for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) mxc = std::max( mxc, g.val[ e ] );
                const TF sfo = force * mxc;
                TF diaf = 0;
                for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e )
                    if ( g.val[ e ] >= sfo ) diaf += g.val[ e ];
                const SI ai = m[ i ];
                if ( ! mk[ ai ] ) { mk[ ai ] = 1; tc.push_back( ai ); }
                ac[ ai ] += 1 - omega_p;
                const TF f = diaf > 0 ? omega_p / diaf : TF( 0 );
                for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) {
                    if ( ! ( g.val[ e ] >= sfo ) ) continue;
                    const SI b = m[ g.col[ e ] ];
                    if ( ! mk[ b ] ) { mk[ b ] = 1; tc.push_back( b ); }
                    ac[ b ] += f * g.val[ e ];
                }
                // LA TRONCATURE, PUIS LA RENORMALISATION. On garde les entrees qui pesent, et on
                // remet la somme a un : c'est ce qui laisse le vecteur constant dans l'image de `P`.
                TF mx = 0;
                for ( SI k = deb; k < SI( tc.size() ); ++k ) mx = std::max( mx, std::fabs( ac[ tc[ k ] ] ) );
                const TF seuil = tronque * mx;
                SI garde = deb;
                TF som = 0;
                for ( SI k = deb; k < SI( tc.size() ); ++k ) {
                    const SI b = tc[ k ];
                    const TF v = ac[ b ];
                    ac[ b ] = 0; mk[ b ] = 0;
                    if ( ! ( std::fabs( v ) >= seuil ) || v == 0 ) continue;
                    tc[ garde ] = b; tv.push_back( v ); ++garde; som += v;
                }
                tc.resize( garde );
                if ( som > 0 )
                    for ( SI k = deb; k < SI( tv.size() ); ++k ) tv[ k ] /= som;
                loc[ i ] = deb; len[ i ] = garde - deb; fil[ i ] = t;
            }
            assemble_csr( nf, len, loc, fil, P.row, P.col, P.val );
            P.lignes = nf; P.colonnes = nc;
        }

        // ---- 2. `P^t`, SANS PASSE DE TRANSPOSITION. `P[ i ][ a ] != 0` demande `m_i = a` ou un
        //         voisin de `i` dans le paquet `a` : la ligne `a` de `P^t` est donc portee par les
        //         membres du paquet et leurs voisins, que la carte inverse donne gratuitement.
        {
            prepare_tampons( T, nc, nf );
            std::vector<SI> len( nc ), loc( nc ), fil( nc );
            #pragma omp parallel for schedule( static ) if( nc >= MG_SEUIL_PAR )
            for ( SI a = 0; a < nc; ++a ) {
                const int t = omp_get_thread_num();
                std::vector<char> &mk = mkf[ t ];
                std::vector<SI> &ls = lst[ t ];
                std::vector<SI> &tc = tcol[ t ];
                std::vector<TF> &tv = tval[ t ];
                ls.clear();
                for ( int q = 0; q < taille_paquet(); ++q ) {
                    const SI i = membre( l, a, q, nf );
                    if ( i < 0 ) continue;
                    if ( ! mk[ i ] ) { mk[ i ] = 1; ls.push_back( i ); }
                    for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) {
                        const SI j = g.col[ e ];
                        if ( ! mk[ j ] ) { mk[ j ] = 1; ls.push_back( j ); }
                    }
                }
                const SI deb = SI( tc.size() );
                for ( SI i : ls ) {
                    mk[ i ] = 0;
                    for ( SI e = P.row[ i ]; e < P.row[ i + 1 ]; ++e )
                        if ( P.col[ e ] == a ) { tc.push_back( i ); tv.push_back( P.val[ e ] ); break; }
                }
                loc[ a ] = deb; len[ a ] = SI( tc.size() ) - deb; fil[ a ] = t;
            }
            assemble_csr( nc, len, loc, fil, Pt.row, Pt.col, Pt.val );
            Pt.lignes = nc; Pt.colonnes = nf;
        }

        // ---- 3. `AP = A P`, une ligne FINE par fil
        CsrMg AP;
        {
            prepare_tampons( T, nc, 0 );
            std::vector<SI> len( nf ), loc( nf ), fil( nf );
            #pragma omp parallel for schedule( static ) if( nf >= MG_SEUIL_PAR )
            for ( SI i = 0; i < nf; ++i ) {
                const int t = omp_get_thread_num();
                std::vector<TF> &ac = acc[ t ];
                std::vector<char> &mk = mkc[ t ];
                std::vector<SI> &tc = tcol[ t ];
                std::vector<TF> &tv = tval[ t ];
                const SI deb = SI( tc.size() );
                for ( SI e = P.row[ i ]; e < P.row[ i + 1 ]; ++e ) {   // `dia_i P[ i ][ . ]`
                    const SI b = P.col[ e ];
                    if ( ! mk[ b ] ) { mk[ b ] = 1; tc.push_back( b ); }
                    ac[ b ] += g.dia[ i ] * P.val[ e ];
                }
                for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) {   // `- somme_j c_ij P[ j ][ . ]`
                    const SI j = g.col[ e ];
                    const TF c = g.val[ e ];
                    for ( SI f = P.row[ j ]; f < P.row[ j + 1 ]; ++f ) {
                        const SI b = P.col[ f ];
                        if ( ! mk[ b ] ) { mk[ b ] = 1; tc.push_back( b ); }
                        ac[ b ] -= c * P.val[ f ];
                    }
                }
                for ( SI k = deb; k < SI( tc.size() ); ++k ) {
                    tv.push_back( ac[ tc[ k ] ] );
                    ac[ tc[ k ] ] = 0; mk[ tc[ k ] ] = 0;
                }
                loc[ i ] = deb; len[ i ] = SI( tc.size() ) - deb; fil[ i ] = t;
            }
            assemble_csr( nf, len, loc, fil, AP.row, AP.col, AP.val );
            AP.lignes = nf; AP.colonnes = nc;
        }

        // ---- 4. `A_c = P^t ( AP )`, une ligne GROSSIERE par fil
        NiveauMg c;
        c.n = nc;
        {
            prepare_tampons( T, nc, 0 );
            std::vector<SI> len( nc ), loc( nc ), fil( nc );
            std::vector<TF> dia( nc, TF( 0 ) );
            #pragma omp parallel for schedule( static ) if( nc >= MG_SEUIL_PAR )
            for ( SI a = 0; a < nc; ++a ) {
                const int t = omp_get_thread_num();
                std::vector<TF> &ac = acc[ t ];
                std::vector<char> &mk = mkc[ t ];
                std::vector<SI> &tc = tcol[ t ];
                std::vector<TF> &tv = tval[ t ];
                const SI deb = SI( tc.size() );
                for ( SI k = Pt.row[ a ]; k < Pt.row[ a + 1 ]; ++k ) {
                    const SI i = Pt.col[ k ];
                    const TF w = Pt.val[ k ];
                    for ( SI f = AP.row[ i ]; f < AP.row[ i + 1 ]; ++f ) {
                        const SI b = AP.col[ f ];
                        if ( ! mk[ b ] ) { mk[ b ] = 1; tc.push_back( b ); }
                        ac[ b ] += w * AP.val[ f ];
                    }
                }
                // LA DIAGONALE SORT DE LA LISTE, et les hors-diagonaux changent de signe : notre
                // convention est `y = dia x - somme val x`, donc `val_ab = - A_c[ a ][ b ]`.
                SI garde = deb;
                for ( SI k = deb; k < SI( tc.size() ); ++k ) {
                    const SI b = tc[ k ];
                    const TF v = ac[ b ];
                    ac[ b ] = 0; mk[ b ] = 0;
                    if ( b == a ) { dia[ a ] = v; continue; }
                    tc[ garde ] = b; tv.push_back( -v ); ++garde;
                }
                tc.resize( garde );
                loc[ a ] = deb; len[ a ] = garde - deb; fil[ a ] = t;
            }
            assemble_csr( nc, len, loc, fil, c.arow, c.acol, c.aval );
            c.nnz = c.arow[ nc ];
            c.adia.resize( nc );
            #pragma omp parallel for schedule( static ) if( nc >= MG_SEUIL_PAR )
            for ( SI a = 0; a < nc; ++a ) c.adia[ a ] = dia[ a ] > 0 ? dia[ a ] : TF( 1 );
        }
        pose( std::move( c ) );
    }

    void pose( NiveauMg &&c ) {
        niv.push_back( std::move( c ) );
        NiveauMg &v = niv.back();                    // apres le deplacement : les tampons ont bouge
        v.row = v.arow.data(); v.col = v.acol.data(); v.val = v.aval.data(); v.dia = v.adia.data();
    }

    /// le niveau le plus grossier, factorise UNE FOIS par hierarchie
    void factorise_le_fond() {
#ifdef SF_EIGEN
        gros_pret = false;
        if ( ! exact )
            return;
        const NiveauMg &g = niv.back();
        const SI m = g.n - 1;
        if ( m <= 0 )
            return;
        std::vector<Eigen::Triplet<double>> tr;
        tr.reserve( size_t( g.nnz + m ) );
        for ( SI i = 1; i < g.n; ++i ) {
            tr.emplace_back( int( i - 1 ), int( i - 1 ), double( g.dia[ i ] ) );
            for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e )
                if ( g.col[ e ] >= 1 )
                    tr.emplace_back( int( i - 1 ), int( g.col[ e ] - 1 ), -double( g.val[ e ] ) );
        }
        Eigen::SparseMatrix<double> A( m, m );
        A.setFromTriplets( tr.begin(), tr.end() );
        fgros.compute( A );
        gros_pret = fgros.info() == Eigen::Success;
        egb.resize( m ); egx.resize( m );
#endif
    }

    // ---------------------------------------------------------------- LES BRIQUES
    static void matvec( const NiveauMg &v, const std::vector<TF> &x, std::vector<TF> &y ) {
        const SI n = v.n;
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) {
            TF s = v.dia[ i ] * x[ i ];
            for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) s -= v.val[ e ] * x[ v.col[ e ] ];
            y[ i ] = s;
        }
    }
    /// `nb` lissages, DEUX TAMPONS ; `net` : on part de `x = 0`. Le coefficient par ligne est
    /// precalcule ( `rlx` ), donc la boucle chaude n'a plus de division.
    void jacobi( NiveauMg &v, int nb, bool net ) const {
        const SI n = v.n;
        if ( net ) std::fill( v.x.begin(), v.x.end(), TF( 0 ) );
        for ( int k = 0; k < nb; ++k ) {
            const TF *xi = v.x.data();
            TF *xo = v.y.data();
            #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
            for ( SI i = 0; i < n; ++i ) {
                TF s = v.dia[ i ] * xi[ i ];
                for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) s -= v.val[ e ] * xi[ v.col[ e ] ];
                xo[ i ] = xi[ i ] + v.rlx[ i ] * ( v.b[ i ] - s );
            }
            v.x.swap( v.y );
        }
    }
    static TF dot( const std::vector<TF> &u, const std::vector<TF> &v ) {
        const SI n = SI( u.size() );
        TF s = 0;
        #pragma omp parallel for schedule( static ) reduction( + : s ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) s += u[ i ] * v[ i ];
        return s;
    }

    // ---------------------------------------------------------------- LE CYCLE
    void cycle( int l ) {
        NiveauMg &g = niv[ l ];
        if ( l + 1 == int( niv.size() ) ) {
#ifdef SF_EIGEN
            if ( gros_pret ) {
                for ( SI i = 1; i < g.n; ++i ) egb[ i - 1 ] = double( g.b[ i ] );
                egx = fgros.solve( egb );
                g.x[ 0 ] = 0;
                for ( SI i = 1; i < g.n; ++i ) g.x[ i ] = TF( egx[ i - 1 ] );
                return;
            }
#endif
            lisse_un( g, gros, true );
            return;
        }
        NiveauMg &c = niv[ l + 1 ];
        const SI nf = g.n, ncc = c.n;

        lisse_un( g, nu, true );
        #pragma omp parallel for schedule( static ) if( nf >= MG_SEUIL_PAR )
        for ( SI i = 0; i < nf; ++i ) {                  // `r = b - A x`
            TF s = g.dia[ i ] * g.x[ i ];
            for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) s -= g.val[ e ] * g.x[ g.col[ e ] ];
            g.r[ i ] = g.b[ i ] - s;
        }
        // LA RESTRICTION SE FAIT PAR RAMASSAGE, jamais par dispersion : pas d'atomique. Sans
        // lissage le paquet connait ses membres ; avec, `P^t` est la et c'est un produit CSR.
        const CsrMg &Pt = prolt[ l ];
        if ( Pt.vrai() ) {
            #pragma omp parallel for schedule( static ) if( ncc >= MG_SEUIL_PAR )
            for ( SI a = 0; a < ncc; ++a ) {
                TF s = 0;
                for ( SI k = Pt.row[ a ]; k < Pt.row[ a + 1 ]; ++k ) s += Pt.val[ k ] * g.r[ Pt.col[ k ] ];
                c.b[ a ] = s;
            }
        } else {
            #pragma omp parallel for schedule( static ) if( ncc >= MG_SEUIL_PAR )
            for ( SI a = 0; a < ncc; ++a ) {
                TF s = 0;
                for ( int q = 0; q < taille_paquet(); ++q ) { const SI i = membre( l, a, q, nf ); if ( i >= 0 ) s += g.r[ i ]; }
                c.b[ a ] = s;
            }
        }

        if ( l >= kcycle ) cycle( l + 1 );
        else               kcycle_deux( l + 1 );

        const CsrMg &P = prol[ l ];
        if ( P.vrai() ) {
            #pragma omp parallel for schedule( static ) if( nf >= MG_SEUIL_PAR )
            for ( SI i = 0; i < nf; ++i ) {
                TF s = 0;
                for ( SI e = P.row[ i ]; e < P.row[ i + 1 ]; ++e ) s += P.val[ e ] * c.x[ P.col[ e ] ];
                g.x[ i ] += s;
            }
        } else {
            const std::vector<SI> &m = carte[ l ];
            #pragma omp parallel for schedule( static ) if( nf >= MG_SEUIL_PAR )
            for ( SI i = 0; i < nf; ++i ) g.x[ i ] += c.x[ m[ i ] ];
        }
        lisse_un( g, nu, false );
    }

    /// DEUX PAS DE GRADIENT CONJUGUE SUR LE NIVEAU `l`, preconditionnes par le niveau d'en
    /// dessous. C'est le K-cycle : on ne change pas `P`, on accelere chaque niveau.
    void kcycle_deux( int l ) {
        NiveauMg &c = niv[ l ];
        const SI n = c.n;
        c.rc = c.b;                                      // le second membre du premier pas
        cycle( l );                                      // `v1 = M^-1 b`
        c.v1 = c.x;
        matvec( c, c.v1, c.t );                          // `t = A v1`
        const TF rho1 = dot( c.v1, c.t );
        const TF a1   = dot( c.v1, c.rc );
        const TF k1   = rho1 != 0 ? a1 / rho1 : TF( 0 );
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) c.b[ i ] = c.rc[ i ] - k1 * c.t[ i ];
        cycle( l );                                      // `v2 = M^-1 r`
        c.v2 = c.x;
        const TF g2 = dot( c.v2, c.t );                  // `t` porte encore `A v1`
        const TF a2 = dot( c.v2, c.b );
        matvec( c, c.v2, c.t );
        const TF b2   = dot( c.v2, c.t );
        const TF rho2 = b2 - ( rho1 != 0 ? g2 * g2 / rho1 : TF( 0 ) );
        const TF k2   = rho2 != 0 ? a2 / rho2 : TF( 0 );
        const TF k1c  = k1 - ( rho1 != 0 && rho2 != 0 ? g2 * a2 / ( rho1 * rho2 ) : TF( 0 ) );
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) c.x[ i ] = k1c * c.v1[ i ] + k2 * c.v2[ i ];
        c.b.swap( c.rc );                                // rendu tel qu'on l'a trouve
    }

    /// LE PETIT SYSTEME DENSE `G y = f`, `G` symetrique definie positive et `k <= 32`. Cholesky
    /// a la main, avec une crete de securite : deux solutions successives peuvent etre presque
    /// colineaires, et `G` devient alors singuliere -- on decline plutot que de rendre n'importe
    /// quoi, le demarrage n'etant qu'un bonus.
    static bool resout_dense( std::vector<TF> &G, std::vector<TF> &f, int k ) {
        TF tr = 0;
        for ( int j = 0; j < k; ++j ) tr += G[ size_t( j ) * k + j ];
        if ( ! ( tr > 0 ) ) return false;
        const TF eps = tr / TF( k ) * TF( 1e-12 );
        for ( int j = 0; j < k; ++j ) G[ size_t( j ) * k + j ] += eps;
        for ( int j = 0; j < k; ++j ) {                  // Cholesky en place, triangle inferieur
            TF s = G[ size_t( j ) * k + j ];
            for ( int q = 0; q < j; ++q ) s -= G[ size_t( j ) * k + q ] * G[ size_t( j ) * k + q ];
            if ( ! ( s > 0 ) ) return false;
            const TF dj = std::sqrt( s );
            G[ size_t( j ) * k + j ] = dj;
            for ( int i = j + 1; i < k; ++i ) {
                TF t = G[ size_t( i ) * k + j ];
                for ( int q = 0; q < j; ++q ) t -= G[ size_t( i ) * k + q ] * G[ size_t( j ) * k + q ];
                G[ size_t( i ) * k + j ] = t / dj;
            }
        }
        for ( int i = 0; i < k; ++i ) {                  // descente
            TF t = f[ i ];
            for ( int q = 0; q < i; ++q ) t -= G[ size_t( i ) * k + q ] * f[ q ];
            f[ i ] = t / G[ size_t( i ) * k + i ];
        }
        for ( int i = k - 1; i >= 0; --i ) {             // remontee
            TF t = f[ i ];
            for ( int q = i + 1; q < k; ++q ) t -= G[ size_t( q ) * k + i ] * f[ q ];
            f[ i ] = t / G[ size_t( i ) * k + i ];
        }
        return true;
    }

    /// la jauge : moyenne nulle, le noyau du laplacien
    static void centre( std::vector<TF> &v ) {
        const SI n = SI( v.size() );
        if ( n <= 0 ) return;
        TF s = 0;
        #pragma omp parallel for schedule( static ) reduction( + : s ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) s += v[ i ];
        const TF mu = s / TF( n );
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) v[ i ] -= mu;
    }

    // ---------------------------------------------------------------- LE CG EXTERNE
    bool cg( const std::vector<TF> &b, std::vector<TF> &d ) {
        const SI n = niv[ 0 ].n;
        d.assign( n, TF( 0 ) );
        cr.assign( b.begin(), b.begin() + n );
        centre( cr );
        cz.assign( n, TF( 0 ) ); cp.assign( n, TF( 0 ) ); cq.assign( n, TF( 0 ) );

        auto precond = [ & ]( const std::vector<TF> &rr, std::vector<TF> &zz ) {
            if ( niv.size() > 1 ) {
                niv[ 0 ].b = rr;
                cycle( 0 );
                zz = niv[ 0 ].x;
            } else {
                const TF *rl = niv[ 0 ].rlx.data();
                #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
                for ( SI i = 0; i < n; ++i ) zz[ i ] = rl[ i ] * rr[ i ];
            }
            centre( zz );
        };

        const TF bb = dot( cr, cr );
        if ( ! ( bb > 0 ) ) return true;                 // rien a resoudre

        // ---- LE DEMARRAGE DE GALERKIN SUR LE SOUS-ESPACE RECYCLE
        //
        // `x0 = U ( U^t A U )^-1 U^t b` est la MEILLEURE approximation dans `span( U )` au sens
        // de l'energie, et son residu sort du calcul deja fait : `r0 = b - ( AU ) y`. Le prix est
        // `k` produits matrice-vecteur, a comparer aux iterations qu'on espere epargner.
        if ( recycle > 0 && ! Uv.empty() ) {
            const int k = int( Uv.size() );
            AUv.resize( k );
            for ( int j = 0; j < k; ++j ) {
                AUv[ j ].resize( n );
                matvec( niv[ 0 ], Uv[ j ], AUv[ j ] );
            }
            std::vector<TF> G( size_t( k ) * k ), f( k );
            for ( int j = 0; j < k; ++j ) {
                f[ j ] = dot( Uv[ j ], cr );
                for ( int l = 0; l <= j; ++l ) {
                    const TF g = dot( Uv[ j ], AUv[ l ] );
                    G[ size_t( j ) * k + l ] = g;
                    G[ size_t( l ) * k + j ] = g;
                }
            }
            if ( resout_dense( G, f, k ) ) {
                #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
                for ( SI i = 0; i < n; ++i ) {
                    TF sx = 0, sr = 0;
                    for ( int j = 0; j < k; ++j ) { sx += f[ j ] * Uv[ j ][ i ]; sr += f[ j ] * AUv[ j ][ i ]; }
                    d[ i ] = sx;
                    cr[ i ] -= sr;
                }
                centre( d );
                centre( cr );
            }
        }

        precond( cr, cz );
        cp = cz;
        TF rz = dot( cr, cz );
        const TF cible = tol * tol * bb;
        TF rr = bb;
        int it = 0;
        for ( ; it < maxit && rr > cible; ++it ) {
            matvec( niv[ 0 ], cp, cq );
            const TF pq = dot( cp, cq );
            if ( ! ( pq > 0 ) ) break;                   // la direction est dans le noyau : fini
            const TF al = rz / pq;
            TF s = 0;
            #pragma omp parallel for schedule( static ) reduction( + : s ) if( n >= MG_SEUIL_PAR )
            for ( SI i = 0; i < n; ++i ) {
                d[ i ] += al * cp[ i ];
                cr[ i ] -= al * cq[ i ];
                s += cr[ i ] * cr[ i ];
            }
            rr = s;
            if ( rr <= cible ) { ++it; break; }
            precond( cr, cz );
            const TF rz2 = dot( cr, cz );
            const TF be = rz != 0 ? rz2 / rz : TF( 0 );
            rz = rz2;
            #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
            for ( SI i = 0; i < n; ++i ) cp[ i ] = cz[ i ] + be * cp[ i ];
        }
        st.nb_iter += it;
        // LA SOLUTION REJOINT LE SOUS-ESPACE, a moyenne nulle -- c'est la jauge dans laquelle on
        // resout, et donc celle dans laquelle `U` doit vivre.
        if ( recycle > 0 ) {
            if ( int( Uv.size() ) >= recycle ) Uv.erase( Uv.begin() );
            Uv.push_back( d );
        }
        // ON REND LA JAUGE `d[ 0 ] = 0` : voir l'entete. Les deux jauges decrivent la meme
        // direction a une constante pres, mais `Newton.h` ecrase `w2[ 0 ]` apres le pas.
        const TF d0 = d[ 0 ];
        #pragma omp parallel for schedule( static ) if( n >= MG_SEUIL_PAR )
        for ( SI i = 0; i < n; ++i ) d[ i ] -= d0;
        const TF err = std::sqrt( rr / bb );
        st.pire = std::max( st.pire, err );
        return err < 1;
    }
};

} // namespace sf
