#pragma once

// =====================================================================================
// LE TRANSPORT SEMI-DISCRET, RESOLU : trouver `w` tel que `|Lag_i( w )| = nu_i` pour tout `i`.
//
// = Le dual, et pourquoi Newton
//
//     Phi( w ) = integrale min_i ( |x - p_i|^2 - w_i ) dx + sum_i w_i nu_i
//
// est CONCAVE, de gradient `nu_i - |Lag_i( w )|`, et sa hessienne est au signe pres le laplacien
// du graphe de Laguerre ( `Laplacien.h` ). Son noyau est les constantes : on fixe `w_0 = 0`.
//
// = L'amortissement ( Kitagawa-Merigot-Thibert ), et ce qu'il protege
//
// La hessienne n'est definie que tant qu'aucune cellule n'est vide. Le pas essaye doit donc
// garder toute aire au-dessus d'un plancher `eps` fixe au depart, et faire decroitre le residu
// d'au moins `1 - t / 2`. On part de `w = 0`, le diagramme de VORONOI, dont aucune cellule n'est
// vide : le depart est toujours admissible.
//
// La decroissance est demandee STRICTE ( `n2 < nr` ) : sans cela un pas qui tend vers zero passe
// le test par egalite des que `t` est negligeable, et Newton tourne sur place indefiniment
// ( mesure : 54 diagrammes par iteration a residu constant ). On sort alors en STAGNATION -- le
// plancher numerique, pas un echec, et la difference se lit sur `reste`.
//
// = Le pas par les limites ( `pas != ESSAIS`, 2D )
//
// Au lieu d'essayer `t = 1, 1/2, 1/4, ...` a un diagramme l'essai, on calcule la LIMITE de
// chaque cellule le long de `d` ( `Ecrasement.h` : le polynome predit, une cellule exacte
// verifie et corrige ), et `alpha* = min_i` dit jusqu'ou on peut aller sans passer sous `eps`.
// Le pas est la puissance de deux sous `alpha*` ( DYADIQUE ) ou `facteur * alpha*` ( FACTEUR ),
// et le diagramme de ce pas -- qu'il faut de toute facon pour l'iteration suivante -- confirme
// la decroissance du residu. S'il refuse ( residu, ou une cellule non monotone ), on recule
// comme avant depuis la.
//
// = Ce que coute une iteration
//
// UN diagramme par pas essaye, et rien de plus : le pas accepte livre a la fois les mesures ( le
// residu ) et les facettes ( la hessienne suivante ). Le temps est compte par poste -- majorants,
// diagrammes, assemblage, resolution -- parce que c'est la REPARTITION qu'on veut lire.
// =====================================================================================

#include "diagram/PowerDiagram.h"
#include "solver/Cible.h"
#include "solver/Densite.h"
#include "solver/Ecrasement.h"
#include "solver/Laplacien.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

namespace sf {

struct NewtonOptions {
    TF   tol        = 1e-6;    ///< arret : `max_i |a_i - nu_i| <= tol * nu_i`
    int  maxit      = 100;
    int  max_reculs = 60;      ///< divisions par deux du pas, au plus, par iteration
    TF   t_min      = 1e-10;   ///< en dessous, on declare la STAGNATION ( 34 diagrammes pour y
                               ///< descendre depuis 1 : `1e-3` en coute 10 )
    /// LE PROGRES MINIMAL PAR ITERATION, en fraction de `|r|_2`. `0` : eteint.
    ///
    /// `t_min` ne suffit pas a arreter une agonie, parce qu'une iteration mourante ACCEPTE son
    /// pas -- elle gagne un pour cent sur le residu pour huit diagrammes. Mesure sur l'uniforme
    /// 3D a `n = 1e5` en `--kernel float` : les six premieres iterations sont identiques a celles
    /// du `double` et coutent NEUF diagrammes ; les quatorze suivantes en coutent CENT QUARANTE
    /// CINQ pour faire passer `|r|_2` de 4.3e-10 a 2.0e-10. En simple precision c'est le regime
    /// normal de fin de course -- le residu a atteint le bruit de la mesure -- donc il faut un
    /// critere qui le voie : une iteration qui gagne moins que `progres_min` est la derniere.
    TF   progres_min = 0;
    /// LE PLANCHER DE BRUIT DE LA MESURE, en unites de `|r|_2`. `0` : eteint.
    ///
    /// Il est PREVISIBLE et ne coute rien a poser : l'ecart de mesure d'une cellule vaut
    /// `kappa eps nu_i`, donc `|bruit|_2 = kappa eps sqrt( n ) nu = kappa eps / sqrt( n )`. A
    /// `n = 1e5` en `float` ca donne 1.9e-10 -- et la trace montre Newton bloque a 2.0e-10.
    /// S'arreter LA, plutot que d'y descendre a coups de demi-pas, est ce qui rend la bascule
    /// `--kernel mixte` rentable : c'est la difference entre rendre la main apres SIX diagrammes
    /// inutiles et apres CENT QUARANTE CINQ.
    TF   plancher = 0;
    bool trace      = true;
    int  extraire   = -1;      ///< >= 0 : s'arreter des que la DIRECTION de cette iteration est
                               ///< calculee ( `w` et `d` sont alors ceux du pas propose )
    /// `MERITE` : le pas qui MINIMISE le merite le long de la direction, au lieu du premier pas qui le
    /// fait decroitre assez. C'est la question posee dans l'autre sens : si le but est de minimiser le
    /// merite `log`, autant le minimiser -- d'autant que le profil ( § 21.2 ) montre qu'il a un vrai
    /// minimum interieur, et que ce minimum est SOUS le pas que le plancher d'aire autorise.
    ///
    /// Attention, ca sort de la theorie KMT : celle-ci exige la decroissance `1 - t/2` d'une norme `l2`
    /// du residu, et minimiser n'est pas la meme condition. On n'exige plus qu'une decroissance stricte.
    enum Pas : int { ESSAIS = 0, DYADIQUE, FACTEUR, TENSEUR, ESSAI_LIMITES, MODELE, MERITE, GRILLE2 };
    /// MERITE : combien de barreaux de l'echelle on accepte de voir REMONTER avant de s'arreter ( le
    /// profil est unimodal, donc 1 suffit ).
    int  mer_patience = 1;
    /// MERITE : evaluations de RAFFINEMENT du pas apres l'argmin dyadique ( `0` : aucun ).
    ///
    /// L'echelle est dyadique, donc l'argmin trouve est le meilleur BARREAU, pas le vrai minimum : la
    /// grille est d'un facteur 2, ce qui est grossier, et rien ne dit que le minimum tombe dessus. On
    /// raffine par SECTION DOREE sur `[ tb / 2, min( t0, 2 tb ) ]`, ou le profil est unimodal. Chaque
    /// evaluation coute un diagramme -- c'est ce que le modele polynomial du § 22 saurait rendre
    /// gratuit, et c'est la raison d'etre de cette mesure : savoir si ca vaut la peine.
    int  mer_raffine = 0;
    /// MERITE : partir du pas des LIMITES EXACTES ( `facteur * alpha*` ) au lieu de `t0`.
    ///
    /// C'est ce que le mode faisait depuis le debut sans que ce soit dit : la passe des limites tourne
    /// pour tout `pas != essais`, donc le premier barreau de l'echelle etait deja `facteur * alpha*` et
    /// non `1`. D'ou des pas comme `0.107` ou `0.123` la ou j'annoncais une echelle dyadique. Les deux
    /// variantes sont maintenant separees et mesurees.
    bool mer_limites = true;
    /// le PLANCHER D'AIRE de l'amortissement. L'eteindre est exactement l'experience que `log` invite a
    /// faire : son merite penalise deja les cellules vides ( mesure, § 21.2 : 643 pour 1061 vides, 334
    /// pour 59, 322 pour 2 ), donc le plancher est peut-etre redondant avec lui.
    bool plancher_aire = true;
    /// L'ECRETAGE de `g` : `x` est borne par le bas a cette valeur. `0` : PAS D'ECRETAGE -- une cellule
    /// vide coute alors `+infini` au merite, donc AUCUN pas qui en vide une n'est acceptable, et le
    /// plancher d'aire devient inutile PAR CONSTRUCTION.
    ///
    /// L'ecretage n'est retire QUE DU MERITE. Dans la direction il reste, parce que
    /// `b_i = nu_i x_i ( c - log x_i )` vaut `0 x infini` sur une cellule vide : la direction doit rester
    /// calculable pour pouvoir REMPLIR une cellule vide ( ce dont on a besoin au multi-echelle, § 8.7 ),
    /// et l'ecretage lui donne exactement la bonne demande finie.
    TF   g_ecrete   = 1e-8;
    /// Note sur l'interaction des deux : le plancher ne peut etre ETEINT que pendant que le merite
    /// interdit lui-meme les cellules vides, c'est-a-dire avec un `log`/`puissance` NON ECRETE. Des que
    /// la bascule est passee a `lin`, le merite `lin` RECOMPENSE le vidage ( § 21.2 : son minimum le long
    /// de la direction est en `t = 1`, la ou 50 030 cellules sur 100 000 sont vides ), donc le plancher
    /// redevient indispensable. C'est mesure : sans cette regle, les plans 3D perdent une cellule a
    /// exactement zero a l'iteration 4 -- apres la bascule -- et le solveur lineaire echoue.
    int  mod_q      = 4;       ///< MODELE : le pas du simplexe cherche ( `1/mod_q` ), 4 = les 15 points de l'oracle
    /// MODELE : combien de directions ( 1 = Newton seul, 2 = + log, 3 = + barriere ).
    ///
    /// DEUX SUFFIT, et le defaut y est revenu. La troisieme direction ne gagne qu'UN diagramme, sur un
    /// seul cas ( `s0.1` ), et elle en perd un sur un autre ( `s0.005` nettoye : 13 a deux directions,
    /// 14 a trois ) -- pour une resolution lineaire de plus a chaque iteration, qui est le vrai prix du
    /// span. Elle avait paru indispensable tant que `s0.005` portait ses 56 paires de germes confondus
    /// ( 13 diagrammes avec elle, 47 sans ) ; le nuage nettoye ( § 23.7 ) fait tomber cette raison.
    int  mod_k      = 2;
    /// MODELE : prendre le pas de la passe des LIMITES EXACTES ( bissection par cellule, `alpha*`,
    /// puis `facteur * alpha*` ) au lieu de la racine du modele lui-meme.
    ///
    /// Le polynome est PESSIMISTE sur le plancher : a l'iteration 0 des lignes `sigma = 0.02` il
    /// annonce `alpha* = 0.118` quand l'exact vaut `0.273` -- une cellule qu'il voit passer sous `eps`
    /// n'y passe pas, parce que sa combinatoire change avant. C'est le meme mensonge qu'au § 7.
    ///
    /// Et pourtant le defaut est NON, parce que la mesure dit que le pessimisme coute moins que la
    /// parade : les limites exactes gagnent un diagramme sur deux cas sains ( 10 contre 11 ) mais en
    /// perdent 34 sur le nuage degenere ( 47 et STAGNATION contre 13 et CONVERGE ), et leur passe
    /// GLOBALE coute 0.41 s par iteration, soit 30 % de temps en plus.
    bool mod_limites = false;
    /// MODELE : SUR QUOI le modele choisit son melange. `PIRE` ( `max|a - nu|/nu` ) est le vrai critere
    /// d'arret et c'est le bon choix -- sauf quand une cellule ne peut PAS etre reparee : un `L-infini`
    /// est otage de cette cellule-la, le choix devient du bruit, et l'amortissement refuse tout
    /// ( mesure sur le nuage degenere `s0.005`, ou deux germes sont a 1e-8 : 1 139 diagrammes et
    /// 1 078 reculs ). `LOG` est la version robuste : tous les ecarts comptent, aucun ne decide seul.
    int  mod_juge   = PIRE;
    /// MODELE, juge `PIRE` : la fraction des cellules qu'on laisse DEHORS du maximum -- le critere
    /// devient le `k`-ieme pire ecart, `k = mod_hors * n`. Zero : le maximum strict.
    ///
    /// C'est la synthese des deux mesures : `max` strict est le bon critere quand toute cellule est
    /// reparable, et il devient du bruit des qu'une seule ne l'est pas. Enjamber une poignee
    /// d'aberrantes garde la nature extremale du critere sans lui donner d'otage.
    TF   mod_hors   = 1e-4;
    /// GRILLE2 : LA RECHERCHE A DEUX VARIABLES, `w + alpha d + beta ( t_prec d_prec )`.
    ///
    /// C'est un INSTRUMENT, pas un algorithme : la question est seulement de savoir si une seconde
    /// direction a de l'interet. Le § 24.10 a montre que le merite `log` a un vrai minimum interieur
    /// une fois la contrainte inactive, donc il y a enfin quelque chose de regulier a minimiser.
    ///
    /// La parametrisation est choisie pour que la GRILLE CONTIENNE L'ALGORITHME ACTUEL : la colonne
    /// `beta = 0` est la descente dyadique le long de la direction de Newton, donc tout gain se lit
    /// comme un ecart a cette colonne, et un argmin qui reste en `beta = 0` est un resultat NEGATIF
    /// franc. La seconde direction est le DEPLACEMENT precedent ( `t_prec d_prec` ), pas la direction
    /// brute : sa norme est celle d'un pas qui a ete accepte, donc `beta` est sans dimension.
    ///
    /// Cout : `g2_na * g2_nb` diagrammes par iteration. Personne ne propose ca comme defaut.
    /// GRILLE2 : D'OU VIENT LA SECONDE DIRECTION.
    ///
    /// `PREC` : le deplacement precedent. Mesure au § 24.12 : quasi colineaire a `d` en champ
    /// lointain ( cos 0.9 a 0.99 ), donc sans resolution, et sa moitie utile est refusee par le
    /// plancher parce que ce deplacement avait deja ete pousse a sa limite.
    ///
    /// `SONDE` : la direction de Newton recalculee AU BOUT DU RAYON, en `w + t d` ou `t` est le pas
    /// que l'amortissement s'apprete a prendre ( juste avant qu'une cellule casse ). Comme le
    /// probleme est non lineaire, `d` n'est valable qu'au depart : `d_s - d` est une difference
    /// finie de la direction le long du rayon, donc la COURBURE en `t`. La grille devient alors un
    /// melange, `w + alpha ( d + beta ( d_s - d ) )` : `beta = 0` est Newton pur ( l'algorithme
    /// actuel ), `beta = 1` est la direction de l'arrivee, et les deux axes sont sans dimension.
    ///
    /// Le point de sonde n'est pas gratuit mais il est BON MARCHE : son diagramme est celui du pas
    /// qu'on allait prendre, et il ne reste qu'un assemblage et un solve de plus.
    enum G2Dir : int { PREC = 0, SONDE };
    int  g2_dir     = PREC;    ///< GRILLE2 : d'ou vient la seconde direction
    int  g2_na      = 5;       ///< GRILLE2 : barreaux en `alpha` ( `alpha = t / 2^k` )
    int  g2_nb      = 5;       ///< GRILLE2 : barreaux en `beta` ( symetriques autour de 0 )
    TF   g2_bmax    = 1;       ///< GRILLE2 : `beta` balaye `[ -g2_bmax, g2_bmax ]`
    /// GRILLE2 : ne balayer que `beta >= 0`, donc `[ 0, g2_bmax ]`. Mesure ( § 24.13 ) : avec la
    /// direction SONDEE le merite est monotone en `beta` et c'est le cote POSITIF qui gagne -- aller
    /// vers la direction de l'arrivee, pas s'en eloigner. La moitie negative ne coute que des
    /// diagrammes.
    bool g2_bpos    = false;
    /// GRILLE2 : EVALUER LA GRILLE PAR LE MODELE POLYNOMIAL au lieu de diagrammes ( 2D ).
    ///
    /// A connectivite fixe l'aire de chaque cellule est un polynome EXACT du deplacement ( § 22 ) :
    /// `A_i( t ) = c0 + g . t + sum q_kl t_k t_l` sur le span, avec un RAYON sous lequel l'exactitude
    /// est prouvee. Explorer `( alpha, beta )` ne coute alors plus rien -- ni diagramme, ni cellule --
    /// donc la grille peut etre aussi fine qu'on veut, et une DESCENTE DE GRADIENT devient possible
    /// ( `PolyMulti::gradient` ). Le seul cout est la construction, un diagramme par iteration.
    bool g2_modele  = false;
    int  g2_desc    = 0;       ///< GRILLE2 MODELE : evaluations de DESCENTE DE GRADIENT ( 0 : aucune )
    int  g2_back    = 4;       ///< GRILLE2 MODELE : divisions par deux permises si le vrai merite ne descend pas
    /// GRILLE2 MODELE : prendre les aires du point de sonde d'un VRAI DIAGRAMME au lieu du modele.
    ///
    /// Un diagramme de plus par iteration, mais la direction sondee est alors exacte. C'est le
    /// temoin qui dit si la precision de `e` est ce qui achetait les iterations ( § 24.15 ).
    bool g2_sonde_reelle = false;
    TF   g2_tol     = 1e-3;    ///< GRILLE2 MODELE : tolerance du solve de la SONDE ( 0 : la meme que le reste )
    bool g2_verif   = true;    ///< GRILLE2 MODELE : verifier l'argmin du modele par un vrai diagramme
    bool g2_trace   = true;    ///< GRILLE2 : imprimer la matrice des merites
    int  pas        = ESSAIS;  ///< comment choisir `t` ( voir en tete )
    TF   facteur    = 0.9;     ///< `t = facteur * alpha*` en mode FACTEUR
    TF   theta_mult = 5;       ///< TENSEUR : la cible partielle est `theta = theta_mult * alpha*`
    TF   confiance  = 0;       ///< ESSAI_LIMITES : apres un pas CORRIGE, le prochain essai est au moins `confiance * t`
                               ///< ( 0 : `beta` inchange -- mesure meilleur : le pas admissible croit vite )
    TF   beta0      = 0.25;    ///< ESSAI_LIMITES : le tout premier essai ( 1 : un diagramme a moitie vide sur Voronoi )
    TF   mult_ok    = 2;       ///< ESSAI_LIMITES : apres un essai passe DIRECT, `beta *= mult_ok` ( plafonne a 1 )
    OptionsLimites lim;        ///< les reglages de la passe des limites ( `niveau` est mis ici )
    OptionsCible   cible;      ///< LA CIBLE MODIFIEE : deformer `nu` pour allonger le pas ( `Cible.h` )
    /// LE RESIDU : `g( a_i / nu_i )` au lieu de `a_i - nu_i`. Meme solution, autre Newton et autre
    /// merite pour l'amortissement. BARRIERE `g = x - 1/x` : une cellule minuscule ( `x << 1` ) recoit
    /// `x -> 2x` au lieu de sa masse entiere d'un coup, et `|g| ~ 1/x` refuse les pas qui la pincent ;
    /// LOG `g = log x` : `x -> x ( 1 - log x )`.
    /// LA FAMILLE DES PUISSANCES, qui contient les deux bouts et tout ce qu'il y a entre :
    ///
    ///     g_p( x ) = ( x^p - 1 ) / p,   g_p'( x ) = x^( p - 1 ),   g_0 = log
    ///
    /// `p = 1` EST `lin` ( au signe pres ), `p = 0` EST `log`, `p = -1` est `1 - 1/x`. Comparer trois
    /// residus ne dit pas POURQUOI l'un gagne ; un exposant continu, si -- parce qu'il donne la
    /// pente. Ce que `p` regle est le PARTAGE DU TRAVAIL entre les deux queues : le second membre
    /// de Newton vaut `b_i = a_i x^( -p ) ( c - g_p( x ) )`, donc
    ///
    ///     cellule GLOUTONNE ( x >> 1 ) : b ~ -a / p ( et -a log x en p = 0 )
    ///     cellule AFFAMEE   ( x << 1 ) : b ~ nu x^( 1 - p ) / p -- qui TEND VERS ZERO des que p < 1
    ///
    /// A `p = 1` seulement, une cellule affamee reclame son deficit ENTIER `nu`. Or elle ne peut pas
    /// le prendre : elle est enserree par des voisines dont le poids est trop haut, et elle ne
    /// grandira que quand CELLES-LA baisseront. La demande est donc inexaucable, et c'est elle qui
    /// force le pas minuscule. Tout `p < 1` l'annule et laisse le travail aux gloutonnes -- qui,
    /// elles, peuvent le faire.
    /// `PIRE` n'est pas un residu -- il n'a pas de derivee utilisable -- mais il fait un MERITE :
    /// `max_i |a_i - nu_i| / nu_i`, c'est-a-dire LE CRITERE D'ARRET LUI-MEME. L'amortissement KMT
    /// exige une norme `l2` pour sa preuve de decroissance ; on mesure ce que coute de la remplacer
    /// par celle qu'on veut vraiment faire baisser.
    /// `LOG2` n'est PAS un residu -- c'est un MERITE : `sqrt( sum ( log x_i )^2 )`, NON CENTRE.
    ///
    /// Le merite `LOG` est centre ( `| g - moyenne( g ) |_2` ) parce que le SECOND MEMBRE doit sommer a
    /// zero : on resout `g( x_i ) = c` et non `g( x_i ) = 0`, la jauge rayant une ligne. Mais rien
    /// n'oblige le JUGE a l'etre, et un merite centre peut decroitre le long d'un rayon qui n'approche
    /// pas la solution -- il ne mesure que la DISPERSION des `g`, pas leur ecart a zero. `LOG2` mesure
    /// l'ecart a zero, donc il s'annule exactement a la solution.
    enum Residu : int { LIN = 0, BARRIERE, LOG, PUISSANCE, PIRE, LOG2 };
    /// LE DEFAUT EST `LOG`, avec la bascule vers `LIN` ci-dessous. Mesure ( § 24 ) : -50 % de diagrammes
    /// sur le cas dur 2D, -37 % en 3D, rien de perdu nulle part. Et la bascule est ce qui le rend sans
    /// danger hors du solve direct : en continuation de densite, ou `log` SEUL est catastrophique
    /// ( § 9.6 : des dizaines d'iterations a pas 2e-3, residu fige ), chaque etape repart d'un residu
    /// deja petit -- donc la bascule tire a l'iteration 0 et le residu est `lin` du debut a la fin.
    int  residu     = LOG;
    TF   puis       = 0.5;     ///< l'exposant `p` de PUISSANCE
    /// LA BASCULE DE RESIDU : repasser a `LIN` des que `max|a - nu|/nu <= bascule_residu`. `0` : jamais.
    ///
    /// Pourquoi c'est la bonne forme. Les deux bouts ne servent pas au meme moment : `log` (`p = 0`)
    /// ne reclame a une cellule affamee qu'une fraction de son ecart LOGARITHMIQUE, ce qui est ce
    /// qu'il faut tant que la dynamique de `a / nu` est de 1 a 2000 ( § 21.3 ) ; `lin` est le vrai
    /// Newton du probleme et c'est lui qui donne la convergence quadratique a la fin. Et la bascule ne
    /// peut pas nuire tard, parce que pres de la solution `b_i = nu_i / g'( x_i ) ( c - g( x_i ) )`
    /// tend vers `nu_i - a_i` POUR TOUT RESIDU : les directions deviennent colineaires, donc le choix
    /// n'a plus d'objet.
    ///
    /// LE DEFAUT EST `2`, dans la fenetre plate `[ 0.5, 10 ]` mesuree au § 21.7. Il est INERTE quand
    /// `residu` vaut deja `LIN` ( le defaut ) -- la bascule ne peut que retourner A `lin`. Autrement
    /// dit, `--residu log` veut maintenant dire « log puis lin », qui est le seul usage de `log` que
    /// la mesure recommande. `--bascule-residu 0` rend le `log` pur.
    TF   bascule_residu = 2;
    /// LA BASCULE PAR LE PAS, au lieu d'un seuil sur `max|a - nu|/nu`. `0` : inactive.
    ///
    /// `bascule_residu` est un SEUIL, donc une constante a regler. Or le § 24.10 a montre ce que ce
    /// seuil approxime : la transition entre le regime ou la contrainte d'aire est ACTIVE ( le pas
    /// optimal est au bord ) et celui ou elle est INACTIVE ( le pas optimal est `t = 1` ). Cette
    /// transition, on ne l'approxime pas : on la MESURE. Des que l'amortissement a accepte un pas
    /// `>= bascule_pas`, la contrainte n'a pas mordu, et c'est exactement la condition sous laquelle
    /// `lin` est sans danger. Aucune constante d'echelle, donc, et rien qui depende du cas.
    ///
    /// Le test porte sur le pas de l'iteration PRECEDENTE ( la bascule se decide avant la direction ),
    /// donc il tire une iteration apres la transition. Il est latche comme l'autre.
    TF   bascule_pas = 0;
    /// LE MERITE DE L'AMORTISSEMENT, SEPAREMENT DE LA DIRECTION. `-1` : le meme que `residu`.
    ///
    /// `--residu log` changeait DEUX choses a la fois -- le second membre de Newton et le juge qui
    /// accepte le pas -- donc on ne pouvait pas savoir laquelle gagnait. Les deux roles n'ont rien
    /// a voir : la direction est un modele local ( ou `log` SURESTIME les cellules trop grosses,
    /// puisque sa tangente demande `a -> a ( 1 - log x )` < 0 des que `x > e` ), le merite est une
    /// norme ( ou `log` est le seul des trois a ne pas etre domine par une queue ). On les separe.
    int  merite_res = -1;
    /// >= 0 : a CETTE iteration, balayer le pas sur une echelle geometrique et imprimer, pour chaque
    /// `t`, LES TROIS merites, l'aire minimale et le pire ecart -- puis s'arreter. C'est le profil
    /// le long de la direction : ce que chaque juge voit du MEME deplacement.
    int  profil     = -1;
    int  oracle     = 0;       ///< > 0 : le MEILLEUR melange des trois directions a chaque iteration, a la force
                               ///< brute ( pas du simplexe `1/oracle` ). Borne superieure, pas un algorithme.
    bool oracle_pire = false;  ///< l'oracle choisit sur le VRAI critere d'arret, `max |a - nu| / nu`,
                               ///< au lieu du merite ( qui est un mauvais juge : cf. le profil )
    int  span       = -1;      ///< >= 0 : a CETTE iteration, construire le span progressivement ( § 24.21 )
    int  span_k     = 4;       ///< SPAN : dimension maximale ( `PolyMulti::KMAX` )
    int  span_desc  = 80;      ///< SPAN : pas de descente de gradient par dimension
    /// SPAN : `0` = variante A ( minimiser dans le span, puis Newton au point optimal ), `1` =
    /// variante B ( les derivees de `w( t )` au depart, UNE construction de modele ).
    int  span_mode  = 0;
    int  span_grille = 0;      ///< SPAN : cote d'une grille de controle de la minimisation ( 0 : aucune )
    bool span_carte = false;   ///< SPAN : imprimer la carte de `log2` et le controle du gradient
    int  span_hess  = 20;      ///< SPAN : pas de NEWTON sur la barriere ( 0 : aucun, cf. § 24.26 )
    int  modele     = -1;      ///< >= 0 : a CETTE iteration, batir le modele multi-directions et le
                               ///< confronter a l'evaluateur exact PUIS au vrai diagramme
    int  combi      = -1;      ///< >= 0 : a CETTE iteration, balayer le SIMPLEXE des trois directions ( lin, log, barriere )
                               ///< et dire, pour chaque melange, le plus grand pas admissible et ce qu.il gagne
    int  profil_nb  = 24;      ///< nombre de pas essayes par le profil ( `t = t0 / ratio^k` )
    TF   profil_ratio = 2;     ///< le rapport entre deux barreaux du profil ( `2` : dyadique )
    TF   t0         = 1;       ///< LE COEFFICIENT DE RELAXATION : le premier pas essaye ( 1 = Newton entier )
    /// tracer, a chaque iteration, CE QUI REND `L` DURE : l'etalement de la diagonale, celui des poids
    /// d'aretes, et l'ANISOTROPIE par ligne ( `max_j c_ij / sum_j c_ij` ). La derniere est la vraie
    /// question pour un multigrille : une ligne proche de 1 est un noeud couple a UN seul voisin, donc
    /// une chaine, et c'est ce qui met l'agregation en echec.
    ///
    /// Le residu n'y est POUR RIEN : `L` ne depend que des facettes ( `c_ij = |facette| / 2|p_i - p_j|` ),
    /// jamais de `g`. `log` ne change que le second membre.
    bool diag_lap   = false;
    /// > 0 : a l'iteration 0, DETECTER les grappes de germes a moins de `agglo` l'un de l'autre, et dire
    /// combien il y en a et ce que ca coute. C'est la premiere phase du § 23.6, et elle ne demande
    /// AUCUNE structure de donnees nouvelle :
    ///
    /// le diagramme en `w = 0` EST celui de Voronoi, donc sa liste de facettes EST le graphe de Delaunay.
    /// Or Delaunay contient l'arbre couvrant minimal euclidien, et les grappes du LIEN SIMPLE au seuil
    /// `delta` sont exactement les composantes connexes des aretes de l'ACM sous `delta`. Donc balayer
    /// `fa` et faire un union-find rend EXACTEMENT les grappes voulues -- pas de kd-tree, pas de grille,
    /// pas de tri. Et le diagramme, on le paye de toute facon.
    ///
    /// ATTENTION : ca ne vaut qu'en `w = 0`. Un diagramme de Laguerre n'est pas Delaunay et ne contient
    /// plus l'ACM -- la detection doit donc se faire AVANT la resolution, ce qui est justement le moment.
    TF   agglo      = 0;
    /// MODELE : combien de fractions de `alpha*` on essaye. LE DEFAUT EST UN -- aucune recherche.
    ///
    /// Elle etait gratuite ( sur le modele ) mais inutile, et c'est mesure : `1` contre `5` donne les
    /// MEMES chiffres sur les quatre nuages ( 6, 8, 11, 13 diagrammes ), et la trace montre que le choix
    /// tombait sur la plus longue fraction neuf fois sur dix. La raison est dans le profil du § 21.2 :
    /// le critere DECROIT de facon monotone le long du rayon jusqu'a ce que le plancher morde, donc le
    /// meilleur point admissible est TOUJOURS au bord. Il n'y a rien a chercher -- seulement a s'arreter
    /// juste avant le bord, et `0.99` le fait.
    int  mod_frac   = 1;
    int  refus      = -1;      ///< >= 0 : tracer, a CETTE iteration, laquelle des deux clauses de
                               ///< l.amortissement refuse chaque essai ( aire ou merite ), et sur quelle cellule
    bool memo       = false;   ///< 3D : les facettes du dernier diagramme ACCEPTE proposees en premier au suivant ( § 11 )
    /// appele apres chaque pas ACCEPTE ( et au depart, `it = -1` ) : `pd` porte alors `w`
    std::function<void( int it, TF t, int reculs )> apres_pas;
};

struct NewtonStats {
    const char *fin = "?";     ///< pourquoi la boucle s'est arretee
    TF     reste = 0;          ///< le `max_i |a_i - nu_i| / nu_i` atteint
    TF     reste0 = 0;         ///< le meme AU DEPART ( ce que vaut le point de depart )
    int    nb_iter = 0, nb_diag = 0, nb_recul = 0;
    int    nb_back = 0;           ///< GRILLE2 MODELE : backtrackings ( le vrai merite n'a pas descendu )
    SI     nb_deborde = 0;     ///< cellules qui ont deborde `MaxNv`, en tout ( mesure fausse )
    SI     nb_cell_lim = 0;    ///< cellules calculees par la passe des limites, en tout
    int    nb_tenseur = 0;     ///< pas tensoriels tentes
    double t_tenseur = 0;
    SI     nb_cell_mauvaises = 0; ///< ESSAI_LIMITES : cellules trouvees sous `eps` par les essais, en tout
    int    nb_tours_essai = 0;    ///< ESSAI_LIMITES : essais corriges par des limites locales
    int    nb_lim_refus = 0;   ///< pas proposes par les limites et refuses par le diagramme
    int    nb_cible_res = 0;   ///< CIBLE : resolutions lineaires depensees, en tout
    int    nb_cible_pris = 0;  ///< CIBLE : iterations ou la direction deformee a ete retenue
    int    nb_cible_refus = 0; ///< CIBLE : ... et ou l'amortissement l'a rejetee, filet declenche
    TF     cible_gain = 0;     ///< CIBLE : produit des `U*_apres / U*_avant`, pour la moyenne geometrique
    double t_cible = 0;
    TF     amp_d0 = 0;         ///< `| d |inf` de la PREMIERE direction resolue. Une continuation qui
                               ///< enchaine des Newton s'en sert pour choisir son pas ( README
                               ///< § 12.6.1 ) : c'est la correction de poids que l'etape a reclamee,
                               ///< et elle est deja calculee -- la relire ne coute rien.
    double t_maj = 0, t_diag = 0, t_asm = 0, t_lin = 0, t_lim = 0, t_memo = 0;
};

template<class PD, class Rho = Densite>
struct Newton {
    PD             &pd;
    Lineaire       &lin;
    const TF *const *P;        ///< les positions, dans l'ordre des identifiants
    Parallel        par;
    NewtonOptions   o;

    std::vector<TF> nu;        ///< la mesure cible, par germe
    std::vector<TF> w;         ///< les poids courants, `w[ 0 ] == 0`
    std::vector<TF> a;         ///< les mesures courantes
    std::vector<TF> d;         ///< la derniere direction de Newton ( `d[ 0 ] == 0` )
    std::vector<Facette> fa;   ///< les facettes du diagramme courant ( celui de `w` )
    NewtonStats     st;
    int             res_cur = NewtonOptions::LIN;  ///< le residu EN COURS ( `o.bascule_residu` le change )

    /// UNE DENSITE au lieu de Lebesgue ( 2D ) : la mesure d'une cellule est sa masse. Le pas par les
    /// limites ( ESSAI_LIMITES ) passe alors par `limites_masse` : la bissection, pas le polynome.
    /// `Rho` n'a qu'a offrir `mesure( cel, facette, dl )` -- `Densite.h` ( gaussiennes ) et
    /// `Image.h` ( une grille de pixels ) le font, et Newton ne distingue pas les deux.
    const Rho      *rho = nullptr;
    bool            derivee = false; ///< avec `rho` : calculer aussi `da = d a / d lambda` a chaque diagramme
    std::vector<TF> da;        ///< `d a_i / d lambda` pour `w` ( le parametre du chemin de `rho` )

    Newton( PD &pd, Lineaire &lin, const TF *const *P, Parallel par, NewtonOptions o = {} )
        : pd( pd ), lin( lin ), P( P ), par( par ), o( o ) {}

    /// LES MESURES ET LES FACETTES pour les poids `W` : un diagramme, et la conversion
    /// `c_ij = |facette| / ( 2 |p_i - p_j| )` faite par le thread qui a mesure la cellule.
    void mesures_et_facettes( const std::vector<TF> &W, std::vector<TF> &res, std::vector<Facette> &fa, std::vector<TF> *dres = nullptr ) {
        constexpr int D = PD::dim;
        double t0 = now();
        pd.set_weights( W.data(), par );
        st.t_maj += now() - t0;
        if constexpr ( D == 3 ) {                        // la memoire : les facettes du diagramme accepte
            if ( o.memo && ! this->fa.empty() ) {
                t0 = now();
                std::vector<d2::SI32> ii( this->fa.size() ), jj( this->fa.size() );
                for ( SI q = 0; q < SI( this->fa.size() ); ++q ) { ii[ q ] = d2::SI32( this->fa[ q ].i ); jj[ q ] = d2::SI32( this->fa[ q ].j ); }
                pd.memorise( ii.data(), jj.data(), SI( ii.size() ) );
                st.t_memo += now() - t0;
            } else
                pd.oublie();
        }

        t0 = now();
        std::vector<std::vector<Facette>> par_th( std::max( par.threads, 1 ) );
        auto facette = [ & ]( int t, SI i, SI j, TF mes ) {
            TF d2 = 0;
            for ( int d = 0; d < D; ++d ) {
                const TF e = P[ d ][ j ] - P[ d ][ i ];
                d2 += e * e;
            }
            if ( d2 > 0 )
                par_th[ t ].push_back( Facette{ i, j, mes / ( 2 * std::sqrt( d2 ) ) } );
        };
        if ( rho ) {
            if constexpr ( D == 2 ) {
                if ( dres ) dres->assign( pd.n, TF( 0 ) );
                st.nb_deborde += pd.measures_and_facets_avec( res, par, facette, [ & ]( const typename PD::Cell &cel, auto &&fac, SI i ) {
                    return rho->mesure( cel, fac, dres ? &( *dres )[ i ] : nullptr );
                } );
            }
        } else
            st.nb_deborde += pd.measures_and_facets( res, par, facette );
        fa.clear();
        for ( auto &v : par_th )
            fa.insert( fa.end(), v.begin(), v.end() );
        st.t_diag += now() - t0;
        ++st.nb_diag;
    }

    static TF norme2( const std::vector<TF> &v ) {
        TF s = 0;
        for ( TF x : v ) s += x * x;
        return std::sqrt( s );
    }

    /// `g( x )` et `g'( x )` du residu `r`, `x = a / nu` borne loin de zero
    static TF g_de( TF x, int r, TF p = 0 ) {
        x = std::max( x, TF( 1e-8 ) );
        if ( r == NewtonOptions::PUISSANCE )
            return p == 0 ? std::log( x ) : ( std::pow( x, p ) - 1 ) / p;
        return r == NewtonOptions::BARRIERE ? x - 1 / x : r == NewtonOptions::LOG ? std::log( x ) : x - 1;
    }
    static TF gp_de( TF x, int r, TF p = 0 ) {
        x = std::max( x, TF( 1e-8 ) );
        if ( r == NewtonOptions::PUISSANCE )
            return std::pow( x, p - 1 );
        return r == NewtonOptions::BARRIERE ? 1 + 1 / ( x * x ) : r == NewtonOptions::LOG ? 1 / x : 1;
    }
    /// ORTHOGONALISER LA NOUVELLE DIRECTION CONTRE LE SPAN, ET LA RENORMALISER.
    ///
    /// C'est LA SOUSTRACTION QUI MANQUAIT, et ce n'est pas cosmetique. Mathematiquement
    /// `{ d_1, d_2 }` et `{ d_1, d_2 - proj( d_2 ) }` sont le MEME span, donc l'orthogonalisation
    /// ne change rien a ce qui est atteignable. Mais numeriquement elle change tout : avec une
    /// base quasi degeneree ( `cos = 0.98`, § 24.21 ), la coordonnee utile de la seconde direction
    /// ne vaut que quelques pour cent de sa norme, et une descente de gradient a PAS UNIQUE --
    /// cale sur l'echelle de `t_1` -- l'affame. Apres Gram-Schmidt et renormalisation a la norme
    /// de `d_1`, les coordonnees sont comparables et la descente les traite a egalite.
    ///
    /// Rend `|d_perp| / |d|` AVANT renormalisation : la fraction de la direction qui est
    /// reellement neuve, qui est la grandeur a lire ( le cosinus, lui, devient nul par
    /// construction ).
    TF ortho( std::vector<TF> *dd, int K, TF n2_avant ) const {
        const SI n = SI( dd[ 0 ].size() );
        for ( int k = 0; k < K; ++k ) {
            TF ps = 0, nk = 0;
            for ( SI i = 0; i < n; ++i ) { ps += dd[ K ][ i ] * dd[ k ][ i ]; nk += dd[ k ][ i ] * dd[ k ][ i ]; }
            if ( ! ( nk > 0 ) ) continue;
            const TF c = ps / nk;
            for ( SI i = 0; i < n; ++i ) dd[ K ][ i ] -= c * dd[ k ][ i ];
        }
        TF np = 0, n0 = 0;
        for ( SI i = 0; i < n; ++i ) { np += dd[ K ][ i ] * dd[ K ][ i ]; n0 += dd[ 0 ][ i ] * dd[ 0 ][ i ]; }
        const TF frac = n2_avant > 0 ? std::sqrt( np / n2_avant ) : TF( 0 );
        if ( np > 0 && n0 > 0 ) {                        // a la norme de `d_1`, pour que les
            const TF e = std::sqrt( n0 / np );           // coordonnees soient comparables
            for ( SI i = 0; i < n; ++i ) dd[ K ][ i ] *= e;
        }
        return frac;
    }

    /// `g( x )` et `g'( x )` du residu choisi, `x = a / nu` borne loin de zero
    TF g( TF x ) const { return g_de( x, res_cur, o.puis ); }
    TF gp( TF x ) const { return gp_de( x, res_cur, o.puis ); }

    /// LE SECOND MEMBRE de Newton pour le residu `r` : `b_i = nu_i / g'( x_i ) ( c - g( x_i ) )`, avec
    /// `c` la moyenne ponderee qui le fait sommer a zero ( la jauge raye une ligne : sans ca elle
    /// porterait toute l'incoherence ). `r = LIN` redonne `nu - a`.
    void membre( int r, TF p, std::vector<TF> &bb ) const { membre_de( a, r, p, bb ); }
    /// Le meme, pour des aires QUELCONQUES -- le second membre en un point de SONDE ( § 24.13 ).
    void membre_de( const std::vector<TF> &A, int r, TF p, std::vector<TF> &bb ) const {
        const SI n = SI( A.size() );
        bb.assign( n, TF( 0 ) );
        if ( r == NewtonOptions::LIN ) {
            for ( SI i = 0; i < n; ++i ) bb[ i ] = nu[ i ] - A[ i ];
            return;
        }
        TF su = 0, sug = 0;
        for ( SI i = 0; i < n; ++i ) {
            const TF x = A[ i ] / nu[ i ], u = nu[ i ] / gp_de( x, r, p );
            su += u; sug += u * g_de( x, r, p );
        }
        const TF c = sug / su;
        for ( SI i = 0; i < n; ++i ) {
            const TF x = A[ i ] / nu[ i ];
            bb[ i ] = nu[ i ] / gp_de( x, r, p ) * ( c - g_de( x, r, p ) );
        }
    }
    /// LE MERITE de l'amortissement : `| a - nu |_2` pour LIN ( les chiffres de reference ), et la norme
    /// SANS DIMENSION `| g( a / nu ) - moyenne |_2` pour les autres. La moyenne : `sum a = sum nu` est
    /// automatique, donc `g( x_i ) = 0` pour tout `i` fait `n` equations pour `n - 1` inconnues ; c'est
    /// `g( x_i ) = c` pour tout `i` qu'on resout ( qui force `x_i = 1` puisque la moyenne des `x` est 1 ),
    /// et le second membre projete somme a zero comme il faut ( sans ca, la ligne rayee par la jauge
    /// porte toute l'incoherence : mesure, le germe 0 explose et Newton stagne a la premiere etape ).
    TF merite_de( const std::vector<TF> &A, int r, TF p = 0 ) const {
        const SI n = SI( A.size() );
        // LE LOG NON ECRETE : une cellule vide coute `+infini`, donc le pas est inacceptable, point.
        if ( o.g_ecrete <= 0 && r != NewtonOptions::LIN && r != NewtonOptions::PIRE )
            for ( SI i = 0; i < n; ++i )
                if ( ! ( A[ i ] > 0 ) ) return INFINI;
        if ( r == NewtonOptions::LOG2 ) {            // NON CENTRE : l'ecart a zero, pas la dispersion
            TF s2 = 0;
            for ( SI i = 0; i < n; ++i ) {
                const TF g = g_de( A[ i ] / nu[ i ], NewtonOptions::LOG );
                s2 += g * g;
            }
            return std::sqrt( s2 );
        }
        if ( r == NewtonOptions::PIRE ) {
            TF m = 0;
            for ( SI i = 0; i < n; ++i ) m = std::max( m, std::fabs( nu[ i ] - A[ i ] ) / nu[ i ] );
            return m;
        }
        if ( r == NewtonOptions::LIN ) {
            TF s = 0;
            for ( SI i = 0; i < n; ++i ) s += ( nu[ i ] - A[ i ] ) * ( nu[ i ] - A[ i ] );
            return std::sqrt( s );
        }
        TF m = 0;
        for ( SI i = 0; i < n; ++i ) m += g_de( A[ i ] / nu[ i ], r, p );
        m /= n;
        TF s = 0;
        for ( SI i = 0; i < n; ++i ) { const TF e = g_de( A[ i ] / nu[ i ], r, p ) - m; s += e * e; }
        return std::sqrt( s );
    }
    /// LE PLANCHER D'AIRE EST-IL ACTIF ? On ne peut l'eteindre que pendant que le merite interdit
    /// lui-meme les cellules vides -- donc avec un `log`/`puissance` NON ECRETE, et jamais en `lin`.
    bool plancher_actif() const {
        return o.plancher_aire || o.g_ecrete > 0
            || res_cur == NewtonOptions::LIN || res_cur == NewtonOptions::PIRE;
    }

    /// le merite EFFECTIF de l'amortissement : celui de `merite_res`, ou celui de `residu` par defaut
    TF merite( const std::vector<TF> &A ) const {
        return merite_de( A, o.merite_res < 0 ? res_cur : o.merite_res, o.puis );
    }

    /// LA BOUCLE, depuis `w_init` ( zero : Voronoi ). Rend `true` si le critere d'arret est atteint.
    /// `deja_mesure` : `a`, `fa` ( et `da` ) sont DEJA ceux de `w_init` ( qui porte la jauge ) --
    /// l'appelant les a calcules en choisissant son depart, on ne refait pas ce diagramme.
    bool resout( const std::vector<TF> &w_init, bool deja_mesure = false ) {
        const SI n = pd.n;
        TF nr_prec = 0;                                  // `|r|_2` de l'iteration precedente
        std::vector<TF> a2, b, w2, da2, wb, ab;
        std::vector<Facette> fa2, fab;
        std::vector<LimiteCellule> lim;
        std::vector<TF> ucel, u2, b2, d2, sflux, dgard, del;  // la passe CIBLE
        std::vector<char> touche, dumm;
        std::vector<TF> aso;                         // MODELE : les aires MODELISEES au point de sonde
        std::vector<TF> d_next;                      // la direction resolue AU POINT DE SONDE
        std::vector<PolyMulti> pm2;                  // GRILLE2 MODELE : le polynome par cellule
        std::vector<TF> d_pre, bson, as_;            // GRILLE2 : la SECONDE DIRECTION, son second membre
        std::vector<Facette> fas_;                   // ... et les mesures du point de SONDE
        Laplacien Lson;                              // GRILLE2 SONDE : le laplacien au point de sonde
        std::vector<TF> d_sur;                       // LE FILET : la direction de Newton, avant
        std::vector<ModeleCellule> mods;             // ... son juge POLYNOME
        std::vector<SI> imod;
        std::vector<TF> ecor, dlt, alim, t1, t2;
        std::vector<char> coeurp;
        TF seuil_prec = 0;                           // la limite du polynome de l'iteration d'avant
        std::vector<char> dans;
        Laplacien L;
        std::vector<TF> *pda = rho && derivee ? &da : nullptr, *pda2 = pda ? &da2 : nullptr;

        w = w_init;
        const TF jauge = w[ 0 ];
        for ( SI i = 0; i < n; ++i )                     // la jauge, imposee ici et maintenue par
            w[ i ] -= jauge;                             // `d[ 0 ] = 0` ensuite
        if ( ! deja_mesure )
            mesures_et_facettes( w, a, fa, pda );
        if ( o.apres_pas ) o.apres_pas( -1, 0, 0 );

        // ---- LA PHASE D'AGGLOMERATION, PREMIERE MOITIE : detecter les grappes ( § 23.6 )
        //
        // Sans aucune structure de donnees nouvelle. En `w = 0` le diagramme EST celui de Voronoi, donc
        // `fa` EST le graphe de Delaunay ; Delaunay contient l'arbre couvrant minimal euclidien ; et les
        // grappes du lien simple au seuil `delta` sont exactement les composantes connexes des aretes de
        // l'ACM sous `delta`. Un balayage de `fa` et un union-find suffisent donc, et sont EXACTS.
        if ( o.agglo > 0 ) {
            const double ta0 = now();
            std::vector<SI> pere( n );
            for ( SI i = 0; i < n; ++i ) pere[ i ] = i;
            auto trouve = [ & ]( SI i ) {
                while ( pere[ i ] != i ) { pere[ i ] = pere[ pere[ i ] ]; i = pere[ i ]; }
                return i;
            };
            SI nb_aretes = 0, nb_proches = 0;
            TF dmin = INFINI, dmed_ech = 0;
            std::vector<TF> dists;
            dists.reserve( fa.size() / 2 + 1 );
            for ( const Facette &e : fa ) {
                if ( e.i >= e.j ) continue;              // l'autre vue de la meme facette
                ++nb_aretes;
                TF d2 = 0;
                for ( int k = 0; k < PD::dim; ++k ) {
                    const TF u = P[ k ][ e.i ] - P[ k ][ e.j ];
                    d2 += u * u;
                }
                const TF dd = std::sqrt( d2 );
                dists.push_back( dd );
                dmin = std::min( dmin, dd );
                if ( dd < o.agglo ) {
                    ++nb_proches;
                    const SI a = trouve( e.i ), b = trouve( e.j );
                    if ( a != b ) pere[ a < b ? b : a ] = a < b ? a : b;
                }
            }
            if ( ! dists.empty() ) {
                std::nth_element( dists.begin(), dists.begin() + dists.size() / 2, dists.end() );
                dmed_ech = dists[ dists.size() / 2 ];
            }
            // les grappes, et leurs tailles
            std::vector<SI> taille( n, 0 );
            for ( SI i = 0; i < n; ++i ) ++taille[ trouve( i ) ];
            SI nb_grappes = 0, nb_dedans = 0, tmax = 0;
            for ( SI i = 0; i < n; ++i )
                if ( taille[ i ] > 1 ) { ++nb_grappes; nb_dedans += taille[ i ]; tmax = std::max( tmax, taille[ i ] ); }
            const double t_ag = now() - ta0;
            std::printf( "    AGGLO seuil %.3e : %d aretes de Delaunay ( longueur min %.3e, mediane %.3e ),"
                         " %d sous le seuil -> %d grappes, %d germes dedans, taille max %d, en %.4f s"
                         " ( %.1f %% d'un diagramme )\n",
                         double( o.agglo ), int( nb_aretes ), double( dmin ), double( dmed_ech ),
                         int( nb_proches ), int( nb_grappes ), int( nb_dedans ), int( tmax ), t_ag,
                         100.0 * t_ag / std::max( st.t_diag, 1e-9 ) );
            std::fflush( stdout );
            st.fin = "AGGLO";
            return false;
        }

        res_cur = o.residu;                              // ... que `o.bascule_residu` fera passer a `LIN`
        TF eps = 0, t_prec = 0, beta = o.beta0, t_sur = -1;
        std::vector<char> protegee;                      // les cellules NON VIDES au depart : celles que `eps` defend
        for ( int it = 0; it < o.maxit; ++it ) {
            TF pire = 0;
            SI nvide = 0;
            for ( SI i = 0; i < n; ++i ) {
                nvide += ! ( a[ i ] > 0 );
                pire = std::max( pire, std::fabs( nu[ i ] - a[ i ] ) / nu[ i ] );
            }
            // ---- LA BASCULE DE RESIDU : `log` tant que la dynamique est large, `lin` pour finir.
            // Elle se decide ICI, avant le second membre ET avant le merite de l'iteration, pour que
            // `b`, `nr` et `n2r` parlent tous du meme residu. Elle est LATCHEE : `pire` n'est pas
            // monotone, et on ne veut pas revenir en arriere.
            if ( o.bascule_residu > 0 && res_cur != NewtonOptions::LIN && pire <= o.bascule_residu ) {
                res_cur = NewtonOptions::LIN;
                if ( o.trace )
                    std::printf( "      bascule : residu -> lin ( max|a-nu|/nu %.3e <= %.3e )\n",
                                 double( pire ), double( o.bascule_residu ) );
            }
            // La MEME bascule, mais declenchee par la contrainte elle-meme : un pas plein accepte dit
            // qu'elle n'a pas mordu, donc que `lin` est sans danger ( cf. `o.bascule_pas` ).
            if ( o.bascule_pas > 0 && res_cur != NewtonOptions::LIN && t_prec >= o.bascule_pas ) {
                res_cur = NewtonOptions::LIN;
                if ( o.trace )
                    std::printf( "      bascule : residu -> lin ( pas precedent %.3e >= %.3e )\n",
                                 double( t_prec ), double( o.bascule_pas ) );
            }
            membre( res_cur, o.puis, b );                // `J = diag( g' / nu ) L` : `L d = ( nu / g' ) ( c - g )`
            if ( it == 0 ) {
                // LE PLANCHER D'AIRE DE L'AMORTISSEMENT, ET LES CELLULES QU'IL DEFEND.
                //
                // Il se lisait `0.5 min( min nu, min a )` sur TOUTES les cellules. Une seule cellule
                // vide au depart mettait donc `eps` a ZERO -- et alors le critere d'acceptation
                // `m2 >= eps` est satisfait par n'importe quel pas : le garde-fou d'aire disparaissait
                // en silence, et le premier pas pouvait vider des milliers de cellules ( mesure :
                // 1 vide au depart -> 7750 apres un pas, README § 8.6 ). Or un depart avec une poignee
                // de vides est exactement ce que rend une reparation incomplete, et il n'a rien de
                // fatal en soi : a nombre de vides nul, un residu de depart quatre fois pire ne coute
                // qu'une iteration ( § 8.6 ).
                //
                // Le plancher se lit donc sur les cellules VIVANTES, et il ne defend que celles-la :
                // une cellule deja vide ne peut pas etre remontee par l'amortissement, et l'exiger
                // au-dessus du plancher refuserait TOUT pas.
                TF am = INFINI, nm = nu[ 0 ];
                protegee.assign( n, 0 );
                for ( SI i = 0; i < n; ++i ) {
                    nm = std::min( nm, nu[ i ] );
                    if ( a[ i ] > 0 ) { am = std::min( am, a[ i ] ); protegee[ i ] = 1; }
                }
                eps = TF( 0.5 ) * std::min( nm, am < INFINI ? am : nm );
            }
            const TF nr = merite( a );
            if ( o.plancher > 0 && nr < o.plancher ) {
                st.fin = "PLANCHER DE BRUIT";            // la mesure ne sait plus rien dire
                st.reste = pire;
                return false;
            }
            if ( o.progres_min > 0 && it > 0 && nr > ( 1 - o.progres_min ) * nr_prec ) {
                st.fin = "PROGRES INSUFFISANT";          // le bruit de la mesure, pas un echec
                st.reste = pire;
                return false;
            }
            nr_prec = nr;
            st.reste = pire;
            if ( it == 0 ) st.reste0 = pire;

            if ( pire <= o.tol ) {
                if ( o.trace )
                    std::printf( "    it %2d  |r|_2 %.3e  max|a-nu|/nu %.3e  CONVERGE\n",
                                 it, double( nr ), double( pire ) );
                st.fin = "CONVERGE";
                return true;
            }
            ++st.nb_iter;
            const double d0 = st.t_diag, l0 = lin.st.total(), s0 = st.t_asm, m0 = st.t_maj;
            const int    g0 = st.nb_diag, i0 = lin.st.nb_iter;

            double t0 = now();
            L.assemble( n, fa );
            st.t_asm += now() - t0;
            if ( o.diag_lap ) {
                auto quant = []( std::vector<TF> v, double q ) {
                    if ( v.empty() ) return TF( 0 );
                    const size_t k = size_t( q * double( v.size() - 1 ) );
                    std::nth_element( v.begin(), v.begin() + k, v.end() );
                    return v[ k ];
                };
                std::vector<TF> dd( L.dia.begin() + 1, L.dia.end() ), cc, an;
                SI nz = 0;
                for ( SI i = 1; i < n; ++i ) {
                    TF sm = 0, mx = 0;
                    for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e )
                        if ( L.col[ e ] >= 1 ) { cc.push_back( L.c[ e ] ); sm += L.c[ e ]; mx = std::max( mx, L.c[ e ] ); }
                    if ( sm > 0 ) an.push_back( mx / sm );
                }
                const TF cmed = quant( cc, 0.5 );
                for ( TF v : cc ) nz += v < TF( 1e-6 ) * cmed;
                SI chaine = 0;
                for ( TF v : an ) chaine += v > TF( 0.9 );
                std::printf( "      lap : diag %.2e / %.2e / %.2e ( max/min %.1e ) | aretes %d,"
                             " c %.2e / %.2e / %.2e, %d sous 1e-6 med | anisotropie med %.3f, %.2f %% de lignes > 0.9\n",
                             double( quant( dd, 0 ) ), double( quant( dd, 0.5 ) ), double( quant( dd, 1 ) ),
                             double( quant( dd, 1 ) / std::max( quant( dd, 0 ), TF( 1e-300 ) ) ),
                             int( cc.size() ), double( quant( cc, 0 ) ), double( cmed ), double( quant( cc, 1 ) ),
                             int( nz ), double( quant( an, 0.5 ) ), 100.0 * double( chaine ) / double( std::max<size_t>( an.size(), 1 ) ) );
                std::fflush( stdout );
            }
            t0 = now();
            const bool fait = lin.resout( L, b, d );
            st.t_lin += now() - t0;
            if ( ! fait ) {
                st.fin = "SOLVEUR LINEAIRE EN ECHEC";
                return false;
            }
            if ( it == 0 ) {                             // ce que l'etape a reclame, pour qui enchaine
                TF m = 0;
                for ( SI i = 0; i < n; ++i ) m = std::max( m, std::fabs( d[ i ] ) );
                st.amp_d0 = m;
            }
            if ( it == o.extraire ) {                    // la direction est ce qu'on venait chercher
                st.fin = "DIRECTION EXTRAITE";
                return false;
            }

            // ---- LE PROFIL LE LONG DE LA DIRECTION : ce que chaque juge voit du MEME deplacement.
            //
            // L'amortissement ne montre que sa decision. Ici on balaye `t` et on imprime les TROIS
            // merites cote a cote, plus l'aire minimale : on lit d'un coup lequel decroit, sur quelle
            // plage, et si c'est l'aire ou le merite qui ferme la porte.
            if ( it == o.profil ) {
                const TF r_lin = merite_de( a, NewtonOptions::LIN ), r_bar = merite_de( a, NewtonOptions::BARRIERE ),
                         r_log = merite_de( a, NewtonOptions::LOG ), r_pui = merite_de( a, NewtonOptions::PUISSANCE, o.puis ),
                         r_lg2 = merite_de( a, NewtonOptions::LOG2 );
                std::printf( "    PROFIL it %d, direction %s, n %d : depart merites lin %.6e  barriere %.6e  log %.6e  p=%g %.6e\n",
                             it, o.residu == NewtonOptions::BARRIERE ? "barriere" : o.residu == NewtonOptions::LOG ? "log"
                               : o.residu == NewtonOptions::PUISSANCE ? "puissance" : "lin",
                             int( n ), double( r_lin ), double( r_bar ), double( r_log ), double( o.puis ), double( r_pui ) );
                std::printf( "      %-10s %-12s %-12s %-12s %-12s %-12s  %-10s %-10s %-9s %s\n", "t", "lin", "barriere", "log",
                             "puissance", "log2", "aire min", "max ecart", "vides", "qui decroit ( >= 1 - t/2 exige )" );
                w2.resize( n );
                TF tp_k = o.t0 * o.profil_ratio;
                for ( int k = 0; k < o.profil_nb; ++k ) {
                    tp_k /= o.profil_ratio;
                    const TF tp = tp_k;
                    for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + tp * d[ i ];
                    w2[ 0 ] = 0;
                    mesures_et_facettes( w2, a2, fa2, pda2 );
                    TF m2 = INFINI, pir = 0;
                    SI nv2 = 0;
                    for ( SI i = 0; i < n; ++i ) {
                        if ( protegee[ i ] && a2[ i ] < m2 ) m2 = a2[ i ];
                        nv2 += ! ( a2[ i ] > 0 );
                        pir = std::max( pir, std::fabs( nu[ i ] - a2[ i ] ) / nu[ i ] );
                    }
                    const TF s_lin = merite_de( a2, NewtonOptions::LIN ), s_bar = merite_de( a2, NewtonOptions::BARRIERE ),
                             s_log = merite_de( a2, NewtonOptions::LOG ), s_pui = merite_de( a2, NewtonOptions::PUISSANCE, o.puis ),
                             s_lg2 = merite_de( a2, NewtonOptions::LOG2 );
                    const TF ex = 1 - tp / 2;            // la decroissance exigee par l'amortissement
                    char qui[ 64 ];
                    std::snprintf( qui, sizeof( qui ), "%s %s %s %s %s", s_lin <= ex * r_lin ? "lin" : "---",
                                   s_bar <= ex * r_bar ? "bar" : "---", s_log <= ex * r_log ? "log" : "---",
                                   s_pui <= ex * r_pui ? "pui" : "---", s_lg2 <= ex * r_lg2 ? "lg2" : "---" );
                    std::printf( "      %-10.3e %-12.6e %-12.6e %-12.6e %-12.6e %-12.6e  %-10.3e %-10.3e %-9d %s\n",
                                 double( tp ), double( s_lin ), double( s_bar ), double( s_log ), double( s_pui ),
                                 double( s_lg2 ), double( m2 ), double( pir ), int( nv2 ), qui );
                    std::fflush( stdout );
                }
                st.fin = "PROFIL";
                return false;
            }

            // ---- LA COMBINAISON DE PLUSIEURS DIRECTIONS : le SPAN contient-il mieux que ses bouts ?
            //
            // Trois residus donnent trois directions, et les trois se resolvent SUR LA MEME
            // FACTORISATION -- donc deux descentes de plus, aucun diagramme, aucun assemblage. La
            // question qui decide s'il vaut la peine de modeliser l'aire sur tout le span ( le
            // polynome multi-directions ) est : un MELANGE fait-il nettement mieux qu'aucune des
            // trois seule ? On le mesure a la force brute -- pour chaque `lambda` du simplexe, le
            // plus grand pas qui respecte le plancher, et le pire ecart qu'il atteint.
            if ( it == o.combi ) {
                const int NR = 3;
                const int rs[ NR ] = { NewtonOptions::LIN, NewtonOptions::LOG, NewtonOptions::BARRIERE };
                std::vector<TF> dd[ NR ], bb;
                for ( int k = 0; k < NR; ++k ) {
                    membre( rs[ k ], o.puis, bb );
                    dd[ k ].assign( n, TF( 0 ) );
                    if ( k == 0 || ! lin.sait_encore() ) lin.resout( L, bb, dd[ k ] );
                    else                                 lin.resout_encore( bb, dd[ k ] );
                }
                std::printf( "    COMBI it %d, n %d, eps %.3e : depart max ecart %.3e, merites lin %.6e log %.6e\n",
                             it, int( n ), double( eps ), double( pire ),
                             double( merite_de( a, NewtonOptions::LIN ) ), double( merite_de( a, NewtonOptions::LOG ) ) );
                std::printf( "      %-16s %-10s %-4s %-10s %-10s %-12s %-12s\n", "lambda (lin,log,bar)",
                             "t admis", "diag", "aire min", "max ecart", "merite lin", "merite log" );
                w2.resize( n );
                const int Q = 4;                         // le pas du simplexe : `lambda` multiple de 1/Q
                for ( int i1 = 0; i1 <= Q; ++i1 )
                for ( int i2 = 0; i2 + i1 <= Q; ++i2 ) {
                    const TF l0 = TF( Q - i1 - i2 ) / Q, l1 = TF( i1 ) / Q, l2 = TF( i2 ) / Q;
                    int nd = 0;
                    TF tp = o.t0;
                    for ( ; tp > TF( 1e-6 ); tp /= 2 ) {
                        for ( SI i = 0; i < n; ++i )
                            w2[ i ] = w[ i ] + tp * ( l0 * dd[ 0 ][ i ] + l1 * dd[ 1 ][ i ] + l2 * dd[ 2 ][ i ] );
                        w2[ 0 ] = 0;
                        mesures_et_facettes( w2, a2, fa2, pda2 );
                        ++nd;
                        TF m2 = INFINI;
                        for ( SI i = 0; i < n; ++i )
                            if ( protegee[ i ] && a2[ i ] < m2 ) m2 = a2[ i ];
                        if ( m2 >= eps ) {
                            TF pir = 0;
                            for ( SI i = 0; i < n; ++i ) pir = std::max( pir, std::fabs( nu[ i ] - a2[ i ] ) / nu[ i ] );
                            std::printf( "      %4.2f %4.2f %4.2f     %-10.3e %-4d %-10.3e %-10.3e %-12.6e %-12.6e\n",
                                         double( l0 ), double( l1 ), double( l2 ), double( tp ), nd, double( m2 ),
                                         double( pir ), double( merite_de( a2, NewtonOptions::LIN ) ),
                                         double( merite_de( a2, NewtonOptions::LOG ) ) );
                            break;
                        }
                    }
                    if ( tp <= TF( 1e-6 ) )
                        std::printf( "      %4.2f %4.2f %4.2f     %-10s %-4d\n", double( l0 ), double( l1 ),
                                     double( l2 ), "AUCUN", nd );
                    std::fflush( stdout );
                }
                st.fin = "COMBI";
                return false;
            }

            // ---- LE MODELE MULTI-DIRECTIONS, MIS A L'EPREUVE ( `Ecrasement.h` : `PolyMulti` )
            //
            // Trois comparaisons, et elles ne disent pas la meme chose :
            //   * contre `ModeleCellule::aire`, qui evalue la meme aire a combinatoire figee mais
            //     point par point : c'est un controle PUREMENT ALGEBRIQUE, il doit tomber au
            //     dernier chiffre. S'il ne tombe pas, les coefficients sont faux.
            //   * contre le VRAI diagramme : c'est l'erreur de combinatoire, la seule qui decide si
            //     le modele sert a quelque chose.
            //   * `rayon` contre `|t|inf` : la ou le modele est PROUVE exact.
            // ---- LE SPAN CONSTRUIT PROGRESSIVEMENT, A CONNECTIVITE GELEE ( `--span K` )
            //
            // LE PROTOCOLE, pose explicitement parce que c'est lui qu'on mesure. On est dans la phase
            // `log`, densite fixe. Le diagramme du point de depart `w` donne les aires, le laplacien
            // `L`, et le polynome EXACT des aires sur un span de directions ( § 22 ). Ensuite, et
            // jusqu'a la fin de la construction, PLUS AUCUN DIAGRAMME :
            //
            //   1. `d_1` est la direction de Newton `log` en `w` ;
            //   2. sur le span courant `{ d_1 ... d_k }`, on minimise le merite `log2` PAR LE MODELE
            //      -- descente de gradient, une grille etant hors de portee des K = 4 dimensions ;
            //   3. au point optimal, les aires MODELISEES donnent le second membre `log`, qu'on
            //      resout avec LE MEME laplacien : c'est `d_{k+1}`. Le span grandit, on retourne en 2.
            //
            // DEUX APPROXIMATIONS, a connaitre avant de lire les chiffres. Le laplacien reste celui du
            // depart ( a connectivite fixe la vraie matrice y serait calculable, les longueurs de
            // facette etant affines en `w`, mais ce n'est pas branche ) ; et les aires du point
            // optimal viennent du modele, pas d'un diagramme. C'est exactement ce que faisait la sonde
            // du § 24.15, ou `cos( d, e ) ~ -0.9` montrait que la direction obtenue est franchement
            // neuve.
            //
            // CE QUE LA SORTIE DONNE, et pourquoi : `log2` atteignable par dimension du span, MODELE
            // CONTRE VERITE ( un vrai diagramme au point retenu, pour le diagnostic seulement ), plus
            // l'aire minimale et le cosinus de la nouvelle direction au span deja la. C'est ca qui dit
            // ce qu'il faut attendre du solveur lineaire : si un span de trois ou quatre directions
            // descend `log2` loin sans toucher au diagramme, alors le solveur devient le poste
            // dominant et c'est lui qu'il faut rendre rapide.
            if ( it == o.span ) {
                if constexpr ( PD::dim == 2 ) {
                    // EN MODE B ON NE DISPOSE QUE DE `w'` ET `w''` : l'ordre trois demanderait le
                    // terme croise `Q( w', w'' )`, donc une construction a deux directions, ce qui
                    // perdrait l'avantage de la variante. On plafonne donc a 2 -- sans ca la boucle
                    // lisait un vecteur vide et plantait.
                    const int KM = std::max( 1, std::min( o.span_mode == 1 ? 2 : o.span_k,
                                                          int( PolyMulti::KMAX ) ) );
                    const int nth = std::max( 1, par.threads );
                    std::vector<TF> dd[ PolyMulti::KMAX ];
                    dd[ 0 ] = d;
                    std::vector<PolyMulti> pm;
                    std::vector<TF> tb( PolyMulti::KMAX, 0 ), aso, bb;
                    w2.resize( n );
                    const TF l2_0 = merite_de( a, NewtonOptions::LOG2 );
                    std::printf( "    SPAN it %d, n %d, log2 au depart %.6e\n", it, int( n ), double( l2_0 ) );
                    std::printf( "      %-3s %-11s %-12s %-12s %-9s %-10s %-8s %s\n", "K", "|t|inf",
                                 "log2 modele", "log2 REEL", "ecart", "aire min", "|dperp|", "coefficients" );

                    // le merite `log2` du modele et l'aire minimale, pour `np` points de `K` coords
                    auto mods = [ & ]( int K, const std::vector<TF> &pts, std::vector<TF> &s2,
                                       std::vector<TF> &mn ) {
                        const int np = int( pts.size() / K );
                        std::vector<TF> as2( size_t( nth ) * np, 0 ), amn( size_t( nth ) * np, INFINI );
                        parallel_for( n, par, [ & ]( SI i, int th ) {
                            const PolyMulti &q = pm[ i ];
                            if ( q.etat != PolyCellule::OK ) return;
                            TF *ps = &as2[ size_t( th ) * np ], *pmn = &amn[ size_t( th ) * np ];
                            for ( int p = 0; p < np; ++p ) {
                                const TF A = q( &pts[ size_t( p ) * K ], K );
                                if ( protegee[ i ] && A < pmn[ p ] ) pmn[ p ] = A;
                                if ( A > 0 ) { const TF g = std::log( A / nu[ i ] ); ps[ p ] += g * g; }
                                else ps[ p ] = INFINI;
                            }
                        } );
                        s2.assign( np, 0 ); mn.assign( np, INFINI );
                        for ( int th = 0; th < nth; ++th )
                            for ( int p = 0; p < np; ++p ) {
                                const TF v = as2[ size_t( th ) * np + p ];
                                s2[ p ] = s2[ p ] == INFINI || v == INFINI ? INFINI : s2[ p ] + v;
                                mn[ p ] = std::min( mn[ p ], amn[ size_t( th ) * np + p ] );
                            }
                        for ( int p = 0; p < np; ++p ) s2[ p ] = s2[ p ] == INFINI ? INFINI : std::sqrt( s2[ p ] );
                    };

                    // ---- VARIANTE B : LES DERIVEES DE `w( t )` AU DEPART, en UNE construction
                    //
                    // A connectivite fixe, `A` est EXACTEMENT quadratique en `w` : `A = a + L d + Q( d, d )`.
                    // Le long d'un chemin, `A' = L w'` et `A'' = L w'' + 2 Q( w', w' )`, et le dernier
                    // terme est exactement le coefficient quadratique du modele a UNE direction.
                    //
                    // On impose au residu `log` de decroitre lineairement, `r( t ) = ( 1 - t ) r_0`.
                    // En derivant deux fois avec `u_i = g'( x_i ) / nu_i = 1 / A_i` ( le `log` ) :
                    //
                    //      u_i A'_i = cste   =>   A''_i = - ( u'_i / u_i ) A'_i = ( A'_i )^2 / A_i
                    //
                    // et comme `A'_i = ( L w' )_i = b_i` au depart, il vient
                    //
                    //      L w'' = b^2 / a - 2 q
                    //
                    // avec `q` le coefficient quadratique du modele a une direction. UNE construction,
                    // un solve de plus, zero diagramme -- la ou la variante A demande une construction
                    // par direction et rend des colineaires ( § 24.21 ).
                    if ( o.span_mode == 1 ) {
                        const TF *dp1[ 1 ] = { dd[ 0 ].data() };
                        pd.set_weights( w.data(), par );
                        polynomes_multi( pd, P, w, dp1, 1, par, pm );
                        // CONTROLE : `dA/dt` du modele doit valoir `b` = `L d_1`. S'il ne tombe pas,
                        // ou le modele ou la derivation est fausse, et rien de ce qui suit ne vaut.
                        TF e1 = 0, n1 = 0;
                        for ( SI i = 0; i < n; ++i ) {
                            if ( pm[ i ].etat != PolyCellule::OK ) continue;
                            e1 = std::max( e1, std::fabs( pm[ i ].g[ 0 ] - b[ i ] ) );
                            n1 = std::max( n1, std::fabs( b[ i ] ) );
                        }
                        std::printf( "      controle dA/dt contre L d_1 : ecart max %.3e pour |b|max %.3e"
                                     " ( relatif %.2e )\n", double( e1 ), double( n1 ),
                                     double( n1 > 0 ? e1 / n1 : TF( 0 ) ) );
                        std::vector<TF> rhs( n, 0 );
                        for ( SI i = 0; i < n; ++i ) {
                            const TF q2 = pm[ i ].etat == PolyCellule::OK ? 2 * pm[ i ].q[ 0 ] : TF( 0 );
                            rhs[ i ] = b[ i ] * b[ i ] / std::max( a[ i ], eps ) - q2;
                        }
                        dd[ 1 ].assign( n, 0 );
                        if ( lin.sait_encore() ) lin.resout_encore( rhs, dd[ 1 ] );
                        else                     lin.resout( L, rhs, dd[ 1 ] );
                        dd[ 1 ][ 0 ] = 0;
                        TF ps = 0, na = 0, nb = 0;
                        for ( SI i = 0; i < n; ++i ) {
                            ps += dd[ 1 ][ i ] * dd[ 0 ][ i ];
                            na += dd[ 1 ][ i ] * dd[ 1 ][ i ];
                            nb += dd[ 0 ][ i ] * dd[ 0 ][ i ];
                        }
                        std::printf( "      w'' : cos( w', w'' ) %.4f, |w''|/|w'| %.3f\n",
                                     double( na > 0 && nb > 0 ? ps / std::sqrt( na * nb ) : TF( 0 ) ),
                                     double( nb > 0 ? std::sqrt( na / nb ) : TF( 0 ) ) );
                    }

                    for ( int K = 1; K <= KM; ++K ) {
                        // ---- 1. le modele EXACT sur le span courant ( `pd` est aux poids `w` )
                        const TF *dp[ PolyMulti::KMAX ];
                        for ( int k = 0; k < K; ++k ) dp[ k ] = dd[ k ].data();
                        // LE VRAI DIAGRAMME DE L'ETAPE 3 A DEPLACE `pd` : il faut le ramener en `w`,
                        // sinon le modele est bati autour du point de verification et tout ce qui
                        // suit est faux. Mesure avant correction : `log2` du modele a `inf` des
                        // `K = 2`, et un span qui ne grandissait pas. Ce cout n'existe que parce
                        // qu'on verifie ; l'algorithme, lui, ne verifierait qu'une fois.
                        pd.set_weights( w.data(), par );
                        polynomes_multi( pd, P, w, dp, K, par, pm );

                        // ---- 2. minimiser `log2` sur le span. Depart : l'optimum precedent avec une
                        // coordonnee neuve a zero ; pour `K = 1` une echelle dyadique depuis `t0`.
                        // L'ADMISSIBILITE EST `A > 0`, PAS LE PLANCHER `eps`. Pour `log2` le merite
                        // vaut l'infini des qu'une cellule se vide, donc le plancher est REDONDANT
                        // ( § 21.1 et § 24.10 ) -- et l'imposer bloque tout : l'optimum a `K = 1` est
                        // colle au plancher, et de la aucune direction de descente ne passe. Mesure :
                        // avec `>= eps` le span ne grandissait pas du tout.
                        std::vector<TF> cur( K, 0 ), best( K, 0 ), s2, mn;
                        TF mb = INFINI;
                        if ( K == 1 ) {
                            std::vector<TF> pts;
                            for ( int ia = 0; ia < 24; ++ia ) pts.push_back( o.t0 / TF( SI( 1 ) << ia ) );
                            mods( 1, pts, s2, mn );
                            for ( int p = 0; p < int( s2.size() ); ++p )
                                if ( s2[ p ] < mb ) { mb = s2[ p ]; best[ 0 ] = pts[ p ]; }
                        } else {
                            // ON REPART DE L'OPTIMUM PRECEDENT, EXACTEMENT. Le span le contient, donc
                            // `log2` ne peut que descendre : ce qu'on lit a `K` est alors exactement ce
                            // que la direction ajoutee APPORTE. Un depart en retrait ( j'avais mis
                            // 0.95 ) melange deux effets et fait croire a une perte.
                            for ( int k = 0; k + 1 < K; ++k ) best[ k ] = tb[ k ];
                            std::vector<TF> pts( best.begin(), best.end() );
                            mods( K, pts, s2, mn );
                            mb = s2[ 0 ];
                        }
                        // ---- LA GRILLE, quand elle est a portee ( `--span-grille N`, K <= 2 ).
                        //
                        // Le modele ne coute rien a evaluer, donc on peut verifier la descente par une
                        // GRILLE COMPLETE : si elle trouve mieux, la descente etait piegee. C'est le
                        // controle que la minimisation elle-meme n'est pas le probleme.
                        if ( o.span_grille > 1 && K <= 2 ) {
                            const int NG = o.span_grille;
                            const TF t1m = std::fabs( best[ 0 ] ) > 0 ? 2 * std::fabs( best[ 0 ] ) : o.t0;
                            std::vector<TF> pts;
                            if ( K == 1 ) {
                                for ( int i1 = 1; i1 <= NG; ++i1 ) pts.push_back( t1m * i1 / TF( NG ) );
                            } else {
                                for ( int i1 = 1; i1 <= NG; ++i1 )
                                for ( int i2 = 0; i2 <= NG; ++i2 ) {
                                    pts.push_back( t1m * i1 / TF( NG ) );
                                    pts.push_back( t1m * ( 2 * i2 - NG ) / TF( NG ) );
                                }
                            }
                            mods( K, pts, s2, mn );
                            TF mg = INFINI;
                            std::vector<TF> bg( K, 0 );
                            for ( int p = 0; p < int( s2.size() ); ++p )
                                if ( s2[ p ] < mg ) {
                                    mg = s2[ p ];
                                    for ( int k = 0; k < K; ++k ) bg[ k ] = pts[ size_t( p ) * K + k ];
                                }
                            std::printf( "      grille %d^%d sur [ 0, %.4g ] x +/- : min %.6e en ( %.4g, %.4g )%s\n",
                                         NG, K, double( t1m ), double( mg ), double( bg[ 0 ] ),
                                         double( K > 1 ? bg[ 1 ] : TF( 0 ) ),
                                         mg < mb ? "  ( MIEUX que le depart de la descente )" : "" );
                            // ---- LA CARTE, pour VOIR le paysage : si `log2` est lisse et unimodal,
                            // une descente correcte doit y arriver, et l'echec est dans la descente.
                            if ( K == 2 && o.span_carte ) {
                                std::printf( "        carte log2 ( lignes t1 croissant, colonnes t2 de -%.3g a %.3g )\n",
                                             double( t1m ), double( t1m ) );
                                const int SP = 8;
                                for ( int i1 = NG; i1 >= 1; i1 -= std::max( 1, NG / SP ) ) {
                                    std::printf( "        t1=%-8.4g", double( t1m * i1 / TF( NG ) ) );
                                    for ( int i2 = 0; i2 <= NG; i2 += std::max( 1, NG / SP ) ) {
                                        const int p = ( i1 - 1 ) * ( NG + 1 ) + i2;
                                        const TF v = s2[ p ];
                                        if ( v == INFINI ) std::printf( " %9s", "inf" );
                                        else std::printf( " %9.4g", double( v ) );
                                    }
                                    std::printf( "\n" );
                                }
                            }
                            if ( mg < mb ) { mb = mg; best = bg; }
                        }

                        // ---- LA MINIMISATION PAR NEWTON SUR LA BARRIERE ( `span_hess` )
                        //
                        // Le § 24.25 a montre pourquoi une descente de gradient echoue ici : le
                        // domaine admissible est une BANDE etroite bordee de `+infini`, et ces parois
                        // ne sont pas une contrainte exterieure -- c'est l'objectif lui-meme, `log2`
                        // valant l'infini des qu'une cellule se vide. On minimise donc une BARRIERE
                        // sur un domaine mince, le cas d'ecole ou le gradient seul rampe et ou la
                        // hessienne rattrape tout : elle est enorme EN TRAVERS de la vallee et petite
                        // LE LONG, donc elle reechelonne exactement ce qu'il faut.
                        //
                        // Avec `F = sum g^2` et `g_i = log( A_i / nu_i )` :
                        //
                        //      dF/dt_k    = sum_i 2 g_i A'_k / A_i
                        //      d2F/dt_kdt_l = sum_i 2 [ ( 1 - g_i ) A'_k A'_l / A_i^2 + g_i A''_kl / A_i ]
                        //
                        // avec `A''_kk = 2 q_kk` et `A''_kl = q_kl`. Tout est analytique, le systeme
                        // est `K x K` avec `K <= 4`, et la recherche lineaire FAIT CROITRE le pas --
                        // l'autre defaut du § 24.25.
                        if ( o.span_hess > 0 ) {
                            const int NQ2 = PolyMulti::KMAX * PolyMulti::KMAX;
                            TF lam = 0;
                            for ( int pas = 0; pas < o.span_hess; ++pas ) {
                                std::vector<TF> acc( size_t( nth ) * ( K + NQ2 ), 0 );
                                parallel_for( n, par, [ & ]( SI i, int th ) {
                                    const PolyMulti &q = pm[ i ];
                                    if ( q.etat != PolyCellule::OK ) return;
                                    const TF A = q( best.data(), K );
                                    if ( ! ( A > 0 ) ) return;
                                    TF da[ PolyMulti::KMAX ];
                                    q.gradient( best.data(), K, da );
                                    const TF g = std::log( A / nu[ i ] );
                                    TF *gr = &acc[ size_t( th ) * ( K + NQ2 ) ];
                                    TF *he = gr + K;
                                    for ( int k = 0; k < K; ++k ) gr[ k ] += 2 * g * da[ k ] / A;
                                    for ( int k = 0; k < K; ++k )
                                        for ( int l = 0; l <= k; ++l ) {
                                            const TF d2 = k == l ? 2 * q.q[ k * ( k + 1 ) / 2 + k ]
                                                                 : q.q[ k * ( k + 1 ) / 2 + l ];
                                            he[ k * PolyMulti::KMAX + l ] +=
                                                2 * ( ( 1 - g ) * da[ k ] * da[ l ] / ( A * A ) + g * d2 / A );
                                        }
                                } );
                                TF gr[ PolyMulti::KMAX ] = {}, H[ PolyMulti::KMAX ][ PolyMulti::KMAX ] = {};
                                for ( int th = 0; th < nth; ++th ) {
                                    const TF *src = &acc[ size_t( th ) * ( K + NQ2 ) ];
                                    for ( int k = 0; k < K; ++k ) gr[ k ] += src[ k ];
                                    for ( int k = 0; k < K; ++k )
                                        for ( int l = 0; l <= k; ++l )
                                            H[ k ][ l ] += src[ K + k * PolyMulti::KMAX + l ];
                                }
                                for ( int k = 0; k < K; ++k )
                                    for ( int l = k + 1; l < K; ++l ) H[ k ][ l ] = H[ l ][ k ];
                                TF ng = 0;
                                for ( int k = 0; k < K; ++k ) ng = std::max( ng, std::fabs( gr[ k ] ) );
                                if ( ! ( ng > 0 ) ) break;

                                // `H + lam diag( H )` puis Gauss : la regularisation de
                                // Levenberg-Marquardt, qui ramene vers le gradient si Newton derape
                                bool pris_un = false;
                                for ( int essai = 0; essai < 24 && ! pris_un; ++essai ) {
                                    TF M[ PolyMulti::KMAX ][ PolyMulti::KMAX + 1 ];
                                    for ( int k = 0; k < K; ++k ) {
                                        for ( int l = 0; l < K; ++l ) M[ k ][ l ] = H[ k ][ l ];
                                        M[ k ][ k ] += lam * ( std::fabs( H[ k ][ k ] ) + TF( 1e-30 ) );
                                        M[ k ][ K ] = -gr[ k ];
                                    }
                                    bool ok_lin = true;
                                    for ( int c = 0; c < K && ok_lin; ++c ) {
                                        int piv = c;
                                        for ( int r = c + 1; r < K; ++r )
                                            if ( std::fabs( M[ r ][ c ] ) > std::fabs( M[ piv ][ c ] ) ) piv = r;
                                        if ( ! ( std::fabs( M[ piv ][ c ] ) > 0 ) ) { ok_lin = false; break; }
                                        if ( piv != c ) for ( int l = 0; l <= K; ++l ) std::swap( M[ c ][ l ], M[ piv ][ l ] );
                                        for ( int r = 0; r < K; ++r ) {
                                            if ( r == c ) continue;
                                            const TF f = M[ r ][ c ] / M[ c ][ c ];
                                            for ( int l = c; l <= K; ++l ) M[ r ][ l ] -= f * M[ c ][ l ];
                                        }
                                    }
                                    if ( ! ok_lin ) { lam = lam > 0 ? 4 * lam : TF( 1e-3 ); continue; }
                                    TF de[ PolyMulti::KMAX ];
                                    for ( int k = 0; k < K; ++k ) de[ k ] = M[ k ][ K ] / M[ k ][ k ];
                                    // LA RECHERCHE LINEAIRE, QUI FAIT CROITRE LE PAS
                                    TF mu = 1;
                                    for ( int j = 0; j < 40; ++j ) {
                                        std::vector<TF> pts( K );
                                        for ( int k = 0; k < K; ++k ) pts[ k ] = best[ k ] + mu * de[ k ];
                                        mods( K, pts, s2, mn );
                                        if ( s2[ 0 ] < mb ) {
                                            // mieux : on essaye PLUS LOIN avant d'accepter
                                            std::vector<TF> pg( K );
                                            for ( int k = 0; k < K; ++k ) pg[ k ] = best[ k ] + 2 * mu * de[ k ];
                                            std::vector<TF> sg, ng2;
                                            mods( K, pg, sg, ng2 );
                                            if ( sg[ 0 ] < s2[ 0 ] && mu < TF( 64 ) ) { mu *= 2; continue; }
                                            mb = s2[ 0 ]; best = pts; pris_un = true;
                                            lam = lam > TF( 1e-12 ) ? lam / 4 : TF( 0 );
                                            break;
                                        }
                                        mu /= 2;
                                        if ( mu < TF( 1e-12 ) ) break;
                                    }
                                    if ( ! pris_un ) lam = lam > 0 ? 4 * lam : TF( 1e-3 );
                                }
                                if ( ! pris_un ) break;
                            }
                        }

                        cur = best;
                        // ---- LE CONTROLE DU GRADIENT : analytique contre difference finie centree.
                        // S'ils ne tombent pas, la derivee est fausse et l'echec de la descente vient
                        // de la ; s'ils tombent, c'est la descente elle-meme qu'il faut accuser.
                        if ( o.span_carte ) {
                            std::vector<TF> agr( size_t( nth ) * K, 0 );
                            parallel_for( n, par, [ & ]( SI i, int th ) {
                                const PolyMulti &q = pm[ i ];
                                if ( q.etat != PolyCellule::OK ) return;
                                const TF A = q( cur.data(), K );
                                if ( ! ( A > 0 ) ) return;
                                TF da[ PolyMulti::KMAX ];
                                q.gradient( cur.data(), K, da );
                                const TF c = 2 * std::log( A / nu[ i ] ) / A;
                                for ( int k = 0; k < K; ++k ) agr[ size_t( th ) * K + k ] += c * da[ k ];
                            } );
                            std::vector<TF> gra( K, 0 );
                            for ( int th = 0; th < nth; ++th )
                                for ( int k = 0; k < K; ++k ) gra[ k ] += agr[ size_t( th ) * K + k ];
                            // la difference finie sur `sum g^2`, donc sur le CARRE du merite
                            std::printf( "        gradient de sum g^2 : analytique / difference finie\n" );
                            for ( int k = 0; k < K; ++k ) {
                                const TF ek = std::max( std::fabs( cur[ k ] ), TF( 1e-3 ) ) * TF( 1e-5 );
                                std::vector<TF> pp( cur ), pq( cur ), ss, nn;
                                pp[ k ] += ek; pq[ k ] -= ek;
                                std::vector<TF> pts( pp );
                                pts.insert( pts.end(), pq.begin(), pq.end() );
                                mods( K, pts, ss, nn );
                                const TF df = ( ss[ 0 ] * ss[ 0 ] - ss[ 1 ] * ss[ 1 ] ) / ( 2 * ek );
                                std::printf( "          k=%d  %-14.6e %-14.6e  ecart relatif %.2e\n",
                                             k, double( gra[ k ] ), double( df ),
                                             double( df != 0 ? std::fabs( gra[ k ] - df ) / std::fabs( df ) : TF( 0 ) ) );
                            }
                        }
                        // la descente de gradient projetee, sur le modele : rien ne coute un diagramme
                        TF h = std::fabs( best[ 0 ] ) / 4 + TF( 1e-12 );
                        for ( int pas = 0; pas < o.span_desc && h > TF( 1e-14 ); ++pas ) {
                            std::vector<TF> agr( size_t( nth ) * K, 0 );
                            parallel_for( n, par, [ & ]( SI i, int th ) {
                                const PolyMulti &q = pm[ i ];
                                if ( q.etat != PolyCellule::OK ) return;
                                const TF A = q( cur.data(), K );
                                if ( ! ( A > 0 ) ) return;
                                TF da[ PolyMulti::KMAX ];
                                q.gradient( cur.data(), K, da );
                                const TF c = 2 * std::log( A / nu[ i ] ) / A;
                                for ( int k = 0; k < K; ++k ) agr[ size_t( th ) * K + k ] += c * da[ k ];
                            } );
                            std::vector<TF> gr( K, 0 );
                            for ( int th = 0; th < nth; ++th )
                                for ( int k = 0; k < K; ++k ) gr[ k ] += agr[ size_t( th ) * K + k ];
                            TF ng = 0;
                            for ( int k = 0; k < K; ++k ) ng = std::max( ng, std::fabs( gr[ k ] ) );
                            if ( ! ( ng > 0 ) ) break;
                            std::vector<TF> pts( K );
                            for ( int k = 0; k < K; ++k ) pts[ k ] = cur[ k ] - h * gr[ k ] / ng;
                            mods( K, pts, s2, mn );
                            if ( s2[ 0 ] < mb ) { mb = s2[ 0 ]; cur = pts; best = pts; }
                            else h /= 2;
                        }

                        // ---- 3. LA VERITE au point retenu : un vrai diagramme, pour le diagnostic
                        for ( SI i = 0; i < n; ++i ) {
                            TF v = w[ i ];
                            for ( int k = 0; k < K; ++k ) v += best[ k ] * dd[ k ][ i ];
                            w2[ i ] = v;
                        }
                        w2[ 0 ] = 0;
                        mesures_et_facettes( w2, a2, fa2, pda2 );
                        const TF l2_vrai = merite_de( a2, NewtonOptions::LOG2 );
                        TF am = INFINI, tinf = 0;
                        for ( SI i = 0; i < n; ++i ) if ( protegee[ i ] ) am = std::min( am, a2[ i ] );
                        for ( int k = 0; k < K; ++k ) tinf = std::max( tinf, std::fabs( best[ k ] ) );

                        // ---- 4. la nouvelle direction : aires MODELISEES, second membre `log`, meme `L`
                        TF cosn = 0;
                        if ( K < KM && o.span_mode == 1 ) {
                            TF n1 = 0;                   // deja calculee : on l'orthogonalise
                            for ( SI i = 0; i < n; ++i ) n1 += dd[ K ][ i ] * dd[ K ][ i ];
                            cosn = ortho( dd, K, n1 );
                        } else if ( K < KM ) {
                            aso.assign( n, 0 );
                            for ( SI i = 0; i < n; ++i ) {
                                const TF A = pm[ i ].etat == PolyCellule::OK ? pm[ i ]( best.data(), K ) : a[ i ];
                                aso[ i ] = std::max( A, eps );
                            }
                            membre_de( aso, NewtonOptions::LOG, o.puis, bb );
                            dd[ K ].assign( n, 0 );
                            if ( lin.sait_encore() ) lin.resout_encore( bb, dd[ K ] );
                            else                     lin.resout( L, bb, dd[ K ] );
                            dd[ K ][ 0 ] = 0;
                            TF n1 = 0;
                            for ( SI i = 0; i < n; ++i ) n1 += dd[ K ][ i ] * dd[ K ][ i ];
                            cosn = ortho( dd, K, n1 );
                        }
                        char co[ 96 ] = { 0 };
                        int cp = 0;
                        for ( int k = 0; k < K && cp < 80; ++k )
                            cp += std::snprintf( co + cp, sizeof( co ) - cp, "%s%.4g", k ? " " : "", double( best[ k ] ) );
                        std::printf( "      %-3d %-11.4g %-12.6e %-12.6e %-9.2f %-10.3e %-8.4f %s\n",
                                     K, double( tinf ), double( mb ), double( l2_vrai ),
                                     double( mb > 0 ? 100 * ( l2_vrai - mb ) / mb : TF( 0 ) ),
                                     double( am ), double( cosn ), co );
                        std::fflush( stdout );
                        for ( int k = 0; k < K; ++k ) tb[ k ] = best[ k ];
                    }
                    st.fin = "SPAN";
                    return false;
                }
            }

            if ( it == o.modele ) {
                if constexpr ( PD::dim == 2 ) {
                    const int NR = 3;
                    const int rs[ NR ] = { NewtonOptions::LIN, NewtonOptions::LOG, NewtonOptions::BARRIERE };
                    std::vector<TF> dd[ NR ], bb;
                    for ( int k = 0; k < NR; ++k ) {
                        membre( rs[ k ], o.puis, bb );
                        dd[ k ].assign( n, TF( 0 ) );
                        if ( k == 0 || ! lin.sait_encore() ) lin.resout( L, bb, dd[ k ] );
                        else                                 lin.resout_encore( bb, dd[ k ] );
                    }
                    const TF *dp[ NR ] = { dd[ 0 ].data(), dd[ 1 ].data(), dd[ 2 ].data() };
                    const double tm0 = now();
                    std::vector<PolyMulti> pm;
                    polynomes_multi( pd, P, w, dp, NR, par, pm );
                    const double t_mod = now() - tm0;
                    std::vector<ModeleCellule> mod( n );  // le temoin algebrique
                    parallel_for( n, par, [ & ]( SI k, int ) {
                        typename PD::Cell cel;
                        pd.cellule( k, cel );
                        mod[ pd.ids[ k ] ].depuis( cel, pd.ids[ k ], P, w.data() );
                    } );
                    TF rmin = INFINI;
                    SI n_ok = 0;
                    for ( SI i = 0; i < n; ++i ) { rmin = std::min( rmin, pm[ i ].rayon ); n_ok += pm[ i ].etat == PolyCellule::OK; }
                    std::printf( "    MODELE it %d, n %d, K %d : bati en %.3f s, %d cellules saines, rayon min %.3e\n",
                                 it, int( n ), NR, t_mod, int( n_ok ), double( rmin ) );
                    std::printf( "      %-22s %-9s %-7s %-10s %-10s %-10s  %-12s %-12s  %s\n",
                                 "t ( lin, log, bar )", "|t|inf", "% prouv", "alg. med", "reel med", "reel max",
                                 "pire modele", "pire reel", "sous eps mod/reel" );
                    w2.resize( n );
                    std::vector<TF> del( n ), tk( NR );
                    const int Q = 4;
                    for ( int i1 = 0; i1 <= Q; ++i1 )
                    for ( int i2 = 0; i2 + i1 <= Q; ++i2 ) {
                        if ( ( i1 + i2 ) % 2 ) continue;  // une candidate sur deux : le tableau reste lisible
                        for ( TF tp : { TF( 1 ), TF( 0.25 ), TF( 0.0625 ) } ) {
                            tk[ 0 ] = tp * TF( Q - i1 - i2 ) / Q; tk[ 1 ] = tp * TF( i1 ) / Q; tk[ 2 ] = tp * TF( i2 ) / Q;
                            TF tinf = 0;
                            for ( int k = 0; k < NR; ++k ) tinf = std::max( tinf, std::fabs( tk[ k ] ) );
                            for ( SI i = 0; i < n; ++i ) {
                                del[ i ] = tk[ 0 ] * dd[ 0 ][ i ] + tk[ 1 ] * dd[ 1 ][ i ] + tk[ 2 ] * dd[ 2 ][ i ];
                                w2[ i ] = w[ i ] + del[ i ];
                            }
                            w2[ 0 ] = w[ 0 ];             // la jauge ne bouge pas : `d[ 0 ] = 0` deja
                            mesures_et_facettes( w2, a2, fa2, pda2 );
                            // les trois aires, cellule par cellule
                            std::vector<TF> ealg, ereel;
                            ealg.reserve( n ); ereel.reserve( n );
                            TF pire_mod = 0, pire_reel = 0;
                            SI sous_mod = 0, sous_reel = 0, prouve = 0;
                            for ( SI i = 0; i < n; ++i ) {
                                if ( pm[ i ].etat != PolyCellule::OK ) continue;
                                const TF am = pm[ i ]( tk.data(), NR ), ax = mod[ i ].aire( del.data() );
                                ealg.push_back( std::fabs( am - ax ) / nu[ i ] );
                                ereel.push_back( std::fabs( am - a2[ i ] ) / nu[ i ] );
                                pire_mod = std::max( pire_mod, std::fabs( nu[ i ] - am ) / nu[ i ] );
                                pire_reel = std::max( pire_reel, std::fabs( nu[ i ] - a2[ i ] ) / nu[ i ] );
                                sous_mod += am < eps;
                                sous_reel += a2[ i ] < eps;
                                prouve += pm[ i ].rayon >= tinf;
                            }
                            auto med = []( std::vector<TF> &v ) {
                                if ( v.empty() ) return TF( 0 );
                                std::nth_element( v.begin(), v.begin() + v.size() / 2, v.end() );
                                return v[ v.size() / 2 ];
                            };
                            TF rmax = 0;
                            for ( TF e : ereel ) rmax = std::max( rmax, e );
                            std::printf( "      %6.3f %6.3f %6.3f   %-9.2e %6.1f%%  %-10.2e %-10.2e %-10.2e  %-12.4e %-12.4e  %d / %d\n",
                                         double( tk[ 0 ] ), double( tk[ 1 ] ), double( tk[ 2 ] ), double( tinf ),
                                         100.0 * double( prouve ) / double( std::max<SI>( n_ok, 1 ) ),
                                         double( med( ealg ) ), double( med( ereel ) ), double( rmax ),
                                         double( pire_mod ), double( pire_reel ), int( sous_mod ), int( sous_reel ) );
                            std::fflush( stdout );
                        }
                    }
                }
                st.fin = "MODELE";
                return false;
            }

            // ---- LE PAS PAR LES LIMITES, s'il est demande
            TF t = o.t0, alpha_lim = -1;
            t_sur = -1;
            d_sur.clear();
            TF gain = 1;                                 // ce que `t = 1` vise : `nu` pour `d`, la cible
                                                         // partielle `theta` pour un pas tensoriel

            // ---- L'ORACLE : le MEILLEUR melange des trois directions, trouve a la force brute.
            //
            // Ce n'est pas un algorithme -- il paye ~50 diagrammes par iteration pour choisir. C'est
            // la BORNE SUPERIEURE de ce qu'un modele d'aire sur le span rendrait, et donc le seul
            // chiffre qui dit s'il vaut la peine de le construire. On ne compte que les ITERATIONS.
            if ( o.oracle > 0 ) {
                const int NR = 3;
                const int rs[ NR ] = { NewtonOptions::LIN, NewtonOptions::LOG, NewtonOptions::BARRIERE };
                std::vector<TF> dd[ NR ], bb;
                for ( int k = 0; k < NR; ++k ) {
                    membre( rs[ k ], o.puis, bb );
                    dd[ k ].assign( n, TF( 0 ) );
                    if ( k == 0 || ! lin.sait_encore() ) lin.resout( L, bb, dd[ k ] );
                    else                                 lin.resout_encore( bb, dd[ k ] );
                }
                const int Q = o.oracle;
                TF best = INFINI, bl[ 3 ] = { 1, 0, 0 }, bt = 0;
                w2.resize( n );
                for ( int i1 = 0; i1 <= Q; ++i1 )
                for ( int i2 = 0; i2 + i1 <= Q; ++i2 ) {
                    const TF ll[ 3 ] = { TF( Q - i1 - i2 ) / Q, TF( i1 ) / Q, TF( i2 ) / Q };
                    for ( TF tp = o.t0; tp > o.t_min; tp /= 2 ) {
                        for ( SI i = 0; i < n; ++i )
                            w2[ i ] = w[ i ] + tp * ( ll[ 0 ] * dd[ 0 ][ i ] + ll[ 1 ] * dd[ 1 ][ i ] + ll[ 2 ] * dd[ 2 ][ i ] );
                        w2[ 0 ] = 0;
                        mesures_et_facettes( w2, a2, fa2, pda2 );
                        TF m2 = INFINI;
                        for ( SI i = 0; i < n; ++i )
                            if ( protegee[ i ] && a2[ i ] < m2 ) m2 = a2[ i ];
                        if ( m2 < eps ) continue;        // le plancher ferme : on raccourcit
                        TF v = merite( a2 );             // le juge, celui de `--merite`
                        if ( o.oracle_pire ) {           // ... ou LE VRAI CRITERE D'ARRET, `max |a - nu| / nu`
                            v = 0;
                            for ( SI i = 0; i < n; ++i ) v = std::max( v, std::fabs( nu[ i ] - a2[ i ] ) / nu[ i ] );
                        }
                        if ( v < best ) { best = v; bl[ 0 ] = ll[ 0 ]; bl[ 1 ] = ll[ 1 ]; bl[ 2 ] = ll[ 2 ]; bt = tp; }
                        break;                           // pour ce `lambda`, le plus long pas admissible suffit
                    }
                }
                if ( bt > 0 ) {
                    for ( SI i = 0; i < n; ++i )
                        d[ i ] = bl[ 0 ] * dd[ 0 ][ i ] + bl[ 1 ] * dd[ 1 ][ i ] + bl[ 2 ] * dd[ 2 ][ i ];
                    t = bt;
                    if ( o.trace )
                        std::printf( "      oracle : lambda ( lin %.2f, log %.2f, bar %.2f ), pas %.3e, merite %.6e\n",
                                     double( bl[ 0 ] ), double( bl[ 1 ] ), double( bl[ 2 ] ), double( bt ), double( best ) );
                }
            }

            // ---- LE PAS PAR LE MODELE MULTI-DIRECTIONS ( `--pas modele` )
            //
            // On a mesure trois choses au § 21.6 : le span de plusieurs directions contient des pas
            // bien meilleurs qu'aucune seule ; il faut les choisir sur `max|a - nu|/nu` et pas sur le
            // merite ; et les trouver coutait un diagramme par essai. Ici le modele les evalue tous
            // GRATUITEMENT -- `n` quadratiques en `K` variables -- et un seul diagramme verifie.
            //
            // Le modele est CONSERVATEUR sur le critere, et c'est ce qui rend la recherche sure : des
            // que la combinatoire d'une cellule casse, son aire predite part n'importe ou, donc le
            // `max` predit EXPLOSE. La recherche fuit donc d'elle-meme les regions ou le modele ne
            // vaut rien ( mesure : a `|t|inf = 1` le modele annonce 2533 quand la realite fait 21 ).
            if ( o.pas == NewtonOptions::MODELE ) {
                if constexpr ( PD::dim == 2 ) {
                    const int NR = std::min( o.mod_k, int( PolyMulti::KMAX ) );
                    const int rs[ 4 ] = { res_cur, NewtonOptions::LOG, NewtonOptions::BARRIERE, NewtonOptions::LIN };
                    // LES DESCENTES DE PLUS SONT COMPTEES AVEC L'ALGEBRE LINEAIRE, pas avec le modele :
                    // c'est LA le vrai prix du span ( une resolution par direction supplementaire ), et
                    // les melanger au modele donnait a lire un chiffre pour un autre.
                    double ts0 = now();
                    std::vector<TF> dd[ 4 ], bb;
                    for ( int k = 0; k < NR; ++k ) {
                        dd[ k ].assign( n, TF( 0 ) );
                        if ( k == 0 ) { dd[ 0 ] = d; continue; }  // la direction de Newton est deja resolue
                        membre( rs[ k ], o.puis, bb );
                        if ( lin.sait_encore() ) lin.resout_encore( bb, dd[ k ] );
                        else                     lin.resout( L, bb, dd[ k ] );
                    }
                    st.t_lin += now() - ts0;
                    const double tm0 = now();
                    const TF *dp[ 4 ] = { dd[ 0 ].data(), dd[ 1 ].data(), dd[ 2 ].data(), dd[ 3 ].data() };
                    std::vector<PolyMulti> pm;
                    polynomes_multi( pd, P, w, dp, NR, par, pm );
                    TF best = INFINI, bl[ 4 ] = { 1, 0, 0, 0 }, bt = 0, balim = 0;
                    SI sans_modele = 0;
                    for ( SI i = 0; i < n; ++i ) sans_modele += protegee[ i ] && pm[ i ].etat != PolyCellule::OK;

                    // ---- LA RECHERCHE DANS LE SPAN, sans un seul diagramme
                    //
                    // Pour un `lambda` FIXE, le modele se reduit a une quadratique SCALAIRE en `t` --
                    // donc le plus grand pas qui respecte le plancher est une RACINE, exactement comme
                    // pour une direction seule ( § 7 ), pas une echelle dyadique. Et une fois
                    // `alpha*( lambda )` connu, evaluer le critere a plusieurs fractions de ce pas ne
                    // coute rien : c'est le profil du § 21.2, gratuit, et le coefficient de relaxation
                    // du § 21.5 choisi par la mesure au lieu d'etre regle a la main.
                    //
                    // LES CANDIDATES SE BALAYENT DEDANS, PAS DEHORS. Le premier essai mettait la boucle
                    // sur les candidates AUTOUR de la boucle sur les cellules : 90 passages sur un
                    // tableau de 12.8 Mo, donc 1.15 Go de trafic et 1.09 s par iteration -- QUATRE
                    // diagrammes, ce qui annulait tout le gain. Ici la cellule est chargee UNE fois et
                    // les 75 candidates se jugent sur ses coefficients restes en registres : deux
                    // passages en tout, un par phase.
                    const int Q = std::max( 1, o.mod_q );
                    const int npt = NR == 1 ? 1 : NR == 2 ? Q + 1 : ( Q + 1 ) * ( Q + 2 ) / 2;
                    const int nth = std::max( 1, par.threads );
                    std::vector<TF> lam( size_t( npt ) * 4, TF( 0 ) );
                    for ( int p = 0, i1 = 0, i2 = 0; p < npt; ++p ) {
                        TF *ll = &lam[ size_t( p ) * 4 ];
                        if ( NR == 1 )      ll[ 0 ] = 1;
                        else if ( NR == 2 ) { ll[ 1 ] = TF( p ) / Q; ll[ 0 ] = 1 - ll[ 1 ]; }
                        else {
                            ll[ 1 ] = TF( i1 ) / Q; ll[ 2 ] = TF( i2 ) / Q; ll[ 0 ] = 1 - ll[ 1 ] - ll[ 2 ];
                            if ( ++i2 + i1 > Q ) { i2 = 0; ++i1; }
                        }
                    }

                    // ---- PHASE 1 : le pas admissible de chaque `lambda`, par les racines
                    std::vector<TF> par_al( size_t( nth ) * npt, o.t0 );
                    parallel_for( n, par, [ & ]( SI i, int th ) {
                        if ( ! protegee[ i ] || pm[ i ].etat != PolyCellule::OK ) return;
                        const PolyMulti &pmi = pm[ i ];
                        TF *al = &par_al[ size_t( th ) * npt ];
                        for ( int p = 0; p < npt; ++p ) {
                            const TF *ll = &lam[ size_t( p ) * 4 ];
                            PolyCellule sc;
                            sc.a0 = pmi.c0;
                            for ( int k = 0; k < NR; ++k ) {
                                sc.a1 += pmi.g[ k ] * ll[ k ];
                                for ( int m = 0; m <= k; ++m )
                                    sc.a2 += pmi.q[ k * ( k + 1 ) / 2 + m ] * ll[ k ] * ll[ m ];
                            }
                            al[ p ] = std::min( al[ p ], sc.premiere_racine( eps ) );
                        }
                    } );
                    std::vector<TF> alim( npt, o.t0 );
                    for ( int th = 0; th < nth; ++th )
                        for ( int p = 0; p < npt; ++p )
                            alim[ p ] = std::min( alim[ p ], par_al[ size_t( th ) * npt + p ] );

                    // ---- LES CANDIDATES : `lambda` x fraction du pas admissible
                    // LA RECHERCHE DE RELAXATION, et `--mod-frac 1` la supprime. Elle est gratuite sur le
                    // modele, mais gratuite n'est pas utile : si le choix tombe toujours sur la fraction
                    // la plus longue, autant ne pas la chercher. C'est ce que `--mod-frac` mesure.
                    const TF frac[] = { TF( 0.99 ), TF( 0.9 ), TF( 0.75 ), TF( 0.5 ), TF( 0.25 ) };
                    const int NF = std::min( o.mod_frac, int( sizeof( frac ) / sizeof( frac[ 0 ] ) ) );
                    std::vector<TF> cand;                // `4` coordonnees par candidate
                    std::vector<int> cpt;                // ... et de quel `lambda` elle vient
                    for ( int p = 0; p < npt; ++p ) {
                        if ( ! ( alim[ p ] > 0 ) ) continue;
                        for ( int f = 0; f < NF; ++f ) {
                            const TF tp = std::min( o.t0, frac[ f ] * alim[ p ] );
                            for ( int k = 0; k < 4; ++k ) cand.push_back( k < NR ? tp * lam[ size_t( p ) * 4 + k ] : TF( 0 ) );
                            cpt.push_back( p );
                        }
                    }
                    const int nc = int( cpt.size() );

                    // ---- PHASE 2 : le critere de chaque candidate. `kh` = le rang du pire qu'on
                    // retient ( 1 : le maximum strict ), et on ne garde que les `kh` premiers par thread
                    const SI kh = std::min<SI>( 256, std::max<SI>( 1, SI( o.mod_hors * TF( n ) ) ) );
                    const bool jpire = o.mod_juge == NewtonOptions::PIRE;
                    std::vector<TF> tops, cs1, cs2;
                    std::vector<SI> cnb;
                    if ( nc > 0 ) {
                        if ( jpire ) { tops.assign( size_t( nth ) * nc * kh, TF( 0 ) ); cnb.assign( size_t( nth ) * nc, 0 ); }
                        else { cs1.assign( size_t( nth ) * nc, TF( 0 ) ); cs2.assign( size_t( nth ) * nc, TF( 0 ) );
                               cnb.assign( size_t( nth ) * nc, 0 ); }
                        parallel_for( n, par, [ & ]( SI i, int th ) {
                            if ( pm[ i ].etat != PolyCellule::OK ) return;
                            const PolyMulti &pmi = pm[ i ];
                            const TF nui = nu[ i ], inv = TF( 1 ) / nui;
                            for ( int c = 0; c < nc; ++c ) {
                                const TF am = pmi( &cand[ size_t( c ) * 4 ], NR );
                                const size_t ic = size_t( th ) * nc + c;
                                if ( jpire ) {
                                    const TF e = std::fabs( nui - am ) * inv;
                                    TF *tp = &tops[ ic * size_t( kh ) ];
                                    if ( e > tp[ kh - 1 ] ) {   // le cas frequent est ce seul test
                                        SI j = kh - 1;
                                        while ( j > 0 && tp[ j - 1 ] < e ) { tp[ j ] = tp[ j - 1 ]; --j; }
                                        tp[ j ] = e;
                                    }
                                    ++cnb[ ic ];
                                } else {
                                    const TF gv = g_de( am * inv, o.mod_juge, o.puis );
                                    cs1[ ic ] += gv; cs2[ ic ] += gv * gv; ++cnb[ ic ];
                                }
                            }
                        } );
                    }

                    // ---- ET LE CHOIX
                    std::vector<TF> fus;
                    for ( int c = 0; c < nc; ++c ) {
                        TF pir;
                        if ( jpire ) {
                            // le `kh`-ieme pire du TOUT : il est dans l'union des `kh`-premiers par thread
                            fus.clear();
                            for ( int th = 0; th < nth; ++th ) {
                                const size_t ic = size_t( th ) * nc + c;
                                const SI m = std::min<SI>( kh, cnb[ ic ] );
                                for ( SI j = 0; j < m; ++j ) fus.push_back( tops[ ic * size_t( kh ) + j ] );
                            }
                            if ( fus.empty() ) continue;
                            const SI r = std::min<SI>( kh, SI( fus.size() ) ) - 1;
                            std::nth_element( fus.begin(), fus.begin() + r, fus.end(), std::greater<TF>() );
                            pir = fus[ r ];
                        } else {
                            TF s1 = 0, s2 = 0;
                            SI nv = 0;
                            for ( int th = 0; th < nth; ++th ) {
                                const size_t ic = size_t( th ) * nc + c;
                                s1 += cs1[ ic ]; s2 += cs2[ ic ]; nv += cnb[ ic ];
                            }
                            if ( ! nv ) continue;
                            pir = std::sqrt( std::max( TF( 0 ), s2 - s1 * s1 / TF( nv ) ) );
                        }
                        if ( pir < best ) {
                            best = pir; balim = alim[ cpt[ c ] ];
                            bt = 0;
                            for ( int k = 0; k < NR; ++k ) { bl[ k ] = lam[ size_t( cpt[ c ] ) * 4 + k ]; }
                            for ( int k = 0; k < NR; ++k ) bt = std::max( bt, bl[ k ] > 0 ? cand[ size_t( c ) * 4 + k ] / bl[ k ] : TF( 0 ) );
                        }
                    }
                    if ( bt > 0 ) {
                        for ( SI i = 0; i < n; ++i ) {
                            TF s = 0;
                            for ( int k = 0; k < NR; ++k ) s += bl[ k ] * dd[ k ][ i ];
                            d[ i ] = s;
                        }
                        t = bt;
                    }
                    st.t_lim += now() - tm0;
                    st.nb_cible_res += NR - 1;           // les descentes de plus, sur la meme factorisation
                    if ( o.trace )
                        std::printf( "      modele : lambda ( %.2f %.2f %.2f ), alpha*_mod %.3e, pas %.3e%s,"
                                     " pire PREDIT %.4e%s\n",
                                     double( bl[ 0 ] ), double( bl[ 1 ] ), double( bl[ 2 ] ), double( balim ),
                                     double( bt ), o.mod_limites ? " ( repris par les limites exactes )" : "",
                                     double( best ), sans_modele ? "  ( des cellules protegees SANS modele )" : "" );
                } else if ( it == 0 ) {
                    // le modele n'est ecrit qu'en 2D, et sans lui `--pas modele` retombe en silence sur
                    // les essais : on le DIT, plutot que de laisser lire un chiffre pour un autre
                    std::printf( "      ATTENTION : --pas modele n'existe qu'en 2D ; on retombe sur --pas essais.\n" );
                }
            }

            // ---- LA CIBLE MODIFIEE ( `Cible.h` ) : on ne touche QU'AU SECOND MEMBRE, et on
            // resout sur la meme factorisation -- aucun diagramme, aucun assemblage
            // ( SEULES se branche plus bas, dans la boucle de `essai-limites` )
            if ( o.cible.mode == OptionsCible::PLAFOND || o.cible.mode == OptionsCible::GEL ) {
                const double tc0 = now();
                EtatCible ec;
                ec.u0 = pas_par_flux( L, a, d, eps, ucel, ec.lie0, &sflux );
                ec.vise = std::min( TF( 1 ), o.cible.facteur * ec.u0 );
                if ( ec.lie0 >= 0 && ec.u0 < o.cible.seuil ) {
                    // `resoudre2` : une descente de plus, sur la factorisation deja faite
                    auto resoudre2 = [ & ]( TF lam ) {
                        melange_cible( b, nu, dans, lam, b2 );
                        ++ec.nb_res;
                        if ( lin.sait_encore() ) { lin.resout_encore( b2, d2 ); return true; }
                        return lin.resout( L, b2, d2 );
                    };
                    auto noyau = [ & ]( int anneaux ) {
                        dans.assign( n, 0 );
                        ec.nb_noyau = 0;
                        for ( SI i = 0; i < n; ++i )
                            if ( ucel[ i ] < ec.vise ) { dans[ i ] = 1; ++ec.nb_noyau; }
                        ec.nb_patch = dilate( L, dans, anneaux );
                    };

                    if ( o.cible.mode == OptionsCible::PLAFOND ) {
                        // LE PLAFOND, sans rien modifier : le patch gele rend `d` harmonique
                        // dedans, ce qui est le plus petit gradient qu'un second membre atteigne
                        if ( o.trace )
                            std::printf( "      cible : U* %.3e ( cellule %d ), vise %.3e -- plafond harmonique :\n",
                                         double( ec.u0 ), int( ec.lie0 ), double( ec.vise ) );
                        for ( int r : { 1, 2, 4, 8 } ) {
                            noyau( r );
                            resoudre2( 0 );
                            SI li = -1;
                            const TF u = pas_par_flux( L, a, d2, eps, u2, li );
                            if ( o.trace )
                                std::printf( "              %d anneaux : noyau %d, patch %d ( %.2f %% ) -> U* %.3e"
                                             "  ( x%.2f ), borne par %d %s\n",
                                             r, int( ec.nb_noyau ), int( ec.nb_patch ),
                                             100.0 * double( ec.nb_patch ) / double( n ), double( u ),
                                             double( u / ec.u0 ), int( li ),
                                             li >= 0 && dans[ li ] ? "DEDANS" : "dehors" );
                        }
                        // ... ET L'AUTRE SENS : nourrir le noyau au lieu de le taire
                        dans.assign( n, 0 );
                        for ( SI i = 0; i < n; ++i ) dans[ i ] = ucel[ i ] < ec.vise;
                        TF sn = 0;
                        for ( SI i = 0; i < n; ++i ) sn += nu[ i ];
                        for ( TF kap : { TF( 0.25 ), TF( 1 ), TF( 4 ), TF( 16 ), TF( 64 ) } ) {
                            const TF don = nourri_cible( b, nu, sflux, dans, kap, b2 );
                            ++ec.nb_res;
                            if ( lin.sait_encore() ) lin.resout_encore( b2, d2 ); else lin.resout( L, b2, d2 );
                            SI li = -1;
                            const TF u = pas_par_flux( L, a, d2, eps, u2, li );
                            TF s2 = 0, sb = 0;
                            for ( SI i = 0; i < n; ++i ) { const TF e = b2[ i ] - b[ i ]; s2 += e * e; sb += b[ i ] * b[ i ]; }
                            if ( o.trace )
                                std::printf( "              nourri kappa %5.2f : masse donnee %.2e = %.2f %% de nu,"
                                             " |db|/|b| %.2e -> U* %.3e  ( x%.2f ), borne par %d %s\n",
                                             double( kap ), double( don ), 100.0 * double( don / sn ),
                                             double( sb > 0 ? std::sqrt( s2 / sb ) : TF( 0 ) ), double( u ),
                                             double( u / ec.u0 ), int( li ), li >= 0 && dans[ li ] ? "le noyau" : "une autre" );
                        }
                    } else {
                        // LA DOSE AUTO-REGLEE : `delta_i = kappa ( S_i - ( a_i - eps ) / F )+`, nulle
                        // d'elle-meme sur les cellules qui tiennent deja. `U*( kappa )` n'est PAS
                        // monotone -- sauver les unes en tue d'autres -- donc on balaye et on garde
                        // le meilleur. Chaque essai est une descente de plus, jamais un diagramme.
                        // L'HORIZON : le pas qu'on veut vraiment atteindre. En ESSAI_LIMITES
                        // c'est `beta`, l'essai que l'iteration va faire -- pas un multiple du pas
                        // courant, qui n'est qu'une lecture.
                        // ... mais seulement pour le critere POPULATION : la DOSE, elle, se
                        // regle sur `vise`, sinon le deficit se calcule contre un horizon que
                        // presque aucune cellule n'atteint et tout le diagramme devient « malade ».
                        const TF horiz = o.cible.critere == OptionsCible::POPULATION
                                       && o.pas == NewtonOptions::ESSAI_LIMITES
                                       ? std::min( TF( 1 ), beta ) : ec.vise;
                        auto malades = [ & ]( const std::vector<TF> &U ) {
                            SI c = 0;
                            for ( SI i = 0; i < n; ++i ) c += U[ i ] < horiz;
                            return c;
                        };
                        // ================= LE JUGE POLYNOME ( § 7 ) =================
                        // `U` n'est pas un predicteur d'extinction, c'est une linearisation des
                        // flux en `t = 0`. Le predicteur, c'est l'aire a combinatoire figee. On la
                        // construit pour les seules candidates, puis on lit TOUT le balayage
                        // dessus -- zero calcul de cellule par essai, une seule resolution.
                        bool poly_fait = false;
                        if constexpr ( PD::dim == 2 ) {
                        if ( o.cible.juge == OptionsCible::POLYNOME ) {
                            const double tp0 = now();
                            // LE FILTRE. `U` est pessimiste, donc `U_i < X` contient toutes les
                            // cellules dont la limite du polynome est sous `X` : c'est un bon
                            // crible, a condition de le regler a l'echelle du POLYNOME et non a
                            // celle de `U`. On prend donc la limite trouvee au tour d'avant, et
                            // on compte apres coup ce qu'on a MANQUE.
                            const TF seuil_mod = std::max( o.cible.filtre * ec.vise, seuil_prec );
                            imod.assign( n, -1 );
                            SI nm = 0;
                            for ( SI i = 0; i < n; ++i )
                                if ( ucel[ i ] < seuil_mod ) imod[ i ] = nm++;
                            mods.assign( nm, ModeleCellule{} );
                            parallel_for( n, par, [ & ]( SI k, int ) {
                                const SI id = pd.ids[ k ];
                                if ( imod[ id ] < 0 ) return;
                                typename PD::Cell cel;
                                pd.cellule( k, cel );
                                mods[ imod[ id ] ].depuis( cel, id, P, w.data() );
                            } );
                            // `e` : la correction de direction par unite de `kappa`. Le coeur et
                            // les donneurs ne dependent pas de `kappa`, donc `delta` lui est
                            // proportionnel et UNE resolution sert tout le balayage.
                            SI nc = 0, na = 0, nd = 0;
                            // LA LIMITE LUE SUR LE POLYNOME, pour une dose donnee : trois
                            // evaluations d'aire par candidate suffisent ( elle est de degre 2 ).
                            // LA LIMITE LUE SUR LE POLYNOME, pour une dose donnee : trois
                            // evaluations d'aire par candidate suffisent ( elle est de degre 2 ),
                            // et la geometrie ne bouge pas -- zero calcul de cellule par essai.
                            t1.resize( n ); t2.resize( n );
                            alim.assign( n, INFINI );
                            auto limite_poly = [ & ]( TF kap, SI &lie, bool garde ) {
                                for ( SI i = 0; i < n; ++i ) {
                                    const TF v = d[ i ] + kap * ecor[ i ];
                                    t1[ i ] = v; t2[ i ] = 2 * v;
                                }
                                TF am = INFINI;
                                lie = -1;
                                for ( SI i = 0; i < n; ++i ) {
                                    if ( imod[ i ] < 0 ) continue;
                                    const ModeleCellule &m = mods[ imod[ i ] ];
                                    if ( m.nb < 3 ) continue;
                                    const TF A0 = m.aire( nullptr ), A1 = m.aire( t1.data() ), A2 = m.aire( t2.data() );
                                    PolyCellule q;
                                    q.a0 = A0;
                                    q.a1 = 2 * A1 - TF( 1.5 ) * A0 - TF( 0.5 ) * A2;
                                    q.a2 = TF( 0.5 ) * ( A2 - 2 * A1 + A0 );
                                    const TF r = q.premiere_racine( eps );
                                    if ( garde ) alim[ i ] = r;
                                    if ( r < am ) { am = r; lie = i; }
                                }
                                return am;
                            };
                            SI lp = -1;
                            ecor.assign( n, TF( 0 ) );   // pas encore de correction : `kappa = 0`
                            const TF a_ref = limite_poly( 0, lp, true );
                            seuil_prec = a_ref;
                            // CE QU'ON A MANQUE : une cellule non modelisee dont `U` tombe sous la
                            // limite trouvee aurait pu border. On le compte plutot que l'ignorer.
                            SI manques = 0;
                            for ( SI i = 0; i < n; ++i ) manques += imod[ i ] < 0 && ucel[ i ] < a_ref;
                            // LE COEUR, DESIGNE PAR LE BON PREDICTEUR : les cellules dont le
                            // POLYNOME dit qu'elles s'eteignent avant la cible.
                            const TF cible_poly = std::min( TF( 1 ), o.cible.facteur * a_ref );
                            coeurp.assign( n, 0 );
                            SI ncp = 0;
                            for ( SI i = 0; i < n; ++i )
                                if ( imod[ i ] >= 0 && alim[ i ] < cible_poly ) { coeurp[ i ] = 1; ++ncp; }
                            nourri_local( L, b, nu, sflux, a, d, eps, ec.vise, TF( 1 ), o.cible.repris,
                                          o.cible.ep_anneau, b2, dlt, touche, nc, na, nd, &coeurp );
                            ++ec.nb_res;
                            if ( lin.sait_encore() ) lin.resout_encore( dlt, ecor );
                            else if ( ! lin.resout( L, dlt, ecor ) ) ecor.assign( n, TF( 0 ) );
                            TF abest = a_ref, kp = 0;
                            SI lbest = lp;
                            for ( TF kap : { TF( 0.25 ), TF( 0.5 ), TF( 1 ), TF( 2 ), TF( 4 ) } ) {
                                SI li = -1;
                                const TF al = limite_poly( kap, li, false );
                                if ( al > abest ) { abest = al; kp = kap; lbest = li; }
                            }
                            if ( o.trace )
                                std::printf( "      cible/poly : %d modeles ( U < %.2e, %d manques ), limite"
                                             " %.3e  ( U* dit %.3e, soit x%.0f trop petit ), %d a nourrir"
                                             " -> kappa %.2f, limite %.3e  ( x%.2f ), %.3f s\n",
                                             int( nm ), double( seuil_mod ), int( manques ), double( a_ref ),
                                             double( ec.u0 ), double( ec.u0 > 0 ? a_ref / ec.u0 : 0 ),
                                             int( ncp ), double( kp ), double( abest ),
                                             double( a_ref > 0 ? abest / a_ref : 1 ), now() - tp0 );
                            if ( kp > 0 ) {
                                nourri_local( L, b, nu, sflux, a, d, eps, ec.vise, kp, o.cible.repris,
                                              o.cible.ep_anneau, b2, del, touche, nc, na, nd, &coeurp );
                                TF s2 = 0, sb = 0, sn = 0, don = 0;
                                for ( SI i = 0; i < n; ++i ) {
                                    const TF e = b2[ i ] - b[ i ];
                                    s2 += e * e; sb += b[ i ] * b[ i ]; sn += nu[ i ];
                                    if ( del[ i ] > 0 ) don += del[ i ];
                                }
                                for ( SI i = 0; i < n; ++i ) d[ i ] += kp * ecor[ i ];
                                d[ 0 ] = 0;
                                ec.u1 = abest; ec.u0 = a_ref; ec.lie1 = lbest;
                                ec.nb_noyau = nc; ec.lambda = kp;
                                ec.masse = sn > 0 ? don / sn : TF( 0 );
                                ec.ecart = sb > 0 ? std::sqrt( s2 / sb ) : TF( 0 );
                                gain = std::max( TF( 0 ), TF( 1 ) - ec.ecart );
                                ++st.nb_cible_pris;
                                st.cible_gain += std::log( abest / a_ref );
                            }
                            poly_fait = true;
                        }
                        }
                        if ( ! poly_fait ) {
                        ec.u1 = ec.u0;
                        ec.lie1 = ec.lie0;
                        const SI mal0 = malades( ucel );
                        SI mbest = mal0, nc = 0, na = 0, nd = 0;
                        TF kbest = 0, fuite = 0, emax = 0;
                        for ( TF kap : { TF( 0.25 ), TF( 0.5 ), TF( 1 ), TF( 2 ), TF( 4 ) } ) {
                            nourri_local( L, b, nu, sflux, a, d, eps, horiz, kap, o.cible.repris,
                                          o.cible.ep_anneau, b2, del, touche, nc, na, nd );
                            ++ec.nb_res;
                            if ( lin.sait_encore() ) lin.resout_encore( b2, d2 );
                            else if ( ! lin.resout( L, b2, d2 ) ) break;
                            SI li = -1;
                            const TF u = pas_par_flux( L, a, d2, eps, u2, li );
                            const SI m = malades( u2 );
                            const bool mieux = o.cible.critere == OptionsCible::POPULATION
                                             ? m < mbest : u > ec.u1;
                            if ( mieux ) {
                                mbest = m; ec.u1 = u; ec.lie1 = li; kbest = kap; dgard = d2;
                                fuite = fuite_direction( d, d2, touche, emax );
                            }
                        }
                        ec.lie0 = mal0;                      // la trace : malades au depart
                        ec.nb_patch = mbest;                 // ... et ce qu'il en reste
                        ec.lambda = kbest;
                        if ( kbest > 0 ) {
                            const TF don = nourri_local( L, b, nu, sflux, a, d, eps, horiz, kbest,
                                                         o.cible.repris, o.cible.ep_anneau,
                                                         b2, del, touche, nc, na, nd );
                            TF s2 = 0, sb = 0, sn = 0;
                            for ( SI i = 0; i < n; ++i ) {
                                const TF e = b2[ i ] - b[ i ];
                                s2 += e * e; sb += b[ i ] * b[ i ]; sn += nu[ i ];
                            }
                            ec.nb_noyau = nc;
                            ec.masse = sn > 0 ? don / sn : TF( 0 );
                            ec.ecart = sb > 0 ? std::sqrt( s2 / sb ) : TF( 0 );
                            d.swap( dgard );
                            ++st.nb_cible_pris;
                            // LE MERITE RESTE SUR `nu` : a l'ordre un le residu devient
                            // `( 1 - t ) r - t delta`, donc la decroissance garantie n'est plus
                            // `t / 2` mais `( 1 - |delta| / |r| ) t / 2`. On le dit au test.
                            gain = std::max( TF( 0 ), TF( 1 ) - ec.ecart );
                            st.cible_gain += std::log( ec.u1 / ec.u0 );
                        }
                        if ( o.trace )
                            std::printf( "      cible : U* %.3e -> %.3e ( x%.2f ), malades %d -> %d, horizon %.3e,"
                                         " kappa %.2f, %d nourries en %d amas, masse %.2f %%,"
                                         " |db|/|b| %.2e, FUITE %.1f %%, %d res%s\n",
                                         double( ec.u0 ), double( ec.u1 ), double( ec.u1 / ec.u0 ),
                                         int( ec.lie0 ), int( ec.nb_patch ), double( horiz ),
                                         double( ec.lambda ), int( ec.nb_noyau ), int( na ),
                                         100.0 * double( ec.masse ), double( ec.ecart ),
                                         100.0 * double( fuite ), ec.nb_res,
                                         kbest > 0 ? "" : "  ( RIEN NE FAIT MIEUX )" );
                        }
                    }
                }
                st.nb_cible_res += ec.nb_res;
                st.t_cible += now() - tc0;
            }

            // LEVIER 2 : en `grille2 --g2-modele`, `alpha*` sort des RACINES DU MODELE ( plus bas ),
            // pas d'une passe de limites globale -- qui coutait 0.12 s par iteration.
            if ( o.pas != NewtonOptions::ESSAIS && o.pas != NewtonOptions::ESSAI_LIMITES
                 && ( o.pas != NewtonOptions::GRILLE2 || ! o.g2_modele )
                 && ( o.pas != NewtonOptions::MODELE || o.mod_limites )
                 && ( o.pas != NewtonOptions::MERITE || o.mer_limites ) ) {
                if constexpr ( PD::dim == 2 ) {
                    t0 = now();
                    OptionsLimites ol = o.lim;
                    ol.niveau = eps;
                    ol.global = true;
                    pd.set_weights( w.data(), par );
                    limites( pd, P, w, d, par, ol, lim, Voisinage{ L.row.data(), L.col.data() } );
                    // le minimum des limites TROUVEES ; une borne ( HORIZON ) ne compte que si elle
                    // est sous ce minimum, et alors comme lui. Rien avant l'horizon : le pas plein.
                    alpha_lim = INFINI;
                    for ( SI i = 0; i < n; ++i ) {
                        st.nb_cell_lim += lim[ i ].tours;
                        if ( lim[ i ].etat != LimiteCellule::VIDE_AU_DEPART && lim[ i ].etat != LimiteCellule::HORIZON )
                            alpha_lim = std::min( alpha_lim, lim[ i ].alpha );
                    }
                    for ( SI i = 0; i < n; ++i )
                        if ( lim[ i ].etat == LimiteCellule::HORIZON && lim[ i ].alpha < ol.horizon )
                            alpha_lim = std::min( alpha_lim, lim[ i ].alpha );
                    // ---- LE PAS TENSORIEL : quand la direction de Newton est bloquee bien avant le
                    // pas plein, on demande au modele quadratique la direction de la cible partielle
                    // `theta = theta_mult * alpha*`, puis ses propres limites
                    if ( o.pas == NewtonOptions::TENSEUR && alpha_lim < TF( 0.2 ) ) {
                        const TF theta = std::min( TF( 1 ), o.theta_mult * alpha_lim );
                        std::vector<ModeleCellule> mod( n );
                        parallel_for( n, par, [ & ]( SI k, int ) {
                            typename PD::Cell cel;
                            pd.cellule( k, cel );
                            mod[ pd.ids[ k ] ].depuis( cel, pd.ids[ k ], P, w.data() );
                        } );
                        PasTensoriel pt;
                        std::vector<TF> delta;
                        const TF rm = pt.resout( mod, a, nu, d, theta, par, lin, delta );
                        st.t_tenseur += pt.t;
                        ++st.nb_tenseur;
                        // les limites de la direction corrigee ( `delta` deja a l'echelle : horizon 1 )
                        limites( pd, P, w, delta, par, ol, lim, Voisinage{ L.row.data(), L.col.data() } );
                        TF al2 = INFINI;
                        for ( SI i = 0; i < n; ++i ) {
                            st.nb_cell_lim += lim[ i ].tours;
                            if ( lim[ i ].etat != LimiteCellule::VIDE_AU_DEPART && lim[ i ].etat != LimiteCellule::HORIZON )
                                al2 = std::min( al2, lim[ i ].alpha );
                        }
                        for ( SI i = 0; i < n; ++i )
                            if ( lim[ i ].etat == LimiteCellule::HORIZON && lim[ i ].alpha < ol.horizon )
                                al2 = std::min( al2, lim[ i ].alpha );
                        if ( o.trace )
                            std::printf( "      tenseur : theta %.3e, %d it, residu du modele %.2e, limite de delta %.3e\n",
                                         double( theta ), pt.nb_it, double( rm ), double( al2 ) );
                        // on garde `delta` si elle porte plus loin que `alpha* d` ( en unites de theta )
                        const TF portee = std::min( al2, TF( 1 ) ) * theta;
                        if ( portee > alpha_lim ) {
                            d.swap( delta );                     // `d` devient `delta`, `t` en fraction de delta
                            alpha_lim = std::min( al2, ol.horizon );
                            gain = theta;
                        }
                    }
                    if ( alpha_lim >= ol.horizon )
                        t = 1;
                    else if ( o.pas == NewtonOptions::DYADIQUE ) {
                        t = 1;
                        while ( t > alpha_lim && t > TF( 1e-10 ) ) t /= 2;
                    } else
                        t = std::min( TF( 1 ), o.facteur * alpha_lim );
                    st.t_lim += now() - t0;
                }
            }

            // ---- L'ESSAI PUIS LES LIMITES LOCALES : le diagramme du pas d'abord, et si des cellules
            // y passent sous `eps`, leurs limites ( a elles seules ), le pas ramene sous la plus
            // petite, et on recommence -- la non-monotonie peut en reveler d'autres
            bool deja = false;                           // ESSAI_LIMITES : le diagramme en `t` est deja fait
            if ( o.pas == NewtonOptions::ESSAI_LIMITES ) {
                if constexpr ( PD::dim == 2 ) {
                    t = beta;
                    OptionsLimites ol = o.lim;
                    ol.niveau = eps;
                    ol.global = true;
                    std::vector<SI> mauvaises;
                    w2.resize( n );
                    TF t_fait = -1;                      // le pas dont le diagramme est dans `a2`
                    const TF t_max_essai = beta;         // on ne depasse pas l'essai qu'on visait

                    // ---- LE DOSEUR BRANCHE SUR `seules` ( `Cible.h`, mode SEULES )
                    //
                    // Le bon predicteur -- le polynome d'aire a combinatoire figee -- n'est pas
                    // cher PAR CELLULE ; ce qui coute, c'est de savoir LESQUELLES modeliser
                    // ( § 20.6 : un crible en `U` en attrape 99 % ). Ici la liste est deja la,
                    // elle est courte, et elle ne soupconne pas : un diagramme a VU ces cellules
                    // passer sous `eps`. On les nourrit juste assez pour que le polynome les donne
                    // vivantes en `t`, et on re-essaye le meme pas au lieu de le diviser.
                    //
                    // `ModeleCellule::aire` est une aire de LEBESGUE : pas de densite ici.
                    int def_faites = 0;
                    // Rend la limite ATTEINTE ( 0 : rien a faire ). On ne vise pas `beta` -- aller
                    // de 4e-3 a 0.25 serait un facteur soixante sur six mille cellules, et le
                    // doseur a raison de refuser -- mais `facteur` fois la limite COURANTE.
                    auto deformer = [ & ]( const std::vector<SI> &cibles, TF t_max ) -> TF {
                        const double tc0 = now();
                        imod.assign( n, -1 );
                        SI nm = 0;
                        for ( SI i : cibles ) if ( imod[ i ] < 0 ) imod[ i ] = nm++;
                        mods.assign( nm, ModeleCellule{} );
                        pd.set_weights( w.data(), par );
                        parallel_for( n, par, [ & ]( SI k, int ) {
                            const SI id = pd.ids[ k ];
                            if ( imod[ id ] < 0 ) return;
                            typename PD::Cell cel;
                            pd.cellule( k, cel );
                            mods[ imod[ id ] ].depuis( cel, id, P, w.data() );
                        } );
                        // le debit sortant de la direction COURANTE : l'echelle de la dose
                        SI lx = -1;
                        pas_par_flux( L, a, d, eps, ucel, lx, &sflux );
                        coeurp.assign( n, 0 );
                        for ( SI i : cibles ) coeurp[ i ] = 1;
                        SI nc = 0, na = 0, nd = 0;
                        nourri_local( L, b, nu, sflux, a, d, eps, t_max, TF( 1 ), o.cible.repris,
                                      o.cible.ep_anneau, b2, dlt, touche, nc, na, nd, &coeurp );
                        ++st.nb_cible_res;
                        if ( lin.sait_encore() ) lin.resout_encore( dlt, ecor );
                        else if ( ! lin.resout( L, dlt, ecor ) ) { st.t_cible += now() - tc0; return 0; }
                        TF dose = 0, emx = 0, dmx = 0;
                        SI ndose = 0;
                        for ( SI i = 0; i < n; ++i ) {
                            if ( dlt[ i ] > 0 ) { dose += dlt[ i ]; ++ndose; }
                            emx = std::max( emx, std::fabs( ecor[ i ] ) );
                            dmx = std::max( dmx, std::fabs( d[ i ] ) );
                        }
                        t1.resize( n ); t2.resize( n );
                        auto limite = [ & ]( TF kap ) {
                            for ( SI i = 0; i < n; ++i ) {
                                const TF v = d[ i ] + kap * ecor[ i ];
                                t1[ i ] = v; t2[ i ] = 2 * v;
                            }
                            TF am = INFINI;
                            for ( SI i : cibles ) {
                                const ModeleCellule &m = mods[ imod[ i ] ];
                                if ( m.nb < 3 ) continue;
                                const TF A0 = m.aire( nullptr ), A1 = m.aire( t1.data() ), A2 = m.aire( t2.data() );
                                PolyCellule q;
                                q.a0 = A0;
                                q.a1 = 2 * A1 - TF( 1.5 ) * A0 - TF( 0.5 ) * A2;
                                q.a2 = TF( 0.5 ) * ( A2 - 2 * A1 + A0 );
                                am = std::min( am, q.premiere_racine( eps ) );
                            }
                            return am;
                        };
                        const TF a_av = limite( 0 );
                        // UNE CELLULE DEJA AU PLANCHER rend `a_av` nul, et alors tout `kappa`
                        // « atteint » une cible nulle : le doseur declarerait victoire en imposant
                        // un pas nul. On rend la main aux limites.
                        if ( ! ( a_av > 0 ) ) { st.t_cible += now() - tc0; return 0; }
                        // ... ET LE POLYNOME QUI MENT. Ces cellules-la, un DIAGRAMME les a vues
                        // sous `eps` en `t_max` ; si le polynome les donne vivantes au-dela, c'est
                        // qu'un voisin NOUVEAU les a mangees ( § 7 : invisible depuis la cellule
                        // seule ). On ne dose pas sur une prediction qu'on sait fausse.
                        if ( a_av >= t_max ) { st.t_cible += now() - tc0; return 0; }
                        // LE COMPROMIS, et c'est le seul reglage qui compte : on ne vise pas, on
                        // prend la MEILLEURE limite achetable dans un budget de deformation
                        // `| db | / | b |`. Le budget se traduit exactement en plafond sur
                        // `kappa`, `delta` lui etant proportionnel -- aucun essai perdu.
                        TF nb2 = 0, nd1 = 0;
                        for ( SI i = 0; i < n; ++i ) { nb2 += b[ i ] * b[ i ]; nd1 += dlt[ i ] * dlt[ i ]; }
                        nb2 = std::sqrt( nb2 ); nd1 = std::sqrt( nd1 );
                        const TF kmax = o.cible.budget > 0 && nd1 > 0
                                      ? o.cible.budget * nb2 / nd1 : INFINI;
                        TF kb = 0, ab = a_av;
                        for ( TF kap : { TF( 0.0625 ), TF( 0.125 ), TF( 0.25 ), TF( 0.5 ), TF( 1 ),
                                         TF( 2 ), TF( 4 ), TF( 8 ), TF( 16 ), TF( 32 ) } ) {
                            if ( kap > kmax ) break;
                            const TF al = limite( kap );
                            if ( al > ab * TF( 1.02 ) ) { ab = al; kb = kap; }
                        }
                        const bool pris = kb > 0;
                        TF ecart = 0;
                        if ( pris && d_sur.empty() ) d_sur = d;   // LE FILET, pose une seule fois
                        if ( pris ) {
                            nourri_local( L, b, nu, sflux, a, d, eps, t_max, kb, o.cible.repris,
                                          o.cible.ep_anneau, b2, del, touche, nc, na, nd, &coeurp );
                            TF s2 = 0, sb = 0;
                            for ( SI i = 0; i < n; ++i ) {
                                const TF e = b2[ i ] - b[ i ];
                                s2 += e * e; sb += b[ i ] * b[ i ];
                            }
                            ecart = sb > 0 ? std::sqrt( s2 / sb ) : TF( 0 );
                            for ( SI i = 0; i < n; ++i ) d[ i ] += kb * ecor[ i ];
                            d[ 0 ] = 0;
                            // le merite reste sur `nu` : la decroissance garantie tombe a
                            // `( 1 - |delta| / |r| ) t / 2` ( § 20.3 )
                            gain = std::max( TF( 0 ), TF( 1 ) - ecart );
                            ++st.nb_cible_pris;
                            st.cible_gain += std::log( std::min( ab / a_av, TF( 1e6 ) ) );
                        }
                        if ( o.trace )
                            std::printf( "      cible/seules : %d mauvaises modelisees, polynome %.3e -> %.3e"
                                         " ( x%.2f ), %d dosees, kappa %.3f ( plafond %.3f ),"
                                         " |db|/|b| %.2e, masse %.1f %% -- %s ( %.3f s )\n",
                                         int( nm ), double( a_av ), double( ab ), double( ab / a_av ),
                                         int( ndose ), double( kb ), double( std::min( kmax, TF( 99 ) ) ),
                                         double( ecart ), 100.0 * double( kb * dose ),
                                         pris ? "on REESSAYE plus loin" : "rien a acheter",
                                         now() - tc0 );
                        st.t_cible += now() - tc0;
                        return pris ? std::min( ab, t_max ) : TF( 0 );
                    };

                    for ( int tour = 0; tour < 8; ++tour ) {
                        for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
                        w2[ 0 ] = 0;
                        mesures_et_facettes( w2, a2, fa2, pda2 );
                        t_fait = t;
                        mauvaises.clear();
                        for ( SI i = 0; i < n; ++i ) if ( protegee[ i ] && a2[ i ] < eps ) mauvaises.push_back( i );
                        if ( mauvaises.empty() ) break;
                        st.nb_cell_mauvaises += SI( mauvaises.size() );
                        ++st.nb_tours_essai;
                        t0 = now();
                        ol.horizon = t;
                        pd.set_weights( w.data(), par );
                        if ( rho )                       // en masse : la bissection, pas le polynome
                            limites_masse( pd, P, w, d, par, ol, lim, Voisinage{ L.row.data(), L.col.data() }, mauvaises,
                                           [ & ]( const typename PD::Cell &cel ) { return rho->mesure( cel, []( int, TF ) {} ); } );
                        else
                            limites( pd, P, w, d, par, ol, lim, Voisinage{ L.row.data(), L.col.data() }, &mauvaises );
                        TF al = t;
                        for ( SI i : mauvaises ) { st.nb_cell_lim += lim[ i ].tours; al = std::min( al, lim[ i ].alpha ); }
                        st.t_lim += now() - t0;
                        if ( o.trace )
                            std::printf( "      essai t %.3e : %d cellules sous eps, limite locale %.3e ( %.3f s, %d cellules calculees )\n",
                                         double( t ), int( mauvaises.size() ), double( al ), now() - t0,
                                         int( [ & ]{ SI c = 0; for ( SI i : mauvaises ) c += lim[ i ].tours; return c; }() ) );
                        t = o.facteur * al;
                        if ( o.cible.mode == OptionsCible::SEULES && rho == nullptr
                             && def_faites < o.cible.essais
                             && ( o.cible.max_seules <= 0 || SI( mauvaises.size() ) <= o.cible.max_seules ) ) {
                            // LE PAS SUR, celui que les limites viennent de donner a la direction
                            // de Newton : on n'y touche plus, et on y revient si la deformation
                            // ne rend rien.
                            const TF t_garde = t;
                            const bool neuf = d_sur.empty();
                            const TF na = deformer( mauvaises, t_max_essai );
                            if ( na > 0 ) {
                                if ( neuf ) t_sur = t_garde;
                                ++def_faites;
                                t = std::min( t_max_essai, o.facteur * na );
                                continue;                // direction corrigee, pas plus long
                            }
                        }
                        if ( t < o.t_min ) break;
                    }
                    // une limite nulle ( la cellule est deja au plancher, ou la bissection n'a rien
                    // trouve ) n'est pas une raison de stagner : on rend la main aux essais, depuis la
                    // moitie du dernier pas calcule
                    if ( t < o.t_min ) t = t_fait / 2;
                    deja = t == t_fait;
                    alpha_lim = t;                       // pour la trace : le pas retenu
                    // le prochain essai : `mult_ok` fois celui-ci s'il est passe direct, et jamais moins
                    // que `confiance` fois le pas retenu
                    const bool direct = t >= beta;
                    beta = std::min( TF( 1 ), std::max( direct ? o.mult_ok * beta : beta, o.confiance * t ) );
                    // le diagramme en `t` est fait : on rejoint l'amortissement au test du residu
                }
            }

            // ---- L'AMORTISSEMENT, EN DEUX PASSES
            //
            // Une cible deformee peut allonger le pas ET ne plus faire descendre le residu : la
            // direction n'est plus celle de Newton. Sans filet, l'amortissement echoue et Newton
            // sort en STAGNATION -- a un residu qui n'a presque pas bouge, ce qui rend toute
            // comparaison de diagrammes MENSONGERE. On revient donc a la direction de Newton et
            // au pas que les limites lui avaient donne, et on recommence.
            bool pris = false;
            TF t_lim0 = t;
            w2.resize( n );

            // ---- LA RECHERCHE A DEUX VARIABLES ( `--pas grille2` )
            //
            // `w + alpha d + beta ( deplacement precedent )`, sur une grille. La colonne `beta = 0`
            // EST l'echelle dyadique le long de la direction de Newton, donc la grille contient
            // l'algorithme actuel et le resultat se lit comme un ecart a cette colonne. Le premier
            // tour ( pas de direction precedente ) se reduit a cette colonne.
            //
            // Comme la minimisation du merite ( § 24.10 ), ca n'a de sens QUE dans la phase `log` :
            // apres la bascule le juge est `lin`, dont le § 21.1 a etabli qu'il est mauvais.
            if ( o.pas == NewtonOptions::GRILLE2 && ! o.g2_modele && res_cur != NewtonOptions::LIN ) {
                const bool son = o.g2_dir == NewtonOptions::SONDE;

                // ---- LEVIER 2 : `alpha*` PAR LES RACINES DU MODELE, pas par une passe de limites.
                //
                // Ici `pd` est encore aux poids `w` ( le diagramme du point accepte a la fin de
                // l'iteration precedente ), donc un modele a UNE direction ne coute rien. Pour chaque
                // cellule protegee, `A_i( t )` est une quadratique scalaire et le plus grand pas qui
                // respecte le plancher est une RACINE ( § 7 ) : `alpha*` est leur minimum. Ca remplace
                // la passe de limites globale, qui coutait 0.12 s par iteration.
                if ( o.g2_modele ) {
                    if constexpr ( PD::dim == 2 ) {
                        const TF *dp1[ 1 ] = { d.data() };
                        polynomes_multi( pd, P, w, dp1, 1, par, pm2 );
                        TF am = INFINI;
                        for ( SI i = 0; i < n; ++i ) {
                            if ( ! protegee[ i ] || pm2[ i ].etat != PolyCellule::OK ) continue;
                            PolyCellule sc;
                            sc.a0 = pm2[ i ].c0; sc.a1 = pm2[ i ].g[ 0 ]; sc.a2 = pm2[ i ].q[ 0 ];
                            am = std::min( am, sc.premiere_racine( eps ) );
                        }
                        if ( am < INFINI && am > 0 ) {
                            alpha_lim = am;                // pour la trace
                            t = std::min( o.t0, o.facteur * am );
                        }
                    }
                }

                TF best = INFINI, al_b = 0, be_b = 0, al_s = 0;
                int essais = 0;
                std::vector<TF> mat0( size_t( o.g2_na ), INFINI );

                // ---- 1. LA COLONNE `beta = 0` : c'est l'echelle de KMT le long de `d`, et son PREMIER
                // POINT ADMISSIBLE est le point de sonde. L'ordre n'est pas un detail : sonder a `t`
                // sans verifier l'admissibilite donne un point A CELLULES VIDES, dont le laplacien a
                // des lignes nulles. Mesure en 3D, ou aucune passe de limites ne ramene `t` : le
                // solveur y a passe 20032 iterations et 363 s avant d'echouer. Le bon point de sonde
                // est juste AVANT le vidage -- et il est gratuit, c'est celui que l'amortissement
                // allait prendre.
                auto evalue = [ & ]( TF al, TF be ) {     // rend le merite, INFINI si le plancher refuse
                    for ( SI i = 0; i < n; ++i )
                        w2[ i ] = w[ i ] + al * ( d[ i ] + ( be != 0 && son ? be * d_pre[ i ] : TF( 0 ) ) )
                                        + ( be != 0 && ! son ? be * d_pre[ i ] : TF( 0 ) );
                    w2[ 0 ] = 0;
                    mesures_et_facettes( w2, a2, fa2, pda2 );
                    ++essais;
                    TF m2 = INFINI;
                    for ( SI i = 0; i < n; ++i )
                        if ( protegee[ i ] && a2[ i ] < m2 ) m2 = a2[ i ];
                    const TF v = merite( a2 );
                    if ( plancher_actif() && m2 < eps ) return INFINI;
                    if ( v < best ) {
                        best = v; al_b = al; be_b = be;
                        wb = w2; ab = a2; fab = fa2;
                    }
                    return v;
                };
                for ( int ia = 0; ia < o.g2_na; ++ia ) {
                    const TF al = t / TF( SI( 1 ) << ia );
                    if ( al < o.t_min ) break;
                    mat0[ ia ] = evalue( al, 0 );
                    if ( al_s == 0 && mat0[ ia ] < INFINI ) { al_s = al; as_ = a2; fas_ = fa2; }
                    // AVEC LE MODELE, LE RESTE DE LA COLONNE EST GRATUIT : on s'arrete au premier
                    // point admissible, qui est tout ce dont la sonde a besoin. Descendre l'echelle
                    // entiere en diagrammes coutait `g2_na - 1` diagrammes par iteration pour rien
                    // ( mesure : 54 diagrammes au lieu de 24 sur 2D lignes s0.005 ).
                    if ( o.g2_modele && al_s > 0 ) break;
                }

                // ---- 2. LA SECONDE DIRECTION
                bool sonde_resolue = false;
                if ( son ) {
                    d_pre.clear();
                    if ( al_s > 0 ) {
                        double ts0 = now();
                        Lson.assemble( n, fas_ );
                        st.t_asm += now() - ts0;
                        membre_de( as_, res_cur, o.puis, bson );
                        ts0 = now();
                        if ( lin.resout( Lson, bson, d_next ) ) {
                            sonde_resolue = true;
                            d_pre = d_next;               // `d_s - d` : LA COURBURE de la direction en `t`
                            for ( SI i = 0; i < n; ++i ) d_pre[ i ] -= d[ i ];
                            d_pre[ 0 ] = 0;
                        }
                        st.t_lin += now() - ts0;
                    }
                }
                const bool avec = SI( d_pre.size() ) == n;

                // ---- 3bis. LE MODELE POLYNOMIAL EXACT SUR LE SPAN `{ d, e }` ( `--g2-modele` )
                //
                // A connectivite fixe, `A_i` est un POLYNOME exact du deplacement, avec un rayon sous
                // lequel l'exactitude est PROUVEE ( § 22 ). Explorer `( alpha, beta )` ne coute alors
                // plus rien : la grille peut etre fine, et le gradient donne une vraie descente. Le
                // seul cout est la construction ( un diagramme, compte comme tel ).
                bool fait_mod = false;
                if ( o.g2_modele && avec ) {
                    if constexpr ( PD::dim == 2 ) {
                        double tm0 = now();
                        pd.set_weights( w.data(), par );  // la descente a bouge les poids du diagramme
                        st.t_maj += now() - tm0;
                        ++st.nb_diag;
                        const TF *dp[ 2 ] = { d.data(), d_pre.data() };
                        tm0 = now();
                        polynomes_multi( pd, P, w, dp, 2, par, pm2 );
                        const double t_mod = now() - tm0;
                        SI sans = 0;
                        TF rmin = INFINI;
                        for ( SI i = 0; i < n; ++i ) {
                            if ( pm2[ i ].etat != PolyCellule::OK ) ++sans;
                            else rmin = std::min( rmin, pm2[ i ].rayon );
                        }
                        // ---- LE MERITE `log2` DU MODELE, POUR UNE LISTE DE POINTS, EN UN SEUL PASSAGE
                        //
                        // En mono-thread ca coutait plus cher que les diagrammes qu'on economisait :
                        // 126 points x 100000 cellules, mesure a 1.69 s sur 8.32 s de total. Donc un
                        // seul `parallel_for` sur les cellules, avec un accumulateur par ( fil, point )
                        // -- le motif du § 22 -- et une seule traversee de `pm2` pour toute la grille.
                        const int nth = std::max( 1, par.threads );
                        auto mods = [ & ]( const std::vector<TF> &pts, std::vector<TF> &s2,
                                           std::vector<TF> &mn, std::vector<SI> &deh ) {
                            const int np = int( pts.size() / 2 );
                            std::vector<TF> as2( size_t( nth ) * np, 0 ), amn( size_t( nth ) * np, INFINI );
                            std::vector<SI> adh( size_t( nth ) * np, 0 );
                            parallel_for( n, par, [ & ]( SI i, int th ) {
                                const PolyMulti &q = pm2[ i ];
                                if ( q.etat != PolyCellule::OK ) return;
                                TF *ps = &as2[ size_t( th ) * np ], *pm = &amn[ size_t( th ) * np ];
                                SI *pd = &adh[ size_t( th ) * np ];
                                for ( int p = 0; p < np; ++p ) {
                                    const TF tk[ 2 ] = { pts[ 2 * p ], pts[ 2 * p + 1 ] };
                                    if ( q.rayon < std::max( std::fabs( tk[ 0 ] ), std::fabs( tk[ 1 ] ) ) ) ++pd[ p ];
                                    const TF A = q( tk, 2 );
                                    if ( protegee[ i ] && A < pm[ p ] ) pm[ p ] = A;
                                    if ( A > 0 ) { const TF gg = std::log( A / nu[ i ] ); ps[ p ] += gg * gg; }
                                    else ps[ p ] = INFINI;
                                }
                            } );
                            s2.assign( np, 0 ); mn.assign( np, INFINI ); deh.assign( np, 0 );
                            for ( int th = 0; th < nth; ++th )
                                for ( int p = 0; p < np; ++p ) {
                                    const TF v = as2[ size_t( th ) * np + p ];
                                    s2[ p ] = s2[ p ] == INFINI || v == INFINI ? INFINI : s2[ p ] + v;
                                    mn[ p ] = std::min( mn[ p ], amn[ size_t( th ) * np + p ] );
                                    deh[ p ] += adh[ size_t( th ) * np + p ];
                                }
                            for ( int p = 0; p < np; ++p ) s2[ p ] = s2[ p ] == INFINI ? INFINI : std::sqrt( s2[ p ] );
                        };
                        auto mod = [ & ]( TF t1, TF t2, TF *pmin, SI *pdeh ) {
                            const std::vector<TF> pts = { t1, t2 };
                            std::vector<TF> s2, mn; std::vector<SI> deh;
                            mods( pts, s2, mn, deh );
                            if ( pmin ) *pmin = mn[ 0 ];
                            if ( pdeh ) *pdeh = deh[ 0 ];
                            return s2[ 0 ];
                        };
                        // ---- LA GRILLE FINE : tous ses points en UN passage
                        TF mb = INFINI, t1b = 0, t2b = 0;
                        const int NA = std::max( o.g2_na, 1 ), NB = std::max( o.g2_nb, 1 );
                        {
                            std::vector<TF> pts;
                            for ( int ia = 0; ia < NA; ++ia ) {
                                const TF al = t / TF( SI( 1 ) << ia );
                                if ( al < o.t_min ) break;
                                for ( int ib = 0; ib < NB; ++ib ) {
                                    const TF be = o.g2_bpos ? o.g2_bmax * ib / TF( std::max( NB - 1, 1 ) )
                                                            : o.g2_bmax * ( 2 * ib - ( NB - 1 ) ) / TF( std::max( NB - 1, 1 ) );
                                    pts.push_back( al ); pts.push_back( al * be );
                                }
                            }
                            std::vector<TF> s2, mn; std::vector<SI> deh;
                            mods( pts, s2, mn, deh );
                            for ( int p = 0; p < int( s2.size() ); ++p )
                                if ( s2[ p ] < mb && ( ! plancher_actif() || mn[ p ] >= eps ) ) {
                                    mb = s2[ p ]; t1b = pts[ 2 * p ]; t2b = pts[ 2 * p + 1 ];
                                }
                        }
                        const TF m_grille = mb;
                        // ---- LA DESCENTE DE GRADIENT sur le modele, depuis l'argmin de la grille
                        int nd_ok = 0;
                        if ( o.g2_desc > 0 && mb < INFINI ) {
                            TF t1 = t1b, t2 = t2b, h = std::fabs( t1b ) / 4;
                            for ( int k = 0; k < o.g2_desc && h > TF( 1e-12 ); ++k ) {
                                TF gr[ 2 ] = { 0, 0 };
                                const TF tk[ 2 ] = { t1, t2 };
                                std::vector<TF> agr( size_t( nth ) * 2, 0 );
                                parallel_for( n, par, [ & ]( SI i, int th ) {
                                    const PolyMulti &q = pm2[ i ];   // d( sum g^2 )/dt = 2 sum g / A dA/dt
                                    if ( q.etat != PolyCellule::OK ) return;
                                    const TF A = q( tk, 2 );
                                    if ( ! ( A > 0 ) ) return;
                                    TF da[ 2 ];
                                    q.gradient( tk, 2, da );
                                    const TF c = 2 * std::log( A / nu[ i ] ) / A;
                                    agr[ size_t( th ) * 2 ] += c * da[ 0 ];
                                    agr[ size_t( th ) * 2 + 1 ] += c * da[ 1 ];
                                } );
                                for ( int th = 0; th < nth; ++th ) { gr[ 0 ] += agr[ size_t( th ) * 2 ]; gr[ 1 ] += agr[ size_t( th ) * 2 + 1 ]; }
                                const TF ng = std::max( std::fabs( gr[ 0 ] ), std::fabs( gr[ 1 ] ) );
                                if ( ! ( ng > 0 ) ) break;
                                const TF n1 = t1 - h * gr[ 0 ] / ng, n2 = t2 - h * gr[ 1 ] / ng;
                                TF mn;
                                const TF v = mod( n1, n2, &mn, nullptr );
                                if ( v < mb && ( ! plancher_actif() || mn >= eps ) ) {
                                    mb = v; t1 = n1; t2 = n2; t1b = n1; t2b = n2; ++nd_ok;
                                } else
                                    h /= 2;
                            }
                        }
                        // ---- LA VERIFICATION : ELLE N'EST PAS UN SURCOUT, c'est le diagramme de
                        // l'etape d'apres. On est optimiste : on va au point que le modele designe, et
                        // s'il ne passe pas on retombe sur le point de sonde -- deja calcule, deja
                        // admissible -- donc le backtracking est GRATUIT. Et si l'argmin du modele EST
                        // le point de sonde, il n'y a meme pas de second diagramme a faire.
                        TF v_reel = INFINI;
                        const bool argmin_sonde = t1b == al_s && t2b == 0;
                        if ( mb < INFINI && ! argmin_sonde ) {
                            for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t1b * d[ i ] + t2b * d_pre[ i ];
                            w2[ 0 ] = 0;
                            mesures_et_facettes( w2, a2, fa2, pda2 );
                            ++essais;
                            TF m2 = INFINI;
                            for ( SI i = 0; i < n; ++i )
                                if ( protegee[ i ] && a2[ i ] < m2 ) m2 = a2[ i ];
                            v_reel = merite( a2 );
                            if ( ( ! plancher_actif() || m2 >= eps ) && v_reel < best ) {
                                best = v_reel; al_b = t1b; be_b = t1b != 0 ? t2b / t1b : TF( 0 );
                                wb = w2; ab = a2; fab = fa2;
                            }
                        }
                        if ( o.trace && o.g2_trace ) {
                            TF mn; SI deh;
                            mod( t1b, t2b, &mn, &deh );
                            std::printf( "      MODELE sur { d, e } : bati en %.3f s, %d cellules sans modele,"
                                         " rayon min %.3e\n        grille %d x %d : merite modele %.6e"
                                         " -> descente ( %d pas retenus ) %.6e\n        argmin t = ( %.4g, %.4g )"
                                         " soit alpha %.4g beta %.4g | aire min modele %.3e | %d cellules HORS rayon"
                                         " | merite REEL %.6e ( ecart %.2f %% )\n",
                                         t_mod, int( sans ), double( rmin ), NA, NB, double( m_grille ),
                                         nd_ok, double( mb ), double( t1b ), double( t2b ), double( t1b ),
                                         double( t1b != 0 ? t2b / t1b : TF( 0 ) ), double( mn ), int( deh ),
                                         double( v_reel ),
                                         double( v_reel < INFINI && mb > 0 ? 100 * ( v_reel - mb ) / mb : TF( 0 ) ) );
                            std::fflush( stdout );
                        }
                        fait_mod = true;
                    }
                }

                // ---- 3. LE BALAYAGE EN `beta`, aux memes `alpha`
                std::vector<TF> bs;
                if ( avec && o.g2_nb > 1 ) {
                    const int m = o.g2_nb / 2;
                    for ( int k = o.g2_bpos ? 0 : -m; k <= m; ++k ) bs.push_back( o.g2_bmax * k / TF( m ) );
                } else
                    bs.push_back( 0 );
                const int nb = int( bs.size() );
                std::vector<TF> mat( size_t( o.g2_na ) * nb, INFINI );
                for ( int ia = 0; ia < o.g2_na && ! fait_mod; ++ia ) {
                    const TF al = t / TF( SI( 1 ) << ia );
                    if ( al < o.t_min ) break;
                    for ( int ib = 0; ib < nb; ++ib )
                        mat[ size_t( ia ) * nb + ib ] = bs[ ib ] == 0 ? mat0[ ia ] : evalue( al, bs[ ib ] );
                }

                if ( o.trace && o.g2_trace && ! fait_mod ) {
                    // LE COSINUS DIT SI LA GRILLE A DE LA RESOLUTION. Si la seconde direction est
                    // colineaire a `d`, `beta` ne fait que rehausser `alpha` : le plan est un rayon et
                    // un argmin en `beta = 0` ne veut rien dire. A lire AVANT la matrice.
                    TF ps = 0, nd = 0, np = 0;
                    if ( avec )
                        for ( SI i = 0; i < n; ++i ) { ps += d[ i ] * d_pre[ i ]; nd += d[ i ] * d[ i ]; np += d_pre[ i ] * d_pre[ i ]; }
                    const TF cos = avec && nd > 0 && np > 0 ? ps / std::sqrt( nd * np ) : TF( 0 );
                    const TF rap = avec && nd > 0 ? std::sqrt( np / nd ) : TF( 0 );
                    std::printf( "      GRILLE2 ( merite %s, depart %.6e, %s, sonde en %.4g, cos( d, e ) %.4f,"
                                 " |e|/|d| %.3f, PART NEUVE |e_perp|/|d| %.3f ) : lignes alpha, colonnes beta\n        %-10s",
                                 res_cur == NewtonOptions::LOG ? "log" : "?", double( nr ),
                                 son ? "e = d( sonde ) - d" : "e = deplacement precedent", double( al_s ),
                                 double( cos ), double( rap ), double( rap * std::sqrt( std::max( TF( 0 ), 1 - cos * cos ) ) ), "alpha" );
                    for ( int ib = 0; ib < nb; ++ib ) std::printf( " %12.3g", double( bs[ ib ] ) );
                    std::printf( "\n" );
                    for ( int ia = 0; ia < o.g2_na; ++ia ) {
                        const TF al = t / TF( SI( 1 ) << ia );
                        if ( al < o.t_min ) break;
                        std::printf( "        %-10.4g", double( al ) );
                        for ( int ib = 0; ib < nb; ++ib ) {
                            const TF v = mat[ size_t( ia ) * nb + ib ];
                            if ( v == INFINI ) std::printf( " %12s", "refuse" );
                            else std::printf( " %12.6g", double( v ) );
                        }
                        std::printf( "\n" );
                    }
                    std::printf( "        argmin : alpha %.4g, beta %.4g, merite %.6e%s\n",
                                 double( al_b ), double( be_b ), double( best ),
                                 be_b == 0 ? "   ( beta = 0 : RIEN A GAGNER )" : "   ( HORS de la colonne beta = 0 )" );
                    std::fflush( stdout );
                }
                st.nb_recul += essais - 1;
                if ( best < nr ) {
                    w2.swap( wb ); a2.swap( ab ); fa2.swap( fab );
                    t = al_b;
                    pris = true;
                }
            }

            // ---- LE PAS SANS AUCUN DIAGRAMME ( `--pas grille2 --g2-modele`, 2D )
            //
            // Tout le choix du pas se fait a connectivite FIXE, donc sans reconstruire une seule
            // cellule. Il ne reste qu'UN diagramme par iteration, celui du point ou l'on va -- qui est
            // le diagramme de l'iteration suivante, pas une verification.
            //
            //   1. le modele a UNE direction, bati la ou `pd` est deja ( aux poids `w` ) : `alpha*`
            //      sort des RACINES ( § 7 ), et le modele donne les AIRES au point de sonde ;
            //   2. le residu en ce point se calcule depuis ces aires, et son systeme se resout avec
            //      LE MEME laplacien -- donc sans assemblage et SANS nouvelle hierarchie AMG, la
            //      seule depense etant les iterations de Krylov ;
            //   3. `e = d( sonde ) - d` est la courbure ( § 24.13 ), et le modele a DEUX directions
            //      rend l'exploration de `( alpha, beta )` gratuite ;
            //   4. on va a l'argmin. Si le vrai merite ne descend pas la-bas, on divise le pas par
            //      deux et on recommence -- un backtracking qui n'arrive pas en pratique.
            if ( o.pas == NewtonOptions::GRILLE2 && o.g2_modele && res_cur != NewtonOptions::LIN ) {
                if constexpr ( PD::dim == 2 ) {
                    const int nth = std::max( 1, par.threads );
                    int essais = 0;

                    // ---- 1. LE POINT DE SONDE AU PREMIER ORDRE : GRATUIT, sans modele du tout.
                    //
                    // Le systeme resolu est `L d = b` ( cf. `membre` ), et `L` est exactement la
                    // derivee des aires : `da = L d`. Donc `A_i( alpha ) = a_i + alpha b_i` au premier
                    // ordre, avec le second membre DEJA calcule -- pas un produit matrice-vecteur a
                    // faire. Et le `alpha*` du premier ordre se lit de la meme facon : le plus grand
                    // `alpha` tel que `a_i + alpha b_i >= eps` pour tout `i`.
                    //
                    // Ca supprime la construction a UNE direction, qui ne servait qu'a ces deux
                    // choses et coutait plus qu'un diagramme ( § 24.16 ).
                    TF a1 = INFINI;
                    for ( SI i = 0; i < n; ++i ) {
                        if ( ! protegee[ i ] || ! ( b[ i ] < 0 ) ) continue;
                        const TF r = ( eps - a[ i ] ) / b[ i ];
                        if ( r > 0 ) a1 = std::min( a1, r );
                    }
                    const TF ts = a1 < INFINI && a1 > 0 ? std::min( o.t0, o.facteur * a1 ) : t;

                    // ---- 2. LA DIRECTION SONDEE, SANS DIAGRAMME NI HIERARCHIE
                    bool ok_s = false;
                    {
                        const TF tk[ 1 ] = { ts };
                        aso.assign( n, 0 );
                        if ( o.g2_sonde_reelle ) {        // le temoin : les aires EXACTES du point
                            for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + ts * d[ i ];
                            w2[ 0 ] = 0;
                            mesures_et_facettes( w2, a2, fa2, pda2 );
                            ++essais;
                            aso = a2;
                            for ( SI i = 0; i < n; ++i ) aso[ i ] = std::max( aso[ i ], eps );
                        } else
                            for ( SI i = 0; i < n; ++i )
                                aso[ i ] = std::max( a[ i ] + ts * b[ i ], eps );
                        (void) tk;
                        membre_de( aso, res_cur, o.puis, bson );
                        const double ts0 = now();
                        // LA SONDE NE SERT QU'A DEFINIR UNE DIRECTION : une tolerance lache suffit,
                        // et c'etait la moitie du surcout du mode ( 1164 iterations de Krylov contre
                        // 681 pour la reference, dont la moitie venait d'ici ).
                        const TF tol0 = lin.tolerance();
                        if ( o.g2_tol > 0 && tol0 > 0 ) lin.tolerance( o.g2_tol );
                        if ( lin.sait_encore() ) { lin.resout_encore( bson, d_next ); ok_s = true; }
                        else                       ok_s = lin.resout( L, bson, d_next );
                        if ( o.g2_tol > 0 && tol0 > 0 ) lin.tolerance( tol0 );
                        st.t_lin += now() - ts0;
                    }

                    // ---- 3. `e = d( sonde ) - d`, ET LE MODELE A DEUX DIRECTIONS
                    TF cos_de = 0, rap_de = 0;
                    if ( ok_s ) {
                        d_pre = d_next;
                        for ( SI i = 0; i < n; ++i ) d_pre[ i ] -= d[ i ];
                        d_pre[ 0 ] = 0;
                        TF ps = 0, nd = 0, np = 0;
                        for ( SI i = 0; i < n; ++i ) { ps += d[ i ] * d_pre[ i ]; nd += d[ i ] * d[ i ]; np += d_pre[ i ] * d_pre[ i ]; }
                        cos_de = nd > 0 && np > 0 ? ps / std::sqrt( nd * np ) : TF( 0 );
                        rap_de = nd > 0 ? std::sqrt( np / nd ) : TF( 0 );
                        const TF *dp2[ 2 ] = { d.data(), d_pre.data() };
                        polynomes_multi( pd, P, w, dp2, 2, par, pm2 );
                    } else {
                        const TF *dp1[ 1 ] = { d.data() };
                        polynomes_multi( pd, P, w, dp1, 1, par, pm2 );
                    }
                    const int nk = ok_s ? 2 : 1;

                    // ---- `alpha*` EXACT, tire du meme modele restreint a `beta = 0`. Le premier
                    // ordre a servi a placer la sonde ; l'echelle, elle, part du vrai `alpha*`.
                    TF am = INFINI;
                    for ( SI i = 0; i < n; ++i ) {
                        if ( ! protegee[ i ] || pm2[ i ].etat != PolyCellule::OK ) continue;
                        PolyCellule sc;
                        sc.a0 = pm2[ i ].c0; sc.a1 = pm2[ i ].g[ 0 ]; sc.a2 = pm2[ i ].q[ 0 ];
                        am = std::min( am, sc.premiere_racine( eps ) );
                    }
                    const TF th = am < INFINI && am > 0 ? std::min( o.t0, o.facteur * am ) : ts;
                    if ( am < INFINI && am > 0 ) alpha_lim = am;

                    // ---- LE MERITE `log2` DU MODELE, POUR UNE LISTE DE POINTS, EN UN PASSAGE
                    auto mods = [ & ]( const std::vector<TF> &pts, std::vector<TF> &s2,
                                       std::vector<TF> &mn ) {
                        const int np = int( pts.size() / nk );
                        std::vector<TF> as2( size_t( nth ) * np, 0 ), amn( size_t( nth ) * np, INFINI );
                        parallel_for( n, par, [ & ]( SI i, int th ) {
                            const PolyMulti &q = pm2[ i ];
                            if ( q.etat != PolyCellule::OK ) return;
                            TF *ps = &as2[ size_t( th ) * np ], *pmn = &amn[ size_t( th ) * np ];
                            for ( int p = 0; p < np; ++p ) {
                                const TF A = q( &pts[ size_t( p ) * nk ], nk );
                                if ( protegee[ i ] && A < pmn[ p ] ) pmn[ p ] = A;
                                if ( A > 0 ) { const TF gg = std::log( A / nu[ i ] ); ps[ p ] += gg * gg; }
                                else ps[ p ] = INFINI;
                            }
                        } );
                        s2.assign( np, 0 ); mn.assign( np, INFINI );
                        for ( int th = 0; th < nth; ++th )
                            for ( int p = 0; p < np; ++p ) {
                                const TF v = as2[ size_t( th ) * np + p ];
                                s2[ p ] = s2[ p ] == INFINI || v == INFINI ? INFINI : s2[ p ] + v;
                                mn[ p ] = std::min( mn[ p ], amn[ size_t( th ) * np + p ] );
                            }
                        for ( int p = 0; p < np; ++p ) s2[ p ] = s2[ p ] == INFINI ? INFINI : std::sqrt( s2[ p ] );
                    };

                    // ---- 4. L'EXPLORATION, GRATUITE : la grille puis la descente de gradient
                    TF mb = INFINI, t1b = th, t2b = 0;
                    {
                        std::vector<TF> pts;
                        const int NA = std::max( o.g2_na, 1 ), NB = ok_s ? std::max( o.g2_nb, 1 ) : 1;
                        for ( int ia = 0; ia < NA; ++ia ) {
                            const TF al = th / TF( SI( 1 ) << ia );
                            if ( al < o.t_min ) break;
                            for ( int ib = 0; ib < NB; ++ib ) {
                                const TF be = NB == 1 ? TF( 0 )
                                            : o.g2_bpos ? o.g2_bmax * ib / TF( std::max( NB - 1, 1 ) )
                                                        : o.g2_bmax * ( 2 * ib - ( NB - 1 ) ) / TF( std::max( NB - 1, 1 ) );
                                pts.push_back( al );
                                if ( nk == 2 ) pts.push_back( al * be );
                            }
                        }
                        std::vector<TF> s2, mn;
                        mods( pts, s2, mn );
                        for ( int p = 0; p < int( s2.size() ); ++p )
                            if ( s2[ p ] < mb && ( ! plancher_actif() || mn[ p ] >= eps ) ) {
                                mb = s2[ p ]; t1b = pts[ size_t( p ) * nk ];
                                t2b = nk == 2 ? pts[ size_t( p ) * nk + 1 ] : TF( 0 );
                            }
                    }
                    int nd_ok = 0;
                    if ( o.g2_desc > 0 && mb < INFINI && nk == 2 ) {
                        TF t1 = t1b, t2 = t2b, h = std::fabs( t1b ) / 4;
                        for ( int k = 0; k < o.g2_desc && h > TF( 1e-12 ); ++k ) {
                            const TF tk[ 2 ] = { t1, t2 };
                            std::vector<TF> agr( size_t( nth ) * 2, 0 );
                            parallel_for( n, par, [ & ]( SI i, int th ) {
                                const PolyMulti &q = pm2[ i ];  // d( sum g^2 )/dt = 2 sum g / A dA/dt
                                if ( q.etat != PolyCellule::OK ) return;
                                const TF A = q( tk, 2 );
                                if ( ! ( A > 0 ) ) return;
                                TF da[ 2 ];
                                q.gradient( tk, 2, da );
                                const TF c = 2 * std::log( A / nu[ i ] ) / A;
                                agr[ size_t( th ) * 2 ] += c * da[ 0 ];
                                agr[ size_t( th ) * 2 + 1 ] += c * da[ 1 ];
                            } );
                            TF gr[ 2 ] = { 0, 0 };
                            for ( int th = 0; th < nth; ++th ) { gr[ 0 ] += agr[ size_t( th ) * 2 ]; gr[ 1 ] += agr[ size_t( th ) * 2 + 1 ]; }
                            const TF ng = std::max( std::fabs( gr[ 0 ] ), std::fabs( gr[ 1 ] ) );
                            if ( ! ( ng > 0 ) ) break;
                            const std::vector<TF> pts = { t1 - h * gr[ 0 ] / ng, t2 - h * gr[ 1 ] / ng };
                            std::vector<TF> s2, mn;
                            mods( pts, s2, mn );
                            if ( s2[ 0 ] < mb && ( ! plancher_actif() || mn[ 0 ] >= eps ) ) {
                                mb = s2[ 0 ]; t1 = pts[ 0 ]; t2 = pts[ 1 ]; t1b = pts[ 0 ]; t2b = pts[ 1 ]; ++nd_ok;
                            } else
                                h /= 2;
                        }
                    }

                    // ---- 5. ON Y VA. Le diagramme est celui de l'iteration suivante ; le
                    // backtracking ne sert que si le vrai merite ne descend pas.
                    for ( int k = 0; k < std::max( o.g2_back, 1 ); ++k ) {
                        const TF f = TF( 1 ) / TF( SI( 1 ) << k );
                        for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + f * ( t1b * d[ i ] + t2b * d_pre[ i ] );
                        w2[ 0 ] = 0;
                        mesures_et_facettes( w2, a2, fa2, pda2 );
                        ++essais;
                        TF m2 = INFINI;
                        for ( SI i = 0; i < n; ++i )
                            if ( protegee[ i ] && a2[ i ] < m2 ) m2 = a2[ i ];
                        const TF v = merite( a2 );
                        const bool ok = ( ! plancher_actif() || m2 >= eps ) && v < nr;
                        if ( o.trace && o.g2_trace )
                            std::printf( "      MODELE : alpha* %.3e ( 1er ordre %.3e ), sonde en %.3e, cos( d, e ) %.4f,"
                                         " |e|/|d| %.3f | argmin ( %.4g, %.4g ) apres %d pas de descente,"
                                         " merite modele %.6e | facteur %.3g : merite REEL %.6e ( %s )\n",
                                         double( am ), double( a1 ), double( ts ), double( cos_de ), double( rap_de ),
                                         double( f * t1b ), double( f * t2b ), nd_ok, double( mb ),
                                         double( f ), double( v ), ok ? "PRIS" : "recule" );
                        if ( ok ) { t = f * t1b; pris = true; break; }
                        ++st.nb_back;
                    }
                    st.nb_recul += essais - 1;
                    if ( o.trace && o.g2_trace ) std::fflush( stdout );
                }
            }

            // ---- LE PAS QUI MINIMISE LE MERITE ( `--pas merite` )
            //
            // Le profil du § 21.2 montre que le merite `log` a un MINIMUM INTERIEUR le long de la
            // direction, et qu'a l'iteration 0 le plancher d'aire refuse le pas ou il se trouve. Ici on
            // descend l'echelle jusqu'a ce que le merite remonte, et on prend l'argmin -- ce qui coute
            // en general LE MEME nombre de diagrammes, puisque KMT descendait de toute facon plus bas.
            // LA MINIMISATION N'A DE SENS QUE PENDANT LA PHASE `log`. Apres la bascule le merite est
            // `lin`, et le § 21.1 a etabli que celui-la est un MAUVAIS JUGE : le minimiser -- a plus
            // forte raison finement -- revient a sur-ajuster un substitut, et le vrai critere remonte
            // ( mesure : `max|a-nu|/nu` de 0.976 a 2.47 en quatre iterations ). Donc dans la phase
            // `lin` on rend la main a KMT, qui protege l'aire et se contente du premier pas acceptable.
            if ( o.pas == NewtonOptions::MERITE && res_cur != NewtonOptions::LIN ) {
                TF best = INFINI, tb = 0;
                int monte = 0, essais = 0;
                for ( TF tp = t; tp > o.t_min; tp /= 2 ) {
                    ++essais;
                    for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + tp * d[ i ];
                    w2[ 0 ] = 0;
                    mesures_et_facettes( w2, a2, fa2, pda2 );
                    TF m2 = INFINI;
                    for ( SI i = 0; i < n; ++i )
                        if ( protegee[ i ] && a2[ i ] < m2 ) m2 = a2[ i ];
                    const TF v = merite( a2 );
                    const bool pl = plancher_actif();
                    // KMT D'ABORD : si le PREMIER barreau passe deja la condition de KMT, on le prend et
                    // on ne cherche pas plus loin. Sans ca la minimisation paye un diagramme de plus par
                    // iteration sur les cas ou `t = 1` etait deja bon -- mesure : 13 diagrammes au lieu de
                    // 7 sur l'uniforme 2D, pour le meme resultat.
                    if ( tp == t && ( ! pl || m2 >= eps )
                         && v <= ( 1 - gain * tp / 2 ) * nr && v < nr ) {
                        best = v; tb = tp;
                        wb = w2; ab = a2; fab = fa2;
                        break;
                    }
                    if ( ( ! pl || m2 >= eps ) && v < best ) {
                        best = v; tb = tp; monte = 0;
                        wb = w2; ab = a2; fab = fa2;         // le meilleur, a garder
                    } else if ( best < INFINI && ++monte >= o.mer_patience )
                        break;
                }
                // ---- LE RAFFINEMENT : section doree autour du meilleur barreau
                if ( o.mer_raffine > 0 && tb > 0 ) {
                    const TF phi = TF( 0.6180339887498949 );
                    TF lo = tb / 2, hi = std::min( o.t0, 2 * tb );
                    auto evalue = [ & ]( TF tp ) {          // rend le merite, et garde le meilleur
                        for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + tp * d[ i ];
                        w2[ 0 ] = 0;
                        mesures_et_facettes( w2, a2, fa2, pda2 );
                        ++essais;
                        TF m2 = INFINI;
                        for ( SI i = 0; i < n; ++i )
                            if ( protegee[ i ] && a2[ i ] < m2 ) m2 = a2[ i ];
                        const TF v = merite( a2 );
                        if ( ( ! plancher_actif() || m2 >= eps ) && v < best ) {
                            best = v; tb = tp;
                            wb = w2; ab = a2; fab = fa2;
                        }
                        return ( ! plancher_actif() || m2 >= eps ) ? v : INFINI;
                    };
                    TF x1 = hi - phi * ( hi - lo ), x2 = lo + phi * ( hi - lo );
                    TF f1 = evalue( x1 ), f2 = evalue( x2 );
                    for ( int k = 2; k < o.mer_raffine; ++k ) {
                        if ( f1 < f2 ) { hi = x2; x2 = x1; f2 = f1; x1 = hi - phi * ( hi - lo ); f1 = evalue( x1 ); }
                        else           { lo = x1; x1 = x2; f1 = f2; x2 = lo + phi * ( hi - lo ); f2 = evalue( x2 ); }
                    }
                }
                st.nb_recul += essais - 1;                  // tout barreau essaye sauf un est un recul
                if ( tb > 0 && best < nr ) {
                    w2.swap( wb ); a2.swap( ab ); fa2.swap( fab );
                    t = tb;
                    pris = true;
                }
            }

            for ( int passe = 0; passe < 2 && ! pris; ++passe ) {
                if ( passe == 1 ) {                      // la deformation n'a rien rendu
                    if ( d_sur.empty() || t_sur <= 0 ) break;
                    d.swap( d_sur );
                    d_sur.clear();
                    t = t_sur;
                    t_lim0 = t;
                    gain = 1;
                    deja = false;
                    ++st.nb_cible_refus;
                }
                for ( int essai = 0; essai < o.max_reculs; ++essai ) {
                    if ( ! ( essai == 0 && deja ) ) {    // sinon, deja fait en `t`
                        for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
                        w2[ 0 ] = 0;                     // la jauge, imposee et non esperee
                        mesures_et_facettes( w2, a2, fa2, pda2 );
                    }
                    TF m2 = INFINI;                      // le plancher `eps` est une aire ABSOLUE, et il ne
                    SI i_m2 = -1;                        // porte que sur les cellules VIVANTES AU DEPART
                    for ( SI i = 0; i < n; ++i )
                        if ( protegee[ i ] && a2[ i ] < m2 ) { m2 = a2[ i ]; i_m2 = i; }
                    const TF n2r = merite( a2 );
                    // POURQUOI CE PAS EST REFUSE. Les deux clauses ne disent pas la meme chose et on
                    // ne savait pas laquelle mordait : `--refus` les separe, a l'iteration `it` seule.
                    if ( o.refus == it ) {
                        const bool c_aire = m2 >= eps;
                        const bool c_mer  = n2r <= ( 1 - gain * t / 2 ) * nr && n2r < nr;
                        std::printf( "      refus it %d  t %.3e : AIRE %s ( m2 %.3e %s eps %.3e", it, double( t ),
                                     c_aire ? "ok " : "NON", double( m2 ), c_aire ? ">=" : "<", double( eps ) );
                        if ( i_m2 >= 0 )
                            std::printf( ", cellule %d : a0 %.3e -> a2 %.3e, nu %.3e, a2/nu %.2e",
                                         int( i_m2 ), double( a[ i_m2 ] ), double( a2[ i_m2 ] ),
                                         double( nu[ i_m2 ] ), double( a2[ i_m2 ] / nu[ i_m2 ] ) );
                        std::printf( " )  MERITE %s ( %.6e -> %.6e, exige <= %.6e )\n",
                                     c_mer ? "ok " : "NON", double( nr ), double( n2r ),
                                     double( std::min( ( 1 - gain * t / 2 ) * nr, nr ) ) );
                        std::fflush( stdout );
                    }
                    if ( ( ! plancher_actif() || m2 >= eps ) && n2r <= ( 1 - gain * t / 2 ) * nr && n2r < nr ) { pris = true; break; }
                    t /= 2;
                    ++st.nb_recul;
                    if ( t < o.t_min )
                        break;
                }
            }
            if ( pris && alpha_lim >= 0 && t < t_lim0 ) ++st.nb_lim_refus;
            if ( o.trace ) {
                std::printf( "    it %2d  |r|_2 %.3e  max|a-nu|/nu %.3e  %d vides  pas %.2e  %d diag"
                             "  [maj %.2f  diag %.2f  asm %.2f  lin %.2f%s]",
                             it, double( nr ), double( pire ), int( nvide ), double( t ),
                             st.nb_diag - g0, st.t_maj - m0, st.t_diag - d0, st.t_asm - s0,
                             lin.st.total() - l0, it_txt( lin.st.nb_iter - i0 ) );
                if ( alpha_lim >= 0 )
                    std::printf( "  alpha* %.2e%s", double( alpha_lim ), t < t_lim0 ? " REFUSE" : "" );
                std::printf( "\n" );
                std::fflush( stdout );
            }
            if ( ! pris ) {
                st.fin = "STAGNATION";                   // le plancher numerique, pas un echec
                return false;
            }
            t_prec = t;
            if ( o.pas == NewtonOptions::GRILLE2 && o.g2_dir == NewtonOptions::PREC ) {
                d_pre.resize( n );                      // le DEPLACEMENT reellement fait, melange compris
                for ( SI i = 0; i < n; ++i ) d_pre[ i ] = w2[ i ] - w[ i ];
            }
            w.swap( w2 );
            a.swap( a2 );
            fa.swap( fa2 );
            if ( pda ) da.swap( da2 );
            if ( o.apres_pas ) o.apres_pas( it, t, st.nb_recul );
        }
        st.fin = "MAX ITERATIONS";
        return false;
    }

private:
    static const char *it_txt( int nb ) {
        static thread_local char buf[ 32 ];
        if ( ! nb ) return "";
        std::snprintf( buf, sizeof( buf ), " (%d it)", nb );
        return buf;
    }
};

} // namespace sf
