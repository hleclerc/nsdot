#!/usr/bin/env python3
"""
DEDUPLIQUER LES GERMES TROP PROCHES d'un fichier de `cases/`.

Pourquoi ce script existe. `gen_cases.py` tire les germes autour de lignes puis les RABAT dans le
carre unite ( `clip` ). Quand sigma est petit, le rabattement empile plusieurs germes sur la meme
frontiere : sur `lines5_n100000_s0.005_voronoi.txt` il en reste des paires a **1.009e-08** l'une de
l'autre, pour un espacement median de 4.5e-4 -- quatre ordres de grandeur.

Ce que ca casse, et c'est deux choses distinctes :

  * LE PROBLEME LUI-MEME devient insoluble a la tolerance demandee. Deux germes confondus se
    partagent une cellule qu'aucun poids ne separe : `max|a-nu|/nu` plafonne a 2.35e-6 et Newton
    sort en STAGNATION -- ce plancher est celui du NUAGE, pas de la methode ( README § 3 ).
  * LE SYSTEME LINEAIRE devient mal conditionne. Le laplacien porte `c_ij = |facette| / 2|p_i-p_j|`,
    donc une distance de 1e-8 donne un poids de 1.4e4 contre une mediane de 0.29 : cinq decades
    d'etalement sur une poignee d'aretes. Mesure ( § 23 ) : sur ce nuage TOUS les solveurs iteratifs
    se degradent et le Cholesky gagne meme a huit fils, alors que sur les nuages sains c'est
    l'inverse. Un lissage de plus dans le multigrille maison fait passer la partie lineaire de 4.5 s
    a 50 s.

Donc la reparation n'est pas un preconditionneur, c'est le nuage.

CE QUI EST MALADE, RELEVE ( § 23.5 ) : `s0.005` seulement. Les autres nuages de `cases/` ont bien deux
a cinq paires sous le seuil, mais a ~0.3 % de l'espacement median -- donc `c_ij` reste dans la plage
saine, et les deduplquer ne change RIEN ( memes iterations, memes diagrammes ). Les nuages 3D sont
propres. Le mal commence vers 0.2 % de la mediane ; `s0.005` est a 0.002 %.

ET CE N'EST PAS LA BONNE OPERATION EN GENERAL ( § 23.6 ) : retirer un germe fait DISPARAITRE sa masse.
Benin quand `nu` est uniforme, faux des que `nu` est une donnee. Il faudra FUSIONNER -- une grappe
devient un germe de cible `sum nu_i`, on resout, on redivise la cellule entre ses membres ( ce qui est
arbitraire pour des germes exactement confondus, donc gratuit ). Ce script est le prealable, pas la
solution.

    ./nettoie_germes.py fichier.txt --essai                            # relever, sans ecrire
    ./nettoie_germes.py lines5_n100000_s0.005_voronoi.txt              # -> ..._propre.txt
    ./nettoie_germes.py entree.txt -o sortie.txt --seuil 1e-6
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree


def lit( chemin ):
    """Un fichier de `cases/` : des lignes `#`, puis `n`, puis `n` fois `x y [z] w`."""
    entete, corps = [], []
    with open( chemin ) as f:
        for l in f:
            if l.startswith( "#" ) and not corps:
                entete.append( l.rstrip( "\n" ) )
            elif l.strip():
                corps.append( l )
    n = int( corps[ 0 ].split()[ 0 ] )
    d = np.array( [ [ float( v ) for v in l.split() ] for l in corps[ 1 : n + 1 ] ] )
    return entete, d


def ecrit( chemin, entete, d ):
    with open( chemin, "w" ) as f:
        for l in entete:
            f.write( l + "\n" )
        f.write( "%d\n" % len( d ) )
        for r in d:
            f.write( " ".join( "%.17g" % v for v in r ) + "\n" )


def dedup( P, seuil ):
    """Les indices a GARDER : d'une grappe de germes a moins de `seuil`, on garde le premier.

    On ne deplace personne et on ne moyenne rien -- deplacer un germe changerait le probleme,
    alors qu'en retirer un qui est confondu avec son voisin ne change aucune cellule visible.
    """
    arbre = cKDTree( P )
    paires = arbre.query_pairs( seuil, output_type = "ndarray" )
    mort = np.zeros( len( P ), dtype = bool )
    for i, j in paires[ np.lexsort( ( paires[ :, 1 ], paires[ :, 0 ] ) ) ]:
        if not mort[ i ]:                                # `i` survit, donc `j` part
            mort[ j ] = True
    return ~mort, paires


def main():
    p = argparse.ArgumentParser( description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter )
    p.add_argument( "entree" )
    p.add_argument( "-o", "--sortie", default = None )
    p.add_argument( "--seuil", type = float, default = 0.0,
                    help = "distance sous laquelle deux germes sont confondus"
                           " ( 0 : 1 %% de l'espacement median )" )
    p.add_argument( "--essai", action = "store_true", help = "relever seulement, n'ecrire rien" )
    p.add_argument( "--garde-poids", action = "store_true",
                    help = "garder les poids du fichier au lieu de les annuler ( voir ci-dessous )" )
    a = p.parse_args()

    entete, d = lit( a.entree )
    D = d.shape[ 1 ] - 1
    P = d[ :, : D ]
    print( "  %s : %d germes en %dD" % ( a.entree, len( P ), D ) )

    arbre = cKDTree( P )
    dd, _ = arbre.query( P, k = 2 )
    esp = np.median( dd[ :, 1 ] )
    seuil = a.seuil if a.seuil > 0 else 0.01 * esp
    print( "  espacement au plus proche voisin : min %.3e, median %.3e ; seuil %.3e"
           % ( dd[ :, 1 ].min(), esp, seuil ) )

    garde, paires = dedup( P, seuil )
    nb = int( ( ~garde ).sum() )
    print( "  %d paires sous le seuil, %d germes retires" % ( len( paires ), nb ) )
    if nb == 0:
        print( "  rien a faire" )
        return 0
    if a.essai:
        return 0

    # ce que ca change pour le laplacien : `c_ij = |facette| / 2 |p_i - p_j|`, donc l'etalement des
    # poids est borne par celui des distances -- c'est LUI qu'on vient de couper
    d2, _ = cKDTree( P[ garde ] ).query( P[ garde ], k = 2 )
    print( "  apres : min %.3e ( x%.0f ), median %.3e" % ( d2[ :, 1 ].min(),
           d2[ :, 1 ].min() / max( dd[ :, 1 ].min(), 1e-300 ), np.median( d2[ :, 1 ] ) ) )

    # LES POIDS D'UN FICHIER `_equal` NE SURVIVENT PAS A LA DEDUPLICATION, et c'est important : ces
    # fichiers PORTENT la solution, obtenue par un tout autre chemin ( L-BFGS, pysdot ), et le banc s'en
    # sert de temoin independant. Retirer un germe CHANGE le probleme -- sa masse doit etre reprise par
    # les autres -- donc les poids stockes ne resolvent plus rien. On les annule, ce qui fait que le banc
    # les ignore de lui-meme ( `W` reste nul, donc pas de temoin ) plutot que de comparer a un faux.
    poids = d[ :, D ]
    if np.any( poids != 0 ) and not a.garde_poids:
        print( "  ATTENTION : ce fichier PORTE des poids non nuls ( un temoin ). Ils ne resolvent plus le"
               " probleme reduit : on les ANNULE. `--garde-poids` pour passer outre." )
        d = d.copy()
        d[ :, D ] = 0.0

    sortie = a.sortie
    if sortie is None:
        base, ext = os.path.splitext( a.entree )
        sortie = base + "_propre" + ext
    entete = entete + [ "# germes dedupliques par nettoie_germes.py : seuil %.3e, %d retires"
                        % ( seuil, nb ) ]
    if np.any( poids != 0 ) and not a.garde_poids:
        entete = entete + [ "# ATTENTION : les poids du fichier d'origine ont ete ANNULES -- ils"
                            " resolvaient l'ancien nuage, pas celui-ci. Ce fichier n'est plus un temoin." ]
    ecrit( sortie, entete, d[ garde ] )
    print( "  ecrit dans %s ( %d germes )" % ( sortie, int( garde.sum() ) ) )
    return 0


if __name__ == "__main__":
    sys.exit( main() )
