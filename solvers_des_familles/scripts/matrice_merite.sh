#!/bin/zsh
# =====================================================================================
# LA MATRICE DIRECTION x JUGE. `--residu log` changeait DEUX choses a la fois : le second membre
# de Newton ( le modele local ) et le merite de l'amortissement ( la norme qui accepte le pas ).
# On ne pouvait donc pas savoir laquelle des deux gagnait. `--merite` les separe, et cette matrice
# le lit sur les cas usuels 2D et 3D.
#
#   scripts/matrice_merite.sh              tout
#   scripts/matrice_merite.sh 2d           les cas 2D seulement
# =====================================================================================
set -u
cd ${0:a:h}/..
C=../2d_des_familles/cases
FILTRE=${1:-tout}

cas=(
  "2d-uniforme:--2d --load uniforme -n 100000"
  "2d-lignes-s0.1:--2d --load $C/lines5_n100000_s0.1_voronoi.txt"
  "2d-lignes-s0.02:--2d --load $C/lines5_n100000_s0.02_voronoi.txt"
  "2d-lignes-s0.005:--2d --load $C/lines5_n100000_s0.005_voronoi.txt"
  "3d-uniforme:--3d --load uniforme -n 100000"
  "3d-plans-s0.02:--3d --load $C/planes4_n100000_s0.02_voronoi.txt"
)

for e in $cas; do
  nom=${e%%:*}; args=${e#*:}
  [[ $FILTRE == tout || $nom == ${FILTRE}* ]] || continue
  print -- "=== $nom"
  printf "  %-9s %-9s  %4s %5s %5s  %-22s %s\n" direction juge it diag reculs fin reste
  for dir in lin barriere log; do
    for jug in lin barriere log; do
      l=$( xmake run newton ${=args} --quiet --newton-max 60 --residu $dir --merite $jug 2>&1 \
           | grep -E "^  newton " | head -1 )
      fin=$(  print -r -- $l | sed -n 's/^  newton \(.*\) ( max.*/\1/p' )
      res=$(  print -r -- $l | sed -n 's/.*max|a-nu|\/nu = \([^ ]*\) ).*/\1/p' )
      it=$(   print -r -- $l | sed -n 's/.*-- \([0-9]*\) iterations.*/\1/p' )
      dg=$(   print -r -- $l | sed -n 's/.*, \([0-9]*\) diagrammes.*/\1/p' )
      rc=$(   print -r -- $l | sed -n 's/.*diagrammes ( \([0-9]*\) reculs.*/\1/p' )
      printf "  %-9s %-9s  %4s %5s %5s  %-22s %s\n" $dir $jug "$it" "$dg" "$rc" "$fin" "$res"
    done
  done
done
