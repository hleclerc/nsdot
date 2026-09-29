#!/bin/zsh
# =====================================================================================
# LE BALAYAGE DE L'EXPOSANT. `g_p( x ) = ( x^p - 1 ) / p` contient `lin` ( p = 1 ) et `log`
# ( p = 0 ) EXACTEMENT. Comparer trois residus ne dit pas pourquoi l'un gagne ; un exposant
# continu donne la PENTE, donc la raison -- et dit s'il y a un optimum ou si `log` est un bout.
#
#   scripts/scan_puissance.sh            tous les cas
#   scripts/scan_puissance.sh 2d-lignes-s0.02
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
P=( 1 0.9 0.75 0.5 0.25 0.1 0 -0.1 -0.25 -0.5 -1 -2 )

for e in $cas; do
  nom=${e%%:*}; args=${e#*:}
  [[ $FILTRE == tout || $nom == ${FILTRE}* ]] || continue
  print -- "=== $nom"
  printf "  %6s  %4s %5s %5s  %-22s %s\n" p it diag reculs fin reste
  for p in $P; do
    l=$( xmake run newton ${=args} --quiet --newton-max 60 --residu puissance --puis $p 2>&1 \
         | grep -E "^  newton " | head -1 )
    fin=$( print -r -- $l | sed -n 's/^  newton \(.*\) ( max.*/\1/p' )
    res=$( print -r -- $l | sed -n 's/.*max|a-nu|\/nu = \([^ ]*\) ).*/\1/p' )
    it=$(  print -r -- $l | sed -n 's/.*-- \([0-9]*\) iterations.*/\1/p' )
    dg=$(  print -r -- $l | sed -n 's/.*, \([0-9]*\) diagrammes.*/\1/p' )
    rc=$(  print -r -- $l | sed -n 's/.*diagrammes ( \([0-9]*\) reculs.*/\1/p' )
    printf "  %6s  %4s %5s %5s  %-22s %s\n" $p "$it" "$dg" "$rc" "$fin" "$res"
  done
done
