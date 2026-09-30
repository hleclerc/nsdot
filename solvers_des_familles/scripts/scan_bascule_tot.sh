#!/bin/zsh
# =====================================================================================
# REMONTER LA BASCULE PLUS TOT ? Le § 24.10 dit que la contrainte cesse d'etre active quand
# `max|a-nu|/nu` tombe a O(1), et que `R = 2` declenche LA. Basculer plus tot ( `R` plus grand )
# c'est donc confier a `lin` des iterations ou la contrainte est encore active. Ce balayage le
# mesure sur les deux amortissements et sur les six cas du banc.
#
#   job -- scripts/scan_bascule_tot.sh
# =====================================================================================
set -u
cd ${0:a:h}/..
C=../2d_des_familles/cases
T=${T:-8}
Rs=( 0 2 10 50 200 1000 1e9 )

un() {  # $1 = args du cas, $2 = args de la variante
  l=$( xmake run newton ${=1} --quiet --threads $T --newton-max 60 --residu log ${=2} 2>&1 | grep -E "^  newton " | head -1 )
  it=$(  print -r -- $l | sed -E 's/.*-- ([0-9]+) iterations.*/\1/' )
  dg=$(  print -r -- $l | sed -E 's/.*, ([0-9]+) diagrammes.*/\1/' )
  rc=$(  print -r -- $l | sed -E 's/.*\( ([0-9]+) reculs.*/\1/' )
  fin=$( print -r -- $l | sed -E 's/.*newton ([A-Z ]+) \( max.*/\1/' )
  res=$( print -r -- $l | sed -E 's/.*max\|a-nu\|\/nu = ([^ ]*) \).*/\1/' )
  printf "%4s it %5s diag %4s rec  %-12s %s\n" "$it" "$dg" "$rc" "$fin" "$res"
}

for cas in "2D uniforme:--2d --load uniforme -n 100000" \
           "2D lignes s0.005 propre:--2d --load $C/lines5_n100000_s0.005_voronoi.txt" \
           "2D lignes s0.02:--2d --load $C/lines5_n100000_s0.02_voronoi.txt" \
           "2D aires egales DEGENERE:--2d --load $C/lines5_n100000_s0.005_equal.txt" \
           "3D uniforme:--3d --load uniforme -n 100000" \
           "3D plans s0.02:--3d --load $C/planes4_n100000_s0.02_voronoi.txt"; do
  print -- "=== ${cas%%:*}"
  for pas in "dyadique:--pas essais" "limites 0.9:--pas essai-limites --facteur 0.9"; do
    for R in $Rs; do
      printf "  %-12s R=%-6s " "${pas%%:*}" $R
      un "${cas#*:}" "${pas#*:} --bascule-residu $R"
    done
  done
done
