#!/bin/zsh
# =====================================================================================
# LA BASCULE DE RESIDU : `log` tant que la dynamique de `a / nu` est large, `lin` pour finir.
#
# Les deux bouts ne servent pas au meme moment ( § 21.3 ), et la bascule ne peut pas nuire tard
# parce que pres de la solution TOUS les residus donnent la meme direction. `R = 0` : jamais ;
# `R = 1e9` : des l'iteration 0, donc `lin` pur. Les deux bornes du balayage sont donc les deux
# temoins qu'il faut.
#
#   scripts/scan_bascule.sh
# =====================================================================================
set -u
cd ${0:a:h}/..
C=../2d_des_familles/cases
cas=( "uniforme:--load uniforme -n 100000" "s0.1:--load $C/lines5_n100000_s0.1_voronoi.txt" "s0.02:--load $C/lines5_n100000_s0.02_voronoi.txt" "s0.005:--load $C/lines5_n100000_s0.005_voronoi.txt" )
for e in $cas; do
  print -- "=== ${e%%:*}"
  for R in 0 0.5 2 10 50 200 1000 1e9; do
    printf "  bascule=%-6s " $R
    l=$( xmake run newton --2d ${=${e#*:}} --quiet --newton-max 60 --pas essai-limites --facteur 0.9 --residu log --bascule-residu $R 2>&1 | grep -E "^  newton " | head -1 )
    print -r -- $l | sed 's/.*newton \(.*\) ( max.*= \([^ ]*\) ).*-- \([0-9]*\) iterations, \([0-9]*\) diagrammes ( \([0-9]*\) reculs.*/\3 it, \4 diag, \5 reculs, \1, reste \2/'
  done
done
