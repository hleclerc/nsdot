#!/bin/zsh
# =====================================================================================
# LA RELAXATION, A LA MAIN, SOUS `essai-limites`. La passe des limites rend `alpha*` -- le pas
# exact qui met la premiere cellule au plancher -- et le pas retenu est `facteur * alpha*`. Ce
# `facteur` EST le coefficient de relaxation, et on ne l'avait jamais balaye contre l'exposant du
# residu : or les deux reglent la meme chose ( jusqu'ou pousser avant que le plancher morde ), donc
# ils peuvent tres bien se substituer l'un a l'autre -- ou pas, et c'est ce qu'on veut savoir.
#
#   scripts/scan_relax.sh                     sigma = 0.02
#   scripts/scan_relax.sh 0.005
# =====================================================================================
set -u
cd ${0:a:h}/..
C=../2d_des_familles/cases
S=${1:-0.02}

print -- "=== 2d-lignes-s$S, --pas essai-limites"
printf "  %6s %6s  %4s %5s %5s  %-22s %s\n" p facteur it diag reculs fin reste
for p in 1 0.5 0; do
  for f in 0.5 0.7 0.8 0.9 0.95 0.99; do
    l=$( xmake run newton --2d --load $C/lines5_n100000_s${S}_voronoi.txt --quiet --newton-max 60 \
         --pas essai-limites --facteur $f --residu puissance --puis $p 2>&1 | grep -E "^  newton " | head -1 )
    fin=$( print -r -- $l | sed -n 's/^  newton \(.*\) ( max.*/\1/p' )
    res=$( print -r -- $l | sed -n 's/.*max|a-nu|\/nu = \([^ ]*\) ).*/\1/p' )
    it=$(  print -r -- $l | sed -n 's/.*-- \([0-9]*\) iterations.*/\1/p' )
    dg=$(  print -r -- $l | sed -n 's/.*, \([0-9]*\) diagrammes.*/\1/p' )
    rc=$(  print -r -- $l | sed -n 's/.*diagrammes ( \([0-9]*\) reculs.*/\1/p' )
    printf "  %6s %6s  %4s %5s %5s  %-22s %s\n" $p $f "$it" "$dg" "$rc" "$fin" "$res"
  done
done
