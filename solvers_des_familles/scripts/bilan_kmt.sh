#!/bin/zsh
# =====================================================================================
# LE BILAN : l'amortissement de Kitagawa-Merigot-Thibert ( le defaut du banc ) contre tout ce que les
# § 21 a 23 ont ajoute. Iterations, diagrammes, temps de paroi, et `reste` -- parce qu'un compte de
# diagrammes ne veut rien dire si les deux variantes ne s'arretent pas au meme endroit.
#
# KMT = `--pas essais --residu lin`, l'echelle `t = 1, 1/2, 1/4 ...` avec le plancher d'aire et la
# decroissance du merite `l2`. C'est bien le defaut, donc la reference est le banc lui-meme.
#
#   errand -x -- scripts/bilan_kmt.sh
# =====================================================================================
set -u
cd ${0:a:h}/..
C=../2d_des_familles/cases
T=${T:-8}

lit() { sed -E 's/.*newton ([A-Z ]+) \( max[^=]*= ([^ ]*) \).*-- ([0-9]+) iterations, ([0-9]+) diagrammes \( ([0-9]+) reculs.*/\3|\4|\5|\1|\2/' }

un() {  # $1 = nom, $2 = args du cas, $3 = args de la variante
  printf "  %-26s " "$1"
  l=$( xmake run newton --2d ${=2} --quiet --threads $T --newton-max 60 ${=3} 2>&1 \
       | grep -E "^  newton |^         arbre" | tr '\n' ' ' )
  it=$(  print -r -- $l | sed -E 's/.*-- ([0-9]+) iterations.*/\1/' )
  dg=$(  print -r -- $l | sed -E 's/.*, ([0-9]+) diagrammes.*/\1/' )
  fin=$( print -r -- $l | sed -E 's/.*newton ([A-Z ]+) \( max.*/\1/' )
  res=$( print -r -- $l | sed -E 's/.*max\|a-nu\|\/nu = ([^ ]*) \).*/\1/' )
  tt=$(  print -r -- $l | sed -E 's/.*TOTAL ([0-9.]+) s.*/\1/' )
  printf "%4s it  %5s diag  %8s s   %-15s %s\n" "$it" "$dg" "$tt" "$fin" "$res"
}

v2=( "KMT ( le defaut ):--pas essais --residu lin"
     "KMT + p = 0.25:--residu puissance --puis 0.25 --bascule-residu 0"
     "KMT + log + bascule:--residu log"
     "limites + log + bascule:--pas essai-limites --facteur 0.9 --residu log"
     "MODELE ( span, K=2 ):--pas modele" )
v3=( "KMT ( le defaut ):--pas essais --residu lin"
     "KMT + p = 0.25:--residu puissance --puis 0.25 --bascule-residu 0"
     "KMT + log + bascule:--residu log" )

for cas in "2D uniforme:--load uniforme -n 100000" \
           "2D lignes / Voronoi ( propre ):--load $C/lines5_n100000_s0.005_voronoi.txt" \
           "2D lignes / aires egales ( DEGENERE ):--load $C/lines5_n100000_s0.005_equal.txt"; do
  print -- "=== ${cas%%:*}"
  for v in $v2; do un "${v%%:*}" "${cas#*:}" "${v#*:}"; done
done
for cas in "3D uniforme:--3d --load uniforme -n 100000" \
           "3D plans / Voronoi:--3d --load $C/planes4_n100000_s0.02_voronoi.txt" \
           "3D plans / volumes egaux:--3d --load $C/planes4_n100000_s0.02_equal.txt"; do
  print -- "=== ${cas%%:*}"
  for v in $v3; do un "${v%%:*}" "${cas#*:}" "${v#*:}"; done
done
