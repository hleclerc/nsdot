#!/bin/zsh
# =====================================================================================
# LE MODELE MULTI-DIRECTIONS CONTRE TOUTES LES REFERENCES, sur les cas 2D.
#
# Les lignes a comparer, dans l'ordre ou elles se sont ajoutees :
#   lin/essais            la reference du banc
#   p=0.25 /essais        le meilleur exposant seul ( § 21.4 )
#   lin  /limites 0.95    le pas par les limites, relaxation reglee a la main ( § 21.5 )
#   log  /limites 0.9     ... et le meilleur des deux ensemble
#   modele K=1            LE CONTROLE : le modele SANS span ( une seule direction )
#   modele K=2 / K=3      le span, direction choisie par le modele, pas par les limites exactes
#   modele K=3 racine     ... et le pas par la racine du modele lui-meme
#
#   scripts/bilan_modele.sh
# =====================================================================================
set -u
cd ${0:a:h}/..
C=../2d_des_familles/cases

cas=(
  "uniforme:--load uniforme -n 100000"
  "lignes-s0.1:--load $C/lines5_n100000_s0.1_voronoi.txt"
  "lignes-s0.02:--load $C/lines5_n100000_s0.02_voronoi.txt"
  "lignes-s0.005:--load $C/lines5_n100000_s0.005_voronoi.txt"
)
var=(
  "lin /essais:--residu lin"
  "p=0.25 /essais:--residu puissance --puis 0.25"
  "lin /limites 0.95:--pas essai-limites --facteur 0.95 --residu lin"
  "log /limites 0.9:--pas essai-limites --facteur 0.9 --residu log"
  "modele K=1:--pas modele --mod-k 1"
  "modele K=2:--pas modele --mod-k 2"
  "modele K=3:--pas modele --mod-k 3"
  "modele K=3 + limites:--pas modele --mod-k 3 --mod-limites"
)

for e in $cas; do
  nom=${e%%:*}; args=${e#*:}
  print -- "=== 2d-$nom"
  printf "  %-20s %4s %5s %5s  %-22s %s\n" variante it diag reculs fin reste
  for v in $var; do
    vn=${v%%:*}; va=${v#*:}
    l=$( xmake run newton --2d ${=args} --quiet --newton-max 60 ${=va} 2>&1 | grep -E "^  newton " | head -1 )
    fin=$( print -r -- $l | sed -n 's/^  newton \(.*\) ( max.*/\1/p' )
    res=$( print -r -- $l | sed -n 's/.*max|a-nu|\/nu = \([^ ]*\) ).*/\1/p' )
    it=$(  print -r -- $l | sed -n 's/.*-- \([0-9]*\) iterations.*/\1/p' )
    dg=$(  print -r -- $l | sed -n 's/.*, \([0-9]*\) diagrammes.*/\1/p' )
    rc=$(  print -r -- $l | sed -n 's/.*diagrammes ( \([0-9]*\) reculs.*/\1/p' )
    printf "  %-20s %4s %5s %5s  %-22s %s\n" "$vn" "$it" "$dg" "$rc" "$fin" "$res"
  done
done
