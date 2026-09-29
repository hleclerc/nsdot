#!/usr/bin/env bash
# RECONSTRUIRE LE PLAN DE TRAVAIL.
#
# Les quatre paquets ont chacun leur dépôt ; ce dépôt-ci ne les suit plus, il les ACCUEILLE. Le
# couplage est physique -- `errandfile.py` déclare `src = [ "loom/src", "sdot/src", "otrec/src" ]`,
# les conteneurs montent `loom` sur `/opt/sdot/loom` -- donc les quatre doivent être clonés ICI, à
# ces noms-là. C'est ce que fait ce script, et c'est tout ce qu'il fait.
#
#     scripts/bootstrap.sh            clone ce qui manque
#     scripts/bootstrap.sh --pull     ... et met à jour ce qui est déjà là
#
# Ensuite : `errand --setup --env nsdot` fabrique l'environnement et installe les quatre en
# éditable ( voir `errandfile.py` ).
#
# NB le répertoire `sdot/` vient du dépôt `sdot-ffi` : le nom local est celui du paquet python,
# et c'est lui que `errandfile.py` et les conteneurs attendent.
set -euo pipefail

racine="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$racine"

depots=(
    "loom    git@github.com:hleclerc/loom.git"
    "sdot    git@github.com:hleclerc/sdot-ffi.git"
    "otrec   git@github.com:hleclerc/otrec.git"
    "errand  git@github.com:hleclerc/errand.git"
)

maj=0
[ "${1:-}" = "--pull" ] && maj=1

for ligne in "${depots[@]}"; do
    read -r nom url <<< "$ligne"
    if [ -d "$nom/.git" ]; then
        if [ "$maj" = 1 ]; then
            echo "== $nom : mise à jour"
            git -C "$nom" pull --ff-only
        else
            echo "== $nom : déjà là ( $(git -C "$nom" rev-parse --short HEAD) )"
        fi
    elif [ -e "$nom" ]; then
        # un répertoire sans `.git` : on ne l'écrase pas, on le dit.
        echo "!! $nom existe mais n'est pas un dépôt git -- laissé tel quel" >&2
    else
        echo "== $nom : clonage depuis $url"
        git clone "$url" "$nom"
    fi
done

echo
echo "Le plan de travail est en place. Ensuite :"
echo "    errand --setup --env nsdot     # l'environnement + les quatre paquets en éditable"
echo "    errand test_diffusion          # une suite, pour vérifier"
