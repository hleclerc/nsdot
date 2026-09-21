#!/bin/bash
# UN BANC A LA FOIS sur la machine, sur les coeurs 8-15, 8 Go au plus : `scripts/banc.sh <cmd...>`.
# ( les compilations et le reste vont sur 0-7 ; le verrou est partage par toutes les sessions )
exec flock -w 7200 /tmp/banc.lock systemd-run --user --scope -q -p MemoryMax=8G taskset -c 8-15 "$@"
