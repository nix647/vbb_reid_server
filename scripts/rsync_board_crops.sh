#!/usr/bin/env bash
# Pull every sighting crop the test boards still hold (they keep 7 days) to the GPU host.
# READ-ONLY on the board: rsync pulls, never --delete, never writes there. Runs the
# board-side rsync at the lowest CPU and I/O priority so the live pipeline is not slowed.
#
# Usage:  rsync_board_crops.sh <dest_root> <did>=<user@host> [<did>=<user@host> ...]
# Example (cron on gpu-ubun, daily 03:10):
#   10 3 * * * /path/to/rsync_board_crops.sh /vmdata/vbb_eval/crops 45=linaro@192.168.0.X >> /vmdata/vbb_eval/rsync.log 2>&1
#
# Layout written: <dest_root>/<did>/<YYYY-MM-DD>/<sighting_id>/<idx>.jpg — what `--lan-crops` expects.
set -euo pipefail

BOARD_CROPS="${BOARD_CROPS:-/home/linaro/vbb/.devdata/vbb/person_sightings/crops/}"
BWLIMIT_KBPS="${BWLIMIT_KBPS:-20000}"

if [[ $# -lt 2 ]]; then
  sed -n '2,12p' "$0"
  exit 2
fi
dest_root="$1"; shift

for spec in "$@"; do
  did="${spec%%=*}"; host="${spec#*=}"
  if [[ -z "$did" || -z "$host" || "$did" == "$spec" ]]; then
    echo "bad spec '$spec' (want <did>=<user@host>)" >&2; exit 2
  fi
  mkdir -p "$dest_root/$did"
  echo "$(date -Is) pull did=$did from $host:$BOARD_CROPS"
  rsync -a --ignore-existing --partial --bwlimit="$BWLIMIT_KBPS" \
        --include='*/' --include='*.jpg' --exclude='*' \
        --rsync-path='nice -n 19 ionice -c3 rsync' \
        -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' \
        "$host:$BOARD_CROPS" "$dest_root/$did/"
  echo "$(date -Is) done did=$did: $(find "$dest_root/$did" -name '*.jpg' | wc -l) crops on disk"
done
