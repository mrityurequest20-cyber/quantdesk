#!/usr/bin/env bash
# journal.sh restore | save MESSAGE
# The desk's memory between runs on throwaway machines (GitHub Actions): the journal database,
# the paper broker's cash and positions, session reviews and the recorded 1m bars live on the
# orphan `journal` branch. Option-chain snapshots are too big for git; the workflow keeps them
# as run artifacts instead.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
branch=${JOURNAL_BRANCH:-journal}
rt=${QD_RUNTIME:-runtime}
case "${1:-}" in
  restore)
    mkdir -p "$rt"
    if git fetch -q --depth 1 origin "$branch" 2>/dev/null; then
      git archive FETCH_HEAD | tar -x -C "$rt"
      echo "journal: restored $(du -sh "$rt/intraday" | cut -f1) from the $branch branch"
    else
      echo "journal: no $branch branch yet; starting a fresh paper account"
    fi
    ;;
  save)
    [ -d "$rt/intraday" ] || { echo "journal: nothing to save"; exit 0; }
    snap=$(mktemp -d)
    trap 'rm -rf "$snap"' EXIT
    mkdir -p "$snap/intraday"
    ( cd "$rt/intraday" && tar -c --exclude='./data/*/chains' --exclude='*.tmp' --exclude='*-journal' . ) | tar -x -C "$snap/intraday"
    cat > "$snap/README.md" <<'MD'
# QuantDesk journal (machine-written)

State the live paper desk carries from one trading day to the next: `intraday/journal.db` (every
thought, decision, fill, trade and review), `intraday/broker.json` (paper cash and positions),
`intraday/reviews/*.md` (one written review per session) and `intraday/data/<date>/*_1m.csv` (the
bars it saw). Rewritten by the workflow after every run. Don't edit by hand.

Read it locally: `git fetch origin journal && git archive FETCH_HEAD | tar -x -C runtime`, then
`python -m quantdesk intraday trades` / `stats` / `review`, or `python -m quantdesk serve`.
MD
    "$here/push-dir.sh" "$snap" "$branch" "${2:-journal $(date '+%F %H:%M')}"
    ;;
  *)
    echo "usage: journal.sh restore | save MESSAGE" >&2
    exit 2
    ;;
esac
