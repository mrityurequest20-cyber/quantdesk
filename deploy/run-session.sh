#!/usr/bin/env bash
# run-session.sh [quantdesk intraday live args...]
# One runner's share of the trading day: the engine in the background, and the read-only live
# site re-exported every $PUBLISH_EVERY_MIN minutes (pushed to the gh-pages branch only when
# something changed), plus once more when the engine stops. On cancel the engine is stopped
# (every minute is committed to the journal, so nothing is lost); the workflow's `if: cancelled()`
# step then squares off open paper positions (`live --close-out`) with its own time budget.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
every=${PUBLISH_EVERY_MIN:-6}
site=${SITE_DIR:-_site}
pages=${PAGES_BRANCH:-gh-pages}

publish() {
  [ "${PUBLISH:-1}" = "1" ] || return 0
  python -m quantdesk intraday export-site --dir "$site" --sessions 2 >/dev/null \
    && "$here/push-dir.sh" "$site" "$pages" "site $(TZ=Asia/Kolkata date '+%F %H:%M') IST" \
    || echo "publish failed; next try in ${every} min"
}

python -m quantdesk intraday live "$@" &
engine=$!
on_cancel() {
  echo "cancelled: stopping the engine"
  kill -TERM "$engine" 2>/dev/null
  wait "$engine" 2>/dev/null
  exit 130
}
trap on_cancel TERM INT

publish
while kill -0 "$engine" 2>/dev/null; do
  for _ in $(seq $((every * 6))); do
    kill -0 "$engine" 2>/dev/null || break
    sleep 10
  done
  kill -0 "$engine" 2>/dev/null && publish
done
wait "$engine"
rc=$?
publish
exit $rc
