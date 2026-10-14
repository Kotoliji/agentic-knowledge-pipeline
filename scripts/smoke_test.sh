#!/usr/bin/env sh
# End-to-end check of the running compose stack (offline agents unless .env says otherwise):
# answer with a citation, a checkpointed run, decline an out-of-scope question, and degrade
# when retrieval is down.
set -eu
API=${API:-http://localhost:8000}

fail() {
  echo "$1" >&2
  # Shows up as a check annotation on GitHub, readable without downloading logs.
  if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::error::$1"; fi
  exit 1
}

ask() {
  curl -fsS -X POST "$API/ask" -H 'content-type: application/json' -d "{\"question\":\"$1\"}" \
    || fail "POST /ask failed for: $1"
}

expect() {
  case "$2" in
    *"\"status\":\"$1\""*) echo "ok: $1" ;;
    *) fail "expected status $1, got: $2" ;;
  esac
}

curl -fsS "$API/health" && echo
answer=$(ask 'How many days per week can I work remotely?')
expect answered "$answer"

# The run was checkpointed in Redis: resuming a finished run is refused with 409.
# printf, not echo: dash's echo would expand the \n escapes inside the JSON.
thread=$(printf '%s\n' "$answer" | sed -n 's/.*"thread_id":"\([A-Za-z0-9_-]*\)".*/\1/p')
[ -n "$thread" ] || fail "no thread_id in: $answer"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API/ask/$thread/resume" || true)
[ "$code" = 409 ] || fail "resume of a finished run gave HTTP $code, expected 409"
echo "ok: run $thread is checkpointed"

expect insufficient_evidence "$(ask "What was the company's revenue last year?")"

# Async ingestion: 202 at once, stored by the worker, a re-send is flagged as duplicate.
title="Smoke test $(date +%s)"
doc="{\"title\":\"$title\",\"text\":\"Visitors sign in at the front desk on floor 2.\"}"
post_doc() {
  curl -s -w ' %{http_code}' -X POST "$API/documents" -H 'content-type: application/json' -d "$doc"
}
case "$(post_doc)" in *'"duplicate":false'*' 202') echo "ok: queued" ;; *) fail "POST /documents was not queued" ;; esac
case "$(post_doc)" in *'"duplicate":true'*' 202') echo "ok: duplicate detected" ;; *) fail "re-send not flagged as duplicate" ;; esac
for _ in $(seq 1 30); do
  curl -fsS "$API/documents" | grep -q "\"$title\"" && break
  sleep 1
done
curl -fsS "$API/documents" | grep -q "\"$title\"" || fail "worker did not store the queued document"
echo "ok: stored by the ingest worker"

docker compose stop retrieval
expect retrieval_failed "$(ask 'How many days per week can I work remotely?')"
docker compose start retrieval
echo "smoke test passed"
