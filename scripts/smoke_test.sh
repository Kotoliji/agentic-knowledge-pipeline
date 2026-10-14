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

docker compose stop retrieval
expect retrieval_failed "$(ask 'How many days per week can I work remotely?')"
docker compose start retrieval
echo "smoke test passed"
