#!/usr/bin/env sh
# End-to-end check of the running compose stack (offline agents unless .env says otherwise):
# answer with a citation, decline an out-of-scope question, degrade when retrieval is down.
set -eu
API=${API:-http://localhost:8000}

ask() {
  curl -fsS -X POST "$API/ask" -H 'content-type: application/json' -d "{\"question\":\"$1\"}"
}

expect() {
  case "$2" in
    *"\"status\":\"$1\""*) echo "ok: $1" ;;
    *) echo "expected status $1, got: $2" >&2; exit 1 ;;
  esac
}

curl -fsS "$API/health" && echo
answer=$(ask 'How many days per week can I work remotely?')
expect answered "$answer"

# The run was checkpointed in Redis: resuming a finished run is refused with 409.
thread=$(echo "$answer" | sed 's/.*"thread_id":"\([^"]*\)".*/\1/')
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API/ask/$thread/resume")
[ "$code" = 409 ] && echo "ok: checkpoint found for $thread" || { echo "resume gave $code" >&2; exit 1; }
expect insufficient_evidence "$(ask "What was the company's revenue last year?")"

docker compose stop retrieval
expect retrieval_failed "$(ask 'How many days per week can I work remotely?')"
docker compose start retrieval
echo "smoke test passed"
