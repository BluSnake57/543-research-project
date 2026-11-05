#!/bin/bash
set -euo pipefail
for attempt in {1..100}; do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        exec "$@"
    fi
    sleep 0.1
done
echo 'X display did not become ready within 10 seconds.' >&2
exit 1
