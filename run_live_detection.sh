#!/usr/bin/env bash
set -u
cd -- "$(dirname -- "$0")"
while true; do
    env -u PYTHONPATH .venv/bin/python live_detection_demo.py "$@"
    status=$?
    if [ "$status" -eq 0 ] || [ "$status" -eq 130 ]; then
        exit "$status"
    fi
    printf 'Live detection exited with status %s; restarting in 2 seconds.\n' "$status" >&2
    sleep 2
done
