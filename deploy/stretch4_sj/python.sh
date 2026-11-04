#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/activate.sh"
exec "$VIRTUAL_ENV/bin/python" "$@"
