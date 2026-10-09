#!/usr/bin/env bash
# Load mounted CANN libraries, then replace the shell so signals reach the app.
set -e
pi05_cann_root="${PI05_CANN_ROOT:-/usr/local/Ascend/cann-8.5.0}"
source "$pi05_cann_root/set_env.sh"
# Upstream ACLLite modules use bare sibling imports as well as package imports.
export PYTHONPATH="/app/runtime/acllite:/app/runtime/acllite/acllite:$pi05_cann_root/python/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cd /app
exec "$@"
