#!/usr/bin/env bash
# Load mounted CANN libraries, then replace the shell so signals reach the app.
set -e
source /app/include/environment.sh
cd /app
exec "$@"
