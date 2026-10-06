#!/usr/bin/env bash
set -euo pipefail

# Keep one pinned environment definition for all model families.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/rebuild_predicate_env.sh" "$@"
