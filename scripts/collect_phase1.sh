#!/usr/bin/env bash
# Kept for compatibility: phase-1 collection is now part of collect_results.sh.
exec bash "$(dirname "$0")/collect_results.sh" "$@"
