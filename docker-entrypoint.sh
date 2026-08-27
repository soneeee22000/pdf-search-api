#!/bin/sh
# Two entrypoints from one image: `ingest` runs the CLI, `api` serves.
# Anything else is executed verbatim, so `docker run ... sh` still works.
set -e

case "$1" in
  ingest)
    shift
    exec python -m pdf_search.ingest "$@"
    ;;
  api)
    shift
    exec uvicorn pdf_search.api:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
