#!/bin/bash
set -e
echo Preparing models...
python -m if_rest.prepare_models

echo Starting InsightFace-REST using $NUM_WORKERS workers.

exec gunicorn --log-level $LOG_LEVEL\
     -w $NUM_WORKERS\
     -k uvicorn.workers.UvicornWorker\
     --keep-alive 60\
     --timeout 60\
     if_rest.api.main:app -b 0.0.0.0:18080