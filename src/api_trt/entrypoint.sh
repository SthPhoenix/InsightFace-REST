#!/bin/bash
set -e
echo Preparing models...
python prepare_models.py

echo Starting InsightFace-REST using $NUM_WORKERS workers.

exec gunicorn --log-level $LOG_LEVEL\
     -w $NUM_WORKERS\
     -k uvicorn.workers.UvicornWorker\
     --keep-alive 60\
     --timeout 60\
     app:app -b 0.0.0.0:18080