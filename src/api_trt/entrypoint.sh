#!/bin/bash

echo Preparing models...
python prepare_models.py

echo Starting InsightFace-REST using $NUM_WORKERS workers.
#uvicorn app:app --port 18080 --host 0.0.0.0 --workers $NUM_WORKERS  --timeout-keep-alive 20000
exec gunicorn --log-level $LOG_LEVEL -w $NUM_WORKERS -k uvicorn.workers.UvicornWorker --keep-alive 20 app:app -b 0.0.0.0:18080