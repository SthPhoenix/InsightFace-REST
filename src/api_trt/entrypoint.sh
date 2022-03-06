#!/bin/bash

echo Preparing models...
python3 prepare_models.py

echo Starting InsightFace-REST using $NUM_WORKERS workers.
uvicorn app:app --port 18080 --host 0.0.0.0 --workers $NUM_WORKERS  --timeout-keep-alive 20000
