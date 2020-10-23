#!/bin/bash

echo Starting InsightFace-REST using $NUM_WORKERS workers.
uvicorn fapp:app --port 18080 --host 0.0.0.0 --workers $NUM_WORKERS