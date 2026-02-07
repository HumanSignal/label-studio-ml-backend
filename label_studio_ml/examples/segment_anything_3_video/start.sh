#!/bin/bash

echo "Running pre-flight checks..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python not found"
    exit 1
fi

# Check SAM3 model availability
python -c "from transformers import Sam3TrackerVideoModel; print('SAM3 Tracker available')" 2>/dev/null || \
    echo "Warning: Sam3TrackerVideoModel not available"
python -c "from transformers import Sam3VideoModel; print('SAM3 Video available')" 2>/dev/null || \
    echo "Warning: Sam3VideoModel not available"

# Check GPU availability if DEVICE=cuda
if [ "$DEVICE" = "cuda" ]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "nvidia-smi available"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "Warning: nvidia-smi not found but DEVICE=cuda"
    fi
fi

echo "Pre-flight checks complete"
echo "Starting gunicorn..."

# Execute the gunicorn command
# Note: Changed timeout from 0 (infinite) to 3600 (1 hour) for video processing
# This prevents infinite hangs while allowing long-running video operations
exec gunicorn --bind :${PORT:-9090} \
    --workers ${WORKERS:-1} \
    --threads ${THREADS:-4} \
    --timeout 3600 \
    --graceful-timeout 30 \
    --log-level ${LOG_LEVEL:-info} \
    _wsgi:app
