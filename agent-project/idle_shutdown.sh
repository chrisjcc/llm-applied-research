#!/bin/bash
# Auto-shutdown if GPU idle for X minutes

IDLE_LIMIT=30   # minutes of idle allowed
CHECK_INTERVAL=60  # seconds between checks
IDLE_COUNT=0

while true; do
    # Query GPU utilization (%)
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')

    if [ "$UTIL" -eq 0 ]; then
        IDLE_COUNT=$((IDLE_COUNT+1))
        echo "$(date): GPU idle ($IDLE_COUNT/$IDLE_LIMIT)"
    else
        IDLE_COUNT=0
        echo "$(date): GPU active, reset counter"
    fi

    # If idle limit reached → shutdown
    if [ "$IDLE_COUNT" -ge "$IDLE_LIMIT" ]; then
        echo "$(date): GPU idle for $IDLE_LIMIT minutes, shutting down..."
        sudo shutdown now
    fi

    sleep $CHECK_INTERVAL
done
