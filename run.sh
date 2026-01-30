#!/bin/bash

# Get real script directory, handling symlinks (macOS compatible)
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE" )"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
PID_FILE="$SCRIPT_DIR/.proxy.pid"

# Check if already running and restart if needed
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Claude Code Proxy is running (PID: $OLD_PID). Restarting..."
        kill "$OLD_PID"
        sleep 2
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "Force killing stubborn process (PID: $OLD_PID)..."
            kill -9 "$OLD_PID"
        fi
    else
        echo "Found stale PID file, cleaning up."
    fi
    rm -f "$PID_FILE"
fi

# Load environment variables if .env exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# Check for required environment variables
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Warning: GEMINI_API_KEY is not set"
    echo "Please set it in .env file or export it"
fi

# Create logs directory if it doesn't exist (same as logger.py)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Generate timestamp for log filename (same format as logger.py)
CURRENT_HOUR=$(date '+%Y-%m-%d_%H')
RUN_LOG="$LOG_DIR/run-${CURRENT_HOUR}.log"

# Start the proxy server with nohup to detach from terminal
echo "Starting Claude Code Proxy..."
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting Claude Code Proxy..." >> "$RUN_LOG"
cd "$SCRIPT_DIR"
nohup .venv/bin/python main.py >> "$RUN_LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "Claude Code Proxy started (PID: $(cat $PID_FILE))"
echo "$(date '+%Y-%m-%d %H:%M:%S') Claude Code Proxy started (PID: $(cat $PID_FILE))" >> "$RUN_LOG"
