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

echo "Stopping Claude Code Proxy..."

if [ ! -f "$PID_FILE" ]; then
    echo "PID file not found. Claude Code Proxy may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Process $PID is not running. Cleaning up PID file."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Found process PID: $PID"
kill "$PID" 2>/dev/null

# Wait and verify
sleep 1
if ps -p "$PID" > /dev/null 2>&1; then
    echo "Process still running, force killing..."
    kill -9 "$PID" 2>/dev/null
fi

rm -f "$PID_FILE"
echo "Claude Code Proxy stopped"
