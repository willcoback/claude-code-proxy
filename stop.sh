#!/bin/bash

# Claude Code Proxy Stop Script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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
