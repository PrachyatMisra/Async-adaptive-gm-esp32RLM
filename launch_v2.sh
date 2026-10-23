#!/bin/bash

trap 'echo -e "\nShutting down services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' SIGINT SIGTERM

echo "Starting RLM AAGM v2 Orchestration..."

echo "-> Starting FastAPI Backend on port 8000..."
make backend > /dev/null 2>&1 &
BACKEND_PID=$!

echo "-> Starting Vite Frontend on port 5173..."
make frontend > /dev/null 2>&1 &
FRONTEND_PID=$!

echo "====================================================="
echo " Services are live!"
echo " - Backend API & WS: http://127.0.0.1:8000"
echo " - React Dashboard:  http://127.0.0.1:5173"
echo "====================================================="
echo "Press Ctrl+C to stop all services."

wait
