#!/bin/bash
# start-demo.sh — starts and keeps alive all port-forwards for the ecommerce demo
# Run once: ./start-demo.sh
# Stops on Ctrl+C

set -e

echo "=== Starting demo port-forwards ==="
echo "  API/UI  → http://localhost:8080"
echo "  Grafana → http://localhost:3001"
echo "  Prometheus → http://localhost:9090"
echo ""
echo "  Frontend: http://localhost:8080/app"
echo "  API docs: http://localhost:8080/docs"
echo ""
echo "Press Ctrl+C to stop all."
echo ""

# Kill any existing port-forwards on these ports
for port in 8080 3001 9090; do
    fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 1

# Start all port-forwards in background
kubectl port-forward svc/ecommerce-fastapi 8080:80 -n ecommerce > /tmp/pf-fastapi.log 2>&1 &
PF_FASTAPI=$!

kubectl port-forward svc/ecommerce-grafana 3001:3001 -n ecommerce > /tmp/pf-grafana.log 2>&1 &
PF_GRAFANA=$!

kubectl port-forward svc/ecommerce-prometheus 9090:9090 -n ecommerce > /tmp/pf-prometheus.log 2>&1 &
PF_PROMETHEUS=$!

# Keep port-forwards alive — restart if they die
trap 'echo ""; echo "Stopping..."; kill $PF_FASTAPI $PF_GRAFANA $PF_PROMETHEUS 2>/dev/null; exit 0' INT TERM

while true; do
    if ! kill -0 $PF_FASTAPI 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] fastapi port-forward died — restarting..."
        kubectl port-forward svc/ecommerce-fastapi 8080:80 -n ecommerce > /tmp/pf-fastapi.log 2>&1 &
        PF_FASTAPI=$!
    fi
    if ! kill -0 $PF_GRAFANA 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] grafana port-forward died — restarting..."
        kubectl port-forward svc/ecommerce-grafana 3001:3001 -n ecommerce > /tmp/pf-grafana.log 2>&1 &
        PF_GRAFANA=$!
    fi
    if ! kill -0 $PF_PROMETHEUS 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] prometheus port-forward died — restarting..."
        kubectl port-forward svc/ecommerce-prometheus 9090:9090 -n ecommerce > /tmp/pf-prometheus.log 2>&1 &
        PF_PROMETHEUS=$!
    fi
    sleep 5
done
