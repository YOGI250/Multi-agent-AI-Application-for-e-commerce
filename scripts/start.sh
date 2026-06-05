#!/bin/bash
# Starts all port-forwards for the ecommerce-agent kind deployment.
# Each forward runs in a loop so it auto-reconnects when a pod restarts.

NAMESPACE=ecommerce

pf() {
  local label=$1; local local_port=$2; local svc=$3; local svc_port=$4
  while true; do
    kubectl port-forward "svc/$svc" "${local_port}:${svc_port}" -n "$NAMESPACE" \
      > "/tmp/pf-${label}.log" 2>&1
    echo "[$(date +%T)] $label port-forward dropped — reconnecting in 3s..."
    sleep 3
  done
}

# Kill any stale port-forwards on these ports
for port in 8000 3001 9090; do
  lsof -ti tcp:$port | xargs -r kill -9 2>/dev/null
done

echo "Starting port-forwards (auto-reconnect on pod restarts)..."
pf fastapi    8000 ecommerce-fastapi   80   &
pf grafana    3001 ecommerce-grafana   3001 &
pf prometheus 9090 ecommerce-prometheus 9090 &

echo "FastAPI   → http://localhost:8000"
echo "Grafana   → http://localhost:3001  (admin / admin)"
echo "Prometheus→ http://localhost:9090"
echo ""
echo "All services running with auto-reconnect."
wait
