#!/bin/bash
# Starts all port-forwards for the ecommerce-agent kind deployment.
# Run once after system boot. Re-run if any connection drops.

NAMESPACE=ecommerce

echo "Starting port-forwards for ecommerce-agent..."

# Kill any existing port-forwards for these ports
for port in 8000 3001 9090; do
  lsof -ti tcp:$port | xargs -r kill -9 2>/dev/null
done

# FastAPI app
kubectl port-forward svc/ecommerce-fastapi 8000:80 -n $NAMESPACE \
  > /tmp/pf-fastapi.log 2>&1 &
echo "FastAPI   → http://localhost:8000  (docs: http://localhost:8000/docs)"

# Grafana
kubectl port-forward svc/ecommerce-grafana 3001:3001 -n $NAMESPACE \
  > /tmp/pf-grafana.log 2>&1 &
echo "Grafana   → http://localhost:3001  (login: admin / admin)"

# Prometheus (optional)
kubectl port-forward svc/ecommerce-prometheus 9090:9090 -n $NAMESPACE \
  > /tmp/pf-prometheus.log 2>&1 &
echo "Prometheus→ http://localhost:9090"

echo ""
echo "All services running. Press Ctrl+C to stop."
wait
