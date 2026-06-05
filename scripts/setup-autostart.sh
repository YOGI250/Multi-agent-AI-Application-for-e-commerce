#!/bin/bash
# Installs a systemd user service that auto-starts all port-forwards
# after login. Run once: ./scripts/setup-autostart.sh

SERVICE_DIR="$HOME/.config/systemd/user"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$SERVICE_DIR"

cat > "$SERVICE_DIR/ecommerce-portforward.service" << EOF
[Unit]
Description=ecommerce-agent kubectl port-forwards
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=on-failure
RestartSec=10
ExecStartPre=/bin/bash -c 'until kubectl get nodes --context kind-ecommerce &>/dev/null; do sleep 3; done'
ExecStart=$SCRIPT_DIR/start.sh
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable ecommerce-portforward.service
systemctl --user start ecommerce-portforward.service

echo "Service installed and started."
echo ""
echo "Useful commands:"
echo "  systemctl --user status ecommerce-portforward   # check status"
echo "  systemctl --user restart ecommerce-portforward  # restart"
echo "  journalctl --user -u ecommerce-portforward -f   # view logs"
