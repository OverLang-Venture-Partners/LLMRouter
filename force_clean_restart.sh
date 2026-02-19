#!/bin/bash
# Force clean restart - remove Python cache and restart service

echo "Stopping OpenClaw service..."
sudo systemctl stop openclaw.service

echo "Removing Python cache files..."
find /home/ubuntu/.openclaw/LLMRouter -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find /home/ubuntu/.openclaw/LLMRouter -type f -name "*.pyc" -delete 2>/dev/null || true

echo "Waiting 2 seconds..."
sleep 2

echo "Starting OpenClaw service..."
sudo systemctl start openclaw.service

echo "Waiting for service to start..."
sleep 3

echo "Service status:"
sudo systemctl status openclaw.service --no-pager | head -20

echo ""
echo "Watching logs (Ctrl+C to stop):"
sudo journalctl -u openclaw.service -f
