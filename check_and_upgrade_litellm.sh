#!/bin/bash
# Check and upgrade LiteLLM to latest version

echo "Current LiteLLM version:"
python3 -c "import litellm; print(litellm.__version__)" 2>/dev/null || echo "LiteLLM not found"

echo ""
echo "Upgrading LiteLLM to latest version..."
pip install --upgrade litellm

echo ""
echo "New LiteLLM version:"
python3 -c "import litellm; print(litellm.__version__)"

echo ""
echo "Restarting OpenClaw service..."
sudo systemctl restart openclaw.service

echo ""
echo "Done! Check logs with: sudo journalctl -u openclaw.service -f"
