#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh
# One script to prepare the whole environment for the Phishing Email
# Detection System:
#   1. Create a virtual environment (if one doesn't already exist)
#   2. Install all required Python dependencies
#   3. Train the ML model on the bundled sample dataset
#      (this also creates and populates the SQLite database with the
#       first model_metadata row)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ---------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/3] Creating virtual environment (venv) ..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "[2/3] Installing dependencies from requirements.txt ..."
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo "[3/3] Training the model on the sample dataset (also initializes the database) ..."
python train_model.py

echo ""
echo "Setup complete."
echo "Activate the environment with: source venv/bin/activate"
echo "Then try:"
echo "  python main.py analyze --subject \"Verify your account\" --body \"Click here http://fake-bank.tk immediately\""
echo "  python main.py list"
echo "  python main.py stats"
