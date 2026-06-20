#!/bin/bash
# Vault Linker — first-time setup
# Run: bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════════╗"
echo "║       Vault Linker — Setup           ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Install Python 3.11+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✓ Python ${PYTHON_VERSION} found"

# Create virtual environment
if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${SCRIPT_DIR}/venv"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate and install
source "${SCRIPT_DIR}/venv/bin/activate"
echo "Installing dependencies..."
pip install -q -r "${SCRIPT_DIR}/requirements.txt"
echo "✓ Dependencies installed"

# Check for OpenAI key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "⚠ No OPENAI_API_KEY environment variable found."
    echo "  You can either:"
    echo "    1. Export it:  export OPENAI_API_KEY='sk-...'"
    echo "    2. Set it in:  vault_linker/config.py"
    echo ""
fi

# Offer to create shell alias
echo ""
echo "Add a shell alias? This lets you type 'vault' to launch."
read -p "Add to ~/.zshrc? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    ALIAS_LINE="alias vault='${SCRIPT_DIR}/venv/bin/python ${SCRIPT_DIR}/run.py'"
    if ! grep -q "alias vault=" ~/.zshrc 2>/dev/null; then
        echo "" >> ~/.zshrc
        echo "# Vault Linker" >> ~/.zshrc
        echo "${ALIAS_LINE}" >> ~/.zshrc
        echo "✓ Alias added. Run 'source ~/.zshrc' or open a new terminal."
    else
        echo "✓ Alias already exists in ~/.zshrc"
    fi
fi

echo ""
echo "╔══════════════════════════════════════╗"
echo "║            Setup complete!           ║"
echo "╠══════════════════════════════════════╣"
echo "║  Launch:  python run.py              ║"
echo "║  Or:      vault  (after alias)       ║"
echo "║                                      ║"
echo "║  Keys:                               ║"
echo "║    ↑/↓   Navigate notes              ║"
echo "║    Enter  Select note                ║"
echo "║    h      Hide/show linked notes     ║"
echo "║    r      Re-embed all notes         ║"
echo "║    q      Quit                       ║"
echo "╚══════════════════════════════════════╝"
