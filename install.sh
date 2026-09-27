#!/usr/bin/env bash
#
# Point CLI: One-line installer script
# Usage: curl -sSL https://raw.githubusercontent.com/khokharsnehil45/point/main/install.sh | bash
#
set -euo pipefail

REPO_URL="https://github.com/khokharsnehil45/point.git"
INSTALL_DIR="${HOME}/.local/share/point"
BIN_DIR="${HOME}/.local/bin"
VENV_DIR="${INSTALL_DIR}/.venv"

# Terminal formatting
BOLD="\033[1m"
GREEN="\033[32m"
CYAN="\033[36m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

echo ""
echo -e "${CYAN}=======================================================${RESET}"
echo -e "${BOLD}${CYAN}|                    INSTALLING POINT                 |${RESET}"
echo -e "${CYAN}=======================================================${RESET}"
echo ""

# 1. Check for Python 3.10+
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}Error: python3 is required but was not found on your system.${RESET}" >&2
    echo "Please install Python 3.10 or higher and re-run this script." >&2
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10+ is required (found Python ${PY_VERSION}).${RESET}" >&2
    exit 1
fi

echo -e "  [+] Found Python ${GREEN}${PY_VERSION}${RESET}"

# 2. Check for git
if ! command -v git >/dev/null 2>&1; then
    echo -e "${RED}Error: git is required but was not found on your system.${RESET}" >&2
    exit 1
fi
echo -e "  [+] Found git"

# 3. Clone or update repository
mkdir -p "${INSTALL_DIR}"
if [ -d "${INSTALL_DIR}/.git" ]; then
    echo -e "  [*] Updating existing installation in ${INSTALL_DIR}..."
    git -C "${INSTALL_DIR}" pull --ff-only || true
else
    echo -e "  [*] Cloning Point into ${INSTALL_DIR}..."
    git clone "${REPO_URL}" "${INSTALL_DIR}"
fi

# 4. Create isolated virtual environment
echo -e "  [*] Setting up Python virtual environment..."
python3 -m venv "${VENV_DIR}"

# 5. Install package inside virtual environment
echo -e "  [*] Installing dependencies (ultralytics, pillow)..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install -e "${INSTALL_DIR}" --quiet

# 6. Symlink binary into ~/.local/bin
mkdir -p "${BIN_DIR}"
ln -sf "${VENV_DIR}/bin/point" "${BIN_DIR}/point"
chmod +x "${BIN_DIR}/point"

echo -e "  [+] Binary linked to ${GREEN}${BIN_DIR}/point${RESET}"

# 7. Check if ~/.local/bin is in PATH
PATH_NOTE=""
if [[ ":$PATH:" != *":${BIN_DIR}:"* ]]; then
    PATH_NOTE="Note: ${BIN_DIR} is not in your current PATH.\nAdd it by running:\n  export PATH=\"${BIN_DIR}:\$PATH\"\nOr append that line to your ~/.bashrc or ~/.zshrc."
fi

# 8. Success Banner
echo ""
echo -e "${GREEN}=======================================================${RESET}"
echo -e "${BOLD}${GREEN}|            POINT INSTALLED SUCCESSFULLY!            |${RESET}"
echo -e "${GREEN}=======================================================${RESET}"
echo -e "| Version    : v0.1.0                                 |"
echo -e "| Binary     : ${BIN_DIR}/point                        |"
echo -e "| Quickstart : point -model -catalog                  |"
echo -e "| Detect     : point -i image.jpg -show               |"
echo -e "${GREEN}=======================================================${RESET}"
echo ""

if [ -n "${PATH_NOTE}" ]; then
    echo -e "${YELLOW}${PATH_NOTE}${RESET}"
    echo ""
fi
