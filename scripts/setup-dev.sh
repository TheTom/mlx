#!/bin/bash
# Set up the local development environment so commits stay lint-clean.
#
# What this does:
#   1. Ensures `pre-commit` is installed (via brew on macOS, pip elsewhere).
#   2. Installs the pre-commit hook into `.git/hooks/pre-commit`, so every
#      future `git commit` runs the .pre-commit-config.yaml hooks
#      (clang-format, black, isort, cmake-format, check-yaml). This matches
#      the upstream `Check Lint` CI job exactly — if commits succeed locally,
#      CI lint passes too.
#
# Run once per fresh clone:
#   ./scripts/setup-dev.sh
#
# Re-run safely; the script is idempotent.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERR]${NC} $1"; }

# ─────────────────────────────────────────────
# 1. pre-commit binary
# ─────────────────────────────────────────────
if command -v pre-commit >/dev/null 2>&1; then
    info "pre-commit already installed ($(pre-commit --version))"
else
    if [[ "$OSTYPE" == "darwin"* ]] && command -v brew >/dev/null 2>&1; then
        info "Installing pre-commit via brew..."
        brew install pre-commit
    elif command -v pipx >/dev/null 2>&1; then
        info "Installing pre-commit via pipx..."
        pipx install pre-commit
    elif command -v pip3 >/dev/null 2>&1; then
        info "Installing pre-commit via pip3 (--user)..."
        pip3 install --user pre-commit || pip3 install --break-system-packages pre-commit
    else
        error "Neither brew, pipx, nor pip3 available. Install pre-commit manually:"
        error "  See https://pre-commit.com/#install"
        exit 1
    fi
fi

# ─────────────────────────────────────────────
# 2. Install the hook
# ─────────────────────────────────────────────
if [ -f ".git/hooks/pre-commit" ] && grep -q "pre-commit.com" ".git/hooks/pre-commit" 2>/dev/null; then
    info "pre-commit hook already installed at .git/hooks/pre-commit"
else
    info "Installing pre-commit hook..."
    pre-commit install
fi

# ─────────────────────────────────────────────
# 3. Warm the hook environments (optional — speeds up the first commit)
# ─────────────────────────────────────────────
if [ "${SKIP_INSTALL_HOOKS:-0}" != "1" ]; then
    info "Pre-fetching hook environments (clang-format, black, isort, cmake-format)..."
    pre-commit install-hooks
fi

# ─────────────────────────────────────────────
# 4. Sanity check
# ─────────────────────────────────────────────
info "Running pre-commit on the current working tree (this should pass on a clean checkout)..."
if pre-commit run --all-files >/dev/null 2>&1; then
    info "Lint clean. You're good to go."
else
    warn "pre-commit found issues on the current tree. Re-run \`pre-commit run --all-files\` to see them."
    warn "These may be pre-existing — your future commits won't be affected."
fi

echo ""
info "Done. Future \`git commit\` operations will run lint automatically."
info "Manual run: \`pre-commit run\` (changed files) or \`pre-commit run --all-files\`."
