#!/usr/bin/env bash
# Installs Agent Reach (https://github.com/Panniantong/Agent-Reach) — a CLI
# that gives an agent read access to the web, YouTube, GitHub, RSS, etc.
#
# Everything this script does lives outside this repo (a venv under
# ~/.agent-reach-venv, plus a Claude Code skill under ~/.claude/skills/), so
# it needs to be re-run in every fresh session/container — nothing here
# persists on its own. Re-running is safe (idempotent).
#
# Usage:
#   bash scripts/install_agent_reach.sh            # zero-config channels only
#   bash scripts/install_agent_reach.sh --system    # + gh CLI, mcporter, yt-dlp config
#
# Login-gated channels (Twitter, Reddit, Xiaohongshu, Xueqiu, Facebook,
# Instagram, LinkedIn, podcast transcription) are NOT installed here since
# they need per-user credentials. Ask your agent to "install <channel>" once
# this base install is done, or see docs/install.md in the upstream repo.

set -euo pipefail

ARCHIVE_URL="https://github.com/Panniantong/agent-reach/archive/main.zip"
VENV_DIR="${HOME}/.agent-reach-venv"

if command -v pipx >/dev/null 2>&1; then
    pipx install "$ARCHIVE_URL" --force
else
    python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade "$ARCHIVE_URL"
fi

# In sandboxes where the archive download itself is blocked (e.g. a Claude
# Code Remote session whose git proxy only allows attached repos), fall back
# to cloning and installing from the local checkout:
#   git clone --depth 1 https://github.com/Panniantong/agent-reach /tmp/agent-reach
#   pip install /tmp/agent-reach

agent-reach install --env=auto "$@"
agent-reach doctor

if [ -d "$VENV_DIR" ]; then
    echo
    echo "Installed into a venv. In new shells, run:"
    echo "  source $VENV_DIR/bin/activate"
    echo "before using the 'agent-reach' or upstream (yt-dlp, gh, ...) commands."
fi
