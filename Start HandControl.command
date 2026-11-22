#!/bin/zsh
# Double-click to start HandControl. Close this window (or press Ctrl+C) to stop it.
cd "$(dirname "$0")"
[ -x .venv/bin/python ] && [ -d models ] || ./setup.sh
exec .venv/bin/python -m handcontrol
