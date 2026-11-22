#!/bin/zsh
# One-time setup: creates .venv, installs packages and downloads the models.
set -e
cd "$(dirname "$0")"

PY=$(command -v python3.12 || command -v python3)
echo "Using $PY ($("$PY" --version))"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
.venv/bin/python -m pip install -r requirements.txt -q

mkdir -p models
if [ ! -f models/hand_landmarker.task ]; then
  echo "Downloading hand tracking model…"
  curl -fsSL -o models/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
fi
if [ ! -d models/vosk-model-small-en-us-0.15 ]; then
  echo "Downloading speech model…"
  curl -fsSL -o models/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
  unzip -q models/vosk.zip -d models && rm models/vosk.zip
fi
echo "Done. Start HandControl with: .venv/bin/python -m handcontrol"
