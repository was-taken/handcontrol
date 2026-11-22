# HandControl

Control your Mac with hand gestures (through the webcam) and voice commands.
Everything runs on this computer: no video or audio is sent anywhere.

## Setup

Needs macOS and Python 3.10–3.12. Run once:

```bash
./setup.sh
```

This creates `.venv`, installs the packages in `requirements.txt` and downloads the hand-tracking
and speech models (about 80 MB) into `models/`.

## Start

Double-click **Start HandControl.command**, or run:

```bash
.venv/bin/python -m handcontrol
```

A settings page opens at http://127.0.0.1:8765. Keep the Terminal window open while you use it; Ctrl+C stops it.

The first time, macOS asks for three permissions for the app you launched it from (usually Terminal).
Grant them under System Settings → Privacy & Security, then restart HandControl:

- **Camera**: to see your hand
- **Microphone**: for voice commands
- **Accessibility**: to press keys and move the mouse

## Use

- **Hand gestures tab**: pick a gesture, which hand, and what it does. Hold a pose still for a moment
  to trigger it; swipe with an open hand. Tick *Repeat* to keep firing while held (volume, scrolling).
- **Voice commands tab**: say the wake word ("computer") then a phrase, e.g. "computer, next slide".
  "computer, stop hand control" / "start hand control" turn gestures off and on by voice.
- **Custom poses tab**: record your own hand shape and bind it like a built-in gesture.
- **Mouse mode**: point to move the cursor, pinch thumb and index to click or drag.
- **Settings tab**: wake word, hold time, swipe sensitivity, camera.

Actions can press key combos, control media and volume, type text, scroll, click, open apps or
websites, or run a shell command. Your setup is saved in `config.json`.

## Layout

- `handcontrol/gestures.py`: landmarks → gestures (poses, holds, swipes, custom poses)
- `handcontrol/actions.py`: keystrokes, mouse, apps
- `handcontrol/voice.py`: offline speech recognition (Vosk)
- `handcontrol/app.py`: camera loop, config, local web server
- `web/index.html`: settings page
- `tests/test_gestures.py`: `.venv/bin/python tests/test_gestures.py`
