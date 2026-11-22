"""Carry out actions (keystrokes, mouse, apps, shell) on a worker thread."""
import queue
import subprocess
import threading

from pynput import keyboard, mouse
from pynput.keyboard import Key

KEY_NAMES = {
    "cmd": Key.cmd, "command": Key.cmd, "ctrl": Key.ctrl, "control": Key.ctrl,
    "alt": Key.alt, "option": Key.alt, "opt": Key.alt, "shift": Key.shift,
    "enter": Key.enter, "return": Key.enter, "esc": Key.esc, "escape": Key.esc,
    "space": Key.space, "tab": Key.tab, "backspace": Key.backspace, "delete": Key.delete,
    "up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right,
    "home": Key.home, "end": Key.end, "pageup": Key.page_up, "pagedown": Key.page_down,
    "volume_up": Key.media_volume_up, "volume_down": Key.media_volume_down,
    "mute": Key.media_volume_mute, "play_pause": Key.media_play_pause,
    "next_track": Key.media_next, "prev_track": Key.media_previous,
}
for _n in range(1, 13):
    KEY_NAMES[f"f{_n}"] = getattr(Key, f"f{_n}")

MODIFIERS = {Key.cmd, Key.ctrl, Key.alt, Key.shift}

INTERNAL = {
    "pause": "Pause hand gestures", "resume": "Resume hand gestures",
    "toggle_pause": "Pause / resume hand gestures",
    "toggle_mouse": "Toggle mouse mode", "mouse_on": "Mouse mode on", "mouse_off": "Mouse mode off",
}


def parse_combo(combo):
    keys = []
    for part in str(combo).strip().lower().split("+"):
        part = part.strip()
        if not part:
            continue
        if part in KEY_NAMES:
            keys.append(KEY_NAMES[part])
        elif len(part) == 1:
            keys.append(part)
        else:
            raise ValueError(f"Unknown key '{part}'")
    if not keys:
        raise ValueError("No key given")
    return keys


def describe(action):
    t, v = action.get("type"), str(action.get("value", ""))
    if t in ("key", "media"):
        return f"press {v}"
    if t == "text":
        return f"type “{v[:30]}”"
    if t == "scroll":
        return "scroll " + ("up" if _num(v) > 0 else "down")
    if t == "click":
        return f"{v or 'left'} click"
    if t == "open_app":
        return f"open {v}"
    if t == "open_url":
        return f"open {v}"
    if t == "shell":
        return f"run `{v[:40]}`"
    if t == "internal":
        return INTERNAL.get(v, v)
    return "do nothing"


def _num(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def accessibility_ok():
    """macOS only delivers synthetic keys/clicks from apps with Accessibility access."""
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except Exception:
        return True


def screen_size():
    try:
        import Quartz
        size = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID()).size
        return int(size.width), int(size.height)
    except Exception:
        return 1440, 900


class ActionRunner:
    """Create on the main thread (keyboard layout lookup needs it on macOS)."""

    def __init__(self, on_internal, log):
        self.kb = keyboard.Controller()
        self.mouse = mouse.Controller()
        self.on_internal = on_internal
        self.log = log
        self.q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, action, source, trigger):
        self.q.put((action, source, trigger))

    def _worker(self):
        while True:
            action, source, trigger = self.q.get()
            try:
                self.run(action)
                self.log(source, trigger, describe(action), True)
            except Exception as e:  # keep the worker alive whatever an action does
                self.log(source, trigger, f"{describe(action)} failed: {e}", False)

    def run(self, action):
        t, v = action.get("type"), str(action.get("value", ""))
        if t in ("key", "media"):
            keys = parse_combo(v)
            mods = [k for k in keys if k in MODIFIERS]
            rest = [k for k in keys if k not in MODIFIERS]
            for k in mods:
                self.kb.press(k)
            try:
                for k in rest:
                    self.kb.press(k)
                    self.kb.release(k)
            finally:
                for k in reversed(mods):
                    self.kb.release(k)
        elif t == "text":
            self.kb.type(v)
        elif t == "scroll":
            self.mouse.scroll(0, _num(v, 3))
        elif t == "click":
            button = mouse.Button.right if v == "right" else mouse.Button.left
            self.mouse.click(button, 2 if v == "double" else 1)
        elif t == "open_app":
            if not v.strip():
                raise ValueError("No app name given")
            subprocess.run(["open", "-a", v.strip()], check=True, capture_output=True, timeout=15)
        elif t == "open_url":
            url = v.strip()
            if not url:
                raise ValueError("No address given")
            if "://" not in url:
                url = "https://" + url
            subprocess.run(["open", url], check=True, capture_output=True, timeout=15)
        elif t == "shell":
            if not v.strip():
                raise ValueError("No command given")
            subprocess.Popen(v, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif t == "internal":
            self.on_internal(v)
        else:
            raise ValueError("No action set")
