"""HandControl: camera loop, settings store and the local settings web page."""
import json
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .actions import INTERNAL, ActionRunner, accessibility_ok, screen_size
from .gestures import GESTURES, MOUSE_POSES, HandTracker, classify, normalize
from .voice import VoiceListener

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
VOICE_MODEL = ROOT / "models" / "vosk-model-small-en-us-0.15"
WEB = ROOT / "web" / "index.html"
PORT = 8765

# name: (default, min, max) for numbers
SETTINGS = {
    "camera_index": (0, 0, 9),
    "hold_ms": (400, 100, 3000),
    "repeat_ms": (300, 100, 3000),
    "swipe_distance": (0.22, 0.08, 0.6),
    "swipe_cooldown_ms": (800, 200, 3000),
    "swipe_open_hand_only": True,
    "pose_tolerance": (0.4, 0.1, 1.0),
    "mouse_smoothing": (0.6, 0.0, 0.95),
    "start_paused": False,
    "voice_enabled": True,
    "wake_word": "computer",
}


def _a(type_, value):
    return {"type": type_, "value": value}


def default_config():
    def b(gesture, action, repeat=False):
        return {"id": uuid.uuid4().hex[:8], "gesture": gesture, "hand": "any",
                "action": action, "repeat": repeat, "enabled": True}

    def v(phrase, action):
        return {"id": uuid.uuid4().hex[:8], "phrase": phrase, "action": action, "enabled": True}

    return {
        "settings": {k: (d[0] if isinstance(d, tuple) else d) for k, d in SETTINGS.items()},
        "bindings": [
            b("swipe_right", _a("key", "right")),
            b("swipe_left", _a("key", "left")),
            b("swipe_up", _a("key", "ctrl+up")),
            b("swipe_down", _a("key", "ctrl+down")),
            b("thumbs_up", _a("media", "volume_up"), repeat=True),
            b("thumbs_down", _a("media", "volume_down"), repeat=True),
            b("peace", _a("media", "play_pause")),
            b("rock", _a("internal", "toggle_mouse")),
        ],
        "voice": [
            v("start hand control", _a("internal", "resume")),
            v("stop hand control", _a("internal", "pause")),
            v("mouse mode", _a("internal", "toggle_mouse")),
            v("next slide", _a("key", "right")),
            v("previous slide", _a("key", "left")),
            v("volume up", _a("media", "volume_up")),
            v("volume down", _a("media", "volume_down")),
            v("play music", _a("media", "play_pause")),
            v("pause music", _a("media", "play_pause")),
            v("take a screenshot", _a("key", "cmd+shift+3")),
        ],
        "custom_poses": [],
    }


def _action(raw):
    raw = raw if isinstance(raw, dict) else {}
    return {"type": str(raw.get("type", "key"))[:20], "value": str(raw.get("value", ""))[:2000]}


def sanitize(raw):
    """Coerce anything posted by the page into a well-formed config."""
    base = default_config()
    out = {"settings": base["settings"], "bindings": [], "voice": [], "custom_poses": []}
    for k, spec in SETTINGS.items():
        val = (raw.get("settings") or {}).get(k)
        if val is None:
            continue
        try:
            if isinstance(spec, tuple):
                d, lo, hi = spec
                out["settings"][k] = max(lo, min(hi, type(d)(float(val))))
            elif isinstance(spec, bool):
                out["settings"][k] = bool(val)
            else:
                out["settings"][k] = str(val)[:60]
        except (TypeError, ValueError):
            pass
    for b in raw.get("bindings") or []:
        hand = b.get("hand")
        out["bindings"].append({
            "id": str(b.get("id") or uuid.uuid4().hex[:8])[:40],
            "gesture": str(b.get("gesture", ""))[:80],
            "hand": hand if hand in ("Left", "Right") else "any",
            "action": _action(b.get("action")),
            "repeat": bool(b.get("repeat")), "enabled": bool(b.get("enabled", True)),
        })
    for v in raw.get("voice") or []:
        out["voice"].append({
            "id": str(v.get("id") or uuid.uuid4().hex[:8])[:40],
            "phrase": str(v.get("phrase", ""))[:80],
            "action": _action(v.get("action")), "enabled": bool(v.get("enabled", True)),
        })
    for p in raw.get("custom_poses") or []:
        tpl = p.get("template")
        try:
            tpl = [[float(x), float(y)] for x, y in tpl]
        except (TypeError, ValueError):
            continue
        if len(tpl) == 21:
            out["custom_poses"].append({"id": str(p.get("id") or uuid.uuid4().hex[:8])[:40],
                                        "name": str(p.get("name") or "Custom pose")[:40],
                                        "template": tpl})
    return out


class Store:
    def __init__(self):
        self.lock = threading.Lock()
        try:
            self.cfg = sanitize(json.loads(CONFIG_PATH.read_text()))
        except FileNotFoundError:
            self.cfg = default_config()
        except Exception as e:
            print(f"config.json unreadable ({e}); starting from defaults", file=sys.stderr)
            self.cfg = default_config()

    def save(self, raw):
        with self.lock:
            cfg = sanitize(raw)
            tmp = CONFIG_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(cfg, indent=1))
            tmp.replace(CONFIG_PATH)
            self.cfg = cfg  # readers hold a snapshot; swap is atomic
            return cfg


CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
               (18, 19), (19, 20), (0, 17)]
GESTURE_LABELS = {g["id"]: g["label"] for g in GESTURES}


class Engine:
    def __init__(self, store):
        self.store = store
        self.paused = store.cfg["settings"]["start_paused"]
        self.mouse_mode = False
        self.events = deque(maxlen=40)
        self.runner = ActionRunner(self.internal, self.log)
        self.trackers = {}
        self.hand = None
        self.fps = 0.0
        self.camera_error = "Starting camera…"
        self.frame_cond = threading.Condition()
        self.jpeg = None
        self.frame_id = 0
        self.viewer_ping = 0.0
        self.record = None
        self.screen = screen_size()
        self.mouse_pos = None
        self.mouse_down = False
        self._ax = (0.0, True)
        self.voice = VoiceListener(VOICE_MODEL, store, self.on_voice)

    # ---- shared helpers -------------------------------------------------
    def log(self, source, trigger, text, ok):
        self.events.appendleft({"t": time.time(), "source": source, "trigger": trigger,
                                "text": text, "ok": ok})

    def internal(self, name):
        if name == "pause":
            self.paused = True
        elif name == "resume":
            self.paused = False
        elif name == "toggle_pause":
            self.paused = not self.paused
        elif name == "toggle_mouse":
            self.mouse_mode = not self.mouse_mode
        elif name == "mouse_on":
            self.mouse_mode = True
        elif name == "mouse_off":
            self.mouse_mode = False
        else:
            raise ValueError(f"Unknown HandControl action '{name}'")

    def on_voice(self, entry):
        self.runner.submit(entry["action"], "voice", f"“{entry['phrase']}”")

    def gesture_label(self, gid):
        if gid and gid.startswith("custom:"):
            for p in self.store.cfg["custom_poses"]:
                if p["id"] == gid[7:]:
                    return p["name"]
            return "Custom pose"
        return GESTURE_LABELS.get(gid, "")

    def status(self):
        now = time.time()
        if now - self._ax[0] > 2:
            self._ax = (now, accessibility_ok())
        v = self.voice.status
        return {"paused": self.paused, "mouse_mode": self.mouse_mode,
                "camera_error": self.camera_error, "fps": round(self.fps),
                "hand": self.hand, "accessibility": self._ax[1],
                "voice": {**v, "armed": self.voice.armed},
                "events": list(self.events)}

    def record_pose(self, seconds=1.2):
        """Average the hand shape seen over the next moment; None if no hand."""
        rec = {"until": time.time() + seconds, "samples": [], "done": threading.Event(),
               "template": None}
        self.record = rec
        rec["done"].wait(seconds + 2)
        self.record = None
        return rec["template"]

    # ---- per-frame work -------------------------------------------------
    def _dispatch(self, kind, gid, label):
        for b in self.store.cfg["bindings"]:
            if not b["enabled"] or b["gesture"] != gid or b["hand"] not in ("any", label):
                continue
            if kind == "repeat" and not b["repeat"]:
                continue
            act = b["action"]
            # While paused, only a gesture that un-pauses still works.
            if self.paused and not (act["type"] == "internal"
                                    and act["value"] in ("resume", "toggle_pause")):
                continue
            self.runner.submit(act, "gesture", self.gesture_label(gid))

    def _release_mouse(self):
        if self.mouse_down:
            self.runner.mouse.release(self._button())
            self.mouse_down = False
        self.mouse_pos = None

    @staticmethod
    def _button():
        from pynput.mouse import Button
        return Button.left

    def _steer_mouse(self, tracker, lms, s):
        pose = tracker.stable
        if pose in MOUSE_POSES:
            w, h = self.screen
            # The index knuckle stays put while pinching, so clicks don't jerk the cursor.
            tx = max(0.0, min(1.0, (lms[5].x - 0.25) / 0.5)) * (w - 1)
            ty = max(0.0, min(1.0, (lms[5].y - 0.25) / 0.45)) * (h - 1)
            if self.mouse_pos is None:
                self.mouse_pos = (tx, ty)
            else:
                a = 1.0 - s["mouse_smoothing"]
                px, py = self.mouse_pos
                self.mouse_pos = (px + (tx - px) * a, py + (ty - py) * a)
            self.runner.mouse.position = self.mouse_pos
        pinching = pose == "pinch"
        if pinching and not self.mouse_down:
            self.runner.mouse.press(self._button())
            self.mouse_down = True
        elif not pinching and self.mouse_down:
            self.runner.mouse.release(self._button())
            self.mouse_down = False

    def _process(self, result, frame, t):
        cfg = self.store.cfg
        s = cfg["settings"]
        h, w = frame.shape[:2]
        aspect = h / w
        self.hand = None
        first = True
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            # MediaPipe's handedness comes out reversed on our mirrored frames.
            label = "Left" if handed[0].category_name == "Right" else "Right"
            pts = [(lm.x, lm.y * aspect) for lm in lms]
            pose = classify(pts, label, cfg["custom_poses"], s["pose_tolerance"])
            center = tuple(sum(pts[i][k] for i in (0, 5, 17)) / 3 for k in (0, 1))
            tracker = self.trackers.setdefault(label, HandTracker())
            events = tracker.update(pose, center, t, s)

            rec = self.record
            if rec and first and t <= rec["until"]:
                rec["samples"].append(normalize(pts, label))

            steering = self.mouse_mode and not self.paused and first
            if steering:
                self._steer_mouse(tracker, lms, s)
            for kind, gid in events:
                if self.mouse_mode and (kind == "swipe" or gid in MOUSE_POSES):
                    continue
                self._dispatch(kind, gid, label)

            if first:
                self.hand = {"label": label, "gesture": tracker.stable,
                             "name": self.gesture_label(tracker.stable),
                             "progress": round(tracker.progress, 2)}
            self._draw_hand(frame, lms, label, tracker)
            first = False

        if not result.hand_landmarks or not self.mouse_mode or self.paused:
            self._release_mouse()
        for tracker in self.trackers.values():
            if tracker.stable is not None and t - tracker.last_seen > 0.4:
                tracker.reset()

        rec = self.record
        if rec and t > rec["until"] and not rec["done"].is_set():
            if len(rec["samples"]) >= 5:
                n = len(rec["samples"])
                rec["template"] = [[round(sum(smp[i][k] for smp in rec["samples"]) / n, 4)
                                    for k in (0, 1)] for i in range(21)]
            rec["done"].set()

    def _draw_hand(self, frame, lms, label, tracker):
        h, w = frame.shape[:2]
        px = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        color = (120, 220, 90) if not self.paused else (150, 150, 150)
        for a, b in CONNECTIONS:
            cv2.line(frame, px[a], px[b], color, 2, cv2.LINE_AA)
        for p in px:
            cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
        name = self.gesture_label(tracker.stable)
        text = f"{label}: {name}" if name else label
        cv2.putText(frame, text, (px[0][0] - 40, min(h - 10, px[0][1] + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (px[0][0] - 40, min(h - 10, px[0][1] + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def _publish(self, frame, t):
        if t - self.viewer_ping > 3:
            return  # nobody is watching the preview
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.frame_cond:
                self.jpeg = buf.tobytes()
                self.frame_id += 1
                self.frame_cond.notify_all()

    def run(self):
        """Camera loop. Must run on the main thread (macOS camera permission)."""
        landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2))
        self.voice.start()
        cap, cam_index, retry_at, fails = None, None, 0.0, 0
        t0, last_ts, last_pub, fps_t, fps_n = time.monotonic(), -1, 0.0, time.time(), 0
        while True:
            want = self.store.cfg["settings"]["camera_index"]
            if cap is None or cam_index != want:
                if cap is None and time.time() < retry_at:
                    time.sleep(0.1)
                    continue
                if cap is not None:
                    cap.release()
                cam_index, fails = want, 0
                cap = cv2.VideoCapture(cam_index)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if not ok:
                fails += 1
                if fails >= 20 or not cap.isOpened():
                    self.camera_error = (
                        f"Can't read from camera {cam_index}. Allow camera access for the app "
                        "you started HandControl from (System Settings → Privacy & Security → "
                        "Camera), then restart it. Or pick another camera in Settings.")
                    self.hand = None
                    cap.release()
                    cap, retry_at = None, time.time() + 5
                else:
                    time.sleep(0.05)
                continue
            fails, self.camera_error = 0, ""
            t = time.time()
            frame = cv2.flip(frame, 1)  # mirror, so moving right looks like moving right
            ts = max(last_ts + 1, int((time.monotonic() - t0) * 1000))
            last_ts = ts
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, ts)
            self._process(result, frame, t)
            if t - last_pub > 0.06:
                last_pub = t
                self._publish(frame, t)
            fps_n += 1
            if t - fps_t >= 1:
                self.fps, fps_t, fps_n = fps_n / (t - fps_t), t, 0


def make_handler(engine):
    store = engine.store
    hosts = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _trusted(self):
            # Only this machine's own page may talk to us: blocks other websites
            # (cross-origin posts, DNS rebinding) from rebinding gestures.
            if self.headers.get("Host") not in hosts:
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin in {"http://" + h for h in hosts}

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._trusted():
                return self._send(403, {"error": "forbidden"})
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, WEB.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/config":
                self._send(200, {"config": store.cfg, "gestures": GESTURES, "internal": INTERNAL})
            elif path == "/api/status":
                self._send(200, engine.status())
            elif path == "/stream":
                self._stream()
            else:
                self._send(404, {"error": "not found"})

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            seen = -1
            try:
                while True:
                    engine.viewer_ping = time.time()
                    with engine.frame_cond:
                        engine.frame_cond.wait_for(
                            lambda: engine.jpeg is not None and engine.frame_id != seen, timeout=1)
                        if engine.jpeg is None or engine.frame_id == seen:
                            continue  # no new frame yet (the camera may still be starting)
                        seen, jpeg = engine.frame_id, engine.jpeg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if not self._trusted() or "json" not in (self.headers.get("Content-Type") or ""):
                return self._send(403, {"error": "forbidden"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                data = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._send(400, {"error": "bad json"})
            if self.path == "/api/config":
                self._send(200, {"config": store.save(data)})
            elif self.path == "/api/control":
                if "paused" in data:
                    engine.paused = bool(data["paused"])
                if "mouse_mode" in data:
                    engine.mouse_mode = bool(data["mouse_mode"])
                self._send(200, engine.status())
            elif self.path == "/api/test":
                engine.runner.submit(data.get("action") or {}, "test", "Test button")
                self._send(200, {"ok": True})
            elif self.path == "/api/record_pose":
                template = engine.record_pose()
                if not template:
                    return self._send(200, {"ok": False, "error": "No hand was visible. "
                                            "Hold the pose in view of the camera and try again."})
                cfg = dict(store.cfg)
                pose = {"id": uuid.uuid4().hex[:8],
                        "name": str(data.get("name") or "Custom pose")[:40], "template": template}
                cfg["custom_poses"] = cfg["custom_poses"] + [pose]
                self._send(200, {"ok": True, "id": pose["id"], "config": store.save(cfg)})
            else:
                self._send(404, {"error": "not found"})

    return Handler


def main():
    for path in (HAND_MODEL, VOICE_MODEL):
        if not path.exists():
            sys.exit(f"Missing model: {path}")
    engine = Engine(Store())
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(engine))
    except OSError as e:
        sys.exit(f"Can't open http://127.0.0.1:{PORT} ({e}). Is HandControl already running?")
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    print(f"HandControl is running. Settings page: {url}   (Ctrl+C to quit)", flush=True)
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        engine.run()
    except KeyboardInterrupt:
        print("\nBye.")
