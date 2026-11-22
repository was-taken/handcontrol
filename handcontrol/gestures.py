"""Turn hand landmarks into named gestures.

Pure logic, no camera or OS access, so it can be tested with synthetic points.
Points are 21 (x, y) tuples in the MediaPipe hand layout, with x and y in the
same unit (frame widths), on a mirrored (selfie) image.
"""
import math
from collections import deque

GESTURES = [
    {"id": "swipe_left", "label": "Swipe left", "emoji": "⬅️", "kind": "motion"},
    {"id": "swipe_right", "label": "Swipe right", "emoji": "➡️", "kind": "motion"},
    {"id": "swipe_up", "label": "Swipe up", "emoji": "⬆️", "kind": "motion"},
    {"id": "swipe_down", "label": "Swipe down", "emoji": "⬇️", "kind": "motion"},
    {"id": "open_palm", "label": "Open palm", "emoji": "🖐️", "kind": "pose"},
    {"id": "fist", "label": "Fist", "emoji": "✊", "kind": "pose"},
    {"id": "point", "label": "Point (index finger)", "emoji": "☝️", "kind": "pose"},
    {"id": "peace", "label": "Peace (two fingers)", "emoji": "✌️", "kind": "pose"},
    {"id": "three", "label": "Three fingers", "emoji": "3️⃣", "kind": "pose"},
    {"id": "four", "label": "Four fingers", "emoji": "4️⃣", "kind": "pose"},
    {"id": "thumbs_up", "label": "Thumbs up", "emoji": "👍", "kind": "pose"},
    {"id": "thumbs_down", "label": "Thumbs down", "emoji": "👎", "kind": "pose"},
    {"id": "pinch", "label": "Pinch (thumb + index)", "emoji": "🤏", "kind": "pose"},
    {"id": "rock", "label": "Rock (index + pinky)", "emoji": "🤘", "kind": "pose"},
    {"id": "shaka", "label": "Shaka (thumb + pinky)", "emoji": "🤙", "kind": "pose"},
    {"id": "l_shape", "label": "L shape (thumb + index)", "emoji": "👆", "kind": "pose"},
]

# Poses that steer the cursor while mouse mode is on.
MOUSE_POSES = ("point", "pinch", "l_shape")

SWIPE_WINDOW = 0.4  # seconds of palm history a swipe is measured over
STILL_LIMIT = 0.07  # palm travel (frame widths) above which a pose hold restarts
DEBOUNCE_FRAMES = 3


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_states(pts):
    """(thumb, index, middle, ring, pinky) -> True when extended."""
    wrist = pts[0]
    palm = _d(wrist, pts[9]) or 1e-6
    fingers = [
        _d(pts[tip], wrist) > _d(pts[pip], wrist) * 1.12
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    ]
    thumb = _d(pts[4], pts[5]) / palm > 0.6
    return (thumb, *fingers)


def classify_builtin(pts):
    palm = _d(pts[0], pts[9]) or 1e-6
    t, i, m, r, p = finger_states(pts)
    # Pinch: thumb and index tips touching, index not curled into a fist.
    if _d(pts[4], pts[8]) / palm < 0.25 and _d(pts[8], pts[5]) / palm > 0.5:
        return "pinch"
    if i and m and r and p:
        return "open_palm" if t else "four"
    if i and m and r and not p:
        return "three"
    if i and m and not r and not p:
        return "peace"
    if i and p and not m and not r:
        return "rock"
    if t and p and not i and not m and not r:
        return "shaka"
    if i and not m and not r and not p:
        return "l_shape" if t else "point"
    if not (i or m or r or p):
        if not t:
            return "fist"
        # The thumb has to point mostly up or down, not sideways.
        dx = (pts[4][0] - pts[2][0]) / palm
        dy = (pts[4][1] - pts[2][1]) / palm
        if abs(dy) > abs(dx) and abs(dy) > 0.3:
            return "thumbs_up" if dy < 0 else "thumbs_down"
    return None


def normalize(pts, hand_label):
    """Wrist-centred, palm-scaled copy of the hand; left hands are mirrored
    so one recorded pose matches either hand."""
    wx, wy = pts[0]
    palm = _d(pts[0], pts[9]) or 1e-6
    sign = -1.0 if hand_label == "Left" else 1.0
    return [[sign * (x - wx) / palm, (y - wy) / palm] for x, y in pts]


def pose_distance(a, b):
    """Largest single-joint mismatch, in palm lengths. A mean would let one
    differently-held finger hide among twenty matching joints."""
    return max(_d(p, q) for p, q in zip(a, b))


def classify(pts, hand_label, custom_poses=(), tolerance=0.4):
    """Return a gesture id: a matching custom pose wins over built-ins."""
    if custom_poses:
        norm = normalize(pts, hand_label)
        best, best_d = None, tolerance
        for cp in custom_poses:
            tpl = cp.get("template")
            if not tpl or len(tpl) != 21:
                continue
            dist = pose_distance(norm, tpl)
            if dist < best_d:
                best, best_d = cp["id"], dist
        if best:
            return "custom:" + best
    return classify_builtin(pts)


class HandTracker:
    """Per-hand state: debounces poses, times holds, and detects swipes."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.stable = None
        self._cand = None
        self._cand_n = 0
        self.since = 0.0
        self.fired = False
        self.last_fire = 0.0
        self.hist = deque()
        self.block_until = 0.0
        self.last_seen = 0.0
        self.progress = 0.0

    def update(self, pose, center, t, s):
        """Feed one frame. Returns events: ("pose"|"repeat"|"swipe", gesture_id)."""
        events = []
        self.last_seen = t

        if pose == self.stable:
            self._cand, self._cand_n = None, 0
        else:
            if pose == self._cand:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = pose, 1
            if self._cand_n >= DEBOUNCE_FRAMES:
                self.stable, self.since, self.fired = pose, t, False
                self._cand, self._cand_n = None, 0

        self.hist.append((t, center[0], center[1]))
        while self.hist and t - self.hist[0][0] > SWIPE_WINDOW:
            self.hist.popleft()
        _, x0, y0 = self.hist[0]
        dx, dy = center[0] - x0, center[1] - y0

        dist = s["swipe_distance"]
        open_ok = (not s["swipe_open_hand_only"]) or self.stable in ("open_palm", "four")
        if t >= self.block_until and open_ok and len(self.hist) >= 4:
            swipe = None
            if abs(dx) > dist and abs(dx) > 1.8 * abs(dy):
                swipe = "swipe_right" if dx > 0 else "swipe_left"
            elif abs(dy) > dist * 0.75 and abs(dy) > 1.8 * abs(dx):
                swipe = "swipe_down" if dy > 0 else "swipe_up"
            if swipe:
                events.append(("swipe", swipe))
                self.hist.clear()
                # Ignore the return stroke, and restart any pose hold.
                self.block_until = t + s["swipe_cooldown_ms"] / 1000.0
                self.since = t
                dx = dy = 0.0

        if not self.fired and math.hypot(dx, dy) > STILL_LIMIT:
            self.since = t  # a pose only counts while the hand is held still

        hold = s["hold_ms"] / 1000.0
        self.progress = 0.0
        if self.stable:
            if not self.fired:
                self.progress = min(1.0, (t - self.since) / hold) if hold > 0 else 1.0
                if t - self.since >= hold:
                    self.fired, self.last_fire, self.progress = True, t, 1.0
                    events.append(("pose", self.stable))
            else:
                self.progress = 1.0
                if t - self.last_fire >= s["repeat_ms"] / 1000.0:
                    self.last_fire = t
                    events.append(("repeat", self.stable))
        return events
