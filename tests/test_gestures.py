"""Synthetic-hand tests for the gesture logic. Run: .venv/bin/python tests/test_gestures.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from handcontrol.actions import parse_combo  # noqa: E402
from handcontrol.gestures import HandTracker, classify, classify_builtin, normalize  # noqa: E402
from handcontrol.voice import VoiceListener, clean  # noqa: E402

S = {"hold_ms": 400, "repeat_ms": 300, "swipe_distance": 0.22, "swipe_cooldown_ms": 800,
     "swipe_open_hand_only": True}


def hand(thumb="out", index=True, middle=True, ring=True, pinky=True, pinch=False):
    """Upright right hand, wrist at the bottom. y grows downward, palm length = 1."""
    pts = [(0.0, 0.0)] * 21
    pts[0] = (0.0, 0.0)
    for base, x, up in ((5, -0.35, index), (9, 0.0, middle), (13, 0.3, ring), (17, 0.55, pinky)):
        mcp_y = -1.0 if base != 17 else -0.8
        pts[base] = (x, mcp_y)
        if up:
            pts[base + 1], pts[base + 2], pts[base + 3] = (x, mcp_y - 0.4), (x, mcp_y - 0.65), (x, mcp_y - 0.9)
        else:  # curled back toward the palm
            pts[base + 1], pts[base + 2], pts[base + 3] = (x, mcp_y - 0.3), (x, mcp_y - 0.1), (x, mcp_y + 0.2)
    pts[1], pts[2] = (-0.35, -0.25), (-0.6, -0.5)
    if thumb == "out":
        pts[3], pts[4] = (-0.85, -0.7), (-1.1, -0.9)
    elif thumb == "up":
        pts[3], pts[4] = (-0.65, -0.9), (-0.7, -1.3)
    elif thumb == "down":
        pts[3], pts[4] = (-0.7, -0.3), (-0.75, 0.0)
    else:  # tucked
        pts[3], pts[4] = (-0.5, -0.7), (-0.3, -0.8)
    if pinch:
        pts[6], pts[7], pts[8] = (-0.5, -1.35), (-0.75, -1.5), (-0.95, -1.35)
        pts[3], pts[4] = (-0.8, -0.85), (-0.95, -1.2)
    return pts


def check(name, got, want):
    assert got == want, f"{name}: got {got!r}, want {want!r}"
    print(f"ok  {name}")


def test_poses():
    check("open palm", classify_builtin(hand()), "open_palm")
    check("four", classify_builtin(hand(thumb="in")), "four")
    check("fist", classify_builtin(hand("in", False, False, False, False)), "fist")
    check("point", classify_builtin(hand("in", True, False, False, False)), "point")
    check("peace", classify_builtin(hand("in", True, True, False, False)), "peace")
    check("three", classify_builtin(hand("in", True, True, True, False)), "three")
    check("rock", classify_builtin(hand("in", True, False, False, True)), "rock")
    check("shaka", classify_builtin(hand("out", False, False, False, True)), "shaka")
    check("L shape", classify_builtin(hand("out", True, False, False, False)), "l_shape")
    # A real thumbs up is a fist turned sideways with the thumb sticking out.
    side_fist = hand("out", False, False, False, False)
    check("thumbs up", classify_builtin([(-y, x) for x, y in side_fist]), "thumbs_up")
    check("thumbs down", classify_builtin([(y, -x) for x, y in side_fist]), "thumbs_down")
    check("sideways thumb is neither", classify_builtin(side_fist), None)
    check("pinch", classify_builtin(hand(pinch=True)), "pinch")


def test_custom():
    spock = hand("out", True, True, False, True)
    custom = [{"id": "abc", "name": "x", "template": normalize(spock, "Right")}]
    check("custom match", classify(spock, "Right", custom), "custom:abc")
    mirrored = [(-x, y) for x, y in spock]
    check("custom matches other hand", classify(mirrored, "Left", custom), "custom:abc")
    check("custom doesn't steal others", classify(hand(), "Right", custom), "open_palm")


def run(tracker, frames):
    out = []
    for t, pose, x, y in frames:
        out += tracker.update(pose, (x, y), t, S)
    return out


def test_tracker():
    tr = HandTracker()
    still = [(i / 30, "peace", 0.5, 0.4) for i in range(40)]
    ev = run(tr, still)
    check("hold fires once, then repeats", ev, [("pose", "peace"), ("repeat", "peace"), ("repeat", "peace")])

    tr = HandTracker()
    ev = run(tr, [(i / 30, "peace", 0.5, 0.4) for i in range(8)])
    check("short hold doesn't fire", ev, [])

    tr = HandTracker()
    frames = [(i / 30, "open_palm", 0.3, 0.4) for i in range(6)]
    frames += [(0.2 + i / 30, "open_palm", 0.3 + i * 0.04, 0.4) for i in range(1, 9)]  # sweep right
    frames += [(0.5 + i / 30, "open_palm", 0.62 - i * 0.04, 0.4) for i in range(1, 9)]  # hand returns
    check("swipe right once, return ignored", run(tr, frames), [("swipe", "swipe_right")])

    tr = HandTracker()
    frames = [(i / 30, "fist", 0.3 + i * 0.04, 0.4) for i in range(12)]
    check("no swipe with closed hand", run(tr, frames), [])

    tr = HandTracker()
    frames = [(i / 30, "open_palm", 0.5, 0.5 - i * 0.03) for i in range(12)]
    check("swipe up", run(tr, frames), [("swipe", "swipe_up")])

    tr = HandTracker()
    frames = [(i / 30, "peace" if i % 5 else None, 0.5, 0.4) for i in range(20)]
    check("one-frame flicker doesn't break a hold", run(tr, frames), [("pose", "peace")])


def test_misc():
    check("combo", [str(k) for k in parse_combo("Cmd+Shift+t")], ["Key.cmd", "Key.shift", "t"])
    check("clean", clean("  Next   SLIDE! "), "next slide")
    fired = []
    v = VoiceListener(None, None, fired.append)
    phrases = {"next slide": "NEXT"}
    v._handle("next slide", phrases, "computer")
    check("wake word required", fired, [])
    v._handle("computer next slide", phrases, "computer")
    check("wake + command", fired, ["NEXT"])
    v._handle("computer", phrases, "computer")
    v._handle("[unk] next slide", phrases, "computer")
    check("wake, then command", fired, ["NEXT", "NEXT"])
    v._handle("next slide", phrases, "")
    check("no wake word set", fired, ["NEXT", "NEXT", "NEXT"])


if __name__ == "__main__":
    test_poses(), test_custom(), test_tracker(), test_misc()
    print("all passed")
