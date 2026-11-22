"""Offline voice commands with Vosk. Audio never leaves the computer."""
import json
import queue
import re
import threading
import time

ARM_SECONDS = 6  # how long the wake word alone keeps listening for a command


def clean(phrase):
    return " ".join(re.sub(r"[^a-z' ]", " ", str(phrase).lower()).split())


class VoiceListener(threading.Thread):
    def __init__(self, model_dir, store, on_command):
        super().__init__(daemon=True)
        self.model_dir = model_dir
        self.store = store
        self.on_command = on_command  # (voice_entry) -> None
        self.model = None
        self.armed_until = 0.0
        self.status = {"state": "off", "error": "", "heard": "", "heard_at": 0, "unknown": []}

    def _signature(self):
        cfg = self.store.cfg
        s = cfg["settings"]
        return json.dumps([s["voice_enabled"], s["wake_word"],
                           [(v["phrase"], v.get("enabled", True)) for v in cfg["voice"]]])

    def run(self):
        try:
            import sounddevice as sd
            import vosk
            vosk.SetLogLevel(-1)
            self.status["state"] = "loading"
            self.model = vosk.Model(str(self.model_dir))
        except Exception as e:
            self.status.update(state="error", error=f"Voice engine failed to load: {e}")
            return
        while True:
            if not self.store.cfg["settings"]["voice_enabled"]:
                self.status.update(state="off", error="")
                time.sleep(0.5)
                continue
            try:
                self._listen(sd, vosk)
            except Exception as e:
                self.status.update(state="error", error=f"Microphone problem: {e}")
                time.sleep(3)

    def _known(self, phrase):
        missing = [w for w in phrase.split() if self.model.vosk_model_find_word(w) < 0]
        return missing

    def _listen(self, sd, vosk):
        sig = self._signature()
        cfg = self.store.cfg
        wake = clean(cfg["settings"]["wake_word"])
        unknown = []
        if wake and self._known(wake):
            unknown += self._known(wake)
            wake = ""
        phrases = {}
        for entry in cfg["voice"]:
            p = clean(entry["phrase"])
            if not p or not entry.get("enabled", True):
                continue
            missing = self._known(p)
            if missing:
                unknown += missing
            else:
                phrases[p] = entry
        self.status["unknown"] = sorted(set(unknown))
        if not phrases:
            self.status.update(state="listening", error="")
            while self._signature() == sig:
                time.sleep(0.5)
            return

        grammar = list(phrases) + ["[unk]"]
        if wake:
            grammar += [wake] + [f"{wake} {p}" for p in phrases]
        rec = vosk.KaldiRecognizer(self.model, 16000, json.dumps(grammar))
        audio = queue.Queue()

        def on_audio(indata, frames, time_info, status):
            audio.put(bytes(indata))

        # Opening the microphone waits here until macOS's permission prompt is answered.
        self.status.update(state="mic", error="")
        with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype="int16",
                               channels=1, callback=on_audio):
            self.status.update(state="listening", error="")
            while self._signature() == sig:
                try:
                    data = audio.get(timeout=0.5)
                except queue.Empty:
                    continue
                if rec.AcceptWaveform(data):
                    text = json.loads(rec.Result()).get("text", "")
                    self._handle(text, phrases, wake)

    def _handle(self, text, phrases, wake):
        words = [w for w in text.split() if w != "[unk]"]
        if not words:
            return
        said = " ".join(words)
        now = time.time()
        self.status.update(heard=said, heard_at=now)
        if wake:
            wake_words = wake.split()
            if words[:len(wake_words)] == wake_words:
                said = " ".join(words[len(wake_words):])
                if not said:
                    self.armed_until = now + ARM_SECONDS
                    return
            elif now > self.armed_until:
                return
        entry = phrases.get(said)
        if entry:
            self.armed_until = 0.0
            self.on_command(entry)

    @property
    def armed(self):
        return time.time() < self.armed_until
