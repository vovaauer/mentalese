"""mentalese: speak one language, Discord hears another, in your own voice.

  uv run mentalese record        record your voice sample (once)
  uv run mentalese run --to ru   start translating; type a language code + Enter to switch
  uv run mentalese say --to zh "text"   hear your clone say something (translated) in your headphones
  uv run mentalese remove-mic    remove the virtual mic from PipeWire
"""

import argparse
import collections
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio
from .langs import LANGS

ROOT = Path(__file__).resolve().parent.parent
VOICE_WAV = ROOT / "voices/me.wav"
VOICE_TXT = ROOT / "voices/me.txt"
SR = 16000
FRAME = 512  # 32 ms, what Silero VAD expects at 16 kHz

READ_ALOUD = (
    "Hey, it's me. I'm recording this so my voice can speak other languages. "
    "Honestly, I have no idea how it's going to sound, but let's find out. "
    "One, two, three, four, five. The quick brown fox jumps over the lazy dog."
)


def cmd_record(args):
    from .asr import ASR

    VOICE_WAV.parent.mkdir(exist_ok=True)
    print("Read this out loud, at your normal pace and volume:\n")
    print(f"  {READ_ALOUD}\n")
    input("Press Enter to start recording (it stops by itself after 15 seconds)...")
    print("recording...")
    proc = subprocess.Popen(["parec", "--rate=24000", "--channels=1", "--format=float32le",
                             "--client-name=mentalese"], stdout=subprocess.PIPE)
    data = proc.stdout.read(24000 * 4 * 15)
    proc.terminate()
    wav = np.frombuffer(data, dtype=np.float32)
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    if peak < 0.02:
        sys.exit("That was nearly silent. Check your default mic in your sound settings and try again.")
    wav = wav / peak * 0.9
    sf.write(VOICE_WAV, wav, 24000)
    print("transcribing your sample...")
    import librosa

    lang, text = ASR().transcribe(librosa.resample(wav, orig_sr=24000, target_sr=SR))
    VOICE_TXT.write_text(text)
    print(f"saved {VOICE_WAV.relative_to(ROOT)} ({lang}): {text}")


class Segmenter:
    """Cuts the mic stream into utterances with Silero VAD."""

    def __init__(self, silence_ms: int):
        import torch
        from silero_vad import VADIterator, load_silero_vad

        torch.set_num_threads(1)
        self.vad = VADIterator(load_silero_vad(), threshold=0.5, sampling_rate=SR,
                               min_silence_duration_ms=silence_ms, speech_pad_ms=100)
        self.torch = torch
        self.preroll = collections.deque(maxlen=10)  # ~320 ms before speech starts
        self.buf = None

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        ev = self.vad(self.torch.from_numpy(frame.copy()))
        if self.buf is None:
            self.preroll.append(frame)
            if ev and "start" in ev:
                self.buf = list(self.preroll)
            return None
        self.buf.append(frame)
        if (ev and "end" in ev) or len(self.buf) * FRAME > SR * 15:
            utt, self.buf = np.concatenate(self.buf), None
            self.preroll.clear()
            if "end" not in (ev or {}):
                self.vad.reset_states()
            return utt
        return None


def cmd_run(args):
    import torch

    if not VOICE_WAV.exists():
        sys.exit("No voice sample yet. Run:  uv run mentalese record")
    state = {"to": args.to, "muted": False}

    audio.ensure_virtual_mic()
    print("loading models...")
    from .asr import ASR
    from .mt import Translator
    from .tts import TTS

    asr, mt, tts = ASR(), Translator(), TTS()
    tts.set_voice(str(VOICE_WAV), VOICE_TXT.read_text())
    # warm up every stage so the first real sentence isn't slow
    asr.transcribe(np.zeros(SR, np.float32))
    mt.translate("hello", "en", "ru")
    tts.warmup()
    player = audio.Player(24000, device=audio.SINK)
    monitor = audio.Player(24000) if args.monitor else None

    utterances: queue.Queue = queue.Queue()

    def listen():
        mic = audio.MicReader(SR, FRAME, device=args.input)
        seg = Segmenter(args.silence_ms)
        while True:
            utt = seg.feed(mic.read())
            if utt is not None and not state["muted"]:
                utterances.put((time.perf_counter(), utt))

    def commands():
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in LANGS:
                state["to"] = cmd
                print(f"now translating into {LANGS[cmd][0]}")
            elif cmd == "m":
                state["muted"] = not state["muted"]
                print("muted" if state["muted"] else "unmuted")
            elif cmd in ("q", "quit", "exit"):
                break
            print("> ", end="", flush=True)
        utterances.put(None)

    threading.Thread(target=listen, daemon=True).start()
    threading.Thread(target=commands, daemon=True).start()

    print(f"\nReady. In Discord pick input device 'Mentalese-Mic' and turn OFF noise suppression.")
    print(f"Translating into {LANGS[state['to']][0]}. Type a code + Enter to switch "
          f"({' '.join(LANGS)}), 'm' to mute/unmute, 'q' to quit.\n> ", end="", flush=True)

    # GPU work stays on the main thread, where the TTS CUDA graphs were captured
    try:
        while (item := utterances.get()) is not None:
            t_end, utt = item
            if len(utt) < SR * 0.3:
                continue
            src, text = asr.transcribe(utt)
            t_asr = time.perf_counter()
            if not text:
                continue
            to = state["to"]
            out = text if src == to else mt.translate(text, src, to)
            t_mt = time.perf_counter()
            t_tts = None
            for chunk, _ in tts.stream(out, to):
                t_tts = t_tts or time.perf_counter()
                player.play(chunk)
                if monitor:
                    monitor.play(chunk)
            t_tts = t_tts or time.perf_counter()
            ms = lambda a, b: f"{1000 * (b - a):.0f}"
            print(f"\n[{src or '?'}] {text}\n[{to}] {out}\n"
                  f"   asr {ms(t_end, t_asr)} + mt {ms(t_asr, t_mt)} + voice {ms(t_mt, t_tts)} ms"
                  f" = they hear you {ms(t_end, t_tts)} ms after the pause (pause = {args.silence_ms} ms)\n> ",
                  end="", flush=True)
    except KeyboardInterrupt:
        pass
    mt.close()


def cmd_say(args):
    if not VOICE_WAV.exists():
        sys.exit("No voice sample yet. Run:  uv run mentalese record")
    from .mt import Translator
    from .tts import TTS

    mt, tts = Translator(), TTS()
    tts.set_voice(str(VOICE_WAV), VOICE_TXT.read_text())
    text = " ".join(args.text)
    out = mt.translate(text, None, args.to) if args.to != "en" else text
    print(f"[{args.to}] {out}")
    player = audio.Player(24000)
    for chunk, _ in tts.stream(out, args.to):
        player.play(chunk)
    player.close()
    mt.close()


def main():
    p = argparse.ArgumentParser(prog="mentalese")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record", help="record your voice sample")
    r = sub.add_parser("run", help="start translating")
    r.add_argument("--to", default="ru", choices=list(LANGS), help="language Discord hears")
    r.add_argument("--silence-ms", type=int, default=400,
                   help="how long a pause counts as the end of a sentence (lower = faster, may cut you off)")
    r.add_argument("--monitor", action="store_true", help="also play the translation in your headphones")
    r.add_argument("--input", help=argparse.SUPPRESS)  # read from another PipeWire source (testing)
    s = sub.add_parser("say", help="hear your cloned voice say something (translated from English)")
    s.add_argument("--to", default="ru", choices=list(LANGS))
    s.add_argument("text", nargs="+")
    sub.add_parser("remove-mic", help="remove the virtual mic")
    args = p.parse_args()
    {"record": cmd_record, "run": cmd_run, "say": cmd_say, "remove-mic": lambda a: audio.remove_virtual_mic()}[args.cmd](args)


if __name__ == "__main__":
    main()
