"""mentalese: speak one language, Discord hears another, in your own voice.

  uv run mentalese record        record your voice sample (once)
  uv run mentalese run --to ru   start translating; type to speak, /zh etc. to switch, /quit to stop
  uv run mentalese say --to zh "text"   hear your clone say something (translated) in your headphones
  uv run mentalese remove-mic    remove the virtual mic from PipeWire
"""

import argparse
import collections
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio, langid
from .appaudio import AppCapture
from .incoming import Incoming
from .langs import LANGS
from .mt import DEFAULT_STYLE

ROOT = Path(__file__).resolve().parent.parent
VOICES = ROOT / "voices"
SR = 16000
FRAME = 512  # 32 ms, what Silero VAD expects at 16 kHz
COLORS = ["\033[96m", "\033[93m", "\033[95m", "\033[92m", "\033[94m", "\033[91m"]  # one per person
DIM, RESET = "\033[2m", "\033[0m"
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")  # the voice can't say these

def voice_files(name: str) -> tuple[Path, Path]:
    """A voice is a 15 s recording plus its transcript: voices/<name>.wav and .txt"""
    return VOICES / f"{name}.wav", VOICES / f"{name}.txt"


def require_voice(name: str) -> tuple[Path, Path]:
    wav, txt = voice_files(name)
    if not wav.exists():
        hint = "" if name == "me" else f" --voice {name}"
        sys.exit(f"No voice called '{name}' yet. Record it with:  uv run mentalese record{hint}")
    return wav, txt


READ_ALOUD = (
    "Hey, it's me. I'm recording this so my voice can speak other languages. "
    "Honestly, I have no idea how it's going to sound, but let's find out. "
    "One, two, three, four, five. The quick brown fox jumps over the lazy dog."
)


def cmd_record(args):
    from .asr import ASR

    wav_path, txt_path = voice_files(args.voice)
    VOICES.mkdir(exist_ok=True)
    print(f"Recording voice '{args.voice}'. Read this out loud at a normal pace and volume,")
    print("or just talk naturally for 15 seconds in any language:\n")
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
    sf.write(wav_path, wav, 24000)
    print("transcribing your sample...")
    import librosa

    lang, text = ASR().transcribe(librosa.resample(wav, orig_sr=24000, target_sr=SR))
    txt_path.write_text(text)
    print(f"saved {wav_path.relative_to(ROOT)} ({lang}): {text}")


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

    voice_wav, voice_txt = require_voice(args.voice)
    if args.practice:  # hear yourself in your headphones only; no Discord, no subtitles
        args.monitor, args.no_subtitles, args.subtitles_from = True, True, None
    state = {"to": args.to, "muted": False, "playing_until": 0.0}

    if not args.practice:
        audio.ensure_virtual_mic()
    print("loading models...")
    from .asr import ASR
    from .mt import Translator
    from .tts import TTS

    asr, mt, tts = ASR(), Translator(style=args.style), TTS()
    voice_text = voice_txt.read_text()
    tts.set_voice(str(voice_wav), voice_text, lang=langid.detect(voice_text, fallback="en"),
                  keep_accent=args.keep_accent)
    # warm up every stage so the first real sentence isn't slow
    asr.transcribe(np.zeros(SR, np.float32))
    mt.translate("hello", "en", "ru")
    tts.warmup()
    torch.cuda.empty_cache()  # hand back what loading and warmup used but no longer need
    player = None if args.practice else audio.Player(24000, device=audio.SINK)
    monitor = audio.Player(24000) if args.monitor else None

    utterances: queue.Queue = queue.Queue()  # ("out"|"in", stream, t_end, audio)

    def listen():
        mic = audio.MicReader(SR, FRAME, device=args.input)
        seg = Segmenter(args.silence_ms)
        while True:
            utt = seg.feed(mic.read())
            # with --monitor on speakers, the mic hears the translation; don't translate it again
            echo = monitor and time.perf_counter() < state["playing_until"] + 0.3
            if utt is not None and not state["muted"] and not echo:
                utterances.put(("out", 0, time.perf_counter(), utt))

    # each call participant gets their own VAD, so people talking over each other stay separate
    segmenters: dict[int, Segmenter] = {}

    def on_frame(num, frame):
        seg = segmenters.get(num) or segmenters.setdefault(num, Segmenter(args.their_silence_ms))
        utt = seg.feed(frame)
        if utt is not None:
            utterances.put(("in", num, time.perf_counter(), utt))

    if args.subtitles_from:
        AppCapture(args.subtitles_from, on_frame, frame=FRAME)
    if not args.no_subtitles:
        Incoming(on_open=lambda num, user: segmenters.setdefault(num, Segmenter(args.their_silence_ms)),
                 on_frame=on_frame, on_close=lambda num: segmenters.pop(num, None), frame=FRAME)

    def commands():
        for line in sys.stdin:
            line = line.strip()
            cmd = line.lower()
            if not line:
                pass
            elif not cmd.startswith("/"):
                # typed text: spoken in your voice, in the target language, like a spoken sentence
                utterances.put(("typed", 0, time.perf_counter(), line))
                continue  # the result line prints the next prompt
            elif cmd[1:] in LANGS:
                state["to"] = cmd[1:]
                print(f"now translating into {LANGS[cmd[1:]][0]}")
            elif cmd in ("/m", "/mute"):
                state["muted"] = not state["muted"]
                print("mic muted (typing still works)" if state["muted"] else "mic unmuted")
            elif cmd in ("/q", "/quit", "/exit"):
                break
            else:
                print(f"commands: /{' /'.join(LANGS)}  /mute  /quit — anything else gets spoken")
            print("> ", end="", flush=True)
        utterances.put(None)

    threading.Thread(target=listen, daemon=True).start()
    threading.Thread(target=commands, daemon=True).start()

    if args.practice:
        print(f"\nReady (practice). Talk, and hear yourself in your headphones as '{args.voice}'.")
    else:
        print(f"\nReady. In Discord pick input device 'Mentalese-Mic' and turn OFF noise suppression.")
    others = "" if args.practice else f" Others → subtitles in {LANGS[args.me][0]}."
    print(f"You speak → {LANGS[state['to']][0]}.{others}\n"
          f"Type anything + Enter to say it in your voice. Commands: /{' /'.join(LANGS)}  /mute  /quit\n> ",
          end="", flush=True)

    ms = lambda a, b: f"{1000 * (b - a):.0f}"
    # GPU work stays on the main thread, where the TTS CUDA graphs were captured
    try:
        while (item := utterances.get()) is not None:
            kind, num, t_end, utt = item
            t_start = time.perf_counter()  # later than t_end if earlier sentences were still being processed
            if kind == "typed":
                src, text = langid.detect(utt, fallback=args.me), utt
            else:
                if len(utt) < SR * 0.3:
                    continue
                src, text = asr.transcribe(utt)
            t_asr = time.perf_counter()
            if not text:
                continue

            if kind == "in":
                if src == args.me:
                    continue  # you understand it already
                sub = mt.translate(text, src, args.me)
                color = COLORS[num % len(COLORS)]
                print(f"\r{color}▌ {sub}{RESET}  {DIM}({text}){RESET}\n> ", end="", flush=True)
                torch.cuda.empty_cache()
                continue

            to = state["to"]
            out = text if src == to else mt.translate(text, src, to)
            t_mt = time.perf_counter()
            t_tts = None
            for chunk, _ in tts.stream(EMOJI.sub("", out).strip() or out, to):
                t_tts = t_tts or time.perf_counter()
                if player:
                    player.play(chunk)
                if monitor:
                    monitor.play(chunk)
                    state["playing_until"] = max(state["playing_until"], time.perf_counter()) + len(chunk) / 24000
            t_tts = t_tts or time.perf_counter()
            torch.cuda.empty_cache()  # keeps VRAM ~0.8 GB lower; no measurable latency cost
            wait = f"wait {ms(t_end, t_start)} + " if t_start - t_end > 0.05 else ""
            asr_part = "typed" if kind == "typed" else f"asr {ms(t_start, t_asr)}"
            print(f"\r{DIM}you: {text}\n  → {out}   "
                  f"[{ms(t_end, t_tts)} ms: {wait}{asr_part} + mt {ms(t_asr, t_mt)} + voice {ms(t_mt, t_tts)}]"
                  f"{RESET}\n> ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    mt.close()


def cmd_say(args):
    voice_wav, voice_txt = require_voice(args.voice)
    from .mt import Translator
    from .tts import TTS

    mt, tts = Translator(style=args.style), TTS()
    voice_text = voice_txt.read_text()
    tts.set_voice(str(voice_wav), voice_text, lang=langid.detect(voice_text, fallback="en"),
                  keep_accent=args.keep_accent)
    text = " ".join(args.text)
    out = mt.translate(text, None, args.to) if args.to != "en" else text
    print(f"[{args.to}] {out}")
    player = audio.Player(24000)
    for chunk, _ in tts.stream(EMOJI.sub("", out).strip() or out, args.to):
        player.play(chunk)
    player.close()
    mt.close()


def main():
    p = argparse.ArgumentParser(prog="mentalese")
    sub = p.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="record a voice sample (yours by default)")
    rec.add_argument("--voice", default="me", help="name to save it as, e.g. mom")
    r = sub.add_parser("run", help="start translating")
    r.add_argument("--to", default="ru", choices=list(LANGS), help="language Discord hears")
    r.add_argument("--voice", default="me", help="which recorded voice to speak with")
    r.add_argument("--keep-accent", action="store_true",
                   help="other languages keep your sample's accent (closer to your voice, less native)")
    r.add_argument("--practice", action="store_true",
                   help="hear yourself translated in your headphones; no Discord, no subtitles")
    r.add_argument("--style", default=DEFAULT_STYLE,
                   help='tone of translations, in plain words; --style "" for neutral')
    r.add_argument("--me", default="en", choices=list(LANGS), help="language for subtitles of what others say")
    r.add_argument("--silence-ms", type=int, default=400,
                   help="how long a pause counts as the end of a sentence (lower = faster, may cut you off)")
    r.add_argument("--their-silence-ms", type=int, default=700,
                   help="same for other people; longer so subtitles get whole sentences, not fragments")
    r.add_argument("--no-subtitles", action="store_true", help="don't listen to the Vesktop plugin")
    r.add_argument("--subtitles-from", metavar="APP",
                   help="also subtitle an app's audio, e.g. Telegram (best for 1:1 calls)")
    r.add_argument("--monitor", action="store_true", help="also play the translation in your headphones")
    r.add_argument("--input", help=argparse.SUPPRESS)  # read from another PipeWire source (testing)
    s = sub.add_parser("say", help="hear your cloned voice say something (translated from English)")
    s.add_argument("--to", default="ru", choices=list(LANGS))
    s.add_argument("--style", default=DEFAULT_STYLE)
    s.add_argument("--voice", default="me")
    s.add_argument("--keep-accent", action="store_true")
    s.add_argument("text", nargs="+")
    sub.add_parser("voices", help="list recorded voices")
    sub.add_parser("remove-mic", help="remove the virtual mic")
    args = p.parse_args()
    commands = {
        "record": cmd_record, "run": cmd_run, "say": cmd_say,
        "voices": lambda a: print("\n".join(sorted(v.stem for v in VOICES.glob("*.wav"))) or "no voices yet"),
        "remove-mic": lambda a: audio.remove_virtual_mic(),
    }
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
