"""End-to-end test without a real mic or Discord: plays clips into a fake mic, records what
Mentalese-Mic outputs, prints mentalese's own timing lines. Needs out/bench_*.wav (bench.py).
Anything recording Mentalese-Mic hears the test clips: don't run it while you're in a Discord call.

uv run python tools/e2e.py [extra mentalese run args...]
"""

import re
import signal
import subprocess
import sys
import time

import numpy as np

TEST_SINK = "mentalese_test"
LOG = "out/e2e.log"


def main():
    mod = subprocess.run(["pactl", "load-module", "module-null-sink", f"sink_name={TEST_SINK}"],
                         capture_output=True, text=True, check=True).stdout.strip()
    try:
        log = open(LOG, "w")
        app = subprocess.Popen([".venv/bin/mentalese", "run", "--no-subtitles", "--input", f"{TEST_SINK}.monitor",
                                *sys.argv[1:]], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        while "Ready" not in (txt := open(LOG).read()):
            if "Traceback" in txt or app.poll() is not None:
                sys.exit(txt[-1500:])
            time.sleep(1)
        rec = subprocess.Popen(["parec", "--device=mentalese_mic", "--rate=24000", "--channels=1",
                                "--format=float32le"], stdout=subprocess.PIPE)
        for clip in ["bench_en", "bench_ru", "bench_en"]:
            subprocess.run(["paplay", f"--device={TEST_SINK}", f"out/{clip}.wav"])
            time.sleep(3)
        time.sleep(3)
        rec.terminate()
        a = np.frombuffer(rec.stdout.read(), dtype=np.float32)
        app.send_signal(signal.SIGTERM)
        app.wait()
        speech = (np.abs(a[: len(a) // 2400 * 2400].reshape(-1, 2400)).max(1) > 0.02).sum() / 10
        out = re.sub(r"\x1b\[[0-9;]*m", "", open(LOG).read())
        for line in out.splitlines():
            if line.startswith(("> you:", "you:")) or "→" in line:
                print(line.lstrip("> "))
        print(f"Mentalese-Mic got {speech:.1f}s of speech")
    finally:
        subprocess.run(["pactl", "unload-module", mod])


if __name__ == "__main__":
    main()
