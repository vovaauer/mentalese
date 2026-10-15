#!/usr/bin/env bash
# Points Vesktop at our Vencord build and enables the Mentalese plugin.
# Vesktop rewrites its config while running, so this waits for it to quit, edits, then relaunches it.
set -euo pipefail
DIST="$(cd "$(dirname "$0")/.." && pwd)/vendor/Vencord/dist"
CFG="$HOME/.config/vesktop"
[ -f "$DIST/package.json" ] || { echo "build first: vencord/build.sh"; exit 1; }

if pgrep -x vesktop >/dev/null; then
    echo "waiting for Vesktop to quit (right-click its tray icon → Quit)..."
    while pgrep -x vesktop >/dev/null; do sleep 1; done
fi

cp "$CFG/state.json" "$CFG/state.json.bak-mentalese"
cp "$CFG/settings/settings.json" "$CFG/settings/settings.json.bak-mentalese"
python3 - "$CFG" "$DIST" <<'PY'
import json, sys
cfg, dist = sys.argv[1], sys.argv[2]
p = f"{cfg}/state.json"; d = json.load(open(p)); d["vencordDir"] = dist
json.dump(d, open(p, "w"), indent=4)
p = f"{cfg}/settings/settings.json"; d = json.load(open(p))
d.setdefault("plugins", {}).setdefault("Mentalese", {})["enabled"] = True
json.dump(d, open(p, "w"), indent=4)
PY
echo "Vesktop now uses $DIST with Mentalese enabled; starting Vesktop"
setsid -f vesktop >/dev/null 2>&1
