#!/usr/bin/env bash
# CI: uploads this runner's installers to the GitHub release and writes the auto-update entry for
# its platform (build/updater-*.json, merged into latest*.json by the workflow's last job).
#   tools/release-upload.sh v0.2.0 [-nvidia]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$1 FLAVOR=${2:-}
REPO=${GITHUB_REPOSITORY:-vovaauer/mentalese}
B=target/release/bundle
case "$(uname -s)" in
    Linux) PLATFORM=linux-x86_64 ;;
    Darwin) PLATFORM=darwin-aarch64 ;;
    *) PLATFORM=windows-x86_64 ;;
esac

mkdir -p build/upload
shopt -s nullglob
for f in $B/deb/*.deb $B/rpm/*.rpm $B/appimage/*.AppImage $B/nsis/*-setup.exe $B/dmg/*.dmg $B/macos/*.app.tar.gz; do
    name=$(basename "$f")
    # Mentalese_0.2.0_amd64.deb -> Mentalese_0.2.0_amd64-nvidia.deb
    [ -n "$FLAVOR" ] && name=$(echo "$name" | sed -E "s/(\.(deb|rpm|AppImage|exe|dmg|app\.tar\.gz))$/$FLAVOR\1/")
    cp "$f" "build/upload/$name"
    [ -f "$f.sig" ] && cp "$f.sig" "build/upload/$name.sig"
done
gh release upload "$TAG" build/upload/* --clobber --repo "$REPO"

# the file the updater installs: the AppImage, the NSIS installer, or the .app archive
upd=$(ls build/upload/*.AppImage build/upload/*-setup*.exe build/upload/*.app.tar.gz 2>/dev/null | head -1 || true)
if [ -n "$upd" ] && [ -f "$upd.sig" ]; then
    printf '{"platform":"%s","flavor":"%s","signature":"%s","url":"https://github.com/%s/releases/download/%s/%s"}\n' \
        "$PLATFORM" "$FLAVOR" "$(cat "$upd.sig")" "$REPO" "$TAG" "$(basename "$upd")" > "build/updater-$PLATFORM$FLAVOR.json"
fi
