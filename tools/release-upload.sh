#!/usr/bin/env bash
# CI: uploads this runner's installers to the GitHub release and writes the auto-update entries for
# them (build/updater-*.json, merged into latest*.json by the workflow's last job).
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

# Auto-update entries. The updater installs the same format the app was installed from (a .deb
# updates through dpkg, an AppImage replaces itself), looking for "<platform>-<format>" first,
# then plain "<platform>".
out="build/updater-$PLATFORM$FLAVOR.json"
: > "$out"
entry() { # manifest key, file
    [ -f "$2.sig" ] || return 0
    printf '{"platform":"%s","flavor":"%s","signature":"%s","url":"https://github.com/%s/releases/download/%s/%s"}\n' \
        "$1" "$FLAVOR" "$(cat "$2.sig")" "$REPO" "$TAG" "$(basename "$2")" >> "$out"
}
for f in build/upload/*; do
    case $f in
        *.AppImage) entry "$PLATFORM-appimage" "$f"; entry "$PLATFORM" "$f" ;;
        *.deb) entry "$PLATFORM-deb" "$f" ;;
        *.rpm) entry "$PLATFORM-rpm" "$f" ;;
        *-setup*.exe) entry "$PLATFORM-nsis" "$f"; entry "$PLATFORM" "$f" ;;
        *.app.tar.gz) entry "$PLATFORM-app" "$f"; entry "$PLATFORM" "$f" ;;
    esac
done
[ -s "$out" ] || rm "$out"
