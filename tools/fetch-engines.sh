#!/usr/bin/env bash
# Gets the engine sources into vendor/ and patches them.
#   tools/fetch-engines.sh           the newest versions (and records them in tools/engines.lock)
#   tools/fetch-engines.sh --locked  exactly the versions in tools/engines.lock (CI, releases)
set -euo pipefail
cd "$(dirname "$0")/.."
LOCKED=0
[ "${1:-}" = --locked ] && LOCKED=1

while read -r name url commit; do
    case $name in ''|'#'*) continue ;; esac
    dir=vendor/$name
    if [ ! -d "$dir/.git" ]; then
        git clone --filter=blob:none "$url" "$dir"
    else
        git -C "$dir" checkout -- . # drop our patch so updating is clean
        git -C "$dir" fetch --quiet origin
    fi
    if [ $LOCKED = 1 ]; then
        git -C "$dir" checkout --quiet "$commit"
    else
        git -C "$dir" checkout --quiet origin/HEAD
    fi
    git -C "$dir" submodule update --init --recursive --quiet
    echo "$name: $(git -C "$dir" log -1 --format='%h %cs %s')"
done < tools/engines.lock

# The voice engine's talker attention cache holds 1024 positions instead of 4096: 670 MB less
# VRAM. mentalese splits long text so one request never needs more (crates/core/src/tts.rs).
f=vendor/qwentts.cpp/src/pipeline-tts.cpp
sed -i.orig 's/pt->talker.head_dim, 4096, pt->max_batch/pt->talker.head_dim, 1024, pt->max_batch/' $f && rm -f $f.orig
grep -q 'pt->talker.head_dim, 1024, pt->max_batch' $f \
    || { echo "qwentts.cpp changed: the cache patch in tools/fetch-engines.sh no longer applies" >&2; exit 1; }
grep -q '#define QT_ABI_VERSION 6$' vendor/qwentts.cpp/src/qwen.h \
    || { echo "qwentts.cpp's ABI changed: review engines/voice/mx_voice.cpp against src/qwen.h" >&2; exit 1; }

if [ $LOCKED = 0 ]; then
    tmp=$(mktemp)
    while read -r name url commit; do
        case $name in ''|'#'*) echo "$name $url $commit" | sed 's/ *$//' ;; *) echo "$name $url $(git -C vendor/$name rev-parse HEAD)" ;; esac
    done < tools/engines.lock > "$tmp"
    mv "$tmp" tools/engines.lock
fi
