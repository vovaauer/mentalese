#!/usr/bin/env bash
# Builds Vencord with the mentalese plugin. Point Vesktop's "Vencord Location" at vendor/Vencord/dist.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d vendor/Vencord ]; then
    git -C vendor/Vencord pull --ff-only  # pick up the latest official Vencord
else
    git clone --depth 1 https://github.com/Vendicated/Vencord vendor/Vencord
fi
mkdir -p vendor/Vencord/src/userplugins
rm -rf vendor/Vencord/src/userplugins/mentalese
cp -r vencord/mentalese vendor/Vencord/src/userplugins/mentalese
cd vendor/Vencord
PNPM="npx -y pnpm@$(node -p "require('./package.json').packageManager.split('@')[1].split('+')[0]")"
$PNPM install --frozen-lockfile
$PNPM build
echo '{}' > dist/package.json  # Vesktop rejects a Vencord folder without one
echo "built: $(pwd)/dist"
