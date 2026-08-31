#!/usr/bin/env bash
# Downloads Sleep-EDF Expanded from PhysioNet into ./data/
#
# Full dataset: ~200 recordings (sleep-cassette + sleep-telemetry), ~8GB.
# Never commit the output of this script -- data/ is gitignored.
#
# Usage:
#   scripts/download_data.sh            # full dataset
#   scripts/download_data.sh --sample   # just 1 subject (2 files) for local dev/testing

set -euo pipefail

BASE_URL="https://physionet.org/files/sleep-edfx/1.0.0"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
mkdir -p "$DATA_DIR"

if [[ "${1:-}" == "--sample" ]]; then
    echo "Downloading a single sample subject (SC4001) into $DATA_DIR ..."
    mkdir -p "$DATA_DIR/sleep-cassette"
    wget -N -c -P "$DATA_DIR/sleep-cassette" \
        "$BASE_URL/sleep-cassette/SC4001E0-PSG.edf" \
        "$BASE_URL/sleep-cassette/SC4001EC-Hypnogram.edf"
    echo "Sample download complete."
elif command -v aria2c >/dev/null 2>&1; then
    # Preferred path. PhysioNet throttles bandwidth hard per TCP connection,
    # so a plain sequential `wget -r` over ~400 files can take hours. aria2c
    # opens multiple connections per file and downloads multiple files at
    # once, which is the difference between this taking ~15 minutes and
    # ~3+ hours for the same 8GB. On Colab: `!apt-get -qq install aria2` first.
    echo "Downloading full Sleep-EDF Expanded (~8GB) into $DATA_DIR via aria2c (parallel) ..."
    MANIFEST="$(mktemp)"
    URLS_FILE="$(mktemp)"
    curl -sS "$BASE_URL/SHA256SUMS.txt" -o "$MANIFEST"
    awk -v base="$BASE_URL" '
        $2 ~ /^sleep-cassette\// || $2 ~ /^sleep-telemetry\// {
            print base "/" $2
            print "  out=" $2
        }
    ' "$MANIFEST" > "$URLS_FILE"
    aria2c -x 8 -s 8 -j 8 -c -d "$DATA_DIR" -i "$URLS_FILE"
    rm -f "$MANIFEST" "$URLS_FILE"
    echo "Full download complete."
else
    echo "aria2c not found -- falling back to sequential wget (slow: can take"
    echo "hours on PhysioNet's throttled connection). Install aria2c for a"
    echo "much faster download: apt-get install aria2 (or 'brew install aria2')."
    echo "Downloading full Sleep-EDF Expanded (~8GB) into $DATA_DIR ..."
    wget -r -N -c -np -nH --cut-dirs=3 -P "$DATA_DIR" \
        --reject "index.html*" \
        "$BASE_URL/"
    echo "Full download complete."
fi
