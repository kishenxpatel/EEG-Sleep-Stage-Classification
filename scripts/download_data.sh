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
else
    echo "Downloading full Sleep-EDF Expanded (~8GB) into $DATA_DIR ..."
    wget -r -N -c -np -nH --cut-dirs=3 -P "$DATA_DIR" \
        --reject "index.html*" \
        "$BASE_URL/"
    echo "Full download complete."
fi
