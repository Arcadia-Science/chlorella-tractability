#!/usr/bin/env bash
# Download the segmented data for the pub from the BioImage Archive (S-BIAD4378)
# into zarrs/ and jsonls/ at the top of this repository, unzip the zarr stores,
# and check that all 7 zarr stores and 7 JSONL files are present.
#
# Usage (from anywhere):   bash download_data.sh
# Needs only curl and unzip, which come with macOS and Linux.
# Disk space: about 240 GB while downloading (zips + unzipped stores);
# about 125 GB once the .zarr.zip files are deleted.
# Safe to rerun: partial downloads resume, finished files are skipped.

set -euo pipefail

ACCESSION="S-BIAD4378"
IDS=(
  20260105_203447_582   # 16 °C, early stationary
  20260105_125955_886   # 22 °C, early stationary
  20260105_161353_802   # 30 °C, early stationary
  20260112_161051_665   # 16 °C, late stationary
  20260112_112101_320   # 22 °C, late stationary
  20260112_135035_609   # 30 °C, late stationary
  20251209_091846_235   # wild type, 22 °C, early stationary
)

# Always work from the top of the repository (where this script lives).
cd "$(dirname "$0")"

for tool in curl unzip; do
  command -v "$tool" >/dev/null 2>&1 || { echo "Error: '$tool' is not installed." >&2; exit 1; }
done

# 1. Look up the study's file location (read from the archive's JSON without needing jq).
INFO=$(curl -fsS "https://www.ebi.ac.uk/biostudies/api/v1/studies/${ACCESSION}/info")
FTP_ROOT=$(printf '%s' "$INFO" | sed -n 's/.*"ftpLink"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
if [[ -z "$FTP_ROOT" ]]; then
  echo "Error: could not find the file location for ${ACCESSION}." >&2
  exit 1
fi
# The same files are served over HTTPS, which works on networks that block FTP.
HTTPS_ROOT="${FTP_ROOT/#ftp:\/\//https://}"

# Download one file with resume; try HTTPS first, then FTP.
fetch() {  # fetch <path under Files/> <local path>
  local rel="$1" out="$2"
  if [[ -f "$out" && ! -f "$out.part" ]]; then
    echo "  $out: already downloaded"
    return 0
  fi
  echo "  downloading $out"
  touch "$out.part"
  if curl -fL --retry 5 --retry-delay 10 -C - -o "$out.tmp" "${HTTPS_ROOT}/Files/${rel}" \
     || curl -fL --retry 5 --retry-delay 10 -C - -o "$out.tmp" "${FTP_ROOT}/Files/${rel}"; then
    mv "$out.tmp" "$out"
    rm -f "$out.part"
  else
    echo "Error: could not download ${rel}. Rerun this script to resume." >&2
    exit 1
  fi
}

# 2. Download the 7 JSONL files and 7 zipped zarr stores.
mkdir -p zarrs jsonls
echo "Downloading ${ACCESSION} into $(pwd) ..."
for id in "${IDS[@]}"; do
  name="experiment_batches_${id}_features"
  fetch "jsonls/${name}.jsonl" "jsonls/${name}.jsonl"
  [[ -d "zarrs/${name}.zarr" ]] || fetch "zarrs/${name}.zarr.zip" "zarrs/${name}.zarr.zip"
done

# 3. Unzip each zarr store (skipped if that store is already unzipped).
echo "Unzipping zarr stores ..."
for id in "${IDS[@]}"; do
  store="zarrs/experiment_batches_${id}_features.zarr"
  if [[ -d "$store" ]]; then
    echo "  ${store}: already unzipped"
  else
    echo "  unzipping ${store}.zip"
    unzip -q "${store}.zip" -d zarrs
  fi
done

# 4. Check that everything the notebook needs is here.
missing=0
for id in "${IDS[@]}"; do
  [[ -f "zarrs/experiment_batches_${id}_features.zarr/zarr.json" ]] \
    || { echo "MISSING: zarrs/experiment_batches_${id}_features.zarr"; missing=1; }
  [[ -f "jsonls/experiment_batches_${id}_features.jsonl" ]] \
    || { echo "MISSING: jsonls/experiment_batches_${id}_features.jsonl"; missing=1; }
done

if [[ $missing -eq 0 ]]; then
  echo "All 7 zarr stores and 7 JSONL files are in place."
  echo "The .zarr.zip files in zarrs/ are no longer needed; delete them to free about 110 GB:"
  echo "  rm zarrs/*.zarr.zip"
else
  echo "Some files are missing. Rerun this script to resume the download." >&2
  exit 1
fi
