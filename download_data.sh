#!/usr/bin/env bash
# Download the segmented data for the pub from the BioImage Archive (S-BIAD4378)
# into zarrs/ and jsonls/ at the top of this repository, unzip the zarr stores,
# and check that all 7 zarr stores and 7 JSONL files are present.
#
# Usage (from anywhere):   bash download_data.sh
# Needs: curl, jq, wget, unzip. About 125 GB of free disk space.
# Safe to rerun: interrupted downloads resume, and finished stores are not re-unzipped.

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

for tool in curl jq wget unzip; do
  command -v "$tool" >/dev/null 2>&1 || { echo "Error: '$tool' is not installed." >&2; exit 1; }
done

# 1. Look up the study's FTP location.
FTP_ROOT=$(curl -fsS "https://www.ebi.ac.uk/biostudies/api/v1/studies/${ACCESSION}/info" | jq -r .ftpLink)
if [[ -z "$FTP_ROOT" || "$FTP_ROOT" == "null" ]]; then
  echo "Error: could not find the FTP location for ${ACCESSION}." >&2
  exit 1
fi
echo "Downloading ${ACCESSION} from ${FTP_ROOT}/Files/ ..."

# 2. Download zarrs/, jsonls/ and file_list.tsv into this folder.
#    --cut-dirs drops the archive's path prefix so the folders land here directly.
CUT_DIRS=$(echo "${FTP_ROOT#ftp://ftp.ebi.ac.uk/}/Files" | tr '/' '\n' | wc -l | tr -d ' ')
wget -r -np -nH -c --cut-dirs="$CUT_DIRS" "${FTP_ROOT}/Files/"

# 3. Unzip each zarr store (skipped if that store is already unzipped).
echo "Unzipping zarr stores ..."
for id in "${IDS[@]}"; do
  store="zarrs/experiment_batches_${id}_features.zarr"
  zip="${store}.zip"
  if [[ -d "$store" ]]; then
    echo "  ${store}: already unzipped"
  elif [[ -f "$zip" ]]; then
    echo "  unzipping ${zip}"
    unzip -q "$zip" -d zarrs
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
  echo "The .zarr.zip files in zarrs/ are no longer needed and can be deleted to save about 110 GB."
else
  echo "Some files are missing. Rerun this script to resume the download." >&2
  exit 1
fi
