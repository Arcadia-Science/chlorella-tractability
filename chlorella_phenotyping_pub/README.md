# Chlorella phenotyping pub: analysis code

This folder reproduces every quantitative result and data figure in the pub *Chlorella vulgaris phenotypes are structured and predictable across environmental conditions*, plus the supplemental analyses behind the methods.

| File | What it is |
|---|---|
| `chlorella_phenotyping_pub_analyses.ipynb` | The analysis, in pub order. All parameters are in one cell at the top. |
| `cvp_pub.py` | Every function the notebook calls (`import cvp_pub as cvpp`). The header maps each pub result to the functions that compute it. |
| `env.yml` | The exact conda environment used for the analysis (Apple Silicon Mac only). |
| `env_portable.yml` | The same package versions for any OS, without platform-specific builds. |
| `requirements.txt` | Unpinned Python dependencies for a plain `pip` install. |

## Running it

1. Data: the notebook expects `zarrs/` and `jsonls/` one level up (`DATA_DIR = ".."`). Point `DATA_DIR` elsewhere if needed. It uses the six mutant experiments (16/22/30 °C × early/late stationary) and the wild-type plate (`20251209_091846_235`).
2. Set up the Python environment. Pick one:
   * **Apple Silicon Mac, exact environment:** `conda env create -f env.yml`, then `conda activate hina1`.
   * **Any OS:** `conda env create -f env_portable.yml`, then `conda activate chlorella-pub`.
   * **pip only:** `pip install -r requirements.txt` (Python 3.11 or later; versions not pinned).
3. Open the notebook with that environment's kernel and run all cells.

Outputs:

* `plots/`: figures, mostly `chlorella_pub_fig_<name>.{png,pdf,svg}`, plus interactive `.html` heatmaps.
* `results/`: CSV tables behind the numbers quoted in the text, plus a cache of the field-of-view feature matrices (`feature_matrices_cache*.npz`). With the cache, later runs skip the slow JSONL read. Set `USE_CACHE = False` to rebuild it.

The supplemental sections (alpha scan, strain identity, interaction examples, batch effects, feature correlations) can each be switched off in `RUN_SUPPLEMENTAL`.

## Notes

* `cvp_pub.py` is self-contained. Its functions were copied verbatim from the working modules (`cvp.py`, `cvp_baseline_additions.py`, `cvp_departures.py`, `cvp_diversity.py`, `cvp_standardized.py`); only cross-module references were rewired. The segmentation pipeline (`arcadia_microscopy_tools`) is not needed here.
* Feature matrices have one row per (well, field of view), laid out FoV-major: `row = well + fov * 96`.
