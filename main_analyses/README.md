# Chlorella phenotyping pub: analysis code

This folder reproduces every quantitative result and data figure in the pub ["Label-free imaging of *Chlorella vulgaris* captures structured phenotypes"](https://thestacks.org/publications/result-chlorella-tractability), plus the supplemental analyses behind the methods.

| File | What it is |
|---|---|
| `chlorella_phenotyping_pub_analyses.ipynb` | The analysis, in pub order. All parameters are in one cell at the top. |
| `cvp_pub.py` | Every function the notebook calls (`import cvp_pub as cvpp`). The header maps each pub result to the functions that compute it. |
| `env.yml` | The exact conda environment used for the analysis (Apple Silicon Mac only). |
| `env_portable.yml` | The same package versions for any OS, without platform-specific builds. |
| `requirements.txt` | Unpinned Python dependencies for a plain `pip` install. |

## Requirements

* About 130 GB of free disk space for the data (111 GB of zarr image stores + 14 GB of JSONL feature files)
* [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda) and Git
* Python 3.11 or later (the conda environments install 3.13)

## 1. Get the code

```bash
git clone https://github.com/Arcadia-Science/chlorella-tractability.git
cd chlorella-tractability
```

## 2. Set up the Python environment

Pick one:

* **Any OS (recommended):**
  ```bash
  conda env create -f main_analyses/env_portable.yml
  conda activate chlorella-pub
  ```
* **Apple Silicon Mac, the exact environment used for the pub:**
  ```bash
  conda env create -f main_analyses/env.yml
  conda activate hina1
  ```

Then make the environment available as a Jupyter kernel:

```bash
python -m ipykernel install --user --name chlorella-pub --display-name "Python (chlorella-pub)"
```

## 3. Download the data

The data are in the BioImage Archive under accession [**S-BIAD4378**](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD4378).
(about 124 GB: seven zipped Zarr v3 stores of about 16 GB each and seven JSONL feature files of 1.3–3.4 GB each)

You need 7 experiments, each as one `.zarr` folder plus one `.jsonl` file:

| Condition | Experiment ID | zarr | jsonl |
|---|---|---|---|
| 16 °C, early stationary | `20260105_203447_582` | 16 GB | 1.6 GB |
| 22 °C, early stationary | `20260105_125955_886` | 16 GB | 1.9 GB |
| 30 °C, early stationary | `20260105_161353_802` | 16 GB | 1.8 GB |
| 16 °C, late stationary | `20260112_161051_665` | 15 GB | 1.3 GB |
| 22 °C, late stationary | `20260112_112101_320` | 16 GB | 1.8 GB |
| 30 °C, late stationary | `20260112_135035_609` | 16 GB | 2.1 GB |
| Wild type, 22 °C, early stationary | `20251209_091846_235` | 16 GB | 3.3 GB |

Files are named `experiment_batches_<ID>_features.zarr` and `experiment_batches_<ID>_features.jsonl`.

Create two folders at the top of the repository:

```bash
mkdir zarrs jsonls
```

### Everything (command line)

The quickest way is the script at the top of the repository. It downloads everything below, unzips the zarr stores, and checks that all 14 items are in place:

```bash
bash download_data.sh
```

It needs `curl`, `jq`, `wget`, and `unzip`, and is safe to rerun: interrupted downloads resume. To do the same steps by hand (no account is needed):

```bash
# Look up the study's FTP location
FTP_ROOT=$(curl -s https://www.ebi.ac.uk/biostudies/api/v1/studies/S-BIAD4378/info | jq -r .ftpLink)

# Download zarrs/, jsonls/, and file_list.tsv into the current folder.
# Run this from the top of the repository, so the data land where the notebook looks.
# (-c resumes interrupted downloads; rerun the same command to continue)
wget -r -np -nH -c --cut-dirs=$(echo "${FTP_ROOT#ftp://ftp.ebi.ac.uk/}/Files" | tr '/' '\n' | wc -l) \
  "$FTP_ROOT/Files/"
```

### A single file

```bash
wget -c "$FTP_ROOT/Files/jsonls/experiment_batches_20251209_091846_235_features.jsonl"
```

### Unpacking

Each `.zarr.zip` unzips to a Zarr v3 store:

```bash
cd zarrs
for f in *.zarr.zip; do unzip -q "$f"; done
cd ..
```

The analysis notebook expects `zarrs/` and `jsonls/` in the folder that
`DATA_DIR` points to.

### Other options

- **Browser:** open the study page and use "Download all files," or select
  individual files.
- **Globus or Aspera:** faster for the full dataset; see the BioImage Archive download guide (https://www.ebi.ac.uk/bioimage-archive/help-download).

Put each `.zarr` folder in `zarrs/` and each `.jsonl` file in `jsonls/`. Keep the original file names; the notebook finds the files by name.

When you're done, the repository should look like this:

```
chlorella-tractability/
├── zarrs/
│   ├── experiment_batches_20251209_091846_235_features.zarr/
│   ├── experiment_batches_20260105_125955_886_features.zarr/
│   └── ... (7 .zarr folders in total)
├── jsonls/
│   ├── experiment_batches_20251209_091846_235_features.jsonl
│   ├── experiment_batches_20260105_125955_886_features.jsonl
│   └── ... (7 .jsonl files in total)
└── main_analyses/
    ├── chlorella_phenotyping_pub_analyses.ipynb
    ├── cvp_pub.py
    └── ...
```

Check that all 14 items are there:

```bash
ls zarrs | wc -l    # 7
ls jsonls | wc -l   # 7
```

To keep the data somewhere else (for example on an external drive), set `DATA_DIR` in the notebook's Parameters cell to the folder that contains `zarrs/` and `jsonls/`.

## 4. Run the notebook

```bash
cd main_analyses
jupyter lab chlorella_phenotyping_pub_analyses.ipynb
```

Select the **Python (chlorella-pub)** kernel, check the Parameters cell, and run all cells (*Run → Run All Cells*). The notebook creates `plots/` and `results/` next to itself.

* **First run:** reading the feature files takes a while, and the full run takes ~15 min. The feature matrices are then cached in `results/feature_matrices_cache*.npz`, so later runs skip the slow read. Set `USE_CACHE = False` to rebuild the cache.
* **Shorter run:** each supplemental section (alpha scan, strain identity, interaction examples, batch effects, feature correlations) can be switched off in `RUN_SUPPLEMENTAL`.

## Outputs

* `plots/`: figures, mostly `chlorella_pub_fig_<name>.{png,pdf,svg}`, plus interactive `.html` heatmaps.
* `results/`: CSV tables behind the numbers quoted in the text, including the values plotted in each interactive heatmap.

## Troubleshooting

* **`FileNotFoundError` or `GroupNotFoundError` when loading data:** the notebook can't find the data. Check the folder layout above and `DATA_DIR`.
* **Zarr errors on a working path:** the stores are zarr v3, which needs `zarr>=3` and Python 3.11 or later. Use one of the conda environments above.
* **"The fonts were not found" warning:** this is harmless. Figures fall back to default fonts. For the Arcadia fonts, install [Atkinson Hyperlegible Next](https://fonts.google.com/specimen/Atkinson+Hyperlegible+Next) and [Atkinson Hyperlegible Mono](https://fonts.google.com/specimen/Atkinson+Hyperlegible+Mono).

## Notes

* `cvp_pub.py` is self-contained. Its functions were copied verbatim from the working analysis modules; only cross-module references were rewired. The segmentation pipeline is not needed here.
* Feature matrices have one row per (well, field of view), laid out FoV-major: `row = well + fov * 96`.
