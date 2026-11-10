# Segmentation: ND2 → Zarr + JSONL

This folder turns raw microscope files (`.nd2`) into the segmented data used by the main analysis: one `zarr` image store and one `jsonl` feature file per experiment. You only need it to segment new data. To reproduce the pub, download the already-segmented data instead (see [`../main_analyses/README.md`](../main_analyses/README.md)).

| File | What it is |
|---|---|
| `cvseg.py` | The segmentation and streaming-save code. |
| `segment_chlorella_strains.ipynb` | Segments the 7 experiments used in the pub. |
| `segmentation_2_zarr_example.ipynb` | A minimal example that segments one field of view of one experiment. Start here. |

## Why we stream to zarr

Our original segmentation pipeline loaded an entire experiment into memory, segmented it, and saved the results at the end. At Arcadia we run that on GPU machines with plenty of memory. On a laptop it fails: one experiment here is 96 wells × 9 fields of view × 2 channels of 1024 × 1024 images, plus every derived image and per-cell table. That's far more than fits in RAM, and the Jupyter kernel crashes partway through.

Most researchers don't have a GPU cluster, so we changed the pipeline to work in small pieces and write each piece to disk as soon as it's done:

* **One field of view at a time.** Each "batch" is one field-of-view position across all 96 wells. Batches are loaded lazily: an ND2 file is only opened when its batch is processed.
* **One channel at a time.** Bright-field and Cy5 are preprocessed, segmented, and saved separately.
* **Write, then free.** After each channel is saved, its images and segmentation objects are dropped from memory before the next one starts.

Memory use therefore depends on the size of one batch, not of the whole experiment. In our runs it stayed at a few GB (about 3–6 GB), so you can segment arbitrarily large ND2 collections on an ordinary computer, without a GPU. Cellpose runs on the CPU, which is slower but works.

## Background: ND2 vs. zarr

**ND2** is the proprietary file format written by Nikon's NIS-Elements acquisition software. In this dataset each ND2 file holds one field of view of one well, for example `WellH01_PointH01_0015_ChannelCy5 BPDIC_Seq1535.nd2` (well H01, field of view 15). Inside it are both channels, Cy5 (`Cy5 BP`) and bright-field (`DIC`), as a stack of raw 16-bit images of shape `2 × 1024 × 1024`, plus acquisition metadata (channel names, pixel size, objective, timestamps). ND2 works well for acquisition, but less well for analysis:

* An experiment is thousands of separate files (96 wells × 9 fields of view = 864 per experiment).
* Reading needs a specialized library, and each file is read as a whole.
* There's nowhere to put derived data like processed images or segmentation masks.

**Zarr** is an open format for large N-dimensional arrays. A zarr "file" is a folder: small JSON files describe each array (shape, data type, compression), and the data are split into compressed chunks stored as separate files. You can read a single image without touching the rest, append data as you go, and keep many related arrays (raw, processed, masks) together in one store. Zarr is readable from Python and many other languages.

```
ND2 input: one folder per experiment              Zarr + JSONL output: one store per experiment

<experiment_folder>/                              experiment_batches_<ID>_features.zarr/
├── Well…_Point…_0000_….nd2  ─┐                    ├── raw_intensities/{BF,CY5}            [FoV, well, 1024, 1024] uint16
├── Well…_Point…_0001_….nd2   │ 96 wells × 9 FoVs ├── processed_intensities/{BF,CY5}      [FoV, well, 1024, 1024] float32
├── …                         │ = 864 files       └── segmentation/
└── Well…_Point…_0008_….nd2  ─┘                        ├── label_image/{BF,CY5}           [FoV, well, 1024, 1024] int64
    each: 2 channels × 1024 × 1024, uint16              └── mask_image/{BF,CY5}            [FoV, well, 1024, 1024] int64
                                                  experiment_batches_<ID>_features.jsonl
                                                       one line per (FoV, well, channel):
                                                       cell outlines + per-cell properties
```

Every array is chunked one image per chunk and compressed (zstd), so reading one well's image reads one chunk. The store's attributes record `n_batches` (fields of view), `n_images_per_batch` (wells), and `channels`.

**Why a separate JSONL file?** Each image has a different number of cells, and each cell outline has a different length. That doesn't fit a rectangular array, so outlines and per-cell properties (area, perimeter, eccentricity, Hu moments, etc.) go in a JSON Lines file. Each line is one image in one channel:

```json
{"batch_index": 0, "image_index": 12, "channel": "BF", "cell_outlines": [[[367, 2], [366, 3], ...], ...], "cell_properties": {...}}
```

`batch_index` is the field of view and `image_index` is the well.

## Requirements

* **Python environment.** Segmentation needs [arcadia-microscopy-tools](https://github.com/Arcadia-Science/arcadia-microscopy-tools) at the exact commit we used. Later versions changed the API and won't work with `cvseg.py`. In a fresh environment:
  ```bash
  conda create -n chlorella-seg python=3.12
  conda activate chlorella-seg
  pip install "arcadia-microscopy-tools @ git+https://github.com/Arcadia-Science/arcadia-microscopy-tools.git@e541caa8fdb685507f115fbd3e3b236ddb0318d9" "zarr>=3" psutil natsort jupyterlab ipykernel
  ```
  This also installs Cellpose and PyTorch. On an Apple Silicon Mac you can instead use the full environment in [`../main_analyses/env.yml`](../main_analyses/env.yml).
* **Internet on the first run.** Cellpose downloads its model weights the first time it runs.
* **Disk space.** About 10 GB per 96-well, 9-FoV, 2-channel experiment.
* **No GPU needed.**

## How to segment ND2 files

### 1. Organize the ND2 files

Put all ND2 files for one experiment (one plate in one condition) in one folder. Each file name must contain its field-of-view number as a 4-digit, zero-padded string, which our files have after the well name: `WellH01_PointH01_0015_…` is field of view 15 of well H01. Files are grouped into batches by matching that string, and sorted naturally within each batch, so the wells must sort in the same order in every field of view. With `n_ims_w = 9`, only fields of view `0000`–`0008` are used, even if more were acquired. Our files also end in a 4-digit sequence number (`Seq1535`), so make sure that number can't also match a field-of-view string.

### 2. Try one field of view first

Open `segmentation_2_zarr_example.ipynb` and set:

```python
n_fovs_2_segment = 1                                   # fields of view per well to segment
data_loc_<ID> = "/path/to/your/nd2_folder"             # folder from step 1
model_ran_<ID> = False                                 # True skips segmentation
```

Run it. It writes a small `.zarr` and `.jsonl` into this folder and prints memory use (`[MEM] …`) at each step. Use it to check that your files are found and memory stays within your machine's limits.

### 3. Segment a full experiment

The core is two calls:

```python
import cvseg

model_batches = cvseg.prep_new_segmentation(
    data_loc="/path/to/your/nd2_folder",
    n_ims_w=9,                                          # fields of view per well
)

cvseg.run_and_stream_save_channelwise_stripped(
    model_batches=model_batches,
    channels=[cvseg.Channel.BF, cvseg.Channel.CY5],
    out_zarr="../zarrs/experiment_batches_<ID>_features.zarr",
    out_jsonl="../jsonls/experiment_batches_<ID>_features.jsonl",
    show_progress=True,
)
```

`segment_chlorella_strains.ipynb` does this for each of the 7 pub experiments. Set each experiment's `model_ran_<ID>` flag to `False` to rerun it. The outputs go to `../zarrs/` and `../jsonls/`, where the main analysis expects them. Name them `experiment_batches_<ID>_features.{zarr,jsonl}` so the main notebook can find them by experiment ID.

For each field of view, then for each channel, the pipeline:

1. Loads that field of view's ND2 files (one per well).
2. Preprocesses: background subtraction (0th percentile), then rescaling intensities to the 0.3–99.7 percentile range. The processed images are cast to `float32`.
3. Segments cells with Cellpose (`flow_threshold = 0.4`, `cellprob_threshold = 0`; both are arguments of `run_and_stream_save_channelwise_stripped`).
4. Writes raw images (original integer type), processed images, and label and mask images to the zarr store, and outlines and cell properties to the JSONL.
5. Frees that channel's data before moving on.

### 4. Read the results

The main analysis reads these files with `ExperimentStore` from `../main_analyses/cvp_pub.py`. It builds an index of the JSONL once, then fetches any single image or table directly:

```python
import sys
sys.path.append("../main_analyses")
import cvp_pub as cvpp

store = cvpp.ExperimentStore("../zarrs/experiment_batches_<ID>_features.zarr",
                             "../jsonls/experiment_batches_<ID>_features.jsonl")
fov, well = 0, 12
img   = store.get_processed_intensities(fov, well, "BF")   # 1024 × 1024 array
cells = store.get_cell_properties(fov, well, "BF")         # pandas DataFrame, one row per cell
```

## Channel names

The code calls the two channels `BF` (bright-field; `DIC` in our ND2 files) and `CY5` (`Cy5 BP` in our ND2 files). Our ND2 files store Cy5 as the first plane and DIC as the second, so channels are always looked up by name, never by position:

* When `cvseg` is imported, it tells arcadia-microscopy-tools that `Cy5 BP` means Cy5 and `DIC` means bright-field. The pinned version of the library doesn't know these names and would otherwise refuse to load the files.
* Raw planes are matched to channels by name through `CHANNEL_NAME_ALIASES` at the top of `cvseg.py`. If your microscope uses other channel names, add them there. If a channel can't be identified, the run stops with an error instead of guessing.

**Known issue in the deposited data.** For the 6 mutant experiments (IDs starting `202601…`), the zarr stores on the BioImage Archive have `raw_intensities/BF` and `raw_intensities/CY5` swapped. An earlier version of `cvseg.py` assumed bright-field was the first plane. Processed images, masks, outlines, and cell properties are correct, and the pub's results don't use raw intensities. To fix a downloaded store, swap the names of its two `raw_intensities` subfolders. The wild-type store (`20251209_091846_235`) is correct as deposited.

**The deposited stores also contain `segmentation/intensity_image/`.** Earlier versions of `cvseg.py` saved Cellpose's intensity image, which is identical to `processed_intensities`. It's no longer saved, and nothing in this repository reads it, so you can ignore or delete it (about 6 GB per store).

## Things to know

* **Reruns overwrite.** Each run creates its zarr store and JSONL from scratch; there's no resume. If a run stops partway, rerun that experiment.
* **Equal-sized batches.** Every field of view must have the same number of wells, and all images the same size.
* **File-name matching.** The field-of-view number is matched anywhere in the file's full path. Avoid folder names that contain zero-padded numbers like `0001`.
