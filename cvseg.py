#Imports

import os 
abra=os.getcwd() #when lost, teleport to abra

from pathlib import Path

import cellpose
import copy
import numpy as np
import skimage as ski
from arcadia_microscopy_tools import Channel
from arcadia_microscopy_tools.batch import ImageBatch
from arcadia_microscopy_tools.model import SegmentationModel
from arcadia_microscopy_tools.operations import (
    get_background_subtracted_intensities,
    rescale_intensity_by_percentiles,
)
from arcadia_microscopy_tools.pipeline import ImageOperation, Pipeline
from natsort import natsorted

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot_2samples
from statsmodels.graphics.gofplots import ProbPlot

import random
import time
from sklearn.manifold import TSNE
#faster tsne
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.stats import gaussian_kde
import math

#PCA import tools
from scipy import linalg as LA
import itertools
import pickle


import json
import zarr
import pandas as pd
import json
import zarr
import psutil
import gc
from dataclasses import is_dataclass, asdict
from datetime import date, datetime
from enum import Enum

def consecutive_number_strings(num_digits: int, length: int, start: int = 0):
    return [str(start + i).zfill(num_digits) for i in range(length)]

def prep_new_segmentation(data_loc,n_ims_w):

    # Set input directory


    input_directory = Path(data_loc)
    unsorted_experiment_nd2_paths = natsorted(input_directory.glob("*.nd2"))

    #Organize experiment into batches. Each batch contains one image (length=1) per well. Max Number of batches are the total images taken per well


    
    nd2_paths_batches=[]
    
    for i in range(n_ims_w):
        experiment_nd2_paths=[]
        for p in unsorted_experiment_nd2_paths:
            if any(item in str(p) for item in consecutive_number_strings(num_digits=4,length=1,start=i)):
                experiment_nd2_paths.append(p)
        nd2_paths_batches.append(experiment_nd2_paths)
    
    #Will Run the model in batches, Each batch will contain one image from each well (as organized above)
     
    
    model_batches=[]
    
    for batch in nd2_paths_batches:
    
        # Create batch
        experiment_batch = ImageBatch.from_paths(
            batch,
            batch_size=len(batch),
            shuffle=False,
            random_state=37,
        )
        model_batches.append(experiment_batch)
    return model_batches
    


# ---------- JSONL: build offset index once ----------
def build_jsonl_offset_index(jsonl_path):
    idx = {}
    with open(jsonl_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            rec = json.loads(line.decode("utf-8"))
            idx[(rec["batch_index"], rec["image_index"], rec["channel"])] = pos
    return idx

def load_jsonl_record_fast(jsonl_path, idx, b, im, channel_name):
    key = (b, im, channel_name)
    pos = idx.get(key)
    if pos is None:
        return None
    with open(jsonl_path, "rb") as f:
        f.seek(pos)
        line = f.readline()
    return json.loads(line.decode("utf-8"))

# ---------- Zarr ----------
def open_experiment_store(zarr_path):
    root = zarr.open_group(zarr_path, mode="r")
    N = root.attrs.get("n_images_per_batch", None)
    return root, N

def load_processed_intensities(root, b, im, channel_name="BF"):
    return root[f"processed_intensities/{channel_name}"][b, im]

def load_segmentation_field(root, field, b, im, channel_name="BF"):
    return root[f"segmentation/{field}/{channel_name}"][b, im]

# ---------- all-in-one ----------
def load_experiment_view(zroot, jsonl_path, jsonl_idx, b, im, channel_name="BF"):
    view = {
        "processed_intensities": load_processed_intensities(zroot, b, im, channel_name),
        "intensity_image": load_segmentation_field(zroot, "intensity_image", b, im, channel_name),
        "label_image": load_segmentation_field(zroot, "label_image", b, im, channel_name),
        "mask_image": load_segmentation_field(zroot, "mask_image", b, im, channel_name),
    }
    rec = load_jsonl_record_fast(jsonl_path, jsonl_idx, b, im, channel_name) or {}
    view["cell_outlines"] = rec.get("cell_outlines")
    view["cell_properties"] = rec.get("cell_properties")
    return view



#  Fully updated: channelwise streaming saver + stripper + psutil memory checks
#  - Saves one channel at a time (no duplicate JSONL records)
#  - Strips segmentation objects down to ONLY:
#      mask_image, label_image, intensity_image, cell_outlines, cell_properties
#  - Uses Zarr v3-compatible resize (tuple shape) to avoid BoundsCheckError
#  - Prints memory (RSS) at key points so you can see where it grows
#  - CHATGPT helped a TON with this section (mostly the stream and save function



# ---------- memory helper (psutil) ----------
def mem_gb(label=""):
    p = psutil.Process(os.getpid())
    gb = p.memory_info().rss / (1024**3)
    print(f"[MEM] {label:28s}: {gb:6.2f} GB")

# ---------- general helpers ----------
def ch_name(ch):
    return getattr(ch, "name", str(ch))

def to_jsonable(x):
    if is_dataclass(x):
        return to_jsonable(asdict(x))
    if isinstance(x, Enum):
        return x.name
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    try:
        import numpy as _np
        if isinstance(x, _np.generic):
            return x.item()
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return str(x) if hasattr(x, "__dict__") else x

# ---------- Option A: minimal stripper ----------
def strip_segmentation_mask(sm):
    """
    Extract only what you need into a lightweight dict:
      mask_image, label_image, intensity_image, cell_outlines, cell_properties
    """
    out = {}
    for f in ("mask_image", "label_image", "intensity_image"):
        out[f] = np.asarray(getattr(sm, f)) if hasattr(sm, f) else None

    outlines = getattr(sm, "cell_outlines", None)
    out["cell_outlines"] = [np.asarray(poly) for poly in outlines] if outlines is not None else None

    out["cell_properties"] = getattr(sm, "cell_properties", None)
    return out

def _get(sm, key):
    """Read field from either a stripped dict or an original object."""
    if isinstance(sm, dict):
        return sm.get(key)
    return getattr(sm, key, None)

# ---------- streaming writer ----------
class ExperimentStoreWriter:
    """
    Incrementally writes:
      - processed_intensities_dict[channel][im] -> Zarr
      - segmentation_masks_dict[channel][im].{intensity_image,label_image,mask_image} -> Zarr
      - segmentation_masks_dict[channel][im].cell_outlines + cell_properties -> JSONL

    Arrays are created lazily and resized along batch axis.
    """
    def __init__(self, out_zarr: str, out_jsonl: str):
        self.out_zarr = out_zarr
        self.out_jsonl = out_jsonl

        self.root = zarr.open_group(out_zarr, mode="w")
        with open(out_jsonl, "w") as f:
            pass

        self.arrays = {}  # path -> zarr.Array
        self.N = None
        self.B = 0

    def _infer_N(self, batch, channels_subset):
        if self.N is not None:
            return
        # infer N from first available channel in either dict
        for ch in channels_subset:
            if hasattr(batch, "processed_intensities_dict") and ch in batch.processed_intensities_dict:
                self.N = len(batch.processed_intensities_dict[ch]); break
            if hasattr(batch, "segmentation_masks_dict") and ch in batch.segmentation_masks_dict:
                self.N = len(batch.segmentation_masks_dict[ch]); break
        if self.N is None:
            raise ValueError("Could not infer N (images per batch).")
        self.root.attrs["n_images_per_batch"] = self.N

    def _ensure_array(self, path, sample_arr):
        if path in self.arrays:
            return self.arrays[path]
        sample_arr = np.asarray(sample_arr)
        arr = self.root.create_array(
            path,
            shape=(0, self.N) + sample_arr.shape,   # (B,N,...) with B resizable
            chunks=(1, 1) + sample_arr.shape,
            dtype=sample_arr.dtype,
            overwrite=True,
        )
        self.arrays[path] = arr
        return arr

    def _resize_for_batch(self, arr, b_idx):
        # Zarr v3 safe: resize expects a shape tuple
        if arr.shape[0] <= b_idx:
            arr.resize((b_idx + 1,) + arr.shape[1:])

    def append_batch(self, batch, b_idx: int, channels_subset):
        """
        Append ONLY the channels in channels_subset for this batch index.
        This prevents duplicate JSONL records when saving channel-by-channel.
        """
        channels_subset = list(channels_subset)
        self._infer_N(batch, channels_subset)

        # track channels written
        prev = set(self.root.attrs.get("channels", []))
        now = prev | {ch_name(ch) for ch in channels_subset}
        self.root.attrs["channels"] = sorted(now)

        with open(self.out_jsonl, "a") as fj:
            for ch in channels_subset:
                cname = ch_name(ch)

                # ---- processed intensities ----
                if hasattr(batch, "processed_intensities_dict") and ch in batch.processed_intensities_dict:
                    proc_list = batch.processed_intensities_dict[ch]
                    if len(proc_list) != self.N:
                        raise ValueError(f"{cname}: expected N={self.N}, got {len(proc_list)}")

                    proc_arr = self._ensure_array(f"processed_intensities/{cname}", proc_list[0])
                    self._resize_for_batch(proc_arr, b_idx)
                    for im in range(self.N):
                        proc_arr[b_idx, im] = np.asarray(proc_list[im])

                # ---- segmentation + JSONL ----
                if hasattr(batch, "segmentation_masks_dict") and ch in batch.segmentation_masks_dict:
                    seg_list = batch.segmentation_masks_dict[ch]
                    if len(seg_list) != self.N:
                        raise ValueError(f"{cname}: expected N={self.N}, got {len(seg_list)}")

                    for im in range(self.N):
                        sm = seg_list[im]

                        for field in ["intensity_image", "label_image", "mask_image"]:
                            arr = _get(sm, field)
                            if arr is not None:
                                arr = np.asarray(arr)
                                z = self._ensure_array(f"segmentation/{field}/{cname}", arr)
                                self._resize_for_batch(z, b_idx)
                                z[b_idx, im] = arr

                        outlines = _get(sm, "cell_outlines")
                        outlines_list = [np.asarray(poly).tolist() for poly in outlines] if outlines is not None else None
                        cell_properties = _get(sm, "cell_properties")

                        rec = {
                            "batch_index": b_idx,
                            "image_index": im,
                            "channel": cname,
                            "cell_outlines": to_jsonable(outlines_list),
                            "cell_properties": to_jsonable(cell_properties),
                        }
                        fj.write(json.dumps(rec) + "\n")

        self.B = max(self.B, b_idx + 1)
        self.root.attrs["n_batches"] = self.B

# ---------- runner: channelwise + stripping + memory prints ----------
def run_and_stream_save_channelwise_stripped(
    model_batches,
    channels,
    out_zarr,
    out_jsonl,
    flow_threshold=0.4,
    cellprob_threshold=0,
    show_progress=True,
):
    """
    For each batch, for each channel:
      - pipeline -> segment
      - cast processed intensities -> float32
      - strip segmentation objects (label/mask/outlines/properties only)
      - append ONLY that channel to disk (raw + processed + seg + jsonl)
      - drop channel outputs from the batch
      - print memory at key steps
    """
    writer = ExperimentStoreWriter(out_zarr=out_zarr, out_jsonl=out_jsonl)

    for b_idx, batch in enumerate(model_batches):
        print(f"\n===== BATCH {b_idx+1} START =====")
        mem_gb("batch start")

        for ch in channels:
            cname = ch_name(ch)
            print(f"\n--- Channel {cname} ---")
            mem_gb("before pipeline")

            experiment_pipeline = Pipeline([
                ImageOperation(get_background_subtracted_intensities, percentile=0),
                ImageOperation(rescale_intensity_by_percentiles, percentile_range=(0.3, 99.7)),
            ])
            _ = batch.apply_pipeline(experiment_pipeline, ch, show_progress=show_progress)
            mem_gb("after pipeline")

            # ✅ reduce memory: force processed intensities to float32
            if hasattr(batch, "processed_intensities_dict") and ch in batch.processed_intensities_dict:
                batch.processed_intensities_dict[ch] = [
                    np.asarray(x, dtype=np.float32) for x in batch.processed_intensities_dict[ch]
                ]
            mem_gb("after cast proc->float32")

            model = SegmentationModel(flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
            _ = batch.segment(model, channel=ch)
            mem_gb("after segment")

            # --- strip segmentation outputs and drop original refs ASAP ---
            if hasattr(batch, "segmentation_masks_dict") and ch in batch.segmentation_masks_dict:
                original_seg_list = batch.segmentation_masks_dict[ch]
                stripped = [strip_segmentation_mask(sm) for sm in original_seg_list]
                batch.segmentation_masks_dict[ch] = stripped
                del original_seg_list
                gc.collect()

            mem_gb("after strip (and del originals)")

            # ✅ append ONLY this channel (no duplicates)
            writer.append_batch(batch, b_idx=b_idx, channels_subset=[ch])
            mem_gb("after save")

            # ✅ drop in-memory outputs for this channel
            if hasattr(batch, "processed_intensities_dict"):
                batch.processed_intensities_dict.pop(ch, None)
            if hasattr(batch, "segmentation_masks_dict"):
                batch.segmentation_masks_dict.pop(ch, None)

            del model, experiment_pipeline
            gc.collect()
            mem_gb("after cleanup")

            print(f"    completed channel {cname}")

        del batch
        gc.collect()
        mem_gb("batch end")

    print("\nDone. Saved:", out_zarr, "and", out_jsonl)

class ExperimentStore:
    """
    Fast read-only interface to:
      - Zarr arrays (raw + processed intensities, segmentation label/mask)
      - JSONL (cell_outlines + cell_properties), indexed for O(1) lookup
    """
    def __init__(self, zarr_path, jsonl_path):
        self.zarr_path = zarr_path
        self.jsonl_path = jsonl_path

        self.root = zarr.open_group(zarr_path, mode="r")
        self.N = self.root.attrs.get("n_images_per_batch", None)
        self.channels = self.root.attrs.get("channels", [])

        self._jsonl_index = self._build_jsonl_offset_index()

    # ---------- JSONL indexing ----------
    def _build_jsonl_offset_index(self):
        idx = {}
        with open(self.jsonl_path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                rec = json.loads(line.decode("utf-8"))
                idx[(rec["batch_index"], rec["image_index"], rec["channel"])] = pos
        return idx

    def _load_jsonl_record(self, b, im, channel_name):
        key = (b, im, channel_name)
        pos = self._jsonl_index.get(key)
        if pos is None:
            return None
        with open(self.jsonl_path, "rb") as f:
            f.seek(pos)
            line = f.readline()
        return json.loads(line.decode("utf-8"))

    # ---------- Zarr getters ----------
    def get_raw_intensities(self, b, im, channel_name="CY5"):
        """Raw intensities array; use .ravel() to match your old .flatten() usage."""
        return self.root[f"raw_intensities/{channel_name}"][b, im]

    def get_processed_intensities(self, b, im, channel_name="BF"):
        return self.root[f"processed_intensities/{channel_name}"][b, im]

    def get_label_image(self, b, im, channel_name="BF"):
        return self.root[f"segmentation/label_image/{channel_name}"][b, im]

    def get_mask_image(self, b, im, channel_name="BF"):
        return self.root[f"segmentation/mask_image/{channel_name}"][b, im]

    # ---------- JSONL getters ----------
    def get_cell_outlines(self, b, im, channel_name="BF"):
        rec = self._load_jsonl_record(b, im, channel_name)
        return None if rec is None else rec.get("cell_outlines")

    def get_cell_properties(self, b, im, channel_name="BF"):
        """Returns a pandas DataFrame equivalent to pd.DataFrame(segmentation_mask.cell_properties)."""
        rec = self._load_jsonl_record(b, im, channel_name)
        if rec is None:
            return None
        props = rec.get("cell_properties")
        return None if props is None else pd.DataFrame(props)

    # ---------- convenience ----------
    def get(self, b, im, channel_name="BF", include_raw=False):
        """
        Returns a dict with arrays + outlines/properties.
        include_raw=True also returns raw intensities.
        """
        out = {
            "processed_intensities": self.get_processed_intensities(b, im, channel_name),
            "label_image": self.get_label_image(b, im, channel_name),
            "mask_image": self.get_mask_image(b, im, channel_name),
        }
        if include_raw:
            out["raw_intensities"] = self.get_raw_intensities(b, im, channel_name)

        rec = self._load_jsonl_record(b, im, channel_name) or {}
        out["cell_outlines"] = rec.get("cell_outlines")
        out["cell_properties"] = rec.get("cell_properties")
        return out