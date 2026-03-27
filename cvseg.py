#imports: general
import os 
from pathlib import Path
from collections.abc import Sequence
import numpy as np

#imports: segmentation
from arcadia_microscopy_tools import Channel
from arcadia_microscopy_tools.batch import ImageBatch
from arcadia_microscopy_tools.model import SegmentationModel
from arcadia_microscopy_tools.operations import (
    get_background_subtracted_intensities,
    rescale_intensity_by_percentiles,
)
from arcadia_microscopy_tools.pipeline import ImageOperation, Pipeline
from natsort import natsorted

#imports: data management
import json
import zarr
import zarr
import psutil
import gc
from dataclasses import is_dataclass, asdict
from datetime import date, datetime
from enum import Enum

def consecutive_number_strings(num_digits: int, length: int, start: int = 0):
    return [str(start + i).zfill(num_digits) for i in range(length)]

def debug_batch_attrs(batch, label=""):
    print(f"\n[BATCH ATTRS] {label}")
    for k, v in sorted(vars(batch).items()):
        try:
            if isinstance(v, dict):
                info = f"dict len={len(v)}"
            elif isinstance(v, list):
                info = f"list len={len(v)}"
            else:
                info = type(v).__name__
            print(f"  {k}: {info}")
        except Exception as e:
            print(f"  {k}: <error inspecting: {e}>")

def clear_image_payloads(batch):
    if not hasattr(batch, "images") or not isinstance(batch.images, list):
        return

    for img in batch.images:
        for attr_name, attr_value in list(vars(img).items()):
            try:
                if isinstance(attr_value, dict):
                    attr_value.clear()
                elif isinstance(attr_value, list):
                    attr_value.clear()
                elif isinstance(attr_value, np.ndarray):
                    setattr(img, attr_name, None)
            except Exception:
                pass


class _LazyImageBatchSequence(Sequence):
    """
    Sequence-like wrapper that lazily constructs ImageBatch objects on demand.

    Preserves:
      - iteration: for batch in model_batches
      - len(model_batches)
      - indexing: model_batches[i]

    while avoiding construction of all ImageBatch objects up front.
    """
    def __init__(self, nd2_paths_batches):
        self._nd2_paths_batches = nd2_paths_batches

    def __len__(self):
        return len(self._nd2_paths_batches)

    def __getitem__(self, idx):
        batch_paths = self._nd2_paths_batches[idx]
        return ImageBatch.from_paths(
            batch_paths,
            batch_size=len(batch_paths),
            shuffle=False,
            random_state=37,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def prep_new_segmentation(data_loc, n_ims_w):
    # Set input directory
    input_directory = Path(data_loc)
    unsorted_experiment_nd2_paths = natsorted(input_directory.glob("*.nd2"))

    # Organize experiment into batches. Each batch contains one image (length=1)
    # per well. Max number of batches is the total images taken per well.
    nd2_paths_batches = []

    for i in range(n_ims_w):
        experiment_nd2_paths = []
        target_str = str(i).zfill(4)

        for p in unsorted_experiment_nd2_paths:
            if target_str in str(p):
                experiment_nd2_paths.append(p)

        nd2_paths_batches.append(experiment_nd2_paths)

    # Return a lazy sequence so downstream code can iterate exactly as before,
    # but ImageBatch objects are only created when needed.
    return _LazyImageBatchSequence(nd2_paths_batches)
    

#  Fully updated: channelwise streaming saver + stripper + psutil memory checks
#  - Saves one channel at a time (no duplicate JSONL records)
#  - Strips segmentation objects down to ONLY:
#      mask_image, label_image, intensity_image, cell_outlines, cell_properties
#  - Uses Zarr v3-compatible resize (tuple shape) to avoid BoundsCheckError
#  - Prints memory (RSS) at key points so you can see where it grows
#  - CHATGPT helped a TON with this section (mostly the stream and save function)


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
        val = getattr(sm, f, None)
        if val is None:
            out[f] = None
        elif isinstance(val, np.ndarray):
            out[f] = val
        else:
            out[f] = np.asarray(val)

    outlines = getattr(sm, "cell_outlines", None)
    if outlines is None:
        out["cell_outlines"] = None
    else:
        out["cell_outlines"] = [
            poly if isinstance(poly, np.ndarray) else np.asarray(poly)
            for poly in outlines
        ]

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

    Same function signature and same output layout as before.
    """
    writer = ExperimentStoreWriter(out_zarr=out_zarr, out_jsonl=out_jsonl)

    for b_idx, batch in enumerate(model_batches):
        print(f"\n===== BATCH {b_idx+1} START =====")
        mem_gb("batch start")

        try:
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

                # Reduce memory pressure: cast processed intensities to float32
                # as early as possible, avoiding unnecessary copies when possible.
                if hasattr(batch, "processed_intensities_dict") and ch in batch.processed_intensities_dict:
                    proc_list = batch.processed_intensities_dict[ch]
                    batch.processed_intensities_dict[ch] = [
                        x.astype(np.float32, copy=False) if isinstance(x, np.ndarray)
                        else np.asarray(x, dtype=np.float32)
                        for x in proc_list
                    ]
                    del proc_list

                gc.collect()
                #mem_gb("after cast proc->float32")

                model = SegmentationModel(
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                )
                _ = batch.segment(model, channel=ch)
                mem_gb("after segment")

                # Strip segmentation outputs immediately so we do not retain
                # heavyweight segmentation objects longer than needed.
                if hasattr(batch, "segmentation_masks_dict") and ch in batch.segmentation_masks_dict:
                    original_seg_list = batch.segmentation_masks_dict[ch]
                    batch.segmentation_masks_dict[ch] = [
                        strip_segmentation_mask(sm) for sm in original_seg_list
                    ]
                    del original_seg_list
                    gc.collect()

                #mem_gb("after strip (and del originals)")

                # Append ONLY this channel to disk
                writer.append_batch(batch, b_idx=b_idx, channels_subset=[ch])
                #mem_gb("after save")

                # Drop in-memory outputs for this channel immediately after save
                if hasattr(batch, "processed_intensities_dict"):
                    batch.processed_intensities_dict.pop(ch, None)

                if hasattr(batch, "segmentation_masks_dict"):
                    batch.segmentation_masks_dict.pop(ch, None)

                # Clear large locals
                del model
                del experiment_pipeline
                gc.collect()
                mem_gb("after cleanup")

                print(f"    completed channel {cname}")

            #debug_batch_attrs(batch, "before cleanup")

            # After all channels, clear any remaining dict contents on the batch
            if hasattr(batch, "processed_intensities_dict") and isinstance(batch.processed_intensities_dict, dict):
                batch.processed_intensities_dict.clear()

            if hasattr(batch, "segmentation_masks_dict") and isinstance(batch.segmentation_masks_dict, dict):
                batch.segmentation_masks_dict.clear()

            clear_image_payloads(batch)

            if hasattr(batch, "images") and isinstance(batch.images, list):
                batch.images.clear()

            # Defensive cleanup for other possible large dict-like caches
            for attr_name in [
                "raw_intensities_dict",
                "background_subtracted_intensities_dict",
                "intensity_images_dict",
                "label_images_dict",
                "mask_images_dict",
            ]:
                if hasattr(batch, attr_name):
                    attr = getattr(batch, attr_name)
                    if isinstance(attr, dict):
                        attr.clear()

            #debug_batch_attrs(batch, "after cleanup")

        finally:
            del batch
            gc.collect()
            mem_gb("batch end")

    print("\nDone. Saved:", out_zarr, "and", out_jsonl)