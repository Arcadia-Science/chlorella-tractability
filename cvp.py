#Imports

#plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import arcadia_pycolor as apc
apc.mpl.setup()
apc.plotly.setup()

#matrices
import numpy as np
import pandas as pd

#segmentation
import cellpose
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

#basic tools
import random
import time
import math
import copy
import itertools

#file management and reading
import json
import zarr
import os, psutil
import gc
from dataclasses import is_dataclass, asdict
from datetime import date, datetime
from enum import Enum
import pickle
from pathlib import Path
import sys
sys.path.append("..")

#stats
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.stats import gaussian_kde
from scipy import linalg as LA
from scipy.stats import zscore
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import leaves_list
from statsmodels.graphics.gofplots import qqplot_2samples
from statsmodels.graphics.gofplots import ProbPlot
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

#ml
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE

# Functions


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




# =========================
# FAST LOADER (UPDATED)
# Works with the NEW on-disk schema:
#   raw_intensities/<CHANNEL>                (B,N, ...)
#   processed_intensities/<CHANNEL>          (B,N, ...)
#   segmentation/label_image/<CHANNEL>       (B,N, H, W)
#   segmentation/mask_image/<CHANNEL>        (B,N, H, W)
# JSONL:
#   batch_index, image_index, channel, cell_outlines, cell_properties
#
# NOTE: "intensity_image" is intentionally NOT saved anymore.
# If you want a replacement, use processed_intensities.
# =========================



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

# =========================
# Example usage:
# =========================
# store = ExperimentStore(
#     "experiment_batches_20251209_091846_235_features.zarr",
#     "experiment_batches_20251209_091846_235_features.jsonl",
# )
#
# b, im = 0, 0
# raw_flat = store.get_raw_intensities(b, im, "CY5").ravel()
# proc = store.get_processed_intensities(b, im, "CY5")
# label = store.get_label_image(b, im, "CY5")
# mask  = store.get_mask_image(b, im, "CY5")
# outlines = store.get_cell_outlines(b, im, "CY5")
# df = store.get_cell_properties(b, im, "CY5")


def euclidean_distance_ignore_nans(a, b):
    # Calculate squared differences, NaNs here are fine for intermediate calculation
    sq_diffs = (a - b) ** 2
    # Sum the squared differences, ignoring NaNs
    nansum_sq_diffs = np.nansum(sq_diffs)
    # Take the square root
    distance = np.sqrt(nansum_sq_diffs)
    return distance


def remove_rows_cols(array, row_indices, col_indices):
    """
    Removes specified rows and columns from a 2D numpy array.
    """
    # Remove rows (axis 0)
    temp_array = np.delete(array, row_indices, axis=0)
    # Remove columns (axis 1)
    final_array = np.delete(temp_array, col_indices, axis=1)
    return final_array

# --- Usage ---
# Create example 96x22 array (using smaller numbers for illustration)
# arr = np.random.randint(0, 100, size=(96, 22))

# Rows and columns 36 and 41 are removed (0-indexed, so 35 and 40)
# modified_arr = remove_rows_cols(arr, [35, 40], [35, 40])

def keep_indices(arr, row_indices, col_indices):
    """
    Returns a subarray containing only the specified row and column indices.
    """
    # Integer array indexing allows selecting specific rows and columns simultaneously
    # By indexing with [row_indices, :], we grab rows; then [:, col_indices] grabs columns.
    return arr[np.ix_(row_indices, col_indices)]

# Example usage:
# my_array = np.random.rand(96, 22)
# rows_to_keep = [0, 1, 2, 36, 41]
# cols_to_keep = [5, 10, 15]
# new_array = keep_indices(my_array, rows_to_keep, cols_to_keep)


def view_clones(zarr_roots,exp_names,clone_numbers=[0],image_number=0):

    fig,ax=plt.subplots(1,len(clone_numbers),figsize=(int(len(clone_numbers)*10),10))

    for e in range(len(clone_numbers)):


        mask_arr_bf=copy.deepcopy(zarr_roots[e].get_mask_image(image_number, clone_numbers[e], "BF"))
        new_mask_arr_bf = np.where(mask_arr_bf != 0, 1, 0)
    
        mask_arr_cy5=copy.deepcopy(zarr_roots[e].get_mask_image(image_number, clone_numbers[e], "CY5"))
        new_mask_arr_cy5 = np.where(mask_arr_cy5 != 0, 1, 0)

        if len(clone_numbers) == 1:
            ax.imshow(np.multiply(zroots[e].get_processed_intensities(image_number, clone_numbers[e], "BF"),new_mask_arr_bf),cmap='Greys')
            ax.imshow(np.multiply(zroots[e].get_processed_intensities(image_number, clone_numbers[e], "CY5"),new_mask_arr_cy5),cmap='Greens',vmin=0,vmax=1,alpha=.8)
            ax.set_xticks([]) 
            ax.set_yticks([])


        if len(clone_numbers) > 1:
            ax[e].imshow(np.multiply(zroots[e].get_processed_intensities(image_number, clone_numbers[e], "BF"),new_mask_arr_bf),cmap='cubehelix',alpha=1)
            ax[e].imshow(np.multiply(zroots[e].get_processed_intensities(image_number, clone_numbers[e], "CY5"),new_mask_arr_cy5),cmap='Greens',vmin=0,vmax=1,alpha=.8)
            ax[e].set_xticks([]) 
            ax[e].set_yticks([])
            ax[e].set_title(exp_names[e] + ': Clone '  + str(clone_numbers[e]), fontsize=18)


    #axes[0, e].set_title(col_titles[e], fontsize=18)
            
    plt.show()


def pca_on_np_array(nparray):
    '''rows are instances , columns are dimensions(variables)'''
    #center data
    mean_value = np.mean(nparray, axis = 0) 
    nparray -= mean_value 
    #calculate covariance matrix
    cov = np.cov(nparray, rowvar = False)
    #get eigenvectors and eigenvalues of the covariance matrix
    evals , evecs = LA.eigh(cov)

    #formatting and SORT by eigenvalues (aka sort by how much variance each eigen vector explains)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    #this is your data represented as PCS 
    nparray_represented_as_pcs = np.dot(nparray, evecs)
    
    return nparray_represented_as_pcs,evecs,evals,mean_value


def represent_data_with_n_pcs (pc_scores,eigenvectors,npcs,mean_value):
    
    n_pc_representation=[]
    
    for n in pc_scores:
        
        x=n[0:npcs]
        
        y=np.dot(x, eigenvectors.T[0:npcs]) + mean_value
 
        n_pc_representation.append(y)
        
    return np.array(n_pc_representation)


def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    # Handle the case where all elements are the same to avoid division by zero
    if max_val == min_val:
        return np.zeros_like(arr, dtype=float) 
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr


def nan_cosine_distance(u, v):
    """
    Calculate the cosine distance between two vectors, ignoring NaNs in either vector.

    Args:
        u (np.ndarray): First vector.
        v (np.ndarray): Second vector.

    Returns:
        float: The cosine distance (1 - cosine similarity).
               Returns NaN if there are no common non-NaN elements.
    """
    # Ensure inputs are numpy arrays
    u = np.asarray(u)
    v = np.asarray(v)

    # Find indices where neither vector has a NaN value
    # isfinite() checks for non-NaN and non-inf values
    common_indices = np.isfinite(u) & np.isfinite(v)

    # Check if there are common indices
    if np.sum(common_indices) == 0:
        return np.nan # Or some other appropriate value/error

    # Filter vectors to keep only common, non-NaN elements
    u_clean = u[common_indices]
    v_clean = v[common_indices]

    # Calculate cosine distance on the clean data
    # scipy.spatial.distance.cosine computes the distance (1 - similarity)
    distance = cosine(u_clean, v_clean)
    
    return distance


def train_and_predict(
    data,
    train_conditions,
    test_condition,
    model_type="ridge",
    alpha=100,
    n_pls_components=2
):
    """
    data: dict[str, np.ndarray]  (96, 22)
    train_conditions: list[str]
    test_condition: str
    """

    # --- Build training matrix ---
    X_train = np.hstack([data[c] for c in train_conditions])
    y_train = data[test_condition]

    # --- Choose model ---
    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "pls":
        model = PLSRegression(n_components=n_pls_components)
    else:
        raise ValueError("Unknown model_type")

    # --- Scale + fit ---
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # --- Predict ---
    y_pred = pipeline.predict(X_train)

    # --- Metrics ---
    metrics = {
        "r2_overall": r2_score(y_train, y_pred),
        "rmse_overall": np.sqrt(mean_squared_error(y_train, y_pred)),
        "r2_per_feature": [
            r2_score(y_train[:, i], y_pred[:, i])
            for i in range(y_train.shape[1])
        ]
    }

    return y_pred, metrics, pipeline


#hold out strains for testing

def cross_validated_performance(
    data,
    train_conditions,
    test_condition,
    model_type="ridge",
    alpha=1,
    n_splits=12
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true = data[test_condition]
    X_full = np.hstack([data[c] for c in train_conditions])

    r2_scores = []

    for train_idx, test_idx in kf.split(X_full):
        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_true[train_idx], y_true[test_idx]

        if model_type == "ridge":
            model = Ridge(alpha=alpha)
        else:
            model = PLSRegression(n_components=3)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))

    return r2_scores, y_test, y_pred


#hold out strains for testing

def cross_validated_performance(
    data,
    train_conditions,
    test_condition,
    model_type="ridge",
    alpha=1,
    n_splits=12
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true = data[test_condition]
    X_full = np.hstack([data[c] for c in train_conditions])

    r2_scores = []

    for train_idx, test_idx in kf.split(X_full):
        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_true[train_idx], y_true[test_idx]

        if model_type == "ridge":
            model = Ridge(alpha=alpha)
        else:
            model = PLSRegression(n_components=3)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))

    return r2_scores, y_test, y_pred


def evaluate_all_condition_holdouts(
    data,
    model_type="ridge",
    alpha=1.0,
    n_splits=5,
    random_state=42
):
    """
    data: dict[str, np.ndarray] of shape (n_strains, n_features)
    """

    conditions = list(data.keys())
    n_conditions = len(conditions)
    n_features = data[conditions[0]].shape[1]

    overall_r2 = np.zeros(n_conditions)
    rmse = np.zeros(n_conditions)
    r2_per_feature = np.zeros((n_conditions, n_features))

    for c_idx, holdout_condition in enumerate(conditions):

        train_conditions = [c for c in conditions if c != holdout_condition]

        # Build full design matrix
        X_full = np.hstack([data[c] for c in train_conditions])
        y_full = data[holdout_condition]

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in kf.split(X_full):

            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y_full[train_idx], y_full[test_idx]

            if model_type == "ridge":
                model = Ridge(alpha=alpha)
            else:
                raise ValueError("Unsupported model")

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)

        # Concatenate across folds
        y_true_all = np.vstack(y_true_all)
        y_pred_all = np.vstack(y_pred_all)

        # Metrics
        overall_r2[c_idx] = r2_score(y_true_all, y_pred_all)
        rmse[c_idx] = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

        for f in range(n_features):
            r2_per_feature[c_idx, f] = r2_score(
                y_true_all[:, f],
                y_pred_all[:, f]
            )

    return {
        "conditions": conditions,
        "overall_r2": overall_r2,
        "rmse": rmse,
        "r2_per_feature": r2_per_feature
    }




def predict_new(pipeline, new_data_by_condition, train_conditions):
    """
    new_data_by_condition: dict[str, np.ndarray] where each array is (n_new, 44)
    train_conditions: list[str] in the same order used during training
    """
    X_new = np.hstack([new_data_by_condition[c] for c in train_conditions])
    y_new_pred = pipeline.predict(X_new)  # (n_new, 22)
    return y_new_pred

def _cv_predict_metrics(X, y, alpha=1.0, n_splits=5, random_state=42):
    """
    Strain-level KFold CV: each strain is predicted out-of-sample exactly once.
    Returns:
      - overall R² (variance-weighted across outputs)
      - overall RMSE
      - per-feature R² (length n_features)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    r2_overall = r2_score(y_true_all, y_pred_all, multioutput="variance_weighted")
    rmse_overall = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

    r2_per_feature = np.array([
        r2_score(y_true_all[:, j], y_pred_all[:, j])
        for j in range(y.shape[1])
    ])

    return r2_overall, rmse_overall, r2_per_feature


def condition_holdout_permutation_test_featurewise(
    data,
    alpha=1.0,
    n_splits=5,
    n_permutations=500,
    random_state=42
):
    """
    For each holdout condition:
      - X = concat(features from all other conditions)
      - y = features for holdout condition
      - Compute real CV metrics
      - Permute y across strains, recompute metrics, and compute p-values:
          * p overall R²
          * p overall RMSE
          * p per-feature R² (shape: n_conditions x n_features)

    Returns a dict with:
      real metrics, and p-values for overall and per-feature R².
    """
    rng = np.random.default_rng(random_state)

    conditions = list(data.keys())
    n_conditions = len(conditions)
    n_strains, n_features = next(iter(data.values())).shape

    # Real metrics
    real_r2 = np.zeros(n_conditions)
    real_rmse = np.zeros(n_conditions)
    real_r2_feat = np.zeros((n_conditions, n_features))

    # P-values
    p_r2 = np.zeros(n_conditions)
    p_rmse = np.zeros(n_conditions)
    p_r2_feat = np.zeros((n_conditions, n_features))

    # We’ll compute p-values by counting how often null >= real (for R²)
    # and null <= real (for RMSE)
    for ci, holdout in enumerate(conditions):
        train_conditions = [c for c in conditions if c != holdout]

        X = np.hstack([data[c] for c in train_conditions])  # (n_strains, 22*(C-1))
        y = data[holdout]                                   # (n_strains, 22)

        # ---- Real ----
        r2o_real, rmse_real, r2f_real = _cv_predict_metrics(
            X, y, alpha=alpha, n_splits=n_splits, random_state=random_state
        )
        real_r2[ci] = r2o_real
        real_rmse[ci] = rmse_real
        real_r2_feat[ci] = r2f_real

        # ---- Null counts (memory-efficient) ----
        count_r2_overall = 0
        count_rmse_overall = 0
        count_r2_feat = np.zeros(n_features, dtype=int)

        for _ in range(n_permutations):
            perm = rng.permutation(n_strains)
            y_perm = y[perm]

            r2o_null, rmse_null, r2f_null = _cv_predict_metrics(
                X, y_perm, alpha=alpha, n_splits=n_splits, random_state=random_state
            )

            # R²: bigger is better → count null >= real
            if r2o_null >= r2o_real:
                count_r2_overall += 1

            # RMSE: smaller is better → count null <= real
            if rmse_null <= rmse_real:
                count_rmse_overall += 1

            count_r2_feat += (r2f_null >= r2f_real)

        # Unbiased permutation p-values with +1 correction
        p_r2[ci] = (1 + count_r2_overall) / (n_permutations + 1)
        p_rmse[ci] = (1 + count_rmse_overall) / (n_permutations + 1)
        p_r2_feat[ci] = (1 + count_r2_feat) / (n_permutations + 1)

    return {
        "conditions": conditions,
        "real": {
            "r2_overall": real_r2,
            "rmse_overall": real_rmse,
            "r2_per_feature": real_r2_feat
        },
        "p_values": {
            "r2_overall": p_r2,
            "rmse_overall": p_rmse,
            "r2_per_feature": p_r2_feat
        },
        "settings": {
            "alpha": alpha,
            "n_splits": n_splits,
            "n_permutations": n_permutations,
            "random_state": random_state
        }
    }



def _cv_predict_metrics(X, y, alpha=1.0, n_splits=5, random_state=42):
    """
    Strain-level KFold CV: each strain is predicted out-of-sample exactly once.
    Returns:
      - overall R² (variance-weighted across outputs)
      - overall RMSE
      - per-feature R² (length n_features)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    r2_overall = r2_score(y_true_all, y_pred_all, multioutput="variance_weighted")
    rmse_overall = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

    r2_per_feature = np.array([
        r2_score(y_true_all[:, j], y_pred_all[:, j])
        for j in range(y.shape[1])
    ])

    return r2_overall, rmse_overall, r2_per_feature


def condition_holdout_permutation_test_featurewise(
    data,
    alpha=1.0,
    n_splits=5,
    n_permutations=500,
    random_state=42
):
    """
    For each holdout condition:
      - X = concat(features from all other conditions)
      - y = features for holdout condition
      - Compute real CV metrics
      - Permute y across strains, recompute metrics, and compute p-values:
          * p overall R²
          * p overall RMSE
          * p per-feature R² (shape: n_conditions x n_features)

    Returns a dict with:
      real metrics, and p-values for overall and per-feature R².
    """
    rng = np.random.default_rng(random_state)

    conditions = list(data.keys())
    n_conditions = len(conditions)
    n_strains, n_features = next(iter(data.values())).shape

    # Real metrics
    real_r2 = np.zeros(n_conditions)
    real_rmse = np.zeros(n_conditions)
    real_r2_feat = np.zeros((n_conditions, n_features))

    # P-values
    p_r2 = np.zeros(n_conditions)
    p_rmse = np.zeros(n_conditions)
    p_r2_feat = np.zeros((n_conditions, n_features))

    # We’ll compute p-values by counting how often null >= real (for R²)
    # and null <= real (for RMSE)
    for ci, holdout in enumerate(conditions):
        train_conditions = [c for c in conditions if c != holdout]

        X = np.hstack([data[c] for c in train_conditions])  # (n_strains, 22*(C-1))
        y = data[holdout]                                   # (n_strains, 22)

        # ---- Real ----
        r2o_real, rmse_real, r2f_real = _cv_predict_metrics(
            X, y, alpha=alpha, n_splits=n_splits, random_state=random_state
        )
        real_r2[ci] = r2o_real
        real_rmse[ci] = rmse_real
        real_r2_feat[ci] = r2f_real

        # ---- Null counts (memory-efficient) ----
        count_r2_overall = 0
        count_rmse_overall = 0
        count_r2_feat = np.zeros(n_features, dtype=int)

        for _ in range(n_permutations):
            perm = rng.permutation(n_strains)
            y_perm = y[perm]

            r2o_null, rmse_null, r2f_null = _cv_predict_metrics(
                X, y_perm, alpha=alpha, n_splits=n_splits, random_state=random_state
            )

            # R²: bigger is better → count null >= real
            if r2o_null >= r2o_real:
                count_r2_overall += 1

            # RMSE: smaller is better → count null <= real
            if rmse_null <= rmse_real:
                count_rmse_overall += 1

            count_r2_feat += (r2f_null >= r2f_real)

        # Unbiased permutation p-values with +1 correction
        p_r2[ci] = (1 + count_r2_overall) / (n_permutations + 1)
        p_rmse[ci] = (1 + count_rmse_overall) / (n_permutations + 1)
        p_r2_feat[ci] = (1 + count_r2_feat) / (n_permutations + 1)

    return {
        "conditions": conditions,
        "real": {
            "r2_overall": real_r2,
            "rmse_overall": real_rmse,
            "r2_per_feature": real_r2_feat
        },
        "p_values": {
            "r2_overall": p_r2,
            "rmse_overall": p_rmse,
            "r2_per_feature": p_r2_feat
        },
        "settings": {
            "alpha": alpha,
            "n_splits": n_splits,
            "n_permutations": n_permutations,
            "random_state": random_state
        }
    }



def _cv_predict_metrics(X, y, alpha=10, n_splits=5, random_state=42):
    """
    Strain-level KFold CV producing out-of-sample predictions for every strain exactly once.
    Returns overall R² (variance-weighted), RMSE, and per-feature R².
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    # Recommended for multioutput: variance-weighted overall R²
    r2_overall = r2_score(y_true_all, y_pred_all, multioutput="variance_weighted")
    rmse_overall = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

    r2_per_feature = np.array([
        r2_score(y_true_all[:, j], y_pred_all[:, j])
        for j in range(y.shape[1])
    ])

    return r2_overall, rmse_overall, r2_per_feature


def condition_holdout_permutation_test(
    data,
    alpha=10,
    n_splits=5,
    n_permutations=200,
    random_state=42
):
    """
    For each holdout condition:
      - Train on all other conditions (inputs concatenated)
      - Predict holdout condition (24-d target)
      - Compute real CV metrics
      - Compute null distribution by permuting y across strains
    Returns a dict with real metrics, null distributions, and permutation p-values.
    """
    rng = np.random.default_rng(random_state)

    conditions = list(data.keys())
    n_conditions = len(conditions)
    n_strains, n_features = next(iter(data.values())).shape

    # Results containers
    real_r2 = np.zeros(n_conditions)
    real_rmse = np.zeros(n_conditions)
    real_r2_feat = np.zeros((n_conditions, n_features))

    null_r2 = np.zeros((n_conditions, n_permutations))
    null_rmse = np.zeros((n_conditions, n_permutations))
    # (optional) null per-feature is large; keep if you want
    # null_r2_feat = np.zeros((n_conditions, n_permutations, n_features))

    p_r2 = np.zeros(n_conditions)
    p_rmse = np.zeros(n_conditions)

    for ci, holdout in enumerate(conditions):
        train_conditions = [c for c in conditions if c != holdout]

        X = np.hstack([data[c] for c in train_conditions])     # (n_strains, 22*(C-1))
        y = data[holdout]                                      # (n_strains, 22)

        # --- Real CV ---
        r2o, rmseo, r2f = _cv_predict_metrics(
            X, y, alpha=alpha, n_splits=n_splits, random_state=random_state
        )
        real_r2[ci] = r2o
        real_rmse[ci] = rmseo
        real_r2_feat[ci] = r2f

        # --- Null via permutation of y rows ---
        for p in range(n_permutations):
            perm = rng.permutation(n_strains)
            y_perm = y[perm]

            r2o_p, rmseo_p, _ = _cv_predict_metrics(
                X, y_perm, alpha=alpha, n_splits=n_splits, random_state=random_state
            )
            null_r2[ci, p] = r2o_p
            null_rmse[ci, p] = rmseo_p

        # Permutation p-values (one-sided, “is real better than null?”)
        # Add +1 to numerator/denominator for an unbiased finite-sample estimate.
        p_r2[ci] = (1 + np.sum(null_r2[ci] >= real_r2[ci])) / (n_permutations + 1)
        p_rmse[ci] = (1 + np.sum(null_rmse[ci] <= real_rmse[ci])) / (n_permutations + 1)

    return {
        "conditions": conditions,
        "real": {
            "r2_overall": real_r2,
            "rmse_overall": real_rmse,
            "r2_per_feature": real_r2_feat,
        },
        "null": {
            "r2_overall": null_r2,
            "rmse_overall": null_rmse,
        },
        "p_values": {
            "r2_overall": p_r2,
            "rmse_overall": p_rmse,
        }
    }







def showcase_clone_across_experimental_conditions(zroots,clone,experiments,fov=0,experiment_labels=['1','2','3','4','5','6'],show_labels=False,fontsize=48):

    figtitle='images_across_experiments_clone_' + str(clone) + '_fov' + str(fov)
    
    fig,ax=plt.subplots(2,3,layout='tight')
    fig.set_size_inches(66,44)
    ax=ax.flatten()
    
    
    for i, e in zip(range(len(experiments)),experiments):
    
    
            label_arr_bf=copy.deepcopy(zroots[e].get_label_image(fov, clone, "BF"))
            mask_arr_bf = np.where(label_arr_bf != 0, 1, 0)
        
            label_arr_cy5=copy.deepcopy(zroots[e].get_label_image(fov, clone, "CY5"))
            mask_arr_cy5 = np.where(label_arr_cy5 != 0, 1, 0)
    
            ax[i].imshow(np.multiply(zroots[e].get_processed_intensities(fov, clone, "BF"),mask_arr_bf),cmap="apc:sunset",alpha=.8)
            ax[i].imshow(np.multiply(zroots[e].get_processed_intensities(fov, clone, "CY5"),mask_arr_cy5),cmap="apc:sages_r",vmin=0,vmax=1,alpha=.8)
            ax[i].set_xticks([]) 
            ax[i].set_yticks([])
            ax[i].set_title(experiment_labels[e], fontsize=fontsize)
    
            if show_labels:
                try:
                    outlines_cy5=zroots[e].get_cell_outlines(fov, clone, "CY5")
                    for labels in range(len(outlines_cy5)):
                        ax[i].text(x=outlines_cy5[labels][0][0],
                                   y=outlines_cy5[labels][0][1],
                                   s=str(labels),
                                   size=8,
                                   c='red')
                except:
                    pass
        
        
        
                try:
                    outlines_bf=zroots[e].get_cell_outlines(fov, clone, "BF")
                    for labels in range(len(outlines_bf)):
                        ax[i].text(x=outlines_bf[labels][0][0],
                                   y=outlines_bf[labels][0][1],
                                   s=str(labels),
                                   size=8,
                                   c='yellow')
                except:
                    pass

    fig.savefig('chlorella_pub_fig_' + figtitle + '.svg',bbox_inches='tight')
    fig.savefig('chlorella_pub_fig_' + figtitle + '.pdf',bbox_inches='tight')
    fig.savefig('chlorella_pub_fig_' + figtitle + '.png',bbox_inches='tight')

    
    


def boxplot_two_phase_temps_with_connections(
    zroots,
    dead_ids,
    cellpose_feature='area',
    non_cellpose_feature='',
    channel="BF",
    n_fovs=9,
    ax=None,
    phase_labels=("Early stationary phase", "Late stationary phase"),
    temp_labels=("16°C", "22°C", "30°C"),
    colors_early=("lightblue", "grey", "pink"),
    colors_late=("blue", "black", "red"),
    show_points=True,
    point_alpha=0.7,
    jitter=0.08,
    box_alpha=0.35,
    widths=0.6,
    title=None,
    ylabel="Value",
    connect_points=True,
    line_alpha=0.35,
    line_width=1.0,
    figsize=(10, 5),
    ylim=[500,1800],
    figtitle='feat_change_with_temp_across_time'
):
    """
    data6: list of 6 lists/arrays
      [early_16, early_22, early_30, late_16, late_22, late_30]

    IMPORTANT: For connection lines to make sense, within each phase
    the three lists must have the same length and be aligned by index
    (replicate i at 16C is the same replicate i at 22C and 30C).
    """

    if non_cellpose_feature=='':

        dist_of_clone_mean_feat=[]
        for e in range(len(zroots)):
            stats=np.empty((n_fovs*96))
            stats.fill(np.nan)
            for pov in range(n_fovs):
                for well in range(96):
                    if well not in dead_ids:
                        dat=np.array(zroots[e].get_cell_properties(pov, well, channel)[cellpose_feature].astype('float32'))
                        if dat.any():
                            stats[well+(n_fovs*pov)]=np.nanmean(dat)
            dist_of_clone_mean_feat.append(stats)


    if non_cellpose_feature=='pixel_intensity':

        dist_of_clone_mean_feat=[]
        for e in range(len(zroots)):
            stats=np.empty((n_fovs*96))
            stats.fill(np.nan)
            for pov in range(n_fovs):
                for well in range(96):
                    if well not in dead_ids:

                        try:
                            label_arr_cy5=copy.deepcopy(zroots[e].get_label_image(pov, well, channel))
                            mask_arr_cy5 = np.where(label_arr_cy5 != 0, 1, 0)
                            test=np.multiply(zroots[e].get_processed_intensities(pov, well, channel),mask_arr_cy5)
                            stats[well+(n_fovs*pov)]=np.sum(test)/np.sum(mask_arr_cy5)
                        except:
                            pass

            dist_of_clone_mean_feat.append(stats)
      




    if len(zroots) != 6:
        raise ValueError(f"Expected data6 to have length 6, got {len(zroots)}")

    # Convert to arrays (allow lists)
    arrs = [np.asarray(d) for d in dist_of_clone_mean_feat]

    # Validate pairing lengths within each phase if connecting
    if connect_points:
        n_e = [len(arrs[i]) for i in (0, 1, 2)]
        n_l = [len(arrs[i]) for i in (3, 4, 5)]
        if len(set(n_e)) != 1:
            raise ValueError(
                f"Early phase lists must have equal length to connect points, got {n_e}."
            )
        if len(set(n_l)) != 1:
            raise ValueError(
                f"Late phase lists must have equal length to connect points, got {n_l}."
            )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # positions with a gap between phases
    pos_early = np.array([1, 2, 3])
    pos_late  = np.array([5, 6, 7])
    positions = np.concatenate([pos_early, pos_late])

    box_colors = list(colors_early) + list(colors_late)

    bp = ax.boxplot(
        arrs,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # Color boxes + lines to match
    for i, (box, c) in enumerate(zip(bp["boxes"], box_colors)):
        box.set_facecolor(c)
        box.set_alpha(box_alpha)
        box.set_edgecolor(c)

        bp["medians"][i].set_color(c)
        bp["whiskers"][2*i].set_color(c)
        bp["whiskers"][2*i + 1].set_color(c)
        bp["caps"][2*i].set_color(c)
        bp["caps"][2*i + 1].set_color(c)

    # Optional jittered points (store x-positions per point so lines can match the same jitter)
    rng = np.random.default_rng(0)
    jittered_x = [None] * 6

    if show_points:
        for idx, (x, ys, c) in enumerate(zip(positions, arrs, box_colors)):
            ys = np.asarray(ys)
            xs = x + rng.uniform(-jitter, jitter, size=len(ys))
            jittered_x[idx] = xs
            ax.scatter(xs, ys, s=18, alpha=point_alpha, color=c, edgecolors="none")
    else:
        # If no points drawn, still create deterministic x positions for lines
        for idx, (x, ys) in enumerate(zip(positions, arrs)):
            ys = np.asarray(ys)
            jittered_x[idx] = np.full(len(ys), x, dtype=float)

    # Connection lines within each phase: (0->1->2) and (3->4->5)
    if connect_points:
        # Early
        for i in range(len(arrs[0])):
            y0, y1, y2 = arrs[0][i], arrs[1][i], arrs[2][i]
            if np.any(np.isnan([y0, y1, y2])):
                continue
            xs = [jittered_x[0][i], jittered_x[1][i], jittered_x[2][i]]
            ys = [y0, y1, y2]
            ax.plot(xs, ys, color="k", alpha=line_alpha, linewidth=line_width)

        # Late
        for i in range(len(arrs[3])):
            y0, y1, y2 = arrs[3][i], arrs[4][i], arrs[5][i]
            if np.any(np.isnan([y0, y1, y2])):
                continue
            xs = [jittered_x[3][i], jittered_x[4][i], jittered_x[5][i]]
            ys = [y0, y1, y2]
            ax.plot(xs, ys, color="k", alpha=line_alpha, linewidth=line_width)

    # Bottom x-axis: temperature labels
    ax.set_xticks(positions)
    ax.set_xticklabels(list(temp_labels) + list(temp_labels))
    ax.tick_params(axis="x", pad=2)

    # Second bottom x-axis: phase labels
    ax_phase = ax.secondary_xaxis("bottom")
    ax_phase.set_xticks([pos_early.mean(), pos_late.mean()])
    ax_phase.set_xticklabels(phase_labels)
    ax_phase.spines["bottom"].set_visible(False)
    ax_phase.tick_params(axis="x", length=0, pad=22)

    # Divider between phases
    ax.axvline(4, linewidth=.1, alpha=0.1)

    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(ylim[0], ylim[1])
    #ax.set_xlabel("Temperature")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    
    plt.show()

    fig.tight_layout()
    fig.savefig('chlorella_pub_fig_' + figtitle + '.svg',bbox_inches='tight')
    fig.savefig('chlorella_pub_fig_' + figtitle + '.pdf',bbox_inches='tight')
    fig.savefig('chlorella_pub_fig_' + figtitle + '.png',bbox_inches='tight')
    return ax

def showcase_clone_cp_features(
    zroots,
    experiment=0,
    field_of_view=0,
    well=0,
    experiment_labels=['1','2','3','4','5'],
    f_size=[50, 50],

    # Outer layout: left block vs bar column
    w_ratios=[3, 2],            # [left_block_width, bar_column_width]
    outer_wspace=0.15,          # spacing between left block and bar column
    outer_hspace=0.4,           # spacing between the two rows of the OUTER grid

    # Image-block-only spacing (NEW)
    img_wspace=0.02,
    img_hspace=0.02,

    img_titles=False,
    bar_titles=False,
    sup_title=True,
    sup_title_params=[.36, .94, 24],
):
    figtitle = (
        'cellpose_feature_outputs_experiment_' + str(experiment) +
        '_fov' + str(field_of_view) +
        '_clone' + str(well)
    )
    plot_title = experiment_labels[experiment] + ' FoV_' + str(field_of_view) + ' Clone_' + str(well)

    # --- Cellpose features ---
    clone_cp_feats_bf = zroots[experiment].get_cell_properties(field_of_view, well, "BF")
    clone_cp_feats_bf_mean = np.nanmean(zscore(clone_cp_feats_bf), axis=0)
    cellpose_features = clone_cp_feats_bf.columns.to_list()

    clone_cp_feats_cy5 = zroots[experiment].get_cell_properties(field_of_view, well, "CY5")
    clone_cp_feats_cy5_mean = np.nanmean(zscore(clone_cp_feats_cy5), axis=0)

    cellpose_features[0] = 'count'

    # --- Images ---
    img1 = zroots[experiment].get_label_image(field_of_view, well, "BF")
    img2 = zroots[experiment].get_raw_intensities(field_of_view, well, "BF")
    img3 = zroots[experiment].get_processed_intensities(field_of_view, well, "BF")

    img4 = zroots[experiment].get_label_image(field_of_view, well, "CY5")
    img5 = zroots[experiment].get_raw_intensities(field_of_view, well, "CY5")
    img6 = zroots[experiment].get_processed_intensities(field_of_view, well, "CY5")

    # --- Figure + OUTER gridspec (2 rows x 2 cols: left image block, right barplots) ---
    fig = plt.figure(figsize=f_size)

    outer = fig.add_gridspec(
        2, 2,
        width_ratios=w_ratios,   # left block vs bar column
        wspace=outer_wspace,
        hspace=outer_hspace
    )

    # LEFT: nested gridspec for 2x3 images, with its own spacing
    left_top = outer[0, 0].subgridspec(1, 3, wspace=img_wspace)
    left_bot = outer[1, 0].subgridspec(1, 3, wspace=img_wspace)

    # If you also want independent vertical spacing *within* the left block,
    # we do it by making a 2x3 subgridspec instead:
    left = outer[:, 0].subgridspec(2, 3, wspace=img_wspace, hspace=img_hspace)

    axes_img = [
        fig.add_subplot(left[0, 0]),
        fig.add_subplot(left[0, 1]),
        fig.add_subplot(left[0, 2]),
        fig.add_subplot(left[1, 0]),
        fig.add_subplot(left[1, 1]),
        fig.add_subplot(left[1, 2]),
    ]

    # RIGHT: barplots (one per row)
    axes_bar = [
        fig.add_subplot(outer[0, 1]),
        fig.add_subplot(outer[1, 1]),
    ]

    # --- Plot images ---
    titles = [
        "Label Image (BF)", "Raw Intensities (BF)", "Processed Intensities (BF)",
        "Label Image (CY5)", "Raw Intensities (CY5)", "Processed Intensities (CY5)"
    ]

    for ax, img, title in zip(axes_img, [img1, img2, img3, img4, img5, img6], titles):
        ax.imshow(img)
        if img_titles:
            ax.set_title(title)
        ax.axis("off")

    # --- Plot bar charts ---
    axes_bar[0].bar(cellpose_features, clone_cp_feats_bf_mean, color=apc.aster)
    axes_bar[0].set_xticklabels([])
    if bar_titles:
        axes_bar[0].set_title("CellPose Features (BF)")
    axes_bar[0].set_ylabel("mean z-score")

    axes_bar[1].bar(cellpose_features, clone_cp_feats_cy5_mean, color=apc.aster)
    axes_bar[1].set_xticklabels(cellpose_features, rotation=45, ha="right")
    if bar_titles:
        axes_bar[1].set_title("CellPose Features (CY5)")
    axes_bar[1].set_ylabel("mean z-score")

    # --- Suptitle ---
    if sup_title:
        fig.suptitle(
            plot_title,
            x=sup_title_params[0],
            y=sup_title_params[1],
            fontsize=sup_title_params[2]
        )
    plt.show()

    fig.savefig('chlorella_pub_fig_' + figtitle + '.svg', bbox_inches='tight')
    fig.savefig('chlorella_pub_fig_' + figtitle + '.pdf', bbox_inches='tight')
    fig.savefig('chlorella_pub_fig_' + figtitle + '.png', bbox_inches='tight')



def cluster_by_feats(zroots,clone,exp,fov,channel,feat_id_range,fig_size=(10,10),figtitle='',col_map="apc:oranges_r"):
    
    # cluster cells within an image by feature
    moments_df=zroots[exp].get_cell_properties(fov, clone, "CY5").iloc[:, feat_id_range[0]:feat_id_range[1]]
    
    # zscore before taking cosing distance
    z_moments=zscore(moments_df.to_numpy(), axis=0)
    cosine_dist_of_moments=np.zeros((len(moments_df),len(moments_df)))
    
    for i in range(len(moments_df)):
        for j in range(len(moments_df)):
    
            m1=z_moments[i]
            m2=z_moments[j]
            cosine_dist_of_moments[i,j]=nan_cosine_distance(m1,m2)

    # convert square distance matrix to condensed form
    condensed_dist = squareform(cosine_dist_of_moments, checks=False)

    # build linkage from the PRECOMPUTED distances
    #row_linkage = linkage(condensed_dist, method="average")
    row_linkage = linkage(squareform(cosine_dist_of_moments, checks=False), method="average")


    g=sns.clustermap(cosine_dist_of_moments,
                     row_linkage=row_linkage,
                     col_linkage=row_linkage,
                     figsize=fig_size,
                     yticklabels=False,
                     xticklabels=False,
                     cmap=col_map,
                     cbar=False,
                     cbar_pos=[1, 0.003, 0.01, .796])
    
    
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])
    
    
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    
    
    plt.show()
    
    
    g.savefig('chlorella_pub_fig_' + figtitle + '.svg',bbox_inches='tight',pad_inches=0)
    g.savefig('chlorella_pub_fig_' + figtitle + '.pdf',bbox_inches='tight',pad_inches=0)
    g.savefig('chlorella_pub_fig_' + figtitle + '.png',bbox_inches='tight',pad_inches=0)

    return g,row_linkage


def plot_tsne(input_dat,
              exp_embedding,
              exp_colors,
              n_fovs,
              dead_well_ids,
              figtitle,
              tsne_alpha=.9,
              feat_2_color_by=-1,
              feat_cmap="apc:verde",
              feat_alpha=.7):

    
    
    scatter_colors=[]
    for b in range(len(exp_colors)):
        for l in range(n_fovs*96):
            if l not in dead_well_ids:
                scatter_colors.append(exp_colors[b])
    
    
            
    fig,ax=plt.subplots()
    fig.set_size_inches(7,7)

    if feat_2_color_by < 0:
        plt.scatter(np.array(exp_embedding)[:,0],np.array(exp_embedding)[:,1],c=scatter_colors,alpha=tsne_alpha)

    if feat_2_color_by > -1:
        plt.scatter(np.array(exp_embedding)[:,0],
                    np.array(exp_embedding)[:,1],
                    c=zscore(input_dat[:,feat_2_color_by]),
                    cmap=feat_cmap,
                    alpha=feat_alpha,
                    vmax=pd.Series(zscore(input_dat[:,feat_2_color_by])).quantile(.99),
                    vmin=pd.Series(zscore(input_dat[:,feat_2_color_by])).quantile(.01))

    
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    fig.savefig('chlorella_pub_fig_' + figtitle + '.svg',bbox_inches='tight')
    fig.savefig('chlorella_pub_fig_' + figtitle + '.pdf',bbox_inches='tight')
    fig.savefig('chlorella_pub_fig_' + figtitle + '.png',bbox_inches='tight')

    return pd.Series(zscore(input_dat[:,feat_2_color_by])).quantile(.95)
    



def cross_validation_and_alpha_scan(training_dat,
                                    exp_colors,
                                    exp_labels,
                                    figtitle='scanning_for_alpha_cross_validation',
                                    fig_size=(8, 5),
                                    alphas_2_try=[.01,1,100],
                                    tv_splits=96,
                                    n_permuts=5,  #5 for test, 200 for quick(ish), 1000 if you can afford
                                    rand_state=42,
                                    show_legend=False,
                                    
    
):


    r_2s=[]

    for a in alphas_2_try:
    
        res = condition_holdout_permutation_test(training_dat,
                                                 alpha=a,
                                                 n_splits=tv_splits,
                                                 n_permutations=n_permuts, # 200 is ok for quick, 1000 is nicer if you can afford it
                                                 random_state=rand_state)
        r_2s.append(res)

    all_rs=[]
    all_ps=[]
    for r in r_2s:
        rs=[]
        ps=[]
        for c, r2, p in zip(r["conditions"], r["real"]["r2_overall"], r["p_values"]["r2_overall"]):
            rs.append(r2)
            ps.append(p)
        all_rs.append(rs)
        all_ps.append(ps)


    # X-axis values
    x_labels = alphas_2_try
    x = np.arange(len(x_labels))
    
    
    fig,ax=plt.subplots(figsize=fig_size)
    
    for rs, color, label in zip(np.array(all_rs).T, exp_colors, exp_labels):
        plt.plot(x, rs, marker="o", color=color, linewidth=2, label=label)
    
    if show_legend:
        plt.legend(frameon=False)
    
    # X-axis ticks + labels
    plt.xticks(x, x_labels)
    
    # Labels and title
    plt.xlabel("alpha")
    plt.ylabel("R²")
    #plt.title("Cross Validation Alpha Scan")
    
    # Optional: nicer grid
    # plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    fig.savefig('chlorella_pub_fig_' + figtitle + '.svg',bbox_inches='tight')
    fig.savefig('chlorella_pub_fig_' + figtitle + '.pdf',bbox_inches='tight')
    fig.savefig('chlorella_pub_fig_' + figtitle + '.png',bbox_inches='tight')

    return all_rs,all_ps

    
def showcase_clone_images_1_by_1(
    zroots,
    experiment=0,
    field_of_view=0,
    well=0,
    f_size=[5, 5],
    f_size_2=[5, 5],
    show_labels=True,
    marker_size_1=8,
    marker_size_2=8,
    marker_color_bf='yellow',
    marker_color_cy='red',

):

    # --- Cellpose features ---
    clone_cp_feats_bf = zroots[experiment].get_cell_properties(field_of_view, well, "BF")
    clone_cp_feats_bf_mean = np.nanmean(zscore(clone_cp_feats_bf), axis=0)

    clone_cp_feats_cy5 = zroots[experiment].get_cell_properties(field_of_view, well, "CY5")
    clone_cp_feats_cy5_mean = np.nanmean(zscore(clone_cp_feats_cy5), axis=0)


    # --- Images ---

    img1 = zroots[experiment].get_label_image(field_of_view, well, "BF")
    img2 = zroots[experiment].get_raw_intensities(field_of_view, well, "BF")
    img3 = zroots[experiment].get_processed_intensities(field_of_view, well, "BF")

    img4 = zroots[experiment].get_label_image(field_of_view, well, "CY5")
    img5 = zroots[experiment].get_raw_intensities(field_of_view, well, "CY5")
    img6 = zroots[experiment].get_processed_intensities(field_of_view, well, "CY5")

    ims=[img1,img2,img3,img4,img5,img6]

    genroot='exp_' + str(experiment) + '_strain_' + str(well) + '_fov_' + str(field_of_view)

    bfroot='exp_' + str(experiment) + '_strain_' + str(well) + '_fov_' + str(field_of_view) + '_BF_'
    cy5root='exp_' + str(experiment) + '_strain_' + str(well) + '_fov_' + str(field_of_view) + '_CY5_'
    plt_names_roots=[bfroot+'label_image',
                     bfroot+'raw_intensities',
                     bfroot+'processed_intensities',
                     cy5root+'label_image',
                     cy5root+'raw_intensities',
                     cy5root+'processed_intensities']

        
    for i in range(len(ims)):
        fig, ax = plt.subplots(figsize=(f_size[0], f_size[1]))

        if i == 0:

            ax.imshow(ims[i])

            if show_labels:
            
                try:
                    outlines_bf=zroots[experiment].get_cell_outlines(field_of_view, well, "BF")
                    for labels in range(len(outlines_bf)):
                        ax.text(x=outlines_bf[labels][0][0],
                                    y=outlines_bf[labels][0][1],
                                    s=str(labels),
                                    size=marker_size_1,
                                    c=marker_color_bf)
                except:
                    pass

            ax.axis('off')

            fig.savefig(
                plt_names_roots[i] + '.png',
                bbox_inches='tight',
                pad_inches=0
            )
            fig.savefig(
                plt_names_roots[i] + '.pdf',
                bbox_inches='tight',
                pad_inches=0
            )
            fig.savefig(
                plt_names_roots[i] + '.svg',
                bbox_inches='tight',
                pad_inches=0
            )

        if i == 3:

            ax.imshow(ims[i])

            if show_labels:

                try:
                    outlines_cy5=zroots[experiment].get_cell_outlines(field_of_view, well, "CY5")
                    for labels in range(len(outlines_cy5)):
                        ax.text(x=outlines_cy5[labels][0][0],
                                    y=outlines_cy5[labels][0][1],
                                    s=str(labels),
                                    size=marker_size_1,
                                    c=marker_color_cy)
                except:
                    pass

            ax.axis('off')

            fig.savefig(
                plt_names_roots[i] + '.png',
                bbox_inches='tight',
                pad_inches=0
            )
            fig.savefig(
                plt_names_roots[i] + '.pdf',
                bbox_inches='tight',
                pad_inches=0
            )
            fig.savefig(
                plt_names_roots[i] + '.svg',
                bbox_inches='tight',
                pad_inches=0
            )

        if i not in [0,3]:

            ax.imshow(ims[i])
            ax.axis('off')

            fig.savefig(
                plt_names_roots[i] + '.png',
                bbox_inches='tight',
                pad_inches=0
            )
            fig.savefig(
                plt_names_roots[i] + '.pdf',
                bbox_inches='tight',
                pad_inches=0
            )
            fig.savefig(
                plt_names_roots[i] + '.svg',
                bbox_inches='tight',
                pad_inches=0
            )

        plt.show()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(f_size_2[0], f_size_2[1]))


    label_arr_bf=copy.deepcopy(zroots[experiment].get_label_image(field_of_view, well, "BF"))
    mask_arr_bf = np.where(label_arr_bf != 0, 1, 0)

    label_arr_cy5=copy.deepcopy(zroots[experiment].get_label_image(field_of_view, well, "CY5"))
    mask_arr_cy5 = np.where(label_arr_cy5 != 0, 1, 0)


    ax.imshow(np.multiply(zroots[experiment].get_processed_intensities(field_of_view, well, "BF"),mask_arr_bf),cmap="apc:sunset",alpha=.8)
    ax.imshow(np.multiply(zroots[experiment].get_processed_intensities(field_of_view, well, "CY5"),mask_arr_cy5),cmap="apc:sages_r",vmin=0,vmax=1,alpha=.8)
    ax.set_xticks([]) 
    ax.set_yticks([])

    if show_labels:
        try:
            outlines_cy5=zroots[experiment].get_cell_outlines(field_of_view, well, "CY5")
            for labels in range(len(outlines_cy5)):
                ax.text(x=outlines_cy5[labels][0][0],
                            y=outlines_cy5[labels][0][1],
                            s=str(labels),
                            size=marker_size_2,
                            c=marker_color_cy)
        except:
            pass



        try:
            outlines_bf=zroots[experiment].get_cell_outlines(field_of_view, well, "BF")
            for labels in range(len(outlines_bf)):
                ax.text(x=outlines_bf[labels][0][0],
                            y=outlines_bf[labels][0][1],
                            s=str(labels),
                            size=marker_size_2,
                            c=marker_color_bf)
        except:
            pass

    fig.savefig(
    genroot + 'merged_image' + '.png',
    bbox_inches='tight',
    pad_inches=0)
    fig.savefig(
    genroot + 'merged_image' + '.pdf',
    bbox_inches='tight',
    pad_inches=0)
    fig.savefig(
    genroot + 'merged_image' + '.svg',
    bbox_inches='tight',
    pad_inches=0)

    plt.show()
    plt.close(fig)




def plot_sorted_r2_per_feature(zroots,
                               ridge_model_residulals,
                               xtick_colors=[apc.vital, 
                                             apc.wish, 
                                             apc.dress, 
                                             apc.lapis, 
                                             apc.aster, 
                                             apc.candy],
                               col_titles=['16C e.stat', 
                                           '22C e.stat', 
                                           '30C e.stat', 
                                           '16C l.stat',
                                           '22C l.stat', 
                                           '30C l.stat'],
                               figsize=[60,8],
                               ):

    cellpose_features=zroots[0].get_cell_properties(0, 0, "BF").columns.tolist()
    cellpose_features[0]='density'
    training_features_labels=[]
    for l in range(44):
        if l<22:
            training_features_labels.append(cellpose_features[l]+str( (' (bf)')))     
        if l>=22:
            training_features_labels.append(cellpose_features[l-22]+str( (' (cy5)')))
                               
    # for visualization, sort columns by the highest R2
    biggest_label=np.mean(ridge_model_residulals['real']['r2_per_feature'],axis=0)
    sorted_by_predictability=biggest_label.argsort()[::-1]
    sorted_res=[]
    
    for r in ridge_model_residulals['real']['r2_per_feature']:
        sorted_res.append(r[sorted_by_predictability])
    
    l_sorted=np.array(training_features_labels)[sorted_by_predictability]
    
    #plot
    fig,ax=plt.subplots()
    fig.set_size_inches(figsize[0],figsize[1])
    
        
    for i, tick_label in enumerate(ax.axes.get_xticklabels()):
        if i < len(xtick_colors):
            tick_label.set_color(xtick_colors[i])


    sns.heatmap(np.array(sorted_res).T,
                cmap="apc:verde",
                vmin=0,
                vmax=1,
                annot=True,
                xticklabels=np.array(col_titles),
                yticklabels=l_sorted
           )

    plt.show()
