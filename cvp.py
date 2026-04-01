#Imports

#plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import arcadia_pycolor as apc
font_path="~/Library/Fonts"
apc.mpl.setup(font_dirpath=font_path)
apc.plotly.setup()

#matrices
import numpy as np
import pandas as pd

#segmentation
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    def _get_array_from_possible_paths(self, possible_paths, b, im):
        for path in possible_paths:
            try:
                return self.root[path][b, im]
            except KeyError:
                continue
        raise KeyError(f"None of these dataset paths exist: {possible_paths}")

    def get_raw_intensities(self, b, im, channel_name="CY5"):
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

    genroot='plots/exp_' + str(experiment) + '_strain_' + str(well) + '_fov_' + str(field_of_view)

    bfroot='plots/exp_' + str(experiment) + '_strain_' + str(well) + '_fov_' + str(field_of_view) + '_BF_'
    cy5root='plots/exp_' + str(experiment) + '_strain_' + str(well) + '_fov_' + str(field_of_view) + '_CY5_'
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_single_clone_features(
    zroots,
    experiment=0,
    field_of_view=0,
    well=0,
    channel="BF",
    figsize=(8, 6),
    title=None,
    show_title=True,
    include_pixel_intensity=True,
    zscore_features=False,
    color=apc.aster,
    rotation=45,
    orientation="vertical",
    show_xticklabels=True,
    ylabel=None,
    axis_color="black",
    font_family=None,
    font_size=10,
    title_size=None,
    exclude_features=None,
    save=False,
    figtitle=None,
):
    """
    Plot mean feature values for a single clone as a bar plot.

    Parameters
    ----------
    exclude_features : list or None
        Feature names to exclude, e.g.
        ["moments_hu-4", "moments_hu-5", "moments_hu-6"]
    """

    if exclude_features is None:
        exclude_features = []

    # --- Load cell properties ---
    props = zroots[experiment].get_cell_properties(field_of_view, well, channel)
    if props is None or len(props) == 0:
        raise ValueError(
            f"No cell properties found for experiment={experiment}, "
            f"field_of_view={field_of_view}, well={well}, channel={channel}"
        )

    # rename first feature if needed
    feature_names = props.columns.to_list()
    if len(feature_names) > 0 and feature_names[0] == "label":
        feature_names[0] = "count"
        props = props.copy()
        props.columns = feature_names

    # --- Exclude requested features ---
    keep_features = [f for f in props.columns if f not in exclude_features]
    props = props[keep_features]

    # --- Compute mean feature values ---
    props_np = props.astype("float32").to_numpy()

    if zscore_features:
        props_np = zscore(props_np, axis=0, nan_policy="omit")

    feature_means = np.nanmean(props_np, axis=0)

    feature_df = pd.DataFrame({
        "feature": props.columns.tolist(),
        "value": feature_means
    })

    # --- Pixel intensity ---
    if include_pixel_intensity:
        try:
            label_arr = copy.deepcopy(
                zroots[experiment].get_label_image(field_of_view, well, channel)
            )
            mask_arr = np.where(label_arr != 0, 1, 0)
            proc_img = zroots[experiment].get_processed_intensities(
                field_of_view, well, channel
            )

            if np.sum(mask_arr) > 0:
                pixel_vals = proc_img[mask_arr.astype(bool)]
                if zscore_features:
                    pixel_vals = zscore(pixel_vals, nan_policy="omit")
                mean_pixel_intensity = np.nanmean(pixel_vals)
            else:
                mean_pixel_intensity = np.nan

            feature_df = pd.concat(
                [
                    feature_df,
                    pd.DataFrame(
                        {"feature": ["pixel_intensity"], "value": [mean_pixel_intensity]}
                    ),
                ],
                ignore_index=True,
            )
        except Exception as e:
            print(f"Warning: could not compute pixel intensity: {e}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(feature_df))

    if orientation == "vertical":
        ax.bar(x, feature_df["value"], color=color)
        ax.set_xticks(x)

        if show_xticklabels:
            ax.set_xticklabels(feature_df["feature"], rotation=rotation, ha="right")
        else:
            ax.set_xticklabels([])

        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(
                "Mean z-scored feature value" if zscore_features else "Mean feature value"
            )

    elif orientation == "horizontal":
        ax.barh(x, feature_df["value"], color=color)
        ax.set_yticks(x)

        if show_xticklabels:
            ax.set_yticklabels(feature_df["feature"])
        else:
            ax.set_yticklabels([])

        if ylabel is not None:
            ax.set_xlabel(ylabel)
        else:
            ax.set_xlabel(
                "Mean z-scored feature value" if zscore_features else "Mean feature value"
            )

    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    # --- Axis styling ---
    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis='x', colors=axis_color)
    ax.tick_params(axis='y', colors=axis_color)

    # --- Font styling ---
    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)

        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size)
    else:
        ax.tick_params(labelsize=font_size)
        ax.xaxis.label.set_size(font_size)
        ax.yaxis.label.set_size(font_size)

    # --- Title ---
    if show_title:
        if title is None:
            title = f"Exp {experiment} | FoV {field_of_view} | Clone {well} | {channel}"

        if font_family is not None:
            ax.set_title(title, fontfamily=font_family, fontsize=title_size or font_size)
        else:
            ax.set_title(title, fontsize=title_size or font_size)

    plt.tight_layout()
    plt.show()

    # --- Save ---
    if save:
        if figtitle is None:
            figtitle = (
                f"single_clone_features_exp_{experiment}"
                f"_fov_{field_of_view}"
                f"_clone_{well}"
                f"_{channel}"
            )

        fig.savefig("plots/"+figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/"+figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/"+figtitle + ".svg", bbox_inches="tight")

    return fig, ax, feature_df
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_feature_cv_across_conditions(
    zroots,
    channel="BF",
    n_wells=96,
    n_fovs=9,
    dead_ids=None,
    include_pixel_intensity=True,
    zscore_features=False,
    figsize=(10, 6),
    title=None,
    show_title=True,
    show_xticklabels=True,
    ylabel=None,
    orientation="vertical",
    rotation=45,
    bar_color=apc.aster,
    dot_colors=None,
    dot_size=40,
    dot_alpha=1.0,
    dot_edgecolor="none",
    axis_color="black",
    font_family=None,
    font_size=10,
    title_size=None,
    log_transform=None,   # None, "log10", or "log2"
    exclude_features=None,
    save=False,
    figtitle=None,
):
    """
    Plot the coefficient of variation (CV = std / mean) for each feature.

    CV is always converted to absolute value before averaging or plotting.

    Parameters
    ----------
    exclude_features : list or None
        Feature names to exclude, e.g.
        ["moments_hu-4", "moments_hu-5", "moments_hu-6"]
    """

    if dead_ids is None:
        dead_ids = []

    if exclude_features is None:
        exclude_features = []

    if dot_colors is None:
        dot_colors = [apc.vital, apc.wish, apc.dress, apc.lapis, apc.aster, apc.candy]

    if len(dot_colors) < len(zroots):
        raise ValueError(
            f"dot_colors has length {len(dot_colors)} but there are {len(zroots)} conditions."
        )

    # ----------------------------
    # CV helper (always positive)
    # ----------------------------
    def _cv(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return np.nan
        mean_val = np.nanmean(arr)
        std_val = np.nanstd(arr, ddof=1)
        if np.isclose(mean_val, 0):
            return np.nan
        return abs(std_val / mean_val)

    # ----------------------------
    # Feature names
    # ----------------------------
    reference_props = zroots[0].get_cell_properties(0, 0, channel)
    if reference_props is None or len(reference_props) == 0:
        raise ValueError("Could not determine feature names")

    feature_names = reference_props.columns.tolist()
    if len(feature_names) > 0 and feature_names[0] == "label":
        feature_names[0] = "count"

    feature_names = [f for f in feature_names if f not in exclude_features]

    n_conditions = len(zroots)

    feature_values_by_condition = {
        fname: [[] for _ in range(n_conditions)]
        for fname in feature_names
    }

    if include_pixel_intensity:
        feature_values_by_condition["pixel_intensity"] = [[] for _ in range(n_conditions)]

    # ----------------------------
    # Collect values
    # ----------------------------
    for cond_idx, z in enumerate(zroots):
        for well in range(n_wells):
            if well in dead_ids:
                continue

            per_fov_feature_means = []
            per_fov_pixel_means = []

            for fov in range(n_fovs):
                props = z.get_cell_properties(fov, well, channel)

                if props is None or len(props) == 0:
                    continue

                current_names = props.columns.tolist()
                if len(current_names) > 0 and current_names[0] == "label":
                    current_names[0] = "count"
                    props = props.copy()
                    props.columns = current_names

                props = props[feature_names]

                props_np = props.astype("float32").to_numpy()

                if zscore_features:
                    props_np = zscore(props_np, axis=0, nan_policy="omit")

                mean_vals = np.nanmean(props_np, axis=0)
                per_fov_feature_means.append(mean_vals)

                if include_pixel_intensity:
                    try:
                        label_arr = z.get_label_image(fov, well, channel)
                        mask = label_arr != 0
                        img = z.get_processed_intensities(fov, well, channel)

                        vals = img[mask]
                        if len(vals) > 0:
                            if zscore_features:
                                vals = zscore(vals, nan_policy="omit")
                            per_fov_pixel_means.append(np.nanmean(vals))
                    except:
                        pass

            if len(per_fov_feature_means) == 0:
                continue

            well_vals = np.nanmean(per_fov_feature_means, axis=0)

            for i, fname in enumerate(feature_names):
                feature_values_by_condition[fname][cond_idx].append(well_vals[i])

            if include_pixel_intensity and len(per_fov_pixel_means) > 0:
                feature_values_by_condition["pixel_intensity"][cond_idx].append(
                    np.nanmean(per_fov_pixel_means)
                )

    # ----------------------------
    # Compute CV
    # ----------------------------
    rows = []

    for feature, groups in feature_values_by_condition.items():
        cv_vals = np.array([_cv(g) for g in groups], dtype=float)
        mean_cv = np.nanmean(cv_vals)

        if log_transform is not None:
            eps = 1e-8
            if log_transform == "log10":
                cv_vals = np.log10(cv_vals + eps)
                mean_cv = np.log10(mean_cv + eps)
            elif log_transform == "log2":
                cv_vals = np.log2(cv_vals + eps)
                mean_cv = np.log2(mean_cv + eps)
            else:
                raise ValueError("log_transform must be None, 'log10', or 'log2'")

        row = {"feature": feature, "mean_cv": mean_cv}

        for i, val in enumerate(cv_vals):
            row[f"cv_condition_{i}"] = val

        rows.append(row)

    results_df = pd.DataFrame(rows)

    # ----------------------------
    # Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=figsize)
    pos = np.arange(len(results_df))
    labels = results_df["feature"].tolist()

    if orientation == "vertical":
        ax.bar(pos, results_df["mean_cv"], color=bar_color)

        for i in range(n_conditions):
            ax.scatter(
                pos,
                results_df[f"cv_condition_{i}"],
                color=dot_colors[i],
                s=dot_size,
                alpha=dot_alpha,
                edgecolors=dot_edgecolor,
                zorder=3,
            )

        ax.set_xticks(pos)
        if show_xticklabels:
            ax.set_xticklabels(labels, rotation=rotation, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels([""] * len(labels))

        ax.set_xlim(-0.6, len(labels) - 0.4)

        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            if log_transform == "log10":
                ax.set_ylabel("log10(CV)")
            elif log_transform == "log2":
                ax.set_ylabel("log2(CV)")
            else:
                ax.set_ylabel("Mean CV")

    else:
        ax.barh(pos, results_df["mean_cv"], color=bar_color)

        for i in range(n_conditions):
            ax.scatter(
                results_df[f"cv_condition_{i}"],
                pos,
                color=dot_colors[i],
                s=dot_size,
                alpha=dot_alpha,
                edgecolors=dot_edgecolor,
                zorder=3,
            )

        ax.set_yticks(pos)
        if show_xticklabels:
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([""] * len(labels))

        ax.set_ylim(-0.6, len(labels) - 0.4)

        if ylabel is not None:
            ax.set_xlabel(ylabel)
        else:
            if log_transform == "log10":
                ax.set_xlabel("log10(CV)")
            elif log_transform == "log2":
                ax.set_xlabel("log2(CV)")
            else:
                ax.set_xlabel("Mean CV")

    # styling
    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)

        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size)
    else:
        ax.tick_params(labelsize=font_size)
        ax.xaxis.label.set_size(font_size)
        ax.yaxis.label.set_size(font_size)

    if show_title:
        if title is None:
            title = f"Feature CV across conditions | {channel}"
        if font_family is not None:
            ax.set_title(title, fontfamily=font_family, fontsize=title_size or font_size)
        else:
            ax.set_title(title, fontsize=title_size or font_size)

    if orientation == "vertical" and show_xticklabels:
        plt.subplots_adjust(bottom=0.3)
    else:
        plt.tight_layout()

    plt.show()

    if save:
        if figtitle is None:
            figtitle = f"feature_cv_{channel}"
        fig.savefig("plots/"+figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/"+figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/"+figtitle + ".svg", bbox_inches="tight")

    return fig, ax, results_df
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    
    
    g.savefig('plots/chlorella_pub_fig_' + figtitle + '.svg',bbox_inches='tight',pad_inches=0)
    g.savefig('plots/chlorella_pub_fig_' + figtitle + '.pdf',bbox_inches='tight',pad_inches=0)
    g.savefig('plots/chlorella_pub_fig_' + figtitle + '.png',bbox_inches='tight',pad_inches=0)

    return g,row_linkage
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.svg',bbox_inches='tight')
    fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.pdf',bbox_inches='tight')
    fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.png',bbox_inches='tight')
    return ax
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def compute_dead_wells_and_ids(
    zroots,
    experiments_to_check,
    n_fovs,
    n_wells=96,
    cell_count_cutoff=1,
    channels=("BF", "CY5"),
):
    """
    Identify dead wells and corresponding row indices (dead_ids) across FoVs.

    A well is considered "dead" if, in ANY:
        - experiment
        - field of view
        - channel
    the number of segmented cells is below `cell_count_cutoff`.

    Parameters
    ----------
    zroots : list
        List of ExperimentStore objects.
    experiments_to_check : list[int]
        Indices of experiments in zroots to evaluate.
    n_fovs : int
        Number of fields of view per experiment.
    n_wells : int, default=96
        Number of wells per plate.
    cell_count_cutoff : int
        Minimum number of cells required for a well to be considered valid.
    channels : tuple[str]
        Channels to check (e.g., ("BF", "CY5")).

    Returns
    -------
    dead_wells : list[int]
        Sorted list of well indices that failed the cutoff.
    dead_ids : list[int]
        Row indices corresponding to dead wells across all FoVs
        (compatible with row indexing: well + fov * n_wells).
    """

    dead_wells = set()

    for well in range(n_wells):
        is_dead = False

        for fov in range(n_fovs):
            for e in experiments_to_check:
                for channel in channels:
                    props = zroots[e].get_cell_properties(fov, well, channel)

                    if props is None or len(props) < cell_count_cutoff:
                        is_dead = True
                        break

                if is_dead:
                    break
            if is_dead:
                break

        if is_dead:
            dead_wells.add(well)

    dead_wells = sorted(dead_wells)

    # Expand to row indices across FoVs
    dead_ids = [
        well + (fov * n_wells)
        for fov in range(n_fovs)
        for well in dead_wells
    ]

    return dead_wells, dead_ids
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def build_training_data(
    zroots,
    exp_conditions=None,
    experiments_to_compare=None,
    n_fovs_2_analyze=9,
    n_wells=96,
    include_channels=("BF", "CY5"),
    exclude_features=None,
    include_pixel_intensity=False,
    pixel_intensity_channels=("BF", "CY5"),
):
    """
    Build training_data as a dictionary of condition -> numpy array.

    By default, this reproduces the original behavior exactly:
    - rows = n_wells * n_fovs_2_analyze
    - columns = 22 BF features + 22 CY5 features = 44
    - one row per (well, field_of_view)

    Optional behavior:
    - exclude selected Cellpose features
    - append pixel intensity for selected channels

    Parameters
    ----------
    zroots : list
        List of ExperimentStore-like objects, one per condition.
    exp_conditions : list[str] or None
        Names for the conditions. If None, defaults to:
        ["e_stat_16","e_stat_22","e_stat_30","l_stat_16","l_stat_22","l_stat_30"]
    experiments_to_compare : list[int] or None
        Indices into zroots. If None, defaults to [0,1,2,3,4,5].
    n_fovs_2_analyze : int
        Number of FoVs per experiment to use.
    n_wells : int
        Number of wells per experiment.
    include_channels : tuple[str]
        Channels to include, typically ("BF","CY5").
    exclude_features : list[str] or None
        Feature names to exclude, e.g.
        ["moments_hu-4", "moments_hu-5", "moments_hu-6"]
    include_pixel_intensity : bool
        If True, append pixel intensity features for channels in pixel_intensity_channels.
    pixel_intensity_channels : tuple[str]
        Channels for which to append pixel intensity if include_pixel_intensity=True.

    Returns
    -------
    training_data : dict[str, np.ndarray]
        Condition -> array of shape (n_wells * n_fovs_2_analyze, n_features)
    training_feature_labels : list[str]
        Column labels in the exact order used in training_data.
    feature_slices : dict[str, slice]
        Column slices for each channel block and pixel intensity block.
        Example:
            feature_slices["BF"]
            feature_slices["CY5"]
            feature_slices["BF_pixel_intensity"]
    """

    if exp_conditions is None:
        exp_conditions = [
            "e_stat_16",
            "e_stat_22",
            "e_stat_30",
            "l_stat_16",
            "l_stat_22",
            "l_stat_30",
        ]

    if experiments_to_compare is None:
        experiments_to_compare = [0, 1, 2, 3, 4, 5]

    if exclude_features is None:
        exclude_features = []

    if len(exp_conditions) != len(experiments_to_compare):
        raise ValueError(
            f"exp_conditions has length {len(exp_conditions)} but "
            f"experiments_to_compare has length {len(experiments_to_compare)}"
        )

    # --------------------------------------------------
    # Determine feature names per channel from a known-good reference
    # --------------------------------------------------
    channel_feature_names = {}
    training_feature_labels = []
    feature_slices = {}

    col_start = 0

    for channel in include_channels:
        reference_props = None

        # find the first valid dataframe for this channel
        for exp_idx in experiments_to_compare:
            found = False
            for fov in range(n_fovs_2_analyze):
                for well in range(n_wells):
                    props = zroots[exp_idx].get_cell_properties(fov, well, channel)
                    if props is not None and len(props) > 0:
                        reference_props = props
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if reference_props is None or len(reference_props) == 0:
            raise ValueError(f"Could not find reference features for channel {channel}")

        feature_names = reference_props.columns.tolist()

        if len(feature_names) > 0 and feature_names[0] == "label":
            feature_names[0] = "count"

        feature_names = [f for f in feature_names if f not in exclude_features]
        channel_feature_names[channel] = feature_names

        # labels for this channel block
        channel_labels = [f"{f} ({channel.lower()})" for f in feature_names]
        training_feature_labels.extend(channel_labels)

        col_end = col_start + len(feature_names)
        feature_slices[channel] = slice(col_start, col_end)
        col_start = col_end

        # optional pixel intensity
        if include_pixel_intensity and channel in pixel_intensity_channels:
            training_feature_labels.append(f"pixel_intensity ({channel.lower()})")
            feature_slices[f"{channel}_pixel_intensity"] = slice(col_start, col_start + 1)
            col_start += 1

    n_features_total = len(training_feature_labels)

    # --------------------------------------------------
    # Build arrays condition by condition
    # --------------------------------------------------
    experiment_cell_pose_means = []

    for e in experiments_to_compare:
        means = np.full((n_wells * n_fovs_2_analyze, n_features_total), np.nan, dtype=float)

        for fov in range(n_fovs_2_analyze):
            for well in range(n_wells):
                row_idx = well + (fov * n_wells)
                col_idx = 0

                for channel in include_channels:
                    props = zroots[e].get_cell_properties(fov, well, channel)

                    if props is None or len(props) == 0:
                        # skip this channel block; leave NaNs in place
                        col_idx = feature_slices[channel].stop
                        if include_pixel_intensity and channel in pixel_intensity_channels:
                            col_idx += 1
                        continue

                    current_names = props.columns.tolist()
                    if len(current_names) > 0 and current_names[0] == "label":
                        current_names[0] = "count"
                        props = props.copy()
                        props.columns = current_names

                    # subset + preserve order
                    selected_feature_names = channel_feature_names[channel]
                    props = props[selected_feature_names]

                    # mean across cells in this FoV/well/channel
                    try:
                        means[row_idx, feature_slices[channel]] = np.array(
                            np.nanmean(props, axis=0).astype("float32")
                        )
                    except Exception:
                        pass

                    col_idx = feature_slices[channel].stop

                    # optional pixel intensity
                    if include_pixel_intensity and channel in pixel_intensity_channels:
                        try:
                            label_arr = zroots[e].get_label_image(fov, well, channel)
                            mask_arr = label_arr != 0
                            proc_img = zroots[e].get_processed_intensities(fov, well, channel)

                            pixel_vals = proc_img[mask_arr]
                            if len(pixel_vals) > 0:
                                means[row_idx, col_idx] = float(np.nanmean(pixel_vals))
                        except Exception:
                            pass

                        col_idx += 1

        experiment_cell_pose_means.append(means)

    # --------------------------------------------------
    # Package into training_data dict
    # --------------------------------------------------
    training_data = {
        cond: experiment_cell_pose_means[i]
        for i, cond in enumerate(exp_conditions)
    }

    return training_data, training_feature_labels, feature_slices
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_mean_relative_feature_values_across_conditions(
    training_data,
    training_feature_labels,
    condition_order=None,
    col_titles=None,
    figsize=(22, 3),
    cmap="apc:orange_sage_r",
    title="Mean Relative Feature Values Across Conditions",
    show_title=True,
    title_fontsize=18,
    xtick_colors=None,
    ytick_colors=None,
    xtick_rotation=45,
    xtick_fontsize=None,
    ytick_fontsize=None,
    axis_color="black",
    font_family=None,
    font_size=10,
    title_size=None,
    double_zscore=True,
    orientation="horizontal",  # "horizontal" or "vertical"
    cbar=True,
    cbar_label=None,
    cbar_shrink=1.0,
    cbar_aspect=30,
    vmin=None,
    vmax=None,
    save=False,
    figtitle="mean_relative_feature_values_across_conditions",
):
    """
    Plot mean relative feature values across conditions as a heatmap.

    orientation:
        "horizontal" -> conditions on y-axis, features on x-axis
        "vertical"   -> conditions on x-axis, features on y-axis
    """

    if condition_order is None:
        condition_order = list(training_data.keys())

    if col_titles is None:
        col_titles = condition_order

    if len(condition_order) != len(col_titles):
        raise ValueError(
            f"condition_order has length {len(condition_order)} but col_titles has length {len(col_titles)}"
        )

    n_features = training_data[condition_order[0]].shape[1]

    if len(training_feature_labels) != n_features:
        raise ValueError(
            f"training_feature_labels has length {len(training_feature_labels)} "
            f"but training_data arrays have {n_features} columns"
        )

    if xtick_fontsize is None:
        xtick_fontsize = font_size
    if ytick_fontsize is None:
        ytick_fontsize = font_size

    # --- Compute condition means ---
    condition_means = np.vstack([
        np.nanmean(training_data[c], axis=0) for c in condition_order
    ])

    # --- Z-score ---
    if double_zscore:
        # Preserves your original notebook behavior
        plot_mat = zscore(zscore(condition_means, axis=0), axis=None)
    else:
        plot_mat = condition_means.copy()

    plot_df = pd.DataFrame(
        plot_mat,
        index=col_titles,
        columns=training_feature_labels
    )

    # --- Orientation ---
    if orientation == "vertical":
        plot_df = plot_df.T
    elif orientation != "horizontal":
        raise ValueError('orientation must be "horizontal" or "vertical"')

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    hm = sns.heatmap(
        plot_df,
        cmap=cmap,
        cbar=cbar,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={
            "shrink": cbar_shrink,
            "aspect": cbar_aspect
        } if cbar else None
    )

    # --- Colorbar label ---
    if cbar and cbar_label is not None:
        hm.collections[0].colorbar.set_label(
            cbar_label,
            fontsize=font_size,
            fontfamily=font_family
        )

    # --- Title ---
    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize or font_size,
            fontfamily=font_family
        )

    # --- Tick label formatting ---
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=xtick_rotation,
        ha="right" if xtick_rotation != 0 else "center",
        rotation_mode="anchor",
        fontsize=xtick_fontsize,
        fontfamily=font_family
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=ytick_fontsize,
        fontfamily=font_family
    )

    # --- Default tick label color first ---
    for tick in ax.get_xticklabels():
        tick.set_color(axis_color)
    for tick in ax.get_yticklabels():
        tick.set_color(axis_color)

    # --- Apply custom tick colors LAST ---
    if orientation == "horizontal":
        # x-axis = features
        if xtick_colors is not None:
            if len(xtick_colors) != len(ax.get_xticklabels()):
                raise ValueError(
                    f"xtick_colors has length {len(xtick_colors)} but there are {len(ax.get_xticklabels())} x-axis labels"
                )
            for tick, color in zip(ax.get_xticklabels(), xtick_colors):
                tick.set_color(color)

        # y-axis = conditions
        if ytick_colors is not None:
            if len(ytick_colors) != len(ax.get_yticklabels()):
                raise ValueError(
                    f"ytick_colors has length {len(ytick_colors)} but there are {len(ax.get_yticklabels())} y-axis labels"
                )
            for tick, color in zip(ax.get_yticklabels(), ytick_colors):
                tick.set_color(color)

    elif orientation == "vertical":
        # x-axis = conditions
        if xtick_colors is not None:
            if len(xtick_colors) != len(ax.get_xticklabels()):
                raise ValueError(
                    f"xtick_colors has length {len(xtick_colors)} but there are {len(ax.get_xticklabels())} x-axis labels"
                )
            for tick, color in zip(ax.get_xticklabels(), xtick_colors):
                tick.set_color(color)

        # y-axis = features
        if ytick_colors is not None:
            if len(ytick_colors) != len(ax.get_yticklabels()):
                raise ValueError(
                    f"ytick_colors has length {len(ytick_colors)} but there are {len(ax.get_yticklabels())} y-axis labels"
                )
            for tick, color in zip(ax.get_yticklabels(), ytick_colors):
                tick.set_color(color)

    # --- Axis styling ---
    for spine in ax.spines.values():
        spine.set_color(axis_color)

    # style tick MARKS, not label colors
    ax.tick_params(axis="x", color=axis_color)
    ax.tick_params(axis="y", color=axis_color)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.svg', bbox_inches='tight')
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.pdf', bbox_inches='tight')
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.png', bbox_inches='tight')

    return fig, ax, plot_df
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_condition_matrix_finite(arr):
    arr = arr.copy().astype(float)

    # treat inf as nan
    arr[~np.isfinite(arr)] = np.nan

    # compute column means ignoring nan
    col_means = np.nanmean(arr, axis=0)

    # fail loudly if an entire feature column is missing
    bad_cols = np.where(np.isnan(col_means))[0]
    if len(bad_cols) > 0:
        raise ValueError(
            f"These feature columns are all-NaN and cannot be imputed: {bad_cols}"
        )

    # fill nan entries with column means
    nan_rows, nan_cols = np.where(np.isnan(arr))
    arr[nan_rows, nan_cols] = col_means[nan_cols]

    return arr
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def remove_rows_cols(array, row_indices, col_indices):
    """
    Removes specified rows and columns from a 2D numpy array.
    """
    # Remove rows (axis 0)
    temp_array = np.delete(array, row_indices, axis=0)
    # Remove columns (axis 1)
    final_array = np.delete(temp_array, col_indices, axis=1)
    return final_array
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    
    fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.svg',bbox_inches='tight')
    fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.pdf',bbox_inches='tight')
    fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.png',bbox_inches='tight')

    return pd.Series(zscore(input_dat[:,feat_2_color_by])).quantile(.95)
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    
    fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.svg',bbox_inches='tight')
    fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.pdf',bbox_inches='tight')
    fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.png',bbox_inches='tight')

    return all_rs,all_ps
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def condition_holdout_permutation_test(
    data,
    alpha=1000,
    n_splits=10,
    n_permutations=100,
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_sorted_r2_per_feature(
    ridge_model_residulals,
    training_feature_labels,
    xtick_colors=None,
    ytick_colors=None,
    col_titles=None,
    figsize=(8, 20),
    figtitle='sorted_r2_per_feature',
    cmap="apc:verde",
    vmin=0,
    vmax=1,
    annot=True,
    annot_fontsize=10,
    annot_color="black",
    sort_by_mean=True,
    show_plot=True,
    save=True,
    # orientation
    orientation="horizontal",  # "horizontal" or "vertical"
    # font controls
    font_family=None,
    font_size=10,
    title=None,
    show_title=False,
    title_size=None,
    x_tick_fontsize=None,
    y_tick_fontsize=None,
    x_tick_rotation=0,
    y_tick_rotation=0,
    label_color="black",
    axis_color="black",
    # colorbar controls
    cbar=True,
    cbar_label=None,
    cbar_shrink=1.0,
    cbar_aspect=30,
):
    """
    Plot per-feature R² values from condition_holdout_permutation_test or related output.

    Parameters
    ----------
    ridge_model_residulals : dict
        Output from cvp.condition_holdout_permutation_test(...)
        Must contain:
            ridge_model_residulals['real']['r2_per_feature']
            ridge_model_residulals['conditions']   (optional, used if col_titles=None)
    training_feature_labels : list[str]
        Feature labels in the exact column order used to build training_data.
    xtick_colors : list or None
        Colors for x-axis tick labels. Must match number of x-axis labels.
    ytick_colors : list or None
        Colors for y-axis tick labels. Must match number of y-axis labels.
    col_titles : list[str] or None
        Condition titles. If None, uses ridge_model_residulals['conditions'].
    figsize : tuple
        Figure size.
    figtitle : str
        Root filename for saving.
    cmap : str
        Heatmap colormap.
    vmin, vmax : float
        Heatmap color scale limits.
    annot : bool
        Whether to annotate heatmap cells.
    annot_fontsize : int
        Font size of annotation text in heatmap cells.
    annot_color : str
        Color of annotation text in heatmap cells.
    sort_by_mean : bool
        If True, sort features by mean R² across conditions (descending).
    show_plot : bool
        Whether to display the figure.
    save : bool
        Whether to save png/pdf/svg.
    orientation : str
        "horizontal" = features on y-axis, conditions on x-axis
        "vertical"   = features on x-axis, conditions on y-axis
    font_family : str or None
        Optional font family.
    font_size : int
        Base font size.
    title : str or None
        Optional title.
    show_title : bool
        Whether to show title.
    title_size : int or None
        Title fontsize override.
    x_tick_fontsize : int or None
        X tick label fontsize. Defaults to font_size.
    y_tick_fontsize : int or None
        Y tick label fontsize. Defaults to font_size.
    x_tick_rotation : int or float
        Rotation of x tick labels.
    y_tick_rotation : int or float
        Rotation of y tick labels.
    label_color : str
        Default color for tick labels.
    axis_color : str
        Color of axes spines and tick marks.
    cbar : bool
        Whether to show a colorbar.
    cbar_label : str or None
        Optional colorbar label.
    cbar_shrink : float
        Colorbar shrink factor.
    cbar_aspect : int
        Colorbar aspect ratio (higher = skinnier).

    Returns
    -------
    fig, ax, plot_df
        plot_df is the plotted dataframe.
    """

    r2_per_feature = np.array(ridge_model_residulals['real']['r2_per_feature'])

    if r2_per_feature.ndim != 2:
        raise ValueError(
            f"Expected ridge_model_residulals['real']['r2_per_feature'] to be 2D, "
            f"got shape {r2_per_feature.shape}"
        )

    n_conditions, n_features = r2_per_feature.shape

    if len(training_feature_labels) != n_features:
        raise ValueError(
            f"training_feature_labels has length {len(training_feature_labels)} "
            f"but r2_per_feature has {n_features} features"
        )

    if col_titles is None:
        if 'conditions' in ridge_model_residulals:
            col_titles = ridge_model_residulals['conditions']
        else:
            col_titles = [f"cond_{i}" for i in range(n_conditions)]

    if len(col_titles) != n_conditions:
        raise ValueError(
            f"col_titles has length {len(col_titles)} but there are {n_conditions} conditions"
        )

    if x_tick_fontsize is None:
        x_tick_fontsize = font_size
    if y_tick_fontsize is None:
        y_tick_fontsize = font_size

    # Base dataframe: rows = features, cols = conditions
    plot_df = pd.DataFrame(
        r2_per_feature.T,
        index=training_feature_labels,
        columns=col_titles
    )

    # Sort before optional transpose so feature ordering is preserved
    if sort_by_mean:
        sort_order = plot_df.mean(axis=1).sort_values(ascending=False).index
        plot_df = plot_df.loc[sort_order]

    # Orientation handling
    if orientation == "vertical":
        plot_df = plot_df.T
    elif orientation != "horizontal":
        raise ValueError('orientation must be "horizontal" or "vertical"')

    fig, ax = plt.subplots(figsize=figsize)

    hm = sns.heatmap(
        plot_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=".2f" if annot else "",
        cbar=cbar,
        ax=ax,
        cbar_kws={
            "shrink": cbar_shrink,
            "aspect": cbar_aspect
        } if cbar else None
    )

    # Colorbar styling
    if cbar and cbar_label is not None:
        cbar_obj = hm.collections[0].colorbar
        cbar_obj.set_label(
            cbar_label,
            fontsize=font_size,
            fontfamily=font_family
        )

    # Annotation styling
    if annot:
        for text in hm.texts:
            text.set_fontsize(annot_fontsize)
            text.set_color(annot_color)
            if font_family is not None:
                text.set_fontfamily(font_family)

    # Title
    if show_title:
        if title is None:
            title = "Per-feature cross-validated R²"
        ax.set_title(
            title,
            fontsize=title_size or font_size,
            fontfamily=font_family,
            color=label_color
        )

    # Set tick labels with base formatting first
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=x_tick_rotation,
        ha="center" if x_tick_rotation == 0 else "right",
        rotation_mode="anchor",
        fontsize=x_tick_fontsize,
        fontfamily=font_family
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=y_tick_rotation,
        fontsize=y_tick_fontsize,
        fontfamily=font_family
    )

    # Default label colors
    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    # Apply custom x tick colors LAST so they are not overridden
    if xtick_colors is not None:
        if len(xtick_colors) != len(ax.get_xticklabels()):
            raise ValueError(
                f"xtick_colors has length {len(xtick_colors)} but there are {len(ax.get_xticklabels())} x-axis labels"
            )
        for tick_label, color in zip(ax.get_xticklabels(), xtick_colors):
            tick_label.set_color(color)

    # Apply custom y tick colors LAST so they are not overridden
    if ytick_colors is not None:
        if len(ytick_colors) != len(ax.get_yticklabels()):
            raise ValueError(
                f"ytick_colors has length {len(ytick_colors)} but there are {len(ax.get_yticklabels())} y-axis labels"
            )
        for tick_label, color in zip(ax.get_yticklabels(), ytick_colors):
            tick_label.set_color(color)

    # Axis styling
    for spine in ax.spines.values():
        spine.set_color(axis_color)

    # Only style tick MARKS here, not label colors
    ax.tick_params(axis="x", color=axis_color)
    ax.tick_params(axis="y", color=axis_color)

    plt.tight_layout()

    if show_plot:
        plt.show()

    if save:
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.svg', bbox_inches='tight')
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.pdf', bbox_inches='tight')
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.png', bbox_inches='tight')

    return fig, ax, plot_df
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_true_vs_predicted_pca_by_condition(
    training_data,
    exp_conditions,
    dead_ids,
    ytick_colors,
    col_titles,
    alpha=100,
    model_type="ridge",
    figsize=(30, 20),
    nrows=2,
    ncols=3,
    prediction_color=apc.shell,
    prediction_alpha=0.8,
    ground_truth_alpha=0.8,
    quiver_color="gray",
    quiver_alpha=0.2,
    quiver_width=0.001,
    title_fontsize=24,
    axis_label_fontsize=12,
    legend_loc="lower left",
    show_legend=True,
    hide_ticks=True,
    xlabel="PC1",
    ylabel="PC2",
    save=False,
    figtitle="true_vs_predicted_phenotype_visualized_in_pc_space",
):
    """
    For each condition in exp_conditions:
      1. train a model to predict that condition from all others
      2. stack ground-truth and predicted matrices
      3. run PCA jointly on the stacked matrix
      4. remove dead_ids rows
      5. plot ground truth, predictions, and connecting arrows

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Dictionary mapping condition name -> matrix of shape (n_rows, n_features).
        This should usually be training_data_finite.
    exp_conditions : list[str]
        Condition names in the order to plot.
    dead_ids : list[int]
        Row indices to remove from each condition matrix.
    ytick_colors : list
        One color per condition for ground-truth points.
    col_titles : list[str]
        One plot title per condition.
    alpha : float
        Ridge alpha used in cvp.train_and_predict.
    model_type : str
        Model type passed to cvp.train_and_predict.
    figsize : tuple
        Figure size.
    nrows, ncols : int
        Subplot grid dimensions.
    prediction_color : matplotlib color
        Color for predicted points.
    prediction_alpha : float
        Alpha for predicted points.
    ground_truth_alpha : float
        Alpha for ground-truth points.
    quiver_color : matplotlib color
        Color for connecting arrows.
    quiver_alpha : float
        Alpha for arrows.
    quiver_width : float
        Arrow width.
    title_fontsize : int
        Font size for subplot titles.
    axis_label_fontsize : int
        Font size for axis labels.
    legend_loc : str
        Legend location.
    show_legend : bool
        Whether to show legend on each subplot.
    hide_ticks : bool
        Whether to hide x and y ticks.
    xlabel, ylabel : str
        Axis labels.
    save : bool
        Whether to save png/pdf/svg.
    figtitle : str
        Root filename for saving.

    Returns
    -------
    fig, ax, data_for_PCA_graphs, metrics_by_condition
        data_for_PCA_graphs is a list of [ground_truth_as_pcs_cleaned, prediction_as_pcs_cleaned]
        metrics_by_condition is a list of metrics dicts returned by train_and_predict
    """

    if len(exp_conditions) != len(col_titles):
        raise ValueError(
            f"exp_conditions has length {len(exp_conditions)} but col_titles has length {len(col_titles)}"
        )

    if len(exp_conditions) != len(ytick_colors):
        raise ValueError(
            f"exp_conditions has length {len(exp_conditions)} but ytick_colors has length {len(ytick_colors)}"
        )

    if nrows * ncols < len(exp_conditions):
        raise ValueError(
            f"subplot grid {nrows}x{ncols} cannot hold {len(exp_conditions)} conditions"
        )

    data_for_PCA_graphs = []
    metrics_by_condition = []

    for i in range(len(exp_conditions)):
        test_condition = exp_conditions[i]
        train_conditions = [e for e in exp_conditions if e != test_condition]

        y_pred, metrics, model = train_and_predict(
            training_data,
            train_conditions,
            test_condition,
            model_type=model_type,
            alpha=alpha
        )

        stacked_dat = np.vstack([training_data[test_condition], y_pred])

        experiment_as_pcs, experiment_pc_evecs, experiment_pc_evals, experiment_pc_mean_value = pca_on_np_array(stacked_dat)

        n_rows = training_data[test_condition].shape[0]

        ground_truth_as_pcs = experiment_as_pcs[0:n_rows]
        prediction_as_pcs = experiment_as_pcs[n_rows:(2 * n_rows)]

        ground_truth_as_pcs_cleaned = remove_rows_cols(
            ground_truth_as_pcs,
            row_indices=dead_ids,
            col_indices=[]
        )
        prediction_as_pcs_cleaned = remove_rows_cols(
            prediction_as_pcs,
            row_indices=dead_ids,
            col_indices=[]
        )

        data_for_PCA_graphs.append([ground_truth_as_pcs_cleaned, prediction_as_pcs_cleaned])
        metrics_by_condition.append(metrics)

    fig, ax = plt.subplots(nrows, ncols)
    fig.set_size_inches(figsize[0], figsize[1])

    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = np.array([ax])

    for cond in range(len(exp_conditions)):
        gt = data_for_PCA_graphs[cond][0]
        pred = data_for_PCA_graphs[cond][1]

        ax[cond].quiver(
            gt[:, 0],
            gt[:, 1],
            pred[:, 0] - gt[:, 0],
            pred[:, 1] - gt[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=quiver_color,
            alpha=quiver_alpha,
            width=quiver_width
        )

        ax[cond].scatter(
            pred[:, 0],
            pred[:, 1],
            label="Prediction",
            color=prediction_color,
            alpha=prediction_alpha
        )

        ax[cond].scatter(
            gt[:, 0],
            gt[:, 1],
            label="Ground Truth",
            color=ytick_colors[cond],
            alpha=ground_truth_alpha
        )

        ax[cond].set_title(col_titles[cond], fontsize=title_fontsize)

        if hide_ticks:
            ax[cond].set_xticks([])
            ax[cond].set_yticks([])

        ax[cond].set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax[cond].set_ylabel(ylabel, fontsize=axis_label_fontsize)

        if show_legend:
            ax[cond].legend(loc=legend_loc)

    # hide unused axes
    for j in range(len(exp_conditions), len(ax)):
        ax[j].axis("off")

    plt.show()

    if save:
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.svg', bbox_inches='tight')
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.pdf', bbox_inches='tight')
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.png', bbox_inches='tight')

    return fig, ax, data_for_PCA_graphs, 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



