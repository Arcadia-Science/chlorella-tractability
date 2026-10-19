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
from arcadia_microscopy_tools.pipeline import ImageOperation
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
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
    show_only_merged=False,
    hide_in_cell=False,

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

    if show_only_merged==False:   
        for i in range(len(ims)):
            fig, ax = plt.subplots(figsize=(f_size[0], f_size[1]))
            if hide_in_cell:
                plt.close(fig)

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
    show_feature_axis=True,
    reverse_feature_order=False,
    xlim=[0,20]
):
    """
    Plot the coefficient of variation (CV = std / mean) for each feature.

    CV is always converted to absolute value before averaging or plotting.

    Parameters
    ----------
    exclude_features : list or None
        Feature names to exclude, e.g.
        ["moments_hu-4", "moments_hu-5", "moments_hu-6"]

    show_feature_axis : bool
        If False, hide the axis label, tick marks, and tick labels on the axis
        that contains the feature names.
        - vertical orientation -> hides x-axis feature labels/ticks
        - horizontal orientation -> hides y-axis feature labels/ticks

    reverse_feature_order : bool
        If True, reverse the plotting order of features.
        Default False preserves the original behavior.
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
    # Reverse feature order if requested
    # ----------------------------
    if reverse_feature_order:
        results_df = results_df.iloc[::-1].reset_index(drop=True)

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
        if show_xticklabels and show_feature_axis:
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

        if not show_feature_axis:
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax.set_xlabel("")

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
        if show_xticklabels and show_feature_axis:
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

        if not show_feature_axis:
            ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
            ax.set_ylabel("")
    ax.set_xlim(xlim)

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

    if orientation == "vertical" and show_xticklabels and show_feature_axis:
        plt.subplots_adjust(bottom=0.3)
    else:
        plt.tight_layout()

    plt.show()

    if save:
        if figtitle is None:
            figtitle = f"feature_cv_{channel}"
        fig.savefig("plots/" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".svg", bbox_inches="tight")

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
              feat_alpha=.7,
              hide_in_cell=True):

    
    
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
    if hide_in_cell:
        plt.close(fig)
    
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
    alpha=.8,
    model_type="ridge",
    figsize=(30, 20),
    marker_size=100,
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



        ax[cond].scatter(
            pred[:, 0],
            pred[:, 1],
            label="Prediction",
            color=prediction_color,
            alpha=prediction_alpha,
            s=marker_size
        )

        ax[cond].scatter(
            gt[:, 0],
            gt[:, 1],
            label="Ground Truth",
            color=ytick_colors[cond],
            alpha=ground_truth_alpha,
            s=marker_size
        )

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

def compute_within_between_strain_variation(
    training_data,
    condition,
    n_wells=96,
    n_fovs=None,
    feature_labels=None,
    use_cv=True,
    eps=1e-8
):
    """
    Compute within-strain vs between-strain variability for each feature.

    Returns
    -------
    within_var : np.array (n_features,)
    between_var : np.array (n_features,)
    ratio : np.array (n_features,)  # between / within
    """

    dat = training_data[condition]  # shape: (n_rows, n_features)

    n_rows, n_features = dat.shape

    if n_fovs is None:
        n_fovs = n_rows // n_wells

    # --- collect per-well data ---
    well_data = []

    for well in range(n_wells):
        rows = [well + (fov * n_wells) for fov in range(n_fovs)]
        well_data.append(dat[rows, :])  # shape: (n_fovs, n_features)

    well_data = np.array(well_data)  # shape: (n_wells, n_fovs, n_features)

    # --- within-strain variability ---
    # variance across FoVs for each well, then average across wells
    within_var = np.nanmean(np.nanvar(well_data, axis=1), axis=0)

    # --- between-strain variability ---
    # mean per well, then variance across wells
    well_means = np.nanmean(well_data, axis=1)  # (n_wells, n_features)
    between_var = np.nanvar(well_means, axis=0)

    if use_cv:
        # convert to coefficient of variation-like scaling
        within_var = np.sqrt(within_var) / (np.abs(np.nanmean(well_means, axis=0)) + eps)
        between_var = np.sqrt(between_var) / (np.abs(np.nanmean(well_means, axis=0)) + eps)

    ratio = between_var / (within_var + eps)

    return within_var, between_var, ratio

def plot_within_vs_between_variation(
    within_var,
    between_var,
    ratio,
    feature_labels,
    sort=True,
    figsize=(6, 10),
    title="Within vs Between Strain Variation",
    save=False,
    figtitle="within_vs_between_variation"
):

    import matplotlib.pyplot as plt
    import numpy as np

    if sort:
        order = np.argsort(ratio)[::-1]
        within_var = within_var[order]
        between_var = between_var[order]
        ratio = ratio[order]
        feature_labels = np.array(feature_labels)[order]

    x = np.arange(len(feature_labels))

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(x, between_var, label="Between-strain", alpha=0.7)
    ax.barh(x, within_var, label="Within-strain", alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(feature_labels)

    ax.set_xlabel("Variation")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.svg', bbox_inches='tight')
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.pdf', bbox_inches='tight')
        fig.savefig('plots/chlorella_pub_fig_' + figtitle + '.png', bbox_inches='tight')

    return fig, ax


def compute_within_between_ratio_matrix(
    training_data,
    condition_order=None,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    log2_transform=True,
    eps=1e-8,
):
    """
    Compute between-strain / within-strain variation ratio for each feature
    across all conditions.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Dict mapping condition name -> matrix of shape (n_rows, n_features)
    condition_order : list[str] or None
        Order of conditions to compute. If None, uses training_data.keys().
    n_wells : int
        Number of wells per condition.
    n_fovs : int or None
        Number of FoVs per well. If None, inferred from n_rows // n_wells.
    dead_ids : list[int] or None
        Row indices corresponding to dead wells across FoVs.
        Used to infer dead wells to skip.
    log2_transform : bool
        If True, return log2(between / within)
    eps : float
        Small constant to avoid divide-by-zero.

    Returns
    -------
    ratio_mat : np.ndarray
        Array of shape (n_conditions, n_features)
    condition_order : list[str]
        Final condition order used
    """

    if condition_order is None:
        condition_order = list(training_data.keys())

    if dead_ids is None:
        dead_ids = []

    dead_wells = sorted(set([d % n_wells for d in dead_ids]))

    ratio_rows = []

    for condition in condition_order:
        dat = np.asarray(training_data[condition], dtype=float)
        n_rows, n_features = dat.shape

        if n_fovs is None:
            this_n_fovs = n_rows // n_wells
        else:
            this_n_fovs = n_fovs

        # collect per-well FoV matrices
        well_blocks = []
        valid_wells = []

        for well in range(n_wells):
            if well in dead_wells:
                continue

            row_idx = [well + (fov * n_wells) for fov in range(this_n_fovs)]
            block = dat[row_idx, :]  # shape: (n_fovs, n_features)

            if np.all(~np.isfinite(block)):
                continue

            well_blocks.append(block)
            valid_wells.append(well)

        if len(well_blocks) == 0:
            raise ValueError(f"No valid wells found for condition {condition}")

        well_blocks = np.array(well_blocks)  # (n_valid_wells, n_fovs, n_features)

        # within-strain variation:
        # variance across FoVs within each well, then mean across wells
        within_var = np.nanmean(np.nanvar(well_blocks, axis=1), axis=0)

        # between-strain variation:
        # mean feature vector per well, then variance across wells
        well_means = np.nanmean(well_blocks, axis=1)  # (n_valid_wells, n_features)
        between_var = np.nanvar(well_means, axis=0)

        ratio = between_var / (within_var + eps)

        if log2_transform:
            ratio = np.log2(ratio + eps)

        ratio_rows.append(ratio)

    ratio_mat = np.vstack(ratio_rows)
    return ratio_mat, condition_order

def plot_within_between_ratio_heatmap(
    ratio_mat,
    training_feature_labels,
    condition_order,
    col_titles=None,
    orientation="horizontal",
    figsize=(22, 4),
    cmap="apc:orange_sage_r",
    title="Between- vs within-strain variation across conditions",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    xtick_rotation=45,
    ytick_rotation=0,
    axis_color="black",
    label_color="black",
    xtick_colors=None,
    ytick_colors=None,
    # NEW
    sort_features=True,
    sort_by="mean",  # "mean" (default), later extensible
    # colorbar
    cbar=True,
    cbar_label="log2(Between / Within)",
    cbar_shrink=1.0,
    cbar_aspect=30,
    vmin=None,
    vmax=None,
    save=False,
    figtitle="within_between_ratio_heatmap",
):

    if col_titles is None:
        col_titles = condition_order

    if xtick_fontsize is None:
        xtick_fontsize = font_size
    if ytick_fontsize is None:
        ytick_fontsize = font_size

    if ratio_mat.shape[0] != len(condition_order):
        raise ValueError("Condition mismatch")

    if ratio_mat.shape[1] != len(training_feature_labels):
        raise ValueError("Feature mismatch")

    ratio_mat = np.array(ratio_mat, dtype=float)
    feature_labels = np.array(training_feature_labels)

    # --- SORT FEATURES ---
    if sort_features:
        if sort_by == "mean":
            sort_metric = np.nanmean(ratio_mat, axis=0)
        else:
            raise ValueError("Currently only sort_by='mean' is supported")

        order = np.argsort(sort_metric)[::-1]  # descending
        ratio_mat = ratio_mat[:, order]
        feature_labels = feature_labels[order]

    # --- Build dataframe ---
    plot_df = pd.DataFrame(
        ratio_mat,
        index=col_titles,
        columns=feature_labels
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

    # --- Tick formatting ---
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
        rotation=ytick_rotation,
        fontsize=ytick_fontsize,
        fontfamily=font_family
    )

    # --- Default label color ---
    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    # --- Apply custom colors LAST ---
    if xtick_colors is not None:
        if len(xtick_colors) != len(ax.get_xticklabels()):
            raise ValueError("xtick_colors length mismatch")
        for tick, color in zip(ax.get_xticklabels(), xtick_colors):
            tick.set_color(color)

    if ytick_colors is not None:
        if len(ytick_colors) != len(ax.get_yticklabels()):
            raise ValueError("ytick_colors length mismatch")
        for tick, color in zip(ax.get_yticklabels(), ytick_colors):
            tick.set_color(color)

    # --- Axis styling ---
    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", color=axis_color)
    ax.tick_params(axis="y", color=axis_color)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, plot_df

def plot_feature_distribution_across_fovs(
    zroots,
    experiment=0,
    well=0,
    feature="area",
    channel="BF",
    n_fovs=9,
    exclude_fovs=None,
    use_density=True,
    bins=40,
    zscore_feature=False,
    absolute_value=False,
    log_transform=None,   # None, "log10", or "log2"
    alpha=0.35,
    histtype="stepfilled",   # "stepfilled", "step", "bar"
    linewidth=1.5,
    colors=None,
    figsize=(8, 6),
    title=None,
    show_title=True,
    xlabel=None,
    ylabel=None,
    axis_color="black",
    font_family=None,
    font_size=10,
    title_size=None,
    xlim=None,
    ylim=None,
    legend=True,
    legend_loc="best",
    save=False,
    figtitle=None
):
    """
    Overlay histograms of a single-cell feature across fields of view for one strain.

    Parameters
    ----------
    zroots : list
        List of ExperimentStore objects.
    experiment : int
        Condition / experiment index.
    well : int
        Well / strain index.
    feature : str
        Feature name from get_cell_properties(...), or "pixel_intensity".
    channel : str
        "BF" or "CY5".
    n_fovs : int
        Number of FoVs to include.
    exclude_fovs : list[int] or None
        FoVs to skip.
    use_density : bool
        If True, normalize histograms to density.
    bins : int or sequence
        Histogram bins.
    zscore_feature : bool
        If True, z-score the feature values within each FoV before plotting.
    absolute_value : bool
        If True, take absolute value before plotting.
    log_transform : str or None
        Optional log transform: None, "log10", or "log2".
    alpha : float
        Histogram alpha.
    histtype : str
        Matplotlib histtype.
    linewidth : float
        Line width for histogram outlines.
    colors : list or None
        One color per FoV. If None, matplotlib default cycle is used.
    figsize : tuple
        Figure size.
    title : str or None
        Plot title.
    show_title : bool
        Whether to show title.
    xlabel, ylabel : str or None
        Axis labels. Auto-generated if None.
    axis_color : str
        Axis spine / tick color.
    font_family : str or None
        Optional font family.
    font_size : int
        Base font size.
    title_size : int or None
        Title font size override.
    xlim, ylim : tuple or None
        Optional axis limits.
    legend : bool
        Whether to show legend.
    legend_loc : str
        Legend location.
    save : bool
        Whether to save png/pdf/svg.
    figtitle : str or None
        Root filename for saving.

    Returns
    -------
    fig, ax, feature_data_by_fov
        feature_data_by_fov is a dict mapping fov -> 1D np.ndarray
    """

    if exclude_fovs is None:
        exclude_fovs = []

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    feature_data_by_fov = {}

    for fov in range(n_fovs):
        if fov in exclude_fovs:
            continue

        vals = None

        if feature == "pixel_intensity":
            try:
                label_arr = zroots[experiment].get_label_image(fov, well, channel)
                mask_arr = label_arr != 0
                proc_img = zroots[experiment].get_processed_intensities(fov, well, channel)
                vals = proc_img[mask_arr]
            except Exception:
                vals = None
        else:
            try:
                props = zroots[experiment].get_cell_properties(fov, well, channel)
                if props is not None and len(props) > 0:
                    current_names = props.columns.tolist()
                    if len(current_names) > 0 and current_names[0] == "label":
                        current_names[0] = "count"
                        props = props.copy()
                        props.columns = current_names

                    if feature not in props.columns:
                        raise ValueError(
                            f"Feature '{feature}' not found in channel '{channel}'. "
                            f"Available features include: {props.columns.tolist()}"
                        )

                    vals = np.asarray(props[feature].astype("float32"))
            except Exception:
                vals = None

        if vals is None:
            continue

        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            continue

        if absolute_value:
            vals = np.abs(vals)

        if log_transform is not None:
            eps = 1e-8
            if log_transform == "log10":
                vals = np.log10(vals + eps)
            elif log_transform == "log2":
                vals = np.log2(vals + eps)
            else:
                raise ValueError("log_transform must be None, 'log10', or 'log2'")

        if zscore_feature:
            vals = zscore(vals, nan_policy="omit")
            vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            continue

        feature_data_by_fov[fov] = vals

    if len(feature_data_by_fov) == 0:
        raise ValueError(
            f"No valid data found for feature='{feature}', channel='{channel}', "
            f"experiment={experiment}, well={well}"
        )

    fig, ax = plt.subplots(figsize=figsize)

    sorted_fovs = sorted(feature_data_by_fov.keys())

    for i, fov in enumerate(sorted_fovs):
        vals = feature_data_by_fov[fov]
        color = colors[i % len(colors)]

        ax.hist(
            vals,
            bins=bins,
            density=use_density,
            alpha=alpha,
            histtype=histtype,
            linewidth=linewidth,
            color=color,
            label=f"FoV {fov}"
        )

    # labels
    if xlabel is None:
        xlabel = feature
        if feature != "pixel_intensity":
            xlabel += f" ({channel.lower()})"
        if log_transform is not None:
            xlabel = f"{log_transform}({xlabel})"
        if zscore_feature:
            xlabel = f"z-scored {xlabel}"

    if ylabel is None:
        ylabel = "Density" if use_density else "Cell count"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # title
    if show_title:
        if title is None:
            title = f"Feature distribution across FoVs | Exp {experiment} | Well {well} | {feature} | {channel}"
        if font_family is not None:
            ax.set_title(title, fontfamily=font_family, fontsize=title_size or font_size)
        else:
            ax.set_title(title, fontsize=title_size or font_size)

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

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if legend:
        leg = ax.legend(loc=legend_loc, frameon=False)
        if font_family is not None:
            for text in leg.get_texts():
                text.set_fontfamily(font_family)
                text.set_fontsize(font_size)
        else:
            for text in leg.get_texts():
                text.set_fontsize(font_size)
    plt.tight_layout()
    plt.show()

    if save:
        if figtitle is None:
            safe_feature = (
                str(feature)
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
                .replace("-", "_")
            )
            figtitle = f"feature_distribution_across_fovs_exp_{experiment}_well_{well}_{channel}_{safe_feature}"

        fig.savefig("plots/" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, feature_data_by_fov

def compute_variance_partitioning_by_feature(
    training_data,
    training_feature_labels,
    condition_order=None,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    feature_subset=None,
    typ=2,
):
    """
    Partition variance for each feature into:
        strain
        environment
        strain:environment
        residual

    using an OLS model:
        value ~ C(strain) + C(environment) + C(strain):C(environment)

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Dict mapping condition -> array of shape (n_rows, n_features)
        Usually training_data_finite
    training_feature_labels : list[str]
        Feature labels matching training_data column order
    condition_order : list[str] or None
        Order of conditions to analyze. If None, uses training_data.keys()
    n_wells : int
        Number of wells
    n_fovs : int or None
        Number of FoVs per well. If None, inferred from n_rows // n_wells
    dead_ids : list[int] or None
        Row indices expanded across FoVs; used to infer dead wells
    feature_subset : list[str] or None
        Optional subset of features to analyze
    typ : int
        ANOVA type passed to sm.stats.anova_lm (usually 2)

    Returns
    -------
    results_df : pd.DataFrame
        Long-form dataframe with one row per feature and variance fractions:
            feature
            ss_strain
            ss_environment
            ss_interaction
            ss_residual
            frac_strain
            frac_environment
            frac_interaction
            frac_residual
    """

    if condition_order is None:
        condition_order = list(training_data.keys())

    if dead_ids is None:
        dead_ids = []

    dead_wells = sorted(set([d % n_wells for d in dead_ids]))

    if feature_subset is None:
        feature_idx = np.arange(len(training_feature_labels))
        feature_names = list(training_feature_labels)
    else:
        feature_names = list(feature_subset)
        feature_idx = [training_feature_labels.index(f) for f in feature_subset]

    # ----------------------------
    # Build master dataframe
    # ----------------------------
    records = []

    for condition in condition_order:
        dat = np.asarray(training_data[condition], dtype=float)
        n_rows, n_features = dat.shape

        if n_fovs is None:
            this_n_fovs = n_rows // n_wells
        else:
            this_n_fovs = n_fovs

        for well in range(n_wells):
            if well in dead_wells:
                continue

            for fov in range(this_n_fovs):
                row_idx = well + (fov * n_wells)
                row_vals = dat[row_idx, :]

                if not np.all(np.isfinite(row_vals)):
                    continue

                rec = {
                    "strain": str(well),
                    "environment": str(condition),
                    "fov": int(fov),
                }

                for fi, fname in zip(feature_idx, feature_names):
                    rec[fname] = row_vals[fi]

                records.append(rec)

    master_df = pd.DataFrame(records)

    if len(master_df) == 0:
        raise ValueError("No valid rows found for variance partitioning.")

    # ----------------------------
    # Per-feature ANOVA
    # ----------------------------
    out_rows = []

    for fname in feature_names:
        df = master_df[["strain", "environment", fname]].copy()
        df = df.rename(columns={fname: "value"})

        # fit model
        model = smf.ols(
            "value ~ C(strain) + C(environment) + C(strain):C(environment)",
            data=df
        ).fit()

        anova = sm.stats.anova_lm(model, typ=typ)

        ss_strain = anova.loc["C(strain)", "sum_sq"] if "C(strain)" in anova.index else 0.0
        ss_environment = anova.loc["C(environment)", "sum_sq"] if "C(environment)" in anova.index else 0.0
        ss_interaction = (
            anova.loc["C(strain):C(environment)", "sum_sq"]
            if "C(strain):C(environment)" in anova.index else 0.0
        )
        ss_residual = anova.loc["Residual", "sum_sq"] if "Residual" in anova.index else 0.0

        ss_total = ss_strain + ss_environment + ss_interaction + ss_residual

        if np.isclose(ss_total, 0):
            frac_strain = np.nan
            frac_environment = np.nan
            frac_interaction = np.nan
            frac_residual = np.nan
        else:
            frac_strain = ss_strain / ss_total
            frac_environment = ss_environment / ss_total
            frac_interaction = ss_interaction / ss_total
            frac_residual = ss_residual / ss_total

        out_rows.append({
            "feature": fname,
            "ss_strain": ss_strain,
            "ss_environment": ss_environment,
            "ss_interaction": ss_interaction,
            "ss_residual": ss_residual,
            "frac_strain": frac_strain,
            "frac_environment": frac_environment,
            "frac_interaction": frac_interaction,
            "frac_residual": frac_residual,
        })

    results_df = pd.DataFrame(out_rows)
    return results_df

def plot_variance_partitioning_heatmap(
    variance_df,
    orientation="horizontal",
    sort_features=True,
    sort_by="environment",
    figsize=(14, 8),
    cmap="apc:orange_sage_r",
    title="Variance partitioning by feature",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    xtick_rotation=45,
    ytick_rotation=0,
    axis_color="black",
    label_color="black",
    xtick_colors=None,
    ytick_colors=None,
    cbar=True,
    cbar_label="Fraction of variance explained",
    cbar_shrink=1.0,
    cbar_aspect=30,
    vmin=0,
    vmax=1,
    save=False,
    figtitle="variance_partitioning_heatmap",
):
    """
    Plot heatmap of variance fractions across features.

    variance_df must contain:
        feature
        frac_strain
        frac_environment
        frac_interaction
        frac_residual
    """

    required_cols = [
        "feature",
        "frac_strain",
        "frac_environment",
        "frac_interaction",
        "frac_residual"
    ]
    for col in required_cols:
        if col not in variance_df.columns:
            raise ValueError(f"variance_df missing required column: {col}")

    if xtick_fontsize is None:
        xtick_fontsize = font_size
    if ytick_fontsize is None:
        ytick_fontsize = font_size

    plot_df = variance_df[
        ["feature", "frac_strain", "frac_environment", "frac_interaction", "frac_residual"]
    ].copy()

    plot_df = plot_df.set_index("feature")
    plot_df.columns = ["strain", "environment", "interaction", "residual"]

    if sort_features:
        sort_col = f"frac_{sort_by}" if f"frac_{sort_by}" in variance_df.columns else None
        if sort_col is None:
            raise ValueError("sort_by must be one of: strain, environment, interaction, residual")
        order = variance_df.sort_values(sort_col, ascending=False)["feature"]
        plot_df = plot_df.loc[order]

    if orientation == "vertical":
        plot_df = plot_df.T
    elif orientation != "horizontal":
        raise ValueError('orientation must be "horizontal" or "vertical"')

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

    if cbar and cbar_label is not None:
        hm.collections[0].colorbar.set_label(
            cbar_label,
            fontsize=font_size,
            fontfamily=font_family
        )

    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize or font_size,
            fontfamily=font_family
        )

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
        rotation=ytick_rotation,
        fontsize=ytick_fontsize,
        fontfamily=font_family
    )

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if xtick_colors is not None:
        if len(xtick_colors) != len(ax.get_xticklabels()):
            raise ValueError("xtick_colors length mismatch")
        for tick, color in zip(ax.get_xticklabels(), xtick_colors):
            tick.set_color(color)

    if ytick_colors is not None:
        if len(ytick_colors) != len(ax.get_yticklabels()):
            raise ValueError("ytick_colors length mismatch")
        for tick, color in zip(ax.get_yticklabels(), ytick_colors):
            tick.set_color(color)

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", color=axis_color)
    ax.tick_params(axis="y", color=axis_color)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, plot_df

'''

def predict_with_baseline(
    data,
    train_conditions,
    test_condition,
    baseline_type="row_mean",
):
    """
    Predict a held-out condition using a simple baseline.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        condition -> array of shape (n_rows, n_features)
    train_conditions : list[str]
        Conditions used as input
    test_condition : str
        Condition to predict
    baseline_type : str
        "global_mean" : predict each feature using the grand mean across all rows and train conditions
        "row_mean"    : predict each row using the mean across train conditions for that same row

    Returns
    -------
    y_pred : np.ndarray
        Predicted matrix of shape (n_rows, n_features)
    metrics : dict
        Same basic structure as train_and_predict
    """

    y_true = data[test_condition]

    train_arrays = [data[c] for c in train_conditions]

    if baseline_type == "global_mean":
        stacked = np.vstack(train_arrays)  # (n_train_conditions * n_rows, n_features)
        mean_vec = np.nanmean(stacked, axis=0)
        y_pred = np.tile(mean_vec, (y_true.shape[0], 1))

    elif baseline_type == "row_mean":
        stacked = np.stack(train_arrays, axis=0)  # (n_train_conditions, n_rows, n_features)
        y_pred = np.nanmean(stacked, axis=0)

    else:
        raise ValueError("baseline_type must be 'global_mean' or 'row_mean'")

    metrics = {
        "r2_overall": r2_score(y_true, y_pred),
        "rmse_overall": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2_per_feature": [
            r2_score(y_true[:, i], y_pred[:, i])
            for i in range(y_true.shape[1])
        ]
    }

    return y_pred, metrics
'''

def predict_with_baseline(
    data,
    train_conditions,
    test_condition,
    baseline_type="within_strain_mean",
):
    """
    Predict a held-out condition using a simple baseline.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        condition -> array of shape (n_rows, n_features)
    train_conditions : list[str]
        Conditions used as input
    test_condition : str
        Condition to predict
    baseline_type : str
        "global_mean"         : predict each feature using the grand mean across all rows and train conditions
        "row_mean"            : predict each row using the mean across train conditions for that same row
                                (legacy name; equivalent to within_strain_mean if row alignment is preserved)
        "within_strain_mean"  : predict each row using the mean across training conditions for that same row

    Returns
    -------
    y_pred : np.ndarray
        Predicted matrix of shape (n_rows, n_features)
    metrics : dict
        Dict with:
            r2_overall
            rmse_overall
            r2_per_feature
    """

    y_true = np.asarray(data[test_condition], dtype=float)
    train_arrays = [np.asarray(data[c], dtype=float) for c in train_conditions]

    if baseline_type == "global_mean":
        stacked = np.vstack(train_arrays)
        mean_vec = np.nanmean(stacked, axis=0)
        y_pred = np.tile(mean_vec, (y_true.shape[0], 1))

    elif baseline_type in ("row_mean", "within_strain_mean"):
        # same row across conditions: preserves strain/FoV identity
        stacked = np.stack(train_arrays, axis=0)   # (n_conditions, n_rows, n_features)
        y_pred = np.nanmean(stacked, axis=0)

    else:
        raise ValueError(
            "baseline_type must be 'global_mean', 'row_mean', or 'within_strain_mean'"
        )

    metrics = {
        "r2_overall": r2_score(y_true, y_pred),
        "rmse_overall": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2_per_feature": [
            r2_score(y_true[:, i], y_pred[:, i])
            for i in range(y_true.shape[1])
        ]
    }

    return y_pred, metrics

def compare_models_condition_holdout(
    training_data,
    exp_conditions=None,
    ridge_alpha=100,
    include_global_mean=True,
    include_row_mean=True,
):
    """
    Compare ridge to simple baselines across all held-out conditions.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Usually training_data_finite
    exp_conditions : list[str] or None
        Condition order. If None, uses training_data.keys()
    ridge_alpha : float
        Alpha for ridge regression
    include_global_mean : bool
        Whether to include global mean baseline
    include_row_mean : bool
        Whether to include row-mean baseline

    Returns
    -------
    results : dict
        Nested dict:
            results["ridge"]["overall_r2"]
            results["ridge"]["rmse"]
            results["ridge"]["r2_per_feature"]
            etc.
    """

    if exp_conditions is None:
        exp_conditions = list(training_data.keys())

    model_names = ["ridge"]
    if include_global_mean:
        model_names.append("global_mean")
    if include_row_mean:
        model_names.append("row_mean")

    n_conditions = len(exp_conditions)
    n_features = training_data[exp_conditions[0]].shape[1]

    results = {}
    for model_name in model_names:
        results[model_name] = {
            "conditions": exp_conditions,
            "overall_r2": np.zeros(n_conditions),
            "rmse": np.zeros(n_conditions),
            "r2_per_feature": np.zeros((n_conditions, n_features)),
        }

    for i, test_condition in enumerate(exp_conditions):
        train_conditions = [c for c in exp_conditions if c != test_condition]

        # ridge
        y_pred, metrics, model = train_and_predict(
            training_data,
            train_conditions,
            test_condition,
            model_type="ridge",
            alpha=ridge_alpha
        )
        results["ridge"]["overall_r2"][i] = metrics["r2_overall"]
        results["ridge"]["rmse"][i] = metrics["rmse_overall"]
        results["ridge"]["r2_per_feature"][i, :] = np.array(metrics["r2_per_feature"])

        # global mean baseline
        if include_global_mean:
            y_pred, metrics = predict_with_baseline(
                training_data,
                train_conditions,
                test_condition,
                baseline_type="global_mean"
            )
            results["global_mean"]["overall_r2"][i] = metrics["r2_overall"]
            results["global_mean"]["rmse"][i] = metrics["rmse_overall"]
            results["global_mean"]["r2_per_feature"][i, :] = np.array(metrics["r2_per_feature"])

        # row mean baseline
        if include_row_mean:
            y_pred, metrics = predict_with_baseline(
                training_data,
                train_conditions,
                test_condition,
                baseline_type="row_mean"
            )
            results["row_mean"]["overall_r2"][i] = metrics["r2_overall"]
            results["row_mean"]["rmse"][i] = metrics["rmse_overall"]
            results["row_mean"]["r2_per_feature"][i, :] = np.array(metrics["r2_per_feature"])

    return results

def plot_model_comparison_overall(
    comparison_results,
    metric="overall_r2",   # "overall_r2" or "rmse"
    col_titles=None,
    model_colors=None,
    figsize=(10, 5),
    title=None,
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    xtick_rotation=45,
    axis_color="black",
    save=False,
    figtitle="model_comparison_overall",
):
    """
    Plot overall model performance across held-out conditions.
    """

    model_names = list(comparison_results.keys())
    conditions = comparison_results[model_names[0]]["conditions"]

    if col_titles is None:
        col_titles = conditions

    if model_colors is None:
        model_colors = {
            "ridge": apc.aster,
            "global_mean": apc.lapis,
            "row_mean": apc.vital,
        }

    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=figsize)

    for model_name in model_names:
        vals = comparison_results[model_name][metric]
        color = model_colors.get(model_name, None)
        ax.plot(
            x,
            vals,
            marker="o",
            linewidth=2,
            label=model_name,
            color=color
        )

    ax.set_xticks(x)
    ax.set_xticklabels(col_titles, rotation=xtick_rotation, ha="right")

    if metric == "overall_r2":
        ylabel = "Overall R²"
    elif metric == "rmse":
        ylabel = "RMSE"
    else:
        ylabel = metric

    ax.set_ylabel(ylabel)

    if show_title:
        if title is None:
            title = f"Model comparison: {ylabel}"
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    ax.legend(frameon=False)

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
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax

def plot_model_comparison_per_feature(
    comparison_results,
    training_feature_labels,
    model_names=("ridge", "row_mean", "global_mean"),
    condition_average=True,
    figsize=(8, 12),
    orientation="horizontal",
    sort_features=True,
    sort_by_model="ridge",
    title="Per-feature model comparison",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    colors=None,
    axis_color="black",
    save=False,
    figtitle="model_comparison_per_feature",
):
    """
    Compare models by average per-feature R² across conditions.
    """

    if colors is None:
        colors = {
            "ridge": apc.aster,
            "row_mean": apc.vital,
            "global_mean": apc.lapis,
        }

    # build matrix
    per_model = {}
    for model_name in model_names:
        mat = np.array(comparison_results[model_name]["r2_per_feature"])
        if condition_average:
            per_model[model_name] = np.nanmean(mat, axis=0)
        else:
            raise ValueError("Currently only condition_average=True is supported")

    plot_df = pd.DataFrame(per_model, index=training_feature_labels)

    if sort_features:
        plot_df = plot_df.sort_values(sort_by_model, ascending=False)

    labels = plot_df.index.to_numpy()
    pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)

    width = 0.25
    model_names = list(model_names)

    if orientation == "horizontal":
        for i, model_name in enumerate(model_names):
            ax.barh(
                pos + (i - (len(model_names)-1)/2) * width,
                plot_df[model_name].to_numpy(),
                height=width,
                label=model_name,
                color=colors.get(model_name, None)
            )

        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Mean per-feature R²")

    elif orientation == "vertical":
        for i, model_name in enumerate(model_names):
            ax.bar(
                pos + (i - (len(model_names)-1)/2) * width,
                plot_df[model_name].to_numpy(),
                width=width,
                label=model_name,
                color=colors.get(model_name, None)
            )

        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Mean per-feature R²")

    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    ax.legend(frameon=False)

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
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, plot_df

'''

def compare_models_condition_holdout_cv(
    training_data,
    exp_conditions=None,
    ridge_alpha=100,
    n_splits=10,
    random_state=42,
    include_global_mean=True,
    include_row_mean=True,
):
    """
    Compare ridge to simple baselines using out-of-sample CV within each held-out condition.

    Returns
    -------
    results : dict
        results[model_name]["overall_r2"]
        results[model_name]["rmse"]
        results[model_name]["r2_per_feature"]
    """

    if exp_conditions is None:
        exp_conditions = list(training_data.keys())

    model_names = ["ridge"]
    if include_global_mean:
        model_names.append("global_mean")
    if include_row_mean:
        model_names.append("row_mean")

    n_conditions = len(exp_conditions)
    n_features = training_data[exp_conditions[0]].shape[1]

    results = {}
    for model_name in model_names:
        results[model_name] = {
            "conditions": exp_conditions,
            "overall_r2": np.zeros(n_conditions),
            "rmse": np.zeros(n_conditions),
            "r2_per_feature": np.zeros((n_conditions, n_features)),
        }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for i, test_condition in enumerate(exp_conditions):
        train_conditions = [c for c in exp_conditions if c != test_condition]

        X_full = np.hstack([training_data[c] for c in train_conditions])
        y_full = training_data[test_condition]

        # containers for CV predictions
        y_true_all = []
        y_pred_ridge_all = []
        y_pred_global_all = []
        y_pred_row_all = []

        for train_idx, test_idx in kf.split(X_full):
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y_full[train_idx], y_full[test_idx]

            # ---- ridge ----
            ridge_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha))
            ])
            ridge_pipeline.fit(X_train, y_train)
            y_pred_ridge = ridge_pipeline.predict(X_test)

            y_true_all.append(y_test)
            y_pred_ridge_all.append(y_pred_ridge)

            # ---- global mean baseline ----
            if include_global_mean:
                mean_vec = np.nanmean(y_train, axis=0)
                y_pred_global = np.tile(mean_vec, (len(test_idx), 1))
                y_pred_global_all.append(y_pred_global)

            # ---- row mean baseline ----
            if include_row_mean:
                # row-matched baseline using training conditions only
                stacked_train_conditions = np.stack(
                    [training_data[c][test_idx] for c in train_conditions],
                    axis=0
                )  # (n_train_conditions, n_test_rows, n_features)
                y_pred_row = np.nanmean(stacked_train_conditions, axis=0)
                y_pred_row_all.append(y_pred_row)

        y_true_all = np.vstack(y_true_all)
        y_pred_ridge_all = np.vstack(y_pred_ridge_all)

        # ridge metrics
        results["ridge"]["overall_r2"][i] = r2_score(y_true_all, y_pred_ridge_all)
        results["ridge"]["rmse"][i] = np.sqrt(mean_squared_error(y_true_all, y_pred_ridge_all))
        results["ridge"]["r2_per_feature"][i, :] = np.array([
            r2_score(y_true_all[:, j], y_pred_ridge_all[:, j])
            for j in range(n_features)
        ])

        if include_global_mean:
            y_pred_global_all = np.vstack(y_pred_global_all)
            results["global_mean"]["overall_r2"][i] = r2_score(y_true_all, y_pred_global_all)
            results["global_mean"]["rmse"][i] = np.sqrt(mean_squared_error(y_true_all, y_pred_global_all))
            results["global_mean"]["r2_per_feature"][i, :] = np.array([
                r2_score(y_true_all[:, j], y_pred_global_all[:, j])
                for j in range(n_features)
            ])

        if include_row_mean:
            y_pred_row_all = np.vstack(y_pred_row_all)
            results["row_mean"]["overall_r2"][i] = r2_score(y_true_all, y_pred_row_all)
            results["row_mean"]["rmse"][i] = np.sqrt(mean_squared_error(y_true_all, y_pred_row_all))
            results["row_mean"]["r2_per_feature"][i, :] = np.array([
                r2_score(y_true_all[:, j], y_pred_row_all[:, j])
                for j in range(n_features)
            ])

    return results
'''

def compare_models_condition_holdout_cv(
    training_data,
    exp_conditions=None,
    ridge_alpha=3000,
    n_splits=16,
    random_state=42,
    include_global_mean=True,
    include_within_strain_mean=True,
    n_wells=96,
    n_fovs=None,
):
    """
    Compare ridge regression to simple baselines using out-of-sample CV
    within each held-out condition.

    Models
    ------
    ridge
        Ridge regression using all other conditions as predictors.

    global_mean
        Predict each feature using the grand mean of y_train in that fold.

    within_strain_mean
        Predict each row using the mean phenotype of the SAME strain (well)
        across all FoVs and all training conditions. This preserves strain identity
        but does not use exact FoV alignment.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Dict mapping condition -> array of shape (n_rows, n_features).
        Usually this should be training_data_finite.
    exp_conditions : list[str] or None
        Condition order. If None, uses training_data.keys().
    ridge_alpha : float
        Alpha for ridge regression.
    n_splits : int
        Number of cross-validation folds.
    random_state : int
        Random seed.
    include_global_mean : bool
        Whether to include the global mean baseline.
    include_within_strain_mean : bool
        Whether to include the within-strain mean baseline.
    n_wells : int
        Number of wells (strains).
    n_fovs : int or None
        Number of FoVs per well. If None, inferred from row count.

    Returns
    -------
    results : dict
        Nested dict with keys for each model.
        Each model contains:
            "conditions"      : list of conditions
            "overall_r2"      : np.ndarray, shape (n_conditions,)
            "rmse"            : np.ndarray, shape (n_conditions,)
            "r2_per_feature"  : np.ndarray, shape (n_conditions, n_features)
    """

    if exp_conditions is None:
        exp_conditions = list(training_data.keys())

    model_names = ["ridge"]
    if include_global_mean:
        model_names.append("global_mean")
    if include_within_strain_mean:
        model_names.append("within_strain_mean")

    n_conditions = len(exp_conditions)
    n_features = training_data[exp_conditions[0]].shape[1]

    results = {}
    for model_name in model_names:
        results[model_name] = {
            "conditions": exp_conditions,
            "overall_r2": np.zeros(n_conditions),
            "rmse": np.zeros(n_conditions),
            "r2_per_feature": np.zeros((n_conditions, n_features)),
        }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for i, test_condition in enumerate(exp_conditions):
        train_conditions = [c for c in exp_conditions if c != test_condition]

        X_full = np.hstack([np.asarray(training_data[c], dtype=float) for c in train_conditions])
        y_full = np.asarray(training_data[test_condition], dtype=float)

        if n_fovs is None:
            this_n_fovs = y_full.shape[0] // n_wells
        else:
            this_n_fovs = n_fovs

        # containers for out-of-sample predictions
        y_true_all = []
        y_pred_ridge_all = []
        y_pred_global_all = []
        y_pred_within_all = []

        for train_idx, test_idx in kf.split(X_full):
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y_full[train_idx], y_full[test_idx]

            # --------------------
            # Ridge
            # --------------------
            ridge_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha))
            ])
            ridge_pipeline.fit(X_train, y_train)
            y_pred_ridge = ridge_pipeline.predict(X_test)

            y_true_all.append(y_test)
            y_pred_ridge_all.append(y_pred_ridge)

            # --------------------
            # Global mean baseline
            # --------------------
            if include_global_mean:
                mean_vec = np.nanmean(y_train, axis=0)
                y_pred_global = np.tile(mean_vec, (len(test_idx), 1))
                y_pred_global_all.append(y_pred_global)

            # --------------------
            # True within-strain baseline
            # --------------------
            if include_within_strain_mean:
                y_pred_within = np.zeros_like(y_test)

                for j, row in enumerate(test_idx):
                    well = row % n_wells

                    preds = []
                    for c in train_conditions:
                        cond_data = np.asarray(training_data[c], dtype=float)

                        # all FoVs corresponding to this well
                        rows_for_well = [well + f * n_wells for f in range(this_n_fovs)]
                        well_vals = cond_data[rows_for_well, :]

                        preds.append(np.nanmean(well_vals, axis=0))

                    # average across training conditions
                    y_pred_within[j] = np.nanmean(preds, axis=0)

                y_pred_within_all.append(y_pred_within)

        # concatenate all held-out predictions
        y_true_all = np.vstack(y_true_all)
        y_pred_ridge_all = np.vstack(y_pred_ridge_all)

        # --------------------
        # Ridge metrics
        # --------------------
        results["ridge"]["overall_r2"][i] = r2_score(y_true_all, y_pred_ridge_all)
        results["ridge"]["rmse"][i] = np.sqrt(mean_squared_error(y_true_all, y_pred_ridge_all))
        results["ridge"]["r2_per_feature"][i, :] = np.array([
            r2_score(y_true_all[:, j], y_pred_ridge_all[:, j])
            for j in range(n_features)
        ])

        # --------------------
        # Global mean metrics
        # --------------------
        if include_global_mean:
            y_pred_global_all = np.vstack(y_pred_global_all)
            results["global_mean"]["overall_r2"][i] = r2_score(y_true_all, y_pred_global_all)
            results["global_mean"]["rmse"][i] = np.sqrt(mean_squared_error(y_true_all, y_pred_global_all))
            results["global_mean"]["r2_per_feature"][i, :] = np.array([
                r2_score(y_true_all[:, j], y_pred_global_all[:, j])
                for j in range(n_features)
            ])

        # --------------------
        # Within-strain mean metrics
        # --------------------
        if include_within_strain_mean:
            y_pred_within_all = np.vstack(y_pred_within_all)
            results["within_strain_mean"]["overall_r2"][i] = r2_score(y_true_all, y_pred_within_all)
            results["within_strain_mean"]["rmse"][i] = np.sqrt(mean_squared_error(y_true_all, y_pred_within_all))
            results["within_strain_mean"]["r2_per_feature"][i, :] = np.array([
                r2_score(y_true_all[:, j], y_pred_within_all[:, j])
                for j in range(n_features)
            ])

    return results

def plot_model_delta_overall(
    comparison_results,
    model_a="ridge",
    model_b="row_mean",
    metric="overall_r2",
    col_titles=None,
    bar_color=apc.aster,
    figsize=(8, 5),
    title=None,
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    xtick_rotation=45,
    axis_color="black",
    label_color="black",
    xtick_colors=None,
    show_values=False,
    value_fontsize=None,
    zero_line_color="gray",
    zero_line_style="--",
    save=False,
    figtitle="model_delta_overall",
):
    """
    Plot condition-wise delta between two models.

    For R²:
        delta = model_a - model_b
        positive means model_a is better

    For RMSE:
        delta = model_b - model_a
        positive means model_a is better (lower error)

    Parameters
    ----------
    comparison_results : dict
        Output from compare_models_condition_holdout_cv(...)
    model_a : str
        Usually "ridge"
    model_b : str
        Usually "row_mean"
    metric : str
        "overall_r2" or "rmse"
    col_titles : list[str] or None
        Display labels for conditions
    """

    if model_a not in comparison_results:
        raise ValueError(f"{model_a} not found in comparison_results")
    if model_b not in comparison_results:
        raise ValueError(f"{model_b} not found in comparison_results")

    conditions = comparison_results[model_a]["conditions"]

    if col_titles is None:
        col_titles = conditions

    if len(col_titles) != len(conditions):
        raise ValueError("col_titles length mismatch")

    vals_a = np.array(comparison_results[model_a][metric], dtype=float)
    vals_b = np.array(comparison_results[model_b][metric], dtype=float)

    if metric == "overall_r2":
        delta = vals_a - vals_b
        ylabel = f"ΔR² ({model_a} - {model_b})"
    elif metric == "rmse":
        delta = vals_b - vals_a
        ylabel = f"ΔRMSE advantage ({model_a} better if positive)"
    else:
        raise ValueError("metric must be 'overall_r2' or 'rmse'")

    if value_fontsize is None:
        value_fontsize = font_size

    x = np.arange(len(delta))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, delta, color=bar_color)

    ax.axhline(0, color=zero_line_color, linestyle=zero_line_style, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(col_titles, rotation=xtick_rotation, ha="right")
    ax.set_ylabel(ylabel)

    if show_title:
        if title is None:
            title = f"{model_a} vs {model_b}"
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    if show_values:
        for rect, val in zip(bars, delta):
            y = rect.get_height()
            va = "bottom" if y >= 0 else "top"
            offset = 0.01 * max(1, np.nanmax(np.abs(delta)))
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y + (offset if y >= 0 else -offset),
                f"{val:.2f}",
                ha="center",
                va=va,
                fontsize=value_fontsize,
                fontfamily=font_family,
                color=label_color
            )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if xtick_colors is not None:
        if len(xtick_colors) != len(ax.get_xticklabels()):
            raise ValueError("xtick_colors length mismatch")
        for tick, color in zip(ax.get_xticklabels(), xtick_colors):
            tick.set_color(color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, delta

def plot_model_delta_per_feature(
    comparison_results,
    training_feature_labels,
    model_a="ridge",
    model_b="row_mean",
    average_across_conditions=True,
    orientation="horizontal",
    sort_features=True,
    figsize=(8, 12),
    bar_color=apc.aster,
    title=None,
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    rotation=45,
    axis_color="black",
    label_color="black",
    zero_line_color="gray",
    zero_line_style="--",
    save=False,
    figtitle="model_delta_per_feature",
):
    """
    Plot delta per feature:
        ΔR² = model_a - model_b

    Positive means model_a performs better.
    """

    if model_a not in comparison_results:
        raise ValueError(f"{model_a} not found in comparison_results")
    if model_b not in comparison_results:
        raise ValueError(f"{model_b} not found in comparison_results")

    mat_a = np.array(comparison_results[model_a]["r2_per_feature"], dtype=float)
    mat_b = np.array(comparison_results[model_b]["r2_per_feature"], dtype=float)

    if average_across_conditions:
        delta = np.nanmean(mat_a - mat_b, axis=0)
    else:
        raise ValueError("Currently only average_across_conditions=True is supported")

    if len(delta) != len(training_feature_labels):
        raise ValueError("training_feature_labels length mismatch")

    labels = np.array(training_feature_labels)

    if sort_features:
        order = np.argsort(delta)[::-1]
        delta = delta[order]
        labels = labels[order]

    pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)

    if orientation == "horizontal":
        ax.barh(pos, delta, color=bar_color)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"ΔR² ({model_a} - {model_b})")
        ax.axvline(0, color=zero_line_color, linestyle=zero_line_style, linewidth=1)

    elif orientation == "vertical":
        ax.bar(pos, delta, color=bar_color)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=rotation, ha="right")
        ax.set_ylabel(f"ΔR² ({model_a} - {model_b})")
        ax.axhline(0, color=zero_line_color, linestyle=zero_line_style, linewidth=1)

    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    if show_title:
        if title is None:
            title = f"Per-feature gain: {model_a} over {model_b}"
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, delta, labels

def merge_variance_and_ridge_results(
    variance_df,
    comparison_results,
    training_feature_labels,
    model_name="ridge",
):
    """
    Merge variance partitioning with mean per-feature ridge performance.

    Parameters
    ----------
    variance_df : pd.DataFrame
        Output from compute_variance_partitioning_by_feature(...)
    comparison_results : dict
        Output from compare_models_condition_holdout_cv(...)
    training_feature_labels : list[str]
        Feature labels in model order
    model_name : str
        Which model to use, usually "ridge"

    Returns
    -------
    merged_df : pd.DataFrame
        variance_df plus:
            mean_r2
            std_r2
    """

    if model_name not in comparison_results:
        raise ValueError(f"{model_name} not found in comparison_results")

    r2_mat = np.array(comparison_results[model_name]["r2_per_feature"], dtype=float)

    if r2_mat.shape[1] != len(training_feature_labels):
        raise ValueError("training_feature_labels length mismatch with r2_per_feature")

    ridge_df = pd.DataFrame({
        "feature": training_feature_labels,
        "mean_r2": np.nanmean(r2_mat, axis=0),
        "std_r2": np.nanstd(r2_mat, axis=0),
    })

    merged_df = variance_df.merge(ridge_df, on="feature", how="inner")
    return merged_df

def plot_ridge_vs_variance_component(
    merged_df,
    component="frac_environment",
    figsize=(6, 5),
    color=apc.aster,
    alpha=0.8,
    title=None,
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    axis_color="black",
    label_color="black",
    annotate_points=False,
    annotation_fontsize=8,
    save=False,
    figtitle="ridge_vs_variance_component",
):
    """
    Scatterplot: mean ridge R² vs one variance component.
    """

    if component not in merged_df.columns:
        raise ValueError(f"{component} not found in merged_df")

    x = merged_df[component].to_numpy(dtype=float)
    y = merged_df["mean_r2"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color=color, alpha=alpha)

    ax.set_xlabel(component.replace("_", " "))
    ax.set_ylabel("Mean per-feature ridge R²")

    if show_title:
        if title is None:
            title = f"Ridge performance vs {component.replace('_', ' ')}"
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    if annotate_points:
        for _, row in merged_df.iterrows():
            ax.text(
                row[component],
                row["mean_r2"],
                row["feature"],
                fontsize=annotation_fontsize,
                fontfamily=font_family,
                color=label_color
            )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax

def plot_ridge_variance_component_correlations(
    merged_df,
    components=("frac_strain", "frac_environment", "frac_interaction", "frac_residual"),
    method="pearson",
    figsize=(6, 4),
    color=apc.aster,
    title="Correlation with ridge predictability",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    axis_color="black",
    label_color="black",
    xtick_rotation=45,
    save=False,
    figtitle="ridge_variance_component_correlations",
):
    """
    Plot correlation between mean ridge R² and variance fractions.
    """

    corrs = []
    valid_components = []

    for comp in components:
        if comp not in merged_df.columns:
            continue

        x = merged_df[comp].to_numpy(dtype=float)
        y = merged_df["mean_r2"].to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            corr = np.nan
        else:
            if method == "pearson":
                corr = np.corrcoef(x, y)[0, 1]
            elif method == "spearman":
                corr = pd.Series(x).corr(pd.Series(y), method="spearman")
            else:
                raise ValueError("method must be 'pearson' or 'spearman'")

        corrs.append(corr)
        valid_components.append(comp.replace("frac_", ""))

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(valid_components))
    ax.bar(x, corrs, color=color)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(valid_components, rotation=xtick_rotation, ha="right")
    ax.set_ylabel(f"{method} correlation with mean ridge R²")

    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    corr_df = pd.DataFrame({
        "component": valid_components,
        "correlation": corrs
    })

    return fig, ax, corr_df

def compute_global_strain_outlier_scores(
    training_data,
    condition_order=None,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    distance_metric="euclidean",   # "euclidean" or "cosine"
    aggregate="mean",              # "mean", "max", or "median"
    zscore_features_across_strains=True,
    return_condition_distances=False,
):
    """
    Compute a global phenotypic outlier score for each strain (well).

    Each strain is represented by its FoV-level mean phenotype within each condition.
    For each condition, the strain's phenotype is compared to the centroid of all valid strains.
    Distances are then aggregated across conditions.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Dict mapping condition -> array of shape (n_rows, n_features).
        Usually this should be training_data_finite.
    condition_order : list[str] or None
        Order of conditions to use. If None, uses training_data.keys().
    n_wells : int
        Number of wells (strains).
    n_fovs : int or None
        Number of FoVs per well. If None, inferred from n_rows // n_wells.
    dead_ids : list[int] or None
        Expanded dead row indices. Used to infer dead wells to exclude.
    distance_metric : str
        "euclidean" or "cosine".
    aggregate : str
        How to combine condition-wise distances for each strain:
        "mean", "max", or "median".
    zscore_features_across_strains : bool
        If True, z-score strain means within each condition before distance calculation.
        Recommended, since features are on different scales.
    return_condition_distances : bool
        If True, also return the per-condition distance matrix.

    Returns
    -------
    outlier_df : pd.DataFrame
        Columns:
            well
            outlier_score
            mean_distance
            max_distance
            median_distance
            n_conditions_used
        Sorted descending by outlier_score.
    condition_distance_df : pd.DataFrame (optional)
        Rows = wells, columns = conditions, values = distance from centroid.
    """

    if condition_order is None:
        condition_order = list(training_data.keys())

    if dead_ids is None:
        dead_ids = []

    dead_wells = sorted(set([d % n_wells for d in dead_ids]))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]

    if len(valid_wells) == 0:
        raise ValueError("No valid wells found after excluding dead wells.")

    # ----------------------------
    # helper: distance to centroid
    # ----------------------------
    def _distance_to_centroid(X, metric="euclidean"):
        centroid = np.nanmean(X, axis=0)

        if metric == "euclidean":
            d = np.sqrt(np.nansum((X - centroid) ** 2, axis=1))

        elif metric == "cosine":
            d = np.full(X.shape[0], np.nan, dtype=float)

            if not np.all(np.isfinite(centroid)):
                return d

            centroid_norm = np.linalg.norm(centroid)

            for i in range(X.shape[0]):
                xi = X[i]

                if not np.all(np.isfinite(xi)):
                    continue

                xi_norm = np.linalg.norm(xi)

                # both zero vectors -> identical
                if np.isclose(xi_norm, 0) and np.isclose(centroid_norm, 0):
                    d[i] = 0.0
                    continue

                # one zero, one nonzero -> maximally different
                if np.isclose(xi_norm, 0) or np.isclose(centroid_norm, 0):
                    d[i] = 1.0
                    continue

                sim = np.dot(xi, centroid) / (xi_norm * centroid_norm)

                # numerical safety
                sim = np.clip(sim, -1.0, 1.0)

                d[i] = 1.0 - sim

        else:
            raise ValueError("distance_metric must be 'euclidean' or 'cosine'")

        return d

    # ----------------------------
    # compute strain means per condition
    # ----------------------------
    condition_distance_records = {}

    for condition in condition_order:
        dat = np.asarray(training_data[condition], dtype=float)
        n_rows, n_features = dat.shape

        if n_fovs is None:
            this_n_fovs = n_rows // n_wells
        else:
            this_n_fovs = n_fovs

        strain_means = []
        strain_ids_used = []

        for well in valid_wells:
            row_idx = [well + (fov * n_wells) for fov in range(this_n_fovs)]
            block = dat[row_idx, :]   # shape: (n_fovs, n_features)

            if np.all(~np.isfinite(block)):
                continue

            mean_vec = np.nanmean(block, axis=0)

            if not np.any(np.isfinite(mean_vec)):
                continue

            strain_means.append(mean_vec)
            strain_ids_used.append(well)

        if len(strain_means) == 0:
            raise ValueError(f"No valid strain means found for condition {condition}")

        strain_means = np.vstack(strain_means)

        # optional z-score across strains within this condition
        if zscore_features_across_strains:
            strain_means = zscore(strain_means, axis=0, nan_policy="omit")
            # fill any all-NaN columns with 0 after z-scoring
            strain_means = np.where(np.isfinite(strain_means), strain_means, 0.0)

        distances = _distance_to_centroid(strain_means, metric=distance_metric)

        condition_distance_records[condition] = pd.Series(
            distances,
            index=strain_ids_used
        )

    # ----------------------------
    # assemble well x condition table
    # ----------------------------
    condition_distance_df = pd.DataFrame(condition_distance_records)
    condition_distance_df.index.name = "well"
    condition_distance_df = condition_distance_df.sort_index()

    # ----------------------------
    # aggregate across conditions
    # ----------------------------
    mean_distance = condition_distance_df.mean(axis=1, skipna=True)
    max_distance = condition_distance_df.max(axis=1, skipna=True)
    median_distance = condition_distance_df.median(axis=1, skipna=True)
    n_conditions_used = condition_distance_df.notna().sum(axis=1)

    if aggregate == "mean":
        outlier_score = mean_distance
    elif aggregate == "max":
        outlier_score = max_distance
    elif aggregate == "median":
        outlier_score = median_distance
    else:
        raise ValueError("aggregate must be 'mean', 'max', or 'median'")

    outlier_df = pd.DataFrame({
        "well": condition_distance_df.index.astype(int),
        "outlier_score": outlier_score.values,
        "mean_distance": mean_distance.values,
        "max_distance": max_distance.values,
        "median_distance": median_distance.values,
        "n_conditions_used": n_conditions_used.values,
    })

    outlier_df = outlier_df.sort_values("outlier_score", ascending=False).reset_index(drop=True)

    if return_condition_distances:
        return outlier_df, condition_distance_df

    return outlier_df

def plot_global_strain_outlier_scores(
    outlier_df,
    top_n=20,
    score_column="outlier_score",
    orientation="vertical",
    figsize=(10, 4),
    color=apc.aster,
    title="Global phenotypic outlier strains",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    rotation=45,
    axis_color="black",
    label_color="black",
    save=False,
    figtitle="global_strain_outlier_scores",
):
    """
    Plot top strain outlier scores.

    Parameters
    ----------
    outlier_df : pd.DataFrame
        Output from compute_global_strain_outlier_scores(...)
    top_n : int
        Number of top strains to plot
    score_column : str
        Which score column to plot, usually "outlier_score"
    orientation : str
        "vertical" or "horizontal"
    """

    if score_column not in outlier_df.columns:
        raise ValueError(f"{score_column} not found in outlier_df")

    plot_df = outlier_df.head(top_n).copy()

    labels = [f"strain_{w}" for w in plot_df["well"].tolist()]
    vals = plot_df[score_column].to_numpy()
    pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)

    if orientation == "vertical":
        ax.bar(pos, vals, color=color)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=rotation, ha="right")
        ax.set_ylabel(score_column.replace("_", " "))
    elif orientation == "horizontal":
        ax.barh(pos, vals, color=color)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(score_column.replace("_", " "))
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax

def compute_strain_response_outlier_scores(
    training_data,
    condition_order=None,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    distance_metric="euclidean",   # "euclidean" or "cosine"
    zscore_within_condition=True,
    center_within_strain=True,
    return_response_matrix=False,
):
    """
    Compute strain-level outlier scores based on environmental response profiles.

    For each strain:
        1. compute mean phenotype in each condition
        2. optionally z-score strain means within each condition
        3. remove the strain's baseline phenotype across conditions
        4. concatenate condition-specific centered vectors into one response vector
        5. measure distance from the average response vector across strains

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Dict mapping condition -> array of shape (n_rows, n_features).
        Usually training_data_finite.
    condition_order : list[str] or None
        Order of conditions to use. If None, uses training_data.keys().
    n_wells : int
        Number of wells.
    n_fovs : int or None
        Number of FoVs per well. If None, inferred from n_rows // n_wells.
    dead_ids : list[int] or None
        Expanded dead row indices. Used to infer dead wells to exclude.
    distance_metric : str
        "euclidean" or "cosine"
    zscore_within_condition : bool
        If True, z-score strain means within each condition before centering.
    center_within_strain : bool
        If True, subtract each strain's mean phenotype across conditions before
        concatenating, so the score reflects environmental response rather than
        absolute phenotype.
    return_response_matrix : bool
        If True, also return the strain x concatenated-response matrix.

    Returns
    -------
    outlier_df : pd.DataFrame
        Columns:
            well
            response_outlier_score
    response_matrix_df : pd.DataFrame (optional)
        Rows = wells, columns = concatenated condition-feature dimensions
    """

    if condition_order is None:
        condition_order = list(training_data.keys())

    if dead_ids is None:
        dead_ids = []

    dead_wells = sorted(set([d % n_wells for d in dead_ids]))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]

    if len(valid_wells) == 0:
        raise ValueError("No valid wells found after excluding dead wells.")

    def _distance_to_centroid(X, metric="euclidean"):
        centroid = np.nanmean(X, axis=0)

        if metric == "euclidean":
            d = np.sqrt(np.nansum((X - centroid) ** 2, axis=1))

        elif metric == "cosine":
            d = np.full(X.shape[0], np.nan, dtype=float)

            if not np.all(np.isfinite(centroid)):
                return d

            centroid_norm = np.linalg.norm(centroid)

            for i in range(X.shape[0]):
                xi = X[i]

                if not np.all(np.isfinite(xi)):
                    continue

                xi_norm = np.linalg.norm(xi)

                if np.isclose(xi_norm, 0) and np.isclose(centroid_norm, 0):
                    d[i] = 0.0
                    continue

                if np.isclose(xi_norm, 0) or np.isclose(centroid_norm, 0):
                    d[i] = 1.0
                    continue

                sim = np.dot(xi, centroid) / (xi_norm * centroid_norm)
                sim = np.clip(sim, -1.0, 1.0)
                d[i] = 1.0 - sim

        else:
            raise ValueError("distance_metric must be 'euclidean' or 'cosine'")

        return d

    # ----------------------------
    # build per-condition strain means
    # ----------------------------
    per_condition_strain_means = {}

    for condition in condition_order:
        dat = np.asarray(training_data[condition], dtype=float)
        n_rows, n_features = dat.shape

        if n_fovs is None:
            this_n_fovs = n_rows // n_wells
        else:
            this_n_fovs = n_fovs

        strain_means = []
        strain_ids_used = []

        for well in valid_wells:
            row_idx = [well + (fov * n_wells) for fov in range(this_n_fovs)]
            block = dat[row_idx, :]   # (n_fovs, n_features)

            if np.all(~np.isfinite(block)):
                continue

            mean_vec = np.nanmean(block, axis=0)

            if not np.any(np.isfinite(mean_vec)):
                continue

            strain_means.append(mean_vec)
            strain_ids_used.append(well)

        if len(strain_means) == 0:
            raise ValueError(f"No valid strain means found for condition {condition}")

        strain_means = np.vstack(strain_means)

        if zscore_within_condition:
            strain_means = zscore(strain_means, axis=0, nan_policy="omit")
            strain_means = np.where(np.isfinite(strain_means), strain_means, 0.0)

        per_condition_strain_means[condition] = pd.DataFrame(
            strain_means,
            index=strain_ids_used
        )

    # wells present in all conditions
    common_wells = set(valid_wells)
    for condition in condition_order:
        common_wells = common_wells.intersection(set(per_condition_strain_means[condition].index))

    common_wells = sorted(common_wells)

    if len(common_wells) == 0:
        raise ValueError("No wells found in common across all conditions.")

    # ----------------------------
    # build centered response vectors
    # ----------------------------
    response_matrix = []

    for well in common_wells:
        condition_vectors = []
        for condition in condition_order:
            condition_vectors.append(
                per_condition_strain_means[condition].loc[well].to_numpy(dtype=float)
            )

        condition_vectors = np.vstack(condition_vectors)  # (n_conditions, n_features)

        if center_within_strain:
            strain_baseline = np.nanmean(condition_vectors, axis=0, keepdims=True)
            condition_vectors = condition_vectors - strain_baseline

        response_vec = condition_vectors.reshape(-1)
        response_matrix.append(response_vec)

    response_matrix = np.vstack(response_matrix)

    # distance from average response profile
    response_outlier_scores = _distance_to_centroid(response_matrix, metric=distance_metric)

    outlier_df = pd.DataFrame({
        "well": common_wells,
        "response_outlier_score": response_outlier_scores,
    }).sort_values("response_outlier_score", ascending=False).reset_index(drop=True)

    if return_response_matrix:
        response_matrix_df = pd.DataFrame(response_matrix, index=common_wells)
        response_matrix_df.index.name = "well"
        return outlier_df, response_matrix_df

    return outlier_df

def plot_strain_response_outlier_scores(
    outlier_df,
    top_n=20,
    score_column="response_outlier_score",
    orientation="vertical",
    figsize=(10, 4),
    color=apc.aster,
    title="Strain environmental-response outlier scores",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    rotation=45,
    axis_color="black",
    label_color="black",
    save=False,
    figtitle="strain_response_outlier_scores",
):
    """
    Plot top response outlier strains.
    """

    if score_column not in outlier_df.columns:
        raise ValueError(f"{score_column} not found in outlier_df")

    plot_df = outlier_df.head(top_n).copy()

    labels = [f"strain_{w}" for w in plot_df["well"].tolist()]
    vals = plot_df[score_column].to_numpy()
    pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)

    if orientation == "vertical":
        ax.bar(pos, vals, color=color)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=rotation, ha="right")
        ax.set_ylabel(score_column.replace("_", " "))
    elif orientation == "horizontal":
        ax.barh(pos, vals, color=color)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(score_column.replace("_", " "))
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax

def plot_global_vs_response_outlier_scores(
    global_outlier_df,
    response_outlier_df,
    global_score_col="outlier_score",
    response_score_col="response_outlier_score",
    figsize=(6, 6),
    color=apc.aster,
    alpha=0.8,
    s=300,
    title="Global phenotype vs environmental-response outlier scores",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=20,
    title_size=None,
    axis_color="black",
    label_color="black",
    annotate_top_n=0,
    annotation_fontsize=16,
    annotate_by="sum",   # "sum", "global", or "response"
    reference_lines=True,
    reference_line_color="gray",
    reference_line_style="--",
    save=False,
    figtitle="global_vs_response_outlier_scores",
):
    """
    Joint scatter plot of:
        x = global phenotypic outlier score
        y = environmental-response outlier score

    Parameters
    ----------
    global_outlier_df : pd.DataFrame
        Output from compute_global_strain_outlier_scores(...)
    response_outlier_df : pd.DataFrame
        Output from compute_strain_response_outlier_scores(...)
    global_score_col : str
        Column in global_outlier_df to use for x
    response_score_col : str
        Column in response_outlier_df to use for y
    annotate_top_n : int
        Number of wells to label
    annotate_by : str
        "sum"      -> rank by x + y
        "global"   -> rank by x
        "response" -> rank by y

    Returns
    -------
    fig, ax, merged_df
    """

    if global_score_col not in global_outlier_df.columns:
        raise ValueError(f"{global_score_col} not found in global_outlier_df")
    if response_score_col not in response_outlier_df.columns:
        raise ValueError(f"{response_score_col} not found in response_outlier_df")

    merged_df = global_outlier_df[["well", global_score_col]].merge(
        response_outlier_df[["well", response_score_col]],
        on="well",
        how="inner"
    )

    x = merged_df[global_score_col].to_numpy(dtype=float)
    y = merged_df[response_score_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color=color, alpha=alpha, s=s)

    ax.set_xlabel(global_score_col.replace("_", " "))
    ax.set_ylabel(response_score_col.replace("_", " "))

    if reference_lines:
        ax.axvline(np.nanmedian(x), color=reference_line_color, linestyle=reference_line_style, linewidth=1)
        ax.axhline(np.nanmedian(y), color=reference_line_color, linestyle=reference_line_style, linewidth=1)

    if annotate_top_n > 0:
        ann_df = merged_df.copy()

        if annotate_by == "sum":
            ann_df["_rank_score"] = ann_df[global_score_col] + ann_df[response_score_col]
        elif annotate_by == "global":
            ann_df["_rank_score"] = ann_df[global_score_col]
        elif annotate_by == "response":
            ann_df["_rank_score"] = ann_df[response_score_col]
        else:
            raise ValueError("annotate_by must be 'sum', 'global', or 'response'")

        ann_df = ann_df.sort_values("_rank_score", ascending=False).head(annotate_top_n)

        for _, row in ann_df.iterrows():
            ax.text(
                row[global_score_col],
                row[response_score_col],
                f"mt_{int(row['well'])}",
                fontsize=annotation_fontsize,
                fontfamily=font_family,
                color=label_color
            )

    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, merged_df

def compute_fov_split_half_reproducibility(
    training_data,
    condition_order=None,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    n_splits=100,
    similarity_metric="pearson",   # "pearson", "spearman", "cosine", "euclidean"
    random_state=42,
    return_null=False,
):
    """
    Compute split-half reproducibility across FoVs within each strain and condition.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        condition -> array of shape (n_rows, n_features)
        Usually training_data_finite
    condition_order : list[str] or None
        Order of conditions to use
    n_wells : int
        Number of wells
    n_fovs : int or None
        Number of FoVs per well. If None, inferred from n_rows // n_wells
    dead_ids : list[int] or None
        Expanded dead row indices. Used to infer dead wells to exclude
    n_splits : int
        Number of random split-halves to perform per strain
    similarity_metric : str
        "pearson", "spearman", "cosine", or "euclidean"
    random_state : int
        RNG seed
    return_null : bool
        If True, also compute a null distribution using mismatched split-halves from different strains

    Returns
    -------
    reproducibility_df : pd.DataFrame
        One row per strain-condition with columns:
            condition
            well
            mean_similarity
            median_similarity
            std_similarity
            n_valid_splits
    null_df : pd.DataFrame (optional)
        Null distribution summary if return_null=True
    """

    if condition_order is None:
        condition_order = list(training_data.keys())

    if dead_ids is None:
        dead_ids = []

    rng = np.random.default_rng(random_state)
    dead_wells = sorted(set([d % n_wells for d in dead_ids]))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]

    def _similarity(u, v, metric="pearson"):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        mask = np.isfinite(u) & np.isfinite(v)
        u = u[mask]
        v = v[mask]

        if len(u) < 2:
            return np.nan

        if metric == "pearson":
            if np.std(u) == 0 or np.std(v) == 0:
                return np.nan
            return np.corrcoef(u, v)[0, 1]

        elif metric == "spearman":
            return pd.Series(u).corr(pd.Series(v), method="spearman")

        elif metric == "cosine":
            nu = np.linalg.norm(u)
            nv = np.linalg.norm(v)
            if np.isclose(nu, 0) or np.isclose(nv, 0):
                return np.nan
            return np.dot(u, v) / (nu * nv)

        elif metric == "euclidean":
            # convert distance to similarity by negating it
            return -np.sqrt(np.sum((u - v) ** 2))

        else:
            raise ValueError("similarity_metric must be 'pearson', 'spearman', 'cosine', or 'euclidean'")

    rows = []
    null_rows = []

    for condition in condition_order:
        dat = np.asarray(training_data[condition], dtype=float)
        n_rows, n_features = dat.shape

        if n_fovs is None:
            this_n_fovs = n_rows // n_wells
        else:
            this_n_fovs = n_fovs

        if this_n_fovs < 2:
            raise ValueError("Need at least 2 FoVs per strain for split-half reproducibility")

        # precompute per-well FoV blocks
        well_blocks = {}
        for well in valid_wells:
            row_idx = [well + (fov * n_wells) for fov in range(this_n_fovs)]
            block = dat[row_idx, :]   # (n_fovs, n_features)
            if np.all(~np.isfinite(block)):
                continue
            well_blocks[well] = block

        for well in valid_wells:
            if well not in well_blocks:
                continue

            block = well_blocks[well]
            sims = []

            for _ in range(n_splits):
                perm = rng.permutation(this_n_fovs)
                half = this_n_fovs // 2

                idx1 = perm[:half]
                idx2 = perm[half:]

                if len(idx1) == 0 or len(idx2) == 0:
                    continue

                vec1 = np.nanmean(block[idx1, :], axis=0)
                vec2 = np.nanmean(block[idx2, :], axis=0)

                sim = _similarity(vec1, vec2, metric=similarity_metric)
                if np.isfinite(sim):
                    sims.append(sim)

                # null: compare one half to a half from a random different strain
                if return_null:
                    other_candidates = [w for w in valid_wells if w != well and w in well_blocks]
                    if len(other_candidates) > 0:
                        other_well = rng.choice(other_candidates)
                        other_block = well_blocks[other_well]

                        perm_other = rng.permutation(this_n_fovs)
                        idx_other = perm_other[:half]

                        if len(idx_other) > 0:
                            vec_other = np.nanmean(other_block[idx_other, :], axis=0)
                            sim_null = _similarity(vec1, vec_other, metric=similarity_metric)
                            if np.isfinite(sim_null):
                                null_rows.append({
                                    "condition": condition,
                                    "well": well,
                                    "other_well": int(other_well),
                                    "null_similarity": sim_null,
                                })

            rows.append({
                "condition": condition,
                "well": well,
                "mean_similarity": np.nanmean(sims) if len(sims) > 0 else np.nan,
                "median_similarity": np.nanmedian(sims) if len(sims) > 0 else np.nan,
                "std_similarity": np.nanstd(sims) if len(sims) > 0 else np.nan,
                "n_valid_splits": len(sims),
            })

    reproducibility_df = pd.DataFrame(rows)

    if return_null:
        null_df = pd.DataFrame(null_rows)
        return reproducibility_df, null_df

    return reproducibility_df

def plot_fov_split_half_reproducibility(
    reproducibility_df,
    condition_order=None,
    col_titles=None,
    ytick_colors=None,
    metric_col="mean_similarity",
    figsize=(8, 4),
    color=apc.aster,
    dot_alpha=0.7,
    dot_size=25,
    box_alpha=0.35,
    title="FoV split-half reproducibility across conditions",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    axis_color="black",
    label_color="black",
    xtick_rotation=45,
    ylabel=None,
    save=False,
    figtitle="fov_split_half_reproducibility",
):
    """
    Plot distribution of split-half reproducibility values across conditions.
    """

    if condition_order is None:
        condition_order = list(reproducibility_df["condition"].unique())

    if col_titles is None:
        col_titles = condition_order

    vals_by_condition = []
    for cond in condition_order:
        vals = reproducibility_df.loc[
            reproducibility_df["condition"] == cond, metric_col
        ].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        vals_by_condition.append(vals)

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        vals_by_condition,
        patch_artist=True,
        showfliers=False
    )

    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(box_alpha)
        box.set_edgecolor(color)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(vals_by_condition, start=1):
        x = rng.normal(loc=i, scale=0.05, size=len(vals))
        ax.scatter(x, vals, color=color, alpha=dot_alpha, s=dot_size)

    ax.set_xticks(np.arange(1, len(condition_order) + 1))
    ax.set_xticklabels(col_titles, rotation=xtick_rotation, ha="right")

    if ylabel is None:
        ylabel = metric_col.replace("_", " ")
    ax.set_ylabel(ylabel)

    if ytick_colors is not None:
        if len(ytick_colors) != len(ax.get_xticklabels()):
            raise ValueError("ytick_colors length mismatch")
        for tick, c in zip(ax.get_xticklabels(), ytick_colors):
            tick.set_color(c)

    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        if ytick_colors is None:
            tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax

def compute_within_between_fov_similarity(
    training_data,
    condition_order=None,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    similarity_metric="pearson",   # "pearson", "spearman", "cosine", "euclidean"
    n_between_samples_per_well=20,
    random_state=42,
):
    """
    Compare within-strain FoV similarity to between-strain FoV similarity.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        condition -> array of shape (n_rows, n_features)
    condition_order : list[str] or None
        Order of conditions to use
    n_wells : int
        Number of wells
    n_fovs : int or None
        Number of FoVs per well
    dead_ids : list[int] or None
        Expanded dead row indices. Used to infer dead wells to exclude
    similarity_metric : str
        "pearson", "spearman", "cosine", or "euclidean"
    n_between_samples_per_well : int
        Number of between-strain comparisons to sample per well
    random_state : int
        RNG seed

    Returns
    -------
    similarity_df : pd.DataFrame
        Columns:
            condition
            comparison_type   ("within" or "between")
            similarity
    """

    if condition_order is None:
        condition_order = list(training_data.keys())

    if dead_ids is None:
        dead_ids = []

    rng = np.random.default_rng(random_state)
    dead_wells = sorted(set([d % n_wells for d in dead_ids]))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]

    def _similarity(u, v, metric="pearson"):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        mask = np.isfinite(u) & np.isfinite(v)
        u = u[mask]
        v = v[mask]

        if len(u) < 2:
            return np.nan

        if metric == "pearson":
            if np.std(u) == 0 or np.std(v) == 0:
                return np.nan
            return np.corrcoef(u, v)[0, 1]

        elif metric == "spearman":
            return pd.Series(u).corr(pd.Series(v), method="spearman")

        elif metric == "cosine":
            nu = np.linalg.norm(u)
            nv = np.linalg.norm(v)
            if np.isclose(nu, 0) or np.isclose(nv, 0):
                return np.nan
            return np.dot(u, v) / (nu * nv)

        elif metric == "euclidean":
            return -np.sqrt(np.sum((u - v) ** 2))

        else:
            raise ValueError("similarity_metric must be 'pearson', 'spearman', 'cosine', or 'euclidean'")

    rows = []

    for condition in condition_order:
        dat = np.asarray(training_data[condition], dtype=float)
        n_rows, n_features = dat.shape

        if n_fovs is None:
            this_n_fovs = n_rows // n_wells
        else:
            this_n_fovs = n_fovs

        well_blocks = {}
        for well in valid_wells:
            row_idx = [well + (fov * n_wells) for fov in range(this_n_fovs)]
            block = dat[row_idx, :]
            if np.all(~np.isfinite(block)):
                continue
            well_blocks[well] = block

        # within-strain
        for well in valid_wells:
            if well not in well_blocks:
                continue

            block = well_blocks[well]
            for i in range(this_n_fovs):
                for j in range(i + 1, this_n_fovs):
                    sim = _similarity(block[i, :], block[j, :], metric=similarity_metric)
                    if np.isfinite(sim):
                        rows.append({
                            "condition": condition,
                            "comparison_type": "within",
                            "similarity": sim,
                        })

        # between-strain
        for well in valid_wells:
            if well not in well_blocks:
                continue

            other_candidates = [w for w in valid_wells if w != well and w in well_blocks]
            if len(other_candidates) == 0:
                continue

            block = well_blocks[well]

            for _ in range(n_between_samples_per_well):
                other_well = rng.choice(other_candidates)
                other_block = well_blocks[other_well]

                i = rng.integers(0, this_n_fovs)
                j = rng.integers(0, this_n_fovs)

                sim = _similarity(block[i, :], other_block[j, :], metric=similarity_metric)
                if np.isfinite(sim):
                    rows.append({
                        "condition": condition,
                        "comparison_type": "between",
                        "similarity": sim,
                    })

    similarity_df = pd.DataFrame(rows)
    return similarity_df

def plot_within_between_fov_similarity(
    similarity_df,
    condition_order=None,
    col_titles=None,
    xtick_colors=None,
    figsize=(10, 4),
    within_color=apc.aster,
    between_color=apc.lapis,
    title="FoV similarity within vs between strains",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    axis_color="black",
    label_color="black",
    xtick_rotation=45,
    ylabel="similarity",
    save=False,
    figtitle="within_between_fov_similarity",
):
    """
    Boxplot comparing within-strain and between-strain FoV similarity across conditions.
    """

    if condition_order is None:
        condition_order = list(similarity_df["condition"].unique())

    if col_titles is None:
        col_titles = condition_order

    positions = []
    data = []
    colors = []
    labels = []

    pos = 1
    for cond, label in zip(condition_order, col_titles):
        vals_within = similarity_df.loc[
            (similarity_df["condition"] == cond) &
            (similarity_df["comparison_type"] == "within"),
            "similarity"
        ].to_numpy(dtype=float)
        vals_between = similarity_df.loc[
            (similarity_df["condition"] == cond) &
            (similarity_df["comparison_type"] == "between"),
            "similarity"
        ].to_numpy(dtype=float)

        data.extend([vals_within, vals_between])
        positions.extend([pos, pos + 0.35])
        colors.extend([within_color, between_color])
        labels.append(label)
        pos += 1.2

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False
    )

    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.35)
        box.set_edgecolor(c)

    center_positions = []
    pos = 1
    for _ in condition_order:
        center_positions.append(pos + 0.175)
        pos += 1.2

    ax.set_xticks(center_positions)
    ax.set_xticklabels(col_titles, rotation=xtick_rotation, ha="right")
    ax.set_ylabel(ylabel)

    if xtick_colors is not None:
        if len(xtick_colors) != len(ax.get_xticklabels()):
            raise ValueError("xtick_colors length mismatch")
        for tick, c in zip(ax.get_xticklabels(), xtick_colors):
            tick.set_color(c)

    handles = [
        plt.Line2D([0], [0], color=within_color, lw=6, alpha=0.6),
        plt.Line2D([0], [0], color=between_color, lw=6, alpha=0.6),
    ]
    ax.legend(handles, ["within strain", "between strains"], frameon=False)

    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        if xtick_colors is None:
            tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax

def plot_fov_split_half_vs_null(
    reproducibility_df,
    null_df,
    condition_order=None,
    col_titles=None,
    xtick_colors=None,
    metric_col="mean_similarity",
    figsize=(10, 4),
    same_color=apc.aster,
    null_color=apc.lapis,
    title="FoV split-half reproducibility vs mismatched null",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    axis_color="black",
    label_color="black",
    xtick_rotation=45,
    ylabel="similarity",
    save=False,
    figtitle="fov_split_half_vs_null",
):
    """
    Compare same-strain split-half reproducibility to mismatched-strain null similarity
    across conditions.

    Parameters
    ----------
    reproducibility_df : pd.DataFrame
        Output from compute_fov_split_half_reproducibility(...)
        Must contain:
            condition
            well
            mean_similarity (or metric_col)
    null_df : pd.DataFrame
        Null output from compute_fov_split_half_reproducibility(..., return_null=True)
        Must contain:
            condition
            null_similarity
    condition_order : list[str] or None
        Order of conditions to plot
    col_titles : list[str] or None
        Display labels for conditions
    xtick_colors : list or None
        Colors for condition labels
    metric_col : str
        Column in reproducibility_df to use for same-strain summary
    """

    if metric_col not in reproducibility_df.columns:
        raise ValueError(f"{metric_col} not found in reproducibility_df")

    required_null_cols = ["condition", "null_similarity"]
    for col in required_null_cols:
        if col not in null_df.columns:
            raise ValueError(f"{col} not found in null_df")

    if condition_order is None:
        condition_order = list(reproducibility_df["condition"].unique())

    if col_titles is None:
        col_titles = condition_order

    positions = []
    data = []
    colors = []

    pos = 1
    for cond in condition_order:
        vals_same = reproducibility_df.loc[
            reproducibility_df["condition"] == cond,
            metric_col
        ].to_numpy(dtype=float)
        vals_same = vals_same[np.isfinite(vals_same)]

        vals_null = null_df.loc[
            null_df["condition"] == cond,
            "null_similarity"
        ].to_numpy(dtype=float)
        vals_null = vals_null[np.isfinite(vals_null)]

        data.extend([vals_same, vals_null])
        positions.extend([pos, pos + 0.35])
        colors.extend([same_color, null_color])

        pos += 1.2

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False
    )

    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.35)
        box.set_edgecolor(c)

    # center tick labels per condition pair
    center_positions = []
    pos = 1
    for _ in condition_order:
        center_positions.append(pos + 0.175)
        pos += 1.2

    ax.set_xticks(center_positions)
    ax.set_xticklabels(col_titles, rotation=xtick_rotation, ha="right")
    ax.set_ylabel(ylabel)

    if xtick_colors is not None:
        if len(xtick_colors) != len(ax.get_xticklabels()):
            raise ValueError("xtick_colors length mismatch")
        for tick, c in zip(ax.get_xticklabels(), xtick_colors):
            tick.set_color(c)

    handles = [
        plt.Line2D([0], [0], color=same_color, lw=6, alpha=0.6),
        plt.Line2D([0], [0], color=null_color, lw=6, alpha=0.6),
    ]
    ax.legend(handles, ["same strain split-halves", "mismatched-strain null"], frameon=False)

    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        if xtick_colors is None:
            tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax

def plot_condition_holdout_model_comparison(
    comparison_results,
    model_order=("ridge", "within_strain_mean", "global_mean"),
    metric="overall_r2",   # "overall_r2" or "rmse"
    col_titles=None,
    xtick_colors=None,
    plot_mode="line",      # "line" or "bar"
    figsize=(8, 5),
    model_colors=None,
    linewidth=2.5,
    marker="o",
    markersize=7,
    bar_width=0.24,
    alpha=0.9,
    show_values=False,
    value_fontsize=None,
    title=None,
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    xtick_rotation=45,
    axis_color="black",
    label_color="black",
    zero_line=True,
    zero_line_color="gray",
    zero_line_style="--",
    legend=True,
    legend_loc="best",
    save=False,
    figtitle="condition_holdout_model_comparison",
):
    """
    Plot held-out condition performance for multiple models.

    Parameters
    ----------
    comparison_results : dict
        Output from compare_models_condition_holdout_cv(...)
    model_order : tuple/list[str]
        Models to plot, in order.
    metric : str
        "overall_r2" or "rmse"
    col_titles : list[str] or None
        Display labels for conditions. If None, uses comparison_results conditions.
    xtick_colors : list or None
        Colors for x-axis condition labels.
    plot_mode : str
        "line" or "bar"
    figsize : tuple
        Figure size.
    model_colors : dict or None
        Mapping model_name -> color.
    linewidth, marker, markersize, bar_width, alpha
        Plot styling parameters.
    show_values : bool
        Whether to label points/bars with numeric values.
    value_fontsize : int or None
        Font size for value labels.
    title : str or None
        Plot title.
    show_title : bool
        Whether to show the title.
    title_fontsize : int
        Title font size.
    font_family : str or None
        Optional font family.
    font_size : int
        Base font size.
    title_size : int or None
        Title font size override.
    xtick_rotation : int or float
        Rotation for x tick labels.
    axis_color : str
        Axis / spine / tick mark color.
    label_color : str
        Default tick label color.
    zero_line : bool
        Whether to draw a horizontal zero line.
    zero_line_color : str
        Color of zero line.
    zero_line_style : str
        Linestyle of zero line.
    legend : bool
        Whether to show legend.
    legend_loc : str
        Legend location.
    save : bool
        Whether to save png/pdf/svg.
    figtitle : str
        Root filename for saved figure.

    Returns
    -------
    fig, ax, plot_df
    """

    # ----------------------------
    # validate model names
    # ----------------------------
    for model_name in model_order:
        if model_name not in comparison_results:
            raise ValueError(f"{model_name} not found in comparison_results")

    conditions = comparison_results[model_order[0]]["conditions"]

    if col_titles is None:
        col_titles = conditions

    if len(col_titles) != len(conditions):
        raise ValueError("col_titles length mismatch")

    if xtick_colors is not None and len(xtick_colors) != len(conditions):
        raise ValueError("xtick_colors length mismatch")

    if metric not in ("overall_r2", "rmse"):
        raise ValueError("metric must be 'overall_r2' or 'rmse'")

    if model_colors is None:
        model_colors = {
            "ridge": apc.aster,
            "within_strain_mean": apc.vital,
            "global_mean": apc.lapis,
        }

    if value_fontsize is None:
        value_fontsize = font_size

    # ----------------------------
    # build plotting dataframe
    # ----------------------------
    plot_df = pd.DataFrame(
        {m: np.asarray(comparison_results[m][metric], dtype=float) for m in model_order},
        index=conditions
    )

    # ----------------------------
    # create plot
    # ----------------------------
    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=figsize)

    if plot_mode == "line":
        for model_name in model_order:
            vals = plot_df[model_name].to_numpy()
            ax.plot(
                x,
                vals,
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                alpha=alpha,
                color=model_colors.get(model_name, None),
                label=model_name
            )

            if show_values:
                for xi, yi in zip(x, vals):
                    ax.text(
                        xi,
                        yi,
                        f"{yi:.2f}",
                        fontsize=value_fontsize,
                        fontfamily=font_family,
                        color=label_color,
                        ha="center",
                        va="bottom"
                    )

    elif plot_mode == "bar":
        n_models = len(model_order)
        offsets = np.linspace(
            -bar_width * (n_models - 1) / 2,
             bar_width * (n_models - 1) / 2,
             n_models
        )

        for offset, model_name in zip(offsets, model_order):
            vals = plot_df[model_name].to_numpy()
            bars = ax.bar(
                x + offset,
                vals,
                width=bar_width,
                alpha=alpha,
                color=model_colors.get(model_name, None),
                label=model_name
            )

            if show_values:
                for rect, yi in zip(bars, vals):
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        yi,
                        f"{yi:.2f}",
                        fontsize=value_fontsize,
                        fontfamily=font_family,
                        color=label_color,
                        ha="center",
                        va="bottom" if yi >= 0 else "top"
                    )

    else:
        raise ValueError("plot_mode must be 'line' or 'bar'")

    # ----------------------------
    # axes / labels
    # ----------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(col_titles, rotation=xtick_rotation, ha="right")

    if metric == "overall_r2":
        ylabel = "Held-out condition R²"
    else:
        ylabel = "Held-out condition RMSE"

    ax.set_ylabel(ylabel)

    if zero_line:
        ax.axhline(0, color=zero_line_color, linestyle=zero_line_style, linewidth=1)

    if show_title:
        if title is None:
            if metric == "overall_r2":
                title = "Held-out condition prediction performance"
            else:
                title = "Held-out condition prediction error"
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    if legend:
        leg = ax.legend(loc=legend_loc, frameon=False)
        if font_family is not None:
            for text in leg.get_texts():
                text.set_fontfamily(font_family)
                text.set_fontsize(font_size)
        else:
            for text in leg.get_texts():
                text.set_fontsize(font_size)

    # ----------------------------
    # styling
    # ----------------------------
    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if xtick_colors is not None:
        for tick, c in zip(ax.get_xticklabels(), xtick_colors):
            tick.set_color(c)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, plot_df

def compute_strain_distance_matrix(
    training_data,
    condition_order=None,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    mode="all_conditions",   # "all_conditions" or "single_condition"
    condition=None,
    distance_metric="euclidean",   # "euclidean", "cosine", or "correlation"
    zscore_within_condition=True,
    return_feature_matrix=False,
):
    """
    Compute a strain x strain distance matrix.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Dict mapping condition -> array of shape (n_rows, n_features).
        Usually training_data_finite.
    condition_order : list[str] or None
        Order of conditions to use. If None, uses training_data.keys().
    n_wells : int
        Number of wells.
    n_fovs : int or None
        Number of FoVs per well. If None, inferred from row count.
    dead_ids : list[int] or None
        Expanded dead row indices. Used to infer dead wells to exclude.
    mode : str
        "all_conditions" -> concatenate strain means across all conditions
        "single_condition" -> use only one condition
    condition : str or None
        Required if mode="single_condition".
    distance_metric : str
        "euclidean", "cosine", or "correlation"
    zscore_within_condition : bool
        If True, z-score strain means within each condition before concatenating.
    return_feature_matrix : bool
        If True, also return the strain x feature matrix used for distance calculation.

    Returns
    -------
    distance_df : pd.DataFrame
        Strain x strain distance matrix.
    feature_matrix_df : pd.DataFrame (optional)
        Strain x feature matrix used to compute distances.
    """

    if condition_order is None:
        condition_order = list(training_data.keys())

    if dead_ids is None:
        dead_ids = []

    dead_wells = sorted(set([d % n_wells for d in dead_ids]))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]

    if len(valid_wells) == 0:
        raise ValueError("No valid wells found after excluding dead wells.")

    if mode not in ("all_conditions", "single_condition"):
        raise ValueError("mode must be 'all_conditions' or 'single_condition'")

    if mode == "single_condition":
        if condition is None:
            raise ValueError("condition must be provided when mode='single_condition'")
        if condition not in training_data:
            raise ValueError(f"{condition} not found in training_data")
        conditions_to_use = [condition]
    else:
        conditions_to_use = condition_order

    per_condition_strain_means = {}

    for cond in conditions_to_use:
        dat = np.asarray(training_data[cond], dtype=float)
        n_rows, n_features = dat.shape

        if n_fovs is None:
            this_n_fovs = n_rows // n_wells
        else:
            this_n_fovs = n_fovs

        strain_means = []
        strain_ids_used = []

        for well in valid_wells:
            row_idx = [well + (fov * n_wells) for fov in range(this_n_fovs)]
            block = dat[row_idx, :]

            if np.all(~np.isfinite(block)):
                continue

            mean_vec = np.nanmean(block, axis=0)

            if not np.any(np.isfinite(mean_vec)):
                continue

            strain_means.append(mean_vec)
            strain_ids_used.append(well)

        if len(strain_means) == 0:
            raise ValueError(f"No valid strain means found for condition {cond}")

        strain_means = np.vstack(strain_means)

        if zscore_within_condition:
            strain_means = zscore(strain_means, axis=0, nan_policy="omit")
            strain_means = np.where(np.isfinite(strain_means), strain_means, 0.0)

        per_condition_strain_means[cond] = pd.DataFrame(
            strain_means,
            index=strain_ids_used
        )

    # keep only wells present in all conditions used
    common_wells = set(valid_wells)
    for cond in conditions_to_use:
        common_wells = common_wells.intersection(set(per_condition_strain_means[cond].index))
    common_wells = sorted(common_wells)

    if len(common_wells) == 0:
        raise ValueError("No common wells found across selected conditions.")

    # build final strain x feature matrix
    feature_matrix = []
    for well in common_wells:
        pieces = []
        for cond in conditions_to_use:
            pieces.append(per_condition_strain_means[cond].loc[well].to_numpy(dtype=float))
        feature_matrix.append(np.concatenate(pieces))

    feature_matrix = np.vstack(feature_matrix)

    # compute pairwise distances
    if distance_metric == "euclidean":
        dist_mat = squareform(pdist(feature_matrix, metric="euclidean"))
    elif distance_metric == "cosine":
        dist_mat = squareform(pdist(feature_matrix, metric="cosine"))
    elif distance_metric == "correlation":
        dist_mat = squareform(pdist(feature_matrix, metric="correlation"))
    else:
        raise ValueError("distance_metric must be 'euclidean', 'cosine', or 'correlation'")

    strain_labels = [f"well_{w}" for w in common_wells]

    distance_df = pd.DataFrame(
        dist_mat,
        index=strain_labels,
        columns=strain_labels
    )

    if return_feature_matrix:
        feature_matrix_df = pd.DataFrame(feature_matrix, index=strain_labels)
        feature_matrix_df.index.name = "strain"
        return distance_df, feature_matrix_df

    return distance_df

def plot_strain_distance_heatmap(
    distance_df,
    figsize=(10, 8),
    cmap="apc:orange_sage_r",
    cluster=False,
    title="Strain-to-strain phenotypic distance",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=8,
    title_size=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    xtick_rotation=90,
    ytick_rotation=0,
    axis_color="black",
    label_color="black",
    cbar=True,
    cbar_label="distance",
    cbar_shrink=1.0,
    cbar_aspect=30,
    vmin=None,
    vmax=None,
    save=False,
    figtitle="strain_distance_heatmap",
):
    """
    Plot a strain x strain distance heatmap.

    Parameters
    ----------
    distance_df : pd.DataFrame
        Output from compute_strain_distance_matrix(...)
    cluster : bool
        If True, reorder rows/columns by hierarchical clustering.
    """

    if xtick_fontsize is None:
        xtick_fontsize = font_size
    if ytick_fontsize is None:
        ytick_fontsize = font_size

    plot_df = distance_df.copy()

    if cluster:
        # cluster using linkage on the distance matrix rows
        linkage_mat = linkage(plot_df.values, method="average")
        order = leaves_list(linkage_mat)
        plot_df = plot_df.iloc[order, order]

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

    if cbar and cbar_label is not None:
        hm.collections[0].colorbar.set_label(
            cbar_label,
            fontsize=font_size,
            fontfamily=font_family
        )

    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

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
        rotation=ytick_rotation,
        fontsize=ytick_fontsize,
        fontfamily=font_family
    )

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", color=axis_color)
    ax.tick_params(axis="y", color=axis_color)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, plot_df

def plot_strain_pca_trajectories(
    training_data,
    exp_conditions,
    strains_to_plot,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    condition_labels=None,
    condition_colors=None,
    figsize=(8, 6),
    title="Strain PCA trajectories across conditions",
    show_title=True,
    title_fontsize=16,
    point_size=80,
    line_width=2,
    alpha=0.9,
    annotate_points=True,
    show_variance_explained=True,
    fit_pca_on="all_strains",   # "all_strains" or "selected_strains"
    pca_conditions=None,
    save=False,
    figtitle="strain_pca_trajectories"
):
    """
    Plot PCA trajectories for selected strains across conditions.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Dict mapping condition -> array of shape (n_rows, n_features).
    exp_conditions : list[str]
        Conditions to plot, in trajectory order.
    strains_to_plot : list[int]
        Wells/strains to show.
    n_wells : int
        Number of wells.
    n_fovs : int or None
        Number of FoVs per well. If None, inferred from row count.
    dead_ids : list[int] or None
        Expanded dead row indices. Used to infer dead wells to exclude.
    condition_labels : list[str] or None
        Labels for legend. If None, uses exp_conditions.
    condition_colors : list or None
        One color per plotted condition.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    show_title : bool
        Whether to show title.
    title_fontsize : int
        Title fontsize.
    point_size : float
        Point size.
    line_width : float
        Trajectory line width.
    alpha : float
        Point/line alpha.
    annotate_points : bool
        Whether to write strain ID on each point.
    show_variance_explained : bool
        Whether to show variance explained in axis labels.
    fit_pca_on : str
        "all_strains" -> fit PCA on all valid strains from pca_conditions
        "selected_strains" -> fit PCA only on selected strains from pca_conditions
    pca_conditions : list[str] or None
        Conditions used to fit PCA space. If None, uses all conditions in training_data.
        This can be broader than exp_conditions.
    save : bool
        Whether to save figure.
    figtitle : str
        Root filename for saving.

    Returns
    -------
    fig, ax
    """

    if dead_ids is None:
        dead_ids = []

    if condition_labels is None:
        condition_labels = exp_conditions

    if condition_colors is None:
        condition_colors = [apc.vital, apc.wish, apc.dress,
                            apc.lapis, apc.aster, apc.candy]

    if len(condition_labels) != len(exp_conditions):
        raise ValueError("condition_labels must match exp_conditions length")

    if len(condition_colors) < len(exp_conditions):
        raise ValueError("condition_colors must have at least as many entries as exp_conditions")

    if fit_pca_on not in ("all_strains", "selected_strains"):
        raise ValueError("fit_pca_on must be 'all_strains' or 'selected_strains'")

    if pca_conditions is None:
        pca_conditions = list(training_data.keys())

    for cond in exp_conditions:
        if cond not in training_data:
            raise ValueError(f"{cond} not found in training_data")

    for cond in pca_conditions:
        if cond not in training_data:
            raise ValueError(f"{cond} not found in training_data")

    # ----------------------------
    # Infer n_fovs
    # ----------------------------
    example = np.asarray(training_data[pca_conditions[0]], dtype=float)
    if n_fovs is None:
        n_fovs = example.shape[0] // n_wells

    dead_wells = sorted(set([d % n_wells for d in dead_ids]))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]

    for s in strains_to_plot:
        if s not in valid_wells:
            raise ValueError(f"strain {s} is not a valid well (may be dead or out of range)")

    # ----------------------------
    # helper: mean vector for one strain in one condition
    # ----------------------------
    def _strain_condition_mean(cond, strain):
        dat = np.asarray(training_data[cond], dtype=float)
        rows = [strain + f * n_wells for f in range(n_fovs)]
        return np.nanmean(dat[rows, :], axis=0)

    # ----------------------------
    # build matrix for PCA fitting
    # ----------------------------
    if fit_pca_on == "all_strains":
        strains_for_pca = valid_wells
    else:
        strains_for_pca = list(strains_to_plot)

    pca_vectors = []
    for strain in strains_for_pca:
        for cond in pca_conditions:
            vec = _strain_condition_mean(cond, strain)
            pca_vectors.append(vec)

    pca_vectors = np.vstack(pca_vectors)

    # ----------------------------
    # fit PCA on chosen population / chosen PCA conditions
    # ----------------------------
    pcs_all, evecs, evals, mean_val = pca_on_np_array(pca_vectors)

    var_explained = evals / np.sum(evals)
    pc1_var = var_explained[0] * 100
    pc2_var = var_explained[1] * 100

    # ----------------------------
    # build selected strain vectors ONLY for plotted conditions
    # and project onto PCA basis fitted above
    # ----------------------------
    selected_vectors = []
    for strain in strains_to_plot:
        for cond in exp_conditions:
            vec = _strain_condition_mean(cond, strain)
            selected_vectors.append(vec)

    selected_vectors = np.vstack(selected_vectors)

    selected_pcs = np.dot(selected_vectors - mean_val, evecs)

    # ----------------------------
    # plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=figsize)

    idx = 0
    for strain in strains_to_plot:
        traj = []

        for _ in range(len(exp_conditions)):
            traj.append(selected_pcs[idx])
            idx += 1

        traj = np.array(traj)

        ax.plot(
            traj[:, 0],
            traj[:, 1],
            linewidth=line_width,
            alpha=alpha,
            color="black",
            zorder=1
        )

        for i in range(len(exp_conditions)):
            ax.scatter(
                traj[i, 0],
                traj[i, 1],
                s=point_size,
                color=condition_colors[i],
                alpha=alpha,
                edgecolor="black",
                zorder=2
            )

            if annotate_points:
                ax.text(
                    traj[i, 0],
                    traj[i, 1],
                    f"{strain}",
                    fontsize=10,
                    ha="center",
                    va="center"
                )

    # ----------------------------
    # labels
    # ----------------------------
    if show_variance_explained:
        ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
        ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    if show_title:
        ax.set_title(title, fontsize=title_fontsize)

    ax.set_xticks([])
    ax.set_yticks([])

    for i, label in enumerate(condition_labels):
        ax.scatter([], [], color=condition_colors[i], label=label)

    ax.legend(title="Condition", loc="best")

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax

def find_feature_extremes(
    zroots,
    feature_name,
    channel="BF",
    exp_conditions=None,
    condition_labels=None,
    n_wells=96,
    n_fovs=9,
    dead_ids=None,
    exclude_features=None,
    top_n_strains=8,
    top_n_cells=12,
    abs_value=False,
    strain_summary_method="mean_of_fov_means",  # "mean_of_fov_means" or "mean_of_all_cells"
):
    """
    Find high/low examples for a given feature at both the strain level and cell level.

    Returns
    -------
    results : dict with keys
        "feature_name"
        "channel"
        "all_strain_values_df"
        "top_strains_df"
        "bottom_strains_df"
        "all_cells_df"
        "top_cells_df"
        "bottom_cells_df"

    Notes
    -----
    - Strain-level values are computed per condition and well.
    - Cell-level values are computed per individual cell row in get_cell_properties(...).
    - For cell-level output, the key locator fields are:
        condition_idx, condition_label, well, fov, cell_row_in_props
      so you can later recover/crop the corresponding cell manually.
    """

    if dead_ids is None:
        dead_ids = []

    if exclude_features is None:
        exclude_features = []

    if exp_conditions is None:
        exp_conditions = list(range(len(zroots)))

    if condition_labels is None:
        condition_labels = [str(x) for x in exp_conditions]

    if len(condition_labels) != len(exp_conditions):
        raise ValueError("condition_labels must match exp_conditions length")

    dead_wells = sorted(set([d % n_wells for d in dead_ids]))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]

    # ----------------------------
    # Validate feature exists
    # ----------------------------
    feature_exists = False
    checked_any = False

    for exp_idx in exp_conditions:
        z = zroots[exp_idx]
        for fov in range(n_fovs):
            for well in valid_wells[: min(len(valid_wells), 5)]:
                props = z.get_cell_properties(fov, well, channel)
                if props is None or len(props) == 0:
                    continue
                checked_any = True
                cols = props.columns.tolist()
                if len(cols) > 0 and cols[0] == "label":
                    cols[0] = "count"
                if feature_name in cols:
                    feature_exists = True
                    break
            if feature_exists:
                break
        if feature_exists:
            break

    if not checked_any:
        raise ValueError("Could not inspect any non-empty feature tables to validate feature_name.")

    if not feature_exists:
        raise ValueError(
            f"Feature '{feature_name}' was not found in get_cell_properties(...). "
            f"Make sure it matches the dataframe column name exactly."
        )

    if feature_name in exclude_features:
        raise ValueError(f"Feature '{feature_name}' is in exclude_features.")

    # ----------------------------
    # Collect strain-level values
    # ----------------------------
    strain_rows = []
    cell_rows = []

    for exp_idx, cond_label in zip(exp_conditions, condition_labels):
        z = zroots[exp_idx]

        for well in valid_wells:
            per_fov_means = []
            pooled_cell_values = []

            for fov in range(n_fovs):
                props = z.get_cell_properties(fov, well, channel)

                if props is None or len(props) == 0:
                    continue

                props = props.copy()
                cols = props.columns.tolist()
                if len(cols) > 0 and cols[0] == "label":
                    cols[0] = "count"
                    props.columns = cols

                if feature_name not in props.columns:
                    continue

                vals = pd.to_numeric(props[feature_name], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]

                if len(vals) == 0:
                    continue

                if abs_value:
                    vals = np.abs(vals)

                # cell-level records
                full_vals = pd.to_numeric(props[feature_name], errors="coerce").to_numpy(dtype=float)
                for row_idx, v in enumerate(full_vals):
                    if not np.isfinite(v):
                        continue
                    if abs_value:
                        v = abs(v)

                    cell_rows.append({
                        "feature": feature_name,
                        "channel": channel,
                        "condition_idx": exp_idx,
                        "condition_label": cond_label,
                        "well": well,
                        "fov": fov,
                        "cell_row_in_props": row_idx,
                        "feature_value": float(v),
                    })

                per_fov_means.append(np.nanmean(vals))
                pooled_cell_values.extend(vals.tolist())

            # summarize one strain in one condition
            if len(per_fov_means) == 0 and len(pooled_cell_values) == 0:
                continue

            if strain_summary_method == "mean_of_fov_means":
                strain_val = float(np.nanmean(per_fov_means))
            elif strain_summary_method == "mean_of_all_cells":
                strain_val = float(np.nanmean(pooled_cell_values))
            else:
                raise ValueError(
                    "strain_summary_method must be 'mean_of_fov_means' or 'mean_of_all_cells'"
                )

            strain_rows.append({
                "feature": feature_name,
                "channel": channel,
                "condition_idx": exp_idx,
                "condition_label": cond_label,
                "well": well,
                "feature_value": strain_val,
                "n_fovs_used": len(per_fov_means),
                "n_cells_used": len(pooled_cell_values),
            })

    if len(strain_rows) == 0:
        raise ValueError("No valid strain-level values were collected.")

    if len(cell_rows) == 0:
        raise ValueError("No valid cell-level values were collected.")

    all_strain_values_df = pd.DataFrame(strain_rows)
    all_cells_df = pd.DataFrame(cell_rows)

    # ----------------------------
    # Top / bottom strain examples
    # ----------------------------
    top_strains_df = (
        all_strain_values_df
        .sort_values("feature_value", ascending=False)
        .head(top_n_strains)
        .reset_index(drop=True)
    )

    bottom_strains_df = (
        all_strain_values_df
        .sort_values("feature_value", ascending=True)
        .head(top_n_strains)
        .reset_index(drop=True)
    )

    # ----------------------------
    # Top / bottom cell examples
    # ----------------------------
    top_cells_df = (
        all_cells_df
        .sort_values("feature_value", ascending=False)
        .head(top_n_cells)
        .reset_index(drop=True)
    )

    bottom_cells_df = (
        all_cells_df
        .sort_values("feature_value", ascending=True)
        .head(top_n_cells)
        .reset_index(drop=True)
    )

    results = {
        "feature_name": feature_name,
        "channel": channel,
        "all_strain_values_df": all_strain_values_df,
        "top_strains_df": top_strains_df,
        "bottom_strains_df": bottom_strains_df,
        "all_cells_df": all_cells_df,
        "top_cells_df": top_cells_df,
        "bottom_cells_df": bottom_cells_df,
    }

    return results

def plot_feature_extreme_strains(
    feature_extremes_result,
    top_n=6,
    figsize=(10, 5),
    high_color=apc.aster,
    low_color=apc.lapis,
    title=None,
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    rotation=45,
    axis_color="black",
    label_color="black",
    save=False,
    figtitle="feature_extreme_strains",
):
    """
    Plot top and bottom strain-level examples for a selected feature.

    Input should be the output dict from find_feature_extremes(...).
    """

    top_df = feature_extremes_result["top_strains_df"].head(top_n).copy()
    bottom_df = feature_extremes_result["bottom_strains_df"].head(top_n).copy()
    feature_name = feature_extremes_result["feature_name"]
    channel = feature_extremes_result["channel"]

    plot_df = pd.concat([
        bottom_df.assign(group="low"),
        top_df.assign(group="high")
    ], axis=0).reset_index(drop=True)

    labels = [
        f"{row['condition_label']}\nwell_{int(row['well'])}"
        for _, row in plot_df.iterrows()
    ]
    vals = plot_df["feature_value"].to_numpy()
    colors = [low_color if g == "low" else high_color for g in plot_df["group"]]
    pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(pos, vals, color=colors)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=rotation, ha="right")

    ax.set_ylabel(feature_name)

    if show_title:
        if title is None:
            title = f"High and low strain examples | {feature_name} ({channel})"
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, plot_df

def compute_feature_covariation_matrix(
    zroots,
    channel="BF",
    exp_conditions=None,
    condition_labels=None,
    n_wells=96,
    n_fovs=9,
    dead_ids=None,
    include_pixel_intensity=True,
    exclude_features=None,
    summary_level="strain",   # "strain" or "fov"
    summary_method="mean",    # "mean" or "median"
    correlation_method="pearson",   # "pearson" or "spearman"
    abs_value_features=None,
    zscore_within_condition=False,
    return_feature_table=False,
):
    """
    Compute a feature-feature covariation matrix from summarized phenotypes.

    Parameters
    ----------
    zroots : list
        List of ExperimentStore-like objects.
    channel : str
        Channel to read from, usually "BF" or "CY5".
    exp_conditions : list[int] or None
        Indices into zroots to use. If None, uses all.
    condition_labels : list[str] or None
        Human-readable labels for conditions. If None, uses string versions of exp_conditions.
    n_wells : int
        Number of wells.
    n_fovs : int
        Number of FoVs per well.
    dead_ids : list[int] or None
        Expanded dead row ids. Converted to dead wells internally.
    include_pixel_intensity : bool
        Whether to append pixel_intensity as a feature.
    exclude_features : list[str] or None
        Base feature names to exclude, e.g. ["moments_hu-4", "moments_hu-5", "moments_hu-6"]
    summary_level : str
        "strain" -> one row per condition × well
        "fov"    -> one row per condition × well × fov
    summary_method : str
        "mean" or "median" over cells
    correlation_method : str
        "pearson" or "spearman"
    abs_value_features : list[str] or None
        Base feature names for which absolute values should be taken before summarizing.
    zscore_within_condition : bool
        If True, z-score each feature within each condition after summarization.
    return_feature_table : bool
        If True, also return the summarized feature table used for the matrix.

    Returns
    -------
    corr_df : pd.DataFrame
        Feature-feature correlation matrix.
    feature_table_df : pd.DataFrame (optional)
        Table used to compute the matrix.
    """

    if dead_ids is None:
        dead_ids = []

    if exclude_features is None:
        exclude_features = []

    if abs_value_features is None:
        abs_value_features = []

    if exp_conditions is None:
        exp_conditions = list(range(len(zroots)))

    if condition_labels is None:
        condition_labels = [str(c) for c in exp_conditions]

    if len(condition_labels) != len(exp_conditions):
        raise ValueError("condition_labels must match exp_conditions length")

    if summary_level not in ("strain", "fov"):
        raise ValueError("summary_level must be 'strain' or 'fov'")

    if summary_method not in ("mean", "median"):
        raise ValueError("summary_method must be 'mean' or 'median'")

    if correlation_method not in ("pearson", "spearman"):
        raise ValueError("correlation_method must be 'pearson' or 'spearman'")

    dead_wells = sorted(set([d % n_wells for d in dead_ids]))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]

    # ----------------------------
    # determine base feature names
    # ----------------------------
    reference_props = None
    for exp_idx in exp_conditions:
        z = zroots[exp_idx]
        found = False
        for fov in range(n_fovs):
            for well in valid_wells[: min(10, len(valid_wells))]:
                props = z.get_cell_properties(fov, well, channel)
                if props is not None and len(props) > 0:
                    reference_props = props.copy()
                    found = True
                    break
            if found:
                break
        if found:
            break

    if reference_props is None or len(reference_props) == 0:
        raise ValueError("Could not determine feature names from get_cell_properties(...)")

    feature_names = reference_props.columns.tolist()
    if len(feature_names) > 0 and feature_names[0] == "label":
        feature_names[0] = "count"

    feature_names = [f for f in feature_names if f not in exclude_features]

    # ----------------------------
    # helper
    # ----------------------------
    def _summarize_vector(vals, method="mean"):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return np.nan
        if method == "mean":
            return float(np.nanmean(vals))
        return float(np.nanmedian(vals))

    rows = []

    # ----------------------------
    # collect summaries
    # ----------------------------
    for exp_idx, cond_label in zip(exp_conditions, condition_labels):
        z = zroots[exp_idx]

        for well in valid_wells:
            if summary_level == "strain":
                per_fov_feature_vectors = []
                per_fov_pixel_vals = []

                for fov in range(n_fovs):
                    props = z.get_cell_properties(fov, well, channel)

                    if props is None or len(props) == 0:
                        continue

                    props = props.copy()
                    cols = props.columns.tolist()
                    if len(cols) > 0 and cols[0] == "label":
                        cols[0] = "count"
                        props.columns = cols

                    if not all(f in props.columns for f in feature_names):
                        continue

                    feature_vec = []
                    for fname in feature_names:
                        vals = pd.to_numeric(props[fname], errors="coerce").to_numpy(dtype=float)
                        if fname in abs_value_features:
                            vals = np.abs(vals)
                        feature_vec.append(_summarize_vector(vals, method=summary_method))

                    per_fov_feature_vectors.append(feature_vec)

                    if include_pixel_intensity:
                        try:
                            label_arr = z.get_label_image(fov, well, channel)
                            mask = label_arr != 0
                            img = z.get_processed_intensities(fov, well, channel)
                            pix_vals = img[mask]
                            pix_vals = np.asarray(pix_vals, dtype=float)
                            pix_vals = pix_vals[np.isfinite(pix_vals)]
                            per_fov_pixel_vals.append(_summarize_vector(pix_vals, method=summary_method))
                        except Exception:
                            per_fov_pixel_vals.append(np.nan)

                if len(per_fov_feature_vectors) == 0:
                    continue

                per_fov_feature_vectors = np.asarray(per_fov_feature_vectors, dtype=float)

                row = {
                    "condition_idx": exp_idx,
                    "condition_label": cond_label,
                    "well": well,
                }

                for i, fname in enumerate(feature_names):
                    row[f"{fname} ({channel.lower()})"] = _summarize_vector(
                        per_fov_feature_vectors[:, i],
                        method=summary_method
                    )

                if include_pixel_intensity:
                    row[f"pixel_intensity ({channel.lower()})"] = _summarize_vector(
                        per_fov_pixel_vals,
                        method=summary_method
                    )

                rows.append(row)

            else:  # summary_level == "fov"
                for fov in range(n_fovs):
                    props = z.get_cell_properties(fov, well, channel)

                    if props is None or len(props) == 0:
                        continue

                    props = props.copy()
                    cols = props.columns.tolist()
                    if len(cols) > 0 and cols[0] == "label":
                        cols[0] = "count"
                        props.columns = cols

                    if not all(f in props.columns for f in feature_names):
                        continue

                    row = {
                        "condition_idx": exp_idx,
                        "condition_label": cond_label,
                        "well": well,
                        "fov": fov,
                    }

                    for fname in feature_names:
                        vals = pd.to_numeric(props[fname], errors="coerce").to_numpy(dtype=float)
                        if fname in abs_value_features:
                            vals = np.abs(vals)
                        row[f"{fname} ({channel.lower()})"] = _summarize_vector(vals, method=summary_method)

                    if include_pixel_intensity:
                        try:
                            label_arr = z.get_label_image(fov, well, channel)
                            mask = label_arr != 0
                            img = z.get_processed_intensities(fov, well, channel)
                            pix_vals = img[mask]
                            pix_vals = np.asarray(pix_vals, dtype=float)
                            pix_vals = pix_vals[np.isfinite(pix_vals)]
                            row[f"pixel_intensity ({channel.lower()})"] = _summarize_vector(
                                pix_vals, method=summary_method
                            )
                        except Exception:
                            row[f"pixel_intensity ({channel.lower()})"] = np.nan

                    rows.append(row)

    if len(rows) == 0:
        raise ValueError("No summarized feature rows were generated.")

    feature_table_df = pd.DataFrame(rows)

    metadata_cols = ["condition_idx", "condition_label", "well", "fov"]
    feature_cols = [c for c in feature_table_df.columns if c not in metadata_cols]

    # optional z-score within each condition
    if zscore_within_condition:
        for cond_label in feature_table_df["condition_label"].unique():
            mask = feature_table_df["condition_label"] == cond_label
            block = feature_table_df.loc[mask, feature_cols].to_numpy(dtype=float)
            block = zscore(block, axis=0, nan_policy="omit")
            block = np.where(np.isfinite(block), block, np.nan)
            feature_table_df.loc[mask, feature_cols] = block

    corr_df = feature_table_df[feature_cols].corr(method=correlation_method)

    if return_feature_table:
        return corr_df, feature_table_df

    return corr_df

def plot_feature_covariation_heatmap(
    corr_df,
    figsize=(10, 8),
    cmap="apc:orange_sage_r",
    cluster=True,
    title="Feature-feature covariation",
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=8,
    title_size=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    xtick_rotation=90,
    ytick_rotation=0,
    axis_color="black",
    label_color="black",
    cbar=True,
    cbar_label="correlation",
    cbar_shrink=1.0,
    cbar_aspect=30,
    vmin=-1,
    vmax=1,
    center=0,
    save=False,
    figtitle="feature_covariation_heatmap",
):
    """
    Plot a feature-feature correlation matrix as a heatmap.
    """

    if xtick_fontsize is None:
        xtick_fontsize = font_size
    if ytick_fontsize is None:
        ytick_fontsize = font_size

    plot_df = corr_df.copy()

    if cluster:
        linkage_mat = linkage(plot_df.values, method="average")
        order = leaves_list(linkage_mat)
        plot_df = plot_df.iloc[order, order]

    fig, ax = plt.subplots(figsize=figsize)

    hm = sns.heatmap(
        plot_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar=cbar,
        ax=ax,
        cbar_kws={
            "shrink": cbar_shrink,
            "aspect": cbar_aspect
        } if cbar else None
    )

    if cbar and cbar_label is not None:
        hm.collections[0].colorbar.set_label(
            cbar_label,
            fontsize=font_size,
            fontfamily=font_family
        )

    if show_title:
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

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
        rotation=ytick_rotation,
        fontsize=ytick_fontsize,
        fontfamily=font_family
    )

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", color=axis_color)
    ax.tick_params(axis="y", color=axis_color)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, plot_df

def plot_feature_pair_relationship(
    feature_table_df,
    feature_x,
    feature_y,
    color_by="condition",   # "condition", "none"
    condition_palette=None,
    figsize=(6, 5),
    alpha=0.7,
    point_size=35,
    add_regression_line=False,
    title=None,
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    axis_color="black",
    label_color="black",
    save=False,
    figtitle="feature_pair_relationship",
):
    """
    Plot a scatter of two summarized features from the table returned by
    compute_feature_covariation_matrix(..., return_feature_table=True).
    """

    if feature_x not in feature_table_df.columns:
        raise ValueError(f"{feature_x} not found in feature_table_df")
    if feature_y not in feature_table_df.columns:
        raise ValueError(f"{feature_y} not found in feature_table_df")

    plot_df = feature_table_df[[feature_x, feature_y, "condition_label"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()

    fig, ax = plt.subplots(figsize=figsize)

    if color_by == "condition":
        if condition_palette is None:
            unique_conditions = plot_df["condition_label"].unique().tolist()
            default_colors = [apc.vital, apc.wish, apc.dress, apc.lapis, apc.aster, apc.candy]
            condition_palette = {
                c: default_colors[i % len(default_colors)]
                for i, c in enumerate(unique_conditions)
            }

        for cond, sub in plot_df.groupby("condition_label"):
            ax.scatter(
                sub[feature_x],
                sub[feature_y],
                color=condition_palette.get(cond, "gray"),
                alpha=alpha,
                s=point_size,
                label=cond
            )
        ax.legend(frameon=False)
    else:
        ax.scatter(
            plot_df[feature_x],
            plot_df[feature_y],
            color=apc.aster,
            alpha=alpha,
            s=point_size
        )

    if add_regression_line and len(plot_df) >= 2:
        x = plot_df[feature_x].to_numpy(dtype=float)
        y = plot_df[feature_y].to_numpy(dtype=float)
        coeffs = np.polyfit(x, y, deg=1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        yy = coeffs[0] * xx + coeffs[1]
        ax.plot(xx, yy, color="black", linewidth=1.5)

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)

    if show_title:
        if title is None:
            title = f"{feature_x} vs {feature_y}"
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, plot_df


def compute_feature_covariation_matrix_cell_level(
    zroots,
    channel="BF",
    exp_conditions=None,
    condition_labels=None,
    n_wells=96,
    n_fovs=9,
    dead_ids=None,
    exclude_features=None,
    abs_value_features=None,
    correlation_method="pearson",
    max_cells_per_well=None,  # optional subsampling
):
    """
    Compute feature-feature correlation at the SINGLE CELL level.
    """

    if dead_ids is None:
        dead_ids = []

    if exclude_features is None:
        exclude_features = []

    if abs_value_features is None:
        abs_value_features = []

    if exp_conditions is None:
        exp_conditions = list(range(len(zroots)))

    if condition_labels is None:
        condition_labels = [str(c) for c in exp_conditions]

    dead_wells = sorted(set([d % n_wells for d in dead_ids]))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]

    rows = []

    for exp_idx, cond_label in zip(exp_conditions, condition_labels):
        z = zroots[exp_idx]

        for well in valid_wells:
            for fov in range(n_fovs):

                props = z.get_cell_properties(fov, well, channel)

                if props is None or len(props) == 0:
                    continue

                props = props.copy()

                cols = props.columns.tolist()
                if cols[0] == "label":
                    cols[0] = "count"
                    props.columns = cols

                # remove excluded
                cols = [c for c in props.columns if c not in exclude_features]

                df = props[cols].apply(pd.to_numeric, errors="coerce")

                # optional subsampling
                if max_cells_per_well is not None and len(df) > max_cells_per_well:
                    df = df.sample(max_cells_per_well, random_state=0)

                # apply abs if needed
                for f in abs_value_features:
                    if f in df.columns:
                        df[f] = np.abs(df[f])

                df = df.replace([np.inf, -np.inf], np.nan).dropna()

                if len(df) == 0:
                    continue

                rows.append(df)

    if len(rows) == 0:
        raise ValueError("No valid cell-level data collected.")

    all_cells_df = pd.concat(rows, axis=0, ignore_index=True)

    corr_df = all_cells_df.corr(method=correlation_method)

    return corr_df, all_cells_df


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#it's the freakend weekend! #20260403

def find_feature_extremes_within_size_band(
    zroots,
    target_feature,
    size_feature="area",
    channel="BF",
    exp_conditions=None,
    condition_labels=None,
    n_wells=96,
    n_fovs=9,
    dead_ids=None,
    top_n_cells=20,
    size_percentile_low=74.0,
    size_percentile_high=76.0,
    abs_value_target=False,
    abs_value_size=False,
    return_all_cells_df=False,
):
    """
    Find high and low target-feature cells after restricting to a narrow band
    of similarly sized cells.

    This is useful when target features (e.g. inertia tensor, Hu moments) are
    visually dominated by size differences and you want to compare cells that
    are already matched for size.

    Parameters
    ----------
    zroots : list
        List of ExperimentStore-like objects.
    target_feature : str
        Feature whose extremes you want to inspect, e.g. "inertia_tensor-1-1".
    size_feature : str
        Feature used to define the narrow size band, usually "area".
    channel : str
        Channel to read from, usually "BF" or "CY5".
    exp_conditions : list[int] or None
        Indices into zroots to use. If None, uses all.
    condition_labels : list[str] or None
        Human-readable labels for conditions. If None, uses strings of exp_conditions.
    n_wells : int
        Number of wells.
    n_fovs : int
        Number of FoVs per well.
    dead_ids : list[int] or None
        Expanded dead row ids. Converted internally to dead wells.
    top_n_cells : int
        Number of extreme high and low cells to return.
    size_percentile_low : float
        Lower percentile bound for the size filter.
    size_percentile_high : float
        Upper percentile bound for the size filter.
    abs_value_target : bool
        If True, use absolute value of the target feature before ranking.
    abs_value_size : bool
        If True, use absolute value of the size feature before percentile filtering.
    return_all_cells_df : bool
        If True, also return the full dataframe of all cells and the filtered subset.

    Returns
    -------
    results : dict
        Keys:
            "target_feature"
            "size_feature"
            "channel"
            "size_low_cutoff"
            "size_high_cutoff"
            "n_all_cells"
            "n_band_cells"
            "band_cells_df"
            "top_cells_df"
            "bottom_cells_df"
            optionally "all_cells_df"

    Notes
    -----
    The output tables include locator columns:
        condition_idx, condition_label, well, fov, cell_row_in_props
    so you can later go back and crop those specific cells manually.
    """

    if dead_ids is None:
        dead_ids = []

    if exp_conditions is None:
        exp_conditions = list(range(len(zroots)))

    if condition_labels is None:
        condition_labels = [str(c) for c in exp_conditions]

    if len(condition_labels) != len(exp_conditions):
        raise ValueError("condition_labels must match exp_conditions length")

    if size_percentile_low >= size_percentile_high:
        raise ValueError("size_percentile_low must be < size_percentile_high")

    dead_wells = sorted(set([d % n_wells for d in dead_ids]))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]

    # ----------------------------
    # validate features exist
    # ----------------------------
    checked_any = False
    target_exists = False
    size_exists = False

    for exp_idx in exp_conditions:
        z = zroots[exp_idx]
        found = False
        for fov in range(n_fovs):
            for well in valid_wells[: min(len(valid_wells), 5)]:
                props = z.get_cell_properties(fov, well, channel)
                if props is None or len(props) == 0:
                    continue

                checked_any = True
                cols = props.columns.tolist()
                if len(cols) > 0 and cols[0] == "label":
                    cols[0] = "count"

                if target_feature in cols:
                    target_exists = True
                if size_feature in cols:
                    size_exists = True

                found = target_exists and size_exists
                if found:
                    break
            if found:
                break
        if found:
            break

    if not checked_any:
        raise ValueError("Could not inspect any non-empty feature tables.")

    if not target_exists:
        raise ValueError(f"target_feature '{target_feature}' was not found.")
    if not size_exists:
        raise ValueError(f"size_feature '{size_feature}' was not found.")

    # ----------------------------
    # collect all cells
    # ----------------------------
    rows = []

    for exp_idx, cond_label in zip(exp_conditions, condition_labels):
        z = zroots[exp_idx]

        for well in valid_wells:
            for fov in range(n_fovs):
                props = z.get_cell_properties(fov, well, channel)

                if props is None or len(props) == 0:
                    continue

                props = props.copy()
                cols = props.columns.tolist()
                if len(cols) > 0 and cols[0] == "label":
                    cols[0] = "count"
                    props.columns = cols

                if target_feature not in props.columns or size_feature not in props.columns:
                    continue

                target_vals = pd.to_numeric(props[target_feature], errors="coerce").to_numpy(dtype=float)
                size_vals = pd.to_numeric(props[size_feature], errors="coerce").to_numpy(dtype=float)

                for row_idx, (tv, sv) in enumerate(zip(target_vals, size_vals)):
                    if not np.isfinite(tv) or not np.isfinite(sv):
                        continue

                    if abs_value_target:
                        tv = abs(tv)
                    if abs_value_size:
                        sv = abs(sv)

                    rows.append({
                        "target_feature": target_feature,
                        "size_feature": size_feature,
                        "channel": channel,
                        "condition_idx": exp_idx,
                        "condition_label": cond_label,
                        "well": well,
                        "fov": fov,
                        "cell_row_in_props": row_idx,
                        "target_value": float(tv),
                        "size_value": float(sv),
                    })

    if len(rows) == 0:
        raise ValueError("No valid cells were collected.")

    all_cells_df = pd.DataFrame(rows)

    # ----------------------------
    # percentile filtering on size
    # ----------------------------
    size_low_cutoff = np.nanpercentile(all_cells_df["size_value"].to_numpy(dtype=float), size_percentile_low)
    size_high_cutoff = np.nanpercentile(all_cells_df["size_value"].to_numpy(dtype=float), size_percentile_high)

    band_cells_df = all_cells_df.loc[
        (all_cells_df["size_value"] >= size_low_cutoff) &
        (all_cells_df["size_value"] <= size_high_cutoff)
    ].copy()

    if len(band_cells_df) == 0:
        raise ValueError("No cells fell within the requested size percentile band.")

    # ----------------------------
    # extremes within size-matched band
    # ----------------------------
    top_cells_df = (
        band_cells_df
        .sort_values("target_value", ascending=False)
        .head(top_n_cells)
        .reset_index(drop=True)
    )

    bottom_cells_df = (
        band_cells_df
        .sort_values("target_value", ascending=True)
        .head(top_n_cells)
        .reset_index(drop=True)
    )

    results = {
        "target_feature": target_feature,
        "size_feature": size_feature,
        "channel": channel,
        "size_low_cutoff": float(size_low_cutoff),
        "size_high_cutoff": float(size_high_cutoff),
        "n_all_cells": int(len(all_cells_df)),
        "n_band_cells": int(len(band_cells_df)),
        "band_cells_df": band_cells_df,
        "top_cells_df": top_cells_df,
        "bottom_cells_df": bottom_cells_df,
    }

    if return_all_cells_df:
        results["all_cells_df"] = all_cells_df

    return results

def plot_feature_extremes_within_size_band(
    feature_band_result,
    top_n=12,
    figsize=(10, 5),
    high_color=apc.aster,
    low_color=apc.lapis,
    title=None,
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    rotation=45,
    axis_color="black",
    label_color="black",
    save=False,
    figtitle="feature_extremes_within_size_band",
):
    """
    Plot top and bottom target-feature examples after size-band filtering.

    Input should be the output dict from find_feature_extremes_within_size_band(...).
    """

    target_feature = feature_band_result["target_feature"]
    size_feature = feature_band_result["size_feature"]
    channel = feature_band_result["channel"]

    top_df = feature_band_result["top_cells_df"].head(top_n).copy()
    bottom_df = feature_band_result["bottom_cells_df"].head(top_n).copy()

    plot_df = pd.concat([
        bottom_df.assign(group="low"),
        top_df.assign(group="high")
    ], axis=0).reset_index(drop=True)

    labels = [
        f"{row['condition_label']}\nwell_{int(row['well'])}\nfov_{int(row['fov'])}\nrow_{int(row['cell_row_in_props'])}"
        for _, row in plot_df.iterrows()
    ]

    vals = plot_df["target_value"].to_numpy(dtype=float)
    colors = [low_color if g == "low" else high_color for g in plot_df["group"]]
    pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(pos, vals, color=colors)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=rotation, ha="right")

    ax.set_ylabel(target_feature)

    if show_title:
        if title is None:
            title = (
                f"High/low {target_feature} within matched {size_feature} band "
                f"({channel})"
            )
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize,
            fontfamily=font_family
        )

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
    else:
        ax.tick_params(labelsize=font_size)

    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/" + figtitle + ".png", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/" + figtitle + ".svg", bbox_inches="tight")

    return fig, ax, plot_df


import copy
import numpy as np

def signed_log_transform_hu_features(
    training_data,
    training_feature_labels,
    log_base="log10",   # "log10", "log2", or "ln"
    eps=1e-12,
    in_place=False,
):
    """
    Apply signed log transform to Hu-moment columns only:
        sign(H) * log(|H| + eps)

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Usually training_data_finite
    training_feature_labels : list[str]
        Labels returned by build_training_data(...)
    log_base : str
        "log10", "log2", or "ln"
    eps : float
        Small constant to avoid log(0)
    in_place : bool
        If True, modify training_data directly.
        If False, return a transformed copy.

    Returns
    -------
    transformed_data : dict[str, np.ndarray]
    hu_indices : list[int]
    hu_labels : list[str]
    """

    if in_place:
        transformed_data = training_data
    else:
        transformed_data = {
            cond: arr.copy().astype(float)
            for cond, arr in training_data.items()
        }

    hu_indices = [
        i for i, f in enumerate(training_feature_labels)
        if "moments_hu" in f
    ]
    hu_labels = [training_feature_labels[i] for i in hu_indices]

    if len(hu_indices) == 0:
        raise ValueError("No Hu-moment columns found in training_feature_labels.")

    if log_base == "log10":
        log_fn = np.log10
    elif log_base == "log2":
        log_fn = np.log2
    elif log_base == "ln":
        log_fn = np.log
    else:
        raise ValueError("log_base must be 'log10', 'log2', or 'ln'")

    for cond in transformed_data:
        X = transformed_data[cond]
        H = X[:, hu_indices]
        X[:, hu_indices] = np.sign(H) * log_fn(np.abs(H) + eps)

    return transformed_data, hu_indices, hu_labels

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _resolve_feature_label(feature, channel, training_feature_labels):
    """
    Resolve a user-supplied feature name to an exact entry in
    training_feature_labels (labels look like 'area (bf)').

    Accepts either the full label ('area (bf)') or a bare feature name
    ('area') plus an optional channel ('BF'/'CY5').
    """
    if feature in training_feature_labels:
        return feature

    if channel is not None:
        cand = f"{feature} ({channel.lower()})"
        if cand in training_feature_labels:
            return cand

    # match on the part before ' ('
    matches = [l for l in training_feature_labels if l.split(" (")[0] == feature]
    if channel is not None:
        matches = [l for l in matches if l.endswith(f"({channel.lower()})")]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Feature '{feature}' is ambiguous across channels: {matches}. "
            f"Pass channel='BF' or channel='CY5'."
        )
    raise ValueError(
        f"Feature '{feature}' not found. Available labels include: {training_feature_labels}"
    )
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _collect_strain_fov_matrix(training_data, strains, conditions, n_wells, n_fovs):
    """
    Stack per-FOV feature rows for the requested strains and conditions.

    Returns
    -------
    X          : (n_rows, n_features) float array, one row per (condition, strain, fov)
    strain_ids : (n_rows,) int array
    cond_ids   : (n_rows,) object array
    fov_ids    : (n_rows,) int array
    """
    X, strain_ids, cond_ids, fov_ids = [], [], [], []
    for cond in conditions:
        dat = np.asarray(training_data[cond], dtype=float)
        for s in strains:
            for f in range(n_fovs):
                r = s + f * n_wells
                X.append(dat[r, :])
                strain_ids.append(s)
                cond_ids.append(cond)
                fov_ids.append(f)
    return (
        np.vstack(X),
        np.array(strain_ids),
        np.array(cond_ids, dtype=object),
        np.array(fov_ids),
    )


def compare_strain_phenotypes_pca(
    training_data,
    training_feature_labels,
    strains,
    conditions=None,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    fit_pca_on="selected_strains",   # "selected_strains" or "all_strains"
    standardize=True,
    pc_x=0,
    pc_y=1,
    strain_colors=None,
    marker_by_condition=False,
    point_size=60,
    alpha=0.85,
    show_strain_centroids=True,
    figsize=(8, 7),
    show_title=True,
    title=None,
    title_fontsize=16,
    save=False,
    figtitle="phenotype_pca",
):
    """
    Project each field of view of the selected strains into a shared PCA space,
    one dot per FOV, colored by strain.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        condition -> (n_wells*n_fovs, n_features) array (row = well + fov*n_wells).
        Use a finite version (e.g. training_data_finite or training_data_hu_log).
    training_feature_labels : list[str]
    strains : list[int]
        Any number of wells/strains to compare, e.g. [0, 24, 93].
    conditions : list[str] or None
        Conditions to include. None -> all conditions in training_data.
    fit_pca_on : str
        "selected_strains" -> PCA fit only on the strains you passed.
        "all_strains"      -> PCA fit on all valid wells (shared reference space).
    standardize : bool
        z-score features before PCA so large-scale features (area) don't dominate.
    pc_x, pc_y : int
        Which principal components to plot.
    marker_by_condition : bool
        If True, use a different marker shape per condition (color still = strain).

    Returns
    -------
    fig, ax, coords_df
        coords_df has columns: strain, condition, fov, PCx, PCy.
    """
    if dead_ids is None:
        dead_ids = []
    if conditions is None:
        conditions = list(training_data.keys())
    if fit_pca_on not in ("selected_strains", "all_strains"):
        raise ValueError("fit_pca_on must be 'selected_strains' or 'all_strains'")

    for cond in conditions:
        if cond not in training_data:
            raise ValueError(f"{cond} not found in training_data")

    example = np.asarray(training_data[conditions[0]], dtype=float)
    if n_fovs is None:
        n_fovs = example.shape[0] // n_wells

    dead_wells = sorted(set(d % n_wells for d in dead_ids))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]
    for s in strains:
        if s not in valid_wells:
            raise ValueError(f"strain {s} is not a valid well (dead or out of range)")

    # --- population used to FIT the PCA basis ---
    fit_strains = valid_wells if fit_pca_on == "all_strains" else list(strains)
    Xfit, _, _, _ = _collect_strain_fov_matrix(
        training_data, fit_strains, conditions, n_wells, n_fovs
    )

    # --- points to PLOT (selected strains) ---
    Xplot, strain_ids, cond_ids, fov_ids = _collect_strain_fov_matrix(
        training_data, list(strains), conditions, n_wells, n_fovs
    )

    # --- make finite: inf->nan, drop empty FOV rows, impute with fit col-means ---
    Xfit[~np.isfinite(Xfit)] = np.nan
    Xplot[~np.isfinite(Xplot)] = np.nan

    fit_keep = ~np.all(np.isnan(Xfit), axis=1)
    Xfit = Xfit[fit_keep]
    col_means = np.nanmean(Xfit, axis=0)
    bad = np.where(np.isnan(col_means))[0]
    if len(bad) > 0:
        raise ValueError(f"Feature columns all-NaN in fit set, cannot impute: {bad}")

    plot_keep = ~np.all(np.isnan(Xplot), axis=1)
    Xplot = Xplot[plot_keep]
    strain_ids, cond_ids, fov_ids = strain_ids[plot_keep], cond_ids[plot_keep], fov_ids[plot_keep]

    ri, ci = np.where(np.isnan(Xfit));  Xfit[ri, ci] = col_means[ci]
    rp, cp = np.where(np.isnan(Xplot)); Xplot[rp, cp] = col_means[cp]

    # --- standardize on the fit population ---
    if standardize:
        mu = np.mean(Xfit, axis=0)
        sd = np.std(Xfit, axis=0); sd[sd == 0] = 1.0
        Xfit = (Xfit - mu) / sd
        Xplot = (Xplot - mu) / sd

    # --- fit PCA (pca_on_np_array mutates its input, so pass a copy) ---
    _, evecs, evals, mean_val = pca_on_np_array(Xfit.copy())
    var_explained = evals / np.sum(evals)

    proj = np.dot(Xplot - mean_val, evecs)
    px, py = proj[:, pc_x], proj[:, pc_y]

    # --- colors / markers ---
    if strain_colors is None:
        cmap = plt.get_cmap("tab10" if len(strains) <= 10 else "tab20")
        strain_colors = {s: cmap(i % cmap.N) for i, s in enumerate(strains)}
    elif isinstance(strain_colors, (list, tuple)):
        strain_colors = {s: strain_colors[i] for i, s in enumerate(strains)}

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    cond_markers = {c: marker_cycle[i % len(marker_cycle)] for i, c in enumerate(conditions)}

    fig, ax = plt.subplots(figsize=figsize)
    for s in strains:
        for cond in conditions:
            m = (strain_ids == s) & (cond_ids == cond)
            if not np.any(m):
                continue
            ax.scatter(
                px[m], py[m],
                s=point_size, alpha=alpha,
                color=strain_colors[s],
                marker=cond_markers[cond] if marker_by_condition else "o",
                edgecolor="none",
                zorder=2,
            )
        if show_strain_centroids:
            ms = (strain_ids == s)
            ax.scatter(
                np.mean(px[ms]), np.mean(py[ms]),
                s=point_size * 4, color=strain_colors[s],
                edgecolor="black", linewidth=1.5, zorder=3,
            )
            #ax.text(np.mean(px[ms]), np.mean(py[ms]), f"{s}",
                    #fontsize=10, ha="center", va="center", zorder=4)

    ax.set_xlabel(f"PC{pc_x + 1} ({var_explained[pc_x] * 100:.1f}%)")
    ax.set_ylabel(f"PC{pc_y + 1} ({var_explained[pc_y] * 100:.1f}%)")
    ax.set_xticks([]); ax.set_yticks([])
    '''
    for s in strains:
        ax.scatter([], [], color=strain_colors[s], label=f"strain {s}")
    leg1 = ax.legend(title="Strain", loc="best")
    if marker_by_condition:
        ax.add_artist(leg1)
        handles = [plt.Line2D([], [], marker=cond_markers[c], color="grey",
                              linestyle="none", label=str(c)) for c in conditions]
        ax.legend(handles=handles, title="Condition", loc="lower right")
    '''

    if show_title:
        if title is None:
            fit_txt = "all strains" if fit_pca_on == "all_strains" else "selected strains"
            title = f"Per-FOV PCA (fit on {fit_txt}) — strains {list(strains)}"
        ax.set_title(title, fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()

    coords_df = pd.DataFrame({
        "strain": strain_ids, "condition": cond_ids, "fov": fov_ids,
        "PCx": px, "PCy": py,
    })

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}", bbox_inches="tight")

    return fig, ax, coords_df

def compare_strain_feature(
    training_data,
    training_feature_labels,
    strains,
    feature,
    channel=None,
    conditions=None,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    plot_kind="box",            # "box", "bar", or "strip"
    split_by_condition=False,
    show_points=True,
    point_alpha=0.6,
    strain_colors=None,
    ylabel=None,
    show_xticklabels=False,
    show_axis_labels=False,
    figsize=(8, 5),
    show_title=True,
    title=None,
    title_fontsize=15,
    save=False,
    figtitle="strain_feature_comparison",
):
    """
    Compare a single feature (e.g. 'area') across strains, one point per FOV.

    feature : str
        Full label ('area (bf)') or bare name ('area') + channel='BF'/'CY5'.
    split_by_condition : bool
        If True, split each strain's distribution by condition (hue).

    Returns
    -------
    fig, ax, tidy_df  (columns: strain, condition, fov, value)
    """
    if dead_ids is None:
        dead_ids = []
    if conditions is None:
        conditions = list(training_data.keys())

    label = _resolve_feature_label(feature, channel, training_feature_labels)
    feat_idx = training_feature_labels.index(label)

    example = np.asarray(training_data[conditions[0]], dtype=float)
    if n_fovs is None:
        n_fovs = example.shape[0] // n_wells

    dead_wells = sorted(set(d % n_wells for d in dead_ids))
    for s in strains:
        if s in dead_wells:
            raise ValueError(f"strain {s} is a dead well")

    recs = []
    for cond in conditions:
        dat = np.asarray(training_data[cond], dtype=float)
        for s in strains:
            for f in range(n_fovs):
                v = dat[s + f * n_wells, feat_idx]
                if np.isfinite(v):
                    recs.append({"strain": s, "condition": cond, "fov": f, "value": v})
    tidy_df = pd.DataFrame(recs)

    # seaborn casts the categorical axis to str -> keep strain as str everywhere
    tidy_df["strain"] = tidy_df["strain"].astype(str)
    order = [str(s) for s in strains]

    # build strain palette keyed by the (string) strain labels
    if strain_colors is None:
        cmap = plt.get_cmap("tab10" if len(strains) <= 10 else "tab20")
        strain_palette = {str(s): cmap(i % cmap.N) for i, s in enumerate(strains)}
    elif isinstance(strain_colors, (list, tuple)):
        strain_palette = {str(s): strain_colors[i] for i, s in enumerate(strains)}
    else:  # dict
        strain_palette = {str(s): strain_colors[s] for s in strains}

    fig, ax = plt.subplots(figsize=figsize)

    # when not splitting, color by strain via hue="strain" (avoids the palette/hue warning)
    if split_by_condition:
        main_hue, main_palette, main_legend = "condition", None, True
    else:
        main_hue, main_palette, main_legend = "strain", strain_palette, False

    if plot_kind == "box":
        sns.boxplot(data=tidy_df, x="strain", y="value", order=order,
                    hue=main_hue, palette=main_palette, legend=main_legend,
                    showfliers=not show_points, ax=ax)
    elif plot_kind == "bar":
        sns.barplot(data=tidy_df, x="strain", y="value", order=order,
                    hue=main_hue, palette=main_palette, legend=main_legend,
                    errorbar="sd", ax=ax)
    elif plot_kind == "strip":
        show_points = False
        sns.stripplot(data=tidy_df, x="strain", y="value", order=order,
                      hue=main_hue, palette=main_palette, legend=main_legend,
                      dodge=split_by_condition, alpha=point_alpha, ax=ax)
    else:
        raise ValueError("plot_kind must be 'box', 'bar', or 'strip'")

    # black FOV points on top of box/bar
    if show_points and plot_kind != "strip":
        strip_kwargs = dict(data=tidy_df, x="strain", y="value", order=order,
                            alpha=point_alpha, size=4, legend=False, ax=ax)
        if split_by_condition:
            strip_kwargs.update(hue="condition", dodge=True,
                                palette={c: "black" for c in conditions})
        else:
            strip_kwargs.update(color="black")
        sns.stripplot(**strip_kwargs)

    if show_axis_labels:
        ax.set_xlabel("Strain")
        ax.set_ylabel(ylabel if ylabel is not None else label)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if not show_xticklabels:
        ax.set_xticklabels([])
    if show_title:
        ax.set_title(title if title is not None else f"{label} across strains",
                     fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}", bbox_inches="tight")

    return fig, ax, tidy_df

def plot_strain_pair_feature_heatmap(
    training_data,
    training_feature_labels,
    strain_a,
    strain_b,
    conditions=None,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    reference_strains=None,     # population for z-scoring; None -> all valid wells
    top_n=None,                 # show only the top_n most-different features
    show_diff_row=False,        # add a 3rd row = z(strain_a) - z(strain_b)
    vmax=3.5,
    figsize=(14, 4),
    title=None,
    figtitle="strain_pair_feature_heatmap",
    save=True,
):
    """
    Heatmap of the full feature set for two strains, z-scored across a strain
    population. Features sorted by |z(strain_a) - z(strain_b)| (left = most
    different). Style mirrors plot_strain_heatmap.

    Returns
    -------
    fig, ax, diff_df   (columns: feature, z_a, z_b, diff, abs_diff; sorted)
    """
    if dead_ids is None:
        dead_ids = []
    if conditions is None:
        conditions = list(training_data.keys())

    example = np.asarray(training_data[conditions[0]], dtype=float)
    if n_fovs is None:
        n_fovs = example.shape[0] // n_wells

    dead_wells = sorted(set(d % n_wells for d in dead_ids))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]
    for s in (strain_a, strain_b):
        if s not in valid_wells:
            raise ValueError(f"strain {s} is not a valid well (dead or out of range)")
    if reference_strains is None:
        reference_strains = valid_wells

    # mean feature vector for one strain, pooled over FoVs and conditions
    def _strain_mean(strain):
        rows = []
        for cond in conditions:
            dat = np.asarray(training_data[cond], dtype=float)
            for f in range(n_fovs):
                rows.append(dat[strain + f * n_wells, :])
        rows = np.vstack(rows)
        rows[~np.isfinite(rows)] = np.nan
        return np.nanmean(rows, axis=0)

    # reference population -> per-feature mean / std for z-scoring
    ref = np.vstack([_strain_mean(s) for s in reference_strains])
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0); sd[sd == 0] = 1.0

    z_a = (_strain_mean(strain_a) - mu) / sd
    z_b = (_strain_mean(strain_b) - mu) / sd
    diff = z_a - z_b

    # sort features by absolute difference between the two strains (descending)
    feat_order = np.argsort(np.abs(diff))[::-1]
    if top_n is not None:
        feat_order = feat_order[:top_n]

    feat_sorted = [training_feature_labels[i] for i in feat_order]
    rows = [z_a[feat_order], z_b[feat_order]]
    row_labels = [f"strain {strain_a}", f"strain {strain_b}"]
    if show_diff_row:
        rows.append(diff[feat_order])
        row_labels.append(f"{strain_a} − {strain_b}")
    z_sorted = np.vstack(rows)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(
        z_sorted,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(feat_sorted)))
    ax.set_xticklabels(feat_sorted, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Feature (sorted by |z difference| between strains)", fontsize=10)
    if title is None:
        title = f"Strain {strain_a} vs {strain_b} — feature z-scores (vs strain population)"
    ax.set_title(title, fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label("Z-score vs strain population", fontsize=9)

    if save:
        for ext in ("svg", "pdf", "png"):
            fig.savefig(f"plots/{figtitle}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Saved to plots/{figtitle}.*")
    plt.show()

    diff_df = pd.DataFrame({
        "feature": [training_feature_labels[i] for i in feat_order],
        "z_a": z_a[feat_order],
        "z_b": z_b[feat_order],
        "diff": diff[feat_order],
        "abs_diff": np.abs(diff[feat_order]),
    })
    return fig, ax, diff_df
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _resolve_condition_key(temperature, growth_stage, available_conditions):
    """
    Resolve a temperature (16/22/30) and growth stage ('e_stat'/'l_stat') to a
    condition key in training_data (keys look like 'e_stat_16').

    Accepts temperature as 16, "16", "16C", "16C" and growth stage as
    "e_stat"/"early"/"e"/"early_stationary" or "l_stat"/"late"/"l"/"late_stationary".
    """
    temp_digits = "".join(ch for ch in str(temperature) if ch.isdigit())
    if temp_digits == "":
        raise ValueError(f"Could not read a temperature from {temperature!r}")

    stage_raw = str(growth_stage).strip().lower().replace(".", "_").replace("-", "_")
    if stage_raw.startswith("e"):
        stage = "e_stat"
    elif stage_raw.startswith("l"):
        stage = "l_stat"
    else:
        raise ValueError(
            f"Could not read a growth stage from {growth_stage!r}. "
            "Use 'e_stat'/'early' or 'l_stat'/'late'."
        )

    candidate = f"{stage}_{temp_digits}"
    if candidate in available_conditions:
        return candidate

    # fall back to any key that carries both the stage and the temperature
    matches = [
        c for c in available_conditions
        if stage in str(c).lower() and temp_digits in str(c)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Temperature {temperature} + growth stage {growth_stage} is ambiguous: {matches}"
        )
    raise ValueError(
        f"No condition found for temperature {temperature} and growth stage {growth_stage}. "
        f"Available conditions: {list(available_conditions)}"
    )
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _select_feature_indices(training_feature_labels, exclude_features=None,
                            include_features=None, exclude_channels=None):
    """
    Pick which feature columns to keep for a heatmap.

    exclude_features / include_features accept, for each entry:
      - a full label      -> 'moments_hu-0 (bf)'  (that column only)
      - a bare name       -> 'moments_hu-0'       (that feature, both channels)
      - a partial name    -> 'moments_hu'         (every Hu moment, both channels)
    Exact matches win: 'area' drops only 'area (bf)'/'area (cy5)', never
    'area_convex'. A token that matches nothing exactly falls back to a
    substring match, so 'moments_hu' drops all of them at once.
    exclude_channels accepts e.g. ['CY5'].
    """
    labels = list(training_feature_labels)
    bare_labels = [l.split(" (")[0] for l in labels]

    def _matched_indices(tokens, strict):
        if isinstance(tokens, str):
            tokens = [tokens]
        hits = set()
        for token in tokens:
            token = str(token)
            exact = [
                i for i, l in enumerate(labels)
                if l == token or bare_labels[i] == token
            ]
            if exact:
                hits.update(exact)
                continue
            partial = [i for i, l in enumerate(labels) if token in l]
            if partial:
                hits.update(partial)
            elif strict:
                raise ValueError(
                    f"Feature '{token}' not found in training_feature_labels."
                )
        return hits

    keep = list(range(len(labels)))

    if include_features is not None:
        wanted = _matched_indices(include_features, strict=True)
        keep = [i for i in keep if i in wanted]

    # features already dropped upstream (e.g. in build_training_data) are ignored
    if exclude_features is not None:
        dropped = _matched_indices(exclude_features, strict=False)
        keep = [i for i in keep if i not in dropped]

    if exclude_channels is not None:
        if isinstance(exclude_channels, str):
            exclude_channels = [exclude_channels]
        drop = tuple(f"({c.lower()})" for c in exclude_channels)
        keep = [i for i in keep if not labels[i].endswith(drop)]

    if len(keep) == 0:
        raise ValueError("No features left after applying include/exclude filters.")

    return keep, [labels[i] for i in keep]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_all_strain_feature_heatmap(
    training_data,
    training_feature_labels,
    temperature,
    growth_stage,
    fovs=0,                       # int, list of ints, or "all"
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    strains=None,                 # None -> every valid well
    exclude_features=None,        # e.g. ["moments_hu-0", "moments_hu"] or full labels
    include_features=None,
    exclude_channels=None,        # e.g. ["CY5"]
    normalize="zscore",           # "zscore", "double_zscore", "minmax", or None
    sort_features=False,
    sort_by="variance",           # "variance" or "mean"
    cluster_strains=False,
    cluster_features=False,
    cluster_metric="euclidean",
    cluster_method="average",
    orientation="vertical",       # "vertical" -> features on y, strains on x
    figsize=(18, 10),
    cmap="apc:orange_sage_r",
    title=None,
    show_title=True,
    title_fontsize=18,
    font_family=None,
    font_size=10,
    title_size=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    xtick_rotation=90,
    ytick_rotation=0,
    show_xticklabels=True,
    show_yticklabels=True,
    axis_color="black",
    label_color="black",
    xtick_colors=None,
    ytick_colors=None,
    cbar=True,
    cbar_label=None,
    cbar_shrink=1.0,
    cbar_aspect=30,
    vmin=None,
    vmax=None,
    save=False,
    figtitle=None,
):
    """
    Heatmap of every strain x every feature for one experimental condition,
    selected by temperature and growth stage.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Condition -> array of shape (n_wells * n_fovs, n_features).
        Usually training_data_finite or training_data_hu_log.
    training_feature_labels : list[str]
        Labels returned by build_training_data(...).
    temperature : int or str
        16, 22, or 30 (also accepts "16C").
    growth_stage : str
        "e_stat"/"early" or "l_stat"/"late".
    fovs : int, list[int], or "all"
        Field(s) of view to use. Default 0 (a single FoV). Pass a list, e.g.
        [0, 1, 2], or "all" to average feature values across several FoVs.
    dead_ids : list[int] or None
        Expanded dead row indices; the corresponding wells are dropped.
    strains : list[int] or None
        Wells to show. None -> every valid well.
    exclude_features / include_features : list[str] or None
        Accepts full labels ('moments_hu-0 (bf)'), bare names ('moments_hu-0'),
        or partial names ('moments_hu' drops every Hu moment in both channels).
    normalize : str or None
        "zscore"        -> per feature, across strains (default)
        "double_zscore" -> z-score across strains, then across the whole matrix
        "minmax"        -> per feature, scaled to [0, 1]
        None            -> raw feature means
    sort_features : bool
        Order features by `sort_by` ("variance" or "mean" across strains).
    cluster_strains / cluster_features : bool
        Hierarchically order strains and/or features (overrides sort_features).
    orientation : str
        "vertical"   -> features on y-axis, strains on x-axis
        "horizontal" -> strains on y-axis, features on x-axis

    Returns
    -------
    fig, ax, plot_df
    """
    if dead_ids is None:
        dead_ids = []

    condition = _resolve_condition_key(temperature, growth_stage, list(training_data.keys()))

    dat = np.asarray(training_data[condition], dtype=float)
    n_rows, n_features_total = dat.shape

    if len(training_feature_labels) != n_features_total:
        raise ValueError(
            f"training_feature_labels has length {len(training_feature_labels)} "
            f"but training_data['{condition}'] has {n_features_total} columns"
        )

    if n_fovs is None:
        n_fovs = n_rows // n_wells

    # --- Which FoVs ---
    if isinstance(fovs, str):
        if fovs.lower() != "all":
            raise ValueError('fovs must be an int, a list of ints, or "all"')
        fov_list = list(range(n_fovs))
    elif isinstance(fovs, (int, np.integer)):
        fov_list = [int(fovs)]
    else:
        fov_list = [int(f) for f in fovs]

    bad_fovs = [f for f in fov_list if f < 0 or f >= n_fovs]
    if bad_fovs:
        raise ValueError(f"FoV(s) {bad_fovs} out of range; this dataset has {n_fovs} FoVs (0-{n_fovs - 1})")

    # --- Which strains ---
    dead_wells = sorted(set(d % n_wells for d in dead_ids))
    valid_wells = [w for w in range(n_wells) if w not in dead_wells]
    if strains is None:
        strains_to_use = valid_wells
    else:
        strains_to_use = [int(s) for s in strains]
        for s in strains_to_use:
            if s in dead_wells:
                raise ValueError(f"strain {s} is a dead well")
            if s < 0 or s >= n_wells:
                raise ValueError(f"strain {s} is out of range (0-{n_wells - 1})")

    # --- Which features ---
    feat_idx, feature_labels = _select_feature_indices(
        training_feature_labels,
        exclude_features=exclude_features,
        include_features=include_features,
        exclude_channels=exclude_channels,
    )

    # --- Strain x feature matrix (mean across the selected FoVs) ---
    strain_mat, strain_labels_used = [], []
    for well in strains_to_use:
        block = dat[[well + f * n_wells for f in fov_list], :][:, feat_idx]
        block = np.where(np.isfinite(block), block, np.nan)
        if np.all(np.isnan(block)):
            continue
        strain_mat.append(np.nanmean(block, axis=0))
        strain_labels_used.append(well)

    if len(strain_mat) == 0:
        raise ValueError(f"No strains with finite values found for condition {condition}")

    strain_mat = np.vstack(strain_mat)

    # --- Normalize ("formalize") the values ---
    if normalize in (None, "none", False):
        plot_mat = strain_mat.copy()
        default_cbar_label = "Feature value"
    elif normalize == "zscore":
        plot_mat = zscore(strain_mat, axis=0, nan_policy="omit")
        plot_mat = np.where(np.isfinite(plot_mat), plot_mat, 0.0)
        default_cbar_label = "Z-score across strains"
    elif normalize == "double_zscore":
        plot_mat = zscore(strain_mat, axis=0, nan_policy="omit")
        plot_mat = np.where(np.isfinite(plot_mat), plot_mat, 0.0)
        plot_mat = zscore(plot_mat, axis=None)
        default_cbar_label = "Double z-score"
    elif normalize == "minmax":
        col_min = np.nanmin(strain_mat, axis=0)
        col_max = np.nanmax(strain_mat, axis=0)
        span = col_max - col_min
        span[span == 0] = 1.0
        plot_mat = (strain_mat - col_min) / span
        plot_mat = np.where(np.isfinite(plot_mat), plot_mat, 0.0)
        default_cbar_label = "Min-max scaled value"
    else:
        raise ValueError('normalize must be "zscore", "double_zscore", "minmax", or None')

    feature_labels = np.array(feature_labels)
    strain_labels_used = np.array(strain_labels_used)

    # --- Feature ordering ---
    if cluster_features:
        link_f = linkage(plot_mat.T, method=cluster_method, metric=cluster_metric)
        order = leaves_list(link_f)
        plot_mat = plot_mat[:, order]
        feature_labels = feature_labels[order]
    elif sort_features:
        if sort_by == "variance":
            sort_metric = np.nanvar(plot_mat, axis=0)
        elif sort_by == "mean":
            sort_metric = np.nanmean(plot_mat, axis=0)
        else:
            raise ValueError("sort_by must be 'variance' or 'mean'")
        order = np.argsort(sort_metric)[::-1]  # descending
        plot_mat = plot_mat[:, order]
        feature_labels = feature_labels[order]

    # --- Strain ordering ---
    if cluster_strains:
        link_s = linkage(plot_mat, method=cluster_method, metric=cluster_metric)
        order = leaves_list(link_s)
        plot_mat = plot_mat[order, :]
        strain_labels_used = strain_labels_used[order]

    plot_df = pd.DataFrame(
        plot_mat,
        index=[str(s) for s in strain_labels_used],
        columns=feature_labels,
    )
    plot_df.index.name = "strain"

    # --- Orientation ---
    if orientation == "vertical":
        plot_df = plot_df.T
    elif orientation != "horizontal":
        raise ValueError('orientation must be "horizontal" or "vertical"')

    if xtick_fontsize is None:
        xtick_fontsize = font_size
    if ytick_fontsize is None:
        ytick_fontsize = font_size
    if cbar_label is None:
        cbar_label = default_cbar_label

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    hm = sns.heatmap(
        plot_df,
        cmap=cmap,
        cbar=cbar,
        vmin=vmin,
        vmax=vmax,
        xticklabels=show_xticklabels,
        yticklabels=show_yticklabels,
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
        if title is None:
            fov_txt = ("FoVs " + ", ".join(str(f) for f in fov_list)) if len(fov_list) > 1 \
                else f"FoV {fov_list[0]}"
            title = f"All strains x features — {condition} ({fov_txt})"
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize or font_size,
            fontfamily=font_family
        )

    # --- Tick formatting ---
    if show_xticklabels:
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=xtick_rotation,
            ha="right" if xtick_rotation not in (0, 90) else "center",
            rotation_mode="anchor" if xtick_rotation not in (0, 90) else None,
            fontsize=xtick_fontsize,
            fontfamily=font_family
        )
    if show_yticklabels:
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=ytick_rotation,
            fontsize=ytick_fontsize,
            fontfamily=font_family
        )

    # --- Default label color ---
    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    # --- Apply custom colors LAST ---
    if xtick_colors is not None:
        if len(xtick_colors) != len(ax.get_xticklabels()):
            raise ValueError("xtick_colors length mismatch")
        for tick, color in zip(ax.get_xticklabels(), xtick_colors):
            tick.set_color(color)

    if ytick_colors is not None:
        if len(ytick_colors) != len(ax.get_yticklabels()):
            raise ValueError("ytick_colors length mismatch")
        for tick, color in zip(ax.get_yticklabels(), ytick_colors):
            tick.set_color(color)

    # --- Axis labels / styling ---
    if orientation == "vertical":
        ax.set_xlabel("Strain (well)", fontsize=font_size, fontfamily=font_family, color=axis_color)
        ax.set_ylabel("")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("Strain (well)", fontsize=font_size, fontfamily=font_family, color=axis_color)

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", color=axis_color)
    ax.tick_params(axis="y", color=axis_color)

    plt.tight_layout()
    plt.show()

    if save:
        if figtitle is None:
            figtitle = f"all_strain_feature_heatmap_{condition}_fov" + "_".join(str(f) for f in fov_list)
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")

    return fig, ax, plot_df
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _well_fov_cube(matrix, n_wells=96, n_fovs=9, dead_ids=None):
    """
    Reshape a (n_wells * n_fovs, n_features) condition matrix into
    (n_fovs, n_strains_kept, n_features).

    Rows are laid out as well + fov * n_wells, matching build_training_data.
    Dead wells are dropped by well identity, so the same wells are removed from
    every FoV.

    Returns
    -------
    cube  : (n_fovs, n_strains_kept, n_features) float array
    keep  : (n_strains_kept,) int array of well indices retained
    """
    matrix = np.asarray(matrix, dtype=float)
    n_features = matrix.shape[1]

    cube = np.full((n_fovs, n_wells, n_features), np.nan)
    for fov in range(n_fovs):
        lo = fov * n_wells
        cube[fov] = matrix[lo: lo + n_wells, :]

    if dead_ids:
        dead_wells = sorted({int(r) % n_wells for r in dead_ids})
        keep = np.array([w for w in range(n_wells) if w not in dead_wells])
    else:
        keep = np.arange(n_wells)

    return cube[:, keep, :], keep
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _icc_oneway(Y):
    """
    One-way random-effects ICC, vectorized over any trailing axes.

    Y : (n_fovs, n_strains) for a single feature, or
        (n_fovs, n_strains, n_features) for a whole condition at once.
    Strains (wells) are the random factor, FoVs are technical replicates.

    Returns
    -------
    icc_single : ICC(1,1) — the fraction of the variance in ONE FoV measurement
                 that is strain-level rather than field-to-field noise
    icc_mean   : ICC(1,k) — reliability of the strain mean over all k FoVs
                 (the Spearman-Brown stepped-up version of ICC(1,1))
    fstat      : MS_between / MS_within, the one-way ANOVA F statistic
    msb, msw   : the two mean squares

    This is the bias-corrected version of a between/within variance ratio: the
    naive ratio uses var(strain means), whose expectation is
    sigma_between^2 + sigma_within^2 / k, so it counts a 1/k share of the
    technical noise as strain signal — worst for the weakest features.
    """
    Y = np.asarray(Y, dtype=float)
    k, n = Y.shape[0], Y.shape[1]

    if k < 2 or n < 2:
        raise ValueError("ICC needs at least 2 FoVs and 2 strains")

    grand = np.nanmean(Y, axis=(0, 1))
    strain_means = np.nanmean(Y, axis=0)

    msb = k * np.nansum((strain_means - grand) ** 2, axis=0) / (n - 1)
    msw = np.nansum((Y - strain_means[None, ...]) ** 2, axis=(0, 1)) / (n * (k - 1))

    with np.errstate(divide="ignore", invalid="ignore"):
        bad = ~np.isfinite(msb) | ~np.isfinite(msw) | (msw <= 0)
        icc_single = (msb - msw) / (msb + (k - 1) * msw)
        icc_mean = (msb - msw) / msb
        fstat = msb / msw

    icc_single = np.where(bad, np.nan, icc_single)
    icc_mean = np.where(bad, np.nan, icc_mean)
    fstat = np.where(bad, np.nan, fstat)

    return icc_single, icc_mean, fstat, msb, msw
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _spearman_columns(A, B):
    """
    Column-wise Spearman correlation between two (n_strains, n_features) arrays.
    Returns a (n_features,) array.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    ra = stats.rankdata(A, axis=0).astype(float)
    rb = stats.rankdata(B, axis=0).astype(float)

    ra = ra - ra.mean(axis=0)
    rb = rb - rb.mean(axis=0)

    denom = np.sqrt((ra ** 2).sum(axis=0) * (rb ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (ra * rb).sum(axis=0) / denom
    return np.where(np.isfinite(rho), rho, np.nan)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def compute_feature_icc(
    training_data,
    training_feature_labels,
    condition_order=None,
    n_wells=96,
    n_fovs=None,
    dead_ids=None,
    n_permutations=0,
    n_split_half_repeats=100,
    icc_threshold=0.8,
    random_state=42,
):
    """
    How much of each feature's variance is strain-level rather than FoV noise?

    For every (feature, condition) pair this runs a one-way random-effects ANOVA
    with wells (= strains) as the random factor and the FoVs as technical
    replicates, and reports ICC(1,1). Features with a high ICC separate strains;
    features with a low ICC are dominated by field-to-field noise and cannot
    support a screen no matter how many strains you measure.

    It also reports how consistently each feature ranks strains ACROSS
    conditions (cross_condition_rho), which is the complementary question: a
    feature can be highly reliable within a condition and still reshuffle the
    strain ranking between environments, and those two properties support
    different kinds of screen.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Condition -> array of shape (n_wells * n_fovs, n_features).
        Usually training_data_finite or training_data_hu_log.
    training_feature_labels : list[str]
        Labels returned by build_training_data(...).
    condition_order : list[str] or None
        Conditions to analyze. None -> every key in training_data.
    n_fovs : int or None
        FoVs per well. None -> inferred from the row count.
    dead_ids : list[int] or None
        Expanded dead row indices; the corresponding wells are dropped.
    n_permutations : int
        Permutation test for the F statistic, shuffling measurements across
        strains within each feature. 0 skips it (fast). 500 reproduces the
        original reliability analysis; the p-value floor is 1/(n_permutations+1).
    n_split_half_repeats : int
        Repeats for the split-half check (Spearman correlation of strain
        rankings between disjoint halves of the FoVs, stepped up by
        Spearman-Brown). 0 skips it.
    icc_threshold : float
        Bar used to classify features as always / conditionally / never usable.

    Returns
    -------
    icc_results : dict
        icc_single           (n_features, n_conditions) ICC(1,1)
        icc_mean             (n_features, n_conditions) ICC(1,k) of the FoV mean
        fstat, pvalue        (n_features, n_conditions)
        split_half           (n_features, n_conditions)
        mean_icc, min_icc, max_icc, cross_condition_rho   (n_features,)
        best_condition, worst_condition                   (n_features,) labels
        always / conditional / never                      lists of feature labels
        table                pd.DataFrame, one row per feature, sorted by mean ICC
        labels, conditions, n_strains, n_fovs, icc_threshold
    """
    rng = np.random.default_rng(random_state)

    if condition_order is None:
        condition_order = list(training_data.keys())

    n_features = len(training_feature_labels)
    n_cond = len(condition_order)

    example = np.asarray(training_data[condition_order[0]], dtype=float)
    if example.shape[1] != n_features:
        raise ValueError(
            f"training_feature_labels has length {n_features} but "
            f"training_data['{condition_order[0]}'] has {example.shape[1]} columns"
        )
    if n_fovs is None:
        n_fovs = example.shape[0] // n_wells

    out = {
        key: np.full((n_features, n_cond), np.nan)
        for key in ("icc_single", "icc_mean", "fstat", "pvalue", "split_half")
    }
    strain_means_by_cond = {}
    n_strains = None

    for ci, cond in enumerate(condition_order):
        cube, keep = _well_fov_cube(training_data[cond], n_wells, n_fovs, dead_ids)
        n_strains = len(keep)
        strain_means_by_cond[cond] = np.nanmean(cube, axis=0)

        icc1, icck, f, _, _ = _icc_oneway(cube)
        out["icc_single"][:, ci] = icc1
        out["icc_mean"][:, ci] = icck
        out["fstat"][:, ci] = f

        # --- permutation null: break the strain labels, keep the values ---
        if n_permutations:
            flat = cube.reshape(n_fovs * n_strains, n_features)
            hits = np.zeros(n_features)
            for _ in range(n_permutations):
                shuffled = rng.permuted(flat, axis=0)
                f_null = _icc_oneway(shuffled.reshape(n_fovs, n_strains, n_features))[2]
                hits += np.where(np.isfinite(f_null) & (f_null >= f), 1.0, 0.0)
            out["pvalue"][:, ci] = (hits + 1) / (n_permutations + 1)

        # --- split-half reproducibility of the strain ranking ---
        half = n_fovs // 2
        if n_split_half_repeats and half >= 1:
            rs = np.full((n_split_half_repeats, n_features), np.nan)
            for r in range(n_split_half_repeats):
                idx = rng.permutation(n_fovs)
                a = np.nanmean(cube[idx[:half]], axis=0)
                b = np.nanmean(cube[idx[half: 2 * half]], axis=0)
                rs[r] = _spearman_columns(a, b)
            raw = np.nanmedian(rs, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                stepped = 2 * raw / (1 + raw)
            out["split_half"][:, ci] = np.where(raw > -1, stepped, np.nan)

    # --- does a feature rank strains the same way in every environment? ---
    cross_rho = np.full(n_features, np.nan)
    if n_cond > 1:
        pair_rhos = []
        for i in range(n_cond):
            for j in range(i + 1, n_cond):
                pair_rhos.append(
                    _spearman_columns(
                        strain_means_by_cond[condition_order[i]],
                        strain_means_by_cond[condition_order[j]],
                    )
                )
        cross_rho = np.nanmean(np.vstack(pair_rhos), axis=0)

    icc = out["icc_single"]
    mean_icc = np.nanmean(icc, axis=1)
    min_icc = np.nanmin(icc, axis=1)
    max_icc = np.nanmax(icc, axis=1)
    conditions_arr = np.array(condition_order, dtype=object)
    best_condition = conditions_arr[np.nanargmax(np.nan_to_num(icc, nan=-np.inf), axis=1)]
    worst_condition = conditions_arr[np.nanargmin(np.nan_to_num(icc, nan=np.inf), axis=1)]

    passes = icc > icc_threshold
    labels_arr = np.array(training_feature_labels, dtype=object)
    always = list(labels_arr[passes.all(axis=1)])
    never = list(labels_arr[~passes.any(axis=1)])
    conditional = list(labels_arr[passes.any(axis=1) & ~passes.all(axis=1)])

    table = pd.DataFrame({
        "feature": list(training_feature_labels),
        "mean_icc": mean_icc,
        "min_icc": min_icc,
        "max_icc": max_icc,
        "best_condition": best_condition,
        "worst_condition": worst_condition,
        "icc_mean_of_fovs": np.nanmean(out["icc_mean"], axis=1),
        "split_half": np.nanmean(out["split_half"], axis=1),
        "max_pvalue": np.nanmax(out["pvalue"], axis=1) if n_permutations else np.nan,
        "cross_condition_rho": cross_rho,
        "n_conditions_passing": passes.sum(axis=1),
    }).sort_values("mean_icc", ascending=False).reset_index(drop=True)

    out.update({
        "labels": list(training_feature_labels),
        "conditions": list(condition_order),
        "mean_icc": mean_icc,
        "min_icc": min_icc,
        "max_icc": max_icc,
        "best_condition": best_condition,
        "worst_condition": worst_condition,
        "cross_condition_rho": cross_rho,
        "passes_per_feature": passes.sum(axis=1),
        "passes_per_condition": passes.sum(axis=0),
        "always": always,
        "conditional": conditional,
        "never": never,
        "table": table,
        "n_strains": n_strains,
        "n_fovs": n_fovs,
        "icc_threshold": icc_threshold,
    })
    return out
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_feature_icc_heatmap(
    icc_results,
    metric="icc_single",          # "icc_single", "icc_mean", "split_half", "cross_condition_rho"
    column="mean",                # "mean", a single condition name, or "per_condition"
    sort_features=True,
    ascending=False,
    top_n=None,
    exclude_features=None,
    include_features=None,
    exclude_channels=None,
    col_titles=None,
    figsize=(2, 12),
    cmap="apc:verde",
    vmin=0,
    vmax=1,
    icc_threshold=None,           # outline cells at or below this value
    threshold_color="black",
    threshold_lw=0.6,
    annot=True,
    annot_fontsize=8,
    annot_fmt=".2f",
    annot_color="auto",           # "auto" picks black or white per cell, or pass a color
    linewidths=0,
    linecolor="white",
    title=None,
    show_title=False,
    title_fontsize=14,
    font_family=None,
    font_size=10,
    title_size=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    xtick_rotation=45,
    ytick_rotation=0,
    show_xticklabels=False,
    show_yticklabels=True,
    axis_color="black",
    label_color="black",
    xtick_colors=None,
    ytick_colors=None,
    cbar=False,                   # no colorbar by default
    cbar_label=None,
    cbar_shrink=1.0,
    cbar_aspect=30,
    save=False,
    figtitle="feature_icc_heatmap",
):
    """
    Single-column heatmap of per-feature reliability, sorted by ICC.

    Parameters
    ----------
    icc_results : dict
        Output of compute_feature_icc(...).
    metric : str
        "icc_single"           ICC(1,1), reliability of one FoV (default)
        "icc_mean"             ICC(1,k), reliability of the strain mean
        "split_half"           split-half reproducibility of the strain ranking
        "cross_condition_rho"  consistency of the strain ranking across conditions
    column : str
        "mean"            one column, averaged over conditions (default)
        a condition name  one column, that condition only (e.g. "l_stat_30")
        "per_condition"   one column per condition
    sort_features : bool
        Sort features by the plotted value (descending unless ascending=True).
        With column="per_condition", sorting uses the row mean.
    top_n : int or None
        Keep only the top_n features after sorting.
    exclude_features / include_features / exclude_channels
        Same matching rules as plot_all_strain_feature_heatmap.
    icc_threshold : float or None
        Outline every cell at or below this value. None -> use the threshold
        stored in icc_results.
    cbar : bool
        Off by default — the figure is the heatmap alone.

    Returns
    -------
    fig, ax, plot_df
    """
    labels = icc_results["labels"]
    conditions = icc_results["conditions"]

    # --- pick the metric ---
    if metric == "cross_condition_rho":
        values = np.asarray(icc_results["cross_condition_rho"], dtype=float)[:, None]
        value_conditions = ["across conditions"]
        default_label = "Cross-condition Spearman rho"
    elif metric in ("icc_single", "icc_mean", "split_half"):
        values = np.asarray(icc_results[metric], dtype=float)
        value_conditions = list(conditions)
        default_label = {
            "icc_single": "ICC(1,1)  (strain variance / total variance)",
            "icc_mean": f"ICC(1,{icc_results.get('n_fovs', 'k')})  (reliability of the strain mean)",
            "split_half": "Split-half rho (Spearman-Brown)",
        }[metric]
    else:
        raise ValueError(
            'metric must be "icc_single", "icc_mean", "split_half", or "cross_condition_rho"'
        )

    # --- pick the column(s) ---
    if metric == "cross_condition_rho" or column == "mean":
        if metric == "cross_condition_rho":
            plot_values = values
            column_titles = value_conditions
        else:
            plot_values = np.nanmean(values, axis=1, keepdims=True)
            column_titles = ["mean"]
    elif column == "per_condition":
        plot_values = values
        column_titles = list(col_titles) if col_titles is not None else value_conditions
    elif column in value_conditions:
        ci = value_conditions.index(column)
        plot_values = values[:, [ci]]
        column_titles = [column]
    else:
        raise ValueError(
            f'column must be "mean", "per_condition", or one of {value_conditions}'
        )

    if len(column_titles) != plot_values.shape[1]:
        raise ValueError("col_titles length does not match the number of columns plotted")

    # --- feature filtering ---
    feat_idx, feature_labels = _select_feature_indices(
        labels,
        exclude_features=exclude_features,
        include_features=include_features,
        exclude_channels=exclude_channels,
    )
    plot_values = plot_values[feat_idx, :]
    feature_labels = np.array(feature_labels, dtype=object)

    # --- sorting ---
    if sort_features:
        sort_metric = np.nanmean(plot_values, axis=1)
        order = np.argsort(np.nan_to_num(sort_metric, nan=-np.inf))
        if not ascending:
            order = order[::-1]
        plot_values = plot_values[order, :]
        feature_labels = feature_labels[order]

    if top_n is not None:
        plot_values = plot_values[:top_n, :]
        feature_labels = feature_labels[:top_n]

    plot_df = pd.DataFrame(plot_values, index=feature_labels, columns=column_titles)
    plot_df.index.name = "feature"

    if xtick_fontsize is None:
        xtick_fontsize = font_size
    if ytick_fontsize is None:
        ytick_fontsize = font_size
    if cbar_label is None:
        cbar_label = default_label
    if icc_threshold is None:
        icc_threshold = icc_results.get("icc_threshold", None)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    hm = sns.heatmap(
        plot_df,
        cmap=cmap,
        cbar=cbar,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=annot_fmt,
        annot_kws={"fontsize": annot_fontsize} if annot else None,
        linewidths=linewidths,
        linecolor=linecolor,
        xticklabels=show_xticklabels,
        yticklabels=show_yticklabels,
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
        # --- Annotation color: "auto" keeps the number legible on any cell ---
    if annot:
        mesh = hm.collections[0]
        for text in ax.texts:
            if annot_color != "auto":
                text.set_color(annot_color)
                continue
            xi, yi = text.get_position()
            value = plot_df.to_numpy(dtype=float)[int(yi), int(xi)]
            if not np.isfinite(value):
                continue
            r, g, b = mesh.cmap(mesh.norm(value))[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text.set_color("black" if luminance > 0.55 else "white")

    # --- Mark the cells that fail the reliability bar ---
    if icc_threshold is not None:
        M = plot_df.to_numpy(dtype=float)
        for yi in range(M.shape[0]):
            for xi in range(M.shape[1]):
                if np.isfinite(M[yi, xi]) and M[yi, xi] <= icc_threshold:
                    ax.add_patch(plt.Rectangle(
                        (xi, yi), 1, 1, fill=False,
                        edgecolor=threshold_color, lw=threshold_lw
                    ))

    # --- Title ---
    if show_title:
        if title is None:
            title = default_label
        ax.set_title(
            title,
            fontsize=title_size or title_fontsize or font_size,
            fontfamily=font_family
        )

    # --- Tick formatting ---
    if show_xticklabels:
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=xtick_rotation,
            ha="right" if xtick_rotation not in (0, 90) else "center",
            rotation_mode="anchor" if xtick_rotation not in (0, 90) else None,
            fontsize=xtick_fontsize,
            fontfamily=font_family
        )
    if show_yticklabels:
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=ytick_rotation,
            fontsize=ytick_fontsize,
            fontfamily=font_family
        )

    # --- Default label color ---
    for tick in ax.get_xticklabels():
        tick.set_color(label_color)
    for tick in ax.get_yticklabels():
        tick.set_color(label_color)

    # --- Apply custom colors LAST ---
    if xtick_colors is not None:
        if len(xtick_colors) != len(ax.get_xticklabels()):
            raise ValueError("xtick_colors length mismatch")
        for tick, color in zip(ax.get_xticklabels(), xtick_colors):
            tick.set_color(color)

    if ytick_colors is not None:
        if len(ytick_colors) != len(ax.get_yticklabels()):
            raise ValueError("ytick_colors length mismatch")
        for tick, color in zip(ax.get_yticklabels(), ytick_colors):
            tick.set_color(color)

    # --- Axis styling ---
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="x", color=axis_color)
    ax.tick_params(axis="y", color=axis_color)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".svg", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".pdf", bbox_inches="tight")
        fig.savefig("plots/chlorella_pub_fig_" + figtitle + ".png", bbox_inches="tight")

    return fig, ax, plot_df
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from cvp_baseline_additions import (
    compare_models_condition_holdout_cv_recalibrated as compare,
    summarize_interaction_capture,
)

rng = np.random.default_rng(0)
N_STRAIN, N_FOV, N_FEAT = 96, 9, 10
CONDS = ["16_e", "22_e", "30_e", "16_l", "22_l", "30_l"]
temp = np.array([16, 22, 30, 16, 22, 30], dtype=float)
temp_z = (temp - temp.mean()) / temp.std()

def simulate(interaction_scale, structured, env_scale=3.0, noise=0.3):
    """structured=True -> strain-specific slope in temperature (learnable)
       structured=False -> independent strain x condition deviations (not learnable)"""
    strain = rng.normal(0, 1, (N_STRAIN, N_FEAT))
    env = rng.normal(0, env_scale, (len(CONDS), N_FEAT))
    if structured:
        slope = rng.normal(0, 1, (N_STRAIN, N_FEAT))
        inter = slope[None] * temp_z[:, None, None]
    else:
        inter = rng.normal(0, 1, (len(CONDS), N_STRAIN, N_FEAT))
    data, groups = {}, np.repeat(np.arange(N_STRAIN), N_FOV)
    for c_i, c in enumerate(CONDS):
        vals = strain + env[c_i] + interaction_scale * inter[c_i]
        vals = np.repeat(vals, N_FOV, axis=0) + rng.normal(0, noise, (N_STRAIN*N_FOV, N_FEAT))
        data[c] = vals
    return data, groups

def run(label, **kw):
    data, groups = simulate(**kw)
    res = compare(data, exp_conditions=CONDS, ridge_alpha=10, n_splits=8, groups=groups)
    s = summarize_interaction_capture(res, [f"f{i}" for i in range(N_FEAT)])
    print(f"{label:<44} ridge {s.r2_ridge.mean():+.3f} | recal {s.r2_strain_mean_recal.mean():+.3f} | "
          f"raw strain_mean {s.r2_strain_mean.mean():+.3f} | global {s.r2_global_mean.mean():+.3f} | "
          f"captured {s.interaction_captured.mean():+.3f}")

