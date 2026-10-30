"""
cvp_pub -- helper functions for the Chlorella vulgaris phenotyping pub
=======================================================================

Every function used by chlorella_phenotyping_pub_analyses.ipynb lives in this
one file (import it as `import cvp_pub as cvpp`). It is self-contained: it does
not import cvp.py, cvp_baseline_additions.py, cvp_departures.py,
cvp_diversity.py, or cvp_standardized.py.

Provenance
----------
Functions were copied verbatim from those five modules, together with every
helper they call. The only edits are mechanical:
  * cross-module calls (cvp.X, cba.X, _cvp_module().X, ...) now call the local
    function directly;
  * where cvp.py and cvp_baseline_additions.py both defined a function with the
    same name, the cvp_baseline_additions.py version (the one the notebook calls)
    keeps the public name and the cvp.py version is renamed `_cvp_<name>`. The
    cvp_baseline_additions.py versions are thin wrappers that run the cvp.py
    version and optionally add an interactive Plotly figure, so the math lives
    in `_cvp_<name>`;
  * segmentation code (arcadia_microscopy_tools) and module-level test code are
    not included.
One functional addition: the interactive heatmap wrappers take
`save_csv=True` / `csv_dir="results"` and write the plotted heatmap values to
<csv_dir>/<figtitle>_values.csv (see _save_heatmap_csv).
Each function is preceded by a `# source: <file>` comment. Functions at the
end of the file (WILD-TYPE COMPARISON, RIDGE ALPHA SCAN, SMALL NOTEBOOK
HELPERS) were moved out of notebook cells into functions; their math is
unchanged.

Data layout
-----------
A condition matrix has one row per (well, field of view) and one column per
feature. Rows are FoV-major: row = well + fov * n_wells, so a well's fields sit
n_wells rows apart.

Where to check the math for each pub result
-------------------------------------------
  Dead-well filter ..................... compute_dead_wells_and_ids
  Feature matrices (FoV means) ......... build_training_data, make_condition_matrix_finite,
                                         signed_log_transform_hu_features
  Strain x feature heatmap ............. plot_all_strain_feature_heatmap (-> _cvp_plot_all_strain_feature_heatmap)
  Mutants vs wild type ................. correct_plate_position_effects, get_strain_means,
                                         batch_correct_median, compute_wt_zscores_and_distances,
                                         plot_wt_distance
  Feature reliability (ICC) ............ compute_feature_icc, plot_feature_icc_heatmap
  t-SNE ................................ tsne_standardized, plot_tsne
  PCA trajectories ..................... plot_strain_pca_trajectories_standardized,
                                         standardize_training_data, plot_strain_pca_trajectories
  Model schematic ...................... plot_model_schematic
  Model performance A (R2 vs baseline) . compare_models_condition_holdout_cv_recalibrated,
                                         plot_ridge_vs_recalibrated_r2
  Model performance B (displacement) ... _oof_by_condition, zscored_oof_predictions,
                                         plot_prediction_displacements
  Model performance C (calibration) .... compute_prediction_calibration, plot_prediction_calibration
  Model performance D (R2 per feature) . plot_sorted_r2_per_feature
  Variance partitioning ................ compute_variance_partitioning_by_feature,
                                         plot_variance_partitioning_heatmap,
                                         merge_variance_and_ridge_results,
                                         plot_ridge_variance_component_correlations,
                                         plot_ridge_vs_variance_component
  Strain departures (37 vs 94) ......... compute_strain_departures, plot_reaction_norm_vs_prediction,
                                         plot_strain_pair_cell_grid
  Supplemental ......................... ridge_alpha_scan, identify_strains_from_predictions,
                                         find_interaction_example_strains, plate_position_effects,
                                         cross_condition_gradient_consistency,
                                         compute_feature_covariation_matrix(_cell_level)

Figures are written to ./plots/ (relative to the working directory), mostly as
plots/chlorella_pub_fig_<figtitle>.{png,pdf,svg}; interactive versions as .html.
"""

# --- standard library ---
import copy
import inspect as _inspect
import json
import re as _re
import warnings

# --- numerics ---
import numpy as np
import pandas as pd
from scipy import linalg as LA
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import norm, zscore
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --- machine learning ---
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- data access ---
import zarr

# --- plotting ---
import matplotlib.colors as _mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import arcadia_pycolor as apc


# ==============================================================================================================
# CORE: data access, feature matrices, ICC, t-SNE, variance partitioning, feature covariation
# (from cvp.py)
# ==============================================================================================================


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
def remove_rows_cols(array, row_indices, col_indices):
    """
    Removes specified rows and columns from a 2D numpy array.
    """
    # Remove rows (axis 0)
    temp_array = np.delete(array, row_indices, axis=0)
    # Remove columns (axis 1)
    final_array = np.delete(temp_array, col_indices, axis=1)
    return final_array


# source: cvp.py
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


# source: cvp.py
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


# standalone colorbar export (same style as the t-SNE colorbars)
def _save_standalone_cbar(mesh, figtitle, cbar_label=None):
    """Save a horizontal colorbar matching `mesh` (cmap + vmin/vmax) as its own
    svg/pdf/png: plots/chlorella_pub_fig_<figtitle>_cbar.*"""
    cfig, cax = plt.subplots(figsize=(3.5, 0.25))
    cb = cfig.colorbar(plt.cm.ScalarMappable(norm=mesh.norm, cmap=mesh.cmap),
                       cax=cax, orientation="horizontal")
    cb.set_label(cbar_label or "")
    cb.ax.xaxis.set_label_position("top")
    base = 'plots/chlorella_pub_fig_' + figtitle + '_cbar'
    for ext in ("svg", "pdf", "png"):
        cfig.savefig(base + '.' + ext, bbox_inches='tight')
    plt.close(cfig)


# source: cvp.py  (originally `plot_sorted_r2_per_feature`)
def _cvp_plot_sorted_r2_per_feature(
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
    save_cbar_separately=False,
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
    save_cbar_separately : bool
        If True, also save a standalone horizontal colorbar to
        plots/chlorella_pub_fig_<figtitle>_cbar.{svg,pdf,png}.

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

    # Standalone colorbar file
    if save_cbar_separately:
        _save_standalone_cbar(hm.collections[0], figtitle, cbar_label)

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


# source: cvp.py
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


# source: cvp.py  (originally `plot_variance_partitioning_heatmap`)
def _cvp_plot_variance_partitioning_heatmap(
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py  (originally `plot_all_strain_feature_heatmap`)
def _cvp_plot_all_strain_feature_heatmap(
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py
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


# source: cvp.py  (originally `plot_feature_icc_heatmap`)
def _cvp_plot_feature_icc_heatmap(
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
    save_cbar_separately=False,  # also save a standalone colorbar file
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
    # --- Standalone colorbar file ---
    if save_cbar_separately:
        _save_standalone_cbar(hm.collections[0], figtitle, cbar_label)
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


# ==============================================================================================================
# MODELING: grouped ridge CV, recalibrated baseline, displacement, calibration, strain identity,
# reaction norms, plate-position (batch) effects, interactive heatmaps (from cvp_baseline_additions.py)
# ==============================================================================================================


# source: cvp_baseline_additions.py
def _well_rows(w, n_wells, n_fovs):
    """Row indices of all fields of view for well `w`."""
    return int(w) + np.arange(n_fovs) * n_wells


# source: cvp_baseline_additions.py
def well_groups(n_wells=96, n_fovs=9):
    """Well id for every row, for GroupKFold: np.tile(arange(n_wells), n_fovs)."""
    return np.tile(np.arange(n_wells), n_fovs)


# source: cvp_baseline_additions.py
def _fit_affine(x, y):
    """
    Least-squares fit of y = a + b*x for one feature.

    Returns (a, b). Degenerate cases (constant x, too few finite points) fall
    back to the intercept-only model, which is the global-mean baseline.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return (np.nanmean(y[mask]) if mask.any() else 0.0), 0.0

    xm, ym = x[mask], y[mask]
    var_x = np.var(xm)
    if not np.isfinite(var_x) or var_x <= 0:
        return ym.mean(), 0.0

    b = np.cov(xm, ym, bias=True)[0, 1] / var_x
    a = ym.mean() - b * xm.mean()
    return a, b


# source: cvp_baseline_additions.py
def _recalibrated_strain_mean_prediction(rowmean_train, y_train, rowmean_test):
    """
    Fit one affine map per feature on the training fold, apply to the test fold.

    rowmean_* : (n_rows, n_features) strain means across the training conditions
    y_train   : (n_train_rows, n_features) values in the held-out condition
    """
    n_features = y_train.shape[1]
    y_pred = np.empty((rowmean_test.shape[0], n_features), dtype=float)

    for j in range(n_features):
        a, b = _fit_affine(rowmean_train[:, j], y_train[:, j])
        y_pred[:, j] = a + b * rowmean_test[:, j]

    return y_pred


# source: cvp_baseline_additions.py
def compare_models_condition_holdout_cv_recalibrated(
    training_data,
    exp_conditions=None,
    ridge_alpha=3000,
    n_splits=16,
    random_state=42,
    groups=None,
    profile_alpha="match_per_predictor",
):
    """
    Leave-one-condition-out comparison of ridge against four baselines.

    Models returned
    ---------------
    ridge                 : full model, strain's profile across training conditions
    global_mean           : held-out condition mean (no strain information)
    strain_mean           : raw within-strain mean, uncalibrated (the existing one)
    strain_mean_recal     : within-strain mean, affinely recalibrated per feature
    ridge_strain_profile  : ridge on the CONDITION-AVERAGED strain profile

    `ridge_strain_profile` is the control that isolates interaction. It gets the
    same multivariate information as the full model -- every feature, pooled
    across features to denoise each strain's level -- but its input is averaged
    over the training conditions, so it cannot see how a strain MOVED between
    them. It therefore cannot express a condition-specific response.

        ridge - strain_mean_recal    : includes multivariate denoising
        ridge - ridge_strain_profile : interaction only, capacity held constant

    The second is the number to report.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        condition -> (n_rows, n_features). Usually training_data_finite.
    exp_conditions : list[str] or None
    ridge_alpha : float
        Must match the alpha reported in the methods section.
    n_splits : int
    random_state : int
        Used only for ungrouped KFold; GroupKFold is deterministic.
    groups : array-like of shape (n_rows,) or None
        Well index per row. If given, GroupKFold is used so no field of a
        held-out well appears in training.
    profile_alpha : "match_per_predictor", None, or float
        Penalty for `ridge_strain_profile`. The full model sees n_train_conditions
        copies of each feature, so it can reach a given prediction with a smaller
        L2 penalty than the profile model can -- reusing the same alpha would
        handicap the control and inflate the apparent interaction.
        "match_per_predictor" (default) uses ridge_alpha / n_train_conditions,
        which equalizes shrinkage. None reuses ridge_alpha unchanged.

    Returns
    -------
    results : dict
        results[model]["overall_r2"]      : (n_conditions,)
        results[model]["rmse"]            : (n_conditions,)
        results[model]["r2_per_feature"]  : (n_conditions, n_features)
    """
    if exp_conditions is None:
        exp_conditions = list(training_data.keys())

    model_names = [
        "ridge", "global_mean", "strain_mean", "strain_mean_recal",
        "ridge_strain_profile",
    ]

    n_train_conditions = len(exp_conditions) - 1
    if profile_alpha is None:
        profile_alpha_eff = ridge_alpha
    elif profile_alpha == "match_per_predictor":
        profile_alpha_eff = ridge_alpha / n_train_conditions
    else:
        profile_alpha_eff = float(profile_alpha)

    n_conditions = len(exp_conditions)
    n_features = training_data[exp_conditions[0]].shape[1]

    results = {
        m: {
            "conditions": exp_conditions,
            "overall_r2": np.zeros(n_conditions),
            "rmse": np.zeros(n_conditions),
            "r2_per_feature": np.zeros((n_conditions, n_features)),
        }
        for m in model_names
    }

    if groups is not None:
        groups = np.asarray(groups)
        splitter = GroupKFold(n_splits=n_splits)
        split_args = lambda X: (X, None, groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_args = lambda X: (X,)

    for i, test_condition in enumerate(exp_conditions):
        train_conditions = [c for c in exp_conditions if c != test_condition]

        X_full = np.hstack([training_data[c] for c in train_conditions])
        y_full = np.asarray(training_data[test_condition], dtype=float)

        # each strain's mean phenotype across the training conditions
        rowmean_full = np.nanmean(
            np.stack([training_data[c] for c in train_conditions], axis=0), axis=0
        )

        collected = {m: [] for m in model_names}
        y_true_all = []

        for train_idx, test_idx in splitter.split(*split_args(X_full)):
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y_full[train_idx], y_full[test_idx]

            y_true_all.append(y_test)

            # ---- ridge ----
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha)),
            ])
            pipe.fit(X_train, y_train)
            collected["ridge"].append(pipe.predict(X_test))

            # ---- global mean of the held-out condition ----
            collected["global_mean"].append(
                np.tile(np.nanmean(y_train, axis=0), (len(test_idx), 1))
            )

            # ---- raw within-strain mean (existing baseline) ----
            collected["strain_mean"].append(rowmean_full[test_idx])

            # ---- recalibrated within-strain mean ----
            collected["strain_mean_recal"].append(
                _recalibrated_strain_mean_prediction(
                    rowmean_full[train_idx], y_train, rowmean_full[test_idx]
                )
            )

            # ---- ridge on the condition-averaged strain profile ----
            # same features and same model class as the full ridge, but the
            # input carries no condition-resolved structure
            profile_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=profile_alpha_eff)),
            ])
            profile_pipe.fit(rowmean_full[train_idx], y_train)
            collected["ridge_strain_profile"].append(
                profile_pipe.predict(rowmean_full[test_idx])
            )

        y_true_all = np.vstack(y_true_all)

        for m in model_names:
            y_pred_all = np.vstack(collected[m])
            results[m]["overall_r2"][i] = r2_score(y_true_all, y_pred_all)
            results[m]["rmse"][i] = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
            results[m]["r2_per_feature"][i, :] = [
                r2_score(y_true_all[:, j], y_pred_all[:, j]) for j in range(n_features)
            ]

    return results


# source: cvp_baseline_additions.py
def _matrix_glyph(ax, x, y, w, h, color, n_rows=5, n_cols=4,
                  edge="black", lw=0.9, fill_alpha=0.30, zorder=2):
    """A small rectangle with grid lines, standing in for a feature matrix."""
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge,
                               linewidth=lw, alpha=fill_alpha, zorder=zorder))
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="none", edgecolor=edge,
                               linewidth=lw, zorder=zorder + 2))
    for i in range(1, n_rows):
        ax.plot([x, x + w], [y + h * i / n_rows] * 2, color=edge,
                lw=0.3, alpha=0.45, zorder=zorder + 1)
    for j in range(1, n_cols):
        ax.plot([x + w * j / n_cols] * 2, [y, y + h], color=edge,
                lw=0.3, alpha=0.45, zorder=zorder + 1)


# source: cvp_baseline_additions.py
def _arrow(ax, x0, y0, x1, y1, color="black", lw=1.2, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0))


# source: cvp_baseline_additions.py
def plot_model_schematic(
    condition_labels=("16 °C e.stat", "22 °C e.stat", "30 °C e.stat",
                      "16 °C l.stat", "22 °C l.stat", "30 °C l.stat"),
    condition_colors=None,
    held_out_index=0,
    n_rows_label="96 strains × 9 FoV",
    n_features=36,
    ridge_alpha=500,
    held_out_color=None,
    model_color=None,
    figsize=(11, 5.4),
    font_size=9,
    label_size=8,
    save=False,
    figtitle="model_schematic",
):
    """
    Schematic of the leave-one-condition-out ridge design (Figure 1A).

    Five conditions are concatenated into the predictor matrix; the sixth is the
    target. The strip along the bottom shows that this repeats six times, each
    condition held out once. Colors default to the six condition colors used
    throughout the pub.
    """
    if condition_colors is None:
        condition_colors = [apc.vital, apc.wish, apc.dress,
                            apc.lapis, apc.aster, apc.candy]
    if held_out_color is None:
        held_out_color = apc.gray if hasattr(apc, "gray") else "#9AA0A6"
    if model_color is None:
        model_color = apc.aster

    n_cond = len(condition_labels)
    train_idx = [i for i in range(n_cond) if i != held_out_index]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    # ======================= top row: the model ===========================
    gw, gh, gap = 5.4, 14.0, 1.1
    x0, ytop = 4.0, 30.0
    for k, ci in enumerate(train_idx):
        gx = x0 + k * (gw + gap)
        _matrix_glyph(ax, gx, ytop, gw, gh, condition_colors[ci])
        ax.text(gx + gw / 2, ytop + gh + 0.8, condition_labels[ci],
                ha="left", va="bottom", fontsize=label_size, rotation=30,
                rotation_mode="anchor")

    span = len(train_idx) * gw + (len(train_idx) - 1) * gap

    # brace under the predictor block
    ax.plot([x0, x0 + span], [ytop - 1.5] * 2, color="black", lw=0.9)
    for xx in (x0, x0 + span):
        ax.plot([xx, xx], [ytop - 1.5, ytop - 0.8], color="black", lw=0.9)
    ax.text(x0 + span / 2, ytop - 2.6,
            f"{n_rows_label}  ×  {len(train_idx)} conditions × {n_features} features",
            ha="center", va="top", fontsize=label_size)

    ymid = ytop + gh / 2

    # ridge box
    bx, bw, bh = 41.4, 16.0, 8.0
    by = ymid - bh / 2
    _arrow(ax, x0 + span + 1.2, ymid, bx - 0.6, ymid)
    ax.add_patch(plt.Rectangle((bx, by), bw, bh, facecolor=model_color,
                               edgecolor="black", lw=1.1, alpha=0.30, zorder=2))
    ax.text(bx + bw / 2, by + bh / 2 + 1.2, "ridge regression",
            ha="center", va="center", fontsize=font_size, zorder=4)
    ax.text(bx + bw / 2, by + bh / 2 - 1.9, f"$\\alpha$ = {ridge_alpha:,}",
            ha="center", va="center", fontsize=label_size, zorder=4)

    # predicted vs observed
    px, pw, ph = 62.4, gw, 7.0
    _arrow(ax, bx + bw + 0.6, ymid, px - 0.8, ymid + 2.0 + ph / 2)

    _matrix_glyph(ax, px, ymid + 2.0, pw, ph, model_color)
    ax.text(px + pw / 2, ymid + 2.0 + ph + 0.7, "predicted",
            ha="center", va="bottom", fontsize=label_size)

    _matrix_glyph(ax, px, ymid - 2.0 - ph, pw, ph,
                  condition_colors[held_out_index])
    ax.text(px + pw / 2, ymid - 2.0 - ph - 0.7,
            f"observed\n{condition_labels[held_out_index]}",
            ha="center", va="top", fontsize=label_size)

    rx = px + pw + 2.2
    ax.annotate("", xy=(rx, ymid - 2.0 - ph + 0.5),
                xytext=(rx, ymid + 2.0 + ph - 0.5),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    ax.text(rx + 1.2, ymid, "$R^2$", ha="left", va="center", fontsize=font_size)

    # ==================== bottom row: the rotation ========================
    sx, cw, ch, sgap = 4.0, 3.4, 2.0, 0.5
    ax.text(sx, 23.5, "repeated six times, each condition held out once",
            ha="left", va="bottom", fontsize=label_size)
    for r in range(n_cond):
        yy = 21.0 - r * (ch + sgap)
        for c in range(n_cond):
            xx = sx + c * (cw + sgap)
            is_out = (c == r)
            ax.add_patch(plt.Rectangle(
                (xx, yy), cw, ch,
                facecolor="white" if is_out else condition_colors[c],
                edgecolor="black", lw=0.7,
                alpha=1.0 if is_out else 0.45, zorder=2))
        ax.text(sx + n_cond * (cw + sgap) + 0.6, yy + ch / 2,
                f"predict {condition_labels[r]}",
                ha="left", va="center", fontsize=label_size - 0.5)

    ax.text(sx, 21.0 - (n_cond - 1) * (ch + sgap) - 1.2,
            "white = held out   |   filled = used as predictors",
            ha="left", va="top", fontsize=label_size - 1, color="#555555")

    # note block, filling the lower right
    ax.text(58.0, 21.5,
            "Rows are well-field pairs: 9 fields of view per well.\n"
            "Cross-validation folds are grouped by well, so no field\n"
            "of a held-out well appears in training.",
            ha="left", va="top", fontsize=label_size - 0.5, color="#555555",
            linespacing=1.6)

    plt.tight_layout()
    plt.show()

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)

    return fig, ax


# source: cvp_baseline_additions.py
def _strain_means_by_condition(training_data, conditions, feat_idx,
                               n_wells=96, n_fovs=9, dead_ids=None,
                               return_sem=False):
    """(n_conditions, n_wells) strain means, z-scored across strains within
    each condition. Z-scoring removes the environment main effect, so a
    difference between two strains that CHANGES across conditions is
    interaction, not a constant offset."""
    dead_wells = sorted({int(d) % n_wells for d in (dead_ids or [])})
    z = np.full((len(conditions), n_wells), np.nan)
    sem = np.full((len(conditions), n_wells), np.nan)

    for ci, cond in enumerate(conditions):
        mat = np.asarray(training_data[cond], dtype=float)[:, feat_idx]
        blocks = [mat[_well_rows(w, n_wells, n_fovs)] for w in range(n_wells)]
        per_strain = np.array([np.nanmean(b) for b in blocks])
        # standard error of each strain mean, from field-to-field spread
        per_sem = np.array([
            np.nanstd(b, ddof=1) / np.sqrt(max(np.isfinite(b).sum(), 1))
            for b in blocks
        ])
        per_strain[dead_wells] = np.nan
        mu, sd = np.nanmean(per_strain), np.nanstd(per_strain)
        sd = sd if sd > 0 else 1.0
        z[ci] = (per_strain - mu) / sd
        sem[ci] = per_sem / sd          # same units as z

    return (z, sem) if return_sem else z


# source: cvp_baseline_additions.py
def find_interaction_example_strains(
    training_data,
    training_feature_labels,
    feature,
    conditions,
    n_wells=96,
    n_fovs=9,
    dead_ids=None,
    apart_min=1.2,
    together_max=0.3,
    min_snr=3.0,
    top_n=10,
):
    """
    Find strain pairs that make the interaction visible by eye.

    Two kinds of pair are returned, labelled in the `kind` column:

    "interaction"  : indistinguishable in at least one condition and clearly
                     separated in another. Their difference is condition-
                     specific, which is the strain x environment term.
                     Filtered by `min_snr`: the CHANGE in the gap must exceed
                     the field-to-field noise on both strain means by that
                     factor. Without this guard, screening every pair (~4,200
                     of them for 92 strains) reliably turns up noise that
                     looks like a striking interaction.
    "strain"       : separated by a similar amount in EVERY condition. Their
                     difference is a constant offset, which is the strain
                     main effect.

    Showing both side by side is what makes the point: the reader sees that
    "these two strains differ" and "these two strains differ *here*" are not
    the same statement.

    Values are z-scored across strains within each condition, so the units are
    between-strain standard deviations and the environment main effect is
    already removed.
    """
    labels = list(training_feature_labels)
    if feature in labels:
        feat_idx = labels.index(feature)
    else:
        hits = [i for i, l in enumerate(labels) if feature in l]
        if len(hits) != 1:
            raise ValueError(
                f"'{feature}' matched {len(hits)} features; be more specific. "
                f"Candidates: {[labels[i] for i in hits][:8]}"
            )
        feat_idx = hits[0]

    z, sem = _strain_means_by_condition(training_data, conditions, feat_idx,
                                        n_wells=n_wells, n_fovs=n_fovs,
                                        dead_ids=dead_ids, return_sem=True)
    alive = [w for w in range(n_wells) if np.all(np.isfinite(z[:, w]))]

    rows = []
    for a_i, a in enumerate(alive):
        for b in alive[a_i + 1:]:
            d = z[:, a] - z[:, b]
            ad = np.abs(d)
            c_near, c_far = int(np.argmin(ad)), int(np.argmax(ad))
            # noise on the CHANGE in gap between those two conditions
            noise = np.sqrt(
                sem[c_near, a] ** 2 + sem[c_near, b] ** 2
                + sem[c_far, a] ** 2 + sem[c_far, b] ** 2
            )
            snr = (ad[c_far] - ad[c_near]) / noise if noise > 0 else np.inf
            entry = {
                "gap_change_snr": snr,
                "strain_a": a, "strain_b": b,
                "min_abs_gap": ad.min(), "max_abs_gap": ad.max(),
                "mean_gap": np.abs(d.mean()), "gap_sd": d.std(),
                "closest_condition": conditions[int(np.argmin(ad))],
                "farthest_condition": conditions[int(np.argmax(ad))],
            }
            for ci, cond in enumerate(conditions):
                entry[f"gap_{cond}"] = d[ci]
            rows.append(entry)

    df = pd.DataFrame(rows)

    inter = df[(df["min_abs_gap"] <= together_max) &
               (df["max_abs_gap"] >= apart_min) &
               (df["gap_change_snr"] >= min_snr)].copy()
    inter["score"] = inter["max_abs_gap"] - inter["min_abs_gap"]
    inter["kind"] = "interaction"

    main = df[df["gap_sd"] <= together_max].copy()
    main["score"] = main["mean_gap"]
    main["kind"] = "strain"

    out = pd.concat([
        inter.sort_values("score", ascending=False).head(top_n),
        main.sort_values("score", ascending=False).head(top_n),
    ], ignore_index=True)

    n_inter = int((out["kind"] == "interaction").sum())
    n_main = int((out["kind"] == "strain").sum())

    if out.empty:
        raise ValueError(
            "No qualifying pairs. Loosen apart_min / together_max / min_snr, "
            "or try a feature with a larger interaction fraction."
        )

    # a silently under-populated category is the usual cause of an IndexError
    # downstream, so say so here instead
    for label, n, hint in [
        ("interaction", n_inter,
         "raise together_max, lower apart_min, or lower min_snr"),
        ("strain", n_main, "raise together_max"),
    ]:
        if n == 0:
            print(f"WARNING: no '{label}' pairs passed the filters — {hint}.")
        elif n < 3:
            print(f"NOTE: only {n} '{label}' pair(s) passed the filters — "
                  f"{hint} if you want more to choose from.")

    return out


# source: cvp_baseline_additions.py
def _representative_cell_crops(
    zroot, fov, well, channel, n_cells, crop, rng, min_area=30,
    verbose=True, label=None,
):
    """
    Square crops centred on cells of typical size for that well.

    Two things here used to produce blank or part-black panels:

    1. Candidates were chosen FIRST and edge-clipped ones discarded AFTER, so a
       cell whose window ran off the image was lost rather than replaced, and
       the tiler padded the gap with black. Now the window is shifted inward so
       every cell yields a full crop, and candidates are drawn until n_cells
       are collected.
    2. The old clamp was asymmetric -- cells near the top or left produced a
       valid (shifted) window, cells near the bottom or right were dropped.

    A genuinely empty panel now means the well really has no segmented objects
    above `min_area`: usually a strain that never grew, or a channel that was
    not segmented for that experiment. It says so instead of failing quietly.
    """
    from skimage.measure import regionprops

    label_img = np.asarray(zroot.get_label_image(fov, well, channel))
    intensity = np.asarray(zroot.get_processed_intensities(fov, well, channel))
    H, W = intensity.shape[-2:]

    if crop > H or crop > W:
        raise ValueError(
            f"crop={crop} is larger than the {H}x{W} image; reduce crop."
        )

    props = [p for p in regionprops(label_img) if p.area >= min_area]
    if not props:
        if verbose:
            print(f"WARNING: no segmented objects >= {min_area} px for "
                  f"well {well}, fov {fov}, channel {channel}"
                  + (f" ({label})" if label else "")
                  + " -- dead well, or this channel was not segmented.")
        return []

    areas = np.array([p.area for p in props])
    # cells closest to this well's median size: typical, not cherry-picked
    order = np.argsort(np.abs(areas - np.median(areas)))
    candidates = [props[i] for i in order]
    rng.shuffle(candidates)

    half = crop // 2
    crops = []
    for p in candidates:
        if len(crops) >= n_cells:
            break
        cy, cx = int(round(p.centroid[0])), int(round(p.centroid[1]))
        # shift the window inward instead of dropping the cell
        y0 = int(np.clip(cy - half, 0, H - crop))
        x0 = int(np.clip(cx - half, 0, W - crop))
        crops.append(intensity[y0:y0 + crop, x0:x0 + crop])

    if verbose and len(crops) < n_cells:
        print(f"NOTE: only {len(crops)} of {n_cells} cells available for "
              f"well {well}, fov {fov}, channel {channel}"
              + (f" ({label})" if label else "")
              + " -- the panel is padded; try another fov or a smaller crop.")

    return crops


# source: cvp_baseline_additions.py
def plot_strain_pair_reaction_norm(
    training_data,
    training_feature_labels,
    feature,
    conditions,
    strain_a,
    strain_b,
    condition_labels=None,
    n_wells=96,
    n_fovs=9,
    dead_ids=None,
    strain_colors=None,
    show_population=True,
    show_errorbars=True,
    annotate_gaps="extremes",     # "extremes", "all", or None
    shade_gap=True,
    figsize=(4.6, 4.0),
    font_size=10,
    title=None,
    ylabel=None,
    save=False,
    figtitle="strain_pair_reaction_norm",
):
    """
    Reaction norm for one strain pair -- no images, one figure per call.

    Call it once for the pair whose gap CHANGES across conditions (interaction)
    and once for the pair whose gap HOLDS (strain identity). Non-parallel lines
    versus parallel lines is the whole argument, and keeping them as separate
    figures lets each carry its own caption.

    Values are z-scored within each condition, so the environment main effect
    is removed and the y-axis is a strain's position in the population for that
    condition, in between-strain SD. Error bars are +/-1 SEM across fields of
    view, which is what makes "indistinguishable" a claim rather than an
    impression: if the bars overlap, the strains are not resolved there.
    """
    conds = list(conditions)
    n_cond = len(conds)
    if condition_labels is None:
        condition_labels = conds
    if strain_colors is None:
        strain_colors = [apc.aster, apc.dress]

    labels = list(training_feature_labels)
    if feature in labels:
        feat_idx = labels.index(feature)
    else:
        hits = [i for i, l in enumerate(labels) if feature in l]
        if len(hits) != 1:
            raise ValueError(
                f"'{feature}' matched {len(hits)} features; pass the exact name "
                f"used for the pair selection."
            )
        feat_idx = hits[0]

    z, sem = _strain_means_by_condition(
        training_data, conds, feat_idx, n_wells=n_wells, n_fovs=n_fovs,
        dead_ids=dead_ids, return_sem=True,
    )

    xs = np.arange(n_cond)
    za, zb = z[:, strain_a], z[:, strain_b]
    ea, eb = sem[:, strain_a], sem[:, strain_b]
    gaps = np.abs(za - zb)

    fig, ax = plt.subplots(figsize=figsize)

    # every other strain, recessive, for context on what 1 SD means here
    if show_population:
        for w in range(n_wells):
            if w in (strain_a, strain_b) or not np.all(np.isfinite(z[:, w])):
                continue
            ax.plot(xs, z[:, w], "-", color="#B8B8B8", lw=0.6, alpha=0.35,
                    zorder=0)

    if shade_gap:
        ax.fill_between(xs, za, zb, color="#9AA0A6", alpha=0.18, lw=0, zorder=1)

    for (strain, vals, errs, color) in [
        (strain_a, za, ea, strain_colors[0]),
        (strain_b, zb, eb, strain_colors[1]),
    ]:
        if show_errorbars:
            ax.errorbar(xs, vals, yerr=errs, fmt="o-", color=color, lw=2,
                        ms=8, capsize=3, elinewidth=1.2,
                        markeredgecolor="white", markeredgewidth=1.2,
                        label=f"strain {strain}", zorder=3)
        else:
            ax.plot(xs, vals, "o-", color=color, lw=2, ms=8,
                    markeredgecolor="white", markeredgewidth=1.2,
                    label=f"strain {strain}", zorder=3)

    # label the gap only where it carries the argument
    if annotate_gaps == "all":
        to_label = list(xs)
    elif annotate_gaps == "extremes":
        to_label = sorted({int(np.argmin(gaps)), int(np.argmax(gaps))})
    else:
        to_label = []

    for i in to_label:
        lo, hi = sorted((za[i], zb[i]))
        ax.annotate("", xy=(i, lo), xytext=(i, hi),
                    arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0))
        if (hi - lo) < 0.4:
            # too tight to sit between the lines without covering them
            anchor, offset, ha, va = (i, hi), (0, 9), "center", "bottom"
        elif i == n_cond - 1:
            # last column: label inward so it does not clip the axes
            anchor, offset, ha, va = (i, (lo + hi) / 2), (-7, 0), "right", "center"
        else:
            anchor, offset, ha, va = (i, (lo + hi) / 2), (7, 0), "left", "center"
        ax.annotate(f"{gaps[i]:.2f} SD", xy=anchor, xytext=offset,
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=font_size - 2, color="#555555")

    ax.set_xticks(xs)
    ax.set_xticklabels(condition_labels, fontsize=font_size - 1)
    ax.set_ylabel(ylabel or f"{feature}\nz-scored within each condition",
                  fontsize=font_size - 1)
    ax.set_xlim(-0.45, n_cond - 0.55)
    ax.axhline(0, color="#CCCCCC", lw=0.8, ls="--", zorder=0)
    ax.set_title(title or f"gap: {gaps.min():.2f} to {gaps.max():.2f} SD",
                 fontsize=font_size)
    ax.legend(fontsize=font_size - 1, frameon=False, loc="best")
    ax.tick_params(labelsize=font_size - 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.show()

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)

    return fig, ax


# source: cvp_baseline_additions.py
def plot_strain_pair_cell_grid(
    zroots,
    strain_a,
    strain_b,
    experiment_indices,
    condition_labels,
    channel="CY5",
    fov=0,
    n_cells=6,
    crop=96,
    cells_per_row=None,
    cmap="gray",
    strain_colors=None,
    row_labels=None,
    row_label_position="left",     # "left" or "inside"
    shared_intensity=True,
    intensity_percentiles=(1.0, 99.5),
    um_per_pixel=None,
    scalebar_um=10,
    random_state=0,
    panel_size=1.7,
    font_size=9,
    save=False,
    figtitle="strain_pair_cell_grid",
):
    """
    Cell images only: two strains down the rows, conditions across the columns.

    Companion to plot_strain_pair_reaction_norm -- keep the quantitative panel
    as its own figure and compose them side by side at layout time, so no axis
    label can collide with an image panel.

    All panels share one intensity range by default (`shared_intensity`).
    Without that, imshow autoscales each panel independently and apparent
    brightness differences between conditions are a rendering artifact rather
    than data.
    """
    rng = np.random.default_rng(random_state)
    n_cond = len(experiment_indices)
    if len(condition_labels) != n_cond:
        raise ValueError(
            f"condition_labels has {len(condition_labels)} entries but "
            f"experiment_indices has {n_cond}."
        )
    if strain_colors is None:
        strain_colors = [apc.aster, apc.dress]
    strains = [strain_a, strain_b]
    if row_labels is None:
        row_labels = [f"strain {s}" for s in strains]

    # tile the cells into a compact block instead of one long strip, so the
    # panel is not 5:1 and the figure has no dead vertical space
    ncols = cells_per_row or int(np.ceil(np.sqrt(n_cells)))
    nrows = int(np.ceil(n_cells / ncols))

    def _tile(crops):
        if not crops:
            return None
        pad = [np.zeros_like(crops[0])] * (nrows * ncols - len(crops))
        cells = (crops + pad)[:nrows * ncols]
        return np.vstack([
            np.hstack(cells[i * ncols:(i + 1) * ncols]) for i in range(nrows)
        ])

    # ---- gather every crop first, so the intensity range can be shared ----
    panels = {}
    for r, strain in enumerate(strains):
        for c, exp_i in enumerate(experiment_indices):
            crops = _representative_cell_crops(
                zroots[exp_i], fov, strain, channel, n_cells, crop, rng,
                label=f"strain {strain}, {condition_labels[c]}",
            )
            panels[(r, c)] = _tile(crops)

    vmin = vmax = None
    if shared_intensity:
        allvals = np.concatenate([
            p.ravel() for p in panels.values() if p is not None
        ]) if any(p is not None for p in panels.values()) else None
        if allvals is not None and allvals.size:
            vmin, vmax = np.percentile(allvals, intensity_percentiles)

    panel_h = panel_size
    panel_w = panel_size * ncols / nrows          # match the tile's aspect
    fig, axes = plt.subplots(
        2, n_cond,
        figsize=(panel_w * n_cond + 0.9, panel_h * 2 + 0.7),
        squeeze=False,
    )

    for r in range(2):
        for c in range(n_cond):
            ax = axes[r][c]
            strip = panels[(r, c)]
            if strip is not None:
                ax.imshow(strip, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax.text(0.5, 0.5, "no cells", ha="center", va="center",
                        transform=ax.transAxes, fontsize=font_size - 1,
                        color="#999999")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(strain_colors[r]); sp.set_linewidth(2.0)

            if r == 0:
                ax.set_title(condition_labels[c], fontsize=font_size)

            if c == 0:
                if row_label_position == "left":
                    ax.set_ylabel(row_labels[r], fontsize=font_size,
                                  color=strain_colors[r], labelpad=6)
                else:
                    ax.text(0.02, 0.96, row_labels[r], transform=ax.transAxes,
                            ha="left", va="top", fontsize=font_size - 1,
                            color="white",
                            bbox=dict(facecolor=strain_colors[r], edgecolor="none",
                                      boxstyle="round,pad=0.25", alpha=0.85))

            # scale bar in the last panel of the bottom row only
            if um_per_pixel and r == 1 and c == n_cond - 1 and strip is not None:
                bar_px = scalebar_um / um_per_pixel
                h, w = strip.shape[:2]
                y = h * 0.92
                ax.plot([w - bar_px - w * 0.04, w - w * 0.04], [y, y],
                        color="white", lw=3, solid_capstyle="butt")
                ax.text(w - bar_px / 2 - w * 0.04, y - h * 0.06,
                        f"{scalebar_um} µm", color="white", ha="center",
                        va="bottom", fontsize=font_size - 2)

    plt.tight_layout()
    plt.show()

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)

    return fig, axes


# source: cvp_baseline_additions.py
def _out_of_fold_predictions(training_data, exp_conditions, test_condition,
                             ridge_alpha, n_splits, groups, random_state):
    """Cross-validated predictions for one held-out condition."""
    from sklearn.model_selection import KFold, GroupKFold

    train_conditions = [c for c in exp_conditions if c != test_condition]
    X = np.hstack([training_data[c] for c in train_conditions])
    y = np.asarray(training_data[test_condition], dtype=float)

    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, np.asarray(groups))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)
        splits = splitter.split(X)

    y_pred = np.empty_like(y)
    for train_idx, test_idx in splits:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("ridge", Ridge(alpha=ridge_alpha))])
        pipe.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = pipe.predict(X[test_idx])
    return y_pred


# source: cvp_baseline_additions.py
def prediction_displacements(
    data_for_pca_graphs,
    exp_conditions,
    n_components=2,
    normalize=True,
):
    """
    Per-row distance between observed and predicted position -- the length of
    each arrow in plot_true_vs_predicted_pca_by_condition.

    `data_for_pca_graphs` is that function's third return value: a list of
    [observed_as_pcs, predicted_as_pcs], one entry per condition.

    n_components : 2 matches the arrows you see (PC1-PC2 only, so it ignores
        error in every higher PC and is therefore a LOWER BOUND on the real
        displacement). None uses all components, i.e. the true distance in
        feature space.

    normalize : raw distances are close to uninterpretable here. `pca_on_np_array`
        only mean-centres, so a raw distance is in unstandardized feature units
        and is dominated by whichever feature has the largest variance -- cell
        count, in this dataset. Each condition also gets its own PCA fit, so raw
        lengths are not comparable between conditions.

        With normalize=True each distance is divided by the median distance from
        an observed point to the observed centroid IN THAT CONDITION. That
        denominator is exactly the error a "predict the population centroid"
        model would make, so the resulting number reads directly:

            1.0  = no better than predicting the population mean
            0.5  = half the error of that null
            > 1  = worse than the null

    Returns a long-form DataFrame: condition, row, displacement.
    """
    records = []
    for cond, (obs, pred) in zip(exp_conditions, data_for_pca_graphs):
        obs = np.asarray(obs, dtype=float)
        pred = np.asarray(pred, dtype=float)
        k = obs.shape[1] if n_components is None else min(n_components, obs.shape[1])

        d = np.linalg.norm(pred[:, :k] - obs[:, :k], axis=1)

        if normalize:
            centroid = obs[:, :k].mean(axis=0)
            null_error = np.median(np.linalg.norm(obs[:, :k] - centroid, axis=1))
            d = d / (null_error if null_error > 0 else 1.0)

        for i, val in enumerate(d):
            records.append({"condition": cond, "row": i, "displacement": val})

    return pd.DataFrame(records)


# source: cvp_baseline_additions.py
def plot_prediction_displacements(
    data_for_pca_graphs,
    exp_conditions,
    col_titles=None,
    condition_colors=None,
    n_components=2,
    normalize=True,
    show_points=True,
    point_alpha=0.25,
    point_size=9,
    jitter=0.08,
    random_state=0,
    figsize=(6.4, 4.2),
    font_size=10,
    save=False,
    figtitle="prediction_displacement_distributions",
):
    """
    Box plots of the observed-to-predicted displacement, one box per condition,
    with the individual rows overlaid.

    The dashed line at 1.0 (when normalize=True) is the null: a model that
    predicts the population centroid for every strain. A box sitting well below
    it is the summary claim the arrow figure makes qualitatively.
    """
    df = prediction_displacements(data_for_pca_graphs, exp_conditions,
                                  n_components=n_components, normalize=normalize)
    labels = list(col_titles) if col_titles is not None else list(exp_conditions)
    if condition_colors is None:
        condition_colors = [apc.aster] * len(exp_conditions)

    rng = np.random.default_rng(random_state)
    groups = [df.loc[df.condition == c, "displacement"].to_numpy()
              for c in exp_conditions]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(groups, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.6),
                    whiskerprops=dict(color="#666666", lw=1.0),
                    capprops=dict(color="#666666", lw=1.0))
    for patch, color in zip(bp["boxes"], condition_colors):
        patch.set_facecolor(color); patch.set_alpha(0.35)
        patch.set_edgecolor("#444444"); patch.set_linewidth(1.0)

    if show_points:
        for i, (vals, color) in enumerate(zip(groups, condition_colors), start=1):
            x = i + rng.uniform(-jitter, jitter, size=len(vals))
            ax.scatter(x, vals, s=point_size, color=color, alpha=point_alpha,
                       linewidth=0, zorder=3)

    if normalize:
        ax.axhline(1.0, color="#999999", ls="--", lw=1.0, zorder=0)
        ax.annotate("predict the population mean", xy=(0.99, 1.0),
                    xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=font_size - 2, color="#777777")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=font_size - 1, rotation=30, ha="right")
    ax.set_ylabel(
        ("displacement / null model error" if normalize
         else "observed-to-predicted distance (PC units)")
        + f"\n{'PC1–PC2' if n_components == 2 else 'all components'}",
        fontsize=font_size - 1,
    )
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=font_size - 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.show()

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)

    return fig, ax, df


# source: cvp_baseline_additions.py
def _oof_by_condition(training_data, exp_conditions, ridge_alpha, n_splits,
                      groups, random_state):
    """{condition: (observed, out-of-fold predicted)} at row level."""
    return {
        c: (np.asarray(training_data[c], dtype=float),
            _out_of_fold_predictions(training_data, exp_conditions, c,
                                     ridge_alpha, n_splits, groups, random_state))
        for c in exp_conditions
    }


# source: cvp_baseline_additions.py
def _to_strain_means(mat, n_wells, n_fovs, keep_wells):
    out = np.array([np.nanmean(mat[_well_rows(w, n_wells, n_fovs)], axis=0)
                    for w in range(n_wells)])
    return out[keep_wells]


# source: cvp_baseline_additions.py
def compute_prediction_calibration(
    training_data, exp_conditions, training_feature_labels,
    ridge_alpha=500, n_splits=16, groups=None, random_state=42,
    n_wells=96, n_fovs=9, dead_ids=None, aggregate_to_strain=True,
):
    """
    Per feature and condition, regress OBSERVED on PREDICTED: observed = a + b*predicted.

    b is the calibration slope and 1.0 is the target.

    Ridge shrinks coefficients by construction, so its predictions are usually
    under-dispersed -- the spread of predictions is narrower than the spread of
    reality. That shows up as b > 1, and it matters for screening: if
    predictions are compressed toward the middle, residuals grow with distance
    from the mean and phenotypically extreme strains get flagged as outliers
    for a reason that has nothing to do with biology. R^2 will not reveal this;
    only the slope will.

    Noise in the observed values inflates scatter but does not bias b, so the
    row-level and strain-level versions estimate the same quantity.
    """
    oof = _oof_by_condition(training_data, exp_conditions, ridge_alpha,
                            n_splits, groups, random_state)
    dead_wells = sorted({int(d) % n_wells for d in (dead_ids or [])})
    keep_wells = [w for w in range(n_wells) if w not in dead_wells]

    rows = []
    for cond, (obs, pred) in oof.items():
        if aggregate_to_strain:
            obs = _to_strain_means(obs, n_wells, n_fovs, keep_wells)
            pred = _to_strain_means(pred, n_wells, n_fovs, keep_wells)

        for j, name in enumerate(training_feature_labels):
            x, y = pred[:, j], obs[:, j]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3 or np.var(x[m]) <= 0:
                continue
            b = np.cov(x[m], y[m], bias=True)[0, 1] / np.var(x[m])
            a = y[m].mean() - b * x[m].mean()
            r = np.corrcoef(x[m], y[m])[0, 1]
            rows.append({"condition": cond, "feature": name,
                         "slope": b, "intercept": a, "pearson_r": r,
                         "pred_sd": x[m].std(), "obs_sd": y[m].std(),
                         "dispersion_ratio": x[m].std() / y[m].std()})
    return pd.DataFrame(rows)


# source: cvp_baseline_additions.py
def _box_by_condition(
    df, value_col, exp_conditions, col_titles, condition_colors,
    ylabel, reference=None, reference_label=None, null_col=None,
    ylim=None, figsize=(6.4, 4.2), font_size=10, jitter=0.09,
    point_alpha=0.45, point_size=14, random_state=0,
    save=False, figtitle="figure", title=None,
):
    """Shared box-plus-points panel: one box per condition, one point per feature."""
    rng = np.random.default_rng(random_state)
    labels = list(col_titles) if col_titles is not None else list(exp_conditions)
    if condition_colors is None:
        condition_colors = [apc.aster] * len(exp_conditions)

    groups_ = [df.loc[df.condition == c, value_col].dropna().to_numpy()
               for c in exp_conditions]

    fig, ax = plt.subplots(figsize=figsize)

    if null_col is not None:
        nulls = np.concatenate([df.loc[df.condition == c, null_col].dropna().to_numpy()
                                for c in exp_conditions])
        lo, hi = np.percentile(nulls, [2.5, 97.5])
        ax.axhspan(lo, hi, color="#BBBBBB", alpha=0.30, zorder=0)
        ax.annotate("shuffled-strain null (95%)", xy=(0.99, hi),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=font_size - 2, color="#777777")

    bp = ax.boxplot(groups_, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.6),
                    whiskerprops=dict(color="#666666", lw=1.0),
                    capprops=dict(color="#666666", lw=1.0))
    for patch, color in zip(bp["boxes"], condition_colors):
        patch.set_facecolor(color); patch.set_alpha(0.35)
        patch.set_edgecolor("#444444"); patch.set_linewidth(1.0)

    for i, (vals, color) in enumerate(zip(groups_, condition_colors), start=1):
        x = i + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(x, vals, s=point_size, color=color, alpha=point_alpha,
                   linewidth=0, zorder=3)

    if reference is not None:
        ax.axhline(reference, color="#999999", ls="--", lw=1.0, zorder=1)
        if reference_label:
            ax.annotate(reference_label, xy=(0.99, reference),
                        xycoords=("axes fraction", "data"), xytext=(0, 4),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=font_size - 2, color="#777777")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=font_size - 1, rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=font_size - 1)
    if ylim: ax.set_ylim(*ylim)
    if title: ax.set_title(title, fontsize=font_size)
    ax.tick_params(labelsize=font_size - 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    plt.tight_layout(); plt.show()
    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig, ax


# source: cvp_baseline_additions.py
def plot_prediction_calibration(calibration_df, exp_conditions, col_titles=None,
                                condition_colors=None, **kwargs):
    """One box per condition, one point per feature. The dashed line at 1.0 is
    perfect calibration; above it the model is under-dispersed."""
    kwargs.setdefault("figtitle", "prediction_calibration_slopes")
    return _box_by_condition(
        calibration_df, "slope", exp_conditions, col_titles, condition_colors,
        ylabel="calibration slope\n(observed on predicted)",
        reference=1.0, reference_label="perfect calibration", **kwargs)


# source: cvp_baseline_additions.py
def _zscore(mat, mu=None, sd=None):
    mu = mat.mean(axis=0) if mu is None else mu
    sd = mat.std(axis=0) if sd is None else sd
    sd = np.where(sd > 0, sd, 1.0)
    return (mat - mu) / sd, mu, sd


# source: cvp_baseline_additions.py
def _pairwise_distance(A, B, metric="euclidean"):
    """(n_a, n_b) distances between rows of A and rows of B."""
    if metric == "euclidean":
        return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    if metric == "correlation":
        Az = (A - A.mean(axis=1, keepdims=True))
        Bz = (B - B.mean(axis=1, keepdims=True))
        Az /= np.where(np.linalg.norm(Az, axis=1, keepdims=True) > 0,
                       np.linalg.norm(Az, axis=1, keepdims=True), 1)
        Bz /= np.where(np.linalg.norm(Bz, axis=1, keepdims=True) > 0,
                       np.linalg.norm(Bz, axis=1, keepdims=True), 1)
        return 1 - Az @ Bz.T
    raise ValueError("metric must be 'euclidean' or 'correlation'")


# source: cvp_baseline_additions.py
def identify_strains_from_predictions(
    training_data, exp_conditions,
    ridge_alpha=500, n_splits=16, groups=None, random_state=42,
    n_wells=96, n_fovs=9, dead_ids=None, metric="euclidean",
    probabilities=True,
):
    """
    Can the model's prediction re-identify the strain it belongs to?

    For each held-out condition the model produces, for every strain, a
    predicted phenotype built ONLY from that strain's behaviour in the other
    five conditions. Treat those predictions as templates and assign each
    strain's OBSERVED phenotype to the nearest template. Getting it right means
    the prediction is specific to that strain rather than generically
    plausible -- which is exactly what a screen needs, and a much more
    demanding test than R^2.

    Because the templates are out-of-fold and never see the held-out condition,
    there is no self-matching shortcut.

    PROBABILITIES. Every observed strain is scored against every template, so
    the natural output is a posterior over identities rather than a yes/no.
    Treating the observation as Gaussian around the template in z-scored
    feature space, with a uniform prior over strains:

        P(strain j | observation i)  proportional to  exp(-d_ij^2 / 2 sigma^2)

    sigma^2 is the per-feature residual variance, estimated LEAVE-ONE-OUT: for
    strain i it uses the observed-to-own-template distances of every OTHER
    strain, so the probability assigned to i never depends on how well i itself
    was predicted. Without that, sigma would be fit partly on the answer and
    the probabilities would be optimistic.

    `p_correct` is the probability mass the model puts on the true identity --
    the number to report instead of a binary hit rate. Chance is 1/n_strains.
    `margin` is p_correct minus the largest probability given to any other
    strain: positive means the true strain is the single best explanation,
    negative means some impostor beat it.

    Returns {condition: {"confusion", "probability", "p_correct", "margin",
             "mean_p_correct", "median_p_correct", "accuracy", "top5",
             "mean_rank", "median_rank", "chance", "strains", "ranks"}}.
    """
    oof = _oof_by_condition(training_data, exp_conditions, ridge_alpha,
                            n_splits, groups, random_state)
    dead = sorted({int(d) % n_wells for d in (dead_ids or [])})
    keep = np.array([w for w in range(n_wells) if w not in dead])
    n = len(keep)

    out = {}
    for cond, (obs, pred) in oof.items():
        O = _to_strain_means(obs, n_wells, n_fovs, keep)
        P = _to_strain_means(pred, n_wells, n_fovs, keep)
        Oz, mu, sd = _zscore(O)                 # scale set by the observed data
        Pz, _, _ = _zscore(P, mu, sd)

        D = _pairwise_distance(Oz, Pz, metric=metric)   # rows observed, cols templates
        assigned = D.argmin(axis=1)
        order = D.argsort(axis=1)
        ranks = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(n)])

        conf = np.zeros((n, n), dtype=int)
        for i, j in enumerate(assigned):
            conf[i, j] += 1

        entry = {
            "confusion": conf,
            "accuracy": float((assigned == np.arange(n)).mean()),
            "top5": float((ranks <= 5).mean()),
            "mean_rank": float(ranks.mean()),
            "median_rank": float(np.median(ranks)),
            "chance": 1.0 / n,
            "strains": keep,
            "ranks": ranks,
        }

        if probabilities:
            D2 = D ** 2 if metric == "euclidean" else D ** 2
            own = np.array([D2[i, i] for i in range(n)])
            n_feat = Oz.shape[1]
            # leave-one-out per-feature residual variance
            sigma2 = np.array([
                (own.sum() - own[i]) / max(n - 1, 1) / n_feat for i in range(n)
            ])
            sigma2 = np.where(sigma2 > 0, sigma2, np.nanmean(sigma2))

            logits = -D2 / (2 * sigma2[:, None])
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)

            p_correct = np.array([P[i, i] for i in range(n)])
            off = P.copy(); off[np.arange(n), np.arange(n)] = -np.inf
            margin = p_correct - off.max(axis=1)

            entry.update({
                "probability": P,
                "p_correct": p_correct,
                "margin": margin,
                "mean_p_correct": float(p_correct.mean()),
                "median_p_correct": float(np.median(p_correct)),
            })

        out[cond] = entry
    return out


# source: cvp_baseline_additions.py
def plot_strain_identification(
    results, exp_conditions, col_titles=None, cmap="Greys",
    figsize=(5.4, 5.0), font_size=10, matrix="probability", save=False,
    figtitle="strain_identification_confusion",
):
    """One matrix per condition, as separate figures. Rows are the true strain,
    columns the candidate identity.

    matrix="probability" (default) shades each cell by P(candidate | observed),
    so a confident identification is a dark diagonal and an ambiguous one
    spreads along its row. matrix="confusion" gives the binary argmin version."""
    labels = list(col_titles) if col_titles is not None else list(exp_conditions)
    figures = []
    for cond, label in zip(exp_conditions, labels):
        r = results[cond]
        use_prob = matrix == "probability" and "probability" in r
        M = r["probability"] if use_prob else r["confusion"]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(M, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        ax.set_xlabel("candidate strain" if use_prob else "assigned strain",
                      fontsize=font_size - 1)
        ax.set_ylabel("true strain", fontsize=font_size - 1)
        subtitle = (f"median P(correct) {r['median_p_correct']:.2f}   "
                    if use_prob else "")
        ax.set_title(
            f"{label}\n{subtitle}top-1 {r['accuracy']:.1%}   "
            f"top-5 {r['top5']:.1%}   chance {r['chance']:.1%}",
            fontsize=font_size - 1,
        )
        if use_prob:
            fig.colorbar(im, ax=ax, shrink=0.85, label="P(identity)")
        ax.tick_params(labelsize=font_size - 2)
        plt.tight_layout(); plt.show()
        if save:
            safe = cond.replace(" ", "_").replace("/", "_")
            for ext in ("png", "pdf", "svg"):
                fig.savefig(f"plots/chlorella_pub_fig_{figtitle}_{safe}.{ext}",
                            bbox_inches="tight", dpi=300)
        figures.append((fig, ax))
    return figures


# source: cvp_baseline_additions.py
def identify_condition_from_phenotype(
    training_data, exp_conditions,
    method="centroid",              # "centroid" or "ridge_template"
    ridge_alpha=500, n_splits=16, groups=None, random_state=42,
    n_wells=96, n_fovs=9, dead_ids=None, metric="euclidean",
):
    """
    Given a strain's phenotype, which condition was it grown in?

    IMPORTANT CONFOUND. In this design each condition is a separate imaging
    session on a separate day, so condition is perfectly confounded with
    imaging batch. A classifier that separates conditions may be reading plate
    or session artifacts rather than the biological response to temperature,
    and nothing in this dataset can distinguish the two. Treat a high accuracy
    here as "the conditions are distinguishable", never as "temperature
    produces these phenotypes". Say so in the legend.

    method="centroid": leave-one-strain-out nearest-centroid. Each strain is
        assigned to the condition whose mean phenotype (computed WITHOUT that
        strain) is closest. Model-free.
    method="ridge_template": assign to the condition whose ridge-predicted
        phenotype for that strain is closest. Uses the model, but note the
        prediction for every condition except the true one was built partly
        from the observation being classified, which biases against the correct
        answer -- so it is conservative rather than optimistic.

    Returns {"confusion", "accuracy", "chance", "labels"}.
    """
    dead = sorted({int(d) % n_wells for d in (dead_ids or [])})
    keep = np.array([w for w in range(n_wells) if w not in dead])
    n_cond = len(exp_conditions)

    # strain-level observed matrix per condition, z-scored on the pooled data
    obs = {c: _to_strain_means(np.asarray(training_data[c], dtype=float),
                               n_wells, n_fovs, keep) for c in exp_conditions}
    pooled = np.vstack([obs[c] for c in exp_conditions])
    _, mu, sd = _zscore(pooled)
    obsz = {c: (obs[c] - mu) / sd for c in exp_conditions}

    if method == "ridge_template":
        oof = _oof_by_condition(training_data, exp_conditions, ridge_alpha,
                                n_splits, groups, random_state)
        predz = {c: (_to_strain_means(oof[c][1], n_wells, n_fovs, keep) - mu) / sd
                 for c in exp_conditions}

    conf = np.zeros((n_cond, n_cond), dtype=int)
    for ti, true_c in enumerate(exp_conditions):
        for s in range(len(keep)):
            x = obsz[true_c][s]
            if method == "centroid":
                # centroid of each condition excluding this strain
                cands = np.array([
                    np.delete(obsz[c], s, axis=0).mean(axis=0)
                    for c in exp_conditions
                ])
            else:
                cands = np.array([predz[c][s] for c in exp_conditions])
            d = _pairwise_distance(x[None, :], cands, metric=metric)[0]
            conf[ti, int(np.argmin(d))] += 1

    return {
        "confusion": conf,
        "accuracy": float(np.trace(conf) / conf.sum()),
        "chance": 1.0 / n_cond,
        "labels": list(exp_conditions),
        "method": method,
    }


# source: cvp_baseline_additions.py
def plot_condition_identification(
    result, col_titles=None, cmap="Blues", figsize=(5.6, 4.8), font_size=10,
    annotate=True, normalize=True, save=False,
    figtitle="condition_identification_confusion",
):
    """Single confusion matrix across conditions, with per-cell counts."""
    conf = result["confusion"].astype(float)
    labels = list(col_titles) if col_titles is not None else result["labels"]
    shown = conf / conf.sum(axis=1, keepdims=True) if normalize else conf

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=1 if normalize else shown.max())
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=font_size - 1)
    ax.set_yticklabels(labels, fontsize=font_size - 1)
    ax.set_xlabel("assigned condition", fontsize=font_size - 1)
    ax.set_ylabel("true condition", fontsize=font_size - 1)
    ax.set_title(f"accuracy {result['accuracy']:.1%}  "
                 f"(chance {result['chance']:.1%}, {result['method']})",
                 fontsize=font_size)
    if annotate:
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                val = shown[i, j]
                ax.text(j, i, f"{val:.2f}" if normalize else f"{int(val)}",
                        ha="center", va="center", fontsize=font_size - 2,
                        color="white" if val > 0.55 * shown.max() else "#333333")
    fig.colorbar(im, ax=ax, shrink=0.85,
                 label="fraction of true class" if normalize else "count")
    plt.tight_layout(); plt.show()
    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig, ax


# source: cvp_baseline_additions.py
def plot_identity_probability_distribution(
    results, exp_conditions, col_titles=None, condition_colors=None,
    figsize=(6.4, 4.2), font_size=10, save=False,
    figtitle="identity_probability_distribution",
):
    """
    P(correct identity) per strain, one box per condition.

    This is the soft counterpart to the confusion matrices: instead of "92% of
    strains were identified", it shows how much probability mass the model puts
    on the right answer, strain by strain. The dashed line is chance (1/n).
    Strains sitting near chance are ones whose predicted phenotype is not
    distinctive -- worth listing, since they are the strains a screen will
    struggle with.
    """
    rows = []
    for cond in exp_conditions:
        r = results[cond]
        if "p_correct" not in r:
            raise ValueError("run identify_strains_from_predictions with "
                             "probabilities=True")
        for s, p in zip(r["strains"], r["p_correct"]):
            rows.append({"condition": cond, "strain": int(s), "p_correct": p})
    df = pd.DataFrame(rows)

    chance = results[exp_conditions[0]]["chance"]
    fig, ax = _box_by_condition(
        df, "p_correct", exp_conditions, col_titles, condition_colors,
        ylabel="P(correct identity)", reference=chance,
        reference_label=f"chance ({chance:.1%})", ylim=(0, 1.02),
        figsize=figsize, font_size=font_size, save=save, figtitle=figtitle,
    )
    return fig, ax, df


# source: cvp_baseline_additions.py
def _well_to_rc(wells, plate_rows, plate_cols, layout):
    if layout == "row_major":
        return wells // plate_cols, wells % plate_cols
    if layout == "column_major":
        return wells % plate_rows, wells // plate_rows
    raise ValueError("layout must be 'row_major' or 'column_major'")


# source: cvp_baseline_additions.py
def _median_polish(M, max_iter=20, tol=1e-6):
    """Tukey median polish. Returns (overall, row_effects, col_effects, residual)."""
    M = M.astype(float).copy()
    grand = 0.0
    r = np.zeros(M.shape[0])
    c = np.zeros(M.shape[1])
    for _ in range(max_iter):
        rm = np.nanmedian(M, axis=1)
        rm = np.where(np.isfinite(rm), rm, 0.0)
        M -= rm[:, None]; r += rm
        delta = np.nanmedian(c)
        c -= delta; grand += delta

        cm = np.nanmedian(M, axis=0)
        cm = np.where(np.isfinite(cm), cm, 0.0)
        M -= cm[None, :]; c += cm
        delta = np.nanmedian(r)
        r -= delta; grand += delta

        if max(np.nanmax(np.abs(rm)), np.nanmax(np.abs(cm))) < tol:
            break
    return grand, r, c, M


# source: cvp_baseline_additions.py
def plate_position_effects(
    training_data, exp_conditions, training_feature_labels,
    n_wells=96, n_fovs=9, dead_ids=None,
    plate_rows=8, plate_cols=12, layout="row_major",
):
    """
    How much of each feature's between-strain variance is explained by where
    the well sits on the plate?

    Fits strain means on an 8 x 12 grid and reports the fraction of variance
    taken by additive row and column effects (median polish). A large value
    means position is masquerading as genotype.

    Returns a DataFrame: condition, feature, frac_row, frac_col, frac_position.
    """
    dead = sorted({int(d) % n_wells for d in (dead_ids or [])})
    keep = np.array([w for w in range(n_wells) if w not in dead])
    rr, cc = _well_to_rc(np.arange(n_wells), plate_rows, plate_cols, layout)

    rows = []
    for cond in exp_conditions:
        M = np.asarray(training_data[cond], dtype=float)
        strain_means = np.array([np.nanmean(M[_well_rows(w, n_wells, n_fovs)], axis=0)
                                 for w in range(n_wells)])
        strain_means[dead] = np.nan

        for j, name in enumerate(training_feature_labels):
            grid = np.full((plate_rows, plate_cols), np.nan)
            grid[rr, cc] = strain_means[:, j]
            total = np.nanvar(grid)
            if not np.isfinite(total) or total <= 0:
                continue
            _, reff, ceff, resid = _median_polish(grid)
            row_only = np.nanvar(np.broadcast_to(reff[:, None], grid.shape))
            col_only = np.nanvar(np.broadcast_to(ceff[None, :], grid.shape))
            rows.append({
                "condition": cond, "feature": name,
                "frac_row": float(row_only / total),
                "frac_col": float(col_only / total),
                "frac_position": float(1 - np.nanvar(resid) / total),
            })
    return pd.DataFrame(rows)


# source: cvp_baseline_additions.py
def plot_plate_heatmap(
    training_data, condition, feature_index, training_feature_labels=None,
    n_wells=96, n_fovs=9, dead_ids=None, plate_rows=8, plate_cols=12,
    layout="row_major", cmap="RdBu_r", figsize=(5.6, 3.8), font_size=10,
    title=None, save=False, figtitle="plate_heatmap",
):
    """Strain means laid out on the plate. A gradient across rows or columns is
    a positional effect you can see rather than infer."""
    dead = sorted({int(d) % n_wells for d in (dead_ids or [])})
    rr, cc = _well_to_rc(np.arange(n_wells), plate_rows, plate_cols, layout)
    M = np.asarray(training_data[condition], dtype=float)
    vals = np.array([np.nanmean(M[_well_rows(w, n_wells, n_fovs), feature_index])
                     for w in range(n_wells)])
    vals[dead] = np.nan
    z = (vals - np.nanmean(vals)) / (np.nanstd(vals) or 1.0)

    grid = np.full((plate_rows, plate_cols), np.nan)
    grid[rr, cc] = z

    fig, ax = plt.subplots(figsize=figsize)
    lim = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks(range(plate_cols)); ax.set_xticklabels(range(1, plate_cols + 1),
                                                         fontsize=font_size - 2)
    ax.set_yticks(range(plate_rows))
    ax.set_yticklabels([chr(ord("A") + i) for i in range(plate_rows)],
                       fontsize=font_size - 2)
    name = (training_feature_labels[feature_index]
            if training_feature_labels is not None else f"feature {feature_index}")
    ax.set_title(title or f"{name} — {condition}", fontsize=font_size)
    fig.colorbar(im, ax=ax, shrink=0.85, label="z-score across strains")
    plt.tight_layout(); plt.show()
    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig, ax


# source: cvp_baseline_additions.py
def cross_condition_gradient_consistency(
    training_data, exp_conditions, training_feature_labels,
    n_wells=96, n_fovs=9, dead_ids=None,
    plate_rows=8, plate_cols=12, layout="row_major",
    n_null=20, random_state=0,
):
    """
    Is the plate gradient the same on every plate, or does each plate have its own?

    This is the number that decides how much the strain x environment claims
    need to change.

    Each condition was imaged on its own plate on its own day. A positional
    effect that is IDENTICAL across plates behaves like a constant per-well
    offset: it inflates the strain main effect but cancels out of the
    interaction, leaving the G x E results intact. A positional effect that
    DIFFERS between plates is, by construction, a well-by-condition effect --
    indistinguishable from strain x environment interaction in the variance
    partitioning.

    Per feature:

    gradient_consistency  -- DIAGNOSTIC ONLY, DO NOT READ AS "the gradient is
        shared". Mean pairwise correlation, across conditions, of the fitted
        per-well positional surface. It is confounded: the strain main effect
        is by definition identical across conditions, and its projection onto
        rows and columns is picked up by the fit, so this runs high even with
        no plate gradient whatsoever. On simulated data with NO gradient it
        read 0.51, higher than a case where every plate genuinely had its own
        gradient (0.25). Use it to see whether surfaces resemble each other at
        all, never to decide the question below.

    excess_interaction_from_position  -- THIS IS THE NUMBER TO USE.
        Variance of the CONDITION-SPECIFIC part of the positional fit over the
        variance of the empirical strain x environment term, minus the same
        quantity under the permutation null. Positive means plate-specific
        position explains interaction variance beyond what fitting noise
        produces; at or below zero means it does not, and the G x E claims are
        safe from this particular artifact.

        Validated on simulated data: no gradient -58%, one shared gradient
        across all plates -92%, each plate its own gradient +12%. Only the last
        case comes out positive.

    frac_strain_from_position
        the same idea for the strain main effect: variance of the SHARED
        positional surface over the variance of the shared strain effect.

    THE NULL MATTERS HERE. Median polish fits a surface to data that also
    contains strain and interaction variation, so it absorbs some of that noise
    and the fitted surfaces differ between conditions even when the true
    gradient is identical. On simulated data with one shared gradient the raw
    consistency was 0.75, not 1.0, and 52.8% of interaction variance was
    attributed to position purely by fitting noise. So both statistics come
    with a permutation null (`null_*` columns) that shuffles wells across plate
    positions independently per condition, destroying spatial structure while
    keeping everything else. Read the EXCESS over the null, not the raw value.

    Returns a DataFrame with one row per feature.
    """
    dead = sorted({int(d) % n_wells for d in (dead_ids or [])})
    alive = np.array([w for w in range(n_wells) if w not in dead])
    rr, cc = _well_to_rc(np.arange(n_wells), plate_rows, plate_cols, layout)
    n_cond = len(exp_conditions)

    # strain means per condition, dead wells masked
    means = {}
    for cond in exp_conditions:
        M = np.asarray(training_data[cond], dtype=float)
        sm = np.array([np.nanmean(M[_well_rows(w, n_wells, n_fovs)], axis=0)
                       for w in range(n_wells)])
        sm[dead] = np.nan
        means[cond] = sm

    rows = []
    for j, name in enumerate(training_feature_labels):
        # centre each condition: removes the environment main effect
        centred = np.vstack([
            means[c][:, j] - np.nanmean(means[c][:, j]) for c in exp_conditions
        ])                                                  # (n_cond, n_wells)
        A = centred[:, alive]
        shared = A.mean(axis=0)                              # strain main effect
        interaction = A - shared[None, :]                    # empirical S x E
        var_int, var_shared = np.var(interaction), np.var(shared)

        def _stats(values_by_cond):
            """values_by_cond: (n_cond, n_alive) laid out on the plate."""
            fits = np.full((n_cond, len(alive)), np.nan)
            for ci in range(n_cond):
                grid = np.full((plate_rows, plate_cols), np.nan)
                grid[rr[alive], cc[alive]] = values_by_cond[ci]
                if not np.isfinite(np.nanvar(grid)) or np.nanvar(grid) <= 0:
                    return None
                _, reff, ceff, _ = _median_polish(grid)
                fits[ci] = reff[rr[alive]] + ceff[cc[alive]]
            if not np.isfinite(fits).all():
                return None
            cors = [np.corrcoef(fits[a], fits[b])[0, 1]
                    for a in range(n_cond) for b in range(a + 1, n_cond)
                    if np.std(fits[a]) > 0 and np.std(fits[b]) > 0]
            shared_pos = fits.mean(axis=0)
            specific_pos = fits - shared_pos[None, :]
            return (float(np.mean(cors)) if cors else np.nan,
                    float(np.var(specific_pos) / var_int) if var_int > 0 else np.nan,
                    float(np.var(shared_pos) / var_shared) if var_shared > 0 else np.nan)

        obs = _stats(A)
        if obs is None:
            continue

        # null: break the link between well and plate position, per condition
        rng = np.random.default_rng(random_state + j)
        null = []
        for _ in range(n_null):
            shuffled = np.vstack([A[ci][rng.permutation(len(alive))]
                                  for ci in range(n_cond)])
            st = _stats(shuffled)
            if st is not None:
                null.append(st)
        null = np.array(null) if null else np.full((1, 3), np.nan)

        rows.append({
            "feature": name,
            "gradient_consistency": obs[0],
            "null_gradient_consistency": float(np.nanmean(null[:, 0])),
            "excess_consistency": obs[0] - float(np.nanmean(null[:, 0])),
            "frac_interaction_from_position": obs[1],
            "null_frac_interaction_from_position": float(np.nanmean(null[:, 1])),
            "excess_interaction_from_position": obs[1] - float(np.nanmean(null[:, 1])),
            "frac_strain_from_position": obs[2],
            "null_frac_strain_from_position": float(np.nanmean(null[:, 2])),
        })
    return pd.DataFrame(rows)


# source: cvp_baseline_additions.py
def plot_gradient_consistency(
    gradient_df, figsize=(7.2, 4.0), font_size=10, color=None,
    save=False, figtitle="cross_condition_gradient_consistency",
):
    """Two panels: how similar the plate gradients are across conditions, and
    how much of the interaction term plate-specific position could explain."""
    if color is None:
        color = apc.aster
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    specs = [
        ("excess_consistency",
         "shared plate gradient across\nconditions, above the null",
         (-1.02, 1.02), 0.0, "no shared gradient"),
        ("excess_interaction_from_position",
         "interaction variance from plate-specific\nposition, above the null",
         None, 0.0, "nothing above fitting noise"),
    ]
    for ax, (col, ylabel, ylim, ref, ref_label) in zip(axes, specs):
        vals = gradient_df[col].dropna().to_numpy()
        bp = ax.boxplot([vals], widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.6),
                        whiskerprops=dict(color="#666666", lw=1.0),
                        capprops=dict(color="#666666", lw=1.0))
        bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.35)
        bp["boxes"][0].set_edgecolor("#444444")
        rng = np.random.default_rng(0)
        ax.scatter(1 + rng.uniform(-0.09, 0.09, len(vals)), vals, s=14,
                   color=color, alpha=0.45, linewidth=0, zorder=3)
        if ref is not None:
            ax.axhline(ref, color="#999999", ls="--", lw=1.0)
            if ref_label:
                ax.annotate(ref_label, xy=(0.99, ref),
                            xycoords=("axes fraction", "data"), xytext=(0, 3),
                            textcoords="offset points", ha="right", va="bottom",
                            fontsize=font_size - 2, color="#777777")
        if ylim: ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=font_size - 1)
        ax.set_xticks([]); ax.tick_params(labelsize=font_size - 1)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"median {np.median(vals):.2f}", fontsize=font_size)

    plt.tight_layout(); plt.show()
    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig, axes


# source: cvp_baseline_additions.py
_PLOTLY_EXTRAS = dict(
    backend="static",
    hover_fields=None,
    hover_data=None,
    hover_labels=None,
    hover_formats=None,
    hover_template=None,
    plotly_show_xticklabels=None,
    plotly_show_yticklabels=None,
    plotly_width=None,
    plotly_height=None,
    plotly_dpi=100,
    plotly_show=True,
    plotly_save_formats=("html",),
    plotly_include_plotlyjs="cdn",
    plotly_figtitle=None,
    plotly_layout=None,
    plotly_hover_space="auto",
    plotly_annot=None,
    plotly_cbar=None,
    save_csv=True,          # write the heatmap's values (plot_df) to a CSV
    csv_dir="results",      # folder for that CSV: <csv_dir>/<figtitle>_values.csv
)


# source: cvp_baseline_additions.py
def _split_plotly_kwargs(kwargs):
    extras = {k: kwargs.pop(k, v) for k, v in _PLOTLY_EXTRAS.items()}
    if extras["backend"] not in ("static", "plotly", "both"):
        raise ValueError('backend must be "static", "plotly", or "both"')
    return extras


# source: cvp_baseline_additions.py
def _run_static_silently(fn, args, kwargs):
    """Run the cvp.py function without showing or saving its figure."""
    params = _inspect.signature(fn).parameters
    kw = dict(kwargs)
    bound = _inspect.signature(fn).bind(*args, **kw)
    if "save" in params:
        bound.arguments["save"] = False
    if "show_plot" in params:
        bound.arguments["show_plot"] = False
    original_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        out = fn(*bound.args, **bound.kwargs)
    finally:
        plt.show = original_show
    plt.close(out[0])
    return out


# source: cvp_baseline_additions.py
def _resolved_arguments(fn, args, kwargs):
    bound = _inspect.signature(fn).bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


# source: cvp_baseline_additions.py
def _css_color(c):
    try:
        return _mcolors.to_hex(c)
    except (ValueError, TypeError):
        return str(getattr(c, "hex_code", c))


# source: cvp_baseline_additions.py
def _plotly_colorscale(cmap, n=64):
    return [[i / (n - 1), _mcolors.to_hex(cmap(i / (n - 1)))] for i in range(n)]


# source: cvp_baseline_additions.py
def _pt(lw):
    """matplotlib points -> CSS pixels."""
    return float(lw) * 96.0 / 72.0


# source: cvp_baseline_additions.py
def _align_to(obj, plot_df, name="hover field"):
    """Return an object array shaped like plot_df from a scalar/Series/dict/array/DataFrame."""
    nr, nc = plot_df.shape
    rows = [str(r) for r in plot_df.index]
    cols = [str(c) for c in plot_df.columns]

    if isinstance(obj, dict):
        obj = pd.Series(obj)

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        df.index = [str(i) for i in df.index]
        df.columns = [str(c) for c in df.columns]
        hit = len(set(rows) & set(df.index)) + len(set(cols) & set(df.columns))
        hit_t = len(set(rows) & set(df.columns)) + len(set(cols) & set(df.index))
        if hit_t > hit:
            df = df.T
        if not (set(rows) & set(df.index)) or not (set(cols) & set(df.columns)):
            raise ValueError(f"hover_data['{name}']: DataFrame labels do not match the plotted rows/columns")
        return df.reindex(index=rows, columns=cols).to_numpy(dtype=object)

    if isinstance(obj, pd.Series):
        s = obj.copy()
        s.index = [str(i) for i in s.index]
        on_rows = len(set(rows) & set(s.index))
        on_cols = len(set(cols) & set(s.index))
        if on_rows == 0 and on_cols == 0:
            raise ValueError(f"hover_data['{name}']: Series index matches neither the rows nor the columns")
        if on_rows >= on_cols:
            v = s.reindex(rows).to_numpy(dtype=object)
            return np.repeat(v[:, None], nc, axis=1)
        v = s.reindex(cols).to_numpy(dtype=object)
        return np.repeat(v[None, :], nr, axis=0)

    arr = np.asarray(obj, dtype=object)
    if arr.ndim == 0:
        return np.full((nr, nc), arr.item(), dtype=object)
    if arr.ndim == 1:
        if len(arr) == nr:
            return np.repeat(arr[:, None], nc, axis=1)
        if len(arr) == nc:
            return np.repeat(arr[None, :], nr, axis=0)
    if arr.ndim == 2:
        if arr.shape == (nr, nc):
            return arr
        if arr.shape == (nc, nr):
            return arr.T
    raise ValueError(f"hover_data['{name}']: shape {arr.shape} does not fit plot_df shape {plot_df.shape}")


# source: cvp_baseline_additions.py
def _is_numeric_array(a):
    for v in np.ravel(a):
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            return False
    return True


# source: cvp_baseline_additions.py
def _jsonable(a):
    out = np.empty(a.shape, dtype=object)
    for idx, v in np.ndenumerate(a):
        if isinstance(v, (np.floating, float)):
            out[idx] = None if not np.isfinite(v) else float(v)
        elif isinstance(v, np.integer):
            out[idx] = int(v)
        elif v is None or (isinstance(v, float) and np.isnan(v)):
            out[idx] = None
        else:
            out[idx] = v if isinstance(v, (int, float, str)) else str(v)
    return out


# source: cvp_baseline_additions.py
def _build_hover(fields, plot_df, extras, default_names):
    """
    fields : dict name -> dict(data=..., label=str, fmt=str)
    Returns customdata (nr, nc, nf), hovertemplate, field_order.
    """
    fields = dict(fields)
    user = extras["hover_data"] or {}
    for name, data in user.items():
        fields[name] = dict(data=data, label=name, fmt=None)

    requested = extras["hover_fields"]
    if requested is None:
        order = [n for n in default_names if n in fields] + [n for n in user if n not in default_names]
    elif isinstance(requested, str) and requested == "all":
        order = list(fields)
    else:
        order = list(requested)
        missing = [n for n in order if n not in fields]
        if missing:
            raise ValueError(f"Unknown hover field(s) {missing}. Available: {list(fields)}")

    # a custom template may use fields that are not in the default list
    template = extras["hover_template"]
    if template:
        for n in _re.findall(r"(?<!%)\{(\w+)(?::[^}]*)?\}", template):
            if n in fields and n not in order:
                order.append(n)

    labels = dict(extras["hover_labels"] or {})
    formats = dict(extras["hover_formats"] or {})

    arrays, lines = [], []
    for i, name in enumerate(order):
        arr = _align_to(fields[name]["data"], plot_df, name)
        numeric = _is_numeric_array(arr)
        arrays.append(_jsonable(arr))
        fmt = formats.get(name, fields[name].get("fmt"))
        if fmt is None:
            fmt = ".4g" if numeric else ""
        if not numeric:
            fmt = ""
        fmt_txt = f":{fmt}" if fmt else ""
        label = labels.get(name, fields[name].get("label", name))
        lines.append(f"{label}: %{{customdata[{i}]{fmt_txt}}}")

    customdata = np.stack(arrays, axis=-1) if arrays else None

    if template:
        index = {n: i for i, n in enumerate(order)}

        def _sub(m):
            n, fmt = m.group(1), m.group(2)
            fmt_txt = f":{fmt}" if fmt else ""
            if n in index:
                return f"%{{customdata[{index[n]}]{fmt_txt}}}"
            if n in ("x", "y", "z"):
                return f"%{{z{fmt_txt}}}" if n == "z" else m.group(0)
            return m.group(0)

        hovertemplate = _re.sub(r"(?<!%)\{(\w+)(?::([^}]*))?\}", _sub, template)
    else:
        hovertemplate = "<br>".join(lines)
    if "<extra>" not in hovertemplate:
        hovertemplate += "<extra></extra>"
    return customdata, hovertemplate, order


# source: cvp_baseline_additions.py
def _label_grid(plot_df, axis):
    """Broadcast the row (axis=0) or column (axis=1) labels over the grid."""
    if axis == 0:
        return pd.Series(list(plot_df.index), index=plot_df.index)
    return pd.Series(list(plot_df.columns), index=plot_df.columns)


# source: cvp_baseline_additions.py
def _estimate_hover_width(customdata, hovertemplate, font_px):
    """Rough pixel width of the widest line of the hover box."""
    lens = []
    if customdata is not None:
        cd = np.asarray(customdata, dtype=object)
        for k in range(cd.shape[-1]):
            vals = cd[..., k].ravel()
            if _is_numeric_array(vals):
                lens.append(8)
            else:
                # multi-line string fields (e.g. icc_by_condition) contain <br>
                lens.append(max((len(_re.sub(r"<[^>]+>", "", seg))
                                 for v in vals if v is not None
                                 for seg in _re.split(r"<br\s*/?>", str(v))), default=0))

    def _sub(m):
        k = int(m.group(1))
        return "x" * (lens[k] if k < len(lens) else 8)

    text = _re.sub(r"<extra>.*?</extra>", "", hovertemplate)
    text = _re.sub(r"%\{customdata\[(\d+)\][^}]*\}", _sub, text)
    text = _re.sub(r"%\{[^}]*\}", "x" * 8, text)
    widest = 0
    for line in _re.split(r"<br\s*/?>", text):
        widest = max(widest, len(_re.sub(r"<[^>]+>", "", line)))
    return int(widest * font_px * 0.6 + 24)


# source: cvp_baseline_additions.py
def _plotly_heatmap_from_static(
    fig, ax, plot_df, P, extras, fields, default_fields,
    annot=False, annot_fmt=".2f", annot_fontsize=10, annot_color="auto",
    outline_at_or_below=None, outline_color="black", outline_lw=0.6,
    linewidths=0, linecolor="white",
):
    import plotly.graph_objects as go

    mesh = ax.collections[0]
    cmap = mesh.cmap
    zmin, zmax = float(mesh.norm.vmin), float(mesh.norm.vmax)

    Z = plot_df.to_numpy(dtype=float)
    nr, nc = Z.shape

    customdata, hovertemplate, _ = _build_hover(fields, plot_df, extras, default_fields)

    font_size = P.get("font_size", 10) or 10
    font_family = P.get("font_family", None)
    xtick_fs = P.get("xtick_fontsize", P.get("x_tick_fontsize")) or font_size
    ytick_fs = P.get("ytick_fontsize", P.get("y_tick_fontsize")) or font_size
    xrot = P.get("xtick_rotation", P.get("x_tick_rotation", 0)) or 0
    yrot = P.get("ytick_rotation", P.get("y_tick_rotation", 0)) or 0
    label_color = _css_color(P.get("label_color", "black"))
    axis_color = _css_color(P.get("axis_color", "black"))

    show_x = extras["plotly_show_xticklabels"]
    show_y = extras["plotly_show_yticklabels"]
    if show_x is None:
        show_x = bool(P.get("show_xticklabels", True))
    if show_y is None:
        show_y = bool(P.get("show_yticklabels", True))

    def _ticktext(labels, colors):
        if colors is None:
            colors = [label_color] * len(labels)
        if len(colors) != len(labels):
            raise ValueError("tick color list length does not match the number of labels")
        return [f"<span style='color:{_css_color(c)}'>{l}</span>" for l, c in zip(labels, colors)]

    xticktext = _ticktext([str(c) for c in plot_df.columns], P.get("xtick_colors"))
    yticktext = _ticktext([str(r) for r in plot_df.index], P.get("ytick_colors"))

    show_cbar = bool(P.get("cbar", True)) if extras["plotly_cbar"] is None else bool(extras["plotly_cbar"])

    # --- size ---
    figsize = P.get("figsize", (8, 8))
    width = extras["plotly_width"] or int(figsize[0] * extras["plotly_dpi"])
    height = extras["plotly_height"] or int(figsize[1] * extras["plotly_dpi"])
    # matplotlib squeezes labels into figsize; Plotly would squeeze the plot
    # area instead, so leave room for the labels on very narrow/short figures
    if not extras["plotly_width"]:
        left = (max((len(str(r)) for r in plot_df.index), default=0) * ytick_fs * 0.62 + 20) if show_y else 10
        right = 90 if show_cbar else 10
        width = max(width, int(left + right + max(20, 3 * nc)))
    if not extras["plotly_height"]:
        bottom = (max((len(str(c)) for c in plot_df.columns), default=0) * xtick_fs * 0.62
                  * abs(np.sin(np.deg2rad(xrot))) + xtick_fs + 20) if show_x else 10
        height = max(height, int(bottom + 60 + max(20, 3 * nr)))

    # --- colorbar ---
    cbar_title = None
    if show_cbar and getattr(mesh, "colorbar", None) is not None:
        cbar_title = mesh.colorbar.ax.get_ylabel() or None
    elif show_cbar and isinstance(P.get("cbar_label"), str):
        cbar_title = P["cbar_label"]
    shrink = P.get("cbar_shrink", 1.0) or 1.0
    aspect = P.get("cbar_aspect", 30) or 30
    colorbar = dict(
        len=float(shrink),
        thickness=max(8, int(height * float(shrink) / float(aspect))),
        title=dict(text=cbar_title, side="right", font=dict(size=font_size)) if cbar_title else None,
        tickfont=dict(size=font_size),
        outlinewidth=0,
    )

    heat_kw = dict(
        z=Z,
        x=list(range(nc)),
        y=list(range(nr)),
        colorscale=_plotly_colorscale(cmap),
        zmin=zmin,
        zmax=zmax,
        customdata=customdata,
        hovertemplate=hovertemplate,
        showscale=show_cbar,
        colorbar=colorbar,
        hoverongaps=False,
    )
    if linewidths:
        heat_kw.update(xgap=_pt(linewidths), ygap=_pt(linewidths))
    if annot:
        heat_kw["texttemplate"] = "%{z:" + str(annot_fmt).lstrip(":") + "}"
        tf = dict(size=annot_fontsize)
        if font_family:
            tf["family"] = font_family
        if annot_color not in (None, "auto"):
            tf["color"] = _css_color(annot_color)
        heat_kw["textfont"] = tf

    pfig = go.Figure(go.Heatmap(**heat_kw))

    # --- outlines (e.g. ICC cells at or below threshold) ---
    if outline_at_or_below is not None:
        shapes = []
        for yi in range(nr):
            for xi in range(nc):
                if np.isfinite(Z[yi, xi]) and Z[yi, xi] <= outline_at_or_below:
                    shapes.append(dict(
                        type="rect", xref="x", yref="y",
                        x0=xi - 0.5, x1=xi + 0.5, y0=yi - 0.5, y1=yi + 0.5,
                        line=dict(color=_css_color(outline_color), width=max(0.5, _pt(outline_lw))),
                        fillcolor="rgba(0,0,0,0)", layer="above",
                    ))
        pfig.update_layout(shapes=shapes)

    title = ax.get_title() if P.get("show_title", False) else ""
    title_size = P.get("title_size") or P.get("title_fontsize") or font_size

    axis_common = dict(
        showgrid=False, zeroline=False, showline=False,
        ticks="outside", tickcolor=axis_color, ticklen=4,
        color=axis_color, automargin=True,
    )
    pfig.update_layout(
        width=width,
        height=height,
        font=dict(family=font_family, size=font_size, color=label_color) if font_family
        else dict(size=font_size, color=label_color),
        title=dict(text=title, font=dict(size=title_size), x=0.5) if title else None,
        plot_bgcolor=_css_color(linecolor) if linewidths else "white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50 if title else 10, b=10),
        xaxis=dict(
            axis_common,
            tickmode="array", tickvals=list(range(nc)), ticktext=xticktext,
            showticklabels=show_x, tickangle=-float(xrot), tickfont=dict(size=xtick_fs),
            title=dict(text=ax.get_xlabel() or None, font=dict(size=font_size)),
            range=[-0.5, nc - 0.5],
            ticks="outside" if show_x else "",
        ),
        yaxis=dict(
            axis_common,
            tickmode="array", tickvals=list(range(nr)), ticktext=yticktext,
            showticklabels=show_y, tickangle=-float(yrot), tickfont=dict(size=ytick_fs),
            title=dict(text=ax.get_ylabel() or None, font=dict(size=font_size)),
            range=[nr - 0.5, -0.5],   # first row on top, like seaborn
            ticks="outside" if show_y else "",
        ),
        hoverlabel=dict(align="left", font=dict(size=max(font_size, 11))),
    )
    # --- room for the hover box on narrow figures ---
    hover_space = extras["plotly_hover_space"]
    if hover_space == "auto":
        hover_font = max(font_size, 11)
        need = _estimate_hover_width(customdata, hovertemplate, hover_font) + 20
        right_existing = 90 if show_cbar else 10
        # Plotly opens the box on whichever side of the cursor has room. The
        # worst cursor position is `need` px from the left edge (or the plot's
        # right edge, if that comes first); add only what that position lacks.
        x_worst = min(need, width - right_existing)
        hover_space = max(0, need - (width - x_worst))
    hover_space = int(hover_space or 0)
    if hover_space > 0:
        pfig.update_layout(width=width + hover_space, margin=dict(r=10 + hover_space))

    if extras["plotly_layout"]:
        pfig.update_layout(**extras["plotly_layout"])
    return pfig


# source: cvp_baseline_additions.py
def _save_plotly(pfig, figtitle, extras):
    stem = "plots/chlorella_pub_fig_" + (extras["plotly_figtitle"] or (figtitle + "_interactive"))
    config = {"displaylogo": False, "toImageButtonOptions": {"format": "svg", "filename": stem.split("/")[-1]}}
    for fmt in extras["plotly_save_formats"] or ():
        if fmt == "html":
            pfig.write_html(stem + ".html", include_plotlyjs=extras["plotly_include_plotlyjs"], config=config)
        elif fmt == "json":
            pfig.write_json(stem + ".json")
        elif fmt in ("png", "svg", "pdf"):
            pfig.write_image(stem + "." + fmt)   # needs kaleido
        else:
            raise ValueError(f"Unknown plotly_save_format {fmt!r}")


# added for the pub notebook: CSV export of the plotted heatmap values
def _save_heatmap_csv(plot_df, P, default_figtitle, extras):
    """Write the values shown in the heatmap (rows/columns exactly as plotted) to a CSV."""
    import os
    figtitle = P.get("figtitle") or default_figtitle(P)
    os.makedirs(extras["csv_dir"], exist_ok=True)
    path = os.path.join(extras["csv_dir"], f"{figtitle}_values.csv")
    plot_df.to_csv(path)
    print(f"Saved heatmap values to {path}")


# source: cvp_baseline_additions.py
def _dispatch(fn, args, kwargs, builder, default_figtitle, style=None):
    """Shared driver for every wrapper."""
    extras = _split_plotly_kwargs(kwargs)
    backend = extras["backend"]

    if backend == "static":
        out = fn(*args, **kwargs)
        if extras["save_csv"]:
            _save_heatmap_csv(out[2], _resolved_arguments(fn, args, kwargs), default_figtitle, extras)
        return out

    P = _resolved_arguments(fn, args, kwargs)
    if backend == "both":
        fig, ax, plot_df = fn(*args, **kwargs)
    else:
        fig, ax, plot_df = _run_static_silently(fn, args, kwargs)
    if extras["save_csv"]:
        _save_heatmap_csv(plot_df, P, default_figtitle, extras)

    fields, default_fields = builder(P, plot_df)
    style = style(P) if style else {}
    if extras["plotly_annot"] is not None:
        style["annot"] = bool(extras["plotly_annot"])
    pfig = _plotly_heatmap_from_static(fig, ax, plot_df, P, extras, fields, default_fields, **style)

    if P.get("save", False):
        figtitle = P.get("figtitle") or default_figtitle(P)
        _save_plotly(pfig, figtitle, extras)
    if extras["plotly_show"]:
        pfig.show(config={"displaylogo": False})

    if backend == "both":
        return fig, ax, plot_df, pfig
    return pfig, plot_df


# source: cvp_baseline_additions.py
def _f(data, label, fmt=None):
    return dict(data=data, label=label, fmt=fmt)


# source: cvp_baseline_additions.py
def _rank_desc(df, axis):
    """1 = highest value along the given axis."""
    return df.rank(axis=axis, ascending=False, method="min")


# source: cvp_baseline_additions.py
def _fields_all_strain(P, plot_df):
    td = P["training_data"]
    condition = _resolve_condition_key(P["temperature"], P["growth_stage"], list(td.keys()))
    dat = np.asarray(td[condition], dtype=float)
    n_wells = P["n_wells"]
    n_fovs = P["n_fovs"] if P["n_fovs"] is not None else dat.shape[0] // n_wells
    fovs = P["fovs"]
    if isinstance(fovs, str):
        fov_list = list(range(n_fovs))
    elif isinstance(fovs, (int, np.integer)):
        fov_list = [int(fovs)]
    else:
        fov_list = [int(f) for f in fovs]

    vertical = P["orientation"] == "vertical"
    canon = plot_df.T if vertical else plot_df            # rows = strains, cols = features
    strains = [str(s) for s in canon.index]
    features = list(canon.columns)
    labels = list(P["training_feature_labels"])
    fcols = [labels.index(f) for f in features]

    raw = np.full((len(strains), len(features)), np.nan)
    sd = np.full_like(raw, np.nan)
    nused = np.zeros_like(raw)
    for i, s in enumerate(strains):
        block = dat[[int(s) + f * n_wells for f in fov_list], :][:, fcols]
        block = np.where(np.isfinite(block), block, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw[i] = np.nanmean(block, axis=0)
            sd[i] = np.nanstd(block, axis=0, ddof=1) if block.shape[0] > 1 else np.nan
        nused[i] = np.isfinite(block).sum(axis=0)

    raw_df = pd.DataFrame(raw, index=strains, columns=features)
    value_df = canon.copy()
    value_df.index = strains
    rank = _rank_desc(value_df, axis=0)

    feat_axis, strain_axis = (0, 1) if vertical else (1, 0)
    cbar_label = {
        None: "Feature value", "none": "Feature value", False: "Feature value",
        "zscore": "Z-score", "double_zscore": "Double z-score", "minmax": "Min-max value",
    }.get(P["normalize"], "Value")

    fields = {
        "strain": _f(_label_grid(plot_df, strain_axis), "Strain (well)"),
        "feature": _f(_label_grid(plot_df, feat_axis), "Feature"),
        "condition": _f(condition, "Condition"),
        "value": _f(plot_df.to_numpy(dtype=float), cbar_label, ".2f"),
        "raw_mean": _f(raw_df, "Raw mean", ".4g"),
        "fov_sd": _f(pd.DataFrame(sd, index=strains, columns=features), "SD across FoVs", ".3g"),
        "fov_cv": _f(pd.DataFrame(sd / np.abs(raw), index=strains, columns=features), "CV across FoVs", ".2f"),
        "n_fovs_used": _f(pd.DataFrame(nused, index=strains, columns=features), "FoVs used", ".0f"),
        "rank": _f(rank, f"Rank among {len(strains)} strains", ".0f"),
    }
    return fields, ["strain", "feature", "value", "raw_mean", "fov_sd", "rank"]


# source: cvp_baseline_additions.py
def _figtitle_all_strain(P):
    condition = _resolve_condition_key(P["temperature"], P["growth_stage"], list(P["training_data"].keys()))
    fovs = P["fovs"]
    if isinstance(fovs, str):
        n_rows = np.asarray(P["training_data"][condition]).shape[0]
        fov_list = list(range(P["n_fovs"] or n_rows // P["n_wells"]))
    elif isinstance(fovs, (int, np.integer)):
        fov_list = [int(fovs)]
    else:
        fov_list = list(fovs)
    return f"all_strain_feature_heatmap_{condition}_fov" + "_".join(str(f) for f in fov_list)


# source: cvp_baseline_additions.py
def plot_all_strain_feature_heatmap(*args, **kwargs):
    """
    cvp.plot_all_strain_feature_heatmap with an optional interactive version.

    All cvp arguments work unchanged. Extras: backend, hover_*, plotly_* (see the
    block comment above "INTERACTIVE (PLOTLY) VERSIONS").

    Hover fields: strain, feature, condition, value (plotted, e.g. z-score),
    raw_mean (un-normalized FoV mean), fov_sd, fov_cv, n_fovs_used, rank
    (1 = highest value of that feature among the plotted strains).
    Default: strain, feature, value, raw_mean, fov_sd, rank.

    Example
    -------
    pfig, df = cba.plot_all_strain_feature_heatmap(
        ...same arguments as before...,
        backend="plotly",
        plotly_show_xticklabels=False,             # strain labels only on hover
        hover_data={"genotype": strain_genotypes}, # Series indexed by well
    )
    """
    return _dispatch(_cvp_plot_all_strain_feature_heatmap, args, kwargs,
                     _fields_all_strain, _figtitle_all_strain)


# source: cvp_baseline_additions.py
def _fields_icc(P, plot_df):
    R = P["icc_results"]
    labels = list(R["labels"])
    conds = list(R["conditions"])
    col_titles = P["col_titles"]
    cond_names = list(col_titles) if col_titles is not None and len(col_titles) == len(conds) else conds
    metric, column = P["metric"], P["column"]
    feats = list(plot_df.index)
    fi = [labels.index(f) for f in feats]
    cols = list(plot_df.columns)

    per_condition = metric != "cross_condition_rho" and column == "per_condition"

    def cell_metric(key, reducer):
        arr = np.asarray(R[key], dtype=float)[fi, :]          # (features, conditions)
        if per_condition:
            return pd.DataFrame(arr, index=feats, columns=cols)
        if column not in ("mean", "per_condition") and column in conds:
            return pd.Series(arr[:, conds.index(column)], index=feats)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.Series(reducer(arr, axis=1), index=feats)

    def feat(key):
        return pd.Series(np.asarray(R[key])[fi], index=feats)

    icc = np.asarray(R["icc_single"], dtype=float)[fi, :]
    by_cond = pd.Series(
        ["<br>   " + "<br>   ".join(f"{c}: {v:.2f}" for c, v in zip(cond_names, row)) for row in icc],
        index=feats,
    )
    thr = P["icc_threshold"] if P["icc_threshold"] is not None else R.get("icc_threshold")
    Z = plot_df.to_numpy(dtype=float)

    value_label = {"icc_single": "ICC(1,1)", "icc_mean": f"ICC(1,{R.get('n_fovs', 'k')})",
                   "split_half": "Split-half rho", "cross_condition_rho": "Cross-condition rho"}[metric]
    if column == "mean" and metric != "cross_condition_rho":
        value_label = "Mean " + value_label

    worst = pd.Series([cond_names[conds.index(c)] for c in np.asarray(R["worst_condition"])[fi]], index=feats)
    best = pd.Series([cond_names[conds.index(c)] for c in np.asarray(R["best_condition"])[fi]], index=feats)

    fields = {
        "feature": _f(_label_grid(plot_df, 0), "Feature"),
        "column": _f(_label_grid(plot_df, 1), "Condition"),
        "value": _f(Z, value_label, ".3f"),
        "icc_single": _f(cell_metric("icc_single", np.nanmean), "ICC(1,1)", ".3f"),
        "icc_mean": _f(cell_metric("icc_mean", np.nanmean), f"ICC(1,{R.get('n_fovs', 'k')})", ".3f"),
        "split_half": _f(cell_metric("split_half", np.nanmean), "Split-half rho", ".3f"),
        "fstat": _f(cell_metric("fstat", np.nanmean), "F statistic", ".3g"),
        "pvalue": _f(cell_metric("pvalue", np.nanmax), "Permutation p", ".3g"),
        "mean_icc": _f(feat("mean_icc"), "Mean ICC(1,1)", ".3f"),
        "min_icc": _f(feat("min_icc"), "Min ICC(1,1)", ".3f"),
        "max_icc": _f(feat("max_icc"), "Max ICC(1,1)", ".3f"),
        "best_condition": _f(best, "Best condition"),
        "worst_condition": _f(worst, "Worst condition"),
        "cross_condition_rho": _f(feat("cross_condition_rho"), "Cross-condition rho", ".3f"),
        "n_conditions_passing": _f(pd.Series(np.asarray(R["passes_per_feature"])[fi], index=feats),
                                   f"Conditions with ICC > {R.get('icc_threshold')}", ".0f"),
        "passes_threshold": _f(np.where(np.isfinite(Z), np.where(Z > (thr if thr is not None else -np.inf),
                                                                 "yes", "no"), ""),
                               f"Above {thr}" if thr is not None else "Above threshold"),
        "icc_by_condition": _f(by_cond, "ICC(1,1) by condition"),
        "rank": _f(_rank_desc(pd.DataFrame(Z, index=feats, columns=cols), axis=0),
                   f"Rank of {len(feats)}", ".0f"),
    }
    default = ["feature"] + (["column"] if Z.shape[1] > 1 else []) + \
              ["value", "min_icc", "worst_condition", "n_conditions_passing", "cross_condition_rho"]
    return fields, default


# source: cvp_baseline_additions.py
def _style_icc(P):
    thr = P["icc_threshold"]
    if thr is None:
        thr = P["icc_results"].get("icc_threshold", None)
    return dict(
        annot=P["annot"], annot_fmt=P["annot_fmt"], annot_fontsize=P["annot_fontsize"],
        annot_color=P["annot_color"], outline_at_or_below=thr,
        outline_color=P["threshold_color"], outline_lw=P["threshold_lw"],
        linewidths=P["linewidths"], linecolor=P["linecolor"],
    )


# source: cvp_baseline_additions.py
def plot_feature_icc_heatmap(*args, **kwargs):
    """
    cvp.plot_feature_icc_heatmap with an optional interactive version.

    Hover fields: feature, column, value, icc_single, icc_mean, split_half,
    fstat, pvalue (averaged over conditions for column="mean"; pvalue uses the
    max), mean_icc, min_icc, max_icc, best_condition, worst_condition,
    cross_condition_rho, n_conditions_passing, passes_threshold,
    icc_by_condition (ICC(1,1) in every condition, one per line), rank.
    Condition names follow col_titles when given.
    """
    return _dispatch(_cvp_plot_feature_icc_heatmap, args, kwargs,
                     _fields_icc, lambda P: P["figtitle"], _style_icc)


# source: cvp_baseline_additions.py
def _fields_r2(P, plot_df):
    res = P["ridge_model_residulals"]
    r2 = np.asarray(res["real"]["r2_per_feature"], dtype=float)
    n_cond = r2.shape[0]
    col_titles = P["col_titles"]
    if col_titles is None:
        col_titles = res.get("conditions", [f"cond_{i}" for i in range(n_cond)])
    col_titles = [str(c) for c in col_titles]
    labels = list(P["training_feature_labels"])

    canon = pd.DataFrame(r2.T, index=[str(l) for l in labels], columns=col_titles)   # features x conditions
    vertical = P["orientation"] == "vertical"
    feat_axis, cond_axis = (1, 0) if vertical else (0, 1)

    mean_r2 = canon.mean(axis=1)
    rank_by_mean = mean_r2.rank(ascending=False, method="min")
    rank_in_cond = _rank_desc(canon, axis=0)
    by_cond = pd.Series(
        ["<br>   " + "<br>   ".join(f"{c}: {v:.2f}" for c, v in zip(col_titles, row)) for row in canon.to_numpy()],
        index=canon.index,
    )

    fields = {
        "feature": _f(_label_grid(plot_df, feat_axis), "Feature"),
        "condition": _f(_label_grid(plot_df, cond_axis), "Held-out condition"),
        "value": _f(plot_df.to_numpy(dtype=float), "R²", ".3f"),
        "mean_r2": _f(mean_r2, "Mean R² (all conditions)", ".3f"),
        "min_r2": _f(canon.min(axis=1), "Min R²", ".3f"),
        "max_r2": _f(canon.max(axis=1), "Max R²", ".3f"),
        "rank_by_mean": _f(rank_by_mean, f"Rank by mean R² (of {len(labels)})", ".0f"),
        "rank_in_condition": _f(rank_in_cond, "Rank in this condition", ".0f"),
        "r2_by_condition": _f(by_cond, "R² by condition"),
    }
    real = res.get("real", {})
    if "r2_overall" in real:
        fields["condition_r2_overall"] = _f(pd.Series(np.asarray(real["r2_overall"], float), index=col_titles),
                                            "Overall R² (this condition)", ".3f")
    if "rmse_overall" in real:
        fields["condition_rmse"] = _f(pd.Series(np.asarray(real["rmse_overall"], float), index=col_titles),
                                      "Overall RMSE (this condition)", ".3g")
    pv = res.get("p_values", {})
    if "r2_overall" in pv:
        fields["condition_p_value"] = _f(pd.Series(np.asarray(pv["r2_overall"], float), index=col_titles),
                                         "Permutation p (overall R²)", ".3g")
    return fields, ["feature", "condition", "value", "mean_r2", "rank_by_mean", "condition_r2_overall"]


# source: cvp_baseline_additions.py
def _style_r2(P):
    return dict(annot=P["annot"], annot_fmt=".2f", annot_fontsize=P["annot_fontsize"],
                annot_color=P["annot_color"])


# source: cvp_baseline_additions.py
def plot_sorted_r2_per_feature(*args, **kwargs):
    """
    cvp.plot_sorted_r2_per_feature with an optional interactive version.

    Hover fields: feature, condition, value (R²), mean_r2, min_r2, max_r2,
    rank_by_mean, rank_in_condition, r2_by_condition, and, when present in the
    results dict, condition_r2_overall, condition_rmse, condition_p_value.

    Note: like the original, save defaults to True.
    """
    return _dispatch(_cvp_plot_sorted_r2_per_feature, args, kwargs,
                     _fields_r2, lambda P: P["figtitle"], _style_r2)


# source: cvp_baseline_additions.py
_VP_LABELS = {
    "frac_strain": "Strain", "frac_environment": "Environment",
    "frac_interaction": "Strain x environment", "frac_residual": "Residual",
    "ss_strain": "SS strain", "ss_environment": "SS environment",
    "ss_interaction": "SS interaction", "ss_residual": "SS residual",
}


# source: cvp_baseline_additions.py
def _fields_variance(P, plot_df):
    vdf = P["variance_df"].copy()
    vdf["feature"] = vdf["feature"].astype(str)
    vdf = vdf.drop_duplicates("feature").set_index("feature")
    vertical = P["orientation"] == "vertical"
    feat_axis, comp_axis = (1, 0) if vertical else (0, 1)

    comps = ["strain", "environment", "interaction", "residual"]
    fr = vdf[[f"frac_{c}" for c in comps]]
    dominant = pd.Series(
        [comps[int(np.nanargmax(r))] if np.isfinite(r).any() else "" for r in fr.to_numpy(dtype=float)],
        index=vdf.index,
    )

    fields = {
        "feature": _f(_label_grid(plot_df, feat_axis), "Feature"),
        "component": _f(_label_grid(plot_df, comp_axis), "Component"),
        "value": _f(plot_df.to_numpy(dtype=float), "Fraction of variance", ".3f"),
        "dominant_component": _f(dominant, "Largest component"),
    }
    # every other column of variance_df (frac_*, ss_*, and anything you added) is available by name
    for col in vdf.columns:
        if col in fields:
            continue
        s = vdf[col]
        fmt = ".3f" if col.startswith("frac_") else (".4g" if pd.api.types.is_numeric_dtype(s) else None)
        fields[col] = _f(s, _VP_LABELS.get(col, col), fmt)
    if P["sort_features"] and f"frac_{P['sort_by']}" in vdf.columns:
        fields["rank"] = _f(vdf[f"frac_{P['sort_by']}"].rank(ascending=False, method="min"),
                            f"Rank by {P['sort_by']} fraction", ".0f")
    return fields, ["feature", "component", "value",
                    "frac_strain", "frac_environment", "frac_interaction", "frac_residual"]


# source: cvp_baseline_additions.py
def plot_variance_partitioning_heatmap(*args, **kwargs):
    """
    cvp.plot_variance_partitioning_heatmap with an optional interactive version.

    Hover fields: feature, component, value, dominant_component, rank, and every
    column of variance_df by its own name (frac_strain, frac_environment,
    frac_interaction, frac_residual, ss_strain, ...). Add a column to
    variance_df and it becomes a hover field automatically.
    Default: feature, component, value, and the four frac_* columns.
    """
    return _dispatch(_cvp_plot_variance_partitioning_heatmap, args, kwargs,
                     _fields_variance, lambda P: P["figtitle"])


# source: cvp_baseline_additions.py
def plot_ridge_vs_recalibrated_r2(
    results,
    models=("strain_mean_recal", "ridge"),
    model_labels=None,
    model_colors=None,
    col_titles=None,
    xtick_colors=None,
    metric="overall_r2",
    show_values=True,
    value_fmt="{:.2f}",
    value_fontsize=8,
    bar_width=0.38,
    figsize=(6, 3),
    ylabel=r"$R^2$ in held-out condition",
    ylim=None,
    legend_loc="upper left",
    save=False,
    figtitle="ridge_vs_recalibrated_strain_mean_r2",
):
    """
    Side-by-side bars of R^2 for each held-out condition, one bar per model.

    `results` is the output of compare_models_condition_holdout_cv_recalibrated.
    The default compares the recalibrated within-strain mean (strain identity
    plus each condition's shift and scale, i.e. an additive strain + environment
    model) against the full ridge model. The gap between them is what ridge
    learns beyond that additive expectation.

    Add "ridge_strain_profile" to `models` to also show the condition-averaged
    ridge control, which isolates the interaction component.

    Returns fig, ax, plot_df (long-form: condition, model, value).
    """
    for m in models:
        if m not in results:
            raise ValueError(f"{m} not found in results; available: {list(results)}")

    conditions = list(results[models[0]]["conditions"])
    if col_titles is None:
        col_titles = conditions
    if len(col_titles) != len(conditions):
        raise ValueError("col_titles length must match the number of conditions")

    default_labels = {
        "ridge": "ridge regression",
        "strain_mean_recal": "recalibrated strain mean",
        "ridge_strain_profile": "condition-averaged ridge",
        "strain_mean": "raw strain mean",
        "global_mean": "population mean",
    }
    default_colors = {
        "ridge": apc.aster,
        "strain_mean_recal": apc.gray,
        "ridge_strain_profile": apc.lapis,
        "strain_mean": apc.dress,
        "global_mean": "black",
    }
    if model_labels is None:
        model_labels = [default_labels.get(m, m) for m in models]
    if model_colors is None:
        model_colors = [default_colors.get(m, None) for m in models]

    n_models = len(models)
    x = np.arange(len(conditions))
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=figsize)
    records = []
    for k, (m, lab, col) in enumerate(zip(models, model_labels, model_colors)):
        vals = np.asarray(results[m][metric], dtype=float)
        bars = ax.bar(x + offsets[k], vals, width=bar_width, color=col, label=lab)
        if show_values:
            for b, v in zip(bars, vals):
                va = "bottom" if v >= 0 else "top"
                ax.text(b.get_x() + b.get_width() / 2, v, value_fmt.format(v),
                        ha="center", va=va, fontsize=value_fontsize)
        for cond, v in zip(conditions, vals):
            records.append({"condition": cond, "model": m, "value": v})

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(col_titles, rotation=45, ha="right")
    if xtick_colors is not None:
        for tick, c in zip(ax.get_xticklabels(), xtick_colors):
            tick.set_color(c)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        lo = min(0, min(r["value"] for r in records))
        ax.set_ylim(lo - 0.05 if lo < 0 else 0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc=legend_loc, fontsize=8)

    plt.tight_layout()
    plt.show()

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}", bbox_inches="tight")

    return fig, ax, pd.DataFrame(records)


# ==============================================================================================================
# STRAIN DEPARTURES from expected reaction norms (from cvp_departures.py)
# ==============================================================================================================

_DEFAULT_COLORS = [apc.aster, apc.dress, apc.lapis, apc.candy]


# source: cvp_departures.py
def _resolve_feature(training_feature_labels, feature):
    labels = list(training_feature_labels)
    if feature in labels:
        return labels.index(feature)
    hits = [i for i, l in enumerate(labels) if feature in l]
    if len(hits) != 1:
        raise ValueError(
            f"'{feature}' matched {len(hits)} features; pass the exact name. "
            f"Candidates: {[labels[i] for i in hits][:8]}"
        )
    return hits[0]


# source: cvp_departures.py
def _bh_fdr(p):
    """Benjamini-Hochberg q-values; NaNs are ignored and returned as NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


# source: cvp_departures.py
def _oof_predictions_grouped(training_data, exp_conditions, test_condition,
                             keep_rows, groups, ridge_alpha, n_splits):
    """
    Out-of-fold ridge predictions for every column of one held-out condition,
    fit only on `keep_rows`, with folds grouped by well. Rows outside
    `keep_rows` come back as NaN.
    """
    train_conditions = [c for c in exp_conditions if c != test_condition]
    X = np.hstack([np.asarray(training_data[c], dtype=float)[keep_rows]
                   for c in train_conditions])
    y = np.asarray(training_data[test_condition], dtype=float)[keep_rows]
    g = groups[keep_rows]

    y_pred = np.full_like(y, np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("ridge", Ridge(alpha=ridge_alpha))])
        pipe.fit(X[tr], y[tr])
        y_pred[te] = pipe.predict(X[te])

    n_rows = np.asarray(training_data[test_condition]).shape[0]
    full = np.full((n_rows, y.shape[1]), np.nan)
    full[keep_rows] = y_pred
    return full


# source: cvp_departures.py
def compute_strain_departures(
    training_data,
    exp_conditions,
    training_feature_labels,
    feature,
    ridge_alpha=1000,
    n_splits=16,
    n_wells=96,
    n_fovs=9,
    dead_ids=None,
    calibrate=True,
    n_permutations=20000,
    random_state=42,
):
    """
    Expected vs observed reaction norm for every strain, with a permutation
    test of each strain's departure in each condition.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        condition -> (n_wells * n_fovs, n_features), with rows laid out as
        well + fov * n_wells (the layout cvp.build_training_data produces).
        Pass the FULL arrays (e.g. training_data_finite); dead wells are
        dropped here via `dead_ids` so row indexing stays aligned.
    exp_conditions : list[str]
        All conditions. The model uses all of them as predictors even if you
        only plot a subset later.
    feature : str
        Feature to test, e.g. "axis_major_length (bf)".
    ridge_alpha : float
        Use the same value as the model reported in the pub.
    dead_ids : list[int] or None
        Row indices of dead wells, as returned by
        cvp.compute_dead_wells_and_ids (well + fov * n_wells).
    calibrate : bool
        Fit observed = a + b * predicted across strains within each condition
        and measure residuals against the calibrated prediction. Ridge
        compresses predictions toward the mean, which would otherwise make
        phenotypically extreme strains look like they depart for purely
        statistical reasons.
    n_permutations : int
        Size of the null per condition. With ~550 strain x condition tests,
        use at least 20,000 so the smallest p-values can survive FDR.

    Returns
    -------
    pd.DataFrame, one row per (strain, condition):
        strain, condition,
        obs_z, pred_z      strain mean, observed and expected, z-scored across
                           strains within the condition (between-strain SD units)
        sem_z              field-to-field SEM of the observed mean, same units
        resid_z            obs_z - pred_z
        z_emp, p_emp       departure relative to the rest of the collection
                           (empirical null, robust z). THIS is the test to report.
        p_perm             field-shuffle permutation p: is the departure beyond
                           field-to-field noise? Necessary but not sufficient,
                           since it ignores the model's typical per-strain error.
        q                  Benjamini-Hochberg q of p_emp across all strains x
                           conditions
        calib_slope        calibration slope used in that condition
    """
    rng = np.random.default_rng(random_state)
    feat_idx = _resolve_feature(training_feature_labels, feature)

    # Row layout matches cvp.build_training_data: row = well + fov * n_wells
    # (FoV-major). Every well therefore appears once in each block of n_wells
    # rows, and its fields sit n_wells rows apart.
    n_rows = n_wells * n_fovs
    row_well = np.arange(n_rows) % n_wells
    groups = row_well                       # CV folds grouped by well
    dead_wells = sorted({int(d) % n_wells for d in (dead_ids or [])})
    keep_rows = np.where(~np.isin(row_well, dead_wells))[0]
    alive = [w for w in range(n_wells) if w not in dead_wells]

    records = []
    for cond in exp_conditions:
        obs_rows = np.asarray(training_data[cond], dtype=float)[:, feat_idx]
        pred_rows = _oof_predictions_grouped(
            training_data, exp_conditions, cond, keep_rows, groups,
            ridge_alpha, n_splits,
        )[:, feat_idx]

        # (n_fovs, n_wells) -> (n_wells, n_fovs), then keep live wells
        obs_blk = obs_rows.reshape(n_fovs, n_wells).T[alive]
        pred_blk = pred_rows.reshape(n_fovs, n_wells).T[alive]

        obs_mean = np.nanmean(obs_blk, axis=1)
        pred_mean = np.nanmean(pred_blk, axis=1)

        # calibration across strains (strain-level, as in the calibration panel)
        a, b = 0.0, 1.0
        if calibrate:
            m = np.isfinite(obs_mean) & np.isfinite(pred_mean)
            vx = np.var(pred_mean[m])
            if vx > 0:
                b = np.cov(pred_mean[m], obs_mean[m], bias=True)[0, 1] / vx
                a = obs_mean[m].mean() - b * pred_mean[m].mean()
        pred_blk_cal = a + b * pred_blk
        pred_mean_cal = a + b * pred_mean

        # field-level residuals; a strain's departure is their mean
        resid_blk = obs_blk - pred_blk_cal
        resid_mean = np.nanmean(resid_blk, axis=1)

        # permutation null: shuffle field residuals across strains, so a
        # "strain" is just n_fovs random fields from this condition
        pool = resid_blk[np.isfinite(resid_blk)]
        pool = pool - pool.mean()       # null is centred on zero
        draws = rng.choice(pool, size=(n_permutations, n_fovs), replace=True)
        null = draws.mean(axis=1)
        abs_null = np.sort(np.abs(null))
        n_ge = n_permutations - np.searchsorted(abs_null, np.abs(resid_mean),
                                                side="left")
        p_perm = (1 + n_ge) / (1 + n_permutations)

        # empirical null: how far this strain's departure sits from the
        # departures of the rest of the collection. Robust centre and scale
        # (median, MAD) so a few true outliers don't inflate the null.
        # This absorbs the model's typical strain-specific shortfall, which
        # the field-shuffle null above cannot (on simulated additive data the
        # field-shuffle null alone gives ~12% p < 0.05).
        med = np.nanmedian(resid_mean)
        mad = 1.4826 * np.nanmedian(np.abs(resid_mean - med))
        mad = mad if mad > 0 else np.nanstd(resid_mean)
        z_emp = (resid_mean - med) / mad
        p_emp = 2 * norm.sf(np.abs(z_emp))

        # z-score in the same units as the reaction-norm figure
        mu, sd = np.nanmean(obs_mean), np.nanstd(obs_mean)
        sd = sd if sd > 0 else 1.0
        n_finite = np.isfinite(obs_blk).sum(axis=1).clip(min=1)
        sem = np.nanstd(obs_blk, axis=1, ddof=1) / np.sqrt(n_finite)

        for i, w in enumerate(alive):
            records.append({
                "strain": w,
                "condition": cond,
                "obs_z": (obs_mean[i] - mu) / sd,
                "pred_z": (pred_mean_cal[i] - mu) / sd,
                "sem_z": sem[i] / sd,
                "resid_z": resid_mean[i] / sd,
                "z_emp": z_emp[i],
                "p_emp": p_emp[i],
                "p_perm": p_perm[i],
                "calib_slope": b,
            })

    df = pd.DataFrame(records)
    df["q"] = _bh_fdr(df["p_emp"].values)
    return df


# source: cvp_departures.py
def plot_reaction_norm_vs_prediction(
    departures_df,
    strains,
    conditions=None,
    condition_labels=None,
    strain_colors=None,
    q_threshold=0.05,
    show_population=True,
    figsize=(4.6, 4.0),
    font_size=10,
    title=None,
    ylabel=None,
    save=False,
    figtitle="reaction_norm_vs_prediction",
):
    """
    Observed reaction norm (solid, +/- SEM) and the collection's expectation
    (dashed) for one or more strains. Conditions where the departure passes
    FDR are marked with an asterisk above the observed point.

    Gray lines are every other strain's observed reaction norm, for scale.
    """
    df = departures_df
    if conditions is None:
        conditions = list(dict.fromkeys(df["condition"]))
    if condition_labels is None:
        condition_labels = conditions
    if strain_colors is None:
        strain_colors = _DEFAULT_COLORS
    strains = list(strains)
    xs = np.arange(len(conditions))

    wide = {col: df.pivot(index="strain", columns="condition", values=col)
                   .reindex(columns=conditions)
            for col in ("obs_z", "pred_z", "sem_z", "q")}

    fig, ax = plt.subplots(figsize=figsize)

    if show_population:
        for s, row in wide["obs_z"].iterrows():
            if s in strains or not np.all(np.isfinite(row.values)):
                continue
            ax.plot(xs, row.values, "-", color="#B8B8B8", lw=0.6, alpha=0.35,
                    zorder=0)

    for k, s in enumerate(strains):
        c = strain_colors[k % len(strain_colors)]
        obs = wide["obs_z"].loc[s].values
        pred = wide["pred_z"].loc[s].values
        sem = wide["sem_z"].loc[s].values
        q = wide["q"].loc[s].values

        ax.plot(xs, pred, "--", color=c, lw=1.5, alpha=0.9, zorder=2,
                label=f"strain {s}, expected")
        ax.errorbar(xs, obs, yerr=sem, fmt="o-", color=c, lw=2, ms=8,
                    capsize=3, elinewidth=1.2, markeredgecolor="white",
                    markeredgewidth=1.2, zorder=3, label=f"strain {s}, observed")
        # thin connector from expectation to observation
        for x, o, p in zip(xs, obs, pred):
            ax.plot([x, x], [p, o], color=c, lw=0.8, alpha=0.5, zorder=1)
        for x, o, e, qq in zip(xs, obs, sem, q):
            if np.isfinite(qq) and qq < q_threshold:
                ax.annotate("*", xy=(x, o + (e if np.isfinite(e) else 0)),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=font_size + 4,
                            color=c)

    ax.axhline(0, color="#CCCCCC", lw=0.8, ls="--", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(condition_labels, fontsize=font_size - 1)
    ax.set_xlim(-0.45, len(conditions) - 0.55)
    ax.set_ylabel(ylabel or "z-scored within each condition",
                  fontsize=font_size - 1)
    if title:
        ax.set_title(title, fontsize=font_size)
    ax.legend(fontsize=font_size - 2, frameon=False, loc="best")
    ax.tick_params(labelsize=font_size - 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.show()

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig, ax


# ==============================================================================================================
# PLATE POSITION CORRECTION + STRAIN MEANS for the WT comparison (from cvp_diversity.py)
# ==============================================================================================================


# source: cvp_diversity.py
def get_strain_means(
    training_data,
    condition,
    n_wells=96,
    n_fovs=9,
    dead_wells=None,
):
    """
    Compute the mean feature vector per strain (well) for one condition.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Output of cvp.build_training_data().
    condition : str
        Key in training_data to use.
    n_wells : int
        Number of wells per plate.
    n_fovs : int or None
        Number of fields of view per well.  If None, inferred from row count.
    dead_wells : list[int] or None
        Well indices to exclude.

    Returns
    -------
    strain_means : np.ndarray, shape (n_valid_wells, n_features)
    valid_well_ids : list[int]
    """
    if dead_wells is None:
        dead_wells = []

    dat = np.asarray(training_data[condition], dtype=float)
    n_rows, n_features = dat.shape

    if n_fovs is None:
        n_fovs = n_rows // n_wells

    strain_means = []
    valid_well_ids = []

    for well in range(n_wells):
        if well in dead_wells:
            continue
        row_idx = [well + fov * n_wells for fov in range(n_fovs)]
        block = dat[row_idx, :]
        if np.all(~np.isfinite(block)):
            continue
        mean_vec = np.nanmean(block, axis=0)
        strain_means.append(mean_vec)
        valid_well_ids.append(well)

    return np.vstack(strain_means), valid_well_ids


# source: cvp_diversity.py
def correct_plate_position_effects(
    training_data,
    n_wells=96,
    n_rows=8,
    n_cols=12,
    dead_wells=None,
    n_fovs=9,
    n_iter=10,
):
    """
    Remove additive plate row + column effects from feature data.

    Systematic plate-position effects (illumination gradients, temperature
    gradients, edge effects) produce reproducible between-well differences that
    are unrelated to biology.  Because these effects are stable across FoVs,
    a classifier trained to predict well identity can exploit them to achieve
    high accuracy even when all strains are phenotypically identical (wild type).

    This function estimates row and column effects from the per-well mean feature
    vectors and subtracts them from every FoV observation.  After correction the
    classifier is forced to use only biology — not plate geometry — to identify
    wells.

    Well indexing convention (standard 96-well plate):
        row = well_index // n_cols      (0 = row A, 7 = row H)
        col = well_index  % n_cols      (0 = col 1, 11 = col 12)

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Preprocessed feature matrix (e.g., training_data_hu_log).
        Each value has shape (n_wells * n_fovs, n_features).
    n_wells : int
    n_rows, n_cols : int
        Plate dimensions.  Default 8 × 12 for a standard 96-well plate.
    dead_wells : list[int] or None
        Wells excluded from effect estimation and correction.
    n_fovs : int or None
        FoVs per well.  Inferred from row count if None.
    n_iter : int
        Alternating demeaning iterations.  Converges in ≤ 5 steps for a
        balanced plate; 10 is conservative.

    Returns
    -------
    corrected_data   : dict[str, np.ndarray]   same shape as training_data
    position_effects : dict[str, np.ndarray]   shape (n_wells, n_features)
        The estimated position component subtracted from each well.
        Dead wells hold zeros.
    """
    if dead_wells is None:
        dead_wells = set()
    else:
        dead_wells = set(dead_wells)

    valid_wells = [w for w in range(n_wells) if w not in dead_wells]
    well_row = np.array([w // n_cols for w in range(n_wells)])
    well_col = np.array([w  % n_cols for w in range(n_wells)])

    corrected_data   = {}
    position_effects = {}

    for cond, dat in training_data.items():
        dat = np.asarray(dat, dtype=float)
        n_total_rows, n_features = dat.shape
        this_n_fovs = n_total_rows // n_wells if n_fovs is None else n_fovs

        # ── Step 1: per-well means (average over FoVs) ────────────────────────
        well_means = np.full((n_wells, n_features), np.nan)
        for well in valid_wells:
            row_idx = [well + fov * n_wells for fov in range(this_n_fovs)]
            well_means[well] = np.nanmean(dat[row_idx, :], axis=0)

        # ── Step 2: iterative alternating row / column demeaning ──────────────
        # After convergence: residuals[w] = well_means[w] - row_effect - col_effect
        residuals = well_means.copy()

        for _ in range(n_iter):
            # subtract row means from valid wells
            for r in range(n_rows):
                wells_r = [w for w in valid_wells if well_row[w] == r]
                if wells_r:
                    rm = np.nanmean(residuals[wells_r], axis=0)
                    for w in wells_r:
                        residuals[w] -= rm

            # subtract column means from valid wells
            for c in range(n_cols):
                wells_c = [w for w in valid_wells if well_col[w] == c]
                if wells_c:
                    cm = np.nanmean(residuals[wells_c], axis=0)
                    for w in wells_c:
                        residuals[w] -= cm

        # position effect = the part of well_means explained by row + col
        pos_eff = np.zeros((n_wells, n_features))
        for w in valid_wells:
            pos_eff[w] = well_means[w] - residuals[w]

        # ── Step 3: subtract position effect from every FoV observation ───────
        corrected_dat = dat.copy()
        for well in valid_wells:
            row_idx = [well + fov * n_wells for fov in range(this_n_fovs)]
            corrected_dat[row_idx, :] -= pos_eff[well]

        corrected_data[cond]   = corrected_dat
        position_effects[cond] = pos_eff

    return corrected_data, position_effects


# ==============================================================================================================
# Z-SCORED t-SNE AND PCA TRAJECTORIES (from cvp_standardized.py)
# ==============================================================================================================


# source: cvp_standardized.py
def zscore_columns(X, mean=None, sd=None):
    """
    Z-score each column of X. Returns (Xz, mean, sd) and never modifies X.

    Columns with zero or undefined spread are set to 0 instead of dividing by
    zero. Pass `mean` and `sd` to apply an existing scaling to new data.
    """
    X = np.array(X, dtype=float, copy=True)
    if mean is None:
        mean = np.nanmean(X, axis=0)
    if sd is None:
        sd = np.nanstd(X, axis=0)
    sd_safe = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    Xz = (X - mean) / sd_safe
    Xz[:, ~(np.isfinite(sd) & (sd > 0))] = 0.0
    return Xz, mean, sd


# source: cvp_standardized.py
def tsne_standardized(X, perplexity=200, random_state=28, init="random",
                      learning_rate="auto", n_components=2, **tsne_kwargs):
    """
    t-SNE on z-scored features. Rows of the output line up with rows of X.

    `random_state` makes the embedding reproducible. (The notebook's
    random.seed(28) seeds Python's `random`, not scikit-learn, so the old
    embedding changes from run to run.)
    """
    Xz, _, _ = zscore_columns(X)
    return TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init=init,
        learning_rate=learning_rate,
        random_state=random_state,
        **tsne_kwargs,
    ).fit_transform(Xz)


# source: cvp_standardized.py
def standardize_training_data(training_data, conditions=None, n_wells=96,
                              n_fovs=None, dead_ids=None):
    """
    Z-score every feature using one mean and SD pooled across all live rows of
    all `conditions`, then apply that same scaling to every condition.

    Pooling across conditions (rather than z-scoring within each condition)
    keeps the differences BETWEEN conditions, which is what the trajectory plot
    is meant to show. Dead wells (row = well + fov * n_wells) are excluded
    from the mean and SD.

    Returns (standardized_dict, mean, sd).
    """
    if conditions is None:
        conditions = list(training_data.keys())
    n_rows = np.asarray(training_data[conditions[0]]).shape[0]
    if n_fovs is None:
        n_fovs = n_rows // n_wells
    dead_wells = sorted({int(d) % n_wells for d in (dead_ids or [])})
    live = ~np.isin(np.arange(n_rows) % n_wells, dead_wells)

    pooled = np.vstack([np.asarray(training_data[c], dtype=float)[live]
                        for c in conditions])
    _, mean, sd = zscore_columns(pooled)

    out = {c: zscore_columns(training_data[c], mean=mean, sd=sd)[0]
           for c in training_data}
    return out, mean, sd


# source: cvp_standardized.py
def plot_strain_pca_trajectories_standardized(training_data, exp_conditions,
                                              strains_to_plot, pca_conditions=None,
                                              n_wells=96, n_fovs=None, dead_ids=None,
                                              figtitle="strain_pca_trajectories_zscored",
                                              **kwargs):
    """
    cvp.plot_strain_pca_trajectories on z-scored features.

    Takes the same arguments as the original. Features are z-scored with one
    mean and SD pooled across `pca_conditions` (or all conditions), then the
    original plotting function runs unchanged. The default figtitle ends in
    "_zscored" so the raw-feature figures aren't overwritten.
    """
    std_conds = pca_conditions or list(training_data.keys())
    std_data, _, _ = standardize_training_data(
        training_data, conditions=std_conds, n_wells=n_wells,
        n_fovs=n_fovs, dead_ids=dead_ids,
    )
    return plot_strain_pca_trajectories(
        training_data=std_data,
        exp_conditions=exp_conditions,
        strains_to_plot=strains_to_plot,
        pca_conditions=pca_conditions,
        n_wells=n_wells,
        n_fovs=n_fovs,
        dead_ids=dead_ids,
        figtitle=figtitle,
        **kwargs,
    )


# ==============================================================================================================
# WILD-TYPE COMPARISON  (moved here from mutant_vs_wt_per_strain.ipynb; math unchanged)
# ==============================================================================================================
#
# The WT plate (wt_e22) was imaged on a different day from the mutant plates, so
# the two are fully confounded with imaging batch. We use the median-strain
# assumption: the median well on a UV-mutagenized plate carries no effective
# mutation, so the per-feature batch offset is
#     b_k = median_j(x_mut,jk) - median_i(x_wt,ik)
# and is subtracted from every mutant well. Each mutant strain is then z-scored
# against the WT strain distribution, and its Euclidean distance from the WT
# centroid (the origin in that z-score space) is its phenotypic distance from WT.

def batch_correct_median(mut_means, wt_means):
    """
    Remove additive plate-level batch offset.

    Estimates the offset as median(mut_plate) - median(wt_plate) per feature
    and subtracts it from every mutant well.

    Parameters
    ----------
    mut_means : ndarray (n_mut_wells, n_features)
    wt_means  : ndarray (n_wt_wells,  n_features)

    Returns
    -------
    corrected    : ndarray (n_mut_wells, n_features)
    batch_offset : ndarray (n_features,)  -- the estimated offset
    """
    batch_offset = np.nanmedian(mut_means, axis=0) - np.nanmedian(wt_means, axis=0)
    return mut_means - batch_offset, batch_offset


def batch_correct_to_wt(strain_means, wt_key="wt_e22", mutant_keys=("mut_e22", "mut_l22"), verbose=True):
    """
    Apply batch_correct_median to every mutant condition. The WT plate is the
    reference and is returned unchanged.

    Returns (strain_means_bc, batch_offsets).
    """
    wt_means = strain_means[wt_key]
    strain_means_bc = {wt_key: wt_means.copy()}
    batch_offsets = {}
    for mut_key in mutant_keys:
        corrected, offset = batch_correct_median(strain_means[mut_key], wt_means)
        strain_means_bc[mut_key] = corrected
        batch_offsets[mut_key] = offset
        if verbose:
            print(f"  {mut_key}: mean |batch offset| = {np.abs(offset).mean():.4f}  "
                  f"(max = {np.abs(offset).max():.4f})")
    return strain_means_bc, batch_offsets


def zscore_vs_wt(means_bc, mu_wt, sigma_wt):
    """z_jk = (x_jk - mu_wt,k) / sigma_wt,k   (sigma_wt uses ddof=1)."""
    return (means_bc - mu_wt) / (sigma_wt + 1e-8)


def compute_wt_zscores_and_distances(strain_means_bc, wt_key="wt_e22"):
    """
    Z-score every condition against the WT strain distribution and return each
    strain's Euclidean distance from the WT centroid (= origin in z-score space).

    Returns (z_scores, distances, mu_wt, sigma_wt); z_scores and distances are
    dicts keyed like strain_means_bc.
    """
    mu_wt = np.nanmean(strain_means_bc[wt_key], axis=0)
    sigma_wt = np.nanstd(strain_means_bc[wt_key], axis=0, ddof=1)
    z_scores = {c: zscore_vs_wt(m, mu_wt, sigma_wt) for c, m in strain_means_bc.items()}
    distances = {c: np.sqrt(np.nansum(z ** 2, axis=1)) for c, z in z_scores.items()}
    return z_scores, distances, mu_wt, sigma_wt


def summarize_wt_distances(distances, group_labels, cond_order=("wt_e22", "mut_e22", "mut_l22"),
                           wt_key="wt_e22"):
    """Summary table of the per-strain distances (one row per group)."""
    rows = []
    for cond in cond_order:
        d = distances[cond]
        rows.append({
            "condition": cond,
            "label": group_labels[cond],
            "n_strains": len(d),
            "mean_dist": np.nanmean(d),
            "median_dist": np.nanmedian(d),
            "std_dist": np.nanstd(d, ddof=1),
            "frac_gt2x_wt_median": np.mean(d > 2 * np.nanmedian(distances[wt_key])),
        })
    return pd.DataFrame(rows).set_index("condition")


def plot_wt_distance(distances, group_labels, group_colors,
                     cond_order=("wt_e22", "mut_e22", "mut_l22"), wt_key="wt_e22",
                     figsize=(6, 5.5), random_state=42, save=True,
                     figtitle="per_strain_wt_distance"):
    """
    Strip + box plot of per-strain distance from the WT centroid, with two-sided
    Mann-Whitney U tests of each mutant group against WT.

    Returns fig, ax, stats_df (one row per mutant-vs-WT test).
    """
    rng = np.random.default_rng(random_state)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    positions = list(range(len(cond_order)))

    for pos, cond in zip(positions, cond_order):
        d = distances[cond]
        color = group_colors[cond]
        ax.boxplot(
            d, positions=[pos], widths=0.35,
            patch_artist=True,
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(color="#555555"),
            capprops=dict(color="#555555"),
            flierprops=dict(marker="o", markersize=3, alpha=0.4,
                            markerfacecolor=color, markeredgecolor="none"),
            boxprops=dict(facecolor=color, alpha=0.55, edgecolor="none"),
        )
        jitter = rng.normal(0, 0.07, size=len(d))
        ax.scatter(pos + jitter, d, color=color, s=20, alpha=0.55,
                   edgecolors="white", linewidths=0.3, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([group_labels[c] for c in cond_order], fontsize=10)
    ax.set_ylabel("Euclidean distance from WT centroid\n(z-score space)", fontsize=10)
    ax.set_title("Per-strain distance from average WT phenotype", fontsize=11, fontweight="bold")

    test_rows = []
    mutant_conds = [c for c in cond_order if c != wt_key]
    for i, mut_cond in enumerate(mutant_conds):
        stat, pval = stats.mannwhitneyu(distances[wt_key], distances[mut_cond], alternative="two-sided")
        test_rows.append({"comparison": f"{mut_cond} vs {wt_key}", "U": stat, "p": pval,
                          "n_wt": len(distances[wt_key]), "n_mut": len(distances[mut_cond])})
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
        y_top = max(distances[wt_key].max(), distances[mut_cond].max()) * 1.05
        x0, x1 = cond_order.index(wt_key), cond_order.index(mut_cond)
        ax.plot([x0, x0, x1, x1],
                [y_top + i * 3, y_top + i * 3 + 1.5, y_top + i * 3 + 1.5, y_top + i * 3],
                lw=1.2, color="#444444")
        ax.text((x0 + x1) / 2, y_top + i * 3 + 1.7, f"{sig}\np={pval:.2e}",
                ha="center", va="bottom", fontsize=8.5, color="#333333")

    plt.show()
    if save:
        for ext in ("svg", "pdf", "png"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}", dpi=150, bbox_inches="tight")
    return fig, ax, pd.DataFrame(test_rows)


def plot_wt_batch_correction_pca(strain_means, strain_means_bc, conditions, group_labels,
                                 group_colors, n_std=1.5, figsize=(13, 5.5), save=True,
                                 figtitle="per_strain_pca_batch_correction"):
    """
    Diagnostic: PCA (on standardized strain means) before vs after the
    between-plate batch correction, with a 1.5-SD covariance ellipse per group.
    """
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA

    def _panel(ax, means_dict, title):
        combined = np.vstack([means_dict[c] for c in conditions])
        pcs_model = PCA(n_components=2)
        pcs = pcs_model.fit_transform(StandardScaler().fit_transform(combined))
        idx = 0
        for cond in conditions:
            n = means_dict[cond].shape[0]
            pts = pcs[idx:idx + n]
            color = group_colors[cond]
            ax.scatter(pts[:, 0], pts[:, 1], c=color, s=18, alpha=0.65,
                       label=group_labels[cond], edgecolors="white", linewidths=0.3)
            vals, vecs = np.linalg.eigh(np.cov(pts.T))
            angle = np.degrees(np.arctan2(*vecs[:, -1][::-1]))
            w, h = 2 * n_std * np.sqrt(vals[::-1])
            ax.add_patch(Ellipse(xy=pts.mean(axis=0), width=w, height=h, angle=angle,
                                 color=color, alpha=0.12, zorder=0))
            idx += n
        ax.set_xlabel(f"PC1 ({pcs_model.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC2 ({pcs_model.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.8)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    _panel(axes[0], strain_means, "Before batch correction")
    _panel(axes[1], strain_means_bc, "After batch correction")
    plt.show()
    if save:
        for ext in ("svg", "pdf", "png"):
            fig.savefig(f"plots/{figtitle}.{ext}", dpi=150, bbox_inches="tight")
    return fig, axes


# ==============================================================================================================
# RIDGE ALPHA SCAN  (moved here from explore_segmented_strains.ipynb)
# ==============================================================================================================

def ridge_alpha_scan(training_data, exp_conditions, alphas, n_splits=16, grouped=False,
                     groups=None, verbose=True):
    """
    Overall held-out-condition R^2 of the ridge model for each alpha.

    grouped=False reproduces the original scan: compare_models_condition_holdout_cv
    with shuffled KFold over all rows (fields of the same well can land in both
    train and test folds).
    grouped=True uses compare_models_condition_holdout_cv_recalibrated with
    GroupKFold on `groups` (well ids), i.e. the same CV scheme as the final model.

    Returns a DataFrame: one row per alpha, one column per condition (overall R^2),
    plus "mean_r2" and "median_feature_r2".
    """
    rows = []
    for a in alphas:
        if grouped:
            res = compare_models_condition_holdout_cv_recalibrated(
                training_data, exp_conditions=exp_conditions, ridge_alpha=a,
                n_splits=n_splits, groups=groups)
        else:
            res = compare_models_condition_holdout_cv(
                training_data=training_data, exp_conditions=exp_conditions,
                ridge_alpha=a, n_splits=n_splits)
        r = res["ridge"]
        row = {"alpha": a}
        row.update(dict(zip(exp_conditions, np.asarray(r["overall_r2"], dtype=float))))
        row["mean_r2"] = float(np.mean(r["overall_r2"]))
        row["median_feature_r2"] = float(np.mean(np.median(r["r2_per_feature"], axis=1)))
        rows.append(row)
        if verbose:
            print(f"alpha = {a:>7}  mean overall R2 = {row['mean_r2']:.3f}  "
                  f"(mean of per-condition median feature R2 = {row['median_feature_r2']:.3f})")
    return pd.DataFrame(rows)


def plot_alpha_scan(scan_df, exp_conditions, condition_colors, col_titles=None,
                    chosen_alpha=None, figsize=(10, 6), save=True,
                    figtitle="alpha_scan_conditional_cross_validated_holdout_16split"):
    """One line per held-out condition: overall R^2 vs alpha (log2 axis)."""
    fig, ax = plt.subplots(figsize=figsize)
    labels = col_titles if col_titles is not None else exp_conditions
    for cond, color, lab in zip(exp_conditions, condition_colors, labels):
        ax.plot(scan_df["alpha"], scan_df[cond], c=color, marker="o", label=lab)
    ax.set_xscale("log", base=2)
    if chosen_alpha is not None:
        ax.axvline(chosen_alpha, color="grey", ls="--", lw=1)
    ax.set_xlabel("ridge alpha")
    ax.set_ylabel("R$^2$ (mean of all features)")
    ax.legend(frameon=False, fontsize=8)
    plt.show()
    if save:
        for ext in ("svg", "png", "pdf"):
            fig.savefig(f"plots/{figtitle}.{ext}", bbox_inches="tight")
    return fig, ax


# ==============================================================================================================
# SMALL NOTEBOOK HELPERS
# ==============================================================================================================

def stack_live_rows(training_data, exp_conditions, dead_ids):
    """Stack conditions (in order) into one matrix with dead-well rows removed (the notebook's `all_dat`)."""
    return np.vstack([remove_rows_cols(training_data[c], row_indices=dead_ids, col_indices=[])
                      for c in exp_conditions])


def safe_feature_name(label):
    """'moments_hu-0 (bf)' -> 'moments_hu_0_bf' (used in figure file names)."""
    s = (label.replace(" ", "_").replace("(", "").replace(")", "")
         .replace("/", "_").replace("-", "_").replace(",", "").lower())
    while "__" in s:
        s = s.replace("__", "_")
    return s


def zscored_oof_predictions(training_data, exp_conditions, ridge_alpha, n_splits, groups,
                            n_wells, n_fovs, dead_ids, random_state=42):
    """
    Grouped out-of-fold ridge predictions for every held-out condition, with dead
    wells dropped and each feature z-scored within condition using the OBSERVED
    mean and SD. Returns a list of [observed_z, predicted_z] per condition, the
    input format of plot_prediction_displacements.
    """
    oof = _oof_by_condition(training_data, exp_conditions, ridge_alpha=ridge_alpha,
                            n_splits=n_splits, groups=groups, random_state=random_state)
    row_well = np.arange(n_wells * n_fovs) % n_wells          # row = well + fov * n_wells
    dead_wells = sorted({int(d) % n_wells for d in dead_ids})
    live = ~np.isin(row_well, dead_wells)
    out = []
    for cond in exp_conditions:
        obs, pred = oof[cond]
        obs, pred = obs[live], pred[live]
        mu, sd = obs.mean(axis=0), obs.std(axis=0)
        sd[sd == 0] = 1.0
        out.append([(obs - mu) / sd, (pred - mu) / sd])
    return out


# source: generated from plot_strain_pair_cell_grid (cvp_baseline_additions.py)
def plot_strain_cell_grid(
    zroots,
    strains,
    experiment_indices,
    condition_labels,
    channel="CY5",
    fov=0,
    n_cells=6,
    crop=96,
    cells_per_row=None,
    cmap="gray",
    strain_colors=None,
    row_labels=None,
    row_label_position="left",     # "left" or "inside"
    shared_intensity=True,
    intensity_percentiles=(1.0, 99.5),
    um_per_pixel=None,
    scalebar_um=10,
    random_state=0,
    panel_size=1.7,
    font_size=9,
    save=False,
    figtitle="strain_cell_grid",
):
    """
    Same as plot_strain_pair_cell_grid, for any number of strains (one row each).
    Added for the pub notebook; the image handling is unchanged.
    """
    rng = np.random.default_rng(random_state)
    n_cond = len(experiment_indices)
    if len(condition_labels) != n_cond:
        raise ValueError(
            f"condition_labels has {len(condition_labels)} entries but "
            f"experiment_indices has {n_cond}."
        )
    if strain_colors is None:
        strain_colors = [apc.aster, apc.dress, apc.lapis, apc.candy, apc.vital, apc.wish]
    strains = [int(s) for s in strains]
    strain_colors = [strain_colors[i % len(strain_colors)] for i in range(len(strains))]
    if row_labels is None:
        row_labels = [f"strain {s}" for s in strains]

    # tile the cells into a compact block instead of one long strip, so the
    # panel is not 5:1 and the figure has no dead vertical space
    ncols = cells_per_row or int(np.ceil(np.sqrt(n_cells)))
    nrows = int(np.ceil(n_cells / ncols))

    def _tile(crops):
        if not crops:
            return None
        pad = [np.zeros_like(crops[0])] * (nrows * ncols - len(crops))
        cells = (crops + pad)[:nrows * ncols]
        return np.vstack([
            np.hstack(cells[i * ncols:(i + 1) * ncols]) for i in range(nrows)
        ])

    # ---- gather every crop first, so the intensity range can be shared ----
    panels = {}
    for r, strain in enumerate(strains):
        for c, exp_i in enumerate(experiment_indices):
            crops = _representative_cell_crops(
                zroots[exp_i], fov, strain, channel, n_cells, crop, rng,
                label=f"strain {strain}, {condition_labels[c]}",
            )
            panels[(r, c)] = _tile(crops)

    vmin = vmax = None
    if shared_intensity:
        allvals = np.concatenate([
            p.ravel() for p in panels.values() if p is not None
        ]) if any(p is not None for p in panels.values()) else None
        if allvals is not None and allvals.size:
            vmin, vmax = np.percentile(allvals, intensity_percentiles)

    panel_h = panel_size
    panel_w = panel_size * ncols / nrows          # match the tile's aspect
    fig, axes = plt.subplots(
        len(strains), n_cond,
        figsize=(panel_w * n_cond + 0.9, panel_h * len(strains) + 0.7),
        squeeze=False,
    )

    for r in range(len(strains)):
        for c in range(n_cond):
            ax = axes[r][c]
            strip = panels[(r, c)]
            if strip is not None:
                ax.imshow(strip, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax.text(0.5, 0.5, "no cells", ha="center", va="center",
                        transform=ax.transAxes, fontsize=font_size - 1,
                        color="#999999")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(strain_colors[r]); sp.set_linewidth(2.0)

            if r == 0:
                ax.set_title(condition_labels[c], fontsize=font_size)

            if c == 0:
                if row_label_position == "left":
                    ax.set_ylabel(row_labels[r], fontsize=font_size,
                                  color=strain_colors[r], labelpad=6)
                else:
                    ax.text(0.02, 0.96, row_labels[r], transform=ax.transAxes,
                            ha="left", va="top", fontsize=font_size - 1,
                            color="white",
                            bbox=dict(facecolor=strain_colors[r], edgecolor="none",
                                      boxstyle="round,pad=0.25", alpha=0.85))

            # scale bar in the last panel of the bottom row only
            if um_per_pixel and r == len(strains) - 1 and c == n_cond - 1 and strip is not None:
                bar_px = scalebar_um / um_per_pixel
                h, w = strip.shape[:2]
                y = h * 0.92
                ax.plot([w - bar_px - w * 0.04, w - w * 0.04], [y, y],
                        color="white", lw=3, solid_capstyle="butt")
                ax.text(w - bar_px / 2 - w * 0.04, y - h * 0.06,
                        f"{scalebar_um} µm", color="white", ha="center",
                        va="bottom", fontsize=font_size - 2)

    plt.tight_layout()
    plt.show()

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)

    return fig, axes


# ==============================================================================================================
# WILD-TYPE PLATE AS A POSITION-ONLY REFERENCE
# ==============================================================================================================
#
# Every mutant plate uses the same layout, so a strain and its well position are
# perfectly confounded: a gradient shared by all plates looks exactly like a
# strain main effect. The wild-type plate breaks the confound. All of its wells
# carry the same genotype, so any row/column structure on it is position alone.

def _rc_surface_fraction(values, rr, cc, plate_rows, plate_cols):
    """Median-polish one plate. Returns (frac_position, fitted surface at every
    plate position as a (plate_rows, plate_cols) grid), or (nan, None)."""
    grid = np.full((plate_rows, plate_cols), np.nan)
    grid[rr, cc] = values
    total = np.nanvar(grid)
    if not np.isfinite(total) or total <= 0:
        return np.nan, None
    _, reff, ceff, resid = _median_polish(grid)
    return float(1 - np.nanvar(resid) / total), reff[:, None] + ceff[None, :]


def _r2_from_reference(target, reference):
    """R^2 of a least-squares line predicting `target` from `reference`."""
    ok = np.isfinite(target) & np.isfinite(reference)
    if ok.sum() < 3 or np.std(reference[ok]) == 0 or np.std(target[ok]) == 0:
        return np.nan
    return float(np.corrcoef(target[ok], reference[ok])[0, 1] ** 2)


def wt_plate_gradient_reference(
    mutant_data, wt_data, exp_conditions, training_feature_labels,
    n_wells=96, n_fovs=9, dead_ids=None, wt_dead_wells=None,
    plate_rows=8, plate_cols=12, layout="row_major",
    mutant_reference_condition=None, n_null=200, random_state=0,
):
    """
    Use the wild-type plate to measure plate position on its own, then ask how
    much of the mutant strain differences that position could account for.

    Per feature:

    wt_frac_position / wt_null_frac_position / wt_excess_position
        Fraction of the WT plate's well-to-well variance taken by additive row
        and column effects (median polish), the same quantity under a null that
        shuffles WT wells across plate positions, and the difference. The null
        matters: an 8 x 12 row + column fit absorbs some variance even when
        there is no spatial structure. Excess > 0 means the WT plate has a real
        positional gradient.

    mut_frac_position
        The same fraction for the mutant strain main effect (each condition
        centred, then averaged across `exp_conditions`). This is confounded with
        genotype; it is reported only for comparison with the WT value.

    surface_r
        Pearson correlation, across live wells, between the WT fitted surface and
        the mutant fitted surface. Positive and large means the mutant plates
        carry the same gradient as the WT plate.

    r2_mut_from_wt_surface / null_r2_mut_from_wt_surface / excess_r2_mut_from_wt_surface
        Fraction of the mutant strain main effect predicted by the WT surface
        alone (a straight-line fit, so differences in scale between plates do
        not matter), the null obtained by shuffling the WT wells before fitting
        the surface, and the difference. This is the unconfounded estimate of
        how much apparent strain variation is plate position.

    null_r2_95 / p_r2 / q_r2
        95th percentile of the shuffled R² values (the per-feature threshold for
        p < 0.05 before correction), a one-sided permutation p-value
        (1 + #null >= observed) / (1 + n_null), and Benjamini-Hochberg q-values
        across features. With n_null shuffles the smallest attainable p is
        1 / (n_null + 1), so use at least 1,000 when correcting across 40
        features.

    r2_ref_from_wt_surface
        As above but for one mutant condition (`mutant_reference_condition`,
        e.g. the plate grown and imaged like the WT plate), if given.

    Caveat: the WT plate was grown and imaged on a different day from the
    mutant plates. A gradient that changed between runs (for example, a plate
    moved to another shelf) would lower these numbers without meaning the
    mutant plates are free of position effects.
    """
    mut_dead = {int(d) % n_wells for d in (dead_ids or [])}
    wt_dead = {int(w) for w in (wt_dead_wells or [])}
    wt_alive = np.array([w for w in range(n_wells) if w not in wt_dead])
    both_alive = np.array([w for w in range(n_wells) if w not in wt_dead | mut_dead])
    rr, cc = _well_to_rc(np.arange(n_wells), plate_rows, plate_cols, layout)

    def _means(M):
        M = np.asarray(M, dtype=float)
        return np.array([np.nanmean(M[_well_rows(w, n_wells, n_fovs)], axis=0)
                         for w in range(n_wells)])

    wt_means = _means(wt_data)
    mut_means = {c: _means(mutant_data[c]) for c in exp_conditions}

    rng = np.random.default_rng(random_state)
    rows = []
    for j, name in enumerate(training_feature_labels):
        # WT plate: position only
        wt_vals = np.full(n_wells, np.nan)
        wt_vals[wt_alive] = wt_means[wt_alive, j]
        wt_frac, wt_surf = _rc_surface_fraction(wt_vals, rr, cc, plate_rows, plate_cols)
        if wt_surf is None:
            continue

        # mutant strain main effect (environment main effect removed per condition)
        centred = np.vstack([mut_means[c][:, j] - np.nanmean(mut_means[c][both_alive, j])
                             for c in exp_conditions])
        shared = np.full(n_wells, np.nan)
        shared[both_alive] = centred[:, both_alive].mean(axis=0)
        mut_frac, mut_surf = _rc_surface_fraction(shared, rr, cc, plate_rows, plate_cols)

        wt_at_wells = wt_surf[rr[both_alive], cc[both_alive]]
        surface_r = (float(np.corrcoef(wt_at_wells, mut_surf[rr[both_alive], cc[both_alive]])[0, 1])
                     if mut_surf is not None and np.std(wt_at_wells) > 0 else np.nan)
        r2 = _r2_from_reference(shared[both_alive], wt_at_wells)

        r2_ref = np.nan
        if mutant_reference_condition is not None:
            r2_ref = _r2_from_reference(mut_means[mutant_reference_condition][both_alive, j],
                                        wt_at_wells)

        # null: shuffle WT wells across plate positions, refit
        null_frac, null_r2 = [], []
        for _ in range(n_null):
            shuffled = np.full(n_wells, np.nan)
            shuffled[wt_alive] = wt_vals[rng.permutation(wt_alive)]
            f, s = _rc_surface_fraction(shuffled, rr, cc, plate_rows, plate_cols)
            if s is None:
                continue
            null_frac.append(f)
            null_r2.append(_r2_from_reference(shared[both_alive], s[rr[both_alive], cc[both_alive]]))
        null_frac_m = float(np.nanmean(null_frac)) if null_frac else np.nan
        null_r2 = np.array(null_r2, dtype=float)
        null_r2 = null_r2[np.isfinite(null_r2)]
        null_r2_m = float(null_r2.mean()) if null_r2.size else np.nan
        null_r2_95 = float(np.quantile(null_r2, 0.95)) if null_r2.size else np.nan
        p_r2 = (float((1 + np.sum(null_r2 >= r2)) / (1 + null_r2.size))
                if null_r2.size and np.isfinite(r2) else np.nan)

        rows.append({
            "feature": name,
            "wt_frac_position": wt_frac,
            "wt_null_frac_position": null_frac_m,
            "wt_excess_position": wt_frac - null_frac_m,
            "wt_position_p": float((1 + np.sum(np.array(null_frac) >= wt_frac)) / (1 + len(null_frac))),
            "mut_frac_position": mut_frac,
            "surface_r": surface_r,
            "r2_mut_from_wt_surface": r2,
            "null_r2_mut_from_wt_surface": null_r2_m,
            "excess_r2_mut_from_wt_surface": r2 - null_r2_m,
            "null_r2_95": null_r2_95,
            "p_r2": p_r2,
            "r2_ref_from_wt_surface": r2_ref,
        })
    out = pd.DataFrame(rows)
    out["q_r2"] = np.nan
    ok = out.p_r2.notna()
    if ok.any():
        from statsmodels.stats.multitest import multipletests
        out.loc[ok, "q_r2"] = multipletests(out.loc[ok, "p_r2"], method="fdr_bh")[1]
    return out


def plot_wt_vs_mutant_plate(
    mutant_data, wt_data, exp_conditions, feature_index, training_feature_labels,
    n_wells=96, n_fovs=9, dead_ids=None, wt_dead_wells=None,
    plate_rows=8, plate_cols=12, layout="row_major", cmap=None,
    figsize=(10.5, 3.4), font_size=10, save=False, figtitle="wt_vs_mutant_plate",
):
    """WT plate next to the mutant strain main effect for one feature, both
    z-scored across live wells. Same gradient on both = position, not genotype."""
    if cmap is None:
        cmap = "RdBu_r"
    mut_dead = {int(d) % n_wells for d in (dead_ids or [])}
    wt_dead = {int(w) for w in (wt_dead_wells or [])}
    rr, cc = _well_to_rc(np.arange(n_wells), plate_rows, plate_cols, layout)

    def _vals(M, dead):
        M = np.asarray(M, dtype=float)
        v = np.array([np.nanmean(M[_well_rows(w, n_wells, n_fovs), feature_index])
                      for w in range(n_wells)])
        v[sorted(dead)] = np.nan
        return v

    wt = _vals(wt_data, wt_dead)
    centred = [(_vals(mutant_data[c], mut_dead | wt_dead)) for c in exp_conditions]
    centred = np.vstack([v - np.nanmean(v) for v in centred])
    mut = np.full(n_wells, np.nan)
    live = np.isfinite(centred).all(axis=0)
    mut[live] = centred[:, live].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    name = training_feature_labels[feature_index]
    for ax, v, title in [(axes[0], wt, f"Wild-type plate — {name}"),
                         (axes[1], mut, f"Mutant strain effect — {name}")]:
        z = (v - np.nanmean(v)) / (np.nanstd(v) or 1.0)
        grid = np.full((plate_rows, plate_cols), np.nan)
        grid[rr, cc] = z
        lim = np.nanmax(np.abs(grid))
        im = ax.imshow(grid, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_xticks(range(plate_cols)); ax.set_xticklabels(range(1, plate_cols + 1), fontsize=font_size - 2)
        ax.set_yticks(range(plate_rows))
        ax.set_yticklabels([chr(ord("A") + i) for i in range(plate_rows)], fontsize=font_size - 2)
        ax.set_title(title, fontsize=font_size - 1)
        fig.colorbar(im, ax=ax, shrink=0.85, label="z-score across wells")
    plt.show()
    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}", bbox_inches="tight", dpi=300)
    return fig, axes


def plot_wt_gradient_reference(
    ref_df, label_features=("count (bf)", "count (cy5)", "eccentricity (bf)", "area (bf)"),
    q_threshold=0.05, figsize=(11, 7.5), font_size=10, color=None,
    save=False, figtitle="wt_plate_gradient_reference",
):
    """
    Left: positional structure on the WT plate (above the shuffle null) against
    the same fraction for the mutant strain effect. Right: how much of the mutant
    strain effect the WT surface predicts. Bars are the observed R², black ticks
    the per-feature 95th percentile of the shuffle null, and filled bars with an
    asterisk pass Benjamini-Hochberg correction (q < q_threshold).
    """
    if color is None:
        color = apc.aster
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    ax = axes[0]
    ax.scatter(ref_df.mut_frac_position, ref_df.wt_excess_position, s=22, color=color, alpha=0.7, lw=0)
    ax.axhline(0, color="#999999", ls="--", lw=1)
    ax.set_xlabel("Mutant strain effect: fraction from row + column\n(confounded with genotype)", fontsize=font_size - 1)
    ax.set_ylabel("WT plate: fraction from row + column,\nabove shuffle null", fontsize=font_size - 1)
    for _, r in ref_df[ref_df.feature.isin(label_features)].iterrows():
        ax.annotate(r.feature, (r.mut_frac_position, r.wt_excess_position), fontsize=font_size - 3,
                    xytext=(3, 2), textcoords="offset points", color="#444444")

    ax = axes[1]
    d = ref_df.sort_values("r2_mut_from_wt_surface", ascending=True).reset_index(drop=True)
    sig = (d.q_r2 < q_threshold).to_numpy() if "q_r2" in d else np.zeros(len(d), bool)
    y = np.arange(len(d))
    ax.barh(y, d.r2_mut_from_wt_surface, color=[color if s_ else "#D9D9D9" for s_ in sig],
            alpha=0.9, height=0.75)
    if "null_r2_95" in d:
        ax.scatter(d.null_r2_95, y, marker="|", s=90, color="black", lw=1.4, zorder=3,
                   label="95th percentile of shuffle null")
        ax.legend(loc="lower right", fontsize=font_size - 3, frameon=False)
    for yi, (v, s_, q) in enumerate(zip(d.r2_mut_from_wt_surface, sig, d.get("q_r2", [np.nan] * len(d)))):
        if s_:
            ax.annotate(f"* q = {q:.3f}", (v, yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=font_size - 3)
    ax.set_yticks(y); ax.set_yticklabels(d.feature, fontsize=font_size - 4)
    ax.set_xlim(0, max(d.r2_mut_from_wt_surface.max(), d.get("null_r2_95", d.r2_mut_from_wt_surface).max()) * 1.25)
    ax.set_xlabel(f"Mutant strain effect predicted by WT surface (R²)\n* = BH q < {q_threshold}", fontsize=font_size - 1)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=font_size - 2)
    plt.show()
    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}", bbox_inches="tight", dpi=300)
    return fig, axes
