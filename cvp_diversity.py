"""
cvp_diversity.py

Self-contained analysis functions for comparing phenotypic diversity between
mutagenized Chlorella strains and wild-type clones.

Core utilities (get_strain_means, compute_within_between_strain_variation,
pca_on_np_array, remove_rows_cols) are adapted directly from cvp.py to keep
this module free of cross-dependencies. New analysis and plotting functions are
written to match the style, parameter naming, and save conventions already
established in cvp.py.

Typical usage
-------------
Import alongside cvp for data loading, then call these functions on the
training_data dicts produced by cvp.build_training_data():

    import cvp
    import cvp_diversity as cvpd

    strain_means_mut_e22, well_ids = cvpd.get_strain_means(
        training_data_hu_log, "mut_e22", n_wells=96, n_fovs=9,
        dead_wells=dead_wells_all
    )
    pairwise_dists, _, _ = cvpd.compute_all_pairwise_distances(
        training_data_hu_log, "mut_e22", dead_wells=dead_wells_all
    )
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore, mannwhitneyu
from scipy.linalg import eigh

import arcadia_pycolor as apc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import accuracy_score, f1_score

font_path = "~/Library/Fonts"
apc.mpl.setup(font_dirpath=font_path)

# ─────────────────────────────────────────────────────────────────────────────
# Utilities adapted from cvp.py
# ─────────────────────────────────────────────────────────────────────────────

def remove_rows_cols(array, row_indices, col_indices):
    """Remove specified rows and columns from a 2D numpy array."""
    tmp = np.delete(array, row_indices, axis=0)
    return np.delete(tmp, col_indices, axis=1)


def pca_on_np_array(nparray):
    """
    PCA on a 2-D numpy array.  Rows are instances, columns are dimensions.
    Adapted from cvp.py (modified to avoid mutating the input).

    Returns
    -------
    pcs : np.ndarray   (n_samples, n_features)
    evecs : np.ndarray (n_features, n_features)
    evals : np.ndarray (n_features,) — sorted descending
    mean_value : np.ndarray (n_features,)
    """
    X = nparray.copy().astype(float)
    mean_value = np.mean(X, axis=0)
    X -= mean_value
    cov = np.cov(X, rowvar=False)
    evals, evecs = eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    pcs = np.dot(X, evecs)
    return pcs, evecs, evals, mean_value


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


def compute_within_between_strain_variation(
    training_data,
    condition,
    n_wells=96,
    n_fovs=9,
    dead_wells=None,
    use_cv=True,
    eps=1e-8,
):
    """
    Compute within-strain vs between-strain variability for each feature.
    Adapted from cvp.py.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
    condition : str
    n_wells : int
    n_fovs : int
    dead_wells : list[int] or None
    use_cv : bool
        If True, scale by the grand-mean magnitude (coefficient-of-variation style).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    within_var  : np.ndarray (n_features,)
    between_var : np.ndarray (n_features,)
    ratio       : np.ndarray (n_features,)  — between / within
    """
    if dead_wells is None:
        dead_wells = []

    dat = np.asarray(training_data[condition], dtype=float)
    n_rows, n_features = dat.shape

    if n_fovs is None:
        n_fovs = n_rows // n_wells

    well_blocks = []
    for well in range(n_wells):
        if well in dead_wells:
            continue
        rows = [well + fov * n_wells for fov in range(n_fovs)]
        well_blocks.append(dat[rows, :])

    well_blocks = np.array(well_blocks)   # (n_valid_wells, n_fovs, n_features)

    within_var = np.nanmean(np.nanvar(well_blocks, axis=1), axis=0)
    well_means = np.nanmean(well_blocks, axis=1)          # (n_valid_wells, n_features)
    between_var = np.nanvar(well_means, axis=0)

    if use_cv:
        grand_mean = np.abs(np.nanmean(well_means, axis=0))
        within_var  = np.sqrt(within_var)  / (grand_mean + eps)
        between_var = np.sqrt(between_var) / (grand_mean + eps)

    ratio = between_var / (within_var + eps)
    return within_var, between_var, ratio


# ─────────────────────────────────────────────────────────────────────────────
# New analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_pairwise_distances(
    training_data,
    condition,
    n_wells=96,
    n_fovs=9,
    dead_wells=None,
    zscore_features=True,
    metric="euclidean",
):
    """
    Compute all pairwise distances between strains within one condition.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
    condition : str
    n_wells, n_fovs : int
    dead_wells : list[int] or None
    zscore_features : bool
        If True, z-score each feature column before computing distances.
    metric : str
        Distance metric passed to scipy.spatial.distance.pdist.

    Returns
    -------
    pairwise_dists : np.ndarray
        Flat upper-triangle pairwise distances.
    strain_means : np.ndarray  (n_valid_wells, n_features)
    valid_well_ids : list[int]
    """
    strain_means, valid_well_ids = get_strain_means(
        training_data, condition, n_wells=n_wells, n_fovs=n_fovs,
        dead_wells=dead_wells,
    )

    X = strain_means.copy()
    if zscore_features:
        X = zscore(X, axis=0, nan_policy="omit")
        X = np.where(np.isfinite(X), X, 0.0)

    n = X.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    dist_mat = squareform(pdist(X, metric=metric))
    pairwise_dists = dist_mat[upper_idx]

    return pairwise_dists, strain_means, valid_well_ids


def compute_diversity_summary(pairwise_dists_dict, ratio_dict=None):
    """
    Build a summary DataFrame of diversity statistics per group.

    Parameters
    ----------
    pairwise_dists_dict : dict[str, np.ndarray]
        Condition key → flat pairwise distance array.
    ratio_dict : dict[str, np.ndarray] or None
        Condition key → between/within ratio array (n_features,).

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for key, dists in pairwise_dists_dict.items():
        row = {
            "group": key,
            "mean_pairwise_dist": float(np.nanmean(dists)),
            "median_pairwise_dist": float(np.nanmedian(dists)),
            "std_pairwise_dist": float(np.nanstd(dists)),
            "n_strain_pairs": int(len(dists)),
        }
        if ratio_dict is not None and key in ratio_dict:
            r = ratio_dict[key]
            row["mean_between_within_ratio"] = float(np.nanmean(r))
            row["median_between_within_ratio"] = float(np.nanmedian(r))
            row["n_features"] = int(len(r))
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_covariance_ellipse(ax, x, y, color, n_std=1.5, alpha=0.15):
    """
    Draw a covariance ellipse centered on (mean_x, mean_y).
    n_std controls how many standard deviations the ellipse spans.
    """
    cov = np.cov(x, y)
    pearson = cov[0, 1] / (np.sqrt(cov[0, 0] * cov[1, 1]) + 1e-12)

    # eigenvalue-based half-axes of the unit circle
    ell_rx = np.sqrt(1 + pearson)
    ell_ry = np.sqrt(max(1 - pearson, 0))

    ellipse = mpatches.Ellipse(
        (0, 0),
        width=ell_rx * 2,
        height=ell_ry * 2,
        facecolor=color,
        edgecolor=color,
        linewidth=1.0,
        alpha=alpha,
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x  = np.mean(x)
    mean_y  = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def _apply_spine_style(ax, axis_color, font_size, font_family=None):
    """Shared axis styling used by all plot functions."""
    for spine in ax.spines.values():
        spine.set_color(axis_color)
    ax.tick_params(axis="both", color=axis_color, labelsize=font_size)
    if font_family is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)


def _save_figure(fig, figtitle, save_dir="plots"):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{save_dir}/{figtitle}.{ext}", bbox_inches="tight")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_pairwise_distance_violin(
    pairwise_dists_dict,
    group_labels=None,
    group_colors=None,
    figsize=(6, 5),
    title="Pairwise phenotypic distance between strains",
    show_title=True,
    title_fontsize=14,
    font_family=None,
    font_size=11,
    axis_color="black",
    ylabel="Pairwise euclidean distance",
    show_stripplot=True,
    strip_alpha=0.25,
    strip_size=1.8,
    violin_alpha=0.75,
    add_stats=True,
    save=False,
    figtitle="diversity_pairwise_distance_violin",
):
    """
    Violin + strip-plot comparing the distribution of pairwise phenotypic
    distances among strains, one distribution per condition/group.

    Parameters
    ----------
    pairwise_dists_dict : dict[str, np.ndarray]
        Mapping of condition key → flat array of pairwise distances.
    group_labels : list[str] or None
        Display labels shown on the x-axis (same order as dict keys).
    group_colors : list or None
        One matplotlib-compatible color per group.
    add_stats : bool
        If True, annotate with Mann-Whitney U p-value between first two groups.

    Returns
    -------
    fig, ax
    """
    keys = list(pairwise_dists_dict.keys())

    if group_labels is None:
        group_labels = keys
    if group_colors is None:
        group_colors = [apc.wish, apc.aster, apc.vital]

    # Build long-form DataFrame
    records = []
    for key, dists in pairwise_dists_dict.items():
        for d in dists:
            records.append({"group": key, "distance": float(d)})
    df = pd.DataFrame(records)

    palette = {k: c for k, c in zip(keys, group_colors)}
    label_map = {k: lbl for k, lbl in zip(keys, group_labels)}

    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=df, x="group", y="distance",
        palette=palette, order=keys,
        inner=None, alpha=violin_alpha, ax=ax,
    )
    if show_stripplot:
        sns.stripplot(
            data=df, x="group", y="distance",
            color="white", edgecolor="none",
            order=keys, size=strip_size, alpha=strip_alpha, ax=ax, jitter=True,
        )

    ax.set_xticklabels(
        group_labels,
        fontsize=font_size,
        fontfamily=font_family,
    )
    ax.set_ylabel(ylabel, fontsize=font_size, fontfamily=font_family)
    ax.set_xlabel("", fontsize=font_size)

    if show_title:
        ax.set_title(title, fontsize=title_fontsize, fontfamily=font_family)

    # Optional Mann-Whitney annotation between first two groups
    if add_stats and len(keys) >= 2:
        dists_a = pairwise_dists_dict[keys[0]]
        dists_b = pairwise_dists_dict[keys[1]]
        _, p = mannwhitneyu(dists_a, dists_b, alternative="two-sided")
        p_str = f"p = {p:.2e}" if p >= 1e-300 else "p < 1e-300"
        y_max = df["distance"].max()
        y_range = y_max - df["distance"].min()
        ax.annotate(
            p_str,
            xy=(0.5, 1.02), xycoords="axes fraction",
            ha="center", va="bottom",
            fontsize=font_size - 2,
            fontfamily=font_family,
            color="gray",
        )

    _apply_spine_style(ax, axis_color, font_size, font_family)
    plt.tight_layout()
    plt.show()

    if save:
        _save_figure(fig, figtitle)

    return fig, ax


def plot_pca_group_spread(
    strain_means_dict,
    group_labels=None,
    group_colors=None,
    figsize=(7, 6),
    title="Phenotype-space spread: mutants vs. wild type",
    show_title=True,
    title_fontsize=14,
    font_family=None,
    font_size=11,
    point_size=18,
    point_alpha=0.6,
    axis_color="black",
    show_ellipses=True,
    ellipse_alpha=0.12,
    ellipse_n_std=1.5,
    show_variance_explained=True,
    legend_loc="best",
    save=False,
    figtitle="diversity_pca_group_spread",
):
    """
    PCA scatter plot of strain mean vectors, colored by group.

    PCA is fitted on all strains combined (after global z-scoring).  Each
    point is one strain; colored by experimental group.  Optional covariance
    ellipses illustrate each group's spread.

    Parameters
    ----------
    strain_means_dict : dict[str, np.ndarray]
        Condition key → (n_valid_wells, n_features) array of strain means.
        The values should already be z-scored or preprocessed as desired.
    group_labels : list[str] or None
    group_colors : list or None

    Returns
    -------
    fig, ax, pcs, evals
    """
    keys = list(strain_means_dict.keys())

    if group_labels is None:
        group_labels = keys
    if group_colors is None:
        group_colors = [apc.wish, apc.aster, apc.vital]

    # Stack all strain means, globally z-score, run PCA
    all_means = np.vstack([strain_means_dict[k] for k in keys])

    col_std = np.nanstd(all_means, axis=0)
    col_std[col_std == 0] = 1.0
    col_mean = np.nanmean(all_means, axis=0)
    all_z = (all_means - col_mean) / col_std
    all_z = np.where(np.isfinite(all_z), all_z, 0.0)

    pcs, evecs, evals, _ = pca_on_np_array(all_z)

    total_var = np.sum(evals[evals > 0])
    var_exp = evals / (total_var + 1e-12)

    fig, ax = plt.subplots(figsize=figsize)

    idx = 0
    for key, label, color in zip(keys, group_labels, group_colors):
        n = strain_means_dict[key].shape[0]
        grp_pcs = pcs[idx: idx + n, :]

        ax.scatter(
            grp_pcs[:, 0], grp_pcs[:, 1],
            color=color, label=label,
            s=point_size, alpha=point_alpha, edgecolors="none",
        )
        if show_ellipses and n > 3:
            _draw_covariance_ellipse(
                ax, grp_pcs[:, 0], grp_pcs[:, 1],
                color=color, n_std=ellipse_n_std, alpha=ellipse_alpha,
            )
        idx += n

    pc1_lbl = f"PC1 ({var_exp[0] * 100:.1f}%)" if show_variance_explained else "PC1"
    pc2_lbl = f"PC2 ({var_exp[1] * 100:.1f}%)" if show_variance_explained else "PC2"
    ax.set_xlabel(pc1_lbl, fontsize=font_size, fontfamily=font_family)
    ax.set_ylabel(pc2_lbl, fontsize=font_size, fontfamily=font_family)

    if show_title:
        ax.set_title(title, fontsize=title_fontsize, fontfamily=font_family)

    ax.legend(fontsize=font_size - 1, loc=legend_loc)
    _apply_spine_style(ax, axis_color, font_size, font_family)

    plt.tight_layout()
    plt.show()

    if save:
        _save_figure(fig, figtitle)

    return fig, ax, pcs, evals


def plot_between_within_ratio_comparison(
    ratio_dict,
    feature_labels,
    group_labels=None,
    group_colors=None,
    figsize=(7, 5),
    title="Between- vs. within-strain variation ratio",
    show_title=True,
    title_fontsize=14,
    font_family=None,
    font_size=11,
    axis_color="black",
    log2_transform=True,
    ylabel=None,
    add_stats=True,
    save=False,
    figtitle="diversity_between_within_ratio",
):
    """
    Box + strip plot comparing the distribution of feature-level between/within
    variation ratios across groups (mutants vs. wild type).

    A ratio > 1 (log2 > 0) means between-strain variation exceeds within-strain
    variation — i.e. phenotypic diversity is visible across strains.

    Parameters
    ----------
    ratio_dict : dict[str, np.ndarray]
        Condition key → ratio array of shape (n_features,).
    feature_labels : list[str]
        Feature labels corresponding to the ratio arrays.
    log2_transform : bool
        If True, show log2 of the ratio (ratio values from
        compute_within_between_strain_variation are already linear).

    Returns
    -------
    fig, ax
    """
    keys = list(ratio_dict.keys())

    if group_labels is None:
        group_labels = keys
    if group_colors is None:
        group_colors = [apc.wish, apc.aster, apc.vital]
    if ylabel is None:
        ylabel = "log2(between / within)" if log2_transform else "Between / within ratio"

    records = []
    for key, ratios in ratio_dict.items():
        vals = np.log2(ratios + 1e-8) if log2_transform else np.asarray(ratios)
        for feat, v in zip(feature_labels, vals):
            records.append({"group": key, "ratio": float(v), "feature": feat})
    df = pd.DataFrame(records)

    palette = {k: c for k, c in zip(keys, group_colors)}

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df, x="group", y="ratio",
        palette=palette, order=keys,
        width=0.45,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        ax=ax,
    )
    sns.stripplot(
        data=df, x="group", y="ratio",
        palette=palette, order=keys,
        size=2.5, alpha=0.3, ax=ax, jitter=True, dodge=False,
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xticklabels(group_labels, fontsize=font_size, fontfamily=font_family)
    ax.set_ylabel(ylabel, fontsize=font_size, fontfamily=font_family)
    ax.set_xlabel("", fontsize=font_size)

    if show_title:
        ax.set_title(title, fontsize=title_fontsize, fontfamily=font_family)

    if add_stats and len(keys) >= 2:
        vals_a = df.loc[df["group"] == keys[0], "ratio"].dropna().values
        vals_b = df.loc[df["group"] == keys[1], "ratio"].dropna().values
        _, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        p_str = f"p = {p:.2e}" if p >= 1e-300 else "p < 1e-300"
        ax.annotate(
            p_str,
            xy=(0.5, 1.02), xycoords="axes fraction",
            ha="center", va="bottom",
            fontsize=font_size - 2,
            fontfamily=font_family,
            color="gray",
        )

    _apply_spine_style(ax, axis_color, font_size, font_family)
    plt.tight_layout()
    plt.show()

    if save:
        _save_figure(fig, figtitle)

    return fig, ax


def plot_pca_variance_explained(
    evals,
    n_components=20,
    figsize=(7, 4),
    title="PCA variance explained",
    show_title=True,
    title_fontsize=14,
    font_family=None,
    font_size=11,
    color=None,
    axis_color="black",
    save=False,
    figtitle="diversity_pca_variance_explained",
):
    """
    Bar chart of variance explained by each PC.

    Parameters
    ----------
    evals : np.ndarray
        Eigenvalues from pca_on_np_array(), sorted descending.
    n_components : int
        Number of PCs to display.

    Returns
    -------
    fig, ax
    """
    if color is None:
        color = apc.wish

    total_var = np.sum(evals[evals > 0])
    var_exp = (evals[:n_components] / (total_var + 1e-12)) * 100

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(1, len(var_exp) + 1), var_exp, color=color, alpha=0.8)
    ax.set_xlabel("Principal component", fontsize=font_size, fontfamily=font_family)
    ax.set_ylabel("Variance explained (%)", fontsize=font_size, fontfamily=font_family)

    if show_title:
        ax.set_title(title, fontsize=title_fontsize, fontfamily=font_family)

    _apply_spine_style(ax, axis_color, font_size, font_family)
    plt.tight_layout()
    plt.show()

    if save:
        _save_figure(fig, figtitle)

    return fig, ax


def plot_mean_pairwise_distance_bar(
    pairwise_dists_dict,
    group_labels=None,
    group_colors=None,
    figsize=(5, 5),
    title="Mean pairwise phenotypic distance",
    show_title=True,
    title_fontsize=14,
    font_family=None,
    font_size=11,
    axis_color="black",
    ylabel="Mean pairwise distance (z-scored features)",
    error_style="sem",   # "sem" or "std"
    save=False,
    figtitle="diversity_mean_pairwise_distance_bar",
):
    """
    Bar chart of the mean pairwise distance per group ± SEM or SD.

    Parameters
    ----------
    pairwise_dists_dict : dict[str, np.ndarray]
    error_style : str
        "sem" for standard error of the mean, "std" for standard deviation.

    Returns
    -------
    fig, ax, summary_df
    """
    keys = list(pairwise_dists_dict.keys())

    if group_labels is None:
        group_labels = keys
    if group_colors is None:
        group_colors = [apc.wish, apc.aster, apc.vital]

    means, errors = [], []
    for key in keys:
        d = pairwise_dists_dict[key]
        means.append(np.nanmean(d))
        if error_style == "sem":
            errors.append(np.nanstd(d) / np.sqrt(np.sum(np.isfinite(d))))
        else:
            errors.append(np.nanstd(d))

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=figsize)

    for i, (key, m, e, color) in enumerate(zip(keys, means, errors, group_colors)):
        ax.bar(i, m, yerr=e, color=color, alpha=0.85, capsize=4, error_kw={"elinewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=font_size, fontfamily=font_family)
    ax.set_ylabel(ylabel, fontsize=font_size, fontfamily=font_family)
    ax.set_xlabel("", fontsize=font_size)

    if show_title:
        ax.set_title(title, fontsize=title_fontsize, fontfamily=font_family)

    _apply_spine_style(ax, axis_color, font_size, font_family)
    plt.tight_layout()
    plt.show()

    if save:
        _save_figure(fig, figtitle)

    summary_df = pd.DataFrame({
        "group": group_labels,
        "mean_pairwise_dist": means,
        "error": errors,
        "error_type": error_style,
    })
    return fig, ax, summary_df


# ─────────────────────────────────────────────────────────────────────────────
# Plate position correction
# ─────────────────────────────────────────────────────────────────────────────

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


def plot_plate_position_heatmap(
    position_effects,
    feature_labels,
    condition,
    feature=None,
    n_wells=96,
    n_rows=8,
    n_cols=12,
    figsize=(6, 4),
    cmap="RdBu_r",
    title=None,
    font_family=None,
    font_size=10,
    axis_color="black",
    save=False,
    figtitle="plate_position_heatmap",
):
    """
    Visualise the estimated plate position effect as an 8 × 12 heatmap.

    Spatial gradients or ring-shaped edge effects in the heatmap confirm the
    correction is capturing real plate artefacts rather than removing biology.

    Parameters
    ----------
    position_effects : dict[str, np.ndarray]
        Output of correct_plate_position_effects().
    feature_labels : list[str]
        Feature labels from cvp.build_training_data().
    condition : str
        Which condition's position effects to show.
    feature : str or None
        Feature name to display (e.g., ``"area (bf)"``).
        If None, the first principal component of the full position-effect
        matrix is shown (summarises the dominant spatial pattern).

    Returns
    -------
    fig, ax, plate  (plate is the 8 × 12 numpy array plotted)
    """
    pos_eff = position_effects[condition].copy()   # (n_wells, n_features)

    if feature is None:
        # Summarise all features: project onto PC1 of position effects
        # (only use wells with non-zero effects)
        valid_mask = np.any(pos_eff != 0, axis=1)
        if np.sum(valid_mask) > 2:
            pos_z = pos_eff[valid_mask].copy()
            std = np.nanstd(pos_z, axis=0)
            std[std == 0] = 1.0
            pos_z /= std
            pcs, _, _, _ = pca_on_np_array(pos_z)
            pc1_vals = np.full(n_wells, np.nan)
            pc1_vals[valid_mask] = pcs[:, 0]
            values = pc1_vals
        else:
            values = pos_eff[:, 0]
        cbar_label = "PC1 of position effect"
        if title is None:
            title = f"Plate position effect (PC1) — {condition}"
    else:
        if feature not in feature_labels:
            raise ValueError(f"Feature '{feature}' not found in feature_labels.")
        feat_idx = feature_labels.index(feature)
        values = pos_eff[:, feat_idx]
        cbar_label = f"Position effect: {feature}"
        if title is None:
            title = f"Plate position effect: {feature} — {condition}"

    # Reshape to 8 × 12 plate layout
    plate = np.full((n_rows, n_cols), np.nan)
    for well in range(n_wells):
        plate[well // n_cols, well % n_cols] = values[well]

    row_labels = list("ABCDEFGH")[:n_rows]
    col_labels  = [str(i + 1) for i in range(n_cols)]

    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(
        plate,
        cmap=cmap,
        center=0,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
    )
    hm.collections[0].colorbar.set_label(
        cbar_label, fontsize=font_size, fontfamily=font_family
    )

    ax.set_title(title, fontsize=font_size + 2, fontfamily=font_family)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=font_size - 1,
                       fontfamily=font_family, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size - 1,
                       fontfamily=font_family, rotation=0)

    _apply_spine_style(ax, axis_color, font_size, font_family)
    plt.tight_layout()
    plt.show()

    if save:
        _save_figure(fig, figtitle)

    return fig, ax, plate


# ─────────────────────────────────────────────────────────────────────────────
# Strain-identity classifier
# ─────────────────────────────────────────────────────────────────────────────

def run_strain_classifier_cv(
    training_data,
    condition,
    n_wells=96,
    n_fovs=9,
    dead_wells=None,
    C=1.0,
    max_iter=500,
    random_state=42,
    run_permutation_test=True,
    n_permutations=20,
    verbose=True,
):
    """
    Leave-one-FoV-out cross-validation strain identity classifier.

    For each held-out FoV, a logistic regression is trained on the remaining
    FoVs to predict which well (strain) each FoV-mean feature vector belongs to.
    High accuracy means strains are phenotypically distinguishable — i.e. diverse.

    Parameters
    ----------
    training_data : dict[str, np.ndarray]
        Output of cvp.build_training_data().  Rows: n_wells × n_fovs.
    condition : str
        Key in training_data to evaluate.
    n_wells : int
    n_fovs  : int
    dead_wells : list[int] or None
        Wells to exclude from both train and test.
    C : float
        Logistic regression inverse regularisation strength.
    max_iter : int
        Maximum solver iterations.
    random_state : int
    run_permutation_test : bool
        If True, also run `n_permutations` shuffled-label trials to build a
        null distribution.  Each permutation shuffles well labels globally and
        reruns the full leave-one-FoV-out CV.
    n_permutations : int
        Number of permutation trials (used only when run_permutation_test=True).
    verbose : bool
        Print fold-by-fold accuracy.

    Returns
    -------
    dict with keys
        fold_accuracies      : list[float]   — accuracy per CV fold
        fold_f1s             : list[float]   — macro F1 per CV fold
        mean_accuracy        : float
        std_accuracy         : float
        mean_f1              : float
        std_f1               : float
        chance_level         : float         — 1 / n_valid_wells
        n_classes            : int
        n_folds              : int
        permutation_mean_accs: list[float]   (present if run_permutation_test)
        permutation_p_value  : float         (present if run_permutation_test)
    """
    if dead_wells is None:
        dead_wells = set()
    else:
        dead_wells = set(dead_wells)

    dat = np.asarray(training_data[condition], dtype=float)

    # Build aligned arrays: one entry per (valid_well, fov)
    row_indices = []
    fov_ids     = []
    well_ids    = []

    for fov in range(n_fovs):
        for well in range(n_wells):
            if well in dead_wells:
                continue
            row_indices.append(well + fov * n_wells)
            fov_ids.append(fov)
            well_ids.append(well)

    row_indices = np.array(row_indices)
    fov_ids     = np.array(fov_ids)
    well_ids    = np.array(well_ids)

    X = dat[row_indices, :]
    X = np.where(np.isfinite(X), X, 0.0)   # guard against residual NaN/inf
    y = well_ids.copy()

    unique_fovs = np.unique(fov_ids)
    n_classes   = len(np.unique(y))
    chance      = 1.0 / n_classes

    def _one_cv_pass(labels):
        """Run leave-one-FoV-out CV with `labels` as class targets."""
        accs, f1s = [], []
        for test_fov in unique_fovs:
            train_mask = fov_ids != test_fov
            test_mask  = fov_ids == test_fov

            X_tr, y_tr = X[train_mask], labels[train_mask]
            X_te, y_te = X[test_mask],  labels[test_mask]

            clf = SklearnPipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                    multi_class="multinomial",
                    solver="lbfgs",
                    n_jobs=1,
                )),
            ])
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            accs.append(accuracy_score(y_te, y_pred))
            f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))

        return accs, f1s

    # ── Real CV ──────────────────────────────────────────────────────────────
    fold_accs, fold_f1s = _one_cv_pass(y)

    if verbose:
        for i, (a, f) in enumerate(zip(fold_accs, fold_f1s)):
            print(f"    {condition}  fold {i} (held-out FoV {unique_fovs[i]}): "
                  f"acc={a:.3f}  macro-F1={f:.3f}")

    results = {
        "fold_accuracies":  fold_accs,
        "fold_f1s":         fold_f1s,
        "mean_accuracy":    float(np.mean(fold_accs)),
        "std_accuracy":     float(np.std(fold_accs)),
        "mean_f1":          float(np.mean(fold_f1s)),
        "std_f1":           float(np.std(fold_f1s)),
        "chance_level":     float(chance),
        "n_classes":        int(n_classes),
        "n_folds":          int(len(unique_fovs)),
    }

    # ── Permutation test ─────────────────────────────────────────────────────
    if run_permutation_test:
        rng = np.random.default_rng(random_state)
        perm_mean_accs = []

        for perm_i in range(n_permutations):
            # Shuffle well labels globally (same shuffle applied to all FoVs)
            unique_wells = np.unique(y)
            shuffled_well_map = dict(zip(unique_wells, rng.permutation(unique_wells)))
            y_perm = np.array([shuffled_well_map[w] for w in y])

            perm_accs, _ = _one_cv_pass(y_perm)
            perm_mean_accs.append(float(np.mean(perm_accs)))

        results["permutation_mean_accs"] = perm_mean_accs
        results["permutation_p_value"] = float(
            np.mean(np.array(perm_mean_accs) >= results["mean_accuracy"])
        )
        if verbose:
            print(f"    {condition}  permutation p = {results['permutation_p_value']:.4f}  "
                  f"(null mean = {np.mean(perm_mean_accs):.4f})")

    return results


def plot_classifier_cv_results(
    cv_results_dict,
    group_labels=None,
    group_colors=None,
    figsize=(7, 5),
    title="Strain identity classification accuracy\n(leave-one-FoV-out CV)",
    show_title=True,
    title_fontsize=14,
    font_family=None,
    font_size=11,
    axis_color="black",
    metric="accuracy",           # "accuracy" or "f1"
    show_chance_line=True,
    show_permutation_null=True,
    point_size=60,
    point_alpha=0.8,
    null_alpha=0.4,
    save=False,
    figtitle="diversity_classifier_accuracy",
):
    """
    Strip + mean±std plot of leave-one-FoV-out classification results.

    Each point is one CV fold (held-out FoV).  The group mean ± std is shown
    as a thick horizontal bar with error cap.  A dashed line marks chance level.
    When permutation results are present, the null distribution is overlaid as
    a transparent strip.

    Parameters
    ----------
    cv_results_dict : dict[str, dict]
        Mapping condition key → result dict from run_strain_classifier_cv().
    metric : str
        "accuracy" or "f1" — which metric to plot.

    Returns
    -------
    fig, ax
    """
    keys = list(cv_results_dict.keys())
    if group_labels is None:
        group_labels = keys
    if group_colors is None:
        group_colors = [apc.wish, apc.aster, apc.vital]

    fold_key = "fold_accuracies" if metric == "accuracy" else "fold_f1s"
    mean_key = "mean_accuracy"   if metric == "accuracy" else "mean_f1"
    std_key  = "std_accuracy"    if metric == "accuracy" else "std_f1"
    ylabel   = "Classification accuracy" if metric == "accuracy" else "Macro F1"

    fig, ax = plt.subplots(figsize=figsize)
    x_positions = np.arange(len(keys))

    for xi, (key, label, color) in enumerate(zip(keys, group_labels, group_colors)):
        res = cv_results_dict[key]
        fold_vals = np.array(res[fold_key])
        mean_val  = res[mean_key]
        std_val   = res[std_key]
        chance    = res["chance_level"]

        # Permutation null: show as a thin shaded box
        if show_permutation_null and "permutation_mean_accs" in res:
            perm_vals = np.array(res["permutation_mean_accs"])
            ax.fill_between(
                [xi - 0.28, xi + 0.28],
                np.percentile(perm_vals, 5),
                np.percentile(perm_vals, 95),
                color=color, alpha=null_alpha * 0.5,
                linewidth=0, zorder=1,
            )

        # Individual fold dots
        jitter = np.random.default_rng(42 + xi).uniform(-0.12, 0.12, len(fold_vals))
        ax.scatter(
            np.full(len(fold_vals), xi) + jitter, fold_vals,
            color=color, s=point_size, alpha=point_alpha,
            edgecolors="white", linewidths=0.5, zorder=3,
        )

        # Mean ± std as a thick horizontal tick
        ax.plot([xi - 0.25, xi + 0.25], [mean_val, mean_val],
                color=color, linewidth=2.5, zorder=4)
        ax.plot([xi, xi], [mean_val - std_val, mean_val + std_val],
                color=color, linewidth=1.5, zorder=4)

        # Annotate mean
        ax.text(
            xi + 0.30, mean_val,
            f"{mean_val:.2f}",
            va="center", ha="left",
            fontsize=font_size - 2, color=color,
            fontfamily=font_family,
        )

        # Chance line (first group only, then shared)
        if xi == 0 and show_chance_line:
            ax.axhline(
                chance, color="gray", linestyle="--",
                linewidth=1.0, alpha=0.7, zorder=2,
                label=f"Chance ({chance:.3f})",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_labels, fontsize=font_size, fontfamily=font_family)
    ax.set_ylabel(ylabel, fontsize=font_size, fontfamily=font_family)
    ax.set_xlabel("", fontsize=font_size)

    if show_chance_line:
        ax.legend(fontsize=font_size - 2)

    if show_title:
        ax.set_title(title, fontsize=title_fontsize, fontfamily=font_family)

    _apply_spine_style(ax, axis_color, font_size, font_family)
    plt.tight_layout()
    plt.show()

    if save:
        _save_figure(fig, figtitle)

    return fig, ax


def classifier_summary_table(cv_results_dict, group_labels=None):
    """
    Build a summary DataFrame from a dict of cv_results.

    Returns
    -------
    pd.DataFrame
    """
    keys = list(cv_results_dict.keys())
    if group_labels is None:
        group_labels = keys

    rows = []
    for key, label in zip(keys, group_labels):
        res = cv_results_dict[key]
        row = {
            "condition":       key,
            "label":           label,
            "n_classes":       res["n_classes"],
            "chance_level":    round(res["chance_level"], 4),
            "mean_accuracy":   round(res["mean_accuracy"], 4),
            "std_accuracy":    round(res["std_accuracy"], 4),
            "mean_f1":         round(res["mean_f1"], 4),
            "std_f1":          round(res["std_f1"], 4),
            "fold_accuracies": res["fold_accuracies"],
        }
        if "permutation_p_value" in res:
            row["permutation_p_value"] = round(res["permutation_p_value"], 4)
        rows.append(row)
    return pd.DataFrame(rows)
