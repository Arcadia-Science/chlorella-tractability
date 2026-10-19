"""
Standardized (z-scored) versions of the PCA, t-SNE, and strain-trajectory analyses.

Why this exists
---------------
cvp.pca_on_np_array only mean-centres, and the notebook's t-SNE runs on those
PCs, so both work in raw feature units. Features span very different scales
(cell counts in the hundreds, areas in the thousands of pixels, Hu moments near
zero), so the raw versions are dominated by whichever features have the largest
numbers. Here every feature is z-scored first, so each one contributes on equal
footing.

The originals in cvp.py are untouched; these are separate functions.

Usage
-----
    import cvp_standardized as cvs

    # PCA + t-SNE on the same stacked matrix the notebook builds (all_dat)
    pcs_z, evecs_z, evals_z, var_ratio_z = cvs.pca_standardized(all_dat)
    tsne_z = cvs.tsne_standardized(all_dat, perplexity=200, random_state=28)

    # plot with the existing cvp.plot_tsne (rows line up with all_dat)
    cvp.plot_tsne(input_dat=all_dat, exp_embedding=tsne_z, ...,
                  figtitle="tsne_colored_by_experimental_condition_zscored")

    # strain trajectories with z-scored features
    cvs.plot_strain_pca_trajectories_standardized(training_data_finite, ...)
"""

import numpy as np
from scipy import linalg as LA
from sklearn.manifold import TSNE


# ----------------------------------------------------------------------------
# core helpers
# ----------------------------------------------------------------------------
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


def pca_standardized(X):
    """
    PCA on z-scored features (i.e., on the correlation matrix).

    Same first four outputs as cvp.pca_on_np_array, so it can drop in where
    that function was used:
        pcs, evecs, evals, explained_variance_ratio

    Unlike cvp.pca_on_np_array, the input array is NOT modified in place.
    """
    Xz, _, _ = zscore_columns(X)
    cov = np.cov(Xz, rowvar=False)
    evals, evecs = LA.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evals = np.clip(evals, 0, None)          # tiny negative values from rounding
    pcs = Xz @ evecs
    ratio = evals / evals.sum() if evals.sum() > 0 else evals
    return pcs, evecs, evals, ratio


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


# ----------------------------------------------------------------------------
# plotting wrapper
# ----------------------------------------------------------------------------
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
    import cvp
    std_conds = pca_conditions or list(training_data.keys())
    std_data, _, _ = standardize_training_data(
        training_data, conditions=std_conds, n_wells=n_wells,
        n_fovs=n_fovs, dead_ids=dead_ids,
    )
    return cvp.plot_strain_pca_trajectories(
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


# ----------------------------------------------------------------------------
# strain trajectories in t-SNE space
# ----------------------------------------------------------------------------
def tsne_row_index(exp_conditions, n_wells=96, n_fovs=9, dead_ids=None):
    """
    Which condition / well / FoV each row of `all_dat` (and of any t-SNE
    embedding computed from it) belongs to.

    Matches how the notebook builds all_dat: conditions stacked in
    `exp_conditions` order, each with dead rows removed, rows laid out as
    well + fov * n_wells.
    """
    import pandas as pd
    dead = set(int(d) for d in (dead_ids or []))
    recs = []
    for cond in exp_conditions:
        for r in range(n_wells * n_fovs):
            if r in dead:
                continue
            recs.append({"condition": cond, "well": r % n_wells, "fov": r // n_wells})
    return pd.DataFrame(recs)


def strain_mean_tsne(training_data, conditions, n_wells=96, n_fovs=None,
                     dead_ids=None, feature_idx=None, perplexity=30,
                     random_state=28, **tsne_kwargs):
    """
    t-SNE on STRAIN MEANS (one point per strain x condition) instead of on
    fields of view. Features are z-scored (pooled across `conditions`) first;
    pass `feature_idx` to restrict to, e.g., the reliable features.

    Returns (embedding, row_index) ready for plot_strain_tsne_trajectories.
    With ~92 x 6 = 552 points, a perplexity around 20-50 is appropriate.
    """
    import pandas as pd
    n_rows = np.asarray(training_data[conditions[0]]).shape[0]
    if n_fovs is None:
        n_fovs = n_rows // n_wells
    dead_wells = sorted({int(d) % n_wells for d in (dead_ids or [])})
    alive = [w for w in range(n_wells) if w not in dead_wells]

    std, _, _ = standardize_training_data(training_data, conditions=conditions,
                                          n_wells=n_wells, n_fovs=n_fovs,
                                          dead_ids=dead_ids)
    vecs, recs = [], []
    for cond in conditions:
        M = std[cond] if feature_idx is None else std[cond][:, feature_idx]
        cube = M.reshape(n_fovs, n_wells, -1)          # row = well + fov * n_wells
        for w in alive:
            vecs.append(np.nanmean(cube[:, w, :], axis=0))
            recs.append({"condition": cond, "well": w})
    emb = TSNE(n_components=2, perplexity=perplexity, init="random",
               learning_rate="auto", random_state=random_state,
               **tsne_kwargs).fit_transform(np.vstack(vecs))
    return emb, pd.DataFrame(recs)


def plot_strain_tsne_trajectories(embedding, row_index, conditions, strains_to_plot,
                                  condition_colors=None, condition_labels=None,
                                  background="points", background_alpha=0.15,
                                  background_size=6, point_size=80, line_color="#555555",
                                  line_width=1.5, annotate_points=True, figsize=(5, 6),
                                  show_title=False, title="Strain trajectories in t-SNE space",
                                  save=False, figtitle="strain_tsne_trajectories"):
    """
    Trajectory plot like cvp.plot_strain_pca_trajectories, but in t-SNE space.

    Each strain's position in a condition is the centroid of its points in the
    embedding (its nine fields of view for a field-level t-SNE, or its single
    point for strain_mean_tsne). Lines connect those positions in the order of
    `conditions`.

    background : "points"    every row of the embedding, faintly, colored by condition
                 "centroids" every strain's centroid, faintly, colored by condition
                 None        no background

    Note: t-SNE preserves neighborhoods, not distances. Treat which region a
    strain moves into as meaningful; don't compare path lengths across strains.

    Returns fig, ax, centroids (DataFrame: condition, well, x, y).
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    emb = np.asarray(embedding)
    df = row_index.reset_index(drop=True).copy()
    df["x"], df["y"] = emb[:, 0], emb[:, 1]
    cent = df.groupby(["condition", "well"], as_index=False)[["x", "y"]].mean()

    if condition_colors is None:
        cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        condition_colors = [cyc[i % len(cyc)] for i in range(len(conditions))]
    if condition_labels is None:
        condition_labels = list(conditions)
    cmap = dict(zip(conditions, condition_colors))

    fig, ax = plt.subplots(figsize=figsize)

    bg = df if background == "points" else cent if background == "centroids" else None
    if bg is not None:
        for cond in conditions:
            sub = bg[bg.condition == cond]
            ax.scatter(sub.x, sub.y, s=background_size, color=cmap[cond],
                       alpha=background_alpha, lw=0, zorder=0)

    for s in strains_to_plot:
        path = (cent[(cent.well == s) & (cent.condition.isin(conditions))]
                .set_index("condition").reindex(conditions).dropna())
        if path.empty:
            print(f"strain {s} not found (dead well?)")
            continue
        ax.plot(path.x, path.y, "-", color=line_color, lw=line_width, zorder=2)
        for i in range(len(path) - 1):
            ax.annotate("", xy=(path.x.iloc[i + 1], path.y.iloc[i + 1]),
                        xytext=(path.x.iloc[i], path.y.iloc[i]),
                        arrowprops=dict(arrowstyle="-|>", color=line_color,
                                        lw=line_width, shrinkA=6, shrinkB=6), zorder=2)
        for cond, row in path.iterrows():
            ax.scatter(row.x, row.y, s=point_size, color=cmap[cond],
                       edgecolor="white", lw=1.2, zorder=3)
            if annotate_points:
                ax.annotate(str(s), (row.x, row.y), ha="center", va="center",
                            fontsize=7, zorder=4)

    for cond, lab in zip(conditions, condition_labels):
        ax.scatter([], [], s=40, color=cmap[cond], label=lab)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if show_title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}", bbox_inches="tight")
    return fig, ax, cent
