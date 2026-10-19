"""
Paste-ready additions for cvp.py: a recalibrated strain-mean baseline.

WHY THIS EXISTS
---------------
`compare_models_condition_holdout_cv` compares ridge against two baselines:

    global_mean : mean of the held-out condition (fit on the training fold)
    row_mean    : each strain's raw mean across the five training conditions

The `row_mean` baseline is not on equal footing with ridge. Ridge fits an
intercept and standardizes its inputs, so it absorbs the held-out condition's
mean and scale for free. `row_mean` gets neither. A feature whose values are
systematically larger at 30 degC than at the average of the other conditions
will make `row_mean` look terrible, and ridge will "beat" it purely by
re-centering -- that is the environment main effect, not a strain-specific
response.

So "ridge > row_mean" cannot distinguish:

    (a) the model learned how each strain responds to the new condition, from
    (b) the model learned that everything is bigger at 30 degC.

`strain_mean_recal` closes that gap. For each feature it fits

    y_heldout = a + b * (that strain's mean across the training conditions)

on the training fold and applies it to the held-out fold. It carries strain
identity plus the environment offset and scale, but it has no way to express a
strain-specific, condition-specific deviation -- it applies the SAME affine map
to every strain. Anything ridge gains over it is interaction structure the model
genuinely recovered.

    ridge - strain_mean_recal  =  the strain x environment interaction captured

That difference, not the raw R^2, is the number that supports a claim about
predicting phenotypic *responses*.

USAGE
-----
    results = compare_models_condition_holdout_cv_recalibrated(
        training_data_finite,
        exp_conditions=exp_conditions,
        ridge_alpha=3000,
        n_splits=16,
        groups=well_ids,          # omit for ungrouped CV; see note below
    )
    summary = summarize_interaction_capture(results, training_feature_labels)
    plot_interaction_capture(summary)

`groups` should be the well index for each row. Rows in this dataset are
well-field pairs, so ungrouped CV puts eight of a held-out well's nine fields
into training and inflates every model in the comparison.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
import arcadia_pycolor as apc


# ----------------------------------------------------------------------------
# row layout
# ----------------------------------------------------------------------------
# cvp.build_training_data lays out rows FoV-major: row = well + fov * n_wells.
# Rows 0..n_wells-1 are FoV 0 for every well, the next n_wells rows are FoV 1,
# and so on. A well's fields therefore sit n_wells rows apart -- they are NOT
# contiguous. Every helper below goes through these two functions so the
# layout lives in one place.
def _well_rows(w, n_wells, n_fovs):
    """Row indices of all fields of view for well `w`."""
    return int(w) + np.arange(n_fovs) * n_wells


def well_groups(n_wells=96, n_fovs=9):
    """Well id for every row, for GroupKFold: np.tile(arange(n_wells), n_fovs)."""
    return np.tile(np.arange(n_wells), n_fovs)



# ----------------------------------------------------------------------------
# per-feature affine recalibration
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# main comparison
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------
def summarize_interaction_capture(results, feature_labels):
    """
    Per-feature table of each model's mean R^2 and the interaction ridge captured.

    `interaction_captured` = ridge - strain_mean_recal. It is the share of
    between-strain variance in a held-out condition that ridge explains and a
    strain-identity-plus-environment model cannot.
    """
    out = {"feature": list(feature_labels)}

    for m, res in results.items():
        mat = np.asarray(res["r2_per_feature"], dtype=float)
        out[f"r2_{m}"] = np.nanmean(mat, axis=0)
        out[f"std_{m}"] = np.nanstd(mat, axis=0)

    df = pd.DataFrame(out)
    df["interaction_captured"] = df["r2_ridge"] - df["r2_strain_mean_recal"]
    df["recalibration_gain"] = df["r2_strain_mean_recal"] - df["r2_strain_mean"]

    sort_key = "interaction_captured"
    if "r2_ridge_strain_profile" in df.columns:
        # capacity-matched: the only thing the full model has that this lacks
        # is condition-resolved structure
        df["interaction_only"] = df["r2_ridge"] - df["r2_ridge_strain_profile"]
        df["multivariate_gain"] = (
            df["r2_ridge_strain_profile"] - df["r2_strain_mean_recal"]
        )
        sort_key = "interaction_only"

    return df.sort_values(sort_key, ascending=False).reset_index(drop=True)


def plot_interaction_capture(
    summary_df,
    value_column=None,
    top_n=None,
    color=apc.aster,
    baseline_color=apc.gray,
    figsize=(7, 8),
    title="Interaction structure captured beyond strain identity",
    save=False,
    figtitle="interaction_capture",
):
    """
    Horizontal bars: ridge R^2 minus recalibrated strain-mean R^2, per feature.

    Bars near zero mean the model is reproducing what strain identity plus the
    environment main effect already gives you.
    """
    if value_column is None:
        value_column = (
            "interaction_only"
            if "interaction_only" in summary_df.columns
            else "interaction_captured"
        )

    df = summary_df if top_n is None else summary_df.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(df))[::-1]

    ax.barh(y, df[value_column], color=color)
    ax.axvline(0, color=baseline_color, linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    xlabels = {
        "interaction_only": r"ridge $R^2$ $-$ condition-averaged ridge $R^2$",
        "interaction_captured": r"ridge $R^2$ $-$ recalibrated strain-mean $R^2$",
    }
    ax.set_xlabel(xlabels.get(value_column, value_column))
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}", bbox_inches="tight")

    return fig, ax


# ----------------------------------------------------------------------------
# Figure 1A: leave-one-condition-out schematic
# ----------------------------------------------------------------------------
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


def _arrow(ax, x0, y0, x1, y1, color="black", lw=1.2, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0))


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


# ----------------------------------------------------------------------------
# Figure 2: making the interaction visible in the images themselves
# ----------------------------------------------------------------------------
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


def plot_strain_pair_cell_montage(
    zroots,
    strain_a,
    strain_b,
    experiment_indices,
    condition_labels,
    training_data=None,
    conditions=None,
    training_feature_labels=None,
    feature=None,
    channel="CY5",
    fov=0,
    n_cells=5,
    crop=96,
    n_wells=96,
    n_fovs=9,
    dead_ids=None,
    cmap="gray",
    strain_colors=None,
    random_state=0,
    figsize=None,
    font_size=9,
    save=False,
    figtitle="strain_pair_cell_montage",
):
    """
    Two strains down the rows, conditions across the columns, real cells in
    every panel, and the reaction norm for the chosen feature on the right.

    Parallel lines in the right-hand panel mean the two strains differ by a
    constant amount -- strain identity. Converging or crossing lines mean the
    difference depends on the condition -- interaction. The images show what
    that difference actually looks like.
    """
    rng = np.random.default_rng(random_state)
    n_cond = len(experiment_indices)
    if strain_colors is None:
        strain_colors = [apc.aster, apc.dress]
    if figsize is None:
        figsize = (2.0 * n_cond + 3.4, 4.4)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, n_cond + 1, width_ratios=[1] * n_cond + [1.5],
                          wspace=0.12, hspace=0.12)

    for r, strain in enumerate([strain_a, strain_b]):
        for c, exp_i in enumerate(experiment_indices):
            ax = fig.add_subplot(gs[r, c])
            crops = _representative_cell_crops(
                zroots[exp_i], fov, strain, channel, n_cells, crop, rng
            )
            if crops:
                strip = np.hstack(crops)
                ax.imshow(strip, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(strain_colors[r]); sp.set_linewidth(1.8)
            if r == 0:
                ax.set_title(condition_labels[c], fontsize=font_size)
            if c == 0:
                ax.set_ylabel(f"strain {strain}", fontsize=font_size,
                              color=strain_colors[r])

    # ---- reaction norm ----
    ax = fig.add_subplot(gs[:, n_cond])
    if training_data is not None and feature is not None:
        conds = list(conditions) if conditions is not None else list(training_data.keys())
        if len(conds) != n_cond:
            raise ValueError(
                f"conditions has {len(conds)} entries but experiment_indices "
                f"has {n_cond}; they must line up or the lines describe "
                f"different conditions than the images."
            )

        labels = list(training_feature_labels)
        if feature in labels:
            feat_idx = labels.index(feature)
        else:
            hits = [i for i, l in enumerate(labels) if feature in l]
            if len(hits) != 1:
                raise ValueError(
                    f"'{feature}' matched {len(hits)} features. Pass the same "
                    f"unambiguous name used for the pair selection, or the "
                    f"lines plot a different feature than the strains were "
                    f"chosen on."
                )
            feat_idx = hits[0]

        z = _strain_means_by_condition(
            training_data, conds, feat_idx,
            n_wells=n_wells, n_fovs=n_fovs, dead_ids=dead_ids,
        )
        xs = np.arange(n_cond)
        za, zb = z[:, strain_a], z[:, strain_b]

        # the GAP between the two strains is the claim, so draw it instead of
        # leaving the reader to subtract two lines by eye
        for i in xs:
            lo, hi = sorted((za[i], zb[i]))
            ax.vlines(i, lo, hi, color="#555555", lw=1.0, ls=":", zorder=1)
            ax.annotate(f"{abs(za[i] - zb[i]):.2f}", xy=(i, (lo + hi) / 2),
                        xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=font_size - 2,
                        color="#555555")

        for r, (strain, vals) in enumerate([(strain_a, za), (strain_b, zb)]):
            ax.plot(xs, vals, "o-", color=strain_colors[r], lw=2, ms=7,
                    label=f"strain {strain}", zorder=3)

        ax.set_xticks(xs)
        ax.set_xticklabels(condition_labels, fontsize=font_size - 1,
                           rotation=30, ha="right")
        ax.set_ylabel(f"{feature}\nz-scored within each condition",
                      fontsize=font_size - 1)
        ax.set_xlim(-0.35, n_cond - 0.35)
        ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=0)
        gaps = np.abs(za - zb)
        ax.set_title(f"gap: {gaps.min():.2f} $\\rightarrow$ {gaps.max():.2f} SD",
                     fontsize=font_size - 1)
        ax.legend(fontsize=font_size - 1, frameon=False, loc="best")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.show()

    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)

    return fig


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


# ----------------------------------------------------------------------------
# Ranking features for two different jobs
# ----------------------------------------------------------------------------
def rank_features_for_use(
    merged_df,
    reliability_df=None,
    training_data=None,
    conditions=None,
    n_wells=96,
    n_fovs=9,
    dead_ids=None,
    keep_verdicts=("use", "conditional"),
    max_mde=None,
    redundancy_threshold=0.9,
):
    """
    Rank features separately for the two jobs they get used for.

    The ranking is two steps, and conflating them is the usual mistake.

    STEP 1 -- reliability gate. `reliab` = (strain + interaction) / (strain +
    interaction + residual): how much of a feature's variance is structured at
    all. This is NOT a tie-breaker between the two jobs -- a low-noise feature
    scores well for both, which is why noise-normalized scores for the two jobs
    correlate at about +0.98 and rank almost identically. Gate on it, then stop
    using it.

    STEP 2 -- allocation. Among features that pass, the structured variance
    splits between strain identity and condition-specific response:

        strain_share = strain / (strain + interaction)
        gxe_share    = interaction / (strain + interaction) = 1 - strain_share

    THAT is the trade-off. Rank by `strain_share` to discriminate strains
    inside one condition; rank by `gxe_share` to detect strain-specific
    environmental response. The two orderings are exact opposites.

    STEP 3 (optional) -- redundancy pruning. Pass `training_data` and a feature
    correlated above `redundancy_threshold` with one already chosen is marked
    redundant. Without this, a top-10 list is one measurement repeated: the
    brightfield size block (area, area_convex, the inertia tensor diagonals and
    eigenvalues) moves together almost perfectly.

    Returns the table sorted by strain_share, with `rank_discriminate`,
    `rank_gxe`, and (when computed) `redundant_with`.
    """
    df = merged_df.copy()

    df["reliab"] = (df["frac_strain"] + df["frac_interaction"]) / (
        df["frac_strain"] + df["frac_interaction"] + df["frac_residual"]
    )
    structured = df["frac_strain"] + df["frac_interaction"]
    df["strain_share"] = df["frac_strain"] / structured
    df["gxe_share"] = df["frac_interaction"] / structured

    if reliability_df is not None:
        cols = [c for c in ("feature", "verdict", "mean_ICC",
                            "MDE_between_strain_SD", "cross_condition_rho")
                if c in reliability_df.columns]
        df = df.merge(reliability_df[cols], on="feature", how="left")
        if keep_verdicts:
            df = df[df["verdict"].isin(keep_verdicts)]
        if max_mde is not None and "MDE_between_strain_SD" in df.columns:
            # a feature can top the G x E ranking and still be unusable because
            # the effect it would take to detect anything is too large
            df = df[df["MDE_between_strain_SD"] <= max_mde]

    df = df.reset_index(drop=True)
    df["rank_discriminate"] = df["strain_share"].rank(ascending=False).astype(int)
    df["rank_gxe"] = df["gxe_share"].rank(ascending=False).astype(int)

    # ---- step 3: redundancy ----
    if training_data is not None:
        conds = list(conditions) if conditions is not None else list(training_data.keys())
        feats = list(df["feature"])
        profiles = {}
        for fi, name in enumerate(feats):
            idx = list(merged_df["feature"]).index(name)
            z = _strain_means_by_condition(training_data, conds, idx,
                                           n_wells=n_wells, n_fovs=n_fovs,
                                           dead_ids=dead_ids)
            profiles[name] = z.ravel()

        ok = np.isfinite(np.vstack([profiles[f] for f in feats])).all(axis=0)
        mat = np.vstack([profiles[f][ok] for f in feats])
        corr = np.corrcoef(mat)

        for col, key in [("redundant_with", "strain_share"),
                         ("redundant_with_gxe", "gxe_share")]:
            chosen, marks = [], {}
            for pos in df[key].sort_values(ascending=False).index:
                dup = next((feats[c] for c in chosen
                            if abs(corr[pos, c]) >= redundancy_threshold), None)
                marks[feats[pos]] = dup
                if dup is None:
                    chosen.append(pos)
            df[col] = df["feature"].map(marks)

    return df.sort_values("strain_share", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# True vs predicted, in t-SNE space, one figure per condition
# ----------------------------------------------------------------------------
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


def plot_true_vs_predicted_tsne_by_condition(
    training_data,
    exp_conditions,
    condition_colors,
    col_titles,
    dead_ids=None,
    ridge_alpha=500,
    n_splits=16,
    groups=None,
    embed="global",                # "global" or "per_condition"
    precomputed_embeddings=None,
    share_limits=None,             # default: True for "global", False otherwise
    perplexity=30,
    tsne_random_state=42,
    random_state=42,
    prediction_color=None,
    marker_size=45,
    ground_truth_alpha=0.85,
    prediction_alpha=0.75,
    arrow_color="#8A8A8A",
    arrow_alpha=0.30,
    arrow_width=0.0025,
    figsize=(5.2, 5.2),
    font_size=10,
    show_legend=True,
    save=False,
    figtitle="true_vs_predicted_tsne",
):
    """
    t-SNE counterpart to plot_true_vs_predicted_pca_by_condition, returning one
    figure per condition instead of a grid.

    Two things differ from the PCA version, and both matter for how you read it.

    t-SNE has no projection step. You cannot fit an embedding on observed data
    and then place predictions into it -- so observed and predicted rows are
    embedded TOGETHER in a single fit. Embedding them separately would produce
    two incomparable spaces and the arrows would be meaningless.

    t-SNE distances are not metric. In the PCA version an arrow's length is a
    real distance in feature space; here it is not. Arrows show whether a
    prediction lands in the right neighbourhood, not how large the error is.
    Report error with R^2, never by describing arrow lengths.

    `embed`:
      "global"        -- ONE t-SNE fit over every condition's observed and
                         predicted rows stacked together (6 conditions x 2
                         matrices x n rows), then sliced back apart per
                         condition. All figures share one coordinate system,
                         so a point in the 16 degC figure and a point in the
                         30 degC figure at the same position really are
                         neighbours in phenotype space. Because the axes are
                         only meaningful if they also share limits, this mode
                         sets identical xlim/ylim on every figure -- so a
                         condition occupying one corner of the global map
                         renders as a cluster in that corner, with empty space
                         around it. That is the honest picture.
      "per_condition" -- an independent t-SNE per condition, matching the PCA
                         function's behaviour. Each figure uses its whole
                         canvas for within-condition structure, which reads
                         better on its own, but the axes mean something
                         different in each one and positions must NOT be
                         compared between figures. Say so in the legend.

    Predictions are out-of-fold, unlike `train_and_predict`, which predicts on
    its own training rows. Pass `groups` (well index per row) so no field of a
    held-out well is used to predict it.

    The embedding cannot reuse a t-SNE fit from elsewhere in the notebook.
    sklearn's TSNE has no transform(), so an embedding fit on observed data
    alone has no position to offer the predicted rows -- they must be in the
    same fit. To avoid refitting when you only want to restyle the figures,
    pass the `embeddings` dict returned by an earlier call back in as
    `precomputed_embeddings`; prediction and t-SNE are then both skipped.

    Returns (figures, embeddings): a list of (fig, ax) in condition order, and
    a dict mapping condition -> (observed_xy, predicted_xy).
    """
    from sklearn.manifold import TSNE

    if not (len(exp_conditions) == len(col_titles) == len(condition_colors)):
        raise ValueError(
            f"exp_conditions ({len(exp_conditions)}), col_titles "
            f"({len(col_titles)}) and condition_colors "
            f"({len(condition_colors)}) must be the same length."
        )
    if prediction_color is None:
        prediction_color = apc.gray if hasattr(apc, "gray") else "#9AA0A6"

    if precomputed_embeddings is not None:
        missing = [c for c in exp_conditions if c not in precomputed_embeddings]
        if missing:
            raise ValueError(
                f"precomputed_embeddings is missing {missing}; pass the dict "
                f"returned by an earlier call with the same exp_conditions."
            )
        embeddings = {c: precomputed_embeddings[c] for c in exp_conditions}
        n_rows = embeddings[exp_conditions[0]][0].shape[0]
    else:
        # ---- out-of-fold predictions per condition ---------------------
        stacks = {}
        for cond in exp_conditions:
            y_true = np.asarray(training_data[cond], dtype=float)
            y_pred = _out_of_fold_predictions(
                training_data, exp_conditions, cond, ridge_alpha, n_splits,
                groups, random_state,
            )
            stacks[cond] = (y_true, y_pred)

        n_rows = stacks[exp_conditions[0]][0].shape[0]

        def _fit(mat):
            scaled = StandardScaler().fit_transform(mat)
            n = scaled.shape[0]
            return TSNE(
                n_components=2,
                perplexity=min(perplexity, max(5, (n - 1) // 3)),
                init="pca",
                random_state=tsne_random_state,
            ).fit_transform(scaled)

        embeddings = {}
        if embed == "global":
            big = np.vstack([np.vstack(stacks[c]) for c in exp_conditions])
            emb = _fit(big)
            for k, cond in enumerate(exp_conditions):
                block = emb[k * 2 * n_rows:(k + 1) * 2 * n_rows]
                embeddings[cond] = (block[:n_rows], block[n_rows:])
        elif embed == "per_condition":
            for cond in exp_conditions:
                emb = _fit(np.vstack(stacks[cond]))
                embeddings[cond] = (emb[:n_rows], emb[n_rows:])
        else:
            raise ValueError("embed must be 'global' or 'per_condition'")

    # ---- one figure per condition --------------------------------------
    keep = np.setdiff1d(np.arange(n_rows), np.asarray(dead_ids or [], dtype=int))

    if share_limits is None:
        # shared axes are what make a global embedding comparable across
        # figures; a per-condition fit has nothing to share
        share_limits = (embed == "global")
    if share_limits:
        allpts = np.vstack([np.vstack([e[keep] for e in embeddings[c]])
                            for c in exp_conditions])
        pad = 0.06 * np.ptp(allpts, axis=0)
        xlim = (allpts[:, 0].min() - pad[0], allpts[:, 0].max() + pad[0])
        ylim = (allpts[:, 1].min() - pad[1], allpts[:, 1].max() + pad[1])

    figures = []
    for i, cond in enumerate(exp_conditions):
        gt, pred = (e[keep] for e in embeddings[cond])

        fig, ax = plt.subplots(figsize=figsize)
        ax.quiver(gt[:, 0], gt[:, 1], pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1],
                  angles="xy", scale_units="xy", scale=1, color=arrow_color,
                  alpha=arrow_alpha, width=arrow_width, zorder=1)
        ax.scatter(pred[:, 0], pred[:, 1], s=marker_size, color=prediction_color,
                   alpha=prediction_alpha, label="predicted", zorder=2,
                   edgecolor="white", linewidth=0.5)
        ax.scatter(gt[:, 0], gt[:, 1], s=marker_size, color=condition_colors[i],
                   alpha=ground_truth_alpha, label="observed", zorder=3,
                   edgecolor="white", linewidth=0.5)

        ax.set_title(col_titles[i], fontsize=font_size + 1)
        ax.set_xlabel("t-SNE 1", fontsize=font_size - 1)
        ax.set_ylabel("t-SNE 2", fontsize=font_size - 1)
        ax.set_xticks([]); ax.set_yticks([])
        if share_limits:
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box" if share_limits else "datalim")
        if show_legend:
            ax.legend(fontsize=font_size - 1, frameon=False, loc="best")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        plt.tight_layout()
        plt.show()

        if save:
            safe = cond.replace(" ", "_").replace("/", "_")
            for ext in ("png", "pdf", "svg"):
                fig.savefig(f"plots/chlorella_pub_fig_{figtitle}_{safe}.{ext}",
                            bbox_inches="tight", dpi=300)
        figures.append((fig, ax))

    return figures, embeddings


# ----------------------------------------------------------------------------
# Distribution of prediction displacements (the arrow lengths)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Is the model calibrated, and are its residuals reproducible?
# ----------------------------------------------------------------------------
def _oof_by_condition(training_data, exp_conditions, ridge_alpha, n_splits,
                      groups, random_state):
    """{condition: (observed, out-of-fold predicted)} at row level."""
    return {
        c: (np.asarray(training_data[c], dtype=float),
            _out_of_fold_predictions(training_data, exp_conditions, c,
                                     ridge_alpha, n_splits, groups, random_state))
        for c in exp_conditions
    }


def _to_strain_means(mat, n_wells, n_fovs, keep_wells):
    out = np.array([np.nanmean(mat[_well_rows(w, n_wells, n_fovs)], axis=0)
                    for w in range(n_wells)])
    return out[keep_wells]


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


def compute_residual_reproducibility(
    training_data, exp_conditions, training_feature_labels,
    ridge_alpha=500, n_splits=16, groups=None, random_state=42,
    n_wells=96, n_fovs=9, dead_ids=None, n_repeats=50, calibrate=True,
):
    """
    Is a strain's departure from prediction a stable property of that strain?

    For each condition: take out-of-fold residuals (observed - predicted) at
    field level, split each well's fields into two halves, average within each
    half, and correlate the two halves across strains. Repeated `n_repeats`
    times with different random halvings.

    WHAT THIS DOES AND DOES NOT ESTABLISH.

    It shows whether a strain's deviation from prediction exceeds field-to-field
    measurement noise -- i.e. whether "departs from prediction" is a stable,
    repeatable property of that well rather than a coin flip. That is exactly
    what a screen needs: an unreproducible residual makes any knockout screen
    built on it worthless.

    It does NOT show that the residual is biology the model could never have
    learned. Any error that is a function of the STRAIN reproduces across field
    halves, and model deficiency is a function of the strain too. On simulated
    data with no reproducible strain quirk whatsoever, residuals still gave a
    corrected split-half correlation of 0.89, because ridge cannot recover each
    strain's level exactly and the shortfall is consistent across that well's
    fields. Calibration removes the part of that which is linear in the
    prediction; the rest is not separable by this test.

    So report it as "residuals are reproducible, therefore screenable", never as
    "residuals are unlearnable biology". Separating those two needs a different
    experiment -- a learning curve, or asking whether the residuals are
    themselves predictable from the strain's other features.

    `split_half_r` is the reliability of a half-sized measurement;
    `spearman_brown` corrects it to the full nine-field measurement.
    `null_r` shuffles strain labels between halves.

    `calibrate=True` (the default, and important) measures residuals against
    the CALIBRATED prediction a + b*predicted rather than the raw prediction.
    Ridge shrinks, so a raw residual contains a systematic term proportional to
    the strain's own effect -- which reproduces perfectly across field halves
    and has nothing to do with biology. On simulated data with no reproducible
    strain quirk at all, raw residuals still gave a split-half correlation of
    0.91; calibrating first removes that term so the number means what it
    claims to. Set calibrate=False only to see the size of the artifact.
    """
    oof = _oof_by_condition(training_data, exp_conditions, ridge_alpha,
                            n_splits, groups, random_state)
    dead_wells = sorted({int(d) % n_wells for d in (dead_ids or [])})
    keep_wells = np.array([w for w in range(n_wells) if w not in dead_wells])
    rng = np.random.default_rng(random_state)

    def _corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or np.var(a[m]) <= 0 or np.var(b[m]) <= 0:
            return np.nan
        return np.corrcoef(a[m], b[m])[0, 1]

    rows = []
    for cond, (obs, pred) in oof.items():
        if calibrate:
            # strip the systematic under-dispersion before asking what is left
            adj = np.empty_like(pred)
            for j in range(pred.shape[1]):
                x, y = pred[:, j], obs[:, j]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 3 or np.var(x[m]) <= 0:
                    adj[:, j] = x
                    continue
                b = np.cov(x[m], y[m], bias=True)[0, 1] / np.var(x[m])
                a = y[m].mean() - b * x[m].mean()
                adj[:, j] = a + b * x
            resid = obs - adj
        else:
            resid = obs - pred
        n_feat = resid.shape[1]
        obs_r = np.full((n_repeats, n_feat), np.nan)
        null_r = np.full((n_repeats, n_feat), np.nan)

        for rep in range(n_repeats):
            order = rng.permutation(n_fovs)
            ha, hb = order[:n_fovs // 2], order[n_fovs // 2:]
            A = np.array([resid[w + ha * n_wells].mean(axis=0) for w in keep_wells])
            B = np.array([resid[w + hb * n_wells].mean(axis=0) for w in keep_wells])
            shuffled = rng.permutation(len(keep_wells))
            for j in range(n_feat):
                obs_r[rep, j] = _corr(A[:, j], B[:, j])
                null_r[rep, j] = _corr(A[:, j], B[shuffled, j])

        mean_r = np.nanmean(obs_r, axis=0)
        sb = 2 * mean_r / (1 + mean_r)          # Spearman-Brown, half -> full
        for j, name in enumerate(training_feature_labels):
            rows.append({"condition": cond, "feature": name,
                         "split_half_r": mean_r[j],
                         "spearman_brown": sb[j],
                         "null_r": np.nanmean(null_r[:, j])})
    return pd.DataFrame(rows)


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


def plot_prediction_calibration(calibration_df, exp_conditions, col_titles=None,
                                condition_colors=None, **kwargs):
    """One box per condition, one point per feature. The dashed line at 1.0 is
    perfect calibration; above it the model is under-dispersed."""
    kwargs.setdefault("figtitle", "prediction_calibration_slopes")
    return _box_by_condition(
        calibration_df, "slope", exp_conditions, col_titles, condition_colors,
        ylabel="calibration slope\n(observed on predicted)",
        reference=1.0, reference_label="perfect calibration", **kwargs)


def plot_residual_reproducibility(residual_df, exp_conditions, col_titles=None,
                                  condition_colors=None, value_col="spearman_brown",
                                  **kwargs):
    """One box per condition, one point per feature, against the shuffled-strain
    null band. Boxes clear of the band mean residuals are a strain property."""
    kwargs.setdefault("figtitle", "residual_reproducibility")
    return _box_by_condition(
        residual_df, value_col, exp_conditions, col_titles, condition_colors,
        ylabel="split-half correlation of residuals\n(Spearman-Brown corrected)",
        reference=0.0, null_col="null_r", **kwargs)


# ----------------------------------------------------------------------------
# Identification: can the model's prediction pick the right strain / condition?
# ----------------------------------------------------------------------------
def _zscore(mat, mu=None, sd=None):
    mu = mat.mean(axis=0) if mu is None else mu
    sd = mat.std(axis=0) if sd is None else sd
    sd = np.where(sd > 0, sd, 1.0)
    return (mat - mu) / sd, mu, sd


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


def identify_strains_vs_n_fov(
    training_data, exp_conditions,
    n_fov_list=(1, 2, 3, 5, 7, 9),
    ridge_alpha=500, n_splits=16, groups=None, random_state=42,
    n_wells=96, n_fovs=9, dead_ids=None, n_repeats=20,
    independence_factor=1.0,
):
    """
    How does identification improve as you use more fields of view per strain?

    THE MATH FIRST, because it decides what this can and cannot buy you.
    Under the Gaussian model, accumulating k independent observations gives

        sum_f ||x_f - m||^2  =  k * ||xbar_k - m||^2  +  (scatter within the well)

    and the second term does not depend on the candidate m. So the posterior
    from k fields is IDENTICAL to the posterior from their mean with sigma^2/k.
    Averaging fields already uses every cell in them; extra fields do not add
    new kinds of information, they sharpen the same estimate. That is why this
    function reports a curve rather than a better number: it tells you how many
    fields a screen needs per strain, which is a design question, not a claim
    that the method got smarter.

    Genuinely new information requires going beyond the mean -- per-cell
    distributions (spread, skew, quantiles of each feature within a well),
    which the current feature set discards at extraction time.

    independence_factor: fields from one well share well-level and positional
    effects, so treating k fields as k independent draws overstates confidence.
    Values above 1 temper the accumulated log-likelihood by that factor; set it
    to the ratio of nominal to effective sample size if you can estimate it
    (the split-half reliability of your field means is a reasonable guide).

    Returns a DataFrame: condition, n_fov, accuracy, top5, median_p_correct.
    """
    oof = _oof_by_condition(training_data, exp_conditions, ridge_alpha,
                            n_splits, groups, random_state)
    dead = sorted({int(d) % n_wells for d in (dead_ids or [])})
    keep = np.array([w for w in range(n_wells) if w not in dead])
    n = len(keep)
    rng = np.random.default_rng(random_state)

    records = []
    for cond, (obs, pred) in oof.items():
        # z-score on the STRAIN-level observed distribution -- the same scaling
        # identify_strains_from_predictions uses, so that k = n_fovs reproduces
        # its result exactly rather than measuring distance in a different space
        _, mu, sd = _zscore(_to_strain_means(obs, n_wells, n_fovs, keep))
        O = np.stack([(obs[_well_rows(w, n_wells, n_fovs)] - mu) / sd for w in keep])
        P = np.stack([((pred[_well_rows(w, n_wells, n_fovs)] - mu) / sd).mean(axis=0)
                      for w in keep])                      # (n, n_feat) templates
        n_feat = P.shape[1]

        # per-field residual variance, leave-one-out over strains
        own = np.array([np.mean(np.sum((O[i] - P[i]) ** 2, axis=1)) for i in range(n)])
        sigma2 = np.array([(own.sum() - own[i]) / max(n - 1, 1) / n_feat
                           for i in range(n)])
        sigma2 = np.where(sigma2 > 0, sigma2, np.nanmean(sigma2))

        for k in n_fov_list:
            k = int(min(k, n_fovs))
            acc, t5, pc = [], [], []
            for _ in range(n_repeats):
                sel = np.array([rng.choice(n_fovs, size=k, replace=False)
                                for _ in range(n)])
                # sum of squared distances over the chosen fields
                D2 = np.zeros((n, n))
                for i in range(n):
                    X = O[i][sel[i]]                       # (k, n_feat)
                    D2[i] = np.sum((X[:, None, :] - P[None, :, :]) ** 2, axis=(0, 2))
                logits = -D2 / (2 * sigma2[:, None] * independence_factor)
                logits -= logits.max(axis=1, keepdims=True)
                Pr = np.exp(logits); Pr /= Pr.sum(axis=1, keepdims=True)

                assigned = D2.argmin(axis=1)
                order = D2.argsort(axis=1)
                ranks = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(n)])
                acc.append((assigned == np.arange(n)).mean())
                t5.append((ranks <= 5).mean())
                pc.append(np.median([Pr[i, i] for i in range(n)]))

            records.append({"condition": cond, "n_fov": k,
                            "accuracy": float(np.mean(acc)),
                            "top5": float(np.mean(t5)),
                            "median_p_correct": float(np.mean(pc)),
                            "chance": 1.0 / n})
    return pd.DataFrame(records)


def plot_identification_vs_n_fov(
    df, exp_conditions, col_titles=None, condition_colors=None,
    value_col="median_p_correct", figsize=(6.0, 4.2), font_size=10,
    save=False, figtitle="identification_vs_n_fov",
):
    """Identification performance against the number of fields of view used.
    Read it as a design curve: how much imaging buys how much certainty."""
    labels = list(col_titles) if col_titles is not None else list(exp_conditions)
    if condition_colors is None:
        condition_colors = [apc.aster] * len(exp_conditions)

    fig, ax = plt.subplots(figsize=figsize)
    for cond, label, color in zip(exp_conditions, labels, condition_colors):
        sub = df[df.condition == cond].sort_values("n_fov")
        ax.plot(sub.n_fov, sub[value_col], "o-", color=color, lw=2, ms=7,
                markeredgecolor="white", markeredgewidth=1.0, label=label)

    chance = float(df["chance"].iloc[0])
    ax.axhline(chance, color="#999999", ls="--", lw=1.0)
    ax.annotate(f"chance ({chance:.1%})", xy=(0.99, chance),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=font_size - 2, color="#777777")

    ax.set_xlabel("fields of view used per strain", fontsize=font_size - 1)
    ax.set_ylabel({"median_p_correct": "median P(correct identity)",
                   "accuracy": "top-1 accuracy",
                   "top5": "top-5 accuracy"}.get(value_col, value_col),
                  fontsize=font_size - 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=font_size - 2, frameon=False, loc="best")
    ax.tick_params(labelsize=font_size - 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    plt.tight_layout(); plt.show()
    if save:
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"plots/chlorella_pub_fig_{figtitle}.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig, ax


def diagnose_confusion_structure(
    results, condition, plate_rows=8, plate_cols=12, layouts=("row_major", "column_major"),
    top_offsets=8,
):
    """
    Why does the confusion matrix have off-diagonal stripes?

    A stripe at offset k means strain i is repeatedly mistaken for strain i+k.
    On a plate that is the signature of PLATE POSITION: neighbouring wells share
    edge effects, evaporation, illumination and thermal gradients, so their
    phenotypes resemble each other for reasons that have nothing to do with
    genotype. Offsets of +/-1 are neighbours along one plate axis; +/-8 or
    +/-12 are neighbours along the other, depending on fill order.

    INDEX VS WELL. The matrix is indexed over SURVIVING strains, not well
    numbers -- dead wells were dropped -- so an offset of 12 in the matrix is
    not an offset of 12 on the plate. This maps every index back through
    results[condition]["strains"] before computing anything.

    Returns {"by_offset": DataFrame(well_offset, mean_probability, n_pairs),
             "by_layout": {layout: DataFrame(relation, mean_probability, n_pairs)},
             "baseline": mean off-diagonal probability}.

    Compare each relation's mean probability to `baseline`. Neighbour relations
    well above baseline mean position is leaking into identity, and the honest
    fix is to include plate row and column as covariates, or to report
    identification among non-adjacent wells only.
    """
    r = results[condition]
    P = r.get("probability")
    if P is None:
        P = r["confusion"].astype(float)
    wells = np.asarray(r["strains"])
    n = len(wells)

    iu = np.arange(n)
    off_mask = ~np.eye(n, dtype=bool)
    baseline = float(P[off_mask].mean())

    # --- by raw well-number offset ---
    offs = wells[None, :] - wells[:, None]
    rows = []
    for k in range(1, plate_rows * plate_cols):
        m = (np.abs(offs) == k) & off_mask
        if m.sum() == 0:
            continue
        rows.append({"well_offset": k, "mean_probability": float(P[m].mean()),
                     "n_pairs": int(m.sum()),
                     "vs_baseline": float(P[m].mean() / baseline)})
    by_offset = (pd.DataFrame(rows)
                 .sort_values("mean_probability", ascending=False)
                 .head(top_offsets).reset_index(drop=True))

    # --- by plate geometry, under each candidate fill order ---
    by_layout = {}
    for layout in layouts:
        if layout == "row_major":
            rr, cc = wells // plate_cols, wells % plate_cols
        elif layout == "column_major":
            rr, cc = wells % plate_rows, wells // plate_rows
        else:
            raise ValueError("layout must be 'row_major' or 'column_major'")

        dr = np.abs(rr[None, :] - rr[:, None])
        dc = np.abs(cc[None, :] - cc[:, None])
        rel = {
            "same row, adjacent column": (dr == 0) & (dc == 1),
            "same column, adjacent row": (dc == 0) & (dr == 1),
            "diagonal neighbour": (dr == 1) & (dc == 1),
            "2-4 wells away": (np.maximum(dr, dc) >= 2) & (np.maximum(dr, dc) <= 4),
            "far apart (>4)": np.maximum(dr, dc) > 4,
        }
        recs = []
        for name, m in rel.items():
            m = m & off_mask
            if m.sum() == 0:
                continue
            recs.append({"relation": name, "mean_probability": float(P[m].mean()),
                         "n_pairs": int(m.sum()),
                         "vs_baseline": float(P[m].mean() / baseline)})
        by_layout[layout] = pd.DataFrame(recs)

    return {"by_offset": by_offset, "by_layout": by_layout, "baseline": baseline}


# ----------------------------------------------------------------------------
# Plate positional effects: measure them, then remove them
# ----------------------------------------------------------------------------
def _well_to_rc(wells, plate_rows, plate_cols, layout):
    if layout == "row_major":
        return wells // plate_cols, wells % plate_cols
    if layout == "column_major":
        return wells % plate_rows, wells // plate_rows
    raise ValueError("layout must be 'row_major' or 'column_major'")


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


def correct_positional_effects(
    training_data, exp_conditions,
    n_wells=96, n_fovs=9, dead_ids=None,
    plate_rows=8, plate_cols=12, layout="row_major",
    method="median_polish", return_effects=False,
):
    """
    Subtract additive plate row and column effects, returning a corrected copy
    of `training_data` with the same shape and row order.

    THE CONFOUND YOU MUST STATE. Each strain occupies exactly one well, so
    genotype and position are partially confounded: any correction that removes
    position also removes whatever genuine strain variation happens to align
    with the plate axes. The correction is valid only if strains were assigned
    to wells at random with respect to phenotype. If they were arrayed
    systematically -- by picking order, by colony size, controls at fixed
    positions -- median polish will delete real biology, and the fix is a
    randomised re-array rather than an analysis step.

    Because of that, treat this as a SENSITIVITY ANALYSIS, not a replacement
    for the raw data: report the main result on uncorrected values, then show
    that it survives correction. A result that only appears in one of the two
    is not a result.

    method="median_polish" is robust to a few extreme wells; "row_col_mean"
    subtracts plain row and column means and is more easily perturbed by them.

    Returns the corrected dict, or (corrected, effects_df) with return_effects.
    """
    dead = sorted({int(d) % n_wells for d in (dead_ids or [])})
    rr, cc = _well_to_rc(np.arange(n_wells), plate_rows, plate_cols, layout)

    corrected, effect_rows = {}, []
    for cond in exp_conditions:
        M = np.asarray(training_data[cond], dtype=float).copy()
        strain_means = np.array([np.nanmean(M[_well_rows(w, n_wells, n_fovs)], axis=0)
                                 for w in range(n_wells)])
        strain_means[dead] = np.nan

        for j in range(M.shape[1]):
            grid = np.full((plate_rows, plate_cols), np.nan)
            grid[rr, cc] = strain_means[:, j]
            if not np.isfinite(np.nanvar(grid)) or np.nanvar(grid) <= 0:
                continue

            if method == "median_polish":
                _, reff, ceff, _ = _median_polish(grid)
            elif method == "row_col_mean":
                gm = np.nanmean(grid)
                reff = np.nan_to_num(np.nanmean(grid, axis=1) - gm)
                ceff = np.nan_to_num(np.nanmean(grid, axis=0) - gm)
            else:
                raise ValueError("method must be 'median_polish' or 'row_col_mean'")

            adjust = reff[rr] + ceff[cc]                 # per well
            M[:, j] -= np.tile(adjust, n_fovs)          # rows are well + fov * n_wells

            for w in range(n_wells):
                effect_rows.append({"condition": cond, "feature_index": j,
                                    "well": w, "row": int(rr[w]),
                                    "col": int(cc[w]), "adjustment": float(adjust[w])})
        corrected[cond] = M

    if return_effects:
        return corrected, pd.DataFrame(effect_rows)
    return corrected


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


# ==============================================================================================================
# INTERACTIVE (PLOTLY) VERSIONS OF THE PUB HEATMAPS
# ==============================================================================================================
#
# Drop-in wrappers for five heatmap functions in cvp.py:
#
#     cba.plot_all_strain_feature_heatmap
#     cba.plot_feature_icc_heatmap
#     cba.plot_sorted_r2_per_feature
#     cba.plot_within_between_ratio_heatmap
#     cba.plot_variance_partitioning_heatmap
#
# Each takes exactly the same arguments as its cvp.py counterpart, plus a few
# keyword-only extras (backend=..., hover_*=..., plotly_*=...).
#
#   backend="static"  (default) -> calls the cvp.py function untouched and returns
#                                  exactly what it returns: fig, ax, plot_df.
#   backend="plotly"            -> interactive figure only: returns pfig, plot_df.
#                                  The static figure is built silently (never
#                                  shown or saved) so cell order, colors, and
#                                  color limits are identical to the static one.
#   backend="both"              -> static figure shown/saved as usual AND the
#                                  interactive one: returns fig, ax, plot_df, pfig.
#
# Nothing in cvp.py is modified. cvp.py is imported lazily inside each call
# because cvp.py itself imports from this module.
#
# HOVER CUSTOMIZATION
# -------------------
#   hover_fields   : list of field names to show, in order. None -> a sensible
#                    default per function. "all" -> every available field.
#                    Call cba.list_hover_fields("<function name>") to see them.
#   hover_data     : dict {name: data} of your own fields. Each value can be
#                      - a DataFrame shaped like plot_df (either orientation)
#                      - a Series / dict keyed by row OR column labels
#                        (e.g. strain -> genotype, feature -> description);
#                        it is broadcast along the other axis
#                      - a 2D array shaped like plot_df, or a scalar
#                    Custom fields are appended to the default fields unless
#                    you name them in hover_fields.
#   hover_labels   : dict {name: "Display label"} to rename fields.
#   hover_formats  : dict {name: d3 format} e.g. {"value": ".2f", "raw_mean": ".3g"}.
#   hover_template : full custom template. Use {name} or {name:.2f} for any field,
#                    e.g. "<b>{feature}</b><br>ICC = {value:.2f}<br>{genotype}".
#                    Native Plotly syntax (%{customdata[0]}, %{z}) also works.
#
# LAYOUT
# ------
#   plotly_show_xticklabels / plotly_show_yticklabels : None -> same as the
#       static figure; False drops the labels (they remain in the hover).
#   plotly_width / plotly_height : pixels. None -> figsize * plotly_dpi.
#   plotly_show  : display the figure in the notebook (default True).
#   plotly_save_formats : tuple from ("html", "json", "png", "svg", "pdf").
#       Written when the wrapped function's save=True. Image formats need kaleido.
#   plotly_include_plotlyjs : passed to write_html ("cdn" keeps files small).
#   plotly_layout : dict of extra layout properties applied last.
#   plotly_annot : True/False shows/hides the numbers in the Plotly cells only.
#       None (default) follows the function's own annot setting.
#   plotly_cbar : True/False shows/hides the colorbar on the Plotly figure only.
#       None (default) follows the function's own cbar setting.
#   plotly_hover_space : blank pixels added to the right of the figure so the
#       hover box has room to open. "auto" (default) adds just enough when the
#       heatmap is too narrow for the box; an int sets it exactly; 0 turns it off.
#       The heatmap itself keeps its size (plotly_width is not reduced).
# ==============================================================================================================

import inspect as _inspect
import warnings
import re as _re

import matplotlib.colors as _mcolors


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
)


def _cvp_module():
    import cvp  # lazy: cvp.py imports this module at load time
    return cvp


def _split_plotly_kwargs(kwargs):
    extras = {k: kwargs.pop(k, v) for k, v in _PLOTLY_EXTRAS.items()}
    if extras["backend"] not in ("static", "plotly", "both"):
        raise ValueError('backend must be "static", "plotly", or "both"')
    return extras


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


def _resolved_arguments(fn, args, kwargs):
    bound = _inspect.signature(fn).bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _css_color(c):
    try:
        return _mcolors.to_hex(c)
    except (ValueError, TypeError):
        return str(getattr(c, "hex_code", c))


def _plotly_colorscale(cmap, n=64):
    return [[i / (n - 1), _mcolors.to_hex(cmap(i / (n - 1)))] for i in range(n)]


def _pt(lw):
    """matplotlib points -> CSS pixels."""
    return float(lw) * 96.0 / 72.0


# ---- hover data alignment ------------------------------------------------------------------------------------
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


def _is_numeric_array(a):
    for v in np.ravel(a):
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            return False
    return True


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


def _grid(df_rows, df_cols, fn):
    """DataFrame (rows x cols) with fn(r, c) in each cell."""
    return pd.DataFrame([[fn(r, c) for c in df_cols] for r in df_rows], index=df_rows, columns=df_cols)


def _label_grid(plot_df, axis):
    """Broadcast the row (axis=0) or column (axis=1) labels over the grid."""
    if axis == 0:
        return pd.Series(list(plot_df.index), index=plot_df.index)
    return pd.Series(list(plot_df.columns), index=plot_df.columns)


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


# ---- the figure ----------------------------------------------------------------------------------------------
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


def _dispatch(fn, args, kwargs, builder, default_figtitle, style=None):
    """Shared driver for every wrapper."""
    extras = _split_plotly_kwargs(kwargs)
    backend = extras["backend"]

    if backend == "static":
        return fn(*args, **kwargs)

    P = _resolved_arguments(fn, args, kwargs)
    if backend == "both":
        fig, ax, plot_df = fn(*args, **kwargs)
    else:
        fig, ax, plot_df = _run_static_silently(fn, args, kwargs)

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


def _f(data, label, fmt=None):
    return dict(data=data, label=label, fmt=fmt)


def _rank_desc(df, axis):
    """1 = highest value along the given axis."""
    return df.rank(axis=axis, ascending=False, method="min")


# ---- 1. plot_all_strain_feature_heatmap ----------------------------------------------------------------------
def _fields_all_strain(P, plot_df):
    cvp = _cvp_module()
    td = P["training_data"]
    condition = cvp._resolve_condition_key(P["temperature"], P["growth_stage"], list(td.keys()))
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


def _figtitle_all_strain(P):
    cvp = _cvp_module()
    condition = cvp._resolve_condition_key(P["temperature"], P["growth_stage"], list(P["training_data"].keys()))
    fovs = P["fovs"]
    if isinstance(fovs, str):
        n_rows = np.asarray(P["training_data"][condition]).shape[0]
        fov_list = list(range(P["n_fovs"] or n_rows // P["n_wells"]))
    elif isinstance(fovs, (int, np.integer)):
        fov_list = [int(fovs)]
    else:
        fov_list = list(fovs)
    return f"all_strain_feature_heatmap_{condition}_fov" + "_".join(str(f) for f in fov_list)


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
    return _dispatch(_cvp_module().plot_all_strain_feature_heatmap, args, kwargs,
                     _fields_all_strain, _figtitle_all_strain)


# ---- 2. plot_feature_icc_heatmap -----------------------------------------------------------------------------
_ICC_LABELS = {
    "icc_single": "ICC(1,1)",
    "icc_mean": "ICC(1,k)",
    "split_half": "Split-half rho",
    "cross_condition_rho": "Cross-condition rho",
    "fstat": "F statistic",
    "pvalue": "Permutation p",
}


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
    return _dispatch(_cvp_module().plot_feature_icc_heatmap, args, kwargs,
                     _fields_icc, lambda P: P["figtitle"], _style_icc)


# ---- 3. plot_sorted_r2_per_feature ---------------------------------------------------------------------------
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


def _style_r2(P):
    return dict(annot=P["annot"], annot_fmt=".2f", annot_fontsize=P["annot_fontsize"],
                annot_color=P["annot_color"])


def plot_sorted_r2_per_feature(*args, **kwargs):
    """
    cvp.plot_sorted_r2_per_feature with an optional interactive version.

    Hover fields: feature, condition, value (R²), mean_r2, min_r2, max_r2,
    rank_by_mean, rank_in_condition, r2_by_condition, and, when present in the
    results dict, condition_r2_overall, condition_rmse, condition_p_value.

    Note: like the original, save defaults to True.
    """
    return _dispatch(_cvp_module().plot_sorted_r2_per_feature, args, kwargs,
                     _fields_r2, lambda P: P["figtitle"], _style_r2)


# ---- 4. plot_within_between_ratio_heatmap --------------------------------------------------------------------
def _fields_ratio(P, plot_df):
    ratio = np.asarray(P["ratio_mat"], dtype=float)
    cond = P["col_titles"] if P["col_titles"] is not None else P["condition_order"]
    cond = [str(c) for c in cond]
    labels = [str(l) for l in P["training_feature_labels"]]
    canon = pd.DataFrame(ratio, index=cond, columns=labels)       # conditions x features

    vertical = P["orientation"] == "vertical"
    feat_axis, cond_axis = (0, 1) if vertical else (1, 0)

    mean_log2 = canon.mean(axis=0)
    by_cond = pd.Series(
        ["<br>   " + "<br>   ".join(f"{c}: {v:.2f}" for c, v in zip(cond, canon[f].to_numpy())) for f in labels],
        index=labels,
    )
    Z = plot_df.to_numpy(dtype=float)
    fields = {
        "feature": _f(_label_grid(plot_df, feat_axis), "Feature"),
        "condition": _f(_label_grid(plot_df, cond_axis), "Condition"),
        "value": _f(Z, "log2(between / within)", ".2f"),
        "fold_ratio": _f(np.power(2.0, Z), "Between / within", ".3g"),
        "mean_log2_ratio": _f(mean_log2, "Mean log2 ratio (all conditions)", ".2f"),
        "min_log2_ratio": _f(canon.min(axis=0), "Min log2 ratio", ".2f"),
        "max_log2_ratio": _f(canon.max(axis=0), "Max log2 ratio", ".2f"),
        "rank_by_mean": _f(mean_log2.rank(ascending=False, method="min"),
                           f"Rank by mean (of {len(labels)})", ".0f"),
        "rank_in_condition": _f(_rank_desc(canon, axis=1), "Rank in this condition", ".0f"),
        "ratio_by_condition": _f(by_cond, "log2 ratio by condition"),
    }
    return fields, ["feature", "condition", "value", "fold_ratio", "mean_log2_ratio", "rank_by_mean"]


def plot_within_between_ratio_heatmap(*args, **kwargs):
    """
    cvp.plot_within_between_ratio_heatmap with an optional interactive version.

    Hover fields: feature, condition, value (log2 ratio), fold_ratio (2**value),
    mean_log2_ratio, min_log2_ratio, max_log2_ratio, rank_by_mean,
    rank_in_condition, ratio_by_condition.
    """
    return _dispatch(_cvp_module().plot_within_between_ratio_heatmap, args, kwargs,
                     _fields_ratio, lambda P: P["figtitle"])


# ---- 5. plot_variance_partitioning_heatmap -------------------------------------------------------------------
_VP_LABELS = {
    "frac_strain": "Strain", "frac_environment": "Environment",
    "frac_interaction": "Strain x environment", "frac_residual": "Residual",
    "ss_strain": "SS strain", "ss_environment": "SS environment",
    "ss_interaction": "SS interaction", "ss_residual": "SS residual",
}


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


def plot_variance_partitioning_heatmap(*args, **kwargs):
    """
    cvp.plot_variance_partitioning_heatmap with an optional interactive version.

    Hover fields: feature, component, value, dominant_component, rank, and every
    column of variance_df by its own name (frac_strain, frac_environment,
    frac_interaction, frac_residual, ss_strain, ...). Add a column to
    variance_df and it becomes a hover field automatically.
    Default: feature, component, value, and the four frac_* columns.
    """
    return _dispatch(_cvp_module().plot_variance_partitioning_heatmap, args, kwargs,
                     _fields_variance, lambda P: P["figtitle"])


# ---- discoverability -----------------------------------------------------------------------------------------
_HOVER_FIELD_DOCS = {
    "plot_all_strain_feature_heatmap": ["strain", "feature", "condition", "value", "raw_mean", "fov_sd",
                                        "fov_cv", "n_fovs_used", "rank"],
    "plot_feature_icc_heatmap": ["feature", "column", "value", "icc_single", "icc_mean", "split_half", "fstat",
                                 "pvalue", "mean_icc", "min_icc", "max_icc", "best_condition",
                                 "worst_condition", "cross_condition_rho", "n_conditions_passing",
                                 "passes_threshold", "icc_by_condition", "rank"],
    "plot_sorted_r2_per_feature": ["feature", "condition", "value", "mean_r2", "min_r2", "max_r2",
                                   "rank_by_mean", "rank_in_condition", "r2_by_condition",
                                   "condition_r2_overall", "condition_rmse", "condition_p_value"],
    "plot_within_between_ratio_heatmap": ["feature", "condition", "value", "fold_ratio", "mean_log2_ratio",
                                          "min_log2_ratio", "max_log2_ratio", "rank_by_mean",
                                          "rank_in_condition", "ratio_by_condition"],
    "plot_variance_partitioning_heatmap": ["feature", "component", "value", "dominant_component", "rank",
                                           "frac_strain", "frac_environment", "frac_interaction",
                                           "frac_residual", "ss_strain", "ss_environment", "ss_interaction",
                                           "ss_residual", "<any other variance_df column>"],
}


def list_hover_fields(function_name):
    """Built-in hover fields for one of the five interactive heatmap wrappers."""
    name = function_name.split(".")[-1]
    if name not in _HOVER_FIELD_DOCS:
        raise ValueError(f"Choose one of {list(_HOVER_FIELD_DOCS)}")
    return list(_HOVER_FIELD_DOCS[name])

# ----------------------------------------------------------------------------
# Grouped bar plot: ridge R^2 vs recalibrated strain-mean R^2 per condition
# ----------------------------------------------------------------------------
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