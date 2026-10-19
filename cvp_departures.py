"""
Does a strain respond differently from what the rest of the collection predicts?

For one feature, this module
  1. builds each strain's EXPECTED reaction norm with leave-one-condition-out
     ridge regression, cross-validated by well so a strain never informs its
     own prediction;
  2. measures each strain's departure from that expectation (residual =
     observed - predicted) in every condition;
  3. tests each departure against a permutation null; and
  4. plots observed (solid) vs expected (dashed) reaction norms.

What a significant departure means
----------------------------------
Two nulls are computed per condition:

  p_perm  shuffles field-of-view residuals across strains. It asks whether a
          departure is bigger than field-to-field noise. On its own it is
          anti-conservative, because ridge never recovers a strain's level
          exactly and that shortfall repeats across all of the strain's
          fields.
  p_emp   compares the strain's departure with the departures of every other
          strain in that condition (robust z from median and MAD). It asks
          whether this strain breaks from the collection's expectation more
          than strains typically do. q is computed from p_emp.

So "significant" (q < 0.05) means: this strain departs from the collection's
expectation by more than strains typically do, given the model's usual error.

It does NOT prove the departure is biology the model could never learn. A
model shortfall that is specific to one strain also reproduces across that
strain's fields. Report significant departures as "responds differently from
what the collection predicts", not as "has a unique gene-environment
interaction".

Usage
-----
    import cvp_departures as cdp

    dep_df = cdp.compute_strain_departures(
        training_data_finite, exp_conditions, training_feature_labels,
        feature="axis_major_length (bf)",
        ridge_alpha=1000, n_wells=96, n_fovs=n_fovs_2_analyze, dead_ids=dead_ids,
    )
    fig, ax = cdp.plot_reaction_norm_vs_prediction(
        dep_df, strains=[27, 88],
        conditions=["16C e.stat", "22C e.stat", "30C e.stat"],
        condition_labels=["16 °C", "22 °C", "30 °C"],
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import arcadia_pycolor as apc
    _DEFAULT_COLORS = [apc.aster, apc.dress, apc.lapis, apc.candy]
except Exception:  # plotting still works without the Arcadia palette
    _DEFAULT_COLORS = ["#5A4FA0", "#E2A6A0", "#2B65A1", "#C8508A"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# main computation
# ----------------------------------------------------------------------------
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


def summarize_departures(departures_df, q_threshold=0.05):
    """
    One row per strain: how many conditions it departs in, and its largest
    departure. Sort by `n_significant` then `max_abs_resid_z` to find the
    strains that most consistently break from the collection's expectation.
    """
    d = departures_df.copy()
    d["significant"] = d["q"] < q_threshold
    d["abs_resid_z"] = d["resid_z"].abs()
    g = d.groupby("strain")
    out = pd.DataFrame({
        "n_significant": g["significant"].sum(),
        "max_abs_resid_z": g["abs_resid_z"].max(),
        "condition_of_max": g.apply(
            lambda x: x.loc[x["abs_resid_z"].idxmax(), "condition"]),
        "min_q": g["q"].min(),
    }).reset_index()
    return out.sort_values(["n_significant", "max_abs_resid_z"],
                           ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------
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
