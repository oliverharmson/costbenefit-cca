# Distributed under the MIT license; see LICENSE.
# CCA dimensionality (project 382702): which pairs of areas share significant canonical correlations (CCA) in each task
# epoch, mirroring the data processing steps in Stoll et al. 2024 https://doi.org/10.1016/j.cub.2024.08.034.
# The CCA itself (cross-validated canonical correlations with a permutation null, per session x epoch x area pair) was
# run on the 25 ms firing-rate time series of project 480152 and is stored in data/382702_canonical_corr.feather; this
# script tests which edges/dimensions are significant across animals and draws FIGURE 2 of the manuscript:
#   Fig. 2a  network of area pairs with significant dimensions, per epoch
#   Fig. 2b  bootstrapped Delta-rho per circuit class, per epoch
#   Fig. 2c  significant CCA components per session by epoch and circuit class
# (saved as output/382702_all_area_pair_cca_corr_ndim), together with the repeated-measures ANOVA on the number of
# significant components.

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, gaussian_kde
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd(), '..', 'lib'))
import helpers as hs
import framework as fw
import matplotlib as mpl
import pingouin as pg
from block import BlockObj as BlO
from block import var_labels, approximate_var_label
from catalogue import get_file_root, get_output_root, PNG_DIR
from matplotlib.lines import Line2D

PROJECT_ID = 382702
OPEN_FIGS = False  # open the saved PNGs in the OS viewer
conf_dict = {
    'thresh': .01,  # FDR-corrected alpha for edges (Benjamini-Hochberg) and the per-model permutation p-value
    'circuit_map': {'acc_stn': 'Fr-BG',
                    'stn_str': 'BG-BG',
                    'acc_str': 'Fr-BG',
                    'mo_str': 'Fr-BG',
                    'mo_stn': 'Fr-BG',
                    'acc_mo': 'Fr-Fr',
                    'acc_vp': 'Fr-BG',
                    'str_vp': 'BG-BG'},
    'circuit_colour': {'Fr-Fr': '#1864AB',
                       'BG-BG': '#C2255C',
                       'Fr-BG': '#862E9C'}
}
mpl.rcParams['svg.fonttype'] = 'none'
f_root = get_file_root(PROJECT_ID)      # input data: data/382702_*
out_root = get_output_root(PROJECT_ID)  # figures: output/382702_*


def load_SpikeStats(src=PROJECT_ID):
    st = BlO.SpikeStats(get_file_root(src))
    st.version = fw.spike_stats_ids['canonical_corr']
    st.load()
    return st


st = load_SpikeStats()
st.canonical_corr['subject'] = [_.split('_')[0] for _ in st.canonical_corr.subject_day]
st.canonical_corr['∆rho'] = (np.abs(st.canonical_corr['corr_cv']) -
                             np.abs(st.canonical_corr['corr_cv_perm_avg']))
st.canonical_corr = st.canonical_corr[st.canonical_corr["rank"] != -1]
conf_dict['epochs'] = st.get_sorted_epochs(st.canonical_corr.epoch.unique())
conf_dict['bin_max'] = {e: 1 for i, e in enumerate(conf_dict['epochs'])}
conf_dict['edges'] = {e: [i, i] for i, e in enumerate(conf_dict['epochs'])}
conf_dict['col_n'] = np.array(list(conf_dict['bin_max'].values())).sum()


def pre_process_cca_for_plotting_wilcoxon(cca_df, stat_var='corr_cv', perm_var='corr_cv_perm_avg', edge_var='∆rho',
                                          fdr=True):
    rows = []
    for epoch in conf_dict['epochs']:
        for area_pair in cca_df.area_pair.unique(): # the default variable label is used here
            tbl = cca_df.query(f'epoch=="{epoch}" & area_pair=="{area_pair}"')
            if tbl.empty:
                area_pair = '_'.join(area_pair.split("_")[::-1])
                tbl = cca_df.query(f'epoch=="{epoch}" & area_pair=="{area_pair}"')
            edge_label=tuple([str(fw.region_labels()[area_pair.split("_")[0]]),
                              str(fw.region_labels()[area_pair.split("_")[1]])])
            for rank in np.unique(np.sort(tbl['rank'])):
                real_corrs = tbl.loc[tbl['rank'] == rank][stat_var].values
                perm_corrs = tbl.loc[tbl['rank'] == rank][perm_var].values
                plot_corrs = tbl.loc[tbl['rank'] == rank][edge_var if stat_var != edge_var else stat_var].values
                try:
                    _, p = wilcoxon(real_corrs, perm_corrs, alternative='greater')
                except ValueError:
                    p = np.nan
                row = pd.Series({'epoch': epoch,
                                 'edge_label': edge_label,
                                 'area_pair': area_pair,
                                 'rank': rank,
                                 'edge_value': hs.average_rhos(plot_corrs),
                                 'edge_pval': p})
                rows.append(row)

    edge_params = pd.DataFrame(rows)
    edge_params.dropna(subset=["edge_pval"]).reset_index(drop=True)


    if fdr:
        ps = edge_params['edge_pval'].values
        ps[np.isnan(ps)] = 1 # replace any non-existent pairs with 1s to be able to preserve the analysis
        p_mask = hs.benjamini_hochberg(ps, alpha=conf_dict['thresh'])
        for i, mask in enumerate(p_mask):
            edge_params.loc[i, 'edge_pval'] = np.invert(mask).astype(int)
    return edge_params

corr_cv_df = pre_process_cca_for_plotting_wilcoxon(st.canonical_corr)

#%%
def plot_canonical_corr_network_kdim(edge_df, inp_cvs=None, ylabel=None, plot_ns_edges=False, edge_var='∆rho',
                                     node_fn=st.plot_node_graph_3_1, colour_map=None, circuit_map=None):
    where = np.divide(hs.config.landscape_A(4), [1, 1]) if inp_cvs is None else inp_cvs
    cv = hs.gridspec(2, 1, wspace=.1, where=where, height_ratios=[0.2, 0.8])
    bottom_gs = iter(hs.gridspec(1, len(conf_dict["bin_max"]), wspace=.05, where=cv[1]))

    nodes_with_finite_edges = set()
    df_finite = edge_df[np.isfinite(pd.to_numeric(edge_df["edge_value"], errors="coerce"))]

    for (u, v) in df_finite["edge_label"]:
        nodes_with_finite_edges.update([u, v])

    if circuit_map is not None:
        legend_labels = list(dict.fromkeys(circuit_map.values()))
    else:
        legend_labels = []

    if colour_map is None:
        colour_map = {}

    legend_handles = [
        Line2D([0], [0], color=colour_map.get(lbl, "black"), lw=3, marker="o",
            markersize=6, markerfacecolor=colour_map.get(lbl, "black"), markeredgecolor="black",
            markeredgewidth=0.8,) for lbl in legend_labels]

    with mpl.rc_context(hs.no_axes):
        leg_ax = hs.get_ax(cv[0])

        if legend_handles:
            leg_ax.legend(legend_handles, legend_labels, loc="center", ncol=len(legend_labels), frameon=False,
                handlelength=2.0, columnspacing=1.8)

    for epoch in conf_dict['epochs']:
        node_labels = {n: n for n in fw.node_plot_coords()}
        png_path = os.path.join(PNG_DIR, epoch + '.png')

        node_fn(
            coords=fw.node_plot_coords(),
            edge_params=edge_df.loc[edge_df.epoch == epoch].drop('epoch', axis=1),
            thresh=conf_dict['thresh'],
            node_labels=node_labels,
            rc_context=hs.no_axes,
            node_size=180,
            label_fontsize=4,
            where=next(bottom_gs),
            annotate_edges=False,
            static_lw=1.2,
            background_png=os.path.join(PNG_DIR, 'brain_slice.png') if epoch == 'PRE' else None,
            ylabel=ylabel if epoch == 'PRE' else None,
            epoch_png_path=png_path,
            title=var_labels[epoch],
            plot_ns_edges=plot_ns_edges,
            nodes_with_finite_edges=nodes_with_finite_edges,
            circuit_map=circuit_map,
            colour_map=colour_map,
        )

#%%

def plot_corr_cv_group_counts(
    corr_cv_df,
    circuit_map,
    circuit_colour=None,
    colors=None,
    epoch_col="epoch",
    pair_col="area_pair",
    sig_col="edge_pval",
    thresh=0.01,
    binary_zero_is_sig=True,
    where=(3, 3),
    epoch_order=None,
    group_order=None,
    marker="o",
    markersize=4,
    alpha=0.9,
    show_points=True,
):
    hs.config.plot_kwargs["linewidth"] = 0.4
    df = corr_cv_df.copy()

    def _canon(x):
        if isinstance(x, (tuple, list)) and len(x) == 2:
            a, b = str(x[0]), str(x[1])
            return "_".join(sorted([a, b]))
        if isinstance(x, str):
            if "_" in x:
                parts = x.split("_")
            elif "-" in x:
                parts = x.split("-")
            else:
                return x
            return "_".join(sorted([str(parts[0]), str(parts[1])])) if len(parts) >= 2 else x
        return str(x)

    if pair_col not in df.columns:
        raise ValueError(f"corr_cv_df is missing '{pair_col}'")
    if epoch_col not in df.columns or sig_col not in df.columns:
        raise ValueError(f"corr_cv_df must contain '{epoch_col}' and '{sig_col}'")

    area_to_group = {_canon(k): v for k, v in circuit_map.items()}
    df["_canon_pair"] = df[pair_col].apply(_canon)
    df = df[df["_canon_pair"].isin(area_to_group)].copy()

    if df.empty:
        raise ValueError(
            f"No rows matched circuit_map. Unique '{pair_col}' values were: "
            f"{df[pair_col].dropna().unique()[:10]}"
        )

    df["circuit_group"] = df["_canon_pair"].map(area_to_group)

    s = df[sig_col]
    if s.dtype == bool:
        df["_sig"] = s
    elif set(pd.unique(s.dropna())).issubset({0, 1}):
        df["_sig"] = s.eq(0) if binary_zero_is_sig else s.astype(bool)
    else:
        df["_sig"] = pd.to_numeric(s, errors="coerce").lt(thresh)

    counts = (
        df.groupby([epoch_col, "circuit_group"], dropna=False)["_sig"]
        .sum()
        .reset_index(name="sig_n")
    )

    if epoch_order is None:
        epoch_order = list(pd.unique(df[epoch_col]))
    if group_order is None:
        group_order = list(dict.fromkeys(circuit_map.values()))

    full_index = pd.MultiIndex.from_product(
        [epoch_order, group_order], names=[epoch_col, "circuit_group"]
    )
    counts = counts.set_index([epoch_col, "circuit_group"]).reindex(full_index, fill_value=0).reset_index()

    colour_map = circuit_colour if circuit_colour is not None else colors
    if colour_map is None:
        palette = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["black"])
        colour_map = {g: palette[i % len(palette)] for i, g in enumerate(group_order)}

    with mpl.rc_context(hs.rc0):
        ax = hs.get_ax(where)
        x = np.arange(len(epoch_order))
        handles, labels = [], []
        for group in group_order:
            grp = counts[counts["circuit_group"] == group].set_index(epoch_col).reindex(epoch_order)
            y = grp["sig_n"].to_numpy(dtype=float)
            col = colour_map.get(group, "black")

            ax.plot(x, y, marker=marker, ms=markersize, color=col, label='_nolegend_', lw=2)
            if show_points:
                ax.scatter(
                    x, y, color=col, alpha=alpha, zorder=3, edgecolor="black", label='_nolegend_',
                    **{k: hs.config.plot_kwargs[k] for k in hs.config.scat_defs}
                )

            col = colour_map.get(group, "black")
            handles.append(Line2D([0], [0], color=col, lw=1.0, marker="o", markersize=markersize,
                    markerfacecolor=col, markeredgecolor="black", markeredgewidth=0.8))
            labels.append(group)
        ax.set_xticks(x)
        ax.set_xticklabels([approximate_var_label(e) for e in epoch_order], rotation=30)
        ax.set_yticks(np.arange(0, int(counts["sig_n"].max()) + 1 if not counts.empty else 1))
        ax.set_xlabel("Epoch", labelpad=2)
        ax.set_ylabel("Count of sig. CCA components", labelpad=2)
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", pad=2)
        ax.margins(x=0.1, y=0.1)
        ax.legend(handles, labels, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        # Clip axis bars to terminal ticks
        xt = ax.get_xticks()
        yt = ax.get_yticks()
        if len(xt) >= 2:
            ax.spines["bottom"].set_bounds(xt[0], xt[-1])
        if len(yt) >= 2:
            ax.spines["left"].set_bounds(yt[0], yt[-1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return counts

#%%

def within_subject_sem(df, subject_col, condition_cols, dv_col):

    work = df[[subject_col, *condition_cols, dv_col]].copy()
    work = work.dropna(subset=[subject_col, *condition_cols, dv_col])

    # -----------------------------------------------------
    # Cousineau normalization
    # -----------------------------------------------------
    grand_mean = work[dv_col].mean()
    subj_mean = work.groupby(subject_col)[dv_col].transform("mean")

    work["_ws_norm"] = (
        work[dv_col]
        - subj_mean
        + grand_mean
    )

    # -----------------------------------------------------
    # Number of within-subject conditions
    # -----------------------------------------------------
    k = (
        work[condition_cols]
        .drop_duplicates()
        .shape[0]
    )

    # Morey correction factor
    morey_corr = np.sqrt(k / (k - 1)) if k > 1 else 1.0

    def _sem(x):

        x = pd.Series(x).dropna().to_numpy()

        if x.size <= 1:
            return 0.0

        return (
            np.std(x, ddof=1)
            / np.sqrt(x.size)
        )

    # -----------------------------------------------------
    # Raw means
    # -----------------------------------------------------
    raw_mean = (
        work.groupby(condition_cols, dropna=False)[dv_col]
        .mean()
        .reset_index(name="mean")
    )

    # -----------------------------------------------------
    # Within-subject SEM
    # -----------------------------------------------------
    ws_sem = (
        work.groupby(condition_cols, dropna=False)["_ws_norm"]
        .agg(sem=_sem)
        .reset_index()
    )

    # Apply Morey correction
    ws_sem["sem"] *= morey_corr

    out = raw_mean.merge(
        ws_sem,
        on=condition_cols,
        how="left"
    )

    return out

def plot_sig_model_counts_by_epoch(
    cca_df,
    circuit_map,
    thresh=0.01,
    pval_col='pval',
    epoch_col='epoch',
    area_col='area_pair',
    day_col='subject_day',
    where=(3, 3),
    collapse_subjects=False,
    # FDR
    fdr=False,
    fdr_within_epoch=True,

    # plotting
    epoch_order=None,
    group_order=None,
    circuit_colour=None,   # NEW: dict mapping circuit_group -> colour
    colors=None,           # kept for backwards compatibility
    marker='o',
    markersize=4,
    alpha=0.9,
    show_points=True,
    # chance line
    show_fdr_line=False,
    fdr_line_kwargs=None,
):
    """
    Count significant CCA models (rows with pval < thresh) for each:

        subject_day x epoch x circuit_group

    then average across subject-days within each epoch.

    This gives a measure of session-wise dimensionality.

    Area-pair mapping is order-invariant:
        acc_stn == stn_acc
        mo_str == str_mo
    """
    hs.config.plot_kwargs['linewidth'] = 0.4
    df = cca_df.copy()

    required = {
        epoch_col,
        area_col,
        pval_col,
        day_col,
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"cca_df is missing required columns: {missing}"
        )

    # ---------------------------------------------------------
    # Canonicalize area-pair labels
    # ---------------------------------------------------------
    def _canonical_pair(x):

        if isinstance(x, (tuple, list)) and len(x) == 2:
            parts = [str(x[0]), str(x[1])]

        elif isinstance(x, str):

            if "_" in x:
                parts = x.split("_")

            elif "-" in x:
                parts = x.split("-")

            else:
                return str(x)

        else:
            return str(x)

        if len(parts) != 2:
            return str(x)

        return "_".join(sorted(parts))

    df["_canonical_area_pair"] = df[area_col].apply(_canonical_pair)

    # canonicalize circuit map
    area_to_group = {
        _canonical_pair(area_pair): group_name
        for area_pair, group_name in circuit_map.items()
    }

    # keep only mapped area pairs
    df = df[df["_canonical_area_pair"].isin(area_to_group)].copy()

    if df.empty:
        raise ValueError(
            "No rows matched entries in circuit_map."
        )

    df["circuit_group"] = df["_canonical_area_pair"].map(area_to_group)

    # ---------------------------------------------------------
    # Significance / FDR
    # ---------------------------------------------------------
    pvals = pd.to_numeric(df[pval_col], errors="coerce")
    if fdr:
        df["_sig"] = False
        if fdr_within_epoch:
            for ep, grp_idx in df.groupby(epoch_col).groups.items():
                grp_p = pvals.loc[grp_idx].to_numpy()
                pmat = grp_p.reshape(-1, 1).flatten()
                sigmat = hs.benjamini_hochberg(pmat, alpha=thresh)
                df.loc[grp_idx, "_sig"] = sigmat.astype(bool)
        else:
            sigmat = hs.benjamini_hochberg(pvals.values.flatten(), alpha=thresh)
            df["_sig"] = sigmat.astype(bool)
    else:
        df["_sig"] = pvals.lt(thresh)

    # ---------------------------------------------------------
    # Count significant rows PER SESSION
    # ---------------------------------------------------------
    per_session = (
        df.groupby(
            [
                day_col,
                epoch_col,
                "circuit_group",
            ],
            dropna=False
        )["_sig"]
        .sum()
        .reset_index(name="sig_n")
    )
    if not collapse_subjects:
        anova_df = (
            per_session
            .rename(columns={epoch_col: "epoch"})
            [[day_col, "circuit_group", "epoch", "sig_n"]]
            .copy()
        )
    else:
        # ---------------------------------------------------------
        # Collapse days down to independent subjects
        # ---------------------------------------------------------
        # Use the existing subject column if present; otherwise derive it
        # from subject_day, e.g. gh001_210601 -> gh001
        if "subject" in df.columns:
            subject_lookup = (
                df[[day_col, "subject"]]
                .drop_duplicates()
                .set_index(day_col)["subject"]
            )
            per_session["subject"] = per_session[day_col].map(subject_lookup)
        else:
            per_session["subject"] = (
                per_session[day_col]
                .astype(str)
                .str.split("_")
                .str[0]
            )

        # One value per subject x epoch x circuit_group
        anova_df = (
            per_session
            .groupby(["subject", epoch_col, "circuit_group"], dropna=False)["sig_n"]
            .mean()
            .reset_index()
            .rename(columns={epoch_col: "epoch"})
        )

    # ---------------------------------------------------------
    # Within-subject-day SEM on the plotted conditions
    # ---------------------------------------------------------
    summary = within_subject_sem(
        per_session,
        subject_col=day_col,
        condition_cols=[epoch_col, "circuit_group"],
        dv_col="sig_n",
    )
    # maximum count observed in any session, used for y ticks
    max_sig_n = int(np.nanmax(summary['mean'])+1) if not per_session.empty else 0
    y_ticks = np.arange(0, max_sig_n + 1, 1)
    # ---------------------------------------------------------
    # Ordering
    # ---------------------------------------------------------
    if epoch_order is None:
        epoch_order = list(pd.unique(df[epoch_col]))

    if group_order is None:
        group_order = list(dict.fromkeys(circuit_map.values()))

    # ---------------------------------------------------------
    # Colours
    # ---------------------------------------------------------
    group_colour_map = circuit_colour if circuit_colour is not None else colors

    if group_colour_map is None:
        palette = list(
            mpl.rcParams["axes.prop_cycle"]
            .by_key()
            .get("color", ["black"])
        )
        group_colour_map = {
            g: palette[i % len(palette)]
            for i, g in enumerate(group_order)
        }

    def _group_colour(g):
        return group_colour_map.get(g, "black")


    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    with mpl.rc_context(hs.rc0):

        ax = hs.get_ax(where)

        x = np.arange(len(epoch_order))

        for group_name in group_order:

            grp = (
                summary[
                    summary["circuit_group"] == group_name
                ]
                .copy()
            )

            grp = (
                grp
                .set_index(epoch_col)
                .reindex(epoch_order)
            )

            y = grp["mean"].to_numpy(dtype=float)
            yerr = grp["sem"].to_numpy(dtype=float)

            col = _group_colour(group_name)

            err = ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt=f"{marker}-",
                    ms=markersize,
                    color=col,
                    label=group_name,
                    **{i: hs.config.plot_kwargs[i] for i in hs.config.err_bar_defs}
                )
            for cap in err[1]:
                cap.set_solid_capstyle("round")
            for barcol in err[2]:
                barcol.set_capstyle("round")
            if show_points and np.isfinite(y).any():

                ax.scatter(
                    x,
                    y,
                    color=col,
                    alpha=alpha,
                    zorder=3,
                    edgecolor="black",
                    **{i: hs.config.plot_kwargs[i] for i in hs.config.scat_defs}
                )

        # -----------------------------------------------------
        # Expected false positives
        # -----------------------------------------------------
        if show_fdr_line:

            n_tests = (
                df.groupby(
                    [
                        day_col,
                        epoch_col,
                        "circuit_group",
                    ]
                )
                .size()
                .mean()
            )

            expected_fp = n_tests * thresh

            if fdr_line_kwargs is None:
                fdr_line_kwargs = dict(
                    color="0.4",
                    lw=1,
                    ls="--",
                    alpha=0.8,
                )

            ax.axhline(expected_fp, **fdr_line_kwargs)

            ax.text(
                x[-1] + 0.1,
                expected_fp,
                "expected FP",
                va="center",
                ha="left",
                fontsize=8,
                color=fdr_line_kwargs.get("color", "0.4"),
            )

        # -----------------------------------------------------
        # Labels
        # -----------------------------------------------------
        ax.set_xticks(x)
        ax.set_xticklabels([approximate_var_label(e) for e in epoch_order], rotation=30)
        ax.set_yticks(y_ticks)
        ax.set_xlabel("Epoch")

        ylabel = (
            "Significant models"
        )
        ax.set_ylabel(f"{ylabel} per session")

        ax.legend(
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        # -----------------------------------------------------
        # Clip axis bars to terminal ticks
        # -----------------------------------------------------
        xt = ax.get_xticks()
        yt = ax.get_yticks()

        if len(xt) >= 2:
            ax.spines["bottom"].set_bounds(xt[0], xt[-1])
        if len(yt) >= 2:
            ax.spines["left"].set_bounds(yt[0], yt[-1])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # hs.displace_axes(ax, .02)

    return summary, per_session, anova_df

summary, per_pair, anova_df = plot_sig_model_counts_by_epoch(
    st.canonical_corr,
    circuit_map=conf_dict['circuit_map'],
    circuit_colour=conf_dict['circuit_colour'],
    thresh=conf_dict['thresh'],
    pval_col="pval",
    fdr=True,
    fdr_within_epoch=True,
    epoch_col="epoch",
    area_col="area_pair",
)

def run_twoway_rm_anova_with_posthoc(
    df,
    subject_col="subject_day",
    group_col="circuit_group",
    epoch_col="epoch",
    dv_col="sig_n",
    alpha=conf_dict['thresh'],
    verbose=True,
):
    """
    Two-way repeated-measures ANOVA with post-hoc testing.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with columns:
        subject_col, group_col, epoch_col, dv_col
    subject_col : str
        Subject / repeated-measures identifier, e.g. subject_day.
    group_col : str
        Within-subject factor 1, e.g. circuit_group.
    epoch_col : str
        Within-subject factor 2, e.g. dimensionality / rank bin.
    dv_col : str
        Dependent variable, e.g. significant-model count.
    alpha : float
        False discovery rate for the post-hoc tests' Benjamini-Hochberg correction
        (hs.benjamini_hochberg, applied within each post-hoc family separately, i.e.
        circuit_group pairs, epoch pairs, and interaction-cell pairs are each their own
        family -- consistent with how FDR correction is applied everywhere else in this
        script, e.g. pre_process_cca_for_plotting_wilcoxon, rather than pingouin's own
        padjust methods).
    verbose : bool
        If True, print the ANOVA and post-hoc tables.

    Returns
    -------
    anova_table : pd.DataFrame
    posthoc_group : pd.DataFrame
        Pairwise post-hoc comparisons for circuit group, collapsed across dimensionality.
    posthoc_dim : pd.DataFrame
        Pairwise post-hoc comparisons for dimensionality, collapsed across circuit group.
    posthoc_interaction : pd.DataFrame
        Pairwise comparisons on the interaction cells.
    Each post-hoc table carries the usual pingouin columns (uncorrected p in 'p-unc') plus the
    'sig_bh' column (always True in what's returned -- see below) marking comparisons that
    survive Benjamini-Hochberg correction at `alpha` within that table.

    A comparison can come back with 'p-unc' = NaN when both conditions being compared are
    identical for every subject (e.g. sig_n=0 everywhere), which leaves no variance to compute
    a paired t-statistic from; such rows are excluded from the FDR family entirely (not
    counted as non-significant) since they were never actually a valid test. Since every
    returned row is filtered to sig_bh == True, rows that were non-significant (sig_bh=False)
    or untestable (sig_bh=pd.NA) are dropped -- these tables list only the comparisons that
    survived correction.
    """

    required = {subject_col, group_col, epoch_col, dv_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    dat = df[[subject_col, group_col, epoch_col, dv_col]].copy()
    dat = dat.dropna(subset=[subject_col, group_col, epoch_col, dv_col])

    dat[subject_col] = dat[subject_col].astype(str)
    dat[group_col] = dat[group_col].astype(str)
    dat[epoch_col] = dat[epoch_col].astype(str)

    # Optional: enforce balance by dropping subjects lacking any cell
    cell_counts = dat.groupby(subject_col).size()
    expected_cells = dat[group_col].nunique() * dat[epoch_col].nunique()
    keep_subjects = cell_counts[cell_counts == expected_cells].index
    dat = dat[dat[subject_col].isin(keep_subjects)].copy()

    if dat.empty:
        raise ValueError("No balanced subject_day data remain after filtering.")

    # Two-way repeated-measures ANOVA
    anova_table = pg.rm_anova(
        data=dat,
        dv=dv_col,
        within=[group_col, epoch_col],
        subject=subject_col,
        detailed=True,
        effsize="ng2",
    )

    def _bh_posthoc(**kwargs):
        # pingouin gives the uncorrected p-values (padjust='none'); hs.benjamini_hochberg
        # then flags which of those survive FDR correction at `alpha`, treating this call's
        # comparisons as one family, matching how FDR is applied elsewhere in this script.
        res = pg.pairwise_tests(padjust='none', effsize='hedges', **kwargs)
        p = res['p-unc'].to_numpy(dtype=float)
        # A pair can come back p-unc=NaN when both conditions are identical for every
        # subject (zero-variance paired difference -> pingouin can't form a t-statistic).
        # That's not a real test, so it's excluded from the FDR family entirely rather than
        # counted as non-significant -- hs.benjamini_hochberg can't accept NaN as input.
        finite = np.isfinite(p)
        sig_bh = pd.array([pd.NA] * len(p), dtype="boolean")
        if finite.any():
            sig_bh[finite] = hs.benjamini_hochberg(p[finite], alpha=alpha)
        res['sig_bh'] = sig_bh
        # keep only comparisons that survived correction; drop sig_bh==False and pd.NA rows
        return res[res['sig_bh'].fillna(False)].reset_index(drop=True)

    # Post-hoc: main effect of circuit group
    posthoc_group = _bh_posthoc(data=dat, dv=dv_col, within=group_col, subject=subject_col)

    # Post-hoc: main effect of dimensionality
    posthoc_dim = _bh_posthoc(data=dat, dv=dv_col, within=epoch_col, subject=subject_col)

    # Post-hoc on interaction cells:
    # create a single cell label for each group x dimensionality combination
    dat = dat.copy()
    dat["cell"] = dat[group_col].astype(str) + " | " + dat[epoch_col].astype(str)

    posthoc_interaction = _bh_posthoc(data=dat, dv=dv_col, within="cell", subject=subject_col)

    if verbose:
        print("\nTwo-way repeated-measures ANOVA")
        print(anova_table)

        print("\nPost-hoc: circuit group")
        print(posthoc_group)

        print("\nPost-hoc: epoch")
        print(posthoc_dim)

        print("\nPost-hoc: interaction cells")
        print(posthoc_interaction)

    return anova_table, posthoc_group, posthoc_dim, posthoc_interaction

anova_table, posthoc_group, posthoc_dim, posthoc_interaction = run_twoway_rm_anova_with_posthoc(
    df=anova_df,
    subject_col="subject_day",
    group_col="circuit_group",
    epoch_col="epoch",
    dv_col="sig_n",
)

posthoc_interaction = posthoc_interaction.loc[:, ['A', 'B', 'hedges', 'p-unc']]
#%%

def plot_edge_variance_by_session(edge_df, cca_df, edge_var='∆rho', thresh=conf_dict['thresh'],
                                  epochs=None, circuit_map=None, colour_map=None, group_order=None,
                                  where=None, n_boot=5000, ci=95, violin_width=0.35, seed=0, ylabel=None,
                                  group_by_circuit=True):
    """
    Companion to plot_canonical_corr_network_kdim: for every edge drawn there as significant
    (edge_pval < thresh), shows the bootstrapped sampling distribution of the mean edge_var
    across subject-days -- 5,000 resamples, kernel-density-shaded curve, observed mean (dot) and
    percentile 95% CI (vertical bar), dashed line at zero -- following the estimation-plot convention in
    McHugh et al. 2024 Neuron Fig. 3B/C/F/G/K/L (https://pmc.ncbi.nlm.nih.gov/articles/PMC7616703/):
    "computed from 5,000 bootstrapped resamples ... black dot, mean; black ticks, 95% confidence
    interval; filled curve, sampling-error distribution".

    group_by_circuit : bool
        If True, pool the per-session-day edge_var values across all significant edges sharing a
        circuit_group (per circuit_map) within an epoch, and bootstrap that pooled distribution
        instead of one distribution per edge. Yields at most one column per circuit sub-class
        (i.e. up to len(group_order), typically 3) per epoch, rather than one per edge.
    """
    epochs_ = conf_dict['epochs'] if epochs is None else epochs
    where = np.divide(hs.config.landscape_A(4), [1, 3]) if where is None else where
    circuit_map = {} if circuit_map is None else circuit_map
    colour_map = {} if colour_map is None else colour_map
    group_order = list(dict.fromkeys(circuit_map.values())) if group_order is None else group_order

    def _canon(pair):
        parts = pair.split('_')
        return '_'.join(sorted(parts)) if len(parts) == 2 else pair

    area_to_group = {_canon(k): v for k, v in circuit_map.items()}

    sig = edge_df[(pd.to_numeric(edge_df['edge_pval'], errors='coerce') < thresh) &
                 np.isfinite(pd.to_numeric(edge_df['edge_value'], errors='coerce'))].copy()
    sig['circuit_group'] = sig['area_pair'].apply(lambda p: area_to_group.get(_canon(p)))
    sig['_group_order'] = sig['circuit_group'].apply(
        lambda g: group_order.index(g) if g in group_order else len(group_order))
    sig = sig.sort_values(['_group_order', 'area_pair', 'rank'])

    def _session_values(epoch, area_pair, rank):
        tbl = cca_df.query('epoch == @epoch & area_pair == @area_pair')
        if tbl.empty:
            area_pair = '_'.join(area_pair.split('_')[::-1])
            tbl = cca_df.query('epoch == @epoch & area_pair == @area_pair')
        return tbl.loc[tbl['rank'] == rank, edge_var].dropna().to_numpy()

    def _pooled_circuit_values(epoch, group):
        group_rows = sig[(sig['epoch'] == epoch) & (sig['circuit_group'] == group)]
        parts = [_session_values(epoch, r['area_pair'], r['rank']) for _, r in group_rows.iterrows()]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.array([])

    rng = np.random.default_rng(seed)
    lo_pct, hi_pct = (100 - ci) / 2, 100 - (100 - ci) / 2

    def _bootstrap_mean_dist(vals):
        # resample sessions with replacement, n_boot times, and take the mean of each resample
        idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
        return vals[idx].mean(axis=1)

    def _draw_column(ax, x, vals, col):
        mean = vals.mean()
        boot = _bootstrap_mean_dist(vals)
        lo, hi = np.percentile(boot, [lo_pct, hi_pct])
        y_all.extend([float(boot.min()), float(boot.max())])

        if np.ptp(boot) > 0:
            y_grid = np.linspace(boot.min(), boot.max(), 200)
            density = gaussian_kde(boot)(y_grid)
            width = density / density.max() * violin_width
            ax.fill_betweenx(y_grid, x, x + width, color=col, alpha=0.5, lw=0, zorder=2)

        ax.plot([x, x], [lo, hi], color='black', lw=1.0, zorder=3, solid_capstyle='butt')
        ax.scatter([x], [mean], color='black', s=13, zorder=4)

    gs = iter(hs.gridspec(1, len(epochs_), wspace=.05, where=where))
    axes, y_all = [], []

    for i, epoch in enumerate(epochs_):
        rc_c = hs.rc0 if not i else hs.no_yaxis
        with mpl.rc_context(rc_c):
            ax = hs.get_ax(next(gs))
            xtick_labels = []

            if group_by_circuit:
                for group in group_order:
                    vals = _pooled_circuit_values(epoch, group)
                    if not vals.size:
                        continue
                    _draw_column(ax, len(xtick_labels), vals, colour_map.get(group, 'black'))
                    xtick_labels.append(group)
            else:
                for _, row in sig[sig['epoch'] == epoch].iterrows():
                    vals = _session_values(epoch, row['area_pair'], row['rank'])
                    if not vals.size:
                        continue
                    _draw_column(ax, len(xtick_labels), vals, colour_map.get(row['circuit_group'], 'black'))
                    u, v = row['edge_label']
                    xtick_labels.append(f'{u}-{v} (k{int(row["rank"])})')

            ax.axhline(0, color='0.6', lw=0.6, ls='--', zorder=1)
            ax.set_yticks(np.arange(0.1, 0.6, .1))
            ax.set_xlim(-0.5, max(len(xtick_labels), 1) - 0.5 + violin_width)
            ax.set_xticks(np.arange(len(xtick_labels)))
            ax.set_xticklabels(xtick_labels, rotation=90)
            if i == 0:
                if ylabel is None:
                    ylab = f'{edge_var}\n(mean, {n_boot}-resample bootstrap, {ci}% CI)'
                    if group_by_circuit:
                        ylab += '\npooled within circuit class'
                else:
                    ylab = ylabel
                ax.set_ylabel(ylab)
            xt = ax.get_xticks()
            if len(xt) >= 2:
                ax.spines['bottom'].set_bounds(xt[0], xt[-1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        axes.append(ax)

    if y_all:
        pad = 0.05 * ((max(y_all) - min(y_all)) or 1)
        ylim = (min(y_all) - pad, max(y_all) + pad)
        for ax in axes:
            ax.set_ylim(ylim)

    for ax in axes:
        yt = ax.get_yticks()
        if len(yt) >= 2:
            ax.spines['left'].set_bounds(yt[0], yt[-1])

    if group_order and not group_by_circuit:
        handles = [Line2D([0], [0], color=colour_map.get(g, 'black'), lw=0, marker='s',
                          markersize=8, markerfacecolor=colour_map.get(g, 'black'),
                          markeredgecolor='none', alpha=0.7) for g in group_order]
        axes[-1].legend(handles, group_order, frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

    return sig

#%%
# Figure 2: panels a-c (letters drawn on the figure)
cvs = iter(hs.gridspec(1, 2, wspace=.1, width_ratios=(.85, .15), where=np.divide(hs.config.landscape_A(4), [1, 3])))
cv_left = iter(hs.gridspec(2, 1, wspace=.1, height_ratios=[.7, .3], where=next(cvs)))

# FIGURE 2a -- network diagram (top-left): skip=1 leaves the circuit-colour legend strip unlabelled -- it's
# created before the per-epoch panels but isn't itself a data panel; count=1 labels only the
# first epoch panel, so the whole multi-epoch block reads as a single lettered subplot
hs.label_panels(plot_canonical_corr_network_kdim, start='a', skip=1, count=1)(
    edge_df=corr_cv_df,
    edge_var='∆rho',
    inp_cvs=next(cv_left),
    circuit_map=conf_dict['circuit_map'],
    colour_map=conf_dict['circuit_colour'])

# FIGURE 2b -- bootstrap edge-variance panels (bottom-left): one label for the whole block
hs.label_panels(plot_edge_variance_by_session, start='b', count=1)(
    edge_df=corr_cv_df,
    cca_df=st.canonical_corr,
    edge_var=r'∆rho',
    ylabel=r'mean ∆$\rho$',
    circuit_map=conf_dict['circuit_map'],
    colour_map=conf_dict['circuit_colour'],
    where=next(cv_left)
)

# FIGURE 2c -- significant CCA component counts by epoch (right)
group_counts = hs.label_panels(plot_corr_cv_group_counts, start='c')(
    corr_cv_df,
    circuit_map=conf_dict['circuit_map'],
    circuit_colour=conf_dict['circuit_colour'],
    epoch_order=conf_dict['epochs'],
    where=next(cvs),
)
hs.savefig(f'{out_root}_all_area_pair_cca_corr_ndim.svg', wspace=0.1, dpi=600, open=False)
hs.savefig(f'{out_root}_all_area_pair_cca_corr_ndim.png', wspace=0.1, dpi=600, open=OPEN_FIGS)

