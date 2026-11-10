# Distributed under the MIT license; see LICENSE.
# Task decoding from the shared (CCA) subspace (project 624290): are task variables -- reward (trial_order), effort
# (effort_lev), trial count and running velocity -- linearly decodable from the canonical components shared between pairs of
# areas in each epoch? Uses CCA components with inspiration from Stoll et al. 2024
# https://doi.org/10.1016/j.cub.2024.08.034 as templates.
# The CCA components (cross-validated component activity per session x epoch x area pair, project 382702/480152) and
# the decoder fits (cross-validated R^2 against a permutation null, and the per-component betas; data/624290_regression.feather)
# are precomputed; this script summarises them across sessions and animals and draws FIGURE 3 of the manuscript
# (saved as output/624290_regression_r^2_breakdown):
#   Fig. 3a  network of area pairs whose task variable decoding exceeds chance, per epoch
#   Fig. 3b  decoding R^2 by task variable and epoch (bars: mean across sessions; dots: sessions)
#   Fig. 3c  composition of the decoded variance by task variable and epoch
#   Fig. 3d  accept-vs-reject decoder comparison (project 258066), with the per-epoch Delta-r swarms as insets
# The script also computes the angles between the decoders' CC beta vectors (reward vs effort, and reward vs velocity /
# trial count). Those two angle figures are additional analyses, not main-figure panels; their letters (a, b) are
# local to each figure.

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from scipy.stats import wilcoxon, gaussian_kde, norm, ttest_1samp
from typing import Iterable, Optional
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd(), '..', 'lib'))
import helpers as hs
import framework as fw
from block import BlockObj as BlO
from matplotlib.lines import Line2D
from statsmodels.formula.api import mixedlm
from block import var_labels, approximate_var_label
from catalogue import get_file_root, get_output_root, PNG_DIR

PROJECT_ID = 624290
OPEN_FIGS = False  # open the saved PNGs in the OS viewer
conf_dict = {
    'p_thresh': .01,
    'task_vars': ["trial_order", "effort_lev", "trial_count", "velocity"],
    'angle_vars': ["trial_order", "effort_lev"],
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
                       'Fr-BG': '#862E9C'},
}
mpl.rcParams['svg.fonttype'] = 'none'
f_root = get_file_root(PROJECT_ID)      # input data: data/624290_*
out_root = get_output_root(PROJECT_ID)  # figures: output/624290_*


def load_SpikeStats(src=PROJECT_ID):
    st = BlO.SpikeStats(get_file_root(src))
    st.version = fw.spike_stats_ids['canonical_corr']
    st.load()
    return st


def collapse_beta_df_to_models(
    beta_df: pd.DataFrame,
    drop_cols: Optional[Iterable[str]] = ("beta", "cc_beta_norm", "n_trials", "rank", "predictor", "task", "subspace",
                                          "score_metric", "n_cc"),
    row_type_col: Optional[str] = None,
    row_type_value: str = "beta",
) -> pd.DataFrame:
    """
    Collapse a beta-level regression dataframe down to one row per model.

    This removes coefficient-specific rows/columns and keeps only the
    model-level information repeated across the beta rows.

    Parameters
    ----------
    beta_df : pd.DataFrame
        Input dataframe containing one row per coefficient.

    drop_cols : iterable of str, optional
        Columns to ignore when defining a unique model. By default:
        ("beta", "rank", "predictor")

    row_type_col : str or None, optional
        If provided, filter rows to row_type_col == row_type_value before
        collapsing. Useful if beta_df contains mixed row types.

    row_type_value : str
        Value to keep in row_type_col if row_type_col is provided.

    Returns
    -------
    pd.DataFrame
        One row per unique model.
    """
    df = beta_df.copy()

    if row_type_col is not None:
        if row_type_col not in df.columns:
            raise ValueError(f"{row_type_col!r} not found in beta_df.")
        df = df[df[row_type_col] == row_type_value].copy()

    # beta_fold_<j> (cca_component_decoder(save_fold_betas=True)) are per-coefficient too, so they
    # must go with `beta`, otherwise every coefficient row of a model would count as its own model
    drop_cols = [c for c in df.columns
                 if c in (drop_cols or []) or str(c).startswith("beta_fold_")]
    keep_cols = [c for c in df.columns if c not in drop_cols]

    if len(keep_cols) == 0:
        raise ValueError("No columns left after dropping beta/rank/predictor.")

    # Drop exact duplicates across the remaining model-defining columns.
    model_df = df[keep_cols].drop_duplicates().reset_index(drop=True)

    return model_df


st = load_SpikeStats()  # decoder results (regression) of project 624290 on the CC activity of the shared subspaces

# The epochs are those of the CCA dimensionality analysis (project 382702)
st_stat = load_SpikeStats(382702)
st_stat.canonical_corr = st_stat.canonical_corr[st_stat.canonical_corr["rank"] != -1]
epochs = st_stat.get_sorted_epochs(st_stat.canonical_corr.epoch.unique())
conf_dict['epochs'] = epochs
conf_dict['bin_max'] = {e: 1 for i, e in enumerate(epochs)}
conf_dict['edges'] = {e: [i, i] for i, e in enumerate(epochs)}
conf_dict['col_n'] = np.array(list(conf_dict['bin_max'].values())).sum()

# CC-space task-vector coefficients only
beta_df = st.regression.query("rank >= 0").copy()
summary_beta_df_ = collapse_beta_df_to_models(beta_df)

# Discard all instances where R^2 < 0 and R^2 smaller than the permuted average
summary_beta_df = summary_beta_df_[(summary_beta_df_['score'] > 0) &
                                  (summary_beta_df_['score'] > summary_beta_df_['score_perm_avg'])]
summary_beta_df['rank'] = np.zeros_like(summary_beta_df['score'])
edge_df = pd.DataFrame()
for p in conf_dict['task_vars']:
    res = st.get_cca_df_edge_params(alpha=conf_dict['p_thresh'],
                                    agg_fn=np.median,
                                    stat_var=f'score',
                                    perm_var=f'score_perm_avg',
                                    edge_var=f'score',
                                    fdr=False,
                                    best_rank_only=False,
                                    source_df=summary_beta_df.query(f'outcome == "{p}"'))
    res.drop_duplicates(subset=['epoch', 'area_pair', 'rank'], inplace=True)  # one row per edge; deduplicating on edge_pval alone dropped distinct edges that share a Wilcoxon p-value
    res['coefficient'] = np.repeat(p, res.shape[0])
    edge_df = pd.concat([edge_df, res[res['edge_pval'] < conf_dict['p_thresh']]])
edge_df.drop(columns=['edge_sem'], inplace=True)
print(edge_df)

# Task variables with at least one significant cohort-level decoding model somewhere in edge_df
# (i.e. in any epoch/area_pair) - reused below to drop task variables with none at all (currently
# effort_lev) from panel (b)'s fixed x-axis and from the later participation-ratio analysis,
# rather than leaving them as an empty-but-labelled column/group.
sig_task_vars = [tv for tv in conf_dict['task_vars'] if tv in set(edge_df['coefficient'])]

#%%
def build_global_network_legend(edge_params, coef_cmap=None):

    # ---- all coefficients across dataset ----
    all_coefs = sorted(edge_params['coefficient'].unique())

    if coef_cmap is None:
        cmap = mpl.cm.get_cmap("tab10")
        coef_to_color = {c: cmap(i % cmap.N) for i, c in enumerate(all_coefs)}
    else:
        coef_to_color = {k: coef_cmap[k] for k in all_coefs}

    # ---- all ranks across dataset ----
    all_ranks = sorted(edge_params['rank'].unique())

    # same mapping as your plotting function
    rank_to_alpha = {
        r: 1.0 - (i / max(1, len(all_ranks)-1)) * 0.7
        for i, r in enumerate(all_ranks)
    }

    return {
        "coefs": all_coefs,
        "coef_to_color": coef_to_color,
        "ranks": all_ranks,
        "rank_to_alpha": rank_to_alpha
    }

def plot_coef_legend(ax, coef_to_color):
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=coef_to_color[c], lw=4, label=approximate_var_label(str(c)).capitalize())
        for c in coef_to_color
    ]
    with mpl.rc_context(hs.no_axes):
        ax = hs.get_ax(ax)
        ax.legend(
            handles=handles,
            loc="center",
            ncol=len(handles),   # 🔥 horizontal
            frameon=False,
            title="Regressor"
        )

def plot_scalebar(ax, edge_values, global_edge_range, linewidth_range=(1, 2), **scalebar_kwargs):
    hs.add_edge_width_scalebar_quantiles(
        ax=ax,
        edge_values=edge_values,
        linewidth_range=linewidth_range,
        global_edge_range=global_edge_range,
        **(scalebar_kwargs or {})
    )

# Data window onto fw.node_plot_coords() (the nodes only span x 0.26-0.92, y 0.15-0.73 of the unit
# square they live in): cropped so each node diagram fills its axes instead of the whole square
NODE_VIEW = ((0.16, 1.02), (0.05, 0.83))  # ~0.1 margin round the nodes: their circles are ~0.08 in data units
FULL_VIEW = ((0.0, 1.0), (0.0, 1.0))  # the whole unit square, eg. so the brain-slice background is fully visible

def plot_network(df, where, node_size=180, view=NODE_VIEW):
    cv = iter(hs.gridspec(3, 1, wspace=.3, where=where, height_ratios=[.05, .15, .8]))
    ax_cv = hs.gridspec(1, 1, wspace=.2, # axis for coefficient legend
                        # width_ratios=[.2, .4, .4],
                        where=next(cv))
    next(cv)
    node_cv = hs.gridspec(1, len(conf_dict['bin_max'])+1, wspace=.1,
                          width_ratios=np.hstack([np.tile(.9, len(conf_dict['bin_max'])), .05]),
                          where=next(cv))

    node_labels = {n: n for n in fw.node_plot_coords()}
    # Check which area pairs have edges to plot
    nodes_with_finite_edges = set()

    df_finite = df[np.isfinite(df["edge_value"])]

    for (u, v) in df_finite["edge_label"]:
        nodes_with_finite_edges.update([u, v])

    # Configure line widths across epochs and area pairs
    all_edges = df["edge_value"].values
    global_vmin, global_vmax = np.quantile(all_edges, [0.01, 0.99])

    legend = build_global_network_legend(edge_df, fw.venn_prm_colours())

    for epoch, ax in zip(epochs, node_cv):
        png_path = os.path.join(PNG_DIR, epoch + '.png')
        ax = st.plot_node_graph_4(coords=fw.node_plot_coords(),
                                  edge_params=edge_df[edge_df['epoch']==epoch].drop('epoch', axis=1),
                                  thresh=conf_dict['p_thresh'],
                                  node_labels=node_labels,
                                  rc_context=hs.no_axes,
                                  node_size=node_size,
                                  label_fontsize=4,
                                  where=ax,
                                  background_png=os.path.join(PNG_DIR, 'brain_slice.png') if epoch == 'PRE' else None,
                                  ylabel=None,
                                  epoch_png_path=png_path,
                                  title=None,
                                  plot_ns_edges=False,
                                  annotate_edges=False,
                                  nodes_with_finite_edges=nodes_with_finite_edges,
                                  scalebar_title=r'R$^2$',
                                  coef_cmap=fw.venn_prm_colours(),
                                  show_legends=False,
                                  show_scalebar=False,
                                  view=view,
                                  )
        if epoch == 'PRE':
            plot_scalebar(ax, edge_values=edge_df['edge_value'],
                          global_edge_range=(global_vmin, global_vmax),
                          scalebar_title=r'R$^2$')
            np.hstack([.1, np.tile(.9, len(conf_dict['bin_max']))])
    plot_coef_legend(ax_cv[0], legend["coef_to_color"])
    # plot_rank_legend(node_cv[-1], legend["ranks"])

def plot_r2_stacked_composition(
    df: pd.DataFrame,
    outcomes,
    score_col="score",
    outcome_col="outcome",
    epoch_col="epoch",
    score_p_col="score_p",
    p_thresh=None,
    color_map=None,
    epoch_sorter=None,
    where=(4, 3),
    normalize="positive",   # "positive", "abs", or None
    x_col_label="epoch",
    title=None,
    ylabel="Fraction of total R$^2$",
    rotation=30,
    alpha=0.85,
):
    """
    Plot stacked model R^2 composition across epochs from a collapsed model dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Collapsed dataframe, typically one row per model.

    outcomes : list
        Ordered list of outcomes to display as stacked components.

    score_col : str
        Column containing the model score, typically R^2.

    outcome_col : str
        Column naming the task/outcome regressor.

    epoch_col : str
        Column naming the epoch.

    score_p_col : str
        Optional p-value column. Used only if p_thresh is not None.

    p_thresh : float or None
        If supplied, keep only rows with score_p_col < p_thresh.

    color_map : dict or None
        Mapping from outcome name to colour. If None, fw.venn_prm_colours()
        is used.

    epoch_sorter : callable or None
        Function that returns an ordered list of epochs, e.g.
        block.get_sorted_epochs(epochs).

    normalize : {"positive", "abs", None}
        How to form the stacked composition:
            "positive" -> clip negative scores at 0
            "abs"      -> use absolute values
            None       -> use raw values

    Returns
    -------
    ax : matplotlib axis
    """
    if outcome_col not in df.columns:
        raise KeyError(f"Missing outcome column: {outcome_col}")
    if epoch_col not in df.columns:
        raise KeyError(f"Missing epoch column: {epoch_col}")
    if score_col not in df.columns:
        raise KeyError(f"Missing score column: {score_col}")

    plot_df = df.copy()

    if p_thresh is not None:
        if score_p_col not in plot_df.columns:
            raise KeyError(f"p_thresh was supplied but {score_p_col!r} is missing.")
        plot_df = plot_df.loc[plot_df[score_p_col] < p_thresh].copy()

    plot_df = plot_df.loc[plot_df[outcome_col].isin(outcomes)].copy()

    if plot_df.empty:
        raise ValueError("No data available after filtering.")

    if color_map is None:
        color_map = fw.venn_prm_colours()
    else:
        # Allow caller overrides while keeping the project palette as default.
        base = fw.venn_prm_colours()
        base.update(color_map)
        color_map = base

    pivot = (
        plot_df.groupby([epoch_col, outcome_col], as_index=False)[score_col].mean()
        .pivot(index=epoch_col, columns=outcome_col, values=score_col)
        .reindex(columns=outcomes)
    )

    if pivot.empty:
        raise ValueError("No data available for stacked composition plot.")

    # Order epochs using the external sorter if supplied.
    if callable(epoch_sorter):
        try:
            epoch_order = epoch_sorter(list(plot_df[epoch_col].dropna().unique()))
            epoch_order = [e for e in epoch_order if e in pivot.index]
        except Exception:
            epoch_order = list(pivot.index)
    else:
        epoch_order = list(pivot.index)

    pivot = pivot.reindex(epoch_order)

    x_labels = list(pivot.index)
    x_pos = np.arange(len(x_labels), dtype=float)

    if normalize == "positive":
        vals = pivot.clip(lower=0)
    elif normalize == "abs":
        vals = pivot.abs()
    elif normalize is None:
        vals = pivot.fillna(0.0)
    else:
        raise ValueError("normalize must be 'positive', 'abs', or None")

    denom = vals.sum(axis=1).replace(0, np.nan)
    frac = vals.div(denom, axis=0).fillna(0.0)
    frac.index = x_labels

    with mpl.rc_context(hs.rc0):
        ax = hs.get_ax(where)

        y = frac.T.to_numpy()

        ax.stackplot(
            x_pos,
            y,
            labels=[approximate_var_label(str(c)) for c in outcomes],
            colors=[color_map.get(c, "grey") for c in outcomes],
            alpha=alpha,
        )

        ax.set_xlim(-0.5, len(x_labels) - 0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([approximate_var_label(str(v)) for v in x_labels], rotation=rotation, ha="center")
        ax.set_xlabel(x_col_label)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)

        if len(x_pos) > 0:
            ax.spines["bottom"].set_bounds(x_pos[0], x_pos[-1])

        if title:
            ax.set_title(title)

        legend_handles = [
            Line2D([0], [0], color=color_map.get(c, "grey"), lw=6,
                   label=approximate_var_label(str(c)).capitalize())
            for c in outcomes
        ]
        ax.legend(
            handles=legend_handles,
            ncol=min(4, len(outcomes)),
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
        )



#%%

def plot_r2_bar_swarm_by_session(edge_df, beta_df, score_var='score', thresh=0.05,
                                 epochs=None, colour_map=None, group_order=None,
                                 where=None, bar_width=0.6, dot_size=3, ylabel=None,
                                 group_by_task_var=True):
    """
    Companion to plot_network / plot_r2_variance_by_session: for every (epoch, area_pair,
    coefficient) edge drawn there as significant (edge_pval < thresh), shows the per-session R^2
    as a bar at the mean with individual subject-day values swarm-plotted on top - simpler and
    more robust than the bootstrapped-KDE version when there are too few subject-days for a smooth
    sampling distribution (a handful of sessions gives a visibly "wavy"/undersampled KDE curve).

    group_by_task_var : bool
        If True, pool the per-subject-day score values across all significant area_pairs sharing a
        task variable (coefficient) within an epoch, and plot that pooled set instead of one column
        per edge. Every epoch panel then shows the *same* fixed x-axis - one position per entry of
        group_order (default conf_dict['task_vars']) - so a task variable sits at the same x
        position in every epoch regardless of whether it was significant there, and an epoch with
        nothing significant is a normal, fully-labelled but empty panel rather than a hidden
        placeholder. Aligns column order/count with plot_network's per-epoch panels above. If
        False, one column per significant (area_pair, coefficient) edge instead, with the dynamic
        (per-epoch-only) column set and placeholder-for-empty-epochs behaviour from before.
    """
    epochs_ = epochs if epochs is not None else st.get_sorted_epochs(edge_df['epoch'].unique())
    where = np.divide(hs.config.landscape_A(4), [1, 3]) if where is None else where
    colour_map = fw.venn_prm_colours() if colour_map is None else colour_map
    group_order = list(conf_dict['task_vars']) if group_order is None else group_order

    sig = edge_df[(pd.to_numeric(edge_df['edge_pval'], errors='coerce') < thresh) &
                 np.isfinite(pd.to_numeric(edge_df['edge_value'], errors='coerce'))].copy()

    def _session_values(epoch, area_pair, coefficient):
        tbl = beta_df.query('epoch == @epoch & area_pair == @area_pair & outcome == @coefficient')
        if tbl.empty:
            area_pair = '_'.join(area_pair.split('_')[::-1])
            tbl = beta_df.query('epoch == @epoch & area_pair == @area_pair & outcome == @coefficient')
        return tbl.groupby('subject_day')[score_var].mean().to_numpy()

    def _pooled_task_var_values(epoch, coefficient):
        pairs = sig.loc[(sig['epoch'] == epoch) & (sig['coefficient'] == coefficient), 'area_pair'].unique()
        parts = [_session_values(epoch, ap, coefficient) for ap in pairs]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.array([])

    def _draw_column(ax, x, vals, col):
        mean = vals.mean()
        y_all.extend([float(vals.min()), float(vals.max()), float(mean)])
        ax.bar(x, mean, width=bar_width, color=col, alpha=0.4, zorder=2, edgecolor=col)
        sns.swarmplot(x=np.full(vals.shape, float(x)), y=vals, color=col, edgecolor='black',
                      linewidth=0.6, size=dot_size, ax=ax, zorder=4, native_scale=True)

    # Fixed x-axis labels, shared by every epoch panel, so a given task variable sits at the same
    # x position regardless of which epochs it turned out significant in - this is what makes a
    # "nothing significant here" epoch legible as an empty panel (still labelled with every task
    # variable) rather than either a shrunken subset of columns or a hidden placeholder, and keeps
    # column order aligned with plot_network's per-epoch panels above.
    group_labels = [approximate_var_label(c).capitalize() for c in group_order]

    gs = iter(hs.gridspec(1, len(epochs_), wspace=.05, where=where))
    axes, has_data_list, y_all = [], [], []

    for i, epoch in enumerate(epochs_):
        rc_c = hs.rc0 if not i else hs.no_yaxis
        with mpl.rc_context(rc_c):
            ax = hs.get_ax(next(gs))
            ax.set_ylim(-1, 1)

            if group_by_task_var:
                for xi, coefficient in enumerate(group_order):
                    vals = _pooled_task_var_values(epoch, coefficient)
                    if vals.size:
                        _draw_column(ax, xi, vals, colour_map.get(coefficient, 'black'))

                # Every epoch gets a fully-formatted panel regardless of whether anything was
                # significant in it - an epoch with nothing to show is then an empty-but-labelled
                # set of axes, not a special-cased blank.
                has_data = True
                ax.axhline(0, color='0.6', alpha=.5, lw=.8, ls='--', zorder=1)
                ax.set_xticks(np.arange(len(group_labels)))
                ax.set_xticklabels(group_labels, rotation=90)
                ax.set_yticks(np.arange(0, 0.9, 0.1))
                for y in np.arange(0, 0.8, 0.2):
                    ax.axhline(y, color='0.6', lw=0.4, zorder=1)
                ax.set(xlim=(-0.5, len(group_labels) - 0.5))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                xtick_labels = []
                for _, row in sig[sig['epoch'] == epoch].drop_duplicates(['area_pair', 'coefficient']).iterrows():
                    vals = _session_values(epoch, row['area_pair'], row['coefficient'])
                    if not vals.size:
                        continue
                    _draw_column(ax, len(xtick_labels), vals, colour_map.get(row['coefficient'], 'black'))
                    u, v = row['edge_label']
                    xtick_labels.append(f'{u}-{v}\n({approximate_var_label(row["coefficient"])})')

                has_data = bool(xtick_labels)
                if has_data:
                    ax.axhline(0, color='0.6', lw=0.6, ls='--', zorder=1)
                    for y in np.arange(0, 0.8, 0.2):
                        ax.axhline(y, color='0.6', lw=0.4, zorder=1)
                    ax.set_xticks(np.arange(len(xtick_labels)))
                    ax.set_xticklabels(xtick_labels, rotation=90)
                    ax.set_yticks(np.arange(0, 0.9, 0.1))
                    ax.set(xlim=(-0.5, len(xtick_labels) - 0.5))
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                else:
                    ax.set_xlim(-0.5, 0.5)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.text(0.5, 0.5, 'n.s.', transform=ax.transAxes, ha='center', va='center',
                           fontsize=8, color='0.6')
            ax.set_title(approximate_var_label(epoch))
            if i == 0 and has_data:
                ylab = ylabel if ylabel is not None else 'R$^2$\n(mean bar, per-session dots)'
                if group_by_task_var and ylabel is None:
                    ylab += '\npooled within task variable'
                ax.set_ylabel(ylab)
        axes.append(ax)
        has_data_list.append(has_data)

    if y_all:
        pad = 0.05 * ((max(y_all) - min(y_all)) or 1)
        ylim = (min(y_all) - pad, max(y_all) + pad)
    else:
        ylim = (-1, 1)
    if not np.isfinite(ylim[0]) or not np.isfinite(ylim[1]) or ylim[0] == ylim[1]:
        ylim = (-1, 1)
    for ax, has_data in zip(axes, has_data_list):
        if has_data:
            ax.set_ylim(ylim)

    for ax, has_data in zip(axes, has_data_list):
        if not has_data:
            continue
        ax.spines['bottom'].set_bounds(*ax.get_xlim())
        ax.spines['left'].set_bounds(*ax.get_ylim())
    if where is None:
        hs.savefig(f'{out_root}_r2_bar_swarm{"" if group_by_task_var else "_by_edge"}.svg',
                  wspace=0.05, dpi=300, open=False)
        hs.savefig(f'{out_root}_r2_bar_swarm{"" if group_by_task_var else "_by_edge"}.png',
                  wspace=0.05, dpi=300, open=OPEN_FIGS)

    return sig


#%%

def calc_cc_task_variable_angles(
    regression: pd.DataFrame,
    var_a: str = "trial_order",
    var_b: str = "effort_lev",
    group_cols=("subject_day", "area", "area_pair", "epoch", "bin", "subspace"),
    min_shared_ranks: int = 2,
    min_norm: float = 1e-8,
) -> pd.DataFrame:
    """
    Compute the angle between two task-variable beta vectors in CC space.

    var_a ~ CC_1 + ... + CC_K and var_b ~ CC_1 + ... + CC_K are fit on the
    same CC columns within a given group (see cca_component_decoder), so
    their fold-averaged beta vectors live in the same CC-dimensional space
    and are directly comparable. Because each vector comes from its own
    regression, its sign is well-defined (unlike raw CCA canonical weights,
    which are sign-ambiguous), so the full 0-180 deg range is meaningful:
    0/180 deg = (anti-)collinear coding of reward and effort, 90 deg =
    orthogonal coding.

    Parameters
    ----------
    regression : pd.DataFrame
        Beta-level output of Block.cca_component_decoder() (st.regression),
        or any row subset of it (e.g. pre-filtered by significance).

    var_a, var_b : str
        Values of the `outcome` column to compare.

    group_cols : tuple of str
        Columns to compute a separate angle within. Only columns actually
        present in `regression` are used.

    min_shared_ranks : int
        Minimum number of shared, non-NaN CC ranks required to trust an
        angle estimate (a 1-D "vector" always gives 0 deg or 180 deg).

    min_norm : float
        Drop groups where either beta vector's norm is ~0 (the target isn't
        represented in the CC subspace at all, so the angle is undefined).

    sig_only : bool
        Whether to filter the regression table by models that survive the significance threshold
    Returns
    -------
    pd.DataFrame with one row per group:
        <group_cols>, n_cc, norm_a, norm_b, cos_theta, angle_deg
    """
    group_cols = [c for c in group_cols if c in regression.columns]
    if not group_cols:
        raise ValueError("None of the requested group_cols are present in `regression`.")

    cc_df = regression.loc[
        (regression["rank"] >= 0) &
        (regression["outcome"].isin([var_a, var_b]))
    ].copy()
    rows = []
    for keys, g in cc_df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)

        pivot = g.pivot_table(index="rank", columns="outcome", values="beta", aggfunc="mean")
        if var_a not in pivot.columns or var_b not in pivot.columns:
            continue

        pivot = pivot.dropna(subset=[var_a, var_b])
        if pivot.shape[0] < min_shared_ranks:
            continue

        v_a = pivot[var_a].to_numpy(dtype=float)
        v_b = pivot[var_b].to_numpy(dtype=float)

        norm_a = np.linalg.norm(v_a)
        norm_b = np.linalg.norm(v_b)
        if norm_a < min_norm or norm_b < min_norm:
            continue

        # deliberately not clipped to [-1, 1]: |cos_theta| > 1 means the normalisation is broken, and
        # arccos then returns NaN (with a RuntimeWarning) instead of that being silently masked
        cos_theta = np.dot(v_a, v_b) / (norm_a * norm_b)
        angle_deg = np.degrees(np.arccos(cos_theta))

        row = dict(zip(group_cols, keys))
        row.update(dict(
            n_cc=pivot.shape[0],
            norm_a=norm_a,
            norm_b=norm_b,
            cos_theta=cos_theta,
            angle_deg=angle_deg,
        ))
        rows.append(row)

    return pd.DataFrame(rows)


def test_angle_orthogonality(
    regression: pd.DataFrame,
    var_a: str = "trial_order",
    var_b: str = "effort_lev",
    group_cols=("subject_day", "area_pair", "epoch"),
    min_shared_ranks: int = 2,
    min_norm: float = 1e-8,
    min_shuffle_n: int = 5,
    n_null: int = 2000,
    seed: int = 0,
):
    """
    Test whether observed reward/effort CC-beta angles are significantly *tighter* around 90 deg
    than expected by chance -- evidence of consistently orthogonal coding, as opposed to angles
    that merely average out near 90 deg because unrelated random vectors in a shared CC-space also
    concentrate there (concentration of measure; see null_angle_stats).

    Null construction (mismatch-pairing within n_cc strata)
    ---------------------------------------------------------
    For each group (same group_cols as calc_cc_task_variable_angles), the var_a and var_b beta
    vectors are extracted. Groups are then stratified by n_cc (shared CC-rank count), since the
    null variance of cos_theta itself shrinks as n_cc grows (~1/n_cc) -- comparing angle spread
    across groups of different dimensionality directly would conflate that scaling with a real
    effect. Within a stratum with at least min_shuffle_n groups, the null is built by repeatedly
    permuting which var_b vector each var_a vector is paired with (excluding self-pairs) and
    computing cos_theta for every mismatched pair -- this reuses the *actual* fitted beta vectors,
    so it captures any real anisotropy in how they're distributed (eg. if CC1 systematically
    carries larger beta magnitude than CC5 for unrelated reasons) that an idealised
    uniform-random-vector null would miss. Strata too sparse to shuffle meaningfully fall back to
    a simulated null of independent random unit vectors in the same n_cc-dimensional space (as
    null_angle_stats), so no group is dropped just because its n_cc is rare.

    Each group's observed cos_theta is then standardised by its own stratum's null SD (z-score),
    which is what makes groups of different n_cc poolable on one footing downstream (eg. in
    plot_angle_orthogonality_test's per-column significance test).

    Returns
    -------
    df : pd.DataFrame
        One row per group: <group_cols>, n_cc, cos_theta, angle_deg, null_std, z.
    null_pools : dict[int, np.ndarray]
        n_cc -> pooled null cos_theta draws for that stratum (shared across all groups with that
        n_cc), for reuse by the plotting/significance-test function.
    """
    group_cols = [c for c in group_cols if c in regression.columns]
    if not group_cols:
        raise ValueError("None of the requested group_cols are present in `regression`.")

    cc_df = regression.loc[
        (regression["rank"] >= 0) &
        (regression["outcome"].isin([var_a, var_b]))].copy()
    records = []
    for keys, g in cc_df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        pivot = g.pivot_table(index="rank", columns="outcome", values="beta", aggfunc="mean")
        if var_a not in pivot.columns or var_b not in pivot.columns:
            continue
        pivot = pivot.dropna(subset=[var_a, var_b])
        if pivot.shape[0] < min_shared_ranks:
            continue
        v_a, v_b = pivot[var_a].to_numpy(dtype=float), pivot[var_b].to_numpy(dtype=float)
        norm_a, norm_b = np.linalg.norm(v_a), np.linalg.norm(v_b)
        if norm_a < min_norm or norm_b < min_norm:
            continue
        row = dict(zip(group_cols, keys))
        row.update(n_cc=pivot.shape[0], v_a=v_a / norm_a, v_b=v_b / norm_b)
        records.append(row)

    if not records:
        raise ValueError("No groups survived rank/norm filtering - nothing to test.")

    df = pd.DataFrame(records)
    rng = np.random.default_rng(seed)

    null_pools = {}
    for n_cc, sub in df.groupby("n_cc"):
        v_a_stack = np.stack(sub["v_a"].to_numpy())
        v_b_stack = np.stack(sub["v_b"].to_numpy())
        n = len(sub)
        pool = np.array([])
        if n >= min_shuffle_n:
            draws = []
            for _ in range(n_null):
                perm = rng.permutation(n)
                mask = perm != np.arange(n)
                if mask.any():
                    draws.append(np.sum(v_a_stack[mask] * v_b_stack[perm[mask]], axis=1))
            pool = np.concatenate(draws) if draws else np.array([])
        if pool.size < 200:
            # Too few real groups at this n_cc to trust an empirical shuffle null -- fall back to
            # the idealised random-unit-vector null (same logic as null_angle_stats).
            x = rng.normal(size=(n_null * max(n, 1), n_cc))
            y = rng.normal(size=(n_null * max(n, 1), n_cc))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            pool = np.sum(x * y, axis=1)
        null_pools[n_cc] = pool

    df["cos_theta"] = df.apply(lambda r: float(np.dot(r["v_a"], r["v_b"])), axis=1)
    df["angle_deg"] = np.degrees(np.arccos(df["cos_theta"]))
    df["null_std"] = df["n_cc"].map(lambda n: null_pools[n].std())
    df["z"] = df["cos_theta"] / df["null_std"]
    df = df.drop(columns=["v_a", "v_b"])

    return df, null_pools

def plot_angle_orthogonality_test(
    angle_test_df: pd.DataFrame,
    null_pools: dict,
    epoch_col: str = "epoch",
    area_col: str = "area_pair",
    epoch_sorter=None,
    circuit_map=None,
    colour_map=None,
    group_order=None,
    p_thresh: float = 0.05,
    n_perm: int = 5000,
    violin_width: float = 0.35,
    where=None,
    seed: int = 1,
    ylabel=None,
    collapse_epochs: bool = False,
    bh_correct: bool = False,
):
    """
    Companion to test_angle_orthogonality: for every (epoch, circuit_group) column, overlays the
    observed reward-effort angle distribution (right half) against its matched permutation null
    (left half -- mismatch-paired within n_cc strata, or simulated-random-vector fallback for
    sparse strata) as back-to-back kernel-density violins, following the same estimation-plot
    convention as plot_edge_variance_by_session/plot_r2_variance_by_session (black dot = observed
    mean, black ticks = observed 95% percentile range).

    Significance: within each column, observed cos_theta values are standardised by their own
    n_cc stratum's null SD (z), and the column's observed z-variance is compared against a
    resampled null-column-z-variance (drawing surrogate z's from each group's own stratum null
    pool, n_perm times) -- a one-sided test of "the observed angles are more tightly clustered
    than chance". Columns with p < p_thresh (tighter than null) are marked with a gold background
    and an asterisk. The per-column raw p-value, z-variance ratio (var_obs of z -- <1 means
    tighter than null, by construction ~1 under H0) and post-correction significance flag are
    printed and returned as stats_df, so a "looks tighter" read on the KDEs can be checked against
    the actual numbers rather than the asterisk alone.

    collapse_epochs : bool
        If True, pool rows across all epochs within each circuit_group before testing/plotting,
        giving one column (one test) per circuit motif instead of one per (epoch, circuit_group)
        pair. With ~6 epochs x ~3 circuits normally tested independently at the same p_thresh,
        that's a lot of implicit comparisons for any single asterisk to survive -- collapsing
        epochs is the simplest way to cut that down to one test per motif when you don't need
        epoch-resolved detail. n_cc-stratified standardisation already treats rows from different
        epochs as poolable (null_pools are keyed by n_cc alone, not by epoch), so pooling here is
        just a change to which rows land in the same column, not to the underlying null.

    bh_correct : bool
        If True (default), apply Benjamini-Hochberg FDR correction across every column actually
        drawn in this call (ie. the family size shrinks automatically under collapse_epochs=True)
        before deciding which columns get the significance highlight, rather than flagging each
        column at the raw per-column p_thresh -- a figure with ~16 columns each tested at raw
        p<0.05 should expect ~1 false positive from chance alone, which a single-column asterisk
        can't be distinguished from without this. If False, falls back to the raw per-column test.

    Parameters
    ----------
    angle_test_df, null_pools : outputs of test_angle_orthogonality().
    circuit_map, colour_map : as in plot_taskvar_angle_by_epoch -- required here, since columns
        are always grouped by circuit (there's no raw-area_pair mode for this plot).

    Returns
    -------
    df : pd.DataFrame
        angle_test_df restricted to rows matched by circuit_map, with a circuit_group column.
    stats_df : pd.DataFrame
        One row per (epoch, circuit_group) column actually drawn: n, var_ratio, p_value,
        significant -- sorted by p_value ascending.
    """
    if circuit_map is None:
        raise ValueError("circuit_map is required (canonicalises area_col into circuit groups).")

    df = angle_test_df.copy()

    def _canon(pair):
        parts = str(pair).replace("-", "_").split("_")
        return "_".join(sorted(parts)) if len(parts) == 2 else str(pair)

    area_to_group = {_canon(k): v for k, v in circuit_map.items()}
    df["circuit_group"] = df[area_col].apply(lambda p: area_to_group.get(_canon(p)))
    df = df[df["circuit_group"].notna()].copy()
    if df.empty:
        raise ValueError(
            f"No rows matched circuit_map after canonicalising {area_col!r} - nothing to plot."
        )

    colour_map = fw.region_colours() if colour_map is None else colour_map
    group_order = list(dict.fromkeys(circuit_map.values())) if group_order is None else group_order

    if collapse_epochs:
        # None is a sentinel meaning "don't filter by epoch" below -- a single pooled column-set.
        epoch_order = [None]
    elif callable(epoch_sorter):
        try:
            epoch_order = epoch_sorter(list(df[epoch_col].dropna().unique()))
            epoch_order = [e for e in epoch_order if e in df[epoch_col].unique()]
        except Exception:
            epoch_order = sorted(df[epoch_col].dropna().unique())
    else:
        epoch_order = sorted(df[epoch_col].dropna().unique())

    rng = np.random.default_rng(seed)

    def _column_stats(sub):
        # var_obs is already on the "ratio to null variance" scale (not just an arbitrary
        # number): z = cos_theta / null_std, and null_std is exactly the null pool's own SD, so
        # a null column's z-variance is ~1 by construction -- var_obs < 1 means tighter than null.
        z_obs = sub["z"].to_numpy()
        if len(z_obs) < 2:
            return np.nan, np.nan
        var_obs = np.var(z_obs, ddof=1)
        draws = np.stack([
            rng.choice(null_pools[ncc], size=n_perm) / std
            for ncc, std in zip(sub["n_cc"], sub["null_std"])
        ])
        surrogate_var = np.var(draws, axis=0, ddof=1)
        p_val = float(np.mean(surrogate_var <= var_obs))
        return p_val, float(var_obs)

    def _null_angle_sample(sub, n_per_row=300):
        parts = [
            np.degrees(np.arccos(rng.choice(null_pools[ncc], size=n_per_row)))
            for ncc in sub["n_cc"]
        ]
        return np.concatenate(parts) if parts else np.array([])

    def _bh_significant(pvals, alpha):
        # Benjamini-Hochberg step-up procedure: flag every p at or below the largest rank whose
        # sorted p-value still clears its own (rank/m)*alpha threshold.
        order = np.argsort(pvals)
        ranked = pvals[order]
        m = len(pvals)
        below = ranked <= (np.arange(1, m + 1) / m) * alpha
        if not below.any():
            return np.zeros(m, dtype=bool)
        cutoff = ranked[np.max(np.where(below)[0])]
        return pvals <= cutoff

    # Pass 1: identify every (epoch, circuit_group) column that will actually be drawn, and
    # compute its raw p-value -- needed up front so BH correction sees the true family size
    # (which columns exist depends on collapse_epochs and which groups have any data at all).
    columns = []
    for epoch in epoch_order:
        for group in group_order:
            group_mask = df["circuit_group"] == group
            sub = df[group_mask] if epoch is None else df[(df[epoch_col] == epoch) & group_mask]
            if sub.empty:
                continue
            p_val, var_obs = _column_stats(sub)
            columns.append(dict(epoch=epoch, group=group, sub=sub, p=p_val, var_obs=var_obs))

    pvals = np.array([c["p"] for c in columns], dtype=float)
    finite = np.isfinite(pvals)
    significant = np.zeros(len(pvals), dtype=bool)
    if finite.any():
        if bh_correct:
            significant[finite] = _bh_significant(pvals[finite], alpha=p_thresh)
        else:
            significant[finite] = pvals[finite] < p_thresh
    for c, sig in zip(columns, significant):
        c["significant"] = sig

    stats_df = pd.DataFrame([
        dict(epoch="all (collapsed)" if c["epoch"] is None else c["epoch"],
             circuit_group=c["group"], n=len(c["sub"]), var_ratio=c["var_obs"],
             p_value=c["p"], significant=c["significant"])
        for c in columns
    ]).sort_values("p_value", na_position="last").reset_index(drop=True)
    print(f"Orthogonality test per column (var_ratio = observed/null z-variance, <1 = tighter; "
         f"bh_correct={bh_correct}):")
    print(stats_df.to_string(index=False))

    # Pass 2: draw each epoch's axis using the columns/significance computed above.
    gs = iter(hs.gridspec(1, len(epoch_order), wspace=.05,
                          where=np.divide(hs.config.landscape_A(4), [1, 2]) if where is None else where))
    axes = []

    for i, epoch in enumerate(epoch_order):
        rc_c = hs.rc0 if not i else hs.no_yaxis
        with mpl.rc_context(rc_c):
            ax = hs.get_ax(next(gs))
            ax.set_ylim(0, 180)
            xtick_labels = []

            for c in [c for c in columns if c["epoch"] == epoch]:
                group, sub = c["group"], c["sub"]
                x = len(xtick_labels)
                obs_vals = sub["angle_deg"].to_numpy()
                null_vals = _null_angle_sample(sub)
                col = colour_map.get(group, "black")

                for vals, side, color, alpha in [
                    (null_vals, -1, "0.6", 0.5), (obs_vals, 1, col, 0.6)
                ]:
                    if vals.size >= 2 and np.ptp(vals) > 0:
                        y_grid = np.linspace(max(vals.min(), 0), min(vals.max(), 180), 200)
                        density = gaussian_kde(vals)(y_grid)
                        width = density / density.max() * violin_width
                        ax.fill_betweenx(y_grid, x, x + side * width, color=color, alpha=alpha,
                                        lw=0, zorder=2)

                ax.scatter([x], [obs_vals.mean()], color="black", s=13, zorder=4)
                lo, hi = np.percentile(obs_vals, [2.5, 97.5])
                ax.plot([x, x], [lo, hi], color="black", lw=1.0, zorder=3, solid_capstyle="butt")

                if c["significant"]:
                    ax.axvspan(x - violin_width - 0.05, x + violin_width + 0.05,
                              color="gold", alpha=0.15, zorder=0)
                    ax.text(x, 177, "*", ha="center", va="top", fontsize=12, fontweight="bold")

                xtick_labels.append(group)

            ax.axhline(90, color="0.6", lw=0.6, ls="--", zorder=1)
            if xtick_labels:
                ax.set_xlim(-0.5, len(xtick_labels) - 0.5 + violin_width)
                ax.set_xticks(np.arange(len(xtick_labels)))
                ax.set_xticklabels(xtick_labels, rotation=90)
            ax.set_title("All epochs (pooled)" if epoch is None else approximate_var_label(epoch))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i == 0:
                ax.set_ylabel(
                    ylabel
                    or "Angle between reward and effort\nCC beta vectors (deg)\n"
                )
        axes.append(ax)

    for ax in axes:
        ax.spines["bottom"].set_bounds(*ax.get_xlim())
        ax.spines["left"].set_bounds(*ax.get_ylim())

    if where is None:
        hs.savefig(f'{out_root}_reward_effort_angle_orthogonality_test.svg', wspace=0.05, dpi=300,
                  open=False)
        hs.savefig(f'{out_root}_reward_effort_angle_orthogonality_test.png', wspace=0.05, dpi=300,
                  open=OPEN_FIGS)

    return df, stats_df


def fit_cos_theta_lmm(sub: pd.DataFrame):
    """
    Intercept-only linear mixed model of cos_theta (the dot product of the two unit-normalised
    beta vectors), cos_theta ~ 1, testing H1: mean dot product != 0 (zero = orthogonal, which is
    also the expectation for randomly oriented vectors). Testing the dot product rather than the
    angle keeps the response linear, so the mean across sessions is a meaningful estimate and the
    CLT justifies a Wald test even though individual sessions are not distinguishable from chance.

    Random effects: an intercept per subject (animal), plus a session (subject_day) variance
    component nested within subject, since several area pairs of one circuit share a session and
    so are not independent. If that nested fit fails (only a few animals, so it can be singular),
    falls back to the subject-only random intercept. REML, so the fixed-effect SE is not
    downward-biased (unlike the ML fits used for the likelihood-ratio comparisons below).

    Returns (mean dot product, SEM, Wald-z p-value, diagnostics), all NaN if there are fewer than 2
    subjects. diagnostics is a dict for checking whether the SEM/p can be trusted: which model was
    fit ('nested' or 'subject-only'), whether it converged, the fitted subject / session / residual
    variances (a subject or session variance of ~0 means that level of clustering collapsed and the
    SEM is then too optimistic), the mean dot product per subject, and the mean and p-value of a
    plain one-sample t-test across session means (rows averaged within session first, so no
    mixed model and no pseudo-replication from several area pairs per session).
    """
    sub = sub.copy()
    sub["subject"] = sub["subject_day"].str.split("_").str[0]
    diag = dict(model=None, converged=np.nan, var_subject=np.nan, var_session=np.nan,
                var_resid=np.nan, subject_means="", mean_cos_session=np.nan, p_session=np.nan)
    if sub["subject"].nunique() < 2:
        return np.nan, np.nan, np.nan, diag

    sess_means = sub.groupby("subject_day")["cos_theta"].mean()
    diag["mean_cos_session"] = float(sess_means.mean())
    if sess_means.size >= 2:
        diag["p_session"] = float(ttest_1samp(sess_means.to_numpy(), popmean=0).pvalue)
    diag["subject_means"] = "; ".join(
        f"{subj}={g['cos_theta'].mean():+.3f} (n_sess={g['subject_day'].nunique()})"
        for subj, g in sub.groupby("subject")
    )

    try:
        # re_formula="1" is required: when vc_formula is given, statsmodels drops the default
        # per-group random intercept, which would leave only the session component (no animal term)
        res = mixedlm("cos_theta ~ 1", sub, groups=sub["subject"], re_formula="1",
                      vc_formula={"session": "0 + C(subject_day)"}).fit(reml=True)
        diag["model"] = "nested"
        diag["var_session"] = float(res.vcomp[0])
    except (np.linalg.LinAlgError, ValueError):
        res = mixedlm("cos_theta ~ 1", sub, groups=sub["subject"]).fit(reml=True)
        diag["model"] = "subject-only"
    diag["converged"] = bool(res.converged)
    diag["var_subject"] = float(np.asarray(res.cov_re).ravel()[0])
    diag["var_resid"] = float(res.scale)
    return (float(res.params["Intercept"]), float(res.bse["Intercept"]),
            float(res.pvalues["Intercept"]), diag)


def plot_taskvar_angle_mean_by_epoch(
    angle_df: pd.DataFrame,
    var_a: str = "trial_order",
    var_b: str = "effort_lev",
    epoch_col: str = "epoch",
    area_col: str = "area_pair",
    epoch_sorter=None,
    colour_map=None,
    circuit_map=None,
    where=(5, 6),
    p_thresh: float = None,
    bh_correct: bool = False,
    trend_thresh: float = 0.05,
    equiv_delta: Optional[float] = 0.2,
    equiv_alpha: float = 0.05,
    ci_mult: float = 1.96,
    dodge: float = 0.25,
    title=None,
    ylabel=None,
    rotation=30,
):
    """
    Replaces plot_taskvar_angle_by_epoch's per-session dots with one point per (epoch, circuit
    group): the angle of the mean dot product across sessions, with error bars at
    arccos(mean + ci_mult*SEM) and arccos(mean - ci_mult*SEM) -- ie. a 95% interval for the
    default ci_mult=1.96, so a bar clearing 90 deg matches p < .05 rather than the much weaker
    +/-1 SEM (|z| > 1) -- where mean and SEM come from fit_cos_theta_lmm
    (the angle is only ever taken of the *averaged* dot product, never averaged itself). Sessions
    differ in subspace size (n_cc), so individual angles have very different null spreads and
    plotting each one makes the data look noisier than they are.

    A column gets `*` when its mean dot product differs from 0 (Wald test from the mixed model,
    p < p_thresh, Benjamini-Hochberg corrected across the drawn columns if bh_correct); columns
    that miss that but have a raw p < trend_thresh get `#` instead. The
    dashed line at 90 deg is the null (dot product 0). circuit_map/colour_map as in
    plot_taskvar_angle_by_epoch; circuit_map is required.

    A mean dot product that can't be told from 0 is NOT evidence that the two vectors are unrelated
    or orthogonal -- only that no consistent alignment was detected. So each column also gets an
    equivalence test (TOST, normal approximation on the same mean and SEM) against +/- equiv_delta:
    `equivalent` (marked `≈`) means |mean dot product| < equiv_delta is established at
    equiv_alpha (BH-corrected across the drawn columns if bh_correct), ie. the 1 - 2*equiv_alpha
    interval of the mean dot product lies inside (-equiv_delta, +equiv_delta). equiv_delta is the
    smallest alignment you would still care about (0.2 = 78.5 deg, 4% shared variance); the grey band
    on the plot is the corresponding range of angles, arccos(equiv_delta) to arccos(-equiv_delta).
    Whatever delta you pick, `equiv_delta_min` (and `equiv_angle_min`, its angle) in stats_df is the
    smallest margin at which equivalence would hold for that column: |mean| + z(1 - equiv_alpha)*SEM
    -- the bound you can report ("any alignment is weaker than |cos| = ...") without choosing delta
    in advance. equiv_delta=None skips the band, the markers and the TOST p-values (the
    bound columns are still returned).

    Returns
    -------
    ax : matplotlib axis
    stats_df : pd.DataFrame
        One row per (epoch, circuit_group): n_rows, n_sessions, n_subjects, mean_cos, sem, z,
        p_value, significant, trend, angle_deg (of the mean), angle_lo/angle_hi (of mean + ci_mult*SEM /
        mean - ci_mult*SEM), plus the fit_cos_theta_lmm diagnostics (model, converged, var_subject,
        var_session, var_resid, subject_means, mean_cos_session, p_session).
    """
    if circuit_map is None:
        raise ValueError("circuit_map is required (canonicalises area_col into circuit groups).")
    if angle_df.empty:
        raise ValueError("angle_df is empty - nothing to plot.")
    p_thresh = conf_dict['p_thresh'] if p_thresh is None else p_thresh

    def _canon(pair):
        parts = str(pair).replace("-", "_").split("_")
        return "_".join(sorted(parts)) if len(parts) == 2 else str(pair)

    area_to_group = {_canon(k): v for k, v in circuit_map.items()}
    df = angle_df.copy()
    df["circuit_group"] = df[area_col].apply(lambda p: area_to_group.get(_canon(p)))
    df = df[df["circuit_group"].notna()].copy()
    if df.empty:
        raise ValueError(f"No rows matched circuit_map after canonicalising {area_col!r}.")

    colour_map = fw.region_colours() if colour_map is None else colour_map
    group_order = [g for g in dict.fromkeys(circuit_map.values()) if g in df["circuit_group"].unique()]
    if callable(epoch_sorter):
        epoch_order = [e for e in epoch_sorter(list(df[epoch_col].dropna().unique()))
                       if e in df[epoch_col].unique()]
    else:
        epoch_order = sorted(df[epoch_col].dropna().unique())

    rows = []
    for epoch in epoch_order:
        for group in group_order:
            sub = df[(df[epoch_col] == epoch) & (df["circuit_group"] == group)]
            if sub.empty:
                continue
            mean_cos, sem, p, diag = fit_cos_theta_lmm(sub)
            delta_min = abs(mean_cos) + norm.ppf(1 - equiv_alpha) * sem
            p_equiv = (max(norm.sf((mean_cos + equiv_delta) / sem), norm.cdf((mean_cos - equiv_delta) / sem))
                       if equiv_delta is not None else np.nan)
            rows.append(dict(
                epoch=epoch, circuit_group=group, n_rows=len(sub),
                n_sessions=sub["subject_day"].nunique(),
                n_subjects=sub["subject_day"].str.split("_").str[0].nunique(),
                mean_cos=mean_cos, sem=sem, z=mean_cos / sem, p_value=p,
                angle_deg=np.degrees(np.arccos(mean_cos)),
                angle_lo=np.degrees(np.arccos(mean_cos + ci_mult * sem)),
                angle_hi=np.degrees(np.arccos(mean_cos - ci_mult * sem)),
                p_equiv=p_equiv,
                equiv_delta_min=delta_min,
                equiv_angle_min=np.degrees(np.arccos(np.clip(delta_min, -1, 1))),
                **diag,
            ))
    stats_df = pd.DataFrame(rows)

    pvals = stats_df["p_value"].to_numpy(dtype=float)
    finite = np.isfinite(pvals)
    significant = np.zeros(len(pvals), dtype=bool)
    if finite.any():
        if bh_correct:
            significant[finite] = hs.benjamini_hochberg(pvals[finite], alpha=p_thresh)
        else:
            significant[finite] = pvals[finite] < p_thresh
    stats_df["significant"] = significant
    stats_df["trend"] = finite & (pvals < trend_thresh) & ~significant

    p_eq = stats_df["p_equiv"].to_numpy(dtype=float)
    finite_eq = np.isfinite(p_eq)
    equivalent = np.zeros(len(p_eq), dtype=bool)
    if finite_eq.any():
        if bh_correct:
            equivalent[finite_eq] = hs.benjamini_hochberg(p_eq[finite_eq], alpha=equiv_alpha)
        else:
            equivalent[finite_eq] = p_eq[finite_eq] < equiv_alpha
    stats_df["equivalent"] = equivalent
    print(f"Mean dot product vs 0, per column (mixed model, bh_correct={bh_correct}):")
    print(stats_df.to_string(index=False))

    with mpl.rc_context(hs.rc0):
        ax = hs.get_ax(where)
        offsets = np.linspace(-dodge, dodge, len(group_order)) if len(group_order) > 1 else [0.0]
        x_of = {g: o for g, o in zip(group_order, offsets)}

        for _, r in stats_df.iterrows():
            if not np.isfinite(r["angle_deg"]):
                continue
            x = epoch_order.index(r["epoch"]) + x_of[r["circuit_group"]]
            col = colour_map.get(r["circuit_group"], "black")
            ax.plot([x, x], [r["angle_lo"], r["angle_hi"]], color=col, lw=1.4, zorder=3,
                    solid_capstyle="butt")
            ax.scatter([x], [r["angle_deg"]], color=col, edgecolor="black", linewidth=0.5, s=22,
                       zorder=4)
            if r["significant"] or r["trend"]:
                ax.text(x, r["angle_lo"] - 0.5, "*" if r["significant"] else "#", ha="center",
                        va="bottom", fontsize=12, fontweight="bold", color="black")
            if r["equivalent"]:
                ax.text(x, r["angle_hi"] + 0.5, "≈", ha="center", va="bottom", fontsize=10,
                        color="black")

        if equiv_delta is not None:
            ax.axhspan(np.degrees(np.arccos(equiv_delta)), np.degrees(np.arccos(-equiv_delta)),
                       color="grey", alpha=0.12, lw=0, zorder=0)
        ax.axhline(90, color="grey", ls="--", lw=1, zorder=0)
        ax.set_xticks(np.arange(len(epoch_order)))
        ax.set_xticklabels([approximate_var_label(e) for e in epoch_order], rotation=rotation,
                           ha="center")
        ax.set_xlim(-0.5, len(epoch_order) - 0.5)
        finite_angles = stats_df[["angle_lo", "angle_hi"]].to_numpy(dtype=float)
        finite_angles = finite_angles[np.isfinite(finite_angles)]
        lo = min(finite_angles.min(), 90) if finite_angles.size else 80
        hi = max(finite_angles.max(), 90) if finite_angles.size else 100
        if equiv_delta is not None:  # keep the whole equivalence band in view
            lo = min(lo, np.degrees(np.arccos(equiv_delta)))
            hi = max(hi, np.degrees(np.arccos(-equiv_delta)))
        pad = 0.1 * (hi - lo) or 1
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("")
        ax.set_ylabel(
            ylabel
            or f"Angle of mean dot product between {approximate_var_label(var_a)} and "
               f"{approximate_var_label(var_b)}\nCC beta vectors (deg, mean $\\pm$ 95% CI)"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            handles=[Line2D([0], [0], marker="o", linestyle="None", color=colour_map.get(g, "grey"),
                            label=g) for g in group_order],
            frameon=False, ncol=min(4, len(group_order)), loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
        )
        if title:
            ax.set_title(title)

    return ax, stats_df
#%%

# Restrict to the same significance criterion already used for summary_beta_df/edge_df
# above (positive, above-chance R^2), applied row-wise so the CC-rank rows needed for
# the angle calculation are preserved.

angle_df = calc_cc_task_variable_angles(
    st.regression,
    var_a=conf_dict['angle_vars'][0],
    var_b=conf_dict['angle_vars'][1],
)
print(angle_df.groupby("area")["angle_deg"].describe())
angle_test_df, angle_null_pools = test_angle_orthogonality(
    st.regression,
    var_a=conf_dict['angle_vars'][0],
    var_b=conf_dict['angle_vars'][1],
)

#%%
# Plot angle statistics panel (additional analysis, not a main-figure panel; panel letters a-b are local to this figure)
where = np.divide(hs.config.landscape_A(4), [1, 1])
cvs = iter(hs.gridspec(3, 1, wspace=.2, where=where))
cv1 = iter(hs.gridspec(1, 2, wspace=.1, where=next(cvs), width_ratios=[.3, .7]))

hs.label_panels(plot_taskvar_angle_mean_by_epoch, start='a', skip=0, count=1)(
    angle_df,
    var_a=conf_dict['angle_vars'][0],
    var_b=conf_dict['angle_vars'][1],
    epoch_sorter=st.get_sorted_epochs,
    circuit_map=conf_dict['circuit_map'],
    colour_map=conf_dict['circuit_colour'],
    where=next(cv1),
    title="Reward-effort CC geometry across epochs",
)
hs.label_panels(plot_angle_orthogonality_test, start='b', skip=0, count=1)(
    angle_test_df,
    angle_null_pools,
    epoch_sorter=st.get_sorted_epochs,
    circuit_map=conf_dict['circuit_map'],
    colour_map=conf_dict['circuit_colour'],
    collapse_epochs=False,
    where=next(cv1)
)

lab = f'{out_root}_{conf_dict["angle_vars"][0]}_{conf_dict["angle_vars"][1]}_angle_breakdown.png'
hs.savefig(lab, wspace=0.1, dpi=600, open=OPEN_FIGS)

#%%
# Do the task variables share an axis in the CC subspace? (a collaborator's question: "is reward in the same axis as velocity and
# trial number")
#
# Generalises the reward-effort angle panel above to other pairs of task variables: per session /
# area pair / epoch, the dot product of the two (unit-normalised) CC beta vectors, summarised per
# (epoch, circuit) by plot_taskvar_angle_mean_by_epoch as the angle of the mean dot product (mixed
# model over animals and sessions; H0: mean dot product = 0, ie. unrelated axes). cos > 0 means the
# two variables are read out along the same direction of the shared subspace, cos < 0 along
# opposite directions, cos ~ 0 along independent directions.
#
# Three things to keep in mind when reading it:
#  * Only (epoch, area_pair) combinations where BOTH variables have a cohort-level significant
#    decoder (edge_df) are compared. Where a variable isn't decodable its "beta vector" is noise,
#    and noise vectors are orthogonal to everything, so including them would read as "separate
#    axes" for no neural reason.
#  * Comparisons are WITHIN epoch only. Every epoch has its own CCA fit, so CC rank k is a different
#    neural direction in each epoch (and only the cross-validated CC *scores* are saved, not the
#    canonical weights): the betas of reward at CUE and velocity at CORR1 are not in the same
#    coordinate system, and their angle would be meaningless. "Is the pursuit velocity code a
#    rotation of the decision-making reward code" needs the weights saved so the betas can be
#    mapped back to unit space (or one CCA fitted across both epochs).
#  * The decoders are fit one variable at a time (control_other_task_vars=False), so two variables
#    that are correlated in the behaviour (eg. reward and trial number across a session, or
#    velocity and reward within pursuit) get aligned betas even if the neural population only
#    carries one of them. A positive angle shift therefore means "decoded along the same axis",
#    not yet "the same neural code" -- re-running cca_component_decoder with
#    control_other_task_vars=True gives the *unique* betas and is the control for this.

def restrict_to_jointly_decoded(angle_df, edge_df, var_a, var_b, area_col='area_pair'):
    """Keep only the rows of angle_df from (epoch, area_pair) combinations where both var_a and
    var_b have a significant cohort-level decoder in edge_df (order-invariant on the pair name)."""
    def _canon(pair):
        parts = str(pair).replace('-', '_').split('_')
        return '_'.join(sorted(parts)) if len(parts) == 2 else str(pair)

    sig = edge_df.assign(_ap=edge_df['area_pair'].map(_canon))
    keys = {v: set(zip(sig.loc[sig['coefficient'] == v, 'epoch'], sig.loc[sig['coefficient'] == v, '_ap']))
            for v in (var_a, var_b)}
    both = keys[var_a] & keys[var_b]
    keep = np.array([(e, _canon(ap)) in both for e, ap in zip(angle_df['epoch'], angle_df[area_col])],
                    dtype=bool)
    return angle_df[keep].copy()


# The pairs shown: reward (trial_order) against velocity and against trial number. (Any other pair works
# the same way -- eg. itertools.combinations over the variables in sig_task_vars, which drops those with
# no significant decoder at all, such as effort_lev.)
var_pairs = [('trial_order', 'velocity'), ('trial_order', 'trial_count')]
# smallest dot product (alignment) still worth caring about: 0.2 = 78.5 deg = 4% shared variance. Columns whose
# 90% CI lies inside +/- this are marked "≈" (equivalent to unrelated); stats' equiv_delta_min is the smallest
# delta each column would pass, if you'd rather report that than commit to a delta
equiv_delta = 0.2
pair_angle_dfs = {}
for var_a, var_b in var_pairs:
    ang = restrict_to_jointly_decoded(
        calc_cc_task_variable_angles(st.regression, var_a=var_a, var_b=var_b), edge_df, var_a, var_b)
    if ang.empty:
        print(f'{var_a} vs {var_b}: no (epoch, area pair) with a significant decoder for both - skipped')
        continue
    pair_angle_dfs[(var_a, var_b)] = ang
    print(f'{var_a} vs {var_b}: {len(ang)} rows from {ang["subject_day"].nunique()} sessions in epochs '
          f'{st.get_sorted_epochs(ang["epoch"].unique())}')

#%%
# All task-variable pairs: angle of the mean dot product between their CC beta vectors, per epoch and circuit
# (additional analysis, not a main-figure panel; one panel per variable pair, lettered a, b, ...)
where = np.divide(hs.config.landscape_A(4), [1, 1])
cvs = iter(hs.gridspec(3, 1, wspace=.2, where=where))
cv_pairs = iter(hs.gridspec(1, len(pair_angle_dfs), wspace=.35, where=next(cvs)))

pair_axes, pair_stats = [], []
for i, ((var_a, var_b), ang) in enumerate(pair_angle_dfs.items()):
    ax, stats = hs.label_panels(plot_taskvar_angle_mean_by_epoch, start=chr(ord('a') + i), skip=0, count=1)(
        ang,
        var_a=var_a,
        var_b=var_b,
        epoch_sorter=st.get_sorted_epochs,
        circuit_map=conf_dict['circuit_map'],
        colour_map=conf_dict['circuit_colour'],
        where=next(cv_pairs),
        bh_correct=True,  # BH across this pair's epoch x circuit columns
        equiv_delta=equiv_delta,
        title=f'{approximate_var_label(var_a).capitalize()} vs {approximate_var_label(var_b)}',
        # shared y-axis: label the first panel only (' ' because the function falls back to its long
        # default label for an empty string)
        ylabel='Angle of mean dot product (deg, mean $\\pm$ 95% CI)' if i == 0 else ' ',
    )
    # dashed grey lines at the borders between neighbouring epochs (epochs sit at x = 0, 1, 2, ...)
    for x_border in np.arange(stats['epoch'].nunique() - 1) + 0.5:
        ax.axvline(x_border, color='0.6', ls='--', lw=0.8, zorder=0)
    pair_axes.append(ax)
    pair_stats.append(stats.assign(var_a=var_a, var_b=var_b))
pair_stats_df = pd.concat(pair_stats, ignore_index=True)

# one shared y-axis so the panels are directly comparable
lims = pair_stats_df[['angle_lo', 'angle_hi']].to_numpy(dtype=float)
lims = lims[np.isfinite(lims)]
y_lo = min(lims.min(), 90, np.degrees(np.arccos(equiv_delta)))
y_hi = max(lims.max(), 90, np.degrees(np.arccos(-equiv_delta)))
y_pad = 0.1 * (y_hi - y_lo) or 1
for ax in pair_axes:
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

hs.savefig(f'{out_root}_task_var_pair_angles.png', wspace=0.1, dpi=600, open=OPEN_FIGS)
hs.savefig(f'{out_root}_task_var_pair_angles.svg', wspace=0.1, dpi=600)

#%%
# Per-session precision of the CC dot products (jackknife over the decoder's cross-validation folds).
# Needs cca_component_decoder(save_fold_betas=True): the beta_fold_<j> columns of st.regression.

def calc_cc_jackknife_dot_products(
    regression: pd.DataFrame,
    var_a: str = "trial_order",
    var_b: str = "effort_lev",
    group_cols=("subject_day", "area", "area_pair", "epoch", "bin", "subspace"),
    min_shared_ranks: int = 2,
    min_norm: float = 1e-8,
    min_folds: int = 3,
    return_folds: bool = False,
) -> pd.DataFrame:
    """
    Dot product (cos theta) between the var_a and var_b CC beta vectors per group, as in
    calc_cc_task_variable_angles, plus a jackknife standard error for it.

    Every cross-validation fold of cca_component_decoder is a delete-one-block jackknife replicate:
    the fold-j model is fit on all of the session's trials except the j-th contiguous block, so
    cos_j = cos(beta_a^(j), beta_b^(j)) is "the dot product recomputed without block j" (the two
    variables' fold j leave out the same block -- see the save_fold_betas docstring of
    cca_component_decoder). With K usable folds the jackknife variance of the session's dot product
    is

        SE^2 = (K - 1) / K * sum_j (cos_j - mean_j cos_j)^2

    -- not the plain variance of the cos_j: the folds share (K-1)/K of their training trials, so
    they are far more alike than independent samples would be and their raw spread understates the
    sampling uncertainty by the factor sqrt(K - 1).

    Returns one row per group: <group_cols>, n_cc, n_folds (usable folds), cos_theta (from the stored
    fold-mean betas, ie. the same estimate calc_cc_task_variable_angles uses), angle_deg,
    cos_fold_mean (mean of the replicates), se_jack, and cos_folds (the per-fold values) if
    return_folds. se_jack is NaN where fewer than min_folds folds are usable.

    Calibration (simulated sessions, 150 trials, 10 folds, 4 ranks, se_jack vs the true spread of the
    dot product over repeated datasets): within ~5% when the decoders carry real signal (orthogonal
    strong 0.114 vs 0.121, orthogonal weak 0.346 vs 0.366, aligned cos = .7 0.081 vs 0.085); for
    decoders with no signal at all it OVER-estimates (mean 0.66 vs true 0.50, rms 0.71) -- the
    conservative direction, since it then reads as "noise-limited", which is what such a session is.
    """
    fold_cols = sorted((c for c in regression.columns if str(c).startswith("beta_fold_")),
                       key=lambda c: int(str(c).rsplit("_", 1)[1]))
    if not fold_cols:
        raise ValueError("No beta_fold_* columns in `regression` - re-run cca_component_decoder with "
                         "save_fold_betas=True.")
    group_cols = [c for c in group_cols if c in regression.columns]
    if not group_cols:
        raise ValueError("None of the requested group_cols are present in `regression`.")

    cc_df = regression.loc[(regression["rank"] >= 0) & regression["outcome"].isin([var_a, var_b]),
                           group_cols + ["rank", "outcome", "beta"] + fold_cols]

    def _cos(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if not (nu >= min_norm and nv >= min_norm):  # also catches NaN norms
            return np.nan
        return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))

    rows = []
    for keys, g in cc_df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        per = g.groupby(["outcome", "rank"])[["beta"] + fold_cols].mean()
        outcomes = per.index.get_level_values("outcome")
        if var_a not in outcomes or var_b not in outcomes:
            continue
        a, b = per.loc[var_a], per.loc[var_b]
        shared = a.index.intersection(b.index)
        a, b = a.loc[shared], b.loc[shared]
        ok = (a["beta"].notna() & b["beta"].notna()).to_numpy()
        a, b = a[ok], b[ok]
        if len(a) < min_shared_ranks:
            continue

        cos_full = _cos(a["beta"].to_numpy(float), b["beta"].to_numpy(float))
        cos_k = np.array([_cos(a[c].to_numpy(float), b[c].to_numpy(float)) for c in fold_cols])
        cos_k = cos_k[np.isfinite(cos_k)]  # a failed fold (NaN betas) drops out for both variables
        K = cos_k.size
        if K >= min_folds:
            cos_bar = float(cos_k.mean())
            se = float(np.sqrt((K - 1) / K * np.sum((cos_k - cos_bar) ** 2)))
        else:
            cos_bar = se = np.nan

        row = dict(zip(group_cols, keys))
        row.update(n_cc=len(a), n_folds=K, cos_theta=cos_full,
                   angle_deg=float(np.degrees(np.arccos(cos_full))) if np.isfinite(cos_full) else np.nan,
                   cos_fold_mean=cos_bar, se_jack=se)
        if return_folds:
            row["cos_folds"] = cos_k
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_dot_product_precision(
    jack_df: pd.DataFrame,
    circuit_map: dict,
    area_col: str = "area_pair",
    epoch_col: str = "epoch",
    epoch_sorter=None,
    n_sim: int = 20000,
    seed: int = 0,
    min_rows: int = 3,
) -> pd.DataFrame:
    """
    Per (epoch, circuit): how close to 0 are the dot products, compared with (i) what unrelated random
    directions would give and (ii) how precisely each session's dot product is measured.

    Columns (rms = root mean square over rows, so it is centred on 0 = orthogonal, not on the mean):
      rms_cos       observed rms dot product
      chance_rms    sqrt(mean(1/n_cc)): rms of the dot product of two random directions in n_cc dims
      tightness     mean(n_cc * cos^2); 1 = chance, < 1 = dot products cluster nearer 0 than chance
      p_tight       one-sided Monte Carlo p for tightness < chance (random directions: cos^2 ~
                    Beta(1/2, (n_cc-1)/2), n_sim draws of the whole column). Treats rows as
                    independent (several area pairs of one session are not).
      mean_se       sqrt(mean(se_jack^2)): typical sampling error of one session's dot product
      noise_ratio   mean(se_jack^2) / mean(1/n_cc): sampling variance relative to the chance variance
      signal_rms    sqrt(max(mean(cos^2) - mean(se_jack^2), 0)): what is left of the rms dot product
                    once the average sampling noise is removed

    Reading it: tightness well below 1 (small p_tight) with noise_ratio well below 1 = dot products
    sit at 0 and are measured precisely enough that this is not just noise: consistently orthogonal.
    noise_ratio near or above 1 = single sessions cannot distinguish orthogonal from unrelated, so a
    non-significant tightness says little. signal_rms clearly above 0 = the true dot product varies
    from session to session beyond sampling noise (some aligned / anti-aligned), which a mean of ~0
    would hide.
    """
    def _canon(pair):
        parts = str(pair).replace("-", "_").split("_")
        return "_".join(sorted(parts)) if len(parts) == 2 else str(pair)

    area_to_group = {_canon(k): v for k, v in circuit_map.items()}
    df = jack_df.copy()
    df["circuit_group"] = df[area_col].apply(lambda p: area_to_group.get(_canon(p)))
    df = df[df["circuit_group"].notna() & np.isfinite(df["cos_theta"])]
    group_order = [g for g in dict.fromkeys(circuit_map.values()) if g in df["circuit_group"].unique()]
    epochs_ = df[epoch_col].dropna().unique()
    epoch_order = ([e for e in epoch_sorter(list(epochs_)) if e in epochs_] if callable(epoch_sorter)
                   else sorted(epochs_))

    rng = np.random.default_rng(seed)
    out = []
    for epoch in epoch_order:
        for group in group_order:
            sub = df[(df[epoch_col] == epoch) & (df["circuit_group"] == group)]
            if len(sub) < min_rows:
                continue
            n_cc = sub["n_cc"].to_numpy(float)
            cos = sub["cos_theta"].to_numpy(float)
            se2 = sub["se_jack"].to_numpy(float) ** 2
            chance_var = np.mean(1.0 / n_cc)
            tight_obs = np.mean(n_cc * cos ** 2)
            sim = np.mean(n_cc * rng.beta(0.5, (n_cc - 1) / 2, size=(n_sim, len(sub))), axis=1)
            mean_se2 = np.nanmean(se2) if np.isfinite(se2).any() else np.nan
            out.append(dict(
                epoch=epoch, circuit_group=group, n_rows=len(sub),
                n_sessions=sub["subject_day"].nunique() if "subject_day" in sub else np.nan,
                mean_cos=cos.mean(), rms_cos=np.sqrt(np.mean(cos ** 2)), chance_rms=np.sqrt(chance_var),
                tightness=tight_obs, p_tight=(1 + np.sum(sim <= tight_obs)) / (n_sim + 1),
                mean_se=np.sqrt(mean_se2), noise_ratio=mean_se2 / chance_var,
                signal_rms=np.sqrt(max(np.mean(cos ** 2) - mean_se2, 0.0)) if np.isfinite(mean_se2) else np.nan,
            ))
    return pd.DataFrame(out)


# Example (needs the fold betas, ie. st.regression from a decoder run with save_fold_betas=True):
if 'beta_fold_0' in st.regression.columns:
    jack_dfs = []
    for var_a, var_b in var_pairs:
        jack = calc_cc_jackknife_dot_products(st.regression, var_a=var_a, var_b=var_b)
        jack = restrict_to_jointly_decoded(jack, edge_df, var_a, var_b)
        jack_dfs.append(jack.assign(var_a=var_a, var_b=var_b))
    jack_df = pd.concat(jack_dfs, ignore_index=True)
    precision_df = pd.concat(
        [summarise_dot_product_precision(g, conf_dict['circuit_map'], epoch_sorter=st.get_sorted_epochs
                                         ).assign(var_a=a, var_b=b)
         for (a, b), g in jack_df.groupby(['var_a', 'var_b'])], ignore_index=True)
    print(precision_df.round(3).to_string(index=False))
else:
    print('st.regression has no beta_fold_* columns: re-run cca_component_decoder(save_fold_betas=True)')

#%%
# Accept-vs-reject decoder panel (d, with the per-epoch ∆r swarms as insets) for the main figure,
# brought in from 258066.py (the v6 task
# decoder, cca_component_decoder_2: both trial groups train one pooled decoder but are scored
# separately, ∆corr = score_accept - score_reject per model). Everything is kept under v6_* /
# st_v6 names so this project's own st / conf_dict / edge_df stay untouched, and the two plotting
# functions take `epochs` explicitly instead of reading 258066's conf_dict. Only the pre-cue (PRE),
# cue (CUE) and action (NP_EX) epochs go in the main figure -- the corridor pursuit epochs
# (CORR1/CORR6) are for supplemental figure S3.
from collections import defaultdict
from matplotlib.collections import LineCollection

v6_src_id = 258066
v6_fig_epochs = ['PRE', 'CUE', 'NP_EX']

st_v6 = load_SpikeStats(src=v6_src_id)
v6_beta_df = st_v6.regression.query("rank >= 0").copy()
v6_summary_df_ = collapse_beta_df_to_models(v6_beta_df)
v6_score_df = v6_summary_df_[
    np.isfinite(v6_summary_df_['score_accept']) & np.isfinite(v6_summary_df_['score_reject'])].copy()
v6_score_df['zero'] = 0.0
v6_score_df['∆corr'] = v6_score_df['score_accept'] - v6_score_df['score_reject']
v6_score_df['rank'] = 0
# get_cca_df_edge_params groups by (epoch, area_pair, rank) -- the collapsed model-level table
v6_diff_edge_df = pd.DataFrame()
for p in conf_dict['task_vars']:
    res = st_v6.get_cca_df_edge_params(alpha=conf_dict['p_thresh'],
                                       agg_fn=np.nanmean,
                                       fdr=False,
                                       stat_var='∆corr',
                                       perm_var='zero',   # wilcoxon(x, zeros) == one-sample test of x vs 0
                                       edge_var='∆corr',
                                       best_rank_only=False,
                                       alternative='two-sided',  # the sign of accept - reject is itself
                                                                 # part of the question
                                       source_df=v6_score_df.query('outcome == @p'))
    res['coefficient'] = p
    v6_diff_edge_df = pd.concat([v6_diff_edge_df, res], ignore_index=True)

# Restrict to the epochs shown in the main figure. fdr=False above, so each epoch's test is
# independent of which other epochs were included -- filtering afterwards is equivalent to
# running on these epochs only. (The swarm panel's default drop_vars also keys off this table, so
# task variables with no significant edge in these three epochs drop out of panel e.)
v6_diff_edge_df = v6_diff_edge_df[v6_diff_edge_df['epoch'].isin(v6_fig_epochs)].reset_index(drop=True)
v6_score_df = v6_score_df[v6_score_df['epoch'].isin(v6_fig_epochs)]
print(v6_diff_edge_df)


def r2_diff_swarm_columns(edge_df, group_order, thresh, drop_vars=None):
    """
    Task variables kept as columns of the ∆r swarm. By default (drop_vars=None) that is every task
    variable in group_order with at least one significant edge (finite edge_value, edge_pval <
    thresh) somewhere in edge_df, i.e. one that never survives the decoders' accept/reject test at
    all is dropped; pass an explicit tuple (or ()) to override.
    """
    if drop_vars is None:
        surviving = edge_df.loc[(pd.to_numeric(edge_df['edge_pval'], errors='coerce') < thresh) &
                                np.isfinite(pd.to_numeric(edge_df['edge_value'], errors='coerce')),
                                'coefficient'].unique()
        drop_vars = [c for c in group_order if c not in surviving]
    return [c for c in group_order if c not in drop_vars]


def draw_r2_diff_swarm(ax, epoch, edge_df, beta_df, group_order, groups=('accept', 'reject'),
                       thresh=None, colour_map=None, bar_width=0.6, dot_size=3, sig_test=False,
                       sig_fdr=False, sig_marker='*', ylabel=None, title=None, x_labels=True,
                       ylim=(-.5, 1), yticks=None):
    """
    One epoch of plot_r2_diff_swarm_by_session, drawn onto an existing axes: the per-session
    ∆r = score_<groups[0]> - score_<groups[1]> from cca_component_decoder_2, pooled per task
    variable across every area_pair edge_df marks significant (edge_pval < thresh) in `epoch` --
    one bar + swarm per entry of group_order. Split out so the same panel can be used as an inset
    (see plot_r2_diff_multiplex_network) as well as in the one-panel-per-epoch layout.

    Each column additionally gets its own group-level significance test: a one-sample Wilcoxon
    signed-rank test of that column's per-session ∆r values against zero, FDR-corrected across the
    columns drawn in this panel if sig_fdr (otherwise raw p < thresh). Columns surviving it get
    `sig_marker` drawn above them if sig_test.

    :param beta_df: the collapsed, model-level regression table (e.g. v6_score_df) -- NOT the raw
    per-coefficient st.regression, which would double/triple-count the same score across its
    Intercept/CC predictor rows.
    :param x_labels: False drops the task-variable tick labels (and ticks), eg. for an inset whose
    bar colours are already explained by a legend.
    """
    thresh = conf_dict['p_thresh'] if thresh is None else thresh
    colour_map = fw.venn_prm_colours() if colour_map is None else colour_map
    yticks = np.arange(-1, 1.5, .5) if yticks is None else yticks
    score_a, score_b = f'score_{groups[0]}', f'score_{groups[1]}'

    sig = edge_df[(pd.to_numeric(edge_df['edge_pval'], errors='coerce') < thresh) &
                 np.isfinite(pd.to_numeric(edge_df['edge_value'], errors='coerce'))]

    def _session_diff_values(epoch, area_pair, coefficient):
        # (epoch is an argument, not a closure variable: DataFrame.query's @epoch only sees the
        # calling frame's own locals)
        # pools A/B sides of the pair per subject-day, matching how get_cca_df_edge_params/
        # plot_r2_bar_swarm_by_session's own _session_values aggregates
        tbl = beta_df.query('epoch == @epoch & area_pair == @area_pair & outcome == @coefficient')
        if tbl.empty:
            area_pair = '_'.join(area_pair.split('_')[::-1])
            tbl = beta_df.query('epoch == @epoch & area_pair == @area_pair & outcome == @coefficient')
        if tbl.empty:
            return np.array([])
        diff = tbl[score_a] - tbl[score_b]
        return diff.groupby(tbl['subject_day']).mean().to_numpy()

    def _pooled_task_var_values(coefficient):
        pairs = sig.loc[(sig['epoch'] == epoch) & (sig['coefficient'] == coefficient), 'area_pair'].unique()
        parts = [_session_diff_values(epoch, ap, coefficient) for ap in pairs]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.array([])

    # Fix the ticks and limits BEFORE plotting: sns.swarmplot draws the whole figure mid-build (to
    # place the dots), and on a small inset the default auto-locators then hit the tiny-axes
    # IndexError in matplotlib's tick search (index 0 out of bounds ... _raw_ticks). Fixed
    # locators + manual limits (which also switch autoscaling off) never reach that code.
    ax.set_xticks(np.arange(len(group_order)))
    ax.set_yticks(yticks)
    ax.set(xlim=(-0.5, len(group_order) - 0.5), ylim=ylim)

    vals_by_col = [_pooled_task_var_values(c) for c in group_order]

    # first pass: Wilcoxon (vs 0) per column, FDR-corrected across the columns in this panel
    p_by_col = np.full(len(group_order), np.nan)
    for xi, vals in enumerate(vals_by_col):
        if vals.size >= 2:
            try:
                _, p_by_col[xi] = wilcoxon(vals, alternative='two-sided')
            except ValueError:
                pass
    finite = np.isfinite(p_by_col)
    is_sig = np.zeros(len(p_by_col), dtype=bool)
    if sig_fdr:
        if finite.any():
            is_sig[finite] = hs.benjamini_hochberg(p_by_col[finite], alpha=thresh)
    else:
        is_sig[finite] = p_by_col[finite] < thresh

    # second pass: bar + swarm, with sig_marker above significant columns
    for xi, (coefficient, vals) in enumerate(zip(group_order, vals_by_col)):
        if not vals.size:
            continue
        col = colour_map.get(coefficient, 'black')
        mean = vals.mean()
        sns.swarmplot(x=np.full(vals.shape, float(xi)), y=vals, color=col, edgecolor='black',
                      linewidth=0.6, size=dot_size, ax=ax, zorder=4, alpha=.7, native_scale=True)
        ax.bar(xi, mean, width=bar_width, color=col, alpha=.9, zorder=2, edgecolor=col)
        if is_sig[xi] and sig_test:
            sem = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            top = max(mean + sem, vals.max())
            ax.text(xi, top, sig_marker, ha='center', va='bottom', fontsize=9, color='black', zorder=5)

    ax.axhline(0, color='0.6', alpha=.5, lw=.8, ls='--', zorder=1)
    if x_labels:
        ax.set_xticklabels([approximate_var_label(c).capitalize() for c in group_order], rotation=90)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(*ax.get_xlim())
    ax.spines['left'].set_bounds(*ax.get_ylim())
    return ax


def plot_multiplex_node_graph(coords, edge_df, colour_map, node_labels=None, node_size=180,
                              label_fontsize=4, lw=1.2, where=(2.1, 2), rc_context=hs.no_axes,
                              background_png=None, title=None, ylabel=None, epoch_png_path=None,
                              nodes_with_finite_edges=None, max_offset_fraction=0.06, view=None):
    """
    Multiplex analogue of st.plot_node_graph_3_1: `edge_df` is expected pre-filtered to
    significant (epoch, area_pair, coefficient) rows by the caller. Edges sharing an area pair
    (same 'edge_label') are drawn with parallel offsets -- reusing plot_node_graph_3_1's own
    edges_by_pair/offset logic -- but coloured per-row by `edge_df['coefficient']` via
    `colour_map`, rather than per-pair by circuit_map, since a pair can now carry more than one
    task variable's edge at once.
    """
    nodes = list(coords.keys())
    node_labels = {n: str(n) for n in nodes} if node_labels is None else node_labels
    nodes_with_finite_edges = set(nodes) if nodes_with_finite_edges is None else set(nodes_with_finite_edges)

    edges_by_pair = defaultdict(list)
    for idx, row in edge_df.iterrows():
        edges_by_pair[tuple(row['edge_label'])].append(idx)

    pair_offsets = {}
    for (u, v), idxs in edges_by_pair.items():
        m = len(idxs)
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        dx, dy = x1 - x0, y1 - y0
        L = np.hypot(dx, dy)
        if L == 0 or m == 1:
            offsets = [(0.0, 0.0)] * m
        else:
            nx, ny = -dy / L, dx / L
            span = max_offset_fraction * L * (m - 1)
            offsets = [(nx * off, ny * off) for off in np.linspace(-span / 2, span / 2, m)]
        for idx, off in zip(idxs, offsets):
            pair_offsets[idx] = off

    segs, cols = [], []
    for idx, row in edge_df.iterrows():
        u, v = row['edge_label']
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        ox, oy = pair_offsets[idx]
        segs.append([(x0 + ox, y0 + oy), (x1 + ox, y1 + oy)])
        cols.append(colour_map.get(row['coefficient'], 'black'))

    with mpl.rc_context(rc_context):
        ax = hs.get_ax(where)
        hs.add_image_background_norm(ax, fname=background_png)

        for n in nodes:
            if n in nodes_with_finite_edges:
                ax.scatter(*coords[n], s=node_size, color='white',
                          edgecolor=fw.region_colours()[n], linewidth=1, zorder=2)
        for n in nodes:
            ax.text(coords[n][0], coords[n][1], node_labels[n], ha='center', va='center',
                   fontsize=label_fontsize, zorder=3)

        if segs:
            ax.add_collection(LineCollection(segs, linewidths=lw, colors=cols, zorder=1))

        if view is not None:
            ax.set_xlim(*view[0])
            ax.set_ylim(*view[1])
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title, pad=14)
        if epoch_png_path is not None:
            hs.add_png_to_plot(ax=ax, img_path=epoch_png_path, x=.5, y=1.55, zoom=0.08,
                               coords='axes fraction')
        if ylabel is not None:
            ax.text(0, 0.5, ylabel, rotation=90, va='center')

    return ax

def plot_r2_diff_multiplex_network(edge_df, ylabel=None, inp_cvs=None, colour_map=None,
                                   task_var_order=None, thresh=None, epochs=None, view=None,
                                   inset_beta_df=None, inset_groups=('accept', 'reject'),
                                   inset_width=0.84, inset_gap=0.36, inset_bounds=None,
                                   inset_hw_ratio=4 / 3, inset_ylabel=None, inset_x_labels=True,
                                   inset_dot_size=2):
    """
    One network per epoch in `epochs` (default conf_dict['epochs']), with every task variable's
    significant edges (edge_pval < thresh) pooled onto the same diagram.

    view : data window onto the node diagram (default NODE_VIEW, cropped to the nodes; FULL_VIEW
        shows the whole unit square, which is needed to see the PRE brain-slice background in full).
    inset_beta_df : if given (the collapsed model-level score table, e.g. v6_score_df), each epoch's
        ∆r swarm (draw_r2_diff_swarm) is drawn as a small inset in the upper right of that
        epoch's diagram. `view` is widened on the right by `inset_width` (data units, of which
        `inset_gap` is left clear for the inset's y tick labels and label, so they don't sit on
        top of the diagram/background) to make room for it -- the diagrams are height-limited, so
        this costs nothing in diagram size as long as the widened axes stay no wider than their
        cell. `inset_bounds` (axes-fraction x, y, w, h of the region the inset may use) is derived
        from that unless given. `inset_hw_ratio` is the inset's physical height / width (it is
        shrunk to fit its region and anchored top-right). `inset_ylabel` defaults to
        "∆r (<inset_groups[0]> - <inset_groups[1]>)", on the first epoch's inset only;
        `inset_x_labels` toggles the task-variable tick labels.
    """
    thresh = conf_dict['p_thresh'] if thresh is None else thresh
    epochs = list(conf_dict['epochs']) if epochs is None else list(epochs)
    colour_map = fw.venn_prm_colours() if colour_map is None else colour_map
    task_var_order = list(conf_dict['task_vars']) if task_var_order is None else task_var_order

    base_view = NODE_VIEW if view is None else view
    view = base_view
    if inset_beta_df is not None:
        view = ((base_view[0][0], base_view[0][1] + inset_width), base_view[1])
    if inset_bounds is None:
        span = view[0][1] - view[0][0]
        inset_bounds = (1 - (inset_width - inset_gap) / span, 0.30, (inset_width - inset_gap) / span, 0.70)

    where = np.divide(hs.config.landscape_A(4), [1, 1]) if inp_cvs is None else inp_cvs
    cv = hs.gridspec(2, 1, wspace=.1, where=where, height_ratios=[0.2, 0.8])
    bottom_gs = iter(hs.gridspec(1, len(epochs), wspace=.05, where=cv[1]))

    df_finite = edge_df[np.isfinite(pd.to_numeric(edge_df['edge_value'], errors='coerce'))]
    nodes_with_finite_edges = set()
    for u, v in df_finite['edge_label']:
        nodes_with_finite_edges.update([u, v])

    sig = edge_df[(pd.to_numeric(edge_df['edge_pval'], errors='coerce') < thresh) &
                 np.isfinite(pd.to_numeric(edge_df['edge_value'], errors='coerce'))].copy()

    legend_labels = [p for p in task_var_order if p in sig['coefficient'].unique()]
    legend_handles = [
        Line2D([0], [0], color=colour_map.get(lbl, 'black'), lw=3, marker='o', markersize=6,
              markerfacecolor=colour_map.get(lbl, 'black'), markeredgecolor='black',
              markeredgewidth=0.8) for lbl in legend_labels]

    with mpl.rc_context(hs.no_axes):
        leg_ax = hs.get_ax(cv[0])
        if legend_handles:
            leg_ax.legend(legend_handles, [approximate_var_label(l).capitalize() for l in legend_labels],
                          loc='center', ncol=len(legend_labels), frameon=False,
                          handlelength=2.0, columnspacing=1.8)

    inset_cols = r2_diff_swarm_columns(edge_df, task_var_order, thresh)
    inset_ylab = (f'∆$r$ ({inset_groups[0]} - {inset_groups[1]})' if inset_ylabel is None
                  else inset_ylabel)
    for epoch in epochs:
        node_labels = {n: n for n in fw.node_plot_coords()}
        png_path = os.path.join(PNG_DIR, epoch + '.png')

        ax = plot_multiplex_node_graph(
            coords=fw.node_plot_coords(),
            edge_df=sig.loc[sig.epoch == epoch],
            colour_map=colour_map,
            node_labels=node_labels,
            node_size=180,
            label_fontsize=4,
            where=next(bottom_gs),
            background_png=os.path.join(PNG_DIR, 'brain_slice.png') if epoch == 'PRE' else None,
            ylabel=ylabel if epoch == 'PRE' else None,
            epoch_png_path=png_path,
            title=var_labels[epoch],
            nodes_with_finite_edges=nodes_with_finite_edges,
            view=view,
        )
        if inset_beta_df is not None:
            with mpl.rc_context(hs.rc0):
                inset = ax.inset_axes(list(inset_bounds))
                inset.set_box_aspect(inset_hw_ratio)  # physical height / width, whatever the region
                inset.set_anchor('NE')                # ... and tucked into its upper-right corner
                draw_r2_diff_swarm(inset, epoch, edge_df, inset_beta_df, inset_cols,
                                   groups=inset_groups, thresh=thresh, colour_map=colour_map,
                                   dot_size=inset_dot_size, x_labels=inset_x_labels,
                                   ylabel=inset_ylab if epoch == epochs[0] else None)
    if inp_cvs is None:
        hs.savefig(f'{out_root}_v6_∆corr_network_multiplex.svg', wspace=0.1, dpi=300, open=False)
        hs.savefig(f'{out_root}_v6_∆corr_network_multiplex.png', wspace=0.1, dpi=300, open=OPEN_FIGS)


#%%
# FIGURE 3 of the manuscript: panels a-d (letters drawn on the figure)
where = np.divide(hs.config.landscape_A(4), [1, 1])
cvs = iter(hs.gridspec(3, 1, wspace=.2, where=where))
cv1 = iter(hs.gridspec(1, 2, wspace=.1, where=next(cvs), width_ratios=[.8, .2]))
cv_left = iter(hs.gridspec(2, 1, wspace=.1, height_ratios=[.6, .4], where=next(cv1)))

# Fig. 3a
hs.label_panels(plot_network, start='a', skip=0, count=1)(edge_df, next(cv_left), node_size=130)
# Fig. 3b
hs.label_panels(plot_r2_bar_swarm_by_session, start='b', skip=0, count=1)(
    edge_df=edge_df,
    beta_df=summary_beta_df,
    ylabel=r'R$^2$',
    epochs=conf_dict['epochs'],
    group_order=sig_task_vars,  # drop effort_lev: no significant model in any epoch
    where=next(cv_left)
)
# Fig. 3c
hs.label_panels(plot_r2_stacked_composition, start='c', skip=0, count=1)(
    df=summary_beta_df,
    outcomes=conf_dict['task_vars'],
    epoch_sorter=st.get_sorted_epochs,
    color_map=fw.venn_prm_colours(),
    p_thresh=conf_dict["p_thresh"],
    normalize="positive",
    title="Stacked R$^2$ composition across epochs",
    where=next(cv1)
)

# Fig. 3d -- accept vs reject decoder comparison (project 258066), pre-cue/cue/action epochs only -- see the
# preceding cell for where v6_diff_edge_df/v6_score_df come from. The per-epoch ∆r (accept - reject)
# swarms are insets in the upper right of each epoch's network diagram.
hs.label_panels(plot_r2_diff_multiplex_network, start='d', skip=0, count=1)(
    edge_df=v6_diff_edge_df,
    ylabel='',
    epochs=v6_fig_epochs,
    view=FULL_VIEW,  # whole unit square: the PRE brain-slice background must be completely visible
    inset_beta_df=v6_score_df,
    inp_cvs=next(cvs),
)

hs.savefig(f'{out_root}_regression_r^2_breakdown.png', wspace=0.1, dpi=600, open=OPEN_FIGS)
hs.savefig(f'{out_root}_regression_r^2_breakdown.svg', wspace=0.1, dpi=600)
