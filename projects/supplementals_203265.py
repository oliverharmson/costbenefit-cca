# Distributed under the MIT license; see LICENSE.
# Supplemental figures and tables of the manuscript (project 203265), built from the precomputed results of the analysis
# projects in this folder:
#   Fig. S1  unit yield: firing-rate heatmaps (a), units per day by area (b), sessions per area pair and animal (c),
#            combined daily unit yield per area pair and circuit class (d) + mixed-model comparison of the yields
#   Fig. S2  between-animal reliability of the dominant canonical correlation (ICC, mixed models) and Table S1
#   Fig. S3  axes of the replacing task variables, running speed, accept-vs-reject decoding and variance of the true
#            task variables
#   Fig. S4  relation of each area pair's dominant CCA rho to the trial-count coefficient of the per-session choice GLM

import os
import sys
import numpy as np
import pandas as pd
import itertools
from scipy.stats import wilcoxon, gaussian_kde, chi2, norm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd(), '..', 'lib'))
import helpers as hs
import framework as fw
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from statsmodels.formula.api import mixedlm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from typing import Iterable, Optional
from block import BlockObj as BlO
from block import var_labels, approximate_var_label, var_mapping
from catalogue import get_file_root, get_output_root, load_trial_behaviour, load_source_object, PNG_DIR

PROJECT_ID = 203265
OPEN_FIGS = False      # open the saved PNGs in the OS viewer
RECOMPUTE_ICC = False  # re-run the (slow) bootstrapped ICCs for Fig. S2 even if they were computed before (output/203265_icc_*)
RECOMPUTE_VELOCITY = False  # re-extract the running-speed table (Fig. S3b) from the tracking data even if it exists
win_size = 25
conf_dict = {
    'omit': {'trial_order': [-1, 0]},
    'fr_crit': .1,  # Hz, the firing rate criterion based on which units may be discarded
    'fr_var': 'smooth_fr',
    'areas': list(fw.region_colours().keys())[:5],
    'thresh': .01,
    # Task epochs (as in the CCA projects): the behavioural states defining each epoch's window (qs), and the window
    # before/after its alignment point in ms (pres/posts)
    'qs': {'PRE': [f"beh_ch_lab == 'PRE' and beh_ch_lab == 'run_trial'"],
           'CUE': [f"beh_ch_lab == 'CUE' and beh_ch_lab == 'run_trial'"],
           'NP_EX': ["beh_ch_lab == 'EVAL' and beh_ch_lab == 'side_act_0' and beh_ch_lab != 'NP_LG_A' "
                     f"and beh_ch_lab == 'run_trial'",
                     "beh_ch_lab == 'EVAL' and beh_ch_lab == 'side_act_1' and beh_ch_lab != 'NP_LG_B' "
                     f"and beh_ch_lab == 'run_trial'"],
           'CORR1': [f"beh_ch_lab == 'EVAL' and beh_ch_lab == 'side_act_0' and beh_ch_lab == 'C_LG_1_A' "
                     f"and beh_ch_lab == 'run_trial'",
                     f"beh_ch_lab == 'EVAL' and beh_ch_lab == 'side_act_1' and beh_ch_lab != 'C_LG_1_B' "
                     f"and beh_ch_lab == 'run_trial'"],
           'CORR6': [f"beh_ch_lab == 'EVAL' and beh_ch_lab == 'side_act_0' and beh_ch_lab == 'C_LG_1_B' "
                     f"and beh_ch_lab == 'run_trial'",
                     f"beh_ch_lab == 'EVAL' and beh_ch_lab == 'side_act_1' and beh_ch_lab != 'C_LG_1_A' "
                     f"and beh_ch_lab == 'run_trial'"],
           'DEL': [f"beh_ch_lab == DEL'"]},  # This state only exists in accept trials
    'pres': {'PRE': 0, 'CUE': 0, 'NP_EX': 0, 'CORR1': 250, 'CORR6': 250, 'DEL': 0},
    'posts': {'PRE': 300,   # minimum pre-cue interval is 300 ms
              'CUE': 600,   # minimum cue interval is 600 ms
              'NP_EX': 500,  # accept/reject speed profiles diverge 0.9 s before the mean light-gate-1 latency (4.4 s)
              'CORR1': 250,
              'CORR6': 250,
              'DEL': 500},  # the pre-reward delay is 1 s; 500 ms matches the duration of the other windows
    'post_lg_crits': {'PRE': 0, 'CUE': 0, 'NP_EX': 0, 'CORR1': 0, 'CORR6': 0, 'DEL': 0},
    'start_aligned': {'PRE': True, 'CUE': True, 'NP_EX': True, 'CORR1': True, 'CORR6': True, 'DEL': True},
    'bin_size': {
        'mo-right': win_size,
        'acc-right': win_size,
        'str-right': win_size,
        'vp-right': win_size,
        'stn-right': win_size,
    },
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
out_root = get_output_root(PROJECT_ID)  # everything this script generates (figures, tables, cached intermediates): output/203265_*


def load_SpikeStats(src=PROJECT_ID, canon_corr=True):
    st = BlO.SpikeStats(get_file_root(src))
    if canon_corr:
        st.version = fw.spike_stats_ids['canonical_corr']
    st.load()
    return st


#%%
# Supplemental figure 1: unit counts, area pair combination counts, firing rate profiles
st_src_id = 480152
st = load_SpikeStats(src=st_src_id)
st.create_epoch_spike_dfs()
st.get_zero_ps()
conf_dict['bin_max'], conf_dict['col_n'], conf_dict['edges'] = st.get_bin_max_in_epochs(st.spike_times)
conf_dict['subj_days'] = np.unique(np.sort([hs.get_subj_date(b) for b in st.spike_times.stage.unique()]))
conf_dict['subjects'] = sorted(set(s.split('_')[0] for s in conf_dict['subj_days']))  # sorted: a set's order differs between runs
st.spike_times.subject = [s.split('_')[0] for s in st.spike_times.stage]
sp_df = st.sanitise_spike_times_df(fr_var=f'raw_fr_{list(conf_dict["bin_size"].values())[0]}ms_bin',
                                   sp_df=st.spike_times,fr_crit=.1)
unit_stats = pd.DataFrame(columns=list(conf_dict['subjects']))

# Unit-by-time firing-rate matrices behind the heatmaps (a): trial-averaged, one row per unit, from the unit table with
# error trials and trials of the nil reward/effort level dropped. Built once from the unit table and reused afterwards
# (output/480152_fr_mats).
fr_mats_root = get_output_root(st_src_id)
hm_metric = f'{conf_dict["fr_var"]}_{list(conf_dict["bin_size"].values())[0]}ms_bin'
if not all(os.path.exists(os.path.join(fr_mats_root + '_fr_mats', ar, f'{hm_metric}.npy')) for ar in conf_dict['areas']):
    hm_df = st.drop_levs(conf_dict['omit'], st.spike_times.copy())
    hm_df = st.sanitise_spike_times_df(fr_var=f'raw_fr_{list(conf_dict["bin_size"].values())[0]}ms_bin', sp_df=hm_df,
                                       fr_crit=conf_dict['fr_crit'])
    BlO.SpikeStats(fr_mats_root).make_fr_matrices_1_1(sp_df=hm_df, sigma_ms=0, fr_metric=hm_metric, path=fr_mats_root,
                                                      parallel=False)
    del hm_df

def populate_conf_with_corr_unit_pool(upstream, downstream, spike_times=sp_df):
    # Create lists of units to cross-correlate
    # Return a dataframe with the number of unique recording blocks with at least one unit in each area of the pair
    units = {2: spike_times.query(f'area=="{downstream}"')['unit_lab'].unique(),
             1: spike_times.query(f'area=="{upstream}"')['unit_lab'].unique()}
    counter = {s: 0 for s in conf_dict['subjects']}
    if 'unit_pools' not in conf_dict.keys():
        conf_dict['unit_pools'] = {}
    if 'unit_counts' not in conf_dict.keys():
        conf_dict['unit_counts'] = {}
    conf_dict['unit_counts'][downstream] = len(units[2])
    conf_dict['unit_counts'][upstream] = len(units[1])
    for s_d in conf_dict['subj_days']:
        s, d = s_d.split('_')
        units_ = {}
        for k, v in units.items():
            unit_list = [item for item in v if s in item and d in item]
            units_[k] = unit_list
        if all([len(u) for u in units_.values()]):
            units_[1].extend(units_[2])
            if len(units_[1]) and len(units_[2]):
                counter[s] += 1
            if s_d not in conf_dict['unit_pools'].keys():
                conf_dict['unit_pools'][s_d] = [tuple(units_[1])]
            else:
                conf_dict['unit_pools'][s_d].append(tuple(units_[1]))
    stats = pd.DataFrame(index=[f'{fw.region_labels()[upstream]}-{fw.region_labels()[downstream]}'],
                         columns=list(counter.keys()))
    stats.loc[f'{fw.region_labels()[upstream]}-{fw.region_labels()[downstream]}'] = counter
    return stats

for efferent, afferent in list(
        itertools.combinations(conf_dict['areas'], 2)):  # efferent and afferent are random here
    unit_stats_ = populate_conf_with_corr_unit_pool(afferent, efferent, sp_df)
    unit_stats = pd.concat([unit_stats, unit_stats_], ignore_index=False)
    print(f'Finished processing {afferent} and {efferent}')
print(unit_stats)


def plot_per_session_neuron_counts_matplotlib(
    bins_step=2,
    kde_smooth=2.0,
    hist_alpha=0.5,
    kde_linewidth=2.0,
    kde_width_scale=0.9,
    where=(12, 3),
    xlabel="Proportion",
):
    """
    Matplotlib version of the vertical seaborn distplot per area.
    Each panel shows:
      - horizontal histogram of per-day neuron counts (units surviving the firing-rate criterion)
      - a KDE overlaid (scaled to histogram height)
      - dashed median line

    Inputs:
      - where: unused placeholder to match your API; left for compatibility
      - bins_step: bin width (same as bins=np.arange(0, max+2, bins_step))
      - kde_smooth: bandwidth multiplier for gaussian_kde (1.0 is default)
      - kde_width_scale: scale factor to control KDE horizontal width relative to hist max
    """
    # Build summary table: number of unique units per (area, subject-day), after the firing-rate criterion (sp_df).
    # A unit recorded across several blocks of one day is counted once
    day_of_block = {b: hs.get_subj_date(b) for b in sp_df['block'].unique()}
    grp = (sp_df.assign(subj_day=sp_df['block'].map(day_of_block))
           .groupby(['area', 'subj_day'])['unit_lab'].nunique().reset_index())
    areas = st.get_sorted_areas(grp['area'].unique())
    grp = grp.sort_values(
        "area",
        key=lambda s: pd.Categorical(s, categories=areas, ordered=True)
    )

    # region colours mapping (assumes fw.region_colours() returns dict region->color)
    region_col_map = fw.region_colours()
    # create colours list in same order as areas (fallback to a default color)
    colours = [region_col_map.get(a, "lightgray") for a in areas]
    cv = hs.gridspec(1, len(areas), where=where, wspace=.4)

    # compute global y grid / bins
    max_units = int(grp['unit_lab'].max())
    bins = np.arange(0, max_units + bins_step + 1, bins_step)

    for i, (ax, area, col) in enumerate(zip(cv, areas, colours)):
        data = grp.loc[grp['area'] == area, 'unit_lab'].to_numpy()

        # histogram (counts per bin)
        hist_counts, bin_edges = np.histogram(data, bins=bins, density=True)
        # center positions for horizontal bars
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # KDE: estimate on a fine y-grid and scale to histogram peak for overlay
        if len(data) >= 2:
            y_grid = np.linspace(bin_edges[0], bin_edges[-1], 200)
            kde = gaussian_kde(data, bw_method='scott')
            # optional scaling of bandwidth
            if kde_smooth != 1.0:
                kde.set_bandwidth(kde.factor * kde_smooth)
            density = kde(y_grid)
            # scale density to histogram max width so overlay is comparable
            hist_peak = hist_counts.max() if hist_counts.max() > 0 else 1.0
            density_scaled = density / (density.max() if density.max() > 0 else 1.0) * (hist_peak * kde_width_scale)
        with mpl.rc_context(hs.rc0):
            # draw horizontal bar chart
            ax = hs.get_ax(ax)
            ax.barh(bin_centers, hist_counts, height=(bin_edges[1] - bin_edges[0]) * 1,
                    color=col, alpha=hist_alpha, linewidth=0, antialiased=False)
            ax.plot(density_scaled, y_grid, linewidth=kde_linewidth, color=col, zorder=5)

        # median line across panel
        med = np.median(data) if len(data) else np.nan
        if np.isfinite(med):
            ax.axhline(med, color='black', linestyle='--', linewidth=1, alpha=0.6)

        # formatting
        # a zero-width xlim (an area with no/degenerate data) would leave matplotlib's tick
        # locator nothing to work with and crash at draw time, so floor the upper bound
        ax.set(ylim=[0, 50], xlim=[0, max(hist_counts.max(), 1e-6)])
        ax.set_yticks(np.arange(0, 60, 10))
        ax.set_title(fw.region_labels()[str(area)], color=col)
        ax.set_xlabel(xlabel)
        ax.grid(False)


def plot_unit_area_pair_stats_frequency(df=unit_stats, where=(2.5,2.05)):
    from matplotlib.ticker import MaxNLocator

    # make sure values are numeric
    df_plot = df.astype(float)

    ax = hs.get_ax(where)
    ax = sns.heatmap(
        df_plot,
        cmap="rocket",
        annot=True,           # print values
        fmt=".0f",             # integer formatting
        linewidths=1,          # white gridlines
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Number of unique recordings",},
        annot_kws={"size": mpl.rcParams['xtick.labelsize']}  # ← controls annotation font size
    )

    # axis labels
    ax.set_xlabel("Subject")
    ax.set_ylabel("Area pair")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # ---- remove ALL ticks cleanly ----
    ax.minorticks_off()
    ax.tick_params(axis='both', which='major', length=0)
    ax.tick_params(axis='both', which='minor', length=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(which='both', length=0)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Create a figure with heatplots and psths
def plot_heatmaps_grid(normalise=True, where=hs.config.portrait_A(4)):
    """
    :param normalise:
    :param where:
    :return:
    """

    def get_axs(axs) -> mpl.gridspec.GridSpec:
        return hs.gridspec(1, len(conf_dict['bin_max'].keys()) + 1, where=axs, wspace=.15, hspace=0.3,
                           width_ratios=np.append(np.repeat(1, len(conf_dict['bin_max'].keys())), 0.05))

    def get_heatmap_matrix(ar: str) -> (np.array, np.array):
        outcome = st.format_map(conf_dict['fr_var'])(conf_dict['bin_size'][ar])
        ar_p = os.path.join(fr_mats_root + '_fr_mats', ar)
        unit_ids = np.load(os.path.join(ar_p, f'{area}_units.npy'), allow_pickle=True)
        fr_mat = np.load(os.path.join(ar_p, f'{outcome}.npy'))
        return fr_mat, unit_ids

    def plot_heatmaps(ar, cv, sort_fn=hs.SpikeTrains.sorted_spike_matrix_0):
        axs_ = get_axs(cv)
        iter_axs = iter(axs_)
        mat, unit_ids = get_heatmap_matrix(ar)
        mat = hs.normalise_along_dim(mat)
        _, idx = sort_fn(mat, sort_by='max')
        for i, (epoch, edges) in enumerate(zip(conf_dict['bin_max'].keys(), conf_dict['edges'].values())):
            title = True if area == 'mo-right' else False
            ep_mat = mat[:, edges[0]:edges[1] + 1]
            with mpl.rc_context(hs.no_xaxis if area != 'stn-right' else hs.rc0):
                ax = hs.get_ax(next(iter_axs))
                if mat.shape[0]:
                    yticks = [0.5, conf_dict['unit_counts'][area]-0.5] if not i else []
                    cbar_ax = None
                    if i == len(conf_dict['bin_max'].keys()) - 1 and area == 'str-right':
                        spec = 'Normalised ' if normalise else ''
                        clabel = f'{spec} firing rate (A.U.)'
                        cbar_ax = hs.get_ax(next(iter_axs))
                    else:
                        clabel = None
                    if title:
                        ax.set_title(f'{approximate_var_label(epoch).capitalize()}',
                                     fontsize=mpl.rcParams['axes.titlesize'], style='italic')
                    if area == 'stn-right':
                        if conf_dict['pres'][epoch] == 0:
                            xticks = [0, st.posts[epoch]/2, st.posts[epoch]]
                            xticklabs = [0, st.e_sp_df[epoch]['post'].iloc[0] / 2000,
                                         st.e_sp_df[epoch]['post'].iloc[0] / 1000]
                        else:
                            xticks = [0, st.zero_ps[epoch], st.zero_ps[epoch] + st.posts[epoch]]
                            xticklabs = [np.negative(st.e_sp_df[epoch]['pre'].iloc[0] / 1000), 0,
                                         st.e_sp_df[epoch]['post'].iloc[0] / 1000]
                    else:
                        xticks, xticklabs = [], []
                    st.plot_heatmap_1(mat=ep_mat[idx],
                                      yticks_on=True,
                                      yticks=yticks,
                                      yticklabs_on=True if not i else False,
                                      yticklabels=[conf_dict['unit_counts'][area], str(0)] if not i else [],
                                      xticks_on=True,
                                      xticks=xticks,
                                      xticklabels=xticklabs,
                                      xticklabs_on=True,
                                      clabel=clabel,
                                      cbar_ax=cbar_ax,
                                      ax=ax,
                                      palette='rocket')
                    ax.axvline(st.zero_ps[epoch], alpha=0.5, lw=1, color=[0, 0, 0])
                if area == 'stn-right':
                    png_path = os.path.join(PNG_DIR, epoch + '.png')
                    y = ax.get_ylim()[0] - .24 * np.diff(ax.get_ylim())
                    hs.add_png_to_plot(ax=ax, img_path=png_path, x=st.zero_ps[epoch], y=y, zoom=.06)
                if not i:
                    ax.set_ylabel(fw.region_labels()[area], color=fw.region_colours()[area],
                                  fontsize=mpl.rcParams['axes.labelsize']+2)
                    axx_10th = np.diff(ax.get_xlim())[0] * 0.1
                    if area == 'mo-right':
                        ax.text(-1.5 * axx_10th, conf_dict['unit_counts'][area]/2, 'Units', rotation=90, va='center',
                                fontsize=mpl.rcParams['axes.labelsize'])
                    if area == 'stn-right':
                        ax.set_xlabel('Time (s)')

    c = list(conf_dict['unit_counts'].values())
    vert_cvs = [x for pair in zip(c, [45]*len(c)) for x in pair]
    cvs = iter(hs.gridspec(len(vert_cvs), 1, height_ratios=vert_cvs, where=where, wspace=.1))
    for area in conf_dict['areas']:
        plot_heatmaps(area, next(cvs))
        with mpl.rc_context(hs.no_axes):
            next(cvs)

# Daily unit yield (units retained after the firing-rate criterion; a unit is counted once per subject-day even if it
# was recorded across several blocks) and the combined yield of each area pair on each day on which both of its areas
# were recorded (the population that enters that pair's CCA). Used by panel d and the yield analysis below the figure
yield_df = sp_df[['block', 'area', 'unit_lab']].copy()
yield_df['subj_day'] = [hs.get_subj_date(b) for b in yield_df['block']]
daily_yield = yield_df.groupby(['area', 'subj_day'])['unit_lab'].nunique().rename('n_units').reset_index()
yield_wide = daily_yield.pivot(index='subj_day', columns='area', values='n_units')
frontal = ('mo', 'acc')  # the remaining regions (str, stn, vp) are basal ganglia
pair_rows = []
for a1, a2 in itertools.combinations(st.get_sorted_areas(yield_wide.columns), 2):
    n_front = sum(a.split('-')[0] in frontal for a in (a1, a2))
    for day, (n1, n2) in yield_wide[[a1, a2]].dropna().iterrows():
        pair_rows.append(dict(subj_day=day, subject=day.split('_')[0], pair=f'{fw.region_labels()[a1]}-{fw.region_labels()[a2]}',
                              circuit={2: 'Fr-Fr', 1: 'Fr-BG', 0: 'BG-BG'}[n_front], yield_=n1 + n2))
pair_yield_df = pd.DataFrame(pair_rows)
pair_yield_df['log_yield'] = np.log(pair_yield_df['yield_'])
yield_circuit_colour = {'Fr-Fr': '#1864AB', 'Fr-BG': '#862E9C', 'BG-BG': '#C2255C'}  # as in Fig. S4


def plot_pair_yield_boxplots(where=(12, 2.5), df=pair_yield_df, width_ratios=(.7, .3), wspace=.16, jitter=.18):
    """
    Box plots of the combined unit yield per recording day (units in both areas of a pair, days on which both areas were
    recorded). Left: every area pair, grouped by circuit class and coloured by it (width_ratios[0] of the panel).
    Right: the same days pooled by circuit class (width_ratios[1]). Boxes show the median, IQR and 1.5 x IQR whiskers;
    the dots are the individual days.
    """
    circuits = list(yield_circuit_colour)
    pair_circuit = df.drop_duplicates('pair').set_index('pair')['circuit']
    pairs = sorted(pair_circuit.index, key=lambda pr: circuits.index(pair_circuit[pr]))  # stable within a class
    gs = hs.gridspec(1, 2, where=where, width_ratios=list(width_ratios), wspace=wspace)
    rng = np.random.default_rng(0)

    def draw(ax, groups, labels, colours):
        bp = ax.boxplot(groups, positions=np.arange(len(groups)), widths=.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', lw=1), whiskerprops=dict(lw=.8), capprops=dict(lw=.8),
                        boxprops=dict(lw=.8))
        for patch, g, c, x in zip(bp['boxes'], groups, colours, range(len(groups))):
            patch.set(facecolor=mpl.colors.to_rgba(c, .35), edgecolor=c)
            ax.scatter(x + rng.uniform(-jitter, jitter, len(g)), g, s=6, facecolors=mpl.colors.to_rgba(c, .35),
                       edgecolors='black', linewidths=.3, zorder=3)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlim(-.6, len(groups) - .4)
        ax.grid(False)

    with mpl.rc_context(hs.rc0):
        ax_l = hs.get_ax(gs[0])
        draw(ax_l, [df.loc[df['pair'] == pr, 'yield_'].to_numpy() for pr in pairs], pairs,
             [yield_circuit_colour[pair_circuit[pr]] for pr in pairs])
        ax_l.set_ylim(0, df['yield_'].max() * 1.05)
        ax_l.set_ylabel('Combined units per day')
        ax_l.set_xlabel('Area pair')
        ax_r = gs.figure.add_subplot(gs[1], sharey=ax_l)
        draw(ax_r, [df.loc[df['circuit'] == c, 'yield_'].to_numpy() for c in circuits], circuits,
             [yield_circuit_colour[c] for c in circuits])
        ax_r.tick_params(labelleft=True)
        ax_r.set_ylabel('Combined units per session')
        ax_r.set_xlabel('Circuit class')


# Pull it all together
s1_w, s1_h = hs.config.landscape_A(4)
s1_d_frac = .32  # height of the bottom row (panel d) relative to the rows above, which keep their original size
rows = iter(hs.gridspec(2, 1, hspace=.22, height_ratios=[1, s1_d_frac], where=(s1_w, s1_h * (1 + s1_d_frac))))
cvs = iter(hs.gridspec(1, 2, hspace=.3, wspace=.4, width_ratios=[.6, .4], where=next(rows)))
hs.label_panels(plot_heatmaps_grid, start='a', skip=0, count=1)(where=next(cvs))
cv1 = iter(hs.gridspec(2, 1, height_ratios=[.5, .5], wspace=.3, where=next(cvs)))
with mpl.rc_context(hs.rc0):
    hs.label_panels(plot_per_session_neuron_counts_matplotlib, start='b', skip=0, count=1)(
        kde_smooth=1,
        kde_width_scale=0.7,
        where=next(cv1)
    )
with mpl.rc_context(hs.rc0):
    hs.label_panels(plot_unit_area_pair_stats_frequency, start='c', skip=0, count=1)(
        unit_stats, where=next(cv1)
    )
hs.label_panels(plot_pair_yield_boxplots, start='d', skip=0, count=1)(where=next(rows))
hs.savefig(f"{out_root}_fig_S1.png", open=OPEN_FIGS)
hs.savefig(f"{out_root}_fig_S1.svg", dpi=600)

# Number of sessions (blocks) available per animal, and the subject-days they fall on.
for label, df_ in [('recorded (all spike_times)', st.spike_times), (f"after fr_crit = {conf_dict['fr_crit']} Hz", sp_df)]:
    blocks = pd.Series(df_['block'].unique())
    block_counts = pd.DataFrame({'block': blocks, 'subject': blocks.str.split('_').str[0],
                                 'subj_day': [hs.get_subj_date(b) for b in blocks]})
    summary = block_counts.groupby('subject').agg(n_blocks=('block', 'nunique'), n_subject_days=('subj_day', 'nunique'))
    summary.loc['total'] = summary.sum()
    print(f'\nSessions available per animal - {label}:\n{summary.to_string()}')

# Cell yield per recording day by region (units retained after the firing-rate criterion). A unit is counted once
# per subject-day even if it was recorded across several blocks of that day; days on which a region yielded no units
# do not contribute to that region's average (n_days is reported alongside)
daily_total = daily_yield.groupby('subj_day')['n_units'].sum()
region_yield = daily_yield.groupby('area')['n_units'].agg(
    median='median', q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75), n_days='count')
region_yield = region_yield.loc[st.get_sorted_areas(region_yield.index)]
region_yield.index = [fw.region_labels()[a] for a in region_yield.index]
tot_q1, tot_q3 = daily_total.quantile([.25, .75])
print(f'\nCell yield per recording day by region (median, IQR):\n{region_yield.round(1).to_string()}')
print(f'All regions combined: {daily_total.median():.1f} [{tot_q1:.1f}, {tot_q3:.1f}] units/day over {len(daily_total)} days')
print('Summary sentence: Across recording days, the median [IQR] yield was '
      + ', '.join(f'{r["median"]:.1f} [{r["q1"]:.1f}, {r["q3"]:.1f}] units in {name}' for name, r in region_yield.iterrows())
      + f' (total {daily_total.median():.1f} [{tot_q1:.1f}, {tot_q3:.1f}] units per day; n = {len(daily_total)} days).')

# Is the combined daily unit yield different between area pairs? The yield of a pair on a day is the number of units
# in its two areas combined (the population that enters that pair's CCA), for days on which both areas were recorded.
# Days contribute to several pairs and cluster within animals, so the pairs are compared with a linear mixed model of
# log yield (effects are ratios) with a random intercept per animal and a day-within-animal variance component. Pair
# effect: likelihood-ratio test (ML fits) against the intercept-only model; post-hoc: Bonferroni-corrected Wald
# contrasts from the REML fit. Repeated for the circuit class (Fr-Fr / Fr-BG / BG-BG) and, as an assumption-light
# check, a Friedman test over the days on which every pair was recorded.
import re
import warnings
from scipy.stats import friedmanchisquare

def fit_yield_lmm(df, rhs, reml):
    """log_yield ~ rhs with a random intercept per animal and a variance component for each day (nested in animal)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return smf.mixedlm(f'log_yield ~ {rhs}', data=df, groups=df['subject'], re_formula='1',
                           vc_formula={'day': '0 + C(subj_day)'}).fit(reml=reml, method='lbfgs')


def yield_factor_test(df, factor):
    """LRT of `factor` (ML) and Bonferroni-corrected pairwise Wald contrasts of its level means (REML)."""
    full = fit_yield_lmm(df, f'0 + C({factor})', reml=False)
    null = fit_yield_lmm(df, '1', reml=False)
    n_levels = df[factor].nunique()
    lrt = 2 * (full.llf - null.llf)
    lrt_p = chi2.sf(lrt, n_levels - 1)
    reml_fit = fit_yield_lmm(df, f'0 + C({factor})', reml=True)
    fe = reml_fit.fe_params
    cov = reml_fit.cov_params().loc[fe.index, fe.index]
    level = {name: re.search(r'\[(.*)\]', name).group(1) for name in fe.index}
    contrasts = []
    for n1, n2 in itertools.combinations(fe.index, 2):
        diff = fe[n1] - fe[n2]
        se = np.sqrt(cov.loc[n1, n1] + cov.loc[n2, n2] - 2 * cov.loc[n1, n2])
        p = 2 * norm.sf(abs(diff / se))
        contrasts.append(dict(a=level[n1], b=level[n2], ratio=np.exp(diff), lo=np.exp(diff - 1.96 * se),
                              hi=np.exp(diff + 1.96 * se), p=p))
    contrasts = pd.DataFrame(contrasts)
    contrasts['p_bonf'] = np.minimum(contrasts['p'] * len(contrasts), 1)
    return lrt, n_levels - 1, lrt_p, contrasts


for factor in ('pair', 'circuit'):
    lrt, lrt_df, lrt_p, contrasts = yield_factor_test(pair_yield_df, factor)
    summary_ = pair_yield_df.groupby(factor)['yield_'].agg(
        n_days='count', median='median', q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75))
    print(f'\nCombined daily unit yield by {factor} ({pair_yield_df.subject.nunique()} animals):\n{summary_.round(1).to_string()}')
    print(f'Likelihood-ratio test of {factor} on log yield: chi2({lrt_df}) = {lrt:.2f}, p = {lrt_p:.3g}')
    sig_ = contrasts[contrasts['p_bonf'] < .05]
    print(f'Bonferroni-corrected pairwise contrasts ({len(contrasts)} comparisons), yield ratio a/b [95% CI]: '
          + (f'{len(sig_)} significant' if len(sig_) else 'none significant'))
    for _, r in (sig_ if len(sig_) else contrasts.nsmallest(3, 'p')).iterrows():
        print(f'  {r["a"]} / {r["b"]}: {r["ratio"]:.2f} [{r["lo"]:.2f}, {r["hi"]:.2f}], p_bonf = {r["p_bonf"]:.3g}')

complete_days = pair_yield_df.pivot(index='subj_day', columns='pair', values='yield_').dropna()
if len(complete_days) >= 3:
    fr_chi, fr_p = friedmanchisquare(*[complete_days[c] for c in complete_days.columns])
    print(f'\nFriedman test over the {len(complete_days)} days on which all {complete_days.shape[1]} pairs were recorded: '
          f'chi2({complete_days.shape[1] - 1}) = {fr_chi:.2f}, p = {fr_p:.3g}')
else:
    print(f'\nFriedman test skipped: only {len(complete_days)} days have all pairs recorded')

del sp_df


#%%
# Supplemental figure 2: reliability of CCA stats across animals

st_src_id = 382702
st = load_SpikeStats(src=st_src_id)
st.canonical_corr.dropna(subset=['corr_cv_perm_avg'], inplace=True)
conf_dict['bin_max'], conf_dict['col_n'], conf_dict['edges'] = st.get_bin_max_in_epochs(st.canonical_corr)
conf_dict['subj_days'] = np.unique(st.canonical_corr.subject_day.unique())
conf_dict['subjects'] = sorted(set(s.split('_')[0] for s in conf_dict['subj_days']))  # sorted: a set's order differs between runs
st.canonical_corr['subject'] = [_.split('_')[0] for _ in st.canonical_corr.subject_day]
st.canonical_corr['∆rho'] = (np.abs(st.canonical_corr['corr_cv']) -
                               np.abs(st.canonical_corr['corr_cv_perm_avg']))
conf_dict['epochs'] = st.get_sorted_epochs(st.canonical_corr.epoch.unique())


# Examine the between-and within-animal consistency of the first CCA rho
def icc_from_mixedlm(df):
    """
    Random-intercept mixed model: y ~ 1 + (1|id)
    ICC = var_between / var_within
    If ICC ≪ 1, then within-animal variance dominates, meaning animals are similar
    "Here, we define consistency as small between-animal variability relative to within-animal variability.
    Accordingly, lower ICC values indicate greater cross-animal agreement.”
    """

    model = mixedlm("y ~ 1", df, groups=df['id'])
    result = model.fit()

    # Variance components
    var_between = result.cov_re.iloc[0, 0]
    var_within = result.scale
    icc = var_between / var_within
    return max(icc, 0.0), var_between, var_within

def bootstrap_icc_mixedlm(y, ids, B=1000, seed=0):
    """
    Bootstrap CI for ICC via mixed model; handles unbalanced replicates and string IDs.
    """
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({'y': np.asarray(y), 'id': np.asarray(ids, dtype=object)})
    n_dropped = base['y'].isna().sum()
    if n_dropped:
        print(f'Dropping {n_dropped} NaN rows in y before fitting mixedlm')
        base = base.dropna(subset=['y']).reset_index(drop=True)

    # point estimate
    icc_hat, vb_hat, vw_hat = icc_from_mixedlm(base)

    icc_vals = []
    animals = base['id'].unique()

    for _ in range(B):
        rows = []
        for a in animals:
            sub = base[base['id'] == a]
            res = sub.sample(n=len(sub), replace=True, random_state=rng.integers(0, 2**31-1))
            rows.append(res)
        boot = pd.concat(rows, ignore_index=True)
        try:
            icc_b, _, _ = icc_from_mixedlm(boot)
        except Exception:
            icc_b = np.nan
        icc_vals.append(icc_b)

    icc_vals = np.array(icc_vals, float)
    ci = np.nanpercentile(icc_vals, [2.5, 97.5])
    return {'icc': icc_hat, 'ci95': ci, 'icc_boot': icc_vals, 'var_between': vb_hat, 'var_within': vw_hat}


def icc_to_long_df(res_dict: dict, label: str, df=pd.DataFrame()):
    # bootstrap values
    boot_df = pd.DataFrame({
        "source": "bootstrap",
        "condition": label,
        "value": res_dict["icc_boot"]
    })
    # point estimate
    point_df = pd.DataFrame({
        "source": "point",
        "condition": label,
        "value": [res_dict["icc"]]
    })
    if df.empty:
        return pd.concat([point_df, boot_df], ignore_index=True)
    else:
        return pd.concat([df, pd.concat([point_df, boot_df], ignore_index=True)], ignore_index=True)

def compute_icc_across_area_pairs(cca_df=st.canonical_corr, f_root=out_root, recompute=RECOMPUTE_ICC):
    # The 5,000-resample bootstrap per area pair is slow, so its results are cached (output/203265_icc_*) and reused:
    # set RECOMPUTE_ICC = True (top of the script) to repeat it (seeded, so the estimates are reproducible).
    if not recompute:
        try:
            return pd.read_csv(f'{f_root}_icc_df.csv', index_col=0), hs.load_df(f'{f_root}_icc_raw_df')
        except FileNotFoundError:
            pass
    icc_ci_df = pd.DataFrame(columns=['icc', 'ci_low', 'ci_high'], index=cca_df.area_pair.unique())
    res, res_df = {}, pd.DataFrame()
    for area_pair in cca_df.area_pair.unique():
        df = cca_df.query(f'area_pair=="{area_pair}" and rank==0').reset_index(drop=True)
        df['subject'] = [sd.split('_')[0] for sd in df['subject_day']]
        res[area_pair] = bootstrap_icc_mixedlm(df['corr_cv'], df['subject'], B=5000, seed=0)
        res_ = res[area_pair]
        res_df = icc_to_long_df(res[area_pair], label=area_pair, df=res_df)
        icc_ci_df.loc[area_pair] = dict(icc=res_['icc'], ci_low=res_['ci95'][0], ci_high=res_['ci95'][1])
        icc_ci_df.to_csv(f'{f_root}_icc_df.csv')
        print(f"{area_pair}: ICC = {res_['icc']:.3f}, 95% CI = [{res_['ci95'][0]:.3f}, {res_['ci95'][1]:.3f}]")
    hs.save_df(res_df, f'{f_root}_icc_raw_df')
    return icc_ci_df, res_df

icc_ci_df, res_df = compute_icc_across_area_pairs()

def plot_canonical_corr_network(cca_df, inp_cvs=None, var='corr_cv', perm_var='corr_cv_perm_avg', edge_var='corr_cv',
                                ylabel=None, plot_ns_edges=False, annotate_edges=False):
    where = np.divide(hs.config.landscape_A(4), [1, 1]) if inp_cvs is None else inp_cvs
    cvs = iter(hs.gridspec(1, len(conf_dict['bin_max']), wspace=.1, where=where))
    edge_params = dict()
    for epoch in conf_dict['epochs']:
        # edges and values (width encodes value)
        edge_params[epoch] = dict(edge_labels=[], edge_values=[], edge_pvals=[])
        for area_pair in cca_df['area_pair'].unique(): # the default variable label is used here
            print(f'Plotting {epoch} {area_pair} {var}')
            edge_params[epoch]['edge_labels'].append(tuple([str(fw.region_labels()[area_pair.split("_")[0]]),
                                                     str(fw.region_labels()[area_pair.split("_")[1]])]))
            tbl = cca_df.query(f'epoch=="{epoch}" & area_pair=="{area_pair}"')
            if tbl.empty:
                area_pair = '_'.join(area_pair.split("_")[::-1])
                tbl = cca_df.query(f'epoch=="{epoch}" & area_pair=="{area_pair}"')
            real_corrs = tbl[var]
            perm_corrs = tbl[perm_var]
            try:
                assert not sum(np.isnan(real_corrs) & np.isnan(perm_corrs))
            except AssertionError:
                print(f'NaN in real or permuted correlations!')
                print(pd.DataFrame(np.vstack([real_corrs, perm_corrs]).transpose(), columns=['Real', 'Permuted']))
                sys.exit()
            edge_vals = real_corrs if edge_var == var else tbl[edge_var]
            edge_params[epoch]['edge_values'].append(hs.average_rhos(edge_vals))
            p = 0  # no statistical testing here: every edge is depicted as significant
            edge_params[epoch]['edge_pvals'].append(p)

    nodes_with_finite_edges = set()

    for epoch in conf_dict['epochs']:
        labs = edge_params[epoch]["edge_labels"]
        vals = np.asarray(edge_params[epoch]["edge_values"], dtype=float)

        for (u, v), val in zip(labs, vals):
            if np.isfinite(val):
                nodes_with_finite_edges.add(u)
                nodes_with_finite_edges.add(v)

    all_edges = np.array([edge_params[k]['edge_values'] for k in conf_dict['epochs']])
    global_vmin, global_vmax = np.quantile(all_edges, [0.01, 0.99])

    for epoch in conf_dict['epochs']:
        node_labels = {n: n for n in fw.node_plot_coords()}
        png_path = os.path.join(PNG_DIR, epoch + '.png')
        st.plot_node_graph_1(coords=fw.node_plot_coords(),
                             edge_params=edge_params[epoch],
                             thresh=conf_dict['thresh'],
                             node_labels=node_labels,
                             rc_context=hs.no_axes,
                             node_size=180,
                             label_fontsize=4,
                             where=next(cvs),
                             background_png=os.path.join(PNG_DIR, 'brain_slice.png') if epoch == 'PRE' else None,
                             ylabel=ylabel if epoch == 'PRE' else None,
                             epoch_png_path=png_path,
                             title=var_labels[epoch],
                             plot_ns_edges=plot_ns_edges,
                             annotate_edges=annotate_edges,
                             nodes_with_finite_edges=nodes_with_finite_edges,
                             global_edge_range=(global_vmin, global_vmax),
                             scalebar_title=r"∆$\rho$ (edges)",
                             show_scalebar=False)

def plot_individual_animal_ccas(where):
    # Plot node diagram individually for each animal
    for i, subject in enumerate(np.unique(st.canonical_corr['subject'])):
        plot_canonical_corr_network(
            cca_df=st.canonical_corr.loc[((st.canonical_corr.subject_day.str.contains(subject)) &
                                          (st.canonical_corr['rank'] == 0)), :], inp_cvs=next(where),
            ylabel=subject, plot_ns_edges=True, annotate_edges=True)

def plot_icc_eb_and_hist(icc: pd.DataFrame, where):

    def plot_dist(dist, bins, ax=None, lw=0.5, col=(0, 0, 0), alpha=0.7, offs=0, max_width=.7, plot_outline=False):
        if np.ptp(dist) == 0:
            # No spread across bootstrap resamples (e.g. only one subject contributed to this area
            # pair, so between- and within-animal variance are both 0/degenerate) - a histogram has
            # no bins to show, so mark the constant value with a flat bar instead.
            ax.plot([offs, offs + max_width], [dist[0], dist[0]], color=col, linewidth=2, alpha=alpha,
                    solid_capstyle='butt')
            return
        y, x = np.histogram(dist, bins, density=True)
        if max_width is not None and y.max() > 0:
            y = y / y.max() * max_width
        # outline
        if plot_outline:
            ax.plot(y + offs, x[:-1], linewidth=lw, color=col, label=label)
        # fill between vertical edges
        ax.fill_betweenx(x[:-1], offs, y + offs, alpha=alpha, color=col)

    ymax = np.round(np.max(icc.value.values), 2)
    with mpl.rc_context(hs.rc0):
        ax = hs.get_ax(where)
        edge_labels = []
        n_subjects = []
        for x, area_pair in enumerate(st.canonical_corr.area_pair.unique()):
            print(f'Plotting {area_pair} icc statistics')
            edge_label = f'{str(fw.region_labels()[area_pair.split("_")[0]])}&' \
                         f'{str(fw.region_labels()[area_pair.split("_")[1]])}'
            edge_labels.append(edge_label)
            n_subjects.append(len(set(
                sd.split('_')[0] for sd in
                st.canonical_corr.query(f'area_pair=="{area_pair}"')['subject'].unique()
            )))
            colour = hs.mix_colours(fw.region_colours()[edge_label.split('&')[0]],
                                    fw.region_colours()[edge_label.split('&')[1]])
            eb_defs = {i: hs.config.plot_kwargs[i] for i in hs.config.err_bar_defs}
            eb_defs['linewidth'] = 1
            eb_defs['alpha'] = 1
            boot_dist = icc.query(f'condition=="{area_pair}" and source == "bootstrap"').value.values
            if np.all(boot_dist == 1):
                print(f'Skipping {area_pair} - bootstrap ICC distribution is degenerate (all ones)')
                continue
            y = np.nanmedian(boot_dist)
            ci_low, ci_high = np.nanpercentile(boot_dist, [2.5, 97.5])
            ax.scatter(x, y, edgecolor=[0, 0, 0], zorder=3, facecolor=[0, 0, 0], s=7)
            _, _, cols = ax.errorbar([x], [np.percentile(boot_dist, 50)],
                                     yerr=np.array([[y-ci_low],
                                                     [ci_high-y]]),
                                     zorder=1,
                                     capsize=0,
                                     ecolor=[0, 0, 0],
                                     **eb_defs)
            plt.setp(cols[0], capstyle="round")
            plot_dist(boot_dist, bins=np.arange(boot_dist.min(), boot_dist.max(), .025), ax=ax,
                      offs=x+0.05, col=colour)
        ax.set_xticks(np.arange(0, x+1))
        ax.set_xlim([0-.7, x+.7])
        ax.set_yticks(np.arange(0, ymax+.1, .25))
        ax.set_ylim([0, 1.25])
        ax.set_xticklabels(edge_labels, rotation=45, ha='right')
        ax.set_ylabel(r'CCA $\rho_1$ agreeability')
        ax.axhline(y=1, color='k', linestyle='-.', linewidth=.1)
        for x_, n in enumerate(n_subjects):
            ax.annotate(f'n={n}', xy=(x_, 1), xycoords='data',
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=mpl.rcParams['xtick.labelsize'] - 1,
                        clip_on=False)

# Summarise between-animal reliability / agreeability in CCA estimates
def plot_canonical_corr_bars_across_epoch(cca_df=st.canonical_corr,
                                          var='corr_cv',
                                          ylim=.6,
                                          where=None):
    if where is None:
        where = np.divide(hs.config.landscape_A(4), [6, 6])
    cvs = iter(hs.gridspec(1, 1, wspace=.3, where=where))
    with mpl.rc_context(hs.rc0):
        ax = hs.get_ax(next(cvs))
        edge_labels = []
        for x, area_pair in enumerate(st.canonical_corr.area_pair.unique()):
            print(f'Plotting {area_pair} {var}')
            edge_label = f'{str(fw.region_labels()[area_pair.split("_")[0]])}&' \
                         f'{str(fw.region_labels()[area_pair.split("_")[1]])}'
            edge_labels.append(edge_label)
            tbl = cca_df.query(f'area_pair=="{area_pair}"')
            colour = hs.mix_colours(fw.region_colours()[edge_label.split('&')[0]],
                                    fw.region_colours()[edge_label.split('&')[1]])
            if tbl.empty:
                area_pair = '_'.join(area_pair.split("_")[::-1])
                tbl = cca_df.query(f'area_pair=="{area_pair}"')
            # Plot the average across animals across epochs
            ax.bar(x, tbl.groupby(['subject'])[var].mean().mean(), color=colour, alpha=.7,
                   **{i: hs.config.plot_kwargs[i] for i in hs.config.bar_defs})
            for subj in tbl.subject.unique():
                ax.scatter(x, hs.average_rhos(tbl.query(f'subject=="{subj}"')[var]), edgecolor=[0, 0, 0],
                           zorder=3, facecolor=colour,
                           **{i: hs.config.plot_kwargs[i] for i in hs.config.scat_defs})
        ax.set_xticks(np.arange(0, x+1))
        ax.set_xlim([0-.7, x+.7])
        ax.set_yticks(np.arange(0, ylim+.1, 0.1))
        ax.set_ylim([0, ylim])
        ax.set_xticklabels(edge_labels, rotation=45, ha='right')
        ax.set_ylabel(r'Cross-validated $\rho_1$')



# Pull it all together
cvs = iter(hs.gridspec(2, 1, hspace=.3, wspace=.4, height_ratios=[.8, .2], where=hs.config.landscape_A(4)))
hs.label_panels(plot_individual_animal_ccas, start='a', skip=0, count=1)(
    iter(hs.gridspec(st.canonical_corr['subject'].nunique(), 1, wspace=.1, where=next(cvs)))
)
cv1 = iter(hs.gridspec(1, 2, wspace=.3, where=next(cvs)))
hs.label_panels(plot_icc_eb_and_hist, start='b', skip=0, count=1)(
    icc=hs.load_df(f'{out_root}_icc_raw_df'),
    where=next(cv1)
)
hs.label_panels(plot_canonical_corr_bars_across_epoch, start='c', skip=0, count=1)(
    cca_df=st.canonical_corr.loc[(st.canonical_corr['rank'] == 0), :], where=next(cv1)
)
hs.savefig(f"{out_root}_fig_S2.png", open=OPEN_FIGS)
hs.savefig(f"{out_root}_fig_S2.svg", dpi=600)

# Table 1 (related to fig. S2a): the edge values annotated on the per-animal network diagrams in Fig. S2a, as a table
# (rows: animal > epoch; columns: area pair) that can be pasted straight into MS Word.
def build_s1_edge_table(cca_df, var='corr_cv', rank=0, decimals=2):
    """
    Fisher-z-averaged cross-validated rho (hs.average_rhos, as in plot_canonical_corr_network) of the
    leading canonical dimension for every animal x epoch x area pair, mirroring Fig. S2a's edge labels.
    """
    cca_df = cca_df.loc[cca_df['rank'] == rank]
    pair_cols = {}  # area_pair as stored -> column label
    for ap in sorted(cca_df['area_pair'].unique(),
                     key=lambda a: [conf_dict['areas'].index(f'{r}-right') for r in a.split('_')]):
        pair_cols[ap] = '–'.join(fw.region_labels()[r] for r in ap.split('_'))
    rows = {}
    for subject in sorted(cca_df['subject'].unique()):
        for epoch in conf_dict['epochs']:
            row = {}
            for ap, lab in pair_cols.items():
                tbl = cca_df.query(f'subject=="{subject}" & epoch=="{epoch}" & area_pair=="{ap}"')
                vals = tbl[var].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                row[lab] = round(float(hs.average_rhos(vals)), decimals) if vals.size else np.nan
            rows[(subject, approximate_var_label(epoch).capitalize())] = row
    table = pd.DataFrame.from_dict(rows, orient='index')
    table.index = pd.MultiIndex.from_tuples(table.index, names=['Animal', 'Epoch'])
    return table


def table_to_html(table, decimals=2):
    """HTML table with the animal cell merged (rowspan) down its epochs, so Word keeps the hierarchy."""
    th = 'style="border-top:1px solid #000;border-bottom:1px solid #000;padding:2px 8px;text-align:center;"'
    out = ['<table style="border-collapse:collapse;font-family:Arial;font-size:9pt;">', '<tr>',
           f'<th {th}>Animal</th><th {th}>Epoch</th>']
    out += [f'<th {th}>{c}</th>' for c in table.columns] + ['</tr>']
    n_ep = table.groupby(level=0).size()
    for i, ((subject, epoch), row) in enumerate(table.iterrows()):
        out.append('<tr>')
        if i == 0 or subject != table.index[i - 1][0]:
            border = 'border-top:1px solid #888;' if i else ''
            out.append(f'<td rowspan="{n_ep[subject]}" style="padding:2px 8px;text-align:center;'
                       f'vertical-align:middle;{border}"><b>{subject}</b></td>')
        border = 'border-top:1px solid #888;' if (i and subject != table.index[i - 1][0]) else ''
        out.append(f'<td style="padding:2px 8px;text-align:left;{border}">{epoch}</td>')
        out += [f'<td style="padding:2px 8px;text-align:center;{border}">'
                f'{"–" if pd.isna(v) else f"{v:.{decimals}f}"}</td>' for v in row]
        out.append('</tr>')
    out.append(f'<tr><td colspan="{table.shape[1] + 2}" style="border-top:1px solid #000;"></td></tr></table>')
    return '\n'.join(out)


s1_table = build_s1_edge_table(st.canonical_corr)
print(s1_table.to_string(na_rep='–'))
s1_html = table_to_html(s1_table)
s1_html_path = f'{out_root}_table_S1.html'
with open(s1_html_path, 'w', encoding='utf-8') as f:
    f.write(f'<html><body>{s1_html}</body></html>')
s1_table.to_csv(f'{out_root}_table_s1.csv')

#%%
# Supplemental figure 3: axes of the replacing task variables relative to earlier reward coding (a; project 781039),
# running-speed profiles across task epochs (b), accept-vs-reject decoder comparison (∆r) in the corridor epochs
# (c; project 258066), and accept-vs-reject variance of the true task variables (d; formerly Fig. S4, project 258066)

ar_src_id = 258066         # oh_cue_run_outcome_cca_1_bin_v6_task_decoder (accept vs. reject) - source of former Fig. S4

st_ar = load_SpikeStats(src=ar_src_id)  # accept-vs-reject decoder (cca_component_decoder_2)


def collapse_beta_df_to_models(beta_df, drop_cols=("beta", "cc_beta_norm", "n_trials", "rank", "predictor", "task",
                                                   "subspace", "score_metric", "n_cc")):
    """One row per fitted model: drop the coefficient-specific columns of a beta-level regression table."""
    keep_cols = [c for c in beta_df.columns if c not in drop_cols]
    return beta_df[keep_cols].drop_duplicates().reset_index(drop=True)


# ∆r (accept - reject) of the group-aware, trial-balanced held-out decoding scores, built as in 258066.py: per
# subject-day and (epoch, area pair) the difference in held-out correlation between accepted and rejected trials,
# then a two-sided Wilcoxon test of that difference against 0 across subject-days (the sign of accept - reject is
# itself part of the question), uncorrected, p < conf_dict['thresh']
task_vars = ["trial_order", "effort_lev", "trial_count", "velocity"]
beta_df = st_ar.regression.query("rank >= 0").copy()  # CC-space coefficients only (drops the Intercept row)
summary_score_df = collapse_beta_df_to_models(beta_df)
summary_score_df = summary_score_df[np.isfinite(summary_score_df['score_accept']) &
                                    np.isfinite(summary_score_df['score_reject'])].copy()
summary_score_df['zero'] = 0.0
summary_score_df['∆corr'] = summary_score_df['score_accept'] - summary_score_df['score_reject']
summary_score_df['rank'] = 0  # get_cca_df_edge_params groups by (epoch, area_pair, rank)
diff_edge_df = pd.DataFrame()
for tv in task_vars:
    res = st_ar.get_cca_df_edge_params(alpha=conf_dict['thresh'],
                                       agg_fn=np.nanmean,
                                       fdr=False,
                                       stat_var='∆corr',
                                       perm_var='zero',  # wilcoxon(x, zeros) == one-sample test of x vs 0
                                       edge_var='∆corr',
                                       best_rank_only=False,
                                       alternative='two-sided',
                                       source_df=summary_score_df.query('outcome == @tv'))
    res['coefficient'] = tv
    diff_edge_df = pd.concat([diff_edge_df, res], ignore_index=True)

class VelDf:
    """
    Running speed per trial, epoch and bin, from the DeepLabCut tracking of the overhead video (data/773894_xy_tables: one
    table of the animal's position, in pixels, per recording stage, 100 Hz). For every trial window of every epoch (the
    windows of conf_dict['qs'], 'pres' and 'posts', cut from the stage's behavioural events in project 958690) the
    position is down-sampled to 10 Hz, converted to cm and differentiated, and the mean speed over the window is stored.
    The reward (trial_order) and effort (effort_lev) level of each trial is merged on from the behaviour table.
    """

    def __init__(self, behaviour_id=958690, dlc_id=773894):
        self.src = {name: load_source_object(behaviour_id, name)
                    for name in ('SpikeStats', 'DinChannels', 'FineBehaviour', 'TrialBehaviour')}
        self.vid_project = BlO.Video(meta=self.src['FineBehaviour'])
        self.xy_tables = f'{get_file_root(dlc_id)}_xy_tables'
        self.vel_df = self.create_velocity_table()
        self.merge_trial_vars()

    def merge_trial_vars(self, trial_vars=('trial_order', 'effort_lev')):
        # Attach each trial's reward level (trial_order) and effort level (effort_lev) from the behavioural table to every
        # epoch/bin row of that trial (keyed on stage + trial_idx)
        beh_df = self.src['TrialBehaviour'].beh_df
        self.vel_df = self.vel_df.merge(beh_df[['stage', 'trial_idx', *trial_vars]], on=['stage', 'trial_idx'], how='left')

    def get_xy_table_dir(self, s_lab):
        return os.path.join(self.xy_tables, f'{s_lab}{self.vid_project.dlc_suffix}.csv')

    def iter_stage(self):
        """Stages with an event-channel ('rise') signal that were recorded in the task box."""
        for _ in self.src['DinChannels'].ol:
            if hasattr(_, 'rise') and _.meta.environment == 'radmac':
                yield _, _.meta.stage

    @staticmethod
    def get_stage_xy(t_dir):
        try:
            return pd.read_csv(t_dir, skiprows=2)
        except FileNotFoundError:
            return None

    def create_empty_dict_1(self):
        n_stages = len(os.listdir(self.xy_tables))
        dtypes = {'str': ['stage', 'epoch'],
                  'int64': ['trial_idx', 'bin'],
                  'float64': ['velocity']}
        d = hs.create_dict_from_types(dtypes)
        return hs.malloc_check(d, dtypes, idx=0, n_fill=int(self.vid_project.n_frames * n_stages)), dtypes, n_stages

    def create_velocity_table(self, estimate=np.mean) -> pd.DataFrame:
        d, dtypes, n_stages = self.create_empty_dict_1()
        idx = 0
        prog_bar = hs.ProgressBar(n_stages, 'selected window of interest xy data', process='pooling')
        for stage_idx, (din, s_lab) in enumerate(self.iter_stage()):
            xy_table = self.get_stage_xy(self.get_xy_table_dir(s_lab))
            if xy_table is None:
                print(f'Could not find stage table for {s_lab}')
            else:
                try:
                    for i_bin, (k, v) in enumerate(conf_dict['qs'].items()):
                        tw = BlO.SpikeStats().TrialWindow(s_lab,
                                                          self.src['SpikeStats'],
                                                          self.src['DinChannels'],
                                                          self.src['FineBehaviour'],
                                                          self.src['TrialBehaviour'])
                        tw_args = dict(sm_query={k: v},
                                       pre=conf_dict['pres'][k],
                                       post=conf_dict['posts'][k],
                                       post_lg_crit=conf_dict['post_lg_crits'][k],
                                       start_aligned=conf_dict['start_aligned'][k])
                        try:
                            for _, (bin_edges, spec, tr_i) in enumerate(getattr(tw, 'yield_epochs')(**tw_args)):
                                try:
                                    v_sl = hs.flatten_list(np.round(np.array([bin_edges]) / (fw.beh_rate / fw.vid_rate)))
                                    v_sl = [i.astype(int) for i in v_sl]
                                    v_sl_rng = np.arange(v_sl[0], v_sl[1])
                                    xy_sel = xy_table.iloc[v_sl_rng]
                                    d['stage'][idx] = s_lab
                                    d['epoch'][idx] = k
                                    d['bin'][idx] = i_bin
                                    d['trial_idx'][idx] = tr_i
                                    d['velocity'][idx] = estimate(self.vid_project.velocity(xy_sel.x, xy_sel.y))
                                    idx += 1
                                except IndexError:
                                    print(f'{s_lab}: indices exceed table of size {xy_table.shape[0]}')
                                    break
                        except (AssertionError, hs.DinRadmacTtlMismatchError):
                            print(f'Skipping {s_lab}')
                except hs.DinRadmacTtlMismatchError:
                    print(f'Skipping {s_lab}')
            prog_bar.print()
        return pd.DataFrame(hs.trim_dict_to_len(d, hs.find_dict_len(d, dtypes)))


# Extracted from the tracking data on the first run (a few minutes) and reused afterwards (output/203265_vel_df_table_v2)
try:
    if RECOMPUTE_VELOCITY:
        raise FileNotFoundError
    vel_df = hs.load_df(f'{out_root}_vel_df_table_v2')
except FileNotFoundError:
    vel_df = VelDf().vel_df
    hs.save_df(vel_df, f'{out_root}_vel_df_table_v2')


def plot_velocity_by_epoch(vel_df=None, where=np.divide(hs.config.landscape_A(4), [1, 4])):
    epochs = st_ar.get_sorted_epochs(vel_df['epoch'].unique())
    cvs = iter(hs.gridspec(1, len(epochs), wspace=.15, where=where))
    with mpl.rc_context(hs.rc0):
        for i, epoch in enumerate(epochs):
            ax = hs.get_ax(next(cvs))
            epoch_df = vel_df.query(f'epoch=="{epoch}"')
            data = epoch_df['velocity'].to_numpy()
            data = data[np.isfinite(data)]  # dropna() alone would let a stray inf through
            # Never let this axis fall back to matplotlib's automatic tick locator: for certain
            # real (vmin, vmax) pairs MaxNLocator._raw_ticks raises IndexError (a matplotlib bug,
            # not something traceable to a specific data pathology we can special-case around) -
            # every other panel in this script sidesteps it by setting ticks explicitly, so do the
            # same here for both axes in every branch.
            xlo = 0  # velocity is non-negative, so anchor every panel's x axis at a true 0
            if data.size == 0:
                xhi_view, ymax = 1, 1
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes,
                       fontsize=mpl.rcParams['axes.labelsize'])
            else:
                data_range = np.ptp(data)
                if data_range <= 0 or not np.isfinite(data_range):
                    # single repeated value: pad on the upper side only, xlo stays a true 0
                    xhi_hist = xhi_view = data[0] + max(data[0] * 0.1, 1e-3)
                    counts, _, _ = ax.hist(data, bins=1, range=(xlo, xhi_hist), color=[.3, .3, .3], alpha=.7)
                else:
                    # bin the histogram out to the 95th percentile (so a long right tail doesn't
                    # wash out the bins), but only show the axis out to the 75th percentile so the
                    # visible window stays on the bulk of the distribution
                    xhi_hist = max(np.percentile(data, 95), xlo + 1e-3)
                    xhi_view = max(np.percentile(data, 95), xlo + 1e-3)
                    counts, _, _ = ax.hist(data, bins=30, range=(xlo, xhi_hist), color=[.3, .3, .3], alpha=.7)
                ax.axvline(np.median(data), color='black', linestyle='--', linewidth=1)
                ymax = counts.max() if counts.size and np.isfinite(counts.max()) and counts.max() > 0 else 1
            ax.set_xlim(xlo, xhi_view)
            ax.set_ylim(0, ymax)
            ax.set_xticks(np.round(np.linspace(xlo, xhi_view, 5), 1))
            ax.set_yticks(np.unique(np.round(np.linspace(0, ymax, 5)).astype(int)))
            ax.set_title(var_labels[epoch], style='italic')
            if i == 0:
                ax.set_xlabel('Velocity (cm/s)')
                ax.set_ylabel('Trial-epoch windows')
            ax.grid(False)

            # inset (top right, currently empty for these left-skewed distributions): mean
            # velocity broken down by each trial's reward (trial_order) and effort (effort_lev)
            # level, to check whether near-zero velocities concentrate in high-effort trials
            # (barrier-scaling trials can fool the top-down DLC tracking into reading near-zero
            # speed, since it only sees the two floor-plane axes and has no height/z perception)
            if {'trial_order', 'effort_lev'}.issubset(epoch_df.columns) and data.size:
                # categorical tick labels/order for reward and effort levels, matching how these
                # columns are actually coded in beh_df (block.var_mapping) - NOT
                # fw.reward_size_keys()/fw.implanted_effort(), which are the labels for a
                # differently-coded variable (rew_offer) and a collapsed 3-level effort grouping
                # respectively, and silently mislabel/drop real trial_order==2/3 and
                # effort_lev==3 trials if used here
                reward_labels = var_mapping['trial_order']
                effort_labels = var_mapping['effort_lev']
                reward_order = [k for k in sorted(reward_labels)
                               if k not in conf_dict['omit'].get('trial_order', [])]
                # drop effort categories with no data anywhere (checked against the full vel_df,
                # not just this epoch, so every epoch panel keeps the same fixed row set) rather
                # than showing an always-empty row
                effort_order = [k for k in sorted(effort_labels) if (vel_df['effort_lev'] == k).any()]

                re_df = epoch_df[~epoch_df['trial_order'].isin(conf_dict['omit'].get('trial_order', []))]
                pivot_re = (re_df.pivot_table(index='effort_lev', columns='trial_order',
                                              values='velocity', aggfunc='median')
                           .reindex(index=effort_order, columns=reward_order))
                if pivot_re.size and pivot_re.notna().any().any():
                    heat_ax = ax.inset_axes([0.52, 0.54, 0.32, 0.42])
                    cbar_ax = ax.inset_axes([0.88, 0.54, 0.05, 0.42])
                    sns.heatmap(pivot_re, mask=pivot_re.isna(), cmap='rocket',
                               annot=True, fmt='.1f',
                               annot_kws={'size': mpl.rcParams['xtick.labelsize'] - 2},
                               linewidths=.3, linecolor='white', ax=heat_ax,
                               cbar=True, cbar_ax=cbar_ax)
                    heat_ax.set_facecolor('#e8e8e8')
                    heat_ax.set_xlabel('Reward', fontsize=mpl.rcParams['xtick.labelsize'] - 1, labelpad=1)
                    heat_ax.set_ylabel('Effort', fontsize=mpl.rcParams['xtick.labelsize'] - 1, labelpad=1)
                    heat_ax.set_xticklabels([reward_labels[k] for k in reward_order], rotation=45, ha='right')
                    heat_ax.set_yticklabels([effort_labels[k] for k in effort_order], rotation=0)
                    heat_ax.tick_params(axis='both', which='both', length=0,
                                       labelsize=mpl.rcParams['xtick.labelsize'] - 2)
                    cbar_ax.tick_params(axis='both', which='both', length=0,
                                       labelsize=mpl.rcParams['xtick.labelsize'] - 2)

# (b) Accept-vs-reject decoder comparison (∆r = accept - reject) in the corridor epochs, ported from 258066.py:
# a multiplex node diagram (every task variable's significant edges on one diagram per epoch) above a bar + swarm
# of the per-session ∆r. Extends the accept-vs-reject panel of Fig. 3 to the corridor epochs.
corridor_epochs = ['CORR1', 'CORR6']


def plot_multiplex_node_graph(coords, edge_df, colour_map, node_labels=None, node_size=180, label_fontsize=4,
                              lw=1.2, where=(2.1, 2), rc_context=hs.no_axes, background_png=None, title=None,
                              ylabel=None, epoch_png_path=None, nodes_with_finite_edges=None,
                              max_offset_fraction=0.06):
    """
    Node diagram of one epoch; `edge_df` is expected pre-filtered to significant (epoch, area_pair, coefficient)
    rows. Edges sharing an area pair are drawn as parallel offset lines, coloured per row by the task variable
    (`edge_df['coefficient']`) via `colour_map`, so a pair can carry more than one task variable's edge at once.
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
                ax.scatter(*coords[n], s=node_size, color='white', edgecolor=fw.region_colours()[n],
                           linewidth=1, zorder=2)
        for n in nodes:
            ax.text(coords[n][0], coords[n][1], node_labels[n], ha='center', va='center',
                    fontsize=label_fontsize, zorder=3)
        if segs:
            ax.add_collection(LineCollection(segs, linewidths=lw, colors=cols, zorder=1))
        ax.set_ylim(.05, .82)  # crop the empty band above the nodes so titles/icons sit close to the diagram
        ax.set_aspect('equal')
        ax.set_anchor('S')  # keep the drawing at the bottom of its slot, next to the swarm below
        if title is not None:
            ax.set_title(title, pad=2)
        if epoch_png_path is not None:
            hs.add_png_to_plot(ax=ax, img_path=epoch_png_path, x=.5, y=1.2, zoom=0.08, coords='axes fraction')
        if ylabel is not None:
            ax.text(0, 0.5, ylabel, rotation=90, va='center')
    return ax


def plot_r2_diff_multiplex_network(edge_df, epochs, where, task_var_order=None, thresh=None):
    """One multiplex node diagram per epoch in `epochs`, with a task-variable colour legend above them."""
    thresh = conf_dict['thresh'] if thresh is None else thresh
    colour_map = fw.venn_prm_colours()
    task_var_order = list(task_vars) if task_var_order is None else task_var_order
    cv = hs.gridspec(2, 1, wspace=.1, where=where, height_ratios=[0.1, 0.9])
    bottom_gs = iter(hs.gridspec(1, len(epochs), wspace=.05, where=cv[1]))

    nodes_with_finite_edges = set()
    for u, v in edge_df[np.isfinite(pd.to_numeric(edge_df['edge_value'], errors='coerce'))]['edge_label']:
        nodes_with_finite_edges.update([u, v])
    sig = edge_df[(pd.to_numeric(edge_df['edge_pval'], errors='coerce') < thresh) &
                  np.isfinite(pd.to_numeric(edge_df['edge_value'], errors='coerce'))].copy()

    legend_labels = [t for t in task_var_order if t in sig['coefficient'].unique()]
    legend_handles = [Line2D([0], [0], color=colour_map.get(lbl, 'black'), lw=3, marker='o', markersize=6,
                             markerfacecolor=colour_map.get(lbl, 'black'), markeredgecolor='black',
                             markeredgewidth=0.8) for lbl in legend_labels]
    with mpl.rc_context(hs.no_axes):
        leg_ax = hs.get_ax(cv[0])
        if legend_handles:
            leg_ax.legend(legend_handles, [approximate_var_label(l).capitalize() for l in legend_labels],
                          loc='upper center', ncol=len(legend_labels), frameon=False, handlelength=2.0,
                          columnspacing=1.8)

    for epoch in epochs:
        plot_multiplex_node_graph(
            coords=fw.node_plot_coords(),
            edge_df=sig.loc[sig.epoch == epoch],
            colour_map=colour_map,
            node_labels={n: n for n in fw.node_plot_coords()},
            node_size=180,
            label_fontsize=4,
            where=next(bottom_gs),
            background_png=None,
            ylabel=None,
            epoch_png_path=os.path.join(PNG_DIR, epoch + '.png'),
            title=var_labels[epoch],
            nodes_with_finite_edges=nodes_with_finite_edges)


def plot_r2_diff_swarm_by_session(edge_df, beta_df, epochs, where, groups=('accept', 'reject'), thresh=None,
                                  group_order=None, bar_width=0.6, dot_size=3, ylabel=None):
    """
    Per-session ∆r = score_<groups[0]> - score_<groups[1]>, pooled per task variable across every area pair that
    edge_df marks significant (edge_pval < thresh) in that epoch: a bar (mean) plus a swarm of per-session values
    per task variable, one panel per epoch. Task variables with no significant edge in any plotted epoch are
    dropped from the x axis (effort level).

    :param beta_df: the collapsed, model-level table (summary_score_df), not the raw per-coefficient regression,
    which would repeat the same score once per CC predictor row.
    """
    thresh = conf_dict['thresh'] if thresh is None else thresh
    score_a, score_b = f'score_{groups[0]}', f'score_{groups[1]}'
    colour_map = fw.venn_prm_colours()
    group_order = list(task_vars) if group_order is None else group_order
    sig = edge_df[(pd.to_numeric(edge_df['edge_pval'], errors='coerce') < thresh) &
                  np.isfinite(pd.to_numeric(edge_df['edge_value'], errors='coerce')) &
                  edge_df['epoch'].isin(epochs)].copy()
    group_order = [c for c in group_order if c in sig['coefficient'].unique()]

    def _session_diff_values(epoch, area_pair, coefficient):
        tbl = beta_df.query('epoch == @epoch & area_pair == @area_pair & outcome == @coefficient')
        if tbl.empty:
            area_pair = '_'.join(area_pair.split('_')[::-1])
            tbl = beta_df.query('epoch == @epoch & area_pair == @area_pair & outcome == @coefficient')
        if tbl.empty:
            return np.array([])
        return (tbl[score_a] - tbl[score_b]).groupby(tbl['subject_day']).mean().to_numpy()

    def _pooled_task_var_values(epoch, coefficient):
        pairs = sig.loc[(sig['epoch'] == epoch) & (sig['coefficient'] == coefficient), 'area_pair'].unique()
        parts = [p for p in (_session_diff_values(epoch, ap, coefficient) for ap in pairs) if p.size]
        return np.concatenate(parts) if parts else np.array([])

    x_labels = [approximate_var_label(c).capitalize() for c in group_order]
    gs = iter(hs.gridspec(1, len(epochs), wspace=.05, where=where))
    axes = []
    for i, epoch in enumerate(epochs):
        with mpl.rc_context(hs.rc0 if not i else hs.no_yaxis):
            ax = hs.get_ax(next(gs))
            for xi, coefficient in enumerate(group_order):
                vals = _pooled_task_var_values(epoch, coefficient)
                if not vals.size:
                    continue
                col = colour_map.get(coefficient, 'black')
                ax.bar(xi, vals.mean(), width=bar_width, color=col, alpha=0.4, zorder=2, edgecolor=col)
                sns.swarmplot(x=np.full(vals.shape, float(xi)), y=vals, color=col, edgecolor='black',
                              linewidth=0.6, size=dot_size, ax=ax, zorder=4, native_scale=True)
            ax.axhline(0, color='0.6', alpha=.5, lw=.8, ls='--', zorder=1)
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=90)
            ax.set_yticks(np.arange(-.5, 1.01, .5))
            ax.set(xlim=(-0.5, len(x_labels) - 0.5), ylim=(-.5, 1))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i == 0:
                ax.set_ylabel(ylabel if ylabel is not None else f'∆$r$ ({groups[0]} - {groups[1]})')
        axes.append(ax)
    for ax in axes:
        ax.spines['bottom'].set_bounds(*ax.get_xlim())
        ax.spines['left'].set_bounds(*ax.get_ylim())
    return axes


def plot_corridor_decoding(where):
    """Panel (b): multiplex ∆corr node diagrams (top) and per-session ∆r bar + swarm (bottom), corridor epochs."""
    cv_left = iter(hs.gridspec(2, 1, wspace=.1, height_ratios=[.6, .4], where=where))
    plot_r2_diff_multiplex_network(diff_edge_df, epochs=corridor_epochs, where=next(cv_left))
    plot_r2_diff_swarm_by_session(edge_df=diff_edge_df, beta_df=summary_score_df, epochs=corridor_epochs,
                                  where=next(cv_left), groups=('accept', 'reject'))


# (c) Formerly Fig. S4 (project 258066): is the true task variable more variable in accepted than rejected trials?
# Per session, the variance of the *true* trial-level value within accepted and within rejected trials (the larger
# group balanced down to the smaller group's trial count, mirroring cca_component_decoder_2's own balanced scoring),
# summarised as log2(accept variance / reject variance); the per-session ratios are tested against 0 with a Wilcoxon
# signed-rank test.
def _balanced_group_variance(df, trial_id_col, group_col, value_col, rng, B, min_trials=3):
    """
    Session-level mirror of cca_component_decoder_2's balance_val_trials block: the larger group's variance is the
    mean across B random min_n-*trial* subsamples (drawn as whole trials via trial_id_col so multi-row-per-trial
    velocity bins are kept or dropped together); the smaller group is left as-is. This reuses the decoder's seed and
    sampling procedure at the session level - the closest practical approximation, not a bit-for-bit replica of the
    per-fold draws.
    """
    trial_ids = {g: df.loc[df[group_col] == g, trial_id_col].unique() for g in ('accept', 'reject')}
    n = {g: len(t) for g, t in trial_ids.items()}
    if min(n.values()) == 0:
        return {'accept': np.nan, 'reject': np.nan}
    min_n = min(n.values())
    out = {}
    for g in ('accept', 'reject'):
        g_df = df.loc[df[group_col] == g]
        if n[g] > min_n and min_n >= min_trials:
            draws = []
            for _ in range(B):
                sel = rng.choice(trial_ids[g], size=min_n, replace=False)
                draws.append(g_df.loc[g_df[trial_id_col].isin(sel), value_col].var())
            out[g] = np.nanmean(draws)
        else:
            out[g] = g_df[value_col].var()
    return out


def build_outcome_variance_df(B=20, seed=347219):
    """Per-session log2(accept var / reject var) of each task variable's true value (see the cell comment above)."""
    # accepted (480152) + rejected (828817) spike sources pooled, as in 258066.py's decoder; the DEL epoch has no
    # equivalent state in rejected trials
    ar_omit = {'trial_order': [-1]}
    sp_df = pd.DataFrame()
    for cond, src_id in dict(reject=828817, accept=480152).items():
        st_src = load_SpikeStats(src=src_id)
        if cond == 'accept':
            st_src.spike_times = st_src.spike_times[st_src.spike_times['epoch'] != 'DEL']
        sp_df = pd.concat([sp_df, st_src.spike_times], ignore_index=True)
    ar_epochs = st_src.get_sorted_epochs(sp_df.epoch.unique())
    sp_df = st_src.drop_levs(ar_omit, sp_df)
    sp_df = st_src.sanitise_spike_times_df(fr_var=f'raw_fr_{list(conf_dict["bin_size"].values())[0]}ms_bin',
                                           sp_df=sp_df, fr_crit=.1)
    ar_subj_days = np.unique(np.sort([hs.get_subj_date(b) for b in sp_df.stage.unique()]))
    trial_groups = BlO.SpikeStats().build_trial_groups(sp_df, ar_subj_days, ar_epochs, sort_trials=True,
                                                       return_order_df=True)
    del sp_df, st_src

    beh_df = load_trial_behaviour(809107).beh_df
    trial_group_df = pd.concat([d.assign(subject_day=sd) for sd, d in trial_groups.items()], ignore_index=True)
    tvar_df = trial_group_df.merge(
        beh_df[['stage', 'trial_idx', 'trial_order', 'effort_lev']].drop_duplicates(subset=['stage', 'trial_idx']),
        on=['stage', 'trial_idx'], how='inner')
    tvar_df['trial_count'] = tvar_df['trial_idx']  # ordinal position of the trial within its stage
    tvar_df['trial_uid'] = list(zip(tvar_df['stage'], tvar_df['trial_idx']))

    # velocity per trial/epoch/bin, with its accept/reject group, cached by project 258066's own VelDf
    vel_ar_df = pd.read_feather(f'{get_file_root(ar_src_id)}_xy_table.feather')
    vel_ar_df['subject_day'] = [hs.get_subj_date(s) for s in vel_ar_df['stage']]
    vel_ar_df['trial_uid'] = list(zip(vel_ar_df['stage'], vel_ar_df['trial_idx']))

    sources = {'trial_order': (tvar_df, 'trial_order'), 'effort_lev': (tvar_df, 'effort_lev'),
               'trial_count': (tvar_df, 'trial_count'), 'velocity': (vel_ar_df, 'velocity')}
    rng = np.random.default_rng(seed=seed)
    var_df = pd.DataFrame()
    for tvar, (df_, col) in sources.items():
        rows = []
        for sd, sess_df in df_.groupby('subject_day'):
            bal = _balanced_group_variance(sess_df, trial_id_col='trial_uid', group_col='trial_group',
                                           value_col=col, rng=rng, B=B)
            if not (np.isfinite(bal['accept']) and np.isfinite(bal['reject'])):
                continue
            if bal['accept'] <= 0 or bal['reject'] <= 0:
                continue
            rows.append({'subject_day': sd, 'accept': bal['accept'], 'reject': bal['reject']})
        v = pd.DataFrame(rows).dropna()
        v['task_var'] = tvar
        # a ratio of the two groups' variances is scale-invariant (units cancel), so bars are comparable across
        # task variables; log2 makes it symmetric around 0 (equal spread)
        v['log2_ratio'] = np.log2(v['accept'] / v['reject'])
        var_df = pd.concat([var_df, v], ignore_index=True)
    return var_df


try:
    ov_var_df = hs.load_df(f'{out_root}_ov_var_df')
except FileNotFoundError:
    ov_var_df = build_outcome_variance_df()
    hs.save_df(ov_var_df, f'{out_root}_ov_var_df')


def plot_outcome_variance_ratio_bars(var_df, task_var_order, where, bar_width=0.6, thresh=None, ylabel=None):
    """
    One bar per task variable: the across-session mean of log2(accept variance / reject variance) of the true
    trial-level value; >0 means accept trials show more spread in that variable than reject trials, <0 the reverse.
    `*` marks task variables whose per-session log2 ratio differs from 0 (Wilcoxon signed-rank test).
    """
    colour_map = fw.venn_prm_colours()
    thresh = conf_dict['thresh'] if thresh is None else thresh
    with mpl.rc_context(hs.rc0):
        ax = hs.get_ax(where)
        x_labels = [approximate_var_label(c).capitalize() for c in task_var_order]
        for xi, tvar in enumerate(task_var_order):
            sub = var_df.query('task_var == @tvar')
            if sub.empty:
                continue
            col = colour_map.get(tvar, 'black')
            vals = sub['log2_ratio'].to_numpy()
            mean = vals.mean()
            ax.bar(xi, mean, width=bar_width, color=col, alpha=0.6, zorder=2, edgecolor=col)
            if vals.size >= 2:
                try:
                    _, p = wilcoxon(vals, alternative='two-sided')
                except ValueError:
                    p = np.nan
                if p < thresh:
                    ax.text(xi, mean, '*', ha='center', va='bottom' if mean >= 0 else 'top', fontsize=9,
                            color='black', zorder=5)
        ax.axhline(0, color='0.6', alpha=.5, lw=.8, ls='--', zorder=1)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set(xlim=(-0.5, len(x_labels) - 0.5))
        ax.set_ylabel(ylabel if ylabel is not None else 'log$_2$ variance ratio (accept / reject)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return ax


# Panel a: is the axis of a variable that replaces reward decoding a rotation of the earlier reward axis? Project 781039
# fits one non-cross-validated CCA per session, area pair and epoch, maps each task variable's CC-space decoder
# coefficients back onto the units (u = W b) and compares axes between epochs (cosine over the units in both epochs' CCAs).
# The table (781039_replacement_summary.csv) holds, for the three situations in which another variable replaces reward
# decoding, the mixed-model mean cosine between the replacing variable's axis in a corridor epoch and the reward axis of each
# earlier epoch in which reward is decodable, with its Wald / equivalence (TOST against +/-.2) tests, BH-corrected across the
# 24 comparisons. Drawn here with the (1 - 2 * thresh) = 98% interval the equivalence test at thresh uses.
ang_src_id = 781039        # oh_cue_run_outcome_cca_1_bin_v5_task_decoder_angles
EPOCH_COLOURS = ['#F0E442', '#E69F00', '#56B4E9', '#009E73', '#0072B2', '#CC79A7']  # Okabe & Ito, colour-blind safe
REPLACEMENT_CASES = [dict(area_pair='mo_str', replaced='trial_order', replacers=['velocity', 'trial_count']),
                     dict(area_pair='mo_stn', replaced='trial_order', replacers=['velocity', 'trial_count']),
                     dict(area_pair='acc_str', replaced='trial_order', replacers=['velocity'])]


def plot_replacement_summary(where, equiv_delta=0.2, thresh=None, dodge=0.7):
    """
    One panel per replacement case, in a row. x: the replacing variable's axis in a corridor epoch; points within a group:
    the reward axis of each earlier epoch (colour). y: mean cosine between the two axes with its (1 - 2 * thresh)
    interval; the grey band is +/-equiv_delta. `≈` = equivalent to 0 (interval within the band, BH-corrected across the
    24 comparisons), `*` = differs from 0 (BH-corrected), no mark = inconclusive. Returns the axes.
    """
    thresh = conf_dict['thresh'] if thresh is None else thresh
    z = norm.ppf(1 - thresh)
    epoch_order = ['PRE', 'CUE', 'NP_EX', 'CORR1', 'CORR6', 'DEL']
    cols = dict(zip(epoch_order, EPOCH_COLOURS))
    lab = lambda x: approximate_var_label(x) or x
    pair_lab = lambda p: '-'.join(str(fw.region_labels().get(a, a)) for a in p.split('_'))
    stats_df = pd.read_csv(f'{get_file_root(ang_src_id)}_replacement_summary.csv')
    to_epochs = [e for e in epoch_order if e in set(stats_df['epoch_to'])]
    with mpl.rc_context(hs.rc0):
        gs = hs.gridspec(1, len(REPLACEMENT_CASES), where=where, wspace=0.12)
        axes, y_all = [], []
        for i, case in enumerate(REPLACEMENT_CASES):
            ax = hs.get_ax(gs[0, i])
            axes.append(ax)
            sub = stats_df[(stats_df['area_pair'] == case['area_pair']) & (stats_df['replaced'] == case['replaced'])]
            groups = [(rp, ep) for rp in case['replacers'] for ep in to_epochs
                      if ((sub['replacer'] == rp) & (sub['epoch_to'] == ep)).any()]
            ax.axhspan(-equiv_delta, equiv_delta, color='grey', alpha=0.15, lw=0, zorder=0)
            ax.axhline(0, color='grey', ls='--', lw=0.8, zorder=0)
            for g, (rp, ep_to) in enumerate(groups):
                g_sub = sub[(sub['replacer'] == rp) & (sub['epoch_to'] == ep_to)]
                offs = np.linspace(-dodge / 2, dodge / 2, len(g_sub)) if len(g_sub) > 1 else [0.0]
                for off, r in zip(offs, g_sub.itertuples()):
                    if not np.isfinite(r.mean):
                        continue
                    lo, hi = r.mean - z * r.sem, r.mean + z * r.sem
                    ax.plot([g + off] * 2, [lo, hi], color=cols[r.epoch_from], lw=1.4, solid_capstyle='butt', zorder=3)
                    ax.scatter([g + off], [r.mean], color=cols[r.epoch_from], edgecolor='black', linewidth=0.5, s=24,
                               zorder=4)
                    mark = ('*' if r.significant else '') + ('≈' if r.equivalent else '')
                    if mark:
                        ax.text(g + off, hi + 0.02, mark, ha='center', va='bottom', fontsize=9, fontweight='bold')
                    y_all += [lo, hi]
                if g:
                    ax.axvline(g - 0.5, color='0.6', ls=':', lw=0.8, zorder=0)
            ax.set_xticks(np.arange(len(groups)))
            ax.set_xticklabels([f'{lab(rp)}\n{lab(ep)}' for rp, ep in groups], fontsize=6)
            ax.set_xlim(-0.6, max(len(groups), 1) - 0.4)
            n_s = sub['n_sessions'][sub['n_sessions'] > 0]
            ax.set_title(f'{pair_lab(case["area_pair"])}: ' + ' / '.join(lab(r) for r in case['replacers']) +
                         f' replace{"s" if len(case["replacers"]) == 1 else ""} {lab(case["replaced"])}\n'
                         + (f'{n_s.min() if n_s.min() == n_s.max() else f"{n_s.min()}–{n_s.max()}"} sessions, '
                            f'{sub["n_subjects"].max()} animals' if len(n_s) else ''), fontsize=7)
            ax.set_xlabel("replacing variable's axis in epoch", fontsize=7)
            if not i:
                ax.set_ylabel(f'cosine with {lab(REPLACEMENT_CASES[0]["replaced"])} axis of earlier epoch\n'
                              f'(mean, {int(round(100 * (1 - 2 * thresh)))}% CI)', fontsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        lim = max([abs(v) for v in y_all] + [equiv_delta]) * 1.15 if y_all else 0.5
        for ax in axes:
            ax.set_ylim(-lim, lim)
            if ax is not axes[0]:
                ax.set_yticklabels([])
        handles = [Line2D([0], [0], marker='o', linestyle='None', markeredgecolor='black', markeredgewidth=0.5,
                          color=cols[e], label=lab(e)) for e in epoch_order if e in set(stats_df['epoch_from'])]
        axes[-1].legend(handles=handles, title=f'{lab(REPLACEMENT_CASES[0]["replaced"])} axis in', frameon=False,
                        fontsize=6, title_fontsize=6, loc='center left', bbox_to_anchor=(1.0, 0.5))
    return axes


# Pull it all together: the summary row (a) is added above the previous layout, whose panels (the running speed profiles,
# the corridor decoding and the variance ratios) therefore move one letter down
fig_w, fig_h = hs.config.landscape_A(4)
summary_h = 3.0  # inches
cvs = iter(hs.gridspec(3, 1, hspace=.4, height_ratios=[summary_h, .3 * fig_h, .7 * fig_h],
                       where=(fig_w, fig_h + summary_h)))
hs.label_panels(plot_replacement_summary, start='a', skip=0, count=1)(where=next(cvs))
hs.label_panels(plot_velocity_by_epoch, start='b', skip=0, count=1)(vel_df, where=next(cvs))
cv_row = iter(hs.gridspec(1, 2, wspace=.3, width_ratios=[.7, .3], where=next(cvs)))
hs.label_panels(plot_corridor_decoding, start='c', skip=0, count=1)(where=next(cv_row))
hs.label_panels(plot_outcome_variance_ratio_bars, start='d', skip=0, count=1)(
    ov_var_df, task_var_order=task_vars, where=next(cv_row))
hs.savefig(f"{out_root}_fig_S3.png", open=OPEN_FIGS)
hs.savefig(f"{out_root}_fig_S3.svg", dpi=600)


#%%
# Supplemental figure 4: relation of each area pair's dominant CCA rho to the trial-count coefficient (trial_count) of the
# per-session choice GLM, laid out as the reward/effort figure built in project 746484 (this section's analysis and panel functions are adapted from that project's script, which
# computes the per-session GLM coefficients from the same behavioural data and the canonical correlations of its decision
# window). Columns: (a) per-session fitted choice curves with the cohort mean, (b) slope of the coefficient
# on z-scored dominant rho for every area pair (mixed model, random intercept per animal; bars = beta +/- s.e.), (c) session
# scatter and fit for the area pair with the smallest p-value, (d) rho across pooled quartiles of the coefficient.

sens_src_id = 746484       # oh_decision_cca_1_bin_connectivity_behaviour_regression
sens_formula = 'far_mag_entry ~ trial_order + effort_lev + trial_count'
conf_dict['circuit_map'] = {'acc_stn': 'Fr-BG', 'stn_str': 'BG-BG', 'acc_str': 'Fr-BG', 'mo_str': 'Fr-BG',
                            'mo_stn': 'Fr-BG', 'acc_mo': 'Fr-Fr', 'acc_vp': 'Fr-BG', 'str_vp': 'BG-BG'}
conf_dict['circuit_colour'] = {'Fr-Fr': '#1864AB', 'BG-BG': '#C2255C', 'Fr-BG': '#862E9C'}
s4_axis_labels = {'trial_count': 'Trial-count β (GLM)'}
s4_subscripts = {'trial_count': 'trial count'}


def s4_pair_label(area_pair: str) -> str:
    """Display label of an area pair 'a_b' as 'A-B'; the striatum-STN pair is written DMS-STA (striatum first)."""
    a1, a2 = area_pair.split('_')
    if (a1, a2) == ('stn', 'str'):
        a1, a2 = a2, a1
    return f"{fw.region_labels()[a1]}-{fw.region_labels()[a2]}"


def fit_session_behaviour(beh_df: pd.DataFrame, formula: str, session_col: str = 'day_subj') -> pd.DataFrame:
    """
    Fit `formula` as a binomial GLM separately within each session (day_subj). Returns the session's intercept
    ('sess_intercept', named so it can sit beside the other coefficients in a `corr_cv ~ ...` formula) and betas for
    trial_order (reward), effort_lev (effort) and trial_count (trial count), the fit itself, and the reward level at which the
    fitted curve crosses P(accept) = 0.5. Sessions that cannot support a stable fit keep NaN coefficients.
    """
    required_cols = ['far_mag_entry', 'trial_order', 'effort_lev', 'trial_count']
    fit_params = ['Intercept', 'trial_order', 'effort_lev', 'trial_count']
    out_col = {'Intercept': 'sess_intercept', 'trial_order': 'trial_order', 'effort_lev': 'effort_lev',
               'trial_count': 'trial_count'}
    out_cols = [session_col, 'subject_id'] + list(out_col.values()) + ['indiff_reward', 'rmse', 'n_trials', 'fit']
    records = []
    for session, df_s in beh_df.groupby(session_col):
        df_s = df_s.dropna(subset=required_cols)
        reward_levels = np.sort(df_s['trial_order'].unique())
        record = {session_col: session, 'subject_id': hs.get_subj_lab(session), **{c: np.nan for c in out_col.values()},
                  'indiff_reward': np.nan, 'rmse': np.nan, 'n_trials': df_s.shape[0], 'fit': None}
        if df_s['far_mag_entry'].nunique() < 2 or len(reward_levels) < 2:
            records.append(record)
            continue
        try:
            fit = smf.glm(formula, data=df_s, family=sm.families.Binomial()).fit()
        except (PerfectSeparationError, np.linalg.LinAlgError):
            records.append(record)
            continue
        rmse = np.sqrt(np.mean((df_s['far_mag_entry'].to_numpy() - fit.predict(df_s).to_numpy()) ** 2))
        pred_df = pd.DataFrame({'trial_order': reward_levels, 'effort_lev': df_s['effort_lev'].median(),
                                'trial_count': df_s['trial_count'].median()})
        pred_df['p_accept'] = fit.predict(pred_df)
        best = pred_df.iloc[(pred_df['p_accept'] - 0.5).abs().argmin()]
        record.update({out_col[p]: fit.params.get(p, np.nan) for p in fit_params})
        record.update(indiff_reward=best['trial_order'], rmse=rmse, fit=fit)
        records.append(record)
    return pd.DataFrame.from_records(records, columns=out_cols)


def get_session_beta_rho_df(session_beh_df: pd.DataFrame, cca_df: pd.DataFrame, sig_area_pairs: Iterable[str],
                            session_col: str = 'day_subj') -> pd.DataFrame:
    """Merge each session's GLM coefficients onto its dominant (rank 0) CCA rho for every area pair in sig_area_pairs."""
    rho_df = cca_df.loc[(cca_df['rank'] == 0) & (cca_df['area_pair'].isin(list(sig_area_pairs))),
                        ['subject_day', 'area_pair', 'corr_cv']]
    return session_beh_df.merge(rho_df, left_on=session_col, right_on='subject_day', how='inner')


def regress_rho_on_beta(model_df: pd.DataFrame, outcome: str, group_col: str = 'subject_id') -> pd.DataFrame:
    """
    For every area pair, `outcome ~ corr_cv` with corr_cv z-scored within the pair (beta = change in the coefficient per SD of
    rho), as a linear mixed model with a random intercept per animal (OLS if only one animal contributes).
    """
    formula = f'{outcome} ~ corr_cv'
    rows = []
    for area_pair, sub in model_df.groupby('area_pair'):
        sub = sub.dropna(subset=['corr_cv']).copy()
        rho_sd = sub['corr_cv'].std()
        if not rho_sd > 0:
            continue
        sub['corr_cv'] = (sub['corr_cv'] - sub['corr_cv'].mean()) / rho_sd
        df_p = sub.dropna(subset=[outcome, group_col])
        if df_p.shape[0] < 3:
            continue
        n_subjects = df_p[group_col].nunique()
        # fit on a unit-SD outcome and scale beta/se back: the mixed model's optimiser is unreliable for tiny coefficients
        y_scale = df_p[outcome].std()
        df_p = df_p.assign(**{outcome: df_p[outcome] / y_scale})
        fit = (smf.mixedlm(formula, data=df_p, groups=df_p[group_col]).fit() if n_subjects > 1
               else smf.ols(formula, data=df_p).fit())
        if not np.isfinite(fit.bse['corr_cv']):  # singular mixed-model fit (zero animal variance): equivalent to OLS
            print(f'{area_pair} {outcome}: singular mixed-model fit (zero animal variance), falling back to OLS')
            fit = smf.ols(formula, data=df_p).fit()
        rows.append(dict(area_pair=area_pair, outcome=outcome, beta=fit.params['corr_cv'] * y_scale,
                         se=fit.bse['corr_cv'] * y_scale,
                         pval=fit.pvalues['corr_cv'], n_sessions=df_p.shape[0], n_subjects=n_subjects))
    return pd.DataFrame(rows).sort_values('beta', ascending=False).reset_index(drop=True)


def compute_session_choice_curves(beh_df: pd.DataFrame, session_beh_df: pd.DataFrame, x_var: str,
                                  sessions: Optional[Iterable] = None, n_grid: int = 30,
                                  session_col: str = 'day_subj') -> pd.DataFrame:
    """
    Each session's fitted P(accept) against the regressor `x_var`. For the discrete regressors (trial_order, effort_lev) it is
    evaluated at the session's observed levels; for trial_count at n_grid points across the session's observed range. The other
    two regressors are held at the session median.
    :return: long DataFrame [session_col, subject_id, level, p_accept].
    """
    regs = ('trial_order', 'effort_lev', 'trial_count')
    others = [v for v in regs if v != x_var]
    sel = session_beh_df.dropna(subset=['fit'])
    if sessions is not None:
        sel = sel[sel[session_col].isin(list(sessions))]
    rows = []
    for _, r in sel.iterrows():
        df_s = beh_df[beh_df[session_col] == r[session_col]].dropna(subset=['far_mag_entry', *regs])
        levels = (np.linspace(df_s[x_var].min(), df_s[x_var].max(), n_grid) if x_var == 'trial_count'
                  else np.sort(df_s[x_var].unique()))
        pred = pd.DataFrame({x_var: levels, **{v: df_s[v].median() for v in others}})
        rows.append(pd.DataFrame({session_col: r[session_col], 'subject_id': r['subject_id'], 'level': levels,
                                  'p_accept': r['fit'].predict(pred).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def plot_session_choice_curves(curves: pd.DataFrame, x_label: str, ax, session_col: str = 'day_subj'):
    """
    Fitted P(accept) against one regressor: a thin line per session (coloured by animal, the 'Reds' shades of project 553431),
    a thick black cohort-mean line (for the continuous trial count: the mean of the sessions' interpolated curves, up to the
    25th percentile of the sessions' maxima so that at least 75% of sessions contribute throughout) and a one-row legend above
    the axes.
    """
    subjects = sorted(curves['subject_id'].unique())
    shades = dict(zip(subjects, mpl.colormaps['Reds'](np.linspace(.15, .9, len(subjects)))))
    per_session = [g for _, g in curves.groupby(session_col)]
    levels = np.sort(curves['level'].unique())
    ax.set_ylim(0, 1)  # fix both limits before drawing: a pending autoscale under hs.rc0 crashes matplotlib's tick locator
    ax.set_xlim(0, max(g['level'].max() for g in per_session) * 1.02) if len(levels) > 8 else ax.set_xlim(
        levels.min() - .5, levels.max() + .5)
    for g in per_session:
        ax.plot(g['level'], g['p_accept'], color=shades[g['subject_id'].iloc[0]], lw=.6, alpha=.75, zorder=1)
    if len(levels) <= 8:
        mean = curves.groupby('level')['p_accept'].mean()
        mx, my = mean.index.to_numpy(), mean.to_numpy()
        ax.set_xticks(levels)
    else:
        grid = np.linspace(levels.min(), np.percentile([g['level'].max() for g in per_session], 25), 60)
        vals = np.full((len(per_session), grid.size), np.nan)
        for i, g in enumerate(per_session):
            inside = (grid >= g['level'].min()) & (grid <= g['level'].max())
            vals[i, inside] = np.interp(grid[inside], g['level'], g['p_accept'])
        keep = np.isfinite(vals).mean(0) >= .75
        mx, my = grid[keep], np.nanmean(vals[:, keep], axis=0)
    ax.plot(mx, my, color='k', lw=2.2, zorder=3)
    ax.axhline(.5, ls='--', color='grey', lw=.6, zorder=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r'$P_{accept}$')
    handles = [Line2D([0], [0], color=shades[subj], lw=1) for subj in subjects] + [Line2D([0], [0], color='k', lw=2.2)]
    ax.legend(handles, subjects + ['Mean'], ncol=3, loc='lower left', bbox_to_anchor=(.1, 1.0), frameon=False,
              fontsize=5, columnspacing=.7, handlelength=1.1, handletextpad=.3, borderaxespad=.2)


def plot_area_beta_ranking(beta_df: pd.DataFrame, outcome: str, ax):
    """
    Horizontal bars of the slope of `outcome` on z-scored dominant rho for every area pair (rank-ordered, coloured by circuit
    class, thin black rim, 70% opaque fill), with +/- 1 s.e. of the mixed-model slope; * marks p < conf_dict['thresh'] (uncorrected).
    """
    sub = beta_df[beta_df['outcome'] == outcome].sort_values('beta').reset_index(drop=True)
    labels = [s4_pair_label(a) for a in sub['area_pair']]
    colours = [conf_dict['circuit_colour'][conf_dict['circuit_map'].get(a, 'Fr-BG')] for a in sub['area_pair']]
    y = np.arange(sub.shape[0])
    ax.barh(y, sub['beta'], color=[mpl.colors.to_rgba(c, .7) for c in colours], edgecolor='k', linewidth=.5, zorder=2)
    ax.errorbar(sub['beta'], y, xerr=sub['se'], fmt='none', ecolor='k', elinewidth=.6, capsize=1.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, ls='--', color='grey', lw=.8)
    sub_lab = s4_subscripts[outcome].replace(' ', chr(92) + ' ')
    ax.set_xlabel(rf'$\beta_{{\mathrm{{{sub_lab}}}}}$ ~ z-scored' + '\n' + r'dominant CCA $\rho$')
    ax.set_ylabel('Area')
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))  # slopes can be tiny (e.g. 1e-3), keep tick labels apart
    x_pad = .02 * sub['beta'].abs().max()
    for yi, (beta, se, pval) in enumerate(zip(sub['beta'], sub['se'], sub['pval'])):
        if pval < conf_dict['thresh']:
            ax.text(beta + np.sign(beta) * (se + x_pad), yi, '*', ha='left' if beta >= 0 else 'right', va='center',
                    fontsize=12, fontweight='bold')
    ax.legend([mpl.patches.Patch(facecolor=mpl.colors.to_rgba(c, .7), edgecolor='k', linewidth=.5)
               for c in conf_dict['circuit_colour'].values()], list(conf_dict['circuit_colour']), frameon=False, fontsize=6,
              loc='upper left')
    ax.spines['left'].set_bounds(0, len(y) - 1)  # truncate the category axis at its first and last ticks
    hs.displace_axes(ax, .1)


def plot_session_scatter(model_df: pd.DataFrame, area_pair: str, outcome: str, ax, group_col: str = 'subject_id'):
    """
    Session scatter of `outcome` against the area pair's dominant rho, dots coloured by animal, with the population-level fit
    and 95% CI of the random-intercept mixed model of regress_rho_on_beta (drawn against raw rho).
    """
    sub = model_df[model_df['area_pair'] == area_pair].dropna(subset=['corr_cv', outcome, group_col]).copy()
    mu, sd = sub['corr_cv'].mean(), sub['corr_cv'].std()
    sub['rho_z'] = (sub['corr_cv'] - mu) / sd
    n_subjects = sub[group_col].nunique()
    y_scale = sub[outcome].std()  # fit on a unit-SD outcome (see regress_rho_on_beta) and scale back for plotting
    sub['y_unit'] = sub[outcome] / y_scale
    fit = (smf.mixedlm('y_unit ~ rho_z', data=sub, groups=sub[group_col]).fit() if n_subjects > 1
           else smf.ols('y_unit ~ rho_z', data=sub).fit())
    x = np.linspace(sub['corr_cv'].min(), sub['corr_cv'].max(), 100)
    design = np.column_stack([np.ones_like(x), (x - mu) / sd])
    names = ['Intercept', 'rho_z']
    y_hat = design @ fit.params[names].to_numpy() * y_scale
    y_se = np.sqrt(np.einsum('ij,jk,ik->i', design, fit.cov_params().loc[names, names].to_numpy(), design)) * y_scale
    subjects = sorted(sub[group_col].unique())
    shades = mpl.colormaps['Reds'](np.linspace(.15, .9, len(subjects)))
    ax.fill_between(x, y_hat - 1.96 * y_se, y_hat + 1.96 * y_se, color='#868E96', alpha=.25, lw=0, zorder=1)
    ax.plot(x, y_hat, color='k', lw=1.2, zorder=2)
    for subj, colour in zip(subjects, shades):
        d = sub[sub[group_col] == subj]
        ax.scatter(d['corr_cv'], d[outcome], s=24, color=colour, edgecolor='k', lw=.6, zorder=3,
                   label=f'{subj} (n={len(d)})')
    ax.set_xlabel(rf'Dominant CCA $\rho$ ({s4_pair_label(area_pair)})')
    ax.set_ylabel(s4_axis_labels[outcome])
    ax.legend(title='Animal', loc='best', frameon=True, fontsize=5, title_fontsize=6)
    ci = fit.conf_int().loc['rho_z']
    print(f'{area_pair} {outcome}: {sub.shape[0]} sessions, {n_subjects} animals, beta per SD rho = '
          f"{fit.params['rho_z'] * y_scale:+.4g} (95% CI {ci[0] * y_scale:+.4g} to {ci[1] * y_scale:+.4g}), "
          f"p = {fit.pvalues['rho_z']:.4f}")


def plot_rho_by_quartile(model_df: pd.DataFrame, area_pair: str, outcome: str, ax, group_col: str = 'subject_id',
                         n_bins: int = 4):
    """Dominant rho of `area_pair` across quartiles of `outcome` (sessions pooled): box plots over jittered session dots coloured by animal."""
    from scipy.stats import kruskal, spearmanr
    sub = model_df[model_df['area_pair'] == area_pair].dropna(subset=['corr_cv', outcome, group_col]).copy()
    sub['q'] = pd.qcut(sub[outcome].rank(method='first'), n_bins, labels=False)
    subjects = sorted(sub[group_col].unique())
    shades = dict(zip(subjects, mpl.colormaps['Reds'](np.linspace(.15, .9, len(subjects)))))
    groups = [sub.loc[sub['q'] == q, 'corr_cv'].to_numpy() for q in range(n_bins)]
    ax.boxplot(groups, positions=range(n_bins), widths=.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='#F1F3F5', edgecolor='k', lw=.8), medianprops=dict(color='k', lw=1.4),
               whiskerprops=dict(lw=.8), capprops=dict(lw=.8), zorder=1)
    rng = np.random.default_rng(0)
    for subj in subjects:
        d = sub[sub[group_col] == subj]
        ax.scatter(d['q'] + rng.uniform(-.17, .17, len(d)), d['corr_cv'], s=16, color=shades[subj], edgecolor='k', lw=.5,
                   zorder=3)
    xlab = {0: 'Q1\n(lowest)', n_bins - 1: f'Q{n_bins}\n(highest)'}
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([xlab.get(q, f'Q{q + 1}') + f'\n(n={len(groups[q])})' for q in range(n_bins)], fontsize=5)
    ax.set_xlabel(f'{s4_axis_labels[outcome]} quartile')
    ax.set_ylabel(rf'Dominant CCA $\rho$ ({s4_pair_label(area_pair)})')
    kw, rs = kruskal(*groups), spearmanr(sub['q'], sub['corr_cv'])
    print(f'{area_pair} {outcome} quartiles: medians ' + ', '.join(f'Q{q + 1} {np.median(g):+.3f}' for q, g in enumerate(groups)) +
          f' | Kruskal-Wallis p = {kw.pvalue:.4f}; Spearman rs = {rs.statistic:+.3f}, p = {rs.pvalue:.4f}')


# 1. Per-session choice GLMs and the dominant rho of each area pair with a significant rank-0 dimension
st_sens = load_SpikeStats(src=sens_src_id)
st_sens.canonical_corr['subject'] = [_.split('_')[0] for _ in st_sens.canonical_corr.subject_day]
st_sens.canonical_corr['∆rho'] = (np.abs(st_sens.canonical_corr['corr_cv']) -
                                  np.abs(st_sens.canonical_corr['corr_cv_perm_avg']))
st_sens.canonical_corr = st_sens.canonical_corr[st_sens.canonical_corr['rank'] != -1]
sens_beh_df = load_trial_behaviour(809107).beh_df
sens_session_df = fit_session_behaviour(sens_beh_df, sens_formula)
sens_edges = st_sens.get_cca_df_edge_params(conf_dict['thresh'])
sens_edges = sens_edges[(sens_edges['rank'] == 0) & (sens_edges['edge_pval'] < 1)]
sens_model_df = get_session_beta_rho_df(sens_session_df, st_sens.canonical_corr, sens_edges['area_pair'].unique())
sens_outcomes = ('trial_count',)
sens_beta_df = pd.concat([regress_rho_on_beta(sens_model_df, outcome=o) for o in sens_outcomes], ignore_index=True)
print(sens_beta_df.round(5).to_string(index=False))

# 2. The figure: choice curves against trial count (a); the area pair shown in columns c and d is the one with the smallest
#    p-value for that coefficient. Drawn on a canvas as wide as a portrait A4 page and 31% of its height.
s4_curve_specs = {'trial_count': dict(x_var='trial_count', x_label='Trial count')}

def plot_sensitivity_figure(where):
    """Panels (a-d) for the single outcome in `sens_outcomes`, in one row of the canvas `where`."""
    # explicit margins so the panels run to the canvas edges (room is left only for the y-axis labels, the panel
    # letters, and the two-line x-axis labels); the layout engines are switched off so that nothing re-pads the figure
    gs = hs.gridspec(1, 4, where=where, wspace=.55, width_ratios=[.85, 1.05, 1, 1],
                     left=.045, right=.998, top=.83, bottom=.19)
    gs.figure.set_layout_engine('none')
    for cell, outcome in zip([gs], sens_outcomes):
        pair = sens_beta_df[sens_beta_df['outcome'] == outcome].sort_values('pval')['area_pair'].iloc[0]
        print(f'Fig. S4, {outcome}: showing area pair {pair} (smallest p-value)')
        spec = s4_curve_specs[outcome]
        curves = compute_session_choice_curves(sens_beh_df, sens_session_df, spec['x_var'],
                                               sessions=sens_model_df.loc[sens_model_df['area_pair'] == pair, 'day_subj'])
        with mpl.rc_context(s4_rc):  # create and draw inside the same rc context (as the former plt.subplots version)
            axes = [hs.get_ax(g) for g in cell]
            plot_session_choice_curves(curves, spec['x_label'], axes[0])
            plot_area_beta_ranking(sens_beta_df, outcome, axes[1])
            plot_session_scatter(sens_model_df, pair, outcome, axes[2])
            plot_rho_by_quartile(sens_model_df, pair, outcome, axes[3])


s4_w, s4_h = hs.portrait_A(4)[0], .2 * hs.portrait_A(4)[1]
s4_rc = {**hs.rc0, 'axes.labelsize': 6, 'xtick.labelsize': 5, 'ytick.labelsize': 5, 'legend.fontsize': 5,
         'legend.title_fontsize': 6}  # the canvas is only 8.3 in wide for four panels, so use smaller type than hs.rc0
with mpl.rc_context(s4_rc):  # the figure must also be saved (i.e. its pending autoscaling resolved) under hs.rc0
    hs.label_panels(plot_sensitivity_figure, start='a', skip=0, count=4)(where=(s4_w, s4_h))
    plt.savefig(f"{out_root}_fig_S4.svg", dpi=300)  # plain savefig: hs.savefig would re-run tight_layout over the margins above
    plt.savefig(f"{out_root}_fig_S4.png", dpi=300)
    if OPEN_FIGS:
        hs.os_based_open(f"{out_root}_fig_S4.png")
