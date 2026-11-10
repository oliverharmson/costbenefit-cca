# Distributed under the MIT license; see LICENSE.
# Connectivity-behaviour relationship (project 746484): is a session's trait connectivity -- the dominant canonical
# correlation (CCA rho) between two areas over the decision window -- related to that session's choice behaviour?
# Choices (accept vs reject) are regressed per session on reward (trial_order), effort (effort_lev) and trial count with a
# binomial GLM; each coefficient is then regressed on the session's dominant rho for every area pair whose first
# canonical dimension is significant (mixed model with animal as grouping factor). The CCA results
# (746484_canonical_corr.feather) were computed from the decision-window firing rates of accepted and rejected trials.
# The script draws FIGURE 4 of the manuscript (output/746484_sensitivity_summary; one row per behavioural coefficient):
#   reward sensitivity (trial_order; pair STA-DMS):  Fig. 4a session choice curves, 4b area-pair hierarchy of the slope of
#       sensitivity on dominant rho, 4c session scatter and mixed-model fit, 4d rho across quartiles of sensitivity
#   effort sensitivity (effort_lev; pair MO/VO-DMS): Fig. 4e choice curves, 4f area-pair hierarchy, 4g scatter, 4h quartiles
# It also writes the area-pair hierarchy for every coefficient (output/746484_<coefficient>_on_rho_area_ranking); the
# reward and effort versions are panels 4b and 4f.

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import statsmodels.api as sm
import matplotlib.pyplot as plt
from typing import Iterable, Optional
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd(), '..', 'lib'))
import helpers as hs
import framework as fw
from block import BlockObj as BlO
from matplotlib.lines import Line2D
from catalogue import get_file_root, get_output_root, load_trial_behaviour
from block import approximate_var_label

PROJECT_ID = 746484
OPEN_FIGS = False  # open the saved PNGs in the OS viewer
conf_dict = {
    'p_thresh': .01,
    'formula': 'far_mag_entry ~ trial_order + effort_lev + trial_count',
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
    'outcome_labels': {'trial_order': 'Reward sensitivity (GLM β)', 'effort_lev': 'Effort sensitivity (GLM β)',
                       'trial_count': 'Trial-count β (GLM)', 'sess_intercept': 'Session intercept (GLM)'},
}
mpl.rcParams['svg.fonttype'] = 'none'
out_root = get_output_root(PROJECT_ID)  # figures are written to output/746484_*


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

# ---------------------------------------------------------------------------
# Per-session behaviour regression, then regress each behavioural coefficient
# on that session's dominant (rank 0, i.e. strongest in the training set) CCA
# rho for every area pair whose rank-0 dimension is significant.
# ---------------------------------------------------------------------------


def fit_session_behaviour(beh_df: pd.DataFrame,
                          formula: str,
                          session_col: str = 'day_subj') -> pd.DataFrame:
    """
    Fit `formula` as a binomial GLM separately within each session (day_subj), returning that
    session's intercept and betas for each predictor in the formula -- trial_order (reward),
    effort_lev (effort) and trial_count (trial count) -- alongside the reward level at which the
    fitted curve crosses P(accept) = 0.5, i.e. that session's behavioural indifference point.

    trial_order is swept over its session-specific observed levels while effort_lev and
    trial_count are held at that session's median to locate indiff_reward, mirroring
    fit_session_indifference in project 284485.

    Sessions that can't support a stable fit (a single far_mag_entry outcome, fewer than two
    reward levels, or a perfectly-separated fit) are kept in the output with NaN coefficients
    rather than dropped, so downstream code can see which sessions were excluded and why.

    :param beh_df: trial-level behaviour dataframe (e.g. BehDf().beh_df), one row per trial.
    :param formula: patsy formula for the per-session GLM (conf_dict['formula']).
    :param session_col: column identifying a single recording session.
    :return: pd.DataFrame, one row per session, with [session_col, 'subject_id',
        'sess_intercept', 'trial_order', 'effort_lev', 'trial_count', 'indiff_reward', 'rmse',
        'n_trials', 'fit']. The GLM's own constant is reported as 'sess_intercept' rather than
        'Intercept' so it can later sit alongside trial_order/effort_lev/trial_count as a regressor
        in a `corr_cv ~ ...` formula without colliding with patsy's own intercept term.
    """
    required_cols = ['far_mag_entry', 'trial_order', 'effort_lev', 'trial_count']
    fit_params = ['Intercept', 'trial_order', 'effort_lev', 'trial_count']
    out_col = {'Intercept': 'sess_intercept', 'trial_order': 'trial_order',
              'effort_lev': 'effort_lev', 'trial_count': 'trial_count'}
    out_cols = [session_col, 'subject_id'] + list(out_col.values()) + \
              ['indiff_reward', 'rmse', 'n_trials', 'fit']
    records = []
    for session, df_s in beh_df.groupby(session_col):
        df_s = df_s.dropna(subset=required_cols)
        reward_levels = np.sort(df_s['trial_order'].unique())
        record = {session_col: session, 'subject_id': hs.get_subj_lab(session),
                  **{c: np.nan for c in out_col.values()},
                  'indiff_reward': np.nan, 'rmse': np.nan, 'n_trials': df_s.shape[0], 'fit': None}
        if df_s['far_mag_entry'].nunique() < 2 or len(reward_levels) < 2:
            # can't estimate a GLM without both outcomes and >1 reward level
            records.append(record)
            continue
        try:
            fit = smf.glm(formula, data=df_s, family=sm.families.Binomial()).fit()
        except (PerfectSeparationError, np.linalg.LinAlgError):
            records.append(record)
            continue
        rmse = np.sqrt(np.mean((df_s['far_mag_entry'].to_numpy() - fit.predict(df_s).to_numpy()) ** 2))
        pred_df = pd.DataFrame({
            'trial_order': reward_levels,
            'effort_lev': df_s['effort_lev'].median(),
            'trial_count': df_s['trial_count'].median(),
        })
        pred_df['p_accept'] = fit.predict(pred_df)
        best = pred_df.iloc[(pred_df['p_accept'] - 0.5).abs().argmin()]
        record.update({out_col[p]: fit.params.get(p, np.nan) for p in fit_params})
        record.update(indiff_reward=best['trial_order'], rmse=rmse, fit=fit)
        records.append(record)
    return pd.DataFrame.from_records(records, columns=out_cols)


def get_session_beta_rho_df(session_beh_df: pd.DataFrame,
                            cca_df: pd.DataFrame,
                            sig_area_pairs: Iterable[str],
                            session_col: str = 'day_subj') -> pd.DataFrame:
    """
    Merge each session's behavioural GLM coefficients (from fit_session_behaviour) onto that
    session's dominant (rank 0) CCA rho, for every area pair in sig_area_pairs.

    :param session_beh_df: output of fit_session_behaviour.
    :param cca_df: st.canonical_corr, with columns subject_day/area_pair/rank/corr_cv.
    :param sig_area_pairs: area_pair labels whose rank-0 dimension was significant.
    :return: pd.DataFrame, one row per session x area_pair.
    """
    rho_df = cca_df.loc[(cca_df['rank'] == 0) & (cca_df['area_pair'].isin(list(sig_area_pairs))),
                        ['subject_day', 'area_pair', 'corr_cv']]
    return session_beh_df.merge(rho_df, left_on=session_col, right_on='subject_day', how='inner')


def regress_rho_on_beta(model_df: pd.DataFrame,
                        outcome: str = 'sess_intercept',
                        group_col: str = 'subject_id',
                        ) -> pd.DataFrame:
    """
    For every area pair, regress a per-session behavioural coefficient (`outcome`) on that area
    pair's per-session dominant CCA rho (corr_cv): `outcome ~ corr_cv`. corr_cv is z-scored
    within each area pair (across that pair's sessions) before fitting, so the beta is the change
    in the behavioural coefficient per 1 SD of that pair's rho -- comparable across area pairs
    whose rho spreads differ. Its sign says whether stronger coupling in that area pair predicts
    higher (+) or lower (-) values of the coefficient -- e.g. for outcome='trial_order', higher
    vs lower reward sensitivity.

    Fit as a linear mixed model with a random intercept per `group_col` (subject), since sessions
    are repeated measures within an animal rather than independent draws. An area pair whose
    surviving sessions come from a single subject falls back to OLS (with a console note), since
    a random intercept isn't identifiable from only one group.

    :param model_df: output of get_session_beta_rho_df.
    :param outcome: the behavioural coefficient column to use as the response (e.g. 'sess_intercept',
        'trial_order', 'effort_lev', 'trial_count'; see fit_session_behaviour).
    :param group_col: column identifying the repeated-measures grouping factor (animal).
    :return: pd.DataFrame, one row per area_pair, sorted by descending |beta|.
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
            # need at least 3 sessions for a meaningful slope + residual df
            continue
        n_subjects = df_p[group_col].nunique()
        # fit on a unit-SD outcome and scale beta/se back: the mixed model's optimiser is unreliable for coefficients of tiny
        # magnitude (e.g. per-trial slopes ~1e-3), and beta/se (but not p) are the only things that depend on the scaling
        y_scale = df_p[outcome].std()
        df_p = df_p.assign(**{outcome: df_p[outcome] / y_scale})
        if n_subjects > 1:
            fit = smf.mixedlm(formula, data=df_p, groups=df_p[group_col]).fit()
        else:
            print(f'{area_pair}: sessions come from a single subject ({df_p[group_col].iloc[0]}) -- '
                 f'a random intercept is not identifiable, falling back to OLS')
            fit = smf.ols(formula, data=df_p).fit()
        if not np.isfinite(fit.bse['corr_cv']):
            # singular mixed-model fit (animal variance estimated at zero, no standard errors): equivalent to OLS
            print(f'{area_pair} {outcome}: singular mixed-model fit (zero animal variance), falling back to OLS')
            fit = smf.ols(formula, data=df_p).fit()
        rows.append({
            'area_pair': area_pair,
            'outcome': outcome,
            'beta': fit.params['corr_cv'] * y_scale,
            'se': fit.bse['corr_cv'] * y_scale,
            'pval': fit.pvalues['corr_cv'],
            'n_sessions': df_p.shape[0],
            'n_subjects': n_subjects,
        })
    beta_df = pd.DataFrame(rows)
    beta_df['abs_beta'] = beta_df['beta'].abs()
    return beta_df.sort_values('abs_beta', ascending=False).reset_index(drop=True)


def regress_rho_on_beta_across_dims(cca_df: pd.DataFrame,
                                    session_beh_df: pd.DataFrame,
                                    outcome: str = 'sess_intercept',
                                    group_col: str = 'subject_id',
                                    session_col: str = 'day_subj',
                                    alpha: Optional[float] = None,
                                    ) -> pd.DataFrame:
    """
    Complementary analysis to regress_rho_on_beta: rather than testing only each area pair's
    dominant (rank 0) CCA dimension against `outcome`, this loops over every significant
    dimension (rank) of every area pair -- as determined the same way as elsewhere in this
    script, via st.get_cca_df_edge_params -- fitting the same subject-random-intercept mixed
    model (outcome ~ corr_cv, with corr_cv z-scored within each area_pair/rank as in
    regress_rho_on_beta) separately for each (area_pair, rank) combination. This asks
    whether a *higher-order* communication subspace (a CC dimension beyond each area pair's
    strongest one) also predicts `outcome` (by default 'sess_intercept'), rather than assuming
    only the dominant component can carry a behaviourally-relevant signal.

    This loop runs one test per (area_pair, rank) combination -- many more tests than the
    rank-0-only analyses above -- so the resulting beta p-values are Benjamini-Hochberg corrected
    (statsmodels fdr_bh) as a single family, separately from the per-dimension significance test
    that get_cca_df_edge_params already applies internally to decide which dimensions to test in
    the first place.

    :param cca_df: st.canonical_corr, with columns subject_day/area_pair/rank/corr_cv.
    :param session_beh_df: output of fit_session_behaviour.
    :param outcome: behavioural coefficient column regressed on each dimension's corr_cv.
    :param group_col: column identifying the repeated-measures grouping factor (animal).
    :param session_col: column identifying a single recording session in session_beh_df.
    :param alpha: FDR level for the BH correction, and the alpha passed to
        get_cca_df_edge_params to decide which dimensions count as significant; defaults to
        conf_dict['p_thresh'].
    :return: pd.DataFrame, one row per (area_pair, rank) tested, with columns area_pair, rank,
        beta, se, pval, pval_bh, sig, n_sessions, n_subjects -- sorted by pval_bh ascending.
    """
    alpha = conf_dict['p_thresh'] if alpha is None else alpha
    sig_dims = st.get_cca_df_edge_params(alpha, best_rank_only=False)
    sig_dims = sig_dims[sig_dims['edge_pval'] < 1]  # every (area_pair, rank) that cleared BH, not just rank 0
    rows = []
    for _, dim in sig_dims.iterrows():
        area_pair, rank = dim['area_pair'], dim['rank']
        rho_df = cca_df.loc[(cca_df['rank'] == rank) & (cca_df['area_pair'] == area_pair),
                            ['subject_day', 'corr_cv']].dropna()
        rho_sd = rho_df['corr_cv'].std()
        if not rho_sd > 0:
            continue
        rho_df = rho_df.assign(corr_cv=(rho_df['corr_cv'] - rho_df['corr_cv'].mean()) / rho_sd)
        df_p = session_beh_df.merge(rho_df, left_on=session_col, right_on='subject_day', how='inner')
        df_p = df_p.dropna(subset=[outcome, 'corr_cv', group_col])
        if df_p.shape[0] < 3:
            # need at least 3 sessions for a meaningful slope + residual df
            continue
        formula = f'{outcome} ~ corr_cv'
        n_subjects = df_p[group_col].nunique()
        if n_subjects > 1:
            fit = smf.mixedlm(formula, data=df_p, groups=df_p[group_col]).fit()
        else:
            print(f'{area_pair} rank {rank}: sessions come from a single subject '
                 f'({df_p[group_col].iloc[0]}) -- a random intercept is not identifiable, falling back to OLS')
            fit = smf.ols(formula, data=df_p).fit()
        rows.append({
            'area_pair': area_pair,
            'rank': rank,
            'beta': fit.params['corr_cv'],
            'se': fit.bse['corr_cv'],
            'pval': fit.pvalues['corr_cv'],
            'n_sessions': df_p.shape[0],
            'n_subjects': n_subjects,
        })
    dim_beta_df = pd.DataFrame(rows)
    if dim_beta_df.empty:
        print('regress_rho_on_beta_across_dims: no (area_pair, rank) combination had enough sessions to fit.')
        return dim_beta_df
    reject, pval_bh, _, _ = multipletests(dim_beta_df['pval'].values, alpha=alpha, method='fdr_bh')
    dim_beta_df['pval_bh'] = pval_bh
    dim_beta_df['sig'] = reject
    return dim_beta_df.sort_values('pval_bh').reset_index(drop=True)


def plot_area_beta_ranking(beta_df: pd.DataFrame, outcome: str, where=(4, 3), mark_significance: bool = True, ax=None):
    """
    Horizontal bar plot of the slope of the per-session `outcome` coefficient on each area pair's
    dominant CCA rho (output of regress_rho_on_beta), rank-ordered by slope and coloured by
    circuit (Fr-Fr/BG-BG/Fr-BG, via conf_dict['circuit_map']/['circuit_colour']). Bars at the
    top are the area pairs whose stronger coupling predicts the highest values of `outcome`
    (e.g. higher reward sensitivity for 'trial_order'); bars at the bottom predict the lowest.

    By default (mark_significance=True) every area pair is plotted and the ones that reached
    conf_dict['p_thresh'] are flagged with an asterisk, so the full hierarchy is visible alongside
    which of it is statistically supported. With mark_significance=False, area pairs that didn't
    clear that threshold are dropped from the ranking instead.

    :param beta_df: output of regress_rho_on_beta.
    :param outcome: which behavioural coefficient's slopes to plot (e.g. 'trial_order').
    :param mark_significance: if True, plot all area pairs and star the significant ones instead
        of dropping non-significant ones.
    :param ax: draw onto this existing Axes (e.g. a panel of a composite figure) instead of creating and saving a figure.
    :return: the Figure, or None if nothing to plot.
    """
    sub = beta_df[beta_df['outcome'] == outcome].copy()
    if not mark_significance:
        sub = sub[sub['pval'] < conf_dict['p_thresh']]
    sub = sub.sort_values('beta').reset_index(drop=True)
    if sub.empty:
        print(f"No area pairs to plot for {outcome} (p < {conf_dict['p_thresh']} gating: {not mark_significance}) "
              f"-- skipping plot")
        return None
    labels = [f"{fw.region_labels()[a.split('_')[0]]}-{fw.region_labels()[a.split('_')[1]]}"
             for a in sub['area_pair']]
    colours = [conf_dict['circuit_colour'][conf_dict['circuit_map'].get(a, 'Fr-BG')] for a in sub['area_pair']]
    own_fig = ax is None
    with mpl.rc_context(hs.rc0):
        if own_fig:
            ax = hs.get_ax(where=where)
        y = np.arange(sub.shape[0])
        ax.barh(y, sub['beta'], color=[mpl.colors.to_rgba(c, .7) for c in colours], edgecolor='k', linewidth=.5, zorder=2)
        ax.errorbar(sub['beta'], y, xerr=sub['se'], fmt='none', ecolor='k', elinewidth=.6, capsize=1.5, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.axvline(0, ls='--', color='grey', lw=0.8)
        var_lab = approximate_var_label(outcome)
        var_lab = var_lab if var_lab != outcome else outcome.replace('_', ' ')  # unmapped names: just drop underscores
        ax.set_xlabel(rf"$\beta_{{\mathrm{{{var_lab.replace(' ', chr(92) + ' ')}}}}}$ ~ z-scored dominant CCA $\rho$")
        ax.set_ylabel('Area')
        if mark_significance:
            x_pad = 0.02 * sub['beta'].abs().max()
            for yi, (beta, se, pval) in enumerate(zip(sub['beta'], sub['se'], sub['pval'])):
                if pval < conf_dict['p_thresh']:
                    ax.text(beta + np.sign(beta) * (se + x_pad), yi, '*',
                           ha='left' if beta >= 0 else 'right', va='center',
                           fontsize=12, fontweight='bold')
        legend_handles = [mpl.patches.Patch(facecolor=mpl.colors.to_rgba(c, .7), edgecolor='k', linewidth=.5)
                          for c in conf_dict['circuit_colour'].values()]
        ax.legend(legend_handles, list(conf_dict['circuit_colour'].keys()), frameon=False, fontsize=6,
                  loc='upper left')
        ax.spines['left'].set_bounds(0, len(y) - 1)  # truncate the category axis at its first and last ticks
        hs.displace_axes(ax, .1)
        fig = ax.get_figure()
        if own_fig:
            hs.savefig(f'{out_root}_{outcome}_on_rho_area_ranking.svg', fig=fig, dpi=300)
            hs.savefig(f'{out_root}_{outcome}_on_rho_area_ranking.png', open=OPEN_FIGS, fig=fig, dpi=300)
    return fig

def plot_session_scatter(model_df: pd.DataFrame, area_pair: str = 'stn_str', outcome: str = 'trial_order',
                         group_col: str = 'subject_id', where=(3.6, 3.4), ax=None):
    """
    Session-level scatter of the per-session behavioural coefficient `outcome` (y) against that session's dominant CCA rho
    for `area_pair` (x), one dot per session coloured by animal (the 'Reds' shades and black-edged dots used in project
    553431). The line and shaded 95% CI are the population-level (fixed-effect) fit of the same random-intercept mixed model
    used by regress_rho_on_beta (`outcome ~ z-scored rho`, random intercept per animal; OLS if only one animal), drawn
    against raw rho; the fitted beta (per SD of that pair's rho, matching the area-ranking plots) is printed to the console.

    :param model_df: output of get_session_beta_rho_df.
    :param ax: draw onto this existing Axes (e.g. a panel of a composite figure) instead of creating and saving a figure.
    :return: the Figure.
    """
    sub = model_df[model_df['area_pair'] == area_pair].dropna(subset=['corr_cv', outcome, group_col]).copy()
    mu, sd = sub['corr_cv'].mean(), sub['corr_cv'].std()
    sub['rho_z'] = (sub['corr_cv'] - mu) / sd
    y_scale = sub[outcome].std()  # fit on a unit-SD outcome (see regress_rho_on_beta) and scale back for plotting
    sub['y_unit'] = sub[outcome] / y_scale
    formula = 'y_unit ~ rho_z'
    n_subjects = sub[group_col].nunique()
    fit = (smf.mixedlm(formula, data=sub, groups=sub[group_col]).fit() if n_subjects > 1
           else smf.ols(formula, data=sub).fit())
    x = np.linspace(sub['corr_cv'].min(), sub['corr_cv'].max(), 100)
    design = np.column_stack([np.ones_like(x), (x - mu) / sd])
    names = ['Intercept', 'rho_z']
    cov = fit.cov_params().loc[names, names].to_numpy()
    y_hat = design @ fit.params[names].to_numpy() * y_scale
    y_se = np.sqrt(np.einsum('ij,jk,ik->i', design, cov, design)) * y_scale
    subjects = sorted(sub[group_col].unique())
    shades = mpl.colormaps['Reds'](np.linspace(.15, .9, len(subjects)))
    labs = fw.region_labels()
    a1, a2 = area_pair.split('_')
    own_fig = ax is None
    with mpl.rc_context(hs.rc0):
        if own_fig:
            ax = hs.get_ax(where=where)
        ax.fill_between(x, y_hat - 1.96 * y_se, y_hat + 1.96 * y_se, color='#868E96', alpha=.25, lw=0, zorder=1)
        ax.plot(x, y_hat, color='k', lw=1.2, zorder=2)
        for subj, colour in zip(subjects, shades):
            d = sub[sub[group_col] == subj]
            ax.scatter(d['corr_cv'], d[outcome], s=24, color=colour, edgecolor='k', lw=.6, zorder=3,
                       label=f'{subj} (n={len(d)})')
        ax.set_xlabel(rf'Dominant CCA $\rho$ ({labs[a1]}-{labs[a2]})')
        ax.set_ylabel(conf_dict['outcome_labels'].get(outcome, outcome))
        ax.legend(title='Animal', loc='best', frameon=True, fontsize=5, title_fontsize=6)
        fig = ax.get_figure()
        if own_fig:
            hs.displace_axes(ax, .1)
            hs.savefig(f'{out_root}_{area_pair}_{outcome}_session_scatter.svg', fig=fig, dpi=300, legend=False)
            hs.savefig(f'{out_root}_{area_pair}_{outcome}_session_scatter.png', open=OPEN_FIGS, fig=fig, dpi=300, legend=False)
    print(f'{area_pair} {outcome}: {sub.shape[0]} sessions, {n_subjects} animals, beta per SD rho = '
          f"{fit.params['rho_z'] * y_scale:+.4g} (95% CI {fit.conf_int().loc['rho_z', 0] * y_scale:+.4g} to "
          f"{fit.conf_int().loc['rho_z', 1] * y_scale:+.4g}), p = {fit.pvalues['rho_z']:.4f}")
    return fig


def plot_rho_by_sensitivity_quartile(model_df: pd.DataFrame, area_pair: str = 'stn_str', outcome: str = 'trial_order',
                                     group_col: str = 'subject_id', n_bins: int = 4, ax=None, legend: bool = True):
    """
    Dominant CCA rho of `area_pair` across quartiles of the sessions' behavioural coefficient `outcome` (default: reward
    sensitivity), as box plots (median, IQR, 1.5 IQR whiskers) over jittered session dots coloured by animal (the 'Reds'
    shades of project 553431).
    Left: quartiles of the pooled sessions. As animals differ systematically in reward sensitivity (and in mean rho), pooled
    quartiles are partly an animal contrast, so the right panel defines each session's quartile *within its own animal* and
    plots rho minus that animal's mean rho, isolating the within-animal relationship. The Kruskal-Wallis test across the
    quartiles and the Spearman correlation between quartile and rho are printed to the console.

    :param model_df: output of get_session_beta_rho_df.
    :param ax: draw only the pooled-sessions panel onto this existing Axes (e.g. a panel of a composite figure) instead of
        creating and saving the two-panel figure.
    :param legend: whether to draw the animal legend.
    :return: the Figure.
    """
    from scipy.stats import kruskal, spearmanr
    sub = model_df[model_df['area_pair'] == area_pair].dropna(subset=['corr_cv', outcome, group_col]).copy()
    sub['q_pooled'] = pd.qcut(sub[outcome].rank(method='first'), n_bins, labels=False)
    sub['q_within'] = np.floor(sub.groupby(group_col)[outcome].rank(method='first', pct=True).sub(1e-9) * n_bins).astype(int)
    sub['rho_within'] = sub['corr_cv'] - sub.groupby(group_col)['corr_cv'].transform('mean')
    subjects = sorted(sub[group_col].unique())
    shades = dict(zip(subjects, mpl.colormaps['Reds'](np.linspace(.15, .9, len(subjects)))))
    labs = fw.region_labels()
    a1, a2 = area_pair.split('_')
    panels = [('q_pooled', 'corr_cv', rf'Dominant CCA $\rho$ ({labs[a1]}-{labs[a2]})', 'Pooled sessions'),
              ('q_within', 'rho_within', r'$\rho$ minus animal mean', 'Quartiles within animal')]
    rng = np.random.default_rng(0)
    xlab = {0: 'Q1\n(lowest)', n_bins - 1: f'Q{n_bins}\n(highest)'}
    own_fig = ax is None
    with mpl.rc_context(hs.rc0):
        if own_fig:
            fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.3))
        else:
            fig, axes, panels = ax.get_figure(), [ax], panels[:1]
        for ax, (qcol, ycol, ylab, title) in zip(axes, panels):
            groups = [sub.loc[sub[qcol] == q, ycol].to_numpy() for q in range(n_bins)]
            ax.boxplot(groups, positions=range(n_bins), widths=.5, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor='#F1F3F5', edgecolor='k', lw=.8), medianprops=dict(color='k', lw=1.4),
                       whiskerprops=dict(lw=.8), capprops=dict(lw=.8), zorder=1)
            for subj in subjects:
                d = sub[sub[group_col] == subj]
                ax.scatter(d[qcol] + rng.uniform(-.17, .17, len(d)), d[ycol], s=16, color=shades[subj], edgecolor='k',
                           lw=.5, zorder=3, label=f'{subj} (n={len(d)})')
            kw = kruskal(*groups)
            rs = spearmanr(sub[qcol], sub[ycol])
            ax.set_xticks(range(n_bins))
            ax.set_xticklabels([xlab.get(q, f'Q{q + 1}') + f'\n(n={len(groups[q])})' for q in range(n_bins)], fontsize=6)
            ax.set_xlabel(f'{conf_dict["outcome_labels"].get(outcome, outcome)} quartile')
            ax.set_ylabel(ylab)
            ax.set_title(title, fontsize=8)
            print(f'{area_pair} {title}: group medians ' + ', '.join(f'Q{q + 1} {np.median(g):+.3f}' for q, g in enumerate(groups)) +
                  f' | Kruskal-Wallis p = {kw.pvalue:.4f}; Spearman rs = {rs.statistic:+.3f}, p = {rs.pvalue:.4f}')
        if legend:
            axes[0].legend(title='Animal', loc='lower left', frameon=True, fontsize=5, title_fontsize=6)
        if own_fig:
            fig.tight_layout()
            hs.savefig(f'{out_root}_{area_pair}_rho_by_{outcome}_quartile.svg', fig=fig, dpi=300, legend=False)
            hs.savefig(f'{out_root}_{area_pair}_rho_by_{outcome}_quartile.png', open=OPEN_FIGS, fig=fig, dpi=300, legend=False)
    return fig


def compute_session_choice_curves(beh_df: pd.DataFrame, session_beh_df: pd.DataFrame, outcome: str,
                                  sessions: Optional[Iterable] = None, session_col: str = 'day_subj') -> pd.DataFrame:
    """
    Each session's fitted P(accept) as a function of one regressor of the per-session choice GLM (fit_session_behaviour),
    evaluated at that session's observed levels of `outcome` ('trial_order' = reward, 'effort_lev' = effort) with the other
    two regressors held at that session's median -- the same convention as the indifference curves of project 284485.

    :param beh_df: trial-level behaviour dataframe (BehDf().beh_df).
    :param session_beh_df: output of fit_session_behaviour (its 'fit' column holds each session's GLM).
    :param sessions: restrict to these sessions (default: every session with a fit).
    :return: long DataFrame [session_col, subject_id, level, p_accept].
    """
    others = [v for v in ('trial_order', 'effort_lev', 'trial_count') if v != outcome]
    sel = session_beh_df.dropna(subset=['fit'])
    if sessions is not None:
        sel = sel[sel[session_col].isin(list(sessions))]
    rows = []
    for _, r in sel.iterrows():
        df_s = beh_df[beh_df[session_col] == r[session_col]].dropna(subset=['far_mag_entry', 'trial_order', 'effort_lev',
                                                                         'trial_count'])
        levels = np.sort(df_s[outcome].unique())
        pred = pd.DataFrame({outcome: levels, **{v: df_s[v].median() for v in others}})
        rows.append(pd.DataFrame({session_col: r[session_col], 'subject_id': r['subject_id'], 'level': levels,
                                  'p_accept': r['fit'].predict(pred).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def plot_session_choice_curves(curves: pd.DataFrame, outcome: str, ax, session_col: str = 'day_subj'):
    """
    Fitted P(accept) against one regressor: a thin line per session (coloured by animal, the 'Reds' shades of project 553431)
    and a thick black line for the cohort mean across sessions at each level, with a one-row legend above the axes
    explaining the line colours.

    :param curves: output of compute_session_choice_curves.
    :param outcome: the regressor on the x axis (labelled via approximate_var_label).
    :param ax: Axes to draw on.
    """
    subjects = sorted(curves['subject_id'].unique())
    shades = dict(zip(subjects, mpl.colormaps['Reds'](np.linspace(.15, .9, len(subjects)))))
    for _, g in curves.groupby(session_col):
        ax.plot(g['level'], g['p_accept'], color=shades[g['subject_id'].iloc[0]], lw=.6, alpha=.75, zorder=1)
    mean = curves.groupby('level')['p_accept'].mean()
    ax.plot(mean.index, mean.values, color='k', lw=2.2, zorder=3)
    ax.axhline(.5, ls='--', color='grey', lw=.6, zorder=0)
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(curves['level'].unique()))
    ax.set_xlabel(approximate_var_label(outcome).capitalize())
    ax.set_ylabel(r'$P_{accept}$')
    handles = [Line2D([0], [0], color=shades[subj], lw=1) for subj in subjects] + [Line2D([0], [0], color='k', lw=2.2)]
    ax.legend(handles, subjects + ['Mean'], ncol=len(handles), loc='lower left', bbox_to_anchor=(-.1, 1.07), frameon=False,
              fontsize=5, columnspacing=.7, handlelength=1.1, handletextpad=.3, borderaxespad=.2)


def rescale_svg_canvas(path: str, scale: float):
    """
    Shrink (or grow) the nominal size of a saved SVG by `scale` while leaving its viewBox untouched, so the whole drawing -
    fonts, strokes, markers - is scaled uniformly when the file is opened or placed at its stated size.
    """
    import re
    svg = open(path, encoding='utf-8').read()
    head = re.search(r'<svg[^>]*>', svg).group(0)
    new_head = re.sub(r'(width|height)="([\d.]+)pt"', lambda m: f'{m.group(1)}="{float(m.group(2)) * scale:.2f}pt"', head)
    open(path, 'w', encoding='utf-8').write(svg.replace(head, new_head, 1))


def plot_sensitivity_summary(beta_df: pd.DataFrame, model_df: pd.DataFrame, session_beh_df: pd.DataFrame,
                             beh_df: pd.DataFrame, area_pair='stn_str',
                             outcomes: Iterable[str] = ('trial_order', 'effort_lev'), legend_fontsize: float = 3,
                             label_fontsize: float = 4, letter_fontsize: float = 10, final_scale: float = .5,
                             rescale_svg: bool = True):
    """
    Composite figure with one row per behavioural coefficient in `outcomes` (default: reward then effort sensitivity), each
    row made of four panels drawn by their own plotting functions: (a) every session's fitted choice curve against the
    regressor, with the cohort mean (plot_session_choice_curves), (b) the area-pair hierarchy of the slope of the outcome on
    z-scored dominant CCA rho (plot_area_beta_ranking), (c) the session scatter and mixed-model fit for the area pair
    (plot_session_scatter), (d) rho across pooled quartiles of the outcome for the area pair (first panel of
    plot_rho_by_sensitivity_quartile). The choice curves show the same sessions as the scatter and quartile panels.

    :param beta_df: output of regress_rho_on_beta (concatenated over outcomes is fine).
    :param model_df: output of get_session_beta_rho_df.
    :param session_beh_df: output of fit_session_behaviour.
    :param beh_df: trial-level behaviour dataframe (BehDf().beh_df).
    :param area_pair: the area pair shown in the scatter and quartile panels; a single pair for every row, or a dict
        {outcome: pair} to pick a different pair per row.
    :param legend_fontsize, label_fontsize, letter_fontsize: font sizes (pt) of the legend text (incl. legend titles), the axis
        labels and the bold panel letters *as they will appear in the final layout*.
    :param final_scale: factor by which the saved figure is reduced in the final layout (0.5: the 14.2 in wide figure is
        printed about 7 in wide), so the fonts above are drawn at size / final_scale in the saved file.
    :param rescale_svg: also shrink the saved SVG's stated size by final_scale (see rescale_svg_canvas), so it opens at the
        final size and its fonts read as legend_fontsize/label_fontsize/letter_fontsize pt. The PNG keeps the full size.
    :return: the created matplotlib Figure.
    """
    outcomes = list(outcomes)
    with mpl.rc_context(hs.rc0):
        fig, axes = plt.subplots(len(outcomes), 4, figsize=(14.2, 3.5 * len(outcomes)), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[.85, 1.05, 1, 1]))
        for row, outcome in zip(axes, outcomes):
            pair = area_pair[outcome] if isinstance(area_pair, dict) else area_pair
            curves = compute_session_choice_curves(beh_df, session_beh_df, outcome,
                                                   sessions=model_df.loc[model_df['area_pair'] == pair, 'day_subj'])
            plot_session_choice_curves(curves, outcome, row[0])
            plot_area_beta_ranking(beta_df, outcome=outcome, mark_significance=True, ax=row[1])
            plot_session_scatter(model_df, area_pair=pair, outcome=outcome, ax=row[2])
            plot_rho_by_sensitivity_quartile(model_df, area_pair=pair, outcome=outcome, ax=row[3], legend=False)
            row[3].set_title('')
        offsets = {0: -.3, 1: -.35, 2: -.2, 3: -.2}
        for ax, letter in zip(axes.flat, 'abcdefghijkl'):
            ax.xaxis.label.set_fontsize(label_fontsize / final_scale)
            ax.yaxis.label.set_fontsize(label_fontsize / final_scale)
            if ax.get_legend() is not None:
                for text in [*ax.get_legend().get_texts(), ax.get_legend().get_title()]:
                    text.set_fontsize(legend_fontsize / final_scale)
            ax.text(offsets[ax.get_subplotspec().colspan.start], 1.06, letter, transform=ax.transAxes,
                    fontsize=letter_fontsize / final_scale, fontweight='bold', va='bottom')
        fig.tight_layout()
        hs.savefig(f'{out_root}_sensitivity_summary.svg', fig=fig, dpi=300, legend=False)
        if rescale_svg:
            rescale_svg_canvas(f'{out_root}_sensitivity_summary.svg', final_scale)
        hs.savefig(f'{out_root}_sensitivity_summary.png', open=OPEN_FIGS, fig=fig, dpi=300, legend=False)
    return fig


#%%
# Per-session choice behaviour, then regress each behavioural coefficient on the session's dominant CCA rho
beh_df = load_trial_behaviour(809107).beh_df  # implanted-animal trial behaviour (tethered_behaviour_1)
session_beh_df = fit_session_behaviour(beh_df, conf_dict['formula'])
sig_edges = st.get_cca_df_edge_params(conf_dict['p_thresh'])
sig_edges = sig_edges[(sig_edges['rank'] == 0) & (sig_edges['edge_pval'] < 1)]
model_df = get_session_beta_rho_df(session_beh_df, st.canonical_corr, sig_edges['area_pair'].unique())

single_beta_dfs = {}
for outcome in ('sess_intercept', 'trial_order', 'effort_lev', 'trial_count'):
    single_beta_dfs[outcome] = regress_rho_on_beta(model_df, outcome=outcome)
    print(single_beta_dfs[outcome])
    plot_area_beta_ranking(single_beta_dfs[outcome], outcome=outcome, mark_significance=True)
beta_df = pd.concat(single_beta_dfs.values(), ignore_index=True)

#%%
# Frequency of light-gate crossings on rejected trials
rej = beh_df[beh_df['far_mag_entry'] == 0]
crossed = rej[['lg2_lat', 'lg6_lat']].notna()  # True where the light gate was crossed
print(f'\nLight-gate entries on rejected trials (n = {len(rej)} trials):\n{crossed.agg(["sum", "mean"])}')

#%%
# Does a higher-order (non-dominant) CC dimension within an area pair also correspond to the session intercept, not just
# that area pair's dominant rank-0 component?
dim_beta_df = regress_rho_on_beta_across_dims(st.canonical_corr, session_beh_df, outcome='sess_intercept')
summary_cols = ['area_pair', 'rank', 'beta', 'se', 'pval', 'pval_bh', 'sig', 'n_sessions', 'n_subjects']
print(dim_beta_df[summary_cols].to_string(index=False))

#%%
# FIGURE 4 (panels a-h): choice curves, area hierarchy, session scatter and rho across sensitivity quartiles, for the
# reward row (a-d, STA-DMS) and the effort row (e-h, MO/VO-DMS)
plot_sensitivity_summary(beta_df, model_df, session_beh_df, beh_df,
                         area_pair={'trial_order': 'stn_str', 'effort_lev': 'mo_str'},
                         outcomes=('trial_order', 'effort_lev'), final_scale=.35)
