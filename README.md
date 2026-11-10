# Shared cortico-basal ganglia population activity: analysis code

Code to reproduce the statistics, figures and supplemental figures/tables of:

Title: Dynamic information sharing across frontal-basal ganglia neurons during cost-benefit decision-making and reward pursuit

Authors: Oliver Härmson<sup>1</sup>, Isaac Grennan<sup>1</sup>, Brook Perry<sup>1</sup>, Robert Toth<sup>2</sup>, Mark E. Walton<sup>3,4</sup>, Sanjay G. Manohar<sup>3,5</sup>, Andrew Sharott<sup>1, 6, 7</sup>

<sup>1</sup>Medical Research Council Centre of Research Excellence in Restorative Neural Dynamics, Nuffield Department of Clinical Neurosciences, University of Oxford, Oxford, United Kingdom.
<sup>2</sup>Department of Engineering Science, University of Oxford, Oxford, OX1 3PJ, United Kingdom.
<sup>3</sup>Department of Experimental Psychology, University of Oxford, Oxford, United Kingdom.
<sup>4</sup>Wellcome Centre for Integrative Neuroimaging, University of Oxford, Oxford, United Kingdom.
<sup>5</sup>Nuffield Department of Clinical Neurosciences, University of Oxford, Oxford, United Kingdom.
<sup>6</sup>These authors contributed equally to this work
<sup>7</sup>Lead Contact 

The analyses ask which pairs of frontal and basal ganglia areas share population activity (canonical correlation
analysis, CCA, of simultaneously recorded single units in rats performing a cost-benefit decision task), how many
dimensions they share, what task variables (reward, effort, trial number, running speed) those dimensions carry,
whether that coding differs between accepted and rejected offers, and whether a session's connectivity relates to
that session's choice behaviour.

`data/` holds two kinds of files (see [Data](#data)): **processed tables** produced by the pipeline that precedes these
scripts (unit firing-rate tables, CCA results, decoder fits), which are the inputs of the statistics and figures, and the
**source files** needed to regenerate the running-speed table of Fig. S3 (event channels and behaviour of the recording
stages, and the DeepLabCut tracking of the video). The scripts regenerate everything derived from the source files and
the processed tables that the supplemental figures need (running-speed table, firing-rate matrices of the heatmaps,
bootstrapped ICCs, variance table) and write it to `output/`.

**What is not included:** the code that produced the processed tables from the raw recordings (spike sorting, trial
alignment and firing-rate binning of the unit tables, the CCA and decoder fits with their permutation nulls). Those tables
are shipped as they were used for the paper and cannot be regenerated from this repository.

## Repository layout

```
costbenefit-cca/
├── projects/        one script per analysis project (see below)
├── lib/             the small library the scripts use (plotting helpers, SpikeStats loader, constants)
├── data/            processed tables and source files, named <project id>_<name>.<ext>  (~5.2 GB)
├── pngs/            epoch icons and the brain-slice drawing used by the network figures
├── output/          everything the scripts generate: figures, tables and cached intermediates (created on first run)
└── environment.yml  Python environment
```

Each analysis project has a six-digit **project id** that prefixes its files in `data/` and its outputs in
`output/`, e.g. `data/382702_canonical_corr.feather` is the CCA result of project 382702.

## Getting started

* Download the data on https://doi.org/10.60964/rnd-1m60-zh48.

```bash
conda env create -f environment.yml
conda activate cca-paper
python projects/cca_dimensionality_382702.py
```

* Run the scripts from any directory; they find `lib/`, `data/` and `pngs/` relative to their own location. To keep
  the data or outputs elsewhere, set `CCA_DATA_DIR` / `CCA_OUTPUT_DIR`.
* The scripts are written in cell style (`#%%`) and can equally be run cell by cell in Spyder, PyCharm or VS Code.
* Figures are written to `output/` as `.png` (and `.svg` where the script saves one). Statistics are printed to the
  console. Set `OPEN_FIGS = True` at the top of a script to also open each saved PNG in the system viewer.
* Run times on a laptop: under a minute for the first three scripts. The first run of `supplementals_203265.py` takes
  about 15 minutes and ~10 GB of memory: it loads the 0.9 GB unit table of project 480152, extracts the running-speed
  table from the 663 tracking files, rebuilds the heatmap matrices and the variance table of Fig. S3d. These are cached
  in `output/` (`203265_vel_df_table_v2`, `480152_fr_mats/`, `203265_ov_var_df`, `203265_icc_*`), so later runs take ~6
  minutes. Delete a cached file to regenerate it, or set `RECOMPUTE_VELOCITY` / `RECOMPUTE_ICC` at the top of the script.
  The bootstrapped ICCs of Fig. S2 (5,000 resamples x 10 area pairs) are the slowest step of a first run (not
  timed; expect well over half an hour).

## Scripts

| Script | Project | Reads (in `data/`) | Produces |
|---|---|---|---|
| `projects/cca_dimensionality_382702.py` | 382702 | `382702_canonical_corr` | **Figure 2 - CCA dimensionality.** Per epoch and area pair, the cross-validated canonical correlations are tested against their permutation null (Wilcoxon signed-rank across sessions, Benjamini-Hochberg FDR). Panels: network of area pairs with significant dimensions per epoch (a), bootstrapped Δρ per circuit class (b), significant components per session by epoch and circuit class (c) → `382702_all_area_pair_cca_corr_ndim`. Statistics: two-way repeated-measures ANOVA (circuit class × epoch) on the number of significant components, with BH-corrected post-hoc tests. |
| `projects/decode_from_cc_space_624290.py` | 624290 (+ 258066, 382702) | `624290_regression`, `258066_regression`, `382702_canonical_corr` | **Figure 3 - task decoding from the shared subspaces.** Cross-validated R² of reward, effort, trial count and running speed decoded from the CC components (against a permutation null); angles between the decoders' CC beta vectors (reward vs effort, and reward vs velocity / trial count; mixed models over animals and sessions, with equivalence tests); accept-vs-reject decoder comparison (project 258066). Figure 3 → `624290_regression_r^2_breakdown`. The angle analyses are additional (not main-figure panels) → `624290_trial_order_effort_lev_angle_breakdown`, `624290_task_var_pair_angles`. |
| `projects/connectivity_behaviour_relationship_746484.py` | 746484 (+ 809107) | `746484_canonical_corr`, `809107_TrialBehaviour` | **Figure 4 - connectivity-behaviour relationship.** Per session, a binomial GLM of accept/reject choice on reward, effort and trial count; each coefficient is regressed (mixed model, animal as grouping factor) on the session's dominant CCA ρ for every area pair with a significant first dimension. Prints the coefficients/statistics and writes Figure 4 → `746484_sensitivity_summary`, and the area-pair hierarchy for every coefficient → `746484_<coefficient>_on_rho_area_ranking`. |
| `projects/supplementals_203265.py` | 203265 (+ 480152, 382702, 624290, 258066, 746484, 781039, 809107, 828817, 958690, 773894) | see the script header | **Supplemental Figures S1-S4 and Table S1.** S1: firing-rate heatmaps, units per day and area, sessions per area pair and animal, combined daily unit yield per area pair and circuit class (with the mixed-model comparison of yields). S2: between-animal reliability of the dominant canonical correlation (ICC). S3: axes of the replacing task variables, running speed, accept-vs-reject decoding, variance of the true task variables. S4: dominant ρ against the trial-count coefficient of the per-session choice GLM. → `203265_fig_S1` … `203265_fig_S4`, `203265_table_S1.csv`. |

## Figure map

Where each figure and panel of the manuscript is produced. Panel letters are the ones drawn on the saved figures. Output
files are written to `output/` (`.png`, and `.svg` where the script saves one).

| Figure | Panel | Content | Script | Output file |
|---|---|---|---|---|
| **2** | a | Network of area pairs with significant shared dimensions, per epoch | `cca_dimensionality_382702.py` | `382702_all_area_pair_cca_corr_ndim` |
| | b | Bootstrapped Δρ per circuit class, per epoch | | |
| | c | Significant CCA components per session, by epoch and circuit class | | |
| **3** | a | Network of area pairs whose task-variable decoding exceeds chance, per epoch | `decode_from_cc_space_624290.py` | `624290_regression_r^2_breakdown` |
| | b | Decoding R² by task variable and epoch (bars: mean across sessions; dots: sessions) | | |
| | c | Composition of the decoded variance by task variable and epoch | | |
| | d | Accept-vs-reject decoder comparison (Δr), with per-epoch swarms as insets | | |
| **4** | a-d | Reward sensitivity (STA-DMS): a choice curves, b area-pair hierarchy, c session scatter with mixed-model fit, d ρ across sensitivity quartiles | `connectivity_behaviour_relationship_746484.py` | `746484_sensitivity_summary` |
| | e-h | Effort sensitivity (MO/VO-DMS): e choice curves, f area-pair hierarchy, g session scatter, h ρ across sensitivity quartiles | | |
| **S1** | a | Firing-rate heatmaps of all units, per area and epoch | `supplementals_203265.py` | `203265_fig_S1` |
| | b | Units per recording day by area | | |
| | c | Recordings per area pair and animal | | |
| | d | Combined daily unit yield per area pair and circuit class | | |
| **S2** | a | Dominant canonical correlation per area pair for each animal (networks) | `supplementals_203265.py` | `203265_fig_S2` |
| | b | Between-animal agreement (ICC) of the dominant canonical correlation | | |
| | c | Cross-validated ρ₁ per area pair, by animal | | |
| **S3** | a | Axes of the replacing task variables relative to earlier reward coding | `supplementals_203265.py` | `203265_fig_S3` |
| | b | Running speed by epoch | | |
| | c | Accept-vs-reject decoding (Δr) in the corridor epochs | | |
| | d | Accept-vs-reject variance of the true task variables | | |
| **S4** | a-d | Trial-count sensitivity (for the area pair with the smallest p-value, ACC-STA): a choice curves, b area-pair hierarchy, c session scatter, d ρ across quartiles | `supplementals_203265.py` | `203265_fig_S4` |
| Table S1 | | Fisher-z-averaged cross-validated ρ of the leading canonical dimension for every animal, epoch and area pair (the values shown as edge labels in Fig. S2a) | `supplementals_203265.py` | `203265_table_S1.html`, `203265_table_s1.csv` |

The reward-vs-effort and pair-angle analyses of `decode_from_cc_space_624290.py` are not main-figure panels.

## Data

Data files are available at: https://doi.org/10.60964/rnd-1m60-zh48

Files are in `data/` and named `<project id>_<name>`.

**Source files** (used to regenerate the running-speed table, `VelDf` in `supplementals_203265.py`):

| Project | Files | Contents |
|---|---|---|
| 958690 | `_DinChannels.pbz2` | event channels (digital inputs) of every recording stage: task events, nose pokes, light gates, ... |
| | `_FineBehaviour.pbz2` | behavioural state sequences of every stage (used to cut the task epochs) |
| | `_TrialBehaviour.pbz2`, `_SpikeStats.pbz2` | trial table of the same stages and the analysis container the epoch windows are cut with |
| 773894 | `_xy_tables/` | 663 DeepLabCut tracking tables (animal position in pixels, 100 Hz) of the overhead video, one per stage |
| 809107 | `_TrialBehaviour.pbz2` | trial-by-trial behaviour of the implanted animals |

**Processed tables** produced upstream of these scripts (inputs; not regenerable from this repository):

| Project | Files | Contents |
|---|---|---|
| 480152 | `_spike_times.feather` | time-resolved firing rates of all units (25 ms bins, causal exponential smoothing, tau = 50 ms) per epoch, accepted trials |
| 828817 | `_spike_times.feather` | the same for rejected trials |
| 382702 | `_canonical_corr.feather` | cross-validated CCA between all pairs of areas, per session, epoch and canonical dimension, with permutation null |
| 624290 | `_regression.feather` | decoding of task variables from the CC components (R², per-component betas) |
| 258066 | `_regression.feather`, `_xy_table.feather` | decoder trained on accepted + rejected trials, scored separately for each; per-trial running speed |
| 746484 | `_canonical_corr.feather` | CCA over the decision window of accepted/rejected trials |
| 781039 | `_replacement_summary.csv` | summary of the replacing task variables (Fig. S3a) |

The four largest files (`773894_xy_tables/`, `958690_DinChannels.pbz2`, `480152_spike_times.feather`, `828817_spike_times.feather`;
~5 GB together) are listed in `.gitignore` and are not tracked by git; obtain them from https://doi.org/10.60964/rnd-1m60-zh48 and place them in `data/`.

`*.pbz2` files are bz2-compressed Python pickles. **Do not unpickle files from sources you do not trust.**

Subjects: gh001, gh009, gh010, oh011. Areas: MO/VO, ACC (frontal); DMS, STA, VP (basal ganglia); circuit classes Fr-Fr,
Fr-BG and BG-BG. Epochs: PRE (pre-cue), CUE, NP_EX (nose-poke exit/action), CORR1 and CORR6 (corridor light gates
1 and 6), DEL (pre-reward delay, accepted trials only).

## Notes on reproducibility

* Randomised procedures (bootstrap resamples, trial subsampling, null draws) are seeded in the scripts.
* The figures in this repository were generated with the versions in `environment.yml`.
* The code in `lib/` is a trimmed version of the lab's analysis library: only the functions needed to run the four
  scripts are included.

## Citation and licence

If you use this code, please cite the paper above. The code is released under the MIT licence (see `LICENSE`). The data files in `data/` are released under the [Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

Contact: Dr. Oliver Härmson, oliver.harmson@some.ox.ac.uk
