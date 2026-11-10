# -*- coding: utf-8 -*-
# Created by Oliver Haermson  03/2020
# Copyright © 2020-2026 University of Oxford. All rights reserved.
# Distributed under the MIT license; see LICENSE.
"""
Data containers for the analysis projects (trimmed to what the scripts in ``projects/`` use).

``BlockObj.SpikeStats`` loads a project's processed tables (``data/<project id>_<table>.feather``) and provides the helpers
the scripts call on it: sorting of epochs/areas, the firing-rate criterion, the CCA edge statistics
(``get_cca_df_edge_params``), the firing-rate matrices of the heatmaps, the network/heatmap plotting functions and
``TrialWindow``, which cuts task-epoch windows out of a stage's behavioural events. ``BlockObj.TrialBehaviour`` holds the
trial-level behaviour table (``beh_df``), ``BlockObj.DinChannels`` / ``FineBehaviour`` the event channels and behavioural
state sequences of the recording stages and ``BlockObj.Video`` the video-tracking geometry (pixels to cm, speed).
"""
from __future__ import division
from __future__ import print_function

import os.path
import re
import ray
import time
import numbers
import numpy as np
import pandas as pd
import helpers as hs
import seaborn as sns
import framework as fw
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from matplotlib.lines import Line2D
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from matplotlib.collections import LineCollection

plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_style.mplstyle'))

# ------------ Global variables ------------ #
# These specify which types of data can be analysed


var_mapping = {'trial_order': {0: 'nil', 1: 'small', 2: 'medium', 3: 'large'},
               'rew_offer': {0: 'nil', 1: 'small', 3: 'medium', 6: 'large'},
               'forced_order': {0: 'Choice', 1: 'Forced'},
               'effort_lev': {0: '0 barriers', 1: '1 barrier', 2: '2 barriers', 3: '3 barriers'},
               'tmin1_far_rew': {0: 't-1 reject', 1: 't-1 accept'},
               'tmin1_fm': {0: 'reject', 1: 'accept'},
               'tr_spec': {0: 'reject', 1: 'accept'},
               'decision': {0: 'reject', 1: 'accept'},
               'var_names': {'far_mag_entry': 'Far magazine entry', 'corridor': 'Corridor entry'}}
var_labels = {'trial_order': 'reward',
              'rew_offer': 'reward',
              'treatment': 'satiety',
              'Intercept': 'intercept',
              '(Intercept)': 'intercept',
              'np.power(trial_order, 2)': 'reward^2',
              'trial_order_cpd': 'reward',
              'trial_order_beta': 'reward',
              'effort_lev': 'effort',
              'np.power(effort_lev, 2)': 'effort^2',
              'effort_lev_cpd': 'effort',
              'effort_lev_beta': 'effort',
              'corr_cv': r'cross-validated $\rho$',
              'corr_cv_perm_avg': r'permuted cross-validated $\rho$',
              'tmin1_reward': 'previous reward',
              'tmin1_trial': 'reward offer t-1',
              'tmin1_fm': 'previous accept',
              'cue_err': 'cue nose-poke exit',
              'cue_err_cpd': 'cue nose-poke exit',
              'pre_err': 'pre-cue nose-poke error',
              'run_trial': 'accept',
              'omit_trial': 'reject',
              'tr_spec': 'choice',
              'decision': 'choice',
              'run_trial_cpd': 'choice',
              'right': 'right side',
              'right_side_cpd': 'right side',
              'rew_count': 'cum. reward',
              'far_mag_entry': 'accept',
              'velocity_cpd': 'velocity',
              'velocity': 'velocity',
              'std_fr': 'Z-Scored Firing Rate',
              'raw_fr': 'Firing Rate (Hz)',
              'norm_fr': 'Normalised Firing Rate',
              'np_exit_lat': 'nose-poke exit latency',
              'np_exit_lat_cpd': 'nose-poke exit latency',
              'lg1_lat': 'corridor entry\nlatency',
              'far_mag_lat': 'reward magazine\nlatency',
              'iti_pass_dur': 're-initiation\nlatency',
              'trial_dur': 'trial duration',
              'trial_count': 'trial count',
              'fr_area': 'firing rate',
              'nonimplanted': 'non-implanted',
              'implanted': 'implanted',
              'non_coding_cpd': 'non-coding',
              'run_trial:right_side_cpd': 'choice * side',
              'PRE': 'Pre-Cue',
              'CUE': 'Offer',
              'EVAL': 'Go Cue',
              'NP_EX': 'Action',
              'DEC': 'Anticipation',
              'CORR1': 'Corridor Lg1',
              'CORR2': 'Corridor Lg2',
              'CORR3': 'Corridor Lg3',
              'CORR4': 'Corridor Lg4',
              'CORR5': 'Corridor Lg5',
              'CORR6': 'Corridor Lg6',
              'DEL': 'Pre-reward delay',
              'REW': 'Reward'}
epoch_order = ['ITI_MOT', 'ITI_ACT', 'ITI_PASS', 'PRE', 'CUE_ERR', 'CUE', 'EVAL', 'NP_EX', 'CORR1',
               'CORR2', 'DEC', 'LG2', 'CORR3', 'CORR4', 'CORR5', 'CORR6', 'FM', 'DEL',
               'REW', 'REW_1', 'REW_2', 'ACCEPT', 'REJECT']
area_order = ['mo-right', 'acc-right', 'str-right', 'vp-right', 'stn-right']


def elementary_var_label(var):
    if var in var_labels.keys():
        return var_labels[var]
    return var


def approximate_var_label(var, interact_marker=':', interact_token=' x ', abbreviate=False):
    var_ = elementary_var_label(var)
    if var_ == var:
        if interact_marker in var:
            var_ = ''
            for i_, p in enumerate(var.split(interact_marker)):
                var_ += elementary_var_label(p) if not abbreviate else f'{elementary_var_label(p)[:3]}.'
                if i_ != len(var.split(':')) - 1:
                    var_ += interact_token
            if len(var_):
                return var_
    else:
        return var_ if not abbreviate else f'{var_[:3]}.'
    return var


class MixedData:

    @property
    def __name__(self):
        return 'MixedData'


class BlockObj:
    # This class collates together stages from a block in order to be able to do
    # cross-stage analyses.

    # --- Metadata handling --- #


    class TrialBehaviour:
        def __init__(self, meta=None):
            if meta is not None:
                self._meta = meta.env_meta
                self._lookup = meta.env_lookup
            self.beh_df, self.log_df = pd.DataFrame(), pd.DataFrame()
            self.buffer = {}

        def load(self, path=None, appended=True):
            if path is not None:
                spl = os.path.split(path)
                path = os.path.join(spl[0], spl[1])
                p = (path + f'_{self.__name__}') if appended else os.path.join(path, f'{self.__name__}')
                return hs.load_pkl(p)

        @property
        def meta(self):
            return self._meta.copy()

        @property
        def __name__(self):
            return 'TrialBehaviour'


        # ---- Plotting methods ----

        class TwoLevGrpPlt:

            def drop_na(self):
                refin_df = self.df.dropna(subset=[self.xgrp, self.ygrp])
                if refin_df.shape[0] < self.df.shape[0]:
                    print(f'Warning: dropping {self.df.shape[0]-refin_df.shape[0]} rows in the behaviour DataFrame'
                          f'due to NaN values!')
                    self.df = refin_df


    class MixedModel:
        """
            Model class to hold hierarchical regression data.
            Attributes
            ---------
            create_config_file : creates a yaml file to store project metadata
            load : loads associated project input data and the config file
            add_derivatives : adds on derivatives of input data columns
            add_discount_models :
            preprocess_data : processes the data into a Lmer-suitable format with the help of the config file
            plot_collinearity :
            train_model :
            predict :
            """


        @property
        def __name__(self):
            return 'MixedModel'


        @property
        def __name__(self):
            return 'MixedModel'


    # --- Video editing tools --- #

    class Video:
        def __init__(self, meta=None):
            if meta is None:
                self.env_meta, self.env_meta = pd.DataFrame(), pd.DataFrame()
                self.buffer = {}
            else:
                self.env_meta = meta.env_meta
                self.env_lookup = meta.env_lookup
                self.buffer = meta.buffer
            self.n_frames = 20 * 60 * 10 ** 2  # upper estimate on number of frames per stage
            self.thresh = 20 # how many dropped video frames per stage are accepted
            self.xlim, self.ylim = [0, 1020], [130, 0] # crop borders
            self.x_dim = dict(all=179,  # cm - upper estimate of apparatus length
                              end_to_lg1=24.5 + 1,
                              # cm - length between (including) light-gate 1 and end of apparatus. Add 1 due to
                              # camera angle distortion
                              mag=4.3,  # cm - length between (including) reward magazine and end of apparatus
                              end_to_np=2.9,  # cm - length between the end of the apparatus and beginning of nose-poke
                              np=5.6)  # cm - width of the nose-poke
            self.y_dim = dict(all=17.6,  # cm - diameter of the arena
                              end_to_mag=5.0,  # cm - length between end of corridor and beginning of magazine
                              mag=10.3,  # cm - width of the magazine
                              np=3.9)  # cm - width of the nose-poke
            self.example_frame_orig = 'gh009_220109b_0'
            self.dlc_suffix = 'DLC_resnet50_cost_benefit_DLCFeb7shuffle1_900000'
            self.dlc_dir = r'E:\Oliver\cost_benefit_DLC'
            self.vel_dt = 0.1 # the time resolution (s) that the velocity should be based on


        def x_px_to_cm(self, px):
            return (px * self.x_dim['all']) / self.xlim[1]


        def y_px_to_cm(self, px):
            return (px * self.y_dim['all']) / self.ylim[0]


        def get_roll(self):
            return int(np.round(self.vel_dt / (1 / fw.vid_rate), 1))

        def velocity(self, dx: pd.Series, dy: pd.Series):
            roll = self.get_roll()
            vx = self.x_px_to_cm(dx.rolling(roll).mean().iloc[roll::roll].diff()).iloc[1:] / self.vel_dt
            vy = self.y_px_to_cm(dy.rolling(roll).mean().iloc[roll::roll].diff()).iloc[1:] / self.vel_dt
            return np.sqrt(vx ** 2 + vy ** 2)


        @property
        def __name__(self):
            return 'Video'


    class FineBehaviour:
        def __init__(self, meta=None, path=None, **lookup):
            if meta is None and path is not None:
                blocks = BlockObj.check_env_meta(path)
                if blocks is None:
                    blocks = BlockObj(reuse=False)
                    self.blockobj = blocks
                else:
                    self.blockobj = blocks.blockobj
                self.ch_meta, self.lookup, self.buffer = \
                    self.blockobj.get_meta(blocks.env_meta, self.meta_prompts(), {}, **blocks.env_lookup)
                self.env_meta = blocks.env_meta
                hs.save_pkl(self, path + f'_{self.__name__}')
                self.ol = self.get_channels()
            elif meta is not None and path is None:
                self.env_meta = meta.env_meta
                self.env_lookup = meta.env_lookup
                look = {**meta.env_lookup} if lookup is None else {**meta.env_lookup, **lookup}
                if hasattr(meta, 'ch_meta') is False:
                    self.ch_meta, self.lookup, self.buffer = \
                        meta.get_meta(meta.env_meta, self.meta_prompts(), {}, **look)
                else:
                    self.ch_meta = meta.ch_meta
                    self.lookup = meta.lookup
                    self.buffer = meta.buffer


        @property
        def __name__(self):
            return 'FineBehaviour'

        def load(self, path=None, appended=True):
            # Load a metadata object, if path is given and points to a metadata pickle.
            if path is not None:
                p = path + f'_{self.__name__}' if appended else os.path.join(path, f'{self.__name__}')
                return hs.load_pkl(p)


        @staticmethod
        def find_truths(ol: list, expr: str, samp_rate: int = 1000, din_obj=None) -> np.array:
            """
            A high level method for extracting logical highs based on logical expressions.
            Parameters
            ---------
            :param ol : FineBehaviour object list
            :param expr : expression follows pd.DataFrame.query convention
            :param samp_rate : a new sampling rate for behaviour data (e.g. 20000). Defaults to 1000.
            :param din_obj: block.BlockObj.DinChannels object
            Returns
            -------
            :return tr_ar:the logical array
            Example
            ------
            idxs = BlockObj.FineBehaviour.find_truths(ol, "beh_ch_lab == 'EVAL' and beh_ch_lab == 'REW_LG_A'")
            """
            ex_expr = "beh_ch_lab == 'EVAL' and beh_ch_lab == 'REW_LG_A'"
            _, vals, bops, wops = hs.parse_logical_statement(expr)
            ar = []
            ol = [ol] if not isinstance(ol, list) else ol
            for o in ol:
                for val, wop in zip(vals, wops):
                    if val == o.meta.beh_ch_lab:
                        if wop == '==':
                            ar.append(o.high_logic_ar(samp_rate=samp_rate, level=None, din_obj=din_obj))
                        else:
                            ar.append(np.invert(o.high_logic_ar(samp_rate=samp_rate, level=None, din_obj=din_obj)))

                    if len(ar) == len(vals):
                        if len(vals) == 1:  # If just one logical expression was given
                            return hs.bool_highs(ar[0])
                        for j in range(1, len(ar)):
                            if 1 >= j < len(ar):
                                truths = hs.truth(bops[j - 1], ar[j - 1], ar[j])
                            else:
                                truths = hs.truth(bops[j - 1], truths, ar[j])
                        return hs.bool_highs(truths)
            raise ValueError(f'Expression did not find any matches for {expr}. Make sure the formatting meets the '
                             f'requirements, e.g: ' + ex_expr)

    class AmplifierChannels:


        @property
        def __name__(self):
            return 'AmplifierChannels'


    class DinChannels:
        def __init__(self, meta=None, path=None, **lookup):
            if path is not None and meta is None:
                blocks = BlockObj.check_env_meta(path)
                if blocks is None:
                    blocks = BlockObj(reuse=False)
                    self.blockobj = blocks
                else:
                    self.blockobj = blocks.blockobj
                time.sleep(2)
            elif meta is not None:
                if hasattr(meta, 'ch_meta'):
                    self.env_meta = meta.env_meta
                    self.ch_meta = meta.ch_meta
                    self.lookup = meta.lookup
                    self.buffer = meta.lookup
                    self.ol = self.get_objects()
                else:
                    look = {**meta.env_lookup} if lookup is None else {**meta.env_lookup, **lookup}
                    self.ch_meta, self.lookup, self.buffer = \
                        meta.get_meta(meta.env_meta, self.meta_prompts, {}, **look)
                    self.env_meta = meta.env_meta


        @property
        def __name__(self):
            return 'DinChannels'

        def load(self, path=None, appended=True):
            # Load a metadata object, if path is given and points to a metadata pickle.
            if path is not None:
                p = path + f'_{self.__name__}' if appended else os.path.join(path, f'{self.__name__}')
                return hs.load_pkl(p)


    class SpikeTrains:


        @property
        def __name__(self):
            return 'SpikeTrains'


    class SpikeStats:
        """
        A class for storing higher level spike timing information in formats such as DataFrames for direct statistical
        analysis.
        """

        def __init__(self, f_root=None, sp_o=None, din_o=None, fbeh_o=None, trbeh_o=None, version=592261):
            self.f_root = f_root
            self.version = version
            self.fr_st = {}
            self.sp_o, self.din_o, self.fbeh_o, self.trbeh_o = sp_o, din_o, fbeh_o, trbeh_o
            self.bulky = ['sp_o', 'din_o', 'fbeh_o']  # Items to be optionally discarded when saving
            for o, dtype in self.objects.items(): # Instantiate the data objects
                setattr(self, o, dtype)

        @property
        def __name__(self):
            return 'SpikeStats'

        @property
        def attrib(self):
            # Return a list of all the processed data items that can be members of this class
            attributes = {592261: ['spike_times', 'regression', 'r_models', 'decoder'],
                          573012: ['spike_times', 'regression', 'r_models', 'decoder', 'pairwise_corr'],
                          113873: ['spike_times', 'regression', 'r_models', 'decoder', 'canonical_corr'],}
            return attributes[self.version]

        @property
        def objects(self):
            dtypes = {592261: {'spike_times': pd.DataFrame(), 'regression': pd.DataFrame(),
                               'decoder': pd.DataFrame(), 'r_models': type(None)},
                      573012: {'spike_times': pd.DataFrame(), 'regression': pd.DataFrame(),
                               'decoder': pd.DataFrame(), 'r_models': type(None), 'pairwise_corr': pd.DataFrame()},
                      113873: {'spike_times': pd.DataFrame(), 'regression': pd.DataFrame(),
                               'decoder': pd.DataFrame(), 'r_models': type(None), 'canonical_corr': pd.DataFrame()}
                      }
            return dtypes[self.version]

        @property
        def dtypes(self):
            return self.objects

        def attr_check(self, *v, path=None):
            """
            A method for checking whether a class attribute exists and
            :param path: str : path to load if necessary
            :param var: str or list
            :return:
            """
            for var in v:
                obj = getattr(self, var)
                if not isinstance(obj, type(None)):
                    if isinstance(obj, pd.DataFrame):
                        if obj.empty:
                            self.load(path=path)
                        if getattr(self, var).empty:
                            return ValueError(f'{var} is an empty DataFrame. Please run the analysis method.')
                else:
                    self.load(path=path)
                    if isinstance(getattr(self, var), type(None)):
                        return ValueError(f'{var} is a NoneType object. Please run the analysis method.')

        def load(self, path=None, appended=True):
            path = path if path is not None else self.f_root
            p = path + f'_{self.__name__}' if appended else os.path.join(path, f'{self.__name__}')
            try:
                inst = hs.load_pkl(p)
                path = path if path is not None else inst.f_root
            except FileNotFoundError:  # the small pickle of the object itself is optional: the tables are what matter
                pass
            for a in self.attrib:
                # Load dataframes separately as they may be very large and pickle will struggle.
                p = path + f'_{a}' if appended else os.path.join(path, f'{a}')
                try:
                    if isinstance(getattr(self, a), pd.DataFrame):
                        setattr(self, a, hs.load_df(p))
                    else:
                        setattr(self, a, hs.load_pkl(p))
                except (FileNotFoundError, AttributeError):
                    setattr(self, a, self.dtypes[a])
                    try:
                        if isinstance(getattr(self, a), pd.DataFrame):
                            setattr(self, a, hs.load_df(p))
                        else:
                            setattr(self, a, hs.load_pkl(p))
                    except FileNotFoundError:
                        pass
            return self


        def get_bin_max_in_epochs(self, df):
            bin_max, edges, col_n = {}, {}, 0
            for epoch in self.get_sorted_epochs(df['epoch'].unique()):
                e_df = df.query(f'epoch=="{epoch}"')
                bin_max[epoch] = e_df['bin'].max() + 1
                edges[epoch] = [col_n, col_n + bin_max[epoch] - 1]
                col_n += bin_max[epoch]
            return bin_max, col_n, edges


        @staticmethod
        def drop_levs(omit, df):
            if omit is not None:
                for omit_k, omit_v in omit.items():
                    for omit_v_ in [omit_v] if not isinstance(omit_v, list) else omit_v:
                        if isinstance(omit_v_, str):
                            df.query(f'{omit_k} != "{omit_v_}"', inplace=True)
                        else:
                            df.query(f'{omit_k} != {omit_v_}', inplace=True)
                        print(f'Dropping  {omit_k} == {omit_v_}')
            return df

        @staticmethod
        def get_sorted_epochs(epochs):
            return [x for _, x in sorted(zip([epoch_order.index(i) for i in epochs], epochs))]

        @staticmethod
        def get_sorted_areas(areas):
            return [x for _, x in sorted(zip([area_order.index(i) for i in areas], areas))]


        @staticmethod
        def yield_epochs(fbeh_obj: list, query: dict, din_obj: list) -> iter:
            """
            Get D-IN pulse counts to window time periods of interest
            :param fbeh_obj: a list of the FineBehaviour objects to query
            :param query: a dictionary containing the list of string formatted queries as values
            :param din_obj:a list of selected DinChannels objects (from the same stages as the FineBehaviour obj list)
            :return: iterator consisting of the epoch index and a list with the edges
            """
            for query_ in query.values():
                for query__ in query_:
                    for idx, edges in enumerate(BlockObj.FineBehaviour.find_truths(fbeh_obj, query__, din_obj=din_obj)):
                        yield idx, [edges[0], edges[1]]

        @staticmethod
        def get_highs(fbeh_obj: list, query: str, din_obj: list) -> tuple:
            return tuple(BlockObj.FineBehaviour.find_truths(fbeh_obj, query, din_obj=din_obj))


        @staticmethod
        def prospective_bin(bin_start, bin_size) -> (int, int):
            return int(bin_start), int(bin_start + bin_size)


        @staticmethod
        def f_std_fr(bin_: int):
            return f'std_fr_{bin_}ms_bin'

        @staticmethod
        def f_raw_fr(bin_: int):
            return f'raw_fr_{bin_}ms_bin'

        @staticmethod
        def f_norm_fr(bin_: int):
            return f'norm_fr_{bin_}ms_bin'

        @staticmethod
        def f_smooth_fr(bin_: int):
            return f'smooth_fr_{bin_}ms_bin'


        def format_map(self, var):
            m = {'raw_fr': self.f_raw_fr,
                 'std_fr': self.f_std_fr,
                 'norm_fr': self.f_norm_fr,
                 'smooth_fr': self.f_smooth_fr}
            return m[var]


        class TrialWindow:
            # A class to configure the trial window of interest, mainly used in SpikeStats
            def __init__(self, s_lab, SpikeStats, DinCh, FineBeh, TrBeh):
                run_expr = "beh_ch_lab == 'run_trial'"
                omit_expr = "beh_ch_lab == 'omit_trial'"
                din_idx = DinCh.db.oli[DinCh.db.stage == s_lab].values
                self.SpikeStats = SpikeStats
                self.din_sel = DinCh.ol[int(din_idx)]
                self.fbeh_sel = self.get_f_beh_sel(s_lab, FineBeh)
                self.tr_beh = TrBeh
                self.stage_df = TrBeh.beh_df.query(f'stage == "{s_lab}"')
                self.block_df = TrBeh.beh_df.query(f'day_subj == "{hs.get_subj_lab(s_lab)}_{hs.get_date_lab(s_lab)}"')
                r_highs = SpikeStats.get_highs(self.fbeh_sel, run_expr, self.din_sel)
                o_highs = SpikeStats.get_highs(self.fbeh_sel, omit_expr, self.din_sel)
                self.trial_sls, self.trial_ids = self.trial_map(runs=r_highs, omits=o_highs)

            @staticmethod
            def get_f_beh_sel(s_lab, FineBeh):
                return [FineBeh.ol[_] for _ in FineBeh.db.oli[FineBeh.db.stage == s_lab].values]

            @staticmethod
            def trial_map(runs=None, omits=None) -> (list, list):
                trial_sls, trial_ids, starts, stops, labs = [], [], [], [], []
                for idx in runs:
                    starts.append(idx[0])
                    stops.append(idx[1])
                    labs.append('run_trial')
                for idx in omits:
                    starts.append(idx[0])
                    stops.append(idx[1])
                    labs.append('omit_trial')
                ord_idxs = np.argsort(starts)
                for sl in ord_idxs:
                    trial_sls.append(tuple([starts[sl], stops[sl]]))
                    trial_ids.append(labs[sl])
                return trial_sls, trial_ids

            def get_trial_id(self, e_sl, trial_idx):
                # Configure the "repeat" to indicate whether the given slice is a repeat of a trial or not
                spec, repeat = 'non_trial', False
                for trial_idx_, tr_sl in enumerate(self.trial_sls):
                    if any([tr_sl[0] <= i <= tr_sl[1] for i in e_sl]):
                        if trial_idx_ != trial_idx:
                            spec, trial_idx = self.trial_ids[self.trial_sls.index(tr_sl)], trial_idx_
                            assert trial_idx < self.stage_df.shape[0], \
                                f'TrialBehaviour dataframe, based on log.csv, had only {trial_idx - 1} trials which did' \
                                f'not line up with the output of FineBehaviour channel data. Use ' \
                                f'{BlockObj.FineBehaviour.gen_ch} to re-create the log.csv and FineBehaviour object and' \
                                f'then re-create {BlockObj.TrialBehaviour.__name__} to line it up.'
                            if spec == 'run_trial':
                                assert self.stage_df.iloc[trial_idx, :]['far_mag_entry'] == 1, \
                                    f'Based on FineBehaviour data, trial was expected to be an accepted trial but was ' \
                                    f'instead a reject trial.'
                            else:
                                assert self.stage_df.iloc[trial_idx, :]['far_mag_entry'] == 0, \
                                    f'Based on FineBehaviour data, trial was expected to be a rejected trial but was ' \
                                    f'instead an accept trial.'
                        else:
                            repeat = True
                        break
                return spec, repeat, trial_idx


            def yield_epochs(self, sm_query, pre=0, pre_sign=np.positive, post=0, post_corr=0, post_corr_1: str = None,
                             post_corr_1_sign=np.negative, post_lg_crit='lg1_lat', start_aligned=True, sorted=False,
                             ) -> \
                    (list, str, int):
                """
                # If not in need of stitching, use this method instead of the one below
                :param sm_query: the epoch to be queried
                :param pre: the amount of time to be included before/after the start of the query
                :param pre_sign: the sign of the pre interval to be added
                :param post: the time included pre/post the end (or start) of the looking period
                :param post_corr: the correction done on the end timing, if any
                :param post_corr_1: any stage_df based correction that might want to be applied to the post-duration
                :param post_corr_1_sign: whether to subtract or add the post_corr_1
                :param post_lg_crit: any additional light-gate criteria that should be applied for the post-duration
                :param start_aligned: whether to align the end timing to the start or end of the query epoch
                :param sorted: whether to sort by trial_idx
                :return:
                """
                trial_idx = -1
                sls = pd.DataFrame(columns=['start', 'stop', 'spec', 'trial_idx'])
                for sl_idx, e_sl in self.SpikeStats.yield_epochs(self.fbeh_sel, sm_query, din_obj=self.din_sel):
                    spec, repeat, trial_idx = self.get_trial_id(e_sl, trial_idx)
                    if not repeat and trial_idx > -1:
                        _, __ = self.conf_pre_post(spec, trial_idx, pre, post, post_corr, post_corr_1,
                                                   post_corr_1_sign, post_lg_crit)
                        if not (np.isnan(_) and not np.isnan(__)):
                            res = [e_sl[0]+1 - pre_sign(_), ((e_sl[0]+1+__) if start_aligned else (e_sl[1]) + __)], \
                                   spec, trial_idx
                            if not sorted:
                                yield res
                            else:
                                sls = pd.concat([sls, pd.DataFrame(dict(start=res[0][0], stop=res[0][1], spec=res[1],
                                                                        trial_idx=res[2]), index=[0])],
                                               ignore_index=True)
                for _, r in sls.sort_values(by=['trial_idx'], ascending=True).iterrows():
                    yield [[r.start, r.stop], r.spec, r.trial_idx]


            def conf_pre_post(self, tr_spec, trial_idx, pre=0, post=0, post_corr=0, post_corr_1=0,
                              post_corr_1_sign=np.negative, post_lg_crit=None, fun=np.nanmedian, median_src='stage') ->\
                (int, int):
                def pre_post_omit():
                    if isinstance(pre, numbers.Integral):
                        pre__ = pre
                    else:
                        pre__ = 0
                        for lat in pre.split(' + '):
                            pre__ += self.stage_df[lat].iloc[trial_idx]
                    if isinstance(post, numbers.Integral):
                        post__ = post
                    else:
                        post__ = 0
                        for lat in post.split(' + '):
                            try:
                                post__ += int(self.stage_df[lat].iloc[trial_idx])
                            except ValueError:
                                print(f'Warning: {self.stage_df.stage.iloc[0]} trial {trial_idx} had a missing value '
                                      f'for {lat}!\n Substituting with 0.')
                                post__ = 0
                    # Add a latency on omission trials to estimate decision time
                    if post_lg_crit:
                        dist = self.stage_df[post_lg_crit] if median_src == 'stage' else self.block_df[post_lg_crit]
                        post__ += int(np.round(fun(dist)))
                    post__ += post_corr
                    if post_corr_1 is not None:
                        post__ += post_corr_1_sign(self.stage_df[post_corr_1].iloc[trial_idx])
                    return pre__, post__

                def pre_post_run():
                    if isinstance(pre, numbers.Integral):
                        pre__ = pre
                    else:
                        pre__ = 0
                        for lat in pre.split(' + '):
                            pre__ += self.stage_df[lat].iloc[trial_idx]
                    if isinstance(post, numbers.Integral):
                        post__ = post
                    else:
                        post__ = 0
                        for lat in post.split(' + '):
                            try:
                                post__ += int(self.stage_df[lat].iloc[trial_idx])
                            except ValueError:
                                print(f'Warning: {self.stage_df.stage.iloc[0]} trial {trial_idx} had a missing value '
                                      f'for {lat}!\n Substituting with 0.')
                                post__ = 0
                    if post_lg_crit:
                        post__ += int(np.round(self.stage_df[post_lg_crit].iloc[trial_idx]))
                    post__ += post_corr
                    if post_corr_1 is not None:
                        post__ += post_corr_1_sign(self.stage_df[post_corr_1].iloc[trial_idx])
                    return pre__, post__

                def pre_post_none():
                    if isinstance(pre, numbers.Integral):
                        pre__ = pre
                    else:
                        pre__ = 0
                    if isinstance(post, numbers.Integral):
                        post__ = post
                    else:
                        post__ = 0
                    return pre__, post__

                def try_pre_post():
                    if tr_spec == 'omit_trial':
                        return pre_post_omit()
                    elif tr_spec == 'run_trial':
                        return pre_post_run()
                    else:
                        return pre_post_none()

                try:
                    pre_, post_ = try_pre_post()
                except ValueError:  # deal with prematurely ended oh011 stages
                    new_stage = self.stage_df.stage.iloc[0].replace('a', 'b')
                    self.stage_df = self.tr_beh.beh_df.query(f'stage == "{new_stage}"')
                    pre_, post_ = try_pre_post()

                return pre_, post_


        # ------- Analysis methods


        @staticmethod
        def build_trial_groups(sp_df, subj_days, epochs, sort_trials=True, return_order_df=False):
            """
            Recovers, for each session, the trial-level accept/reject group label ('tr_spec') in the
            same trial order that make_fr_matrices_4_1/trial_filter puts into the firing-rate matrix
            rows -- i.e. per stage (sorted), the trials common to every epoch (trial_filter's
            epoch-intersection), sorted by trial_idx iff sort_trials=True, stages concatenated in
            order. This is done at the SESSION level rather than per-unit/per-area: valid because
            canonical_correlation_5/6/7 already assume every unit (and area) within a session shares
            one common trial axis (that's what lets them hstack per-unit columns together), so any one
            unit's trial structure represents the whole session.
            :param sp_df: the combined accepted+rejected spike_times dataframe
            :param subj_days: iterable of 'subject_date' session labels to build groups for
            :param epochs: the epoch list, identical to what make_fr_matrices_4_1 derives internally
            from sp_df['epoch'].unique() -- pass the same `epochs` used for that call
            :param sort_trials: must match the sort_trials passed to make_fr_matrices_4_1
            :param return_order_df: if False (default), return {subject_day: np.array of 'reject'/
            'accept' labels, one per trial row, in canonical_correlation_7 group_labels order} -- the
            positional form canonical_correlation_7 expects, aligned to the firing-rate matrix's own
            row order. If True, return {subject_day: pd.DataFrame(columns=['stage', 'trial_idx',
            'trial_group'])} instead -- the same trials/labels, but keyed by (stage, trial_idx) so
            they can be merged onto a trial-level dataframe (e.g. cca_component_decoder_2's
            trial_df, which is addressed by (stage, trial_idx) rather than firing-rate-matrix row
            position).
            :return: dict, keyed by subject_day, in one of the two forms described above
            """
            tr_spec_map = {k: approximate_var_label(k) for k in ['run_trial', 'omit_trial']}
            sess_lookup = sp_df.copy()
            sess_lookup['subject_day'] = [hs.get_subj_date(s) for s in sess_lookup['stage']]

            trial_groups = {}
            for subj_day in subj_days:
                sess_df = sess_lookup.query('subject_day == @subj_day')
                order_df = pd.DataFrame()
                for stage in np.sort(sess_df['stage'].unique()):
                    stage_df = sess_df[sess_df['stage'] == stage]
                    trial_idxs = [np.sort(stage_df.query('epoch == @ep')['trial_idx'].unique()) for ep in epochs]
                    lens = np.array([len(tt) for tt in trial_idxs])
                    min_idx = np.where(lens == int(np.min(lens)))[0][0]
                    trs = trial_idxs[min_idx]
                    stage_sel = stage_df.loc[stage_df['trial_idx'].isin(trs), ['stage', 'trial_idx', 'tr_spec']]
                    # one row per trial -- tr_spec is constant within a trial regardless of epoch/bin
                    stage_sel = stage_sel.drop_duplicates(subset='trial_idx')
                    if sort_trials:
                        stage_sel = stage_sel.sort_values('trial_idx', kind='mergesort')
                    order_df = pd.concat([order_df, stage_sel], ignore_index=True)
                unmapped = set(order_df['tr_spec'].unique()) - set(tr_spec_map)
                assert not unmapped, f"{subj_day}: unrecognised tr_spec value(s) {unmapped}"
                order_df['trial_group'] = order_df['tr_spec'].map(tr_spec_map)
                if return_order_df:
                    trial_groups[subj_day] = order_df[['stage', 'trial_idx', 'trial_group']].reset_index(drop=True)
                else:
                    trial_groups[subj_day] = order_df['trial_group'].to_numpy()
            return trial_groups


        def get_cca_df_edge_params(self,
                                   alpha: float,
                                   agg_fn=hs.average_rhos,
                                   stat_var: str ='corr_cv',
                                   perm_var: str ='corr_cv_perm_avg',
                                   edge_var: str = '∆rho',
                                   fdr: bool = True,
                                   best_rank_only = False,
                                   source_df: pd.DataFrame = None,
                                   alternative: str = 'greater',
                                   ) -> pd.DataFrame:
            cca_df = self.canonical_corr.copy() if source_df is None else source_df
            rows = []
            for epoch in self.get_sorted_epochs(cca_df['epoch'].unique()):
                for area_pair in cca_df.area_pair.unique():  # the default variable label is used here
                    tbl = cca_df.query(f'epoch=="{epoch}" & area_pair=="{area_pair}"')
                    if tbl.empty:
                        area_pair = '_'.join(area_pair.split("_")[::-1])
                        tbl = cca_df.query(f'epoch=="{epoch}" & area_pair=="{area_pair}"')
                    if tbl.empty:
                        continue
                    edge_label = tuple([str(fw.region_labels()[area_pair.split("_")[0]]),
                                        str(fw.region_labels()[area_pair.split("_")[1]])])

                    # ---- find best rank ----
                    best_rank = None
                    best_value = -np.inf

                    for rank in np.unique(tbl['rank']):
                        plot_corrs = tbl.loc[tbl['rank'] == rank][stat_var].values

                        if len(plot_corrs) == 0:
                            continue

                        val = agg_fn(plot_corrs)

                        if val > best_value:
                            best_value = val
                            best_rank = rank

                    # skip if nothing valid
                    if best_rank is None:
                        continue

                    if best_rank_only:
                        # ---- compute stats only for best rank ----
                        sub = tbl.loc[tbl['rank'] == best_rank]
                    else:
                        sub = tbl

                    for rank in np.unique(np.sort(sub['rank'])):
                        real_corrs = sub.loc[sub['rank'] == rank][stat_var].values
                        perm_corrs = sub.loc[sub['rank'] == rank][perm_var].values
                        plot_corrs = sub.loc[sub['rank'] == rank][edge_var if stat_var != edge_var else stat_var].values
                        try:
                            _, p = wilcoxon(real_corrs, perm_corrs, alternative=alternative)
                        except ValueError:
                            p = np.nan

                        if len(plot_corrs) > 1:
                            edge_sem = np.nanstd(plot_corrs, ddof=1) / np.sqrt(np.sum(np.isfinite(plot_corrs)))
                        else:
                            edge_sem = np.nan

                        row = pd.Series({
                            'epoch': epoch,
                            'edge_label': edge_label,
                            'area_pair': area_pair,
                            'rank': rank,
                            'edge_value': agg_fn(plot_corrs),
                            'edge_sem': edge_sem,
                            'edge_pval': p
                        })
                        rows.append(row)

            # explicit columns so an empty `rows` (e.g. source_df had no qualifying rows for this
            # epoch/area_pair combination at all) still produces the expected schema rather than a
            # zero-column DataFrame that the dropna below -- and any caller indexing 'edge_pval' --
            # would KeyError on
            edge_params = pd.DataFrame(
                rows, columns=['epoch', 'edge_label', 'area_pair', 'rank', 'edge_value', 'edge_sem', 'edge_pval']
            )
            edge_params.dropna(subset=["edge_pval"]).reset_index(drop=True)

            if fdr:
                ps = edge_params['edge_pval'].values
                ps[np.isnan(ps)] = 1  # replace any non-existent pairs with 1s to be able to preserve the analysis
                p_mask = hs.benjamini_hochberg(ps, alpha=alpha)
                for i, mask in enumerate(p_mask):
                    edge_params.loc[i, 'edge_pval'] = np.invert(mask).astype(int)
            return edge_params


        # -------- PLotting methods


        @staticmethod
        def plot_node_graph_1(
                coords: dict,  # node -> (x, y)
                edge_params: dict,  # edge_labels, edge_values, edge_pvals
                thresh: float,
                node_colors=None,
                node_labels=None,
                linewidth_range=(1, 2),
                node_size=100,
                label_fontsize=3,
                where=(2.1, 2),
                rc_context=hs.no_axes,
                background_png=None,
                annotate_edges=True,
                plot_ns_edges=True,
                edge_label_fontsize=5,
                edge_label_offset=0.015,
                title=None,
                ylabel=None,
                epoch_png_path=None,
                nodes_with_finite_edges=None,
                global_edge_range=None,
                show_scalebar=False,
                **scalebar_kwargs,
        ):
            """
            Nodes:
              - Always draw text labels for all nodes in coords
              - Draw the node circle ONLY if the node participates in >=1 finite edge_value
            Edges:
              - Never draw edges with NaN edge_value
              - Significance only controls linestyle / inclusion when plot_ns_edges=False
            """

            edge_labels = list(edge_params['edge_labels'])
            edge_values = np.asarray(edge_params['edge_values'], dtype=float)
            edge_pvals = np.asarray(edge_params['edge_pvals'], dtype=float)

            if not (len(edge_labels) == len(edge_values) == len(edge_pvals)):
                raise ValueError("edge_labels, edge_values and edge_pvals must have equal length")

            nodes = list(coords.keys())

            # node colors/labels
            if node_colors is None:
                node_colors = {n: "white" for n in nodes}
            elif not isinstance(node_colors, dict):
                node_colors = {n: node_colors for n in nodes}

            if node_labels is None:
                node_labels = {n: str(n) for n in nodes}

            # -------------------------
            # Determine which nodes have ANY finite-valued edge
            # -------------------------
            nodes_with_finite_edges = set() if nodes_with_finite_edges is None else nodes_with_finite_edges
            if not nodes_with_finite_edges:
                for (u, v), val in zip(edge_labels, edge_values):
                    if np.isfinite(val):
                        nodes_with_finite_edges.add(u)
                        nodes_with_finite_edges.add(v)

            # -------------------------
            # Decide which edges are drawable
            # -------------------------
            finite_edge = np.isfinite(edge_values)

            if plot_ns_edges:
                draw_edge = finite_edge
            else:
                draw_edge = finite_edge & (edge_pvals < thresh)

            # -------------------------
            # Width scaling (finite edges only)
            # -------------------------
            widths = {}
            vals_for_width = edge_values[finite_edge]
            labs_for_width = np.array(edge_labels, dtype=object)[finite_edge]

            if len(vals_for_width):
                vmin, vmax = float(np.min(vals_for_width)), float(np.max(vals_for_width))
                a, b = linewidth_range
                if np.isclose(vmin, vmax):
                    scaled = np.full_like(vals_for_width, np.mean(linewidth_range), dtype=float)
                else:
                    scaled = a + (vals_for_width - vmin) * (b - a) / (vmax - vmin)
                for lab, w in zip(labs_for_width, scaled):
                    widths[tuple(lab)] = float(w)

            # -------------------------
            # Split edges by style
            # -------------------------
            edge_style = {'-': [], '--': []}
            for (u, v), val, pval, keep in zip(edge_labels, edge_values, edge_pvals, draw_edge):
                if not keep:
                    continue
                style = '-' if pval < thresh else '--'
                edge_style[style].append((u, v))

            # -------------------------
            # Plot
            # -------------------------
            with mpl.rc_context(rc_context):
                ax = hs.get_ax(where)
                hs.add_image_background_norm(ax, fname=background_png)

                # ---- Node circles: ONLY for nodes with finite edges ----
                for n in nodes:
                    if n in nodes_with_finite_edges:
                        ax.scatter(
                            *coords[n],
                            s=node_size,
                            color=node_colors[n],
                            edgecolor=fw.region_colours()[n],
                            linewidth=1,
                            zorder=2
                        )

                # ---- Node text: ALWAYS ----
                for n in nodes:
                    ax.text(
                        coords[n][0], coords[n][1],
                        node_labels[n],
                        ha="center", va="center",
                        fontsize=label_fontsize,
                        zorder=3
                    )

                # ---- Edges ----
                for style, pairs in edge_style.items():
                    if not pairs:
                        continue
                    segs = [[coords[u], coords[v]] for u, v in pairs]
                    cols = [
                        hs.mix_colours(
                            fw.region_colours()[fw.region_labels()[u]],
                            fw.region_colours()[fw.region_labels()[v]]
                        )
                        for u, v in pairs
                    ]
                    lws = [widths[(u, v)] for u, v in pairs]  # safe: pairs are finite_edge subset

                    ax.add_collection(LineCollection(
                        segs,
                        linewidths=lws,
                        colors=cols,
                        linestyle=style,
                        zorder=1
                    ))

                # ---- Edge annotations (finite edges only, and only if drawn) ----
                if annotate_edges:
                    for (u, v), val, keep in zip(edge_labels, edge_values, draw_edge):
                        if not keep:
                            continue

                        x0, y0 = coords[u]
                        x1, y1 = coords[v]
                        mx, my = (x0 + x1) / 2, (y0 + y1) / 2

                        dx, dy = x1 - x0, y1 - y0
                        L = np.hypot(dx, dy)
                        if L > 0:
                            mx += (-dy / L) * edge_label_offset
                            my += (dx / L) * edge_label_offset

                        if not np.isnan(val):
                            val = np.round(val, 2)
                            val = f"{val:.2f}"
                        else:
                            val = ""
                        ax.text(mx, my, val, ha="center", va="center", fontsize=edge_label_fontsize)

                ax.set_aspect("equal")

                if title is not None:
                    ax.set_title(title)

                if epoch_png_path is not None:
                    hs.add_png_to_plot(ax=ax, img_path=epoch_png_path, x=.5, y=1.03, zoom=0.08)

                if ylabel is not None:
                    ax.text(0, 0.5, ylabel, rotation=90, va="center")

                if show_scalebar:
                    hs.add_edge_width_scalebar_quantiles(
                        ax=ax,
                        edge_values=edge_values,
                        linewidth_range=linewidth_range,
                        global_edge_range=global_edge_range,
                        **(scalebar_kwargs or {})
                    )

                return ax


        @staticmethod
        def plot_node_graph_3_1(
                coords: dict,  # node -> (x, y)
                edge_params: "pd.DataFrame",  # DataFrame with columns: edge_label, rank, edge_value, edge_pval
                thresh: float,
                node_colors=None,
                node_labels=None,
                linewidth_range=(1, 2),
                static_lw=None,  # if set, use a constant linewidth
                edge_color="black",  # fallback colour
                color_by_rank=False,  # fallback rank-colouring
                circuit_map=None,  # NEW: area_pair -> higher-level group
                colour_map=None,  # NEW: higher-level group -> matplotlib colour
                node_size=100,
                label_fontsize=3,
                where=(2.1, 2),
                rc_context=hs.no_axes,
                background_png=None,
                annotate_edges=True,
                plot_ns_edges=True,
                edge_label_fontsize=5,
                title=None,
                ylabel=None,
                epoch_png_path=None,
                nodes_with_finite_edges=None,
                global_edge_range=None,
                show_scalebar=False,  # will be ignored if static_lw is not None
                cmap_name="tab10",
                add_rank_legend=True,  # will be ignored if color_by_rank is False
                **scalebar_kwargs,
        ):
            if not isinstance(edge_params, pd.DataFrame):
                raise ValueError(
                    "edge_params must be a pandas DataFrame with columns "
                    "['edge_label','rank','edge_value','edge_pval']"
                )

            required_cols = {"edge_label", "rank", "edge_value", "edge_pval"}
            if not required_cols.issubset(set(edge_params.columns)):
                raise ValueError(f"edge_params must contain columns: {required_cols}")

            def _parse_label(l):
                if isinstance(l, (tuple, list)):
                    if len(l) != 2:
                        raise ValueError(f"Unsupported edge_label tuple/list length: {len(l)}")
                    return (str(l[0]), str(l[1]))
                if isinstance(l, str):
                    if "_" in l:
                        parts = l.split("_")
                    elif "-" in l:
                        parts = l.split("-")
                    else:
                        raise ValueError(f"Cannot parse edge_label string '{l}' to pair")
                    if len(parts) != 2:
                        raise ValueError(f"edge_label string '{l}' must parse into two parts")
                    return (str(parts[0]), str(parts[1]))
                raise ValueError(f"Unsupported edge_label type: {type(l)}")

            def _canonical_pair_key(pair):
                u, v = pair
                return "_".join(sorted([str(u), str(v)]))

            def _canonical_pair_from_string(s):
                if "_" in s:
                    parts = s.split("_")
                elif "-" in s:
                    parts = s.split("-")
                else:
                    raise ValueError(f"Cannot parse circuit_map key '{s}'")
                if len(parts) != 2:
                    raise ValueError(f"circuit_map key '{s}' must have two parts")
                return "_".join(sorted([str(parts[0]), str(parts[1])]))

            def _canonical_region_token(x):
                """
                Map any region label style onto the canonical lower-case tokens:
                mo, acc, str, vp, stn
                """
                labels = fw.region_labels()

                # canonical tokens already
                canon = {"mo", "acc", "str", "vp", "stn"}
                if x in canon:
                    return x

                if not isinstance(x, str):
                    return str(x).lower()

                # Handle "-right" labels first
                if x.endswith("-right"):
                    x = x[:-6]

                # Build inverse map from displayed labels to canonical tokens
                # e.g. "MO/VO" -> "mo", "ACC" -> "acc", "STA" -> "stn"
                inv = {}
                for tok in canon:
                    disp = labels.get(tok, None)
                    if disp is not None:
                        inv[disp] = tok

                if x in inv:
                    return inv[x]

                # If we already reduced "stn-right" -> "stn", etc.
                if x in canon:
                    return x

                return x.lower()

            def _canonical_pair_key(pair):
                """
                Turn a pair into an order-invariant canonical key like:
                acc_stn, mo_str, stn_vp, ...
                """
                if isinstance(pair, (tuple, list)) and len(pair) == 2:
                    a, b = pair
                elif isinstance(pair, str):
                    if "_" in pair:
                        parts = pair.split("_")
                    elif "-" in pair:
                        parts = pair.split("-")
                    else:
                        raise ValueError(f"Cannot parse pair label '{pair}'")
                    if len(parts) != 2:
                        raise ValueError(f"Pair label '{pair}' must contain two parts")
                    a, b = parts
                else:
                    raise ValueError(f"Unsupported pair type: {type(pair)}")

                a = _canonical_region_token(a)
                b = _canonical_region_token(b)

                return "_".join(sorted([a, b]))
            # -------------------------
            # Copy and canonicalise input
            # -------------------------
            df = edge_params.copy().reset_index(drop=True)
            df["pair"] = df["edge_label"].apply(_parse_label)
            df["pair_canon"] = df["edge_label"].apply(_canonical_pair_key)

            pair_to_group = {}
            if circuit_map is not None:
                for k, v in circuit_map.items():
                    pair_to_group[_canonical_pair_key(k)] = v

            df["rank"] = df["rank"].astype(int)
            df["edge_value"] = pd.to_numeric(df["edge_value"], errors="coerce")
            df["edge_pval"] = pd.to_numeric(df["edge_pval"], errors="coerce")

            pairs = list(df["pair"])
            ranks = df["rank"].to_numpy()
            edge_values = df["edge_value"].to_numpy(dtype=float)
            edge_pvals = df["edge_pval"].to_numpy(dtype=float)
            pair_canons = df["pair_canon"].tolist()

            if not (len(pairs) == len(edge_values) == len(edge_pvals) == len(ranks)):
                raise ValueError("edge_params columns must have equal length")

            nodes = list(coords.keys())

            # node colors/labels
            if node_colors is None:
                node_colors = {n: "white" for n in nodes}
            elif not isinstance(node_colors, dict):
                node_colors = {n: node_colors for n in nodes}

            if node_labels is None:
                node_labels = {n: str(n) for n in nodes}

            # -------------------------
            # Canonicalise circuit map
            # -------------------------
            pair_to_group = {}
            if circuit_map is not None:
                for pair_name, group_name in circuit_map.items():
                    if isinstance(pair_name, str):
                        canon = _canonical_pair_from_string(pair_name)
                    elif isinstance(pair_name, (tuple, list)) and len(pair_name) == 2:
                        canon = _canonical_pair_key(pair_name)
                    else:
                        raise ValueError(f"Unsupported circuit_map key type: {type(pair_name)}")
                    pair_to_group[canon] = group_name

            # -------------------------
            # Determine which nodes have ANY finite-valued edge
            # -------------------------
            nodes_with_finite_edges = set() if nodes_with_finite_edges is None else set(nodes_with_finite_edges)
            if not nodes_with_finite_edges:
                for (u, v), val in zip(pairs, edge_values):
                    if np.isfinite(val):
                        nodes_with_finite_edges.add(u)
                        nodes_with_finite_edges.add(v)

            # -------------------------
            # Decide which edges are drawable
            # -------------------------
            finite_edge_mask = np.isfinite(edge_values)
            if plot_ns_edges:
                draw_edge_mask = finite_edge_mask
            else:
                draw_edge_mask = finite_edge_mask & (edge_pvals < thresh)

            # -------------------------
            # Widths: either static or scaled
            # -------------------------
            widths = {}
            if static_lw is not None:
                for i in range(len(edge_values)):
                    if draw_edge_mask[i]:
                        widths[i] = float(static_lw)
                show_scalebar = False  # suppress scale bar when width is fixed
            else:
                vals_for_width = edge_values[finite_edge_mask]
                idxs_for_width = np.nonzero(finite_edge_mask)[0]

                if len(vals_for_width):
                    vmin, vmax = float(np.min(vals_for_width)), float(np.max(vals_for_width))
                    a, b = linewidth_range
                    if np.isclose(vmin, vmax):
                        scaled = np.full_like(vals_for_width, np.mean(linewidth_range), dtype=float)
                    else:
                        scaled = a + (vals_for_width - vmin) * (b - a) / (vmax - vmin)
                    for i, w in zip(idxs_for_width, scaled):
                        widths[i] = float(w)

            # -------------------------
            # Group edges by pair so we can compute offsets for multi-edge pairs
            # -------------------------
            edges_by_pair = defaultdict(list)
            for i, ((u, v), rnk, val, pval, draw) in enumerate(
                    zip(pairs, ranks, edge_values, edge_pvals, draw_edge_mask)
            ):
                if not draw:
                    continue
                key = (u, v)
                edges_by_pair[key].append(i)

            # For each pair compute offset vectors for each edge index
            pair_offsets = {}
            max_offset_fraction = 0.05
            for (u, v), idx_list in edges_by_pair.items():
                m = len(idx_list)
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                dx, dy = x1 - x0, y1 - y0
                L = np.hypot(dx, dy)
                if L == 0:
                    normals = [(0.0, 0.0)] * m
                else:
                    nx, ny = -dy / L, dx / L
                    span = max_offset_fraction * L * max(1.0, m - 1)
                    if m == 1:
                        offsets = [0.0]
                    else:
                        offsets = np.linspace(-span / 2.0, span / 2.0, m)
                    normals = [(nx * off, ny * off) for off in offsets]
                for idx, off in zip(idx_list, normals):
                    pair_offsets[idx] = off

            # -------------------------
            # Optional rank-colouring fallback
            # -------------------------
            drawn_idxs = [i for i, d in enumerate(draw_edge_mask) if d]
            unique_ranks = sorted(set(ranks[i] for i in drawn_idxs))

            if color_by_rank and len(unique_ranks) > 0:
                cmap = mpl.cm.get_cmap(cmap_name)
                n_rank_colors = max(1, len(unique_ranks))
                if cmap_name in ("tab10", "tab20", "tab20b", "tab20c"):
                    colors_list = [cmap(i % cmap.N) for i in range(n_rank_colors)]
                else:
                    colors_list = [cmap(i / max(1, n_rank_colors - 1)) for i in range(n_rank_colors)]
                rank_to_color = {r: colors_list[i] for i, r in enumerate(unique_ranks)}
            else:
                rank_to_color = {}

            # -------------------------
            # Build segments, LWs, colours
            # -------------------------
            edge_style = {"-": [], "--": []}
            edge_cols = {"-": [], "--": []}
            edge_lws = {"-": [], "--": []}
            edge_label_infos = []

            for i, ((u, v), val, pval, draw, rnk, canon) in enumerate(
                    zip(pairs, edge_values, edge_pvals, draw_edge_mask, ranks, pair_canons)
            ):
                if not draw:
                    continue

                style = "-" if pval < thresh else "--"
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                off = pair_offsets.get(i, (0.0, 0.0))
                mx_off, my_off = mx + off[0], my + off[1]
                x0_off, y0_off = x0 + off[0], y0 + off[1]
                x1_off, y1_off = x1 + off[0], y1 + off[1]

                segs = [(x0_off, y0_off), (x1_off, y1_off)]
                edge_style[style].append(segs)

                # colour priority:
                # 1) circuit_map + colour_map
                # 2) rank colour if requested
                # 3) fallback edge_color
                col = edge_color
                if circuit_map is not None and colour_map is not None:
                    group = pair_to_group.get(canon, None)  # canon = df row’s pair_canon
                    if group is not None:
                        col = colour_map.get(group, edge_color)
                elif color_by_rank:
                    col = rank_to_color.get(rnk, edge_color)

                edge_cols[style].append(col)

                # linewidth
                lw = widths.get(i, static_lw if static_lw is not None else np.mean(linewidth_range))
                edge_lws[style].append(lw)

                # annotation
                if not np.isnan(val):
                    edge_label_text = f"{val:.2f}"
                else:
                    edge_label_text = ""
                edge_label_text = f"{edge_label_text} (r={int(rnk)})" if edge_label_text else f"r={int(rnk)}"
                edge_label_infos.append((mx_off, my_off, edge_label_text, i))

            # -------------------------
            # Plot
            # -------------------------
            with mpl.rc_context(rc_context):
                ax = hs.get_ax(where)
                hs.add_image_background_norm(ax, fname=background_png)

                # ---- Node circles ----
                for n in nodes:
                    if n in nodes_with_finite_edges:
                        ax.scatter(
                            *coords[n],
                            s=node_size,
                            color=node_colors[n],
                            edgecolor=fw.region_colours()[n],
                            linewidth=1,
                            zorder=2,
                        )

                # ---- Node text ----
                for n in nodes:
                    ax.text(
                        coords[n][0], coords[n][1],
                        node_labels[n],
                        ha="center", va="center",
                        fontsize=label_fontsize,
                        zorder=3,
                    )

                # ---- Edges ----
                for style, seg_list in edge_style.items():
                    if not seg_list:
                        continue
                    segs = [[(x0, y0), (x1, y1)] for (x0, y0), (x1, y1) in seg_list]
                    cols = edge_cols[style]
                    lws = edge_lws[style]
                    ax.add_collection(LineCollection(
                        segs,
                        linewidths=lws,
                        colors=cols,
                        linestyle=style,
                        zorder=1,
                    ))

                # ---- Edge annotations ----
                if annotate_edges:
                    for mx, my, txt, i in edge_label_infos:
                        ax.text(
                            mx, my, txt,
                            ha="center",
                            va="center",
                            fontsize=edge_label_fontsize,
                            zorder=4,
                        )

                ax.set_aspect("equal")

                if title is not None:
                    ax.set_title(title, pad=14)

                if epoch_png_path is not None:
                    # axes-fraction placement (not 'data') so the icon sits a fixed distance
                    # above the title regardless of how each panel's data range happens to scale
                    hs.add_png_to_plot(ax=ax, img_path=epoch_png_path, x=.5, y=1.55, zoom=0.08,
                                       coords='axes fraction')

                if ylabel is not None:
                    ax.text(0, 0.5, ylabel, rotation=90, va="center")

                # ---- Scalebar ----
                if show_scalebar:
                    hs.add_edge_width_scalebar_quantiles(
                        ax=ax,
                        edge_values=edge_values,
                        linewidth_range=linewidth_range,
                        global_edge_range=global_edge_range,
                        **(scalebar_kwargs or {}),
                    )

                # ---- Rank legend only makes sense if edges are coloured by rank ----
                if add_rank_legend and color_by_rank and len(unique_ranks) > 0:
                    handles = [
                        Line2D([0], [0], color=rank_to_color[r], lw=3, label=f"rank {r}")
                        for r in unique_ranks
                    ]
                    ax.legend(
                        handles=handles,
                        title="CCA rank",
                        loc="upper right",
                        bbox_to_anchor=(1.18, 1.02),
                        frameon=False,
                    )

                return ax

        @staticmethod
        def plot_node_graph_4(
                coords: dict,
                edge_params: "pd.DataFrame",
                thresh: float,
                node_colors=None,
                node_labels=None,
                linewidth_range=(1, 2),
                node_size=100,
                label_fontsize=3,
                where=(2.1, 2),
                rc_context=hs.no_axes,
                background_png=None,
                annotate_edges=True,
                plot_ns_edges=True,
                edge_label_fontsize=5,
                title=None,
                ylabel=None,
                epoch_png_path=None,
                nodes_with_finite_edges=None,
                global_edge_range=None,
                show_scalebar=False,
                coef_cmap=None,
                add_coef_legend=True,
                add_rank_alpha_legend=True,
                # 🔥 NEW
                edge_var_col="edge_var",
                show_variance_halo=False,
                variance_alpha=0,
                variance_linewidth_scale=0,
                # 🔥 NEW
                coef_leg_ax=None,
                rank_leg_ax=None,
                show_legends=True,
                view=None,

                **scalebar_kwargs,
        ):
            """
            view : None or ((x0, x1), (y0, y1))
                Data-space window shown in the axes. The default (None) keeps the unit square the
                coordinates in `coords` live in; fw.node_plot_coords() only spans roughly
                x 0.26-0.92, y 0.15-0.73 of it, so passing a tighter window blows the diagram up
                to fill its axes.
            """

            required_cols = {'edge_label', 'rank', 'edge_value', 'edge_pval', 'coefficient'}
            if not required_cols.issubset(edge_params.columns):
                raise ValueError(f"edge_params must contain {required_cols}")

            def _parse_label(l):
                if isinstance(l, (tuple, list)):
                    return (l[0], l[1])
                parts = l.replace("-", "_").split("_")
                return (parts[0], parts[1])

            df = edge_params.copy().reset_index(drop=True)
            df['pair'] = df['edge_label'].apply(_parse_label)
            edge_vars = df[edge_var_col].to_numpy(dtype=float) if edge_var_col in df.columns else None

            pairs = list(df['pair'])
            ranks = df['rank'].astype(int).to_numpy()
            edge_values = df['edge_value'].to_numpy(dtype=float)
            edge_pvals = df['edge_pval'].to_numpy(dtype=float)
            coefs = df['coefficient'].to_numpy()

            nodes = list(coords.keys())

            if node_colors is None:
                node_colors = {n: "white" for n in nodes}
            if node_labels is None:
                node_labels = {n: str(n) for n in nodes}

            # -------------------------
            # Node inclusion
            # -------------------------
            nodes_with_finite_edges = set() if nodes_with_finite_edges is None else set(nodes_with_finite_edges)
            if not nodes_with_finite_edges:
                for (u, v), val in zip(pairs, edge_values):
                    if np.isfinite(val):
                        nodes_with_finite_edges.add(u)
                        nodes_with_finite_edges.add(v)

            # -------------------------
            # Edge mask
            # -------------------------
            finite_mask = np.isfinite(edge_values)
            draw_mask = finite_mask if plot_ns_edges else finite_mask & (edge_pvals < thresh)
            drawn_idxs = np.where(draw_mask)[0]

            # -------------------------
            # Width scaling
            # -------------------------
            widths = {}
            vals = edge_values[finite_mask]
            idxs = np.where(finite_mask)[0]

            if len(vals):
                vmin, vmax = np.min(vals), np.max(vals)
                a, b = linewidth_range
                scaled = np.full_like(vals, np.mean(linewidth_range)) if np.isclose(vmin, vmax) \
                    else a + (vals - vmin) * (b - a) / (vmax - vmin)
                for i, w in zip(idxs, scaled):
                    widths[i] = float(w)

            # -------------------------
            # Edge offsets
            # -------------------------
            edges_by_pair = defaultdict(list)
            for i, (p, draw) in enumerate(zip(pairs, draw_mask)):
                if draw:
                    edges_by_pair[p].append(i)

            pair_offsets = {}
            for (u, v), idx_list in edges_by_pair.items():
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                dx, dy = x1 - x0, y1 - y0
                L = np.hypot(dx, dy)

                if L == 0:
                    offsets = [(0, 0)] * len(idx_list)
                else:
                    nx, ny = -dy / L, dx / L
                    offs = np.linspace(-0.05 * L, 0.05 * L, len(idx_list))
                    offsets = [(nx * o, ny * o) for o in offs]

                for idx, off in zip(idx_list, offsets):
                    pair_offsets[idx] = off

            # -------------------------
            # Colour (coefficient)
            # -------------------------
            unique_coefs = sorted(set(coefs[i] for i in drawn_idxs))
            if coef_cmap is None:
                cmap = mpl.cm.get_cmap("tab10")
                coef_to_color = {c: cmap(i % cmap.N) for i, c in enumerate(unique_coefs)}
            else:
                coef_to_color = coef_cmap

            # -------------------------
            # Rank → opacity
            # -------------------------
            unique_ranks = sorted(set(ranks[i] for i in drawn_idxs))
            rank_to_alpha = {
                r: 1.0 - (i / max(1, len(unique_ranks) - 1)) * 0.7
                for i, r in enumerate(unique_ranks)
            }

            # -------------------------
            # Build edges
            # -------------------------
            edge_style = {'-': [], '--': []}
            edge_cols = {'-': [], '--': []}
            edge_lws = {'-': [], '--': []}

            edge_style_var = {'-': [], '--': []}
            edge_cols_var = {'-': [], '--': []}
            edge_lws_var = {'-': [], '--': []}

            edge_label_infos = []

            for i, ((u, v), val, pval, draw, rnk) in enumerate(
                    zip(pairs, edge_values, edge_pvals, draw_mask, ranks)):

                if not draw:
                    continue

                style = '-' if pval < thresh else '--'
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                off = pair_offsets.get(i, (0, 0))

                seg = [(x0 + off[0], y0 + off[1]), (x1 + off[0], y1 + off[1])]

                # main edge
                edge_style[style].append(seg)

                coef = coefs[i]
                base_color = coef_to_color.get(coef, "black")
                alpha = rank_to_alpha.get(rnk, 1.0)
                col = (*mpl.colors.to_rgb(base_color), alpha)

                edge_cols[style].append(col)
                edge_lws[style].append(widths.get(i, np.mean(linewidth_range)))

                # variance halo
                if edge_vars is not None and np.isfinite(edge_vars[i]) and show_variance_halo:
                    var_val = edge_vars[i]
                    edge_style_var[style].append(seg)
                    edge_cols_var[style].append((*mpl.colors.to_rgb(base_color), variance_alpha))

                    # make the halo wider for larger variance
                    halo_lw = widths.get(i, np.mean(linewidth_range)) + (
                            variance_linewidth_scale * var_val
                    )
                    edge_lws_var[style].append(float(halo_lw))

                mx = (x0 + x1) / 2 + off[0]
                my = (y0 + y1) / 2 + off[1]
                edge_label_infos.append((mx, my, f"{val:.2f} (r={rnk})"))

            # -------------------------
            # Plot
            # -------------------------
            with mpl.rc_context(rc_context):
                ax = hs.get_ax(where)
                hs.add_image_background_norm(ax, fname=background_png)

                for n in nodes:
                    if n in nodes_with_finite_edges:
                        ax.scatter(*coords[n], s=node_size,
                                   color=node_colors[n],
                                   edgecolor=fw.region_colours()[n],
                                   linewidth=1,
                                   zorder=3)

                for n in nodes:
                    ax.text(*coords[n], node_labels[n], zorder=4,
                            fontsize=label_fontsize, ha="center", va="center")

                # variance halo first
                if show_variance_halo and edge_vars is not None:
                    for style in edge_style_var:
                        if edge_style_var[style]:
                            ax.add_collection(LineCollection(
                                edge_style_var[style],
                                colors=edge_cols_var[style],
                                linewidths=edge_lws_var[style],
                                linestyle=style,
                                zorder=0
                            ))

                # main edges on top
                for style in edge_style:
                    if edge_style[style]:
                        ax.add_collection(LineCollection(
                            edge_style[style],
                            colors=edge_cols[style],
                            linewidths=edge_lws[style],
                            linestyle=style,
                            zorder=1
                        ))

                if annotate_edges:
                    for x, y, txt in edge_label_infos:
                        ax.text(x, y, txt, fontsize=edge_label_fontsize, zorder=2,
                                ha="center", va="center")

                if view is not None:
                    ax.set_xlim(*view[0])
                    ax.set_ylim(*view[1])
                ax.set_aspect("equal")

                if title:
                    ax.set_title(title)

                if epoch_png_path:
                    # axes-fraction coordinates: identical to the old data coordinates for the default unit-square
                    # view, but stays put above the axes when `view` crops the data window (a data-coordinate
                    # annotation would also be dropped, since its anchor point falls outside the cropped axes)
                    hs.add_png_to_plot(ax=ax, img_path=epoch_png_path, x=.5, y=1.03, zoom=0.08, coords='axes fraction')

                if ylabel:
                    ax.text(0, 0.5, ylabel, rotation=90, va="center")

                # -------------------------
                # Legends (NEW SYSTEM)
                # -------------------------
                if show_legends:

                    # fallback: draw everything on main axis if none provided
                    coef_ax = hs.get_ax(coef_leg_ax) if coef_leg_ax is not None else ax
                    rank_ax = hs.get_ax(rank_leg_ax) if rank_leg_ax is not None else ax

                    # ---- 1. COEFFICIENT LEGEND (HORIZONTAL) ----
                    if add_coef_legend and len(unique_coefs):

                        coef_handles = [
                            Line2D([0], [0], color=coef_to_color[c], lw=3, label=str(c))
                            for c in unique_coefs
                        ]

                        leg1 = coef_ax.legend(
                            handles=coef_handles,
                            title="Regressor",
                            loc="upper center",
                            bbox_to_anchor=(0.5, 1.2),
                            ncol=len(unique_coefs),  # 🔥 horizontal layout
                            frameon=False
                        )

                        if coef_ax is not ax:
                            coef_ax.axis("off")

                    # ---- 2. RANK GRADIENT (HORIZONTAL BAR) ----
                    if add_rank_alpha_legend and len(unique_ranks) > 1:

                        norm = mpl.colors.Normalize(vmin=min(unique_ranks), vmax=max(unique_ranks))

                        # Create custom colormap: black → transparent
                        cmap = mpl.cm.Greys
                        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])

                        cbar = plt.colorbar(
                            sm,
                            ax=rank_ax,
                            orientation="horizontal",  # 🔥 horizontal
                            fraction=0.8,
                            pad=0.2
                        )

                        cbar.set_label("CCA rank (opacity)")
                        cbar.ax.invert_xaxis()  # high rank → more transparent

                        if rank_ax is not ax:
                            rank_ax.axis("off")

                # ---- 3. SCALEBAR (BOTTOM LEFT, NON-OVERLAPPING) ----
                if show_scalebar:
                    hs.add_edge_width_scalebar_quantiles(
                        ax=ax,
                        edge_values=edge_values,
                        linewidth_range=linewidth_range,
                        global_edge_range=global_edge_range,
                        **(scalebar_kwargs or {})
                    )

                return ax


        @staticmethod
        def plot_heatmap_1(mat: np.array, ax: plt.subplot, ytitle: str = None, xtitle: str = None,
                           vs: list = None, vlims: list = None, clabel: str = None,
                           yticks=None, yticks_on=True, yticklabs_on=False, yticklabels: list = None,
                           xticks=None, xticks_on=True, xticklabs_on=False, xticklabels: list = None,
                           palette='rocket', gaussian_sd=0, mask=None, cbar_ax=None):
            if gaussian_sd:
                _, smoother = hs.init_smoother()
                mat = smoother(mat, gaussian_sd)
            vs = [None, None] if vs is None else vs
            try:
                sns.heatmap(mat, ax=ax, cmap=palette, vmin=vs[0] if vlims is None else vlims[0],
                            vmax=vs[1] if vlims is None else vlims[1], cbar=True if clabel is not None else False,
                            cbar_kws={'label': clabel} if clabel is not None else {}, mask=mask, cbar_ax=cbar_ax)
            except ValueError:
                print(f'Scalebar was not added because of an error!')
            if yticks_on:
                ax.set_yticks(np.arange(0.5, mat.shape[0] + 0.5) if yticks is None else yticks)
            else:
                ax.tick_params(axis='y', which='major', length=0)
                ax.tick_params(axis='y', which='minor', length=0)

            if xticks_on:
                ax.set_xticks(np.arange(0.5, mat.shape[1] + 0.5) if xticks is None else xticks)
            else:
                ax.tick_params(axis='x', which='major', length=0)
                ax.tick_params(axis='x', which='minor', length=0)

            if mat.shape[0] > 1:
                yticklabels = ([1, mat.shape[0]] if yticklabels is None else yticklabels) if yticklabs_on else []
                ax.set_yticklabels(yticklabels, rotation=45)

            xticklabels = ([1, mat.shape[1]] if xticklabels is None else xticklabels) if xticklabs_on else []
            ax.set_xticklabels(xticklabels, rotation=45)

            if ytitle is not None:
                ax.set_ylabel(ytitle)

            if xtitle is not None:
                ax.set_xlabel(xtitle)

            if clabel is not None:
                cbar_ax = cbar_ax if cbar_ax is not None else ax
                if len(cbar_ax.collections):
                    try:
                        cbar_ax.collections[0].colorbar.ax.yaxis.set_ticks([], minor=True)
                    except AttributeError:
                        pass
            # Adjust legend (colorbar) font size
            try:
                colorbar = ax.collections[0].colorbar
                colorbar.ax.tick_params(labelsize=mpl.rcParams['xtick.labelsize'])  # Change to any desired font size
                colorbar.ax.tick_params(labelsize=10, labelrotation=45)
            except AttributeError:
                print(f'Could not adjust heatmap colourbar axis as it was empty.')


        def create_epoch_spike_dfs(self):
            if not hasattr(self, 'e_sp_df'):
                self.e_sp_df = {}
                if not hasattr(self, 'bin_max') and not hasattr(self, 'edges'):
                    self.bin_max, _, self.edges = self.get_bin_max_in_epochs(self.spike_times)
                for epoch, edges in self.edges.items():
                    if epoch not in self.e_sp_df.keys():
                        self.e_sp_df[epoch] = self.spike_times.query(f'epoch=="{epoch}"')


        def get_zero_ps(self):
            """A useful method if wanting to find out where the zero point is in epochs based on pre and post values"""
            if not hasattr(self, 'zero_ps'):
                self.zero_ps, self.posts = dict(), dict()
                if not hasattr(self, 'bin_max') and not hasattr(self, 'edges'):
                    self.bin_max, _, self.edges = self.get_bin_max_in_epochs(self.spike_times)
                for epoch, edges in self.edges.items():
                    if epoch not in self.zero_ps.keys():
                        self.zero_ps[epoch] = int(self.e_sp_df[epoch]['pre'].iloc[0] /
                                                  self.e_sp_df[epoch]['samp_int'].iloc[0])
                        self.posts[epoch] = int(self.e_sp_df[epoch]['post'].iloc[0] /
                                                self.e_sp_df[epoch]['samp_int'].iloc[0])


            # animation.save(f'{path}.gif', writer='pillow')


        def make_fr_matrices_1_1(self, fr_metric: str, sp_df: pd.DataFrame = None, sigma_ms: int = 0,
                               path: str = None, unit_col='unit_lab', parallel: bool = True) -> None:
            """
            Method to generate numpy arrays with aggregate data for populations of neurons from SpikeStats spike_times
            dataframe.
            Variant that AVERAGES across levels of `grp_var` (i.e. does NOT produce level-by-level output).
            :param sp_df: DataFrame:  to use instead of the one belonging to the class
            :param sigma_ms: apply a gaussian smoothing filter with a std of the specified ms
            :param fr_metric: str : one of either std_fr, norm_fr or raw_fr firing rate on the y axis
            :param path: path where to save the figure
            :param omit: dict: whether any levels of reward/effort need to be excluded
            :param unit_col: a specifier for the column containing unit labels in spike_times df
            :param parallel: whether to parallel compute the matrices
            :return:
            """
            path = path if path is not None else self.f_root
            if sp_df is None: self.attr_check('spike_times', path=path)
            areas, subjects = list(sp_df['area'].unique()), list(sp_df['subject'].unique())
            _, col_n, edges = self.get_bin_max_in_epochs(sp_df)
            cols, col_sel = sp_df.columns, []
            col_sel.append(fr_metric)
            [col_sel.append(i) for i in ['subject', 'area', 'epoch', 'bin', 'unit_lab']]
            spike_times = sp_df[col_sel]
            mat_p = os.path.join(path + '_fr_mats')  # Create a folder to store the firing rate matrices
            if not os.path.exists(mat_p): os.mkdir(mat_p)
            # Set up smoothing parameters
            bin_ms = int(re.search(r'\d+', fr_metric).group())
            sigma_bins = sigma_ms / bin_ms

            def worker(sp_t, area_, col_n_, edges_, filep_, epochs_):
                ar_df = sp_t.query(f'area=="{area_}"')
                fr_mat = np.zeros((ar_df[unit_col].nunique(), col_n_))
                for epoch in epochs_:
                    grp = ar_df.query(f'epoch=="{epoch}"').groupby(['unit_lab', 'bin'])[fr_metric].mean().unstack()
                    grp = grp.sort_values(by=['unit_lab'])[np.sort(grp.columns)]
                    fr_inp = grp.values
                    # Average over each bin in the epoch
                    if sigma_bins:
                        for i, trial in enumerate(fr_inp):
                            fr_inp[i, :] = gaussian_filter1d(trial, sigma=sigma_bins, mode='reflect')
                    fr_mat[:, edges_[epoch][0]:edges_[epoch][1] + 1] = fr_inp
                return fr_mat, filep_, grp.index.values

            if parallel:
                if not ray.is_initialized():
                    hs.init_ray()
                spt_ad = ray.put(spike_times)
                workers = []

                @ray.remote
                def ray_worker(*args):
                    return worker(*args)

            epochs = self.get_sorted_epochs(spike_times['epoch'].unique())
            for area in areas:
                ar_mat_p = os.path.join(mat_p, area)
                os.makedirs(ar_mat_p, exist_ok=True)
                filep = os.path.join(ar_mat_p, fr_metric)
                if parallel:
                    workers.append(ray_worker.remote(spt_ad, area, col_n, edges, filep, epochs))
                else:
                    res, _, units = worker(spike_times, area, col_n, edges, filep, epochs)
                    np.save(filep, res)
                    print(f'Saved {filep}')
                    np.save(os.path.join(ar_mat_p, f'{area}_units'), units)
            if parallel:
                while len(workers):
                    done_id, workers = ray.wait(workers)
                    res = ray.get(done_id[0])
                    if res is not None:
                        np.save(res[1], res[0])
                        print(f'Saved {res[1]}')
                        spl = hs.split_path(res[1])[:-3]
                        np.save(os.path.join(*spl, f'{spl[-1]}_units'), res[2])
                del spt_ad
                del workers
                ray.shutdown()


        def _sanitise_spike_times_df(self, fr_var: str = None, sp_df=None, fr_crit: float = 0.1, parallel=True,
                                     ray_initialised=False) -> pd.DataFrame:
            """
            Throw out units that have an overall firing rate less than fr_crit.
            :param fr_var: the firing rate variable to base the sanitisation on
            :param sp_df: spike timing and meta-data dataframe
            :param fr_crit: critical firing rate
            :param parallel: whether to parallelise code using Ray
            :return: sanitised spike_times dataframe
            """
            print(f'Cleaning spike_times dataframe from units with an overall firing rate < {fr_crit} Hz')
            spike_times = self.spike_times if sp_df is None else sp_df
            cols, col_sel = spike_times.columns, []
            if fr_var is None:
                if 'raw_fr' in cols:
                    fr_var = 'raw_fr'
                else:
                    fr_var = cols[np.where(['raw_fr' in i for i in list(cols)])[-1][-1]]
            col_sel.append(fr_var)
            [col_sel.append(i) for i in ['subject', 'area', 'unit_lab']]
            spike_times_ = spike_times[col_sel]
            areas = spike_times_['area'].unique().tolist()
            subjects = spike_times_['subject'].unique().tolist()

            def worker(df, area_, subject_):
                df.query(f'area=="{area_}" and subject=="{subject_}"', inplace=True)
                rm_idxs, removed = np.array([]), 0
                for unit in df['unit_lab'].unique():
                    if df.query(f'unit_lab == "{unit}"')[fr_var].mean() < fr_crit:
                        rm_idxs = np.append(rm_idxs, df.query(f'unit_lab == "{unit}"').index)
                        removed += 1
                if len(rm_idxs):
                    print(f'Dropped {removed} units from {subject_} {area_}')
                return rm_idxs

            if parallel:
                spike_times_ad = ray.put(spike_times_)
                if not ray.is_initialized():
                    hs.init_ray()

                @ray.remote
                def ray_worker(*args):
                    return worker(*args)

            workers, drop_idxs = [], np.array([])
            for area in areas:
                for subject in subjects:
                    print(f'Processing {subject} {area}')
                    if not parallel:
                        res = worker(spike_times_, area, subject)
                        drop_idxs = np.append(drop_idxs, res)
                    else:
                        workers.append(ray_worker.remote(spike_times_ad, area, subject))
            if parallel:
                while len(workers):
                    done_id, workers = ray.wait(workers)
                    res = ray.get(done_id[0])
                    if res is not None:
                        drop_idxs = np.append(drop_idxs, res)
                del spike_times_ad
                del workers
                if not ray_initialised:
                    ray.shutdown()

            return spike_times.drop(drop_idxs)

        def sanitise_spike_times_df(self, **args):
            return self._sanitise_spike_times_df(**args)
