# -*- coding: utf-8 -*-
# Created by Oliver Haermson  03/2020
# Copyright © 2020-2026 University of Oxford. All rights reserved.
# Distributed under the MIT license; see LICENSE.
"""
Helpers shared by the analysis scripts: plotting utilities (gridspec/axes handling, panel lettering, savefig, matplotlib rc
contexts), small statistics functions, pickle/feather I/O, and the classes that the stored source objects (event channels,
spike trains) are built from.
"""
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import functools

import bz2
import string
import pickle


import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import _pickle as c_pickle
from datetime import datetime
# from moviepy.editor import *
from skimage import transform
from numpy import append as ap
from numpy import where as wh
from numpy import diff as dif
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mticker
import subprocess
#from moviepy.editor import VideoFileClip
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.gridspec import SubplotSpec


# adc bit to micro volt scaling factor for intan rhd digitiser chips


class Config:
    """Figure-size helpers and the default keyword arguments for the plotting functions."""

    @staticmethod
    def landscape_A(n):
        """Returns a tuple with the size in inches of the corresponding A series
        paper size. e.g. for A4 call: ``landscape_A(4)``"""
        alpha = 1000 / 25.4 * 2 ** 0.25
        return alpha / 2 ** (n / 2.0), alpha / 2 ** ((n + 1) / 2.0)

    def portrait_A(self, n):
        """Returns a tuple with the size in inches of the corresponding A series
        paper size. e.g. for A4 call: ``portrait_A(4)``"""
        return self.landscape_A(n)[::-1]

    @property
    def plot_kwargs(self):
        return {'linestyle': '-',
                'linewidth': 1.0,
                'line_zorder': 2,
                'scat_zorder': 3,
                'color': [0, 0, 0],
                'offs': -0.1,
                'offs_incr': 0.05,
                'figsize': (8, 10),
                'marker': 'o', 's': 13,
                'legend': True,
                'grid': True,
                'fill_alpha': 0.3,
                'line_alpha': 0.7,
                'alpha': 0.5,
                'solid_capstyle': 'butt',
                'solid_joinstyle': 'round',
                'width': 0.5,
                'bottom': 0,
                'align': 'center',
                'edgecolor': "#212121",
                'capsize': 4}

    @property
    def scat_defs(self):
        return ['marker', 's', 'linewidth']

    @property
    def bar_defs(self):
        return ['width', 'bottom', 'align', 'edgecolor']

    @property
    def err_bar_defs(self):
        return ['alpha', 'linewidth', 'solid_capstyle', 'solid_joinstyle']


config = Config()
"""
archive_paths = config.archive_paths
archive_roots = config.archive_roots
"""


def landscape_A(n):
    """Returns a tuple with the size in inches of the corresponding A series
    paper size. e.g. for A4 call: ``landscape_A(4)``"""
    alpha = 1000/25.4 * 2**0.25
    return alpha / 2**(n/2.0), alpha / 2**((n+1)/2.0)


def portrait_A(n):
    """Returns a tuple with the size in inches of the corresponding A series
    paper size. e.g. for A4 call: ``portrait_A(4)``"""
    return landscape_A(n)[::-1]

# ---- Decorator functions ----


def buffered_property(func):
    """For use as the ``@buffered_property`` decorator. It is similar to the
    ``@property`` decorator but the decorated function is only called if
    there is no key matching the decorated function name in the dictionary
    ``self.buffer``. In addition to being returned, the calculated value
    is also added it to ``self.buffer``. If there is a match the value in
    ``self.buffer`` is returned without calling the decorated function.

    Note: While self.buffer will be created automatically if absent it is best
    to explicitly create it by adding ``self.buffer = {}`` to ``__init__()``
    """
    def wrapper(self):
        try:
            return self.buffer[func.__name__]
        except KeyError:
            self.buffer[func.__name__] = func(self)
            return self.buffer[func.__name__]
        except AttributeError:
            self._buffer = {}
            self._buffer[func.__name__] = func(self)
            return self._buffer[func.__name__]
    return property(wrapper)


# ----- Data container objects ------


class DinRadmacTtlMismatchError(Exception):
    pass


class RadmacChannels:
    """
    This class holds register data from Teensy3.6 recordings on the Radmac system. Channels
    are listed in the radmac_chanels.csv table in meta/raw. Methods found here are similar to
    those in AmplifierChannels and DinChannels.
    """

    def __init__(self, meta=None, data=None):
        self.buffer = {}
        if meta is not None:
            self.meta = meta
            self.meta['beh_samp_rate'] = 1000
        if data is not None:
            channel = data.channels[meta.beh_ch_lab]
            if bool(meta.bitdata):
                self._rise = channel['rise']
                self._fall = channel['fall']
                self._begin_level = channel['begin_level']
                self._end_time = int(channel['end_time'])
                self._end_level = channel['end_level']
            else:
                self._discrete_ser = channel['log']
                self.levels = np.unique(channel['log'])
                if isinstance(self._discrete_ser[0], str):
                    if len(np.where(self.levels == 'none')) > 0:
                        self.levels = np.delete(self.levels, np.where(self.levels == 'none'))
                self._time_points = channel['time']
                self._end_time = int(channel['end_time'])
            self.prm_meta = data.prm_meta


    @property
    def end_time(self):
        return self._end_time


    def rise(self, level=None):
        if bool(self.meta.bitdata):
            return self._rise
        else:
            f_times, f_reps = self.extract_timings(level)
            if f_times is not None:
                if f_reps is not None:
                    return np.array(ap(f_times[0], f_times[f_reps+1]), dtype=np.int64)
                return np.array(f_times[0], dtype=np.int64)
            else:
                return np.empty(0, dtype=np.int64)

    def fall(self, level=None):
        if bool(self.meta.bitdata):
            return self._fall
        else:
            f_times, f_reps = self.extract_timings(level)
            if f_times is not None:
                if f_reps is not None:
                    return np.array(ap(f_times[f_reps], f_times[-1]), dtype=np.int64)
                return np.array(f_times[-1], dtype=np.int64)
            else:
                return np.empty(0, dtype=np.int64)


    @property
    def begin_level(self):
        if bool(self.meta.bitdata):
            return self._begin_level
        else:
            return self._discrete_ser[0]

    @property
    def end_level(self):
        if bool(self.meta.bitdata):
            return self._end_level
        else:
            return self._discrete_ser[-1]

    @property
    def end_time(self):
        return self._end_time


    @staticmethod
    def logic_array(samp_rate: int = 1000, din_obj=None) -> np.array:
        """
        Initialise an array of 0s (Falses) for the entire length of the recording
        :arg samp_rate: desired sampling rate
        :arg din_obj: DinChannels object
        :return np array (dtype=bool)
        """
        n_samples = din_obj.rise[-1]-1-din_obj.rise[0]
        assert isinstance(n_samples, np.int64), "Digital-input sample time points have non-integer values"
        if samp_rate != din_obj.meta.ephys_samp_rate:
            n_samples = int(n_samples/(din_obj.meta.ephys_samp_rate/samp_rate))
        return np.zeros(n_samples, dtype=bool)

    def high(self, level: str = None, samp_rate: int = 1000, din_obj=None) -> list:
        """
        :arg level: str (the level of a non-boolean behavioural channel)
        :arg samp_rate: desired sampling rate
        :arg din_obj: DinChannels object
        :return:
        """
        sc = samp_rate/self.meta.beh_samp_rate  # Compute the scalar for up/down-scaling
        idx_sc = (din_obj.rise[-1]-1-din_obj.rise[0]) / \
                 (self.end_time*(din_obj.meta.ephys_samp_rate/self.meta.beh_samp_rate))
        sc *= idx_sc  # Correct for timing imprecision when up/down-sampling
        rise, fall = [], []
        for s in self.itr('high', level):
            rise.append(int(s[0]*sc))
            fall.append(int(s[1]*sc))
        z = zip(rise, fall)
        return [s for s in z]

    def high_logic_ar(self, samp_rate=1000, level=None, din_obj=None):
        """
        Retrieve an array of logical highs in an array constructed with the method .logic_array()
        :arg level: str (the level of a non-boolean behavioural channel)
        :arg samp_rate: desired sampling rate
        :arg din_obj: DinChannels object
        :return np array (dtype=bool)
        """
        error = 0.0001  # The allowed proportional mismatch between the number of rising edges
        try:
            assert len(din_obj.rise) - self.end_time < len(din_obj.rise)*error, \
                'The number of D-IN ttl pulses does not match the number of behaviour samples'
        except AssertionError:
            print(f'DIN and behaviour ttls have a differnce of {len(din_obj.rise) - self.end_time} '
                  f'in {din_obj.meta.stage}')
            raise DinRadmacTtlMismatchError
        except AttributeError:
            print(f'DIN object lacks necessary attributes in {din_obj.meta.stage}')
            raise EmptyDinError
        ar, high = self.logic_array(samp_rate, din_obj), self.high(level, samp_rate, din_obj)
        for isl in high:
            idxs = np.arange(isl[0], isl[1]).astype(np.int64)
            ar[idxs] = np.ones_like(idxs, dtype=bool)
        return ar


        # return [slice(*s) for s in self.itr('low')]

    def itr(self, logic, level=None):
        if logic == 'high':
            if bool(self.meta.bitdata):
                rise, fall = self.rise(), self.fall()
                if self.begin_level and 0 not in rise:
                    rise = np.concatenate([[0], rise])
                if self.end_level and self.end_time not in fall:
                    fall = np.concatenate([fall, [self.end_time]])
                # assert len(rise) == len(fall), f'length of rises is {len(rise)}, whereas that of falls is {len(fall)}'
            else:
                rise = self.rise(level)
                fall = self.fall(level)
            try: # If the whole stage is a series of logical highs, turn into lists for zip to work
                return zip(rise, fall)
            except TypeError:
                return zip([rise], [fall])
        elif logic == 'low':
            if bool(self.meta.bitdata):
                rise, fall = self.rise(), self.fall()
                if not self.begin_level and 0 not in fall:
                    fall = np.concatenate([[0], fall])
                if not self.end_level and self.end_time not in rise:
                    rise = np.concatenate([rise, [self.end_time]])
                # assert len(rise) == len(fall), f'length of rises is {len(rise)}, whereas that of falls is {len(fall)}'
            else:
                rise = self.rise(level)[1:]
                fall = self.fall(level)[:-1]
            try:
                return zip(rise, fall)
            except TypeError:
                return zip([rise], [fall])
        else:
            raise ValueError('invalid logic level string')


class DinChannels:
    """needs doc"""
    def __init__(self, meta=None, time_series=None, rftl=None, store_original=False):
        self.buffer = {}
        if meta is not None:
            self.meta = meta
        if time_series is not None:
            time_series = np.array(time_series)
            diff = dif(np.array(time_series).astype(np.int64))
            self._rise = wh(diff > 0)[0] + 1
            self._fall = wh(diff < 0)[0] + 1
            self._begin_level = bool(time_series[0])
            self._end_time = time_series.shape[-1]
            if store_original:
                self.buffer['time_series'] = time_series
        elif rftl is not None:
            self._rise = np.array(rftl[0])
            self._fall = np.array(rftl[1])
            self._end_time = int(rftl[2])
            if len(rftl) > 3:
                self._begin_level = bool(rftl[3])
                if len(self.rise) != 0 and len(self.fall) != 0:
                    assert(self._begin_level == self._calc_begin_level())
            else:
                self._begin_level = self._calc_begin_level()
        else:
            self.buffer['msg'] = 'No time series data given. Empty class'

    @property
    def rise(self):
        return self._rise

    @property
    def fall(self):
        return self._fall

    @property
    def begin_level(self):
        return self._begin_level

    @buffered_property
    def end_level(self):
        if len(self.rise) == 0:
            if len(self.fall) == 0:
                return self.begin_level
            elif len(self.fall) == 1:
                return False
        elif len(self.fall) == 0:
            if len(self.rise) == 1:
                return True
        elif self.rise[-1] > self.fall[-1]:
            return True
        else:
            return False
        raise ValueError

    @property
    def end_time(self):
        return self._end_time


    @buffered_property
    def high(self):
        return [slice(*s) for s in self.itr('high')]


    def itr(self, level):
        # Method that creates zipped lists of highs or lows
        rise, fall = self.rise, self.fall
        if level == 'high':
            if self.begin_level:
                rise = np.concatenate([[0], rise])
            if self.end_level:
                fall = np.concatenate([fall, [self.end_time]])
            assert len(rise) == len(fall)
            return zip(rise, fall)
        elif level == 'low':
            if not self.begin_level:
                fall = np.concatenate([[0], fall])
            if not self.end_level:
                rise = np.concatenate([rise, [self.end_time]])
            assert len(rise) == len(fall)
            return zip(fall, rise)
        elif level == 'full_high':
            if self.begin_level:
                fall = fall[1:]
            if self.end_level:
                rise = rise[:-1]
            assert len(rise) == len(fall)
            return zip(rise, fall)
        elif level == 'full_low':
            if not self.begin_level:
                rise = rise[1:]
            if not self.end_level:
                fall = fall[:-1]
            assert len(rise) == len(fall)
            return zip(fall, rise)
        else:
            raise ValueError('invalid logic level string')


class SpikeTrains:
    def __init__(self, stage_meta=None, spike_obj=None):
        if stage_meta is not None:
            self.meta = stage_meta.squeeze()  # Convert back from a Dataframe
            self.meta.region = spike_obj['target']
            self.meta.origin = spike_obj['origin']
            self.spike_times, self.unit_ids = spike_obj['spike_times'], spike_obj['unit_ids']
            self.stage_order = spike_obj['stage_order'] # Reflects the stage ordering in the concatenated recording
            self.ttl_trim = False # a flag for whether the spikes have been restricted to stage time


    @staticmethod
    def sorted_spike_matrix_0(x: np.ndarray, sort_by: str = 'max', abs=False, drop_na=True) -> (np.ndarray, np.ndarray):
        """
        Obtain a firing rate matrix that is sorted by the time incidence of maximum firing rate along the 0th dimension,
        which gives a tiled sorting artefact/impression.
        :param x: numpy array where units occupy individual rows and binned firing rates are ordered on the 1st dim
        :param sort_by: str: whether to sort by maximum or minimum along each row
        :param abs: bool: whether to look at the absolute values in the matrix
        :return: sorted numpy array, sorting indexes
        """
        if sum(np.isnan(x.flatten()) == True) and drop_na:
            drop_rows = np.unique(np.where(np.isnan(x))[0])
            x = np.delete(x, np.unique(np.where(np.isnan(x))[0]), axis=0)
            print(f'Dropped {len(drop_rows)} units from matrix due to a NaN coefficient value')
        if abs: x = np.abs(x)
        n_units = np.shape(x)[0]
        extremes = np.nanmax(x, axis=1) if sort_by == 'max' else np.nanmin(x, axis=1)
        ex_ts = np.zeros((1, n_units)).flatten()
        for i in range(n_units):
            ex_ts[i] = np.where(x[i, :] == extremes[i])[0][0]
        idxs = np.arange(0, n_units) # Initialise indexers
        mat = np.zeros((n_units, 2)) # Initialise sorting matrix
        mat[:, 0] = idxs.astype(int)
        mat[:, 1] = ex_ts.transpose()
        mat = mat[np.argsort(mat[:, 1])] # Sort by extremes of firing rate
        return x[mat[:, 0].astype(int)], mat[:, 0].astype(int)


# ------------- User input processing -------------------- #


# ---- Tables, meta data, and data load_ functions ----
# note this function is specific to a particular table structure


def extract_subject_from_meta_label(label):
    """
    A function for extracting the date from a block string
    :param label : str : for example 'oh011_201210a'. NB subject name is expected to be encoded
    as two ASCII characters and a 3-digit number combination.
    :return : str for example oh
    """
    bl_constr = re.findall('^[a-z][a-z][0-9][0-9][0-9]_', label)
    if len(bl_constr) == 0:
        bl_constr = re.findall('^[a-z]+_', label) # match the name 'test'
    return bl_constr[0][:-1]


def get_date_lab(label):
    """
    Get a date label from a stage label
    :param stage: str
    :return: str
    """
    date_s = list(re.findall('([0-9][0-9][0-9][0-9][0-9][0-9][a-z])', label))[0]
    return date_s[:-1]


def get_subj_lab(label):
    return extract_subject_from_meta_label(label)


def get_subj_date(label):
    return f'{get_subj_lab(label)}_{get_date_lab(label)}'


# --- Version control --- #


# --- File IO functions --- #


def load_pkl(*path_arg, decompress=True):
    ext = '.pbz2' if decompress else '.pkl'
    load_m = c_pickle if decompress else pickle
    if not path_arg[-1].endswith(ext):
        path_arg = *path_arg[:-1], path_arg[-1] + ext
    fp = os.path.join(*path_arg)
    if os.path.exists(fp) is False:
        fp = fp.replace('.pbz2', '.pkl') if decompress else fp.replace('.pkl', '.pbz2')
        load_m = pickle
        if os.path.exists(fp) is False:
            raise FileNotFoundError
    f = bz2.BZ2File(fp, 'rb') if '.pbz2' in fp else open(fp, 'rb')
    start = datetime.now()
    obj = load_m.load(f)
    print(f'Loaded in {fp} in {datetime.now()-start}')
    return obj


def save_df(df, path, ext='.feather', print_output=True):
    """
    Utility function that helps save pandas dataframes with compression and file type operations that enhance read/write
    speed.
    :param df: pandas Dataframe
    :param ext: str : one of either .feather or .parquet or csv
    :return: None
    """
    start = datetime.now()
    if ext == '.feather':
        df.to_feather(path + ext, compression='zstd')
    elif ext == '.parquet':
        df.to_parquet(path + ext, engine='pyarrow', compression='brotli')
    elif ext == '.csv':
        df.to_csv(path + ext, index=False)
    if print_output:
        print(f'Saved {path + ext} in in {datetime.now()-start}')


def load_df(path, ext='.feather', print_output=True):
    path = path + ext if ext not in path else path
    start = datetime.now()
    if ext == '.csv' or ext == '.zip':
        df = pd.read_csv(path, keep_default_na=False)
    elif ext == '.parquet':
        df = pd.read_parquet(path)
    elif ext == '.feather':
        df = pd.read_feather(path)
    if print_output:
        print(f'Loaded in {path} in {datetime.now()-start}')
    return df


# --- Misc --- #


def flatten_list(ls):
    return [item for sublist in ls for item in sublist]


# Terminal communication utilities


# --- Logical operations --- #

def parse_logical_statement(expr: str) -> (list, list, list, list):
    """
    Parse a logical expression into the constituent keyword arguments, their values and the
    connecting logical expressions. Only simple non-nested logical expressions are supported at this point.
    The value of the searches (on the right side of the == side NEED TO BE SURROUNDED BY '', regardless of
    whether the value is a string or not.
    Regex cheatsheet: https://www.dataquest.io/wp-content/uploads/2019/03/python-regular-expressions-cheat-sheet.pdf
    :param expr: the logical expression
        example: "trial_type == 'large' and beh_ch_lab == 'CUE'"
        example: "trial_count == '2' and state == 'EVAL'"
    :return
        vars: argument names
        vals: search value names
        b_ops: the operators between expressions
        w_ops: the within-expression operators
    """
    assert '(' not in expr and ')' not in expr, 'Only simple non-nested logic expressions are allowed at this point'
    b_ops, w_ops = [], []
    for bop in re.findall(r"(\sand\s|\sor\s)", expr, re.I):  # find the between-expression operators
        b_ops.append(bop.replace(' ', ''))
    for wop in re.findall(r"(\s==\s|\s!=\s)", expr, re.I):  # find the within-expression operators
        w_ops.append(wop.replace(' ', ''))
    expr = re.sub(r"((\sand\s)|(\sor\s))", ' ', expr)  # Format to get rid of the between-expression ops
    expr = re.sub(r"((\s==\s)|(\s!=\s))", ' ', expr)  # format to get rid of the within-expression ops
    expr = expr.split(' ')  # Parse the remainder as search arguments (left) and values (right, of the operator)
    vals, vars = [_.replace("'", '') for _ in expr[1::2]], expr[::2]
    assert len(vars) + len(vals), 'No arguments or values found'
    assert len(vals) == len(vars) == len(w_ops), 'The number of arguments, values and internal operators does not match'
    assert len(b_ops) < len(vals), 'The number of between-expression operators cannot equal the number of search vals'
    return vars, vals, b_ops, w_ops


def truth(op, *ars):
    """
    A function for finding truth values of pairs of vectors in *ars. Each vector is compared side-by-side and
    the logical AND operation performed.
    Parameters
    ---------
    op : str
        one of either 'AND' or 'OR' or '&' or '|' or 'and' or 'or'
    *ars : list of numpy arrays
    """
    ops = len(ars)-1
    for idx in range(ops):
        if not ((ars[idx] == 0) | (ars[idx] == 1)).all():
            raise ValueError('Array must be comprised of only 1s or 0s.')
        if len(ars[idx]) != len(ars[idx+1]):
            raise ValueError('Logical arrays must be the same length')
        if op in ['AND', 'and', '&']:
            tr = ars[idx] & ars[idx+1]
        elif op in ['OR', 'or', '|']:
            tr = ars[idx] | ars[idx + 1]
        else:
            raise ValueError('Operation {} has not been specified in instructions.'.format(op))
    return tr


def bool_highs(ar):
    """
    :param ar: numpy array of dtype=bool
    :return: zip object
    """
    if not ((ar == 0) | (ar == 1)).all():
        raise ValueError('Array must be a seriess of booleans.')
    if ar[-1]:  # If the last element of an array is a high, add a falling edge for meaningful extrapolation
        ar = np.append(ar, False)
    edges = wh(dif(ar))[0]
    starts, ends = edges[::2], edges[1::2]
    return zip(starts, ends)


# --- Array operations --- #


# --- General plotting helpers --- #


def mix_colours(col1, col2, col3=None):
    """
    Mixes two colors to compute a "mixed" color (for purposes of computing
    colors of the intersection regions based on the colors of the sets.
    Note that we do not simply compute averages of given colors as those seem
    too dark for some default configurations. Thus, we lighten the combination up a bit.
    https://github.com/konstantint/matplotlib-venn/blob/master/matplotlib_venn/_common.py
    Inputs are (up to) three RGB triples of floats 0.0-1.0 given as numpy arrays.

    > mix_colors(np.array([1.0, 0., 0.]), np.array([1.0, 0., 0.])).tolist()
    [1.0, 0.0, 0.0]
    > np.round(mix_colors(np.array([1.0, 1., 0.]), np.array([1.0, 0.9, 0.]), np.array([1.0, 0.8, 0.1])), 3).tolist()
    [1.0, 1.0, 0.04]
    """
    if isinstance(col1, str):
        col1 = matplotlib.colors.to_rgb(col1)
        col2 = matplotlib.colors.to_rgb(col2)
        if col3 is not None:
            col3 = matplotlib.colors.to_rgb(col3)
    if col3 is None:
        mix_color = 0.7 * np.array(col1) + np.array(col2)
    else:
        mix_color = 0.4 * (np.array(col1) + np.array(col2) + np.array(col3))
    mix_color = np.min([mix_color, [1.0, 1.0, 1.0]], 0)
    return mix_color

# matplotlib rc context dictionaries. Use: with mpl.rc_context(rc0):

rc0 = {
    'figure.constrained_layout.use': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'axes.labelsize': 8,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.pad': .25,
    'ytick.major.pad': .25,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'legend.fontsize': 6,
    'legend.frameon': False,
    'legend.title_fontsize': 8,
    'legend.borderpad': 0,
    'legend.edgecolor': 'None',
    'legend.labelspacing': 0.4,
    'legend.handletextpad': 0.6,
    'legend.loc': 'upper right',
    'lines.dash_capstyle': 'round',
    'figure.dpi': 600,
    'axes.titlesize': 10,
    'svg.fonttype': 'none',
    'font.family': ['Arial'],
    'mathtext.default': 'regular',
    }

no_axes = {
    'figure.constrained_layout.use': True,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 8,
    'xtick.top': False,
    'xtick.bottom': False,
    'xtick.minor.visible': False,
    'ytick.left': False,
    'ytick.right': False,
    'xtick.labeltop': False,
    'xtick.labelbottom': False,
    'ytick.labelleft': False,
    'ytick.labelright': False,
    'ytick.minor.visible': False,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'axes.grid': False,
    'axes.xmargin': 0,
    'axes.ymargin': 0,
    'legend.fontsize': 6,
    'legend.frameon': False,
    'legend.borderpad': 0,
    'legend.edgecolor': 'None',
    'legend.labelspacing': 0.4,
    'legend.handletextpad': 0.6,
    'legend.title_fontsize': 8,
    'legend.loc': 'upper right',
    'lines.dash_capstyle': 'round',
    'axes.titlesize': 10,
    'svg.fonttype': 'none',
    'font.family': ['Arial'],
    'mathtext.default': 'regular',
    'figure.dpi': 600,
    }

no_xaxis = {
    'figure.constrained_layout.use': True,
    'axes.spines.left': True,
    'axes.spines.bottom': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 8,
    'xtick.top': False,
    'xtick.minor.visible': False,
    'xtick.bottom': False,
    'ytick.left': True,
    'ytick.right': False,
    'xtick.labeltop': False,
    'xtick.labelbottom': False,
    'ytick.labelleft': True,
    'ytick.labelright': False,
    'ytick.minor.visible': False,
    'ytick.major.width': 0.5,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'ytick.major.size': 3,
    'ytick.major.pad': .25,
    'axes.grid': False,
    'axes.xmargin': 0,
    'axes.ymargin': 0,
    'axes.linewidth': 0.5,
    'legend.title_fontsize': 8,
    'legend.fontsize': 6,
    'legend.frameon': False,
    'legend.borderpad': 0,
    'legend.edgecolor': 'None',
    'legend.labelspacing': 0.4,
    'legend.handletextpad': 0.6,
    'legend.loc': 'upper right',
    'lines.dash_capstyle': 'round',
    'axes.titlesize': 10,
    'svg.fonttype': 'none',
    'font.family': ['Arial'],
    'mathtext.default': 'regular',
    'figure.dpi': 600,
    }

no_yaxis = {
    'figure.constrained_layout.use': True,
    'axes.spines.left': False,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 8,
    'xtick.top': False,
    'xtick.bottom': True,
    'xtick.minor.visible': False,
    'ytick.left': False,
    'ytick.right': False,
    'xtick.labeltop': False,
    'xtick.major.width': 0.5,
    'xtick.labelbottom': True,
    'ytick.labelleft': False,
    'ytick.labelright': False,
    'ytick.minor.visible': False,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.size': 3,
    'xtick.major.pad': .25,
    'axes.grid': False,
    'axes.linewidth': 0.5,
    'axes.xmargin': 0,
    'axes.ymargin': 0,
    'axes.titlesize': 10,
    'legend.title_fontsize': 8,
    'legend.fontsize': 6,
    'legend.frameon': False,
    'legend.borderpad': 0,
    'legend.edgecolor': 'None',
    'legend.labelspacing': 0.4,
    'legend.handletextpad': 0.6,
    'legend.loc': 'upper right',
    'lines.dash_capstyle': 'round',
    'svg.fonttype': 'none',
    'font.family': ['Arial'],
    'mathtext.default': 'regular',
    'figure.dpi': 600,
    }


def displace_axes(ax, prop=0.01, move_x=False):
    """
    Displace the y and x axis relative to each other
    :param ax: axis object
    :param prop: percentage of y/x limit to be displaced by
    :return: None
    """
    axy_prop = np.abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * prop
    axx_prop = np.abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * prop
    ax.spines["left"].set_position(("data", ax.get_xlim()[0]-0.5*axx_prop))
    if move_x:
        ax.spines["bottom"].set_position(("data", ax.get_ylim()[0]-0.5*axy_prop))


def savefig(path, ax=None, fig=None, legend=True, right=0.9, wspace=0.1, dpi=330, h_pad=0.1, pad_inches=.01,
            transparent=False, open=False, bbox_inches=None):
    if ax is not None:
        sanitise_empty_axes_for_gridspec(ax)
    if legend:
        # Adjust the scaling factor to fit your legend text completely outside the plot
        # (smaller "right" value results in more space being made for the legend)
        plt.subplots_adjust(right=right, wspace=wspace, hspace=wspace)
        bbox_inches = 'tight'
    if fig is not None:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, transparent=transparent)
    else:
        try:
            plt.tight_layout(h_pad=h_pad)
        except RuntimeError: # Sometimes using constrained and tight layout together causes an error
            pass
        plt.savefig(path, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi, transparent=transparent)
    if open:
        os_based_open(path)


def legend(ax, edge=False, loc='upper right', title=None, **kwargs):
    ax.legend(bbox_to_anchor=(1.2, 1), loc=loc, borderaxespad=0, edgecolor="None" if not edge else None,
              title=title, **kwargs)


def add_image_background_norm(ax, fname=None):
    if fname is not None:
        img = mpimg.imread(fname)
        ax.imshow(img, extent=[0, 1, 0, 1], origin='upper', aspect='auto', zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)          # keep image upright with top-left origin
    ax.set_axis_off()
    ax.set_autoscale_on(False)


def add_png_to_plot(ax, img_path: str, x=0, y=0, size_lim=None, zoom=0.2, coords='data') -> None:
    """
    :param coords: coordinate system for (x, y): 'data' (default, preserves prior behaviour) or
    'axes fraction'. Use 'axes fraction' when the placement should stay put relative to the axes
    box regardless of the data range (e.g. an icon anchored above a plot whose data limits vary
    panel to panel) -- 'data' placement drifts if the data range differs between panels.
    """
    img = matplotlib.image.imread(img_path)
    if size_lim is not None:
        if img.size > size_lim:
            while img.size > size_lim:
                img = transform.rescale(img, 0.5)
        elif img.size < size_lim:
            while img.size < size_lim:
                img = transform.rescale(img, 2)
    imagebox = OffsetImage(img, zoom=zoom)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy=(0, 0), xybox=(x, y), xycoords=coords, boxcoords=coords, frameon=False)
    ax.add_artist(ab)


def add_edge_width_scalebar_quantiles(
        ax,
        edge_values,
        linewidth_range,
        global_edge_range,
        qs=(0.1, 0.5, 0.9),
        loc=(0.05, 0.05),
        length=0.12,
        row_gap=0.06,
        text_offset=0.02,
        fontsize=6,
        scalebar_title="Edge value",
        color='k'):
    """
    Draw a linewidth scale bar using quantiles,
    but mapped using a global edge-value range.
    """

    ev = np.asarray(edge_values, float)
    ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        return

    if global_edge_range is None:
        q_vals = np.quantile(ev, qs)
    else:
        q_vals = np.quantile(global_edge_range, qs)

    # --- Edge width scale bar ---
    if global_edge_range is None:
        vmin, vmax = np.nanmin(edge_values), np.nanmax(edge_values)
    else:
        vmin, vmax = global_edge_range

    a, b = linewidth_range

    if np.isclose(vmin, vmax):
        w_vals = np.full_like(q_vals, np.mean(linewidth_range))
    else:
        w_vals = a + (q_vals - vmin) * (b - a) / (vmax - vmin)

    x0, y0 = loc

    ax.text(
        x0, y0 + row_gap * len(q_vals),
        scalebar_title,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha='left',
        va='bottom'
    )

    for i, (val, lw) in enumerate(zip(q_vals, w_vals)):
        y = y0 + i * row_gap
        ax.plot(
            [x0, x0 + length], [y, y],
            transform=ax.transAxes,
            lw=float(lw),
            color=color,
            solid_capstyle='round',
            clip_on=False
        )
        ax.text(
            x0 + length + text_offset, y,
            f"{val:.2g}",
            transform=ax.transAxes,
            va='center',
            fontsize=fontsize
        )


def _same_sps(ax_sps, target_sps):
    """Return True if two SubplotSpecs refer to the same slot."""
    if ax_sps is target_sps:
        return True
    # sometimes == works
    try:
        if ax_sps == target_sps:
            return True
    except Exception:
        pass
    # fallback: compare parent gridspec + indices
    try:
        return (
            ax_sps.get_gridspec() is target_sps.get_gridspec()
            and ax_sps.num1 == target_sps.num1
            and ax_sps.num2 == target_sps.num2
        )
    except Exception:
        return False

def get_ax_1(where=None, *, clear=False, create_if_missing=True):
    """
    Smart Axes getter/creator that REUSES axes in a gridspec slot.

    Parameters
    ----------
    where :
        - Axes -> returned as is
        - SubplotSpec -> returns existing Axes in that slot, or creates it
        - Figure -> add subplot(111) (or reuse if exists)
        - (w, h) -> create new figure with that size and add one Axes
        - None -> current figure / create new fig
    clear : bool
        If True, and an Axes is found, call ax.cla().
    create_if_missing : bool
        If False, and no Axes is found for `where`, return None.
    """
    # 1. Already an axes
    if isinstance(where, mpl.axes.Axes):
        ax = where
        if clear:
            ax.cla()
        return ax

    # 2. Specific slot in a GridSpec
    if isinstance(where, SubplotSpec):
        fig = where.get_gridspec().figure
        # try to FIND an existing axes in this slot
        for ax in fig.axes:
            # each axes made from a SubplotSpec has .get_subplotspec()
            sps = getattr(ax, "get_subplotspec", None)
            if sps is not None:
                try:
                    ax_sps = ax.get_subplotspec()
                except Exception:
                    continue
                if _same_sps(ax_sps, where):
                    if clear:
                        ax.cla()
                    return ax
        # no existing one -> create
        if not create_if_missing:
            return None

        ax = fig.add_subplot(where)
        if clear:
            ax.cla()
        return ax

    # 3. A figure object: reuse first axes or create one
    if isinstance(where, mpl.figure.Figure):
        fig = where
        if fig.axes:
            ax = fig.axes[0]
            if clear:
                ax.cla()
            return ax
        ax = fig.add_subplot(111)
        if clear:
            ax.cla()
        return ax

    # 4. A figure size: (width, height)
    if isinstance(where, (tuple, list)):
        fig = plt.figure(figsize=where)
        ax = fig.add_subplot(111)
        return ax

    # 5. Default: use current figure if exists
    fig = plt.gcf() if plt.get_fignums() else plt.figure()
    if fig.axes:
        ax = fig.axes[0]
        if clear:
            ax.cla()
        return ax
    ax = fig.add_subplot(111)
    if clear:
        ax.cla()
    return ax


def get_ax(where):
    return get_ax_1(where)


def gridspec(nrows=1, ncols=1, where=None, reuse_axes=False, **kwargs):
    """
    GridSpec or GridSpecFromSubplotSpec wrapper.
    Automatically creates the correct object depending on where.

    Parameters
    ----------
    nrows : int, optional
        Number of rows in grid. The default is 1.
    ncols : int, optional
        Number or columns in grid. The default is 1.
    where : Figure, SubplotSpec, None or (float, float), optional
        Object in which to place the grid. (float, float) specifies the size
        of a new figure. None creates a new figure of default size.
        The default is None.
    If `where` is:
      - Figure        -> add_gridspec to that figure
      - SubplotSpec   -> subgridspec
      - None          -> use *current* figure (or make one)
      - (w, h) tuple  -> make a new figure of that size
    reuse_axes: bool, optional : whether to reuse existing global axes or not (warning: may create awkward canvases!)
    **kwargs : keyword arguments
        All other parameters are passed to matplotlib.gridspec.GridSpec
        or GridSpecFromSubplotSpec

    Returns
    -------
    GridSpec or GridSpecFromSubplotSpec
        Newly created grid.
    """
    # 1) Figure
    if isinstance(where, mpl.figure.Figure):
        return where.add_gridspec(nrows, ncols, **kwargs)

    # 2) SubplotSpec
    if isinstance(where, mpl.gridspec.SubplotSpec):
        return where.subgridspec(nrows, ncols, **kwargs)

    # 3) Explicit tuple -> new figure
    if isinstance(where, tuple):
        fig = plt.figure(figsize=where)
        return fig.add_gridspec(nrows, ncols, **kwargs)

    # 4) None -> current figure (or create one)
    if reuse_axes:
        fig = plt.gcf()
    else:
        fig = plt.figure(figsize=where)
    return fig.add_gridspec(nrows, ncols, **kwargs)


def label_axes(axes, start='a', fontsize=20, fontweight='bold', pad=4, order='reading', **text_kwargs):
    """
    Stamp each axes in `axes` with sequential, bold panel letters (a, b, c, ...) just outside the
    top-left corner of each subpanel (in the margin above and to the left of the axes box, rather
    than overlapping its data area), in axes-fraction/offset-points coordinates so placement
    tracks each panel regardless of its data range.

    :param axes: iterable of matplotlib Axes to label. None entries are skipped.
    :param start: letter to start from, e.g. 'a' (default) or 'c' to continue a sequence.
    :param fontsize: label font size in points (default 20, per house style).
    :param fontweight: label font weight (default 'bold').
    :param pad: outward offset from the top-left corner, in points (shifts the letter up and to
        the left, i.e. outside the axes).
    :param order: 'reading' (default) sorts axes top-to-bottom then left-to-right by their
        figure position before labelling, so panel order matches visual layout regardless of
        the order axes were created in. 'given' preserves the input order as-is.
    :param text_kwargs: forwarded to Axes.annotate (e.g. color).
    :return: list of the created Text (annotation) artists, in label order.

    Note: a panel that also sets its own `ax.set_title(...)` places that title in the same margin
    above the axes, so the two can collide - increase `pad` (or adjust the title) for such panels.
    """
    axes = [ax for ax in axes if ax is not None]
    if not axes:
        return []

    if order == 'reading':
        axes = sorted(axes, key=lambda a: (-round(a.get_position().y1, 3), round(a.get_position().x0, 3)))
    elif order != 'given':
        raise ValueError(f"order must be 'reading' or 'given', got {order!r}")

    start_idx = string.ascii_lowercase.index(start)
    labels = []
    for i, ax in enumerate(axes):
        letter = string.ascii_lowercase[(start_idx + i) % len(string.ascii_lowercase)]
        txt = ax.annotate(letter, xy=(0, 1), xycoords='axes fraction',
                          xytext=(-pad, pad), textcoords='offset points',
                          fontsize=fontsize, fontweight=fontweight, ha='left', va='bottom', **text_kwargs)
        labels.append(txt)
    return labels


def label_panels(func=None, *, start='a', fontsize=20, fontweight='bold', pad=4, order='reading', skip=0,
                 count=None, **text_kwargs):
    """
    Decorator that alphabetically labels new subpanel(s) a wrapped plotting function creates.

    Wrap any function that builds one or more subplots via hs.get_ax(...)/hs.gridspec(...)
    (whether it starts a fresh figure via `where=(w, h)` or adds panels to an existing one) and
    each Axes it creates gets a sequential, bold panel letter just outside its top-left corner
    -- no need to change how the wrapped function is written or what it returns.

    Usage:
        @hs.label_panels
        def plot_my_figure(...):
            ...

        # or with options:
        @hs.label_panels(start='c', fontsize=16)
        def plot_my_figure(...):
            ...

    :param func: the plotting function being wrapped (supplied automatically by @hs.label_panels).
    :param skip: number of axes to leave unlabelled, counted in the order the wrapped function
        created them (before the 'reading order' sort). Use this when the function creates
        auxiliary axes -- e.g. a legend strip -- ahead of its real data panels.
    :param count: how many of the (post-skip) new axes to label, in `order`. Default None labels
        all of them. Use count=1 when a function internally builds several subplots (e.g. one per
        epoch) that should still read as a single lettered panel in the overall figure.
    :param start, fontsize, fontweight, pad, order, text_kwargs: forwarded to label_axes.
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            before = set(plt.gcf().axes) if plt.get_fignums() else set()
            result = f(*args, **kwargs)
            new_axes = [ax for ax in plt.gcf().axes if ax not in before][skip:]
            label_axes(new_axes[:count], start=start, fontsize=fontsize, fontweight=fontweight,
                      pad=pad, order=order, **text_kwargs)
            return result
        return wrapper
    return decorator(func) if func is not None else decorator


# ------------------ Signal processing and mathematical functions --------------------- #


def minmaxscale_1(x):
    """
    A custom implementation of scaling between 0 and 1
    :param x:
    :return:
    """
    return (x-np.min(x))/(np.max(x)-np.min(x))


def normalise_along_dim(data: np.ndarray, fn=minmaxscale_1, axis=0):
    data = np.array(data).astype(np.float64) if not isinstance(data, np.ndarray) else data
    if data.ndim == 1:
        return fn(data)
    else:
        for i, sl in enumerate(data.copy() if not axis else data.copy().transpose()):
            if not axis:
                try:
                    data[i, :] = fn(sl)
                except TypeError:
                    data[i, :] = fn.fit_transform(sl.reshape(-1, 1)).flatten()
            else:
                try:
                    data[:, i] = fn(sl)
                except TypeError:
                    data[:, i] = fn.fit_transform(sl.reshape(-1, 1))
    return data


def average_rhos(rhos: np.array, axis=0):
    z = np.arctanh(rhos)  # warning: atanh(nan) = nan
    z_mean = np.nanmean(z, axis=axis)  # (dim,)
    return np.tanh(z_mean)


def wilcoxon(x, y, alternative='two-sided', mode='approx') -> pd.Series:
    """
    A non-parametric test to compare paired samples
    This is a wrapper around stats.wilcoxon
    method = approx allows return of z-statistic in newer scipy versions
    otherwise try http://www.statisticslectures.com/topics/wilcoxonsignedranks/
    """
    test = stats.wilcoxon(x, y, alternative=alternative, mode=mode)
    ret = [test.zstatistic if hasattr(test, 'zstatistic') else test.statistic, test.pvalue]
    return pd.Series(ret, index=['statistic', 'pvalue'])


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05, lower=None) -> np.ndarray:
    """
    A function to detect below-threshold entries in a vector of family-wise error rate-corrected p-values,
    adapted from Isaac Grennan's ensemble toolbox. Theory: https://www.biostathandbook.com/multiplecomparisons.html
    :param p_values: p-values in a 1d numpy array
    :param alpha: the false discovery rate
    :param lower: a lower bound for detection where needing to band the p-values
    :return: an array of booleans where p-values corresponding to indices of True elements are below the Benjamini-
    Hochberg critical value
    """
    p_values = np.array(p_values) if not isinstance(p_values, np.ndarray) else p_values
    assert not sum(p_values == np.nan), f'nan values found in vector'
    assert p_values.ndim == 1, 'p-values must be in a 1d array'
    assert p_values.size > 0, 'p-values.size must be greater than 0'
    assert np.count_nonzero((0 <= p_values) & (p_values <= 1)) == p_values.size, \
        'p-values must be between 0 and 1, inclusive'
    assert (0 <= alpha) & (alpha <= 1), 'alpha must be between 0 and 1'

    p_sorted = np.sort(p_values) # Put the individual P values in order, from smallest to largest
    m = p_sorted.shape[0]
    # Compare each individual P value to its Benjamini-Hochberg critical value, Q(i/m), where i is the rank, m is the
    # total number of tests, and Q is the false discovery rate you choose.
    cutoff = alpha*np.arange(1, m+1)/m
    if lower is not None:
        lower = lower*np.arange(1, m+1)/m
        index = np.where((p_sorted < cutoff) & (p_sorted >= lower))[0]
    else:
        index = np.where(p_sorted < cutoff)[0]

    if index.size == 0:
        return np.repeat(False, p_values.shape[0])
    return p_values <= p_sorted[index[-1]]


# ------------------ Pandas DataFrame utilities ----------------- #


# ------------------ Ephys analysis helpers --------------------- #


# ------------------ Memory management --------------------- #


def create_dict_from_types(dtypes, n_fill=None, **char_len):
    d, kwargs = {}, {}
    if n_fill is not None:
        kwargs['vec_len'] = n_fill
    for tp, items in dtypes.items():
        for item in items:
            if item in char_len.keys():
                kwargs['char_len'] = char_len[item]
            d[item] = malloc(tp, **kwargs)
    return d


def malloc(tp, char_len=30, vec_len=10**6):
    """
    A convenience function for initialising memory for different types of vectors.
    :param tp: type : the type of a unit entry into the vector
    :param char_len: int : the length of a a unit of in a string array
    :param vec_len : int : the length of the initialising vector
    :return: numpy array
    """
    if isinstance(tp, str):
        tp = get_type_from_str(tp)

    if tp == str:
        ch_init = ''
        for _ in range(char_len):
            ch_init += ' '
        return np.repeat(ch_init, vec_len)
    elif tp == bool:
        return np.repeat(False, vec_len).astype(tp)
    else:
        return np.repeat(-1, vec_len).astype(tp)


def malloc_check(d: dict, d_types: dict, idx: int, n_fill=None) -> dict:
    """
    A function for checking if more memory needs to be allocated to arrays in a dictionary.
    :param d: dict
    :param d_types: dict: a dictionary with keys for variable types and values as lists containing entry names
    :param idx: the current index of operations in any of the values in dictionary d
    :param n_fill: int : if known, the the number of elements to be added before the next round of malloc_check
    :return: dict
    """
    def add_memory(dtypes, vec_len):
        for tp, items in dtypes.items():
            for item in items:
                d[item] = np.append(d[item], malloc(tp, vec_len=vec_len))

    if n_fill is None:
        if idx == d[list(d.keys())[0]].shape[0]:
            add_memory(d_types)
        return d
    if d[list(d.keys())[0]].shape[0] - idx < n_fill:
        add_memory(d_types, n_fill)
    return d


def get_type_from_str(s):
    if s == 'str':
        return str
    elif s == 'bool':
        return np.bool_
    else:
        if 'int' in s:
            if '8' in s:
                return np.int8
            if '16' in s:
                return np.int16
            if '32' in s:
                return np.int32
            if '64' in s:
                return np.int64
        elif 'float' in s:
            if '8' in s:
                return np.float8
            if '16' in s:
                return np.float16
            if '32' in s:
                return np.float32
            if '64' in s:
                return np.float64


def trim_dict_to_len(d: dict, n: int) -> dict:
    for i in d.keys():
        d[i] = d[i][:n]
    return d


def find_dict_len(d, dtypes, tp='int', filler=-1):
    for dtp, val in dtypes.items():
        if tp in dtp:
            try:
                return np.delete(d[val[0]], np.where(d[val[0]] == filler)[0], axis=0).shape[0]
            except IndexError:
                pass


# --- Performance and timing --- #


class ProgressBar:
    def __init__(self, sum_proc, add_msg=None, print_first=True, num_bars=20, process='analysing'):
        self.sum_proc = sum_proc
        self.bar_space = np.linspace(0, 1, num_bars)  # Create a progress bar percentile spaces
        self.add_msg = add_msg
        start = f'| '
        end = f' |'
        mid = "".join([' ' for _ in range(num_bars)])
        self.msg = start + mid + end
        self.proc_idx = 0
        self.n_bars = 0
        self.tic = datetime.now()
        self.proc_keyword = process
        print('') # Leave an empty line
        if print_first:
            self.print(first=True)

    def print(self, first=False):
        loop_enter = False
        if not first:
            self.proc_idx += 1
        temp = list(self.msg)
        while self.proc_idx / self.sum_proc >= self.bar_space[self.n_bars]:
            self.n_bars += 1
            temp[self.n_bars + 1] = '-'
            loop_enter = True
            if self.n_bars == len(self.bar_space):
                break
        if loop_enter:
            self.msg = "".join(temp)
            print(f'{self.msg} {int((self.proc_idx / self.sum_proc * 100))} % done {self.proc_keyword} '
                  f'{self.add_msg} in {datetime.now() - self.tic}')


def os_based_open(path):
    """Open a file with the operating system's default viewer."""
    opener = "open" if sys.platform == "darwin" else "start"
    try:
        subprocess.call([opener, path])
    except FileNotFoundError:
        os.startfile(path)


def sanitise_empty_axes_for_gridspec(ax, xlim=(0, 1), ylim=(0, 1), strategy="null"):
    """
    Make all axes in `gs` safe to save even if empty:
      - sets finite, non-identical limits on empty/degenerate axes
      - optionally disables ticks to avoid AutoLocator errors

    Parameters
    ----------
    gs : matplotlib.gridspec.GridSpec or SubplotSpec
    xlim, ylim : tuple
        Limits to apply to empty/degenerate axes.
    strategy : {"null", "linear", "off"}
        How to handle tick locators on sanitized axes:
          - "null": no ticks (NullLocator)  [safest default]
          - "linear": simple LinearLocator(2) (min & max)
          - "off": leave current locator but turn autoscale off
    """
    axes = _axes_for_gridspec(ax)
    for ax in axes:
        needs_sanitize = (not ax.has_data()) or _is_degenerate(ax)
        if not needs_sanitize:
            continue

        # Stop autoscaling so savefig won't try to re-autoscale an empty axis
        ax.set_autoscale_on(False)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if strategy == "null":
            ax.xaxis.set_major_locator(mticker.NullLocator())
            ax.yaxis.set_major_locator(mticker.NullLocator())
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.yaxis.set_minor_locator(mticker.NullLocator())
        elif strategy == "linear":
            ax.xaxis.set_major_locator(mticker.LinearLocator(2))
            ax.yaxis.set_major_locator(mticker.LinearLocator(2))
            # Optional: remove minor ticks to be extra safe
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.yaxis.set_minor_locator(mticker.NullLocator())
        elif strategy == "off":
            # Keep existing locators but ensure limits are fixed
            pass
        else:
            raise ValueError("strategy must be one of {'null','linear','off'}")
    return axes  # handy if you want to inspect which axes were touched


def _axes_for_gridspec(gs):
    """
    Return a flat list of Axes that belong to the given GridSpec (or SubplotSpec).
    Works for nested subgrids as well.
    """
    fig = gs.figure if hasattr(gs, "figure") else gs.get_gridspec().figure
    out = []
    for ax in fig.axes:
        try:
            sps = ax.get_subplotspec()
            # Walk up to the top-level gridspec for comparison
            while isinstance(sps, mpl.gridspec.SubplotSpec) and hasattr(sps, "get_gridspec"):
                if sps.get_gridspec() is gs:
                    out.append(ax)
                    break
                # If this SubplotSpec is a subgrid entry, climb one level
                parent = getattr(sps, "_gridspec", None)
                sps = parent.subplot_spec if getattr(parent, "subplot_spec", None) is not None else parent
        except Exception:
            # Axes without a SubplotSpec (e.g., inset axes) are ignored
            pass
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for ax in out:
        if ax not in seen:
            uniq.append(ax); seen.add(ax)
    return uniq


def _is_degenerate(ax):
    """True if limits are non-finite or identical (which triggers tick locator errors)."""
    try:
        xl = ax.get_xlim(); yl = ax.get_ylim()
        bad = (
            not np.isfinite(xl).all() or not np.isfinite(yl).all() or
            xl[0] == xl[1] or yl[0] == yl[1]
        )
        return bad
    except Exception:
        return True


class EmptyDinError(Exception):
    pass
