# -*- coding: utf-8 -*-
# Created by Oliver Haermson  03/2020
# Copyright © 2020-2026 University of Oxford. All rights reserved.
# Distributed under the MIT license; see LICENSE.
"""
Constants shared by the analysis scripts: sampling rates, region colours and labels, the node layout of the
network diagrams and the data-version ids of the SpikeStats tables.
"""
vid_rate = 100 # Video sampling rate (100)
beh_rate = 1000  # The Teensy behaviour sampling rate (Hz)
ephys_rate = 20000  # Intan sampling rate (Hz)
trial_count = 180 # Ceiling of trial count in any recording session

version_ids = {2976638579: 1,  # task_params_oliver_v1
               1071738542: 2,  # task_params_isaac_v1
               1966638597: 3,  # task_params_oliver_v2
               1205839512: 4,  # task_params_isaac_v2
               }

# If nothing is specified, the base version without the below attributes will be loaded
spike_stats_ids = {'pairwise_corr': 573012,
                   'canonical_corr': 113873,
}

def region_colours():
    return {'mo-right': '#005cff',
            'acc-right': '#9c27b0',
            'str-right': '#1e1e1c',
            'vp-right': '#036838',
            'stn-right': '#a20045',
            'MO/VO': '#005cff',
            'ACC': '#9c27b0',
            'DMS': '#1e1e1c',
            'VP': '#036838',
            'STA': '#a20045',
            }


def venn_prm_colours():
    return {
        'trial_order': '#9400d3',
        'effort_lev': '#00bfff',
        'run_trial': '#0000ff',
        'np_exit_lat': '#ff00f3',
        'cue_err': '#6df900',
        'velocity': '#ff8500',
        'non_coding': '#c1c1c1',
        'run_trial:right_side': '#f92b58',
        'right_side': '#f92b58',
        'trial_count': '#d32f2f',
        'fr_area': '#37474f'
    }


def region_labels():
    return {'mo-right': 'MO/VO',
            'acc-right': 'ACC',
            'str-right': 'DMS',
            'vp-right': 'VP',
            'stn-right': 'STA',
            'mo': 'MO/VO',
            'acc': 'ACC',
            'str': 'DMS',
            'vp': 'VP',
            'stn': 'STA',
            'MO/VO': 'mo-right',
            'ACC': 'acc-right',
            'DMS': 'str-right',
            'VP': 'vp-right',
            'STA': 'stn-right',
            }

def node_plot_coords():
    return {
        "MO/VO": (0.26, 0.45),
        "ACC":   (0.69, 0.73),
        "DMS":   (0.53, 0.50),
        "VP":    (0.50, 0.15),
        "STA":   (0.92, 0.15),
    }



































