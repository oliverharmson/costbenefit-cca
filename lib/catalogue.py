# Distributed under the MIT license; see LICENSE.
"""
Locations of the data, figure and output folders of this codebase.

Every analysis project of the paper is identified by a six-digit project id (e.g. 382702). The source files and the
processed tables produced upstream of the scripts are stored flat in ``data/`` as ``<project id>_<name>.<ext>`` -- for
example ``data/382702_canonical_corr.feather`` -- and everything the scripts generate (figures, tables, cached
intermediates such as the running-speed table) is written to ``output/``.

This replaces the lab-wide catalogue (a table of project folders spread over several drives) the scripts were
originally written against. The folders can be moved with the environment variables ``CCA_DATA_DIR`` and
``CCA_OUTPUT_DIR``.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('CCA_DATA_DIR', os.path.join(ROOT, 'data'))
OUTPUT_DIR = os.environ.get('CCA_OUTPUT_DIR', os.path.join(ROOT, 'output'))
PNG_DIR = os.path.join(ROOT, 'pngs')  # epoch icons and the brain-slice drawing used by the network plots


def get_file_root(project_id) -> str:
    """Path prefix of a project's data files: ``get_file_root(382702) + '_canonical_corr'`` -> data/382702_canonical_corr."""
    return os.path.join(DATA_DIR, str(project_id))


def get_output_root(project_id) -> str:
    """Path prefix for the figures/tables a project's script writes (the folder is created if necessary)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, str(project_id))


def load_trial_behaviour(project_id):
    """Load the trial-level behaviour object (``.beh_df``) stored as ``data/<project id>_TrialBehaviour.pbz2``."""
    from block import BlockObj
    return BlockObj.TrialBehaviour().load(get_file_root(project_id))


def load_source_object(project_id, name):
    """
    Load one of the source objects stored as ``data/<project id>_<name>.pbz2``: ``TrialBehaviour``, ``FineBehaviour``,
    ``DinChannels`` (event channels), ``SpikeTrains`` or ``SpikeStats`` (e.g. ``load_source_object(958690, 'DinChannels')``).
    """
    from block import BlockObj
    return getattr(BlockObj, name)().load(get_file_root(project_id))
