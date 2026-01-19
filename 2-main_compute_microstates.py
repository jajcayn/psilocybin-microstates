"""
Compute microstates per subject and session, compute mean microstate maps and
all microstate stats.

(c) Nikola Jajcay
"""

import logging
import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import xarray as xr
from src.helpers import (
    DATA_ROOT,
    PLOTS_ROOT,
    RESULTS_ROOT,
    make_dirs,
    set_logger,
)
from src.microstates import (
    load_Koenig_microstate_templates,
    match_reorder_microstates,
    plot_microstate_maps,
    segment,
)
from src.recording import load_all_data
from tqdm.rich import tqdm

# 4: PSI-T3 - no data
# 13: PSI-T3 - lot of artefacts
# 14: PSI-T5 - no data
# 20: no PLA data
# 22: PSI-T1 - very short data
EXCLUDE_SUBJECTS = [4, 13, 14, 20, 22]


# as (filter low, filter high, no. states)
MS_OPTIONS = [(2.0, 20.0, 4), (1.0, 40.0, 3)]
# MS_OPTIONS = [(1.0, 40.0, 4)]


def _append_or_create(dict_, key_, value):
    """
    If key_ is already in the dict_ append the value into the list, otherwise
    create list with value in it.

    :param dict_: dictionary to append to
    :type dict_: dict
    :param key_: key under which the value is appended
    :type key_: str
    :param value: value to append
    :type value: Any
    :return: appended dictionary
    :rtype: dict
    """
    if key_ not in dict_:
        dict_[key_] = [value]
    else:
        assert isinstance(dict_[key_], list)
        dict_[key_].append(value)
    return dict_


def _compute_microstates(args):
    """
    Wrapper to process EEG recording. Computes microstates.

    :param args: arguments for multiprocessing map, here recording, number
        of microstates and filter tuple
    :type args: tuple[`PsilocybinRecording` & tuple]
    """
    recording, ms_opts, n_inits = args
    recording.preprocess(ms_opts[0], ms_opts[1])
    recording.run_microstates(n_states=ms_opts[2], n_inits=n_inits)
    ms_templates, channels_templates = load_Koenig_microstate_templates(
        n_states=ms_opts[2]
    )
    recording.match_reorder_microstates(ms_templates, channels_templates)
    recording.reassign_segmentation_by_midpoints()
    recording.compute_segmentation_stats()
    recording.attrs = {"ms_opts": ms_opts}
    assert recording.computed_stats
    return recording


def main(
    folder: str,
    n_inits: int = 500,
    save_plots: bool = False,
    save_microstates: bool = False,
    plot_ext: Literal[".png", ".eps", ".pdf"] = ".png",
    workers: int = cpu_count(),
) -> None:
    working_folder = os.path.join(RESULTS_ROOT, folder)
    make_dirs(working_folder)
    set_logger(log_filename=os.path.join(working_folder, "log"))
    if save_plots:
        plotting_folder = os.path.join(PLOTS_ROOT, folder)
        make_dirs(plotting_folder)
        plt.style.use("default")
    data = load_all_data(os.path.join(DATA_ROOT, "processed"), EXCLUDE_SUBJECTS)
    assert len(data) % 5 == 0, len(data)

    logging.info("Computing microstates per subject and session...")
    pool = Pool(workers)
    data_ms = []
    for result in pool.imap_unordered(
        _compute_microstates,
        tqdm(
            [
                (deepcopy(recording), option, n_inits)
                for recording in data
                for option in MS_OPTIONS
            ]
        ),
    ):
        data_ms.append(result)

    # assert len(data_ms) == 2 * len(data)
    pool.close()
    pool.join()
    logging.info("Microstates computed.")

    if save_plots:
        # save individual maps
        logging.info("Plotting individual maps...")
        for recording in tqdm(data_ms):
            opts = recording.attrs["ms_opts"]
            filt_folder = os.path.join(
                plotting_folder, f"individual_maps_{opts[0]}-{opts[1]}filt"
            )
            if not os.path.exists(filt_folder):
                make_dirs(filt_folder)
            fname = os.path.join(
                filt_folder,
                f"{recording.subject}_{recording.session}_ind_maps{plot_ext}",
            )
            title = f"Subj.{recording.subject} ~ {recording.session}: "
            title += f"{opts[0]}-{opts[1]} Hz"
            plot_microstate_maps(
                recording.microstates,
                recording.info,
                xlabels=[
                    f"r={np.abs(corr):.3f} vs. template"
                    for corr in recording.corrs_template
                ],
                title=title,
                fname=fname,
                transparent=True,
            )
        # save group means
        logging.info("Computing group mean maps...")
        group_folder = os.path.join(plotting_folder, "group_maps")
        make_dirs(group_folder)
        ms_groups = {}
        for recording in tqdm(data_ms):
            opts = recording.attrs["ms_opts"]
            filt_str = f"{opts[0]}-{opts[1]}filt_{recording.session}"
            ms_groups = _append_or_create(
                ms_groups, filt_str, recording.microstates
            )
        for key, group_maps in ms_groups.items():
            n_states = 3 if "1.0-40.0" in key else 4
            # n_states = 4
            group_mean, _, _, _ = segment(
                np.concatenate(group_maps, axis=0).T,
                n_states=n_states,
                use_gfp=False,
                n_inits=n_inits,
                return_polarity=False,
            )
            ms_templates, channels_templates = load_Koenig_microstate_templates(
                n_states=n_states
            )
            # match channels
            _, idx_input, idx_sortby = np.intersect1d(
                data_ms[0].info["ch_names"],
                channels_templates,
                return_indices=True,
            )
            attribution, corrs_template = match_reorder_microstates(
                group_mean[:, idx_input],
                ms_templates[:, idx_sortby],
                return_correlation=True,
                return_attribution_only=True,
            )
            plot_microstate_maps(
                group_mean[attribution, :],
                data_ms[0].info,
                xlabels=[
                    f"r={np.abs(corr):.3f} vs. template"
                    for corr in corrs_template
                ],
                title=f"{key.replace('filt', 'Hz').replace('_', ' ')} group mean",
                fname=os.path.join(group_folder, f"group_mean_{key}{plot_ext}"),
                transparent=True,
            )

    if save_microstates:
        topographies = xr.combine_by_coords(
            [
                recording.get_segmentation_xarray().expand_dims(
                    ["subject", "session"]
                )
                for recording in data_ms
            ]
        )
        topographies.to_netcdf(os.path.join(working_folder, "topographies.nc"))
    # save ms stats
    logging.info("All done, saving.")
    full_df = pd.concat(
        [recording.get_stats_pandas(write_attrs=True) for recording in data_ms],
        axis=0,
    )
    full_df.to_csv(os.path.join(working_folder, "ms_stats.csv"))


if __name__ == "__main__":
    typer.run(main)
