"""
Compute GFP peak stats and test for ideal number of microstates.

(c) Nikola Jajcay
"""
import os
from copy import deepcopy

import numpy as np
import pandas as pd

import src.clustering_scores as scores
from src.helpers import (
    DATA_ROOT,
    RESULTS_ROOT,
    make_dirs,
    run_in_parallel,
    set_logger,
)
from src.recording import load_all_data

# 4: PSI-T3 - no data
# 13: PSI-T3 - lot of artefacts
# 14: PSI-T5 - no data
# 20: no PLA data
# 22: PSI-T1 - very short data
EXCLUDE_SUBJECTS = [4, 13, 14, 20, 22]

WORKERS = 5
NO_STATES_RANGE = np.arange(4, 11)
FILTER_OPTIONS = [(2.0, 20.0), (1.0, 30.0), (1.0, 40.0)]
# number of initialisations for each microstate computation
N_INITS = 200


def _process_recording(args):
    """
    Wrapper to process EEG recording. Computes number of GFP peaks.

    :param args: arguments for multiprocessing map, here recording and filter
        tuple
    :type args: tuple[`PsilocybinRecording` & tuple]
    """
    recording, filter_, n_states = args
    df = {
        "subject": recording.subject,
        "session": recording.session,
        "filter": f"{filter_[0]}-{filter_[1]}",
    }
    n_channels = recording.info["nchan"]
    recording.preprocess(filter_[0], filter_[1])
    recording.gfp()
    df["# GFP peaks"] = len(recording.gfp_peaks)

    df_corr = deepcopy(df)
    df_corr["method"] = "corr"
    recording.run_microstates(n_states=n_states, n_inits=N_INITS)
    recording.reassign_segmentation_by_midpoints(method="corr")
    df_corr["# states"] = n_states
    df_corr["PM variance total"] = scores.pascual_marqui_variance_test(
        1.0 - recording.gev_tot, n_states, n_channels
    )
    df_corr["PM variance GFP"] = scores.pascual_marqui_variance_test(
        1.0 - recording.gev_gfp, n_states, n_channels
    )
    df_corr["Davies-Bouldin"] = scores.davies_bouldin_test(
        recording.data.T, recording.segmentation
    )
    df_corr["Dunn"] = scores.dunn_test(recording.data.T, recording.segmentation)
    df_corr["Silhouette"] = scores.silhouette_test(
        recording.data.T, recording.segmentation
    )
    df_corr["Calinski-Harabasz"] = scores.calinski_harabasz_test(
        recording.data.T, recording.segmentation
    )

    df_gmd = deepcopy(df)
    df_gmd["method"] = "GMD"
    recording.run_microstates(n_states=n_states, n_inits=N_INITS)
    recording.reassign_segmentation_by_midpoints(method="GMD")
    df_gmd["# states"] = n_states
    df_gmd["PM variance total"] = scores.pascual_marqui_variance_test(
        1.0 - recording.gev_tot, n_states, n_channels
    )
    df_gmd["PM variance GFP"] = scores.pascual_marqui_variance_test(
        1.0 - recording.gev_gfp, n_states, n_channels
    )
    df_gmd["Davies-Bouldin"] = scores.davies_bouldin_test(
        recording.data.T, recording.segmentation
    )
    df_gmd["Dunn"] = scores.dunn_test(recording.data.T, recording.segmentation)
    df_gmd["Silhouette"] = scores.silhouette_test(
        recording.data.T, recording.segmentation
    )
    df_gmd["Calinski-Harabasz"] = scores.calinski_harabasz_test(
        recording.data.T, recording.segmentation
    )

    return pd.concat(
        [pd.DataFrame(df_corr, index=[0]), pd.DataFrame(df_gmd, index=[1])]
    )


def main():
    working_folder = os.path.join(
        RESULTS_ROOT, "gfp_and_no_mstates_4states_min"
    )
    make_dirs(working_folder)
    set_logger(log_filename=os.path.join(working_folder, "log"))
    data = load_all_data(os.path.join(DATA_ROOT, "processed"), EXCLUDE_SUBJECTS)
    assert len(data) % 5 == 0, len(data)

    results = run_in_parallel(
        _process_recording,
        [
            (deepcopy(recording), filter_, n_states)
            for recording in data
            for filter_ in FILTER_OPTIONS
            for n_states in NO_STATES_RANGE
        ],
        workers=WORKERS,
    )

    results = pd.concat(list(results), axis=0)
    results.to_csv(os.path.join(working_folder, "gfp_peaks_var_test.csv"))


if __name__ == "__main__":
    main()
