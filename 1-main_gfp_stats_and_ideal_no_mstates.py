"""
Compute GFP peak stats and test for ideal number of microstates.

(c) Nikola Jajcay
"""
import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from src.helpers import DATA_ROOT, RESULTS_ROOT, make_dirs, set_logger
from src.recording import load_all_data
from tqdm import tqdm

# 4: PSI-T3 - no data
# 13: PSI-T3 - lot of artefacts
# 14: PSI-T5 - no data
# 20: no PLA data
# 22: PSI-T1 - very short data
EXCLUDE_SUBJECTS = [4, 13, 14, 20, 22]

WORKERS = cpu_count()
NO_STATES_RANGE = np.arange(1, 11)
FILTER_OPTIONS = [(2.0, 20.0), (1.0, 40.0)]
# number of initialisation for each microstate computation
N_INITS = 500


def pm_variance_test(gev, no_states, n_channels):
    """
    Compute variance test according to Pascual-Marqui, R. D., Michel, C. M., &
    Lehmann, D. (1995). Segmentation of brain electrical activity into
    microstates: model estimation and validation. IEEE Transactions on
    Biomedical Engineering, 42(7), 658-665 ~ eq. (20).

    :param gev: global `unexplained` (i.e. 1 - explained variance) variance by
        microstate decomposition
    :type gev: float
    :param no_states: number of canonical states for decomposition
    :type no_states: int
    :param n_channels: number of channels in EEG recording
    :type n_channels: int
    """
    return gev * np.power(
        (1.0 / (n_channels - 1)) * (n_channels - 1 - no_states), -2
    )


def _process_recording(args):
    """
    Wrapper to process EEG recording. Computes number of GFP peaks.

    :param args: arguments for multiprocessing map, here recording and filter
        tuple
    :type args: tuple[`PsilocybinRecording` & tuple]
    """
    recording, filter_ = args
    out = [recording.subject, recording.session, filter_]
    n_channels = recording.info["nchan"]
    recording.preprocess(filter_[0], filter_[1])
    recording.gfp()
    out.append(len(recording.gfp_peaks))
    for n_states in NO_STATES_RANGE:
        recording.run_microstates(n_states=n_states, n_inits=N_INITS)
        out += [
            pm_variance_test(
                gev=1.0 - recording.gev_tot,
                no_states=n_states,
                n_channels=n_channels,
            ),
            pm_variance_test(
                gev=1.0 - recording.gev_gfp,
                no_states=n_states,
                n_channels=n_channels,
            ),
        ]
    assert len(out) == 4 + 2 * len(NO_STATES_RANGE)
    return out


def main():
    working_folder = os.path.join(RESULTS_ROOT, "gfp_and_no_mstates")
    make_dirs(working_folder)
    set_logger(log_filename=os.path.join(working_folder, "log"))
    data = load_all_data(os.path.join(DATA_ROOT, "processed"), EXCLUDE_SUBJECTS)
    assert len(data) % 5 == 0, len(data)

    pool = Pool(WORKERS)
    results = pool.map(
        _process_recording,
        tqdm(
            [
                (deepcopy(recording), filter_)
                for recording in data
                for filter_ in FILTER_OPTIONS
            ]
        ),
    )
    assert len(results) == 2 * len(data)
    pool.close()
    pool.join()

    results = pd.DataFrame(
        results,
        columns=["subject", "session", "filter", "# GFP peaks"]
        + sum(
            [
                [f"sigma_MCV total: {i} mstates", f"sigma_MCV GFP: {i} mstates"]
                for i in NO_STATES_RANGE
            ],
            [],
        ),
    )
    results.to_csv(os.path.join(working_folder, "gfp_peaks_var_test.csv"))


if __name__ == "__main__":
    main()
