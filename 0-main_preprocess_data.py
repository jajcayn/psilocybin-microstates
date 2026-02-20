"""
Preprocess raw data exported from BrainVision.

(c) Nikola Jajcay
"""

import logging
import os

import mne
import pandas as pd
from tqdm import tqdm

from src.helpers import DATA_ROOT, make_dirs, set_logger

# sampling rate of raw data
SAMPLING_RATE = 1000.0
# target sampling rate
RESAMPLE_TO = 256.0
# whether to rename data
RENAME = True
# target length seconds
TARGET_LENGTH = 40

RAW_DATA = (
    "/Volumes/Q/science-brain/UI-microstates/"
    "data_v2 - Palenicek-raw/raw_continuous"
)

KEEP_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
    "AFz",
]


def load_ascii_data_to_mne(filename: str) -> mne.io.RawArray:
    """
    Load ASCII data in text file to mne.

    :param filename: filename to load
    :type filename: str
    :return: mne raw array with EEG data
    :rtype: `mne.io.RawArray`
    """
    data = pd.read_csv(filename, delimiter=";")
    if len(data.columns) == 30:
        ch_types = ["eeg"] * 22 + ["eog"] * 4 + ["ecg", "misc", "misc", "misc"]
    elif len(data.columns) == 28:
        ch_types = ["eeg"] * 20 + ["eog"] * 4 + ["ecg", "misc", "misc", "misc"]
    info = mne.create_info(
        ch_names=list(data.columns), sfreq=SAMPLING_RATE, ch_types=ch_types
    )

    montage = "standard_1005"
    info.set_montage(montage)
    mne_data = mne.io.RawArray(data.values.T, info)
    mne_data.pick_channels(KEEP_CHANNELS)
    return mne_data


def main() -> None:
    target_folder = os.path.join(DATA_ROOT, "processed")
    make_dirs(target_folder)
    set_logger(log_filename=os.path.join(target_folder, "log"))
    # load info in excel file
    info = pd.read_excel(os.path.join(DATA_ROOT, "info_analyzy_psi_EC.xlsx"))
    info["Subj. & session No."] = info["Subj. & session No."].fillna("same")
    current_subject = None

    problematic = []
    for idx, row in tqdm(info.iterrows()):
        if row["Subj. & session No."] != "same":
            current_subject = row["Subj. & session No."]
        try:
            seg_suffix = "_Segmentation 2s nonoverlap skip bad_continuous.txt"
            data = load_ascii_data_to_mne(
                os.path.join(RAW_DATA, f"{row['filename']}{seg_suffix}")
            )
        except FileNotFoundError:
            problematic.append(f"{row['filename']} - file not found")
            continue

        assert isinstance(data, mne.io.RawArray)
        fin_time = row[
            "final legngth of non-overalpped artefact & sleep free data (s)"
        ]
        if data.n_times / SAMPLING_RATE != fin_time:
            problematic.append(f"{row['filename']} - time not matching")
        data.resample(RESAMPLE_TO)
        if RENAME:
            fname = (
                f"{current_subject}_T{row['EO']}_"
                f"{TARGET_LENGTH}s_256samp_raw.fif"
            )
            fname = os.path.join(target_folder, fname)
        else:
            fname = os.path.join(
                target_folder, row["filename"] + "256samp_raw.fif"
            )
        data.save(fname, picks=["eeg"], tmax=TARGET_LENGTH, overwrite=True)

    for problem in problematic:
        logging.warning(problem)


if __name__ == "__main__":
    main()
