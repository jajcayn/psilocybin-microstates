"""
Base subject class for microstate analyses. Assumes within-subject design with
multiple sessions / recording times.

(c) Nikola Jajcay
"""

import logging
import os
import string
from glob import glob
from itertools import groupby

import mne
import numpy as np
import pandas as pd
import scipy.stats as sts
import xarray as xr

from src.microstates import (
    get_gfp_peaks,
    global_map_dissimilarity,
    match_reorder_microstates,
    segment,
)

MNE_LOGGING_LEVEL = "WARNING"
mne.set_log_level(MNE_LOGGING_LEVEL)


class PsilocybinRecording:
    """
    Base class for each individual subject and recording.
    """

    @classmethod
    def load_processed(cls, filename):
        """
        Init class from preprocessed .fif file.

        :param filename: filename for EEG data in fif format
        :type filename: str
        """
        basename = os.path.basename(filename)
        session, subj_no, time, _, _, _ = basename.split("_")
        data = mne.io.read_raw_fif(filename, preload=True)
        return cls(subj_no, f"{session}-{time}", data)

    def __init__(self, subject_no, session, data):
        """
        :param subject_no: subject identifier
        :type subject_no: int|str
        :param session: session / time of the recording
        :type session: str
        :param data: EEG data for the recording
        :type data: `mne.io.RawArray`|`mne.io.fiff.raw.Raw`
        """
        self.subject = int(subject_no)
        self.session = session
        assert isinstance(data, (mne.io.RawArray, mne.io.fiff.raw.Raw))
        self._data = data
        self._data.pick_types(eeg=True)
        self.gfp_peaks = None
        self.gfp_curve = None
        self.microstates = None
        self.segmentation = None
        self.computed_stats = False
        self.attrs = {}

    @property
    def info(self):
        """
        Return mne info.

        :return: EEG recording info
        :rtype: dict
        """
        return self._data.info

    @property
    def data(self):
        """
        Return data as numpy array.

        :return: EEG data in mne structure, channels x time
        :rtype: np.ndarray
        """
        return self._data.get_data()

    def preprocess(self, low, high):
        """
        Preprocess data - average reference and band-pass filter.

        :param low | high: low - high frequencies for bandpass filter
        :type low | high: float
        """
        self._data.set_eeg_reference("average")
        self._data.filter(low, high)

    def gfp(self):
        """
        Compute GFP curve and peaks from eeg.
        """
        self.gfp_peaks, self.gfp_curve = get_gfp_peaks(
            self.data, min_peak_dist=2, smoothing=None, smoothing_window=100
        )

    def run_microstates(self, n_states, n_inits=200):
        """
        Run microstate segmentation. Gets canonical microstates and timeseries
        segmentation using the dummy rule - maximal activation.

        :param n_states: number of canonical microstates
        :type n_states: int
        :param n_inits: number of initialisations for the modified KMeans
            algorithm
        :type n_inits: int
        """
        (
            self.microstates,
            self.segmentation,
            self.polarity,
            self.gev_tot,
            self.gev_gfp,
        ) = segment(
            self.data,
            n_states=n_states,
            use_gfp=True,
            n_inits=n_inits,
            return_polarity=True,
        )

    def reassign_segmentation_by_midpoints(self, method="corr"):
        """
        Redo segmentation based by midpoints - the GFP peaks are labelled based
        on correspondence and neighbours are smooth between them.

        :param method: which assignment method to use
        :type method: str
        """
        assert method in ["corr", "GMD"]
        if self.gfp_peaks is None:
            self.gfp()
        segmentation = np.ones_like(self.gfp_curve, dtype=int)
        segmentation *= self.microstates.shape[0] * 2
        for peak in self.gfp_peaks:
            # list of corr. coefs between original map and 4 microstates
            if method == "corr":
                similarity = [
                    np.abs(sts.pearsonr(mic, self.data[:, peak])[0])
                    for mic in self.microstates
                ]
                # pick microstate with max corr.
                segmentation[peak] = similarity.index(max(similarity))
            elif method == "GMD":
                similarity = [
                    np.abs(global_map_dissimilarity(mic, self.data[:, peak]))
                    for mic in self.microstates
                ]
                # pick microstate with min GMD
                segmentation[peak] = similarity.index(min(similarity))

        # midpoints between microstates (temporal sense)
        peaks = self.gfp_peaks.copy()
        midpoints = [
            (peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)
        ]

        for idx in range(len(midpoints) - 1):
            # fill between two midpoints with microstate at peak
            segmentation[midpoints[idx] : midpoints[idx + 1] + 1] = (
                segmentation[peaks[idx + 1]]
            )

        # beginning and end of ts, since these were omitted in the loop
        segmentation[: midpoints[0]] = segmentation[peaks[0]]
        segmentation[midpoints[-1] :] = segmentation[peaks[-1]]
        self.segmentation = segmentation

    def match_reorder_microstates(
        self, microstate_templates, template_channels
    ):
        """
        Match and reorder microstates based on template [typically group mean
        maps or Koenig's microstates templates]. Finds maximum average
        correlation among all possible attributions

        :param microstate_templates: templates for sorting / sort by
        :type microstate_templates: np.ndarray
        :param template_channels: list of channels in the template
        :type template_channels: list
        """
        # match channels
        _, idx_input, idx_sortby = np.intersect1d(
            self.info["ch_names"], template_channels, return_indices=True
        )
        attribution, self.corrs_template = match_reorder_microstates(
            self.microstates[:, idx_input],
            microstate_templates[:, idx_sortby],
            return_correlation=True,
            return_attribution_only=True,
        )
        self.microstates = self.microstates[attribution, :]

    def _compute_lifespan(self):
        """
        Computes average lifespan of microstates in segmented time series in ms.
        """
        assert self.segmentation is not None
        consec = np.array(
            [(x, len(list(y))) for x, y in groupby(self.segmentation)]
        )[1:-1]
        self.avg_lifespan = {
            ms_no: (
                consec[consec[:, 0] == ms_no].mean(axis=0)[1]
                / self.info["sfreq"]
            )
            * 1000.0
            for ms_no in np.unique(self.segmentation)
        }

    def _compute_coverage(self):
        """
        Computes total coverage of microstates in segmented time series.
        """
        assert self.segmentation is not None
        self.coverage = {
            ms_no: count / self.segmentation.shape[0]
            for ms_no, count in zip(
                *np.unique(self.segmentation, return_counts=True)
            )
        }

    def _compute_freq_of_occurrence(self):
        """
        Computes average frequency of occurrence of microstates in segmented
        time series per second.
        """
        assert self.segmentation is not None
        freq_occurence = {}
        for ms_no in np.unique(self.segmentation):
            idx = np.where(self.segmentation[:-1] == ms_no)[0]
            count_ = np.nonzero(np.diff(self.segmentation)[idx])[0].shape[0]
            freq_occurence[ms_no] = count_ / (
                self.segmentation.shape[0] / self.info["sfreq"]
            )
        self.freq_occurence = freq_occurence

    def _compute_transition_matrix(self):
        """
        Computes transition probability matrix.
        """
        assert self.segmentation is not None
        prob_matrix = np.zeros(
            (self.microstates.shape[0], self.microstates.shape[0])
        )
        for from_, to_ in zip(self.segmentation, self.segmentation[1:]):
            prob_matrix[from_, to_] += 1
        self.transition_mat = prob_matrix / np.nansum(
            prob_matrix, axis=1, keepdims=True
        )

    def compute_segmentation_stats(self):
        """
        Compute statistics on segmented time series, i.e. coverage, frequency of
        occurence, average lifespan and transition probablity matrix.
        """
        self._compute_coverage()
        self._compute_freq_of_occurrence()
        self._compute_lifespan()
        self._compute_transition_matrix()
        self.computed_stats = True

    def get_segmentation_xarray(self, return_segmentation=False):
        """
        Return microstate topographies and optionally segmentation as
        xr.DataArray.

        :param return_segmentation: whether to return also segmentations
        :type return_segmentation: bool
        :return: microstate topographies and optionally segmentation as
            xr.DataArray
        :rtype: `xr.DataArray`
        """
        extra_coords = {"subject": self.subject, "session": self.session}
        topo = xr.DataArray(
            self.microstates,
            dims=["microstate", "channel"],
            coords={
                "microstate": list(string.ascii_uppercase)[
                    : self.microstates.shape[0]
                ],
                "channel": self.info["ch_names"],
            },
        ).assign_coords(extra_coords)
        if return_segmentation:
            segmentation = xr.DataArray(
                self.segmentation,
                dims=["time"],
                coords={"time": self.data.times},
            ).assign_coords(extra_coords)

            return topo, segmentation
        else:
            return topo

    def get_stats_pandas(self, write_attrs=False):
        """
        Return all segmentation statistics as pd.DataFrame for subsequent
        statistical analysis.

        :param write_attrs: whether to write attributes to dataframe
        :type write_attrs: bool
        :return: dataframe with segmented time series statistics per microstate
        :rtype: pd.DataFrame
        """
        assert self.computed_stats
        ms_names = list(string.ascii_uppercase)[: self.microstates.shape[0]]
        df = pd.DataFrame(
            columns=[
                "subject",
                "session",
                "microstate",
                "var_GFP",
                "var_total",
                "template_corr",
                "coverage",
                "occurrence",
                "lifespan",
            ]
            + [f"transition->{to_ms}" for to_ms in ms_names]
        )
        for ms_idx, ms_name in enumerate(ms_names):
            df.loc[ms_idx] = [
                self.subject,
                self.session,
                ms_name,
                self.gev_gfp,
                self.gev_tot,
                self.corrs_template[ms_idx],
                self.coverage[ms_idx],
                self.freq_occurence[ms_idx],
                self.avg_lifespan[ms_idx],
            ] + self.transition_mat[ms_idx, :].tolist()
        if write_attrs:
            for key, val in self.attrs.items():
                df[key] = str(val)
        return df


def load_all_data(path, exclude_subjects=None):
    """
    Load all .fif data from path and exclude individual subjects.

    :param path: path with data
    :type path: str
    :param exclude_subjects: subject no. to exclude
    :type exclude_subjects: list|None
    :return: all loaded data
    :rtype: list[`PsilocybinRecording`]
    """
    exclude_subjects = exclude_subjects or []
    data = []
    logging.info(f"Will load all data from {path}")
    for filename in sorted(glob(f"{path}/*.fif")):
        basename = os.path.basename(filename)
        _, subj_no, _, _, _, _ = basename.split("_")
        if int(subj_no) in exclude_subjects:
            logging.info(f"Skipping {filename}")
            continue
        logging.info(f"Loading {filename}")
        data.append(PsilocybinRecording.load_processed(filename))
    return data
