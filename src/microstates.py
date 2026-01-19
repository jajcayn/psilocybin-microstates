"""
Functions to segment EEG into microstates. Based on the Microsegment toolbox
for EEGlab, written by Andreas Trier Poulsen [1]_.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>

References
----------
.. [1]  Poulsen, A. T., Pedroni, A., Langer, N., &  Hansen, L. K. (2018).
        Microstate EEGlab toolbox: An introductionary guide. bioRxiv.


Taken from https://github.com/wmvanvliet/mne_microstates

Little changes / additions by me.
"""

import logging
import os
import string
from itertools import permutations

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.signal import convolve, find_peaks, get_window
from scipy.stats import zscore

from src.helpers import DATA_ROOT

DEFAULT_TEMPLATES = os.path.join(DATA_ROOT, "MS-templates_Koenig")
MNE_LOGGING_LEVEL = "WARNING"
mne.set_log_level(MNE_LOGGING_LEVEL)


def global_map_dissimilarity(map1, map2):
    """
    Computes a global map dissimilarity between two maps.

    https://github.com/emma-holmes/Source-Dissimilarity-Index-for-Python/blob/master/SourceDissimilarityIndex.py

    :param map1: first map to compare
    :type map1: first map to compare
    :param map2: second map to compare
    :type map2: second map to compare
    :return: global map dissimilarity
    :rtype: float
    """
    assert map1.shape == map2.shape

    def normalize(data):
        data_norm = data.reshape(1, np.size(data), order="F").copy()
        data_norm = data_norm - np.mean(data_norm)
        if np.mean(data_norm**2) != 0:
            data_norm = data_norm / np.sqrt(np.mean(data_norm**2))

        return data_norm

    map1_norm = normalize(map1)
    map2_norm = normalize(map2)
    gmd = np.sqrt(np.mean((map1_norm - map2_norm) ** 2))

    return gmd


def get_gfp_peaks(data, min_peak_dist=2, smoothing=None, smoothing_window=100):
    """
    Compute GFP peaks.

    :param data: data for GFP peaks, channels x samples
    :type data: np.ndarray
    :param min_peak_dist: minimum distance between two peaks
    :type min_peak_dist: int
    :param smoothing: smoothing window if some, None means to smoothing
    :type smoothing: str|None
    :param smoothing_window: window for smoothing, in samples
    :type smoothing_window: int
    :return: GFP peaks and GFP curve
    :rtype: (list, np.ndarray)
    """
    gfp_curve = np.std(data, axis=0)
    if smoothing is not None:
        gfp_curve = convolve(
            gfp_curve,
            get_window(smoothing, Nx=smoothing_window),
        )
    gfp_peaks, _ = find_peaks(gfp_curve, distance=min_peak_dist)

    return gfp_peaks, gfp_curve


def segment(
    data,
    n_states=4,
    use_gfp=True,
    n_inits=10,
    max_iter=1000,
    thresh=1e-6,
    normalize=False,
    return_polarity=False,
    random_state=None,
    **kwargs,
):
    """
    Segment a continuous signal into microstates.

    Peaks in the global field power (GFP) are used to find microstates, using a
    modified K-means algorithm. Several runs of the modified K-means algorithm
    are performed, using different random initializations. The run that
    resulted in the best segmentation, as measured by global explained variance
    (GEV), is used.

    Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
    Additions: Nikola Jajcay
    Code: https://github.com/wmvanvliet/mne_microstates

    :param data: data to find the microstates in, channels x samples
    :type data: np.ndarray
    :param n_states: number of states to find
    :type n_states: int
    :param use_gfp: whether to use GFP peaks to find microstates or whole data
    :type use_gfp: bool
    :param n_inits: number of random initialisations to use for algorithm
    :type n_inits: int
    :param max_iter: the maximum number of iterations to perform in the
        microstate algorithm
    :type max_iter: int
    :param thresh: threshold for convergence of the microstate algorithm
    :type thresh: float
    :param normalize: whether to z-score the data
    :type normalize: bool
    :param return_polarity: whether to return the polarity of the activation
    :type return_polarity: bool
    :param random_state: seed or RandomState` for the random number generator
    :type random_state: int|np.random.RandomState|None
    :return: microstate maps, dummy segmentation (maximum activation), polarity
        (if `return_polarity` == True), global explained variance for whole
        timeseries, global explained variance for GFP peaks
    :rtype: (np.ndarray, np.ndarray, np.ndarray, float, float)

    References:
    Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995). Segmentation of
        brain electrical activity into microstates: model estimation and
        validation. IEEE Transactions on Biomedical Engineering, 42(7), 658-665.
    """
    logging.debug(
        "Finding %d microstates, using %d random intitializations"
        % (n_states, n_inits)
    )

    if normalize:
        data = zscore(data, axis=1)

    if use_gfp:
        (peaks, gfp_curve) = get_gfp_peaks(data, **kwargs)
    else:
        peaks = np.arange(data.shape[1])
        gfp_curve = np.std(data, axis=0)

    # Cache this value for later
    gfp_sum_sq = np.sum(gfp_curve**2)
    peaks_sum_sq = np.sum(data[:, peaks].std(axis=0) ** 2)

    # Do several runs of the k-means algorithm, keep track of the best
    # segmentation.
    best_gev = 0
    best_gfp_gev = 0
    best_maps = None
    best_segmentation = None
    best_polarity = None
    for _ in range(n_inits):
        maps = _mod_kmeans(
            data[:, peaks],
            n_states,
            max_iter,
            thresh,
            random_state,
        )
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        map_corr = _corr_vectors(data, maps[segmentation].T)
        gfp_corr = _corr_vectors(data[:, peaks], maps[segmentation[peaks]].T)
        # assigned_activations = np.choose(segmentations, activation)

        # Compare across iterations using global explained variance (GEV) of
        # the found microstates.
        gev = sum((gfp_curve * map_corr) ** 2) / gfp_sum_sq
        gev_gfp = (
            sum((data[:, peaks].std(axis=0) * gfp_corr) ** 2) / peaks_sum_sq
        )
        logging.debug("GEV of found microstates: %f" % gev)
        if gev > best_gev:
            best_gev, best_maps, best_segmentation = gev, maps, segmentation
            best_gfp_gev = gev_gfp
            best_polarity = np.sign(np.choose(segmentation, activation))

    if return_polarity:
        return (
            best_maps,
            best_segmentation,
            best_polarity,
            best_gev,
            best_gfp_gev,
        )
    else:
        return best_maps, best_segmentation, best_gev, best_gfp_gev


def _mod_kmeans(
    data,
    n_states=4,
    max_iter=1000,
    thresh=1e-6,
    random_state=None,
):
    """
    The modified K-means clustering algorithm.

    See :func:`segment` for the meaning of the parameters and return
    values.

    Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
    Code: https://github.com/wmvanvliet/mne_microstates
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    n_channels, n_samples = data.shape

    # Cache this value for later
    data_sum_sq = np.sum(data**2)

    # Select random timepoints for our initial topographic maps
    init_times = random_state.choice(n_samples, size=n_states, replace=False)
    maps = data[:, init_times].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)  # Normalize the maps

    prev_residual = np.inf
    for iteration in range(max_iter):
        # Assign each sample to the best matching microstate
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        # Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in range(n_states):
            idx = segmentation == state
            if np.sum(idx) == 0:
                logging.warning("Some microstates are never activated")
                maps[state] = 0
                continue

            # Find largest eigenvector
            # cov = data[:, idx].dot(data[:, idx].T)
            # _, vec = eigh(cov, eigvals=(n_channels - 1, n_channels - 1))
            # maps[state] = vec.ravel()
            maps[state] = data[:, idx].dot(activation[state, idx])
            maps[state] /= np.linalg.norm(maps[state])

        # Estimate residual noise
        act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
        residual = abs(data_sum_sq - act_sum_sq)
        residual /= float(n_samples * (n_channels - 1))

        # Have we converged?
        if (prev_residual - residual) < (thresh * residual):
            logging.debug("Converged at %d iterations." % iteration)
            break

        prev_residual = residual
    else:
        logging.warning("Modified K-means algorithm failed to converge.")

    return maps


def _corr_vectors(A, B, axis=0):
    """
    Compute pairwise correlation of multiple pairs of vectors.

    Fast way to compute correlation of multiple pairs of vectors without
    computing all pairs as would with corr(A,B). Borrowed from Oli at Stack
    overflow. Note the resulting coefficients vary slightly from the ones
    obtained from corr due differences in the order of the calculations.
    (Differences are of a magnitude of 1e-9 to 1e-17 depending of the tested
    data).

    Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
    Additions: Nikola Jajcay
    Code: https://github.com/wmvanvliet/mne_microstates

    :param A: first collection of vectors
    :type A: np.ndarray
    :param B: second collection of vectors
    :type B: np.ndarray
    :param axis: axis along which to perform correlations
    :type axis: int
    :return: correlation between pairs of vector
    :rtype: np.ndarray
    """
    An = A - np.mean(A, axis=axis, keepdims=True)
    Bn = B - np.mean(B, axis=axis, keepdims=True)
    An /= np.linalg.norm(An, axis=axis, keepdims=True)
    Bn /= np.linalg.norm(Bn, axis=axis, keepdims=True)
    return np.sum(An * Bn, axis=axis)


def plot_microstate_maps(
    microstates,
    mne_info,
    xlabels=None,
    title="",
    plot_minmax_vec=True,
    fname=None,
    **kwargs,
):
    """
    Plots microstate maps.

    :param microstates: microstate topographies to plot, no states x channels
    :type microstates: np.ndarray
    :param mne_info: info from mne as per channels and locations
    :type mne_info: `mne.io.meas_info.Info`
    :param xlabels: labels for microstates maps, usually correlation with
        template
    :type xlabels: list[str]
    :param title: title for the plot
    :type title: str
    :param plot_minmax_vec: whether to plot vector between minimum and maximum
        loading of the topographies
    :type plot_minmax_vec: bool
    :param fname: filename for the plot, if None, will show
    :type fname: str|None
    """

    plt.figure(figsize=((np.ceil(microstates.shape[0] / 2.0)) * 5, 12))

    ms_names = list(string.ascii_uppercase)[: microstates.shape[0]]

    if xlabels is None:
        xlabels = ["" for i in range(microstates.shape[0])]

    for i, t, xlab in zip(range(microstates.shape[0]), ms_names, xlabels):
        plt.subplot(2, int(np.ceil(microstates.shape[0] / 2.0)), i + 1)
        mne.viz.plot_topomap(
            microstates[i, :], mne_info, show=False, contours=10
        )

        if plot_minmax_vec:
            max_sen = np.argmax(microstates[i, :])
            min_sen = np.argmin(microstates[i, :])
            pos_int = mne.channels.layout._find_topomap_coords(
                mne_info, picks="eeg"
            )
            plt.gca().plot(
                [pos_int[min_sen, 0], pos_int[max_sen, 0]],
                [pos_int[min_sen, 1], pos_int[max_sen, 1]],
                "ko-",
                markersize=7,
                lw=2.2,
            )
        plt.title(t, fontsize=25)
        plt.xlabel(xlab, fontsize=22)

    plt.suptitle(title, fontsize=30)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight", dpi=150, **kwargs)
    plt.close()


def match_reorder_microstates(
    maps_input,
    maps_sortby,
    return_correlation=False,
    return_attribution_only=False,
):
    """
    Match and reorder microstates. `maps_input` will be reorderer based on
    correlations with `maps_sortby`. Disregards polarity as usual in microstates
    analyses.

    :param maps_input: maps to be reordered, no maps x channels
    :type maps_input: np.ndarray
    :param maps_sortby: reference maps for sorting, no maps x channels
    :type maps_sortby: np.ndarray
    :param return_correlation: whether to return correlations of the best
        attribution
    :type return_correlation: bool
    :param return_attribution_only: whether to return only attribution list, i.e.
        list of indices of the highest correlation, if False, will return
        reordered maps
    :type return_attribution_only: bool
    :return: best attribution or reordered maps, correlation of best attribution
        (if `return_correlation` == True)
    :rtype: np.ndarray or list[int]|n.ndarray
    """
    assert maps_input.shape == maps_sortby.shape
    n_maps = maps_input.shape[0]
    best_corr_mean = -1
    best_attribution = None
    best_corr = None
    for perm in permutations(range(n_maps)):
        corr_attr = np.abs(
            _corr_vectors(maps_sortby, maps_input[perm, :], axis=1)
        )
        if corr_attr.mean() > best_corr_mean:
            best_corr_mean = corr_attr.mean()
            best_corr = corr_attr
            best_attribution = perm
    to_return = (
        best_attribution
        if return_attribution_only
        else maps_input[best_attribution, :]
    )
    if return_correlation:
        return to_return, best_corr
    else:
        return to_return


def load_Koenig_microstate_templates(n_states=4, path=DEFAULT_TEMPLATES):
    """
    Load microstate canonical maps as per Koening et al. Neuroimage, 2015.

    :param n_states: number of canonical templates to load
    :type n_states: int
    :param path: folder with templates
    :type path: str
    :return: template maps (state x channels), channel names
    :rtype: (np.ndarray, list)
    """
    assert n_states <= 6
    template_maps = loadmat(os.path.join(path, "MS_templates_Koenig.mat"))[
        "data"
    ]
    channels = pd.read_csv(os.path.join(path, "channel_info.csv"))["labels"]
    # keep only relevant maps
    template_maps = template_maps[:, :n_states, n_states - 1]
    assert template_maps.shape == (len(channels), n_states)

    return template_maps.T, channels.values.tolist()
