from scipy.signal import lfilter, butter, find_peaks
import numpy as np
import typing as tp
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = (Path(__file__).parent / 'data').resolve()

def get_contrast(t: np.ndarray, stim: np.ndarray, contrast_eps: float = 1., axis:int = -1) -> np.ndarray:
    """Compute Concentration Contrast from Stimulus
    
    Arguments:
        t: time vector
        stim: stimulus
        contrast_eps: epsilon added to denominator to avoid devision by 0
        axis: axis along which the contarst is computed
    
    Returns:
        Concentration contrast
    """
    dt = t[1] - t[0]
    b, a = butter(5, 15, fs=1 / dt)
    stim_smooth = lfilter(b, a, np.clip(stim, 0, np.inf), axis=axis)
    d_stim = np.diff(stim_smooth, axis=axis, prepend=stim_smooth[:, [0]]) / dt
    contrast_stim = d_stim / (contrast_eps + stim_smooth)
    return contrast_stim

def decompose_signal(
    t: np.ndarray, stim: np.ndarray, out: np.ndarray, 
    axis: int = -1, 
    contrast_eps:float=1.0, ss_window:float=0.5
) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose output signal given stimulus
    
    Arguments:
        t: time vector
        stim: stimulus, from which contrast is computed
        out: output signal
        axis: axis along which the decomposition is performed
        contrast_eps: epsilon value for computing contrast
        ss_window: window for computing average response before jump times, in unit of seconds
        
    Returns:
        pos_peaks_contrast: position (index) of peak timing computed from contrast
        ss_vals: amplitudes of steady-state values
        ss: steady-state signal component
        pk: transient signal component
    """
    dt = t[1] - t[0]
    stim = np.atleast_2d(stim)
    out = np.atleast_2d(out)
    # compute contrast
    contrast_stim = get_contrast(t, stim, axis=axis)
    # find timing of peak contrasts (both positive and negative)
    pos_peaks_contrast = np.sort(
        np.concatenate(
            [
                find_peaks(
                    contrast_stim[0], height=2.5, width=100, distance=int(1.7 // dt)
                )[0],
                find_peaks(
                    -contrast_stim[0], height=2.5, width=100, distance=int(1.7 // dt)
                )[0],
            ]
        )
    )
    pos_peaks_contrast = np.concatenate([pos_peaks_contrast, [len(t)]])
    
    # find average response ss_window seconds before each contrast peak
    ss = np.zeros_like(out)
    ss_vals = np.zeros((out.shape[0], len(pos_peaks_contrast) - 1))
    for n, (start_idx, stop_idx) in enumerate(
        zip(pos_peaks_contrast[:-1], pos_peaks_contrast[1:])
    ):
        ss_amp = out[
            :,
            stop_idx - int((0.2 + ss_window) // (dt)) : 
            stop_idx - int(0.2 // (dt)),
        ].mean(axis=1)
        ss_vals[:, n] = ss_amp
        ss[:, start_idx:stop_idx] = ss_amp[:, None]
    
    # compute peak as residual
    pk = out - ss
    return pos_peaks_contrast, ss_vals, ss, pk

def yyaxis(ax: plt.Axes, c: "color" = "red") -> plt.Axes:
    """Create A second axis with colored spine/ticks/label

    Note:
        This method will only make the twinx look like the color in
        MATLAB's :code:`yyaxis` function. However, unlike in MATLAB,
        it will not set the linestyle and linecolor of the lines that
        are plotted after twinx creation.

    Arguments:
        ax: the main axis to generate a twinx from
        c: color of the twinx, see https://matplotlib.org/stable/gallery/color/color_demo.html
            for color specifications accepted by matplotlib.
    """
    ax2 = ax.twinx()
    ax2.spines["right"].set_color(c)
    ax2.tick_params(axis="y", colors=c, which="both")
    ax2.yaxis.label.set_color(c)
    return ax2


def corrcoef(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return Correlation Coefficient of two matricces"""
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    x_center = x - mu_x[:, None]
    y_center = y - mu_y[:, None]
    num = (x_center * y_center).sum(1)
    den = np.linalg.norm(x_center, axis=1) * np.linalg.norm(y_center, axis=1)
    return num / (1e-15 + den)


def angular_dist(X: np.ndarray, Y: np.ndarray=None, paired: bool=False) -> np.ndarray:
    """Compute angular distances
    
    Arguments:
        X: 2D data of shape (N_samples_X, N_features)
        Y: 2D data of shape (N_samples_Y, N_features). If not specified, same as X
        paired: either paired distance between X,Y or pairwise.
          - if `paired=True`, `X, Y` must have the same dimensionality
        
    Returns:
        Angular distance of shape (N_samples_X, N_samples_Y) if `paired` is False,
        else (N_samples_X,).
        Entries where both X,Y are all 0 are set to `np.inf`.
    """
    if Y is None:
        Y = X
    if paired is True:
        from sklearn.metrics.pairwise import paired_distances as metric_func
        positive_mask = np.logical_and(np.all(X >= 0, axis=1), np.all(Y >= 0, axis=1))
        zero_mask = np.logical_or(np.all(X == 0, axis=1), np.all(Y == 0, axis=1))
    else:
        from sklearn.metrics import pairwise_distances as metric_func
        positive_mask = np.logical_and(np.all(X >= 0, axis=1), np.all(Y >= 0, axis=1))
        zero_mask = np.logical_or(np.all(X == 0, axis=1), np.all(Y == 0, axis=1))
    cos_sim = np.atleast_1d(np.squeeze(1.0 - metric_func(X=X, Y=Y, metric="cosine")))
    dists = np.arccos(cos_sim) / np.pi
    dists[positive_mask] *= 2
    dists[zero_mask] = np.inf
    return dists

def estimate_current(spike_rates: np.ndarray, resting:float=8.) -> np.ndarray:
    """Estimate synaptic current from spike rate for ConnorStevens neuron model
    
    Arguments:
        spike_rates: spike rate vector
        resting: resting spike rate of CSN model
    
    Returns:
        injected current in uA
    """
    import pandas as pd

    _df = pd.read_json(DATA_DIR / "records_fi.json").T
    _df = _df[["Currents", "Frequencies", "Params"]]
    resting_ref = _df.Frequencies.apply(lambda r: r[0])
    idx = np.argmin(np.abs(resting_ref - resting))
    if np.abs(resting_ref[idx] - resting) > 0.5:
        raise ValueError(
            f"Closest resting rate is {resting_ref[idx]}, required as {resting}"
        )
    I = np.interp(spike_rates, _df.iloc[idx].Frequencies, _df.iloc[idx].Currents)
    if resting == 0:
        I[I < 7.6] = 0.0  # noise-free CSN neuron has no spiking for current below 7.6 uA
    return I

def estimate_spike_rate(currents: np.ndarray, resting:float=8.) -> np.ndarray:
    """Estimate spike rate from input current to CSN neuron
    
    Arguments:
        currents: injected current vector
        resting: resting spike rate of CSN model
    
    Returns:
        spike rate in Hz
    """
    import pandas as pd

    _df = pd.read_json(DATA_DIR / "records_fi.json").T
    _df = _df[["Currents", "Frequencies", "Params"]]
    resting_ref = _df.Frequencies.apply(lambda r: r[0])
    idx = np.argmin(np.abs(resting_ref - resting))
    if np.abs(resting_ref[idx] - resting) > 0.5:
        raise ValueError(
            f"Closest resting rate is {resting_ref[idx]}, required as {resting}"
        )
    return np.interp(currents, _df.iloc[idx].Currents, _df.iloc[idx].Frequencies)