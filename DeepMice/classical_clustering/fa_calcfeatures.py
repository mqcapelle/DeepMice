from scipy.signal import find_peaks
from statistics import stdev, mean
import numpy as np
import warnings

import ephys_features as ft  # Import allensdk's ephys_feature
from fa_misc import smooth_func


def propagation_speed(first_spike_ts, positions):
    """
    Function to calculate the one-dimensional propagation speed along a
    neuron, by averaging the local speeds between measurement
    positions P.
        speed = mean(dx/dt)
    Cases where dt is nan or dt is zero are removed.

    Parameters
    ----------
    first_spike_ts : list or array of size (N,)
        Time stamps of (first) action potential peak, measured for every
        position P in N
    positions : list or array of size (N,)
        Distances between monitors, measured along the neuron (NOT 'as the
        crow flies' but 'as the road winds')

    Returns
    -------
    speed : float
    """
    if np.size(first_spike_ts) != np.size(positions):
        raise Warning("Unequal size of spikes ({}) and positions ({})".format(
            np.size(first_spike_ts), np.size(positions)))
    dt = np.diff(first_spike_ts)
    dx = np.diff(positions)
    # Remove all cases where dt has nan
    dx = dx[~np.isnan(dt)]
    dt = dt[~np.isnan(dt)]
    # Keep only cases where dt is not zero
    dx = dx[np.nonzero(dt)]
    dt = dt[np.nonzero(dt)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed = np.nanmean(np.divide(dx, dt))
    return speed


def attenuation(first_spike_vs, positions):
    """
        Function to calculate the voltage attenuation along a neuron,
        by averaging the local attenuations between measurement positions P.
            attenuation = mean(dV/dx)
        Cases where dx is nan or dx is zero are removed.

        Parameters
        ----------
        first_spike_vs : list or array of size (N,)
            Amplitude of (first) action potential peak, measured for every
            position P in N
        positions : list or array of size (N,)
            Distances between monitors, measured along the neuron (NOT 'as
            the crow flies' but 'as the road winds')

        Returns
        -------
        attenuation : float
        """
    if np.size(first_spike_vs) != np.size(positions):
        raise Warning("Unequal size of spikes ({}) and positions ({})".format(
            np.size(first_spike_vs), np.size(positions)))
    dv = np.diff(first_spike_vs)
    dx = np.diff(positions)
    # Remove all cases where dx has nan
    dv = dv[~np.isnan(dx)]
    dx = dx[~np.isnan(dx)]
    # Keep only cases where dx is not zero
    dv = dv[np.nonzero(dx)]
    dx = dx[np.nonzero(dx)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        att = np.nanmean(dv / dx)
    return att


def _calc_local_feature(f_type, v, t, peaks, start=0, end=None, *args,
                        **kwargs):
    """
    Function to calculate local feature value, by calling corresponding
    function in AllenSDK.
    Currently implemented local feature types are:
    Spike-time features:
        avg_rate, latency, first_ISI, CV_ISI, avg_ISI, adapt
    Wave-form features:
        mean AP_peak, mean AP_FWHM, mean AP_trough_depth, first AP_peak,
        first AP_FWHM, first AP_trough_depth

    Parameters
    ----------
    f_type : str
        Feature type
    v : array of shape (N,), in mV
        Membrane potential trace
    t : array of shape (N,), in seconds
        Timestamps corresponding to membrane potential trace
    peaks : array with int(s)
        Timestamps of action potential detection
    start : float (optional), in seconds
        Time of trace from which to start feature calculation
    end : float (optional), in seconds
        Time of trace until which the feature is calculated

    Returns
    -------
    value : float
        Calculated feature value. If f_type not found set or number of peaks is
        to low, to NaN
    unit : str
        Unit corresponding to feature value. If f_type not found or no feature
        has no unit, set to emtpy string.
    """
    unit = ""
    value = np.nan
    # # Spike-time features
    if str(f_type) == "avg_rate":
        # firing frequency
        unit = "Hz"
        if peaks.size != 0:
            value = ft.average_rate(t, peaks, start=start, end=end)
    elif str(f_type) == "latency":
        # latency to first action potential
        unit = "ms"
        if peaks.size != 0:
            value = ft.latency(t, peaks, start=start) * 1E3
    elif str(f_type) == "first_ISI":
        # duration of first ISI
        unit = "ms"
        if peaks.size > 1:
            isis = ft.get_isis(t, peaks) * 1E3
            value = isis[0]
    elif str(f_type) == 'CV_ISI':
        # coefficient of variation of the ISI
        unit = ""
        if peaks.size > 1:
            isis = ft.get_isis(t, peaks)
            isis_SD = stdev(isis)
            isis_mean = mean(isis)
            value = isis_SD / isis_mean
    elif str(f_type) == "avg_ISI":
        # average ISI
        unit = "ms"
        if peaks.size > 1:
            isis = ft.get_isis(t, peaks) * 1E3
            value = mean(isis)
    elif str(f_type) == "adapt":
        # adaptation index
        unit = ""
        if peaks.size > 1:
            isis = ft.get_isis(t, peaks)
            value = ft.norm_diff(isis)

    # # Wave-form based features
    elif str(f_type) == "first_AP_peak":
        # Action potential maximum peak
        unit = "mV"
        if peaks.size != 0:
            value = np.max(v[peaks])
    elif str(f_type) == "mean_AP_peak":
        # Action potential average peak
        unit = 'mV'
        if peaks.size != 0:
            value = np.nanmean(v[peaks])
    elif str(f_type) == "first_AP_FWHM":
        # First action potential width at half-height
        unit = "ms"
        if peaks.size != 0:
            spike_indexes = ft.detect_putative_spikes(v, t, start, end)
            peak_indexes = ft.find_peak_indexes(v, t, spike_indexes, end)
            spike_indexes, peak_indexes = ft.filter_putative_spikes(v, t,
                                                    spike_indexes, peak_indexes)
            trough_indexes = ft.find_trough_indexes(v, t, spike_indexes,
                                                    peak_indexes, end=end)
            widths = ft.find_widths(v, t, spike_indexes, peak_indexes,
                                                    trough_indexes)
            if widths.size == 0:
                value = np.nan
            else:
                value = widths[0] * 1E3
    elif str(f_type) == "mean_AP_FWHM":
        # Mean action potential width at half-height
        unit = "ms"
        if peaks.size != 0:
            spike_indexes = ft.detect_putative_spikes(v, t, start, end)
            peak_indexes = ft.find_peak_indexes(v, t, spike_indexes, end)
            spike_indexes, peak_indexes = ft.filter_putative_spikes(v, t,
                                                    spike_indexes, peak_indexes)
            trough_indexes = ft.find_trough_indexes(v, t, spike_indexes,
                                                    peak_indexes, end=end)
            widths = ft.find_widths(v, t, spike_indexes, peak_indexes,
                                    trough_indexes)
            if widths.size == 0:
                value = np.nan
            else:
                # try:
                #    value = np.nanmean(widths) * 1E3
                # except RuntimeWarning:
                #   value = np.nan
                value = np.nanmean(widths) * 1E3
    elif str(f_type) == "first_AP_trough_depth":
        # First AP trough depth
        unit = "mV"
        if peaks.size != 0:
            spike_indexes = ft.detect_putative_spikes(v, t, start, end)
            peak_indexes = ft.find_peak_indexes(v, t, spike_indexes, end)
            spike_indexes, peak_indexes = ft.filter_putative_spikes(v, t,
                                                spike_indexes, peak_indexes)
            trough_indexes = ft.find_trough_indexes(v, t, spike_indexes,
                                                    peak_indexes, end=end)
            if trough_indexes.size == 0:
                value = np.nan
            else:
                value = v[int(trough_indexes[0])]
    elif str(f_type) == "mean_AP_trough_depth":
        # Mean AP trough depth
        unit = "mV"
        if peaks.size != 0:
            spike_indexes = ft.detect_putative_spikes(v, t, start, end)
            peak_indexes = ft.find_peak_indexes(v, t, spike_indexes, end)
            spike_indexes, peak_indexes = ft.filter_putative_spikes(v, t,
                                                spike_indexes, peak_indexes)
            trough_indexes = ft.find_trough_indexes(v, t, spike_indexes,
                                                    peak_indexes, end=end)
            if trough_indexes.size == 0:
                value = np.nan
            else:
                trough_indexes = list(map(int, trough_indexes))
                value = np.nanmean(v[trough_indexes])
    # # Catch undefined features
    else:
        raise Warning("_calc_local_feature(): Feature name"
                      "'{}' not defined".format(f_type))

    return value, unit


def _calc_spatial_feature(f_type, first_peak_vs, first_peak_ts, positions):
    """
    Function to calculate spatial feature value.
    Currenty implemented spatial feature types are:
    speed, attenuation

    Parameters
    ----------
    f_type : str
        Feature type
    first_peak_vs : list or array of size (N,)
            Amplitude of (first) action potential peak, measured for every
            position P in N
    first_peak_ts : list or array of size (N,)
        Time stamps of (first) action potential peak, measured for every
        position P in N
    positions : list or array of size (N,)
        Distances between monitors, measured along the neuron (NOT 'as the
        crow flies' but 'as the road winds')

    Returns
    -------
    value : float
        Calculated feature value. If f_type not found set or number of peaks is
        to low, to NaN
    unit : str
        Unit corresponding to feature value. If f_type not found or no feature
        has no unit, set to emtpy string.
    """
    unit = []
    value = np.nan
    if str(f_type) == "speed":
        unit = "m/s"
        value = propagation_speed(first_peak_ts, positions) * 1E-6
    elif str(f_type) == "attenuation":
        value = attenuation(first_peak_vs, positions)
        unit = "mV/um"
    else:
        raise Warning("Feature name '{}' not defined".format(f_type))

    return value, unit


def calc_features(v, t, f_parameter, start=0, positions=None,
                  compartments=None, min_peak_height=-60, smooth=False):
    """
    Function to calculate feature values of membrane potential(s) by calling
    _calc_local_feature() and
    _calc_spatial_feature(). Features that cannot be calculated, are set to NaN
    Currently implemented local feature types are:
    Spike-time features:
        avg_rate, latency, first_ISI, CV_ISI, avg_ISI, adapt
    Wave-form features:
        AP_peak, AP_FWHM, AP_trough_depth
    Spatial features:
        speed, attenuation

    Parameters
    ----------
    v : array of shape (T,N), in mV
        Membrane potential trace
    t : array of hape (T,), in seconds
        Timestamps corresponding to membrane potential trace
    f_parameter : dict
        Dictionary describing which features and corresponding compartments to
        be calculated
    start : float (optional), in seconds
        Time of trace from which to start feature calculation
    positions : list or array of size (N,) (optional, only required for spatial
    features)
        Distances between monitors, measured along the neuron (NOT 'as the crow
        flies' but 'as the road winds')
    compartments : list or array of size (N,) (optional)
        Can be used to constrain the selection of used compartments
    min_peak_height : float, in mV
        Membrane potential threshold value to be eligible for peak detection
    smooth : Bool
        Whether to smooth the data

    Returns
    -------
    f : list of float(s)
        Calculated feature value(s), in order as described by f_names
    f_names : list of str(s)
        Names describing calculated features
    f_compartments : list of int(s)
        Corresponding compartments, in order as described by f_names
    f_units : list of str(s)
        Corresponding units, in order as described by f_names
    """
    # Distinguish between local and spatial features
    local_features = ["avg_rate", "latency", "first_ISI", "CV_ISI",
                      "avg_ISI", "adapt", "first_AP_peak", "first_AP_FWHM",
                      "first_AP_trough_depth", "mean_AP_peak",
                      "mean_AP_FWHM", "mean_AP_trough_depth"]
    spatial_features = None  # ["speed", "attenuation"]  Not required for DeepMice project
    # Create compartments iterable
    if compartments is None:
        compartments = np.arange(np.size(v, 0))
    # Create empty list for appending
    f = []
    f_names = []
    f_compartments = []
    f_units = []
    first_peak_ts = []
    first_peak_vs = []
    # Loop over compartments
    for comp_ind, compartment in enumerate(compartments):
        if smooth and compartment == 0:
            peaks, _ = find_peaks(v[comp_ind], height=-20)
        elif smooth and compartment > 0:
            v_smooth = smooth_func(v[comp_ind], 5)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(v[comp_ind])
            plt.plot(v_smooth)
            peaks, _ = find_peaks(v_smooth, height=min_peak_height)
            print(peaks)
        else:
            peaks, _ = find_peaks(v[comp_ind], height=min_peak_height)
        if peaks.size > 0:
            first_peak = peaks[0]
            first_peak_ts.append(t[first_peak])
            first_peak_vs.append(v[comp_ind][first_peak])
        else:  # No peaks detected: write NaN values
            first_peak_ts.append(np.nan)
            first_peak_vs.append(np.nan)
        # Loop over local features of interest
        for f_key, f_value in f_parameter.items():
            feature_type = f_value["type"]
            if feature_type not in local_features and feature_type not in \
                    spatial_features:
                raise UserWarning("calc_features(): Feature type"
                                  "{} is not defined.".format(feature_type))
            if feature_type in local_features:
                if f_value["compartment"] == compartment:
                    # Feature of interest corresponds to compartment
                    feature_value, feature_unit = _calc_local_feature(
                        feature_type, v[comp_ind], t, peaks, start, end=None)
                    f.append(feature_value)
                    f_names.append(f_key)
                    f_compartments.append(compartment)
                    f_units.append(feature_unit)
    # Loop over spatial features of interest
    for f_key, f_value in f_parameter.items():
        feature_type = f_value["type"]
        if feature_type in spatial_features:
            feature_value, feature_unit = _calc_spatial_feature(
                feature_type, first_peak_vs, first_peak_ts, positions)
            f.append(feature_value)
            f_names.append(f_key)
            f_compartments.append(0)
            f_units.append(feature_unit)

    return f, f_names, f_compartments, f_units


