import numpy as np
import sys
import h5py
import argparse
from pathlib import Path
from scipy.signal import savgol_filter
import mcfit
from scipy.interpolate import InterpolatedUnivariateSpline


def pad_and_damp_Pk(k_in, P_in, pad_factor=100):
    """
    Extends P(k) to very low and high k to suppress FFT ringing.

    Args:
        k_in (array): Original k array.
        P_in (array): Original P(k).
        pad_factor (float): How much to extend the range (e.g., 100x).

    Returns:
        k_padded, P_padded
    """
    # 1. Determine Power Law Slopes at edges
    # Low-k slope (n_low)
    n_low = np.log(P_in[1] / P_in[0]) / np.log(k_in[1] / k_in[0])
    # High-k slope (n_high)
    n_high = np.log(P_in[-1] / P_in[-2]) / np.log(k_in[-1] / k_in[-2])

    # 2. Create Padded k array
    k_min_new = k_in[0] / pad_factor
    k_max_new = k_in[-1] * pad_factor

    # Generate extensions (maintaining log spacing)
    # Note: We use the same dl (delta log k) as the input array
    dln_k = np.log(k_in[1] / k_in[0])

    k_left = np.exp(np.arange(np.log(k_min_new), np.log(k_in[0]), dln_k))
    k_right = np.exp(np.arange(np.log(k_in[-1]) + dln_k, np.log(k_max_new), dln_k))

    # 3. Extrapolate P(k)
    P_left = P_in[0] * (k_left / k_in[0]) ** n_low
    P_right = P_in[-1] * (k_right / k_in[-1]) ** n_high

    # 4. Apply High-k Damping (Window Function)
    # This is critical! Softens the upper cutoff.
    # We apply a Gaussian cutoff to the extended part only.
    k_damp = k_in[-1] * 10  # Dampen well outside the data range
    damping = np.exp(-((k_right / k_damp) ** 2))
    P_right *= damping

    # 5. Concatenate
    k_final = np.concatenate([k_left, k_in, k_right])
    P_final = np.concatenate([P_left, P_in, P_right])

    return k_final, P_final


def compute_xi_from_Pk(
    k_input, P_input, r_output, r_min=0.1, r_max=150.0, smooth_xi=True
):
    r"""
    Compute xi(r) from P(k) using mcfit based on Hankel transform.
    Args:
        k_input (array): Input k values.
        P_input (array): Input P(k) values.
        r_output (array): Desired output r values.
        r_min (float): Minimum r to consider.
        r_max (float): Maximum r to consider.
        smooth_xi (bool): Whether to apply smoothing to the result.
    Returns:
        xi(r_output) (array): Correlation function at r_output.
    """

    k_interp = np.logspace(np.log10(k_input[0]), np.log10(k_input[-1]), 2000)
    P_interp = np.interp(k_interp, k_input, P_input)
    k_pad, P_pad = pad_and_damp_Pk(k_interp, P_interp, pad_factor=100)
    r, xi = mcfit.P2xi(k_pad)(P_pad)
    mask_r = (r <= r_max) & (r >= r_min)
    r = r[mask_r]
    xi = xi[mask_r]
    spline_r2xi = InterpolatedUnivariateSpline(r, r**2 * xi, k=3)

    if smooth_xi:
        # Apply Savitzky-Golay filter for smoothing
        r2xi_smooth = savgol_filter(
            spline_r2xi(r_output), window_length=11, polyorder=3
        )
        return r2xi_smooth / r_output**2
    return spline_r2xi(r_output) / r_output**2
