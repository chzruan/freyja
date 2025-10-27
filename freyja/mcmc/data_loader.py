"""Functions for loading correlation functions, covariance, and galaxy density data."""

import numpy as np
import h5py
import jax.numpy as jnp
from typing import Tuple

# def load_correlation_functions(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
#     """Loads s bins and stacked [xi0, xi2] vector from HDF5 file."""
#     with h5py.File(h5_path, "r") as f:
#         boxes = [k for k in f.keys() if k.startswith("box")]
#         if not boxes:
#             raise RuntimeError("No 'box*' groups found in xiS file.")
        
#         s = np.array(f[boxes[0]]["s_bincentre_gg"][...], dtype=np.float64)

#         xi0_list = [np.asarray(f[b]["xiS0_gg"][...], dtype=np.float64) for b in boxes]
#         xi2_list = [np.asarray(f[b]["xiS2_gg"][...], dtype=np.float64) for b in boxes]

#         xi0 = np.mean(np.stack(xi0_list, axis=0), axis=0)
#         xi2 = np.mean(np.stack(xi2_list, axis=0), axis=0)

#         if not (np.all(np.isfinite(xi0)) and np.all(np.isfinite(xi2))):
#             raise ValueError("Non-finite values found in xiS data.")
        
#         return s, np.concatenate([xi0, xi2])

def load_correlation_functions(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads s bins and stacked [xi0, xi2] vector from HDF5 file.
    The expected structure is that the file contains datasets:
    's_bincentre_gg', 'xiS0_gg', and 'xiS2_gg'.
    """
    with h5py.File(h5_path, "r") as f:
        s   = f["s_bincentre_gg"][...].astype(np.float64)
        xi0 = f["xiS0_gg"][...].astype(np.float64)
        xi2 = f["xiS2_gg"][...].astype(np.float64)

        if not (np.all(np.isfinite(xi0)) and np.all(np.isfinite(xi2))):
            raise ValueError("Non-finite values found in xiS data.")
        
        return s, np.concatenate([xi0, xi2])


def load_covariance(npy_path: str) -> np.ndarray:
    """Loads the covariance matrix from a .npy file."""
    cov = np.load(npy_path)
    if not np.all(np.isfinite(cov)):
        raise ValueError("Non-finite values found in covariance matrix.")
    return cov.astype(np.float64)

def load_galaxy_density(
    ng_path: str, 
    use_log_ng: bool = False,
):
    ng, nc, ns = np.loadtxt(
        ng_path,
        unpack=True,
        skiprows=1,
    )
    ng = jnp.array(ng)
    
    if use_log_ng:
        return jnp.log10(ng)
    else:
        return ng

