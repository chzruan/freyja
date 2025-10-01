"""
Handles physical predictions by wrapping trained emulators and implementing
the Halo Occupation Distribution (HOD) model.
"""

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jax.scipy.special import erfc
from jax import random
import numpy as np
from typing import Callable, Dict, Any

from .models import FCNStd, fcn_from_config, load_and_map_params_dynamic, inspect_ckpt_dims, _cfg_to_fcnstd, _load_yaml


def xiS2_trans_inverse(xiS2_trans: jnp.ndarray, aa: float = 0.1) -> jnp.ndarray:
    r"""
    Inverse transformation for the xi2 emulator output.
    
    transformation of quadrupole: :math:`y \equiv \xi_2(s) \to \tilde{y} = \mathrm{sign}(y) \log(1 + |y|/a)`
    
    inverse transformation: :math:`y = \mathrm{sign}(\tilde{y}) a (\mathrm{e}^{|y|} - 1)`
    """
    return jnp.sign(xiS2_trans) * aa * (jnp.exp(jnp.abs(xiS2_trans)) - 1.0)


def _load_emulator(config: Dict[str, Any], key: random.PRNGKey, n_input: int, n_output: int, hidden: tuple) -> (FCNStd, Dict):
    """Helper to initialize a Flax model and load its weights from a checkpoint."""
    model = FCNStd(n_input=n_input, n_output=n_output, n_hidden=hidden)
    variables = model.init(key, jnp.zeros((1, n_input)))
    variables = load_and_map_params_dynamic(
        ckpt_path=config['ckpt_path'],
        variables=variables,
        prefix=config.get('prefix', 'mlp.mlp'),
    )
    return model, variables

def setup_emulators(config: Dict[str, Any], N_bins: int) -> Dict[str, Any]:
    """Loads and initializes all emulators specified in the config."""
    emulators = {}
    
    # xi0 emulator
    cfg0 = _load_yaml(config['xi0']['hparams_path'])
    std0 = _cfg_to_fcnstd(cfg0)
    assert std0['n_output'] == N_bins, f"xi0 emulator output ({std0['n_output']}) != data bins ({N_bins})"
    emulators['xi0_model'], emulators['xi0_vars'] = _load_emulator(
        config['xi0'], random.PRNGKey(123), std0['n_input'], std0['n_output'], std0['n_hidden'])
    
    # xi2 emulator
    cfg2 = _load_yaml(config['xi2']['hparams_path'])
    std2 = _cfg_to_fcnstd(cfg2)
    assert std2['n_output'] == N_bins, f"xi2 emulator output ({std2['n_output']}) != data bins ({N_bins})"
    emulators['xi2_model'], emulators['xi2_vars'] = _load_emulator(
        config['xi2'], random.PRNGKey(124), std2['n_input'], std2['n_output'], std2['n_hidden'])

    # HMF emulator
    n_in, n_out, hidden = inspect_ckpt_dims(config['hmf']['ckpt_path'], config['hmf']['prefix'])
    assert n_in == 4, f"HMF emulator expects 4 cosmology inputs, got {n_in}"
    emulators['hmf_model'], emulators['hmf_vars'] = _load_emulator(
        config['hmf'], random.PRNGKey(456), n_in, n_out, hidden)
    
    # Load HMF mass bins
    log10M_edges = np.load(config['hmf']['log10M_edges_path'])
    assert n_out == len(log10M_edges), "HMF output dim must match mass bins"
    emulators['log10M_edges'] = jnp.asarray(log10M_edges)
    emulators['dlog10M'] = jnp.asarray(log10M_edges[1] - log10M_edges[0])


    # log10ng emulator
    n_in, n_out, hidden = inspect_ckpt_dims(config['log10ng']['ckpt_path'], config['log10ng']['prefix'])
    emulators['log10ng_model'], emulators['log10ng_vars'] = _load_emulator(
        config['log10ng'], random.PRNGKey(456), n_in, n_out, hidden)

    return emulators


def get_xi_predictor(emulators: Dict, s_full: jnp.ndarray, cut_indices: jnp.ndarray) -> Callable:
    """Returns a JIT-compiled function that predicts xi(theta)."""
    s2 = s_full ** 2
    
    @jax.jit
    def predict_xi(theta: jnp.ndarray) -> jnp.ndarray:
        y1_full = emulators['xi0_model'].apply(emulators['xi0_vars'], theta[None, :])[0] / s2
        y2_full = xiS2_trans_inverse(emulators['xi2_model'].apply(emulators['xi2_vars'], theta[None, :])[0])
        y_all = jnp.concatenate([y1_full, y2_full])
        return y_all[cut_indices]
        
    return predict_xi

@jax.jit
def lambda_sat_smooth(halo_mass, galaxy_params, T=1e1):
    """Differentiable satellite occupation number."""
    logMcut, logM1, logsigma, alpha = galaxy_params
    kappa = 1.0  # M_sat_min = kappa * M_cen_cut
    Mmin = 10.0**logMcut
    M1 = 10.0**logM1
    
    gate = jax.nn.sigmoid((halo_mass - kappa * Mmin) / T)
    base = ((halo_mass - kappa * Mmin).clip(min=0.0) / M1)**alpha
    return gate * base

def get_density_predictor(emulators: Dict, use_log_ng: bool) -> Callable:
    """Returns a JIT-compiled function that predicts n_g(theta) or log10n_g(theta)."""
    
    @jax.jit
    def predict_density(theta: jnp.ndarray) -> jnp.ndarray:
        _log10density_emu = emulators['log10ng_model'].apply(emulators['log10ng_vars'], theta[None, :-2])[0]
        return _log10density_emu if use_log_ng else 10**_log10density_emu

    return predict_density
