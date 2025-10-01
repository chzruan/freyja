# freyja/mcmc/run.py
"""
Main script to run the MCMC analysis for cosmological and HOD parameters.
This can be run as a command-line tool via `freyja-run`.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import argparse
from pathlib import Path
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

# Use relative imports for package modules
from . import data_loader
from . import theory
from . import plot_posterior

def run_analysis(config_path: str, progress_bar: bool = True):
    """Orchestrates the MCMC analysis pipeline from a config file."""
    
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cfg_basic       = config['basic']
    gravity         = cfg_basic['gravity']
    cfg_mcmc        = config['mcmc']
    cfg_analysis    = config['analysis']

    cfg_data        = config[gravity]['data_paths']
    cfg_params      = config[gravity]['parameters']

    param_names = list(cfg_params.keys())
    low_bounds  = jnp.array([v[0] for v in cfg_params.values()])
    high_bounds = jnp.array([v[1] for v in cfg_params.values()])


    # 2. Load and Prepare Data
    s_full, data_full = data_loader.load_correlation_functions(cfg_data['xi_data'])
    cov_full = data_loader.load_covariance(cfg_data['covariance'])
    N_bins = len(s_full)
    
    ikeep = np.where(s_full > cfg_analysis['smin'])[0]
    if ikeep.size == 0:
        raise ValueError(f"No bins satisfy s > {cfg_analysis['smin']}.")
    icut = np.concatenate([ikeep, ikeep + N_bins])
    data_vec = jnp.asarray(data_full[icut])
    cov_mat = jnp.asarray(cov_full[np.ix_(icut, icut)])
    
    print(f"Scale cut s > {cfg_analysis['smin']:.2f} Mpc/h: Kept {ikeep.size}/{N_bins} bins.")
    
    L = jnp.linalg.cholesky(cov_mat)
    prec_mat = jsp.linalg.solve_triangular(L, jnp.eye(L.shape[0]), lower=True).T @ \
               jsp.linalg.solve_triangular(L, jnp.eye(L.shape[0]), lower=True)
    
    n_data = data_loader.load_galaxy_density(cfg_data['galaxy_density'], cfg_analysis['use_log_ng'])
    sigma_n = np.abs(cfg_analysis['density_relerr'] * n_data)
    if cfg_analysis['use_log_ng']:
        print(f"Using n_data = {n_data:.4f} with sigma_n = {sigma_n:.4f}")
    else:
        print(f"Using n_data = {n_data:.4e} with sigma_n = {sigma_n:.4e}")

    # 3. Setup Theory Predictors
    emulators = theory.setup_emulators(config[gravity]['emulators'], N_bins)
    predict_xi = theory.get_xi_predictor(emulators, jnp.asarray(s_full), jnp.asarray(icut))
    predict_density = theory.get_density_predictor(emulators, cfg_analysis['use_log_ng'])

    # 4. Define NumPyro Model
    def numpyro_model():
        theta = numpyro.sample("theta", dist.Independent(dist.Uniform(low_bounds, high_bounds), 1))

        # RSD xi0 + xi2 likelihood (multivariate normal)
        mu_xi = predict_xi(theta)
        numpyro.sample("xi_obs", dist.MultivariateNormal(loc=mu_xi, precision_matrix=prec_mat), obs=data_vec)
        
        # Galaxy number density likelihood (normal)
        mu_n = predict_density(theta)
        numpyro.sample("n_obs", dist.Normal(mu_n, sigma_n), obs=n_data)

    # 5. Run MCMC
    kernel = NUTS(numpyro_model, target_accept_prob=cfg_mcmc['target_accept_prob'])
    mcmc = MCMC(
        kernel, 
        num_warmup=cfg_mcmc['warmup'],
        num_samples=cfg_mcmc['samples'],
        num_chains=cfg_mcmc['chains'],
        chain_method="parallel" if jax.device_count() > 1 else "sequential",
        progress_bar=progress_bar,
    )
    mcmc.run(random.PRNGKey(cfg_mcmc.get('prng_key', 0)))
    mcmc.print_summary()

    # 6. Save Results
    posterior_samples = mcmc.get_samples()["theta"]
    
    # Dynamically generate the output filename
    samples = cfg_mcmc['samples']
    smin = cfg_analysis['smin']
    density_str = "uselogng" if cfg_analysis['use_log_ng'] else "useng"
    output_filename = f"L{samples:.0e}_smin{smin:.0f}_{density_str}.npz"
    Path(cfg_mcmc['output_path']).mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(cfg_mcmc['output_path']) / Path(output_filename),
        theta=np.array(posterior_samples),
        names=np.array(param_names),
        smin=np.array(cfg_analysis['smin']),
        ikeep=np.array(ikeep),
        config=str(config) # Save the config for provenance
    )
    print(f"\nSaved posterior samples to: {output_filename}")
    plot_posterior.create_triangle_plot(
        Path(cfg_mcmc['output_path']) / Path(output_filename), 
        cfg_mcmc['output_path'],
    )


def cli_entrypoint():
    """Handles command-line argument parsing for the main analysis."""
    parser = argparse.ArgumentParser(description="Run MCMC analysis using the Freyja package.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    run_analysis(args.config)

