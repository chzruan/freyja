# models.py
"""
Contains the Flax FCN model definition and utilities for loading weights
from PyTorch checkpoints.
"""
import torch
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Any, Dict, Tuple, List
import yaml


class SiLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jax.nn.silu(x)

class FCNStd(nn.Module):
    """
    Generic MLP with input/output standardization.
    """
    n_input: int
    n_output: int
    n_hidden: Sequence[int]
    standarize_input: bool = True
    standarize_output: bool = True
    act: Any = SiLU

    def setup(self):
        self.x_mean = self.variable("scalers", "x_mean", lambda: jnp.zeros((self.n_input,)))
        self.x_std  = self.variable("scalers", "x_std",  lambda: jnp.ones((self.n_input,)))
        self.y_mean = self.variable("scalers", "y_mean", lambda: jnp.zeros((self.n_output,)))
        self.y_std  = self.variable("scalers", "y_std",  lambda: jnp.ones((self.n_output,)))

    @nn.compact
    def __call__(self, x):
        if self.standarize_input:
            x = (x - self.x_mean.value) / (self.x_std.value + 1e-12)

        Act = self.act
        for h in self.n_hidden:
            x = Act()(nn.Dense(h)(x))

        x = nn.Dense(self.n_output)(x)

        if self.standarize_output:
            x = x * (self.y_std.value + 1e-12) + self.y_mean.value
        return x

def fcn_from_config(cfg: dict) -> FCNStd:
    act_map = { "silu": SiLU, "relu": nn.relu, "gelu": nn.gelu, "tanh": jax.nn.tanh }
    act = act_map.get(str(cfg.get("act_fn", "silu")).lower(), SiLU)
    return FCNStd(
        n_input=int(cfg["n_input"]),
        n_output=int(cfg["n_output"]),
        n_hidden=tuple(int(x) for x in cfg["n_hidden"]),
        standarize_input=bool(cfg.get("standarize_input", True)),
        standarize_output=bool(cfg.get("standarize_output", True)),
        act=act,
    )

def torch_linear_to_flax(weight: torch.Tensor, bias: torch.Tensor) -> Dict[str, jnp.ndarray]:
    W = jnp.array(weight.detach().cpu().numpy().T)
    b = jnp.array(bias.detach().cpu().numpy())
    return {"kernel": W, "bias": b}

def _collect_torch_mlp_layers(sd: Dict[str, torch.Tensor], prefix: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    pairs = []
    for k in sd.keys():
        if k.startswith(prefix) and k.endswith(".weight"):
            tail = k[len(prefix):]
            try:
                idx_str = tail.split(".")[0]
                idx = int(idx_str)
            except Exception:
                continue
            wkey = f"{prefix}{idx}.weight"
            bkey = f"{prefix}{idx}.bias"
            if wkey in sd and bkey in sd:
                pairs.append((idx, sd[wkey], sd[bkey]))
    pairs.sort(key=lambda t: t[0])
    return [(w, b) for _, w, b in pairs]

def inspect_ckpt_dims(ckpt_path: str, prefix: str) -> Tuple[int, int, Tuple[int, ...]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    layers = _collect_torch_mlp_layers(sd, prefix=prefix)
    if not layers:
        raise ValueError(f"No layers found with prefix '{prefix}' in checkpoint.")
    n_input = int(layers[0][0].shape[1])
    n_output = int(layers[-1][0].shape[0])
    n_hidden = tuple(int(w.shape[0]) for w, _ in layers[:-1])
    return n_input, n_output, n_hidden

def load_and_map_params_dynamic(ckpt_path: str, variables: Dict[str, Any], prefix: str, scaler_keys: Dict[str, str] = None) -> Dict[str, Any]:
    if scaler_keys is None:
        scaler_keys = {"x_mean": "mean_input", "x_std": "std_input", "y_mean": "mean_output", "y_std": "std_output"}
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    new_params, new_scalers = dict(variables["params"]), dict(variables["scalers"])
    layers = _collect_torch_mlp_layers(sd, prefix=prefix)
    if not layers:
        raise ValueError(f"No MLP layers found with prefix '{prefix}'.")
    for i, (W, b) in enumerate(layers):
        new_params[f"Dense_{i}"] = torch_linear_to_flax(W, b)

    def _to_jnp(name: str) -> jnp.ndarray:
        return jnp.array(sd[name].detach().cpu().numpy())
    
    new_scalers["x_mean"] = _to_jnp(scaler_keys["x_mean"])
    new_scalers["x_std"] = jnp.where(_to_jnp(scaler_keys["x_std"])==0., 1., _to_jnp(scaler_keys["x_std"]))
    new_scalers["y_mean"] = _to_jnp(scaler_keys["y_mean"])
    new_scalers["y_std"] = jnp.where(_to_jnp(scaler_keys["y_std"])==0., 1., _to_jnp(scaler_keys["y_std"]))
    return {"params": new_params, "scalers": new_scalers}

def _cfg_to_fcnstd(cfg: Dict[str, Any]) -> Dict[str, Any]:
    n_input  = cfg.get("n_input",  cfg.get("in_features",  cfg.get("input_dim")))
    n_output = cfg.get("n_output", cfg.get("out_features", cfg.get("output_dim")))
    n_hidden  = cfg.get("n_hidden",  cfg.get("mlp_hidden_dims", cfg.get("hidden_dims")))
    if n_input is None or n_output is None or n_hidden is None:
        raise KeyError("Cannot find n_input/n_output/n_hidden in hparams.yaml")
    return {
        "n_input": int(n_input), "n_output": int(n_output),
        "n_hidden": [int(h) for h in n_hidden], "activation": cfg.get("activation", "silu"),
        "standarize_input": bool(cfg.get("standarize_input", True)),
        "standarize_output": bool(cfg.get("standarize_output", True)),
    }

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)