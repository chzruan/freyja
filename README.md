# Freyja

`freyja` provides pretrained cosmology emulators for halo statistics and related correlation functions. The repository currently ships checkpoint files for the halo mass function, linear halo bias, scale-dependent halo bias, and matter correlation function emulators at `z = 0.25`.

## Installation

From the repository root, install the package in editable mode:

```bash
pip install -e .
```

This installs `freyja` and its Python dependencies from the local checkout.

## Calling the Halo Emulators

The halo emulator APIs use:

- cosmology parameters ordered as `[Om0, h, S8, ns]`
- halo masses in `log10(M / (Msun / h))`
- the default pretrained checkpoints bundled under `freyja/emulators/checkpoints/`

### Halo mass function

`HMFEmulator` returns the cumulative halo mass function `n(>M)` and can also be used to derive `dn / dlog10M`.

```python
import numpy as np
from freyja.emulators import HMFEmulator

cosmo = np.array([0.315, 0.674, 0.811, 0.965])
log10M_edges = np.arange(12.0, 15.6, 0.1)
log10M_centres = 0.5 * (log10M_edges[:-1] + log10M_edges[1:])

hmf = HMFEmulator()
cumulative_hmf = hmf.cumulative_hmf(cosmo, log10M_edges)
dndlog10M = hmf.get_dndlog10M(cosmo, log10M_centres, dlog10M=0.1)
```

### Linear halo bias

`HaloLinearBiasEmulator` predicts the large-scale halo bias as a function of halo mass.

```python
import numpy as np
from freyja.emulators import HaloLinearBiasEmulator

cosmo = np.array([0.315, 0.674, 0.811, 0.965])
logM = np.array([12.5, 13.0, 13.5, 14.0])

bias_emu = HaloLinearBiasEmulator()
bias = bias_emu.predict(cosmo, logM)
```

### Scale-dependent halo bias

`HaloBetaEmulator` predicts
`beta(r | M1, M2) = xi_hh(r | M1, M2) / xi_mm(r)`.

For a single mass pair:

```python
import numpy as np
from freyja.emulators import HaloBetaEmulator

cosmo = np.array([0.315, 0.674, 0.811, 0.965])

beta_emu = HaloBetaEmulator()
r_beta, beta = beta_emu.predict_from_masses(cosmo, logM1=13.0, logM2=13.5)
```

For a full symmetric mass-pair matrix:

```python
import numpy as np
from freyja.emulators import HaloBetaEmulator

cosmo = np.array([0.315, 0.674, 0.811, 0.965])
logM_bins = np.array([12.8, 13.2, 13.6])

beta_emu = HaloBetaEmulator()
r_beta, beta_matrix = beta_emu.predict(cosmo, logM_bins)
```

`beta_matrix` has shape `(N_mass, N_mass, N_r)`.

### Halo-halo correlation function

`HaloXiDiffMCalculator` combines the halo-bias and matter-correlation emulators to return `xi_hh(r | M1, M2)` directly.

```python
import numpy as np
from freyja.emulators import HaloXiDiffMCalculator

cosmo = np.array([0.315, 0.674, 0.811, 0.965])
r = np.geomspace(1.0, 100.0, 64)

xi_calc = HaloXiDiffMCalculator()
xi_hh = xi_calc.calculate_M12(cosmo, logM1=13.0, logM2=13.5, r=r)
```

To evaluate a full mass matrix on the same `r` grid:

```python
import numpy as np
from freyja.emulators import HaloXiDiffMCalculator

cosmo = np.array([0.315, 0.674, 0.811, 0.965])
logM_bins = np.array([12.8, 13.2, 13.6])
r = np.geomspace(1.0, 100.0, 64)

xi_calc = HaloXiDiffMCalculator()
xi_hh_matrix = xi_calc.calculate_mass_matrix(cosmo, logM_bins, r)
```

## Notes

- The default pretrained emulators are configured for `z = 0.25`.
- If you want to use a different checkpoint file, pass the relevant path to the emulator constructor, for example `HMFEmulator(gp_emulator_path=...)` or `HaloBetaEmulator(checkpoint_path=...)`.
