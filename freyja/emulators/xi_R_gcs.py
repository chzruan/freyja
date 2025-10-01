import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from sunbird.emulators import FCN
from astropy.cosmology import FlatLambdaCDM, Planck15
from hmd.galaxy import Galaxy

aPATH = Path(__file__).parent
_r_emu = np.load(aPATH / Path('data/r_xi_R_cc.npy'))
emulator_r2xiRcc = FCN.load_from_checkpoint(
    aPATH / Path('checkpoints/xi_R_cc.ckpt'), 
    strict=True,
).to('cpu')

def get_xiRcc(
    cosmology: FlatLambdaCDM = Planck15,
    galaxy: Galaxy = Galaxy(
        logM_min    = 13.6,
        logM1       = 14.2,
        sigma_logM  = 0.5,
        alpha       = 0.5,
        v_bias_centrals     = 1.0,
        v_bias_satellites   = 1.0,
        kappa = 1.0,
        B_cen = 0.,
        B_sat = 0.,
    ),
    r_input: Optional[np.array] = None,
):  
    x_input_cc = np.array([
        cosmology.Om0, 
        cosmology.h, 
        cosmology.S8, 
        cosmology.ns, 
        galaxy.logMcut, 
        galaxy.logM1, 
        galaxy.logsigma, 
    ])
    _r2xiRcc = emulator_r2xiRcc.get_prediction(torch.Tensor(x_input)).numpy()
    
    if (r_input is not None):
        return r_input, IUS(
            np.log10(_r_emu),
            _r2xiRcc,
            ext='zeros',
        )(np.log10(r_input)) / r_input**2
    else:
        return _r_emu, _r2xiRcc / (_r_emu**2)



