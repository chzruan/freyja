from typing import List, Optional, Union, Tuple
from pydantic import BaseModel


class ConfigHOD(BaseModel):
    gravity: "str"
    dataflag: "str"
    snapnum: int
    redshift: float
    log10M_halo_range: List[float]
    datasets: List[str]
    boxsize: float



class ConfigMeasure(BaseModel):
    r_xiR_binning: "str"
    r_xiR_min: float
    r_xiR_max: float
    r_xiR_binnum: int
    r_velmom_binning: "str"
    r_velmom_min: float
    r_velmom_max: float
    r_velmom_binnum: int
    s_xiS_binning: "str"
    s_xiS_min: float
    s_xiS_max: float
    s_xiS_binnum: int
    mu_xiS_binning: "str"
    mu_xiS_min: float
    mu_xiS_max: float
    mu_xiS_binnum: int



