from typing import List, Optional, Union
from pydantic import BaseModel, Extra, ConfigDict


class ConfigHOD(BaseModel):
    model_config = ConfigDict(extra='allow')

    gravity: str
    dataflag: str
    snapnum: int
    redshift: float
    log10M_halo_range: List[float]
    datasets: List[str]
    boxsize: float

    # Optional fields for future extensions
    extra_params: Optional[dict] = None



class ConfigMeasure(BaseModel):
    model_config = ConfigDict(extra='allow')

    r_xiR_binning: str
    r_xiR_min: float
    r_xiR_max: float
    r_xiR_binnum: int

    r_velmom_binning: str
    r_velmom_min: float
    r_velmom_max: float
    r_velmom_binnum: int

    s_xiS_binning: str
    s_xiS_min: float
    s_xiS_max: float
    s_xiS_binnum: int

    # Optional for any other future parameters
    extra_params: Optional[dict] = None

