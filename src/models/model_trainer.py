from typing import List
from models.filters import CFFilter
from src.models.utils import SampleDataPack
from src.schema import CFNAMES
from src.models.model_trainer import build_style_dataset

def build_dataset(d: SampleDataPack, d_filters: List[CFFilter], d_filter_config: dict) -> SampleDataPack:
    """
    Build the dataset for car-following model training.
    """
    d = d.normalize_kilopost() # normalize KILO to rising
    d.append_col(d[:, :, CFNAMES.LEAD_V] - d[:, :, CFNAMES.SELF_V], CFNAMES.DELTA_V)
    d.append_col(d[:, :, CFNAMES.LEAD_V] - d[:, :, CFNAMES.SELF_X], CFNAMES.DELTA_X)
    d_filter = CFFilter(d, d_filter_config)
    d = d_filter.filter(d_filters)

    d.force_consistent()

    

    return d

def build_style_dataset(d: SampleDataPack, d_filters: List[CFFilter], d_filter_config: dict) -> SampleDataPack:
    """
    Build the dataset for style-based car-following model training.
    """
    d = build_dataset(d, d_filters, d_filter_config)
    

    return d
