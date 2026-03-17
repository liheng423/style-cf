
from typing import List
from .filters import CFFilter
from ..utils.utils import SampleDataPack
from ...schema import CFNAMES as CF


def build_dataset(d: SampleDataPack, d_filters: List, d_filter_config: dict) -> SampleDataPack:
    """
    Build the dataset for car-following model training.
    """
    d = d.normalize_kilopost()  # normalize KILO to rising
    d.append_col(d[:, :, CF.LEAD_V] - d[:, :, CF.SELF_V], CF.DELTA_V)
    d.append_col(d[:, :, CF.LEAD_X] - d[:, :, CF.SELF_X], CF.DELTA_X)
    d_filter = CFFilter(d, d_filter_config)
    if d_filters and isinstance(d_filters[0], str):
        d_filters = [getattr(d_filter, name) for name in d_filters]
    d = d_filter.filter(d_filters)

    d.force_consistent()
    return d
