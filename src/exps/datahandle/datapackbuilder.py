
from typing import List
from .filters import CFFilter, filter_min_self_id_samples
from ..utils.datapack import SampleDataPack
from ...schema import CFNAMES as CF
from ...utils.logger import logger


def build_dataset(d: SampleDataPack, d_filters: List, d_filter_config: dict) -> SampleDataPack:
    """
    Build the dataset for car-following model training.
    """
    d = d.normalize_kilopost()  # normalize KILO to rising
    d.append_col(d[:, :, CF.LEAD_V] - d[:, :, CF.SELF_V], CF.DELTA_V)
    d.append_col(d[:, :, CF.LEAD_X] - d[:, :, CF.SELF_X], CF.DELTA_X)

    logger.info(f"Data build start | raw_samples={d.data.shape[0]}")

    if d_filters and isinstance(d_filters[0], str):
        # Resolve method from the current filtered datapack at every step.
        for fname in d_filters:
            before = int(d.data.shape[0])
            d_filter = CFFilter(d, d_filter_config)
            ffunc = getattr(d_filter, fname)
            mask = ffunc()
            d = SampleDataPack(d.data[mask], d.names.copy(), d.rise, d.kph, d.kilo_norm, d.dt)
            kept = int(d.data.shape[0])
            ratio = (kept / before * 100.0) if before > 0 else 0.0
            logger.info(f"Data filter | {fname}: {kept}/{before} ({ratio:.2f}%)")
    else:
        # Keep backward behavior for custom callables.
        before = int(d.data.shape[0])
        d_filter = CFFilter(d, d_filter_config)
        d = d_filter.filter(d_filters)
        kept = int(d.data.shape[0])
        ratio = (kept / before * 100.0) if before > 0 else 0.0
        logger.info(f"Data filter | custom_callable_filters: {kept}/{before} ({ratio:.2f}%)")

    min_self_id_samples = int(d_filter_config.get("min_self_id_samples", 1))
    if min_self_id_samples > 1:
        before = int(d.data.shape[0])
        d = filter_min_self_id_samples(d, min_samples=min_self_id_samples)
        kept = int(d.data.shape[0])
        ratio = (kept / before * 100.0) if before > 0 else 0.0
        logger.info(
            "Data filter | "
            f"min_self_id_samples>={min_self_id_samples}: {kept}/{before} ({ratio:.2f}%)"
        )

    d.force_consistent()
    logger.info(f"Data build done | kept_samples={d.data.shape[0]}")
    return d
