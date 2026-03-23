from __future__ import annotations

REGIMES = ["A", "D", "C", "S", "Am", "Dm", "As", "Ds"]

DEC_REGIME_IDX = [1, 5, 7]
ACC_REGIME_IDX = [0, 4, 6]


def map_segment_to_regime_index(slope: float, mean_speed: float, thres_dict: dict[str, float]) -> int:
    if slope > 0.1:
        if slope >= thres_dict["acute_acc"]:
            return REGIMES.index("As")
        if slope >= thres_dict["mod_acc"]:
            return REGIMES.index("Am")
        return REGIMES.index("A")

    if slope < -0.1:
        if slope <= thres_dict["acute_dec"]:
            return REGIMES.index("Ds")
        if slope <= thres_dict["mod_dec"]:
            return REGIMES.index("Dm")
        return REGIMES.index("D")

    if mean_speed < 1.0:
        return REGIMES.index("S")
    return REGIMES.index("C")


__all__ = ["REGIMES", "DEC_REGIME_IDX", "ACC_REGIME_IDX", "map_segment_to_regime_index"]
