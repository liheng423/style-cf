from typing import List, Mapping

# `namebuilder` parses the data structure in this shape, and enables to map the name back to index

#    
#    "x_groups": {
#        "enc_x": {"features": [CF.SELF_V, CF.DELTA_X, CF.DELTA_V, CF.SELF_L, CF.LEAD_L], "transform": True},
#        "dec_x": {"features": [CF.SELF_V, CF.LEAD_V], "transform": True},
#        "style": {"features": [CF.REACT, CF.THW, CF.DELTA_X, CF.SELF_V], "transform": True},
#    },
#    "y_groups": {
#        "y_seq": {"features": [CF.SELF_X, CF.SELF_V, CF.SELF_A, CF.DELTA_X, CF.LEAD_X]},


def _build_name_dict(feature_dict: Mapping[str, object]) -> dict[str, dict[str, int]]:
    name_dict: dict[str, dict[str, int]] = {}
    for group, spec in feature_dict.items():
        if isinstance(spec, dict):
            features = spec.get("features")
            if features is None:
                raise ValueError(f"Missing features for group '{group}'")
        else:
            features = spec
        name_dict[group] = {feat: idx for idx, feat in enumerate(features)}
    return name_dict

def _build_scaler_dict(feature: List[str], data_config: dict) -> dict[str, object]:

    scalers = {}
    for key, group in feature.items():
        if not group.get("transform", True):
            continue
        scalers[key] = data_config["scaler"]()
    
    return scalers