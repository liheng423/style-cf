from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ....schema import HH_Col as HH
from ..datapack import BASE_CF_NAMES, SampleDataPack


def _cut_retrieve_data(npdata: np.ndarray, roller) -> np.ndarray:
    T = npdata.shape[0]
    window_len = int(roller.window_len)
    results = []
    i = 0
    while i + window_len <= T:
        results.append(npdata[i : i + window_len])
        i += roller.jump()
    if T - window_len > 0 and (len(results) == 0 or not np.array_equal(results[-1], npdata[-window_len:])):
        results.append(npdata[-window_len:])
    return np.asarray(results, dtype=np.float32)


def _retrieve_hh_data(
    df: pd.DataFrame,
    roller,
    time_col: str = HH.TIME,
    id_col: str = HH.ID,
    lead_id_col: str | None = None,
) -> np.ndarray:
    sort_cols = [id_col] + ([time_col] if time_col in df.columns else [])
    sorted_df = df.sort_values(sort_cols)

    all_windows = []
    for _, group in sorted_df.groupby(id_col, sort=False):
        self_id = group[id_col].to_numpy(dtype=np.float32)
        lead_id = (
            group[lead_id_col].to_numpy(dtype=np.float32)
            if lead_id_col is not None and lead_id_col in group.columns
            else self_id.copy()
        )

        group_values = np.column_stack(
            [
                self_id,
                group[HH.X_FOL].to_numpy(dtype=np.float32),
                group[HH.V_FOL].to_numpy(dtype=np.float32),
                group[HH.A_FOL].to_numpy(dtype=np.float32),
                group[HH.L_FOL].to_numpy(dtype=np.float32),
                lead_id,
                group[HH.X_LEAD].to_numpy(dtype=np.float32),
                group[HH.V_LEAD].to_numpy(dtype=np.float32),
                group[HH.A_LEAD].to_numpy(dtype=np.float32),
                group[HH.L_LEAD].to_numpy(dtype=np.float32),
            ]
        )
        windows = _cut_retrieve_data(group_values, roller=roller)
        all_windows.extend(windows)
    return np.asarray(all_windows, dtype=np.float32).reshape(-1, int(roller.window_len), 10)


def _concat_hh_h5(paths: list[str], key: str | None = None) -> pd.DataFrame:
    if key is None:
        return pd.concat([pd.read_hdf(path) for path in paths], ignore_index=True)
    return pd.concat([pd.read_hdf(path, key=key) for path in paths], ignore_index=True)


def load_hh_data(
    path_or_df: str | Path | pd.DataFrame,
    rise: bool,
    roller,
    in_kph: bool = False,
    kilo_norm: bool = False,
    time_col: str = HH.TIME,
    id_col: str = HH.ID,
    lead_id_col: str | None = None,
    dt: float = 0.1,
):
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        source_path = Path(path_or_df)
        suffix = source_path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            df = pd.read_hdf(source_path)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(source_path)
        else:
            df = pd.read_csv(source_path)

    data = _retrieve_hh_data(
        df=df,
        roller=roller,
        time_col=time_col,
        id_col=id_col,
        lead_id_col=lead_id_col,
    ).astype(np.float32, copy=False)
    print(f"Data Shape: {data.shape}")
    return SampleDataPack(data, BASE_CF_NAMES.copy(), rise=rise, kph=in_kph, kilo_norm=kilo_norm, dt=float(dt))


def load_hh_h5_list(
    paths: list[str],
    rise: bool,
    roller,
    key: str | None = None,
    in_kph: bool = False,
    kilo_norm: bool = False,
    time_col: str = HH.TIME,
    id_col: str = HH.ID,
    lead_id_col: str | None = None,
    dt: float = 0.1,
):
    df = _concat_hh_h5(paths, key=key)
    return load_hh_data(
        path_or_df=df,
        rise=rise,
        roller=roller,
        in_kph=in_kph,
        kilo_norm=kilo_norm,
        time_col=time_col,
        id_col=id_col,
        lead_id_col=lead_id_col,
        dt=dt,
    )
