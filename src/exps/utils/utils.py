from typing import Any, Callable, Dict, List, Iterable, Self, cast, overload
import os
from datetime import datetime
import torch
from tensordict import TensorDict, TensorDictBase
from torch.utils.data._utils.collate import default_collate
import numpy as np
from src.exps.utils.utils_kine import _predict_kinematics_np
from src.schema import CFNAMES as CF
from src.stylecf.schema import TensorNames
##
# batch = TensorDict({
#     "enc_x": torch.randn([N, F, T], names=["N", "T", "F"]),
#     "dec_x": torch.randn([N, F, T], names=["N", "T", "F"]),
#     "style": torch.randn([N, F], , names=["N", "F"])
# }, batch_size=[N])
## 
def drop_tensor_names(tensor: torch.Tensor) -> torch.Tensor:
    if getattr(tensor, "names", None) is None:
        return tensor
    return tensor.rename(None)

def _stack_named_tensors(tensors):
    names = tensors[0].names
    unnamed = [t.rename(None) if t.names is not None else t for t in tensors]
    stacked = torch.stack(unnamed, dim=0)
    if names is not None:
        stacked = stacked.refine_names(TensorNames.N, *names)
    return stacked


def _stack_tensordict(batch):
    if not batch:
        return TensorDict({}, batch_size=[0])
    first = batch[0]
    stacked = {}
    for key in first.keys():
        tensors = [td[key] for td in batch]
        stacked[key] = _stack_named_tensors(tensors)
    return TensorDict(stacked, batch_size=[len(batch)])


def _collate(batch):
    if not batch:
        return batch
    first = batch[0]
    if isinstance(first, TensorDict):
        return _stack_tensordict(batch)
    if (
        isinstance(first, tuple)
        and len(first) == 2
        and isinstance(first[0], TensorDict)
        and isinstance(first[1], TensorDict)
    ):
        xs, ys = zip(*batch)
        return _stack_tensordict(list(xs)), _stack_tensordict(list(ys))
    return default_collate(batch)

def td_cat(tensordicts: list[TensorDict]) -> TensorDict:
    td = cast(list, tensordicts)
    return cast(TensorDict, torch.cat(td, dim=0))
    
def stack_name(tensordict_list: list[TensorDict], dim_name: str):
    """
    Stacks a TensorDict along the specified dimension name.
    If the dimension is not present in a tensor, it remains unchanged.
    The rest dimensions are assumed to be consistent across all TensorDicts. (i.e., same size).

    Args:
        tensordict (List[TensorDict]): The input TensorDict.
        dim_name (str): The name of the dimension to stack.

    Returns:
        TensorDict: A new TensorDict with the dimension stacked into the indicated dimension.
    """    
    first_type = type(tensordict_list[0])
    if not all(isinstance(td, first_type) for td in tensordict_list):
        raise TypeError("All items in tensordict_list must have the same class")
    
    if not tensordict_list:
        return TensorDict({}, batch_size=[])
    
    # Check if dim_name is present in at least one tensor's names within the first TensorDict
    keys = list(cast(Iterable[str], tensordict_list[0].keys()))
    assert any(
        tensordict_list[0][key].names is not None and dim_name in tensordict_list[0][key].names
        for key in keys
    ), \
        f"Dimension name '{dim_name}' not found in the names of any tensor in the first TensorDict."
    
    # Get all keys from the first TensorDict
    keys = cast(Iterable[str], keys)
    stacked_data = {}

    for key in keys:
        tensors_to_stack = []
        for td in tensordict_list:
            if key in td.keys():
                tensors_to_stack.append(td[key])
            else:
                raise ValueError(f"Key '{key}' not found in all TensorDicts.")

        if dim_name in tensordict_list[0][key].names:
            stacked_data[key] = torch.concat(tensors_to_stack, dim=dim_name)
        else:
            # For static features, ensure they are consistent or pick the first one
            stacked_data[key] = tensors_to_stack[0]

    # If the dim_name is one of the batch dimensions, update batch_size
    new_batch_size = list(tensordict_list[0].batch_size)
    names = cast(List[str], tensordict_list[0].names)
    if dim_name in names:
        dim_idx = names.index(dim_name)
        new_batch_size[dim_idx] *= len(tensordict_list)

    return first_type(stacked_data, batch_size=new_batch_size, names=names)


class SliceableTensorDict(TensorDict):
    def __init__(self, source, batch_size, names):
        super().__init__(source=source, batch_size=batch_size, names=names)
    
    def __new__(cls, *args, **kwargs) -> Self:
        return cast(Self,super().__new__(cls))


    def get(self, key) -> Any:
        result = super().__getitem__(cast(Any, key))
        if (
            isinstance(key, list)
            and all(isinstance(k, str) for k in key)
            and isinstance(result, TensorDictBase)
        ):
            return SliceableTensorDict(
                result,
                batch_size=result.batch_size,
                names=result.names,
            )
        return result

    def sel(self, item=None, **indexers) -> 'SliceableTensorDict':
        """
        Slice tensors along named dimensions.

        Examples:
            td.sel(T=0)                       # int index
            td.sel(T=slice(1, 5))             # slice index
            td.sel(T=[0, 2, 4], X=slice(0, 3))
            td.sel(("T", 0))                  # backwards-compatible
            td.sel({"T": 0, "X": [1, 3]})     # dict form

        If a named dimension does not exist on a tensor, that tensor is left unchanged.
        """
        if item is not None and indexers:
            raise ValueError("Provide either item or keyword indexers, not both")

        if item is None:
            selectors = indexers
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
        ):
            selectors = {item[0]: item[1]}
        elif isinstance(item, dict):
            selectors = item
        else:
            raise TypeError(
                "sel() only accepts named-dimension selectors; "
                "use get() or another method for key-based indexing."
            )

        if not selectors:
            return self

        for dim_name in selectors.keys():
            if self.names is not None and dim_name in self.names:
                raise ValueError(f"Slicing along batch axis '{dim_name}' is not allowed")

        def _validate_selector(selector):
            if not isinstance(selector, (int, slice, list, torch.Tensor)):
                raise TypeError(
                    "Selector must be int, slice, list or torch.Tensor when indexing by name"
                )

        for selector in selectors.values():
            _validate_selector(selector)

        def _index_tensor(tensor):
            result = tensor
            for dim_name, selector in selectors.items():
                if dim_name not in result.names:
                    continue
                dim = result.names.index(dim_name)
                if isinstance(selector, int):
                    result = result.select(dim=dim, index=selector)
                    continue

                idx = [slice(None)] * result.ndim
                idx[dim] = (
                    torch.as_tensor(selector, device=result.device)
                    if isinstance(selector, list)
                    else selector
                )
                result = result[tuple(idx)]
            return result

        new_data = {k: _index_tensor(v) for k, v in self.items()}

        return SliceableTensorDict(new_data, batch_size=self.batch_size, names=self.names)
    
    def to(self, device):
        new_data = {k: v.to(device) for k, v in self.items()}
        new_td = SliceableTensorDict(new_data, batch_size=self.batch_size, names=self.names)
        return cast("SliceableTensorDict", new_td)
    



class SampleDataPack:
    """
    A class to manage and manipulate car-following data stored in a 3D numpy array format.
    """

    def __init__(self, data: np.ndarray, name_dict: dict, rise: bool, kph: bool, kilo_norm, dt: float):
        """
        Args:
            data (np.ndarray): the data in the format of (sample, time, num_feature)
            name_dict (dict): the mapping between column name and index
        """
        self.data = data
        self.names = name_dict
        self.rise = rise
        self.kph = kph
        self.kilo_norm = kilo_norm
        self.dt = dt

    
    def append_col(self, col: np.ndarray, col_name: str):
        """
        Args:
            col (np.ndarray): New column data to append. Shape must be (samples, T) or (samples, T, 1).
            col_name (str): Name of the new column.
        """

        assert col_name not in self.names

        assert col.shape[:2] == self.data.shape[:2]

        if len(col.shape) == 2:
            col = col[:, :, np.newaxis]
        
        col_index = self.data.shape[2]
        self.data = np.concatenate([self.data, col], axis=2)
        self.names[col_name] = col_index

    def replace_col(self, col: np.ndarray, col_name: str):
        """
        Replace an existing column in the data by name.

        Args:
            col (np.ndarray): New column data to replace with. Shape must be (samples, T) or (samples, T, 1).
            col_name (str): Name of the column to replace.
        """
        assert col_name in self.names, f"Column '{col_name}' not found."
        assert col.shape[:2] == self.data.shape[:2], "Shape mismatch with existing data."

        if len(col.shape) == 2:
            col = col[:, :, np.newaxis]

        col_index = self.names[col_name]
        self.data[:, :, col_index] = col[:, :, 0]

    def head(self, n: int) -> 'SampleDataPack':
        """
        Get the first n samples of the data.

        Args:
            n (int): Number of samples to retrieve.

        Returns:
            DataPack: A new DataPack instance with the first n samples.
        """
        return SampleDataPack(self.data[:n], self.names.copy(), self.rise, self.kph, self.kilo_norm, self.dt)

    def __getitem__(self, key) -> np.ndarray:
        """
        Supports:
        - [i, j, "feature_name"]
        - [:, :, "feature_name"]
        - [i, j, ["feature1", "feature2"]]
        """
        if isinstance(key, tuple) and len(key) == 3:
            i, j, feat = key
            if isinstance(feat, str):
                k = self.names[feat]
                return self.data[i, j, k]
            elif isinstance(feat, list):
                k = [self.names[f] for f in feat]
                return self.data[i, j, k]
            else:
                raise TypeError("Third index must be a string or list of strings.")
        else:
            return self.data[key]
        

    def reorder_features(self, new_name_dict: dict):
        """
        Reorder the feature dimension of data based on new_name_dict.

        Args:
            new_name_dict (dict): New mapping from feature name to index (0-based).
                                Must contain the same keys as the old mapping,
                                but possibly in a different order.

        Raises:
            ValueError: if the keys in the new_name_dict do not match the original.
        """
        # check keys consistency
        if set(new_name_dict.keys()) != set(self.names.keys()):
            raise ValueError("New name_dict must contain the same keys as the original.")

        new_order = [self.names[name] for name in sorted(new_name_dict, key=lambda x: new_name_dict[x])]
        
        self.data = self.data[:, :, new_order]

        self.names = new_name_dict


    def normalize_kilopost(self, column_keys: List[str]= [CF.LEAD_X, CF.SELF_X]) -> 'SampleDataPack':
        """
        Normalize selected columns (e.g., 'LEAD_X', 'SELF_X') across all samples.
        
        Args:
            column_keys (List[str], optional): List of feature keys to normalize. 
                                            Defaults to ['LEAD_X', 'SELF_X'].

        Returns:
            DataPack: A new DataPack instance with normalized kilopost columns.
        """
        if self.kilo_norm is True:
            print("The kilopost is already normalized, nothing is done")
            return self


        # Get indices of the columns to normalize
        column_indices = []
        for key in column_keys:
            idx = self.names.get(key)
            if idx is None:
                raise KeyError(f"Key '{key}' not found in names dict.")
            column_indices.append(idx)

        # Make a copy of the original data
        new_data = self.data.copy()

        # Extract and concatenate selected columns along axis=1
        extracted = np.stack([new_data[:, :, idx] for idx in column_indices], axis=1)  # shape: (N, C, T)

        # Normalize
        mins = np.min(extracted, axis= (1, 2), keepdims=True)
        maxs = np.max(extracted, axis= (1, 2), keepdims=True)
        normalized = extracted - mins if self.rise else maxs - extracted

        # Replace original data columns with normalized values
        for i, idx in enumerate(column_indices):
            new_data[:, :, idx] = normalized[:, i, :]

        return SampleDataPack(new_data, self.names.copy(), kilo_norm=True, kph=self.kph, rise=True, dt=self.dt)

    def split_by_time_windows(self, windows: list[tuple[int, int]]) -> 'SampleDataPack':
        """
        Efficiently split each sample along the time axis using numpy slicing,
        and return a new DataPack with stacked split samples.

        Args:
            windows (list of tuple): Each tuple is a (start, end) window on the time axis.

        Returns:
            DataPack: New DataPack with additional samples from time splits.
        """

        # Check windows validity
        for start, end in windows:
            if not (0 <= start < end <= self.data.shape[1]):
                raise ValueError(f"Invalid window ({start}, {end})")

        # Slice and stack using numpy
        split_data = np.concatenate(
            [self.data[:, start:end, :].copy() for (start, end) in windows],
            axis=0
        )  # Result shape: (samples * len(windows), time_window, features)

        return SampleDataPack(split_data, self.names.copy(), self.rise, self.kph, self.kilo_norm, dt=self.dt)

    def convert_speed_to_ms(self, cols: list[str]) -> 'SampleDataPack':
        """
        Convert the given speed columns from km/h to m/s, and return a new DataPack.

        Args:
            cols (list[str]): List of column names to convert.

        Returns:
            DataPack: A new DataPack with specified columns converted to m/s.
        """

        if not self.kph: 
            print("The data is already in m/s, nothing is executed.")
            return self

        for col in cols:
            if col not in self.names:
                raise ValueError(f"Column '{col}' not found in data.")

        kph2ms = lambda x: x / 3.6

        new_data = self.data.copy()
        for col in cols:
            idx = self.names[col]
            new_data[:, :, idx] = kph2ms(new_data[:, :, idx])

        return SampleDataPack(new_data, self.names.copy(), self.rise, False, self.kilo_norm, self.dt)
    
              
    def check_consistency(self, start_idx: int = 1):
        """
        Check physical consistency between acceleration, speed and displacement.

        Args:
            start_idx (int): Start index in the time axis for comparison (skip unstable initial frames).
           dt (float): Time interval between steps in seconds.

        Returns:
            Tuple of:
                - position error: tensor of shape (N, T-start_idx)
                - speed error: tensor of shape (N, T-start_idx)
        """
        assert self.kph == False

        # Extract tensors
        accs = self[:, :, CF.SELF_A][:, start_idx:]  # (N, T')
        x = self[:, :, CF.SELF_X]  # (N, T)
        v = self[:, :, CF.SELF_V]  # (N, T)

        # Build ground truth: shape (N, T, 2) with (x, v, a)
        initial_states = np.stack([x, v], axis=2)  # (N, T, 2)
        initial_states = initial_states[:, start_idx - 1]

        # Predict kinematics using acceleration
        preds = _predict_kinematics_np(accs, initial_states, self.dt)

        # Compute error
        pos_error = x[:, start_idx:] - preds[:, :, 0]
        spd_error = v[:, start_idx:] - preds[:, :, 1]

        return pos_error, spd_error
    
    def force_consistent(self):
        self.replace_col(np.gradient(self[:, :, CF.SELF_X], self.dt, axis=1), CF.SELF_V)
        self.replace_col(np.gradient(self[:, :, CF.SELF_V], self.dt, axis=1), CF.SELF_A)
        self.replace_col(np.gradient(self[:, :, CF.LEAD_X], self.dt, axis=1), CF.LEAD_V)
        self.replace_col(np.gradient(self[:, :, CF.LEAD_V], self.dt, axis=1), CF.LEAD_A)
        self.replace_col(self[:, :, CF.LEAD_V] - self[:, :, CF.SELF_V], CF.DELTA_V)
        self.replace_col(self[:, :, CF.LEAD_X] - self[:, :, CF.SELF_X], CF.DELTA_X)

def load_zen_data(path, rise, in_kph=False, kilo_norm=False):
    names = {
        CF.SELF_ID: 0,
        CF.SELF_X: 1,
        CF.SELF_V: 2,
        CF.SELF_A: 3,
        CF.SELF_L: 4,
        CF.LEAD_ID: 5,
        CF.LEAD_X: 6,
        CF.LEAD_V: 7,
        CF.LEAD_A: 8,
        CF.LEAD_L: 9
    }

    data: np.ndarray = np.load(path, allow_pickle=True).astype(np.float32)
    print(f"Data Shape: {data.shape}")

    datapack = SampleDataPack(data, names, rise=rise, kph=in_kph, kilo_norm=kilo_norm, dt=0.1)

    return datapack
# 

def build_id_datapack(
    datapack: SampleDataPack,
    require_const_self_id: bool = True,
    key_by_id: bool = False,
) -> Dict[int, SampleDataPack]:
    """
    Group samples by SELF_ID to create an id_datapack.

    Args:
        datapack: Source SampleDataPack with SELF_ID in feature names.
        require_const_self_id: If True, only keep samples where SELF_ID is
            constant over time.
        key_by_id: If True, use the vehicle ID as dict key; otherwise use
            0..N-1 indices for compatibility with calibrate_idm.
    """
    if CF.SELF_ID not in datapack.names:
        raise KeyError(f"{CF.SELF_ID} not found in datapack.names")

    id_idx = datapack.names[CF.SELF_ID]
    ids = datapack.data[:, :, id_idx]
    first_ids = ids[:, 0]

    if require_const_self_id:
        same_id = np.all(ids == ids[:, [0]], axis=1)
    else:
        same_id = np.ones(ids.shape[0], dtype=bool)

    unique_ids = np.unique(first_ids[same_id])

    id_datapack: Dict[int, SampleDataPack] = {}
    for i, vid in enumerate(unique_ids):
        mask = (first_ids == vid) & same_id
        if not np.any(mask):
            continue
        sub = SampleDataPack(
            datapack.data[mask].copy(),
            datapack.names.copy(),
            datapack.rise,
            datapack.kph,
            datapack.kilo_norm,
            datapack.dt,
        )
        key = int(vid) if key_by_id else i
        id_datapack[key] = sub

    return id_datapack


## saving functions

def model_save(model_dict, path):
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = model_dict.state_dict() if hasattr(model_dict, "state_dict") else model_dict
    torch.save(payload, path)
    return path


def ensure_dir(folder):
    """
        Ensure the folder exists. If not, create it.
    """
    os.makedirs(folder, exist_ok=True)
    


    
