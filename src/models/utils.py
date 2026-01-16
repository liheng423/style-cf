from typing import List
import torch
from tensordict import TensorDict
import numpy as np
from src.schema import CFNAMES
##
# batch = TensorDict({
#     "enc_x": torch.randn([N, F, T], names=["N", "T", "F"]),
#     "dec_x": torch.randn([N, F, T], names=["N", "T", "F"]),
#     "style": torch.randn([N, F], , names=["N", "F"])
# }, batch_size=[N])
## 



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
    assert any(dim_name in tensordict_list[0][key].names for key in tensordict_list[0].keys()), \
        f"Dimension name '{dim_name}' not found in the names of any tensor in the first TensorDict."
    
    # Get all keys from the first TensorDict
    keys = tensordict_list[0].keys()
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
    if dim_name in tensordict_list[0].names:
        dim_idx = tensordict_list[0].names.index(dim_name)
        new_batch_size[dim_idx] *= len(tensordict_list)

    return first_type(stacked_data, batch_size=new_batch_size, names=tensordict_list[0].names)


class SliceableTensorDict(TensorDict):
    def __init__(self, source=None, batch_size=None, names=None):
        super().__init__(source=source, batch_size=batch_size, names=names)

    def sel(self, item=None, **indexers):
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
            return super().__getitem__(item)

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


    def __getitem__(self, key):
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

        new_order = [self.names[name] for name in sorted(new_name_dict, key=new_name_dict.get)]
        
        self.data = self.data[:, :, new_order]

        self.names = new_name_dict


    def normalize_kilopost(self, column_keys: List[str] = None) -> 'SampleDataPack':
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

        # Default columns to normalize
        if column_keys is None:
            column_keys = [LEAD_X, SELF_X]

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

        return SampleDataPack(new_data, self.names.copy(), kilo_norm=True, kph=self.kph, rise=True)

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

        return SampleDataPack(split_data, self.names.copy(), self.rise, self.kph, self.kilo_norm)

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

        return SampleDataPack(new_data, self.names.copy(), self.rise, False, self.kilo_norm)
    
              
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
        accs = self[:, :, CFNAMES.SELF_A][:, start_idx:]  # (N, T')
        x = self[:, :, CFNAMES.SELF_X]  # (N, T)
        v = self[:, :, CFNAMES.SELF_V]  # (N, T)

        # Build ground truth: shape (N, T, 2) with (x, v, a)
        initial_states = np.stack([x, v], dim=2)  # (N, T, 2)
        initial_states = initial_states[:, start_idx - 1]

        # Predict kinematics using acceleration
        preds = agent._predict_kinematics_np_batch(accs, initial_states, self.dt)  

        # Compute error
        pos_error = x[:, start_idx:] - preds[:, :, 0]
        spd_error = v[:, start_idx:] - preds[:, :, 1]

        return pos_error, spd_error
    
    def force_consistent(self):
        self.replace_col(np.gradient(self[:, :, CFNAMES.SELF_X], self.dt, axis=1), CFNAMES.SELF_V)
        self.replace_col(np.gradient(self[:, :, CFNAMES.SELF_V], self.dt, axis=1), CFNAMES.SELF_A)
        self.replace_col(np.gradient(self[:, :, CFNAMES.LEAD_X], self.dt, axis=1), CFNAMES.LEAD_V)
        self.replace_col(np.gradient(self[:, :, CFNAMES.LEAD_V], self.dt, axis=1), CFNAMES.LEAD_A)
        self.replace_col(self[:, :, CFNAMES.LEAD_V] - self[:, :, CFNAMES.SELF_V], CFNAMES.DELTA_V)
        self.replace_col(self[:, :, CFNAMES.LEAD_X] - self[:, :, CFNAMES.SELF_X], CFNAMES.DELTA_X)

def load_zen_data(path, rise, in_kph=False, kilo_norm=False):
    names = {
        CFNAMES.SELF_ID: 0,
        CFNAMES.SELF_X: 1,
        CFNAMES.SELF_V: 2,
        CFNAMES.SELF_A: 3,
        CFNAMES.SELF_L: 4,
        CFNAMES.LEAD_ID: 5,
        CFNAMES.LEAD_X: 6,
        CFNAMES.LEAD_V: 7,
        CFNAMES.LEAD_A: 8,
        CFNAMES.LEAD_L: 9
    }

    data: np.ndarray = np.load(path, allow_pickle=True).astype(np.float32)
    print(f"Data Shape: {data.shape}")

    datapack = SampleDataPack(data, names, rise=rise, kph=in_kph, kilo_norm=kilo_norm, dt=0.1)

    return datapack
