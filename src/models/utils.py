import torch
from tensordict import TensorDict

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

    return TensorDict(stacked_data, batch_size=new_batch_size, names=tensordict_list[0].names)