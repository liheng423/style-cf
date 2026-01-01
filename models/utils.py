import torch
from tensordict import TensorDict
from typing import List

##
# batch = TensorDict({
#     "enc_x": torch.randn(N, T, F),
#     "dec_x": torch.randn(N, T, F),
#     # 注意：如果將 Time 納入 batch_size，靜態數據通常需要 expand
#     "style": torch.randn(N, F)
# }, batch_size=[N, T], names=["S", "T"])
## 

def stack(tensordict_list: List[TensorDict], dim_name: str) -> TensorDict:
    """
    Stacks a TensorDict along the specified dimension.
    If the dimension is not present in a tensor, it remains unchanged.
    The rest dimensions are assumed to be consistent across all TensorDicts. (i.e., same size).

    Args:
        tensordict (List[TensorDict]): The input TensorDict.
        dim_name (str): The name of the dimension to stack.

    Returns:
        TensorDict: A new TensorDict with the dimension stacked into the indicated dimension.
    """    
  
    if not tensordict_list:
        return TensorDict({}, batch_size=[0])
    
    assert torch.all([dim_name in tensordict_list[i].batch_names for i in range(len(tensordict_list))]), f"Dimension name '{dim_name}' must be present in all TensorDicts' batch names."

    # Get all keys from the first TensorDict (assuming all have the same keys)
    keys = tensordict_list[0].keys()
    
    stacked_data = {}
    
    # Determine the index of the dim_name in the batch_names
    # If dim_name is not in batch_names, it means we are stacking along a new dimension
    # or stacking static tensors.
    dim_idx = -1
    if dim_name in tensordict_list[0].batch_names:
        dim_idx = tensordict_list[0].batch_names.index(dim_name)

    for key in keys:
        tensors_to_stack = []
        for td in tensordict_list:
            tensor = td.get(key)
            tensors_to_stack.append(tensor)
        
        if dim_idx != -1: # If the dimension exists in the batch_names
            stacked_data[key] = torch.cat(tensors_to_stack, dim=dim_idx)
        else: # If the dimension does not exist, assume it's a static tensor or a new dimension to be created
            # For static tensors, we assume they are identical across the list, so just take the first one
            # If we are stacking along a new dimension, this logic needs to be more complex
            # For now, we assume if dim_name is not in batch_names, the tensor is static
            stacked_data[key] = tensors_to_stack[0]

    # Determine the new batch_size and batch_names
    new_batch_names = list(tensordict_list[0].batch_names)
    new_batch_size = list(tensordict_list[0].batch_size)

    if dim_idx != -1:
        # Sum the sizes of the stacked dimension
        new_batch_size[dim_idx] = sum(td.batch_size[dim_idx] for td in tensordict_list)
    else:
        # If dim_name was not in batch_names, we are effectively adding a new batch dimension
        #
