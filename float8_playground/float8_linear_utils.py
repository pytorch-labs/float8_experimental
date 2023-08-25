import torch

from float8_utils import (
    tensor_to_amax,
    amax_to_scale,
)

def _maybe_initialize_amaxes_for_float8_cast(x, cur_amax, amax_history, is_initialized):
    """
    If x is about to be cast to `float8` and the amax buffers are not initialized,
    initializes them inplace.
    """
    if is_initialized:
        return
    with torch.no_grad():
        new_amax = tensor_to_amax(x)
        cur_amax.fill_(new_amax)
        amax_history[0] = new_amax

def _update_history_with_new_amax(new_amax, amax_history):
    """
    Updates `amax_history` (the last N cur_amax values) inplace with the value 
    of `new_amax`.
    """
    new_amax_history = torch.roll(amax_history, 1)
    new_amax_history[0] = new_amax
    amax_history.copy_(new_amax_history)
