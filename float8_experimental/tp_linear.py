import torch

from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear, 
    RowParallelLinear,
)

from fairscale.nn.model_parallel.mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    scatter_to_model_parallel_region,
    reduce_from_model_parallel_region,
)

from float8_experimental.float8_linear import Float8LinearMixin, float8_linear

from float8_experimental.distributed_utils import (
    _AllGatherFloat8FwReduceScatterBw,
    _ReduceScatterFwAllGatherFloat8Bw,
)


class Float8ColumnParallelLinear(Float8LinearMixin, ColumnParallelLinear):
    """
    Same as `ColumnParallelLinear`, but with single GPU compute in float8.
    """
    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Float8 bookkeeping
        self.float8_pre_forward(input_)

        if self.use_sequence_parallel:
            # forward: all-gather
            #   Float8 comms: yes
            # backward: reduce-scatter
            #   Float8 comms: none, because we can't reduce in float8
            # cast activation and weight to float8
            input_fp8_ = self.cast_x_to_float8(input_, self.is_amax_initialized)
            input_parallel_fp8 = _AllGatherFloat8FwReduceScatterBw.apply(input_fp8_)
        else:
            # forward: no-op
            #   Float8 comms: no
            # backward: all-reduce
            #   Float8 comms: no, because we can't reduce in float8
            input_parallel = copy_to_model_parallel_region(input_)
            input_parallel_fp8 = self.cast_x_to_float8(
                input_parallel, self.is_amax_initialized)

        w_fp8 = self.cast_w_to_float8(
            self.weight, self.is_amax_initialized)

        # Matrix multiply.
        output_parallel = self.float8_mm(
            input_parallel_fp8, w_fp8, self.is_amax_initialized)
        output_parallel = self.cast_y_to_float8_in_bw(output_parallel)

        if self.bias is not None:
            output_parallel = output_parallel + self.bias.to(output_parallel.dtype)

        if self.gather_output:
            assert not self.use_sequence_parallel, 'unsupported'
            # All-gather across the partitions.
            # forward: gather
            #   Float8 comms: 
            #     - in the general case where the next op's input is in high 
            #       precision, nothing to do.
            #     - in the special case where the next op's input is in float8
            #       we could do the comms in float8. TODO(later) implement this
            #       if it becomes important
            # backward: split
            #   Float8 comms - no, split does not do any comms
            output = gather_from_model_parallel_region(output_parallel)
        else:
            # no-op
            output = output_parallel

        # Float8 bookkeeping
        self.float8_post_forward()
        return output

    @classmethod
    def from_float(cls, mod, emulate=False):
        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features,
            fake_out_features)
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.gather_output = mod.gather_output
        new_mod.output_size_per_partition = mod.output_size_per_partition
        new_mod.master_weight = mod.master_weight
        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        new_mod.emulate = emulate
        # TODO: test when creation is on cuda
        return new_mod

class Float8RowParallelLinear(Float8LinearMixin, RowParallelLinear):
    """
    Same as `RowParallelLinear`, but with single GPU compute in float8.
    """
    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Float8 bookkeeping
        self.float8_pre_forward(input_)

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            # no-op
            input_parallel = input_
        else:
            assert not self.use_sequence_parallel, 'unsupported'
            # forward: split
            #   Float8 comms: none
            # backward: gather
            #   Float8 comms: 
            #     - in the general case where the prev op's grad output is in high 
            #       precision, nothing to do.
            #     - in the special case where the prev op's grad output is in float8
            #       we could do the comms in float8. TODO(later) implement this
            #       if it becomes important
            input_parallel = scatter_to_model_parallel_region(input_)

        # cast activation and weight to float8
        input_parallel_fp8 = self.cast_x_to_float8(
            input_parallel, self.is_amax_initialized)
        w_fp8 = self.cast_w_to_float8(
            self.weight, self.is_amax_initialized)

        # Matrix multiply.
        output_parallel = self.float8_mm(
            input_parallel_fp8, w_fp8, self.is_amax_initialized)

        if self.use_sequence_parallel:
            # forward: reduce-scatter
            #   Float8 comms: none
            # backward: all-gather
            #   Float8 comms: yes
            output_ = _ReduceScatterFwAllGatherFloat8Bw.apply(output_parallel)
            output_ = self.cast_y_to_float8_in_bw(output_)
        else:
            # All-reduce across all the partitions.
            # forward: reduce
            #   Float8 comms: none
            # backward: no-op
            #   Float8 comms: none
            output_parallel = self.cast_y_to_float8_in_bw(output_parallel)

            # adding zero below is a hack
            # without this hack, we see the following error: https://gist.github.com/vkuzo/0ed84e35081c8c7d20d0f46ed4322704
            # pointing to autograd internals: https://fburl.com/code/eiipnnty
            # it seems like there are some issues within chaining torch.autograd.Function instances
            # together
            # TODO(future) figure out a workaround
            output_parallel = output_parallel + 0.0

            output_ = reduce_from_model_parallel_region(output_parallel)

        if self.bias is not None:
            output_ = output_ + self.bias

        # Float8 bookkeeping
        self.float8_post_forward()

        return output_

    @classmethod
    def from_float(cls, mod, emulate=False):
        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features,
            fake_out_features)
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.input_is_parallel = mod.input_is_parallel
        new_mod.input_size_per_partition = mod.input_size_per_partition
        new_mod.master_weight = mod.master_weight
        # TODO: test when creation is on cuda
        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        new_mod.emulate = emulate
        return new_mod

def swap_tp_linear_with_float8_linear(model, emulate=False):
    """
    Replaces all instances of {Column|Row}ParallelLinear in the given model
    with their Float8 enabled versions.

    Args:
        model (torch.nn.Module): The model to modify.
        emulate (bool, optional): Whether to emulate the fp8 matmul logic in float32.
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if isinstance(child, ColumnParallelLinear):
            new_child = Float8ColumnParallelLinear.from_float(child, emulate)
            setattr(model, name, new_child)
        elif isinstance(child, RowParallelLinear):
            new_child = Float8RowParallelLinear.from_float(child, emulate)
            setattr(model, name, new_child)
        else:
            swap_tp_linear_with_float8_linear(child, emulate)
