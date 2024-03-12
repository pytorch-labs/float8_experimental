import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

# subclass the ColwiseParallel and RowwiseParallel classes
# to add the float8 support
# The parameter sharding stays the same as the core
# ColwiseParallel and RowwiseParallel, the only difference
# here is that in input/output handling we do casting after
# creating the DTensor.

# NOTE: This only works and tested with the DynamicLinear


class Float8ColwiseParallel(ColwiseParallel):
    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        input_tensor = mod.cast_to_float8_e4m3fn(input_tensor)  # DTensor(Float8Tensor)

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        outputs = outputs.redistribute(
            placements=output_layouts, async_op=True
        )  # DTensor(torch.Tensor)

        # fwd noop bwd cast to DTensor(Float8Tensor)
        outputs = mod.cast_to_float8_e5m2_bw(outputs)

        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from float8_experimental.float8_dynamic_linear import Float8DynamicLinear

        if not isinstance(module, Float8DynamicLinear):
            raise ValueError(
                f"Expecting module to be Float8DynamicLinear but found {type(module)}"
            )

        return super()._apply(module, device_mesh)


class Float8RowwiseParallel(RowwiseParallel):
    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        input_tensor = mod.cast_to_float8_e4m3fn(input_tensor)  # DTensor(Float8Tensor)

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        outputs = outputs.redistribute(placements=output_layouts, async_op=True)

        # fwd noop bwd cast to DTensor(Float8Tensor)
        outputs = mod.cast_to_float8_e5m2_bw(outputs)

        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from float8_experimental.float8_dynamic_linear import Float8DynamicLinear

        if not isinstance(module, Float8DynamicLinear):
            raise ValueError(
                f"Expecting module to be Float8DynamicLinear but found {type(module)}"
            )

        return super()._apply(module, device_mesh)
