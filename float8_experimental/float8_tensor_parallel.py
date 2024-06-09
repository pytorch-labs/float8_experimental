import torch.nn as nn
from float8_experimental.float8_dynamic_linear import (
    cast_to_float8_e4m3fn,
    cast_to_float8_e5m2_bw,
)
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, PrepareModuleInput

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

        input_tensor = cast_to_float8_e4m3fn(
            input_tensor, mod.forward_config
        )  # DTensor(Float8Tensor)

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(
                placements=output_layouts, async_op=True
            )  # DTensor(torch.Tensor)

        # fwd noop bwd cast to DTensor(Float8Tensor)
        outputs = cast_to_float8_e5m2_bw(outputs, mod.backward_config)

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

        input_tensor = cast_to_float8_e4m3fn(
            input_tensor, mod.forward_config
        )  # DTensor(Float8Tensor)

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
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)

        # fwd noop bwd cast to DTensor(Float8Tensor)
        outputs = cast_to_float8_e5m2_bw(outputs, mod.backward_config)

        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from float8_experimental.float8_dynamic_linear import Float8DynamicLinear

        if not isinstance(module, Float8DynamicLinear):
            raise ValueError(
                f"Expecting module to be Float8DynamicLinear but found {type(module)}"
            )

        return super()._apply(module, device_mesh)


class PrepareFloat8ModuleInput(PrepareModuleInput):
    # subclass the PrepareModuleInput classes, the only difference is that after we prepare
    # the input DTensor, we cast the input to DTensor(Float8Tensor)
    def _prepare_input_fn(self, inputs, device_mesh):
        if self.input_layouts is None:
            return inputs
        prepared_inputs = []
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if len(inputs) != len(self.input_layouts):
            raise ValueError("module inputs and input_layouts should have same length!")

        assert self.desired_input_layouts is not None, "desired module inputs should not be None!"
        for inp, input_layout, desired_layout in zip(inputs, self.input_layouts, self.desired_input_layouts):
            if input_layout is not None:
                if isinstance(inp, DTensor):
                    # TODO: re-enable the check once we fix the compile path
                    # assert inp.placements[0] == input_layout
                    dt_inp = inp
                else:
                    dt_inp = DTensor.from_local(inp, device_mesh, (input_layout,), run_check=False)

                dt_inp = cast_to_float8_e4m3fn(
                    dt_inp, self.fwd_linear_config
                )  # DTensor(Float8Tensor)
                if desired_layout is not None and input_layout != desired_layout:
                    # i.e. Shard -> Replicate: allgather
                    dt_inp = dt_inp.redistribute(placements=(desired_layout,))
                prepared_inputs.append(dt_inp.to_local() if self.use_local_output else dt_inp)
            else:
                prepared_inputs.append(inp)
        return tuple(prepared_inputs)

    def _prepare_input_kwarg_fn(self, inputs, kwarg_inputs, device_mesh):
        prepared_arg_inputs = self._prepare_input_fn(inputs, device_mesh)
        prepared_kwarg_inputs = {}
        for kwarg_key in kwarg_inputs.keys():
            kwarg_val = kwarg_inputs[kwarg_key]
            input_layout = None
            if kwarg_key in self.input_kwarg_layouts:
                input_layout = self.input_kwarg_layouts[kwarg_key]
                assert isinstance(kwarg_val, torch.Tensor), f"input of key {kwarg_key} to the module should be a Tensor!"
                kwarg_val = DTensor.from_local(kwarg_val, device_mesh, (input_layout,), run_check=False)

                kwarg_val = cast_to_float8_e4m3fn(
                    kwarg_val, self.fwd_linear_config
                )  # DTensor(Float8Tensor)
                if kwarg_key in self.desired_input_kwarg_layouts:
                    desired_layout = self.desired_input_kwarg_layouts[kwarg_key]
                    if desired_layout != input_layout:
                        kwarg_val = kwarg_val.redistribute(placements=(desired_layout,))

                prepared_kwarg_inputs[kwarg_key] = kwarg_val.to_local() if self.use_local_output else kwarg_val
            else:
                prepared_kwarg_inputs[kwarg_key] = kwarg_val

        return (prepared_arg_inputs, prepared_kwarg_inputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
        # search for ScaledMM configs for all the submodules and make sure they are the same
        fwd_linear_config = None
        for mod in module.modules():
            if isinstance(mod, Float8DynamicLinear):
                if fwd_linear_config is None:
                    fwd_linear_config = mod.forward_config
                else:
                    assert fwd_linear_config == mod.forward_config, "All the Float8DynamicLinear modules should have same forward config!"

        self.fwd_linear_config = fwd_linear_config
        super()._apply(module, device_mesh)
        return module
