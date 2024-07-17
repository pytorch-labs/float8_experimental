import torch
import torch.nn as nn
from float8_experimental.float8_dynamic_utils import (
    cast_to_float8_e4m3_dynamic,
    cast_to_float8_e5m2_dynamic_bw,
)
from float8_experimental.float8_linear import TensorScalingType
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
)

# subclass the ColwiseParallel and RowwiseParallel classes
# to add the float8 support
# The parameter sharding stays the same as the core
# ColwiseParallel and RowwiseParallel, the only difference
# here is that in input/output handling we do casting after
# creating the DTensor.

# NOTE: This only works and tested with the dynamic scaling


def _float8_linear_supports_float8_allgather(m):
    # TODO(future): add support for delayed scaling for activations
    # and gradients
    return (
        m.scaling_type_x == TensorScalingType.DYNAMIC
        and m.scaling_type_dL_dY == TensorScalingType.DYNAMIC
    )


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

        input_tensor = cast_to_float8_e4m3_dynamic(
            input_tensor, mod.forward_config, mod.scaling_granularity
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
        outputs = cast_to_float8_e5m2_dynamic_bw(
            outputs, mod.backward_config, mod.scaling_granularity
        )

        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from float8_experimental.float8_linear import Float8Linear

        if not isinstance(module, Float8Linear):
            raise ValueError(
                f"Expecting module to be Float8Linear but found {type(module)}"
            )
        elif isinstance(
            module, Float8Linear
        ) and not _float8_linear_supports_float8_allgather(module):
            raise AssertionError("unsupported")

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

        input_tensor = cast_to_float8_e4m3_dynamic(
            input_tensor, mod.forward_config, mod.scaling_granularity
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
        outputs = cast_to_float8_e5m2_dynamic_bw(
            outputs, mod.backward_config, mod.scaling_granularity
        )

        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from float8_experimental.float8_linear import Float8Linear

        if not isinstance(module, Float8Linear):
            raise ValueError(
                f"Expecting module to be Float8Linear but found {type(module)}"
            )
        elif isinstance(
            module, Float8Linear
        ) and not _float8_linear_supports_float8_allgather(module):
            raise AssertionError("unsupported")

        return super()._apply(module, device_mesh)


class PrepareFloat8ModuleInput(PrepareModuleInput):
    # subclass the PrepareModuleInput classes to implement fp8 specific logic, the only difference is that
    # after we prepare the input DTensor, we cast the input to DTensor(Float8Tensor)
    # This is to ensure the float8 cast happens before the all-gather (i.e. Shard -> Replicate)
    # so that if there are multiple float8 users of the input activation, we perform fp8 allgather
    # only once.
    # FP8 Args:
    #   float8_dtype (torch.dtype, optional): control what float8 dtype to cast to when prepare the module input,
    #       we currently only support torch.float8_e4m3fn. default: torch.float8_e4m3fn
    #   fwd_config_submodule_fqn (str, optional): the fqn of the submodule that contains the forward config 
    #       and scaling_granularity used
    #       for the float8 cast. If not specified, we will search for the Float8Linear in the submodules
    #       and use the forward config from that module, in this case all module's forward config must be
    #       the same.

    def __init__(
        self,
        *,
        input_layouts=None,
        desired_input_layouts=None,
        input_kwarg_layouts=None,
        desired_input_kwarg_layouts=None,
        use_local_output=False,
        float8_dtype=torch.float8_e4m3fn,
        fwd_config_submodule_fqn=None,
    ):
        super().__init__(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_output,
        )

        # fp8 specific fields
        self.float8_dtype = float8_dtype
        self.fwd_config_submodule_fqn = fwd_config_submodule_fqn

        if self.float8_dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                "PrepareFloat8ModuleInput only support casting to float8_e4m3fn for now"
            )

    def _prepare_input_arg(self, input, mesh, input_layout, desired_layout):
        if input_layout is not None:
            if isinstance(input, DTensor):
                # TODO: re-enable the check once we fix the compile path
                # assert inp.placements[0] == input_layout
                dt_inp = input
            else:
                assert isinstance(
                    input, torch.Tensor
                ), "expecting input to be a torch.Tensor!"
                dt_inp = DTensor.from_local(
                    input, mesh, (input_layout,), run_check=False
                )

            dt_inp = cast_to_float8_e4m3_dynamic(
                dt_inp,
                mm_config=self.fwd_linear_config,
                scaling_granularity=self.scaling_granularity,
            )  # DTensor(Float8Tensor)
            if desired_layout is not None and input_layout != desired_layout:
                dt_inp = dt_inp.redistribute(placements=(desired_layout,))

            return dt_inp.to_local() if self.use_local_output else dt_inp
        else:
            return input

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from float8_experimental.float8_linear import Float8Linear

        fwd_linear_config = None
        scaling_granularity = None
        if self.fwd_config_submodule_fqn is not None:
            fwd_linear = module.get_submodule(self.fwd_config_submodule_fqn)
            assert isinstance(fwd_linear, Float8Linear)
            fwd_linear_config = fwd_linear.forward_config
            scaling_granularity = fwd_linear.scaling_granularity
        else:
            # search for ScaledMM configs for all the submodules and make sure they are the same
            for mod in module.modules():
                if isinstance(mod, Float8Linear):
                    if fwd_linear_config is None:
                        fwd_linear_config = mod.forward_config
                        scaling_granularity = mod.scaling_granularity
                    else:
                        assert (
                            fwd_linear_config == mod.forward_config
                        ), "All the Float8Linear modules should have same forward config!"
                        assert (
                            scaling_granularity == mod.scaling_granularity
                        ), "All the Float8DynamicLinear modules should have same scaling granularity!"
        self.fwd_linear_config = fwd_linear_config
        self.scaling_granularity = scaling_granularity
        super()._apply(module, device_mesh)
        return module
