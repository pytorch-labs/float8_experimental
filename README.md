# float8_experimental

This is an early version of a library for accelerating training with float8 in native PyTorch
according to the recipes laid out in https://arxiv.org/pdf/2209.05433.pdf.
The codebase strives to stay small, easily hackable, and debuggable with native PyTorch tooling.
``torch.compile`` is supported out of the box. With ``torch.compile`` on, initial results show
throughput speedups of up to 1.2x on small scale (8 GPUs) LLaMa pretraining jobs.

:warning: <em>See the [feature tracker](https://github.com/pytorch-labs/float8_experimental/issues/187) for upcoming features. Key features such as weight cast recomputation in backward and large scale distributed support are not ready yet. </em>

:warning: <em>Backwards compatibility is not guaranteed at this point. The codebase is in active development and
will change rapidly.</em>

# installation

:warning: <em>For now, use the latest PyTorch nightly for best results with torch.compile.</em>

```Shell
pip install .

# Optionally install editable
pip install -e .

# Optionally Install dev tooling
pip install -e ".[dev]"
```

# User API

We provide two per-tensor scaling strategies: dynamic and delayed.  See https://arxiv.org/pdf/2209.05433.pdf, Section 4.3 for more details. These strategies are configurable separately for activations (`x`), weights (`w`) and gradients (`dL_dY`).

## float8 linear with dynamic scaling for `x`, `w` and `dL_dY`

This is the most accurate recipe as every tensor is scaled dynamically.

```python
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
)
from float8_experimental.fsdp_utils import precompute_float8_dynamic_scale_for_fsdp
from float8_experimental.float8_linear import Float8Linear

# create model
m = Model(...)

# convert all `torch.nn.Linear` modules to `Float8Linear`
swap_linear_with_float8_linear(m, Float8Linear)

# optional: use FSDP
model = FSDP(model, use_orig_params=True)

# optional: enable torch.compile for improved performance
m = torch.compile(m)

# toy training loop
for _ in range(N_ITER):
    optimizer.zero_grad()
    y = m(x)
    y.sum().backward()
    optimizer.step()

    # specific to fsdp2 + dynamic scaling, when fp8 all-gather is turned on
    # this method is optional but is highly recommended for performance
    # it calcuclates scales for all parameters in a single all-reduce
    precompute_float8_dynamic_scale_for_fsdp(model)

```

## float8 linear with delayed scaling

This is theoretically the most performant recipe as it minimizes memory reads.

```python
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from float8_experimental.float8_linear import Float8Linear, TensorScalingType

# create model
m = Model(...)

# convert all `torch.nn.Linear` modules to `Float8Linear`, specifying scaling
# type
swap_linear_with_float8_linear(
    m,
    Float8Linear,
    scaling_type_x=TensorScalingType.DELAYED,
    scaling_type_w=TensorScalingType.DELAYED,
    scaling_type_dL_dY=TensorScalingType.DELAYED,
)

# optional: use FSDP. Note that workarounds gated with config.enable_amax_init and
# config.enable_pre_and_post_forward are needed for autocast + compile + FSDP + float8 to work
from float8_experimental import config
config.enable_amax_init = False  # only needed for autocast + compile + FSDP +  float8 delayed
config.enable_pre_and_post_forward = False  # only needed for autocast + compile + FSDP +  float8 delayed
model = FSDP(model, use_orig_params=True)

# optional: enable torch.compile for improved performance
m = torch.compile(m)

# toy training loop
for _ in range(N_ITER):
    optimizer.zero_grad()
    y = m(x)
    y.sum().backward()

    # specific to float8 with delayed scaling: separate step to sync scales/amaxes
    # in the future, this may move to a context manager
    sync_float8_amax_and_scale_history(model)

    optimizer.step()
```

# 🧭 Code Organization

* `float8_experimental/float8_linear.py`
    - `Float8Linear` (main user facing entry point for Float8Linear)
* `float8_experimental/float8_tensor.py`
    - `Float8Tensor`, which allows `Float8Linear` to abide by the `x.dtype == x.grad.dtype` restriction
    - `ScaledMMConfig` defines the semantics for matmul in the forward and backwards pass

# Testing

```bash
# run single-GPU unit tests
pytest test/test_base.py

# run a single-GPU integration test on SAM
pytest test/test_sam.py

# run single-GPU compile tests
pytest test/test_compile.py
# run a two-GPU integration test on FSDP
./test/test_fsdp.sh

# run integration tests for TP/SP (outdated)
./test/test_tp.sh

# run all of these tests
./test/test_everything.sh
```

# Benchmarking

```bash
# benchmark the torch._scaled_mm function on LLaMa 2 70B shapes
./benchmarks/bench_matmul.py

# benchmark fw/bw of `Linear` and `Float8Linear` on LLaMa 2 70B shapes
# make sure to turn on torch.compile to get the best performance
./benchmarks/bench_linear_float8.py -o ../tmp/test.txt --compile
```

# License
PyTorch has a BSD 3-Clause License, as found in the LICENSE file.
