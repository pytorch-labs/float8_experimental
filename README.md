# float8_experimental

This is a prototype of a float8 training UX in native PyTorch, with full torch.compile and distributed support.
The codebase strives to stay small, easily hackable, and debuggable with native PyTorch tooling.

:warning: Not all distributed support is implemented yet. See the 
[upcoming feature tracker](https://github.com/pytorch-labs/float8_experimental/issues/187) for details.

:warning: Backwards compatibility is not guaranteed at this point. The codebase is in active development and
will change rapidly.

# installation

```Shell
pip install .

# Optionally install editable
pip install -e .

# Optionally Install dev tooling
pip install -e ".[dev]"
```

# User API, subject to change

We provide two scaling strategies: per-tensor dynamic and delayed.

## float8 linear with dynamic scaling

```python
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
)
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear

# create model
m = Model(...)

# convert all `torch.nn.Linear` modules to `Float8DynamicLinear`
swap_linear_with_float8_linear(m, Float8DynamicLinear)

# optional: use FSDP
model = FSDP(model, use_orig_params=True)

# optional: enable torch.compile for improved performance
m = torch.compile(m)

# train/finetune (not shown)
```

## float8 linear with delayed scaling

```python
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from float8_experimental.float8_linear import Float8Linear

# create model
m = Model(...)

# convert all `torch.nn.Linear` modules to `Float8Linear`
swap_linear_with_float8_linear(m, Float8Linear)

# optional: use FSDP. Note that workarounds gated with config.enable_amax_init and
# config.enable_pre_and_post_forward are needed for autocast+compile+FSDP+float8 to work
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

# code tips

* `float8_experimental/float8_linear.py` - `Float8Linear` (main user facing entry point for delayed scaling)
* `float8_experimental/float8_dynamic_linear.py` - `Float8DynamicLinear` (main user facing entry point for dynamic scaling)
* `float8_experimental/float8_tensor.py` - `Float8Tensor`, which allows `Float8Linear` to abide by the `x.dtype == x.grad.dtype` restriction

# testing

```bash
# run single-GPU unit tests
pytest test/test_base.py

# run a single-GPU integration test on SAM
pytest test/test_sam.py

# run single-GPU compile tests
pytest test/test_compile.py
# run a two-GPU integration test on FSDP
./test/test_fsdp.sh

# run integration tests for TP/SP
./test/test_tp.sh

# run all of these tests
./test/run_everything.sh
```

# benchmarking

```bash
# benchmark the torch._scaled_mm function on LLaMa 2 70B shapes
./benchmarks/bench_matmul.py

# benchmark fw/bw of `Linear`, `Float8Linear` on LLaMa 2 70B shapes
# make sure to turn on torch.compile to get the best performance
./benchmarks/bench_linear_float8.py -o ../tmp/test.txt --compile
```

# License
PyTorch has a BSD 3-Clause License, as found in the LICENSE file.

