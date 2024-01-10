# float8_experimental

This is a prototype of a float8 training UX in native PyTorch, with full PT2.0 and distributed support.
The codebase strives to stay small, easily hackable, and debuggable with native PyTorch tooling.

Backwards compatibility is not guaranteed at this point. The codebase is in active development and
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

## single GPU

```python
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from float8_experimental.float8_linear import Float8Linear

# create fp32 model
m = Model(...)

# convert all `torch.nn.Linear` modules to `Float8Linear`
swap_linear_with_float8_linear(m, Float8Linear)

# toy training loop
for _ in range(N_ITER):
    optimizer.zero_grad()
    y = m(x)
    y.sum().backward()

    # specific to float8: separate step to sync scales/amaxes
    # in the future, this may move to a context manager
    sync_float8_amax_and_scale_history(model)

    optimizer.step()
```

## multi GPU

```python
from float8_experimental.tp_linear import swap_tp_linear_with_float8_linear

# swaps the fairscale `ColumnParallelLinear` with `Float8ColumnParallelLinear`,
# and the fairscale `RowParallelLinear` with `Float8RowParallelLinear`
swap_tp_linear_with_float8_linear(model)

# if applicable, enable sequence parallel on the right modules
# TODO make the API for this nicer
model.foo.bar.fc1.sequence_parallel = True
model.foo.bar.fc2.sequence_parallel = True

# the rest of the flow is the same as the single GPU flow
```

# high level technical design

## UX

We are using a module swap UX to keep things simple. If the user model has `torch.nn.Linear` modules or their `fairscale` TP/SP equivalents,
we can convert them to float8. `F.linear`, `torch.mm`, `torch.matmul` are not supported at the moment.

User is responsible for calling the `sync_float8_amax_and_scale_history` function once per fw/bw,
this function updates the amax history. If distributed is enabled, this function also syncs amax values across workers.
This is a separate model level function (as opposed to each module owning the syncing of its buffers) to
make it easier to optimize performance (for example, reduce all the amaxes once in a single tensor instead of doing N reductions).

Composability with `DTensor` is on our radar and we plan to look into this after the manual flow works e2e.

A user facing tensor subclass UX is not being considered at the moment because delayed scaling requires persistent state for
activations, and there isn't a clean and sound way to implement this with tensor subclasses.

## single GPU

### separation of concerns

1. `Float8Linear` owns casting X, W and dL/dY to float8 and does all the bookkeeping of the amax, amax_history and scale buffers
2. user is responsible for applying `Float8Linear` to the right parts of their model with module swaps


### Tensor subclasses

We are using tensor subclasses (`Float8Tensor`) to write modular code which satisfies
autograd's restriction that `x.dtype == x.grad.dtype`.  The way we achieve this is by
ensuring that instances of `Float8Tensor` set their dtype attribute to the original
dtype (float32/float16/bfloat16) while the underlying data representation is in float8.
If you look in `float8_linear.py` and `te_linear.py`, you will see that we pass instances of `Float8Tensor`
around various `torch.autograd.Function` calls, enabling us to have modular code.

## multi GPU

### TP/SP

`Float8ColumnParallelLinear` and `Float8RowParallelLinear` are replacements for the non-float8 TP/SP primitives.

### FSDP with fp16 weight all-gather

No change from single GPU code - it just works.

### FSDP with fp8 weight all-gather

FSDP with fp8 weight-all gather is currently under design.  The problem can be separated into three parts:

a. separation of concerns between user code and FSDP
b. user code interaction with FSDP
c. FSDP implementation of fp8 all-gather

#### Separation of concerns between user code and FSDP

We have alignment on the separation of concerns that we want:
1. user code is responsible for making the model fp8 aware and adding the right buffers
2. user code is responsible to passing FSDP the information necessary to cast weights to fp8: a way to tell if a weight should be cast to fp8, the weight's scale, and the Float8Tensor constructor
3. FSDP is responsible for performing the fp8 cast and providing the unsharded fp8 weight to each worker
4. user code is responsible for syncing amax metadata across workers and calculating scales

This way, FSDP knows as little as possible about user logic - it just gets a list of weights + amax buffers + scales,
and does the float8 fused cast + amax calculation.  User code does everything else.

#### User code interaction with FSDP

We expect this to be trivial. First, when initializing FSDP, we will provide the necessary configuration
to it as described above. Second, instead of `w_fp8 = cast_to_fp8(w)`, we will just check if `w` is already in fp8.

#### FSDP implementation of fp8 all-gather

This is in early design. The current `FlatParameter` design does not work cleanly with heterogeneous dtypes,
and heterogeneous dtypes are required for a good UX, since for realistic models not all parameters
(norm parameters, biases, etc) will be in float8.

We are working on a new FSDP implementation that uses per-parameter sharding that will allow flexible fp8 all-gather. This is being prototyped currently.

# code tips

* `float8_experimental/float8_linear.py` - `Float8Linear` (main user facing entry point for delayed scaling)
* `float8_experimental/float8_dynamic_linear.py` - `Float8DynamicLinear` (main user facing entry point for dynamic scaling)
* `float8_experimental/float8_tensor.py` - `Float8Tensor`, which allows `Float8Linear` to abide by the `x.dtype == x.grad.dtype` restriction
* `float8_experimental/tp_linear.py` - `Float8ColumnParallelLinear` / `Float8RowParallelLinear` (TP/SP versions of float8 linear)

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

# benchmark fw/bw of `Linear`, `Float8Linear` and `te.Linear` on LLaMa 2 70B shapes
# make sure to turn on torch.compile to get the best performance
./benchmarks/bench_linear_float8.py -o ../tmp/test.txt --compile

```

# License
PyTorch has a BSD 3-Clause License, as found in the LICENSE file.

