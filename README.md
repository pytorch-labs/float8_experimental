# float8_experimental

This is a prototype of a float8 training UX in native PyTorch, with full PT2.0 and distributed support. 
The codebase strives to stay small, easily hackable, and debuggable with native PyTorch tooling.

For now the goal is to move quickly and validate our design by achieving promising e2e training performance/accuracy 
benchmark data on LLaMa 2 70B by the end of 2023H2. Production readiness, backwards compatibility, etc right now is 
an explicit non-goal at this point. Once we are farther along, we will discuss how to make this public.

:warning: **This repository is shared with external partners, so do not post/commit any confidential Meta information here**

# installation

```Shell
pip install .
# Optionally install editable
pip install -e .
```

# User API, subject to change

## single GPU

```python
from float8_experimental.float8_linear import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from float8_experimental.float8_linear_utils import sync_float8_amax_and_scale_history

# create fp32 model
m = Model(...)

# convert all `torch.nn.Linear` modules to `Float8Linear`
swap_linear_with_float8_linear(m)

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

# high level milestone status

## M0: building blocks in core

* :white_check_mark: float8 dtypes in core: `torch.float8_e4m3fn` and `torch.float8_e5m2`
* :white_check_mark: `torch._scaled_mm` in core

## M1: fp8 enabled linear with correct numerics

* :white_check_mark: `Float8Linear` works with real compute and correct numerics, note that performance is a non-goal for this milestone

## M2: single GPU performance

### no tensor subclass branch
* :white_check_mark: PT2.0 compatibility of this repository: dynamo
* :white_check_mark: PT2.0 compatibility of this repository: aot_autograd
* :black_square_button: PT2.0 compatibility of this repository: inductor
* :black_square_button: reach 80% of TransformerEngine performance (current SOTA)
  * current torch._scaled_mm speedup over bf16: 1.9x to 2.5x
  * current pt_float8 speedup over bf16 nn.Linear: 0.6x to 1.1x (optimizations in progress)
  * current te_float8 speedup over bf16 nn.Linear: 1.1x to 1.6x
  * main gap: fused amax+cast+transpose+contiguous kernel (ETA for inductor support is 2023-10-15)
* :black_square_button: match TransformerEngine performance (current SOTA)

### tensor subclass branch
* :black_square_button: PT2.0 compatibility of this repository: dynamo
* :black_square_button: PT2.0 compatibility of this repository: aot_autograd
* :black_square_button: PT2.0 compatibility of this repository: inductor
* :black_square_button: reach 80% of TransformerEngine performance (current SOTA)
* :black_square_button: match TransformerEngine performance (current SOTA)

## M3: distributed

* :white_check_mark: validate FSDP with fp16 weight all-gather still works
* :white_check_mark: TP/SP primitives
* :black_square_button: FSDP with fp8 weight all-gather
* :black_square_button: e2e benchmarking

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

### PT2.0 compatibility and tensor_subclass vs no_tensor_subclass

#### Tensor subclass branch

We are using tensor subclasses (`Float8Tensor`) to write modular code which satisfies
autograd's restriction that `x.dtype == x.grad.dtype`.  The way we achieve this is by
ensuring that instances of `Float8Tensor` set their dtype attribute to the original
dtype (float32/float16/bfloat16) while the underlying data representation is in float8.
If you look in `float8_linear.py` and `te_linear.py`, you will see that we pass instances of `Float8Tensor`
around various `torch.autograd.Function` calls, enabling us to have modular code.

#### No tensor subclass branch

As of 2023-09-16 PT2.0 does not yet support tracing through tensor subclasses.
The ETA of this support is 2023Q4, @bdhirsh is working on it. In the meanwhile, we
have a "no tensor subclass" branch of our single GPU code in `float8_linear_nots.py` 
to enable real performance testing. If you look inside this file, you will notice a single, giant
`torch.autograd.Function`.  This is done in order to hide all the float8 data conversions
from autograd to avoid violating autograd's `x.dtype == x.grad.dtype` restriction.

In the short term, all of our 2023H2 goals can met without tensor subclasses with exception of float8 all-gather FSDP.
However, we plan to start using tensor subclasses as soon as the tracing support lands
to get the code modularity benefits.

Float8 all-gather FSDP will require tensor subclass support because the interface between FSDP and
single GPU code is visible to autograd and has to abide by the `x.dtype == x.grad.dtype restriction`.

Future work such as supporting scaling for float8 output from the matmul (for chaining float8 aware 
ops such as dual gemms) will also need tensor subclasses in order for the implementation to be clean.

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

#### separation of concerns between user code and FSDP

We have alignment on the separation of concerns that we want:
1. user code is responsible for making the model fp8 aware and adding the right buffers
2. user code is responsible to passing FSDP a data structure with all the information necessary to cast weights to fp8: fqns of fp8 enabled weights, and their amax and scale buffers
3. FSDP is responsible for performing the fp8 cast and providing the unsharded fp8 weight to each worker
4. user code is responsible for syncing amax metadata across workers and calculating scales

This way, FSDP knows as little as possible about user logic - it just gets a list of weights + amax buffers + scales, 
and does the float8 fused cast + amax calculation.  User code does everything else.

#### user code interaction with FSDP

We expect this to be trivial. First, when initializing FSDP, we will provide the necessary configuration 
to it as described above. Second, instead of `w_fp8 = cast_to_fp8(w)`, we will just check if `w` is already in fp8.

#### FSDP implementation of fp8 all-gather

This is in early design. The current `FlatParameter` design does not work cleanly with heterogeneous dtypes,
and heterogeneous dtypes are required for a good UX, since for realistic models not all parameters 
(norm parameters, biases, etc) will be in float8.

The hope is that we can refactor FSDP to use per-parameter sharding. @awgu is driving viability studies
on whether this will work, as this will require major changes to FSDP.

# code tips

* `float8_experimental/float8_linear.py` - `Float8Linear` (main user facing entry point)
* `float8_experimental/float8_tensor.py` - `Float8Tensor`, which allows `Float8Linear` to abide by the `x.dtype == x.grad.dtype` restriction
* `float8_experimental/float8_linear_nots.py` - `Float8LinearNoTensorSubclass` (version of `Float8Linear` without tensor subclass, where `torch.compile` works today)
* `float8_experimental/tp_linear.py` - `Float8ColumnParallelLinear` / `Float8RowParallelLinear` (TP/SP versions of float8 linear)

# testing

```bash
# run single-GPU unit tests
python tests/test.py

# run a single-GPU integration test on SAM
python tests/test_sam.py

# run a two-GPU integration test on FSDP
./tests/test_fsdp.sh

# run integration tests for TP/SP
./tests/test_tp.sh

# run all of these tests
./tests/run_everything.sh
```

# benchmarking

```bash
# benchmark the torch._scaled_mm function on LLaMa 2 70B shapes
./benchmarks/bench_matmul.py

# benchmark fw/bw of `Linear`, `Float8Linear` and `te.Linear` on LLaMa 2 70B shapes
# make sure to turn on torch.compile to get the best performance
./benchmarks/bench_linear_float8_nots.py -o ../tmp/test.txt --compile

# dump chrome traces of fw/bw of `Linear`, `Float8Linear` and `te.Linear` on a single shape
./benchmarks/profile_linear_float8_nots.py -o ../tmp/ --compile
```

# License
PyTorch has a BSD 3-Clause License, as found in the LICENSE file.