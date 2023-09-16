# float8_experimental

This is a prototype of a float8 training UX in native PyTorch, with full PT2.0 and distributed support. 
The codebase strives to stay small, easily hackable, and debuggable with native PyTorch tooling.

For now the goal is to move quickly and validate our design by achieving promising e2e training performance/accuracy 
benchmark data on LLaMa 2 70B by the end of 2023H2. Production readiness, backwards compatibility, etc right now is 
an explicit non-goal at this point. Once we are farther along, we will discuss how to make this public.

:warning: **This repository is shared with external partners, so do not post/commit any confidential Meta information here**

# installation

```bash
# install requirements
pip install -r requirements.txt
```

# User API, subject to change

## single GPU

```python
from float8_linear import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)

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
from tp_linear import swap_tp_linear_with_float8_linear

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

* :black_square_button: PT2.0 compatibility of this repository: dynamo
* :black_square_button: PT2.0 compatibility of this repository: aot_autograd
* :black_square_button: PT2.0 compatibility of this repository: inductor
* :black_square_button: e2e benchmarking

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

1. `Float8Linear` owns casting X, W and dL/dY to float8 and does all the bookkeeping of the amax, amax_history and scale buffers
2. user is responsible for applying `Float8Linear` to the right parts of their model with module swaps


## multi GPU

### FSDP with fp16 weight all-gather

No change from single GPU code

### FSDP with fp8 weight all-gather

1. user code is responsible for making the model fp8 aware and adding the right buffers
2. user code is responsible to passing FSDP a data structure with all the information necessary to cast weights to fp8
3. FSDP is responsible for performing the fp8 cast and providing the unsharded fp8 weight to each worker
4. user code is responsible for syncing amax metadata across workers

More details TBD

### TP/SP

`Float8ColumnParallelLinear` and `Float8RowParallelLinear` are replacements for the non-float8 TP/SP primitives.

# code tips

* `float8_experimental/float8_linear.py` - `Float8Linear` (user facing entry point), and custom fw/bw
* `float8_experimental/float8_tensor.py` - `Float8Tensor`, which contains syntactic sugar for passing float8 data + scale around and converting to/from fp8
* `float8_experimental/float8_python_apy.py` - interface between Python functions which know about `Float8Tensor` and aten functions which know about raw data + scale

# testing

```python
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

