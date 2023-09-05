# float8_playground

This repository is a prototype of a float8 training UX written in native PyTorch. For now the goal is to move quickly and validate our design. Production 
readiness, backwards compatibility, etc right now is an explicit non-goal at this point. Once we are farther along, we will discuss how to make this public.

# installation

```
# install requirements
pip install -r requirements.txt
```

# single GPU user API (not final)

```
from float8_linear import (
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)

# create fp32 model
m = Model(...)

# convert linears to float8
swap_linear_with_float8_linear(m)

# training loop
optimizer.zero_grad()
# specific to float8: separate step to sync scales/amaxes
# in the future, this may move to a context manager
sync_float8_amax_and_scale_history(model)
# run forward
y = m(x)
# run backward
y.sum().backward()
# update weights
optimizer.step()
```

# high level progress tracking

## M0: building blocks in core

* [done] float8 dtypes in core
* [done] torch._scaled_mm in core
* [not started] saturated casts to float8 in core

## M1: fp8 enabled linear with correct numerics

* [done] Float8Linear with emulation and just-in-time scaling
* [done] swap to real fp8 compute
* [done] swap to delayed scaling

Note that performance is a non-goal for this milestone

## M2: single GPU performance

* [in progress] PT2.0 compatibility of this repository: dynamo
* [in progress] PT2.0 compatibility of this repository: aot_autograd
* [in progress] PT2.0 compatibility of this repository: inductor
* [not started] e2e benchmarking

## M3: distributed

* [in progress] validate FSDP with fp16 weight all-gather still works
* [in progress] design for FSDP with fp8 weight all-gather
* [design] implementation for FSDP with fp8 weight all-gather

# high level design

## single GPU

1. `Float8Linear` owns casting inputs/weights/outputs/grads to float8 and keeps track of the relevant buffers
2. user is responsible for applying `Float8Linear` to the right parts of their model with module swaps
3. eager mode performance is a non-goal. PT2.0 graph capture -> inductor graph lowering to fp8 enabled fused kernels is the blessed path for competitive performance.

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

For now, we plan to start with:
1. moving the input/output casts out from `Float8Linear` and in to module hooks
2. asking the user to apply the hooks to the right places in their model, to compose with the activation distributed primitives

More details TBD.



# code tips

* `float8_playground/float8_linear.py` - `Float8Linear` (user facing entry point), and custom fw/bw
* `float8_playground/float8_tensor.py` - `Float8Tensor`, which contains syntactic sugar for passing float8 data + scale around and converting to/from fp8
* `float8_playground/float8_python_apy.py` - interface between Python functions which know about `Float8Tensor` and aten functions which know about raw data + scale

# testing

```
# run single-GPU unit tests
python tests/test.py

# run a single-GPU integration test on SAM
python tests/test_sam.py

# run a two-GPU integration test on FSDP
./tests/test_fsdp.sh

# run all of these tests
./tests/run_everything.sh
```

