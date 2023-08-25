import context

import torch
from torch.utils._pytree import tree_map
from torch.utils._mode_utils import no_dispatch

from float8_tensor import ToFloat8ConstrFunc, Float8Tensor

from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor, Shard

class FooTensor(torch.Tensor):
    """
    This is a tensor subclass similar to `Float8Tensor` which I am using
    to debug why tracing with PT2.0 + eager backend is broken.
    """
    @staticmethod
    def __new__(cls, data, config, scale):
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            # data.size(),
            # strides=data.stride(),
            # storage_offset=data.storage_offset(),
            # dtype=data.dtype,
            # layout=data.layout,
            # requires_grad=data.requires_grad,
            config[0], 
            strides=config[1],
            storage_offset=config[2],
            dtype=config[3],
            layout=config[4],
            requires_grad=config[5],
            device=data.device,
        )
        self._data = data
        self._config = config
        self._scale = scale
        return self

    def __repr__(self):
        # having a real __repr__ masks some useful stack traces, comment it out for now
        # return f"FooTensor(dtype={self._data.dtype}, scale={self._scale}, data={self._data})"
        return f"FooTensor"

    def __tensor_flatten__(self):
        return (self._data,), (self._config, self._scale,)

    @staticmethod
    def __tensor_unflatten__(tensors, metadatas):
        return FooTensor(tensors[0], metadatas[0], metadatas[1])

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        with no_dispatch():
            print('func', func, type(func), types, args, kwargs)

        # DTensor is not doing special handling for clone, and we probably should not either
        if func == torch.ops.aten.clone.default:
            # is this right? ideally this should work without special handling
            return FooTensor(args[0]._data.clone(), args[0]._config, args[0]._scale)
        elif func == torch.ops.aten.view.default:
            # is this right? ideally this should work without special handling
            new_data = args[0]._data.view(*args[1:])
            return FooTensor(new_data, args[0]._config, args[0]._scale)

        raise NotImplementedError()

        # for all ops that get here, for `Float8Tensor` we want to fall back 
        # to original precision, so we unwrap
        def unwrap(t):
            if isinstance(t, FooTensor):
                return t._data
            return t

        args = tree_map(unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(unwrap, kwargs)
        with no_dispatch():
            print('new args', args, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    # Do not force the FooTensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl

class foo_autograd_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x2 = x._data + 1.0
        x3 = FooTensor(x2, x._config, x._scale)
        return x3

    @staticmethod
    def backward(ctx, g):
        return g
        
def run_foo():
    x = torch.zeros(4, 4)
    scale = torch.tensor(1.0)
    orig_dtype = torch.float

    from torch._dynamo import allow_in_graph
    allow_in_graph(FooTensor)

    def foo(x, scale):
        config = (x.size(), x.stride(), x.storage_offset(), x.dtype, x.layout, x.requires_grad)
        x = FooTensor(x, config, scale)
        x = foo_autograd_fn.apply(x)
        return x
        # return (x._data * x._scale) + x._data

    foo = torch.compile(foo, backend='eager')
    print(x)
    y = foo(x, scale)
    print(y)

def run_float8():
    x = torch.ones(4, 4)
    scale = torch.tensor(2.0)
    orig_dtype = torch.float
    config = {
        'size': x.size(),
        'strides': x.stride(),
        'storage_offset': x.storage_offset(),
    }
    # config = (x.size(), x.stride(), x.storage_offset(), x.dtype, x.layout, x.requires_grad)

    x = Float8Tensor(x, scale, x.dtype, config)

    def foo(x):
        x = foo_autograd_fn.apply(x)
        return (x._data * x._scale) + x._data

    foo = torch.compile(foo, backend='eager')
    print(x)
    y = foo(x)
    print(y)

def run_float8_harder():

        
    from torch._dynamo import allow_in_graph
    allow_in_graph(Float8Tensor)

    x = torch.ones(4, 4)
    scale = torch.tensor(2.0)
    orig_dtype = torch.float

    def foo(x, scale, orig_dtype):

        config = {
            'size': x.size(),
            'strides': x.stride(),
            'storage_offset': x.storage_offset(),
        }

        x = Float8Tensor(x, scale, x.dtype, config)
        # return x
        x = foo_autograd_fn.apply(x)
        return (x._data * x._scale) + x._data

    # foo = torch.compile(foo, backend='eager')
    print(x)
    y = foo(x, scale, orig_dtype)
    print(y)
    


def run_dtensor():
    """
    Known good example of tensor subclass + torch.compile with eager 
    backend, we can run this to see how it's supposed to work
    """

    # copied from https://fburl.com/code/5pei3374
    device = 'cpu'
    world_size = 1
    mesh = DeviceMesh(device, torch.arange(world_size))

    # test passing in DTensor as inputs/outputs and run some tensor computation
    def fn(x):
        return x * x + 2

    local_x = torch.randn(4, 4)

    x = DTensor.from_local(local_x, mesh, [Shard(0)], run_check=False)
    opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
    y = opt_fn(x)


# torchrun --nproc_per_node 1 tests/test_dynamo.py
if __name__ == '__main__':
    # known good
    # run_dtensor()
    # print('done with dtensor\n\n')

    # TODO make this work
    # run_foo()

    # run_float8()
    run_float8_harder()
