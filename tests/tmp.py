import torch
import context
from float8_linear import Float8Linear
from functorch.compile import aot_module_simplified, min_cut_rematerialization_partition
from functools import partial
from torch.fx.experimental.proxy_tensor import make_fx
import torch.utils._pytree as pytree
from torch._functorch.aot_autograd import create_functional_call

def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g

m = torch.nn.Linear(4, 4, device='cpu', bias=False)
x = torch.randn(4, 4, device='cpu')
m = Float8Linear.from_float(m, emulate=True)
import pdb; pdb.set_trace()
m_compiled = torch.compile(m, backend="eager")
out = m_compiled(x)

params = {
    **dict(m.named_parameters(remove_duplicate=False)),
    **dict(m.named_buffers(remove_duplicate=False)),
}
params_flat, params_spec = pytree.tree_flatten(params)
params_flat = list(params_flat)
params_len = len(params_flat)
functional_call = create_functional_call(m, params_spec, params_len)

full_args = []
full_args.extend(params_flat)
full_args.extend([x])


fw_cell, bw_cell = [None], [None]
m_compiled = aot_module_simplified(
    m,
    (x,),
    fw_compiler=partial(extract_graph, graph_cell=fw_cell),
    bw_compiler=partial(extract_graph, graph_cell=bw_cell),
    partition_fn=min_cut_rematerialization_partition,

)

# verify things don't crash
out = m_compiled(x)
out[0].sum().backward()
print(fw_cell[0].code)
print(bw_cell[0].code)
