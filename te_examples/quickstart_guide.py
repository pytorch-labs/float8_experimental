# copy-paste of
# https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/quickstart.html

import sys

sys.path.insert(0, "/home/vasiliy/local/TransformerEngine/docs/examples")

#
# Let’s build a Transformer layer!
#
print("Section: Let’s build a Transformer layer!")

import quickstart_utils as utils
import torch


class BasicTransformerLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: int = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln1 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps)
        self.qkv_projection = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attention = utils.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        self.projection = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = torch.nn.Dropout(hidden_dropout)
        self.ln2 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps)
        self.mlp = utils.BasicMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.ln1(x)

        # Fused QKV projection
        qkv = self.qkv_projection(x)
        qkv = qkv.view(
            qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels
        )
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout(x)
        x = res + x
        res = x
        x = self.ln2(x)
        x = self.mlp(x)

        return x + res


# Layer configuration
hidden_size = 4096
sequence_length = 2048
batch_size = 4
ffn_hidden_size = 16384
num_attention_heads = 32
dtype = torch.float16

# Synthetic data
x = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)
dy = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)

basic_transformer = BasicTransformerLayer(
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
)
basic_transformer.to(dtype=dtype).cuda()

torch.manual_seed(1234)
y = basic_transformer(x, attention_mask=None)

utils.speedometer(
    basic_transformer,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
)

#
# Meet Transformer Engine
#
print("Section: Meet Transformer Engine")

import transformer_engine.pytorch as te


class BasicTEMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.linear1 = te.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.linear2 = te.Linear(ffn_hidden_size, hidden_size, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.linear2(x)
        return x


class BasicTETransformerLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: int = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln1 = te.LayerNorm(hidden_size, eps=layernorm_eps)
        self.qkv_projection = te.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attention = utils.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        self.projection = te.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = torch.nn.Dropout(hidden_dropout)
        self.ln2 = te.LayerNorm(hidden_size, eps=layernorm_eps)
        self.mlp = BasicTEMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        res = x
        x = self.ln1(x)

        # Fused QKV projection
        qkv = self.qkv_projection(x)
        qkv = qkv.view(
            qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels
        )
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout(x)
        x = res + x
        res = x
        x = self.ln2(x)
        x = self.mlp(x)

        return x + res


basic_te_transformer = BasicTETransformerLayer(
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
)
basic_te_transformer.to(dtype=dtype).cuda()
utils.share_parameters_with_basic_te_model(basic_te_transformer, basic_transformer)

torch.manual_seed(1234)
y = basic_te_transformer(x, attention_mask=None)

utils.speedometer(
    basic_te_transformer,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
)

#
# Fused TE Modules
#
print("Section: Fused TE Modules")


class FusedTETransformerLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: int = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln_qkv = te.LayerNormLinear(
            hidden_size, 3 * hidden_size, eps=layernorm_eps, bias=True
        )
        self.attention = utils.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        self.projection = te.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = torch.nn.Dropout(hidden_dropout)
        self.ln_mlp = te.LayerNormMLP(
            hidden_size, ffn_hidden_size, eps=layernorm_eps, bias=True
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        res = x
        qkv = self.ln_qkv(x)

        # Split qkv into query, key and value
        qkv = qkv.view(
            qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels
        )
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout(x)
        x = res + x
        res = x
        x = self.ln_mlp(x)

        return x + res


fused_te_transformer = FusedTETransformerLayer(
    hidden_size, ffn_hidden_size, num_attention_heads
)
fused_te_transformer.to(dtype=dtype).cuda()
utils.share_parameters_with_fused_te_model(fused_te_transformer, basic_transformer)

torch.manual_seed(1234)
y = fused_te_transformer(x, attention_mask=None)

utils.speedometer(
    fused_te_transformer,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
)

te_transformer = te.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads)
te_transformer.to(dtype=dtype).cuda()
utils.share_parameters_with_transformerlayer_te_model(te_transformer, basic_transformer)

torch.manual_seed(1234)
y = te_transformer(x, attention_mask=None)

utils.speedometer(
    te_transformer,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
)

#
# Enabling FP8
#
# Note: below section currently fails with a segmentation fault, probably related
# to LayerNorm backward: https://gist.github.com/vkuzo/10399ea1e2edb9715b7ce2b10c82dd2a
print("Section: Enabling FP8")
import faulthandler

faulthandler.enable()

from transformer_engine.common.recipe import DelayedScaling, Format

te_transformer = te.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads)
te_transformer.to(dtype=dtype).cuda()
utils.share_parameters_with_transformerlayer_te_model(te_transformer, basic_transformer)

fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(
    fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
)
torch.manual_seed(1234)
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    y = te_transformer(x, attention_mask=None)

utils.speedometer(
    te_transformer,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
    fp8_autocast_kwargs={"enabled": True, "fp8_recipe": fp8_recipe},
)
