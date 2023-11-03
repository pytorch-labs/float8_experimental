# Tests SAM with real weights with float8
# if we want finetuning later, we can use
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb

import copy

import pytest

import torch

from float8_experimental.float8_linear import (
    Float8Linear,
    sync_float8_amax_and_scale_history,
)
from float8_experimental.float8_linear_utils import (
    swap_linear_with_float8_linear,
)
from float8_experimental.float8_utils import compute_error

from transformers import SamModel

torch.manual_seed(0)


class TestFloat8SAMIntegrationTest:
    @pytest.mark.parametrize("data_dtype", [torch.float16, torch.bfloat16])
    def test_encoder_fw_bw(self, data_dtype):
        model = SamModel.from_pretrained("facebook/sam-vit-base").to(data_dtype).cuda()
        # print(model)

        # for now just test the encoder to simplify things
        encoder_ref = model.vision_encoder
        encoder_fp8 = copy.deepcopy(encoder_ref)
        swap_linear_with_float8_linear(encoder_fp8, Float8Linear, emulate=False)

        # an image
        # Note: bsz==4 or a larger power of 2 for this model is needed to
        # ensure all matmuls have arguments with dimensions divisible by 16
        data = torch.randn(4, 3, 1024, 1024).to(data_dtype).cuda()

        encoder_ref_out = encoder_ref(data)
        last_hidden_ref = encoder_ref_out.last_hidden_state
        last_hidden_ref.max().backward()

        sync_float8_amax_and_scale_history(encoder_fp8)
        encoder_fp8_out = encoder_fp8(data)
        last_hidden_fp8 = encoder_fp8_out.last_hidden_state
        last_hidden_fp8.max().backward()

        hidden_sqnr = compute_error(last_hidden_ref, last_hidden_fp8)
        assert hidden_sqnr > 20.0

        ref_name_to_grad = {
            name: param.grad for name, param in encoder_ref.named_parameters()
        }
        sqnr_threshold = 1.0 if data_dtype == torch.float16 else -4
        for name, param in encoder_fp8.named_parameters():
            ref_grad = ref_name_to_grad[name]
            cur_grad = param.grad
            sqnr = compute_error(ref_grad, cur_grad)
            assert sqnr > sqnr_threshold


if __name__ == "__main__":
    pytest.main([__file__])
