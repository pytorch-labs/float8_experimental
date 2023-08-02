# Tests SAM with real weights with float8
# if we want finetuning later, we can use
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb

import copy
import unittest

import torch

from transformers import SamModel

# set up float8 path
import context

from float8_linear import swap_linear_with_float8_linear
from float8_utils import compute_error

torch.manual_seed(0)

class Float8SAMIntegrationTest(unittest.TestCase):

    def test_encoder_fw_bw(self):
        model = SamModel.from_pretrained("facebook/sam-vit-base").cuda()
        # print(model)

        # for now just test the encoder to simplify things
        encoder_ref = model.vision_encoder
        encoder_fp8 = copy.deepcopy(encoder_ref)
        swap_linear_with_float8_linear(encoder_fp8)

        # an image 
        data = torch.randn(1, 3, 1024, 1024).cuda()

        encoder_ref_out = encoder_ref(data)
        last_hidden_ref = encoder_ref_out.last_hidden_state
        last_hidden_ref.sum().backward()

        encoder_fp8_out = encoder_fp8(data)
        last_hidden_fp8 = encoder_fp8_out.last_hidden_state
        last_hidden_fp8.sum().backward()

        hidden_sqnr = compute_error(last_hidden_ref, last_hidden_fp8)
        self.assertTrue(hidden_sqnr > 20.0)

        ref_name_to_grad = \
            {name: param.grad for name, param in encoder_ref.named_parameters()}
        for name, param in encoder_fp8.named_parameters():
            ref_grad = ref_name_to_grad[name]
            cur_grad = param.grad
            # For now below is for debugging only, numerical values of
            # fp32 baseline vs fp8 for grads are not that close for a lot
            # of the layers in this network
            sqnr = compute_error(ref_grad, cur_grad)


if __name__ == '__main__':
    unittest.main()
