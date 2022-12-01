"""
Unit test for Attention in GPT2. This is different from BERT for several reasons.
1) GPT2 attention is unidirectional. Meaning that a token can only attend to previous tokens
2) May use previous state as an input instead of computing attention for old tokens again (to prevent recomputing the same thing)
"""

from __future__ import annotations
import sys
import os
import logging

sys.path.insert(1, os.path.abspath("./../../../"))
import click
import numpy as np
from utils import map_np_type_to_onnx
import onnx
import onnxruntime as ort


"""
Past states support
Unfused version : doable
flash attention : not supported in the AITFrontend (it takes a single projected qkv), how about the actual FlashAttention kernel?
mem_eff_attention : need to check, if takes q, k, v separately, so it might be possible?
"""

"""
Unidirectional attention (for sequence length = 1, it doesn't matter)
unfused : doable
flash attention : causal = True? (see ops.flash_attentio(causal=True))
mem_eff_attention : causal = True would do that?
"""





def create_onnx_graph():
    from onnx import helper
    from onnx import TensorProto
    
    dtype = np.float16
    inputs = ["input_hidden_states", "qkv_weight", "qkv_bias", ]

def unfused_ait_attention():
    pass


if __name__ == "__main__":
    pass
