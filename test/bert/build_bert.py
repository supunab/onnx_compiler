"""
Build a custom op that wraps the AIT generated code for BERT
------------------------------------------------------------
(Can find more details about args here: https://github.com/huggingface/transformers/blob/aa39967b2898d1056d51ec3b710468ca95773074/src/transformers/models/bert/modeling_bert.py#L812)
Inputs:
    input_ids: [batch_size, seq_length] - input token ids
    token_type_ids: [batch_size, seq_lenght] - to identify the segment tokens belong to. 0 for segment "A", 1 for segment "B"
                                               (used for cases where input_ids corresponds to two [or more?] sentences)
    attention_mask: [batch_size, seq_length] - masks padding as 0

Outputs:
    (embed_size = 768)
    output1: [batch_size, seq_length, embed_size] - embeddings of each token
    output2: take (gather) the first output token embedding of each sentence --> matmul + bias --> tanh 

**inputs and outputs are provided in float32, but all the ops are float16 ops. Therefore, casts are added for inputs and outputs
    to convert them back to/from float32
"""
import onnx
import sys
import os

sys.path.insert(1, os.path.abspath("./../../"))
from converter import optimize_graph

# TODO: attention_mask is ignored at the moment since there's no matching param in AIT Bert
if __name__ == "__main__":
    
    # TODO: temp, just to test optimize_model
    model_path = "/work/models/bert_base/onnx_models/" + "bert_base_cased_3_fp16_gpu.onnx"

    model = onnx.load_model(model_path)
    optimize_graph(model)
    onnx.save_model(model, "test.onnx")
