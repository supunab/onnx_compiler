batch_size = 2
hh = 64
ww = 64
dim = 320
embedding_size = 768
prompt_length = 64
input_channels = 4

import torch.nn as nn

def get_token():
    # token_path = "/home/supun/work/hf_token"
    token_path = "/work/hf_token"
    token = ""
    with open(token_path) as f:
        token = f.read()
    return token[:-1]


def get_pt_sd_unet_from_hf() -> nn.Module:
    from diffusers import UNet2DConditionModel
    import torch
    access_token = get_token()
    # pipe = StableDiffusionPipeline.from_pretrained(
    # "CompVis/stable-diffusion-v1-4",
    # revision="fp16",
    # torch_dtype=torch.float16,
    # use_auth_token=access_token,
    # ).to("cuda")
    # return pipe.unet
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        revision="fp16", 
        torch_dtype=torch.float16, 
        use_auth_token=access_token, 
        subfolder="unet")
    unet.cuda()
    unet.eval()
    return unet