import torch
from diffusers import FluxPipeline
import os

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

prompts = [
    "A blonde, fair-skinned woman on a beach",
    "A dark-skinned woman in a forest",
]
os.system("rm -rf mask_folder")
os.system("rm mask_nd")
os.system("rm heatmap_nd")
os.system("rm heatmap_bval_nd")

os.system("mkdir mask_folder")

gens = [torch.Generator(device="cuda").manual_seed(11),
        torch.Generator(device="cuda").manual_seed(22)]

out = pipe(
    prompts,
    guidance_scale=0.4,
    generator=gens,
    num_inference_steps=20
)

for i, img in enumerate(out.images):
    img.save(f"flux_batch_{i}.png")

os.system("python grid.py")






