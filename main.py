import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionLatentUpscalePipeline
from diffusers import StableDiffusionUpscalePipeline


location = "images/init_milad.jpg"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
# upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
#     "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True
# )
super_res = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)

pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()
# upscaler = upscaler.to("cuda")
# upscaler.enable_model_cpu_offload()
super_res = super_res.to("cuda")
super_res.enable_model_cpu_offload()

init_image = load_image(location)#.convert("RGB") #Image.open(location)
prompt = "Godlike AI looking into camera, male, dark hair,artificial intelligence, brown eyes, cold color palette, digital art, fantasy, artstation, dystopian, highly detailed, sci-fi, cyberware, 8k"
negative_prompt = "ugly, deformed, many faces, disfigured, long neck, poor details, bad anatomy"
list_of_result_images = [init_image]
for i in range(1,21):
    image_1 = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=i*0.05).images[0]#, output_type="latent"
    # image_2 = upscaler(prompt=prompt, image=image_1).images[0]
    image_3 = super_res(prompt=prompt, image=image_1).images[0]
    image_3.save(f"images/milad_AI_strength_{i*0.05: 0.2f}.png")

    list_of_result_images.append(image_3)

make_image_grid(list_of_result_images, rows=3, cols=7).save("images/all_images.png")