import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionLatentUpscalePipeline
from diffusers import StableDiffusionUpscalePipeline


location = "images/init_milad.jpg"
init_image = load_image(location)#.convert("RGB") #Image.open(location)
prompt = "Godlike AI looking into camera, male, dark hair,artificial intelligence, red eyes, cold color palette, digital art, fantasy, artstation, dystopian, highly detailed, sci-fi, cyberware"
negative_prompt = "ugly, deformed, many faces, disfigured, long neck, poor details, bad anatomy"

#Loading models
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)


pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(33)

list_of_result_images = [init_image]
list_of_upscaled_result_images = []
for i in range(1,21):
    image_1 = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=i*0.05, output_type="latent").images[0]#, output_type="latent"
    # image_1.save(f"images/milad_AI_strength_{i*0.05: 0.2f}.png")
    list_of_result_images.append(image_1)

#Clear gpu
del pipe
torch.cuda.empty_cache()

upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
    "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True
)
upscaler = upscaler.to("cuda")
upscaler.enable_attention_slicing()
upscaler.enable_model_cpu_offload()


# super_res = StableDiffusionUpscalePipeline.from_pretrained(
#     "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )

# super_res = super_res.to("cuda")
# super_res.enable_attention_slicing()
# super_res.enable_model_cpu_offload()


for c,image in enumerate (list_of_result_images):
    image_2 = upscaler(prompt=prompt,image=image,num_inference_steps = 20+c, generator=generator, guidance_scale=0).images[0]
    image_2.save(f"images/upscaled_image_{c/20: 0.2f}.png")
    list_of_upscaled_result_images.append(image_2)

make_image_grid(list_of_upscaled_result_images, rows=3, cols=7).save("images/all_images.png")