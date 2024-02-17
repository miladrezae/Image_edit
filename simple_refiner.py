import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image, make_image_grid


location = "images/init_milad.jpg"
init_image = load_image(location)
prompt = "wired AI, dark hair, mustache, middle east, linen cloth, connectors, clear pale skin, red cheeks, red eyeline, hostile, takeover, cold color palette, future, artstation, dystopian, highly detailed, sci-fi, cyberpunk"
negative_prompt = "ugly, deformed, many faces, disfigured, long neck, poor details, bad anatomy, uneven, uneven eyes, different eyes, skin wrinkles"

#Loading models
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)


pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(11)

list_of_result_images = [init_image]
for i in range(5):
    image_1 = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=0.35+(i*0.08),num_inference_steps = 30+(i*6)).images[0]
    image_1.save(f"images/milad_AI_strength_{0.5+(i*0.05): 0.2f}.png")
    list_of_result_images.append(image_1)

make_image_grid(list_of_result_images, rows=1, cols=6).save("images/all_images.png")