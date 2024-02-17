import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image, make_image_grid


pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
location = "images/init_milad.jpg"
init_image = load_image(location)#.convert("RGB") #Image.open(location)
# init_image.save("images/init_image.png")
prompt = "metallic Robot looking into camera, male, contemplating, dark hair, brown eyes, cold color palette, digital art, fantasy, dark art, artstation, dystopian, highly detailed, sci-fi, cyberpunk, 8k"
negative_prompt = "ugly, deformed, many faces, disfigured, long neck, poor details, bad anatomy"
list_of_result_images = [init_image]
for i in range(1,21):
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=i*0.05).images
    # make_image_grid([init_image, image], rows=1, cols=2)
    image[0].save(f"images/milad_AI_strength_{i*0.05: 0.2f}.png")
    list_of_result_images.append(image[0])

make_image_grid(list_of_result_images, rows=3, cols=7).save("images/all_images.png")