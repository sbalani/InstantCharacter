import torch
import random
import numpy as np
import os
from PIL import Image

import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

from pipeline import InstantCharacterFluxPipeline

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# pre-trained weights
ip_adapter_path = hf_hub_download(repo_id="Tencent/InstantCharacter", filename="instantcharacter_ip-adapter.bin")
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
birefnet_path = 'ZhengPeng7/BiRefNet'
makoto_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai", filename="Makoto_Shinkai_style.safetensors")
ghibli_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Ghibli", filename="ghibli_style.safetensors")

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

# init InstantCharacter pipeline
pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
pipe.to(device)

# load InstantCharacter
pipe.init_adapter(
    image_encoder_path=image_encoder_path, 
    image_encoder_2_path=image_encoder_2_path, 
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
)

# load matting model
birefnet = AutoModelForImageSegmentation.from_pretrained(birefnet_path, trust_remote_code=True)
birefnet.to(device)
birefnet.eval()
birefnet_transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def remove_bkg(subject_image):

    def infer_matting(img_pil):
        input_images = birefnet_transform_image(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(img_pil.size)
        mask = np.array(mask)
        mask = mask[..., None]
        return mask

    def get_bbox_from_mask(mask, th=128):
        height, width = mask.shape[:2]
        x1, y1, x2, y2 = 0, 0, width - 1, height - 1

        sample = np.max(mask, axis=0)
        for idx in range(width):
            if sample[idx] >= th:
                x1 = idx
                break
        
        sample = np.max(mask[:, ::-1], axis=0)
        for idx in range(width):
            if sample[idx] >= th:
                x2 = width - 1 - idx
                break

        sample = np.max(mask, axis=1)
        for idx in range(height):
            if sample[idx] >= th:
                y1 = idx
                break

        sample = np.max(mask[::-1], axis=1)
        for idx in range(height):
            if sample[idx] >= th:
                y2 = height - 1 - idx
                break

        x1 = np.clip(x1, 0, width-1).round().astype(np.int32)
        y1 = np.clip(y1, 0, height-1).round().astype(np.int32)
        x2 = np.clip(x2, 0, width-1).round().astype(np.int32)
        y2 = np.clip(y2, 0, height-1).round().astype(np.int32)

        return [x1, y1, x2, y2]

    def pad_to_square(image, pad_value = 255, randomize = False):
        H,W = image.shape[0], image.shape[1]
        if H == W:
            return image

        padd = abs(H - W)
        if randomize:
            padd_1 = int(np.random.randint(0,padd))
        else:
            padd_1 = int(padd / 2)
        padd_2 = padd - padd_1

        if H > W:
            pad_param = ((0,0),(padd_1,padd_2),(0,0))
        else:
            pad_param = ((padd_1,padd_2),(0,0),(0,0))

        image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
        return image

    salient_object_mask = infer_matting(subject_image)[..., 0]
    x1, y1, x2, y2 = get_bbox_from_mask(salient_object_mask)
    subject_image = np.array(subject_image)
    salient_object_mask[salient_object_mask > 128] = 255
    salient_object_mask[salient_object_mask < 128] = 0
    sample_mask = np.concatenate([salient_object_mask[..., None]]*3, axis=2)
    obj_image = sample_mask / 255 * subject_image + (1 - sample_mask / 255) * 255
    crop_obj_image = obj_image[y1:y2, x1:x2]
    crop_pad_obj_image = pad_to_square(crop_obj_image, 255)
    subject_image = Image.fromarray(crop_pad_obj_image.astype(np.uint8))
    return subject_image

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        return random.randint(0, MAX_SEED)
    return int(seed)

def get_example():
    return [
        [
            "assets/girl.jpg",
            "A girl is playing a guitar in street",
            0.9,
            'Makoto Shinkai style',
        ],
        [
            "assets/boy.jpg",
            "A boy is riding a bike in snow",
            0.9,
            'Makoto Shinkai style',
        ],
    ]

def run_for_examples(source_image, prompt, scale, style_mode):
    images = create_image(
        input_image=source_image,
        prompt=prompt,
        scale=scale,
        guidance_scale=3.5,
        num_inference_steps=28,
        seed=123456,
        style_mode=style_mode,
    )
    # Gallery expects list of images
    return images if isinstance(images, list) else [images]

def create_image(
    input_image,
    prompt,
    scale, 
    guidance_scale,
    num_inference_steps,
    seed,
    style_mode=None
):
    if input_image is None:
        return None
    input_image = remove_bkg(input_image)
    # Style handling
    if style_mode == "None" or style_mode is None:
        images = pipe(
            prompt=prompt, 
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            width=1024,
            height=1024,
            subject_image=input_image,
            subject_scale=float(scale),
            generator=torch.manual_seed(int(seed)),
        ).images
    else:
        if style_mode == 'Makoto Shinkai style':
            lora_file_path = makoto_style_lora_path
            trigger = 'Makoto Shinkai style'
        elif style_mode == 'Ghibli style':
            lora_file_path = ghibli_style_lora_path
            trigger = 'ghibli style'
        else:
            return pipe(
                prompt=prompt, 
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                width=1024,
                height=1024,
                subject_image=input_image,
                subject_scale=float(scale),
                generator=torch.manual_seed(int(seed)),
            ).images

        images = pipe.with_style_lora(
            lora_file_path=lora_file_path,
            trigger=trigger,
            prompt=prompt, 
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            width=1024,
            height=1024,
            subject_image=input_image,
            subject_scale=float(scale),
            generator=torch.manual_seed(int(seed)),
        ).images
    # Ensure output is a list (for Gallery)
    return images if isinstance(images, list) else [images]

description = r"""
**InstantCharacter SECourses Improved App V4**  
[https://www.patreon.com/posts/126995127](https://www.patreon.com/posts/126995127)
"""

with gr.Blocks(css="footer {visibility: hidden;}") as block:
    gr.Markdown(description)
    with gr.Row():
        with gr.Column(scale=1):
            image_pil = gr.Image(label="Source Image", type='pil')
            prompt = gr.Textbox(label="Prompt", value="a character is riding a bike in snow")
            scale = gr.Slider(minimum=0, maximum=1.5, step=0.01, value=1.0, label="Scale")
            style_mode = gr.Dropdown(
                label='Style',
                choices=["None", "Makoto Shinkai style", "Ghibli style"],
                value='Makoto Shinkai style'
            )
            with gr.Accordion("Advanced Options", open=False):
                guidance_scale = gr.Slider(minimum=1, maximum=7.0, step=0.01, value=3.5, label="Guidance Scale")
                num_inference_steps = gr.Slider(minimum=5, maximum=50, step=1, value=28, label="Num Inference Steps")
                seed = gr.Number(value=123456, label="Seed Value", precision=0)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            generate_button = gr.Button("Generate Image")
        with gr.Column(scale=1):
            generated_image = gr.Gallery(label="Generated Image", show_label=True, elem_id="gallery").style(grid=(1, 2))
    # Place Examples outside of row/column for proper display
    gr.Examples(
        examples=get_example(),
        inputs=[image_pil, prompt, scale, style_mode],
        outputs=generated_image,
        fn=run_for_examples,
        cache_examples=False,
    )

    # Button logic: first, randomize seed, then generate
    def generate_with_random_seed(
        input_image, prompt, scale, guidance_scale, num_inference_steps, seed, randomize_seed, style_mode
    ):
        new_seed = randomize_seed_fn(seed, randomize_seed)
        images = create_image(
            input_image=input_image,
            prompt=prompt,
            scale=scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=new_seed,
            style_mode=style_mode,
        )
        return images

    generate_button.click(
        generate_with_random_seed,
        inputs=[image_pil, prompt, scale, guidance_scale, num_inference_steps, seed, randomize_seed, style_mode],
        outputs=generated_image,
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run InstantCharacter Gradio app')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    args = parser.parse_args()
    block.queue(max_size=10)
    block.launch(inbrowser=True, share=args.share)