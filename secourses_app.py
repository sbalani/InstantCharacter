# --- START OF REVISED FILE secourses_app.py ---

import torch
import random
import numpy as np
import os
from PIL import Image

import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

# Ensure pipeline module is accessible (e.g., in the same directory or Python path)
from pipeline import InstantCharacterFluxPipeline

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use bfloat16 if available on CUDA, otherwise float16, fallback to float32 for CPU
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
elif torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")


# pre-trained weights
print("Downloading/Loading weights...")
ip_adapter_path = hf_hub_download(repo_id="Tencent/InstantCharacter", filename="instantcharacter_ip-adapter.bin")
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
birefnet_path = 'ZhengPeng7/BiRefNet'
makoto_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai", filename="Makoto_Shinkai_style.safetensors")
ghibli_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Ghibli", filename="ghibli_style.safetensors")
print("Finished downloading/loading weights.")

# Create assets directory if it doesn't exist
print("Ensuring 'assets' directory exists...")
os.makedirs("assets", exist_ok=True)
# --- IMPORTANT ---
# Ensure 'assets/girl.jpg' and 'assets/boy.jpg' exist in the 'assets' folder
# for the examples to load correctly.
# You might need to download them separately if they are not included.
# Example placeholder check (replace with actual file download/copy if needed):
if not os.path.exists("assets/boy2.jpg"):
    print("Warning: assets/boy2.jpg not found. Examples may fail.")
    # You could add code here to download example images if necessary
if not os.path.exists("assets/boy.jpg"):
    print("Warning: assets/boy.jpg not found. Examples may fail.")
# --- END IMPORTANT ---

# init InstantCharacter pipeline
print("Initializing InstantCharacter pipeline...")
pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=dtype) # Use determined dtype
pipe.to(device)
print("Pipeline initialized.")

# load InstantCharacter adapters
print("Initializing adapters...")
pipe.init_adapter(
    image_encoder_path=image_encoder_path,
    image_encoder_2_path=image_encoder_2_path,
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024),
)
print("Adapters initialized.")

# load matting model
print("Loading matting model...")
birefnet = AutoModelForImageSegmentation.from_pretrained(birefnet_path, trust_remote_code=True)
birefnet.to(device) # Use global device
birefnet.eval()
birefnet_transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("Matting model loaded.")


def remove_bkg(subject_image: Image.Image) -> Image.Image:
    """Removes background, crops, and pads image to square."""
    print("Removing background...")
    img_pil = subject_image.convert("RGB") # Ensure 3 channels

    def infer_matting(img_pil):
        input_images = birefnet_transform_image(img_pil).unsqueeze(0).to(device) # Use global device

        with torch.no_grad():
            # Select the appropriate output tensor if BiRefNet returns multiple
            output = birefnet(input_images)
            # Assuming the segmentation map is the last element, adjust if necessary
            if isinstance(output, (list, tuple)):
                preds = output[-1].sigmoid().cpu()
            else: # Handle cases where output might be directly the logits tensor
                 preds = output.logits.sigmoid().cpu() # Adjust based on actual model output structure

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(img_pil.size)
        mask = np.array(mask)
        mask = mask[..., None] # Add channel dim
        return mask

    def get_bbox_from_mask(mask, th=128):
        # Find non-zero points
        rows, cols = np.where(mask[:, :, 0] >= th)
        if len(rows) == 0 or len(cols) == 0:
            # If mask is empty or below threshold, return full image bounds
            return [0, 0, mask.shape[1] - 1, mask.shape[0] - 1]

        y1 = np.min(rows)
        y2 = np.max(rows)
        x1 = np.min(cols)
        x2 = np.max(cols)

        # Clip to ensure bounds are within image dimensions
        height, width = mask.shape[:2]
        x1 = np.clip(x1, 0, width - 1).round().astype(np.int32)
        y1 = np.clip(y1, 0, height - 1).round().astype(np.int32)
        x2 = np.clip(x2, 0, width - 1).round().astype(np.int32)
        y2 = np.clip(y2, 0, height - 1).round().astype(np.int32)

        return [x1, y1, x2, y2]

    def pad_to_square(image, pad_value = 255):
        H, W = image.shape[0], image.shape[1]
        if H == W:
            return image

        diff = abs(H - W)
        pad1 = diff // 2
        pad2 = diff - pad1

        if H > W:
            pad_width = ((0, 0), (pad1, pad2), (0, 0)) # Pad width
        else:
            pad_width = ((pad1, pad2), (0, 0), (0, 0)) # Pad height

        # Ensure image has 3 channels before padding
        if image.ndim == 2:
             image = np.stack([image]*3, axis=-1) # Convert grayscale to RGB-like
        elif image.shape[2] == 4: # Handle RGBA
             image = image[..., :3] # Drop alpha for padding calculation, apply alpha later?

        # Pad only RGB channels
        padded_image = np.pad(image, pad_width, 'constant', constant_values=pad_value)
        return padded_image

    salient_object_mask = infer_matting(img_pil) # Mask shape (H, W, 1)
    x1, y1, x2, y2 = get_bbox_from_mask(salient_object_mask)

    # Ensure coordinates are valid
    if x1 >= x2 or y1 >= y2:
        print("Warning: Invalid bounding box from mask, using original image.")
        subject_image_np = pad_to_square(np.array(img_pil), 255)
        return Image.fromarray(subject_image_np.astype(np.uint8))

    subject_image_np = np.array(img_pil)
    # Create RGBA image: original image with alpha channel from mask
    alpha_mask = (salient_object_mask > 128).astype(np.uint8) * 255
    rgba_image = np.concatenate((subject_image_np, alpha_mask), axis=2)

    # Crop the RGBA image
    crop_rgba_image = rgba_image[y1:y2, x1:x2, :]

    # Create a white background image of the same size as the crop
    h_crop, w_crop = crop_rgba_image.shape[:2]
    white_bkg = np.ones((h_crop, w_crop, 3), dtype=np.uint8) * 255

    # Alpha blending: composite cropped object onto white background
    alpha = crop_rgba_image[:, :, 3:] / 255.0
    rgb = crop_rgba_image[:, :, :3]
    composite_image = (rgb * alpha + white_bkg * (1 - alpha)).astype(np.uint8)

    # Pad the composite image to square
    crop_pad_obj_image = pad_to_square(composite_image, 255)

    subject_image_processed = Image.fromarray(crop_pad_obj_image.astype(np.uint8))
    print("Background removal complete.")
    return subject_image_processed


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        new_seed = random.randint(0, MAX_SEED)
        print(f"Randomizing seed to: {new_seed}")
        return new_seed
    print(f"Using fixed seed: {seed}")
    return seed

def get_example():
    # IMPORTANT: Ensure these files exist in the 'assets' directory
    return [
        [
            "assets/boy2.jpg",
            "A man is playing a guitar in street, detailed illustration",
            0.9,
            'Makoto Shinkai style',
        ],
        [
            "assets/boy.jpg",
            "A man is riding a bike in snow, cinematic lighting",
            0.9,
            'Makoto Shinkai style',
        ],
         [
            "assets/boy2.jpg", # Example using Ghibli style
            "A man is reading a book under a large tree, Ghibli style",
            1.0,
            'Ghibli style',
        ],
         [
            "assets/boy.jpg", # Example using no style
            "photo of a man holding a camera",
            1.1,
            'None',
        ],
    ]

def run_for_examples(source_image_path, prompt, scale, style_mode):
    print(f"Running example: {source_image_path}, Prompt: '{prompt}', Scale: {scale}, Style: {style_mode}")
    # Load the image from the path for the example
    try:
        input_image_pil = Image.open(source_image_path)
    except FileNotFoundError:
        print(f"Error: Example image not found at {source_image_path}")
        # Return an empty list or raise an error that Gradio can handle
        gr.Warning(f"Example image {source_image_path} not found! Please ensure it exists in the 'assets' folder.")
        return [] # Return empty list for Gallery output
    except Exception as e:
        print(f"Error loading example image {source_image_path}: {e}")
        gr.Error(f"Could not load example image: {e}")
        return []


    # Use a fixed seed for examples for reproducibility
    example_seed = 12345
    fixed_guidance = 3.5
    fixed_steps = 28

    # Call the main generation function
    generated_images = create_image(
        input_image=input_image_pil,
        prompt=prompt,
        scale=scale,
        guidance_scale=fixed_guidance,
        num_inference_steps=fixed_steps,
        seed=example_seed,
        style_mode=style_mode,
    )
    print("Example run finished.")
    return generated_images # Should be a list of PIL Images

def create_image(input_image,
                 prompt,
                 scale,
                 guidance_scale,
                 num_inference_steps,
                 seed,
                 style_mode=None):

    if input_image is None:
        gr.Warning("Input image not provided!")
        return [] # Return empty list for Gallery

    print(f"Starting image generation with seed: {seed}")
    # Process input image (remove background, etc.)
    try:
        processed_image = remove_bkg(input_image)
    except Exception as e:
        print(f"Error during background removal: {e}")
        gr.Error(f"Failed to process input image: {e}")
        return []

    generator = torch.Generator(device=device).manual_seed(int(seed))
    images = [] # Initialize images as empty list

    common_args = dict(
        prompt=prompt,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=guidance_scale,
        width=1024,
        height=1024,
        subject_image=processed_image,
        subject_scale=scale,
        generator=generator,
    )

    try:
        if style_mode == "None" or style_mode is None:
            print("Generating image without specific style LoRA...")
            images = pipe(**common_args).images
        else:
            if style_mode == 'Makoto Shinkai style':
                lora_file_path = makoto_style_lora_path
                trigger = 'Makoto Shinkai style' # Trigger phrase might be needed in prompt too
                print(f"Generating image with style: {style_mode}")
            elif style_mode == 'Ghibli style':
                lora_file_path = ghibli_style_lora_path
                trigger = 'ghibli style' # Trigger phrase might be needed in prompt too
                print(f"Generating image with style: {style_mode}")
            else:
                # Fallback to no style if style_mode is unexpected (shouldn't happen with dropdown)
                print(f"Warning: Unknown style '{style_mode}', generating without style LoRA...")
                images = pipe(**common_args).images
                style_mode = "None" # Set style_mode to None to skip LoRA loading below

            # Only call with_style_lora if a valid style was selected
            if style_mode != "None":
                 # Ensure trigger phrase is included in the prompt if required by the LoRA
                if trigger not in prompt.lower():
                     final_prompt = f"{prompt}, {trigger}"
                     print(f"Adding trigger phrase to prompt: '{trigger}'")
                else:
                     final_prompt = prompt

                common_args["prompt"] = final_prompt # Update prompt in args

                images = pipe.with_style_lora(
                    lora_file_path=lora_file_path,
                    trigger=trigger, # Pass trigger if needed by the function internally
                    **common_args
                ).images

        print(f"Image generation complete. Generated {len(images)} image(s).")

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        gr.Error(f"Image generation failed: {e}")
        return [] # Return empty list on failure

    # Ensure output is always a list
    if not isinstance(images, list):
        return [images]
    return images

# --- Gradio UI ---
description = r""" InstantCharacter SECourses Improved App V6 - https://www.patreon.com/posts/126995127"""

css = """
footer {visibility: hidden;}
/* You can add more custom CSS here if needed */
"""

print("Building Gradio interface...")
with gr.Blocks(css=css) as block:

    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            # Input Components
            image_pil = gr.Image(label="Source Character Image", type='pil', height=400)
            prompt = gr.Textbox(label="Prompt", info="Describe the scene and action.", value="a character is riding a bike in snow")
            scale = gr.Slider(minimum=0.0, maximum=1.5, step=0.01, value=1.0, label="Character Scale", info="How strongly to adhere to the source character (0=ignore, 1.5=max adherence).")
            style_mode = gr.Dropdown(
                label='Artistic Style',
                choices=["None", 'Makoto Shinkai style', 'Ghibli style'],
                value='Makoto Shinkai style', # Default style
                info="Select an artistic style LoRA or None."
            )

            # Use gr.Group for better visual grouping if desired, or keep Accordion
            with gr.Accordion("Advanced Options", open=False) as advanced_options:
                 guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.5, label="Guidance Scale (CFG)", info="How strongly the prompt guides generation.")
                 num_inference_steps = gr.Slider(minimum=5, maximum=50, step=1, value=28, label="Inference Steps", info="More steps take longer but can improve detail.")
                 seed = gr.Slider(minimum=0, maximum=MAX_SEED, value=123456, step=1, label="Seed", info="Set to -1 for random, or keep fixed for reproducibility.")
                 randomize_seed = gr.Checkbox(label="Randomize seed on generate", value=True)


            generate_button = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=1):
            # Output Component
            generated_image = gr.Gallery(label="Generated Image", height=512, object_fit="contain", columns=1) # Show single image better


    # Examples Section
    # Ensure the example file paths are correct relative to script execution dir
    example_list = get_example()
    if example_list: # Only show examples if list is not empty
        gr.Examples(
            examples=example_list,
            inputs=[image_pil, prompt, scale, style_mode], # Match order and type of example data
            outputs=generated_image, # Output component
            fn=run_for_examples,       # Function to run for examples
            cache_examples=False,      # Re-run examples each time
            label="Examples (Click to Run)",
            # run_on_click=True # Explicitly set, though it's the default
        )


    # Event Handling
    generate_button.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=[seed], # Update the seed slider UI
        queue=False, # Run this quickly UI-side
        show_progress="hidden" # Hide progress for this small step
    ).then(
        fn=create_image,
        inputs=[
            image_pil,
            prompt,
            scale,
            guidance_scale,
            num_inference_steps,
            seed, # Use the potentially updated seed value
            style_mode,
        ],
        outputs=[generated_image], # Send result to the gallery
        # api_name="generate_image" # Optional: Define API endpoint name
        show_progress="full" # Show progress for the main generation step
    )

print("Gradio interface built.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run InstantCharacter Gradio app')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing link')
    parser.add_argument('--host', type=str, default=None, help='Host name to bind to (e.g., 0.0.0.0 for public access)')
    parser.add_argument('--port', type=int, default=None, help='Port number to use')
    args = parser.parse_args()

    print("Launching Gradio app...")
    block.queue(max_size=100) # Enable queue for handling concurrent requests
    block.launch(inbrowser=True, share=args.share)

# --- END OF REVISED FILE secourses_app.py ---