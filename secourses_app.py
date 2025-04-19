# --- START OF REVISED FILE secourses_app.py ---

import torch
import random
import numpy as np
import os
import time # Added for unique filenames
import platform # Added for opening folder
import subprocess # Added for opening folder
from PIL import Image

import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

# Ensure pipeline module is accessible (e.g., in the same directory or Python path)
try:
    from pipeline import InstantCharacterFluxPipeline
except ImportError:
    print("Error: 'pipeline.py' not found. Please ensure it's in the same directory or your Python path.")
    exit()


# --- Global Variables and Setup ---
MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "outputs" # Define output directory name

# Determine device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
elif torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

# Create output directory
print(f"Ensuring output directory '{OUTPUT_DIR}' exists...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Loading ---
print("Downloading/Loading weights...")
# Use try-except blocks for better error handling during download/load
try:
    ip_adapter_path = hf_hub_download(repo_id="Tencent/InstantCharacter", filename="instantcharacter_ip-adapter.bin")
    base_model = 'black-forest-labs/FLUX.1-dev'
    image_encoder_path = 'google/siglip-so400m-patch14-384'
    image_encoder_2_path = 'facebook/dinov2-giant'
    birefnet_path = 'ZhengPeng7/BiRefNet'
    makoto_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai", filename="Makoto_Shinkai_style.safetensors")
    ghibli_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Ghibli", filename="ghibli_style.safetensors")
except Exception as e:
    print(f"Error downloading or finding model weights: {e}")
    print("Please check your internet connection and Hugging Face Hub access.")
    exit()
print("Finished downloading/loading weights.")

# --- Example Asset Check ---
print("Ensuring 'assets' directory exists...")
os.makedirs("assets", exist_ok=True)
# --- IMPORTANT ---
# Ensure 'assets/girl.jpg' and 'assets/boy.jpg' exist in the 'assets' folder
# for the examples to load correctly.
if not os.path.exists("assets/boy2.jpg"):
    print("Warning: assets/boy2.jpg not found. Examples may fail or be skipped.")
if not os.path.exists("assets/boy.jpg"):
    print("Warning: assets/boy.jpg not found. Examples may fail or be skipped.")
# --- END IMPORTANT ---

# --- Initialize Pipelines and Models ---
print("Initializing InstantCharacter pipeline...")
try:
    pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.to(device)
except Exception as e:
    print(f"Error initializing the main pipeline: {e}")
    exit()
print("Pipeline initialized.")

print("Initializing adapters...")
try:
    pipe.init_adapter(
        image_encoder_path=image_encoder_path,
        image_encoder_2_path=image_encoder_2_path,
        subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024),
    )
except Exception as e:
    print(f"Error initializing adapters: {e}")
    exit()
print("Adapters initialized.")

print("Loading matting model...")
try:
    birefnet = AutoModelForImageSegmentation.from_pretrained(birefnet_path, trust_remote_code=True)
    birefnet.to(device)
    birefnet.eval()
    birefnet_transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
except Exception as e:
    print(f"Error loading matting model: {e}")
    exit()
print("Matting model loaded.")

# --- Helper Functions ---

def remove_bkg(subject_image: Image.Image) -> Image.Image:
    """Removes background, crops, and pads image to square."""
    if subject_image is None:
        raise ValueError("Input image cannot be None for background removal.")
    print("Processing image for background removal...")
    img_pil = subject_image.convert("RGB") # Ensure 3 channels

    # --- Matting Inference ---
    input_images = birefnet_transform_image(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = birefnet(input_images)
        # Handle potential variations in model output structure
        if isinstance(output, (list, tuple)):
            # Common case: logits might be in output.logits or similar
            if hasattr(output, 'logits'):
                preds = output.logits.sigmoid().cpu()
            else: # Fallback: assume last element is segmentation map
                preds = output[-1].sigmoid().cpu()
        elif hasattr(output, 'logits'): # If output is an object with logits
             preds = output.logits.sigmoid().cpu()
        else: # Assume output is directly the tensor we need
             preds = output.sigmoid().cpu() # Adjust if model outputs unnormalized scores

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(img_pil.size)
    mask_np = np.array(mask)
    mask_np = mask_np[..., None] # Add channel dim -> (H, W, 1)

    # --- Bounding Box Calculation ---
    def get_bbox_from_mask(mask_arr, th=128):
        rows, cols = np.where(mask_arr[:, :, 0] >= th)
        if len(rows) == 0 or len(cols) == 0:
            print("Warning: Mask appears empty or below threshold. Using full image bounds.")
            return [0, 0, mask_arr.shape[1] - 1, mask_arr.shape[0] - 1]
        y1, y2 = np.min(rows), np.max(rows)
        x1, x2 = np.min(cols), np.max(cols)
        # Clip to ensure bounds are within image dimensions
        height, width = mask_arr.shape[:2]
        x1 = np.clip(x1, 0, width - 1).round().astype(np.int32)
        y1 = np.clip(y1, 0, height - 1).round().astype(np.int32)
        x2 = np.clip(x2, 0, width - 1).round().astype(np.int32)
        y2 = np.clip(y2, 0, height - 1).round().astype(np.int32)
        return [x1, y1, x2, y2]

    x1, y1, x2, y2 = get_bbox_from_mask(mask_np)

    # Ensure coordinates are valid
    if x1 >= x2 or y1 >= y2:
        print("Warning: Invalid bounding box from mask, padding original image.")
        subject_image_np_orig = np.array(img_pil)
        subject_image_np = pad_to_square(subject_image_np_orig) # Pad original if bbox fails
        return Image.fromarray(subject_image_np.astype(np.uint8))

    # --- Cropping and Compositing ---
    subject_image_np = np.array(img_pil)
    alpha_mask = (mask_np > 128).astype(np.uint8) * 255 # Binary alpha mask
    rgba_image = np.concatenate((subject_image_np, alpha_mask), axis=2)

    crop_rgba_image = rgba_image[y1:y2, x1:x2, :]
    h_crop, w_crop = crop_rgba_image.shape[:2]
    white_bkg = np.ones((h_crop, w_crop, 3), dtype=np.uint8) * 255

    alpha = crop_rgba_image[:, :, 3:] / 255.0
    rgb = crop_rgba_image[:, :, :3]
    composite_image = (rgb * alpha + white_bkg * (1 - alpha)).astype(np.uint8)

    # --- Padding ---
    def pad_to_square(image, pad_value = 255):
        # Expects HWC format numpy array
        if image.ndim != 3 or image.shape[2] != 3:
             print(f"Warning: Unexpected image shape {image.shape} for padding. Trying to proceed.")
             # Attempt to handle grayscale or RGBA conservatively
             if image.ndim == 2: # Grayscale
                 image = np.stack([image]*3, axis=-1)
             elif image.ndim == 3 and image.shape[2] == 4: # RGBA
                 image = image[..., :3] # Use only RGB channels for padding calculation
             elif image.ndim == 3 and image.shape[2] == 1: # Single channel grayscale
                 image = np.concatenate([image]*3, axis=-1)
             else:
                 raise ValueError(f"Cannot pad image with shape {image.shape}")

        H, W, C = image.shape
        if H == W: return image

        diff = abs(H - W)
        pad1, pad2 = diff // 2, diff - (diff // 2)
        pad_width = ((pad1, pad2), (0, 0), (0, 0)) if H < W else ((0, 0), (pad1, pad2), (0, 0))
        padded_image = np.pad(image, pad_width, 'constant', constant_values=pad_value)
        return padded_image

    crop_pad_obj_image = pad_to_square(composite_image, 255)
    subject_image_processed = Image.fromarray(crop_pad_obj_image.astype(np.uint8))
    print("Background removal and processing complete.")
    return subject_image_processed


def get_example():
    """Returns a list of examples for the Gradio interface."""
    # IMPORTANT: Ensure these file paths exist relative to where the script is run
    examples = []
    base_examples = [
        [
            "assets/boy2.jpg", "A man is playing a guitar in street, detailed illustration", 0.9, 'Makoto Shinkai style'
        ],
        [
            "assets/boy.jpg", "A man is riding a bike in snow, cinematic lighting", 0.9, 'Makoto Shinkai style'
        ],
        [
            "assets/boy2.jpg", "A man is reading a book under a large tree, Ghibli style", 1.0, 'Ghibli style'
        ],
        [
            "assets/boy.jpg", "photo of a man holding a camera", 1.1, 'None'
        ],
    ]
    # Check if example files exist before adding them
    for ex in base_examples:
        if os.path.exists(ex[0]):
            examples.append(ex)
        else:
            print(f"Skipping example, file not found: {ex[0]}")
    return examples

def run_for_examples(source_image_path, prompt, scale, style_mode):
    """Wrapper function to run examples."""
    print(f"Running example: {source_image_path}, Prompt: '{prompt}', Scale: {scale}, Style: {style_mode}")
    try:
        input_image_pil = Image.open(source_image_path)
    except FileNotFoundError:
        gr.Warning(f"Example image {source_image_path} not found! Please ensure it exists in the 'assets' folder.")
        return [], 12345 # Return empty list for gallery and default seed
    except Exception as e:
        gr.Error(f"Could not load example image {source_image_path}: {e}")
        return [], 12345

    # Use a fixed seed, guidance, steps, and 1 generation for examples
    example_seed = 12345
    fixed_guidance = 3.5
    fixed_steps = 40 # Use the new default steps
    num_generations = 1
    randomize = False # Don't randomize seed for examples

    # Call the main generation loop function
    generated_images, next_seed = run_generation_loop(
        input_image=input_image_pil,
        prompt=prompt,
        scale=scale,
        guidance_scale=fixed_guidance,
        num_inference_steps=fixed_steps,
        seed=example_seed,
        randomize_seed=randomize,
        style_mode=style_mode,
        num_generations=num_generations,
    )
    print("Example run finished.")
    # Examples only expect the gallery update
    return generated_images


def save_image(image: Image.Image, seed: int, index: int, prompt: str = "output") -> str:
    """Saves a PIL image to the output directory with a unique name."""
    timestamp = int(time.time())
    # Sanitize prompt for filename (basic)
    safe_prompt = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in prompt[:30]).rstrip()
    filename = f"{OUTPUT_DIR}/img_{seed}_{index}_{timestamp}_{safe_prompt}.png"
    try:
        image.save(filename)
        print(f"Saved image: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving image {filename}: {e}")
        return None

def open_folder(folder_path):
    """Opens the specified folder in the default file explorer."""
    print(f"Attempting to open folder: {folder_path}")
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        gr.Warning(f"Output folder '{folder_path}' not found. Generate images first.")
        return

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(os.path.abspath(folder_path)) # Use absolute path
        elif system == "Darwin": # macOS
            subprocess.run(["open", os.path.abspath(folder_path)], check=True)
        else: # Linux and other Unix-like
            subprocess.run(["xdg-open", os.path.abspath(folder_path)], check=True)
        print(f"Opened folder: {folder_path}")
    except FileNotFoundError:
         # This might happen if 'xdg-open' or 'open' isn't available
         gr.Warning(f"Could not automatically open the folder. Please navigate to: {os.path.abspath(folder_path)}")
    except Exception as e:
        gr.Error(f"Failed to open folder: {e}")
        print(f"Error opening folder: {e}")

def run_generation_loop(input_image,
                        prompt,
                        scale,
                        guidance_scale,
                        num_inference_steps,
                        seed,
                        randomize_seed,
                        style_mode,
                        num_generations):
    """Handles the image generation loop, seed management, and saving."""

    if input_image is None:
        gr.Warning("Input image not provided!")
        return [], seed # Return empty list and original seed

    print(f"Starting generation loop: {num_generations} image(s)")
    try:
        processed_image = remove_bkg(input_image)
    except Exception as e:
        print(f"Error during background removal: {e}")
        gr.Error(f"Failed to process input image: {e}")
        return [], seed # Return empty list and original seed

    all_generated_images = []
    current_seed = int(seed) # Ensure seed is integer

    for i in range(int(num_generations)):
        iteration_seed = random.randint(0, MAX_SEED) if randomize_seed else current_seed
        print(f"--- Generation {i+1}/{int(num_generations)} --- Seed: {iteration_seed} ---")

        generator = torch.Generator(device=device).manual_seed(iteration_seed)

        # --- Prepare Pipeline Arguments ---
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

        images_batch = [] # To store images from this iteration
        try:
            if style_mode == "None" or style_mode is None:
                print("Generating image without specific style LoRA...")
                images_batch = pipe(**common_args).images
            else:
                lora_file_path = None
                trigger = None
                if style_mode == 'Makoto Shinkai style':
                    lora_file_path = makoto_style_lora_path
                    trigger = 'Makoto Shinkai style'
                elif style_mode == 'Ghibli style':
                    lora_file_path = ghibli_style_lora_path
                    trigger = 'ghibli style'

                if lora_file_path and trigger:
                    print(f"Generating image with style: {style_mode}")
                    # Add trigger phrase if not already present (case-insensitive check)
                    if trigger.lower() not in common_args["prompt"].lower():
                        final_prompt = f"{common_args['prompt']}, {trigger}"
                        print(f"Adding trigger phrase to prompt: '{trigger}'")
                        common_args["prompt"] = final_prompt

                    images_batch = pipe.with_style_lora(
                        lora_file_path=lora_file_path,
                        trigger=trigger, # Pass trigger if needed by the function internally
                        **common_args
                    ).images
                else:
                    # Fallback if style selected but not configured correctly
                    print(f"Warning: Style '{style_mode}' selected but LoRA path/trigger missing. Generating without style LoRA.")
                    images_batch = pipe(**common_args).images


            print(f"Iteration {i+1} complete. Generated {len(images_batch)} image(s).")

            # --- Save and Collect Images ---
            if isinstance(images_batch, list):
                for idx, img in enumerate(images_batch):
                    saved_path = save_image(img, iteration_seed, idx, prompt)
                    if saved_path: # Only append if save was successful
                         all_generated_images.append(img) # Append PIL image for gallery
            elif isinstance(images_batch, Image.Image): # Handle single image output
                 saved_path = save_image(images_batch, iteration_seed, 0, prompt)
                 if saved_path:
                      all_generated_images.append(images_batch)

        except Exception as e:
            print(f"!!! Error during pipeline execution (Iteration {i+1}): {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            gr.Warning(f"Image generation failed on iteration {i+1}: Check logs for details.")
            # Continue to next iteration if possible, or break if fatal

        # --- Update Seed for Next Iteration ---
        if not randomize_seed:
            current_seed += 1 # Increment seed for the next non-random run


    print(f"--- Generation loop finished. Total images generated: {len(all_generated_images)} ---")

    # Return all collected images and the *next* seed if not randomized
    next_seed = current_seed if not randomize_seed else seed # Return the potentially incremented seed
    return all_generated_images, next_seed


# --- Gradio UI Definition ---
description = r"""
InstantCharacter SECourses Improved App V7 - https://www.patreon.com/posts/126995127
"""

css = """
footer {visibility: hidden;}
.gr-image { min-width: 250px !important; } /* Try to prevent input image squishing */
.gr-gallery { min-height: 400px !important; } /* Ensure gallery has decent height */
"""

print("Building Gradio interface...")
with gr.Blocks(css=css, theme=gr.themes.Soft()) as block:

    gr.Markdown(description)

    with gr.Row():
        # --- Left Column: Inputs ---
        with gr.Column(scale=1):
            image_pil = gr.Image(
                label="Source Character Image",
                type='pil',
                height=640
                # Removed fixed height to allow natural aspect ratio
                # height=400 # <-- Removed this
            )
            generate_button = gr.Button("Generate Image", variant="primary", scale=1) # Moved button up

            prompt = gr.Textbox(
                label="Prompt",
                info="Describe the scene and action.",
                value="a character is riding a bike in snow"
            )

            with gr.Row(): # Character Scale and Artistic Style side-by-side
                scale = gr.Slider(
                    minimum=0.0, maximum=1.5, step=0.01, value=1.0,
                    label="Character Scale",
                    info="Adherence to source (0=ignore, 1.5=max).",
                    scale=1 # Give equal space in the row
                )
                style_mode = gr.Dropdown(
                    label='Artistic Style',
                    choices=["None", 'Makoto Shinkai style', 'Ghibli style'],
                    value='Makoto Shinkai style', # Default style
                    info="Select style LoRA or None.",
                    scale=1 # Give equal space in the row
                )

            num_generations = gr.Number(
                label="Number of Generations",
                value=1, minimum=1, step=1,
                info="How many images to generate in sequence."
            )




        # --- Right Column: Outputs ---
        with gr.Column(scale=1):
            generated_image = gr.Gallery(
                label="Generated Image(s)",
                # height=512, # Auto height might be better with Gallery
                object_fit="contain",
                columns=2, # Show potentially multiple images side-by-side
                preview=True, # Allow clicking image for larger view,
                height=640
            )
            open_folder_button = gr.Button("Open Outputs Folder")
                        # --- Advanced Options ---
            with gr.Accordion("Advanced Options", open=True) as advanced_options: # Open by default
                 with gr.Row(): # CFG and Steps side-by-side
                    guidance_scale = gr.Slider(
                        minimum=1.0, maximum=10.0, step=0.1, value=3.5,
                        label="Guidance Scale (CFG)",
                        info="Prompt guidance strength.",
                        scale=1
                    )
                    num_inference_steps = gr.Slider(
                        minimum=5, maximum=50, step=1, value=40, # Default steps = 40
                        label="Inference Steps",
                        info="More steps = more detail (longer).",
                        scale=1
                    )
                 with gr.Row(): # Seed and Randomize side-by-side
                     seed = gr.Slider(
                         minimum=0, maximum=MAX_SEED, value=random.randint(0, MAX_SEED), step=1,
                         label="Seed",
                         info="Leave random or set for reproducibility.",
                         scale=3 # Give more space to seed slider
                     )
                     randomize_seed = gr.Checkbox(
                         label="Randomize seed",
                         value=True,
                         scale=1 # Give less space to checkbox
                     )


    # --- Examples Section ---
    example_list = get_example()
    if example_list:
        gr.Examples(
            examples=example_list,
            inputs=[image_pil, prompt, scale, style_mode], # Match inputs to the function
            outputs=[generated_image], # Match output components (only gallery needed for examples)
            fn=run_for_examples,
            cache_examples=False, # Re-run examples for consistency if needed
            label="Examples (Click to Run)",
            # run_on_click=True # Default behavior
        )
    else:
        gr.Markdown("_(No example images found in 'assets' folder)_")


    # --- Event Handling ---
    generate_button.click(
        fn=run_generation_loop, # Call the main loop function
        inputs=[
            image_pil,
            prompt,
            scale,
            guidance_scale,
            num_inference_steps,
            seed,
            randomize_seed,
            style_mode,
            num_generations,
        ],
        outputs=[generated_image, seed], # Update gallery and the seed slider
        # api_name="generate_image" # Optional: Define API endpoint name
        show_progress="full" # Show progress for the generation loop
    )

    # Button to open the outputs folder
    open_folder_button.click(
        fn=lambda: open_folder(OUTPUT_DIR), # Call the helper function
        inputs=[], # No inputs needed
        outputs=[] # No Gradio outputs to update
    )

print("Gradio interface built.")

# --- Launch the Application ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run InstantCharacter Gradio app')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing link')
    parser.add_argument('--host', type=str, default=None, help='Host name to bind to (e.g., 0.0.0.0 for public access)')
    parser.add_argument('--port', type=int, default=None, help='Port number to use (default: Gradio chooses)')
    args = parser.parse_args()

    print("Launching Gradio app...")
    block.queue(max_size=10) # Enable queue for handling concurrent requests (adjust size as needed)
    block.launch(
        inbrowser=True,
        share=args.share
        )

# --- END OF REVISED FILE secourses_app.py ---