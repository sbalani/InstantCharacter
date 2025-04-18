import torch
import random
import numpy as np
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageSegmentation, T5EncoderModel # Added T5EncoderModel
from torchvision import transforms

# Added imports for FP8 quantization
try:
    from optimum.quanto import quantize, qfloat8, freeze
    print("Optimum Quanto found. FP8 quantization will be attempted.")
    fp8_available = True
except ImportError:
    print("Optimum Quanto not found. Install with 'pip install optimum-quanto'. Falling back to default precision.")
    fp8_available = False

from diffusers import FluxTransformer2DModel # Added FluxTransformer2DModel

from pipeline import InstantCharacterFluxPipeline

# global variable
MAX_SEED = np.iinfo(np.int32).max
# Using FP8 primarily targets memory reduction on CUDA, fallback to CPU if no CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
# Keep non-quantized parts in bf16/fp16 on CUDA, fp32 on CPU
dtype = torch.bfloat16 if str(device).__contains__("cuda") else torch.float32
print(f"Using device: {device}, dtype for non-quantized models: {dtype}")

# --- Configuration ---
USE_FP8_QUANTIZATION = fp8_available # Set to False to force fallback
# ** IMPORTANT: Replace with your actual FP8 checkpoint path/URL if different **
fp8_transformer_path = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors"
# --- End Configuration ---


# pre-trained weights
ip_adapter_path = hf_hub_download(repo_id="Tencent/InstantCharacter", filename="instantcharacter_ip-adapter.bin")
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
birefnet_path = 'ZhengPeng7/BiRefNet'
makoto_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai", filename="Makoto_Shinkai_style.safetensors")
ghibli_style_lora_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-LoRA-Ghibli", filename="ghibli_style.safetensors")

# --------------- FP8 Pipeline Initialization ---------------
if USE_FP8_QUANTIZATION:
    print("Initializing InstantCharacter Flux Pipeline with FP8 Quantization...")

    # 1. Load Transformer FP8
    print(f"Loading FP8 Transformer from: {fp8_transformer_path}")
    transformer_fp8 = FluxTransformer2DModel.from_single_file(
        fp8_transformer_path,
        torch_dtype=dtype, # Load initially in higher precision if needed, then quantize
        low_cpu_mem_usage=True, # Use less CPU RAM during load
    )
    print("Quantizing Transformer to FP8...")
    quantize(transformer_fp8, weights=qfloat8)
    freeze(transformer_fp8)
    print("Transformer quantized and frozen.")

    # 2. Load and Quantize Text Encoder 2 (T5) FP8
    print("Loading Text Encoder 2 (T5)...")
    text_encoder_2_fp8 = T5EncoderModel.from_pretrained(
        base_model,
        subfolder="text_encoder_2",
        torch_dtype=dtype, # Load initially in higher precision
        low_cpu_mem_usage=True,
    )
    print("Quantizing Text Encoder 2 to FP8...")
    quantize(text_encoder_2_fp8, weights=qfloat8)
    freeze(text_encoder_2_fp8)
    print("Text Encoder 2 quantized and frozen.")

    # 3. Load Base Pipeline without Transformer and Text Encoder 2
    print("Loading base pipeline components (VAE, Text Encoder 1, Tokenizers)...")
    pipe = InstantCharacterFluxPipeline.from_pretrained(
        base_model,
        transformer=None, # Don't load standard transformer
        text_encoder_2=None, # Don't load standard text encoder 2
        torch_dtype=dtype, # VAE, Text Encoder 1 will use this
        low_cpu_mem_usage=True,
        variant=None, # Avoid bf16/fp16 variants here if loading components manually
    )

    # 4. Assign Quantized Components
    print("Assigning FP8 components to the pipeline...")
    pipe.transformer = transformer_fp8
    pipe.text_encoder_2 = text_encoder_2_fp8

    # 5. Enable Offloading and VAE optimizations
    print("Enabling model CPU offload and VAE optimizations...")
    # pipe.enable_model_cpu_offload() # Preferred for FP8 memory saving
    # If enable_model_cpu_offload causes issues with custom pipeline/LoRA,
    # try moving individual components to device carefully, but this is complex.
    # As a simpler start, let's try moving the whole pipe and see.
    # If OOM occurs, enable_model_cpu_offload() is the way to go.
    print(f"Moving pipeline components to {device}...")
    pipe.to(device) # Try moving all at once first
    # Consider pipe.enable_model_cpu_offload() if pipe.to(device) fails due to memory

    # VAE optimizations might help further
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        print("VAE slicing and tiling enabled.")
    except Exception as e:
        print(f"Could not enable VAE slicing/tiling: {e}")


else:
    # Original BF16/FP16 initialization
    print("Initializing InstantCharacter Flux Pipeline with default precision (BF16/FP16 or FP32)...")
    pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.to(device)

# --------------- End FP8 Initialization Modification ---------------


# load InstantCharacter Adapter System (Common for both FP8 and default)
print("Initializing InstantCharacter Adapter System (Image Encoders, IP-Adapter)...")
pipe.init_adapter(
    image_encoder_path=image_encoder_path,
    image_encoder_2_path=image_encoder_2_path,
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024),
)
print("Adapter system initialized.")

# load matting model (Common for both FP8 and default)
print("Loading Matting Model (BiRefNet)...")
try:
    birefnet = AutoModelForImageSegmentation.from_pretrained(birefnet_path, trust_remote_code=True)
    birefnet.to(device) # Use main device
    birefnet.eval()
    birefnet_transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    matting_model_loaded = True
    print("Matting model loaded successfully.")
except Exception as e:
    print(f"ERROR loading Matting Model: {e}. Background removal will fail.")
    birefnet = None
    matting_model_loaded = False


def remove_bkg(subject_image):
    if not matting_model_loaded or birefnet is None:
        print("Matting model not loaded, skipping background removal.")
        # Pad to square directly without removal
        subject_image_np = np.array(subject_image)
        return Image.fromarray(pad_to_square(subject_image_np, 255).astype(np.uint8))

    def infer_matting(img_pil):
        # Ensure model is on the correct device (might be offloaded)
        birefnet.to(device)
        input_images = birefnet_transform_image(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            # Assuming the model outputs logits, apply sigmoid if needed based on model specifics
            # The original code used [-1].sigmoid(), let's keep that structure
            outputs = birefnet(input_images)
            # Check if output is tuple/list or single tensor
            if isinstance(outputs, (list, tuple)):
                 # Find the segmentation map, often the last element or in 'logits'
                 if hasattr(outputs, 'logits'):
                      preds = outputs.logits
                 else: # Assume last element if no logits attribute
                      preds = outputs[-1]
            else: # Assuming single tensor output (logits or probs)
                 preds = outputs

            # Apply sigmoid if they are logits (common case)
            if torch.min(preds) < 0 or torch.max(preds) > 1: # Heuristic check for logits
                preds = preds.sigmoid()

            preds = preds.cpu() # Move preds to CPU before numpy conversion

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(img_pil.size)
        mask = np.array(mask)
        mask = mask[..., None] # Add channel dim
        return mask

    # Extracted from original remove_bkg for clarity
    def get_bbox_from_mask(mask, th=128):
        if mask.ndim > 2: # Ensure mask is 2D
             mask = np.max(mask, axis=-1)

        height, width = mask.shape[:2]
        rows = np.any(mask > th, axis=1)
        cols = np.any(mask > th, axis=0)

        if not np.any(rows) or not np.any(cols): # Handle empty mask
            print("Warning: Empty mask detected in get_bbox_from_mask.")
            return [0, 0, width - 1, height - 1]

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Clip to bounds just in case
        x1 = np.clip(x1, 0, width-1).round().astype(np.int32)
        y1 = np.clip(y1, 0, height-1).round().astype(np.int32)
        x2 = np.clip(x2, 0, width-1).round().astype(np.int32)
        y2 = np.clip(y2, 0, height-1).round().astype(np.int32)

        # Ensure x1 < x2 and y1 < y2
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        # Ensure minimum size of 1 pixel
        if x1 == x2: x2 = min(x1 + 1, width - 1)
        if y1 == y2: y2 = min(y1 + 1, height - 1)


        return [x1, y1, x2, y2]

    # Extracted from original remove_bkg
    def pad_to_square(image, pad_value = 255, random_padding = False):
        H,W = image.shape[0], image.shape[1]
        if H == W:
            return image

        diff = abs(H - W)
        if random_padding: # Renamed variable for clarity
            pad_1 = int(np.random.randint(0, diff + 1)) # Correct randint upper bound
        else:
            pad_1 = int(diff / 2)
        pad_2 = diff - pad_1

        if H > W: # Pad width
            pad_param = ((0, 0), (pad_1, pad_2), (0, 0))
        else: # Pad height
            pad_param = ((pad_1, pad_2), (0, 0), (0, 0))

        # Ensure image is 3 channels for padding
        if image.ndim == 2:
             image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 4: # Handle RGBA, convert to RGB
             # Simple white background blend
             alpha = image[..., 3:4] / 255.0
             image = image[..., :3] * alpha + pad_value * (1 - alpha)
             image = image.astype(np.uint8)


        padded_image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
        return padded_image

    # Main logic of remove_bkg
    print("Performing background removal...")
    try:
        salient_object_mask = infer_matting(subject_image) # Get mask [h, w, 1]
        x1, y1, x2, y2 = get_bbox_from_mask(salient_object_mask)

        subject_image_np = np.array(subject_image)

        # Ensure mask is binary 0 or 255
        binary_mask = np.where(salient_object_mask > 128, 255, 0).astype(np.uint8)

        # Ensure mask is broadcastable to image (stack if needed)
        if binary_mask.ndim == 2:
            binary_mask = binary_mask[..., None] # Add channel dim if missing
        if binary_mask.shape[2] == 1 and subject_image_np.shape[2] == 3:
             binary_mask = np.concatenate([binary_mask]*3, axis=2)

        # Apply mask: object + white background
        # Ensure shapes match before broadcasting
        if binary_mask.shape != subject_image_np.shape:
             print(f"Warning: Mask shape {binary_mask.shape} mismatch with image shape {subject_image_np.shape}. Attempting broadcast.")
             # Attempt simple broadcast fix if only channel differs (e.g., mask is (H,W,1), image is (H,W,3))
             if binary_mask.shape[:2] == subject_image_np.shape[:2] and binary_mask.shape[2] == 1 and subject_image_np.shape[2] == 3:
                  binary_mask = np.repeat(binary_mask, 3, axis=2)
             else:
                  # More complex mismatch, resize mask? Fallback?
                  print("ERROR: Cannot reconcile mask and image shapes for background removal. Skipping masking.")
                  obj_image = subject_image_np # Fallback to original image
             
        obj_image = (binary_mask / 255.0) * subject_image_np + (1 - (binary_mask / 255.0)) * 255
        obj_image = obj_image.astype(np.uint8)


        # Crop and pad
        crop_obj_image = obj_image[y1:y2, x1:x2]
        # Handle cases where crop is empty (e.g., empty mask)
        if crop_obj_image.size == 0:
             print("Warning: Cropped object image is empty. Returning original image padded.")
             return Image.fromarray(pad_to_square(subject_image_np, 255).astype(np.uint8))

        crop_pad_obj_image = pad_to_square(crop_obj_image, 255)
        result_image = Image.fromarray(crop_pad_obj_image.astype(np.uint8))
        print("Background removal complete.")
        return result_image

    except Exception as e:
        print(f"ERROR during background removal: {e}. Returning original image padded.")
        # Fallback: Pad original image to square
        subject_image_np = np.array(subject_image)
        return Image.fromarray(pad_to_square(subject_image_np, 255).astype(np.uint8))



def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def get_example():
    # Make sure these assets exist or change paths
    case = [
        [
            "./assets/girl.jpg",
            "A girl is playing a guitar in street, Makoto Shinkai style", # Include style in prompt if needed
            0.9,
            'Makoto Shinkai style',
        ],
        [
            "./assets/boy.jpg",
            "A boy is riding a bike in snow, Makoto Shinkai style", # Include style in prompt if needed
            0.9,
            'Makoto Shinkai style',
        ],
    ]
    return case

def run_for_examples(source_image, prompt, scale, style_mode):
    # Ensure example images are loaded correctly if they are paths
    if isinstance(source_image, str):
        try:
            source_image = Image.open(source_image).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Example image not found at {source_image}")
            # Return a placeholder or raise error
            return [Image.new('RGB', (512, 512), color = 'red')]


    return create_image(
        input_image=source_image,
        prompt=prompt,
        scale=scale,
        guidance_scale=3.5,
        num_inference_steps=28, # Use fewer steps for FP8 potentially
        seed=123456,
        style_mode=style_mode,
    )

def create_image(input_image,
                 prompt,
                 scale,
                 guidance_scale,
                 num_inference_steps,
                 seed,
                 style_mode=None):

    if input_image is None:
         gr.Error("Please provide a source image.")
         return [] # Return empty list for gallery

    # Ensure input is PIL Image
    if not isinstance(input_image, Image.Image):
        try:
            # Assuming input_image might be numpy array from Gradio input
            input_image = Image.fromarray(input_image).convert("RGB")
        except Exception as e:
            gr.Error(f"Invalid input image format: {e}")
            return []

    print("Processing input image (background removal)...")
    input_image_processed = remove_bkg(input_image)
    print("Input image processed.")

    # Adjust steps for FP8 if desired (often faster/fewer steps work)
    if USE_FP8_QUANTIZATION:
        # num_inference_steps = max(10, num_inference_steps // 2) # Example: Halve steps for FP8
        print(f"Using {num_inference_steps} steps (Consider adjusting for FP8 performance)")
        pass # Keep user setting for now

    generator = torch.Generator(device=device).manual_seed(seed)

    images = []
    try:
        print(f"Generating image with seed: {seed}, scale: {scale}, guidance: {guidance_scale}, steps: {num_inference_steps}")
        common_args = dict(
            prompt=prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=guidance_scale,
            width=1024,
            height=1024,
            subject_image=input_image_processed,
            subject_scale=scale,
            generator=generator,
        )

        if style_mode is None or style_mode == 'None':
            print("Generating without style LoRA...")
            images = pipe(**common_args).images
        else:
            if style_mode == 'Makoto Shinkai style':
                lora_file_path = makoto_style_lora_path
                trigger = 'Makoto Shinkai style'
            elif style_mode == 'Ghibli style':
                lora_file_path = ghibli_style_lora_path
                trigger = 'ghibli style'
            else:
                 # Fallback if style_mode is unexpected
                 print(f"Unknown style mode '{style_mode}', generating without LoRA.")
                 images = pipe(**common_args).images
                 return images # Exit early if style is invalid

            print(f"Generating with LoRA: {style_mode}...")
            # Note: LoRA application with FP8 might be unstable.
            # Ensure flux_load_lora in models/utils.py (imported by pipeline) handles devices/dtypes correctly.
            images = pipe.with_style_lora(
                lora_file_path=lora_file_path,
                trigger=trigger, # Trigger is added to prompt inside with_style_lora
                **common_args
            ).images
        print("Image generation complete.")

    except RuntimeError as e:
         if "out of memory" in str(e).lower():
             gr.Error(f"CUDA Out of Memory Error. Try reducing image size (if possible), using CPU offloading (if not already), or simplifying the model/task. FP8 helps, but large inputs/models can still exceed memory. Error: {e}")
         else:
             gr.Error(f"Runtime Error during generation: {e}")
         # Clean up memory if possible
         if torch.cuda.is_available():
             torch.cuda.empty_cache()
         return [] # Return empty list on error
    except Exception as e:
         gr.Error(f"An unexpected error occurred: {e}")
         if torch.cuda.is_available():
             torch.cuda.empty_cache()
         return [] # Return empty list on error


    return images

# --- Gradio UI ---
# Description (keep as is)
title = r"""
<h1 align="center">InstantCharacter (FP8 Quantized Demo)</h1>
<p align="center">Personalize Any Characters with a Scalable Diffusion Transformer Framework.<br><strong>NOTE:</strong> Using FP8 Quantization for reduced memory. LoRA/IP-Adapter compatibility may vary.</p>
"""

description = r"""
Upload a source image of a character. The background will be automatically removed (if matting model loads).
Provide a prompt describing the desired scene. Select a style LoRA (optional).
Adjust scale, guidance, steps, and seed as needed.
"""

# Define article (assuming it's defined elsewhere or provide content)
article = """
## Notes on FP8 Version:
*   This demo uses FP8 quantization for the main Transformer and Text Encoder 2 via `optimum-quanto`. This significantly reduces VRAM requirements compared to the original BF16/FP16 model.
*   Image quality might differ slightly from the original model due to quantization.
*   Inference speed may or may not be faster depending on hardware and implementation details.
*   Compatibility with LoRAs and the custom IP-Adapter is attempted but not guaranteed with FP8 models. Results may vary.
*   Ensure `optimum-quanto` is installed: `pip install optimum-quanto`
*   The background removal relies on the `ZhengPeng7/BiRefNet` model. If it fails to load, background removal will be skipped.
*   Check console output for initialization status and potential errors.
"""

block = gr.Blocks(css="footer {visibility: hidden}").queue(max_size=10, api_open=False)
with block:

    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row(): # Main layout Row
        with gr.Column(scale=1): # Input Column
            image_pil = gr.Image(label="Source Image", type='pil', height=300) # Use PIL for consistency
            prompt = gr.Textbox(label="Prompt", value="a character is riding a bike in snow")
            scale = gr.Slider(minimum=0, maximum=1.5, step=0.01, value=1.0, label="Character Scale (Subject Scale)")
            style_mode = gr.Dropdown(label='Style LoRA (Optional)', choices=['None', 'Makoto Shinkai style', 'Ghibli style'], value='Makoto Shinkai style')

            with gr.Accordion(open=False, label="Advanced Options"):
                guidance_scale = gr.Slider(minimum=0.0, maximum=15.0, step=0.1, value=3.5, label="Guidance Scale (CFG)")
                num_inference_steps = gr.Slider(minimum=4, maximum=100, step=1, value=28, label="Num Inference Steps")
                seed = gr.Slider(minimum=0, maximum=MAX_SEED, value=123456, step=1, label="Seed Value")
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            generate_button = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=1): # Output Column
            generated_image = gr.Gallery(label="Generated Image", height=512, object_fit="contain") # Use Gallery

    # Examples Section
    with gr.Row():
         gr.Examples(
             label="Examples (Click to Run)",
             examples=get_example(),
             inputs=[image_pil, prompt, scale, style_mode],
             outputs=[generated_image],
             fn=run_for_examples,
             cache_examples=False, # Caching might be tricky with model state changes
             run_on_click=True,
         )

    gr.Markdown(article) # Display notes

    # Event Handling
    generate_button.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False, # Seed randomization is fast
        api_name=False,
    ).then(
        fn=create_image,
        inputs=[image_pil,
                prompt,
                scale,
                guidance_scale,
                num_inference_steps,
                seed,
                style_mode,
               ],
        outputs=[generated_image]
        # Queueing enabled by default for the block
    )


# Launch the Gradio app
print("Launching Gradio interface...")
block.launch(inbrowser=True, share=False) # Set share=True if you need external access