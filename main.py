import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import torch
import random
from diffusers import (AutoPipelineForImage2Image, StableDiffusionControlNetPipeline, ControlNetModel)

def choose_device(torch_device=None):
    print('...Is CUDA available in your computer?', '\n... Yes!' if torch.cuda.is_available() else "\n... No D': ")
    print('...Is MPS available in your computer?', '\n... Yes' if torch.backends.mps.is_available() else "\n... No D':")

    if torch_device is None:
        if torch.cuda.is_available():
            torch_device = "cuda"
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available() and not torch.cuda.is_available():
            torch_device = "mps"
            torch_dtype = torch.float16
        else:
            torch_device = "cpu"
            torch_dtype = torch.float32

    print("......using ", torch_device)

    return torch_device, torch_dtype

DEFAULT_PROMPT = "Above the peak, the sky is lit up with the Northern Lights, featuring vibrant green and purple hues. The stars are visible in the dark sky, adding to the serene and otherworldly atmosphere. The horizon has a warm glow, contrasting beautifully with the cool colors of the aurora and the snowy mountain."
MODEL = "lcm"  # "lcm" or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION = 'models/sd1.5'
LCM_MODEL_LOCATION = 'models/LCM_Dreamshaper_v7'
CONTROLNET_CANNY_LOCATION = "models/control_v11p_sd15_canny"
TORCH_DEVICE, TORCH_DTYPE = choose_device()
GUIDANCE_SCALE = 9  # 0 for sdxl turbo (hardcoded already)
INFERENCE_STEPS = 4  # 4 for lcm (high quality) #2 for turbo
DEFAULT_NOISE_STRENGTH = 0.7  # 0.5 works well too
CONDITIONING_SCALE = 0.7  # 0.5 works well too
GUIDANCE_START = 0.
GUIDANCE_END = 1.
HEIGHT = 864  # 512 #384 #512
WIDTH = 1536  # 512 #384 #512

def prepare_seed():
    random_seed = random.randint(0, 10000)
    generator = torch.manual_seed(random_seed)
    return generator

def convert_numpy_image_to_pil_image(image):
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

def get_result_and_mask(frame, center_x, center_y, width, height):
    "just gets full frame and the mask for cutout"
    
    mask = np.zeros_like(frame)
    mask[center_y:center_y+height, center_x:center_x+width, :] = 255
    cutout = frame[center_y:center_y+height, center_x:center_x+width, :]

    return frame, cutout

def process_lcm(image, lower_threshold=100, upper_threshold=100, aperture=3): 
    image = np.array(image)
    image = cv.Canny(image, lower_threshold, upper_threshold, apertureSize=aperture)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image

def process_sdxlturbo(image):
    return image

def prepare_lcm_controlnet_or_sdxlturbo_pipeline():
    if MODEL == "lcm":
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_LOCATION, torch_dtype=TORCH_DTYPE,
                                                     use_safetensors=True)
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(LCM_MODEL_LOCATION,
                                                                     controlnet=controlnet,
                                                                     torch_dtype=TORCH_DTYPE, safety_checker=None).to(TORCH_DEVICE)
    elif MODEL == "sdxlturbo":
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            SDXLTURBO_MODEL_LOCATION, torch_dtype=TORCH_DTYPE,
            safety_checker=None).to(TORCH_DEVICE)
        
    return pipeline

def run_lcm(pipeline, ref_image):
    generator = prepare_seed()
    gen_image = pipeline(prompt=DEFAULT_PROMPT,
                         num_inference_steps=INFERENCE_STEPS,
                         guidance_scale=GUIDANCE_SCALE,
                         width=WIDTH,
                         height=HEIGHT,
                         generator=generator,
                         image=ref_image,
                         controlnet_conditioning_scale=CONDITIONING_SCALE,
                         control_guidance_start=GUIDANCE_START,
                         control_guidance_end=GUIDANCE_END).images[0]

    return gen_image

def run_sdxlturbo(pipeline, ref_image):
    generator = prepare_seed()
    gen_image = pipeline(prompt=DEFAULT_PROMPT,
                         num_inference_steps=INFERENCE_STEPS,
                         guidance_scale=0.0,
                         width=WIDTH,
                         height=HEIGHT,
                         generator=generator,
                         image=ref_image,
                         strength=DEFAULT_NOISE_STRENGTH).images[0]
                        
    return gen_image

def run_lcm_or_sdxl():
    ###
    ### PREPARE MODELS
    ###
    pipeline = prepare_lcm_controlnet_or_sdxlturbo_pipeline()
    
    processor = process_lcm if MODEL == "lcm" else process_sdxlturbo

    run_model = run_lcm if MODEL == "lcm" else run_sdxlturbo

    ###
    ### RUN DIFFUSION WITH RANDOM BACKGROUND COLOR
    ###
    # Generate a random color background
    background_color = np.random.randint(0, 256, 3, dtype=np.uint8)
    screen = np.full((HEIGHT, WIDTH, 3), background_color, dtype=np.uint8)

    # Calculate the center position for the black and white filter
    center_x = (screen.shape[1] - WIDTH) // 2
    center_y = (screen.shape[0] - HEIGHT) // 2

    result_image, masked_image = get_result_and_mask(screen, center_x, center_y, WIDTH, HEIGHT)

    numpy_image = processor(masked_image)
    pil_image = convert_numpy_image_to_pil_image(numpy_image)
    pil_image = run_model(pipeline, pil_image)

    result_image[center_y:center_y+HEIGHT, center_x:center_x+WIDTH] = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    # Convert the result image to PIL format
    result_pil_image = Image.fromarray(cv.cvtColor(result_image, cv.COLOR_BGR2RGB))
    
    # Convert the PIL image to ImageTk format
    result_tk_image = ImageTk.PhotoImage(result_pil_image)
    
    # Update the Tkinter label with the new image
    label.config(image=result_tk_image)
    label.image = result_tk_image

    root.after(1000, run_lcm_or_sdxl)  # Schedule the next frame generation

# Initialize Tkinter window
root = tk.Tk()
root.attributes("-fullscreen", True)
root.configure(bg='black')

# Label to display the generated images
label = Label(root)
label.pack(expand=True)

# Start the image generation process
root.after(0, run_lcm_or_sdxl)

# Start the Tkinter event loop
root.mainloop()
