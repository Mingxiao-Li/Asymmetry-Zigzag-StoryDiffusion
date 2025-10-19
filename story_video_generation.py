import os 
import torch 
import random 
import torch.utils 
import flux.utils as utils 
import argparse 
import numpy as np
import diffusers
from datetime import datetime
from flux.flux_controller import FluxController
from PIL import Image
from story_image_generation import zigzag_image_generation
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
import os 
import yaml

def two_step_video_story_generation(
    device,
    image_model_path,
    precision,
    instances,
    spsa_top_percent,
):
    """
    first step: flux asysmmetry zigzag to generation story images 
    second step: use i2v mode to animate the story images
    """

    model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

   

    max_area = 720 * 1280

    for instance in instances:
        save_dir, id_prompt, frame_prompt_list, concept_token, seed, window_length = instance
        story_images = zigzag_image_generation(
        device=device,
        model_path=image_model_path,
        precision=precision,
        instances=[instance],
        spsa_top_percent=spsa_top_percent,
        return_image=True
    )
        for i, image in enumerate(story_images):
            aspect_ratio = image.height / image.width
            mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            image = image.resize((width, height))

            prompt = id_prompt + " " + frame_prompt_list[i]
            negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
            output = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height, width=width,
                num_frames=129,
                num_inference_steps=20, 
                guidance_scale=5.0
            ).frames[0]
            export_to_video(output, f"{save_dir}_{prompt}.mp4", fps=16)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux Zigzag")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation (e.g. cuda:0, cpu)")
    parser.add_argument("--image_model_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--random_seed", action='store_true', help='Use random seed')
    parser.add_argument("--subject_id_prompt",type=str,default="A 3D animation of A black and white dog with yellow collar",help="Subject id prompt")
    parser.add_argument("--prompt_list", nargs="+", type=str, help="Prompt list", default=[
        "wearing a bandana ",
        "on a beach",
        "in a city alley",
        "in a snowy backyard",
        "at a veterinarian's office",
        "playing with a toy",
        "barking at a squirrel",
    ])
    parser.add_argument("--concept_token", type=str, default="dog", help="Concept token")
    parser.add_argument("--save_dir", type=str, default=".", help="Save directory")
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp16", help="Precision")
    parser.add_argument("--seed", type=int, default=46, help="Random seed")
    parser.add_argument("--window_length", type=int, default=10, help="Window length")
    parser.add_argument("--benchmark_path", type=str, default="/data/mingxiao/ZigZag-Story-Diffusion/resource/consistory+.yaml", help="Benchmark path")
    parser.add_argument("--fix_seed", type=int, default=46, help="Fix seed")
    parser.add_argument("--spsa_top_percent", type=float, default=0.2, help="SPSA top percent")

    args = parser.parse_args()
    if args.random_seed:
        args.seed = random.randint(0, 100000)

    current_time =  datetime.now().strftime("%Y%m%d%H")
    current_time_ = datetime.now().strftime("%M%S")
    save_dir = os.path.join(args.save_dir, f"{current_time}/{current_time_}_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    instance = [(save_dir, args.subject_id_prompt, args.prompt_list, args.concept_token, args.seed, args.window_length)]

    print("🚀 🚀 🚀 Start generating videos >>>>>>>>>>>> ")

    two_step_video_story_generation(
        device=args.device,
        image_model_path=args.image_model_path,
        precision=args.precision,
        instances=instance,
        spsa_top_percent=args.spsa_top_percent,
    )