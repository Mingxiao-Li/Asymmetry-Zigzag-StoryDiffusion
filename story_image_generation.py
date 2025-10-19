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

diffusers.utils.logging.set_verbosity_error()

def load_flux_controller(pipe, device):
    flux_controller = FluxController()
    flux_controller.device = device 
    flux_controller.tokenizer = pipe.tokenizer_2 
    return flux_controller 

def zigzag_image_generation(
    device,
    model_path,
    precision,
    instances,
    spsa_top_percent,
    verbose=True,
    rotate_spsa=True,
    return_image=False,
    num_inference_steps=28,
):  
    print(f"🔥 🔥 🔥 Loading pipe from {model_path}")
    pipe = utils.load_pipe_from_path(
        model_path,
        device,
        torch.float16 if precision == "fp16" else torch.float32,
        variant=precision,
        use_zigzag=True
    )
    
    save_dir, id_prompt, frame_prompt_list, concept_token, seed, window_length = instances[0]
    print(f"🔥 🔥 🔥 Generating images for {id_prompt} >>>>>>>>>>>> ")
    
    generator = torch.Generator().manual_seed(seed)
    flux_controller = load_flux_controller(pipe, device)

    tokenizer = pipe.tokenizer_2 
    id_text_input = tokenizer(
        id_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    concept_token_id = tokenizer(concept_token, add_special_tokens=False).input_ids[0]
    concept_token_index = (id_text_input.input_ids[0] == concept_token_id).nonzero(as_tuple=True)[0][0].item()

    flux_controller.concept_token_index = concept_token_index 
    flux_controller.concept_token = concept_token 
    flux_controller.concept_token_id = concept_token_id 

    if flux_controller.Use_ipca is True:
        flux_controller.Store_qkv = True 
        flux_controller.id_prompt = id_prompt

        flux_controller.Store_subject_qkv = True 
        flux_controller.Use_spsa = True 
        flux_controller.top_percent = spsa_top_percent

        original_prompt_embeds_mode = flux_controller.Prompt_embeds_mode
        flux_controller.Prompt_embeds_mode = "original"
        _ = pipe(prompt=id_prompt,generator=generator, flux_controller=flux_controller, num_inference_steps=num_inference_steps,lambda_step=-1).images
        flux_controller.Prompt_embeds_mode = original_prompt_embeds_mode
    
    flux_controller.Store_qkv = False 
    flux_controller.Store_subject_qkv = False 
    flux_controller.rotate_spsa = rotate_spsa
    max_window_length = utils.get_max_window_length(flux_controller, id_prompt, frame_prompt_list)
    window_length = min(max_window_length, window_length) 

    if window_length < len(frame_prompt_list):
        movement_lists = utils.circular_sliding_windows(frame_prompt_list, window_length)

    else:
        movement_lists = [movement for movement in frame_prompt_list]
    story_images = []
    if verbose:
        print("seed:", seed)
    
    flux_controller.id_prompt = id_prompt 
    for index, movement in enumerate(frame_prompt_list):
        if window_length < len(frame_prompt_list):
            flux_controller.frame_prompt_suppress = movement_lists[index][1:]
            flux_controller.frame_prompt_express = movement_lists[index][0]
            m = " ".join(movement_lists[index])
            gen_prompts = [f"{id_prompt} {m}"]
        else:
            flux_controller.frame_prompt_suppress = movement_lists[:index] + movement_lists[index+1:]
            flux_controller.frame_prompt_express = movement_lists[index]
            m = "".join(movement_lists[index])
            gen_prompts = [f"{id_prompt} {m}"]

        
        if verbose:
            print(f"suppress: {flux_controller.frame_prompt_suppress}")
            print(f"express: {flux_controller.frame_prompt_express}")
            print(f"id_prompt: {id_prompt}")
            print(f"gen_prompts: {gen_prompts}")
       
        if flux_controller is not None and flux_controller.Use_same_init_noise is True:
            generator = torch.Generator().manual_seed(seed)
     
        inv_prompt =[""]
        images = pipe(
            inv_prompt=inv_prompt,
            prompt=gen_prompts,
            generator=generator,
            flux_controller=flux_controller,
            num_inference_steps=num_inference_steps,
        ).images
        story_images.append(images[0])
        if not return_image:
            images[0].save(os.path.join(save_dir, f'{id_prompt} {flux_controller.frame_prompt_express}.jpg'))
    
    if not return_image:
        image_array_list = [np.array(pil_img) for pil_img in story_images]
        story_image = np.concatenate(image_array_list, axis=1)
        story_image = Image.fromarray(story_image.astype(np.uint8))
        story_image.save(os.path.join(save_dir, f'{id_prompt}.jpg'))
        
        print(f"🔥 🔥 🔥 Generated images saved to {save_dir}")
    else:
        return story_images
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux Zigzag")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation (e.g. cuda:0, cpu)")
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-dev")
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
    parser.add_argument("--save_dir", type=str, default="./long_story_result", help="Save directory")
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp16", help="Precision")
    parser.add_argument("--seed", type=int, default=46, help="Random seed")
    parser.add_argument("--window_length", type=int, default=10, help="Window length")
    parser.add_argument("--benchmark_path", type=str, default=None, help="Benchmark path")
    parser.add_argument("--fix_seed", type=int, default=46, help="Fix seed")
    parser.add_argument("--spsa_top_percent", type=float, default=0.2, help="SPSA top percent")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of inference steps")
    args = parser.parse_args()
    
    if args.random_seed:
        args.seed = random.randint(0, 100000)

    current_time =  datetime.now().strftime("%Y%m%d%H")
    current_time_ = datetime.now().strftime("%M%S")
    save_dir = os.path.join(args.save_dir, f"{current_time}/{current_time_}_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    instance = [(save_dir, args.subject_id_prompt, args.prompt_list, args.concept_token, args.seed, args.window_length)]

    print("🚀 🚀 🚀 Start generating images >>>>>>>>>>>> ")
    zigzag_image_generation(
        device=args.device,
        model_path=args.model_path,
        precision=args.precision,
        spsa_top_percent=args.spsa_top_percent,
        instances=instance,
        verbose=True,
        num_inference_steps=args.num_inference_steps,
    )
    