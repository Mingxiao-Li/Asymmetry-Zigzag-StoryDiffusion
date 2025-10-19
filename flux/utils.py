from flux.flux_pipeline import FluxPipeline
from flux.zigzag_pipeline import FluxZigzagPipeline
from diffusers.schedulers import  FlowMatchEulerDiscreteScheduler
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from flux.flux_model import FluxTransformer2DModel
from flux.scheduling_flow_match_inverse_euler_discrete import FlowMatchInverseEulerDiscreteScheduler
from scipy.spatial.distance import cdist
from flux.flux_controller import FluxController
from typing import Optional, Tuple
import torch 

def load_pipe_from_path(model_path, device, torch_dtype, variant, use_zigzag=False):

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path,subfolder="scheduler",torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae",torch_dtype=torch_dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer",torch_dtype=torch_dtype)
    tokenizer_2 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_2",torch_dtype=torch_dtype)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",torch_dtype=torch_dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder_2",torch_dtype=torch_dtype)
    flux_transformer = FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer",torch_dtype=torch_dtype)
   
    if not use_zigzag:
        pipe = FluxPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=flux_transformer,
            scheduler=scheduler,
        )

    else:
        inv_scheduer = FlowMatchInverseEulerDiscreteScheduler(
            num_train_timesteps=scheduler.num_train_timesteps,
            shift = scheduler.shift,
            use_dynamic_shifting = True,
            invert_sigmas = False,
        )
        pipe = FluxZigzagPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=flux_transformer,
            scheduler=scheduler,
            inv_scheduler=inv_scheduer,
        )

    pipe.to(device)
    return pipe

def spsa2(attn_weights=None, gather_index=False, text_length=None, visual_length=None, visual_key=None, visual_value=None, flux_controller: Optional[FluxController] = None):

    if flux_controller.spsa_time_step != flux_controller.current_time_step:
        flux_controller.spsa_time_step = flux_controller.current_time_step
        flux_controller.spsa2_index = 0 
    else:
        flux_controller.spsa2_index += 1

    if flux_controller.Store_subject_qkv:
        if  gather_index:
            key = f"transformer {flux_controller.current_time_step}"
            concept_index = flux_controller.concept_token_index
            concept_visual_score = attn_weights[:,:,concept_index, text_length:]
            if key not in flux_controller.subject_attn_weights:
                flux_controller.subject_attn_weights[key] = concept_visual_score
                # print("store --> ",flux_controller.spsa2_index)
            elif key in flux_controller.subject_attn_weights:
                flux_controller.subject_attn_weights[key] += concept_visual_score
                # print("add --> ",flux_controller.spsa2_index)
    
        else:

            key = f"transformer {flux_controller.current_time_step} {flux_controller.current_transformer_position} {flux_controller.spsa2_index}"
            attn_key = f"transformer {flux_controller.current_time_step}"
            num_top_visual_tokens = flux_controller.top_percent * visual_length
            attn_weights = flux_controller.subject_attn_weights[attn_key]
         
            _, top_indices = torch.topk(attn_weights, k=int(num_top_visual_tokens), dim=-1, largest=True, sorted=True)
            visual_index = top_indices.unsqueeze(-1).expand(-1, -1, -1, visual_value.shape[-1])
            store_k = torch.gather(visual_key, dim=2, index=visual_index)
            store_v = torch.gather(visual_value, dim=2, index=visual_index)

            # store_k = torch.cat([visual_key,store_k], dim=2)
            # store_v = torch.cat([visual_value,store_v], dim=2)
            
            flux_controller.subject_k_store[key] = store_k.to("cpu")
            flux_controller.subject_v_store[key] = store_v.to("cpu")
          
            # print("store_representation to key ->", key)
    else:
        
        key_index = 19 + flux_controller.spsa2_index
        key = f"transformer {flux_controller.current_time_step} {flux_controller.current_transformer_position} {key_index}"
        # print("get representation from -> ",key)
        store_key = flux_controller.subject_k_store[key]
        store_value = flux_controller.subject_v_store[key]

        return store_key, store_value


def ipca2(q=None, k=None ,v=None, retrieva_store_states=False,flux_controller: Optional[FluxController] = None):
    # k,v encoder input 
    if flux_controller.ipca_time_step != flux_controller.current_time_step:
        flux_controller.ipca_time_step = flux_controller.current_time_step 
        flux_controller.ipca2_index = 0 
    else:
        flux_controller.ipca2_index += 1 
    
    if flux_controller.Store_qkv is True:

        key = f"transformer {flux_controller.current_time_step} {flux_controller.current_transformer_position} {flux_controller.ipca2_index}"
        
        id_prompt = flux_controller.id_prompt
        end_of_s_index = get_end_s_index(flux_controller.tokenizer, id_prompt)
        flux_controller.k_store[key] = k[:,:,:end_of_s_index+1,:]
        flux_controller.q_store[key] = q[:,:,:end_of_s_index+1,:]
        flux_controller.v_store[key] = v[:,:,:end_of_s_index+1,:]
    
    # if flux_controller.Store_subject_qkv is True:
    #     key = f"transformer {flux_controller.current_time_step} {flux_controller.current_transformer_position} {flux_controller.ipca2_index}"
    #     import ipdb;ipdb.set_trace()

    if retrieva_store_states is True:
        key = f"transformer {flux_controller.current_time_step} {flux_controller.current_transformer_position} {flux_controller.ipca2_index}"
        
        story_query = flux_controller.q_store[key]
        story_key = flux_controller.k_store[key]
        story_value = flux_controller.v_store[key]
     
        return story_query, story_key, story_value
        



def get_end_s_index(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    end_of_s_index = torch.where(text_input_ids == tokenizer.eos_token_id)[1].item()
    return end_of_s_index


def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokens = []
    for text_input_id in text_input_ids[0]:
        token = tokenizer.convert_ids_to_tokens(text_input_id.item())
        tokens.append(token)
    return tokens


def find_sublist_index(list1, list2):
    for i in range(len(list1) - len(list2) + 1):
        if list1[i:i + len(list2)] == list2:
            return i
    return -1 

def punish_weights(tensor, latent_size, alpha=1.0, beta=1.2, calc_similarity=False):
    U, S, Vh = torch.linalg.svd(tensor)
    U = U[:, :latent_size]  
    zero_idx = int(latent_size * alpha)
    
    if calc_similarity:
        _s = S.clone()
        _s *= torch.exp(-alpha * _s) * beta 
        _s[zero_idx:] = 0.0
        _tensor = U @ torch.diag(_s) @ Vh
        dist = cdist(tensor[:,0].unsqueeze(0).cpu(), _tensor[:,0].unsqueeze(0).cpu(), metric='cosine')
        print(f'The distance between the word embedding before and after the punishment: {dist}')
    S *= torch.exp(-alpha*S) * beta
    tensor =U @ torch.diag(S) @ Vh
    return tensor



def swr_single_prompt_embeds(swr_words, prompt_embeds, prompt, tokenizer, alpha=1.0, beta=1.2, zero_eot=False):

    punish_indices = []

    prompt_tokens = prompt2tokens(tokenizer, prompt)
    
    words_tokens = prompt2tokens(tokenizer, swr_words)
    words_tokens = [word for word in words_tokens if word != "<pad>" and word != "</s>"]
    index_of_words = find_sublist_index(prompt_tokens, words_tokens)

    if index_of_words != -1:
        punish_indices.extend([num for num in range(index_of_words, index_of_words + len(words_tokens))])
    
    if zero_eot:
        ept_indices = [index for index,word in enumerate(prompt_tokens) if word == "</s>"]
        prompt_embeds[eot_indices] *= 9e-1
        
    else:
        punish_indices.extend([index for index, word in enumerate(prompt_tokens) if word == '<pad>'])
    
  
    punish_indices = list(set(punish_indices))
    wo_batch = prompt_embeds[punish_indices]
    wo_batch = punish_weights(wo_batch.T.to(float), wo_batch.size(0), alpha=alpha, beta=beta, calc_similarity=False).T.to(prompt_embeds.dtype)

    prompt_embeds[punish_indices] = wo_batch
         

def get_max_window_length(flux_controller: Optional[FluxController],id_prompt, frame_prompt_list):
    single_long_prompt = id_prompt
    max_window_length = 0
    for index, movement in enumerate(frame_prompt_list):
        single_long_prompt += ' ' + movement
        token_length = len(single_long_prompt.split())
        if token_length >= 77:
            break
        max_window_length += 1
    return max_window_length


def circular_sliding_windows(lst, w):
    n = len(lst)
    windows = []
    for i in range(n):
        window = [lst[(i + j) % n] for j in range(w)]
        windows.append(window)
    return windows

def apply_embeds_mask(flux_controller: Optional[FluxController], dropout_mask, add_eot=False):
    id_prompt = flux_controller.id_prompt
    prompt_tokens = prompt2tokens(flux_controller.tokenizer, flux_controller.prompts[0])
    
    words_tokens = prompt2tokens(flux_controller.tokenizer, id_prompt)
    words_tokens = [word for word in words_tokens if word != '<|endoftext|>' and word != '<|startoftext|>']
    index_of_words = find_sublist_index(prompt_tokens,words_tokens)  

    index_list = [index+77 for index in range(index_of_words, index_of_words+len(words_tokens))]
    if add_eot:
        index_list.extend([index+77 for index, word in enumerate(prompt_tokens) if word == '<|endoftext|>'])

    mask_indices = torch.arange(dropout_mask.size(-1), device=dropout_mask.device)
    mask = (mask_indices >= 78) & (~torch.isin(mask_indices, torch.tensor(index_list, device=dropout_mask.device)))
    dropout_mask[0, :, mask] = 0


def gen_dropout_mask(out_shape, flux_controller: Optional[FluxController], drop_out):
    gen_length = out_shape[3]
    attn_map_side_length = out_shape[2]

    batch_num = out_shape[0]
    mask_list = []
    
    for prompt_index in range(batch_num):
        start = prompt_index * int(gen_length / batch_num)
        end = (prompt_index + 1) * int(gen_length / batch_num)
    
        mask = torch.bernoulli(torch.full((attn_map_side_length,gen_length), 1 - drop_out, dtype=unet_controller.torch_dtype, device=unet_controller.device))        
        mask[:, start:end] = 1

        mask_list.append(mask)

    concatenated_mask = torch.stack(mask_list, dim=0)
    return concatenated_mask
