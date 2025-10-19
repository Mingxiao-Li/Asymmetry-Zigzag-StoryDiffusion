from typing import Optional, Tuple, Dict, Any
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from flux.flux_controller import FluxController
import flux.utils as utils
import torch 
import math 
import torch.nn.functional as F

class FluxAttentionProcessor:
  
    def __call__(
        self,
        attention: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        flux_controller: Optional[FluxController] = None,
    ) -> torch.FloatTensor:
        
        if encoder_hidden_states is not None:
          
            return self._call_flux_transformer_block_attention(
                attention,
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                flux_controller=flux_controller,
            ) 
        else:

            return self._call_flux_single_transformer_block_attention(
                attention,
                hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                flux_controller=flux_controller,
            )


    def _call_flux_transformer_block_attention(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        flux_controller: Optional[FluxController] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape 
        
        # 'sample' projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

      
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # `context` projections
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
        
        if flux_controller is not None and flux_controller.Store_qkv is True and flux_controller.Use_ipca is True:
            utils.ipca2(
                q = encoder_hidden_states_query_proj,
                k = encoder_hidden_states_key_proj,
                v = encoder_hidden_states_value_proj,
                flux_controller=flux_controller,
            )
        elif flux_controller is not None and flux_controller.Store_qkv is False and flux_controller.Use_ipca is True:

            store_query, store_key, store_value = utils.ipca2(
                retrieva_store_states=True,
                flux_controller=flux_controller,
            )
            
            encoder_hidden_states_query_proj = torch.cat([store_query,encoder_hidden_states_query_proj], dim=2)
            encoder_hidden_states_key_proj = torch.cat([store_key, encoder_hidden_states_key_proj ], dim=2)
            encoder_hidden_states_value_proj = torch.cat([store_value,encoder_hidden_states_value_proj], dim=2)
            
            end_of_s_index, dim= store_query.shape[2], store_query.shape[-1]
            story_pos_0, story_pos_1 = torch.ones((end_of_s_index, dim)).to(encoder_hidden_states_query_proj.device), torch.zeros((end_of_s_index, dim)).to(encoder_hidden_states_query_proj.device)
            
            image_rotary_emb_0 = torch.cat([story_pos_0, image_rotary_emb[0]], dim=0)
            image_rotary_emb_1 = torch.cat([story_pos_1, image_rotary_emb[1]], dim=0)
            image_rotary_emb = (image_rotary_emb_0, image_rotary_emb_1)
       
            
        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )   

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) 
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attention_mask + attn_bias
        
        attn_weights = query @ key.transpose(-2, -1) * scale_factor 
        attn_weights = attn_weights + attn_bias
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = torch.dropout(attn_weights, p=0,train=True)
        hidden_states = attn_weights @ value 
        
        if flux_controller is not None and flux_controller.Store_subject_qkv and flux_controller.Use_spsa:
            utils.spsa2(attn_weights=attn_weights, text_length=encoder_hidden_states_query_proj.shape[2], flux_controller=flux_controller, gather_index=True)
       

        if flux_controller is not None and flux_controller.Store_qkv is False and flux_controller.Use_ipca is True:
            hidden_states = hidden_states[:,:, end_of_s_index:,:]   
            

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        return hidden_states, encoder_hidden_states


    def _call_flux_single_transformer_block_attention(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        flux_controller: Optional[FluxController] = None,
    ) -> torch.Tensor:
       
        batch_size, _ , _ = hidden_states.shape 

        # 'sample' projections 
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads 

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        if flux_controller is not None and flux_controller.Store_qkv is True and flux_controller.Use_spsa:
            utils.spsa2(visual_key=key, visual_value=value, visual_length=key.shape[-2]-512,flux_controller=flux_controller)
        
       
        
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        

        if flux_controller is not None and flux_controller.Store_subject_qkv is False and flux_controller.Use_spsa:
            if not flux_controller.rotate_spsa:
                store_key, store_value = utils.spsa2(flux_controller=flux_controller)
                store_key, store_value = store_key.to(query.device), store_value.to(query.device)
                key = torch.cat([store_key, key],dim=2)
                value = torch.cat([store_value, value], dim=2)
            else:
                store_key, store_value = utils.spsa2(flux_controller=flux_controller)
                store_key, store_value = store_key.to(query.device), store_value.to(query.device)
                store_key = apply_rotary_emb(store_key, 
                (image_rotary_emb[0][:store_key.shape[2]], 
                 image_rotary_emb[1][:store_key.shape[2]]))
                
                store_value = apply_rotary_emb(store_value,
                (image_rotary_emb[0][:store_value.shape[2]],
                 image_rotary_emb[1][:store_value.shape[2]]))
                
                key = torch.cat([store_key, key], dim=2)
                value = torch.cat([store_value, value], dim=2)
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states
     

    
    