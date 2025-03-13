import torch
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .configuration_gpt2_bias import GPT2BiasConfig
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union

class GPT2BiasClass(GPT2LMHeadModel):
    """
    Custom GPT2 model that adds a bias term to the `lm_head` output.
    """
    config_class = GPT2BiasConfig

    def __init__(self, config):
        super().__init__(config)
        # Add the custom bias to the `lm_head`
        self.lm_head_bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
        
    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        **kwargs
    ):
        
        model_embeds = super().resize_token_embeddings(new_num_tokens, **kwargs)
        
        # Resize the bias term
        with torch.no_grad():
            old_bias = self.lm_head_bias.data.clone()
            self.lm_head_bias = torch.nn.Parameter(
                torch.zeros(
                    new_num_tokens,
                    device=old_bias.device,
                    dtype=old_bias.dtype
                    )
                )
            self.lm_head_bias.data[:old_bias.size(0)] = old_bias
        
        return model_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # Add the custom bias to the logits
        lm_logits = self.lm_head(hidden_states) + self.lm_head_bias

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )