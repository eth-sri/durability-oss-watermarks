import torch
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_llama2_bias import Llama2BiasConfig
from typing import Optional, Union, Tuple
class Llama2BiasModel(LlamaForCausalLM):
    """
    Custom LLaMA 2 model that adds a bias term to the `lm_head` output.
    """
    config_class = Llama2BiasConfig

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
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        logits = logits + self.lm_head

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )