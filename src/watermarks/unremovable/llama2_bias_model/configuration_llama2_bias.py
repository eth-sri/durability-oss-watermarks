from transformers import LlamaConfig

class Llama2BiasConfig(LlamaConfig):
    """
    Custom configuration for LLaMA 2 with an additional bias term in the `lm_head`.
    """
    model_type = "llama2_bias"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
