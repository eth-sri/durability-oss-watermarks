from transformers import GPT2Config

class GPT2BiasConfig(GPT2Config):
    """
    Custom configuration for GPT2 with an additional bias term in the `lm_head`.
    """
    model_type = "gpt2_bias"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
