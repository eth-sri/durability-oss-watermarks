import torch
import scipy
from math import sqrt
from transformers import LogitsProcessor

from ..watermark_detector import WatermarkDetector


def watermark_model(
    model,
    std: float = 1.0,
    key: int = 0,
    device: str = "cpu",
    use_kgw: bool = False,
    ):
    """Assumes a custom model that has a lm_head_bias parameter (see implementation in GPT2BiasClass)."""
    
    assert hasattr(model, "lm_head_bias"), "Model must have a custom bias parameter"

    vocab_size = model.config.vocab_size
    torch.manual_seed(key)
    delta_wm = torch.normal(mean=0.0, std=std, size=(vocab_size,), device=device)
    if use_kgw:
        zeros = torch.zeros_like(delta_wm)
        delta_wm = zeros
        delta_wm[torch.randperm(vocab_size)[:int(vocab_size * 0.25)]] = 2
    
    # Match dtype to model
    delta_wm = delta_wm.type(model.dtype)
    
    model.lm_head_bias.data = delta_wm
    
    return model
    
class LogitProcessorUnremovable(LogitsProcessor):
    
    def __init__(self, watermark_detector):
        self.delta_wm = watermark_detector.delta_wm
        
    def __call__(self, input_ids, logits):
        return logits + self.delta_wm

class WatermarkDetectorUnremovable(WatermarkDetector):
    
    def __init__(
        self,
        tokenizer,
        std: float = 1.0,
        key: int = 0,
        wm_device: str = "cpu",
        ignore_repeated_ngrams: bool = False,
        use_kgw: bool = False,
        ):

        self.device = wm_device
        self.key = key
        self.std = std
        
            
        # also configure the metrics returned/preprocessing options
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer

        vocab_size = len(tokenizer)
        torch.manual_seed(key)
        self.delta_wm = torch.normal(mean=0.0, std=std, size=(vocab_size,), device=self.device)
        if use_kgw:
            zeros = torch.zeros_like(self.delta_wm)
            self.delta_wm = zeros
            self.delta_wm[torch.randperm(vocab_size)[:int(vocab_size * 0.25)]] = 2
        self.wm_std = std

        self.ignore_repeated_ngrams = ignore_repeated_ngrams
        
    def detect(
        self,
        input_ids: torch.LongTensor, 
        attention_mask: torch.FloatTensor = None
    ) -> torch.LongTensor:
        
        pvalues = []
        
        input_ids_device = input_ids.device
        input_ids = input_ids.to(self.device)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)
        
        for tokenized_text, mask in zip(input_ids, attention_mask):
        
            tokenized_text = tokenized_text[mask.bool()]
        
            # Remove repeated tokens
            tokenized_text = torch.unique(tokenized_text)
            scores = self.delta_wm[tokenized_text]
            z_score = (torch.sum(scores) - scores.shape[0]*torch.mean(self.delta_wm)) / (sqrt(scores.shape[0]) * self.wm_std)
            # One-sided test
            p_value = 1 - scipy.stats.norm.cdf(z_score.item())
            
            pvalues.append(p_value)
            
        pvalues = torch.tensor(pvalues).to(input_ids_device)
            
        return pvalues

    def get_config(self):
        config = {
            "type": "unremovable",
            "key": self.key,
            "std": self.std,
            "wm_device": self.device,
        }
        return config

    def get_name(self):
        name = f"unremovable_{self.key}_{self.std}"
        return name

    def watermark_logits(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.FloatTensor:
        """Returns watermarked logits to be used as distillation target."""
    
        return logits + self.delta_wm
    def spawn_logit_processor(self):
        return LogitProcessorUnremovable(self)