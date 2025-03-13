import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..watermark_detector import WatermarkDetector
from torch import Tensor

class CalibrationCDF:
    def __init__(self, scores: Tensor):
        self.sorted_scores, _ = torch.sort(scores)

    def __call__(self, scores: Tensor) -> Tensor:
        
        device = scores.device
        
        self.sorted_scores = self.sorted_scores.to(device)
        
        indices = torch.searchsorted(self.sorted_scores, scores, right=True)
        pvals = indices.float() / self.sorted_scores.numel()
        pvals = 1 - pvals # LLM scores are positive whereas human scores are negative.
        return pvals
        
class RewardModelDetector(nn.Module):
    def __init__(self, base_model, tokenizer):
        super(RewardModelDetector, self).__init__()
        self.config = base_model.config

        if hasattr(self.config, "word_embed_proj_dim"):
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,1,bias=False)
        else:
            self.v_head = nn.Linear(self.config.hidden_size, 1, bias=False)
        self.rwtransformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.tokenizer = tokenizer

        if "OPT" in base_model.__class__.__name__:
            self.num_padding_at_beginning = 1
        else:
            assert "Llama" in base_model.__class__.__name__ or "llama" in base_model.__class__.__name__
            self.num_padding_at_beginning = 0

    @classmethod
    def load(self, reward_model_state_dict, reward_model_config) -> "RewardModelDetector":
        """Creates a RewardModelDetector from a RewardModel."""
        reward_model_base = AutoModel.from_pretrained(reward_model_config)
        tokenizer = AutoTokenizer.from_pretrained(reward_model_config)
        
        tokenizer.pad_token = tokenizer.eos_token
        
        reward_model_detector = RewardModelDetector(reward_model_base, tokenizer)
        reward_model_detector.load_state_dict(reward_model_state_dict)
        return reward_model_detector

    def detect(self, input_ids, attention_mask, use_cache=False):
        
        device = input_ids.device
        
        self.rwtransformer = self.rwtransformer.to(device)
        self.v_head = self.v_head.to(device)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
                
        chosen_outputs = self.rwtransformer(input_ids, attention_mask, use_cache=use_cache)
        chosen_rewards = self.v_head(chosen_outputs[0]).squeeze(-1)
        seq_len = input_ids.shape[1]

        chosen_mean_scores = []

        for i in range(len(input_ids)):
            input_id = input_ids[i]
            chosen_reward = chosen_rewards[i]

            start_ind = attention_mask[i].nonzero()[0].item()

            c_inds = (input_id[start_ind:] == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item()+start_ind if len(c_inds)>self.num_padding_at_beginning else seq_len

            r_ind = min(c_ind, seq_len)
            c_ind = min(c_ind, r_ind)
      
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        return chosen_mean_scores

class RLWatermarkDetector(WatermarkDetector):
    
    def __init__(
        self,
        reward_model_config: str,
        reward_model_path: str,
        calibration_cdf_path: str,
        device: str
        ):
        
        reward_model = torch.load(reward_model_path)
        self.reward_model = RewardModelDetector.load(reward_model, reward_model_config)
        if calibration_cdf_path is not None:
            self.calibration_cdf = torch.load(calibration_cdf_path)
        else:
            self.calibration_cdf = None
            
        self.device = device
        self.reward_model.eval()

    def detect(
        self,
        input_ids: torch.LongTensor, 
        attention_mask: torch.FloatTensor = None
    ) -> torch.LongTensor:
        
        scores = self.reward_model.detect(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.calibration_cdf is not None:
            scores = self.calibration_cdf(scores)
        
        return scores