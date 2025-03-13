import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from ..watermark_detector import WatermarkDetector
from math import sqrt

def _apply_watermark(model, key, std, sign = 1, valid_params=None):
    with torch.random.fork_rng():
        torch.manual_seed(key)
        for name, param in model.named_parameters():
            
            if valid_params is not None and name not in valid_params:
                continue
            
            noise, current_std = load_noise_individual(key, std, param)
            #print(current_std)
            param.data = sign * noise + param.data
            
def load_noise_individual(key, std, param):
    """WARNING - This function is not deterministic by default."""
    current_std = std 
    noise = torch.randn_like(param) * current_std
    return noise, current_std
            
def fullweigthsmark_model(model, std, key, valid_params=None):
    
    
    _apply_watermark(model, key, std, valid_params=valid_params)
    
    return model

def get_valid_params(model_name, layer_ids, param_names):
    
    if layer_ids is None:
        return None
    
    if len(param_names) == 0:
        raise ValueError("No parameter names provided")
    
    valid_params = []
    
    if model_name == "meta-llama/Llama-2-7b-hf":
        
        for id in layer_ids:
            
            for layer_name in param_names:
                valid_params.append(f"model.layers.{id}.{layer_name}.weight")

    else:
        raise NotImplementedError("Model not supported")
    
    return valid_params    

class FullWeightsMarkWatermark(WatermarkDetector):
    def __init__(
        self,
        model_name: str,
        tokenizer,
        std: float = 5e-3,
        key: int = 0,
        torch_dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        layer_ids: list = None,
        param_names: list = None,
        ctv_path: str = None,
        mean_adjust: float = 0.0,
        std_adjust: float = 1.0,
    ):
        self.tokenizer = tokenizer

        # Save the watermark parameters
        self.model_name = model_name
        self.std = std
        self.key = key
        self.torch_dtype = torch_dtype
        self.device = device
        self.mean_adjust = mean_adjust
        self.std_adjust = std_adjust
        self.layer_ids = layer_ids
        self.valid_params = get_valid_params(model_name, layer_ids, param_names)
        
        print(self.valid_params)
        
        self.load_model()
        
        # Enable grad only for valid parameters
        for name, param in self.model.named_parameters():
            if self.valid_params is not None and name not in self.valid_params:
                param.requires_grad = False
                
        if ctv_path=="None":
            ctv_path = None
        
        if ctv_path is not None:
            self.ctv = ctv_path
        else:
            self.ctv = "None"        
            
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=self.torch_dtype,
            attn_implementation="sdpa"
        ).to(self.device)
        self.model.eval()
        self.model_state = None
        
        self.model_weights = self.model.state_dict()
        
    def apply_watermark(self, key):
        assert self.model_state is None
        self.model_state = key
        _apply_watermark(self.model, self.model_state, self.std, 1, self.valid_params)
        
    def remove_watermark(self):
        assert self.model_state is not None
        self.model.load_state_dict(self.model_weights)   
        self.model_state = None
            
    def probabilities(self, text: str = None, apply_wm: bool = False):
        with torch.no_grad():
            tokenized_text = self.tokenizer(
                text, return_tensors="pt"
            )
            input_ids = tokenized_text["input_ids"].to(self.device)
            attention_mask = tokenized_text["attention_mask"].to(self.device)
                        
            if apply_wm:
                self.apply_watermark(self.key)
                model_scores = self.model(input_ids=input_ids, attention_mask=attention_mask)
                self.remove_watermark()
            else:
                model_scores = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = model_scores.logits
            
            shifted_logits = logits[:, :-1, :]  # Ignore last token prediction
            shifted_input_ids = input_ids[:, 1:]  # Ignore first input token
                        
            model_logprobs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)

            sentence_logprobs = model_logprobs[0, torch.arange(len(shifted_input_ids[0])), shifted_input_ids[0]]
            sentence = [self.tokenizer.decode(token) for token in shifted_input_ids[0]]

        return sentence_logprobs, sentence


    def detect(
        self,
        input_ids: torch.LongTensor, 
        attention_mask: torch.FloatTensor = None
    ) -> torch.LongTensor:
        return self._detect_gaussmark(input_ids, attention_mask)
        #return self._detect_logprobratio(input_ids, attention_mask)
    
    def _detect_gaussmark(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor = None
    ) -> torch.LongTensor:
        
        
        model_scores = self.model(input_ids=input_ids)
            
        mask = torch.ones_like(input_ids)
        if attention_mask is not None:
            mask = attention_mask.bool()
        mask = mask.bool()
        
        # Shifting the logits and input_ids to ignore the last token prediction and the first input token
        shifted_model_logits = model_scores.logits[:, :-1, :]  # shape: (B, seq_len-1, vocab_size)
        shifted_input_ids = input_ids[:, 1:]  # shape: (B, seq_len-1, vocab_size)
        shifted_mask = mask[:, 1:]  # shape: (B, seq_len-1)
    
        model_logprobs = F.log_softmax(shifted_model_logits, dim=-1)

        model_realized_logprobs = model_logprobs.gather(
            dim=-1, index=shifted_input_ids.unsqueeze(-1)
        ).squeeze(-1)

        model_realized_logprobs = model_realized_logprobs.masked_fill(~shifted_mask, 0.0)

        model_sums = model_realized_logprobs.sum(dim=1) # shape (B,)
        
        z_scores = torch.zeros(input_ids.shape[0])
        
        
        for batch in range(input_ids.shape[0]):
            
            logprob = model_sums[batch]
            # Zero the gradients
            self.model.zero_grad()
            logprob.backward(retain_graph=True)
            
            grad_logprob = {}
            # Get the gradients
            for name, param in self.model.named_parameters():
                if self.valid_params is not None and name not in self.valid_params:
                        continue
                grad_logprob[name] = param.grad.detach()
            
            # Compute the scalar product of the gradient and the noise
            z_score = 0
                
            with torch.random.fork_rng():
                torch.manual_seed(self.key)
                for name, grad in grad_logprob.items():
                    noise, current_std = load_noise_individual(self.key, self.std, grad)
                    z_score += torch.sum(grad * noise) / current_std 
                    
            l2_norm = 0
            for name, grad in grad_logprob.items():
                l2_norm += torch.sum(grad ** 2)
            l2_norm = torch.sqrt(l2_norm)
            
            z_score = z_score / (l2_norm )                
            
            z_scores[batch] = z_score
            
        p_values = 1-torch.distributions.Normal(0, 1).cdf(z_scores)
        p_values = p_values.to(input_ids.device)
        
        return p_values
    
    def _detect_logprobratio(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor = None
    ) -> torch.LongTensor:
        
        raise NotImplementedError("Not implemented yet")
        
        with torch.no_grad():
            
            model_scores = self.model(input_ids=input_ids)
            
            self.apply_watermark(self.key)
            detection_scores = self.model(input_ids=input_ids)
            self.remove_watermark()

            mask = torch.ones_like(input_ids)
            if attention_mask is not None:
                mask = attention_mask.bool()
            mask = mask.bool()
            
            # Shifting the logits and input_ids to ignore the last token prediction and the first input token
            shifted_watermark_logits = detection_scores.logits[:, :-1, :]  # shape: (B, seq_len-1, vocab_size)
            shifted_model_logits = model_scores.logits[:, :-1, :]  # shape: (B, seq_len-1, vocab_size)
            shifted_input_ids = input_ids[:, 1:]  # shape: (B, seq_len-1, vocab_size)
            shifted_mask = mask[:, 1:]  # shape: (B, seq_len-1)

            z_scores = self.get_zscore(
                watermark_logits=shifted_watermark_logits,
                model_logits=shifted_model_logits,
                realizations=shifted_input_ids,
                mask=shifted_mask
            )
            
            z_scores = (z_scores - self.mean_adjust) / self.std_adjust
            p_values = 0.5 * torch.erfc(z_scores / torch.sqrt(torch.tensor(2.0, device=z_scores.device)))
            
        return p_values

    def get_zscore(
        self, 
        watermark_logits: torch.Tensor,  # (B, T, vocab)
        model_logits: torch.Tensor,      # (B, T, vocab)
        realizations: torch.Tensor,      # (B, T)
        mask: torch.Tensor,              # (B, T) boolean
    ) -> torch.Tensor:

        watermarked_logprobs = F.log_softmax(watermark_logits, dim=-1)
        model_logprobs       = F.log_softmax(model_logits, dim=-1)

        # 2. Gather the log-prob of the "realized" tokens
        #    shape => (B, T, 1), then squeeze => (B, T)
        wm_realized_logprobs = watermarked_logprobs.gather(
            dim=-1, index=realizations.unsqueeze(-1)
        ).squeeze(-1)

        model_realized_logprobs = model_logprobs.gather(
            dim=-1, index=realizations.unsqueeze(-1)
        ).squeeze(-1)

        # 3. Mask out positions not used
        #    We'll replace them with 0.0 so sums are unaffected by those positions.
        wm_realized_logprobs = wm_realized_logprobs.masked_fill(~mask, 0.0)
        model_realized_logprobs = model_realized_logprobs.masked_fill(~mask, 0.0)

        # 4. Sum across the sequence dimension => shape (B,)
        wm_sums = wm_realized_logprobs.sum(dim=1)
        model_sums = model_realized_logprobs.sum(dim=1)

        # 5. z_score = (sum of watermarked logprobs) - (sum of normal model logprobs)
        z_scores = wm_sums - model_sums  # (B,)

        return z_scores

    def get_config(self):
        config = {
            "type": "weightsmark",
            "key": self.key,
            "std": self.std,
            "wm_device": self.device,
            "model_name": self.model_name,
            "ctv": self.ctv,
            "mean_adjust": self.mean_adjust,
            "std_adjust": self.std_adjust,
            "layer_ids": self.layer_ids,
            "param_names": self.valid_params,
        }
        return config

    def get_name(self):
        name = f"weightsmark_key{self.key}_std{self.std}"
        return name

    def watermark_logits(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.FloatTensor:
        """Returns watermarked logits to be used as distillation target."""

        return logits + self.detector(input_ids).logits

