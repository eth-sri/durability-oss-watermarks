from typing import Optional

import torch
from transformers import AutoTokenizer

from ..watermark_detector import WatermarkDetector

DEFAULT_SEED = 42


class AarWatermark:
    def __init__(
        self,
        vocab_size: int,
        k: int,
        seed: int = DEFAULT_SEED,
        eps: float = 1e-20,
        device: Optional[str] = None,
    ):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        generator = torch.Generator()  # generator is always cpu for reproducibility
        generator.manual_seed(seed)

        # clamp to avoid NaNs
        uniform = torch.clamp(
            torch.rand(
                (vocab_size * k, vocab_size), generator=generator, dtype=torch.float32
            ),
            min=eps,
        )
        self.gumbel = (-torch.log(torch.clamp(-torch.log(uniform), min=eps))).to(device)

        self.k = k
        self.vocab_size = vocab_size
        self.seed = seed
        self.eps = eps
        self.device = device

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.k:
            return scores
        prev_token = torch.sum(input_ids[:, -self.k :], dim=-1)  # (batch_size,)
        gumbel = self.gumbel[prev_token]  # (batch_size, vocab_size)
        return scores[..., : gumbel.shape[-1]] + gumbel

    def watermark_logits_argmax(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.LongTensor:
        """Finds argmax token for watermark, returns token indexes to be used for cross-entropy loss.

        Returns tensor of shape (batch, seq_len), where each element is a token index.
        """
        hashes = torch.sum(
            input_ids.unfold(-1, self.k, 1), dim=-1
        )  # (batch, seq_len - k + 1)
        gumbel = self.gumbel[hashes]  # (batch, seq_len - k + 1, vocab_size)
        # tokenizer vocab size and model outputs vocab size may be different
        logits[..., self.k - 1 :, : gumbel.shape[-1]] += (
            gumbel  # (batch, seq_len, vocab_size)
        )
        tokens = torch.argmax(logits, dim=-1)  # (batch, seq_len)
        return tokens


class AarWatermarkDetector(WatermarkDetector):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model_device: str,
        k: int = 1,
        seed: int = DEFAULT_SEED,
        eps: float = 1e-20,
    ):
        # Prevent OOM due to the uniform tensor size
        model_device = "cpu"

        generator = torch.Generator()  # generator is always cpu for reproducibility
        generator.manual_seed(seed)
        vocab_size = len(tokenizer)
        self.uniform = torch.clamp(
            torch.rand(
                (vocab_size * k, vocab_size), generator=generator, dtype=torch.float32
            ),
            min=eps,
            max=1 - eps,
        ).to(model_device)

        self.tokenizer = tokenizer
        self.k = k
        self.seed = seed
        self.eps = eps
        self.vocab_size = vocab_size

        self.device = model_device

    def detect(
        self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor = None
    ) -> torch.LongTensor:
        """
        Returns p-value, where null hypothesis is that the text is not watermarked.

        Under null hypothesis, each u is Uniform(0, 1), so each score (-log(1 -u )) is Exp(1).
        So the sum of scores is distributed as Gamma(n_tokens, 1).
        """

        input_ids_device = input_ids.device
        input_ids = input_ids.to(self.device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)

        uniform = self.uniform
        sums = input_ids.unfold(1, self.k, 1).sum(-1)[:, :-1]
        tokens = input_ids[:, self.k :]
        u = uniform[sums, tokens]
        mask = attention_mask[:, self.k :].float()
        score = -torch.log(1 - u.clamp(max=1 - 1e-9)) * mask
        score = score.sum(dim=1)
        alpha = mask.sum(dim=1)
        alpha_safe = alpha.clone()
        alpha_safe[alpha_safe == 0] = 1
        pvalues = 1 - torch.special.gammainc(alpha_safe, score)
        pvalues[alpha == 0] = 1

        pvalues = pvalues.to(input_ids_device)

        return pvalues
