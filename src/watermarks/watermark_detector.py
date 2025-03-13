import torch
from typing import Dict


class WatermarkDetector:
    def __init__(self):
        pass

    def detect(
        self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        """Should work with batched input and return a tensor of pvalues corresponding to the batch dimenstion."""
        raise NotImplementedError("detect method must be implemented in the subclass.")
