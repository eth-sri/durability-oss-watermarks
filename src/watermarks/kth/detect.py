import torch
from ..watermark_detector import WatermarkDetector

from levenshtein_rust import permutation_test_parallel, detect


def permutation_test(tokens, key, n, k, vocab_size, n_runs=1000):
    generator = torch.Generator()  # generator is always cpu for reproducibility
    generator.manual_seed(key)

    xi = torch.rand((n, vocab_size), generator=generator, dtype=torch.float32)
    xi = xi.numpy()

    test_result = detect(tokens, n, k, xi, 0.0)
    p_val = permutation_test_parallel(tokens, n, k, vocab_size, test_result, n_runs)

    return p_val

class KTHWatermarkDetector(WatermarkDetector):
    def __init__(self, tokenizer, key, key_len):
        self.vocab_size = len(tokenizer)
        self.key = key
        self.key_len = key_len

    def detect(
        self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor = None
    ) -> torch.LongTensor:
        
        input_ids_device = input_ids.device
        input_ids = input_ids.to("cpu")
        
        pvalues = []
        for tokens in input_ids.numpy():
            pvalues.append(permutation_test(tokens, self.key, self.key_len, len(tokens), self.vocab_size))
            print(pvalues[-1])
        pvalues = torch.tensor(pvalues, device=input_ids_device)
            
        return pvalues