import torch
import gc
from huggingface_hub import HfApi
import json
import requests
import yaml


def free_memory():
    """Free memory by running the garbage collector and emptying the cache."""
    gc.collect()
    torch.cuda.empty_cache()
    
def push_to_hub(repo_id: str, model, tokenizer, watermark_config: dict = None):
    model.push_to_hub(repo_id, use_temp_dir=True, private=True)
    tokenizer.push_to_hub(repo_id, use_temp_dir=True, private=True)

    file_path = "/tmp/watermark_config.json"
    with open(file_path, "w") as f:
        json.dump(watermark_config, f)

    if watermark_config is not None:

        api = HfApi()
        api.upload_file(
            path_or_fileobj=file_path,  
            path_in_repo="watermark_config.json",  
            repo_id=repo_id, 
            repo_type="model",
            commit_message="Upload watermark config",
        )