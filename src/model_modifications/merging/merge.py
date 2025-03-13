# actually do merge
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import hashlib

import yaml
import mergekit.config
from mergekit.merge import MergeOptions, run_merge
import os
import shutil

class MergeConfiguration(mergekit.config.MergeConfiguration):
    merge_name: str = "default_name"
    
    @staticmethod
    def _parse_yaml(yaml_str: str) -> "MergeConfiguration":
        data = yaml.safe_load(yaml_str)
        return MergeConfiguration.model_validate(data)

    def short_str(self):
        return f"{self.base_model}/{self.merge_method}/{self.merge_name}"
    
def merge_models(
    merge_config: MergeConfiguration,
    output_dir: str = "/tmp/merged_model",
    lora_merge_cache: str = "/tmp",
    copy_tokenizer: bool = True,  # Whether to have a tokenizer bundled with the merged model
    lazy_unpickle: bool = False,  # Experimental low-memory model loader
    low_cpu_memory: bool = False,  # Enable if you somehow have more VRAM than RAM+swap
    return_model: bool = True,  # Return the merged model
    use_path_id: bool = False,  # Use the path identifier as the output path
    delete_cache: bool = True,  # Delete the cache after merging
):
    
    assert not delete_cache or return_model, "Can't delete cache without returning model"
    
    if use_path_id:
        yaml_str = merge_config.to_yaml()
        identifier = hashlib.sha256(yaml_str.encode("utf-8")).hexdigest()   
        output_dir = f"{output_dir}/{merge_config.base_model}/{identifier}"

        # Check if the model already exists
        if os.path.exists(output_dir):
            print("Model already exists, skipping merge")
            if return_model:
                return AutoModelForCausalLM.from_pretrained(
                    output_dir
                ), AutoTokenizer.from_pretrained(output_dir)
            return

    if delete_cache:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    run_merge(
        merge_config,
        out_path=output_dir,
        options=MergeOptions(
            lora_merge_cache=lora_merge_cache,
            cuda=torch.cuda.is_available(),
            copy_tokenizer=copy_tokenizer,
            lazy_unpickle=lazy_unpickle,
            low_cpu_memory=low_cpu_memory,
        ),
    )
    print("Done")
    
    if return_model:
        model, tokenizer = (
            AutoModelForCausalLM.from_pretrained(output_dir,device_map="auto"),
            AutoTokenizer.from_pretrained(output_dir),
        )
    
    if delete_cache:
        shutil.rmtree(output_dir)

    if return_model:
        return model, tokenizer
