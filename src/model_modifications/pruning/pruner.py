from .pruner_configuration import PruningConfiguration, PruningMethod

from transformers import AutoModelForCausalLM, AutoTokenizer, pytorch_utils
import torch
import torch.nn as nn

from.gblm_pruner import GBLMPruner
from .wanda_pruner import WandaPruner
from .sparsegpt_pruner import SparseGPTPruner

def convert_conv1d_to_linear(conv1d_layer):
    """
    Converts a Conv1D layer into an equivalent nn.Linear layer. Ensure it computes gradients correctly.
    """
    weight = conv1d_layer.weight  # Shape: (input_dim, output_dim)
    bias = conv1d_layer.bias      # Shape: (output_dim,)

    linear_layer = nn.Linear(in_features=weight.shape[0], out_features=weight.shape[1], bias=True, device=weight.device)

    with torch.no_grad():
        linear_layer.weight.copy_(weight.t())  
        linear_layer.bias.copy_(bias)     
        
    linear_layer.weight.requires_grad = True
    linear_layer.bias.requires_grad = True

    return linear_layer

def replace_nested_module(root, module_name, new_module):
    # Split the module name by dots to navigate the hierarchy
    parts = module_name.split('.')
    parent = root
    for p in parts[:-1]:  # navigate to the parent
        parent = getattr(parent, p)
    # The last part is the attribute name of the module we want to replace
    setattr(parent, parts[-1], new_module)

def standardize_model(model):
    replacements = []
    
    # Gather all replacements first
    for name, module in model.named_modules():
        if isinstance(module, pytorch_utils.Conv1D):
            new_module = convert_conv1d_to_linear(module)
            replacements.append((name, new_module))
    
    # Now apply replacements using the helper function
    for name, new_module in replacements:
        replace_nested_module(model, name, new_module)
    
    return model


def prune_model(pruning_config: PruningConfiguration):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(
        pruning_config.base_model,
        torch_dtype=dtype_map[pruning_config.dtype],
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(pruning_config.base_model)
    
    pruner_map = {
        PruningMethod.sparsegpt: SparseGPTPruner,
        PruningMethod.gblm: GBLMPruner,
        PruningMethod.wanda: WandaPruner,
    }
    
    print(pruning_config.pruning_method_config.dict())
    
    model = standardize_model(model)
    
    pruner = pruner_map[pruning_config.pruning_method](
        model=model,
        tokenizer=tokenizer,
        **pruning_config.pruning_method_config.dict()
    )
    
    pruner.prune_model(model)
    
    return model, tokenizer