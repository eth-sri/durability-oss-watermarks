import torch
import torch.nn as nn

def add_position_ids_support(layer):
    """
    Overrides the forward method of the provided layer to accept `position_ids`.

    Args:
        layer (nn.Module): The layer whose forward method will be overridden.

    Returns:
        nn.Module: The same layer with an updated forward method.
    """

    original_forward = layer.forward

    def new_forward(self, *args, position_ids=None, **kwargs):
        return original_forward(*args, **kwargs)
    
    layer.forward = new_forward.__get__(layer, nn.Module)
    return layer


def get_model_seq_length(model):
    try:
        return min(model.config.max_position_embeddings, 2048)
    except AttributeError:
        return 2048

def _get_layers(model):
    
    if "LlamaForCausalLM" in model.config.architectures:
        layers = model.model.layers
    elif "GPT2LMHeadModel" in model.config.architectures:
        layers = model.transformer.h
        layers= [add_position_ids_support(layer) for layer in layers] # GPT2LMHeadModel has a different forward signature
    else:
        raise ValueError("Model not supported for pruning")

    return layers

def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def prepare_calibration_input(model, dataloader, nsamples, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    model_seq_length = get_model_seq_length(model)
    
    layers = _get_layers(model)
    
    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model_seq_length, model.config.hidden_size), dtype=dtype, device=device
    )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]

            cache["position_ids"] = kwargs.get("position_ids",None)

            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity