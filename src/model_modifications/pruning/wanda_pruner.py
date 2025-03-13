"""
Code forked from: https://github.com/VILA-Lab/GBLM-Pruner
"""

from .pruner_configuration import Pruner
import torch
from .utils.layerwrapper import WrappedGPT
from .utils.data_utils import get_loaders
from .utils.model_utils import (
    prepare_calibration_input,
    find_layers,
    return_given_alpha,
    get_model_seq_length,
    _get_layers
)


class WandaPruner(Pruner):
    def __init__(
        self,
        model,
        tokenizer,
        n_samples: int = 128,
        seed: int = 0,
        prune_n: int = 0,
        prune_m: int = 0,
        use_variant: bool = False,
        sparsity_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(model, **kwargs)

        self.n_samples = n_samples
        self.tokenizer = tokenizer
        self.seed = seed
        self.device = model.device

        self.sparsity_ratio = sparsity_ratio
        self.prune_n = prune_n
        self.prune_m = prune_m

        self.use_variant = use_variant

    @torch.no_grad()
    def prune_model(self, model):
        use_cache = model.config.use_cache
        model.config.use_cache = False
        
        model_seq_length = get_model_seq_length(model)

        tokenizer = self.tokenizer
        device = model.device

        dataloader, _ = get_loaders(
            "c4",
            nsamples=self.n_samples,
            seed=self.seed,
            seqlen=model_seq_length,
            tokenizer=tokenizer,
        )

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, self.n_samples, device
            )

        layers = _get_layers(model)

        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            if (
                f"model.layers.{i}" in model.hf_device_map
            ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    attention_mask.to(dev),
                    position_ids.to(dev),
                )

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(
                    subset[name], layer_id=i, layer_name=name
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(
                    subset[name].register_forward_hook(add_batch(name))
                )  ## this is a important function.
            for j in range(self.n_samples):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]

            for h in handles:
                h.remove()

            for sub_i, name in enumerate(subset):
                print(f"pruning layer {i} name {name}")
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )

                W_mask = (
                    torch.zeros_like(W_metric) == 1
                )  ## initialize a mask to be all False
                if self.prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % self.prune_m == 0:
                            tmp = W_metric[:, ii : (ii + self.prune_m)].float()
                            W_mask.scatter_(
                                1,
                                ii
                                + torch.topk(tmp, self.prune_n, dim=1, largest=False)[
                                    1
                                ],
                                True,
                            )
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if self.use_variant:
                        # wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0.0, 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                        while (
                            torch.abs(cur_sparsity - self.sparsity_ratio) > 0.001
                        ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > self.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][
                            :, : int(W_metric.shape[1] * self.sparsity_ratio)
                        ]
                        W_mask.scatter_(1, indices, True)

                subset[name].weight.data[W_mask] = 0  ## set weights to zero

            for j in range(self.n_samples):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
            inps, outs = outs, inps

        model.config.use_cache = use_cache
        torch.cuda.empty_cache()

        return model
