"""
Code forked from: https://github.com/VILA-Lab/GBLM-Pruner
"""

from .pruner_configuration import Pruner
from .utils.data_utils import get_loaders
from .utils.layerwrapper import WrappedGPT
from .utils.model_utils import (
    find_layers,
    prepare_calibration_input,
    return_given_alpha,
    get_model_seq_length,
    _get_layers
)
import torch
from transformers import AdamW
from tqdm import tqdm
import os


class GBLMPruner(Pruner):
    def __init__(
        self,
        model,
        tokenizer,
        base_model: str,
        n_samples: int = 128,
        scale: int = 100,
        seed: int = 0,
        prune_n: int = 0,
        prune_m: int = 0,
        gradient_inv: bool = False,
        use_variant: bool = False,
        sparsity_ratio: float = 0.5,
        gradient_norm: str = "l1",
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(model, **kwargs)

        self.n_samples = n_samples
        self.scale = scale
        self.tokenizer = tokenizer
        self.seed = seed
        self.device = model.device
        self.model_name = base_model
        self.gradient_norm = gradient_norm

        self.sparsity_ratio = sparsity_ratio
        self.prune_n = prune_n
        self.prune_m = prune_m

        self.gradient_inv = gradient_inv
        self.use_variant = use_variant

        # Whether to use cache for gradients
        self.use_cache = use_cache

    def get_gradients(self, model):
        try:
            gradients_l1, gradients_l2 = self.load_gradients()
        except FileNotFoundError:
            gradients_l1, gradients_l2 = self._compute_gradients(model)
            self.save_gradients(gradients_l1, gradients_l2)

        if self.gradient_norm == "l1":
            return gradients_l1
        elif self.gradient_norm == "l2":
            return gradients_l2
        else:
            raise ValueError("gradient_norm must be either 'l1' or 'l2'")

    def _compute_gradients(self, model):
        tokenizer = self.tokenizer
        device = self.device

        model_seq_length = get_model_seq_length(model)

        dataloader, _ = get_loaders(
            "c4",
            nsamples=self.n_samples,
            seed=self.seed,
            seqlen=model_seq_length,
            tokenizer=tokenizer,
        )

        optimizer = AdamW(model.parameters(), lr=0.01, eps=0.01)
        optimizer.zero_grad()
        scale = self.scale
        grad_up = gradient_computation(model, scale)
        nsample = 0
        model.train()
        for input_ids, labels in dataloader:
            nsample += 1
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            grad_up.update_gradient(model, nsample)
            optimizer.zero_grad()
        gradients_l2 = grad_up.gradients_l2

        for name in gradients_l2:
            grad_sqrt = torch.sqrt(gradients_l2[name])
            gradients_l2[name] = grad_sqrt.to(dtype=torch.float16)

        return grad_up.gradients_l1, gradients_l2

    def load_gradients(self):
        _, g1_path, g2_path = self._get_gradients_path()
        with open(g1_path, "rb") as f:
            gradients_l1 = torch.load(f)
        with open(g2_path, "rb") as f:
            gradients_l2 = torch.load(f)
        return gradients_l1, gradients_l2

    def save_gradients(self, gradients_l1, gradients_l2):
        if not self.use_cache:
            return

        path, g1_path, g2_path = self._get_gradients_path()
        os.makedirs(path, exist_ok=True)

        print(f"Saving gradients to {g1_path} and {g2_path}")

        with open(g2_path, "wb") as f:
            torch.save(gradients_l2, f)
        with open(g1_path, "wb") as f:
            torch.save(gradients_l1, f)

    def _get_gradients_path(self):
        path = f".cache/gradients/{self.model_name}"
        g1_path = f"{path}/gradients_aggregrate_norm_l1_model.pth"
        g2_path = f"{path}/gradients_aggregrate_norm_l2_model.pth"
        return path, g1_path, g2_path

    def prune_model(self, model):
        use_cache = model.config.use_cache
        model.config.use_cache = False

        model_seq_length = get_model_seq_length(model)

        tokenizer = self.tokenizer
        device = self.device

        gradients = self.get_gradients(model)

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
                indexed_name = f"{name}_layer_{i}"
                print(f"pruning layer {i} name {name}")
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )
                if not self.gradient_inv:
                    # small_value = torch.tensor(1e-8, dtype=gradients[indexed_name].dtype, device=gradients[indexed_name].device)
                    W_metric_grad = torch.abs(subset[name].weight.data) * torch.abs(
                        gradients[indexed_name].to(device=W_metric.device)
                    )
                    W_metric = W_metric.to(dtype=torch.float32) + W_metric_grad.to(
                        dtype=torch.float32
                    )  # + small_value)
                else:
                    small_value = torch.tensor(
                        1e-8,
                        dtype=gradients[indexed_name].dtype,
                        device=gradients[indexed_name].device,
                    )
                    gradient_inv = 1 / (
                        torch.abs(gradients[indexed_name]) + small_value
                    )
                    W_metric = W_metric.to(dtype=torch.float32) * gradient_inv.to(
                        device=W_metric.device
                    ).to(dtype=torch.float32)

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


class gradient_computation:
    def __init__(self, model, scale):
        self.model = model
        self.gradients_l1 = dict()
        self.gradients_l2 = dict()
        self.nsample = 0
        self.scale = scale
        self.device = torch.device("cpu")
        self.gradients_init()

    def gradients_init(self):
        layers = _get_layers(self.model)
        for i in tqdm(range(len(layers)), desc="initializing the gradient list ...."):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f"{name}_layer_{i}"
                self.gradients_l1[indexed_name] = torch.zeros_like(
                    subset[name].weight, dtype=torch.float16, device=self.device
                )
                self.gradients_l2[indexed_name] = torch.zeros_like(
                    subset[name].weight, dtype=torch.float32, device=self.device
                )

    def update_gradient(self, model, nsample):
        assert nsample - self.nsample == 1, "number of samples must be incremented by 1"
        layers = _get_layers(model)
        for i in tqdm(
            range(len(layers)),
            desc=f"updating the gradient of sample no: {self.nsample}",
        ):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f"{name}_layer_{i}"
                if subset[name].weight.grad is None:
                    print(f"Error: {name} has none gradient")
                if subset[name].weight.grad is not None:
                    assert (
                        subset[name].weight.requires_grad is True
                    ), f"Required grad must be true ( {name}: {subset[name].weight.requires_grad})"
                    grad = (
                        subset[name]
                        .weight.grad.detach()
                        .clone()
                        .to(dtype=torch.float32)
                    )  # Cast to float32
                    all_zero = (torch.abs(grad) == 0).all()
                    assert (
                        int(all_zero) == 0
                    ), f"all the elements in the tensor are zero.: {all_zero}"
                    assert (
                        self.gradients_l1[indexed_name].shape == grad.shape
                    ), "shape mismatch"
                    self.gradients_l1[indexed_name] = self.gradients_l1[
                        indexed_name
                    ] + torch.abs(grad * self.scale).to(device=self.device).to(
                        dtype=torch.float16
                    )
                    self.gradients_l2[indexed_name] = self.gradients_l2[
                        indexed_name
                    ] + torch.abs((grad * self.scale) ** 2).to(device=self.device)
        self.nsample = nsample
