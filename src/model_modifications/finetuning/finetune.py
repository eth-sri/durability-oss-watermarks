from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel
import yaml
from typing import Optional, Dict, List
import torch
import numpy as np
import evaluate
import glob
from huggingface_hub import snapshot_download, repo_exists, HfFileSystem
import shutil
import os

from .dataset import get_dataset, DatasetType
from src.watermarks.watermark_config import WatermarkEvalConfiguration
from src.utils import free_memory


class FinetuningConfiguration(BaseModel):
    base_model: str
    dtype: Optional[str] = "float32"

    training_args: Dict
    lora_config: Optional[LoraConfig] = None

    train_dataset: (
        DatasetType  # Name of training dataset. See dataset.py to implement a dataset.
    )

    metric: Optional[str] = None  # Metric name from evaluate.
    watermark_eval_config: List[
        WatermarkEvalConfiguration
    ] = []  # Path to custom watermark evaluation config (for task specific eval).

    @staticmethod
    def parse_yaml(file_path: str) -> "FinetuningConfiguration":
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return FinetuningConfiguration.load_configuration(data)

    @staticmethod
    def _parse_yaml(yaml_as_string: str) -> "FinetuningConfiguration":
        data = yaml.safe_load(yaml_as_string)
        return FinetuningConfiguration.load_configuration(data)

    @staticmethod
    def load_configuration(data):
        configs = []
        for config in data.get("watermark_eval_config", []):
            configs.append(WatermarkEvalConfiguration.parse_yaml(config))
        data["watermark_eval_config"] = configs
        if data.get("lora_config", None) is not None:
            data["lora_config"] = LoraConfig(**data.get("lora_config", {}))
        return FinetuningConfiguration(**data)

    def short_str(self):
        lora = ""
        if self.lora_config is not None:
            lora = "-lora"
        return f"{self.train_dataset}{lora}"


class CustomTrainer(Trainer):
    def __init__(
        self,
        results_output_dir,
        evaluation,
        tokenizer_wm,
        finetuning_config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.results_output_dir = results_output_dir
        self.evaluation = evaluation
        self.tokenizer_wm = tokenizer_wm

        self.finetuning_config = finetuning_config

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False,
        is_checkpoint: bool = True,
    ):
        """
        While saving the model, we also evaluate the watermark.
        Additionaly, we disable the saving process according to the MainConfiguration.
        """

        if is_checkpoint:
            checkpoint_folder = f"-{self.state.global_step}"
        else:
            checkpoint_folder = ""

        super().save_model(output_dir, _internal_call)

        with torch.no_grad():
            self.evaluation.eval_model(
                self.model,
                self.tokenizer_wm,
                self.finetuning_config,
                f"finetuning{checkpoint_folder}",
                custom_eval=self.finetuning_config.watermark_eval_config,
            )


def finetune_model(
    finetuning_config: FinetuningConfiguration,
    model_output_dir: str,
    results_output_dir: str,
    evaluation,
    save_pretrained: bool,
    overwrite_results: bool,
):
    
    print(f"Finetuning {finetuning_config.base_model} on {finetuning_config.train_dataset}.")
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if finetuning_config.training_args.get("push_to_hub", False):
        save_pretrained = True  # Always save to hub if push_to_hub is enabled

    try:
        tokenizer = AutoTokenizer.from_pretrained(finetuning_config.base_model)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    training_args = finetuning_config.training_args
    training_args["output_dir"] = model_output_dir
    training_args["hub_strategy"] = "all_checkpoints"
    training_args["report_to"]="none"
    training_args = TrainingArguments(
        **training_args,
    )

    if finetuning_config.metric is not None:
        metric = evaluate.load(finetuning_config.metric)
        compute_metrics = lambda eval_pred: compute_metrics(eval_pred, metric)  # noqa: E731
    else:
        compute_metrics = None

    train_ds, eval_ds, tokenizer = get_dataset(
        tokenizer, finetuning_config.train_dataset
    )

    resume_from_checkpoint = evaluate_previous_checkpoints(
        save_pretrained,
        finetuning_config,
        model_output_dir,
    )
    free_memory()

    model = AutoModelForCausalLM.from_pretrained(
        finetuning_config.base_model,
        device_map="cuda",
        torch_dtype=dtype_map[finetuning_config.dtype],
    )

    if finetuning_config.lora_config is not None:
        model = get_peft_model(model, finetuning_config.lora_config)

    model = resize_model_if_needed(
        tokenizer, model
    )  # Due to potential addition of chat template


    trainer = CustomTrainer(
        results_output_dir=results_output_dir,
        evaluation=evaluation,
        finetuning_config=finetuning_config,
        tokenizer_wm=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
        
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(is_checkpoint=False)

    # Delete the repository clone if saving to hub
    if finetuning_config.training_args.get("push_to_hub", False) or not save_pretrained:
        output_dir = model_output_dir

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def resize_model_if_needed(tokenizer, model):
    """
    Resizes the model's embedding layer if the tokenizer's vocabulary size
    is larger than the current embedding layer. Useful when using chat template.
    """
    # Get tokenizer and model vocabulary sizes
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = model.get_input_embeddings().weight.size(0)

    # Check if resizing is needed
    if tokenizer_vocab_size > model_vocab_size:
        print(
            f"Resizing model embeddings from {model_vocab_size} to {tokenizer_vocab_size}."
        )
        model.resize_token_embeddings(tokenizer_vocab_size)

    return model


def evaluate_previous_checkpoints(
    save_pretrained: bool,
    finetuning_config,
    model_output_dir: str,
):
    """
    Evaluates all previous checkpoints for a finetuning run.
    """

    if not save_pretrained:
        return False

    # Get all checkpoint directories
    push_to_hub = finetuning_config.training_args.get("push_to_hub", False)

    if push_to_hub:
        if not repo_exists(model_output_dir):
            return False

        fs = HfFileSystem()
        checkpoints = fs.glob(f"{model_output_dir}/checkpoint-*")

        # Download the repository
        snapshot_download(model_output_dir, local_dir=model_output_dir)

    else:
        checkpoints = glob.glob(f"{model_output_dir}/checkpoint-*")

    if len(checkpoints) == 0:
        return False

    return True
