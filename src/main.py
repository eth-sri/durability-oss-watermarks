import argparse
import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import torch
from config import MainConfiguration
from utils import free_memory

from model_modifications.pruning.pruner import prune_model
from model_modifications.merging.merge import merge_models
from model_modifications.quantization.quantization import quantize_model
from model_modifications.finetuning.finetune import finetune_model

from eval import get_output_dir, Evaluation

from strenum import StrEnum

import traceback


def preprocess_dict(data):
    """Recursively convert Enums to strings in a dictionary."""
    if isinstance(data, dict):
        return {key: preprocess_dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [preprocess_dict(item) for item in data]
    elif isinstance(data, StrEnum):
        return data.value  # Convert Enum to its value
    return data


def dump_config_to_yaml(config, output_path: str) -> None:
    """Dump the configuration object to a YAML file."""
    processed_data = preprocess_dict(config.dict())
    with open(output_path, "w") as file:
        yaml.safe_dump(processed_data, file, default_flow_style=False)


# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # Default log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
        datefmt="%Y-%m-%d %H:%M:%S",  # Timestamp format
    )
    logger = logging.getLogger(__name__)  # Create a logger with the module's name
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evalute the OSS watermark resilience")

    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--n_samples", type=int, help="Number of samples to evaluate", default=None)

    return parser.parse_args()


def save(model, tokenizer, modification_config, configuration, type: str):
    output_dir = get_output_dir(configuration, modification_config, type)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    dump_config_to_yaml(modification_config, f"{output_dir}/modification_config.yaml")


def load_model(configuration, modification_config, type):
    output_dir = get_output_dir(configuration, modification_config, type)

    if not os.path.exists(output_dir):
        return False, None, None

    return (
        True,
        AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto"),
        AutoTokenizer.from_pretrained(output_dir),
    )


def modify_and_evaluate(
    configuration: MainConfiguration,
    modification_config,
    modification_type,
    evaluation: Evaluation,
):
    loaded = False

    if not configuration.overwrite_results:
        if evaluation.check_results_exist(modification_type, modification_config):
            raise ValueError(
                "Results already exist. Set overwrite_results to True to overwrite them."
            )

    if configuration.caching_models:
        loaded, model, tokenizer = load_model(
            configuration, modification_config, modification_type
        )
    if not loaded:
        if modification_type == "prune":
            model, tokenizer = prune_model(modification_config)
        elif modification_type == "merge":
            model, tokenizer = merge_models(modification_config)
        elif modification_type == "quantize":
            model, tokenizer = quantize_model(modification_config)
        if configuration.caching_models:
            save(
                model,
                tokenizer,
                modification_config,
                configuration,
                type=modification_type,
            )

    evaluation.eval_model(model, tokenizer, modification_config, modification_type)
    free_memory()


def finetuning_evaluation(
    configuration: MainConfiguration,
    finetuning_config,
    evaluation: Evaluation,
):
    """Fine-tuning because of checkpointing uses a custom evaluation logic built into the trainer."""

    results_output_dir = get_output_dir(
        configuration, finetuning_config, "finetuning", result_type="results"
    )
    model_output_dir = get_output_dir(
        configuration, finetuning_config, "finetuning", result_type="models"
    )

    if not configuration.overwrite_results:
        if evaluation.check_results_exist(
            modification_type="finetuning",
            modification_config=finetuning_config,
            custom_eval=finetuning_config.watermark_eval_config,
        ):
            raise ValueError(
                "Results already exist. Set overwrite_results to True to overwrite them."
            )

    finetune_model(
        finetuning_config,
        model_output_dir,
        results_output_dir,
        evaluation,
        save_pretrained=configuration.caching_models,
        overwrite_results=configuration.overwrite_results,
    )
    free_memory()

def evaluate_original(configuration, evaluation, logger = None):
    
    if (not configuration.overwrite_results) and evaluation.check_results_exist(
        modification_type="original", modification_config=None
    ):
        if logger:
            logger.error(
                "Results already exist. Set overwrite_results to True to overwrite them."
            )
    else:
        model, tokenizer = (
            AutoModelForCausalLM.from_pretrained(
                configuration.base_model, device_map="auto"
            ),
            AutoTokenizer.from_pretrained(configuration.base_model, padding_side="left"),
        )
        evaluation.eval_model(model, tokenizer, None, "original")
        del model
        del tokenizer
        free_memory()

def main(configuration: MainConfiguration):
    logger = setup_logging()

    configuration.check_config(logger)

    evaluation = Evaluation(
        configuration=configuration, watermark_config=configuration.watermark_config
    )

    if configuration.evaluate_original:
        evaluate_original(configuration, evaluation, logger)  

    merge_configs = configuration.merge_configs
    if len(merge_configs) != 0:
        logger.info("Merging models")
        for merge_config in merge_configs:
            try:
                modify_and_evaluate(
                    configuration=configuration,
                    modification_config=merge_config,
                    modification_type="merge",
                    evaluation=evaluation,
                )
            except Exception as e:
                logger.error(f"Error merging models: {e}")
                free_memory()
                continue

    prune_configs = configuration.pruning_configs
    if len(prune_configs) != 0:
        logger.info("Pruning models")
        for prune_config in prune_configs:
            try:
                modify_and_evaluate(
                    configuration=configuration,
                    modification_config=prune_config,
                    modification_type="prune",
                    evaluation=evaluation,
                )
            except Exception as e:
                logger.error(f"Error pruning models: {e}")
                free_memory()
                continue

    quantization_configs = configuration.quantization_configs
    if len(quantization_configs) != 0:
        logger.info("Quantizing models")
        for quantization_config in quantization_configs:
            with torch.cuda.amp.autocast():
                modify_and_evaluate(
                    configuration=configuration,
                    modification_config=quantization_config,
                    modification_type="quantize",
                    evaluation=evaluation,
                )

    finetuning_configs = configuration.finetuning_configs
    if len(finetuning_configs) != 0:
        logger.info("Fine-tuning models")
        for finetuning_config in finetuning_configs:
            try:
                finetuning_evaluation(
                    configuration=configuration,
                    finetuning_config=finetuning_config,
                    evaluation=evaluation,
                )
            except Exception as e:
                logger.error(
                    f"Error fine-tuning models during {finetuning_config.short_str()}: {e}"
                )
                print(traceback.format_exc())
                free_memory()
                continue


if __name__ == "__main__":
    args = parse_args()
    configuration = MainConfiguration.parse_yaml(args.config)
    
    if args.n_samples:
        for wm_config in configuration.watermark_config.watermark_eval_config:
            wm_config.n_samples = args.n_samples
    
    main(configuration)
