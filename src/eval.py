from transformers import AutoModelForCausalLM, AutoTokenizer

from watermarks.watermark_config import WatermarkConfiguration
from watermarks.watermark_benchmark import evaluate_watermark

from config import MainConfiguration
import torch

from typing import Optional, List, Dict
import json
import os


class DummyDetector():
    """Useful to generate the wm samples without running the detector to prevent OOM errors"""
    
    def __init__(self):
        pass
    
    def detect(self, input_ids, attention_mask = None):
        return torch.zeros(input_ids.shape[0], dtype=torch.float32).to(input_ids.device)
    
def shorten_name(name):
    
    name = name.replace("huihui-ai-Llama-3.2-1B-Instruct-abliterated", "LLama-3-1B-Harm")
    name = name.replace("HarmfulAssistant", "HA")
    name = name.replace("HelpfulAssistant", "HeA")
    name = name.replace("meta-llama-Llama", "llama")
    
    name = name.replace("AlpacaGPT4", "Al4")
    name = name.replace("OpenWebText", "OWT")
    name = name.replace("RefusalData", "Ref")
    name = name.replace("-ft-", "-")
    
    return name

def get_output_dir(
    configuration, modification_config, modification_type, result_type: str = "models"
):
    # Handling checkpointing + hub saving
    if "finetuning" in modification_type:
        push_to_hub = modification_config.training_args.get("push_to_hub", False)
        if modification_type == "finetuning":
            if push_to_hub and result_type == "models":
                base_model = modification_config.base_model.replace("/", "-")
                out= f"{configuration.huggingface_name}/{base_model}-ft-{modification_config.short_str()}".replace("meta-llama-", "").replace("OpenMathInstruct-AlpacaGPT4-OpenWebText", "M-A-O")
                out = shorten_name(out)
                return out
        
            return f"{configuration.output_directory}/{configuration.base_model}/{result_type}/{modification_type}/{modification_config.short_str()}/final"

        else:
            ckpt = modification_type.split("-")[1]

            if push_to_hub and result_type == "models":
                base_model = modification_config.base_model.replace("/", "-")
                out = f"{configuration.huggingface_name}/{base_model}-ft-{modification_config.short_str()}-ckpt-{ckpt}".replace("meta-llama-", "").replace("OpenMathInstruct-AlpacaGPT4-OpenWebText", "M-A-O")
                out = shorten_name(out)
                return out
                
            return f"{configuration.output_directory}/{configuration.base_model}/{result_type}/finetuning/{modification_config.short_str()}/ckpt-{ckpt}"

    modification_config_short_str = (
        f"/{modification_config.short_str()}" if modification_config is not None else ""
    )
    return f"{configuration.output_directory}/{configuration.base_model}/{result_type}/{modification_type}{modification_config_short_str}".replace("meta-llama-", "").replace("OpenMathInstruct-AlpacaGPT4-OpenWebText", "M-A-O")


class Evaluation:
    def __init__(
        self,
        configuration: MainConfiguration,
        watermark_config: WatermarkConfiguration,
    ):
        self.configuration = configuration
        self.watermark_config = watermark_config
        self.detector = None # Manual detector override

    def eval_model(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        modification_config,
        modification_type: str,
        custom_eval: List = [],
    ):
        res = {}

        if self.detector is not None:
            print("WARNING - Detector was manually overriden. If this is not intended, results WILL be incorrect.")
            detector = self.detector
        else:
            if self.configuration.disable_wm_detector:
                detector = DummyDetector()
            else:
                detector = self.watermark_config.get_detector(model.device, tokenizer)

        for wm_eval_config in self.watermark_config.watermark_eval_config:
            res = self._eval_watermark(model, tokenizer, wm_eval_config, detector)

            self.save_results(
                results=res,
                modification_type=modification_type,
                modification_config=modification_config,
                name=wm_eval_config.name,
            )

        for wm_eval_config in custom_eval:
            res = self._eval_watermark(model, tokenizer, wm_eval_config, detector)

            self.save_results(
                results=res,
                modification_type=modification_type,
                modification_config=modification_config,
                name=wm_eval_config.name,
            )

    def _eval_watermark(self, model, tokenizer, wm_eval_config, detector):
        prompts, completions, pvalues, ppls = evaluate_watermark(
            model, tokenizer, detector, wm_eval_config
        )
        res = {
            "prompts": prompts,
            "completions": completions,
            "pvalues": [p.item() for p in pvalues],
        }
        if ppls is not None:
            res["ppls"] = ppls
        return res
    
    def check_results_exist(
        self,
        modification_type: str,
        modification_config,
        custom_eval: List = []
    ):
        
        wm_eval_configs = self.watermark_config.watermark_eval_config + custom_eval
        results_exist = True
        
        for wm_eval_config in wm_eval_configs:
            
            name = wm_eval_config.name
        
            output_dir = get_output_dir(
                configuration=self.configuration,
                modification_config=modification_config,
                modification_type=modification_type,
                result_type="results",
            )
        
            results_exist *= os.path.exists(f"{output_dir}/results_{name}.jsonl")
            
        return results_exist

    def save_results(
        self,
        results: Dict[str, List],
        modification_type: str,
        modification_config,
        name: str,
    ):
        output_dir = get_output_dir(
            configuration=self.configuration,
            modification_config=modification_config,
            modification_type=modification_type,
            result_type="results",
        )

        os.makedirs(output_dir, exist_ok=True)

        # Save results in JSONL format
        with open(f"{output_dir}/results_{name}.jsonl", "w") as file:
            for values in zip(*results.values()):
                line_dict = {key: value for key, value in zip(results.keys(), values)}
                file.write(json.dumps(line_dict) + "\n")
