from pydantic import BaseModel
from strenum import StrEnum
from typing import Optional, List, Type, TypeVar
import yaml

from model_modifications.pruning.pruner_configuration import PruningConfiguration
from model_modifications.merging.merge import MergeConfiguration
from model_modifications.quantization.quantization import QuantizationConfiguration
from model_modifications.finetuning.finetune import FinetuningConfiguration

from watermarks.watermark_config import WatermarkConfiguration

# Define a generic type for configuration classes
T = TypeVar("T", bound=BaseModel)


class ModificationType(StrEnum):
    merge = "merge"
    prune = "prune"
    quantize = "quantize"
    finetuning = "finetuning"


class MainConfiguration(BaseModel):

    base_model: str
    caching_models: bool = False
    
    evaluate_original: bool = True
    
    watermark_config: WatermarkConfiguration
    disable_wm_detector: bool = False 
    overwrite_results: bool = False
    output_directory: Optional[str] = None
    
    huggingface_name: Optional[str] = None

    pruning_configs: Optional[List[PruningConfiguration]] = None
    merge_configs: Optional[List[MergeConfiguration]] = None
    quantization_configs: Optional[List[QuantizationConfiguration]] = None
    finetuning_configs: Optional[List[FinetuningConfiguration]] = None
    
    def check_config(self, logger):
                
        assert self.watermark_config is not None or self.caching_models, "Either watermark configuration or caching models must be enabled. Else there is nothing to do."
        assert self.output_directory is not None, "Output directory must be provided. Else there is no place to save the results."
        
        if self.watermark_config is None:
            logger.warning("No watermark configuration provided. The watermark will not be evaluated.")
            
        if not self.caching_models:  
            logger.warning("Caching models is disabled. The models will not be saved after modification.")
            
        if len(self.pruning_configs) == 0:
            logger.warning("No pruning configurations provided. The models will not be pruned.")
        
        if len(self.merge_configs) == 0:
            logger.warning("No merge configurations provided. The models will not be merged.")
            
        if len(self.quantization_configs) == 0:
            logger.warning("No quantization configurations provided. The models will not be quantized.")

        if len(self.finetuning_configs) == 0:
            logger.warning("No finetuning configurations provided. The models will not be finetuned.")


    @staticmethod
    def _load_configs(file_paths: List[str], config_class: Type[T], base_model: str) -> List[T]:
        """Load and parse configurations from file paths."""
        return [
            config_class._parse_yaml(
                open(file, "r").read().replace("PLACEHOLDER", base_model)
            )
            for file in file_paths
        ]
        
    @classmethod
    def load_watermark_config(cls, watermark_type, watermark_config_path, watermark_eval_config_path):
        """Load watermark configuration from the YAML content."""
        watermark_config = WatermarkConfiguration.load_configuration(
            watermark_type=watermark_type,
            watermark_config_path=watermark_config_path,
            watermark_eval_config_path=watermark_eval_config_path,
        )
        return watermark_config


    @classmethod
    def parse_yaml(cls, yaml_path: str) -> "MainConfiguration":
        """Parse the main configuration from a YAML file."""
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        return cls._parse_yaml(data)

    @classmethod
    def _parse_yaml(cls, data: str) -> "MainConfiguration":

        # Load nested configurations
        data["watermark_config"] = cls.load_watermark_config(data["watermark_type"], data["watermark_config"], data["watermark_evaluation_config"])
        data["pruning_configs"] = cls._load_configs(data.get("pruning_config_files", []), PruningConfiguration, data["base_model"])
        data["merge_configs"] = cls._load_configs(data.get("merge_config_files", []), MergeConfiguration, data["base_model"])
        data["quantization_configs"] = cls._load_configs(data.get("quantization_config_files", []), QuantizationConfiguration, data["base_model"])
        data["finetuning_configs"] = cls._load_configs(data.get("finetuning_config_files", []), FinetuningConfiguration, data["base_model"])

        return cls.model_validate(data)

    @staticmethod
    def _load_nested_config(config_path: Optional[str], config_class: Type[T]) -> Optional[T]:
        """Load a single nested configuration."""
        if config_path:                
            return config_class.parse_yaml(config_path)
        return None
