from pydantic import BaseModel
import yaml
from .kgw.kgw_watermark import KGWWatermark
from .weightsmark.weightsmark import FullWeightsMarkWatermark
from .aar.aar_watermark import AarWatermarkDetector
from .kth.detect import KTHWatermarkDetector
from .rl.rl_watermark import RLWatermarkDetector
from .unremovable.unremovable import WatermarkDetectorUnremovable
from typing import Optional

class WatermarkSchemeConfiguration(BaseModel):
    pass

    @classmethod
    def parse_yaml(cls, yaml_path: str) -> "WeigthsmarkWatermarkConfiguration":
        """Parses a YAML file into a WeigthsmarkWatermarkConfiguration object."""
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return cls.model_validate(data)
    
    def get_detector(self, model_device, tokenizer, **kwargs):
        raise NotImplementedError("get_detector method must be implemented in the subclass.")

class KGWWatermarkConfiguration(WatermarkSchemeConfiguration):
    gamma: float = 0.25
    delta: float = 2
    k: int = 1
    seeding_scheme: str = "simple_1"
    kgw_device: str = "cuda"
    

    def get_detector(self, model_device, tokenizer, **kwargs):
        return KGWWatermark(
            vocab=tokenizer.get_vocab(),
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme=self.seeding_scheme,
            tokenizer = tokenizer,
            device = model_device,
            kgw_device=self.kgw_device,   
        )
        
class WeigthsmarkWatermarkConfiguration(WatermarkSchemeConfiguration):
    modelname: str
    std: float = 5e-3
    key: int = 0
    device: str = "cuda"
    layer_ids: Optional[list] = None
    param_names: Optional[list] = None
    
    def get_detector(self, model_device, tokenizer, **kwargs):
        return FullWeightsMarkWatermark(
            model_name=self.modelname,
            tokenizer=tokenizer,
            std=self.std,
            key=self.key,
            device=self.device,
            layer_ids=self.layer_ids,
            param_names=self.param_names
        )
        
class AARWatermarkConfiguration(WatermarkSchemeConfiguration):
    k: int
    seed: int = 42
    
    def get_detector(self, model_device, tokenizer, **kwargs):
        return AarWatermarkDetector(
            tokenizer=tokenizer,
            model_device=model_device,
            k=self.k,
            seed=self.seed  
        )
        
class KTHWatermarkConfiguration(WatermarkSchemeConfiguration):
    key_len: int
    seed: int = 42
    
    def get_detector(self, model_device, tokenizer, **kwargs):
        return KTHWatermarkDetector(
            tokenizer=tokenizer,
            key=self.seed,
            key_len=self.key_len
        )
        
class RLWatermarkConfiguration(WatermarkSchemeConfiguration):
    reward_model_config: str
    reward_model_path: str
    calibration_cdf_path: str = None
    
    def get_detector(self, model_device, tokenizer, **kwargs):
        return RLWatermarkDetector(
            reward_model_config=self.reward_model_config,
            reward_model_path=self.reward_model_path,
            calibration_cdf_path=self.calibration_cdf_path,
            device=model_device
        )
        
class UnremovableWatermarkConfiguration(WatermarkSchemeConfiguration):
    std: float 
    key: int 
    
    def get_detector(self, model_device, tokenizer, **kwargs):
        return WatermarkDetectorUnremovable(
            tokenizer=tokenizer,
            std=self.std,
            key=self.key,
        )