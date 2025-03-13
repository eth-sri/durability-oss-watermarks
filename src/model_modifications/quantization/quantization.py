import yaml
from pydantic import BaseModel, Field
from strenum import StrEnum
from typing import Optional
import torch

from awq import AutoAWQForCausalLM

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GPTQConfig,
    AwqConfig,
    HqqConfig,
)


class QuantizationMethod(StrEnum):
    bitsandbytes = "bitsandbytes"
    gptq = "gptq"
    awq = "awq"
    hqq = "hqq"


class QuantizationBitsAndBytesConfiguration(BaseModel):
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    llm_int8_threshold: Optional[float] = 6.0
    bnb_4bit_quant_type: Optional[str] = "fp4"  # Either fp4 or nf4
    bnb_4bit_use_double_quant: Optional[bool] = False  # Nested quantization

    def __post_init__(self):
        assert (
            self.load_in_8bit or self.load_in_4bit
        ), "At least one of load_in_8bit or load_in_4bit should be True"
        assert not (
            self.load_in_8bit and self.load_in_4bit
        ), "Only one of load_in_8bit or load_in_4bit should be True"
        assert self.bnb_4bit_quant_type in [
            "fp4",
            "nf4",
        ], "bnb_4bit_quant_type should be either fp4 or nf4"

    def to_dict(self, **kwargs):
        return BitsAndBytesConfig.from_dict(self.model_dump())

    def short_str(self):
        if self.load_in_8bit:
            return "8bit"
        else:
            s = f'4bit-{self.bnb_4bit_quant_type}-{"double" if self.bnb_4bit_use_double_quant else "single"}'
            return s


class QuantizationGPTQ(BaseModel):
    bits: int  # Number of bits to quantize to (2,3,4,8)
    batch_size: int = 1
    dataset: str = "wikitext2"

    def __post_init__(self):
        assert self.bits in [2, 3, 4, 8], "bits should be one of 2,3,4,8"

    def short_str(self):
        return f"{self.bits}"

    def to_dict(self, tokenizer: AutoTokenizer, **kwargs):
        dict_config = self.model_dump()
        dict_config["tokenizer"] = tokenizer
        dict_config["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        return GPTQConfig.from_dict(dict_config)


class QuantizationAWQ(BaseModel):
    w_bit: int
    version: str = "GEMM"
    q_group_size: int = 128
    zero_point: bool = True

    def short_str(self):
        return f"{self.w_bit}-gs{self.q_group_size}"

    def to_dict(self,**kwargs):
        return self.model_dump() # Quantize from AWQ takes dict as input
    

class QuantizationHQQ(BaseModel):
    nbits: int   # Number of bits to quantize to (1,2,3,4,8)
    group_size: int = 64
    
    def __post_init__(self):
        assert self.nbits in [1, 2, 3, 4, 8], "nbits should be one of 1,2,3,4,8"
        
    def short_str(self):
        return f"{self.nbits}"
    
    def to_dict(self, **kwargs):
        return HqqConfig(**self.model_dump())


class QuantizationConfiguration(BaseModel):
    quantization_method: QuantizationMethod
    quantization_config: dict = Field(...)
    base_model: str

    def __init__(self, **data):
        super().__init__(**data)
        # Map pruning_method to configuration
        method_to_config_map = {
            QuantizationMethod.bitsandbytes: QuantizationBitsAndBytesConfiguration,
            QuantizationMethod.gptq: QuantizationGPTQ,
            QuantizationMethod.awq: QuantizationAWQ,
            QuantizationMethod.hqq: QuantizationHQQ,
        }
        config_class = method_to_config_map[self.quantization_method]
        # Convert dict to appropriate config class
        self.quantization_config = config_class(**self.quantization_config)

    @staticmethod
    def parse_yaml(file_path: str) -> "QuantizationConfiguration":
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return QuantizationConfiguration(**data)

    @staticmethod
    def _parse_yaml(yaml_as_string: str) -> "QuantizationConfiguration":
        data = yaml.safe_load(yaml_as_string)
        return QuantizationConfiguration(**data)

    def to_dict(self, tokenizer: AutoTokenizer, **kwargs):
        kwargs["tokenizer"] = tokenizer
        return self.quantization_config.to_dict(**kwargs)

    def short_str(self):
        return f"{self.quantization_method}/{self.quantization_config.short_str()}"


def quantize_model(quantization_config: QuantizationConfiguration):
    model_name = quantization_config.base_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config_dict = quantization_config.to_dict(tokenizer)
    
    if quantization_config.quantization_method == QuantizationMethod.awq:
        try:
            model = AutoAWQForCausalLM.from_pretrained(
                model_name, device_map="cuda", safetensors=False
            )
        except Exception as e: # In case of safetensors=True
            model = AutoAWQForCausalLM.from_pretrained(
                model_name, device_map="cuda", safetensors=True
            )
        model.quantize(tokenizer, quant_config = quantization_config_dict)
        model.device = torch.device("cuda")
        model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config_dict, device_map="cuda"
        )
    
    return model, tokenizer
