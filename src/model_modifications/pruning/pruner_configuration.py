import yaml
from pydantic import BaseModel, Field
from strenum import StrEnum


class PruningMethod(StrEnum):
    sparsegpt = "sparseGPT"
    gblm = "gblm"
    wanda = "wanda"


class MagnitudePruningConfiguration(BaseModel):
    n_samples: int = 128
    seed: int = 0
    prune_n: int = 0
    prune_m: int = 0
    sparsity_ratio: float = 0.5

    def __post_init__(self):
        assert (self.prune_n == 0 and self.prune_m == 0) or (
            self.prune_n != 0 and self.prune_m != 0
        ), "Both prune_n and prune_m should be set to 0 or non-zero"

    def short_str(self):
        main_str = f"n_samples={self.n_samples}, seed={self.seed},"
        if self.prune_n == 0:
            return f"{main_str} sparsity_ratio={self.sparsity_ratio}"
        return f"{main_str} sparsity_ratio={self.prune_n}:{self.prune_m}"


class GBLMPruningConfiguration(MagnitudePruningConfiguration):
    gradient_norm: str = "l1"
    scale: int = 100
    gradient_inv: bool = False
    use_variant: bool = False
    use_cache: bool = False
    base_model: str 

    def short_str(self):
        main_str = super().short_str()
        return f"{main_str}, gradient_norm={self.gradient_norm}, scale={self.scale}, gradient_inv={self.gradient_inv}, use_variant={self.use_variant}"

class WandaPruningConfiguration(MagnitudePruningConfiguration):
    use_variant: bool = False
    
    def short_str(self):
        main_str = super().short_str()
        return f"{main_str}, use_variant={self.use_variant}"


class SparseGPTPruningConfiguration(MagnitudePruningConfiguration):
    pass

    def short_str(self):
        main_str = super().short_str()
        return f"{main_str}"


class PruningConfiguration(BaseModel):
    pruning_method: PruningMethod
    pruning_method_config: dict = Field(...)
    base_model: str
    dtype: str = "float32"

    def __init__(self, **data):
        super().__init__(**data)
        # Map pruning_method to configuration
        method_to_config_map = {
            PruningMethod.sparsegpt: SparseGPTPruningConfiguration,
            PruningMethod.gblm: GBLMPruningConfiguration,
            PruningMethod.wanda: WandaPruningConfiguration,
        }
        config_class = method_to_config_map[self.pruning_method]
        # Convert dict to appropriate config class
        self.pruning_method_config = config_class(**self.pruning_method_config)

    @staticmethod
    def parse_yaml(file_path: str) -> "PruningConfiguration":
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return PruningConfiguration(**data)
    
    @staticmethod
    def _parse_yaml(yaml_as_string: str) -> "PruningConfiguration":
        data = yaml.safe_load(yaml_as_string)
        return PruningConfiguration(**data)
    
    def short_str(self):
        return f"{self.pruning_method}/{self.pruning_method_config.short_str()}"


# Prune model
class Pruner:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def prune_model(self, model, **kwargs):
        raise NotImplementedError
