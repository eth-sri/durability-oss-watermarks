from strenum import StrEnum
from .scheme_config import (
    WeigthsmarkWatermarkConfiguration,
    KGWWatermarkConfiguration,
    AARWatermarkConfiguration,
    KTHWatermarkConfiguration,
    RLWatermarkConfiguration,
    UnremovableWatermarkConfiguration,
)


class WatermarkType(StrEnum):
    AAR = "aar"
    KGW = "kgw"
    KTH = "kth"
    UNREMOVABLE = "unremovable"
    UNFORGABLE = "unforgeable"
    WEIGHTSMARK = "weightsmark"
    RESAMPLE = "resample"
    RL = "rl"

    def get_config(self):
        if self.value == "kgw":
            return KGWWatermarkConfiguration
        elif self.value == "weightsmark":
            return WeigthsmarkWatermarkConfiguration
        elif self.value == "aar":
            return AARWatermarkConfiguration
        elif self.value == "kth":
            return KTHWatermarkConfiguration
        elif self.value == "rl":
            return RLWatermarkConfiguration
        elif self.value == "unremovable":
            return UnremovableWatermarkConfiguration
        else:
            raise NotImplementedError("Watermark type not implemented")
