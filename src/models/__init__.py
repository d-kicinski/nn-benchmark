import torch

from .mobilenetv1 import *
from .mobilenetv2 import *
from .mobilenetv3 import *
from .resnet import *
from .shufflenetv2 import *
from .mnasnet import *


class LazyModel:
    def __init__(self, name: str):
        self._name = name

    def __call__(self) -> torch.nn.Module:
        return self._construct_model_by_name()

    @property
    def name(self) -> str:
        return self._name

    def _construct_model_by_name(self) -> torch.nn.Module:
        if self._name == "MobileNetV1(1.0)":
            return MobileNetV1(3, 1000, alpha=1.0)
        elif self._name == "MobileNetV1(0.5)":
            return MobileNetV1(3, 1000, alpha=0.5)
        if self._name == "MobileNetV2(1.0)":
            return MobileNetV2(width_mult=1.0)
        elif self._name == "MobileNetV2(0.5)":
            return MobileNetV2(width_mult=0.5)
        if self._name == "MobileNetV3(Large)":
            return mobilenet_v3_large()
        elif self._name == "MobileNetV3(Small)":
            return mobilenet_v3_small()
        if self._name == "ShuffleNetV2(0.5)":
            return shufflenet_v2_x0_5()
        elif self._name == "ShuffleNetV2(1.0)":
            return shufflenet_v2_x1_0()
        elif self._name == "ResNet18":
            return resnet18()
        elif self._name == "ResNet34":
            return resnet34()
        elif self._name == "ResNet50":
            return resnet50()
        elif self._name == "MNASNet(1.0)":
            return mnasnet1_0()
        elif self._name == "MNASNet(0.5)":
            return mnasnet0_5()
        else:
            raise ValueError(f"Model {self._name} is unknown!")
