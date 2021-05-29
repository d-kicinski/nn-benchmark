from dataclasses import dataclass
from typing import List

import time
import torch

from models.mobilenetv1 import MobileNetV1


class ExperimentProvider:
    def __init__(self, name: str, model: torch.nn.Module, data: torch.Tensor, device: torch.device):
        self._name = name
        self._model = model.to(device)
        self._data = data.to(device)
        self._device = device

        self._loss = torch.nn.CrossEntropyLoss()
        self._dummy_labels = torch.zeros((data.shape[0],), device=device, dtype=torch.long)

    @property
    def name(self) -> str:
        return self._name

    def step_once(self) -> None:
        self._model(self._data)

    def train_step_once(self) -> None:
        out = self._model(self._data)
        loss = self._loss(out, self._dummy_labels)
        loss.backward()


@dataclass
class BenchmarkResult:
    name: str
    forward_step: float
    train_step: float


def _mean_time(start, end, steps):
    return (end - start) / steps


def benchmark(experiment: ExperimentProvider, steps: int) -> BenchmarkResult:
    start_time_step = time.time()
    for i in range(steps):
        experiment.step_once()
    end_time_step = time.time()

    start_time_train = time.time()
    for i in range(steps):
        experiment.train_step_once()
    end_time_train = time.time()

    return BenchmarkResult(experiment.name,
                           _mean_time(start_time_step, end_time_step, steps),
                           _mean_time(start_time_train, end_time_train, steps))


if __name__ == '__main__':
    steps = 10
    input_size = 224
    data = torch.randn((32, 3, input_size, input_size))

    models = [
        ExperimentProvider("MobileNet(1.0), cpu", MobileNetV1(3, 1000, 1.0), data,
                           torch.device("cpu")),
        ExperimentProvider("MobileNet(1.0), gpu", MobileNetV1(3, 1000, 1.0), data,
                           torch.device("cuda")),

        ExperimentProvider("MobileNet(0.5), cpu", MobileNetV1(3, 1000, 0.5), data,
                           torch.device("cpu")),
        ExperimentProvider("MobileNet(0.5), gpu", MobileNetV1(3, 1000, 0.5), data,
                           torch.device("cuda")),
    ]

    for m in models:
        result = benchmark(m, steps)
        print(f"{result.name}: {result.forward_step:.4f} {result.train_step:.4f}")
