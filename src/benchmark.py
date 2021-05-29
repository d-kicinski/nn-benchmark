import itertools
from dataclasses import dataclass
from typing import List

import time
import torch

from models import MobileNetV1, MobileNetV2


class ExperimentProvider:
    def __init__(self, name: str, model: torch.nn.Module, data: torch.Tensor):
        self._name = name
        self._model = model
        self._data = data

        self._loss = torch.nn.CrossEntropyLoss()
        self._dummy_labels = torch.zeros((data.shape[0],), dtype=torch.long)

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, data: torch.Tensor):
        self._data = data
        self._dummy_labels = self._dummy_labels.to(data.device)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module):
        self._model = model

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


class Runner:
    def __init__(self, device: str):
        self._device = device

    def run_experiments(self, experiments: List[ExperimentProvider]):
        for e in experiments:
            e.model = e.model.to(self._device)
            e.data = e.data.to(self._device)
            result = benchmark(e, steps)
            print(f"{result.name}: {result.forward_step:.4f} {result.train_step:.4f}")


if __name__ == '__main__':
    steps = 5
    input_size = 224
    data = torch.randn((32, 3, input_size, input_size))

    cpu_runner = Runner("cpu")
    gpu_runner = Runner("cuda")

    experiments = [
        ExperimentProvider("MobileNet(1.0)", MobileNetV1(3, 1000, 1.0), data),
        ExperimentProvider("MobileNet(0.5)", MobileNetV1(3, 1000, 0.5), data),
        ExperimentProvider("MobileNetV2(1.0)", MobileNetV2(width_mult=1.0), data),
        ExperimentProvider("MobileNetv2(0.5)", MobileNetV2(width_mult=0.5), data)
    ]

    gpu_runner.run_experiments(experiments)




