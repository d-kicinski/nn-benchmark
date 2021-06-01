from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import time
import torch

from models import MobileNetV1, MobileNetV2, mobilenet_v3_small, mobilenet_v3_large
from models import resnet18, resnet34, resnet50
from models import shufflenet_v2_x0_5, shufflenet_v2_x1_0

torch.set_num_threads(4)


class ExperimentProvider:
    def __init__(self, name: str, model: torch.nn.Module, data: Data):
        self._name = name
        self._model = model
        self._data = data

        self._loss = torch.nn.CrossEntropyLoss()
        self._dummy_labels = torch.zeros((data.train.shape[0],), dtype=torch.long)

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> Data:
        return self._data

    @data.setter
    def data(self, data: Data):
        self._data = data
        self._dummy_labels = self._dummy_labels.to(data.train.device)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module):
        self._model = model

    def step_once(self) -> None:
        self._model(self._data.eval)

    def train_step_once(self) -> None:
        out = self._model(self._data.train)
        loss = self._loss(out, self._dummy_labels)
        loss.backward()


@dataclass
class BenchmarkResult:
    name: str
    input_size: int
    forward_step: float
    train_step: float


def _throughput(start, end, steps):
    return 1 / (1e-7 + (end - start) / steps)


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
                           experiment.data.eval.shape[2],
                           _throughput(start_time_step, end_time_step, steps),
                           _throughput(start_time_train, end_time_train, steps))


class Runner:
    RUN_DELAY_SECONDS = 3.0

    def __init__(self, device: str):
        self._device = device

    def run_experiments(self, experiments: List[ExperimentProvider], verbose: bool = True) \
            -> List[BenchmarkResult]:
        results: List[BenchmarkResult] = []
        for e in experiments:
            time.sleep(Runner.RUN_DELAY_SECONDS)
            e.model = e.model.to(self._device)
            e.data = Data(e.data.eval.to(self._device), e.data.train.to(self._device))
            r = benchmark(e, steps)
            results.append(r)
            if verbose:
                print(f"{r.name}: {r.forward_step:.4f} {r.train_step:.4f}")
        return results


def update_stats_(stats: Dict[str, Dict[int, float]], results: List[BenchmarkResult]):
    for r in results:
        stats[r.name]["forward"][r.input_size] = r.forward_step
        stats[r.name]["backward"][r.input_size] = r.train_step


@dataclass
class Data:
    eval: torch.Tensor
    train: torch.Tensor


if __name__ == '__main__':
    steps = 3
    batch_size = 8
    sizes = [64, 124, 160, 224]

    datas = [
        Data(
            eval=torch.randn((1, 3, input_size, input_size)),
            train=torch.randn((batch_size, 3, input_size, input_size))
        ) for input_size in sizes
    ]

    cpu_runner = Runner("cpu")
    gpu_runner = Runner("cuda")

    cpu_stats = defaultdict(lambda: defaultdict(dict))
    gpu_stats = defaultdict(lambda: defaultdict(dict))

    for data, size in zip(datas, sizes):
        print(f"=== {size} ===")
        experiments = [
            ExperimentProvider("MobileNet(1.0)", MobileNetV1(3, 1000, 1.0), data),
            ExperimentProvider("MobileNet(0.5)", MobileNetV1(3, 1000, 0.5), data),
            ExperimentProvider("MobileNetV2(1.0)", MobileNetV2(width_mult=1.0), data),
            ExperimentProvider("MobileNetV2(0.5)", MobileNetV2(width_mult=0.5), data),
            ExperimentProvider("MobileNetV3(Large)", mobilenet_v3_large(), data),
            ExperimentProvider("MobileNetV3(Small)", mobilenet_v3_small(), data),
            ExperimentProvider("ShuffleNetV2(1.0)", shufflenet_v2_x1_0(), data),
            ExperimentProvider("ShuffleNetV2(0.5)", shufflenet_v2_x0_5(), data),
            ExperimentProvider("ResNet18", resnet18(), data),
            ExperimentProvider("ResNet34", resnet34(), data),
            ExperimentProvider("ResNet50", resnet50(), data),
        ]

        if torch.cuda.is_available():
            print("Running on CUDA")
            results = gpu_runner.run_experiments(experiments)
            update_stats_(gpu_stats, results)
            print()

        print("Running on CPU")
        results = cpu_runner.run_experiments(experiments)
        update_stats_(cpu_stats, results)

        print()
        print()

    with Path("benchmark_cpu.json").open("w") as fp:
        json.dump(cpu_stats, fp, indent=4)

    with Path("benchmark_gpu.json").open("w") as fp:
        json.dump(cpu_stats, fp, indent=4)
