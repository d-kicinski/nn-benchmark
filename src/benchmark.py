from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import time
import torch

from models import LazyModel

torch.set_num_threads(1)


class ExperimentProvider:
    def __init__(self, lazy_model: LazyModel, data: Data):
        self._lazy_model = lazy_model
        self._data = data

        self._loss = torch.nn.CrossEntropyLoss()
        self._dummy_labels = torch.zeros((data.train.shape[0],), dtype=torch.long)

        self._model: Optional[torch.nn.Module] = None

    @property
    def name(self) -> str:
        return self._lazy_model.name

    @property
    def data(self) -> Data:
        return self._data

    @data.setter
    def data(self, data: Data):
        self._data = data
        self._dummy_labels = self._dummy_labels.to(data.train.device)

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = self._lazy_model()
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module):
        self._model = model

    def initialize_model(self):
        if self._model is None:
            self._model = self._lazy_model()

    def assign_device(self, device: str):
        self._dummy_labels = self._dummy_labels.to(device)
        self._data.assign_device(device)
        if self._model is None:
            raise ValueError("Explicitly initialize model with `initialize_model` or `model`")
        else:
            self._model = self._model.to(device)

    def clear(self):
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None

    def step_once(self) -> None:
        self.model(self._data.eval)

    def train_step_once(self) -> None:
        out = self.model(self._data.train)
        loss = self._loss(out, self._dummy_labels)
        loss.backward()


@dataclass
class BenchmarkResult:
    name: str
    input_size: int
    forward_step: float
    train_step: float


def _throughput(start, end, steps):
    return 1 / (1e-7 + ((end - start) / steps))


def benchmark(experiment: ExperimentProvider, steps: int) -> BenchmarkResult:
    start_time_step = time.time()
    for i in range(steps):
        experiment.step_once()
    end_time_step = time.time()

    start_time_train = time.time()
    for i in range(steps):
        experiment.train_step_once()
    end_time_train = time.time()

    # start_time_train = 0
    # end_time_train = 1

    return BenchmarkResult(experiment.name,
                           experiment.data.eval.shape[2],
                           _throughput(start_time_step, end_time_step, steps),
                           _throughput(start_time_train, end_time_train, steps))


class Runner:
    RUN_DELAY_SECONDS = 10.0

    def __init__(self, device: str):
        self._device = device

    def run_experiments(self, experiments: List[ExperimentProvider], verbose: bool = True) \
            -> List[BenchmarkResult]:
        results: List[BenchmarkResult] = []
        for e in experiments:
            e.initialize_model()
            e.assign_device(self._device)
            e.train_step_once()  # dry run for sanity-check
            time.sleep(Runner.RUN_DELAY_SECONDS)
            r = benchmark(e, steps)
            e.clear()
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

    def assign_device(self, device: str):
        self.eval = self.eval.to(device)
        self.train = self.train.to(device)


if __name__ == '__main__':
    steps = 10
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
            ExperimentProvider(LazyModel("MobileNetV1(1.0)"), data),
            ExperimentProvider(LazyModel("MobileNetV1(0.5)"), data),
            ExperimentProvider(LazyModel("MobileNetV2(1.0)"), data),
            ExperimentProvider(LazyModel("MobileNetV2(0.5)"), data),
            ExperimentProvider(LazyModel("MobileNetV3(Large)"), data),
            ExperimentProvider(LazyModel("MobileNetV3(Small)"), data),
            ExperimentProvider(LazyModel("ShuffleNetV2(1.0)"), data),
            ExperimentProvider(LazyModel("ShuffleNetV2(0.5)"), data),
            ExperimentProvider(LazyModel("ResNet18"), data),
            ExperimentProvider(LazyModel("ResNet34"), data),
            ExperimentProvider(LazyModel("ResNet50"), data),
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

    with Path("../results/benchmark_cpu.json").open("w") as fp:
        json.dump(cpu_stats, fp, indent=4)

    if torch.cuda.is_available():
        with Path("../results/benchmark_gpu.json").open("w") as fp:
            json.dump(gpu_stats, fp, indent=4)
