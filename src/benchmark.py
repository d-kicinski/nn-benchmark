from __future__ import annotations

import json
import platform
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import time
import torch
import torch.utils.mobile_optimizer as mobile_optimizer
import onnxruntime as ort

from models import LazyModel

torch.set_num_threads(4)


class Runtime:
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

    @property
    def unsupported(self) -> List[str]:
        return []

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


class OnnxRuntime(Runtime):
    def __init__(self, lazy_model: LazyModel, data: Data):
        super().__init__(lazy_model, data)

        self._ort_session: Optional[ort.InferenceSession] = None
        self._dummy_input = self._data.eval.numpy()

    @property
    def unsupported(self) -> List[str]:
        return ["MobileNetV3(Large)", "MobileNetV3(Small)"]

    def initialize_model(self):
        if self._ort_session is None:
            self._model = self._lazy_model()
            Path("onnx").mkdir(exist_ok=True, parents=True)
            torch.onnx.export(self._model, self._data.eval, f"onnx/{self.name}.onnx",
                              input_names=["input"], output_names=["output"])
            self._ort_session = ort.InferenceSession(f"onnx/{self.name}.onnx")

    def clear(self):
        super(OnnxRuntime, self).clear()
        del self._ort_session
        self._ort_session = None

    def step_once(self) -> None:
        self._ort_session.run(None, {"input": self._dummy_input})

    def train_step_once(self) -> None:
        # ONNX doesn't yet support model training
        pass


class JITRuntime(Runtime):
    def __init__(self, lazy_model: LazyModel, data: Data):
        super().__init__(lazy_model, data)
        self._torchscript_model: Optional[torch.jit.ScriptModule] = None

    def initialize_model(self):
        if self._torchscript_model is None:
            self._model = self._lazy_model()
            Path("jit").mkdir(exist_ok=True, parents=True)
            self._torchscript_model = torch.jit.trace(self._model, self.data.eval,
                                                      check_trace=False)
            self._torchscript_model = mobile_optimizer.optimize_for_mobile(self._torchscript_model)
            self._torchscript_model.save(f"jit/{self.name}.pt")

    def clear(self):
        super(JITRuntime, self).clear()
        del self._torchscript_model
        self._torchscript_model = None

    def step_once(self) -> None:
        self._torchscript_model(self.data.eval)

    def train_step_once(self) -> None:
        # JIT doesn't yet support model training (TODO: Make sure of it)
        pass


@dataclass
class BenchmarkResult:
    name: str
    input_size: int
    forward_step: float
    train_step: float


def _throughput(start, end, steps):
    return 1 / (1e-7 + ((end - start) / steps))


def benchmark(experiment: Runtime, steps: int) -> BenchmarkResult:
    start_time_step = time.time()
    for _ in range(steps):
        experiment.step_once()
    end_time_step = time.time()

    start_time_train = time.time()
    for _ in range(steps):
        experiment.train_step_once()
    end_time_train = time.time()

    return BenchmarkResult(experiment.name,
                           experiment.data.eval.shape[2],
                           _throughput(start_time_step, end_time_step, steps),
                           _throughput(start_time_train, end_time_train, steps))


class Runner:
    RUN_DELAY_SECONDS = 3.0
    TEMP_DIR = "onnx"

    def __init__(self, device: str, onnx: bool = False, steps: int = 10):
        self._device = device
        self._steps = steps
        self._onnx = onnx

    def run_experiments(self, experiments: List[Runtime], verbose: bool = True) \
            -> List[BenchmarkResult]:
        results: List[BenchmarkResult] = []
        for e in experiments:
            e.initialize_model()
            e.assign_device(self._device)
            e.train_step_once()  # dry run for sanity-check
            time.sleep(Runner.RUN_DELAY_SECONDS)
            r = benchmark(e, self._steps)
            e.clear()
            results.append(r)
            if verbose:
                print(f"{r.name}: {r.forward_step:.4f} {r.train_step:.4f}")
        return results


def update_stats_(stats: Dict[str, Dict[str, Dict[int, float]]], results: List[BenchmarkResult]):
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


def run_benchmark():
    steps = 10
    batch_size = 8
    sizes = [64, 124, 160, 224]
    runtime = "jit"

    if runtime == "onnx":
        runtime_cls = OnnxRuntime
    elif runtime == "jit":
        runtime_cls = JITRuntime
    else:
        runtime_cls = Runtime

    model_list = [
        "MobileNetV1(1.0)",
        "MobileNetV1(0.5)",
        "MobileNetV2(1.0)",
        "MobileNetV2(0.5)",
        "MobileNetV3(Large)",
        "MobileNetV3(Small)",
        "ShuffleNetV2(1.0)",
        "ShuffleNetV2(0.5)",
        "ResNet18",
        "ResNet34",
        "ResNet50",
    ]

    dataset = [
        Data(
            eval=torch.randn((1, 3, input_size, input_size)),
            train=torch.randn((batch_size, 3, input_size, input_size))
        ) for input_size in sizes
    ]

    cpu_runner = Runner("cpu", steps)
    gpu_runner = Runner("cuda", steps)

    cpu_stats = defaultdict(lambda: defaultdict(dict))
    gpu_stats = defaultdict(lambda: defaultdict(dict))

    for data, size in zip(dataset, sizes):
        print(f"=== {size} ===")
        experiments = [runtime_cls(LazyModel(m), data) for m in model_list
                       if m not in runtime_cls.unsupported]

        if torch.cuda.is_available():
            print("Running on CUDA")
            results = gpu_runner.run_experiments(experiments)
            update_stats_(gpu_stats, results)
            print()

        print("Running on CPU")
        results = cpu_runner.run_experiments(experiments)
        update_stats_(cpu_stats, results)
        print()

    Path(f"../results").mkdir(exist_ok=True, parents=True)
    with Path(f"../results/benchmark_cpu_{runtime}_{platform.machine()}.json").open("w") as fp:
        json.dump(cpu_stats, fp, indent=4)

    if torch.cuda.is_available():
        with Path("../results/benchmark_gpu.json").open("w") as fp:
            json.dump(gpu_stats, fp, indent=4)


if __name__ == '__main__':
    run_benchmark()
