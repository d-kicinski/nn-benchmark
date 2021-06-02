import json
from pathlib import Path
import matplotlib.pyplot as plt

markers = ["s", "d", "^", "v", "x", "X", "h", "P", "p", "*", "<", ">", "1"]

BLACKLIST = ["ResNet18", "ResNet34", "ResNet50"]

def plot_benchmark(input_path: str, output_path: str, run_type: str = "forward", show: bool = True):
    with Path(input_path).open("r") as fp:
        stats = json.load(fp)
    sizes = set()
    for i, (name, data) in enumerate(stats.items()):
        if name in BLACKLIST:
            continue
        data = data[run_type]
        x = list(map(int, data.keys()))
        y = list(map(float, data.values()))
        plt.plot(x, y, markers[i], label=f"{name}")
        sizes.update(x)
    plt.xlabel("input size")
    plt.ylabel("throughput (img/s)")
    plt.legend(numpoints=1, loc="upper left", bbox_to_anchor=(1.04, 1))
    plt.tight_layout()
    plt.xticks(list(sizes))
    plt.savefig(output_path)

    if show:
        plt.show()


if __name__ == '__main__':
    plot_benchmark("../results/benchmark_cpu.json", "../results/cpu_forward.png", "forward")
    plot_benchmark("../results/benchmark_cpu.json", "../results/cpu_train.png", "backward")
    plot_benchmark("../results/benchmark_gpu.json", "../results/gpu_forward.png", "forward")
    plot_benchmark("../results/benchmark_gpu.json", "../results/gpu_train.png", "backward")
