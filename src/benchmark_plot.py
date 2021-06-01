import json
from pathlib import Path
import matplotlib.pyplot as plt

DATA_PATH = "benchmark_cpu.json"

markers = ["s", "d", "^", "v", "x", "X", "h", "P", "p", "*", "<", ">", "1"]


def plot_benchmark():
    with Path(DATA_PATH).open("r") as fp:
        stats = json.load(fp)
    sizes = set()
    print(len(stats.items()))
    for i, (name, data) in enumerate(stats.items()):
        data = data["forward"]
        x = list(map(int, data.keys()))
        y = list(map(float, data.values()))
        plt.plot(x, y, markers[i], label=f"{name}")
        sizes.update(x)
    plt.xlabel("input size")
    plt.ylabel("throughput (img/s)")
    plt.legend(numpoints=1, loc="upper left", bbox_to_anchor=(1.04, 1))
    plt.tight_layout()
    plt.xticks(list(sizes))
    plt.show()


if __name__ == '__main__':
    plot_benchmark()
