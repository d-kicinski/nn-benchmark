from typing import Dict, List

from pytablewriter import MarkdownTableWriter
from models import LazyModel
import torchinfo
from torchinfo import ModelStatistics

model_list = [
    "MNASNet(0.5)",
    "MNASNet(1.0)",
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


def get_memory_stats(image_size: int) -> Dict[str, ModelStatistics]:
    memory_stats: Dict[str, ModelStatistics] = {}
    for m in model_list:
        memory_stats[m] = torchinfo.summary(LazyModel(m)(),
                                            input_size=(1, 3, image_size, image_size),
                                            verbose=0)
    return memory_stats


def memory_table(memory_stats: Dict[str, ModelStatistics]) -> MarkdownTableWriter:
    matrix = []
    for name, s in memory_stats.items():
        matrix.append([name,
                       s.total_params / 1e6,
                       s.total_mult_adds / 1e6,
                       ModelStatistics.to_bytes(s.total_params),
                       ModelStatistics.to_bytes(s.total_output) + ModelStatistics.to_bytes(
                           s.total_params),
                       ])
    return MarkdownTableWriter(
        headers=["model", "#params(1e6)", "#mult-adds(1e6)", "disk size(Mb)", "RAM size(Mb)"],
        value_matrix=matrix
    )


def write_table(image_size: int):
    writer = memory_table(get_memory_stats(image_size))
    writer.write_table()


if __name__ == '__main__':
    # sizes = [112, 140, 160, 192, 224]
    write_table(160)
