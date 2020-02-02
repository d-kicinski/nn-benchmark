import argparse
from time import time
import numpy as np

import torch
from facenet_pytorch import MTCNN

BATCH_SIZE: int = 1
IMAGES_TO_PROCESS: int = 1
IMAGE_SIZE: int = 256
MARGIN: int = 10
USE_CPU: bool = True


def benchmark_mtcnn(images_to_process: int,
                    image_size: int,
                    use_cpu: bool):
    if use_cpu:
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Running processing on device: {}".format(device))

    mtcnn = MTCNN(image_size=image_size, min_face_size=20,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709, device=device)
    images = [np.random.randint(low=0, high=256, size=(256, 256, 3))
              for _ in range(images_to_process)]

    time_start = time()
    mtcnn(images)
    time_end = time()
    print(f"Processing {images_to_process} took {time_start-time_end:.4f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MTCNN")
    parser.add_argument("--images_to_process", type=int, default=IMAGES_TO_PROCESS)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--margin", type=int, default=MARGIN)
    parser.add_argument("--cpu", action="store_true", default=USE_CPU)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark_mtcnn(args.images_to_process, args.image_size, args.margin, args.cpu)
