import argparse
import math

import numpy as np

from stylegan.generate import get_mean_style, sample
from stylegan.model import StyledGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
    parser.add_argument('--seed', type=int, default=0, help='initial seed')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('path', type=str, help='path to checkpoint file')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device

    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.path)['g_running'])
    generator.eval()

    mean_style = get_mean_style(generator, device)

    step = int(math.log(args.size, 2)) - 2

    img = sample(generator, step, mean_style, args.n_row * args.n_col, device)

