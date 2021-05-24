import argparse
import os
import pprint
from pathlib import Path

import torch

from plt2pix import plt2pix

def parse_args():
    desc = "plt2pix: Line Art Colorization using palettes"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model', type=str, default='tag2pix', choices=['tag2pix', 'senet', 'resnext', 'catconv', 'catall', 'adain', 'seadain'],
                        help='Model Types. (default: tag2pix == SECat)')

    parser.add_argument('--cpu', action='store_true', help='If set, use cpu only')

    parser.add_argument('--input_size', type=int, default=256, help='Width / Height of input image (must be rectangular)')
    parser.add_argument('--load', type=str, default="", help='Path to load network weights (if non-empty)')
    parser.add_argument('--color_space', type=str, default='rgb', choices=['lab', 'rgb', 'hsv'], help='color space of images')
    parser.add_argument('--layers', type=int, nargs='+', default=[12,8,5,5],
        help='Block counts of each U-Net Decoder blocks of generator. The first argument is count of bottom block.')

    parser.add_argument('--use_relu', action='store_true', help='Apply ReLU to colorFC')
    parser.add_argument('--no_bn', action='store_true', help='Remove every BN Layer from Generator')
    parser.add_argument('--no_guide', action='store_true', help='Remove guide decoder from Generator. If set, Generator will return same G_f: like (G_f, G_f)')

    # Palette based colorization
    parser.add_argument('--palette_num', type=int, default=5, help='Number of palette colors')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    model = plt2pix(args)

    # Example colorization
    import numpy as np
    from PIL import Image

    img = Image.open('testset/000099.png')
    # (R, G, B, R, G, B, ..., R, G, B)
    palette = np.array([201, 207, 216, 83, 127, 164, 124, 156, 189, 116, 132, 146, 124, 140, 148])

    output = model.colorize(img, palette)

    output = Image.fromarray(output)
    output.save('test_result.png')


if __name__ == '__main__':
    main()
