import argparse
from pathlib import Path

from src.generator import Generator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int,
                        help='The seed used to generate the dataset (default: None).')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Recursively iterate through the input path (default: False).')
    parser.add_argument('--flatten-output', action='store_true',
                        help='Store all images in the output path "gt" and "source" folder instead of mirroring the '
                             'folder structure of the input path (default: False). Only useful with -r.')
    parser.add_argument('--input-path', type=str, default='data/in',
                        help='The path from where to read the images (default: data/in).')
    parser.add_argument('--output-path', type=str, default='data/out',
                        help='The path to where to save the images (default: data/out). The ground truth and blurred '
                             'imaged will be saved to the subdirectories "gt" and "source", respectively.')
    parser.add_argument('--separate-by-blur', action='store_true',
                        help='Store each level of blur in a separate folder (default: False).')
    parser.add_argument('-d', '--divisible-by', type=int, default=None,
                        help='Make crops divisible by this number.')
    parser.add_argument('--size', type=int, default=None,
                        help='Make crops exactly this number. Ignores -d and default max size.')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    generator = Generator(Path(arguments.input_path), Path(arguments.output_path))
    generator.generate(seed=arguments.seed, recursive=arguments.recursive, flatten_output=arguments.flatten_output,
                       max_size=384, blur_levels=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 5], num_crops=2,
                       separate_blur=arguments.separate_by_blur, divisible_by=arguments.divisible_by, size=arguments.size)
