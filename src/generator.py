import random
from pathlib import Path

import magic
import numpy as np
from PIL import Image, ImageFilter, ImageChops
from scipy.ndimage import gaussian_filter

from . import cvl_xml_parser
from . import mask_generators


def save_image(image: Image, filename: str) -> None:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    image.save(filename)


class Generator:
    supported_files = {
        'image/jpeg': 'jpg',
        'image/tiff': 'tiff',
        'image/bmp': 'bmp',
    }

    bounding_box_names = {
        0: 'computer',
        1: 'handwritten',
    }

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.output_ground_truth_path = output_path.joinpath('gt')
        self.output_source_path = output_path.joinpath('source')

    def randfloat(self, min: float = 0, max: float = 1) -> float:
        return min + random.random() * (max - min)

    def generate(self, seed: None | int | float | str | bytes | bytearray = None, recursive: bool = False,
                 flatten_output=False, max_size: int = 256, blur_levels: list[float | int] = (3, 6), num_crops: int = 2,
                 separate_blur=False, divisible_by=None, size=None) -> None:
        random.seed(seed)

        files = list(self.input_path.glob('*'))

        while len(files) != 0:
            file = files.pop()
            if flatten_output:
                relative_file_name_without_extension = file.stem
            else:
                relative_file_name_without_extension = file.relative_to(self.input_path).with_suffix('')

            if file.is_dir():
                if recursive:
                    files.extend(file.glob('*'))
                continue

            mime_type = magic.from_file(file, mime=True)
            if mime_type not in self.supported_files.keys():
                continue

            print(f'Processing file {file}')

            xml_attribute_file = file.parent.joinpath(f'xml/{file.stem}_attributes.xml')
            bounding_boxes = None
            if xml_attribute_file.exists():
                bounding_boxes = cvl_xml_parser.get_text_bounding_boxes(xml_attribute_file)

            img = Image.open(file)

            output_source_path = self.output_source_path.joinpath(
                f'{"{}"}{relative_file_name_without_extension}{"{}"}.{self.supported_files[mime_type]}')
            output_gt_path = self.output_ground_truth_path.joinpath(
                f'{"{}"}{relative_file_name_without_extension}{"{}"}.{self.supported_files[mime_type]}')

            sigma = random.choice(blur_levels)

            if bounding_boxes is None:
                try:
                    source, gt = self.create_image_and_mask(img, sigma=sigma, max_size=max_size, divisible_by=divisible_by, _size=size)
                except AttributeError:
                    print('Image is too small for the configured size')
                    continue
                save_image(source, str(output_source_path).format('', ''))
                save_image(gt, str(output_gt_path).format('', ''))
            else:
                bounding_box_index = 0
                for bounding_box in bounding_boxes:
                    cropped_img = img.crop((*bounding_box[0], *bounding_box[3]))

                    if cropped_img.size[0] < 16 or cropped_img.size[1] < 16:
                        print(f'Cropped image is too small. '
                              f'Bounding box "{self.bounding_box_names[bounding_box_index]}": {bounding_box}.')
                        continue

                    for i in range(num_crops):
                        try:
                            source, gt = self.create_image_and_mask(cropped_img, sigma=sigma, max_size=max_size, divisible_by=divisible_by, _size=size)
                        except AttributeError:
                            print('Image is too small for the configured size')
                            continue
                        # Only check for white crops in computer dataset
                        if bounding_box_index == 0 and not self.check_quality(source):
                            print(f'Image does not meet quality criteria.')
                            continue

                        folder_prefix = f'{self.bounding_box_names[bounding_box_index]}/'
                        if separate_blur:
                            folder_prefix += f'{sigma}/'
                        save_image(source, str(output_source_path).format(folder_prefix, f'_{i}'))
                        save_image(gt, str(output_gt_path).format(folder_prefix, f'_{i}'))
                    bounding_box_index += 1

    def check_quality(self, image: Image, white_level_threshold: float = 0.90) -> bool:
        white_level = ((np.asarray(image) > 250).sum() /
                       (image.width * image.height * len(image.getbands()))) < white_level_threshold

        return white_level

    def create_image_and_mask(self, img: Image, sigma: float, max_size: int, divisible_by: int | None,
                              _size: int | None) -> tuple[Image, Image]:
        """
        :raises AttributeError: if the given image is too small for the given _size
        """
        size = min(max_size, *img.size)
        if divisible_by is not None:
            size = divisible_by * (size // divisible_by)
        if _size is not None:
            size = _size
            if size > min(*img.size):
                raise AttributeError('Image size is too small for the configured size.')

        rand_x = random.randint(0, img.size[0] - size)
        rand_y = random.randint(0, img.size[1] - size)
        _img = img.crop((rand_x, rand_y, rand_x + size, rand_y + size))

        blurred = Image.fromarray(gaussian_filter(np.asarray(_img.copy()), sigma=(sigma, sigma, 0), mode='mirror'))

        mask_img = self.get_random_mask(blurred.size)

        if self.randfloat(0, 1) > 0.9:
            mask_img = ImageChops.subtract_modulo(mask_img, self.get_random_mask(blurred.size))

        _mask_img = mask_img.copy().filter(ImageFilter.GaussianBlur(2))
        return Image.composite(_img, blurred, _mask_img), mask_img

    def get_random_mask(self, size: tuple[int, int]) -> Image:
        choice = random.randint(0, 6)
        match choice:
            case 1:
                t = self.randfloat(0.1, 0.9)
                mask_img = mask_generators.horizontal_mask(size, t0=t, t1=t, w0=1, w1=1,
                                                           invert=bool(random.getrandbits(1)))
            case 2:
                mask_img = mask_generators.horizontal_mask(size, t0=self.randfloat(0.1, 0.9),
                                                           t1=self.randfloat(0.1, 0.9), w0=1, w1=1,
                                                           invert=bool(random.getrandbits(1)))
            case 3:
                mask_img = mask_generators.horizontal_mask(size, t0=self.randfloat(0.1, 0.9),
                                                           t1=self.randfloat(0.1, 0.9),
                                                           w0=self.randfloat(0.1, 0.4),
                                                           w1=self.randfloat(0.1, 0.4),
                                                           invert=bool(random.getrandbits(1)))
            case 4:
                mask_img = mask_generators.vertical_mask(size, t0=self.randfloat(0.1, 0.9),
                                                         t1=self.randfloat(0.1, 0.9), w0=1, w1=1,
                                                         invert=bool(random.getrandbits(1)))
            case 5:
                mask_img = mask_generators.vertical_mask(size, t0=self.randfloat(0.1, 0.9),
                                                         t1=self.randfloat(0.1, 0.9),
                                                         w0=self.randfloat(0.1, 0.4),
                                                         w1=self.randfloat(0.1, 0.4),
                                                         invert=bool(random.getrandbits(1)))
            case 6:
                mask_img = mask_generators.ellipse_mask(size, x=self.randfloat(0.1, 0.9),
                                                        y=self.randfloat(0.1, 0.9),
                                                        rx=self.randfloat(0.1, 0.4),
                                                        ry=self.randfloat(0.1, 0.4),
                                                        invert=bool(random.getrandbits(1)),
                                                        rotation=self.randfloat(-180, 180))
            case _:
                t = self.randfloat(0.1, 0.9)
                mask_img = mask_generators.vertical_mask(size, t0=t, t1=t, w0=1, w1=1,
                                                         invert=bool(random.getrandbits(1)))
        return mask_img
