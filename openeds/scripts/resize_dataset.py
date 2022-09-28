import shutil
import pathlib
from argparse import ArgumentParser

import cv2
from tqdm.auto import tqdm

from ..utils import (
    make_directory,
    check_path,
    glob_by_extension,
)


def assemble_path_with_preservation(
        image_path: pathlib.Path,
        target_path: pathlib.Path,
        up_to: int
) -> pathlib.Path:
    if up_to < 0 or len(image_path.parts) < up_to:
        raise ValueError(f'selected preservation level {up_to} is incorrect')

    structure_parts = pathlib.Path(*image_path.parts[-up_to:])
    return target_path / structure_parts


def run() -> None:
    parser = ArgumentParser()
    parser.add_argument('-in', '--dataset_path', type=str, help='path to the original dataset directory')
    parser.add_argument('-out', '--destination_path', type=str, help='path to the rescaled dataset directory')
    parser.add_argument('-f', '--fx', type=float, default=0.4, help='fraction for image resize')
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path)
    destination_path = pathlib.Path(args.destination_path)

    if not check_path(dataset_path):
        raise FileNotFoundError(f'given {dataset_path} doesnt exist')

    all_images = glob_by_extension(dataset_path, 'png')
    all_labels = glob_by_extension(dataset_path, 'txt')

    if not check_path(destination_path):
        make_directory(destination_path)

    for img_path in tqdm(all_images, total=len(all_images), desc='png files'):
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(
            image,
            (0, 0),
            fx=args.fx,
            fy=args.fx,
            interpolation=cv2.INTER_LANCZOS4
        )
        new_path = assemble_path_with_preservation(img_path, destination_path, 4)
        if not check_path(new_path):
            make_directory(new_path.parent)
        cv2.imwrite(str(new_path), image, [cv2.IMWRITE_PNG_STRATEGY_DEFAULT])

    for txt_path in tqdm(all_labels, total=len(all_labels), desc='txt files'):
        new_path = assemble_path_with_preservation(txt_path, destination_path, up_to=3)
        if not check_path(new_path):
            make_directory(new_path.parent)
        shutil.copy(txt_path, new_path)
    return None
