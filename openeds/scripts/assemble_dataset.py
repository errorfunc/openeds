import typing
import pathlib
from collections import ChainMap
from argparse import ArgumentParser

from dataclasses import (
    dataclass,
    asdict,
)

import pandas as pd
from tqdm.auto import tqdm

from ..utils import (
    save_frame,
    strip_and_split,
    check_path,
    glob_by_extension,
)


@dataclass(frozen=True)
class GazeTrackingRecord:
    seq_id: str
    img_id: str

    image_path: pathlib.Path
    x: float
    y: float
    z: float
    split: str


def get_image_split(record: pathlib.Path) -> str:
    return record.parts[-4]


def transform_target(file_path: pathlib.Path) -> typing.Dict[str, typing.Dict]:
    stem = file_path.stem
    record = {stem: dict()}

    with open(str(file_path)) as target_file:
        for line in target_file:
            img_id, *vector = strip_and_split(line)
            record[stem].update({img_id: vector})

    return record


def assemble_target(files: typing.List) -> ChainMap:
    return ChainMap(*[transform_target(_file) for _file in files])


def assemble_dataset(dataset_root_path: str) -> pd.DataFrame:
    root_path = pathlib.Path(dataset_root_path)
    if not check_path(root_path):
        raise ValueError(f'wrong root path: {root_path}')

    images = glob_by_extension(root_path, 'png')
    labels = glob_by_extension(root_path, 'txt')
    labels = assemble_target(labels)
    dataset = list()

    for image in tqdm(images, total=len(images), desc='dataset assembling'):
        seq_id = image.parent.name
        img_id = image.stem

        split = get_image_split(image)
        target = labels[seq_id][img_id]

        record = GazeTrackingRecord(
            seq_id,
            img_id,
            image.resolve(),
            *target,
            split=split
        )
        dataset.append(asdict(record))
    return pd.DataFrame(dataset)


def run() -> None:
    parser = ArgumentParser()
    parser.add_argument('-in', '--dataset_path', type=str, help='path to the dataset directory')
    parser.add_argument(
        '-out',
        '--destination_path',
        type=str,
        help='desired path for the dataset (should be csv file)'
    )
    args = parser.parse_args()

    save_frame(assemble_dataset(args.dataset_path), args.destination_path)
    return None
