import math
import typing
from collections import defaultdict

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from .utils import (
    gaze_aug,
    image_to_tensor,
    make_internal_id,
)


def convert_target(data: pd.DataFrame) -> typing.Dict[str, typing.Dict[str, torch.Tensor]]:
    required_fields = ['seq_id', 'img_id']
    vector_fields = ['x', 'y', 'z']
    all_fields = required_fields + vector_fields

    fields = data[all_fields].copy()
    result = defaultdict(dict)
    for _, row in tqdm(fields.iterrows(), total=len(fields), desc='target transformation'):
        vector = row[vector_fields].values.astype(np.float32)
        result[row.seq_id].update({row.img_id: vector})
    return result


class GazeEstimationDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            is_test: bool = False,
            aug_prob: float = 0.5
    ) -> None:
        self.data = data.copy()
        self.is_train = not is_test
        self.is_test = is_test

        self.aug_prob = aug_prob
        self.aug = gaze_aug(p=self.aug_prob)

        if not is_test:
            self.labels = convert_target(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> typing.Tuple:
        target = None
        seq_id, img_id, image_path = self.data.iloc[index, :3]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if self.is_train:
            target = self.labels[seq_id][img_id]

        if self.is_train and not math.isclose(self.aug_prob, 0.0, abs_tol=1e-2):
            data = {'image': image, 'gaze_vector': target}
            augmented = self.aug(**data)
            image = augmented['image']
            target = augmented['gaze_vector']

        image = image_to_tensor(image)
        unique_id = make_internal_id(seq_id, img_id)

        if self.is_train:
            return unique_id, image, torch.tensor(target, dtype=torch.float)

        return unique_id, image
