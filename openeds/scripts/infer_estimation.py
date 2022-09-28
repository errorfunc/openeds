import pathlib
from itertools import chain
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from ..resnet import grey_resnet18
from ..dataset import GazeEstimationDataset

from ..utils import (
    cast_to_numpy,
    read_frame,
    save_frame,
)


def run_inference(
        checkpoint: pathlib.Path,
        test_loader: DataLoader,
        target_device: torch.device
) -> pd.DataFrame:
    ids_acc = list()
    predictions_acc = list()

    model = grey_resnet18()
    model.load_state_dict(torch.load(checkpoint))
    model.to(target_device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch_ids, img_data = batch
            prediction = model(img_data.to(target_device))
            ids_acc.append(batch_ids)
            predictions_acc.append(cast_to_numpy(prediction))

    whole_ids = list(chain.from_iterable(ids_acc))
    whole_predictions = np.concatenate(predictions_acc, axis=0)
    x, y, z = np.split(whole_predictions, 3, axis=1)
    return pd.DataFrame({'internal_id': whole_ids, 'x': x.flatten(), 'y': y.flatten(), 'z': z.flatten()})


def run() -> None:
    parser = ArgumentParser()
    parser.add_argument('-in', '--dataset_path', type=str, help='path to the dataset (should be csv file)')
    parser.add_argument(
        '-out',
        '--destination_path',
        type=str,
        help='desired path for inference results (should be csv file)'
    )
    parser.add_argument('-c', '--checkpoint_path', type=str, help='path to the model checkpoint')
    parser.add_argument('-bs', '--batch_size', type=int, default=1600, help='batch size for inference')
    parser.add_argument('-dev', '--device', type=str, default='cuda', help='device to process')
    args = parser.parse_args()

    device = torch.device(args.device)
    data = read_frame(args.dataset_path)

    dataset = DataLoader(
        GazeEstimationDataset(data, is_test=True),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4
    )

    predictions = run_inference(args.checkpoint_path, dataset, device)
    save_frame(predictions, args.destination_path)
    return None