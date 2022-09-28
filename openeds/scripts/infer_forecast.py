import json
import pathlib
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def run_forecast(data: pd.DataFrame, submit_path: pathlib.Path, threshold: float) -> None:
    forecast_frames = ('50', '51', '52', '53', '54')

    submit_prediction = dict()
    unique_seqs = pd.unique(data['seq'])

    for seq_num in tqdm(unique_seqs):
        predictions = dict()
        seq = data.query(f'seq == "{seq_num}"')
        gaze_data = seq[['x', 'y', 'z']].values
        gradients, _ = np.gradient(gaze_data)
        deviations = abs(np.mean(gradients[-2:], axis=0))

        if (deviations >= threshold).any():
            last_row = gaze_data[-1]
            for frame in forecast_frames:
                last_row = last_row + gradients[-1]
                predictions[frame] = tuple(last_row)
        else:
            last_rows = gaze_data[-2:]
            for frame in forecast_frames:
                avg_frames = np.mean(last_rows, axis=0)
                last_rows = np.vstack((last_rows[-1], avg_frames))
                predictions[frame] = tuple(avg_frames)
        submit_prediction[seq_num] = predictions

    with open(submit_path, 'w') as submit_file:
        json.dump(submit_prediction, submit_file)
    return None


def run() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        '-in',
        '--dataset_path',
        type=str,
        help='path to the estimations predicted by the network (should be csv file)'
    )
    parser.add_argument(
        '-out',
        '--destination_path',
        type=str,
        help='desired path for submit (should be json file)'
    )
    parser.add_argument(
        '-t'
        '--threshold',
        type=float,
        default=0.0283,
        help='heuristic cutoff for gradient method'
    )
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset_path)
    dataset['seq'], dataset['frame'] = list(
        zip(*dataset['internal_id'].str.split('_'))
    )
    dataset = dataset.drop(columns='internal_id')
    run_forecast(dataset, pathlib.Path(args.destination_path), args.threshold)
    return None
