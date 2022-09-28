import json
import pathlib
from collections import defaultdict
from argparse import ArgumentParser

import numpy as np


def make_ensemble(path: str, submit_path: str) -> None:
    submit_prediction = defaultdict(lambda: defaultdict(list))
    submissions = list(pathlib.Path(path).glob('*.json'))

    for submission_path in submissions:
        with open(submission_path, 'r') as submission_file:
            submission = json.load(submission_file)
            for user_key in submission.keys():
                for frame_key in submission[user_key].keys():
                    submit_prediction[user_key][frame_key].append(submission[user_key][frame_key])

    for user_key in submit_prediction.keys():
        for frame_key in submit_prediction[user_key].keys():
            submit_prediction[user_key][frame_key] = np.mean(submit_prediction[user_key][frame_key], axis=0)

    with open(submit_path, 'w') as submit_file:
        json.dump(submit_prediction, submit_file)
    return None


def run() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        '-in',
        '--submit_path',
        type=str,
        help='path to the directory containing submits'
    )
    parser.add_argument(
        '-out',
        '--destination_path',
        type=str,
        help='desired path for results (should be json file)'
    )

    args = parser.parse_args()
    make_ensemble(args.submit_path, args.destination_path)
    return None
