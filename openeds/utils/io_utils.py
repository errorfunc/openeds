import typing
import pathlib
import pandas as pd


def read_frame(frame_path: typing.Union[pathlib.Path, str]) -> pd.DataFrame:
    return pd.read_csv(frame_path, dtype={'seq_id': str, 'img_id': str})


def save_frame(frame: pd.DataFrame, target_path: typing.Union[pathlib.Path, str]) -> None:
    frame.to_csv(target_path, index=False)
