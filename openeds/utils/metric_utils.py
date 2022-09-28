import typing
import numpy as np
import pandas as pd


def unit_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_u = unit_vector(left)
    right_u = unit_vector(right)
    return np.arccos(np.clip(np.dot(left_u, right_u), -1.0, 1.0))


def per_batch_angle(
        y_true: typing.Union[np.ndarray, pd.DataFrame],
        prediction: typing.Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    esum = np.einsum('ij,ij->i', y_true, prediction)
    left_norm = np.linalg.norm(y_true, axis=1)
    right_norm = np.linalg.norm(prediction, axis=1)
    cos_angle = esum / (left_norm * right_norm)
    return np.sum(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
