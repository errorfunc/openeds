import typing
import pandas as pd

from sklearn.model_selection import GroupKFold


def strip_and_split(record: str) -> typing.Tuple[str, float, float, float]:
    image_id, x, y, z = record.strip().split(',')
    return image_id, float(x), float(y), float(z)


def make_internal_id(seq_id: str, img_id: str) -> str:
    return f'{seq_id}_{img_id}'


def get_gkfold_split(
        dataset: pd.DataFrame,
        n_splits: int = 5,
        split: int = 0
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    seq_ids = dataset.seq_id.values
    group_kfold = GroupKFold(n_splits=n_splits)
    splits = [x for x in group_kfold.split(dataset, None, seq_ids)]
    train_index, test_index = splits[split]
    x_train, x_valid = dataset.iloc[train_index, :], dataset.iloc[test_index, :]
    x_train, x_valid = x_train.reset_index(drop=True), x_valid.reset_index(drop=True)
    print(f'splitting dataset: '
          f'n_folds {n_splits}; fold {split}; train size {x_train.shape}; valid size {x_valid.shape}')
    return x_train, x_valid
