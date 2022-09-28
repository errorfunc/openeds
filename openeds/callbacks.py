import sys
import typing
import logging
import pathlib

from operator import (
    lt,
    gt,
)

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def _recent_k(paths: typing.List[pathlib.Path], k: int = 3) -> typing.List[pathlib.Path]:
    if len(paths) <= k:
        return paths
    return sorted(paths, key=lambda p: p.stat().st_ctime)[-k:]


def _get_checkpoints(path: pathlib.Path, prefix: str) -> typing.List:
    return list(path.glob(f'{prefix}*.pt'))


def _purge_files(paths: typing.List[pathlib.Path]) -> None:
    for checkpoint in paths:
        checkpoint.unlink()
    return None


def _save_checkpoint(
        path: pathlib.Path,
        prefix: str,
        train_loss: float,
        valid_loss: float,
        model: nn.Module
) -> None:
    torch.save(model.state_dict(), path / f'{prefix}_{train_loss:.4f}_{valid_loss:.4f}.pt')
    logging.info(f'train: {train_loss:.4f} | valid: {valid_loss:.4f}')
    return None


def _clean_mess(path: pathlib.Path, prefix: str, best_k: int) -> None:
    checkpoints = _get_checkpoints(path, prefix)
    recent_checkpoints = _recent_k(checkpoints, best_k)
    goners = [x for x in checkpoints if x not in recent_checkpoints]
    return _purge_files(goners)


def _try_to_terminate(summary: typing.Optional[SummaryWriter] = None) -> None:
    if summary:
        summary.close()
    return sys.exit()


class EarlyStopping:
    def __init__(
            self,
            patience: int,
            mode: str,
            path_to_save: pathlib.Path,
            prefix: str
    ) -> None:
        self.patience = patience
        self.mode = gt if mode == 'min' else lt
        self.path_to_save = pathlib.Path(path_to_save)
        if not self.path_to_save.exists():
            self.path_to_save.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

        self.counter = 0
        self.best_training_score = None
        self.best_validation_score = None

    def __call__(self, train_loss: float, val_loss: float, model: nn.Module) -> bool:
        if self.best_validation_score is None:
            self.best_training_score = train_loss
            self.best_validation_score = val_loss
            _save_checkpoint(self.path_to_save, self.prefix, train_loss, val_loss, model)
            return False
        if self.mode(val_loss, self.best_validation_score):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_training_score = train_loss
            self.best_validation_score = val_loss
            _save_checkpoint(self.path_to_save, self.prefix, train_loss, val_loss, model)
            self.counter = 0
        return False


class Terminator:
    def __init__(
            self,
            path_to_save: pathlib.Path,
            prefix: str,
            best_k: int,
            summary: typing.Optional[SummaryWriter] = None
    ) -> None:
        self.path_to_save = path_to_save
        self.prefix = prefix
        self.best_k = best_k
        self.summary = summary

    def __call__(self, trigger: bool) -> None:
        if trigger:
            logging.info('cleaning remaining model files')
            _clean_mess(
                self.path_to_save,
                self.prefix,
                self.best_k
            )
            logging.info('terminating application')
            _try_to_terminate(self.summary)
        return None
