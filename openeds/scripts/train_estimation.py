import logging
import pathlib
from time import asctime
from statistics import mean
from argparse import ArgumentParser

import torch
import pandas as pd
import torch.nn as nn

from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..resnet import grey_resnet18
from ..callbacks import (
    EarlyStopping,
    Terminator,
)
from ..dataset import GazeEstimationDataset

from ..utils import (
    get_gkfold_split,
    cast_to_numpy,
    TqdmStream,
)


def run() -> None:
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.DEBUG,
        stream=TqdmStream
    )
    logger = logging.getLogger(__name__)

    parser = ArgumentParser()
    parser.add_argument('-in', '--dataset_path', type=str, help='path to the preprocessed csv file')
    parser.add_argument(
        '-out',
        '--path_to_save',
        type=str,
        default='models',
        help='directory to save the model checkpoints'
    )
    parser.add_argument('-p', '--prefix', type=str, default='resnet18', help='prefix for the model checkpoints')

    parser.add_argument('-bs', '--batch_size', type=int, default=450, help='batch size to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('-pat', '--patience', type=int, default=7, help='patience of the learning rate scheduler')
    parser.add_argument('-es_pat', '--early_stop_patience', type=int, default=5, help='patience of the early stopping')
    parser.add_argument('-k', '--best_k', type=int, default=3, help='number of best models to keep')

    parser.add_argument('-e', '--n_epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('-nf', '--n_folds', type=int, default=5, help='number of folds in validation scheme')
    parser.add_argument('-fn', '--fold', type=int, default=0, help='number of particular fold to use')
    parser.add_argument(
        '-a',
        '--aug_prob',
        type=float,
        default=0.95,
        help='probability to apply augmentations (from 0 to 1)'
    )
    parser.add_argument('-dev', '--device', type=str, default='cuda', help='device to process')

    args = parser.parse_args()

    device = torch.device(args.device)
    summary = SummaryWriter(f'model_logs/{asctime()}')

    dataset = pd.read_csv(args.dataset_path)

    train, valid = get_gkfold_split(
        dataset,
        n_splits=args.n_folds,
        split=args.fold
    )

    train_dataset = GazeEstimationDataset(train, is_test=False, aug_prob=args.aug_prob)
    valid_dataset = GazeEstimationDataset(valid, is_test=False, aug_prob=0.0)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )
    logger.info(f'length of the train dataloader is {len(train_dataloader)}')

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False,
        drop_last=True
    )

    logger.info(f'length of the valid dataloader is {len(valid_dataloader)}')

    model = grey_resnet18()
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=args.patience,
        verbose=True
    )

    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        mode='min',
        path_to_save=pathlib.Path(args.path_to_save),
        prefix=f'{args.prefix}_fold_{args.fold}',
    )

    terminator = Terminator(
        path_to_save=pathlib.Path(args.path_to_save),
        prefix=f'{args.prefix}_fold_{args.fold}',
        best_k=args.best_k,
        summary=summary
    )

    with tqdm(range(args.n_epochs)) as progress_bar:
        for epoch in progress_bar:
            model.train()

            train_loss_acc = list()
            valid_loss_acc = list()

            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()

                _, x, y = batch
                x = x.to(device)
                y = y.to(device)
                prediction = model(x)

                loss = nn.functional.mse_loss(prediction, y)
                loss.backward()
                optimizer.step()
                train_loss_acc.append(cast_to_numpy(loss).item())

            mean_training_loss = mean(train_loss_acc)
            summary.add_scalar(
                f'training mse per epoch',
                mean_training_loss,
                epoch
            )

            model.eval()
            with torch.no_grad():
                for valid_batch in valid_dataloader:
                    _, x, y = valid_batch
                    x = x.to(device)
                    y = y.to(device)
                    prediction = model(x)
                    valid_loss = nn.functional.mse_loss(prediction, y)
                    valid_loss_acc.append(cast_to_numpy(valid_loss).item())

            mean_valid_loss = mean(valid_loss_acc)
            summary.add_scalar(
                f'validation mse per epoch',
                mean_valid_loss,
                epoch
            )

            progress_bar.set_postfix({
                f'train mse per epoch': mean_training_loss,
                f'valid mse per epoch': mean_valid_loss
            }, refresh=False)

            scheduler.step(mean_valid_loss)
            es = early_stopper(mean_training_loss, mean_valid_loss, model)
            terminator(es)
    return None
