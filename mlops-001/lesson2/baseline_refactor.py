import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from types import SimpleNamespace
from pathlib import Path
import tqdm
import params
import argparse

import wandb
import rave
import rave.dataset
import rave.blocks
import rave.pqmf

DEBUG = False

default_config = SimpleNamespace(
    framework="RAVEV2Encoder",
    batch_size=32,
    epochs=10,
    seed=42,
    n_labels=11,
    sampling_rate=16000,
    n_signal=64000,
    lr=2e-3,
    n_layers=2,
    n_features_start=128
)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size,
                           help='batch size')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs,
                           help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr,
                           help='learning rate')
    argparser.add_argument('--seed', type=int, default=default_config.seed,
                           help='random seed')
    argparser.add_argument('--n_layers', type=int, default=default_config.n_layers,
                           help='number of fully-connected layers after encoder')
    argparser.add_argument('--n_features_start', type=int, default=default_config.n_features_start,
                           help='size of starting fc layer - subsequent layers are each half of the previous')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


def get_data_loaders(batch_size, data_path):
    data_path = Path(data_path)
    db_path = data_path / 'processed-labeled'
    num_workers = 0

    dataset = rave.dataset.get_dataset(str(db_path), default_config.sampling_rate, default_config.n_signal)
    train, val = rave.dataset.split_dataset(dataset, 80)
    train_loader = DataLoader(
        train,
        batch_size,
        True,
        drop_last=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val,
        batch_size,
        False,
        num_workers=num_workers
    )
    return train_loader, val_loader


def get_logger():
    # Create logger and update config
    if DEBUG:
        logger = None
        config = default_config
    else:
        logger = WandbLogger(project=params.WANDB_PROJECT, config=default_config, log_model=True)
        config = wandb.config
    return logger, config


def train(config):

    logger, config = get_logger()

    # Set seed for reproducability
    seed_everything(config.seed)

    # Link to dataset artifact
    if not DEBUG:
        data_at = logger.use_artifact(f'{params.RAW_DATA_AT}:latest')

    # Build model
    pt_model = torch.jit.load('C:/Users/Griffin/ml/models/rave/vintage.ts')
    model = rave.model.TimbreClassifier(
        n_labels=config.n_labels,
        sampling_rate=config.sampling_rate,
        lr=config.lr,
        n_layers=config.n_layers,
        n_features_start=config.n_features_start,
        encoder=pt_model._rave.encoder,
        pqmf=pt_model._rave.pqmf
    )

    print(model)

    # Setup callbacks and trainer
    callbacks = [ModelCheckpoint()]
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.epochs,
        logger=logger,
        callbacks=callbacks
    )

    # Load data
    train_loader, val_loader = get_data_loaders(config.batch_size, 'C:/Users/Griffin/ml/datasets/nsynth/nsynth-valid')

    # Execute
    trainer.fit(model, train_loader, val_loader)

    table_data = []
    for i, batch in tqdm.tqdm(enumerate(val_loader)):
        _, y = batch
        p = model.predict_step(batch, i)
        for inst_y, inst_p in zip(y.tolist(), p.tolist()):
            table_data.append([
                params.NSYNTH_CLASSES[inst_y],
                params.NSYNTH_CLASSES[inst_p]
            ])

    if not DEBUG:
        logger.log_table(
            key='pred_table',
            columns=['True_Instrument', 'Predicted_Instrument'],
            data=table_data
        )


if __name__ == '__main__':
    parse_args()
    train(default_config)
