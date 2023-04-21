import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from types import SimpleNamespace
from pathlib import Path
import json
import params

import wandb
import rave
import rave.dataset
import rave.blocks
import rave.pqmf


if __name__ == '__main__':

    train_config = SimpleNamespace(
        framework="RAVEV2Encoder",
        batch_size=32,
        epochs=10,
        seed=42,
        n_labels=10,
        sampling_rate=16000,
        n_signal=64000,
        n_band=16,
        attenuation=100,
        latent_size=32,
        n_out=2,
        lr=2e-5,
        kernel_size=3,
        dilations=[1, 3],
        ratios=[4, 4, 2],
        fc_sizes=[],
        capacity=24
    )

    # Create logger and update config
    wandb_logger = WandbLogger(project=params.WANDB_PROJECT, config=train_config)
    config = wandb.config

    # Set seed for reproducability
    seed_everything(train_config.seed)

    # Link to dataset artifact
    data_at = wandb_logger.use_artifact(f'{params.RAW_DATA_AT}:latest')

    # Prepare dataset
    data_path = Path('C:/Users/Griffin/Documents/datasets/nsynth/nsynth-valid')
    db_path = data_path / 'processed-labeled'
    num_workers = 0

    dataset = rave.dataset.get_dataset(str(db_path), config.sampling_rate, config.n_signal)
    train, val = rave.dataset.split_dataset(dataset, 80)
    train_loader = DataLoader(
        train,
        train_config.batch_size,
        True,
        drop_last=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val,
        train_config.batch_size,
        False,
        num_workers=num_workers
    )

    # Build model
    pqmf = rave.pqmf.PQMF(config.attenuation, config.n_band)
    model = rave.model.TimbreEncoder(
        n_labels=config.n_labels,
        data_size=config.n_band,
        n_out=config.n_out,
        latent_size=config.latent_size,
        sampling_rate=config.sampling_rate,
        lr=config.lr,
        kernel_size=config.kernel_size,
        dilations=config.dilations,
        ratios=config.ratios,
        fc_sizes=config.fc_sizes,
        capacity=config.capacity,
        pqmf=pqmf
    )

    print(model)

    # Setup callbacks and trainer
    callbacks = [ModelCheckpoint()]
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.epochs,
        logger=wandb_logger,
        callbacks=callbacks
    )

    # Execute
    trainer.fit(model, train_loader, val_loader)
