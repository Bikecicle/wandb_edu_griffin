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

debug = False

if __name__ == '__main__':

    train_config = SimpleNamespace(
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

    # Create logger and update config
    if debug:
        logger = None
        config = train_config
    else:
        logger = WandbLogger(project=params.WANDB_PROJECT, config=train_config)
        config = wandb.config


    # Set seed for reproducability
    seed_everything(train_config.seed)

    # Link to dataset artifact
    if not debug:
        data_at = logger.use_artifact(f'{params.RAW_DATA_AT}:latest')

    # Prepare dataset
    data_path = Path('C:/Users/Griffin/ml/datasets/nsynth/nsynth-valid')
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

    # Execute
    trainer.fit(model, train_loader, val_loader)
