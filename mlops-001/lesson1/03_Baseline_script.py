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
        n_signal=16000,
        n_band=16,
        attenuation=100,
        latent_size=32,
        n_out=2,
        lr=2e-3,
        kernel_size=3,
        dilations=[],
        ratios=[4, 2],
        fc_sizes=[128, 64],
        capacity=24
    )

    # Create logger and update config
    if debug:
        config = train_config
    else:
        wandb_logger = WandbLogger(project=params.WANDB_PROJECT, config=train_config)
        config = wandb.config


    # Set seed for reproducability
    seed_everything(train_config.seed)

    # Link to dataset artifact
    if not debug:
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
    '''
    pqmf = rave.pqmf.CachedPQMF(config.attenuation, config.n_band)
    encoder = VariationalEncoder(EncoderV2(
            data_size=config.data_size,
            capacity=capacity,
            ratios=ratios,
            latent_size=latent_size,
            n_out=n_out,
            kernel_size=kernel_size,
            dilations=dilations,
            keep_dim=False,
            recurrent_layer=None
        ))
    '''

    model_pt = torch.jit.load('C:/Users/Griffin/Downloads/vintage.ts')
    model = rave.model.TimbreClassifier(
        n_labels=config.n_labels,
        latent_size=config.latent_size,
        sampling_rate=config.sampling_rate,
        lr=config.lr,
        fc_sizes=config.fc_sizes,
        encoder=model_pt._rave.encoder,
        pqmf=model_pt._rave.pqmf
    )

    if not debug:
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
