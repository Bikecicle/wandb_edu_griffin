from pathlib import Path
import wandb

import torch
import torchaudio
import pandas as pd

import params
from dataset import NoteDataset

if __name__ == '__main__':

    #run = wandb.init(project=params.WANDB_PROJECT, job_type='data_preprocessing')
    data_path = Path('C:/Users/Griffin/ml/datasets/nsynth/nsynth-valid')
    model_path = Path('C:/Users/Griffin/ml/models/rave/vintage.ts')

    dataset = NoteDataset(data_path / 'audio', data_path / 'examples.json')

    print(dataset[0])

    model = torch.jit.load(model_path)
    pqmf = model._rave.pqmf
    encoder = model._rave.encoder
    with torch.no_grad():
        x = dataset[0]['data'].unsqueeze(0)
        x = pqmf(x)
        x = encoder(x)
        print(x)
