import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class NoteDataset(Dataset):

    def __init__(self, audio_dir, json_file):
        self.audio_dir = audio_dir
        with open(json_file) as jf:
            self.examples = list(json.load(jf).values())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        note = self.examples[idx]
        note_path = '{}/{}.wav'.format(self.audio_dir, note['note_str'])
        audio, sr = torchaudio.load(note_path)
        note['data'] = audio
        note['sample_rate'] = sr
        return note