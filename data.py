from torch.utils.data import Dataset
import os
import pandas as pd

class SpeechDataset(Dataset):
    def __init__(self, annotations, data_dir, transform=None, target_transform=None):
        """
        Annotations format: 0. "label-0", "path-to-input" 
        Annotation headline: label, path
        """
        self.annotations_df = pd.read_csv(annotations)
        self.data_dir = data_dir 
        self.transform = transform
        self.target_transform = target_transform

        

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        file = self.annotations_df.iloc[(idx, 0)]
        label = self.annotations_df.iloc[(idx, 1)]
        if self.transform:
            file = self.transform(file)
        if self.target_transform:
            label = self.target_transform(label)
        return file, label






