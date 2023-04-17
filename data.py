from torch.utils.data import Dataset
import os
import pandas as pd

""" Sample Use Cases """
# from torch.utils.data import DataLoader

# training_data = SpeechDataset(training_annotations, data_dir, transform, target_transform)
# testing_data = SpeechDataset(testing_annotations, data_dir, tranform, target_transform)




""" Split between train and test data by creating seperate annotations"""
# train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)


""" Sample use cases of the dataloders defined above"""
# features, label = next(iter(train_dataloader))
# features, label = next(iter(test_dataloader))

# # for batch, (xi, yi) in enumerate(train_dataloader):
    # ...
# for batch, (xi, yi) in enumerate(trst_dataloader):
    # ...



class SpeechDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        """
        Annotations format: 0. "label-0", "path-to-input" 
        Annotation headline: label, path
        """
        self.annotations_df = pd.read_csv(annotations_file)
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






