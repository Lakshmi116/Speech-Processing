"""
# Python 3
# PyTorch Dataset and Dataload
+---------------------------------------+
| Creator: Gollapudi N Lakshmi Narayana |
| Date   : 14-04-2023                   |
| Project: Speech =:= Words             |
+---------------------------------------+
 #
 #
 #
 """
# PyTorch Imports
from torch.utils.data import Dataset

# IO Imports
import os
import pandas as pd
from scipy.io import wavfile

# Linear ALgebra Imports
from sklearn.preprocessing import StandardScaler

# Speech Imports
from python_speech_features import mfcc


class Dataset_SpeechToString(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        """
        Annotations format: 0. "label-0", "path-to-input" 
        Annotation headline: label, path
        """
        self.annotations_df = pd.read_csv(annotations_file)
        self.data_dir = data_dir 
        self.transform = transform
        self.target_transform = target_transform

        # Utility Functions
        self.scaler = StandardScaler()

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.data_dir, self.annotations_df.iloc[(idx, 1)])
        mfcc_ = self.read_audio_to_mfcc(audio_path)
        mfcc_n = self.normalize_features(mfcc_)
        label = self.annotations_df.iloc[(idx, 0)]
        if self.transform:
            mfcc_nt = self.transform(mfcc_n)
        if self.target_transform:
            label_t = self.target_transform(label)
        return mfcc_nt, label_t
    
    def read_audio_to_mfcc(self, audio_path):
        file_path = os.path.join(self.data_dir, audio_path)
        sampling_frequency, audio_signal = wavfile.read(file_path)
        return mfcc(audio_signal, sampling_frequency)
        
    
    def normalize_features(self, features_):
        self.scaler = self.scaler.fit(features_)
        return self.scaler.transform(features_)



"""
Boiler plate code to utilize the above created custom dataset
 - Copy paste to define training and testing batch loaders
 - Sample Use functions are also given below
 - Type of loading depends on the nature of the data
"""

# from torch.utils.data import DataLoader

# training_data = Dataset_SpeechToString(training_annotations, data_dir, transfor, target_transform)
# testing_data = Dataset_SpeechToString(testing_annotations, data_dir, tranform, target_transform)
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

