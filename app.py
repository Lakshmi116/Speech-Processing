# Imports
import data as dt
import lstm as ll  
import os
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda, ToTensor
from python_speech_features import mfcc
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler


# Get Device for trainng 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on Device: {device}")
print(f"Using PyTorch Version: {torch.__version__}")
print(f"")

# Data ROOT 
data_dir =  "./home/gdata/narayana/Lakshmi/Data/"

# Dataset and Dataloader
train_l = dt.SpeechDataset(
             annotations=os.path.join(data_dir, "train_annotations.csv"),
             data_dir=data_dir,
             transform=Lambda(lambda x: input_transform(x)),
             target_transform=None
)
train_dl = DataLoader(train_l, batch_size=4, shuffle=True )

test_l = dt.SpeechDataset(
             annotations=os.path.join(data_dir, "test_annotations.csv"),
             data_dir=data_dir,
             transform=Lambda(lambda x: input_transform(x)),
             target_transform=None
)
test_dl = DataLoader(test_l, batch_size=4, shuffle=True)

# Model Architechture

# Training
for batch, (X, y) in enumerate(train_dl):
   print(batch, X.shape, y)

# Validation

# Summary






# Tranforms
def input_transform(audio_path):
        # Reading from wav
        sampling_frequency, audio_signal = wavfile.read(audio_path)

        # Calculating MFCC
        mfcc_ = mfcc(audio_signal, sampling_frequency)

        # Normalization
        scaler = StandardScaler()
        scaler = scaler.fit(mfcc_)
        mfcc_ = scaler.transform(mfcc_)

        x = torch.from_numpy(mfcc_)

        # Convert to 250 frame tape
        # Input form (None, 13)
        # Pad with zeros at the end
        MAX_FRAMES = 250
        no_of_frames = x.shape[0]
        if no_of_frames < MAX_FRAMES:
            pad = MAX_FRAMES - no_of_frames
            padder = nn.ZeroPad2d((0,0,0,pad))
            x = padder(x)
        
        x = x[:MAX_FRAMES, :] # Truncate the signal to max_frames
        return x 

# Target transform
def output_tranform(X):
   idx_ = torch.max(X, dim=1)
   alpha = "abcdefghijklmnopqrstuvwxyz "
   s = ""
   for idx in idx_:
      if idx == 26:
         continue
      s += alpha[idx]
   return s

