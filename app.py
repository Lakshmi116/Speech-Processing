# Imports
import data as dt
import lstm as ll  
import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda, ToTensor
import torchvision.models as models
from python_speech_features import mfcc
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler


# Get Device for trainng 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on Device: {DEVICE}")
print(f"Using PyTorch Version: {torch.__version__}")
print(f"")

# Constants
BATCH_SIZE = 1
ALPHABET_SIZE = 27
MAX_LABEL = 20
SPEAKER = "M01"
DATA_DIR =  "./home/gdata/narayana/Lakshmi/Data/"

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
      #   mfcc_ = mfcc_.astype('float')

        x = torch.from_numpy(mfcc_)
        x = x.type(torch.DoubleTensor)
        x = x.cuda()
      #   x = x.requires_grad_(True)
      #   print(x.dtype)

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
        return x.requires_grad_(True)

# Target transform
word_dict = {}
def output_transform_onehot(y):
   y = y.lower()
   out = torch.zeros(460, device=DEVICE, dtype=torch.long)
   if y in word_dict:
      idx = word_dict[y]
      out[idx] = 1
   else:
      word_dict[y] = len(word_dict)+1
      idx = word_dict[y]
      out[idx] = 1
   return out 



def output_tranform(y):
   y_len = len(y)
   pad = 20 - y_len
   if(pad < 0):
      print("Output size greater than 15")
      return

   out = torch.zeros(20, dtype=torch.long)
   alpha = " abcdefghijklmnopqrstuvwxyz"

   for i in range(y_len):
      idx = alpha.find(y[i])
      out[i] = idx
   
   for i in range(y_len, 20):
      out[i] = 0
   return out.cuda()


# Dataset and Dataloader
train_l = dt.SpeechDataset(
             annotations=os.path.join(DATA_DIR, f"{SPEAKER}_train_annotations.csv"),
             data_dir=DATA_DIR,
             transform=Lambda(lambda x: input_transform(x)),
             target_transform=Lambda(lambda y: output_transform_onehot(y))
)
train_dl = DataLoader(train_l, batch_size=BATCH_SIZE, shuffle=False)

test_l = dt.SpeechDataset(
             annotations=os.path.join(DATA_DIR, f"{SPEAKER}_test_annotations.csv"),
             data_dir=DATA_DIR,
             transform=Lambda(lambda x: input_transform(x)),
             target_transform=Lambda(lambda y: output_transform_onehot(y))
)
test_dl = DataLoader(test_l, batch_size=BATCH_SIZE, shuffle=False)

# Model


model = ll.Model(batch_size=BATCH_SIZE)
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss(reduction='mean')


# Training and Testing loops
def training_loop(dataloader, model, loss_fn, optimizer):
   size = len(dataloader.dataset)
   for batch, (X, y) in enumerate(dataloader):
      output = model(X)

      print(X.shape, output.shape, y.shape)
      print(y[0])
      loss = loss_fn(output, y)

      print(loss.shape)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


      if batch % 100 == 0:
         loss, current = loss.item(), (batch+1)*len(X)
         print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
   size = len(dataloader.dataset)
   num_batches = len(dataloader)
   test_loss, levenshtein_loss = 0,0

   with torch.no_grad():
      for X, y in dataloader:
         pred = model(X)
         test_loss += loss_fn(pred, y).item()
   
   test_loss /= num_batches
   print(f"Avg. Loss: , {test_loss:>8f}")


"""
-----------------------------------------------------------
# Simulation
-----------------------------------------------------------
"""

import time 
start_time = time.time() 
training_loop(train_dl, model, loss_fn, optim)
end_time = time.time()

print(f"runtime (1 epoch): {end_time - start_time}s")

# model = models.vgg16(weights='IMAGENET1K_V1')
models_dir = "./home/gdata/narayana/Lakshmi/SavedModels"
torch.save(model.state_dict(), os.path.join(models_dir, "model_M01") )