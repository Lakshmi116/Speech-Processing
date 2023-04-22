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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on Device: {device}")
print(f"Using PyTorch Version: {torch.__version__}")
print(f"")

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
        return x.cuda()

# Target transform
def output_tranform(y):
   y_len = len(y)
   pad = 20 - y_len 
   out = torch.zeros(20, 27)
   alpha = "abcdefghijklmnop"
   for i in range(y_len):
      idx = alpha.find(y[i])
      out[i][idx] = 1
   if(pad < 0):
      print("Output size greater than 15")
   for i in range(y_len, 20):
      out[i][26] = 1
   return out.cuda()

# Levenshtein's loss
# def levenshtein_loss(output, target):


# Data ROOT 
data_dir =  "./home/gdata/narayana/Lakshmi/Data/"

# Dataset and Dataloader
train_l = dt.SpeechDataset(
             annotations=os.path.join(data_dir, "train_annotations.csv"),
             data_dir=data_dir,
             transform=Lambda(lambda x: input_transform(x)),
             target_transform=Lambda(lambda y: output_tranform(y))
)
train_dl = DataLoader(train_l, batch_size=16, shuffle=False)

test_l = dt.SpeechDataset(
             annotations=os.path.join(data_dir, "test_annotations.csv"),
             data_dir=data_dir,
             transform=Lambda(lambda x: input_transform(x)),
             target_transform=Lambda(lambda y: output_tranform(y))
)
test_dl = DataLoader(test_l, batch_size=16, shuffle=False)


x, target = next(iter(train_dl))
# print(x)
# Model Architechture
mf = ll.Model(batch_size=16)

# Training
batch_size = 16
alphabet_size = 27
max_label = 20

weights = torch.tensor([1/27]*27, device=device)

model = ll.Model(batch_size=16)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def bcewl(output, target):
   return F.binary_cross_entropy_with_logits(output, target, weight=weights)

def training_loop(dataloader, model, loss_fn, optimizer):
   size = len(dataloader.dataset)
   for batch, (X, y) in enumerate(dataloader):
      output = model(X)
      loss = loss_fn(output, y)

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


import time 
start_time = time.time() 
training_loop(train_dl, model, bcewl, optimizer)
end_time = time.time()

print(f"runtime (1 epoch): {end_time - start_time}s")

# model = models.vgg16(weights='IMAGENET1K_V1')
models_dir = "./home/gdata/narayana/Lakshmi/SavedModels"
torch.save(model.state_dict(), os.path.join(models_dir, "model_1") )