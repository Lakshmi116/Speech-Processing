print("Importing required libraries")

# Imports
import data as dt
import lstm as ll  
import os
import time 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda, ToTensor
import torchvision.models as models
from python_speech_features import mfcc
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
import gc


print("Emptying the gpu cache")
torch.cuda.empty_cache()

# Get Device for trainng 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on Device: {DEVICE}")
print(f"Using PyTorch Version: {torch.__version__}")
print(f"")
# torch.backends.cudnn.enabled = False 

# Constants
BATCH_SIZE = 1
ALPHABET_SIZE = 27
MAX_LABEL = 20
SPEAKER = "M01"
MODEL_NAME = SPEAKER + "sample1"
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
      return x

# Target transform
word_dict = {}
def output_transform_onehot_idx(y):
   y = y.lower()
   idx = 459

   if y not in word_dict:
      word_dict[y] = len(word_dict)+1

   idx = word_dict[y]
   return torch.tensor(idx, device=DEVICE, dtype=torch.long)

# Training and Testing loops
def training_loop(trainloader, net, criterion, optimizer, num_epochs):
   
   for epoch in range(num_epochs):  # loop over the dataset multiple times
      accuracy = 0
      total = 0
      running_loss = 0.0
      print(f"Starting epoch {epoch}")
      size = len(trainloader)
      print(f"size = {size}")
      for i, data in enumerate(trainloader, 0):
         # get the inputs; data is a list of [inputs, labels]
         inputs, labels = data

         # zero the parameter gradients
         net.zero_grad()

         # forward + backward + optimize
         outputs = net(inputs)
         loss = criterion(outputs, labels)
         print(loss)
         state_dict_prev = optimizer.state_dict()
         loss.backward()
         # exit()
         optimizer.step()

         
         print("Emptying the gpu cache")
         gc.collect()
         torch.cuda.empty_cache()

         print(f"optimizer step succesfull")
         state_dict_curr = optimizer.state_dict()
         print(i)
         print(state_dict_curr == state_dict_prev)
      #   exit()

         # Update Accuracy
         classes = torch.argmax(outputs, dim=1)
         for j in range(len(classes)):
            if classes[j] == labels[j]:
                  accuracy+=1
            total+=1

         # print statistics
         running_loss += loss.item()
      # print every 100 mini-batches
      print('epoch: %d, loss: %.3f, accuracy: %.3f' %(epoch + 1 , running_loss / size, accuracy/total))
      running_loss = 0.0

   print('Finished Training')  



# Dataset and Dataloader
print(f"Dataset loading")
train_l = dt.SpeechDataset(
             annotations=os.path.join(DATA_DIR, f"{SPEAKER}_train_annotations.csv"),
             data_dir=DATA_DIR,
             transform=Lambda(lambda x: input_transform(x)),
             target_transform=Lambda(lambda y: output_transform_onehot_idx(y))
)
train_dl = DataLoader(train_l, batch_size=BATCH_SIZE, shuffle=False)

test_l = dt.SpeechDataset(
             annotations=os.path.join(DATA_DIR, f"{SPEAKER}_test_annotations.csv"),
             data_dir=DATA_DIR,
             transform=Lambda(lambda x: input_transform(x)),
             target_transform=Lambda(lambda y: output_transform_onehot_idx(y))
)
test_dl = DataLoader(test_l, batch_size=BATCH_SIZE, shuffle=False)




# Model
print(f"Model Initialization")
model = ll.Model(batch_size=BATCH_SIZE)
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()


# Loading from the saved models
print(f"Loading the model")
MODELS_DIR = "./home/gdata/narayana/Lakshmi/SavedModels"
PATH = os.path.join(MODELS_DIR, MODEL_NAME)
model.load_state_dict(torch.load(PATH))
print(model)

# Simulation
print(f"Simulation")
start_time = time.time() 
EPOCHS = 1
training_loop(train_dl, model, loss_fn, optim, EPOCHS)
print(f"runtime ({EPOCHS} epochs): {time.time() - start_time}s")


# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# print()

# Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optim.state_dict():
#     print(var_name, "\t", optim.state_dict()[var_name])


torch.save(model.state_dict(),  PATH)






























# Testing Loop 
"""
def testing_loop(testloader, net, criterion):
   accuracy = 0
   total = 0
   running_loss = 0.0
   size = len(testloader)
   for i, data in enumerate(testloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      
      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)

      # Update Accuracy
      classes = torch.argmax(outputs, dim=1)
      for j in range(len(classes)):
         if classes[j] == labels[j]:
            accuracy+=1
         total+=1

      # print statistics
      running_loss += loss.item()
   # print every 100 mini-batches
   print('loss: %.3f, accuracy: %.3f' %(epoch + 1 , running_loss / size, accuracy/total))
   print('Finished Testing') 
   """
