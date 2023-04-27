# Imports
import torch
import torch.nn as nn
import data as dt
import os
import time 
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda, ToTensor
from python_speech_features import mfcc
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

# Device
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on Device: {device}")
print(f"Using PyTorch Version: {torch.__version__}")
print(f"")



# Constants
BATCH_SIZE = 16
SPEAKER = "M09"
TAG = "start51"
MODEL_NAME = SPEAKER + TAG + "_260423_" + device
DATA_DIR =  "./home/gdata/narayana/Lakshmi/Data/"
MODELS_DIR = "./home/gdata/narayana/Lakshmi/SavedModels"
LOG_DIR = "./home/gdata/narayana/Lakshmi/ModelsSummary"
PATH = os.path.join(MODELS_DIR, MODEL_NAME + ".nn")
LOG_PATH = os.path.join(LOG_DIR, MODEL_NAME + ".log")
EVAL_LOG_PATH = os.path.join(LOG_DIR, MODEL_NAME + ".elog")
CLASSMAP_PATH = os.path.join()




# Transform
def path_to_tensor(audio_path):
   # Reading from wav
   sampling_frequency, audio_signal = wavfile.read(audio_path)

   # Calculating MFCC
   mfcc_ = mfcc(audio_signal, sampling_frequency)

   # Normalization
   scaler = StandardScaler()
   scaler = scaler.fit(mfcc_)
   mfcc_ = scaler.transform(mfcc_)

   x = torch.from_numpy(mfcc_).double().to(device)
   return x

# Target transform

word_dict = {}
def word_to_idx(y):
   y = y.lower()
   idx = 0

   if y not in word_dict:
      word_dict[y] = len(word_dict)

   idx = word_dict[y]
   return torch.tensor(idx, device=device, dtype=torch.long)

# Collate_fn
def pad_seq(batch):
   """
   batch: (tensor, label, length)
   """
   features, labels = zip(*batch)
   # print("@ collate fn")
   # for f in features:
   #    print(f.shape)
   return (pad_sequence(features, batch_first=True).double(), torch.tensor(labels).to(device))




# Dataset and Dataloader
print(f"Dataset loading")
train_l = dt.SpeechDataset(
             annotations=os.path.join(DATA_DIR, f"{SPEAKER}_train_annotations_{TAG}.csv"),
             data_dir=DATA_DIR,
             transform=Lambda(lambda x: path_to_tensor(x)),
             target_transform=Lambda(lambda y: word_to_idx(y))
)
train_dl = DataLoader(train_l, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_seq)

test_l = dt.SpeechDataset(
             annotations=os.path.join(DATA_DIR, f"{SPEAKER}_test_annotations_{TAG}.csv"),
             data_dir=DATA_DIR,
             transform=Lambda(lambda x: path_to_tensor(x)),
             target_transform=Lambda(lambda y: word_to_idx(y))
)
test_dl = DataLoader(test_l, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_seq)


# Train loop
def train_loop(trainloader, model, criterion, optimizer, epochs):
   loger = open(LOG_PATH, "a")
   for epoch in range(epochs):
      total = 0
      correct = 0
      running_loss = 0
      size = len(trainloader)

      print(f"running at epoch {epoch}")
      # batch = -1
      # for idx, inputs, labels in enumerate(trainloader):
      for idx, (inputs, labels) in enumerate(trainloader):
         # print(int)
         # print(len(data), data[0].shape, data[1])
         # exit()
         # batch += 1

         # zero the parameter gradients
         optimizer.zero_grad()

         # forward + backward + optimize
         outputs = model(inputs)
         # print("output shape:", outputs.shape)
         # print("target shape:", labels.shape)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()

         classes = torch.argmax(outputs, dim=1)
         correct_t = (classes == labels)
         correct += correct_t.sum()
         total += len(classes)
         running_loss += loss.item()

         if idx % 10 == 9:
            loger.write("epoch: %d, loss: %.4f, accuracy: %.4f\n" %(epoch+1, running_loss/10, correct/total))
            running_loss = 0
      loger.write("epoch: %d, loss: %.4f, accuracy: %.4f\n" %(epoch+1, running_loss/10, correct/total))
   print("Finished Training")

def test_loop(testloader, model):
   loger = open(EVAL_LOG_PATH, "a")
   correct = 0
   total = 0
   size = len(testloader)
   for idx, (inputs, labels) in enumerate(testloader):
      outputs = model(inputs)

      classes = torch.argmax(outputs, dim=1)
      correct_t = (classes == labels)
      correct += correct_t.sum()
      total += len(classes)
   
   loger.writelines([
      f"\n",
      f"Validation set size: {total}\n",
      f"Accurate: {correct}\n",
      f"Validation Accuracy: {correct / total}\n"
   ])


# Model
class Model52(nn.Module):
   def __init__(self, input_size, num_layers, hidden_size, target_size):

      super(Model52, self).__init__()
      self.input_size = input_size
      self.num_layers = num_layers
      self.hidden_size = hidden_size
      self.target_size = target_size
      self.D = 2
      
      self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=(self.D==2)).double().to(device)
      self.fc = nn.Linear(self.D*self.hidden_size, self.target_size).double().to(device)
      
   def forward(self, x):
      output, _ = self.lstm(x)
      output = output[:, -1]
      output = self.fc(output)
      return output




# Model Definition
model = Model52(13, 3, 128, 52)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()



# # Training 
# EPOCHS = 100
# starttime = time.time()
# train_loop(train_dl, model, criterion, optimizer, EPOCHS)
# endtime = time.time()

# # Save model
# print(f"Runtime: {endtime-starttime}, epochs: {EPOCHS}")
# torch.save(model.state_dict(), PATH)


"""
Evaluating Trained Models
"""
# Loading 
# model = Model52
model.load_state_dict(torch.load(PATH))
model.eval()

with torch.no_grad():
   test_loop(test_dl, model)
   test_loop(train_dl, model)


