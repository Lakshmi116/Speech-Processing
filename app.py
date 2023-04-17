# Imports
import data as dt
import lstm as ll  
import os
import torch
from torch.utils.data import DataLoader

# Get Device for trainng 
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on Device: {device}")
print(f"Using PyTorch Version: {torch.__version__}")



# Storing all the model parameters in a dictionary for hyper parameter tuning
"""Model Parameters"""
model_parameters = {
   "MAX_FRAMES": 250,
   "MAX_LABEL": 15,
   "ALPHABET_SIZE":27,
   "LSTM_FEATURES": 13,
   "LINEAR_FEATURES": 6,
   "RELU_FEATURES": 3,
   "TANH_FEATURES": 2
}

# --------------------------------------------------------
# --------------------------------------------------------

"""
    Tasks:

    1. Complete tranform 
    2. Complete target transform
    3. Complete neural network baseline definitions
    4. Play with the hyperparameters
    5. Fine tune the hyperparameters
    6. State all the parameters that are playing vital role 
       in the entire simulation
    """


# --------------------------------------------------------
# --------------------------------------------------------

# Helper Functions
def print_model_params(model_params, model_name="LSTM"):
   print(f"Model Parameters for {model_name}")
   for (param, val) in iter(model_parameters):
      print(f"{param}: {val}")
   print(f"\n")

def print_model_output(y):
   return 

def print_model_input(x):
   return 




# Initialize lstm network
sample_lstm_stack = ll.LSTM_Stack(
         lstm_features=model_parameters["LSTM_FEATURES"],
         linear_features=model_parameters["LINEAR_FEATURES"],
         relu_features=model_parameters["RELU_FEATURES"],
         tanh_features=model_parameters["TANH_FEATURES"],
         max_frames=model_parameters["MAX_FRAMES"],
         max_label=model_parameters["MAX_LABEL"],
         alphabet_size=model_parameters["ALPHABET_SIZE"])

# Input shape = (None, 13)
sample_input_X = [torch.randn(300, 13), torch.randn(235, 13)]
sample_output_y = [sample_lstm_stack(tensor) for tensor in sample_input_X]

for batch, tensor in enumerate(sample_output_y):
   print(f"")
   print(f"Tensor {batch}:")
   print(f"Shape: ", tensor.shape)
   print(f"Space:", tensor)
   print(f"Max (dim=1):", tensor.max(dim=1))
   

""" Network Backlog:

1. Loss function
2. Network definition


"""


""" Data Backlog:

1. Transform
2. Target transform
3. Annotations
4. Train/Test Logic

 """

# training_data = dt.Dataset_SpeechToString(annotations_file="annotations/path",
#                                           data_dir="data-root-dir",
#                                           transform=None,
#                                           target_transform=None)

# testing_data = dt.Dataset_SpeechToString(annotations_file="testing-annotations-file",
#                                          data_dir = "data-root-dir",
#                                          transform=None,
#                                          target_transform=None)

# train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
# test_dataloader = DataLoader(testing_data, batch_size=4, shuffle=True)









