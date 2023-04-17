#
# Imports
import torch
from torch import nn


#
# Model Architecture
#
#

class LSTM_Stack(nn.Module):
    def __init__(self, lstm_features, linear_features, relu_features, tanh_features, max_frames, max_label, alphabet_size):
        super().__init__()
        self.lstm_features = lstm_features
        self.linear_features = linear_features 
        self.tanh_features = tanh_features
        self.relu_features = relu_features
        self.alphabet_size= alphabet_size
        self.max_label = max_label
        self.max_frames = max_frames 

        self.flatten_1 = nn.Flatten(start_dim=0, end_dim=-1)
        self.unflatten_1 = nn.Unflatten(dim=0, unflattened_size=(1, self.max_frames*self.lstm_features))
        # A conical network to slowly learn the equivalences b/w speech and vocabulary
        
        self.lstm_1 = nn.LSTM(input_size=self.max_frames*self.lstm_features, 
                    hidden_size=self.max_frames*self.linear_features,
                    num_layers = 3, 
                    dropout = 0.2, 
                    bidirectional = True), # 250 x 13
        
        self.sequential_1 = nn.Sequential(
            nn.Linear(in_features=self.max_frames*self.linear_features,
                      out_features=self.max_frames*self.linear_features), # 250 x 6
            nn.Linear(in_features=self.max_frames*self.linear_features,
                      out_features=self.max_frames*self.relu_features), # 250 x 3
            nn.LeakyReLU(), # 250 x 3
            nn.Linear(in_features=self.max_frames*self.relu_features,
                      out_features=self.max_frames*self.tanh_features), # 250 x 2
            nn.Tanh(), # 250 x 2
            nn.Linear(in_features=self.max_frames*self.tanh_features,
                      out_features=self.max_label*self.alphabet_size), # 250 x 2 -> 15 x 27 (output size)
            nn.Unflatten(dim=0, unflattened_size=(15,27)), # (15,27)
            nn.Softmax(dim=1), # (15, 27)
        )

    def forward(self, x):
        
        # x = self.flatten_1(x) # Flatten the input
        # x = self.unflatten_1(x) # Unflatten for feeding it to the neural network
        
        """ Complete this part """
        x1 = self.lstm_1(self.unflatten_1(self.flatten_1(x)))

        return x1
        x1 = x1[1][1][0] # Hidden state of layer 3
        
        y = self.sequential_1(x1) 
        return y

"""Usage: """
# model = NeuralNetwork().to(device)
# print(model)
#
#


"""
Use the first loss functions to train the network
Use the second loss function to train GAN ...
"""
class LevenshteinLoss(nn.Module):
    def __init__(self):
        super(LevenshteinLoss, self).__init__()
    
    def forward(self, output, target):
        loss = torch.zeros(1)
        # loss = lossfnlogic(output, target)
        return loss 
    
    # Fucntion to skip spaces in the word
    def skip_spaces(word):
        n = len(word)
        new_word = ""
        for ch in word:
            if ch == " ":
                continue
            else:
                new_word+=ch
        return new_word
    
    # We use levensthtein distance between two words for calculating the loss function
    def levenshtein_distance(a, b):
        # minimum edits requried to transform a to b
        n = len(a)
        m = len(b)
        dp = torch.zeros(size=(1+n, 1+m))
        for j in range(m+1):
            dp[0][j] = j
        for i in range(n+1):
            dp[i][0] = i
        
        # insert, replace and delete are possible operations
        for i in range(1, n+1):
            for j in range(1, m+1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = min([dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]])
                else:
                    dp[i][j] = min([dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1])
        # print(dp)
        return dp[n][m]

class DynammicTimeWarpingLoss(nn.Module):
    def __init__(self):
        super(DynammicTimeWarpingLoss, self).__init__()
    
    def forward(self, output, target):
        return 0
    
    def DTW_measure(shape1, shape2):
        return 0



"""Usage: In the training loop """
# my_loss = CustomLossFn()
# for epoch in range(epochs):
    # model.train() # IDK what does this mean
    # output = model(input)
    # loss = my_loss(output, target)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
# print(model)
#
#


