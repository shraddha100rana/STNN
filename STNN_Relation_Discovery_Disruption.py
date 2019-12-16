import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# X is input data
# Size 247 (weeks) x 136 (nodes) x 35 (features) 
# Size 1733 (days) x 136 (nodes) x 35 (features)
X = np.load('X_Input_Daily.npy')

# Y is output data
# Size 247 (weeks) x 136 (nodes) x 4 (features) 
# Size 1733 (days) x 136 (nodes) x 4 (features)
y = np.load('Y_Output_Daily.npy')

# Set device as GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Send data to device
x_tensor = torch.from_numpy(X).float().to(device)
y_tensor = torch.from_numpy(y).float().to(device)

train_data = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset = train_data, batch_size = 1)

# The Encoder captures the latent representation of the inputs
# X_{t} --> Z_{t}
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(136*35, 136*16)
        self.fc2 = nn.Linear(136*16, 136*8)
        self.fc3 = nn.Linear(136*8, 136*4)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim = 1)
    
# The Dynamic part captures the temporal relationship the latent representation
# We also specify a spatial relationship between the nodes which is learnt
# Z_{t+1} = g(Z_{t}*Theta0 + W*Z_{t}*Theta1)
class Dynamic(nn.Module):
    def __init__(self, hidden_size):
        super(Dynamic, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Parameter(torch.randn(136, 136, requires_grad = True, dtype = torch.float))
        self.W2 = nn.Parameter(torch.randn(136, 136, requires_grad = True, dtype = torch.float))
        self.theta0 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad = True, dtype = torch.float))
        self.theta1 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad = True, dtype = torch.float))
        self.theta2 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad = True, dtype = torch.float))
        
        self.gru = nn.GRU(136*hidden_size, 136*hidden_size)

    def forward(self, inp, hidden, k):
        actual_inp = torch.matmul(inp.view(136, self.hidden_size), self.theta0) + \
                     k*torch.matmul(torch.matmul(self.W1, inp.view(136, self.hidden_size)), self.theta1) + \
                     (1-k)*torch.matmul(torch.matmul(self.W2, inp.view(136, self.hidden_size)), self.theta2)
        actual_inp = actual_inp.view(1, 1, 136*self.hidden_size)
        output, hidden = self.gru(actual_inp, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, 136*self.hidden_size, device = device)

# The Decoder predicts the output using the latent representation
# Y_{t} = d(Z_{t})
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(136*4, 136*4)
        
    def forward(self, x):
        x = self.fc1(x)
        return x  
    
# The Dynamic process and the Decoding process, both affect the overall loss on a pass  

n_epochs = 100
rate = 0.01

encoder = Encoder().to(device)
dynamic = Dynamic(hidden_size = 4).to(device)
decoder = Decoder().to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr = rate)
dynamic_optimizer = optim.SGD(dynamic.parameters(), lr = rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr = rate)

criterion = nn.MSELoss()
lam = 0.01

for epoch in range(n_epochs):
    
    # Initializing gradients to 0
    encoder_optimizer.zero_grad()
    dynamic_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    # Encoding process
    encoder_outputs = []
    encoder.train()
    
    for x_batch, y_batch in train_loader:
        
        x_batch = x_batch.to(device)
        encoder_output = encoder(x_batch.view(-1,136*35))
    
        # List of true Z_{t} 
        encoder_outputs.append(encoder_output)
    
    # Dynamic process
    Z = encoder_outputs.copy()
    
    # True Z_{t}
    Z_t = torch.stack(Z[0:-1])
    # True Z_{t+1}
    Z_t1 = torch.stack(Z[1:])
    
    Z_data = TensorDataset(Z_t, Z_t1)
    Z_loader = DataLoader(dataset = Z_data, batch_size = 1)
       
    dynamic.train()
    dynamic_loss = 0
    # Initializing hidden state of Dynamic process
    dynamic_hidden = dynamic.initHidden()
    
    # time step
    t = 0
    for Z_t_batch, Z_t1_batch in Z_loader:
        x = X[t,:,34]
        #if disaster was declared, disaster indicator is 1
        disaster_indicator = int(1 in x)
        t = t+1
        
        Z_t_batch = Z_t_batch.to(device)
        Z_t1_batch = Z_t1_batch.to(device)

        # Calculated Z_{t+1}
        dynamic_output, dynamic_hidden = dynamic(Z_t_batch, dynamic_hidden, disaster_indicator)
    
        # Adding MSE at each time step
        dynamic_loss += criterion(Z_t1_batch, dynamic_output)
       
    # Taking average of MSE for all time steps
    dynamic_loss /= t

    # Decoder process
    decoder.train()
    decoder_loss = 0
    
    # time step
    t = 0
    for x_batch, y_batch in train_loader:
        y_batch = y_batch.to(device)

        decoder_output = decoder(encoder_outputs[t])
        t = t+1
    
        # Adding MSE at each time step
        decoder_loss += criterion(decoder_output.view(-1,136,4), y_batch)
    
    # Taking average of MSE for all time steps
    decoder_loss /= t
    
    # Both losses contribute to the updates
    loss = decoder_loss + (lam*dynamic_loss)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    dynamic_optimizer.step() 
    
    if (epoch%10 == 0):
        print(loss)

W1_final = dynamic.W1.detach().numpy()
W2_final = dynamic.W2.detach().numpy()
np.save('W_Relation_Disaster', W1_final)
np.save('W_Relation_Regular', W1_final)