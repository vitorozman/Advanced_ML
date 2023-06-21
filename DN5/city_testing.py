import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# Define LSTM model for time series prediction
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        out = out.view(batch_size, seq_len, -1)

        return out[:, -7:, :]

import pandas as pd

okuzeni_all = pd.read_csv('DN5/okuzeni.csv')
NAMES = [ "ljubljana", "maribor", "kranj", "koper", "celje", "novo_mesto", "velenje", "nova_gorica", "krško", "ptuj", "murska_sobota", "slovenj_gradec"]
NAMES = 'velenje'
#NAMES = ['ljubljana', 'maribor']
data = torch.tensor(okuzeni_all[NAMES].values).float()


scaler = MinMaxScaler()
from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
data_normalized = scaler.fit_transform(data.reshape(-1, 1))

# Prepare input and target tensors

seq_len = 15  # Previous 20 steps
pred_len = 7  # Predict next 7 steps

def ceate_sequence(data, seq_len, pred_len):
    input_seq = []
    target_seq = []
    for i in range(len(data) - seq_len - pred_len + 1):
        input_seq.append(data[i:i+seq_len])
        target_seq.append(data[i+seq_len:i+seq_len+pred_len])

    input_seq = torch.tensor(input_seq, dtype=torch.float32)
    target_seq = torch.tensor(target_seq, dtype=torch.float32)
    return input_seq, target_seq

input_seq, target_seq = ceate_sequence(data_normalized, seq_len, pred_len)

# Define hyperparameters
input_size = 1  # Dimension of input at each time step
hidden_size = 64  # Number of LSTM units
output_size = 1  # Dimension of output at each time step
batch_size = 16
num_epochs = 100
learning_rate = 0.001



model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for i in range(0, len(input_seq), batch_size):
        optimizer.zero_grad()
        inputs = input_seq[i:i+batch_size]
        targets = target_seq[i:i+batch_size]
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')


# Make predictions
model.eval()

# Use the last 20 steps of data to make predictions for the next 7 steps
test_input = data_normalized[:seq_len].reshape(1, seq_len, input_size)
#print(test_input.shape)
test_input = torch.tensor(test_input, dtype=torch.float32)
#print(test_input.shape)
from sklearn.metrics import mean_squared_error
err = 0
for i in range(len(target_seq)):
    with torch.no_grad():
        inputs = test_input[:,i:,:]
        #inputs = input_seq[i].reshape(1, 20, 1)
        predictions = model(inputs)
    err += mean_squared_error(target_seq[i], predictions[0])
    popped_element = predictions[:, 0:1, :]
    # Reshape the popped element to size [1, 1, 1]
    reshaped_element = popped_element.reshape(1, 1, 1)
    
    test_input = torch.cat((test_input, reshaped_element), dim=1)
#print(predictions.view(-1, output_size).numpy())
# Denormalize predictions
#predictions = scaler.inverse_transform(predictions.view(-1, output_size).numpy())

pred = scaler.inverse_transform(test_input.view(-1, output_size).numpy())
#print(scaler.inverse_transform(test_input.view(-1, output_size).numpy()))
print(f"MSE: {err}")
#print(predictions)

import matplotlib.pyplot as plt

plt.plot(data)
#plt.show()
plt.plot(pred)
plt.show()


