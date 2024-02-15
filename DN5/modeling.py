import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

okuzeni_all = pd.read_csv('okuzeni.csv').fillna(0)
NAMES = [ "ljubljana", "maribor", "kranj", "koper", "celje", "novo_mesto", "velenje", "nova_gorica", "krško", "ptuj", "murska_sobota", "slovenj_gradec"]

okuzeni = torch.tensor(np.array(okuzeni_all[NAMES].values)).float()
okuzeni_diff = okuzeni.diff(axis=0)
okuzeni_log = torch.log(okuzeni)

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

torch.manual_seed(1234)


TOT_err = []
##########################################################################################
# Hiperparems
##########################################################################################
    
seq_len = 20  
pred_len = 7  
input_size = 1 
hidden_size = 32  
output_size = 1  
batch_size = 32
num_epochs = 100
learning_rate = 0.001

err_day = {}
for i in range(pred_len):
    dan = f"dan{i+1}"
    err_day[dan] = 0

for name in NAMES:


    ##########################################################################################
    # Prepar data
    ##########################################################################################
    data = torch.tensor(okuzeni_all[name].values).float()
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    
    input_seq = []
    target_seq = []
    for i in range(len(data_normalized) - seq_len - pred_len + 1):
        input_seq.append(data_normalized[i:i+seq_len])
        target_seq.append(data_normalized[i+seq_len:i+seq_len+pred_len])

    input_seq = torch.tensor(input_seq, dtype=torch.float32)
    target_seq = torch.tensor(target_seq, dtype=torch.float32)


    ##########################################################################################
    # Train
    ##########################################################################################
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
        # v casu ucenja
        #if (epoch+1) % 10 == 0:
        #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')



    ##########################################################################################
    # Evaluation
    ##########################################################################################
    model.eval()
    test_input = data_normalized[:seq_len].reshape(1, seq_len, input_size)
    test_input = torch.tensor(test_input, dtype=torch.float32)
    # evaluacija modela in izracun napak
    tot_err = 0
    for i in range(len(target_seq)):
        with torch.no_grad():
            inputs = test_input[:,i:,:]
            #t = input_seq[i].reshape(1, 20, 1)
            predictions = model(inputs)
        y_true = target_seq[i,:,:]
        y_pred = predictions.reshape(pred_len,1)
    
        tot_err += mean_squared_error(y_true, y_pred)
        for j, (dan, _) in enumerate(err_day.items()):
            err_day[dan] += mean_squared_error(y_true[j], y_pred[j])

        popped_element = predictions[:, 0:1, :]
        reshaped_element = popped_element.reshape(1, 1, 1)        
        test_input = torch.cat((test_input, reshaped_element), dim=1)

    TOT_err.append(tot_err)
    pred = scaler.inverse_transform(test_input.view(-1, output_size).numpy())

    
    ##########################################################################################
    # izris grafov 
    ##########################################################################################


    #plt.plot(data)
    #plt.plot(pred)
    #plt.xlabel('Days')
    #plt.ylabel('Number')
    #plt.title(f"M = {pred_len}, Location: {name}")
    #plt.show()

##########################################################################################
# izris grafov 
##########################################################################################


print(f"MSE {pred_len}: {np.mean(TOT_err)}")

x = [i+1 for i,_ in enumerate(err_day)]
y = [ls/len(NAMES) for _,ls in err_day.items()]

plt.plot(x,y)
plt.xlabel('Day')
plt.ylabel('MSE')
plt.title(f"Prediction error by days M={pred_len}")
plt.show()



# Define LSTM model for time series prediction
class LSTMModel30(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel30, self).__init__()
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
        
        return out[:, -30:, :]
    


TOT_err = []

##########################################################################################
# Hiperparems
##########################################################################################

seq_len = 30  
pred_len = 30  
input_size = 1 
hidden_size = 64  
output_size = 1  
batch_size = 32
num_epochs = 100
learning_rate = 0.001

err_day = {}
for i in range(pred_len):
    dan = f"dan{i+1}"
    err_day[dan] = 0

for name in NAMES:

    ##########################################################################################
    # Prepar data
    ##########################################################################################

    data = torch.tensor(okuzeni_all[name].values).float()
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    input_seq = []
    target_seq = []
    for i in range(len(data_normalized) - seq_len - pred_len + 1):
        input_seq.append(data_normalized[i:i+seq_len])
        target_seq.append(data_normalized[i+seq_len:i+seq_len+pred_len])

    input_seq = torch.tensor(input_seq, dtype=torch.float32)
    target_seq = torch.tensor(target_seq, dtype=torch.float32)

    
    ##########################################################################################
    # Train
    ##########################################################################################
    model = LSTMModel30(input_size, hidden_size, output_size)
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
        # v casu ucenja
        #if (epoch+1) % 10 == 0:
        #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')


    ##########################################################################################
    # Evaluation
    ##########################################################################################
    model.eval()
    test_input = data_normalized[:seq_len].reshape(1, seq_len, input_size)
    test_input = torch.tensor(test_input, dtype=torch.float32)
    # evaluacija modela in izracun napak
    tot_err = 0
    for i in range(len(target_seq)):
        with torch.no_grad():
            inputs = test_input[:,i:,:]
            #t = input_seq[i].reshape(1, 20, 1)
            predictions = model(inputs)
        y_true = target_seq[i,:,:]
        y_pred = predictions.reshape(pred_len,1)
    
        tot_err += mean_squared_error(y_true, y_pred)
        for j, (dan, _) in enumerate(err_day.items()):
            err_day[dan] += mean_squared_error(y_true[j], y_pred[j])

        popped_element = predictions[:, 0:1, :]
        reshaped_element = popped_element.reshape(1, 1, 1)        
        test_input = torch.cat((test_input, reshaped_element), dim=1)

    TOT_err.append(tot_err)
    pred = scaler.inverse_transform(test_input.view(-1, output_size).numpy())

    ##########################################################################################
    # Izris garfov
    ##########################################################################################
 
    #plt.plot(data)
    #plt.plot(pred)
    #plt.xlabel('Days')
    #plt.ylabel('Number')
    #plt.title(f"M = {pred_len}, Location: {name}")
    #plt.show()


##########################################################################################
# Izris grafa
##########################################################################################

print(f"MSE {pred_len}: {np.mean(TOT_err)}")

x = [i+1 for i,_ in enumerate(err_day)]
y = [ls/len(NAMES) for _,ls in err_day.items()]

plt.plot(x,y)
plt.xlabel('Day')
plt.ylabel('MSE')
plt.title(f"Prediction error by days M={pred_len}")
plt.show()