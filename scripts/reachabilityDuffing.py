import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo
import os 

from qutils.ml.utils import printModelParmSize, getDevice, Adam_mini
from qutils.tictoc import timer
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.regression import LSTM
from qutils.ml.utils import findDecAcc

#import for superweight identification
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight

# from nets import Adam_mini


modelString = 'mamba'  # 'mamba' or 'lstm'


problemDim = 2

device = getDevice()


# hyperparameters
n_epochs = 10
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
horizon = 10


# load data
dataFile = './data/test/duffing_monte_carlo_trajectories_dv_0.3_dt_100_n_20000.npz'

system_data = np.load(dataFile,allow_pickle=True)
dt = system_data['dt']
trajs = system_data['trajectories'] # shape (num_trajectories, num_time_steps, problemDim)
t = np.arange(0,trajs.shape[1]*dt,dt)

# reshape numericResult to be (num_time_steps, num_trajectories, problemDim)
trajs_t = np.transpose(trajs, (1, 0, 2))  # (num_time_steps, num_trajectories, problemDim)
num_time_steps = trajs_t.shape[0]
numericResult = trajs_t

# generate data sets

train_size = 5
test_size = numericResult.shape[1] - train_size

def create_datasets(data, seq_length, train_size, device):
    # Split across dimension 0 (time): use edge windows for train (size ~2*horizon)
    # and middle window for test, while keeping 80-20 split across trajectories.
    split_idx = int(data.shape[1] * 0.8)
    time_end = min(num_time_steps, data.shape[0])
    train_time = np.concatenate([data[:horizon], data[time_end - horizon:time_end]], axis=0)
    test_time = data[horizon:time_end - horizon]

    train_data = train_time[:, :split_idx, :]
    test_data = test_time[:, split_idx:, :]

    def build_xy(d):
        xs, ys = [], []
        for i in range(len(d) - seq_length):
            x = d[i:(i + seq_length)]
            y = d[i + seq_length]
            y = y[np.newaxis, ...]  # (1, num_trajectories, problemDim)
            xs.append(x)
            ys.append(y)
        return xs, ys

    X_train, Y_train = build_xy(train_data)
    X_test, Y_test = build_xy(test_data)
    # Convert to PyTorch tensors
    X_train = torch.tensor(np.array(X_train)).double().to(device).squeeze()
    Y_train = torch.tensor(np.array(Y_train)).double().to(device).squeeze()
    X_test = torch.tensor(np.array(X_test)).double().to(device).squeeze()
    Y_test = torch.tensor(np.array(Y_test)).double().to(device).squeeze()


    return X_train,Y_train,X_test,Y_test
train_in,train_out,test_in,test_out = create_datasets(numericResult,1,train_size,device)
print(train_in.shape)
print(train_out.shape)
print(test_in.shape)
print(test_out.shape)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=256)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=16)

def returnModel(modelString = 'mamba'):
    if modelString == 'mamba':
        model = Mamba(config).to(device).double()
    elif modelString == 'lstm':
        model = LSTM(input_size,30,output_size,num_layers,0).double().to(device)
    return model

model = returnModel(modelString)

optimizer = Adam_mini(model,lr=lr)
# optimizer = Adam_mini(model,lr=lr)

criterion = F.smooth_l1_loss
criterion = torch.nn.HuberLoss()

trainTime = timer()
for epoch in range(n_epochs):

    # trajPredition = plotPredition(epoch,model,'target',t=t*TU,output_seq=pertNR)

    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    with torch.no_grad():
        y_pred_train = model(train_in)
        train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
        y_pred_test = model(test_in)
        test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

        decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
        decAcc, err2 = findDecAcc(test_out,y_pred_test)
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

trainTime.toc()

test_loader = data.DataLoader(data.TensorDataset(test_in, test_out), shuffle=False, batch_size=256)

# plot some predictions
model.eval()
with torch.no_grad():
    xb, yb = next(iter(test_loader))
    pred = model(xb).cpu().numpy()
    yb = yb.cpu().numpy()
    xb = xb.cpu().numpy()
    seq_length = xb.shape[1]
    train_size = train_in.shape[0]
    output_seq = np.zeros((seq_length + train_size + test_in.shape[0], problemDim))
    # shift train predictions for plotting
    train_plot = np.ones_like(output_seq) * np.nan
    train_plot[seq_length:train_size+seq_length] = model(train_in)[:, -1, :].cpu()
    # shift test predictions for plotting
    test_plot = np.ones_like(output_seq) * np.nan
    test_plot[train_size+seq_length:] = model(test_in)[:, -1, :].cpu()
    # combine for full output sequence
    output_seq[seq_length:train_size+seq_length] = train_plot[seq_length:train_size+seq_length]
    output_seq[train_size+seq_length:] = test_plot[train_size+seq_length:]
    # fill in initial conditions
    output_seq[:seq_length] = xb[0]
    # plot
    plt.figure()
    # plt.plot(output_seq[:,0], output_seq[:,1], label='Predicted Trajectory')
    # color train and test segments differently
    plt.plot(train_plot[:,0], train_plot[:,1], 'g.', label='Train Predictions')
    plt.plot(test_plot[:,0], test_plot[:,1], 'r.', label='Test Predictions')
    plt.title(modelString+' Reachability Prediction')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    # save plot
    plt.savefig(modelString+f'_reachability_new_prediction_epoch_{n_epochs}.png')
    plt.close()
