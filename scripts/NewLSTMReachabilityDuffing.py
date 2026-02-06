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


modelString = 'lstm'  # 'mamba' or 'lstm'


problemDim = 2

device = getDevice()


# hyperparameters
n_epochs = 5
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
numericResult = trajs_t

# generate data sets

train_size = 5
test_size = numericResult.shape[1] - train_size

def create_datasets(data,seq_length,train_size,device):
    xs, ys = [], []
    for i in range (len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        y = y[np.newaxis, ...]  # (1, num_trajectories, problemDim)
        xs.append(x)
        ys.append(y)
    
    X_train, X_test = xs[:train_size], xs[train_size:]
    Y_train, Y_test = ys[:train_size], ys[train_size:]
    # Convert to PyTorch tensors
    X_train = torch.tensor(np.array(X_train)).double().to(device).squeeze()
    Y_train = torch.tensor(np.array(Y_train)).double().to(device).squeeze()
    X_test = torch.tensor(np.array(X_test)).double().to(device).squeeze()
    Y_test = torch.tensor(np.array(Y_test)).double().to(device).squeeze()

    # reduce test size to 500 for faster training
    X_test = X_test[:500]
    Y_test = Y_test[:500]

    return X_train,Y_train,X_test,Y_test
train_in,train_out,test_in,test_out = create_datasets(numericResult,1,train_size,device)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)

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

# plot some predictions
model.eval()
with torch.no_grad():
    xb, yb = next(iter(loader))
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
    plt.plot(output_seq[:,0,0], output_seq[:,0,1], label='Predicted Trajectory')
    plt.title('LSTM Reachability Prediction')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    # save plot
    plt.savefig(f'lstm_reachability_prediction_epoch_{n_epochs}.png')
    plt.close()