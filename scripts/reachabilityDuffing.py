import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import argparse

from qutils.ml.utils import printModelParmSize, getDevice, Adam_mini
from qutils.tictoc import timer
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.regression import LSTM
from qutils.ml.utils import findDecAcc

#import for superweight identification
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight

# from nets import Adam_mini

# args parsing for model, horizon, traj_index
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mamba', help='Model to use')
parser.add_argument('--horizon', type=int, default=5, help='Horizon for prediction')
parser.add_argument('--traj_index', type=int, default=0, help='Trajectory index to plot')
parser.add_argument('--dv', type=float, default=0.3, help='Delta v for dataset')
parser.add_argument('--dt', type=float, default=0.02, help='Time step for dataset')
parser.add_argument('--n', type=int, default=20000, help='Number of trajectories in dataset')

args = parser.parse_args()
modelString = args.model
traj_index = args.traj_index


problemDim = 2

device = getDevice()


# hyperparameters
n_epochs = 10
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
horizon = 5


# load data
dataFile = './data/test/duffing_monte_carlo_trajectories_dv_{}_dt_{}_n_{}.npz'.format(args.dv, args.dt, args.n)

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
    # Convert to PyTorch tensors (keep on CPU; move batches to GPU in the loop)
    X_train = torch.tensor(np.array(X_train)).double().squeeze()
    Y_train = torch.tensor(np.array(Y_train)).double().squeeze()
    X_test = torch.tensor(np.array(X_test)).double().squeeze()
    Y_test = torch.tensor(np.array(Y_test)).double().squeeze()


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
        model = LSTM(input_size,30,output_size,num_layers,0).to(device).double()
    printModelParmSize(model)
    return model

model = returnModel(modelString)

optimizer = Adam_mini(model,lr=lr)
# optimizer = Adam_mini(model,lr=lr)

criterion = F.smooth_l1_loss
criterion = torch.nn.HuberLoss()

trainTime = timer()
for epoch in range(n_epochs):

    model.train()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    with torch.no_grad():
        def eval_batches(x_all, y_all, batch_size=256):
            loader_eval = data.DataLoader(
                data.TensorDataset(x_all, y_all),
                shuffle=False,
                batch_size=batch_size,
            )
            preds = []
            targets = []
            total_loss = 0.0
            total_count = 0
            for xb, yb in loader_eval:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                batch_loss = criterion(pred, yb).detach()
                total_loss += batch_loss.item() * xb.shape[0]
                total_count += xb.shape[0]
                preds.append(pred.cpu())
                targets.append(yb.cpu())
            pred_all = torch.cat(preds, dim=0)
            target_all = torch.cat(targets, dim=0)
            rmse = np.sqrt(total_loss / max(total_count, 1))
            return rmse, pred_all, target_all

        train_loss, y_pred_train, y_true_train = eval_batches(train_in, train_out)
        test_loss, y_pred_test, y_true_test = eval_batches(test_in, test_out)

        decAcc, err1 = findDecAcc(y_true_train, y_pred_train, printOut=False)
        decAcc, err2 = findDecAcc(y_true_test, y_pred_test)
        err = np.concatenate((err1,err2),axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

trainTime.toc()

test_loader = data.DataLoader(data.TensorDataset(test_in, test_out), shuffle=False, batch_size=256)

# plot some predictions
model.eval()
with torch.no_grad():
    xb, yb = next(iter(test_loader))
    xb = xb.to(device)
    pred = model(xb).cpu().numpy()
    yb = yb.cpu().numpy()
    xb = xb.cpu().numpy()
    traj_idx = traj_index
    def predict_last_step(x_all, batch_size=256, slice_traj_idx=None):
        loader_eval = data.DataLoader(
            data.TensorDataset(x_all),
            shuffle=False,
            batch_size=batch_size,
        )
        preds = []
        for (xb_eval,) in loader_eval:
            xb_eval = xb_eval.to(device)
            pred = model(xb_eval).cpu()
            if pred.ndim == 3:
                # If second dim isn't trajectory count, treat it as sequence and take last step.
                if x_all.ndim >= 2 and pred.shape[1] != x_all.shape[1]:
                    pred = pred[:, -1, :]
                if slice_traj_idx is not None:
                    pred = pred[:, slice_traj_idx, :]
            elif pred.ndim == 2:
                # Already (batch, problemDim); no trajectory axis to slice.
                pass
            preds.append(pred)
        return torch.cat(preds, dim=0)

    train_pred = predict_last_step(train_in, slice_traj_idx=traj_idx).numpy()
    test_pred = predict_last_step(test_in, slice_traj_idx=traj_idx).numpy()
    test_pred_full = predict_last_step(test_in)
    def build_full_seq(x_all, y_all, traj_idx):
        x_np = x_all.numpy()
        y_np = y_all.numpy()
        if x_np.ndim == 4:
            init = x_np[0, :, traj_idx, :]
        else:
            init = x_np[0, traj_idx, :][np.newaxis, :]
        y_seq = y_np[:, traj_idx, :]
        print("init shape:", init.shape)
        print("y_seq shape:", y_seq.shape)
        return np.concatenate([init, y_seq], axis=0)

    true_test_seq = build_full_seq(test_in, test_out, traj_idx)
    pred_test_seq = build_full_seq(test_in, test_pred_full, traj_idx)
    # plot
    plt.figure()
    # color train and test segments differently
    # plt.plot(train_pred[:,0], train_pred[:,1], 'g.', label='Train Predictions')
    # plt.plot(test_pred[:,0], test_pred[:,1], 'r.', label='Test Predictions')
    plt.plot(true_test_seq[:,0], true_test_seq[:,1], 'k-', label='True Trajectory')
    plt.plot(pred_test_seq[:,0], pred_test_seq[:,1], '--', label='Predicted Trajectory')
    plt.title(modelString+' Reachability Prediction: Trajectory Index '+str(traj_idx))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='best')
    # save plot
    plt.savefig("plots/"+modelString+f'_reachability_new_prediction_epoch_{n_epochs}_index_{traj_idx}.png')
    plt.close()
