import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchinfo
import os 

from qutils.ml.utils import printModelParmSize, getDevice, Adam_mini
from qutils.tictoc import timer

#import for superweight identification
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight

# from nets import Adam_mini


DEBUG = True
plotOn = True
randomIC = False
periodic = False
printoutSuperweight = True
compareLSTM = True

problemDim = 2

device = getDevice()


# hyperparameters
n_epochs = 5
lr = 0.001
input_size = problemDim
output_size = problemDim
num_layers = 1
n_seq = 5
horizon = 10
teacher_forcing_ratio = 0.5


# load data
dataFile = './data/test/duffing_monte_carlo_trajectories_dv_0.3_dt_100_n_20000.npz'

system_data = np.load(dataFile,allow_pickle=True)
numericResult = system_data['trajectories'] # shape (num_trajectories, num_time_steps, problemDim)

dt = system_data['dt']
t = np.arange(0,numericResult.shape[1]*dt,dt)
# generate data sets

train_size = 5
test_size = numericResult.shape[1] - train_size


def make_sequence_dataset(trajectories, lookback, horizon=1):
    # Build (X, y) sequences from trajectories.
    # trajectories: (num_trajectories, num_time_steps, problemDim)
    x_list = []
    y_list = []
    for traj in trajectories:
        t_steps = traj.shape[0]
        end_idx = t_steps - lookback - horizon + 1
        for i in range(end_idx):
            x_list.append(traj[i:i + lookback])
            y_list.append(traj[i + lookback:i + lookback + horizon])
    x = np.stack(x_list, axis=0)
    y = np.stack(y_list, axis=0)
    return x, y


class TrajectoryDataset(data.Dataset):
    def __init__(self, x, y, device):
        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LSTMReachability(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, horizon):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.encoder = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = torch.nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, y=None, teacher_forcing_ratio=0.0):
        # x: (batch, n_seq, input_size)
        _, (h, c) = self.encoder(x)
        decoder_input = x[:, -1, :]  # start from last input state

        outputs = []
        for t in range(self.horizon):
            dec_out, (h, c) = self.decoder(decoder_input.unsqueeze(1), (h, c))
            pred = self.fc(dec_out[:, -1, :])
            outputs.append(pred)
            if y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y[:, t, :]
            else:
                decoder_input = pred
        return torch.stack(outputs, dim=1)


hidden_size = 32
batch_size = 256
train_frac = 0.8

num_traj = numericResult.shape[0]
split_idx = int(num_traj * train_frac)
train_traj = numericResult[:split_idx]
test_traj = numericResult[split_idx:]

x_train, y_train = make_sequence_dataset(train_traj, n_seq, horizon=horizon)
x_test, y_test = make_sequence_dataset(test_traj, n_seq, horizon=horizon)

train_ds = TrajectoryDataset(x_train, y_train, device)
test_ds = TrajectoryDataset(x_test, y_test, device)

train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

model = LSTMReachability(input_size, hidden_size, output_size, num_layers, horizon).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

if DEBUG:
    print(model)
    printModelParmSize(model)

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb, yb, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb, teacher_forcing_ratio=0.0)
            loss = loss_fn(pred, yb)
            test_loss += loss.item() * xb.size(0)
    test_loss /= len(test_loader.dataset)

    print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_loss:.6f} - test_loss: {test_loss:.6f}")
    
# plot some predictions
model.eval()
with torch.no_grad():
    xb, yb = next(iter(test_loader))
    pred = model(xb, teacher_forcing_ratio=0.0).cpu().numpy()
    yb = yb.cpu().numpy()
    xb = xb.cpu().numpy()
    for i in range(5):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        input_steps = np.arange(n_seq)
        pred_steps = np.arange(n_seq, n_seq + horizon)
        transition_x = n_seq - 0.5

        axes[0].plot(input_steps, xb[i, :, 0], color='k', label='Input x1')
        axes[0].plot(pred_steps, yb[i, :, 0], color='g', label='True x1')
        axes[0].plot(pred_steps, pred[i, :, 0], color='g', linestyle='--', label='Pred x1')
        axes[0].axvline(transition_x, color='0.5', linestyle=':')
        axes[0].axvspan(n_seq - 0.5, n_seq + horizon - 0.5, color='0.9', alpha=0.5)
        axes[0].set_ylabel('x1')
        axes[0].legend(loc='upper left')

        axes[1].plot(input_steps, xb[i, :, 1], color='k', label='Input x2')
        axes[1].plot(pred_steps, yb[i, :, 1], color='r', label='True x2')
        axes[1].plot(pred_steps, pred[i, :, 1], color='r', linestyle='--', label='Pred x2')
        axes[1].axvline(transition_x, color='0.5', linestyle=':')
        axes[1].axvspan(n_seq - 0.5, n_seq + horizon - 0.5, color='0.9', alpha=0.5)
        axes[1].set_ylabel('x2')
        axes[1].set_xlabel('Time Step')
        axes[1].legend(loc='upper left')

        fig.suptitle(f'Seq2Seq Prediction (Test Sample {i+1})')
        fig.tight_layout()
        fig.savefig(f'seq2seq_prediction_sample_{i+1}.png')
        plt.close(fig)
# compute average error on test set
model.eval()
total_error = 0.0
num_samples = 0
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb, teacher_forcing_ratio=0.0)
        error = torch.sqrt(torch.mean((pred - yb) ** 2, dim=(1, 2)))  # RMSE per sample
        total_error += torch.sum(error).item()
        num_samples += xb.size(0)
avg_error = total_error / num_samples
print(f'Average RMSE on test set: {avg_error:.6f}')
