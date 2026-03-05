import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import argparse
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError

from qutils.ml.utils import printModelParmSize, getDevice, Adam_mini
from qutils.tictoc import timer
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.regression import LSTM
from qutils.ml.utils import findDecAcc

from qutils.ml.superweight import printoutMaxLayerWeight, getSuperWeight, plotSuperWeight, findMambaSuperActivation, plotSuperActivation

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mamba', help='Model to use')
parser.add_argument('--horizon', type=int, default=1, help='Predict this many steps ahead (target at t+horizon)')
parser.add_argument('--lookback', type=int, default=10, help='Number of past steps fed to the model')
parser.add_argument('--train-timesteps', type=int, default=20, help='Number of time steps from each edge used as training time region')
parser.add_argument('--traj-index', type=int, default=0, help='Trajectory index to plot')
parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of trajectories to use for training (rest used for testing)')
parser.add_argument('--batch', type=int, default=256, help='Batch size for training')
parser.add_argument('--batch-test', type=int, default=128, help='Batch size for evaluation')
parser.add_argument('--n-epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training')
parser.add_argument('--ood', action='store_true', help='Whether to evaluate on OOD data with larger deltaV')
parser.add_argument('--jetson', action='store_true', help='use flag to run on jetson with smaller test size')
parser.add_argument('--dv', type=float, default=0.3, help="amount of delta v used for picking dataset")
parser.add_argument('--dt', type=float, default=0.02, help="numerical value for timestep")
parser.add_argument('--n', type=int, default=20000, help='amount of trajectories used for picking dataset')
parser.add_argument('--pdf', action='store_true', help='Whether to save plots in PDF format instead of PNG')

args = parser.parse_args()
modelString = args.model
traj_index = args.traj_index

if args.pdf:
    saveType = 'pdf'
else:
    saveType = 'png'

problemDim = 2
device = getDevice()

n_epochs = args.n_epochs
lr = args.lr
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = args.lookback
horizon = args.horizon
train_timesteps = args.train_timesteps

dataFile = './data/test/duffing_single_monte_carlo_trajectories_dv_{}_dt_{}_n_{}.npz'.format(args.dv, args.dt, args.n)

system_data = np.load(dataFile, allow_pickle=True)
dt = system_data['dt']
trajs = system_data['trajectories']  # (num_trajectories, num_time_steps, problemDim)
t = np.arange(0, trajs.shape[1] * dt, dt)

trajs_t = np.transpose(trajs, (1, 0, 2))  # (num_time_steps, num_trajectories, problemDim)
num_time_steps = trajs_t.shape[0]
numericResult = trajs_t

train_size = 5
test_size = numericResult.shape[1] - train_size


def create_datasets_spatial(data_arr, lookback, horizon, train_size, device, tw=None):
    seq_length = lookback
    if tw is None:
        tw = train_timesteps
    split_idx = int(data_arr.shape[1] * args.train_ratio)
    time_end = min(num_time_steps, data_arr.shape[0])
    train_time = np.concatenate([data_arr[:tw], data_arr[time_end - tw:time_end]], axis=0)
    test_time = data_arr[tw:time_end - tw]

    train_data = train_time[:, :split_idx, :]
    if args.jetson:
        test_data = test_time[:, split_idx:split_idx + 1000, :]
    else:
        test_data = test_time[:, split_idx:, :]

    def build_xy(d):
        xs, ys = [], []
        for i in range(len(d) - seq_length - horizon + 1):
            x = d[i:(i + seq_length)]
            y = d[i + seq_length + horizon - 1]
            xs.append(x)
            ys.append(y)
        X = np.stack(xs, axis=0)
        Y = np.stack(ys, axis=0)
        return X, Y

    X_train, Y_train = build_xy(train_data)
    X_test, Y_test = build_xy(test_data)
    X_train = torch.tensor(np.array(X_train)).float()
    Y_train = torch.tensor(np.array(Y_train)).float()
    X_test = torch.tensor(np.array(X_test)).float()
    Y_test = torch.tensor(np.array(Y_test)).float()
    return X_train, Y_train, X_test, Y_test


def create_datasets(data_arr, lookback, horizon, train_size, device, tw=None):
    seq_length = lookback
    if tw is None:
        tw = train_timesteps
    split_idx = int(data_arr.shape[1] * args.train_ratio)
    time_end = min(num_time_steps, data_arr.shape[0])
    train_time = np.concatenate([data_arr[:tw], data_arr[time_end - tw:time_end]], axis=0)
    test_time = data_arr[tw:time_end - tw]

    train_data = train_time[:, :split_idx, :]
    if args.jetson:
        test_data = test_time[:, split_idx:split_idx + 1000, :]
    else:
        test_data = test_time[:, split_idx:, :]

    def build_xy(d):
        xs, ys = [], []
        for i in range(len(d) - seq_length - horizon + 1):
            x = d[i:(i + seq_length)]
            y = d[i + seq_length + horizon - 1]
            xs.append(x)
            ys.append(y)
        X = np.stack(xs, axis=0)
        Y = np.stack(ys, axis=0)
        return X, Y

    X_train, Y_train = build_xy(train_data)
    X_test, Y_test = build_xy(test_data)
    X_train = torch.tensor(np.array(X_train)).float()
    Y_train = torch.tensor(np.array(Y_train)).float()
    X_test = torch.tensor(np.array(X_test)).float()
    Y_test = torch.tensor(np.array(Y_test)).float()
    return X_train, Y_train, X_test, Y_test


if modelString == 'mamba':
    train_in, train_out, test_in, test_out = create_datasets_spatial(numericResult, lookback, horizon, train_size, device, tw=train_timesteps)
else:
    numericalResult = numericResult.transpose(1, 0, 2)
    train_in, train_out, test_in, test_out = create_datasets(numericResult, lookback, horizon, train_size, device, tw=train_timesteps)
print(train_in.shape)
print(train_out.shape)
print(test_in.shape)
print(test_out.shape)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=args.batch)

config = MambaConfig(d_model=problemDim, n_layers=num_layers, d_conv=16)


def returnModel(modelString='mamba'):
    if modelString == 'mamba':
        model = Mamba(config).to(device).float()
    elif modelString == 'lstm':
        model = LSTM(input_size, 30, output_size, num_layers, 1).to(device).float()
    printModelParmSize(model)
    return model


def alpha_shape_segments_and_area(points, radius_quantile=0.65):
    n = points.shape[0]
    if n < 4:
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    try:
        tri = Delaunay(points)
    except QhullError:
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    simplices = tri.simplices
    p = points[simplices]
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    s = 0.5 * (a + b + c)
    area_sq = s * (s - a) * (s - b) * (s - c)
    area_sq = np.maximum(area_sq, 0.0)
    tri_area = np.sqrt(area_sq)

    valid = tri_area > 1e-12
    if not np.any(valid):
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    circum_r = np.full_like(tri_area, np.inf)
    circum_r[valid] = (a[valid] * b[valid] * c[valid]) / (4.0 * tri_area[valid])
    r_thresh = np.quantile(circum_r[valid], radius_quantile)
    keep = valid & (circum_r <= r_thresh)

    if not np.any(keep):
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    kept = simplices[keep]
    edges = np.concatenate([kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq_edges[counts == 1]

    return points[boundary_edges], tri_area[keep].sum()


model = returnModel(modelString)
modelString = modelString + '_ood' if args.ood else modelString

optimizer = Adam_mini(model, lr=lr)
criterion = torch.nn.HuberLoss()

trainTime = timer()
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        b, L, T, D_sz = X_batch.shape
        X_mamba = X_batch.permute(1, 0, 2, 3).reshape(L, b * T, D_sz)
        y_flat = y_batch.reshape(b * T, D_sz)
        y_pred = model(X_mamba)[-1]
        loss = criterion(y_pred, y_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        def eval_batches(x_all, y_all, batch_size=args.batch_test):
            loader_eval = data.DataLoader(
                data.TensorDataset(x_all, y_all),
                shuffle=True,
                batch_size=batch_size,
            )
            preds = []
            targets = []
            total_loss = 0.0
            total_count = 0
            for xb, yb in loader_eval:
                xb = xb.to(device)
                yb = yb.to(device)
                b, L, T, D_sz = xb.shape
                xb_mamba = xb.permute(1, 0, 2, 3).reshape(L, b * T, D_sz)
                yb_flat = yb.reshape(b * T, D_sz)
                pred = model(xb_mamba)[-1]
                batch_loss = criterion(pred, yb_flat).detach()
                total_loss += batch_loss.item() * (b * T)
                total_count += b * T
                preds.append(pred.reshape(b, T, D_sz).cpu())
                targets.append(yb.cpu())
            pred_all = torch.cat(preds, dim=0)
            target_all = torch.cat(targets, dim=0)
            rmse = np.sqrt(total_loss / max(total_count, 1))
            return rmse, pred_all, target_all

        train_loss, y_pred_train, y_true_train = eval_batches(train_in, train_out)
        test_loss, y_pred_test, y_true_test = eval_batches(test_in, test_out)

        decAcc, err1 = findDecAcc(y_true_train, y_pred_train, printOut=False)
        decAcc, err2 = findDecAcc(y_true_test, y_pred_test)
        err = np.concatenate((err1, err2), axis=0)

    print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

trainTime.toc()

if args.ood:
    ood_dataFile = './data/test/duffing_single_monte_carlo_trajectories_dv_{}_dt_{}_n_{}_ood.npz'.format(args.dv * 2, args.dt, args.n)
    ood_system_data = np.load(ood_dataFile, allow_pickle=True)
    ood_trajs = ood_system_data['trajectories']
    ood_trajs_t = np.transpose(ood_trajs, (1, 0, 2))
    ood_numericResult = ood_trajs_t

    if modelString == 'mamba_ood':
        _, _, test_in, test_out = create_datasets_spatial(ood_numericResult, lookback, horizon, train_size, device, tw=train_timesteps)
    else:
        ood_numericResult = ood_numericResult.transpose(1, 0, 2)
        _, _, test_in, test_out = create_datasets(ood_numericResult, lookback, horizon, train_size, device, tw=train_timesteps)


def autoregressive_rollout(model, init_window, n_steps, device):
    """
    Autoregressively roll out the model from an initial window.

    Args:
        init_window: numpy array of shape (lookback, num_trajs, D)
        n_steps: number of steps to predict forward
        device: torch device

    Returns:
        numpy array of shape (n_steps, num_trajs, D)
    """
    model.eval()
    window = torch.tensor(init_window).float()  # (L, T, D)
    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = window.to(device)  # (L, T, D) — already in Mamba (L, B, D) format
            pred = model(x)[-1].cpu()  # (T, D)
            preds.append(pred)
            # slide window forward: drop oldest step, append new prediction
            window = torch.cat([window[1:], pred.unsqueeze(0)], dim=0)
    return torch.stack(preds, dim=0).numpy()  # (n_steps, T, D)


model.eval()
with torch.no_grad():
    # test_in shape: (num_windows, L, num_trajs, D)
    # Use first window as the starting context for rollout
    init_window = test_in[0].numpy()   # (L, num_trajs, D)
    n_steps = test_out.shape[0]        # number of steps to predict = number of test windows

    print(f"Running autoregressive rollout for {n_steps} steps from initial window...")
    ar_pred = autoregressive_rollout(model, init_window, n_steps, device)
    # ar_pred shape: (n_steps, num_trajs, D)

    # True sequence: initial lookback + test targets
    # test_in[0]: (L, num_trajs, D) → use as initial steps
    # test_out: (n_steps, num_trajs, D) → true targets

    def build_true_seq(traj_idx):
        init = test_in[0, :, traj_idx, :].numpy()   # (L, D)
        targets = test_out[:, traj_idx, :].numpy()   # (n_steps, D)
        return np.concatenate([init, targets], axis=0)  # (L + n_steps, D)

    def build_ar_seq(traj_idx):
        init = test_in[0, :, traj_idx, :].numpy()   # (L, D)
        rollout = ar_pred[:, traj_idx, :]            # (n_steps, D)
        return np.concatenate([init, rollout], axis=0)  # (L + n_steps, D)

    # --- Plot: single trajectory predicted vs true ---
    traj_idx = traj_index
    true_seq = build_true_seq(traj_idx)
    pred_seq = build_ar_seq(traj_idx)

    fig, ax = plt.subplots()
    ax.plot(true_seq[:, 0], true_seq[:, 1], 'k-', label='True Trajectory')
    ax.plot(pred_seq[:, 0], pred_seq[:, 1], 'r--', label='AR Predicted Trajectory')
    ax.set_title(modelString + ' AR Rollout: Trajectory Index ' + str(traj_idx))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best')
    plt.savefig("plots/" + modelString + f'_ar_trajectory_ratio_{args.train_ratio}_epoch_{n_epochs}_index_{traj_idx}_lr_{lr}_train_timesteps_{args.horizon * 2}.{saveType}')
    plt.close()

    # --- Plot: per-dimension vs time for a random trajectory ---
    rng = np.random.default_rng(12)
    rand_traj_idx = rng.integers(0, test_in.shape[2])
    traj_true = build_true_seq(rand_traj_idx)
    traj_pred = build_ar_seq(rand_traj_idx)
    time_axis = np.arange(traj_true.shape[0]) * float(dt)

    labels = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    fig = plt.figure(figsize=(8, 6))
    for i in range(problemDim):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.plot(time_axis, traj_true[:, i], 'k-', lw=1.5, label='True')
        ax.plot(time_axis, traj_pred[:, i], 'r--', lw=1.5, label='AR Predicted')
        ax.set_xlabel('time [s]')
        ax.set_ylabel(labels[i])
        if i == 0:
            ax.legend(loc='best')
    fig.suptitle(f'AR Rollout Trajectory #{rand_traj_idx} (Pred vs True)')
    plt.tight_layout()
    plt.savefig("plots/" + modelString + f'_ar_random_trajectory_{rand_traj_idx}_epoch_{n_epochs}_lr_{lr}_train_timesteps_{args.horizon * 2}.{saveType}')
    plt.close()

    # --- Reachability: final states from AR rollout vs true ---
    final_true = test_out[-1].numpy()        # (num_trajs, D)
    final_pred_ar = ar_pred[-1]              # (num_trajs, D)

    true_segments, area_true = alpha_shape_segments_and_area(final_true, radius_quantile=0.95)
    pred_segments, area_pred = alpha_shape_segments_and_area(final_pred_ar, radius_quantile=0.95)

    area_true = float(area_true)
    area_pred = float(area_pred)
    area_ratio = area_pred / area_true if area_true > 0 else float('inf')

    print(f"True Alpha-Shape Area: {area_true:.4f}, AR Pred Alpha-Shape Area: {area_pred:.4f}, Area Ratio (Pred/True): {area_ratio:.4f}")

    fig, ax = plt.subplots()
    ax.scatter(final_true[:, 0], final_true[:, 1], s=6, alpha=0.4, label='True Final States')
    ax.scatter(final_pred_ar[:, 0], final_pred_ar[:, 1], s=6, alpha=0.4, label='AR Pred Final States')
    for i, seg in enumerate(true_segments):
        ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]], c='k', lw=2,
                label='True Alpha Shape' if i == 0 else None)
    for i, seg in enumerate(pred_segments):
        ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]], c='r', lw=2, ls='--',
                label='AR Pred Alpha Shape' if i == 0 else None)
    ax.set_title(modelString + ' AR Final-State Alpha Shapes: Area Ratio {:.4f}'.format(area_ratio))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best')
    plt.savefig("plots/" + modelString + f'_ar_final_state_alpha_shapes_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}_train_timesteps_{args.horizon * 2}.{saveType}')
    plt.close()

    fig, ax = plt.subplots()
    ax.scatter(final_true[:, 0], final_true[:, 1], s=6, alpha=0.4, label='True Final States')
    ax.scatter(final_pred_ar[:, 0], final_pred_ar[:, 1], s=6, alpha=0.4, label='AR Pred Final States')
    ax.set_title(modelString + ' AR Final-State Points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best')
    plt.savefig("plots/" + modelString + f'_ar_final_state_points_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}_train_timesteps_{args.horizon * 2}.{saveType}')
    plt.close()

    if modelString == 'mamba':
        test_loader = data.DataLoader(data.TensorDataset(test_in, test_out), shuffle=False, batch_size=args.batch_test)
        xb, yb = next(iter(test_loader))
        b, L, T, D_sz = xb.shape
        xb_one_traj = xb[:, :, traj_index:traj_index + 1, :]
        xb_one_traj = xb_one_traj.permute(1, 0, 2, 3).reshape(L, b, D_sz)
        magnitude, index = findMambaSuperActivation(model, xb_one_traj.to(device))

        normedMagsMRP = np.zeros((len(magnitude),))
        for i in range(len(magnitude)):
            normedMagsMRP[i] = magnitude[i].norm().detach().cpu()

        printoutMaxLayerWeight(model)
        getSuperWeight(model)
        plotSuperWeight(model)
        plotSuperActivation(magnitude, index, printOutValues=True)
        plt.title("Mamba AR Reachability Super Activations")
        plt.savefig("plots/" + modelString + f'_ar_super_activations_ratio_{args.train_ratio}_epoch_{n_epochs}_index_{traj_index}_lr_{lr}_train_timesteps_{args.horizon * 2}.{saveType}')
