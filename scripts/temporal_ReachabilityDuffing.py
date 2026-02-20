import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
import torch.nn.functional as F
import torch.utils.data as data
import argparse
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError # import here for p36 compatibility


from qutils.ml.utils import printModelParmSize, getDevice, Adam_mini
from qutils.tictoc import timer
from qutils.ml.utils import findDecAcc
from qutils.ml.mamba import Mamba, MambaConfig
#import for superweight identification
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight, findMambaSuperActivation,plotSuperActivation


# args parsing for model, horizon, traj_index
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm', help='Model to use')
parser.add_argument('--horizon', type=int, default=1, help='Predict this many steps ahead (target at t+horizon)')
parser.add_argument('--lookback', type=int, default=4, help='Number of past steps fed to the LSTM')
parser.add_argument('--hidden', type=int, default=64, help='LSTM hidden size')
parser.add_argument('--layers', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout (only effective if layers>1, depending on implementation)')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay for AdamW')
parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping norm')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--train-timesteps', type=int, default=5, help='Use first M timesteps in each trajectory for training')

parser.add_argument('--traj-index', type=int, default=0, help='Trajectory index to plot')
parser.add_argument('--dv', type=float, default=0.3, help='Delta v for dataset')
parser.add_argument('--dt', type=float, default=0.02, help='Time step for dataset')
parser.add_argument('--n', type=int, default=20000, help='Number of trajectories in dataset')
parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of trajectories to use for training (rest used for testing)')
parser.add_argument('--batch', type=int, default=256, help='Batch size for training')
parser.add_argument('--batch-test', type=int, default=512, help='Batch size for evaluation')
parser.add_argument('--n-epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
parser.add_argument('--ood', action='store_true', help='Whether to evaluate on OOD data with larger deltaV')
parser.add_argument('--jetson', action='store_true', help='use flag to run on jetson with smaller test size')

parser.add_argument("--mamba-d-state", type=int, default=16, help="Mamba state size")
parser.add_argument("--mamba-expand", type=int, default=2, help="Mamba expand factor")
parser.add_argument("--mamba-d-conv", type=int, default=4, help="Mamba local conv kernel size")
parser.add_argument("--mamba-dt-rank", type=str, default="auto", help="Mamba dt rank ('auto' or integer)")
parser.add_argument("--mamba-no-pscan", action="store_true", help="Disable parallel scan path in Mamba")

args = parser.parse_args()
modelString = args.model
traj_index = args.traj_index


problemDim = 2

device = getDevice()


# hyperparameters
input_size = problemDim
output_size = problemDim

n_epochs = args.n_epochs
lr = args.lr
num_layers = args.layers
lookback = args.lookback
horizon = args.horizon


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

def create_datasets(data_TND, lookback, horizon, train_ratio=0.8, train_timesteps=None, jetson=False):
    """
    data_TND: (T, N, D)
    Returns:
      X_train: (S_train, lookback, D)
      Y_train: (S_train, D)
      X_test : (S_test,  lookback, D)
      Y_test : (S_test,  D)
      norm: dict with mean/std for de/normalization
      meta: dict with window counts and traj counts for extracting per-time slices
    """
    T, N, D = data_TND.shape
    min_required = lookback + horizon
    split_t = train_timesteps if train_timesteps is not None else int(T * train_ratio)
    if split_t < min_required:
        raise ValueError(
            f"train_timesteps must be >= lookback+horizon ({min_required}), got {split_t}"
        )
    if T - split_t < min_required:
        raise ValueError(
            f"Not enough test timesteps after split: T={T}, split_t={split_t}, "
            f"required test timesteps >= {min_required}"
        )

    train = data_TND[:split_t, :, :]   # (Ttr, N, D)
    test  = data_TND[split_t:, :, :]   # (Tte, N, D)

    if jetson:
        test = test[:, :min(test.shape[1], 1000), :]

    def build_xy(block_TND):
        T_, N_, D_ = block_TND.shape
        W = T_ - lookback - horizon + 1
        if W <= 0:
            raise ValueError(f"Not enough timesteps: T={T_}, lookback={lookback}, horizon={horizon}")
        # X: (W, lookback, N_, D_)
        X = np.stack([block_TND[i:i+lookback] for i in range(W)], axis=0)
        # Y at time i+lookback+horizon-1: (W, N_, D_)
        Y = block_TND[lookback + horizon - 1 : lookback + horizon - 1 + W]
        # reshape to samples per trajectory
        X = X.transpose(0, 2, 1, 3).reshape(W * N_, lookback, D_)  # (W*N_, lookback, D_)
        Y = Y.reshape(W * N_, D_)                                  # (W*N_, D_)
        return X, Y, W, N_

    Xtr, Ytr, Wtr, Ntr = build_xy(train)
    Xte, Yte, Wte, Nts = build_xy(test)

    # Normalization from TRAIN only (apply to X and Y)
    mu = Xtr.reshape(-1, D).mean(axis=0)
    sig = Xtr.reshape(-1, D).std(axis=0)
    sig = np.where(sig < 1e-8, 1.0, sig)

    Xtr = (Xtr - mu) / sig
    Ytr = (Ytr - mu) / sig
    Xte = (Xte - mu) / sig
    Yte = (Yte - mu) / sig

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Ytr = torch.tensor(Ytr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    Yte = torch.tensor(Yte, dtype=torch.float32)

    norm = {"mu": torch.tensor(mu, dtype=torch.float32), "sig": torch.tensor(sig, dtype=torch.float32)}
    meta = {"W_train": Wtr, "N_train": Ntr, "W_test": Wte, "N_test": Nts, "split_t": split_t}
    return Xtr, Ytr, Xte, Yte, norm, meta



train_in, train_out, test_in, test_out, norm, meta = create_datasets(
    numericResult,
    lookback=lookback,
    horizon=horizon,
    train_ratio=args.train_ratio,
    train_timesteps=args.train_timesteps,
    jetson=args.jetson
)

print(train_in.shape, train_out.shape, test_in.shape, test_out.shape)
loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=args.batch, pin_memory=True)



import torch
import torch.nn as nn
class MambaRegressor(nn.Module):
    def __init__(self, input_size, d_model, output_size, n_layers=2, dropout=0.1, d_state=16, expand_factor=2, d_conv=4, dt_rank="auto", pscan=True):
        super().__init__()
        config = MambaConfig(
            d_model=d_model,
            n_layers=n_layers,
            dt_rank=dt_rank,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            pscan=pscan,
            classifer=False,
        )
        self.in_proj = nn.Linear(input_size, d_model)
        self.backbone = Mamba(config)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_size),
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected x of shape (B, T, D), got {tuple(x.shape)}")
        h = self.in_proj(x)
        h = self.backbone(h)
        h_last = h[:, -1, :]
        y = self.head(h_last)
        return y


def parse_dt_rank(value):
    if value == "auto":
        return value
    return int(value)

class SimpleLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

        # Better default init than PyTorch’s raw defaults for regression
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x):
        # x: (B, T, D)
        if x.ndim != 3:
            raise ValueError(f"Expected x of shape (B, T, D), got {tuple(x.shape)}")

        out, _ = self.lstm(x)       # out: (B, T, H)
        h_last = out[:, -1, :]      # (B, H)
        y = self.head(h_last)       # (B, output_size)
        return y


def returnModel(modelString='lstm'):
    if modelString == 'lstm':
        model = SimpleLSTMRegressor(
            input_size=input_size,
            hidden_size=args.hidden,      # from argparse
            output_size=output_size,
            num_layers=args.layers,
            dropout=args.dropout,
        ).to(device).float()
    elif modelString == 'mamba':
        model = MambaRegressor(
            input_size=input_size,
            d_model=args.hidden,
            output_size=output_size,
            n_layers=args.layers,
            dropout=args.dropout,
            d_state=args.mamba_d_state,
            expand_factor=args.mamba_expand,
            d_conv=args.mamba_d_conv,
            dt_rank=parse_dt_rank(args.mamba_dt_rank),
            pscan=(not args.mamba_no_pscan),
        ).to(device).float()

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
    edges = np.concatenate(
        [kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]],
        axis=0
    )
    edges = np.sort(edges, axis=1)
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq_edges[counts == 1]

    return points[boundary_edges], tri_area[keep].sum()

model = returnModel(modelString)
modelString = modelString + '_ood' if args.ood else modelString

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
criterion = torch.nn.MSELoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
)

use_amp = (device.type == 'cuda')
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

trainTime = timer()
best_test = float('inf')

for epoch in range(n_epochs):
    model.train()
    total_train_loss = 0.0
    total_train_count = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.detach().item() * X_batch.shape[0]
        total_train_count += X_batch.shape[0]

    model.eval()
    with torch.no_grad():
        def eval_rmse(x_all, y_all, batch_size):
            loader_eval = data.DataLoader(
                data.TensorDataset(x_all, y_all),
                shuffle=False,
                batch_size=batch_size,
                pin_memory=True
            )
            se_sum = 0.0
            n_sum = 0
            preds = []
            targets = []
            for xb, yb in loader_eval:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                se_sum += torch.sum((pred - yb) ** 2).item()
                n_sum += yb.numel()
                preds.append(pred.cpu())
                targets.append(yb.cpu())
            rmse = np.sqrt(se_sum / max(n_sum, 1))
            return rmse, torch.cat(preds, dim=0), torch.cat(targets, dim=0)

        train_rmse, y_pred_train, y_true_train = eval_rmse(train_in, train_out, args.batch_test)
        test_rmse,  y_pred_test,  y_true_test  = eval_rmse(test_in,  test_out,  args.batch_test)

        # Optional diagnostic metric you already use
        decAcc, err1 = findDecAcc(y_true_train, y_pred_train, printOut=False)
        decAcc, err2 = findDecAcc(y_true_test, y_pred_test)
        err = np.concatenate((err1, err2), axis=0)

    scheduler.step(test_rmse)

    if test_rmse < best_test:
        best_test = test_rmse

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch:03d}: train RMSE {train_rmse:.6f}, test RMSE {test_rmse:.6f}, lr {lr_now:.2e}")

trainTime.toc()


if args.ood:
    # Load OOD test data with larger deltaV
    ood_dataFile = './data/test/duffing_monte_carlo_trajectories_dv_{}_dt_{}_n_{}_ood.npz'.format(args.dv*2, args.dt, args.n)
    ood_system_data = np.load(ood_dataFile, allow_pickle=True)
    ood_trajs = ood_system_data['trajectories']
    ood_trajs_t = np.transpose(ood_trajs, (1, 0, 2))
    ood_numericResult = ood_trajs_t

    # Create OOD test dataset
    _, _, test_in, test_out, _, _ = create_datasets(
        ood_numericResult,
        lookback=lookback,
        horizon=horizon,
        train_ratio=args.train_ratio,
        train_timesteps=args.train_timesteps,
        jetson=args.jetson,
    )
# plot some predictions
# De-normalize helper
mu = norm["mu"]
sig = norm["sig"]

def denorm(x):
    # x: torch or np
    if isinstance(x, np.ndarray):
        return x * sig.numpy() + mu.numpy()
    return x * sig + mu

# Extract last window slice (time = final available) across ALL test trajectories
Wte = meta["W_test"]
Nts = meta["N_test"]
start = (Wte - 1) * Nts
end = Wte * Nts

model.eval()
with torch.no_grad():
    xb_last = test_in[start:end].to(device)
    pred_last = model(xb_last).cpu()
    true_last = test_out[start:end].cpu()

final_true = denorm(true_last).numpy()
final_pred = denorm(pred_last).numpy()

true_segments, area_true = alpha_shape_segments_and_area(final_true, radius_quantile=0.95)
pred_segments, area_pred = alpha_shape_segments_and_area(final_pred, radius_quantile=0.95)

area_true = float(area_true)
area_pred = float(area_pred)
area_ratio = (area_pred) / area_true if area_true > 0 else float('inf')

print(f"True Alpha-Shape Area: {area_true:.4f}, Pred Alpha-Shape Area: {area_pred:.4f}, Area Ratio (Pred/True): {area_ratio:.4f}")

true_segments, area_true = alpha_shape_segments_and_area(final_true,radius_quantile=0.95)
pred_segments, area_pred = alpha_shape_segments_and_area(final_pred,radius_quantile=0.95)

# calculate reachable-set areas and find ratio of pred area to true area
area_true = float(area_true)
area_pred = float(area_pred)

area_ratio = (area_pred) / area_true if area_true > 0 else float('inf')

print(f"True Alpha-Shape Area: {area_true:.4f}, Pred Alpha-Shape Area: {area_pred:.4f}, Area Ratio (Pred/True): {area_ratio:.4f}")

plt.figure()
plt.scatter(final_true[:, 0], final_true[:, 1], s=6, alpha=0.4, label='True Final States')
plt.scatter(final_pred[:, 0], final_pred[:, 1], s=6, alpha=0.4, label='Pred Final States')
ax = plt.gca()
ax.add_collection(LineCollection(true_segments, colors='k', linewidths=2, label='True Alpha Shape'))
ax.add_collection(LineCollection(pred_segments, colors='r', linewidths=2, linestyles='--', label='Pred Alpha Shape'))
ax.autoscale_view()

plt.title(modelString + ' Final-State Alpha Shapes: Area Ratio {:.4f}'.format(area_ratio))
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.savefig("plots/" + modelString + f'_final_state_alpha_shapes_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}.png')
plt.close()

plt.figure()
plt.scatter(final_true[:, 0], final_true[:, 1], s=6, alpha=0.4, label='True Final States')
plt.scatter(final_pred[:, 0], final_pred[:, 1], s=6, alpha=0.4, label='Pred Final States')

plt.title(modelString + ' Final-State Points')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.savefig("plots/" + modelString + f'_final_state_points_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}.png')
plt.close()
