import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import torch.nn.functional as F
import torch.utils.data as data
import argparse
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError # import here for p36 compatibility


from qutils.ml.utils import printModelParmSize, getDevice, Adam_mini
from qutils.tictoc import timer
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.regression import LSTM
from qutils.ml.utils import findDecAcc
from qutils.orbital import dim2NonDim6, nonDim2Dim6

#import for superweight identification
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight, findMambaSuperActivation,plotSuperActivation

# args parsing for model, horizon, traj_index
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mamba', help='Model to use')
parser.add_argument('--horizon', type=int, default=3, help='Horizon for prediction')
parser.add_argument('--traj-index', type=int, default=0, help='Trajectory index to plot')
parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of trajectories to use for training (rest used for testing)')
parser.add_argument('--batch', type=int, default=256, help='Batch size for training')
parser.add_argument('--batch-test', type=int, default=128, help='Batch size for evaluation')
parser.add_argument('--n-epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training')
parser.add_argument('--ood', action='store_true', help='Whether to evaluate on OOD data with larger deltaV')
parser.add_argument('--jetson', action='store_true', help='use flag to run on jetson with smaller test size')
parser.add_argument('--dim',action="store_true",help="train WITHOUT non dimensional coordinates")
parser.add_argument('--dv',type=float,default=5.0,help="amount of delta v used for picking dataset")
parser.add_argument('--n',type=int,default=10000,help='amount of trajectories used for picking dataset')
parser.add_argument('--pdf', action='store_true', help='Whether to save plots in PDF format instead of PNG')

args = parser.parse_args()
modelString = args.model
traj_index = args.traj_index


if args.pdf:
    saveType = 'pdf'
else:
    saveType = 'png'

problemDim = 6

device = getDevice()


# hyperparameters
n_epochs = args.n_epochs
lr = args.lr
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = 1
horizon = args.horizon


# import gmat dataset
dataset_loc = f"./data/gmat/{args.dv}km-{args.n}"
dataset_file = "/statesArrayImpBurn.npy"

dataset = np.load(dataset_loc+dataset_file)["statesArrayImpBurn"] # (n_traj,min_prop,problemDim)
num_trajs = dataset.shape[0]
num_time_steps = dataset.shape[1]

# convert to nondim for better ML -- turn off with args
if not args.dim:
    for i in range(num_trajs):
        dataset[i,:,:]=dim2NonDim6(dataset[i,:,:])

trajs_t = np.transpose(dataset, (1, 0, 2))  # (num_time_steps, num_trajectories, problemDim)
numericResult = trajs_t
train_size = 5
test_size = numericResult.shape[1] - train_size

def create_datasets(data, seq_length, train_size, device):
    # Split across dimension 0 (time): use edge windows for train (size ~2*horizon)
    # and middle window for test, while keeping 80-20 split across trajectories.
    split_idx = int(data.shape[1] * args.train_ratio)
    time_end = min(num_time_steps, data.shape[0])
    train_time = np.concatenate([data[:horizon], data[time_end - horizon:time_end]], axis=0)
    test_time = data[horizon:time_end - horizon]

    train_data = train_time[:, :split_idx, :]
    if args.jetson: 
        # for jetson testing, use smaller test set to reduce memory requirements for test loss evaluation
        test_data = test_time[:, split_idx:split_idx+1000, :]
    else:
        test_data = test_time[:, split_idx:, :]

    def build_xy(d):
        xs, ys = [], []
        for i in range(len(d) - seq_length):
            x = d[i:(i + seq_length)]  # (seq_length, num_trajectories, problemDim)
            y = d[i + seq_length]      # (num_trajectories, problemDim)
            xs.append(x)
            ys.append(y)
        X = np.stack(xs, axis=0)  # (num_windows, seq_length, num_trajectories, problemDim)
        Y = np.stack(ys, axis=0)  # (num_windows, num_trajectories, problemDim)
        return X, Y

    X_train, Y_train = build_xy(train_data)
    X_test, Y_test = build_xy(test_data)
    # Convert to PyTorch tensors (keep on CPU; move batches to GPU in the loop)
    X_train = torch.tensor(np.array(X_train)).float().squeeze()
    Y_train = torch.tensor(np.array(Y_train)).float()
    X_test = torch.tensor(np.array(X_test)).float().squeeze()
    Y_test = torch.tensor(np.array(Y_test)).float()


    return X_train,Y_train,X_test,Y_test
train_in,train_out,test_in,test_out = create_datasets(numericResult,1,train_size,device)
print(train_in.shape)
print(train_out.shape)
print(test_in.shape)
print(test_out.shape)

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=args.batch)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=16)

def returnModel(modelString = 'mamba'):
    if modelString == 'mamba':
        model = Mamba(config).to(device).float()
    elif modelString == 'lstm':
        model = LSTM(input_size,30,output_size,num_layers,1).to(device).float()
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

def alpha_shape_faces_and_volume(points, edge_quantile=0.95):
    n = points.shape[0]
    if n < 5:
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    try:
        tet = Delaunay(points)
    except QhullError:
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    simplices = tet.simplices  # (m, 4)
    p = points[simplices]      # (m, 4, 3)

    e01 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    e02 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    e03 = np.linalg.norm(p[:, 3] - p[:, 0], axis=1)
    e12 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    e13 = np.linalg.norm(p[:, 3] - p[:, 1], axis=1)
    e23 = np.linalg.norm(p[:, 3] - p[:, 2], axis=1)
    max_edge = np.maximum.reduce([e01, e02, e03, e12, e13, e23])

    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    tet_vol = np.abs(np.einsum("ij,ij->i", cross, p[:, 3] - p[:, 0])) / 6.0
    valid = tet_vol > 1e-14
    if not np.any(valid):
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    thresh = np.quantile(max_edge[valid], edge_quantile)
    keep = valid & (max_edge <= thresh)
    if not np.any(keep):
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    kept = simplices[keep]
    faces = np.concatenate(
        [
            kept[:, [0, 1, 2]],
            kept[:, [0, 1, 3]],
            kept[:, [0, 2, 3]],
            kept[:, [1, 2, 3]],
        ],
        axis=0,
    )
    faces_sorted = np.sort(faces, axis=1)
    uniq_faces, counts = np.unique(faces_sorted, axis=0, return_counts=True)
    boundary_faces = uniq_faces[counts == 1]
    return points[boundary_faces], tet_vol[keep].sum()

model = returnModel(modelString)
modelString = modelString + '_ood' if args.ood else modelString

optimizer = Adam_mini(model,lr=lr)

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


if args.ood:
    # Load OOD test data with larger deltaV
    ood_dataFile = "./data/gmat/5.0km-{}".format(args.n)
    dataset_file = "/statesArrayImpBurn.npy"

    ood_system_data = np.load(ood_dataFile + dataset_file, allow_pickle=True)
    ood_trajs = ood_system_data['statesArrayImpBurn']
    
    # convert to nondim for better ML -- turn off with args
    if not args.dim:
        for i in range(num_trajs):
            ood_trajs[i,:,:]=dim2NonDim6(ood_trajs[i,:,:])


    ood_trajs_t = np.transpose(ood_trajs, (1, 0, 2))
    ood_numericResult = ood_trajs_t

    # Create OOD test dataset
    _, _, test_in, test_out = create_datasets(ood_numericResult, 1, train_size, device)



# plot some predictions
model.eval()
with torch.no_grad():
    test_loader = data.DataLoader(data.TensorDataset(test_in, test_out), shuffle=False, batch_size=args.batch_test)

    xb, yb = next(iter(test_loader))
    xb = xb.to(device)
    pred = model(xb)
    pred = pred.cpu().numpy()
    yb = yb.cpu().numpy()
    xb = xb.cpu().numpy()
    traj_idx = traj_index
    def predict_last_step(x_all, batch_size=args.batch_test, slice_traj_idx=None):
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

    # convert to dim for plotting
    if not args.dim:
        true_test_seq=nonDim2Dim6(true_test_seq)
        pred_test_seq=nonDim2Dim6(pred_test_seq)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # color train and test segments differently
    # ax.plot(train_pred[:,0], train_pred[:,1], train_pred[:,2], 'g.', label='Train Predictions')
    # ax.plot(test_pred[:,0], test_pred[:,1], test_pred[:,2], 'r.', label='Test Predictions')
    if true_test_seq.shape[1] >= 3 and pred_test_seq.shape[1] >= 3:
        true_z = true_test_seq[:, 2]
        pred_z = pred_test_seq[:, 2]
    else:
        true_z = np.zeros(true_test_seq.shape[0])
        pred_z = np.zeros(pred_test_seq.shape[0])

    ax.plot(true_test_seq[:, 0], true_test_seq[:, 1], true_z, 'k-', label='True Trajectory')
    ax.plot(pred_test_seq[:, 0], pred_test_seq[:, 1], pred_z, '--', label='Predicted Trajectory')
    ax.set_title(modelString+' Reachability Prediction: Trajectory Index '+str(traj_idx))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(loc='best')
    # save plot
    plt.savefig("plots/"+modelString+f'_reachability_ratio_{args.train_ratio}_prediction_epoch_{n_epochs}_index_{traj_idx}_lr_{lr}_train_window_{args.horizon*2}.{saveType}')
    plt.close()

    final_true = test_out[-1].numpy()
    final_pred = test_pred_full[-1].numpy()

    if not args.dim:
        final_true=nonDim2Dim6(final_true)
        final_pred=nonDim2Dim6(final_pred)


    if final_true.shape[1] == 6:
        pos_true = final_true[:, :3]
        pos_pred = final_pred[:, :3]
        vel_true = final_true[:, 3:]
        vel_pred = final_pred[:, 3:]

        pos_faces_true, pos_vol_true = alpha_shape_faces_and_volume(pos_true, edge_quantile=0.95)
        pos_faces_pred, pos_vol_pred = alpha_shape_faces_and_volume(pos_pred, edge_quantile=0.95)
        vel_faces_true, vel_vol_true = alpha_shape_faces_and_volume(vel_true, edge_quantile=0.95)
        vel_faces_pred, vel_vol_pred = alpha_shape_faces_and_volume(vel_pred, edge_quantile=0.95)

        pos_ratio = float(pos_vol_pred) / float(pos_vol_true) if pos_vol_true > 0 else float('inf')
        vel_ratio = float(vel_vol_pred) / float(vel_vol_true) if vel_vol_true > 0 else float('inf')
        print(f"Pos Alpha-Shape Volume True: {pos_vol_true:.4f}, Pred: {pos_vol_pred:.4f}, Ratio: {pos_ratio:.4f}")
        print(f"Vel Alpha-Shape Volume True: {vel_vol_true:.4f}, Pred: {vel_vol_pred:.4f}, Ratio: {vel_ratio:.4f}")

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        for ax, t_pts, p_pts, t_faces, p_faces, title, labels in [
            (ax1, pos_true, pos_pred, pos_faces_true, pos_faces_pred, f'Position Alpha Shapes (Ratio {pos_ratio:.4f})', ('x', 'y', 'z')),
            (ax2, vel_true, vel_pred, vel_faces_true, vel_faces_pred, f'Velocity Alpha Shapes (Ratio {vel_ratio:.4f})', ('vx', 'vy', 'vz')),
        ]:
            ax.scatter(t_pts[:, 0], t_pts[:, 1], t_pts[:, 2], s=3, alpha=0.2, c='k')
            ax.scatter(p_pts[:, 0], p_pts[:, 1], p_pts[:, 2], s=3, alpha=0.2, c='r')
            ax.add_collection3d(Poly3DCollection(t_faces, facecolor='k', alpha=0.08, edgecolor='none'))
            ax.add_collection3d(Poly3DCollection(p_faces, facecolor='r', alpha=0.08, edgecolor='none'))
            all_pts = np.vstack((t_pts, p_pts))
            mins = all_pts.min(axis=0)
            maxs = all_pts.max(axis=0)
            spans = np.maximum(maxs - mins, 1e-6)
            ax.set_box_aspect(spans)
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])
            ax.set_title(title)

        fig.suptitle(modelString + ' Final-State 3D Alpha Shapes')
        plt.tight_layout()
        plt.savefig("plots/" + modelString + f'_final_state_alpha_shapes_3d_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}_train_window_{args.horizon*2}.{saveType}')
        plt.close()
    else:
        true_segments, area_true = alpha_shape_segments_and_area(final_true,radius_quantile=0.95)
        pred_segments, area_pred = alpha_shape_segments_and_area(final_pred,radius_quantile=0.95)

        # calculate reachable-set areas and find ratio of pred area to true area
        area_true = float(area_true)
        area_pred = float(area_pred)

        area_ratio = (area_pred) / area_true if area_true > 0 else float('inf')

        print(f"True Alpha-Shape Area: {area_true:.4f}, Pred Alpha-Shape Area: {area_pred:.4f}, Area Ratio (Pred/True): {area_ratio:.4f}")

        true_z = final_true[:, 2] if final_true.shape[1] >= 3 else np.zeros(final_true.shape[0])
        pred_z = final_pred[:, 2] if final_pred.shape[1] >= 3 else np.zeros(final_pred.shape[0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(final_true[:, 0], final_true[:, 1], true_z, s=6, alpha=0.4, label='True Final States')
        ax.scatter(final_pred[:, 0], final_pred[:, 1], pred_z, s=6, alpha=0.4, label='Pred Final States')
        for i, seg in enumerate(true_segments):
            ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]], [0.0, 0.0], c='k', lw=2,
                    label='True Alpha Shape' if i == 0 else None)
        for i, seg in enumerate(pred_segments):
            ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]], [0.0, 0.0], c='r', lw=2, ls='--',
                    label='Pred Alpha Shape' if i == 0 else None)

        ax.set_title(modelString + ' Final-State Alpha Shapes: Area Ratio {:.4f}'.format(area_ratio))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend(loc='best')
        plt.savefig("plots/" + modelString + f'_final_state_alpha_shapes_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}_train_window_{args.horizon*2}.{saveType}')
        plt.close()

    true_z = final_true[:, 2] if final_true.shape[1] >= 3 else np.zeros(final_true.shape[0])
    pred_z = final_pred[:, 2] if final_pred.shape[1] >= 3 else np.zeros(final_pred.shape[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(final_true[:, 0], final_true[:, 1], true_z, s=6, alpha=0.4, label='True Final States')
    ax.scatter(final_pred[:, 0], final_pred[:, 1], pred_z, s=6, alpha=0.4, label='Pred Final States')

    ax.set_title(modelString + ' Final-State Points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(loc='best')
    plt.savefig("plots/" + modelString + f'_final_state_points_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}_train_window_{args.horizon*2}.{saveType}')
    plt.close()
    # Plot one random test trajectory: predicted vs true across time
    rng = np.random.default_rng(12)
    rand_traj_idx = rng.integers(0, test_in.shape[1])
    traj_true = build_full_seq(test_in, test_out, rand_traj_idx)
    traj_pred = build_full_seq(test_in, test_pred_full, rand_traj_idx)
    time_axis = np.arange(traj_true.shape[0]) * 60.0  # Assuming 60s time step

    traj_true=nonDim2Dim6(traj_true)
    traj_pred=nonDim2Dim6(traj_pred)


    fig = plt.figure(figsize=(8, 6))
    labels = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    for i in range(problemDim):
        ax = fig.add_subplot(3, 2, i + 1)
        ax.plot(time_axis, traj_true[:, i], 'k-', lw=1.5, label='True')
        ax.plot(time_axis, traj_pred[:, i], 'r--', lw=1.5, label='Predicted')
        ax.set_xlabel('time [s]')
        ax.set_ylabel(labels[i])
        if i == 0:
            ax.legend(loc='best')
    fig.suptitle(f'Random Test Trajectory #{rand_traj_idx} (Pred vs True)')
    plt.tight_layout()
    plt.savefig("plots/" + modelString + f'_random_test_trajectory_{rand_traj_idx}_epoch_{n_epochs}_lr_{lr}_train_window_{args.horizon*2}.{saveType}')
    plt.close()

    if modelString == 'mamba':
        xb, yb = next(iter(test_loader))
        xb_one_traj = xb[:, traj_index:traj_index+1, :]
        magnitude, index = findMambaSuperActivation(model, xb_one_traj.to(device))

        normedMagsMRP = np.zeros((len(magnitude),))
        for i in range(len(magnitude)):
            normedMagsMRP[i] = magnitude[i].norm().detach().cpu()

        printoutMaxLayerWeight(model)
        getSuperWeight(model)
        plotSuperWeight(model)
        plotSuperActivation(magnitude, index,printOutValues=True)
        plt.title("Mamba Reachability Super Activations")
        plt.savefig("plots/" + modelString + f'_super_activations_ratio_{args.train_ratio}_epoch_{n_epochs}_index_{traj_index}_lr_{lr}_train_window_{args.horizon*2}.{saveType}')
