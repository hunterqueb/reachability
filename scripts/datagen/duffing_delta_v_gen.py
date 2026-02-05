import numpy as np
try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


# -----------------------------
# Dynamics + numerical integrator
# -----------------------------
def duffing_oscillator_f(x, t,u, omega, zeta, alpha,beta,gamma, total_mass=2.0):
    """
    x = [x1, x2]
    x1dot = x2
    x2dot = -omega^2 x1 - beta*x1^3 - 2*zeta*omega x2 + u
    """
    x1, x2 = x
    dx1 = x2
    dx2 = -alpha * x1 - beta * (x1**3) - zeta * x2 + u / total_mass + gamma * np.cos(omega * t)
    
    return np.array([dx1, dx2], dtype=float)


def rk4_step(x, t, u, dt, omega, zeta, alpha, beta, gamma, total_mass=2.0):
    k1 = duffing_oscillator_f(x, t, u, omega, zeta, alpha, beta, gamma, total_mass=total_mass)
    k2 = duffing_oscillator_f(x + 0.5 * dt * k1, t + 0.5 * dt, u, omega, zeta, alpha, beta, gamma, total_mass=total_mass)
    k3 = duffing_oscillator_f(x + 0.5 * dt * k2, t + 0.5 * dt, u, omega, zeta, alpha, beta, gamma, total_mass=total_mass)
    k4 = duffing_oscillator_f(x + dt * k3,t + dt,u , omega,zeta,alpha,beta,gamma,total_mass=total_mass)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# -----------------------------
# Numba-accelerated kernels
# -----------------------------
if _HAVE_NUMBA:
    @njit
    def _duffing_oscillator_f_nb(x, t, u, omega, zeta, alpha, beta, gamma, total_mass):
        x1 = x[0]
        x2 = x[1]
        dx1 = x2
        dx2 = -alpha * x1 - beta * (x1**3) - zeta * x2 + u / total_mass + gamma * np.cos(omega * t)
        out = np.empty(2, dtype=np.float64)
        out[0] = dx1
        out[1] = dx2
        return out


    @njit
    def _rk4_step_nb(x, t, u, dt, omega, zeta, alpha, beta, gamma, total_mass):
        k1 = _duffing_oscillator_f_nb(x, t, u, omega, zeta, alpha, beta, gamma, total_mass)
        k2 = _duffing_oscillator_f_nb(x + 0.5 * dt * k1, t + 0.5 * dt, u, omega, zeta, alpha, beta, gamma, total_mass)
        k3 = _duffing_oscillator_f_nb(x + 0.5 * dt * k2, t + 0.5 * dt, u, omega, zeta, alpha, beta, gamma, total_mass)
        k4 = _duffing_oscillator_f_nb(x + dt * k3,t + dt,u , omega,zeta,alpha,beta,gamma,total_mass=total_mass)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


    @njit(parallel=True)
    def _monte_carlo_reachable_set_numba(
        x0_posvel,
        delta_v,
        steps,
        dt,
        omega,
        zeta,
        alpha,
        beta,
        gamma,
        total_mass,
        snapshot_indices,
        snapshots,
        X_final
    ):
        n_traj = x0_posvel.shape[0]
        n_snaps = snapshot_indices.shape[0]

        for i in prange(n_traj):
            x = np.empty(2, dtype=np.float64)
            x[0] = x0_posvel[i, 0]
            x[1] = x0_posvel[i, 1]

            for s in range(n_snaps):
                if snapshot_indices[s] == 0:
                    snapshots[s, i, 0] = x[0]
                    snapshots[s, i, 1] = x[1]

            for k in range(steps):
                if k == 2:
                    x[1] = x[1] + delta_v[i]
                x = _rk4_step_nb(x, k * dt, 0.0, dt, omega, zeta, alpha, beta, gamma, total_mass)
                k1 = k + 1
                for s in range(n_snaps):
                    if snapshot_indices[s] == k1:
                        snapshots[s, i, 0] = x[0]
                        snapshots[s, i, 1] = x[1]

            X_final[i, 0] = x[0]
            X_final[i, 1] = x[1]


# -----------------------------
# Convex hull (2D) - monotone chain
# -----------------------------
def _cross(o, a, b):
    # 2D cross product (OA x OB)
    return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])


def convex_hull_2d(points):
    """
    points: (N,2)
    Returns hull vertices in CCW order as (H,2). If N<3, returns unique points.
    Monotone chain. O(N log N).
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return pts

    # sort lexicographically, unique
    pts = np.unique(pts, axis=0)
    if pts.shape[0] <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]  # sort by x then y

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    # concatenate, removing duplicate endpoints
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)


# -----------------------------
# Monte Carlo reachability
# -----------------------------
def monte_carlo_reachable_set(
    x0_mean,
    x0_box_radius,
    omega,
    zeta,
    alpha,
    beta,
    gamma,
    total_mass,
    delta_v_radius,
    dt,
    steps,
    n_traj,
    snapshot_indices=(0, 200, 400, 800),
    seed=0
):
    rng = np.random.default_rng(seed)

    x0_mean = np.asarray(x0_mean, dtype=float).reshape(2)
    rad = np.asarray(x0_box_radius, dtype=float).reshape(2)

    snapshot_indices = tuple(int(i) for i in snapshot_indices if 0 <= i <= steps)
    snapshot_idx_arr = np.asarray(snapshot_indices, dtype=np.int64)

    # Precompute randomness in NumPy for Numba compatibility.
    x0_posvel = x0_mean + rng.uniform(-1.0, 1.0, size=(n_traj, 2)) * rad

    delta_v = rng.uniform(-delta_v_radius, delta_v_radius, size=n_traj)

    snapshots_arr = np.zeros((len(snapshot_indices), n_traj, 2), dtype=float)
    X_final = np.zeros((n_traj, 2), dtype=float)

    if _HAVE_NUMBA:
        _monte_carlo_reachable_set_numba(
            x0_posvel,
            delta_v,
            steps,
            dt,
            omega,
            zeta,
            alpha,
            beta,
            gamma,
            total_mass,
            snapshot_idx_arr,
            snapshots_arr,
            X_final
        )
    else:
        for i in range(n_traj):
            # sample initial condition from a box
            x = x0_posvel[i].copy()

            for s, k in enumerate(snapshot_indices):
                if k == 0:
                    snapshots_arr[s, i] = x

            for k in range(steps):
                if k == 2:
                    x[1] = x[1] + delta_v[i]
                x = rk4_step(x, k * dt, 0.0, dt, omega, zeta, alpha=alpha, beta=beta, gamma=gamma, total_mass=total_mass)
                k1 = k + 1
                for s, snap_k in enumerate(snapshot_indices):
                    if snap_k == k1:
                        snapshots_arr[s, i] = x

            X_final[i] = x

    snapshots = {int(k): snapshots_arr[idx] for idx, k in enumerate(snapshot_indices)}
    return snapshots, X_final



def compute_hulls_for_snapshots(snapshots, downsample=None, seed=0):
    """
    snapshots: dict {k: (N,2) array}
    downsample: if not None, randomly pick this many points per snapshot for hull
    Returns dict {k: hull_vertices (H,2)}
    """
    rng = np.random.default_rng(seed)
    hulls = {}

    for k, X in snapshots.items():
        X = np.asarray(X, dtype=float)
        if X.shape[0] == 0:
            hulls[k] = X
            continue

        if downsample is not None and X.shape[0] > downsample:
            idx = rng.choice(X.shape[0], size=downsample, replace=False)
            X_use = X[idx]
        else:
            X_use = X

        hulls[k] = convex_hull_2d(X_use)

    return hulls    

if __name__ == "__main__":
    # System
    k = 0.5
    m = 2.0
    omega = np.sqrt(k / m)
    zeta = 0.05
    alpha = 0.0
    beta = 1.0
    gamma = 0.2
    total_mass = 2.0
    delta_v_radius = 0.3

    # args parse for delta v_radius
    import argparse
    parser = argparse.ArgumentParser(description="Duffing Oscillator Monte Carlo Reachability")
    parser.add_argument("--deltaV", type=float, default=0.3, help="Radius of delta-v perturbation at control time")
    parser.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    parser.add_argument("--steps", type=int, default=100, help="Number of snapshots to compute")
    parser.add_argument("--T", type=float, default=16.0, help="Total simulation time in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Time step for integration")
    parser.add_argument("--n", type=int, default=20000, help="Number of Monte Carlo trajectories to simulate")
    if parser.parse_args().no_numba:
        _HAVE_NUMBA = False
    args = parser.parse_args()
    delta_v_radius = args.deltaV

    save_dir = "./data/test/"
    save_file = f"duffing_monte_carlo_trajectories_dv_{delta_v_radius}_dt_{args.steps}_n_{args.n}.npz"

    # Time
    dt = args.dt
    steps = int(args.T / dt)

    # Monte Carlo
    n_traj = args.n  # increase for tighter hull
    x0_mean = [0.2, 0.0]
    x0_box_radius = [0.02, 0.02]

    # Choose snapshot times (indices)
    snapshot_indices = (0, 100, 200, 400, 800)
    
    zoneNums = args.steps
    snapshot_indices = [0]

    for i in range(1, zoneNums + 1):
        snapshot_indices.append(int(i * steps / zoneNums))

    snapshots, X_final = monte_carlo_reachable_set(
        x0_mean=x0_mean,
        x0_box_radius=x0_box_radius,
        omega=omega,
        zeta=zeta,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        total_mass=total_mass,
        delta_v_radius=delta_v_radius,
        dt=dt,
        steps=steps,
        n_traj=n_traj,
        snapshot_indices=snapshot_indices,
        seed=1
    )
    
    # using snapshots, construct n_traj trajectories
    n_traj = X_final.shape[0]
    traj_list = []
    for i in prange(n_traj):
        traj = np.zeros((steps + 1, 2), dtype=float)
        traj[0] = snapshots[0][i]
        for k in range(steps):
            if k in snapshots:
                traj[k + 1] = snapshots[k][i]
            else:
                traj[k + 1] = traj[k]
        traj_list.append(traj)
    traj_list = np.array(traj_list) # n,steps,2


    # save trajectories to disk
    np.savez(save_dir + save_file, trajectories=traj_list, dt=dt)