import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Dynamics + numerical integrator
# -----------------------------
def linear_oscillator_f(x, u, omega, zeta):
    """
    x = [x1, x2]
    x1dot = x2
    x2dot = -omega^2 x1 - 2*zeta*omega x2 + u
    """
    x1, x2 = x
    dx1 = x2
    dx2 = -(omega**2) * x1 - 2.0 * zeta * omega * x2 + u
    return np.array([dx1, dx2], dtype=float)


def rk4_step(x, u, dt, omega, zeta):
    k1 = linear_oscillator_f(x, u, omega, zeta)
    k2 = linear_oscillator_f(x + 0.5 * dt * k1, u, omega, zeta)
    k3 = linear_oscillator_f(x + 0.5 * dt * k2, u, omega, zeta)
    k4 = linear_oscillator_f(x + dt * k3, u, omega, zeta)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# -----------------------------
# Control sampling
# -----------------------------
def sample_control_sequence(steps, u_max, rng, kind="bangbang", switch_prob=0.05):
    """
    Returns u[k] for k=0..steps-1

    kind:
      - "bangbang": u in {+u_max, -u_max}, with random switching
      - "uniform":  u ~ U[-u_max, u_max] i.i.d.
      - "piecewise_constant": random value held, switches with prob switch_prob
    """
    if kind == "uniform":
        return rng.uniform(-u_max, u_max, size=steps)

    if kind == "piecewise_constant":
        u = np.empty(steps, dtype=float)
        cur = rng.uniform(-u_max, u_max)
        for k in range(steps):
            if rng.random() < switch_prob:
                cur = rng.uniform(-u_max, u_max)
            u[k] = cur
        return u

    # default: bang-bang
    u = np.empty(steps, dtype=float)
    cur = u_max if rng.random() < 0.5 else -u_max
    for k in range(steps):
        if rng.random() < switch_prob:
            cur = -cur
        u[k] = cur
    return u


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
    u_max,
    dt,
    steps,
    n_traj,
    control_kind="bangbang",
    switch_prob=0.05,
    snapshot_indices=(0, 200, 400, 800),
    seed=0
):
    rng = np.random.default_rng(seed)

    x0_mean = np.asarray(x0_mean, dtype=float).reshape(2)
    rad = np.asarray(x0_box_radius, dtype=float).reshape(2)

    snapshot_indices = tuple(int(i) for i in snapshot_indices if 0 <= i <= steps)
    snapshots = {i: np.zeros((n_traj, 2), dtype=float) for i in snapshot_indices}

    X_final = np.zeros((n_traj, 2), dtype=float)

    for i in range(n_traj):
        # sample initial condition from a box
        x = x0_mean + rng.uniform(-1.0, 1.0, size=2) * rad

        u_seq = sample_control_sequence(
            steps=steps,
            u_max=u_max,
            rng=rng,
            kind=control_kind,
            switch_prob=switch_prob
        )

        if 0 in snapshots:
            snapshots[0][i] = x

        for k in range(steps):
            x = rk4_step(x, u_seq[k], dt, omega, zeta)
            if (k + 1) in snapshots:
                snapshots[k + 1][i] = x

        X_final[i] = x

    return snapshots, X_final


# -----------------------------
# Plotting
# -----------------------------
def plot_snapshots_and_final_hull(snapshots, X_final, dt, title="Monte Carlo + Convex Hull"):
    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot snapshots as faint clouds
    snap_keys = sorted(snapshots.keys())
    for k in snap_keys:
        Xk = snapshots[k]
        ax.scatter(Xk[:, 0], Xk[:, 1], s=3, alpha=0.06, label=f"t={k*dt:.2f}s" if k != snap_keys[0] else None)

    # Plot final points
    ax.scatter(X_final[:, 0], X_final[:, 1], s=6, alpha=0.18, label="final samples")

    # Convex hull of final points
    hull = convex_hull_2d(X_final)
    if hull.shape[0] >= 3:
        hull_closed = np.vstack([hull, hull[0]])
        ax.plot(hull_closed[:, 0], hull_closed[:, 1], linewidth=2.5, label="final convex hull")
    elif hull.shape[0] > 0:
        ax.scatter(hull[:, 0], hull[:, 1], s=50, label="degenerate hull")

    ax.set_xlabel("x1 (position)")
    ax.set_ylabel("x2 (velocity)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    

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


def plot_snapshot_hulls(hulls, dt, show_points=False, snapshots=None, title="Snapshot Hulls"):
    """
    hulls: dict {k: (H,2)}
    If show_points=True, also scatter the underlying snapshot points (provide snapshots dict).
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    keys = sorted(hulls.keys())

    if show_points:
        if snapshots is None:
            raise ValueError("snapshots must be provided if show_points=True")
        for k in keys:
            Xk = snapshots[k]
            ax.scatter(Xk[:, 0], Xk[:, 1], s=2, alpha=0.03)

    for k in keys:
        hull = hulls[k]
        if hull.shape[0] >= 3:
            hull_closed = np.vstack([hull, hull[0]])
            ax.plot(hull_closed[:, 0], hull_closed[:, 1], linewidth=2.0, label=f"t={k*dt:.2f}s")
        elif hull.shape[0] == 2:
            ax.plot(hull[:, 0], hull[:, 1], linewidth=2.0, label=f"t={k*dt:.2f}s")
        elif hull.shape[0] == 1:
            ax.scatter(hull[0, 0], hull[0, 1], s=40, label=f"t={k*dt:.2f}s")

    ax.set_xlabel("x1 (position)")
    ax.set_ylabel("x2 (velocity)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    

if __name__ == "__main__":
    # System
    omega = 1.5
    zeta  = 0.05
    u_max = 0.3

    # Time
    dt = 0.02
    steps = 800  # 16 seconds

    # Monte Carlo
    n_traj = 20000  # increase for tighter hull
    x0_mean = [0.2, 0.0]
    x0_box_radius = [0.02, 0.02]

    # Choose snapshot times (indices)
    snapshot_indices = (0, 100, 200, 400, 800)

    snapshots, X_final = monte_carlo_reachable_set(
        x0_mean=x0_mean,
        x0_box_radius=x0_box_radius,
        omega=omega,
        zeta=zeta,
        u_max=u_max,
        dt=dt,
        steps=steps,
        n_traj=n_traj,
        control_kind="bangbang",     # "uniform" or "piecewise_constant" also supported
        switch_prob=0.03,            # more switching explores more directions
        snapshot_indices=snapshot_indices,
        seed=1
    )

    plot_snapshots_and_final_hull(
        snapshots,
        X_final,
        dt,
        title="Linear Oscillator Reachability (Monte Carlo Trajectories + Final Convex Hull)"
    )

    hulls = compute_hulls_for_snapshots(
        snapshots,
        downsample=8000,  # None for full set; use a number to speed up
        seed=123
    )

    plot_snapshot_hulls(
        hulls,
        dt,
        show_points=True,  # set True if you want clouds underneath
        snapshots=snapshots,
        title="Reachable Set Over Time (Convex Hulls per Snapshot)"
    )
    # solve nominal system for reference
    x_nom = np.array(x0_mean, dtype=float)
    traj_nom = np.zeros((steps + 1, 2), dtype=float)
    traj_nom[0] = x_nom
    for k in range(steps):
        x_nom = rk4_step(x_nom, 0.0, dt, omega, zeta)
        traj_nom[k + 1] = x_nom
    # plot nominal trajectory on top of last figure
    plt.plot(traj_nom[:, 0], traj_nom[:, 1], 'k--', linewidth=2.5, label="nominal trajectory")
    plt.legend(loc="best", frameon=True)
    # plot nominal trajectory location of snapshots
    for k in snapshot_indices:
        plt.plot(traj_nom[k, 0], traj_nom[k, 1], 'o', markersize=8,color='C'+str(snapshot_indices.index(k)))
    plt.show()
