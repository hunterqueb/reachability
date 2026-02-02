import numpy as np
import matplotlib.pyplot as plt


BVP = False

# -----------------------------
# Dynamics + numerical integrator
# -----------------------------
def linear_oscillator_f(x, u, omega, zeta, mass=1.0, fuel_burn_rate=0.1):
    """
    x = [x1, x2, mf]
    x1dot = x2
    x2dot = -omega^2 x1 - 2*zeta*omega x2 + u / (mass + mf)
    mfdot = -fuel_burn_rate * |u|
    """
    x1, x2, mf = x
    mf = max(0.0, mf)
    u_eff = u if mf > 0.0 else 0.0
    total_mass = mass + mf
    dx1 = x2
    dx2 = -(omega**2) * x1 - 2.0 * zeta * omega * x2 + (u_eff / total_mass)
    dmf = -fuel_burn_rate * abs(u_eff)
    return np.array([dx1, dx2, dmf], dtype=float)


def rk4_step(x, u, dt, omega, zeta, mass=1.0, fuel_burn_rate=0.1):
    k1 = linear_oscillator_f(x, u, omega, zeta, mass=mass, fuel_burn_rate=fuel_burn_rate)
    k2 = linear_oscillator_f(x + 0.5 * dt * k1, u, omega, zeta, mass=mass, fuel_burn_rate=fuel_burn_rate)
    k3 = linear_oscillator_f(x + 0.5 * dt * k2, u, omega, zeta, mass=mass, fuel_burn_rate=fuel_burn_rate)
    k4 = linear_oscillator_f(x + dt * k3, u, omega, zeta, mass=mass, fuel_burn_rate=fuel_burn_rate)
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
    mass=1.0,
    fuel_mass0=1.0,
    fuel_radius=0.0,
    fuel_burn_rate=0.1,
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
    snapshots_full = {i: np.zeros((n_traj, 3), dtype=float) for i in snapshot_indices}

    X_final = np.zeros((n_traj, 2), dtype=float)
    X_final_full = np.zeros((n_traj, 3), dtype=float)

    for i in range(n_traj):
        # sample initial condition from a box
        x_posvel = x0_mean + rng.uniform(-1.0, 1.0, size=2) * rad
        mf0 = fuel_mass0 + rng.uniform(-1.0, 1.0) * fuel_radius
        mf0 = max(0.0, mf0)
        x = np.array([x_posvel[0], x_posvel[1], mf0], dtype=float)

        u_seq = sample_control_sequence(
            steps=steps,
            u_max=u_max,
            rng=rng,
            kind=control_kind,
            switch_prob=switch_prob
        )

        if 0 in snapshots:
            snapshots[0][i] = x[:2]
            snapshots_full[0][i] = x

        for k in range(steps):
            x = rk4_step(
                x,
                u_seq[k],
                dt,
                omega,
                zeta,
                mass=mass,
                fuel_burn_rate=fuel_burn_rate
            )
            if (k + 1) in snapshots:
                snapshots[k + 1][i] = x[:2]
                snapshots_full[k + 1][i] = x

        X_final[i] = x[:2]
        X_final_full[i] = x

    return snapshots, snapshots_full, X_final, X_final_full


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


def plot_final_hull_3d(points3d, title="Reachable Set (3D Convex Hull)"):
    try:
        from scipy.spatial import ConvexHull
    except Exception as exc:
        ConvexHull = None
        print("scipy not available; plotting 3D points only:", exc)

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        pass

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    P = np.asarray(points3d, dtype=float)
    # ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=4, alpha=0.25, label="final samples")

    if ConvexHull is not None and P.shape[0] >= 4:
        ranges = np.ptp(P, axis=0)
        if np.any(ranges < 1e-9):
            print("3D hull skipped: points are nearly coplanar.")
        else:
            try:
                hull = ConvexHull(P, qhull_options="QJ")
                for simplex in hull.simplices:
                    tri = P[simplex]
                    ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], linewidth=0.8, color="C1", alpha=0.6)
            except Exception as exc:
                print("3D hull failed; showing points only:", exc)

    ax.set_xlabel("x1 (position)")
    ax.set_ylabel("x2 (velocity)")
    ax.set_zlabel("mass (fuel)")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()


def plot_snapshot_hulls_3d(snapshots_full, dt, title="Reachable Set Over Time (3D Hulls)"):
    try:
        from scipy.spatial import ConvexHull
    except Exception as exc:
        ConvexHull = None
        print("scipy not available; plotting 3D points only:", exc)

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        pass

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    keys = sorted(snapshots_full.keys())
    for k in keys:
        edge_color = 'C'+str(keys.index(k))
        P = np.asarray(snapshots_full[k], dtype=float)
        # ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, alpha=0.06)

        if ConvexHull is None or P.shape[0] < 4:
            continue

        ranges = np.ptp(P, axis=0)
        if np.any(ranges < 1e-9):
            continue

        try:
            hull = ConvexHull(P, qhull_options="QJ")
            for simplex in hull.simplices:
                tri = P[simplex]
                ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], linewidth=0.6, alpha=0.5,color = edge_color)
        except Exception:
            continue

    ax.set_xlabel("x1 (position)")
    ax.set_ylabel("x2 (velocity)")
    ax.set_zlabel("mass (fuel)")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    

if __name__ == "__main__":
    # System
    omega = 1.5
    zeta  = 0.05
    u_max = 0.3
    mass = 1.0
    fuel_mass0 = 1.0
    fuel_burn_rate = 0.05

    # Time
    dt = 0.02
    steps = 800  # 16 seconds

    # Monte Carlo
    n_traj = 20000  # increase for tighter hull
    x0_mean = [0.2, 0.0]
    x0_box_radius = [0.02, 0.02]

    # Choose snapshot times (indices)
    snapshot_indices = (0, 100, 200, 400, 800)

    snapshots, snapshots_full, X_final, X_final_full = monte_carlo_reachable_set(
        x0_mean=x0_mean,
        x0_box_radius=x0_box_radius,
        omega=omega,
        zeta=zeta,
        u_max=u_max,
        dt=dt,
        steps=steps,
        n_traj=n_traj,
        mass=mass,
        fuel_mass0=fuel_mass0,
        fuel_radius=0.5,
        fuel_burn_rate=fuel_burn_rate,
        control_kind="bangbang",     # "uniform" or "piecewise_constant" also supported
        switch_prob=0.05,            # more switching explores more directions
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
        show_points=False,  # set True if you want clouds underneath
        snapshots=snapshots,
        title="Reachable Set Over Time (Convex Hulls per Snapshot)"
    )
    # solve nominal system for reference
    x_nom = np.array([x0_mean[0], x0_mean[1], fuel_mass0], dtype=float)
    traj_nom = np.zeros((steps + 1, 3), dtype=float)
    traj_nom[0] = x_nom
    for k in range(steps):
        x_nom = rk4_step(x_nom, 0.0, dt, omega, zeta, mass=mass, fuel_burn_rate=fuel_burn_rate)
        traj_nom[k + 1] = x_nom
    # plot nominal trajectory on top of last figure
    plt.plot(traj_nom[:, 0], traj_nom[:, 1], 'k--', linewidth=2.5, label="nominal trajectory")
    plt.legend(loc="best", frameon=True)
    # plot nominal trajectory location of snapshots
    for k in snapshot_indices:
        plt.plot(traj_nom[k, 0], traj_nom[k, 1], 'o', markersize=8,color='C'+str(snapshot_indices.index(k)))
    plot_final_hull_3d(
        X_final_full,
        title="Reachable Set (3D Convex Hull: x1, x2, mass)"
    )
    plot_snapshot_hulls_3d(
        snapshots_full,
        dt,
        title="Reachable Set Over Time (3D Hulls per Snapshot)"
    )
    # get current 3d figure
    ax = plt.gca()
    # plot nominal trajectory on top of 3d figure
    ax.plot(traj_nom[:, 0], traj_nom[:, 1], traj_nom[:, 2], 'k--', linewidth=2.5, label="nominal trajectory")

    if BVP:
        # -----------------------------
        # Two-point BVP on outer hull (fuel system, CasADi)
        # -----------------------------
        try:
            import casadi as ca
        except Exception as exc:
            ca = None
            print("CasADi not available; skipping BVP block:", exc)

        if ca is not None:
            def solve_bvp_fuel(
                x0,
                xf,
                dt,
                steps,
                omega,
                zeta,
                u_max,
                mass=mass,
                fuel0=fuel_mass0,
                burn_rate=fuel_burn_rate,
                terminal_slack_weight=5e3
            ):
                opti = ca.Opti()
                X = opti.variable(3, steps + 1)  # [x1, x2, mf]
                U = opti.variable(1, steps)
                S = opti.variable(2, 1)  # terminal slack for feasibility

                opti.subject_to(X[:, 0] == ca.vertcat(x0[0], x0[1], fuel0))
                opti.subject_to(X[0, -1] + S[0] == xf[0])
                opti.subject_to(X[1, -1] + S[1] == xf[1])

                for k in range(steps):
                    xk = X[:, k]
                    uk = U[0, k]
                    mf = ca.fmax(0, xk[2])
                    u_eff = ca.if_else(mf > 0, uk, 0.0)
                    total_mass = mass + mf
                    def f(x):
                        mf_k = ca.fmax(0, x[2])
                        u_eff_k = ca.if_else(mf_k > 0, uk, 0.0)
                        total_mass_k = mass + mf_k
                        return ca.vertcat(
                            x[1],
                            -(omega**2) * x[0] - 2.0 * zeta * omega * x[1] + (u_eff_k / total_mass_k),
                            -burn_rate * ca.fabs(u_eff_k)
                        )

                    k1 = f(xk)
                    k2 = f(xk + 0.5 * dt * k1)
                    k3 = f(xk + 0.5 * dt * k2)
                    k4 = f(xk + dt * k3)
                    x_next = xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                    opti.subject_to(X[:, k + 1] == x_next)

                opti.subject_to(opti.bounded(-u_max, U, u_max))
                opti.subject_to(X[2, :] >= 0)
                opti.subject_to(ca.fabs(S[0]) <= 0.25)
                opti.subject_to(ca.fabs(S[1]) <= 0.25)

                # Minimum control effort + small terminal error penalty for feasibility
                obj = ca.sumsqr(U) + terminal_slack_weight * ca.sumsqr(S)
                opti.minimize(obj)

                # Initial guesses improve feasibility
                opti.set_initial(X, np.tile(np.array([[x0[0]], [x0[1]], [fuel0]]), (1, steps + 1)))
                opti.set_initial(U, 0.0)
                opti.set_initial(S, 0.0)

                opti.solver("ipopt", {"print_time": False}, {"print_level": 0})
                sol = opti.solve()
                return np.array(sol.value(X)).T, np.array(sol.value(U)).reshape(-1)

            # pick a few outer hull points from the final set
            hull_final = convex_hull_2d(X_final)
            if hull_final.shape[0] >= 3:
                # take the farthest points (by radius) as boundary targets
                radii = np.linalg.norm(hull_final, axis=1)
                idx = int(np.argsort(radii)[-1])  # single farthest boundary point
                xf = hull_final[idx]

                try:
                    traj_bvp, u_bvp = solve_bvp_fuel(
                        x0=x0_mean,
                        xf=xf,
                        dt=dt,
                        steps=snapshot_indices[-1],  # longer horizon for feasibility
                        omega=omega,
                        zeta=zeta,
                        u_max=u_max,
                        mass=mass,
                        fuel0=fuel_mass0,
                        burn_rate=fuel_burn_rate,
                        terminal_slack_weight=5e3
                    )
                    plt.plot(traj_bvp[:, 0], traj_bvp[:, 1], '--', linewidth=2.0, label="optimal BVP traj")
                except Exception as exc:
                    print(f"BVP failed: {exc}")
            else:
                print("Not enough hull points for BVP targets.")
        plt.legend(loc="best", frameon=True)  
        # plot control from BVP on a separate figure
        if ca is not None and hull_final.shape[0] >= 3:
            plt.figure(figsize=(9,4))
            plt.plot(u_bvp, label="BVP optimal control")
            plt.xlabel("Time step")
            plt.ylabel("Control input u")
            plt.title("Optimal Control from BVP Solution")
            plt.grid(True)
            plt.legend(loc="best", frameon=True)
    plt.show()
