import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from scipy.linalg import expm
except ImportError:
    expm = None


class Zonotope:
    """
    Zonotope Z = { c + G*xi : xi in [-1,1]^p }.
    c: (n,)
    G: (n,p)
    """
    def __init__(self, c: np.ndarray, G: np.ndarray):
        self.c = np.asarray(c, dtype=float).reshape(-1)
        self.G = np.asarray(G, dtype=float)
        assert self.G.ndim == 2
        assert self.G.shape[0] == self.c.shape[0]

    @property
    def n(self):
        return self.c.shape[0]

    @property
    def p(self):
        return self.G.shape[1]

    def interval_bounds(self):
        r = np.sum(np.abs(self.G), axis=1)
        return self.c - r, self.c + r

    def affine_map(self, M: np.ndarray, b: np.ndarray = None):
        M = np.asarray(M, dtype=float)
        assert M.shape[1] == self.n
        c2 = M @ self.c
        G2 = M @ self.G
        if b is not None:
            b = np.asarray(b, dtype=float).reshape(-1)
            assert b.shape[0] == c2.shape[0]
            c2 = c2 + b
        return Zonotope(c2, G2)

    def minkowski_sum(self, other: "Zonotope"):
        assert self.n == other.n
        c2 = self.c + other.c
        if self.p == 0 and other.p == 0:
            G2 = np.zeros((self.n, 0))
        elif self.p == 0:
            G2 = other.G.copy()
        elif other.p == 0:
            G2 = self.G.copy()
        else:
            G2 = np.concatenate([self.G, other.G], axis=1)
        return Zonotope(c2, G2)

    def add_generator(self, g: np.ndarray):
        g = np.asarray(g, dtype=float).reshape(-1, 1)
        assert g.shape[0] == self.n
        G2 = np.concatenate([self.G, g], axis=1)
        return Zonotope(self.c.copy(), G2)

    def reduce(self, max_gens: int):
        """
        Conservative generator reduction:
        Keep largest generators, lump the rest into a box (diagonal generators).
        """
        if self.p <= max_gens:
            return self

        norms = np.linalg.norm(self.G, axis=0)
        idx = np.argsort(-norms)
        keep = idx[:max_gens]
        drop = idx[max_gens:]

        G_keep = self.G[:, keep]
        rad = np.sum(np.abs(self.G[:, drop]), axis=1)
        G_box = np.diag(rad)

        G2 = np.concatenate([G_keep, G_box], axis=1)
        return Zonotope(self.c.copy(), G2)


def discretize_lti_exact(A, B, dt):
    """
    Exact discretization using augmented matrix exponential:
    [Ad Bd] = expm([A B; 0 0] dt)
    """
    if expm is None:
        raise RuntimeError("scipy is required for exact discretization (scipy.linalg.expm not found).")

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    m = B.shape[1]

    M = np.zeros((n + m, n + m), dtype=float)
    M[:n, :n] = A
    M[:n, n:] = B

    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from scipy.linalg import expm
except ImportError:
    expm = None


class Zonotope:
    """
    Zonotope Z = { c + G*xi : xi in [-1,1]^p }.
    c: (n,)
    G: (n,p)
    """
    def __init__(self, c: np.ndarray, G: np.ndarray):
        self.c = np.asarray(c, dtype=float).reshape(-1)
        self.G = np.asarray(G, dtype=float)
        assert self.G.ndim == 2
        assert self.G.shape[0] == self.c.shape[0]

    @property
    def n(self):
        return self.c.shape[0]

    @property
    def p(self):
        return self.G.shape[1]

    def interval_bounds(self):
        r = np.sum(np.abs(self.G), axis=1)
        return self.c - r, self.c + r

    def affine_map(self, M: np.ndarray, b: np.ndarray = None):
        M = np.asarray(M, dtype=float)
        assert M.shape[1] == self.n
        c2 = M @ self.c
        G2 = M @ self.G
        if b is not None:
            b = np.asarray(b, dtype=float).reshape(-1)
            assert b.shape[0] == c2.shape[0]
            c2 = c2 + b
        return Zonotope(c2, G2)

    def minkowski_sum(self, other: "Zonotope"):
        assert self.n == other.n
        c2 = self.c + other.c
        if self.p == 0 and other.p == 0:
            G2 = np.zeros((self.n, 0))
        elif self.p == 0:
            G2 = other.G.copy()
        elif other.p == 0:
            G2 = self.G.copy()
        else:
            G2 = np.concatenate([self.G, other.G], axis=1)
        return Zonotope(c2, G2)

    def add_generator(self, g: np.ndarray):
        g = np.asarray(g, dtype=float).reshape(-1, 1)
        assert g.shape[0] == self.n
        G2 = np.concatenate([self.G, g], axis=1)
        return Zonotope(self.c.copy(), G2)

    def reduce(self, max_gens: int):
        """
        Conservative generator reduction:
        Keep largest generators, lump the rest into a box (diagonal generators).
        """
        if self.p <= max_gens:
            return self

        norms = np.linalg.norm(self.G, axis=0)
        idx = np.argsort(-norms)
        keep = idx[:max_gens]
        drop = idx[max_gens:]

        G_keep = self.G[:, keep]
        rad = np.sum(np.abs(self.G[:, drop]), axis=1)
        G_box = np.diag(rad)

        G2 = np.concatenate([G_keep, G_box], axis=1)
        return Zonotope(self.c.copy(), G2)


def discretize_lti_exact(A, B, dt):
    """
    Exact discretization using augmented matrix exponential:
    [Ad Bd] = expm([A B; 0 0] dt)
    """
    if expm is None:
        raise RuntimeError("scipy is required for exact discretization (scipy.linalg.expm not found).")

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    m = B.shape[1]

    M = np.zeros((n + m, n + m), dtype=float)
    M[:n, :n] = A
    M[:n, n:] = B

    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


def discretize_lti_euler(A, B, dt):
    """
    Fallback discretization if SciPy isn't available.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    Ad = np.eye(n) + dt * A
    Bd = dt * B
    return Ad, Bd


def linear_oscillator_reach_zonotope(
    Z0: Zonotope,
    dt: float,
    steps: int,
    omega: float,
    zeta: float,
    u_max: float,
    max_gens: int = 30,
    exact_discretization: bool = True
):
    """
    Reachable tube for a damped linear oscillator with bounded input using zonotopes.

    xdot = A x + B u,  u in [-u_max, u_max]
    Z_{k+1} = Ad Z_k ⊕ (Bd * [-u_max, u_max])
    """
    A = np.array([[0.0, 1.0],
                  [-(omega ** 2), -2.0 * zeta * omega]], dtype=float)
    B = np.array([[0.0],
                  [1.0]], dtype=float)

    if exact_discretization:
        Ad, Bd = discretize_lti_exact(A, B, dt)
    else:
        Ad, Bd = discretize_lti_euler(A, B, dt)

    g_u = (Bd[:, 0] * u_max).reshape(2, 1)  # single generator from input bound
    Z_u = Zonotope(np.zeros(2), g_u)

    Zs = [Z0]
    Z = Z0
    for _ in range(steps):
        Z_next = Z.affine_map(Ad).minkowski_sum(Z_u).reduce(max_gens=max_gens)
        Zs.append(Z_next)
        Z = Z_next

    return Zs


def sample_zonotope(Z, n_samples=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    if Z.p == 0:
        return Z.c.reshape(1, -1)

    Xi = rng.uniform(-1.0, 1.0, size=(Z.p, n_samples))  # (p, N)
    X = Z.c.reshape(-1, 1) + Z.G @ Xi                   # (n, N)
    return X.T                                          # (N, n)


def plot_reach_tube_boxes_and_samples(
    Zs,
    dt,
    box_stride=50,
    sample_stride=200,
    samples_per_slice=3000,
    rng_seed=0
):
    rng = np.random.default_rng(rng_seed)
    fig, ax = plt.subplots(figsize=(8, 6))

    for k in range(0, len(Zs), sample_stride):
        X = sample_zonotope(Zs[k], n_samples=samples_per_slice, rng=rng)
        ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.08)

    for k in range(0, len(Zs), box_stride):
        lo, hi = Zs[k].interval_bounds()
        rect = Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                         fill=False, linewidth=1.0, alpha=0.6)
        ax.add_patch(rect)

    lo, hi = Zs[-1].interval_bounds()
    rect = Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                     fill=False, linewidth=2.5)
    ax.add_patch(rect)

    ax.set_xlabel("x1 (position)")
    ax.set_ylabel("x2 (velocity)")
    ax.set_title("Linear Oscillator Reachable Tube (Zonotope Boxes + Samples)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Initial set: small box around (x1, x2)
    c0 = np.array([0.2, 0.0])
    rad0 = np.array([0.02, 0.02])
    Z0 = Zonotope(c0, np.diag(rad0))

    # Linear oscillator parameters
    omega = 1.5     # rad/s
    zeta  = 0.05    # damping ratio
    u_max = 0.3

    dt = 0.02
    steps = 800  # 16 seconds

    # If you don't have SciPy, set exact_discretization=False.
    Zs = linear_oscillator_reach_zonotope(
        Z0=Z0,
        dt=dt,
        steps=steps,
        omega=omega,
        zeta=zeta,
        u_max=u_max,
        max_gens=40,
        exact_discretization=True
    )

    plot_reach_tube_boxes_and_samples(
        Zs,
        dt,
        box_stride=40,
        sample_stride=160,
        samples_per_slice=2500,
        rng_seed=1
    )


def discretize_lti_euler(A, B, dt):
    """
    Fallback discretization if SciPy isn't available.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    Ad = np.eye(n) + dt * A
    Bd = dt * B
    return Ad, Bd


def linear_oscillator_reach_zonotope(
    Z0: Zonotope,
    dt: float,
    steps: int,
    omega: float,
    zeta: float,
    u_max: float,
    max_gens: int = 30,
    exact_discretization: bool = True
):
    """
    Reachable tube for a damped linear oscillator with bounded input using zonotopes.

    xdot = A x + B u,  u in [-u_max, u_max]
    Z_{k+1} = Ad Z_k ⊕ (Bd * [-u_max, u_max])
    """
    A = np.array([[0.0, 1.0],
                  [-(omega ** 2), -2.0 * zeta * omega]], dtype=float)
    B = np.array([[0.0],
                  [1.0]], dtype=float)

    if exact_discretization:
        Ad, Bd = discretize_lti_exact(A, B, dt)
    else:
        Ad, Bd = discretize_lti_euler(A, B, dt)

    g_u = (Bd[:, 0] * u_max).reshape(2, 1)  # single generator from input bound
    Z_u = Zonotope(np.zeros(2), g_u)

    Zs = [Z0]
    Z = Z0
    for _ in range(steps):
        Z_next = Z.affine_map(Ad).minkowski_sum(Z_u).reduce(max_gens=max_gens)
        Zs.append(Z_next)
        Z = Z_next

    return Zs


def sample_zonotope(Z, n_samples=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    if Z.p == 0:
        return Z.c.reshape(1, -1)

    Xi = rng.uniform(-1.0, 1.0, size=(Z.p, n_samples))  # (p, N)
    X = Z.c.reshape(-1, 1) + Z.G @ Xi                   # (n, N)
    return X.T                                          # (N, n)


def plot_reach_tube_boxes_and_samples(
    Zs,
    dt,
    box_stride=50,
    sample_stride=200,
    samples_per_slice=3000,
    rng_seed=0
):
    rng = np.random.default_rng(rng_seed)
    fig, ax = plt.subplots(figsize=(8, 6))

    for k in range(0, len(Zs), sample_stride):
        X = sample_zonotope(Zs[k], n_samples=samples_per_slice, rng=rng)
        ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.08)

    for k in range(0, len(Zs), box_stride):
        lo, hi = Zs[k].interval_bounds()
        rect = Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                         fill=False, linewidth=1.0, alpha=0.6)
        ax.add_patch(rect)

    lo, hi = Zs[-1].interval_bounds()
    rect = Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                     fill=False, linewidth=2.5)
    ax.add_patch(rect)

    ax.set_xlabel("x1 (position)")
    ax.set_ylabel("x2 (velocity)")
    ax.set_title("Linear Oscillator Reachable Tube (Zonotope Boxes + Samples)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Initial set: small box around (x1, x2)
    c0 = np.array([0.2, 0.0])
    rad0 = np.array([0.02, 0.02])
    Z0 = Zonotope(c0, np.diag(rad0))

    # Linear oscillator parameters
    omega = 1.5     # rad/s
    zeta  = 0.05    # damping ratio
    u_max = 0.3

    dt = 0.02
    steps = 800  # 16 seconds

    # If you don't have SciPy, set exact_discretization=False.
    Zs = linear_oscillator_reach_zonotope(
        Z0=Z0,
        dt=dt,
        steps=steps,
        omega=omega,
        zeta=zeta,
        u_max=u_max,
        max_gens=40,
        exact_discretization=True
    )

    plot_reach_tube_boxes_and_samples(
        Zs,
        dt,
        box_stride=40,
        sample_stride=160,
        samples_per_slice=2500,
        rng_seed=1
    )

