import numpy as np
import matplotlib.pyplot as plt
from qutils.orbital import dim2NonDim6, orbitalEnergy

def apply_noise(data, pos_noise_std, vel_noise_std):
    # assumes data shape is (num_samples, seq_length, 6)
    mid = data.shape[2] // 2  # Split index
    pos_noise = np.random.normal(0, pos_noise_std, size=data[:, :, :mid].shape)
    vel_noise = np.random.normal(0, vel_noise_std, size=data[:, :, mid:].shape)
    noisy_data = data.copy()
    noisy_data[:, :, :mid] += pos_noise
    noisy_data[:, :, mid:] += vel_noise
    return noisy_data


dt = 60

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--propMin', type=int, default=10, help='Number of minutes of propagation')
parser.add_argument('--n', type=int, default=10000, help='Number of random systems')
parser.add_argument('--randPlots', type=int, default=50, help='Number of random plots to generate')
parser.add_argument('--trainSteps', type=int, default=5, help='Number of initial time steps to color as training region')
parser.add_argument('--noise', action='store_true', help='Apply noise to the states')
parser.add_argument('--velNoise', type=float, default=1e-3, help='Velocity noise standard deviation in km/s')
parser.add_argument('--dv',type=float,default="5.0",help="dataset to analyze. defaults to cont thrust dataset")
args = parser.parse_args()
numMinProp = args.propMin
numRandSys = args.n
randPlots = args.randPlots
train_steps = max(0, args.trainSteps)
useNoise = args.noise
vel_noise_std = args.velNoise
pos_noise_std = vel_noise_std * 1e3 


dataLoc = f"./data/gmat/{args.dv}km-{numRandSys}"

# get npz files in folder and load them into script


a = np.load(f"{dataLoc}/statesArrayImpBurn.npy")
statesArrayImpBurn = a['statesArrayImpBurn']

print(statesArrayImpBurn.shape)

if useNoise:
    statesArrayImpBurn = apply_noise(statesArrayImpBurn, pos_noise_std, vel_noise_std)


energyImpBurn= np.zeros((statesArrayImpBurn.shape[0],statesArrayImpBurn.shape[1],1))
for i in range(statesArrayImpBurn.shape[0]):
    energyImpBurn[i,:,0] = orbitalEnergy(statesArrayImpBurn[i,:,:])

t = np.linspace(0,numMinProp*dt,len(statesArrayImpBurn[0,:,0]))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(statesArrayImpBurn[0,:,0],statesArrayImpBurn[0,:,1],statesArrayImpBurn[0,:,2],label='Impulsive')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Trajectory of a Single Earth Orbiter')
ax.legend(loc='lower left')
ax.axis('equal')

from matplotlib.lines import Line2D
colors = ['C0', 'C1', 'C2', 'C3']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
labels = ['Chemical Thrust', 'Electrical Thrust', 'Impulsive Thrust', 'No Thrust']
ax.legend(lines, labels)
ax.axis('equal')

plt.figure()
for j in range(randPlots):
    i = np.random.randint(0, len(statesArrayImpBurn))

    plt.plot(t, energyImpBurn[i,:,0], label='Impulsive',color='C2')
plt.legend(lines, labels)
plt.grid()
plt.xlabel('Time (s)')
plt.title("Energy of "+str(randPlots*4)+" Earth Orbiters")

plt.figure()
plt.plot(t, statesArrayImpBurn[0,:,0], label='Impulsive X')
plt.plot(t, statesArrayImpBurn[0,:,1], label='Impulsive Y')
plt.plot(t, statesArrayImpBurn[0,:,2], label='Impulsive Z')
plt.xlabel('Time (s)')
plt.ylabel('Position (km)')
plt.title('Position vs Time for Different Thruster Profiles')
plt.legend(loc='lower left')
plt.grid()


R_E = 6378.137  # km

def plot_earth_fast(ax, R=R_E, u_res=60, v_res=30):
    u = np.linspace(0, 2*np.pi, u_res)
    v = np.linspace(0, np.pi, v_res)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    surf = ax.plot_surface(
        x, y, z,
        linewidth=0, antialiased=False, shade=False,
        color="#A8922D", alpha=0.3, zorder=0
    )
    # make sure the sphere sits between back and front lines in the 3D sort
    try: surf.set_sort_zpos(0.0)
    except Exception: pass
    surf.set_rasterized(True)
    return surf

def _view_vec(ax):
    A = np.deg2rad(ax.azim); E = np.deg2rad(ax.elev)
    v = np.array([np.cos(E)*np.cos(A), np.cos(E)*np.sin(A), np.sin(E)])
    return v / np.linalg.norm(v)

def draw_orbits_translucent(ax, orbits, R=R_E, tag="__orbit__",
                            front_kw=None, back_kw=None, train_steps=0,
                            train_color="#E24A33"):
    # remove previously drawn segments from earlier views
    for ln in list(ax.lines):
        if getattr(ln, "_tag", None) == tag:
            ln.remove()

    v = _view_vec(ax)
    for kind, (x, y, z) in orbits:
        p_dot_v = x*v[0] + y*v[1] + z*v[2]
        r2 = x*x + y*y + z*z
        perp2 = r2 - p_dot_v**2
        visible = (p_dot_v >= 0) | (perp2 >= R*R)
        m = min(train_steps, x.size)
        is_train = np.zeros(x.size, dtype=bool)
        if m > 0:
            is_train[:m] = True

        # split into contiguous segments whenever visibility or train region changes
        state = visible.astype(np.int8) * 2 + is_train.astype(np.int8)
        cuts = np.flatnonzero(np.diff(state) != 0) + 1
        for seg in np.split(np.arange(x.size), cuts):
            if seg.size < 2:
                continue
            is_front = bool(visible[seg[0]])
            seg_is_train = bool(is_train[seg[0]])
            kw = (front_kw or {}).get(kind, {}) if is_front else (back_kw or {}).get(kind, {})
            if seg_is_train:
                kw = dict(kw)
                kw["color"] = train_color
                kw["lw"] = max(float(kw.get("lw", 1.2)), 2.2)
            zord = 4 if is_front else 2
            ln, = ax.plot(x[seg], y[seg], z[seg], zorder=zord, **kw)
            ln._tag = tag
            # hard-order in 3D painter: back ≪ sphere ≪ front
            try: ln.set_sort_zpos(+1e9 if is_front else -1e9)
            except Exception: pass

# --- shuffle your datasets (unchanged) ---
indices = np.random.permutation(statesArrayImpBurn.shape[0])
statesArrayImpBurn = statesArrayImpBurn[indices] 

# --- figure ---
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = plot_earth_fast(ax, R=R_E, u_res=48, v_res=24)
ax.set_proj_type('ortho')  # cheaper + avoids perspective ambiguity

# collect random trajectories once
orbits = []
for _ in range(randPlots):
    iI = np.random.randint(statesArrayImpBurn.shape[0])

    orbits.append(("Impulsive", (statesArrayImpBurn[iI,:,0],
                                 statesArrayImpBurn[iI,:,1],
                                 statesArrayImpBurn[iI,:,2])))

# styles
front_kw = {
    "Chemical": {"color": "C0", "lw": 1.8},
    "Electric": {"color": "C1", "lw": 1.8},
    "Impulsive": {"color": "C2", "lw": 1.8},
    "NoThrust": {"color": "C3", "lw": 1.8},
}
# slightly faded for the hidden half (still visible through Earth)
back_kw = {
    "Chemical": {"color": "C0", "lw": 1.2, "alpha": 0.5},
    "Electric": {"color": "C1", "lw": 1.2, "alpha": 0.5},
    "Impulsive": {"color": "C2", "lw": 1.2, "alpha": 0.5},
    "NoThrust": {"color": "C3", "lw": 1.2, "alpha": 0.5},
}

# initial draw (order: back lines, sphere, front lines) achieved via sort_zpos
draw_orbits_translucent(ax, orbits, R=R_E, front_kw=front_kw, back_kw=back_kw, train_steps=train_steps)

# axes, limits, labels
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
ax.set_title(f'Trajectories of {randPlots} Earth Orbiters')
ax.set_box_aspect((1, 1, 1))
max_r = max([np.sqrt(np.nanmax(x*x + y*y + z*z)) for _, (x, y, z) in orbits] + [R_E])
lim = 1.05 * max_r
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
legend_lines = [
    Line2D([0], [0], color="#E24A33", lw=2.2),
    Line2D([0], [0], color=front_kw["Impulsive"]["color"], lw=1.8),
]
ax.legend(legend_lines, [f"Training region (first {train_steps} minutes)","Testing Region"], loc="upper right")
plt.tight_layout()

# redraw orbits when camera changes
def _redraw(_evt=None):
    draw_orbits_translucent(ax, orbits, R=R_E, front_kw=front_kw, back_kw=back_kw, train_steps=train_steps)
    fig.canvas.draw_idle()
fig.canvas.mpl_connect('button_release_event', _redraw)
fig.canvas.mpl_connect('key_release_event', _redraw)

plt.show()
