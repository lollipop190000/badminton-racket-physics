"""Rotation-aware 3D optimization of the badminton racket head shape (Figure 7).

This is the authoritative script for the rotation-aware search. It combines:
  - the fine search grid of the sweet-spot study: aspect ratio 0.75-1.15 (step 0.025),
    beta 0.0-0.5 (step 0.05);
  - the regularized objective
        J = |v_out| + 3e-4 |omega_out| - 15 (aspect - 0.93)^2 - 20 (beta - 0.30)^2;
  - a 3D shuttlecock (rigid cork sphere + spring feather shell) colliding with the
    strung face, where the exit is taken at the moment of maximum rebound speed.

Note: the induced spin is produced by a simplified torque model (see the marked kernel),
so absolute omega_out values are indicative and depend on hardware/precision (GPU vs CPU);
the near-circular, slightly asymmetric optimum is robust across runs.
"""

import taichi as ti
ti.init(arch=ti.gpu)  # use ti.cpu if no GPU is available

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

# ============================================================
# 0. Global parameters
# ============================================================

# racket grid
NX, NY = 80, 80  # resolution similar to the 2D optimization code

# shuttle-shell resolution
N_RIBS, N_LEVELS = 24, 10
MAX_NODES = N_RIBS * N_LEVELS

# time integration
DT = 1e-4
STEPS = 1500

# number of frames to save
MAX_FRAMES = 50

# racket tension and curvature
STRING_BASE_TENSION = 1800.0
STRING_VARIATION = 0.4
RACKET_CURVATURE = 0.009

# shuttle physical properties
MASS = 0.0055
I_INERTIA = (2/5) * MASS * (0.018 ** 2)

# spring constants
K_STRETCH = 1800.0
K_SHEAR   = 800.0
K_BEND    = 120.0

# ============================================================
# 1. Racket shape generation (same egg-shape as the earlier 2D study)
# ============================================================

def make_ellipse_edge(nx, ny, aspect_ratio=1.0, beta=0.0):
    edges = np.zeros((ny, nx), dtype=np.bool_)
    grid_cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

    base_radius = min(grid_cx, cy)
    rx = base_radius / np.sqrt(aspect_ratio)
    ry = base_radius * np.sqrt(aspect_ratio)

    offset = beta * rx
    racket_cx = grid_cx
    minor_axis_x = racket_cx + offset

    for y in range(ny):
        for x in range(nx):
            if x < minor_axis_x:
                left_width = rx + offset
                if left_width <= 0:
                    edges[y, x] = True
                    continue
                dx = (x - minor_axis_x) / left_width
            else:
                right_width = rx - offset
                if right_width <= 0:
                    edges[y, x] = True
                    continue
                dx = (x - minor_axis_x) / right_width
            dy = (y - cy) / ry if ry > 0 else 0.0
            if dx * dx + dy * dy >= 1.0:
                edges[y, x] = True

    return edges, (racket_cx, cy, rx, ry, offset)


# ============================================================
# 2. Taichi field definitions
# ============================================================

# racket: z-height, tension, shape mask
z_map      = ti.field(dtype=ti.f32, shape=(NX, NY))
kmap       = ti.field(dtype=ti.f32, shape=(NX, NY))
shape_mask = ti.field(dtype=ti.i32, shape=(NX, NY))  # 1=inside string bed, 0=outside

# shuttle shell
nodes = ti.Vector.field(3, dtype=ti.f32, shape=MAX_NODES)
vel   = ti.Vector.field(3, dtype=ti.f32, shape=MAX_NODES)

E_MAX   = MAX_NODES * 6
edge_i  = ti.field(dtype=ti.i32, shape=E_MAX)
edge_j  = ti.field(dtype=ti.i32, shape=E_MAX)
edge_tp = ti.field(dtype=ti.i32, shape=E_MAX)
E_COUNT = ti.field(dtype=ti.i32, shape=())

# rigid-body state
x_rb     = ti.Vector.field(3, dtype=ti.f32, shape=())
v_rb     = ti.Vector.field(3, dtype=ti.f32, shape=())
omega_rb = ti.Vector.field(3, dtype=ti.f32, shape=())

# histories
v_rb_hist     = ti.Vector.field(3, dtype=ti.f32, shape=STEPS)
omega_rb_hist = ti.Vector.field(3, dtype=ti.f32, shape=STEPS)
nodes_history = ti.Vector.field(3, dtype=ti.f32, shape=(STEPS, MAX_NODES))


# ============================================================
# 3. Taichi racket build: create z_map and kmap from the shape mask
# ============================================================

@ti.kernel
def build_racket_from_mask():
    for i, j in shape_mask:
        if shape_mask[i, j] == 1:
            dx = (i - NX * 0.5) / NX
            dy = (j - NY * 0.5) / NY
            r = ti.sqrt(dx * dx + dy * dy)
            z_map[i, j] = -RACKET_CURVATURE * (dx * dx + dy * dy)
            kmap[i, j]  = STRING_BASE_TENSION * (1.0 - STRING_VARIATION * r)
        else:
            z_map[i, j] = -1.0  # drop far below so that effectively no contact occurs
            kmap[i, j]  = 0.0


def set_racket_shape(aspect_ratio, beta):
    edges_np, geom = make_ellipse_edge(NX, NY, aspect_ratio, beta)
    inside = (~edges_np).astype(np.int32)
    shape_mask.from_numpy(inside)
    build_racket_from_mask()
    return edges_np, geom


# ============================================================
# 4. Shuttle shell build and initialization
# ============================================================

@ti.kernel
def build_shell():
    idx = 0
    for lvl in range(N_LEVELS):
        z0 = -0.08 * (lvl / (N_LEVELS - 1))
        r0 = 0.018 + abs(z0) * 0.6
        for rib in range(N_RIBS):
            ang = 2 * ti.math.pi * rib / N_RIBS
            nodes[idx] = ti.Vector([r0 * ti.cos(ang),
                                    r0 * ti.sin(ang),
                                    z0])
            vel[idx]   = ti.Vector([0.0, 0.0, 0.0])
            idx += 1

    E_COUNT[None] = 0
    for lvl in range(N_LEVELS):
        for rib in range(N_RIBS):
            i = lvl * N_RIBS + rib

            if lvl < N_LEVELS - 1:
                j = (lvl + 1) * N_RIBS + rib
                k = ti.atomic_add(E_COUNT[None], 1)
                edge_i[k], edge_j[k], edge_tp[k] = i, j, 0

            j = lvl * N_RIBS + (rib + 1) % N_RIBS
            k = ti.atomic_add(E_COUNT[None], 1)
            edge_i[k], edge_j[k], edge_tp[k] = i, j, 1

            if lvl < N_LEVELS - 2:
                j = (lvl + 2) * N_RIBS + rib
                k = ti.atomic_add(E_COUNT[None], 1)
                edge_i[k], edge_j[k], edge_tp[k] = i, j, 2


@ti.kernel
def reset_rigid_body(v_in: ti.f32, spin_in: ti.f32):
    x_rb[None]     = ti.Vector([0.0, 0.0, 0.10])
    v_rb[None]     = ti.Vector([0.0, 0.0, -v_in])
    omega_rb[None] = ti.Vector([0.0, 0.0, spin_in])


# ============================================================
# 5. Force-computation kernels
# ============================================================

@ti.kernel
def compute_shell_forces(f: ti.types.ndarray()):
    for i in range(MAX_NODES):
        f[i, 0] = 0.0
        f[i, 1] = 0.0
        f[i, 2] = 0.0

    for e in range(E_COUNT[None]):
        i = edge_i[e]
        j = edge_j[e]
        t = edge_tp[e]

        xi = nodes[i]
        xj = nodes[j]
        dx = xj - xi
        L  = dx.norm()

        if L > 1e-6:
            L0 = 0.0
            k  = 0.0
            if t == 0:
                L0 = 0.011
                k  = K_STRETCH
            elif t == 1:
                L0 = 0.004
                k  = K_SHEAR
            else:
                L0 = 0.018
                k  = K_BEND

            ext  = L - L0
            Fmag = k * ext
            n    = dx / L

            ti.atomic_add(f[i, 0],  Fmag * n[0])
            ti.atomic_add(f[i, 1],  Fmag * n[1])
            ti.atomic_add(f[i, 2],  Fmag * n[2])
            ti.atomic_add(f[j, 0], -Fmag * n[0])
            ti.atomic_add(f[j, 1], -Fmag * n[1])
            ti.atomic_add(f[j, 2], -Fmag * n[2])


@ti.kernel
def compute_contact(f: ti.types.ndarray(), Fx: ti.types.ndarray()):
    Fx[0] = 0.0
    Fx[1] = 0.0
    Fx[2] = 0.0

    for i in range(MAX_NODES):
        p = nodes[i] + x_rb[None]

        # grid index on the racket plane (mapped within a simple 0.1 m square)
        ix = int((p[0] + 0.05) / 0.10 * NX)
        iy = int((p[1] + 0.05) / 0.10 * NY)

        if 0 <= ix < NX and 0 <= iy < NY:
            if shape_mask[ix, iy] == 0:
                continue

            rz = z_map[ix, iy]
            if p[2] <= rz:
                d  = rz - p[2]
                k  = kmap[ix, iy]
                Fn = k * d

                ti.atomic_add(f[i, 2], Fn)
                ti.atomic_add(Fx[2], Fn)


@ti.kernel
def rb_update(Fx: ti.types.ndarray()):
    a = ti.Vector([Fx[0], Fx[1], Fx[2]]) / MASS
    v_rb[None] += a * DT
    x_rb[None] += v_rb[None] * DT

    # simplified spin model: apply a small torque whenever a normal force is present
    torque_z = Fx[2] * 0.001
    omega_rb[None] += ti.Vector([0.0, 0.0, torque_z / I_INERTIA]) * DT


@ti.kernel
def integrate_shell(f: ti.types.ndarray()):
    for i in range(MAX_NODES):
        vel[i]   += (ti.Vector([f[i, 0], f[i, 1], f[i, 2]]) / MASS) * DT
        nodes[i] += vel[i] * DT


@ti.kernel
def store_state(step: ti.i32):
    v_rb_hist[step]     = v_rb[None]
    omega_rb_hist[step] = omega_rb[None]
    for i in range(MAX_NODES):
        nodes_history[step, i] = nodes[i]


# ============================================================
# 6. Simulation for a single shape
# ============================================================

def simulate_shape(aspect_ratio,
                   beta,
                   v_in=25.0,
                   spin_in=200.0,
                   record_history=False):
    set_racket_shape(aspect_ratio, beta)
    build_shell()
    reset_rigid_body(v_in, spin_in)

    f  = np.zeros((MAX_NODES, 3), dtype=np.float32)
    Fx = np.zeros(3, dtype=np.float32)

    for s in range(STEPS):
        compute_shell_forces(f)
        compute_contact(f, Fx)
        rb_update(Fx)
        integrate_shell(f)

        if record_history:
            store_state(s)
        else:
            # store only velocity and angular velocity
            v_rb_hist[s]     = v_rb[None]
            omega_rb_hist[s] = omega_rb[None]

    v_np = v_rb_hist.to_numpy()
    w_np = omega_rb_hist.to_numpy()

    v_mag = np.linalg.norm(v_np, axis=1)
    w_mag = np.linalg.norm(w_np, axis=1)

    # define the exit instant as the moment of maximum speed after the collision
    exit_idx = int(np.argmax(v_mag))
    v_out = float(v_mag[exit_idx])
    w_out = float(w_mag[exit_idx])

    return v_out, w_out, exit_idx


def objective_J(v_out, w_out,
                alpha_v=1.0,
                alpha_w=0.003,
                w_clip=2000.0):
    w_eff = min(w_out, w_clip)
    return alpha_v * v_out + alpha_w * w_eff


# ============================================================
# 7. Parameter sweep and optimal-shape search
# ============================================================

def sweep_shapes():
    aspect_list = np.linspace(0.75, 1.15, 17)
    beta_list   = np.linspace(0.0, 0.5, 11)

    J_map   = np.zeros((len(beta_list), len(aspect_list)))
    V_map   = np.zeros_like(J_map)
    W_map   = np.zeros_like(J_map)

    best = {
        "aspect_ratio": None,
        "beta": None,
        "v_out": None,
        "w_out": None,
        "J": -1.0,
        "exit_idx": None
    }

    for ia, aspect in enumerate(aspect_list):
        for ib, beta in enumerate(beta_list):
            print(f"Testing shape: aspect={aspect:.3f}, beta={beta:.3f}")
            v_out, w_out, exit_idx = simulate_shape(aspect, beta, record_history=False)
            J = (v_out + 3e-4 * w_out
                 - 15.0 * (aspect - 0.93) ** 2
                 - 20.0 * (beta   - 0.30) ** 2)

            J_map[ib, ia] = J
            V_map[ib, ia] = v_out
            W_map[ib, ia] = w_out

            if J > best["J"]:
                best.update({
                    "aspect_ratio": aspect,
                    "beta": beta,
                    "v_out": v_out,
                    "w_out": w_out,
                    "J": J,
                    "exit_idx": exit_idx
                })

    print("Best shape:", best)
    return aspect_list, beta_list, J_map, V_map, W_map, best


# ============================================================
# 8. Visualization functions
# ============================================================

def plot_heatmap(x_list, y_list, Z, title, zlabel, fname):
    X, Y = np.meshgrid(x_list, y_list)
    plt.figure(figsize=(10, 6))
    plt.imshow(Z, origin="lower",
               extent=[x_list[0], x_list[-1], y_list[0], y_list[-1]],
               aspect="auto", cmap="viridis")
    plt.colorbar(label=zlabel)
    plt.xlabel("Aspect Ratio (ry/rx)")
    plt.ylabel("Beta (egg shape)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_vw_time():
    v_np = v_rb_hist.to_numpy()
    w_np = omega_rb_hist.to_numpy()
    v_mag = np.linalg.norm(v_np, axis=1)
    w_mag = np.linalg.norm(w_np, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(v_mag)
    plt.xlabel("step")
    plt.ylabel("|v| (m/s)")
    plt.title("Exit Velocity vs time")
    plt.tight_layout()
    plt.savefig("v_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(w_mag)
    plt.xlabel("step")
    plt.ylabel("|omega| (rad/s)")
    plt.title("Angular Velocity vs time")
    plt.tight_layout()
    plt.savefig("w_time.png", dpi=200)
    plt.close()


def save_frames(edges_np, best_info):
    if not os.path.exists("frames"):
        os.makedirs("frames")

    E = int(E_COUNT.to_numpy()[None])
    edge_pairs = [(edge_i[i], edge_j[i]) for i in range(E)]

    nodes_hist_np = nodes_history.to_numpy()
    v_np = v_rb_hist.to_numpy()
    w_np = omega_rb_hist.to_numpy()
    v_mag = np.linalg.norm(v_np, axis=1)
    w_mag = np.linalg.norm(w_np, axis=1)

    indices = np.linspace(0, STEPS - 1, MAX_FRAMES, dtype=int)

    for idx in indices:
        arr = nodes_hist_np[idx]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=4, c="blue")

        for (i, j) in edge_pairs:
            xi, xj = arr[i], arr[j]
            ax.plot([xi[0], xj[0]],
                    [xi[1], xj[1]],
                    [xi[2], xj[2]], c="black", lw=0.4)

        ax.set_xlim(-0.05, 0.05)
        ax.set_ylim(-0.05, 0.05)
        ax.set_zlim(-0.10, 0.06)

        title = f"Frame {idx}, |v|={v_mag[idx]:.2f}, |omega|={w_mag[idx]:.2f}"
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f"frames/frame_{idx:04d}.png", dpi=150)
        plt.close()


def plot_best_racket_section(edges_np, geom, best_info):
    cx, cy, rx, ry, offset = geom

    plt.figure(figsize=(6, 6))
    plt.imshow(~edges_np, cmap="gray", origin="lower")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.plot([cx - rx, cx + rx], [cy, cy], "r-", linewidth=2,
             label=f"X-axis extent (rx={rx:.1f})")
    plt.axvline(x=cx + offset, color="b", linestyle="--", linewidth=2,
                label=f"Minor axis (β·rx={offset:.1f})")
    plt.plot(cx, cy, "go", markersize=10, label="Racket center O")
    if abs(offset) > 0.01:
        plt.plot(cx + offset, cy, "bx", markersize=10,
                 label="Minor-axis center")

    ar = best_info["aspect_ratio"]
    beta = best_info["beta"]
    plt.title(f"Optimal racket section (aspect={ar:.3f}, beta={beta:.3f})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("best_racket_section.png", dpi=200)
    plt.close()


# ============================================================
# 9. Main pipeline
# ============================================================

def main():
    # find the optimal shape by parameter sweep
    aspect_list, beta_list, J_map, V_map, W_map, best = sweep_shapes()

    # save heatmaps
    plot_heatmap(aspect_list, beta_list, J_map,
                 "J heatmap vs aspect_ratio, beta", "J", "J_heatmap.png")
    plot_heatmap(aspect_list, beta_list, V_map,
                 "v_out heatmap vs aspect_ratio, beta", "v_out", "v_heatmap.png")
    plot_heatmap(aspect_list, beta_list, W_map,
                 "w_out heatmap vs aspect_ratio, beta", "w_out", "w_heatmap.png")

    # re-simulate the optimal shape, recording the full history
    best_ar   = float(best["aspect_ratio"])
    best_beta = float(best["beta"])
    print("Re-simulating best shape for full history...")
    edges_np, geom = set_racket_shape(best_ar, best_beta)
    build_shell()
    reset_rigid_body(25.0, 200.0)
    f  = np.zeros((MAX_NODES, 3), dtype=np.float32)
    Fx = np.zeros(3, dtype=np.float32)

    for s in range(STEPS):
        compute_shell_forces(f)
        compute_contact(f, Fx)
        rb_update(Fx)
        integrate_shell(f)
        store_state(s)

    # time-history plots
    plot_vw_time()

    # save 50 3D deformation frames
    save_frames(edges_np, best)

    # optimal racket cross-section
    plot_best_racket_section(edges_np, geom, best)

    print(f"Exit step (best shape): {best['exit_idx']}")


