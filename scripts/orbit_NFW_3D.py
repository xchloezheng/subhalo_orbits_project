#!/usr/bin/env python3
"""
3D Subhalo Orbit in NFW Host + AHF Measured Centers

- Reads AHF halo catalogs, extracts one halo center per snapshot
- Models the host halo as an NFWPotential (galpy, analytic)
- Integrates a test-particle orbit in that NFW host
- Plots in 3D:
    * Virial radius sphere (R_VIRIAL)        -> solid light black wireframe
    * Outer radius sphere (R_HALO_MAX)       -> dashed light black wireframe
    * Measured subhalo centers (AHF)         -> small blue points
    * Measured subhalo orbit (AHF)           -> thin blue line
    * Predicted NFW orbit (galpy)            -> orange line
    * Host center                             -> black star
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from galpy.potential import NFWPotential
from galpy.orbit import Orbit

# ==================== CONFIGURATION ====================

# --- AHF input ---
AHF_DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Data/subhalo_evp_3_1_1/AHFout/")
TOTAL_SNAPSHOTS = 280          # number of snapshots from AHF (0 to TOTAL_SNAPSHOTS)

# --- Output ---
OUTPUT_DIR      = "./"
SIMULATION_NAME = "subhalo_evp_3_1_1"
SHOW_FIG        = False      # display figure interactively

# --- Host halo geometry (in kpc) ---
HOST_CENTER = np.array([1e4, 1e4, 1e4])  
R_VIRIAL    = 974.407                    
R_HALO_MAX  = 2000.0                     # outer radius for plotting

# --- Host halo mass model (for NFWPotential) ---
#   NFWPotential(mvir=...) expects mvir in units of 1e12 Msun
LOG10_MVIR_MSUN = 13.5                   # host Mvir = 10^13.5 Msun
M_VIR_1E12      = 10**(LOG10_MVIR_MSUN - 12.0)  # in 1e12 Msun units
C_HOST          = 6.5                    # concentration

# --- galpy unit system (dimensionless; we interpret these physically) ---
# I define:
RO = 1.0    # kpc
VO = 220.0  # km/s

# Conversion: 1 km/s * 1 Gyr ≈ 1.022712 kpc
Kpc_per_Gyr_Per_Kms = 1.022712
RO_KPC  = 1.0
VO_KMS  = VO
T_UNIT_GYR = RO_KPC / (VO_KMS * Kpc_per_Gyr_Per_Kms)  # the physical time (in Gyr) corresponding to one galpy time unit
# ≈ 0.00445 Gyr

# --- Initial conditions for predicted orbit (in physical units) ---
R0_KPC   = 0.7 * R_VIRIAL     # starting radius from host center (Rapo)
z0_KPC   = 0.0
phi0_RAD = 0.0

vR0_KMS  = 0.0                # ~apocenter → vR0 = 0 radially
vT0_KMS  = 33.1515            # initial vel (km/s) tangentially
vz0_KMS  = 0.0

# --- Integration parameters for predicted orbit ---
T_MAX_GYR = 14.0              # total integration time for predicted orbit
N_STEPS   = 2001              # time steps

# --- Plot settings ---
FIGSIZE_ORBIT_3D = (8, 7)
POINT_SIZE_3D    = 2          # subhalo points
LINE_WIDTH_3D    = 1.0        # measured orbit line width

# =======================================================


def get_valid_snapshots(data_path, total_snapshots):
    """
    Detect which AHF snapshot files exist and are non-empty.

    Returns:
        list: Valid snapshot numbers
    """
    valid_snapshots = []

    print("Detecting valid AHF files...")
    for i in range(total_snapshots + 1):
        filepath = f"{data_path}{i}-halos"

        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    lines = f.readlines()
                    # Require header + at least one data line
                    if len(lines) > 1:
                        valid_snapshots.append(i)
            except Exception:
                continue

    print(
        f"Found {len(valid_snapshots)} non-empty snapshot files "
        f"out of {total_snapshots + 1} total"
    )
    return valid_snapshots


def load_subhalo_centers_from_ahf(data_path, snapshot_list, host_center):
    """
    Load one halo center per snapshot from AHF files.

    Currently:
        - Uses the first halo listed in each snapshot file.

    Returns:
        centers      : (N, 3) array of [Xc, Yc, Zc]
        snapshots    : (N,) array of snapshot numbers
        orbital_dist : (N,) array of distances from host center
    """
    centers = []
    snapshots = []
    orbital_dist = []

    print("\nExtracting centers from AHF files...")
    for snap in snapshot_list:
        filepath = f"{data_path}{snap}-halos"
        print(f"  snapshot {snap}: {filepath}")

        with open(filepath, "r") as file:
            next(file)  # skip header

            first_line = None
            for line in file:
                if line.strip():
                    first_line = line
                    break

            if first_line is None:
                print(f"    WARNING: no halo lines found in {filepath}")
                continue

            tokens = first_line.strip().split()
            # AHF standard: tokens[5], [6], [7] are Xc, Yc, Zc in kpc
            Xc = float(tokens[5])
            Yc = float(tokens[6])
            Zc = float(tokens[7])

            centers.append([Xc, Yc, Zc])
            snapshots.append(snap)

            orbital_dist.append(
                np.linalg.norm(np.array([Xc, Yc, Zc]) - host_center)
            )

    centers = np.array(centers)
    snapshots = np.array(snapshots)
    orbital_dist = np.array(orbital_dist)

    return centers, snapshots, orbital_dist


def plot_sphere(ax, center, radius, alpha=0.15, **kwargs):
    """
    Plot a light wireframe sphere for the host halo.

    Uses a coarse grid so it doesn't obscure the orbit.
    """
    cx, cy, cz = center
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 9)

    xs = cx + radius * np.outer(np.cos(u), np.sin(v))
    ys = cy + radius * np.outer(np.sin(u), np.sin(v))
    zs = cz + radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(xs, ys, zs, alpha=alpha, rstride=1, cstride=1, **kwargs)


def set_equal_3d(ax):
    """
    Make 3D axes have equal scale (so spheres/orbits aren't distorted).
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    x_middle = 0.5 * (x_limits[0] + x_limits[1])
    y_middle = 0.5 * (y_limits[0] + y_limits[1])
    z_middle = 0.5 * (z_limits[0] + z_limits[1])

    half = 0.5 * max_range
    ax.set_xlim3d(x_middle - half, x_middle + half)
    ax.set_ylim3d(y_middle - half, y_middle + half)
    ax.set_zlim3d(z_middle - half, z_middle + half)


# ---------- NFW predicted orbit (galpy) ----------

def make_host_potential():
    """
    Create NFW host halo potential (galpy units).
    """
    pot = NFWPotential(
        mvir=M_VIR_1E12,  # in 1e12 Msun
        conc=C_HOST,
        ro=RO,
        vo=VO
    )
    print(dir(pot))
    return pot


def make_predicted_orbit():
    """
    Create a galpy Orbit object for the predicted subhalo COM.

    We work in the host-centered frame with:
        R0, z0, phi0, vR0, vT0, vz0
    given in kpc and km/s, converted to galpy units.
    """
    R0_g  = R0_KPC / RO_KPC
    z0_g  = z0_KPC / RO_KPC
    vR0_g = vR0_KMS / VO_KMS
    vT0_g = vT0_KMS / VO_KMS
    vz0_g = vz0_KMS / VO_KMS
    phi0  = phi0_RAD

    o = Orbit(vxvv=[R0_g, vR0_g, vT0_g, z0_g, vz0_g, phi0],
              ro=RO, vo=VO)
    return o


def integrate_predicted_orbit(o, pot):
    """
    Integrate predicted orbit for T_MAX_GYR (physical) in galpy units.

    Returns:
        t_galpy : (N,) times in galpy units
        t_Gyr   : (N,) times in Gyr
    """
    tmax_galpy = T_MAX_GYR / T_UNIT_GYR
    t_galpy = np.linspace(0.0, tmax_galpy, N_STEPS)

    o.integrate(t_galpy, pot)

    t_Gyr = t_galpy * T_UNIT_GYR
    return t_galpy, t_Gyr


def compute_predicted_orbit_positions():
    """
    Compute predicted orbit positions in AHF coordinates (absolute).

    1) Integrate orbit in host-centered frame using NFWPotential.
    2) Get x,y,z in kpc relative to host center.
    3) Shift by HOST_CENTER to align with AHF coordinates.

    Returns:
        x_pred, y_pred, z_pred : arrays of positions in kpc (absolute)
        t_Gyr                  : times in Gyr (for possible diagnostics)
    """
    pot = make_host_potential()
    orb = make_predicted_orbit()
    t_galpy, t_Gyr = integrate_predicted_orbit(orb, pot)

    # Positions in host-centered frame, in kpc
    x_rel = orb.x(t_galpy) * RO_KPC
    y_rel = orb.y(t_galpy) * RO_KPC
    z_rel = orb.z(t_galpy) * RO_KPC

    # Shift to absolute coordinates used in AHF
    x_abs = HOST_CENTER[0] + x_rel
    y_abs = HOST_CENTER[1] + y_rel
    z_abs = HOST_CENTER[2] + z_rel

    return x_abs, y_abs, z_abs, t_Gyr


# ================================
# MAIN: combine AHF + NFW prediction
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) AHF: detect valid snapshots and load measured centers
valid_snapshots = get_valid_snapshots(AHF_DATA_PATH, TOTAL_SNAPSHOTS)
if not valid_snapshots:
    raise RuntimeError("No valid snapshot files found!")

centers, snapshots, orbital_dist = load_subhalo_centers_from_ahf(
    AHF_DATA_PATH, valid_snapshots, HOST_CENTER
)

if centers.size == 0:
    raise RuntimeError("No centers extracted from AHF files.")

x_meas = centers[:, 0]
y_meas = centers[:, 1]
z_meas = centers[:, 2]

# 2) Predicted NFW orbit (galpy)
try:
    x_pred, y_pred, z_pred, t_Gyr = compute_predicted_orbit_positions()
    have_predicted = True
except Exception as e:
    print(f"WARNING: Could not compute predicted orbit with galpy: {e}")
    have_predicted = False
    x_pred = y_pred = z_pred = None

# 3) 3D visualization
fig = plt.figure(figsize=FIGSIZE_ORBIT_3D)
ax = fig.add_subplot(111, projection="3d")

# Host halo: virial radius (solid, light black)
plot_sphere(
    ax,
    HOST_CENTER,
    R_VIRIAL,
    color="k",
    linewidth=0.4,
    linestyle="-"
)

# Host halo: outer radius (dashed, light black)
plot_sphere(
    ax,
    HOST_CENTER,
    R_HALO_MAX,
    color="k",
    linewidth=0.4,
    linestyle="--"
)

# Measured subhalo orbit (AHF): blue line + points
ax.plot(
    x_meas, y_meas, z_meas,
    linewidth=LINE_WIDTH_3D, color="b", label="Subhalo (AHF)"
)
ax.scatter(
    x_meas, y_meas, z_meas,
    s=POINT_SIZE_3D, color="b"
)

# Predicted analytic orbit: orange NFW rosette
if have_predicted:
    ax.plot(
        x_pred, y_pred, z_pred,
        color="orange", linewidth=1.2,
        label="Predicted orbit (NFW)"
    )

# Host center marker
ax.scatter(
    HOST_CENTER[0], HOST_CENTER[1], HOST_CENTER[2],
    s=70, marker="*", color="k", label="Host center"
)

# Labels and title
ax.set_xlabel("x [kpc]")
ax.set_ylabel("y [kpc]")
ax.set_zlabel("z [kpc]")
ax.set_title(f"3D Subhalo Orbit + NFW Prediction - {SIMULATION_NAME}")

# Plot range: include halo, measured orbit, and predicted orbit
rel_meas = centers - HOST_CENTER
max_meas_radius = np.max(np.linalg.norm(rel_meas, axis=1))

candidates = [R_HALO_MAX, max_meas_radius, R0_KPC]
if have_predicted:
    rel_pred = np.vstack(
        [
            x_pred - HOST_CENTER[0],
            y_pred - HOST_CENTER[1],
            z_pred - HOST_CENTER[2],
        ]
    ).T
    max_pred_radius = np.max(np.linalg.norm(rel_pred, axis=1))
    candidates.append(max_pred_radius)

r_plot = 1.1 * max(candidates)

cx, cy, cz = HOST_CENTER
ax.set_xlim(cx - r_plot, cx + r_plot)
ax.set_ylim(cy - r_plot, cy + r_plot)
ax.set_zlim(cz - r_plot, cz + r_plot)

set_equal_3d(ax)
ax.legend(loc="upper left")

plt.tight_layout()

outpath = os.path.join(OUTPUT_DIR, f"orbit3D_with_NFW_{SIMULATION_NAME}.png")
plt.savefig(outpath, dpi=200)
print(f"\nSaved 3D combined orbit plot to {outpath}")

if SHOW_FIG:
    plt.show()
else:
    plt.close(fig)