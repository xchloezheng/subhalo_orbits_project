#!/usr/bin/env python3
"""
3D Subhalo Orbit Visualization from AHF Outputs

- Reads AHF halo catalogs, extracts one halo center per snapshot
- Plots the subhalo orbit in 3D around a host halo
- Draws:
    * Virial radius (R_VIRIAL)  -> solid light black wireframe
    * Outer radius (R_HALO_MAX) -> dashed light black wireframe
    * Subhalo centers           -> small blue points
    * Subhalo orbit             -> thin blue line
    * Predicted orbit (galpy NFW rosette) -> orange line
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- NEW: galpy / astropy imports for predicted orbit ---
from galpy.potential import NFWPotential
from galpy.orbit import Orbit
import astropy.units as u

# ==================== CONFIGURATION ====================

# Input path to AHF halo catalogs
AHF_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Data/subhalo_evp_3_1_1/AHFout/')

# Output directory for the 3D orbit plot
OUTPUT_DIR = './'

# Simulation name (used in filename and plot title)
SIMULATION_NAME = 'subhalo_evp_3_1_1'

# Total number of snapshots to check (0 .. TOTAL_SNAPSHOTS)
TOTAL_SNAPSHOTS = 280

# Host halo center coordinates (kpc)
HOST_CENTER = np.array([1e4, 1e4, 1e4])

# Host radii (kpc)
R_VIRIAL   = 974.407      # from your sheet
R_HALO_MAX = 2000.0       # outer radius (e.g. max particle radius)

# ---------- Host halo parameters for galpy NFWPotential ----------
# Adjust if you update the host properties
M_VIR_HOST = (10**13.5) * u.Msun  # host virial mass
C_HOST     = 6.5                  # concentration
R_VIR_HOST = R_VIRIAL * u.kpc     # should match R_VIRIAL above
R_S_HOST   = R_VIR_HOST / C_HOST  # scale radius (for your info)

# galpy unit system
RO = 1.0 * u.kpc          # length unit
VO = 200.0 * u.km/u.s     # velocity unit (roughly halo Vc; tweak if desired)

# Approximate apocenter radius for initial condition (kpc)
R_APO  = 0.7 * R_VIRIAL          # ~682.085 kpc
# Tangential velocity at apocenter for initial condition (km/s)
V_T_INIT = 33.1515 * u.km/u.s    # from your sheet / sanity-check calc
# We start at apocenter, so v_R = 0.

# Integration time for predicted orbit (Gyr)
T_MAX_GYR     = 10.0             # total integration time for predicted orbit
N_PRED_POINTS = 1000             # number of time steps for predicted orbit

# 3D plot settings
FIGSIZE_ORBIT_3D = (8, 7)
POINT_SIZE_3D = 2          # smaller subhalo points
LINE_WIDTH_3D = 1.0        # thin solid line for measured orbit

SHOW_FIG = True            # show interactive window on laptop

# =======================================================


def get_valid_snapshots(data_path, total_snapshots):
    """
    Detect which snapshot files exist and are non-empty.

    Returns:
        list: Valid snapshot numbers
    """
    valid_snapshots = []

    print("Detecting valid AHF files...")
    for i in range(total_snapshots + 1):
        filepath = f'{data_path}{i}-halos'

        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    # Require header + at least one data line
                    if len(lines) > 1:
                        valid_snapshots.append(i)
            except Exception:
                continue

    print(f"Found {len(valid_snapshots)} non-empty snapshot files out of {total_snapshots + 1} total")
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
        filepath = f'{data_path}{snap}-halos'
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
    # Coarse sampling: fewer u, v points -> fewer grid lines
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 9)

    xs = cx + radius * np.outer(np.cos(u), np.sin(v))
    ys = cy + radius * np.outer(np.sin(u), np.sin(v))
    zs = cz + radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(xs, ys, zs, alpha=alpha, rstride=1, cstride=1, **kwargs)


def set_equal_3d(ax):
    """
    Make 3D axes have equal scale (so spheres look like spheres).
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    half = 0.5 * max_range
    ax.set_xlim3d(x_middle - half, x_middle + half)
    ax.set_ylim3d(y_middle - half, y_middle + half)
    ax.set_zlim3d(z_middle - half, z_middle + half)


# ----------- NEW: galpy-predicted orbit in NFW potential -----------
def compute_predicted_orbit_galpy():
    """
    Compute a predicted orbit using galpy in an NFWPotential.

    - Host is modeled as an NFW halo with (M_VIR_HOST, C_HOST, R_VIR_HOST).
    - We start the subhalo at apocenter:
        R0   = R_APO
        vR0  = 0
        vT0  = V_T_INIT
        z0   = 0, vz0 = 0
      in the orbital plane, and integrate for T_MAX_GYR.
    - Orbit is computed in a frame where the host is at the origin, then
      shifted by HOST_CENTER for plotting.
    """
    # Define host potential
    host_pot = NFWPotential(mvir=M_VIR_HOST, conc=C_HOST, ro=RO, vo=VO)

    # Initial conditions in cylindrical coordinates (galpy units)
    R0   = (R_APO * u.kpc).to(RO.unit).value / RO.value  # in units of ro
    vR0  = 0.0                                           # in units of vo
    vT0  = (V_T_INIT.to(VO.unit).value / VO.value)       # in units of vo
    z0   = 0.0                                           # in units of ro
    vz0  = 0.0                                           # in units of vo
    phi0 = 0.0                                           # radians

    o = Orbit(vxvv=[R0, vR0, vT0, z0, vz0, phi0],
              ro=RO, vo=VO)

    # Time grid in Gyr
    t = np.linspace(0.0, T_MAX_GYR, N_PRED_POINTS) * u.Gyr

    # Integrate orbit
    o.integrate(t, host_pot)

    # Extract Cartesian coordinates in kpc (host at origin)
    x_rel = o.x(t) * RO.to(u.kpc).value
    y_rel = o.y(t) * RO.to(u.kpc).value
    z_rel = o.z(t) * RO.to(u.kpc).value

    # Shift to actual host center position
    x = HOST_CENTER[0] + x_rel
    y = HOST_CENTER[1] + y_rel
    z = HOST_CENTER[2] + z_rel

    return x, y, z


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Detect valid snapshots and load measured centers
    valid_snapshots = get_valid_snapshots(AHF_DATA_PATH, TOTAL_SNAPSHOTS)
    if not valid_snapshots:
        print("Error: No valid snapshot files found!")
        return

    centers, snapshots, orbital_dist = load_subhalo_centers_from_ahf(
        AHF_DATA_PATH, valid_snapshots, HOST_CENTER
    )

    if centers.size == 0:
        print("Error: No centers extracted from AHF files.")
        return

    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]

    # 2) Predicted orbit from galpy NFWPotential
    try:
        x_pred, y_pred, z_pred = compute_predicted_orbit_galpy()
        have_predicted_orbit = True
    except Exception as e:
        print(f"WARNING: Could not compute predicted orbit with galpy: {e}")
        have_predicted_orbit = False

    # 3) 3D orbit visualization
    fig = plt.figure(figsize=FIGSIZE_ORBIT_3D)
    ax = fig.add_subplot(111, projection="3d")

    # Host halo: virial radius (solid, light black)
    plot_sphere(
        ax,
        HOST_CENTER,
        R_VIRIAL,
        color='k',
        linewidth=0.4,
        linestyle='-'
    )

    # Host halo: outer radius (dashed, light black)
    plot_sphere(
        ax,
        HOST_CENTER,
        R_HALO_MAX,
        color='k',
        linewidth=0.4,
        linestyle='--'
    )

    # Measured subhalo orbit: thin blue line + small blue points
    ax.plot(x, y, z, linewidth=LINE_WIDTH_3D, color='b', label="Subhalo (AHF)")
    ax.scatter(x, y, z, s=POINT_SIZE_3D, color='b')

    # Predicted analytic orbit: NFW rosette from galpy
    if have_predicted_orbit:
        ax.plot(x_pred, y_pred, z_pred, color='orange', linewidth=1.2,
                label="Predicted orbit (NFW, galpy)")

    # Host center marker
    ax.scatter(
        HOST_CENTER[0], HOST_CENTER[1], HOST_CENTER[2],
        s=70, marker="*", color='k', label="Host center"
    )

    # Labels and title
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_zlabel("z [kpc]")
    ax.set_title(f"3D Subhalo Orbit - {SIMULATION_NAME}")

    # Plot range: include both halo and orbit
    rel_positions = centers - HOST_CENTER
    max_orbit_radius = np.max(np.linalg.norm(rel_positions, axis=1))
    r_plot = 1.1 * max(R_HALO_MAX, max_orbit_radius, R_APO)

    cx, cy, cz = HOST_CENTER
    ax.set_xlim(cx - r_plot, cx + r_plot)
    ax.set_ylim(cy - r_plot, cy + r_plot)
    ax.set_zlim(cz - r_plot, cz + r_plot)

    set_equal_3d(ax)
    ax.legend(loc="upper left")

    plt.tight_layout()

    orbit3d_plot_path = os.path.join(OUTPUT_DIR, f'orbit3D_{SIMULATION_NAME}.png')
    plt.savefig(orbit3d_plot_path, dpi=200)
    print(f"\nSaved 3D orbit plot to {orbit3d_plot_path}")

    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()