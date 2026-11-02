#!/usr/bin/env python3
"""
3D Subhalo Orbit in a TABULATED (multi-time) spherical host potential + AHF measured centers

What changes vs your NFW version?
- Replaces galpy's NFWPotential with a custom galpy Potential that reads host_sigv.txt
- host_sigv.txt contains multi-time blocks of [r, rho(r), M(<r), sigv(r)]
- We build Menc(r,t) by:
    (1) interpolating in r within each time slice
    (2) interpolating in t between time slices
- Orbit integration stays in galpy (Orbit.integrate)

Required inputs:
- host_sigv.txt : multi-time table (stacked blocks of identical r-grid)
- host_times.txt: a 1-column file listing the times (Gyr) corresponding to each block
  (If you don't have this yet, you MUST create it from your param file / snapshots.
   There's a fallback option to assume uniform spacing, but it's usually not what you want.)

Notes:
- We use enclosed mass M(<r) to compute forces:
      a_r(r,t) = -G M(<r,t) / r^2
  then project to cylindrical forces (R,z).
- Spherical host assumed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from galpy.potential import Potential
from galpy.orbit import Orbit

# ==================== CONFIGURATION ====================

# --- AHF input ---
AHF_DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Data/subhalo_evp_3_1_1/AHFout/")
TOTAL_SNAPSHOTS = 280          # check 0..TOTAL_SNAPSHOTS

# --- Host table input ---
HOST_SIGV_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Data/host_halo/host_sigv_subhalo_evp_3_1_1.txt")
# MUST correspond to the number of time blocks in host_sigv.txt
HOST_TIME_FILE  = None   # 1 column, times in Gyr (recommended)
# If you don't have host_times.txt, set HOST_TIME_FILE=None and use fallback below:
ASSUME_UNIFORM_HOST_TIMES_IF_MISSING = False
HOST_TMIN_GYR_FALLBACK = 0.0
HOST_TMAX_GYR_FALLBACK = 14.0  # only used if fallback is enabled

# --- Output ---
OUTPUT_DIR      = "./"
SIMULATION_NAME = "subhalo_evp_3_1_1"
SHOW_FIG        = True

# --- Host halo geometry (in kpc) ---
HOST_CENTER = np.array([1e4, 1e4, 1e4])  # AHF frame
R_VIRIAL    = 974.407                    # virial radius
R_HALO_MAX  = 2000.0                     # outer radius for plotting

# --- galpy unit system (dimensionless; we interpret these physically) ---
# ro_unit = 1 kpc, vo_unit = 220 km/s
RO = 1.0
VO = 220.0  # km/s

# Conversion: 1 km/s * 1 Gyr ≈ 1.022712 kpc
KPC_PER_GYR_PER_KMS = 1.022712
RO_KPC  = 1.0
VO_KMS  = VO
T_UNIT_GYR = RO_KPC / (VO_KMS * KPC_PER_GYR_PER_KMS)  # galpy time unit in Gyr

# --- Initial conditions for predicted orbit (in physical units) ---
R0_KPC   = 0.7 * R_VIRIAL
z0_KPC   = 0.0
phi0_RAD = 0.0

vR0_KMS  = 0.0
vT0_KMS  = 33.1515
vz0_KMS  = 0.0

# --- Integration parameters for predicted orbit ---
T_MAX_GYR = 50.0
N_STEPS   = 2001

# --- Plot settings ---
FIGSIZE_ORBIT_3D = (8, 7)
POINT_SIZE_3D    = 2
LINE_WIDTH_3D    = 1.0

# Physical G in kpc (km/s)^2 / Msun
G_KPC_KMS2_PER_MSUN = 4.30091e-6

# =======================================================


# =========================
# AHF reading (unchanged)
# =========================

def get_valid_snapshots(data_path, total_snapshots):
    valid_snapshots = []
    print("Detecting valid AHF files...")
    for i in range(total_snapshots + 1):
        filepath = f"{data_path}{i}-halos"
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        valid_snapshots.append(i)
            except Exception:
                continue
    print(f"Found {len(valid_snapshots)} non-empty snapshot files out of {total_snapshots + 1} total")
    return valid_snapshots


def load_subhalo_centers_from_ahf(data_path, snapshot_list, host_center):
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
            Xc = float(tokens[5])
            Yc = float(tokens[6])
            Zc = float(tokens[7])

            centers.append([Xc, Yc, Zc])
            snapshots.append(snap)
            orbital_dist.append(np.linalg.norm(np.array([Xc, Yc, Zc]) - host_center))

    return np.array(centers), np.array(snapshots), np.array(orbital_dist)


# =========================
# Plot helpers (unchanged)
# =========================

def plot_sphere(ax, center, radius, alpha=0.15, **kwargs):
    cx, cy, cz = center
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 9)

    xs = cx + radius * np.outer(np.cos(u), np.sin(v))
    ys = cy + radius * np.outer(np.sin(u), np.sin(v))
    zs = cz + radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(xs, ys, zs, alpha=alpha, rstride=1, cstride=1, **kwargs)


def set_equal_3d(ax):
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


# ==========================================
# TABULATED multi-time spherical host potential
# ==========================================

def _split_multitime_blocks_from_r(r):
    """
    Given an r array that is stacked as [r0..rN, r0..rN, ...],
    find block size Nr and number of blocks Nt.
    """
    r = np.asarray(r)
    # indices where r decreases -> boundary between blocks
    resets = np.where(np.diff(r) < 0)[0]
    if len(resets) == 0:
        Nr = len(r)
        Nt = 1
        return Nr, Nt

    Nr = resets[0] + 1
    if len(r) % Nr != 0:
        raise ValueError(f"host_sigv.txt length {len(r)} is not divisible by inferred Nr={Nr}")
    Nt = len(r) // Nr
    return Nr, Nt


def load_host_sigv_multitime(host_sigv_file, host_time_file=None):
    """
    Load host_sigv.txt and return:
      r_grid (Nr,)
      t_grid_gyr (Nt,)
      Menc_grid (Nt, Nr)  [Msun]
    """
    data = np.loadtxt(host_sigv_file)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("host_sigv.txt must have at least 3 columns: r, rho, Menc (and optionally sigv).")

    r_all   = data[:, 0]
    M_all   = data[:, 2]

    Nr, Nt = _split_multitime_blocks_from_r(r_all)
    r_grid = r_all[:Nr].copy()
    M_grid = M_all.reshape(Nt, Nr).copy()

    # Ensure r_grid is strictly increasing
    if not np.all(np.diff(r_grid) > 0):
        raise ValueError("Radius grid in host_sigv.txt is not strictly increasing within a block.")

    # Load times
    if host_time_file is not None and os.path.exists(host_time_file):
        t_grid = np.loadtxt(host_time_file).astype(float)
        if t_grid.ndim != 1:
            t_grid = t_grid.reshape(-1)
        if len(t_grid) != Nt:
            raise ValueError(f"host_times.txt has {len(t_grid)} times, but host_sigv.txt implies Nt={Nt} blocks.")
        # Ensure times increasing
        if not np.all(np.diff(t_grid) >= 0):
            raise ValueError("host_times.txt must be non-decreasing.")
        t_grid_gyr = t_grid
    else:
        if not ASSUME_UNIFORM_HOST_TIMES_IF_MISSING:
            raise FileNotFoundError(
                f"Missing host time file '{host_time_file}'.\n"
                f"Create host_times.txt with Nt={Nt} rows (times in Gyr), or set:\n"
                f"  ASSUME_UNIFORM_HOST_TIMES_IF_MISSING = True\n"
            )
        t_grid_gyr = np.linspace(HOST_TMIN_GYR_FALLBACK, HOST_TMAX_GYR_FALLBACK, Nt)

    return r_grid, t_grid_gyr, M_grid


def _interp1d_linear(x, xp, fp):
    """Vectorized linear interp (np.interp wrapper) for 1D."""
    return np.interp(x, xp, fp)


def _interp_Menc_rt(r_query, t_query, r_grid, t_grid, M_grid):
    """
    Bilinear interpolation for Menc(r,t) using only NumPy.

    Inputs:
      r_query: array (...)
      t_query: array (...) same shape as r_query
      r_grid: (Nr,)
      t_grid: (Nt,)
      M_grid: (Nt, Nr)
    Returns:
      Menc: array (...) Msun
    """
    r_query = np.asarray(r_query, dtype=float)
    t_query = np.asarray(t_query, dtype=float)
    if r_query.shape != t_query.shape:
        raise ValueError("r_query and t_query must have the same shape.")

    Nr = r_grid.size
    Nt = t_grid.size

    # --- radius handling (we will clamp and do a soft inner extrapolation) ---
    rmin = r_grid[0]
    rmax = r_grid[-1]

    r_clamp = np.clip(r_query, rmin, rmax)

    # --- time handling (clamp to table range) ---
    tmin = t_grid[0]
    tmax = t_grid[-1]
    t_clamp = np.clip(t_query, tmin, tmax)

    # For each query time, find neighboring time indices
    # idx_hi is first index where t_grid[idx_hi] >= t
    idx_hi = np.searchsorted(t_grid, t_clamp, side="left")
    idx_hi = np.clip(idx_hi, 0, Nt - 1)
    idx_lo = np.maximum(idx_hi - 1, 0)

    t_lo = t_grid[idx_lo]
    t_hi = t_grid[idx_hi]

    # Avoid division by zero when t_lo == t_hi (at edges or repeated times)
    denom = (t_hi - t_lo)
    w = np.zeros_like(t_clamp)
    mask = denom > 0
    w[mask] = (t_clamp[mask] - t_lo[mask]) / denom[mask]  # in [0,1]

    # Interpolate in r for both time slices
    # We need fp arrays for each query; simplest is loop over unique indices (fast enough for orbit steps)
    M_lo = np.empty_like(r_clamp)
    M_hi = np.empty_like(r_clamp)

    for it in np.unique(idx_lo):
        m = idx_lo == it
        M_lo[m] = _interp1d_linear(r_clamp[m], r_grid, M_grid[it, :])

    for it in np.unique(idx_hi):
        m = idx_hi == it
        M_hi[m] = _interp1d_linear(r_clamp[m], r_grid, M_grid[it, :])

    M = (1.0 - w) * M_lo + w * M_hi

    # r > rmax: use Mtot(t) (total mass at outermost bin, time-interpolated)
    mask_hi_r = r_query > rmax
    if np.any(mask_hi_r):
        # total mass at rmax for each query time
        Mtot_lo = M_grid[idx_lo[mask_hi_r], -1]
        Mtot_hi = M_grid[idx_hi[mask_hi_r], -1]
        w_hi = w[mask_hi_r]
        M[mask_hi_r] = (1.0 - w_hi) * Mtot_lo + w_hi * Mtot_hi

    # r < rmin: assume constant-density core scaling M ~ r^3 using M(rmin,t)
    mask_lo_r = r_query < rmin
    if np.any(mask_lo_r):
        Mrmin_lo = M_grid[idx_lo[mask_lo_r], 0]
        Mrmin_hi = M_grid[idx_hi[mask_lo_r], 0]
        w_lo = w[mask_lo_r]
        Mrmin = (1.0 - w_lo) * Mrmin_lo + w_lo * Mrmin_hi
        M[mask_lo_r] = Mrmin * (r_query[mask_lo_r] / rmin) ** 3

    return M


class HostTabulatedSphericalPotential(Potential):
    """
    galpy Potential that computes forces from a tabulated enclosed mass M(<r,t).

    Forces are spherical:
      a_r = -G M(<r,t)/r^2   [ (km/s)^2 / kpc ]
    Project to cylindrical:
      a_R = a_r * R/r
      a_z = a_r * z/r

    galpy expects _Rforce/_zforce in INTERNAL units:
      force_internal = a_physical / (vo^2/ro)

    Time handling:
      galpy passes t in its own time units.
      We convert to Gyr using t_gyr = t * t_unit_gyr.
    """
    def __init__(self, r_grid_kpc, t_grid_gyr, M_grid_msun,
                 ro=1.0, vo=220.0, t_unit_gyr=1.0):
        Potential.__init__(self, amp=1.0, ro=ro, vo=vo)
        self.r_grid = np.asarray(r_grid_kpc, dtype=float)
        self.t_grid = np.asarray(t_grid_gyr, dtype=float)
        self.M_grid = np.asarray(M_grid_msun, dtype=float)
        self.t_unit_gyr = float(t_unit_gyr)

        if self.M_grid.shape != (self.t_grid.size, self.r_grid.size):
            raise ValueError("M_grid shape must be (Nt, Nr) matching (t_grid, r_grid).")

        self._force_unit = (self._vo**2 / self._ro)  # (km/s)^2/kpc in galpy internal units

    def _Menc(self, r_kpc, t_gyr):
        return _interp_Menc_rt(r_kpc, t_gyr, self.r_grid, self.t_grid, self.M_grid)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        R = np.asarray(R)
        z = np.asarray(z)

        # Convert to physical kpc
        R_kpc = R * self._ro
        z_kpc = z * self._ro
        r_kpc = np.sqrt(R_kpc**2 + z_kpc**2)

        # Convert galpy time to Gyr
        t_gyr = np.asarray(t) * self.t_unit_gyr
        # broadcast to same shape
        t_gyr = np.broadcast_to(t_gyr, r_kpc.shape)

        # Avoid division by zero at center
        eps = 1e-12
        r_safe = np.maximum(r_kpc, eps)

        Menc = self._Menc(r_safe, t_gyr)  # Msun
        a_r = -G_KPC_KMS2_PER_MSUN * Menc / (r_safe**2)  # (km/s)^2 / kpc
        a_R = a_r * (R_kpc / r_safe)                     # (km/s)^2 / kpc

        return a_R / self._force_unit

    def _zforce(self, R, z, phi=0.0, t=0.0):
        R = np.asarray(R)
        z = np.asarray(z)

        R_kpc = R * self._ro
        z_kpc = z * self._ro
        r_kpc = np.sqrt(R_kpc**2 + z_kpc**2)

        t_gyr = np.asarray(t) * self.t_unit_gyr
        t_gyr = np.broadcast_to(t_gyr, r_kpc.shape)

        eps = 1e-12
        r_safe = np.maximum(r_kpc, eps)

        Menc = self._Menc(r_safe, t_gyr)
        a_r = -G_KPC_KMS2_PER_MSUN * Menc / (r_safe**2)
        a_z = a_r * (z_kpc / r_safe)

        return a_z / self._force_unit


# ==========================================
# Predicted orbit pieces (mostly unchanged)
# ==========================================

def make_host_potential():
    """
    Create the multi-time host halo potential from host_sigv.txt (+ host_times.txt).
    Returns a galpy-compatible Potential.
    """
    r_grid, t_grid_gyr, M_grid = load_host_sigv_multitime(HOST_SIGV_FILE, HOST_TIME_FILE)

    print("\nLoaded host potential table:")
    print(f"  Nr = {len(r_grid)} radial bins: r in [{r_grid[0]:.4g}, {r_grid[-1]:.4g}] kpc")
    print(f"  Nt = {len(t_grid_gyr)} time slices: t in [{t_grid_gyr[0]:.4g}, {t_grid_gyr[-1]:.4g}] Gyr")

    pot = HostTabulatedSphericalPotential(
        r_grid_kpc=r_grid,
        t_grid_gyr=t_grid_gyr,
        M_grid_msun=M_grid,
        ro=RO,
        vo=VO,
        t_unit_gyr=T_UNIT_GYR,  # converts galpy time -> Gyr inside the potential
    )
    return pot


def make_predicted_orbit():
    """
    Create a galpy Orbit object for the predicted subhalo COM in host-centered frame.
    """
    R0_g  = R0_KPC / RO_KPC
    z0_g  = z0_KPC / RO_KPC
    vR0_g = vR0_KMS / VO_KMS
    vT0_g = vT0_KMS / VO_KMS
    vz0_g = vz0_KMS / VO_KMS
    phi0  = phi0_RAD

    o = Orbit(vxvv=[R0_g, vR0_g, vT0_g, z0_g, vz0_g, phi0], ro=RO, vo=VO)
    return o


def integrate_predicted_orbit(o, pot):
    """
    Integrate predicted orbit for T_MAX_GYR (physical) in galpy time units.
    """
    tmax_galpy = T_MAX_GYR / T_UNIT_GYR
    t_galpy = np.linspace(0.0, tmax_galpy, N_STEPS)
    o.integrate(t_galpy, pot)
    t_Gyr = t_galpy * T_UNIT_GYR
    return t_galpy, t_Gyr


def compute_predicted_orbit_positions():
    """
    Integrate orbit in host-centered frame, then shift to AHF coordinates.
    """
    pot = make_host_potential()
    orb = make_predicted_orbit()
    t_galpy, t_Gyr = integrate_predicted_orbit(orb, pot)

    # host-centered positions in kpc
    x_rel = orb.x(t_galpy) * RO_KPC
    y_rel = orb.y(t_galpy) * RO_KPC
    z_rel = orb.z(t_galpy) * RO_KPC

    # shift to AHF absolute frame
    x_abs = HOST_CENTER[0] + x_rel
    y_abs = HOST_CENTER[1] + y_rel
    z_abs = HOST_CENTER[2] + z_rel

    return x_abs, y_abs, z_abs, t_Gyr


# ================================
# MAIN
# ================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) AHF: detect valid snapshots and load measured centers
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

    x_meas = centers[:, 0]
    y_meas = centers[:, 1]
    z_meas = centers[:, 2]

    # 2) Predicted orbit in tabulated host potential
    try:
        x_pred, y_pred, z_pred, t_Gyr = compute_predicted_orbit_positions()
        have_predicted = True
    except Exception as e:
        print(f"WARNING: Could not compute predicted orbit in tabulated host potential: {e}")
        have_predicted = False
        x_pred = y_pred = z_pred = None

    # 3) Plot
    fig = plt.figure(figsize=FIGSIZE_ORBIT_3D)
    ax = fig.add_subplot(111, projection="3d")

    plot_sphere(ax, HOST_CENTER, R_VIRIAL,   color="k", linewidth=0.4, linestyle="-")
    plot_sphere(ax, HOST_CENTER, R_HALO_MAX, color="k", linewidth=0.4, linestyle="--")

    ax.plot(x_meas, y_meas, z_meas,
            linewidth=LINE_WIDTH_3D, color="b", label="Subhalo (AHF)")
    ax.scatter(x_meas, y_meas, z_meas, s=POINT_SIZE_3D, color="b")

    if have_predicted:
        ax.plot(x_pred, y_pred, z_pred,
                color="orange", linewidth=1.2, label="Predicted orbit (tabulated host)")

    ax.scatter(HOST_CENTER[0], HOST_CENTER[1], HOST_CENTER[2],
               s=70, marker="*", color="k", label="Host center")

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_zlabel("z [kpc]")
    ax.set_title(f"3D Subhalo Orbit + Tabulated Host Potential - {SIMULATION_NAME}")

    # Plot range
    rel_meas = centers - HOST_CENTER
    max_meas_radius = np.max(np.linalg.norm(rel_meas, axis=1))

    candidates = [R_HALO_MAX, max_meas_radius, R0_KPC]
    if have_predicted:
        rel_pred = np.vstack([x_pred - HOST_CENTER[0],
                              y_pred - HOST_CENTER[1],
                              z_pred - HOST_CENTER[2]]).T
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

    outpath = os.path.join(OUTPUT_DIR, f"orbit3D_with_HOSTTABU_{SIMULATION_NAME}.png")
    plt.savefig(outpath, dpi=200)
    print(f"\nSaved 3D combined orbit plot to {outpath}")

    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()