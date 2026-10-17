#!/usr/bin/env python3
"""
GNSS Positioning Solver
=======================

Simulates a 24-satellite GPS-like constellation (Walker 24/6/1, 55 deg
inclination, 26,560 km orbit radius) and computes a receiver position from
noisy pseudoranges using weighted iterative least squares (Gauss-Newton),
solving for position AND receiver clock bias.

What it demonstrates:
  * WGS84 geodetic <-> ECEF and ENU frame maths
  * Satellite visibility with an elevation mask
  * Dilution of Precision (GDOP / PDOP / HDOP / VDOP / TDOP) from geometry
  * Monte-Carlo validation: the measured position error must match the
    theoretical covariance  sqrt(trace(inv(H' W H)))
  * 24-hour DOP / satellite-count timeline and a sky plot

Usage:
    python gnss_positioning.py
    python gnss_positioning.py --lat 30.0444 --lon 31.2357 --mask 15 --sigma 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = 299_792_458.0                 # speed of light, m/s
MU = 3.986_004_418e14             # m^3/s^2
OMEGA_E = 7.292_115_0e-5          # rad/s
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257_223_563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


# ------------------------------------------------------------ geometry
def geodetic_to_ecef(lat_deg, lon_deg, h=0.0):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - WGS84_E2) + h) * np.sin(lat)])


def enu_matrix(lat_deg, lon_deg):
    """Rows are the East, North, Up unit vectors expressed in ECEF."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl, sp, cp = np.sin(lon), np.cos(lon), np.sin(lat), np.cos(lat)
    return np.array([[-sl, cl, 0.0],
                     [-sp * cl, -sp * sl, cp],
                     [cp * cl, cp * sl, sp]])


def constellation_ecef(t, n_planes=6, per_plane=4, inc_deg=55.0, a=26_560e3, phasing=1):
    """ECEF positions (N x 3, m) of a circular Walker constellation at time t (s)."""
    n = np.sqrt(MU / a**3)
    inc = np.radians(inc_deg)
    total = n_planes * per_plane
    pos = []
    for p in range(n_planes):
        raan = 2.0 * np.pi * p / n_planes
        for s in range(per_plane):
            u = 2.0 * np.pi * s / per_plane + 2.0 * np.pi * phasing * p / total + n * t
            xo, yo = a * np.cos(u), a * np.sin(u)
            pos.append([xo * np.cos(raan) - yo * np.cos(inc) * np.sin(raan),
                        xo * np.sin(raan) + yo * np.cos(inc) * np.cos(raan),
                        yo * np.sin(inc)])
    eci = np.array(pos)
    th = OMEGA_E * t
    c, s = np.cos(th), np.sin(th)
    return np.column_stack([c * eci[:, 0] + s * eci[:, 1], -s * eci[:, 0] + c * eci[:, 1], eci[:, 2]])


def az_el(rx, sats, rot_enu):
    enu = (rot_enu @ (sats - rx).T).T
    el = np.degrees(np.arcsin(enu[:, 2] / np.linalg.norm(enu, axis=1)))
    az = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360.0
    return az, el


# ---------------------------------------------------------- estimation
def design_matrix(rx, sats):
    d = sats - rx
    rho = np.linalg.norm(d, axis=1)
    return np.hstack([-d / rho[:, None], np.ones((len(sats), 1))]), rho


def solve_position(sats, pseudoranges, sigmas, iters=15):
    """Weighted Gauss-Newton. Returns state [x, y, z, c*dt] (m) and covariance."""
    x = np.zeros(4)
    w = np.diag(1.0 / sigmas**2)
    for _ in range(iters):
        h, rho = design_matrix(x[:3], sats)
        resid = pseudoranges - (rho + x[3])
        cov = np.linalg.inv(h.T @ w @ h)
        dx = cov @ h.T @ w @ resid
        x = x + dx
        if np.linalg.norm(dx[:3]) < 1e-4:
            break
    h, _ = design_matrix(x[:3], sats)
    return x, np.linalg.inv(h.T @ w @ h)


def dop(rx, sats, rot_enu):
    h, _ = design_matrix(rx, sats)
    q = np.linalg.inv(h.T @ h)
    q_enu = rot_enu @ q[:3, :3] @ rot_enu.T
    return {"GDOP": np.sqrt(np.trace(q)), "PDOP": np.sqrt(np.trace(q[:3, :3])),
            "HDOP": np.sqrt(q_enu[0, 0] + q_enu[1, 1]), "VDOP": np.sqrt(q_enu[2, 2]),
            "TDOP": np.sqrt(q[3, 3])}


def elevation_sigma(el_deg, sigma0):
    """Simple elevation-dependent pseudorange noise model (m)."""
    return sigma0 / np.sin(np.radians(np.maximum(el_deg, 5.0))) ** 0.7


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, default=30.0444, help="receiver latitude, deg")
    ap.add_argument("--lon", type=float, default=31.2357, help="receiver longitude, deg")
    ap.add_argument("--alt", type=float, default=23.0, help="receiver altitude, m")
    ap.add_argument("--mask", type=float, default=10.0, help="elevation mask, deg")
    ap.add_argument("--sigma", type=float, default=2.0, help="zenith pseudorange noise (1-sigma), m")
    ap.add_argument("--trials", type=int, default=3000, help="Monte-Carlo trials")
    ap.add_argument("--epoch", type=float, default=0.0, help="snapshot time, s from start")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rx_true = geodetic_to_ecef(args.lat, args.lon, args.alt)
    rot = enu_matrix(args.lat, args.lon)
    clock_bias_m = 1.0e-3 * C        # a deliberately large 1 ms receiver clock error

    # ---------------------------------------------------- snapshot
    sats_all = constellation_ecef(args.epoch)
    az, el = az_el(rx_true, sats_all, rot)
    vis = el >= args.mask
    sats, az_v, el_v = sats_all[vis], az[vis], el[vis]
    print(f"Epoch {args.epoch:.0f} s: {vis.sum()} satellites above {args.mask:.0f} deg mask")
    if vis.sum() < 4:
        raise SystemExit("Fewer than 4 satellites visible: try another --epoch or a lower --mask.")

    d = dop(rx_true, sats, rot)
    print("DOP: " + "  ".join(f"{k} {v:.2f}" for k, v in d.items()))

    sig = elevation_sigma(el_v, args.sigma)
    rho_true = np.linalg.norm(sats - rx_true, axis=1) + clock_bias_m
    _, cov = solve_position(sats, rho_true, sig)           # noiseless sanity check
    x_clean, _ = solve_position(sats, rho_true, sig)
    print(f"Noise-free solve: position error {np.linalg.norm(x_clean[:3] - rx_true) * 1e3:.4f} mm, "
          f"clock bias error {abs(x_clean[3] - clock_bias_m) * 1e3:.4f} mm")

    # ------------------------------------------- Monte-Carlo validation
    errs = np.zeros((args.trials, 3))
    for i in range(args.trials):
        x, _ = solve_position(sats, rho_true + rng.normal(0.0, sig), sig)
        errs[i] = rot @ (x[:3] - rx_true)
    rms_h = np.sqrt(np.mean(errs[:, 0] ** 2 + errs[:, 1] ** 2))
    rms_v = np.sqrt(np.mean(errs[:, 2] ** 2))
    rms_3d = np.sqrt(np.mean(np.sum(errs**2, axis=1)))
    cov_enu = rot @ cov[:3, :3] @ rot.T
    th_h = np.sqrt(cov_enu[0, 0] + cov_enu[1, 1])
    th_v = np.sqrt(cov_enu[2, 2])
    th_3d = np.sqrt(np.trace(cov[:3, :3]))
    print(f"\nMonte-Carlo ({args.trials} trials) vs theoretical covariance:")
    print(f"  horizontal RMS: {rms_h:.3f} m   (theory {th_h:.3f} m)")
    print(f"  vertical   RMS: {rms_v:.3f} m   (theory {th_v:.3f} m)")
    print(f"  3-D        RMS: {rms_3d:.3f} m   (theory {th_3d:.3f} m, "
          f"difference {abs(rms_3d - th_3d) / th_3d * 100:.1f} %)")

    # ------------------------------------------------ 24 h timeline
    times = np.arange(0.0, 86_400.0, 300.0)
    n_vis, hd, vd, pd = [], [], [], []
    for t in times:
        s_all = constellation_ecef(t)
        _, e_all = az_el(rx_true, s_all, rot)
        m = e_all >= args.mask
        n_vis.append(m.sum())
        if m.sum() >= 4:
            q = dop(rx_true, s_all[m], rot)
            hd.append(q["HDOP"]); vd.append(q["VDOP"]); pd.append(q["PDOP"])
        else:
            hd.append(np.nan); vd.append(np.nan); pd.append(np.nan)
    print(f"\n24 h: satellites visible {min(n_vis)}-{max(n_vis)}, "
          f"median PDOP {np.nanmedian(pd):.2f}, worst PDOP {np.nanmax(pd):.2f}")

    # ------------------------------------------------------ figures
    fig = plt.figure(figsize=(16, 4.8))
    ax0 = fig.add_subplot(1, 3, 1, projection="polar")
    ax0.set_theta_zero_location("N")
    ax0.set_theta_direction(-1)
    ax0.scatter(np.radians(az_v), 90.0 - el_v, s=90, c="#1f77b4", zorder=3)
    idx = np.where(vis)[0]
    for a_, e_, k in zip(az_v, el_v, idx):
        ax0.annotate(f"G{k + 1:02d}", (np.radians(a_), 90.0 - e_), ha="center", va="center",
                     color="white", fontsize=7, zorder=4)
    ax0.set_rlim(0, 90)
    ax0.set_rticks([0, 30, 60, 90])
    ax0.set_yticklabels(["90", "60", "30", "0"])
    ax0.set_title(f"Sky plot ({vis.sum()} satellites)", pad=14)

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.scatter(errs[:, 0], errs[:, 1], s=4, alpha=0.35, color="#2ca02c")
    ax1.add_patch(plt.Circle((0, 0), 2.0 * th_h, fill=False, color="#d62728", ls="--", label="2x theoretical 1-sigma"))
    ax1.set(xlabel="East error (m)", ylabel="North error (m)", title="Horizontal error scatter", aspect="equal")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 3)
    hrs = times / 3600.0
    ax2.plot(hrs, pd, label="PDOP", color="#9467bd")
    ax2.plot(hrs, hd, label="HDOP", color="#1f77b4")
    ax2.plot(hrs, vd, label="VDOP", color="#ff7f0e")
    ax2.set(xlabel="Time (h)", ylabel="DOP", title="DOP and satellite count over 24 h")
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.step(hrs, n_vis, where="post", color="#7f7f7f", alpha=0.5, label="visible sats")
    ax3.set_ylabel("Visible satellites")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    fig.tight_layout()
    path = args.out / "gnss_positioning.png"
    fig.savefig(path, dpi=140)
    print(f"Figure saved: {path}")


if __name__ == "__main__":
    main()
