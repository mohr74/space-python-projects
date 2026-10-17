#!/usr/bin/env python3
"""
Orbit Propagator with J2 perturbation
=====================================

Numerically integrates a satellite's motion around the Earth (two-body + J2
oblateness) with a fixed-step RK4 integrator, then validates the result
against analytical theory:

  1. Two-body run  -> specific orbital energy must stay constant.
  2. J2 run        -> the orbital plane (RAAN) must drift at the rate predicted
                      by secular perturbation theory:
                          dOmega/dt = -1.5 * n * J2 * (Re/p)^2 * cos(i)
  3. Sun-synchronous mode (default) picks the inclination so that this drift
     equals +0.9856 deg/day (the Earth's mean motion around the Sun).

Outputs a ground-track / RAAN-drift / altitude figure in ./output.

Usage:
    python orbit_propagator.py                   # 700 km sun-synchronous, 3 days
    python orbit_propagator.py --alt 420 --inc 51.6 --days 5   # ISS-like
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- constants
MU = 398_600.4418            # Earth gravitational parameter, km^3/s^2
RE = 6_378.137               # Earth equatorial radius, km
J2 = 1.082_626_68e-3         # second zonal harmonic
OMEGA_E = 7.292_115_9e-5     # Earth rotation rate, rad/s
SOLAR_YEAR_S = 365.242_189_7 * 86_400.0


# ------------------------------------------------------------ rotations
def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# ------------------------------------------------- orbital element helpers
def elements_to_state(a, e, inc, raan, argp, nu):
    """Classical elements (km, rad) -> ECI position (km) and velocity (km/s)."""
    p = a * (1.0 - e**2)
    r_pf = p / (1.0 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(MU / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    rot = rot_z(raan) @ rot_x(inc) @ rot_z(argp)
    return rot @ r_pf, rot @ v_pf


def state_to_elements(r, v):
    """ECI state -> (a km, e, inclination rad, RAAN rad)."""
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    node = np.cross([0.0, 0.0, 1.0], h)
    evec = ((vn**2 - MU / rn) * r - np.dot(r, v) * v) / MU
    a = -MU / (2.0 * (vn**2 / 2.0 - MU / rn))
    inc = np.arccos(h[2] / np.linalg.norm(h))
    raan = np.arctan2(node[1], node[0]) % (2.0 * np.pi)
    return a, np.linalg.norm(evec), inc, raan


def j2_raan_rate(a, e, inc):
    """Secular RAAN drift due to J2, rad/s."""
    n = np.sqrt(MU / a**3)
    p = a * (1.0 - e**2)
    return -1.5 * n * J2 * (RE / p) ** 2 * np.cos(inc)


def sun_sync_inclination(a, e=0.0):
    """Inclination (rad) whose J2 nodal drift equals the Sun's mean motion."""
    target = 2.0 * np.pi / SOLAR_YEAR_S
    n = np.sqrt(MU / a**3)
    p = a * (1.0 - e**2)
    return np.arccos(-target / (1.5 * n * J2 * (RE / p) ** 2))


# --------------------------------------------------------------- dynamics
def accel(r: np.ndarray, j2: bool = True) -> np.ndarray:
    """Gravitational acceleration (km/s^2): point mass + optional J2."""
    x, y, z = r
    rn = np.linalg.norm(r)
    a = -MU * r / rn**3
    if j2:
        k = 1.5 * J2 * MU * RE**2 / rn**5
        zz = 5.0 * z * z / rn**2
        a = a + k * np.array([x * (zz - 1.0), y * (zz - 1.0), z * (zz - 3.0)])
    return a


def propagate(r0, v0, duration, dt=10.0, j2=True, sample_every=6):
    """Fixed-step RK4. Returns sample times (s) and states (N x 6)."""

    def f(y):
        return np.concatenate([y[3:], accel(y[:3], j2)])

    y = np.concatenate([r0, v0]).astype(float)
    n_steps = int(round(duration / dt))
    times, states = [0.0], [y.copy()]
    for k in range(1, n_steps + 1):
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if k % sample_every == 0:
            times.append(k * dt)
            states.append(y.copy())
    return np.array(times), np.array(states)


def ground_track(times, states):
    """ECI -> sub-satellite (lat, lon) in degrees (spherical Earth)."""
    th = OMEGA_E * times
    c, s = np.cos(th), np.sin(th)
    x = c * states[:, 0] + s * states[:, 1]
    y = -s * states[:, 0] + c * states[:, 1]
    z = states[:, 2]
    lat = np.degrees(np.arcsin(z / np.linalg.norm(states[:, :3], axis=1)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


# --------------------------------------------------------------- analysis
def raan_history(states):
    raan = np.array([state_to_elements(s[:3], s[3:])[3] for s in states])
    return np.unwrap(raan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alt", type=float, default=700.0, help="mean altitude, km")
    ap.add_argument("--ecc", type=float, default=0.0, help="eccentricity")
    ap.add_argument("--inc", type=float, default=None, help="inclination, deg (default: sun-synchronous)")
    ap.add_argument("--days", type=float, default=3.0, help="simulated duration, days")
    ap.add_argument("--dt", type=float, default=10.0, help="RK4 step, s")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    a = RE + args.alt
    inc = np.radians(args.inc) if args.inc is not None else sun_sync_inclination(a, args.ecc)
    r0, v0 = elements_to_state(a, args.ecc, inc, 0.0, 0.0, 0.0)
    period = 2.0 * np.pi * np.sqrt(a**3 / MU)
    duration = args.days * 86_400.0

    print(f"Orbit: a = {a:.1f} km, e = {args.ecc}, i = {np.degrees(inc):.3f} deg, "
          f"period = {period / 60:.2f} min")

    t, y_j2 = propagate(r0, v0, duration, args.dt, j2=True)
    _, y_2b = propagate(r0, v0, duration, args.dt, j2=False)

    # 1) two-body energy conservation (validates the integrator)
    r_n = np.linalg.norm(y_2b[:, :3], axis=1)
    v_n = np.linalg.norm(y_2b[:, 3:], axis=1)
    eps = v_n**2 / 2.0 - MU / r_n
    drift = np.max(np.abs((eps - eps[0]) / eps[0]))
    print(f"[two-body] max relative energy drift over {args.days:g} d: {drift:.2e}")

    # 2) J2 nodal regression vs theory (validates the perturbation model)
    days = t / 86_400.0
    slope_sim = np.polyfit(days, np.degrees(raan_history(y_j2)), 1)[0]
    slope_2b = np.polyfit(days, np.degrees(raan_history(y_2b)), 1)[0]
    slope_th = np.degrees(j2_raan_rate(a, args.ecc, inc)) * 86_400.0
    print(f"[RAAN drift] simulated J2  : {slope_sim:+.4f} deg/day")
    print(f"[RAAN drift] theory        : {slope_th:+.4f} deg/day "
          f"(difference {abs(slope_sim - slope_th) / abs(slope_th) * 100:.2f} %)")
    print(f"[RAAN drift] simulated, no J2: {slope_2b:+.5f} deg/day")
    if args.inc is None:
        print(f"[sun-sync] target = {360.0 / 365.2421897:+.4f} deg/day")

    # ---------------------------------------------------------- figure
    lat, lon = ground_track(t, y_j2)
    n_orb = 3 * period
    m = t <= max(n_orb, 1.0)
    lon_p = lon[m].copy()
    lat_p = lat[m].copy()
    jump = np.where(np.abs(np.diff(lon_p)) > 180.0)[0]
    lon_p[jump] = np.nan
    lat_p[jump] = np.nan

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(lon_p, lat_p, lw=1.2, color="#1f77b4")
    ax[0].plot(lon[0], lat[0], "o", color="#d62728", label="start")
    ax[0].set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude (deg)", ylabel="Latitude (deg)",
              title="Ground track (first 3 orbits)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc="lower left")

    ax[1].plot(days, np.degrees(raan_history(y_j2)), label="with J2", color="#2ca02c")
    ax[1].plot(days, np.degrees(raan_history(y_2b)), label="two-body only", color="#7f7f7f", ls="--")
    ax[1].plot(days, slope_th * days, ":", color="#d62728", label=f"theory ({slope_th:+.3f} deg/day)")
    ax[1].set(xlabel="Time (days)", ylabel="RAAN (deg, unwrapped)", title="Orbital plane drift")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    alt = np.linalg.norm(y_j2[:, :3], axis=1) - RE
    ax[2].plot(days, alt, color="#9467bd")
    ax[2].set(xlabel="Time (days)", ylabel="Altitude above equatorial radius (km)",
              title="Altitude (J2 short-period variation)")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    path = args.out / "orbit_analysis.png"
    fig.savefig(path, dpi=140)
    print(f"Figure saved: {path}")


if __name__ == "__main__":
    main()
