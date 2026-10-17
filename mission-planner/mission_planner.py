#!/usr/bin/env python3
"""
Mission Delta-V Planner
=======================

Orbital-maneuver calculator for early mission design:

  * Hohmann and bi-elliptic transfers (and the r2/r1 ratios where bi-elliptic wins:
    11.94 in the rb -> infinity limit, 15.58 for any rb > r2)
  * LEO -> GEO transfer from a launch-site inclination, with the plane change
    split optimally between the perigee and apogee burns
  * Earth -> Mars Hohmann transfer: v_infinity, trans-Mars injection from LEO,
    flight time and the launch phase angle
  * Tsiolkovsky rocket equation: propellant needed for each Isp class

Figures are written to ./output.

Usage:
    python mission_planner.py
    python mission_planner.py --leo-alt 400 --launch-lat 5.2   # e.g. equatorial-ish site
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MU_EARTH = 398_600.4418          # km^3/s^2
MU_SUN = 1.327_124_400_18e11     # km^3/s^2
RE = 6_378.137                   # km
GEO_R = 42_164.17                # km (geostationary radius)
AU = 149_597_870.7               # km
R_MARS = 1.523_679 * AU          # km (mean heliocentric distance, circular approx.)
G0 = 9.806_65e-3                 # km/s^2


# ----------------------------------------------------------- transfers
def hohmann(mu, r1, r2):
    """Return (dv1, dv2, transfer_time_s). Circular -> circular, coplanar."""
    dv1 = np.sqrt(mu / r1) * (np.sqrt(2.0 * r2 / (r1 + r2)) - 1.0)
    dv2 = np.sqrt(mu / r2) * (1.0 - np.sqrt(2.0 * r1 / (r1 + r2)))
    t = np.pi * np.sqrt(((r1 + r2) / 2.0) ** 3 / mu)
    return dv1, dv2, t


def bielliptic(mu, r1, r2, rb):
    """Return (dv_total, transfer_time_s). Intermediate apoapsis rb >= r2."""
    a1, a2 = (r1 + rb) / 2.0, (r2 + rb) / 2.0
    dv1 = np.sqrt(2 * mu / r1 - mu / a1) - np.sqrt(mu / r1)
    dv2 = abs(np.sqrt(2 * mu / rb - mu / a2) - np.sqrt(2 * mu / rb - mu / a1))
    dv3 = abs(np.sqrt(2 * mu / r2 - mu / a2) - np.sqrt(mu / r2))
    t = np.pi * (np.sqrt(a1**3 / mu) + np.sqrt(a2**3 / mu))
    return dv1 + dv2 + dv3, t


def plane_change_dv(v, d_inc):
    """Pure plane change at speed v (km/s) through d_inc (rad)."""
    return 2.0 * v * np.sin(d_inc / 2.0)


def combined_burn(v_a, v_b, angle):
    """Burn changing speed v_a -> v_b while rotating the velocity by `angle`."""
    return np.sqrt(v_a**2 + v_b**2 - 2.0 * v_a * v_b * np.cos(angle))


def leo_to_geo(r_leo, d_inc, split_deg):
    """Total dv when `split_deg` of the plane change is done at perigee."""
    a_t = (r_leo + GEO_R) / 2.0
    v_leo = np.sqrt(MU_EARTH / r_leo)
    v_geo = np.sqrt(MU_EARTH / GEO_R)
    v_p = np.sqrt(MU_EARTH * (2.0 / r_leo - 1.0 / a_t))
    v_a = np.sqrt(MU_EARTH * (2.0 / GEO_R - 1.0 / a_t))
    alpha = np.radians(split_deg)
    return combined_burn(v_leo, v_p, alpha) + combined_burn(v_a, v_geo, d_inc - alpha)


# ------------------------------------------------------ rocket equation
def mass_ratio(dv, isp):
    """m0 / mf for a given dv (km/s) and Isp (s)."""
    return np.exp(dv / (isp * G0))


# ------------------------------------------------------------- reports
def report_geo(args):
    r_leo = RE + args.leo_alt
    d_inc = np.radians(abs(args.launch_lat))
    v_leo = np.sqrt(MU_EARTH / r_leo)
    v_geo = np.sqrt(MU_EARTH / GEO_R)

    dv1, dv2, t = hohmann(MU_EARTH, r_leo, GEO_R)
    print(f"=== LEO ({args.leo_alt:.0f} km) -> GEO, launch latitude {args.launch_lat:.1f} deg ===")
    print(f"Coplanar Hohmann: dv1 = {dv1:.3f} km/s, dv2 = {dv2:.3f} km/s, "
          f"total = {dv1 + dv2:.3f} km/s, time = {t / 3600:.2f} h")

    pure = plane_change_dv(v_geo, d_inc)
    print(f"Plane change {np.degrees(d_inc):.1f} deg as a separate burn at GEO: +{pure:.3f} km/s "
          f"-> total {dv1 + dv2 + pure:.3f} km/s")

    splits = np.linspace(0.0, np.degrees(d_inc), 400)
    totals = np.array([leo_to_geo(r_leo, d_inc, s) for s in splits])
    k = int(np.argmin(totals))
    print(f"Optimal split: {splits[k]:.2f} deg at perigee + {np.degrees(d_inc) - splits[k]:.2f} deg at apogee"
          f" -> total {totals[k]:.3f} km/s (saves {dv1 + dv2 + pure - totals[k]:.3f} km/s)")
    return splits, totals, totals[k], v_leo


def report_bielliptic(args):
    r1 = RE + args.leo_alt
    ratios = np.linspace(2.0, 40.0, 300)
    hoh = np.array([sum(hohmann(MU_EARTH, r1, r1 * q)[:2]) / np.sqrt(MU_EARTH / r1) for q in ratios])
    # rb -> infinity limit approximated by a very distant apoapsis
    bie = np.array([bielliptic(MU_EARTH, r1, r1 * q, r1 * 1e4)[0] / np.sqrt(MU_EARTH / r1) for q in ratios])
    cross = ratios[np.argmax(bie < hoh)]
    print("\n=== Hohmann vs bi-elliptic (rb -> infinity) ===")
    print(f"Bi-elliptic (rb -> infinity) becomes cheaper above r2/r1 ~ {cross:.1f} (theory: 11.94). "
          f"Above 15.58 any bi-elliptic with rb > r2 beats Hohmann.")
    print(f"LEO->GEO ratio is {GEO_R / r1:.2f}, so Hohmann wins there.")
    return ratios, hoh, bie


def report_mars(args):
    r_e = AU
    dv1, dv2, t = hohmann(MU_SUN, r_e, R_MARS)   # heliocentric legs = v_infinity values
    r_leo = RE + args.leo_alt
    v_leo = np.sqrt(MU_EARTH / r_leo)
    tmi = np.sqrt(dv1**2 + 2.0 * MU_EARTH / r_leo) - v_leo
    n_mars = np.sqrt(MU_SUN / R_MARS**3)
    lead = np.degrees(np.pi - n_mars * t)
    n_e, n_m = np.sqrt(MU_SUN / r_e**3), n_mars
    synodic = 2 * np.pi / (n_e - n_m)
    print("\n=== Earth -> Mars Hohmann ===")
    print(f"v_infinity at Earth = {dv1:.3f} km/s, at Mars = {dv2:.3f} km/s")
    print(f"Trans-Mars injection from {args.leo_alt:.0f} km LEO: {tmi:.3f} km/s")
    print(f"Flight time = {t / 86400:.1f} days; Mars must lead Earth by {lead:.1f} deg at departure")
    print(f"Launch window repeats every {abs(synodic) / 86400:.0f} days (synodic period)")
    return tmi


def report_rocket(dv_geo, dv_mars):
    print("\n=== Tsiolkovsky rocket equation ===")
    print(f"{'Engine class':<28}{'Isp (s)':>8}{'GEO prop. %':>14}{'TMI prop. %':>14}")
    for name, isp in [("Hydrazine monoprop", 230), ("Bipropellant (hypergolic)", 320),
                      ("LOX/LH2", 450), ("Ion thruster", 3000)]:
        pf_geo = (1.0 - 1.0 / mass_ratio(dv_geo, isp)) * 100
        pf_mars = (1.0 - 1.0 / mass_ratio(dv_mars, isp)) * 100
        print(f"{name:<28}{isp:>8}{pf_geo:>13.1f}%{pf_mars:>13.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leo-alt", type=float, default=300.0, help="parking orbit altitude, km")
    ap.add_argument("--launch-lat", type=float, default=28.5, help="launch site latitude = initial inclination, deg")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    splits, totals, best, _ = report_geo(args)
    ratios, hoh, bie = report_bielliptic(args)
    tmi = report_mars(args)
    report_rocket(best, tmi)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(splits, totals, color="#1f77b4")
    k = int(np.argmin(totals))
    ax[0].plot(splits[k], totals[k], "o", color="#d62728", label=f"optimum: {splits[k]:.1f} deg at perigee")
    ax[0].set(xlabel="Plane change done at perigee burn (deg)", ylabel="Total delta-v (km/s)",
              title="LEO -> GEO: where to do the plane change")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(ratios, hoh, label="Hohmann", color="#2ca02c")
    ax[1].plot(ratios, bie, label="Bi-elliptic (rb -> inf)", color="#9467bd")
    ax[1].axvline(11.94, color="#7f7f7f", ls=":", label="11.94: bi-elliptic (rb -> inf) starts to win")
    ax[1].axvline(15.58, color="#7f7f7f", ls="--", label="15.58: any rb > r2 wins")
    ax[1].set(xlabel="r2 / r1", ylabel="delta-v / circular speed at r1", title="Hohmann vs bi-elliptic")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    fig.tight_layout()
    path = args.out / "mission_planner.png"
    fig.savefig(path, dpi=140)
    print(f"\nFigure saved: {path}")


if __name__ == "__main__":
    main()
