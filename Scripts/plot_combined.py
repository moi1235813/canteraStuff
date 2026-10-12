"""
Re-plot the ignition-delay sweep and the laminar flame-speed sweep from their
saved text files (no Cantera runs) in one figure with a shared Z axis.

    top:    ignition delay tau_ign (log, solid) + mixing temperature T_mix (dashed, right axis)
    bottom: laminar flame speed S_L (solid)     + flame thickness delta (dashed, right axis)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullFormatter

IGN_FILE = "ignition_sweep.txt"   # T_fuel, Z, T_mix [K], tau_ign [ms]
SL_FILE  = "SL_data.txt"          # T_fuel, Z, S_L [m/s], delta [m]

T_OX   = 900.0                    # K
P_BAR  = 60
COLORS = ["black", "red", "blue"]

# Stoichiometric mixture fraction of pure H2 with air (O2:1, N2:3.76):
# 2 H2 + O2 -> 2 H2O, so 2 mol H2 per mol O2.
M_H2, M_O2, M_N2 = 2.01588, 31.9988, 28.0134
Z_ST = 2 * M_H2 / (2 * M_H2 + M_O2 + 3.76 * M_N2)   # ~0.0285


def read_sweep(path):
    """
    Read 4-column 'T, Z, a, b' rows. Separator and summary lines are skipped.
    Because the sweep scripts open their files in append mode, a file may hold
    several runs; for each (T, Z) the last value written wins.
    Returns {T: array with columns [Z, a, b]}, sorted by T and Z.
    """
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            try:
                T, Z, a, b = map(float, parts)   # 'nan' parses fine
            except ValueError:
                continue
            rows[(T, Z)] = (a, b)

    grouped = {}
    for (T, Z), (a, b) in rows.items():
        grouped.setdefault(T, []).append((Z, a, b))
    return {T: np.array(sorted(v)) for T, v in sorted(grouped.items())}


def add_legend(ax_main, ax_twin, solid_label, dashed_label, loc):
    """One legend (drawn on the twin axis so it sits on top): colours = T_fuel, line style = quantity."""
    color_handles, _ = ax_main.get_legend_handles_labels()
    style_handles = [
        Line2D([], [], color="gray", linestyle="-",  label=solid_label),
        Line2D([], [], color="gray", linestyle="--", label=dashed_label),
    ]
    ax_twin.legend(handles=color_handles + style_handles, loc=loc)


ign = read_sweep(IGN_FILE)   # T_fuel -> [Z, T_mix, tau_ms]
sl  = read_sweep(SL_FILE)    # T_fuel -> [Z, S_L, delta_m]

fig, (ax_ign, ax_sl) = plt.subplots(2, 1, sharex=True, figsize=(8, 9))
ax_ign_T = ax_ign.twinx()
ax_sl_d  = ax_sl.twinx()

# --- top: ignition delay -------------------------------------------------------
for k, (T_fuel, d) in enumerate(ign.items()):
    ax_ign.semilogy(d[:, 0], d[:, 2], "-", color=COLORS[k],
                    label=rf"$T_{{fuel}}$ = {T_fuel:.0f} K")
    ax_ign_T.plot(d[:, 0], d[:, 1], linestyle="dashed", color=COLORS[k])

ax_ign.yaxis.set_major_formatter(ScalarFormatter())
ax_ign.yaxis.set_minor_formatter(NullFormatter())   # label decades only (100, 1000, ...)
ax_ign.set_ylabel(r"$\tau_{ign}$ [ms]")
ax_ign_T.set_ylabel(r"$T_{mix}$ [K]")
ax_ign.set_title(f"Ignition delay vs H2 mass fraction (T_air = {T_OX:.0f} K, p = {P_BAR}bar)")
add_legend(ax_ign, ax_ign_T, r"$\tau_{ign}$", r"$T_{mix}$", loc="upper right")

# --- bottom: laminar flame speed and thickness ------------------------------------
for k, (T_fuel, d) in enumerate(sl.items()):
    ax_sl.plot(d[:, 0], d[:, 1], color=COLORS[k], label=f"{T_fuel:.0f}K")
    ax_sl_d.plot(d[:, 0], d[:, 2] * 1e6, color=COLORS[k], linestyle="dashed")  # m -> um

ax_sl.set_ylabel("S_L [m/s]")
ax_sl_d.set_ylabel("δ [μm]")
ax_sl.set_xlabel("Z")
ax_sl.set_title("Laminar flame speed and flame thickness vs mixture fraction\n"
                f"T_fuel = 300/400/500K T_air={T_OX:.0f}K p={P_BAR}bar")
add_legend(ax_sl, ax_sl_d, r"$S_L$", r"$\delta$", loc="upper right")

# --- shared bits -----------------------------------------------------------------
for ax in (ax_ign, ax_sl):
    ax.axvline(x=Z_ST, color="gray", linestyle="dotted")
    ax.grid()

fig.tight_layout()
plt.show()
