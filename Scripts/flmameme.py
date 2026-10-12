from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from si import compute_adiabatic_T
from matplotlib.lines import Line2D



# Simulation parameters

#gas = ct.Solution('Mevel2017.yaml')
#gas = ct.Solution('example_data/ammonia-CO-H2-Alzueta-2023.yaml')

#plt.clf()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()



temps = [300, 400, 500]
fractions = np.geomspace(0.008, 0.16, 200)

air = "O2:1,N2:3.76"

colors = ["black", "red", "blue"]

gas = ct.Solution('output2.yaml')
gas.set_equivalence_ratio(1.0, fuel="H2", oxidizer=air)
Z_st = gas.Y[gas.species_index('H2')]

Sl_max = []
Z_at_max = []
T_list = []

delta_min = []
Z_at_minD = []


for i, temp in enumerate(temps):
    data = []
    for fraction in fractions:
        p = 6e6 # pressure [Pa]

        Tfuel = temp
        Tox = 900

        width = 0.005  # m
        loglevel = 2  # amount of diagnostic output (0 to 8)

        #gas = ct.Solution('h2o2.yaml')
        gas = ct.Solution('output2.yaml')
        gas.set_mixture_fraction(fraction, fuel="H2", oxidizer=air)

        T_mix, X_target = compute_adiabatic_T(gas, T_oxidiser=Tox, P_pa=p, T_fuel=Tfuel)
        gas.TPX = T_mix, p, X_target

        # Set up flame object
        f = ct.FreeFlame(gas, width=width)
        f.set_refine_criteria(ratio=3, slope=0.1, curve=0.2, prune=0.05)

        f.transport_model = 'multicomponent'
        f.solve(loglevel=loglevel, auto=True)

        flame_speed = f.velocity[0]
        dTdx = np.gradient(f.T, f.grid)
        flameThic = (f.T[-1]-f.T[0]) / (np.max(dTdx))


        with open("SL_data.txt", "a") as file:
            file.write(f"{temp},{fraction},{flame_speed},{flameThic}\n")
        data.append([fraction, flame_speed, flameThic])

    data = np.array(data)
    ax1.plot(fractions, data[:, 1], color=colors[i], label=f"{temp}K")
    ax2.plot(fractions, data[:, 2], color=colors[i], linestyle="dashed")
    idx = np.argmax(data[:, 1])          # index of the peak S_L
    Sl_max.append(data[idx, 1])
    Z_at_max.append(fractions[idx])      # Z where the peak occurs
    idx_d = np.argmin(data[:, 2])        # index of the smallest δ
    delta_min.append(data[idx_d, 2])
    Z_at_minD.append(fractions[idx_d])
    T_list.append(temp)

with open("SL_data.txt", "a", encoding="utf-8") as file:
    file.write("////////////////////////////////////////////////////////////////////////////////////////////////////////\n")
    for i in range(len(Sl_max)):
        file.write(f"T = {T_list[i]}K | max S_L = {Sl_max[i]:.3f} m/s @ Z = {Z_at_max[i]:.3f}, min δ = {delta_min[i]*1e6:.3f} μm @ Z = {Z_at_minD[i]:.3f}\n")


ax1.axvline(x=Z_st, color="gray", linestyle="dotted")

plt.title(f"Laminar flame speed and flame thickness vs mixture fraction\nT_fuel = 300/400/500K T_air=900K p=60bar")
ax1.set_ylabel('S_L [m/s]')
ax1.set(xlabel="Z")
ax2.set_ylabel('δ [μm]')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

style_handles = [
    Line2D([], [], color='gray', linestyle='-',  label=r'$S_L$'),
    Line2D([], [], color='gray', linestyle='--', label=r'$\delta$'),
]

color_handles, color_labels = ax1.get_legend_handles_labels()  # 300K, 400K, 500K
ax2.legend(handles=color_handles + style_handles, loc='upper right')

ax1.grid()

plt.tight_layout()
plt.show()