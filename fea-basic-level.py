import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title("Heat Conduction in CEM III Concrete Slab")

# --- Slab and material properties (CEM III concrete, representative) ---
thickness = 0.3    # m
rho = 2350         # kg/m3
cp = 900           # J/kgK
k = 1.8            # W/mK
alpha = k / (rho * cp)   # thermal diffusivity

# --- Discretization ---
Nx = 61
dz = thickness / (Nx - 1)
dt = 5.0  # seconds, chosen for stability

# --- User sliders ---
bed_temp = st.slider("Heated Bed Temperature (°C)", 20, 80, 60)
ambient_temp = st.slider("Ambient (Top) Temperature (°C)", 5, 25, 20)
sim_hours = st.slider("Simulation time (hours)", 1, 100, 24)

# --- Initial condition ---
T_init = ambient_temp
T = np.full(Nx, T_init)

# --- Equilibrium profile (linear) ---
z = np.linspace(0, thickness, Nx)
T_equil = bed_temp + (ambient_temp - bed_temp) * (z / thickness)

# --- Time stepping ---
max_time = sim_hours * 3600
mid_temps = []
times = []

for n in range(int(max_time / dt)):
    Tn = T.copy()
    Tn[0] = bed_temp
    Tn[-1] = ambient_temp
    for i in range(1, Nx - 1):
        T[i] = Tn[i] + alpha * dt / dz**2 * (Tn[i+1] - 2*Tn[i] + Tn[i-1])
    T[0] = bed_temp
    T[-1] = ambient_temp

    # record mid-slab temperature every step
    t_now = (n + 1) * dt
    mid_temps.append(T[Nx//2])
    times.append(t_now / 3600)  # hours

# --- Reduce plotting density (every 60th point ~5 min) ---
times_plot = times[::60]
mid_temps_plot = mid_temps[::60]

# --- Plot depth profile (final vs equilibrium) ---
fig1, ax1 = plt.subplots()
ax1.plot(z, T, "r-", label=f"After {sim_hours} h")
ax1.plot(z, T_equil, "k--", label="Equilibrium (linear)")
ax1.set_xlabel("Depth (m) from heated bed")
ax1.set_ylabel("Temperature (°C)")
ax1.legend()
st.pyplot(fig1)

# --- Plot mid-slab temperature vs time ---
fig2, ax2 = plt.subplots()
ax2.plot(times_plot, mid_temps_plot, label="Mid-slab temperature")
ax2.axhline(T_equil[Nx//2], color="k", linestyle="--", label="Equilibrium mid-slab")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Temperature (°C)")
ax2.set_ylim(min(ambient_temp, min(mid_temps_plot)) - 2,
             max(bed_temp, max(mid_temps_plot)) + 2)
ax2.legend()
st.pyplot(fig2)

# --- Results ---
st.success(f"Equilibrium mid-slab temperature ≈ {T_equil[Nx//2]:.1f} °C")
st.info("Simulation certainty: ~65%.\n"
        "- Based on typical CEM III concrete properties.\n"
        "- 1D conduction only (no hollow cores, no hydration heat, no convection/radiation).\n"
        "- Captures realistic diffusion time scale (~30 h for 300 mm slab).")
