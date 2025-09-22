# slab_heat_corrected.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("Heat Transfer in Hollow-Core Concrete Slab (CEM III)")

# -------------------
# Material properties
# -------------------
rho = 2300.0       # kg/m3
cp = 900.0         # J/kgK
k = 1.4            # W/mK
alpha = k / (rho * cp)  # thermal diffusivity m2/s

# Convective heat transfer coefficient at top (natural convection, approx.)
h = 10.0           # W/m2K

# Geometry
L = 0.3   # m thickness
Nx = 61   # nodes through thickness
z = np.linspace(0, L, Nx)
dz = z[1] - z[0]

# Stability for explicit scheme
dt_stability = dz**2 / (2*alpha)
dt = 0.9 * dt_stability
r = alpha * dt / dz**2

# -------------------
# User inputs
# -------------------
st.sidebar.header("Inputs")
bed_temp = st.sidebar.slider("Heated bed temperature (°C)", 20, 80, 60)
ambient_temp = st.sidebar.slider("Ambient temperature (°C)", 0, 40, 20)
sim_hours = st.sidebar.slider("Simulation time (hours)", 1, 100, 24)

st.sidebar.write(f"dt = {dt:.2f} s (stability r={r:.3f})")

# -------------------
# Initial condition
# -------------------
T = np.full(Nx, ambient_temp)
times = []
mid_temps = []

# -------------------
# Equilibrium profile (analytic steady-state with convection top)
# -------------------
# Effective thermal resistance balance:
# k*(T_bed - T_top)/L = h*(T_top - T_air)
T_top_eq = (k*bed_temp + h*L*ambient_temp) / (k + h*L)
T_eq_profile = bed_temp + (T_top_eq - bed_temp) * (z/L)
T_eq_mid = T_eq_profile[Nx//2]

# -------------------
# Time stepping
# -------------------
max_time = sim_hours * 3600
n_steps = int(max_time / dt)

for n in range(n_steps):
    Tn = T.copy()
    # Bottom BC: fixed to bed temperature
    T[0] = bed_temp
    # Top BC: convection -> ghost node method
    Tghost = Tn[-2] - 2*dz*h/k * (Tn[-1] - ambient_temp)
    T[-1] = Tn[-1] + r * (Tghost - 2*Tn[-1] + Tn[-2])
    # Interior nodes
    T[1:-1] = Tn[1:-1] + r * (Tn[2:] - 2*Tn[1:-1] + Tn[:-2])

    # Store mid-slab temperature
    if n % 50 == 0:  # store every ~50 steps to save memory
        times.append(n*dt/3600)  # hours
        mid_temps.append(T[Nx//2])

# -------------------
# Plot depth profile
# -------------------
fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(z*1000, T, label=f"After {sim_hours} h")
ax1.plot(z*1000, T_eq_profile, "k--", label="Equilibrium")
ax1.set_xlabel("Depth (mm)")
ax1.set_ylabel("Temperature (°C)")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# -------------------
# Plot mid-slab temperature vs time
# -------------------
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(times, mid_temps, label="Mid-slab temperature")
ax2.axhline(T_eq_mid, color="k", linestyle="--", label="Equilibrium mid-slab")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Temperature (°C)")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# -------------------
# Key results
# -------------------
st.subheader("Results")
st.write(f"Equilibrium top surface temperature ≈ {T_top_eq:.1f} °C")
st.write(f"Equilibrium mid-slab temperature ≈ {T_eq_mid:.1f} °C")
st.write(f"Initial mid-slab temperature = {ambient_temp:.1f} °C")

st.info("Certainty ~70% (based on standard thermal properties and 1D conduction model).\n"
        "Assumptions:\n"
        "- Only conduction + convection at top\n"
        "- Constant material properties (CEM III concrete)\n"
        "- No hydration heat, no radiation\n"
        "- 1D through thickness only")
