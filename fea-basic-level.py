import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title("Heat Conduction in CEM III Concrete Slab")

# Slab and material properties (CEM III concrete, representative)
thickness = 0.3    # m
rho = 2350         # kg/m3
cp = 900           # J/kgK
k = 1.8            # W/mK
alpha = k / (rho * cp)

# Discretization
Nx = 61
dz = thickness / (Nx - 1)
dt = 5.0  # seconds, chosen for stability
max_time = 8 * 3600  # simulate up to 8 hours

# User sliders
bed_temp = st.slider("Heated Bed Temperature (°C)", 20, 80, 50)
ambient_temp = st.slider("Ambient (Top) Temperature (°C)", 5, 25, 20)

# Initial condition
T_init = ambient_temp
T = np.full(Nx, T_init)

# Track time to near-bed temperature
target_temp = bed_temp - 1.0  # within 1°C of bed
time_to_target = None

# Save snapshots for plotting
snapshots = {}
snapshot_times = [600, 1800, 3600, 7200, 14400, max_time]  # 10min, 30min, 1h, 2h, 4h, 8h

for n in range(int(max_time / dt)):
    Tn = T.copy()
    Tn[0] = bed_temp
    Tn[-1] = ambient_temp
    for i in range(1, Nx - 1):
        T[i] = Tn[i] + alpha * dt / dz**2 * (Tn[i+1] - 2*Tn[i] + Tn[i-1])
    T[0] = bed_temp
    T[-1] = ambient_temp

    # record snapshots
    t_now = (n+1) * dt
    if t_now in snapshot_times:
        snapshots[t_now] = T.copy()

    # check if mid-depth reached target
    if time_to_target is None and T[Nx//2] >= target_temp:
        time_to_target = t_now

# Plot snapshots
z = np.linspace(0, thickness, Nx)
fig, ax = plt.subplots()
for t, profile in snapshots.items():
    ax.plot(z, profile, label=f"{int(t/60)} min")
ax.set_xlabel("Depth (m) from heated bed")
ax.set_ylabel("Temperature (°C)")
ax.legend()
st.pyplot(fig)

# Show results
if time_to_target:
    hours = int(time_to_target // 3600)
    minutes = int((time_to_target % 3600) // 60)
    st.success(f"Mid-slab warms to within 1°C of bed in ~{hours}h {minutes}min.")
else:
    st.warning("Mid-slab did not reach near-bed temperature within 8 hours.")

# Certainty estimate
st.info("Simulation certainty: ~65%.\n"
        "- Good for relative trends.\n"
        "- Simplifies geometry (no hollow cores), ignores hydration heat, convection, radiation.\n"
        "- Based on typical CEM III concrete properties.")
