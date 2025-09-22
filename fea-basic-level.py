import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title("Heat Conduction in CEM III Concrete Slab")

# --- Slab and material properties (CEM III concrete, representative) ---
thickness = 0.3    # m
rho = 2350         # kg/m3
cp = 900           # J/kgK
k = 1.8            # W/mK
alpha = k / (rho * cp)

# --- Discretization ---
Nx = 61
dz = thickness / (Nx - 1)
dt = 5.0  # seconds, chosen for stability
max_time = 8 * 3600  # simulate up to 8 hours

# --- User sliders ---
bed_temp = st.slider("Heated Bed Temperature (°C)", 20, 80, 50)
ambient_temp = st.slider("Ambient (Top) Temperature (°C)", 5, 25, 20)

# --- Initial condition ---
T_init = ambient_temp
T = np.full(Nx, T_init)

# --- Tracking ---
target_temp = bed_temp - 1.0  # within 1°C of bed
time_to_target = None
mid_temps = []   # mid-slab temperature vs. time
times = []       # time points
snapshots = {}
snapshot_times = [600, 1800, 3600, 7200, 14400, max_time]

# --- Time stepping ---
for n in range(int(max_time / dt)):
    Tn = T.copy()
    Tn[0] = bed_temp
    Tn[-1] = ambient_temp
    for i in range(1, Nx - 1):
        T[i] = Tn[i] + alpha * dt / dz**2 * (Tn[i+1] - 2*Tn[i] + Tn[i-1])
    T[0] = bed_temp
    T[-1] = ambient_temp

    t_now = (n + 1) * dt
    mid_temps.append(T[Nx//2])
    times.append(t_now / 3600)  # hours

    if t_now in snapshot_times:
        snapshots[t_now] = T.copy()

    if time_to_target is None and T[Nx//2] >= target_temp:
        time_to_target = t_now

# --- Plot depth profiles ---
z = np.linspace(0, thickness, Nx)
fig1, ax1 = plt.subplots()
for t, profile in snapshots.items():
    ax1.plot(z, profile, label=f"{int(t/60)} min")
ax1.set_xlabel("Depth (m) from heated bed")
ax1.set_ylabel("Temperature (°C)")
ax1.legend()
st.pyplot(fig1)

# --- Plot mid-slab temperature vs time ---
fig2, ax2 = plt.subplots()
ax2.plot(times, mid_temps, label="Mid-slab temperature")
ax2.axhline(bed_temp, color="r", linestyle="--", label="Bed temperature")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Temperature (°C)")
ax2.set_ylim(min(ambient_temp, min(mid_temps)) - 2,
             max(bed_temp, max(mid_temps)) + 2)
ax2.legend()
st.pyplot(fig2)

# --- Results ---
if time_to_target:
    hours = int(time_to_target // 3600)
    minutes = int((time_to_target % 3600) // 60)
    st.success(f"Mid-slab warms to within 1°C of bed in ~{hours}h {minutes}min.")
else:
    st.warning("Mid-slab did not reach near-bed temperature within 8 hours.")

# --- Certainty estimate ---
st.info("Simulation certainty: ~65%.\n"
        "- Based on typical CEM III concrete properties.\n"
        "- 1D conduction only (no hollow cores, no hydration heat, no convection/radiation).\n"
        "- Good for trends, not exact curing prediction.")
