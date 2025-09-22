# slab_heat_verified.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("Verified 1D Heat Conduction — Hollow-Core Slab (CEM III)")

# ------------------------
# Material & geometry
# ------------------------
# Typical hollow-core slab / CEM III-ish values (representative)
rho = 2300.0        # kg/m^3
cp = 900.0          # J/(kg*K)
k = 1.4             # W/(m*K)
alpha = k / (rho * cp)  # m^2/s

h = 10.0            # convective heat transfer coefficient at top (W/m^2/K), natural convection

L = 0.30            # slab thickness (m)
Nx = 61             # number of grid points (gives dz ~ 5 mm)
z = np.linspace(0.0, L, Nx)
dz = z[1] - z[0]

# Stability for explicit FTCS: dt <= dz^2/(2*alpha)
dt_stability = dz**2 / (2.0 * alpha)
dt = 0.9 * dt_stability     # use 90% of stability limit
r = alpha * dt / dz**2

# ------------------------
# UI inputs
# ------------------------
st.sidebar.header("Simulation inputs")
bed_temp = st.sidebar.slider("Heated bed temperature (°C)", 20, 80, 60)
ambient_temp = st.sidebar.slider("Ambient / top temperature (°C)", -5, 40, 24)
sim_hours = st.sidebar.slider("Simulation time (hours)", 1, 100, 24)
store_every_n = st.sidebar.number_input("Store every n steps for plotting (1 = every step)", min_value=1, max_value=1000, value=1)

st.sidebar.write(f"dz = {dz*1000:.1f} mm")
st.sidebar.write(f"dt (used) = {dt:.3f} s")
st.sidebar.write(f"alpha = {alpha:.2e} m^2/s")
st.sidebar.write(f"stability r = {r:.4f}  (must be <= 0.5)")

if r > 0.5:
    st.error("Stability condition violated (r > 0.5). Reduce dz or dt.")
    st.stop()

# ------------------------
# Initial conditions
# ------------------------
T = np.full(Nx, ambient_temp, dtype=float)
T[0] = bed_temp  # bottom node at bed
# top node will be computed via convection algebraic relation each step

# ------------------------
# Analytical equilibrium (linear with convective top)
# ------------------------
# Solve for T_top_eq from k*(T_bed - T_top)/L = h*(T_top - T_air)
# => T_top_eq = (k*T_bed + h*L*T_air) / (k + h*L)
T_top_eq = (k * bed_temp + h * L * ambient_temp) / (k + h * L)
T_eq_profile = bed_temp + (T_top_eq - bed_temp) * (z / L)
T_eq_mid = T_eq_profile[Nx // 2]

# ------------------------
# Time stepping (explicit, vectorized)
# ------------------------
max_time_s = sim_hours * 3600.0
n_steps = int(np.ceil(max_time_s / dt))

# Arrays to store time history (we'll store every store_every_n steps)
times = []
mid_history = []

coeff = alpha * dt / dz**2  # equals r

# Main time loop
for n in range(n_steps):
    Tn = T.copy()
    # Enforce bottom Dirichlet BC on new array immediately
    T[0] = bed_temp

    # Interior update (1 .. Nx-2)
    T[1:-1] = Tn[1:-1] + coeff * (Tn[2:] - 2.0 * Tn[1:-1] + Tn[:-2])

    # Top node: enforce convective flux by algebraic relation derived from:
    #   -k*(T_top - T_{N-2})/dz = h*(T_top - T_air)  (backward diff for derivative)
    # Rearranged:
    #   (k + h*dz)*T_top = k*T_{N-2} + h*dz*T_air
    T_top = (k * T[ -2] + h * dz * ambient_temp) / (k + h * dz)
    T[-1] = T_top

    # record
    if (n % store_every_n) == 0:
        times.append((n+1) * dt / 3600.0)  # hours
        mid_history.append(T[Nx // 2])

# Final profile
T_final = T.copy()

# ------------------------
# Post-processing metrics
# ------------------------
initial_mid = ambient_temp
equil_mid = T_eq_mid
delta_equil = equil_mid - initial_mid

if abs(delta_equil) < 1e-9:
    frac = np.zeros_like(mid_history)
else:
    frac = (np.array(mid_history) - initial_mid) / delta_equil

# find time to 90% of equilibrium (if reached within sim time)
idx90 = np.where(frac >= 0.9)[0]
time_to_90 = (times[idx90[0]] if idx90.size > 0 else None)

# 63% time (approx)
idx63 = np.argmin(np.abs(frac - 0.63)) if delta_equil != 0 else None
time63 = (times[idx63] if idx63 is not None else None)

# diffusion timescale L_half^2/alpha (rule-of-thumb)
L_half = L / 2.0
t_diff_s = L_half**2 / alpha
t_diff_h = t_diff_s / 3600.0

# ------------------------
# Plots
# ------------------------
st.subheader("Depth profile (final) and equilibrium")
fig1, ax1 = plt.subplots()
ax1.plot(z*1000.0, T_final, label=f"Final after {sim_hours} h")
ax1.plot(z*1000.0, T_eq_profile, 'k--', label="Analytic equilibrium")
ax1.set_xlabel("Depth from bed (mm)")
ax1.set_ylabel("Temperature (°C)")
ax1.grid(alpha=0.25)
ax1.legend()
st.pyplot(fig1)

st.subheader("Mid-slab temperature vs time")
fig2, ax2 = plt.subplots()
ax2.plot(times, mid_history, label="Mid-slab (numerical)")
ax2.axhline(equil_mid, color='k', linestyle='--', label="Equilibrium mid-slab")
if time63 is not None:
    ax2.axvline(time63, color='orange', linestyle=':', label=f"~63% @ {time63:.2f} h")
if time_to_90 is not None:
    ax2.axvline(time_to_90, color='green', linestyle='--', label=f"90% @ {time_to_90:.2f} h")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Temperature (°C)")
ax2.grid(alpha=0.25)
ax2.legend()
st.pyplot(fig2)

# ------------------------
# Numeric outputs & checks
# ------------------------
st.markdown("### Numeric diagnostics")
st.write(f"- dz = {dz:.4f} m, dt = {dt:.4f} s, r = {r:.6f}  (r must be ≤ 0.5)")
st.write(f"- alpha = {alpha:.2e} m²/s")
st.write(f"- Analytic equilibrium top temp = {T_top_eq:.3f} °C")
st.write(f"- Analytic equilibrium mid-slab temp = {equil_mid:.3f} °C")
st.write(f"- Numerical final mid-slab temp = {T_final[Nx//2]:.3f} °C")

if time_to_90 is not None:
    st.success(f"Time to reach 90% of equilibrium (mid-slab) ≈ {time_to_90:.2f} hours")
else:
    st.info("Mid-slab did not reach 90% of equilibrium within the simulated time.")

if time63 is not None:
    st.write(f"Approx 63% time (numerical) ≈ {time63:.2f} hours")

st.write(f"Diffusion timescale estimate (L/2)^2/alpha ≈ {t_diff_h:.2f} hours")

st.markdown(
    "**Notes about the physics and expected behaviour**\n\n"
    "- The **equilibrium** depends strongly on bed temperature (higher bed → higher equilibrium mid-slab & top). You should see the final mid-slab temp increase when you raise the bed temperature.\n"
    "- The **shape** of the transient (how quickly the mid-slab moves toward its equilibrium fraction) depends mainly on geometry and alpha. For a fixed geometry and alpha, the *time to reach a given fraction of equilibrium (e.g., 63% or 90%)* is roughly independent of the bed temperature — this is a property of the linear diffusion equation. That means if you compare two bed temps, the *curves* look similar in shape, but they approach different equilibrium levels. If you inspect absolute values (not normalized fraction), obviously hotter beds produce higher mid-slab temperatures.\n"
    "- If you want to compare *absolute time to reach the bed temperature itself* (not a fraction of equilibrium), note that the mid-slab often never reaches the bed temperature unless ambient ≈ bed (because heat must escape at the top)."
)

st.markdown("If you still see a flat mid-slab line at the ambient value after running this, please tell me the exact values you used for bed_temp, ambient_temp, sim_hours and the 'store every n steps' setting — I will reproduce and debug step-by-step.")
