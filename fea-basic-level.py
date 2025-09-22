# slab_heat_app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("1D Heat Conduction — CEM III Concrete Slab (fixed)")

# --------------------
# Material & geometry
# --------------------
thickness = 0.3       # m (300 mm)
width = 1.2           # m (not used in 1D)
length = 2.0          # m (not used in 1D)

# Representative CEM III values (you asked for as-close-as-possible)
rho = 2350.0          # kg/m^3
cp = 900.0            # J/(kg*K)
k = 1.8               # W/(m*K)
alpha = k / (rho * cp)  # m^2/s

# --------------------
# Numerical discretization
# --------------------
Nx = 61                       # nodes through thickness (~5 mm spacing)
z = np.linspace(0, thickness, Nx)
dz = z[1] - z[0]

# Stability limit for explicit FTCS: dt <= dz^2/(2*alpha)
dt_stability = dz**2 / (2.0 * alpha)
# Choose dt = 0.9 * stability limit (safe) but not too small
dt = max(0.05, 0.9 * dt_stability)   # don't go below 0.05s to avoid extremely long runs
# Note: dt_stability will typically be ~1-2 seconds for these parameters; we use 90% of that.
r = alpha * dt / dz**2

# --------------------
# User controls
# --------------------
st.sidebar.header("Simulation inputs")
bed_temp = st.sidebar.slider("Heated bed temperature (°C)", 20, 80, 69)
ambient_temp = st.sidebar.slider("Ambient / top temperature (°C)", -5, 40, 24)
sim_hours = st.sidebar.slider("Simulate for (hours)", 1, 100, 89)
show_equilibrium = st.sidebar.checkbox("Show linear equilibrium profile", True)

# Show key numeric diagnostics
st.sidebar.write(f"dz = {dz*1000:.1f} mm")
st.sidebar.write(f"dt (used) = {dt:.3f} s")
st.sidebar.write(f"alpha = {alpha:.2e} m²/s")
st.sidebar.write(f"stability r = {r:.4f}  (should be <= 0.5)")

# --------------------
# Initial / boundary conditions
# --------------------
T0 = np.full(Nx, ambient_temp, dtype=float)  # initial temperature
T = T0.copy()

# Boundary condition indices:
# z=0 -> bottom (in contact with heated bed)
# z=thickness -> top (ambient)
# We will enforce T[0] = bed_temp, T[-1] = ambient_temp each step

# Equilibrium (steady-state linear) profile when bottom & top fixed:
T_equil = bed_temp + (ambient_temp - bed_temp) * (z / thickness)
T_equil_mid = T_equil[Nx // 2]

# --------------------
# Time setup
# --------------------
max_time_s = sim_hours * 3600.0
n_steps = int(np.ceil(max_time_s / dt))
# For memory/plotting reasons we will record every step but downsample when plotting.
times = np.empty(n_steps, dtype=float)        # hours
mid_temps = np.empty(n_steps, dtype=float)    # mid-depth temp

# Precompute coefficient for the update
coeff = alpha * dt / dz**2  # equal to r

# Vector indices for interior nodes
interior = slice(1, Nx - 1)

# --------------------
# Time-stepping (vectorized inner update)
# --------------------
# We'll use the explicit formula:
# T_new[i] = T[i] + coeff * (T[i+1] - 2*T[i] + T[i-1])
# Implemented vectorized: T[1:-1] = T[1:-1] + coeff * (T[2:] - 2*T[1:-1] + T[:-2])
# Save mid-slab temp each step

T[0] = bed_temp
T[-1] = ambient_temp

for n in range(n_steps):
    # store time
    t_now = (n + 1) * dt
    times[n] = t_now / 3600.0  # convert to hours

    # compute next interior temperatures
    Tn = T.copy()
    # Enforce BCs on Tn (just to be safe)
    Tn[0] = bed_temp
    Tn[-1] = ambient_temp

    # Vectorized update
    T[interior] = Tn[interior] + coeff * (Tn[2:] - 2.0 * Tn[1:-1] + Tn[:-2])

    # Re-apply BCs (ensure exact)
    T[0] = bed_temp
    T[-1] = ambient_temp

    # record mid-slab temperature
    mid_temps[n] = T[Nx // 2]

# --------------------
# Post-processing & metrics
# --------------------
# Final profile after sim_hours
T_final = T.copy()

# Equilibrium mid-slab
equil_mid = T_equil_mid

# Compute how close mid_temp gets to equilibrium and time to reach 90% of the change
initial_mid = T0[Nx // 2]
delta_equil = equil_mid - initial_mid

# If delta_equil is 0 (ambient==bed) avoid division by zero
if abs(delta_equil) < 1e-6:
    frac_mid = (mid_temps - initial_mid) * 0.0
    time_to_90pct = None
else:
    frac_mid = (mid_temps - initial_mid) / delta_equil  # fraction of approach to equilibrium
    # find first index where fraction >= 0.9
    idx_90 = np.where(frac_mid >= 0.9)[0]
    time_to_90pct = (idx_90[0] * dt / 3600.0) if idx_90.size > 0 else None

# Simple time-constant estimate (diffusion timescale):
# For 1D slab of half-thickness L = thickness/2, t_diff ~ L^2 / alpha (order-of-magnitude)
L = thickness / 2.0
t_diff_s = L**2 / alpha
t_diff_h = t_diff_s / 3600.0

# 63% estimate: find index closest to 0.63 fraction
idx_63 = np.argmin(np.abs(frac_mid - 0.63)) if abs(delta_equil) >= 1e-6 else None
time_63 = (idx_63 * dt / 3600.0) if idx_63 is not None else None

# --------------------
# Plotting
# --------------------
st.subheader("Results")

# Left: Depth profiles (final and equilibrium)
fig1, ax1 = plt.subplots(figsize=(6, 3.5))
ax1.plot(z * 1000.0, T_final, label=f"After {sim_hours} h (final)")
if show_equilibrium:
    ax1.plot(z * 1000.0, T_equil, "k--", label="Equilibrium (linear)")
ax1.set_xlabel("Depth from bed (mm)")
ax1.set_ylabel("Temperature (°C)")
ax1.grid(alpha=0.2)
ax1.legend()
st.pyplot(fig1)

# Right / below: Mid-slab vs time
# Downsample for plotting (keep every kth point so the plot is responsive)
k = max(1, int(len(times) / 1200))  # aim for <=1200 points
times_plot = times[::k]
mid_plot = mid_temps[::k]

fig2, ax2 = plt.subplots(figsize=(8, 3.5))
ax2.plot(times_plot, mid_plot, label="Mid-slab temperature")
ax2.axhline(equil_mid, color="k", linestyle="--", label="Equilibrium mid-slab")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Temperature (°C)")
ax2.grid(alpha=0.2)

# Mark times if found
if time_63 is not None:
    ax2.axvline(time_63, color="orange", linestyle=":", label=f"~63% at {time_63:.2f} h")
if time_to_90pct is not None:
    ax2.axvline(time_to_90pct, color="green", linestyle="--", label=f"90% at {time_to_90pct:.2f} h")

ax2.set_ylim(min(ambient_temp, min(mid_plot)) - 2, max(bed_temp, max(mid_plot)) + 2)
ax2.legend()
st.pyplot(fig2)

# --------------------
# Numeric outputs / explanations
# --------------------
st.markdown("### Key numbers")
st.write(f"- Equilibrium mid-slab temperature ≈ **{equil_mid:.2f} °C**")
st.write(f"- Initial mid-slab temperature = **{initial_mid:.2f} °C**")
st.write(f"- Diffusion timescale estimate t ~ L²/α = **{t_diff_h:.1f} hours** (order-of-magnitude)")
if time_63 is not None:
    st.write(f"- Approx. time to reach ~63% of equilibrium (numerical) = **{time_63:.2f} h**")
else:
    st.write(f"- 63% time: not applicable (no change expected)")

if time_to_90pct is not None:
    st.success(f"- Time to reach 90% of equilibrium (numerical) = **{time_to_90pct:.2f} h**")
else:
    st.info("- Mid-slab did not reach 90% of equilibrium within the simulated time.")

# Certainty & assumptions (transparent)
st.markdown("### Simulation certainty & assumptions")
st.info(
    "Certainty: **~65%** (trend-level accuracy).\n\n"
    "**Assumptions / simplifications:**\n"
    "- 1D conduction only (no hollow cores, no hydration heat, no convection/radiation).\n"
    "- Constant material properties (no temperature-dependent properties).\n"
    "- Explicit FTCS finite-difference scheme (stability enforced by dt).\n\n"
    "This model is useful for qualitative and semi-quantitative trends, not final structural/curing design."
)

st.caption("If you still see a flat mid-slab line at ambient temperature, please verify the bed temperature differs from ambient (otherwise equilibrium equals ambient).")
