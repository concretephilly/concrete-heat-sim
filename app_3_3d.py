# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Hollow-Core Slab Heat (CEM III defaults)")

st.title("Hollow-Core Slab — 2D Cross-section Heat Simulation (CEM III defaults)")

# -------------------------
# Defaults (CEM III-like)
# -------------------------
DEFAULT_DENSITY = 2300.0        # kg/m3
DEFAULT_CP = 900.0              # J/kgK
DEFAULT_K = 2.5                 # W/mK
DEFAULT_CEMENT = 350.0          # kg cement / m3 concrete (typical)
DEFAULT_QTOT = 250e3            # J/kg cement (total hydration heat)
DEFAULT_K_RATE_H = 0.05         # 1/hour reaction rate (user input)
DEFAULT_OUTSIDE_TEMP = 20.0     # °C
DEFAULT_H = 10.0                # W/m2K (natural convection)

# -------------------------
# Sidebar: user inputs
# -------------------------
st.sidebar.header("Geometry")
length = st.sidebar.number_input("Length (m)", 1.0, 20.0, 4.0, 0.1,
                                 help="Slab length along x (used for true aspect ratio).")
width = st.sidebar.number_input("Width (m)", 0.5, 5.0, 1.2, 0.05,
                                help="Slab cross-section width (y).")
height = st.sidebar.number_input("Height (m)", 0.08, 0.5, 0.20, 0.01,
                                 help="Slab thickness (z).")

st.sidebar.header("Hollow-c cores")
n_voids = st.sidebar.number_input("Number of circular hollow cores", 1, 8, 5, 1)
void_radius = st.sidebar.number_input("Core radius (m)", 0.02, 0.25, 0.06, 0.005,
                                      help="Radius of each circular void in the cross-section (m).")

st.sidebar.header("Material (CEM III defaults)")
density = st.sidebar.number_input("Concrete density ρ (kg/m³)", 1800.0, 2800.0, DEFAULT_DENSITY, 10.0)
cp = st.sidebar.number_input("Concrete specific heat c_p (J/kg·K)", 600.0, 1200.0, DEFAULT_CP, 10.0)
k_conc = st.sidebar.number_input("Concrete thermal conductivity k (W/m·K)", 0.5, 4.0, DEFAULT_K, 0.1)

st.sidebar.header("Hydration")
cement_content = st.sidebar.number_input("Cement content (kg cement / m³ concrete)", 150.0, 600.0, DEFAULT_CEMENT, 10.0)
Q_total = st.sidebar.number_input("Total heat Q_total (J/kg cement)", 50e3, 600e3, DEFAULT_QTOT, 1000.0,
                                  help="Total heat release per kg of cement (J/kg).")
k_rate_h = st.sidebar.number_input("Reaction rate k (1/hour)", 0.001, 1.0, DEFAULT_K_RATE_H, 0.001,
                                   help="Controls speed of hydration (larger = faster).")

st.sidebar.header("Environment & Simulation")
outside_temp = st.sidebar.number_input("Outside air temperature (°C)", -20.0, 40.0, DEFAULT_OUTSIDE_TEMP, 0.5)
h_conv = st.sidebar.number_input("Convection h (W/m²·K)", 0.5, 100.0, DEFAULT_H, 0.1,
                                 help="Surface heat transfer coefficient to air.")
sim_hours = st.sidebar.number_input("Simulation duration (hours)", 1, 200, 72, 1)
dt_seconds = st.sidebar.number_input("Timestep (s)", 1.0, 600.0, 60.0, 1.0,
                                    help="Solver timestep in seconds (smaller = more accurate & slower).")

st.sidebar.header("Resolution / performance")
res_preset = st.sidebar.selectbox("Simulation resolution", ["Low", "Medium", "High"], index=1,
                                  help="Lower resolution = faster. Medium is good for demos.")
if res_preset == "Low":
    nx, ny, nz = 60, 30, 12
elif res_preset == "Medium":
    nx, ny, nz = 120, 60, 20
else:
    nx, ny, nz = 180, 90, 28

# Cap resolution so app doesn't explode
nx = int(min(nx, 240))
ny = int(min(ny, 120))
nz = int(min(nz, 48))

# -------------------------
# Derived quantities
# -------------------------
dx = length / nx
dy = width / ny
dz = height / nz
total_seconds = float(sim_hours) * 3600.0
n_steps = max(1, int(total_seconds / dt_seconds))

# Air properties for hollow cores (approx)
rho_air = 1.225
cp_air = 1005.0
k_air = 0.025

# Thermal diffusivity arrays
alpha_conc = k_conc / (density * cp)
alpha_air = k_air / (rho_air * cp_air)

# Convert hydration constants
# k_rate_h is per hour; convert to per second for dQ/dt = Q_total * k_s * exp(-k_s * t)
k_rate_s = k_rate_h / 3600.0
Q_total_per_m3 = Q_total * cement_content  # J per m^3 total (assuming Q_total is J/kg cement)

# -------------------------
# Build grid and mask (vectorized)
# -------------------------
x = np.linspace(0, length, nx)
y = np.linspace(0, width, ny)
z = np.linspace(0, height, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")  # shape (nx, ny, nz)

# mask True=concrete, False=void
mask = np.ones((nx, ny, nz), dtype=bool)
centers_y = np.linspace(width/(n_voids+1), width - width/(n_voids+1), int(n_voids))
centers_z = np.full_like(centers_y, height / 2.0)
# vectorized mask: for each center, mark if inside radius
for cy, cz in zip(centers_y, centers_z):
    r2 = (Y - cy) ** 2 + (Z - cz) ** 2
    mask &= (r2 >= void_radius ** 2)

# per-voxel properties arrays
k_arr = np.where(mask, k_conc, k_air)
rho_arr = np.where(mask, density, rho_air)
cp_arr = np.where(mask, cp, cp_air)
alpha_arr = k_arr / (rho_arr * cp_arr)  # diffusivity per voxel

# -------------------------
# Initial temperature field
# -------------------------
T0 = np.full((nx, ny, nz), outside_temp, dtype=float)
# Initialize concrete slightly warmer than outside to mimic casting (user can change outside_temp)
T0[mask] = outside_temp + 5.0

# -------------------------
# Simulation: vectorized using numpy slicing (no python triple loops)
# -------------------------
save_every = max(1, n_steps // 120)  # store ~120 snapshots max
snapshots = []
T = T0.copy()
time_seconds = 0.0

st.info("Running simulation — this may take some seconds depending on resolution. Progress shown below.")
progress = st.progress(0)

for step in range(n_steps):
    Tn = T.copy()

    # compute laplacian for interior points (vectorized)
    lap = np.zeros_like(Tn)
    lap[1:-1, 1:-1, 1:-1] = (
        (Tn[2:, 1:-1, 1:-1] - 2.0 * Tn[1:-1, 1:-1, 1:-1] + Tn[:-2, 1:-1, 1:-1]) / dx**2 +
        (Tn[1:-1, 2:, 1:-1] - 2.0 * Tn[1:-1, 1:-1, 1:-1] + Tn[1:-1, :-2, 1:-1]) / dy**2 +
        (Tn[1:-1, 1:-1, 2:] - 2.0 * Tn[1:-1, 1:-1, 1:-1] + Tn[1:-1, 1:-1, :-2]) / dz**2
    )

    # diffusion update (vectorized) only for interior voxels
    T[1:-1, 1:-1, 1:-1] = (
        Tn[1:-1, 1:-1, 1:-1]
        + alpha_arr[1:-1, 1:-1, 1:-1] * dt_seconds * lap[1:-1, 1:-1, 1:-1]
    )

    # hydration volumetric heat rate (J/m3/s) using single-exponential kinetics
    # dQ/dt = Q_total_per_m3 * k_s * exp(-k_s * t)
    q_vol = Q_total_per_m3 * k_rate_s * np.exp(-k_rate_s * time_seconds)
    # temperature rise from volumetric heat: dT = q_vol / (rho * cp) * dt
    # apply only to concrete voxels
    dT_hyd = (q_vol / (rho_arr * cp_arr)) * dt_seconds
    T += dT_hyd * mask  # add to all concrete voxels (air voxels get 0)

    # convection at exposed faces (vectorized)
    # temperature change per voxel due to one outside face: dT = - h * dt / (rho*cp*dx) * (T - T_inf)
    # X faces (i=0 and i=nx-1), area = dy*dz, volume = dx*dy*dz -> factor reduces to h*dt/(rho*cp*dx)
    face_coeff_x = (h_conv * dt_seconds) / (rho_arr * cp_arr * dx)
    face_coeff_y = (h_conv * dt_seconds) / (rho_arr * cp_arr * dy)
    face_coeff_z = (h_conv * dt_seconds) / (rho_arr * cp_arr * dz)

    # apply to X faces where mask True (concrete surface only)
    # i=0
    mask_face = mask[0, :, :]
    if np.any(mask_face):
        Tn_face = Tn[0, :, :].copy()
        T[0, :, :][mask_face] = Tn_face[mask_face] - face_coeff_x[0, :, :][mask_face] * (Tn_face[mask_face] - outside_temp)
    # i=nx-1
    mask_face = mask[-1, :, :]
    if np.any(mask_face):
        Tn_face = Tn[-1, :, :].copy()
        T[-1, :, :][mask_face] = Tn_face[mask_face] - face_coeff_x[-1, :, :][mask_face] * (Tn_face[mask_face] - outside_temp)

    # Y faces j=0 and j=ny-1
    mask_face = mask[:, 0, :]
    if np.any(mask_face):
        Tn_face = Tn[:, 0, :].copy()
        T[:, 0, :][mask_face] = Tn_face[mask_face] - face_coeff_y[:, 0, :][mask_face] * (Tn_face[mask_face] - outside_temp)
    mask_face = mask[:, -1, :]
    if np.any(mask_face):
        Tn_face = Tn[:, -1, :].copy()
        T[:, -1, :][mask_face] = Tn_face[mask_face] - face_coeff_y[:, -1, :][mask_face] * (Tn_face[mask_face] - outside_temp)

    # Z faces k=0 and k=nz-1
    mask_face = mask[:, :, 0]
    if np.any(mask_face):
        Tn_face = Tn[:, :, 0].copy()
        T[:, :, 0][mask_face] = Tn_face[mask_face] - face_coeff_z[:, :, 0][mask_face] * (Tn_face[mask_face] - outside_temp)
    mask_face = mask[:, :, -1]
    if np.any(mask_face):
        Tn_face = Tn[:, :, -1].copy()
        T[:, :, -1][mask_face] = Tn_face[mask_face] - face_coeff_z[:, :, -1][mask_face] * (Tn_face[mask_face] - outside_temp)

    # For voids (air) — update by diffusion using alpha_air (already in alpha_arr)
    # We already updated interior voxels above; ensure voids use alpha_air by alpha_arr values.

    # clamp physically: don't go below outside temp by a lot
    T = np.maximum(T, outside_temp)

    time_seconds += dt_seconds

    # save snapshots periodically
    if step % save_every == 0 or step == n_steps - 1:
        snapshots.append((time_seconds / 3600.0, T.copy()))

    # progress bar
    if step % max(1, n_steps // 100) == 0:
        progress.progress(min(100, int(step / n_steps * 100)))

progress.progress(100)
st.success("Simulation complete.")

# -------------------------
# Prepare 2D cross-section display (higher effective resolution)
# -------------------------
st.subheader("2D Cross-section (primary view)")

# precompute global color scale from concrete voxels across all snapshots
valid_mins, valid_maxs = [], []
for _, Tsnap in snapshots:
    vals = Tsnap[mask]
    if vals.size:
        valid_mins.append(vals.min())
        valid_maxs.append(vals.max())
if valid_mins:
    global_min = float(min(valid_mins))
    global_max = float(max(valid_maxs))
else:
    global_min = outside_temp
    global_max = outside_temp + 1.0
if np.isclose(global_min, global_max):
    global_max = global_min + 1.0

# slider for time index
frame_idx = st.slider("Snapshot index", 0, len(snapshots) - 1, 0)
time_h, Tcurr = snapshots[frame_idx]

# slice options (default to cross-section through slab center width x height)
st.write("Choose slice axis and index (center default).")
axis = st.radio("Slice axis", ["X (lengthwise)", "Y (width)", "Z (height / thickness)"], index=1)

if axis.startswith("X"):
    slice_idx = nx // 2
    data = Tcurr[slice_idx, :, :].T  # shape (nz, ny) => display z (height) vertical, y horizontal
    x_axis = y  # width
    y_axis = z  # height
    xlabel = "Width (m)"
    ylabel = "Height (m)"
elif axis.startswith("Y"):
    slice_idx = ny // 2
    data = Tcurr[:, slice_idx, :].T  # shape (nz, nx) -> x along length
    x_axis = x  # length
    y_axis = z  # height
    xlabel = "Length (m)"
    ylabel = "Height (m)"
else:
    slice_idx = nz // 2
    data = Tcurr[:, :, slice_idx].T  # shape (ny, nx) -> length x width
    x_axis = x
    y_axis = y
    xlabel = "Length (m)"
    ylabel = "Width (m)"

# heatmap with Jet colormap (blue cold -> red hot)
fig = go.Figure(data=go.Heatmap(
    z=data,
    x=x_axis,
    y=y_axis,
    colorscale="Jet",
    zmin=global_min,
    zmax=global_max,
    colorbar=dict(title="Temperature (°C)")
))
fig.update_layout(title=f"Cross-section at t = {time_h:.2f} h",
                  xaxis_title=xlabel,
                  yaxis_title=ylabel,
                  yaxis=dict(scaleanchor="x", scaleratio=1),
                  height=650)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Temperature vs time plot (avg & max) + equalization marker
# -------------------------
st.subheader("Temperature vs time (concrete voxels)")

times = [t for t, _ in snapshots]
avg_temps = [np.mean(Tsnap[mask]) for _, Tsnap in snapshots]
max_temps = [np.max(Tsnap[mask]) for _, Tsnap in snapshots]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=times, y=avg_temps, mode="lines", name="Average Temp", line=dict(color="blue")))
fig2.add_trace(go.Scatter(x=times, y=max_temps, mode="lines", name="Max Temp", line=dict(color="red")))

# equalization (average concrete within tol of outside temp)
tolerance = 0.5
eq_time = None
for t, avg in zip(times, avg_temps):
    if abs(avg - outside_temp) <= tolerance:
        eq_time = t
        break

if eq_time is not None:
    fig2.add_vline(x=eq_time, line=dict(color="green", dash="dash"), annotation_text="Equalized", annotation_position="top left")
    fig2.add_trace(go.Scatter(
        x=[eq_time], y=[np.interp(eq_time, times, max_temps)],
        mode="markers+text", text=["Equalized"], textposition="bottom center",
        marker=dict(color="green", size=10, symbol="x"), showlegend=False
    ))

fig2.update_layout(xaxis_title="Time (hours)", yaxis_title="Temperature (°C)",
                   title="Concrete temperature evolution", height=350)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("**Equalization estimate**")
if eq_time is not None:
    st.success(f"Average concrete temperature equals outside air (±{tolerance}°C) after about **{eq_time:.1f} hours**.")
else:
    st.info("Average concrete temperature did not equalize with outside air within simulated time.")

# -------------------------
# Short notes + export
# -------------------------
st.markdown("---")
st.markdown("**Notes / assumptions**")
st.markdown(
    "- Solver: explicit finite-difference, vectorized. Stability requires small dt at high resolution.\n"
    "- Hollow cores are modeled as air (separate thermal properties) — they remain near ambient and gain heat mainly by conduction through the concrete walls.\n"
    "- Hydration: single-exponential kinetics. Volumetric heat rate derived from Q_total (J/kg cement) × cement_content (kg/m³).\n"
    "- Surface convection uses `h` (W/m²K) applied to exposed concrete surface voxels.\n"
)
if st.button("Export time–temperature data (CSV)"):
    import io, pandas as pd
    df = pd.DataFrame({
        "time_h": times,
        "avg_temp": avg_temps,
        "max_temp": max_temps
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button("Download CSV", buf, file_name="temp_history.csv", mime="text/csv")

