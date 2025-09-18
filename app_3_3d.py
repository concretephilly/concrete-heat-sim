import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Concrete Heat Simulation", layout="wide")

st.title("Concrete Hollow-Core Slab Heat Simulation")

# -------------------------
# Sidebar: User Parameters
# -------------------------
st.sidebar.header("Slab & Environment Settings")

length = st.sidebar.number_input("Slab length (m)", 4.0, 20.0, 8.0, 0.5,
    help="Total span of the slab in meters.")
width = st.sidebar.number_input("Slab width (m)", 0.5, 3.0, 1.2, 0.1,
    help="Full width of the slab cross-section.")
height = st.sidebar.number_input("Slab height (m)", 0.1, 1.0, 0.3, 0.05,
    help="Thickness of the slab.")

outside_temp = st.sidebar.slider("Outside Temperature (°C)", -10, 40, 20, 1,
    help="Ambient temperature around the slab.")
initial_temp = st.sidebar.slider("Initial Concrete Temp (°C)", 0, 40, 25, 1,
    help="Fresh concrete casting temperature.")

hydration_heat = st.sidebar.slider("Hydration Heat Peak (°C rise)", 0, 50, 20, 1,
    help="Maximum internal temperature rise due to curing reaction.")
hydration_rate = st.sidebar.slider("Hydration Rate", 0.01, 0.2, 0.05, 0.01,
    help="Speed at which hydration heat develops (higher = faster).")

sim_time = st.sidebar.number_input("Simulation time (hours)", 1, 72, 24, 1,
    help="Total curing duration to simulate.")

dt = 0.1  # hours per step
steps = int(sim_time / dt)

# Grid resolution
nx, ny, nz = 40, 20, 10
x = np.linspace(0, length, nx)
y = np.linspace(0, width, ny)
z = np.linspace(0, height, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# -------------------------
# Build hollow-core geometry mask
# -------------------------
mask = np.ones_like(X, dtype=bool)

n_cores = 5
core_radius = height / 4
spacing = width / (n_cores + 1)
for i in range(n_cores):
    cy = (i + 1) * spacing
    cz = height / 2
    r = np.sqrt((Y - cy) ** 2 + (Z - cz) ** 2)
    mask &= r > core_radius

# -------------------------
# Initial Temperature
# -------------------------
T = np.full_like(X, initial_temp, dtype=float)
T[~mask] = outside_temp  # voids start at outside temp

alpha = 1e-6  # thermal diffusivity (m²/s) simplified
alpha_h = alpha * 3600  # per hour for dt

# -------------------------
# Simulation loop
# -------------------------
snapshots = []
for step in range(steps):
    t = step * dt

    # Heat equation with simple finite differences
    Tnew = T.copy()
    lap = (
        np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
        np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) +
        np.roll(T, 1, axis=2) + np.roll(T, -1, axis=2) -
        6 * T
    )
    Tnew[mask] += alpha_h * lap[mask]

    # Hydration heat release (exponential growth curve)
    Q = hydration_heat * (1 - np.exp(-hydration_rate * t))
    Tnew[mask] += Q * dt

    # Boundary conditions: exposed to outside temp
    Tnew[0, :, :] = outside_temp
    Tnew[-1, :, :] = outside_temp
    Tnew[:, 0, :] = outside_temp
    Tnew[:, -1, :] = outside_temp
    Tnew[:, :, 0] = outside_temp
    Tnew[:, :, -1] = outside_temp

    T = Tnew
    if step % int(1/dt) == 0:  # save every hour
        snapshots.append((t, T.copy()))

# -------------------------
# Visualization
# -------------------------
frame = st.slider("Time step (hours)", 0, len(snapshots)-1, 0)
time_h, Tcurr = snapshots[frame]

# Mask voids with dummy low value so they vanish
Tplot = Tcurr.copy()
Tplot[~mask] = np.nanmin(Tcurr) - 100

tmin, tmax = np.nanmin(Tplot), np.nanmax(Tplot)
if np.isclose(tmin, tmax):
    tmax = tmin + 0.1

mode = st.radio("Rendering mode", ["Isosurface", "Volume"], horizontal=True)

if mode == "Isosurface":
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=Tplot.flatten(),
        isomin=tmin,
        isomax=tmax,
        surface_count=6,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale="Inferno",
        showscale=True,
        colorbar=dict(title="Temperature (°C)")
    ))
else:
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=Tplot.flatten(),
        isomin=tmin,
        isomax=tmax,
        opacity=0.1,
        surface_count=20,
        colorscale="Inferno",
        showscale=True,
        colorbar=dict(title="Temperature (°C)")
    ))

# CAD-style orbit camera
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=2, y=2, z=1.5)
)

fig.update_layout(
    title=f"t = {time_h:.2f} h",
    scene_aspectmode="manual",
    scene_aspectratio=dict(x=length, y=width, z=height),
    scene=dict(
        xaxis_title="Length (m)",
        yaxis_title="Width (m)",
        zaxis_title="Height (m)"
    ),
    height=750,
    uirevision="constant",
    scene_camera=camera,
    dragmode="orbit"
)

st.plotly_chart(fig, use_container_width=True)

