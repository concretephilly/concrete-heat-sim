import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Concrete Heat Simulation", layout="wide")

st.title("Concrete Hollow-Core Slab Heat Simulation")

# -------------------------
# Sidebar: Slab & Environment
# -------------------------
st.sidebar.header("Slab & Environment Settings")

length = st.sidebar.number_input("Slab length (m)", 4.0, 20.0, 8.0, 0.5)
width = st.sidebar.number_input("Slab width (m)", 0.5, 3.0, 1.2, 0.1)
height = st.sidebar.number_input("Slab height (m)", 0.1, 1.0, 0.3, 0.05)

outside_temp = st.sidebar.slider("Outside Temp (°C)", -10, 40, 20, 1)
initial_temp = st.sidebar.slider("Initial Concrete Temp (°C)", 0, 40, 25, 1)

sim_time = st.sidebar.number_input("Simulation time (hours)", 1, 72, 24, 1)

# -------------------------
# Sidebar: Concrete Properties
# -------------------------
st.sidebar.header("Concrete Properties")

density = st.sidebar.number_input("Density (kg/m³)", 1800, 2800, 2400, 50)
c_heat = st.sidebar.number_input("Specific Heat (J/kg·K)", 600, 1500, 900, 50)
hydration_energy = st.sidebar.number_input("Hydration Energy (J/kg)", 0, 600000, 350000, 5000)
hydration_rate = st.sidebar.slider("Hydration Rate", 0.001, 0.2, 0.05, 0.005)

# Derived max temperature rise from hydration
temp_rise_max = hydration_energy / (density * c_heat)

# -------------------------
# Grid setup
# -------------------------
dt = 0.1  # hours
steps = int(sim_time / dt)

nx, ny, nz = 40, 20, 10
x = np.linspace(0, length, nx)
y = np.linspace(0, width, ny)
z = np.linspace(0, height, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# -------------------------
# Hollow-core geometry
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
T[~mask] = outside_temp  # voids = air at outside temp

alpha = 1e-6  # thermal diffusivity (m²/s)
alpha_h = alpha * 3600  # per hour

# -------------------------
# Simulation
# -------------------------
snapshots = []
for step in range(steps):
    t = step * dt
    Tnew = T.copy()

    # Heat equation (finite diff)
    lap = (
        np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
        np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) +
        np.roll(T, 1, axis=2) + np.roll(T, -1, axis=2) -
        6 * T
    )
    Tnew[mask] += alpha_h * lap[mask]

    # Hydration heat release
    Q = temp_rise_max * (1 - np.exp(-hydration_rate * t))
    Tnew[mask] += Q * dt

    # Boundary conditions (slab surfaces to outside temp)
    Tnew[0, :, :] = outside_temp
    Tnew[-1, :, :] = outside_temp
    Tnew[:, 0, :] = outside_temp
    Tnew[:, -1, :] = outside_temp
    Tnew[:, :, 0] = outside_temp
    Tnew[:, :, -1] = outside_temp

    # Clamp temps (cannot go below outside temp)
    Tnew = np.maximum(Tnew, outside_temp)

    T = Tnew
    if step % int(1/dt) == 0:
        snapshots.append((t, T.copy()))

# -------------------------
# Visualization
# -------------------------
frame = st.slider("Time step (hours)", 0, len(snapshots)-1, 0)
time_h, Tcurr = snapshots[frame]

view_mode = st.radio("View Mode", ["3D Isosurface", "3D Volume", "2D Cross-section"], horizontal=True)

if view_mode == "2D Cross-section":
    axis = st.radio("Slice Axis", ["X", "Y", "Z"], horizontal=True)
    if axis == "X":
        slice_idx = nx // 2
        data = Tcurr[slice_idx, :, :]
        x_axis, y_axis = y, z
        xlabel, ylabel = "Width (m)", "Height (m)"
    elif axis == "Y":
        slice_idx = ny // 2
        data = Tcurr[:, slice_idx, :]
        x_axis, y_axis = x, z
        xlabel, ylabel = "Length (m)", "Height (m)"
    else:
        slice_idx = nz // 2
        data = Tcurr[:, :, slice_idx]
        x_axis, y_axis = x, y
        xlabel, ylabel = "Length (m)", "Width (m)"

    fig = go.Figure(data=go.Heatmap(
        z=data.T,
        x=x_axis,
        y=y_axis,
        colorscale="Inferno",
        colorbar=dict(title="Temp (°C)")
    ))
    fig.update_layout(
        title=f"Cross-section at t = {time_h:.2f} h",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

else:
    tmin, tmax = np.min(Tcurr), np.max(Tcurr)
    if np.isclose(tmin, tmax):
        tmax = tmin + 0.1

    if view_mode == "3D Isosurface":
        fig = go.Figure(data=go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=Tcurr.flatten(),
            isomin=tmin,
            isomax=tmax,
            surface_count=6,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale="Inferno",
            showscale=True,
            colorbar=dict(title="Temp (°C)")
        ))
    else:
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=Tcurr.flatten(),
            isomin=tmin,
            isomax=tmax,
            opacity=0.1,
            surface_count=20,
            colorscale="Inferno",
            showscale=True,
            colorbar=dict(title="Temp (°C)")
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

# -------------------------
# Temperature history plot
# -------------------------
st.subheader("Temperature Evolution")

times = [t for t, _ in snapshots]
avg_temps = [np.mean(T[mask]) for _, T in snapshots]
max_temps = [np.max(T[mask]) for _, T in snapshots]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=times, y=avg_temps, mode="lines", name="Average Temp"))
fig2.add_trace(go.Scatter(x=times, y=max_temps, mode="lines", name="Max Temp"))

# Add marker for peak max temp
peak_idx = int(np.argmax(max_temps))
fig2.add_trace(go.Scatter(
    x=[times[peak_idx]],
    y=[max_temps[peak_idx]],
    mode="markers+text",
    text=["Peak"],
    textposition="top center",
    marker=dict(size=10, color="red"),
    name="Peak Temp"
))

fig2.update_layout(
    xaxis_title="Time (hours)",
    yaxis_title="Temperature (°C)",
    title="Concrete Temperature vs Time",
    height=400,
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig2, use_container_width=True)
