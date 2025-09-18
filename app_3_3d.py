import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Concrete Hollow-Core Slab Heat Simulation (CEM III Defaults + Convection)")

# ------------------------
# Default material properties (CEM III)
# ------------------------
default_density = 2300.0        # kg/m3
default_cp = 900.0              # J/kgK
default_k = 2.5                 # W/mK
default_hydration = 250e3       # J/kg cement (simplified scaling)
default_outside_temp = 20.0     # °C
default_h = 10.0                # W/m²K, natural convection

# ------------------------
# Sidebar settings
# ------------------------
st.sidebar.header("Material & Environmental Settings")
density = st.sidebar.number_input("Density (kg/m³)", 1500.0, 3000.0, default_density)
cp = st.sidebar.number_input("Specific heat (J/kgK)", 500.0, 1500.0, default_cp)
k = st.sidebar.number_input("Thermal conductivity (W/mK)", 0.5, 5.0, default_k)
hydration_heat = st.sidebar.number_input("Hydration heat (J/kg cement)", 1e4, 5e5, default_hydration)
outside_temp = st.sidebar.number_input("Outside air temperature (°C)", -20.0, 40.0, default_outside_temp)
h_conv = st.sidebar.number_input("Convection coefficient h (W/m²K)", 2.0, 100.0, default_h)

st.sidebar.header("Geometry")
length = st.sidebar.number_input("Slab length (m)", 1.0, 20.0, 4.0)
width = st.sidebar.number_input("Slab width (m)", 0.5, 5.0, 1.2)
height = st.sidebar.number_input("Slab height (m)", 0.1, 1.0, 0.2)
n_voids = st.sidebar.number_input("Number of hollow cores", 1, 10, 5)
void_radius = st.sidebar.number_input("Hollow core radius (m)", 0.05, 0.2, 0.08)

st.sidebar.header("Simulation Settings")
nx = st.sidebar.slider("Grid resolution lengthwise", 20, 200, 120)
ny = st.sidebar.slider("Grid resolution widthwise", 10, 100, 60)
nz = st.sidebar.slider("Grid resolution heightwise", 5, 50, 20)
dt = 60.0   # timestep (s)
tmax = st.sidebar.number_input("Simulation duration (hours)", 1, 200, 72)

# ------------------------
# Grid setup
# ------------------------
dx = length / nx
dy = width / ny
dz = height / nz
alpha = k / (density * cp)

x = np.linspace(0, length, nx)
y = np.linspace(0, width, ny)
z = np.linspace(0, height, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# mask: True = concrete, False = void
mask = np.ones((nx, ny, nz), dtype=bool)
spacing = width / (n_voids + 1)
cy = np.linspace(spacing, width - spacing, n_voids)
cz = np.full_like(cy, height / 2)

for i in range(nx):
    for j in range(ny):
        for k_ in range(nz):
            for c_y, c_z in zip(cy, cz):
                if (Y[i, j, k_] - c_y) ** 2 + (Z[i, j, k_] - c_z) ** 2 < void_radius**2:
                    mask[i, j, k_] = False

# ------------------------
# Simulation
# ------------------------
T = np.ones((nx, ny, nz)) * (outside_temp + 10.0)  # slightly warmer initial slab
snapshots = [(0.0, T.copy())]

nsteps = int((tmax * 3600) / dt)
save_interval = max(1, nsteps // 50)

for step in range(nsteps):
    Tnew = T.copy()
    for i in range(nx):
        for j in range(ny):
            for k_ in range(nz):
                if mask[i, j, k_]:
                    # internal conduction
                    if 0 < i < nx-1 and 0 < j < ny-1 and 0 < k_ < nz-1:
                        lap = (T[i+1, j, k_] + T[i-1, j, k_] +
                               T[i, j+1, k_] + T[i, j-1, k_] +
                               T[i, j, k_+1] + T[i, j, k_-1] - 6*T[i, j, k_]) / dx**2
                    else:
                        lap = 0.0
                    source = (hydration_heat / (density*cp)) * np.exp(-step*dt/3600/10.0)
                    dT = alpha * dt * lap + source * dt
                    Tnew[i, j, k_] = T[i, j, k_] + dT

                    # surface convection cooling (outer faces only)
                    if i == 0 or i == nx-1:
                        Tnew[i, j, k_] += - (h_conv/(density*cp)) * (2/dx) * (T[i, j, k_] - outside_temp) * dt
                    if j == 0 or j == ny-1:
                        Tnew[i, j, k_] += - (h_conv/(density*cp)) * (2/dy) * (T[i, j, k_] - outside_temp) * dt
                    if k_ == 0 or k_ == nz-1:
                        Tnew[i, j, k_] += - (h_conv/(density*cp)) * (2/dz) * (T[i, j, k_] - outside_temp) * dt

                else:
                    # Hollow cores exchange with outside air only
                    Tnew[i, j, k_] = outside_temp + 0.05*(np.mean([
                        T[min(nx-1,i+1),j,k_],T[max(0,i-1),j,k_],
                        T[i,min(ny-1,j+1),k_],T[i,max(0,j-1),k_],
                        T[i,j,min(nz-1,k_+1)],T[i,j,max(0,k_-1)]
                    ]) - outside_temp)
    T = Tnew
    if step % save_interval == 0:
        snapshots.append((step*dt/3600, T.copy()))

# ------------------------
# Cross-sectional visualization
# ------------------------
st.subheader("Cross-sectional Temperature Field")

frame = st.slider("Time step (index)", 0, len(snapshots)-1, 0)
time_h, Tcurr = snapshots[frame]

axis = st.radio("Slice Axis", ["X", "Y", "Z"], horizontal=True)
if axis == "X":
    slice_idx = nx // 2
    data = Tcurr[slice_idx, :, :]
    xlabel, ylabel = "Width (m)", "Height (m)"
    x_axis, y_axis = y, z
elif axis == "Y":
    slice_idx = ny // 2
    data = Tcurr[:, slice_idx, :]
    xlabel, ylabel = "Length (m)", "Height (m)"
    x_axis, y_axis = x, z
else:
    slice_idx = nz // 2
    data = Tcurr[:, :, slice_idx]
    xlabel, ylabel = "Length (m)", "Width (m)"
    x_axis, y_axis = x, y

fig = go.Figure(data=go.Heatmap(
    z=data.T,
    x=x_axis,
    y=y_axis,
    colorscale="Jet",
    colorbar=dict(title="Temp (°C)")
))
fig.update_layout(
    title=f"Cross-section at t = {time_h:.2f} h",
    xaxis_title=xlabel,
    yaxis_title=ylabel,
    yaxis=dict(scaleanchor="x", scaleratio=1)
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Temperature history plot
# ------------------------
st.subheader("Temperature Evolution")

times = [t for t, _ in snapshots]
avg_temps = [np.mean(Tsnap[mask]) for _, Tsnap in snapshots]
max_temps = [np.max(Tsnap[mask]) for _, Tsnap in snapshots]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=times, y=avg_temps, mode="lines", name="Average Temp", line=dict(color="blue")))
fig2.add_trace(go.Scatter(x=times, y=max_temps, mode="lines", name="Max Temp", line=dict(color="red")))

# Equalization estimate
tolerance = 0.5
eq_time = None
for t, avg in zip(times, avg_temps):
    if abs(avg - outside_temp) <= tolerance:
        eq_time = t
        break

if eq_time is not None:
    fig2.add_vline(x=eq_time, line=dict(color="green", dash="dash"))
    fig2.add_trace(go.Scatter(
        x=[eq_time], y=[outside_temp],
        mode="markers+text",
        text=["Equalized"], textposition="top right",
        marker=dict(color="green", size=10, symbol="x"),
        name="Equalization Point"
    ))

fig2.update_layout(xaxis_title="Time (hours)", yaxis_title="Temperature (°C)",
                   title="Concrete Temperature vs Time", height=400)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Equalization Estimate")
if eq_time is not None:
    st.success(f"The slab reaches equilibrium with outside air (≈{outside_temp}°C) at about **{eq_time:.1f} hours**.")
else:
    st.info("The slab has not equalized with outside air within the simulated time.")
