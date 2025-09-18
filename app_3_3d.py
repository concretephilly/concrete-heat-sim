# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Hollow-Core Slab Heat (CEM III defaults)")

st.title("Hollow-Core Slab — Heat Development (CEM III defaults)")

# -------------------------
# Defaults (CEM III-like)
# -------------------------
DEFAULT_DENSITY = 2300.0       # kg/m³
DEFAULT_CP = 900.0             # J/kgK
DEFAULT_K = 2.5                # W/mK
DEFAULT_CEMENT = 350.0         # kg/m³
DEFAULT_QTOT = 250e3           # J/kg cement
DEFAULT_K_RATE_H = 0.05        # 1/h
DEFAULT_OUTSIDE_TEMP = 20.0    # °C
DEFAULT_H = 10.0               # W/m²K

# -------------------------
# Sidebar parameters
# -------------------------
st.sidebar.header("Geometry")
length = st.sidebar.number_input("Length (m)", 1.0, 20.0, 4.0, 0.1)
width = st.sidebar.number_input("Width (m)", 0.5, 5.0, 1.2, 0.05)
height = st.sidebar.number_input("Height (m)", 0.08, 0.5, 0.20, 0.01)

st.sidebar.header("Hollow-cores")
n_voids = st.sidebar.number_input("Number of cores", 1, 8, 5, 1)
void_radius = st.sidebar.number_input("Core radius (m)", 0.02, 0.25, 0.06, 0.005)

st.sidebar.header("Material (CEM III defaults)")
density = st.sidebar.number_input("Density ρ (kg/m³)", 1800.0, 2800.0, DEFAULT_DENSITY, 10.0)
cp = st.sidebar.number_input("Specific heat cₚ (J/kg·K)", 600.0, 1200.0, DEFAULT_CP, 10.0)
k_conc = st.sidebar.number_input("Conductivity k (W/m·K)", 0.5, 4.0, DEFAULT_K, 0.1)

st.sidebar.header("Hydration")
cement_content = st.sidebar.number_input("Cement content (kg/m³)", 150.0, 600.0, DEFAULT_CEMENT, 10.0)
Q_total = st.sidebar.number_input("Total heat (J/kg cement)", 50e3, 600e3, DEFAULT_QTOT, 1000.0)
k_rate_h = st.sidebar.number_input("Reaction rate k (1/h)", 0.001, 1.0, DEFAULT_K_RATE_H, 0.001)

st.sidebar.header("Environment & Simulation")
outside_temp = st.sidebar.number_input("Outside air temp (°C)", -20.0, 40.0, DEFAULT_OUTSIDE_TEMP, 0.5)
h_conv = st.sidebar.number_input("Convection h (W/m²K)", 0.5, 100.0, DEFAULT_H, 0.1)
sim_hours = st.sidebar.number_input("Simulation duration (h)", 1, 200, 72, 1)
dt_seconds = st.sidebar.number_input("Timestep (s)", 60.0, 1800.0, 300.0, 60.0)

st.sidebar.header("Resolution")
nx, ny = st.sidebar.slider("Grid resolution (nx, ny)", 40, 200, (120, 60))

# -------------------------
# Simulation setup
# -------------------------
dx = length / nx
dy = width / ny
n_steps = int(sim_hours * 3600 / dt_seconds)

xv = np.linspace(0, length, nx)
yv = np.linspace(0, width, ny)
X, Y = np.meshgrid(xv, yv, indexing="ij")

# Initial temperature field
T = np.full((nx, ny), outside_temp, dtype=np.float64)
alpha = np.zeros((nx, ny), dtype=np.float64)  # hydration degree

# Mask for voids
void_mask = np.zeros_like(T, dtype=bool)
void_spacing = width / (n_voids + 1)
for i in range(n_voids):
    cy = (i + 1) * void_spacing
    void_mask |= ((Y - cy) ** 2 + (X - length/2) ** 2) < void_radius ** 2

# -------------------------
# Run simulation
# -------------------------
avg_temps = []
for step in range(n_steps):
    # hydration progress
    d_alpha = k_rate_h * (1 - alpha) * (dt_seconds / 3600.0)
    alpha += d_alpha
    alpha = np.clip(alpha, 0.0, 1.0)
    q_hyd = cement_content * Q_total * d_alpha / (density * cp)

    # diffusion (2D finite difference)
    T_pad = np.pad(T, 1, mode="edge")
    lap = (
        T_pad[2:,1:-1] + T_pad[:-2,1:-1] +
        T_pad[1:-1,2:] + T_pad[1:-1,:-2] -
        4 * T_pad[1:-1,1:-1]
    ) / (dx * dy)
    dT_diff = k_conc / (density * cp) * lap * dt_seconds

    # convection on boundaries
    surface_loss = np.zeros_like(T)
    surface_loss[0,:] += h_conv * (outside_temp - T[0,:]) * dt_seconds / (density * cp * dx)
    surface_loss[-1,:] += h_conv * (outside_temp - T[-1,:]) * dt_seconds / (density * cp * dx)
    surface_loss[:,0] += h_conv * (outside_temp - T[:,0]) * dt_seconds / (density * cp * dy)
    surface_loss[:,-1] += h_conv * (outside_temp - T[:,-1]) * dt_seconds / (density * cp * dy)

    # update temperature
    T += dT_diff + q_hyd + surface_loss

    # voids stay close to outside air temp
    T[void_mask] = outside_temp

    avg_temps.append(np.mean(T[~void_mask]))

# -------------------------
# Visualization
# -------------------------

col1, col2 = st.columns([1,1])

# 3D geometry slab
with col1:
    z = np.zeros_like(T)
    fig3d = go.Figure(data=[go.Surface(
        z=z,
        x=xv,
        y=yv,
        surfacecolor=~void_mask,
        colorscale=[[0,"lightgrey"], [1,"grey"]],
        showscale=False,
        opacity=0.8
    )])
    fig3d.update_layout(
        title="3D Slab Geometry",
        scene=dict(
            xaxis_title="Length (m)",
            yaxis_title="Width (m)",
            zaxis_title="Height (m)",
            aspectmode="data"
        )
    )
    st.plotly_chart(fig3d, use_container_width=True)

# 2D cross-section heatmap
with col2:
    fig2d = go.Figure(data=go.Heatmap(
        z=T.T,
        x=xv,
        y=yv,
        colorscale=[
            [0.0, "blue"],
            [0.5, "yellow"],
            [1.0, "red"]
        ],
        colorbar=dict(title="°C")
    ))
    fig2d.update_layout(
        title="Cross-section Temperature Distribution",
        xaxis_title="Length (m)",
        yaxis_title="Width (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    st.plotly_chart(fig2d, use_container_width=True)

# Line plot: avg temp over time
times = np.arange(n_steps) * dt_seconds / 3600
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=times, y=avg_temps, mode="lines", name="Avg Temp"))
fig_line.update_layout(
    title="Average Concrete Temperature Over Time",
    xaxis_title="Time (hours)",
    yaxis_title="Temperature (°C)"
)
st.plotly_chart(fig_line, use_container_width=True)
