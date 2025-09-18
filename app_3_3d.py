import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Simulation Settings")
T_plate = st.sidebar.slider("Heating Plate Temperature (°C)", 20, 200, 80)
T_air = st.sidebar.slider("Air Temperature (°C)", -10, 50, 20)
sim_time = st.sidebar.slider("Simulation Duration (hours)", 1, 48, 12)
timestep = 600  # seconds
n_steps = int((sim_time * 3600) / timestep)

# -----------------------------
# Cube geometry
# -----------------------------
nx, ny, nz = 20, 20, 20
dx = 1.0 / nx
alpha = 1e-6  # thermal diffusivity (m²/s), approx for concrete

# Initialize temperature field
T = np.ones((nx, ny, nz)) * T_air

# Precompute all timesteps
temps = []
for step in range(n_steps):
    T_new = T.copy()

    # finite difference for conduction
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                lap = (
                    T[i+1,j,k] + T[i-1,j,k] +
                    T[i,j+1,k] + T[i,j-1,k] +
                    T[i,j,k+1] + T[i,j,k-1] - 6*T[i,j,k]
                )
                T_new[i,j,k] += alpha * timestep / (dx*dx) * lap

    # Boundary conditions
    T_new[:,:,0] = T_plate   # bottom face = plate temp
    T_new[:,:, -1] = T_air   # top = air temp
    T_new[0,:,:] = T_air     # sides = air temp
    T_new[-1,:,:] = T_air
    T_new[:,0,:] = T_air
    T_new[:,-1,:] = T_air

    T = T_new
    if step % max(1, n_steps // 50) == 0:
        temps.append(T.copy())

# -----------------------------
# Time slider
# -----------------------------
st.sidebar.header("View")
frame = st.sidebar.slider("Time step", 0, len(temps)-1, 0)

# -----------------------------
# 3D visualization
# -----------------------------
T_show = temps[frame]

x, y, z = np.mgrid[0:1:nx*1j, 0:1:ny*1j, 0:1:nz*1j]

fig = go.Figure(data=go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=T_show.flatten(),
    isomin=np.min(T_show),
    isomax=np.max(T_show),
    opacity=0.1,
    surface_count=15,
    colorscale="Jet",
    colorbar=dict(title="°C")
))

fig.update_layout(
    scene=dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        aspectmode="cube"
    )
)

st.plotly_chart(fig, use_container_width=True)
