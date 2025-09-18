import streamlit as st
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# -------------------------
# Session state init
# -------------------------
if "frames" not in st.session_state:
    st.session_state.frames = None
if "times" not in st.session_state:
    st.session_state.times = None

# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.header("Simulation Settings")

T_plate = st.sidebar.slider("Heating Plate Temperature (°C)", 20, 200, 80)
T_air = st.sidebar.slider("Air Temperature (°C)", -10, 50, 20)
sim_time = st.sidebar.slider("Simulation Duration (hours)", 1, 48, 12)

nx, ny, nz = 15, 15, 15
dx = 1.0 / nx

# Concrete properties (CEM III typical values)
rho = 2400        # kg/m³
c = 900           # J/kgK
k = 2.0           # W/mK
alpha = k / (rho * c)  # thermal diffusivity ≈ 9.3e-7 m²/s

dt = 300  # s
n_steps = int((sim_time * 3600) / dt)

# -------------------------
# Helper functions
# -------------------------
def build_system(nx, ny, nz, dx, alpha, dt, T_air, T_plate):
    N = nx * ny * nz
    main_diag = np.ones(N)
    off_diag = np.ones(N-1)
    diags = [main_diag, -off_diag, -off_diag, -off_diag, -off_diag, -off_diag]
    A = sp.diags(diags, [0, -1, 1, -nx, nx, -nx*ny, nx*ny], format="csr")
    A = (sp.eye(N) - (alpha*dt/dx**2) * A)
    return A

def idx(i,j,k,nx,ny,nz):
    return i + j*nx + k*nx*ny

def run_simulation(nx, ny, nz, dx, alpha, dt, n_steps, T_air, T_plate):
    # Build implicit solver matrix
    A = build_system(nx, ny, nz, dx, alpha, dt, T_air, T_plate)
    N = nx*ny*nz
    T = np.ones(N) * T_air

    frames = []
    times = []
    for step in range(n_steps):
        # Boundary conditions
        # bottom face = plate
        for i in range(nx):
            for j in range(ny):
                T[idx(i,j,0,nx,ny,nz)] = T_plate
        # top face & sides = air
        for i in range(nx):
            for j in range(ny):
                T[idx(i,j,nz-1,nx,ny,nz)] = T_air
        for i in range(nx):
            for k in range(nz):
                T[idx(i,0,k,nx,ny,nz)] = T_air
                T[idx(i,ny-1,k,nx,ny,nz)] = T_air
        for j in range(ny):
            for k in range(nz):
                T[idx(0,j,k,nx,ny,nz)] = T_air
                T[idx(nx-1,j,k,nx,ny,nz)] = T_air

        # Solve system
        T = spla.spsolve(A, T)

        if step % max(1, n_steps//50) == 0:
            frames.append(T.reshape((nx,ny,nz)))
            times.append(step*dt/3600.0)

    return frames, times

# -------------------------
# Run button
# -------------------------
if st.button("(Re)run simulation now"):
    with st.spinner("Running FEA simulation..."):
        frames, times = run_simulation(nx, ny, nz, dx, alpha, dt, n_steps, T_air, T_plate)
    st.session_state.frames = frames
    st.session_state.times = times
    st.success("Simulation complete ✅")

# -------------------------
# Visualization
# -------------------------
if st.session_state.frames is not None:
    frames = st.session_state.frames
    times = st.session_state.times

    view = st.selectbox("View", ["3D Volume (cube)", "2D cross-section", "Temperature vs time"])
    frame_idx = st.slider("Frame index", 0, len(frames)-1, 0)
    Tframe = frames[frame_idx]

    if view == "3D Volume (cube)":
        x, y, z = np.mgrid[0:1:nx*1j, 0:1:ny*1j, 0:1:nz*1j]
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=Tframe.flatten(),
            isomin=np.min(Tframe),
            isomax=np.max(Tframe),
            opacity=0.1,
            surface_count=15,
            colorscale="Jet",
            colorbar=dict(title="°C")
        ))
        fig.update_layout(scene=dict(aspectmode="cube"))
        st.plotly_chart(fig, use_container_width=True)

    elif view == "2D cross-section":
        mid = nz//2
        fig = go.Figure(data=go.Heatmap(
            z=Tframe[:,:,mid],
            colorscale="Jet",
            colorbar=dict(title="°C")
        ))
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Temperature vs time":
        avg_temp = [np.mean(F) for F in frames]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=avg_temp, mode="lines", name="Avg temp"))
        fig.update_layout(xaxis_title="Time (h)", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Press '(Re)run simulation now' to build and solve the system.")
