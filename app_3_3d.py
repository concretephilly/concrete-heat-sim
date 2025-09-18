import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm

st.set_page_config(layout="wide")
st.title("Concrete Heat Simulation (Homogeneous Block)")

# ----------------------------
# Session state setup
# ----------------------------
if "frames" not in st.session_state:
    st.session_state.frames = None
if "times" not in st.session_state:
    st.session_state.times = None

# ----------------------------
# Parameters
# ----------------------------
st.sidebar.header("Simulation Settings")

nx = st.sidebar.slider("Grid resolution X", 10, 40, 20)
ny = st.sidebar.slider("Grid resolution Y", 10, 40, 20)
nz = st.sidebar.slider("Grid resolution Z", 10, 40, 20)

slab_length = st.sidebar.number_input("Block length (m)", 0.5, 5.0, 1.0, 0.1)
slab_width = st.sidebar.number_input("Block width (m)", 0.5, 5.0, 1.0, 0.1)
slab_height = st.sidebar.number_input("Block height (m)", 0.5, 5.0, 1.0, 0.1)

T_init = st.sidebar.number_input("Initial temperature (°C)", -20.0, 80.0, 20.0)
T_air = st.sidebar.number_input("Air temperature (°C)", -20.0, 60.0, 20.0)
T_bed = st.sidebar.number_input("Heating bed temperature (°C)", 0.0, 100.0, 60.0)

alpha = 8.5e-7   # thermal diffusivity for concrete, m2/s
dt = 60.0        # timestep in seconds
nsteps = st.sidebar.slider("Number of timesteps", 10, 200, 50)

run_button = st.sidebar.button("(Re)run simulation now")

# ----------------------------
# Simulation
# ----------------------------
def run_simulation(nx, ny, nz, T_init, T_air, T_bed, dt, nsteps):
    T = np.ones((nx, ny, nz)) * T_init
    frames = [T.copy()]
    times = [0.0]

    dx = slab_length / nx
    dy = slab_width / ny
    dz = slab_height / nz

    for step in range(1, nsteps+1):
        Tnew = T.copy()

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    d2x = (T[i+1, j, k] - 2*T[i, j, k] + T[i-1, j, k]) / dx**2
                    d2y = (T[i, j+1, k] - 2*T[i, j, k] + T[i, j-1, k]) / dy**2
                    d2z = (T[i, j, k+1] - 2*T[i, j, k] + T[i, j, k-1]) / dz**2
                    Tnew[i, j, k] = T[i, j, k] + alpha * dt * (d2x + d2y + d2z)

        # boundary conditions
        Tnew[0, :, :] = T_air
        Tnew[-1, :, :] = T_air
        Tnew[:, 0, :] = T_air
        Tnew[:, -1, :] = T_air
        Tnew[:, :, -1] = T_air
        Tnew[:, :, 0] = T_bed  # bottom heating

        T = Tnew
        frames.append(T.copy())
        times.append(step * dt / 3600.0)  # hours

    return frames, times

# Run sim only when button clicked
if run_button:
    with st.spinner("Running simulation..."):
        frames, times = run_simulation(nx, ny, nz, T_init, T_air, T_bed, dt, nsteps)
        st.session_state.frames = frames
        st.session_state.times = times
        st.success("Simulation complete ✅")

# ----------------------------
# Visualization
# ----------------------------
if st.session_state.frames is not None:
    frames = st.session_state.frames
    times = st.session_state.times

    view = st.selectbox("View", ["3D Volume (cube)", "2D cross-section", "Temperature vs time"])
    frame_idx = st.slider("Frame index", 0, len(frames) - 1, 0)
    Tframe = frames[frame_idx]

    if view == "3D Volume (cube)":
        fig = go.Figure(data=go.Volume(
            x=np.repeat(np.arange(nx), ny * nz),
            y=np.tile(np.repeat(np.arange(ny), nz), nx),
            z=np.tile(np.arange(nz), nx * ny),
            value=Tframe.flatten(),
            isomin=Tframe.min(),
            isomax=Tframe.max(),
            opacity=0.1,
            surface_count=20,
            colorscale="Jet"
        ))
        fig.update_layout(scene=dict(aspectmode="cube"))
        st.plotly_chart(fig, use_container_width=True)

    elif view == "2D cross-section":
        midz = nz // 2
        fig, ax = plt.subplots()
        im = ax.imshow(Tframe[:, :, midz].T, origin="lower", cmap="jet",
                       extent=[0, slab_length, 0, slab_width], aspect="auto")
        ax.set_title(f"Mid-plane temperature at {times[frame_idx]:.2f} h")
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Width (m)")
        plt.colorbar(im, ax=ax, label="Temperature (°C)")
        st.pyplot(fig)

    elif view == "Temperature vs time":
        midx, midy, midz = nx//2, ny//2, nz//2
        center_temp = [f[midx, midy, midz] for f in frames]
        fig, ax = plt.subplots()
        ax.plot(times, center_temp, label="Center of block")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Temperature evolution at block center")
        ax.legend()
        st.pyplot(fig)
else:
    st.info("Press '(Re)run simulation now' to build and solve the system.")
