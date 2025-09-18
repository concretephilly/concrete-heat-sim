import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve

st.set_page_config(layout="wide")
st.title("Concrete Heat Simulation (Crank–Nicolson, CEM III defaults)")

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

nx = st.sidebar.slider("Grid resolution X", 6, 20, 10)
ny = st.sidebar.slider("Grid resolution Y", 6, 20, 10)
nz = st.sidebar.slider("Grid resolution Z", 6, 20, 10)

slab_length = st.sidebar.number_input("Block length (m)", 0.5, 5.0, 1.0, 0.1)
slab_width = st.sidebar.number_input("Block width (m)", 0.5, 5.0, 1.0, 0.1)
slab_height = st.sidebar.number_input("Block height (m)", 0.5, 5.0, 1.0, 0.1)

# Concrete thermal properties (defaults = CEM III concrete)
st.sidebar.subheader("Concrete properties (CEM III defaults)")
rho = st.sidebar.number_input("Density ρ (kg/m³)", 2000.0, 2600.0, 2300.0)
c = st.sidebar.number_input("Specific heat c (J/kg·K)", 700.0, 1200.0, 900.0)
k = st.sidebar.number_input("Thermal conductivity k (W/m·K)", 1.0, 3.0, 2.0)
alpha = k / (rho * c)

# Boundary and initial conditions
st.sidebar.subheader("Boundary conditions")
T_init = st.sidebar.number_input("Initial temperature (°C)", -20.0, 60.0, 20.0)
T_air = st.sidebar.number_input("Air temperature (°C)", -20.0, 60.0, 20.0)
T_bed = st.sidebar.number_input("Max bed temperature (°C)", 0.0, 100.0, 60.0)
ramp_time = st.sidebar.number_input("Bed heating ramp time (hours)", 0.0, 12.0, 2.0)

# Time control
st.sidebar.subheader("Time settings")
dt = st.sidebar.number_input("Time step (s)", 10.0, 600.0, 60.0, 10.0)
nsteps = st.sidebar.slider("Number of timesteps", 10, 200, 50)

run_button = st.sidebar.button("(Re)run simulation now")

# ----------------------------
# Simulation (Crank–Nicolson)
# ----------------------------
def crank_nicolson_heat(nx, ny, nz, dx, dy, dz, dt, nsteps, T_init, T_air, T_bed, ramp_time, alpha):
    # Grid and initial temperature
    T = np.ones((nx, ny, nz)) * T_init
    frames = [T.copy()]
    times = [0.0]

    # 1D diffusion operators
    rx = alpha * dt / (2 * dx**2)
    ry = alpha * dt / (2 * dy**2)
    rz = alpha * dt / (2 * dz**2)

    def laplace_1d(n, r):
        main = (1 + 2*r) * np.ones(n)
        off = -r * np.ones(n-1)
        return diags([main, off, off], [0, -1, 1])

    def laplace_1d_rhs(n, r):
        main = (1 - 2*r) * np.ones(n)
        off = r * np.ones(n-1)
        return diags([main, off, off], [0, -1, 1])

    Ax = laplace_1d(nx, rx)
    Ay = laplace_1d(ny, ry)
    Az = laplace_1d(nz, rz)

    Bx = laplace_1d_rhs(nx, rx)
    By = laplace_1d_rhs(ny, ry)
    Bz = laplace_1d_rhs(nz, rz)

    A = kron(kron(Ax, identity(ny)), identity(nz)) \
      + kron(kron(identity(nx), Ay), identity(nz)) \
      + kron(kron(identity(nx), identity(ny)), Az)

    B = kron(kron(Bx, identity(ny)), identity(nz)) \
      + kron(kron(identity(nx), By), identity(nz)) \
      + kron(kron(identity(nx), identity(ny)), Bz)

    # Time loop
    for step in range(1, nsteps+1):
        t = step * dt
        hours = t / 3600.0
        # ramp bed temp
        Tbed_now = T_bed * min(hours / ramp_time, 1.0) if ramp_time > 0 else T_bed

        # flatten temperature
        Tflat = T.flatten()

        rhs = B @ Tflat
        Tnew = spsolve(A, rhs).reshape((nx, ny, nz))

        # boundary conditions (Dirichlet)
        Tnew[0, :, :] = T_air
        Tnew[-1, :, :] = T_air
        Tnew[:, 0, :] = T_air
        Tnew[:, -1, :] = T_air
        Tnew[:, :, -1] = T_air
        Tnew[:, :, 0] = Tbed_now

        T = Tnew
        frames.append(T.copy())
        times.append(hours)

    return frames, times

# Run sim only when button clicked
if run_button:
    with st.spinner("Running Crank–Nicolson simulation..."):
        dx = slab_length / nx
        dy = slab_width / ny
        dz = slab_height / nz
        frames, times = crank_nicolson_heat(nx, ny, nz, dx, dy, dz,
                                            dt, nsteps, T_init, T_air,
                                            T_bed, ramp_time, alpha)
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
