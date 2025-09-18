import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")
st.title("2D Hollow-Core Concrete Slab Heat Simulation")

# -----------------------
# Sidebar inputs
# -----------------------
st.sidebar.header("Geometry")
nx = st.sidebar.slider("Grid cells (x - width)", 20, 200, 60)
ny = st.sidebar.slider("Grid cells (y - height)", 10, 100, 30)
width = st.sidebar.number_input("Width (m)", 0.5, 5.0, 1.2)
height = st.sidebar.number_input("Height (m)", 0.1, 1.0, 0.25)

st.sidebar.header("Material & Environment")
initial_temp = st.sidebar.number_input("Initial concrete temp (°C)", -10.0, 80.0, 20.0)
outside_temp = st.sidebar.number_input("Outside temp (°C)", -40.0, 60.0, 5.0)
rho_conc = st.sidebar.number_input("Concrete density (kg/m³)", 1500.0, 3000.0, 2400.0)
cp_conc = st.sidebar.number_input("Concrete heat capacity (J/kgK)", 400.0, 2000.0, 900.0)
k_conc = st.sidebar.number_input("Concrete conductivity (W/mK)", 0.05, 5.0, 1.7)
h = st.sidebar.slider("Convection coefficient h (W/m²K)", 1, 50, 10)

st.sidebar.header("Time / simulation")
total_time = st.sidebar.number_input("Total time (hours)", 0.1, 24.0, 2.0)
dt = st.sidebar.number_input("Time step (s)", 0.1, 600.0, 5.0)

# hollow-core parameters
n_cores = st.sidebar.slider("Number of voids", 0, 6, 3)
core_radius_fraction = st.sidebar.slider("Core radius (fraction of height)", 0.05, 0.5, 0.25)

# -----------------------
# Grid setup
# -----------------------
dx = width / nx
dy = height / ny
alpha_conc = k_conc / (rho_conc * cp_conc)
k_air = 0.025
rho_air = 1.225
cp_air = 1005.0
alpha_air = k_air / (rho_air * cp_air)

nt = int(total_time * 3600 / dt)

# coordinate grids
xs = np.linspace(-width/2, width/2, nx)
ys = np.linspace(-height/2, height/2, ny)
X, Y = np.meshgrid(xs, ys)

# hollow-core mask
mask = np.ones((ny, nx), dtype=bool)
if n_cores > 0:
    spacing = width / (n_cores + 1)
    core_radius = core_radius_fraction * height
    for i in range(n_cores):
        cx = -width/2 + (i+1)*spacing
        mask &= (X - cx)**2 + Y**2 >= core_radius**2

# material property map
alpha = np.where(mask, alpha_conc, alpha_air)
rho = np.where(mask, rho_conc, rho_air)
cp = np.where(mask, cp_conc, cp_air)

# -----------------------
# Simulation
# -----------------------
@st.cache_data(show_spinner=False)
def run_simulation(nx, ny, alpha, rho, cp, initial_temp, outside_temp, h, dx, dy, dt, nt):
    T = np.full((ny, nx), initial_temp, dtype=float)
    results = []
    store_every = max(1, nt // 200)

    for n in range(nt):
        Tn = T.copy()
        lap = np.zeros_like(Tn)
        lap[1:-1,1:-1] = (
            (Tn[1:-1,2:] - 2*Tn[1:-1,1:-1] + Tn[1:-1,:-2]) / dx**2 +
            (Tn[2:,1:-1] - 2*Tn[1:-1,1:-1] + Tn[:-2,1:-1]) / dy**2
        )
        T = Tn + alpha * dt * lap

        # convection at boundaries
        T[0,:]   = Tn[0,:]   - (h*dt/(rho[0,:]*cp[0,:]*dy))*(Tn[0,:]   - outside_temp)
        T[-1,:]  = Tn[-1,:]  - (h*dt/(rho[-1,:]*cp[-1,:]*dy))*(Tn[-1,:] - outside_temp)
        T[:,0]   = Tn[:,0]   - (h*dt/(rho[:,0]*cp[:,0]*dx))*(Tn[:,0]   - outside_temp)
        T[:,-1]  = Tn[:,-1]  - (h*dt/(rho[:,-1]*cp[:,-1]*dx))*(Tn[:,-1] - outside_temp)

        if n % store_every == 0:
            results.append((n*dt/3600, T.copy()))
    return results

with st.spinner("Running simulation..."):
    results = run_simulation(nx, ny, alpha, rho, cp,
                             initial_temp, outside_temp, h,
                             dx, dy, dt, nt)

# -----------------------
# Playback controls
# -----------------------
if "frame" not in st.session_state:
    st.session_state.frame = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

st.sidebar.header("Playback")
if st.sidebar.button("Play"):
    st.session_state.playing = True
if st.sidebar.button("Pause"):
    st.session_state.playing = False
if st.sidebar.button("Reset"):
    st.session_state.frame = 0
    st.session_state.playing = False

frame = st.sidebar.slider("Frame", 0, len(results)-1, st.session_state.frame)
st.session_state.frame = frame

# auto-play without experimental_rerun
if st.session_state.playing:
    st.session_state.frame = (st.session_state.frame + 1) % len(results)
    time.sleep(0.2)

# -----------------------
# Plot
# -----------------------
time_h, temp = results[st.session_state.frame]
fig, ax = plt.subplots(figsize=(6,4))
im = ax.imshow(temp, origin="lower",
               extent=(xs.min(), xs.max(), ys.min(), ys.max()),
               cmap="inferno", aspect="auto")
ax.set_title(f"Hollow-core slab at t={time_h:.2f} h")
ax.set_xlabel("Width (m)")
ax.set_ylabel("Height (m)")
plt.colorbar(im, ax=ax, label="°C")
st.pyplot(fig)
