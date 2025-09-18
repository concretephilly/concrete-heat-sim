# app.py
import streamlit as st
import numpy as np
import time
import math
import matplotlib.pyplot as plt

# optional interactive plotting with Plotly; fallback to matplotlib if missing
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

st.set_page_config(layout="wide")
st.title("3D Hollow-Core Concrete Slab Heat Simulation (with Play/Pause)")

# -----------------------
# Sidebar inputs
# -----------------------
st.sidebar.header("Geometry")
nx = st.sidebar.slider("Grid cells (x - length)", 8, 80, 30)
ny = st.sidebar.slider("Grid cells (y - width)", 8, 60, 20)
nz = st.sidebar.slider("Grid cells (z - height)", 4, 40, 8)
length = st.sidebar.number_input("Length (m)", 0.5, 20.0, 4.0)
width = st.sidebar.number_input("Width (m)", 0.2, 4.0, 1.2)
height = st.sidebar.number_input("Height (m)", 0.05, 1.0, 0.25)

st.sidebar.header("Material & Environment")
initial_temp = st.sidebar.number_input("Initial concrete temp (°C)", -10.0, 80.0, 20.0)
outside_temp = st.sidebar.number_input("Outside temp (°C)", -40.0, 60.0, 5.0)
rho_conc = st.sidebar.number_input("Concrete density (kg/m³)", 1500.0, 3000.0, 2400.0)
cp_conc = st.sidebar.number_input("Concrete heat capacity (J/kgK)", 400.0, 2000.0, 900.0)
k_conc = st.sidebar.number_input("Concrete conductivity (W/mK)", 0.05, 5.0, 1.7)
h = st.sidebar.slider("Convection coefficient h (W/m²K)", 1, 50, 10)

st.sidebar.header("Time / simulation")
total_time = st.sidebar.number_input("Total time (hours)", 0.01, 48.0, 2.0)
dt_user = st.sidebar.number_input("Requested time step (s)", 0.1, 600.0, 10.0)

# hollow-core parameters (simple repeating circular voids)
n_cores = st.sidebar.slider("Number of voids across width", 0, 6, 3)
core_radius_fraction = st.sidebar.slider("Core radius (fraction of height)", 0.05, 1.0, 0.25)

# -----------------------
# Derived geometry
# -----------------------
dx = length / nx
dy = width / ny
dz = height / nz

# air properties for voids
k_air = 0.025
rho_air = 1.225
cp_air = 1005.0

# thermal diffusivities (will be per-cell)
alpha_conc = k_conc / (rho_conc * cp_conc)
alpha_air = k_air / (rho_air * cp_air)

# -----------------------
# Stability check
# -----------------------
max_alpha = max(alpha_conc, alpha_air)
den = (1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2)
dt_stable = 0.5 / (max_alpha * den) if den > 0 else dt_user
if dt_user > dt_stable:
    st.warning(
        f"Selected dt={dt_user:.3f}s may be unstable for an explicit scheme. "
        f"Using dt={dt_stable*0.9:.6f}s instead for stability."
    )
dt = min(dt_user, dt_stable * 0.9)

nt = int(total_time * 3600.0 / dt)
nt = max(1, nt)
if nt > 5000:
    st.warning(
        f"Long simulation ({nt} steps). For responsiveness the app will store fewer frames. "
        "Consider increasing dt or reducing total time or grid resolution."
    )

# -----------------------
# Create hollow-core mask (vectorized)
# -----------------------
# 2D cross-section grid (y across width, z through height)
ys = np.linspace(-width/2.0, width/2.0, ny)
zs = np.linspace(-height/2.0, height/2.0, nz)
Zg, Yg = np.meshgrid(zs, ys, indexing='xy')  # shape (ny, nz) if indexing='xy'
# (we prefer shape (nz,ny) later)
Zg = Zg.T  # now (nz, ny)
Yg = Yg.T  # (nz, ny)

mask2d = np.ones((nz, ny), dtype=bool)  # True = concrete, False = void

if n_cores > 0:
    core_radius = core_radius_fraction * height
    # place cores evenly across width (y direction)
    centers = np.linspace(-width/2.0 + width/(n_cores+1),
                          width/2.0 - width/(n_cores+1), n_cores)
    for c in centers:
        dist2 = (Yg - c)**2 + (Zg)**2
        mask2d &= dist2 >= core_radius**2

# extend mask in x-direction (voids run along the length)
mask = np.repeat(mask2d[:, :, np.newaxis], nx, axis=2)  # shape (nz, ny, nx)

# per-cell properties arrays
k_arr = np.where(mask, k_conc, k_air)
rho_arr = np.where(mask, rho_conc, rho_air)
cp_arr = np.where(mask, cp_conc, cp_air)
alpha_arr = k_arr / (rho_arr * cp_arr)

# -----------------------
# Simulation function (vectorized interior using slices)
# -----------------------
@st.cache_data(show_spinner=False)
def run_simulation(nx, ny, nz, dx, dy, dz, alpha_arr, rho_arr, cp_arr,
                   initial_temp, outside_temp, h, dt, nt, store_max=200):
    """Run transient explicit heat conduction with convection boundaries.
       Returns list of (time_hours, temperature_array) snapshots.
    """
    T = np.full((nz, ny, nx), initial_temp, dtype=float)
    results = []

    # determine store_every so we don't store too many frames
    store_every = max(1, nt // store_max)

    for step in range(nt):
        Tn = T.copy()

        # compute laplacian for interior points
        lap = np.zeros_like(Tn)
        lap[1:-1,1:-1,1:-1] = (
            (Tn[1:-1,1:-1,2:] - 2.0*Tn[1:-1,1:-1,1:-1] + Tn[1:-1,1:-1,:-2]) / dx**2 +
            (Tn[1:-1,2:,1:-1] - 2.0*Tn[1:-1,1:-1,1:-1] + Tn[1:-1,:-2,1:-1]) / dy**2 +
            (Tn[2:,1:-1,1:-1] - 2.0*Tn[1:-1,1:-1,1:-1] + Tn[:-2,1:-1,1:-1]) / dz**2
        )

        # update all cells using per-cell alpha
        T = Tn + alpha_arr * dt * lap

        # convection on faces (per-cell rho & cp used)
        # X faces (along length)
        rho_face = rho_arr[:,:,0]
        cp_face = cp_arr[:,:,0]
        T[:,:,0] = Tn[:,:,0] - (h*dt/(rho_face*cp_face*dx)) * (Tn[:,:,0] - outside_temp)

        rho_face = rho_arr[:,:,-1]
        cp_face = cp_arr[:,:,-1]
        T[:,:,-1] = Tn[:,:,-1] - (h*dt/(rho_face*cp_face*dx)) * (Tn[:,:,-1] - outside_temp)

        # Y faces (width)
        rho_face = rho_arr[:,0,:]
        cp_face = cp_arr[:,0,:]
        T[:,0,:] = Tn[:,0,:] - (h*dt/(rho_face*cp_face*dy)) * (Tn[:,0,:] - outside_temp)

        rho_face = rho_arr[:,-1,:]
        cp_face = cp_arr[:,-1,:]
        T[:,-1,:] = Tn[:,-1,:] - (h*dt/(rho_face*cp_face*dy)) * (Tn[:,-1,:] - outside_temp)

        # Z faces (top/bottom)
        rho_face = rho_arr[0,:,:]
        cp_face = cp_arr[0,:,:]
        T[0,:,:] = Tn[0,:,:] - (h*dt/(rho_face*cp_face*dz)) * (Tn[0,:,:] - outside_temp)

        rho_face = rho_arr[-1,:,:]
        cp_face = cp_arr[-1,:,:]
        T[-1,:,:] = Tn[-1,:,:] - (h*dt/(rho_face*cp_face*dz)) * (Tn[-1,:,:] - outside_temp)

        if step % store_every == 0:
            results.append((step*dt/3600.0, T.copy()))

    return results

# run (cached) simulation
with st.spinner("Running simulation (this may take a moment)..."):
    results = run_simulation(nx, ny, nz, dx, dy, dz, alpha_arr, rho_arr, cp_arr,
                             initial_temp, outside_temp, h, dt, nt, store_max=200)

if len(results) == 0:
    st.error("No simulation frames were produced. Try changing dt / total time or reduce grid size.")
    st.stop()

# -----------------------
# Playback controls (session state)
# -----------------------
if 'frame' not in st.session_state:
    st.session_state.frame = 0
if 'playing' not in st.session_state:
    st.session_state.playing = False

st.sidebar.header("Playback")
if st.sidebar.button("Play"):
    st.session_state.playing = True
if st.sidebar.button("Pause"):
    st.session_state.playing = False
if st.sidebar.button("Reset"):
    st.session_state.frame = 0
    st.session_state.playing = False

# slider (sync with session state)
max_frame = len(results) - 1
frame = st.sidebar.slider("Frame index", 0, max_frame, st.session_state.frame)
st.session_state.frame = frame

# autoplay loop: increment frame and rerun quickly
if st.session_state.playing:
    if st.session_state.frame < max_frame:
        st.session_state.frame += 1
        # small pause for animation speed
        time.sleep(0.12)
        st.experimental_rerun()
    else:
        st.session_state.playing = False

# -----------------------
# Show slice selection & plot
# -----------------------
time_h, temp = results[st.session_state.frame]
st.sidebar.markdown(f"Frame time: **{time_h:.3f} h**  (dt used = {dt:.4f} s)")

axis = st.sidebar.radio("Slice axis", ["X (length)", "Y (width)", "Z (height)"])
# allow choosing slice index
if axis.startswith("X"):
    slice_idx = st.sidebar.slider("X index (0..nx-1)", 0, nx-1, nx//2)
    slice_data = temp[:, :, slice_idx]  # shape (nz, ny) -> (height, width)
    x = np.linspace(0, width, ny)
    y = np.linspace(0, height, nz)
    xlabel, ylabel = "Width (m)", "Height (m)"
    plot_z = slice_data.T  # orient to (y,x) correctly for plotting
elif axis.startswith("Y"):
    slice_idx = st.sidebar.slider("Y index (0..ny-1)", 0, ny-1, ny//2)
    slice_data = temp[:, slice_idx, :]  # (nz, nx)
    x = np.linspace(0, length, nx)
    y = np.linspace(0, height, nz)
    xlabel, ylabel = "Length (m)", "Height (m)"
    plot_z = slice_data.T
else:
    slice_idx = st.sidebar.slider("Z index (0..nz-1)", 0, nz-1, nz//2)
    slice_data = temp[slice_idx, :, :]  # (ny, nx)
    x = np.linspace(0, length, nx)
    y = np.linspace(0, width, ny)
    xlabel, ylabel = "Length (m)", "Width (m)"
    plot_z = slice_data  # already (ny, nx)

# Plot with Plotly if available, else Matplotlib fallback
if PLOTLY_AVAILABLE:
    fig = go.Figure(go.Heatmap(
        z=plot_z,
        x=x,
        y=y,
        colorscale="Inferno",
        colorbar=dict(title="°C")
    ))
    fig.update_layout(title=f"Slice ({axis}) at index {slice_idx} — t={time_h:.3f} h",
                      xaxis_title=xlabel, yaxis_title=ylabel, height=600)
    st.plotly_chart(fig, use_container_width=True)
else:
    fig2, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(plot_z, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()), aspect='auto', cmap='inferno')
    ax.set_title(f"Slice ({axis}) index {slice_idx} — t={time_h:.3f} h")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, label="°C")
    st.pyplot(fig2)

st.markdown("**Notes:** convection uses Newton's law (h). Voids are treated as air (low conductivity). "
            "If you tweak dt and grid size, watch the stability warning.")
