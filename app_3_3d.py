# app.py
import streamlit as st
import numpy as np
import time

# Try to import Plotly; fallback is not provided because 3D viewport uses Plotly
try:
    import plotly.graph_objects as go
except Exception as e:
    st.error("Plotly is required to run this demo. Add 'plotly' to requirements.txt and redeploy.")
    st.stop()

st.set_page_config(layout="wide")
st.title("3D Voxel Demo — Hollow-Core Slab Heat Diffusion (Interactive)")

# -------------------------
# User controls (sidebar)
# -------------------------
st.sidebar.header("Grid & geometry (keep moderate sizes for cloud)")
nx = st.sidebar.slider("cells (length, x)", 12, 64, 28)
ny = st.sidebar.slider("cells (width, y)", 8, 48, 16)
nz = st.sidebar.slider("cells (height, z)", 6, 40, 10)

length = st.sidebar.number_input("Length (m)", 0.5, 12.0, 4.0)
width  = st.sidebar.number_input("Width (m)", 0.2, 4.0, 1.2)
height = st.sidebar.number_input("Height (m)", 0.05, 1.0, 0.25)

st.sidebar.header("Material & environment")
initial_temp = st.sidebar.number_input("Initial temp (°C)", -10.0, 90.0, 20.0)
outside_temp = st.sidebar.number_input("Outside temp (°C)", -40.0, 60.0, 5.0)

rho_conc = st.sidebar.number_input("concrete density (kg/m³)", 1500.0, 3000.0, 2400.0)
cp_conc  = st.sidebar.number_input("concrete cp (J/kgK)", 400.0, 2000.0, 900.0)
k_conc   = st.sidebar.number_input("concrete k (W/mK)", 0.05, 5.0, 1.7)
h_coeff  = st.sidebar.slider("h (W/m²K) convection coeff", 1, 50, 10)

st.sidebar.header("Hollow core")
n_cores = st.sidebar.slider("number of cores across width", 0, 6, 3)
core_radius_frac = st.sidebar.slider("core radius (fraction of height)", 0.05, 0.5, 0.25)

st.sidebar.header("Time")
total_hours = st.sidebar.number_input("total time (hours)", 0.01, 48.0, 2.0)
dt_user     = st.sidebar.number_input("requested dt (s)", 0.05, 600.0, 5.0)

# -------------------------
# derived quantities
# -------------------------
dx = length / nx
dy = width / ny
dz = height / nz

# air properties (for voids)
k_air = 0.025
rho_air = 1.225
cp_air = 1005.0

# per-material alpha
alpha_conc = k_conc / (rho_conc * cp_conc)
alpha_air  = k_air   / (rho_air  * cp_air)

# stability estimate for explicit scheme (3D)
max_alpha = max(alpha_conc, alpha_air)
den = (1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2)
if den <= 0:
    dt_stable = dt_user
else:
    dt_stable = 0.5 / (max_alpha * den)

if dt_user > dt_stable:
    st.warning(f"Requested dt={dt_user:.4g}s unstable for explicit solver. Using dt={dt_stable*0.9:.6g}s.")
dt = min(dt_user, dt_stable * 0.9)

nt = int(max(1, (total_hours*3600.0) / dt))
if nt > 4000:
    st.warning("Large number of time steps — consider increasing dt or reducing total time / grid size.")

# -------------------------
# create voxel geometry & hollow-core mask
# -------------------------
xs = np.linspace(0, length, nx)
ys = np.linspace(-width/2, width/2, ny)
zs = np.linspace(-height/2, height/2, nz)

Z, Y = np.meshgrid(zs, ys, indexing='xy')  # shapes (ny, nz)
Z = Z.T
Y = Y.T

mask2d = np.ones((nz, ny), dtype=bool)
if n_cores > 0:
    core_radius = core_radius_frac * height
    centers = np.linspace(-width/2 + width/(n_cores+1),
                          width/2 - width/(n_cores+1), n_cores)
    for c in centers:
        dist2 = (Y - c)**2 + (Z - 0.0)**2
        mask2d &= (dist2 >= core_radius**2)

mask = np.repeat(mask2d[:, :, np.newaxis], nx, axis=2)

k_arr   = np.where(mask, k_conc, k_air)
rho_arr = np.where(mask, rho_conc, rho_air)
cp_arr  = np.where(mask, cp_conc, cp_air)
alpha_arr = k_arr / (rho_arr * cp_arr)

# -------------------------
# Simulation
# -------------------------
@st.cache_data(show_spinner=False)
def run_voxel_simulation(nx, ny, nz, dx, dy, dz, alpha_arr, rho_arr, cp_arr,
                         init_temp, outside_temp, h, dt, nt, store_max=180):
    T = np.full((nz, ny, nx), init_temp, dtype=float)
    results = []
    store_every = max(1, nt // store_max)

    for step in range(nt):
        Tn = T.copy()
        lap = np.zeros_like(Tn)

        # interior laplacian (vectorized) ✅ fixed commas
        lap[1:-1,1:-1,1:-1] = (
            (Tn[1:-1,1:-1,2:] - 2.0*Tn[1:-1,1:-1,1:-1] + Tn[1:-1,1:-1,:-2]) / dx**2 +
            (Tn[1:-1,2:,1:-1] - 2.0*Tn[1:-1,1:-1,1:-1] + Tn[1:-1,:-2,1:-1]) / dy**2 +
            (Tn[2:,1:-1,1:-1] - 2.0*Tn[1:-1,1:-1,1:-1] + Tn[:-2,1:-1,1:-1]) / dz**2
        )

        T = Tn + alpha_arr * dt * lap

        # convection on external faces
        T[:,:,0]  = Tn[:,:,0]  - (h*dt/(rho_arr[:,:,0]*cp_arr[:,:,0]*dx)) * (Tn[:,:,0]  - outside_temp)
        T[:,:,-1] = Tn[:,:,-1] - (h*dt/(rho_arr[:,:,-1]*cp_arr[:,:,-1]*dx)) * (Tn[:,:,-1] - outside_temp)
        T[:,0,:]  = Tn[:,0,:]  - (h*dt/(rho_arr[:,0,:]*cp_arr[:,0,:]*dy)) * (Tn[:,0,:]  - outside_temp)
        T[:,-1,:] = Tn[:,-1,:] - (h*dt/(rho_arr[:,-1,:]*cp_arr[:,-1,:]*dy)) * (Tn[:,-1,:] - outside_temp)
        T[0,:,:]  = Tn[0,:,:]  - (h*dt/(rho_arr[0,:,:]*cp_arr[0,:,:]*dz)) * (Tn[0,:,:]  - outside_temp)
        T[-1,:,:] = Tn[-1,:,:] - (h*dt/(rho_arr[-1,:,:]*cp_arr[-1,:,:]*dz)) * (Tn[-1,:,:] - outside_temp)

        if step % store_every == 0:
            results.append((step*dt/3600.0, T.copy()))
    return results

with st.spinner("Running simulation..."):
    snapshots = run_voxel_simulation(nx, ny, nz, dx, dy, dz, alpha_arr, rho_arr, cp_arr,
                                     initial_temp, outside_temp, h_coeff, dt, nt, store_max=160)

if not snapshots:
    st.error("No snapshots produced (check dt / total_time / grid).")
    st.stop()

# -------------------------
# Playback controls
# -------------------------
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

col1, col2 = st.columns([1, 3])
with col1:
    st.sidebar.header("Playback")
    if st.button("Prev"):
        st.session_state.frame_idx = max(0, st.session_state.frame_idx - 1)
    if st.button("Next"):
        st.session_state.frame_idx = min(len(snapshots)-1, st.session_state.frame_idx + 1)

    play = st.button("Play animation")
    if play:
        placeholder = st.empty()
        for i in range(st.session_state.frame_idx, len(snapshots)):
            st.session_state.frame_idx = i
            time_h, Tnow = snapshots[i]
            fig = go.Figure(data=go.Volume(
                x=np.repeat(xs, ny*nz),
                y=np.tile(np.repeat(ys, nz), nx),
                z=np.tile(np.tile(zs, ny), nx),
                value=Tnow.flatten(order='F'),
                opacity=0.1,
                surface_count=20,
                colorscale="Inferno",
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=True
            ))
            fig.update_layout(title=f"t = {time_h:.2f} h  (frame {i+1}/{len(snapshots)})",
                              scene=dict(xaxis_title='Length (m)',
                                         yaxis_title='Width (m)',
                                         zaxis_title='Height (m)'),
                              height=700)
            placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.12)
        placeholder.empty()

with col2:
    idx = st.session_state.frame_idx
    time_h, Tcurr = snapshots[idx]
    st.write(f"Frame {idx+1}/{len(snapshots)} · t = {time_h:.3f} h")

    X_coords = np.repeat(xs, ny*nz)
    Y_coords = np.tile(np.repeat(ys, nz), nx)
    Z_coords = np.tile(np.tile(zs, ny), nx)
    values = Tcurr.flatten(order='F')

    vol = go.Figure(data=go.Volume(
        x=X_coords, y=Y_coords, z=Z_coords, value=values,
        opacity=0.1, surface_count=20, colorscale="Inferno",
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=True
    ))
    vol.update_layout(title=f"3D temperature (t={time_h:.2f} h)",
                      scene=dict(xaxis_title='Length (m)',
                                 yaxis_title='Width (m)',
                                 zaxis_title='Height (m)'),
                      height=750)
    st.plotly_chart(vol, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Notes:**\n- This is a voxel-based demo (not finite-element). For real CAD meshes and FEM you'd need a mesh-based solver.\n- Keep grid sizes moderate (nx*ny*nz < ~60k) for Streamlit Cloud.\n- Voids are treated as air (low conductivity).")
