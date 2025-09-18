# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("3D Hollow-Core Slab Heat Transfer + Hydration")

# -------------------------
# Sidebar controls with tooltips
# -------------------------
st.sidebar.header("Grid & Geometry")
nx = st.sidebar.slider("Cells (length, x)", 12, 64, 28,
    help="Number of grid cells along the slab length. More = finer resolution but slower.")
ny = st.sidebar.slider("Cells (width, y)", 8, 48, 16,
    help="Number of grid cells along the slab width.")
nz = st.sidebar.slider("Cells (height, z)", 6, 40, 10,
    help="Number of grid cells along the slab thickness.")

length = st.sidebar.number_input("Length (m)", 0.5, 12.0, 4.0,
    help="Total length of the slab in meters.")
width  = st.sidebar.number_input("Width (m)", 0.2, 4.0, 1.2,
    help="Total width of the slab in meters.")
height = st.sidebar.number_input("Height (m)", 0.05, 1.0, 0.25,
    help="Total thickness of the slab in meters.")

st.sidebar.header("Material & Environment")
initial_temp = st.sidebar.number_input("Initial temp (°C)", -10.0, 90.0, 20.0,
    help="Starting slab temperature before cooling/heating.")
outside_temp = st.sidebar.number_input("Outside temp (°C)", -40.0, 60.0, 5.0,
    help="Ambient air temperature around the slab.")

rho_conc = st.sidebar.number_input("ρ concrete (kg/m³)", 1500.0, 3000.0, 2400.0,
    help="Density of concrete. Typical ≈ 2400 kg/m³.")
cp_conc  = st.sidebar.number_input("cp concrete (J/kgK)", 400.0, 2000.0, 900.0,
    help="Specific heat capacity of concrete. Typical ≈ 900 J/kgK.")
k_conc   = st.sidebar.number_input("k concrete (W/mK)", 0.05, 5.0, 1.7,
    help="Thermal conductivity of concrete. Typical ≈ 1.7 W/mK.")
h_coeff  = st.sidebar.slider("h convection (W/m²K)", 1, 50, 10,
    help="Convection coefficient between slab surface and air. Higher = faster heat exchange.")

st.sidebar.header("Hydration Heat")
cement_content = st.sidebar.number_input("Cement content (kg/m³)", 200.0, 600.0, 350.0,
    help="Cement per cubic meter of concrete.")
Q_total = st.sidebar.number_input("Total hydration heat (J/kg cement)", 100e3, 600e3, 250e3,
    help="Total heat released per kg of cement (≈ 250–400 kJ/kg typical).")
k_rate = st.sidebar.number_input("Reaction rate constant (1/h)", 0.01, 2.0, 0.1,
    help="Controls how fast the hydration progresses. Larger = faster heat release.")

st.sidebar.header("Hollow Cores")
n_cores = st.sidebar.slider("Number of cores across width", 0, 6, 3,
    help="Number of circular voids spanning the slab width.")
core_radius_frac = st.sidebar.slider("Core radius (fraction of height)", 0.05, 0.5, 0.25,
    help="Core radius as a fraction of slab thickness.")

st.sidebar.header("Time")
total_hours = st.sidebar.number_input("Total time (hours)", 0.01, 72.0, 24.0,
    help="Total simulated duration of heat transfer.")
dt_user = st.sidebar.number_input("Requested dt (s)", 0.05, 600.0, 5.0,
    help="Solver timestep. Smaller = more accurate but slower.")

# -------------------------
# Derived params
# -------------------------
dx, dy, dz = length/nx, width/ny, height/nz
k_air, rho_air, cp_air = 0.025, 1.225, 1005.0
alpha_conc = k_conc/(rho_conc*cp_conc)
alpha_air  = k_air/(rho_air*cp_air)
max_alpha = max(alpha_conc, alpha_air)
den = (1/dx**2 + 1/dy**2 + 1/dz**2)
dt_stable = 0.5/(max_alpha*den)

if dt_user > dt_stable:
    st.warning(f"Requested dt={dt_user:.4g}s > stable {dt_stable:.4g}s, using {dt_stable*0.9:.4g}s.")
dt = min(dt_user, dt_stable*0.9)
nt = int(max(1, (total_hours*3600)/dt))

# -------------------------
# Geometry + hollow cores
# -------------------------
xs = np.linspace(0, length, nx)
ys = np.linspace(-width/2, width/2, ny)
zs = np.linspace(-height/2, height/2, nz)
Z, Y = np.meshgrid(zs, ys, indexing="xy")
Z = Z.T; Y = Y.T
mask2d = np.ones((nz, ny), dtype=bool)

if n_cores > 0:
    core_radius = core_radius_frac * height
    centers = np.linspace(-width/2+width/(n_cores+1), width/2-width/(n_cores+1), n_cores)
    for c in centers:
        dist2 = (Y-c)**2 + (Z-0)**2
        mask2d &= (dist2 >= core_radius**2)

mask = np.repeat(mask2d[:, :, np.newaxis], nx, axis=2)
k_arr   = np.where(mask, k_conc, k_air)
rho_arr = np.where(mask, rho_conc, rho_air)
cp_arr  = np.where(mask, cp_conc, cp_air)
alpha_arr = k_arr/(rho_arr*cp_arr)

# -------------------------
# Hydration heat function
# -------------------------
def hydration_rate(t):
    """Heat generation rate per m³ of concrete at time t (s)."""
    t_h = t/3600.0
    dQdt = Q_total * k_rate * np.exp(-k_rate*t_h)  # J/kg/h
    dQdt /= 3600.0  # J/kg/s
    q_vol = cement_content * dQdt  # W/m³
    return q_vol

# -------------------------
# Simulation
# -------------------------
@st.cache_data(show_spinner=False)
def run_sim(nx, ny, nz, dx, dy, dz, alpha_arr, rho_arr, cp_arr,
            init_temp, outside_temp, h, dt, nt, store_max=120):
    T = np.full((nz, ny, nx), init_temp, dtype=float)
    results = []
    store_every = max(1, nt//store_max)

    for step in range(nt):
        Tn = T.copy()
        lap = np.zeros_like(Tn)
        lap[1:-1,1:-1,1:-1] = (
            (Tn[1:-1,1:-1,2:] - 2*Tn[1:-1,1:-1,1:-1] + Tn[1:-1,1:-1,:-2])/dx**2 +
            (Tn[1:-1,2:,1:-1] - 2*Tn[1:-1,1:-1,1:-1] + Tn[1:-1,:-2,1:-1])/dy**2 +
            (Tn[2:,1:-1,1:-1] - 2*Tn[1:-1,1:-1,1:-1] + Tn[:-2,1:-1,1:-1])/dz**2
        )
        # heat conduction
        T = Tn + alpha_arr*dt*lap

        # hydration heat
        q = hydration_rate(step*dt)
        T += (q*dt)/(rho_arr*cp_arr)

        # convection boundaries
        T[:,:,0]  -= (h*dt/(rho_arr[:,:,0]*cp_arr[:,:,0]*dx))*(Tn[:,:,0]-outside_temp)
        T[:,:,-1] -= (h*dt/(rho_arr[:,:,-1]*cp_arr[:,:,-1]*dx))*(Tn[:,:,-1]-outside_temp)
        T[:,0,:]  -= (h*dt/(rho_arr[:,0,:]*cp_arr[:,0,:]*dy))*(Tn[:,0,:]-outside_temp)
        T[:,-1,:] -= (h*dt/(rho_arr[:,-1,:]*cp_arr[:,-1,:]*dy))*(Tn[:,-1,:]-outside_temp)
        T[0,:,:]  -= (h*dt/(rho_arr[0,:,:]*cp_arr[0,:,:]*dz))*(Tn[0,:,:]-outside_temp)
        T[-1,:,:] -= (h*dt/(rho_arr[-1,:,:]*cp_arr[-1,:,:]*dz))*(Tn[-1,:,:]-outside_temp)

        if step % store_every == 0:
            results.append((step*dt/3600, T.copy()))
    return results

snapshots = run_sim(nx, ny, nz, dx, dy, dz, alpha_arr, rho_arr, cp_arr,
                    initial_temp, outside_temp, h_coeff, dt, nt)

# -------------------------
# Slider playback
# -------------------------
frame = st.slider("Frame", 0, len(snapshots)-1, 0,
    help="Select a simulation snapshot in time.")
time_h, Tcurr = snapshots[frame]

X_coords, Y_coords, Z_coords = np.meshgrid(xs, ys, zs, indexing="ij")
fig = go.Figure(data=go.Volume(
    x=X_coords.flatten(),
    y=Y_coords.flatten(),
    z=Z_coords.flatten(),
    value=Tcurr.flatten(),
    opacity=0.1, surface_count=20,
    colorscale="Inferno",
    caps=dict(x_show=False, y_show=False, z_show=False),
    showscale=True,
    colorbar=dict(title="Temperature (°C)")
))
fig.update_layout(
    title=f"t = {time_h:.2f} h",
    scene=dict(xaxis_title="Length (m)", yaxis_title="Width (m)", zaxis_title="Height (m)"),
    height=750
)
st.plotly_chart(fig, use_container_width=True)
