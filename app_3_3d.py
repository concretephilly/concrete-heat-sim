import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Hollow-Core Slab Heat Simulation (Isosurface)")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Geometry")
length = st.sidebar.number_input("Length (m)", 0.5, 12.0, 4.0)
width  = st.sidebar.number_input("Width (m)", 0.2, 4.0, 1.2)
height = st.sidebar.number_input("Height (m)", 0.05, 1.0, 0.25)

nx = st.sidebar.slider("Grid cells (length)", 12, 64, 28)
ny = st.sidebar.slider("Grid cells (width)", 8, 48, 16)
nz = st.sidebar.slider("Grid cells (height)", 6, 40, 10)

st.sidebar.header("Material & Environment")
initial_temp = st.sidebar.number_input("Initial temp (°C)", -10.0, 90.0, 20.0)
outside_temp = st.sidebar.number_input("Outside temp (°C)", -40.0, 60.0, 5.0)

rho_conc = st.sidebar.number_input("ρ concrete (kg/m³)", 1500.0, 3000.0, 2400.0)
cp_conc  = st.sidebar.number_input("cp concrete (J/kgK)", 400.0, 2000.0, 900.0)
k_conc   = st.sidebar.number_input("k concrete (W/mK)", 0.05, 5.0, 1.7)
h_coeff  = st.sidebar.slider("h convection (W/m²K)", 1, 50, 10)

st.sidebar.header("Hydration Heat")
cement_content = st.sidebar.number_input("Cement content (kg/m³)", 200.0, 600.0, 350.0)
Q_total = st.sidebar.number_input("Total hydration heat (J/kg cement)", 100e3, 600e3, 250e3)
k_rate = st.sidebar.number_input("Reaction rate constant (1/h)", 0.01, 2.0, 0.1)

st.sidebar.header("Time")
total_hours = st.sidebar.number_input("Total time (hours)", 0.01, 72.0, 24.0)
dt_user = st.sidebar.number_input("Requested dt (s)", 0.05, 600.0, 5.0)

# -------------------------
# Grid + stability
# -------------------------
dx, dy, dz = length/nx, width/ny, height/nz
k_air, rho_air, cp_air = 0.025, 1.225, 1005.0
alpha_conc = k_conc/(rho_conc*cp_conc)
alpha_air  = k_air/(rho_air*cp_air)
max_alpha = max(alpha_conc, alpha_air)
den = (1/dx**2 + 1/dy**2 + 1/dz**2)
dt_stable = 0.5/(max_alpha*den)
dt = min(dt_user, dt_stable*0.9)
nt = int(max(1, (total_hours*3600)/dt))

# -------------------------
# Geometry: hollow cores
# -------------------------
xs = np.linspace(0, length, nx)
ys = np.linspace(-width/2, width/2, ny)
zs = np.linspace(-height/2, height/2, nz)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

mask = np.ones((nx, ny, nz), dtype=bool)

# 5 evenly spaced circular cores
n_cores = 5
core_radius = 0.25 * height
centers = np.linspace(-width/2+width/(n_cores+1), width/2-width/(n_cores+1), n_cores)

for c in centers:
    dist2 = (Y - c)**2 + (Z - 0.0)**2
    mask &= (dist2 >= core_radius**2)

# Material properties
k_arr   = np.where(mask, k_conc, k_air)
rho_arr = np.where(mask, rho_conc, rho_air)
cp_arr  = np.where(mask, cp_conc, cp_air)
alpha_arr = k_arr/(rho_arr*cp_arr)

# -------------------------
# Hydration heat function
# -------------------------
def hydration_rate(t):
    t_h = t/3600.0
    dQdt = Q_total * k_rate * np.exp(-k_rate*t_h)  # J/kg/h
    dQdt /= 3600.0  # J/kg/s
    q_vol = cement_content * dQdt  # W/m³
    return q_vol

# -------------------------
# Simulation
# -------------------------
@st.cache_data(show_spinner=False)
def run_sim(alpha_arr, rho_arr, cp_arr, init_temp, outside_temp, h, dt, nt):
    T = np.full(alpha_arr.shape, init_temp, dtype=float)
    results = []
    store_every = max(1, nt//60)

    for step in range(nt):
        Tn = T.copy()
        lap = np.zeros_like(Tn)
        lap[1:-1,1:-1,1:-1] = (
            (Tn[2:,1:-1,1:-1] - 2*Tn[1:-1,1:-1,1:-1] + Tn[:-2,1:-1,1:-1])/dx**2 +
            (Tn[1:-1,2:,1:-1] - 2*Tn[1:-1,1:-1,1:-1] + Tn[1:-1,:-2,1:-1])/dy**2 +
            (Tn[1:-1,1:-1,2:] - 2*Tn[1:-1,1:-1,1:-1] + Tn[1:-1,1:-1,:-2])/dz**2
        )
        T = Tn + alpha_arr*dt*lap
        q = hydration_rate(step*dt)
        T += (q*dt)/(rho_arr*cp_arr)
        # convection
        T[0,:,:]   -= (h*dt/(rho_arr[0,:,:]*cp_arr[0,:,:]*dx))*(Tn[0,:,:]-outside_temp)
        T[-1,:,:]  -= (h*dt/(rho_arr[-1,:,:]*cp_arr[-1,:,:]*dx))*(Tn[-1,:,:]-outside_temp)
        T[:,0,:]   -= (h*dt/(rho_arr[:,0,:]*cp_arr[:,0,:]*dy))*(Tn[:,0,:]-outside_temp)
        T[:,-1,:]  -= (h*dt/(rho_arr[:,-1,:]*cp_arr[:,-1,:]*dy))*(Tn[:,-1,:]-outside_temp)
        T[:,:,0]   -= (h*dt/(rho_arr[:,:,0]*cp_arr[:,:,0]*dz))*(Tn[:,:,0]-outside_temp)
        T[:,:,-1]  -= (h*dt/(rho_arr[:,:,-1]*cp_arr[:,:,-1]*dz))*(Tn[:,:,-1]-outside_temp)

        if step % store_every == 0:
            results.append((step*dt/3600, T.copy()))
    return results

snapshots = run_sim(alpha_arr, rho_arr, cp_arr,
                    initial_temp, outside_temp, h_coeff, dt, nt)

# -------------------------
# Slider + Visualization
# -------------------------
frame = st.slider("Frame", 0, len(snapshots)-1, 0)
time_h, Tcurr = snapshots[frame]

# Mask out voids
Tplot = Tcurr.copy()
Tplot[~mask] = np.nan

fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=Tplot.flatten(),
    isomin=np.nanmin(Tplot),
    isomax=np.nanmax(Tplot),
    surface=dict(show=True, fill=0.9),
    caps=dict(x_show=False, y_show=False, z_show=False),
    colorscale="Inferno",
    showscale=True,
    colorbar=dict(title="Temperature (°C)")
))
fig.update_layout(
    title=f"t = {time_h:.2f} h",
    scene_aspectmode="manual",
    scene_aspectratio=dict(x=length, y=width, z=height),
    scene=dict(
        xaxis_title="Length (m)",
        yaxis_title="Width (m)",
        zaxis_title="Height (m)"
    ),
    height=750
)
st.plotly_chart(fig, use_container_width=True)
