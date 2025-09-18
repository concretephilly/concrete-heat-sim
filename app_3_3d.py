import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import cm

st.set_page_config(layout="wide", page_title="Hollow-core Slab Curing Simulator")

st.title("Hollow-core Concrete Slab Curing Simulator")
st.markdown("""
This app simulates transient heat conduction and hydration heat release in a hollow-core concrete slab sitting on a heated casting bed.
Physics is simplified (2D cross-section across width × thickness, transient heat equation with internal heat generation).
""")

# --- Defaults & Material properties (CEM III realistic defaults) ---
rho_default = 2400.0  # kg/m3
k_default = 2.0       # W/mK
cp_default = 900.0    # J/kgK
heat_hydration_per_kg = 250e3  # J/kg cement (250 kJ/kg)
cement_content_default = 300.0  # kg cement per m3 concrete
Q_total_per_m3_default = heat_hydration_per_kg * cement_content_default  # J per m3

# --- Sidebar inputs ---
st.sidebar.header("Simulation inputs")
col = st.sidebar
casting_bed_temp = col.number_input("Casting bed temperature (°C)", value=40.0, step=1.0)
air_temp = col.number_input("Air temperature (°C)", value=20.0, step=1.0)

col.subheader("Concrete / hydration parameters")
rho = col.number_input("Density (kg/m³)", value=rho_default)
k = col.number_input("Thermal conductivity (W/m·K)", value=k_default)
cp = col.number_input("Specific heat capacity (J/kg·K)", value=cp_default)
heat_hydration = col.number_input("Heat of hydration (J/kg cement)", value=heat_hydration_per_kg)
cement_content = col.number_input("Cement content (kg/m³)", value=cement_content_default)
# total energy per m3
Q_total_per_m3 = heat_hydration * cement_content

col.subheader("Slab geometry")
length = col.number_input("Length (m)", value=4.0, step=0.1)
width = col.number_input("Width (m)", value=2.0, step=0.1)
thickness = col.number_input("Thickness (m)", value=0.25, step=0.01)

col.subheader("Hollow cores")
num_cores = col.slider("Number of cores across width", min_value=0, max_value=10, value=3)
core_diameter = col.number_input("Core diameter (m)", value=0.2, step=0.01)
core_cover = col.number_input("Concrete cover each side of cores (m)", value=0.05, step=0.01)

col.subheader("Simulation control")
sim_hours = col.number_input("Simulation time (hours)", value=72.0, step=1.0)
resolution = col.slider("Cross-section resolution (cells across thickness)", min_value=20, max_value=200, value=80)
# let width resolution scale with ratio
res_x = int(max(20, round(resolution * (width / thickness))))

col.subheader("Hydration kinetics (simple exp model)")
tau_hours = col.number_input("Characteristic hydration time (hours)", value=24.0, step=1.0)
# convective heat transfer on top
h_conv = col.number_input("Top convective coefficient h (W/m²K)", value=10.0)

# derived
alpha = k / (rho * cp)  # thermal diffusivity (m2/s)

st.sidebar.markdown(f"**Derived thermal diffusivity:** {alpha:.2e} m²/s")
st.sidebar.markdown(f"**Total hydration heat per m³:** {Q_total_per_m3/1e6:.3f} MJ/m³")

run_button = st.sidebar.button("Run simulation")

# Provide a quick explanation of the numerical scheme
st.markdown("""
**Notes on the model**
- 2D explicit finite-difference transient heat conduction across width × thickness.
- Boundary conditions: bottom is in contact with casting bed (Dirichlet, constant temperature); top convects to air (Neumann with h); sides insulated.
- Hollow cores are modelled as air-filled regions with air thermal properties (low k, low rho*cp), and they *do not* generate hydration heat.
- Hydration heat is applied as volumetric heat generation with a simple exponential kinetics: q(t) = (Q_tot / tau) * exp(-t/tau) where Q_tot is per-m³.
""")

if not run_button:
    st.info("Adjust parameters on the left and click **Run simulation**")
    st.stop()

# --- Build computational grid (2D cross-section: x across width, y through thickness) ---
nx = res_x
ny = resolution
Lx = width
Ly = thickness
dx = Lx / nx
dy = Ly / ny
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# hollow core geometry (positions across width centered)
core_centers = []
if num_cores > 0:
    usable_width = Lx - 2 * core_cover
    if num_cores == 1:
        centers_x = [Lx/2]
    else:
        centers_x = np.linspace(core_cover + core_diameter/2, Lx - (core_cover + core_diameter/2), num_cores)
    for cx in centers_x:
        core_centers.append((cx, Ly/2))

# material map: 1 = concrete, 0 = air (core)
material = np.ones((ny, nx), dtype=int)
for (cx, cy) in core_centers:
    r = core_diameter / 2.0
    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    material[mask] = 0

# properties arrays (per cell)
air_k = 0.026
air_rho = 1.2
air_cp = 1005.0

k_mat = np.where(material == 1, k, air_k)
rho_mat = np.where(material == 1, rho, air_rho)
cp_mat = np.where(material == 1, cp, air_cp)

# initial temperature: assume freshly cast at bed temperature at bottom grading to air at top? Start uniform at casting bed temp
T0 = casting_bed_temp * np.ones_like(k_mat)

# time stepping
total_seconds = sim_hours * 3600.0
tau = tau_hours * 3600.0
Q_tot = Q_total_per_m3  # J/m3

# Stability (explicit 2D) dt <= 1/4 * min(dx^2,dy^2)/alpha
min_space = min(dx, dy)
alpha_local = k_mat / (rho_mat * cp_mat)
alpha_max = np.max(alpha_local)
if alpha_max <= 0:
    alpha_max = alpha

dt_stable = 0.25 * (min_space ** 2) / alpha_max
# choose dt to target about 500-2000 timesteps if possible
target_steps = 800
dt = min(dt_stable, total_seconds / target_steps)
num_steps = int(np.ceil(total_seconds / dt))
# safety cap
if num_steps > 20000:
    num_steps = 20000
    dt = total_seconds / num_steps

st.sidebar.markdown(f"Estimated timesteps: {num_steps}")

# Precompute coefficients for explicit FD (uniform grid)
T = T0.copy().astype(float)
T_new = T.copy()
avg_temps = []
times = []

# For efficiency, precompute i,j neighbor indices vectorized
# We'll compute Laplacian with second-order central; treat k varying by harmonic averaging at interfaces

# helper to compute harmonic conductivity between neighbor cells

# Precompute position arrays for boundaries
nyi, nxi = ny, nx

# Precompute cell volumes (1 m in length direction) -> area in cross-section * length
cell_area = dx * dy  # m2 per unit length in length direction

# vectorize indices for interior updates
for step in range(num_steps):
    t = step * dt
    # hydration generation per m3 (only in concrete cells)
    q_gen_rate = (Q_tot / tau) * np.exp(-t / tau)  # J/(m3·s)
    q = q_gen_rate * (material == 1)  # zero in air

    # compute conduction term using finite differences with variable k (harmonic average)
    # pad arrays to handle boundaries
    T_pad = np.pad(T, ((1,1),(1,1)), mode='edge')
    k_pad = np.pad(k_mat, ((1,1),(1,1)), mode='edge')
    rho_cp = rho_mat * cp_mat

    # indices
    # T at center is T_pad[1:-1,1:-1]
    Tc = T_pad[1:-1,1:-1]
    Txp = T_pad[1:-1,2:]
    Txm = T_pad[1:-1,0:-2]
    Typ = T_pad[2:,1:-1]
    Tym = T_pad[0:-2,1:-1]

    kcxp = (k_pad[1:-1,1:-1] * k_pad[1:-1,2:]) / (k_pad[1:-1,1:-1] + k_pad[1:-1,2:] + 1e-12)
    kcxm = (k_pad[1:-1,1:-1] * k_pad[1:-1,0:-2]) / (k_pad[1:-1,1:-1] + k_pad[1:-1,0:-2] + 1e-12)
    kcyp = (k_pad[1:-1,1:-1] * k_pad[2:,1:-1]) / (k_pad[1:-1,1:-1] + k_pad[2:,1:-1] + 1e-12)
    kcym = (k_pad[1:-1,1:-1] * k_pad[0:-2,1:-1]) / (k_pad[1:-1,1:-1] + k_pad[0:-2,1:-1] + 1e-12)

    # second derivative components (flux divergence) per unit volume (W/m3)
    fx = (kcxp * (Txp - Tc) - kcxm * (Tc - Txm)) / (dx * dx)
    fy = (kcyp * (Typ - Tc) - kcym * (Tc - Tym)) / (dy * dy)
    div_q = fx + fy  # W/m3

    # top boundary (y = Ly) index corresponds to last row (y-index ny-1)
    # apply convective boundary at top: implemented via ghost cell in pad above top row
    # For the top row, replace Typ for that row using convective flux: -k*(dT/dy) = h*(T_surface - T_air)
    # Implement by modifying fy for top-most interior nodes

    # Compute convective correction for top row
    top_row = ny - 1
    # indices in pad: center at [1+top_row,...]
    i_center = 1 + top_row
    j_all = slice(1,1+nx)
    # surface temperature Ts = Tc[top_row,:]
    Ts = T_pad[i_center, j_all]
    # conductive flux to ghost: k * (Tghost - Ts)/dy; set so that -k*(Tghost-Ts)/dy = h*(Ts - T_air) -> Tghost = Ts - (h*dy/k)*(Ts - T_air)
    k_surface = k_pad[i_center, j_all]
    Tghost = Ts - (h_conv * dy / (k_surface + 1e-12)) * (Ts - air_temp)
    Typ_top = Tghost
    # Compute fy correction for top row explicitly for those cells
    # For top interior nodes, Typ used above should be Typ_top
    Typ_full = Typ.copy()
    Typ_full[top_row,:] = Typ_top
    kcyp_full = kcyp.copy()
    kcyp_full[top_row,:] = (k_pad[i_center,j_all] * k_pad[i_center+1,j_all]) / (k_pad[i_center,j_all] + k_pad[i_center+1,j_all] + 1e-12)

    # recompute fy for top row only
    fy_full = (kcyp_full * (Typ_full - Tc) - kcym * (Tc - Tym)) / (dy * dy)
    div_q[top_row,:] = fx[top_row,:] + fy_full[top_row,:]

    # bottom row: Dirichlet contact with casting bed (keep temperature equal to casting bed temperature)
    # We'll enforce after update.

    # explicit update: T_new = T + dt/(rho*cp) * (div_q + q)
    T_new = T + (dt / rho_cp) * (div_q + q)

    # enforce bottom Dirichlet (y=0)
    T_new[0,:] = casting_bed_temp
    # ensure air cavities don't generate heat (already zeroed) but allow them to exchange heat

    T = T_new

    # store averages per unit length (exclude cores or include? compute concrete volume avg)
    # average over all cells weighted by material volume
    avg = np.sum(T * cell_area) / (nx * ny * cell_area)
    avg_temps.append(avg)
    times.append(t / 3600.0)

# --- Prepare outputs ---
st.markdown("## Results")

# Left column: 3D geometry
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("3D slab geometry (length view)")
    # Create box mesh for slab
    L = length
    W = width
    H = thickness
    # create a semi-transparent box
    # box vertices
    verts = np.array([
        [0,0,0], [L,0,0], [L,W,0], [0,W,0],
        [0,0,H], [L,0,H], [L,W,H], [0,W,H]
    ])
    i = [0,0,0,1,4,5,6,7,0,1,2,3]
    j = [1,2,3,2,5,6,7,4,4,5,6,7]
    k = [4,5,6,7,0,1,2,3,1,2,3,0]
    mesh = go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=i, j=j, k=k,
        opacity=0.25,
        color='lightgray',
        name='Slab body'
    )

    data = [mesh]
    # add cylinders for cores (as parametric surfaces)
    for (cx, cy) in core_centers:
        # cylinder axis along length (x direction) center located at cx on y, cy on z? careful coordinates: we'll put cylinder along length, so param coords u along length, v angle
        u = np.linspace(0, L, 20)
        v = np.linspace(0, 2*np.pi, 24)
        U, V = np.meshgrid(u, v)
        Xc = U
        Yc = cx + (core_diameter/2.0) * np.cos(V)
        Zc = cy + (core_diameter/2.0) * np.sin(V)
        cyl = go.Surface(x=Xc, y=Yc, z=Zc, showscale=False, opacity=1.0, colorscale=[[0, 'white'], [1,'white']], name='core')
        data.append(cyl)

    fig3d = go.Figure(data=data)
    fig3d.update_layout(scene=dict(aspectmode='data', xaxis_title='Length (m)', yaxis_title='Width (m)', zaxis_title='Thickness (m)'), height=500)
    st.plotly_chart(fig3d, use_container_width=True)

# Right column: heatmap and avg plot
with col2:
    st.subheader("2D cross-section temperature (width × thickness)")
    # prepare heatmap: Y vertical (thickness), X horizontal (width)
    # plot with origin at bottom (y=0 bottom)
    fig_heat = go.Figure()
    # mask cores by NaN to show white holes
    T_plot = T.copy()
    T_plot[material == 0] = np.nan
    fig_heat.add_trace(go.Heatmap(z=T_plot[::-1,:], x=x, y=y[::-1], colorscale='RdBu', zmin=min(casting_bed_temp, air_temp), zmax=np.max(T_plot[np.isfinite(T_plot)]), colorbar=dict(title='°C')))
    fig_heat.update_yaxes(title='Thickness (m)', autorange='reversed')
    fig_heat.update_xaxes(title='Width (m)')
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Average temperature in cross-section over time")
    fig_line = px.line(x=times, y=avg_temps, labels={'x':'Time (hours)','y':'Avg Temp (°C)'})
    st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")
st.markdown("**Tips / next steps**: You can increase resolution for finer results, add lengthwise conduction, or switch to an implicit solver for larger time steps.")

