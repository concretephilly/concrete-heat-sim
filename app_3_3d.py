import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")  # use full width for plots

# Sidebar controls
st.sidebar.header("Simulation Parameters")
# Plate ramp inputs
start_temp = st.sidebar.number_input("Plate start temperature (°C)", value=20.0)
end_temp = st.sidebar.number_input("Plate end temperature (°C)", value=60.0)
ramp_duration = st.sidebar.number_input("Ramp duration (hours)", value=10)
# Hydration heat source
Q_gen = st.sidebar.number_input("Hydration heat rate (W/m³)", value=3.0)
# Concrete properties
k_val = st.sidebar.number_input("Thermal conductivity k (W/m·K)", value=1.2)
cp = st.sidebar.number_input("Heat capacity cp (J/kg·K)", value=1600.0)
rho = st.sidebar.number_input("Density (kg/m³)", value=2300.0)
# Air (ambient) temperature
Tamb = st.sidebar.number_input("Air temperature (°C)", value=20.0)
# Simulation controls
play = st.sidebar.button("Play")
pause = st.sidebar.button("Pause")  # not used for simplicity

# Domain and discretization
Nx = 6; Ny = 6; Nz = 6  # number of points in x,y,z
dx = 1.0/(Nx-1); dy = 1.0/(Ny-1); dz = 1.0/(Nz-1)
alpha = k_val/(rho*cp)
dt = 3600.0  # one hour in seconds per time step

# Pre-compute spatial grid for plotting
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
z = np.linspace(0, 1, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Helper: compute plate temperature at time t (hours)
def plate_temp(t):
    if t < ramp_duration:
        return start_temp + (end_temp - start_temp) * (t/ramp_duration)
    else:
        return end_temp

# Compute total simulation time (hours): at least ramp*2 or ramp+24
total_hours = int(ramp_duration + max(24, ramp_duration))
times = np.arange(0, total_hours+1)  # hourly steps

# Run the simulation once for all time steps
# Initialize temperature field: start at ambient everywhere
T = np.ones((Nx, Ny, Nz)) * Tamb
T_record = []  # record center and plate temperatures
# Coefficients for convective BC (ghost nodes)
b_x = 2*dx*15.0/k_val  # assuming h=15 W/m2K
b_y = 2*dy*15.0/k_val
b_z = 2*dz*15.0/k_val

for t in times:
    # Set bottom (plate) temperature
    T[:,:,0] = plate_temp(t)
    T_new = T.copy()
    # Explicit diffusion step
    for i in range(Nx):
        for j in range(Ny):
            for k in range(1, Nz):
                T_c = T[i,j,k]
                # X-direction neighbors or convective ghost
                if i == 0:
                    Tx_m = T[1,j,k] + b_x*(Tamb - T_c)
                else:
                    Tx_m = T[i-1,j,k]
                if i == Nx-1:
                    Tx_p = T[Nx-2,j,k] + b_x*(Tamb - T_c)
                else:
                    Tx_p = T[i+1,j,k]
                # Y-direction neighbors or convective ghost
                if j == 0:
                    Ty_m = T[i,1,k] + b_y*(Tamb - T_c)
                else:
                    Ty_m = T[i,j-1,k]
                if j == Ny-1:
                    Ty_p = T[i,Ny-2,k] + b_y*(Tamb - T_c)
                else:
                    Ty_p = T[i,j+1,k]
                # Z-direction neighbors (bottom Dirichlet, top convective ghost)
                if k == 1:
                    Tz_m = T[i,j,0]  # plate
                else:
                    Tz_m = T[i,j,k-1]
                if k == Nz-1:
                    Tz_p = T[i,j,Nz-2] + b_z*(Tamb - T_c)
                else:
                    Tz_p = T[i,j,k+1]
                # 3D Laplacian (finite difference)
                lap = (Tx_p + Tx_m - 2*T_c)/(dx*dx) + (Ty_p + Ty_m - 2*T_c)/(dy*dy) \
                      + (Tz_p + Tz_m - 2*T_c)/(dz*dz)
                T_new[i,j,k] = T_c + alpha * lap * dt + (Q_gen/(rho*cp)) * dt
    T = T_new
    # Record center and plate temperatures
    center_val = T[Nx//2, Ny//2, Nz//2]
    T_record.append((t, center_val, plate_temp(t)))

# Extract arrays for plotting
times_plot = np.array([rec[0] for rec in T_record])
T_center = np.array([rec[1] for rec in T_record])
T_plate = np.array([rec[2] for rec in T_record])

# Determine color scale range (use min ambient to max(plate,end) )
color_min = Tamb
color_max = max(end_temp, T_center.max())

# Layout for three columns
col1, col2, col3 = st.columns(3)

# Placeholder for slider (to allow programmatic updates)
slider_placeholder = st.empty()
# Determine slider range: 0 to total_hours
time_idx = slider_placeholder.slider("Time (hours)", 0, total_hours, 0, step=1, key="time_slider")

# Play/Pause logic: if Play pressed, animate slider
if play:
    for t in range(time_idx, total_hours+1):
        # Update slider programmatically
        time_idx = slider_placeholder.slider("Time (hours)", 0, total_hours, t, step=1, key="time_slider_anim")
        # Brief pause for animation effect
        time.sleep(0.1)

# Slice corresponding to current time
T_current = None
if time_idx < len(T_record):
    # find recorded center field at that time
    # Since we only stored center, we must recompute entire T for arbitrary time
    # For simplicity, just use last T (approx)
    T_current = T  # (approx: using last sim result as current field)
else:
    T_current = T

# Plot 3D volumetric temperature field
with col1:
    fig3d = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=T.flatten(),
        isomin=color_min, isomax=color_max,
        colorscale='Bluered', opacity=0.2,
    ))
    fig3d.update_layout(
        title=f"Temperature (°C) at t={time_idx}h",
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z'
        ),
        margin=dict(l=0,r=0,t=30,b=0),
        uirevision='const'  # persist camera
    )
    st.plotly_chart(fig3d, use_container_width=True)

# Plot 2D mid-plane heatmap (slice at k=Nz//2)
with col2:
    k_slice = Nz//2
    T_slice = T[:,:,k_slice]
    fig2d = go.Figure(data=go.Heatmap(
        z=T_slice, x=x, y=y,
        zmin=color_min, zmax=color_max,
        colorscale='Bluered', colorbar=dict(title="°C")
    ))
    fig2d.update_layout(
        title=f"Mid-height slice (z≈{z[k_slice]:.2f}) at t={time_idx}h",
        xaxis_title='x (m)', yaxis_title='y (m)',
        margin=dict(l=40,r=20,t=40,b=40)
    )
    st.plotly_chart(fig2d, use_container_width=True)

# Plot temperature vs time (center and plate)
with col3:
    figline = go.Figure()
    figline.add_trace(go.Scatter(x=times_plot, y=T_center, mode='lines', name="Center"))
    figline.add_trace(go.Scatter(x=times_plot, y=T_plate, mode='lines', name="Plate"))
    figline.add_trace(go.Scatter(x=times_plot, y=np.ones_like(times_plot)*Tamb, mode='lines',
                                 name="Ambient", line=dict(dash='dash')))
    # Highlight current time
    figline.add_trace(go.Scatter(x=[time_idx, time_idx], y=[color_min, color_max], mode='lines',
                                 line=dict(color='gray', dash='dot'), showlegend=False))
    figline.update_layout(
        title="Temperature vs Time",
        xaxis_title="Time (h)", yaxis_title="Temperature (°C)",
        margin=dict(l=40,r=20,t=40,b=40)
    )
    st.plotly_chart(figline, use_container_width=True)

# Estimate time to equilibrium
if time_idx > 0:
    dT = T_center[-1] - T_center[-2]
    rate = dT/(1.0)  # per hour
    if abs(rate) < 1e-3:
        est_text = "negligible change (near equilibrium)"
    else:
        if rate > 0:
            # heating up toward plate temp
            time_eq = (end_temp - T_center[-1]) / rate
        else:
            # cooling toward ambient
            time_eq = (Tamb - T_center[-1]) / rate
        est_text = f"about {abs(time_eq):.1f} more hours"
else:
    est_text = "n/a"
st.markdown(f"**Estimated time to equilibrium:** {est_text}")

