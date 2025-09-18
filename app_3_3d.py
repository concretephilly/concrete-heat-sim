import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("3D Hollow-Core Concrete Slab Heat Simulation")

# --- User Inputs ---
st.sidebar.header("Geometry")
nx = st.sidebar.slider("Grid size (x)", 10, 60, 30)
ny = st.sidebar.slider("Grid size (y)", 10, 60, 30)
nz = st.sidebar.slider("Grid size (z)", 5, 30, 10)
length = st.sidebar.number_input("Length (m)", 0.5, 10.0, 4.0)
width = st.sidebar.number_input("Width (m)", 0.2, 2.0, 1.2)
height = st.sidebar.number_input("Height (m)", 0.1, 1.0, 0.25)

st.sidebar.header("Material & Environment")
initial_temp = st.sidebar.number_input("Initial concrete temp (°C)", 0.0, 50.0, 20.0)
outside_temp = st.sidebar.number_input("Outside temp (°C)", -20.0, 50.0, 5.0)
rho = st.sidebar.number_input("Concrete density (kg/m³)", 1500.0, 3000.0, 2400.0)
cp = st.sidebar.number_input("Concrete heat capacity (J/kgK)", 500.0, 2000.0, 900.0)
k_concrete = st.sidebar.number_input("Concrete conductivity (W/mK)", 0.1, 10.0, 1.7)
k_air = 0.025  # thermal conductivity of air
h = st.sidebar.slider("Convection coefficient h (W/m²K)", 1, 50, 10)

st.sidebar.header("Time settings")
total_time = st.sidebar.number_input("Total time (hours)", 0.1, 24.0, 2.0)
dt = st.sidebar.number_input("Time step (s)", 1.0, 600.0, 30.0)

# --- Setup grid ---
dx = length / nx
dy = width / ny
dz = height / nz
alpha_concrete = k_concrete / (rho * cp)
alpha_air = k_air / (1.2 * 1005)  # rough values for air

nt = int(total_time * 3600 / dt)

# --- Create hollow-core mask ---
X, Y = np.meshgrid(np.linspace(-width/2, width/2, ny), np.linspace(-height/2, height/2, nz))
mask = np.ones((nz, ny, nx), dtype=bool)  # True = concrete
n_cores = 4
core_radius = 0.15 * height
core_spacing = width / (n_cores + 1)
for i in range(n_cores):
    y_center = -width/2 + (i+1)*core_spacing
    mask[:, :, :] &= True  # default
    for z in range(nz):
        for y in range(ny):
            y_pos = np.linspace(-width/2, width/2, ny)[y]
            z_pos = np.linspace(-height/2, height/2, nz)[z]
            if (y_pos - y_center)**2 + z_pos**2 < core_radius**2:
                mask[z, y, :] = False  # hollow (air)

# --- Initialize temperature field ---
T = np.ones((nz, ny, nx)) * initial_temp

# --- Simulation ---
results = []
for n in range(nt):
    Tn = T.copy()
    # Loop interior (vectorization gets tricky with mask)
    for z in range(1, nz-1):
        for y in range(1, ny-1):
            for x in range(1, nx-1):
                if mask[z,y,x]:  # concrete
                    alpha = alpha_concrete
                else:  # air void
                    alpha = alpha_air
                T[z,y,x] = (
                    Tn[z,y,x] +
                    alpha*dt*(
                        (Tn[z,y,x+1] - 2*Tn[z,y,x] + Tn[z,y,x-1]) / dx**2 +
                        (Tn[z,y+1,x] - 2*Tn[z,y,x] + Tn[z,y-1,x]) / dy**2 +
                        (Tn[z+1,y,x] - 2*Tn[z,y,x] + Tn[z-1,y,x]) / dz**2
                    )
                )

    # Convection at boundaries (applied to all exposed cells)
    T[0,:,:]   -= (h*dt/(rho*cp*dz))*(Tn[0,:,:]   - outside_temp)
    T[-1,:,:]  -= (h*dt/(rho*cp*dz))*(Tn[-1,:,:]  - outside_temp)
    T[:,0,:]   -= (h*dt/(rho*cp*dy))*(Tn[:,0,:]   - outside_temp)
    T[:,-1,:]  -= (h*dt/(rho*cp*dy))*(Tn[:,-1,:]  - outside_temp)
    T[:,:,0]   -= (h*dt/(rho*cp*dx))*(Tn[:,:,0]   - outside_temp)
    T[:,:,-1]  -= (h*dt/(rho*cp*dx))*(Tn[:,:,-1]  - outside_temp)

    if n % 5 == 0:
        results.append((n*dt/3600, T.copy()))

# --- Playback controls ---
st.sidebar.header("Playback")
frame = st.sidebar.slider("Time step", 0, len(results)-1, 0)
time_h, temp = results[frame]

# --- Slice viewer ---
st.sidebar.header("Slice options")
axis = st.sidebar.radio("Slice axis", ["X", "Y", "Z"])
if axis == "X":
    pos = nx // 2
    slice_data = temp[:,:,pos]
    x = np.linspace(0, width, ny)
    y = np.linspace(0, height, nz)
    xlabel, ylabel = "Width (m)", "Height (m)"
elif axis == "Y":
    pos = ny // 2
    slice_data = temp[:,pos,:]
    x = np.linspace(0, length, nx)
    y = np.linspace(0, height, nz)
    xlabel, ylabel = "Length (m)", "Height (m)"
else:  # Z
    pos = nz // 2
    slice_data = temp[pos,:,:]
    x = np.linspace(0, length, nx)
    y = np.linspace(0, width, ny)
    xlabel, ylabel = "Length (m)", "Width (m)"

# --- Plot slice ---
fig = go.Figure(data=go.Heatmap(
    z=slice_data,
    x=x,
    y=y,
    colorscale="Inferno"
))
fig.update_layout(
    title=f"{axis}-slice at t={time_h:.2f} h",
    xaxis_title=xlabel,
    yaxis_title=ylabel
)
st.plotly_chart(fig, use_container_width=True)
