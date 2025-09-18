import streamlit as st
import numpy as np
import plotly.graph_objects as go

# -------------------------
# Heat transfer simulation
# -------------------------

def simulate_heat(length, width, height, nx, ny, nz, outside_temp, heat_rate, total_time, dt, k, rho, c, n_voids):
    dx = length / nx
    dy = width / ny
    dz = height / nz

    # Coordinates
    x = np.linspace(0, length, nx)
    y = np.linspace(0, width, ny)
    z = np.linspace(0, height, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Initialize temperature field
    T = np.ones((nx, ny, nz)) * outside_temp

    # Mask for concrete vs voids (hollow-cores)
    mask = np.ones_like(T, dtype=bool)

    void_radius = min(width / (2 * n_voids), height / 3)
    void_centers_y = np.linspace(width/(2*n_voids), width-width/(2*n_voids), n_voids)

    for cy in void_centers_y:
        r = np.sqrt((Y - cy)**2 + (Z - height/2)**2)
        mask[r < void_radius] = False

    alpha = k / (rho * c)  # thermal diffusivity
    n_steps = int(total_time / dt)
    snapshots = []

    for step in range(n_steps):
        # Finite difference Laplacian
        Tnew = T.copy()
        Tnew[1:-1,1:-1,1:-1] = T[1:-1,1:-1,1:-1] + alpha * dt * (
            (T[2:,1:-1,1:-1] - 2*T[1:-1,1:-1,1:-1] + T[:-2,1:-1,1:-1]) / dx**2 +
            (T[1:-1,2:,1:-1] - 2*T[1:-1,1:-1,1:-1] + T[1:-1,:-2,1:-1]) / dy**2 +
            (T[1:-1,1:-1,2:] - 2*T[1:-1,1:-1,1:-1] + T[1:-1,1:-1,:-2]) / dz**2
        )

        # Apply hydration heat in concrete regions
        Tnew[mask] += (heat_rate / (rho * c)) * dt

        # Boundary = outside temperature
        Tnew[0,:,:] = outside_temp
        Tnew[-1,:,:] = outside_temp
        Tnew[:,0,:] = outside_temp
        Tnew[:,-1,:] = outside_temp
        Tnew[:,:,0] = outside_temp
        Tnew[:,:,-1] = outside_temp

        T = Tnew
        time_h = step * dt / 3600
        if step % max(1, n_steps // 50) == 0:
            snapshots.append((time_h, T.copy()))

    return X, Y, Z, snapshots, mask


# -------------------------
# Streamlit App
# -------------------------

st.title("Concrete Hollow-core Slab Heat Simulation")

# Sidebar settings
st.sidebar.header("Simulation Settings")

# Geometry
length = st.sidebar.number_input("Slab length (m)", 1.0, 20.0, 4.0, help="Total length of the hollow-core slab")
width = st.sidebar.number_input("Slab width (m)", 0.5, 5.0, 1.2, help="Total width of the hollow-core slab")
height = st.sidebar.number_input("Slab height (m)", 0.1, 1.0, 0.3, help="Total height (thickness) of the hollow-core slab")
n_voids = st.sidebar.slider("Number of hollow cores", 1, 10, 5, help="How many circular voids to include across the width")

# Material properties
outside_temp = st.sidebar.number_input("Outside temperature (°C)", -20.0, 40.0, 5.0)
heat_rate = st.sidebar.number_input("Heat generation rate (W/m³)", 0.0, 200000.0, 50000.0,
                                    help="Approximate heat released by hydration per unit volume")
rho = st.sidebar.number_input("Density (kg/m³)", 1000.0, 4000.0, 2400.0)
c = st.sidebar.number_input("Specific heat capacity (J/kgK)", 500.0, 2000.0, 900.0)
k = st.sidebar.number_input("Thermal conductivity (W/mK)", 0.1, 5.0, 1.4)

# Time settings
total_time = st.sidebar.slider("Simulation time (hours)", 1, 72, 24) * 3600
dt = 60  # timestep in seconds

# Mesh resolution
nx, ny, nz = 30, 20, 10

# Run simulation
with st.spinner("Simulating heat transfer..."):
    X, Y, Z, snapshots, mask = simulate_heat(
        length, width, height, nx, ny, nz,
        outside_temp, heat_rate, total_time, dt,
        k, rho, c, n_voids
    )

# -------------------------
# Visualization
# -------------------------

st.subheader("Temperature Visualization")

time_index = st.slider("Select time index", 0, len(snapshots)-1, 0)
time_h, Tcurr = snapshots[time_index]

tmin, tmax = np.min(Tcurr), np.max(Tcurr)

# Toggle 2D/3D
view_mode = st.radio("View mode", ["3D", "2D Cross-section"])

if view_mode == "3D":
    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=Tcurr.flatten(),
        isomin=tmin, isomax=tmax,
        opacity=0.1,
        surface_count=20,
        colorscale="Jet",
        colorbar=dict(title="Temp (°C)")
    ))
    fig.update_layout(
        title=f"t = {time_h:.2f} h",
        scene_aspectmode="manual",
        scene_aspectratio=dict(x=length, y=width, z=height),
        uirevision="constant"
    )
else:
    mid = nz // 2
    fig = go.Figure(data=go.Heatmap(
        x=np.linspace(0, length, nx),
        y=np.linspace(0, width, ny),
        z=Tcurr[:,:,mid].T,
        colorscale="Jet",
        colorbar=dict(title="Temp (°C)")
    ))
    fig.update_layout(
        title=f"Cross-section at mid-height, t = {time_h:.2f} h",
        xaxis_title="Length (m)",
        yaxis_title="Width (m)"
    )

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Temperature history plot
# -------------------------
st.subheader("Temperature Evolution")

times = [t for t, _ in snapshots]
avg_temps = [np.mean(T[mask]) for _, T in snapshots]
max_temps = [np.max(T[mask]) for _, T in snapshots]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=times, y=avg_temps, mode="lines", name="Average Temp"))
fig2.add_trace(go.Scatter(x=times, y=max_temps, mode="lines", name="Max Temp"))

# Equalization estimate
tolerance = 0.5
eq_time = None
for t, avg in zip(times, avg_temps):
    if abs(avg - outside_temp) <= tolerance:
        eq_time = t
        break

if eq_time is not None:
    fig2.add_vline(x=eq_time, line=dict(color="green", dash="dash"))
    fig2.add_trace(go.Scatter(
        x=[eq_time], y=[outside_temp],
        mode="markers+text",
        marker=dict(color="green", size=10, symbol="x"),
        text=["Equalized"],
        textposition="top right",
        name="Equalization Point"
    ))

fig2.update_layout(
    xaxis_title="Time (hours)",
    yaxis_title="Temperature (°C)",
    title="Concrete Temperature vs Time",
    height=400,
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("Equalization Estimate")
if eq_time is not None:
    st.success(f"The slab reaches equilibrium with outside air (≈{outside_temp}°C) at about **{eq_time:.1f} hours**.")
else:
    st.info("The slab has not equalized with outside air within the simulated time.")
