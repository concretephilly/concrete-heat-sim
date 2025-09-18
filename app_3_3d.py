# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Hollow-Core Slab Heat (CEM III defaults)")

st.title("Hollow-Core Slab — 2D Cross-section Heat Simulation (CEM III defaults)")

# -------------------------
# Defaults (CEM III-like)
# -------------------------
DEFAULT_DENSITY = 2300.0
DEFAULT_CP = 900.0
DEFAULT_K = 2.5
DEFAULT_CEMENT = 350.0
DEFAULT_QTOT = 250e3
DEFAULT_K_RATE_H = 0.05
DEFAULT_OUTSIDE_TEMP = 20.0
DEFAULT_H = 10.0

# -------------------------
# Sidebar: user inputs
# -------------------------
st.sidebar.header("Geometry")
length = st.sidebar.number_input("Length (m)", 1.0, 20.0, 4.0, 0.1)
width = st.sidebar.number_input("Width (m)", 0.5, 5.0, 1.2, 0.05)
height = st.sidebar.number_input("Height (m)", 0.08, 0.5, 0.20, 0.01)

st.sidebar.header("Hollow-c cores")
n_voids = st.sidebar.number_input("Number of circular hollow cores", 1, 8, 5, 1)
void_radius = st.sidebar.number_input("Core radius (m)", 0.02, 0.25, 0.06, 0.005)

st.sidebar.header("Material (CEM III defaults)")
density = st.sidebar.number_input("Concrete density ρ (kg/m³)", 1800.0, 2800.0, DEFAULT_DENSITY, 10.0)
cp = st.sidebar.number_input("Concrete specific heat c_p (J/kg·K)", 600.0, 1200.0, DEFAULT_CP, 10.0)
k_conc = st.sidebar.number_input("Concrete thermal conductivity k (W/m·K)", 0.5, 4.0, DEFAULT_K, 0.1)

st.sidebar.header("Hydration")
cement_content = st.sidebar.number_input("Cement content (kg cement / m³ concrete)", 150.0, 600.0, DEFAULT_CEMENT, 10.0)
Q_total = st.sidebar.number_input("Total heat Q_total (J/kg cement)", 50e3, 600e3, DEFAULT_QTOT, 1000.0)
k_rate_h = st.sidebar.number_input("Reaction rate k (1/hour)", 0.001, 1.0, DEFAULT_K_RATE_H, 0.001)

st.sidebar.header("Environment & Simulation")
outside_temp = st.sidebar.number_input("Outside air temperature (°C)", -20.0, 40.0, DEFAULT_OUTSIDE_TEMP, 0.5)
h_conv = st.sidebar.number_input("Convection h (W/m²·K)", 0.5, 100.0, DEFAULT_H, 0.1)
sim_hours = st.sidebar.number_input("Simulation duration (hours)", 1, 200, 72, 1)
dt_seconds = st.sidebar.number_input("Timestep (s)", 60.0, 1800.0, 300.0, 60.0)

st.sidebar.header("Resolution / performance")
res_preset = st.sidebar.selectbox("Simulation resolution", ["Low", "Medium", "High"], index=0)

if res_preset == "Low":
    nx, ny, nz = 60, 30, 10
elif res_preset == "Medium":
    nx, ny, nz = 120, 60, 20
else:
    nx, ny, nz = 180, 90, 28

st.sidebar.markdown("⚡ **Tip:** Lower resolution & larger timestep = much faster.")

# -------------------------
# Run button
# -------------------------
if st.button("Run Simulation"):
    st.info("⏳ Running simulation... please wait. This may take some seconds depending on settings.")

    # Mesh spacing
    dx = length / nx
    dy = width / ny

    # Time steps
    n_steps = int(sim_hours * 3600 / dt_seconds)

    # Initialize temperature field
    T = np.full((nx, ny), outside_temp, dtype=np.float64)

    # Hydration progress
    alpha = np.zeros((nx, ny), dtype=np.float64)

    # Void mask (True where air voids exist)
    xv = np.linspace(0, length, nx)
    yv = np.linspace(0, width, ny)
    X, Y = np.meshgrid(xv, yv, indexing="ij")
    void_mask = np.zeros_like(T, dtype=bool)

    void_spacing = width / (n_voids + 1)
    for i in range(n_voids):
        cy = (i + 1) * void_spacing
        void_mask |= ( (Y - cy) ** 2 + (X - length/2) ** 2 ) < void_radius ** 2

    # Progress bar
    progress = st.progress(0, text="Starting simulation...")

    # Store average temperature history
    avg_temps = []

    # Time loop
    for step in range(n_steps):
        # Hydration (Arrhenius-type model)
        d_alpha = k_rate_h * (1 - alpha) * (dt_seconds / 3600.0)
        alpha += d_alpha
        alpha = np.clip(alpha, 0.0, 1.0)

        q_hyd = cement_content * Q_total * d_alpha / (density * cp)

        # Diffusion (finite differences, simplified 2D)
        T_pad = np.pad(T, 1, mode="edge")
        lap = (
            T_pad[2:,1:-1] + T_pad[:-2,1:-1] +
            T_pad[1:-1,2:] + T_pad[1:-1,:-2] -
            4 * T_pad[1:-1,1:-1]
        ) / (dx * dy)

        dT_diff = k_conc / (density * cp) * lap * dt_seconds

        # Convection on surfaces (approximate)
        surface_loss = np.zeros_like(T)
        surface_loss[0,:] += h_conv * (outside_temp - T[0,:]) * dt_seconds / (density * cp * dx)
        surface_loss[-1,:] += h_conv * (outside_temp - T[-1,:]) * dt_seconds / (density * cp * dx)
        surface_loss[:,0] += h_conv * (outside_temp - T[:,0]) * dt_seconds / (density * cp * dy)
        surface_loss[:,-1] += h_conv * (outside_temp - T[:,-1]) * dt_seconds / (density * cp * dy)

        # Update T
        T += dT_diff + q_hyd + surface_loss

        # Void cells stay at outside air temperature
        T[void_mask] = outside_temp

        # Save average temp
        avg_temps.append(np.mean(T[~void_mask]))

        # Update progress bar
        if step % max(1, n_steps // 100) == 0:
            pct = int((step / n_steps) * 100)
            progress.progress(pct, text=f"Running simulation... {pct}%")

    progress.progress(100, text="Simulation complete ✅")

    # -------------------------
    # Plot cross-section
    # -------------------------
    fig = go.Figure(data=go.Heatmap(
        z=T.T,
        x=xv,
        y=yv,
        colorscale="Jet",
        colorbar=dict(title="°C")
    ))
    fig.update_layout(
        title="Final temperature distribution (cross-section)",
        xaxis_title="Length (m)",
        yaxis_title="Width (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Plot avg temp vs time
    # -------------------------
    times = np.arange(n_steps) * dt_seconds / 3600
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=avg_temps, mode="lines", name="Avg Temp"))
    fig2.update_layout(
        title="Average temperature over time",
        xaxis_title="Time (hours)",
        yaxis_title="Temperature (°C)"
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.warning("Click **Run Simulation** to start. Adjust parameters in the sidebar first.")
