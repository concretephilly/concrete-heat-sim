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
    # ---- import solver code from previous version ----
    # (To keep answer shorter, reuse the full solver I gave in last message,
    # but keep nx,ny,nz and dt_seconds from here.)
    # Just paste that solver section here, unchanged, starting with
    # "dx = length / nx" and ending at "st.download_button..."
    st.info("⏳ Running simulation... please wait. This may take some seconds depending on settings.")

    # (--- solver code block from last message goes here ---)

else:
    st.warning("Click **Run Simulation** to start. Adjust parameters in the sidebar first.")
