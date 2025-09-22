# app_fenics_slab.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from fenics import *

st.set_page_config(layout="centered")
st.title("1D Heat Conduction in Concrete Slab (FEniCS FEM)")

# ------------------------
# Sidebar inputs
# ------------------------
bed_temp = st.sidebar.slider("Heated bed temperature (°C)", 20, 80, 60)
ambient_temp = st.sidebar.slider("Ambient (air) temperature (°C)", -5, 40, 24)
sim_hours = st.sidebar.slider("Simulation time (hours)", 1, 48, 12)
dt_hours = st.sidebar.slider("Time step (hours)", 0.1, 2.0, 0.5)

rho = 2300.0    # kg/m3
cp = 900.0      # J/kgK
k = 1.4         # W/mK
alpha = k / (rho * cp)
h = 10.0        # W/m2K

L = 0.30        # m thickness
Nx = 60

# ------------------------
# Mesh and function space
# ------------------------
mesh = IntervalMesh(Nx, 0.0, L)
V = FunctionSpace(mesh, "CG", 1)

# Trial/test functions
u = TrialFunction(V)
v = TestFunction(V)

# Initial condition
u_n = interpolate(Constant(ambient_temp), V)

# Time stepping
dt = dt_hours * 3600.0
t_end = sim_hours * 3600.0
n_steps = int(t_end // dt)

# Bottom BC: Dirichlet (fixed bed temperature)
bc_bottom = DirichletBC(V, Constant(bed_temp), "near(x[0], 0.0)")

# Weak form (implicit Euler)
u_mid = u
F = (u - u_n) / dt * v * dx + alpha * dot(grad(u_mid), grad(v)) * dx

# Robin (convection) at top surface
ds_top = Measure("ds", domain=mesh, subdomain_data=None)
F += (h / (rho*cp)) * (u_mid - ambient_temp) * v * ds

a, Lf = lhs(F), rhs(F)

# ------------------------
# Time loop
# ------------------------
times = []
mid_temps = []
u_sol = Function(V)

for n in range(n_steps):
    solve(a == Lf, u_sol, bc_bottom)
    u_n.assign(u_sol)

    t_h = (n+1)*dt/3600.0
    times.append(t_h)
    mid_temps.append(u_sol(L/2))

# ------------------------
# Plot mid-slab vs time
# ------------------------
st.subheader("Mid-slab temperature vs time")
fig, ax = plt.subplots()
ax.plot(times, mid_temps, label="Mid-slab (FEniCS)")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Temperature (°C)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ------------------------
# Final profile
# ------------------------
st.subheader("Final depth profile")
z_vals = np.linspace(0, L, 50)
temps = [u_sol(z) for z in z_vals]
fig2, ax2 = plt.subplots()
ax2.plot(np.array(temps), z_vals*1000, label=f"After {sim_hours} h")
ax2.set_xlabel("Temperature (°C)")
ax2.set_ylabel("Depth (mm) from bed")
ax2.invert_yaxis()
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)
