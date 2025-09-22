import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title("Basic 1D Heat Conduction in Concrete Slab")

# Slab properties (300 mm thick, concrete)
thickness = 0.3    # slab thickness (m)
rho = 2400         # density (kg/m^3)
cp = 880           # specific heat (J/kg*K)
k = 1.7            # thermal conductivity (W/m*K)
alpha = k/(rho*cp) # thermal diffusivity

# Discretization parameters
Nx = 61                 # number of grid points in thickness
dz = thickness/(Nx-1)   # spatial step (m)
dt = 5.0                # time step (s), satisfy dt <= dz^2/(2*alpha)
t_final = 7200.0        # simulate 2 hours (7200 s)

# User inputs
bed_temp = st.slider("Heated Bed Temperature (°C)", 20, 80, 50)
T_init = 20.0           # initial slab temp (°C)
T_top = 20.0            # fixed top temp (°C, ambient)

# Initialize temperature array
T = np.full(Nx, T_init)

# Time-stepping loop (explicit FTCS)
for n in range(int(t_final/dt)):
    Tn = T.copy()
    Tn[0] = bed_temp       # enforce bottom BC
    Tn[-1] = T_top         # enforce top BC
    # update interior nodes
    for i in range(1, Nx-1):
        T[i] = Tn[i] + alpha*dt/dz**2 * (Tn[i+1] - 2*Tn[i] + Tn[i-1])
    T[0] = bed_temp         # re-apply BCs after update
    T[-1] = T_top

# Plot temperature distribution vs depth
z = np.linspace(0, thickness, Nx)  # depth axis (0 at bed)
fig, ax = plt.subplots()
ax.plot(z, np.ones_like(z)*T_init, 'k--', label=f"Initial ({T_init}°C)")
ax.plot(z, T, 'r-', label=f"After 2h, bed={bed_temp}°C")
ax.set_xlabel("Depth (m) from heated bed")
ax.set_ylabel("Temperature (°C)")
ax.set_ylim(T_init, bed_temp+5)
ax.legend()
st.pyplot(fig)
