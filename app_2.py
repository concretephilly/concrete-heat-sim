import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Concrete Floor Panel Heat Simulation")

# --- User Inputs ---
st.sidebar.header("Simulation Parameters")
nx = st.sidebar.slider("Grid size (x)", 10, 100, 40)
ny = st.sidebar.slider("Grid size (y)", 10, 100, 40)
length = st.sidebar.number_input("Panel length (m)", 0.5, 10.0, 2.0)
width = st.sidebar.number_input("Panel width (m)", 0.5, 10.0, 2.0)

initial_temp = st.sidebar.number_input("Initial concrete temp (°C)", 0.0, 50.0, 20.0)
outside_temp = st.sidebar.number_input("Outside temp (°C)", -20.0, 50.0, 5.0)

rho = st.sidebar.number_input("Density (kg/m³)", 1500.0, 3000.0, 2400.0)
cp = st.sidebar.number_input("Heat capacity (J/kgK)", 500.0, 2000.0, 900.0)
k = st.sidebar.number_input("Thermal conductivity (W/mK)", 0.1, 10.0, 1.7)

total_time = st.sidebar.number_input("Total time (hours)", 0.1, 48.0, 6.0)
dt = st.sidebar.number_input("Time step (s)", 1.0, 600.0, 30.0)

# --- Setup grid ---
dx = length / nx
dy = width / ny
alpha = k / (rho * cp)  # thermal diffusivity

nt = int(total_time * 3600 / dt)
T = np.ones((ny, nx)) * initial_temp

# --- Simulation ---
results = []
for n in range(nt):
    Tn = T.copy()
    # interior update (finite difference, explicit scheme)
    T[1:-1,1:-1] = (Tn[1:-1,1:-1] +
        alpha*dt*(
            (Tn[1:-1,2:] - 2*Tn[1:-1,1:-1] + Tn[1:-1,:-2]) / dx**2 +
            (Tn[2:,1:-1] - 2*Tn[1:-1,1:-1] + Tn[:-2,1:-1]) / dy**2
        )
    )
    # boundary condition = fixed outside temperature
    T[0,:] = outside_temp
    T[-1,:] = outside_temp
    T[:,0] = outside_temp
    T[:,-1] = outside_temp

    if n % 5 == 0:  # store every 5th step
        results.append((n*dt/3600, T.copy()))

# --- Playback controls ---
st.sidebar.header("Playback")
frame = st.sidebar.slider("Time step", 0, len(results)-1, 0)
time_h, temp = results[frame]

# --- Plot selected frame ---
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(temp, cmap="inferno", origin="lower",
               extent=[0,length,0,width])
ax.set_title(f"t = {time_h:.2f} h")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Temp (°C)")

st.pyplot(fig)

st.markdown(
    "▶️ Use the slider in the sidebar to **play through time**. "
    "This shows how heat diffuses through the slab."
)
