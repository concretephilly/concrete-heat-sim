# slab_heat_skfem.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# scikit-fem imports
from skfem import MeshLine, ElementLine, Basis, FacetBasis
from skfem import bilinear_form, linear_form, asm, condense

st.set_page_config(layout="centered")
st.title("1D Heat Conduction — scikit-fem (CEM III)")

# ------------------------
# Material & geometry (exposed as before)
# ------------------------
st.sidebar.header("Simulation inputs")

bed_temp = st.sidebar.slider("Heated bed temperature (°C)", 20, 80, 60)
ambient_temp = st.sidebar.slider("Ambient / top temperature (°C)", -5, 40, 24)
sim_hours = st.sidebar.slider("Simulation time (hours)", 1, 100, 24)

# Expose material properties as requested previously
st.sidebar.header("Material properties")
rho = st.sidebar.number_input("Density ρ (kg/m³)", 1000.0, 4000.0, 2300.0, step=50.0,
                              help="Concrete density (CEM III ≈ 2300 kg/m³)")
cp = st.sidebar.number_input("Specific heat c (J/kgK)", 500.0, 2000.0, 900.0, step=10.0,
                             help="Concrete specific heat (CEM III ≈ 900 J/kgK)")
k = st.sidebar.number_input("Thermal conductivity k (W/mK)", 0.5, 3.0, 1.4, step=0.1,
                            help="Thermal conductivity (CEM III ≈ 1.4 W/mK)")
h = st.sidebar.number_input("Convective h (W/m²K)", 1.0, 50.0, 10.0, step=1.0,
                            help="Top surface convection coefficient (natural ≈ 10 W/m²K)")

alpha = k / (rho * cp)

# Keep spacing and numerical settings similar to previous app
L = 0.30               # slab thickness (m)
Nx = 61                # number of nodes for 1D mesh (keeps dz ~ previous)
z = np.linspace(0.0, L, Nx)
dz = z[1] - z[0]

st.sidebar.write(f"dz = {dz*1000:.1f} mm")
# For FEM implicit we choose a dt similar to previous stability-based dt but allow user to scale
dt_auto = (dz ** 2) / (2.0 * alpha)
dt = 0.9 * dt_auto
st.sidebar.write(f"dt (auto) = {dt:.3f} s (used internally)")

# ------------------------
# Build 1D mesh & function spaces (scikit-fem)
# ------------------------
mesh = MeshLine(np.linspace(0.0, L, Nx))
element = ElementLine()    # linear Lagrange 1D
basis = Basis(mesh, element)

# facet basis for boundary integrals (1D endpoints)
facet_basis = FacetBasis(mesh, element)

# ------------------------
# Weak forms
# ------------------------
@bilinear_form
def mass(u, v, w):
    # u, v are basis values; w provides geometry
    return rho * cp * u * v

@bilinear_form
def stiffness(u, v, w):
    # w.grad(u) returns gradient arrays; for 1D this works fine
    gu = w.grad(u)
    gv = w.grad(v)
    return k * (gu * gv)

@bilinear_form
def robin_b(u, v, w):
    # boundary bilinear form (for facet assembly), actual multiplication by h done at assembly
    return u * v

@linear_form
def robin_rhs(v, w):
    # boundary linear form for ambient contribution (will multiply by h*ambient)
    return v

# assemble global mass and stiffness matrices
M = asm(mass, basis)
K = asm(stiffness, basis)

# Robin (convection) assembled on facets
K_robin = h * asm(robin_b, facet_basis)   # multiplies u*v on boundary facets by h
f_robin_const = h * ambient_temp * asm(robin_rhs, facet_basis)  # right-hand side boundary contribution

# ------------------------
# Boundary (Dirichlet) DOFs
# ------------------------
# We want bottom (x=0) to be Dirichlet fixed at bed_temp
# Find the dof nearest x=0
coords = mesh.p[0]  # mesh points 1D array
dof_bottom = np.argmin(np.abs(coords - 0.0))
D = np.array([dof_bottom], dtype=int)

# ------------------------
# Time stepping setup
# ------------------------
max_time_s = sim_hours * 3600.0
n_steps = int(np.ceil(max_time_s / dt))

# Precompute left-hand matrix (A = M/dt + K + K_robin) but A depends on dt only not T
A = (M / dt) + K + K_robin

# We'll condense Dirichlet DOFs during solves
# initial temperature vector (ambient)
from skfem.helpers import solve_direct
Tn = np.full(basis.N, ambient_temp, dtype=float)  # initial nodal values

# For storing time-history (mid-slab)
times_hist = []
mid_history = []

# Precompute index of mid node (closest to L/2)
mid_idx = np.argmin(np.abs(coords - (L/2.0)))

# Condense A once (remove Dirichlet rows/cols). But RHS changes every time; use condense(A,b,D)
# Use condense from skfem: condense(A, b, D=D, x=None) returns Acond, bcond, I, J, and basis map if needed
# We'll condense each step to keep it simple and robust.
from skfem import condense

# ------------------------
# Time loop (backward Euler)
# ------------------------
with st.spinner("Running scikit-fem solver..."):
    for n in range(n_steps):
        # RHS: M/dt * Tn + f_robin_const
        rhs = (M @ (Tn / dt)) + f_robin_const

        # Apply Dirichlet by condensing the system
        Acond, rhscond, D_map = condense(A, rhs, D=D)

        # Solve condensed system (direct solver)
        from scipy.sparse.linalg import spsolve
        T_reduced = spsolve(Acond.tocsr(), rhscond)

        # Reconstruct full solution (put Dirichlet values back)
        from skfem import make_balanced
        T_full = make_balanced(T_reduced, D, bed_temp, basis.N)

        # Update for next step
        Tn = T_full.copy()

        # record time & mid-node temp
        if True:
            t_h = (n+1) * dt / 3600.0
            times_hist.append(t_h)
            mid_history.append(Tn[mid_idx])

# Final assembled nodal array
T_final = Tn.copy()

# ------------------------
# Analytic-like equilibrium (same as earlier)
# ------------------------
T_top_eq = (k * bed_temp + h * L * ambient_temp) / (k + h * L)
T_eq_profile = bed_temp + (T_top_eq - bed_temp) * (z / L)
T_eq_mid = T_eq_profile[Nx // 2]

# ------------------------
# Post processing metrics (same logic)
# ------------------------
initial_mid = ambient_temp
equil_mid = T_eq_mid
delta_equil = equil_mid - initial_mid
if abs(delta_equil) < 1e-9:
    frac = np.zeros_like(mid_history)
else:
    frac = (np.array(mid_history) - initial_mid) / delta_equil

idx90 = np.where(frac >= 0.9)[0]
time_to_90 = (times_hist[idx90[0]] if idx90.size > 0 else None)

idx63 = np.argmin(np.abs(frac - 0.63)) if delta_equil != 0 else None
time63 = (times_hist[idx63] if idx63 is not None else None)

L_half = L / 2.0
t_diff_s = L_half**2 / alpha
t_diff_h = t_diff_s / 3600.0

# ------------------------
# Plots (keeps same look/feel)
# ------------------------
st.subheader("Depth profile (final) and equilibrium")
fig1, ax1 = plt.subplots()
ax1.plot(z*1000.0, T_final, label=f"Final after {sim_hours} h")
ax1.plot(z*1000.0, T_eq_profile, 'k--', label="Analytic equilibrium")
ax1.set_xlabel("Depth from bed (mm)")
ax1.set_ylabel("Temperature (°C)")
ax1.grid(alpha=0.25)
ax1.legend()
st.pyplot(fig1)

st.subheader("Mid-slab temperature vs time")
fig2, ax2 = plt.subplots()
ax2.plot(times_hist, mid_history, label="Mid-slab (scikit-fem)")
ax2.axhline(equil_mid, color='k', linestyle='--', label="Equilibrium mid-slab")
if time63 is not None:
    ax2.axvline(time63, color='orange', linestyle=':', label=f"~63% @ {time63:.2f} h")
if time_to_90 is not None:
    ax2.axvline(time_to_90, color='green', linestyle='--', label=f"90% @ {time_to_90:.2f} h")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Temperature (°C)")
ax2.grid(alpha=0.25)
ax2.legend()
st.pyplot(fig2)

# ------------------------
# Numeric outputs & checks
# ------------------------
st.markdown("### Numeric diagnostics")
st.write(f"- dz = {dz:.4f} m, dt = {dt:.4f} s (implicit), alpha = {alpha:.2e} m²/s")
st.write(f"- Analytic equilibrium mid-slab temp = {equil_mid:.3f} °C")
st.write(f"- Numerical final mid-slab temp = {T_final[Nx//2]:.3f} °C")

if time_to_90 is not None:
    st.success(f"Time to reach 90% of equilibrium (mid-slab) ≈ {time_to_90:.2f} hours")
else:
    st.info("Mid-slab did not reach 90% of equilibrium within the simulated time.")

if time63 is not None:
    st.write(f"Approx 63% time (numerical) ≈ {time63:.2f} hours")

st.write(f"Diffusion timescale estimate (L/2)^2/alpha ≈ {t_diff_h:.2f} hours")

st.caption("This run used scikit-fem (FEM) backward-Euler time stepping. Compare with your original FD results to validate.")
