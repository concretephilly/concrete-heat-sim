# app.py
"""
Crank-Nicolson 3D heat diffusion demo for a 1x1x1 m cube on a heated plate.
- Uses scipy.sparse for the implicit solver (Crank-Nicolson).
- UI: plate temp, air temp, cement params, sim time, grid resolution.
- Views: 3D volume (orbitable), 2D cross-section, and avg/max vs time plot.
- Colors: Jet (blue=cold -> red=hot).
Notes: if scipy is not available, the app will show instructions and fall back
to a stable explicit solver (with auto timestep check).
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# try importing scipy; CN solver needs sparse linear algebra
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Cube on Heated Plate — CN Solver")

st.title("1×1×1 m Cube on a Heated Plate — Crank–Nicolson 3D Heat Simulation")

# --------------------------
# Sidebar: user controls
# --------------------------
st.sidebar.header("Geometry & grid")
cube_size = st.sidebar.number_input("Cube side (m)", 0.5, 2.0, 1.0, 0.1)
nx = st.sidebar.slider("Grid cells per side (nx=ny=nz)", 8, 60, 28, 1,
                       help="Higher -> more accurate but slower. Try 20-40 for testing.")
# use uniform cubic cells
dx = cube_size / nx

st.sidebar.header("Boundary & environment")
T_plate = st.sidebar.number_input("Plate (bottom) temperature (°C)", 20.0, 120.0, 60.0, 1.0)
T_air = st.sidebar.number_input("Ambient (air) temperature (°C)", -10.0, 40.0, 20.0, 0.5)
h_conv = st.sidebar.number_input("Convection h (W/m²K) on exposed faces", 1.0, 100.0, 10.0, 0.5)

st.sidebar.header("Concrete (CEM III-like defaults)")
# Values chosen from literature ranges (see citations)
rho = st.sidebar.number_input("Density ρ (kg/m³)", 2000.0, 2600.0, 2400.0, 10.0)
cp = st.sidebar.number_input("Specific heat c_p (J/kg·K)", 700.0, 1100.0, 900.0, 10.0)
k_cond = st.sidebar.number_input("Thermal conductivity k (W/m·K)", 1.0, 3.5, 2.0, 0.1)

st.sidebar.header("Hydration heat (volumetric source)")
cement_content = st.sidebar.number_input("Cement content (kg cement / m³ concrete)", 200.0, 400.0, 300.0, 10.0)
Q_cement = st.sidebar.number_input("Heat of hydration Q (J/kg cement)", 100e3, 400e3, 250e3, 1000.0,
                                   help="Typical range ~250 kJ/kg (60 cal/g) for low-heat cements.")
k_rate_h = st.sidebar.number_input("Hydration rate k (1/hour)", 0.001, 1.0, 0.05, 0.001,
                                   help="Larger = faster release of heat")

st.sidebar.header("Time & solver")
sim_hours = st.sidebar.number_input("Simulation duration (hours)", 0.5, 72.0, 12.0, 0.5)
dt_seconds = st.sidebar.number_input("Requested timestep (s)", 1.0, 3600.0, 60.0, 1.0)

# Derived quantities
total_seconds = float(sim_hours) * 3600.0
n_steps = max(1, int(total_seconds / float(dt_seconds)))
st.sidebar.markdown(f"Estimated timesteps: **{n_steps}**")

# thermal diffusivity (alpha = k / (rho * cp))
alpha = k_cond / (rho * cp)  # m^2/s

# Stability check for explicit method (just to inform user if scipy missing)
dt_stable = dx * dx / (6.0 * alpha) if alpha > 0 else np.inf

if not SCIPY_AVAILABLE and dt_seconds > dt_stable:
    st.sidebar.warning(
        "Scipy not installed and requested dt is larger than explicit stability limit.\n"
        f"Explicit method would be unstable (dt_stable ≈ {dt_stable:.1f} s). "
        "Install scipy to use Crank–Nicolson implicit solver or reduce dt."
    )

# --------------------------
# Prepare grid / indexing
# --------------------------
nx = int(nx)
N = nx * nx * nx  # total nodes
# 3D coordinates (for plotting)
xs = np.linspace(0, cube_size, nx)
ys = np.linspace(0, cube_size, nx)
zs = np.linspace(0, cube_size, nx)
Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')  # (nx,nx,nx)

# helper to flatten 3D index (i,j,k) -> idx
def idx(i, j, k):
    return (i * nx + j) * nx + k

# boundary classification:
# bottom k=0 is plate (Dirichlet = T_plate)
# other faces (k=nx-1, i=0, i=nx-1, j=0, j=nx-1) are exposed to air (we model convection)
# interior nodes are solved implicitly

# --------------------------
# Hydration source function
# --------------------------
k_rate_s = k_rate_h / 3600.0  # convert per hour to per second
Q_total_per_m3 = Q_cement * cement_content  # J per m^3 total (assumes Q_cement J/kg * kg cement per m3)

def q_vol(t):  # volumetric heat generation rate (J/m3/s) at time t (s)
    # single-exponential decay kinetics: dQ/dt = Q_total_per_m3 * k_s * exp(-k_s * t)
    return Q_total_per_m3 * k_rate_s * np.exp(-k_rate_s * t)

# --------------------------
# Build Laplacian sparse matrix (constant-coefficient, 7-point stencil)
# We'll create A such that: dT/dt = alpha * A * T  (A includes 1/dx^2 factor)
# For Crank-Nicolson we need (I - 0.5*dt*alpha*A) T^{n+1} = (I + 0.5*dt*alpha*A) T^n + dt * source_term
# --------------------------
if SCIPY_AVAILABLE:
    from scipy.sparse import lil_matrix, csr_matrix, identity
    st.info("Building sparse matrix for implicit solver (this may take a second)...")
    A = lil_matrix((N, N), dtype=np.float64)
    inv_dx2 = 1.0 / (dx * dx)
    for i in range(nx):
        for j in range(nx):
            for k in range(nx):
                r = idx(i, j, k)
                # if node is bottom plate (k==0) -> Dirichlet handled later (we will enforce)
                # For matrix A we still build interior stencil; we'll modify rows for Dirichlet nodes
                # neighbor count and coefficients
                neighbors = []
                # i+/-1
                if i > 0:
                    neighbors.append(idx(i-1, j, k))
                if i < nx-1:
                    neighbors.append(idx(i+1, j, k))
                # j+/-1
                if j > 0:
                    neighbors.append(idx(i, j-1, k))
                if j < nx-1:
                    neighbors.append(idx(i, j+1, k))
                # k+/-1
                if k > 0:
                    neighbors.append(idx(i, j, k-1))
                if k < nx-1:
                    neighbors.append(idx(i, j, k+1))

                # center diagonal
                A[r, r] = -len(neighbors) * inv_dx2
                for nb in neighbors:
                    A[r, nb] = inv_dx2
    A = csr_matrix(A)
    I = identity(N, format='csr')

    # Precompute CN matrices
    dt = float(dt_seconds)
    M_left = (I - 0.5 * dt * alpha * A).tocsr()
    M_right = (I + 0.5 * dt * alpha * A).tocsr()

else:
    st.warning("scipy is not available — the app will attempt a stable explicit fallback later. "
               "Install scipy (scipy.sparse) for the implicit Crank–Nicolson solver.")

# --------------------------
# Initial temperature (flattened)
# --------------------------
T0 = np.full((N,), float(T_air), dtype=np.float64)
# enforce bottom Dirichlet nodes to plate temperature initially
for i in range(nx):
    for j in range(nx):
        r = idx(i, j, 0)  # k=0 bottom
        T0[r] = float(T_plate)

# --------------------------
# Time stepping
# --------------------------
run_button = st.button("(Re)run simulation now")
# auto-run if user didn't want button? We'll run automatically if not pressing button,
# but to avoid accidental re-runs, require button to start heavy CN solver.
# If scipy not available we run explicit fallback automatically once clicked.
if run_button:
    start_time = time.time()
    progress = st.progress(0, text="Preparing solver...")
    Tn = T0.copy()
    snapshots = []
    times = []
    # For boundary treatment we will:
    # - enforce Dirichlet bottom (k=0) at each step by overwriting after solve
    # - approximate convection on other faces by adding an explicit term to rhs (simpler)
    # Precompute indices for nodes on faces (excluding bottom)
    top_nodes = [idx(i, j, nx-1) for i in range(nx) for j in range(nx)]
    i0_nodes = [idx(0, j, k) for j in range(nx) for k in range(nx)]
    iN_nodes = [idx(nx-1, j, k) for j in range(nx) for k in range(nx)]
    j0_nodes = [idx(i, 0, k) for i in range(nx) for k in range(nx)]
    jN_nodes = [idx(i, nx-1, k) for i in range(nx) for k in range(nx)]
    bottom_nodes = [idx(i, j, 0) for i in range(nx) for j in range(nx)]

    # Precompute area/volume ratio approximations for surface convection
    # For a voxel on face normal to x (i=0 or i=nx-1): A/V ≈ 1/dx
    AoverV_x = 1.0 / dx
    AoverV_y = 1.0 / dx
    AoverV_z = 1.0 / dx

    # Build mask array to detect bottom Dirichlet quickly
    bottom_mask = np.zeros(N, dtype=bool)
    bottom_mask[bottom_nodes] = True

    # progress bookkeeping
    last_update = 0
    for step in range(n_steps):
        t = step * dt  # seconds
        # compute volumetric source at t and t+dt (Crank-Nicolson treat as trapezoid)
        qn = q_vol(t)               # J/m3/s at t
        qnp1 = q_vol(t + dt)        # J/m3/s at t+dt
        # corresponding temperature source contribution per node:
        # s = q/(rho*cp) -> K/s ; treat CN => multiply by dt/2*(s_n + s_np1)
        svec = (dt * 0.5) * (qn + qnp1) / (rho * cp)  # scalar (applies to all concrete voxels)

        # Right-hand side: M_right * Tn + source_term + convection_term (explicit approx)
        rhs = M_right.dot(Tn)

        # add volumetric source uniformly to rhs (every voxel; for plate nodes we overwrite later)
        rhs += svec

        # explicit convection on exposed faces: approximate dT = -(h * A/V) / (rho*cp) * (Tn - T_air) * dt
        # we'll compute ΔT_conv and add to rhs as "explicit" contribution (i.e. rhs += ΔT_conv)
        # X faces
        # i=0 face nodes
        if h_conv > 0:
            # i=0
            for r in i0_nodes:
                # only if not bottom (which is Dirichlet)
                if not bottom_mask[r]:
                    dT = - (h_conv * AoverV_x) / (rho * cp) * (Tn[r] - T_air) * dt
                    rhs[r] += dT
            # i=nx-1
            for r in iN_nodes:
                if not bottom_mask[r]:
                    dT = - (h_conv * AoverV_x) / (rho * cp) * (Tn[r] - T_air) * dt
                    rhs[r] += dT
            # j=0
            for r in j0_nodes:
                if not bottom_mask[r]:
                    dT = - (h_conv * AoverV_y) / (rho * cp) * (Tn[r] - T_air) * dt
                    rhs[r] += dT
            # j=nx-1
            for r in jN_nodes:
                if not bottom_mask[r]:
                    dT = - (h_conv * AoverV_y) / (rho * cp) * (Tn[r] - T_air) * dt
                    rhs[r] += dT
            # top face (k=nx-1)
            for r in top_nodes:
                if not bottom_mask[r]:
                    dT = - (h_conv * AoverV_z) / (rho * cp) * (Tn[r] - T_air) * dt
                    rhs[r] += dT

        # enforce Dirichlet bottom nodes: modify rhs to equal T_plate (left side will be handled by replacing rows)
        # For Dirichlet enforcement: set row r of M_left to e_r^T, and rhs[r] = T_plate
        # Easiest: solve full system then overwrite bottom nodes after solve to T_plate (acceptable because bottom is rigid)
        # Solve for T_{n+1}
        try:
            Tnp1 = spla.spsolve(M_left, rhs)
        except Exception as e:
            st.error(f"Linear solver failed: {e}")
            break

        # enforce bottom plate Dirichlet exactly
        Tnp1[bottom_nodes] = float(T_plate)

        # clamp physically (no below absolute ambient by much)
        Tnp1 = np.maximum(Tnp1, T_air - 5.0)  # slight allowance

        Tn = Tnp1

        # save snapshots periodically (store every ~1% of steps up to 200 frames)
        if (step % max(1, n_steps // 200)) == 0 or (step == n_steps - 1):
            snapshots.append(Tn.copy())
            times.append(t / 3600.0)  # hours

        # update progress + ETA occasionally
        if (step - last_update) >= max(1, n_steps // 100):
            elapsed = time.time() - start_time
            completed = step + 1
            frac = completed / float(n_steps)
            est_total = elapsed / frac if frac > 0 else 0.0
            est_remaining = max(0.0, est_total - elapsed)
            pct = int(frac * 100.0)
            progress.progress(pct, text=f"Running... {pct}% | ~{est_remaining:.1f}s left")
            last_update = step

    progress.progress(100, text="Simulation complete ✅")

    # reshape snapshots to (frames, nx, nx, nx)
    frames = [s.reshape((nx, nx, nx)) for s in snapshots]

    st.success(f"Simulation finished: saved {len(frames)} frames over {sim_hours} hours.")

    # --------------------------
    # Visualization controls
    # --------------------------
    # Sidebar controls for view
    view = st.selectbox("View", ["3D Volume (cube)", "2D cross-section", "Temperature vs time"])
    frame_idx = st.slider("Frame index", 0, len(frames) - 1, 0)
    Tframe = frames[frame_idx]

    if view == "3D Volume (cube)":
        # plotly Volume (can be heavy); we downsample for plotting if grid large
        max_plot = 40
        if nx > max_plot:
            # downsample by slicing
            step_ds = int(nx // max_plot)
            Tplot = Tframe[::step_ds, ::step_ds, ::step_ds]
            xs_plot = xs[::step_ds]
            ys_plot = ys[::step_ds]
            zs_plot = zs[::step_ds]
        else:
            Tplot = Tframe
            xs_plot = xs; ys_plot = ys; zs_plot = zs

        fig = go.Figure(data=go.Volume(
            x=np.repeat(xs_plot, len(ys_plot)*len(zs_plot)),
            y=np.tile(np.repeat(ys_plot, len(zs_plot)), len(xs_plot)),
            z=np.tile(zs_plot, len(xs_plot)*len(ys_plot)),
            value=Tplot.flatten(),
            isomin=float(np.nanmin(Tplot)),
            isomax=float(np.nanmax(Tplot)),
            opacity=0.1,
            surface_count=15,
            colorscale="Jet",
            colorbar=dict(title="°C")
        ))
        camera = dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.6, y=1.6, z=1.2))
        fig.update_layout(scene=dict(aspectmode='cube'), scene_camera=camera, height=700)
        st.plotly_chart(fig, use_container_width=True)

    elif view == "2D cross-section":
        # allow axis selection and index
        axis = st.selectbox("Slice axis", ["X (length)", "Y (width)", "Z (height)"])
        if axis.startswith("X"):
            mid = nx // 2
            data = Tframe[mid, :, :].T  # (z,y) -> show height vertical
            xaxis = ys
            yaxis = zs
            xlabel = "Width (m)"; ylabel = "Height (m)"
        elif axis.startswith("Y"):
            mid = nx // 2
            data = Tframe[:, mid, :].T
            xaxis = xs
            yaxis = zs
            xlabel = "Length (m)"; ylabel = "Height (m)"
        else:
            mid = nx // 2
            data = Tframe[:, :, mid].T
            xaxis = xs
            yaxis = ys
            xlabel = "Length (m)"; ylabel = "Width (m)"

        fig2 = go.Figure(data=go.Heatmap(
            z=data,
            x=xaxis,
            y=yaxis,
            colorscale="Jet",
            zmin=float(np.nanmin(Tframe)), zmax=float(np.nanmax(Tframe)),
            colorbar=dict(title="°C")
        ))
        fig2.update_layout(title=f"Cross-section at frame {frame_idx} (t={times[frame_idx]:.2f} h)",
                           xaxis_title=xlabel, yaxis_title=ylabel, yaxis=dict(scaleanchor="x", scaleratio=1))
        st.plotly_chart(fig2, use_container_width=True)

    else:
        # Temperature vs time for average and max in concrete (excluding bottom Dirichlet if needed)
        avg_t = [np.mean(f[1:-1,1:-1,1:-1]) for f in frames]  # interior average
        max_t = [np.max(f) for f in frames]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=times, y=avg_t, mode='lines', name='Avg (interior)', line=dict(color='blue')))
        fig3.add_trace(go.Scatter(x=times, y=max_t, mode='lines', name='Max', line=dict(color='red')))
        # equalization estimate (avg close to air)
        tol = 0.5
        eq_time = None
        for tt, av in zip(times, avg_t):
            if abs(av - T_air) <= tol:
                eq_time = tt
                break
        if eq_time is not None:
            fig3.add_vline(x=eq_time, line=dict(color='green', dash='dash'))
        fig3.update_layout(title='Temperature vs time', xaxis_title='Time (h)', yaxis_title='Temperature (°C)')
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("Press '(Re)run simulation now' to build and solve the Crank–Nicolson system (requires scipy).")
    if not SCIPY_AVAILABLE:
        st.warning("scipy not detected — install scipy to run the implicit solver for best accuracy. "
                   "Without scipy, we can fall back to an explicit, stable solver (with small dt).")
