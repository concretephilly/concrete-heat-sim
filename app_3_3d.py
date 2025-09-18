# app.py
import time
import math
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Try to import scipy.sparse; CN solver relies on it
try:
    from scipy.sparse import diags, identity, kron, csr_matrix
    from scipy.sparse.linalg import spsolve
    SCIPY = True
except Exception:
    SCIPY = False

st.set_page_config(layout="wide", page_title="Concrete Cube — CN Heat Solver (CEM III defaults)")
st.title("Concrete cube on heated plate — Crank–Nicolson (CEM III defaults)")

# -------------------------
# session state init
# -------------------------
if "frames" not in st.session_state:
    st.session_state.frames = None
if "times" not in st.session_state:
    st.session_state.times = None
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "view" not in st.session_state:
    st.session_state.view = "3D Volume"
if "playing" not in st.session_state:
    st.session_state.playing = False
if "last_run_hash" not in st.session_state:
    st.session_state.last_run_hash = None

# -------------------------
# Sidebar: inputs (all options)
# -------------------------
st.sidebar.header("Domain / Grid")
Lx = st.sidebar.number_input("Block length X (m)", min_value=0.01, max_value=1.0, value=0.10, step=0.01)
Ly = st.sidebar.number_input("Block width Y (m)",  min_value=0.01, max_value=1.0, value=0.10, step=0.01)
Lz = st.sidebar.number_input("Block height Z (m)", min_value=0.01, max_value=1.0, value=0.10, step=0.01)

n_side = st.sidebar.slider("Grid resolution (per side, cubic grid)", 6, 40, 20, 1,
                           help="Cubic grid: nx = ny = nz = n_side. Higher = more accurate but slower.")
nx = ny = nz = int(n_side)
dx = Lx / (nx - 1)
inv_dx2 = 1.0 / (dx * dx)

st.sidebar.header("Plate ramp & environment")
plate_start = st.sidebar.number_input("Plate start temp (°C) at t=0", value=20.0)
plate_end = st.sidebar.number_input("Plate end temp (°C) after ramp", value=60.0)
ramp_hours = st.sidebar.number_input("Plate ramp duration (hours)", min_value=0.0, max_value=72.0, value=2.0)
T_air = st.sidebar.number_input("Ambient air temp (°C)", value=20.0)
h_conv = st.sidebar.number_input("Convective h (W/m²·K)", min_value=0.5, max_value=100.0, value=15.0)

st.sidebar.header("Concrete (CEM III defaults)")
rho = st.sidebar.number_input("Density ρ (kg/m³)", 1000.0, 3000.0, 2300.0)
cp = st.sidebar.number_input("Specific heat c_p (J/kg·K)", 500.0, 3000.0, 1600.0)
k_cond = st.sidebar.number_input("Thermal conductivity k (W/m·K)", 0.1, 5.0, 1.2)

alpha = k_cond / (rho * cp)

st.sidebar.header("Hydration (linear-decay source)")
Q_gen_peak = st.sidebar.number_input("Initial hydration power (W/m³)", min_value=0.0, max_value=200.0, value=3.0,
                                      help="Initial volumetric heat generation (W/m³), decays linearly to zero over release_hours.")
release_hours = st.sidebar.number_input("Hydration release duration (hours)", min_value=0.1, max_value=168.0, value=48.0)

st.sidebar.header("Time & solver")
sim_hours = st.sidebar.number_input("Simulation total duration (hours)", min_value=0.1, max_value=168.0, value=24.0)
dt_seconds = st.sidebar.number_input("Requested timestep (s)", min_value=0.1, max_value=3600.0, value=60.0)
n_steps = max(1, int(math.ceil(sim_hours * 3600.0 / float(dt_seconds))))

st.sidebar.header("Playback")
frames_per_play = st.sidebar.slider("Frames / Play click", 1, 120, 20)
play_fps = st.sidebar.slider("Play FPS", 1, 30, 6)

st.sidebar.markdown("**Notes:** Crank–Nicolson uses sparse linear solves (requires `scipy`). If SciPy is not present a stable explicit fallback is used (slower/smaller grids).")

run_button = st.sidebar.button("(Re)run simulation now")

# -------------------------
# helper functions
# -------------------------
def plate_temperature_at_hour(t_h):
    """Linear ramp from plate_start to plate_end over ramp_hours (hours)."""
    if ramp_hours <= 0.0:
        return float(plate_end)
    frac = min(max(t_h / ramp_hours, 0.0), 1.0)
    return float(plate_start) + frac * (float(plate_end) - float(plate_start))

def q_hydration(t_s):
    """Linear-decay hydration power (W/m3) at time t_s (seconds).
       initial = Q_gen_peak at t=0, decays linearly to 0 at release_hours."""
    t_h = t_s / 3600.0
    if release_hours <= 0.0:
        return 0.0
    frac = max(0.0, 1.0 - t_h / float(release_hours))
    return float(Q_gen_peak) * frac

# Build 1D second-difference operator and 3D Laplacian (sparse)
def build_laplacian_3d(n, inv_dx2):
    # D1 = tridiag [1, -2, 1] * inv_dx2
    from scipy.sparse import diags, identity, kron
    e = np.ones(n)
    D1 = diags([e, -2*e, e], offsets=[-1, 0, 1], shape=(n, n), format='csr') * inv_dx2
    I = identity(n, format='csr')
    L = kron(kron(D1, I), I) + kron(kron(I, D1), I) + kron(kron(I, I), D1)
    return L.tocsr()

# small helper to get 1D coordinates for plotting (meters)
xs = np.linspace(0, Lx, nx)
ys = np.linspace(0, Ly, ny)
zs = np.linspace(0, Lz, nz)

# progress ETA utility
def secs_to_hms(s):
    s = max(0.0, float(s))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h}h {m}m {sec}s" if h else (f"{m}m {sec}s" if m else f"{sec}s")

# -------------------------
# Solver (Crank–Nicolson) - heavy part
# -------------------------
def run_crank_nicolson_solver():
    if not SCIPY:
        st.error("SciPy not found. Install scipy to use the Crank–Nicolson implicit solver. Falling back to stable explicit solver.")
        return run_explicit_fallback()

    # Precompute sparse Laplacian
    N = nx * ny * nz
    dt = float(dt_seconds)
    L = build_laplacian_3d(nx, inv_dx2)  # using same spacing for simplicity
    I = identity(N, format='csr')

    # CN matrices: M_left = I - 0.5 * alpha * dt * L ; M_right = I + 0.5 * alpha * dt * L
    M_left = (I - 0.5 * alpha * dt * L).tocsr()
    M_right = (I + 0.5 * alpha * dt * L).tocsr()

    # index helper: flatten indexing i,j,k -> idx
    def idx(i, j, k):
        return (i * ny + j) * nz + k

    # bottom nodes (k=0) for Dirichlet enforcement
    bottom_nodes = [idx(i, j, 0) for i in range(nx) for j in range(ny)]

    # face node lists for explicit convective correction
    top_nodes = [idx(i, j, nz - 1) for i in range(nx) for j in range(ny)]
    i0_nodes = [idx(0, j, k) for j in range(ny) for k in range(nz)]
    iN_nodes = [idx(nx - 1, j, k) for j in range(ny) for k in range(nz)]
    j0_nodes = [idx(i, 0, k) for i in range(nx) for k in range(nz)]
    jN_nodes = [idx(i, ny - 1, k) for i in range(nx) for k in range(nz)]

    # approximate A/V for face convective term: area per voxel / volume per voxel ~ 1/dx
    AoverV = 1.0 / dx

    # Initial temperature flattened (start at ambient)
    Tn = np.ones((N,), dtype=np.float64) * float(T_air)
    # set initial bottom nodes to plate_start
    for r in bottom_nodes:
        Tn[r] = float(plate_start)

    frames = []
    times_h = []
    start_wall = time.time()
    progress = st.progress(0)
    last_update = 0

    for step in range(n_steps):
        t_s = step * dt
        t_h = t_s / 3600.0

        # plate now
        plate_now = plate_temperature_at_hour(t_h)

        # volumetric source (trapezoidal rule, J/m3/s -> temperature via / (rho*cp) )
        qn = q_hydration(t_s)
        qnp1 = q_hydration(t_s + dt)
        s_temp = (dt * 0.5) * (qn + qnp1) / (rho * cp)  # K per node added to rhs

        # rhs from M_right
        rhs = M_right.dot(Tn)

        # add volumetric source uniformly
        if s_temp != 0.0:
            rhs = rhs + s_temp

        # explicit convective correction (approx) on faces: ΔT_conv = - (h*A/V)/(rho*cp) * (Tn - T_air) * dt
        conv_factor = (h_conv * AoverV * dt) / (rho * cp)
        if conv_factor != 0.0:
            # top
            for r in top_nodes:
                rhs[r] += -conv_factor * (Tn[r] - float(T_air))
            # i faces
            for r in i0_nodes:
                if r not in bottom_nodes:
                    rhs[r] += -conv_factor * (Tn[r] - float(T_air))
            for r in iN_nodes:
                if r not in bottom_nodes:
                    rhs[r] += -conv_factor * (Tn[r] - float(T_air))
            # j faces
            for r in j0_nodes:
                if r not in bottom_nodes:
                    rhs[r] += -conv_factor * (Tn[r] - float(T_air))
            for r in jN_nodes:
                if r not in bottom_nodes:
                    rhs[r] += -conv_factor * (Tn[r] - float(T_air))

        # Solve linear system M_left * Tnp1 = rhs
        try:
            Tnp1 = spsolve(M_left, rhs)
        except Exception as e:
            st.error(f"Linear solver failed: {e}")
            break

        # enforce bottom Dirichlet exactly
        for r in bottom_nodes:
            Tnp1[r] = plate_now

        # clamp to realistic bounds
        Tnp1 = np.maximum(Tnp1, float(T_air) - 50.0)  # allow some below ambient but not ridiculous

        Tn = Tnp1

        # store snapshots every ~1% up to ~200 frames
        if (step % max(1, n_steps // 200) == 0) or (step == n_steps - 1):
            frames.append(Tn.copy().reshape((nx, ny, nz)))
            times_h.append(t_h)

        # progress + ETA
        if (step - last_update) >= max(1, n_steps // 100):
            elapsed = time.time() - start_wall
            fraction = (step + 1) / float(n_steps)
            est_total = elapsed / fraction if fraction > 0 else 0.0
            est_remain = max(0.0, est_total - elapsed)
            progress.progress(int(fraction * 100), text=f"{int(fraction*100)}% — ETA {secs_to_hms(est_remain)}")
            last_update = step

    progress.progress(100, text="Done")
    return frames, times_h

# explicit fallback (stable) if scipy missing
def run_explicit_fallback():
    st.warning("Running explicit, stable fallback solver. This may be slower and needs small dt for stability.")
    # stability limit
    dt_stable = dx * dx / (6.0 * alpha) if alpha > 0 else 1.0
    dt_used = float(dt_seconds)
    if dt_used > dt_stable:
        st.warning(f"Requested dt ({dt_used}s) > explicit stability limit ({dt_stable:.2f}s). Reducing dt to 80% of limit.")
        dt_used = 0.8 * dt_stable
    # compute steps
    steps = max(1, int(math.ceil(sim_hours * 3600.0 / dt_used)))
    Tn = np.ones((nx * ny * nz,), dtype=np.float64) * float(T_air)
    def idx(i, j, k): return (i * ny + j) * nz + k
    bottom_nodes = [idx(i, j, 0) for i in range(nx) for j in range(ny)]
    frames = []
    times_h = []
    for step in range(steps):
        t_s = step * dt_used
        t_h = t_s / 3600.0
        plate_now = plate_temperature_at_hour(t_h)
        T3 = Tn.reshape((nx, ny, nz)).copy()
        Tnew3 = T3.copy()
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    lap = (T3[i+1,j,k] + T3[i-1,j,k] + T3[i,j+1,k] + T3[i,j-1,k] + T3[i,j,k+1] + T3[i,j,k-1] - 6.0*T3[i,j,k]) * inv_dx2
                    Tnew3[i,j,k] = T3[i,j,k] + alpha * dt_used * lap + (q_hydration(t_s) / (rho * cp)) * dt_used
        # enforce bottom plate
        Tnew3[:,:,0] = plate_now
        Tn = Tnew3.flatten()
        if (step % max(1, steps // 200) == 0) or (step == steps - 1):
            frames.append(Tn.copy().reshape((nx, ny, nz)))
            times_h.append(t_h)
    return frames, times_h

# -------------------------
# Run solver when requested
# -------------------------
# compute a simple hash of input parameters to detect changes and avoid unnecessary reruns
def params_hash():
    return hash((
        nx, ny, nz, Lx, Ly, Lz,
        plate_start, plate_end, ramp_hours,
        T_air, h_conv,
        rho, cp, k_cond,
        Q_gen_peak, release_hours,
        sim_hours, dt_seconds
    ))

if run_button:
    cur_hash = params_hash()
    st.session_state.last_run_hash = cur_hash
    st.info("Starting simulation (this may take a while for fine grids)...")
    start_total = time.time()
    if SCIPY:
        frames_out, times_out = run_crank_nicolson_solver()
    else:
        frames_out, times_out = run_explicit_fallback()
    st.session_state.frames = frames_out
    st.session_state.times = times_out
    st.session_state.frame_idx = 0
    st.success(f"Simulation finished in {secs_to_hms(time.time() - start_total)} — {len(frames_out)} frames saved.")

# If simulation results in session_state, use them
frames = st.session_state.frames
times_h = st.session_state.times

# -------------------------
# Visualization controls (no duplicate sliders)
# -------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.session_state.view = st.selectbox("View", ["3D Volume", "2D cross-section", "Temperature vs time"],
                                         index=["3D Volume", "2D cross-section", "Temperature vs time"].index(st.session_state.view))

with col_right:
    # Play / Pause / Next / Prev
    pcol1, pcol2, pcol3, pcol4 = st.columns([1,1,1,1])
    with pcol1:
        if st.button("▶ Play"):
            st.session_state.playing = True
    with pcol2:
        if st.button("⏸ Pause"):
            st.session_state.playing = False
    with pcol3:
        if st.button("⟸ Prev"):
            if frames is not None:
                st.session_state.frame_idx = (st.session_state.frame_idx - 1) % len(frames)
    with pcol4:
        if st.button("Next ⟹"):
            if frames is not None:
                st.session_state.frame_idx = (st.session_state.frame_idx + 1) % len(frames)

# single persistent slider
if frames is not None:
    nframes = len(frames)
    # ensure frame_idx bounds
    if st.session_state.frame_idx >= nframes:
        st.session_state.frame_idx = nframes - 1
    st.session_state.frame_idx = st.slider("Time frame index", 0, nframes - 1, st.session_state.frame_idx, key="time_slider")

    # a placeholder for chart updates during playback
    placeholder = st.empty()

    # Playback: advance a small batch when Play is pressed
    if st.session_state.playing:
        batch = int(min(frames_per_play, nframes))
        sleep_time = 1.0 / max(1.0, float(play_fps))
        for _ in range(batch):
            st.session_state.frame_idx = (st.session_state.frame_idx + 1) % nframes
            idx = st.session_state.frame_idx
            Tframe = frames[idx]
            t_label = f"t = {times_h[idx]:.2f} h"
            # render according to chosen view into placeholder
            if st.session_state.view == "3D Volume":
                # downsample if needed
                max_plot = 40
                nplot = Tframe.shape[0]
                if nplot > max_plot:
                    step_ds = int(math.ceil(nplot / max_plot))
                    Tplot = Tframe[::step_ds, ::step_ds, ::step_ds]
                    xs_plot = np.linspace(0, Lx, Tplot.shape[0])
                    ys_plot = np.linspace(0, Ly, Tplot.shape[1])
                    zs_plot = np.linspace(0, Lz, Tplot.shape[2])
                else:
                    Tplot = Tframe
                    xs_plot = np.linspace(0, Lx, nplot)
                    ys_plot = np.linspace(0, Ly, nplot)
                    zs_plot = np.linspace(0, Lz, nplot)
                xf = np.repeat(xs_plot, len(ys_plot) * len(zs_plot))
                yf = np.tile(np.repeat(ys_plot, len(zs_plot)), len(xs_plot))
                zf = np.tile(zs_plot, len(xs_plot) * len(ys_plot))
                fig = go.Figure(data=go.Volume(
                    x=xf, y=yf, z=zf, value=Tplot.flatten(),
                    isomin=float(np.nanmin(Tplot)), isomax=float(np.nanmax(Tplot)),
                    opacity=0.1, surface_count=15, colorscale="Jet", colorbar=dict(title="°C")
                ))
                fig.update_layout(scene=dict(aspectmode='data'), title=t_label, uirevision='view')
                placeholder.plotly_chart(fig, use_container_width=True)
            elif st.session_state.view == "2D cross-section":
                mid = Tframe.shape[2] // 2
                fig = go.Figure(data=go.Heatmap(
                    z=Tframe[:, :, mid].T,
                    x=np.linspace(0, Lx, Tframe.shape[0]),
                    y=np.linspace(0, Ly, Tframe.shape[1]),
                    colorscale="Jet",
                    zmin=float(np.nanmin(Tframe)), zmax=float(np.nanmax(Tframe)),
                    colorbar=dict(title="°C")
                ))
                fig.update_layout(title=f"Cross-section {t_label}", yaxis=dict(scaleanchor="x", scaleratio=1))
                placeholder.plotly_chart(fig, use_container_width=True)
            else:
                # time history plot
                avg = [np.mean(f) for f in frames]
                mx = [np.max(f) for f in frames]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=times_h, y=avg, mode='lines', name='Avg', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=times_h, y=mx, mode='lines', name='Max', line=dict(color='red')))
                fig.add_vline(x=times_h[idx], line=dict(color='gray', dash='dash'))
                fig.update_layout(title="Temperature vs time", xaxis_title="Time (h)", yaxis_title="°C")
                placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(sleep_time)
        # stop playing automatically (batch mode)
        st.session_state.playing = False

    # If not playing, render the selected frame normally (outside placeholder)
    if not st.session_state.playing:
        idx = st.session_state.frame_idx
        Tframe = frames[idx]
        t_label = f"t = {times_h[idx]:.2f} h"

        if st.session_state.view == "3D Volume":
            # same rendering as above but single display
            max_plot = 40
            nplot = Tframe.shape[0]
            if nplot > max_plot:
                step_ds = int(math.ceil(nplot / max_plot))
                Tplot = Tframe[::step_ds, ::step_ds, ::step_ds]
                xs_plot = np.linspace(0, Lx, Tplot.shape[0])
                ys_plot = np.linspace(0, Ly, Tplot.shape[1])
                zs_plot = np.linspace(0, Lz, Tplot.shape[2])
            else:
                Tplot = Tframe
                xs_plot = np.linspace(0, Lx, nplot)
                ys_plot = np.linspace(0, Ly, nplot)
                zs_plot = np.linspace(0, Lz, nplot)
            xf = np.repeat(xs_plot, len(ys_plot) * len(zs_plot))
            yf = np.tile(np.repeat(ys_plot, len(zs_plot)), len(xs_plot))
            zf = np.tile(zs_plot, len(xs_plot) * len(ys_plot))
            fig = go.Figure(data=go.Volume(
                x=xf, y=yf, z=zf, value=Tplot.flatten(),
                isomin=float(np.nanmin(Tplot)), isomax=float(np.nanmax(Tplot)),
                opacity=0.1, surface_count=15, colorscale="Jet", colorbar=dict(title="°C")
            ))
            fig.update_layout(scene=dict(aspectmode='data'), title=t_label, uirevision='view')
            st.plotly_chart(fig, use_container_width=True)

        elif st.session_state.view == "2D cross-section":
            mid = Tframe.shape[2] // 2
            fig = go.Figure(data=go.Heatmap(
                z=Tframe[:, :, mid].T,
                x=np.linspace(0, Lx, Tframe.shape[0]),
                y=np.linspace(0, Ly, Tframe.shape[1]),
                colorscale="Jet",
                zmin=float(np.nanmin(Tframe)), zmax=float(np.nanmax(Tframe)),
                colorbar=dict(title="°C")
            ))
            fig.update_layout(title=f"Cross-section {t_label}", yaxis=dict(scaleanchor="x", scaleratio=1))
            st.plotly_chart(fig, use_container_width=True)

        else:
            avg = [np.mean(f) for f in frames]
            mx = [np.max(f) for f in frames]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times_h, y=avg, mode='lines', name='Avg', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=times_h, y=mx, mode='lines', name='Max', line=dict(color='red')))
            fig.add_vline(x=times_h[idx], line=dict(color='gray', dash='dash'))
            fig.update_layout(title="Temperature vs time", xaxis_title="Time (h)", yaxis_title="°C")
            st.plotly_chart(fig, use_container_width=True)

    # Estimate time to equalize (center vs ambient or plate)
    center_vals = [np.mean(f[nx//2, ny//2, nz//2]) for f in frames]
    if len(center_vals) >= 2:
        dt_sample = max(1e-6, (times_h[1] - times_h[0]))  # hours
        rate = (center_vals[-1] - center_vals[-2]) / dt_sample  # °C per hour
        if abs(rate) < 1e-6:
            st.info("Temperature change very small — near equilibrium")
        else:
            # if center trending up, target is plate_end, else target is ambient
            if rate > 0:
                target = plate_end
            else:
                target = T_air
            hours_to_eq = (target - center_vals[-1]) / rate if abs(rate) > 1e-9 else math.inf
            if hours_to_eq < 0:
                st.info("Center already past target (no further time to equalize).")
            else:
                st.info(f"Estimated time to reach near target: {hours_to_eq:.1f} hours (extrapolated)")
else:
    st.info("No simulation results yet. Press '(Re)run simulation now' in the sidebar to start.")
    if not SCIPY:
        st.warning("SciPy not detected: CN implicit solver unavailable; an explicit fallback will be used when you press Run.")
