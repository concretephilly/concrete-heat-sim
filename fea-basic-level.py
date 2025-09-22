# ==============================================================
# ADDITION (hollow cores + diagnostics)
# ==============================================================

# Sidebar toggle for diagnostics
show_diag = st.sidebar.checkbox("Show 2D solver diagnostics", value=False)

# Define hollow cores (5 across width, centered vertically)
n_cores = 5
core_radius = 0.08  # 80 mm typical
core_y_center = L/2  # middle depth
x_centers = np.linspace(0.15, Lx-0.15, n_cores)  # evenly spaced

# Mask array for voids (air)
void_mask = np.zeros((Nz, Nx2d), dtype=bool)
X, Z = np.meshgrid(x, z)  # shape (Nz, Nx2d)
for cx in x_centers:
    r2 = (X - cx)**2 + (Z - core_y_center)**2
    void_mask |= (r2 <= core_radius**2)

# Adjust diffusivity field: concrete vs air
alpha_air = 2.0e-5  # much higher diffusivity than concrete
alpha_field = np.where(void_mask, alpha_air, alpha)

# Modify update to use spatially varying alpha (still explicit)
# We'll update the solver loop to use alpha_field instead of constant alpha.

# Reset 2D field with voids included
T2D = np.full((Nz, Nx2d), ambient_temp, dtype=float)
T2D[0, :] = bed_temp

snapshots = []
times2d = []
t_now = 0.0
target_idx = 0
last_target = target_times_s[-1]

for step in range(max_steps_2d):
    Tn = T2D.copy()

    # interior update with local alpha
    # (slightly slower than vectorized uniform-alpha version)
    for iz in range(1, Nz-1):
        for ix in range(1, Nx2d-1):
            a_loc = alpha_field[iz, ix]
            T2D[iz, ix] = (
                Tn[iz, ix]
                + a_loc * dt2d * (
                    (Tn[iz+1, ix] - 2*Tn[iz, ix] + Tn[iz-1, ix]) / dz**2
                    + (Tn[iz, ix+1] - 2*Tn[iz, ix] + Tn[iz, ix-1]) / dx**2
                )
            )

    # bottom row fixed
    T2D[0, :] = bed_temp

    # top convection
    T2D[-1, :] = (k * T2D[-2, :] + h * dz * ambient_temp) / (k + h * dz)

    # insulated sides
    T2D[:, 0] = T2D[:, 1]
    T2D[:, -1] = T2D[:, -2]

    t_now += dt2d

    # snapshot logic
    while target_idx < len(target_times_s) and t_now + 1e-12 >= target_times_s[target_idx]:
        snapshots.append(T2D.copy())
        times2d.append(target_times_s[target_idx] / 3600.0)
        target_idx += 1

    if target_idx >= len(target_times_s):
        break

# Plot
if snapshots:
    idx = st.slider("Select snapshot index (2D)", 0, len(snapshots)-1, 0)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    im = ax3.imshow(
        snapshots[idx],
        origin='lower',
        extent=[0.0, Lx, 0.0, L],
        aspect='auto',
        cmap='jet',
        vmin=all_min,
        vmax=all_max
    )
    # overlay hollow core outlines
    for cx in x_centers:
        circ = plt.Circle((cx, core_y_center), core_radius, color='black', fill=False, lw=1)
        ax3.add_patch(circ)
    fig3.colorbar(im, ax=ax3, label="Temperature (°C)")
    ax3.set_xlabel("Width (m)")
    ax3.set_ylabel("Depth (m) from bed (bottom=0)")
    ax3.set_title(f"2D temp distribution at {times2d[idx]:.2f} h")
    st.pyplot(fig3)

if show_diag:
    st.markdown("### 2D Solver Diagnostics")
    st.write(f"- dt2d = {dt2d:.4e} s (1D dt = {dt:.4e} s)")
    st.write(f"- max_steps_2d = {max_steps_2d}")
    st.write(f"- snapshots stored = {len(snapshots)}")
    st.write(f"- void fraction ≈ {np.mean(void_mask)*100:.1f}% of cross-section")
    st.write(f"- min temp in final snapshot = {np.min(snapshots[-1]):.2f} °C")
    st.write(f"- max temp in final snapshot = {np.max(snapshots[-1]):.2f} °C")
