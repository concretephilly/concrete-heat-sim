# --- Simulation length setting ---
total_hours = st.number_input("Simulation length (hours)", min_value=1, max_value=72, value=24, step=1)

# --- Slider (single, persistent) ---
if "time_idx" not in st.session_state:
    st.session_state.time_idx = 0

time_idx = st.slider("Time (hours)", 0, total_hours, st.session_state.time_idx, step=1, key="time_slider")

# --- Play button ---
if "playing" not in st.session_state:
    st.session_state.playing = False

if st.button("▶ Play"):
    st.session_state.playing = True
if st.button("⏸ Pause"):
    st.session_state.playing = False

# --- Animate by updating session state ---
if st.session_state.playing:
    for t in range(time_idx, total_hours + 1):
        st.session_state.time_idx = t
        time.sleep(0.2)  # playback speed
        st.experimental_rerun()

# Use session state for consistent indexing
t = st.session_state.time_idx
