```python
import streamlit as st
import matplotlib.pyplot as plt
import math

# --- Core slab placement logic ---
def cover_length_with_slabs(target_length, available_lengths, min_cut=0.6):
    available_lengths = sorted(available_lengths, reverse=True)
    result = []
    remaining = target_length
    for L in available_lengths:
        while remaining >= L:
            result.append(L)
            remaining -= L
    # handle remainder
    if remaining > 0:
        for L in available_lengths:
            if L - remaining >= min_cut:
                result.append(remaining)
                remaining = 0
                break
    return result if remaining == 0 else None

def layout_floorplan(width, length, slab_width, available_lengths, min_cut=0.6):
    rows = math.floor(width / slab_width)
    remainder = width - rows * slab_width
    row_widths = [slab_width] * rows
    if remainder >= min_cut:
        row_widths.append(remainder)

    placements = []
    total_pieces = 0
    for idx, rw in enumerate(row_widths):
        row = cover_length_with_slabs(length, available_lengths, min_cut)
        if not row:
            return None
        placements.append(row)
        total_pieces += len(row)

    return {
        "rows": row_widths,
        "placements": placements,
        "total_pieces": total_pieces
    }

# --- Streamlit UI ---
st.title("Hollow-core Slab Planner")

with st.sidebar:
    st.header("Input Parameters")
    floor_width = st.number_input("Floor Width (m)", min_value=1.0, value=6.0, step=0.1)
    floor_length = st.number_input("Floor Length (m)", min_value=1.0, value=10.0, step=0.1)
    slab_width = st.number_input("Slab Width (m)", min_value=0.1, value=1.2, step=0.1)
    available_lengths = st.text_input("Available Slab Lengths (comma separated)", "3,4,6,8,10,12")
    min_cut = st.number_input("Minimum Cut Length (m)", min_value=0.1, value=0.6, step=0.1)

    run_button = st.button("Run")

if run_button:
    try:
        avail_lengths = [float(x.strip()) for x in available_lengths.split(",") if x.strip()]
        solution = layout_floorplan(floor_width, floor_length, slab_width, avail_lengths, min_cut)
    except Exception as e:
        st.error(f"Invalid input: {e}")
        solution = None

    if not solution:
        st.error("No valid layout found with given parameters.")
    else:
        st.success(f"Total pieces: {solution['total_pieces']}")

        # Plot layout
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_aspect("equal")
        ax.set_xlim(0, floor_length)
        ax.set_ylim(0, floor_width)
        ax.set_title(f"Optimised Layout - {solution['total_pieces']} pieces")

        y = 0
        for row_w, pieces in zip(solution["rows"], solution["placements"]):
            x = 0
            for p in pieces:
                rect = plt.Rectangle((x, y), p, row_w, fill=None, edgecolor="blue")
                ax.add_patch(rect)
                x += p
            y += row_w

        st.pyplot(fig)
```
