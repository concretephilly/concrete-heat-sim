import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Slab Layout Optimizer", layout="wide")

st.title("🧱 Concrete Slab Layout Optimizer")

# Step 1: Draw site boundaries
st.subheader("1. Draw your site area")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Transparent orange fill
    stroke_width=2,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=400,
    width=600,
    drawing_mode="rect",  # Draw rectangles for area
    key="canvas",
)

# Step 2: Slab parameters
st.subheader("2. Enter slab parameters")
slab_length = st.number_input("Slab length (m)", min_value=0.1, value=2.0, step=0.1)
slab_width = st.number_input("Slab width (m)", min_value=0.1, value=1.0, step=0.1)
slab_count = st.number_input("Number of slabs", min_value=1, value=10, step=1)

# Step 3: Run optimizer
if st.button("Run Optimization"):
    st.subheader("3. Optimized Layout Plan")

    if canvas_result.json_data is None:
        st.warning("⚠️ Please draw your site boundary first.")
    else:
        # Extract bounding box of drawn site
        objs = canvas_result.json_data["objects"]
        if not objs:
            st.warning("⚠️ No boundary drawn.")
        else:
            site = objs[0]  # take the first rectangle drawn
            x, y, w, h = site["left"], site["top"], site["width"], site["height"]

            # Convert site dimensions to meters (arbitrary scaling factor)
            scale = 0.05  # pixels to meters
            site_w, site_h = w * scale, h * scale

            # Greedy packing algorithm: place slabs row by row
            layout = []
            x_pos, y_pos = 0, 0
            used_slabs = 0

            while y_pos + slab_width <= site_h and used_slabs < slab_count:
                while x_pos + slab_length <= site_w and used_slabs < slab_count:
                    layout.append((x_pos, y_pos))
                    x_pos += slab_length
                    used_slabs += 1
                x_pos = 0
                y_pos += slab_width

            # Plot the optimized layout
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(0, site_w)
            ax.set_ylim(0, site_h)
            ax.set_aspect('equal')

            # Draw site boundary
            ax.add_patch(plt.Rectangle((0, 0), site_w, site_h, fill=None, edgecolor='black', linewidth=2))

            # Draw slabs
            for (x_pos, y_pos) in layout:
                ax.add_patch(plt.Rectangle((x_pos, y_pos), slab_length, slab_width, fill=True, alpha=0.5))

            ax.set_title("Optimized Slab Placement")
            st.pyplot(fig)

            st.success(f"✅ Used {used_slabs} slabs out of {slab_count} available.")
