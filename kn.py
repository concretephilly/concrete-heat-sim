import streamlit as st
import time

st.set_page_config(page_title="Knoetteke Clicker", layout="centered")

st.title("🖱️ Knoetteke Clicker")

# Placeholder for displaying text
placeholder = st.empty()

# Button to simulate click
if st.button("Click me!"):
    placeholder.markdown("<h2 style='text-align:center;'>knoetteke</h2>", unsafe_allow_html=True)
    time.sleep(3)
    placeholder.empty()
