# app.py
import streamlit as st
from PIL import Image
import io
from inference import sharpen_image

st.title("🔍 AI Image Sharpener")

uploaded_file = st.file_uploader("Upload an image to sharpen", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original", use_column_width=True)

    if st.button("Sharpen Image"):
        sharp = sharpen_image(image)
        st.image(sharp, caption="Sharpened", use_column_width=True)

        buf = io.BytesIO()
        sharp.save(buf, format="PNG")
        st.download_button("Download Sharpened Image", buf.getvalue(), file_name="sharpened.png")
