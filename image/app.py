import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(page_title="Image Enhancement & Segmentation", layout="wide")

st.title("🎨 Image Enhancement and Segmentation App")
st.write(
    "Upload an image to enhance contrast, visualize histograms, "
    "and segment automatically using Otsu’s method."
)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image properly in RGB
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    # Contrast Enhancement (Histogram Equalization)
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # Grayscale + Otsu Segmentation
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display original vs enhanced
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Original Image", use_container_width=True)
    with col2:
        st.image(enhanced, caption="Enhanced Image", use_container_width=True)

    # Histogram comparison
    st.subheader("📊 RGB Histogram Comparison")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    colors = ("r", "g", "b")
    for i, col in enumerate(colors):
        ax[0].hist(img_rgb[:, :, i].ravel(), bins=256, color=col, alpha=0.6)
        ax[1].hist(enhanced[:, :, i].ravel(), bins=256, color=col, alpha=0.6)
    ax[0].set_title("Original Histogram")
    ax[1].set_title("Enhanced Histogram")
    st.pyplot(fig)

    # Segmentation Result
    st.subheader("Segmentation Result (Otsu’s Threshold)")
    st.image(segmented, caption="Segmented Image", use_container_width=True)

    # Convert images for download
    enhanced_pil = Image.fromarray(enhanced)
    segmented_pil = Image.fromarray(segmented)

    buf1 = io.BytesIO()
    buf2 = io.BytesIO()
    enhanced_pil.save(buf1, format="JPEG")
    segmented_pil.save(buf2, format="JPEG")
    buf1.seek(0)
    buf2.seek(0)

    st.download_button(
        label="⬇️ Download Enhanced Image",
        data=buf1,
        file_name="enhanced_image.jpg",
        mime="image/jpeg"
    )

    st.download_button(
        label="⬇️ Download Segmented Image",
        data=buf2,
        file_name="segmented_image.jpg",
        mime="image/jpeg"
    )

    st.success("Image processed successfully!")
