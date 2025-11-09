import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(page_title="Ultimate Image Processing App", layout="wide")

st.title("🎨 Ultimate Image Processing and Enhancement App")
st.write("Upload an image and apply multiple image processing techniques!")

uploaded_file = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    # Sidebar options
    st.sidebar.header("🧩 Select Image Processing Features")
    enhance = st.sidebar.checkbox("Contrast Enhancement", False)
    grayscale = st.sidebar.checkbox("Grayscale Conversion", False)
    blur = st.sidebar.checkbox("Blurring", False)
    edge = st.sidebar.checkbox("Edge Detection (Canny)", False)
    sharpen = st.sidebar.checkbox("Sharpening", False)
    segment = st.sidebar.checkbox("Segmentation (Otsu’s Threshold)", False)
    morph = st.sidebar.checkbox("Morphological Operations", False)
    transform = st.sidebar.checkbox("Geometric Transformations", False)
    color_ops = st.sidebar.checkbox("Color Transformations", False)
    noise_ops = st.sidebar.checkbox("Noise Addition / Denoising", False)

    processed = img_rgb.copy()

    # --- Enhancement ---
    if enhance:
        img_yuv = cv2.cvtColor(processed, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        processed = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # --- Grayscale ---
    if grayscale:
        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)

    # --- Blurring ---
    if blur:
        blur_type = st.sidebar.selectbox("Select Blur Type", ["Gaussian", "Median", "Bilateral"])
        if blur_type == "Gaussian":
            processed = cv2.GaussianBlur(processed, (5, 5), 0)
        elif blur_type == "Median":
            processed = cv2.medianBlur(processed, 5)
        else:
            processed = cv2.bilateralFilter(processed, 9, 75, 75)

    # --- Edge Detection ---
    if edge:
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            gray = processed
        processed = cv2.Canny(gray, 100, 200)

    # --- Sharpening ---
    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        processed = cv2.filter2D(processed, -1, kernel)

    # --- Segmentation ---
    if segment:
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            gray = processed
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Morphological Operations ---
    if morph:
        kernel = np.ones((5, 5), np.uint8)
        morph_type = st.sidebar.selectbox("Morphological Operation", ["Erosion", "Dilation", "Opening", "Closing"])
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        if morph_type == "Erosion":
            processed = cv2.erode(binary, kernel, iterations=1)
        elif morph_type == "Dilation":
            processed = cv2.dilate(binary, kernel, iterations=1)
        elif morph_type == "Opening":
            processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif morph_type == "Closing":
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # --- Geometric Transformations ---
    if transform:
        trans_type = st.sidebar.selectbox("Transformation", ["Rotate 90°", "Flip Horizontally", "Flip Vertically", "Resize 50%"])
        if trans_type == "Rotate 90°":
            processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
        elif trans_type == "Flip Horizontally":
            processed = cv2.flip(processed, 1)
        elif trans_type == "Flip Vertically":
            processed = cv2.flip(processed, 0)
        elif trans_type == "Resize 50%":
            processed = cv2.resize(processed, None, fx=0.5, fy=0.5)

    # --- Color Transformations ---
    if color_ops:
        color_type = st.sidebar.selectbox("Color Operation", ["Negative", "Gamma Correction"])
        if color_type == "Negative":
            processed = cv2.bitwise_not(processed)
        elif color_type == "Gamma Correction":
            gamma = st.sidebar.slider("Gamma Value", 0.1, 3.0, 1.2)
            processed = np.array(255 * ((processed / 255) ** (1 / gamma)), dtype=np.uint8)

    # --- Noise Operations ---
    if noise_ops:
        noise_type = st.sidebar.selectbox("Noise Option", ["Add Salt & Pepper Noise", "Denoise"])
        if noise_type == "Add Salt & Pepper Noise":
            row, col, ch = processed.shape
            s_vs_p = 0.5
            amount = 0.02
            out = np.copy(processed)
            num_salt = np.ceil(amount * processed.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in processed.shape]
            out[coords[0], coords[1], :] = 255
            num_pepper = np.ceil(amount * processed.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in processed.shape]
            out[coords[0], coords[1], :] = 0
            processed = out
        elif noise_type == "Denoise":
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)

    # --- Display Results ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="🖼️ Original Image", use_container_width=True)
    with col2:
        st.image(processed, caption="✨ Processed Image", use_container_width=True)

    # --- Histogram ---
    if len(img_rgb.shape) == 3 and enhance:
        st.subheader("📊 Histogram Comparison")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        colors = ("r", "g", "b")
        for i, col in enumerate(colors):
            ax[0].hist(img_rgb[:, :, i].ravel(), bins=256, color=col, alpha=0.6)
            if processed.ndim == 3:
                ax[1].hist(processed[:, :, i].ravel(), bins=256, color=col, alpha=0.6)
        ax[0].set_title("Original Histogram")
        ax[1].set_title("Processed Histogram")
        st.pyplot(fig)

    # --- Download processed image ---
    processed_pil = Image.fromarray(processed if processed.ndim == 2 else cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    buf = io.BytesIO()
    processed_pil.save(buf, format="JPEG")
    buf.seek(0)

    st.download_button(
        label="⬇️ Download Processed Image",
        data=buf,
        file_name="processed_image.jpg",
        mime="image/jpeg"
    )



