import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
import numpy as np

# Allow YOLO DetectionModel class for PyTorch 2.6+ safety
torch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"])

st.title("Issue Image Comparison (Before vs After)")

# Upload before and after images
uploaded_file1 = st.file_uploader("Upload Before Image", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Upload After Image", type=["jpg", "jpeg", "png"])

# Load YOLO model
model = YOLO("best.pt")  # path to your trained model

if uploaded_file1 and uploaded_file2:
    # Open images
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)

    # Show images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption="Before Image", use_column_width=True)
    with col2:
        st.image(image2, caption="After Image", use_column_width=True)

    st.subheader("Detection Results")

    # Run YOLO inference
    results_before = model(np.array(image1))
    results_after = model(np.array(image2))

    # Count detected objects (example: number of bounding boxes)
    count_before = len(results_before[0].boxes) if results_before[0].boxes is not None else 0
    count_after = len(results_after[0].boxes) if results_after[0].boxes is not None else 0

    st.write(f"🔴 Objects detected in Before Image: {count_before}")
    st.write(f"🟢 Objects detected in After Image: {count_after}")

    # Simple "resolved" accuracy calculation
    if count_before > 0:
        resolved_accuracy = max(0, (count_before - count_after) / count_before * 100)
    else:
        resolved_accuracy = 100 if count_after == 0 else 0

    st.subheader("Comparison Accuracy")
    st.write(f"✅ Resolved Accuracy: {resolved_accuracy:.2f} %")

    # Optionally show annotated images
    st.subheader("Annotated Images")
    annotated_before = Image.fromarray(results_before[0].plot())
    annotated_after = Image.fromarray(results_after[0].plot())

    col3, col4 = st.columns(2)
    with col3:
        st.image(annotated_before, caption="Before Image Detection", use_column_width=True)
    with col4:
        st.image(annotated_after, caption="After Image Detection", use_column_width=True)
