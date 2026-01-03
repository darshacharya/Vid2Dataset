import streamlit as st
import cv2
import tempfile
import zipfile
import os
from PIL import Image
import numpy as np
import time

st.set_page_config(
    page_title="Video to Image Dataset",
    layout="centered",
    page_icon="🎥"
)

# Initialize session state
if 'cancel_extraction' not in st.session_state:
    st.session_state.cancel_extraction = False
if 'extracting' not in st.session_state:
    st.session_state.extracting = False

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        flex: 1;
        margin: 0 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.3rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
    }
    div[data-testid="stButton"] button[kind="secondary"] {
        background: linear-gradient(120deg, #f56565 0%, #c53030 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎥 Video to Image Dataset</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Convert your videos into structured image datasets for ML training</p>', unsafe_allow_html=True)

# Upload section
st.markdown("### 📤 Upload Video")
uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed"
)

if uploaded_video is not None:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Failed to open video. Please try another file.")
        st.stop()
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Display video information
    st.markdown("### 📊 Video Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{original_fps:.1f}</div>
            <div class="metric-label">FPS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{total_frames:,}</div>
            <div class="metric-label">Total Frames</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{duration:.1f}s</div>
            <div class="metric-label">Duration</div>
        </div>
        """, unsafe_allow_html=True)
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{width}×{height}</div>
            <div class="metric-label">Resolution</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{uploaded_video.size / (1024*1024):.1f} MB</div>
            <div class="metric-label">File Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Extraction settings
    st.markdown("### ⚙️ Extraction Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        extract_fps = st.slider(
            "Frames per second",
            min_value=1,
            max_value=int(original_fps),
            value=min(5, int(original_fps)),
            help="Higher values extract more frames"
        )
    
    with col2:
        output_format = st.selectbox(
            "Image format",
            ["jpg", "png"],
            help="PNG is lossless but larger, JPG is compressed"
        )
    
    # Calculate estimated output
    estimated_frames = int((total_frames / original_fps) * extract_fps)
    st.info(f"📋 Estimated output: **{estimated_frames:,}** images")
    
    # Generate and Cancel buttons
    col1, col2 = st.columns([3, 1])
    
    with col1:
        generate_clicked = st.button("🚀 Generate Dataset", disabled=st.session_state.extracting)
    
    with col2:
        if st.button("❌ Cancel", type="secondary", disabled=not st.session_state.extracting):
            st.session_state.cancel_extraction = True
    
    # Placeholders for dynamic content
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    result_placeholder = st.empty()
    preview_placeholder = st.empty()
    
    if generate_clicked:
        st.session_state.cancel_extraction = False
        st.session_state.extracting = True
        st.rerun()
    
    if st.session_state.extracting:
        progress_bar = progress_placeholder.progress(0)
        status_text = status_placeholder.empty()
        
        status_text.text("Initializing extraction...")
        
        frame_interval = int(original_fps // extract_fps)
        extracted_frames = 0
        temp_dir = tempfile.mkdtemp()
        frame_count = 0
        preview_frames = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        
        cancelled = False
        
        try:
            while True:
                # Check for cancellation
                if st.session_state.cancel_extraction:
                    status_text.text("⚠️ Cancelling extraction...")
                    cancelled = True
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    img_name = f"frame_{extracted_frames:05d}.{output_format}"
                    img_path = os.path.join(temp_dir, img_name)
                    img.save(img_path)
                    
                    # Store first 5 frames for preview
                    if extracted_frames < 5:
                        preview_frames.append(img_path)
                    
                    extracted_frames += 1
                    
                    # Update progress
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Extracting frames... {extracted_frames} extracted")
                
                frame_count += 1
        
        finally:
            cap.release()
            st.session_state.extracting = False
        
        if cancelled:
            progress_placeholder.empty()
            status_placeholder.empty()
            result_placeholder.warning(f"⚠️ Extraction cancelled. {extracted_frames} frames were extracted before cancellation.")
            st.session_state.cancel_extraction = False
        elif extracted_frames > 0:
            # Create ZIP file
            status_text.text("Creating ZIP archive...")
            zip_path = os.path.join(temp_dir, "dataset.zip")
            
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file in os.listdir(temp_dir):
                    if file.endswith(output_format):
                        zipf.write(
                            os.path.join(temp_dir, file),
                            arcname=file
                        )
            
            progress_bar.progress(1.0)
            progress_placeholder.empty()
            status_placeholder.empty()
            
            result_placeholder.markdown(f"""
            <div class="success-box">
                <h3>✅ Dataset Generated Successfully!</h3>
                <p>Extracted <strong>{extracted_frames}</strong> frames in <strong>{output_format.upper()}</strong> format</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Download button
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Image Dataset (ZIP)",
                    data=f,
                    file_name=f"dataset_{extracted_frames}_frames.zip",
                    mime="application/zip"
                )
            
            # Preview section with first 5 frames
            with preview_placeholder.expander("👀 Preview First 5 Frames", expanded=True):
                if len(preview_frames) > 0:
                    cols = st.columns(min(5, len(preview_frames)))
                    for idx, frame_path in enumerate(preview_frames):
                        with cols[idx]:
                            st.image(frame_path, caption=f"Frame {idx+1}", use_container_width=True)
                else:
                    st.info("No frames available for preview")

else:
    # Instructions when no video is uploaded
    st.markdown("""
    <div class="info-card">
        <h3>🎯 How to Use</h3>
        <ol>
            <li>Upload a video file (MP4, AVI, MOV, or MKV)</li>
            <li>Review the video properties</li>
            <li>Adjust extraction settings (FPS and format)</li>
            <li>Click "Generate Dataset" to extract frames</li>
            <li>Download your image dataset as a ZIP file</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Use Cases")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 🤖 Machine learning datasets
        - 🎨 Video analysis
        - 📸 Frame extraction
        """)
    
    with col2:
        st.markdown("""
        - 🔍 Computer vision training
        - 📊 Data augmentation
        - 🎬 Content analysis
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Developed by <strong>Sudarshan TS</strong></p>
    <p>
        <a href="https://github.com/darshacharya/Vid2Dataset" target="_blank" style="text-decoration: none; color: #667eea;">
            ⭐ GitHub Repository
        </a>
    </p>
</div>
""", unsafe_allow_html=True)