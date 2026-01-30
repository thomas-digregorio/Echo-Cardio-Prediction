"""
EchoNet EF Prediction Demo
Streamlit app to demonstrate VideoMAE-based ejection fraction prediction.
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import os
import sys
import io
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.basic_model import EchoNetRegressor

# ============================================================================
# Configuration
# ============================================================================

# ImageNet normalization values
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# VideoMAE input requirements
NUM_FRAMES = 16  # VideoMAE-base pretrained with 16 frames
FRAME_SIZE = 224

# Paths (relative to project root)
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DATA_ROOT = PROJECT_ROOT / "EchoNet-Dynamic"
CSV_PATH = DATA_ROOT / "FileList.csv"
VIDEOS_DIR = DATA_ROOT / "Videos"


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_resource
def load_model() -> EchoNetRegressor:
    """
    Load the trained EchoNet regressor model.
    Cached to avoid reloading on every interaction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EchoNetRegressor(freeze_backbone=True)  # Architecture only
    
    if CHECKPOINT_PATH.exists():
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        
        # Handle checkpoint from old architecture (before Dropout was added)
        # Old: head = [LayerNorm, Linear] -> keys: head.0.*, head.1.*
        # New: head = [LayerNorm, Dropout, Linear] -> keys: head.0.*, head.2.*
        if "head.1.weight" in state_dict and "head.2.weight" not in state_dict:
            print("Remapping old checkpoint keys to new architecture...")
            state_dict["head.2.weight"] = state_dict.pop("head.1.weight")
            state_dict["head.2.bias"] = state_dict.pop("head.1.bias")
        
        model.load_state_dict(state_dict)
        st.sidebar.success(f"✅ Model loaded: `{CHECKPOINT_PATH.name}`")
    else:
        st.sidebar.warning(f"⚠️ No checkpoint found at {CHECKPOINT_PATH}. Using random weights.")
    
    model.to(device)
    model.eval()
    return model


@st.cache_data
def load_sample_videos(n_samples: int = 10) -> pd.DataFrame:
    """
    Load metadata for sample videos from the dataset.
    Selects videos with diverse EF values for interesting demo.
    """
    if not CSV_PATH.exists():
        st.error(f"FileList.csv not found at {CSV_PATH}")
        return pd.DataFrame()
    
    df = pd.read_csv(CSV_PATH)
    
    # Filter to only existing videos
    df['FullPath'] = df['FileName'].apply(
        lambda x: VIDEOS_DIR / (x if x.endswith('.avi') else f"{x}.avi")
    )
    df = df[df['FullPath'].apply(lambda p: p.exists())]
    
    # Filter to realistic EF range (exclude extreme outliers)
    df = df[(df['EF'] >= 20) & (df['EF'] <= 80)]
    
    # Select diverse EF range: sort by EF and sample evenly
    df_sorted = df.sort_values('EF').reset_index(drop=True)
    indices = np.linspace(0, len(df_sorted) - 1, n_samples, dtype=int)
    samples = df_sorted.iloc[indices].copy()
    
    return samples[['FileName', 'EF', 'FullPath']]


def extract_frames(video_path: Path, num_frames: int = NUM_FRAMES) -> np.ndarray | None:
    """
    Extract evenly-spaced frames from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        
    Returns:
        Array of shape (num_frames, H, W, C) in RGB format, or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        st.error(f"Could not open video: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        st.warning(f"Video has only {total_frames} frames, padding will be used.")
    
    # Calculate frame indices to sample
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to model input size
            frame_resized = cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE))
            frames.append(frame_resized)
        else:
            # Pad with last frame if read fails
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))
    
    cap.release()
    return np.array(frames)


def preprocess_frames(frames: np.ndarray) -> torch.Tensor:
    """
    Preprocess frames for VideoMAE input.
    
    Args:
        frames: Array of shape (F, H, W, C) with uint8 values [0, 255]
        
    Returns:
        Tensor of shape (1, F, C, H, W) normalized for ImageNet
    """
    # Convert to float [0, 1]
    frames_float = frames.astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    frames_norm = (frames_float - IMAGENET_MEAN) / IMAGENET_STD
    
    # Rearrange: (F, H, W, C) -> (F, C, H, W)
    frames_chw = np.transpose(frames_norm, (0, 3, 1, 2))
    
    # Add batch dimension: (1, F, C, H, W)
    tensor = torch.from_numpy(frames_chw).unsqueeze(0).float()
    
    return tensor


def predict_ef(model: EchoNetRegressor, video_path: Path, use_confidence: bool = True) -> dict:
    """
    Run EF prediction with confidence estimation.
    
    Args:
        model: Loaded EchoNetRegressor model
        video_path: Path to the video file
        use_confidence: If True, use MC Dropout for confidence estimation
        
    Returns:
        Dict with predicted_ef, confidence, attention_weights, frames
    """
    device = next(model.parameters()).device
    
    frames = extract_frames(video_path)
    if frames is None:
        return {"error": True}
    
    tensor = preprocess_frames(frames).to(device)
    
    if use_confidence:
        # MC Dropout for confidence
        mean_pred, confidence = model.predict_with_confidence(tensor, n_samples=10)
        predicted_ef = mean_pred.item()
        conf_value = confidence.item()
    else:
        with torch.no_grad():
            prediction, attn_weights = model(tensor, return_attention=True)
        predicted_ef = prediction.item()
        conf_value = None
    
    # Get attention weights for visualization
    attn_weights = model.get_attention_weights()
    
    predicted_ef = max(0, min(100, predicted_ef))
    
    return {
        "predicted_ef": predicted_ef,
        "confidence": conf_value,
        "attention_weights": attn_weights.cpu().numpy() if attn_weights is not None else None,
        "frames": frames,
        "error": False
    }


def create_gif_from_frames(frames: np.ndarray, fps: int = 8) -> bytes:
    """
    Create an animated GIF from frames for browser display.
    
    Args:
        frames: Array of shape (F, H, W, C) with uint8 RGB values
        fps: Frames per second for the GIF
        
    Returns:
        GIF bytes that can be displayed with st.image
    """
    import imageio
    
    # Create GIF in memory
    gif_buffer = io.BytesIO()
    duration = 1.0 / fps
    
    # Resize frames to smaller size for faster GIF
    resized_frames = [cv2.resize(f, (300, 300)) for f in frames]
    
    imageio.mimsave(gif_buffer, resized_frames, format='GIF', duration=duration, loop=0)
    gif_buffer.seek(0)
    
    return gif_buffer.getvalue()


def create_video_preview(frames: np.ndarray, fps: int = 8) -> None:
    """
    Display animated GIF preview of video frames in Streamlit.
    """
    if len(frames) == 0:
        st.warning("No frames to display")
        return
    
    try:
        gif_bytes = create_gif_from_frames(frames, fps)
        st.image(gif_bytes, caption="Echo Video (animated)", use_container_width=True)
    except ImportError:
        # Fallback if imageio not installed
        st.image(frames[0], caption="First frame (install imageio for animation)", use_container_width=True)


# ============================================================================
# Streamlit App
# ============================================================================

def main():
    st.set_page_config(
        page_title="EchoNet EF Prediction",
        page_icon="❤️",
        layout="wide"
    )
    
    st.title("❤️ EchoNet Ejection Fraction Prediction")
    st.markdown("""
    This demo uses a **VideoMAE** model fine-tuned on the EchoNet-Dynamic dataset 
    to predict left ventricular ejection fraction (EF) from echocardiogram videos.
    """)
    
    # Load model
    model = load_model()
    device = next(model.parameters()).device
    st.sidebar.markdown(f"**Device:** `{device}`")
    
    # Load sample videos
    samples = load_sample_videos(n_samples=10)
    
    if samples.empty:
        st.error("No sample videos found. Please check the data directory.")
        return
    
    # ========================================================================
    # Sidebar: Video Selection
    # ========================================================================
    
    st.sidebar.header("📹 Select Sample Video")
    st.sidebar.markdown("Choose from 10 samples with diverse EF values:")
    
    # Create selection options with EF preview
    video_options = {
        f"{row['FileName']} (EF: {row['EF']:.1f}%)": idx 
        for idx, row in samples.iterrows()
    }
    
    selected_option = st.sidebar.radio(
        "Sample Videos",
        options=list(video_options.keys()),
        index=0
    )
    
    selected_idx = video_options[selected_option]
    selected_row = samples.loc[selected_idx]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    - **Model:** VideoMAE-base
    - **Dataset:** EchoNet-Dynamic
    - **Task:** EF Regression
    - **Input:** 16 frames @ 224×224
    """)
    
    # ========================================================================
    # Main Content: Prediction Display
    # ========================================================================
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎬 Selected Video")
        st.markdown(f"**File:** `{selected_row['FileName']}`")
        
        video_path = selected_row['FullPath']
        
        # Extract frames and display as animated GIF (browser can't play .avi)
        with st.spinner("Loading video frames..."):
            display_frames = extract_frames(video_path, num_frames=32)  # More frames for smoother preview
        
        if display_frames is not None:
            create_video_preview(display_frames, fps=10)
    
    with col2:
        st.header("📊 Prediction Results")
        
        # Run prediction button
        if st.button("🔮 Predict EF", type="primary", use_container_width=True):
            with st.spinner("Running inference with confidence estimation..."):
                result = predict_ef(model, video_path, use_confidence=True)
            
            if not result.get("error", True):
                predicted_ef = result["predicted_ef"]
                confidence = result["confidence"]
                frames = result["frames"]
                true_ef = selected_row['EF']
                error = abs(predicted_ef - true_ef)
                
                # Display results with metrics
                st.markdown("### Results")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(label="Predicted EF", value=f"{predicted_ef:.1f}%")
                
                with metric_col2:
                    st.metric(label="True EF", value=f"{true_ef:.1f}%")
                
                with metric_col3:
                    st.metric(
                        label="Error",
                        value=f"{error:.1f}%",
                        delta=f"{-error:.1f}%" if error < 5 else f"+{error:.1f}%",
                        delta_color="normal" if error < 5 else "inverse"
                    )
                
                with metric_col4:
                    # Confidence display
                    if confidence is not None:
                        conf_pct = min(confidence * 10, 100)  # Scale for display
                        conf_label = "High" if conf_pct > 70 else "Medium" if conf_pct > 40 else "Low"
                        st.metric(label="Confidence", value=f"{conf_label}")
                
                # Visual comparison bar
                st.markdown("### EF Comparison")
                st.progress(int(predicted_ef), text=f"Predicted: {predicted_ef:.1f}%")
                st.progress(int(true_ef), text=f"Ground Truth: {true_ef:.1f}%")
                
                # Attention visualization (simplified)
                attn_weights = result.get("attention_weights")
                if attn_weights is not None:
                    st.markdown("### 🔍 Model Attention")
                    st.caption("Higher values indicate which parts of the video the model focuses on for EF prediction.")
                    # Show attention as a bar chart across tokens
                    attn_flat = attn_weights.flatten()
                    st.bar_chart(attn_flat[:50], height=100)  # Show first 50 tokens
                
                # Clinical Interpretation
                st.markdown("### 🩺 Clinical Interpretation")
                
                if predicted_ef >= 55:
                    st.success("**Normal EF** (≥55%): Left ventricular function appears normal.")
                elif predicted_ef >= 40:
                    st.warning("**Mildly Reduced EF** (40-54%): Mild left ventricular dysfunction.")
                elif predicted_ef >= 30:
                    st.warning("**Moderately Reduced EF** (30-39%): Moderate LV dysfunction.")
                else:
                    st.error("**Severely Reduced EF** (<30%): Severe LV dysfunction. Clinical evaluation recommended.")
                    
            else:
                st.error("Prediction failed. Check video file.")
        else:
            st.info("👆 Click 'Predict EF' to run inference on the selected video.")
            st.markdown(f"**Ground Truth EF:** {selected_row['EF']:.1f}%")


if __name__ == "__main__":
    main()
