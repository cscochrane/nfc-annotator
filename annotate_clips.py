import os
import glob
import tempfile
import streamlit as st
from supabase import create_client, Client
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import wavfile
from scipy.signal import spectrogram
from streamlit_drawable_canvas import st_canvas

# === SUPABASE CONFIG ===
SUPABASE_URL = "https://vyndvwdwqyrnzxjdcakf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ5bmR2d2R3cXlybnp4amRjYWtmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDY2NTc0NTksImV4cCI6MjA2MjIzMzQ1OX0._HhABMUYPP_86KJxq5Tnj6-KxU1XK06nUpkV10avMyg"


@st.cache_resource
def load_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = load_supabase()

# --- Page navigation setup ---
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page

# --- Page router ---
if st.session_state.page == "home":
    st.title("🎧 Nocturnal Flight Call Annotation")

    st.markdown("""
    Welcome! Choose one of the options below to contribute to our open dataset:
    """)
    st.button("📤 Upload Your Own Recording", on_click=lambda: go_to("upload"))
    st.button("🖍️ Annotate Public Recordings", on_click=lambda: go_to("annotate"))


# === Upload UI (inserted before annotation UI) ===
elif st.session_state.page == "upload":
    st.title("📤 Upload Your NFC Recording")

    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
    if uploaded_file is not None:
        filename = uploaded_file.name

        # Save to temp file first
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        with open(temp_path, "rb") as f:
            file_data = f.read()

        # Upload to Supabase bucket
        bucket_name = "nfc-uploads"
        response = supabase.storage.from_(bucket_name).upload(f"{filename}", file_data, {"content-type": "audio/wav"})

        if "error" not in response:
            st.success(f"Uploaded {filename} to cloud storage!")
        else:
            st.error(f"Upload failed: {response['error']['message']}")

        os.remove(temp_path)

    st.button("⬅️ Back to Home", on_click=lambda: go_to("home"))  # ✅ OUTSIDE the 'if'

elif st.session_state.page == "annotate":
    st.title("🖍️ Annotate Public Recordings")

    from streamlit.experimental_rerun import rerun  # or use st.experimental_rerun()

    # === Your original annotation setup ===
    DATA_DIR = os.path.expanduser('~/Documents/nfc-detector/data/wav')
    LABELS_FILE = os.path.expanduser('~/Documents/nfc-detector/data/bbox_labels.csv')

    SPECIES_MAP = {
        "AMCO": "American Coot",
        "AMPI": "American Pipit",
        "AMRE": "American Redstart",
        "AMRO": "American Robin",
        "ATSP": "American Tree Sparrow",
        "BAIS": "Baird's Sparrow",
        "BHGR": "Black-headed Grosbeak",
        "CAWA": "Canada Warbler",
        "CCSP_BRSP": "Clay-colored/Brewer's Sparrow",
        "CHSP": "Chipping Sparrow",
        "COYE": "Common Yellowthroat",
        "DBUP": "Double-banded Upland Plover",
        "DEJU": "Dark-eyed Junco",
        "GCKI": "Golden-crowned Kinglet",
        "GRSP": "Grasshopper Sparrow",
        "GRYE": "Greater Yellowlegs",
        "HETH": "Hermit Thrush",
        "High": "High frequency NFC",
        "HOLA": "Hooded Warbler",
        "LALO": "Lapland Longspur",
        "LAZB": "Lazuli Bunting",
        "LBCU": "Long-billed Curlew",
        "LISP": "Lincoln's Sparrow",
        "Low": "Low frequency NFC",
        "MGWA": "MacGillivray's Warbler",
        "NOWA": "Northern Waterthrush",
        "OVEN": "Ovenbird",
        "Other": "Other",
        "PYNU_LBDO": "Pinyon/Long-billed Dowitcher",
        "Peep": "Unidentified Peep",
        "SAVS": "Savannah Sparrow",
        "SORA": "Sora",
        "SOSP": "Song Sparrow",
        "SPSA_SOSA": "Saltmarsh/Seaside Sparrow",
        "SWTH": "Swainson's Thrush",
        "UPSA": "Upland Sandpiper",
        "Unknown": "Unknown",
        "VEER": "Veery",
        "VESP": "Vesper Sparrow",
        "VIRA": "Virginia Rail",
        "WCSP": "White-crowned Sparrow",
        "WEME": "Western Meadowlark",
        "WETA": "Western Tanager",
        "WIWA": "Wilson's Warbler",
        "WTSP": "White-throated Sparrow",
        "Weak": "Weak NFC",
        "YRWA": "Yellow-rumped Warbler",
        "Zeep": "Unidentified Zeep",
        "Noise": "Noise"
    }
    
    LABEL_OPTIONS = sorted(SPECIES_MAP.keys())
    
    # --- Load files ---
    wav_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.wav')))
    if os.path.exists(LABELS_FILE):
        labels_df = pd.read_csv(LABELS_FILE)
    else:
        labels_df = pd.DataFrame(columns=['file', 'label', 'start_time', 'end_time', 'low_freq', 'high_freq'])
    
    # --- UI ---
    st.title("NFC Annotator with Bounding Boxes")
    
    unlabeled_files = [f for f in wav_files if os.path.basename(f) not in labels_df['file'].unique()]
    if not unlabeled_files:
        st.success("All clips labeled!")
        st.stop()
    
    current_file = unlabeled_files[0]
    file_name = os.path.basename(current_file)
    
    st.subheader(f"Annotating: {file_name}")
    st.audio(current_file)  # ✅ Audio playback
    
    # --- FFT Window Size ---
    fft_size = st.selectbox("FFT Window Size", [256, 512, 1024, 2048], index=2)
    
    # --- Load Audio ---
    sample_rate, audio = wavfile.read(current_file)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio / np.max(np.abs(audio)) * 6.0  # normalize + amplify
    
    # --- Compute Spectrogram ---
    freqs, times, Sxx = spectrogram(audio, fs=sample_rate, nperseg=fft_size, noverlap=int(0.75 * fft_size))
    Sxx = 10 * np.log10(Sxx + 1e-10)
    freq_mask = freqs <= 11000
    Sxx = Sxx[freq_mask, :]
    freqs = freqs[freq_mask]
    
    # --- Zoom Sliders ---
    st.markdown("### Zoom Controls")
    zoom_time_range = st.slider("Zoom Time (s)", 0.0, float(times[-1]), (0.0, float(times[-1])), step=0.01)
    zoom_freq_range = st.slider("Zoom Frequency (Hz)", 0, 11000, (0, 11000), step=100)
    
    time_mask = (times >= zoom_time_range[0]) & (times <= zoom_time_range[1])
    freq_mask = (freqs >= zoom_freq_range[0]) & (freqs <= zoom_freq_range[1])
    
    Sxx_zoom = Sxx[freq_mask, :][:, time_mask]
    freqs_zoom = freqs[freq_mask]
    times_zoom = times[time_mask]
    extent = [times_zoom[0], times_zoom[-1], freqs_zoom[0], freqs_zoom[-1]]
    
    # Normalize and invert for Audacity-style (black = loud)
    Sxx_norm = (Sxx_zoom - np.min(Sxx_zoom)) / (np.max(Sxx_zoom) - np.min(Sxx_zoom))
    Sxx_inverted = 1.0 - Sxx_norm
    
    # Create high-resolution spectrogram image
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)  # Bigger and sharper
    ax.imshow(Sxx_inverted, aspect='auto', extent=extent, origin='lower',
              cmap='gray', interpolation='bilinear')  # Smooth rendering
    ax.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img_width, img_height = img.size
    
    
    # --- Drawable Canvas ---
    st.markdown("### Draw bounding boxes and select label")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        background_image=img,
        update_streamlit=True,
        height=600,
        drawing_mode="rect",
        key=f"canvas_{file_name}",  # ✅ unique key for each file
    )
    
    # --- Species label ---
    search_term = st.text_input("Search for species (code or name):")
    filtered_options = [code for code in LABEL_OPTIONS if search_term.lower() in code.lower() or search_term.lower() in SPECIES_MAP[code].lower()]
    final_label = st.selectbox("Select species label", filtered_options if filtered_options else LABEL_OPTIONS)
    
    # --- Skip Clip ---
    if st.button("Skip Clip"):
        st.warning(f"Skipped {file_name}")
        st.experimental_rerun()
    
    # --- Save Annotation ---
    if st.button("Save Annotation"):
        if canvas_result.json_data and canvas_result.json_data["objects"]:
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] != "rect":
                    continue
                left = obj["left"]
                top = obj["top"]
                width = obj["width"]
                height = obj["height"]
                x0 = left
                x1 = left + width
                y0 = top
                y1 = top + height
    
                # Convert canvas pixel coords to spectrogram time/freq
                start_time = extent[0] + (x0 / img_width) * (extent[1] - extent[0])
                end_time = extent[0] + (x1 / img_width) * (extent[1] - extent[0])
                high_freq = extent[2] + ((img_height - y0) / img_height) * (extent[3] - extent[2])
                low_freq = extent[2] + ((img_height - y1) / img_height) * (extent[3] - extent[2])
    
                # Add annotation as a new row
                new_row = pd.DataFrame([{
                    'file': file_name,
                    'label': final_label,
                    'start_time': min(start_time, end_time),
                    'end_time': max(start_time, end_time),
                    'low_freq': min(low_freq, high_freq),
                    'high_freq': max(low_freq, high_freq)
                }])
                labels_df = pd.concat([labels_df, new_row], ignore_index=True)
    
            labels_df.to_csv(LABELS_FILE, index=False)
            st.success(f"Saved {len(canvas_result.json_data['objects'])} annotations for {file_name}")
            st.experimental_rerun()
        else:
            st.warning("No boxes drawn.")
     st.button("⬅️ Back to Home", on_click=lambda: go_to("home"))
