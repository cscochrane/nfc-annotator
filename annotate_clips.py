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

from utils.supabase_utils import (
    get_supabase_client,
    upload_to_supabase,
    list_wav_files,
    download_wav_file,
    get_annotated_filenames,
    insert_annotation
)

from utils.spectrogram import compute_zoomed_spectrogram, render_spectrogram_image


# === SUPABASE CONFIG ===
supabase = get_supabase_client()

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

    # === User login ===
    if "user" not in st.session_state:
        st.session_state.user = ""

    st.session_state.user = st.text_input("Your name or email (for tracking):", value=st.session_state.user)

    if not st.session_state.user.strip():
        st.warning("Please enter your name or email to upload.")
        st.stop()

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
        response = supabase.storage.from_(bucket_name).upload(
            f"{filename}", file_data, {"content-type": "audio/wav", "x-upsert": "true"}
        )

        if "error" not in response:
            st.success(f"Uploaded {filename} to cloud storage!")

            # Log uploader info to Supabase table
            upload_record = {
                "filename": filename,
                "uploader": st.session_state.get("user", "anonymous")
            }
            response = supabase.table("uploads").insert(upload_record).execute()

            if response.get("status_code", 200) >= 400:
                st.warning(f"Upload succeeded but failed to log uploader info: {response}")
        else:
            st.error(f"Upload failed: {response['error']['message']}")

        os.remove(temp_path)

    st.button("⬅️ Back to Home", on_click=lambda: go_to("home"))


elif st.session_state.page == "annotate":
    st.title("🖍️ Annotate Public Recordings")

    # === User login ===
    if "user" not in st.session_state:
        st.session_state.user = ""

    st.session_state.user = st.text_input("Your name or email (for tracking):", value=st.session_state.user)

    if not st.session_state.user.strip():
        st.warning("Please enter your name or email to begin annotating.")
        st.stop()

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
        "PYNU_LBDO": "Pygmy Nuthatch/Long-billed Dowitcher",
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
    # Get all .wav files in the Supabase bucket
    all_remote_files = list_wav_files(supabase)

    # Get already-annotated filenames
    user_annotated_files = set(
        row["filename"]
        for row in supabase.table("annotations")
                           .select("filename")
                           .eq("annotator", st.session_state.user.strip())
                           .execute()
                           .data
    )

    # Filter only files not yet annotated
    unlabeled_files = [f for f in all_remote_files if f not in user_annotated_files]

    # Stop if everything is labeled
    if not unlabeled_files:
        st.success("🎉 All clips labeled!")
        st.button("⬅️ Back to Home", on_click=lambda: go_to("home"))
        st.stop()

    # Pick the first unlabeled file
    current_file = unlabeled_files[0]
    file_name = os.path.basename(current_file)

    # Download file locally for playback and processing
    local_wav_path = download_wav_file(supabase, current_file)
    if not local_wav_path:
        st.error(f"❌ Failed to download {current_file}")
        st.stop()

    st.subheader(f"Annotating: {file_name}")
    st.audio(local_wav_path)  # ✅ Audio playback
  
    if not unlabeled_files:
        st.success("🎉 All clips labeled!")
        st.button("⬅️ Back to Home", on_click=lambda: go_to("home"))  # ✅ Add this here
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
    # Sliders (keep these lines as-is)
    zoom_time_range = st.slider("Zoom Time (s)", 0.0, float(len(audio) / sample_rate), (0.0, float(len(audio) / sample_rate)), step=0.01)
    zoom_freq_range = st.slider("Zoom Frequency (Hz)", 0, 11000, (0, 11000), step=100)

    # Use utility functions
    Sxx_inverted, extent = compute_zoomed_spectrogram(
        audio, sample_rate, fft_size, zoom_time_range, zoom_freq_range
    )

    img, (img_width, img_height) = render_spectrogram_image(Sxx_inverted, extent)


        
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

                # Create and send annotation
                annotation = {
                    "filename": file_name,
                    "label": final_label,
                    "start_time": float(min(start_time, end_time)),
                    "end_time": float(max(start_time, end_time)),
                    "low_freq": float(min(low_freq, high_freq)),
                    "high_freq": float(max(low_freq, high_freq)),
                    "annotator": st.session_state.get("user", "anonymous")
                }

                response = insert_annotation(supabase, annotation)
                if response.get("status_code", 200) >= 400:
                    st.error(f"Failed to insert annotation: {response}")

            st.success(f"Saved {len(canvas_result.json_data['objects'])} annotations for {file_name}")
            st.experimental_rerun()
        else:
            st.warning("No boxes drawn.")
    
    st.button("⬅️ Back to Home", on_click=lambda: go_to("home"))
