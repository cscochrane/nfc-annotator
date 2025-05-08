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
from datetime import datetime


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
# === Upload UI ===
elif st.session_state.page == "upload":
    st.title("📤 Upload Your NFC Recording")

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

        bucket_name = "nfc-uploads"
        try:
            # Upload to Supabase Storage
            response = supabase.storage.from_(bucket_name).upload(
                filename, file_data, {"content-type": "audio/wav", "x-upsert": "true"}
            )

            if not getattr(response, "error", None):
                st.success(f"✅ Uploaded {filename} to cloud storage!")

                # Insert upload metadata into the 'uploads' table
                upload_record = {
                    "filename": filename,
                    "uploader": st.session_state.user.strip(),
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": f"{bucket_name}/{filename}"
                }

                result = supabase.table("uploads").insert(upload_record).execute()

                if getattr(result, "status_code", 200) >= 400:
                    st.warning(f"Upload succeeded but failed to log uploader info: {result}")
            else:
                st.error(f"Upload failed: {response.error['message']}")

        except Exception as e:
            import traceback
            st.error("❌ Upload crashed:")
            st.code(traceback.format_exc())

        finally:
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
    
    # --- FFT Window Size ---
    fft_size = st.selectbox("FFT Window Size", [256, 512, 1024, 2048], index=2, key="fft_size_selector")
    
    # --- Load Audio ---
    sample_rate, audio = wavfile.read(local_wav_path)  # ✅
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio / np.max(np.abs(audio)) * 6.0  # normalize + amplify
        # --- Compute Spectrogram ---
    sample_rate, audio = wavfile.read(local_wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio / np.max(np.abs(audio)) * 6.0

    zoom_time_range = st.slider("Zoom Time (s)", 0.0, float(len(audio) / sample_rate), (0.0, float(len(audio) / sample_rate)), step=0.01)
    zoom_freq_range = st.slider("Zoom Frequency (Hz)", 0, 11000, (0, 11000), step=100)

    Sxx_inverted, extent = compute_zoomed_spectrogram(audio, sample_rate, fft_size, zoom_time_range, zoom_freq_range)
    img, _ = render_spectrogram_image(Sxx_inverted, extent)

    st.image(img, caption="Spectrogram", use_container_width=True)


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
        annotation = {
            "filename": file_name,
            "label": final_label,
            "annotator": st.session_state.get("user", "anonymous")
        }
    
        response = insert_annotation(supabase, annotation)
        if response.get("status_code", 200) >= 400:
            st.error(f"❌ Failed to insert annotation: {response}")
        else:
            st.success(f"✅ Saved annotation for {file_name}")
            st.experimental_rerun()
    

    st.button("⬅️ Back to Home", on_click=lambda: go_to("home"))
