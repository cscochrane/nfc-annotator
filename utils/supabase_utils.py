from supabase import create_client, Client
import os

def get_supabase_client() -> Client:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://vyndvwdwqyrnzxjdcakf.supabase.co")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ5bmR2d2R3cXlybnp4amRjYWtmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDY2NTc0NTksImV4cCI6MjA2MjIzMzQ1OX0._HhABMUYPP_86KJxq5Tnj6-KxU1XK06nUpkV10avMyg")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_to_supabase(client: Client, bucket_name: str, filename: str, file_data: bytes):
    return client.storage.from_(bucket_name).upload(
        path=filename,
        file=file_data,
        file_options={"content-type": "audio/wav"}
    )
def insert_annotation(client: Client, annotation: dict):
    response = client.table("annotations").insert(annotation).execute()
    return response

import tempfile

def list_wav_files(client: Client, bucket_name: str = "nfc-uploads"):
    response = client.storage.from_(bucket_name).list()
    if isinstance(response, list):
        return [f["name"] for f in response if f["name"].endswith(".wav")]
    return []

def download_wav_file(client: Client, filename: str, bucket_name: str = "nfc-uploads"):
    data = client.storage.from_(bucket_name).download(filename)
    if isinstance(data, bytes):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(data)
        temp_file.close()
        return temp_file.name
    return None

def get_annotated_filenames(client: Client):
    response = client.table("annotations").select("filename").execute()
    if response.data:
        return list(set([row["filename"] for row in response.data]))
    return []
import os
import json
import torch
import torchaudio
from tqdm import tqdm
from utils.supabase_utils import get_annotations, download_wav_file

OUTPUT_DIR = "processed_data"
SAMPLE_RATE = 16000
N_MELS = 64

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=N_MELS
)

def process_clip(filepath):
    waveform, sr = torchaudio.load(filepath)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
    mel_spec = mel_transform(waveform)
    return mel_spec.squeeze(0)  # Assume mono

def main():
    annotations = get_annotations()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    label_to_idx = {}
    idx = 0
    metadata = []

    for entry in tqdm(annotations):
        filename = entry["filename"]
        label = entry["label"]
        if not label:
            continue

        if label not in label_to_idx:
            label_to_idx[label] = idx
            idx += 1

        # Download clip if needed
        wav_path = os.path.join("clips", filename)
        if not os.path.exists(wav_path):
            os.makedirs("clips", exist_ok=True)
            download_wav_file(filename, wav_path)

        # Create Mel spectrogram
        try:
            mel_spec = process_clip(wav_path)
            tensor_path = os.path.join(OUTPUT_DIR, filename.replace(".wav", ".pt"))
            torch.save(mel_spec, tensor_path)
            metadata.append({"tensor_path": tensor_path, "label": label_to_idx[label]})
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    # Save metadata
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_to_idx, f)

if __name__ == "__main__":
    main()

