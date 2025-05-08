import os
import json
import torch
import torchaudio
from tqdm import tqdm
from supabase_utils import get_labeled_clips, download_wav

OUTPUT_DIR = "processed_data"
SAMPLE_RATE = 16000
N_MELS = 64

# Set up MelSpectrogram
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=N_MELS
)

def process_clip(filepath):
    waveform, sr = torchaudio.load(filepath)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    mel_spec = mel_transform(waveform)
    return mel_spec.squeeze(0)  # remove channel dim if mono

def main():
    data = get_labeled_clips()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    label_to_idx = {}
    idx = 0
    metadata = []

    for entry in tqdm(data):
        filename, label = entry["filename"], entry["label"]
        if not label:
            continue

        if label not in label_to_idx:
            label_to_idx[label] = idx
            idx += 1

        wav_path = download_wav(filename)
        mel_spec = process_clip(wav_path)

        save_path = os.path.join(OUTPUT_DIR, f"{filename.replace('.wav', '.pt')}")
        torch.save(mel_spec, save_path)

        metadata.append({"filename": save_path, "label": label_to_idx[label]})

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_to_idx, f)

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    main()
