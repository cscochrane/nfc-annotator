import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from scipy.signal import spectrogram

def compute_zoomed_spectrogram(audio, sample_rate, fft_size, time_range, freq_range):
    freqs, times, Sxx = spectrogram(
        audio, fs=sample_rate, nperseg=fft_size, noverlap=int(0.875 * fft_size)
    )
    Sxx = 10 * np.log10(Sxx + 1e-10)
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    time_mask = (times >= time_range[0]) & (times <= time_range[1])

    Sxx_zoom = Sxx[freq_mask][:, time_mask]
    freqs_zoom = freqs[freq_mask]
    times_zoom = times[time_mask]
    extent = [times_zoom[0], times_zoom[-1], freqs_zoom[0], freqs_zoom[-1]]

    # Normalize and invert for black-on-white spectrogram
    Sxx_norm = (Sxx_zoom - np.min(Sxx_zoom)) / (np.max(Sxx_zoom) - np.min(Sxx_zoom))
    Sxx_inverted = 1.0 - Sxx_norm

    return Sxx_inverted, extent

def render_spectrogram_image(Sxx_data, extent):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.imshow(Sxx_data, aspect='auto', extent=extent, origin='lower', cmap='gray', interpolation='bilinear')
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf)
    return img, img.size
