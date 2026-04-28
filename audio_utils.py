import librosa
import matplotlib.pyplot as plt
import scipy.signal as scipysig
import numpy as np

# Load audio
def load_data(uploaded_file):
    # Read in file
    (s, framerate) = librosa.core.load(uploaded_file, sr=None, mono=False)

    # Normalize
    s = s/np.max(s)**2

    return s, framerate

# Create spectrogram parameters
def spectrogram(vid, s, framerate, start=None, end=None):
    # Convert times to samples
    if end is not None:
        end = int(np.ceil(end * framerate))
    if start is not None:
        start = int(np.trunc(start * framerate))
        s = s[start:end]

    s = np.asarray(s).squeeze()
    
    if s.ndim == 2:
        s = s.mean(axis=0)

    # STFT
    freqs, times, spec = scipysig.stft(s, fs=framerate, nperseg=1024)
    spec = np.abs(spec)

    # Power / intensity
    power = np.abs(spec) ** 2

    # Log spectrogram (more stable visually + for ML)
    log_spec = np.log(power + 1e-8)

    return freqs, times, power, log_spec

# Visualize spectrogram
def plot_spectrogram(freqs, times, spec):
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, freqs, spec, shading='gouraud')
    plt.ylim(0, 14000)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Log Spectrogram')
    plt.colorbar(label='Log Power')
    plt.tight_layout()
    plt.show()