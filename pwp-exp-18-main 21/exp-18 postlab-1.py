import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import convolve

import numpy as np
from scipy.io.wavfile import write

# Create a simple impulse response (like a small echo)
fs = 44100  # Sampling rate
impulse = np.zeros(fs)     # 1 second of samples
impulse[0] = 1.0           # Dirac impulse
impulse[5000] = 0.5        # small delayed echo
write(r"C:\Users\user\Downloads\impulse.wav", fs, impulse.astype(np.float32))
print("Impulse file created successfully!")

# === Read input audio and impulse response ===
fs1, x = wavfile.read(r'C:/Users/user/Downloads/BAK.wav')
fs2, h = wavfile.read(r'C:/Users/user/Downloads/impulse.wav')  # <-- put your impulse file path here too

# Convert to mono if stereo
if x.ndim > 1:
    x = x[:, 0]
if h.ndim > 1:
    h = h[:, 0]

# Make sure sampling rates match
assert fs1 == fs2, "Sampling rates must match!"

# === Linear convolution ===
y_linear = convolve(x, h)

# === Circular convolution using FFT ===
N = max(len(x), len(h))
Y_circular = np.fft.ifft(np.fft.fft(x, N) * np.fft.fft(h, N)).real

# === Normalize results ===
y_linear /= np.max(np.abs(y_linear))
Y_circular /= np.max(np.abs(Y_circular))

# === Plot results ===
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(x[:2000])
plt.title("Original Audio Signal")

plt.subplot(3, 1, 2)
plt.plot(y_linear[:2000])
plt.title("Linear Convolution Output")

plt.subplot(3, 1, 3)
plt.plot(Y_circular[:2000])
plt.title("Circular Convolution Output")

plt.tight_layout()
plt.show()
