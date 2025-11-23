import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# === Generate example audio signals ===
fs = 8000
t = np.linspace(0, 1, fs)

clean = np.sin(2*np.pi*440*t)                # clean tone (440 Hz)
noisy = clean + 0.4*np.random.randn(len(t))  # noisy version
periodic = np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*880*t)  # periodic signal

# === Autocorrelation ===
auto_clean = correlate(clean, clean)
auto_noisy = correlate(noisy, noisy)
auto_periodic = correlate(periodic, periodic)

# === Cross-correlation ===
cross_clean_noisy = correlate(clean, noisy)
cross_clean_periodic = correlate(clean, periodic)

# === Plot results ===
plt.figure(figsize=(10,8))
plt.subplot(3,1,1); plt.plot(auto_clean); plt.title("Autocorrelation - Clean")
plt.subplot(3,1,2); plt.plot(auto_noisy); plt.title("Autocorrelation - Noisy")
plt.subplot(3,1,3); plt.plot(auto_periodic); plt.title("Autocorrelation - Periodic")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,6))
plt.subplot(2,1,1); plt.plot(cross_clean_noisy); plt.title("Cross-Correlation - Clean vs Noisy")
plt.subplot(2,1,2); plt.plot(cross_clean_periodic); plt.title("Cross-Correlation - Clean vs Periodic")
plt.tight_layout(); plt.show()
