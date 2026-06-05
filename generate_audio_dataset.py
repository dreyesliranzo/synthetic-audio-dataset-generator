"""
Synthetic Speech + Noise Dataset Generator
-------------------------------------------
Generates labeled audio training samples for speech AI models.
Each sample = clean synthetic speech + background noise at a random SNR.

Mirrors the data pipelines used to train models like Whisper, DeepSpeech,
and RNNoise — where noisy/clean pairs teach the model to separate speech.

Usage:  python generate_audio_dataset.py
Output: ./output/
          audio/clean/      ← clean speech signals
          audio/noisy/      ← speech + noise mixed at target SNR
          audio/noise_only/ ← isolated noise samples
          labels.json       ← SNR, noise type, duration, RMS per sample
          summary.png       ← waveform + spectrogram grid
          stats.png         ← SNR distribution + noise type breakdown
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io.wavfile as wav
import scipy.signal as signal
import json
import os
from pathlib import Path
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
NUM_SAMPLES     = 30          # total audio samples to generate
SAMPLE_RATE     = 16000       # 16kHz — standard for speech AI
DURATION_RANGE  = (2.0, 5.0)  # seconds per clip
SNR_RANGE       = (-5, 30)    # dB — from very noisy to relatively clean
OUTPUT_DIR      = Path("output")
SUMMARY_SAMPLES = 6           # how many to show in the visual grid
# ─────────────────────────────────────────────────────────────────────────────

NOISE_TYPES = ["white", "brown", "crowd", "static", "hum"]


# ── Speech synthesis ──────────────────────────────────────────────────────────

def synthesize_speech(duration, sr=SAMPLE_RATE):
    """
    Generate a speech-like signal using formant synthesis.
    Real speech is built from harmonics modulated by formant frequencies
    (resonances in the vocal tract). We model this procedurally.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Fundamental pitch (F0): 85–255 Hz for human voice
    f0 = np.random.uniform(100, 220)

    # Generate harmonics (voice source)
    source = np.zeros_like(t)
    for k in range(1, 12):
        amp = 1.0 / k  # harmonics fall off with order
        source += amp * np.sin(2 * np.pi * f0 * k * t)

    # Formant filter — shape the spectrum like a vocal tract
    # F1 (500 Hz), F2 (1500 Hz), F3 (2500 Hz) — rough vowel formants
    formants = [
        (np.random.uniform(400, 800),   150),
        (np.random.uniform(1000, 2000), 200),
        (np.random.uniform(2200, 3000), 300),
    ]
    filtered = source.copy()
    for fc, bw in formants:
        b, a = signal.iirpeak(fc / (sr / 2), fc / bw)
        filtered = signal.lfilter(b, a, filtered)

    # Amplitude envelope: natural speech has pauses and syllable bursts
    envelope = _speech_envelope(duration, sr)
    speech = filtered * envelope

    # Normalize
    speech = speech / (np.max(np.abs(speech)) + 1e-9) * 0.8
    return speech.astype(np.float32)


def _speech_envelope(duration, sr):
    """Simulate syllable-level amplitude modulation (~4–8 syllables/sec)."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n)
    syll_rate = np.random.uniform(4, 8)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * syll_rate * t + np.random.uniform(0, 2*np.pi))
    # Add random pauses
    n_pauses = np.random.randint(1, 4)
    for _ in range(n_pauses):
        start = np.random.randint(0, n - sr // 4)
        length = np.random.randint(sr // 8, sr // 3)
        env[start:start+length] *= np.random.uniform(0.0, 0.15)
    return np.clip(env, 0, 1)


# ── Noise generators ──────────────────────────────────────────────────────────

def generate_noise(noise_type, n_samples, sr=SAMPLE_RATE):
    if noise_type == "white":
        return np.random.randn(n_samples).astype(np.float32)

    elif noise_type == "brown":
        # Brown noise = integrated white noise (1/f² spectrum)
        white = np.random.randn(n_samples)
        brown = np.cumsum(white)
        brown -= np.mean(brown)
        return (brown / (np.max(np.abs(brown)) + 1e-9)).astype(np.float32)

    elif noise_type == "crowd":
        # Crowd babble: many overlapping speech-like voices at low level
        babble = np.zeros(n_samples, dtype=np.float32)
        for _ in range(np.random.randint(6, 14)):
            duration = n_samples / sr
            voice = synthesize_speech(duration, sr)
            offset = np.random.randint(0, max(1, n_samples - len(voice)))
            end = min(offset + len(voice), n_samples)
            babble[offset:end] += voice[:end-offset] * np.random.uniform(0.1, 0.4)
        return babble / (np.max(np.abs(babble)) + 1e-9)

    elif noise_type == "static":
        # Static: bandlimited noise around mid frequencies
        white = np.random.randn(n_samples)
        b, a = signal.butter(4, [300/(sr/2), 3400/(sr/2)], btype='band')
        static = signal.lfilter(b, a, white)
        return (static / (np.max(np.abs(static)) + 1e-9)).astype(np.float32)

    elif noise_type == "hum":
        # Electrical hum: 60 Hz + harmonics (common in cheap audio equipment)
        t = np.linspace(0, n_samples/sr, n_samples)
        hum = np.zeros(n_samples)
        for harmonic in [1, 2, 3, 4]:
            hum += (1.0/harmonic) * np.sin(2*np.pi * 60 * harmonic * t)
        hum += np.random.randn(n_samples) * 0.15  # add slight noise floor
        return (hum / (np.max(np.abs(hum)) + 1e-9)).astype(np.float32)

    return np.random.randn(n_samples).astype(np.float32)


# ── SNR mixing ────────────────────────────────────────────────────────────────

def mix_at_snr(speech, noise, snr_db):
    """
    Mix speech + noise at a target SNR (dB).
    SNR = 10 * log10(P_speech / P_noise)
    We scale the noise so the mix hits the requested SNR.
    """
    # Match lengths
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(speech)]

    p_speech = np.mean(speech ** 2) + 1e-9
    p_noise  = np.mean(noise  ** 2) + 1e-9
    target_noise_power = p_speech / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / p_noise)
    noisy = speech + scale * noise
    # Normalize to prevent clipping
    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        noisy = noisy / peak * 0.99
    return noisy.astype(np.float32), noise * scale


def rms_db(x):
    return float(10 * np.log10(np.mean(x**2) + 1e-9))


# ── Save WAV ──────────────────────────────────────────────────────────────────

def save_wav(path, audio, sr=SAMPLE_RATE):
    pcm = (audio * 32767).astype(np.int16)
    wav.write(path, sr, pcm)


# ── Generate one sample ───────────────────────────────────────────────────────

def generate_sample(sample_id):
    duration   = np.random.uniform(*DURATION_RANGE)
    snr_db     = np.random.uniform(*SNR_RANGE)
    noise_type = np.random.choice(NOISE_TYPES)
    n_samples  = int(SAMPLE_RATE * duration)

    speech     = synthesize_speech(duration)
    noise      = generate_noise(noise_type, n_samples)
    noisy, scaled_noise = mix_at_snr(speech, noise, snr_db)

    # Save audio files
    clean_path = OUTPUT_DIR / "audio" / "clean"      / f"sample_{sample_id:04d}_clean.wav"
    noisy_path = OUTPUT_DIR / "audio" / "noisy"      / f"sample_{sample_id:04d}_noisy.wav"
    noise_path = OUTPUT_DIR / "audio" / "noise_only" / f"sample_{sample_id:04d}_noise.wav"

    save_wav(clean_path, speech)
    save_wav(noisy_path, noisy)
    save_wav(noise_path, scaled_noise)

    return {
        "sample_id": sample_id,
        "duration_sec": round(duration, 3),
        "sample_rate": SAMPLE_RATE,
        "snr_db": round(snr_db, 2),
        "noise_type": noise_type,
        "speech_rms_db": round(rms_db(speech), 2),
        "noisy_rms_db": round(rms_db(noisy), 2),
        "files": {
            "clean": str(clean_path),
            "noisy": str(noisy_path),
            "noise_only": str(noise_path),
        }
    }


# ── Summary plot ──────────────────────────────────────────────────────────────

def render_summary(annotations):
    samples = annotations[:SUMMARY_SAMPLES]
    n = len(samples)
    fig = plt.figure(figsize=(14, n * 2.6), facecolor='#0d0d12')
    fig.suptitle("Synthetic Speech + Noise Dataset — Sample Pairs",
                 fontsize=13, color='white', fontweight='bold', y=1.005)

    gs = gridspec.GridSpec(n, 4, figure=fig, hspace=0.55, wspace=0.35)

    for row, ann in enumerate(samples):
        # Load audio back for plotting
        _, clean = wav.read(ann["files"]["clean"])
        _, noisy = wav.read(ann["files"]["noisy"])
        clean = clean.astype(np.float32) / 32767
        noisy = noisy.astype(np.float32) / 32767
        t = np.linspace(0, ann["duration_sec"], len(clean))

        snr   = ann["snr_db"]
        ntype = ann["noise_type"]
        snr_color = _snr_color(snr)

        # Waveform — clean
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(t, clean, color='#4a9eff', linewidth=0.6, alpha=0.9)
        ax1.set_facecolor('#16161c')
        ax1.set_title(f"Clean speech", fontsize=7.5, color='#aaaaaa', pad=2)
        ax1.set_yticks([]); ax1.set_xticks([])
        for sp in ax1.spines.values(): sp.set_color('#2a2a35')

        # Waveform — noisy
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.plot(t, noisy, color=snr_color, linewidth=0.6, alpha=0.9)
        ax2.set_facecolor('#16161c')
        ax2.set_title(f"Noisy ({ntype})  SNR={snr:+.1f} dB",
                      fontsize=7.5, color='#aaaaaa', pad=2)
        ax2.set_yticks([]); ax2.set_xticks([])
        for sp in ax2.spines.values(): sp.set_color('#2a2a35')

        # Spectrogram — clean
        ax3 = fig.add_subplot(gs[row, 2])
        f, t_spec, Sxx = signal.spectrogram(clean, SAMPLE_RATE, nperseg=256)
        ax3.pcolormesh(t_spec, f/1000, 10*np.log10(Sxx + 1e-9),
                       cmap='magma', shading='gouraud', vmin=-80, vmax=0)
        ax3.set_facecolor('#16161c')
        ax3.set_title("Spectrogram (clean)", fontsize=7.5, color='#aaaaaa', pad=2)
        ax3.set_ylabel("kHz", fontsize=6, color='#666'); ax3.set_yticks([0,2,4,6,8])
        ax3.tick_params(colors='#555', labelsize=5)
        for sp in ax3.spines.values(): sp.set_color('#2a2a35')

        # Spectrogram — noisy
        ax4 = fig.add_subplot(gs[row, 3])
        f, t_spec, Sxx = signal.spectrogram(noisy, SAMPLE_RATE, nperseg=256)
        ax4.pcolormesh(t_spec, f/1000, 10*np.log10(Sxx + 1e-9),
                       cmap='magma', shading='gouraud', vmin=-80, vmax=0)
        ax4.set_facecolor('#16161c')
        ax4.set_title("Spectrogram (noisy)", fontsize=7.5, color='#aaaaaa', pad=2)
        ax4.set_yticks([0,2,4,6,8])
        ax4.tick_params(colors='#555', labelsize=5)
        for sp in ax4.spines.values(): sp.set_color('#2a2a35')

        # Row label
        fig.text(0.005, 1 - (row + 0.5)/n,
                 f"#{ann['sample_id']:04d}\n{ann['duration_sec']:.1f}s",
                 va='center', fontsize=6.5, color='#555555')

    plt.tight_layout(pad=0.5)
    out = OUTPUT_DIR / "summary.png"
    fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0d0d12')
    plt.close(fig)
    print(f"  ✓ Summary plot  → {out}")


def _snr_color(snr):
    if snr < 0:   return '#ff5f40'   # very noisy → red
    if snr < 10:  return '#f5a623'   # noisy → amber
    if snr < 20:  return '#4a9eff'   # moderate → blue
    return '#3ecf72'                  # clean → green


# ── Stats plot ────────────────────────────────────────────────────────────────

def render_stats(annotations):
    from collections import Counter
    snrs       = [a["snr_db"] for a in annotations]
    durations  = [a["duration_sec"] for a in annotations]
    noise_cnts = Counter(a["noise_type"] for a in annotations)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor='#0d0d12')
    fig.suptitle("Dataset Statistics", fontsize=12, color='white', fontweight='bold')

    NOISE_COLORS = {
        "white":  "#4a9eff",
        "brown":  "#f5a623",
        "crowd":  "#3ecf72",
        "static": "#ff5f40",
        "hum":    "#b06aff",
    }

    # SNR distribution
    ax1 = axes[0]
    ax1.set_facecolor('#16161c')
    n, bins, patches_list = ax1.hist(snrs, bins=12, edgecolor='#0d0d12', linewidth=0.6)
    for patch, left in zip(patches_list, bins[:-1]):
        patch.set_facecolor(_snr_color(left))
    ax1.set_title("SNR distribution (dB)", color='#cccccc', fontsize=10)
    ax1.set_xlabel("SNR (dB)", color='#888'); ax1.set_ylabel("Count", color='#888')
    ax1.tick_params(colors='#888'); ax1.spines[:].set_color('#333344')
    ax1.axvline(np.mean(snrs), color='white', linewidth=1, linestyle='--', alpha=0.5)
    ax1.text(np.mean(snrs)+0.5, ax1.get_ylim()[1]*0.85,
             f"mean\n{np.mean(snrs):.1f}dB", color='white', fontsize=7)

    # Noise type breakdown
    ax2 = axes[1]
    ax2.set_facecolor('#16161c')
    names  = list(noise_cnts.keys())
    counts = [noise_cnts[n] for n in names]
    colors = [NOISE_COLORS.get(n, '#888') for n in names]
    bars   = ax2.bar(names, counts, color=colors, edgecolor='#0d0d12', linewidth=0.6)
    for bar, val in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(val), ha='center', va='bottom', color='white', fontsize=9)
    ax2.set_title("Noise type distribution", color='#cccccc', fontsize=10)
    ax2.tick_params(colors='#888', axis='x', labelsize=8)
    ax2.tick_params(colors='#888', axis='y')
    ax2.spines[:].set_color('#333344')
    ax2.set_ylabel("Count", color='#888')

    # Duration histogram
    ax3 = axes[2]
    ax3.set_facecolor('#16161c')
    ax3.hist(durations, bins=10, color='#7a6fff', edgecolor='#0d0d12', linewidth=0.6)
    ax3.set_title("Clip duration (seconds)", color='#cccccc', fontsize=10)
    ax3.set_xlabel("Duration (s)", color='#888'); ax3.set_ylabel("Count", color='#888')
    ax3.tick_params(colors='#888'); ax3.spines[:].set_color('#333344')

    plt.tight_layout(pad=1.2)
    out = OUTPUT_DIR / "stats.png"
    fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0d0d12')
    plt.close(fig)
    print(f"  ✓ Stats chart   → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 54)
    print("  Synthetic Speech + Noise Dataset Generator")
    print("=" * 54)

    for subdir in ["audio/clean", "audio/noisy", "audio/noise_only"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Generating {NUM_SAMPLES} audio samples...")
    annotations = []
    for i in range(NUM_SAMPLES):
        ann = generate_sample(i)
        annotations.append(ann)
        print(f"      sample {i:04d}  noise={ann['noise_type']:<7}  "
              f"SNR={ann['snr_db']:+6.1f} dB  dur={ann['duration_sec']:.2f}s")

    dataset = {
        "generated_at": datetime.now().isoformat(),
        "num_samples": NUM_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "noise_types": NOISE_TYPES,
        "snr_range_db": list(SNR_RANGE),
        "samples": annotations,
    }
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\n  ✓ Labels saved  → {labels_path}")

    print(f"\n[2/3] Rendering summary grid...")
    render_summary(annotations)

    print(f"\n[3/3] Generating stats...")
    render_stats(annotations)

    total_dur = sum(a["duration_sec"] for a in annotations)
    print(f"\n{'='*54}")
    print(f"  Done!  {NUM_SAMPLES} samples  |  {total_dur:.1f}s total audio")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()
