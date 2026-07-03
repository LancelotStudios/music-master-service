# Deterministic audio ANALYSIS — measure the things an LLM should never guess.
#
# The app currently asks Gemini to "listen" and report tempo, key, loudness, etc.
# LLMs are near-random at those objective tasks (they confuse half/double-time tempo,
# mis-call the key, and sometimes invent instruments). This module measures them with
# real signal processing instead, so they're reproducible and trustworthy. The LLM's
# job then shrinks to describing the vibe in words — grounded in these numbers, never
# inventing them. (This mirrors how Spotify's LLark and NVIDIA's Music Flamingo work:
# DSP measures, the model only narrates.)
#
# Everything here is BSD/MIT/ISC-licensed (numpy, scipy, librosa, pyloudnorm) — clean
# for a commercial product, CPU-only, installs from pip wheels on Render.
#
# Returns a JSON-able dict; see analyze_audio() at the bottom for the shape.

from __future__ import annotations

import math
from typing import Any

import numpy as np
import soundfile as sf

# librosa pulls in numba/llvmlite; import lazily so the rest of the service still boots
# if librosa ever fails to load on a given runtime.
try:
    import librosa  # type: ignore

    _LIBROSA_OK = True
except Exception as _e:  # pragma: no cover - import guard
    librosa = None  # type: ignore
    _LIBROSA_OK = False
    _LIBROSA_ERR = str(_e)

try:
    import pyloudnorm as pyln  # type: ignore

    _PYLN_OK = True
except Exception:  # pragma: no cover
    pyln = None  # type: ignore
    _PYLN_OK = False

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Key-profile templates. We run TWO independent profiles and treat their agreement as a
# confidence signal — the literature is clear that no single profile wins on every genre.
# Krumhansl-Kessler (probe-tone experiments) and Temperley (corpus-tuned) are the classics.
_KRUMHANSL_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KRUMHANSL_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_TEMPERLEY_MAJ = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
_TEMPERLEY_MIN = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

# Frequency bands for the tonal-balance read (the same kind of profile Matchering matches).
_BANDS = [
    ("sub", 20, 60),
    ("bass", 60, 250),
    ("low_mid", 250, 2000),
    ("high_mid", 2000, 6000),
    ("air", 6000, 20000),
]

MAX_SECONDS = 120  # analyze the first 2 minutes — captures a song's sound, fits a 512MB instance
MIR_SR = 22050     # standard sample rate for tempo/key work (fast, accurate enough)


def _r(x: float, n: int = 2) -> float:
    try:
        return round(float(x), n)
    except Exception:
        return 0.0


# ---------- tempo (with half/double-time octave handling) ----------

def _perceptual_weight(bpm: float) -> float:
    """Log-normal 'preferred tempo' weight centered near 108 BPM (Parncutt/Moelants
    resonance — humans hear tempo around there). Used to choose the metrical level among
    half/normal/double candidates: the true notated tempo of most pop/worship sits 70–140,
    and its half/double lands outside that, so the center-most candidate is almost always right."""
    if bpm <= 0:
        return 0.0
    return math.exp(-0.5 * ((math.log(bpm) - math.log(108.0)) / 0.40) ** 2)


def _analyze_tempo(y: np.ndarray, sr: int) -> dict[str, Any]:
    # FELT tempo — the speed a person taps a foot to, not the strongest mathematical
    # repetition. Bench-tuned 2026-07-03 against ground-truth files after two failure
    # classes hit live: (a) the 3:4 / 4:3 metrical trap (a real ~75 BPM chill groove read
    # as 99.4 — caught by Lance's ear), (b) evidence-by-autocorrelation alone prefers the
    # count-every-little-pulse level (132) because busy hi-hats out-repeat the beat.
    #
    # What survived the bench (see prototype scratch tempo_bench4.py):
    #   1. Work on the LOW BAND only (kick/bass < 100 Hz) — the foot-tap lives there;
    #      full-band envelopes are dominated by subdivisions.
    #   2. Skip the intro (30s) — grooves establish after ambient builds.
    #   3. Dense PHASE-FOLD scan over every beat spacing 0.40–1.10s: fold the low-band
    #      energy modulo two beats; the true beat level shows strong phase structure
    #      (energy concentrated at kick/snare positions), subdivisions fold flat.
    #   4. A strict kick-EVENT detector votes: the median spacing between clean, prominent
    #      low-band hits multiplies matching candidates by 1.8 (the foot-tap vote).
    #   5. A mild ~88 BPM log-normal prior breaks ties only.
    onset_full = librosa.onset.onset_strength(y=y, sr=sr)
    fps = sr / 512.0

    # librosa's raw global estimate — kept for observability, no longer trusted to pick.
    try:
        tempo_ac = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_full, sr=sr, std_bpm=24.0))[0])
    except Exception:
        tempo_ac = 120.0

    # Groove window: skip the intro, keep what remains (y is already capped upstream).
    start = min(int(30 * sr), max(0, len(y) - int(60 * sr)))
    y_g = y[start:]

    # Low-band (kick/bass) onset envelope — the felt pulse lives down here.
    try:
        from scipy.signal import butter, sosfilt
        sos = butter(4, 100, btype="low", fs=sr, output="sos")
        onset_low = librosa.onset.onset_strength(y=sosfilt(sos, y_g).astype(np.float32), sr=sr)
    except Exception:
        onset_low = librosa.onset.onset_strength(y=y_g, sr=sr)

    def _phase_contrast(period_s: float, bins: int = 16) -> float:
        L = 2 * period_s  # fold two beats: captures kick/snare alternation
        idx = np.minimum((((np.arange(len(onset_low)) / fps) % L) / L * bins).astype(int), bins - 1)
        prof = np.array([onset_low[idx == b].mean() if (idx == b).any() else 0.0 for b in range(bins)])
        mu = float(prof.mean())
        return float((prof.max() - mu) / mu) if mu > 0 else 0.0

    # Strict kick events: prominent low-band peaks, half-second refractory.
    bg = np.copy(onset_low)
    acc = 0.0
    for i in range(len(onset_low)):
        acc = 0.985 * acc + 0.015 * onset_low[i]
        bg[i] = acc
    events: list[float] = []
    last = -100.0
    for i in range(1, len(onset_low) - 1):
        t = i / fps
        if onset_low[i] > 2.2 * max(bg[i], 1e-6) and onset_low[i] >= onset_low[i - 1] and onset_low[i] >= onset_low[i + 1] and t - last > 0.5:
            events.append(t)
            last = t
    iv = np.diff(np.array(events)) if len(events) > 1 else np.array([])
    iv = iv[(iv > 0.45) & (iv < 1.5)] if len(iv) else iv
    kick_period = float(np.median(iv)) if len(iv) >= 15 else None
    kick_cv = float(np.std(iv) / np.mean(iv)) if len(iv) >= 15 else 1.0

    def _prior(bpm: float) -> float:
        return math.exp(-0.5 * ((math.log(bpm) - math.log(88.0)) / 0.5) ** 2) if bpm > 0 else 0.0

    scan: list[tuple[float, float]] = []  # (score, bpm)
    p = 0.40
    while p <= 1.10:
        bpm = 60.0 / p
        s = _phase_contrast(p) * (0.55 + 0.9 * _prior(bpm))
        if kick_period and kick_cv < 0.4 and abs(p - kick_period) / kick_period < 0.06:
            s *= 1.8  # the foot-tap vote
        scan.append((s, bpm))
        p += 0.004
    scan.sort(reverse=True)
    picks: list[tuple[float, float]] = []
    for s, bpm in scan:
        if all(abs(bpm - b) / b > 0.03 for _, b in picks):
            picks.append((s, bpm))
        if len(picks) == 4:
            break
    primary = _r(picks[0][1], 1)
    margin = (picks[0][0] - picks[1][0]) / picks[0][0] if len(picks) > 1 and picks[0][0] > 0 else 0.0

    confidence = 0.35 + 0.35 * (1.0 - min(kick_cv, 1.0)) + 0.30 * min(1.0, margin * 2.0)
    confidence = min(0.99, confidence)

    # Alternate metrical levels — lets the recipe say "~78 BPM (some count it ~97)".
    candidates = [_r(b, 1) for _, b in picks[1:]]

    return {
        "bpm": primary,
        "confidence": _r(confidence, 2),
        "candidates": candidates,
        "octave_clarity": _r(margin, 2),
        "kick_bpm": _r(60.0 / kick_period, 1) if kick_period else None,
        "autocorr_bpm": _r(tempo_ac, 1),
    }


# ---------- key + mode ----------

def _profile_key(pcp: np.ndarray, maj: np.ndarray, minr: np.ndarray) -> list[tuple[str, float]]:
    """Correlate a 12-bin pitch-class profile against all 24 rotated key templates.
    Returns the ranked (label, correlation) list."""
    maj = (maj - maj.mean()) / (maj.std() + 1e-9)
    minr = (minr - minr.mean()) / (minr.std() + 1e-9)
    p = (pcp - pcp.mean()) / (pcp.std() + 1e-9)
    scores: list[tuple[str, float]] = []
    for i in range(12):
        scores.append((f"{PITCH_CLASSES[i]} major", float(np.dot(p, np.roll(maj, i)) / 12.0)))
        scores.append((f"{PITCH_CLASSES[i]} minor", float(np.dot(p, np.roll(minr, i)) / 12.0)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _analyze_key(y: np.ndarray, sr: int) -> dict[str, Any]:
    # STFT chroma on a 60s middle slice. The earlier version ran harmonic-percussive
    # separation + CQT chroma over the whole window — better in theory, but each allocates
    # multiple full-spectrogram copies and OOM'd the 512MB instance. STFT chroma with the
    # two-profile agreement check still reads keys reliably, in a fraction of the memory.
    if len(y) > 60 * sr:
        mid = len(y) // 2
        y = y[max(0, mid - 30 * sr): mid + 30 * sr]

    def _pcp(sig: np.ndarray) -> np.ndarray:
        chroma = librosa.feature.chroma_stft(y=sig, sr=sr)
        return chroma.mean(axis=1)  # average pitch-class energy over the slice

    try:
        pcp = _pcp(y)
    except Exception:
        return {"key": "", "confidence": 0.0, "alternatives": []}
    if pcp.sum() <= 1e-6:
        return {"key": "", "confidence": 0.0, "alternatives": []}

    kr = _profile_key(pcp, _KRUMHANSL_MAJ, _KRUMHANSL_MIN)
    tm = _profile_key(pcp, _TEMPERLEY_MAJ, _TEMPERLEY_MIN)
    top_kr, score_kr = kr[0]
    top_tm = tm[0][0]

    # Margin of the winner over the runner-up = how decisive the read is.
    margin = score_kr - kr[1][1]
    agree = top_kr == top_tm
    # Confidence blends raw correlation, decisiveness, and whether the two profiles agree.
    confidence = max(0.0, min(1.0, 0.45 * max(0.0, score_kr) + 2.5 * max(0.0, margin) + (0.25 if agree else 0.0)))

    alts = [{"key": k, "score": _r(s, 3)} for k, s in kr[1:4]]
    return {
        "key": top_kr,
        "confidence": _r(confidence, 2),
        "agreement": bool(agree),
        "alternatives": alts,  # usually relative/parallel/fifth — harmonically compatible neighbors
        "temperley_key": top_tm,
    }


# ---------- loudness + dynamics ----------

def _analyze_loudness(data: np.ndarray, sr: int, mono: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if _PYLN_OK:
        try:
            meter = pyln.Meter(sr)  # ITU-R BS.1770-4 / EBU R128
            out["integrated_lufs"] = _r(meter.integrated_loudness(data), 1)
            try:
                out["loudness_range_lu"] = _r(meter.loudness_range(data), 1)
            except Exception:
                pass
        except Exception:
            pass
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    peak_db = 20 * math.log10(peak) if peak > 0 else -120.0
    rms_db = 20 * math.log10(rms) if rms > 0 else -120.0
    out["peak_db"] = _r(peak_db, 1)
    out["rms_db"] = _r(rms_db, 1)
    out["crest_factor_db"] = _r(peak_db - rms_db, 1)  # punchy/dynamic (high) vs squashed/loud (low)
    return out


# ---------- spectral / tonal balance ----------

def _analyze_spectral(mono: np.ndarray, sr: int) -> dict[str, Any]:
    mono = mono - float(np.mean(mono))  # drop DC so it can't masquerade as sub-bass
    # Averaged STFT magnitude — robust for any length, no giant single FFT.
    spec = librosa.stft(mono, n_fft=4096, hop_length=2048)
    mag = np.abs(spec).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    power = mag ** 2
    audible = freqs >= 20  # ignore sub-sonic rumble / envelope artifacts in the totals
    total = float(power[audible].sum()) + 1e-12
    bands: dict[str, float] = {}
    for name, lo, hi in _BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bands[name] = _r(100.0 * float(power[m].sum()) / total, 1)

    centroid = float((freqs[audible] * power[audible]).sum() / total)
    # Plain-English descriptors so the recipe can use words, not just numbers.
    desc = []
    if bands.get("sub", 0) + bands.get("bass", 0) > 45:
        desc.append("bass-heavy")
    elif bands.get("sub", 0) + bands.get("bass", 0) < 20:
        desc.append("bass-light")
    if bands.get("air", 0) > 6 or centroid > 3000:
        desc.append("bright / airy top end")
    elif centroid < 1200:
        desc.append("dark / warm")
    if bands.get("low_mid", 0) + bands.get("high_mid", 0) < 40:
        desc.append("scooped mids")

    return {"bands_pct": bands, "centroid_hz": _r(centroid, 0), "descriptors": desc}


# ---------- energy arc (arrangement dynamics over time) ----------

def _analyze_energy_arc(mono: np.ndarray, sr: int, segments: int = 8) -> dict[str, Any]:
    rms = librosa.feature.rms(y=mono)[0]
    if rms.size == 0:
        return {"contour": [], "shape": "unknown"}
    # Downsample the RMS curve into N coarse segments and normalize 0..1.
    idx = np.linspace(0, len(rms), segments + 1).astype(int)
    seg = np.array([rms[idx[i]:idx[i + 1]].mean() if idx[i + 1] > idx[i] else 0.0 for i in range(segments)])
    norm = (seg - seg.min()) / (seg.ptp() + 1e-9)
    contour = [_r(v, 2) for v in norm]

    first, last = norm[: max(1, segments // 3)].mean(), norm[-max(1, segments // 3):].mean()
    rng = float(norm.ptp())
    if rng < 0.25:
        shape = "steady — holds one fairly constant energy the whole way"
    elif last - first > 0.2:
        shape = "builds — starts sparser and grows toward the end"
    elif first - last > 0.2:
        shape = "front-loaded — biggest early, easing off later"
    else:
        shape = "dynamic — rises and falls between sections"
    return {"contour": contour, "shape": shape, "range": _r(rng, 2)}


# ---------- orchestration ----------

def analyze_wav(wav_path: str) -> dict[str, Any]:
    """Analyze a decoded WAV file. Returns the structured measurement dict."""
    if not _LIBROSA_OK:
        raise RuntimeError(f"librosa unavailable: {_LIBROSA_ERR}")

    # float32 throughout — float64 doubled every array (and forced complex128 spectrograms),
    # which OOM'd the 512MB Render instance. float32 is plenty for these measurements.
    data, sr = sf.read(wav_path, always_2d=True, dtype="float32")
    if data.shape[0] > sr * MAX_SECONDS:
        data = data[: sr * MAX_SECONDS]
    mono_full = data.mean(axis=1, dtype=np.float32)
    duration = len(mono_full) / float(sr)

    # Loudness on a 24 kHz stereo downsample — pyloudnorm's filters copy the array in float64,
    # which OOM'd the 512MB instance at 44.1k. LUFS reads the same at 24 kHz (K-weighted loudness
    # gets almost nothing from content above 12 kHz). Free the big buffers as soon as possible.
    LOUD_SR = 24000
    if sr != LOUD_SR:
        loud_data = np.ascontiguousarray(
            librosa.resample(np.ascontiguousarray(data.T), orig_sr=sr, target_sr=LOUD_SR).T,
            dtype=np.float32,
        )
    else:
        loud_data = data
    del data
    loud_mono = loud_data.mean(axis=1, dtype=np.float32)
    loudness = _analyze_loudness(loud_data, LOUD_SR, loud_mono)
    del loud_data, loud_mono

    spectral = _analyze_spectral(mono_full[: min(len(mono_full), 90 * sr)], sr)

    # Lower-rate mono for the MIR work (tempo/key/energy); free the 44.1k mono after.
    y = librosa.resample(mono_full, orig_sr=sr, target_sr=MIR_SR) if sr != MIR_SR else mono_full
    if y is not mono_full:
        del mono_full

    result: dict[str, Any] = {
        "duration_sec": _r(duration, 1),
        "tempo": _analyze_tempo(y, MIR_SR),
        "key": _analyze_key(y, MIR_SR),
        "loudness": loudness,
        "spectral": spectral,
        "energy_arc": _analyze_energy_arc(y, MIR_SR),
        "meta": {
            "analyzer": "dsp-v1",
            "sr": sr,
            "pyloudnorm": _PYLN_OK,
        },
    }
    return result
