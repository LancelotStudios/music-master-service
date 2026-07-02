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

MAX_SECONDS = 150  # analyze the first 2.5 minutes — captures a song's sound, fits a 512MB instance
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
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Data-driven global tempo first, with a WIDE prior (std_bpm well above librosa's tight
    # default) so the 120-centered prior can't drag a genuine 75 BPM ballad toward 120.
    try:
        tempo_ac = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, std_bpm=24.0))[0])
    except Exception:
        tempo_ac = 120.0

    # Anchor the beat grid to that tempo (not the 120 default), then read the perceived
    # pulse off the median inter-beat interval (more reliable than one autocorr peak).
    tempo_bt, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=tempo_ac)
    tempo_bt = float(np.atleast_1d(tempo_bt)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)

    if len(beat_times) >= 5:
        ibis = np.diff(beat_times)
        ibis = ibis[(ibis > 0.2) & (ibis < 2.0)]  # 30–300 BPM sanity window
        bpm_grid = 60.0 / float(np.median(ibis)) if len(ibis) else tempo_bt
        # Beat stability: tight, even spacing => trustworthy. Coefficient of variation -> 0..1.
        cov = float(np.std(ibis) / np.mean(ibis)) if len(ibis) else 1.0
        stability = max(0.0, 1.0 - min(cov / 0.25, 1.0))
    else:
        bpm_grid = tempo_bt
        stability = 0.3

    # The beat tracker reliably finds the pulse PERIOD, but often reports the wrong
    # metrical level (half- or double-time) — that's the exact octave error the LLM also hits.
    # Resolve it by scoring the half/normal/double candidates against a "preferred tempo"
    # prior (humans hear tempo near ~108 BPM; most pop/worship sits 70–140). The candidate
    # closest to that center wins, because its half/double almost always falls outside the range.
    base = bpm_grid
    octaves = sorted({_r(base / 2, 1), _r(base, 1), _r(base * 2, 1)})
    octaves = [c for c in octaves if 40 <= c <= 220] or [_r(base, 1)]
    weighted = sorted(((c, _perceptual_weight(c)) for c in octaves), key=lambda x: x[1], reverse=True)
    primary = weighted[0][0]

    # How decisively the prior picked one octave over the next (0 = a genuine 75-vs-150
    # toss-up, 1 = unambiguous). Low clarity must lower the reported confidence.
    best_w = weighted[0][1]
    runner_w = weighted[1][1] if len(weighted) > 1 else 0.0
    octave_clarity = (best_w - runner_w) / best_w if best_w > 0 else 0.0

    agree = abs(tempo_ac - bpm_grid) / bpm_grid < 0.06 if bpm_grid else False
    confidence = 0.35 + 0.4 * stability + 0.25 * min(1.0, octave_clarity * 1.5)
    confidence = min(1.0, confidence)

    # Candidate metrical levels (half / double feels) — lets the recipe say
    # "92 BPM, with a half-time feel" instead of committing to a single number.
    candidates = [c for c in octaves if c != primary]

    return {
        "bpm": primary,
        "confidence": _r(confidence, 2),
        "candidates": candidates,
        "octave_clarity": _r(octave_clarity, 2),
        "beat_grid_bpm": _r(bpm_grid, 1),
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
    def _pcp(sig: np.ndarray) -> np.ndarray:
        chroma = librosa.feature.chroma_cqt(y=sig, sr=sr)
        return chroma.mean(axis=1)  # average pitch-class energy over the whole track

    # Harmonic-percussive separation first: stripping the drums cleans the chroma and
    # measurably helps template-based key detection. Fall back to the raw signal if the
    # harmonic mask comes back empty (can happen on sparse/synthetic material).
    try:
        pcp = _pcp(librosa.effects.harmonic(y))
        if pcp.sum() <= 1e-6:
            pcp = _pcp(y)
    except Exception:
        pcp = _pcp(y)
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

    # Loudness needs the full-channel data — do it FIRST, then free the big stereo array
    # so its memory isn't held while the spectrogram work allocates.
    loudness = _analyze_loudness(data, sr, mono_full)
    del data

    spectral = _analyze_spectral(mono_full, sr)

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
