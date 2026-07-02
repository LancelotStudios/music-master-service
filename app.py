# Reference-match mastering microservice (FastAPI + Matchering).
# The Next.js app can't run Python, so this small service does the audio work:
# given a TARGET song URL and a REFERENCE song URL, it masters the target to match the
# reference's loudness + tonal balance (Matchering), and returns the mastered MP3.
#
# Endpoints:
#   GET  /          → health check
#   POST /master    → { targetUrl, referenceUrl } → mastered MP3 bytes (audio/mpeg)
#
# Run locally:  ./start.sh   (binds 0.0.0.0:$PORT, default 8000)
# Deploy:       Render (render.yaml, python). Set MASTER_TOKEN in the dashboard.

import os
import tempfile
import subprocess
import urllib.request

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel
import matchering as mg

# DSP analysis (tempo/key/loudness/etc). Imported defensively so a problem loading the
# heavier analysis deps (librosa/numba) can never take down the mastering endpoint.
try:
    from analyze import analyze_wav
    _ANALYZE_OK = True
    _ANALYZE_ERR = ""
except Exception as _ae:  # pragma: no cover - import guard
    analyze_wav = None  # type: ignore
    _ANALYZE_OK = False
    _ANALYZE_ERR = str(_ae)

app = FastAPI(title="Reference-match mastering + analysis")


_FFMPEG_CACHE = ""


def _ffmpeg() -> str:
    # Resolve a real ffmpeg binary that works on Render's native Python runtime (no apt).
    # static-ffmpeg ships/fetches a static binary via pip (verified on Linux + macOS).
    global _FFMPEG_CACHE
    override = os.environ.get("FFMPEG_PATH", "").strip()
    if override:
        return override
    if _FFMPEG_CACHE:
        return _FFMPEG_CACHE
    try:
        from static_ffmpeg import run
        ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
        _FFMPEG_CACHE = ffmpeg_path
        return ffmpeg_path
    except Exception:
        return "ffmpeg"  # last-ditch fallback to a system binary

# Optional shared secret so only our app can call it (set in Render + Vercel).
MASTER_TOKEN = os.environ.get("MASTER_TOKEN", "").strip()
MAX_BYTES = 60 * 1024 * 1024  # 60 MB cap per download (a song is a few MB)


class MasterReq(BaseModel):
    targetUrl: str
    referenceUrl: str


class AnalyzeReq(BaseModel):
    audioUrl: str


def _download(url: str, path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "master-service/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large.")
    with open(path, "wb") as f:
        f.write(data)


def _to_wav(src: str, dst: str) -> None:
    # Matchering needs WAV in. ffmpeg comes from imageio-ffmpeg (pip) — no system install needed.
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-i", src, dst],
        check=True,
        timeout=90,
    )


def _to_mp3(src: str, dst: str) -> None:
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-i", src, "-c:a", "libmp3lame", "-q:a", "2", dst],
        check=True,
        timeout=90,
    )


@app.get("/")
def health():
    return {
        "ok": True,
        "service": "reference-match mastering + analysis",
        "matchering": getattr(mg, "__version__", "2"),
        "analyze": _ANALYZE_OK or _ANALYZE_ERR,
    }


@app.post("/analyze")
def analyze(req: AnalyzeReq, x_master_token: str = Header(default="")):
    """Measure a song's objective sound with DSP (tempo/key/loudness/dynamics/tonal balance/
    energy arc) so the app can GROUND its description instead of letting an LLM guess. Returns
    JSON. Same shared-secret auth as /master."""
    if MASTER_TOKEN and x_master_token != MASTER_TOKEN:
        raise HTTPException(status_code=401, detail="Bad or missing token.")
    if not _ANALYZE_OK:
        raise HTTPException(status_code=503, detail=f"Analysis unavailable: {_ANALYZE_ERR}")

    with tempfile.TemporaryDirectory() as d:
        a_in = os.path.join(d, "audio_in")
        a_wav = os.path.join(d, "audio.wav")
        try:
            _download(req.audioUrl, a_in)
            # Decode to a 44.1k WAV, capped at the first 2.5 minutes (captures the sound,
            # fits the 512MB instance). librosa/soundfile read the WAV from here.
            subprocess.run(
                [_ffmpeg(), "-y", "-loglevel", "error", "-t", "150", "-i", a_in, "-ar", "44100", a_wav],
                check=True, timeout=120,
            )
            result = analyze_wav(a_wav)
        except subprocess.CalledProcessError:
            raise HTTPException(status_code=502, detail="Audio conversion failed.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Analysis failed: {e}")

    return result


@app.post("/master")
def master(req: MasterReq, x_master_token: str = Header(default="")):
    if MASTER_TOKEN and x_master_token != MASTER_TOKEN:
        raise HTTPException(status_code=401, detail="Bad or missing token.")

    with tempfile.TemporaryDirectory() as d:
        t_in = os.path.join(d, "target_in")
        r_in = os.path.join(d, "ref_in")
        t_wav = os.path.join(d, "target.wav")
        r_wav = os.path.join(d, "reference.wav")
        out_wav = os.path.join(d, "mastered.wav")
        out_mp3 = os.path.join(d, "mastered.mp3")
        try:
            _download(req.targetUrl, t_in)
            _download(req.referenceUrl, r_in)
            # Convert to 44.1k stereo WAV. Matchering loads whole songs as float arrays, which can
            # OOM a small instance — so keep the reference SHORT (the first ~60s is plenty to learn
            # its loudness/tone) to cut peak memory. The full target is still mastered.
            _to_wav(t_in, t_wav)
            subprocess.run(
                [_ffmpeg(), "-y", "-loglevel", "error", "-t", "60", "-i", r_in, r_wav],
                check=True, timeout=90,
            )
            # The heart of it: match the target to the reference (RMS, frequency response,
            # peak, stereo width). 16-bit PCM result, then encode to MP3.
            mg.process(target=t_wav, reference=r_wav, results=[mg.pcm16(out_wav)])
            _to_mp3(out_wav, out_mp3)
            with open(out_mp3, "rb") as f:
                audio = f.read()
        except subprocess.CalledProcessError:
            raise HTTPException(status_code=502, detail="Audio conversion failed.")
        except HTTPException:
            raise
        except Exception as e:  # matchering or download errors
            raise HTTPException(status_code=502, detail=f"Mastering failed: {e}")

    return Response(content=audio, media_type="audio/mpeg")
