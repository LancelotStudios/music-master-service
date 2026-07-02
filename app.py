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
        # Which build is serving (Render sets RENDER_GIT_COMMIT) — for deploy debugging.
        "commit": os.environ.get("RENDER_GIT_COMMIT", "")[:7],
        # Is the PO-token provider (YouTube bot-wall pass) up?
        "pot": _pot_alive(),
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
                [_ffmpeg(), "-y", "-loglevel", "error", "-t", "120", "-i", a_in, "-ar", "44100", a_wav],
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


# ---- PO-token provider (dodges YouTube's datacenter bot-wall without accounts/cookies) ----
# We launch the single-binary Rust provider (protocol-compatible with the bgutil yt-dlp plugin,
# which auto-detects it at 127.0.0.1:4416). Best-effort: any failure here only means /fetch-youtube
# behaves as before (bot-walled) — it can never affect /master or /analyze.
POT_BIN_URL = os.environ.get(
    "BGUTIL_POT_URL",
    "https://github.com/jim60105/bgutil-ytdlp-pot-provider-rs/releases/latest/download/bgutil-pot-linux-x86_64",
)
_POT_STARTED = False


def _ensure_pot_provider() -> None:
    global _POT_STARTED
    if _POT_STARTED or os.environ.get("DISABLE_POT_PROVIDER"):
        return
    _POT_STARTED = True
    try:
        binp = "/tmp/bgutil-pot"
        if not os.path.exists(binp):
            _download(POT_BIN_URL, binp)
            os.chmod(binp, 0o755)
        subprocess.Popen(
            [binp, "server", "--host", "127.0.0.1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # provider is a bonus, never a dependency


@app.on_event("startup")
def _startup() -> None:
    _ensure_pot_provider()


def _pot_alive() -> bool:
    try:
        req = urllib.request.Request("http://127.0.0.1:4416/ping")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


class FetchYouTubeReq(BaseModel):
    url: str
    startSeconds: float | None = None
    endSeconds: float | None = None


@app.post("/fetch-youtube")
def fetch_youtube(req: FetchYouTubeReq, x_master_token: str = Header(default="")):
    """Pull the AUDIO from a YouTube link as an mp3 (optionally trimmed to a start/stop window),
    so the app can treat a pasted link exactly like an uploaded file (measured analysis + Suno
    covers). Beta-mode convenience; the ToS/legal review lives on the app's future-ideas list."""
    if MASTER_TOKEN and x_master_token != MASTER_TOKEN:
        raise HTTPException(status_code=401, detail="Bad or missing token.")
    try:
        import yt_dlp  # lazy import so a broken dep can never take down /master or /analyze
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"yt-dlp unavailable: {e}")

    start = max(0.0, float(req.startSeconds or 0.0))
    end = float(req.endSeconds or 0.0)
    # Cap the clip at 8 minutes — matches Suno's cover-input limit and keeps files small.
    length = min((end - start) if end > start else 480.0, 480.0)

    with tempfile.TemporaryDirectory() as d:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(d, "yt.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "retries": 2,
            "socket_timeout": 30,
            # No player_client override: with the PO-token provider running, yt-dlp's default
            # clients pass YouTube's datacenter bot-wall (the bgutil plugin auto-detects :4416).
        }
        # Optional escape hatch: a logged-in session's cookies (base64 of a Netscape cookies.txt in
        # the YTDLP_COOKIES_B64 env var) — the reliable fix if the client trick stops working.
        ck = os.environ.get("YTDLP_COOKIES_B64", "").strip()
        if ck:
            try:
                import base64
                cookie_path = os.path.join(d, "cookies.txt")
                with open(cookie_path, "wb") as f:
                    f.write(base64.b64decode(ck))
                opts["cookiefile"] = cookie_path
            except Exception:
                pass  # bad env value → just proceed without cookies
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(req.url, download=True)
                src = ydl.prepare_filename(info)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Couldn't fetch that video's audio: {str(e)[:200]}")
        mp3 = os.path.join(d, "audio.mp3")
        args = [_ffmpeg(), "-y", "-loglevel", "error"]
        if start > 0:
            args += ["-ss", str(start)]
        args += ["-i", src, "-t", str(length), "-c:a", "libmp3lame", "-q:a", "2", mp3]
        try:
            subprocess.run(args, check=True, timeout=180)
        except Exception:
            raise HTTPException(status_code=502, detail="Audio conversion failed.")
        with open(mp3, "rb") as f:
            data = f.read()
    if len(data) < 1000:
        raise HTTPException(status_code=502, detail="Extracted audio looks empty.")
    return Response(content=data, media_type="audio/mpeg")


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
