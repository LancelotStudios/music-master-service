# Reference-match mastering service

A tiny FastAPI service that masters a song to **match a reference track you love**
(loudness + tonal balance), using the open-source [Matchering](https://github.com/sergree/matchering).
The main Next.js app (Vercel) can't run Python, so this runs separately on Render and the
app calls it over HTTP. ffmpeg comes from the `static-ffmpeg` pip package — no apt / system install.

## What it does
`POST /master { targetUrl, referenceUrl }` → downloads both, converts to WAV, runs Matchering
(matches RMS, frequency response, peak, stereo width), returns the mastered song as MP3 bytes.
A `X-Master-Token` header must match the `MASTER_TOKEN` env var.

## Run locally
```bash
./start.sh            # first run builds .venv + installs deps
# health:  curl localhost:8000/
```

## Deploy to Render — exact steps (see the chat for the fully detailed walkthrough)
1. Put this `master-service/` folder in a Git repo (GitHub).
2. Render → **New → Web Service** → connect the repo. If it's a subfolder, set **Root Directory** = `master-service`.
3. Settings (render.yaml already sets these): Runtime **Python 3**, Build `pip install -r requirements.txt`,
   Start `uvicorn app:app --host 0.0.0.0 --port $PORT`, Plan **Starter**.
4. Environment → add **`MASTER_TOKEN`** = a long random secret.
5. Create → wait for "Live". Copy the URL, e.g. `https://music-master-service.onrender.com`. Test: open it → `{"ok":true,...}`.
6. In **Vercel** (the Next.js app) → Settings → Environment Variables, add:
   - `MASTER_SERVICE_URL` = the Render URL
   - `MASTER_TOKEN` = the same secret
   Redeploy. The "Match a song I love" button appears automatically.

If `MASTER_SERVICE_URL` is unset, the app hides the reference-match option and keeps the free
built-in polish/master — nothing breaks.

## Verified (2026-06-01)
Local end-to-end with the exact deploy code (static-ffmpeg, no system ffmpeg): `/master` returned
a valid 2.2 MB MP3; output matched the reference's tone (target highs −32 dB → −30, toward the
reference's −30.7) and loudness; token gate returns 401 without the header.

## `/analyze` — DSP song analysis (added 2026-06-07)
`POST /analyze { audioUrl }` (same `X-Master-Token` header) → measures the song's OBJECTIVE
sound with real signal processing and returns JSON: tempo (BPM + confidence + half/double-time
candidates), key + mode (+ confidence + harmonically-adjacent alternatives), loudness
(LUFS/LRA), dynamics (crest factor), tonal balance (sub→air band %, centroid, descriptors),
and the energy/arrangement arc. This exists because an LLM hallucinates these numbers (it
confuses half/double-time tempo, mis-calls the key); DSP measures them so the app can GROUND
its AI description instead of guessing. Code: `analyze.py` (librosa + pyloudnorm + ffmpeg — all
BSD/MIT/ISC, CPU-only, pip wheels). Imported defensively, so an analysis-dep problem can never
take down `/master`.

The Next.js app calls it when `ANALYZE_SERVICE_URL` (the Render URL) + `MASTER_TOKEN` are set in
Vercel. If `ANALYZE_SERVICE_URL` is unset, the app silently falls back to the pure-Gemini path —
nothing breaks until it's live.

Verified locally (2026-06-07) against synthetic audio with known ground truth: key detection
exact on C-major and A-minor progressions (with relative/parallel/fifth alternatives); tempo
exact on 4/5 test grooves (120/90/100/140 BPM) and the 5th (75 BPM) correctly *flagged
low-confidence* as a genuine 75-vs-152 half/double ambiguity rather than asserting a wrong
number; spectral/loudness sane. Tempo octave resolution is the known weak spot — the accuracy
upgrade is Essentia RhythmExtractor2013 or a madmom/allin1 downbeat tracker (see the research note).
