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
